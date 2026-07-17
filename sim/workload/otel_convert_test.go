package workload

import (
	"testing"
)

// otelTraceJSON builds a minimal OTel trace with the given (input,output) token
// pairs and start times (µs since epoch encoded as ISO-8601). Each span is a
// chat LLM call in status OK.
const twoCallTrace = `{
  "spans": [
    {"span_id":"a","name":"chat gpt","start_time":"2026-01-01T00:00:00.000000+00:00","end_time":"2026-01-01T00:00:00.001000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10,"gen_ai.input.messages":"[]"}},
    {"span_id":"b","name":"chat gpt","start_time":"2026-01-01T00:00:08.000000+00:00","end_time":"2026-01-01T00:00:08.001000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":150,"gen_ai.usage.output_tokens":20,"gen_ai.input.messages":"[]"}},
    {"span_id":"c","name":"chat gpt","start_time":"2026-01-01T00:00:20.000000+00:00","end_time":"2026-01-01T00:00:20.001000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":215,"gen_ai.usage.output_tokens":5,"gen_ai.input.messages":"[]"},"trace_id":"sess-1"}
  ]
}`

func TestConvertOTelTrace_DeltaReconstruction(t *testing.T) {
	recs, err := ConvertOTelTrace([]byte(twoCallTrace), OTelConvertOptions{ContextGrowth: "accumulate", MaxThinkTimeUs: 15_000_000, MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertOTelTrace: %v", err)
	}
	if len(recs) != 3 {
		t.Fatalf("got %d records, want 3", len(recs))
	}
	// Round 0: full first prompt.
	if recs[0].InputTokens != 100 || recs[0].OutputTokens != 10 || recs[0].RoundIndex != 0 {
		t.Errorf("round0 = in %d out %d ri %d, want 100/10/0", recs[0].InputTokens, recs[0].OutputTokens, recs[0].RoundIndex)
	}
	// Round 1 delta: 150 - 100 - 10 = 40.
	if recs[1].InputTokens != 40 {
		t.Errorf("round1 delta = %d, want 40", recs[1].InputTokens)
	}
	// Round 2 delta: 215 - 150 - 20 = 45.
	if recs[2].InputTokens != 45 {
		t.Errorf("round2 delta = %d, want 45", recs[2].InputTokens)
	}
	// Delta reconstruction law: prefix + running(delta+output) == recorded totals.
	wantTotals := []int{100, 150, 215}
	running := 0
	for i, r := range recs {
		if i == 0 {
			running = r.InputTokens
		} else {
			running += r.InputTokens // add this round's new input delta
		}
		if running != wantTotals[i] {
			t.Errorf("round %d reconstructed input = %d, want %d", i, running, wantTotals[i])
		}
		running += r.OutputTokens // account this round's output before next delta
	}
	// Think time carried as arrival gaps (µs): 0, 8s, 20s.
	if recs[0].ArrivalTimeUs != 0 || recs[1].ArrivalTimeUs != 8_000_000 || recs[2].ArrivalTimeUs != 20_000_000 {
		t.Errorf("arrivals = %d/%d/%d, want 0/8e6/20e6", recs[0].ArrivalTimeUs, recs[1].ArrivalTimeUs, recs[2].ArrivalTimeUs)
	}
	// SessionID from trace_id.
	if recs[0].SessionID != "sess-1" {
		t.Errorf("session id = %q, want sess-1", recs[0].SessionID)
	}
	// Model MUST be empty even though every span records "gpt": TraceRecord.Model
	// is routing-significant (buildRouterState filters instances by it), so a
	// recorded name differing from --model would drop every request at routing.
	for i, r := range recs {
		if r.Model != "" {
			t.Errorf("round %d Model = %q, want empty (recorded model must not reach the routing-significant field)", i, r.Model)
		}
	}
}

func TestConvertOTelTrace_FiltersErrorsAndTzNaive(t *testing.T) {
	// One OK tz-naive span + one error span (dropped) + one OK span.
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5},"trace_id":"s"},
	  {"span_id":"e","name":"chat m","start_time":"2026-01-01T00:00:03.000000","status":{"code":2},"attributes":{"gen_ai.usage.input_tokens":0,"gen_ai.usage.output_tokens":0},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"2026-01-01T00:00:05.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":70,"gen_ai.usage.output_tokens":8},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("got %d records, want 2 (error span dropped)", len(recs))
	}
	if recs[1].RoundIndex != 1 || recs[1].InputTokens != (70-50-5) {
		t.Errorf("round1 = ri %d in %d, want 1/15", recs[1].RoundIndex, recs[1].InputTokens)
	}
	if recs[1].ArrivalTimeUs != 5_000_000 {
		t.Errorf("arrival = %d, want 5e6", recs[1].ArrivalTimeUs)
	}
}

func TestConvertOTelTrace_MinRoundsSkips(t *testing.T) {
	j := `{"spans":[{"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5},"trace_id":"s"}]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 2})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if recs != nil {
		t.Fatalf("got %d records, want nil (below MinRounds)", len(recs))
	}
}

func TestConvertOTelTrace_CapsThinkTime(t *testing.T) {
	// 100s gap, capped to 15s.
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"2026-01-01T00:01:40.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":70,"gen_ai.usage.output_tokens":8},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1, MaxThinkTimeUs: 15_000_000})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if recs[1].ArrivalTimeUs != 15_000_000 {
		t.Errorf("capped arrival = %d, want 15e6", recs[1].ArrivalTimeUs)
	}
}

func TestConvertOTelTrace_NonMonotoneClampsToZero(t *testing.T) {
	// Round 1's recorded input (120) is SMALLER than round 0's input+output
	// (200+50=250) — e.g. the agent compacted/summarized context. The raw delta
	// 120-200-50 = -130 must clamp to 0 (never negative). Exact reconstruction
	// does NOT hold in this case (accepted, documented deviation): the accumulate
	// buffer over-counts by the clamped deficit rather than shrinking.
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":200,"gen_ai.usage.output_tokens":50},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"2026-01-01T00:00:05.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":120,"gen_ai.usage.output_tokens":8},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if recs[0].InputTokens != 200 {
		t.Errorf("round0 input = %d, want 200 (full first prompt)", recs[0].InputTokens)
	}
	if recs[1].InputTokens != 0 {
		t.Errorf("round1 delta = %d, want 0 (clamped, never negative)", recs[1].InputTokens)
	}
}
