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

func TestConvertOTelTrace_IncludeErrorsKeepsErrorStatus(t *testing.T) {
	// One OK span + one error span (status.code:2) that still carries
	// non-nil token counts. With IncludeErrors:true both are kept, and the
	// error span's record MUST report Status "error", not "ok".
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5},"trace_id":"s"},
	  {"span_id":"e","name":"chat m","start_time":"2026-01-01T00:00:03.000000","status":{"code":2},"attributes":{"gen_ai.usage.input_tokens":10,"gen_ai.usage.output_tokens":2},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{IncludeErrors: true, MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertOTelTrace: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("got %d records, want 2 (error span kept under IncludeErrors)", len(recs))
	}
	if recs[0].Status != "ok" {
		t.Errorf("round0 Status = %q, want ok", recs[0].Status)
	}
	if recs[1].Status != "error" {
		t.Errorf("round1 Status = %q, want error (span had status.code:2)", recs[1].Status)
	}
}

func TestConvertOTelTrace_FiltersByInputMessagesAttr(t *testing.T) {
	// "llm.call" does NOT have the "chat " name prefix, but carries
	// gen_ai.input.messages — the isLLMSpan disjunct must keep it independent
	// of the name-prefix check. A span with neither signal is dropped.
	j := `{"spans":[
	  {"span_id":"a","name":"llm.call","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5,"gen_ai.input.messages":"[]"},"trace_id":"s"},
	  {"span_id":"x","name":"other.op","start_time":"2026-01-01T00:00:02.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":999,"gen_ai.usage.output_tokens":999},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertOTelTrace: %v", err)
	}
	if len(recs) != 1 {
		t.Fatalf("got %d records, want 1 (only the gen_ai.input.messages span kept)", len(recs))
	}
	if recs[0].InputTokens != 50 || recs[0].OutputTokens != 5 {
		t.Errorf("round0 = in %d out %d, want 50/5", recs[0].InputTokens, recs[0].OutputTokens)
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
	// T6: pin the reconstruction MAGNITUDE, not just delta == 0. The accumulate
	// buffer reconstructs round 1's absolute input as prev_input(200) +
	// prev_output(50) + clamped_delta(0) = 250, which OVER-counts the recorded
	// 120 by the clamped deficit (130). Asserting the reconstructed 250 (not just
	// the zero delta) catches a future clamp-semantics change that still yields a
	// non-negative delta but a different reconstructed absolute.
	reconstructed := recs[0].InputTokens + recs[0].OutputTokens + recs[1].InputTokens
	if reconstructed != 250 {
		t.Errorf("reconstructed round1 absolute input = %d, want 250 (200+50+0; over-counts recorded 120 by the clamped 130)", reconstructed)
	}
}

// TestConvertOTelTrace_SortsOutOfOrderSpans (T1) feeds spans OUT of start-time
// order; ConvertOTelTrace must sort by start time so RoundIndex 0..N and the
// input deltas reflect *time* order, not the order spans appear in the trace.
// Delta correctness depends on this sort: if it were dropped the records would
// follow JSON order and every delta below would differ — a regression this test
// catches (the other converter tests all supply already-ordered spans).
func TestConvertOTelTrace_SortsOutOfOrderSpans(t *testing.T) {
	// JSON order: t=20s (in 215), t=0s (in 100), t=8s (in 150) — scrambled.
	j := `{"spans":[
	  {"span_id":"c","name":"chat m","start_time":"2026-01-01T00:00:20.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":215,"gen_ai.usage.output_tokens":5},"trace_id":"s"},
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"2026-01-01T00:00:08.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":150,"gen_ai.usage.output_tokens":20},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertOTelTrace: %v", err)
	}
	if len(recs) != 3 {
		t.Fatalf("got %d records, want 3", len(recs))
	}
	// Time order 0s → 8s → 20s (inputs 100 → 150 → 215), regardless of the
	// scrambled span order: round 0 = full 100; round 1 delta = 150-100-10 = 40;
	// round 2 delta = 215-150-20 = 45.
	wantIn := []int{100, 40, 45}
	wantArr := []int64{0, 8_000_000, 20_000_000}
	for i, r := range recs {
		if r.RoundIndex != i {
			t.Errorf("record %d RoundIndex = %d, want %d (must follow start-time order)", i, r.RoundIndex, i)
		}
		if r.InputTokens != wantIn[i] {
			t.Errorf("round %d input delta = %d, want %d (start-time order, not JSON order)", i, r.InputTokens, wantIn[i])
		}
		if r.ArrivalTimeUs != wantArr[i] {
			t.Errorf("round %d arrival = %d, want %d", i, r.ArrivalTimeUs, wantArr[i])
		}
	}
}

// TestConvertOTelTrace_DropsSpanMissingTokenCount (T2) drops a chat span that is
// missing the gen_ai.usage.input_tokens key entirely (nil pointer, NOT a
// recorded 0) — it has no ground-truth token count. This is distinct from the
// error-status drop; the surrounding OK spans are kept and renumber contiguously.
func TestConvertOTelTrace_DropsSpanMissingTokenCount(t *testing.T) {
	// Middle span omits gen_ai.usage.input_tokens → InputTokens pointer nil → dropped.
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10},"trace_id":"s"},
	  {"span_id":"n","name":"chat m","start_time":"2026-01-01T00:00:04.000000","status":{"code":1},"attributes":{"gen_ai.usage.output_tokens":9},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"2026-01-01T00:00:08.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":150,"gen_ai.usage.output_tokens":20},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1})
	if err != nil {
		t.Fatalf("ConvertOTelTrace: %v", err)
	}
	if len(recs) != 2 {
		t.Fatalf("got %d records, want 2 (span with absent input_tokens dropped)", len(recs))
	}
	// The two survivors are the 0s and 8s spans; the delta spans the survivors
	// (150-100-10 = 40) and RoundIndex renumbers 0,1 with no gap from the drop.
	if recs[1].RoundIndex != 1 || recs[1].InputTokens != 40 {
		t.Errorf("survivor round1 = ri %d in %d, want 1/40", recs[1].RoundIndex, recs[1].InputTokens)
	}
}

// TestConvertOTelTrace_NoSessionIDErrors (T3) verifies that a trace whose spans
// carry neither session_id nor trace_id cannot be identified: ConvertOTelTrace
// (via sessionIDFromSpans) returns an error rather than emitting records under
// an empty session id. The corpus-level warn+skip propagation of this same error
// is covered in cmd/convert_otel_test.go (TestConvertOtel_SkipsNoSessionIDTrace).
func TestConvertOTelTrace_NoSessionIDErrors(t *testing.T) {
	j := `{"spans":[{"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10}}]}`
	if _, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1}); err == nil {
		t.Fatal("expected error for a trace with no session_id or trace_id, got nil")
	}
}

// TestConvertOTelTrace_UnparseableTimestampErrors (T5) verifies that a malformed
// start_time is a whole-trace error (return), NOT a per-span skip — deliberately
// asymmetric with the error-status / nil-token per-span drops, because an
// unparseable timestamp makes the round ordering itself untrustworthy. A
// well-formed span in the same trace does not rescue it.
func TestConvertOTelTrace_UnparseableTimestampErrors(t *testing.T) {
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"not-a-timestamp","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":150,"gen_ai.usage.output_tokens":20},"trace_id":"s"}
	]}`
	if _, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1}); err == nil {
		t.Fatal("expected error for an unparseable start_time, got nil")
	}
}

// TestOTelSessionID (T7) covers the exported OTelSessionID helper (previously
// uncovered): it prefers session_id, falls back to trace_id, and errors when a
// trace carries neither.
func TestOTelSessionID(t *testing.T) {
	cases := []struct {
		name    string
		json    string
		want    string
		wantErr bool
	}{
		{"prefers session_id over trace_id", `{"spans":[{"span_id":"a","session_id":"sid","trace_id":"tid"}]}`, "sid", false},
		{"falls back to trace_id", `{"spans":[{"span_id":"a","trace_id":"tid"}]}`, "tid", false},
		{"errors when neither present", `{"spans":[{"span_id":"a"}]}`, "", true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := OTelSessionID([]byte(tc.json))
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got id %q", got)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("session id = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestConvertOTelTrace_ZeroMaxThinkTimeUncapped (T8) exercises the MaxThinkTimeUs
// == 0 ("no cap") branch that TestConvertOTelTrace_CapsThinkTime (15s cap) does
// not: a large inter-call gap is preserved verbatim as the arrival time.
func TestConvertOTelTrace_ZeroMaxThinkTimeUncapped(t *testing.T) {
	// 100s gap; with MaxThinkTimeUs:0 it is NOT capped.
	j := `{"spans":[
	  {"span_id":"a","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5},"trace_id":"s"},
	  {"span_id":"b","name":"chat m","start_time":"2026-01-01T00:01:40.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":70,"gen_ai.usage.output_tokens":8},"trace_id":"s"}
	]}`
	recs, err := ConvertOTelTrace([]byte(j), OTelConvertOptions{MinRounds: 1, MaxThinkTimeUs: 0})
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if recs[1].ArrivalTimeUs != 100_000_000 {
		t.Errorf("uncapped arrival = %d, want 100e6 (no cap applied when MaxThinkTimeUs == 0)", recs[1].ArrivalTimeUs)
	}
}
