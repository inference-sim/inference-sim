package cmd

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestConvertOtel_SingleFileEndToEnd(t *testing.T) {
	dir := t.TempDir()
	inPath := filepath.Join(dir, "trace.json")
	traceJSON := `{"spans":[
	  {"span_id":"a","name":"chat gpt","start_time":"2026-01-01T00:00:00.000000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10},"trace_id":"sess-1"},
	  {"span_id":"b","name":"chat gpt","start_time":"2026-01-01T00:00:08.000000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":150,"gen_ai.usage.output_tokens":20},"trace_id":"sess-1"}
	]}`
	if err := os.WriteFile(inPath, []byte(traceJSON), 0644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	outPrefix := filepath.Join(dir, "out")

	if err := runConvertOtel(inPath, outPrefix, workload.OTelConvertOptions{ContextGrowth: "accumulate", MaxThinkTimeUs: 15_000_000, MinRounds: 1}); err != nil {
		t.Fatalf("runConvertOtel: %v", err)
	}

	trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
	if err != nil {
		t.Fatalf("load exported trace: %v", err)
	}
	if trace.Header.SessionContextGrowth != "accumulate" {
		t.Errorf("header growth = %q, want accumulate", trace.Header.SessionContextGrowth)
	}
	if len(trace.Records) != 2 {
		t.Fatalf("records = %d, want 2", len(trace.Records))
	}
	// Global request IDs assigned 0..N in session/round order.
	if trace.Records[0].RequestID != 0 || trace.Records[1].RequestID != 1 {
		t.Errorf("request IDs = %d,%d, want 0,1", trace.Records[0].RequestID, trace.Records[1].RequestID)
	}
	// Delta preserved: round1 = 150-100-10 = 40.
	if trace.Records[1].InputTokens != 40 {
		t.Errorf("round1 delta = %d, want 40", trace.Records[1].InputTokens)
	}
}

func TestConvertOtel_DirectoryDeterministicIDs(t *testing.T) {
	dir := t.TempDir()
	inDir := filepath.Join(dir, "traces")
	if err := os.MkdirAll(inDir, 0755); err != nil {
		t.Fatal(err)
	}
	// Two single-call sessions in separate files, both kept (MinRounds: 1);
	// verifies deterministic global request-ID assignment in sorted-filename order.
	write := func(name, tid string, in, out int) {
		j := `{"spans":[{"span_id":"x","name":"chat m","start_time":"2026-01-01T00:00:00.000000","status":{"code":1},"attributes":{"gen_ai.usage.input_tokens":` +
			itoa(in) + `,"gen_ai.usage.output_tokens":` + itoa(out) + `},"trace_id":"` + tid + `"}]}`
		if err := os.WriteFile(filepath.Join(inDir, name), []byte(j), 0644); err != nil {
			t.Fatal(err)
		}
	}
	write("t0.json", "sess-a", 50, 5)
	write("t1.json", "sess-b", 60, 6)
	outPrefix := filepath.Join(dir, "out")

	if err := runConvertOtel(inDir, outPrefix, workload.OTelConvertOptions{ContextGrowth: "accumulate", MinRounds: 1}); err != nil {
		t.Fatalf("runConvertOtel: %v", err)
	}
	trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(trace.Records) != 2 {
		t.Fatalf("records = %d, want 2", len(trace.Records))
	}
	// Files sorted by name → sess-a (t0) gets ID 0, sess-b (t1) gets ID 1.
	if trace.Records[0].SessionID != "sess-a" || trace.Records[0].RequestID != 0 {
		t.Errorf("record0 = %q/%d, want sess-a/0", trace.Records[0].SessionID, trace.Records[0].RequestID)
	}
	if trace.Records[1].SessionID != "sess-b" || trace.Records[1].RequestID != 1 {
		t.Errorf("record1 = %q/%d, want sess-b/1", trace.Records[1].SessionID, trace.Records[1].RequestID)
	}
}

// itoa is a tiny local helper to keep the test JSON builder readable.
func itoa(n int) string { return strconv.Itoa(n) }

// TestConvertOtel_IndependentGrowthEmptyHeader covers the "independent" →
// empty-header mapping in runConvertOtel: opts.ContextGrowth == "independent"
// must be written to the exported header as SessionContextGrowth == "" (NOT
// the literal string "independent"). TestConvertOtel_SingleFileEndToEnd
// already covers the "accumulate" → "accumulate" case.
func TestConvertOtel_IndependentGrowthEmptyHeader(t *testing.T) {
	dir := t.TempDir()
	inPath := filepath.Join(dir, "trace.json")
	traceJSON := `{"spans":[
	  {"span_id":"a","name":"chat gpt","start_time":"2026-01-01T00:00:00.000000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10},"trace_id":"sess-1"}
	]}`
	if err := os.WriteFile(inPath, []byte(traceJSON), 0644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	outPrefix := filepath.Join(dir, "out")

	if err := runConvertOtel(inPath, outPrefix, workload.OTelConvertOptions{ContextGrowth: "independent", MinRounds: 1}); err != nil {
		t.Fatalf("runConvertOtel: %v", err)
	}

	trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
	if err != nil {
		t.Fatalf("load exported trace: %v", err)
	}
	if trace.Header.SessionContextGrowth != "" {
		t.Errorf("header growth = %q, want empty string for independent growth", trace.Header.SessionContextGrowth)
	}
}

// TestConvertOtel_SkipsBelowMinRoundsAndUnparseable covers both non-fatal skip
// paths in runConvertOtel's conversion loop: a session with fewer usable LLM
// calls than MinRounds (the ConvertOTelTrace (nil,nil) branch) and a file with
// malformed JSON (the parse-error `continue` branch). Neither should abort the
// run or contribute records to the output; only the kept session's records
// should appear.
func TestConvertOtel_SkipsBelowMinRoundsAndUnparseable(t *testing.T) {
	dir := t.TempDir()
	inDir := filepath.Join(dir, "traces")
	if err := os.MkdirAll(inDir, 0755); err != nil {
		t.Fatal(err)
	}

	// (a) Valid session with 2 usable rounds — kept (MinRounds: 2).
	kept := `{"spans":[
	  {"span_id":"a","name":"chat gpt","start_time":"2026-01-01T00:00:00.000000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":100,"gen_ai.usage.output_tokens":10},"trace_id":"sess-keep"},
	  {"span_id":"b","name":"chat gpt","start_time":"2026-01-01T00:00:08.000000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":150,"gen_ai.usage.output_tokens":20},"trace_id":"sess-keep"}
	]}`
	if err := os.WriteFile(filepath.Join(inDir, "a_kept.json"), []byte(kept), 0644); err != nil {
		t.Fatal(err)
	}

	// (b) Valid session with only 1 usable round — below MinRounds:2, skipped
	// via ConvertOTelTrace's (nil,nil) return.
	tooFew := `{"spans":[
	  {"span_id":"c","name":"chat gpt","start_time":"2026-01-01T00:00:00.000000+00:00","status":{"code":1},"attributes":{"gen_ai.request.model":"gpt","gen_ai.usage.input_tokens":50,"gen_ai.usage.output_tokens":5},"trace_id":"sess-toofew"}
	]}`
	if err := os.WriteFile(filepath.Join(inDir, "b_toofew.json"), []byte(tooFew), 0644); err != nil {
		t.Fatal(err)
	}

	// (c) Malformed JSON — skipped via the parse-error `continue` branch, must
	// not abort the run.
	if err := os.WriteFile(filepath.Join(inDir, "c_malformed.json"), []byte(`{not json`), 0644); err != nil {
		t.Fatal(err)
	}

	outPrefix := filepath.Join(dir, "out")
	if err := runConvertOtel(inDir, outPrefix, workload.OTelConvertOptions{ContextGrowth: "accumulate", MinRounds: 2}); err != nil {
		t.Fatalf("runConvertOtel: %v", err)
	}

	trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
	if err != nil {
		t.Fatalf("load exported trace: %v", err)
	}
	if len(trace.Records) != 2 {
		t.Fatalf("records = %d, want 2 (only the kept session's rounds)", len(trace.Records))
	}
	for _, r := range trace.Records {
		if r.SessionID != "sess-keep" {
			t.Errorf("unexpected record from session %q; only sess-keep should have contributed records", r.SessionID)
		}
	}
}
