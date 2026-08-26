package cmd

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// T9 — end-to-end: runConvertWeka → LoadTraceV2 (header accumulate, global IDs
// 0..N) → LoadTraceV2SessionBlueprints + SessionManager reconstructs each
// monotone session's recorded absolute per-round inputs via the accumulate
// buffer, and the ThinkTimeSampler yields the recomputed pure-think values
// (which drive follow-up arrival = completion + think, INV-10).
func TestConvertWeka_EndToEndAndAccumulateReconstruction(t *testing.T) {
	dir := t.TempDir()
	inPath := filepath.Join(dir, "traces.jsonl")
	// One monotone session, 3 main turns (in 100/150/215, out 10/20/5),
	// think gaps: t1−t0−api0 = 8−0−1 = 7s; t2−t1−api1 = 20−8−2 = 10s.
	session := `{"id":"sess-1","models":["claude-sonnet-4"],"requests":[` +
		`{"type":"n","t":0.0,"in":100,"out":10,"api_time":1.0},` +
		`{"type":"s","t":8.0,"in":150,"out":20,"api_time":2.0},` +
		`{"type":"n","t":20.0,"in":215,"out":5,"api_time":1.0}` +
		`]}`
	if err := os.WriteFile(inPath, []byte(session+"\n"), 0o644); err != nil {
		t.Fatalf("write input: %v", err)
	}
	outPrefix := filepath.Join(dir, "out")

	if err := runConvertWeka(inPath, outPrefix, workload.WekaConvertOptions{ContextGrowth: "accumulate", MinRounds: 1}); err != nil {
		t.Fatalf("runConvertWeka: %v", err)
	}

	trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
	if err != nil {
		t.Fatalf("load exported trace: %v", err)
	}
	if trace.Header.SessionContextGrowth != "accumulate" {
		t.Errorf("header growth = %q, want accumulate", trace.Header.SessionContextGrowth)
	}
	if len(trace.Records) != 3 {
		t.Fatalf("records = %d, want 3", len(trace.Records))
	}
	for i, r := range trace.Records {
		if r.RequestID != i {
			t.Errorf("record %d RequestID = %d, want %d (assigned in stream order)", i, r.RequestID, i)
		}
		if r.Model != "" {
			t.Errorf("record %d Model = %q, want empty (routing safety)", i, r.Model)
		}
	}
	// think_time_us column round-trips: recorded 7s / 10s on rounds 1, 2 (non-nil, #1608).
	if trace.Records[1].ThinkTimeUs == nil || *trace.Records[1].ThinkTimeUs != 7_000_000 ||
		trace.Records[2].ThinkTimeUs == nil || *trace.Records[2].ThinkTimeUs != 10_000_000 {
		t.Errorf("think_time_us = [_,%v,%v], want [_,&7e6,&10e6]", trace.Records[1].ThinkTimeUs, trace.Records[2].ThinkTimeUs)
	}

	// Closed-loop reconstruction through the SessionManager (accumulate buffer).
	r0, bps, err := workload.LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("LoadTraceV2SessionBlueprints: %v", err)
	}
	if len(bps) != 1 {
		t.Fatalf("blueprints = %d, want 1", len(bps))
	}
	// ThinkTimeSampler yields the recorded pure-think sequence (INV-10 spacing):
	// round 1 arrives at completion[0]+7s, round 2 at completion[1]+10s.
	ts := bps[0].ThinkTimeSampler
	if ts == nil {
		t.Fatal("expected a ThinkTimeSampler (recorded think present)")
	}
	if g1, g2 := ts.Sample(nil), ts.Sample(nil); g1 != 7_000_000 || g2 != 10_000_000 {
		t.Errorf("think samples = [%d, %d], want recorded [7e6, 10e6]", g1, g2)
	}

	// Reconstruct absolute per-round inputs via the accumulate buffer.
	sm := workload.NewSessionManager(bps)
	round0 := r0[0]
	if round0.InputLen() != 100 {
		t.Fatalf("round0 input len = %d, want 100", round0.InputLen())
	}
	round0.State = sim.StateCompleted
	round0.ProgressIndex = int64(round0.InputLen()) + 10 // 10 output tokens generated
	fu := sm.OnComplete(round0, 8_000_000)
	if len(fu) != 1 {
		t.Fatalf("round0 follow-ups = %d, want 1", len(fu))
	}
	round1 := fu[0]
	// Round 1 absolute input = 100 + 10 + delta(40) = 150 (the recorded in_1).
	if round1.InputLen() != 150 {
		t.Fatalf("round1 reconstructed input len = %d, want 150 (recorded in_1)", round1.InputLen())
	}
	round1.State = sim.StateCompleted
	round1.ProgressIndex = int64(round1.InputLen()) + 20 // 20 output tokens
	fu = sm.OnComplete(round1, 30_000_000)
	if len(fu) != 1 {
		t.Fatalf("round1 follow-ups = %d, want 1", len(fu))
	}
	round2 := fu[0]
	// Round 2 absolute input = 150 + 20 + delta(45) = 215 (the recorded in_2).
	if round2.InputLen() != 215 {
		t.Fatalf("round2 reconstructed input len = %d, want 215 (recorded in_2)", round2.InputLen())
	}
}

// T10 — CLI corpus paths: .jsonl multi-session with deterministic global IDs in
// file order, "independent"→empty-header mapping, warn+skip of a malformed line
// and a below-min-rounds session, and the no-usable-sessions hard error (writes
// no output files).
func TestConvertWeka_JSONLIndependentAndSkips(t *testing.T) {
	t.Run("jsonl multi-session, independent growth, deterministic IDs", func(t *testing.T) {
		dir := t.TempDir()
		inPath := filepath.Join(dir, "traces.jsonl")
		// Two sessions, each 2 main turns. File order sess-a then sess-b.
		a := `{"id":"sess-a","requests":[{"type":"n","t":0.0,"in":100,"out":10,"api_time":1.0},{"type":"n","t":8.0,"in":150,"out":20,"api_time":1.0}]}`
		b := `{"id":"sess-b","requests":[{"type":"n","t":0.0,"in":60,"out":6,"api_time":0.5},{"type":"n","t":5.0,"in":90,"out":9,"api_time":0.5}]}`
		if err := os.WriteFile(inPath, []byte(a+"\n"+b+"\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		outPrefix := filepath.Join(dir, "out")
		if err := runConvertWeka(inPath, outPrefix, workload.WekaConvertOptions{ContextGrowth: "independent", MinRounds: 1}); err != nil {
			t.Fatalf("runConvertWeka: %v", err)
		}
		trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		// "independent" → empty header value (NOT the literal "independent").
		if trace.Header.SessionContextGrowth != "" {
			t.Errorf("header growth = %q, want empty for independent", trace.Header.SessionContextGrowth)
		}
		if len(trace.Records) != 4 {
			t.Fatalf("records = %d, want 4 (2 sessions × 2 rounds)", len(trace.Records))
		}
		// Global IDs 0..3 in file order (sess-a rounds first, then sess-b).
		for i, r := range trace.Records {
			if r.RequestID != i {
				t.Errorf("record %d RequestID = %d, want %d", i, r.RequestID, i)
			}
		}
		if trace.Records[0].SessionID != "sess-a" || trace.Records[2].SessionID != "sess-b" {
			t.Errorf("session order = %q..%q, want sess-a then sess-b (file order)", trace.Records[0].SessionID, trace.Records[2].SessionID)
		}
	})

	t.Run("skips malformed line and below-min-rounds", func(t *testing.T) {
		dir := t.TempDir()
		inPath := filepath.Join(dir, "traces.jsonl")
		kept := `{"id":"sess-keep","requests":[{"type":"n","t":0.0,"in":100,"out":10,"api_time":1.0},{"type":"n","t":8.0,"in":150,"out":20,"api_time":1.0}]}`
		tooFew := `{"id":"sess-toofew","requests":[{"type":"n","t":0.0,"in":50,"out":5,"api_time":1.0}]}` // 1 round < MinRounds:2
		malformed := `{not json`
		if err := os.WriteFile(inPath, []byte(kept+"\n"+tooFew+"\n"+malformed+"\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		outPrefix := filepath.Join(dir, "out")
		if err := runConvertWeka(inPath, outPrefix, workload.WekaConvertOptions{ContextGrowth: "accumulate", MinRounds: 2}); err != nil {
			t.Fatalf("runConvertWeka should skip bad lines, not fail: %v", err)
		}
		trace, err := workload.LoadTraceV2(outPrefix+".yaml", outPrefix+".csv")
		if err != nil {
			t.Fatalf("load: %v", err)
		}
		if len(trace.Records) != 2 {
			t.Fatalf("records = %d, want 2 (only sess-keep)", len(trace.Records))
		}
		for _, r := range trace.Records {
			if r.SessionID != "sess-keep" {
				t.Errorf("unexpected record from %q; only sess-keep should contribute", r.SessionID)
			}
		}
	})

	t.Run("no usable sessions errors and writes no files", func(t *testing.T) {
		dir := t.TempDir()
		inPath := filepath.Join(dir, "traces.jsonl")
		// A single-round session under MinRounds:2 → zero usable sessions.
		j := `{"id":"s","requests":[{"type":"n","t":0.0,"in":50,"out":5,"api_time":1.0}]}`
		if err := os.WriteFile(inPath, []byte(j+"\n"), 0o644); err != nil {
			t.Fatalf("write: %v", err)
		}
		outPrefix := filepath.Join(dir, "out")
		if err := runConvertWeka(inPath, outPrefix, workload.WekaConvertOptions{ContextGrowth: "accumulate", MinRounds: 2}); err == nil {
			t.Fatal("expected error when no usable sessions are found, got nil")
		}
		if _, statErr := os.Stat(outPrefix + ".csv"); !os.IsNotExist(statErr) {
			t.Errorf("out.csv should not exist on the failure path; stat err = %v", statErr)
		}
	})
}
