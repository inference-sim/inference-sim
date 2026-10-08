package sim

import "testing"

// TestEmittedOutputLen_Law pins Request.EmittedOutputLen as a LAW over the
// (ProgressIndex, InputLen, len(OutputTokens)) triple rather than a golden value, so the
// expectations hold under a rewrite of the expression (#1893).
//
// The law: BLIS charges output token #1 to prefill completion and every later token to a
// decode step, so a request that emitted N tokens ends at ProgressIndex == InputLen+N-1.
// The emitted count is therefore the decode-step count plus one whenever the request
// stopped short of its oracle budget, clamped into [0, len(OutputTokens)].
//
// Every case is derived from the serving semantics named in its comment — no case's
// expectation comes from running the code.
func TestEmittedOutputLen_Law(t *testing.T) {
	tests := []struct {
		name          string
		inputLen      int
		oracleOutLen  int
		progressIndex int64
		want          int
		why           string
	}{
		{
			name: "normal completion emits its full oracle output",
			// processCompletions fires at completionProgressIndex == InputLen+N-1, so a
			// fully-generated 10-token round sits 9 decode tokens past its input.
			inputLen: 50, oracleOutLen: 10, progressIndex: 59, want: 10,
			why: "9 decode steps + the prefill-charged token #1",
		},
		{
			name: "single-output request completes at the end of prefill",
			// One token, charged entirely to prefill: zero decode steps.
			inputLen: 50, oracleOutLen: 1, progressIndex: 50, want: 1,
			why: "0 decode steps + the prefill-charged token #1",
		},
		{
			name: "PD 1-output decode sub-request already counts its token",
			// A PD decode sub-request starts at ProgressIndex == InputLen and takes one
			// step, landing at InputLen+1 — its decode-step count ALREADY equals its
			// output length, so the prefill token must not be added again.
			inputLen: 50, oracleOutLen: 1, progressIndex: 51, want: 1,
			why: "decode-step count already equals the output length; no double count",
		},
		{
			name: "length-capped round emits one more than its decode steps",
			// Force-completed at the MaxModelLen boundary well short of the oracle budget.
			inputLen: 50, oracleOutLen: 200, progressIndex: 53, want: 4,
			why: "3 decode steps + the prefill-charged token #1 (#1891 parity)",
		},
		{
			name: "capped at the end of prefill still emitted one token",
			// Prefill completed (emitting token #1) and the request was stopped before
			// any decode step. It emitted 1 token, not 0 — the pre-#1893 accumulate law
			// read this as zero output.
			inputLen: 50, oracleOutLen: 200, progressIndex: 50, want: 1,
			why: "prefill completion itself emits output token #1",
		},
		{
			name: "zero-output request emits nothing",
			// A prefill-only request has no output budget at all.
			inputLen: 50, oracleOutLen: 0, progressIndex: 50, want: 0,
			why: "no output budget, so nothing to emit",
		},
		{
			name: "overshoot past the completion boundary is clamped to the budget",
			// Defence-in-depth for INV-1 (#1528): a request must never report MORE
			// output than it was assigned.
			inputLen: 50, oracleOutLen: 10, progressIndex: 70, want: 10,
			why: "INV-1 conservation: never more output than assigned",
		},
		{
			name: "ProgressIndex below InputLen cannot yield a negative count",
			// Unreachable in normal flow; upstream accounting drift must not produce a
			// negative token count that would corrupt a session buffer.
			inputLen: 100, oracleOutLen: 50, progressIndex: 40, want: 0,
			why: "drift clamps to zero, never negative",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := &Request{
				ID:            tc.name,
				InputTokens:   make([]TokenID, tc.inputLen),
				OutputTokens:  make([]TokenID, tc.oracleOutLen),
				ProgressIndex: tc.progressIndex,
			}
			if got := req.EmittedOutputLen(); got != tc.want {
				t.Errorf("EmittedOutputLen() = %d, want %d (%s)", got, tc.want, tc.why)
			}
		})
	}
}

// TestRecordRequestCompletion_CountsEmittedOutputLen pins the R23 single-source-of-truth
// contract that #1893 rests on: the output-token count recorded at completion IS
// Request.EmittedOutputLen. Everything downstream — the length-capped TPOT denominator,
// the accumulate session context-growth law, and the re-export delta law — reads that
// same method, so a site that re-derived its own expression would silently disagree with
// the recorded metrics. This test fails if recordRequestCompletion stops routing through
// the method (the pre-#1893 accumulate path drifted by exactly one token this way).
func TestRecordRequestCompletion_CountsEmittedOutputLen(t *testing.T) {
	tests := []struct {
		name          string
		inputLen      int
		oracleOutLen  int
		progressIndex int64
		lengthCapped  bool
	}{
		{name: "normal completion", inputLen: 50, oracleOutLen: 10, progressIndex: 59},
		{name: "length-capped", inputLen: 50, oracleOutLen: 200, progressIndex: 53, lengthCapped: true},
		{name: "capped at end of prefill", inputLen: 50, oracleOutLen: 200, progressIndex: 50, lengthCapped: true},
		{name: "pd single-output decode sub-request", inputLen: 50, oracleOutLen: 1, progressIndex: 51},
		{name: "zero-output prefill only", inputLen: 50, oracleOutLen: 0, progressIndex: 50},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := SimConfig{
				Horizon:             1_000_000,
				Seed:                42,
				KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
				BatchConfig:         NewBatchConfig(256, 2048, 0),
				LatencyCoeffs:       NewLatencyCoeffs([]float64{1000, 1, 1}, []float64{0, 0, 0}),
				ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, 1, false, "", "roofline", 0),
			}
			s := mustNewSimulator(t, cfg)

			req := &Request{
				ID:             "r",
				InputTokens:    make([]TokenID, tc.inputLen),
				OutputTokens:   make([]TokenID, tc.oracleOutLen),
				State:          StateCompleted,
				LengthCapped:   tc.lengthCapped,
				ProgressIndex:  tc.progressIndex,
				FirstTokenTime: 10000,
			}
			s.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)
			want := req.EmittedOutputLen()

			s.recordRequestCompletion(req)

			if s.Metrics.TotalOutputTokens != want {
				t.Errorf("TotalOutputTokens = %d, want EmittedOutputLen() = %d", s.Metrics.TotalOutputTokens, want)
			}
		})
	}
}
