package workload

import (
	"math/rand"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestReasoningAccumulate_PerSessionMemoryIsLinear demonstrates BC-7/BC-8/BC-9:
// per-session input-token bytes scale linearly with the round count R, not
// quadratically.
//
// Method: for a fixed (prefixTokens, inputLen, outputLen) workload, run
// GenerateReasoningRequests with two different R values, sum the sizes of
// each round's InputTokens slice header view, and report (a) the sum of
// underlying backing array capacities visible to the rounds (peak resident
// per-session bytes), and (b) the legacy O(R²) bound that the eager copy
// produced.
//
// We assert that the post-change peak bytes are within a small constant
// factor of the optimal O(R) total content size (≤3× — Go's append doubles
// capacity, so ≤2× from realloc slack, +1× margin).
func TestReasoningAccumulate_PerSessionMemoryIsLinear(t *testing.T) {
	// Use small per-round sizes so test is fast and the asymptotic shows up
	// clearly. Test asserts a structural property, not absolute numbers, so
	// this is faithful to the real-workload behavior. Hard win-ratio
	// thresholds are asserted per the PR description (MOD-R4-2, #1445).
	cases := []struct {
		name      string
		maxRounds int
		minWin    float64 // hard floor on legacy/new ratio
	}{
		{"R=15", 15, 5.0},
		{"R=50", 50, 20.0},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			spec := &ReasoningSpec{
				MultiTurn: &MultiTurnSpec{
					MaxRounds:     tc.maxRounds,
					ThinkTimeUs:   1000,
					ContextGrowth: "accumulate",
				},
			}
			inputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 0, "min": 100, "max": 100}})
			outputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 0, "min": 50, "max": 50}})

			reqs, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c", "t", "s", "", nil)
			if err != nil {
				t.Fatalf("GenerateReasoningRequests: %v", err)
			}
			if len(reqs) != tc.maxRounds {
				t.Fatalf("got %d rounds, want %d", len(reqs), tc.maxRounds)
			}

			// Content size = sum of all per-round tokens (input + output) that
			// participate in the accumulated context. With per-round input=100
			// and output=50, total content = R × (100 + 50) = 150R.
			//
			// Round N's InputTokens length = 100 + (N) × 150 (R0: 100; R1: 100+50+100=250; ...).
			// Sum over rounds = sum_{N=0..R-1} of (100 + N × 150) = 100R + 150 × R(R-1)/2.
			// Legacy O(R²) eager total bytes = sum of fresh copies = same formula × 8 bytes.

			// New representation: backing array peak. Rounds share a single
			// growable buffer; the LAST round's underlying array IS the buffer.
			// The buffer's capacity = cap of the last round's InputTokens (which
			// is sess.buf.Slice(0, end)).
			lastRound := reqs[len(reqs)-1]
			newPeakCap := cap(lastRound.InputTokens)

			// Content size (input + output) is the optimal O(R) lower bound.
			//   contentTokens = R × (perInput + perOutput) = R × 150
			contentTokens := tc.maxRounds * (100 + 50)

			// Legacy O(R²) bound (independent reproduction of the formula).
			//   legacyTokens = sum_{N=0..R-1} of accumulatedInputLen(N)
			//   accumulatedInputLen(0) = 100; accumulatedInputLen(N) = N × 150 + 100
			legacyTokens := 0
			for n := 0; n < tc.maxRounds; n++ {
				legacyTokens += 100 + n*150
			}

			// The new peak must be within 3× of content size — Go's append
			// growth bound is 2× cap, plus a constant slack.
			if newPeakCap > 3*contentTokens {
				t.Errorf("BC: peak cap = %d, want ≤ 3*content = %d (linear scaling violated)",
					newPeakCap, 3*contentTokens)
			}

			// The new peak must be strictly smaller than the legacy bound for
			// R ≥ 4 (where R²/2 > 3R). Otherwise the storage win is asymptotically
			// zero.
			if tc.maxRounds >= 4 && newPeakCap >= legacyTokens {
				t.Errorf("BC: peak cap = %d >= legacy bound = %d — no storage win",
					newPeakCap, legacyTokens)
			}

			ratio := float64(legacyTokens) / float64(newPeakCap)
			t.Logf("%s: legacy O(R²) tokens = %d, new peak cap = %d, content = %d, win = %.1fx",
				tc.name, legacyTokens, newPeakCap, contentTokens, ratio)
			if ratio < tc.minWin {
				t.Errorf("%s: win ratio %.2fx < minimum %.2fx — storage benefit regressed", tc.name, ratio, tc.minWin)
			}
		})
	}
}

// TestSessionAccumulate_ClosedLoopMemoryIsLinear demonstrates BC-9: closed-loop
// session accumulate also exhibits O(R) per-session memory growth.
func TestSessionAccumulate_ClosedLoopMemoryIsLinear(t *testing.T) {
	const R = 20
	bp := SessionBlueprint{
		SessionID:     "s",
		MaxRounds:     R,
		ContextGrowth: "accumulate",
		ThinkTimeUs:   1000,
		Horizon:       1 << 30,
		InputSampler:  &constantSampler{value: 100},
		OutputSampler: &constantSampler{value: 50},
		RNG:           rand.New(rand.NewSource(7)),
	}
	sm := NewSessionManager([]SessionBlueprint{bp})

	// Round 0: simulated externally — 100 input + 50 actual output.
	req := &sim.Request{
		ID:            "r0",
		SessionID:     "s",
		RoundIndex:    0,
		State:         sim.StateCompleted,
		ProgressIndex: 150, // 100 + 50
		InputTokens:   sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 100),
		OutputTokens:  sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(2)), 50),
	}

	// Drive the session through R rounds.
	for i := 0; i < R-1; i++ {
		follow := sm.OnComplete(req, int64(1000*(i+1)))
		if len(follow) != 1 {
			t.Fatalf("round %d: expected 1 follow-up, got %d", i, len(follow))
		}
		nextReq := follow[0]
		nextReq.State = sim.StateCompleted
		nextReq.ProgressIndex = int64(len(nextReq.InputTokens) + len(nextReq.OutputTokens))
		req = nextReq
	}

	// The final round's InputTokens spans [0, end_of_rR-1_input). Its cap is the
	// session buffer's underlying array capacity.
	finalCap := cap(req.InputTokens)

	// Legacy O(R²) bound: each round's eager copy = sum_{N=0..R-1} (100 + N×150).
	legacyTokens := 0
	for n := 0; n < R; n++ {
		legacyTokens += 100 + n*150
	}
	contentTokens := R * 150 // theoretical optimal

	if finalCap > 3*contentTokens {
		t.Errorf("BC-9: final cap = %d, want ≤ 3*content = %d", finalCap, 3*contentTokens)
	}
	if finalCap >= legacyTokens {
		t.Errorf("BC-9: final cap = %d >= legacy bound = %d — no closed-loop win", finalCap, legacyTokens)
	}
	t.Logf("R=%d closed-loop: legacy O(R²) = %d tokens, new peak = %d, content = %d, win = %.1fx",
		R, legacyTokens, finalCap, contentTokens, float64(legacyTokens)/float64(finalCap))
}

// TestReasoningAccumulate_WithPrefix_MemoryIsLinear verifies that the storage
// win extends to prefix-bearing workloads — i.e., that prefix is seeded into
// the shared buffer (#1445) rather than freshly prepended per round. Without
// this, every round would carry a fresh O(prefix + accumulated) copy.
func TestReasoningAccumulate_WithPrefix_MemoryIsLinear(t *testing.T) {
	const (
		R         = 30
		prefixLen = 500
		inputLen  = 100
		outputLen = 50
	)
	rng := rand.New(rand.NewSource(123))
	spec := &ReasoningSpec{
		MultiTurn: &MultiTurnSpec{
			MaxRounds:     R,
			ThinkTimeUs:   1000,
			ContextGrowth: "accumulate",
		},
	}
	inputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": float64(inputLen), "std_dev": 0, "min": float64(inputLen), "max": float64(inputLen)}})
	outputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": float64(outputLen), "std_dev": 0, "min": float64(outputLen), "max": float64(outputLen)}})
	prefix := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(2)), prefixLen)

	reqs, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c", "t", "s", "", prefix)
	if err != nil {
		t.Fatal(err)
	}

	// All rounds must carry the prefix in their InputTokens.
	for i, r := range reqs {
		if r.PrefixLength != prefixLen {
			t.Errorf("round %d: PrefixLength = %d, want %d", i, r.PrefixLength, prefixLen)
		}
		// Round N's input starts with prefix, then accumulated context.
		if r.InputLen() < int64(prefixLen) {
			t.Errorf("round %d: InputLen %d < prefixLen %d", i, r.InputLen(), prefixLen)
			continue
		}
		head := r.InputTokenSlice(0, int64(prefixLen))
		for j, tok := range head {
			if tok != prefix[j] {
				t.Errorf("round %d: prefix token[%d] = %d, want %d (prefix not seeded into buffer)", i, j, tok, prefix[j])
				break
			}
		}
	}

	// Memory: peak cap of the last round (≈ session buffer's underlying array).
	finalCap := cap(reqs[len(reqs)-1].InputTokens)
	// Legacy O(R²) bound: each round allocates fresh [prefix | accumulated | newInput].
	//   legacy(N) = prefixLen + N × (inputLen + outputLen) + inputLen
	//   sum_{N=0..R-1} legacy(N) = R*prefixLen + R*inputLen + (R(R-1)/2) * (inputLen + outputLen)
	legacy := 0
	for n := 0; n < R; n++ {
		legacy += prefixLen + n*(inputLen+outputLen) + inputLen
	}
	content := prefixLen + R*(inputLen+outputLen)
	if finalCap > 3*content {
		t.Errorf("peak cap = %d > 3*content (%d) — prefix is not shared", finalCap, 3*content)
	}
	if finalCap >= legacy {
		t.Errorf("peak cap = %d >= legacy bound = %d — no storage win with prefix", finalCap, legacy)
	}
	t.Logf("R=%d prefix=%d: legacy=%d, peak=%d, content=%d, win=%.1fx",
		R, prefixLen, legacy, finalCap, content, float64(legacy)/float64(finalCap))
}

// TestReportMemoryWinForPR captures the win factor under a realistic workload
// shape for the PR description (R=15 / R=50 with input=1500 output=425 — the
// shape from the parent issue #1438). Asserts hard floor win ratios so the
// test fails on regression (susiejojo human review, #1445).
func TestReportMemoryWinForPR(t *testing.T) {
	cases := []struct {
		R      int
		minWin float64
	}{
		{15, 5.0},
		{50, 20.0},
	}
	for _, tc := range cases {
		rng := rand.New(rand.NewSource(42))
		spec := &ReasoningSpec{
			MultiTurn: &MultiTurnSpec{
				MaxRounds:     tc.R,
				ThinkTimeUs:   1000,
				ContextGrowth: "accumulate",
			},
		}
		inputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 1500, "std_dev": 0, "min": 1500, "max": 1500}})
		outputSampler, _ := NewLengthSampler(DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 425, "std_dev": 0, "min": 425, "max": 425}})

		reqs, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c", "t", "s", "", nil)
		if err != nil {
			t.Fatal(err)
		}
		newPeak := cap(reqs[len(reqs)-1].InputTokens)
		legacy := 0
		for n := 0; n < tc.R; n++ {
			legacy += 1500 + n*(1500+425)
		}
		content := tc.R * (1500 + 425)
		ratio := float64(legacy) / float64(newPeak)
		t.Logf("PR-MEM open-loop R=%d input=1500 output=425: legacy=%d toks, new=%d toks, content=%d toks, win=%.1fx, peak-bytes=%.1fKB (int=8B)",
			tc.R, legacy, newPeak, content, ratio, float64(newPeak*8)/1024)
		if ratio < tc.minWin {
			t.Errorf("R=%d: win ratio %.2fx < minimum %.2fx — storage benefit regressed", tc.R, ratio, tc.minWin)
		}
	}
}
