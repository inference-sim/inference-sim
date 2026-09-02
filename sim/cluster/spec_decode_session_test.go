package cluster

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// sessionRun captures the token-accounting observables of one closed-loop
// accumulate session: how many rounds actually ran, each round's input length
// (the accumulate buffer's growth), and each round's terminal ProgressIndex.
// Latency is deliberately NOT captured — speculative decoding is expected to
// change timing, and only timing.
type sessionRun struct {
	rounds     int
	inputLens  []int64
	terminalPI []int64
	completed  int
}

// assertSameSequence compares a spec-decode run's per-round sequence against the K=0
// baseline's, reporting a length mismatch rather than silently skipping the comparison.
func assertSameSequence(t *testing.T, label string, got, want []int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: %d rounds recorded, want %d (K=0 baseline) — got %v, want %v",
			label, len(got), len(want), got, want)
		return
	}
	for i := range got {
		if got[i] != want[i] {
			t.Errorf("%s: round %d = %d, want %d (K=0 baseline)", label, i, got[i], want[i])
		}
	}
}

// runAccumulateSession drives ONE closed-loop accumulate session (constant
// input/output lengths) through the cluster and records its per-round token
// accounting. The session's own RNG is seeded identically on every call, so two
// runs differing only in spec-decode config must produce identical accounting.
func runAccumulateSession(t *testing.T, cfg DeploymentConfig, outLen, maxRounds int) sessionRun {
	t.Helper()

	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type: "constant", Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (input): %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type: "constant", Params: map[string]float64{"value": float64(outLen)},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler (output): %v", err)
	}

	const sessID = "spec_acc_sess"
	bp := workload.SessionBlueprint{
		SessionID:     sessID,
		ClientID:      "spec-client",
		MaxRounds:     maxRounds,
		ContextGrowth: "accumulate",
		ThinkTimeUs:   1000,
		Horizon:       math.MaxInt64,
		InputSampler:  inputSampler,
		OutputSampler: outputSampler,
		RNG:           rand.New(rand.NewSource(1657)),
		Model:         "test-model",
	}
	seed := &sim.Request{
		ID:           sessID + "_r0",
		ArrivalTime:  0,
		InputTokens:  make([]sim.TokenID, 20),
		OutputTokens: make([]sim.TokenID, outLen),
		MaxOutputLen: outLen,
		State:        sim.StateQueued,
		SessionID:    sessID,
		RoundIndex:   0,
	}

	sm := workload.NewSessionManager([]workload.SessionBlueprint{bp})
	out := sessionRun{}
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		// Under PD the per-instance callback also fires for the prefill/decode
		// SUB-requests (which carry no SessionID); the session's own rounds are the
		// parent completions, which the cluster hands us with the SessionID intact.
		if req.SessionID == sessID {
			out.rounds++
			out.inputLens = append(out.inputLens, req.InputLen())
			out.terminalPI = append(out.terminalPI, req.ProgressIndex)
		}
		return sm.OnComplete(req, tick)
	}

	cs := NewClusterSimulator(cfg, NewSliceRequestSource([]*sim.Request{seed}), onDone)
	mustRun(t, cs)
	out.completed = cs.AggregatedMetrics().CompletedRequests
	return out
}

// TestSpecDecode_AccumulateSession_AccountingMatchesBaseline pins BC-1/BC-3/BC-4
// (issue #1657): speculative decoding changes a session's TIMING, never its token
// ACCOUNTING. A decode step that advances g > 1 tokens must land exactly on the
// completion boundary, so every round's terminal ProgressIndex — and therefore the
// accumulate buffer's growth and the session's round count — is identical to a
// K=0 run.
//
// Before the fix, an overshooting final step pushed ProgressIndex past the boundary;
// SessionManager.OnComplete read that back as actualOutputLen > len(OutputTokens) and
// CANCELLED the whole session after round 0 (INV-11 violation: a valid session dropped).
func TestSpecDecode_AccumulateSession_AccountingMatchesBaseline(t *testing.T) {
	const maxRounds = 4

	specConfigs := []struct {
		name string
		k    int
		acc  float64
	}{
		// The reported repro: K=2, α=0.755 (InferenceX Kimi-K3 H200 MTP run).
		{name: "k2_acc0.755", k: 2, acc: 0.755},
		// Whole-number rate (g=3 exactly) and a wide verify block.
		{name: "k2_acc1.0", k: 2, acc: 1.0},
		{name: "k5_acc0.8", k: 5, acc: 0.8},
		// Fractional carry that straddles the boundary from several offsets.
		{name: "k3_acc0.5", k: 3, acc: 0.5},
	}
	// Output lengths 1..13 sweep every phase of the carry relative to the
	// boundary; 4 and 9 are the lengths that cancelled the session pre-fix at
	// K=2/α=0.755.
	outLens := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}

	for _, outLen := range outLens {
		baseCfg := newTestDeploymentConfig(1)
		base := runAccumulateSession(t, baseCfg, outLen, maxRounds)

		// The baseline itself must be a non-vacuous, complete session.
		if base.rounds != maxRounds {
			t.Fatalf("outLen=%d: baseline (K=0) ran %d rounds, want %d — test setup is vacuous",
				outLen, base.rounds, maxRounds)
		}

		for _, sc := range specConfigs {
			t.Run(fmt.Sprintf("outLen%d/%s", outLen, sc.name), func(t *testing.T) {
				cfg := newTestDeploymentConfig(1)
				cfg.SpeculativeConfig = sim.SpeculativeConfig{K: sc.k, Acceptance: sc.acc, Method: "mtp"}
				got := runAccumulateSession(t, cfg, outLen, maxRounds)

				// BC-4 / INV-11: the session completes every round; it is never cancelled.
				if got.rounds != maxRounds {
					t.Errorf("rounds = %d, want %d (session must not be cancelled by a spec-decode overshoot)",
						got.rounds, maxRounds)
				}
				if got.completed != base.completed {
					t.Errorf("CompletedRequests = %d, want %d (baseline)", got.completed, base.completed)
				}
				// BC-1: round 0 lands exactly on the completion boundary (round 0's input
				// length is the seed request's 20 tokens; later rounds' inputs grow, and
				// the baseline comparison below covers them).
				boundary := int64(20 + max(outLen, 1) - 1)
				if len(got.terminalPI) == 0 {
					t.Fatalf("no rounds recorded — cannot check the completion boundary")
				}
				if got.terminalPI[0] != boundary {
					t.Errorf("round 0 terminal ProgressIndex = %d, want %d (InputLen + max(outputLen,1) - 1)",
						got.terminalPI[0], boundary)
				}
				// BC-3: accounting parity with the K=0 baseline, round by round.
				assertSameSequence(t, "terminal ProgressIndex", got.terminalPI, base.terminalPI)
				assertSameSequence(t, "InputLen (accumulate growth must not depend on spec-decode)",
					got.inputLens, base.inputLens)
			})
		}
	}
}

// TestSpecDecode_PDAccumulateSession_AccountingMatchesBaseline is the PD twin of the
// test above: it exercises the Phase-2 decode-sub-request branch of batch formation,
// whose control flow (break-on-<1, floor-at-1) differs from the Phase-1 decode branch.
// outLen=1 specifically pins BC-5: a decode sub-request admitted already AT its
// completion boundary must still be granted one token, or it would be stranded in the
// wait queue forever (INV-8/INV-11).
func TestSpecDecode_PDAccumulateSession_AccountingMatchesBaseline(t *testing.T) {
	const maxRounds = 3

	for _, outLen := range []int{1, 4, 9, 20} {
		baseCfg := newTestDisaggDeploymentConfig(4, 2, 2)
		base := runAccumulateSession(t, baseCfg, outLen, maxRounds)
		if base.rounds != maxRounds {
			t.Fatalf("outLen=%d: PD baseline (K=0) ran %d rounds, want %d — test setup is vacuous",
				outLen, base.rounds, maxRounds)
		}

		for _, k := range []int{2, 5} {
			t.Run(fmt.Sprintf("outLen%d/k%d", outLen, k), func(t *testing.T) {
				cfg := newTestDisaggDeploymentConfig(4, 2, 2)
				cfg.SpeculativeConfig = sim.SpeculativeConfig{K: k, Acceptance: 0.755, Method: "mtp"}
				got := runAccumulateSession(t, cfg, outLen, maxRounds)

				if got.rounds != maxRounds {
					t.Errorf("PD rounds = %d, want %d (session must not be cancelled, and no sub-request stranded)",
						got.rounds, maxRounds)
				}
				assertSameSequence(t, "PD terminal ProgressIndex", got.terminalPI, base.terminalPI)
				assertSameSequence(t, "PD InputLen", got.inputLens, base.inputLens)
			})
		}
	}
}
