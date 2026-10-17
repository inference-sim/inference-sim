package cluster

import (
	"bytes"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// Shape of the accumulate session driven below. The per-round NEW input and the
// round-0 seed length read the same constant so the blueprint and the seed agree.
const (
	preemptSessionNewInput = 40
	preemptSessionOutLen   = 10
	preemptSessionRounds   = 4
	preemptSessionThinkUs  = 1000
	// Background contenders. Short prompts (so the scheduler admits them all) with
	// long outputs (so their decode growth overruns the tight KV cache below) are
	// what makes batch formation evict rather than merely decline to schedule.
	preemptBgCount    = 8
	preemptBgInputLen = 16
	preemptBgOutLen   = 200
)

// accumulateSessionCapture records what a closed-loop accumulate session actually ran:
// the completed round requests (in round order) and the think time each follow-up was
// generated with — exactly the two inputs ReExportClosedLoopRecords consumes — plus the
// preemption counts that make the comparison below non-vacuous.
type accumulateSessionCapture struct {
	rounds    []*sim.Request
	thinkByID map[string]int64
	// preemptions is every eviction the instance performed; roundEvictions is the
	// subset that evicted one of THIS session's own rounds, which is the state the
	// #1893 contract is about (a request preempted mid-session).
	preemptions    int64
	roundEvictions int
}

// evictionLogPrefix is the batch-formation preemption warning
// (sim/batch_formation.go). Counting session-round evictions means reading that log,
// because preemption is recorded per instance as a COUNT, not per request: there is no
// per-request "was preempted" observable on sim.Request to assert against — the rewind
// deliberately erases its own traces (ProgressIndex, ITL, TTFTSet). If this constant
// stops matching, the per-round non-vacuity check below fails loudly rather than
// quietly passing on zero matches.
const evictionLogPrefix = "preemption: evicting "

// runAccumulateSessionUnderLoad drives ONE closed-loop accumulate session through the
// cluster alongside preemptBgCount contending single-turn requests, with kvBlocks KV
// blocks. Sizing kvBlocks down is what forces the REAL preemption path
// (preemptForTokens in sim/batch_formation.go, which rewinds ProgressIndex to 0 and
// re-queues the victim); everything else about the run is held fixed, including the
// session RNG seed, so two calls differing only in kvBlocks run the same session.
func runAccumulateSessionUnderLoad(t *testing.T, kvBlocks int64) accumulateSessionCapture {
	t.Helper()

	constSampler := func(v int) workload.LengthSampler {
		s, err := workload.NewLengthSampler(workload.DistSpec{
			Type: "constant", Params: map[string]float64{"value": float64(v)},
		})
		if err != nil {
			t.Fatalf("NewLengthSampler(%d): %v", v, err)
		}
		return s
	}

	const sessID = "preempt_acc_sess"
	bp := workload.SessionBlueprint{
		SessionID:     sessID,
		ClientID:      "preempt-client",
		MaxRounds:     preemptSessionRounds,
		ContextGrowth: "accumulate",
		ThinkTimeUs:   preemptSessionThinkUs,
		Horizon:       math.MaxInt64,
		InputSampler:  constSampler(preemptSessionNewInput),
		OutputSampler: constSampler(preemptSessionOutLen),
		RNG:           rand.New(rand.NewSource(1893)),
		Model:         "test-model",
	}
	seed := &sim.Request{
		ID:           sessID + "_r0",
		ArrivalTime:  0,
		InputTokens:  make([]sim.TokenID, preemptSessionNewInput),
		OutputTokens: make([]sim.TokenID, preemptSessionOutLen),
		MaxOutputLen: preemptSessionOutLen,
		State:        sim.StateQueued,
		SessionID:    sessID,
		RoundIndex:   0,
		Model:        "test-model",
	}
	// Contenders arrive while the session is mid-flight (the session's own rounds are
	// spaced by think time), so they compete for KV with a running session round.
	arrivals := []*sim.Request{seed}
	for i := 0; i < preemptBgCount; i++ {
		arrivals = append(arrivals, &sim.Request{
			ID:           "bg_" + string(rune('a'+i)),
			ArrivalTime:  int64(i * 200),
			InputTokens:  make([]sim.TokenID, preemptBgInputLen),
			OutputTokens: make([]sim.TokenID, preemptBgOutLen),
			MaxOutputLen: preemptBgOutLen,
			State:        sim.StateQueued,
			Model:        "test-model",
		})
	}

	cfg := newTestDeploymentConfig(1)
	cfg.KVCacheConfig = sim.NewKVCacheConfig(kvBlocks, 16, 0, 0, 0, 0)

	sm := workload.NewSessionManager([]workload.SessionBlueprint{bp})
	capture := accumulateSessionCapture{thinkByID: map[string]int64{}}
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		if req.SessionID == sessID {
			capture.rounds = append(capture.rounds, req)
		}
		followUps := sm.OnComplete(req, tick)
		for _, f := range followUps {
			// What the caller in cmd/replay.go captures: the think time the follow-up
			// was generated with, as arrival minus the completion clock.
			capture.thinkByID[f.ID] = f.ArrivalTime - tick
		}
		return followUps
	}

	cs := NewClusterSimulator(cfg, NewSliceRequestSource(arrivals), onDone)

	// Capture the eviction warnings instead of letting a few hundred of them flood the
	// test output. Restored unconditionally, so a Fatal inside Run cannot leak it.
	var logBuf bytes.Buffer
	prevOut := logrus.StandardLogger().Out
	logrus.SetOutput(&logBuf)
	defer logrus.SetOutput(prevOut)

	mustRun(t, cs)

	for _, inst := range cs.instances {
		capture.preemptions += inst.PreemptionCount()
	}
	for _, line := range strings.Split(logBuf.String(), "\n") {
		if i := strings.Index(line, evictionLogPrefix); i >= 0 && strings.Contains(line[i:], sessID) {
			capture.roundEvictions++
		}
	}
	return capture
}

// TestPreemption_AccumulateGrowthAndReExport_AreTransparent is the real-path half of
// the #1893 preemption contract: a round driven through the ACTUAL preemption rewind
// must grow its session's accumulate context, and re-export to a TraceV2, exactly as an
// uninterrupted round does.
//
// The workload-package twin (TestReExport_DeltaLawInverse_PreemptedRoundIsTransparent)
// walks a synthetic ProgressIndex trajectory, because the rewind itself lives in package
// sim and sim/workload cannot reach it. This test closes that gap from the one package
// that can see both: it shrinks the KV cache until batch formation genuinely evicts a
// running request — ProgressIndex rewound to 0, KV released, re-queued, re-prefilled and
// re-decoded — and then asserts the session's per-round accounting and its re-exported
// records against a run of the SAME session with an ample cache and no preemption.
//
// Why it must hold: both halves of the round-trip read EmittedOutputLen, a TERMINAL
// quantity derived at completion. A re-run request passes through its decode steps
// twice, so any per-step accumulation would double-count them and the preempted
// session's context would outgrow the control session's from the preempted round on.
func TestPreemption_AccumulateGrowthAndReExport_AreTransparent(t *testing.T) {
	// Ample: the whole session plus every contender fits, so nothing is ever evicted.
	control := runAccumulateSessionUnderLoad(t, 10_000)
	// Tight: 20 blocks = 320 tokens, far under what the contenders' decode growth
	// plus a running session round demand, so batch formation has to evict.
	preempted := runAccumulateSessionUnderLoad(t, 20)

	// Non-vacuity, both ways: the control run must not preempt at all, and the tight run
	// must have evicted one of the SESSION's own rounds mid-generation — not merely a
	// background contender. Without this the agreement below could be two identical runs
	// proving nothing.
	if control.preemptions != 0 {
		t.Fatalf("control run preempted %d times, want 0 — it is not a clean baseline", control.preemptions)
	}
	if preempted.roundEvictions == 0 {
		t.Fatalf("tight-KV run evicted a session round 0 times (%d evictions overall) — no round went through the real rewind, so this test proves nothing",
			preempted.preemptions)
	}

	// Both runs must have completed the whole session; a cancelled session would make
	// the sequence comparison vacuous at a shorter length.
	if len(control.rounds) != preemptSessionRounds {
		t.Fatalf("control ran %d rounds, want %d", len(control.rounds), preemptSessionRounds)
	}
	if len(preempted.rounds) != len(control.rounds) {
		t.Fatalf("preempted run completed %d session rounds, want %d (preemption must not cancel or truncate the session)",
			len(preempted.rounds), len(control.rounds))
	}

	// The accumulate growth law: per-round absolute input, and the terminal index it was
	// derived from, are identical across the rewind.
	for i := range control.rounds {
		if got, want := preempted.rounds[i].InputLen(), control.rounds[i].InputLen(); got != want {
			t.Errorf("round %d absolute input = %d, want %d (preemption changed accumulate context growth)", i, got, want)
		}
		if got, want := preempted.rounds[i].ProgressIndex, control.rounds[i].ProgressIndex; got != want {
			t.Errorf("round %d terminal ProgressIndex = %d, want %d (the rewind must leave the terminal index unchanged)", i, got, want)
		}
		if got, want := preempted.rounds[i].EmittedOutputLen(), control.rounds[i].EmittedOutputLen(); got != want {
			t.Errorf("round %d emitted output = %d, want %d", i, got, want)
		}
	}

	// The re-export delta law is the inverse of that growth, so it too must be blind to
	// the rewind: the same records, field for field.
	controlRecs, err := workload.ReExportClosedLoopRecords(control.rounds, control.thinkByID, "accumulate")
	if err != nil {
		t.Fatalf("ReExportClosedLoopRecords(control): %v", err)
	}
	preemptedRecs, err := workload.ReExportClosedLoopRecords(preempted.rounds, preempted.thinkByID, "accumulate")
	if err != nil {
		t.Fatalf("ReExportClosedLoopRecords(preempted): %v", err)
	}
	if len(preemptedRecs) != len(controlRecs) {
		t.Fatalf("re-exported %d records, want %d", len(preemptedRecs), len(controlRecs))
	}
	for i := range controlRecs {
		got, want := withoutTiming(preemptedRecs[i]), withoutTiming(controlRecs[i])
		if !reflect.DeepEqual(got, want) {
			t.Errorf("round %d re-exported record = %+v, want %+v (the delta law must be blind to preemption)", i, got, want)
		}
	}
	// Guard the normalisation itself: the per-round input DELTA — the field the
	// accumulate delta law actually computes, and the one the #1893 shortfall moved — is
	// compared directly, so a withoutTiming that grew too broad cannot hide a drift.
	for i := range controlRecs {
		if got, want := preemptedRecs[i].InputTokens, controlRecs[i].InputTokens; got != want {
			t.Errorf("round %d re-exported input delta = %d, want %d", i, got, want)
		}
		if got, want := preemptedRecs[i].OutputTokens, controlRecs[i].OutputTokens; got != want {
			t.Errorf("round %d re-exported output = %d, want %d", i, got, want)
		}
	}
}

// withoutTiming zeroes the clock-valued columns of a TraceRecord. Preemption is
// EXPECTED to move every one of them — eviction and re-prefill delay the round, which
// delays the round after it, and the deadline is derived from the send time. Everything
// left is the record's token accounting and metadata, which preemption must not touch.
func withoutTiming(r workload.TraceRecord) workload.TraceRecord {
	r.ArrivalTimeUs = 0
	r.SendTimeUs = 0
	r.FirstChunkTimeUs = 0
	r.LastChunkTimeUs = 0
	r.DeadlineUs = 0
	return r
}
