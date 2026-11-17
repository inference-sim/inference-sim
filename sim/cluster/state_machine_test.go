package cluster

import (
	"fmt"
	"math/rand"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

type constLen int

func (c constLen) Sample(*rand.Rand) int { return int(c) }

// SIM-8: a session round that was planned and never issued has exactly one fate, not started,
// and a row. Round 0 times out, which cancels the session, so rounds 1 and 2 are never issued.
func TestStateMachine_UnissuedSessionRoundsEndNotStarted(t *testing.T) {
	bp := workload.SessionBlueprint{SessionID: "s1", ClientID: "c", MaxRounds: 3, ThinkTimeUs: 1000,
		Horizon: 5_000_000, InputSampler: constLen(16), OutputSampler: constLen(4),
		Streams: workload.SessionStreams(42, "s1"), TenantID: "t", SLOClass: "standard"}
	sm := workload.NewSessionManager([]workload.SessionBlueprint{bp})
	reqs := fixedRequests(1, 64, 20_000, 0)
	reqs[0].SessionID, reqs[0].RoundIndex = "s1", 0
	reqs[0].Deadline = 100_000 // far shorter than 20,000 decode steps
	cs := newTestCluster(newTestDeploymentConfig(1), NewSliceRequestSource(reqs), sm.OnComplete)
	cs.SetPlannedRounds(sm.PlannedUnissued)
	mustRun(t, cs)

	tally := cs.Fates().Tally()
	if tally.ByFate[sim.FateTimedOut] != 1 || tally.ByFate[sim.FateNotStarted] != 2 {
		t.Fatalf("fates = %+v, want 1 timed out and 2 not started", tally.ByFate)
	}
	if tally.Created != 3 || tally.Total() != 3 {
		t.Errorf("created %d, fates %d: planned rounds must be counted on both sides", tally.Created, tally.Total())
	}
	for r := 1; r <= 2; r++ {
		id := fmt.Sprintf("session_s1_round_%d_unissued", r)
		rec, ok := cs.Fates().Fate(id)
		if !ok || rec.Fate != sim.FateNotStarted {
			t.Errorf("round %d (%s): fate %+v ok=%v, want not_started", r, id, rec, ok)
		}
		if f := cs.RequestFacts()[id]; f.SessionID != "s1" || f.RoundIndex != r {
			t.Errorf("round %d facts = %+v, want session s1 round %d", r, f, r)
		}
	}
}

// SIM-8: a session that issues every round plans none beyond them.
func TestStateMachine_ACompleteSessionPlansNothingExtra(t *testing.T) {
	bp := workload.SessionBlueprint{SessionID: "s2", ClientID: "c", MaxRounds: 2, ThinkTimeUs: 1000,
		Horizon: 50_000_000, InputSampler: constLen(16), OutputSampler: constLen(4),
		Streams: workload.SessionStreams(42, "s2")}
	sm := workload.NewSessionManager([]workload.SessionBlueprint{bp})
	reqs := fixedRequests(1, 64, 4, 0)
	reqs[0].SessionID = "s2"
	cs := newTestCluster(newTestDeploymentConfig(1), NewSliceRequestSource(reqs), sm.OnComplete)
	cs.SetPlannedRounds(sm.PlannedUnissued)
	mustRun(t, cs)
	if tally := cs.Fates().Tally(); tally.ByFate[sim.FateCompleted] != 2 || tally.Created != 2 || tally.ByFate[sim.FateNotStarted] != 0 {
		t.Errorf("tally = %+v, want 2 created and completed, none not started", tally)
	}
}

// SIM-8: a disaggregated request passes the prefill, handoff and decode states by the written
// edges wherever the horizon cuts it. A move the machine has no edge for fails the run, so a
// sweep of horizons across the request's life proves every stage is entered by an edge, and the
// stage the horizon finds decides the fate.
func TestStateMachine_PDRequestFollowsItsEdgesAtEveryHorizon(t *testing.T) {
	seen := map[sim.Fate]bool{}
	for h := int64(500); h <= 400_000; h += 1_700 {
		cfg := newTestDisaggDeploymentConfig(4, 2, 2)
		cfg.Horizon = h
		editEngine(&cfg, func(c *simConfig) { c.Horizon = h })
		cs, tally := runFates(t, cfg, fixedRequests(3, 512, 3000, 200), &fakeGateway{disaggregate: true})
		if tally.Created != 3 || tally.Total() != 3 {
			t.Fatalf("horizon %d: created %d, fates %d", h, tally.Created, tally.Total())
		}
		for _, rec := range cs.Fates().Records() {
			seen[rec.Fate] = true
		}
	}
	for _, f := range []sim.Fate{sim.FateQueuedAtHorizon, sim.FateInFlightAtHorizon} {
		if !seen[f] {
			t.Errorf("no horizon in the sweep ended a request %s", f)
		}
	}
}

// SIM-8: a preempted request is a state of the machine and resumes on the stage it left.
func TestStateMachine_PreemptedRequestsResume(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	editEngine(&cfg, func(c *simConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(40, 16, 0, 0, 0, 0) })
	cs, tally := runFates(t, cfg, fixedRequests(8, 64, 200, 10), nil)
	if n := cs.Instances()[0].PreemptionCount(); n == 0 {
		t.Fatal("the fixture must force a preemption, and it forced none")
	}
	if tally.ByFate[sim.FateCompleted]+tally.ByFate[sim.FateDropped] != 8 {
		t.Errorf("tally = %+v, want all 8 ended completed or dropped", tally)
	}
}
