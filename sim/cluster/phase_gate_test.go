package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/trace"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// phasedRequests returns n warm-up requests of the same prompt followed by m measured requests
// of that prompt, all recorded at arrival 0 (inference-sim#1926).
func phasedRequests(n, m, prompt, out int) []*sim.Request {
	mk := func(id string, phase sim.Phase) *sim.Request {
		r := &sim.Request{ID: id, InputTokens: make([]sim.TokenID, prompt), OutputTokens: make([]sim.TokenID, out),
			State: sim.StateQueued, Phase: phase}
		for j := range r.InputTokens {
			r.InputTokens[j] = sim.TokenID(j + 1) // the same prompt, so the warm-up leaves it cached
		}
		return r
	}
	var reqs []*sim.Request
	for i := 0; i < n; i++ {
		reqs = append(reqs, mk(fmt.Sprintf("warm_%d", i), sim.PhaseWarmup))
	}
	for i := 0; i < m; i++ {
		reqs = append(reqs, mk(fmt.Sprintf("meas_%d", i), "measured"))
	}
	return reqs
}

func phasedCluster(t *testing.T, reqs []*sim.Request, phases []trace.Phase, horizon int64) *ClusterSimulator {
	t.Helper()
	gate, initial, err := workload.NewPhaseGate(phases, reqs)
	if err != nil {
		t.Fatal(err)
	}
	cfg := newTestDeploymentConfig(1)
	if horizon > 0 {
		cfg.Horizon = horizon
		editEngine(&cfg, func(c *simConfig) { c.Horizon = horizon })
	}
	cs := newTestCluster(cfg, NewSliceRequestSource(initial), nil)
	cs.SetGate(gate)
	mustRun(t, cs)
	return cs
}

// A phase starts after the one before it drains, and the engine's prefix cache carries across:
// the measured requests arrive after the last warm-up request ended, complete, and hit the prefix
// the warm-up left, which sent together at the start they could not.
func TestPhaseGate_MeasuredPhaseStartsAfterWarmupDrainsWithCacheCarried(t *testing.T) {
	phases := []trace.Phase{{Name: "warmup"}, {Name: "measured", OffsetUs: 1000}}
	cs := phasedCluster(t, phasedRequests(2, 2, 256, 8), phases, 0)
	tally := cs.Fates().Tally()
	if tally.ByFate[sim.FateCompleted] != 4 || tally.Created != 4 {
		t.Fatalf("tally = %+v, want 4 created and completed", tally)
	}
	var lastWarmEnd int64
	for _, rec := range cs.Fates().Records() {
		if rec.ID[:4] == "warm" && rec.At > lastWarmEnd {
			lastWarmEnd = rec.At
		}
	}
	for _, id := range []string{"meas_0", "meas_1"} {
		f := cs.RequestFacts()[id]
		if f.ArrivalUs != lastWarmEnd+1000 {
			t.Errorf("%s arrives at %d, want the warm-up's drain %d plus the phase offset 1000", id, f.ArrivalUs, lastWarmEnd)
		}
	}
	hits := 0
	for _, rm := range cs.AggregatedMetrics().RequestRows() {
		if rm.ID[:4] == "meas" {
			hits += rm.CachedLocalTokens
		}
	}
	if hits == 0 {
		t.Errorf("the measured requests hit no cached token: the warm-up's prefix did not carry across the phases")
	}
}

// A concurrency limit sends that many of a phase's requests and one more as each ends.
func TestPhaseGate_ConcurrencyLimitsTheRequestsInFlight(t *testing.T) {
	phases := []trace.Phase{{Name: "warmup"}, {Name: "measured", Concurrency: 2}}
	cs := phasedCluster(t, phasedRequests(1, 6, 128, 16), phases, 0)
	type span struct{ from, to int64 }
	var spans []span
	for _, rec := range cs.Fates().Records() {
		if rec.ID[:4] == "meas" {
			spans = append(spans, span{cs.RequestFacts()[rec.ID].ArrivalUs, rec.At})
		}
	}
	if len(spans) != 6 {
		t.Fatalf("%d measured requests ended, want 6", len(spans))
	}
	maxIn := 0
	for _, s := range spans {
		n := 0
		for _, o := range spans {
			if o.from <= s.from && s.from < o.to {
				n++
			}
		}
		maxIn = max(maxIn, n)
	}
	if maxIn != 2 {
		t.Errorf("at most %d measured requests were in flight at once, want exactly the limit 2 reached and kept", maxIn)
	}
}

// A run cut short before a phase starts leaves its requests not started, each with a row.
func TestPhaseGate_UnsentRequestsEndNotStarted(t *testing.T) {
	phases := []trace.Phase{{Name: "warmup"}, {Name: "measured"}}
	cs := phasedCluster(t, phasedRequests(1, 3, 256, 5000), phases, 20_000)
	tally := cs.Fates().Tally()
	if tally.ByFate[sim.FateNotStarted] != 3 || tally.Created != 4 || tally.Total() != 4 {
		t.Fatalf("tally = %+v, want the 3 unsent measured requests not started, 4 counted", tally)
	}
	if f := cs.RequestFacts()["meas_0"]; f.Phase != "measured" || f.PromptTokens != 256 {
		t.Errorf("unsent request facts = %+v, want phase measured and 256 prompt tokens", f)
	}
}
