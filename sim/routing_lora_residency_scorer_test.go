package sim

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// residencyRig drives a weighted policy holding only the lora-residency scorer
// through the same entry points the cluster uses: Route (which runs the
// routing observer), OnRequestStart and OnRequestCompletion.
type residencyRig struct {
	t     *testing.T
	p     RoutingPolicy
	snaps []RoutingSnapshot
}

func newResidencyRig(t *testing.T, maxLoras int, ids ...string) *residencyRig {
	t.Helper()
	r := &residencyRig{t: t, p: NewRoutingPolicyWithCache("weighted",
		[]ScorerConfig{{Name: "lora-residency", Weight: 1}}, 16, nil, nil)}
	for _, id := range ids {
		r.snaps = append(r.snaps, RoutingSnapshot{ID: id, MaxLoras: maxLoras, ActiveAdapters: map[string]int{}})
	}
	return r
}

func (r *residencyRig) scores(adapter string, tick int64) map[string]float64 {
	req := &Request{ID: "probe", Adapter: adapter, ArrivalTime: tick}
	return newScorerParts("lora-residency", 16, nil).score(req, r.snaps) // fresh, stateless probe
}

// route sends req to target through the policy's observer path by making target
// the only candidate, as the cluster would after picking it.
func (r *residencyRig) route(id, adapter, target string, tick int64) *Request {
	req := &Request{ID: id, Adapter: adapter, ArrivalTime: tick}
	var only []RoutingSnapshot
	for _, s := range r.snaps {
		if s.ID == target {
			only = append(only, s)
		}
	}
	require.NotEmpty(r.t, only, "unknown target %s", target)
	d := r.p.Route(req, &RouterState{Snapshots: only, Clock: tick})
	require.Equal(r.t, target, d.TargetInstance)
	return req
}

func (r *residencyRig) start(req *Request, inst string, tick int64) {
	r.p.(RequestStartObserver).OnRequestStart(req, inst, tick)
}

func (r *residencyRig) complete(req *Request, inst string, tick int64) {
	r.p.(RequestCompletionObserver).OnRequestCompletion(req, inst, tick)
}

// score asks the policy's own scorer (with its accumulated state) to rank all
// instances for adapter.
func (r *residencyRig) score(adapter string, tick int64) map[string]float64 {
	ws := r.p.(*WeightedScoring)
	return ws.scorers[0](&Request{ID: "probe", Adapter: adapter, ArrivalTime: tick}, r.snaps)
}

func TestLoRAResidency_ServedAdapterAttractsNextRequest(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.start(req, "i0", 200)
	r.complete(req, "i0", 300)
	s := r.score("a", 400)
	assert.Greater(t, s["i0"], s["i1"], "i0 served a, so it should be preferred: %v", s)
}

// A routed request that has not produced its first token is pending: it does
// not make the adapter look resident, and an abort leaves no trace.
func TestLoRAResidency_RoutedIsNotResidentUntilFirstToken(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.complete(req, "i0", 150) // never started, e.g. timed out in the queue
	s := r.score("b", 200)
	assert.Equal(t, s["i0"], s["i1"], "an unstarted request must not occupy i0's slot: %v", s)
}

// A re-prefill after preemption fires the first-token event again; it must not
// count the request as running twice, or the adapter would stay pinned and i0
// would look as blocked as i1, whose only slot really is running.
func TestLoRAResidency_RepeatedFirstTokenCountedOnce(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.start(req, "i0", 200)
	r.start(req, "i0", 250) // re-prefill after preemption
	r.complete(req, "i0", 300)
	busy := r.route("r2", "c", "i1", 310)
	r.start(busy, "i1", 320) // still running
	s := r.score("b", 400)
	assert.Greater(t, s["i0"], s["i1"], "a is idle on i0, while i1 is blocked: %v", s)
}

// A first token on an instance other than the one the scorer routed to is not
// evidence about the routed instance; a completion reported anywhere settles
// the request, so it does not stay pending on i0 forever.
func TestLoRAResidency_EventsForAnotherInstance(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.start(req, "i1", 200)
	s := r.score("a", 250)
	assert.Greater(t, s["i0"], s["i1"], "still pending on i0, which attracts a: %v", s)
	r.complete(req, "i1", 300)
	s = r.score("a", 400)
	assert.Equal(t, s["i0"], s["i1"], "settled without ever starting on i0: %v", s)
}

// The scorer reads only router-observable inputs: contradicting ground truth in
// ResidentAdapters must not change its scores.
func TestLoRAResidency_IgnoresResidentAdapters(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	before := r.score("a", 100)
	r.snaps[1].ResidentAdapters = map[string]bool{"a": true}
	after := r.score("a", 100)
	assert.Equal(t, before, after, "ResidentAdapters must not influence lora-residency")
}

// ActiveAdapters protects an adapter estimated on GPU: with its only slot held
// by an active adapter, an instance counts as blocked.
func TestLoRAResidency_ActiveAdaptersProtect(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	for _, inst := range []string{"i0", "i1"} {
		req := r.route("r-"+inst, "a-"+inst, inst, 100)
		r.start(req, inst, 200)
		r.complete(req, inst, 300)
	}
	s := r.score("b", 400)
	assert.Equal(t, s["i0"], s["i1"], "both offer an idle victim: %v", s)
	r.snaps[0].ActiveAdapters = map[string]int{"a-i0": 1}
	s = r.score("b", 400)
	assert.Greater(t, s["i1"], s["i0"], "i0's only slot is active, so i0 is blocked: %v", s)
}

func TestLoRAResidency_BaseModelNeutral(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.start(req, "i0", 200)
	s := r.score("", 300)
	assert.Equal(t, 1.0, s["i0"])
	assert.Equal(t, 1.0, s["i1"])
}

// The probe helper must not share state with the policy.
func TestLoRAResidency_FreshScorerIsStateless(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	req := r.route("r1", "a", "i0", 100)
	r.start(req, "i0", 200)
	s := r.scores("a", 300)
	assert.Equal(t, s["i0"], s["i1"])
}

// route-to-holder delegates both lifecycle events to its inner weighted policy.
func TestLoRAResidency_ThroughRouteToHolder(t *testing.T) {
	p := NewRoutingPolicyWithCache("route-to-holder", []ScorerConfig{{Name: "lora-residency", Weight: 1}}, 16, nil, nil)
	snaps := []RoutingSnapshot{
		{ID: "i0", MaxLoras: 1, ActiveAdapters: map[string]int{}},
		{ID: "i1", MaxLoras: 1, ActiveAdapters: map[string]int{}},
	}
	req := &Request{ID: "r1", Adapter: "a", ArrivalTime: 100}
	d := p.Route(req, &RouterState{Snapshots: snaps[:1], Clock: 100})
	require.Equal(t, "i0", d.TargetInstance)
	p.(RequestStartObserver).OnRequestStart(req, "i0", 200)
	p.(RequestCompletionObserver).OnRequestCompletion(req, "i0", 300)
	inner := p.(*RouteToHolder).inner.(*WeightedScoring)
	s := inner.scorers[0](&Request{ID: "probe", Adapter: "a", ArrivalTime: 400}, snaps)
	assert.Greater(t, s["i0"], s["i1"], "the started-then-finished a must be resident on i0: %v", s)
}

// Capacity comes from each snapshot's MaxLoras: i0 has a free second slot, i1's
// only slot is taken, so i0 is cheaper even though its adapter is hotter.
func TestLoRAResidency_CapacityFromMaxLoras(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	r.snaps[0].MaxLoras = 2
	for i, inst := range []string{"i0", "i0", "i0", "i1"} {
		adapter := map[string]string{"i0": "hot", "i1": "cold"}[inst]
		req := r.route(fmt.Sprintf("r%d", i), adapter, inst, int64(100+i))
		r.start(req, inst, int64(200+i))
		r.complete(req, inst, int64(300+i))
	}
	s := r.score("new", 400)
	assert.Greater(t, s["i0"], s["i1"], "i0 has a free slot at max_loras=2: %v", s)
}

// Base-model requests must not occupy a slot in the estimate. With two slots,
// i0 serves x and then a base request, i1 serves x; a phantom base entry would
// fill i0 and make x its eviction victim, so the two would no longer tie.
func TestLoRAResidency_BaseRequestsNotTracked(t *testing.T) {
	r := newResidencyRig(t, 2, "i0", "i1")
	for i, step := range []struct{ adapter, inst string }{{"x", "i0"}, {"x", "i1"}, {"", "i0"}} {
		req := r.route(fmt.Sprintf("r%d", i), step.adapter, step.inst, int64(100+i))
		r.start(req, step.inst, int64(200+i))
		r.complete(req, step.inst, int64(300+i))
	}
	s := r.score("y", 400)
	assert.Equal(t, s["i0"], s["i1"], "a base-model request left a phantom slot on i0: %v", s)
}

// Review of PR #59, finding 4: demand is evaluated at the routing clock, not
// at arrival. Both requests arrive at 0; a is routed at 600 s (gateway queueing,
// say) and x at 0, so at 600 s a is hot and x has decayed ten half-lives:
// evicting x (on i1) is the cheaper choice.
func TestLoRAResidency_DemandAtRoutingClock(t *testing.T) {
	r := newResidencyRig(t, 1, "i0", "i1")
	const s600 = 600_000_000 // microseconds
	for _, step := range []struct {
		id, adapter, inst string
		clock             int64
	}{{"rx", "x", "i1", 0}, {"ra", "a", "i0", s600}} {
		req := &Request{ID: step.id, Adapter: step.adapter, ArrivalTime: 0}
		var only []RoutingSnapshot
		for _, s := range r.snaps {
			if s.ID == step.inst {
				only = append(only, s)
			}
		}
		require.Equal(t, step.inst, r.p.Route(req, &RouterState{Snapshots: only, Clock: step.clock}).TargetInstance)
		r.start(req, step.inst, step.clock+1)
		r.complete(req, step.inst, step.clock+2)
	}
	ws := r.p.(*WeightedScoring)
	ws.clockSetters[0](s600)
	// The probe also arrived at 0, so scoring at arrival rather than at the
	// routing clock would see neither adapter's demand decayed.
	s := ws.scorers[0](&Request{ID: "probe", Adapter: "y", ArrivalTime: 0}, r.snaps)
	assert.Greater(t, s["i1"], s["i0"], "x has decayed by the routing clock, a has not: %v", s)
}
