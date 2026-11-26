// Original to this repository: H2 of the DP external load-balancing RFC (inference-sim#2699).
// Driver-level pins for the rewritten leader step: one leader drives the shared group's lockstep
// steps in flight (BC-H2-2, BC-H2-5), the member drivers never step one of their own, and every
// request a rank serves completes with the output the workload fixed (BC-H2-4 / INV-1). The engine
// is a real vLLM external data-parallel group behind the dprankview views, as the build site wires
// it; the test stands in for the cluster's event loop with a minimal earliest-event driver pump.

package sim

import (
	"fmt"
	"slices"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm/dprankview"
	"github.com/inference-sim/inference-sim/substrate/contracts/engine"
)

// externalDPGroupEngine builds a dp-rank external-balancing vLLM group at a fixed step price.
func externalDPGroupEngine(t *testing.T, dp int) *vllm.DPGroup {
	return externalDPGroupEngineAsync(t, dp, false)
}

// externalDPGroupEngineAsync builds the group with async scheduling on or off. With async on each
// rank's batch queue holds two steps (InFlightLimit 2), so the leader keeps two lockstep steps in
// flight -- the in-flight path BC-H2-5 is about; with it off, one.
func externalDPGroupEngineAsync(t *testing.T, dp int, async bool) *vllm.DPGroup {
	t.Helper()
	src := fmt.Sprintf("block_size: 16\nmax_num_seqs: 8\nmax_num_batched_tokens: 256\nmax_model_len: 512\n"+
		"num_gpu_blocks: 400\nuse_v2_model_runner: %t\nasync_scheduling: %t\npipeline_parallel_size: 1\n"+
		"enable_prefix_caching: true\ndata_parallel_size: %d\ndata_parallel_external_lb: true\n", async, async, dp)
	var n yaml.Node
	if err := yaml.Unmarshal([]byte(src), &n); err != nil {
		t.Fatal(err)
	}
	f := vllm.Factory{Pricer: func(engine.Spec) (vllm.Pricer, error) { return vllm.FixedPricer(100), nil }}
	eng, err := f.New(engine.Spec{Kind: vllm.Kind, UpstreamVersion: vllm.Version, Config: *n.Content[0]})
	if err != nil {
		t.Fatal(err)
	}
	g, ok := eng.(*vllm.DPGroup)
	if !ok {
		t.Fatalf("a dp=%d external engine is %T, not a *DPGroup", dp, eng)
	}
	return g
}

// externalDPDrivers wraps g's per-rank views in drivers and joins them into one group, rank 0 the
// leader, as sim/cluster's buildDPReplicas does for an external shared-group replica.
func externalDPDrivers(t *testing.T, g *vllm.DPGroup) []*EngineDriver {
	t.Helper()
	views := dprankview.NewEndpointViews(g)
	drivers := make([]*EngineDriver, len(views))
	for k, v := range views {
		drivers[k] = NewEngineDriver(v, EngineDriverConfig{Horizon: 1 << 60})
	}
	for k := range drivers {
		drivers[k].JoinGroup(drivers[0], k)
	}
	return drivers
}

// TestLeaderDrivesGroupInFlightMembersNeverStep pins BC-H2-2 and BC-H2-5: the rewritten leader
// step keeps each lockstep step in flight on the leader driver alone, and the member drivers never
// schedule a group step of their own (their groupInFlight stays empty). Every request routed to a
// rank completes with the workload's output (BC-H2-4 / INV-1), and the per-request output matches a
// direct leaderless engine loop over an identically-assigned group (the leader path does not
// corrupt the engine's own result).
func TestLeaderDrivesGroupInFlightMembersNeverStep(t *testing.T) {
	const dp = 4
	g := externalDPGroupEngine(t, dp)
	drivers := externalDPDrivers(t, g)

	// Distinct prompts, assigned round-robin to the endpoints: request i goes to rank i%dp.
	const n = 24
	reqs := make([]*Request, n)
	for i := range reqs {
		r := &Request{ID: fmt.Sprintf("r%d", i), InputTokens: tokens(20 + i), OutputTokens: tokens(3 + i%4),
			ArrivalTime: int64(i * 10)}
		r.MaxOutputLen = len(r.OutputTokens)
		reqs[i] = r
		drivers[i%dp].InjectArrivalAt(r, r.ArrivalTime)
	}

	// Watch the members' in-flight depth across the whole run: a member that ever stepped the group
	// would hold a batch here.
	memberStepped := false
	checkMembers := func() {
		for _, m := range drivers[1:] {
			if len(m.groupInFlight) != 0 {
				memberStepped = true
			}
		}
	}
	for range 1_000_000 {
		best := -1
		var bt int64
		for i, d := range drivers {
			if !d.HasPendingEvents() {
				continue
			}
			if t := d.PeekNextEventTime(); best == -1 || t < bt {
				best, bt = i, t
			}
		}
		if best == -1 {
			break
		}
		drivers[best].ProcessNextEvent()
		checkMembers()
	}

	if memberStepped {
		t.Error("a member driver held an in-flight group step; only the leader (rank 0) may step the shared group")
	}
	for i, r := range reqs {
		if r.State != StateCompleted {
			t.Errorf("request %s (rank %d) ended in state %q, not completed", r.ID, i%dp, r.State)
			continue
		}
		if want := len(r.OutputTokens); r.GeneratedTokens != want {
			t.Errorf("request %s generated %d tokens, want %d (the workload's output)", r.ID, r.GeneratedTokens, want)
		}
	}

	// The leader path's per-request output must equal a direct leaderless engine loop's over an
	// identically-assigned fresh group: the leader step reproduces the engine's own result.
	want := directLoopOutputs(t, dp, n)
	for _, r := range reqs {
		if got, ok := want[r.ID]; !ok || got != r.GeneratedTokens {
			t.Errorf("request %s generated %d tokens through the leader path, %d (ok=%v) driving the engine directly",
				r.ID, r.GeneratedTokens, got, ok)
		}
	}
}

// TestExternalDP_PerRankKVEventsReachOwnEndpoint pins H2b (BC-H1-4 delivery): the leader routes
// each rank's KV events to that rank's driver sink and no other, so every endpoint's downstream
// index sees only its own rank's blocks. Driving distinct prompts onto two ranks and nothing onto
// the others, each of the two sinks receives store events and the idle ranks' sinks receive none.
func TestExternalDP_PerRankKVEventsReachOwnEndpoint(t *testing.T) {
	const dp = 4
	g := externalDPGroupEngine(t, dp)
	drivers := externalDPDrivers(t, g)

	stores := make([]int, dp)
	for k := range drivers {
		k := k
		if err := drivers[k].SetKVEventSink(func(_ int64, evs []KVEvent) {
			for _, e := range evs {
				if !e.Removed && !e.Cleared {
					stores[k]++
				}
			}
		}); err != nil {
			t.Fatalf("rank %d: SetKVEventSink: %v", k, err)
		}
	}

	// Rank 1 and rank 2 each serve their own prompts; ranks 0 and 3 serve nothing.
	i := 0
	for _, rank := range []int{1, 2} {
		for j := 0; j < 4; j++ {
			r := &Request{ID: fmt.Sprintf("r%d", i), InputTokens: tokens(30 + 7*i), OutputTokens: tokens(3), ArrivalTime: int64(i * 10)}
			r.MaxOutputLen = len(r.OutputTokens)
			drivers[rank].InjectArrivalAt(r, r.ArrivalTime)
			i++
		}
	}

	for range 1_000_000 {
		best := -1
		var bt int64
		for i, d := range drivers {
			if !d.HasPendingEvents() {
				continue
			}
			if t := d.PeekNextEventTime(); best == -1 || t < bt {
				best, bt = i, t
			}
		}
		if best == -1 {
			break
		}
		drivers[best].ProcessNextEvent()
	}

	if stores[1] == 0 || stores[2] == 0 {
		t.Fatalf("rank 1 sink saw %d stores, rank 2 saw %d; each serving endpoint must receive its own KV events", stores[1], stores[2])
	}
	for _, idle := range []int{0, 3} {
		if stores[idle] != 0 {
			t.Errorf("idle rank %d's sink saw %d store events; no rank's events may reach an endpoint that served nothing", idle, stores[idle])
		}
	}
}

// collapsedGroupEngine wraps a shared external DP group as one collapsed instance for the normal
// single-driver kick path: Add places each request on its assigned rank (as the router would place
// it), while Schedule, Complete and everything else step the whole group. It stands in for the
// internal-balancer collapsed instance with a fixed, known per-rank assignment, so the collapsed
// kick and the leader path can run the identical per-rank workload and their step timelines be
// compared (BC-H2-5). Embedding promotes every engine method; only Add is overridden.
type collapsedGroupEngine struct {
	*vllm.DPGroup
	assign map[string]int
}

func (e *collapsedGroupEngine) Add(r engine.Request, now engine.Time) error {
	return e.AddToEndpoint(e.assign[r.ID], r, now)
}

// TestLeaderPathMatchesCollapsedKickUnderAsync is BC-H2-5's evidence: under async scheduling (two
// lockstep steps in flight, the whole point of the in-flight rewrite) the leader path's timeline
// equals the collapsed single-driver kick's over the same engine with the same per-rank
// assignment. It asserts equal max in-flight depth (and that it exceeds one, so the in-flight path
// is actually exercised) and, per request, equal finish time, first-token time and inter-token
// latencies. It fails if groupStepEnd stops re-kicking (requests hang or finish late) or if the
// group is stepped one at a time (depth 1, timeline shifts).
func TestLeaderPathMatchesCollapsedKickUnderAsync(t *testing.T) {
	const dp = 4
	const n = 24
	assign := map[string]int{}
	mkReqs := func() []*Request {
		rs := make([]*Request, n)
		for i := range rs {
			id := fmt.Sprintf("r%d", i)
			r := &Request{ID: id, InputTokens: tokens(20 + i), OutputTokens: tokens(3 + i%4), ArrivalTime: int64(i * 7)}
			r.MaxOutputLen = len(r.OutputTokens)
			rs[i] = r
			assign[id] = i % dp
		}
		return rs
	}

	// Collapsed kick: one driver over the shared group, forced assignment, async on.
	gCollapsed := externalDPGroupEngineAsync(t, dp, true)
	collapsedReqs := mkReqs()
	cd := NewEngineDriver(&collapsedGroupEngine{DPGroup: gCollapsed, assign: assign}, EngineDriverConfig{Horizon: 1 << 60})
	for _, r := range collapsedReqs {
		cd.InjectArrivalAt(r, r.ArrivalTime)
	}
	collapsedDepth := 0
	for range 1_000_000 {
		if !cd.HasPendingEvents() || cd.PeekNextEventTime() > cd.Horizon {
			break
		}
		cd.ProcessNextEvent()
		collapsedDepth = max(collapsedDepth, len(cd.inFlight))
	}

	// Leader path: N view-drivers over a fresh group, same assignment, async on.
	gLeader := externalDPGroupEngineAsync(t, dp, true)
	drivers := externalDPDrivers(t, gLeader)
	leaderReqs := mkReqs()
	for _, r := range leaderReqs {
		drivers[assign[r.ID]].InjectArrivalAt(r, r.ArrivalTime)
	}
	leaderDepth := 0
	for range 1_000_000 {
		best := -1
		var bt int64
		for i, d := range drivers {
			if !d.HasPendingEvents() {
				continue
			}
			if tt := d.PeekNextEventTime(); best == -1 || tt < bt {
				best, bt = i, tt
			}
		}
		if best == -1 {
			break
		}
		drivers[best].ProcessNextEvent()
		leaderDepth = max(leaderDepth, len(drivers[0].groupInFlight))
	}

	if collapsedDepth != leaderDepth {
		t.Errorf("max in-flight depth: collapsed kick %d, leader path %d; the leader must keep as many lockstep steps in flight as the collapsed kick", collapsedDepth, leaderDepth)
	}
	if collapsedDepth < 2 {
		t.Fatalf("async in-flight depth stayed at %d, not above one: the differential does not exercise the two-steps-in-flight path BC-H2-5 is about", collapsedDepth)
	}

	byID := func(rs []*Request) map[string]*Request {
		m := make(map[string]*Request, len(rs))
		for _, r := range rs {
			m[r.ID] = r
		}
		return m
	}
	lead := byID(leaderReqs)
	for _, c := range collapsedReqs {
		l := lead[c.ID]
		if c.State != StateCompleted || l == nil || l.State != StateCompleted {
			t.Errorf("%s: states collapsed=%q leader=%v; both must complete", c.ID, c.State, l)
			continue
		}
		if c.DepartureTime != l.DepartureTime {
			t.Errorf("%s: finish time collapsed=%d leader=%d; the leader path's step ends must match the collapsed kick's", c.ID, c.DepartureTime, l.DepartureTime)
		}
		if c.FirstTokenTime != l.FirstTokenTime {
			t.Errorf("%s: first-token time collapsed=%d leader=%d", c.ID, c.FirstTokenTime, l.FirstTokenTime)
		}
		if !slices.Equal(c.ITL, l.ITL) {
			t.Errorf("%s: inter-token latencies collapsed=%v leader=%v", c.ID, c.ITL, l.ITL)
		}
	}
}

// directLoopOutputs drives a fresh external group with the same round-robin assignment straight
// through Schedule/Complete (no sim drivers), returning each finished request's output-token count.
func directLoopOutputs(t *testing.T, dp, n int) map[string]int {
	t.Helper()
	g := externalDPGroupEngine(t, dp)
	for i := 0; i < n; i++ {
		r := engine.Request{ID: fmt.Sprintf("r%d", i), Prompt: make([]int32, 20+i),
			Output: make([]int32, 3+i%4), MaxOutputTokens: 3 + i%4}
		if err := g.AddToEndpoint(i%dp, r, 0); err != nil {
			t.Fatalf("direct loop: add %s to endpoint %d: %v", r.ID, i%dp, err)
		}
	}
	out := map[string]int{}
	now := engine.Time(0)
	for range 1_000_000 {
		b, err := g.Schedule(now)
		if err != nil {
			t.Fatalf("direct loop: schedule at %d: %v", now, err)
		}
		if b.Empty() {
			if w, ok := g.NextWake(now); ok {
				now = max(w, now+1)
				continue
			}
			return out
		}
		now += g.StepTime(b)
		res, err := g.Complete(b.ID, engine.Outcome{}, now)
		if err != nil {
			t.Fatalf("direct loop: complete at %d: %v", now, err)
		}
		for _, f := range res.Finished {
			out[f.ID] = f.OutputTokens
		}
	}
	t.Fatal("direct loop: the group never went idle")
	return nil
}
