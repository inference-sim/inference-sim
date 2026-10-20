package cluster

import (
	"container/heap"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// The agentic replay (#1937) through the real cluster on the paths the campaign uses: P/D
// disaggregation, flow control with a queue TTL, admission rejection, and a drained
// instance redirecting its queue. Each asserts conservation over the sends (every request
// the replay sent ends exactly once, and the replay's in-flight count drains) and that a
// stream continues or stops as aiperf's stream would after that outcome.

func agxTrace(id string, reqs ...map[string]any) string {
	b, _ := json.Marshal(map[string]any{"id": id, "models": []string{"m"}, "block_size": 64, "hash_id_scope": "local", "requests": reqs})
	return string(b)
}

func agxReq(t, api float64, in, out int, hash ...int64) map[string]any {
	return map[string]any{"type": "s", "t": t, "api_time": api, "in": in, "out": out, "model": "m", "hash_ids": hash}
}

// agxTestCorpus: two traces, one with a subagent that the root's last turn joins.
func agxTestCorpus(t *testing.T) *workload.AgentXCorpus {
	t.Helper()
	lines := []string{
		agxTrace("A",
			agxReq(0, 0.5, 128, 20, 1, 2),
			map[string]any{"type": "subagent", "t": 0.6, "agent_id": "s", "subagent_type": "Explore", "duration_ms": 2000.0,
				"status": "completed", "models": []string{"m"}, "requests": []map[string]any{
					agxReq(0.6, 0.5, 192, 20, 50, 51, 52), agxReq(1.5, 0.5, 256, 20, 50, 51, 52, 53)}},
			agxReq(1.0, 0.5, 192, 20, 1, 2, 3),
			agxReq(4.0, 0.5, 256, 20, 1, 2, 3, 4)),
		agxTrace("B",
			agxReq(0, 0.3, 64, 10, 9),
			agxReq(0.5, 0.3, 128, 10, 9, 10),
			agxReq(1.0, 0.3, 192, 10, 9, 10, 11)),
	}
	c, err := workload.ReadAgentXCorpus(strings.NewReader(strings.Join(lines, "\n")), 0, true)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

type agxRun struct {
	x    *workload.AgentXReplay
	cs   *ClusterSimulator
	sent []*sim.Request
}

// runAgentXOn drives the replay against a real cluster built from cfg, wired exactly as
// `blis replay --agentic-scenario` wires it.
func runAgentXOn(t *testing.T, cfg DeploymentConfig, conc int, extra func(*ClusterSimulator)) *agxRun {
	t.Helper()
	x, err := workload.NewAgentXReplay(agxTestCorpus(t), workload.AgentXReplayConfig{
		Concurrency: conc, Seed: 42, DurationUs: 60_000_000, GraceUs: 5_000_000, IdleCapUs: 10_000_000,
		MinRatio: 0.3, MaxRatio: 0.3, RequestTimeoutUs: workload.AgentXRequestTimeoutUs,
		MaxModelLen: cfg.MaxModelLen,
	})
	if err != nil {
		t.Fatal(err)
	}
	r := &agxRun{x: x}
	initial := x.Start()
	r.cs = NewClusterSimulator(cfg, NewSliceRequestSource(initial), x.OnComplete)
	r.cs.SetArrivalGate(x.Gate)
	r.cs.SetUnservedHook(x.OnUnserved)
	r.cs.SetArrivalHook(func(req *sim.Request) { r.sent = append(r.sent, req) })
	x.BindHorizon(r.cs.SetHorizon)
	if extra != nil {
		extra(r.cs)
	}
	mustRun(t, r.cs)
	x.Finish(r.cs.Clock())
	return r
}

// checkConservation: every send is injected once and ends exactly once, with the outcome
// counts matching the replay's, and nothing the replay counted in flight is lost.
func (r *agxRun) checkConservation(t *testing.T) map[string]int {
	t.Helper()
	st := r.x.Stats()
	sent := st.WarmupRequests + st.MainRequests + st.SubagentRequests
	if len(r.sent) != sent {
		t.Errorf("cluster saw %d fresh arrivals, the replay sent %d", len(r.sent), sent)
	}
	out := map[string]int{}
	seen := map[*sim.Request]bool{}
	for _, req := range r.sent {
		if seen[req] {
			t.Errorf("request %s injected twice as a fresh arrival", req.ID)
		}
		seen[req] = true
		if req.Agentic == nil || req.Agentic.Outcome == "" {
			t.Errorf("request %s (%+v) never ended", req.ID, req.Agentic)
			continue
		}
		out[req.Agentic.Outcome]++
	}
	if out[sim.AgenticCompleted] != st.Completed || out[sim.AgenticError] != st.Errors ||
		out[sim.AgenticContextOverflow] != st.ContextOverflows || out[sim.AgenticCancelled] != st.Cancelled {
		t.Errorf("outcomes %v disagree with the replay's counts %+v", out, st)
	}
	if r.x.InFlight() != st.Cancelled {
		t.Errorf("replay in-flight count %d at the end, %d cancelled", r.x.InFlight(), st.Cancelled)
	}
	// A cancelled request is one the cluster still holds when the run stops (queued or
	// running on an instance, in transfer, or in the gateway queue) -- not one it lost.
	assertClusterINV1Conservation(t, r.cs, len(r.sent), r.cs.RejectedRequests(), "agentx")
	if held := newClusterLedger(r.cs).inSystem() + st.CancelledRefused; held != st.Cancelled {
		t.Errorf("%d requests cancelled at the end, but the cluster holds %d", st.Cancelled, held)
	}
	return out
}

// streams groups the profiling sends by (tree, stream instance), in send order.
func (r *agxRun) streams() map[string][]*sim.Request {
	m := map[string][]*sim.Request{}
	for _, req := range r.sent {
		if req.Phase == sim.PhaseWarmup {
			continue
		}
		k := fmt.Sprintf("%d/%d/%s", req.Agentic.Tree, req.Agentic.StreamInstance, req.Agentic.Stream)
		m[k] = append(m[k], req)
	}
	for _, v := range m {
		sort.SliceStable(v, func(i, j int) bool { return v[i].ArrivalTime < v[j].ArrivalTime })
	}
	return m
}

// checkStreamsContinueAfterErrors: an error ends that request only; aiperf dispatches the
// stream's next turn as usual (callback_handler.py:445-465), so a stream's sends are
// consecutive turns whatever the outcome of each.
func (r *agxRun) checkStreamsContinueAfterErrors(t *testing.T) (continuedAfterError int) {
	t.Helper()
	for k, ss := range r.streams() {
		for i := 1; i < len(ss); i++ {
			if ss[i].Agentic.Turn != ss[i-1].Agentic.Turn+1 {
				t.Errorf("stream %s: turn %d after turn %d", k, ss[i].Agentic.Turn, ss[i-1].Agentic.Turn)
			}
			if ss[i-1].Agentic.Outcome == sim.AgenticError {
				continuedAfterError++
			}
		}
	}
	return continuedAfterError
}

func TestAgentXThroughPDDisaggregation(t *testing.T) {
	r := runAgentXOn(t, newTestDisaggDeploymentConfig(4, 2, 2), 3, nil)
	out := r.checkConservation(t)
	r.checkStreamsContinueAfterErrors(t)
	// Every P/D parent completion reached the replay (by request ID: the cluster reports
	// a disaggregated request's completion with a copy of it), so the closed loop ran on.
	if out[sim.AgenticCompleted] == 0 || r.x.Stats().MainRequests < 6 || r.x.Stats().SubagentRequests == 0 {
		t.Fatalf("P/D replay stalled: outcomes %v, stats %+v", out, r.x.Stats())
	}
	if len(r.cs.ParentRequests()) == 0 {
		t.Fatal("no request was disaggregated")
	}
}

func TestAgentXThroughFlowControlTTL(t *testing.T) {
	cfg := newFlowControlTTLConfig(1, 2000, "concurrency")
	cfg.FlowControlMaxConcurrency = 1
	r := runAgentXOn(t, cfg, 3, nil)
	out := r.checkConservation(t)
	if r.cs.GatewayExpired() == 0 || out[sim.AgenticError] != r.cs.GatewayExpired() {
		t.Fatalf("%d queue expiries, %d error outcomes", r.cs.GatewayExpired(), out[sim.AgenticError])
	}
	if r.checkStreamsContinueAfterErrors(t) == 0 {
		t.Fatal("no stream sent its next turn after an expired request")
	}
}

func TestAgentXThroughAdmissionRejection(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.AdmissionPolicy = "token-bucket"
	cfg.TokenBucketCapacity = 400
	cfg.TokenBucketRefillRate = 50
	r := runAgentXOn(t, cfg, 3, nil)
	out := r.checkConservation(t)
	if r.cs.RejectedRequests() == 0 || out[sim.AgenticError] != r.cs.RejectedRequests() {
		t.Fatalf("%d admission rejections, %d error outcomes", r.cs.RejectedRequests(), out[sim.AgenticError])
	}
	if r.checkStreamsContinueAfterErrors(t) == 0 {
		t.Fatal("no stream sent its next turn after a rejected request")
	}
}

// drainAt drains an instance with the REDIRECT policy at the first poll (every 1 ms from
// at) that finds requests queued on it, so the drain has work to redirect.
type drainAt struct {
	at   int64
	inst int
}

func (e *drainAt) Timestamp() int64 { return e.at }
func (e *drainAt) Priority() int    { return 9 }
func (e *drainAt) Execute(cs *ClusterSimulator) {
	if cs.instances[e.inst].QueueDepth() == 0 {
		heap.Push(&cs.clusterEvents, clusterEventEntry{event: &drainAt{at: e.at + 1000, inst: e.inst}, seqID: cs.nextSeqID()})
		return
	}
	NewDrainPolicy(DrainPolicyRedirect).Drain(cs.instances[e.inst], cs)
}

func TestAgentXThroughRedirectDrain(t *testing.T) {
	cfg := newTestDeploymentConfig(2)
	cfg.InstanceLifecycle = InstanceLifecycleConfig{DrainPolicy: string(DrainPolicyRedirect)}
	cfg.BatchConfig = sim.NewBatchConfig(1, 2048, 0) // one running request: the rest queue
	cfg.RoutingPolicy = "round-robin"
	redirected := 0
	r := runAgentXOn(t, cfg, 8, func(cs *ClusterSimulator) {
		heap.Push(&cs.clusterEvents, clusterEventEntry{event: &drainAt{at: 0, inst: 0}, seqID: cs.nextSeqID()})
	})
	for _, req := range r.sent {
		if req.Redirected {
			redirected++
		}
	}
	if redirected == 0 {
		t.Fatal("no request was redirected by the drain")
	}
	out := r.checkConservation(t)
	// A redirected request is the same send, served elsewhere: it completes. (Refused as a
	// stale arrival it would be lost, and only a client timeout would end it, hours later.)
	if out[sim.AgenticCompleted] == 0 || out[sim.AgenticError] != 0 {
		t.Fatalf("outcomes %v", out)
	}
	r.checkStreamsContinueAfterErrors(t)
}

// TestAgentXContextOverflowEndsTheStream: a prompt past max_model_len is refused by the
// engine as aiperf's context overflow; that stream sends nothing more (agentic_replay.py
// handle_credit_return), and a root's tree drains and its lane recycles.
func TestAgentXContextOverflowEndsTheStream(t *testing.T) {
	cfg := newTestDeploymentConfig(1)
	cfg.MaxModelLen = 260 // trace A's 192- and 256-token turns overflow with the template
	r := runAgentXOn(t, cfg, 2, nil)
	out := r.checkConservation(t)
	if out[sim.AgenticContextOverflow] == 0 {
		t.Fatalf("no context overflow: %v", out)
	}
	for k, ss := range r.streams() {
		for i, req := range ss {
			if req.Agentic.Outcome == sim.AgenticContextOverflow && i != len(ss)-1 {
				t.Errorf("stream %s sent turn %d after its context overflow", k, ss[i+1].Agentic.Turn)
			}
		}
	}
	if r.x.Stats().TreesStarted <= 2 {
		t.Errorf("no lane recycled after its root overflowed: %+v", r.x.Stats())
	}
}
