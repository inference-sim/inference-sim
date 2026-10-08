package cluster

import (
	"container/heap"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// probeEvent runs check when the cluster reaches its time, at its priority.
type probeEvent struct {
	time  int64
	prio  int
	check func()
}

func (e *probeEvent) Timestamp() int64          { return e.time }
func (e *probeEvent) Priority() int             { return e.prio }
func (e *probeEvent) Execute(*ClusterSimulator) { e.check() }

func pushProbe(cs *ClusterSimulator, time int64, prio int, check func()) {
	heap.Push(&cs.clusterEvents, clusterEventEntry{event: &probeEvent{time, prio, check}, seqID: cs.nextSeqID()})
}

// Review of PR #59, finding 1: an instance step runs at its start time but
// reports completions and first tokens at later times. A routing decision in
// between, or at the same tick, must not see them yet.
func TestLifecycleObservers_NotVisibleBeforeTheirTick(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(nil), nil)
	done := &recordingPolicy{subscribe: true}
	started := &startRecordingPolicy{subscribe: true}
	cs.completionObservers = collectCompletionObservers(done)
	cs.startObservers = collectStartObservers(started)
	inst := cs.instances[0]
	cs.wireOnRequestDone(inst)
	cs.wireOnFirstToken(inst)

	cs.clock = 100 // a step starting at 100 reports both events at 500
	inst.sim.OnFirstToken(&sim.Request{ID: "x"}, 500)
	inst.sim.OnRequestDone(&sim.Request{ID: "x"}, 500)
	var at300, atRouting500 int
	pushProbe(cs, 300, 2, func() { at300 = len(done.events) + len(started.events) })
	pushProbe(cs, 500, 2, func() { atRouting500 = len(done.events) + len(started.events) }) // routing priority
	drainClusterEvents(cs)

	assert.Zero(t, at300, "seen 200us before it happened")
	assert.Zero(t, atRouting500, "a routing decision at the same tick ran after it")
	assert.Len(t, done.events, 1)
	assert.Len(t, started.events, 1)
}

// A notification whose tick has already passed (a dropped request reported at
// the instance's clock) is delivered now, not in the past.
func TestLifecycleObservers_PastTickDeliveredNow(t *testing.T) {
	cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource(nil), nil)
	cs.clock = 900
	cs.scheduleLifecycle(100, func() {})
	require.Len(t, cs.clusterEvents, 1)
	assert.Equal(t, int64(900), cs.clusterEvents[0].event.Timestamp())
}

// Review of PR #59, finding 2: a gateway eviction ends a request on its
// instance without OnRequestDone; completion observers must hear of it, for
// running and queued victims alike.
func TestCompletionObserver_GatewayEvictionNotifies(t *testing.T) {
	dc := newTestDeploymentConfig(1)
	dc.BatchConfig = sim.NewBatchConfig(1, 2048, 0) // one running request; the next queues
	cs := NewClusterSimulator(dc, NewSliceRequestSource(nil), nil)
	rec := &recordingPolicy{subscribe: true}
	cs.completionObservers = collectCompletionObservers(rec)
	inst := cs.instances[0]
	id := string(inst.ID())

	running := &sim.Request{ID: "running", State: sim.StateQueued,
		InputTokens: make([]sim.TokenID, 8), OutputTokens: make([]sim.TokenID, 200), MaxOutputLen: 200}
	queued := &sim.Request{ID: "queued", State: sim.StateQueued,
		InputTokens: make([]sim.TokenID, 8), OutputTokens: make([]sim.TokenID, 4), MaxOutputLen: 4}
	inst.InjectRequest(running)
	inst.InjectRequest(queued)
	for i := 0; i < 10000 && (inst.BatchSize() != 1 || inst.QueueDepth() != 1) && inst.HasPendingEvents(); i++ {
		inst.ProcessNextEvent()
	}
	require.Equal(t, 1, inst.BatchSize(), "precondition: one request running")
	require.Equal(t, 1, inst.QueueDepth(), "precondition: one request queued")

	cs.clock = inst.Clock()
	for _, r := range []*sim.Request{running, queued} {
		(&GatewayEvictionEvent{time: cs.clock, request: r, targetInstance: id}).Execute(cs)
	}
	drainClusterEvents(cs)
	assert.ElementsMatch(t, []string{"running@" + id, "queued@" + id}, rec.events)
}

// Review of PR #59, finding 3: lora-residency models aggregated serving only.
func TestLoRAResidency_RejectedUnderDisaggregation(t *testing.T) {
	residency := []sim.ScorerConfig{{Name: "lora-residency", Weight: 1}}
	for name, set := range map[string]func(*DeploymentConfig){
		"main pool":    func(dc *DeploymentConfig) { dc.RoutingScorerConfigs = residency },
		"prefill pool": func(dc *DeploymentConfig) { dc.PrefillScorerConfigs = residency },
		"decode pool":  func(dc *DeploymentConfig) { dc.DecodeScorerConfigs = residency },
	} {
		dc := newTestDeploymentConfig(4)
		dc.PrefillInstances, dc.DecodeInstances = 2, 2
		set(&dc)
		assert.Panics(t, func() { NewClusterSimulator(dc, NewSliceRequestSource(nil), nil) }, name)
	}
	// Aggregated serving with lora-residency is fine, and so is disaggregation without it.
	dc := newTestDeploymentConfig(2)
	dc.RoutingPolicy = "weighted"
	dc.RoutingScorerConfigs = residency
	assert.NotPanics(t, func() { NewClusterSimulator(dc, NewSliceRequestSource(nil), nil) })
	assert.False(t, namesScorer("lora-residency", []sim.ScorerConfig{{Name: "queue-depth"}}, nil))
}

// lifecycleLog is a round-robin policy recording both lifecycle notifications,
// in delivery order, with their ticks.
type lifecycleLog struct {
	inner sim.RoundRobin
	kinds []string
	ticks []int64
}

func (p *lifecycleLog) Route(req *sim.Request, s *sim.RouterState) sim.RoutingDecision {
	return p.inner.Route(req, s)
}
func (p *lifecycleLog) ObservesRequestStart() bool      { return true }
func (p *lifecycleLog) ObservesRequestCompletion() bool { return true }
func (p *lifecycleLog) OnRequestStart(_ *sim.Request, _ string, tick int64) {
	p.kinds, p.ticks = append(p.kinds, "start"), append(p.ticks, tick)
}
func (p *lifecycleLog) OnRequestCompletion(_ *sim.Request, _ string, tick int64) {
	p.kinds, p.ticks = append(p.kinds, "done"), append(p.ticks, tick)
}

// Second review of PR #59: a 0- or 1-token request's first token and
// completion come from the same prefill step, and the step reports completion
// at its end while the first token carries the output-token processing time.
// In a real run, the start must still be delivered first, and the completion
// at the client-visible completion time.
func TestLifecycleObservers_ShortRequestStartsBeforeItCompletes(t *testing.T) {
	for _, outLen := range []int{0, 1, 5} { // 5: the response ends after its last token's processing
		req := &sim.Request{ID: "short", State: sim.StateQueued, InputTokens: make([]sim.TokenID, 16),
			OutputTokens: make([]sim.TokenID, outLen), MaxOutputLen: outLen}
		cs := NewClusterSimulator(newTestDeploymentConfig(1), NewSliceRequestSource([]*sim.Request{req}), nil)
		log := &lifecycleLog{}
		cs.routingPolicy = log
		cs.completionObservers = collectCompletionObservers(log)
		cs.startObservers = collectStartObservers(log)
		for _, inst := range cs.instances {
			cs.wireOnRequestDone(inst)
			cs.wireOnFirstToken(inst)
		}
		require.NoError(t, cs.Run())

		require.Equal(t, []string{"start", "done"}, log.kinds, "output tokens %d", outLen)
		completion, ok := cs.instances[0].sim.Metrics.RequestCompletionTimes["short"]
		require.True(t, ok, "precondition: the request completed")
		assert.LessOrEqual(t, log.ticks[0], log.ticks[1], "output tokens %d", outLen)
		assert.Equal(t, int64(completion), log.ticks[1], "completion delivered at the client-visible time (output tokens %d)", outLen)
	}
}
