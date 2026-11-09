package cluster

// DEV-2 (design §8.2): the cluster orders the events of one instant by (time, class, sender,
// sender's own count), with no global counter, and refuses two queued events with one key.

import (
	"container/heap"
	"crypto/sha256"
	"fmt"
	"math/rand"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
)

// senderTest is the sender of the events tests schedule by hand. It sorts before every
// production sender, so a test event runs first among the events of its time and class, as it
// did when it was queued before Run.
const senderTest senderID = -1

// scheduleTestEvent queues ev as sent by the test.
func (cs *ClusterSimulator) scheduleTestEvent(ev ClusterEvent) {
	if cs.testSender == nil {
		cs.testSender = &eventSender{id: senderTest}
	}
	cs.schedule(ev, cs.testSender)
}

// The queue pops in key order: time, then class, then sender, then the sender's count. The
// order in which the events were pushed does not enter it.
func TestEventKey_QueueOrdersByTimeClassSenderCount(t *testing.T) {
	want := []eventKey{
		{time: 100, class: 0, sender: senderWorkload, count: 3},
		{time: 100, class: 0, sender: senderFirstInstance, count: 0},
		{time: 100, class: 0, sender: senderFirstInstance + 1, count: 0},
		{time: 100, class: 2, sender: senderRouter, count: 0},
		{time: 100, class: 2, sender: senderRouter, count: 1},
		{time: 100, class: 6, sender: senderKVConnector, count: 7},
		{time: 200, class: -1, sender: senderLifecycle, count: 9},
	}
	rng := rand.New(rand.NewSource(7))
	for trial := 0; trial < 50; trial++ {
		var q ClusterEventQueue
		for _, i := range rng.Perm(len(want)) {
			k := want[i]
			q.Push(clusterEventEntry{event: &fixedPriorityEvent{at: k.time, priority: k.class}, key: k})
		}
		cs := &ClusterSimulator{clusterEvents: q}
		heap.Init(&cs.clusterEvents)
		for i, w := range want {
			if got := cs.popEvent().key; got != w {
				t.Fatalf("trial %d pop %d: key %+v, want %+v", trial, i, got, w)
			}
		}
	}
}

// Known-bad input (X18): two queued events with one key are a defect, and schedule refuses the
// second, naming the key.
func TestEventKey_DuplicateKeyPanics(t *testing.T) {
	cs := &ClusterSimulator{}
	a := &eventSender{id: senderRouter}
	b := &eventSender{id: senderRouter} // a second component claiming the router's identity
	cs.schedule(&fixedPriorityEvent{at: 5, priority: 2}, a)
	defer func() {
		if recover() == nil {
			t.Fatal("a second event with key (5, 2, router, 0) was queued; want a panic")
		}
	}()
	cs.schedule(&fixedPriorityEvent{at: 5, priority: 2}, b)
}

// The same key may be used again once its event has left the queue: the rule is about queued
// events only.
func TestEventKey_KeyFreedWhenPopped(t *testing.T) {
	cs := &ClusterSimulator{}
	a := &eventSender{id: senderRouter}
	cs.schedule(&fixedPriorityEvent{at: 5, priority: 2}, a)
	cs.popEvent()
	a.sent = 0
	cs.schedule(&fixedPriorityEvent{at: 5, priority: 2}, a)
	if cs.clusterEvents.Len() != 1 {
		t.Fatalf("queue holds %d events, want 1", cs.clusterEvents.Len())
	}
}

// permutationDigest runs one cluster scenario with ties at every instant it can make (requests
// arriving together on several instances, disaggregated prefills ending together, a drain that
// redirects queued requests) and digests every request's placement, timing and latency, and
// the order of the events the loop took. permute, when non-nil, gives the queue a random
// scheduling order for events of equal key.
func permutationDigest(t *testing.T, permute *rand.Rand, setup func(*ClusterSimulator)) string {
	t.Helper()
	h := sha256.New()
	runOne := func(name string, cfg DeploymentConfig, reqs []*sim.Request, extra func(*ClusterSimulator)) {
		cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
		cs.permuteSchedule = permute
		cs.observePop = func(k eventKey) { _, _ = fmt.Fprintf(h, "%s pop %+v\n", name, k) }
		if extra != nil {
			extra(cs)
		}
		if setup != nil {
			setup(cs)
		}
		mustRun(t, cs)
		m := cs.AggregatedMetrics()
		for _, r := range reqs {
			_, _ = fmt.Fprintf(h, "%s %s %s %d %v %v\n", name, r.ID, r.AssignedInstance, r.GatewayDispatchTime,
				m.RequestTTFTs[r.ID], m.RequestE2Es[r.ID])
		}
		_, _ = fmt.Fprintf(h, "%s ledger %d %d %d %d %d\n", name, m.CompletedRequests, m.StillQueued, m.StillRunning,
			m.DroppedUnservable, m.TimedOutRequests)
	}
	tied := func(n int, per int64) []*sim.Request {
		reqs := newTestRequests(n)
		for i, r := range reqs {
			r.ArrivalTime = int64(i/4) * per // four requests at each instant
		}
		return reqs
	}

	colocated := newTestDeploymentConfig(3)
	editEngine(&colocated, func(c *simConfig) { c.BatchConfig = cengine.NewBatchConfig(2, 2048, 0) })
	runOne("colocated", colocated, tied(24, 5000), nil)

	runOne("pd", newTestDisaggDeploymentConfig(4, 2, 2), tied(16, 2000), nil)

	drain := newTestDeploymentConfig(2)
	drain.InstanceLifecycle = InstanceLifecycleConfig{DrainPolicy: string(DrainPolicyRedirect)}
	editEngine(&drain, func(c *simConfig) { c.BatchConfig = cengine.NewBatchConfig(1, 2048, 0) })
	runOne("drain", drain, tied(12, 0), func(cs *ClusterSimulator) { cs.scheduleTestEvent(&drainAt{at: 0, inst: 0}) })
	return fmt.Sprintf("%x", h.Sum(nil))
}

// DEV-2's permutation test: the order in which same-instant events are scheduled does not move
// the result. Each permuted run gives every queued event a random salt, which would decide the
// order of two events of equal key; the key is total, so every run equals the unpermuted one.
// Breaking the key (dropping its sender or count) lets the salt decide, and the digests differ.
func TestEventKey_SchedulingOrderPermutationLeavesResultUnchanged(t *testing.T) {
	base := permutationDigest(t, nil, nil)
	for seed := int64(1); seed <= 6; seed++ {
		if got := permutationDigest(t, rand.New(rand.NewSource(seed)), nil); got != base {
			t.Fatalf("permutation seed %d changed the result: %s, unpermuted %s", seed, got, base)
		}
	}
}

// No two events the loop takes in one run share a key (DEV-2's "two messages with the same key
// are a defect"), checked over the permutation scenarios.
func TestEventKey_NoTwoProcessedEventsShareAKey(t *testing.T) {
	seen := map[eventKey]int{}
	run := 0
	permutationDigest(t, nil, func(cs *ClusterSimulator) {
		run++
		r := run
		inner := cs.observePop
		cs.observePop = func(k eventKey) {
			inner(k)
			if prev, ok := seen[k]; ok && prev == r {
				t.Errorf("key %+v taken twice in one run", k)
			}
			seen[k] = r
		}
	})
	if len(seen) == 0 {
		t.Fatal("non-vacuity: no event was taken")
	}
}
