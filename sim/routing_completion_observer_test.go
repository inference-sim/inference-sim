package sim

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type completionEvent struct {
	reqID, instance string
	tick            int64
}

// Every built-in scorer except lora-residency leaves the completion hook
// unsubscribed, so the cluster's OnRequestDone guard is unchanged for every
// configuration that does not name it.
func TestRequestCompletionObserver_BuiltInsDoNotSubscribe(t *testing.T) {
	for _, name := range ValidScorerNames() {
		if name == "lora-residency" {
			continue
		}
		t.Run(name, func(t *testing.T) {
			p := NewRoutingPolicyWithCache("weighted", []ScorerConfig{{Name: name, Weight: 1}}, 16, nil, nil)
			o, ok := p.(RequestCompletionObserver)
			require.True(t, ok, "WeightedScoring implements RequestCompletionObserver")
			assert.False(t, o.ObservesRequestCompletion())
		})
	}
	// Default profile and route-to-holder too.
	for _, pol := range []string{"weighted", "route-to-holder"} {
		p := NewRoutingPolicy(pol, nil, 16, nil)
		o, ok := p.(RequestCompletionObserver)
		require.True(t, ok, pol)
		assert.False(t, o.ObservesRequestCompletion(), pol)
	}
	// Non-scoring policies do not implement the interface at all.
	for _, pol := range []string{"round-robin", "least-loaded", "always-busiest"} {
		_, ok := NewRoutingPolicy(pol, nil, 16, nil).(RequestCompletionObserver)
		assert.False(t, ok, pol)
	}
}

// A subscribing scorer is wired through WeightedScoring and route-to-holder, and
// each event is forwarded exactly once, to every subscriber, in config order.
func TestRequestCompletionObserver_WeightedFanOut(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	var got []string
	restore1 := RegisterCompletionScorerForTest("test-complete-1", scoreQueueDepth, nil,
		func(req *Request, inst string, tick int64) { got = append(got, "1:"+req.ID+"@"+inst) })
	defer restore1()
	restore2 := RegisterCompletionScorerForTest("test-complete-2", scoreQueueDepth, nil,
		func(req *Request, inst string, tick int64) { got = append(got, "2:"+req.ID+"@"+inst) })
	defer restore2()

	cfgs := []ScorerConfig{{Name: "test-complete-1", Weight: 1}, {Name: "queue-depth", Weight: 1}, {Name: "test-complete-2", Weight: 1}}
	for _, pol := range []string{"weighted", "route-to-holder"} {
		got = nil
		p := NewRoutingPolicy(pol, cfgs, 16, nil)
		o := p.(RequestCompletionObserver)
		require.True(t, o.ObservesRequestCompletion(), pol)
		o.OnRequestCompletion(&Request{ID: "r1"}, "i0", 5)
		assert.Equal(t, []string{"1:r1@i0", "2:r1@i0"}, got, pol)
	}
}

// The completion observer shares closure state with its scorer (the point of the
// scorerParts constructor), and routing still notifies the routing observer.
func TestRequestCompletionObserver_SharesScorerState(t *testing.T) {
	// no t.Parallel: mutates package-global scorerRegistry
	inflight := map[string]int{}
	score := func(_ *Request, snaps []RoutingSnapshot) map[string]float64 {
		s := make(map[string]float64, len(snaps))
		for _, sn := range snaps {
			s[sn.ID] = 1.0 / float64(1+inflight[sn.ID])
		}
		return s
	}
	var events []completionEvent
	restore := RegisterCompletionScorerForTest("test-inflight", score,
		func(_ *Request, target string) { inflight[target]++ },
		func(req *Request, inst string, tick int64) {
			inflight[inst]--
			events = append(events, completionEvent{req.ID, inst, tick})
		})
	defer restore()

	p := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "test-inflight", Weight: 1}}, 16, nil)
	state := &RouterState{Snapshots: []RoutingSnapshot{{ID: "a"}, {ID: "b"}}}
	d1 := p.Route(&Request{ID: "r1"}, state)
	d2 := p.Route(&Request{ID: "r2"}, state)
	assert.Equal(t, "a", d1.TargetInstance)
	assert.Equal(t, "b", d2.TargetInstance, "observer raised a's in-flight count")

	p.(RequestCompletionObserver).OnRequestCompletion(&Request{ID: "r1"}, "a", 100)
	assert.Equal(t, []completionEvent{{"r1", "a", 100}}, events)
	assert.Equal(t, map[string]int{"a": 0, "b": 1}, inflight)
	d3 := p.Route(&Request{ID: "r3"}, state)
	assert.Equal(t, "a", d3.TargetInstance, "completion released a")
}

// lora-residency subscribes to both lifecycle events, through weighted and
// route-to-holder alike.
func TestLoRAResidencySubscribesToStartAndCompletion(t *testing.T) {
	for _, pol := range []string{"weighted", "route-to-holder"} {
		p := NewRoutingPolicyWithCache(pol, []ScorerConfig{{Name: "lora-residency", Weight: 1}}, 16, nil, nil)
		c, ok := p.(RequestCompletionObserver)
		require.True(t, ok, pol)
		assert.True(t, c.ObservesRequestCompletion(), pol)
		st, ok := p.(RequestStartObserver)
		require.True(t, ok, pol)
		assert.True(t, st.ObservesRequestStart(), pol)
	}
}
