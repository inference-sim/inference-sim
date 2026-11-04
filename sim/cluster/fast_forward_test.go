package cluster

import (
	"container/heap"
	"math"
	"math/rand"
	"testing"
)

type plainEvent struct{ at int64 }

func (e *plainEvent) Timestamp() int64          { return e.at }
func (e *plainEvent) Priority() int             { return 0 }
func (e *plainEvent) Execute(*ClusterSimulator) {}

// nextRequestWork walks the heap from its root and prunes below the first non-daemon event on
// each path (#2072). On random queues it must agree with a scan of every event.
func TestNextRequestWork_MatchesLinearScan(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for trial := 0; trial < 2000; trial++ {
		c := &ClusterSimulator{}
		n := rng.Intn(60)
		daemonShare := rng.Float64()
		for i := 0; i < n; i++ {
			at := rng.Int63n(1000)
			var ev ClusterEvent = &plainEvent{at: at}
			if rng.Float64() < daemonShare {
				ev = &CallbackEvent{at: at, priority: PriorityGatewayBackground, daemon: true, fn: func() {}}
			}
			heap.Push(&c.clusterEvents, clusterEventEntry{event: ev, seqID: c.nextSeqID()})
			if rng.Intn(4) == 0 && len(c.clusterEvents) > 0 {
				heap.Pop(&c.clusterEvents)
			}
		}
		want := int64(math.MaxInt64)
		for _, e := range c.clusterEvents {
			if cb, ok := e.event.(*CallbackEvent); ok && cb.daemon {
				continue
			}
			want = min(want, e.event.Timestamp())
		}
		if got := c.nextRequestWork(); got != want {
			t.Fatalf("trial %d (%d events): nextRequestWork = %d, linear scan = %d", trial, len(c.clusterEvents), got, want)
		}
	}
}
