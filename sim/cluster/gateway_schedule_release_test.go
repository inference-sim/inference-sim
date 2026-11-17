package cluster

import (
	"runtime"
	"testing"
	"weak"
)

// A gateway event that is cancelled stays in the event queue until its time, which can be hours of
// simulated time away (a client deadline), so cancelling must let go of what its closure holds: a
// flight's prompt and router state otherwise stay in memory for the rest of the run
// (inference-sim#2201). The figure is bounded: after N events are cancelled, none of what they
// captured is reachable, whatever N.
func TestScheduleCancelReleasesWhatTheEventCaptured(t *testing.T) {
	for _, n := range []int{20, 200} {
		cs := newTestCluster(newTestDeploymentConfig(1), NewSliceRequestSource(nil), nil)
		type payload struct{ prompt [1 << 14]byte }
		var live []weak.Pointer[payload]
		for i := 0; i < n; i++ {
			p := &payload{}
			live = append(live, weak.Make(p))
			stop := cs.Schedule(cs.clock+1_000_000_000, PriorityGatewayRequest, true, func() { _ = p.prompt[0] })
			stop()
		}
		runtime.GC()
		held := 0
		for _, w := range live {
			if w.Value() != nil {
				held++
			}
		}
		runtime.KeepAlive(cs)
		if held != 0 {
			t.Errorf("n=%d: %d of %d cancelled events still hold what they captured", n, held, n)
		}
	}
}
