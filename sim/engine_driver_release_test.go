package sim

import (
	"runtime"
	"testing"
	"weak"

	"github.com/inference-sim/inference-sim/substrate/contracts/conformance/enginetest"
)

// A request that completed long before its client deadline must not be kept alive by the deadline
// event still waiting in the queue: aiperf's default client timeout is 6 h, so on a long agentic
// replay every completed request's prompt stayed in memory until the end of the run, and the
// resident set grew with simulated time (inference-sim#2201). The figure is bounded: after N
// requests complete with their deadlines still pending, none of the N is reachable, whatever N.
func TestEngineDriver_ACompletedRequestIsNotHeldByItsPendingDeadline(t *testing.T) {
	for _, n := range []int{20, 200} {
		eng, err := enginetest.NewFIFO(enginetest.FIFOConfig{MaxRunning: 8, TokenBudget: 4096, StepBaseUs: 100})
		if err != nil {
			t.Fatal(err)
		}
		const deadline = 5_000_000
		d := NewEngineDriver(eng, EngineDriverConfig{Horizon: 10_000_000})
		var live []weak.Pointer[Request]
		for i := 0; i < n; i++ {
			r := &Request{ID: string(rune('a'+i%26)) + string(rune('0'+i/26%10)) + string(rune('A'+i/260)),
				InputTokens: tokens(64), OutputTokens: tokens(2), ArrivalTime: int64(i) * 10, Deadline: deadline}
			live = append(live, weak.Make(r))
			d.InjectArrival(r)
		}
		for d.HasPendingEvents() && d.PeekNextEventTime() < deadline {
			d.ProcessNextEvent()
		}
		if got := d.Metrics.CompletedRequests; got != n {
			t.Fatalf("n=%d: non-vacuity: %d requests completed before the deadlines, want all", n, got)
		}
		if !d.HasPendingEvents() {
			t.Fatalf("n=%d: non-vacuity: no deadline event is pending", n)
		}
		runtime.GC()
		held := 0
		for _, w := range live {
			if w.Value() != nil {
				held++
			}
		}
		runtime.KeepAlive(d) // the driver, and so its event queue, is reachable while the figure is read
		if held != 0 {
			t.Errorf("n=%d: %d of %d completed requests are still reachable through their pending deadline events", n, held, n)
		}
	}
}
