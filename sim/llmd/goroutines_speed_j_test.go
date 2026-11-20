package llmd

import (
	"runtime"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/cluster"
)

// The router runs on a baton: exactly one of the simulation goroutine and the router's coroutines runs at any time, and the
// coroutines are the only goroutines the router may have (a goroutine started with a bare `go` statement would run outside the baton
// and make the order of operations depend on the Go scheduler, INV-6). This test counts goroutines at every router event of a run
// and requires that, beyond those the process had before the run, there are exactly the live coroutines. It covers the default
// picker and the prefill/decode configuration; a router plugin that starts a raw goroutine in either shows up as an excess.
func TestRouterStartsNoGoroutineOutsideTheBaton(t *testing.T) {
	for _, tc := range []struct {
		name string
		pd   bool
	}{{"default picker", false}, {"prefill-decode picker", true}} {
		t.Run(tc.name, func(t *testing.T) {
			runtime.GC()
			base := runtime.NumGoroutine()
			reqs := testRequests(40, 3, 2000, 0)
			dep := testDeployment(4)
			opts := Options{Model: "test-model"}
			if tc.pd {
				dep.PrefillInstances, dep.DecodeInstances = 2, 2
				setPDPrice(&dep, func(tokens int64) int64 { return 50 + tokens/50 })
				opts.PodLabels = roleLabels
				opts.EPPFlags = append(DefaultEPPFlags(), "--config-text="+pdConfig)
			}
			cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
			gw := New(opts)
			samples, worst := 0, 0
			gw.observe = func(kind, detail string) {
				samples++
				// A coroutine that has just finished still has its goroutine for a moment after the scheduler counts it
				// dead; give such goroutines time to exit before counting an excess.
				extra := 0
				for try := 0; try < 200; try++ {
					if extra = runtime.NumGoroutine() - base - gw.LiveCoroutines(); extra <= 0 {
						break
					}
					runtime.Gosched()
					time.Sleep(50 * time.Microsecond)
				}
				worst = max(worst, extra)
			}
			cs.SetGateway(gw)
			if err := cs.Run(); err != nil {
				t.Fatalf("Run: %v", err)
			}
			if samples == 0 {
				t.Fatal("the run produced no router event to sample")
			}
			if worst > 0 {
				t.Errorf("%d goroutines beyond the baton's coroutines at some router event (samples: %d)", worst, samples)
			}
		})
	}
}
