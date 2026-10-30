package cluster

import (
	"testing"
	"time"

	sim "github.com/inference-sim/inference-sim/sim"
)

// A lone request whose input fits the KV cache and whose input + output budget does not
// is dropped as unservable at enqueue (R19), rather than admitted to preempt itself at its
// first unallocatable block forever. The boundary is TestHashOnce's Rapid seed
// 467318834185904987: a 625-token prompt (40 blocks of 16) with a 31-token budget needs a 41st
// block of a 40-block cache. The same prompt with a 15-token budget fits exactly 40 blocks and
// completes, so the guard does not drop a sequence that ends on the last slot.
func TestCompatEngine_AnInputPlusBudgetPastTheCacheIsDroppedNotLivelocked(t *testing.T) {
	for _, c := range []struct {
		name               string
		out                int
		completed, dropped int64
	}{
		{"input fits, input + budget needs block 41", 31, 0, 1},
		{"input + budget ends on block 40", 15, 1, 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg := newTestDeploymentConfig(1)
			editEngine(&cfg, func(e *simConfig) { e.TotalKVBlocks = 40 })
			reqs := []*sim.Request{holdTestRequest("r0", 1, 0, 625, c.out)}
			cs := newTestCluster(cfg, NewSliceRequestSource(reqs), nil)
			done := make(chan error, 1)
			go func() { done <- cs.Run() }()
			select {
			case err := <-done:
				if err != nil {
					t.Fatalf("run: %v", err)
				}
			case <-time.After(30 * time.Second):
				t.Fatal("the run did not finish: the request livelocks preempting itself")
			}
			agg := cs.AggregatedMetrics()
			if int64(agg.CompletedRequests) != c.completed || int64(agg.DroppedUnservable) != c.dropped {
				t.Errorf("completed %d, dropped unservable %d; want %d and %d",
					agg.CompletedRequests, agg.DroppedUnservable, c.completed, c.dropped)
			}
			assertClusterINV1Conservation(t, cs, 1, 0, c.name)
		})
	}
}
