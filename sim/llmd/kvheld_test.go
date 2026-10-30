package llmd

import (
	"fmt"
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
)

// TestKVHeld_DropsReleasedChains: a removal the router receives releases its store's pin, and a
// chain no store still holds leaves the instance's holdings, so a long run's holdings stay the
// engine's cached set rather than every chain it ever published.
func TestKVHeld_DropsReleasedChains(t *testing.T) {
	var reqs []*sim.Request
	for i, r := range testRequests(12, 77, 20000, 320) { // each needs most of the cache
		r.ID = fmt.Sprintf("unique_%d", i)
		reqs = append(reqs, r)
	}
	dep := testDeployment(1)
	editEngine(&dep, func(c *cengine.SimConfig) { c.KVCacheConfig = cengine.NewKVCacheConfig(40, 16, 0, 0, 0, 0) })
	cs := newCluster(dep, cluster.NewSliceRequestSource(reqs), nil)
	gw := New(Options{Model: "test-model", EPPFlags: append(DefaultEPPFlags(), "--config-text="+preciseConfig(""))})
	removed := 0
	gw.observe = func(kind, detail string) {
		if kind == "kvevent" && len(detail) > 5 && detail[len(detail)-4:] == "true" {
			removed++
		}
	}
	cs.SetGateway(gw)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if removed == 0 {
		t.Fatal("no removal was delivered; the check would be vacuous")
	}
	for id, held := range gw.kvHeld {
		for h, n := range held {
			if n <= 0 {
				t.Errorf("instance %s still lists chain %s with %d holdings", id, h, n)
			}
		}
	}
}
