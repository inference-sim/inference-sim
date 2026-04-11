package cluster

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// newTierTestRequests creates n requests all with the given SLOClass,
// arriving every 100µs starting at time 0.
func newTierTestRequests(n int, sloClass string) []*sim.Request {
	reqs := make([]*sim.Request, n)
	for i := range reqs {
		reqs[i] = &sim.Request{
			ID:          fmt.Sprintf("req_%s_%d", sloClass, i),
			ArrivalTime: int64(i) * 100,
			SLOClass:    sloClass,
			InputTokens: make([]int, 50),
			OutputTokens: make([]int, 20),
			State:       sim.StateQueued,
		}
	}
	return reqs
}

// newTierShedConfig creates a DeploymentConfig with tier-shed admission,
// 2 instances, OverloadThreshold=0 (any load triggers shedding at min priority).
func newTierShedConfig(threshold, minPriority int) DeploymentConfig {
	cfg := newTestDeploymentConfig(2)
	cfg.AdmissionPolicy = "tier-shed"
	cfg.TierShedThreshold = threshold
	cfg.TierShedMinPriority = minPriority
	return cfg
}

// T015 — Invariant: monotonic shedding order under load ramp.
// shed(Sheddable) >= shed(Standard) >= shed(Critical) at simulation end.
func TestTierShed_MonotonicSheddingOrder(t *testing.T) {
	// Create a mixed workload: equal volumes of Critical, Standard, Sheddable.
	// Use high rate to create genuine overload (more requests than cluster can handle).
	const nPerTier = 80
	var requests []*sim.Request
	for i, class := range []string{"critical", "standard", "sheddable"} {
		for j := 0; j < nPerTier; j++ {
			requests = append(requests, &sim.Request{
				ID:           fmt.Sprintf("req_%s_%d", class, j),
				ArrivalTime:  int64(i*nPerTier+j) * 10, // dense arrivals to force overload
				SLOClass:     class,
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			})
		}
	}

	// tier-shed: MinAdmitPriority=3 → Standard and above pass, Sheddable shed.
	cfg := newTierShedConfig(0, 3)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	shedCounts := cs.ShedByTier()
	shedCritical := shedCounts["critical"]
	shedStandard := shedCounts["standard"]
	shedSheddable := shedCounts["sheddable"]

	// Monotonicity invariant: shed(Sheddable) >= shed(Standard) >= shed(Critical)
	if shedSheddable < shedStandard {
		t.Errorf("monotonicity violated: shed(sheddable)=%d < shed(standard)=%d", shedSheddable, shedStandard)
	}
	if shedStandard < shedCritical {
		t.Errorf("monotonicity violated: shed(standard)=%d < shed(critical)=%d", shedStandard, shedCritical)
	}

	// With MinAdmitPriority=3, Critical and Standard should NOT be shed by tier policy.
	if shedCritical > 0 {
		t.Errorf("critical requests should not be shed by tier policy, got shed(critical)=%d", shedCritical)
	}
	if shedStandard > 0 {
		t.Errorf("standard requests should not be shed by tier policy, got shed(standard)=%d", shedStandard)
	}

	// Sheddable should have some rejections (cluster is overloaded).
	if shedSheddable == 0 {
		t.Error("expected some sheddable requests to be shed under overload, got 0")
	}
}

// T016 — Invariant: simulation without tier-shed is unaffected (INV-6 regression guard).
// Two runs with identical seed and default admission produce identical aggregated metrics.
func TestTierShed_NoRegressionWithDefaultPolicy(t *testing.T) {
	requests1 := newTestRequests(50)
	requests2 := newTestRequests(50)

	// Run 1: default admission (always-admit)
	cfg1 := newTestDeploymentConfig(2)
	cs1 := NewClusterSimulator(cfg1, requests1, nil)
	mustRun(t, cs1)

	// Run 2: same config, same seed, same requests — must be byte-identical
	cfg2 := newTestDeploymentConfig(2)
	cs2 := NewClusterSimulator(cfg2, requests2, nil)
	mustRun(t, cs2)

	m1, err1 := json.Marshal(cs1.AggregatedMetrics())
	m2, err2 := json.Marshal(cs2.AggregatedMetrics())
	if err1 != nil || err2 != nil {
		t.Fatalf("json marshal error: %v / %v", err1, err2)
	}
	if !bytes.Equal(m1, m2) {
		t.Error("two identical runs produced different aggregated metrics (INV-6 violated)")
	}
}

// Batch and Background ARE shed by tier-shed when below MinAdmitPriority under overload.
func TestTierShed_BatchBackgroundShedUnderOverload(t *testing.T) {
	const n = 40
	var requests []*sim.Request
	for _, class := range []string{"batch", "background"} {
		for i := 0; i < n; i++ {
			requests = append(requests, &sim.Request{
				ID:           fmt.Sprintf("req_%s_%d", class, i),
				ArrivalTime:  int64(i) * 5, // very dense
				SLOClass:     class,
				InputTokens:  make([]int, 100),
				OutputTokens: make([]int, 50),
				State:        sim.StateQueued,
			})
		}
	}

	cfg := newTierShedConfig(0, 3) // MinAdmitPriority=3 rejects batch(-1) and background(-3)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	// With MinAdmitPriority=3, batch(priority=-1) and background(priority=-3) should be rejected
	if cs.RejectedRequests() == 0 {
		t.Errorf("batch/background should be rejected under overload with MinAdmitPriority=3, got RejectedRequests=0")
	}
}

// Additional: With tier-shed threshold=math.MaxInt32, nothing is ever shed.
func TestTierShed_HighThresholdNeverSheds(t *testing.T) {
	requests := newTierTestRequests(20, "sheddable")
	cfg := newTierShedConfig(math.MaxInt32, 3)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	if shed := cs.ShedByTier()["sheddable"]; shed > 0 {
		t.Errorf("no shedding expected with very high threshold, got %d", shed)
	}
}
