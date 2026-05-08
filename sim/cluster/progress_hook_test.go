package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type clusterCollectingHook struct {
	snapshots *[]sim.ProgressSnapshot
}

func (h *clusterCollectingHook) OnProgress(snap sim.ProgressSnapshot) {
	*h.snapshots = append(*h.snapshots, snap)
}

var _ sim.ProgressHook = (*clusterCollectingHook)(nil)

func TestClusterSimulator_ProgressHook_FiresWithInstances(t *testing.T) {
	config := newTestDeploymentConfig(2)
	reqs := newTestRequests(10)
	cs := NewClusterSimulator(config, reqs, nil)

	var snapshots []sim.ProgressSnapshot
	cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 500_000)

	mustRun(t, cs)

	if len(snapshots) == 0 {
		t.Fatal("expected at least one snapshot")
	}
	last := snapshots[len(snapshots)-1]
	if !last.IsFinal {
		t.Error("final snapshot should have IsFinal=true")
	}
	if last.TotalInstances != 2 {
		t.Errorf("expected TotalInstances=2, got %d", last.TotalInstances)
	}
	if len(last.InstanceSnapshots) != 2 {
		t.Errorf("expected 2 InstanceSnapshots, got %d", len(last.InstanceSnapshots))
	}
	finalCount := 0
	for _, s := range snapshots {
		if s.IsFinal {
			finalCount++
		}
	}
	if finalCount != 1 {
		t.Errorf("expected exactly 1 IsFinal snapshot, got %d", finalCount)
	}
}

func TestClusterSimulator_ProgressHook_NilHookNoImpact(t *testing.T) {
	config := newTestDeploymentConfig(2)
	reqs := newTestRequests(10)
	cs := NewClusterSimulator(config, reqs, nil)

	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("expected some completed requests")
	}
}

func TestClusterSimulator_ProgressHook_ZeroIntervalOnlyFinal(t *testing.T) {
	config := newTestDeploymentConfig(1)
	reqs := newTestRequests(5)
	cs := NewClusterSimulator(config, reqs, nil)

	var snapshots []sim.ProgressSnapshot
	cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 0)

	mustRun(t, cs)

	if len(snapshots) != 1 {
		t.Fatalf("expected exactly 1 snapshot (final only) with intervalUs=0, got %d", len(snapshots))
	}
	if !snapshots[0].IsFinal {
		t.Error("the only snapshot should have IsFinal=true")
	}
}

func TestClusterSimulator_ProgressHook_SnapshotClockMonotonicity(t *testing.T) {
	config := newTestDeploymentConfig(2)
	reqs := newTestRequests(20)
	cs := NewClusterSimulator(config, reqs, nil)

	var snapshots []sim.ProgressSnapshot
	cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 100_000)

	mustRun(t, cs)

	for i := 1; i < len(snapshots); i++ {
		if snapshots[i].Clock < snapshots[i-1].Clock {
			t.Errorf("snapshot clocks not monotonically increasing: snapshot[%d].Clock=%d < snapshot[%d].Clock=%d",
				i, snapshots[i].Clock, i-1, snapshots[i-1].Clock)
		}
	}
}

func TestClusterSimulator_ProgressHook_FinalSnapshotOnHorizon(t *testing.T) {
	config := newTestDeploymentConfig(1)
	config.Horizon = 1_000_000
	reqs := newTestRequests(100)
	cs := NewClusterSimulator(config, reqs, nil)

	var snapshots []sim.ProgressSnapshot
	cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 100_000)

	mustRun(t, cs)

	last := snapshots[len(snapshots)-1]
	if !last.IsFinal {
		t.Fatal("last snapshot should be final")
	}
	if last.Clock > config.Horizon {
		t.Errorf("final snapshot Clock=%d exceeds Horizon=%d", last.Clock, config.Horizon)
	}
}

func TestClusterSimulator_ProgressHook_ShedByTier(t *testing.T) {
	// BC-1: periodic (non-final) snapshots have ShedByTier populated.
	// BC-3: final snapshot ShedByTier matches cs.ShedByTier().
	// Also verifies cumulative monotonicity and conservation invariant.
	var requests []*sim.Request
	for i := 0; i < 40; i++ {
		requests = append(requests, &sim.Request{
			ID:           fmt.Sprintf("req_sheddable_%d", i),
			ArrivalTime:  int64(i) * 50_000, // spread over 2_000_000µs to trigger periodic snapshots
			SLOClass:     "sheddable",
			InputTokens:  make([]int, 50),
			OutputTokens: make([]int, 20),
			State:        sim.StateQueued,
		})
	}

	cfg := newTierShedConfig(0, 3) // MinAdmitPriority=3 → sheddable is rejected
	cs := NewClusterSimulator(cfg, requests, nil)

	var snapshots []sim.ProgressSnapshot
	cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 500_000)

	mustRun(t, cs)

	if len(snapshots) < 2 {
		t.Fatalf("expected multiple snapshots (periodic + final), got %d", len(snapshots))
	}

	// BC-1: at least one non-final snapshot has ShedByTier populated.
	foundPeriodicWithShed := false
	for _, snap := range snapshots {
		if !snap.IsFinal && snap.ShedByTier != nil && snap.ShedByTier["sheddable"] > 0 {
			foundPeriodicWithShed = true
			break
		}
	}
	if !foundPeriodicWithShed {
		t.Error("expected at least one non-final snapshot with ShedByTier[\"sheddable\"] > 0")
	}

	// Cumulative monotonicity: ShedByTier counts never decrease across snapshots.
	for i := 1; i < len(snapshots); i++ {
		prev := snapshots[i-1].ShedByTier
		curr := snapshots[i].ShedByTier
		for tier, count := range prev {
			if curr[tier] < count {
				t.Errorf("monotonicity violated at snapshot[%d]: ShedByTier[%q] decreased from %d to %d",
					i, tier, count, curr[tier])
			}
		}
	}

	// Conservation invariant: sum(ShedByTier) == RejectedRequests + GatewayQueueShed + GatewayEvicted.
	for i, snap := range snapshots {
		if snap.ShedByTier == nil {
			continue
		}
		sum := 0
		for _, count := range snap.ShedByTier {
			sum += count
		}
		expected := snap.RejectedRequests + snap.GatewayQueueShed + snap.GatewayEvicted
		if sum != expected {
			t.Errorf("snapshot[%d]: sum(ShedByTier)=%d != RejectedRequests(%d)+GatewayQueueShed(%d)+GatewayEvicted(%d)=%d",
				i, sum, snap.RejectedRequests, snap.GatewayQueueShed, snap.GatewayEvicted, expected)
		}
	}

	// BC-3: final snapshot matches post-simulation accessor (bidirectional equality).
	last := snapshots[len(snapshots)-1]
	if !last.IsFinal {
		t.Fatal("last snapshot should be final")
	}
	if last.ShedByTier == nil {
		t.Fatal("expected ShedByTier to be non-nil in final snapshot when shedding occurred")
	}
	postRun := cs.ShedByTier()
	for tier, count := range postRun {
		if last.ShedByTier[tier] != count {
			t.Errorf("ShedByTier mismatch for %q: snapshot=%d, postRun=%d", tier, last.ShedByTier[tier], count)
		}
	}
	for tier, count := range last.ShedByTier {
		if postRun[tier] != count {
			t.Errorf("snapshot has unexpected tier %q: snapshot=%d, postRun=%d", tier, count, postRun[tier])
		}
	}
}

func TestClusterSimulator_ProgressHook_ShedByTierNilWhenNoShedding(t *testing.T) {
	// BC-2: always-admit produces nil ShedByTier in all snapshots.
	config := newTestDeploymentConfig(2)
	reqs := newTestRequests(10)
	cs := NewClusterSimulator(config, reqs, nil)

	var snapshots []sim.ProgressSnapshot
	cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 100_000)

	mustRun(t, cs)

	for i, snap := range snapshots {
		if snap.ShedByTier != nil {
			t.Errorf("snapshot[%d]: expected ShedByTier=nil with always-admit, got %v", i, snap.ShedByTier)
		}
	}
}

func TestClusterSimulator_ProgressHook_ShedByTierDeterminism(t *testing.T) {
	// BC-4: hook presence does not affect ShedByTier() counts.
	makeRequests := func() []*sim.Request {
		var reqs []*sim.Request
		for i := 0; i < 40; i++ {
			reqs = append(reqs, &sim.Request{
				ID:           fmt.Sprintf("req_sheddable_%d", i),
				ArrivalTime:  int64(i) * 10,
				SLOClass:     "sheddable",
				InputTokens:  make([]int, 50),
				OutputTokens: make([]int, 20),
				State:        sim.StateQueued,
			})
		}
		return reqs
	}

	run := func(withHook bool) map[string]int {
		cfg := newTierShedConfig(0, 3)
		cs := NewClusterSimulator(cfg, makeRequests(), nil)
		if withHook {
			var snapshots []sim.ProgressSnapshot
			cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 100_000)
		}
		mustRun(t, cs)
		return cs.ShedByTier()
	}

	without := run(false)
	with := run(true)

	for tier, count := range without {
		if with[tier] != count {
			t.Errorf("ShedByTier[%q] differs: without=%d, with=%d", tier, count, with[tier])
		}
	}
	for tier, count := range with {
		if without[tier] != count {
			t.Errorf("ShedByTier[%q] extra in with-hook run: %d (absent without hook)", tier, count)
		}
	}
}

func TestClusterSimulator_ProgressHook_Determinism(t *testing.T) {
	run := func(withHook bool) *sim.Metrics {
		config := newTestDeploymentConfig(2)
		config.Horizon = math.MaxInt64
		reqs := newTestRequests(10)
		cs := NewClusterSimulator(config, reqs, nil)
		if withHook {
			var snapshots []sim.ProgressSnapshot
			cs.SetProgressHook(&clusterCollectingHook{snapshots: &snapshots}, 100_000)
		}
		mustRun(t, cs)
		return cs.AggregatedMetrics()
	}

	without := run(false)
	with := run(true)

	if without.CompletedRequests != with.CompletedRequests {
		t.Errorf("CompletedRequests differs: %d vs %d", without.CompletedRequests, with.CompletedRequests)
	}
	if without.TotalInputTokens != with.TotalInputTokens {
		t.Errorf("TotalInputTokens differs: %d vs %d", without.TotalInputTokens, with.TotalInputTokens)
	}
	if without.TotalOutputTokens != with.TotalOutputTokens {
		t.Errorf("TotalOutputTokens differs: %d vs %d", without.TotalOutputTokens, with.TotalOutputTokens)
	}
}
