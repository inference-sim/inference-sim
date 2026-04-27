package cluster

import (
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
