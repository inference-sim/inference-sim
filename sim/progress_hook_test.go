package sim

import (
	"fmt"
	"math"
	"testing"
)

// collectingHook implements ProgressHook by appending snapshots to a slice.
type collectingHook struct {
	snapshots *[]ProgressSnapshot
}

func (h *collectingHook) OnProgress(snap ProgressSnapshot) {
	*h.snapshots = append(*h.snapshots, snap)
}

var _ ProgressHook = (*collectingHook)(nil)

func TestProgressSnapshot_IsValueType(t *testing.T) {
	snap := ProgressSnapshot{
		Clock:          1000,
		TotalCompleted: 5,
		InstanceSnapshots: []InstanceSnapshot{
			{ID: "inst-0", QueueDepth: 3, BatchSize: 2},
		},
	}
	copied := snap
	copied.TotalCompleted = 99

	if snap.TotalCompleted == 99 {
		t.Error("ProgressSnapshot scalar fields are not value-copied")
	}

	// Slice backing array IS shared on simple assignment (expected Go behavior).
	// The real safety guarantee is fresh allocation per callback (BC-7, tested
	// in FreshSlicePerCall).
	copied.InstanceSnapshots[0].QueueDepth = 999
	if snap.InstanceSnapshots[0].QueueDepth != 999 {
		t.Error("slice backing array should be shared on simple assignment")
	}
}

func newTestSimulatorForHook(t *testing.T) *Simulator {
	t.Helper()
	return mustNewSimulator(t, SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 4, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 1000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 0.5, 0.5}, []float64{100, 0.1, 50}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test-model", "", 1, "roofline", 0),
	})
}

func newTestRequest(id string, arrivalTime int64, inputLen, outputLen int) *Request {
	input := make([]int, inputLen)
	output := make([]int, outputLen)
	for i := range input {
		input[i] = i % MaxTokenID
	}
	for i := range output {
		output[i] = i % MaxTokenID
	}
	return &Request{
		ID:           id,
		InputTokens:  input,
		OutputTokens: output,
		ArrivalTime:  arrivalTime,
		MaxOutputLen: outputLen,
		State:        StateQueued,
	}
}

func TestSimulator_ProgressHook_FiresAtInterval(t *testing.T) {
	sim := newTestSimulatorForHook(t)
	sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))

	var snapshots []ProgressSnapshot
	sim.SetProgressHook(&collectingHook{snapshots: &snapshots}, 500_000)

	sim.Run()

	if len(snapshots) == 0 {
		t.Fatal("expected at least one snapshot")
	}
	last := snapshots[len(snapshots)-1]
	if !last.IsFinal {
		t.Error("last snapshot should have IsFinal=true")
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
	for i, s := range snapshots {
		if len(s.InstanceSnapshots) != 1 {
			t.Errorf("snapshot %d: expected 1 InstanceSnapshot, got %d", i, len(s.InstanceSnapshots))
		}
	}
}

func TestSimulator_ProgressHook_NilHookNoImpact(t *testing.T) {
	sim := newTestSimulatorForHook(t)
	sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))
	sim.Run()
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("expected 1 completed request, got %d", sim.Metrics.CompletedRequests)
	}
}

func TestSimulator_ProgressHook_ZeroIntervalOnlyFinal(t *testing.T) {
	sim := newTestSimulatorForHook(t)
	sim.InjectArrival(newTestRequest("req-1", 0, 100, 50))

	var snapshots []ProgressSnapshot
	sim.SetProgressHook(&collectingHook{snapshots: &snapshots}, 0)

	sim.Run()

	if len(snapshots) != 1 {
		t.Fatalf("expected exactly 1 snapshot (final only) with intervalUs=0, got %d", len(snapshots))
	}
	if !snapshots[0].IsFinal {
		t.Error("the only snapshot should have IsFinal=true")
	}
}

func TestSimulator_ProgressHook_FinalSnapshotClockClamped(t *testing.T) {
	horizon := int64(1_000_000)
	sim := mustNewSimulator(t, SimConfig{
		Horizon:             horizon,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 4, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 1000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 0.5, 0.5}, []float64{100, 0.1, 50}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "", "", 1, "roofline", 0),
	})
	sim.InjectArrival(newTestRequest("req-1", 0, 100, int(math.MaxInt16)))

	var snapshots []ProgressSnapshot
	sim.SetProgressHook(&collectingHook{snapshots: &snapshots}, 100_000)

	sim.Run()

	last := snapshots[len(snapshots)-1]
	if !last.IsFinal {
		t.Fatal("last snapshot should be final")
	}
	if last.Clock > horizon {
		t.Errorf("final snapshot Clock=%d exceeds Horizon=%d", last.Clock, horizon)
	}
}

func TestSimulator_ProgressHook_Determinism(t *testing.T) {
	runSim := func(withHook bool) *Metrics {
		s := newTestSimulatorForHook(t)
		s.InjectArrival(newTestRequest("req-1", 0, 100, 50))
		s.InjectArrival(newTestRequest("req-2", 100_000, 80, 30))
		if withHook {
			var snaps []ProgressSnapshot
			s.SetProgressHook(&collectingHook{snapshots: &snaps}, 100_000)
		}
		s.Run()
		return s.Metrics
	}
	without := runSim(false)
	with := runSim(true)

	if without.CompletedRequests != with.CompletedRequests {
		t.Errorf("CompletedRequests differs: %d vs %d", without.CompletedRequests, with.CompletedRequests)
	}
	if without.TotalInputTokens != with.TotalInputTokens {
		t.Errorf("TotalInputTokens differs: %d vs %d", without.TotalInputTokens, with.TotalInputTokens)
	}
	if without.TotalOutputTokens != with.TotalOutputTokens {
		t.Errorf("TotalOutputTokens differs: %d vs %d", without.TotalOutputTokens, with.TotalOutputTokens)
	}
	if without.SimEndedTime != with.SimEndedTime {
		t.Errorf("SimEndedTime differs: %d vs %d", without.SimEndedTime, with.SimEndedTime)
	}
}

func TestSimulator_ProgressHook_IntervalBoundaries(t *testing.T) {
	s := newTestSimulatorForHook(t)
	for i := 0; i < 10; i++ {
		s.InjectArrival(newTestRequest(
			fmt.Sprintf("req-%d", i),
			int64(i)*500_000,
			100, 50,
		))
	}
	var snapshots []ProgressSnapshot
	intervalUs := int64(2_000_000)
	s.SetProgressHook(&collectingHook{snapshots: &snapshots}, intervalUs)

	s.Run()

	nonFinal := 0
	for i, snap := range snapshots {
		if snap.IsFinal {
			continue
		}
		nonFinal++
		if i > 0 && !snapshots[i-1].IsFinal {
			gap := snap.Clock - snapshots[i-1].Clock
			if gap < intervalUs {
				t.Errorf("snapshot %d fired too soon: gap=%d < interval=%d", i, gap, intervalUs)
			}
		}
	}
	if nonFinal > 10 {
		t.Errorf("too many non-final snapshots (%d) for 2s interval — suggests per-event firing", nonFinal)
	}
}

func TestSimulator_ProgressHook_FreshSlicePerCall(t *testing.T) {
	s := mustNewSimulator(t, SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(100, 4, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(10, 1000, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{100, 0.5, 0.5}, []float64{100, 0.1, 50}),
		ModelHardwareConfig: NewModelHardwareConfig(rooflineModelConfig(), rooflineHWCalib(), "test-model", "", 1, "roofline", 0),
	})
	s.InjectArrival(newTestRequest("req-1", 0, 100, int(math.MaxInt16)))

	var snapshots []ProgressSnapshot
	s.SetProgressHook(&collectingHook{snapshots: &snapshots}, 100_000)

	s.Run()

	if len(snapshots) < 2 {
		t.Fatalf("expected at least 2 snapshots, got %d", len(snapshots))
	}
	snapshots[0].InstanceSnapshots[0].QueueDepth = 999999
	if snapshots[1].InstanceSnapshots[0].QueueDepth == 999999 {
		t.Error("InstanceSnapshots slices are shared between callbacks — BC-7 violated")
	}
}
