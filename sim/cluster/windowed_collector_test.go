package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

type stubCollector struct{}

func (stubCollector) Collect(s *sim.ClusterState) []ModelSignals {
	return []ModelSignals{{ModelID: "m", Replicas: []ReplicaMetrics{{InstanceID: "i0", Variant: NewVariantSpec("A100", 1),
		QueueDepth: int(s.Clock / 1_000_000)}}}}
}

// The windowed collector keeps the last N ticks, oldest first, each with its time and a sequence
// number; with the totals of completed requests it fills the latency fields from each tick's own
// completions, not the run's average so far (#1141).
func TestWindowedCollector_KeepsTheLastTicksAndTheTicksOwnLatency(t *testing.T) {
	totals := map[string]LatencyTotals{}
	w := NewWindowedCollector(stubCollector{}, 3, func(id string) (LatencyTotals, bool) { v, ok := totals[id]; return v, ok })
	state := func(clock int64) *sim.ClusterState { return &sim.ClusterState{Clock: clock} }

	// Tick 1 (t=1s): 10 requests completed so far, mean TTFT 100 µs, mean ITL 10 µs.
	totals["i0"] = LatencyTotals{Completed: 10, TTFTSumUs: 1000, ITLSumUs: 100, InTokens: 1000, OutTokens: 500}
	got := w.Collect(state(1_000_000))
	if r := got[0].Replicas[0]; r.TTFT != 100 || r.ITL != 10 || r.DispatchRate != 10 || r.AvgInTokens != 100 {
		t.Errorf("tick 1: %+v, want TTFT 100, ITL 10, 10 req/s, 100 input tokens", r)
	}
	// Tick 2 (t=2s): 4 more requests, each TTFT 500 and ITL 50: the tick's own mean, not the cumulative.
	totals["i0"] = LatencyTotals{Completed: 14, TTFTSumUs: 3000, ITLSumUs: 300, InTokens: 1400, OutTokens: 700}
	got = w.Collect(state(2_000_000))
	if r := got[0].Replicas[0]; r.TTFT != 500 || r.ITL != 50 || r.DispatchRate != 4 {
		t.Errorf("tick 2: %+v, want TTFT 500, ITL 50, 4 req/s (this tick's completions)", r)
	}
	// Tick 3 (t=3s): nothing completed: the latency fields are zero, as with no samples.
	got = w.Collect(state(3_000_000))
	if r := got[0].Replicas[0]; r.TTFT != 0 || r.DispatchRate != 0 {
		t.Errorf("tick 3: %+v, want zero latency fields for a tick with no completion", r)
	}
	got = w.Collect(state(4_000_000))
	win := got[0].Window
	if len(win) != 3 {
		t.Fatalf("window holds %d ticks, want the last 3", len(win))
	}
	for i := 1; i < len(win); i++ {
		if win[i].TimeUs <= win[i-1].TimeUs || win[i].Seq != win[i-1].Seq+1 {
			t.Errorf("window not oldest-first with consecutive sequence numbers: %+v", win)
		}
	}
	if win[0].TimeUs != 2_000_000 || win[2].TimeUs != 4_000_000 || win[0].Replicas[0].QueueDepth != 2 {
		t.Errorf("window = %+v, want ticks at 2s, 3s and 4s with each tick's own readings", win)
	}
}

// With no window the collector is the plain one and the signals carry none; a window of the
// cluster's own instances is read from their metrics.
func TestWindowedCollector_ClusterWiring(t *testing.T) {
	d := newTestDeploymentConfig(1)
	d.ModelAutoscalerIntervalUs = 1_000_000
	if _, windowed := newTestCluster(d, NewSliceRequestSource(nil), nil).autoscaler.collector.(*WindowedCollector); windowed {
		t.Error("a deployment with no window got a windowed collector")
	}
	d.CollectorWindowSize = 4
	cs := newTestCluster(d, NewSliceRequestSource(nil), nil)
	if _, windowed := cs.autoscaler.collector.(*WindowedCollector); !windowed {
		t.Error("collector_window_size 4 did not give a windowed collector")
	}
	if _, ok := cs.instanceTotals(string(cs.instances[0].ID())); !ok {
		t.Error("the cluster cannot read its own instance's totals")
	}
	d.CollectorWindowSize = -1
	if newTestCluster(d, NewSliceRequestSource(nil), nil).startupErr == nil {
		t.Error("a negative window was accepted")
	}
}
