package cluster

import (
	"math"
	"strings"
	"testing"
)

func thresholdSignals(signal string, values ...float64) ModelSignals {
	ms := ModelSignals{ModelID: "m"}
	for _, v := range values {
		r := ReplicaMetrics{Variant: NewVariantSpec("A100", 1), CostPerHour: 3}
		switch signal {
		case SignalQueueDepth:
			r.QueueDepth = int(v)
		case SignalKVUtilization:
			r.KVUtilization = v
		default:
			r.InFlightCount = int(v)
		}
		ms.Replicas = append(ms.Replicas, r)
	}
	return ms
}

// The threshold analyzer, with the unchanged engine, makes the HPA desired-replicas step:
// ceil(sum(signal) / target) replicas in all, never fewer than the floor, and scale-down is the
// symmetric floor bounded by what the demand leaves unneeded.
func TestThresholdAnalyzer_DesiredReplicasAreCeilOfSumOverTarget(t *testing.T) {
	desired := func(a *ThresholdAnalyzer, ms ModelSignals) int {
		n := len(ms.Replicas)
		res := a.Analyze(ms)
		if res.RequiredCapacity > 0 {
			return n + scaleUpN(res.RequiredCapacity, res.VariantCapacities[0:1])
		}
		if res.SafeRemovals > 0 {
			return n - scaleDownN(res.SpareCapacity, res.VariantCapacities[0], res.SafeRemovals)
		}
		return n
	}
	a := NewThresholdAnalyzer(AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalInFlight, Target: 16}, 1)
	for _, tc := range []struct {
		name   string
		values []float64
		want   int
	}{
		{"saturated: ceil(100/16) = 7", []float64{25, 25, 25, 25}, 7},
		{"exactly at target", []float64{16, 16, 16, 16}, 4},
		{"just over", []float64{16, 16, 16, 17}, 5},
		{"idle: down to the floor of 1", []float64{0, 0, 0, 0, 0}, 1},
		{"light: ceil(20/16) = 2", []float64{4, 4, 4, 4, 4}, 2},
	} {
		if got := desired(a, thresholdSignals(SignalInFlight, tc.values...)); got != tc.want {
			t.Errorf("%s: desired %d replicas, want %d", tc.name, got, tc.want)
		}
	}

	floor2 := NewThresholdAnalyzer(AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalInFlight, Target: 16}, 2)
	if got := desired(floor2, thresholdSignals(SignalInFlight, 0, 0, 0, 0)); got != 2 {
		t.Errorf("idle with a floor of 2: desired %d, want 2", got)
	}

	// Replicas still loading count as supply for the scale-up signal, so a slow load is not asked for again.
	ms := thresholdSignals(SignalInFlight, 25, 25, 25, 25)
	ms.PendingReplicaCount = 3
	if res := a.Analyze(ms); res.RequiredCapacity != 0 {
		t.Errorf("3 replicas loading cover the shortfall of 36 (3 x 16 = 48), yet required capacity = %v", res.RequiredCapacity)
	}

	// No replica to measure is no signal.
	if res := a.Analyze(ModelSignals{ModelID: "m"}); res.RequiredCapacity != 0 || res.SafeRemovals != 0 {
		t.Errorf("a model with no replicas produced a signal: %+v", res)
	}
}

func TestThresholdAnalyzer_ReadsTheNamedSignal(t *testing.T) {
	q := NewThresholdAnalyzer(AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalQueueDepth, Target: 4}, 1)
	if res := q.Analyze(thresholdSignals(SignalQueueDepth, 10, 10)); res.TotalDemand != 20 || res.TotalSupply != 8 {
		t.Errorf("queue_depth: demand %v supply %v, want 20 and 8", res.TotalDemand, res.TotalSupply)
	}
	kv := NewThresholdAnalyzer(AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalKVUtilization, Target: 0.5}, 1)
	if res := kv.Analyze(thresholdSignals(SignalKVUtilization, 0.9, 0.9)); math.Abs(res.TotalDemand-1.8) > 1e-9 || res.RequiredCapacity <= 0 {
		t.Errorf("kv_utilization: demand %v required %v, want 1.8 and a scale-up", res.TotalDemand, res.RequiredCapacity)
	}
}

func TestAnalyzerSelection_Validate(t *testing.T) {
	for _, tc := range []struct {
		sel  AnalyzerSelection
		want string
	}{
		{AnalyzerSelection{}, ""},
		{AnalyzerSelection{Kind: AnalyzerKindV2Saturation}, ""},
		{AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalInFlight, Target: 16}, ""},
		{AnalyzerSelection{Kind: "queueing"}, "not an analyzer"},
		{AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: "cpu", Target: 1}, "not a signal"},
		{AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalInFlight}, "target"},
		{AnalyzerSelection{Signal: SignalInFlight, Target: 4}, "belong to kind"},
	} {
		err := tc.sel.Validate()
		if (tc.want == "") != (err == nil) || (err != nil && !strings.Contains(err.Error(), tc.want)) {
			t.Errorf("%+v: error %v, want %q", tc.sel, err, tc.want)
		}
	}
}

// The selector reaches the pipeline: threshold when asked for, the saturation analyzer otherwise
// (every existing run is unchanged), and a selection that cannot run is a startup error.
func TestAnalyzerSelection_ChoosesThePipelinesAnalyzer(t *testing.T) {
	build := func(sel AnalyzerSelection) *ClusterSimulator {
		d := newTestDeploymentConfig(1)
		d.ModelAutoscalerIntervalUs = 1_000_000
		d.AutoscalerAnalyzer = sel
		return newTestCluster(d, NewSliceRequestSource(nil), nil)
	}
	if got := build(AnalyzerSelection{}).autoscaler.analyzer.Name(); got != AnalyzerKindV2Saturation {
		t.Errorf("default analyzer = %q, want %q", got, AnalyzerKindV2Saturation)
	}
	cs := build(AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: SignalInFlight, Target: 16})
	if got := cs.autoscaler.analyzer.Name(); got != AnalyzerKindThreshold {
		t.Errorf("selected analyzer = %q, want %q", got, AnalyzerKindThreshold)
	}
	if bad := build(AnalyzerSelection{Kind: AnalyzerKindThreshold, Signal: "x", Target: 1}); bad.startupErr == nil {
		t.Error("an invalid analyzer selection was accepted")
	}
}
