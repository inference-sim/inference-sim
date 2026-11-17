package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// signalCollector gives the pipeline one model to analyze; signalAnalyzer reports fixed signals for it.
type signalCollector struct{}

func (signalCollector) Collect(_ *sim.ClusterState) []ModelSignals {
	return []ModelSignals{{ModelID: "test-model"}}
}

type signalAnalyzer struct{}

func (signalAnalyzer) Name() string { return "signals" }
func (signalAnalyzer) Analyze(ms ModelSignals) AnalyzerResult {
	return AnalyzerResult{ModelID: ms.ModelID, Utilization: 0.9, RequiredCapacity: 5, SpareCapacity: 0}
}

func tracedCluster(t *testing.T, level string, instances int) *ClusterSimulator {
	t.Helper()
	d := newTestDeploymentConfig(instances)
	d.Model = "test-model"
	d.TraceLevel = level
	cs := newTestCluster(d, NewSliceRequestSource(nil), nil)
	for _, inst := range cs.instances {
		inst.Model, inst.gpu, inst.TPDegree = "test-model", "A100", 1
		inst.State = sim.InstanceStateActive
	}
	return cs
}

// Every applied scale decision is in the decision trace with the replicas before and after, its
// trigger and the analyzer's signals (#743); a decision with no change writes none, and without the
// decisions trace level nothing is recorded.
func TestScaleRecords_ModelDecisionsAreTraced(t *testing.T) {
	cs := tracedCluster(t, "decisions", 3)
	a := NewDirectActuator(cs)
	down := ScaleDecision{ModelID: "test-model", Variant: NewVariantSpec("A100", 1), Delta: -2,
		Utilization: 0.1, SpareCapacity: 7}
	if err := a.Apply([]ScaleDecision{down, {ModelID: "test-model", Variant: NewVariantSpec("A100", 1)}}); err != nil {
		t.Fatal(err)
	}
	got := cs.Trace().ModelScales
	if len(got) != 1 {
		t.Fatalf("%d model scale record(s), want one for the decision with a change: %+v", len(got), got)
	}
	r := got[0]
	want := trace.ModelScaleRecord{ModelID: "test-model", GPUType: "A100", TPDegree: 1, Delta: -2, OldReplicas: 3,
		NewReplicas: 1, Trigger: "spare_capacity", Utilization: 0.1, SpareCapacity: 7}
	if r != want {
		t.Errorf("record = %+v, want %+v", r, want)
	}

	off := tracedCluster(t, "", 3)
	if err := NewDirectActuator(off).Apply([]ScaleDecision{down}); err != nil {
		t.Fatal(err)
	}
	if off.Trace() != nil {
		t.Error("a trace exists without the decisions trace level")
	}
}

// The signals the analyzer reported at the tick reach the record, through the pipeline.
func TestScaleRecords_CarryTheAnalyzersSignals(t *testing.T) {
	d := newTestDeploymentConfig(1)
	d.Model = "test-model"
	d.TraceLevel = "decisions"
	d.ModelAutoscalerIntervalUs = 1_000_000
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100", GPUsPerNode: 4, InitialNodes: 1, MaxNodes: 1, GPUMemoryGiB: 80}}
	cs := newTestCluster(d, NewSliceRequestSource(newTestRequests(3)), nil)
	cs.autoscaler = newTestPipeline(signalCollector{}, signalAnalyzer{}, &scaleUpOnce{}, NewDirectActuator(cs))
	mustRun(t, cs)
	got := cs.Trace().ModelScales
	if len(got) != 1 {
		t.Fatalf("%d model scale record(s), want the one scale-up: %+v", len(got), got)
	}
	r := got[0]
	if r.Trigger != "required_capacity" || r.Delta != 1 || r.Utilization != 0.9 || r.RequiredCapacity != 5 {
		t.Errorf("record = %+v, want the scale-up with utilization 0.9 and required capacity 5", r)
	}
	if r.NewReplicas != r.OldReplicas+1 {
		t.Errorf("replicas %d -> %d, want one more placed", r.OldReplicas, r.NewReplicas)
	}
	if sum := trace.Summarize(cs.Trace()); len(sum.ModelScales) != 1 {
		t.Errorf("the trace summary (the Result's trace_summary) holds %d model scale records, want 1", len(sum.ModelScales))
	}
}

// A node the cluster autoscaler adds is recorded with the pool's node counts, the pending
// placements that caused it and the pool's utilization.
func TestScaleRecords_ClusterDecisionsAreTraced(t *testing.T) {
	d := newTestDeploymentConfig(2)
	d.Model = "test-model"
	d.TraceLevel = "decisions"
	d.ModelAutoscalerIntervalUs = 1_000_000
	d.NodePools = []NodePoolConfig{{Name: "p", GPUType: "A100", GPUsPerNode: 2, InitialNodes: 1, MaxNodes: 2,
		GPUMemoryGiB: 80, ProvisioningDelay: DelaySpec{Mean: 1}}}
	d.ClusterAutoscaler = ClusterAutoscalerConfig{Policy: ClusterAutoscalerPendingDriven,
		PendingDrivenConfig: PendingDrivenConfig{PendingThresholdUs: 500_000}}
	cs := newTestCluster(d, NewSliceRequestSource(newTestRequests(4)), nil)
	cs.autoscaler = newTestPipeline(&countingCollector{}, &nopAnalyzer{}, &scaleUpOnce{}, NewDirectActuator(cs))
	mustRun(t, cs)
	got := cs.Trace().ClusterScales
	if len(got) != 1 {
		t.Fatalf("%d cluster scale record(s), want the one node added: %+v", len(got), got)
	}
	want := trace.ClusterScaleRecord{Clock: got[0].Clock, PoolName: "p", OldNodeCount: 1, NewNodeCount: 2,
		TriggerReason: "pending_threshold", PendingCount: 1, GPUUtilization: 1}
	if got[0] != want {
		t.Errorf("record = %+v, want %+v", got[0], want)
	}
	if got[0].Clock < 500_000 {
		t.Errorf("node added at %d µs, before the placement had waited the threshold", got[0].Clock)
	}
}
