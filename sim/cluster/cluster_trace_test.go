package cluster

import (
	"testing"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"

	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestClusterSimulator_TraceLevelNone_NilTrace(t *testing.T) {
	// GIVEN trace level none (default)
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             1000000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 2
		d.TraceLevel = "none"
		return d
	}()
	requests := testGenerateRequests(42, 1000000, 1.0/1e6, 3,
		0, 10, 0, 10, 10, 5, 0, 5, 5)
	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)

	// WHEN run
	mustRun(t, cs)

	// THEN trace is nil (zero overhead)
	if cs.Trace() != nil {
		t.Error("expected nil trace for trace-level none")
	}
}

func TestClusterSimulator_TraceLevelDecisions_RecordsAllEvents(t *testing.T) {
	// GIVEN trace level decisions with 5 requests and 2 instances
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 2
		d.TraceLevel = "decisions"
		d.CounterfactualK = 0
		return d
	}()
	requests := testGenerateRequests(42, 10000000, 1.0/1e6, 5,
		0, 10, 0, 10, 10, 5, 0, 5, 5)
	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)

	// WHEN run
	mustRun(t, cs)

	// THEN trace is non-nil with admission and routing records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace for trace-level decisions")
	}
	if len(tr.Admissions) != 5 {
		t.Errorf("expected 5 admission records (one per request), got %d", len(tr.Admissions))
	}
	if len(tr.Routings) != 5 {
		t.Errorf("expected 5 routing records (all admitted with always-admit), got %d", len(tr.Routings))
	}

	// All admission records should be admitted (default always-admit)
	for i, a := range tr.Admissions {
		if !a.Admitted {
			t.Errorf("admission[%d]: expected admitted=true", i)
		}
	}
}

func TestClusterSimulator_TraceLevelDecisions_WithCounterfactual(t *testing.T) {
	// GIVEN trace with counterfactual k=2 and a gateway that scores both instances
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             10000000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 2
		d.TraceLevel = "decisions"
		d.CounterfactualK = 2
		return d
	}()
	requests := testGenerateRequests(42, 10000000, 1.0/1e6, 3,
		0, 10, 0, 10, 10, 5, 0, 5, 5)
	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)
	cs.SetGateway(&fakeGateway{scores: map[string]float64{"instance_0": 1, "instance_1": 0.5}})

	// WHEN run
	mustRun(t, cs)

	// THEN routing records have candidates
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	if len(tr.Routings) == 0 {
		t.Fatal("expected routing records")
	}
	for i, r := range tr.Routings {
		if len(r.Candidates) != 2 {
			t.Errorf("routing[%d]: expected 2 candidates with k=2, got %d", i, len(r.Candidates))
		}
		wantRegret := 0.0
		if r.ChosenInstance == "instance_1" {
			wantRegret = 0.5
		}
		if r.Regret != wantRegret {
			t.Errorf("routing[%d]: chose %s, regret = %v, want %v", i, r.ChosenInstance, r.Regret, wantRegret)
		}
	}
}

func TestClusterSimulator_TraceWithAdmissionRejections_RecordsRejections(t *testing.T) {
	// GIVEN a gateway that rejects every other request at admission
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             5000000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(10, 2048, 0),
			LatencyModel:        testFakeLatency(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "", "", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 1
		d.TraceLevel = "decisions"
		return d
	}()
	requests := testGenerateRequests(42, 5000000, 5.0/1e6, 10,
		0, 10, 0, 10, 10, 5, 0, 5, 5)
	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)
	cs.SetGateway(rejectEveryOtherGateway())

	// WHEN run
	mustRun(t, cs)

	// THEN some admissions are rejected
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	rejections := 0
	for _, a := range tr.Admissions {
		if !a.Admitted {
			rejections++
		}
	}
	if rejections == 0 {
		t.Error("expected some rejections, got 0")
	}
	if rejections != cs.RejectedRequests() {
		t.Errorf("trace rejections %d != RejectedRequests %d", rejections, cs.RejectedRequests())
	}
	// Routing records should be fewer than admissions (rejected requests don't route)
	if len(tr.Routings) >= len(tr.Admissions) {
		t.Errorf("expected fewer routings (%d) than admissions (%d) with rejections",
			len(tr.Routings), len(tr.Admissions))
	}

	// Summarize to verify rejection counts match
	summary := trace.Summarize(tr)
	if summary.RejectedCount != rejections {
		t.Errorf("summary rejected count %d != counted rejections %d", summary.RejectedCount, rejections)
	}
}
