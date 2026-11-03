package cluster

import (
	"bytes"
	"os"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/inference-sim/sim"
)

// captureLogOutput runs fn and returns the log output as a string.
func captureLogOutput(fn func()) string {
	var buf bytes.Buffer
	origOutput := logrus.StandardLogger().Out
	origLevel := logrus.GetLevel()
	logrus.SetOutput(&buf)
	logrus.SetLevel(logrus.WarnLevel)
	defer func() {
		if origOutput != nil {
			logrus.SetOutput(origOutput)
		} else {
			logrus.SetOutput(os.Stderr)
		}
		logrus.SetLevel(origLevel)
	}()
	fn()
	return buf.String()
}

func TestClusterSimulator_AllRejected_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster whose gateway rejects every request at admission
	config := newTestDeploymentConfig(1)
	config.Horizon = 100000
	workload := newTestRequests(5)

	cs := newTestCluster(config, NewSliceRequestSource(workload), nil)
	cs.SetGateway(&fakeGateway{reject: func(*sim.Request) bool { return true }})

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	// THEN a warning about all requests being rejected MUST be logged
	if !strings.Contains(output, "rejected") {
		t.Errorf("expected all-rejected warning, got: %q", output)
	}
}

func TestClusterSimulator_ZeroCompletions_WarnsAfterRun(t *testing.T) {
	// GIVEN a cluster with horizon too short for any request to finish
	config := newTestDeploymentConfig(1)
	config.Horizon = 1 // 1 tick — admits but can't finish
	workload := newTestRequests(5)

	cs := newTestCluster(config, NewSliceRequestSource(workload), nil)

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	// THEN a warning about no completed requests MUST be logged
	if !strings.Contains(output, "no requests completed") {
		t.Errorf("expected zero-completion warning, got: %q", output)
	}
}

// TestClusterSimulator_SomeTimeouts_WarnsEvenWithCompletions verifies issue #2064's
// third expectation: a timeout is reported whenever ANY request times out, not only
// when every request fails. Before the fix the timeout warning lived inside the
// CompletedRequests==0 block, so a run where most requests complete but some time out
// logged nothing about the timeouts.
func TestClusterSimulator_SomeTimeouts_WarnsEvenWithCompletions(t *testing.T) {
	// GIVEN a 1-instance, batch-1 cluster: r0 completes (~6ms), r2 has a 5ms deadline
	// and times out while queued behind r0 (mirrors the metrics test above).
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             500_000_000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(1, 2048, 0), // max 1 running — forces queuing
			LatencyModel:        testFakeZeroQueueing(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 1
		return d
	}()

	requests := []*sim.Request{
		{
			ID: "r0", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5),
			State: sim.StateQueued, MaxOutputLen: 5,
			Deadline: 300_000_000, // 300s — completes well before
		},
		{
			ID: "r2", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5),
			State: sim.StateQueued, MaxOutputLen: 5,
			Deadline: 5_000, // 5ms — times out while queued behind r0
		},
	}

	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)

	// WHEN the simulation runs with at least one completion AND at least one timeout
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	m := cs.AggregatedMetrics()
	if m.CompletedRequests == 0 || m.TimedOutRequests == 0 {
		t.Fatalf("test setup: want some completions and some timeouts, got completed=%d timed_out=%d",
			m.CompletedRequests, m.TimedOutRequests)
	}

	// THEN a timeout warning MUST be logged (it was silent before #2064)
	if !strings.Contains(output, "timed out") {
		t.Errorf("expected a timeout warning even with completions, got: %q", output)
	}
}

func TestClusterSimulator_NormalOperation_NoPostSimWarning(t *testing.T) {
	// GIVEN a properly configured cluster that will complete requests
	config := newTestDeploymentConfig(1)
	config.Horizon = 1000000
	workload := newTestRequests(2)

	cs := newTestCluster(config, NewSliceRequestSource(workload), nil)

	// WHEN the simulation runs to completion
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	// THEN no rejection or zero-completion warnings MUST be logged
	if strings.Contains(output, "rejected") || strings.Contains(output, "no requests completed") {
		t.Errorf("unexpected warning during normal operation, got: %q", output)
	}
}
