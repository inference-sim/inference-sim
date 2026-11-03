package cluster

import (
	"bytes"
	"math"
	"math/rand"
	"os"
	"strings"
	"testing"

	"github.com/sirupsen/logrus"

	cengine "github.com/inference-sim/blis-vllm/compat/engine"
	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
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

// TestClusterSimulator_MixedTimeoutAndDrop_NoOverclaim covers issue #2064 review F3:
// with zero completions, the diagnostic must NOT assert "every request timed out" when
// a dropped-unservable request also contributed. One request times out (5ms deadline
// behind a slow r0 that itself is dropped), one is dropped by the MaxModelLen guard.
func TestClusterSimulator_MixedTimeoutAndDrop_NoOverclaim(t *testing.T) {
	const maxModelLen = 200
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             500_000_000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(1, 2048, 0), // max 1 running — forces queuing
			LatencyModel:        testFakeZeroQueueing(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "test", "H100", 1, 1, false, maxModelLen),
		}, 0)
		d.NumInstances = 1
		return d
	}()

	requests := []*sim.Request{
		// Fits and runs, but its 5ms deadline is shorter than its service time → times out.
		{
			ID: "slow", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 50),
			State: sim.StateQueued, MaxOutputLen: 50,
			Deadline: 5_000,
		},
		// Dropped unservable: input == MaxModelLen (Guard 1a).
		{
			ID: "toobig", ArrivalTime: 0,
			InputTokens: make([]sim.TokenID, maxModelLen), OutputTokens: make([]sim.TokenID, 10),
			State: sim.StateQueued, MaxOutputLen: 10,
		},
	}

	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 0 || m.TimedOutRequests == 0 || m.DroppedUnservable == 0 {
		t.Fatalf("test setup: want 0 completed, >=1 timed out, >=1 dropped; got completed=%d timed_out=%d dropped=%d",
			m.CompletedRequests, m.TimedOutRequests, m.DroppedUnservable)
	}

	// The timeout count is still reported...
	if !strings.Contains(output, "timed out") {
		t.Errorf("expected a timeout warning, got: %q", output)
	}
	// ...but the no-completion line must NOT overclaim that EVERY request timed out,
	// because a drop also contributed.
	if strings.Contains(output, "every request timed out") {
		t.Errorf("must not claim 'every request timed out' when a drop also contributed, got: %q", output)
	}
}

// TestClusterSimulator_SessionUnserved_CancelsThroughHook covers PR #2069 re-review #1
// end-to-end: a real SessionManager wired exactly as cmd wires it (OnComplete as
// onRequestDone, OnUnserved as the unserved hook), with a gateway that ends the session's
// round-0 request WITHOUT serving it — a client-deadline expiry in the gateway, the same
// recordUnserved path an eviction, admission or routing rejection takes. The session must
// be cancelled (not continued): no follow-up round is issued and CancelledSessions()==1.
// (The eviction-tombstone case — where OnComplete alone would misread StateCompleted as a
// successful round — is covered precisely at the unit level in
// TestSession_OnUnserved_EvictedTombstoneCancels and the pool variant.)
func TestClusterSimulator_SessionUnserved_CancelsThroughHook(t *testing.T) {
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 10}})
	if err != nil {
		t.Fatalf("input sampler: %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 5}})
	if err != nil {
		t.Fatalf("output sampler: %v", err)
	}
	const sessID = "unserved-sess"
	bp := workload.SessionBlueprint{
		SessionID: sessID, ClientID: "c", MaxRounds: 5, ThinkTimeUs: 1000,
		Horizon: math.MaxInt64, InputSampler: inputSampler, OutputSampler: outputSampler,
		RNG: rand.New(rand.NewSource(7)), Model: "test-model",
	}
	seed := &sim.Request{
		ID: sessID + "_r0", ArrivalTime: 0, State: sim.StateQueued,
		InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), MaxOutputLen: 5,
		SessionID: sessID, RoundIndex: 0,
	}
	sm := workload.NewSessionManager([]workload.SessionBlueprint{bp})

	rounds := 0
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		if req.SessionID == sessID {
			rounds++
		}
		return sm.OnComplete(req, tick)
	}

	config := newTestDeploymentConfig(1)
	config.Horizon = 1_000_000
	cs := newTestCluster(config, NewSliceRequestSource([]*sim.Request{seed}), onDone)
	// Gateway ends the request unserved (client deadline expired while held).
	cs.SetGateway(&fakeGateway{reject: func(*sim.Request) bool { return true }, rejectAs: RejectedClientTimeout})
	cs.SetUnservedHook(sm.OnUnserved)

	if err := cs.Run(); err != nil {
		t.Fatalf("cs.Run: %v", err)
	}

	if sm.CancelledSessions() != 1 {
		t.Errorf("CancelledSessions = %d, want 1 (an unserved round-0 must cancel the session)", sm.CancelledSessions())
	}
	if rounds != 0 {
		t.Errorf("onRequestDone saw %d session rounds; an unserved round-0 reaches the unserved hook, not onRequestDone", rounds)
	}
	if cs.AggregatedMetrics().CompletedRequests != 0 {
		t.Errorf("CompletedRequests = %d, want 0 (the only request was unserved)", cs.AggregatedMetrics().CompletedRequests)
	}
}

// TestClusterSimulator_MixedTimeoutAndRoutingRejection_NoOverclaim covers PR #2069
// re-review #2: the zero-completion diagnostic must not claim "every request timed out"
// when a router/gateway rejection — counted in routingRejections, OUTSIDE
// aggregatedMetrics — also contributed. One request times out; one is routing-rejected
// (its model matches no instance). Before the fix this logged "every request timed out"
// because DroppedUnservable/StillQueued/StillRunning and rejectedRequests were all 0.
func TestClusterSimulator_MixedTimeoutAndRoutingRejection_NoOverclaim(t *testing.T) {
	config := func() DeploymentConfig {
		d := compatDeployment(cengine.SimConfig{
			Horizon:             500_000_000,
			Seed:                42,
			KVCacheConfig:       cengine.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         cengine.NewBatchConfig(1, 2048, 0),
			LatencyModel:        testFakeZeroQueueing(),
			ModelHardwareConfig: cengine.NewModelHardwareConfig(testModelConfig(), "test-model", "H100", 1, 1, false, 0),
		}, 0)
		d.NumInstances = 1
		return d
	}()
	requests := []*sim.Request{
		// Times out: 5ms deadline, shorter than its service time.
		{ID: "slow", ArrivalTime: 0, State: sim.StateQueued,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 50), MaxOutputLen: 50, Deadline: 5_000},
		// Routing-rejected: its model matches no instance.
		{ID: "wrongmodel", ArrivalTime: 0, State: sim.StateQueued,
			InputTokens: make([]sim.TokenID, 10), OutputTokens: make([]sim.TokenID, 5), MaxOutputLen: 5, Model: "no-such-model"},
	}

	cs := newTestCluster(config, NewSliceRequestSource(requests), nil)
	cs.SetGateway(&fakeGateway{})
	output := captureLogOutput(func() {
		if err := cs.Run(); err != nil {
			t.Fatalf("cs.Run: %v", err)
		}
	})

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 0 || m.TimedOutRequests == 0 || cs.RoutingRejections() == 0 {
		t.Fatalf("test setup: want 0 completed, >=1 timed out, >=1 routing-rejected; got completed=%d timed_out=%d routing_rej=%d",
			m.CompletedRequests, m.TimedOutRequests, cs.RoutingRejections())
	}
	if strings.Contains(output, "every request timed out") {
		t.Errorf("must not claim 'every request timed out' when a routing rejection also contributed, got: %q", output)
	}
	if strings.Contains(output, "all") && strings.Contains(output, "rejected by the router's admission") {
		t.Errorf("must not claim 'all requests rejected' in a mixed timeout + routing-rejection run, got: %q", output)
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
