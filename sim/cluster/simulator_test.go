package cluster

import "testing"

// TestClusterSimulator_Creation tests cluster creation
func TestClusterSimulator_Creation(t *testing.T) {
	sim := NewClusterSimulator(10000)

	if sim.Clock != 0 {
		t.Errorf("Initial clock = %d, want 0", sim.Clock)
	}
	if sim.Horizon != 10000 {
		t.Errorf("Horizon = %d, want 10000", sim.Horizon)
	}
	if len(sim.Instances) != 0 {
		t.Errorf("Initial instance count = %d, want 0", len(sim.Instances))
	}
	if sim.EventQueue.Len() != 0 {
		t.Errorf("Initial event queue len = %d, want 0", sim.EventQueue.Len())
	}
}

// TestClusterSimulator_AddDeployment tests adding deployments
func TestClusterSimulator_AddDeployment(t *testing.T) {
	sim := NewClusterSimulator(10000)

	// Create a deployment with instances
	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 3)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	inst2 := NewInstanceSimulator("inst2", PoolMonolithic, nil, 1000, 16)
	pool.AddInstance(inst1)
	pool.AddInstance(inst2)

	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}

	err := sim.AddDeployment(config)
	if err != nil {
		t.Errorf("AddDeployment() error = %v", err)
	}

	if len(sim.Deployments) != 1 {
		t.Errorf("Deployment count = %d, want 1", len(sim.Deployments))
	}
	if len(sim.Instances) != 2 {
		t.Errorf("Instance count = %d, want 2", len(sim.Instances))
	}

	// Try to add duplicate
	err = sim.AddDeployment(config)
	if err == nil {
		t.Error("AddDeployment() should error on duplicate ConfigID")
	}
}

// TestClusterSimulator_GetInstance tests instance retrieval
func TestClusterSimulator_GetInstance(t *testing.T) {
	sim := NewClusterSimulator(10000)

	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 3)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool.AddInstance(inst1)

	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}
	sim.AddDeployment(config)

	retrieved := sim.GetInstance("inst1")
	if retrieved == nil {
		t.Error("GetInstance(inst1) returned nil")
	}
	if retrieved.ID != "inst1" {
		t.Errorf("Retrieved instance ID = %s, want inst1", retrieved.ID)
	}

	nonexistent := sim.GetInstance("inst999")
	if nonexistent != nil {
		t.Error("GetInstance(inst999) should return nil")
	}
}

// TestClusterSimulator_ListInstances tests listing instances
func TestClusterSimulator_ListInstances(t *testing.T) {
	sim := NewClusterSimulator(10000)

	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 0, 3)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	inst2 := NewInstanceSimulator("inst2", PoolMonolithic, nil, 1000, 16)
	pool.AddInstance(inst1)
	pool.AddInstance(inst2)

	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}
	sim.AddDeployment(config)

	ids := sim.ListInstances()
	if len(ids) != 2 {
		t.Errorf("ListInstances() returned %d instances, want 2", len(ids))
	}

	// Check both IDs are present
	found1, found2 := false, false
	for _, id := range ids {
		if id == "inst1" {
			found1 = true
		}
		if id == "inst2" {
			found2 = true
		}
	}
	if !found1 || !found2 {
		t.Error("ListInstances() missing expected instance IDs")
	}
}

// TestClusterSimulator_BC5_ClockMonotonicity tests BC-5: clock monotonicity
func TestClusterSimulator_BC5_ClockMonotonicity(t *testing.T) {
	sim := NewClusterSimulator(10000)

	// Add instance
	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool.AddInstance(inst1)
	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}
	sim.AddDeployment(config)

	// Schedule events at increasing timestamps
	req1 := &Request{ID: "req1", PromptTokens: 100, OutputTokens: 50}
	req2 := &Request{ID: "req2", PromptTokens: 100, OutputTokens: 50}

	sim.ScheduleEvent(NewRequestArrivalEvent(100, req1))
	sim.ScheduleEvent(NewRequestArrivalEvent(200, req2))

	// Run and verify clock increases
	lastClock := sim.Clock
	eventCount := 0

	for sim.EventQueue.Len() > 0 && eventCount < 10 {
		event := sim.EventQueue.PopNext()
		if event.Timestamp() > sim.Horizon {
			break
		}

		if event.Timestamp() < lastClock {
			t.Errorf("Clock monotonicity violated: %d < %d", event.Timestamp(), lastClock)
		}

		sim.Clock = event.Timestamp()
		event.Execute(sim)
		lastClock = sim.Clock
		eventCount++
	}
}

// TestClusterSimulator_BC7_RequestLifecycle tests BC-7: request lifecycle
func TestClusterSimulator_BC7_RequestLifecycle(t *testing.T) {
	sim := NewClusterSimulator(10000)

	// Add instance
	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool.AddInstance(inst1)
	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}
	sim.AddDeployment(config)

	// Schedule request arrival
	req := &Request{ID: "req1", PromptTokens: 100, OutputTokens: 50}
	sim.ScheduleEvent(NewRequestArrivalEvent(100, req))

	// Process events
	for i := 0; i < 10 && sim.EventQueue.Len() > 0; i++ {
		event := sim.EventQueue.PopNext()
		if event.Timestamp() > sim.Horizon {
			break
		}
		sim.Clock = event.Timestamp()
		event.Execute(sim)
	}

	// Verify request went through lifecycle states
	if req.State != RequestStateQueued && req.State != RequestStateRunning && req.State != RequestStateCompleted {
		// At minimum, should be queued
		if req.State != RequestStateQueued {
			t.Errorf("Request did not progress through lifecycle, state = %s", req.State)
		}
	}

	// Verify state transitions are monotonic
	stateOrder := map[RequestState]int{
		RequestStatePending:   1,
		RequestStateRouted:    2,
		RequestStateQueued:    3,
		RequestStateRunning:   4,
		RequestStateCompleted: 5,
	}

	if req.ArrivalTime > 0 {
		if req.RouteTime > 0 && req.RouteTime < req.ArrivalTime {
			t.Error("RouteTime < ArrivalTime (backward transition)")
		}
		if req.EnqueueTime > 0 && req.EnqueueTime < req.RouteTime {
			t.Error("EnqueueTime < RouteTime (backward transition)")
		}
	}

	_ = stateOrder // Use variable
}

// TestClusterSimulator_BC8_Causality tests BC-8: causality invariant
func TestClusterSimulator_BC8_Causality(t *testing.T) {
	// Test causality check in handleRequestCompleted
	sim := NewClusterSimulator(10000)

	req := &Request{
		ID:             "req1",
		ArrivalTime:    100,
		RouteTime:      150,
		EnqueueTime:    200,
		CompletionTime: 500,
	}

	// Should not panic - valid causality
	event := NewRequestCompletedEvent(500, req, "inst1")
	sim.handleRequestCompleted(event)

	// Test invalid causality (arrival > route)
	req2 := &Request{
		ID:             "req2",
		ArrivalTime:    200,
		RouteTime:      100,  // Invalid: before arrival
		EnqueueTime:    250,
		CompletionTime: 500,
	}

	defer func() {
		if r := recover(); r == nil {
			t.Error("Expected panic for causality violation (arrival > route)")
		}
	}()

	event2 := NewRequestCompletedEvent(500, req2, "inst1")
	sim.handleRequestCompleted(event2)
}

// TestClusterSimulator_BC14_EventQueueBounds tests BC-14: event queue bounded
func TestClusterSimulator_BC14_EventQueueBounds(t *testing.T) {
	sim := NewClusterSimulator(10000)

	// Add instance
	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool.AddInstance(inst1)
	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}
	sim.AddDeployment(config)

	// Schedule multiple requests
	numRequests := 100
	for i := 0; i < numRequests; i++ {
		req := &Request{
			ID:           string(rune('A' + i)),
			PromptTokens: 100,
			OutputTokens: 50,
		}
		sim.ScheduleEvent(NewRequestArrivalEvent(int64(100*i), req))
	}

	initialQueueSize := sim.EventQueue.Len()

	// Event queue should not grow unboundedly
	// Each request generates: 1 arrival + 1 route + some steps + 0-1 completion
	// So maximum queue size is O(numRequests * events_per_request)
	maxExpectedSize := numRequests * 10 // Very generous bound

	if initialQueueSize > maxExpectedSize {
		t.Errorf("Event queue size %d exceeds bound %d", initialQueueSize, maxExpectedSize)
	}

	// Process some events
	for i := 0; i < 50 && sim.EventQueue.Len() > 0; i++ {
		event := sim.EventQueue.PopNext()
		if event.Timestamp() > sim.Horizon {
			break
		}
		sim.Clock = event.Timestamp()
		event.Execute(sim)

		// Verify queue doesn't grow unboundedly
		if sim.EventQueue.Len() > maxExpectedSize {
			t.Errorf("Event queue grew to %d, exceeds bound %d", sim.EventQueue.Len(), maxExpectedSize)
		}
	}
}

// TestClusterSimulator_MetricsAggregation tests metric computation
func TestClusterSimulator_MetricsAggregation(t *testing.T) {
	sim := NewClusterSimulator(10000)

	// Add instances with some metrics
	pool, _ := NewReplicaPool("pool1", PoolMonolithic, 0, 2)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	inst1.CompletedRequests = 5
	inst1.TotalInputTokens = 500
	inst1.TotalOutputTokens = 250

	inst2 := NewInstanceSimulator("inst2", PoolMonolithic, nil, 1000, 16)
	inst2.CompletedRequests = 3
	inst2.TotalInputTokens = 300
	inst2.TotalOutputTokens = 150

	pool.AddInstance(inst1)
	pool.AddInstance(inst2)

	config := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool,
	}
	sim.AddDeployment(config)

	sim.CompletedRequests = 8

	metrics := sim.ComputeMetrics()

	if metrics.CompletedRequests != 8 {
		t.Errorf("CompletedRequests = %d, want 8", metrics.CompletedRequests)
	}
	if metrics.TotalInputTokens != 800 {
		t.Errorf("TotalInputTokens = %d, want 800", metrics.TotalInputTokens)
	}
	if metrics.TotalOutputTokens != 400 {
		t.Errorf("TotalOutputTokens = %d, want 400", metrics.TotalOutputTokens)
	}
	if len(metrics.PerInstance) != 2 {
		t.Errorf("PerInstance count = %d, want 2", len(metrics.PerInstance))
	}
}
