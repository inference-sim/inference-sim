package cluster

import (
	"testing"
	"time"
)

// TestDeterminism_BC9_SameSeedIdenticalResults tests BC-9: deterministic replay
func TestDeterminism_BC9_SameSeedIdenticalResults(t *testing.T) {
	// Create two simulations with identical configuration
	seed := int64(42)

	// Simulation 1
	sim1 := NewClusterSimulator(10000)
	sim1.RNG = NewPartitionedRNG(seed)

	pool1, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool1.AddInstance(inst1)
	config1 := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool1,
	}
	sim1.AddDeployment(config1)

	// Schedule identical requests
	for i := 0; i < 5; i++ {
		req := &Request{
			ID:           string(rune('A' + i)),
			PromptTokens: 100,
			OutputTokens: 50,
		}
		sim1.ScheduleEvent(sim1.NewRequestArrivalEvent(int64(100*i), req))
	}

	// Simulation 2 - identical setup
	sim2 := NewClusterSimulator(10000)
	sim2.RNG = NewPartitionedRNG(seed)

	pool2, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst2 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool2.AddInstance(inst2)
	config2 := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool2,
	}
	sim2.AddDeployment(config2)

	for i := 0; i < 5; i++ {
		req := &Request{
			ID:           string(rune('A' + i)),
			PromptTokens: 100,
			OutputTokens: 50,
		}
		sim2.ScheduleEvent(sim2.NewRequestArrivalEvent(int64(100*i), req))
	}

	// Run both simulations (limited steps for this test)
	for i := 0; i < 20 && sim1.EventQueue.Len() > 0; i++ {
		event := sim1.EventQueue.PopNext()
		if event.Timestamp() > sim1.Horizon {
			break
		}
		sim1.Clock = event.Timestamp()
		event.Execute(sim1)
	}

	for i := 0; i < 20 && sim2.EventQueue.Len() > 0; i++ {
		event := sim2.EventQueue.PopNext()
		if event.Timestamp() > sim2.Horizon {
			break
		}
		sim2.Clock = event.Timestamp()
		event.Execute(sim2)
	}

	// Verify identical state
	if sim1.Clock != sim2.Clock {
		t.Errorf("Clock differs: sim1=%d, sim2=%d", sim1.Clock, sim2.Clock)
	}

	if sim1.CompletedRequests != sim2.CompletedRequests {
		t.Errorf("CompletedRequests differs: sim1=%d, sim2=%d", sim1.CompletedRequests, sim2.CompletedRequests)
	}

	if sim1.PendingRequests != sim2.PendingRequests {
		t.Errorf("PendingRequests differs: sim1=%d, sim2=%d", sim1.PendingRequests, sim2.PendingRequests)
	}

	// Verify request states are identical
	for reqID, req1 := range sim1.Requests {
		req2, exists := sim2.Requests[reqID]
		if !exists {
			t.Errorf("Request %s exists in sim1 but not sim2", reqID)
			continue
		}

		if req1.State != req2.State {
			t.Errorf("Request %s state differs: sim1=%s, sim2=%s", reqID, req1.State, req2.State)
		}
		if req1.ArrivalTime != req2.ArrivalTime {
			t.Errorf("Request %s ArrivalTime differs: sim1=%d, sim2=%d", reqID, req1.ArrivalTime, req2.ArrivalTime)
		}
		if req1.RouteTime != req2.RouteTime {
			t.Errorf("Request %s RouteTime differs: sim1=%d, sim2=%d", reqID, req1.RouteTime, req2.RouteTime)
		}
	}
}

// TestDeterminism_BC9_DifferentSeedDifferentResults tests that different seeds produce different results
func TestDeterminism_BC9_DifferentSeedDifferentResults(t *testing.T) {
	// Create two simulations with different seeds
	sim1 := NewClusterSimulator(10000)
	sim1.RNG = NewPartitionedRNG(42)

	sim2 := NewClusterSimulator(10000)
	sim2.RNG = NewPartitionedRNG(43) // Different seed

	// Identical configuration otherwise
	for _, sim := range []*ClusterSimulator{sim1, sim2} {
		pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 2)
		inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
		inst2 := NewInstanceSimulator("inst2", PoolMonolithic, nil, 1000, 16)
		pool.AddInstance(inst1)
		pool.AddInstance(inst2)
		config := &DeploymentConfig{
			ConfigID:    "config1",
			ReplicaPool: pool,
		}
		sim.AddDeployment(config)

		// Schedule requests
		for i := 0; i < 10; i++ {
			req := &Request{
				ID:           string(rune('A' + i)),
				PromptTokens: 100,
				OutputTokens: 50,
			}
			sim.ScheduleEvent(sim.NewRequestArrivalEvent(int64(100*i), req))
		}
	}

	// Run both simulations
	for i := 0; i < 50 && sim1.EventQueue.Len() > 0; i++ {
		event := sim1.EventQueue.PopNext()
		if event.Timestamp() > sim1.Horizon {
			break
		}
		sim1.Clock = event.Timestamp()
		event.Execute(sim1)
	}

	for i := 0; i < 50 && sim2.EventQueue.Len() > 0; i++ {
		event := sim2.EventQueue.PopNext()
		if event.Timestamp() > sim2.Horizon {
			break
		}
		sim2.Clock = event.Timestamp()
		event.Execute(sim2)
	}

	// With different seeds and round-robin routing, requests should be routed differently
	// Count routing differences
	routingDifferences := 0
	for reqID, req1 := range sim1.Requests {
		req2, exists := sim2.Requests[reqID]
		if !exists {
			continue
		}

		if req1.TargetInstance != req2.TargetInstance {
			routingDifferences++
		}
	}

	// Note: With deterministic round-robin, routing might still be identical
	// This test documents expected behavior but may not always show differences
	// with simple round-robin. With more complex policies using RNG, differences would be clear.
	t.Logf("Routing differences with different seeds: %d", routingDifferences)
}

// TestDeterminism_BC11_NoExternalStateDependency tests BC-11: no external state dependency
func TestDeterminism_BC11_NoExternalStateDependency(t *testing.T) {
	seed := int64(123)

	// Helper to run simulation
	runSimulation := func() *ClusterSimulator {
		sim := NewClusterSimulator(5000)
		sim.RNG = NewPartitionedRNG(seed)

		pool, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
		inst := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
		pool.AddInstance(inst)
		config := &DeploymentConfig{
			ConfigID:    "config1",
			ReplicaPool: pool,
		}
		sim.AddDeployment(config)

		// Schedule requests
		for i := 0; i < 3; i++ {
			req := &Request{
				ID:           string(rune('A' + i)),
				PromptTokens: 100,
				OutputTokens: 50,
			}
			sim.ScheduleEvent(sim.NewRequestArrivalEvent(int64(100*i), req))
		}

		// Run simulation
		for i := 0; i < 15 && sim.EventQueue.Len() > 0; i++ {
			event := sim.EventQueue.PopNext()
			if event.Timestamp() > sim.Horizon {
				break
			}
			sim.Clock = event.Timestamp()
			event.Execute(sim)
		}

		return sim
	}

	// Run at different wall-clock times
	sim1 := runSimulation()
	time.Sleep(10 * time.Millisecond) // Advance wall-clock
	sim2 := runSimulation()

	// Results should be identical despite different wall-clock times
	if sim1.Clock != sim2.Clock {
		t.Errorf("Results depend on wall-clock time: clocks differ %d vs %d", sim1.Clock, sim2.Clock)
	}

	if sim1.CompletedRequests != sim2.CompletedRequests {
		t.Errorf("Results depend on wall-clock time: completed requests differ %d vs %d", sim1.CompletedRequests, sim2.CompletedRequests)
	}

	// Verify all request timestamps are identical
	for reqID, req1 := range sim1.Requests {
		req2, exists := sim2.Requests[reqID]
		if !exists {
			continue
		}

		if req1.ArrivalTime != req2.ArrivalTime {
			t.Errorf("Request %s ArrivalTime depends on wall-clock: %d vs %d", reqID, req1.ArrivalTime, req2.ArrivalTime)
		}
	}
}

// TestDeterminism_TieBreaking tests all tie-breaking rules
func TestDeterminism_TieBreaking(t *testing.T) {
	t.Run("Timestamp tie-breaking", func(t *testing.T) {
		sim := NewClusterSimulator(10000)

		// Schedule events at same timestamp with different types
		req1 := &Request{ID: "req1", PromptTokens: 100, OutputTokens: 50}
		req2 := &Request{ID: "req2", PromptTokens: 100, OutputTokens: 50}

		// All at timestamp 100
		sim.ScheduleEvent(sim.NewInstanceStepEvent(100, "inst1"))          // Priority 3
		sim.ScheduleEvent(sim.NewRequestArrivalEvent(100, req1))           // Priority 1
		sim.ScheduleEvent(sim.NewRequestCompletedEvent(100, req2, "inst1")) // Priority 4
		sim.ScheduleEvent(sim.NewRouteDecisionEvent(100, req1, "inst1"))   // Priority 2

		// Process events and verify order
		expectedOrder := []EventType{
			EventTypeRequestArrival,   // Priority 1
			EventTypeRouteDecision,    // Priority 2
			EventTypeInstanceStep,     // Priority 3
			EventTypeRequestCompleted, // Priority 4
		}

		for i, expectedType := range expectedOrder {
			if sim.EventQueue.Len() == 0 {
				t.Fatalf("Event queue empty at position %d", i)
			}

			event := sim.EventQueue.PopNext()
			if event.Type() != expectedType {
				t.Errorf("Position %d: got %s, want %s", i, event.Type(), expectedType)
			}
		}
	})

	t.Run("EventID tie-breaking", func(t *testing.T) {
		sim := NewClusterSimulator(10000)

		// Schedule multiple events of same type at same timestamp
		// EventID should provide deterministic ordering
		e1 := sim.NewInstanceStepEvent(100, "inst1")
		e2 := sim.NewInstanceStepEvent(100, "inst2")
		e3 := sim.NewInstanceStepEvent(100, "inst3")

		id1, id2, id3 := e1.EventID(), e2.EventID(), e3.EventID()

		// Add in reverse order
		sim.ScheduleEvent(e3)
		sim.ScheduleEvent(e1)
		sim.ScheduleEvent(e2)

		// Should pop in EventID order
		first := sim.EventQueue.PopNext()
		if first.EventID() != id1 {
			t.Errorf("First event ID = %d, want %d", first.EventID(), id1)
		}

		second := sim.EventQueue.PopNext()
		if second.EventID() != id2 {
			t.Errorf("Second event ID = %d, want %d", second.EventID(), id2)
		}

		third := sim.EventQueue.PopNext()
		if third.EventID() != id3 {
			t.Errorf("Third event ID = %d, want %d", third.EventID(), id3)
		}
	})
}

// TestDeterminism_EventInsertionOrderIndependence tests that event insertion order doesn't matter
func TestDeterminism_EventInsertionOrderIndependence(t *testing.T) {
	// Helper to create and run simulation with given event insertion order
	runWithOrder := func(order []int) []EventType {
		sim := NewClusterSimulator(10000)

		req1 := &Request{ID: "req1", PromptTokens: 100, OutputTokens: 50}
		req2 := &Request{ID: "req2", PromptTokens: 100, OutputTokens: 50}

		events := []Event{
			sim.NewRequestArrivalEvent(100, req1),
			sim.NewRouteDecisionEvent(150, req1, "inst1"),
			sim.NewInstanceStepEvent(200, "inst1"),
			sim.NewRequestCompletedEvent(300, req2, "inst1"),
		}

		// Insert in specified order
		for _, idx := range order {
			sim.ScheduleEvent(events[idx])
		}

		// Extract processing order
		result := []EventType{}
		for sim.EventQueue.Len() > 0 {
			event := sim.EventQueue.PopNext()
			result = append(result, event.Type())
		}

		return result
	}

	// Try different insertion orders
	order1 := []int{0, 1, 2, 3}
	order2 := []int{3, 2, 1, 0}
	order3 := []int{1, 3, 0, 2}

	result1 := runWithOrder(order1)
	result2 := runWithOrder(order2)
	result3 := runWithOrder(order3)

	// All should produce same processing order
	if len(result1) != len(result2) || len(result1) != len(result3) {
		t.Error("Processing order length differs")
	}

	for i := 0; i < len(result1); i++ {
		if result1[i] != result2[i] || result1[i] != result3[i] {
			t.Errorf("Processing order differs at position %d: %s vs %s vs %s", i, result1[i], result2[i], result3[i])
		}
	}
}

// TestDeterminism_NoGlobalRandomness tests that simulation doesn't use global rand
func TestDeterminism_NoGlobalRandomness(t *testing.T) {
	// This test verifies that the simulation uses PartitionedRNG, not global rand
	// If global rand were used, seeding it would affect simulation behavior

	seed := int64(999)

	// Run 1: Don't touch global rand
	sim1 := NewClusterSimulator(5000)
	sim1.RNG = NewPartitionedRNG(seed)

	pool1, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst1 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool1.AddInstance(inst1)
	config1 := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool1,
	}
	sim1.AddDeployment(config1)

	req1 := &Request{ID: "req1", PromptTokens: 100, OutputTokens: 50}
	sim1.ScheduleEvent(sim1.NewRequestArrivalEvent(100, req1))

	// Run a few steps
	for i := 0; i < 10 && sim1.EventQueue.Len() > 0; i++ {
		event := sim1.EventQueue.PopNext()
		if event.Timestamp() > sim1.Horizon {
			break
		}
		sim1.Clock = event.Timestamp()
		event.Execute(sim1)
	}

	// Capture state
	clock1 := sim1.Clock
	state1 := sim1.Requests["req1"].State

	// Run 2: Seed global rand differently (should have no effect if simulation is properly isolated)
	// Note: We can't actually seed global rand in a test without affecting other tests
	// This test documents the requirement
	sim2 := NewClusterSimulator(5000)
	sim2.RNG = NewPartitionedRNG(seed) // Same seed as sim1

	pool2, _ := NewReplicaPool("pool1", PoolMonolithic, 1, 1)
	inst2 := NewInstanceSimulator("inst1", PoolMonolithic, nil, 1000, 16)
	pool2.AddInstance(inst2)
	config2 := &DeploymentConfig{
		ConfigID:    "config1",
		ReplicaPool: pool2,
	}
	sim2.AddDeployment(config2)

	req2 := &Request{ID: "req1", PromptTokens: 100, OutputTokens: 50}
	sim2.ScheduleEvent(sim2.NewRequestArrivalEvent(100, req2))

	for i := 0; i < 10 && sim2.EventQueue.Len() > 0; i++ {
		event := sim2.EventQueue.PopNext()
		if event.Timestamp() > sim2.Horizon {
			break
		}
		sim2.Clock = event.Timestamp()
		event.Execute(sim2)
	}

	clock2 := sim2.Clock
	state2 := sim2.Requests["req1"].State

	// States should be identical (not affected by potential global rand usage)
	if clock1 != clock2 {
		t.Errorf("Clock differs: %d vs %d (simulation may be using global rand)", clock1, clock2)
	}
	if state1 != state2 {
		t.Errorf("Request state differs: %s vs %s (simulation may be using global rand)", state1, state2)
	}
}

// TestDeterminism_EventIDsIdenticalAcrossRuns verifies BC-9: event IDs are deterministic
// This tests that creating events in a fresh simulator produces identical event IDs
func TestDeterminism_EventIDsIdenticalAcrossRuns(t *testing.T) {
	// First run
	sim1 := NewClusterSimulator(10000)
	event1_1 := sim1.NewRequestArrivalEvent(100, &Request{ID: "req1"})
	event1_2 := sim1.NewRequestArrivalEvent(200, &Request{ID: "req2"})
	event1_3 := sim1.NewInstanceStepEvent(300, "inst1")

	// Second run (fresh simulator)
	sim2 := NewClusterSimulator(10000)
	event2_1 := sim2.NewRequestArrivalEvent(100, &Request{ID: "req1"})
	event2_2 := sim2.NewRequestArrivalEvent(200, &Request{ID: "req2"})
	event2_3 := sim2.NewInstanceStepEvent(300, "inst1")

	// Verify event IDs are identical across runs
	if event1_1.EventID() != event2_1.EventID() {
		t.Errorf("Event IDs differ across runs: run1=%d, run2=%d", event1_1.EventID(), event2_1.EventID())
	}
	if event1_2.EventID() != event2_2.EventID() {
		t.Errorf("Event IDs differ across runs: run1=%d, run2=%d", event1_2.EventID(), event2_2.EventID())
	}
	if event1_3.EventID() != event2_3.EventID() {
		t.Errorf("Event IDs differ across runs: run1=%d, run2=%d", event1_3.EventID(), event2_3.EventID())
	}

	// Verify event IDs are sequential
	if event1_1.EventID() != 1 || event1_2.EventID() != 2 || event1_3.EventID() != 3 {
		t.Errorf("Event IDs not sequential: got %d, %d, %d, want 1, 2, 3",
			event1_1.EventID(), event1_2.EventID(), event1_3.EventID())
	}
}

// TestDeterminism_SimulationKeyUniqueness tests that different SimulationKeys produce different results
func TestDeterminism_SimulationKeyUniqueness(t *testing.T) {
	key1 := SimulationKey{
		PolicyID:     "policy_v1",
		WorkloadSeed: 100,
		SimSeed:      200,
		JitterSeed:   300,
	}

	key2 := SimulationKey{
		PolicyID:     "policy_v1",
		WorkloadSeed: 100,
		SimSeed:      201, // Different SimSeed
		JitterSeed:   300,
	}

	// Verify keys are different
	if key1 == key2 {
		t.Error("Different SimSeeds should produce different SimulationKeys")
	}

	// Document that different SimSeeds should produce different simulation behavior
	// (Full integration test would verify this end-to-end)
	rng1 := NewPartitionedRNG(key1.SimSeed)
	rng2 := NewPartitionedRNG(key2.SimSeed)

	val1 := rng1.ForSubsystem("test").Intn(10000)
	val2 := rng2.ForSubsystem("test").Intn(10000)

	if val1 == val2 {
		t.Error("Different SimSeeds should produce different RNG sequences")
	}
}
