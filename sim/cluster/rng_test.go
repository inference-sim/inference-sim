package cluster

import (
	"testing"
)

// TestPartitionedRNG_Creation tests RNG creation
func TestPartitionedRNG_Creation(t *testing.T) {
	rng := NewPartitionedRNG(42)

	if rng == nil {
		t.Fatal("NewPartitionedRNG returned nil")
	}
	if rng.masterSeed != 42 {
		t.Errorf("masterSeed = %d, want 42", rng.masterSeed)
	}
	if len(rng.subsystems) != 0 {
		t.Errorf("Initial subsystems count = %d, want 0", len(rng.subsystems))
	}
}

// TestPartitionedRNG_ForSubsystem tests subsystem RNG creation
func TestPartitionedRNG_ForSubsystem(t *testing.T) {
	rng := NewPartitionedRNG(42)

	// Get subsystem RNG
	workloadRNG := rng.ForSubsystem("workload")
	if workloadRNG == nil {
		t.Fatal("ForSubsystem returned nil")
	}

	// Second call should return same instance
	workloadRNG2 := rng.ForSubsystem("workload")
	if workloadRNG != workloadRNG2 {
		t.Error("ForSubsystem should return same instance on repeated calls")
	}

	// Different subsystem should return different instance
	routerRNG := rng.ForSubsystem("router")
	if routerRNG == workloadRNG {
		t.Error("Different subsystems should return different RNG instances")
	}
}

// TestPartitionedRNG_ForInstance tests instance-specific RNG
func TestPartitionedRNG_ForInstance(t *testing.T) {
	rng := NewPartitionedRNG(42)

	inst1RNG := rng.ForInstance("inst1")
	inst2RNG := rng.ForInstance("inst2")

	if inst1RNG == nil || inst2RNG == nil {
		t.Fatal("ForInstance returned nil")
	}
	if inst1RNG == inst2RNG {
		t.Error("Different instances should return different RNG instances")
	}

	// Second call should return same instance
	inst1RNG2 := rng.ForInstance("inst1")
	if inst1RNG != inst1RNG2 {
		t.Error("ForInstance should return same instance on repeated calls")
	}
}

// TestPartitionedRNG_BC10_SubsystemIsolation tests BC-10: RNG subsystem isolation
func TestPartitionedRNG_BC10_SubsystemIsolation(t *testing.T) {
	// Create two RNGs with same seed
	rng1 := NewPartitionedRNG(42)
	rng2 := NewPartitionedRNG(42)

	// Generate sequence from "router" subsystem in rng1
	router1 := rng1.ForSubsystem("router")
	seq1 := make([]int, 10)
	for i := 0; i < 10; i++ {
		seq1[i] = router1.Intn(1000)
	}

	// In rng2, generate from "workload" first (consuming RNG)
	workload2 := rng2.ForSubsystem("workload")
	for i := 0; i < 100; i++ {
		workload2.Intn(1000) // Consume workload RNG
	}

	// Now generate from "router" in rng2
	router2 := rng2.ForSubsystem("router")
	seq2 := make([]int, 10)
	for i := 0; i < 10; i++ {
		seq2[i] = router2.Intn(1000)
	}

	// Sequences should be identical despite different access patterns
	for i := 0; i < 10; i++ {
		if seq1[i] != seq2[i] {
			t.Errorf("Subsystem isolation violated at position %d: seq1=%d, seq2=%d", i, seq1[i], seq2[i])
		}
	}
}

// TestPartitionedRNG_BC10_OrderIndependence tests that seed derivation is order-independent
func TestPartitionedRNG_BC10_OrderIndependence(t *testing.T) {
	// Create two RNGs with same seed
	rng1 := NewPartitionedRNG(123)
	rng2 := NewPartitionedRNG(123)

	// Access subsystems in different order
	// rng1: A, B, C
	rngA1 := rng1.ForSubsystem("A")
	rngB1 := rng1.ForSubsystem("B")
	rngC1 := rng1.ForSubsystem("C")

	// rng2: C, B, A
	rngC2 := rng2.ForSubsystem("C")
	rngB2 := rng2.ForSubsystem("B")
	rngA2 := rng2.ForSubsystem("A")

	// Generate sequences from each subsystem
	seqA1 := rngA1.Intn(10000)
	seqB1 := rngB1.Intn(10000)
	seqC1 := rngC1.Intn(10000)

	seqA2 := rngA2.Intn(10000)
	seqB2 := rngB2.Intn(10000)
	seqC2 := rngC2.Intn(10000)

	// Sequences should match regardless of access order
	if seqA1 != seqA2 {
		t.Errorf("Subsystem A sequences differ: %d vs %d", seqA1, seqA2)
	}
	if seqB1 != seqB2 {
		t.Errorf("Subsystem B sequences differ: %d vs %d", seqB1, seqB2)
	}
	if seqC1 != seqC2 {
		t.Errorf("Subsystem C sequences differ: %d vs %d", seqC1, seqC2)
	}
}

// TestPartitionedRNG_BC10_NoInterference tests that consuming one subsystem doesn't affect another
func TestPartitionedRNG_BC10_NoInterference(t *testing.T) {
	rng := NewPartitionedRNG(999)

	// Generate baseline sequence from subsystem A
	rngA := rng.ForSubsystem("A")
	baseline := make([]int, 5)
	for i := 0; i < 5; i++ {
		baseline[i] = rngA.Intn(1000)
	}

	// Consume lots of values from subsystem B
	rngB := rng.ForSubsystem("B")
	for i := 0; i < 10000; i++ {
		rngB.Intn(1000)
	}

	// Continue generating from subsystem A
	continued := make([]int, 5)
	for i := 0; i < 5; i++ {
		continued[i] = rngA.Intn(1000)
	}

	// Create new RNG with same seed to verify expected sequence
	rng2 := NewPartitionedRNG(999)
	rngA2 := rng2.ForSubsystem("A")
	expected := make([]int, 10)
	for i := 0; i < 10; i++ {
		expected[i] = rngA2.Intn(1000)
	}

	// Verify baseline matches
	for i := 0; i < 5; i++ {
		if baseline[i] != expected[i] {
			t.Errorf("Baseline mismatch at %d: got %d, want %d", i, baseline[i], expected[i])
		}
	}

	// Verify continued matches (subsystem B consumption had no effect)
	for i := 0; i < 5; i++ {
		if continued[i] != expected[5+i] {
			t.Errorf("Continued mismatch at %d: got %d, want %d", i, continued[i], expected[5+i])
		}
	}
}

// TestPartitionedRNG_DifferentSeeds tests that different master seeds produce different sequences
func TestPartitionedRNG_DifferentSeeds(t *testing.T) {
	rng1 := NewPartitionedRNG(42)
	rng2 := NewPartitionedRNG(43)

	// Generate sequences from same subsystem
	workload1 := rng1.ForSubsystem("workload")
	workload2 := rng2.ForSubsystem("workload")

	seq1 := make([]int, 10)
	seq2 := make([]int, 10)

	for i := 0; i < 10; i++ {
		seq1[i] = workload1.Intn(10000)
		seq2[i] = workload2.Intn(10000)
	}

	// Sequences should differ
	allSame := true
	for i := 0; i < 10; i++ {
		if seq1[i] != seq2[i] {
			allSame = false
			break
		}
	}

	if allSame {
		t.Error("Different master seeds should produce different sequences")
	}
}

// TestPartitionedRNG_DeterministicDerivation tests that seed derivation is deterministic
func TestPartitionedRNG_DeterministicDerivation(t *testing.T) {
	// Create multiple RNGs with same seed
	rngs := []*PartitionedRNG{
		NewPartitionedRNG(777),
		NewPartitionedRNG(777),
		NewPartitionedRNG(777),
	}

	// Derive seeds for same subsystem across all RNGs
	seeds := make([]int64, len(rngs))
	for i, rng := range rngs {
		seeds[i] = rng.deriveSeed("test_subsystem")
	}

	// All seeds should be identical
	for i := 1; i < len(seeds); i++ {
		if seeds[i] != seeds[0] {
			t.Errorf("Seed derivation not deterministic: seeds[%d]=%d != seeds[0]=%d", i, seeds[i], seeds[0])
		}
	}

	// Different subsystem names should produce different seeds
	seed1 := rngs[0].deriveSeed("subsystem_A")
	seed2 := rngs[0].deriveSeed("subsystem_B")

	if seed1 == seed2 {
		t.Error("Different subsystem names should produce different seeds")
	}
}

// TestSimulationKey_Structure tests SimulationKey structure
func TestSimulationKey_Structure(t *testing.T) {
	key := SimulationKey{
		PolicyID:     "policy_v1",
		WorkloadSeed: 100,
		SimSeed:      200,
		JitterSeed:   300,
	}

	if key.PolicyID != "policy_v1" {
		t.Errorf("PolicyID = %s, want policy_v1", key.PolicyID)
	}
	if key.WorkloadSeed != 100 {
		t.Errorf("WorkloadSeed = %d, want 100", key.WorkloadSeed)
	}
	if key.SimSeed != 200 {
		t.Errorf("SimSeed = %d, want 200", key.SimSeed)
	}
	if key.JitterSeed != 300 {
		t.Errorf("JitterSeed = %d, want 300", key.JitterSeed)
	}
}

// TestPartitionedRNG_SubsystemConstants tests that subsystem constants are defined
func TestPartitionedRNG_SubsystemConstants(t *testing.T) {
	if SubsystemWorkload == "" {
		t.Error("SubsystemWorkload constant is empty")
	}
	if SubsystemRouter == "" {
		t.Error("SubsystemRouter constant is empty")
	}
	if SubsystemScheduler == "" {
		t.Error("SubsystemScheduler constant is empty")
	}

	// Verify constants have expected values
	if SubsystemWorkload != "workload" {
		t.Errorf("SubsystemWorkload = %s, want workload", SubsystemWorkload)
	}
	if SubsystemRouter != "router" {
		t.Errorf("SubsystemRouter = %s, want router", SubsystemRouter)
	}
	if SubsystemScheduler != "scheduler" {
		t.Errorf("SubsystemScheduler = %s, want scheduler", SubsystemScheduler)
	}
}

// TestPartitionedRNG_InstancePrefix tests that ForInstance uses correct prefix
func TestPartitionedRNG_InstancePrefix(t *testing.T) {
	rng := NewPartitionedRNG(42)

	// Get instance RNG
	instRNG := rng.ForInstance("inst1")

	// Get equivalent subsystem RNG
	subsysRNG := rng.ForSubsystem("instance_inst1")

	// They should be the same RNG instance
	if instRNG != subsysRNG {
		t.Error("ForInstance should use 'instance_' prefix")
	}

	// Generate values to verify they're truly the same
	val1 := instRNG.Intn(10000)
	val2 := subsysRNG.Intn(10000)

	if val1 == val2 {
		t.Error("Values should differ since they're from the same RNG (state advanced)")
	}
}
