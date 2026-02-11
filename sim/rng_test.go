package sim

import (
	"math"
	"math/rand"
	"testing"
)

// === SimulationKey Tests ===

func TestSimulationKey_Creation(t *testing.T) {
	tests := []struct {
		name string
		seed int64
	}{
		{"positive seed", 42},
		{"zero seed", 0},
		{"negative seed", -1},
		{"max int64", math.MaxInt64},
		{"min int64", math.MinInt64},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			key := NewSimulationKey(tt.seed)
			if int64(key) != tt.seed {
				t.Errorf("NewSimulationKey(%d) = %d, want %d", tt.seed, key, tt.seed)
			}
		})
	}
}

// === PartitionedRNG Tests ===

func TestPartitionedRNG_DeterministicDerivation(t *testing.T) {
	// BDD: Same key+name produces same sequence
	rng1 := NewPartitionedRNG(NewSimulationKey(42))
	rng2 := NewPartitionedRNG(NewSimulationKey(42))

	// Draw 3 values from router subsystem in each
	vals1 := make([]float64, 3)
	vals2 := make([]float64, 3)

	for i := 0; i < 3; i++ {
		vals1[i] = rng1.ForSubsystem(SubsystemRouter).Float64()
	}
	for i := 0; i < 3; i++ {
		vals2[i] = rng2.ForSubsystem(SubsystemRouter).Float64()
	}

	for i := 0; i < 3; i++ {
		if vals1[i] != vals2[i] {
			t.Errorf("Value %d: got %v and %v, want identical", i, vals1[i], vals2[i])
		}
	}
}

func TestPartitionedRNG_SubsystemIsolation(t *testing.T) {
	// BDD: Drawing from subsystem A doesn't affect subsystem B
	rngA := NewPartitionedRNG(NewSimulationKey(42))
	rngB := NewPartitionedRNG(NewSimulationKey(42))

	// Draw 10 values from A's workload subsystem (this should NOT affect router)
	for i := 0; i < 10; i++ {
		rngA.ForSubsystem(SubsystemWorkload).Float64()
	}

	// Draw 5 values from B's router subsystem
	for i := 0; i < 5; i++ {
		rngB.ForSubsystem(SubsystemRouter).Float64()
	}

	// Now draw from A's router - should be 1st value in router sequence
	aRouterFirst := rngA.ForSubsystem(SubsystemRouter).Float64()

	// Draw 6th value from B's router
	bRouterSixth := rngB.ForSubsystem(SubsystemRouter).Float64()

	// Create fresh RNG to get expected 1st router value
	fresh := NewPartitionedRNG(NewSimulationKey(42))
	expectedFirst := fresh.ForSubsystem(SubsystemRouter).Float64()

	if aRouterFirst != expectedFirst {
		t.Errorf("A's router first value = %v, want %v (isolation broken)", aRouterFirst, expectedFirst)
	}

	// bRouterSixth should be the 6th value, NOT equal to first
	if bRouterSixth == expectedFirst {
		t.Error("B's 6th router value equals 1st value - unexpected")
	}
}

func TestPartitionedRNG_WorkloadBackwardCompat(t *testing.T) {
	// BDD: "workload" subsystem uses master seed directly
	seed := int64(42)
	rng := NewPartitionedRNG(NewSimulationKey(seed))

	// Get workload RNG
	workloadRNG := rng.ForSubsystem(SubsystemWorkload)

	// Create a direct RNG with same seed (old implementation)
	directRNG := newRandFromSeed(seed)

	// They should produce identical sequences
	for i := 0; i < 10; i++ {
		got := workloadRNG.Float64()
		want := directRNG.Float64()
		if got != want {
			t.Errorf("Value %d: workload RNG = %v, direct RNG = %v", i, got, want)
		}
	}
}

func TestPartitionedRNG_CachesInstance(t *testing.T) {
	// BDD: Same name returns same *rand.Rand instance
	rng := NewPartitionedRNG(NewSimulationKey(42))

	rng1 := rng.ForSubsystem(SubsystemWorkload)
	rng2 := rng.ForSubsystem(SubsystemWorkload)

	if rng1 != rng2 {
		t.Error("ForSubsystem returned different instances for same name")
	}
}

func TestPartitionedRNG_Key(t *testing.T) {
	seed := int64(12345)
	rng := NewPartitionedRNG(NewSimulationKey(seed))

	if rng.Key() != SimulationKey(seed) {
		t.Errorf("Key() = %v, want %v", rng.Key(), seed)
	}
}

func TestPartitionedRNG_EmptySubsystemName(t *testing.T) {
	// BDD: Empty string is valid subsystem name
	rng := NewPartitionedRNG(NewSimulationKey(42))
	result := rng.ForSubsystem("")

	if result == nil {
		t.Error("ForSubsystem(\"\") returned nil")
	}

	// Should be deterministic
	rng2 := NewPartitionedRNG(NewSimulationKey(42))
	result2 := rng2.ForSubsystem("")

	val1 := result.Float64()
	// Need to recreate since result was already used
	rng3 := NewPartitionedRNG(NewSimulationKey(42))
	val2 := rng3.ForSubsystem("").Float64()

	if val1 != val2 {
		t.Errorf("Empty subsystem not deterministic: %v != %v", val1, val2)
	}
	_ = result2 // silence unused warning
}

func TestPartitionedRNG_ZeroSeed(t *testing.T) {
	// BDD: Seed 0 works correctly
	rng := NewPartitionedRNG(NewSimulationKey(0))

	workload := rng.ForSubsystem(SubsystemWorkload)
	router := rng.ForSubsystem(SubsystemRouter)

	if workload == nil || router == nil {
		t.Error("ForSubsystem returned nil with zero seed")
	}

	// workload should use seed 0 directly
	directRNG := newRandFromSeed(0)
	if workload.Float64() != directRNG.Float64() {
		t.Error("Workload with seed 0 not matching direct RNG")
	}
}

func TestPartitionedRNG_NegativeSeed(t *testing.T) {
	// BDD: MinInt64 seed works correctly
	rng := NewPartitionedRNG(NewSimulationKey(math.MinInt64))

	workload := rng.ForSubsystem(SubsystemWorkload)
	router := rng.ForSubsystem(SubsystemRouter)

	if workload == nil || router == nil {
		t.Error("ForSubsystem returned nil with MinInt64 seed")
	}

	// Should produce valid random values
	val := workload.Float64()
	if val < 0 || val >= 1 {
		t.Errorf("Float64() returned %v, want [0, 1)", val)
	}
}

func TestPartitionedRNG_LazyInitialization(t *testing.T) {
	// BDD: Subsystems map is empty until ForSubsystem is called
	rng := NewPartitionedRNG(NewSimulationKey(42))

	if len(rng.subsystems) != 0 {
		t.Errorf("New PartitionedRNG has %d subsystems, want 0", len(rng.subsystems))
	}

	rng.ForSubsystem(SubsystemWorkload)

	if len(rng.subsystems) != 1 {
		t.Errorf("After one ForSubsystem call, have %d subsystems, want 1", len(rng.subsystems))
	}
}

// === fnv1a64 Tests ===

func TestFnv1a64_Deterministic(t *testing.T) {
	// Same input produces same hash
	input := "test_subsystem"
	hash1 := fnv1a64(input)
	hash2 := fnv1a64(input)

	if hash1 != hash2 {
		t.Errorf("fnv1a64(%q) not deterministic: %v != %v", input, hash1, hash2)
	}
}

func TestFnv1a64_Collision(t *testing.T) {
	// Different subsystem names should produce different hashes (spot check)
	names := []string{
		SubsystemWorkload,
		SubsystemRouter,
		"instance_0",
		"instance_1",
		"instance_100",
		"",
	}

	hashes := make(map[int64]string)
	for _, name := range names {
		h := fnv1a64(name)
		if existing, ok := hashes[h]; ok {
			t.Errorf("Hash collision: %q and %q both hash to %d", name, existing, h)
		}
		hashes[h] = name
	}
}

// === SubsystemInstance Tests ===

func TestSubsystemInstance(t *testing.T) {
	tests := []struct {
		id   int
		want string
	}{
		{0, "instance_0"},
		{1, "instance_1"},
		{100, "instance_100"},
		{-1, "instance_-1"},
	}

	for _, tt := range tests {
		got := SubsystemInstance(tt.id)
		if got != tt.want {
			t.Errorf("SubsystemInstance(%d) = %q, want %q", tt.id, got, tt.want)
		}
	}
}

// === Benchmark ===

func BenchmarkPartitionedRNG_ForSubsystem_CacheHit(b *testing.B) {
	rng := NewPartitionedRNG(NewSimulationKey(42))
	// Prime the cache
	rng.ForSubsystem(SubsystemWorkload)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng.ForSubsystem(SubsystemWorkload)
	}
}

func BenchmarkPartitionedRNG_ForSubsystem_CacheMiss(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rng := NewPartitionedRNG(NewSimulationKey(42))
		rng.ForSubsystem(SubsystemWorkload)
	}
}

// === Helper ===

// newRandFromSeed creates a *rand.Rand with the given seed (mirrors old implementation)
func newRandFromSeed(seed int64) *rand.Rand {
	return rand.New(rand.NewSource(seed))
}
