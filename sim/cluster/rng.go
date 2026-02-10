package cluster

import (
	"hash/fnv"
	"math/rand"
)

// SimulationKey uniquely identifies a simulation run and controls all randomness
type SimulationKey struct {
	PolicyID     string // Identifies policy version (Phase 2+)
	WorkloadSeed int64  // Controls workload generation (Phase 4+)
	SimSeed      int64  // Controls simulation randomness
	JitterSeed   int64  // Controls timing noise (if any)
}

// PartitionedRNG provides isolated RNG streams per subsystem for deterministic simulation
type PartitionedRNG struct {
	masterSeed  int64
	subsystems  map[string]*rand.Rand
}

// NewPartitionedRNG creates a new partitioned RNG with the given master seed
func NewPartitionedRNG(masterSeed int64) *PartitionedRNG {
	return &PartitionedRNG{
		masterSeed: masterSeed,
		subsystems: make(map[string]*rand.Rand),
	}
}

// ForSubsystem returns an RNG for the given subsystem name
// The subsystem RNG is created lazily and deterministically derived from master seed
// Multiple calls with same subsystem name return the same RNG instance
func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand {
	if rng, exists := p.subsystems[name]; exists {
		return rng
	}

	// Derive subsystem seed deterministically from master seed and subsystem name
	subsystemSeed := p.deriveSeed(name)
	rng := rand.New(rand.NewSource(subsystemSeed))
	p.subsystems[name] = rng
	return rng
}

// ForInstance returns an RNG for the given instance ID
// This is a convenience method that calls ForSubsystem with "instance_<id>"
func (p *PartitionedRNG) ForInstance(id InstanceID) *rand.Rand {
	return p.ForSubsystem("instance_" + string(id))
}

// deriveSeed deterministically derives a subsystem seed from master seed and subsystem name
// Uses hash-based derivation to ensure order-independence:
// subsystemSeed = masterSeed XOR hash(subsystemName)
func (p *PartitionedRNG) deriveSeed(subsystemName string) int64 {
	h := fnv.New64a()
	h.Write([]byte(subsystemName))
	nameHash := int64(h.Sum64())

	// XOR master seed with name hash for order-independent derivation
	return p.masterSeed ^ nameHash
}

// Subsystem name constants for common subsystems
const (
	SubsystemWorkload  = "workload"
	SubsystemRouter    = "router"
	SubsystemScheduler = "scheduler"
)
