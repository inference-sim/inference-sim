package sim

import (
	"fmt"
	"hash/fnv"
	"math/rand"
)

// === SimulationKey ===

// SimulationKey uniquely identifies a reproducible simulation run.
// Two simulations with the same SimulationKey and identical configuration
// MUST produce bit-for-bit identical results.
type SimulationKey int64

// NewSimulationKey creates a SimulationKey from a seed value.
func NewSimulationKey(seed int64) SimulationKey {
	return SimulationKey(seed)
}

// === Subsystem Constants ===

const (
	// SubsystemWorkload is the RNG subsystem for workload generation.
	// Uses master seed directly for backward compatibility.
	SubsystemWorkload = "workload"

	// SubsystemRouter is the RNG subsystem for routing decisions.
	// Used in PR 6+.
	SubsystemRouter = "router"
)

// SubsystemInstance returns the subsystem name for instance N.
// Used in PR 2+ for per-instance RNG isolation.
func SubsystemInstance(id int) string {
	return fmt.Sprintf("instance_%d", id)
}

// === PartitionedRNG ===

// PartitionedRNG provides deterministic, isolated RNG instances per subsystem.
//
// Derivation formula:
//   - For SubsystemWorkload: uses masterSeed directly (backward compatibility)
//   - For all other subsystems: masterSeed XOR fnv1a64(subsystemName)
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
type PartitionedRNG struct {
	key        SimulationKey
	subsystems map[string]*rand.Rand
}

// NewPartitionedRNG creates a PartitionedRNG from a SimulationKey.
func NewPartitionedRNG(key SimulationKey) *PartitionedRNG {
	return &PartitionedRNG{
		key:        key,
		subsystems: make(map[string]*rand.Rand),
	}
}

// ForSubsystem returns a deterministically-seeded RNG for the named subsystem.
// The same subsystem name always returns the same *rand.Rand instance (cached).
// Never returns nil.
func (p *PartitionedRNG) ForSubsystem(name string) *rand.Rand {
	if rng, ok := p.subsystems[name]; ok {
		return rng
	}

	var derivedSeed int64
	if name == SubsystemWorkload {
		// Backward compatibility: workload uses master seed directly.
		// This ensures existing --seed behavior produces identical output.
		derivedSeed = int64(p.key)
	} else {
		// All other subsystems: XOR with hash for isolation.
		derivedSeed = int64(p.key) ^ fnv1a64(name)
	}

	rng := rand.New(rand.NewSource(derivedSeed))
	p.subsystems[name] = rng
	return rng
}

// Key returns the SimulationKey used to create this PartitionedRNG.
func (p *PartitionedRNG) Key() SimulationKey {
	return p.key
}

// fnv1a64 computes a 64-bit FNV-1a hash of the input string.
func fnv1a64(s string) int64 {
	h := fnv.New64a()
	h.Write([]byte(s))
	return int64(h.Sum64())
}
