package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// PoolRole identifies whether an instance serves as prefill or decode in PD disaggregation.
type PoolRole int

const (
	// PoolRolePrefill indicates the instance handles prefill (prompt processing).
	PoolRolePrefill PoolRole = iota + 1
	// PoolRoleDecode indicates the instance handles decode (token generation).
	PoolRoleDecode
)

// String returns a human-readable name for the pool role.
func (r PoolRole) String() string {
	switch r {
	case PoolRolePrefill:
		return "prefill"
	case PoolRoleDecode:
		return "decode"
	default:
		return fmt.Sprintf("PoolRole(%d)", int(r))
	}
}

// ValidatePoolTopology checks that PD pool configuration is valid.
// Returns nil if pools are disabled (both prefill and decode are 0).
// Returns an error if:
//   - prefill or decode is negative
//   - only one of prefill/decode is set (both must be set or neither)
//   - prefill + decode exceeds total instances
func ValidatePoolTopology(prefill, decode, total int) error {
	if prefill < 0 {
		return fmt.Errorf("prefill-instances must be >= 0, got %d", prefill)
	}
	if decode < 0 {
		return fmt.Errorf("decode-instances must be >= 0, got %d", decode)
	}
	// Both zero = disabled, no further checks
	if prefill == 0 && decode == 0 {
		return nil
	}
	// Both must be set when disaggregation is enabled
	if prefill == 0 || decode == 0 {
		return fmt.Errorf("both --prefill-instances and --decode-instances must be set when PD disaggregation is enabled (got prefill=%d, decode=%d)", prefill, decode)
	}
	if prefill+decode > total {
		return fmt.Errorf("prefill-instances (%d) + decode-instances (%d) = %d exceeds num-instances (%d)", prefill, decode, prefill+decode, total)
	}
	return nil
}

// BuildPoolMembership constructs an immutable map of instance ID → PoolRole.
// Instances 0..prefill-1 are assigned PoolRolePrefill, prefill..prefill+decode-1 are PoolRoleDecode.
// Caller must validate prefill+decode <= len(instances) before calling.
func BuildPoolMembership(instances []*InstanceSimulator, prefill, decode int) map[string]PoolRole {
	if prefill+decode > len(instances) {
		panic(fmt.Sprintf("BuildPoolMembership: prefill(%d)+decode(%d)=%d exceeds len(instances)=%d",
			prefill, decode, prefill+decode, len(instances)))
	}
	membership := make(map[string]PoolRole, prefill+decode)
	for i := 0; i < prefill; i++ {
		membership[string(instances[i].ID())] = PoolRolePrefill
	}
	for i := prefill; i < prefill+decode; i++ {
		membership[string(instances[i].ID())] = PoolRoleDecode
	}
	return membership
}

// BuildPoolMembershipFromIndices constructs a pool membership map from instance indices.
// Uses the same instance ID naming convention as NewClusterSimulator: "instance_N".
// This variant does not require constructed instances, enabling pool membership
// computation before instance construction (needed for per-pool config resolution).
// Caller must validate prefill+decode <= total before calling.
func BuildPoolMembershipFromIndices(total, prefill, decode int) map[string]PoolRole {
	if prefill+decode > total {
		panic(fmt.Sprintf("BuildPoolMembershipFromIndices: prefill(%d)+decode(%d)=%d exceeds total(%d)",
			prefill, decode, prefill+decode, total))
	}
	membership := make(map[string]PoolRole, prefill+decode)
	for i := 0; i < prefill; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRolePrefill
	}
	for i := prefill; i < prefill+decode; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRoleDecode
	}
	return membership
}

// FilterSnapshotsByPool returns only the snapshots for instances in the given pool role.
// Order is preserved (stable relative to the input slice).
func FilterSnapshotsByPool(snapshots []sim.RoutingSnapshot, membership map[string]PoolRole, role PoolRole) []sim.RoutingSnapshot {
	filtered := make([]sim.RoutingSnapshot, 0, len(snapshots))
	for _, snap := range snapshots {
		if membership[snap.ID] == role {
			filtered = append(filtered, snap)
		}
	}
	return filtered
}
