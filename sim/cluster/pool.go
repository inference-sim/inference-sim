package cluster

import (
	"fmt"

	"blis/sim"
)

// PoolRole identifies which PD-disaggregation stages an instance serves.
// Encoded as a bitmask so a single instance can belong to multiple pools
// simultaneously (shared-role pods, matching llm-d role labels
// "prefill-decode" and legacy "both"). See issue #1276.
type PoolRole uint

const (
	// PoolRolePrefill indicates the instance handles prefill (prompt processing).
	PoolRolePrefill PoolRole = 1 << iota
	// PoolRoleDecode indicates the instance handles decode (token generation).
	PoolRoleDecode
	// PoolRoleEncode indicates the instance handles encoding of multimodal
	// input (image/video/audio) into token embeddings. E/P/D disaggregation
	// parity with llm-d (GAP-4, issue #1264). Only active when
	// --encode-instances > 0.
	PoolRoleEncode
	// PoolRolePrefillDecode is the shared-role marker for pods serving both
	// prefill and decode — the BLIS analogue of llm-d's "prefill-decode" /
	// legacy "both" role label.
	PoolRolePrefillDecode = PoolRolePrefill | PoolRoleDecode
)

// Has reports whether r carries every bit of other. Returns false if other is 0.
// Enables set-membership checks over a bitmask role, e.g., a shared-role
// instance satisfies both Has(PoolRolePrefill) and Has(PoolRoleDecode).
func (r PoolRole) Has(other PoolRole) bool {
	return other != 0 && r&other == other
}

// String returns a human-readable name for the pool role.
func (r PoolRole) String() string {
	switch r {
	case 0:
		return "PoolRole(0)"
	case PoolRolePrefill:
		return "prefill"
	case PoolRoleDecode:
		return "decode"
	case PoolRoleEncode:
		return "encode"
	case PoolRolePrefillDecode:
		return "prefill-decode"
	default:
		return fmt.Sprintf("PoolRole(%d)", uint(r))
	}
}

// ValidatePoolTopology checks that PD pool configuration is valid.
// Returns nil if pools are disabled (prefill, decode, shared, and encode are all 0).
// Returns an error if:
//   - prefill, decode, shared, or encode is negative
//   - shared is 0 and only one of prefill/decode is set (both must be set for
//     pure-disjoint PD) — except when encode-only is legitimately considered disabled
//   - prefill + decode + shared + encode exceeds total instances
//   - encode > 0 without any decode-capable pool (prefill+decode+shared == 0):
//     encode requires a decode-capable pool because encode is an optional stage
//     that precedes decode (GAP-4, issue #1264).
//
// When shared > 0, a pure-shared cluster (prefill=0, decode=0, shared>0) is
// legitimate, and a cluster with only prefill-only + shared or only decode-only
// + shared is also legitimate.
func ValidatePoolTopology(prefill, decode, shared, encode, total int) error {
	if prefill < 0 {
		return fmt.Errorf("prefill-instances must be >= 0, got %d", prefill)
	}
	if decode < 0 {
		return fmt.Errorf("decode-instances must be >= 0, got %d", decode)
	}
	if shared < 0 {
		return fmt.Errorf("prefill-decode-instances must be >= 0, got %d", shared)
	}
	if encode < 0 {
		return fmt.Errorf("encode-instances must be >= 0, got %d", encode)
	}
	// All zero = disabled, no further checks
	if prefill == 0 && decode == 0 && shared == 0 && encode == 0 {
		return nil
	}
	// Encode requires a decode-capable pool (see GAP-4 design doc D4).
	if encode > 0 && prefill == 0 && decode == 0 && shared == 0 {
		return fmt.Errorf("encode-instances (%d) requires a decode-capable pool: set --prefill-instances, --decode-instances, or --prefill-decode-instances > 0", encode)
	}
	// PD pools (prefill/decode/shared): apply the existing pairing rule only when at
	// least one PD pool is set. If only encode is set, the check above already returned.
	if prefill > 0 || decode > 0 || shared > 0 {
		if shared == 0 && (prefill == 0 || decode == 0) {
			return fmt.Errorf("both --prefill-instances and --decode-instances must be set when PD disaggregation is enabled without shared-role pods (got prefill=%d, decode=%d)", prefill, decode)
		}
	}
	if prefill+decode+shared+encode > total {
		return fmt.Errorf("prefill-instances (%d) + decode-instances (%d) + prefill-decode-instances (%d) + encode-instances (%d) = %d exceeds num-instances (%d)", prefill, decode, shared, encode, prefill+decode+shared+encode, total)
	}
	return nil
}

// BuildPoolMembershipFromIndices constructs a pool membership map from instance indices.
// Uses the same instance ID naming convention as NewClusterSimulator: "instance_N".
// This variant does not require constructed instances, enabling pool membership
// computation before instance construction (needed for per-pool config resolution).
// Caller must validate prefill+decode+shared+encode <= total before calling.
//
// Index layout:
//   [0, prefill)                          — prefill-only
//   [prefill, prefill+decode)             — decode-only
//   [prefill+decode, prefill+decode+shared) — shared-role (PoolRolePrefillDecode)
//   [prefill+decode+shared, prefill+decode+shared+encode) — encode-only (PoolRoleEncode)
func BuildPoolMembershipFromIndices(total, prefill, decode, shared, encode int) map[string]PoolRole {
	if prefill+decode+shared+encode > total {
		panic(fmt.Sprintf("BuildPoolMembershipFromIndices: prefill(%d)+decode(%d)+shared(%d)+encode(%d)=%d exceeds total(%d)",
			prefill, decode, shared, encode, prefill+decode+shared+encode, total))
	}
	membership := make(map[string]PoolRole, prefill+decode+shared+encode)
	for i := 0; i < prefill; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRolePrefill
	}
	for i := prefill; i < prefill+decode; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRoleDecode
	}
	for i := prefill + decode; i < prefill+decode+shared; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRolePrefillDecode
	}
	for i := prefill + decode + shared; i < prefill+decode+shared+encode; i++ {
		membership[fmt.Sprintf("instance_%d", i)] = PoolRoleEncode
	}
	return membership
}

// FilterSnapshotsByPool returns snapshots for instances whose role includes
// the requested role bits (set-membership). A shared-role instance
// (PoolRolePrefillDecode) appears in both the PoolRolePrefill and
// PoolRoleDecode result lists — BLIS analogue of llm-d's bylabel filter
// semantics (issue #1276, Permalink 3).
//
// Order is preserved (stable relative to the input slice).
func FilterSnapshotsByPool(snapshots []sim.RoutingSnapshot, membership map[string]PoolRole, role PoolRole) []sim.RoutingSnapshot {
	filtered := make([]sim.RoutingSnapshot, 0, len(snapshots))
	for _, snap := range snapshots {
		if membership[snap.ID].Has(role) {
			filtered = append(filtered, snap)
		}
	}
	return filtered
}
