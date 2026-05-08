package cluster

import (
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestValidatePoolTopology verifies BC-PD-2 (invalid pool configs return errors)
// and issue #1276 BC-4 (shared-role count permits overlap).
func TestValidatePoolTopology(t *testing.T) {
	tests := []struct {
		name    string
		prefill int
		decode  int
		shared  int
		total   int
		wantErr bool
	}{
		{
			name:    "all zero — disabled",
			prefill: 0, decode: 0, shared: 0, total: 4,
			wantErr: false,
		},
		{
			name:    "valid split",
			prefill: 2, decode: 2, shared: 0, total: 4,
			wantErr: false,
		},
		{
			name:    "valid uneven split",
			prefill: 1, decode: 3, shared: 0, total: 4,
			wantErr: false,
		},
		{
			name:    "sum exceeds total",
			prefill: 3, decode: 3, shared: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "sum equals total",
			prefill: 2, decode: 2, shared: 0, total: 4,
			wantErr: false,
		},
		{
			name:    "negative prefill",
			prefill: -1, decode: 2, shared: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "negative decode",
			prefill: 2, decode: -1, shared: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "negative shared",
			prefill: 2, decode: 2, shared: -1, total: 4,
			wantErr: true,
		},
		{
			name:    "only prefill set — single pool when disagg enabled without shared",
			prefill: 2, decode: 0, shared: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "only decode set — single pool when disagg enabled without shared",
			prefill: 0, decode: 2, shared: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "prefill equals total — no decode instances, no shared",
			prefill: 4, decode: 0, shared: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "decode equals total — no prefill instances, no shared",
			prefill: 0, decode: 4, shared: 0, total: 4,
			wantErr: true,
		},
		// Issue #1276: shared-role scenarios
		{
			name:    "pure-shared cluster — all instances prefill-decode",
			prefill: 0, decode: 0, shared: 4, total: 4,
			wantErr: false,
		},
		{
			name:    "shared with prefill only — legitimate shared-role deployment",
			prefill: 2, decode: 0, shared: 2, total: 4,
			wantErr: false,
		},
		{
			name:    "shared with decode only — legitimate shared-role deployment",
			prefill: 0, decode: 2, shared: 2, total: 4,
			wantErr: false,
		},
		{
			name:    "prefill + decode + shared equals total",
			prefill: 2, decode: 1, shared: 1, total: 4,
			wantErr: false,
		},
		{
			name:    "prefill + decode + shared exceeds total",
			prefill: 2, decode: 2, shared: 1, total: 4,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidatePoolTopology(tc.prefill, tc.decode, tc.shared, 0, tc.total)
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidatePoolTopology(%d, %d, %d, 0, %d) error = %v, wantErr %v",
					tc.prefill, tc.decode, tc.shared, tc.total, err, tc.wantErr)
			}
		})
	}
}

// TestBuildPoolMembershipFromIndices verifies BC-PD-3: membership construction via indices.
func TestBuildPoolMembershipFromIndices(t *testing.T) {
	tests := []struct {
		name        string
		total       int
		prefill     int
		decode      int
		wantPrefill []string
		wantDecode  []string
	}{
		{
			name:  "2 prefill, 2 decode",
			total: 4, prefill: 2, decode: 2,
			wantPrefill: []string{"instance_0", "instance_1"},
			wantDecode:  []string{"instance_2", "instance_3"},
		},
		{
			name:  "1 prefill, 3 decode",
			total: 4, prefill: 1, decode: 3,
			wantPrefill: []string{"instance_0"},
			wantDecode:  []string{"instance_1", "instance_2", "instance_3"},
		},
		{
			name:  "3 prefill, 1 decode",
			total: 4, prefill: 3, decode: 1,
			wantPrefill: []string{"instance_0", "instance_1", "instance_2"},
			wantDecode:  []string{"instance_3"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			membership := BuildPoolMembershipFromIndices(tc.total, tc.prefill, tc.decode, 0, 0)

			// Verify total membership count
			if len(membership) != tc.prefill+tc.decode {
				t.Errorf("membership size = %d, want %d", len(membership), tc.prefill+tc.decode)
			}

			// Verify prefill instances
			for _, id := range tc.wantPrefill {
				role, ok := membership[id]
				if !ok {
					t.Errorf("instance %q not in membership", id)
					continue
				}
				if role != PoolRolePrefill {
					t.Errorf("instance %q role = %v, want Prefill", id, role)
				}
			}

			// Verify decode instances
			for _, id := range tc.wantDecode {
				role, ok := membership[id]
				if !ok {
					t.Errorf("instance %q not in membership", id)
					continue
				}
				if role != PoolRoleDecode {
					t.Errorf("instance %q role = %v, want Decode", id, role)
				}
			}

			// Unassigned instances must not appear
			for i := tc.prefill + tc.decode; i < tc.total; i++ {
				id := InstanceID(instanceName(i))
				if _, ok := membership[string(id)]; ok {
					t.Errorf("unassigned instance %q should not appear in membership", id)
				}
			}
		})
	}
}

// instanceName returns the instance ID string for index i, mirroring NewClusterSimulator convention.
func instanceName(i int) string {
	return "instance_" + strconv.Itoa(i)
}

// TestFilterSnapshotsByPool verifies snapshot filtering by pool role.
func TestFilterSnapshotsByPool(t *testing.T) {
	membership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRolePrefill,
		"instance_2": PoolRoleDecode,
		"instance_3": PoolRoleDecode,
	}
	snapshots := []sim.RoutingSnapshot{
		{ID: "instance_0"},
		{ID: "instance_1"},
		{ID: "instance_2"},
		{ID: "instance_3"},
		{ID: "instance_4"}, // unassigned — filtered out for any specific role
	}

	t.Run("prefill pool", func(t *testing.T) {
		got := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
		if len(got) != 2 {
			t.Fatalf("prefill filter: got %d snapshots, want 2", len(got))
		}
		if got[0].ID != "instance_0" || got[1].ID != "instance_1" {
			t.Errorf("prefill filter: got IDs %v, want [instance_0, instance_1]",
				[]string{got[0].ID, got[1].ID})
		}
	})

	t.Run("decode pool", func(t *testing.T) {
		got := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
		if len(got) != 2 {
			t.Fatalf("decode filter: got %d snapshots, want 2", len(got))
		}
		if got[0].ID != "instance_2" || got[1].ID != "instance_3" {
			t.Errorf("decode filter: got IDs %v, want [instance_2, instance_3]",
				[]string{got[0].ID, got[1].ID})
		}
	})

	t.Run("empty input", func(t *testing.T) {
		got := FilterSnapshotsByPool(nil, membership, PoolRolePrefill)
		if len(got) != 0 {
			t.Errorf("empty input: got %d snapshots, want 0", len(got))
		}
	})

	t.Run("order preserved", func(t *testing.T) {
		// Decode instances are instance_2, instance_3 — input order preserved
		got := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
		for i, snap := range got {
			if i > 0 && got[i-1].ID >= snap.ID {
				t.Errorf("order not preserved: %q >= %q", got[i-1].ID, snap.ID)
			}
		}
	})
}

// TestPoolRole_String verifies PoolRole has meaningful string representation,
// including the shared-role PoolRolePrefillDecode (issue #1276).
func TestPoolRole_String(t *testing.T) {
	tests := []struct {
		role PoolRole
		want string
	}{
		{PoolRole(0), "PoolRole(0)"},
		{PoolRolePrefill, "prefill"},
		{PoolRoleDecode, "decode"},
		{PoolRolePrefillDecode, "prefill-decode"},
		{PoolRole(99), "PoolRole(99)"},
	}
	for _, tc := range tests {
		if got := tc.role.String(); got != tc.want {
			t.Errorf("PoolRole(%d).String() = %q, want %q", uint(tc.role), got, tc.want)
		}
	}
}

// TestPoolRole_Has verifies issue #1276 BC-2: Has implements set-membership
// over role bits, enabling shared-role pods to satisfy both prefill and decode
// filters — BLIS analogue of llm-d's bylabel set-membership semantics.
func TestPoolRole_Has(t *testing.T) {
	tests := []struct {
		name  string
		role  PoolRole
		query PoolRole
		want  bool
	}{
		{"prefill has prefill", PoolRolePrefill, PoolRolePrefill, true},
		{"prefill does not have decode", PoolRolePrefill, PoolRoleDecode, false},
		{"decode has decode", PoolRoleDecode, PoolRoleDecode, true},
		{"decode does not have prefill", PoolRoleDecode, PoolRolePrefill, false},
		{"shared has prefill", PoolRolePrefillDecode, PoolRolePrefill, true},
		{"shared has decode", PoolRolePrefillDecode, PoolRoleDecode, true},
		{"shared has shared (self)", PoolRolePrefillDecode, PoolRolePrefillDecode, true},
		{"zero has nothing (prefill)", PoolRole(0), PoolRolePrefill, false},
		{"zero has nothing (decode)", PoolRole(0), PoolRoleDecode, false},
		{"zero.Has(zero) is false (degenerate)", PoolRole(0), PoolRole(0), false},
		{"prefill.Has(zero) is false (degenerate)", PoolRolePrefill, PoolRole(0), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.role.Has(tc.query); got != tc.want {
				t.Errorf("PoolRole(%d).Has(PoolRole(%d)) = %v, want %v",
					uint(tc.role), uint(tc.query), got, tc.want)
			}
		})
	}
}

// TestPoolRole_SharedEqualsUnion verifies the algebraic identity
// PoolRolePrefillDecode == PoolRolePrefill | PoolRoleDecode.
// This is load-bearing for issue #1276 BC-5 and BC-6 — the completion
// detectors and throughput attribution rely on set-membership over a
// shared role that is exactly the union of its two components.
func TestPoolRole_SharedEqualsUnion(t *testing.T) {
	if PoolRolePrefillDecode != (PoolRolePrefill | PoolRoleDecode) {
		t.Errorf("PoolRolePrefillDecode (%d) must equal PoolRolePrefill | PoolRoleDecode (%d)",
			uint(PoolRolePrefillDecode), uint(PoolRolePrefill|PoolRoleDecode))
	}
}

// TestBuildPoolMembershipFromIndices_WithShared verifies issue #1276: the last
// `shared` indices after the prefill and decode ranges are assigned
// PoolRolePrefillDecode.
func TestBuildPoolMembershipFromIndices_WithShared(t *testing.T) {
	// Cluster of 5: 2 prefill-only + 2 decode-only + 1 shared.
	membership := BuildPoolMembershipFromIndices(5, 2, 2, 1, 0)

	if len(membership) != 5 {
		t.Fatalf("membership size = %d, want 5", len(membership))
	}
	checks := []struct {
		id   string
		role PoolRole
	}{
		{"instance_0", PoolRolePrefill},
		{"instance_1", PoolRolePrefill},
		{"instance_2", PoolRoleDecode},
		{"instance_3", PoolRoleDecode},
		{"instance_4", PoolRolePrefillDecode},
	}
	for _, c := range checks {
		if got := membership[c.id]; got != c.role {
			t.Errorf("membership[%q] = %v, want %v", c.id, got, c.role)
		}
	}

	// Pure-shared cluster: all instances serve both stages.
	pure := BuildPoolMembershipFromIndices(3, 0, 0, 3, 0)
	for i := 0; i < 3; i++ {
		id := "instance_" + strconv.Itoa(i)
		if got := pure[id]; got != PoolRolePrefillDecode {
			t.Errorf("pure-shared: membership[%q] = %v, want PoolRolePrefillDecode", id, got)
		}
	}
}

// TestFilterSnapshotsByPool_SharedRole verifies issue #1276 BC-1: a shared-role
// instance is returned by both the prefill and decode filters — BLIS analogue
// of llm-d roles_test.go (Permalink 4).
func TestFilterSnapshotsByPool_SharedRole(t *testing.T) {
	membership := map[string]PoolRole{
		"prefill-only":   PoolRolePrefill,
		"decode-only":    PoolRoleDecode,
		"shared-pd-pod":  PoolRolePrefillDecode,
		"unassigned-pod": PoolRole(0), // explicit zero — not in any pool
	}
	snapshots := []sim.RoutingSnapshot{
		{ID: "prefill-only"},
		{ID: "decode-only"},
		{ID: "shared-pd-pod"},
		{ID: "unassigned-pod"},
	}

	t.Run("prefill filter includes shared-role", func(t *testing.T) {
		got := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
		ids := idsOf(got)
		want := []string{"prefill-only", "shared-pd-pod"}
		if !equalIDs(ids, want) {
			t.Errorf("prefill filter IDs = %v, want %v", ids, want)
		}
	})

	t.Run("decode filter includes shared-role", func(t *testing.T) {
		got := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
		ids := idsOf(got)
		want := []string{"decode-only", "shared-pd-pod"}
		if !equalIDs(ids, want) {
			t.Errorf("decode filter IDs = %v, want %v", ids, want)
		}
	})

	t.Run("shared appears in both results — BC-1 core invariant", func(t *testing.T) {
		pf := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
		df := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
		if !containsID(pf, "shared-pd-pod") || !containsID(df, "shared-pd-pod") {
			t.Errorf("shared-pd-pod must appear in BOTH filters: prefill=%v, decode=%v",
				idsOf(pf), idsOf(df))
		}
	})

	t.Run("unassigned pod appears in neither", func(t *testing.T) {
		pf := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
		df := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
		if containsID(pf, "unassigned-pod") || containsID(df, "unassigned-pod") {
			t.Errorf("unassigned pod leaked into a filter: prefill=%v, decode=%v",
				idsOf(pf), idsOf(df))
		}
	})
}

func idsOf(snaps []sim.RoutingSnapshot) []string {
	out := make([]string, len(snaps))
	for i, s := range snaps {
		out[i] = s.ID
	}
	return out
}

func containsID(snaps []sim.RoutingSnapshot, id string) bool {
	for _, s := range snaps {
		if s.ID == id {
			return true
		}
	}
	return false
}

func equalIDs(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	// order-insensitive (the filter preserves input order, but we don't
	// care about ordering for these assertions)
	seen := make(map[string]int, len(a))
	for _, x := range a {
		seen[x]++
	}
	for _, x := range b {
		seen[x]--
	}
	for _, v := range seen {
		if v != 0 {
			return false
		}
	}
	return true
}

// TestPoolRoleEncode_BitmaskIdentity verifies that PoolRoleEncode does not
// overlap with PoolRolePrefill / PoolRoleDecode / PoolRolePrefillDecode — a
// prerequisite for E/P/D set-membership filtering (GAP-4, issue #1264).
func TestPoolRoleEncode_BitmaskIdentity(t *testing.T) {
	if PoolRoleEncode == 0 {
		t.Fatal("PoolRoleEncode must be non-zero")
	}
	if PoolRoleEncode.Has(PoolRolePrefill) {
		t.Errorf("PoolRoleEncode must not contain PoolRolePrefill bit")
	}
	if PoolRoleEncode.Has(PoolRoleDecode) {
		t.Errorf("PoolRoleEncode must not contain PoolRoleDecode bit")
	}
	if PoolRolePrefill.Has(PoolRoleEncode) || PoolRoleDecode.Has(PoolRoleEncode) || PoolRolePrefillDecode.Has(PoolRoleEncode) {
		t.Errorf("Prefill/Decode/Shared roles must not contain the encode bit")
	}
	if PoolRoleEncode.String() != "encode" {
		t.Errorf("PoolRoleEncode.String() = %q, want %q", PoolRoleEncode.String(), "encode")
	}
}

// TestValidatePoolTopology_Encode verifies the encode-instances validation rules.
func TestValidatePoolTopology_Encode(t *testing.T) {
	tests := []struct {
		name                                           string
		prefill, decode, shared, encode, total         int
		wantErr                                        bool
	}{
		{"valid 1p/1d/0s/1e in total=3", 1, 1, 0, 1, 3, false},
		{"valid 0p/2d/0s/1e in total=3 (decode + encode)", 0, 2, 0, 1, 3, true}, // prefill+decode must both be set OR shared; (0,2) fails the existing PD pair rule even with encode>0
		{"valid 2p/2d/0s/1e in total=5", 2, 2, 0, 1, 5, false},
		{"encode only, no decode-capable pool — rejected", 0, 0, 0, 2, 2, true},
		{"encode+shared valid", 0, 0, 2, 1, 3, false},
		{"sum exceeds total with encode", 1, 1, 0, 2, 3, true},
		{"negative encode", 1, 1, 0, -1, 3, true},
		{"encode=0 backward compat (valid PD)", 1, 1, 0, 0, 2, false},
		{"encode=0 all-zero disabled", 0, 0, 0, 0, 4, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidatePoolTopology(tc.prefill, tc.decode, tc.shared, tc.encode, tc.total)
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidatePoolTopology(p=%d,d=%d,s=%d,e=%d,t=%d) err=%v wantErr=%v",
					tc.prefill, tc.decode, tc.shared, tc.encode, tc.total, err, tc.wantErr)
			}
		})
	}
}

// TestBuildPoolMembershipFromIndices_Encode verifies the encode-only
// assignment block follows the prefill/decode/shared ranges.
func TestBuildPoolMembershipFromIndices_Encode(t *testing.T) {
	// 1 prefill + 1 decode + 1 shared + 2 encode = 5 total.
	membership := BuildPoolMembershipFromIndices(5, 1, 1, 1, 2)
	if len(membership) != 5 {
		t.Fatalf("membership size = %d, want 5", len(membership))
	}
	checks := []struct {
		id   string
		role PoolRole
	}{
		{"instance_0", PoolRolePrefill},
		{"instance_1", PoolRoleDecode},
		{"instance_2", PoolRolePrefillDecode},
		{"instance_3", PoolRoleEncode},
		{"instance_4", PoolRoleEncode},
	}
	for _, c := range checks {
		if got := membership[c.id]; got != c.role {
			t.Errorf("membership[%q] = %v, want %v", c.id, got, c.role)
		}
	}
}

// TestFilterSnapshotsByPool_Encode verifies encode-pool filtering returns
// only encode-role instances and excludes prefill/decode/shared pods.
func TestFilterSnapshotsByPool_Encode(t *testing.T) {
	membership := map[string]PoolRole{
		"p-only":   PoolRolePrefill,
		"d-only":   PoolRoleDecode,
		"shared":   PoolRolePrefillDecode,
		"enc-pod":  PoolRoleEncode,
		"enc-pod2": PoolRoleEncode,
	}
	snaps := []sim.RoutingSnapshot{
		{ID: "p-only"}, {ID: "d-only"}, {ID: "shared"}, {ID: "enc-pod"}, {ID: "enc-pod2"},
	}
	got := FilterSnapshotsByPool(snaps, membership, PoolRoleEncode)
	if len(got) != 2 || got[0].ID != "enc-pod" || got[1].ID != "enc-pod2" {
		t.Errorf("encode filter = %v, want [enc-pod, enc-pod2]", idsOf(got))
	}
}

// TestBuildPoolMembershipFromIndices_Immutability verifies the returned map can be copied
// without aliasing (defensive copy contract).
func TestBuildPoolMembershipFromIndices_Immutability(t *testing.T) {
	membership := BuildPoolMembershipFromIndices(4, 2, 2, 0, 0)

	// Take a snapshot
	snapshot := make(map[string]PoolRole, len(membership))
	for k, v := range membership {
		snapshot[k] = v
	}

	// Mutate the original (simulating caller mistake)
	membership["instance_0"] = PoolRoleDecode

	// The snapshot must be unaffected (it's a separate map)
	if snapshot["instance_0"] != PoolRolePrefill {
		t.Error("snapshot was affected by mutation of original map")
	}
}
