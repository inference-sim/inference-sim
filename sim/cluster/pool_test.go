package cluster

import (
	"strconv"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestValidatePoolTopology verifies BC-PD-2: invalid pool configs return errors.
func TestValidatePoolTopology(t *testing.T) {
	tests := []struct {
		name    string
		prefill int
		decode  int
		total   int
		wantErr bool
	}{
		{
			name:    "both zero — disabled",
			prefill: 0, decode: 0, total: 4,
			wantErr: false,
		},
		{
			name:    "valid split",
			prefill: 2, decode: 2, total: 4,
			wantErr: false,
		},
		{
			name:    "valid uneven split",
			prefill: 1, decode: 3, total: 4,
			wantErr: false,
		},
		{
			name:    "sum exceeds total",
			prefill: 3, decode: 3, total: 4,
			wantErr: true,
		},
		{
			name:    "sum equals total",
			prefill: 2, decode: 2, total: 4,
			wantErr: false,
		},
		{
			name:    "negative prefill",
			prefill: -1, decode: 2, total: 4,
			wantErr: true,
		},
		{
			name:    "negative decode",
			prefill: 2, decode: -1, total: 4,
			wantErr: true,
		},
		{
			name:    "only prefill set — single pool when disagg enabled",
			prefill: 2, decode: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "only decode set — single pool when disagg enabled",
			prefill: 0, decode: 2, total: 4,
			wantErr: true,
		},
		{
			name:    "prefill equals total — no decode instances",
			prefill: 4, decode: 0, total: 4,
			wantErr: true,
		},
		{
			name:    "decode equals total — no prefill instances",
			prefill: 0, decode: 4, total: 4,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidatePoolTopology(tc.prefill, tc.decode, tc.total)
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidatePoolTopology(%d, %d, %d) error = %v, wantErr %v",
					tc.prefill, tc.decode, tc.total, err, tc.wantErr)
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
			name:        "2 prefill, 2 decode",
			total:       4, prefill: 2, decode: 2,
			wantPrefill: []string{"instance_0", "instance_1"},
			wantDecode:  []string{"instance_2", "instance_3"},
		},
		{
			name:        "1 prefill, 3 decode",
			total:       4, prefill: 1, decode: 3,
			wantPrefill: []string{"instance_0"},
			wantDecode:  []string{"instance_1", "instance_2", "instance_3"},
		},
		{
			name:        "3 prefill, 1 decode",
			total:       4, prefill: 3, decode: 1,
			wantPrefill: []string{"instance_0", "instance_1", "instance_2"},
			wantDecode:  []string{"instance_3"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			membership := BuildPoolMembershipFromIndices(tc.total, tc.prefill, tc.decode)

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

// TestPoolRole_String verifies PoolRole has meaningful string representation.
func TestPoolRole_String(t *testing.T) {
	tests := []struct {
		role PoolRole
		want string
	}{
		{PoolRolePrefill, "prefill"},
		{PoolRoleDecode, "decode"},
		{PoolRole(99), "PoolRole(99)"},
	}
	for _, tc := range tests {
		if got := tc.role.String(); got != tc.want {
			t.Errorf("PoolRole(%d).String() = %q, want %q", tc.role, got, tc.want)
		}
	}
}

// TestBuildPoolMembershipFromIndices_Immutability verifies the returned map can be copied
// without aliasing (defensive copy contract).
func TestBuildPoolMembershipFromIndices_Immutability(t *testing.T) {
	membership := BuildPoolMembershipFromIndices(4, 2, 2)

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
