package cluster

import (
	"fmt"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestValidatePoolTopology verifies pool topology validation rules (BC-PD-2, BC-PD-11, BC-PD-12, BC-PD-13)
func TestValidatePoolTopology(t *testing.T) {
	tests := []struct {
		name    string
		prefill int
		decode  int
		total   int
		wantErr bool
		errMsg  string
	}{
		{
			name:    "both zero (disabled)",
			prefill: 0,
			decode:  0,
			total:   4,
			wantErr: false,
		},
		{
			name:    "valid split",
			prefill: 2,
			decode:  2,
			total:   4,
			wantErr: false,
		},
		{
			name:    "valid unequal split",
			prefill: 1,
			decode:  3,
			total:   4,
			wantErr: false,
		},
		{
			name:    "negative prefill",
			prefill: -1,
			decode:  2,
			total:   4,
			wantErr: true,
			errMsg:  "prefill-instances must be >= 0",
		},
		{
			name:    "negative decode",
			prefill: 2,
			decode:  -1,
			total:   4,
			wantErr: true,
			errMsg:  "decode-instances must be >= 0",
		},
		{
			name:    "only prefill set",
			prefill: 2,
			decode:  0,
			total:   4,
			wantErr: true,
			errMsg:  "both --prefill-instances and --decode-instances must be set",
		},
		{
			name:    "only decode set",
			prefill: 0,
			decode:  2,
			total:   4,
			wantErr: true,
			errMsg:  "both --prefill-instances and --decode-instances must be set",
		},
		{
			name:    "sum exceeds total",
			prefill: 3,
			decode:  3,
			total:   4,
			wantErr: true,
			errMsg:  "exceeds num-instances",
		},
		{
			name:    "sum equals total",
			prefill: 2,
			decode:  2,
			total:   4,
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ValidatePoolTopology(tt.prefill, tt.decode, tt.total)
			if tt.wantErr {
				if err == nil {
					t.Errorf("ValidatePoolTopology() expected error containing %q, got nil", tt.errMsg)
				} else if tt.errMsg != "" && !contains(err.Error(), tt.errMsg) {
					t.Errorf("ValidatePoolTopology() error = %v, want substring %q", err, tt.errMsg)
				}
			} else {
				if err != nil {
					t.Errorf("ValidatePoolTopology() unexpected error = %v", err)
				}
			}
		})
	}
}

// TestBuildPoolMembership verifies pool membership construction (BC-PD-3)
func TestBuildPoolMembership(t *testing.T) {
	tests := []struct {
		name     string
		prefill  int
		decode   int
		wantSize int
	}{
		{
			name:     "equal split",
			prefill:  2,
			decode:   2,
			wantSize: 4,
		},
		{
			name:     "prefill heavy",
			prefill:  3,
			decode:   1,
			wantSize: 4,
		},
		{
			name:     "decode heavy",
			prefill:  1,
			decode:   3,
			wantSize: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create mock instances
			total := tt.prefill + tt.decode
			instances := make([]*InstanceSimulator, total)
			for i := 0; i < total; i++ {
				cfg := sim.SimConfig{
					Horizon:             100.0,
					Seed:                42,
					KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
					BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
					LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
					ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
				}
				inst := NewInstanceSimulator(InstanceID(fmt.Sprintf("instance_%d", i)), cfg)
				instances[i] = inst
			}

			membership := BuildPoolMembership(instances, tt.prefill, tt.decode)

			// Verify size
			if len(membership) != tt.wantSize {
				t.Errorf("BuildPoolMembership() size = %d, want %d", len(membership), tt.wantSize)
			}

			// Verify prefill instances
			for i := 0; i < tt.prefill; i++ {
				id := string(instances[i].ID())
				if role, ok := membership[id]; !ok {
					t.Errorf("BuildPoolMembership() missing instance %s", id)
				} else if role != PoolRolePrefill {
					t.Errorf("BuildPoolMembership() instance %s role = %v, want PoolRolePrefill", id, role)
				}
			}

			// Verify decode instances
			for i := tt.prefill; i < tt.prefill+tt.decode; i++ {
				id := string(instances[i].ID())
				if role, ok := membership[id]; !ok {
					t.Errorf("BuildPoolMembership() missing instance %s", id)
				} else if role != PoolRoleDecode {
					t.Errorf("BuildPoolMembership() instance %s role = %v, want PoolRoleDecode", id, role)
				}
			}
		})
	}
}

// TestBuildPoolMembership_Immutability verifies pool membership is immutable (BC-PD-9, INV-PD-5)
func TestBuildPoolMembership_Immutability(t *testing.T) {
	// Create mock instances
	instances := make([]*InstanceSimulator, 4)
	for i := 0; i < 4; i++ {
		cfg := sim.SimConfig{
			Horizon:             100.0,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, "blackbox", 0),
		}
		inst := NewInstanceSimulator(InstanceID(fmt.Sprintf("instance_%d", i)), cfg)
		instances[i] = inst
	}

	membership := BuildPoolMembership(instances, 2, 2)

	// Attempt to mutate (should not affect original)
	membership[string(instances[0].ID())] = PoolRoleDecode

	// Rebuild and verify original assignment is preserved
	membership2 := BuildPoolMembership(instances, 2, 2)
	if membership2[string(instances[0].ID())] != PoolRolePrefill {
		t.Errorf("BuildPoolMembership() not immutable: instance 0 role changed")
	}
}

// TestBuildPoolMembershipFromIndices verifies index-based membership construction
func TestBuildPoolMembershipFromIndices(t *testing.T) {
	tests := []struct {
		name     string
		total    int
		prefill  int
		decode   int
		wantSize int
	}{
		{
			name:     "equal split",
			total:    4,
			prefill:  2,
			decode:   2,
			wantSize: 4,
		},
		{
			name:     "prefill heavy",
			total:    4,
			prefill:  3,
			decode:   1,
			wantSize: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			membership := BuildPoolMembershipFromIndices(tt.total, tt.prefill, tt.decode)

			// Verify size
			if len(membership) != tt.wantSize {
				t.Errorf("BuildPoolMembershipFromIndices() size = %d, want %d", len(membership), tt.wantSize)
			}

			// Verify prefill instances
			for i := 0; i < tt.prefill; i++ {
				id := instanceIDFromIndex(i)
				if role, ok := membership[id]; !ok {
					t.Errorf("BuildPoolMembershipFromIndices() missing instance %s", id)
				} else if role != PoolRolePrefill {
					t.Errorf("BuildPoolMembershipFromIndices() instance %s role = %v, want PoolRolePrefill", id, role)
				}
			}

			// Verify decode instances
			for i := tt.prefill; i < tt.prefill+tt.decode; i++ {
				id := instanceIDFromIndex(i)
				if role, ok := membership[id]; !ok {
					t.Errorf("BuildPoolMembershipFromIndices() missing instance %s", id)
				} else if role != PoolRoleDecode {
					t.Errorf("BuildPoolMembershipFromIndices() instance %s role = %v, want PoolRoleDecode", id, role)
				}
			}
		})
	}
}

// TestFilterSnapshotsByPoolRole verifies snapshot filtering by pool role
func TestFilterSnapshotsByPoolRole(t *testing.T) {
	// Create mock snapshots
	snapshots := []sim.RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 1},
		{ID: "instance_1", QueueDepth: 2},
		{ID: "instance_2", QueueDepth: 3},
		{ID: "instance_3", QueueDepth: 4},
	}

	membership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRolePrefill,
		"instance_2": PoolRoleDecode,
		"instance_3": PoolRoleDecode,
	}

	// Filter prefill
	prefillSnaps := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
	if len(prefillSnaps) != 2 {
		t.Errorf("FilterSnapshotsByPool(PoolRolePrefill) len = %d, want 2", len(prefillSnaps))
	}
	if prefillSnaps[0].ID != "instance_0" || prefillSnaps[1].ID != "instance_1" {
		t.Errorf("FilterSnapshotsByPool(PoolRolePrefill) IDs = %v, want [instance_0, instance_1]", []string{prefillSnaps[0].ID, prefillSnaps[1].ID})
	}

	// Filter decode
	decodeSnaps := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
	if len(decodeSnaps) != 2 {
		t.Errorf("FilterSnapshotsByPool(PoolRoleDecode) len = %d, want 2", len(decodeSnaps))
	}
	if decodeSnaps[0].ID != "instance_2" || decodeSnaps[1].ID != "instance_3" {
		t.Errorf("FilterSnapshotsByPool(PoolRoleDecode) IDs = %v, want [instance_2, instance_3]", []string{decodeSnaps[0].ID, decodeSnaps[1].ID})
	}
}

// TestPoolRoleString verifies String() method
func TestPoolRoleString(t *testing.T) {
	tests := []struct {
		role PoolRole
		want string
	}{
		{PoolRolePrefill, "prefill"},
		{PoolRoleDecode, "decode"},
		{PoolRole(99), "PoolRole(99)"},
	}

	for _, tt := range tests {
		t.Run(tt.want, func(t *testing.T) {
			if got := tt.role.String(); got != tt.want {
				t.Errorf("PoolRole.String() = %v, want %v", got, tt.want)
			}
		})
	}
}

// Helper functions

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(substr) == 0 || (len(s) > 0 && len(substr) > 0 && findSubstring(s, substr)))
}

func findSubstring(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

func instanceIDFromIndex(i int) string {
	return fmt.Sprintf("instance_%d", i)
}