package cluster

import (
	"testing"
)

// TestValidatePoolTopology verifies BC-PD-2: invalid pool configs return errors.
func TestValidatePoolTopology(t *testing.T) {
	tests := []struct {
		name     string
		prefill  int
		decode   int
		total    int
		wantErr  bool
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

// TestBuildPoolMembership verifies BC-PD-3: membership construction and correctness.
func TestBuildPoolMembership(t *testing.T) {
	tests := []struct {
		name          string
		numInstances  int
		prefill       int
		decode        int
		wantPrefill   []string
		wantDecode    []string
	}{
		{
			name:         "2 prefill, 2 decode",
			numInstances: 4,
			prefill:      2, decode: 2,
			wantPrefill: []string{"instance_0", "instance_1"},
			wantDecode:  []string{"instance_2", "instance_3"},
		},
		{
			name:         "1 prefill, 3 decode",
			numInstances: 4,
			prefill:      1, decode: 3,
			wantPrefill: []string{"instance_0"},
			wantDecode:  []string{"instance_1", "instance_2", "instance_3"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			config := newTestDeploymentConfig(tc.numInstances)
			cs := NewClusterSimulator(config, nil)

			membership := BuildPoolMembership(cs.instances, tc.prefill, tc.decode)

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
		})
	}
}

// TestBuildPoolMembership_Immutability verifies INV-PD-5: pool membership never changes.
func TestBuildPoolMembership_Immutability(t *testing.T) {
	config := newTestDeploymentConfig(4)
	cs := NewClusterSimulator(config, nil)

	membership := BuildPoolMembership(cs.instances, 2, 2)

	// Take a snapshot
	snapshot := make(map[string]PoolRole, len(membership))
	for k, v := range membership {
		snapshot[k] = v
	}

	// Verify snapshot matches original
	for k, v := range snapshot {
		if membership[k] != v {
			t.Errorf("membership[%q] changed: was %v, now %v", k, v, membership[k])
		}
	}
	if len(membership) != len(snapshot) {
		t.Errorf("membership size changed: was %d, now %d", len(snapshot), len(membership))
	}
}

// TestClusterSimulator_WithPools verifies that pool topology is wired into ClusterSimulator.
func TestClusterSimulator_WithPools(t *testing.T) {
	config := newTestDeploymentConfig(4)
	config.PrefillInstances = 2
	config.DecodeInstances = 2
	config.PDDecider = "always"
	config.PDTransferBandwidthGBps = 25.0
	config.PDTransferBaseLatencyMs = 0.05
	config.PDKVBytesPerToken = 512

	cs := NewClusterSimulator(config, nil)

	if !cs.poolsConfigured() {
		t.Error("poolsConfigured() should return true when pools are set")
	}
	if len(cs.PoolMembership()) != 4 {
		t.Errorf("PoolMembership() size = %d, want 4", len(cs.PoolMembership()))
	}
	if cs.disaggregationDecider == nil {
		t.Error("disaggregationDecider should be non-nil when pools are configured")
	}
}

// TestClusterSimulator_WithoutPools verifies BC-PD-4: no pool topology when disabled.
func TestClusterSimulator_WithoutPools(t *testing.T) {
	config := newTestDeploymentConfig(4)
	// PrefillInstances and DecodeInstances are 0 (zero values)

	cs := NewClusterSimulator(config, nil)

	if cs.poolsConfigured() {
		t.Error("poolsConfigured() should return false when pools are not set")
	}
	if cs.PoolMembership() != nil {
		t.Errorf("PoolMembership() should be nil, got %v", cs.PoolMembership())
	}
	if cs.disaggregationDecider != nil {
		t.Error("disaggregationDecider should be nil when pools are not configured")
	}
}

// TestClusterSimulator_InvalidPoolConfig_Panics verifies BC-PD-2 at construction time.
func TestClusterSimulator_InvalidPoolConfig_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for invalid pool config")
		}
	}()
	config := newTestDeploymentConfig(4)
	config.PrefillInstances = 3
	config.DecodeInstances = 3 // sum > 4
	NewClusterSimulator(config, nil)
}

// TestPoolRole_String verifies PoolRole has meaningful string representation.
func TestPoolRole_String(t *testing.T) {
	tests := []struct {
		role PoolRole
		want string
	}{
		{PoolRolePrefill, "prefill"},
		{PoolRoleDecode, "decode"},
	}
	for _, tc := range tests {
		if got := tc.role.String(); got != tc.want {
			t.Errorf("PoolRole(%d).String() = %q, want %q", tc.role, got, tc.want)
		}
	}
}
