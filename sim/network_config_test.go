package sim

import (
	"math"
	"testing"
)

// TestNewNetworkConfig_FieldEquivalence verifies the canonical constructor sets
// every field (R4) and that the zero value is inert (IsActive false), so an
// unset network config leaves the latency model byte-identical to a pre-feature
// build (INV-6).
func TestNewNetworkConfig_FieldEquivalence(t *testing.T) {
	nc := NewNetworkConfig(8, 50.0, 0.002)
	if nc.GPUsPerNode != 8 {
		t.Errorf("GPUsPerNode = %d, want 8", nc.GPUsPerNode)
	}
	if nc.InterNodeBandwidthGBps != 50.0 {
		t.Errorf("InterNodeBandwidthGBps = %v, want 50.0", nc.InterNodeBandwidthGBps)
	}
	if nc.InterNodeLatencyMs != 0.002 {
		t.Errorf("InterNodeLatencyMs = %v, want 0.002", nc.InterNodeLatencyMs)
	}

	// Zero value is inert.
	var zero NetworkConfig
	if zero.IsActive() {
		t.Error("zero-value NetworkConfig must be inert (IsActive false)")
	}
	if NewNetworkConfig(0, 0, 0).IsActive() {
		t.Error("NewNetworkConfig(0,0,0) must be inert")
	}
	if !NewNetworkConfig(8, 50.0, 0).IsActive() {
		t.Error("NewNetworkConfig(8,50,0) must be active (GPUsPerNode>0)")
	}
}

// TestNetworkConfig_Validate covers the R3 numeric guards and the R11 divisor
// guard: an active config (GPUsPerNode>0) requires a positive finite bandwidth,
// fabric knobs set without GPUsPerNode is a misconfiguration (R1), and all fields
// must be finite and non-negative.
func TestNetworkConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		nc      NetworkConfig
		wantErr bool
	}{
		{"inert zero value", NetworkConfig{}, false},
		{"active well-formed", NewNetworkConfig(8, 50.0, 0.002), false},
		{"active no latency ok", NewNetworkConfig(8, 50.0, 0), false},
		{"negative gpus-per-node", NetworkConfig{GPUsPerNode: -1}, true},
		{"active but zero bandwidth", NetworkConfig{GPUsPerNode: 8, InterNodeBandwidthGBps: 0}, true},
		{"active but negative bandwidth", NetworkConfig{GPUsPerNode: 8, InterNodeBandwidthGBps: -1}, true},
		{"orphan bandwidth (no gpus-per-node)", NetworkConfig{InterNodeBandwidthGBps: 50.0}, true},
		{"orphan latency (no gpus-per-node)", NetworkConfig{InterNodeLatencyMs: 0.002}, true},
		{"NaN bandwidth", NetworkConfig{GPUsPerNode: 8, InterNodeBandwidthGBps: math.NaN()}, true},
		{"Inf bandwidth", NetworkConfig{GPUsPerNode: 8, InterNodeBandwidthGBps: math.Inf(1)}, true},
		{"NaN latency", NetworkConfig{GPUsPerNode: 8, InterNodeBandwidthGBps: 50.0, InterNodeLatencyMs: math.NaN()}, true},
		{"negative latency", NetworkConfig{GPUsPerNode: 8, InterNodeBandwidthGBps: 50.0, InterNodeLatencyMs: -0.001}, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.nc.Validate()
			if tt.wantErr && err == nil {
				t.Errorf("Validate() = nil, want error")
			}
			if !tt.wantErr && err != nil {
				t.Errorf("Validate() = %v, want nil", err)
			}
		})
	}
}
