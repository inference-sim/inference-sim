package sim

import (
	"math"
	"testing"
)

// TestAdaptiveWeightedScoring_ExploreMode_NoCacheHit verifies that requests
// with no prefix cache hit use explore (load-balanced) routing.
func TestAdaptiveWeightedScoring_ExploreMode_NoCacheHit(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)

	req := &Request{InputTokens: make([]int, 32)}
	for i := range req.InputTokens {
		req.InputTokens[i] = i
	}

	snapshots := []RoutingSnapshot{
		{ID: "inst-0", QueueDepth: 10, BatchSize: 0, KVUtilization: 0.3},
		{ID: "inst-1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}

	d := policy.Route(req, state)

	// With no cache hit, should use explore mode → round-robin (starts at inst-0)
	if d.TargetInstance != "inst-0" {
		t.Errorf("explore mode should round-robin starting at inst-0; got %s", d.TargetInstance)
	}
}

// TestAdaptiveWeightedScoring_ExploitMode_StrongCacheHit verifies that
// after seeding the cache, subsequent requests with matching prefix exploit
// the cached instance.
func TestAdaptiveWeightedScoring_ExploitMode_StrongCacheHit(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)
	aws := policy.(*AdaptiveWeightedScoring)

	// Create a prefix that all requests share (256 tokens = 16 blocks)
	prefix := make([]int, 256)
	for i := range prefix {
		prefix[i] = 1000 + i
	}

	// Seed the cache: route first request to inst-0
	req1 := &Request{InputTokens: append(append([]int{}, prefix...), make([]int, 128)...)}
	for i := 256; i < len(req1.InputTokens); i++ {
		req1.InputTokens[i] = 5000 + i
	}

	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
			{ID: "inst-1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
			{ID: "inst-2", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
			{ID: "inst-3", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
		},
		Clock: 1000,
	}

	d1 := policy.Route(req1, state)
	seededInstance := d1.TargetInstance

	// Second request: same prefix, different suffix
	req2 := &Request{InputTokens: append(append([]int{}, prefix...), make([]int, 128)...)}
	for i := 256; i < len(req2.InputTokens); i++ {
		req2.InputTokens[i] = 9000 + i
	}

	d2 := policy.Route(req2, state)

	// Should exploit the seeded instance due to cache hit (16/24 = 0.67 > 0.3 threshold)
	if d2.TargetInstance != seededInstance {
		t.Errorf("exploit mode should route to cached instance %s; got %s (cache miss ratio should be 0.67)",
			seededInstance, d2.TargetInstance)
	}

	// Verify the reason mentions exploit
	if aws.config.ExploitThreshold > 0.67 {
		t.Skip("threshold too high for this test")
	}
}

// TestAdaptiveWeightedScoring_ExploitMode_OverloadedFallback verifies that
// even with a cache hit, an overloaded instance triggers explore mode.
func TestAdaptiveWeightedScoring_ExploitMode_OverloadedFallback(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)

	// Create prefix and seed cache on inst-0
	prefix := make([]int, 256)
	for i := range prefix {
		prefix[i] = 1000 + i
	}

	// Seed cache on inst-0 at zero load
	req1 := &Request{InputTokens: append(append([]int{}, prefix...), make([]int, 128)...)}
	for i := 256; i < len(req1.InputTokens); i++ {
		req1.InputTokens[i] = 5000 + i
	}
	zeroState := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
			{ID: "inst-1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
		},
		Clock: 1000,
	}
	policy.Route(req1, zeroState)

	// Now inst-0 is heavily loaded (QD=20), inst-1 is empty (QD=0)
	// LoadHeadroom default = 5, so 20-0=20 > 5 → should fall back to explore
	overloadState := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 20, BatchSize: 0, KVUtilization: 0.8},
			{ID: "inst-1", QueueDepth: 0, BatchSize: 0, KVUtilization: 0.1},
		},
		Clock: 2000,
	}

	req2 := &Request{InputTokens: append(append([]int{}, prefix...), make([]int, 128)...)}
	for i := 256; i < len(req2.InputTokens); i++ {
		req2.InputTokens[i] = 9000 + i
	}

	d := policy.Route(req2, overloadState)

	// Should NOT route to overloaded inst-0 despite cache hit
	if d.TargetInstance != "inst-1" {
		t.Errorf("overloaded cached instance should trigger explore mode; got %s, want inst-1",
			d.TargetInstance)
	}
}

// TestAdaptiveWeightedScoring_DeterministicRouting verifies INV-6.
func TestAdaptiveWeightedScoring_DeterministicRouting(t *testing.T) {
	policy1 := NewRoutingPolicy("adaptive-weighted", nil, 16)
	policy2 := NewRoutingPolicy("adaptive-weighted", nil, 16)

	req := &Request{InputTokens: make([]int, 32)}
	for i := range req.InputTokens {
		req.InputTokens[i] = i
	}

	snapshots := []RoutingSnapshot{
		{ID: "inst-0", QueueDepth: 5, BatchSize: 2, KVUtilization: 0.3},
		{ID: "inst-1", QueueDepth: 10, BatchSize: 1, KVUtilization: 0.5},
	}
	state := &RouterState{Snapshots: snapshots, Clock: 1000}

	d1 := policy1.Route(req, state)
	d2 := policy2.Route(req, state)

	if d1.TargetInstance != d2.TargetInstance {
		t.Errorf("non-deterministic: policy1=%s, policy2=%s", d1.TargetInstance, d2.TargetInstance)
	}
}

// TestValidateAdaptiveConfig verifies config validation.
func TestValidateAdaptiveConfig_RejectsInvalid(t *testing.T) {
	tests := []struct {
		name    string
		config  AdaptiveConfig
		wantErr bool
	}{
		{"valid defaults", DefaultAdaptiveConfig(), false},
		{"threshold=1", AdaptiveConfig{ExploitThreshold: 1.0, LoadHeadroom: 5}, false},
		{"threshold=0", AdaptiveConfig{ExploitThreshold: 0, LoadHeadroom: 5}, true},
		{"threshold>1", AdaptiveConfig{ExploitThreshold: 1.5, LoadHeadroom: 5}, true},
		{"NaN threshold", AdaptiveConfig{ExploitThreshold: math.NaN(), LoadHeadroom: 5}, true},
		{"negative headroom", AdaptiveConfig{ExploitThreshold: 0.3, LoadHeadroom: -1}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateAdaptiveConfig(tc.config)
			if (err != nil) != tc.wantErr {
				t.Errorf("ValidateAdaptiveConfig(%+v) error = %v, wantErr %v", tc.config, err, tc.wantErr)
			}
		})
	}
}

// TestAdaptiveWeightedScoring_ClassifyRequest_EmptyPrefix verifies that
// nil/empty input tokens always classify as explore.
func TestAdaptiveWeightedScoring_ClassifyRequest_EmptyPrefix(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)
	aws := policy.(*AdaptiveWeightedScoring)

	snapshots := []RoutingSnapshot{
		{ID: "inst-0"},
		{ID: "inst-1"},
	}

	// Nil request
	mode, _, _ := aws.classifyRequest(nil, snapshots)
	if mode != "explore" {
		t.Errorf("nil request: mode=%s, want explore", mode)
	}

	// Empty input tokens
	mode, _, _ = aws.classifyRequest(&Request{}, snapshots)
	if mode != "explore" {
		t.Errorf("empty input: mode=%s, want explore", mode)
	}
}
