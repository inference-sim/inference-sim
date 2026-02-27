package sim

import (
	"math"
	"testing"
)

// TestAdaptiveWeightedScoring_CriticalSLO_RoutesToLeastLoaded verifies that
// critical-SLO requests are routed to the least-loaded instance even when
// a cache hit exists on a busier instance.
func TestAdaptiveWeightedScoring_CriticalSLO_RoutesToLeastLoaded(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)

	// Seed cache on inst-0 with a shared prefix
	prefix := make([]int, 256)
	for i := range prefix {
		prefix[i] = 1000 + i
	}
	seedReq := &Request{
		InputTokens: append(append([]int{}, prefix...), make([]int, 128)...),
		SLOClass:    "batch", // seed with batch to warm cache
	}
	for i := 256; i < len(seedReq.InputTokens); i++ {
		seedReq.InputTokens[i] = 5000 + i
	}

	balancedState := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 0},
			{ID: "inst-1", QueueDepth: 0},
		},
	}
	policy.Route(seedReq, balancedState)

	// Now inst-0 is busier, critical request should go to inst-1
	loadedState := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 5, BatchSize: 2},
			{ID: "inst-1", QueueDepth: 0, BatchSize: 0},
		},
	}
	criticalReq := &Request{
		InputTokens: append(append([]int{}, prefix...), make([]int, 128)...),
		SLOClass:    "critical",
	}
	for i := 256; i < len(criticalReq.InputTokens); i++ {
		criticalReq.InputTokens[i] = 9000 + i
	}

	d := policy.Route(criticalReq, loadedState)

	// Critical SLO has MaxLoadHeadroom=0 + no PA weight → must go to least-loaded
	if d.TargetInstance != "inst-1" {
		t.Errorf("critical SLO should route to least-loaded inst-1; got %s", d.TargetInstance)
	}
}

// TestAdaptiveWeightedScoring_BatchSLO_ExploitsCache verifies that
// batch-SLO requests exploit cache hits even on slightly busier instances.
func TestAdaptiveWeightedScoring_BatchSLO_ExploitsCache(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)

	prefix := make([]int, 256)
	for i := range prefix {
		prefix[i] = 1000 + i
	}

	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 0},
			{ID: "inst-1", QueueDepth: 0},
		},
	}

	// Seed cache on inst-0 with many batch requests to build PA scores
	for i := 0; i < 50; i++ {
		req := &Request{
			InputTokens: append(append([]int{}, prefix...), make([]int, 128)...),
			SLOClass:    "batch",
		}
		for j := 256; j < len(req.InputTokens); j++ {
			req.InputTokens[j] = i*1000 + j
		}
		policy.Route(req, state)
	}

	// Now inst-0 has some load but batch tolerance is high (MaxLoadHeadroom=10)
	loadedState := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 3, BatchSize: 1},
			{ID: "inst-1", QueueDepth: 0, BatchSize: 0},
		},
	}
	batchReq := &Request{
		InputTokens: append(append([]int{}, prefix...), make([]int, 128)...),
		SLOClass:    "batch",
	}
	for i := 256; i < len(batchReq.InputTokens); i++ {
		batchReq.InputTokens[i] = 9000 + i
	}

	d := policy.Route(batchReq, loadedState)

	// Batch SLO has high PA weight + MaxLoadHeadroom=10 → should exploit cache
	// The PA scorer gives inst-0 a high score from the cached prefix.
	// QD=3 vs QD=0 is within headroom=10.
	if d.TargetInstance != "inst-0" {
		t.Errorf("batch SLO should exploit cache on inst-0 (QD diff=3, headroom=10); got %s",
			d.TargetInstance)
	}
}

// TestAdaptiveWeightedScoring_EmptySLOUsesStandardProfile verifies that
// requests with empty SLO class use the standard (default) profile.
func TestAdaptiveWeightedScoring_EmptySLOUsesStandardProfile(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)

	req := &Request{
		InputTokens: make([]int, 32),
		SLOClass:    "", // empty
	}
	for i := range req.InputTokens {
		req.InputTokens[i] = i
	}

	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 10},
			{ID: "inst-1", QueueDepth: 0},
		},
	}

	d := policy.Route(req, state)

	// Should use standard profile (pa:3,qd:2,kv:2) → route to least-loaded
	if d.TargetInstance != "inst-1" {
		t.Errorf("empty SLO should use standard profile → least-loaded inst-1; got %s",
			d.TargetInstance)
	}
}

// TestAdaptiveWeightedScoring_DeterministicRouting verifies INV-6.
func TestAdaptiveWeightedScoring_DeterministicRouting(t *testing.T) {
	policy1 := NewRoutingPolicy("adaptive-weighted", nil, 16)
	policy2 := NewRoutingPolicy("adaptive-weighted", nil, 16)

	req := &Request{
		InputTokens: make([]int, 32),
		SLOClass:    "standard",
	}
	for i := range req.InputTokens {
		req.InputTokens[i] = i
	}

	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0", QueueDepth: 5, BatchSize: 2, KVUtilization: 0.3},
			{ID: "inst-1", QueueDepth: 10, BatchSize: 1, KVUtilization: 0.5},
		},
	}

	d1 := policy1.Route(req, state)
	d2 := policy2.Route(req, state)

	if d1.TargetInstance != d2.TargetInstance {
		t.Errorf("non-deterministic: policy1=%s, policy2=%s", d1.TargetInstance, d2.TargetInstance)
	}
}

// TestValidateAdaptiveConfig_RejectsInvalid verifies config validation.
func TestValidateAdaptiveConfig_RejectsInvalid(t *testing.T) {
	tests := []struct {
		name    string
		config  AdaptiveConfig
		wantErr bool
	}{
		{"valid defaults", DefaultAdaptiveConfig(), false},
		{"threshold=1", AdaptiveConfig{ExploitThreshold: 1.0}, false},
		{"threshold=0", AdaptiveConfig{ExploitThreshold: 0}, true},
		{"threshold>1", AdaptiveConfig{ExploitThreshold: 1.5}, true},
		{"NaN threshold", AdaptiveConfig{ExploitThreshold: math.NaN()}, true},
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

// TestAdaptiveWeightedScoring_NilRequest verifies safe handling of nil request.
func TestAdaptiveWeightedScoring_NilRequest(t *testing.T) {
	policy := NewRoutingPolicy("adaptive-weighted", nil, 16)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{
			{ID: "inst-0"},
			{ID: "inst-1"},
		},
	}

	// Should not panic, route using default profile
	d := policy.Route(nil, state)
	if d.TargetInstance == "" {
		t.Error("expected non-empty target for nil request")
	}
}
