package sim

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
)

func float64Ptr(v float64) *float64 { return &v }

func TestLoadPolicyBundle_ValidYAML(t *testing.T) {
	yaml := `
admission:
  policy: token-bucket
  token_bucket_capacity: 5000
  token_bucket_refill_rate: 500
routing:
  policy: weighted
  scorers:
    - name: queue-depth
      weight: 2.0
    - name: kv-utilization
      weight: 2.0
    - name: load-balance
      weight: 1.0
priority:
  policy: slo-based
scheduler: priority-fcfs
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Admission.Policy != "token-bucket" {
		t.Errorf("expected admission policy 'token-bucket', got %q", bundle.Admission.Policy)
	}
	if bundle.Admission.TokenBucketCapacity == nil || *bundle.Admission.TokenBucketCapacity != 5000 {
		t.Errorf("expected capacity 5000, got %v", bundle.Admission.TokenBucketCapacity)
	}
	if bundle.Routing.Policy != "weighted" {
		t.Errorf("expected routing policy 'weighted', got %q", bundle.Routing.Policy)
	}
	if len(bundle.Routing.Scorers) != 3 {
		t.Fatalf("expected 3 scorers, got %d", len(bundle.Routing.Scorers))
	}
	assert.Equal(t, "queue-depth", bundle.Routing.Scorers[0].Name)
	assert.Equal(t, 2.0, bundle.Routing.Scorers[0].Weight)
	assert.Equal(t, "kv-utilization", bundle.Routing.Scorers[1].Name)
	assert.Equal(t, "load-balance", bundle.Routing.Scorers[2].Name)
	if bundle.Priority.Policy != "slo-based" {
		t.Errorf("expected priority policy 'slo-based', got %q", bundle.Priority.Policy)
	}
	if bundle.Scheduler != "priority-fcfs" {
		t.Errorf("expected scheduler 'priority-fcfs', got %q", bundle.Scheduler)
	}
}

func TestLoadPolicyBundle_ScorersAbsent_IsNil(t *testing.T) {
	yaml := `
routing:
  policy: weighted
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Scorers not specified → nil (will use defaults at factory level)
	assert.Nil(t, bundle.Routing.Scorers)
}

func TestLoadPolicyBundle_EmptyFields(t *testing.T) {
	yaml := `
routing:
  policy: least-loaded
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Admission.Policy != "" {
		t.Errorf("expected empty admission policy, got %q", bundle.Admission.Policy)
	}
	if bundle.Routing.Policy != "least-loaded" {
		t.Errorf("expected 'least-loaded', got %q", bundle.Routing.Policy)
	}
	if bundle.Scheduler != "" {
		t.Errorf("expected empty scheduler, got %q", bundle.Scheduler)
	}
	assert.Nil(t, bundle.Routing.Scorers)
}

func TestLoadPolicyBundle_NonexistentFile(t *testing.T) {
	_, err := LoadPolicyBundle("/nonexistent/path.yaml")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadPolicyBundle_MalformedYAML(t *testing.T) {
	path := writeTempYAML(t, "{{invalid yaml")
	_, err := LoadPolicyBundle(path)
	if err == nil {
		t.Fatal("expected error for malformed YAML")
	}
}

// TestLoadPolicyBundle_OldFieldsRejected verifies BC-17-10: old cache_weight/load_weight
// fields produce a parse error due to KnownFields(true) strict parsing.
func TestLoadPolicyBundle_OldFieldsRejected(t *testing.T) {
	tests := []struct {
		name string
		yaml string
	}{
		{"cache_weight", `
routing:
  policy: weighted
  cache_weight: 0.6
`},
		{"load_weight", `
routing:
  policy: weighted
  load_weight: 0.4
`},
		{"both old fields", `
routing:
  policy: weighted
  cache_weight: 0.6
  load_weight: 0.4
`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeTempYAML(t, tt.yaml)
			_, err := LoadPolicyBundle(path)
			assert.Error(t, err, "old YAML field should be rejected by strict parsing")
		})
	}
}

func TestPolicyBundle_Validate_ValidPolicies(t *testing.T) {
	bundle := &PolicyBundle{
		Admission: AdmissionConfig{Policy: "token-bucket", TokenBucketCapacity: float64Ptr(100)},
		Routing: RoutingConfig{
			Policy: "weighted",
			Scorers: []ScorerConfig{
				{Name: "queue-depth", Weight: 2.0},
				{Name: "load-balance", Weight: 1.0},
			},
		},
		Priority:  PriorityConfig{Policy: "slo-based"},
		Scheduler: "priority-fcfs",
	}
	if err := bundle.Validate(); err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

func TestPolicyBundle_Validate_EmptyIsValid(t *testing.T) {
	bundle := &PolicyBundle{}
	if err := bundle.Validate(); err != nil {
		t.Errorf("empty bundle should be valid, got: %v", err)
	}
}

func TestPolicyBundle_Validate_InvalidPolicy(t *testing.T) {
	tests := []struct {
		name   string
		bundle PolicyBundle
	}{
		{"bad admission", PolicyBundle{Admission: AdmissionConfig{Policy: "invalid"}}},
		{"bad routing", PolicyBundle{Routing: RoutingConfig{Policy: "invalid"}}},
		{"bad priority", PolicyBundle{Priority: PriorityConfig{Policy: "invalid"}}},
		{"bad scheduler", PolicyBundle{Scheduler: "invalid"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.bundle.Validate(); err == nil {
				t.Error("expected validation error")
			}
		})
	}
}

func TestPolicyBundle_Validate_NegativeParameters(t *testing.T) {
	tests := []struct {
		name   string
		bundle PolicyBundle
	}{
		{"negative capacity", PolicyBundle{Admission: AdmissionConfig{
			Policy: "token-bucket", TokenBucketCapacity: float64Ptr(-1),
		}}},
		{"negative refill rate", PolicyBundle{Admission: AdmissionConfig{
			Policy: "token-bucket", TokenBucketRefillRate: float64Ptr(-1),
		}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.bundle.Validate(); err == nil {
				t.Error("expected validation error for negative parameter")
			}
		})
	}
}

// TestPolicyBundle_Validate_InvalidScorerName verifies BC-17-4: unknown scorer name rejected.
func TestPolicyBundle_Validate_InvalidScorerName(t *testing.T) {
	bundle := &PolicyBundle{
		Routing: RoutingConfig{
			Policy:  "weighted",
			Scorers: []ScorerConfig{{Name: "unknown-scorer", Weight: 1.0}},
		},
	}
	err := bundle.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "unknown scorer")
}

// TestPolicyBundle_Validate_InvalidScorerWeight verifies BC-17-4: bad weights rejected.
func TestPolicyBundle_Validate_InvalidScorerWeight(t *testing.T) {
	tests := []struct {
		name   string
		weight float64
	}{
		{"zero weight", 0.0},
		{"negative weight", -1.0},
		{"NaN weight", math.NaN()},
		{"Inf weight", math.Inf(1)},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bundle := &PolicyBundle{
				Routing: RoutingConfig{
					Policy:  "weighted",
					Scorers: []ScorerConfig{{Name: "queue-depth", Weight: tt.weight}},
				},
			}
			err := bundle.Validate()
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "weight must be")
		})
	}
}

// TestPolicyBundle_Validate_DuplicateScorer verifies duplicate scorer names are rejected.
func TestPolicyBundle_Validate_DuplicateScorer(t *testing.T) {
	bundle := &PolicyBundle{
		Routing: RoutingConfig{
			Policy: "weighted",
			Scorers: []ScorerConfig{
				{Name: "queue-depth", Weight: 1.0},
				{Name: "queue-depth", Weight: 2.0},
			},
		},
	}
	err := bundle.Validate()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate scorer")
}

// TestPolicyBundle_Validate_ScorersOnNonWeightedPolicy verifies scorers are validated
// even when attached to a non-weighted policy (validation catches config mistakes early).
func TestPolicyBundle_Validate_ScorersOnNonWeightedPolicy(t *testing.T) {
	bundle := &PolicyBundle{
		Routing: RoutingConfig{
			Policy:  "round-robin",
			Scorers: []ScorerConfig{{Name: "unknown", Weight: 1.0}},
		},
	}
	err := bundle.Validate()
	assert.Error(t, err, "invalid scorers should be caught even on non-weighted policy")
}

// TestPolicyBundle_Validate_TenantBudgets verifies BC-TB-1: invalid tenant budget values are rejected.
func TestPolicyBundle_Validate_TenantBudgets(t *testing.T) {
	tests := []struct {
		name    string
		budgets map[string]float64
		wantErr bool
	}{
		{"valid: zero", map[string]float64{"alice": 0.0}, false},
		{"valid: one", map[string]float64{"alice": 1.0}, false},
		{"valid: fraction", map[string]float64{"alice": 0.3, "bob": 0.7}, false},
		{"invalid: negative", map[string]float64{"alice": -0.1}, true},
		{"invalid: greater than one", map[string]float64{"alice": 1.5}, true},
		{"invalid: NaN", map[string]float64{"alice": math.NaN()}, true},
		{"invalid: +Inf", map[string]float64{"alice": math.Inf(1)}, true},
		{"invalid: -Inf", map[string]float64{"alice": math.Inf(-1)}, true},
		{"nil map: no enforcement", nil, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bundle := &PolicyBundle{TenantBudgets: tt.budgets}
			err := bundle.Validate()
			if tt.wantErr {
				assert.Error(t, err, "expected validation error for budgets=%v", tt.budgets)
			} else {
				assert.NoError(t, err, "unexpected validation error for budgets=%v", tt.budgets)
			}
		})
	}
}

// TestPolicyBundle_Validate_EmptyScorersIsValid verifies nil/empty scorers list is acceptable.
func TestPolicyBundle_Validate_EmptyScorersIsValid(t *testing.T) {
	bundle := &PolicyBundle{
		Routing: RoutingConfig{Policy: "weighted"},
	}
	err := bundle.Validate()
	assert.NoError(t, err, "empty scorers list should be valid (defaults used at factory)")
}

func writeTempYAML(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "policy.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestValidAdmissionPolicyNames_ReturnsAllNames(t *testing.T) {
	// BC-7: Names derived from authoritative map
	names := ValidAdmissionPolicyNames()
	assert.Contains(t, names, "always-admit")
	assert.Contains(t, names, "token-bucket")
	assert.Contains(t, names, "reject-all")
	assert.NotContains(t, names, "")
}

func TestValidRoutingPolicyNames_Sorted(t *testing.T) {
	names := ValidRoutingPolicyNames()
	for i := 1; i < len(names); i++ {
		assert.True(t, names[i-1] < names[i], "names must be sorted: %q >= %q", names[i-1], names[i])
	}
}

func TestValidPriorityPolicyNames_ReturnsAllNames(t *testing.T) {
	names := ValidPriorityPolicyNames()
	assert.Contains(t, names, "constant")
	assert.Contains(t, names, "slo-based")
	assert.Contains(t, names, "inverted-slo")
}

func TestValidSchedulerNames_ReturnsAllNames(t *testing.T) {
	names := ValidSchedulerNames()
	assert.Contains(t, names, "fcfs")
	assert.Contains(t, names, "priority-fcfs")
	assert.Contains(t, names, "sjf")
	assert.Contains(t, names, "reverse-priority")
}

// ---------------------------------------------------------------------------
// Autoscaler + NodePool bundle tests
// ---------------------------------------------------------------------------

func TestLoadPolicyBundle_AutoscalerSection(t *testing.T) {
	yaml := `
autoscaler:
  interval_us: 30000000
  scale_up_cooldown_us: 60000000
  scale_down_cooldown_us: 180000000
  actuation_delay:
    mean: 10.0
    stddev: 2.0
  analyzer:
    kv_cache_threshold: 0.8
    scale_up_threshold: 0.8
    scale_down_boundary: 0.4
    avg_input_tokens: 512.0
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.Autoscaler.IntervalUs != 30_000_000 {
		t.Errorf("IntervalUs = %v, want 30000000", bundle.Autoscaler.IntervalUs)
	}
	if bundle.Autoscaler.Analyzer.KVCacheThreshold != 0.8 {
		t.Errorf("KVCacheThreshold = %v, want 0.8", bundle.Autoscaler.Analyzer.KVCacheThreshold)
	}
	if bundle.Autoscaler.ActuationDelay.Mean != 10.0 {
		t.Errorf("ActuationDelay.Mean = %v, want 10.0", bundle.Autoscaler.ActuationDelay.Mean)
	}
	if bundle.Autoscaler.ScaleUpCooldownUs != 60_000_000 {
		t.Errorf("ScaleUpCooldownUs = %v, want 60000000", bundle.Autoscaler.ScaleUpCooldownUs)
	}
}

func TestLoadPolicyBundle_NodePoolsSection(t *testing.T) {
	yaml := `
node_pools:
  - name: h100-pool
    gpu_type: H100
    gpus_per_node: 8
    gpu_memory_gib: 80.0
    initial_nodes: 1
    min_nodes: 1
    max_nodes: 4
    cost_per_hour: 32.0
    provisioning_delay:
      mean: 30.0
      stddev: 5.0
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if len(bundle.NodePools) != 1 {
		t.Fatalf("NodePools len = %d, want 1", len(bundle.NodePools))
	}
	np := bundle.NodePools[0]
	if np.Name != "h100-pool" {
		t.Errorf("Name = %q, want h100-pool", np.Name)
	}
	if np.MaxNodes != 4 {
		t.Errorf("MaxNodes = %d, want 4", np.MaxNodes)
	}
	if np.ProvisioningDelay.Mean != 30.0 {
		t.Errorf("ProvisioningDelay.Mean = %v, want 30.0", np.ProvisioningDelay.Mean)
	}
}

func TestLoadPolicyBundle_AutoscalerAbsent_IsZero(t *testing.T) {
	// Existing policy-config files without autoscaler section must parse cleanly (backward compat).
	yaml := `
admission:
  policy: always-admit
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.Autoscaler.IntervalUs != 0 {
		t.Errorf("IntervalUs = %v, want 0 (disabled)", bundle.Autoscaler.IntervalUs)
	}
	if len(bundle.NodePools) != 0 {
		t.Errorf("NodePools len = %d, want 0", len(bundle.NodePools))
	}
}

func TestPolicyBundle_Validate_AutoscalerNegativeInterval(t *testing.T) {
	bundle := &PolicyBundle{
		Autoscaler: AutoscalerBundleConfig{IntervalUs: -1},
	}
	if err := bundle.Validate(); err == nil {
		t.Error("expected error for negative interval_us, got nil")
	}
}

func TestPolicyBundle_Validate_AnalyzerThresholds(t *testing.T) {
	cases := []struct {
		name    string
		cfg     AnalyzerBundleConfig
		wantErr bool
	}{
		{"zero values are valid (use defaults)", AnalyzerBundleConfig{}, false},
		{"valid explicit values", AnalyzerBundleConfig{ScaleUpThreshold: 0.8, ScaleDownBoundary: 0.4, AvgInputTokens: 512}, false},
		{"negative scale_up_threshold", AnalyzerBundleConfig{ScaleUpThreshold: -0.1}, true},
		{"zero scale_up_threshold treated as default, not invalid", AnalyzerBundleConfig{ScaleUpThreshold: 0}, false},
		{"negative scale_down_boundary", AnalyzerBundleConfig{ScaleDownBoundary: -1}, true},
		{"negative avg_input_tokens", AnalyzerBundleConfig{AvgInputTokens: -100}, true},
		{"scale_down_boundary >= scale_up_threshold", AnalyzerBundleConfig{ScaleUpThreshold: 0.5, ScaleDownBoundary: 0.5}, true},
		{"scale_down_boundary > scale_up_threshold", AnalyzerBundleConfig{ScaleUpThreshold: 0.4, ScaleDownBoundary: 0.8}, true},
		{"scale_down_boundary above default scale_up_threshold (one-sided)", AnalyzerBundleConfig{ScaleDownBoundary: 0.9}, true},
		{"scale_up_threshold below default scale_down_boundary (one-sided)", AnalyzerBundleConfig{ScaleUpThreshold: 0.3}, true},
		{"kv_cache_threshold > 1", AnalyzerBundleConfig{KVCacheThreshold: 1.5}, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bundle := &PolicyBundle{
				Autoscaler: AutoscalerBundleConfig{Analyzer: tc.cfg},
			}
			err := bundle.Validate()
			if tc.wantErr && err == nil {
				t.Errorf("expected validation error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected validation error: %v", err)
			}
		})
	}
}

func TestPolicyBundle_Validate_NodePool_MissingName(t *testing.T) {
	bundle := &PolicyBundle{
		NodePools: []NodePoolBundleConfig{
			{GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, MaxNodes: 2},
		},
	}
	if err := bundle.Validate(); err == nil {
		t.Error("expected error for missing node pool name, got nil")
	}
}

func TestPolicyBundle_Validate_NodePool_MaxLessThanInitial(t *testing.T) {
	bundle := &PolicyBundle{
		NodePools: []NodePoolBundleConfig{
			{Name: "p", GPUType: "H100", GPUsPerNode: 8, GPUMemoryGiB: 80, InitialNodes: 5, MaxNodes: 2},
		},
	}
	if err := bundle.Validate(); err == nil {
		t.Error("expected error for max_nodes < initial_nodes, got nil")
	}
}

func TestIsValidLatencyBackend(t *testing.T) {
	assert.True(t, IsValidLatencyBackend(""))
	assert.True(t, IsValidLatencyBackend("blackbox"))
	assert.True(t, IsValidLatencyBackend("roofline"))
	assert.True(t, IsValidLatencyBackend("trained-physics"))
	assert.False(t, IsValidLatencyBackend("nonexistent"))
}

func TestValidLatencyBackendNames(t *testing.T) {
	names := ValidLatencyBackendNames()
	assert.Contains(t, names, "blackbox")
	assert.Contains(t, names, "roofline")
	assert.Contains(t, names, "trained-physics")
	assert.NotContains(t, names, "")
}

func TestIsValidLatencyBackend_RemovedBackends(t *testing.T) {
	// GIVEN removed backend names
	removedBackends := []string{"crossmodel", "trained-roofline"}

	for _, backend := range removedBackends {
		t.Run(backend, func(t *testing.T) {
			// WHEN checking if backend is valid
			valid := IsValidLatencyBackend(backend)

			// THEN it must return false
			if valid {
				t.Errorf("IsValidLatencyBackend(%q) = true; want false (backend was removed)", backend)
			}
		})
	}
}

func TestValidLatencyBackendNames_ExcludesRemoved(t *testing.T) {
	// GIVEN the list of valid backend names
	names := ValidLatencyBackendNames()

	// WHEN checking for removed backends
	removedBackends := []string{"crossmodel", "trained-roofline"}
	for _, removed := range removedBackends {
		for _, name := range names {
			// THEN removed backends must not appear in the list
			if name == removed {
				t.Errorf("ValidLatencyBackendNames() contains removed backend %q; want excluded", removed)
			}
		}
	}

	// AND the list must contain exactly 3 backends
	expected := []string{"blackbox", "roofline", "trained-physics"}
	if len(names) != len(expected) {
		t.Errorf("ValidLatencyBackendNames() returned %d backends; want %d: %v", len(names), len(expected), expected)
	}

	// AND they must be the correct backends
	nameSet := make(map[string]bool)
	for _, name := range names {
		nameSet[name] = true
	}
	for _, exp := range expected {
		if !nameSet[exp] {
			t.Errorf("ValidLatencyBackendNames() missing expected backend %q", exp)
		}
	}
}
