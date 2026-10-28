package sim

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestLoadPolicyBundle_ValidYAML(t *testing.T) {
	yaml := `
priority:
  policy: slo-based
  slo_priorities:
    critical: 9
scheduler: priority-fcfs
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assert.Equal(t, 9, bundle.Priority.SLOPriorities["critical"])
	if bundle.Priority.Policy != "slo-based" {
		t.Errorf("expected priority policy 'slo-based', got %q", bundle.Priority.Policy)
	}
	if bundle.Scheduler != "priority-fcfs" {
		t.Errorf("expected scheduler 'priority-fcfs', got %q", bundle.Scheduler)
	}
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
		// Admission, routing, flow control and tenant budgets are the llm-d router's.
		{"admission", `
admission:
  policy: token-bucket
`},
		{"routing", `
routing:
  policy: weighted
`},
		{"tenant_budgets", `
tenant_budgets:
  a: 0.5
`},
		{"old_scale_up_cooldown_us", `
autoscaler:
  scale_up_cooldown_us: 60000000
`},
		{"old_scale_down_cooldown_us", `
autoscaler:
  scale_down_cooldown_us: 60000000
`},
		{"old_actuation_delay", `
autoscaler:
  actuation_delay:
    mean: 5.0
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
		{"bad scheduler", PolicyBundle{Scheduler: "invalid"}},
		{"bad preemption", PolicyBundle{Preemption: PreemptionConfig{Policy: "random"}}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.bundle.Validate(); err == nil {
				t.Error("expected validation error")
			}
		})
	}
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
  scale_up_stabilization_window_us: 60000000
  scale_down_stabilization_window_us: 180000000
  hpa_scrape_delay:
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
	if bundle.Autoscaler.HPAScrapeDelay.Mean != 10.0 {
		t.Errorf("HPAScrapeDelay.Mean = %v, want 10.0", bundle.Autoscaler.HPAScrapeDelay.Mean)
	}
	if bundle.Autoscaler.HPAScrapeDelay.Stddev != 2.0 {
		t.Errorf("HPAScrapeDelay.Stddev = %v, want 2.0", bundle.Autoscaler.HPAScrapeDelay.Stddev)
	}
	if bundle.Autoscaler.ScaleUpStabilizationWindowUs != 60_000_000 {
		t.Errorf("ScaleUpStabilizationWindowUs = %v, want 60000000", bundle.Autoscaler.ScaleUpStabilizationWindowUs)
	}
	if bundle.Autoscaler.ScaleDownStabilizationWindowUs != 180_000_000 {
		t.Errorf("ScaleDownStabilizationWindowUs = %v, want 180000000", bundle.Autoscaler.ScaleDownStabilizationWindowUs)
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
scheduler: fcfs
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

func TestPolicyBundle_Validate_AutoscalerNewFields(t *testing.T) {
	cases := []struct {
		name string
		cfg  AutoscalerBundleConfig
	}{
		{"negative scale_up_stabilization_window_us", AutoscalerBundleConfig{ScaleUpStabilizationWindowUs: -1}},
		{"NaN scale_up_stabilization_window_us", AutoscalerBundleConfig{ScaleUpStabilizationWindowUs: math.NaN()}},
		{"Inf scale_up_stabilization_window_us", AutoscalerBundleConfig{ScaleUpStabilizationWindowUs: math.Inf(1)}},
		{"negative scale_down_stabilization_window_us", AutoscalerBundleConfig{ScaleDownStabilizationWindowUs: -1}},
		{"NaN scale_down_stabilization_window_us", AutoscalerBundleConfig{ScaleDownStabilizationWindowUs: math.NaN()}},
		{"Inf scale_down_stabilization_window_us", AutoscalerBundleConfig{ScaleDownStabilizationWindowUs: math.Inf(1)}},
		{"negative hpa_scrape_delay.mean", AutoscalerBundleConfig{HPAScrapeDelay: DelayBundleSpec{Mean: -1}}},
		{"NaN hpa_scrape_delay.mean", AutoscalerBundleConfig{HPAScrapeDelay: DelayBundleSpec{Mean: math.NaN()}}},
		{"Inf hpa_scrape_delay.mean", AutoscalerBundleConfig{HPAScrapeDelay: DelayBundleSpec{Mean: math.Inf(1)}}},
		{"negative hpa_scrape_delay.stddev", AutoscalerBundleConfig{HPAScrapeDelay: DelayBundleSpec{Stddev: -1}}},
		{"NaN hpa_scrape_delay.stddev", AutoscalerBundleConfig{HPAScrapeDelay: DelayBundleSpec{Stddev: math.NaN()}}},
		{"Inf hpa_scrape_delay.stddev", AutoscalerBundleConfig{HPAScrapeDelay: DelayBundleSpec{Stddev: math.Inf(1)}}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			bundle := &PolicyBundle{Autoscaler: tc.cfg}
			if err := bundle.Validate(); err == nil {
				t.Errorf("expected validation error for %q, got nil", tc.name)
			}
		})
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

func TestPreemptionPolicy_ValidNames(t *testing.T) {
	for _, name := range []string{"", "fcfs", "priority"} {
		if !IsValidPreemptionPolicy(name) {
			t.Errorf("IsValidPreemptionPolicy(%q) = false, want true", name)
		}
	}
	if IsValidPreemptionPolicy("random") {
		t.Error("IsValidPreemptionPolicy(\"random\") = true, want false")
	}
}

func TestLoadPolicyBundle_InstanceLifecycleSection(t *testing.T) {
	yaml := `
instance_lifecycle:
  loading_delay:
    mean: 270.0
    stddev: 30.0
  warm_start_initial_instances: true
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.InstanceLifecycle.LoadingDelay.Mean != 270.0 {
		t.Errorf("LoadingDelay.Mean = %v, want 270.0", bundle.InstanceLifecycle.LoadingDelay.Mean)
	}
	if bundle.InstanceLifecycle.LoadingDelay.Stddev != 30.0 {
		t.Errorf("LoadingDelay.Stddev = %v, want 30.0", bundle.InstanceLifecycle.LoadingDelay.Stddev)
	}
	if !bundle.InstanceLifecycle.WarmStartInitialInstances {
		t.Errorf("WarmStartInitialInstances = false, want true")
	}
}

func TestLoadPolicyBundle_InstanceLifecycleAbsent_IsZero(t *testing.T) {
	yaml := `
scheduler: fcfs
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("LoadPolicyBundle: %v", err)
	}
	if bundle.InstanceLifecycle.LoadingDelay.Mean != 0.0 {
		t.Errorf("LoadingDelay.Mean = %v, want 0.0 (absent section should default to zero)", bundle.InstanceLifecycle.LoadingDelay.Mean)
	}
}

func TestPolicyBundle_Validate_InstanceLifecycle(t *testing.T) {
	cases := []struct {
		name    string
		yaml    string
		wantErr string
	}{
		{
			name: "negative mean",
			yaml: `instance_lifecycle:
  loading_delay:
    mean: -1.0`,
			wantErr: "loading_delay.mean must be >= 0",
		},
		{
			name: "negative stddev",
			yaml: `instance_lifecycle:
  loading_delay:
    stddev: -1.0`,
			wantErr: "loading_delay.stddev must be >= 0",
		},
		{
			name: "zero mean is valid",
			yaml: `instance_lifecycle:
  loading_delay:
    mean: 0.0`,
			wantErr: "",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeTempYAML(t, tc.yaml)
			bundle, err := LoadPolicyBundle(path)
			if err != nil {
				t.Fatalf("LoadPolicyBundle: %v", err)
			}
			err = bundle.Validate()
			if tc.wantErr == "" {
				if err != nil {
					t.Errorf("Validate() = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("Validate() = %v, want error containing %q", err, tc.wantErr)
			}
		})
	}
}
