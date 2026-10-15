package sim

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

// PolicyBundle holds unified policy configuration, loadable from a YAML file.
// Nil pointer fields mean "not set in YAML" — they do not override DeploymentConfig.
// String fields use empty string for "not set".
//
// Admission, routing and flow control are not part of a bundle: they are the llm-d router's,
// configured by the deployment's router inputs (see sim/llmd).
type PolicyBundle struct {
	Priority          PriorityConfig                `yaml:"priority"`
	Scheduler         string                        `yaml:"scheduler"`
	Preemption        PreemptionConfig              `yaml:"preemption"`
	NodePools         []NodePoolBundleConfig        `yaml:"node_pools"`         // nil = no node pools
	Autoscaler        AutoscalerBundleConfig        `yaml:"autoscaler"`         // IntervalUs=0 = disabled
	InstanceLifecycle InstanceLifecycleBundleConfig `yaml:"instance_lifecycle"` // zero = instant loading
}

// PriorityConfig holds priority policy configuration.
type PriorityConfig struct {
	Policy string `yaml:"policy"`
	// SLOPriorities overrides the default SLO class to engine priority mapping used by priority
	// scheduling and preemption. nil = defaults (critical=4, standard=3, batch=-1,
	// sheddable=-2, background=-3).
	SLOPriorities map[string]int `yaml:"slo_priorities,omitempty"`
}

// PreemptionConfig holds preemption policy configuration.
type PreemptionConfig struct {
	Policy string `yaml:"policy"`
}

// NodePoolBundleConfig mirrors cluster.NodePoolConfig for YAML loading in PolicyBundle.
// Converted to cluster.NodePoolConfig in cmd/ to avoid a sim→sim/cluster circular import.
type NodePoolBundleConfig struct {
	Name              string          `yaml:"name"`
	GPUType           string          `yaml:"gpu_type"`
	GPUsPerNode       int             `yaml:"gpus_per_node"`
	GPUMemoryGiB      float64         `yaml:"gpu_memory_gib"`
	InitialNodes      int             `yaml:"initial_nodes"`
	MinNodes          int             `yaml:"min_nodes"`
	MaxNodes          int             `yaml:"max_nodes"`
	ProvisioningDelay DelayBundleSpec `yaml:"provisioning_delay"`
	CostPerHour       float64         `yaml:"cost_per_hour"`
}

// DelayBundleSpec mirrors cluster.DelaySpec for YAML loading. Mean and Stddev in seconds.
type DelayBundleSpec struct {
	Mean   float64 `yaml:"mean"`
	Stddev float64 `yaml:"stddev"`
}

// AnalyzerBundleConfig holds V2SaturationAnalyzer thresholds.
// Zero values mean "use defaults" (applied by effectiveAnalyzerConfig in cluster package).
type AnalyzerBundleConfig struct {
	KVCacheThreshold  float64 `yaml:"kv_cache_threshold"`
	ScaleUpThreshold  float64 `yaml:"scale_up_threshold"`
	ScaleDownBoundary float64 `yaml:"scale_down_boundary"`
	AvgInputTokens    float64 `yaml:"avg_input_tokens"`
}

// InstanceLifecycleBundleConfig holds instance lifecycle timing for YAML loading.
// Mean and Stddev are in seconds; converted to microseconds when building DeploymentConfig.
type InstanceLifecycleBundleConfig struct {
	LoadingDelay              DelayBundleSpec `yaml:"loading_delay"`
	WarmStartInitialInstances bool            `yaml:"warm_start_initial_instances"`
}

// AutoscalerBundleConfig holds autoscaler pipeline configuration.
// IntervalUs == 0 disables the autoscaler (default).
// Note: ScaleDownStabilizationWindowUs = 0 (the Go zero value, used when the field is
// absent from YAML) means no stabilization — scale-down acts on the first signal. To match
// the Kubernetes HPA default scale-down behavior, set it to 300,000,000 (5 min) explicitly.
type AutoscalerBundleConfig struct {
	IntervalUs                     float64              `yaml:"interval_us"`
	ScaleUpStabilizationWindowUs   float64              `yaml:"scale_up_stabilization_window_us"`
	ScaleDownStabilizationWindowUs float64              `yaml:"scale_down_stabilization_window_us"`
	HPAScrapeDelay                 DelayBundleSpec      `yaml:"hpa_scrape_delay"`
	Analyzer                       AnalyzerBundleConfig `yaml:"analyzer"`
}

// LoadPolicyBundle reads and parses a YAML policy configuration file.
// Uses strict parsing: unrecognized keys (typos) are rejected.
func LoadPolicyBundle(path string) (*PolicyBundle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading policy config: %w", err)
	}
	var bundle PolicyBundle
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&bundle); err != nil {
		return nil, fmt.Errorf("parsing policy config: %w", err)
	}
	return &bundle, nil
}

// Valid policy name registries. Unexported to prevent external mutation.
// Used by Validate(), factory functions, and ValidatePolicyName().
var (
	validSchedulers         = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true, "reverse-priority": true}
	validPreemptionPolicies = map[string]bool{"": true, "fcfs": true, "priority": true}
)

// IsValidScheduler returns true if name is a recognized scheduler.
func IsValidScheduler(name string) bool { return validSchedulers[name] }

// ValidSchedulerNames returns sorted valid scheduler names (excluding empty).
func ValidSchedulerNames() []string { return validNamesList(validSchedulers) }

// IsValidPreemptionPolicy returns true if name is a recognized preemption policy.
func IsValidPreemptionPolicy(name string) bool { return validPreemptionPolicies[name] }

// ValidPreemptionPolicyNames returns sorted valid preemption policy names (excluding empty).
func ValidPreemptionPolicyNames() []string { return validNamesList(validPreemptionPolicies) }

// validNamesList returns sorted non-empty keys from a validity map.
func validNamesList(m map[string]bool) []string {
	names := make([]string, 0, len(m))
	for k := range m {
		if k != "" {
			names = append(names, k)
		}
	}
	sort.Strings(names)
	return names
}

// validNames returns a sorted comma-separated list of valid names (excluding empty string).
func validNames(m map[string]bool) string {
	return strings.Join(validNamesList(m), ", ")
}

// Validate checks that all policy names and parameter ranges in the bundle are valid.
func (b *PolicyBundle) Validate() error {
	if !validSchedulers[b.Scheduler] {
		return fmt.Errorf("unknown scheduler %q; valid options: %s", b.Scheduler, validNames(validSchedulers))
	}
	if !validPreemptionPolicies[b.Preemption.Policy] {
		return fmt.Errorf("unknown preemption policy %q; valid options: %s", b.Preemption.Policy, validNames(validPreemptionPolicies))
	}
	// Validate autoscaler config.
	if math.IsNaN(b.Autoscaler.IntervalUs) || math.IsInf(b.Autoscaler.IntervalUs, 0) {
		return fmt.Errorf("autoscaler.interval_us must be a finite number")
	}
	if b.Autoscaler.IntervalUs < 0 {
		return fmt.Errorf("autoscaler.interval_us must be >= 0 (0 = disabled), got %v", b.Autoscaler.IntervalUs)
	}
	if math.IsNaN(b.Autoscaler.ScaleUpStabilizationWindowUs) || math.IsInf(b.Autoscaler.ScaleUpStabilizationWindowUs, 0) || b.Autoscaler.ScaleUpStabilizationWindowUs < 0 {
		return fmt.Errorf("autoscaler.scale_up_stabilization_window_us must be a finite non-negative number, got %v", b.Autoscaler.ScaleUpStabilizationWindowUs)
	}
	if math.IsNaN(b.Autoscaler.ScaleDownStabilizationWindowUs) || math.IsInf(b.Autoscaler.ScaleDownStabilizationWindowUs, 0) || b.Autoscaler.ScaleDownStabilizationWindowUs < 0 {
		return fmt.Errorf("autoscaler.scale_down_stabilization_window_us must be a finite non-negative number, got %v", b.Autoscaler.ScaleDownStabilizationWindowUs)
	}
	if math.IsNaN(b.Autoscaler.HPAScrapeDelay.Mean) || math.IsInf(b.Autoscaler.HPAScrapeDelay.Mean, 0) || b.Autoscaler.HPAScrapeDelay.Mean < 0 {
		return fmt.Errorf("autoscaler.hpa_scrape_delay.mean must be a finite non-negative number, got %v", b.Autoscaler.HPAScrapeDelay.Mean)
	}
	if math.IsNaN(b.Autoscaler.HPAScrapeDelay.Stddev) || math.IsInf(b.Autoscaler.HPAScrapeDelay.Stddev, 0) || b.Autoscaler.HPAScrapeDelay.Stddev < 0 {
		return fmt.Errorf("autoscaler.hpa_scrape_delay.stddev must be a finite non-negative number, got %v", b.Autoscaler.HPAScrapeDelay.Stddev)
	}
	// Validate analyzer thresholds (non-zero = explicitly set; zero = use default).
	// Mirrors the panic guards in NewV2SaturationAnalyzer so invalid values are caught
	// at the CLI boundary (R3) rather than at runtime during simulation.
	a := b.Autoscaler.Analyzer
	if a.KVCacheThreshold != 0 && (a.KVCacheThreshold <= 0 || a.KVCacheThreshold > 1 || math.IsNaN(a.KVCacheThreshold) || math.IsInf(a.KVCacheThreshold, 0)) {
		return fmt.Errorf("autoscaler.analyzer.kv_cache_threshold must be in (0, 1], got %v", a.KVCacheThreshold)
	}
	if a.ScaleUpThreshold != 0 && (a.ScaleUpThreshold <= 0 || math.IsNaN(a.ScaleUpThreshold) || math.IsInf(a.ScaleUpThreshold, 0)) {
		return fmt.Errorf("autoscaler.analyzer.scale_up_threshold must be > 0, got %v", a.ScaleUpThreshold)
	}
	if a.ScaleDownBoundary != 0 && (a.ScaleDownBoundary <= 0 || math.IsNaN(a.ScaleDownBoundary) || math.IsInf(a.ScaleDownBoundary, 0)) {
		return fmt.Errorf("autoscaler.analyzer.scale_down_boundary must be > 0, got %v", a.ScaleDownBoundary)
	}
	if a.AvgInputTokens != 0 && (a.AvgInputTokens <= 0 || math.IsNaN(a.AvgInputTokens) || math.IsInf(a.AvgInputTokens, 0)) {
		return fmt.Errorf("autoscaler.analyzer.avg_input_tokens must be > 0, got %v", a.AvgInputTokens)
	}
	// Cross-check scale_down_boundary < scale_up_threshold using effective (post-default) values.
	// A one-sided override (e.g. scale_down_boundary: 0.9, scale_up_threshold unset → default 0.8)
	// would pass individual checks but panic inside NewV2SaturationAnalyzer (R3).
	effectiveUp := a.ScaleUpThreshold
	if effectiveUp == 0 {
		effectiveUp = 0.8 // WVA reference default (mirrors effectiveAnalyzerConfig)
	}
	effectiveDown := a.ScaleDownBoundary
	if effectiveDown == 0 {
		effectiveDown = 0.4 // WVA reference default (mirrors effectiveAnalyzerConfig)
	}
	if effectiveDown >= effectiveUp {
		return fmt.Errorf("autoscaler.analyzer.scale_down_boundary (%v) must be < effective scale_up_threshold (%v)",
			effectiveDown, effectiveUp)
	}
	// Validate node pools — mirrors cluster.NodePoolConfig.IsValid() (NodePoolBundleConfig is a
	// separate mirror type to avoid sim→sim/cluster circular import, so we inline the checks).
	for i, np := range b.NodePools {
		if np.Name == "" {
			return fmt.Errorf("node_pools[%d]: name must not be empty", i)
		}
		if np.GPUType == "" {
			return fmt.Errorf("node_pools[%d] %q: gpu_type must not be empty", i, np.Name)
		}
		if np.GPUsPerNode < 1 {
			return fmt.Errorf("node_pools[%d] %q: gpus_per_node must be >= 1, got %d", i, np.Name, np.GPUsPerNode)
		}
		if math.IsNaN(np.GPUMemoryGiB) || math.IsInf(np.GPUMemoryGiB, 0) {
			return fmt.Errorf("node_pools[%d] %q: gpu_memory_gib must be finite, got %v", i, np.Name, np.GPUMemoryGiB)
		}
		if np.GPUMemoryGiB <= 0 {
			return fmt.Errorf("node_pools[%d] %q: gpu_memory_gib must be > 0, got %v", i, np.Name, np.GPUMemoryGiB)
		}
		if np.InitialNodes < 0 {
			return fmt.Errorf("node_pools[%d] %q: initial_nodes must be >= 0, got %d", i, np.Name, np.InitialNodes)
		}
		if np.MinNodes < 0 {
			return fmt.Errorf("node_pools[%d] %q: min_nodes must be >= 0, got %d", i, np.Name, np.MinNodes)
		}
		if np.MaxNodes < np.InitialNodes {
			return fmt.Errorf("node_pools[%d] %q: max_nodes (%d) must be >= initial_nodes (%d)", i, np.Name, np.MaxNodes, np.InitialNodes)
		}
		if np.MinNodes > np.MaxNodes {
			return fmt.Errorf("node_pools[%d] %q: min_nodes (%d) must be <= max_nodes (%d)", i, np.Name, np.MinNodes, np.MaxNodes)
		}
		if math.IsNaN(np.ProvisioningDelay.Mean) || math.IsInf(np.ProvisioningDelay.Mean, 0) || np.ProvisioningDelay.Mean < 0 {
			return fmt.Errorf("node_pools[%d] %q: provisioning_delay.mean must be finite and >= 0, got %v", i, np.Name, np.ProvisioningDelay.Mean)
		}
		if math.IsNaN(np.ProvisioningDelay.Stddev) || math.IsInf(np.ProvisioningDelay.Stddev, 0) || np.ProvisioningDelay.Stddev < 0 {
			return fmt.Errorf("node_pools[%d] %q: provisioning_delay.stddev must be finite and >= 0, got %v", i, np.Name, np.ProvisioningDelay.Stddev)
		}
		if math.IsNaN(np.CostPerHour) || math.IsInf(np.CostPerHour, 0) {
			return fmt.Errorf("node_pools[%d] %q: cost_per_hour must be finite, got %v", i, np.Name, np.CostPerHour)
		}
		if np.CostPerHour < 0 {
			return fmt.Errorf("node_pools[%d] %q: cost_per_hour must be >= 0, got %v", i, np.Name, np.CostPerHour)
		}
	}
	// Validate instance lifecycle config.
	lm := b.InstanceLifecycle.LoadingDelay.Mean
	if math.IsNaN(lm) || math.IsInf(lm, 0) {
		return fmt.Errorf("instance_lifecycle.loading_delay.mean must be a finite number, got %v", lm)
	}
	if lm < 0 {
		return fmt.Errorf("instance_lifecycle.loading_delay.mean must be >= 0, got %v", lm)
	}
	ls := b.InstanceLifecycle.LoadingDelay.Stddev
	if math.IsNaN(ls) || math.IsInf(ls, 0) {
		return fmt.Errorf("instance_lifecycle.loading_delay.stddev must be a finite number, got %v", ls)
	}
	if ls < 0 {
		return fmt.Errorf("instance_lifecycle.loading_delay.stddev must be >= 0, got %v", ls)
	}
	return nil
}

// validateFloat checks that a float parameter is non-negative and finite.
func validateFloat(name string, val *float64) error {
	if val == nil {
		return nil
	}
	if math.IsNaN(*val) || math.IsInf(*val, 0) {
		return fmt.Errorf("%s must be a finite number, got %f", name, *val)
	}
	if *val < 0 {
		return fmt.Errorf("%s must be non-negative, got %f", name, *val)
	}
	return nil
}
