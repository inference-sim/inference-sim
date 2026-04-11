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
type PolicyBundle struct {
	Admission     AdmissionConfig        `yaml:"admission"`
	Routing       RoutingConfig          `yaml:"routing"`
	Priority      PriorityConfig         `yaml:"priority"`
	Scheduler     string                 `yaml:"scheduler"`
	TenantBudgets map[string]float64     `yaml:"tenant_budgets"`  // nil = no tenant enforcement
	NodePools     []NodePoolBundleConfig `yaml:"node_pools"`      // nil = no node pools
	Autoscaler    AutoscalerBundleConfig `yaml:"autoscaler"`      // IntervalUs=0 = disabled
}

// AdmissionConfig holds admission policy configuration.
type AdmissionConfig struct {
	Policy                string   `yaml:"policy"`
	TokenBucketCapacity   *float64 `yaml:"token_bucket_capacity"`
	TokenBucketRefillRate *float64 `yaml:"token_bucket_refill_rate"`
	// Tier-shed options (Phase 1B): only used when policy = "tier-shed".
	TierShedThreshold   *int `yaml:"tier_shed_threshold"`   // nil = use default (0)
	TierShedMinPriority *int `yaml:"tier_shed_min_priority"` // nil = use default (3)
	// SLOPriorities overrides default SLO class → priority mappings.
	// nil = use GAIE defaults (critical=4, standard=3, batch=-1, sheddable=-2, background=-3).
	SLOPriorities map[string]int `yaml:"slo_priorities,omitempty"`
}

// RoutingConfig holds routing policy configuration.
type RoutingConfig struct {
	Policy  string         `yaml:"policy"`
	Scorers []ScorerConfig `yaml:"scorers"`
}

// PriorityConfig holds priority policy configuration.
type PriorityConfig struct {
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

// AutoscalerBundleConfig holds autoscaler pipeline configuration.
// IntervalUs == 0 disables the autoscaler (default).
type AutoscalerBundleConfig struct {
	IntervalUs          float64              `yaml:"interval_us"`
	ScaleUpCooldownUs   float64              `yaml:"scale_up_cooldown_us"`
	ScaleDownCooldownUs float64              `yaml:"scale_down_cooldown_us"`
	ActuationDelay      DelayBundleSpec      `yaml:"actuation_delay"`
	Analyzer            AnalyzerBundleConfig `yaml:"analyzer"`
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
	validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true, "reject-all": true, "tier-shed": true}
	validRoutingPolicies   = map[string]bool{"": true, "round-robin": true, "least-loaded": true, "weighted": true, "always-busiest": true}
	validPriorityPolicies  = map[string]bool{"": true, "constant": true, "slo-based": true, "inverted-slo": true}
	validSchedulers        = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true, "reverse-priority": true}
	validLatencyBackends          = map[string]bool{"": true, "blackbox": true, "roofline": true, "crossmodel": true, "trained-roofline": true, "trained-physics": true}
	validDisaggregationDeciders   = map[string]bool{"": true, "never": true, "always": true, "prefix-threshold": true}
	validSaturationDetectors      = map[string]bool{"": true, "never": true, "utilization": true, "concurrency": true}
)

// IsValidAdmissionPolicy returns true if name is a recognized admission policy.
func IsValidAdmissionPolicy(name string) bool { return validAdmissionPolicies[name] }

// IsValidRoutingPolicy returns true if name is a recognized routing policy.
func IsValidRoutingPolicy(name string) bool { return validRoutingPolicies[name] }

// IsValidPriorityPolicy returns true if name is a recognized priority policy.
func IsValidPriorityPolicy(name string) bool { return validPriorityPolicies[name] }

// IsValidScheduler returns true if name is a recognized scheduler.
func IsValidScheduler(name string) bool { return validSchedulers[name] }

// ValidAdmissionPolicyNames returns sorted valid admission policy names (excluding empty).
func ValidAdmissionPolicyNames() []string { return validNamesList(validAdmissionPolicies) }

// ValidRoutingPolicyNames returns sorted valid routing policy names (excluding empty).
func ValidRoutingPolicyNames() []string { return validNamesList(validRoutingPolicies) }

// ValidPriorityPolicyNames returns sorted valid priority policy names (excluding empty).
func ValidPriorityPolicyNames() []string { return validNamesList(validPriorityPolicies) }

// ValidSchedulerNames returns sorted valid scheduler names (excluding empty).
func ValidSchedulerNames() []string { return validNamesList(validSchedulers) }

// IsValidLatencyBackend returns true if name is a recognized latency model backend.
func IsValidLatencyBackend(name string) bool { return validLatencyBackends[name] }

// ValidLatencyBackendNames returns sorted valid latency backend names (excluding empty).
func ValidLatencyBackendNames() []string { return validNamesList(validLatencyBackends) }

// IsValidDisaggregationDecider returns true if name is a recognized disaggregation decider.
func IsValidDisaggregationDecider(name string) bool { return validDisaggregationDeciders[name] }

// ValidDisaggregationDeciderNames returns sorted valid disaggregation decider names (excluding empty).
func ValidDisaggregationDeciderNames() []string { return validNamesList(validDisaggregationDeciders) }

// IsValidSaturationDetector returns true if name is a recognized saturation detector.
func IsValidSaturationDetector(name string) bool { return validSaturationDetectors[name] }

// ValidSaturationDetectorNames returns sorted valid saturation detector names (excluding empty).
func ValidSaturationDetectorNames() []string { return validNamesList(validSaturationDetectors) }

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
	if !validAdmissionPolicies[b.Admission.Policy] {
		return fmt.Errorf("unknown admission policy %q; valid options: %s", b.Admission.Policy, validNames(validAdmissionPolicies))
	}
	if !validRoutingPolicies[b.Routing.Policy] {
		return fmt.Errorf("unknown routing policy %q; valid options: %s", b.Routing.Policy, validNames(validRoutingPolicies))
	}
	if !validPriorityPolicies[b.Priority.Policy] {
		return fmt.Errorf("unknown priority policy %q; valid options: %s", b.Priority.Policy, validNames(validPriorityPolicies))
	}
	if !validSchedulers[b.Scheduler] {
		return fmt.Errorf("unknown scheduler %q; valid options: %s", b.Scheduler, validNames(validSchedulers))
	}
	// Parameter range validation (reject negative, NaN, and Inf)
	if err := validateFloat("token_bucket_capacity", b.Admission.TokenBucketCapacity); err != nil {
		return err
	}
	if err := validateFloat("token_bucket_refill_rate", b.Admission.TokenBucketRefillRate); err != nil {
		return err
	}
	// Validate tier-shed parameters when present.
	if b.Admission.TierShedThreshold != nil && *b.Admission.TierShedThreshold < 0 {
		return fmt.Errorf("tier_shed_threshold must be >= 0, got %d", *b.Admission.TierShedThreshold)
	}
	// No range check on TierShedMinPriority: GAIE priorities are arbitrary integers
	// with no bounds (InferenceObjective.spec.priority is *int, negative allowed).
	// The only semantic boundary is priority < 0 → sheddable.
	// Validate tenant budgets: each value must be in [0, 1].
	for tenantID, v := range b.TenantBudgets {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
			return fmt.Errorf("tenant_budgets[%q] must be in [0, 1], got %v", tenantID, v)
		}
	}
	// Validate scorer configs if present
	scorerSeen := make(map[string]bool, len(b.Routing.Scorers))
	for i, sc := range b.Routing.Scorers {
		if !IsValidScorer(sc.Name) {
			return fmt.Errorf("routing scorer[%d]: unknown scorer %q; valid: %s",
				i, sc.Name, strings.Join(ValidScorerNames(), ", "))
		}
		if scorerSeen[sc.Name] {
			return fmt.Errorf("routing scorer[%d]: duplicate scorer %q; each scorer may appear at most once", i, sc.Name)
		}
		scorerSeen[sc.Name] = true
		if sc.Weight <= 0 || math.IsNaN(sc.Weight) || math.IsInf(sc.Weight, 0) {
			return fmt.Errorf("routing scorer[%d] %q: weight must be a finite positive number, got %v",
				i, sc.Name, sc.Weight)
		}
	}
	// Validate autoscaler config.
	if math.IsNaN(b.Autoscaler.IntervalUs) || math.IsInf(b.Autoscaler.IntervalUs, 0) {
		return fmt.Errorf("autoscaler.interval_us must be a finite number")
	}
	if b.Autoscaler.IntervalUs < 0 {
		return fmt.Errorf("autoscaler.interval_us must be >= 0 (0 = disabled), got %v", b.Autoscaler.IntervalUs)
	}
	if b.Autoscaler.ScaleUpCooldownUs < 0 {
		return fmt.Errorf("autoscaler.scale_up_cooldown_us must be >= 0, got %v", b.Autoscaler.ScaleUpCooldownUs)
	}
	if b.Autoscaler.ScaleDownCooldownUs < 0 {
		return fmt.Errorf("autoscaler.scale_down_cooldown_us must be >= 0, got %v", b.Autoscaler.ScaleDownCooldownUs)
	}
	if b.Autoscaler.ActuationDelay.Mean < 0 {
		return fmt.Errorf("autoscaler.actuation_delay.mean must be >= 0, got %v", b.Autoscaler.ActuationDelay.Mean)
	}
	if b.Autoscaler.ActuationDelay.Stddev < 0 {
		return fmt.Errorf("autoscaler.actuation_delay.stddev must be >= 0, got %v", b.Autoscaler.ActuationDelay.Stddev)
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
