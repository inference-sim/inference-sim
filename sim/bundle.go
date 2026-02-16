package sim

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// PolicyBundle holds unified policy configuration, loadable from a YAML file.
// Nil pointer fields mean "not set in YAML" — they do not override DeploymentConfig.
// String fields use empty string for "not set".
type PolicyBundle struct {
	Admission AdmissionConfig `yaml:"admission"`
	Routing   RoutingConfig   `yaml:"routing"`
	Priority  PriorityConfig  `yaml:"priority"`
	Scheduler string          `yaml:"scheduler"`
}

// AdmissionConfig holds admission policy configuration.
type AdmissionConfig struct {
	Policy                string   `yaml:"policy"`
	TokenBucketCapacity   *float64 `yaml:"token_bucket_capacity"`
	TokenBucketRefillRate *float64 `yaml:"token_bucket_refill_rate"`
}

// RoutingConfig holds routing policy configuration.
type RoutingConfig struct {
	Policy      string   `yaml:"policy"`
	CacheWeight *float64 `yaml:"cache_weight"`
	LoadWeight  *float64 `yaml:"load_weight"`
}

// PriorityConfig holds priority policy configuration.
type PriorityConfig struct {
	Policy string `yaml:"policy"`
}

// LoadPolicyBundle reads and parses a YAML policy configuration file.
func LoadPolicyBundle(path string) (*PolicyBundle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading policy config: %w", err)
	}
	var bundle PolicyBundle
	if err := yaml.Unmarshal(data, &bundle); err != nil {
		return nil, fmt.Errorf("parsing policy config: %w", err)
	}
	return &bundle, nil
}

// ValidAdmissionPolicies is the set of recognized admission policy names.
// Shared by Validate() and NewAdmissionPolicy() to avoid duplication.
var ValidAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true}

// ValidRoutingPolicies is the set of recognized routing policy names.
var ValidRoutingPolicies = map[string]bool{"": true, "round-robin": true, "least-loaded": true, "weighted": true, "prefix-affinity": true}

// ValidPriorityPolicies is the set of recognized priority policy names.
var ValidPriorityPolicies = map[string]bool{"": true, "constant": true, "slo-based": true}

// ValidSchedulers is the set of recognized scheduler names.
var ValidSchedulers = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true}

// Validate checks that all policy names and parameter ranges in the bundle are valid.
func (b *PolicyBundle) Validate() error {
	if !ValidAdmissionPolicies[b.Admission.Policy] {
		return fmt.Errorf("unknown admission policy %q", b.Admission.Policy)
	}
	if !ValidRoutingPolicies[b.Routing.Policy] {
		return fmt.Errorf("unknown routing policy %q", b.Routing.Policy)
	}
	if !ValidPriorityPolicies[b.Priority.Policy] {
		return fmt.Errorf("unknown priority policy %q", b.Priority.Policy)
	}
	if !ValidSchedulers[b.Scheduler] {
		return fmt.Errorf("unknown scheduler %q", b.Scheduler)
	}
	// Parameter range validation
	if b.Admission.TokenBucketCapacity != nil && *b.Admission.TokenBucketCapacity < 0 {
		return fmt.Errorf("token_bucket_capacity must be non-negative, got %f", *b.Admission.TokenBucketCapacity)
	}
	if b.Admission.TokenBucketRefillRate != nil && *b.Admission.TokenBucketRefillRate < 0 {
		return fmt.Errorf("token_bucket_refill_rate must be non-negative, got %f", *b.Admission.TokenBucketRefillRate)
	}
	if b.Routing.CacheWeight != nil && *b.Routing.CacheWeight < 0 {
		return fmt.Errorf("cache_weight must be non-negative, got %f", *b.Routing.CacheWeight)
	}
	if b.Routing.LoadWeight != nil && *b.Routing.LoadWeight < 0 {
		return fmt.Errorf("load_weight must be non-negative, got %f", *b.Routing.LoadWeight)
	}
	return nil
}
