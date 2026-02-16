package sim

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// PolicyBundle holds unified policy configuration, loadable from a YAML file.
// Zero-value fields mean "use default" — they do not override DeploymentConfig.
type PolicyBundle struct {
	Admission AdmissionConfig `yaml:"admission"`
	Routing   RoutingConfig   `yaml:"routing"`
	Priority  PriorityConfig  `yaml:"priority"`
	Scheduler string          `yaml:"scheduler"`
}

// AdmissionConfig holds admission policy configuration.
type AdmissionConfig struct {
	Policy                string  `yaml:"policy"`
	TokenBucketCapacity   float64 `yaml:"token_bucket_capacity"`
	TokenBucketRefillRate float64 `yaml:"token_bucket_refill_rate"`
}

// RoutingConfig holds routing policy configuration.
type RoutingConfig struct {
	Policy      string  `yaml:"policy"`
	CacheWeight float64 `yaml:"cache_weight"`
	LoadWeight  float64 `yaml:"load_weight"`
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

var (
	validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true}
	validRoutingPolicies   = map[string]bool{"": true, "round-robin": true, "least-loaded": true, "weighted": true, "prefix-affinity": true}
	validPriorityPolicies  = map[string]bool{"": true, "constant": true, "slo-based": true}
	validSchedulers        = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true}
)

// Validate checks that all policy names in the bundle are recognized.
func (b *PolicyBundle) Validate() error {
	if !validAdmissionPolicies[b.Admission.Policy] {
		return fmt.Errorf("unknown admission policy %q", b.Admission.Policy)
	}
	if !validRoutingPolicies[b.Routing.Policy] {
		return fmt.Errorf("unknown routing policy %q", b.Routing.Policy)
	}
	if !validPriorityPolicies[b.Priority.Policy] {
		return fmt.Errorf("unknown priority policy %q", b.Priority.Policy)
	}
	if !validSchedulers[b.Scheduler] {
		return fmt.Errorf("unknown scheduler %q", b.Scheduler)
	}
	return nil
}
