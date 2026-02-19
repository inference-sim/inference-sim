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
	Policy  string         `yaml:"policy"`
	Scorers []ScorerConfig `yaml:"scorers"`
}

// PriorityConfig holds priority policy configuration.
type PriorityConfig struct {
	Policy string `yaml:"policy"`
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
	validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true, "reject-all": true}
	validRoutingPolicies   = map[string]bool{"": true, "round-robin": true, "least-loaded": true, "weighted": true, "prefix-affinity": true, "always-busiest": true}
	validPriorityPolicies  = map[string]bool{"": true, "constant": true, "slo-based": true, "inverted-slo": true}
	validSchedulers        = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true, "reverse-priority": true}
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
	// Validate scorer configs if present
	for i, sc := range b.Routing.Scorers {
		if !IsValidScorer(sc.Name) {
			return fmt.Errorf("routing scorer[%d]: unknown scorer %q; valid: %s",
				i, sc.Name, strings.Join(ValidScorerNames(), ", "))
		}
		if sc.Weight <= 0 || math.IsNaN(sc.Weight) || math.IsInf(sc.Weight, 0) {
			return fmt.Errorf("routing scorer[%d] %q: weight must be a finite positive number, got %v",
				i, sc.Name, sc.Weight)
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
