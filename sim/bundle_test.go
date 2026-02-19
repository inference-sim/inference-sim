package sim

import (
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
  cache_weight: 0.7
  load_weight: 0.3
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
	if bundle.Admission.TokenBucketRefillRate == nil || *bundle.Admission.TokenBucketRefillRate != 500 {
		t.Errorf("expected refill rate 500, got %v", bundle.Admission.TokenBucketRefillRate)
	}
	if bundle.Routing.Policy != "weighted" {
		t.Errorf("expected routing policy 'weighted', got %q", bundle.Routing.Policy)
	}
	if bundle.Routing.CacheWeight == nil || *bundle.Routing.CacheWeight != 0.7 {
		t.Errorf("expected cache weight 0.7, got %v", bundle.Routing.CacheWeight)
	}
	if bundle.Routing.LoadWeight == nil || *bundle.Routing.LoadWeight != 0.3 {
		t.Errorf("expected load weight 0.3, got %v", bundle.Routing.LoadWeight)
	}
	if bundle.Priority.Policy != "slo-based" {
		t.Errorf("expected priority policy 'slo-based', got %q", bundle.Priority.Policy)
	}
	if bundle.Scheduler != "priority-fcfs" {
		t.Errorf("expected scheduler 'priority-fcfs', got %q", bundle.Scheduler)
	}
}

func TestLoadPolicyBundle_ZeroValueIsDistinctFromUnset(t *testing.T) {
	yaml := `
routing:
  policy: weighted
  cache_weight: 0.0
  load_weight: 1.0
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// cache_weight: 0.0 should be explicitly set (non-nil), not treated as "unset"
	if bundle.Routing.CacheWeight == nil {
		t.Fatal("expected CacheWeight to be non-nil (explicitly set to 0.0)")
	}
	if *bundle.Routing.CacheWeight != 0.0 {
		t.Errorf("expected CacheWeight 0.0, got %f", *bundle.Routing.CacheWeight)
	}
	// load_weight: 1.0 should be set
	if bundle.Routing.LoadWeight == nil || *bundle.Routing.LoadWeight != 1.0 {
		t.Errorf("expected LoadWeight 1.0, got %v", bundle.Routing.LoadWeight)
	}
	// Unset fields should be nil
	if bundle.Admission.TokenBucketCapacity != nil {
		t.Errorf("expected nil TokenBucketCapacity for unset field, got %f", *bundle.Admission.TokenBucketCapacity)
	}
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
	if bundle.Routing.CacheWeight != nil {
		t.Errorf("expected nil CacheWeight for unset field")
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

func TestPolicyBundle_Validate_ValidPolicies(t *testing.T) {
	bundle := &PolicyBundle{
		Admission: AdmissionConfig{Policy: "token-bucket", TokenBucketCapacity: float64Ptr(100)},
		Routing:   RoutingConfig{Policy: "weighted", CacheWeight: float64Ptr(0.6), LoadWeight: float64Ptr(0.4)},
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
		{"negative cache weight", PolicyBundle{Routing: RoutingConfig{
			Policy: "weighted", CacheWeight: float64Ptr(-0.5),
		}}},
		{"negative load weight", PolicyBundle{Routing: RoutingConfig{
			Policy: "weighted", LoadWeight: float64Ptr(-0.5),
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

func TestPolicyBundle_Validate_ZeroParametersAreValid(t *testing.T) {
	bundle := &PolicyBundle{
		Routing: RoutingConfig{
			Policy:      "weighted",
			CacheWeight: float64Ptr(0.0),
			LoadWeight:  float64Ptr(1.0),
		},
	}
	if err := bundle.Validate(); err != nil {
		t.Errorf("zero cache weight should be valid, got: %v", err)
	}
}

func TestPolicyBundle_Validate_WeightSumNotOne(t *testing.T) {
	// GIVEN routing weights that don't sum to 1.0
	tests := []struct {
		name   string
		cache  float64
		load   float64
	}{
		{"sum 0.8", 0.6, 0.2},
		{"sum 1.5", 1.0, 0.5},
		{"sum 0.0 (both zero)", 0.0, 0.0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bundle := &PolicyBundle{
				Routing: RoutingConfig{
					Policy:      "weighted",
					CacheWeight: float64Ptr(tt.cache),
					LoadWeight:  float64Ptr(tt.load),
				},
			}
			// THEN Validate should return an error for non-unit sum
			if err := bundle.Validate(); err == nil {
				t.Errorf("expected validation error for weights summing to %.1f", tt.cache+tt.load)
			}
		})
	}
}

func TestPolicyBundle_Validate_WeightSumOne(t *testing.T) {
	// GIVEN routing weights that sum to 1.0
	bundle := &PolicyBundle{
		Routing: RoutingConfig{
			Policy:      "weighted",
			CacheWeight: float64Ptr(0.7),
			LoadWeight:  float64Ptr(0.3),
		},
	}
	// THEN Validate should pass
	if err := bundle.Validate(); err != nil {
		t.Errorf("expected no error for weights summing to 1.0, got: %v", err)
	}
}

func TestPolicyBundle_Validate_WeightSumSkippedForNonWeighted(t *testing.T) {
	// GIVEN a non-weighted routing policy with arbitrary weight values
	bundle := &PolicyBundle{
		Routing: RoutingConfig{
			Policy:      "round-robin",
			CacheWeight: float64Ptr(0.6),
			LoadWeight:  float64Ptr(0.2),
		},
	}
	// THEN Validate should not check weight sum (weights are irrelevant)
	if err := bundle.Validate(); err != nil {
		t.Errorf("expected no error for non-weighted policy, got: %v", err)
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
