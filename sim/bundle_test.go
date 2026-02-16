package sim

import (
	"os"
	"path/filepath"
	"testing"
)

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
	if bundle.Admission.TokenBucketCapacity != 5000 {
		t.Errorf("expected capacity 5000, got %f", bundle.Admission.TokenBucketCapacity)
	}
	if bundle.Admission.TokenBucketRefillRate != 500 {
		t.Errorf("expected refill rate 500, got %f", bundle.Admission.TokenBucketRefillRate)
	}
	if bundle.Routing.Policy != "weighted" {
		t.Errorf("expected routing policy 'weighted', got %q", bundle.Routing.Policy)
	}
	if bundle.Routing.CacheWeight != 0.7 {
		t.Errorf("expected cache weight 0.7, got %f", bundle.Routing.CacheWeight)
	}
	if bundle.Routing.LoadWeight != 0.3 {
		t.Errorf("expected load weight 0.3, got %f", bundle.Routing.LoadWeight)
	}
	if bundle.Priority.Policy != "slo-based" {
		t.Errorf("expected priority policy 'slo-based', got %q", bundle.Priority.Policy)
	}
	if bundle.Scheduler != "priority-fcfs" {
		t.Errorf("expected scheduler 'priority-fcfs', got %q", bundle.Scheduler)
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
		Admission: AdmissionConfig{Policy: "token-bucket"},
		Routing:   RoutingConfig{Policy: "weighted"},
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

func writeTempYAML(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "policy.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
