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
