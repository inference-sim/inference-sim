package workload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func TestLoadWorkloadSpec_ValidYAML_LoadsCorrectly(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "spec.yaml")
	yaml := `
version: "1"
seed: 42
category: language
aggregate_rate: 100.0
clients:
  - id: "client-a"
    tenant_id: "tenant-1"
    slo_class: "batch"
    rate_fraction: 0.7
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 128
        min: 10
        max: 4096
    output_distribution:
      type: exponential
      params:
        mean: 256
`
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}

	spec, err := LoadWorkloadSpec(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Version != "1" {
		t.Errorf("version = %q, want %q", spec.Version, "1")
	}
	if spec.Seed != 42 {
		t.Errorf("seed = %d, want 42", spec.Seed)
	}
	if spec.AggregateRate != 100.0 {
		t.Errorf("aggregate_rate = %f, want 100.0", spec.AggregateRate)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("clients count = %d, want 1", len(spec.Clients))
	}
	c := spec.Clients[0]
	if c.ID != "client-a" || c.TenantID != "tenant-1" || c.SLOClass != "batch" {
		t.Errorf("client fields mismatch: id=%q tenant=%q slo=%q", c.ID, c.TenantID, c.SLOClass)
	}
	if c.RateFraction != 0.7 {
		t.Errorf("rate_fraction = %f, want 0.7", c.RateFraction)
	}
	if c.Arrival.Process != "poisson" {
		t.Errorf("arrival process = %q, want poisson", c.Arrival.Process)
	}
	if c.InputDist.Type != "gaussian" {
		t.Errorf("input dist type = %q, want gaussian", c.InputDist.Type)
	}
	if c.InputDist.Params["mean"] != 512 {
		t.Errorf("input dist mean = %f, want 512", c.InputDist.Params["mean"])
	}
}

func TestLoadWorkloadSpec_UnknownKey_ReturnsError(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.yaml")
	yaml := `
version: "1"
seed: 42
aggreate_rate: 100.0
`
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadWorkloadSpec(path)
	if err == nil {
		t.Fatal("expected error for unknown key, got nil")
	}
}

func TestWorkloadSpec_Validate_EmptyClients_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected validation error for empty clients")
	}
}

func TestWorkloadSpec_Validate_InvalidArrivalProcess_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "invalid"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 1000}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for invalid arrival process")
	}
}

func TestWorkloadSpec_Validate_InvalidDistType_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "unknown_dist"},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for invalid distribution type")
	}
}

func TestWorkloadSpec_Validate_InvalidCategory_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		Category:      "invalid_category",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 1000}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for invalid category")
	}
}

func TestWorkloadSpec_Validate_InvalidSLOClass_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			SLOClass:     "premium",
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 1000}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for invalid SLO class")
	}
}

func TestWorkloadSpec_Validate_NegativeRate_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: -10.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 1000}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for negative aggregate rate")
	}
}

func TestWorkloadSpec_Validate_ValidSpec_NoError(t *testing.T) {
	cv := 2.0
	spec := &WorkloadSpec{
		Version:       "1",
		Category:      "language",
		AggregateRate: 100.0,
		Clients: []ClientSpec{
			{
				ID:           "c1",
				TenantID:     "t1",
				SLOClass:     "batch",
				RateFraction: 0.7,
				Arrival:      ArrivalSpec{Process: "gamma", CV: &cv},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 512, "std_dev": 128, "min": 10, "max": 4096}},
				OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 256}},
			},
			{
				ID:           "c2",
				SLOClass:     "realtime",
				RateFraction: 0.3,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "exponential", Params: map[string]float64{"mean": 128}},
				OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 64}},
			},
		},
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("expected no error for valid spec, got: %v", err)
	}
}

func TestWorkloadSpec_Validate_NaNParam_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{
				Type:   "exponential",
				Params: map[string]float64{"mean": nanVal()},
			},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for NaN parameter")
	}
	if !strings.Contains(err.Error(), "NaN") && !strings.Contains(err.Error(), "finite") {
		t.Errorf("error should mention NaN: %v", err)
	}
}

func parseWorkloadSpecFromBytes(data []byte) (*WorkloadSpec, error) {
	var spec WorkloadSpec
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&spec); err != nil {
		return nil, err
	}
	return &spec, nil
}

func TestWorkloadSpec_NumRequests_ParsedFromYAML(t *testing.T) {
	// BC-4: YAML num_requests field is parsed
	yamlData := `
version: "1"
seed: 42
category: language
aggregate_rate: 10.0
num_requests: 200
clients:
  - id: "c1"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 100
    output_distribution:
      type: exponential
      params:
        mean: 50
`
	spec, err := parseWorkloadSpecFromBytes([]byte(yamlData))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.NumRequests != 200 {
		t.Errorf("NumRequests = %d, want 200", spec.NumRequests)
	}
}

func TestWorkloadSpec_NumRequestsOmitted_DefaultsToZero(t *testing.T) {
	// BC-9: omitted num_requests defaults to 0 (unlimited)
	yamlData := `
version: "1"
seed: 42
category: language
aggregate_rate: 10.0
clients:
  - id: "c1"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 100
    output_distribution:
      type: exponential
      params:
        mean: 50
`
	spec, err := parseWorkloadSpecFromBytes([]byte(yamlData))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.NumRequests != 0 {
		t.Errorf("NumRequests = %d, want 0 (default unlimited)", spec.NumRequests)
	}
}

func nanVal() float64 {
	return math.NaN()
}

func TestIsValidSLOClass_V2Tiers_ReturnsTrue(t *testing.T) {
	// BC-6: IsValidSLOClass returns true for all v2 tier names
	validTiers := []string{"", "critical", "standard", "sheddable", "batch", "background"}
	for _, tier := range validTiers {
		if !IsValidSLOClass(tier) {
			t.Errorf("IsValidSLOClass(%q) = false, want true", tier)
		}
	}
}

func TestIsValidSLOClass_Invalid_ReturnsFalse(t *testing.T) {
	// BC-6: IsValidSLOClass returns false for non-v2 names
	invalidTiers := []string{"premium", "realtime", "interactive", "urgent", "low"}
	for _, tier := range invalidTiers {
		if IsValidSLOClass(tier) {
			t.Errorf("IsValidSLOClass(%q) = true, want false", tier)
		}
	}
}

func TestValidate_V2SLOTiers_NoError(t *testing.T) {
	// BC-2: v2 spec validates with all v2 tier names
	tiers := []string{"", "critical", "standard", "sheddable", "batch", "background"}
	for _, tier := range tiers {
		spec := &WorkloadSpec{
			AggregateRate: 100.0,
			Clients: []ClientSpec{{
				ID: "c1", RateFraction: 1.0, SLOClass: tier,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			}},
		}
		if err := spec.Validate(); err != nil {
			t.Errorf("Validate() with SLOClass=%q: unexpected error: %v", tier, err)
		}
	}
}

func TestValidate_UnknownSLOTier_ReturnsError(t *testing.T) {
	// BC-10: Unknown SLO class rejected with descriptive error
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0, SLOClass: "premium",
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for unknown SLO class")
	}
	if !strings.Contains(err.Error(), "premium") {
		t.Errorf("error should mention the invalid class: %v", err)
	}
	if !strings.Contains(err.Error(), "critical") {
		t.Errorf("error should list valid classes: %v", err)
	}
}

func TestUpgradeV1ToV2_EmptyVersion_SetsV2(t *testing.T) {
	spec := &WorkloadSpec{Version: ""}
	UpgradeV1ToV2(spec)
	if spec.Version != "2" {
		t.Errorf("Version = %q, want %q", spec.Version, "2")
	}
}

func TestUpgradeV1ToV2_V1Version_SetsV2(t *testing.T) {
	spec := &WorkloadSpec{Version: "1"}
	UpgradeV1ToV2(spec)
	if spec.Version != "2" {
		t.Errorf("Version = %q, want %q", spec.Version, "2")
	}
}

func TestUpgradeV1ToV2_V2Version_NoChange(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2",
		Clients: []ClientSpec{{SLOClass: "critical"}},
	}
	UpgradeV1ToV2(spec)
	if spec.Version != "2" {
		t.Errorf("Version = %q, want %q", spec.Version, "2")
	}
	if spec.Clients[0].SLOClass != "critical" {
		t.Errorf("SLOClass changed unexpectedly to %q", spec.Clients[0].SLOClass)
	}
}

func TestUpgradeV1ToV2_RealtimeMappedToCritical(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1",
		Clients: []ClientSpec{{SLOClass: "realtime"}},
	}
	UpgradeV1ToV2(spec)
	if spec.Clients[0].SLOClass != "critical" {
		t.Errorf("SLOClass = %q, want %q", spec.Clients[0].SLOClass, "critical")
	}
}

func TestUpgradeV1ToV2_InteractiveMappedToStandard(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1",
		Clients: []ClientSpec{{SLOClass: "interactive"}},
	}
	UpgradeV1ToV2(spec)
	if spec.Clients[0].SLOClass != "standard" {
		t.Errorf("SLOClass = %q, want %q", spec.Clients[0].SLOClass, "standard")
	}
}

func TestUpgradeV1ToV2_EmptySLOClassUnchanged(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1",
		Clients: []ClientSpec{{SLOClass: ""}},
	}
	UpgradeV1ToV2(spec)
	if spec.Clients[0].SLOClass != "" {
		t.Errorf("SLOClass = %q, want empty string", spec.Clients[0].SLOClass)
	}
}

func TestUpgradeV1ToV2_BatchUnchanged(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1",
		Clients: []ClientSpec{{SLOClass: "batch"}},
	}
	UpgradeV1ToV2(spec)
	if spec.Clients[0].SLOClass != "batch" {
		t.Errorf("SLOClass = %q, want %q", spec.Clients[0].SLOClass, "batch")
	}
}

func TestUpgradeV1ToV2_Idempotent(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "1",
		Clients: []ClientSpec{{SLOClass: "realtime"}, {SLOClass: "interactive"}},
	}
	UpgradeV1ToV2(spec)
	UpgradeV1ToV2(spec)
	if spec.Clients[0].SLOClass != "critical" {
		t.Errorf("SLOClass[0] = %q, want %q", spec.Clients[0].SLOClass, "critical")
	}
	if spec.Clients[1].SLOClass != "standard" {
		t.Errorf("SLOClass[1] = %q, want %q", spec.Clients[1].SLOClass, "standard")
	}
}

func TestLoadWorkloadSpec_V1File_AutoUpgradedToV2(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "v1.yaml")
	yamlContent := `
version: "1"
seed: 42
category: language
aggregate_rate: 100.0
clients:
  - id: "c1"
    slo_class: "realtime"
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: exponential
      params:
        mean: 100
    output_distribution:
      type: exponential
      params:
        mean: 50
`
	if err := os.WriteFile(path, []byte(yamlContent), 0644); err != nil {
		t.Fatal(err)
	}
	spec, err := LoadWorkloadSpec(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Version != "2" {
		t.Errorf("Version = %q, want %q", spec.Version, "2")
	}
	if spec.Clients[0].SLOClass != "critical" {
		t.Errorf("SLOClass = %q, want %q", spec.Clients[0].SLOClass, "critical")
	}
}

func TestWorkloadSpec_Validate_WeibullCVOutOfRange_ReturnsError(t *testing.T) {
	cv := 20.0 // > 10.4, outside Weibull convergence range
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "weibull", CV: &cv},
			InputDist:    DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for Weibull CV > 10.4")
	}
}
