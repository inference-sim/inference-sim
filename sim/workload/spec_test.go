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
	if spec.Version != "2" {
		t.Errorf("version = %q, want %q (auto-upgraded from v1)", spec.Version, "2")
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
				SLOClass:     "critical",
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

func TestLoadWorkloadSpec_ModelField_Parsed(t *testing.T) {
	// BC-4: model field parsed from YAML
	dir := t.TempDir()
	path := filepath.Join(dir, "model.yaml")
	yamlContent := `
version: "2"
seed: 42
category: language
aggregate_rate: 100.0
clients:
  - id: "c1"
    model: "llama-3.1-8b"
    slo_class: "standard"
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
	if spec.Clients[0].Model != "llama-3.1-8b" {
		t.Errorf("Model = %q, want %q", spec.Clients[0].Model, "llama-3.1-8b")
	}
}

func TestValidate_ConstantArrival_NoError(t *testing.T) {
	// BC-7: Constant arrival process validates successfully
	cv := 2.0 // should be ignored for constant
	spec := &WorkloadSpec{
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant", CV: &cv},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("unexpected error for constant arrival: %v", err)
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

func TestWorkloadSpec_Validate_WeibullHighCV_WithExplicitShapeScale_Passes(t *testing.T) {
	// GIVEN a Weibull arrival with CV far exceeding [0.01, 10.4] but
	// explicit MLE-fitted shape/scale provided (ServeGen pattern).
	// CV is informational only when shape/scale are present.
	cv := 173.81
	shape := 0.0575
	scale := 0.000573
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival: ArrivalSpec{
				Process: "weibull",
				CV:      &cv,
				Shape:   &shape,
				Scale:   &scale,
			},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err != nil {
		t.Errorf("expected no error for high-CV Weibull with explicit shape/scale, got: %v", err)
	}
}

func TestValidate_ConcurrencyClient_AcceptsZeroRateFraction(t *testing.T) {
	spec := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{{
			ID:          "conc",
			Concurrency: 10,
			ThinkTimeUs: 0,
			Arrival:     ArrivalSpec{Process: "constant"},
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("expected valid concurrency spec, got: %v", err)
	}
}

func TestValidate_ConcurrencyAndRateFraction_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID:           "bad",
			Concurrency:  10,
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "constant"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for client with both concurrency and rate_fraction")
	}
}

func TestValidate_NegativeConcurrency_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{{
			ID:          "bad",
			Concurrency: -1,
			Arrival:     ArrivalSpec{Process: "constant"},
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for negative concurrency")
	}
}

func TestValidate_LifecycleNoWindows_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "bad", RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			Lifecycle:  &LifecycleSpec{Windows: []ActiveWindow{}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for lifecycle with no windows")
	}
}

func TestValidate_LifecycleNegativeStartUs_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "bad", RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: -1, EndUs: 1_000_000}},
			},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for lifecycle window with negative start_us")
	}
}

func TestValidate_LifecycleEndUsNotGreaterThanStartUs_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version: "2", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "bad", RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 1_000_000, EndUs: 1_000_000}},
			},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for lifecycle window with end_us == start_us")
	}
}

func TestValidate_NegativeThinkTimeUs_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{{
			ID:          "bad",
			Concurrency: 10,
			ThinkTimeUs: -1,
			Arrival:     ArrivalSpec{Process: "constant"},
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for negative think_time_us")
	}
}

func TestValidate_AggregateRateNotRequired_WhenAllConcurrency(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: 0,
		Clients: []ClientSpec{{
			ID:          "conc",
			Concurrency: 10,
			Arrival:     ArrivalSpec{Process: "constant"},
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("expected valid spec with all-concurrency clients, got: %v", err)
	}
}

func TestValidate_ConcurrencyAndMultiTurn_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "reasoning",
		AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID:          "conc",
				Concurrency: 10,
				Arrival:     ArrivalSpec{Process: "constant"},
				InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			},
			{
				ID:           "multi",
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Reasoning:    &ReasoningSpec{MultiTurn: &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 1000}},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for mixed concurrency + multi-turn clients")
	}
	if err != nil && !strings.Contains(err.Error(), "concurrency clients and multi-turn clients cannot be mixed") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_ConcurrencyAndCohortMultiTurn_Rejects(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "2",
		Category:      "reasoning",
		AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID:          "conc",
				Concurrency: 10,
				Arrival:     ArrivalSpec{Process: "constant"},
				InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
			},
		},
		Cohorts: []CohortSpec{
			{
				ID:           "cohort-mt",
				Population:   5,
				RateFraction: 1.0,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
				OutputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
				Reasoning:    &ReasoningSpec{MultiTurn: &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 1000}},
			},
		},
	}
	err := spec.Validate()
	if err == nil {
		t.Error("expected error for concurrency client + multi-turn cohort")
	}
	if err != nil && !strings.Contains(err.Error(), "concurrency clients and multi-turn clients cannot be mixed") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidate_ConcurrencyClient_NoArrivalField_Accepted(t *testing.T) {
	spec := &WorkloadSpec{
		Version:  "2",
		Category: "language",
		Clients: []ClientSpec{{
			ID:          "conc",
			Concurrency: 10,
			InputDist:   DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 200}},
			OutputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 50, "std_dev": 5, "min": 1, "max": 100}},
		}},
	}
	if err := spec.Validate(); err != nil {
		t.Errorf("expected valid spec for concurrency client without arrival field, got: %v", err)
	}
}

func TestWorkloadSpec_Validate_InfShape_ReturnsError(t *testing.T) {
	// BC-11: +Inf shape bypasses > 0 check but creates degenerate samplers (R3 compliance)
	infShape := math.Inf(1)
	scale := 0.5
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival: ArrivalSpec{
				Process: "gamma",
				Shape:   &infShape,
				Scale:   &scale,
			},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for +Inf shape parameter")
	}
	if !strings.Contains(err.Error(), "shape") && !strings.Contains(err.Error(), "finite") {
		t.Errorf("error should mention shape and finite: %v", err)
	}
}

func TestWorkloadSpec_Validate_InfScale_ReturnsError(t *testing.T) {
	// BC-11: +Inf scale bypasses > 0 check but creates degenerate samplers (R3 compliance)
	shape := 1.5
	infScale := math.Inf(1)
	spec := &WorkloadSpec{
		Version:       "2",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival: ArrivalSpec{
				Process: "weibull",
				Shape:   &shape,
				Scale:   &infScale,
			},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for +Inf scale parameter")
	}
	if !strings.Contains(err.Error(), "scale") && !strings.Contains(err.Error(), "finite") {
		t.Errorf("error should mention scale and finite: %v", err)
	}
}

func TestWorkloadSpec_YAML_ShapeScaleRoundTrip(t *testing.T) {
	// BC-12: Shape and Scale fields round-trip through YAML with KnownFields(true)
	cv := 2.5
	shape := 0.16
	scale := 6250000.0 // microseconds
	original := &WorkloadSpec{
		Version:       "2",
		Category:      "language",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival: ArrivalSpec{
				Process: "gamma",
				CV:      &cv,
				Shape:   &shape,
				Scale:   &scale,
			},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}

	// Marshal to YAML
	var buf bytes.Buffer
	encoder := yaml.NewEncoder(&buf)
	if err := encoder.Encode(original); err != nil {
		t.Fatalf("encode error: %v", err)
	}
	yamlBytes := buf.Bytes()

	// Unmarshal with KnownFields(true) to catch typos
	var decoded WorkloadSpec
	decoder := yaml.NewDecoder(bytes.NewReader(yamlBytes))
	decoder.KnownFields(true)
	if err := decoder.Decode(&decoded); err != nil {
		t.Fatalf("decode error: %v", err)
	}

	// Verify Shape and Scale round-tripped correctly
	if decoded.Clients[0].Arrival.Shape == nil {
		t.Fatal("Shape not round-tripped")
	}
	if decoded.Clients[0].Arrival.Scale == nil {
		t.Fatal("Scale not round-tripped")
	}
	if *decoded.Clients[0].Arrival.Shape != shape {
		t.Errorf("Shape = %f, want %f", *decoded.Clients[0].Arrival.Shape, shape)
	}
	if *decoded.Clients[0].Arrival.Scale != scale {
		t.Errorf("Scale = %f, want %f", *decoded.Clients[0].Arrival.Scale, scale)
	}
	// Verify CV also round-tripped
	if decoded.Clients[0].Arrival.CV == nil || *decoded.Clients[0].Arrival.CV != cv {
		t.Errorf("CV = %v, want %f", decoded.Clients[0].Arrival.CV, cv)
	}
}

func TestValidateCohort_ShapeScaleValidation(t *testing.T) {
	// Exercise the cohort path (Cohorts field) for the new shape/scale validation logic.
	// Ensures validateCohort applies the same guards as validateClient.
	tests := []struct {
		name      string
		process   string
		cv        *float64
		shape     *float64
		scale     *float64
		wantError bool
		errorText string
	}{
		{
			name:      "weibull_high_cv_with_both_params_passes",
			process:   "weibull",
			cv:        ptrFloat64(173.81),
			shape:     ptrFloat64(0.05),
			scale:     ptrFloat64(1000000.0),
			wantError: false,
		},
		{
			name:      "weibull_high_cv_shape_only_cv_check_fires",
			process:   "weibull",
			cv:        ptrFloat64(173.81),
			shape:     ptrFloat64(0.05),
			scale:     nil, // Missing scale → CV check applies
			wantError: true,
			errorText: "CV",
		},
		{
			name:      "inf_shape_in_cohort_returns_error",
			process:   "gamma",
			cv:        ptrFloat64(2.5),
			shape:     ptrFloat64(math.Inf(1)),
			scale:     ptrFloat64(50000.0),
			wantError: true,
			errorText: "finite",
		},
		{
			name:      "inf_scale_in_cohort_returns_error",
			process:   "weibull",
			cv:        ptrFloat64(1.5),
			shape:     ptrFloat64(1.5),
			scale:     ptrFloat64(math.Inf(1)),
			wantError: true,
			errorText: "finite",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			spec := &WorkloadSpec{
				Version:       "2",
				AggregateRate: 100.0,
				Cohorts: []CohortSpec{{
					ID:           "cohort1",
					RateFraction: 1.0,
					Population:   100,
					Arrival: ArrivalSpec{
						Process: tc.process,
						CV:      tc.cv,
						Shape:   tc.shape,
						Scale:   tc.scale,
					},
					InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
					OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
				}},
			}
			err := spec.Validate()
			if tc.wantError {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.errorText)
				}
				if !strings.Contains(err.Error(), tc.errorText) {
					t.Errorf("error should contain %q: %v", tc.errorText, err)
				}
			} else {
				if err != nil {
					t.Errorf("unexpected error: %v", err)
				}
			}
		})
	}
}

func TestExampleWorkloadFiles_AllValid(t *testing.T) {
	// Validate all example workload specs load and pass validation.
	// Only files that parse as WorkloadSpec are tested — examples/
	// also contains policy configs and inference-perf specs.
	files, err := filepath.Glob("../../examples/*.yaml")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no example YAML files found — check relative path from sim/workload/")
	}
	for _, path := range files {
		t.Run(filepath.Base(path), func(t *testing.T) {
			spec, err := LoadWorkloadSpec(path)
			if err != nil {
				t.Skipf("not a workload spec: %v", err)
			}
			if spec.InferencePerf != nil {
				t.Skip("inference-perf spec — validated via its own pipeline")
			}
			if err := spec.Validate(); err != nil {
				t.Errorf("validation failed for %s: %v", path, err)
			}
		})
	}
}

func TestExampleWorkloadFiles_CanonicalSLOClasses(t *testing.T) {
	// BC-1: Verify raw YAML slo_class values are canonical v2 names,
	// not deprecated v1 names that rely on auto-upgrade.
	files, err := filepath.Glob("../../examples/*.yaml")
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no example YAML files found — check relative path from sim/workload/")
	}
	for _, path := range files {
		t.Run(filepath.Base(path), func(t *testing.T) {
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("read %s: %v", path, err)
			}
			var raw struct {
				Clients []struct {
					SLOClass string `yaml:"slo_class"`
				} `yaml:"clients"`
			}
			if err := yaml.Unmarshal(data, &raw); err != nil {
				t.Fatalf("unmarshal %s: %v", path, err)
			}
			for i, c := range raw.Clients {
				if !IsValidSLOClass(c.SLOClass) {
					t.Errorf("client[%d] slo_class %q is not a canonical v2 tier name in %s",
						i, c.SLOClass, filepath.Base(path))
				}
			}
		})
	}
}
