# W0-1: Spec v2 Schema + SLO Tier Expansion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a versioned workload specification format (v2) with expanded SLO tiers, a constant-rate arrival process, and a model tag — so that every future workload feature has a stable, backward-compatible schema to build on.

**The problem today:** The workload spec has no version field, only 3 SLO tiers (realtime, interactive, batch), no constant-rate arrival process (needed for legacy parity), and no model tag for multi-model workloads. This blocks Phase 0 unification: the unified path needs 5 SLO tiers for autoscaler experiments, a constant sampler to match legacy distribution mode exactly, and a model tag for Phase 2 multi-model routing.

**What this PR adds:**
1. **v2 spec schema** — a `version` field on WorkloadSpec; v1 files auto-upgrade transparently with a deprecation warning (e.g., `realtime` → `critical`)
2. **5-tier SLO classification** — critical, standard, sheddable, batch, background — matching llm-d's priority classes
3. **Constant arrival sampler** — produces requests at exact `1/rate` intervals (zero variance), needed for deterministic legacy parity
4. **Model tag** — a string field on ClientSpec and Request that carries through the pipeline without affecting routing in Phase 0

**Why this matters:** This is the foundation for all Phase 0 PRs. W0-2 (converters), W0-3 (cohort dynamics), and W0-4 (legacy retirement) all depend on the v2 schema being in place.

**Architecture:** All changes are in `sim/workload/` (spec parsing, validation, arrival sampler) and `sim/` (Request struct, RequestMetrics). No CLI changes, no cluster changes, no event loop changes. The `UpgradeV1ToV2` function runs inside `LoadWorkloadSpec` and `GenerateRequests` before validation.

**Source:** W0-1 in Phase 0 macro plan (GitHub issue #420, comment 2)

**Closes:** N/A — source is macro plan, no linked issues

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR extends the `sim/workload/` package with a v2 spec schema. It adds a version field to `WorkloadSpec`, expands the SLO tier taxonomy from 3 to 5 named tiers, adds a constant arrival sampler behind the existing `ArrivalSampler` interface, and adds a `Model` string field that flows from `ClientSpec` through `Request` to `RequestMetrics` without affecting any routing or scheduling decisions.

**Where it fits:** First PR in the Phase 0 workload unification series. Depends on nothing. W0-2 (binary rename + converters), W0-3 (cohort dynamics), and W0-4 (legacy retirement) depend on it.

**Adjacent blocks:** `sim/workload/spec.go` (validation registries), `sim/workload/arrival.go` (sampler factory), `sim/workload/generator.go` (request construction), `sim/request.go` (Request struct), `sim/metrics_utils.go` (RequestMetrics).

**No deviations from source documents.**

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: v1 Auto-Upgrade**
- GIVEN a WorkloadSpec with version "" or "1" and clients using v1 SLO class names
- WHEN loaded via `LoadWorkloadSpec` or passed to `GenerateRequests`
- THEN the spec's version becomes "2", deprecated SLO class names are mapped (realtime→critical, interactive→standard), and a deprecation warning is emitted to stderr
- MECHANISM: `UpgradeV1ToV2` called before validation in both paths

**BC-2: v2 SLO Tier Validation**
- GIVEN a WorkloadSpec with a client whose SLO class is not in {`""`, `critical`, `standard`, `sheddable`, `batch`, `background`}
- WHEN `Validate()` is called (after upgrade)
- THEN validation returns an error listing the valid v2 tier names
- MECHANISM: `validSLOClasses` registry updated to v2 set; `validateClient` checks against it

**BC-3: Constant Arrival Sampler**
- GIVEN a client with `arrival.process: "constant"` and a configured rate
- WHEN the sampler generates inter-arrival times
- THEN every IAT equals exactly `floor(1/rate)` microseconds (minimum 1), regardless of the RNG state
- MECHANISM: `ConstantArrivalSampler.SampleIAT` ignores its `rng` argument

**BC-4: Model Field Propagation**
- GIVEN a ClientSpec with `model: "llama-3.1-8b"`
- WHEN `GenerateRequests` produces requests for that client
- THEN each generated request has `Model == "llama-3.1-8b"`, and the corresponding `RequestMetrics` includes the same model value
- MECHANISM: `generator.go` copies `client.Model` to `req.Model`; `NewRequestMetrics` copies `req.Model`

**BC-5: Model Field Default**
- GIVEN a ClientSpec with no `model` field (empty string)
- WHEN requests are generated
- THEN `Request.Model` is `""` (no model preference), preserving backward compatibility
- MECHANISM: Go zero-value for string

**BC-6: IsValidSLOClass Accessor**
- GIVEN any string value
- WHEN `IsValidSLOClass(value)` is called
- THEN it returns true for v2 tier names {`""`, `critical`, `standard`, `sheddable`, `batch`, `background`} and false for all other values
- MECHANISM: Unexported `validSLOClasses` map with exported accessor (R8 compliance)

**BC-7: Constant Arrival Validation**
- GIVEN a client with `arrival.process: "constant"` and `cv` set
- WHEN `Validate()` is called
- THEN validation succeeds (CV is ignored for constant process; no error)
- MECHANISM: Constant process factory ignores CV parameter

#### Negative Contracts

**BC-8: No Routing Effect from Model Field**
- GIVEN requests with different `Model` values
- WHEN routed through any existing routing policy
- THEN routing decisions are identical to requests without model values (the field is not read by any policy)
- MECHANISM: No routing code reads `Request.Model` in Phase 0

**BC-9: No Process Termination in Library**
- GIVEN any invalid spec input
- WHEN processed by `sim/workload/` functions
- THEN an error is returned; `logrus.Fatalf` is never called (R6 compliance)
- MECHANISM: All functions return `error`; no `Fatalf` in `sim/`

#### Error Handling Contracts

**BC-10: Unknown SLO Class Rejected**
- GIVEN a v2 spec with `slo_class: "premium"`
- WHEN validated
- THEN returns error `unknown slo_class "premium"; valid: critical, standard, sheddable, batch, background, or empty`
- MECHANISM: `validSLOClasses` registry check in `validateClient`

### C) Component Interaction

```
YAML file → [LoadWorkloadSpec] → WorkloadSpec (raw)
                                      ↓
                              [UpgradeV1ToV2] (maps tier names, sets version)
                                      ↓
                              WorkloadSpec (v2)
                                      ↓
                              [Validate] (rejects unknown tiers)
                                      ↓
                              [GenerateRequests] → []*sim.Request (with Model field)
                                      ↓
                              [NewRequestMetrics] → RequestMetrics (with Model field)
```

**API contracts:**
- `UpgradeV1ToV2(spec *WorkloadSpec)` — mutates spec in-place. Idempotent. No error return.
- `IsValidSLOClass(name string) bool` — pure query, no side effects.
- `ConstantArrivalSampler.SampleIAT(rng) int64` — deterministic, ignores rng.

**State changes:** None. All changes are to data types and validation registries. No new mutable state.

**Extension friction:** Adding another SLO tier: 1 line in `validSLOClasses` + docs = 2 touch points. Adding another arrival process: 1 type + factory case + validator entry = 3 touch points.

### D) Deviation Log

No deviations from source document (issue #420 macro plan W0-1 section).

### E) Review Guide

1. **The tricky part:** The v1→v2 auto-upgrade must be idempotent and must run before validation in both `LoadWorkloadSpec` and `GenerateRequests`. If either path is missed, v1 specs with "realtime" will fail validation.
2. **What to scrutinize:** BC-1 (auto-upgrade) and BC-4 (model propagation) — verify all Request construction sites in `generator.go` and `reasoning.go` are updated.
3. **What's safe to skim:** BC-3 (constant sampler) is mechanical — a one-liner that returns a fixed value. BC-6 (IsValidSLOClass) is a trivial accessor.
4. **Known debt:** Two existing tests break from the changes: `TestWorkloadSpec_Validate_ValidSpec_NoError` (uses "realtime") and `TestLoadWorkloadSpec_ValidYAML_LoadsCorrectly` (expects version "1"). Both fixed in Task 5. Also 5 call sites in reasoning_test.go need the new `model` parameter — fixed in Task 4.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- None (all changes to existing files)

**Files to modify:**
- `sim/workload/spec.go` — version upgrade, SLO tiers, model field, IsValidSLOClass, constant arrival validation
- `sim/workload/spec_test.go` — tests for upgrade, validation, model parsing
- `sim/workload/arrival.go` — ConstantArrivalSampler, factory case
- `sim/workload/arrival_test.go` — constant sampler tests
- `sim/workload/generator.go` — call UpgradeV1ToV2, propagate Model field
- `sim/workload/generator_test.go` — model propagation integration test
- `sim/workload/reasoning.go` — propagate Model field in Request construction
- `sim/workload/scenarios.go` — update v1 tier names to v2
- `sim/request.go` — add Model field
- `sim/metrics_utils.go` — add Model to RequestMetrics, update NewRequestMetrics
- `sim/metrics_utils_test.go` — update test for Model field
- `CLAUDE.md` — update SLO tiers, arrival processes, spec.go description

**Key decisions:**
- Auto-upgrade is a separate exported function (`UpgradeV1ToV2`) for testability
- Validator only accepts v2 names (v1 names must go through upgrade first)
- `ConstantArrivalSampler` ignores RNG (deterministic by definition)
- Model field is zero-value safe (empty string = no model preference)

**Confirmation:** No dead code — every field, type, and function is exercised by tests or production code in this PR.

### G) Task Breakdown

---

#### Task 1: SLO Tier Expansion + IsValidSLOClass Accessor

**Contracts Implemented:** BC-2, BC-6, BC-10

**Files:**
- Modify: `sim/workload/spec.go`
- Test: `sim/workload/spec_test.go`

**Step 1: Write failing tests for v2 SLO tier validation**

Context: We expand validSLOClasses to the v2 set and add an IsValidSLOClass accessor.

```go
// In sim/workload/spec_test.go — add these tests

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
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestIsValidSLOClass|TestValidate_V2SLO|TestValidate_UnknownSLO" -v`
Expected: FAIL — `IsValidSLOClass` undefined; "sheddable"/"background" not in valid set

**Step 3: Implement SLO tier expansion and accessor**

In `sim/workload/spec.go`:

Update the `validSLOClasses` map to v2 tier names:
```go
validSLOClasses = map[string]bool{
	"":           true,
	"critical":   true,
	"standard":   true,
	"sheddable":  true,
	"batch":      true,
	"background": true,
}
```

Add the `IsValidSLOClass` accessor:
```go
// IsValidSLOClass reports whether name is a valid v2 SLO class.
// Valid classes: "", "critical", "standard", "sheddable", "batch", "background".
func IsValidSLOClass(name string) bool {
	return validSLOClasses[name]
}
```

Update the error message in `validateClient`:
```go
if !validSLOClasses[c.SLOClass] {
	return fmt.Errorf("%s: unknown slo_class %q; valid: critical, standard, sheddable, batch, background, or empty", prefix, c.SLOClass)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run "TestIsValidSLOClass|TestValidate_V2SLO|TestValidate_UnknownSLO" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/spec.go sim/workload/spec_test.go
git commit -m "feat(workload): expand SLO tiers to v2 set + add IsValidSLOClass accessor (BC-2, BC-6, BC-10)

- Replace v1 SLO classes (realtime, interactive, batch) with v2 set
  (critical, standard, sheddable, batch, background)
- Add IsValidSLOClass() accessor for R8 compliance
- Update validation error messages to list v2 tier names

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: v1→v2 Auto-Upgrade Function

**Contracts Implemented:** BC-1

**Files:**
- Modify: `sim/workload/spec.go`
- Modify: `sim/workload/generator.go`
- Test: `sim/workload/spec_test.go`

**Step 1: Write failing tests for auto-upgrade**

Context: The upgrade function maps v1 tier names to v2 and sets the version field.

```go
// In sim/workload/spec_test.go — add these tests

func TestUpgradeV1ToV2_EmptyVersion_SetsV2(t *testing.T) {
	// BC-1: Empty version auto-upgrades to "2"
	spec := &WorkloadSpec{Version: ""}
	UpgradeV1ToV2(spec)
	if spec.Version != "2" {
		t.Errorf("Version = %q, want %q", spec.Version, "2")
	}
}

func TestUpgradeV1ToV2_V1Version_SetsV2(t *testing.T) {
	// BC-1: Version "1" auto-upgrades to "2"
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
	// BC-1: deprecated "realtime" auto-maps to "critical"
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
	// BC-1: deprecated "interactive" auto-maps to "standard"
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
	// BC-1: Empty SLO class stays empty through upgrade
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
	// BC-1: Calling twice produces the same result
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
	// BC-1: File-loaded v1 spec is auto-upgraded
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
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestUpgradeV1ToV2|TestLoadWorkloadSpec_V1File_AutoUpgraded" -v`
Expected: FAIL — `UpgradeV1ToV2` undefined

**Step 3: Implement auto-upgrade function**

In `sim/workload/spec.go`, add import for `"github.com/sirupsen/logrus"` and add:

```go
// v1ToV2SLOClasses maps deprecated v1 SLO class names to v2 equivalents.
var v1ToV2SLOClasses = map[string]string{
	"realtime":    "critical",
	"interactive": "standard",
}

// UpgradeV1ToV2 auto-upgrades a v1 WorkloadSpec to v2 format in-place.
// Maps deprecated SLO class names (realtime→critical, interactive→standard)
// and sets the version field to "2". Idempotent — calling on a v2 spec is safe.
// Emits logrus.Warn deprecation notices for mapped tier names.
func UpgradeV1ToV2(spec *WorkloadSpec) {
	if spec.Version == "" || spec.Version == "1" {
		spec.Version = "2"
	}
	for i := range spec.Clients {
		if newName, ok := v1ToV2SLOClasses[spec.Clients[i].SLOClass]; ok {
			logrus.Warnf("deprecated SLO class %q auto-mapped to %q; update your spec to use v2 tier names",
				spec.Clients[i].SLOClass, newName)
			spec.Clients[i].SLOClass = newName
		}
	}
}
```

Update `LoadWorkloadSpec` to call upgrade after parsing:
```go
func LoadWorkloadSpec(path string) (*WorkloadSpec, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading workload spec: %w", err)
	}
	var spec WorkloadSpec
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&spec); err != nil {
		return nil, fmt.Errorf("parsing workload spec: %w", err)
	}
	UpgradeV1ToV2(&spec)
	return &spec, nil
}
```

In `sim/workload/generator.go`, add `UpgradeV1ToV2` call before validation:
```go
// Inside GenerateRequests, before the spec.Validate() call:
UpgradeV1ToV2(spec)
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run "TestUpgradeV1ToV2|TestLoadWorkloadSpec_V1File_AutoUpgraded" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/spec.go sim/workload/generator.go sim/workload/spec_test.go
git commit -m "feat(workload): add v1→v2 auto-upgrade for WorkloadSpec (BC-1)

- Add UpgradeV1ToV2() function mapping deprecated SLO class names
  (realtime→critical, interactive→standard) with logrus.Warn
- Call upgrade in LoadWorkloadSpec (after parse) and GenerateRequests
  (before validate) to cover all entry points
- Idempotent: calling on v2 specs is a no-op

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Constant Arrival Sampler

**Contracts Implemented:** BC-3, BC-7

**Files:**
- Modify: `sim/workload/arrival.go`
- Modify: `sim/workload/spec.go` (add "constant" to valid processes)
- Test: `sim/workload/arrival_test.go`
- Test: `sim/workload/spec_test.go`

**Step 1: Write failing tests for constant arrival sampler**

Context: A constant sampler produces fixed-interval IATs, ignoring RNG.

```go
// In sim/workload/arrival_test.go — add these tests

func TestConstantArrivalSampler_ExactIntervals(t *testing.T) {
	// BC-3: Constant sampler produces exact 1/rate intervals
	rate := 10.0 / 1e6 // 10 req/s = 10/1e6 req/us
	sampler := NewArrivalSampler(ArrivalSpec{Process: "constant"}, rate)

	expectedIAT := int64(1.0 / rate) // 100000 us
	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 100; i++ {
		iat := sampler.SampleIAT(rng)
		if iat != expectedIAT {
			t.Fatalf("iteration %d: SampleIAT = %d, want %d", i, iat, expectedIAT)
		}
	}
}

func TestConstantArrivalSampler_DifferentSeeds_SameResult(t *testing.T) {
	// BC-3: Constant sampler is deterministic regardless of RNG state
	rate := 5.0 / 1e6
	sampler := NewArrivalSampler(ArrivalSpec{Process: "constant"}, rate)

	rng1 := rand.New(rand.NewSource(1))
	rng2 := rand.New(rand.NewSource(999))

	for i := 0; i < 50; i++ {
		iat1 := sampler.SampleIAT(rng1)
		iat2 := sampler.SampleIAT(rng2)
		if iat1 != iat2 {
			t.Fatalf("iteration %d: different seeds produced different IATs: %d vs %d", i, iat1, iat2)
		}
	}
}

func TestConstantArrivalSampler_MinimumOneUs(t *testing.T) {
	// BC-3: Floor of 1 microsecond for very high rates
	rate := 1.0 // 1 req/us (extremely high)
	sampler := NewArrivalSampler(ArrivalSpec{Process: "constant"}, rate)

	rng := rand.New(rand.NewSource(42))
	iat := sampler.SampleIAT(rng)
	if iat < 1 {
		t.Errorf("SampleIAT = %d, want >= 1", iat)
	}
}
```

```go
// In sim/workload/spec_test.go — add this test

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
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestConstantArrival|TestValidate_ConstantArrival" -v`
Expected: FAIL — "constant" not a valid arrival process

**Step 3: Implement constant arrival sampler**

In `sim/workload/spec.go`, add "constant" to valid processes:
```go
validArrivalProcesses = map[string]bool{
	"poisson": true, "gamma": true, "weibull": true, "constant": true,
}
```

In `sim/workload/arrival.go`, add the sampler type and factory case:

```go
// ConstantArrivalSampler produces fixed inter-arrival times (zero variance).
// Used for deterministic legacy parity where requests arrive at exact intervals.
type ConstantArrivalSampler struct {
	iatMicros int64 // fixed inter-arrival time in microseconds
}

func (s *ConstantArrivalSampler) SampleIAT(_ *rand.Rand) int64 {
	return s.iatMicros
}
```

Add factory case in `NewArrivalSampler`, before the default:
```go
case "constant":
	iat := int64(1.0 / ratePerMicrosecond)
	if iat < 1 {
		iat = 1
	}
	return &ConstantArrivalSampler{iatMicros: iat}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run "TestConstantArrival|TestValidate_ConstantArrival" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/arrival.go sim/workload/arrival_test.go sim/workload/spec.go sim/workload/spec_test.go
git commit -m "feat(workload): add constant arrival sampler (BC-3, BC-7)

- Add ConstantArrivalSampler producing exact 1/rate interval IATs
- RNG-independent: ignores rng argument (deterministic by definition)
- Add 'constant' to validArrivalProcesses registry
- CV parameter ignored when process is constant

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Model Field on ClientSpec + Request + RequestMetrics

**Contracts Implemented:** BC-4, BC-5, BC-8

**Files:**
- Modify: `sim/workload/spec.go` (ClientSpec.Model)
- Modify: `sim/request.go` (Request.Model)
- Modify: `sim/metrics_utils.go` (RequestMetrics.Model, NewRequestMetrics)
- Modify: `sim/workload/generator.go` (propagate Model)
- Modify: `sim/workload/reasoning.go` (propagate Model)
- Modify: `sim/workload/reasoning_test.go` (update call sites for new parameter)
- NOT modified: `sim/workload/replay.go` — constructs Request from TraceV2 records which have no model field; Model defaults to `""` (correct per BC-5)
- NOT modified: `sim/workload_config.go`, `sim/cluster/workload.go` — legacy paths; Model defaults to `""` (correct per BC-5)
- Test: `sim/workload/spec_test.go`
- Test: `sim/workload/generator_test.go`
- Test: `sim/metrics_utils_test.go`

**Step 1: Write failing tests for model field propagation**

```go
// In sim/workload/spec_test.go — add this test

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
```

```go
// In sim/workload/generator_test.go — add these tests

func TestGenerateRequests_ModelFieldPropagated(t *testing.T) {
	// BC-4: Model field flows from ClientSpec to Request
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", Model: "llama-3.1-8b", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected at least 1 request")
	}
	for i, req := range reqs {
		if req.Model != "llama-3.1-8b" {
			t.Errorf("request[%d].Model = %q, want %q", i, req.Model, "llama-3.1-8b")
		}
	}
}

func TestGenerateRequests_EmptyModel_DefaultsToEmpty(t *testing.T) {
	// BC-5: Empty model field preserved as empty string
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	reqs, err := GenerateRequests(spec, 1_000_000, 5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for i, req := range reqs {
		if req.Model != "" {
			t.Errorf("request[%d].Model = %q, want empty string", i, req.Model)
		}
	}
}
```

```go
// In sim/metrics_utils_test.go — add this test

func TestNewRequestMetrics_IncludesModel(t *testing.T) {
	// BC-4: Model field included in RequestMetrics
	req := &Request{
		ID:          "r1",
		InputTokens: make([]int, 5),
		OutputTokens: make([]int, 10),
		Model:       "llama-3.1-8b",
		SLOClass:    "critical",
	}
	metrics := NewRequestMetrics(req, 1.0)
	if metrics.Model != "llama-3.1-8b" {
		t.Errorf("Model = %q, want %q", metrics.Model, "llama-3.1-8b")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestLoadWorkloadSpec_ModelField|TestGenerateRequests_Model" -v`
Run: `go test ./sim/... -run "TestNewRequestMetrics_IncludesModel" -v`
Expected: FAIL — Model field undefined

**Step 3: Implement model field propagation**

In `sim/workload/spec.go`, add `Model` field to `ClientSpec`:
```go
type ClientSpec struct {
	ID           string        `yaml:"id"`
	TenantID     string        `yaml:"tenant_id"`
	SLOClass     string        `yaml:"slo_class"`
	Model        string        `yaml:"model,omitempty"`
	RateFraction float64       `yaml:"rate_fraction"`
	// ... rest unchanged
}
```

In `sim/request.go`, add `Model` field to `Request`:
```go
// Add after the AssignedInstance field:
Model string // Model tag for multi-model routing (empty = default model). Phase 0: carried, not read by routing.
```

In `sim/metrics_utils.go`, add `Model` to `RequestMetrics` and update `NewRequestMetrics`:
```go
type RequestMetrics struct {
	ArrivedAt        float64 `json:"arrived_at"`
	ID               string  `json:"requestID"`
	NumPrefillTokens int     `json:"num_prefill_tokens"`
	NumDecodeTokens  int     `json:"num_decode_tokens"`
	TTFT             float64 `json:"ttft_ms"`
	ITL              float64 `json:"itl_ms"`
	E2E              float64 `json:"e2e_ms"`
	SchedulingDelay  float64 `json:"scheduling_delay_ms"`
	SLOClass         string  `json:"slo_class,omitempty"`
	TenantID         string  `json:"tenant_id,omitempty"`
	HandledBy        string  `json:"handled_by,omitempty"`
	Model            string  `json:"model,omitempty"`
}

func NewRequestMetrics(req *Request, arrivedAt float64) RequestMetrics {
	return RequestMetrics{
		ID:               req.ID,
		ArrivedAt:        arrivedAt,
		NumPrefillTokens: len(req.InputTokens),
		NumDecodeTokens:  len(req.OutputTokens),
		SLOClass:         req.SLOClass,
		TenantID:         req.TenantID,
		HandledBy:        req.AssignedInstance,
		Model:            req.Model,
	}
}
```

In `sim/workload/generator.go`, add `Model` to the Request construction (around line 175):
```go
req := &sim.Request{
	ID:               "",
	ArrivalTime:      currentTime,
	InputTokens:      inputTokens,
	OutputTokens:     outputTokens,
	State:            sim.StateQueued,
	ScheduledStepIdx: 0,
	FinishedStepIdx:  0,
	TenantID:         client.TenantID,
	SLOClass:         client.SLOClass,
	Streaming:        client.Streaming,
	Model:            client.Model,
	TextTokenCount:   textCount,
	ImageTokenCount:  imageCount,
	AudioTokenCount:  audioCount,
	VideoTokenCount:  videoCount,
}
```

In `sim/workload/reasoning.go`, add `Model` parameter and propagation. Update `GenerateReasoningRequests` signature to accept `model string` and add to the Request construction:
```go
func GenerateReasoningRequests(
	rng *rand.Rand,
	spec *ReasoningSpec,
	inputSampler, outputSampler LengthSampler,
	startTime int64,
	clientID, tenantID, sloClass, model string,
) ([]*sim.Request, error) {
	// ... existing code ...
	req := &sim.Request{
		// ... existing fields ...
		Model:       model,
	}
```

Update the call site in `generator.go` (around line 113):
```go
reasoningReqs, err := GenerateReasoningRequests(
	clientRNG, client.Reasoning,
	inputSampler, outputSampler,
	currentTime,
	client.ID, client.TenantID, client.SLOClass, client.Model,
)
```

Update all call sites in `sim/workload/reasoning_test.go` — add empty string `""` as the new `model` parameter at the end of each `GenerateReasoningRequests(...)` call. There are 5 call sites (lines 20, 63, 90, 117, 158). Example:
```go
// Before:
requests, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c1", "t1", "batch")
// After:
requests, err := GenerateReasoningRequests(rng, spec, inputSampler, outputSampler, 0, "c1", "t1", "batch", "")
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestNewRequestMetrics_IncludesModel" -v`
Run: `go test ./sim/workload/... -run "TestLoadWorkloadSpec_ModelField|TestGenerateRequests_Model" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/... ./sim/workload/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/workload/spec.go sim/request.go sim/metrics_utils.go sim/workload/generator.go sim/workload/reasoning.go sim/workload/reasoning_test.go sim/workload/spec_test.go sim/workload/generator_test.go sim/metrics_utils_test.go
git commit -m "feat(workload): add model field to ClientSpec, Request, RequestMetrics (BC-4, BC-5)

- Add Model string field to ClientSpec (YAML: model, omitempty)
- Add Model string field to Request (zero-value safe)
- Add Model string field to RequestMetrics (included in NewRequestMetrics)
- Propagate model from ClientSpec through generator and reasoning paths
- No routing effect in Phase 0 — field is carried but not read

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: Update Built-in Scenarios + Fix Existing Tests

**Contracts Implemented:** BC-1 (end-to-end proof), BC-2 (existing tests updated)

**Files:**
- Modify: `sim/workload/scenarios.go`
- Modify: `sim/workload/spec_test.go` (fix broken test)
- Test: `sim/workload/scenarios_test.go` (new)

**Step 1: Write test verifying scenarios validate under v2 rules**

Context: Update scenarios to use v2 tier names and verify they pass validation.

```go
// Create sim/workload/scenarios_test.go

package workload

import (
	"testing"
)

func TestScenarios_ValidateUnderV2Rules(t *testing.T) {
	// Verify all built-in scenarios pass v2 validation after auto-upgrade
	scenarios := []struct {
		name string
		spec *WorkloadSpec
	}{
		{"BurstyTraffic", ScenarioBurstyTraffic(42, 10.0)},
		{"UnfairTenants", ScenarioUnfairTenants(42, 10.0)},
		{"PrefixHeavy", ScenarioPrefixHeavy(42, 10.0)},
		{"MixedSLO", ScenarioMixedSLO(42, 10.0)},
	}
	for _, tc := range scenarios {
		t.Run(tc.name, func(t *testing.T) {
			UpgradeV1ToV2(tc.spec)
			if err := tc.spec.Validate(); err != nil {
				t.Errorf("scenario %s failed validation after upgrade: %v", tc.name, err)
			}
		})
	}
}

func TestScenarios_UseV2TierNames(t *testing.T) {
	// Verify scenarios use v2 tier names directly (no upgrade needed)
	validV2Tiers := map[string]bool{
		"": true, "critical": true, "standard": true, "sheddable": true, "batch": true, "background": true,
	}

	scenarios := []*WorkloadSpec{
		ScenarioBurstyTraffic(42, 10.0),
		ScenarioUnfairTenants(42, 10.0),
		ScenarioPrefixHeavy(42, 10.0),
		ScenarioMixedSLO(42, 10.0),
	}
	for _, spec := range scenarios {
		for _, c := range spec.Clients {
			if !validV2Tiers[c.SLOClass] {
				t.Errorf("client %q uses non-v2 SLO class %q", c.ID, c.SLOClass)
			}
		}
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/workload/... -run "TestScenarios_UseV2TierNames" -v`
Expected: FAIL — scenarios still use "realtime", "interactive"

**Step 3: Update scenarios and fix existing test**

In `sim/workload/scenarios.go`, update all SLO class names:
- `ScenarioBurstyTraffic`: "batch" → "batch" (unchanged)
- `ScenarioUnfairTenants`: "batch" → "batch", "realtime" → "critical"
- `ScenarioPrefixHeavy`: "batch" → "batch", "interactive" → "standard"
- `ScenarioMixedSLO`: "realtime" → "critical", "interactive" → "standard", "batch" → "batch"

In `sim/workload/spec_test.go`, fix two existing tests:

1. `TestWorkloadSpec_Validate_ValidSpec_NoError`: Change `SLOClass: "realtime"` to `SLOClass: "critical"` on client c2.

2. `TestLoadWorkloadSpec_ValidYAML_LoadsCorrectly`: Change the version assertion from `spec.Version != "1"` to `spec.Version != "2"` and update the want message, because LoadWorkloadSpec now auto-upgrades v1 to v2.

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/workload/... -run "TestScenarios|TestWorkloadSpec_Validate_ValidSpec" -v`
Expected: PASS

**Step 5: Run full test suite to check for regressions**

Run: `go test ./sim/workload/... -v`
Expected: All tests pass (no regressions)

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 7: Commit**

```bash
git add sim/workload/scenarios.go sim/workload/scenarios_test.go sim/workload/spec_test.go
git commit -m "refactor(workload): update built-in scenarios to v2 SLO tier names (BC-1, BC-2)

- Update ScenarioUnfairTenants: realtime → critical
- Update ScenarioPrefixHeavy: interactive → standard
- Update ScenarioMixedSLO: realtime → critical, interactive → standard
- Fix existing test to use v2 tier name 'critical' instead of 'realtime'
- Add scenario validation tests

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: RNG Subsystem Consistency Test (Blocking Prerequisite)

**Contracts Implemented:** Validation gate (macro plan requirement)

**Files:**
- Test: `sim/workload/generator_test.go`

**Step 1: Write RNG subsystem consistency test**

Context: This test verifies the RNG subsystem assignments documented in the design doc Section 4 are accurate. It's a blocking prerequisite for all Phase 0 PRs.

```go
// In sim/workload/generator_test.go — add this test

func TestRNGSubsystemConsistencyAcrossLegacyPaths(t *testing.T) {
	// Blocking prerequisite: Verify that sim/workload/ path uses
	// SubsystemWorkloadGen as documented in the design doc.
	// This ensures the unification in W0-4 can rely on known RNG behavior.

	// Generate requests twice with the same seed — output must be identical.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}

	reqs1, err := GenerateRequests(spec, 1_000_000, 10)
	if err != nil {
		t.Fatalf("first generation: %v", err)
	}

	// Reset spec (GenerateRequests may have mutated it via UpgradeV1ToV2)
	spec2 := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}

	reqs2, err := GenerateRequests(spec2, 1_000_000, 10)
	if err != nil {
		t.Fatalf("second generation: %v", err)
	}

	// INV-6: Same seed must produce identical requests
	if len(reqs1) != len(reqs2) {
		t.Fatalf("request count mismatch: %d vs %d", len(reqs1), len(reqs2))
	}
	for i := range reqs1 {
		if reqs1[i].ArrivalTime != reqs2[i].ArrivalTime {
			t.Errorf("request[%d] arrival time mismatch: %d vs %d", i, reqs1[i].ArrivalTime, reqs2[i].ArrivalTime)
		}
		if len(reqs1[i].InputTokens) != len(reqs2[i].InputTokens) {
			t.Errorf("request[%d] input token count mismatch: %d vs %d", i, len(reqs1[i].InputTokens), len(reqs2[i].InputTokens))
		}
		if len(reqs1[i].OutputTokens) != len(reqs2[i].OutputTokens) {
			t.Errorf("request[%d] output token count mismatch: %d vs %d", i, len(reqs1[i].OutputTokens), len(reqs2[i].OutputTokens))
		}
	}
}
```

**Step 2: Run test**

Run: `go test ./sim/workload/... -run "TestRNGSubsystemConsistency" -v`
Expected: PASS (this test verifies existing behavior is deterministic)

**Step 3: No implementation needed (verification test only)**

**Step 4: Commit**

```bash
git add sim/workload/generator_test.go
git commit -m "test(workload): add RNG subsystem consistency test (blocking prerequisite)

- Verify GenerateRequests produces identical output for identical seed
- Validates INV-6 determinism through the sim/workload/ path
- Blocking prerequisite for Phase 0 unification (design doc Section 4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 7: v1/v2 Round-Trip Integration Test

**Contracts Implemented:** BC-1 (end-to-end), BC-3 (constant sampler integration)

**Files:**
- Test: `sim/workload/generator_test.go`

**Step 1: Write v1/v2 round-trip integration test**

Context: End-to-end test: load a v1 spec → auto-upgrade → generate → verify determinism and model/SLO propagation.

```go
// In sim/workload/generator_test.go — add these tests

func TestGenerateRequests_V1SpecAutoUpgrade_EndToEnd(t *testing.T) {
	// BC-1 end-to-end: v1 spec with deprecated tier names auto-upgrades and generates
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{
			{
				ID: "realtime-client", RateFraction: 0.5, SLOClass: "realtime",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			},
			{
				ID: "interactive-client", RateFraction: 0.5, SLOClass: "interactive",
				Model: "llama-3.1-8b",
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 200}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
			},
		},
	}

	reqs, err := GenerateRequests(spec, 1_000_000, 20)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) == 0 {
		t.Fatal("expected requests to be generated")
	}

	// Verify SLO classes were upgraded
	hasCritical, hasStandard := false, false
	for _, req := range reqs {
		switch req.SLOClass {
		case "critical":
			hasCritical = true
		case "standard":
			hasStandard = true
		case "realtime", "interactive":
			t.Errorf("found deprecated SLO class %q — should have been upgraded", req.SLOClass)
		}
	}
	if !hasCritical {
		t.Error("expected at least one request with SLOClass 'critical'")
	}
	if !hasStandard {
		t.Error("expected at least one request with SLOClass 'standard'")
	}

	// Verify model propagation
	hasModel := false
	for _, req := range reqs {
		if req.Model == "llama-3.1-8b" {
			hasModel = true
			break
		}
	}
	if !hasModel {
		t.Error("expected at least one request with Model 'llama-3.1-8b'")
	}

	// Verify spec was upgraded
	if spec.Version != "2" {
		t.Errorf("spec.Version = %q, want %q", spec.Version, "2")
	}
}

func TestGenerateRequests_ConstantArrival_EvenSpacing(t *testing.T) {
	// BC-3 integration: Constant sampler produces evenly-spaced requests
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0, // 10 req/s
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "constant"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		}},
	}

	reqs, err := GenerateRequests(spec, 1_000_000, 5) // 1 second, 5 requests max
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(reqs) < 2 {
		t.Fatalf("expected at least 2 requests, got %d", len(reqs))
	}

	// Verify even spacing: all IATs should be identical
	expectedIAT := reqs[1].ArrivalTime - reqs[0].ArrivalTime
	for i := 2; i < len(reqs); i++ {
		iat := reqs[i].ArrivalTime - reqs[i-1].ArrivalTime
		if iat != expectedIAT {
			t.Errorf("request[%d] IAT = %d, want %d (constant spacing)", i, iat, expectedIAT)
		}
	}
}

func TestGenerateRequests_V2NewSLOTiers_Generate(t *testing.T) {
	// BC-2: New v2 SLO tiers generate successfully
	tiers := []string{"critical", "standard", "sheddable", "batch", "background"}
	for _, tier := range tiers {
		spec := &WorkloadSpec{
			Version: "2", Seed: 42, AggregateRate: 10.0,
			Clients: []ClientSpec{{
				ID: "c1", RateFraction: 1.0, SLOClass: tier,
				Arrival:    ArrivalSpec{Process: "poisson"},
				InputDist:  DistSpec{Type: "exponential", Params: map[string]float64{"mean": 100}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
			}},
		}
		reqs, err := GenerateRequests(spec, 1_000_000, 5)
		if err != nil {
			t.Errorf("SLO tier %q: unexpected error: %v", tier, err)
			continue
		}
		for _, req := range reqs {
			if req.SLOClass != tier {
				t.Errorf("SLO tier %q: request has SLOClass %q", tier, req.SLOClass)
			}
		}
	}
}
```

**Step 2: Run tests**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_V1Spec|TestGenerateRequests_ConstantArrival_EvenSpacing|TestGenerateRequests_V2NewSLO" -v`
Expected: PASS

**Step 3: No additional implementation needed (integration tests only)**

**Step 4: Run full test suite**

Run: `go test ./sim/workload/... -v`
Expected: All tests pass

Run: `go test ./sim/... -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add sim/workload/generator_test.go
git commit -m "test(workload): add v1/v2 round-trip and constant arrival integration tests (BC-1, BC-3)

- End-to-end: v1 spec auto-upgrades through GenerateRequests pipeline
- Verify SLO class upgrade, model propagation in generated requests
- Constant arrival sampler produces evenly-spaced requests
- All v2 SLO tiers generate successfully

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 8: CLAUDE.md Documentation Updates

**Contracts Implemented:** Documentation (not contract-backed)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Update the following sections:

1. **File Organization → sim/workload/spec.go description**: Add mention of v2 schema, IsValidSLOClass, UpgradeV1ToV2, model field
2. **File Organization → sim/workload/arrival.go description**: Add mention of Constant arrival sampler
3. **Current Implementation Focus**: Add mention of Phase 0 workload unification starting (W0-1 complete)

In the `spec.go` line of the File Organization tree, update:
```
│   ├── spec.go                # WorkloadSpec v2, ClientSpec (with Model field), ArrivalSpec, DistSpec, YAML loading, v1→v2 auto-upgrade (UpgradeV1ToV2), IsValidSLOClass accessor
```

In the `arrival.go` line:
```
│   ├── arrival.go             # ArrivalSampler: Poisson, Gamma (Marsaglia-Tsang), Weibull (bisection), Constant (fixed-interval)
```

In Current Implementation Focus, add after the scorer framework sentence:
```
Phase 0 workload unification in progress (see issue #420): W0-1 (spec v2 schema + SLO tiers) complete. SLO tiers: critical, standard, sheddable, batch, background. Arrival processes: poisson, gamma, weibull, constant.
```

**Step 2: Run tests (sanity check)**

Run: `go test ./... -count=1`
Expected: All tests pass

**Step 3: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for W0-1 spec v2 schema

- Update spec.go description: v2 schema, IsValidSLOClass, UpgradeV1ToV2
- Update arrival.go description: add Constant sampler
- Update Current Implementation Focus: Phase 0 W0-1 status
- Update SLO tier list and arrival process list

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestUpgradeV1ToV2_* (7 tests) |
| BC-1 | Task 2 | Unit | TestUpgradeV1ToV2_EmptySLOClassUnchanged |
| BC-1 | Task 2 | Integration | TestLoadWorkloadSpec_V1File_AutoUpgradedToV2 |
| BC-1 | Task 7 | Integration | TestGenerateRequests_V1SpecAutoUpgrade_EndToEnd |
| BC-2 | Task 1 | Unit | TestValidate_V2SLOTiers_NoError |
| BC-2 | Task 7 | Integration | TestGenerateRequests_V2NewSLOTiers_Generate |
| BC-3 | Task 3 | Unit | TestConstantArrivalSampler_ExactIntervals |
| BC-3 | Task 3 | Unit | TestConstantArrivalSampler_DifferentSeeds_SameResult |
| BC-3 | Task 3 | Unit | TestConstantArrivalSampler_MinimumOneUs |
| BC-3 | Task 7 | Integration | TestGenerateRequests_ConstantArrival_EvenSpacing |
| BC-4 | Task 4 | Unit | TestLoadWorkloadSpec_ModelField_Parsed |
| BC-4 | Task 4 | Unit | TestGenerateRequests_ModelFieldPropagated |
| BC-4 | Task 4 | Unit | TestNewRequestMetrics_IncludesModel |
| BC-5 | Task 4 | Unit | TestGenerateRequests_EmptyModel_DefaultsToEmpty |
| BC-6 | Task 1 | Unit | TestIsValidSLOClass_V2Tiers_ReturnsTrue |
| BC-6 | Task 1 | Unit | TestIsValidSLOClass_Invalid_ReturnsFalse |
| BC-7 | Task 3 | Unit | TestValidate_ConstantArrival_NoError |
| BC-10 | Task 1 | Unit | TestValidate_UnknownSLOTier_ReturnsError |
| INV-6 | Task 6 | Invariant | TestRNGSubsystemConsistencyAcrossLegacyPaths |

**Golden dataset:** Not affected — this PR adds no new output metrics or format changes. Golden dataset regeneration deferred to W0-4.

**Shared test infrastructure:** No new shared helpers needed. Existing `parseWorkloadSpecFromBytes` helper reused.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| v1 specs fail validation if upgrade path missed | Low | High | UpgradeV1ToV2 called in both LoadWorkloadSpec and GenerateRequests | Task 2 |
| Existing tests break from SLO tier change | Medium | Low | Task 5 explicitly fixes broken test; scenarios updated | Task 5 |
| Model field breaks strict YAML parsing for existing specs | Low | Medium | Field uses `omitempty` — absent fields parsed as zero value | Task 4 |
| ConstantArrivalSampler edge case at very high rates | Low | Low | Floor of 1 microsecond prevents zero/negative IAT | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — UpgradeV1ToV2 is a function, not an interface
- [x] No feature creep — no cohorts, no converters, no legacy retirement
- [x] No unexercised flags or interfaces — all new code tested
- [x] No partial implementations — model field flows end-to-end
- [x] No breaking changes without contract updates — v1 auto-upgrade preserves compatibility
- [x] No hidden global state impact — only validation registries (maps) updated
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (parseWorkloadSpecFromBytes)
- [x] CLAUDE.md updated (Task 8)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY — no canonical sources modified
- [x] Deviation log reviewed — no deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2, 3 independent, 4 independent, 5 depends on 1+2)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration: N/A (no output changes)
- [x] Construction site audit: Request struct gains Model field — 5 production sites identified (generator.go, reasoning.go, replay.go, workload_config.go, cluster/workload.go); 2 updated in this PR (generator.go, reasoning.go); 3 legacy sites use zero-value (correct for empty model)

**Antipattern rules:**
- [x] R1: No silent continue/return dropping data
- [x] R2: No map iteration affecting output order (validSLOClasses used for membership test only)
- [x] R3: N/A — no new CLI flags
- [x] R4: Request construction sites audited — Model field is zero-value safe
- [x] R5: N/A — no resource allocation loops
- [x] R6: No Fatalf in sim/ — UpgradeV1ToV2 uses logrus.Warn (stderr only)
- [x] R7: INV-6 invariant test added (Task 6)
- [x] R8: IsValidSLOClass accessor added; validSLOClasses remains unexported
- [x] R9: N/A — no new YAML float fields where zero is valid
- [x] R10: Existing strict YAML parsing preserved; model field uses omitempty
- [x] R11: N/A — no new division
- [x] R12: N/A — golden dataset unaffected
- [x] R13: N/A — no new interfaces
- [x] R14: N/A — no multi-module methods
- [x] R15: N/A — no PR references to clean up
- [x] R16: N/A — no new config parameters
- [x] R17: N/A — no new routing signals

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/spec.go`

**Purpose:** Add v2 SLO tiers, UpgradeV1ToV2, IsValidSLOClass, model field on ClientSpec.

**Changes:**
1. Add `Model` field to `ClientSpec` struct (after SLOClass)
2. Update `validSLOClasses` map to v2 set
3. Add `v1ToV2SLOClasses` mapping
4. Add `UpgradeV1ToV2(spec *WorkloadSpec)` function
5. Add `IsValidSLOClass(name string) bool` accessor
6. Update `validateClient` error message for SLO classes
7. Add "constant" to `validArrivalProcesses`
8. Call `UpgradeV1ToV2` in `LoadWorkloadSpec`
9. Add import for `"github.com/sirupsen/logrus"`

### File: `sim/workload/arrival.go`

**Purpose:** Add ConstantArrivalSampler.

**Changes:**
1. Add `ConstantArrivalSampler` struct with `iatMicros int64` field
2. Add `SampleIAT(_ *rand.Rand) int64` method (returns fixed value)
3. Add `case "constant":` to `NewArrivalSampler` factory

### File: `sim/request.go`

**Purpose:** Add Model field to Request struct.

**Changes:**
1. Add `Model string` field after `AssignedInstance` with doc comment

### File: `sim/metrics_utils.go`

**Purpose:** Add Model to RequestMetrics and NewRequestMetrics.

**Changes:**
1. Add `Model string` field to `RequestMetrics` struct (json:"model,omitempty")
2. Add `Model: req.Model` to `NewRequestMetrics` constructor

### File: `sim/workload/generator.go`

**Purpose:** Call UpgradeV1ToV2, propagate Model field.

**Changes:**
1. Add `UpgradeV1ToV2(spec)` call before `spec.Validate()` (around line 45)
2. Add `Model: client.Model` to Request construction (around line 175)
3. Add `client.Model` parameter to `GenerateReasoningRequests` call (around line 117)

### File: `sim/workload/reasoning.go`

**Purpose:** Accept and propagate model parameter.

**Changes:**
1. Add `model string` parameter to `GenerateReasoningRequests` signature
2. Add `Model: model` to Request construction (around line 68)

### File: `sim/workload/scenarios.go`

**Purpose:** Update SLO tier names to v2.

**Changes:**
1. `ScenarioUnfairTenants`: "realtime" → "critical"
2. `ScenarioPrefixHeavy`: "interactive" → "standard"
3. `ScenarioMixedSLO`: "realtime" → "critical", "interactive" → "standard"
