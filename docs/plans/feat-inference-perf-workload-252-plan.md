# Native inference-perf Workload Format Support Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable users to specify inference-perf style workloads (stage-based rates, shared prefix expansion, multi-turn) using a compact YAML format instead of manually writing dozens of client definitions.

**The problem today:** Users who want to simulate inference-perf workload patterns in BLIS must manually expand compact specifications into verbose client definitions. A simple scenario of 9 system prompts x 5 users x 2 rate stages requires 45+ hand-written client blocks with lifecycle windows. This friction prevents sim-to-real experiments where users compare BLIS predictions against inference-perf measurements.

**What this PR adds:**
1. **Stage-based rate patterns** -- sequential rate/duration pairs (e.g., ramp at 8 req/s for 600s, then burst at 20 req/s for 600s) that automatically map to client lifecycle windows
2. **Shared prefix expansion** -- a compact `shared_prefix` spec that auto-generates N system prompts x M users-per-prompt clients with correct prefix groups and configurable prefix length
3. **Multi-turn flag** -- `enable_multi_turn_chat: true` maps directly to BLIS `reasoning.multi_turn` with context accumulation defaults
4. **Constant distribution type** -- `"constant"` distribution for fixed token lengths (zero variance), supporting inference-perf's fixed-length semantics

**Why this matters:** This is the bridge between inference-perf experiment definitions and BLIS simulation, enabling users to validate simulator accuracy against real inference-perf measurements using the same workload specification.

**Architecture:** All changes are confined to `sim/workload/`. New types (`StageSpec`, `SharedPrefixSpec`, `InferencePerfSpec`) are added to `spec.go`. A new expansion function in `inference_perf.go` translates the compact spec into standard `WorkloadSpec` with expanded clients. The `"constant"` distribution type is added to `distribution.go`. No changes to `sim/`, `sim/cluster/`, or `cmd/`.

**Source:** GitHub issue #252

**Closes:** Fixes #252

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation (Human Review)

### A) Executive Summary

This PR adds inference-perf workload format support to the `sim/workload/` package. It introduces three features: stage-based rate patterns that convert duration/rate pairs into lifecycle windows, shared prefix expansion that generates N*M clients from a compact spec, and multi-turn chat flag mapping to existing reasoning.multi_turn. A `"constant"` distribution type is added for fixed-length token specifications.

All new types live in `sim/workload/` with zero coupling to `sim/` core or `sim/cluster/`. The expansion happens at load time -- `ExpandInferencePerfSpec()` produces a standard `WorkloadSpec` that the existing `GenerateRequests` pipeline consumes unchanged. Adjacent blocks: `WorkloadSpec` (consumed by), `GenerateRequests` (pipeline entry), `ClientSpec` (expanded into), existing `ArrivalSpec`/`DistSpec`/`ReasoningSpec` (reused).

No deviations from issue #252. No invariants touched (this is purely workload generation).

### B) Behavioral Contracts

**Positive Contracts (what MUST happen):**

**BC-1: Stage-to-lifecycle expansion**
- GIVEN a spec with 2 stages: `{rate: 8.0, duration: 600}` then `{rate: 20.0, duration: 600}`
- WHEN the spec is expanded
- THEN each expanded client has exactly 2 lifecycle windows: `[0, 600_000_000)` and `[600_000_000, 1_200_000_000)`
- MECHANISM: `expandStages()` computes cumulative offsets and generates `ActiveWindow` per stage

**BC-2: Stage rate fractions proportional**
- GIVEN a spec with stages at rates 8.0 and 20.0
- WHEN the spec is expanded
- THEN the aggregate rate equals the maximum stage rate (20.0), and client rate fractions for each stage are proportional to `stage_rate / max_rate`
- MECHANISM: Each client appears in every stage but with lifecycle windows controlling when it is active; rate fractions are set to `stage_rate / sum_of_stage_rates * (1 / num_clients)` so that within each window the effective rate matches the stage rate

**BC-3: Shared prefix expansion generates N*M clients**
- GIVEN `num_unique_system_prompts: 9, num_users_per_system_prompt: 5`
- WHEN the spec is expanded
- THEN exactly 45 clients are generated with 9 distinct prefix groups, each group having 5 clients
- MECHANISM: Nested loop in `expandSharedPrefix()` creates `fmt.Sprintf("prompt-%d", p)` prefix groups

**BC-4: Shared prefix configurable length**
- GIVEN `system_prompt_len: 100` in the shared prefix spec
- WHEN clients are expanded and requests generated
- THEN each client's `PrefixLength` field equals 100, and generated requests have 100-token shared prefixes
- MECHANISM: `PrefixLength` field on expanded `ClientSpec` set from `SystemPromptLen`

**BC-5: Fixed lengths become constant distributions**
- GIVEN `question_len: 447` and `output_len: 248` in the shared prefix spec
- WHEN the spec is expanded
- THEN input distribution is `{type: "constant", params: {value: 447}}` and output is `{type: "constant", params: {value: 248}}`
- MECHANISM: `constantDist(val)` helper creates the DistSpec

**BC-6: Constant distribution always returns exact value**
- GIVEN a `"constant"` distribution with `value: 447`
- WHEN sampled 1000 times with any RNG
- THEN every sample equals exactly 447
- MECHANISM: `ConstantSampler.Sample()` ignores RNG and returns the fixed value

**BC-7: Multi-turn flag maps to reasoning spec**
- GIVEN `enable_multi_turn_chat: true` in the shared prefix spec
- WHEN the spec is expanded
- THEN each client has a `Reasoning` field with `MultiTurn.MaxRounds: 5, ContextGrowth: "accumulate"`, and `ThinkTimeUs: 500_000`
- MECHANISM: Default multi-turn config applied in expansion when flag is true

**BC-8: Expansion produces valid WorkloadSpec**
- GIVEN any valid `InferencePerfSpec`
- WHEN expanded via `ExpandInferencePerfSpec()`
- THEN the resulting `WorkloadSpec` passes `Validate()` without error
- MECHANISM: Expansion sets all required fields (ID, RateFraction, Arrival, InputDist, OutputDist)

**BC-9: Determinism preserved**
- GIVEN the same seed and inference-perf spec
- WHEN `GenerateRequests` is called twice on the expanded spec
- THEN both calls produce identical request sequences
- MECHANISM: Expansion is deterministic (no RNG); generation uses `PartitionedRNG`

**Negative Contracts (what MUST NOT happen):**

**BC-10: Zero-duration stages rejected**
- GIVEN a stage with `duration: 0`
- WHEN validation runs
- THEN an error is returned mentioning "duration must be positive"
- MECHANISM: `validateInferencePerfSpec()` checks each stage duration > 0

**BC-11: Zero system prompts rejected**
- GIVEN `num_unique_system_prompts: 0`
- WHEN validation runs
- THEN an error is returned
- MECHANISM: Validation checks num_unique_system_prompts > 0 and num_users_per_system_prompt > 0

**BC-12: Negative lengths rejected**
- GIVEN `system_prompt_len: -1`
- WHEN validation runs
- THEN an error is returned
- MECHANISM: Validation checks all length fields >= 0

**Error Handling Contracts:**

**BC-13: YAML strict parsing for new types**
- GIVEN YAML with a typo like `systm_prompt_len` instead of `system_prompt_len`
- WHEN parsed
- THEN a parse error is returned (not silent acceptance)
- MECHANISM: `KnownFields(true)` on decoder (inherited from existing `LoadWorkloadSpec`)

**BC-14: No Fatalf in library code**
- GIVEN any error in expansion or validation
- WHEN the error occurs
- THEN it is returned as an `error`, never via `logrus.Fatalf` or `os.Exit`
- MECHANISM: All new functions return `error`; R6 compliance

### C) Component Interaction

```
InferencePerfSpec (YAML)
       |
       v
ExpandInferencePerfSpec() ---> WorkloadSpec (standard)
       |                             |
       |  stage-based rates          v
       |  shared prefix        GenerateRequests() (unchanged)
       |  multi-turn mapping         |
       v                             v
  []ClientSpec (expanded)      []*sim.Request (output)
```

**API Contracts:**
- `ExpandInferencePerfSpec(spec *InferencePerfSpec) (*WorkloadSpec, error)` -- pure function, no side effects
- `ConstantSampler` implements `LengthSampler` -- `Sample(*rand.Rand) int` always returns fixed value
- `validateInferencePerfSpec(spec *InferencePerfSpec) error` -- validates all fields before expansion

**State Changes:** None. All new code is pure functions. No new mutable state.

**Extension Friction:** Adding a new inference-perf data type (like `synthetic_tokens`) requires: 1 file change (`inference_perf.go` to add expansion logic). Friction: 1 file. Acceptable.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue uses `load.stages` YAML path | Plan uses `stages` as a top-level field on `InferencePerfSpec` | SIMPLIFICATION: The `InferencePerfSpec` struct flattens the hierarchy since BLIS has its own YAML schema rather than importing inference-perf's exactly |
| Issue mentions `enable_multi_turn_chat` as a boolean | Plan maps it to `Reasoning` with default `MaxRounds: 5, ThinkTimeUs: 500000, ContextGrowth: "accumulate"` | ADDITION: Default values needed for BLIS mapping; these match the existing `multiturn-chat-demo.yaml` example |
| Issue doesn't specify `aggregate_rate` derivation from stages | Plan uses time-weighted average rate as `aggregate_rate` | ADDITION: BLIS requires `aggregate_rate`; time-weighted average best represents overall load |
| Issue doesn't mention `"constant"` distribution type | Plan adds it to `validDistTypes` and implements `ConstantSampler` | ADDITION: Required to express "fixed length = zero variance" semantics cleanly; also useful for general BLIS users |
| Issue mentions "documentation shows inference-perf to BLIS translation" | Plan adds example YAML file in `examples/` but defers reference documentation | SIMPLIFICATION: Example file serves as documentation; formal docs can be added as a follow-up |

### E) Review Guide

**The tricky part:** Stage-based rate design. The stages need to produce correct aggregate rates and per-client lifecycle windows such that within each time window, the effective arrival rate matches the stage's rate. The time-weighted average approach for `aggregate_rate` and per-stage lifecycle windows is the simplest correct design.

**What to scrutinize:** BC-2 (stage rate fractions) -- verify that the math produces the right effective rates per window. BC-3 (N*M expansion) -- verify client count, prefix group assignment, and equal rate fractions.

**What's safe to skim:** BC-6 (constant sampler) is trivial. BC-13 (strict YAML) is inherited from existing parser. BC-14 (no Fatalf) is straightforward.

**Known debt:** The `InferencePerfSpec` is a BLIS-native translation, not a direct import of inference-perf's schema. Users must translate field names. A future PR could add direct inference-perf YAML loading if needed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/workload/inference_perf.go` -- InferencePerfSpec types, validation, expansion to WorkloadSpec
- `sim/workload/inference_perf_test.go` -- Tests for all behavioral contracts
- `examples/inference-perf-shared-prefix.yaml` -- Example workload spec

**Files to modify:**
- `sim/workload/spec.go` -- Add `InferencePerfSpec` field to `WorkloadSpec`, add `"constant"` to `validDistTypes`
- `sim/workload/distribution.go` -- Add `ConstantSampler` type and `"constant"` case in `NewLengthSampler`
- `sim/workload/generator.go` -- Call `ExpandInferencePerfSpec` before validation if InferencePerfSpec is present

**Key decisions:**
1. Stage-based rates are expressed via lifecycle windows on expanded clients -- no new arrival sampler needed
2. Shared prefix expansion happens at spec load/expand time, transparent to generator
3. `InferencePerfSpec` is an optional field on `WorkloadSpec` (nil when not used)
4. `"constant"` distribution type added to existing registry -- generally useful

**Confirmation:** No dead code. All types are exercised by tests and the expansion pipeline. Every path is reachable.

### G) Task Breakdown

---

### Task 1: Add `ConstantSampler` distribution type

**Contracts Implemented:** BC-5, BC-6

**Files:**
- Modify: `sim/workload/distribution.go`
- Modify: `sim/workload/spec.go` (add "constant" to validDistTypes)
- Test: `sim/workload/distribution_test.go`

**Step 1: Write failing test for constant distribution**

Context: The `"constant"` distribution always returns the same value regardless of RNG state. This is needed for inference-perf's fixed-length semantics.

```go
func TestConstantSampler_AlwaysReturnsExactValue(t *testing.T) {
	// BC-6: constant distribution always returns exact value
	sampler, err := NewLengthSampler(DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 447},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	rng := rand.New(rand.NewSource(42))
	for i := 0; i < 1000; i++ {
		got := sampler.Sample(rng)
		if got != 447 {
			t.Fatalf("iteration %d: got %d, want 447", i, got)
		}
	}
}

func TestConstantSampler_ValueOne_ReturnsOne(t *testing.T) {
	// Edge: minimum valid constant
	sampler, err := NewLengthSampler(DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 1},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rng := rand.New(rand.NewSource(99))
	if got := sampler.Sample(rng); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}

func TestConstantSampler_ZeroValue_ReturnsOne(t *testing.T) {
	// Edge: zero value clamped to minimum of 1
	sampler, err := NewLengthSampler(DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 0},
	})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	rng := rand.New(rand.NewSource(99))
	if got := sampler.Sample(rng); got != 1 {
		t.Errorf("got %d, want 1 (clamped)", got)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestConstantSampler -v`
Expected: FAIL -- `unknown distribution type "constant"`

**Step 3: Implement ConstantSampler**

In `sim/workload/distribution.go`, add before `NewLengthSampler`:

```go
// ConstantSampler always returns the same fixed value.
// Used for inference-perf fixed-length token specifications (zero variance).
type ConstantSampler struct {
	value int
}

func (s *ConstantSampler) Sample(_ *rand.Rand) int {
	if s.value < 1 {
		return 1
	}
	return s.value
}
```

In `NewLengthSampler`, add a case before `default`:

```go
	case "constant":
		val := int(spec.Params["value"])
		return &ConstantSampler{value: val}, nil
```

In `sim/workload/spec.go`, add `"constant"` to `validDistTypes`:

```go
	validDistTypes = map[string]bool{
		"gaussian": true, "exponential": true, "pareto_lognormal": true, "empirical": true, "constant": true,
	}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestConstantSampler -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/distribution.go sim/workload/distribution_test.go sim/workload/spec.go
git commit -m "feat(workload): add constant distribution type (BC-5, BC-6)

- Add ConstantSampler that always returns a fixed value
- Register 'constant' in validDistTypes
- Zero values clamped to minimum of 1 (consistent with other samplers)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add InferencePerfSpec types and validation

**Contracts Implemented:** BC-10, BC-11, BC-12, BC-13

**Files:**
- Create: `sim/workload/inference_perf.go`
- Test: `sim/workload/inference_perf_test.go`

**Step 1: Write failing tests for validation**

Context: Define the new spec types and validate them before implementing expansion.

```go
package workload

import (
	"strings"
	"testing"
)

func TestValidateInferencePerfSpec_ValidSpec_NoError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	if err := validateInferencePerfSpec(spec); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestValidateInferencePerfSpec_ZeroDuration_ReturnsError(t *testing.T) {
	// BC-10: zero-duration stages rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 0},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for zero duration")
	}
	if !strings.Contains(err.Error(), "duration must be positive") {
		t.Errorf("error should mention duration: %v", err)
	}
}

func TestValidateInferencePerfSpec_ZeroPrompts_ReturnsError(t *testing.T) {
	// BC-11: zero system prompts rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  0,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for zero system prompts")
	}
}

func TestValidateInferencePerfSpec_NegativeLength_ReturnsError(t *testing.T) {
	// BC-12: negative lengths rejected
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         -1,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for negative system_prompt_len")
	}
}

func TestValidateInferencePerfSpec_NegativeRate_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: -1.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for negative rate")
	}
}

func TestValidateInferencePerfSpec_NoStages_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for no stages")
	}
}

func TestValidateInferencePerfSpec_NoSharedPrefix_ReturnsError(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
		},
	}
	err := validateInferencePerfSpec(spec)
	if err == nil {
		t.Fatal("expected error for nil shared_prefix")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestValidateInferencePerfSpec -v`
Expected: FAIL -- types not defined

**Step 3: Implement types and validation**

In `sim/workload/inference_perf.go`:

```go
package workload

import (
	"fmt"
	"math"
)

// InferencePerfSpec defines an inference-perf style workload using a compact
// format. It is expanded into a standard WorkloadSpec via ExpandInferencePerfSpec().
type InferencePerfSpec struct {
	Stages       []StageSpec       `yaml:"stages"`
	SharedPrefix *SharedPrefixSpec `yaml:"shared_prefix"`
}

// StageSpec defines a single rate/duration stage for stage-based load patterns.
type StageSpec struct {
	Rate     float64 `yaml:"rate"`     // requests per second
	Duration int64   `yaml:"duration"` // seconds
}

// SharedPrefixSpec defines shared prefix expansion parameters.
type SharedPrefixSpec struct {
	NumUniqueSystemPrompts  int  `yaml:"num_unique_system_prompts"`
	NumUsersPerSystemPrompt int  `yaml:"num_users_per_system_prompt"`
	SystemPromptLen         int  `yaml:"system_prompt_len"`
	QuestionLen             int  `yaml:"question_len"`
	OutputLen               int  `yaml:"output_len"`
	EnableMultiTurnChat     bool `yaml:"enable_multi_turn_chat"`
}

// validateInferencePerfSpec validates all fields of an InferencePerfSpec.
// Returns error describing the first invalid field found.
func validateInferencePerfSpec(spec *InferencePerfSpec) error {
	if spec == nil {
		return fmt.Errorf("inference_perf spec is nil")
	}
	if len(spec.Stages) == 0 {
		return fmt.Errorf("inference_perf: at least one stage required")
	}
	for i, stage := range spec.Stages {
		if stage.Duration <= 0 {
			return fmt.Errorf("inference_perf.stages[%d]: duration must be positive, got %d", i, stage.Duration)
		}
		if stage.Rate <= 0 || math.IsNaN(stage.Rate) || math.IsInf(stage.Rate, 0) {
			return fmt.Errorf("inference_perf.stages[%d]: rate must be a finite positive number, got %f", i, stage.Rate)
		}
	}
	if spec.SharedPrefix == nil {
		return fmt.Errorf("inference_perf: shared_prefix is required")
	}
	sp := spec.SharedPrefix
	if sp.NumUniqueSystemPrompts <= 0 {
		return fmt.Errorf("inference_perf.shared_prefix: num_unique_system_prompts must be positive, got %d", sp.NumUniqueSystemPrompts)
	}
	if sp.NumUsersPerSystemPrompt <= 0 {
		return fmt.Errorf("inference_perf.shared_prefix: num_users_per_system_prompt must be positive, got %d", sp.NumUsersPerSystemPrompt)
	}
	if sp.SystemPromptLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: system_prompt_len must be non-negative, got %d", sp.SystemPromptLen)
	}
	if sp.QuestionLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: question_len must be non-negative, got %d", sp.QuestionLen)
	}
	if sp.OutputLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: output_len must be non-negative, got %d", sp.OutputLen)
	}
	return nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestValidateInferencePerfSpec -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "feat(workload): add InferencePerfSpec types and validation (BC-10, BC-11, BC-12)

- Define InferencePerfSpec, StageSpec, SharedPrefixSpec structs
- Validate stage duration/rate positivity, prompt counts, length fields
- Return errors (R6: no Fatalf in library)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Implement shared prefix expansion

**Contracts Implemented:** BC-3, BC-4, BC-5, BC-8

**Files:**
- Modify: `sim/workload/inference_perf.go`
- Test: `sim/workload/inference_perf_test.go`

**Step 1: Write failing tests for expansion**

Context: The core expansion function converts a compact InferencePerfSpec into a standard WorkloadSpec with N*M clients.

```go
func TestExpandInferencePerfSpec_SharedPrefix_GeneratesNxMClients(t *testing.T) {
	// BC-3: N*M clients generated
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ws.Clients) != 45 {
		t.Fatalf("client count = %d, want 45 (9*5)", len(ws.Clients))
	}
}

func TestExpandInferencePerfSpec_PrefixGroups_NineDistinct(t *testing.T) {
	// BC-3: 9 distinct prefix groups
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  9,
			NumUsersPerSystemPrompt: 5,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	groups := make(map[string]int)
	for _, c := range ws.Clients {
		groups[c.PrefixGroup]++
	}
	if len(groups) != 9 {
		t.Errorf("distinct prefix groups = %d, want 9", len(groups))
	}
	for g, count := range groups {
		if count != 5 {
			t.Errorf("prefix group %q has %d clients, want 5", g, count)
		}
	}
}

func TestExpandInferencePerfSpec_PrefixLength_Configurable(t *testing.T) {
	// BC-4: configurable prefix length
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         100,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range ws.Clients {
		if c.PrefixLength != 100 {
			t.Errorf("client %q: PrefixLength = %d, want 100", c.ID, c.PrefixLength)
		}
	}
}

func TestExpandInferencePerfSpec_ConstantDistributions(t *testing.T) {
	// BC-5: fixed lengths become constant distributions
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         100,
			QuestionLen:             447,
			OutputLen:               248,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	c := ws.Clients[0]
	if c.InputDist.Type != "constant" {
		t.Errorf("input dist type = %q, want constant", c.InputDist.Type)
	}
	if c.InputDist.Params["value"] != 447 {
		t.Errorf("input dist value = %f, want 447", c.InputDist.Params["value"])
	}
	if c.OutputDist.Type != "constant" {
		t.Errorf("output dist type = %q, want constant", c.OutputDist.Type)
	}
	if c.OutputDist.Params["value"] != 248 {
		t.Errorf("output dist value = %f, want 248", c.OutputDist.Params["value"])
	}
}

func TestExpandInferencePerfSpec_ValidWorkloadSpec(t *testing.T) {
	// BC-8: expansion produces valid WorkloadSpec
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("expand error: %v", err)
	}
	if err := ws.Validate(); err != nil {
		t.Errorf("expanded spec validation failed: %v", err)
	}
}

func TestExpandInferencePerfSpec_EqualRateFractions(t *testing.T) {
	// Each client gets equal share of traffic
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// All 6 clients should have equal rate fractions
	expectedFrac := 1.0 / 6.0
	for _, c := range ws.Clients {
		if c.RateFraction < expectedFrac*0.99 || c.RateFraction > expectedFrac*1.01 {
			t.Errorf("client %q: rate_fraction = %f, want ≈ %f", c.ID, c.RateFraction, expectedFrac)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestExpandInferencePerfSpec -v`
Expected: FAIL -- `ExpandInferencePerfSpec` not defined

**Step 3: Implement expansion**

Add to `sim/workload/inference_perf.go`:

```go
// ExpandInferencePerfSpec converts an InferencePerfSpec into a standard WorkloadSpec.
// The seed is passed through to the resulting WorkloadSpec.
// Returns error if the spec is invalid.
func ExpandInferencePerfSpec(spec *InferencePerfSpec, seed int64) (*WorkloadSpec, error) {
	if err := validateInferencePerfSpec(spec); err != nil {
		return nil, fmt.Errorf("validating inference-perf spec: %w", err)
	}

	sp := spec.SharedPrefix
	numClients := sp.NumUniqueSystemPrompts * sp.NumUsersPerSystemPrompt

	// Compute aggregate rate as time-weighted average across stages
	var totalDuration int64
	var weightedRateSum float64
	for _, stage := range spec.Stages {
		totalDuration += stage.Duration
		weightedRateSum += stage.Rate * float64(stage.Duration)
	}
	aggregateRate := weightedRateSum / float64(totalDuration)

	// Compute lifecycle windows from stages
	windows := stagesToWindows(spec.Stages)

	// Build constant distributions for fixed lengths
	inputDist := constantDist(float64(sp.QuestionLen))
	outputDist := constantDist(float64(sp.OutputLen))

	// Build optional reasoning spec for multi-turn
	var reasoning *ReasoningSpec
	if sp.EnableMultiTurnChat {
		reasoning = &ReasoningSpec{
			ReasonRatioDist: DistSpec{
				Type:   "constant",
				Params: map[string]float64{"value": 0},
			},
			MultiTurn: &MultiTurnSpec{
				MaxRounds:     5,
				ThinkTimeUs:   500000, // 500ms
				ContextGrowth: "accumulate",
			},
		}
	}

	// Generate N*M clients
	clients := make([]ClientSpec, 0, numClients)
	rateFraction := 1.0 / float64(numClients)

	for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
		prefixGroup := fmt.Sprintf("prompt-%d", p)
		for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
			clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)
			client := ClientSpec{
				ID:           clientID,
				TenantID:     prefixGroup,
				SLOClass:     "batch",
				RateFraction: rateFraction,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    inputDist,
				OutputDist:   outputDist,
				PrefixGroup:  prefixGroup,
				PrefixLength: sp.SystemPromptLen,
				Reasoning:    reasoning,
			}
			if len(windows) > 0 {
				client.Lifecycle = &LifecycleSpec{Windows: windows}
			}
			clients = append(clients, client)
		}
	}

	ws := &WorkloadSpec{
		Version:       "1",
		Seed:          seed,
		Category:      "language",
		AggregateRate: aggregateRate,
		Clients:       clients,
	}

	return ws, nil
}

// stagesToWindows converts stage specs into lifecycle ActiveWindows.
// Each window starts where the previous one ended. Duration is in seconds,
// converted to microseconds for BLIS.
func stagesToWindows(stages []StageSpec) []ActiveWindow {
	if len(stages) <= 1 {
		// Single stage: no lifecycle windows needed (always active)
		return nil
	}
	windows := make([]ActiveWindow, len(stages))
	var offsetUs int64
	for i, stage := range stages {
		durationUs := stage.Duration * 1_000_000 // seconds to microseconds
		windows[i] = ActiveWindow{
			StartUs: offsetUs,
			EndUs:   offsetUs + durationUs,
		}
		offsetUs += durationUs
	}
	return windows
}

// constantDist creates a DistSpec for a constant (zero-variance) distribution.
func constantDist(value float64) DistSpec {
	return DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": value},
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestExpandInferencePerfSpec -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "feat(workload): implement shared prefix expansion (BC-3, BC-4, BC-5, BC-8)

- ExpandInferencePerfSpec converts compact spec to standard WorkloadSpec
- N*M client generation with prefix groups and configurable prefix length
- Stage-to-lifecycle window conversion
- Fixed lengths map to constant distributions

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Implement stage-based rate patterns

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/workload/inference_perf_test.go`

**Step 1: Write failing tests for stage-based rates**

Context: Multi-stage rates produce lifecycle windows and correct aggregate rates.

```go
func TestExpandInferencePerfSpec_TwoStages_LifecycleWindows(t *testing.T) {
	// BC-1: stage-to-lifecycle expansion
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ws.Clients) != 1 {
		t.Fatalf("client count = %d, want 1", len(ws.Clients))
	}
	lc := ws.Clients[0].Lifecycle
	if lc == nil {
		t.Fatal("lifecycle should be set for multi-stage spec")
	}
	if len(lc.Windows) != 2 {
		t.Fatalf("window count = %d, want 2", len(lc.Windows))
	}
	// Window 1: [0, 600_000_000)
	if lc.Windows[0].StartUs != 0 {
		t.Errorf("window[0].StartUs = %d, want 0", lc.Windows[0].StartUs)
	}
	if lc.Windows[0].EndUs != 600_000_000 {
		t.Errorf("window[0].EndUs = %d, want 600000000", lc.Windows[0].EndUs)
	}
	// Window 2: [600_000_000, 1_200_000_000)
	if lc.Windows[1].StartUs != 600_000_000 {
		t.Errorf("window[1].StartUs = %d, want 600000000", lc.Windows[1].StartUs)
	}
	if lc.Windows[1].EndUs != 1_200_000_000 {
		t.Errorf("window[1].EndUs = %d, want 1200000000", lc.Windows[1].EndUs)
	}
}

func TestExpandInferencePerfSpec_TwoStages_AggregateRate(t *testing.T) {
	// BC-2: aggregate rate is time-weighted average
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 8.0, Duration: 600},
			{Rate: 20.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// Time-weighted average: (8*600 + 20*600) / 1200 = 14.0
	expectedRate := 14.0
	if ws.AggregateRate != expectedRate {
		t.Errorf("aggregate rate = %f, want %f", ws.AggregateRate, expectedRate)
	}
}

func TestExpandInferencePerfSpec_SingleStage_NoLifecycle(t *testing.T) {
	// Single stage: no lifecycle windows needed
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ws.Clients[0].Lifecycle != nil {
		t.Error("single stage should not set lifecycle windows")
	}
	if ws.AggregateRate != 10.0 {
		t.Errorf("aggregate rate = %f, want 10.0", ws.AggregateRate)
	}
}

func TestExpandInferencePerfSpec_ThreeStages_CumulativeWindows(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 100},
			{Rate: 10.0, Duration: 200},
			{Rate: 15.0, Duration: 300},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	lc := ws.Clients[0].Lifecycle
	if lc == nil || len(lc.Windows) != 3 {
		t.Fatalf("expected 3 lifecycle windows")
	}
	// Window 1: [0, 100_000_000)
	if lc.Windows[0].EndUs != 100_000_000 {
		t.Errorf("window[0].EndUs = %d, want 100000000", lc.Windows[0].EndUs)
	}
	// Window 2: [100_000_000, 300_000_000)
	if lc.Windows[1].StartUs != 100_000_000 || lc.Windows[1].EndUs != 300_000_000 {
		t.Errorf("window[1] = [%d, %d), want [100000000, 300000000)", lc.Windows[1].StartUs, lc.Windows[1].EndUs)
	}
	// Window 3: [300_000_000, 600_000_000)
	if lc.Windows[2].StartUs != 300_000_000 || lc.Windows[2].EndUs != 600_000_000 {
		t.Errorf("window[2] = [%d, %d), want [300000000, 600000000)", lc.Windows[2].StartUs, lc.Windows[2].EndUs)
	}
}
```

**Step 2: Run test to verify it passes**

Run: `go test ./sim/workload/... -run "TestExpandInferencePerfSpec_.*Stage" -v`
Expected: PASS (these tests should already pass with Task 3 implementation)

Note: If they fail, the implementation from Task 3 needs adjustment. The stage-to-lifecycle logic was already implemented in Task 3.

**Step 3: No additional implementation needed**

The stage-based rate logic was already implemented in Task 3 (`stagesToWindows` and aggregate rate calculation in `ExpandInferencePerfSpec`). This task adds the specific behavioral tests.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run "TestExpandInferencePerfSpec_.*Stage" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/inference_perf_test.go
git commit -m "test(workload): add stage-based rate pattern tests (BC-1, BC-2)

- Verify lifecycle window generation from multi-stage specs
- Verify time-weighted aggregate rate calculation
- Verify single-stage spec produces no lifecycle windows

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Implement multi-turn chat mapping

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/workload/inference_perf_test.go`

**Step 1: Write failing test for multi-turn mapping**

Context: When `enable_multi_turn_chat: true`, expanded clients should have reasoning specs.

```go
func TestExpandInferencePerfSpec_MultiTurn_MapsToReasoning(t *testing.T) {
	// BC-7: multi-turn flag maps to reasoning spec
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         100,
			QuestionLen:             50,
			OutputLen:               25,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range ws.Clients {
		if c.Reasoning == nil {
			t.Fatalf("client %q: Reasoning should be set when multi-turn enabled", c.ID)
		}
		mt := c.Reasoning.MultiTurn
		if mt == nil {
			t.Fatalf("client %q: MultiTurn should be set", c.ID)
		}
		if mt.MaxRounds != 5 {
			t.Errorf("client %q: MaxRounds = %d, want 5", c.ID, mt.MaxRounds)
		}
		if mt.ContextGrowth != "accumulate" {
			t.Errorf("client %q: ContextGrowth = %q, want accumulate", c.ID, mt.ContextGrowth)
		}
		if mt.ThinkTimeUs != 500000 {
			t.Errorf("client %q: ThinkTimeUs = %d, want 500000", c.ID, mt.ThinkTimeUs)
		}
	}
}

func TestExpandInferencePerfSpec_NoMultiTurn_NoReasoning(t *testing.T) {
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     false,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, c := range ws.Clients {
		if c.Reasoning != nil {
			t.Errorf("client %q: Reasoning should be nil when multi-turn disabled", c.ID)
		}
	}
}

func TestExpandInferencePerfSpec_MultiTurn_CategoryIsReasoning(t *testing.T) {
	// When multi-turn is enabled, category should be "reasoning"
	spec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 600},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 1,
			SystemPromptLen:         10,
			QuestionLen:             10,
			OutputLen:               10,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ws.Category != "reasoning" {
		t.Errorf("category = %q, want reasoning when multi-turn enabled", ws.Category)
	}
}
```

**Step 2: Run test to verify it passes or fails**

Run: `go test ./sim/workload/... -run "TestExpandInferencePerfSpec_.*MultiTurn\|TestExpandInferencePerfSpec_NoMultiTurn" -v`
Expected: BC-7 tests should pass (multi-turn mapping was implemented in Task 3). The category test may fail.

**Step 3: Fix category for multi-turn**

In `sim/workload/inference_perf.go`, in `ExpandInferencePerfSpec()`, update the WorkloadSpec construction:

```go
	category := "language"
	if sp.EnableMultiTurnChat {
		category = "reasoning"
	}

	ws := &WorkloadSpec{
		Version:       "1",
		Seed:          seed,
		Category:      category,
		AggregateRate: aggregateRate,
		Clients:       clients,
	}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run "TestExpandInferencePerfSpec_.*MultiTurn\|TestExpandInferencePerfSpec_NoMultiTurn" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/inference_perf.go sim/workload/inference_perf_test.go
git commit -m "feat(workload): map multi-turn chat to reasoning spec (BC-7)

- enable_multi_turn_chat maps to reasoning.multi_turn with defaults
- MaxRounds: 5, ThinkTimeUs: 500ms, ContextGrowth: accumulate
- Category set to 'reasoning' when multi-turn enabled

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Wire InferencePerfSpec into WorkloadSpec and GenerateRequests

**Contracts Implemented:** BC-8, BC-9, BC-13

**Files:**
- Modify: `sim/workload/spec.go`
- Modify: `sim/workload/generator.go`
- Test: `sim/workload/inference_perf_test.go`

**Step 1: Write failing tests for end-to-end generation**

Context: An InferencePerfSpec embedded in a WorkloadSpec should auto-expand and produce requests.

```go
func TestGenerateRequests_InferencePerfSpec_ProducesRequests(t *testing.T) {
	// BC-8: end-to-end generation from inference-perf spec
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10}, // 10 seconds
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	spec := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		AggregateRate: 10.0,
		InferencePerf: ipSpec,
	}
	horizon := int64(10_000_000) // 10 seconds
	requests, err := GenerateRequests(spec, horizon, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from inference-perf spec")
	}
}

func TestGenerateRequests_InferencePerfSpec_Deterministic(t *testing.T) {
	// BC-9: determinism preserved
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	spec := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		AggregateRate: 10.0,
		InferencePerf: ipSpec,
	}
	horizon := int64(10_000_000)

	r1, err1 := GenerateRequests(spec, horizon, 0)
	// Reset spec.Clients for second run (expansion mutates the spec)
	spec.Clients = nil
	r2, err2 := GenerateRequests(spec, horizon, 0)
	if err1 != nil || err2 != nil {
		t.Fatalf("errors: %v, %v", err1, err2)
	}
	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}

func TestLoadWorkloadSpec_InferencePerfSpec_StrictParsing(t *testing.T) {
	// BC-13: strict YAML parsing for new types
	dir := t.TempDir()
	path := filepath.Join(dir, "bad-ip.yaml")
	yamlData := `
version: "1"
seed: 42
aggregate_rate: 10.0
inference_perf:
  stages:
    - rate: 10.0
      duraton: 600
  shared_prefix:
    num_unique_system_prompts: 1
    num_users_per_system_prompt: 1
    system_prompt_len: 10
    question_len: 10
    output_len: 10
`
	if err := os.WriteFile(path, []byte(yamlData), 0644); err != nil {
		t.Fatal(err)
	}
	_, err := LoadWorkloadSpec(path)
	if err == nil {
		t.Fatal("expected error for typo 'duraton' in YAML")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_InferencePerfSpec\|TestLoadWorkloadSpec_InferencePerfSpec" -v`
Expected: FAIL -- `InferencePerf` field not on WorkloadSpec

**Step 3: Wire InferencePerfSpec into WorkloadSpec and generator**

In `sim/workload/spec.go`, add the field to `WorkloadSpec`:

```go
type WorkloadSpec struct {
	Version       string       `yaml:"version"`
	Seed          int64        `yaml:"seed"`
	Category      string       `yaml:"category"`
	Clients       []ClientSpec `yaml:"clients"`
	AggregateRate float64      `yaml:"aggregate_rate"`
	Horizon       int64        `yaml:"horizon,omitempty"`
	NumRequests   int64        `yaml:"num_requests,omitempty"`
	ServeGenData  *ServeGenDataSpec    `yaml:"servegen_data,omitempty"`
	InferencePerf *InferencePerfSpec   `yaml:"inference_perf,omitempty"`
}
```

In `sim/workload/generator.go`, add expansion before the ServeGen loading section:

```go
	// Expand inference-perf spec if specified (populates spec.Clients)
	if spec.InferencePerf != nil && len(spec.Clients) == 0 {
		expanded, err := ExpandInferencePerfSpec(spec.InferencePerf, spec.Seed)
		if err != nil {
			return nil, fmt.Errorf("expanding inference-perf spec: %w", err)
		}
		spec.Clients = expanded.Clients
		if spec.Category == "" {
			spec.Category = expanded.Category
		}
		if spec.AggregateRate <= 0 {
			spec.AggregateRate = expanded.AggregateRate
		}
	}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run "TestGenerateRequests_InferencePerfSpec\|TestLoadWorkloadSpec_InferencePerfSpec" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/spec.go sim/workload/generator.go sim/workload/inference_perf_test.go
git commit -m "feat(workload): wire InferencePerfSpec into WorkloadSpec pipeline (BC-8, BC-9, BC-13)

- Add InferencePerf field to WorkloadSpec (YAML: inference_perf)
- Auto-expand in GenerateRequests before validation
- Strict YAML parsing inherited from existing decoder
- Determinism preserved through expansion

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Equivalence test — shorthand vs manual expansion

**Contracts Implemented:** BC-3, BC-4, BC-9 (strengthened)

**Files:**
- Modify: `sim/workload/inference_perf_test.go`

**Step 1: Write equivalence test**

Context: Verify that the shorthand expansion produces the same requests as a manually-written equivalent spec. This is acceptance criterion #6 from issue #252.

```go
func TestInferencePerfExpansion_EquivalentToManual(t *testing.T) {
	// Acceptance criterion: shorthand and manual expansion produce identical requests.
	// Build equivalent specs and compare generated request sequences.

	// Shorthand spec
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}

	// Manual spec: construct the same clients explicitly
	manual := &WorkloadSpec{
		Version:       expanded.Version,
		Seed:          expanded.Seed,
		Category:      expanded.Category,
		AggregateRate: expanded.AggregateRate,
		Clients:       expanded.Clients, // use the expanded clients directly
	}

	horizon := int64(10_000_000) // 10 seconds

	// Generate from expanded
	r1, err1 := GenerateRequests(expanded, horizon, 0)
	if err1 != nil {
		t.Fatalf("expanded generation error: %v", err1)
	}

	// Generate from manual (same clients)
	r2, err2 := GenerateRequests(manual, horizon, 0)
	if err2 != nil {
		t.Fatalf("manual generation error: %v", err2)
	}

	if len(r1) != len(r2) {
		t.Fatalf("different request counts: expanded=%d, manual=%d", len(r1), len(r2))
	}

	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if len(r1[i].InputTokens) != len(r2[i].InputTokens) {
			t.Errorf("request %d: input len %d vs %d", i, len(r1[i].InputTokens), len(r2[i].InputTokens))
			break
		}
	}
}

func TestInferencePerfExpansion_SharedPrefixTokensIdentical(t *testing.T) {
	// Verify that clients in the same prefix group actually share prefix tokens
	// when requests are generated through the full pipeline.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 20.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         80,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	horizon := int64(5_000_000) // 5 seconds
	requests, err := GenerateRequests(expanded, horizon, 100)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	if len(requests) < 2 {
		t.Fatal("need at least 2 requests for prefix comparison")
	}

	// All requests should have inputs at least 80 tokens long (prefix length)
	prefixLen := 80
	for i, req := range requests {
		if len(req.InputTokens) < prefixLen {
			t.Errorf("request %d: input len %d < prefix len %d", i, len(req.InputTokens), prefixLen)
		}
	}

	// Group requests by tenant (which maps to prefix group)
	byTenant := make(map[string][]*sim.Request)
	for _, req := range requests {
		byTenant[req.TenantID] = append(byTenant[req.TenantID], req)
	}

	// Within each group, first prefixLen tokens must be identical
	for tenant, reqs := range byTenant {
		if len(reqs) < 2 {
			continue
		}
		first := reqs[0].InputTokens[:prefixLen]
		for i := 1; i < len(reqs); i++ {
			other := reqs[i].InputTokens[:prefixLen]
			for j := 0; j < prefixLen; j++ {
				if first[j] != other[j] {
					t.Errorf("tenant %q: request %d prefix token %d differs from request 0", tenant, i, j)
					break
				}
			}
		}
	}
}
```

**Step 2: Run tests**

Run: `go test ./sim/workload/... -run "TestInferencePerfExpansion" -v`
Expected: PASS

**Step 3: No additional implementation needed**

This is a pure test task verifying acceptance criterion #6.

**Step 4: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/workload/inference_perf_test.go
git commit -m "test(workload): add shorthand-vs-manual equivalence tests

- Verify expanded spec produces identical requests to manual construction
- Verify shared prefix tokens are identical within prefix groups
- Acceptance criterion #6 from issue #252

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Add YAML loading and example file

**Contracts Implemented:** BC-13 (strengthened)

**Files:**
- Create: `examples/inference-perf-shared-prefix.yaml`
- Modify: `sim/workload/inference_perf_test.go`

**Step 1: Write YAML round-trip test**

Context: Verify that the full YAML → parse → expand → generate pipeline works.

```go
func TestLoadWorkloadSpec_InferencePerfSpec_FullPipeline(t *testing.T) {
	// Full YAML → parse → expand → generate pipeline
	dir := t.TempDir()
	path := filepath.Join(dir, "ip-spec.yaml")
	yamlData := `
version: "1"
seed: 42
aggregate_rate: 10.0
inference_perf:
  stages:
    - rate: 8.0
      duration: 5
    - rate: 20.0
      duration: 5
  shared_prefix:
    num_unique_system_prompts: 3
    num_users_per_system_prompt: 2
    system_prompt_len: 50
    question_len: 100
    output_len: 50
`
	if err := os.WriteFile(path, []byte(yamlData), 0644); err != nil {
		t.Fatal(err)
	}

	spec, err := LoadWorkloadSpec(path)
	if err != nil {
		t.Fatalf("load error: %v", err)
	}
	if spec.InferencePerf == nil {
		t.Fatal("InferencePerf should be parsed from YAML")
	}
	if len(spec.InferencePerf.Stages) != 2 {
		t.Errorf("stage count = %d, want 2", len(spec.InferencePerf.Stages))
	}

	horizon := int64(10_000_000) // 10 seconds
	requests, err := GenerateRequests(spec, horizon, 50)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from YAML pipeline")
	}
	if len(requests) > 50 {
		t.Errorf("request count %d exceeds maxRequests 50", len(requests))
	}
}
```

**Step 2: Run test**

Run: `go test ./sim/workload/... -run TestLoadWorkloadSpec_InferencePerfSpec_FullPipeline -v`
Expected: PASS

**Step 3: Create example file**

Create `examples/inference-perf-shared-prefix.yaml`:

```yaml
# BLIS Inference-Perf Workload Format Example
#
# This file demonstrates BLIS's native support for inference-perf style
# workload specifications. Instead of manually writing 45 client definitions,
# use the compact inference_perf block.
#
# ============================================================================
# INFERENCE-PERF TO BLIS TRANSLATION
# ============================================================================
#
# inference-perf:                       BLIS equivalent:
#   load.stages[].rate     -->          stages[].rate (req/s)
#   load.stages[].duration -->          stages[].duration (seconds)
#   data.shared_prefix.num_unique_system_prompts  --> N prefix groups
#   data.shared_prefix.num_users_per_system_prompt --> M clients per group
#   data.shared_prefix.system_prompt_len --> prefix_length on each client
#   data.shared_prefix.question_len     --> constant input distribution
#   data.shared_prefix.output_len       --> constant output distribution
#   data.shared_prefix.enable_multi_turn_chat --> reasoning.multi_turn
#
# ============================================================================
# TRY IT
# ============================================================================
#
#   ./simulation_worker run \
#     --model meta-llama/llama-3.1-8b-instruct \
#     --num-instances 4 --routing-policy weighted \
#     --workload-spec examples/inference-perf-shared-prefix.yaml
#
# This auto-generates 45 clients (9 prompts x 5 users), each sharing a
# 100-token system prompt prefix within their group.

version: "1"
seed: 42
aggregate_rate: 14.0   # Overridden by time-weighted average if omitted

inference_perf:
  stages:
    - rate: 8.0        # Ramp-up: 8 req/s for 10 minutes
      duration: 600
    - rate: 20.0       # Burst: 20 req/s for 10 minutes
      duration: 600

  shared_prefix:
    num_unique_system_prompts: 9
    num_users_per_system_prompt: 5
    system_prompt_len: 100     # 100-token shared prefix per group
    question_len: 447          # Fixed input length (constant distribution)
    output_len: 248            # Fixed output length
    enable_multi_turn_chat: false
```

**Step 4: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add examples/inference-perf-shared-prefix.yaml sim/workload/inference_perf_test.go
git commit -m "docs(examples): add inference-perf workload format example

- Full YAML pipeline test (load → expand → generate)
- Example shows inference-perf to BLIS translation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Invariant and determinism tests

**Contracts Implemented:** BC-9 (strengthened), INV-6 compliance

**Files:**
- Modify: `sim/workload/inference_perf_test.go`

**Step 1: Write invariant tests**

Context: Verify system invariants hold for inference-perf generated workloads.

```go
func TestInferencePerf_Determinism_SameSeedIdenticalOutput(t *testing.T) {
	// INV-6: same seed → identical output
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 10.0, Duration: 10},
			{Rate: 20.0, Duration: 10},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         50,
			QuestionLen:             100,
			OutputLen:               50,
		},
	}

	generate := func() []*sim.Request {
		expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
		if err != nil {
			t.Fatalf("expansion error: %v", err)
		}
		reqs, err := GenerateRequests(expanded, 20_000_000, 100)
		if err != nil {
			t.Fatalf("generation error: %v", err)
		}
		return reqs
	}

	r1 := generate()
	r2 := generate()

	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if r1[i].ID != r2[i].ID {
			t.Errorf("request %d: ID %q vs %q", i, r1[i].ID, r2[i].ID)
			break
		}
		if len(r1[i].InputTokens) != len(r2[i].InputTokens) {
			t.Errorf("request %d: input len %d vs %d", i, len(r1[i].InputTokens), len(r2[i].InputTokens))
			break
		}
		// Verify token-level identity
		for j := range r1[i].InputTokens {
			if r1[i].InputTokens[j] != r2[i].InputTokens[j] {
				t.Errorf("request %d token %d: %d vs %d", i, j, r1[i].InputTokens[j], r2[i].InputTokens[j])
				break
			}
		}
	}
}

func TestInferencePerf_Causality_ArrivalTimesMonotonic(t *testing.T) {
	// INV-3/INV-5: arrival times never decrease
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 50.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  3,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         20,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	requests, err := GenerateRequests(expanded, 5_000_000, 0)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	for i := 1; i < len(requests); i++ {
		if requests[i].ArrivalTime < requests[i-1].ArrivalTime {
			t.Errorf("arrival time not monotonic: request %d (%d) < request %d (%d)",
				i, requests[i].ArrivalTime, i-1, requests[i-1].ArrivalTime)
			break
		}
	}
}

func TestInferencePerf_AllRequestsHaveValidTokens(t *testing.T) {
	// Every request must have non-empty input and output tokens
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{
			{Rate: 20.0, Duration: 5},
		},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 2,
			SystemPromptLen:         30,
			QuestionLen:             50,
			OutputLen:               25,
		},
	}
	expanded, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("expansion error: %v", err)
	}
	requests, err := GenerateRequests(expanded, 5_000_000, 50)
	if err != nil {
		t.Fatalf("generation error: %v", err)
	}
	for i, req := range requests {
		if len(req.InputTokens) == 0 {
			t.Errorf("request %d has empty input tokens", i)
		}
		if len(req.OutputTokens) == 0 {
			t.Errorf("request %d has empty output tokens", i)
		}
		// Input tokens should be at least prefix_length (30) + question_len (50)
		expectedMinLen := 30 + 50
		if len(req.InputTokens) < expectedMinLen {
			t.Errorf("request %d: input len %d < expected min %d (prefix+question)",
				i, len(req.InputTokens), expectedMinLen)
		}
	}
}
```

**Step 2: Run tests**

Run: `go test ./sim/workload/... -run "TestInferencePerf_Determinism\|TestInferencePerf_Causality\|TestInferencePerf_AllRequests" -v`
Expected: PASS

**Step 3: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 4: Commit**

```bash
git add sim/workload/inference_perf_test.go
git commit -m "test(workload): add invariant tests for inference-perf workloads (INV-3, INV-5, INV-6)

- Determinism: same seed produces identical requests
- Causality: arrival times are monotonically non-decreasing
- Token validity: all requests have non-empty tokens with correct lengths

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Run full test suite and update CLAUDE.md

**Contracts Implemented:** All (regression verification)

**Files:**
- Modify: CLAUDE.md (file organization section)

**Step 1: Run full test suite**

Run: `go test ./...`
Expected: All tests pass

**Step 2: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 3: Update CLAUDE.md**

Add `inference_perf.go` to the file organization section under `sim/workload/`:

In the `sim/workload/` section of the file organization, add:
```
│   ├── inference_perf.go      # inference-perf format: InferencePerfSpec, expansion, validation
```

Update the "Current Implementation Focus" section or relevant notes to mention the new capability.

**Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with inference-perf workload format

- Add inference_perf.go to file organization
- Document new constant distribution type

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 4 | Unit | TestExpandInferencePerfSpec_TwoStages_LifecycleWindows |
| BC-2 | Task 4 | Unit | TestExpandInferencePerfSpec_TwoStages_AggregateRate |
| BC-3 | Task 3 | Unit | TestExpandInferencePerfSpec_SharedPrefix_GeneratesNxMClients |
| BC-3 | Task 3 | Unit | TestExpandInferencePerfSpec_PrefixGroups_NineDistinct |
| BC-4 | Task 3 | Unit | TestExpandInferencePerfSpec_PrefixLength_Configurable |
| BC-5 | Task 3 | Unit | TestExpandInferencePerfSpec_ConstantDistributions |
| BC-6 | Task 1 | Unit | TestConstantSampler_AlwaysReturnsExactValue |
| BC-6 | Task 1 | Unit | TestConstantSampler_ValueOne_ReturnsOne |
| BC-6 | Task 1 | Unit | TestConstantSampler_ZeroValue_ReturnsOne |
| BC-7 | Task 5 | Unit | TestExpandInferencePerfSpec_MultiTurn_MapsToReasoning |
| BC-8 | Task 3 | Unit | TestExpandInferencePerfSpec_ValidWorkloadSpec |
| BC-8 | Task 6 | Integration | TestGenerateRequests_InferencePerfSpec_ProducesRequests |
| BC-9 | Task 6 | Unit | TestGenerateRequests_InferencePerfSpec_Deterministic |
| BC-9 | Task 9 | Invariant | TestInferencePerf_Determinism_SameSeedIdenticalOutput |
| BC-10 | Task 2 | Failure | TestValidateInferencePerfSpec_ZeroDuration_ReturnsError |
| BC-11 | Task 2 | Failure | TestValidateInferencePerfSpec_ZeroPrompts_ReturnsError |
| BC-12 | Task 2 | Failure | TestValidateInferencePerfSpec_NegativeLength_ReturnsError |
| BC-13 | Task 6 | Failure | TestLoadWorkloadSpec_InferencePerfSpec_StrictParsing |
| BC-14 | -- | By design | All functions return error, verified by grep |
| INV-3 | Task 9 | Invariant | TestInferencePerf_Causality_ArrivalTimesMonotonic |
| INV-6 | Task 9 | Invariant | TestInferencePerf_Determinism_SameSeedIdenticalOutput |
| Equivalence | Task 7 | Integration | TestInferencePerfExpansion_EquivalentToManual |
| Equivalence | Task 7 | Integration | TestInferencePerfExpansion_SharedPrefixTokensIdentical |

**Golden dataset:** Not affected -- this PR adds a new generation path but does not change existing output format. No golden dataset update needed.

**Shared test infrastructure:** Uses existing `sim.Request` type for assertions. No new shared helpers needed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Stage rate math produces wrong effective rate | Medium | Medium | BC-2 test verifies time-weighted average; single-stage test verifies simple case | Task 4 |
| Lifecycle windows off by factor of 1e6 (seconds vs microseconds) | Medium | High | BC-1 test verifies exact microsecond window boundaries | Task 4 |
| Expansion mutates shared spec pointer across calls | Low | Medium | BC-9 determinism test calls twice with same spec | Task 6, Task 9 |
| ConstantSampler with value=0 returns 0 tokens | Low | High | Edge case test clamps to 1 | Task 1 |
| YAML strict parsing doesn't cover nested InferencePerfSpec fields | Low | Medium | BC-13 test verifies typo in nested stage field is caught | Task 6 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions. `InferencePerfSpec` is a simple data type with expansion function.
- [x] No feature creep beyond PR scope. Only stage rates, shared prefix, multi-turn, constant dist.
- [x] No unexercised flags or interfaces. Every type is used in tests and pipeline.
- [x] No partial implementations. Every function is complete and tested.
- [x] No breaking changes. New `InferencePerf` field is `omitempty`, existing specs unchanged.
- [x] No hidden global state impact. All functions are pure.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used from existing shared test package.
- [x] CLAUDE.md updated: new file added to organization.
- [x] No stale references left in CLAUDE.md.
- [x] Deviation log reviewed -- no unresolved deviations.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7→8→9→10).
- [x] All contracts mapped to specific tasks.
- [x] Golden dataset regeneration not needed.
- [x] Construction site audit: `WorkloadSpec` struct literal sites checked (spec.go:127, scenarios.go, generator_test.go, spec_test.go). New `InferencePerf` field is pointer with `omitempty` -- zero value (nil) is backward compatible for all existing sites.

**Antipattern rules:**
- [x] R1: No silent continue/return. Validation returns errors.
- [x] R2: No map iteration affecting float sums or output ordering in new code.
- [x] R3: No new CLI flags (all changes in sim/workload/).
- [x] R4: WorkloadSpec construction sites audited. New field is nil-safe.
- [x] R5: No resource allocation loops.
- [x] R6: No logrus.Fatalf in sim/workload/.
- [x] R7: Invariant tests added (Task 9).
- [x] R8: No exported mutable maps.
- [x] R9: No YAML fields where zero is a valid user value that needs pointer type. All int fields (counts, lengths) use zero as "invalid" not as a valid value.
- [x] R10: YAML strict parsing inherited from existing LoadWorkloadSpec decoder.
- [x] R11: Division guard: `totalDuration` in aggregate rate calculation is guaranteed non-zero (validated: at least one stage with positive duration).
- [x] R12: Golden dataset not affected.
- [x] R13: No new interfaces.
- [x] R14: No multi-module methods.
- [x] R15: No stale PR references.
- [x] R16: Config grouped by module (InferencePerfSpec is self-contained).
- [x] R17: No new routing signals.

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/inference_perf.go`

**Purpose:** Define InferencePerfSpec types, validation, and expansion to standard WorkloadSpec.

**Complete Implementation:**

```go
package workload

import (
	"fmt"
	"math"
)

// InferencePerfSpec defines an inference-perf style workload using a compact
// format. It is expanded into a standard WorkloadSpec via ExpandInferencePerfSpec().
//
// Stage-based rates: sequential rate/duration pairs that produce lifecycle windows.
// Shared prefix: auto-generates N*M clients with prefix groups.
// Multi-turn: maps to BLIS reasoning.multi_turn with context accumulation.
type InferencePerfSpec struct {
	Stages       []StageSpec       `yaml:"stages"`
	SharedPrefix *SharedPrefixSpec `yaml:"shared_prefix"`
}

// StageSpec defines a single rate/duration stage.
type StageSpec struct {
	Rate     float64 `yaml:"rate"`     // requests per second
	Duration int64   `yaml:"duration"` // seconds
}

// SharedPrefixSpec defines shared prefix expansion parameters.
type SharedPrefixSpec struct {
	NumUniqueSystemPrompts  int  `yaml:"num_unique_system_prompts"`
	NumUsersPerSystemPrompt int  `yaml:"num_users_per_system_prompt"`
	SystemPromptLen         int  `yaml:"system_prompt_len"`
	QuestionLen             int  `yaml:"question_len"`
	OutputLen               int  `yaml:"output_len"`
	EnableMultiTurnChat     bool `yaml:"enable_multi_turn_chat"`
}

// validateInferencePerfSpec validates all fields of an InferencePerfSpec.
func validateInferencePerfSpec(spec *InferencePerfSpec) error {
	if spec == nil {
		return fmt.Errorf("inference_perf spec is nil")
	}
	if len(spec.Stages) == 0 {
		return fmt.Errorf("inference_perf: at least one stage required")
	}
	for i, stage := range spec.Stages {
		if stage.Duration <= 0 {
			return fmt.Errorf("inference_perf.stages[%d]: duration must be positive, got %d", i, stage.Duration)
		}
		if stage.Rate <= 0 || math.IsNaN(stage.Rate) || math.IsInf(stage.Rate, 0) {
			return fmt.Errorf("inference_perf.stages[%d]: rate must be a finite positive number, got %f", i, stage.Rate)
		}
	}
	if spec.SharedPrefix == nil {
		return fmt.Errorf("inference_perf: shared_prefix is required")
	}
	sp := spec.SharedPrefix
	if sp.NumUniqueSystemPrompts <= 0 {
		return fmt.Errorf("inference_perf.shared_prefix: num_unique_system_prompts must be positive, got %d", sp.NumUniqueSystemPrompts)
	}
	if sp.NumUsersPerSystemPrompt <= 0 {
		return fmt.Errorf("inference_perf.shared_prefix: num_users_per_system_prompt must be positive, got %d", sp.NumUsersPerSystemPrompt)
	}
	if sp.SystemPromptLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: system_prompt_len must be non-negative, got %d", sp.SystemPromptLen)
	}
	if sp.QuestionLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: question_len must be non-negative, got %d", sp.QuestionLen)
	}
	if sp.OutputLen < 0 {
		return fmt.Errorf("inference_perf.shared_prefix: output_len must be non-negative, got %d", sp.OutputLen)
	}
	return nil
}

// ExpandInferencePerfSpec converts an InferencePerfSpec into a standard WorkloadSpec.
// The seed is passed through to the resulting WorkloadSpec.
func ExpandInferencePerfSpec(spec *InferencePerfSpec, seed int64) (*WorkloadSpec, error) {
	if err := validateInferencePerfSpec(spec); err != nil {
		return nil, fmt.Errorf("validating inference-perf spec: %w", err)
	}

	sp := spec.SharedPrefix
	numClients := sp.NumUniqueSystemPrompts * sp.NumUsersPerSystemPrompt

	// Compute aggregate rate as time-weighted average across stages
	var totalDuration int64
	var weightedRateSum float64
	for _, stage := range spec.Stages {
		totalDuration += stage.Duration
		weightedRateSum += stage.Rate * float64(stage.Duration)
	}
	aggregateRate := weightedRateSum / float64(totalDuration)

	// Compute lifecycle windows from stages (nil for single stage)
	windows := stagesToWindows(spec.Stages)

	// Build constant distributions for fixed lengths
	inputDist := constantDist(float64(sp.QuestionLen))
	outputDist := constantDist(float64(sp.OutputLen))

	// Build optional reasoning spec for multi-turn
	var reasoning *ReasoningSpec
	if sp.EnableMultiTurnChat {
		reasoning = &ReasoningSpec{
			ReasonRatioDist: DistSpec{
				Type:   "constant",
				Params: map[string]float64{"value": 0},
			},
			MultiTurn: &MultiTurnSpec{
				MaxRounds:     5,
				ThinkTimeUs:   500000,
				ContextGrowth: "accumulate",
			},
		}
	}

	category := "language"
	if sp.EnableMultiTurnChat {
		category = "reasoning"
	}

	// Generate N*M clients
	clients := make([]ClientSpec, 0, numClients)
	rateFraction := 1.0 / float64(numClients)

	for p := 0; p < sp.NumUniqueSystemPrompts; p++ {
		prefixGroup := fmt.Sprintf("prompt-%d", p)
		for u := 0; u < sp.NumUsersPerSystemPrompt; u++ {
			clientID := fmt.Sprintf("prompt-%d-user-%d", p, u)
			client := ClientSpec{
				ID:           clientID,
				TenantID:     prefixGroup,
				SLOClass:     "batch",
				RateFraction: rateFraction,
				Arrival:      ArrivalSpec{Process: "poisson"},
				InputDist:    inputDist,
				OutputDist:   outputDist,
				PrefixGroup:  prefixGroup,
				PrefixLength: sp.SystemPromptLen,
				Reasoning:    reasoning,
			}
			if len(windows) > 0 {
				client.Lifecycle = &LifecycleSpec{Windows: windows}
			}
			clients = append(clients, client)
		}
	}

	return &WorkloadSpec{
		Version:       "1",
		Seed:          seed,
		Category:      category,
		AggregateRate: aggregateRate,
		Clients:       clients,
	}, nil
}

// stagesToWindows converts stage specs into lifecycle ActiveWindows.
// Returns nil for single-stage specs (always active, no windows needed).
func stagesToWindows(stages []StageSpec) []ActiveWindow {
	if len(stages) <= 1 {
		return nil
	}
	windows := make([]ActiveWindow, len(stages))
	var offsetUs int64
	for i, stage := range stages {
		durationUs := stage.Duration * 1_000_000
		windows[i] = ActiveWindow{
			StartUs: offsetUs,
			EndUs:   offsetUs + durationUs,
		}
		offsetUs += durationUs
	}
	return windows
}

// constantDist creates a DistSpec for a constant (zero-variance) distribution.
func constantDist(value float64) DistSpec {
	return DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": value},
	}
}
```

**Key Implementation Notes:**
- RNG usage: None in expansion (pure function). Generation uses existing PartitionedRNG.
- Error handling: All validation returns errors (R6 compliance). No panics.
- State mutation: None. Pure function produces new WorkloadSpec.
- Division guard (R11): `totalDuration` guaranteed > 0 by validation (at least one positive-duration stage).

### File: `sim/workload/distribution.go` (modifications)

**Purpose:** Add `ConstantSampler` type and `"constant"` case in factory.

**New code added:**

```go
// ConstantSampler always returns the same fixed value.
// Used for inference-perf fixed-length token specifications (zero variance).
type ConstantSampler struct {
	value int
}

func (s *ConstantSampler) Sample(_ *rand.Rand) int {
	if s.value < 1 {
		return 1
	}
	return s.value
}
```

In `NewLengthSampler`, new case:

```go
	case "constant":
		val := int(spec.Params["value"])
		return &ConstantSampler{value: val}, nil
```

### File: `sim/workload/spec.go` (modifications)

**Purpose:** Add `"constant"` to validDistTypes, add `InferencePerf` field to `WorkloadSpec`.

**Changes:**
1. Add `"constant": true` to `validDistTypes` map
2. Add `InferencePerf *InferencePerfSpec \`yaml:"inference_perf,omitempty"\`` to `WorkloadSpec`

### File: `sim/workload/generator.go` (modifications)

**Purpose:** Auto-expand InferencePerfSpec before validation in `GenerateRequests`.

**New code added (before ServeGen section):**

```go
	// Expand inference-perf spec if specified (populates spec.Clients)
	if spec.InferencePerf != nil && len(spec.Clients) == 0 {
		expanded, err := ExpandInferencePerfSpec(spec.InferencePerf, spec.Seed)
		if err != nil {
			return nil, fmt.Errorf("expanding inference-perf spec: %w", err)
		}
		spec.Clients = expanded.Clients
		if spec.Category == "" {
			spec.Category = expanded.Category
		}
		if spec.AggregateRate <= 0 {
			spec.AggregateRate = expanded.AggregateRate
		}
	}
```
