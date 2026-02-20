# feat(routing): Composable scorer framework for weighted routing with stateless scorers — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the broken two-dimension `weighted` routing policy with a composable multi-scorer pipeline so researchers can tune independent scoring dimensions (queue depth, KV utilization, load balance) and get meaningful routing tradeoffs instead of feedback latency artifacts.

**The problem today:** The `weighted` routing policy's cache dimension measures capacity headroom (`FreeKVBlocks / maxFreeKVBlocks`), not prefix affinity. In homogeneous clusters, FreeKVBlocks and EffectiveLoad are correlated — both measure instance busyness. Varying `--routing-cache-weight` vs `--routing-load-weight` produces a feedback latency artifact, not a meaningful tradeoff (#229). The README demo commands (#230) are therefore misleading.

**What this PR adds:**
1. **Composable scorer pipeline** — `weighted` routing now aggregates N independent scoring dimensions, each returning [0,1] scores per instance, combined with configurable weights. Example: `--routing-scorers "queue-depth:2,kv-utilization:2,load-balance:1"`.
2. **Three stateless scorers** — `queue-depth` (min-max normalization of effective load), `kv-utilization` (1 − utilization), and `load-balance` (inverse transform 1/(1+load)). Each independently useful for capacity planning experiments.
3. **Clean CLI surface** — replaces the broken `--routing-cache-weight`/`--routing-load-weight` flags with `--routing-scorers` that accepts `name:weight` pairs.

**Why this matters:** This is step 1 of matching llm-d's Endpoint Picker architecture. PR 18 will add a prefix-affinity scorer with router-side cache, enabling the real cache/load tradeoff that #229 identified as missing.

**Architecture:** The scorer framework is internal to the `weighted` routing policy — the frozen `RoutingPolicy` interface is unchanged. New scorer implementations live in `sim/routing_scorers.go`. `ScorerConfig` (name+weight pair) propagates through `RoutingConfig` (YAML) → `DeploymentConfig` → `NewRoutingPolicy` factory. Non-weighted routing policies are completely unaffected.

**Source:** PR 17 in `docs/plans/2026-02-19-weighted-scoring-macro-plan.md`, design doc `docs/plans/2026-02-19-weighted-scoring-evolution-design.md`, parallel dev plan `docs/plans/2026-02-19-parallel-dev-plan-234-plus-4.md`.

**Closes:** Fixes #229, fixes #230

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR replaces the monolithic `WeightedScoring` routing policy (two hardcoded dimensions: cache capacity + load balance) with a composable scorer pipeline accepting N independent scorers with configurable weights. It ships three stateless scorers matching llm-d's scoring dimensions.

**Where it fits:** This is the first of two PRs in the Composable Scorer Framework plan. PR 17 provides the framework + stateless scorers. PR 18 (depends on this) adds a prefix-affinity scorer with router-side cache index. All other routing policies (round-robin, least-loaded, prefix-affinity, always-busiest) are completely unaffected.

**Adjacent blocks:** Counterfactual analysis (`sim/cluster/counterfactual.go`) consumes `RoutingDecision.Scores` — unchanged contract. Decision tracing (`sim/trace/`) records routing decisions — unaffected. Admission policies, priority policies, and schedulers are independent.

**DEVIATION flags from Phase 0:** File paths and line references confirmed accurate. Scope deviations (function type vs interface, LOC estimates, BC renumbering) are documented in Section D.

### B) Behavioral Contracts

#### Positive Contracts (what MUST happen)

**BC-17-1: Score Range Conformance (INV-1, INV-2)**
- GIVEN any scorer (queue-depth, kv-utilization, or load-balance) and any set of instance snapshots
- WHEN the scorer evaluates the snapshots
- THEN every instance receives a score in [0, 1], and every instance in the snapshot list has a score
- MECHANISM: Each scorer function returns scores clamped to [0,1]; the pipeline verifies completeness

**BC-17-2: Weight Normalization (INV-6)**
- GIVEN two `weighted` policy instances configured with weights `[3,2,2]` and `[0.43,0.29,0.29]` (same ratio)
- WHEN both route the same request against the same snapshots
- THEN both produce identical routing decisions (same target instance)
- MECHANISM: Weights are normalized to sum to 1.0 at construction time

**BC-17-3: Non-Weighted Policy Stability (INV-5)**
- GIVEN routing policies round-robin, least-loaded, prefix-affinity, and always-busiest
- WHEN the same workload is run before and after this PR
- THEN all produce byte-identical per-request output (same target instances, same metrics)
- MECHANISM: Only the `case "weighted"` branch in `NewRoutingPolicy` changes; all other cases are untouched

**BC-17-4: Invalid Config Rejection (matches macro plan BC-17-4)**
- GIVEN scorer configs with NaN weight, Inf weight, negative weight, zero weight, or unknown scorer name
- WHEN parsed via CLI (`ParseScorerConfigs`) or validated via YAML (`PolicyBundle.Validate`)
- THEN a clear error is returned (not a panic) at configuration time, before any simulation runs
- MECHANISM: `ParseScorerConfigs` validates each weight; `Validate()` checks scorer names against `IsValidScorer()`

**BC-17-5: Load-Balance-Only Equivalence (matches macro plan BC-17-5)**
- GIVEN a `weighted` policy configured with `load-balance:1` (single scorer)
- WHEN routing requests against the same snapshots as `least-loaded`
- THEN both select the same target instance for every request
- MECHANISM: `1/(1+load)` is monotonically decreasing in load, so `argmax(1/(1+load))` = `argmin(load)`, with identical tie-breaking (first occurrence)

**BC-17-6: Scorer Pipeline Argmax (new — extends macro plan)**
- GIVEN a `weighted` policy with multiple scorers and weighted scores computed per instance
- WHEN routing a request
- THEN the instance with the highest composite score is selected; ties broken by first occurrence in snapshot order
- MECHANISM: Iterate snapshots in order, use strict `>` for best score update

**BC-17-7: Queue-Depth Min-Max Normalization (new)**
- GIVEN instances with varying effective loads
- WHEN the queue-depth scorer evaluates them
- THEN the instance with minimum load scores 1.0, maximum load scores 0.0, and intermediate loads are linearly interpolated
- MECHANISM: `(maxLoad - load) / (maxLoad - minLoad)` with uniform fallback (all 1.0) when loads are equal

#### Negative Contracts (what MUST NOT happen)

**BC-17-8: Empty Snapshots Panic (new)**
- GIVEN a `weighted` policy with any scorer configuration
- WHEN Route() is called with an empty snapshot list
- THEN the policy panics (defensive convention matching all other routing policies)
- MECHANISM: Guard at top of Route()

**BC-17-9: No Division by Zero in Scorers (new)**
- GIVEN any combination of zero-valued snapshot fields (zero load, zero utilization, zero FreeKVBlocks)
- WHEN scorers evaluate the snapshots
- THEN all scores are finite (no NaN, no Inf)
- MECHANISM: Each scorer guards its denominators — queue-depth returns 1.0 for uniform load; load-balance uses `1/(1+x)` which is always finite; kv-utilization is `1 - x` which is always finite

#### Error Handling Contracts

**BC-17-10: Old YAML Fields Rejected (new)**
- GIVEN a YAML policy config using the old `cache_weight` or `load_weight` fields
- WHEN loaded via `LoadPolicyBundle`
- THEN a parse error is returned (strict YAML parsing rejects unknown fields)
- MECHANISM: Existing `decoder.KnownFields(true)` in `LoadPolicyBundle` — removing the fields from `RoutingConfig` makes them unknown

### C) Component Interaction

```
CLI (--routing-scorers) ──→ ParseScorerConfigs() ──→ []ScorerConfig
                                                          │
YAML (routing.scorers) ──→ RoutingConfig.Scorers ─────────┤
                                                          ▼
                                                   DeploymentConfig
                                                          │
                                                          ▼
                                             NewRoutingPolicy("weighted", configs)
                                                          │
                                                          ▼
                                              WeightedScoring{scorers, weights}
                                                          │
                                                    Route(req, state)
                                                          │
                                    ┌─────────────────────┼──────────────────────┐
                                    ▼                     ▼                      ▼
                           scoreQueueDepth     scoreKVUtilization      scoreLoadBalance
                                    │                     │                      │
                                    └─────────────────────┼──────────────────────┘
                                                          ▼
                                              Σ clamp(s_i) × w_i per instance
                                                          │
                                                          ▼
                                                   argmax → RoutingDecision
```

**API contracts:**
- `ScorerConfig{Name, Weight}` — value type, YAML-serializable
- `ParseScorerConfigs(string) ([]ScorerConfig, error)` — CLI parser
- `DefaultScorerConfigs() []ScorerConfig` — default profile
- `IsValidScorer(name string) bool` — validation accessor
- `NewRoutingPolicy(name string, scorerConfigs []ScorerConfig) RoutingPolicy` — changed signature

**State changes:** No new mutable state. Scorer weights are immutable after construction. Intermediate scores are ephemeral (stack-allocated per Route() call).

**Extension friction:** 2 files to add a new scorer: implementation function in `sim/routing_scorers.go` + registration in `validScorerNames` map and `newScorer` factory. Meets the 2-file reference target.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Macro plan says ~120 LOC for `routing_scorers.go` | Plan targets ~100 LOC | SIMPLIFICATION: scorers are simpler as package-level functions than as an interface with Name()/Score() methods. Interface can be introduced in PR 18 when observer hook is needed. |
| Macro plan says `NewRoutingPolicy` "accepts scorer configuration instead of individual weights" | Factory takes `[]ScorerConfig` (nil for non-weighted) | ADDITION: Explicit nil semantics for non-weighted policies — cleaner than empty slice |
| Issue #229 proposes `weighted-affinity` as a new policy | This PR evolves `weighted` into a multi-scorer pipeline instead | SCOPE_CHANGE: The design doc (merged in #250) chose the composable approach over a separate policy. More flexible, matches llm-d architecture. |
| Issue #230 says "Remove lines 172-192" from README | Plan removes the misleading demo AND updates the weighted description | ADDITION: Since we're changing the CLI surface, the README weighted section gets a full rewrite, not just removal |
| Issue #230 requests updating `prefix-affinity` description in README | Deferred to PR 18 | DEFERRAL: The prefix-affinity description update makes more sense when the prefix-affinity scorer ships in PR 18. PR 17 only updates the `weighted` description. |
| Macro plan BC-17-4 = "NaN/Inf rejection", parallel dev plan matches | Micro plan keeps BC-17-1 through BC-17-5 matching macro plan, adds BC-17-6 through BC-17-10 as new contracts | ADDITION: Micro plan expands 5 macro BCs to 10 for implementation detail. First 5 preserve macro plan semantics. |
| Macro plan estimates ~315 LOC for `routing_scorers.go` + other files | Micro plan estimates ~275 LOC total | SIMPLIFICATION: Function type instead of interface reduces LOC. ~40 LOC gap is acceptable estimation variance. |

### E) Review Guide

**The tricky part:** The `NewRoutingPolicy` factory signature change cascades through 8+ files. Every `DeploymentConfig{}` construction site in tests needs updating (remove old weight fields, add scorer config where appropriate). Missing a site causes a silent zero-value bug.

**What to scrutinize:** BC-17-5 (load-balance ≈ least-loaded equivalence) — this is the strongest behavioral invariant and validates the scorer math is correct. Also scrutinize the queue-depth scorer's uniform-load edge case (all loads equal → all score 1.0, not division by zero).

**What's safe to skim:** Test file updates in `sim/cluster/` — these are mechanical field renames (remove `RoutingCacheWeight`/`RoutingLoadWeight`, most tests don't use weighted routing). The `examples/` and doc updates are straightforward.

**Known debt:** The scorer contract is currently a function type (`scorerFunc`), not an interface. PR 18 will promote it to an interface to add the observer hook for prefix-affinity. This is intentional — YAGNI for PR 17.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/routing_scorers.go` — ScorerConfig, scorerFunc, 3 scorer implementations, factory, validation, defaults, CLI parser (~100 LOC)
- `sim/routing_scorers_test.go` — scorer unit tests + config parser tests

**Files to modify:**
- `sim/routing.go` — WeightedScoring struct + Route method + NewRoutingPolicy signature (~80 LOC delta)
- `sim/routing_test.go` — rewrite weighted tests, update all NewRoutingPolicy calls
- `sim/bundle.go` — RoutingConfig schema, Validate() (~30 LOC delta)
- `sim/bundle_test.go` — update YAML tests
- `sim/cluster/deployment.go` — replace weight fields with ScorerConfigs (~10 LOC delta)
- `sim/cluster/cluster.go` — update NewRoutingPolicy call (~5 LOC delta)
- `sim/cluster/cluster_test.go` — update DeploymentConfig construction + weight fields
- `sim/cluster/pending_requests_test.go` — update DeploymentConfig construction
- `sim/cluster/cluster_trace_test.go` — update DeploymentConfig construction
- `sim/cluster/evaluation_test.go` — no weighted-specific fields, but verify still compiles
- `cmd/root.go` — replace flag vars, registration, validation, bundle override logic (~50 LOC delta)

**Files to rewrite:**
- `examples/weighted-routing.yaml` — new scorer YAML schema + documentation
- `examples/policy-config.yaml` — update comments

**Files to update:**
- `CLAUDE.md` — CLI flags, RoutingConfig description, weighted policy description
- `README.md` — remove misleading demo (#230), update weighted description, update CLI flags table

**Key decisions:**
- Scorers are package-level functions (not interface methods) — simplest for stateless scorers
- `validScorerNames` is unexported map + `IsValidScorer()` accessor (antipattern rule 8)
- Scorer weights use bare `float64` in YAML (zero is always an error per INV-6, no ambiguity)

**Confirmation:** No dead code — every scorer is exercisable via `--routing-scorers`, every type is used in the routing pipeline, every function is called in tests.

### G) Task Breakdown

---

#### Task 1: Scorer Config Types + Validation + Parser

**Contracts Implemented:** BC-17-4 (invalid config rejection), partial BC-17-2 (config normalization foundation)

**Files:**
- Create: `sim/routing_scorers.go`
- Create: `sim/routing_scorers_test.go`

**Step 1: Write failing tests for ScorerConfig parsing and validation**

Context: We need types and parsing for the `name:weight` scorer configuration format used by both CLI and YAML.

```go
package sim

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseScorerConfigs_ValidInput(t *testing.T) {
	configs, err := ParseScorerConfigs("queue-depth:2,kv-utilization:3,load-balance:1")
	require.NoError(t, err)
	assert.Len(t, configs, 3)
	assert.Equal(t, "queue-depth", configs[0].Name)
	assert.Equal(t, 2.0, configs[0].Weight)
	assert.Equal(t, "kv-utilization", configs[1].Name)
	assert.Equal(t, 3.0, configs[1].Weight)
	assert.Equal(t, "load-balance", configs[2].Name)
	assert.Equal(t, 1.0, configs[2].Weight)
}

func TestParseScorerConfigs_EmptyString_ReturnsNil(t *testing.T) {
	configs, err := ParseScorerConfigs("")
	require.NoError(t, err)
	assert.Nil(t, configs)
}

func TestParseScorerConfigs_InvalidInput(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{"unknown scorer", "unknown-scorer:1"},
		{"missing weight", "queue-depth"},
		{"negative weight", "queue-depth:-1"},
		{"zero weight", "queue-depth:0"},
		{"NaN weight", "queue-depth:NaN"},
		{"Inf weight", "queue-depth:Inf"},
		{"non-numeric weight", "queue-depth:abc"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := ParseScorerConfigs(tt.input)
			assert.Error(t, err)
		})
	}
}

func TestIsValidScorer_KnownNames(t *testing.T) {
	assert.True(t, IsValidScorer("queue-depth"))
	assert.True(t, IsValidScorer("kv-utilization"))
	assert.True(t, IsValidScorer("load-balance"))
	assert.False(t, IsValidScorer("unknown"))
	assert.False(t, IsValidScorer(""))
}

func TestValidScorerNames_Sorted(t *testing.T) {
	names := ValidScorerNames()
	assert.Len(t, names, 3)
	for i := 1; i < len(names); i++ {
		assert.True(t, names[i-1] < names[i], "names must be sorted")
	}
}

func TestDefaultScorerConfigs_ReturnsThreeScorers(t *testing.T) {
	configs := DefaultScorerConfigs()
	assert.Len(t, configs, 3)
	// Verify all names are valid
	for _, c := range configs {
		assert.True(t, IsValidScorer(c.Name), "default scorer %q must be valid", c.Name)
		assert.True(t, c.Weight > 0, "default weight must be positive")
	}
}

func TestNormalizeScorerWeights_PreservesRatio(t *testing.T) {
	configs := []ScorerConfig{
		{Name: "queue-depth", Weight: 3.0},
		{Name: "load-balance", Weight: 2.0},
	}
	weights := normalizeScorerWeights(configs)
	assert.InDelta(t, 0.6, weights[0], 0.001)
	assert.InDelta(t, 0.4, weights[1], 0.001)
	assert.InDelta(t, 1.0, weights[0]+weights[1], 0.001)
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim -run "TestParseScorerConfigs|TestIsValidScorer|TestValidScorerNames|TestDefaultScorerConfigs|TestNormalizeScorerWeights" -v`
Expected: FAIL — functions not defined

**Step 3: Implement ScorerConfig types, parser, validation, and defaults**

In `sim/routing_scorers.go`:
```go
package sim

import (
	"fmt"
	"math"
	"strconv"
	"strings"
)

// ScorerConfig describes a named scorer with a weight for weighted routing.
type ScorerConfig struct {
	Name   string  `yaml:"name"`
	Weight float64 `yaml:"weight"`
}

// scorerFunc computes per-instance scores in [0,1] for a scoring dimension.
type scorerFunc func(snapshots []RoutingSnapshot) map[string]float64

// validScorerNames maps scorer names to validity. Unexported to prevent mutation (antipattern rule 8).
var validScorerNames = map[string]bool{
	"queue-depth":    true,
	"kv-utilization": true,
	"load-balance":   true,
}

// IsValidScorer returns true if name is a recognized scorer.
func IsValidScorer(name string) bool { return validScorerNames[name] }

// ValidScorerNames returns sorted valid scorer names.
func ValidScorerNames() []string { return validNamesList(validScorerNames) }

// DefaultScorerConfigs returns the default scorer configuration for weighted routing.
// Default profile: queue-depth:2, kv-utilization:2, load-balance:1.
func DefaultScorerConfigs() []ScorerConfig {
	return []ScorerConfig{
		{Name: "queue-depth", Weight: 2.0},
		{Name: "kv-utilization", Weight: 2.0},
		{Name: "load-balance", Weight: 1.0},
	}
}

// ParseScorerConfigs parses a comma-separated string of "name:weight" pairs.
// Returns nil for empty input. Returns error for invalid names, non-positive weights,
// NaN, Inf, or malformed input.
func ParseScorerConfigs(s string) ([]ScorerConfig, error) {
	if s == "" {
		return nil, nil
	}
	parts := strings.Split(s, ",")
	configs := make([]ScorerConfig, 0, len(parts))
	for _, part := range parts {
		kv := strings.SplitN(strings.TrimSpace(part), ":", 2)
		if len(kv) != 2 {
			return nil, fmt.Errorf("invalid scorer config %q (expected name:weight)", strings.TrimSpace(part))
		}
		name := strings.TrimSpace(kv[0])
		if !IsValidScorer(name) {
			return nil, fmt.Errorf("unknown scorer %q; valid: %s", name, strings.Join(ValidScorerNames(), ", "))
		}
		weight, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid weight for scorer %q: %w", name, err)
		}
		if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
			return nil, fmt.Errorf("scorer %q weight must be a finite positive number, got %v", name, weight)
		}
		configs = append(configs, ScorerConfig{Name: name, Weight: weight})
	}
	return configs, nil
}

// normalizeScorerWeights returns weights normalized to sum to 1.0.
// Panics if total weight is zero (should be prevented by validation).
func normalizeScorerWeights(configs []ScorerConfig) []float64 {
	total := 0.0
	for _, c := range configs {
		total += c.Weight
	}
	if total <= 0 {
		panic(fmt.Sprintf("scorer weights sum to %f; must be positive", total))
	}
	weights := make([]float64, len(configs))
	for i, c := range configs {
		weights[i] = c.Weight / total
	}
	return weights
}

// newScorer creates a scorer function by name. Panics on unknown name
// (validation should catch this before reaching here).
func newScorer(name string) scorerFunc {
	switch name {
	case "queue-depth":
		return scoreQueueDepth
	case "kv-utilization":
		return scoreKVUtilization
	case "load-balance":
		return scoreLoadBalance
	default:
		panic(fmt.Sprintf("unknown scorer %q", name))
	}
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim -run "TestParseScorerConfigs|TestIsValidScorer|TestValidScorerNames|TestDefaultScorerConfigs|TestNormalizeScorerWeights" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues (unused `newScorer` is OK — used in Task 3)

**Step 6: Commit**

```bash
git add sim/routing_scorers.go sim/routing_scorers_test.go
git commit -m "feat(routing): add ScorerConfig types, parser, and validation (BC-17-4)

- ScorerConfig type for name:weight pairs (YAML-serializable)
- ParseScorerConfigs for CLI flag parsing with NaN/Inf/negative rejection
- IsValidScorer/ValidScorerNames accessors (unexported map, antipattern rule 8)
- DefaultScorerConfigs: queue-depth:2, kv-utilization:2, load-balance:1
- normalizeScorerWeights for INV-6 weight normalization

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Three Scorer Implementations

**Contracts Implemented:** BC-17-1 (score range), BC-17-7 (queue-depth min-max), BC-17-9 (no division by zero)

**Files:**
- Modify: `sim/routing_scorers.go`
- Modify: `sim/routing_scorers_test.go`

**Step 1: Write failing tests for scorer behavior**

Context: Each scorer must return scores in [0,1] for every instance, handle edge cases (zero load, uniform load), and never produce NaN/Inf.

```go
// Add to sim/routing_scorers_test.go:

func TestScoreQueueDepth_MinMaxNormalization(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 10, BatchSize: 0, PendingRequests: 0}, // load=10 → score=0.0
		{ID: "b", QueueDepth: 5, BatchSize: 0, PendingRequests: 0},  // load=5  → score=0.5
		{ID: "c", QueueDepth: 0, BatchSize: 0, PendingRequests: 0},  // load=0  → score=1.0
	}
	scores := scoreQueueDepth(snapshots)
	assert.InDelta(t, 0.0, scores["a"], 0.001)
	assert.InDelta(t, 0.5, scores["b"], 0.001)
	assert.InDelta(t, 1.0, scores["c"], 0.001)
}

func TestScoreQueueDepth_UniformLoad_AllScoreOne(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 5, BatchSize: 3},
		{ID: "b", QueueDepth: 5, BatchSize: 3},
	}
	scores := scoreQueueDepth(snapshots)
	assert.Equal(t, 1.0, scores["a"])
	assert.Equal(t, 1.0, scores["b"])
}

func TestScoreQueueDepth_IncludesPendingRequests(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0, PendingRequests: 5}, // load=5
		{ID: "b", QueueDepth: 5, PendingRequests: 0}, // load=5
		{ID: "c", QueueDepth: 0, PendingRequests: 0}, // load=0 → best
	}
	scores := scoreQueueDepth(snapshots)
	assert.Equal(t, scores["a"], scores["b"], "equal effective load → equal score")
	assert.Greater(t, scores["c"], scores["a"], "lower load → higher score")
}

func TestScoreKVUtilization_InverseUtilization(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", KVUtilization: 0.0},  // score=1.0
		{ID: "b", KVUtilization: 0.5},  // score=0.5
		{ID: "c", KVUtilization: 1.0},  // score=0.0
	}
	scores := scoreKVUtilization(snapshots)
	assert.InDelta(t, 1.0, scores["a"], 0.001)
	assert.InDelta(t, 0.5, scores["b"], 0.001)
	assert.InDelta(t, 0.0, scores["c"], 0.001)
}

func TestScoreLoadBalance_InverseTransform(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 0},   // load=0 → score=1.0
		{ID: "b", QueueDepth: 9},   // load=9 → score=0.1
	}
	scores := scoreLoadBalance(snapshots)
	assert.InDelta(t, 1.0, scores["a"], 0.001)
	assert.InDelta(t, 0.1, scores["b"], 0.001)
}

func TestAllScorers_ReturnScoreForEveryInstance(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "a", QueueDepth: 1, KVUtilization: 0.3},
		{ID: "b", QueueDepth: 2, KVUtilization: 0.7},
		{ID: "c", QueueDepth: 0, KVUtilization: 0.0},
	}
	scorerFns := []struct {
		name string
		fn   scorerFunc
	}{
		{"queue-depth", scoreQueueDepth},
		{"kv-utilization", scoreKVUtilization},
		{"load-balance", scoreLoadBalance},
	}
	for _, sf := range scorerFns {
		t.Run(sf.name, func(t *testing.T) {
			scores := sf.fn(snapshots)
			// INV-2: score for every instance
			assert.Len(t, scores, len(snapshots))
			for _, snap := range snapshots {
				score, ok := scores[snap.ID]
				assert.True(t, ok, "missing score for %s", snap.ID)
				// INV-1: score in [0,1]
				assert.GreaterOrEqual(t, score, 0.0, "score below 0 for %s", snap.ID)
				assert.LessOrEqual(t, score, 1.0, "score above 1 for %s", snap.ID)
				// BC-17-8: no NaN/Inf
				assert.False(t, math.IsNaN(score), "NaN score for %s", snap.ID)
				assert.False(t, math.IsInf(score, 0), "Inf score for %s", snap.ID)
			}
		})
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim -run "TestScoreQueueDepth|TestScoreKVUtilization|TestScoreLoadBalance|TestAllScorers" -v`
Expected: FAIL — functions not defined

**Step 3: Implement three scorer functions**

Add to `sim/routing_scorers.go`:
```go
// scoreQueueDepth computes per-instance queue depth scores using min-max normalization.
// Lower effective load → higher score. All-equal loads → all score 1.0.
// Matches llm-d's queue-scorer semantics.
func scoreQueueDepth(snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	minLoad, maxLoad := math.MaxInt, 0
	for _, snap := range snapshots {
		load := snap.EffectiveLoad()
		if load < minLoad {
			minLoad = load
		}
		if load > maxLoad {
			maxLoad = load
		}
	}
	for _, snap := range snapshots {
		if maxLoad == minLoad {
			scores[snap.ID] = 1.0
		} else {
			load := snap.EffectiveLoad()
			scores[snap.ID] = float64(maxLoad-load) / float64(maxLoad-minLoad)
		}
	}
	return scores
}

// scoreKVUtilization computes per-instance KV utilization scores.
// Lower utilization → higher score: score = 1 - KVUtilization.
// Matches llm-d's kv-cache-utilization-scorer semantics.
func scoreKVUtilization(snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 - snap.KVUtilization
	}
	return scores
}

// scoreLoadBalance computes per-instance load balance scores using inverse transform.
// Lower effective load → higher score: score = 1/(1 + effectiveLoad).
// BLIS-native formula preserving absolute load differences (alternative to min-max).
func scoreLoadBalance(snapshots []RoutingSnapshot) map[string]float64 {
	scores := make(map[string]float64, len(snapshots))
	for _, snap := range snapshots {
		scores[snap.ID] = 1.0 / (1.0 + float64(snap.EffectiveLoad()))
	}
	return scores
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim -run "TestScoreQueueDepth|TestScoreKVUtilization|TestScoreLoadBalance|TestAllScorers" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/routing_scorers.go sim/routing_scorers_test.go
git commit -m "feat(routing): add queue-depth, kv-utilization, load-balance scorers (BC-17-1, BC-17-6, BC-17-8)

- scoreQueueDepth: min-max normalization of EffectiveLoad (llm-d match)
- scoreKVUtilization: 1 - utilization (llm-d match)
- scoreLoadBalance: 1/(1+load) inverse transform (BLIS-native)
- All scorers return [0,1] for every instance (INV-1, INV-2)
- No NaN/Inf for any input (BC-17-8)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: WeightedScoring Pipeline Refactor + Factory Signature Change

**Contracts Implemented:** BC-17-2 (weight normalization), BC-17-6 (argmax), BC-17-8 (empty snapshots panic)

**Files:**
- Modify: `sim/routing.go` (WeightedScoring struct, Route method, NewRoutingPolicy signature)
- Modify: `sim/routing_test.go` (rewrite weighted tests, update ALL factory calls)

**Step 1: Write tests for new scorer pipeline behavior**

Context: The WeightedScoring struct now uses a list of scorers with weights instead of two hardcoded dimensions. The factory signature changes from `(name, cacheWeight, loadWeight)` to `(name, []ScorerConfig)`. All existing tests must be updated for the new signature.

Write the NEW weighted scoring tests in `sim/routing_test.go`, replacing the old WeightedScoring tests. Also update ALL `NewRoutingPolicy` calls for non-weighted policies to pass `nil` instead of `0, 0`.

See Appendix K for complete test code.

**Step 2: Run tests to verify compilation and expected behavior**

Run: `go test ./sim -v`
Expected: PASS (after implementation — run after Step 3)

**Step 3: Implement the refactored WeightedScoring and factory**

Replace the `WeightedScoring` struct, `Route` method, and `NewRoutingPolicy` factory in `sim/routing.go`. See Appendix K for complete implementation.

Key changes:
- `WeightedScoring` struct: `scorers []scorerFunc` + `weights []float64` + `names []string`
- `Route()`: iterate scorers, accumulate weighted scores with [0,1] clamping, argmax
- `NewRoutingPolicy(name string, scorerConfigs []ScorerConfig)`: `case "weighted"` builds scorer pipeline from configs, defaults when nil

**Step 4: Run tests to verify they pass**

Run: `go test ./sim -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/routing.go sim/routing_test.go
git commit -m "feat(routing): refactor WeightedScoring to composable scorer pipeline (BC-17-2, BC-17-4, BC-17-7)

- WeightedScoring now holds []scorerFunc + normalized weights
- Route() iterates scorers, aggregates weighted [0,1] scores, argmax
- NewRoutingPolicy signature: (name, []ScorerConfig) replaces (name, cacheWeight, loadWeight)
- Non-weighted policies pass nil for scorerConfigs
- Weight normalization ensures [3,2,2] ≡ [0.43,0.29,0.29] (INV-6)

BREAKING: NewRoutingPolicy signature changed — callers must be updated

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Bundle YAML Config Changes

**Contracts Implemented:** BC-17-4 (YAML validation), BC-17-10 (old fields rejected)

**Files:**
- Modify: `sim/bundle.go` (RoutingConfig, Validate)
- Modify: `sim/bundle_test.go` (update tests)

**Step 1: Write tests for new YAML schema**

Context: `RoutingConfig` replaces `CacheWeight *float64` and `LoadWeight *float64` with `Scorers []ScorerConfig`. Old YAML fields must be rejected by strict parsing.

**Tests to DELETE (validation model no longer applies):**
- `TestPolicyBundle_Validate_WeightSumNotOne` — weight-sum-to-one check removed
- `TestPolicyBundle_Validate_WeightSumOne` — same
- `TestPolicyBundle_Validate_WeightSumSkippedForNonWeighted` — same
- `TestPolicyBundle_Validate_ZeroParametersAreValid` — zero is now always invalid for scorer weights
- `TestPolicyBundle_Validate_NegativeParameters` cases for `CacheWeight`/`LoadWeight` — fields removed

**Tests to REWRITE (new schema):**
- `TestLoadPolicyBundle_ValidYAML` — YAML now uses `scorers:` list, not `cache_weight`/`load_weight`
- `TestLoadPolicyBundle_ZeroValueIsDistinctFromUnset` — pointer-type test no longer applicable (bare float64 in ScorerConfig); replace with scorer-list-present-vs-absent test
- `TestLoadPolicyBundle_EmptyFields` — CacheWeight nil check → Scorers nil/empty check
- `TestPolicyBundle_Validate_ValidPolicies` — construct with `Scorers` field

**Tests to ADD:**
- `TestPolicyBundle_Validate_InvalidScorerName` — unknown scorer rejected
- `TestPolicyBundle_Validate_InvalidScorerWeight` — negative/zero/NaN/Inf weight rejected
- `TestLoadPolicyBundle_OldFieldsRejected` — old `cache_weight`/`load_weight` YAML → parse error (BC-17-10)

See Appendix K for complete test code.

**Step 2: Implement RoutingConfig changes**

Replace the `RoutingConfig` struct and update `Validate()` in `sim/bundle.go`. See Appendix K for details.

**Step 3: Run tests**

Run: `go test ./sim -run "TestPolicyBundle|TestLoadPolicyBundle" -v`
Expected: PASS

**Step 4: Lint + commit**

```bash
git add sim/bundle.go sim/bundle_test.go
git commit -m "feat(routing): update RoutingConfig YAML schema for scorer pipeline (BC-17-4, BC-17-10)

- RoutingConfig.Scorers replaces CacheWeight/LoadWeight fields
- Validate() checks scorer names and weights (NaN/Inf/negative/zero)
- Old cache_weight/load_weight YAML fields produce parse error (KnownFields)
- Weight sum validation removed (normalization handles any positive weights)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: Cluster Integration (DeploymentConfig + cluster.go)

**Contracts Implemented:** BC-17-3 (non-weighted stability — verified by unchanged test output)

**Files:**
- Modify: `sim/cluster/deployment.go`
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/cluster_test.go`
- Modify: `sim/cluster/pending_requests_test.go`
- Modify: `sim/cluster/cluster_trace_test.go`

**Step 1: Update DeploymentConfig**

Replace `RoutingCacheWeight float64` and `RoutingLoadWeight float64` with `RoutingScorerConfigs []sim.ScorerConfig` in `sim/cluster/deployment.go`. The `ToSimConfig()` method does not propagate scorer configs (they're cluster-level, not instance-level).

**Step 2: Update cluster.go**

Change `sim/cluster/cluster.go:85` from:
```go
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingCacheWeight, config.RoutingLoadWeight),
```
to:
```go
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs),
```

**Step 3: Update all cluster test files**

Remove all `RoutingCacheWeight`/`RoutingLoadWeight` field assignments. For tests that use `weighted` routing policy, add `RoutingScorerConfigs` with appropriate scorers.

The `newTestDeploymentConfig` helper in `cluster_test.go` does not set weight fields (they default to zero), so no change needed there — the factory will use `DefaultScorerConfigs()` when configs is nil.

For specific test files:
- `cluster_test.go:769-770`: Replace `RoutingCacheWeight: 0.6` / `RoutingLoadWeight: 0.4` with `RoutingScorerConfigs: sim.DefaultScorerConfigs()`
- `pending_requests_test.go:25-26`: Same replacement
- `cluster_trace_test.go:102-103`: Same replacement

**Step 4: Run tests**

Run: `go test ./sim/cluster/... -v`
Expected: PASS

**Step 5: Lint + commit**

```bash
git add sim/cluster/deployment.go sim/cluster/cluster.go sim/cluster/cluster_test.go sim/cluster/pending_requests_test.go sim/cluster/cluster_trace_test.go
git commit -m "feat(routing): update DeploymentConfig and cluster for scorer pipeline (BC-17-3)

- DeploymentConfig: RoutingScorerConfigs replaces RoutingCacheWeight/RoutingLoadWeight
- cluster.go: NewRoutingPolicy call passes scorer configs
- All cluster tests updated for new config field
- Non-weighted routing policies unaffected (BC-17-3)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: CLI Flag Changes

**Contracts Implemented:** BC-17-4 (CLI validation)

**Files:**
- Modify: `cmd/root.go`

**Step 1: Update CLI flags and validation**

In `cmd/root.go`:
1. Replace `routingCacheWeight` / `routingLoadWeight` vars (lines 70-71) with `routingScorers string`
2. Remove old flag registrations (lines 568-569)
3. Add `--routing-scorers` flag registration (default empty = use DefaultScorerConfigs)
4. Remove old weight validation block (lines 358-379)
5. Add scorer parsing and validation where weighted policy is detected
6. Update `DeploymentConfig{}` construction (lines 415-416) to use parsed scorer configs
7. Update bundle override logic (lines 300-305) for scorers
8. Update policy config log message for weighted routing

See Appendix K for exact code changes.

**Step 2: Run tests**

Run: `go test ./cmd/... -v && go test ./... -count=1`
Expected: PASS

**Step 3: Lint + commit**

```bash
git add cmd/root.go
git commit -m "feat(cmd): replace --routing-cache-weight/--routing-load-weight with --routing-scorers (BC-17-9)

- New --routing-scorers flag: comma-separated name:weight pairs
- Default: queue-depth:2,kv-utilization:2,load-balance:1
- Validates scorer names, weight positivity, NaN/Inf
- Removes --routing-cache-weight and --routing-load-weight flags
- Updates bundle override logic for scorers YAML field

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 7: Golden Dataset + Invariant Tests

**Contracts Implemented:** BC-17-3 (non-weighted stability), BC-17-5 (load-balance ≈ least-loaded)

**Files:**
- Modify: `sim/routing_scorers_test.go` (add invariant test)
- Potentially modify: golden dataset test expectations

**Step 1: Add BC-17-5 invariant test**

```go
// Add to sim/routing_scorers_test.go:

// TestLoadBalanceOnly_EquivalentToLeastLoaded verifies BC-17-5:
// weighted with load-balance:1 must select the same instance as least-loaded
// for every request, because argmax(1/(1+load)) = argmin(load).
func TestLoadBalanceOnly_EquivalentToLeastLoaded(t *testing.T) {
	loadBalanceOnly := NewRoutingPolicy("weighted", []ScorerConfig{{Name: "load-balance", Weight: 1.0}})
	leastLoaded := NewRoutingPolicy("least-loaded", nil)

	testCases := [][]RoutingSnapshot{
		{
			{ID: "a", QueueDepth: 10, BatchSize: 2},
			{ID: "b", QueueDepth: 3, BatchSize: 1},
			{ID: "c", QueueDepth: 7, BatchSize: 0},
		},
		{
			{ID: "a", QueueDepth: 5, BatchSize: 5, PendingRequests: 3},
			{ID: "b", QueueDepth: 5, BatchSize: 5, PendingRequests: 0},
		},
		{
			{ID: "a", QueueDepth: 0, BatchSize: 0},
			{ID: "b", QueueDepth: 0, BatchSize: 0},
			{ID: "c", QueueDepth: 0, BatchSize: 0},
		},
	}

	for i, snapshots := range testCases {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			req := &Request{ID: fmt.Sprintf("req_%d", i)}
			state := &RouterState{Snapshots: snapshots, Clock: 1000}

			wDecision := loadBalanceOnly.Route(req, state)
			llDecision := leastLoaded.Route(req, state)

			assert.Equal(t, llDecision.TargetInstance, wDecision.TargetInstance,
				"load-balance-only weighted must select same instance as least-loaded")
		})
	}
}
```

**Step 2: Run all tests including golden dataset**

Run: `go test ./... -count=1 -v 2>&1 | tail -30`
Expected: If golden dataset tests fail for `weighted` policy, that's expected — the scoring formula changed. Non-weighted tests must pass unchanged.

**Step 3: Update golden baselines if needed**

If golden dataset tests reference `weighted` routing and fail, update the expected values. Document the rationale: "scoring formula changed from two-dimension (cache capacity + load) to three-dimension scorer pipeline (queue-depth + kv-utilization + load-balance)."

Note: The golden dataset may not exercise `weighted` routing specifically — check the test configs. If no weighted golden tests exist, no update needed.

**Step 4: Verify all tests pass**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add sim/routing_scorers_test.go
git commit -m "test(routing): add BC-17-5 invariant test — load-balance-only ≡ least-loaded

- Verifies argmax(1/(1+load)) = argmin(load) across multiple scenarios
- Covers equal loads, pending requests, and zero-load edge cases
- Invariant test (specification-derived, not golden value)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 8: Documentation Updates

**Contracts Implemented:** N/A (documentation only)

**Files:**
- Rewrite: `examples/weighted-routing.yaml`
- Modify: `examples/policy-config.yaml`
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Step 1: Rewrite weighted-routing.yaml**

Replace the entire file with new scorer pipeline YAML schema and documentation. See Appendix K for content.

**Step 2: Update policy-config.yaml**

Update routing section comments to reference scorers instead of cache_weight/load_weight.

**Step 3: Update CLAUDE.md**

- Update CLI flags section: remove `--routing-cache-weight`/`--routing-load-weight`, add `--routing-scorers`
- Update `routing.go` description: replace WeightedScoring description
- Update "Adding New Policy Templates" section if it references weight params
- Update "Current Implementation Focus": note PR17 completed

**Step 4: Update README.md**

- Remove misleading demo section (lines 171-191) per #230
- Update `weighted` policy description to reference scorer pipeline
- Update CLI flags table (remove old weight flags, add `--routing-scorers`)
- Update routing policy list to describe new `weighted` behavior

**Step 5: Verify build**

Run: `go build ./... && golangci-lint run ./...`
Expected: PASS

**Step 6: Commit**

```bash
git add examples/weighted-routing.yaml examples/policy-config.yaml CLAUDE.md README.md
git commit -m "docs: update documentation for composable scorer framework

- Rewrite weighted-routing.yaml with scorer pipeline YAML schema
- Update policy-config.yaml comments for scorers
- Update CLAUDE.md: CLI flags, routing.go description, implementation focus
- Update README.md: remove misleading demo (#230), update weighted description

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-17-1 | Task 2 | Unit | TestAllScorers_ReturnScoreForEveryInstance |
| BC-17-2 | Task 3 | Unit | TestWeightedScoring_WeightsNormalized (rewritten) |
| BC-17-3 | Task 7 | Regression | Golden dataset non-weighted tests (unchanged) |
| BC-17-4 | Task 1 | Unit | TestParseScorerConfigs_InvalidInput |
| BC-17-5 | Task 7 | Invariant | TestLoadBalanceOnly_EquivalentToLeastLoaded |
| BC-17-6 | Task 3 | Unit | TestWeightedScoring_HighestScoreWins (rewritten) |
| BC-17-7 | Task 2 | Unit | TestScoreQueueDepth_MinMaxNormalization |
| BC-17-8 | Task 3 | Failure | TestWeightedScoring_EmptySnapshots_Panics |
| BC-17-9 | Task 2 | Unit | TestAllScorers_ReturnScoreForEveryInstance (NaN/Inf check) |
| BC-17-10 | Task 4 | Unit | TestLoadPolicyBundle_OldFieldsRejected |

**Golden dataset update strategy:** The golden dataset (`testdata/goldendataset.json`) uses pre-configured test suites. If any suite uses `weighted` routing, its expected values will change (scoring formula changed). Non-weighted suites must remain byte-identical. Check in Task 7 and update if needed.

**Invariant test (mandatory per template rule 6):** BC-17-5 is a specification-derived invariant (mathematical equivalence of argmax and argmin over monotone functions), not a golden value. It would catch bugs even if the scorer implementation were wrong.

**Design doc invariant coverage:**
- INV-1 (Score Range): Covered by BC-17-1 tests
- INV-2 (Score Completeness): Covered by BC-17-1 tests
- INV-3 (Determinism): Covered implicitly by golden dataset tests (same seed + same workload = same output). No explicit determinism test added because PR 17 scorers are pure functions with no state — determinism is guaranteed by construction.
- INV-4 (Observer Consistency): Deferred to PR 18 — PR 17 has only stateless scorers, no observers.
- INV-5 (Backward Stability): Covered by BC-17-3 golden dataset tests
- INV-6 (Weight Normalization): Covered by BC-17-2 tests
- INV-7 (Prefix Cache Conservation): Deferred to PR 18 — prefix cache not introduced in PR 17.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Golden dataset tests fail for non-weighted policies | Low | High | BC-17-3: non-weighted factory cases are untouched | Task 7 |
| Missing DeploymentConfig construction site | Medium | High | Phase 0 audit found 16 sites in source code; Task 5 addresses all | Task 5 |
| Scorer scores outside [0,1] for edge case | Low | Medium | Clamping in Route() pipeline + per-scorer edge case tests | Task 2, 3 |
| NewRoutingPolicy signature change missed in some caller | Low | High | Compile error catches missing sites; only 1 production caller | Task 3 |
| CLI backward compatibility surprise | Low | Low | Clean break is intentional (design doc D-5); error message references new flag | Task 6 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — scorers are functions, not interfaces (YAGNI for PR 17)
- [x] No feature creep — 3 scorers matching macro plan scope, no observer hook (PR 18)
- [x] No unexercised flags — `--routing-scorers` immediately usable with all 3 scorers
- [x] No partial implementations — every scorer is complete and tested
- [x] No breaking changes without contract updates — BC-17-10 documents old YAML rejection
- [x] No hidden global state — validScorerNames is unexported, scorers are stateless
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (testutil for golden dataset)
- [x] CLAUDE.md updated for new CLI flags and routing description
- [x] No stale references — will grep for "planned for PR 17" after completion
- [x] Deviation log reviewed — 4 deviations, all justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7→8)
- [x] All contracts mapped to tasks
- [x] Golden dataset update documented (Task 7)
- [x] Construction site audit: DeploymentConfig has 16 construction sites in source code (1 in cmd/root.go, 15 in test files). Of the test sites: 3 cluster test files explicitly set weight fields (cluster_test.go:769, pending_requests_test.go:25, cluster_trace_test.go:102) and need changes in Task 5; 2 more (evaluation_test.go, workload_test.go) use struct literals but don't set weight fields — they compile without changes after field removal
- [x] New CLI flag `--routing-scorers` validated for empty, unknown names, NaN, Inf, negative, zero
- [x] No silent `continue` — scorer pipeline processes all instances, no early exit
- [x] No map iteration for float accumulation — scorers iterate slices (snapshots), not maps
- [x] Library code (sim/) never calls logrus.Fatalf — all errors returned or panicked with context
- [x] No resource allocation loops — scorers are pure computations, no rollback needed
- [x] validScorerNames is unexported with IsValidScorer() accessor
- [x] ScorerConfig.Weight is bare float64 — zero is always an error (no ambiguity), pointer not needed
- [x] YAML loading uses existing KnownFields(true) — old cache_weight/load_weight cause parse errors
- [x] scoreQueueDepth guards maxLoad==minLoad (division by zero); scoreLoadBalance uses 1/(1+x) always finite
- [x] Scorer contract accommodates stateless (PR 17) and future stateful (PR 18) implementations
- [x] No method spans multiple concerns — scorers score, pipeline aggregates, factory constructs
- [x] ScorerConfig grouped under routing — not added to SimConfig or DeploymentConfig top-level
- [x] Will grep for "planned for PR 17" references after implementation
- [x] Macro plan status will be updated to mark PR 17 as completed

---

## Appendix K: File-Level Implementation Details

### File: `sim/routing_scorers.go`

**Purpose:** Scorer config types, 3 scorer implementations, factory, validation, CLI parser.

See Task 1 (Step 3) and Task 2 (Step 3) for complete implementation.

**Key implementation notes:**
- `validScorerNames` is unexported (antipattern rule 8)
- `scoreQueueDepth` uses `math.MaxInt` for initial minLoad — safe because `EffectiveLoad()` returns non-negative int
- `normalizeScorerWeights` panics on zero sum — validation prevents this, but defensive
- No RNG usage, no metrics, no events, no state mutation

### File: `sim/routing.go` (modifications)

**Purpose:** Refactored WeightedScoring struct, Route method, and NewRoutingPolicy factory.

**WeightedScoring struct replacement (lines 116-119):**
```go
type WeightedScoring struct {
	scorers []scorerFunc
	weights []float64 // normalized to sum to 1.0
	names   []string  // for debugging/logging
}
```

**Route method replacement (lines 122-167):**
```go
func (ws *WeightedScoring) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// Compute composite scores from all scorers
	scores := make(map[string]float64, len(snapshots))
	for i, scorer := range ws.scorers {
		dimScores := scorer(snapshots)
		for _, snap := range snapshots {
			s := dimScores[snap.ID]
			// Clamp to [0,1] per INV-1
			if s < 0 {
				s = 0
			}
			if s > 1 {
				s = 1
			}
			scores[snap.ID] += s * ws.weights[i]
		}
	}

	// Argmax: select instance with highest composite score.
	// Ties broken by first occurrence in snapshot order (strict >).
	bestScore := -1.0
	bestIdx := 0
	for i, snap := range snapshots {
		if scores[snap.ID] > bestScore {
			bestScore = scores[snap.ID]
			bestIdx = i
		}
	}

	return RoutingDecision{
		TargetInstance: snapshots[bestIdx].ID,
		Reason:         fmt.Sprintf("weighted-scoring (score=%.3f)", bestScore),
		Scores:         scores,
	}
}
```

**NewRoutingPolicy factory replacement (lines 247-274):**
```go
func NewRoutingPolicy(name string, scorerConfigs []ScorerConfig) RoutingPolicy {
	if !IsValidRoutingPolicy(name) {
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	case "least-loaded":
		return &LeastLoaded{}
	case "weighted":
		if len(scorerConfigs) == 0 {
			scorerConfigs = DefaultScorerConfigs()
		}
		scorers := make([]scorerFunc, len(scorerConfigs))
		names := make([]string, len(scorerConfigs))
		for i, cfg := range scorerConfigs {
			scorers[i] = newScorer(cfg.Name)
			names[i] = cfg.Name
		}
		weights := normalizeScorerWeights(scorerConfigs)
		return &WeightedScoring{scorers: scorers, weights: weights, names: names}
	case "prefix-affinity":
		return &PrefixAffinity{prefixMap: make(map[string]string)}
	case "always-busiest":
		return &AlwaysBusiest{}
	default:
		panic(fmt.Sprintf("unhandled routing policy %q", name))
	}
}
```

### File: `sim/bundle.go` (modifications)

**RoutingConfig replacement (lines 32-36):**
```go
type RoutingConfig struct {
	Policy  string         `yaml:"policy"`
	Scorers []ScorerConfig `yaml:"scorers"`
}
```

**Validate() update (lines 130-143):**
Remove old `validateFloat("cache_weight", ...)` / `validateFloat("load_weight", ...)` and weight sum check. Replace with:
```go
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
```

### File: `sim/cluster/deployment.go` (modifications)

**Field replacement (lines 34-35):**
```go
// Replace:
// RoutingCacheWeight float64
// RoutingLoadWeight  float64
// With:
RoutingScorerConfigs []sim.ScorerConfig // for weighted routing scorer pipeline
```

### File: `sim/cluster/cluster.go` (modifications)

**Line 85 replacement:**
```go
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs),
```

### File: `cmd/root.go` (modifications)

**Variable replacement (lines 70-71):**
```go
// Replace:
// routingCacheWeight float64
// routingLoadWeight  float64
// With:
routingScorers string // comma-separated name:weight pairs for weighted routing
```

**Flag registration replacement (lines 568-569):**
```go
// Replace:
// runCmd.Flags().Float64Var(&routingCacheWeight, "routing-cache-weight", 0.6, "...")
// runCmd.Flags().Float64Var(&routingLoadWeight, "routing-load-weight", 0.4, "...")
// With:
runCmd.Flags().StringVar(&routingScorers, "routing-scorers", "", "Scorer weights for weighted routing (e.g., queue-depth:2,kv-utilization:2,load-balance:1). Default: queue-depth:2,kv-utilization:2,load-balance:1")
```

**Bundle override logic (lines 297-305):**
Replace cache_weight/load_weight overrides with scorers override:
```go
if len(bundle.Routing.Scorers) > 0 && !cmd.Flags().Changed("routing-scorers") {
	// Use YAML scorers directly — they're already validated
	parsedScorerConfigs = bundle.Routing.Scorers
}
```

**Weight validation block (lines 358-379):**
Replace with scorer parsing and logging:
```go
var parsedScorerConfigs []sim.ScorerConfig
if routingPolicy == "weighted" {
	if routingScorers != "" {
		var err error
		parsedScorerConfigs, err = sim.ParseScorerConfigs(routingScorers)
		if err != nil {
			logrus.Fatalf("Invalid --routing-scorers: %v", err)
		}
	}
	// Log active scorer configuration
	if len(parsedScorerConfigs) == 0 {
		parsedScorerConfigs = sim.DefaultScorerConfigs()
	}
	scorerStrs := make([]string, len(parsedScorerConfigs))
	for i, sc := range parsedScorerConfigs {
		scorerStrs[i] = fmt.Sprintf("%s:%.1f", sc.Name, sc.Weight)
	}
	logrus.Infof("Weighted routing scorers: %s", strings.Join(scorerStrs, ", "))
}
```

**DeploymentConfig construction (lines 414-416):**
```go
// Replace:
// RoutingCacheWeight: routingCacheWeight,
// RoutingLoadWeight:  routingLoadWeight,
// With:
RoutingScorerConfigs: parsedScorerConfigs,
```

### File: `examples/weighted-routing.yaml` (rewrite)

```yaml
# BLIS Weighted Routing — Composable Scorer Pipeline
#
# The `weighted` routing policy aggregates multiple scoring dimensions,
# each evaluating instances on a [0,1] scale. Scores are combined with
# configurable weights: score = Σ clamp(s_i) × w_i, then argmax.
#
# Available scorers:
#   queue-depth     — min-max normalization of effective load (llm-d match)
#   kv-utilization  — inverse utilization: 1 - KVUtilization (llm-d match)
#   load-balance    — inverse transform: 1/(1 + effectiveLoad) (BLIS-native)
#
# Default profile (when --routing-scorers is not specified):
#   queue-depth:2, kv-utilization:2, load-balance:1
#
# ============================================================================
# TRY IT: Compare scorer configurations
# ============================================================================
#
#   # Default profile (balanced)
#   ./simulation_worker run \
#     --model meta-llama/llama-3.1-8b-instruct \
#     --num-instances 4 --routing-policy weighted \
#     --num-requests 500 --rate 1000 \
#     --trace-level decisions --summarize-trace
#
#   # Load-balance only (equivalent to least-loaded)
#   ./simulation_worker run \
#     --model meta-llama/llama-3.1-8b-instruct \
#     --num-instances 4 --routing-policy weighted \
#     --routing-scorers "load-balance:1" \
#     --num-requests 500 --rate 1000 \
#     --trace-level decisions --summarize-trace
#
#   # KV-utilization dominant (prefer instances with free memory)
#   ./simulation_worker run \
#     --model meta-llama/llama-3.1-8b-instruct \
#     --num-instances 4 --routing-policy weighted \
#     --routing-scorers "kv-utilization:5,queue-depth:1" \
#     --num-requests 500 --rate 1000 \
#     --trace-level decisions --summarize-trace
#
# ============================================================================
# Adding new scorers
# ============================================================================
#
# To add a new scorer (e.g., for prefix affinity — coming in PR 18):
# 1. Implement the scorer function in sim/routing_scorers.go
# 2. Register the name in validScorerNames + newScorer factory
# Extension friction: 2 files.

admission:
  policy: always-admit

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
  policy: constant

scheduler: fcfs
```

### File: `examples/policy-config.yaml` (modifications)

Update routing section comments:
```yaml
routing:
  policy: round-robin
  # scorers:                              # scorer config (only for weighted)
  #   - name: queue-depth
  #     weight: 2.0
  #   - name: kv-utilization
  #     weight: 2.0
  #   - name: load-balance
  #     weight: 1.0
  # See examples/weighted-routing.yaml for weighted routing details.
```
