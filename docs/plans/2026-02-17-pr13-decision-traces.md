# Decision Traces + Counterfactual Analysis Implementation Plan (PR13)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a "flight recorder" to cluster simulations that captures every admission and routing decision, then answers the question: *"For each request, did the router make a good choice — and how much better could it have done?"*

**The problem today:** BLIS tells you *what happened* (latency, throughput, anomaly counts) but not *why*. If a simulation shows poor P99 latency, you can't tell whether the routing policy made bad placement decisions, the admission policy rejected the wrong requests, or load was simply too high. You see the outcome but not the reasoning.

**What this PR adds:**
1. **Decision traces** — a log of every admission decision ("request_42 was rejected by token-bucket at tick 5000") and every routing decision ("request_43 was sent to instance_2 because it had the highest weighted score of 0.87").
2. **Counterfactual analysis** — for each routing decision, the trace also records the top-K alternative instances the request *could* have gone to, ranked by score. The **regret** metric captures how much better the best unchosen alternative was (regret = 0 means the router picked the best option).
3. **Trace summary** — aggregate statistics: how many requests were admitted vs rejected, how evenly were requests distributed across instances, what was the mean and max regret across all decisions.

**Why this matters for evolutionary policy optimization:** The fitness score (PR9) tells the optimizer "policy A scored 0.72 and policy B scored 0.68." Traces tell it *why*: "Policy A had 0 regret on 95% of decisions but made 5 catastrophic routing choices. Policy B was mediocre everywhere." This is the observability layer that makes policy evolution informed rather than blind.

**Architecture:** New `sim/trace/` package (pure data types, no dependencies) stores the trace records. Recording hooks in `sim/cluster/cluster_event.go` capture decisions as they happen. Counterfactual computation and the `EvaluationResult` wrapper live in `sim/cluster/`. Zero overhead when tracing is off (`--trace-level none`, the default).

**Macro Plan Reference:** Phase 4, PR 13 in `docs/plans/2026-02-11-macro-implementation-plan-v2.md`

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

**What this PR builds:** A decision-tracing system that records every admission and routing policy decision during cluster simulation, with optional counterfactual ("what-if") analysis that ranks alternative routing choices and computes regret (how much better the best alternative was).

**Where it fits:** This is the observability layer between the existing metrics system (PR9: "how well did the policy perform?") and the future adapter layer (PR15: "feed results to an evolutionary optimizer"). Traces answer the question metrics cannot: "where specifically did the policy make suboptimal decisions?"

**Adjacent blocks:**
- `sim/cluster/cluster_event.go` — recording hooks in AdmissionDecisionEvent and RoutingDecisionEvent
- `sim/cluster/metrics.go` — EvaluationResult wraps existing RawMetrics + FitnessResult + trace
- `sim/routing.go` — reads RoutingDecision.Scores for counterfactual analysis (no modification)
- `cmd/root.go` — new CLI flags `--trace-level`, `--counterfactual-k`, `--summarize-trace`

**Deviations:** `--trace-level detailed` deferred (see Deviation Log). No interface changes — frozen interfaces untouched.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Zero Overhead Default**
- GIVEN `--trace-level none` (default)
- WHEN simulation runs
- THEN no `SimulationTrace` is allocated, no recording calls are made, no additional memory used
- MECHANISM: `ClusterSimulator.trace` field is nil; event Execute methods check `cs.trace != nil` before recording

**BC-2: Admission Recording**
- GIVEN `--trace-level decisions`
- WHEN a request passes through AdmissionDecisionEvent.Execute
- THEN an AdmissionRecord is appended to the trace with request ID, clock, admitted flag, and reason string
- MECHANISM: `cs.trace.RecordAdmission(...)` called inside Execute after `Admit()` returns

**BC-3: Routing Recording**
- GIVEN `--trace-level decisions`
- WHEN a request passes through RoutingDecisionEvent.Execute
- THEN a RoutingRecord is appended to the trace with request ID, clock, chosen instance, reason, and scores
- MECHANISM: `cs.trace.RecordRouting(...)` called inside Execute after `Route()` returns

**BC-4: Counterfactual Candidates (Scored Policies)**
- GIVEN `--counterfactual-k N` where N > 0, and routing policy produces non-nil Scores (e.g., WeightedScoring)
- WHEN a routing decision is recorded
- THEN the top-N candidates are stored sorted by policy score descending, with tie-breaking by instance ID
- MECHANISM: `computeCounterfactual()` sorts candidates by RoutingDecision.Scores values

**BC-5: Counterfactual Candidates (Non-Scored Policies)**
- GIVEN `--counterfactual-k N` where N > 0, and routing policy produces nil Scores (e.g., RoundRobin, LeastLoaded)
- WHEN a routing decision is recorded
- THEN candidates use synthetic load-based scores: `-(QueueDepth + BatchSize)`; lower load → higher score
- MECHANISM: `computeCounterfactual()` falls back to negative-load scoring when Scores is nil

**BC-6: Regret Computation**
- GIVEN a routing decision with candidates
- WHEN regret is computed
- THEN `regret = max(all scores) - score(chosen)`; clamped to 0 if chosen is best
- MECHANISM: `computeCounterfactual()` returns regret alongside candidates

**BC-7: Trace Summary**
- GIVEN a populated SimulationTrace
- WHEN `Summarize()` is called
- THEN returns TraceSummary with correct TotalDecisions, AdmittedCount, RejectedCount, UniqueTargets, TargetDistribution, MeanRegret, MaxRegret
- MECHANISM: Iterates admission and routing records, aggregates statistics

**BC-8: EvaluationResult Wrapping**
- GIVEN completed simulation with RawMetrics, optional FitnessResult, optional trace, optional summary
- WHEN `NewEvaluationResult()` is called
- THEN returns EvaluationResult bundling all available data plus SimDuration and WallTime metadata
- MECHANISM: Constructor accepts all fields; nil-safe for optional parameters

**BC-9: CLI Exercisability**
- GIVEN `--trace-level decisions --counterfactual-k 5 --summarize-trace`
- WHEN simulation completes
- THEN trace summary is printed to stdout with decision counts, target distribution, and regret statistics
- MECHANISM: CLI constructs EvaluationResult, prints Summary fields when non-nil

#### Negative Contracts

**NC-1: No Interface Changes**
- GIVEN the INTERFACE FREEZE (PR8)
- WHEN PR13 is implemented
- THEN AdmissionPolicy, RoutingPolicy, PriorityPolicy, and InstanceScheduler interfaces MUST NOT be modified
- MECHANISM: Tracing hooks are in event Execute methods, not in policy interfaces

**NC-2: No Simulation Behavior Change**
- GIVEN any trace level
- WHEN simulation runs with identical inputs
- THEN routing decisions, admission decisions, and all metrics are identical regardless of trace level
- MECHANISM: Tracing is observation-only; recording happens after policy decisions, never before

**NC-3: No Golden Dataset Regression**
- GIVEN existing golden dataset tests
- WHEN PR13 code is merged
- THEN all existing tests pass with no changes to testdata/goldendataset.json
- MECHANISM: Default trace-level is "none"; no behavioral changes to simulation

#### Error Handling Contracts

**EC-1: Invalid Trace Level**
- GIVEN `--trace-level foobar`
- WHEN CLI validates inputs
- THEN exits with `logrus.Fatalf("Unknown trace level %q. Valid: none, decisions")`
- MECHANISM: `trace.IsValidTraceLevel()` check in CLI before simulation

**EC-2: Counterfactual with Zero K**
- GIVEN `--counterfactual-k 0` (default)
- WHEN routing is recorded
- THEN no candidates are computed, regret is 0
- MECHANISM: `computeCounterfactual()` returns `nil, 0` when k <= 0

**EC-3: Empty Trace Summary**
- GIVEN an empty SimulationTrace (no records)
- WHEN `Summarize()` is called
- THEN returns TraceSummary with all zero values and empty TargetDistribution
- MECHANISM: Summarize handles empty slices gracefully

### C) Component Interaction

```
┌─────────────────────────────────────────────────────┐
│                  cmd/root.go                         │
│  --trace-level, --counterfactual-k, --summarize-trace│
│                                                       │
│  Constructs DeploymentConfig with TraceLevel/K        │
│  After Run(): builds EvaluationResult, prints summary │
└───────────────┬───────────────────────────────────────┘
                │ creates
                ▼
┌───────────────────────────────────────┐
│       sim/cluster/cluster.go           │
│  ClusterSimulator                      │
│    trace *trace.SimulationTrace (nil?) │
│    Trace() accessor                    │
└───────┬───────────────────┬───────────┘
        │ events call       │
        ▼                   ▼
┌─────────────────┐  ┌───────────────────────┐
│ cluster_event.go│  │ counterfactual.go      │
│ Admission hook: │  │ computeCounterfactual()│
│  RecordAdmission│  │ scores + snapshots → k │
│ Routing hook:   │  │ candidates + regret    │
│  RecordRouting  │  └───────────────────────┘
└────────┬────────┘
         │ records into
         ▼
┌────────────────────────────┐
│      sim/trace/             │
│  trace.go: SimulationTrace  │
│  record.go: Records         │
│  summary.go: TraceSummary   │
│  (no sim/ dependency)       │
└────────────────────────────┘
```

**API Contracts:**
- `trace.NewSimulationTrace(config TraceConfig) *SimulationTrace` — creates collector
- `(*SimulationTrace).RecordAdmission(AdmissionRecord)` — appends record
- `(*SimulationTrace).RecordRouting(RoutingRecord)` — appends record
- `trace.Summarize(*SimulationTrace) *TraceSummary` — computes aggregate stats
- `trace.IsValidTraceLevel(string) bool` — validates level string
- `cluster.computeCounterfactual(chosenID, scores, snapshots, k)` — unexported helper
- `cluster.NewEvaluationResult(...)` — constructs wrapper

**State ownership:**
- `*SimulationTrace` owned by `ClusterSimulator`; created in constructor, populated during Run(), read after Run()
- `*TraceSummary` created from trace after Run(); immutable once created
- `*EvaluationResult` created in CLI after Run(); bundles all results

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| `--trace-level` accepts `none/decisions/detailed` | Only `none` and `decisions` accepted | **DEFERRAL:** `detailed` requires per-step event tracing in per-instance Simulator (frozen interface boundary). Reserved for future PR. |
| `TraceConfig` listed as in-scope type | `TraceConfig` is a simple struct in `sim/trace/` with Level + CounterfactualK | **SIMPLIFICATION:** No additional config fields needed at this stage |
| `EvaluationResult` has `PolicyID` and `WorkloadID` fields | Deferred to PR15 (not included in struct) | **DEFERRAL:** These are adapter-supplied metadata. Adding unused fields is dead code; PR15 will add them when needed. |
| `EvaluationResult.Fitness` is `map[string]float64` (G.3) | `*FitnessResult` pointer | **CORRECTION:** Reuses existing PR9 `FitnessResult` type which contains both `Score` and `Components map[string]float64`. More type-safe than raw map. |
| `DecisionTrace` listed as separate type | Folded into `SimulationTrace` | **SIMPLIFICATION:** Macro plan envisioned `DecisionTrace` as a sub-component. Micro plan uses `SimulationTrace` directly with `Admissions` and `Routings` slices — simpler, no extra indirection. |
| `TopKCandidates` listed as in-scope type | Replaced by `[]CandidateScore` field on `RoutingRecord` | **SIMPLIFICATION:** No wrapper type needed. `RoutingRecord.Candidates` is a `[]CandidateScore` slice. |
| `Regret` listed as in-scope type | `float64` field on `RoutingRecord` | **SIMPLIFICATION:** Single scalar value — a dedicated type adds no value. |
| Macro plan Files Changed: only `sim/trace/*.go` and `cmd/root.go` | Also creates `sim/cluster/counterfactual.go`, `sim/cluster/evaluation.go`; modifies `sim/cluster/deployment.go`, `cluster.go`, `cluster_event.go` | **ADDITION:** Counterfactual computation needs `sim.RoutingSnapshot` (in `sim/cluster/` to avoid `sim/trace/` importing `sim/`). `EvaluationResult` needs `RawMetrics`+`FitnessResult` (both in `sim/cluster/`). Event hooks require `cluster_event.go` modifications. |
| `sim/trace/record.go` estimated ~150 LOC, `summary.go` ~200 LOC, `trace.go` ~100 LOC | ~40 LOC, ~60 LOC, ~50 LOC respectively | **SIMPLIFICATION:** Record types are pure data structs; complexity lives in `computeCounterfactual()` (in `sim/cluster/`). Summary aggregation is straightforward iteration. |
| Macro plan G.1 comment: `Candidates []CandidateScore` on `RoutingDecision` | Candidates stored on `trace.RoutingRecord`, not `sim.RoutingDecision` | **CORRECTION:** Adding fields to `RoutingDecision` would modify the frozen interface surface. Candidates are computed in the event pipeline and stored on the trace record instead. |

### E) Review Guide

1. **THE TRICKY PART:** Counterfactual computation in `computeCounterfactual()` — the fallback from policy scores to load-based synthetic scores when `RoutingDecision.Scores` is nil. Verify deterministic tie-breaking (by instance ID) and correct regret formula.

2. **WHAT TO SCRUTINIZE:** BC-1 (zero overhead) — verify that `--trace-level none` truly does zero work. BC-4 vs BC-5 — scored vs non-scored policy counterfactual paths. The event hook in `cluster_event.go` — verify admission `reason` is now captured (was previously discarded with `_`). The `copyScores()` defensive copy in the routing recording — ensures trace data isn't corrupted by map reference sharing.

3. **WHAT'S SAFE TO SKIM:** Record type definitions (pure data), TraceSummary aggregation (straightforward iteration), EvaluationResult constructor (simple assignment), CLI flag registration (mechanical).

4. **KNOWN DEBT:** `--trace-level detailed` deferred. Raw trace JSON export to file not yet implemented (future PR or adapter concern). Comment on line 119 of `sim/cluster/metrics.go` says "Full decision-trace-based detection deferred to PR13" — should be updated/removed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/trace/trace.go` — TraceLevel, TraceConfig, SimulationTrace, recording methods (~50 LOC)
- `sim/trace/record.go` — AdmissionRecord, RoutingRecord, CandidateScore (~40 LOC)
- `sim/trace/summary.go` — TraceSummary, Summarize() (~60 LOC)
- `sim/trace/trace_test.go` — tests for trace recording + IsValidTraceLevel (~70 LOC)
- `sim/trace/summary_test.go` — tests for Summarize (~50 LOC)
- `sim/cluster/counterfactual.go` — computeCounterfactual() (~55 LOC)
- `sim/cluster/counterfactual_test.go` — tests for counterfactual computation (~90 LOC)
- `sim/cluster/evaluation.go` — EvaluationResult, NewEvaluationResult (~35 LOC)
- `sim/cluster/evaluation_test.go` — tests for EvaluationResult (~40 LOC)

**Files to modify:**
- `sim/cluster/deployment.go` — add TraceLevel, CounterfactualK fields (~5 LOC)
- `sim/cluster/cluster.go` — add trace field, constructor wiring, Trace() accessor (~20 LOC)
- `sim/cluster/cluster_event.go` — admission + routing recording hooks (~25 LOC)
- `cmd/root.go` — new flags, validation, trace config, summary output (~45 LOC)

**Key decisions:**
- `sim/trace/` has zero dependencies on `sim/` — pure data types + recording logic
- Counterfactual computation lives in `sim/cluster/` (needs both `sim.RoutingSnapshot` and `trace.CandidateScore`)
- Load-based fallback scoring: `-(QueueDepth + BatchSize)` for non-scoring policies
- EvaluationResult constructed in CLI, used for trace summary printing

**Confirmation:** No dead code — `--trace-level decisions --counterfactual-k 5 --summarize-trace` exercises all trace + counterfactual + summary code; EvaluationResult used for printing summary; IsValidTraceLevel used by CLI validation.

### G) Task Breakdown

---

### Task 1: Core Trace Types + Admission/Routing Recording

**Contracts Implemented:** BC-1, BC-2, BC-3, EC-2, EC-3

**Files:**
- Create: `sim/trace/trace.go`
- Create: `sim/trace/record.go`
- Create: `sim/trace/trace_test.go`

**Step 1: Write failing tests for trace recording**

Context: We test that SimulationTrace can record admission and routing records and that IsValidTraceLevel rejects invalid levels.

```go
// sim/trace/trace_test.go
package trace

import (
	"testing"
)

func TestSimulationTrace_RecordAdmission_AppendsRecord(t *testing.T) {
	// GIVEN a trace configured for decisions
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions, CounterfactualK: 0})

	// WHEN an admission record is recorded
	st.RecordAdmission(AdmissionRecord{
		RequestID: "req_1",
		Clock:     1000,
		Admitted:  true,
		Reason:    "always-admit",
	})

	// THEN the trace contains one admission record with correct data
	if len(st.Admissions) != 1 {
		t.Fatalf("expected 1 admission, got %d", len(st.Admissions))
	}
	if st.Admissions[0].RequestID != "req_1" {
		t.Errorf("expected request ID req_1, got %s", st.Admissions[0].RequestID)
	}
	if !st.Admissions[0].Admitted {
		t.Error("expected admitted=true")
	}
}

func TestSimulationTrace_RecordRouting_AppendsRecord(t *testing.T) {
	// GIVEN a trace configured for decisions
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions, CounterfactualK: 0})

	// WHEN a routing record is recorded
	st.RecordRouting(RoutingRecord{
		RequestID:      "req_1",
		Clock:          2000,
		ChosenInstance: "instance_0",
		Reason:         "least-loaded (load=0)",
		Scores:         nil,
	})

	// THEN the trace contains one routing record with correct data
	if len(st.Routings) != 1 {
		t.Fatalf("expected 1 routing, got %d", len(st.Routings))
	}
	if st.Routings[0].ChosenInstance != "instance_0" {
		t.Errorf("expected instance_0, got %s", st.Routings[0].ChosenInstance)
	}
}

func TestSimulationTrace_MultipleRecords_PreservesOrder(t *testing.T) {
	// GIVEN a trace
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})

	// WHEN multiple records are added
	st.RecordAdmission(AdmissionRecord{RequestID: "req_1", Clock: 100, Admitted: true, Reason: "ok"})
	st.RecordAdmission(AdmissionRecord{RequestID: "req_2", Clock: 200, Admitted: false, Reason: "rejected"})
	st.RecordRouting(RoutingRecord{RequestID: "req_1", Clock: 150, ChosenInstance: "i_0", Reason: "rr"})

	// THEN order is preserved
	if len(st.Admissions) != 2 {
		t.Fatalf("expected 2 admissions, got %d", len(st.Admissions))
	}
	if st.Admissions[0].RequestID != "req_1" || st.Admissions[1].RequestID != "req_2" {
		t.Error("admission order not preserved")
	}
	if len(st.Routings) != 1 || st.Routings[0].RequestID != "req_1" {
		t.Error("routing record mismatch")
	}
}

func TestIsValidTraceLevel_ValidLevels(t *testing.T) {
	tests := []struct {
		level string
		valid bool
	}{
		{"none", true},
		{"decisions", true},
		{"", true}, // empty defaults to none
		{"detailed", false},
		{"foobar", false},
		{"NONE", false}, // case-sensitive
	}
	for _, tt := range tests {
		t.Run(tt.level, func(t *testing.T) {
			if got := IsValidTraceLevel(tt.level); got != tt.valid {
				t.Errorf("IsValidTraceLevel(%q) = %v, want %v", tt.level, got, tt.valid)
			}
		})
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/trace/... -v`
Expected: FAIL (package does not exist yet)

**Step 3: Implement trace types and recording**

In `sim/trace/record.go`:
```go
// Package trace provides decision-trace recording for cluster-level policy analysis.
// This package has no dependencies on sim/ or sim/cluster/ — it stores pure data types.
package trace

// AdmissionRecord captures a single admission policy decision.
type AdmissionRecord struct {
	RequestID string
	Clock     int64
	Admitted  bool
	Reason    string
}

// CandidateScore captures a counterfactual candidate instance with its score and state.
type CandidateScore struct {
	InstanceID    string
	Score         float64
	QueueDepth    int
	BatchSize     int
	KVUtilization float64
	FreeKVBlocks  int64
}

// RoutingRecord captures a single routing policy decision with optional counterfactual analysis.
type RoutingRecord struct {
	RequestID      string
	Clock          int64
	ChosenInstance string
	Reason         string
	Scores         map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates     []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret         float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}
```

In `sim/trace/trace.go`:
```go
package trace

// TraceLevel controls the verbosity of decision tracing.
type TraceLevel string

const (
	// TraceLevelNone disables tracing (zero overhead).
	TraceLevelNone TraceLevel = "none"
	// TraceLevelDecisions captures all admission and routing policy decisions.
	TraceLevelDecisions TraceLevel = "decisions"
)

// validTraceLevels maps accepted trace level strings.
var validTraceLevels = map[TraceLevel]bool{
	TraceLevelNone:      true,
	TraceLevelDecisions: true,
	"":                  true, // empty defaults to none
}

// IsValidTraceLevel returns true if the given level string is a recognized trace level.
func IsValidTraceLevel(level string) bool {
	return validTraceLevels[TraceLevel(level)]
}

// TraceConfig controls trace collection behavior.
type TraceConfig struct {
	Level           TraceLevel
	CounterfactualK int // number of counterfactual candidates per routing decision
}

// SimulationTrace collects decision records during a cluster simulation.
type SimulationTrace struct {
	Config     TraceConfig
	Admissions []AdmissionRecord
	Routings   []RoutingRecord
}

// NewSimulationTrace creates a SimulationTrace ready for recording.
func NewSimulationTrace(config TraceConfig) *SimulationTrace {
	return &SimulationTrace{
		Config:     config,
		Admissions: make([]AdmissionRecord, 0),
		Routings:   make([]RoutingRecord, 0),
	}
}

// RecordAdmission appends an admission decision record.
func (st *SimulationTrace) RecordAdmission(record AdmissionRecord) {
	st.Admissions = append(st.Admissions, record)
}

// RecordRouting appends a routing decision record.
func (st *SimulationTrace) RecordRouting(record RoutingRecord) {
	st.Routings = append(st.Routings, record)
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/trace/... -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/trace/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/trace/trace.go sim/trace/record.go sim/trace/trace_test.go
git commit -m "feat(trace): add SimulationTrace with admission and routing recording (BC-1, BC-2, BC-3)

- Add TraceLevel (none, decisions) with IsValidTraceLevel validator
- Add TraceConfig, SimulationTrace, NewSimulationTrace constructor
- Add AdmissionRecord, RoutingRecord, CandidateScore record types
- RecordAdmission and RecordRouting append to internal slices

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Counterfactual Computation

**Contracts Implemented:** BC-4, BC-5, BC-6, EC-2

**Files:**
- Create: `sim/cluster/counterfactual.go`
- Create: `sim/cluster/counterfactual_test.go`

**Step 1: Write failing tests for counterfactual computation**

Context: Test both scored (WeightedScoring) and non-scored (RoundRobin/LeastLoaded) paths, plus edge cases.

```go
// sim/cluster/counterfactual_test.go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestComputeCounterfactual_WithScores_TopKSortedByScore(t *testing.T) {
	// GIVEN 3 instances with explicit scores from WeightedScoring
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 5, BatchSize: 2, KVUtilization: 0.3},
		{ID: "i_1", QueueDepth: 1, BatchSize: 0, KVUtilization: 0.1},
		{ID: "i_2", QueueDepth: 3, BatchSize: 1, KVUtilization: 0.5},
	}
	scores := map[string]float64{"i_0": 0.4, "i_1": 0.9, "i_2": 0.6}

	// WHEN computing top-2 counterfactual, chosen is i_2 (score 0.6)
	candidates, regret := computeCounterfactual("i_2", scores, snapshots, 2)

	// THEN top-2 sorted by score desc: i_1 (0.9), i_2 (0.6)
	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates, got %d", len(candidates))
	}
	if candidates[0].InstanceID != "i_1" || candidates[0].Score != 0.9 {
		t.Errorf("first candidate: got %s/%.1f, want i_1/0.9", candidates[0].InstanceID, candidates[0].Score)
	}
	if candidates[1].InstanceID != "i_2" || candidates[1].Score != 0.6 {
		t.Errorf("second candidate: got %s/%.1f, want i_2/0.6", candidates[1].InstanceID, candidates[1].Score)
	}

	// THEN regret = best(0.9) - chosen(0.6) = 0.3
	if regret < 0.299 || regret > 0.301 {
		t.Errorf("expected regret ~0.3, got %.6f", regret)
	}
}

func TestComputeCounterfactual_ChosenIsBest_ZeroRegret(t *testing.T) {
	// GIVEN chosen instance has the highest score
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 1},
		{ID: "i_1", QueueDepth: 5},
	}
	scores := map[string]float64{"i_0": 0.8, "i_1": 0.2}

	// WHEN chosen is i_0 (best score)
	_, regret := computeCounterfactual("i_0", scores, snapshots, 2)

	// THEN regret = 0
	if regret != 0 {
		t.Errorf("expected 0 regret, got %.6f", regret)
	}
}

func TestComputeCounterfactual_NilScores_UsesLoadFallback(t *testing.T) {
	// GIVEN nil scores (RoundRobin/LeastLoaded) and different loads
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 10, BatchSize: 5}, // load=15, score=-15
		{ID: "i_1", QueueDepth: 1, BatchSize: 0},  // load=1,  score=-1
		{ID: "i_2", QueueDepth: 3, BatchSize: 2},  // load=5,  score=-5
	}

	// WHEN computing with nil scores, chosen is i_0 (worst load)
	candidates, regret := computeCounterfactual("i_0", nil, snapshots, 3)

	// THEN sorted by score desc (least loaded first): i_1(-1), i_2(-5), i_0(-15)
	if len(candidates) != 3 {
		t.Fatalf("expected 3 candidates, got %d", len(candidates))
	}
	if candidates[0].InstanceID != "i_1" {
		t.Errorf("first candidate: got %s, want i_1 (least loaded)", candidates[0].InstanceID)
	}
	if candidates[2].InstanceID != "i_0" {
		t.Errorf("last candidate: got %s, want i_0 (most loaded)", candidates[2].InstanceID)
	}

	// THEN regret = best(-1) - chosen(-15) = 14
	if regret < 13.99 || regret > 14.01 {
		t.Errorf("expected regret ~14, got %.6f", regret)
	}
}

func TestComputeCounterfactual_KZero_ReturnsNilAndZero(t *testing.T) {
	// GIVEN k=0
	snapshots := []sim.RoutingSnapshot{{ID: "i_0"}}

	// WHEN computing with k=0
	candidates, regret := computeCounterfactual("i_0", nil, snapshots, 0)

	// THEN nil candidates and zero regret
	if candidates != nil {
		t.Errorf("expected nil candidates, got %v", candidates)
	}
	if regret != 0 {
		t.Errorf("expected 0 regret, got %.6f", regret)
	}
}

func TestComputeCounterfactual_KExceedsInstances_ClampsToLen(t *testing.T) {
	// GIVEN 2 instances but k=10
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 1},
		{ID: "i_1", QueueDepth: 2},
	}
	scores := map[string]float64{"i_0": 0.8, "i_1": 0.3}

	// WHEN k > len(snapshots)
	candidates, _ := computeCounterfactual("i_1", scores, snapshots, 10)

	// THEN returns all instances (clamped)
	if len(candidates) != 2 {
		t.Fatalf("expected 2 candidates (clamped), got %d", len(candidates))
	}
}

func TestComputeCounterfactual_TiedScores_BreaksByInstanceID(t *testing.T) {
	// GIVEN equal scores, different IDs
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_b", QueueDepth: 1},
		{ID: "i_a", QueueDepth: 1},
	}
	scores := map[string]float64{"i_a": 0.5, "i_b": 0.5}

	// WHEN computing candidates
	candidates, _ := computeCounterfactual("i_b", scores, snapshots, 2)

	// THEN tie-broken by ID ascending: i_a before i_b
	if candidates[0].InstanceID != "i_a" {
		t.Errorf("expected i_a first (ID tie-break), got %s", candidates[0].InstanceID)
	}
}

func TestComputeCounterfactual_CandidatesIncludeSnapshotData(t *testing.T) {
	// GIVEN snapshots with KV utilization and free blocks data
	snapshots := []sim.RoutingSnapshot{
		{ID: "i_0", QueueDepth: 3, BatchSize: 2, KVUtilization: 0.7, FreeKVBlocks: 42},
	}
	scores := map[string]float64{"i_0": 0.5}

	// WHEN computing candidates
	candidates, _ := computeCounterfactual("i_0", scores, snapshots, 1)

	// THEN candidate includes full snapshot state
	c := candidates[0]
	if c.QueueDepth != 3 || c.BatchSize != 2 || c.KVUtilization != 0.7 || c.FreeKVBlocks != 42 {
		t.Errorf("candidate snapshot data mismatch: depth=%d batch=%d kv=%.1f freeKV=%d",
			c.QueueDepth, c.BatchSize, c.KVUtilization, c.FreeKVBlocks)
	}
}

// Verify the return type matches trace.CandidateScore (compile-time check)
var _ []trace.CandidateScore = func() []trace.CandidateScore {
	c, _ := computeCounterfactual("x", nil, nil, 0)
	return c
}()
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run TestComputeCounterfactual -v`
Expected: FAIL (function does not exist)

**Step 3: Implement counterfactual computation**

In `sim/cluster/counterfactual.go`:
```go
package cluster

import (
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

// copyScores returns a shallow copy of the scores map.
// Defensive copy: prevents trace data corruption if a future RoutingPolicy reuses its Scores map.
// Returns nil for nil input.
func copyScores(scores map[string]float64) map[string]float64 {
	if scores == nil {
		return nil
	}
	cp := make(map[string]float64, len(scores))
	for k, v := range scores {
		cp[k] = v
	}
	return cp
}

// computeCounterfactual builds a ranked list of candidate instances and computes
// regret (how much better the best alternative was compared to the chosen instance).
//
// When scores is non-nil (from WeightedScoring), candidates are ranked by policy scores.
// When scores is nil (RoundRobin, LeastLoaded), a synthetic load-based score is used:
// -(QueueDepth + BatchSize), so lower-load instances rank higher.
//
// Returns top-k candidates sorted by score descending and regret (≥ 0).
func computeCounterfactual(chosenID string, scores map[string]float64, snapshots []sim.RoutingSnapshot, k int) ([]trace.CandidateScore, float64) {
	if k <= 0 || len(snapshots) == 0 {
		return nil, 0
	}

	type scored struct {
		snap  sim.RoutingSnapshot
		score float64
	}

	all := make([]scored, len(snapshots))
	var chosenScore float64
	for i, snap := range snapshots {
		s := 0.0
		if scores != nil {
			s = scores[snap.ID]
		} else {
			// Load-based fallback: negative load so lower load ranks higher
			s = -float64(snap.QueueDepth + snap.BatchSize)
		}
		all[i] = scored{snap: snap, score: s}
		if snap.ID == chosenID {
			chosenScore = s
		}
	}

	// Sort by score descending; tie-break by instance ID ascending for determinism
	sort.Slice(all, func(i, j int) bool {
		if all[i].score != all[j].score {
			return all[i].score > all[j].score
		}
		return all[i].snap.ID < all[j].snap.ID
	})

	// Clamp k to available instances
	n := k
	if n > len(all) {
		n = len(all)
	}

	result := make([]trace.CandidateScore, n)
	for i := 0; i < n; i++ {
		result[i] = trace.CandidateScore{
			InstanceID:    all[i].snap.ID,
			Score:         all[i].score,
			QueueDepth:    all[i].snap.QueueDepth,
			BatchSize:     all[i].snap.BatchSize,
			KVUtilization: all[i].snap.KVUtilization,
			FreeKVBlocks:  all[i].snap.FreeKVBlocks,
		}
	}

	// Regret = best score - chosen score; 0 if chosen is best
	regret := all[0].score - chosenScore
	if regret < 0 {
		regret = 0
	}

	return result, regret
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run TestComputeCounterfactual -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/counterfactual.go sim/cluster/counterfactual_test.go
git commit -m "feat(cluster): add counterfactual candidate computation (BC-4, BC-5, BC-6)

- computeCounterfactual ranks instances by policy scores or load-based fallback
- Top-k candidates sorted by score desc with deterministic tie-breaking
- Regret = best alternative score - chosen score (clamped to 0)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Trace Summary

**Contracts Implemented:** BC-7, EC-3

**Files:**
- Create: `sim/trace/summary.go`
- Create: `sim/trace/summary_test.go`

**Step 1: Write failing tests for trace summary**

```go
// sim/trace/summary_test.go
package trace

import "testing"

func TestSummarize_EmptyTrace_ZeroValues(t *testing.T) {
	// GIVEN an empty trace
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})

	// WHEN summarized
	summary := Summarize(st)

	// THEN all counts are zero
	if summary.TotalDecisions != 0 {
		t.Errorf("expected 0 total decisions, got %d", summary.TotalDecisions)
	}
	if summary.AdmittedCount != 0 || summary.RejectedCount != 0 {
		t.Error("expected 0 admitted and rejected")
	}
	if summary.UniqueTargets != 0 {
		t.Errorf("expected 0 unique targets, got %d", summary.UniqueTargets)
	}
	if summary.MeanRegret != 0 || summary.MaxRegret != 0 {
		t.Error("expected 0 regret values")
	}
	if len(summary.TargetDistribution) != 0 {
		t.Error("expected empty target distribution")
	}
}

func TestSummarize_PopulatedTrace_CorrectCounts(t *testing.T) {
	// GIVEN a trace with mixed admission and routing records
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordAdmission(AdmissionRecord{RequestID: "r1", Admitted: true, Reason: "ok"})
	st.RecordAdmission(AdmissionRecord{RequestID: "r2", Admitted: false, Reason: "rejected"})
	st.RecordAdmission(AdmissionRecord{RequestID: "r3", Admitted: true, Reason: "ok"})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0", Regret: 0.1})
	st.RecordRouting(RoutingRecord{RequestID: "r3", ChosenInstance: "i_1", Regret: 0.3})

	// WHEN summarized
	summary := Summarize(st)

	// THEN counts match
	if summary.TotalDecisions != 3 {
		t.Errorf("expected 3 total decisions, got %d", summary.TotalDecisions)
	}
	if summary.AdmittedCount != 2 {
		t.Errorf("expected 2 admitted, got %d", summary.AdmittedCount)
	}
	if summary.RejectedCount != 1 {
		t.Errorf("expected 1 rejected, got %d", summary.RejectedCount)
	}
	if summary.UniqueTargets != 2 {
		t.Errorf("expected 2 unique targets, got %d", summary.UniqueTargets)
	}
}

func TestSummarize_RegretStatistics_CorrectMeanAndMax(t *testing.T) {
	// GIVEN routing records with known regrets
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0", Regret: 0.1})
	st.RecordRouting(RoutingRecord{RequestID: "r2", ChosenInstance: "i_0", Regret: 0.5})
	st.RecordRouting(RoutingRecord{RequestID: "r3", ChosenInstance: "i_1", Regret: 0.2})

	// WHEN summarized
	summary := Summarize(st)

	// THEN mean regret = (0.1 + 0.5 + 0.2) / 3 ≈ 0.2667
	expectedMean := (0.1 + 0.5 + 0.2) / 3.0
	if summary.MeanRegret < expectedMean-0.001 || summary.MeanRegret > expectedMean+0.001 {
		t.Errorf("expected mean regret ~%.4f, got %.4f", expectedMean, summary.MeanRegret)
	}

	// THEN max regret = 0.5
	if summary.MaxRegret != 0.5 {
		t.Errorf("expected max regret 0.5, got %.4f", summary.MaxRegret)
	}
}

func TestSummarize_TargetDistribution_CountsPerInstance(t *testing.T) {
	// GIVEN routing to same instance multiple times
	st := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
	st.RecordRouting(RoutingRecord{RequestID: "r1", ChosenInstance: "i_0"})
	st.RecordRouting(RoutingRecord{RequestID: "r2", ChosenInstance: "i_0"})
	st.RecordRouting(RoutingRecord{RequestID: "r3", ChosenInstance: "i_1"})

	// WHEN summarized
	summary := Summarize(st)

	// THEN target distribution reflects counts
	if summary.TargetDistribution["i_0"] != 2 {
		t.Errorf("expected i_0 count 2, got %d", summary.TargetDistribution["i_0"])
	}
	if summary.TargetDistribution["i_1"] != 1 {
		t.Errorf("expected i_1 count 1, got %d", summary.TargetDistribution["i_1"])
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/trace/... -run TestSummarize -v`
Expected: FAIL (Summarize function does not exist)

**Step 3: Implement TraceSummary**

In `sim/trace/summary.go`:
```go
package trace

// TraceSummary aggregates statistics from a SimulationTrace.
type TraceSummary struct {
	TotalDecisions     int
	AdmittedCount      int
	RejectedCount      int
	MeanRegret         float64
	MaxRegret          float64
	UniqueTargets      int
	TargetDistribution map[string]int // instance ID → count of requests routed
}

// Summarize computes aggregate statistics from a SimulationTrace.
// Safe for empty traces (returns zero-value fields).
func Summarize(st *SimulationTrace) *TraceSummary {
	summary := &TraceSummary{
		TargetDistribution: make(map[string]int),
	}

	summary.TotalDecisions = len(st.Admissions)
	for _, a := range st.Admissions {
		if a.Admitted {
			summary.AdmittedCount++
		} else {
			summary.RejectedCount++
		}
	}

	if len(st.Routings) > 0 {
		totalRegret := 0.0
		for _, r := range st.Routings {
			summary.TargetDistribution[r.ChosenInstance]++
			totalRegret += r.Regret
			if r.Regret > summary.MaxRegret {
				summary.MaxRegret = r.Regret
			}
		}
		summary.MeanRegret = totalRegret / float64(len(st.Routings))
	}

	summary.UniqueTargets = len(summary.TargetDistribution)

	return summary
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/trace/... -v`
Expected: PASS (all trace tests)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/trace/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/trace/summary.go sim/trace/summary_test.go
git commit -m "feat(trace): add TraceSummary with Summarize aggregation (BC-7, EC-3)

- TraceSummary: TotalDecisions, Admitted/Rejected counts, regret stats
- TargetDistribution maps instance IDs to routed request counts
- Handles empty traces safely (all zero values)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: EvaluationResult Wrapper

**Contracts Implemented:** BC-8

**Files:**
- Create: `sim/cluster/evaluation.go`
- Create: `sim/cluster/evaluation_test.go`

**Step 1: Write failing tests**

```go
// sim/cluster/evaluation_test.go
package cluster

import (
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestNewEvaluationResult_AllFields_PopulatesCorrectly(t *testing.T) {
	// GIVEN all available result data
	metrics := &RawMetrics{RequestsPerSec: 50.0}
	fitness := &FitnessResult{Score: 0.75, Components: map[string]float64{"throughput": 0.75}}
	tr := trace.NewSimulationTrace(trace.TraceConfig{Level: trace.TraceLevelDecisions})
	summary := &trace.TraceSummary{TotalDecisions: 10}

	// WHEN constructing EvaluationResult
	result := NewEvaluationResult(metrics, fitness, tr, summary, 1000000, 5*time.Second)

	// THEN all fields are populated
	if result.Metrics != metrics {
		t.Error("metrics mismatch")
	}
	if result.Fitness != fitness {
		t.Error("fitness mismatch")
	}
	if result.Trace != tr {
		t.Error("trace mismatch")
	}
	if result.Summary != summary {
		t.Error("summary mismatch")
	}
	if result.SimDuration != 1000000 {
		t.Errorf("expected SimDuration 1000000, got %d", result.SimDuration)
	}
	if result.WallTime != 5*time.Second {
		t.Errorf("expected WallTime 5s, got %v", result.WallTime)
	}
}

func TestNewEvaluationResult_NilOptionals_Accepted(t *testing.T) {
	// GIVEN only required data (metrics), optionals nil
	metrics := &RawMetrics{RequestsPerSec: 10.0}

	// WHEN constructing with nils
	result := NewEvaluationResult(metrics, nil, nil, nil, 500000, time.Second)

	// THEN result is valid with nil optionals
	if result.Metrics == nil {
		t.Error("metrics should not be nil")
	}
	if result.Fitness != nil {
		t.Error("fitness should be nil")
	}
	if result.Trace != nil {
		t.Error("trace should be nil")
	}
	if result.Summary != nil {
		t.Error("summary should be nil")
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run TestNewEvaluationResult -v`
Expected: FAIL (type and function do not exist)

**Step 3: Implement EvaluationResult**

In `sim/cluster/evaluation.go`:
```go
package cluster

import (
	"time"

	"github.com/inference-sim/inference-sim/sim/trace"
)

// EvaluationResult bundles all outputs from a cluster simulation run.
// Used by downstream consumers (GEPA, OpenEvolve adapters) for unified access
// to metrics, fitness scores, decision traces, and trace summaries.
type EvaluationResult struct {
	Metrics  *RawMetrics
	Fitness  *FitnessResult         // nil if no fitness weights provided
	Trace    *trace.SimulationTrace // nil if trace-level is "none"
	Summary  *trace.TraceSummary    // nil if --summarize-trace not set

	SimDuration int64         // simulation clock at end (ticks)
	WallTime    time.Duration // wall-clock duration of Run()
}

// NewEvaluationResult constructs an EvaluationResult.
// metrics is required; fitness, tr, and summary may be nil.
func NewEvaluationResult(metrics *RawMetrics, fitness *FitnessResult, tr *trace.SimulationTrace, summary *trace.TraceSummary, simDuration int64, wallTime time.Duration) *EvaluationResult {
	return &EvaluationResult{
		Metrics:     metrics,
		Fitness:     fitness,
		Trace:       tr,
		Summary:     summary,
		SimDuration: simDuration,
		WallTime:    wallTime,
	}
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run TestNewEvaluationResult -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/evaluation.go sim/cluster/evaluation_test.go
git commit -m "feat(cluster): add EvaluationResult wrapper for unified sim output (BC-8)

- Bundles RawMetrics, FitnessResult, SimulationTrace, TraceSummary
- Nil-safe for optional fields (fitness, trace, summary)
- API surface for downstream adapters (PR15)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: ClusterSimulator Trace Integration

**Contracts Implemented:** BC-1 (wiring), BC-2 (admission hook), BC-3 (routing hook), BC-4/BC-5/BC-6 (counterfactual in routing hook), NC-1, NC-2

**Files:**
- Modify: `sim/cluster/deployment.go:7-40` (add TraceLevel, CounterfactualK)
- Modify: `sim/cluster/cluster.go:15-73` (add trace field, constructor wiring, Trace() accessor)
- Modify: `sim/cluster/cluster_event.go:107-156` (admission + routing hooks)
- Create: `sim/cluster/cluster_trace_test.go`

**Step 1: Write failing integration test**

Context: Run a small cluster simulation with tracing enabled and verify trace records match events.

```go
// sim/cluster/cluster_trace_test.go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
)

func TestClusterSimulator_TraceLevelNone_NilTrace(t *testing.T) {
	// GIVEN trace level none (default)
	config := DeploymentConfig{
		NumInstances:   2,
		Horizon:        1000000,
		Seed:           42,
		TotalKVBlocks:  100,
		BlockSizeTokens: 16,
		MaxRunningReqs: 10,
		MaxScheduledTokens: 2048,
		TraceLevel:     "none",
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, MaxPrompts: 3,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	cs.Run()

	// THEN trace is nil (zero overhead)
	if cs.Trace() != nil {
		t.Error("expected nil trace for trace-level none")
	}
}

func TestClusterSimulator_TraceLevelDecisions_RecordsAllEvents(t *testing.T) {
	// GIVEN trace level decisions with 3 requests and 2 instances
	config := DeploymentConfig{
		NumInstances:       2,
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      100,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 2048,
		TraceLevel:         "decisions",
		CounterfactualK:    0,
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, MaxPrompts: 5,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	cs.Run()

	// THEN trace is non-nil with admission and routing records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace for trace-level decisions")
	}
	if len(tr.Admissions) != 5 {
		t.Errorf("expected 5 admission records (one per request), got %d", len(tr.Admissions))
	}
	if len(tr.Routings) != 5 {
		t.Errorf("expected 5 routing records (all admitted with always-admit), got %d", len(tr.Routings))
	}

	// All admission records should be admitted (default always-admit)
	for i, a := range tr.Admissions {
		if !a.Admitted {
			t.Errorf("admission[%d]: expected admitted=true", i)
		}
	}
}

func TestClusterSimulator_TraceLevelDecisions_WithCounterfactual(t *testing.T) {
	// GIVEN trace with counterfactual k=2 and weighted scoring
	config := DeploymentConfig{
		NumInstances:       2,
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      100,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 2048,
		RoutingPolicy:      "weighted",
		RoutingCacheWeight: 0.6,
		RoutingLoadWeight:  0.4,
		TraceLevel:         "decisions",
		CounterfactualK:    2,
	}
	workload := &sim.GuideLLMConfig{
		Rate: 1.0 / 1e6, MaxPrompts: 3,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	cs.Run()

	// THEN routing records have candidates
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	for i, r := range tr.Routings {
		if len(r.Candidates) == 0 {
			t.Errorf("routing[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("routing[%d]: expected at most 2 candidates, got %d", i, len(r.Candidates))
		}
	}
}

func TestClusterSimulator_TraceWithTokenBucket_RecordsRejections(t *testing.T) {
	// GIVEN token bucket admission that rejects some requests
	config := DeploymentConfig{
		NumInstances:       1,
		Horizon:            5000000,
		Seed:               42,
		TotalKVBlocks:      100,
		BlockSizeTokens:    16,
		MaxRunningReqs:     10,
		MaxScheduledTokens: 2048,
		AdmissionPolicy:    "token-bucket",
		TokenBucketCapacity: 2,   // very small: only 2 tokens
		TokenBucketRefillRate: 0.000001, // near-zero refill
		TraceLevel:          "decisions",
	}
	workload := &sim.GuideLLMConfig{
		Rate: 5.0 / 1e6, MaxPrompts: 10,
		PromptTokens: 10, OutputTokens: 5,
		PromptTokensStdDev: 0, OutputTokensStdDev: 0,
		PromptTokensMin: 10, PromptTokensMax: 10,
		OutputTokensMin: 5, OutputTokensMax: 5,
	}
	cs := NewClusterSimulator(config, workload, "")

	// WHEN run
	cs.Run()

	// THEN some admissions are rejected
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	rejections := 0
	for _, a := range tr.Admissions {
		if !a.Admitted {
			rejections++
		}
	}
	if rejections == 0 {
		t.Error("expected some rejections with tiny token bucket, got 0")
	}
	// Routing records should be fewer than admissions (rejected requests don't route)
	if len(tr.Routings) >= len(tr.Admissions) {
		t.Errorf("expected fewer routings (%d) than admissions (%d) with rejections",
			len(tr.Routings), len(tr.Admissions))
	}

	// Summarize to verify rejection counts match
	summary := trace.Summarize(tr)
	if summary.RejectedCount != rejections {
		t.Errorf("summary rejected count %d != counted rejections %d", summary.RejectedCount, rejections)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_Trace -v`
Expected: FAIL (TraceLevel/CounterfactualK not on DeploymentConfig, Trace() not on ClusterSimulator)

**Step 3: Implement ClusterSimulator trace integration**

Modify `sim/cluster/deployment.go` — add trace fields to DeploymentConfig:

In DeploymentConfig struct, after the Scheduler field, add:
```go
	// Decision trace configuration (PR13)
	TraceLevel      string // "none" (default), "decisions"
	CounterfactualK int    // number of counterfactual candidates, default 0
```

Modify `sim/cluster/cluster.go` — add trace field and wiring:

Add import for trace package and add trace field to ClusterSimulator struct:
```go
	trace *trace.SimulationTrace // nil when trace-level is "none" (BC-1: zero overhead)
```

In `NewClusterSimulator`, add trace initialization before the return statement:
```go
	// Initialize trace collector if tracing is enabled (BC-1: nil when none)
	var simTrace *trace.SimulationTrace
	if config.TraceLevel != "" && trace.TraceLevel(config.TraceLevel) != trace.TraceLevelNone {
		simTrace = trace.NewSimulationTrace(trace.TraceConfig{
			Level:           trace.TraceLevel(config.TraceLevel),
			CounterfactualK: config.CounterfactualK,
		})
	}
```

Add `trace: simTrace,` to the return struct literal.

Add Trace() accessor:
```go
// Trace returns the decision trace collected during simulation.
// Returns nil if trace-level was "none" (default).
func (c *ClusterSimulator) Trace() *trace.SimulationTrace {
	return c.trace
}
```

Modify `sim/cluster/cluster_event.go` — add recording hooks:

In `AdmissionDecisionEvent.Execute`, change `admitted, _` to capture reason, and add trace recording:
```go
func (e *AdmissionDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterState(cs)
	admitted, reason := cs.admissionPolicy.Admit(e.request, state)

	// Record admission decision if tracing is enabled (BC-2)
	if cs.trace != nil {
		cs.trace.RecordAdmission(trace.AdmissionRecord{
			RequestID: e.request.ID,
			Clock:     cs.clock,
			Admitted:  admitted,
			Reason:    reason,
		})
	}

	if !admitted {
		cs.rejectedRequests++
		return
	}
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &RoutingDecisionEvent{
			time:    e.time + cs.routingLatency,
			request: e.request,
		},
		seqID: cs.nextSeqID(),
	})
}
```

In `RoutingDecisionEvent.Execute`, add trace recording after the Route call:
```go
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterState(cs)
	decision := cs.routingPolicy.Route(e.request, state)

	// BC-9: Apply cluster-level priority hint if set by routing policy
	if decision.Priority != 0 {
		e.request.Priority = decision.Priority
	}

	// Record routing decision if tracing is enabled (BC-3, BC-4, BC-5, BC-6)
	// Placed after priority assignment to minimize diff; recording reads decision, not request.Priority
	if cs.trace != nil {
		record := trace.RoutingRecord{
			RequestID:      e.request.ID,
			Clock:          cs.clock,
			ChosenInstance: decision.TargetInstance,
			Reason:         decision.Reason,
			Scores:         copyScores(decision.Scores),
		}
		if cs.trace.Config.CounterfactualK > 0 {
			record.Candidates, record.Regret = computeCounterfactual(
				decision.TargetInstance, decision.Scores,
				state.Snapshots, cs.trace.Config.CounterfactualK,
			)
		}
		cs.trace.RecordRouting(record)
	}

	// Find target instance and inject request
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			inst.InjectRequestOnline(e.request, e.time)
			return
		}
	}

	// Should never reach here (policy contract ensures valid target)
	panic(fmt.Sprintf("RoutingDecisionEvent: invalid TargetInstance %q", decision.TargetInstance))
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -v`
Expected: PASS (all cluster tests including new trace tests)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/deployment.go sim/cluster/cluster.go sim/cluster/cluster_event.go sim/cluster/cluster_trace_test.go
git commit -m "feat(cluster): integrate trace recording into event pipeline (BC-1,BC-2,BC-3)

- Add TraceLevel/CounterfactualK to DeploymentConfig
- ClusterSimulator creates SimulationTrace when trace-level != none
- AdmissionDecisionEvent records admission decisions (captures reason)
- RoutingDecisionEvent records routing decisions with counterfactual

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: CLI Flags + Trace Summary Output

**Contracts Implemented:** BC-9, EC-1

**Files:**
- Modify: `cmd/root.go:19-83` (new flag vars), `cmd/root.go:299-346` (after Run output), `cmd/root.go:357-427` (flag registration)

**Step 1: Implement CLI integration**

Context: This task is primarily integration wiring. The behavioral contracts are tested by the cluster integration tests (Task 5) and end-to-end by the build/run verification below. No separate unit test file needed — the CLI is verified by `go build` and the existing test suite.

In `cmd/root.go`, add flag variables after the existing fitness weights variable (around line 79):

```go
	// Decision trace config (PR13)
	traceLevel      string // Trace verbosity level
	counterfactualK int    // Number of counterfactual candidates
	summarizeTrace  bool   // Print trace summary after simulation
```

In the `run` function (around line 255, after the scheduler validation), add trace level validation:

```go
		// Validate trace level
		if !trace.IsValidTraceLevel(traceLevel) {
			logrus.Fatalf("Unknown trace level %q. Valid: none, decisions", traceLevel)
		}
```

Add import for `"github.com/inference-sim/inference-sim/sim/trace"` in the imports block.

In the DeploymentConfig construction (around line 271), add trace fields:

```go
			TraceLevel:      traceLevel,
			CounterfactualK: counterfactualK,
```

After the existing anomaly counter output (around line 343), add trace summary and EvaluationResult construction:

```go
		// Build trace summary if requested (BC-9)
		var traceSummary *trace.TraceSummary
		if cs.Trace() != nil && summarizeTrace {
			traceSummary = trace.Summarize(cs.Trace())
		}

		// Construct unified EvaluationResult (BC-8)
		var fitnessResult *cluster.FitnessResult
		if fitnessWeights != "" {
			fitnessResult = fitness
		}
		evalResult := cluster.NewEvaluationResult(
			rawMetrics, fitnessResult, cs.Trace(), traceSummary,
			cs.Clock(), time.Since(startTime),
		)

		// Print trace summary if requested
		if evalResult.Summary != nil {
			fmt.Printf("\n=== Trace Summary ===\n")
			fmt.Printf("Total Decisions: %d\n", evalResult.Summary.TotalDecisions)
			fmt.Printf("  Admitted: %d\n", evalResult.Summary.AdmittedCount)
			fmt.Printf("  Rejected: %d\n", evalResult.Summary.RejectedCount)
			fmt.Printf("Unique Targets: %d\n", evalResult.Summary.UniqueTargets)
			if len(evalResult.Summary.TargetDistribution) > 0 {
				fmt.Printf("Target Distribution:\n")
				// Sort keys for deterministic output
				targetKeys := make([]string, 0, len(evalResult.Summary.TargetDistribution))
				for k := range evalResult.Summary.TargetDistribution {
					targetKeys = append(targetKeys, k)
				}
				sort.Strings(targetKeys)
				for _, k := range targetKeys {
					fmt.Printf("  %s: %d\n", k, evalResult.Summary.TargetDistribution[k])
				}
			}
			fmt.Printf("Mean Regret: %.6f\n", evalResult.Summary.MeanRegret)
			fmt.Printf("Max Regret: %.6f\n", evalResult.Summary.MaxRegret)
		}
```

Note: The existing `fitness` variable scoping must be widened. Currently `fitness := cluster.ComputeFitness(rawMetrics, weights)` is declared inside the `if fitnessWeights != ""` block (cmd/root.go ~line 323). Change to:
1. Before the `if fitnessWeights != ""` block, declare: `var fitness *cluster.FitnessResult`
2. Inside the block, change `fitness := cluster.ComputeFitness(...)` to `fitness = cluster.ComputeFitness(...)` (assignment, not declaration)
This allows `fitness` to be referenced later for the EvaluationResult construction.

In the `init()` function, add flag registrations (after the fitness-weights flag):

```go
	// Decision trace config (PR13)
	runCmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
	runCmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Number of counterfactual candidates per routing decision")
	runCmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")
```

**Step 2: Verify build succeeds**

Run: `go build ./...`
Expected: Success

**Step 3: Run full test suite**

Run: `go test ./...`
Expected: PASS (no golden dataset changes, no behavior changes)

**Step 4: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add cmd/root.go
git commit -m "feat(cmd): add --trace-level, --counterfactual-k, --summarize-trace flags (BC-9, EC-1)

- Validate trace level at CLI before simulation
- Pass TraceLevel/CounterfactualK through DeploymentConfig
- Print trace summary when --summarize-trace is set
- Construct EvaluationResult for unified output access

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Documentation + Stale Comment Cleanup

**Contracts Implemented:** (documentation-only)

**Files:**
- Modify: `CLAUDE.md` (file organization, CLI examples, current implementation focus)
- Modify: `sim/cluster/metrics.go:119` (remove stale "deferred to PR13" comment)

**Step 1: Update stale comments**

In `sim/cluster/metrics.go:119`, the comment says "Full decision-trace-based detection deferred to PR13." Now that PR13 is being implemented, update it:
```
// PR9 heuristic: counts pairs where an earlier-arriving request has
// worse E2E than a later-arriving request (with 2× threshold).
```
(Remove the "deferred to PR13" clause.)

**Step 2: Update CLAUDE.md**

In the **Build and Run Commands** section, add trace example:
```bash
# Run with decision tracing and counterfactual analysis
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --trace-level decisions --counterfactual-k 5 --summarize-trace
```

In the **Core Simulation Engine** section, add under `sim/cluster/`:
```
- **cluster/counterfactual.go**: computeCounterfactual() for top-k candidate ranking and regret
- **cluster/evaluation.go**: EvaluationResult wrapper (RawMetrics + FitnessResult + trace + summary)
```

In the **File Organization** section, update `sim/trace/` from "planned" to actual:
```
├── sim/trace/               # Decision trace recording
│   ├── trace.go             # TraceLevel, TraceConfig, SimulationTrace
│   ├── record.go            # AdmissionRecord, RoutingRecord, CandidateScore
│   └── summary.go           # TraceSummary, Summarize()
```

In the **Current Implementation Focus** section, add PR13 to **Completed** list:
```
PR13 (DecisionTrace with RoutingRecord and counterfactual analysis, EvaluationResult wrapper, `--trace-level decisions --counterfactual-k --summarize-trace` CLI flags)
```

In the **CLI flags** comment in `cmd/root.go` header area (around line 76), add:
```go
	// Decision trace config (PR13)
```

**Step 3: Verify everything passes**

Run: `go test ./... && golangci-lint run ./...`
Expected: All pass

**Step 4: Commit**

```bash
git add CLAUDE.md sim/cluster/metrics.go
git commit -m "docs: update CLAUDE.md for PR13 traces, remove stale PR13 deferral comment

- Add trace CLI example to Build and Run Commands
- Add sim/trace/ package to file organization
- Add PR13 to completed PRs list
- Remove 'deferred to PR13' comment in metrics.go (now implemented)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 5 | Integration | TestClusterSimulator_TraceLevelNone_NilTrace |
| BC-2 | Task 1, Task 5 | Unit + Integration | TestSimulationTrace_RecordAdmission_AppendsRecord, TestClusterSimulator_TraceLevelDecisions_RecordsAllEvents |
| BC-3 | Task 1, Task 5 | Unit + Integration | TestSimulationTrace_RecordRouting_AppendsRecord, TestClusterSimulator_TraceLevelDecisions_RecordsAllEvents |
| BC-4 | Task 2, Task 5 | Unit + Integration | TestComputeCounterfactual_WithScores_TopKSortedByScore, TestClusterSimulator_TraceLevelDecisions_WithCounterfactual |
| BC-5 | Task 2 | Unit | TestComputeCounterfactual_NilScores_UsesLoadFallback |
| BC-6 | Task 2 | Unit | TestComputeCounterfactual_ChosenIsBest_ZeroRegret, TestComputeCounterfactual_WithScores_TopKSortedByScore |
| BC-7 | Task 3 | Unit | TestSummarize_PopulatedTrace_CorrectCounts, TestSummarize_RegretStatistics_CorrectMeanAndMax |
| BC-8 | Task 4 | Unit | TestNewEvaluationResult_AllFields_PopulatesCorrectly, TestNewEvaluationResult_NilOptionals_Accepted |
| BC-9 | Task 6 | Build | go build verification |
| NC-1 | — | Structural | No interface files modified (verified by diff) |
| NC-2 | Task 5 | Integration | TestClusterSimulator_TraceLevelDecisions_RecordsAllEvents (same metrics) |
| NC-3 | Task 6 | Regression | go test ./... (all existing golden tests pass) |
| EC-1 | Task 1 | Unit | TestIsValidTraceLevel_ValidLevels |
| EC-2 | Task 2 | Unit | TestComputeCounterfactual_KZero_ReturnsNilAndZero |
| EC-3 | Task 3 | Unit | TestSummarize_EmptyTrace_ZeroValues |

**Golden dataset update:** Not needed. Tracing is observation-only; default trace-level is "none" so existing tests are unaffected.

**Shared test infrastructure:** No new shared helpers needed. Tests use standard Go testing.

---

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Trace memory bloat with many requests | Medium | Low | Configurable via --trace-level; default "none" disables allocation | Task 1 (BC-1) |
| Counterfactual float comparison instability | Low | Low | Deterministic tie-breaking by instance ID; tolerance-based assertions | Task 2 |
| Event pipeline change breaks existing tests | Low | High | Minimal change: only adds nil-check + recording after existing logic | Task 5 |
| Import cycle sim/trace/ ↔ sim/cluster/ | Low | High | sim/trace/ has zero dependencies on sim/ or sim/cluster/ (pure data) | Task 1 design |
| EvaluationResult not exercised in CLI | Low | Medium | Used for trace summary printing; also unit-tested | Task 4, Task 6 |
| Map reference sharing corrupts trace data | Low | High | `copyScores()` defensive copy of `RoutingDecision.Scores` map before storing in trace record | Task 5 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — trace types are simple data structs
- [x] No feature creep beyond PR scope — `detailed` trace level deferred
- [x] No unexercised flags — `--trace-level decisions --counterfactual-k 5 --summarize-trace` exercises all code
- [x] No partial implementations — all types have recording + summary + CLI integration
- [x] No breaking changes — frozen interfaces untouched, default behavior unchanged
- [x] No hidden global state — trace owned by ClusterSimulator instance
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: no new helpers needed
- [x] CLAUDE.md updated: file organization, CLI examples, completed PRs
- [x] No stale references: "deferred to PR13" comment removed
- [x] Deviation log reviewed: `detailed` deferred, record LOC simplified, EvaluationResult metadata fields
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered: 1 → 2 → 3 → 4 → 5 → 6 → 7
- [x] All contracts mapped to tasks
- [x] Golden dataset: no regeneration needed

---

## Appendix K: File-Level Implementation Details

### File: `sim/trace/trace.go`

**Purpose:** Core trace types and recording methods. Zero dependencies on sim/ or sim/cluster/.

**Complete implementation:** See Task 1, Step 3.

**Key notes:**
- `validTraceLevels` is unexported map (follows bundle.go pattern from PR8)
- `IsValidTraceLevel` is the public accessor
- Empty string `""` maps to valid (treated as "none" by CLI default)
- Slices pre-allocated with `make(..., 0)` to avoid nil slice in JSON serialization

### File: `sim/trace/record.go`

**Purpose:** Pure data types for admission and routing records.

**Complete implementation:** See Task 1, Step 3.

**Key notes:**
- No methods on record types — they're pure data containers
- `CandidateScore` mirrors `RoutingSnapshot` fields but avoids importing `sim/`
- `RoutingRecord.Scores` may be nil (depends on routing policy)
- `RoutingRecord.Candidates` is nil when counterfactual-k=0

### File: `sim/trace/summary.go`

**Purpose:** Aggregate statistics from a SimulationTrace.

**Complete implementation:** See Task 3, Step 3.

**Key notes:**
- `Summarize` is a package-level function (not a method on SimulationTrace)
- Handles empty traces gracefully (zero values, empty map)
- MeanRegret computed from routing records only (not admission records)

### File: `sim/cluster/counterfactual.go`

**Purpose:** Compute top-k candidate scores and regret from routing decision context.

**Complete implementation:** See Task 2, Step 3.

**Key notes:**
- Unexported function — only called from RoutingDecisionEvent.Execute
- Two scoring modes: explicit policy scores (non-nil map) or load-based fallback (nil map)
- Load-based score: `-(QueueDepth + BatchSize)` — negative so lower load ranks higher
- Deterministic tie-breaking by instance ID (ascending)
- k clamped to len(snapshots) when k exceeds available instances

### File: `sim/cluster/evaluation.go`

**Purpose:** Unified result wrapper for downstream consumers.

**Complete implementation:** See Task 4, Step 3.

**Key notes:**
- Fitness, Trace, Summary are nil-safe optional fields
- SimDuration comes from `cs.Clock()` (cluster clock at simulation end)
- WallTime comes from `time.Since(startTime)` in CLI
- PolicyID/WorkloadID deferred to PR15 (adapter concern)

### File: `sim/cluster/deployment.go` (modified)

**Purpose:** Add TraceLevel and CounterfactualK to DeploymentConfig.

**Changes:** Two new string/int fields added at end of struct. No ToSimConfig() changes needed — trace config is used by ClusterSimulator directly, not passed to per-instance SimConfig.

### File: `sim/cluster/cluster.go` (modified)

**Purpose:** Wire trace into ClusterSimulator lifecycle.

**Changes:**
- New `trace` field on ClusterSimulator struct
- Conditional allocation in NewClusterSimulator (nil when level=none)
- New `Trace()` accessor method
- Import added for `sim/trace` package

### File: `sim/cluster/cluster_event.go` (modified)

**Purpose:** Record decisions in event pipeline.

**Changes:**
- `AdmissionDecisionEvent.Execute`: capture reason (was `_`), add nil-checked trace recording
- `RoutingDecisionEvent.Execute`: add nil-checked trace recording with counterfactual computation
- Import added for `sim/trace` package

**Behavioral note:** Admission recording happens AFTER `Admit()` but BEFORE the reject counter increment (observation-only). Routing recording happens AFTER both `Route()` and the priority assignment but BEFORE instance injection — recording reads `decision` fields, not mutated request state. This ensures NC-2 (no behavior change) and minimizes diff from current code.

### File: `cmd/root.go` (modified)

**Purpose:** CLI flag registration, validation, and trace summary output.

**Changes:**
- 3 new flag variables: traceLevel, counterfactualK, summarizeTrace
- Trace level validation after scheduler validation
- TraceLevel/CounterfactualK added to DeploymentConfig literal
- After anomaly output: TraceSummary computation + EvaluationResult construction + summary printing
- 3 new flag registrations in init()
- `fitness` variable scope widened for EvaluationResult construction
