# PR7: PriorityPolicy + InstanceScheduler Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable request prioritization and per-instance batch scheduling policies with pluggable templates, while preserving exact backward compatibility with default settings.

**Architecture:** Add `PriorityPolicy` (assigns priority scores to requests) and `InstanceScheduler` (reorders the wait queue before batch formation) as pluggable interfaces in the `sim/` package. The scheduler sorts the WaitQueue in-place before each `makeRunningBatch()` call, leveraging the existing batch formation logic. Default policies (`constant` + `fcfs`) are identity operations preserving existing behavior.

**Macro Plan Reference:** Phase 2, PR 7 in `docs/plans/2026-02-11-macro-implementation-plan-v2.md`

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation (Human Review)

### A) Executive Summary

- **Building block:** PriorityPolicy and InstanceScheduler — two new policy interfaces in `sim/` following the AdmissionPolicy pattern from PR5
- **Adjacent blocks:** `Simulator.Step()` (caller), `WaitQueue` (reordered), `Request` (Priority field added), `DeploymentConfig`/`SimConfig` (config flow), `cmd/root.go` (CLI flags)
- **DEVIATION flags:** 3 deviations from macro plan (see Section D): file location (`sim/` not `sim/policy/`), simplified interface (`OrderQueue` not `MakeBatch`), modified files differ

### B) Behavioral Contracts

**Positive Contracts (what MUST happen):**

**BC-1: Default Backward Compatibility**
- GIVEN default config (`priority-policy=""` or `"constant"`, `scheduler=""` or `"fcfs"`)
- WHEN simulation runs with golden dataset parameters
- THEN output metrics MUST exactly match golden dataset (bit-for-bit identical)
- MECHANISM: `ConstantPriority` assigns 0.0 to all requests; `FCFSScheduler.OrderQueue` is a no-op

**BC-2: Priority Assignment Per Step**
- GIVEN any `PriorityPolicy` and requests in the WaitQueue
- WHEN `Simulator.Step()` is called
- THEN each queued request MUST have its `Priority` field set to the policy's computed value before batch formation begins
- MECHANISM: Loop over `WaitQ.queue` calling `priorityPolicy.Compute()` before `makeRunningBatch()`

**BC-3: Priority Ordering (PriorityFCFS)**
- GIVEN `PriorityFCFSScheduler` and requests with different priorities
- WHEN `OrderQueue` is called
- THEN requests MUST be sorted by Priority descending, then ArrivalTime ascending, then ID ascending (lexicographic)
- MECHANISM: `sort.SliceStable` with three-level composite comparator

**BC-4: SJF Ordering**
- GIVEN `SJFScheduler` and requests with different input token counts
- WHEN `OrderQueue` is called
- THEN requests MUST be sorted by `len(InputTokens)` ascending, then ArrivalTime ascending, then ID ascending
- MECHANISM: `sort.SliceStable` with three-level composite comparator

**BC-5: Priority-Aware Batch Formation**
- GIVEN `PriorityFCFSScheduler`, two queued requests A (priority=10) and B (priority=1), both fit in batch
- WHEN `Step()` runs (which calls `OrderQueue` then `makeRunningBatch`)
- THEN A MUST be scheduled before B in the same or earlier step
- MECHANISM: `OrderQueue` places A before B; `makeRunningBatch` dequeues from front
- NOTE: If A cannot be scheduled due to KV cache pressure, `makeRunningBatch` breaks — B is also not scheduled that step (head-of-line blocking, see BC-10)

**BC-10: Head-of-Line Blocking Under Memory Pressure**
- GIVEN any non-FCFS scheduler and the front-of-queue request cannot allocate KV blocks
- WHEN `makeRunningBatch` attempts to dequeue from the sorted queue
- THEN no additional requests from the queue MUST be scheduled in that step, even if they would fit
- MECHANISM: `makeRunningBatch` (`sim/simulator.go:452-456`) uses a hard `break` on KV allocation failure — this is intentional vLLM-compatible behavior that preserves strict ordering guarantees. The scheduler controls ordering, but `makeRunningBatch` controls admission within a step.

**BC-6: ConstantPriority Returns Fixed Score**
- GIVEN `ConstantPriority{Score: X}` for any float64 X
- WHEN `Compute` is called with any request and any clock value
- THEN MUST return X
- MECHANISM: Returns `c.Score` directly

**BC-7: SLOBasedPriority Age Sensitivity**
- GIVEN `SLOBasedPriority{BaseScore: B, AgeWeight: W}` with W > 0
- WHEN two requests with ArrivalTime t1 < t2 (t1 is older) at clock C
- THEN `Compute(req1, C) > Compute(req2, C)` (older request gets higher priority)
- MECHANISM: `priority = BaseScore + AgeWeight * float64(clock - req.ArrivalTime)`

**BC-8: Deterministic Tie-Breaking**
- GIVEN requests with equal priority scores and equal ArrivalTime
- WHEN `PriorityFCFSScheduler.OrderQueue` is called
- THEN ties MUST be broken by `Request.ID` (lexicographic ascending)
- MECHANISM: Third comparator level: `reqs[i].ID < reqs[j].ID`

**BC-9: CLI Flag Integration**
- GIVEN `--priority-policy` and `--scheduler` CLI flags
- WHEN simulation runs
- THEN the specified policies MUST be used for scheduling
- MECHANISM: Flags → `DeploymentConfig` → `ToSimConfig()` → `SimConfig` → `NewSimulator()`

**Negative Contracts (what MUST NOT happen):**

**NC-1: No Metric Regression with Defaults**
- GIVEN default configuration (no `--priority-policy` or `--scheduler` flags)
- WHEN golden dataset tests run
- THEN all metrics MUST match exactly — zero regression

**NC-2: No Queue Corruption**
- GIVEN any scheduler reordering
- WHEN `OrderQueue` completes
- THEN WaitQueue MUST contain exactly the same set of requests (no additions, no removals, no duplicates)
- MECHANISM: `sort.SliceStable` operates on existing slice elements in-place

**NC-3: No Side Effects from Priority Computation**
- GIVEN `PriorityPolicy.Compute()` is called
- THEN only the return value is used; the request MUST NOT be modified inside `Compute`
- MECHANISM: `Compute` returns `float64`; caller sets `req.Priority`

**Error Handling Contracts:**

**EH-1: Unknown Policy Name Panics**
- GIVEN an unrecognized priority policy name (e.g., `"unknown"`)
- WHEN `NewPriorityPolicy("unknown")` is called
- THEN MUST panic with message containing the unrecognized name
- MECHANISM: `default` case in switch with `panic(fmt.Sprintf(...))`

**EH-2: Unknown Scheduler Name Panics**
- GIVEN an unrecognized scheduler name (e.g., `"unknown"`)
- WHEN `NewScheduler("unknown")` is called
- THEN MUST panic with message containing the unrecognized name
- MECHANISM: `default` case in switch with `panic(fmt.Sprintf(...))`

**EH-3: Empty Name Uses Default**
- GIVEN empty string `""` for policy or scheduler name
- WHEN factory function is called
- THEN MUST return the default (`ConstantPriority{Score: 0}` / `FCFSScheduler{}`)
- MECHANISM: `case ""` in switch falls through to default template

### C) Component Interaction

```
                   cmd/root.go
                       │
                       ▼
              DeploymentConfig
              (PriorityPolicy, Scheduler strings)
                       │
                       ▼ ToSimConfig()
                   SimConfig
              (PriorityPolicy, Scheduler strings)
                       │
                       ▼ NewSimulator()
         ┌─────────Simulator─────────┐
         │  priorityPolicy PriorityPolicy  │
         │  scheduler InstanceScheduler    │
         │                                 │
         │  Step(now):                     │
         │    1. Compute priorities        │
         │    2. OrderQueue (sort WaitQ)   │
         │    3. makeRunningBatch (existing)│
         └─────────────────────────────────┘
```

**API Contracts:**
- `PriorityPolicy.Compute(req *Request, clock int64) float64` — pure function, no mutation
- `InstanceScheduler.OrderQueue(requests []*Request, clock int64)` — sorts slice in-place
- `NewPriorityPolicy(name string) PriorityPolicy` — factory, panics on unknown
- `NewScheduler(name string) InstanceScheduler` — factory, panics on unknown

**State Changes:**
- `Request.Priority float64` — new field, set by Simulator each step, read by schedulers
- `Simulator.priorityPolicy` / `Simulator.scheduler` — set once in constructor, immutable after

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| Files in `sim/policy/priority.go`, `sim/policy/scheduler.go` | Files in `sim/priority.go`, `sim/scheduler.go` | CORRECTION: `sim/policy/` would need to import `sim.Request`, `sim.WaitQueue`, but `Simulator` (in `sim/`) calls the scheduler — creating a circular dependency. Follows PR5 precedent where `AdmissionPolicy` was moved from `sim/policy/` to `sim/admission.go`. |
| `InstanceScheduler` has `MakeBatch(ctx SchedulerContext) BatchDecision` + `OnRequestArrival` | `InstanceScheduler` has `OrderQueue(requests []*Request, clock int64)` | SIMPLIFICATION: Full `MakeBatch` would require reimplementing all of `makeRunningBatch()` (KV allocation, token budgets, preemption — 120 lines). `OrderQueue` achieves the same scheduling effect with ~5 lines per template by reordering the queue before existing batch formation. `SchedulerContext`, `BatchDecision` deferred to when `MakeBatch` is truly needed. |
| Modifies `sim/cluster/instance.go` (~30 LOC) | Modifies `sim/simulator.go` (~25 LOC); no changes to `instance.go` | CORRECTION: The scheduler integrates at the `Simulator.Step()` level (where `makeRunningBatch` is called), not at the `InstanceSimulator` wrapper level. Config flows automatically via `DeploymentConfig.ToSimConfig()`. |

### E) Review Guide

1. **THE TRICKY PART:** The integration point in `Simulator.Step()` — priority assignment + queue ordering MUST happen after `runningBatchFeatures` reset but before `makeRunningBatch(now)`. Incorrect ordering would cause priorities to be stale or the queue to not be sorted when batch formation reads it.

2. **WHAT TO SCRUTINIZE:** BC-1 (backward compatibility) is the highest-risk contract. Verify that `ConstantPriority{Score: 0}` + `FCFSScheduler` (no-op) produces zero behavioral change. The golden dataset test is the definitive check. Also scrutinize `sort.SliceStable` usage — must be stable sort to preserve FIFO within equal-priority groups.

3. **WHAT'S SAFE TO SKIM:** Factory functions (`NewPriorityPolicy`, `NewScheduler`) are mechanical switch statements following the `NewAdmissionPolicy` pattern. CLI flag wiring is boilerplate. `DeploymentConfig.ToSimConfig()` field mapping is mechanical.

4. **KNOWN DEBT:** `SLOBasedPriority` currently uses age-based urgency only (no SLO class input). Full SLO class integration requires `RouterState`/`TenantState` from PR8. The name is kept for macro plan consistency.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/priority.go` — PriorityPolicy interface, ConstantPriority, SLOBasedPriority, factory (~70 LOC)
- `sim/scheduler.go` — InstanceScheduler interface, FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, factory (~90 LOC)
- `sim/priority_test.go` — unit tests for priority policies (~80 LOC)
- `sim/scheduler_test.go` — unit tests for schedulers + integration tests (~150 LOC)

**Files to modify:**
- `sim/request.go:18` — add `Priority float64` field (~1 LOC)
- `sim/simulator.go:73-92` — add fields to `SimConfig` (~2 LOC)
- `sim/simulator.go:95-136` — add fields to `Simulator` struct (~2 LOC)
- `sim/simulator.go:143-198` — wire in `NewSimulator()` constructor (~2 LOC)
- `sim/simulator.go:494-505` — add priority+scheduling to `Step()` (~6 LOC)
- `sim/cluster/deployment.go:7-31` — add config fields (~2 LOC)
- `sim/cluster/deployment.go:37-55` — add to `ToSimConfig()` mapping (~2 LOC)
- `cmd/root.go:57-62` — add flag variables (~2 LOC)
- `cmd/root.go:202-224` — add to `DeploymentConfig` construction (~2 LOC)
- `cmd/root.go:292-297` — add flag registrations (~2 LOC)

**Key decisions:**
- Interfaces in `sim/` package (not `sim/policy/`) to avoid circular deps
- `OrderQueue` sorts WaitQueue in-place before existing `makeRunningBatch()` — minimal change
- Default policies are identity operations (zero behavioral change)

**Confirmation:** No dead code. All templates exercisable via `--priority-policy` and `--scheduler` CLI flags. All code paths tested.

### G) Task Breakdown

---

#### Batch 1: Core Types (Tasks 1-2)

---

### Task 1: PriorityPolicy Interface + Templates + Priority Field on Request

**Contracts Implemented:** BC-6, BC-7, EH-1, EH-3, NC-3

**Files:**
- Modify: `sim/request.go:18-34`
- Create: `sim/priority.go`
- Test: `sim/priority_test.go`

**Step 1: Write failing tests for PriorityPolicy**

Context: We test that ConstantPriority always returns the configured score, SLOBasedPriority increases with request age, and the factory panics on unknown names.

```go
package sim

import (
	"testing"
)

func TestConstantPriority_ReturnsFixedScore(t *testing.T) {
	// BC-6: ConstantPriority returns configured score regardless of request/clock
	policy := &ConstantPriority{Score: 5.0}
	req := &Request{ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50)}

	got := policy.Compute(req, 1000)
	if got != 5.0 {
		t.Errorf("ConstantPriority.Compute: got %f, want 5.0", got)
	}

	// Different request and clock — same result
	req2 := &Request{ID: "r2", ArrivalTime: 500, InputTokens: make([]int, 200)}
	got2 := policy.Compute(req2, 9999)
	if got2 != 5.0 {
		t.Errorf("ConstantPriority.Compute (different req): got %f, want 5.0", got2)
	}
}

func TestConstantPriority_DefaultZero(t *testing.T) {
	// BC-6: Zero-value ConstantPriority returns 0.0
	policy := &ConstantPriority{}
	req := &Request{ID: "r1", ArrivalTime: 0}
	got := policy.Compute(req, 0)
	if got != 0.0 {
		t.Errorf("ConstantPriority (zero): got %f, want 0.0", got)
	}
}

func TestSLOBasedPriority_OlderRequestGetsHigherPriority(t *testing.T) {
	// BC-7: With AgeWeight > 0, older requests get higher priority
	policy := &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	clock := int64(2000000) // 2 seconds in ticks

	older := &Request{ID: "old", ArrivalTime: 0}          // age = 2s
	newer := &Request{ID: "new", ArrivalTime: 1000000}     // age = 1s

	pOlder := policy.Compute(older, clock)
	pNewer := policy.Compute(newer, clock)

	if pOlder <= pNewer {
		t.Errorf("SLOBasedPriority: older=%f should be > newer=%f", pOlder, pNewer)
	}
}

func TestSLOBasedPriority_FormulaCorrectness(t *testing.T) {
	// BC-7: priority = BaseScore + AgeWeight * float64(clock - ArrivalTime)
	policy := &SLOBasedPriority{BaseScore: 1.0, AgeWeight: 0.5}
	req := &Request{ID: "r1", ArrivalTime: 100}
	clock := int64(300)

	got := policy.Compute(req, clock)
	want := 1.0 + 0.5*float64(300-100) // 1.0 + 100.0 = 101.0
	if got != want {
		t.Errorf("SLOBasedPriority formula: got %f, want %f", got, want)
	}
}

func TestNewPriorityPolicy_ValidNames_ReturnsCorrectType(t *testing.T) {
	// EH-3: empty string returns default (ConstantPriority)
	p1 := NewPriorityPolicy("")
	if _, ok := p1.(*ConstantPriority); !ok {
		t.Errorf("NewPriorityPolicy(\"\"): expected *ConstantPriority, got %T", p1)
	}

	p2 := NewPriorityPolicy("constant")
	if _, ok := p2.(*ConstantPriority); !ok {
		t.Errorf("NewPriorityPolicy(\"constant\"): expected *ConstantPriority, got %T", p2)
	}

	p3 := NewPriorityPolicy("slo-based")
	if _, ok := p3.(*SLOBasedPriority); !ok {
		t.Errorf("NewPriorityPolicy(\"slo-based\"): expected *SLOBasedPriority, got %T", p3)
	}
}

func TestNewPriorityPolicy_UnknownName_Panics(t *testing.T) {
	// EH-1: unknown name panics
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("NewPriorityPolicy(\"unknown\"): expected panic, got nil")
		}
	}()
	NewPriorityPolicy("unknown")
}

func TestPriorityPolicy_Compute_NoSideEffects(t *testing.T) {
	// NC-3: Compute must not modify the request
	policies := []struct {
		name   string
		policy PriorityPolicy
	}{
		{"constant", &ConstantPriority{Score: 5.0}},
		{"slo-based", &SLOBasedPriority{BaseScore: 1.0, AgeWeight: 1e-6}},
	}
	for _, tc := range policies {
		t.Run(tc.name, func(t *testing.T) {
			req := &Request{
				ID: "r1", ArrivalTime: 100, InputTokens: make([]int, 50),
				Priority: 0.0, State: "queued",
			}
			tc.policy.Compute(req, 1000)
			if req.Priority != 0.0 {
				t.Errorf("Compute modified req.Priority: got %f, want 0.0", req.Priority)
			}
			if req.State != "queued" {
				t.Errorf("Compute modified req.State: got %q, want %q", req.State, "queued")
			}
			if req.ID != "r1" {
				t.Errorf("Compute modified req.ID: got %q, want %q", req.ID, "r1")
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestConstantPriority -v 2>&1 | head -20`
Expected: FAIL (compilation error — types not defined yet)

**Step 3: Implement PriorityPolicy + add Priority field to Request**

Context: Add `Priority float64` to Request, then create `sim/priority.go` with the interface and templates.

In `sim/request.go`, add the Priority field to the Request struct (after line 33, before the closing brace):

```go
Priority float64 // Priority score assigned by PriorityPolicy (higher = more urgent)
```

Create `sim/priority.go`:

```go
package sim

import "fmt"

// PriorityPolicy computes a priority score for a request.
// Higher scores indicate higher priority (scheduled first by priority-aware schedulers).
// Implementations MUST NOT modify the request — only the return value is used.
type PriorityPolicy interface {
	Compute(req *Request, clock int64) float64
}

// ConstantPriority assigns a fixed priority score to all requests.
type ConstantPriority struct {
	Score float64
}

func (c *ConstantPriority) Compute(_ *Request, _ int64) float64 {
	return c.Score
}

// SLOBasedPriority computes priority based on request age (time waiting).
// Older requests receive higher priority to reduce SLO violation risk.
// Formula: BaseScore + AgeWeight * float64(clock - req.ArrivalTime)
//
// Full SLO class integration (using TenantState) is planned for PR8.
type SLOBasedPriority struct {
	BaseScore float64
	AgeWeight float64
}

func (s *SLOBasedPriority) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	return s.BaseScore + s.AgeWeight*age
}

// NewPriorityPolicy creates a PriorityPolicy by name.
// Valid names: "constant" (default), "slo-based".
// Empty string defaults to ConstantPriority (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewPriorityPolicy(name string) PriorityPolicy {
	switch name {
	case "", "constant":
		return &ConstantPriority{Score: 0.0}
	case "slo-based":
		return &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	default:
		panic(fmt.Sprintf("unknown priority policy %q", name))
	}
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestConstantPriority|TestSLOBasedPriority|TestNewPriorityPolicy|TestPriorityPolicy_Compute" -v`
Expected: PASS (all 8 tests)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/request.go sim/priority.go sim/priority_test.go
git commit -m "$(cat <<'EOF'
feat(sim): add PriorityPolicy interface with ConstantPriority and SLOBasedPriority (BC-2, BC-6, BC-7)

- Add Priority float64 field to Request struct
- Add PriorityPolicy interface with Compute(req, clock) float64
- Implement ConstantPriority (fixed score) and SLOBasedPriority (age-based)
- Add NewPriorityPolicy factory with panic on unknown names
- Unit tests for all templates and factory

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: InstanceScheduler Interface + Templates

**Contracts Implemented:** BC-3, BC-4, BC-8, EH-2, EH-3, NC-2

**Files:**
- Create: `sim/scheduler.go`
- Test: `sim/scheduler_test.go`

**Step 1: Write failing tests for InstanceScheduler**

Context: We test that FCFSScheduler is a no-op, PriorityFCFSScheduler sorts by priority/arrival/ID, SJFScheduler sorts by input length/arrival/ID, and the factory works correctly.

```go
package sim

import (
	"testing"
)

// helper to extract IDs from request slice for assertion
func requestIDs(reqs []*Request) []string {
	ids := make([]string, len(reqs))
	for i, r := range reqs {
		ids[i] = r.ID
	}
	return ids
}

func sliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestFCFSScheduler_PreservesOrder(t *testing.T) {
	// FCFS is a no-op: order unchanged
	sched := &FCFSScheduler{}
	reqs := []*Request{
		{ID: "c", ArrivalTime: 300, Priority: 1.0},
		{ID: "a", ArrivalTime: 100, Priority: 3.0},
		{ID: "b", ArrivalTime: 200, Priority: 2.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"c", "a", "b"}
	if !sliceEqual(got, want) {
		t.Errorf("FCFSScheduler: got %v, want %v", got, want)
	}
}

func TestPriorityFCFSScheduler_SortsByPriorityDescending(t *testing.T) {
	// BC-3: higher priority first
	sched := &PriorityFCFSScheduler{}
	reqs := []*Request{
		{ID: "low", ArrivalTime: 100, Priority: 1.0},
		{ID: "high", ArrivalTime: 200, Priority: 3.0},
		{ID: "mid", ArrivalTime: 50, Priority: 2.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"high", "mid", "low"}
	if !sliceEqual(got, want) {
		t.Errorf("PriorityFCFS priority ordering: got %v, want %v", got, want)
	}
}

func TestPriorityFCFSScheduler_TieBreakByArrivalTime(t *testing.T) {
	// BC-3: same priority → earlier arrival first
	sched := &PriorityFCFSScheduler{}
	reqs := []*Request{
		{ID: "late", ArrivalTime: 300, Priority: 5.0},
		{ID: "early", ArrivalTime: 100, Priority: 5.0},
		{ID: "mid", ArrivalTime: 200, Priority: 5.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"early", "mid", "late"}
	if !sliceEqual(got, want) {
		t.Errorf("PriorityFCFS arrival tiebreak: got %v, want %v", got, want)
	}
}

func TestPriorityFCFSScheduler_TieBreakByID(t *testing.T) {
	// BC-8: same priority + same arrival → lexicographic ID
	sched := &PriorityFCFSScheduler{}
	reqs := []*Request{
		{ID: "charlie", ArrivalTime: 100, Priority: 5.0},
		{ID: "alpha", ArrivalTime: 100, Priority: 5.0},
		{ID: "bravo", ArrivalTime: 100, Priority: 5.0},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"alpha", "bravo", "charlie"}
	if !sliceEqual(got, want) {
		t.Errorf("PriorityFCFS ID tiebreak: got %v, want %v", got, want)
	}
}

func TestSJFScheduler_SortsByInputTokensAscending(t *testing.T) {
	// BC-4: shorter jobs first
	sched := &SJFScheduler{}
	reqs := []*Request{
		{ID: "long", ArrivalTime: 100, InputTokens: make([]int, 500)},
		{ID: "short", ArrivalTime: 200, InputTokens: make([]int, 50)},
		{ID: "medium", ArrivalTime: 50, InputTokens: make([]int, 200)},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"short", "medium", "long"}
	if !sliceEqual(got, want) {
		t.Errorf("SJF input token ordering: got %v, want %v", got, want)
	}
}

func TestSJFScheduler_TieBreakByArrivalTime(t *testing.T) {
	// BC-4: same length → earlier arrival first
	sched := &SJFScheduler{}
	reqs := []*Request{
		{ID: "late", ArrivalTime: 300, InputTokens: make([]int, 100)},
		{ID: "early", ArrivalTime: 100, InputTokens: make([]int, 100)},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"early", "late"}
	if !sliceEqual(got, want) {
		t.Errorf("SJF arrival tiebreak: got %v, want %v", got, want)
	}
}

func TestSJFScheduler_TieBreakByID(t *testing.T) {
	// BC-4 + BC-8: same length + same arrival → lexicographic ID
	sched := &SJFScheduler{}
	reqs := []*Request{
		{ID: "bravo", ArrivalTime: 100, InputTokens: make([]int, 100)},
		{ID: "alpha", ArrivalTime: 100, InputTokens: make([]int, 100)},
	}
	sched.OrderQueue(reqs, 0)

	got := requestIDs(reqs)
	want := []string{"alpha", "bravo"}
	if !sliceEqual(got, want) {
		t.Errorf("SJF ID tiebreak: got %v, want %v", got, want)
	}
}

func TestScheduler_AnyPolicy_PreservesAllRequests(t *testing.T) {
	// NC-2: sorting must not add/remove/duplicate requests
	schedulers := []struct {
		name  string
		sched InstanceScheduler
	}{
		{"fcfs", &FCFSScheduler{}},
		{"priority-fcfs", &PriorityFCFSScheduler{}},
		{"sjf", &SJFScheduler{}},
	}

	for _, tc := range schedulers {
		t.Run(tc.name, func(t *testing.T) {
			reqs := []*Request{
				{ID: "a", ArrivalTime: 100, Priority: 1.0, InputTokens: make([]int, 50)},
				{ID: "b", ArrivalTime: 200, Priority: 2.0, InputTokens: make([]int, 100)},
				{ID: "c", ArrivalTime: 300, Priority: 3.0, InputTokens: make([]int, 25)},
			}
			tc.sched.OrderQueue(reqs, 0)

			if len(reqs) != 3 {
				t.Fatalf("queue length changed: got %d, want 3", len(reqs))
			}
			seen := make(map[string]bool)
			for _, r := range reqs {
				if seen[r.ID] {
					t.Errorf("duplicate request %q", r.ID)
				}
				seen[r.ID] = true
			}
			for _, id := range []string{"a", "b", "c"} {
				if !seen[id] {
					t.Errorf("missing request %q", id)
				}
			}
		})
	}
}

func TestNewScheduler_ValidNames_ReturnsCorrectType(t *testing.T) {
	// EH-3: empty string returns FCFSScheduler
	s1 := NewScheduler("")
	if _, ok := s1.(*FCFSScheduler); !ok {
		t.Errorf("NewScheduler(\"\"): expected *FCFSScheduler, got %T", s1)
	}

	s2 := NewScheduler("fcfs")
	if _, ok := s2.(*FCFSScheduler); !ok {
		t.Errorf("NewScheduler(\"fcfs\"): expected *FCFSScheduler, got %T", s2)
	}

	s3 := NewScheduler("priority-fcfs")
	if _, ok := s3.(*PriorityFCFSScheduler); !ok {
		t.Errorf("NewScheduler(\"priority-fcfs\"): expected *PriorityFCFSScheduler, got %T", s3)
	}

	s4 := NewScheduler("sjf")
	if _, ok := s4.(*SJFScheduler); !ok {
		t.Errorf("NewScheduler(\"sjf\"): expected *SJFScheduler, got %T", s4)
	}
}

func TestNewScheduler_UnknownName_Panics(t *testing.T) {
	// EH-2: unknown name panics
	defer func() {
		r := recover()
		if r == nil {
			t.Errorf("NewScheduler(\"unknown\"): expected panic, got nil")
		}
	}()
	NewScheduler("unknown")
}

func TestScheduler_EmptyQueue_NoOp(t *testing.T) {
	// Edge case: empty queue must not panic or modify slice
	schedulers := []struct {
		name  string
		sched InstanceScheduler
	}{
		{"fcfs", &FCFSScheduler{}},
		{"priority-fcfs", &PriorityFCFSScheduler{}},
		{"sjf", &SJFScheduler{}},
	}
	for _, tc := range schedulers {
		t.Run(tc.name, func(t *testing.T) {
			reqs := []*Request{}
			tc.sched.OrderQueue(reqs, 0)
			if len(reqs) != 0 {
				t.Errorf("empty queue modified: got len %d, want 0", len(reqs))
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestFCFSScheduler -v 2>&1 | head -20`
Expected: FAIL (compilation error — types not defined yet)

**Step 3: Implement InstanceScheduler + templates**

Create `sim/scheduler.go`:

```go
package sim

import (
	"fmt"
	"sort"
)

// InstanceScheduler reorders the wait queue before batch formation.
// Called each step to determine which requests should be considered first.
// Implementations sort the slice in-place using sort.SliceStable for determinism.
type InstanceScheduler interface {
	OrderQueue(requests []*Request, clock int64)
}

// FCFSScheduler preserves First-Come-First-Served order (no-op).
// This is the default scheduler matching existing BLIS behavior.
type FCFSScheduler struct{}

func (f *FCFSScheduler) OrderQueue(_ []*Request, _ int64) {
	// No-op: FIFO order preserved from enqueue order
}

// PriorityFCFSScheduler sorts requests by priority (descending),
// then by arrival time (ascending), then by ID (ascending) for determinism.
type PriorityFCFSScheduler struct{}

func (p *PriorityFCFSScheduler) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].Priority != reqs[j].Priority {
			return reqs[i].Priority > reqs[j].Priority
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}

// SJFScheduler sorts requests by input token count (ascending, shortest first),
// then by arrival time (ascending), then by ID (ascending) for determinism.
// Warning: SJF can cause starvation for long requests under sustained load.
type SJFScheduler struct{}

func (s *SJFScheduler) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		li, lj := len(reqs[i].InputTokens), len(reqs[j].InputTokens)
		if li != lj {
			return li < lj
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}

// NewScheduler creates an InstanceScheduler by name.
// Valid names: "fcfs" (default), "priority-fcfs", "sjf".
// Empty string defaults to FCFSScheduler (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewScheduler(name string) InstanceScheduler {
	switch name {
	case "", "fcfs":
		return &FCFSScheduler{}
	case "priority-fcfs":
		return &PriorityFCFSScheduler{}
	case "sjf":
		return &SJFScheduler{}
	default:
		panic(fmt.Sprintf("unknown scheduler %q", name))
	}
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestFCFS|TestPriorityFCFS|TestSJF|TestScheduler_|TestNewScheduler" -v`
Expected: PASS (all 11 top-level tests)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/scheduler.go sim/scheduler_test.go
git commit -m "$(cat <<'EOF'
feat(sim): add InstanceScheduler interface with FCFS, PriorityFCFS, and SJF templates (BC-3, BC-4, BC-8)

- Add InstanceScheduler interface with OrderQueue(requests, clock)
- Implement FCFSScheduler (no-op, preserves existing behavior)
- Implement PriorityFCFSScheduler (priority desc, arrival asc, ID asc)
- Implement SJFScheduler (input tokens asc, arrival asc, ID asc)
- Add NewScheduler factory with panic on unknown names
- Unit tests for all templates, tie-breaking, queue preservation

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

#### --- CHECKPOINT: Review Batch 1 ---

**Verification:** Run `go test ./sim/... -v` — all new tests pass, all existing tests still pass.

**Review focus:** Are the sort comparators correct? Is `sort.SliceStable` used (not `sort.Slice`)?

---

#### Batch 2: Wiring & Integration (Tasks 3-5)

---

### Task 3: Wire PriorityPolicy + Scheduler into Simulator

**Contracts Implemented:** BC-1, BC-2, BC-5, BC-10

**Files:**
- Modify: `sim/simulator.go:73-92` (SimConfig), `sim/simulator.go:95-136` (Simulator struct), `sim/simulator.go:143-198` (NewSimulator), `sim/simulator.go:494-505` (Step)
- Test: `sim/scheduler_test.go` (append integration test)

**Step 1: Write failing integration test**

Context: Test that a Simulator with PriorityFCFSScheduler schedules higher-priority requests before lower-priority ones. We create a simulator without workload, inject two requests with different priorities, and verify scheduling order.

Append to `sim/scheduler_test.go`:

```go
func TestSimulator_PriorityFCFS_SchedulesHighPriorityFirst(t *testing.T) {
	// BC-2 + BC-5: SLO-based priority assigns higher priority to older requests;
	// priority-fcfs scheduler should schedule the older (higher-priority) request first.
	// Uses MaxRunningReqs=1 to force sequential scheduling so step index proves ordering.
	cfg := SimConfig{
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     1,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		PriorityPolicy:     "slo-based",
		Scheduler:          "priority-fcfs",
	}
	s := NewSimulator(cfg)

	// reqNewer arrives later (lower age → lower priority from SLO-based policy)
	// Inject it first so FCFS would schedule it first — but priority should override.
	reqNewer := &Request{
		ID:           "req_newer",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 5),
		ArrivalTime:  500000,
		State:        "queued",
	}
	// reqOlder arrives earlier (higher age → higher priority)
	reqOlder := &Request{
		ID:           "req_older",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 5),
		ArrivalTime:  0,
		State:        "queued",
	}

	// Inject newer first, then older — priority-fcfs should reorder
	s.InjectArrival(reqNewer)
	s.InjectArrival(reqOlder)

	s.Run()

	// Both should complete
	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed: got %d, want 2", s.Metrics.CompletedRequests)
	}

	// Older request (higher priority) should have been scheduled first
	if reqOlder.ScheduledStepIdx > reqNewer.ScheduledStepIdx {
		t.Errorf("priority inversion: older scheduled at step %d, newer at step %d",
			reqOlder.ScheduledStepIdx, reqNewer.ScheduledStepIdx)
	}
}

func TestSimulator_DefaultConfig_MatchesFCFS(t *testing.T) {
	// BC-1: default config (empty strings) uses ConstantPriority + FCFSScheduler
	cfg := SimConfig{
		Horizon:            1000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		// PriorityPolicy and Scheduler left empty (defaults)
	}
	s := NewSimulator(cfg)

	// Verify correct types were created
	if _, ok := s.priorityPolicy.(*ConstantPriority); !ok {
		t.Errorf("default priorityPolicy: got %T, want *ConstantPriority", s.priorityPolicy)
	}
	if _, ok := s.scheduler.(*FCFSScheduler); !ok {
		t.Errorf("default scheduler: got %T, want *FCFSScheduler", s.scheduler)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run "TestSimulator_PriorityFCFS|TestSimulator_DefaultConfig_MatchesFCFS" -v 2>&1 | head -20`
Expected: FAIL (Simulator doesn't have priorityPolicy/scheduler fields yet)

**Step 3: Implement wiring in Simulator**

In `sim/simulator.go`, make these changes:

**3a.** Add fields to `SimConfig` (after line 91, before closing brace):
```go
PriorityPolicy string // "constant" (default) or "slo-based"
Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf"
```

**3b.** Add fields to `Simulator` struct (after line 135, before closing brace):
```go
priorityPolicy PriorityPolicy
scheduler      InstanceScheduler
```

**3c.** In `NewSimulator()`, initialize the fields (after `s.rng = ...` on line 186, before the workload conditional):
```go
s.priorityPolicy = NewPriorityPolicy(cfg.PriorityPolicy)
s.scheduler = NewScheduler(cfg.Scheduler)
```

**3d.** In `Step()`, add priority assignment and queue ordering (after `runningBatchFeatures` reset on line 503, before `sim.makeRunningBatch(now)` on line 505):
```go
// Assign priorities to queued requests and order queue per scheduler policy
for _, req := range sim.WaitQ.queue {
	req.Priority = sim.priorityPolicy.Compute(req, now)
}
sim.scheduler.OrderQueue(sim.WaitQ.queue, now)
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/... -run "TestSimulator_PriorityFCFS|TestSimulator_DefaultConfig_MatchesFCFS" -v`
Expected: PASS

Also verify all existing tests still pass:
Run: `go test ./sim/... -v 2>&1 | tail -20`
Expected: PASS (all tests including golden dataset)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/scheduler_test.go
git commit -m "$(cat <<'EOF'
feat(sim): wire PriorityPolicy and InstanceScheduler into Simulator.Step (BC-1, BC-2, BC-5)

- Add PriorityPolicy and Scheduler fields to SimConfig
- Add priorityPolicy and scheduler fields to Simulator struct
- Initialize from config in NewSimulator constructor
- Call priority assignment + OrderQueue in Step before makeRunningBatch
- Integration test: priority-fcfs schedules high priority first
- Verify default config creates ConstantPriority + FCFSScheduler

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Wire through DeploymentConfig + CLI Flags

**Contracts Implemented:** BC-9

**Files:**
- Modify: `sim/cluster/deployment.go:7-31` and `sim/cluster/deployment.go:37-55`
- Modify: `cmd/root.go:57-62`, `cmd/root.go:202-224`, `cmd/root.go:292-297`

**Step 1: Write failing test for DeploymentConfig field mapping**

Context: Verify the new fields flow through `ToSimConfig()`.

Append to the existing `sim/cluster/cluster_test.go` (the `TestDeploymentConfig_ToSimConfig_FieldMapping` test should be extended). Since the existing test already checks individual fields, add a new test specifically for the new fields.

Create/append to `sim/cluster/cluster_test.go`:

```go
func TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields(t *testing.T) {
	dc := DeploymentConfig{
		NumInstances:    1,
		Horizon:         999,
		Seed:            7,
		TotalKVBlocks:   500,
		BlockSizeTokens: 32,
		MaxRunningReqs:  128,
		MaxScheduledTokens: 4096,
		BetaCoeffs:      []float64{1, 2, 3},
		AlphaCoeffs:     []float64{4, 5, 6},
		PriorityPolicy:  "slo-based",
		Scheduler:       "priority-fcfs",
	}

	sc := dc.ToSimConfig()

	if sc.PriorityPolicy != "slo-based" {
		t.Errorf("PriorityPolicy: got %q, want %q", sc.PriorityPolicy, "slo-based")
	}
	if sc.Scheduler != "priority-fcfs" {
		t.Errorf("Scheduler: got %q, want %q", sc.Scheduler, "priority-fcfs")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields -v 2>&1 | head -20`
Expected: FAIL (fields don't exist in DeploymentConfig yet)

**Step 3: Implement DeploymentConfig + CLI changes**

**3a.** In `sim/cluster/deployment.go`, add fields to `DeploymentConfig` (after `TokenBucketRefillRate` line 30, before closing brace):
```go
PriorityPolicy string // "constant" (default) or "slo-based"
Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf"
```

**3b.** In `sim/cluster/deployment.go`, add to `ToSimConfig()` return (after `Roofline: d.Roofline,` line 53):
```go
PriorityPolicy: d.PriorityPolicy,
Scheduler:      d.Scheduler,
```

**3c.** In `cmd/root.go`, add flag variables (after `tokenBucketRefillRate` line 62):
```go
// Priority and scheduler config (PR7)
priorityPolicy string // Priority policy name
scheduler      string // Scheduler name
```

**3d.** In `cmd/root.go`, add to `DeploymentConfig` construction (after `TokenBucketRefillRate: tokenBucketRefillRate,` line 223):
```go
PriorityPolicy:    priorityPolicy,
Scheduler:         scheduler,
```

**3e.** In `cmd/root.go`, add flag registrations (after token bucket flags around line 297, before results path):
```go
// Priority and scheduler config (PR7)
runCmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based")
runCmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf")
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/deployment.go sim/cluster/cluster_test.go cmd/root.go
git commit -m "$(cat <<'EOF'
feat(cmd): add --priority-policy and --scheduler CLI flags (BC-9)

- Add PriorityPolicy and Scheduler fields to DeploymentConfig
- Wire through ToSimConfig() for per-instance construction
- Add CLI flags: --priority-policy (constant, slo-based), --scheduler (fcfs, priority-fcfs, sjf)
- Test DeploymentConfig field mapping

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Golden Dataset Verification + End-to-End Tests

**Contracts Implemented:** BC-1, NC-1

**Files:**
- Test: `sim/scheduler_test.go` (append e2e tests)
- Verify: `sim/simulator_test.go` (existing golden dataset test)
- Verify: `sim/cluster/instance_test.go` (existing cluster golden dataset test)

**Step 1: Run golden dataset tests to verify backward compatibility**

Run: `go test ./sim/... -run TestSimulator_GoldenDataset -v`
Expected: PASS (all golden dataset cases match exactly — BC-1, NC-1)

Run: `go test ./sim/cluster/... -run TestCluster -v`
Expected: PASS (cluster tests also match)

**Step 2: Write end-to-end SJF scheduling test**

Append to `sim/scheduler_test.go`:

```go
func TestSimulator_SJF_SchedulesShortJobFirst(t *testing.T) {
	// BC-4 + BC-5: SJF should schedule shorter input request first
	cfg := SimConfig{
		Horizon:            1000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     1, // only 1 slot: forces sequential scheduling
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Scheduler:          "sjf",
	}
	s := NewSimulator(cfg)

	// Long request arrives first
	reqLong := &Request{
		ID:           "req_long",
		InputTokens:  make([]int, 200),
		OutputTokens: make([]int, 2),
		ArrivalTime:  0,
		State:        "queued",
	}
	// Short request arrives second
	reqShort := &Request{
		ID:           "req_short",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 2),
		ArrivalTime:  0,
		State:        "queued",
	}

	s.InjectArrival(reqLong)
	s.InjectArrival(reqShort)

	s.Run()

	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed: got %d, want 2", s.Metrics.CompletedRequests)
	}

	// Short job should be scheduled at an earlier or same step
	if reqShort.ScheduledStepIdx > reqLong.ScheduledStepIdx {
		t.Errorf("SJF violation: short scheduled at step %d, long at step %d",
			reqShort.ScheduledStepIdx, reqLong.ScheduledStepIdx)
	}
}

func TestSimulator_SLOBased_PriorityFCFS_OlderRequestFirst(t *testing.T) {
	// BC-7 + BC-5: SLO-based priority with priority-fcfs scheduler
	// Older requests should get higher priority and schedule first
	cfg := SimConfig{
		Horizon:            10000000,
		Seed:               42,
		TotalKVBlocks:      1000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     1, // only 1 slot: forces sequential scheduling
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		PriorityPolicy:     "slo-based",
		Scheduler:          "priority-fcfs",
	}
	s := NewSimulator(cfg)

	// Newer request injected first (arrives at t=500000)
	reqNew := &Request{
		ID:           "req_new",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 2),
		ArrivalTime:  500000,
		State:        "queued",
	}
	// Older request injected second (arrives at t=0)
	reqOld := &Request{
		ID:           "req_old",
		InputTokens:  make([]int, 20),
		OutputTokens: make([]int, 2),
		ArrivalTime:  0,
		State:        "queued",
	}

	s.InjectArrival(reqNew)
	s.InjectArrival(reqOld)

	s.Run()

	if s.Metrics.CompletedRequests != 2 {
		t.Fatalf("completed: got %d, want 2", s.Metrics.CompletedRequests)
	}

	// Older request should be scheduled first (higher priority from age)
	if reqOld.ScheduledStepIdx > reqNew.ScheduledStepIdx {
		t.Errorf("SLO-based priority violation: old scheduled at step %d, new at step %d",
			reqOld.ScheduledStepIdx, reqNew.ScheduledStepIdx)
	}
}
```

**Step 3: Run all new tests**

Run: `go test ./sim/... -run "TestSimulator_SJF|TestSimulator_SLOBased" -v`
Expected: PASS

**Step 4: Run full test suite**

Run: `go test ./...`
Expected: PASS (all tests including golden dataset)

**Step 5: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/scheduler_test.go
git commit -m "$(cat <<'EOF'
test(sim): add end-to-end scheduling integration tests (BC-1, BC-4, BC-5, BC-7, NC-1)

- Verify SJF schedules short jobs before long jobs
- Verify SLO-based priority with priority-fcfs schedules older requests first
- Golden dataset backward compatibility confirmed

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

#### --- CHECKPOINT: Review Batch 2 ---

**Verification:** Run `go test ./... && golangci-lint run ./...` — all tests pass, no lint issues.

**Review focus:** Check the Step() integration point (priority before ordering before batch formation). Verify golden dataset exact match.

---

#### Batch 3: Documentation (Task 6)

---

### Task 6: Update CLAUDE.md

**Contracts Implemented:** (documentation only)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

Apply these changes:

1. In the "Current Implementation Focus" section, update the completed PRs list:
   - Change `**Next:** PR6 (routing policies), then PR7+ (priority+scheduler, ...)` to include PR7 as completed

2. In the "File Organization" section, add the new files:
   - `sim/priority.go` — PriorityPolicy interface, ConstantPriority, SLOBasedPriority, factory
   - `sim/scheduler.go` — InstanceScheduler interface, FCFSScheduler, PriorityFCFSScheduler, SJFScheduler, factory

3. In the "Core Simulation Engine" section, add entries:
   - **priority.go**: `PriorityPolicy` interface with `ConstantPriority` and `SLOBasedPriority` templates, `NewPriorityPolicy` factory
   - **scheduler.go**: `InstanceScheduler` interface with `FCFSScheduler`, `PriorityFCFSScheduler`, and `SJFScheduler` templates, `NewScheduler` factory

4. Add new CLI flags to documentation:
   - `--priority-policy`: Priority policy (constant, slo-based)
   - `--scheduler`: Instance scheduler (fcfs, priority-fcfs, sjf)

**Step 2: Verify no stale references**

Grep CLAUDE.md for any references to `sim/policy/` (should not exist) and verify all file paths match reality.

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
docs: update CLAUDE.md for PR7 (PriorityPolicy + InstanceScheduler)

- Add priority.go and scheduler.go to file organization and architecture
- Update completed PR list
- Document new CLI flags: --priority-policy, --scheduler

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

#### --- CHECKPOINT: Review Batch 3 ---

**Verification:** Run `go test ./... && golangci-lint run ./...` one final time.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 5 | Golden | `TestSimulator_GoldenDataset` (existing) |
| BC-2 | Task 3 | Integration | `TestSimulator_PriorityFCFS_SchedulesHighPriorityFirst` (exercises per-step Compute loop) |
| BC-2 | Task 3 | Integration | `TestSimulator_DefaultConfig_MatchesFCFS` (verifies default types) |
| BC-3 | Task 2 | Unit | `TestPriorityFCFSScheduler_SortsByPriorityDescending` |
| BC-3 | Task 2 | Unit | `TestPriorityFCFSScheduler_TieBreakByArrivalTime` |
| BC-4 | Task 2 | Unit | `TestSJFScheduler_SortsByInputTokensAscending` |
| BC-4 | Task 2 | Unit | `TestSJFScheduler_TieBreakByArrivalTime` |
| BC-5 | Task 3 | Integration | `TestSimulator_PriorityFCFS_SchedulesHighPriorityFirst` |
| BC-5 | Task 5 | E2E | `TestSimulator_SJF_SchedulesShortJobFirst` |
| BC-6 | Task 1 | Unit | `TestConstantPriority_ReturnsFixedScore` |
| BC-6 | Task 1 | Unit | `TestConstantPriority_DefaultZero` |
| BC-7 | Task 1 | Unit | `TestSLOBasedPriority_OlderRequestGetsHigherPriority` |
| BC-7 | Task 1 | Unit | `TestSLOBasedPriority_FormulaCorrectness` |
| BC-7 | Task 5 | E2E | `TestSimulator_SLOBased_PriorityFCFS_OlderRequestFirst` |
| BC-8 | Task 2 | Unit | `TestPriorityFCFSScheduler_TieBreakByID` |
| BC-8 | Task 2 | Unit | `TestSJFScheduler_TieBreakByID` |
| BC-9 | Task 4 | Unit | `TestDeploymentConfig_ToSimConfig_PrioritySchedulerFields` |
| NC-1 | Task 5 | Golden | `TestSimulator_GoldenDataset` (existing) |
| BC-10 | Task 3 | Documentation | Documented in BC-10 contract + Risk Analysis (verified by design, not test) |
| NC-2 | Task 2 | Unit | `TestScheduler_AnyPolicy_PreservesAllRequests` |
| NC-3 | Task 1 | Unit | `TestPriorityPolicy_Compute_NoSideEffects` |
| EH-1 | Task 1 | Unit | `TestNewPriorityPolicy_UnknownName_Panics` |
| EH-2 | Task 2 | Unit | `TestNewScheduler_UnknownName_Panics` |
| EH-3 | Task 1 | Unit | `TestNewPriorityPolicy_ValidNames_ReturnsCorrectType` |
| EH-3 | Task 2 | Unit | `TestNewScheduler_ValidNames_ReturnsCorrectType` |
| Edge | Task 2 | Unit | `TestScheduler_EmptyQueue_NoOp` |

**Golden dataset update:** Not needed. Default config (constant + fcfs) produces identical output to current code.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Sort instability breaks FIFO within equal-priority groups | Low | High | Use `sort.SliceStable` (not `sort.Slice`); verified by tie-break tests | Task 2 |
| Priority assignment in Step() adds overhead to hot path | Low | Low | Priority assignment is O(n) over WaitQ, negligible vs KV allocation/step time | Task 3 |
| SJF starvation under sustained high load | Medium | Low | Documented limitation; not a bug but an inherent SJF property | Task 2 (doc comment) |
| Float64 priority comparison precision | Low | Medium | Using `!=` for float64; considered OK since values come from deterministic arithmetic, not accumulation | Task 2 |
| Preempted requests lose position after sort | Low | Low | Correct: preempted requests sort by their original priority/arrival, not re-queue position. Note: preemption happens inside `makeRunningBatch()` (after `OrderQueue`), so a re-queued request is not re-prioritized until the next step's `OrderQueue` call — one-step delay. | Task 3 |
| HOL blocking limits non-FCFS scheduling under memory pressure | Medium | Medium | `makeRunningBatch` breaks on KV allocation failure (`sim/simulator.go:452-456`), blocking all subsequent requests. This is intentional vLLM behavior. Documented in BC-10. | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — `OrderQueue` is simpler than full `MakeBatch`
- [x] No feature creep beyond PR scope — no SchedulerContext, BatchDecision, OnRequestArrival
- [x] No unexercised flags or interfaces — all templates reachable via CLI flags
- [x] No partial implementations — all 3 schedulers and 2 priority policies complete
- [x] No breaking changes — default config is identity operation
- [x] No hidden global state impact — policies stored in Simulator, config-driven
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: golden dataset tests use `sim/internal/testutil`; new scheduler/priority unit tests use package-local helpers (appropriate for `sim/` package-internal tests)
- [x] CLAUDE.md updated (Task 6): new files, CLI flags, completed PR
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — 3 deviations, all justified
- [x] Each task produces working, testable code (no scaffolding)
- [x] Task dependencies correctly ordered (1→2→3→4→5→6)
- [x] All contracts mapped to specific tasks (see Test Strategy table)
- [x] Golden dataset regeneration: not needed (defaults preserve behavior)

---

## Appendix: File-Level Implementation Details

### File: `sim/request.go`

**Purpose:** Add Priority field for scheduler-aware ordering.

**Change:** Add one field to the Request struct after line 33:

```go
Priority float64 // Priority score assigned by PriorityPolicy (higher = more urgent)
```

**Key Notes:**
- Zero value (0.0) is the default, matching ConstantPriority default behavior
- Set by Simulator.Step() each step, read by schedulers
- Not persisted in metrics (priority is a transient scheduling signal)

---

### File: `sim/priority.go` (NEW)

**Purpose:** PriorityPolicy interface and two template implementations.

**Complete Implementation:**

```go
package sim

import "fmt"

// PriorityPolicy computes a priority score for a request.
// Higher scores indicate higher priority (scheduled first by priority-aware schedulers).
// Implementations MUST NOT modify the request — only the return value is used.
type PriorityPolicy interface {
	Compute(req *Request, clock int64) float64
}

// ConstantPriority assigns a fixed priority score to all requests.
type ConstantPriority struct {
	Score float64
}

func (c *ConstantPriority) Compute(_ *Request, _ int64) float64 {
	return c.Score
}

// SLOBasedPriority computes priority based on request age (time waiting).
// Older requests receive higher priority to reduce SLO violation risk.
// Formula: BaseScore + AgeWeight * float64(clock - req.ArrivalTime)
//
// Full SLO class integration (using TenantState) is planned for PR8.
type SLOBasedPriority struct {
	BaseScore float64
	AgeWeight float64
}

func (s *SLOBasedPriority) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	return s.BaseScore + s.AgeWeight*age
}

// NewPriorityPolicy creates a PriorityPolicy by name.
// Valid names: "constant" (default), "slo-based".
// Empty string defaults to ConstantPriority (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewPriorityPolicy(name string) PriorityPolicy {
	switch name {
	case "", "constant":
		return &ConstantPriority{Score: 0.0}
	case "slo-based":
		return &SLOBasedPriority{BaseScore: 0.0, AgeWeight: 1e-6}
	default:
		panic(fmt.Sprintf("unknown priority policy %q", name))
	}
}
```

**Key Notes:**
- RNG usage: None
- Metrics: None (priority is transient)
- Error handling: panic on unknown name (matches AdmissionPolicy pattern at `sim/admission.go:63`)
- SLOBasedPriority AgeWeight=1e-6: a request waiting 1 second (1e6 ticks) gets +1.0 priority

---

### File: `sim/scheduler.go` (NEW)

**Purpose:** InstanceScheduler interface and three template implementations.

**Complete Implementation:**

```go
package sim

import (
	"fmt"
	"sort"
)

// InstanceScheduler reorders the wait queue before batch formation.
// Called each step to determine which requests should be considered first.
// Implementations sort the slice in-place using sort.SliceStable for determinism.
type InstanceScheduler interface {
	OrderQueue(requests []*Request, clock int64)
}

// FCFSScheduler preserves First-Come-First-Served order (no-op).
// This is the default scheduler matching existing BLIS behavior.
type FCFSScheduler struct{}

func (f *FCFSScheduler) OrderQueue(_ []*Request, _ int64) {
	// No-op: FIFO order preserved from enqueue order
}

// PriorityFCFSScheduler sorts requests by priority (descending),
// then by arrival time (ascending), then by ID (ascending) for determinism.
type PriorityFCFSScheduler struct{}

func (p *PriorityFCFSScheduler) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].Priority != reqs[j].Priority {
			return reqs[i].Priority > reqs[j].Priority
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}

// SJFScheduler sorts requests by input token count (ascending, shortest first),
// then by arrival time (ascending), then by ID (ascending) for determinism.
// Warning: SJF can cause starvation for long requests under sustained load.
type SJFScheduler struct{}

func (s *SJFScheduler) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		li, lj := len(reqs[i].InputTokens), len(reqs[j].InputTokens)
		if li != lj {
			return li < lj
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}

// NewScheduler creates an InstanceScheduler by name.
// Valid names: "fcfs" (default), "priority-fcfs", "sjf".
// Empty string defaults to FCFSScheduler (for CLI flag default compatibility).
// Panics on unrecognized names.
func NewScheduler(name string) InstanceScheduler {
	switch name {
	case "", "fcfs":
		return &FCFSScheduler{}
	case "priority-fcfs":
		return &PriorityFCFSScheduler{}
	case "sjf":
		return &SJFScheduler{}
	default:
		panic(fmt.Sprintf("unknown scheduler %q", name))
	}
}
```

**Key Notes:**
- RNG usage: None (deterministic sort)
- Event ordering: Scheduler runs before makeRunningBatch, does not interact with event queue
- State mutation: Sorts `WaitQ.queue` in-place (same backing array)
- `sort.SliceStable` is critical — `sort.Slice` would break FIFO within equal groups

---

### File: `sim/simulator.go` (MODIFIED)

**Purpose:** Wire PriorityPolicy and InstanceScheduler into SimConfig, Simulator struct, constructor, and Step.

**Changes:**

1. **SimConfig** — add 2 fields after `TracesWorkloadFilePath`:
```go
PriorityPolicy string // "constant" (default) or "slo-based"
Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf"
```

2. **Simulator struct** — add 2 fields after `rng`:
```go
priorityPolicy PriorityPolicy
scheduler      InstanceScheduler
```

3. **NewSimulator** — add 2 lines after `s.rng = NewPartitionedRNG(...)`:
```go
s.priorityPolicy = NewPriorityPolicy(cfg.PriorityPolicy)
s.scheduler = NewScheduler(cfg.Scheduler)
```

4. **Step** — add 4 lines after `runningBatchFeatures` reset, before `makeRunningBatch`:
```go
// Assign priorities to queued requests and order queue per scheduler policy
for _, req := range sim.WaitQ.queue {
    req.Priority = sim.priorityPolicy.Compute(req, now)
}
sim.scheduler.OrderQueue(sim.WaitQ.queue, now)
```

**Key Notes:**
- Priority assignment iterates all queued requests each step — O(n) per step
- Queue ordering is O(n log n) per step for non-FCFS schedulers
- Both are negligible compared to KV allocation and step time estimation
- Default (constant + fcfs): priority loop assigns all 0.0, OrderQueue is no-op — zero behavioral change

---

### File: `sim/cluster/deployment.go` (MODIFIED)

**Purpose:** Pass PriorityPolicy and Scheduler config through to SimConfig.

**Changes:**

1. **DeploymentConfig** — add 2 fields after `TokenBucketRefillRate`:
```go
PriorityPolicy string // "constant" (default) or "slo-based"
Scheduler      string // "fcfs" (default), "priority-fcfs", "sjf"
```

2. **ToSimConfig** — add 2 fields to return struct after `Roofline`:
```go
PriorityPolicy: d.PriorityPolicy,
Scheduler:      d.Scheduler,
```

---

### File: `cmd/root.go` (MODIFIED)

**Purpose:** Add CLI flags for priority policy and scheduler selection.

**Changes:**

1. **Flag variables** — add after `tokenBucketRefillRate` (line 62):
```go
// Priority and scheduler config (PR7)
priorityPolicy string // Priority policy name
scheduler      string // Scheduler name
```

2. **DeploymentConfig construction** — add after `TokenBucketRefillRate`:
```go
PriorityPolicy:    priorityPolicy,
Scheduler:         scheduler,
```

3. **Flag registrations** — add after token bucket flags, before results path:
```go
// Priority and scheduler config (PR7)
runCmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based")
runCmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf")
```
