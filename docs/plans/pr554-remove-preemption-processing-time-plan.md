# PR #554: Remove PreemptionProcessingTime from LatencyModel Interface

- **Goal:** Remove the dead `PreemptionProcessingTime()` method from the `LatencyModel` interface and the dead `PreemptionEvent` type, simplifying the latency model contract.
- **The problem today:** `PreemptionProcessingTime()` is a method on the `LatencyModel` interface that all three backends implement as `return 0`. The `PreemptionEvent` type's `Execute()` method does nothing but log a debug message. Together, they add boilerplate to every latency backend and create the false impression that preemption overhead is a tunable parameter, when the actual cost of preemption (re-prefill) is already correctly modeled by the existing `ProgressIndex = 0` reset in batch formation.
- **What this PR adds:**
  1. Removes `PreemptionProcessingTime() int64` from the `LatencyModel` interface — one fewer method every backend must implement
  2. Removes the dead `PreemptionEvent` type — simplifies event.go
  3. Removes the `PreemptionDelay` field from `PreemptedRequest` — simpler batch result type
  4. Replaces `PreemptionEvent` scheduling with an inline `logrus.Debugf` call — preserves debug observability
- **Why this matters:** Reducing interface surface area lowers the cost of adding new latency backends (4 methods instead of 5) and eliminates a misleading extension point.
- **Architecture:** Pure removal from `sim/latency_model.go` (interface), `sim/event.go` (event type), `sim/batch_formation.go` (field + call), `sim/simulator.go` (event scheduling), `sim/latency/latency.go` and `sim/latency/crossmodel.go` (implementations), plus documentation updates.
- **Source:** [GitHub issue #554](https://github.com/inference-sim/inference-sim/issues/554)
- **Closes:** Fixes #554
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** LatencyModel interface (`sim/latency_model.go`) — defines the contract for all latency estimation backends.
2. **Adjacent blocks:** BatchFormation (`sim/batch_formation.go`) calls `PreemptionProcessingTime()` during preemption. Simulator (`sim/simulator.go`) schedules `PreemptionEvent` using the delay. Three backends in `sim/latency/` implement the method.
3. **Invariants touched:** None directly. INV-1 (request conservation) is unaffected because `PreemptionCount++` is retained. INV-8 (work-conserving) is unaffected because preemption logic is unchanged — only the delay and event are removed.
4. **Construction Site Audit:**
   - `PreemptedRequest` struct: constructed at `sim/batch_formation.go:177`. One site. Removing `PreemptionDelay` field.
   - `PreemptionEvent` struct: constructed at `sim/simulator.go:333`. One site. Being deleted entirely.
   - No struct fields are being added.

---

## Part 1: Design Validation

### A) Executive Summary

This PR removes dead code from the latency model interface and event system. `PreemptionProcessingTime()` always returns 0 in all three backends and has never been configurable. `PreemptionEvent` is a no-op event type whose only effect is a debug log. The preemption counter (`PreemptionCount++`) and all preemption logic (KV release, state reset, re-enqueue) are preserved — only the delay calculation and the no-op event scheduling are removed. Debug observability is preserved by replacing the event with an inline `logrus.Debugf` at the preemption site.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Interface Simplification
- GIVEN the LatencyModel interface
- WHEN a new backend is implemented
- THEN it needs to implement only 4 methods (StepTime, QueueingTime, OutputTokenProcessingTime, SchedulingProcessingTime)
- MECHANISM: PreemptionProcessingTime() removed from the interface definition
```

```
BC-2: Preemption Metrics Preserved
- GIVEN a simulation with KV cache pressure that triggers preemptions
- WHEN preemption occurs during batch formation
- THEN PreemptionCount is still incremented for each evicted request
- MECHANISM: PreemptionCount++ remains in scheduleBatch, unchanged
```

```
BC-3: Preemption Behavior Preserved
- GIVEN a running request that is preempted
- WHEN batch formation evicts it
- THEN the request is re-enqueued at queue front with ProgressIndex=0 and KV blocks released
- MECHANISM: preemptForTokens logic is unchanged except removal of the delay computation
```

```
BC-4: Debug Observability Preserved
- GIVEN a simulation with preemptions and debug logging enabled
- WHEN preemption occurs
- THEN a debug log message is emitted for each preempted request
- MECHANISM: logrus.Debugf in scheduleBatch replaces the PreemptionEvent.Execute() debug log
```

**Negative contracts:**

```
BC-5: No Golden Dataset Impact
- GIVEN the golden dataset
- WHEN tests run after this change
- THEN all golden dataset tests pass unchanged
- MECHANISM: PreemptionProcessingTime was 0 in all backends; PreemptionEvent.Execute was a no-op
```

```
BC-6: No Behavioral Change
- GIVEN any simulation configuration
- WHEN run before and after this change with the same seed
- THEN the output is byte-identical (INV-6)
- MECHANISM: removed code had zero observable effect (0 delay, no-op execute)
```

### C) Component Interaction

```
LatencyModel interface (sim/latency_model.go)
  - BEFORE: 5 methods including PreemptionProcessingTime()
  - AFTER:  4 methods
  |
  +-- BlackboxLatencyModel (sim/latency/latency.go)
  |     BEFORE: implements PreemptionProcessingTime() { return 0 }
  |     AFTER:  method removed
  |
  +-- RooflineLatencyModel (sim/latency/latency.go)
  |     BEFORE: implements PreemptionProcessingTime() { return 0 }
  |     AFTER:  method removed
  |
  +-- CrossModelLatencyModel (sim/latency/crossmodel.go)
        BEFORE: implements PreemptionProcessingTime() { return 0 }
        AFTER:  method removed

BatchFormation (sim/batch_formation.go)
  - preemptForTokens()
    BEFORE: calls v.latencyModel.PreemptionProcessingTime(), stores in PreemptionDelay
    AFTER:  no delay computation
  - PreemptedRequest struct
    BEFORE: has PreemptionDelay field
    AFTER:  field removed

Simulator (sim/simulator.go)
  - scheduleBatch()
    BEFORE: schedules PreemptionEvent for each preempted request using p.PreemptionDelay
    AFTER:  emits logrus.Debugf for each preempted request; PreemptionCount++ preserved

Event types (sim/event.go)
  - BEFORE: PreemptionEvent type with Timestamp() and Execute() methods
  - AFTER:  type removed entirely
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue says "Update `problem.md` Section 2a" | Updates the PreemptionProcessingTime row | ALIGNED |
| Issue says "Remove `PreemptionEvent` scheduling in `scheduleBatch`" | Replaces with inline `logrus.Debugf` | ADDITION: preserves debug observability that the event provided |
| Issue does not mention documentation updates beyond problem.md | Also updates `docs/concepts/core-engine.md`, `docs/guide/latency-models.md`, `docs/contributing/extension-recipes.md`, `training/research.md`, `CLAUDE.md`, `docs/contributing/templates/design-guidelines.md`, and additional `training/problem.md` references | ADDITION: documentation DRY — all working copies of the interface description must be updated |
| Hypothesis FINDINGS reference PreemptionEvent in mechanism explanation | `hypotheses/h-overload-kv/FINDINGS.md` line 90 — add footnote noting removal in #554 | ADDITION: historical accuracy for future readers |
| Archived/active plan docs reference PreemptionProcessingTime | NOT updated — `docs/plans/` files are historical snapshots | EXCLUSION: plan documents are frozen records of their time |
| `training/problem.md` and `training/research.md` reference interface | NOT updated in this PR — files are untracked (never committed) and absent from worktree | EXCLUSION: untracked files cannot be modified in a worktree-based PR. Updates deferred to when these files are committed. |

### E) Review Guide

**The tricky part:** Ensuring that the `PreemptionEvent` removal doesn't break the event queue's ordering properties. Since `PreemptionEvent.Execute()` was a no-op (only debug log), and the preemption delay was always 0, removing it cannot change event ordering or simulation behavior.

**What to scrutinize:** The `scheduleBatch` change — verify that `PreemptionCount++` is preserved and the debug log replacement covers the same information. Verify no other code references `PreemptionEvent` or `PreemptionDelay`.

**What's safe to skim:** Documentation updates (straightforward removals from tables). The latency backend changes (mechanical deletion of `return 0` methods).

**Known debt:** `SchedulingProcessingTime()` is also always 0 across all backends — but unlike `PreemptionProcessingTime()`, its return value feeds into `ScheduledEvent` timing which is consumed by `RequestSchedulingDelays` at `simulator.go:346`. It is a dormant-but-wired extension point, not dead code. A future issue could track its removal if the wiring is never activated. Out of scope for this PR.

**Debug log interleaving change:** The replacement `logrus.Debugf` runs synchronously during `scheduleBatch`, whereas the original `PreemptionEvent.Execute()` ran when the event was popped from the queue. Since both fire at the same simulation clock value (`now + 0 = now`), the content is identical, but debug log ordering in stderr changes (preemption messages now appear grouped during batch formation rather than interleaved with other events). No structured log parsing exists, so this is cosmetic.

**Verification notes:** Confirmed via grep that `PreemptionDelay` is consumed at exactly one site (`simulator.go:334`). Confirmed `PreemptionEvent` is not referenced in `sim/trace/` or `sim/cluster/`. No test files type-assert on `PreemptionEvent`.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify:
- `sim/latency_model.go` — remove `PreemptionProcessingTime()` from interface (2 lines)
- `sim/event.go` — remove `PreemptionEvent` type (15 lines)
- `sim/batch_formation.go` — remove `PreemptionDelay` field, simplify `preemptForTokens`
- `sim/simulator.go` — replace `PreemptionEvent` scheduling with debug log
- `sim/latency/latency.go` — remove method from BlackboxLatencyModel and RooflineLatencyModel
- `sim/latency/crossmodel.go` — remove method from CrossModelLatencyModel
- `sim/latency/latency_test.go` — update `TestBlackboxLatencyModel_PlaceholderOverheads`
- `CLAUDE.md` — update "5 methods" → "4 methods" (line 221), remove "Preemption" from event types (line 224)
- `sim/doc.go` — remove "preemption/" from LatencyModel overhead list (line 26)
- `docs/concepts/core-engine.md` — remove PreemptionEvent from event table, update Phase 1 description, update preemption section
- `docs/guide/latency-models.md` — update interface method table, "five methods" → "four methods"
- `docs/contributing/extension-recipes.md` — update latency backend recipe, "5 methods" → "4 methods"
- `docs/contributing/templates/design-guidelines.md` — update "5 methods" → "4 methods" (line 243)
- `hypotheses/h-overload-kv/FINDINGS.md` — add footnote about PreemptionEvent removal (line 90)

No dead code introduced. No new files. No new abstractions.

### G) Task Breakdown

#### Task 1: Remove PreemptionProcessingTime, PreemptionEvent, and PreemptionDelay (BC-1, BC-3, BC-4)

**Contracts:** BC-1, BC-3, BC-4

**Note:** All code changes must be applied atomically because removing `PreemptionProcessingTime()` from the interface causes a compile failure in `batch_formation.go` (which calls the method). The interface removal, backend removals, caller removal, event type removal, and struct field removal are all done in one step.

**Test update:**

Modify `sim/latency/latency_test.go` — remove the `PreemptionProcessingTime` assertion from `TestBlackboxLatencyModel_PlaceholderOverheads`:

```go
// TestBlackboxLatencyModel_PlaceholderOverheads verifies placeholders return 0.
func TestBlackboxLatencyModel_PlaceholderOverheads(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	if model.SchedulingProcessingTime() != 0 {
		t.Errorf("SchedulingProcessingTime = %d, want 0", model.SchedulingProcessingTime())
	}
}
```

**Implementation:**

1. `sim/latency_model.go`: Remove lines 21-22 (`PreemptionProcessingTime` from interface)
2. `sim/latency/latency.go`: Remove `BlackboxLatencyModel.PreemptionProcessingTime()` (lines 58-60) and `RooflineLatencyModel.PreemptionProcessingTime()` (lines 107-109)
3. `sim/latency/crossmodel.go`: Remove `CrossModelLatencyModel.PreemptionProcessingTime()` (lines 77-79)
4. `sim/event.go`: Remove lines 82-96 (`PreemptionEvent` type, `Timestamp()`, `Execute()`)
5. `sim/batch_formation.go`:
   - Remove `PreemptionDelay` field from `PreemptedRequest` struct (line 44)
   - In `preemptForTokens`, remove the `preemptionDelay` variable and simplify `PreemptedRequest` construction:
     ```go
     result.Preempted = append(result.Preempted, PreemptedRequest{
         Request: preemptedRequest,
     })
     ```
6. `sim/simulator.go` — in `scheduleBatch`, replace the PreemptionEvent scheduling loop (lines 332-338) with:
   ```go
   // Record preemption metrics and emit debug log for each preempted request
   for _, p := range batchResult.Preempted {
       logrus.Debugf("<< Preemption: %s at %d ticks", p.Request.ID, now)
       sim.Metrics.PreemptionCount++
   }
   ```

**Note:** Do NOT remove the `latencyModel` field from `VLLMBatchFormation` — it is still needed for `SchedulingProcessingTime()` at `batch_formation.go:143`.

**Verify:**
```bash
cd .worktrees/pr554-remove-preemption-processing-time && go build ./... && go test ./... -count=1
```

**Lint:**
```bash
cd .worktrees/pr554-remove-preemption-processing-time && golangci-lint run ./...
```

#### Task 2: Update documentation (BC-1 doc updates)

**Contracts:** Documentation accuracy

**Files:**

1. `CLAUDE.md`:
   - Line 221: Update "5 methods" → "4 methods" in `latency_model.go` comment
   - Line 224: Remove "Preemption" from `event.go` event type list
2. `sim/doc.go`:
   - Line 26: Remove "preemption/" from LatencyModel overhead list → `scheduling/output processing overheads`
3. `docs/concepts/core-engine.md`:
   - Line 30: Remove `PreemptionEvent` row from event table
   - Line 74: Remove "Schedule `PreemptionEvent` for any evicted requests" from Phase 1 steps
   - Line 167: Replace "A `PreemptionEvent` is recorded for tracing" with "A debug log is emitted"
5. `docs/guide/latency-models.md`:
   - Line 159: Remove `PreemptionProcessingTime()` row from interface method table
   - Lines 151, 163: Update "five methods" → "four methods"
6. `docs/contributing/extension-recipes.md`:
   - Line 87: Update "5 methods" → "4 methods"
   - Line 92: Remove `PreemptionProcessingTime()` from method list
7. `docs/contributing/templates/design-guidelines.md`:
   - Line 243: Update "5 methods" → "4 methods"
8. `hypotheses/h-overload-kv/FINDINGS.md`:
   - Line 90: Add footnote: "(Note: `PreemptionEvent` removed in #554; preemptions now emit debug log only, reducing event count per preemption cycle from 3 to 2)"

**Verify:**
```bash
cd .worktrees/pr554-remove-preemption-processing-time && go build ./... && go test ./... -count=1
```

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit (negative) | Compile-time: backends without PreemptionProcessingTime satisfy LatencyModel |
| BC-1 | Task 1 | Unit (update) | TestBlackboxLatencyModel_PlaceholderOverheads — updated |
| BC-2 | Task 1 | Integration | Existing preemption tests verify PreemptionCount is still tracked |
| BC-3 | Task 1 | Integration | Existing preemption tests verify preemption behavior unchanged |
| BC-5 | Task 1 | Regression | Existing golden dataset tests pass unchanged |
| BC-6 | Task 1 | Determinism | Existing determinism tests pass unchanged |

**Key insight:** This is a pure removal of dead code (always-zero return values and no-op execution). Existing tests verify the properties we care about (preemption counting, request conservation, KV cache conservation). No new tests are needed because no new behavior is introduced.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| External code implements LatencyModel | Low | Medium | This is an internal interface; no known external implementors. Issue scope explicitly states all three backends. | Task 1 |
| Debug log format change breaks log parsing | Low | Low | The format string is consistent with other event debug logs. No structured log parsing exists. | Task 2 |
| Documentation references missed | Low | Low | grep for all references completed in Phase 0 analysis | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint (only removals + one debug log)
- [x] Shared test helpers used from existing shared test package (N/A)
- [x] CLAUDE.md updated: interface method count in extension recipes reference
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: all working copies of interface method list updated
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 atomic code change, Task 2 docs)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration: NOT needed (zero behavioral change)
- [x] Construction site audit completed: PreemptedRequest (1 site), PreemptionEvent (1 site, deleted)

**Antipattern rules:**
- [x] R1: No silent data loss (no new error paths)
- [x] R2: No new map iteration
- [x] R3: No new numeric parameters
- [x] R4: PreemptedRequest field removed; single construction site updated
- [x] R5: No new resource allocation loops
- [x] R6: No logrus.Fatalf in sim/
- [x] R7: No new golden tests
- [x] R8-R23: N/A (pure removal, no new code beyond a debug log)

---

## Appendix: File-Level Implementation Details

### File: `sim/latency_model.go`

Remove lines 21-22:
```go
// Before:
	// PreemptionProcessingTime estimates preemption overhead per eviction.
	PreemptionProcessingTime() int64

// After: (lines deleted)
```

### File: `sim/event.go`

Remove lines 82-96 (entire `PreemptionEvent` type):
```go
// DELETED:
// PreemptionEvent represents the pre-emption of an inference request in the system.
type PreemptionEvent struct { ... }
func (e *PreemptionEvent) Timestamp() int64 { ... }
func (e *PreemptionEvent) Execute(sim *Simulator) { ... }
```

### File: `sim/batch_formation.go`

1. Remove `PreemptionDelay` from `PreemptedRequest`:
```go
// Before:
type PreemptedRequest struct {
	Request         *Request
	PreemptionDelay int64
}

// After:
type PreemptedRequest struct {
	Request *Request
}
```

2. Simplify `preemptForTokens`:
```go
// Before:
			preemptionDelay := v.latencyModel.PreemptionProcessingTime()
			...
			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request:         preemptedRequest,
				PreemptionDelay: preemptionDelay,
			})

// After:
			...
			result.Preempted = append(result.Preempted, PreemptedRequest{
				Request: preemptedRequest,
			})
```

3. Remove `latencyModel` dependency from `VLLMBatchFormation` if it's only used for `PreemptionProcessingTime`. CHECK: `SchedulingProcessingTime()` is also called — `latencyModel` is still needed (line 143).

### File: `sim/simulator.go`

Replace PreemptionEvent scheduling in `scheduleBatch`:
```go
// Before (lines 332-338):
	for _, p := range batchResult.Preempted {
		sim.Schedule(&PreemptionEvent{
			time:    now + p.PreemptionDelay,
			Request: p.Request,
		})
		sim.Metrics.PreemptionCount++
	}

// After:
	for _, p := range batchResult.Preempted {
		logrus.Debugf("<< Preemption: %s at %d ticks", p.Request.ID, now)
		sim.Metrics.PreemptionCount++
	}
```

### File: `sim/latency/latency.go`

Remove two methods:
```go
// DELETED from BlackboxLatencyModel:
func (m *BlackboxLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}

// DELETED from RooflineLatencyModel:
func (m *RooflineLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}
```

### File: `sim/latency/crossmodel.go`

Remove one method:
```go
// DELETED from CrossModelLatencyModel:
func (m *CrossModelLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}
```

### File: `sim/latency/latency_test.go`

Update `TestBlackboxLatencyModel_PlaceholderOverheads` — remove PreemptionProcessingTime assertion:
```go
func TestBlackboxLatencyModel_PlaceholderOverheads(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	if model.SchedulingProcessingTime() != 0 {
		t.Errorf("SchedulingProcessingTime = %d, want 0", model.SchedulingProcessingTime())
	}
}
```
