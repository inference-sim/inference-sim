# Micro Plan: Random Tie-Breaking + H29 Erratum

- **Goal:** Eliminate deterministic positional bias in routing tie-breaking and correct stale H29 documentation.
- **The problem today:** `WeightedScoring.Route()` and `LeastLoaded.Route()` both use strict `>` / `<` comparison for argmax/argmin selection. When multiple instances have equal scores/loads, the lowest-index instance always wins. This deterministic bias seeds a prefix-affinity cold-start feedback loop (#565) and causes 12-21% worse TTFT P99 for LeastLoaded at low utilization (H4). Separately, H29's FINDINGS.md documents stale snapshot behavior that PR #467 invalidated (#566).
- **What this PR adds:**
  1. Random uniform tie-breaking in `WeightedScoring.Route()` using `PartitionedRNG`'s `SubsystemRouter` partition
  2. Random uniform tie-breaking in `LeastLoaded.Route()` using the same RNG partition
  3. Erratum in H29 FINDINGS.md documenting that PR #467 changed all signals to Periodic when `--snapshot-refresh-interval > 0`
- **Why this matters:** The positional bias is the most actionable root cause of the extreme latency variance across seeds reported in #562. The H29 erratum prevents users from relying on stale guidance.
- **Architecture:** `NewRoutingPolicy` factory gains a `*rand.Rand` parameter. `LeastLoaded` and `WeightedScoring` structs store the RNG. `ClusterSimulator` passes `rng.ForSubsystem(SubsystemRouter)`. Tests pass `nil` for backward compat (positional tie-breaking when rng is nil).
- **Source:** Issues #565, #566
- **Closes:** Fixes #565, fixes #566
- **Behavioral Contracts:** See Part 1, Section B

---

## Phase 0: Component Context

1. **Building block:** Routing policy layer (`sim/routing.go`)
2. **Adjacent blocks:** `ClusterSimulator` (caller via `RoutingDecisionEvent.Execute`), `PartitionedRNG` (RNG source), scoring pipeline (`routing_scorers.go`, `routing_prefix_scorer.go`)
3. **Invariants touched:** INV-6 (determinism — preserved via PartitionedRNG), INV-7 (signal freshness — H29 erratum updates documentation only)
4. **Construction Site Audit:**
   - `LeastLoaded{}` — constructed in `NewRoutingPolicy` (`routing.go:241`) only
   - `WeightedScoring{scorers, weights, observers}` — constructed in `NewRoutingPolicy` (`routing.go:257`) only
   - `NewRoutingPolicy(...)` — called in `cluster.go:78` (production) + ~40 test call sites

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds random tie-breaking to `WeightedScoring.Route()` and `LeastLoaded.Route()`. When multiple instances have equal composite scores (or equal effective load), the router selects uniformly at random among tied candidates using the `SubsystemRouter` RNG partition, preserving INV-6 determinism. The `NewRoutingPolicy` factory gains a `*rand.Rand` parameter; `nil` preserves positional tie-breaking for backward compatibility in tests. Separately, an erratum is added to `hypotheses/h29-snapshot-staleness/FINDINGS.md` noting that PR #467 changed the snapshot staleness model.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Random Tie-Breaking (WeightedScoring)
- GIVEN WeightedScoring with a non-nil RNG and N >= 2 instances with equal composite scores
- WHEN Route() is called repeatedly
- THEN each tied instance is selected with approximately equal frequency (uniform distribution)
- MECHANISM: Collect tied indices, pick rng.Intn(len(tied))
```

```
BC-2: Random Tie-Breaking (LeastLoaded)
- GIVEN LeastLoaded with a non-nil RNG and N >= 2 instances with equal EffectiveLoad
- WHEN Route() is called repeatedly
- THEN each tied instance is selected with approximately equal frequency (uniform distribution)
- MECHANISM: Collect tied indices, pick rng.Intn(len(tied))
```

```
BC-3: Determinism Preserved (INV-6)
- GIVEN the same seed and configuration
- WHEN the simulation is run twice
- THEN routing decisions are identical across runs
- MECHANISM: SubsystemRouter partition of PartitionedRNG produces identical sequence per seed
```

```
BC-4: Non-Tie Behavior Unchanged
- GIVEN instances with distinct scores/loads and a non-nil RNG
- WHEN Route() is called and then a tie-scenario Route() is called
- THEN the unique best instance is selected AND the subsequent tie-scenario produces the same sequence as if only the tie calls had been made (RNG state not advanced by non-tie calls)
```

```
BC-5: Nil RNG Backward Compatibility
- GIVEN a routing policy created with rng=nil
- WHEN Route() encounters a tie
- THEN the first occurrence (lowest index) wins (existing positional behavior)
- MECHANISM: nil guard in tie-breaking code
```

**Documentation contract:**

```
BC-6: H29 Erratum
- GIVEN H29 FINDINGS.md
- WHEN a user reads the document
- THEN they see a clear erratum noting PR #467 changed snapshot behavior
- MECHANISM: New "Erratum" section with explanation of what changed
```

### C) Component Interaction

```
ClusterSimulator
  │ creates PartitionedRNG(seed)
  │ calls rng.ForSubsystem(SubsystemRouter)
  │ passes *rand.Rand to NewRoutingPolicy(...)
  │
  ▼
NewRoutingPolicy(name, scorers, blockSize, rng)
  │ stores rng in LeastLoaded{rng} or WeightedScoring{..., rng}
  │
  ▼
RoutingDecisionEvent.Execute()
  │ calls routingPolicy.Route(req, state)
  │
  ▼
WeightedScoring.Route() / LeastLoaded.Route()
  │ computes scores/loads
  │ collects tied candidates
  │ if len(tied) > 1 && rng != nil: rng.Intn(len(tied))
  │ else: first occurrence
  │
  ▼
RoutingDecision (returned to cluster)
```

State ownership: `*rand.Rand` owned by the routing policy struct. No new state crosses boundaries.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #565 proposes fix for WeightedScoring and LeastLoaded | Also applies nil-guard for backward compat | ADDITION — minimizes test churn |
| #566 proposes erratum in FINDINGS.md | Also updates the stale code snippet in Critical Design Note section | ADDITION — the code snippet is the most misleading part |

### E) Review Guide

**Tricky part:** The RNG must only be consumed when there's an actual tie (>1 candidate), so non-tie scenarios don't shift the RNG state. This preserves existing behavior for differentiated workloads while fixing the equal-score case.

**Scrutinize:** BC-3 determinism test — ensure same seed produces identical decisions. The tie-breaking rewrite in existing tests that assert positional behavior.

**Safe to skim:** The ~40 mechanical `nil` additions to test call sites. The H29 erratum text.

**Known debt:** `AlwaysBusiest` also has positional tie-breaking but is a pathological test template — intentionally not fixed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action |
|------|--------|
| `sim/routing.go` | Modify: add `rng` field to `LeastLoaded` and `WeightedScoring`, update `NewRoutingPolicy` factory signature, implement random tie-breaking |
| `sim/routing_test.go` | Modify: update call sites with `nil` RNG, rewrite tie-breaking tests for random behavior, add BC-1 through BC-5 tests |
| `sim/cluster/cluster.go` | Modify: pass `rng.ForSubsystem(SubsystemRouter)` to `NewRoutingPolicy` |
| `sim/routing_scorers_test.go` | Modify: update `NewRoutingPolicy` call sites with `nil` |
| `sim/routing_prefix_scorer_test.go` | Modify: update `NewRoutingPolicy` call sites with `nil` |
| `sim/examples_test.go` | Modify: update `NewRoutingPolicy` call sites with `nil` |
| `hypotheses/h29-snapshot-staleness/FINDINGS.md` | Modify: add erratum section, update stale code snippet |

No new files created. No golden dataset regeneration needed (golden tests are single-instance, no routing policies involved).

### G) Task Breakdown

#### Task 1: Update `NewRoutingPolicy` factory and routing structs (BC-4, BC-5)

**TDD deviation note:** This task implements the factory signature change and tie-breaking logic before tests (Task 2) because changing the factory signature breaks all ~45 existing test call sites at compile time. Task 1 updates all call sites to restore compilation; Task 2 adds new behavioral tests for the tie-breaking behavior.

**Files:** `sim/routing.go`

**Step 1 — Implement factory change and tie-breaking logic:**

In `sim/routing.go`:

1. Add `import "math/rand"` to imports
2. Add `rng *rand.Rand` field to `LeastLoaded` struct
3. Add `rng *rand.Rand` field to `WeightedScoring` struct
4. Update `NewRoutingPolicy` signature to accept `rng *rand.Rand`
5. Pass `rng` to `LeastLoaded` and `WeightedScoring` construction
6. Rewrite `LeastLoaded.Route()` tie-breaking:
   ```go
   // Collect all indices with minimum load
   minLoad := snapshots[0].EffectiveLoad()
   for i := 1; i < len(snapshots); i++ {
       if load := snapshots[i].EffectiveLoad(); load < minLoad {
           minLoad = load
       }
   }
   var tied []int
   for i, snap := range snapshots {
       if snap.EffectiveLoad() == minLoad {
           tied = append(tied, i)
       }
   }
   idx := tied[0]
   if len(tied) > 1 && ll.rng != nil {
       idx = tied[ll.rng.Intn(len(tied))]
   }
   target := snapshots[idx]
   ```
7. Rewrite `WeightedScoring.Route()` tie-breaking:
   ```go
   bestScore := -1.0
   for _, snap := range snapshots {
       if scores[snap.ID] > bestScore {
           bestScore = scores[snap.ID]
       }
   }
   var tied []int
   for i, snap := range snapshots {
       if scores[snap.ID] == bestScore {
           tied = append(tied, i)
       }
   }
   bestIdx := tied[0]
   if len(tied) > 1 && ws.rng != nil {
       bestIdx = tied[ws.rng.Intn(len(tied))]
   }
   ```
8. Update doc comments:
   - `LeastLoaded`: change "Ties are broken by first occurrence in snapshot order (lowest index)." to "Ties are broken randomly when rng is non-nil; by first occurrence (lowest index) when rng is nil."
   - `WeightedScoring`: change "Higher scores are preferred. Ties broken by first occurrence in snapshot order." to "Higher scores are preferred. Ties broken randomly when rng is non-nil; by first occurrence (lowest index) when rng is nil."
   - `NewRoutingPolicy` godoc: add "The rng parameter enables random tie-breaking for least-loaded and weighted policies; nil preserves positional tie-breaking. Ignored by round-robin and always-busiest."

**Step 2 — Update all test call sites:**

Add `nil` as the last argument to every `NewRoutingPolicy(...)` call in:
- `sim/routing_test.go` (~25 sites)
- `sim/routing_scorers_test.go` (~2 sites)
- `sim/routing_prefix_scorer_test.go` (~10 sites)
- `sim/examples_test.go` (~6 sites)

**Step 3 — Update cluster.go production call site:**

In `sim/cluster/cluster.go`, change:
```go
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens),
```
to:
```go
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens, sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed)).ForSubsystem(sim.SubsystemRouter)),
```

Note: the `rng` PartitionedRNG is already created at `cluster.go:71` and assigned to `cs.rng`. Since we're inside the struct literal, we can't reference `cs.rng` yet. Extract the PartitionedRNG creation before the struct literal so both the struct field and the routing policy share the same instance:

```go
rng := sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed))
// ... in struct literal:
rng: rng,  // same object, moved out of literal
routingPolicy: sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem(sim.SubsystemRouter)),
```

**Step 4 — Build and run existing tests:**

```bash
cd .worktrees/pr-tiebreak-erratum
go build ./...
go test ./sim/... -count=1
```

Expected: build passes, all tests pass (existing tie-breaking tests now pass `nil` RNG from Step 2, preserving positional behavior per BC-5).

**Commit:** `refactor(sim): add RNG parameter to NewRoutingPolicy for tie-breaking`

#### Task 2: Write tie-breaking behavioral tests (BC-1, BC-2, BC-3, BC-4, BC-5)

**Files:** `sim/routing_test.go`

**Step 1 — Rewrite existing tie-breaking tests:**

Replace `TestLeastLoaded_LoadBasedSelection` subcase "tie broken by first occurrence (lowest index)" and "all instances equal load" with:

```go
// TestLeastLoaded_TieBreaking_Random verifies BC-2: random uniform tie-breaking.
func TestLeastLoaded_TieBreaking_Random(t *testing.T) {
    rng := rand.New(rand.NewSource(42))
    policy := NewRoutingPolicy("least-loaded", nil, 16, rng)
    snapshots := []RoutingSnapshot{
        {ID: "instance_0", QueueDepth: 5, BatchSize: 5},
        {ID: "instance_1", QueueDepth: 5, BatchSize: 5},
        {ID: "instance_2", QueueDepth: 5, BatchSize: 5},
    }

    counts := map[string]int{}
    N := 300
    for i := 0; i < N; i++ {
        req := &Request{ID: fmt.Sprintf("req%d", i)}
        decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
        counts[decision.TargetInstance]++
    }

    // Each instance should get roughly N/3 = 100 requests.
    // With 3 instances and 300 trials, expected = 100 per instance.
    // Allow ±50% tolerance (50-150 range) for random variation.
    for _, id := range []string{"instance_0", "instance_1", "instance_2"} {
        if counts[id] < 50 || counts[id] > 150 {
            t.Errorf("instance %s got %d/%d requests, expected ~100 (uniform)", id, counts[id], N)
        }
    }
}
```

Similar test for `WeightedScoring`:

```go
// TestWeightedScoring_TieBreaking_Random verifies BC-1: random uniform tie-breaking.
func TestWeightedScoring_TieBreaking_Random(t *testing.T) {
    rng := rand.New(rand.NewSource(42))
    policy := NewRoutingPolicy("weighted", []ScorerConfig{
        {Name: "queue-depth", Weight: 1.0},
    }, 16, rng)
    // All instances idle → queue-depth scores all 1.0 → tie
    snapshots := []RoutingSnapshot{
        {ID: "instance_0", QueueDepth: 0},
        {ID: "instance_1", QueueDepth: 0},
        {ID: "instance_2", QueueDepth: 0},
    }

    counts := map[string]int{}
    N := 300
    for i := 0; i < N; i++ {
        req := &Request{ID: fmt.Sprintf("req%d", i)}
        decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
        counts[decision.TargetInstance]++
    }

    for _, id := range []string{"instance_0", "instance_1", "instance_2"} {
        if counts[id] < 50 || counts[id] > 150 {
            t.Errorf("instance %s got %d/%d requests, expected ~100 (uniform)", id, counts[id], N)
        }
    }
}
```

**Step 2 — Add BC-3 determinism test:**

```go
// TestTieBreaking_Determinism verifies BC-3: same seed → same decisions.
func TestTieBreaking_Determinism(t *testing.T) {
    snapshots := []RoutingSnapshot{
        {ID: "a", QueueDepth: 0},
        {ID: "b", QueueDepth: 0},
    }

    for _, policyName := range []string{"least-loaded", "weighted"} {
        t.Run(policyName, func(t *testing.T) {
            // Two policies with same seed
            rng1 := rand.New(rand.NewSource(99))
            rng2 := rand.New(rand.NewSource(99))
            p1 := NewRoutingPolicy(policyName, nil, 16, rng1)
            p2 := NewRoutingPolicy(policyName, nil, 16, rng2)

            for i := 0; i < 50; i++ {
                req := &Request{ID: fmt.Sprintf("req%d", i)}
                d1 := p1.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
                d2 := p2.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
                if d1.TargetInstance != d2.TargetInstance {
                    t.Errorf("request %d: different decisions with same seed: %s vs %s",
                        i, d1.TargetInstance, d2.TargetInstance)
                }
            }
        })
    }
}
```

**Step 3 — Add BC-4 non-tie test (verifies RNG state not consumed):**

```go
// TestTieBreaking_NoTie_PreservesRNGState verifies BC-4: distinct scores → unique winner,
// RNG state not advanced (non-tie calls must not shift the RNG stream).
func TestTieBreaking_NoTie_PreservesRNGState(t *testing.T) {
    // Two RNGs with the same seed
    rng1 := rand.New(rand.NewSource(42))
    rng2 := rand.New(rand.NewSource(42))

    policy := NewRoutingPolicy("least-loaded", nil, 16, rng1)

    // Non-tie snapshots (unique minimum at instance_1)
    nonTieSnaps := []RoutingSnapshot{
        {ID: "instance_0", QueueDepth: 10},
        {ID: "instance_1", QueueDepth: 1},
        {ID: "instance_2", QueueDepth: 5},
    }

    // Make 50 non-tie routing calls — should NOT consume RNG
    for i := 0; i < 50; i++ {
        req := &Request{ID: fmt.Sprintf("req%d", i)}
        decision := policy.Route(req, &RouterState{Snapshots: nonTieSnaps, Clock: 1000})
        if decision.TargetInstance != "instance_1" {
            t.Fatalf("request %d: expected instance_1 (unique min), got %q", i, decision.TargetInstance)
        }
    }

    // Now verify rng1 and rng2 are in the same state:
    // draw from both and compare — if rng1 was consumed by non-tie calls, they'll differ
    val1 := rng1.Intn(1000)
    val2 := rng2.Intn(1000)
    if val1 != val2 {
        t.Errorf("RNG state diverged after non-tie calls: rng1=%d, rng2=%d (RNG consumed on non-tie)", val1, val2)
    }
}
```

**Step 4 — Add BC-5 nil RNG test:**

```go
// TestTieBreaking_NilRNG_Positional verifies BC-5: nil RNG → positional tie-breaking.
func TestTieBreaking_NilRNG_Positional(t *testing.T) {
    policy := NewRoutingPolicy("least-loaded", nil, 16, nil)
    snapshots := []RoutingSnapshot{
        {ID: "instance_0", QueueDepth: 5},
        {ID: "instance_1", QueueDepth: 5},
    }

    for i := 0; i < 10; i++ {
        req := &Request{ID: fmt.Sprintf("req%d", i)}
        decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})
        if decision.TargetInstance != "instance_0" {
            t.Errorf("nil RNG should use positional (first), got %q", decision.TargetInstance)
        }
    }
}
```

**Step 5 — Update existing tests that assert positional tie-breaking:**

In `TestLeastLoaded_LoadBasedSelection`, remove the subcases that test tie-breaking positional behavior (they're replaced by the new tests above). Keep the subcase "instance 1 has lowest load" (non-tie case).

In `TestWeightedScoring_AllIdle_NoDivisionByZero`, change the assertion from:
```go
if decision.TargetInstance != "instance_0" {
```
to verify that the decision is valid (any instance) and scores are finite:
```go
// All idle with nil RNG: positional tie-breaking → instance_0
```
(This test already passes nil RNG from Task 1.)

**Step 6 — Run tests:**

```bash
go test ./sim/... -count=1 -run "TieBreak|Routing|WeightedScoring|LeastLoaded"
```

Expected: all pass.

**Commit:** `test(sim): add behavioral tests for random tie-breaking (BC-1 through BC-5)`

#### Task 3: H29 Erratum (BC-6)

**Files:** `hypotheses/h29-snapshot-staleness/FINDINGS.md`

**Step 1 — Add erratum section after Metadata table:**

Insert after line 15 (after the Metadata table):

```markdown
## Erratum (2026-03-09)

**PR #467** (commit `9a603bc`) changed `sim/cluster/snapshot.go` so that when
`--snapshot-refresh-interval > 0`, **all three signals** (QueueDepth, BatchSize,
KVUtilization) use Periodic mode — not just KVUtilization. This invalidates the
key design note and several conclusions in the original findings:

- **"QueueDepth is always Immediate"** — no longer true when interval > 0. Only
  `InFlightRequests` (injected synchronously by `buildRouterState()`) remains
  unconditionally fresh.
- **"queue-depth scorer is NOT affected"** — partially invalidated. Queue-depth
  uses `EffectiveLoad() = QueueDepth + BatchSize + InFlightRequests`. When
  interval > 0, both `QueueDepth` and `BatchSize` are stale; only
  `InFlightRequests` remains fresh. The composite scorer resilience finding
  (+3.8% mean) may no longer hold at non-zero intervals.
- **"the default composite scorer is inherently resilient"** — this conclusion
  now applies **only to the default `--snapshot-refresh-interval 0`
  configuration**, where all signals are Immediate.

The experiment results themselves remain valid — they were run before #467 and
accurately describe the behavior at the time. INV-7 in
`docs/contributing/standards/invariants.md` was updated in PR #467 to reflect
the current behavior.

**Experiment 2 (negative control) invalidated:** H29's queue-depth:1 negative
control showed 0.0% change between fresh and stale configurations — but that
result depended on QueueDepth being Immediate. Post-#467, queue-depth:1 with
`--snapshot-refresh-interval > 0` would also exhibit staleness-driven
degradation, since both QueueDepth and BatchSize (which feed `EffectiveLoad()`)
are now Periodic.
```

**Step 2 — Update the stale code snippet in "Critical Design Note" section:**

Replace the code block starting at line 27 with the current `newObservabilityConfig` implementation:

```go
func newObservabilityConfig(refreshInterval int64) ObservabilityConfig {
    if refreshInterval <= 0 {
        return DefaultObservabilityConfig()  // all Immediate
    }
    periodic := FieldConfig{Mode: Periodic, Interval: refreshInterval}
    return ObservabilityConfig{
        QueueDepth:    periodic,    // Periodic when interval > 0 (changed in PR #467)
        BatchSize:     periodic,    // Periodic when interval > 0 (changed in PR #467)
        KVUtilization: periodic,
    }
}
```

And update the surrounding explanation to note:

```
When `--snapshot-refresh-interval > 0`, all three Prometheus-sourced signals become Periodic.
Only `InFlightRequests` (gateway-local counter) remains synchronous.
```

**Step 3 — Run lint:**

```bash
# No Go files changed in this task, so no lint needed
```

**Commit:** `docs: add erratum to H29 FINDINGS.md for PR #467 snapshot staleness change`

#### Task 4: Full verification gate

**Step 1 — Build:**
```bash
go build ./...
```

**Step 2 — Run all tests:**
```bash
go test ./... -count=1
```

**Step 3 — Lint:**
```bash
golangci-lint run ./...
```

**Step 4 — Git status:**
```bash
git status
```

Expected: all pass, working tree shows planned modifications only.

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Behavioral | `TestWeightedScoring_TieBreaking_Random` |
| BC-2 | Task 2 | Behavioral | `TestLeastLoaded_TieBreaking_Random` |
| BC-3 | Task 2 | Invariant | `TestTieBreaking_Determinism` |
| BC-4 | Task 2 | Invariant | `TestTieBreaking_NoTie_PreservesRNGState` |
| BC-5 | Task 2 | Behavioral | `TestTieBreaking_NilRNG_Positional` |
| BC-6 | Task 3 | Manual | Read erratum in FINDINGS.md |

Key invariants:
- **INV-6 (Determinism):** BC-3 test verifies same seed → same decisions
- **INV-1 (Conservation):** Existing cluster conservation tests pass (routing change doesn't affect request accounting)

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| RNG consumed on non-tie changes output determinism | Low | High | Guard: only consume RNG when len(tied) > 1 | Task 1, BC-4 test |
| Floating-point equality comparison misses near-ties | Low | Medium | Scores are computed identically for all instances with equal state; exact float equality is correct here | Task 1 |
| Test randomness makes tests flaky | Low | Low | 300 trials with ±50% tolerance; seed is fixed | Task 2 |
| H29 erratum text inaccurate | Low | Low | Cross-reference with current snapshot.go code | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (AlwaysBusiest intentionally not fixed — pathological template)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: uses standard `math/rand` and existing test patterns
- [x] CLAUDE.md update: N/A — CLAUDE.md doesn't describe tie-breaking behavior; doc comments on structs are updated in Task 1
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: INV-7 already updated; H29 erratum is the stale working copy
- [x] Deviation log reviewed
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 before Task 2)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration not needed (single-instance only)
- [x] Construction site audit: `NewRoutingPolicy` is the sole construction site for both structs

**Antipattern rules:**
- [x] R1: No silent data loss
- [x] R2: No map iteration ordering issues
- [x] R3: No new numeric parameters
- [x] R4: Construction site audit — `NewRoutingPolicy` is sole constructor, updated
- [x] R5: N/A (no resource allocation)
- [x] R6: No Fatalf in sim/
- [x] R7: BC-3 invariant test alongside behavioral tests
- [x] R8: No exported maps
- [x] R9-R10: N/A (no YAML changes)
- [x] R11: N/A (no division)
- [x] R12: Golden dataset unaffected
- [x] R13-R23: N/A or checked

---

## Appendix: File-Level Implementation Details

### File: `sim/routing.go`

**Purpose:** Add RNG-based random tie-breaking to `LeastLoaded` and `WeightedScoring`.

**Changes:**
1. Add `"math/rand"` import
2. `LeastLoaded` struct: add `rng *rand.Rand` field
3. `LeastLoaded.Route()`: two-pass (find min, collect tied, random pick)
4. `LeastLoaded` doc comment: update tie-breaking description
5. `WeightedScoring` struct: add `rng *rand.Rand` field
6. `WeightedScoring.Route()`: two-pass (find max, collect tied, random pick)
7. `WeightedScoring` doc comment: update tie-breaking description
8. `NewRoutingPolicy`: add `rng *rand.Rand` parameter, pass to LeastLoaded{rng: rng} and WeightedScoring{..., rng: rng}

**RNG usage:** `SubsystemRouter` from `PartitionedRNG`. Consumed only on ties (len(tied) > 1).

### File: `sim/cluster/cluster.go`

**Purpose:** Pass router RNG to `NewRoutingPolicy`.

**Changes:** Extract `rng` creation before struct literal, pass `rng.ForSubsystem(sim.SubsystemRouter)` to factory.

### File: `sim/routing_test.go`

**Purpose:** Update all `NewRoutingPolicy` call sites with `nil` RNG; add BC-1 through BC-5 tests.

### File: `sim/routing_scorers_test.go`, `sim/routing_prefix_scorer_test.go`, `sim/examples_test.go`

**Purpose:** Update `NewRoutingPolicy` call sites with `nil` RNG parameter.

### File: `hypotheses/h29-snapshot-staleness/FINDINGS.md`

**Purpose:** Add erratum section, update stale code snippet and explanation.
