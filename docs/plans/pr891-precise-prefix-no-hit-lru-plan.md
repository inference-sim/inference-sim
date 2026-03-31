# Precise Prefix Cache + No-Hit LRU Scorers Implementation Plan

**Goal:** Add two new scorers to the weighted routing pipeline — `precise-prefix-cache` (ground-truth KV cache prefix matching) and `no-hit-lru` (cold request distribution) — matching llm-d production implementation.

**The problem today:** BLIS's existing `prefix-affinity` scorer uses a router-side approximate index that diverges from llm-d's production `PrecisePrefixCacheScorer`, which queries actual instance KV cache state. There is no cold-request distribution mechanism (`NoHitLRU`) for requests with zero cache hits.

**What this PR adds:**
1. `precise-prefix-cache` scorer: queries actual instance KV cache via `CacheQueryFn`, min-max normalized across candidates (ground truth, no approximation).
2. `no-hit-lru` scorer: distributes cold requests (no cache hits on any candidate) to least-recently-used endpoints. Warm requests score 0.5 (neutral).
3. `CacheQueryFn` type and cluster-level wiring that maps instance IDs to their `GetCachedBlocks` functions.

**Why this matters:** Enables high-fidelity simulation of llm-d's precise prefix and cold-start routing, which is the production-grade scoring pipeline.

**Architecture:** Both scorers are policy templates behind the existing `scorerFunc`/`observerFunc` interface. `CacheQueryFn` bridges the cluster layer (`sim/cluster/`) to the scoring layer (`sim/`) without import cycles. The `no-hit-lru` scorer maintains a doubly-linked LRU via closure state shared between scorer and observer.

**Source:** [Issue #883](https://github.com/inference-sim/inference-sim/issues/883)
**Closes:** Fixes #883
**Behavioral Contracts:** See Section B below.

---

## Phase 0: Component Context

1. **Building block:** Two new scorers in the weighted routing scorer pipeline (`sim/routing_scorers.go` factory).
2. **Adjacent blocks:** `WeightedScoring` (consumer), `InstanceSimulator.KVCache` (data source via `CacheQueryFn`), `ClusterSimulator` (wiring).
3. **Invariants touched:** INV-7 (signal freshness — both scorers bypass snapshot staleness, reading ground truth synchronously).
4. **Construction site audit:**
   - `CacheQueryFn` is a new type (no existing construction sites).
   - `NewRoutingPolicy` signature extended with variadic `cacheQueryFn ...CacheQueryFn` — backward compatible, no existing call sites broken.
   - `newScorerWithObserver` signature extended with `cacheQueryFn CacheQueryFn` — called only from `NewRoutingPolicy`, single call site.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds two scorers (`precise-prefix-cache`, `no-hit-lru`) that together form llm-d's production precise-prefix scoring pipeline. The `precise-prefix-cache` scorer replaces the approximate router-side prefix index with direct KV cache queries. The `no-hit-lru` scorer distributes cold requests (zero cache hits) to least-recently-used endpoints, preventing hot-spotting during cold starts. Both scorers are stateless/stateful policy templates behind existing interfaces. The `CacheQueryFn` bridge type enables cluster-to-scorer data flow without import cycles.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Precise prefix — min-max normalization
- GIVEN a request with input tokens and multiple candidate instances with varying cached block counts
- WHEN the precise-prefix-cache scorer evaluates the request
- THEN the instance with the most cached blocks scores 1.0, the least scores 0.0, and intermediate instances are linearly interpolated

BC-2: Precise prefix — all-equal normalization
- GIVEN all candidate instances have the same number of cached blocks (including zero)
- WHEN the precise-prefix-cache scorer evaluates the request
- THEN all instances score 1.0

BC-3: No-hit LRU — warm request neutrality
- GIVEN any candidate instance has cached blocks for the request (warm request)
- WHEN the no-hit-lru scorer evaluates the request
- THEN all instances score 0.5 (neutral, deferring to other scorers)

BC-4: No-hit LRU — cold request never-used preference
- GIVEN no candidate instance has cached blocks (cold request) and some candidates have never been used for cold routing
- WHEN the no-hit-lru scorer evaluates the request
- THEN never-used candidates score 1.0 and previously-used candidates score lower

BC-5: No-hit LRU — cold request LRU differentiation
- GIVEN no candidate instance has cached blocks (cold request) and all candidates have been used
- WHEN the no-hit-lru scorer evaluates the request
- THEN the least-recently-used candidate scores highest, the most-recently-used scores lowest, and all scores are in [0, 1]

BC-6: No-hit LRU — LRU state update on cold routing only
- GIVEN a routing decision has been made
- WHEN the observer is called
- THEN the LRU is updated only for cold requests (warm requests do not modify LRU state)

BC-7: No-hit LRU — snapshot-filtered ranking
- GIVEN the LRU contains entries for instances not in the current routing snapshots
- WHEN the scorer ranks candidates
- THEN only snapshot-visible instances are assigned ranks (non-snapshot entries do not inflate ranks)

**Negative contracts:**

BC-8: Nil cacheQueryFn panic
- GIVEN cacheQueryFn is nil
- WHEN either scorer factory is called
- THEN the factory panics with a descriptive message

BC-9: Score bounds
- GIVEN any combination of inputs
- WHEN either scorer produces scores
- THEN all scores are in [0, 1] (no negative scores, no scores > 1)

### C) Component Interaction

```
ClusterSimulator
  |-- builds CacheQueryFn map: instanceID -> inst.GetCachedBlockCount
  |-- passes CacheQueryFn to NewRoutingPolicy(..., cqf)
  |
  v
NewRoutingPolicy("weighted", scorerConfigs, blockSize, rng, cqf)
  |-- newScorerWithObserver("precise-prefix-cache", blockSize, cqf)
  |     |-> newPrecisePrefixCacheScorer(cqf) -> (scorerFunc, nil)
  |-- newScorerWithObserver("no-hit-lru", blockSize, cqf)
  |     |-> newNoHitLRUScorer(cqf) -> (scorerFunc, observerFunc)
  |
  v
WeightedScoring.Route(req, state)
  |-- calls each scorerFunc(req, snapshots)
  |     |-- precise-prefix-cache: queries cqf[snap.ID](req.InputTokens)
  |     |-- no-hit-lru: checks warm/cold via cqf, scores by LRU position
  |-- argmax selection
  |-- calls each observerFunc(req, targetInstance)
  |     |-- no-hit-lru observer: updates LRU if cold request
```

Data ownership:
- `CacheQueryFn`: created by cluster, consumed by scorers (read-only bridge)
- `noHitLRU`: owned by scorer closure, mutated by observer closure (safe: DES single-threaded)
- `lastReqID/lastReqWarm`: shared between scorer and observer via closure (safe: sequential call order)

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue #883 mentions "PrecisePrefixCacheScorer" | Named `precise-prefix-cache` in kebab-case | CLARIFICATION — follows existing naming convention (e.g., `prefix-affinity`, `queue-depth`) |
| Issue #883 mentions "NoHitLRU" | Delivered in same PR as `precise-prefix-cache` | ADDITION — `no-hit-lru` solves the cold-start problem that `precise-prefix-cache` creates (they are a design unit in llm-d) |
| llm-d uses gRPC for cache queries | Uses direct function call via `CacheQueryFn` | SIMPLIFICATION — DES is single-threaded, no need for RPC simulation |

### E) Review Guide

**Scrutinize:** The `noHitLRU.rank()` method's snapshot filtering (BC-7) and the scoring formula's denominator choice (`total` not `total-1`) for BC-5/BC-9. The closure-based state sharing between scorer and observer (`lastReqID`/`lastReqWarm`) is safe due to DES single-threading but non-obvious.

**Safe to skim:** `newPrecisePrefixCacheScorer` — straightforward min-max normalization identical to existing `scoreQueueDepth`. The `noHitLRU` doubly-linked list is textbook.

**Known debt:** None. Both scorers are complete relative to llm-d's production implementation.

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files created:
- `sim/routing_precise_prefix_scorer.go` — precise prefix cache scorer (BC-1, BC-2, BC-8)
- `sim/routing_precise_prefix_scorer_test.go` — 4 tests
- `sim/routing_no_hit_lru_scorer.go` — no-hit LRU scorer + LRU data structure (BC-3 through BC-9)
- `sim/routing_no_hit_lru_scorer_test.go` — 7 tests

Files modified:
- `sim/routing_scorers.go` — `CacheQueryFn` type, registration in `validScorerNames` and `newScorerWithObserver`
- `sim/routing.go` — variadic `cacheQueryFn` parameter on `NewRoutingPolicy`, pass-through to factory
- `sim/cluster/instance.go` — `GetCachedBlockCount` public accessor
- `sim/cluster/cluster.go` — build `CacheQueryFn` map, pass to all `NewRoutingPolicy` calls
- 8 documentation files (CLAUDE.md, README.md, cmd/root.go, docs/*, examples/*)

### G) Task Breakdown

**Task 1: CacheQueryFn type and scorer registration (BC-8)**
- Files: modify `sim/routing_scorers.go`, modify `sim/routing.go`
- Add `CacheQueryFn` type, register both scorer names, extend `newScorerWithObserver` signature
- Extend `NewRoutingPolicy` with variadic `cacheQueryFn` parameter

**Task 2: Precise prefix cache scorer (BC-1, BC-2, BC-8, BC-9)**
- Files: create `sim/routing_precise_prefix_scorer.go`, create `sim/routing_precise_prefix_scorer_test.go`
- Implement min-max normalized scorer with nil-panic guard
- Tests: min-max normalization, all-equal, zero-cached, nil-panic

**Task 3: No-hit LRU scorer (BC-3 through BC-9)**
- Files: create `sim/routing_no_hit_lru_scorer.go`, create `sim/routing_no_hit_lru_scorer_test.go`
- Implement LRU data structure, scorer with warm/cold detection, observer with LRU update
- Tests: warm neutrality, cold never-used preference, single endpoint, all-endpoints-used LRU differentiation, nil-panic, rank filters non-snapshot, combined with precise prefix

**Task 4: Cluster integration (all BCs)**
- Files: modify `sim/cluster/instance.go`, modify `sim/cluster/cluster.go`
- Add `GetCachedBlockCount` accessor, build `CacheQueryFn` map, wire to all `NewRoutingPolicy` calls

**Task 5: Documentation update**
- Files: 8 documentation files
- Update scorer tables, file trees, CLI help text, examples

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | 2 | Unit | TestPrecisePrefixCache_MinMaxNormalization |
| BC-2 | 2 | Unit | TestPrecisePrefixCache_AllEqual_AllScoreOne |
| BC-2 | 2 | Unit | TestPrecisePrefixCache_ZeroCachedBlocks |
| BC-3 | 3 | Unit | TestNoHitLRU_WarmRequest_NeutralScores |
| BC-4 | 3 | Unit | TestNoHitLRU_ColdRequest_NeverUsedEndpointsPreferred |
| BC-5 | 3 | Unit | TestNoHitLRU_AllEndpointsUsed_LRUDifferentiates |
| BC-6 | 3 | Unit | (implicit in BC-4 test — second request verifies LRU update) |
| BC-7 | 3 | Unit | TestNoHitLRU_RankFiltersNonSnapshotInstances |
| BC-8 | 2,3 | Unit | TestPrecisePrefixCache_NilCacheQueryFn_Panics, TestNoHitLRU_NilCacheQueryFn_Panics |
| BC-9 | 3 | Unit | TestNoHitLRU_AllEndpointsUsed_LRUDifferentiates (score bounds check) |
| All | 3 | Unit | TestNoHitLRU_SingleEndpoint_ScoresOne |
| Composition | 3 | Unit | TestNoHitLRU_CombinedWithPrecisePrefix |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| LRU rank inflation from non-snapshot entries | Medium | High (wrong routing) | `rank()` filters to snapshot set; dedicated test (BC-7) | 3 |
| Formula produces negative scores when all endpoints used | Medium | High (clamped to 0, loses differentiation) | Use `total` denominator, not `total-1`; dedicated test (BC-5) | 3 |
| Observer updates LRU for warm requests | Low | Medium (LRU drift) | Shared `lastReqWarm` flag; warm neutrality test (BC-3) | 3 |
| Backward compatibility break from `NewRoutingPolicy` signature change | High | High (compile errors) | Variadic `cacheQueryFn ...CacheQueryFn` preserves all existing call sites | 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `CacheQueryFn` is the minimal bridge type
- [x] No feature creep beyond PR scope — exactly two scorers per issue #883
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact — LRU is closure-scoped
- [x] All new code passes golangci-lint
- [x] Shared test helpers used (`makeCacheQueryFn` in test file)
- [x] CLAUDE.md updated with new scorers and file references
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY — all docs updated (routing.md, architecture.md, configuration.md, project-structure.md, README.md, examples)
- [x] No unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1 before 2,3; 4 after 2,3)
- [x] All contracts mapped to tasks
- [x] Construction site audit completed — `NewRoutingPolicy` is the only construction site for scorer pipeline

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` dropping data — scorer returns empty map for nil req/empty snapshots (documented behavior)
- [x] R2: Map keys sorted before float accumulation — scorer iterates `snapshots` slice (ordered), not map
- [x] R3: Every new numeric parameter validated — `cacheQueryFn != nil` checked with panic
- [x] R4: Construction site audit — `NewRoutingPolicy` variadic parameter, single call site for `newScorerWithObserver`
- [x] R5: No resource allocation loops
- [x] R6: No `logrus.Fatalf` in `sim/` — uses `panic` for factory validation
- [x] R7: No golden tests added (all behavioral/invariant tests)
- [x] R8: No exported mutable maps — `validScorerNames` is unexported, `CacheQueryFn` is a type not a variable
- [x] R9: No YAML fields added
- [x] R10: No YAML parsing added
- [x] R11: Division guarded — `maxRaw == minRaw` check in precise-prefix, `len(snapshots) == 1` short-circuit and `total` denominator (non-zero by construction) in no-hit-lru
- [x] R12: No golden dataset changes
- [x] R13: N/A — no new interfaces, using existing `scorerFunc`/`observerFunc`
- [x] R14: Scorers are single-concern (scoring only, no scheduling/metrics)
- [x] R15: No stale PR references
- [x] R16: No config params added (scorers configured via existing `--routing-scorers` flag)
- [x] R17: Signal freshness documented — both scorers have `Signal freshness (R17, INV-7)` blocks
- [x] R18: No CLI flag defaults modified
- [x] R19: No retry loops
- [x] R20: N/A — no detectors/analyzers
- [x] R21: No `range` over mutable slices — LRU walked via pointer chain, not slice
- [x] R22: No pre-check/operation pairs
- [x] R23: Both scorers handle empty snapshots identically (return empty map)
