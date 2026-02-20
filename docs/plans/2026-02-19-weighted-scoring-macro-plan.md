# BLIS Composable Scorer Framework: Macro-Level Implementation Plan

**Date:** 2026-02-19
**Revision:** v1.1
**Status:** Complete (PR17 merged, PR18 in review)
**Target:** Composable multi-scorer routing within `weighted` policy
**Based on:** [Design Document](2026-02-19-weighted-scoring-evolution-design.md)
**Closes:** #229, #230

---

## A) Executive Summary

The `weighted` routing policy's cache dimension measures capacity headroom, not prefix affinity — making weight tuning a feedback latency artifact rather than a meaningful tradeoff (#229). This plan evolves `weighted` into a composable multi-scorer pipeline matching llm-d's Endpoint Picker architecture.

**What changes:** The monolithic two-dimension `WeightedScoring` is replaced by a pipeline of independent scorers (prefix-affinity, queue-depth, kv-utilization, load-balance), each returning [0,1] scores combined with configurable weights.

**Implementation:** 2 PRs, sequentially dependent.
- **PR 17** — Scorer framework + 3 stateless scorers (queue-depth, kv-utilization, load-balance). Immediately exercisable; fixes the broken cache dimension.
- **PR 18** — Prefix-affinity scorer with router-side approximate prefix cache index. Enables llm-d default profile and meaningful prefix workload experiments.

**Key constraint:** The frozen `RoutingPolicy` interface is not modified. The scorer framework is internal to the `weighted` policy.

---

## B) Repository Recon Summary

### B.1 Affected Packages and Files

| Package | Files Affected | Change Type |
|---|---|---|
| `sim/` | `routing.go` | Major — `WeightedScoring` refactored to scorer pipeline |
| `sim/` | `bundle.go` | Moderate — `RoutingConfig` YAML schema changes, scorer name validation |
| `sim/` | new file(s) for scorers | New — scorer implementations |
| `sim/` | new file for prefix cache index | New (PR 18) — router-side approximate cache |
| `sim/cluster/` | `deployment.go` | Minor — `RoutingCacheWeight`/`RoutingLoadWeight` fields replaced |
| `sim/cluster/` | `cluster.go` | Minor — `NewRoutingPolicy` call signature changes |
| `cmd/` | `root.go` | Moderate — CLI flags replaced (`--routing-scorers` replaces two weight flags) |
| `examples/` | `weighted-routing.yaml` | Rewritten — new YAML schema + documentation |
| `examples/` | `policy-config.yaml` | Minor — update weighted routing comments |

### B.2 Core Data Structures (Frozen)

This plan depends on three frozen types (see Section G for exact signatures):

- **RoutingPolicy** (`sim/routing.go:43-45`) — single-method interface; the scorer framework lives *inside* the `weighted` implementation of this interface, not alongside it.
- **RoutingSnapshot** (`sim/routing.go:10-18`) — immutable value type providing per-instance queue depth, batch size, KV utilization, free KV blocks, cache hit rate, and pending requests. Scorers read these fields to compute their scores.
- **RouterState** (`sim/router_state.go`) — bridge type carrying snapshots and clock to cluster-level policies.
- **RoutingDecision** (`sim/routing.go:28-39`) — return type from routing; `Scores` map carries composite scores consumed by counterfactual analysis.

The scorer framework is internal to the `weighted` routing policy and introduces no new public interfaces.

### B.3 Current WeightedScoring Implementation (Confirmed Facts)

- `sim/routing.go:116-167`: Two scoring dimensions hardcoded — `cacheScore = FreeKVBlocks / maxFreeKVBlocks` and `loadScore = 1 / (1 + effectiveLoad)`. Composite: `cacheScore * cacheWeight + loadScore * loadWeight`.
- `sim/routing.go:247-274`: `NewRoutingPolicy` factory — `case "weighted"` normalizes weights to sum to 1.0 and constructs `WeightedScoring`.
- `sim/bundle.go:32-36`: `RoutingConfig` with `CacheWeight *float64` and `LoadWeight *float64` fields.
- `cmd/root.go:70-71`: CLI flags `routingCacheWeight` (default 0.6) and `routingLoadWeight` (default 0.4).
- `cmd/root.go:563-564`: Flag registration for `--routing-cache-weight` and `--routing-load-weight`.
- `sim/cluster/deployment.go:34-35`: `DeploymentConfig` fields `RoutingCacheWeight` and `RoutingLoadWeight`.
- `sim/cluster/cluster.go:85`: Factory call `sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingCacheWeight, config.RoutingLoadWeight)`.

### B.4 Counterfactual Analysis Integration (Confirmed Fact)

`sim/cluster/counterfactual.go:32-96`: `computeCounterfactual` uses `RoutingDecision.Scores` map directly. Since the refactored `WeightedScoring` will continue to populate `Scores` with composite values, counterfactual analysis requires no changes.

### B.5 Existing Tests (Confirmed Facts)

- `sim/routing_test.go`: Unit tests for all routing policies including `WeightedScoring`
- `sim/bundle_test.go`: YAML parsing and validation tests for `RoutingConfig`
- `sim/cluster/cluster_test.go:765-766`: Integration tests that set `RoutingCacheWeight`/`RoutingLoadWeight`
- `sim/cluster/pending_requests_test.go:25-26`: Tests using weighted routing
- `sim/cluster/cluster_trace_test.go:102-103`: Trace tests with weighted routing
- Golden dataset tests: Exercise multiple routing policies; `weighted` outputs will change (expected)

### B.6 CLI Entrypoints and Flag Surface

**Current CLI** (`cmd/root.go`): `simulation_worker run` with `--routing-policy` (string), `--routing-cache-weight` (float64, default 0.6), `--routing-load-weight` (float64, default 0.4). Policy config YAML via `--policy-config` overrides CLI flags.

**After this plan:** `--routing-cache-weight` and `--routing-load-weight` removed. Replaced by `--routing-scorers` (string, comma-separated `name:weight` pairs).

### B.7 Areas of Coupling

- `NewRoutingPolicy` factory signature is coupled to `DeploymentConfig` and `ClusterSimulator` construction — changing the factory requires updating both call sites.
- `RoutingConfig` YAML schema is coupled to `PolicyBundle.Validate()` — adding scorer fields requires validation updates in the same file.
- No tight coupling or fragility beyond these expected integration points.

### B.8 Open Uncertainties

| Uncertainty | Impact | Resolution |
|---|---|---|
| Router-side prefix cache approximation quality for short prefixes | May produce zero scores for <1 block prefixes | Validated at PR 18 merge (D-2 gate in risk register) |

---

## C) High-Level Objectives + Non-Goals

### Objectives

1. **Replace the broken cache dimension** with meaningful, independent scoring dimensions
2. **Enable llm-d scheduling profile experimentation** by matching the multi-scorer architecture
3. **Provide router-side approximate prefix cache** matching llm-d's information model (not idealized direct query)
4. **Support configurable scorer composition** via YAML and CLI for rapid experimentation
5. **Demonstrate real cache/load tradeoff** with prefix-heavy workloads (unblocking #230)

### Analysis Questions This Feature Helps Answer

1. What is the optimal weight combination for prefix-affinity vs. load-balance vs. KV-utilization for a given workload?
2. How does router-side cache approximation quality degrade under high request rates and prefix diversity?
3. How do llm-d scheduling profiles perform under controlled simulation conditions?
4. How does normalization strategy (min-max vs. inverse transform) affect routing fairness?

### Non-Goals

- Modifying the frozen `RoutingPolicy` interface
- Adding predicted-latency scoring (requires latency model interface extraction)
- Adding LoRA-affinity scoring (multi-LoRA not modeled)
- Picker strategies (weighted-random, power-of-two-choices) — argmax only
- Changes to non-weighted routing policies

### Performance Constraints

- Scorer pipeline must add negligible overhead to routing decisions. Scoring is O(num_instances × num_scorers) per request — with 4 scorers and 16 instances, this is 64 score evaluations, each a simple arithmetic operation. No measurable impact on simulation throughput.
- Router-side prefix cache index lookup is O(num_blocks_per_request) per scorer call. With max 256 blocks (llm-d default), this is bounded.
- LRU eviction is O(1) amortized per access (standard doubly-linked-list + hashmap design).

### Backward Compatibility

- **No CLI backward compatibility**: `--routing-cache-weight` and `--routing-load-weight` are removed
- **No YAML backward compatibility**: `cache_weight` and `load_weight` fields removed; strict parsing ensures old files produce errors
- **Golden dataset stability**: Non-weighted routing policy outputs must remain byte-identical; `weighted` output changes are expected and documented

### Model Scoping Table

| Component | Modeled | Simplified | Omitted | Justification |
|---|---|---|---|---|
| Prefix match ratio | Proportional block-level matching | Router-side LRU approximation (not exact server-side state) | Exact cache query | Studies information asymmetry between router and actual cache; this is what production systems face |
| Queue depth normalization | Min-max normalization (llm-d match) | — | Exponential decay, time-weighted averages | Min-max is simple and proven in llm-d; more sophisticated normalization adds complexity without answering different questions |
| KV utilization scoring | 1 - utilization (llm-d match) | — | Per-layer breakdown | Per-layer breakdown requires KVStore interface changes; aggregate sufficient for routing |
| Score aggregation | Weighted sum with [0,1] clamping | — | Multiplicative, cascaded filtering | Weighted sum is the standard in both llm-d and k8s scheduling; multiplicative scoring creates threshold effects that complicate weight interpretation |
| Load balance scoring | Inverse transform 1/(1+load) | — | — | BLIS-native alternative to min-max; enables normalization strategy comparison |

---

## D) Concept Model

### Building Blocks (4 components, all within Router module)

**1. Scorer Pipeline** — Orchestrates scoring and aggregation within `weighted` routing.
- OBSERVES: Request, RouterState, scorer weights
- CONTROLS: Final composite score per instance, target selection (argmax)
- OWNS: Ordered list of weighted scorers
- INVARIANTS: INV-1 (scores clamped to [0,1]), INV-6 (weights normalized)
- EVENTS: None (synchronous within RoutingDecisionEvent)
- FRICTION: 0 files to add/remove (configuration-driven)

**2. Stateless Scorers** (queue-depth, kv-utilization, load-balance) — Pure scoring functions.
- OBSERVES: Per-instance snapshots from RouterState
- CONTROLS: Per-instance score in [0,1]
- OWNS: No state
- INVARIANTS: INV-1, INV-2 (score for every instance), INV-3 (deterministic)
- EVENTS: None
- FRICTION: 1 file + 1 registration = 2 touch points per new scorer

**3. Prefix-Affinity Scorer** — Stateful scorer with router-side cache.
- OBSERVES: Request token IDs, RouterState snapshots
- CONTROLS: Per-instance prefix match ratio in [0,1]
- OWNS: Router-side prefix cache index (block hash → instance set with LRU)
- INVARIANTS: INV-1, INV-2, INV-3, INV-4 (observer consistency), INV-7 (bounded growth)
- EVENTS: None (observer hook within same event)
- FRICTION: 2 touch points

**4. Scorer Configuration** — YAML/CLI parsing and validation for scorer weights.
- OBSERVES: PolicyBundle YAML, CLI flags
- CONTROLS: Scorer instantiation and weight assignment
- OWNS: Validated scorer config
- INVARIANTS: INV-6 (weight normalization), reject NaN/Inf/negative
- EVENTS: None (startup-time only)
- FRICTION: 0 files (extends existing bundle.go validation)

### Interaction Model

```
CLI/YAML → Config → WeightedScoring{scorers[], weights[]}
Request → Route() → [Score() per scorer] → Σ clamp(s_i)×w_i → Argmax → Decision → [Observe() per observer]
```

### System Invariants

- INV-1 through INV-7 as defined in design document
- Existing invariants unchanged: request conservation, clock monotonicity, KV cache conservation, causality, determinism
- State vs. statistics: prefix cache index is simulation state (evolves routing decisions); per-scorer scores in `RoutingDecision.Scores` are statistics (output for analysis)

### Extension Points

- **Adding a new scorer:** Implement the scorer behavioral contract (name, score function, optional observer). Register in scorer factory. Extension friction: 2 files.
- **Default behavior:** `weighted` without explicit scorers uses `queue-depth:2, kv-utilization:2, load-balance:1` (PR 17) evolving to `prefix-affinity:3, queue-depth:2, kv-utilization:2` (PR 18).
- **First non-default:** `--routing-scorers "load-balance:1"` (single-scorer equivalent to `least-loaded`).

### State Ownership Map

| Mutable State | Owner |
|---|---|
| Scorer weight config | `WeightedScoring` instance (immutable after construction) |
| Prefix cache index | Prefix-affinity scorer (exclusive) |
| Intermediate scores | `Route()` call stack (ephemeral) |

No shared mutable state across module boundaries.

### Real-System Correspondence

| Building Block | llm-d EPP | SGLang | vLLM |
|---|---|---|---|
| Scorer Pipeline | Scheduling profile with weighted scorers | Cache-aware router | N/A |
| Prefix-Affinity Scorer | `prefix-cache-scorer` | RadixCache routing | N/A |
| Queue-Depth Scorer | `queue-scorer` | Queue threshold | N/A |
| KV-Utilization Scorer | `kv-cache-utilization-scorer` | KV threshold | N/A |

---

## E) Architectural Risk Register

| Decision | Assumption | Validation Method | Cost if Wrong | Gate |
|---|---|---|---|---|
| Router-side approximate prefix cache (D-1) | Approximation quality is sufficient for meaningful routing differentiation | Run `servegen-language.yaml` (70% shared prefix, 500 requests, 4 instances) with `prefix-affinity:3,queue-depth:2`. **Success:** at least 2 instances have mean prefix-affinity score > 0.3; distribution std_dev differs by >5× between `prefix-affinity:5,queue-depth:1` and `prefix-affinity:0,queue-depth:1`. **Abort:** add oracle scorer that queries instance KVCache directly. | PR 18 rework (add oracle scorer fallback) — 1 PR | Before PR 18 merge |
| Scorer as internal abstraction (D-3) | Scorers are only needed within `weighted` routing | Grep for scoring patterns in admission policy and autoscaler code paths. **Success:** no scoring logic found outside routing. **Abort:** extract scorer interface to `sim/` as a top-level type. | Extract to `sim/` top-level — 0.5 PR refactor | Before PR15 (adapters) |
| Hierarchical block hashing (D-2) | Block-level matching produces proportional scores for typical prefix lengths | Test with requests sharing 50 tokens (3 blocks at block_size=16) and 500 tokens (31 blocks). **Success:** 500-token match scores ≥ 5× higher than 50-token match scores for the same instance. **Abort:** fall back to binary match (current PrefixAffinity behavior) — zero additional cost since that policy already exists. | Binary fallback exists — 0 additional cost | Before PR 18 merge |
| Four scorers from day one (D-4) | All four are independently useful and testable | Run 500-request simulation with each single-scorer configuration. **Success:** each scorer produces a routing distribution distinguishable from random (std_dev > 0 for at least one non-trivial workload). **Abort:** remove scorer, reduce to three. | Remove unused scorer — trivial | Before PR 17 merge |

No decision has cost-if-wrong >= 3 PRs, so no mandatory spike PR is required. Validation gates are placed at PR merge reviews.

---

## F) Architectural Evolution

### Current State

The `weighted` routing policy is a monolithic two-dimension scorer with hardcoded `FreeKVBlocks / maxFreeKVBlocks` (cache) and `1 / (1 + effectiveLoad)` (load) dimensions. The two dimensions are configured via `--routing-cache-weight` and `--routing-load-weight` CLI flags and `cache_weight`/`load_weight` YAML fields.

### Target State

The `weighted` routing policy contains a composable scorer pipeline. Each scorer is an independent scoring function that returns per-instance scores in [0,1]. The pipeline aggregates scores via weighted sum with per-scorer weights. Scorers are configured via `--routing-scorers` CLI flag and `routing-scorers` YAML list. Stateful scorers (prefix-affinity) observe routing decisions to update internal state.

### What Changes

1. `WeightedScoring` struct: replaces two weight fields with a list of weighted scorers
2. `NewRoutingPolicy` factory: `case "weighted"` accepts a scorer configuration instead of two weight values
3. `RoutingConfig` in bundle: replaces two float fields with a scorer list
4. `DeploymentConfig`: replaces two weight fields with scorer config
5. `cmd/root.go`: replaces two weight flags with `--routing-scorers` flag
6. New: scorer implementations (3 stateless in PR 17, 1 stateful in PR 18)
7. New: router-side prefix cache index (PR 18)

### What Remains Unchanged

- `RoutingPolicy` interface (frozen)
- `RoutingSnapshot`, `RouterState`, `RoutingDecision` types
- All non-weighted routing policies (round-robin, least-loaded, prefix-affinity, always-busiest)
- Counterfactual analysis (uses `RoutingDecision.Scores` — unchanged contract)
- All instance-level policies (priority, scheduler)
- Cluster event pipeline, admission policies, snapshot provider

### Fidelity Trade-offs

| Simplification | Real behavior lost | Conditions where it matters | Upgrade path |
|---|---|---|---|
| Router-side LRU approximation | Perfect prefix cache awareness | High prefix diversity with rapid eviction; approximation diverges from actual cache state | Add oracle scorer that directly queries instance KVCache |
| Weighted sum aggregation | Threshold-based filtering (llm-d stage 1) | When load thresholds create hard cutoffs vs. smooth weight tradeoff | Add filter stage before scoring (future extension) |
| Block-level granularity | Sub-block prefix matching | Prefixes shorter than 1 block (16 tokens) | Reduce block size or add token-level fallback |

---

## G) Frozen Interface Reference

The following interfaces are frozen (merged code) and this plan depends on them without modification:

```go
// sim/routing.go:43-45 — frozen PR8
type RoutingPolicy interface {
    Route(req *Request, state *RouterState) RoutingDecision
}
```

```go
// sim/routing.go:28-39 — frozen PR8
type RoutingDecision struct {
    TargetInstance string
    Reason         string
    Scores         map[string]float64
    Priority       float64
}
```

No new interfaces are frozen by this plan. The scorer behavioral contract is internal to the `weighted` policy implementation and is not promoted to a top-level interface.

---

## H) Cross-Cutting Infrastructure Plan

### H.1 Shared Test Infrastructure

- **Existing:** `sim/internal/testutil/` golden dataset loader. No new shared test packages needed.
- **New test helpers:** Per-scorer edge case generators (zero load, uniform load, all-cached, no-cached). These live within the scorer test files, not in a shared package, since they're scorer-specific.
- **Golden dataset impact:** `weighted` routing golden outputs will change. Non-weighted outputs must remain identical. PR 17 must regenerate golden baselines for `weighted` tests and document the expected change.
- **Invariant tests:** INV-1 through INV-7 each have companion tests (not just golden comparisons).

### H.2 Documentation Maintenance

| PR | CLAUDE.md Update | README Update | Examples Update |
|---|---|---|---|
| PR 17 | Update CLI flags table, RoutingConfig description, weighted policy description | Update `weighted` description to reference scorer pipeline | Rewrite `weighted-routing.yaml`, update `policy-config.yaml` |
| PR 18 | Add "Adding New Scorers" section, document prefix cache index | Add llm-d default profile example, replace misleading demo (#230) | Add prefix-affinity scorer example to `weighted-routing.yaml` |

### H.3 CI Pipeline Changes

No CI changes required. Existing `go build`, `golangci-lint`, `go test` pipeline covers all new code.

### H.4 Dependency Management

No new external dependencies. All scorer logic and the LRU cache use standard library only.

### H.5 Interface Freeze Schedule

No new interfaces are frozen by this plan. The scorer contract is internal and can evolve freely across PRs without breaking external code.

---

## I) PR Plan

### PR 17: Scorer Framework + Stateless Scorers

**Extension Type:** Backend swap (replaces monolithic WeightedScoring with scorer pipeline behind unchanged RoutingPolicy interface)

--- Tier 1: Human Review Summary ---

- **Title:** `feat(routing): Composable scorer framework for weighted routing with stateless scorers`
- **Building Block Change:** Replaces Scorer Pipeline + Stateless Scorers + Scorer Configuration
- **Motivation:** The current `weighted` cache dimension measures capacity, not affinity (#229). This PR replaces the monolithic implementation with a composable scorer pipeline and ships three stateless scorers.
- **Scope:**
  - In: Scorer pipeline, score aggregation, queue-depth/kv-utilization/load-balance scorers, YAML/CLI config, flag removal
  - Out: Prefix-affinity scorer (PR 18), picker strategies, non-weighted policy changes
- **Behavioral Guarantees:**
  - BC-17-1: Each scorer returns scores in [0,1] for every instance (INV-1, INV-2)
  - BC-17-2: Weight normalization: `[3,2,2]` and `[0.43,0.29,0.29]` produce identical routing (INV-6)
  - BC-17-3: Non-weighted policies produce byte-identical output (INV-5)
  - BC-17-4: NaN/Inf/negative weights rejected at config time
  - BC-17-5: `weighted` with `load-balance:1` produces identical distribution to `least-loaded` at same request rates
- **Risks:** (1) Golden test breakage for `weighted` — mitigated by documenting expected change. (2) CLI breaking change — mitigated by clear error messages referencing `--routing-scorers`.
- **Cross-Cutting:** Rewrites `examples/weighted-routing.yaml`, updates `examples/policy-config.yaml`, updates CLAUDE.md
- **Validation Gate:** None (no risk register entry with cost >= 3 PRs)

--- Tier 2: Implementation Guide ---

- **Architectural Impact:** `WeightedScoring` struct changes from two weight fields to a list of weighted scorers. `NewRoutingPolicy` factory signature changes. `RoutingConfig` YAML schema replaces two float fields with scorer list.
- **API Surface Changes:** New internal scorer behavioral contract (name, score function, optional observer). New scorer name validation via `IsValidScorer()`. `NewRoutingPolicy` accepts scorer configuration instead of individual weights. `RoutingConfig` YAML fields replaced.
- **Files Changed:** Modified: `sim/routing.go` (~100 LOC delta), `sim/bundle.go` (~40 LOC), `sim/cluster/deployment.go` (~10 LOC), `sim/cluster/cluster.go` (~5 LOC), `cmd/root.go` (~40 LOC). New: `sim/routing_scorers.go` (~120 LOC: scorer interface + 3 implementations + factory). Rewritten: `examples/weighted-routing.yaml`, `examples/policy-config.yaml`.
- **CLI Changes:**
  - Removed: `--routing-cache-weight`, `--routing-load-weight`
  - Added: `--routing-scorers` (comma-separated `name:weight` pairs)
  - Default when unspecified: `queue-depth:2,kv-utilization:2,load-balance:1`
- **Test Categories:**
  - Unit: Per-scorer edge case tests (zero/uniform/extreme values)
  - Unit: Score aggregation with multiple scorers and weight normalization
  - Unit: YAML/CLI parsing and validation (NaN/Inf/negative/unknown names)
  - Regression: Golden dataset stability for non-weighted policies
  - Invariant: BC-17-5 (load-balance-only ≈ least-loaded distribution)
- **Documentation Updates:** CLAUDE.md (CLI flags, RoutingConfig, weighted description), README (weighted description), examples
- **Extension Friction:** 2 files to add a new scorer (implementation + registration). Meets reference target.
- **Parallel Development:** After PR 17, new stateless scorers can be added independently with no coordination.
- **Why Independently Reviewable:** Ships a complete, exercisable scorer framework with three scorers. `weighted` policy is fully functional without prefix-affinity.
- **Why No Dead Code:** Every scorer is exercisable via `--routing-scorers`. No unused scaffolding.
- **LOC Estimate:** ~315 (non-test)

---

### PR 18: Prefix-Affinity Scorer + Router-Side Cache

**Extension Type:** Policy template (new scorer behind the scorer contract established in PR 17)

--- Tier 1: Human Review Summary ---

- **Title:** `feat(routing): Prefix-affinity scorer with router-side approximate prefix cache index`
- **Building Block Change:** Adds Prefix-Affinity Scorer building block
- **Motivation:** Enables the real cache/load tradeoff that #229 identified as missing. Matches llm-d's `prefix-cache-scorer` with hierarchical block hashing and LRU eviction.
- **Scope:**
  - In: Prefix-affinity scorer, router-side prefix cache index with hierarchical block hashing and LRU, observer hook integration, default profile change to llm-d parity, README demo replacement (#230)
  - Out: Oracle scorer (direct KVCache query), RadixCache-style tree matching (SGLang)
- **Behavioral Guarantees:**
  - BC-18-1: Prefix match ratio is proportional — 80% prefix overlap scores higher than 10% (not binary)
  - BC-18-2: No prefix history → score 0 for all instances (INV-4 initial state)
  - BC-18-3: Prefix cache index bounded by O(num_instances × lru_capacity) blocks (INV-7)
  - BC-18-4: Deterministic scoring given same request sequence and seed (INV-3)
  - BC-18-5: Higher prefix-affinity weight produces more concentrated routing distribution for prefix-heavy workloads
- **Risks:** (1) Router-side approximation quality may be low for short prefixes — mitigated by D-2 validation gate. (2) LRU eviction may cause oscillation — mitigated by capacity default (31,250 blocks per instance).
- **Cross-Cutting:** Updates `weighted` default profile, updates README (#230), updates CLAUDE.md
- **Validation Gate:** D-1 and D-2 validated at PR 18 review (prefix workload sweep, block-level score gradation)

--- Tier 2: Implementation Guide ---

- **Architectural Impact:** New prefix cache index data structure (block hash → instance set with LRU timestamps per instance). Observer hook added to `WeightedScoring.Route()` — after routing decision, notify stateful scorers. Default scorer configuration changes from `queue-depth:2,kv-utilization:2,load-balance:1` to `prefix-affinity:3,queue-depth:2,kv-utilization:2`.
- **API Surface Changes:** New `prefix-affinity` scorer name registered. New optional observer behavioral contract. New internal prefix cache index type.
- **Files Changed:** New: `sim/routing_prefix_scorer.go` (~150 LOC: prefix-affinity scorer + observer), `sim/prefix_cache_index.go` (~120 LOC: hierarchical block hashing + LRU index). Modified: `sim/routing_scorers.go` (~15 LOC: register prefix-affinity + update default), `sim/routing.go` (~10 LOC: observer hook in Route). Rewritten: `examples/weighted-routing.yaml`.
- **CLI Changes:**
  - Default for `--routing-scorers` changes to `prefix-affinity:3,queue-depth:2,kv-utilization:2` (llm-d parity)
  - `prefix-affinity` available as scorer name
- **Test Categories:**
  - Unit: Prefix match proportionality (varying overlap lengths)
  - Unit: LRU eviction boundary (exceed capacity, verify oldest evicted)
  - Unit: Initial state (no history → zero scores)
  - Unit: Observer correctness (routing decision updates prefix index)
  - Invariant: INV-7 (memory bound under high prefix diversity)
  - Integration: Weight sensitivity sweep with `servegen-language.yaml` (BC-18-5)
  - Determinism: Same seed + workload → identical routing across runs
- **Documentation Updates:** CLAUDE.md (scorer list, "Adding New Scorers" section), README (replace misleading demo with real prefix workload comparison), `weighted-routing.yaml` (llm-d default profile example)
- **Extension Friction:** 2 files to add a new stateful scorer. Matches reference target.
- **Parallel Development:** After PR 18, future scorers (predicted-latency, oracle, LoRA-affinity) can proceed independently.
- **Why Independently Reviewable:** PR 17 provides the complete scorer framework; PR 18 adds one scorer and its infrastructure.
- **Why No Dead Code:** Prefix-affinity scorer is immediately exercisable and becomes part of the default profile.
- **LOC Estimate:** ~295 (non-test)

---

## J) Dependency DAG

### PR Dependency Graph

```
PR 17 (Scorer Framework + Stateless Scorers)
  ↓
PR 18 (Prefix-Affinity Scorer + Router-Side Cache)
```

Strictly sequential — PR 18 depends on the scorer framework from PR 17.

No interface freeze points — the scorer contract is internal and not promoted to top-level.

### Merge Sequencing

1. PR 17 merges first. Validate: all tests pass, `weighted` exercisable with stateless scorers, non-weighted policies unchanged.
2. PR 18 merges second. Validate: prefix workload sweep shows real weight sensitivity, llm-d default profile active, README demo is meaningful.

### Integration Risk

Low — both PRs modify the same area (`weighted` policy in `sim/routing.go` and related config). Sequential dependency ensures no merge conflicts. No other in-progress PRs (PR11, PR14, PR15, PR16) touch routing policy internals.

---

## K) Design Bug Prevention Checklist

### Invariants That Must Never Be Broken

- INV-1 through INV-7 (defined in design document)
- Existing: request conservation, clock monotonicity, KV cache conservation, causality, determinism

### Regression Surfaces

| Existing Test | Must Pass? | Expected Change? |
|---|---|---|
| Golden dataset (non-weighted policies) | Yes | No change |
| Golden dataset (`weighted` policy) | Must update baselines | Scoring formula changes — expected |
| `sim/routing_test.go` WeightedScoring tests | Must update | Test new scorer pipeline behavior |
| `sim/bundle_test.go` RoutingConfig tests | Must update | YAML schema changes |
| `sim/cluster/cluster_test.go` weighted tests | Must update | Config field changes |
| `sim/cluster/pending_requests_test.go` | Must update | Config field changes |
| `sim/cluster/cluster_trace_test.go` | Must update | Config field changes |
| All other tests | Must pass unchanged | No change |

### Cross-PR State Migration Risks

PR 17 changes `RoutingConfig` YAML schema (replaces `cache_weight`/`load_weight` with `scorers` list). PR 18 changes the default scorer profile. These are independent — PR 18 extends the config surface introduced by PR 17, no conflicting schema changes. Users who adopt PR 17's YAML format will not need to change their files when PR 18 merges (the default changes, but explicit configs are unaffected).

### Common Failure Modes Prevented

**General:**

| Failure Mode | Prevention |
|---|---|
| Scaffolding creep | Every scorer is exercisable in the PR that introduces it |
| Documentation drift | CLAUDE.md, README, and examples updated in each PR |
| Golden dataset staleness | PR 17 regenerates weighted baselines with documented rationale |
| Interface over-specification | No new public interfaces — scorer contract is internal |
| The Type Catalog trap | Design doc and macro plan describe scorers behaviorally, not as Go types |
| Exported mutable maps | Scorer name registry follows existing `validRoutingPolicies` pattern (unexported map + `IsValid*()` accessor) |
| YAML zero-value ambiguity | Scorer weights use explicit values in config; no zero-value-means-default semantics |
| Strict YAML parsing | Existing `KnownFields(true)` ensures old `cache_weight`/`load_weight` fields produce parse errors |
| Non-deterministic map iteration | Scorer pipeline iterates a slice (ordered), not a map. Prefix cache index sorts instance IDs before scoring (INV-3). |
| NaN/Inf in weights | CLI and YAML validation per antipattern rule 3 |

**DES-specific:**

| Failure Mode | Prevention |
|---|---|
| Golden tests without invariant tests | INV-1 through INV-7 each have dedicated invariant tests, not just golden comparisons. Golden tests are updated with documented rationale, not blindly regenerated. |
| Mixing exogenous and endogenous | N/A — no new events introduced. Scorers are synchronous functions within the existing endogenous RoutingDecisionEvent. |
| Interface leaking implementation | Scorer contract is behavioral (name, score function, optional observer) not implementation-specific. The contract must accommodate both stateless (queue-depth) and stateful (prefix-affinity) implementations. |

**Module architecture:**

| Failure Mode | Prevention |
|---|---|
| Shotgun surgery | `NewRoutingPolicy` factory is the single construction site for `WeightedScoring`. Scorer config propagation: CLI → `DeploymentConfig` → factory — 3 touch points, within reference target. |
| Config mixing concerns | Scorer configuration is scoped to the `weighted` policy within `RoutingConfig`, not mixed into `SimConfig` or `DeploymentConfig` top-level fields. |
