# Composable Scorer Framework for Weighted Routing

**Date:** 2026-02-19
**Status:** Draft (pending review)
**Species:** Specification
**Closes:** #229, #230

---

## 1. Motivation

BLIS's `weighted` routing policy combines two scoring dimensions — KV cache capacity headroom and load balance — into a composite score. The cache dimension measures `FreeKVBlocks / maxFreeKVBlocks`, which is a **capacity** signal, not a **prefix affinity** signal. In homogeneous clusters these two dimensions are correlated (both measure instance busyness from different angles), so varying their weights produces a feedback latency artifact rather than a meaningful tradeoff (#229).

Production LLM routers (llm-d's Endpoint Picker, SGLang's cache-aware router) solve this by decomposing routing into **independent scoring dimensions** — prefix match ratio, queue depth, KV utilization — each scored in [0, 1] and combined with configurable weights. BLIS needs the same composability so researchers can:

1. Prototype and tune llm-d scheduling profiles in simulation before deploying
2. Compare scoring dimension combinations (e.g., prefix affinity vs. capacity headroom)
3. Study how weight sensitivity varies across workload patterns

---

## 2. Scope

**In scope:**
- A composable scorer framework within the `weighted` routing policy
- Four initial scorer implementations (prefix-affinity, queue-depth, kv-utilization, load-balance)
- Router-side approximate prefix cache index (matching llm-d's approach)
- YAML and CLI configuration for scorer weights
- Replacement of the broken cache dimension with prefix affinity scoring

**Out of scope:**
- Changes to the frozen `RoutingPolicy` interface — the scorer framework lives *inside* the `weighted` implementation
- Predicted-latency scoring (llm-d has this but it requires latency model integration — deferred)
- LoRA affinity scoring (llm-d has this for multi-LoRA serving — deferred, not currently modeled in BLIS)
- Picker strategies (max-score vs. weighted-random) — use argmax initially, add picker abstraction later if needed
- Changes to other routing policies (`round-robin`, `least-loaded`, `prefix-affinity`, `always-busiest`)

**Deferred:**
- Predicted-latency scorer (requires latency model interface extraction — tracked in design guidelines as target module)
- Scorer category types (llm-d classifies scorers as Affinity/Distribution/Balance — useful for auto-configuration but not needed for manual weight tuning)
- Per-scorer parameters (e.g., prefix cache LRU capacity as a tunable) — use sensible defaults initially

**Extension type classification (per design guidelines §5.1):**
- PR A (scorer framework + stateless scorers) is a **backend swap** — replacing the monolithic `WeightedScoring` implementation with a composable scorer pipeline, behind the unchanged `RoutingPolicy` interface
- PR B (prefix-affinity scorer) is a **policy template** within the scorer framework — a new algorithm behind the scorer contract established in PR A

---

## 3. Modeling Decisions

| Component | Modeled | Simplified | Omitted | Justification |
|---|---|---|---|---|
| Prefix match ratio | Proportional block-level matching per instance | Hierarchical block hashing with LRU eviction on the router side, matching llm-d's approximation | Exact server-side cache state query | Router-side approximation is what production systems use; studying the information asymmetry between router approximation and actual cache state is an explicit analysis question |
| Queue depth normalization | Min-max normalization matching llm-d's queue-scorer | — | Exponential decay, time-weighted averages | Min-max is simple and proven in llm-d; more sophisticated normalization adds complexity without answering different questions |
| KV utilization scoring | Inverse utilization (1 - util) matching llm-d | — | Per-layer utilization breakdown | Per-layer breakdown requires KVStore interface changes; aggregate utilization is sufficient for routing decisions |
| Score aggregation | Weighted sum with [0,1] clamping per scorer | — | Multiplicative scoring, cascaded filtering | Weighted sum is the standard approach in both llm-d and k8s scheduling; multiplicative scoring creates threshold effects that complicate weight interpretation |
| Load balance scoring | Inverse transform 1/(1+load), preserving BLIS's existing formula | — | llm-d's approach (which uses queue-depth min-max instead) | Offering both normalization strategies enables comparison studies |

---

## 4. Invariants

**INV-1 (Score Range):** Every scorer must return values in [0, 1] per instance. Values outside this range are clamped before weighting.

**INV-2 (Score Completeness):** Every scorer must return a score for every instance in the snapshot list. Missing scores are treated as 0.

**INV-3 (Determinism):** Given the same request sequence, seed, and scorer configuration, the `weighted` policy must produce identical routing decisions across runs. The router-side prefix cache index must be deterministic (no map iteration order dependencies in scoring).

**INV-4 (Observer Consistency):** Stateful scorers (those that observe routing decisions to update internal state) must update their state for every routed request, including the first request when no prior state exists. Initial state must produce well-defined scores (e.g., 0 for prefix affinity when no routing has occurred yet). Note: the observer creates a sequential dependency — scoring for request N depends on all routing decisions for requests 1..N-1. This is correct and deterministic given INV-3, but means scorer state cannot be reconstructed without replaying the full routing history.

**INV-5 (Backward Routing Stability):** The existing `round-robin`, `least-loaded`, `prefix-affinity`, and `always-busiest` routing policies must be completely unaffected by the scorer framework changes. Only `weighted` changes behavior.

**INV-6 (Weight Normalization):** Scorer weights are relative — only ratios matter. The framework normalizes weights so they sum to 1.0 before scoring. Zero or negative weights are rejected at configuration time.

**INV-7 (Prefix Cache Conservation):** The router-side prefix cache index must not grow unboundedly. LRU eviction must bound the per-instance block count. Total memory is O(num_instances × lru_capacity).

---

## 5. Decisions with Trade-offs

### D-1: Router-side approximate prefix cache vs. direct KVCache query

**Decision:** Use a router-side approximate prefix cache index, matching llm-d's architecture.

**Alternatives considered:**
- **(a) Direct query:** Query each instance's KVCache for prefix matches. BLIS has full visibility since it's a simulator. Simpler implementation, always accurate.
- **(b) Router-side approximation (chosen):** Maintain an approximate block-hash index at the router, updated when routing decisions are made, with LRU eviction per instance.

**Why (b) wins:** The router-side approximation is what production systems use because they can't query GPU memory. If BLIS uses direct queries, it simulates an idealized router that doesn't exist in practice. Researchers studying cache-aware routing need to understand how approximation quality degrades under high request rates, prefix diversity, and LRU eviction pressure. The information asymmetry is the phenomenon worth studying.

**What breaks if wrong:** If researchers need idealized routing (upper bound analysis), they'd need a separate "oracle" scorer. This is a straightforward addition later — a scorer that directly queries instance KVCache state.

### D-2: Hierarchical block hashing vs. full-sequence hashing

**Decision:** Use hierarchical block hashing where hash(block_i) chains the previous block's hash, matching llm-d's approach.

**Alternatives considered:**
- **(a) Full-sequence hash:** Hash the entire input token sequence as one unit. Binary match (all or nothing).
- **(b) Hierarchical block hashing (chosen):** Split tokens into blocks of configurable size, hash each block incorporating the previous block's hash. This creates prefix-semantic hashes: two requests sharing the first K blocks produce identical hashes for those K blocks.

**Why (b) wins:** Proportional matching. Two requests sharing 80% of their prefix should score differently from two sharing 10%. Full-sequence hashing reduces to binary (match/no-match), which is what the current `PrefixAffinity` policy already does. The point of the scorer framework is *graduated* signals.

**What breaks if wrong:** Block-level granularity means short prefixes (< 1 block) produce no match. This is acceptable because sub-block prefixes provide negligible KV cache savings.

### D-3: Scorer as internal abstraction vs. new top-level interface

**Decision:** The scorer is an internal abstraction within the `weighted` routing policy, not a new top-level interface visible at the module map level.

**Alternatives considered:**
- **(a) Top-level Scorer interface** alongside RoutingPolicy, registered in bundle.go, with its own factory.
- **(b) Internal abstraction (chosen):** The scorer contract is defined and used only within the routing module. The `weighted` policy owns its scorer instances.

**Why (b) wins:** The frozen `RoutingPolicy` interface already provides the module boundary. Scorers are implementation details of one policy variant — they don't need their own module contract, factory, or CLI surface. Keeping them internal avoids interface proliferation and means the extension friction for adding a new scorer is very low (one file + registration in the weighted policy's configuration parser).

**What breaks if wrong:** If we later need scorers outside of routing (e.g., for admission policy scoring), we'd need to promote the abstraction. This would be a straightforward refactoring — extract the internal interface to `sim/`.

### D-4: Four initial scorers vs. just prefix-affinity + load-balance

**Decision:** Ship four scorers (prefix-affinity, queue-depth, kv-utilization, load-balance) to match llm-d's default profile from day one.

**Alternatives considered:**
- **(a) Minimal (two scorers):** Only prefix-affinity and load-balance, matching the issue #229 proposal.
- **(b) Full llm-d parity (four scorers, chosen):** prefix-affinity, queue-depth, kv-utilization, and load-balance.

**Why (b) wins:** The user wants to experiment with llm-d configurations from day one. llm-d's default profile uses prefix-cache (weight 3), queue-depth (weight 2), and kv-utilization (weight 2). The load-balance scorer (BLIS's existing inverse-transform formula) is a bonus that enables comparing normalization strategies with queue-depth.

**What breaks if wrong:** More code to review and test, but each scorer is small (conceptually ~20-30 lines of behavioral logic) and independently testable.

### D-5: Drop `--routing-cache-weight` / `--routing-load-weight` vs. backward compatibility

**Decision:** Replace the old two-weight CLI flags with a single `--routing-scorers` flag (comma-separated `name:weight` pairs) and YAML `routing-scorers` configuration.

**Alternatives considered:**
- **(a) Keep old flags as aliases** mapping to `[{prefix-affinity, cache_weight}, {load-balance, load_weight}]`.
- **(b) Clean break (chosen):** Remove old flags, introduce new configuration surface.

**Why (b) wins:** The old flags encode a broken mental model (cache capacity vs. load). Keeping them as aliases preserves the misconception. A clean break with new flags forces users to understand what they're configuring. The user explicitly stated backward compatibility is not a necessity here.

**What breaks if wrong:** Users with existing scripts using `--routing-cache-weight` get an error and must update. This is a feature, not a bug — it surfaces the semantic change.

---

## 6. Module Contract: Scorer (Internal to Router Module)

Since the scorer is an internal abstraction within the `weighted` routing policy (D-3), its module contract is subordinate to the Router module's contract:

| Aspect | Contract |
|---|---|
| **Observes** | Request metadata (token IDs for prefix matching, SLO class for future priority-aware scoring), per-instance snapshots (queue depth, batch size, KV utilization, free KV blocks, pending requests, cache hit rate), cluster clock |
| **Controls** | Per-instance score in [0, 1] for its scoring dimension |
| **Owns** | Scorer-specific internal state (e.g., prefix cache index for prefix-affinity scorer). Each scorer owns its state exclusively. |
| **Invariants** | INV-1 (score range), INV-2 (completeness), INV-3 (determinism) |
| **Events** | None — scorers are synchronous functions called within the routing event handler. They do not produce or consume events independently. |
| **Extension friction** | 1 file to add a new scorer implementation + 1 registration in the scorer factory/configuration parser = **2 touch points** (meets reference target) |

### Scorer Implementations

**Prefix-Affinity Scorer:**
- Observes: request token IDs, instance IDs from snapshots
- Owns: router-side prefix cache index (block hash → set of instance IDs with LRU timestamps)
- Score: proportion of request's prefix blocks that match cached blocks for each instance (longest consecutive match / total blocks)
- Stateful: implements the observer contract to update the index after routing decisions
- Parameters: block size (default 16), LRU capacity per instance (default 31,250 blocks)

**Queue-Depth Scorer:**
- Observes: effective load from snapshots (queue depth + batch size + pending requests)
- Owns: no state (stateless)
- Score: min-max normalization — (max_load - instance_load) / (max_load - min_load), returns 1.0 if all loads are equal
- Matches llm-d's queue-scorer semantics

**KV-Utilization Scorer:**
- Observes: KV utilization from snapshots
- Owns: no state (stateless)
- Score: 1 - kv_utilization (higher score for less utilized instances)
- Matches llm-d's kv-cache-utilization-scorer semantics

**Load-Balance Scorer:**
- Observes: effective load from snapshots
- Owns: no state (stateless)
- Score: 1 / (1 + effective_load) (inverse transform, preserves absolute differences)
- BLIS-native formula, alternative to queue-depth's min-max normalization

---

## 7. Extension Points

**Adding a new scorer:** Implement the scorer behavioral contract (name, score function returning map of instance IDs to [0,1] scores, optional observer for post-routing state updates). Register in the scorer factory. The `weighted` policy discovers it via configuration. Extension friction: 2 files.

**Default behavior (staged):**
- After PR A (before prefix-affinity scorer exists): When `routing-policy: weighted` is specified without `--routing-scorers`, the default is `queue-depth:2, kv-utilization:2, load-balance:1` — all stateless scorers with queue-depth and kv-utilization matching llm-d's non-prefix defaults.
- After PR B (prefix-affinity scorer available): Default changes to llm-d's full profile: `prefix-affinity:3, queue-depth:2, kv-utilization:2`.

The default change between PR A and PR B is a deliberate behavior change, not a bug. PR B's CHANGELOG must note this.

**First non-default configuration:** A load-balance-only profile equivalent to the current `least-loaded` behavior: `--routing-scorers "load-balance:1"`.

**Future extensions:**
- Predicted-latency scorer (after latency model interface extraction)
- LoRA-affinity scorer (after multi-LoRA modeling)
- Oracle prefix scorer (direct KVCache query for upper-bound analysis)

---

## 8. Validation Strategy

### Correctness Verification (Invariants)

- **INV-1, INV-2 (score range/completeness):** Unit tests per scorer with edge cases (empty queue, all-equal loads, zero utilization, no prefix match)
- **INV-3 (determinism):** Run identical workload twice with same seed, assert byte-identical routing decisions
- **INV-5 (backward stability):** Golden dataset tests must produce identical output for non-weighted routing policies
- **INV-6 (weight normalization):** Test that `[3,2,2]` and `[0.43,0.29,0.29]` produce identical routing
- **INV-7 (prefix cache bound):** Test that prefix cache index doesn't exceed `num_instances × lru_capacity` blocks

### Existing Test Impact

The golden dataset tests currently exercise `weighted` routing via `routingCacheWeight`/`routingLoadWeight`. After refactoring, the `weighted` policy changes its scoring formula (new scorer pipeline replaces the old two-dimension formula). Golden test output for `weighted` routing will change — this is expected and correct, since the old formula was measuring the wrong signal (#229). Tests for non-weighted policies (round-robin, least-loaded, etc.) must remain byte-identical.

### Fidelity Validation (Against Real Systems)

- **llm-d correspondence:** Configure BLIS with llm-d's default profile (prefix:3, queue:2, kv-util:2) and compare routing distribution patterns against llm-d's EPP simulator logs (qualitative — same request pattern should route to same instances in similar proportions)
- **Prefix workload sensitivity:** Using `servegen-language.yaml` (70% shared prefix), sweep prefix-affinity weight from 0 to 5 and verify that higher weights produce more concentrated routing (measurable via distribution standard deviation)
- **Weight sensitivity:** Demonstrate that varying weights produces *meaningfully different* routing distributions (unlike the current broken `WeightedScoring` where only feedback artifacts vary — see #229 evidence table)

---

## 9. DES Design Review Checklist

| Question | Answer |
|---|---|
| What analysis questions does this design help answer? | (1) What is the optimal weight combination for prefix-affinity vs. load-balance vs. KV-utilization for a given workload? (2) How does router-side cache approximation quality degrade under high request rates and prefix diversity? (3) How do llm-d scheduling profiles perform under controlled simulation? |
| What is modeled, simplified, and deliberately omitted? | See Section 3 table |
| What events are introduced or modified? | None. Scorers are synchronous functions within the existing `RoutingDecisionEvent` handler. The observer hook (for prefix cache updates) executes within the same event, not as a separate event. |
| How do new events interact with existing tie-breaking? | N/A — no new events |
| What new state is introduced? Who owns it? | Router-side prefix cache index, owned exclusively by the prefix-affinity scorer instance within the `weighted` policy. Per-scorer weight configuration, owned by the `weighted` policy. Both are simulation state (they evolve the system by influencing routing decisions), not statistics. |
| What new metrics are derived? | No new system-level metrics (statistics). The existing `RoutingDecision.Scores` map naturally reflects the composite score from the scorer pipeline — this is trace data (observation), not state. Future: per-scorer score breakdown in trace records. |
| How will correctness be verified? | Invariant tests (INV-1 through INV-7), determinism tests, golden dataset stability tests. See Section 8. |
| How will fidelity be validated? | llm-d correspondence checks, prefix workload sensitivity sweeps. See Section 8. |
| Does this introduce new randomness? | No. All scoring is deterministic given the same inputs. The prefix cache index is deterministic (LRU eviction order is determined by access timestamps from the deterministic routing sequence). |
| What is the simplest version that answers the same questions? | A single binary prefix-affinity scorer + load-balance scorer (issue #229's proposal). We chose four scorers to enable llm-d profile comparison from day one, but the framework is functional with just two. |

---

## 10. Real-System Correspondence

| BLIS Component | llm-d EPP | SGLang | vLLM |
|---|---|---|---|
| Scorer framework (weighted policy) | Scheduling profile with weighted scorer plugins | Cache-aware router with load threshold | N/A (single-instance) |
| Prefix-affinity scorer | `prefix-cache-scorer` (block hash index, LRU) | RadixCache-aware routing (tree-based prefix match) | N/A |
| Queue-depth scorer | `queue-scorer` (min-max normalization) | Queue threshold filter | N/A |
| KV-utilization scorer | `kv-cache-utilization-scorer` (1 - util) | KV cache threshold filter | N/A |
| Load-balance scorer | No direct equivalent (queue-scorer subsumes) | No direct equivalent | N/A |
| Score aggregation | Weighted sum, clamp to [0,1] per scorer | Two-stage: threshold filter, then cache match | N/A |
| Picker | Argmax (max-score) | Longest prefix match among eligible | N/A |

**Key difference from llm-d:** llm-d uses a two-stage approach (filter by load thresholds, then score by prefix match among eligible candidates). BLIS uses a single-stage weighted sum. The single-stage approach is more general — it can approximate two-stage behavior by setting very high load-scorer weights — but the behavioral difference matters for studying threshold effects vs. continuous tradeoffs.

**Key difference from SGLang:** SGLang's routing is tree-based (RadixCache prefix tree traversal), not block-hash-based. BLIS's block-hash approach matches llm-d more closely. Modeling SGLang's tree-based approach would require a different scorer implementation (future extension).

---

## 11. Configuration Surface

### YAML (via `--policy-config`)

```yaml
routing-policy: weighted
routing-scorers:
  - name: prefix-affinity
    weight: 3.0
  - name: queue-depth
    weight: 2.0
  - name: kv-utilization
    weight: 2.0
```

### CLI (quick experiments)

```bash
--routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2,kv-utilization:2"
```

If `--routing-policy weighted` is specified without `--routing-scorers`, the default llm-d profile is used.

### Removed

- `--routing-cache-weight` (replaced by per-scorer weights)
- `--routing-load-weight` (replaced by per-scorer weights)

### YAML Schema Migration

The current `RoutingConfig` in `bundle.go` has `cache_weight` and `load_weight` fields. These are replaced by a `scorers` list. Since BLIS uses strict YAML parsing (`KnownFields(true)`), old YAML files with `cache_weight`/`load_weight` will produce a parse error — this is the desired behavior (surfaces the semantic change).

Files requiring update:
- `examples/weighted-routing.yaml` — replace with new scorer configuration and updated documentation
- `examples/policy-config.yaml` — update commented-out weighted routing example

---

## 12. Phasing

This design is implementable in two PRs:

**PR A — Scorer Framework + Stateless Scorers:**
- Scorer behavioral contract (internal to routing module)
- Score aggregation with weighted sum and clamping
- Queue-depth, kv-utilization, and load-balance scorer implementations
- YAML and CLI configuration for scorer weights
- Removal of old `--routing-cache-weight` / `--routing-load-weight` flags
- Tests for all invariants except INV-7 (prefix cache bound)

**PR B — Prefix-Affinity Scorer + Router-Side Cache:**
- Router-side prefix cache index with hierarchical block hashing and LRU eviction
- Prefix-affinity scorer implementation with observer hook
- llm-d default profile activation
- Weight sensitivity demonstration with prefix-heavy workload
- README update replacing misleading demo (#230)
- Tests for INV-7 (prefix cache bound) and fidelity validation

PR A is independently valuable (fixes the broken cache dimension, enables load-only scorer experiments). PR B adds prefix intelligence and llm-d parity.
