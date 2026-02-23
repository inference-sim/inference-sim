# BLIS Extension Recipes

Step-by-step guides for extending BLIS. Each recipe lists the exact files to touch, the order, and examples to follow.

## Adding New Policy Templates

To add a new policy template (e.g., a new routing algorithm):

1. **Implement the interface** in the corresponding file:
   - `AdmissionPolicy` → `sim/admission.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `RoutingPolicy` → `sim/routing.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `PriorityPolicy` → `sim/priority.go` (instance-level: receives `req` + `clock` only)
   - `InstanceScheduler` → `sim/scheduler.go` (instance-level: receives `requests` + `clock` only)
   - Note: `RouterState` is a bridge type in `sim/` to avoid import cycles — see `sim/router_state.go`

2. **Register in two places** (both required):
   - Add policy name to valid names map in `sim/bundle.go` (e.g., `validRoutingPolicies`) and corresponding `IsValid*` function
   - Add `case` to factory function in the same policy file (e.g., `NewRoutingPolicy` in `sim/routing.go`)
   - CLI error messages auto-derive from `ValidAdmissionPolicyNames()` etc. — no manual update needed

3. **Add tests** following BDD naming: `TestMyPolicy_Scenario_Behavior`
   - Test observable behavior, not internal structure
   - Include empty-snapshots panic test for routing policies (defensive programming convention)
   - Use `&RouterState{Snapshots: snapshots, Clock: clock}` in test setup

4. **Update documentation**: CLAUDE.md file organization, README policy lists

**Important:** For load-based routing, use `snap.EffectiveLoad()` — never compute `QueueDepth + BatchSize + PendingRequests` inline. This ensures all routing policies use the same formula.

Examples:
- See `RejectAll` in `sim/admission.go` for a simple admission template (constant return)
- See `PrefixAffinity` in `sim/routing.go` for a stateful routing policy with LeastLoaded fallback. **Known limitation (#259):** hashes the full input sequence, not just the prefix — degrades to LeastLoaded for prefix-sharing workloads. PR18's prefix-affinity scorer with hierarchical block hashing addresses this.

## Adding New Scorers (Weighted Routing)

To add a new scoring dimension for the `weighted` routing policy (e.g., predicted-latency):

1. **Implement the scorer function** in `sim/routing_scorers.go` (stateless) or a new file (stateful) — a `scorerFunc` that takes `(*Request, []RoutingSnapshot)` and returns `map[string]float64` with scores in [0,1] per instance. Stateful scorers also return an `observerFunc` called after each routing decision.
2. **Register the scorer** in `sim/routing_scorers.go`: add to `validScorerNames` map + `newScorerWithObserver` factory switch
3. **Add behavioral tests** — monotonicity, boundary values, INV-1/INV-2 conformance
4. Extension friction: **2 touch points** (implementation + registration in `newScorerWithObserver`). Stateful scorers (like prefix-affinity) may use a separate file (e.g., `sim/routing_prefix_scorer.go`) but the registration point is the same `newScorerWithObserver` switch in `sim/routing_scorers.go`.
5. **Stateful scorers** return an `observerFunc` alongside the `scorerFunc` from `newScorerWithObserver`. The `observerFunc` signature is `func(req *Request, targetInstance string)` and is called after each routing decision to update scorer state. The scorer and observer share state via closure.

Examples:
- See `scoreLoadBalance` in `sim/routing_scorers.go` for a simple stateless scorer
- See `scoreQueueDepth` for a scorer with edge case handling (uniform load)
- See `newPrefixAffinityScorer` in `sim/routing_prefix_scorer.go` for a stateful scorer with observer and router-side cache

## Extending KV Cache Tiers

To add a new KV tier (e.g., NVMe offloading for 3-tier GPU+CPU+NVMe):

1. **Implement the `KVStore` interface** in `sim/kvcache_*.go` (11 methods: allocate, get cached, release, capacity queries, metrics, `SetClock`, `ConsumePendingTransferLatency`)
2. **Compose existing tiers** — e.g., wrap `TieredKVCache` (GPU+CPU) with NVMe logic, following the same delegation pattern
3. **Update `NewKVStore` factory** in `sim/kv_store.go` to instantiate your tier based on `SimConfig` fields
4. **Add CLI flags** in `cmd/root.go` for new parameters (e.g., `--kv-nvme-blocks`)
5. **Aggregate metrics** — combine hit/miss/thrashing counters from all tiers; see `TieredKVCache.CacheHitRate()` for the 2-tier pattern
6. **Add behavioral tests** in `sim/kvcache_*_test.go`
7. **Preserve rollback semantics** — `KVCacheState.AllocateKVBlocks` is transactional: on mid-loop failure, `rollbackAllocation()` undoes all mutations (UsedBlockCnt, CacheMisses, CacheHits, RefCount, InUse, free list, HashToBlock, RequestMap). If your tier adds mutations beyond what delegation to `gpu.AllocateKVBlocks()` handles, you must roll those back too. See `cachedBlockMutation` and `newBlockMutation` types in `sim/kvcache.go`.
8. **`GetCachedBlocks` is a pure query** — it returns cached block IDs without side effects. `CacheHits` are counted by `AllocateKVBlocks` when cached blocks are committed to an allocation (and rolled back on failure). This was fixed in the Phase 3 hardening PR; the previous implementation incremented CacheHits in GetCachedBlocks, causing double-counting in tiered mode.

Examples:
- See `TieredKVCache` in `sim/kvcache_tiered.go` for 2-tier GPU+CPU composition
- See `KVCacheState` in `sim/kvcache.go` for single-tier baseline (also implements `KVStore`)
- See `docs/plans/archive/pr12-architectural-predesign.md` for the design decisions behind the tiered architecture

## Adding New Trace Record Types

To add a new trace record type (e.g., `ScaleRecord` for autoscaling events):

1. **Define the record struct** in `sim/trace/record.go` (pure data, no `sim/` dependency)
2. **Add a slice field** to `SimulationTrace` in `sim/trace/trace.go` (e.g., `Scales []ScaleRecord`)
3. **Add a recording method** to `SimulationTrace` (e.g., `RecordScale(ScaleRecord)`)
4. **Hook recording** into the cluster event pipeline in `sim/cluster/cluster_event.go` (guard with `if cs.trace != nil` for zero-overhead default)
5. **Update `Summarize()`** in `sim/trace/summary.go` to aggregate the new record type
6. **Add behavioral tests** in `sim/trace/*_test.go`

Examples:
- See `AdmissionRecord` in `sim/trace/record.go` for a simple record
- See `RoutingRecord` with `CandidateScore` for a record with nested counterfactual data
- See `computeCounterfactual()` in `sim/cluster/counterfactual.go` for derived computation that lives in `sim/cluster/` (not `sim/trace/`) because it needs `sim.RoutingSnapshot`

## Adding New Latency Model Backends

To add a new latency estimation backend (e.g., SGLang RadixAttention, TensorRT-LLM, neural surrogate):

1. **Implement the `LatencyModel` interface** in `sim/latency_model.go` (or a new file for complex models) — 5 methods:
   - `StepTime(batch []*Request) int64` — estimate batch step duration from request states
   - `QueueingTime(req *Request) int64` — estimate arrival-to-queue delay
   - `OutputTokenProcessingTime() int64` — per-token post-processing overhead
   - `SchedulingProcessingTime() int64` — scheduling overhead per request
   - `PreemptionProcessingTime() int64` — preemption overhead per eviction
2. **Register in `NewLatencyModel` factory** in `sim/latency_model.go`: add a branch based on `SimConfig` fields (e.g., a new string field or boolean)
3. **Add behavioral tests** — monotonicity (more tokens → longer step time), positive output, boundary cases (empty batch)
4. Extension friction: **2 touch points** (implementation + factory branch)

Examples:
- See `BlackboxLatencyModel` in `sim/latency_model.go` for a simple stateless model (alpha/beta regression)
- See `RooflineLatencyModel` in `sim/latency_model.go` for a model that uses hardware config (FLOPs/bandwidth)

## Adding New Batch Formation Strategies

To add a new batch formation strategy (e.g., disaggregated prefill/decode, speculative decoding, continuous batching without preemption):

1. **Implement the `BatchFormation` interface** in `sim/batch_formation.go` (or a new file for complex strategies) — 1 method:
   - `FormBatch(ctx BatchContext) BatchResult` — compose the running batch for the next step
   - The implementation receives `BatchContext` with: RunningBatch, WaitQ, KVCache, token budget, batch size limit, chunked prefill threshold, simulation time, step count, and ComputedTokens map
   - The implementation MUST update `ctx.ComputedTokens[req.ID]` for each request that receives new tokens (Phase 2 of `Step()` reads this map to advance `ProgressIndex`)
   - The implementation may mutate `WaitQ` (dequeue/prepend) and `KVCache` (allocate/release) during batch formation
   - The implementation MUST NOT schedule events or record metrics — return decisions in `BatchResult`, the Simulator applies them
2. **Register in `NewBatchFormation` factory** in `sim/batch_formation.go`: add a selection branch (currently returns `VLLMBatchFormation` unconditionally — a future PR will add `SimConfig.BatchFormation` field for strategy selection)
3. **Add behavioral tests** — token budget enforcement, batch size limits, KV conservation, preemption behavior (if applicable), FCFS ordering
4. Extension friction: **2 touch points** (implementation + factory registration)

**Note:** Currently only `VLLMBatchFormation` exists. Adding a second strategy will also require: (a) a `BatchFormation string` field in `SimConfig`, (b) a CLI flag in `cmd/root.go`, (c) validation in `sim/bundle.go`, (d) selection logic in `NewBatchFormation`.

Examples:
- See `VLLMBatchFormation` in `sim/batch_formation.go` for the vLLM FCFS + chunked-prefill + preemption strategy
- See `preemptForTokens` for the KV allocation + eviction loop pattern

## Adding New Per-Request Metric Fields

To add a new field to per-request JSON output (appears in `--results-path` output):

1. **Add field to `Request`** in `sim/request.go` (runtime state, zero-value safe). When constructing `Request` structs, use `RequestState` typed constants (`StateQueued`, `StateRunning`, `StateCompleted`) — never bare strings.
2. **Add field to `RequestMetrics`** in `sim/metrics_utils.go` (JSON output struct, use `omitempty` for backward compatibility)
3. **Update `NewRequestMetrics()` constructor** in `sim/metrics_utils.go` to propagate the new field from `Request` to `RequestMetrics`
4. **Set the field** at the appropriate event (e.g., `RoutingDecisionEvent` for cluster-level, or completion for computed metrics)
5. **Add behavioral tests** covering multi-instance, single-instance, and standalone boundaries

Examples:
- See `HandledBy` (#181) — set by `RoutingDecisionEvent`, zero-value when used outside cluster pipeline (suppressed from JSON via `omitempty`)
- See `SLOClass`/`TenantID` (PR10) — set during workload generation, propagated at injection
