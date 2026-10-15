# BLIS Extension Recipes

Step-by-step guides for extending BLIS. Each recipe lists the exact files to touch, the order, and examples to follow.

## Adding New Policy Templates

To add a new policy template (e.g., a new routing algorithm):

1. **Implement the interface** in the corresponding file:
   - `AdmissionPolicy` → `sim/admission.go` (cluster-level: receives `*RouterState` with snapshots + clock)
   - `RoutingPolicy` → `sim/routing.go` (cluster-level: receives `*RouterState` with snapshots + clock)
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

4. **Update documentation**: the relevant `docs/` guide and README policy lists. Touch `CLAUDE.md` only if you are adding an operating rule an agent needs every session (see the charter at the top of CLAUDE.md) — it holds pointers, not per-feature detail.

**Important:** For composite load signals, use `snap.EffectiveLoad()` — never compute `QueueDepth + BatchSize + InFlightRequests` inline. For queue-depth-only signals, use `snap.QueueDepth` directly.

Examples:
- See `RejectAll` in `sim/admission.go` for a simple admission template (constant return)
- See `newPrefixAffinityScorer` in `sim/routing_prefix_scorer.go` for a stateful scorer with observer-based state updates (the prefix-affinity scorer uses a router-side `PrefixCacheIndex` to track per-instance block hash history)

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

1. **Implement the `KVStore` interface** in `sim/kv/` (11 methods: allocate, get cached, release, capacity queries, metrics, `SetClock`, `ConsumePendingTransferLatency`)
2. **Compose existing tiers** — e.g., wrap `TieredKVCache` (GPU+CPU) with NVMe logic, following the same delegation pattern
3. **Update `NewKVStore` factory** in `sim/kv/register.go` to instantiate your tier based on `KVCacheConfig` fields (add new fields to `KVCacheConfig` in `sim/config.go`)
4. **Add CLI flags** in `cmd/root.go` for new parameters (e.g., `--kv-nvme-blocks`) and wire them into the `KVCacheConfig` sub-config
   Tier transfer *time* is not computed here: each tier names a catalog `device_class`, and `blis-latency-kernel` prices the transfer (`TierTime`). A new device class is added to `blis-catalog`.
5. **Aggregate metrics** — combine hit/miss/thrashing counters from all tiers; see `TieredKVCache.CacheHitRate()` for the 2-tier pattern
6. **Add behavioral tests** in `sim/kv/*_test.go`
7. **Check-then-act allocation (no rollback)** — `KVCacheState.AllocateKVBlocks` uses a pre-check gate: it computes the total blocks needed (new blocks + cached blocks leaving the free list) and compares against `countFreeBlocks()` before any state mutation. If insufficient, it returns `false` immediately with zero side effects. Post-pre-check `popFreeBlock() == nil` is a `panic` (INV-4 violation, structurally unreachable in single-threaded DES). This mirrors vLLM's `kv_cache_manager.py:334-336` universal pre-check. If your tier adds mutations before delegating to `gpu.AllocateKVBlocks()`, ensure the inner pre-check sees the updated `FreeBlockCnt` (e.g., `commitCachedBlocks` calls `removeFromFreeList` which decrements `FreeBlockCnt` before the inner call).
8. **`GetCachedBlocks` is a pure query** — it returns cached block IDs without side effects. `CacheHits` are counted by `AllocateKVBlocks` when cached blocks are committed to an allocation. The pre-check accounts for cached blocks with `!InUse` (on the free list) via the `cachedFromFreeList` budget, mirroring vLLM's `num_evictable_blocks`.

Examples:
- See `TieredKVCache` in `sim/kv/tiered.go` for 2-tier GPU+CPU composition
- See `KVCacheState` in `sim/kv/cache.go` for single-tier baseline (also implements `KVStore`)
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

## Adding Fields to TraceV2 Format

To add a new field to TraceV2 CSV output (e.g., observability metadata like `vllm_priority`):

1. **Add field to `TraceRecord`** in `sim/workload/tracev2.go` (place logically near related fields)
2. **Add field to `RequestRecord`** in `cmd/observe.go` (if captured during observation)
3. **Capture the value** in `RealClient.Send()` in `cmd/observe.go` (set field on `RequestRecord`)
4. **Wire through Recorder** in `cmd/observe.go`: Update `RecordRequest()` to copy field from `RequestRecord` to `TraceRecord`
5. **Update ExportTraceV2** in `sim/workload/tracev2.go`:
   - For **optional columns**: scan records to determine if column is needed, conditionally add to header, conditionally write values
   - For **always-present columns**: add to `traceV2Columns` slice and write in row construction
6. **Update LoadTraceV2/parseTraceRecord** in `sim/workload/tracev2.go`:
   - For **optional columns**: detect column presence from CSV header, apply offset to subsequent column indices
   - For **always-present columns**: parse at fixed index, update all downstream indices
7. **Enforce simulation isolation** (if observability-only): Verify `LoadTraceV2Requests` and `LoadTraceV2SessionBlueprints` do NOT read the field into `sim.Request` — add tests for this
8. **Add behavioral tests** covering round-trip, conditional logic, and simulation isolation

**Column position guidelines:**
- Observability metadata: place near related fields (e.g., `vllm_priority` after `slo_class`)
- Timing data: group with other timestamps
- Optional columns: prefer conditional inclusion over always-writing zero values

Examples:
- See PR #1220 (`VLLMPriority`) — optional column, simulation-isolated, conditional on SLOClass presence
- See `FinishReason`/`ErrorMessage` fields — always-present optional strings (empty when not set)
- See `ServerInputTokens` field — observability metadata that differs from `InputTokens`

## Extending Latency Pricing

Latency pricing is not extended in this repository. BLIS takes every step time, memory figure, P/D transfer time, offload tier transfer time and host overhead from `blis-latency-kernel` (adapter `sim/kernelmodel`); there is no `LatencyModel` factory or backend flag to add a case to.

- **A new pricing term, engine feature or model shape** → change `blis-latency-kernel`, release it, and bump the pinned module version in `go.mod`.
- **New or refitted coefficients** → publish them in `blis-registry` and name the coefficient sets in the scenario's `coefficients:` list.
- **A new model, chip, fabric or storage device** → add it to `blis-catalog`.
- **A new scenario or engine-setting field** → change the format in `blis-schemas` first.

On the BLIS side, a new kernel capability usually needs only the adapter in `sim/kernelmodel` and the scenario-to-`SimConfig` wiring in `cmd/`. See [Latency models](../guide/latency-models.md).

## Adding New Batch Formation Strategies

To add a new batch formation strategy (e.g., disaggregated prefill/decode, speculative decoding, continuous batching without preemption):

1. **Implement the `BatchFormation` interface** in `sim/batch_formation.go` (or a new file for complex strategies) — 1 method:
   - `FormBatch(ctx BatchContext) BatchResult` — compose the running batch for the next step
   - The implementation receives `BatchContext` with: RunningBatch, WaitQ, KVCache, token budget, batch size limit, chunked prefill threshold, MaxModelLen (0 = unlimited; implementations should clamp token scheduling to `maxModelLen-1-ProgressIndex` when > 0), simulation time, step count, and ComputedTokens map
   - The implementation MUST update `ctx.ComputedTokens[req.ID]` for each request that receives new tokens (Phase 2 of `Step()` reads this map to advance `ProgressIndex`)
   - The implementation may mutate `WaitQ` (dequeue/prepend) and `KVCache` (allocate/release) during batch formation
   - The implementation MUST NOT schedule events or record metrics — return decisions in `BatchResult`, the Simulator applies them
2. **Register in `NewBatchFormation` factory** in `sim/batch_formation.go`: add a selection branch. The factory signature is `NewBatchFormation(preemptionPolicy string)`. For a new batch formation *strategy* (not just a preemption variant), add a `BatchFormation string` field to `PolicyConfig` and a selection branch in `NewBatchFormation`
3. **Add behavioral tests** — token budget enforcement, batch size limits, KV conservation, preemption behavior (if applicable), FCFS ordering
4. Extension friction: **2 touch points** (implementation + factory registration)

**Note:** Currently only `VLLMBatchFormation` exists (with configurable preemption via `--preemption-policy fcfs|priority`). Adding a second batch formation strategy will also require: (a) a `BatchFormation string` field in `PolicyConfig` or `BatchConfig` (in `sim/config.go`), (b) a CLI flag in `cmd/root.go`, (c) validation in `sim/bundle.go`, (d) selection logic in `NewBatchFormation`. For adding a new *preemption* variant (not a new strategy), add a constant to `batch_formation.go`, a case to the `switch` in `preemptForTokens`, and an entry in `validPreemptionPolicies` in `bundle.go`.

Examples:
- See `VLLMBatchFormation` in `sim/batch_formation.go` for the vLLM FCFS + chunked-prefill + preemption strategy
- See `preemptForTokens` for the KV allocation + eviction loop pattern

## Adding New Quantization Formats

Quantization is stated per pool in the scenario (`engine.quantization`, `engine.cache_dtype`) and priced by `blis-latency-kernel`. A new format is added there (and to `blis-schemas` if the field's allowed values change), not in this repository.

## Adding a New Engine

To add a new autoscaler `Engine` implementation (e.g., a cost-minimizing MIP solver or an OpenEvolve-evolved policy):

1. **Implement the `Engine` interface** in `sim/cluster/engine.go` (or a new file for complex engines) — 1 method:
   - `Optimize(results []AnalyzerResult, inventory GPUInventory) []ScaleDecision`
   - Inputs: one `AnalyzerResult` per model (supply/demand signals + per-variant breakdown) and a `GPUInventory` snapshot (free GPU slots per variant, pre-subtracted for Loading/Active/Draining instances).
   - Output: at most one `ScaleDecision` per model per call. `Delta > 0` = add replicas; `Delta < 0` = remove replicas.
   - **Must not** read `RouterState` or `ModelSignals` directly — only `AnalyzerResult`.

2. **Reuse the shared helpers** when appropriate:
   - `scaleUpN(requiredCapacity float64, vcs []VariantCapacity) int` — exact replica count via `ceil(requiredCapacity / prc)`; fallback to 1 when `prc == 0`.
   - `scaleDownN(spareCapacity float64, vc VariantCapacity) int` — exact replica count via `floor(spareCapacity / prc)`, clamped to `[1, ReplicaCount]`; fallback to 1.
   - `sortedByAscCost`, `sortedByDescCost` — deterministic variant sort (R2: copy-and-sort to avoid mutating caller data).
   - Pass only the selected variant to `scaleUpN` (`vcs[0:1]`, not the full slice) so `perReplicaCapacityForScaleUp` uses the chosen variant's own capacity — not a different active variant's.

3. **Wire the engine** in `sim/cluster/cluster.go` — search for `&UnlimitedEngine{}` in the autoscaler pipeline construction and replace it with your engine (or add a config field + factory once multi-engine selection is implemented in the follow-up wiring PR).

4. **Add behavioral tests** in `sim/cluster/engine_test.go`:
   - Scale-up: correct `Delta`, correct `Variant`, inventory check pass/fail.
   - Scale-down: correct `Delta` from spare capacity, `ReplicaCount` clamp.
   - Edge cases: zero `RequiredCapacity`, zero `SpareCapacity`, all variants inactive, `TPDegree > 1`.
   - Cross-model: multiple `AnalyzerResult` entries — verify decisions for each model independently.

5. Extension friction: **2 touch points** (implementation + wiring via `&UnlimitedEngine{}` in `cluster.go`). For research variants behind a config field, 4 touch points: implementation + `AutoscalerConfig` field + factory function + CLI flag.

Examples:
- See `UnlimitedEngine` in `sim/cluster/engine.go` for a simple inventory-ignoring engine
- See `GreedyEngine` in `sim/cluster/engine.go` for an inventory-respecting greedy engine with exact-N sizing

## Adding New Per-Request Metric Fields

To add a new field to per-request JSON output (appears in `--metrics-path` output):

1. **Add field to `Request`** in `sim/request.go` (runtime state, zero-value safe). When constructing `Request` structs, use `RequestState` typed constants (`StateQueued`, `StateRunning`, `StateCompleted`) — never bare strings.
2. **Add field to `RequestMetrics`** in `sim/metrics_utils.go` (JSON output struct, use `omitempty` for backward compatibility)
3. **Update `NewRequestMetrics()` constructor** in `sim/metrics_utils.go` to propagate the new field from `Request` to `RequestMetrics`
4. **Set the field** at the appropriate event (e.g., `RoutingDecisionEvent` for cluster-level, or completion for computed metrics)
5. **Add behavioral tests** covering multi-instance, single-instance, and standalone boundaries

Examples:
- See `HandledBy` (#181) — set by `RoutingDecisionEvent`, zero-value when used outside cluster pipeline (suppressed from JSON via `omitempty`)
- See `SLOClass`/`TenantID` (PR10) — set during workload generation, propagated at injection
