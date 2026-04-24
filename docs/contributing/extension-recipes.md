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

## Adding New Latency Model Backends

To add a new latency estimation backend (e.g., SGLang RadixAttention, TensorRT-LLM, neural surrogate):

1. **Implement the `LatencyModel` interface** in `sim/latency/latency.go` (or a new file in `sim/latency/` for complex models) — 4 methods:
   - `StepTime(batch []*Request) int64` — estimate batch step duration from request states
   - `QueueingTime(req *Request) int64` — estimate arrival-to-queue delay
   - `OutputTokenProcessingTime() int64` — per-token post-processing overhead
   - `PostDecodeFixedOverhead() int64` — fixed per-request completion overhead (return 0 if not applicable)
   - **All `float64 → int64` conversions MUST use `clampToInt64(v)` (defined in `sim/latency/latency.go`).** Direct `int64(v)` casts on float64 values are undefined behavior in Go when the value is out of range. `clampToInt64` handles NaN and positive overflow correctly.
2. **Register the backend name** in `sim/bundle.go`: add `"your-backend": true` to `validLatencyBackends` map.
3. **Register in `NewLatencyModel` factory** in `sim/latency/latency.go`: add a `case` branch in the `switch hw.Backend` block. The backend string (e.g., `"trained-physics"`) is set by the `--latency-model` CLI flag and stored in `ModelHardwareConfig.Backend`. The factory signature is `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`.
4. **Add CLI wiring** (if needed) in `cmd/root.go`: add a loading block for your backend's coefficients from `defaults.yaml`. If your backend needs a custom defaults section, add a struct to `cmd/default_config.go`.
5. **Add behavioral tests** in `sim/latency/` — monotonicity (more tokens → longer step time), positive output, boundary cases (empty batch)
6. Extension friction: **3-5 touch points** (implementation + bundle map + factory branch; optionally CLI wiring + defaults struct)

Examples:
- See `RooflineLatencyModel` in `sim/latency/latency.go` for a simple stateless analytical model (FLOPs/bandwidth roofline)
- See `TrainedPhysicsModel` in `sim/latency/trained_physics_model.go` for a physics-informed model with roofline basis functions, learned corrections, and MoE-aware overhead modeling

## Adding New Batch Formation Strategies

To add a new batch formation strategy (e.g., disaggregated prefill/decode, speculative decoding, continuous batching without preemption):

1. **Implement the `BatchFormation` interface** in `sim/batch_formation.go` (or a new file for complex strategies) — 1 method:
   - `FormBatch(ctx BatchContext) BatchResult` — compose the running batch for the next step
   - The implementation receives `BatchContext` with: RunningBatch, WaitQ, KVCache, token budget, batch size limit, chunked prefill threshold, MaxModelLen (0 = unlimited; implementations should clamp token scheduling to `maxModelLen-1-ProgressIndex` when > 0), simulation time, step count, and ComputedTokens map
   - The implementation MUST update `ctx.ComputedTokens[req.ID]` for each request that receives new tokens (Phase 2 of `Step()` reads this map to advance `ProgressIndex`)
   - The implementation may mutate `WaitQ` (dequeue/prepend) and `KVCache` (allocate/release) during batch formation
   - The implementation MUST NOT schedule events or record metrics — return decisions in `BatchResult`, the Simulator applies them
2. **Register in `NewBatchFormation` factory** in `sim/batch_formation.go`: add a selection branch. The factory signature is `NewBatchFormation()` — a future PR will add a strategy selection parameter (e.g., a string field in `PolicyConfig` or `BatchConfig`)
3. **Add behavioral tests** — token budget enforcement, batch size limits, KV conservation, preemption behavior (if applicable), FCFS ordering
4. Extension friction: **2 touch points** (implementation + factory registration)

**Note:** Currently only `VLLMBatchFormation` exists. Adding a second strategy will also require: (a) a `BatchFormation string` field in `PolicyConfig` or `BatchConfig` (in `sim/config.go`), (b) a CLI flag in `cmd/root.go`, (c) validation in `sim/bundle.go`, (d) selection logic in `NewBatchFormation`.

Examples:
- See `VLLMBatchFormation` in `sim/batch_formation.go` for the vLLM FCFS + chunked-prefill + preemption strategy
- See `preemptForTokens` for the KV allocation + eviction loop pattern

## Adding New Quantization Formats

To add support for a new quantization format (e.g., GGUF, HQQ, Marlin):

1. **Add `quantization_config` parsing** in `sim/latency/config.go` inside `ParseHuggingFaceConfig()`. The three-tier detection order:
   - **Tier 1 — `quantization_config`**: Add a new `else if` branch after the existing `compressed-tensors` case (~line 240). Extract the weight bit-width from the format's config structure and set `weightBytesPerParam = bits / 8.0`. Use case-insensitive matching for `quant_method` (`strings.EqualFold`).
   - **Tier 2 — Model name conventions**: If the format has recognizable naming patterns (like `w4a16` or `FP8`), add a regex to the compiled patterns (`reWxAy`, `reFP8Name`) or add a new pattern near line 283. Update `InferWeightBytesFromModelName()` accordingly.
   - **Tier 3 — `torch_dtype` fallback**: No changes needed — this is automatic via `BytesPerParam`.

2. **Add tests** in `sim/latency/config_test.go`:
   - A `TestParseHuggingFaceConfig_YourFormat_*` test with a synthetic `config.json` containing the new `quantization_config` structure
   - Verify `WeightBytesPerParam` is set correctly
   - If adding name-based detection, add cases to `TestInferWeightBytesFromModelName`

3. **No changes needed to roofline/KV capacity code** — they already use `EffectiveWeightBytesPerParam()` which automatically picks up the new format's weight precision.

4. Extension friction: **1-2 touch points** (config parsing + optional name regex)

Examples:
- See the GPTQ/AWQ `bits` extraction (~line 229) for formats with a top-level `bits` field
- See the `compressed-tensors` branch (~line 240) for formats with nested config structures
- See `InferWeightBytesFromModelName()` for regex-based name pattern detection

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
