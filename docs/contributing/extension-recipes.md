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

**Important:** For load-based routing, use `snap.EffectiveLoad()` — never compute `QueueDepth + BatchSize + InFlightRequests` inline. This ensures all routing policies use the same formula.

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
7. **Preserve rollback semantics** — `KVCacheState.AllocateKVBlocks` is transactional: on mid-loop failure, `rollbackAllocation()` undoes all mutations (UsedBlockCnt, CacheMisses, CacheHits, RefCount, InUse, free list, HashToBlock, RequestMap). If your tier adds mutations beyond what delegation to `gpu.AllocateKVBlocks()` handles, you must roll those back too. See `cachedBlockMutation` and `newBlockMutation` types in `sim/kv/cache.go`.
8. **`GetCachedBlocks` is a pure query** — it returns cached block IDs without side effects. `CacheHits` are counted by `AllocateKVBlocks` when cached blocks are committed to an allocation (and rolled back on failure). This was fixed in the Phase 3 hardening PR; the previous implementation incremented CacheHits in GetCachedBlocks, causing double-counting in tiered mode.

Examples:
- See `TieredKVCache` in `sim/kv/tiered.go` for 2-tier GPU+CPU composition
- See `KVCacheState` in `sim/kv/cache.go` for single-tier baseline (also implements `KVStore`)
- See `docs/plans/archive/pr12-architectural-predesign.md` for the design decisions behind the tiered architecture

## Adding New Trace Record Types

To add a new trace record type (e.g., `ScaleRecord` for autoscaling events):

1. **Define the record struct** in `sim/trace/record.go` (pure data, no `sim/` dependency)
2. **Add a slice field** to `SimulationTrace` in `sim/trace/trace.go` (e.g., `Scales []ScaleRecord`)
3. **Add a recording method** to `SimulationTrace` (e.g., `RecordScale(ScaleRecord)`)
4. **Hook recording** into the cluster event pipeline (guard with `if cs.trace != nil` for zero-overhead default):
   - For standard routing events: `sim/cluster/cluster_event.go`
   - For PD disaggregation events: `sim/cluster/pd_events.go` (PrefillRoutingEvent and DecodeRoutingEvent). Note: `KVTransferRecord` is recorded in `DecodeRoutingEvent.Execute()` (not `KVTransferStartedEvent`) because `DecodeInstanceID` is only populated at decode routing time.
   - **Pool-filtered snapshots:** PD routing events must pass `filteredSnapshots` (not `state.Snapshots`) to `computeCounterfactual()`. Pool-filtered snapshots contain only pool-member instances; passing full-cluster snapshots would produce candidates from the wrong pool.
5. **Update `Summarize()`** in `sim/trace/summary.go` to aggregate the new record type
6. **Update the `--summarize-trace` output block** in `cmd/root.go` to print the new summary fields (guard with a non-zero count check so they only appear when the feature is active)
7. **Add behavioral tests** in `sim/trace/*_test.go`

**Activation conditions for PD trace records:** PD-specific records (`DisaggregationRecord`, `PrefillRoutingRecord`, `DecodeRoutingRecord`, `KVTransferRecord`) are only emitted when **both** of the following are configured:
- `--trace-level decisions` (or higher) — enables the trace recorder
- `--prefill-instances N --decode-instances M` — enables PD disaggregation pool topology

Setting `--trace-level decisions` alone (without pool flags) produces admission and standard routing records but zero PD records.

Examples:
- See `AdmissionRecord` in `sim/trace/record.go` for a simple record
- See `RoutingRecord` with `CandidateScore` for a record with nested counterfactual data
- See `computeCounterfactual()` in `sim/cluster/counterfactual.go` for derived computation that lives in `sim/cluster/` (not `sim/trace/`) because it needs `sim.RoutingSnapshot`
- See `PrefillRoutingRecord`/`DecodeRoutingRecord` for records with pool-scoped counterfactual candidates

## Adding New Latency Model Backends

To add a new latency estimation backend (e.g., SGLang RadixAttention, TensorRT-LLM, neural surrogate):

1. **Implement the `LatencyModel` interface** in `sim/latency/latency.go` (or a new file in `sim/latency/` for complex models) — 3 methods:
   - `StepTime(batch []*Request) int64` — estimate batch step duration from request states
   - `QueueingTime(req *Request) int64` — estimate arrival-to-queue delay
   - `OutputTokenProcessingTime() int64` — per-token post-processing overhead
2. **Register in `NewLatencyModel` factory** in `sim/latency/latency.go`: add a `case` branch in the `switch hw.Backend` block. The backend string (e.g., `"crossmodel"`) is set by the `--latency-model` CLI flag and stored in `ModelHardwareConfig.Backend`. The factory signature is `NewLatencyModel(LatencyCoeffs, ModelHardwareConfig)`.
3. **Add behavioral tests** in `sim/latency/` — monotonicity (more tokens → longer step time), positive output, boundary cases (empty batch)
4. Extension friction: **2 touch points** (implementation + factory branch)

Examples:
- See `BlackboxLatencyModel` in `sim/latency/latency.go` for a simple stateless model (alpha/beta regression)
- See `RooflineLatencyModel` in `sim/latency/latency.go` for a model that uses hardware config (FLOPs/bandwidth)
- See `CrossModelLatencyModel` in `sim/latency/crossmodel.go` for a physics-informed model that derives step time from HuggingFace architecture features (MoE-aware)

## Adding New Batch Formation Strategies

To add a new batch formation strategy (e.g., disaggregated prefill/decode, speculative decoding, continuous batching without preemption):

1. **Implement the `BatchFormation` interface** in `sim/batch_formation.go` (or a new file for complex strategies) — 1 method:
   - `FormBatch(ctx BatchContext) BatchResult` — compose the running batch for the next step
   - The implementation receives `BatchContext` with: RunningBatch, WaitQ, KVCache, token budget, batch size limit, chunked prefill threshold, simulation time, step count, and ComputedTokens map
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

## Adding a New Disaggregation Decider

To add a new disaggregation decider (e.g., a threshold-based decider that disaggregates only long prefills):

1. **Implement `DisaggregationDecider`** in `sim/disaggregation.go`:
   ```go
   type PrefixThresholdDecider struct{ MinPrefillTokens int }
   func (d *PrefixThresholdDecider) Decide(req *sim.Request) DisaggregationDecision {
       if len(req.InputTokens) >= d.MinPrefillTokens {
           return DisaggregationDecision{Disaggregate: true}
       }
       return DisaggregationDecision{Disaggregate: false}
   }
   ```
2. **Register in `NewDisaggregationDecider` factory** in `sim/disaggregation.go`: add a `case` branch for the new decider name (e.g., `"prefix-threshold"`). The decider name is supplied via `--pd-decider` CLI flag and stored in `DeploymentConfig.PDDecider`.
3. **Add CLI parameter** if the decider needs configuration (e.g., `--pd-prefill-threshold-tokens`). Full wiring:
   - Add a field to `DeploymentConfig` in `sim/cluster/deployment.go`:
     ```go
     PDPrefillThresholdTokens int // minimum input tokens to disaggregate
     ```
   - Add a flag in `cmd/root.go` (alongside the other `--pd-*` flags):
     ```go
     runCmd.Flags().IntVar(&pdPrefillThresholdTokens, "pd-prefill-threshold-tokens", 0,
         "Minimum input token count to trigger disaggregation (used with --pd-decider prefix-threshold)")
     ```
   - Validate and wire in the `runCmd` handler (inside the `if prefillInstances > 0` block):
     ```go
     cfg.PDPrefillThresholdTokens = pdPrefillThresholdTokens
     ```
   - Update the `NewDisaggregationDecider` factory signature to accept the new parameter, and add a `case` branch:
     ```go
     // sim/disaggregation.go — extend factory signature for config-bearing deciders:
     func NewDisaggregationDecider(name string, minPrefillTokens int) DisaggregationDecider {
         ...
         case "prefix-threshold":
             return &PrefixThresholdDecider{MinPrefillTokens: minPrefillTokens}
         ...
     }
     ```
   - Update the call site in `sim/cluster/cluster.go` to pass the new parameter:
     ```go
     cs.disaggregationDecider = sim.NewDisaggregationDecider(config.PDDecider, config.PDPrefillThresholdTokens)
     ```
   - Note: `NewDisaggregationDecider` lives in `sim/` and `DeploymentConfig` in `sim/cluster/`, so pass individual fields directly to avoid import cycles.
4. **Add behavioral tests** — disaggregation decision for short vs long requests, boundary value (exactly at threshold), zero-threshold.
5. Extension friction: **2–3 touch points** (implementation + factory + optional CLI flag).

**Contract:** `Decide()` must be pure — no side effects, no access to cluster state. It receives only the `*sim.Request` (input tokens available pre-routing).

Examples:
- See `NeverDisaggregate` in `sim/disaggregation.go` for the simplest implementation (always returns `DisaggregationDecisionLocal`)
- See `AlwaysDisaggregate` for a decider that always routes to the PD pipeline
- See `DisaggregationDecisionEvent.Execute()` in `sim/cluster/cluster_event.go` to understand how the decision is consumed in the cluster event pipeline

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
