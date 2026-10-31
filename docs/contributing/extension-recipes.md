# BLIS Extension Recipes

Step-by-step guides for extending BLIS. Each recipe lists the exact files to touch, the order, and examples to follow.

## Routing, Admission and Flow Control

BLIS has no routing, admission or flow-control code of its own: every cluster run uses the llm-d router embedded in-process ([Routing](../guide/routing.md)). New routing, admission or flow-control behavior belongs in [llm-d-router](https://github.com/llm-d/llm-d-router), as a plugin or configuration, not in this repository.

A new llm-d plugin type becomes usable in BLIS once the fork has audited it for hosting and BLIS's signal policy allows it (the supported list in `sim/llmd/support.go`, passed to the router as `embed.Config.Allow`): in the fork, read every clock read, goroutine, blocking wait, random draw and map iteration it makes and route the decision-relevant ones through the hosting runtime; in BLIS, check that every signal it consumes is produced by the simulated engines, allow it, and add a test in `sim/llmd`. Until then it is refused at startup with its reason. See [The llm-d router fork](llm-d-router-fork.md).

## Adding New Instance Schedulers

The instance scheduler is vLLM's, translated in blis-vllm (`core/sched`), so a change there is a blis-vllm change, released and repinned in `go.mod` ([cross-repo changes](cross-repo-changes.md)).

## Extending KV Cache Tiers

The KV cache is vLLM's, translated in blis-vllm (`core/kv`), so a change there is a blis-vllm change, released and repinned in `go.mod` ([cross-repo changes](cross-repo-changes.md)). Tier transfer *time* is priced by `blis-latency-kernel` from a catalog `device_class`.

## Adding New Trace Record Types

To add a new trace record type (e.g., `ScaleRecord` for autoscaling events):

1. **Define the record struct** in `sim/trace/record.go` (pure data, no `sim/` dependency)
2. **Add a slice field** to `SimulationTrace` in `sim/trace/trace.go` (e.g., `Scales []ScaleRecord`)
3. **Add a recording method** to `SimulationTrace` (e.g., `RecordScale(ScaleRecord)`)
4. **Hook recording** into the cluster pipeline (e.g. `sim/cluster/gateway.go` for router decisions, `sim/cluster/cluster_event.go` for scaling events; guard with `if cs.trace != nil` for zero-overhead default)
5. **Update `Summarize()`** in `sim/trace/summary.go` to aggregate the new record type
6. **Add behavioral tests** in `sim/trace/*_test.go`

Examples:
- See `AdmissionRecord` in `sim/trace/record.go` for a simple record
- See `RoutingRecord` with `CandidateScore` for a record with nested counterfactual data
- See `counterfactual()` in `sim/cluster/gateway.go` for derived computation that lives in `sim/cluster/` (not `sim/trace/`): it ranks the scores the router's primary scheduling profile gave every endpoint

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

Latency pricing is not extended in this repository. BLIS takes every step time, memory figure, P/D transfer time, offload tier transfer time and host overhead from `blis-latency-kernel` (adapter blis-vllm `compat/kernelmodel`); there is no `LatencyModel` factory or backend flag to add a case to.

- **A new pricing term, engine feature or model shape** → change `blis-latency-kernel`, release it, and bump the pinned module version in `go.mod`.
- **New or refitted coefficients** → publish them in `blis-registry` and name the coefficient sets in the scenario's `coefficients:` list.
- **A new model, chip, fabric or storage device** → add it to `blis-catalog`.
- **A new scenario or engine-setting field** → change the format in `blis-schemas` first.

On the BLIS side, a new kernel capability usually needs only the adapter in blis-vllm `compat/kernelmodel` and the scenario-to-`SimConfig` wiring (blis-vllm `compat/build`, called from `cmd/engine_build.go`). See [Latency models](../guide/latency-models.md).

## Adding New Batch Formation Strategies

Batch formation is vLLM's scheduler, translated in blis-vllm (`core/sched`), so a change there is a blis-vllm change, released and repinned in `go.mod` ([cross-repo changes](cross-repo-changes.md)).

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
   - `scaleDownN(spareCapacity float64, vc VariantCapacity, modelReplicas int) int` — exact replica count via `floor(spareCapacity / prc)`, clamped to `[1, ReplicaCount]` and to `modelReplicas-1` so the model keeps one replica; fallback to 1. Returns 0 when nothing may be removed; emit no decision then (INV-19).
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

1. **Add field to `Request`** in blis-vllm `compat/engine/request.go` (runtime state, zero-value safe; `sim.Request` is an alias of it). When constructing `Request` structs, use `RequestState` typed constants (`StateQueued`, `StateRunning`, `StateCompleted`) — never bare strings.
2. **Add field to `RequestMetrics`** in `sim/metrics_utils.go` (JSON output struct, use `omitempty` for backward compatibility)
3. **Update `NewRequestMetrics()` constructor** in `sim/metrics_utils.go` to propagate the new field from `Request` to `RequestMetrics`
4. **Set the field** at the appropriate point (e.g., dispatch in `sim/cluster/gateway.go` for cluster-level, or completion for computed metrics)
5. **Add behavioral tests** covering multi-instance, single-instance, and standalone boundaries

Examples:
- See `HandledBy` (#181) — derived from the instance a request was dispatched to, zero-value outside the cluster pipeline (suppressed from JSON via `omitempty`)
- See `SLOClass`/`TenantID` (PR10) — set during workload generation, propagated at injection
