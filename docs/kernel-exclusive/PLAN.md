# Refactoring BLIS onto blis-latency-kernel exclusively

Scope: this worktree only (`kernel-exclusive` branch). No change to `inference-sim/modeling`,
`blis-catalog`, `blis-registry`, `blis-latency-kernel`, `blis-schemas`, or any document
outside this directory.

## The thesis this tests

`blis-latency-kernel` scores 13.67% shape MAPE against AISimulate's 9.41% on 447 points
(`docs/perf-model/latency-kernel-implementation.html` §6.1). That comparison assumes
**resident batch == client concurrency**, because the snapshot states no resident batch.
§6.3 of that document names this as the largest known error source and says a
discrete-event simulator is what supplies the real value.

AISimulate's own method field reads `randomized_synthetic_engine_replay`: it replays a
scheduler. So the gap is not a coefficient gap — the registry's fits are 1.5x-1.9x on
tens of thousands of measured points per part — it is a scheduling gap. BLIS has the
scheduler. Wiring the kernel into BLIS and letting BLIS's scheduler decide the resident
batch is the experiment that tests whether that is the whole gap.

Falsifiable prediction: if the residual is scheduling, BLIS + kernel lands below 9.41%.
If it does not, the residual is in the kernel or the coefficients and this plan is wrong.
Either outcome is reported.

## What "apples-to-apples" requires

The snapshot's own fields define the comparison. Anything not on this list is a
deviation and must be justified in the results:

| Snapshot field | Value(s) | How BLIS must match it |
|---|---|---|
| `latency_scope` | `client_observed` | Measure TPOT as the client sees it, not step time |
| `measurement_scope` | `end_to_end` | Full request lifecycle through the simulator |
| `metrics` | TTFT, TPOT | Score TPOT (what the 447 points carry) |
| `workload` | `1024:1024`, `8192:1024`, `1024:8192` | ISL:OSL exactly, 31/33/19 sweeps |
| `concurrency` | 1..2048, per-sweep subsets | Closed-loop client concurrency, not arrival rate |
| `serving` | `aggregated` | Single pool, no disaggregation |
| `spec_method` | `none` | Speculation off |
| `parallelism` | tp/pp/dp/moe_ep/moe_tp per sweep | From the same 39 scenario files |
| Normalisation | to each sweep's lowest concurrency | Same anchor on both sides |

Closed loop matters: `concurrency` is a client level, so the harness must hold N requests
in flight and let the scheduler decide the resident batch, rather than driving a rate.

## Approach

1. **Link the kernel locally.** A `replace` directive in this worktree's `go.mod` points
   at `blis-latency-kernel` and `blis-schemas` on disk. No vendoring, no copying.
2. **One adapter, no reimplementation.** BLIS's `sim.LatencyModel` is four methods; the
   kernel already answers all four. The adapter translates `[]*sim.Request` to
   `kernel.Batch` and delegates. It computes nothing.
   - `StepTime(batch)` -> `kernel.StepTime(Batch).Overlap`
   - `OutputTokenProcessingTime()` -> `kernel.OutputTokenOverhead()`
   - `PostDecodeFixedOverhead()` -> `kernel.CompletionOverhead()`
   - `QueueingTime(req)` -> `kernel.AdmissionOverhead(promptTokens)`
3. **Exclusive reliance.** The adapter is registered through the existing
   `sim.NewLatencyModelFunc` seam (`sim/latency/register.go`), so roofline and
   trained-physics are not called. They are left on disk untouched; "exclusive" is about
   what runs, and a behavioural test asserts the other two models are never constructed.
4. **Reuse the existing corpus and scenarios.** The 39 scenario files and the 447-point
   corpus in `blis-latency-kernel/testdata/` are read as-is. No new fixtures.
5. **Compare.** Same normalisation, same points, same anchor as `cmd/shape`.

## Tests: behavioural, not structural

Each test states the behaviour it pins and the failure it would catch.

- **Adapter delegates rather than computes.** Two kernels built from scenarios that differ
  only in a coefficient must produce different step times through the adapter. Catches an
  adapter that ignores the kernel.
- **Batch translation preserves the classifier.** A request with `Computed < PromptLen`
  must be priced as prefill and one with `Computed >= PromptLen` as decode. Catches a
  field-mapping error, which would silently price every decode as a prefill.
- **Neither legacy model is constructed on a kernel run.** Catches a fallback path.
- **Monotonicity through the adapter.** Step time non-decreasing in scheduled tokens.
- **Closed-loop concurrency is honoured.** With N in flight, observed in-flight count
  never exceeds N. Catches a harness that drives a rate instead.
- **The comparison is anchored on both sides.** Reproduce AISimulate's published 10.05%
  whole-snapshot shape error from its own per-point data, as `cmd/shape` does. Catches the
  metric-identity error already made once in this project.

Structural checks to avoid: asserting a file contains a string, asserting a type has a
method, counting lines. Those pass while the model is wrong.

## What could make this fail honestly

- BLIS's scheduler may not reproduce the engine the measurements came from (sglang, trtllm,
  vllm across sweeps). BLIS models vLLM.
- The kernel has no Blackwell recurrent coefficients and only H100 recurrent entries.
- `client_observed` TPOT includes queueing the kernel does not model.
- 39 deployments span 4 chips and 3 frameworks; one scheduler cannot match all three.

If the result lands above 9.41%, the report says so and says which of these it is.
