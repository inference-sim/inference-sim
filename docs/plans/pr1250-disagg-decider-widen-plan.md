# PR1250 Plan — Widen DisaggregationDecider for llm-d parity

**Goal:** Widen `DisaggregationDecider` to receive the selected decode endpoint + a per-endpoint cache query closure, introduce `RequestView` (narrow read-only projection — no `OutputTokens`) for structural INV-9 enforcement, and reconnect the prefix-threshold decider to read the *selected decode pod's* cache instead of a cluster-wide virtual LRU.

**Source:** Issue #1250. Parity target: llm-d-inference-scheduler @ `e52311b7` — `prefix-based-pd-decider`.

**Closes:** #1250

## Behavioral Contracts

- **BEH-1 — Decider receives selected decode endpoint.** GIVEN a request arriving to `DisaggregationDecisionEvent.Execute`, WHEN decode routing has selected an instance, THEN `DisaggregationDecider.Decide(view, ctx)` is called with `ctx.DecodeInstanceID()` equal to the selected decode instance.
- **BEH-2 — Per-endpoint cache state drives the decision.** GIVEN a `PrefixBasedPDDecider(threshold, blockSize)` and a request whose tokens have N cached blocks on the selected decode pod, WHEN `Decide` is called, THEN `Disaggregate == (len(InputTokens) - N*blockSize > threshold)`.
- **BEH-3 — `threshold == 0` disables disaggregation.** GIVEN `PrefixBasedPDDecider(0, blockSize)`, WHEN `Decide` is called on any request, THEN `Disaggregate == false`.
- **BEH-5 — Structural INV-9 enforcement.** `RequestView` has no `OutputTokens` field; attempts to read it fail to compile. `DisaggregationContext` is passed by value with unexported fields; deciders cannot mutate cluster state.
- **BEH-6 — Freshness parity.** `ctx.DecodeCacheQuery(tokens)` returns the same count that `precise-prefix-cache` would observe for the same `(instance, tokens)` — both consume `CachedSnapshotProvider.BuildCacheQueryFn()`.

## Tasks

1. **Add `RequestView` + `DisaggregationContext` types** in `sim/disaggregation.go`. `RequestView` fields: `ID, InputTokens, MaxOutputLen, SLOClass, Priority, ArrivalTime`. No `OutputTokens`. Constructor `NewRequestView(*Request) RequestView`. `DisaggregationContext` fields unexported: `decodeInstanceID string`, `decodeCacheQuery func([]int) int`. Constructor `NewDisaggregationContext(id string, fn func([]int) int)`. Methods `DecodeInstanceID()`, `DecodeCacheQuery([]int) int` (returns 0 when closure nil).

2. **Change `DisaggregationDecider.Decide` signature** to `Decide(view RequestView, ctx DisaggregationContext) DisaggregationDecision`. Update `NeverDisaggregate`, `AlwaysDisaggregate`.

3. **Split PrefixThresholdDecider into two:**
   - `PrefixBasedPDDecider` — stateless; queries `ctx.DecodeCacheQuery`; implements BEH-3 (threshold==0 short-circuit). No `DisaggregationObserver` needed.
   - `GlobalPrefixThresholdDecider` — retains the old `globalVirtualInstance` LRU + `ObserveRouting` for counterfactual baseline.
   - Keep `NewPrefixThresholdDecider(threshold, blockSize)` as a thin alias constructor returning `*PrefixBasedPDDecider` (behavior change: now per-pod, not global).

4. **Register names** in `sim/bundle.go`: add `"prefix-based-pd-decider"`, `"global-prefix-threshold"`. Keep `"prefix-threshold"` (now aliases to `prefix-based-pd-decider`; behavior change).

5. **Update `sim/cluster/cluster.go` constructor wiring** — factory calls based on `PDDecider` name.

6. **Update `DisaggregationDecisionEvent.Execute`** in `sim/cluster/cluster_event.go` — build `DisaggregationContext` from `cs.cacheQueryFn[decodeDecision.TargetInstance]` and pass `NewRequestView(e.request)` plus ctx.

7. **Update tests**: `sim/disaggregation_test.go` (new signature), `sim/cluster/disaggregation_test.go` (integration-level tests cover BEH-1, BEH-2, BEH-3, BEH-6).

8. **Lint + build + test** — `go build ./...`, `go test ./...`, `golangci-lint run ./...`.

## Sanity Checklist

- [ ] `RequestView` has no `OutputTokens` field.
- [ ] `DisaggregationContext` fields are unexported.
- [ ] `PrefixBasedPDDecider` does not import/use `PrefixCacheIndex`.
- [ ] `threshold == 0` → `Disaggregate = false` short-circuit exists.
- [ ] `DisaggregationDecisionEvent.Execute` wires `cs.cacheQueryFn[decodeDecision.TargetInstance]` into the context.
- [ ] `INV-DECIDER-1` test: `Decide` invoked exactly once, after decode routing, before prefill routing / KV transfer.
- [ ] All in-tree deciders (Never, Always, PrefixBasedPD, GlobalPrefixThreshold) satisfy the new interface.
- [ ] `go test ./... -count=1` passes.
- [ ] `golangci-lint run ./...` passes.
