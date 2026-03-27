# Research: Phase 1B — Service Tiers & Tenant Fairness

**Status**: Complete — no unknowns. All decisions resolved by reading the codebase directly.

## Key Findings

### What already exists (no work needed)

| Thing | Location | Note |
|-------|----------|------|
| `Request.SLOClass` string field | `sim/request.go:54` | "critical"/"standard"/"sheddable"/"batch"/"background"/""  |
| `Request.TenantID` string field | `sim/request.go:53` | empty = legacy, zero-value safe |
| `RequestMetrics.TenantID` | `sim/metrics_utils.go:23` | already propagated to per-request JSON output |
| `JainFairnessIndex()` | `sim/cluster/metrics.go:378` | already implemented, just needs to be called |
| `PerSLOClass` metrics | `sim/cluster/metrics.go:94` | per-class TTFT/E2E already computed |
| `ComputePerModelMetrics()` pattern | `sim/cluster/metrics.go:499` | template for `ComputePerTenantMetrics()` |
| `printPerModelMetrics()` pattern | `cmd/root.go:1247` | template for `printPerTenantMetrics()` |

### Design Decisions

**D-1: Overload signal for TierShedAdmission**
- Decision: Max `EffectiveLoad()` (= `QueueDepth + BatchSize + InFlightRequests`) across all `RouterState.Snapshots`, compared against a configurable `OverloadThreshold int` (absolute count).
- Rationale: `RouterState` is already passed to `Admit()`; no new wiring needed. `EffectiveLoad()` already exists on `RoutingSnapshot`. Threshold of 0 = "shed Sheddable whenever any instance has queued work" (sensible default).
- Alternative rejected: KV utilization — lags behind queue depth, harder to configure.

**D-2: How TierShedAdmission handles Batch/Background**
- Decision: TierShedAdmission (#809) passes Batch and Background through (admits them). The deferred queue (#810) intercepts them in `AdmissionDecisionEvent.Execute()` *before* the admission policy call, when system is busy.
- Rationale: Clean separation of concerns. #809 only changes the admission policy. #810 only changes the event handler. Neither PR touches the other's code.

**D-3: Deferred queue promotion trigger**
- Decision: At the end of each cluster event processing tick, check `len(cs.deferredQueue) > 0 && totalEffectiveLoad() == 0`. If so, inject all deferred requests as `ClusterArrivalEvents` into the cluster event queue.
- Rationale: `totalEffectiveLoad() == 0` is the correct definition of "idle" — no queued or running requests on any instance.

**D-4: Tenant budget enforcement location**
- Decision: In `AdmissionDecisionEvent.Execute()`, after the admission policy runs and returns `admitted=true`, check `cs.tenantTracker.IsOverBudget(req.TenantID)`. If over budget AND tier < Standard, override to rejected.
- Rationale: Keeps `AdmissionPolicy` interface stateless. Tenant state is cluster-level, not policy-level. Matches pattern of `rejectedRequests` counter.

**D-5: Tenant budget window**
- Decision: Instantaneous in-flight count (`inFlight map[string]int`) only — no time window, no token tracking.
- Rationale: In-flight count is already maintained by `cs.inFlightRequests`; per-tenant variant is straightforward. Token tracking adds complexity without changing correctness for the Phase 1B invariant test.
- Cross-ref FR-012: spec.md FR-012 requires that per-tenant accounting "MUST reset or decay between observation windows." Instantaneous in-flight count satisfies this — the count drops to zero naturally as requests complete, so no explicit window reset is needed. Each measurement reflects only requests actively in flight at that moment.

**D-6: Per-tenant metrics source**
- Decision: Walk `aggregated.Requests` (the `map[string]RequestMetrics`) and group by `TenantID`. Already contains `NumDecodeTokens` for token totals. Matches how `ComputePerSLODistributions` works.
- Rationale: Zero additional state during simulation run; computed post-hoc like all other cluster metrics.
