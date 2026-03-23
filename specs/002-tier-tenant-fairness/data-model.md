# Data Model: Phase 1B — Service Tiers & Tenant Fairness

## Entities

### SLO Tier Priority (existing, extended)

Canonical mapping from `Request.SLOClass` string to integer priority. Defined in `sim/admission.go`.

| SLOClass string | Priority int | Admission behavior |
|----------------|-------------|-------------------|
| `"critical"`   | 4 | Never shed by tier policy |
| `"standard"`   | 3 | Admitted at default threshold; shed only under extreme overload |
| `"sheddable"`  | 2 | Shed first when overload threshold exceeded |
| `"batch"`      | 1 | Bypasses tier-shed; routed to deferred queue when cluster busy |
| `"background"` | 0 | Bypasses tier-shed; routed to deferred queue when cluster busy |
| `""` (empty)   | 3 | Treated as Standard (backward compatibility) |

**Invariants:**
- `sloTierPriority()` never reads `req.OutputTokens` (INV-9)
- Result is a pure function of the class string — no side effects

---

### TierShedAdmission (new struct, `sim/admission.go`)

An `AdmissionPolicy` that gates requests by tier priority under overload.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `OverloadThreshold` | `int` | `0` | Max effective load before shedding; 0 = shed when any instance has load > 0 |
| `MinAdmitPriority` | `int` | `3` | Minimum tier priority admitted under overload (3 = Standard and above) |

**State transitions**: Stateless — all decisions computed from `RouterState` at call time.

**Zero-value behavior**: `OverloadThreshold=0, MinAdmitPriority=0` admits everything (same as `AlwaysAdmit`).

---

### DeferredQueue (new field on ClusterSimulator, `sim/cluster/cluster.go`)

A slice of `*sim.Request` holding Batch and Background requests awaiting idle capacity.

| Property | Value |
|----------|-------|
| Type | `[]*sim.Request` |
| Capacity | Unbounded (simulation horizon terminates growth) |
| Promotion trigger | `isBusy() == false` after any cluster event is processed |
| Terminal accounting | Requests in queue at horizon counted as `DeferredHorizonInterrupted` |

**State transitions for a deferred request:**

```
Arrived → [Batch/Background + cluster busy] → DeferredQueue
DeferredQueue → [cluster becomes idle] → ClusterArrivalEvent (re-injected)
ClusterArrivalEvent → normal admission → routing → instance → completed
DeferredQueue → [horizon reached] → DeferredHorizonInterrupted (INV-1)
```

---

### TenantTracker (new struct, `sim/cluster/tenant.go`)

Tracks per-tenant in-flight request counts for fair-share enforcement.

| Field | Type | Meaning |
|-------|------|---------|
| `budgets` | `map[string]float64` | tenantID → fraction of `totalCapacity` (0 = unlimited) |
| `inFlight` | `map[string]int` | tenantID → current in-flight count |
| `totalCapacity` | `int` | cluster max in-flight slots (sum of instance KV blocks or NumInstances × batch size) |

**Methods and their contracts:**

| Method | Contract |
|--------|----------|
| `IsOverBudget(tenantID)` | Returns `true` iff budget configured for tenant AND `inFlight[tenantID] > budget * totalCapacity`; always `false` for empty tenantID |
| `OnStart(tenantID)` | Increments `inFlight[tenantID]` by 1; no-op for empty tenantID |
| `OnComplete(tenantID)` | Decrements `inFlight[tenantID]` by 1, floor 0; no-op for empty tenantID |

**Zero-value behavior**: `TenantTracker{budgets: nil}` — `IsOverBudget` always returns `false`.

---

### TenantMetrics (new struct, `sim/cluster/metrics.go`)

Per-tenant output in the simulation metrics JSON.

| Field | JSON key | Meaning |
|-------|----------|---------|
| `TenantID` | `tenant_id` | Tenant identifier |
| `CompletedRequests` | `completed_requests` | Count of completed requests for this tenant |
| `TotalOutputTokens` | `total_output_tokens` | Sum of decode tokens across completed requests |

**Cluster-level**: Jain fairness index is computed over `TotalOutputTokens` across all tenants and printed alongside the per-tenant table (not per-tenant). Computed by existing `JainFairnessIndex()`.

---

## Config Extensions

### `AdmissionConfig` (in `sim/bundle.go`)

```yaml
admission:
  policy: tier-shed
  tier_shed_threshold: 10      # overload kicks in when max effective load > 10
  tier_shed_min_priority: 3    # shed tiers below Standard (3) under overload
```

### `DeploymentConfig` (in `sim/cluster/deployment.go`)

```yaml
tenant_budgets:
  "tenant-a": 0.40   # tenant-a may hold up to 40% of cluster capacity
  "tenant-b": 0.60   # tenant-b may hold up to 60%
```

Zero-value (`tenant_budgets` absent) = all tenants unlimited.

---

## Invariants

| Invariant | Formulation |
|-----------|-------------|
| INV-1 (extended) | `injected == completed + running + queued + shed + dropped + deferred_horizon_interrupted` |
| INV-9 | `sloTierPriority()` and `IsOverBudget()` never read `req.OutputTokens` |
| INV-6 | `TenantBudgets: nil` produces byte-identical output to pre-Phase-1B baseline |
| Monotonic shedding | Under load ramp: `shed(Background) ≥ shed(Batch) ≥ shed(Sheddable) ≥ shed(Standard) ≥ shed(Critical)` |
