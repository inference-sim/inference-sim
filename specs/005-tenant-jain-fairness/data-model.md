# Data Model: Per-Tenant Jain Fairness Index

## New Types

### `TenantMetrics` (`sim/cluster/metrics.go`)

Holds post-simulation aggregates for a single tenant.

| Field | Type | Description |
|-------|------|-------------|
| `TenantID` | `string` | Tenant identifier (non-empty; matches request `TenantID`). |
| `CompletedRequests` | `int` | Count of completed requests attributed to this tenant. |
| `TotalTokensServed` | `int` | Sum of `NumDecodeTokens` across all completed requests for this tenant. |

**Validation rules**:
- `TenantID` is always non-empty (the function only creates entries for non-empty IDs).
- `CompletedRequests ≥ 0`, `TotalTokensServed ≥ 0`.
- Zero values are valid (a tenant could complete requests with zero decode tokens — edge case, not an error).

**Relationships**:
- Produced by `ComputePerTenantMetrics(*sim.Metrics) map[string]*TenantMetrics`.
- The Jain fairness index is computed separately from the map values, not stored in `TenantMetrics` itself (consistent with how `ComputePerModelMetrics` handles cluster-level aggregates).

---

## New Functions

### `ComputePerTenantMetrics` (`sim/cluster/metrics.go`)

```
ComputePerTenantMetrics(aggregated *sim.Metrics) map[string]*TenantMetrics
```

**Inputs**: `*sim.Metrics` — the aggregated cluster-level metrics after simulation.

**Outputs**: `map[string]*TenantMetrics` — keyed by TenantID. Returns `nil` when no completed request has a non-empty TenantID (zero-value safe, no section printed).

**Algorithm**:
1. Iterate `aggregated.RequestE2Es` (authoritative set of completed request IDs).
2. For each ID, look up `aggregated.Requests[id]`. Skip if not found or if `TenantID == ""`.
3. Accumulate `CompletedRequests++` and `TotalTokensServed += NumDecodeTokens` per tenant.
4. Return `nil` if no entries accumulated; otherwise return the map.

**Determinism**: Uses `sortedKeys` (already in `metrics.go`) when iterating during print, not during accumulation. Accumulation is deterministic because it only sums integers.

---

### `printPerTenantMetrics` (`cmd/root.go`)

```
printPerTenantMetrics(w io.Writer, perTenantMetrics map[string]*TenantMetrics)
```

**Inputs**: writer (stdout in production), map from `ComputePerTenantMetrics`.

**Outputs**: Formatted text section to `w`. No-op when map is nil or empty.

**Algorithm**:
1. Guard: return immediately if `len(perTenantMetrics) == 0`.
2. Print `=== Per-Tenant Metrics ===` header.
3. Collect and sort keys (R2).
4. For each tenant (sorted): print `  <id>: requests=<n>, tokens=<n>`.
5. Build `map[string]float64` of `TenantID → float64(TotalTokensServed)`.
6. Compute `jain := cluster.JainFairnessIndex(tokenMap)`.
7. Print `  Jain Fairness Index: <value>` (4 decimal places).

---

## Existing Types Used (no changes)

| Type | Location | Fields read |
|------|----------|-------------|
| `sim.Metrics` | `sim/metrics.go` | `RequestE2Es map[string]float64`, `Requests map[string]RequestMetrics` |
| `sim.RequestMetrics` | `sim/metrics_utils.go` | `TenantID string`, `NumDecodeTokens int` |
| `JainFairnessIndex` | `sim/cluster/metrics.go:382` | Takes `map[string]float64` (tenant → tokens), already implemented |

---

## State Transitions

No new state. This is a pure post-simulation aggregation — all inputs are already materialized in `aggregated.Requests` and `aggregated.RequestE2Es` before `ComputePerTenantMetrics` is called.

---

## INV-1 Impact

None. Per-tenant metrics are read-only aggregates computed after simulation ends. They do not modify request counts or introduce new accounting buckets. INV-1 accounting is unaffected.
