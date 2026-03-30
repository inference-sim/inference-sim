# Research: Per-Tenant Jain Fairness Index

## Decision 1: Implementation pattern

**Decision**: Follow `ComputePerModelMetrics` / `printPerModelMetrics` exactly.

**Rationale**: `ComputePerModelMetrics` in `sim/cluster/metrics.go` (lines 503–570) solves the identical problem — partition `aggregated.Requests` by a string key, accumulate counts, and return a typed map. The pattern is already tested and lint-clean. Diverging from it would require extra justification.

**Alternatives considered**:
- Embed in `MetricsOutput` JSON: blocked by import cycle (`sim/` cannot import `sim/cluster/`).
- Track tokens in `TenantTracker` at admission time: adds state to the admission-path struct, pollutes its single responsibility (budget enforcement), and is not needed since `aggregated.Requests` already carries `NumDecodeTokens` per request.

---

## Decision 2: Fairness resource = output tokens (not requests)

**Decision**: Jain index computed over per-tenant `NumDecodeTokens` totals.

**Rationale**: Output tokens represent compute actually delivered. Request counts are misleading when request sizes vary significantly (one tenant sending long-context completions vs another sending short ones). The Phase 1B master spec (FR-010, SC-004) and the issue description both specify tokens.

**Alternatives considered**:
- Per-tenant request count: simpler but ignores work heterogeneity; discarded.
- Per-tenant TTFT: a latency metric, not a throughput/fairness metric; discarded.

---

## Decision 3: Scope = completed requests only

**Decision**: Only requests in `aggregated.RequestE2Es` (i.e., requests that reached the execution engine and produced output) contribute to per-tenant token totals.

**Rationale**: Consistent with how global `TokensPerSec` is computed in `SaveResults`. Deferred-horizon-interrupted requests (Phase 1B-1b, `#810`) have zero output tokens and are not in `RequestE2Es` — they are naturally excluded without special-casing.

**Alternatives considered**:
- Include all requests in `aggregated.Requests` (completed + still queued + still running): would add zero-token entries for incomplete requests, artificially deflating per-tenant token averages and biasing the Jain index downward.

---

## Decision 4: Where to source per-tenant token data

**Decision**: Iterate `aggregated.RequestE2Es` (for the set of completed request IDs), look up `aggregated.Requests[id].TenantID` and `aggregated.Requests[id].NumDecodeTokens`.

**Rationale**: This is exactly how `ComputePerModelMetrics` sources its data (lines 521–529). `RequestE2Es` is the authoritative set of completed requests; `Requests` carries the metadata. Using the same two-map join is DRY.

**Alternatives considered**:
- Iterate `aggregated.Requests` directly: includes incomplete requests (still queued/running); see Decision 3 above for why this is wrong.

---

## Decision 5: Output format

**Decision**: Plaintext `=== Per-Tenant Metrics ===` section printed to stdout, following `printPerSLOMetrics` and `printPerModelMetrics`.

**Rationale**: Consistent with existing output style. JSON embedding is blocked by import cycle (Decision 1). No user-visible format change for single-tenant workloads (section is absent when no TenantIDs — FR-005).

**Printed format**:
```
=== Per-Tenant Metrics ===
  alice: requests=50, tokens=12500
  bob:   requests=50, tokens=12480
  Jain Fairness Index: 0.9999
```

Tenants listed in lexicographic order (R2/INV-6). Jain index printed after the per-tenant table. No-op when map is nil or len == 0.
