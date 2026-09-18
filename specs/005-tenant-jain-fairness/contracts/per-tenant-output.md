# Contract: Per-Tenant Metrics Output

**Type**: CLI output contract (stdout section)
**Produced by**: `printPerTenantMetrics` in `cmd/root.go`
**Consumed by**: operators reading simulation output; Phase 1D hypothesis experiments

---

## Presence Contract

| Condition | Output |
|-----------|--------|
| Any completed request has a non-empty `TenantID` | Section printed |
| All completed requests have empty `TenantID` | Section absent (no header, no data, no blank line) |
| No completed requests at all | Section absent |

---

## Format Contract

```
=== Per-Tenant Metrics ===
  <tenant_id_1>: requests=<n>, tokens=<n>
  <tenant_id_2>: requests=<n>, tokens=<n>
  ...
  Jain Fairness Index: <jain_value>
```

**Rules**:
- Tenants listed in lexicographically ascending order (deterministic; R2/INV-6).
- `<jain_value>` formatted to 4 decimal places (e.g., `0.9999`).
- Section ends after the Jain line — no trailing blank line.
- Exactly 2 spaces of indentation for each per-tenant line and the Jain line.

---

## Value Contract

| Field | Guarantee |
|-------|-----------|
| `requests` | Count of completed requests from this tenant (non-negative integer). |
| `tokens` | Sum of `NumDecodeTokens` from completed requests for this tenant (non-negative integer). |
| Jain index | Value in [1/N, 1.0] computed over per-tenant token totals. Single-tenant → 1.0. All-zero → 1.0. |
| Jain accuracy | Within 2% relative error of `JainFairnessIndex(tokenMap)` called on the same data. |

---

## Example Output — Balanced Two-Tenant Workload

```
=== Per-Tenant Metrics ===
  alice: requests=100, tokens=25000
  bob:   requests=100, tokens=25000
  Jain Fairness Index: 1.0000
```

## Example Output — Skewed Workload (10× imbalance)

```
=== Per-Tenant Metrics ===
  alice: requests=91, tokens=45500
  bob:   requests=9,  tokens=4500
  Jain Fairness Index: 0.6694
```

## Example Output — No Tenant Labels (section absent)

```
=== Simulation Metrics ===
{ ... }
=== KV Cache Metrics ===
...
=== Per-Model Metrics ===
...
[no per-tenant section]
```

---

## Placement in stdout

The section is emitted immediately after `=== Per-Model Metrics ===` and before `=== PD Metrics ===` (when present). This matches the position of equivalent per-X sections in the output pipeline.

**Calling sites**:
- `cmd/root.go`: after `printPerModelMetrics(os.Stdout, perModelMetrics)`
- `cmd/replay.go`: after `printPerModelMetrics(os.Stdout, perModelMetrics)`
