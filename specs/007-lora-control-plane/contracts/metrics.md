# Contract: Per-Adapter Metrics

**Feature**: `007-lora-control-plane`
**Extends**: `RequestMetrics` (per-request `Adapter`) + aggregate `MetricsOutput`.

## Per-request

| Field | Meaning |
|---|---|
| `RequestMetrics.Adapter` | adapter id serving this request (`""` = base model) |

## Aggregate (stdout JSON)

Emitted **only when adapters are configured** — absent otherwise, so the no-op default
adds no stdout fields (INV-6, SC-001).

```json
{
  "adapters": {
    "adapter_0": { "load_count": 3, "eviction_count": 1,
                   "ttft_p50_us": 8200, "ttft_p99_us": 41000,
                   "throughput_tok_per_s": 512.4 },
    "adapter_1": { "load_count": 5, "eviction_count": 4, "...": "..." }
  }
}
```

### Contract (GIVEN/WHEN/THEN)

- **GIVEN** no adapters configured **WHEN** metrics are emitted **THEN** no `adapters` key appears and stdout is byte-identical to pre-feature (INV-6, SC-001).
- **GIVEN** adapters configured **WHEN** metrics are emitted **THEN** map keys are **sorted** before serialization (R2, INV-6 determinism).
- **GIVEN** a workload with two adapters **WHEN** the run completes **THEN** `ttft_*` and `throughput_tok_per_s` are reported per adapter (US1 scenario 1).
- **GIVEN** a mixed workload **WHEN** metrics are emitted **THEN** base-model requests are attributed to no adapter; adapter requests to their id (US1 scenario 3).
- **Conservation (INV-1)**: `Σ per-adapter completed + base-model completed + …` reconciles with the global request-conservation identity — adapters partition, not duplicate, request accounting.
- **State vs statistics (DES rule)**: these are derived statistics; they MUST NOT be read back into any control-plane decision.
