# Contract: Configuration Schema

**Feature**: `007-lora-control-plane`

## `LoRAConfig` — new module sub-config on `SimConfig`

Grouped by module (R16), strict-parsed (R10, `yaml.KnownFields(true)`), pointer types
where zero is a meaningful user value (R9). Every numeric validated for zero/negative
(R3) at both CLI flags and library constructor.

```yaml
lora:
  # Per-instance adapter slot capacity. Pointer: 0 is meaningful (=> adapters
  # forbidden). nil/absent => subsystem inert (no-op default).
  adapter_capacity: 8            # *int; > 0 when adapters used
  # Cold-load cost shape (deltas onto calibrated base).
  load_base_latency_us: 1500.0   # *float64; >= 0
  load_bandwidth_bytes_us: 2.0e6 # *float64; > 0 (R11 divisor guard)
  # Per-step compute-overhead coefficients, keyed by adapter rank tier (DT Eq.1
  # (K6·A + K7), fitted per rank). The factor uses the batch's MAX-rank tier so
  # a higher max rank yields a longer step (FR-009). Rank-blind (flat) coefficients
  # are NOT sufficient — they cannot track the DT's rank-dependent throughput.
  step_overhead_tiers:           # map: rank -> {k6, k7}; k6 *float64 >= 0, k7 *float64 > 0
    8:  {k6: 0.02,  k7: 1.0}     #   (k7 is the per-tier normalization denominator)
    16: {k6: 0.035, k7: 1.0}
    32: {k6: 0.06,  k7: 1.0}
  # Adapter footprint derivation.
  footprint_bytes_per_rank: 2.0e6 # *float64; > 0 (linear first-cut; R3)
  # Pre-declared adapter registry (id -> rank). Empty => inert.
  adapters:
    - id: adapter_0
      rank: 8
    - id: adapter_1
      rank: 16
```

### Validation contract (GIVEN/WHEN/THEN)

- **GIVEN** `adapters` non-empty **AND** `adapter_capacity` is `0` **WHEN** config is validated **THEN** startup fails with a clear error (`cmd/`→`logrus.Fatalf`; `sim/` constructor→`panic`). *(spec edge case: adapters present but zero capacity)*
- **GIVEN** any adapter with `rank <= 0` **WHEN** validated **THEN** startup fails (R3).
- **GIVEN** `load_bandwidth_bytes_us <= 0` **WHEN** validated **THEN** startup fails (R11 — divisor guard).
- **GIVEN** a `Request.Adapter` / workload adapter id absent from `adapters` **WHEN** validated **THEN** startup fails (registry completeness).
- **GIVEN** `lora:` absent entirely **WHEN** the simulation runs **THEN** behavior and output are byte-identical to pre-feature (INV-6, SC-001).
- **GIVEN** a user-set flag **WHEN** applying `defaults.yaml` **THEN** the flag is honored (`Flags().Changed`, R18).

## Workload spec additions (`ClientSpec` / `CohortSpec`)

```yaml
clients:
  - id: c0
    model: llama-3.1-8b-instruct
    adapter: adapter_0      # NEW: adapter id (registry key). omitempty => base-model-only
cohorts:
  - id: h0
    model: qwen-2.5-7b-instruct
    adapter: adapter_1      # NEW: same semantics
```

- **GIVEN** `adapter` omitted **THEN** generated requests have `Request.Adapter == ""` (no-op path).
- **GIVEN** `adapter` set **THEN** every generated request carries that id; it MUST be a registry key and its base model MUST equal the client/cohort `model`.
