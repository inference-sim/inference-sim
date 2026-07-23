# Contract: CLI Flags

**Feature**: `007-lora-control-plane` | Commands: `blis run`, `blis replay`

New flags mirror `LoRAConfig`. All optional; **absence ⇒ no-op default** (INV-6).
Each flag checked via `cmd.Flags().Changed(...)` before any `defaults.yaml` value is
applied (R18). Config file (`--config`) and flags compose; flags win when set.

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--lora-adapter-capacity` | int | unset (inert) | Per-instance resident adapter slots |
| `--lora-load-base-latency-us` | float | from defaults | Cold-load fixed latency |
| `--lora-load-bandwidth-bytes-us` | float | from defaults | Cold-load bandwidth (>0) |
| `--lora-footprint-bytes-per-rank` | float | from defaults | Footprint per rank unit |
| `--lora-scorer-weight` | float | 0 (not in profile) | Weight of `lora-affinity` in routing profile |

Registry (`adapters`) **and the per-rank compute-overhead coefficients (`step_overhead_tiers`, a rank→{k6,k7} map) are config-file only** (structured values; not scalar flags — a flat flag cannot express a per-rank tier map).

### Contract (GIVEN/WHEN/THEN)

- **GIVEN** no `--lora-*` flags and no `lora:` config **WHEN** `blis run --model X` **THEN** stdout is byte-identical to the pre-feature build (INV-6, SC-001, SC-006).
- **GIVEN** `--lora-adapter-capacity 0` with a workload that assigns adapters **WHEN** `blis run` **THEN** `logrus.Fatalf` with a clear message (edge case).
- **GIVEN** identical flags + seed on two runs **WHEN** `blis run` twice **THEN** byte-identical stdout (INV-6).
- **GIVEN** a trace exported with adapters via `--trace-output` **WHEN** `blis replay` with identical flags **THEN** identical per-request metrics (INV-13 run/replay parity).
- **Unsupported-in-replay** LoRA features (none anticipated) MUST `logrus.Fatalf` at startup, never silently degrade (INV-13).
