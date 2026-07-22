# Contract — CLI Flags

New flags on `blis run` (and `blis replay`, for INV-13 parity). Names are indicative; finalized
at micro-plan time. All default to baseline so the no-op golden is byte-identical (INV-6).

| Flag | Purpose | Default | Notes |
|---|---|---|---|
| `--routing-policy` / existing `--routing-scorers` | select routing policy (incl. `route-to-holder`) | existing profile | route-to-holder pins `ResidentAdapters` Immediate freshness (D7) |
| `--eviction-policy` | select eviction policy (`lru` \| `rank-aware`) | `lru` | B-3/B-4 |
| `--creation-policy` | select creation policy (`on-demand` \| `pre-placement`) | `on-demand` | B-5/B-6 |
| `--lora-adapter-placement` (or in `--lora-config`/deployment YAML) | adapter→instance assignment | none | cluster-scoped (D3); startup-validated (INV-PS2) |
| `--lora-bundle` | select a named strategy bundle (expands to a triple) | none | per-knob flags override (FR-015) |
| `--lora-periodic-interval` | declare the (inert) periodic trigger interval | unset | scaffold only, no effect this round (INV-PS3) |

## Rules
- Unknown policy/bundle name ⇒ `logrus.Fatalf` listing valid names for that seam (FR-004, Principle V).
- Every numeric flag validated for zero/negative/NaN/Inf at the CLI (R3).
- `cmd.Flags().Changed(...)` checked before applying `defaults.yaml` (R18).
- `run` and `replay` resolve policies through the same shared path (R23); the new selections join the **INV-13 sync-point** field set, with a `logrus.Fatalf` fail-fast for any selection replay cannot honor (B-7).
- No CLI knob yet exposes RNG-free (positional) router tie-breaking; strict CRN validity for paired experiments is a tracked §14 follow-up, not enabled by these flags.
