# Contract — Configuration Schema

Cluster-scoped placement config (D3) and policy selection. Strict YAML (`KnownFields(true)`, R10);
pointer types where zero is meaningful (R9); no new `SimConfig` sub-config (Principle VI / R16).

## Cluster-scoped (`DeploymentConfig`)
- **Adapter→instance assignment**: a per-instance-index declaration of adapter ids (consumed by `pre-placement`). Sits beside `NodePools` / routing profile — cluster-topology scope, not `LoRAConfig` (D3).
- **Policy selection**: routing / eviction / creation policy names, or a bundle name (with per-knob override). Resolved in the shared `resolvePolicies` path (run + replay, R23 parity).
- **Validation (startup)**: instance index in `[0, NumInstances)`; per-instance assignment count ≤ resident-adapter capacity; every adapter id present in the adapter registry; every policy/bundle name registered. Any failure ⇒ clear startup error (INV-PS2, FR-012, FR-004).

## Bundle table
- A bundle name resolves to a {routing, eviction, creation} triple (data, not code). An explicit per-knob field overrides only that knob; unset knobs → baseline default (FR-015).

## Defaults & no-op
- With no placement config set: routing = existing profile, eviction = `lru`, creation = `on-demand`, no assignment, no `Periodic` — byte-identical to today (INV-6/INV-L1).
- `cmd.Flags().Changed(...)` guard before applying any `defaults.yaml` default (R18).

## Periodic trigger config (scaffold only)
- A declared, deterministic simulation-time interval field, parseable and selectable, but inert this round (no event scheduled, INV-PS3).
- **SUPERSEDED by Spec 3 (2026-08-15):** the trigger is now live and INV-PS3 is replaced by INV-PS3′ — the interval is inert only when the effective creation policy does not implement `sim.PeriodicCreationPolicy`. The text above describes the B-7 round as shipped and is kept as a record of it; the current contract is `DeploymentConfig.LoRAPeriodicIntervalUs`'s doc comment.
