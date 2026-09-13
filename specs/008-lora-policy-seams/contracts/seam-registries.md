# Contract — Seam Registries

Behavioral contracts for the three named policy registries. Go signatures are finalized at
micro-plan time against the real call sites; this fixes the observable behavior.

## Routing registry (extends `sim/routing_scorers.go`)
- **Selection**: name → routing policy/scorer. Adding a policy = one registered entry + one validation-name entry (no edits to other policies).
- **Default**: existing scorer-composed weighted routing — byte-identical to today.
- **`route-to-holder` (B-2)**: given the request adapter and per-instance snapshots, restrict candidates to holders when ≥1 holder exists, then select via the existing weighted scoring; when no holder exists, fall back to unconstrained weighted routing.
- **Contract tests**: (a) ≥1 holder ⇒ target is a holder (INV-PS1), under both freshness modes incl. the default Periodic scenario (D7 Immediate override); (b) no holder ⇒ baseline fallback; (c) deterministic given inputs + RNG state; (d) property test ≥100 random holder-configs.

## Eviction registry (`sim/lora/eviction`, extracted from hardcoded LRU — Backend Swap)
- **Selection**: name → eviction policy. Baseline `lru` registered by default.
- **Phase-A gate (B-3)**: extract the seam; `lru` default is byte-identical (all existing tests pass, no behavior change).
- **`rank/cost-aware` (B-4)**: given unpinned candidates + eviction context (rank/reload-cost), return the lowest-reload-cost victim (provisional criterion, §14); deterministic id tie-break.
- **Contract tests**: (a) full set, one unpinned ⇒ only that one evictable (any policy); (b) differing ranks ⇒ rank-aware picks declared victim, monotonic in reload-cost (mirrors 2026-07-15 rank-sensitivity gate); (c) all pinned ⇒ no victim, waiting request runs once a pin clears (INV-L5/INV-8 no-deadlock); never evicts a pinned adapter under any policy.

## Creation registry (`sim/lora/creation` — Subsystem Module)
- **Selection**: name → creation policy (two entry points: `Initial`, `OnResidentMiss`). Baseline `on-demand` registered by default.
- **`on-demand` default**: empty seed, always admit — today's behavior (INV-L1).
- **`pre-placement` (B-6)**: `Initial` seeds the declared per-instance assignment as resident at t=0 (no load latency, no load-count, INV-L3); `OnResidentMiss` admits on-demand for non-seeded adapters.
- **Contract tests**: (a) valid assignment ⇒ resident at t=0, zero cold-loads (SC-002); (b) over-capacity / unregistered / out-of-range index ⇒ startup error (INV-PS2); (c) deferred-construction fixture (`TestNodeReadyEvent_*`) ⇒ deferred initial-topology instance is seeded, autoscaler-scaled instance is not (D3).

## Registration & validation (all three)
- Registration via `init()` in `sim/lora` (or the routing file) — no `sim/` self-import (Principle I).
- Validation maps unexported; exposed via `IsValid*()` accessors (R8).
- Unknown policy/bundle name ⇒ CLI `logrus.Fatalf` listing valid names (FR-004, Principle V).
- No-op default: with no placement options set, or all seams at baseline, output is byte-identical (INV-6/INV-L1) — the first test written per PR (Principle IV).
