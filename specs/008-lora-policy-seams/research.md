# Phase 0 Research — LoRA Placement-Policy Seams

All `NEEDS CLARIFICATION` items were resolved during the approved design doc's convergence
review; this file consolidates the decisions in Decision/Rationale/Alternatives form. The
authoritative source is [`docs/plans/2026-07-22-lora-policy-seams-design.md`](../../docs/plans/2026-07-22-lora-policy-seams-design.md)
§7 (D1–D9). Two items remain **provisional pending paper confirmation** (tracked in design §14),
noted below — neither blocks planning.

## R1 — Routing strictness mechanism (D1)

- **Decision**: route-to-holder is a routing *policy* that restricts the candidate set to holders
  (when non-empty) before the existing weighted-scoring argmax runs. When no instance holds the
  adapter, it falls back to unconstrained weighted routing (= baseline; first request becomes a
  normal cold-load).
- **Rationale**: strictness is a hard constraint; the weighted sum only biases, so a scorer cannot
  guarantee INV-PS1. Candidate filtering + reuse of existing scoring keeps tie-break/determinism
  identical to today.
- **Alternatives**: very-large scorer weight (still probabilistic — rejected); reject-on-no-holder
  (breaks liveness — rejected); defer/queue until a holder exists (starvation risk — rejected).
- **Provisional (§14)**: the no-holder fallback is confirmed against `Tantawi2025` at micro-plan time.

## R2 — Rank-aware eviction data path (D2)

- **Decision**: introduce an **eviction context** exposing unpinned candidate ids + rank/reload-cost
  at the eviction site, sourced from the already-defined `AdapterRegistry` (which has `RankOf`),
  wired into the instance simulator alongside the existing cost model. LRU ignores it (byte-identical);
  rank-aware consumes it.
- **Rationale**: rank is unreachable at the eviction site today; reusing the existing registry
  interface is additive (R13) and avoids mutating the frozen `AdapterCost` contract. Reuses the
  already-fitted rank→reload-cost model (no new fidelity claim).
- **Alternatives**: widen `AdapterCost` with `RankOf` (single-consumer, R13 — rejected); store rank in
  resident-set entries (couples LRU to cost semantics — rejected).
- **Provisional (§14)**: the exact victim criterion/tie-break confirmed against `Li2025` (Toppings);
  may resolve to "inspired by" rather than "reproduces" if Toppings prescribes only rank-aware routing.

## R3 — Pre-placement configuration scope (D3)

- **Decision**: the adapter→instance assignment lives at **cluster scope (`DeploymentConfig`)**; the
  cluster layer resolves each instance's own subset in the construction loop and hands only that subset
  across the boundary. Seeding runs at **both** initial-topology construction sites (up-front loop +
  deferred node-ready path), gated so it never fires in the shared `addLiveInstance` constructor's
  autoscaler-scale-up caller.
- **Rationale**: placement is inherently cluster-topology; `LoRAConfig` is instance-agnostic and cannot
  target instance i. Mirrors the existing per-instance GPU-type resolution precedent.
- **Alternatives**: per-instance `LoRAConfig` list (can't target instance i — rejected); thread the full
  cross-instance map into instance config (leaks cluster state — rejected); seed inside the shared
  constructor (leaks onto autoscaler instances, R4 — rejected).

## R4 — Pre-placement accounting (D4)

- **Decision**: pre-placed adapters are resident at t=0 with **no** cold-load latency and **no**
  load-count increment (INV-L3 — t=0 seeding is not a charged cold load).
- **Rationale**: static placement's value is avoiding demand cold-loads; counting seeds would mask the
  headline metric (SC-002). t=0 seeding is instantaneous (no request to block).
- **Alternatives**: count seeds as loads (defeats SC-002 — rejected); charge provisioning latency
  (deferred — no analysis question needs it, Banks criterion 5).

## R5 — Periodic trigger (D5)

- **Decision**: define the trigger taxonomy (reactive + `Periodic`) now; scaffold `Periodic` as
  type + config only, scheduling **no** event this round (INV-PS3 trivial). Future activation reuses the
  existing `ScalingTickEvent` self-rescheduling pattern.
- **Rationale**: BLIS already has a periodic clock event (`ScalingTickEvent`, priority 8); the taxonomy
  is cheap now and expensive to retrofit. No event scheduled → no priority slot consumed, no ordering
  guarantee to validate. Priority space is packed 0–9 but the scheme extends to negatives (−1/−2), so a
  future slot is a low-risk extension.
- **Alternatives**: defer the taxonomy (forces later repaint — rejected); implement/schedule an inert
  event now (consumes a priority slot + ordering risk for no benefit — rejected).

## R6 — Provenance (D6 + D8)

- **Decision**: record the effective {routing, eviction, creation} triple as a **run-level** field in
  `MetricsOutput`, computed once at policy resolution (not per-event). **Omitted** from stdout whenever
  every seam is at baseline (all-baseline run) — preserving the INV-6 no-op golden byte-identity.
- **Rationale**: reproducibility needs the resolved triple, not just a bundle name; run-level matches the
  granularity policies are selected at; omission-when-inert mirrors the existing adapter-metrics/HBM
  pattern.
- **Alternatives**: per-request attribution (bloat with a constant — rejected); always-emit a baseline
  triple (breaks INV-6 golden — rejected); rely on the harness manifest only (not self-describing — rejected).

## R7 — Strict-routing signal freshness (D7)

- **Decision**: when route-to-holder is active, read `ResidentAdapters` at **Immediate** freshness,
  overriding the tiered default. This is an INV-7 exception for one field.
- **Rationale**: `ResidentAdapters` is **Periodic by default** (`--snapshot-refresh-interval` defaults to
  50000µs; CLAUDE.md INV-7), so a stale cached view could route a strict request to a non-holder, breaking
  INV-PS1 in the *default* config. In BLIS's single-threaded DES an Immediate read is exact ground truth at
  the routing event. The router constructor's positional/immediate path exists; a small cluster-layer wiring
  hooks it for the strict policy.
- **Alternatives**: reframe INV-PS1 as snapshot-relative (weaker guarantee — rejected); force Immediate
  globally when LoRA is configured (perturbs the soft `lora-affinity` path — rejected).

## R8 — Creation seam shape (D9)

- **Decision**: Creation is one **two-entry-point** seam (`Initial` seeding + `OnResidentMiss` admission),
  not two separate seams.
- **Rationale**: both express one coherent placement stance (static vs. on-demand); splitting them permits
  incoherent mixes (seed-static + admit-and-reload-elsewhere) no experiment wants.
- **Alternatives**: two orthogonal single-method seams (permits incoherent config, doubles surface — rejected).

## Cross-cutting facts (verified against `lora-integration` code during review)

- `AdapterCost.LoadLatency(id)` exposes cost by id only; `AdapterRegistry.RankOf(id)` exists but is not
  wired into the `Simulator`/eviction site today.
- `addLiveInstance` (cluster.go) has exactly two callers: `NodeReadyEvent.Execute` (deferred initial) and
  `DirectActuator.scaleUp` (autoscaler); no caller-discriminator on its signature.
- The deferred node-ready construction path is **not** live-CLI-reachable today (`ProvisionNode` is
  test-only; initial nodes start Ready) — deferred-path tests use the `TestNodeReadyEvent_*` fixture pattern.
- `--snapshot-refresh-interval` defaults to 50000µs → `ResidentAdapters` is Periodic by default.
- Router tie-break uses `SubsystemRouter` RNG (non-nil by default in cluster mode); nil selects positional
  tie-break but no CLI/cluster knob threads nil today (CRN caveat, §12/§14).
- No new `PartitionedRNG` subsystem is introduced.
