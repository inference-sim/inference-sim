# Phase 0 Research: LoRA Control-Plane Subsystem

**Feature**: `007-lora-control-plane` | **Date**: 2026-07-15

All code citations were read in the worktree on 2026-07-15. The spec's three
spec-level ambiguities were resolved by `/speckit.clarify` (Session 2026-07-15);
this document resolves the two remaining design-doc-level items and records the
technical decisions that ground the plan.

---

## R1. Cold-load gating: concrete mechanism realizing the pre-admission gate

**Decision**: Model the cold load as a **per-instance serialized load event** that
gates the request's entry into a batch. When a request reaches batch formation and
its adapter is not resident on the target instance, the request is held out of the
runnable batch and an adapter-load completion is scheduled at
`now + baseLatency + ceil(adapterBytes / loadBandwidth)`. The adapter becomes
resident (and the request becomes batch-eligible) at that completion time. Only one
adapter loads at a time per instance (serialized); further cold loads queue behind it.

**Rationale**:
- The clarification fixed the *observable* semantics (pre-admission gate; latency in
  TTFT; serialized). This picks the mechanism that produces exactly that.
- It mirrors the Digital Twin's blocking model (`Lat_load(S_A)`, testbed §"Performance
  model"), our only calibrated reference for the adapter axes.
- The latency *shape* reuses the tiered-KV transfer formula
  `baseLatency + ceil(size/bw)` (`sim/kv/tiered.go:319`), keeping one cost idiom.
- Gating **before** batch entry (rather than a per-step stall via
  `ConsumePendingTransferLatency`, `sim/simulator.go:636`) avoids interaction with
  preemption and Phase-1 batch completeness (INV-12) that the source doc §9 flagged,
  and keeps INV-8 (work-conserving) intact: the load event is itself scheduled work,
  so the simulator is not idle.

**Alternatives considered**:
- *Per-step stall (drain `pendingLatency` per step)*: mechanically closest to the KV
  reload path, but charges the delay inside `StepTime` where the request is already
  batched — muddies the cold-vs-warm TTFT signal (SC-004) and complicates preemption.
  Rejected.
- *Non-blocking/prefetch*: loses the cold-start TTFT tail the feature exists to study.
  Rejected (also rejected in clarify Q1).
- *Concurrent (unserialized) loads*: overstates HBM/PCIe bandwidth; the DT serializes.
  Rejected for first cut; a bandwidth-shared model is a future refinement.

**Invariants touched**: INV-5 (`enqueue ≤ schedule` — the gate delays schedule, not
enqueue), INV-8 (load event keeps the queue non-idle), INV-3 (load completion
timestamp ≥ now). No `OutputTokens` read (INV-9).

---

## R2. Adapter memory: static vs dynamic

**Decision**: **Static** per-adapter subtraction from the KV budget in
`CalculateKVBlocks` (`sim/latency/kv_capacity.go`), beside model weights. The reserved
amount is **fixed at startup** as `configured adapter capacity × per-slot footprint`
(the per-slot footprint sized from the **max declared rank**, derived from rank per R3),
so the KV block count is fixed once (matching how BLIS sets the block count today). It is
**not** a running sum over currently-resident adapters (that would be the deferred dynamic
tier, and `CalculateKVBlocks` runs at startup before any adapter has loaded). Adapters
churn within the pre-reserved slots without changing the reservation, preserving INV-4 as
an equality with a constant `adapter-reserved` term.

**Rationale**:
- BLIS's KV budget is a fixed block count set once; a truly dynamic runtime
  KV↔adapter tradeoff needs cross-consumer budget plumbing that does not exist today
  (source doc §9.2). Static is trivial, coarse, and correct for capacity planning.
- Keeps INV-4 a clean conservation equality: `allocated + free + adapter-reserved = total`.

**Alternatives considered**:
- *Dynamic runtime tradeoff* (a second memory tier analogous to `cpuTier`): faithful
  but requires new budget-negotiation machinery. **Deferred** as an explicit follow-on
  (spec Assumptions; design doc flags it). Not in scope for these 7 PRs.

**Invariants touched**: INV-4 (extended with the `adapter-reserved` term).

---

## R3. Adapter footprint & load latency as functions of rank

**Decision**: Both derive from adapter **rank** (single source of truth, clarified
2026-07-15):
- **HBM footprint (bytes)** = `rank · sum_over_target_modules(in_dim + out_dim) ·
  num_layers · dtype_bytes` — the LoRA A/B parameter count (`A ∈ R^{r×in}`, `B ∈ R^{out×r}`
  ⇒ `rank·(in+out)` params per module; no extra factor of 2). A simplified linear
  `bytes = perRankByteCost · rank` is acceptable for the first cut and matches the DT's
  size-indexed table granularity; the exact formula is preferred where the model config
  (`sim.ModelConfig`) already exposes `HiddenDim`/`NumLayers` (as `computeModelWeightBytes`
  does, `sim/latency/kv_capacity.go:256`).
- **Cold-load latency** = `baseLatency + ceil(footprintBytes / loadBandwidth)`, the
  tiered-KV transfer shape. Rank enters through `footprintBytes`.

**Rationale**: One rank→(bytes, latency) derivation avoids the DT's parallel
`served_adapters_sizes` array drifting from a separate byte field, and gives the
calibration import (PR7) a single table to populate.

**Alternatives considered**:
- *Explicit per-adapter byte field on the request/registry*: rejected in clarify Q2
  (rank is the source of truth; a byte field can disagree with rank).
- *Fixed uniform footprint*: rejected — ignores rank-driven size differences that
  placement research depends on.

---

## R4. Adapter identity & registry shape (parity with the Digital Twin)

**Decision**: An adapter is a **pre-declared registry entry** `id → rank`. `id` is a
global opaque string, unique across the simulation, belonging to one base model.
Requests carry `Adapter string` (the id) beside `Request.Model` (`sim/request.go`);
rank is looked up from the registry, never stamped on the request.

**Rationale**: Verified against the DT driver (`~/Projects/blis/lora-control/experiments/dt_driver.py`):
`served = [f"adapter_{i}" ...]` (flat string ids), `served_adapters_sizes = [...]`
(parallel rank array declared once at the manager level), and a single `model=` per
run. Requests reference the adapter by id in a `(in_tok, out_tok, adapter_id)` tuple.
BLIS's `Model`-tag path (`sim/cluster/cluster_event.go` per-model filter) already
carries an analogous opaque tag through workload→request→routing→metrics, so the
adapter id rides the same plumbing with minimal new code.

**Alternatives considered**:
- *`(base_model, name)` compound key*: more robust for multi-base-model clusters but
  more plumbing; unnecessary when ids are globally unique. Deferred.
- *Rank on every request*: rejected (clarify Q2/Q3) — duplicates the registry and can
  disagree across requests for the same adapter.

---

## R5. Compute-overhead term inside `StepTime`

**Decision**: Add a multiplicative adapter-overhead factor to the base step time,
keyed on the unique adapters in the batch and their maximum rank:
`stepTime_with_adapters = baseStepTime(batch) · factor`, where the factor is the DT
overhead term **normalized to the no-adapter baseline**:
`factor = (K6(r_max)·A_B + K7(r_max))/K7(r_max) = 1 + (K6(r_max)/K7(r_max))·A_B` (from DT Eq. 1 `Lat_model = (K4·B + K5)·(K6·A + K7)`, coefficients fitted per rank tier),
`A_B` = count of unique non-empty adapter ids in the batch, rank entering via the fitted
`K6`/`K7` tier. Normalizing by `K7` makes the factor exactly `1.0` at `A_B == 0` **by
construction for any fitted `K7`** (no-op default → INV-6 byte-identity), not only when
`K7` happens to be 1.0.

**Rationale**:
- `LatencyModel.StepTime(batch []*Request)` already receives the batch
  (`sim/latency_model.go`), so reading each request's `Adapter` is the "one net-new
  per-request read" the source doc anticipated — **no interface change** (R13).
- **Rank plumbing**: `Adapter` on the request is an id only (clarification), so rank
  reaches `StepTime` via an **adapter-cost accessor supplied at model construction**
  (threaded from `sim/lora` through `NewLatencyModelFunc`), not a per-request field.
  This keeps the `StepTime(batch []*Request)` signature unchanged (R13) and preserves
  the id-only request; a nil accessor ⇒ `factor = 1.0` (INV-6). Both backends get the
  same accessor (R23). See task T030a.
- Composing as a *relative multiplier* onto BLIS's calibrated base (rather than the
  DT's absolute constants) is the calibration-transfer strategy from spec §Assumptions
  / source doc §6: the DT constants are H100/vLLM-specific and cannot be dropped on
  BLIS's base as absolutes.
- Both backends (`roofline`, `trained-physics`) must apply the identical factor (R23);
  the factor is computed in shared code, not per-backend.

**Alternatives considered**:
- *Additive per-step overhead*: simpler but the DT's fitted form is multiplicative on
  the backbone term; additive would misfit at large batch. Rejected.
- *New `AdapterAwareLatencyModel` interface*: single-impl, violates R13. Rejected.

---

## R6. LoRA-aware scorer & snapshot freshness

**Decision**: Add a `scorerFunc` named `lora-affinity` to the switch in
`sim/routing_scorers.go` and register it in `validScorerNames` (R8). It reads a new
`RoutingSnapshot.ResidentAdapters` field (set of resident adapter ids) and scores an
instance higher when it already holds the request's adapter (warm placement),
min-max normalized like `precise-prefix-cache`. It composes into the weighted routing
profile; it is **not** selected by default (no-op preservation).

**Freshness (R17, INV-7)**: `ResidentAdapters` is populated by `buildRouterState()`
from instance state at snapshot-build time, giving it **Periodic** freshness (default
50ms) like `QueueDepth`/`KVUtilization` — Immediate when `--snapshot-refresh-interval 0`.
The scorer's doc comment MUST declare it reads `ResidentAdapters` at Periodic tier.

**Rationale**: Extending `RoutingSnapshot` (an existing bridge type in `sim/`) keeps
the scorer a Policy Template (≤3 files) and avoids a new interface. The `scorerFunc`
signature `func(req *Request, snapshots []RoutingSnapshot) map[string]float64` already
carries both the request (→ `req.Adapter`) and per-instance snapshots (→ resident set).

**Alternatives considered**:
- *Synchronous per-request resident-set query* (like `InFlightRequests`): higher
  fidelity but couples routing to live instance state and breaks the snapshot
  abstraction; Periodic matches how llm-d's EPP sees adapter state. Rejected for parity.
- *Replacing the default profile*: would break no-op default and INV-6. Rejected.

---

## R7. Determinism & randomness audit

**Decision**: The subsystem introduces **no randomness**. Eviction (LRU), load
ordering (arrival order + serialized per instance), cost terms, and scoring are
deterministic. **No new `PartitionedRNG` subsystem** is declared.

**Rationale**: Per-adapter metric maps must be **key-sorted before output/accumulation**
(R2, INV-6). No-op default (empty adapters, unset capacity) must yield byte-identical
stdout (INV-6, SC-001/006) — guaranteed because every adapter code path is gated on a
non-empty adapter id or configured capacity and reduces to the identity transform when
absent.

---

## Summary of resolved unknowns

| Item | Resolution | Source |
|---|---|---|
| Cold-load gating mechanism | Serialized per-instance load event gating batch entry | R1 |
| Static vs dynamic memory | Static subtraction (dynamic deferred) | R2 |
| Footprint/latency vs rank | Both derived from rank; LoRA A/B formula (linear acceptable first cut) | R3 |
| Adapter identity/registry | Global string id + pre-declared `id → rank` registry | R4, DT driver |
| Compute overhead | Multiplicative `1 + (K6(r_max)/K7(r_max))·A_B` (DT term, per-rank tier, normalized to no-adapter baseline) in `StepTime`, factor=1 at A_B=0 by construction | R5 |
| Scorer + freshness | `lora-affinity` scorerFunc; `ResidentAdapters` at Periodic tier | R6 |
| Randomness | None; no new RNG subsystem | R7 |

**No open NEEDS CLARIFICATION remain.** Ready for Phase 1 design artifacts.
