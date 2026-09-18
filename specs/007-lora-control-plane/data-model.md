# Phase 1 Data Model: LoRA Control-Plane Subsystem

**Feature**: `007-lora-control-plane` | **Date**: 2026-07-15

Entities described at the behavioral level (per BLIS's abstraction rule; exact Go
struct definitions belong in the per-PR micro plans). Field names are indicative.

---

## Entity: Adapter (registry entry)

The identity + physical characteristics of one LoRA adapter, declared once.

| Attribute | Meaning | Validation |
|---|---|---|
| `ID` | Global opaque string, unique across the simulation | non-empty; unique in registry |
| `Rank` | LoRA rank; sole source of footprint & load-latency | integer > 0 |
| `BaseModel` | The base model this adapter attaches to | must match an instance's served `Model` |

**Derived (not stored):**
- `FootprintBytes(rank)` — HBM bytes (R3); function of rank + model config.
- `LoadLatency(rank)` — `baseLatency + ceil(FootprintBytes/loadBandwidth)` (R1, R3).

**Lifecycle**: declared → (per instance) cold → loading → resident → evicted → cold.

---

## Entity: Adapter Registry

Pre-declared mapping `ID → Adapter`. One per simulation. Single source of truth for
rank; requests reference adapters by `ID` only.

| Attribute | Meaning | Validation |
|---|---|---|
| `entries` | `map[string]Adapter` (key-sorted before any output/accumulation — R2) | keys unique; every referenced request adapter id present |

**Invariants**:
- Every `Request.Adapter` (when non-empty) MUST resolve to a registry entry (else config error, `cmd/`→`Fatalf`; library→`panic`).
- Empty registry + no capacity ⇒ subsystem inert (INV-6, SC-001).

---

## Entity: Request (extended)

Existing `sim.Request` gains one field.

| New attribute | Meaning | Validation / rules |
|---|---|---|
| `Adapter string` | Adapter id (registry key); empty ⇒ base-model-only request | empty is valid (no-op path); if non-empty MUST be in registry |

**Rules**:
- **R4**: every literal construction site of `Request` audited and updated; zero value `""` preserves byte-identity (INV-6).
- **INV-9**: `Adapter` is control-plane-visible (routing/scoring may read it); `OutputTokens` remains execution-only.

---

## Entity: Resident-Adapter Set (per instance)

New per-instance state modeling the finite GPU adapter slots. Reuses the `cpuTier`
hash-keyed LRU pattern (`sim/kv/tiered.go`: `store`/`touch`/`unlink`/`appendToTail`).

| Attribute | Meaning | Validation |
|---|---|---|
| `capacity` | Max resident adapters (from `LoRAConfig`) | integer > 0 when adapters configured; adapters-present + capacity 0 ⇒ config error |
| `resident` | id → recency-ordered entry | `len(resident) ≤ capacity` at all times (**new invariant**) |
| `pinned` | ids required by in-flight requests | pinned ids not evictable |
| `loading` | id currently loading (serialized gate) | at most one per instance |

**State transitions**:
- `touch(id)` — request for resident adapter → move to MRU.
- `admit(id)` — cold request: if `len(resident) == capacity`, `evictLRU()` (skipping pinned) then begin `loading`; on load completion → resident + MRU.
- `evictLRU()` — remove least-recently-used **non-pinned** entry; if all pinned, request waits (bounded by in-flight completion — R19 circuit breaker via capacity bound).

**Invariants**:
- `resident ≤ capacity` (new).
- A `pinned` (in-use) adapter is never evicted (spec edge case; US2 scenario 3).
- Deterministic: eviction order is LRU over arrival-ordered touches (no RNG — R7).

---

## Entity: Adapter Cost Model

Pure, queryable derivation of the three DT terms as deltas onto BLIS's calibrated base.

| Query | Returns | Notes |
|---|---|---|
| `LoadLatency(id)` | µs, ≥ 0 | R1/R3; charged once on cold admit |
| `StepOverheadFactor(batch)` | multiplier ≥ 1.0 | `1 + (K6(r_max)/K7(r_max))·A_B` — `K6/K7` selected by the batch's max-rank tier `r_max` (rank enters here, FR-009); normalized so `= 1.0` at `A_B==0` by construction for any `K7` (INV-6) — R5 |
| `FootprintBytes(id)` | bytes ≥ 0 | R3; summed into KV reservation |

**Rules**: pure query methods (Principle III — no mutation); both latency backends
apply `StepOverheadFactor` identically (R23).

---

## Entity: RoutingSnapshot (extended)

Existing `sim.RoutingSnapshot` gains resident-adapter visibility.

| New attribute | Meaning | Freshness (INV-7, R17) |
|---|---|---|
| `ResidentAdapters` | set of adapter ids resident on the instance | **Periodic** (built by `buildRouterState()`); Immediate when `--snapshot-refresh-interval 0` |

**Rules**: read by the `lora-affinity` scorer only; scorer doc comment declares the
field + freshness tier (R17). Zero value (nil/empty) ⇒ scorer neutral.

---

## Entity: Per-Adapter Metrics (statistics, not state)

Derived outputs surfaced in the metrics JSON. Kept separate from state (DES rule).

| Metric | Meaning |
|---|---|
| `AdapterLoadCount[id]` | cold loads charged per adapter |
| `AdapterEvictionCount[id]` | evictions per adapter |
| `TTFTByAdapter[id]` | TTFT distribution per adapter (esp. cold-start) |
| `ThroughputByAdapter[id]` | tokens/s per adapter |

**Rules**: map keys **sorted before output** (R2, INV-6). Absent when no adapters
configured (no new stdout fields ⇒ INV-6 byte-identity). `RequestMetrics.Adapter`
carries the id for per-request attribution.

---

## Configuration: LoRAConfig (new 7th `SimConfig` sub-config)

See [contracts/config-schema.md](./contracts/config-schema.md). Grouped by module
(R16); strict-parsed (R10); pointer types where zero is meaningful (R9). Justified as
a Subsystem Module addition in plan.md Complexity Tracking.

---

## Relationship summary

```text
AdapterRegistry 1──* Adapter (id → rank, base model)
Request *──1 Adapter        (by id; empty = base-model-only)
Instance 1──1 ResidentAdapterSet ──* Adapter (resident, bounded by capacity)
Instance 1──1 RoutingSnapshot.ResidentAdapters (Periodic view of resident set)
AdapterCostModel ──reads── Adapter.rank → {LoadLatency, StepOverheadFactor, FootprintBytes}
CalculateKVBlocks ──subtracts── capacity × per-slot footprint (max declared rank), fixed at startup (INV-4)
Metrics ──attributes── Request.Adapter → per-adapter aggregates
```
