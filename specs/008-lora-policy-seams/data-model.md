# Phase 1 Data Model — LoRA Placement-Policy Seams

Behavioral entities and their relationships. Concrete Go types/fields are a micro-plan
decision (design-guidelines §3.4); this describes what data crosses each boundary and the
validation rules, per the spec's Key Entities and FR-001…FR-018.

## Entities

### RoutingPolicy (extends existing)
- **Represents**: a named strategy selecting the target instance for a request.
- **Key data observed**: request adapter id; per-instance `RoutingSnapshot` (incl. `ResidentAdapters` membership set); existing routing signals.
- **Produces**: a target instance id (from the non-empty candidate list).
- **Variants**: baseline = existing scorer-composed weighted routing; new = `route-to-holder` (candidate-set restriction to holders, D1).
- **Validation**: name must be registered (else CLI fatal, FR-004). Under route-to-holder, if ≥1 holder exists the target is a holder (INV-PS1); reads no `OutputTokens` (INV-9/L6).

### EvictionPolicy (new seam)
- **Represents**: a named strategy selecting the victim among unpinned resident adapters when a slot is needed.
- **Key data observed**: unpinned resident-adapter set + eviction context (rank/reload-cost, D2).
- **Produces**: a victim id, or "no victim" iff every resident adapter is pinned.
- **Variants**: baseline = `lru` (byte-identical to current); new = `rank/cost-aware` (lowest-reload-cost victim, provisional criterion).
- **Validation**: never selects a pinned adapter (INV-L5); preserves `|resident| ≤ capacity` (INV-L2); deterministic id tie-break; no-victim path preserves INV-8 (waiting request retries once a pin clears).

### CreationPolicy (new seam)
- **Represents**: a named strategy deciding start-of-run residency (`Initial`) and admit-on-miss (`OnResidentMiss`) — one two-entry-point policy (D9).
- **Key data observed**: the adapter→instance assignment + adapter registry (at `Initial`); routed request + instance snapshot (at `OnResidentMiss`).
- **Produces**: the set of adapters resident at t=0 per instance; an admit/defer decision on a miss.
- **Variants**: baseline = `on-demand` (empty seed, always admit — today's behavior, INV-L1); new = `pre-placement` (seed the declared assignment at t=0).
- **Validation**: seeded residency ≤ per-instance capacity (INV-L2, else startup error, INV-PS2); seeded adapters must be registered; pre-placed adapters incur no cold-load latency and no load-count (INV-L3, D4).

### Trigger
- **Represents**: the point at which a policy is invoked.
- **Reactive (implemented)**: `Initial` (construction-time seeding hook — both up-front and deferred node-ready sites), `OnRoute`, `OnResidentMiss`, `OnCapacityPressure`.
- **Periodic (scaffolded, inert)**: `Periodic(Δt)` — type + config only this round, no event scheduled (INV-PS3). Reuses the `ScalingTickEvent` pattern when later activated.
- **Validation**: reactive triggers reuse existing decision points (no new event). `Initial` needs no priority (not an event-loop trigger). `Periodic` config must be a declared, deterministic simulation-time interval.

### StrategyBundle
- **Represents**: a named config binding a {routing, eviction, creation} triple.
- **Resolution**: bundle name → three policy selections; an explicit per-knob override replaces only that knob; unset knobs fall back to baseline defaults (FR-015).
- **Validation**: bundle name and every referenced policy name must be registered (FR-004).

### AdapterInstanceAssignment (new, cluster-scoped)
- **Represents**: the declared placement input consumed by `pre-placement` — per-instance-index list of adapter ids.
- **Home**: `DeploymentConfig` (cluster scope, D3), resolved to per-instance subsets before crossing into instance-local state.
- **Validation**: instance index in range; per-instance count ≤ capacity; adapter ids registered — all checked at startup (INV-PS2).

### EffectiveTriple (provenance)
- **Represents**: the resolved {routing, eviction, creation} selection actually used.
- **Home**: run-level field in `MetricsOutput`, computed once at policy resolution (D6).
- **Validation**: **omitted** from stdout when every seam is at baseline (INV-6 byte-identity, D8); present otherwise; reconstructs the exact policy configuration from the record alone (SC-006).

## Relationships

```
StrategyBundle ──expands to──▶ {RoutingPolicy, EvictionPolicy, CreationPolicy}   (per-knob override)
CreationPolicy(pre-placement) ──consumes──▶ AdapterInstanceAssignment (cluster-scoped, D3)
RoutingPolicy(route-to-holder) ──reads──▶ RoutingSnapshot.ResidentAdapters (Immediate freshness, D7)
EvictionPolicy(rank-aware) ──reads──▶ EvictionContext (rank via AdapterRegistry, D2)
Trigger(Initial) ──invokes──▶ CreationPolicy seeding (both construction sites, D3)
Trigger(OnRoute/OnResidentMiss/OnCapacityPressure) ──invokes──▶ Routing/Creation/Eviction
{selected triple} ──recorded as──▶ EffectiveTriple in MetricsOutput (omitted if all-baseline, D8)
```

## State ownership (no shared mutable state across module boundaries)

| State | Owner | Notes |
|---|---|---|
| Resident-adapter set (per instance) | instance `Simulator` | already owned; recency order internal |
| Eviction context (rank source) | instance `Simulator` | constructed alongside the cost model (D2) |
| Adapter→instance assignment | `DeploymentConfig` (cluster) | resolved to per-instance subset at construction (D3); instance code never sees the full map |
| Effective triple (provenance) | `MetricsOutput` (run-level) | computed once at resolution, not per-event (D6) |
| Policies (routing/eviction/creation) | stateless | registry lookup; no mutable policy state |
