# Implementation Plan: LoRA Placement-Policy Seams

**Branch**: `008-lora-policy-seams` | **Date**: 2026-07-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/008-lora-policy-seams/spec.md`
**Design doc**: [`docs/plans/2026-07-22-lora-policy-seams-design.md`](../../docs/plans/2026-07-22-lora-policy-seams-design.md) — APPROVED, convergence review PASSED (6 rounds, 0C/0I).

## Summary

Make LoRA adapter **placement policy** a first-class, selectable concern along three
independent, byte-identical-by-default seams — **Routing** (which instance serves a
request), **Eviction** (victim under capacity pressure), **Creation** (start-of-run
residency + admit-on-miss) — plus a `(trigger → policy)` taxonomy (reactive triggers
implemented; a `Periodic` trigger scaffolded and inert), named strategy bundles, and a
run-level effective-policy-triple provenance field. Three seed policies ship:
**route-to-holder** (strict routing), **rank/cost-aware eviction**, **pre-placement**
(declared adapter→instance seeding at t=0). Technical approach and all non-obvious
decisions (D1–D9) are fixed in the approved design doc; this plan records the
Constitution Check, structure, and Phase 0/1 artifacts, and the B-1…B-7 PR sequence
each PR's micro-plan will elaborate.

## Technical Context

**Language/Version**: Go 1.22+ (single implementation language; no CGO/Python — constitution tech-stack constraint)
**Primary Dependencies**: `cobra` (CLI), `logrus` (stderr diagnostics), `gopkg.in/yaml.v3` (strict config), `gonum` (stats)
**Storage**: In-memory only (no external storage); config via YAML + CLI flags
**Testing**: `go test ./...` (table-driven, TDD; behavioral GIVEN/WHEN/THEN); `golangci-lint run ./...` (v2.9.0)
**Target Platform**: CPU-only, deterministic DES; Linux/macOS
**Project Type**: Single project — a Go library (`sim/`, `sim/cluster/`, `sim/lora/`) with a CLI (`cmd/`)
**Performance Goals**: No new performance target. The load-bearing property is **byte-identical no-op** (INV-6/INV-L1): baseline output unchanged. New policies add no measurable latency/memory when unconfigured.
**Constraints**: Determinism (INV-6; no new `PartitionedRNG` subsystem — all seed policies deterministic); run/replay parity (INV-13); resident-set pin/capacity invariant (INV-L2/L5); oracle boundary (INV-9/INV-L6)
**Scale/Scope**: Multi-instance cluster simulation; the feature is a Subsystem Module (Creation) + Backend Swap (Eviction) + Policy Template (Routing), delivered as 7 Small-tier PRs (B-1…B-7)

**Base branch**: Built on `lora-integration` (the assembled LoRA control-plane subsystem, PRs 1–7), where the resident set, cold-load gate, adapter cost model, adapter registry, and `lora-affinity` scorer already exist. All file:line references are resolved at micro-plan time per PR.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design (below).*

| Principle | Status | Notes |
|---|---|---|
| **I. Architecture & Layering** | ✅ PASS | Registries live in `sim/`/`sim/lora/`; new policies register via `init()` (no `sim/` self-import). Dependency direction preserved (`cmd/ → sim/cluster/ → sim/`). **D3 explicitly keeps cluster-topology (adapter→instance assignment) out of instance-local `sim/` code** — the cluster layer resolves per-instance subsets, honoring "cluster state MUST NOT leak into instance-level code." Bridge types (`RoutingSnapshot`) stay in `sim/`. |
| **II. Determinism** | ✅ PASS | No new `PartitionedRNG` subsystem — eviction/creation deterministic by id/ordered-seeding; routing reuses `SubsystemRouter`. INV-6 byte-identity is the primary safety property (first test per PR). Any map iteration in registries sorts keys (R2). CRN caveat documented (design §12). |
| **III. Interface & Module Design** | ✅ PASS | Each seam is a named registry behind a behavioral contract with ≥2 impls (R13): Routing {existing scorers/policies, route-to-holder}, Eviction {lru, rank-aware}, Creation {on-demand, pre-placement}. Query methods pure. Factory validation via `IsValid*()` + unexported maps (R8). No single-impl interface (D2 reuses the existing `AdapterRegistry` rather than widening the frozen `AdapterCost`). |
| **IV. Test-First (BDD/TDD)** | ✅ PASS | Design §11 lists behavioral contracts per seam + companion invariant tests (INV-PS1/PS2/PS3, INV-6/8/9/13, INV-L*), property tests, and the no-op golden written first. Micro-plans convert to failing-first TDD tasks. |
| **V. Error Handling** | ✅ PASS | Unregistered policy/bundle name → CLI `logrus.Fatalf` (FR-004); over-capacity/unregistered/out-of-range pre-placement → startup error (INV-PS2). `sim/` returns errors/panics, never `os.Exit`. Eviction no-victim path preserves INV-8 (no silent drop, R1). |
| **VI. Configuration Discipline** | ⚠️ PASS w/ note | Part B adds the adapter→instance assignment + policy-selection fields at **cluster scope (`DeploymentConfig`)** — the correct home for topology (R16), **not** a new `SimConfig` sub-config. Strict YAML (`KnownFields(true)`, R10), pointer types where zero is meaningful (R9), `Flags().Changed` guard (R18). **Pre-existing tension (NOT introduced here):** the merged LoRA control-plane subsystem already added a 7th `SimConfig` sub-config (`LoRAConfig`), which exceeds Principle VI's "exactly 6"; that amendment (constitution v1.1.0) is tracked against the control-plane work, not this feature. Part B introduces no new `SimConfig` sub-config. |
| **VII. System Invariants** | ✅ PASS | Preserves INV-1/6/8/9/13 and INV-L1/L2/L3/L5/L6/L7; introduces INV-PS1/PS2/PS3 (design §6.4), each with a verification path (§11). INV-7 exception for `ResidentAdapters` under route-to-holder documented (D7). |
| **VIII. Antipattern Prevention** | ✅ PASS | Design explicitly cites R1 (no silent drop), R4 (construction-site audit for new config fields; the D3 shared-constructor `addLiveInstance` gotcha), R8/R9/R10/R16/R18 (config), R13 (≥2 impls), R17 (scorer freshness — D7), R22/R23. No R21 range-shrink risk in the LRU walk. |

**Development-workflow gates:** worktree ✅ (`inthat-lora-seams`); design doc reviewed to convergence ✅; per-PR micro-plan + plan-review + TDD + code-review + verification gate → enforced per B-1…B-7 at implementation time (out of scope for this planning session).

**Gate result: PASS.** No unjustified violations. The one flagged item (Principle VI 6-sub-config count) is a pre-existing condition of the base branch, not a Part B change; recorded in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```text
specs/008-lora-policy-seams/
├── plan.md              # This file
├── spec.md              # Feature spec (/speckit.specify)
├── research.md          # Phase 0 — decision consolidation (D1–D9)
├── data-model.md        # Phase 1 — entities (policies, triggers, bundle, assignment, provenance)
├── quickstart.md        # Phase 1 — run a pre-placement + route-to-holder scenario
├── contracts/           # Phase 1 — registry + config + CLI + metrics contracts
│   ├── seam-registries.md
│   ├── config-schema.md
│   ├── cli-flags.md
│   └── metrics.md
├── checklists/
│   └── requirements.md  # Spec quality checklist (all pass)
└── tasks.md             # Phase 2 (/speckit.tasks — NOT created here)
```

The design doc (`docs/plans/2026-07-22-lora-policy-seams-design.md`) is the authoritative behavioral source; these artifacts refine it toward TDD tasks without restating decisions.

### Source Code (repository root)

```text
sim/
├── routing.go                 # RoutingPolicy iface, WeightedScoring, RouterState (route-to-holder plugs here)
├── routing_scorers.go         # scorer registry (switch → registry, B-1); route-to-holder (B-2)
├── resident_adapter_set.go    # ResidentAdapterSet iface (eviction seam extraction, B-3)
├── simulator.go               # cold-load gate (maybeStartAdapterLoad), t=0 seeding hook site (B-5/B-6)
├── adapter_registry.go        # AdapterRegistry (RankOf) — wired into eviction context (B-3/B-4)
├── config.go                  # LoRAConfig, AdapterSpec (unchanged shape; assignment lives in cluster/)
├── metrics_utils.go           # MetricsOutput — effective-triple provenance field (B-7)
└── lora/
    ├── resident_set.go        # concrete LRU (default eviction policy, byte-identical, B-3)
    ├── eviction/              # NEW: EvictionPolicy registry + lru default + rank-aware (B-3/B-4)
    ├── creation/              # NEW: CreationPolicy registry + on-demand + pre-placement (B-5/B-6)
    └── register.go            # init()-registration of seam defaults

sim/cluster/
├── deployment.go              # DeploymentConfig — adapter→instance assignment + policy selection (D3, B-5)
├── cluster.go                 # instance-construction loop (up-front) — pre-placement resolution (B-5/B-6)
├── infra_lifecycle_event.go   # NodeReadyEvent.Execute — deferred-path seeding caller site (B-6)
├── snapshot.go                # ResidentAdapters freshness — Immediate override for route-to-holder (D7, B-2)
└── cluster_event.go           # (Periodic trigger scaffold reuses ScalingTickEvent pattern, B-7)

cmd/
├── root.go                    # resolvePolicies, --routing/eviction/creation-policy, --lora-bundle flags; INV-13 sync point (B-1..B-7)
└── replay.go                  # replay-side policy wiring (INV-13 parity)

sim/trace/ + tests           # golden no-op baseline; behavioral + invariant + property tests per seam
```

**Structure Decision**: Single-project Go library + CLI. The three seams map to existing modules (Router, resident set, cold-load gate). New sub-packages `sim/lora/eviction` and `sim/lora/creation` host the two new registries; the routing registry extends the existing `sim/routing_scorers.go`. Cluster-scoped placement config lives in `sim/cluster/deployment.go` (D3). This mirrors the merged control-plane subsystem's layout.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Eviction seam ~3–4 files (target ~3) | Rank is unreachable at the eviction site today; an eviction-context wires the existing `AdapterRegistry` in (D2) | Widening the frozen `AdapterCost` interface with a rank method (single-consumer) violates R13; storing rank in resident-set entries couples the LRU data structure to cost semantics |
| Creation seam ~4–5 files (target ~3) | Per-instance adapter placement is a cluster-topology concern with no home in instance-agnostic config; seeding must apply at **both** initial-topology construction sites, gated off the shared `addLiveInstance` autoscaler caller (D3) | A per-instance `LoRAConfig` list structurally cannot target instance i; seeding inside the shared constructor leaks onto autoscaler-scaled instances (R4 shotgun-surgery) |
| Routing strict policy ~3–4 files (target ~3) | route-to-holder needs a narrow cluster-layer `ResidentAdapters` Immediate-freshness override (D7) beyond the pure policy file, else INV-PS1 is unenforceable under the default Periodic freshness | Leaving strictness as a scorer cannot guarantee INV-PS1 (weighted sum only biases) |
| **Pre-existing:** `SimConfig` has 7 sub-configs (Principle VI says 6) | Introduced by the merged LoRA control-plane subsystem (`LoRAConfig`), not this feature | N/A — constitution v1.1.0 amendment tracked against the control-plane work; Part B adds no new sub-config |

All three seam overages are one-time structural costs (paid once per seam, not per policy); adding a further policy afterward returns to the ~2–3-file target. Each is justified in design doc §10.
