# Tasks: LoRA Placement-Policy Seams

**Input**: Design documents from `specs/008-lora-policy-seams/` + [`docs/plans/2026-07-22-lora-policy-seams-design.md`](../../docs/plans/2026-07-22-lora-policy-seams-design.md)
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅

**Tests**: REQUIRED — BLIS Constitution Principle IV (Test-First, NON-NEGOTIABLE). Every PR writes its behavioral contract / no-op golden test FIRST (must fail before implementation).

**Organization**: Grouped by the **7-PR roadmap (B-1…B-7)** from design §13 — the BLIS-native decomposition. Each PR is independently mergeable, no-op-safe (byte-identical default), and becomes one sub-issue under the epic. Each PR gets its **own micro-plan + convergence plan-review + code-review + verification gate** at implementation time (per constitution Development Workflow); the tasks here are PR-scoped seeds, not a substitute for those micro-plans. User-story coverage is labeled per PR; **US4 (byte-identical no-op default) is cross-cutting — its golden test is the first task of every PR.**

**Story map**: US1 Static-placement reproduction (B-1, B-2, B-5, B-6) · US2 Eviction ablation (B-3, B-4) · US3 Bundle selection (B-7) · US4 No-op default (all PRs) · US5 Periodic scaffold (B-7).

## Format: `[ID] [P?] [Story] Description with file path`

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Confirm the base and the safety harness before any seam work.

- [ ] T001 Confirm worktree `inference-sim-lora-seams` on branch `008-lora-policy-seams` is based on `lora-integration` and `go build ./... && go test ./... && golangci-lint run ./...` are green (baseline sanity)
- [ ] T002 [P] Capture/confirm the pre-feature no-op golden baseline used by the INV-6/INV-L1 byte-identity test (adapter-blind `blis run --seed 42`), recording the regeneration command (R12)

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The one shared safety property every subsequent PR must preserve.

**⚠️ CRITICAL**: The no-op byte-identity harness must exist before any seam PR merges.

- [ ] T003 [US4] Establish/verify the byte-identical no-op golden test (INV-6/INV-L1) as a companion invariant test (R7), asserting adapter-blind stdout matches the committed baseline; this test is re-run unchanged after every seam PR

**Checkpoint**: No-op safety net in place — seam PRs may begin.

---

## Phase 3: PR B-1 — Routing switch → registry (Policy Template infra) [US1, US4]

**Goal**: Convert `sim/routing_scorers.go`'s scorer `switch` into a named registry so adding a routing policy is a localized additive change; register all existing scorers/policies as defaults, byte-identical.

**Independent Test**: All existing routing tests pass unchanged; the no-op golden holds; a newly registered dummy name resolves via the registry.

- [ ] T004 [US4] Write failing byte-identity + existing-routing-behavior contract tests (registry must reproduce current selection exactly) in `sim/routing_scorers_test.go`
- [ ] T005 [US1] Convert the scorer `switch` to a registry (name → constructor) with `init()`-based registration; keep `validScorerNames` unexported, expose via `IsValid*()` (R8); no `sim/` self-import (Principle I) — `sim/routing_scorers.go`
- [ ] T006 [US1] Register all existing scorers/policies as defaults so `DefaultScorerConfigs()` and `newScorerWithObserver` behavior is byte-identical
- [ ] T007 [US4] Confirm no-op golden (T003) + full routing suite pass; run verification gate

**Checkpoint**: Routing is a registry; behavior unchanged.

---

## Phase 4: PR B-2 — route-to-holder strict routing policy + D7 freshness [US1]

**Goal**: Add the `route-to-holder` policy (candidate-set restriction to holders, D1) with the Immediate-freshness override for `ResidentAdapters` (D7), enforcing INV-PS1.

**Independent Test**: given ≥1 holder, request never routed to a non-holder (under default Periodic freshness, via D7 override); no holder → baseline fallback.

- [ ] T008 [US1] Write failing GIVEN/WHEN/THEN contract tests for INV-PS1 (holder-exists ⇒ holder selected; no-holder ⇒ unconstrained fallback) incl. a default-Periodic-freshness stale-snapshot scenario, plus a ≥100-config property test — `sim/routing_scorers_test.go` / `sim/cluster/*_test.go`
- [ ] T009 [P] [US1] Implement `route-to-holder` as a candidate-set-restricting routing policy reusing the existing weighted scoring among holders — `sim/routing_scorers.go` (+ registration)
- [ ] T010 [US1] Wire the D7 cluster-layer Immediate-freshness override for `ResidentAdapters` when route-to-holder is active — `sim/cluster/snapshot.go` / `buildRouterState`
- [ ] T011 [US1] Add `--routing-policy route-to-holder` (or scorer-name) CLI wiring in shared `resolvePolicies` (run + replay, R23); reject unknown names (FR-004, R3) — `cmd/root.go`, `cmd/replay.go`
- [ ] T012 [US4] Confirm no-op golden holds (route-to-holder inactive ⇒ byte-identical); verification gate

**Checkpoint**: Strict routing works and honors INV-PS1 under the default freshness.

---

## Phase 5: PR B-3 — Eviction seam extraction + lru default (Backend Swap Phase A) [US2, US4]

**Goal**: Extract an `EvictionPolicy` seam from the hardcoded LRU; register `lru` as the byte-identical default; introduce the eviction context wiring the `AdapterRegistry` (rank) into the instance simulator (D2).

**Independent Test** (§5.4 Phase-A gate): all existing tests pass, no behavior change, factory returns `lru` by default.

- [ ] T013 [US4] Write failing tests asserting `lru` default is byte-identical (Phase-A gate) + INV-L2/L5 pin-safety + INV-8 no-deadlock (all-pinned ⇒ waiting request runs once a pin clears) — `sim/lora/eviction/*_test.go`, `sim/simulator_test.go`
- [ ] T014 [US2] Define the `EvictionPolicy` seam + registry (`lru` default via `init()`) in `sim/lora/eviction/`; route the single eviction call site (`maybeStartAdapterLoad`) through it — `sim/lora/eviction/`, `sim/simulator.go`
- [ ] T015 [US2] Introduce the eviction context exposing unpinned candidates + rank/reload-cost, sourced from the `AdapterRegistry` wired into the instance simulator (D2) — `sim/adapter_registry.go` wiring, `sim/simulator.go`
- [ ] T016 [US4] Confirm no-op golden + resident-set suite pass; verification gate

**Checkpoint**: Eviction is a seam; `lru` default byte-identical; rank reachable at the eviction site.

---

## Phase 6: PR B-4 — rank/cost-aware eviction policy (Backend Swap Phase B) [US2]

**Goal**: Add the `rank/cost-aware` eviction policy consuming the eviction context (provisional victim criterion, D2/§14).

**Independent Test**: in a scenario where LRU-order and rank/cost-order provably disagree, the victim tracks the reload-cost criterion (monotonic), never a pinned adapter.

- [x] T017 [US2] Write failing rank-sensitivity contract test (skewed popularity + differing ranks ⇒ victim = lowest-reload-cost unpinned, monotonic; not merely "differs from LRU"; deterministic id tie-break) — `sim/lora/eviction/*_test.go`
- [x] T018 [P] [US2] Implement `rank/cost-aware` eviction policy + registration — `sim/lora/eviction/`
- [x] T019 [US2] Add `--eviction-policy rank-aware` CLI wiring (run + replay); reject unknown names — `cmd/root.go`, `cmd/replay.go`
- [x] T020 [US4] Confirm no-op golden holds; verification gate

**Checkpoint**: Eviction ablation (US2) runnable end-to-end.

---

## Phase 7: PR B-5 — CreationPolicy seam + on-demand default + cluster-scoped placement config (Subsystem Module) [US1, US4]

**Goal**: Introduce the `CreationPolicy` seam (two entry points `Initial`/`OnResidentMiss`, D9) with `on-demand` as the byte-identical default; add the cluster-scoped adapter→instance assignment config resolved per-instance (D3).

**Independent Test**: `on-demand` default seeds nothing and always admits (today's behavior); the assignment config parses/validates but with no `pre-placement` selected changes nothing.

- [ ] T021 [US4] Write failing tests: `on-demand` default byte-identical (INV-L1); assignment config strict-parse (R10) + startup validation for over-capacity/unregistered/out-of-range index (INV-PS2) — `sim/lora/creation/*_test.go`, `sim/cluster/deployment_test.go`
- [ ] T022 [US1] Define the `CreationPolicy` seam + registry (`on-demand` default via `init()`); seat the `Initial` + `OnResidentMiss` triggers at the existing construction/cold-load sites — `sim/lora/creation/`, `sim/simulator.go`
- [ ] T023 [US1] Add the cluster-scoped adapter→instance assignment field to `DeploymentConfig`; resolve each instance's subset in the construction loop (D3), never handing the full map into instance-local `sim/` code (Principle I) — `sim/cluster/deployment.go`, `sim/cluster/cluster.go`
- [ ] T024 [US4] Confirm no-op golden holds; verification gate

**Checkpoint**: Creation is a seam; `on-demand` default byte-identical; placement config plumbed.

---

## Phase 8: PR B-6 — pre-placement creation policy (t=0 seeding) [US1]

**Goal**: Add the `pre-placement` policy seeding the declared assignment as resident at t=0 (no load latency, no load-count, INV-L3/D4), seeded at **both** initial-topology construction sites, never on the shared constructor's autoscaler caller (D3).

**Independent Test**: pre-placed adapters resident at t=0 with zero cold-loads; 100% of their requests served by a holder (SC-002); deferred-construction fixture seeds the deferred initial-topology instance but not an autoscaler-scaled one.

- [ ] T025 [US1] Write failing tests: pre-placed adapters resident at t=0 with zero load-count (INV-L3/SC-002); pre-placement under deferred construction via `TestNodeReadyEvent_*` fixture (D3); pre-placement + `--model-autoscaler-interval-us` preserves INV-PS2/INV-1/INV-L2 with scaled-in instance unseeded — `sim/lora/creation/*_test.go`, `sim/cluster/cluster_test.go`
- [ ] T026 [US1] Implement `pre-placement` policy (`Initial` seeding) + registration; seed at the up-front construction loop AND the deferred `NodeReadyEvent.Execute` caller site, gated off the shared `addLiveInstance` autoscaler caller (D3) — `sim/lora/creation/`, `sim/cluster/cluster.go`, `sim/cluster/infra_lifecycle_event.go`
- [ ] T027 [US1] Add `--creation-policy pre-placement` + assignment CLI/YAML wiring (run + replay); reject unknown names — `cmd/root.go`, `cmd/replay.go`
- [ ] T028 [US4] Confirm no-op golden holds; verification gate

**Checkpoint**: Static-placement reproduction (US1) runnable end-to-end.

---

## Phase 9: PR B-7 — Periodic-trigger scaffold + bundles + provenance + INV-13 [US3, US5, US4]

**Goal**: Scaffold the (inert) `Periodic` trigger type/config (D5/INV-PS3); add named strategy bundles (FR-015); add the run-level effective-triple provenance field, omitted when all-baseline (D6/D8); extend the INV-13 sync-point.

**Independent Test**: a declared periodic trigger yields byte-identical output; a bundle name expands to the correct triple with per-knob override; the effective triple round-trips run→replay; provenance absent on all-baseline runs.

- [ ] T029 [US5] Write failing INV-PS3 test (declared periodic trigger ⇒ byte-identical output; no event scheduled) — `sim/cluster/*_test.go`
- [ ] T030 [US3] Write failing bundle-resolution tests (name → triple; per-knob override; unset → baseline) + provenance round-trip test (SC-006) + provenance-omitted-when-baseline test (D8/INV-6) — `sim/*_test.go`, `cmd/*_test.go`
- [ ] T031 [P] [US5] Scaffold the `Periodic` trigger type + config (parseable, selectable, schedules no event — reuses the `ScalingTickEvent` pattern shape) — `sim/cluster/`
- [ ] T032 [P] [US3] Implement strategy-bundle resolution (name → {routing, eviction, creation} triple, per-knob override) in shared `resolvePolicies` — `sim/bundle.go`, `cmd/root.go`
- [ ] T033 [US3] Add the run-level effective-triple field to `MetricsOutput`, computed once at resolution, omitted when all-baseline (D6/D8) — `sim/metrics_utils.go`
- [ ] T034 [US1] Extend the INV-13 run/replay sync-point with the new policy/bundle/assignment selections + `logrus.Fatalf` fail-fast for unsupported replay; add a coverage-matrix parity case — `cmd/root.go`, `cmd/replay.go`, `cmd/*_test.go`
- [ ] T035 [US4] Confirm no-op golden holds; verification gate

**Checkpoint**: All seams, triggers, bundles, and provenance complete.

---

## Phase 10: Polish & Cross-Cutting Concerns

- [ ] T036 [P] Update CLAUDE.md + user docs (new flags, policies, bundles, provenance) — canonical-source-first (constitution Documentation rule)
- [ ] T037 [P] File follow-ups tracked in design §14: paper confirmation of D1 fallback (`Tantawi2025`) and D2 victim rule (`Li2025`); positional-tie-break CLI knob for strict CRN; governance-doc housekeeping (design-guidelines §4.2 AutoScaler status, invariants.md INV-7 `ResidentAdapters` row)
- [ ] T038 Full-feature verification gate on the assembled stack: `go build ./...` + `go test ./... -count=1` + `golangci-lint run ./...`

---

## Dependencies & Execution Order

### Phase / PR dependencies

- **Setup (P1)** → **Foundational (P2, no-op golden)** → seam PRs.
- **B-1** (routing registry) precedes **B-2** (route-to-holder).
- **B-3** (eviction seam) precedes **B-4** (rank-aware).
- **B-5** (creation seam + placement config) precedes **B-6** (pre-placement).
- **B-7** (bundles/provenance/periodic/INV-13) depends on B-1…B-6 being registrable (it composes their names) — last in the stack.
- **Polish (P10)** after B-1…B-7.

### Parallelism (across PRs, per design §3/§13)

Once the no-op golden (T003) is in place, the three seam *chains* are independent and can proceed concurrently, coordinated only by behavioral-contract tests:
- Routing chain: B-1 → B-2
- Eviction chain: B-3 → B-4
- Creation chain: B-5 → B-6

Note B-3 and B-5 both touch the instance-construction region (ordinary file-level merge coordination, not shared runtime state, design §3). B-7 is the join point.

### User-story completion

- **US4 (no-op)**: satisfied continuously (every PR's first + last task).
- **US2 (eviction ablation)**: complete after B-4.
- **US1 (static placement)**: complete after B-6 (needs routing B-1/B-2 + creation B-5/B-6).
- **US3 (bundle) + US5 (periodic)**: complete after B-7.

---

## Implementation Strategy

### MVP scope

The **US1 static-placement reproduction** is the headline MVP: B-1 → B-2 → B-5 → B-6 (routing registry + route-to-holder + creation seam + pre-placement). B-3/B-4 (US2 eviction ablation) is an independent, equally-shippable slice. Each PR is a demoable increment behind the byte-identical no-op default.

### Incremental delivery (recommended order)

B-1 → B-2 (strict routing demoable) → B-3 → B-4 (eviction ablation demoable) → B-5 → B-6 (static-placement reproduction demoable) → B-7 (bundles/provenance/periodic). Each merges independently; the no-op default keeps `main` green throughout.

### Per-PR gate (constitution Development Workflow)

Each B-N: worktree → micro-plan → convergence plan-review (0C/0I) → TDD (failing test first) → convergence code-review → self-audit → verification gate → PR. The tasks above are the PR's seed scope; the micro-plan elaborates exact file:line and GIVEN/WHEN/THEN contracts.

---

## Notes

- `[P]` = different files, no dependency on an incomplete task.
- `[Story]` labels map tasks to spec user stories for traceability; US4 (no-op) is cross-cutting.
- Tests are written FIRST and must fail before implementation (Principle IV).
- Byte-identical no-op default (INV-6/INV-L1) is verified at the start and end of every PR.
- No new `PartitionedRNG` subsystem is introduced (Principle II).
