---
description: "Task list for LoRA Control-Plane Subsystem"
---

# Tasks: LoRA Control-Plane Subsystem

**Input**: Design documents from `/specs/007-lora-control-plane/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/

**Tests**: TDD tasks are INCLUDED and REQUIRED — BLIS Constitution Principle IV
(Test-First) is NON-NEGOTIABLE: write GIVEN/WHEN/THEN behavioral contracts, tests
MUST fail before implementation, every golden test has a companion invariant test.

**Organization**: Tasks grouped by user story (spec.md priorities). This is a layered
subsystem, so stories have real dependencies (documented below) rather than being fully
independent — but each still delivers an independently testable increment.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependency on incomplete tasks)
- **[Story]**: US1–US5 for user-story phases only
- All paths are repo-relative to the worktree root.

## Path Conventions

Single Go project: CLI in `cmd/`, library in `sim/` (+ subpackages `sim/lora/`,
`sim/latency/`, `sim/kv/`, `sim/workload/`, `sim/cluster/`). New subpackage `sim/lora/`
registers into `sim/` via `init()` (no reverse import — Principle I).

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Scaffolding with no observable behavior.

- [X] T001 [P] Create `sim/lora/` package skeleton with `sim/lora/doc.go` documenting the subsystem (no-op default, registers into `sim/` via `init()`, no RNG)
- [X] T002 [P] Capture pre-feature no-op golden baseline: run `./blis run --model qwen/qwen3-14b --seed 42` and store stdout as `specs/007-lora-control-plane/testdata/baseline_noop.json` for INV-6 regression

**Checkpoint**: Package compiles; baseline captured.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Config, registry, and the `Request.Adapter` field — needed by ALL user
stories. No user story can start until this phase completes. All paths reduce to a
no-op when adapters are unconfigured (INV-6).

- [X] T003 Write LoRAConfig validation contract tests in `sim/config_test.go`: adapters-present + `adapter_capacity==0` ⇒ error; `rank<=0` ⇒ error; `load_bandwidth_bytes_us<=0` ⇒ error (R11); request adapter id absent from registry ⇒ error; empty config ⇒ valid/inert (tests MUST fail first)
- [X] T004 Add `LoRAConfig` sub-config + `AdapterSpec{ID string; Rank int}` types in `sim/config.go` — pointer types where zero is meaningful (R9), yaml tags, no logic yet
- [X] T005 Implement `LoRAConfig.Validate()` in `sim/config.go` to pass T003 (R3 numeric guards; library returns error / factory panics, never `logrus.Fatalf` in `sim/` — Principle V)
- [X] T006 Embed `LoRAConfig` as the 7th sub-config in `SimConfig` (`sim/simulator.go`); update factory signatures to accept the narrowest sub-config; **R4: grep and update ALL `SimConfig` construction sites**
- [X] T007 Add inert `lora:` defaults to `defaults.yaml` (capacity nil/unset; coefficients present but only applied when adapters configured)
- [X] T007a Write CLI-flag contract test in `cmd/run_test.go`/`cmd/replay_test.go`: each `--lora-*` config flag maps onto `LoRAConfig`; unset flags do NOT override `defaults.yaml` (R18); `--lora-adapter-capacity 0` with adapters present ⇒ `logrus.Fatalf` (contracts/cli-flags.md) (fail first)
- [X] T007b Register `--lora-adapter-capacity`, `--lora-load-base-latency-us`, `--lora-load-bandwidth-bytes-us`, `--lora-footprint-bytes-per-rank` in `cmd/run.go` and `cmd/replay.go`, each guarded by `cmd.Flags().Changed(...)` before applying defaults (R18); bind onto `SimConfig.LoRAConfig`, to pass T007a. (The per-rank compute-overhead coefficients `step_overhead_tiers` and the adapter registry are **config-file only** — a scalar flag cannot express a per-rank map — so no `--lora-step-overhead-*` flags are registered.)
- [X] T008 [P] Write adapter-registry contract tests in `sim/lora/registry_test.go`: `id→rank` lookup; iteration is key-sorted (R2); completeness check flags unknown ids (tests MUST fail first)
- [X] T009 Implement adapter registry in `sim/lora/registry.go` (built from `LoRAConfig.Adapters`) + `init()` registration into `sim/`, to pass T008
- [X] T010 Add `Request.Adapter string` field in `sim/request.go`; **R4: grep and update ALL `Request{...}` literal construction sites**; zero value `""` MUST preserve byte-identity
- [X] T011 Write INV-6 no-op byte-identity test comparing an adapter-blind run against `testdata/baseline_noop.json` (companion invariant test; must pass after T010)

**Checkpoint**: Config validates, registry resolves, `Request.Adapter` exists, adapter-blind output byte-identical to baseline. User stories can now begin.

---

## Phase 3: User Story 1 — Attribute requests and metrics to LoRA adapters (Priority: P1) 🎯 MVP

**Goal**: Assign adapters in a workload and get per-adapter metrics (load counts, TTFT,
throughput); adapter-blind runs stay byte-identical.

**Independent Test**: Run a 2-adapter workload → results expose per-adapter TTFT &
throughput; run any adapter-free scenario → byte-identical to baseline.

**Dependencies**: Phase 2.

- [X] T012 [US1] Write workload-threading contract test in `sim/workload/spec_test.go`: `ClientSpec`/`CohortSpec.Adapter` → `Request.Adapter`; omitted ⇒ `""`; id must be a registry key and base model must match client/cohort `model` (fail first)
- [X] T013 [P] [US1] Write per-adapter metrics contract test in `sim/metrics_test.go`: `adapters` block present with per-adapter TTFT/throughput when configured; ABSENT when unconfigured (INV-6); keys sorted (R2); base-model requests attributed to no adapter (fail first)
- [X] T014 [US1] Add `Adapter string` (yaml `omitempty`) to `ClientSpec` and `CohortSpec` in `sim/workload/spec.go`
- [X] T015 [US1] Thread `Adapter` through request generation into `Request.Adapter` in `sim/workload/` (generator), with registry+base-model validation, to pass T012
- [X] T016 [US1] Add `Adapter` field to `RequestMetrics` in `sim/metrics_utils.go` (per-request attribution)
- [X] T017 [US1] Add per-adapter aggregate section to `MetricsOutput` in `sim/metrics.go`: `TTFTByAdapter`, `ThroughputByAdapter`, `AdapterLoadCount` (0 for now), sorted keys, omitted when no adapters — to pass T013
- [X] T018 [US1] Companion invariant test in `sim/metrics_test.go`: INV-1 conservation holds with adapters; per-adapter counts partition (not duplicate) global request accounting
- [X] T019 [US1] Re-run T011 no-op byte-identity regression to confirm US1 additions stay inert when unconfigured

**Checkpoint**: MVP complete — per-adapter attribution works; no-op default proven byte-identical.

---

## Phase 4: User Story 2 — Per-instance adapter capacity with LRU eviction (Priority: P1)

**Goal**: Track a per-instance resident-adapter set bounded by capacity; evict LRU
(never a pinned/in-use adapter) under pressure.

**Independent Test**: Capacity N, traffic for M>N adapters → resident set never exceeds
N; evictions follow LRU; in-use adapters not evicted.

**Dependencies**: Phase 2 (registry, `Request.Adapter`). Metrics wiring builds on US1's T017.

- [ ] T020 [US2] Write resident-set contract tests in `sim/lora/resident_set_test.go`: `store`/`touch`/`evictLRU`; invariant `resident ≤ capacity`; pinned (in-use) never evicted; deterministic LRU order, no RNG (fail first)
- [ ] T021 [US2] Implement resident-adapter set (LRU) in `sim/lora/resident_set.go` reusing the `cpuTier` pattern (`store`/`touch`/`unlink`/`appendToTail` from `sim/kv/tiered.go`), to pass T020
- [ ] T022 [US2] Wire a per-instance resident set into instance state in `sim/simulator.go` (InstanceSimulator): `touch` on warm reference, `admit` (+`evictLRU`) on cold — STATE ONLY, no latency effect yet
- [ ] T023 [US2] Companion integration invariant test: resident set never exceeds capacity across all steps of a M>N run (SC-003)
- [ ] T024 [US2] Wire `AdapterLoadCount`/`AdapterEvictionCount` in `sim/metrics.go` from resident-set load/evict events (fills the US1 placeholder)

**Checkpoint**: Finite adapter slots with LRU eviction observable via metrics; capacity invariant holds.

---

## Phase 5: User Story 3 — Adapter cost physics (Priority: P1)

**Goal**: Charge cold-load latency (pre-admission gate), per-step compute overhead
(unique adapters × max rank), and per-adapter HBM footprint (memory conserved).

**Independent Test**: cold TTFT > warm TTFT (same input); more unique adapters/higher
rank → longer step; usable KV shrinks with resident adapters, total conserved.

**Dependencies**: Phase 4 (resident set for cold/warm + footprint), Phase 3 (identity/registry).

### Cost model core

- [ ] T025 [P] [US3] Write cost-model contract tests in `sim/lora/cost_model_test.go`: `LoadLatency(rank) = base + ceil(bytes/bw) ≥ 0`; `FootprintBytes(rank)` monotonic in rank; `StepOverheadFactor` = `1.0` when no adapters; pure/deterministic (fail first)
- [ ] T026 [US3] Implement `sim/lora/cost_model.go` (pure query methods per Principle III) to pass T025; register into `sim/` via `init()`

### Cold-load latency — pre-admission gate (PR3)

- [ ] T027 [US3] Write cold-load contract test: cold adapter adds `LoadLatency` to TTFT (charged once), warm adds zero (US3 scenario 1, SC-004); loads serialized per instance (fail first)
- [ ] T028 [US3] Implement the pre-admission gate in `sim/batch_formation.go` + `sim/simulator.go`: a cold request is held out of the runnable batch; schedule a serialized per-instance adapter-load completion at `now + LoadLatency`; adapter becomes resident + request batch-eligible at completion
- [ ] T029 [US3] Companion invariant tests: INV-5 (`enqueue ≤ schedule`), INV-8 (load event keeps queue non-idle), INV-3 (load completion ≥ now), no RNG (determinism)

### Per-step compute overhead (PR4)

- [X] T030 [P] [US3] Write step-overhead contract test in `sim/latency/` (`*_test.go`): factor `1.0` when `A_B==0` **including a non-unit `K7` case** (normalized factor = 1.0 for any `K7`, guards the no-op regression); **strictly increasing in max rank** — a batch whose max-rank tier is higher yields a strictly larger factor for the same `A_B>0` (non-vacuous; asserts rank actually enters via the tier, FR-009); monotonic in unique-adapter count; duplicate adapter counted once; **out-of-envelope rank/batch is clamped to the nearest calibrated tier (factor stays ≥1, no inversion)**; roofline factor == trained-physics factor (R23) (fail first)
- [X] T030a [US3] Extend latency-model construction to accept an adapter-cost accessor (e.g. `AdapterCost` with `RankOf(id) int` / `StepOverheadFactor(batch)`), threaded from `sim/lora` via the existing `NewLatencyModelFunc` registration; nil accessor ⇒ no adapter effect (INV-6). Both roofline and trained-physics receive the same accessor (R13/R23)
- [X] T031 [US3] Implement the shared adapter-overhead factor `1 + (K6(r_max)/K7(r_max))·A_B` in `sim/latency/` — `K6/K7` looked up from the **rank-tier table** (`step_overhead_tiers`) by the batch's max rank `r_max`, clamped to the nearest calibrated tier when out of envelope; normalized to = 1.0 at `A_B=0` by construction. Rank resolved via the T030a accessor (NOT a per-request field); applied identically by both backends (R23); preserve `StepTime ≥ 1` (INV-3)
- [X] T032 [US3] Companion invariant test: no-op byte-identity preserved when `A_B==0` (INV-6); `StepTime ≥ 1` for empty and adapter batches

### HBM accounting (PR5, static)

- [ ] T033 [P] [US3] Write HBM contract test in `sim/latency/kv_capacity_test.go`: usable KV blocks shrink by the **fixed `capacity × per-slot footprint` (max declared rank)** reservation set once at startup; `allocated + free + adapter_reserved = total` (INV-4/INV-L4); and **`adapter_reserved` stays numerically invariant across adapter load/evict churn** (guards against the rejected dynamic running-sum model, which would nominally satisfy a laxer test); an infeasible reservation is rejected at startup (R22), not a runtime drop (fail first)
- [ ] T034 [US3] Subtract the **fixed capacity-based reservation** (`capacity × per-slot footprint`, from max declared rank) in `CalculateKVBlocks` (`sim/latency/kv_capacity.go`), once at startup beside model weights, to pass T033

**Checkpoint**: All three cost terms live; TTFT/throughput/memory physically meaningful; INV-4 extended and holds.

---

## Phase 6: User Story 4 — LoRA-aware placement scorer (Priority: P2)

**Goal**: A selectable `lora-affinity` scorer that prefers instances already holding a
request's adapter, reducing loads/evictions under skewed popularity.

**Independent Test**: warm instance preferred; skewed workload → fewer loads vs
adapter-blind router; scorer not selected → routing unchanged.

**Dependencies**: Phase 4 (resident set → snapshot), Phase 3 (identity).

- [ ] T035 [US4] Write scorer contract tests in `sim/routing_scorers_test.go`: warm instance scores > cold (US4-1); empty `req.Adapter` ⇒ neutral; scorer absent from profile ⇒ routing unchanged (US4-3, INV-6); reads only `Adapter`/`ResidentAdapters`, never `OutputTokens` (INV-9) (fail first)
- [ ] T036 [US4] Add `ResidentAdapters` field to `RoutingSnapshot` in `sim/routing.go` (zero value ⇒ scorer neutral)
- [ ] T037 [US4] Populate `ResidentAdapters` in `buildRouterState()` (`sim/cluster/cluster_event.go`) at Periodic freshness (INV-7); confirm adapter rides the existing per-model routing filter
- [ ] T038 [US4] Implement `lora-affinity` `scorerFunc` + register in `validScorerNames` (R8) and the `newScorerWithObserver` switch in `sim/routing_scorers.go`; min-max normalize (llm-d parity); doc comment declares `ResidentAdapters` + Periodic tier (R17), to pass T035
- [ ] T039 [US4] Add `--lora-scorer-weight` flag + profile composition in `cmd/run.go` and `cmd/replay.go` with `Flags().Changed` guard (R18)
- [ ] T040 [US4] Integration test in `sim/cluster/`: skewed (Zipfian) adapter popularity → ≥30% fewer total adapter loads with `lora-affinity` vs least-loaded routing (SC-005)

**Checkpoint**: Placement control surface live; routing payoff demonstrated; default routing unchanged.

---

## Phase 7: User Story 5 — Fidelity validation vs Digital Twin (Priority: P3, OPTIONAL)

**Goal**: Import pre-fitted DT rank→latency tables and compare adapter-aware TTFT to the
reference.

**Independent Test**: For a supported reference config, import table, run, produce a
per-adapter TTFT comparison report with an error metric.

**Dependencies**: Phase 5 (cost physics).

- [ ] T041 [US5] Write calibration-import contract test: DT rank-table import + per-adapter **TTFT and throughput** comparison report produced with an error metric (fail first)
- [ ] T042 [US5] Implement DT rank-table import + adapter-reference comparison in `cmd/calibrate.go` (add `--adapter-reference` flag); per-adapter MAPE on **TTFT and throughput**; R12 if output format changes
- [ ] T043 [US5] Validate SC-007 (≤20% MAPE on **both TTFT and throughput**) for the two reference configs via a fixture-based test

**Checkpoint**: Fidelity bounded and reported for the two calibrated configs.

---

## Phase 8: Polish & Cross-Cutting Concerns

**Purpose**: Parity, determinism, docs, and the verification gate.

- [ ] T044 [P] Run/replay parity integration test (INV-13): export a trace with adapters via `--trace-output`, replay with identical flags, assert identical per-request metrics
- [ ] T045 [P] Determinism regression test (INV-6, SC-006): same seed + adapter config → byte-identical stdout across two runs
- [ ] T046 [P] Update docs: `CLAUDE.md` "Current Implementation Focus" + valid scorer list (add `lora-affinity`); `docs/guide/` routing/latency pages; `docs/contributing/extension-recipes.md`
- [ ] T047 R15/R12 sweep: grep for stale `PR N` / `TODO.*PR N` references and resolve; regenerate + document golden dataset if any output format changed
- [ ] T048 Verification gate: `go build ./... && go test ./... -count=1 && golangci-lint run ./...` — all MUST report zero failures

---

## Dependencies & Execution Order

```text
Setup (P1) ─▶ Foundational (P2) ─▶ US1 (P3, MVP) ─▶ US2 (P4) ─▶ US3 (P5) ─▶ US5 (P7)
                                        │              └──────────▶ US4 (P6)
                                        └────────────────────────▶ US4 (P6)
                                                                      │
                              Polish (P8) ◀────────── all stories ────┘
```

- **Foundational blocks everything** — registry + `Request.Adapter` + config. CLI-flag tasks T007a/T007b run after T004–T006 (LoRAConfig type + embed).
- **US3 rank plumbing**: T030a (adapter-cost accessor into the latency model) precedes T031 and unblocks the compute-overhead factor.
- **US1** is the MVP; **US2** depends on Foundational (uses US1's metric struct in T024).
- **US3** depends on US2 (resident set for cold/warm + footprint) and US1.
- **US4** depends on US2 (resident set → snapshot) and US1.
- **US5** depends on US3 (cost physics).
- **Polish** runs after the stories it verifies.

## Parallel Execution Examples

- **Setup**: T001, T002 in parallel.
- **Foundational**: T008 (registry tests, `sim/lora/`) ∥ T003 (config tests, `sim/config_test.go`) — different files.
- **US1**: T013 (metrics test) ∥ T012 (workload test) — different files.
- **US3**: T025 (cost-model test) ∥ T030 (step-overhead test) ∥ T033 (HBM test) — different packages/files; implementations (T026/T031/T034) then proceed per their own test.
- **Polish**: T044 ∥ T045 ∥ T046 — independent files.

## Implementation Strategy

- **MVP first**: Setup + Foundational + **US1** → adapter attribution + per-adapter
  metrics with a proven no-op default. Ship/validate before proceeding.
- **Incremental**: add US2 (capacity/eviction) → US3 (cost physics) → US4 (scorer).
  Each phase is a no-op-safe, individually mergeable Small-tier PR (plan §PR decomposition).
- **TDD every task cluster** (Principle IV): contract test (fails) → implement → pass →
  companion invariant test → lint → commit.
- **US5 optional**: schedule only if DT reference fidelity is needed for the research payload.
