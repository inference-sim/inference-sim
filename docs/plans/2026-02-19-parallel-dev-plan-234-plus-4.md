# Parallel Development Plan: #234 + 4 Concurrent Issues

## Context

We have 5 issues to address. Issue #234 (P0 Critical) must merge first because it touches `sim/cluster/deployment.go` and `cmd/root.go`, which overlap with PR17. After #234 merges, 4 issues proceed in parallel with zero file overlap.

## Phase 0: Quick Fix #234 (Sequential, Merge First) — COMPLETE

**Issue:** `fix(deployment): DeploymentConfig missing KVTransferBaseLatency`
**Branch:** `fix/234-deployment-base-latency`
**PR:** #253 (https://github.com/inference-sim/inference-sim/pull/253) — MERGED
**Micro plan:** `docs/plans/fix-234-deployment-base-latency-plan.md`

### What was done (5 files, 4 commits)

| File | Change |
|------|--------|
| `sim/cluster/deployment.go` | Added `KVTransferBaseLatency int64` field + `ToSimConfig()` propagation |
| `cmd/root.go` | Added `kvTransferBaseLatency` var, `--kv-transfer-base-latency` flag (default 0), negative validation, `DeploymentConfig{}` wiring |
| `sim/cluster/cluster_test.go` | Extended `TestDeploymentConfig_ToSimConfig_FieldMapping` with `KVTransferBaseLatency: 42` assertion |
| `CLAUDE.md` | Added `--kv-transfer-base-latency` to CLI flags list |
| `docs/plans/fix-234-deployment-base-latency-plan.md` | Micro plan with 4 behavioral contracts (BC-1 through BC-4) |

### Status
PR #253 merged 2026-02-19. Main fast-forwarded. Phase 1 started.

---

## Phase 1: Four Parallel Worktrees (After #234 Merges)

### Conflict-Freedom Proof

| Worktree | Primary Files | Overlap |
|----------|--------------|---------|
| PR17 | `sim/routing.go`, `sim/bundle.go`, `sim/routing_scorers.go` (new), `sim/cluster/deployment.go`, `sim/cluster/cluster.go`, `cmd/root.go`, `examples/` | None with others |
| #236 | `sim/roofline_step.go`, `sim/model_hardware_config.go` | None with others |
| #237 | `sim/cluster/metrics.go`, `sim/workload/servegen.go` | None with others |
| #240 | `sim/priority.go`, `sim/queue.go` | None with others |

Shared files: `CLAUDE.md` and `README.md` may get minor doc updates, but only PR17 rewrites README sections. Others only touch source code.

---

### Worktree 1: PR17 — Composable Scorer Framework — PR #260 OPEN

**Issue:** #229 + #230 (closes both)
**Branch:** `feat/pr17-scorer-framework`
**PR:** #260 (https://github.com/inference-sim/inference-sim/pull/260)
**Micro plan:** `docs/plans/pr17-scorer-framework-plan.md`
**Scope:** ~315 LOC non-test (largest of the 4)
**Discovered:** #259 (PrefixAffinity full-sequence hash limitation)

#### Summary
Replace monolithic `WeightedScoring` (two hardcoded dimensions) with composable multi-scorer pipeline. Ships 3 stateless scorers: `queue-depth`, `kv-utilization`, `load-balance`.

#### Key Changes

**`sim/routing.go` (~100 LOC delta):**
- Replace `WeightedScoring` struct (lines 116-119): two weight fields → scorer slice + weight slice
- Replace `Route()` method (lines 121-167): iterate scorers, aggregate weighted scores, clamp [0,1], argmax
- Update `NewRoutingPolicy` factory `case "weighted"` (lines 256-266): accept scorer config instead of two floats

**`sim/routing_scorers.go` (new, ~120 LOC):**
- Scorer behavioral contract (name + score function + optional observer)
- `QueueDepthScorer`: min-max normalization of `EffectiveLoad()`
- `KVUtilizationScorer`: `1 - (UsedBlocks / TotalBlocks)` per snapshot
- `LoadBalanceScorer`: `1 / (1 + EffectiveLoad)` (preserves current load dimension)
- Scorer factory + `IsValidScorer()` accessor

**`sim/bundle.go` (~40 LOC):**
- Replace `RoutingConfig.CacheWeight`/`LoadWeight` (lines 32-36) with scorer list
- Update `Validate()` (lines 137-142): validate scorer names + weights

**`sim/cluster/deployment.go` (~10 LOC):**
- Replace `RoutingCacheWeight`/`RoutingLoadWeight` (lines 34-35) with scorer config field

**`sim/cluster/cluster.go` (~5 LOC):**
- Update `NewRoutingPolicy` call (line 85): pass scorer config

**`cmd/root.go` (~40 LOC):**
- Remove `routingCacheWeight`/`routingLoadWeight` vars (lines 70-71)
- Remove flag registrations (lines 563-564)
- Remove weight validation (lines 354-375)
- Add `--routing-scorers` flag (string, comma-separated `name:weight`)
- Add scorer name + weight parsing/validation (NaN/Inf/negative/unknown)
- Update `DeploymentConfig{}` construction (lines 410-412)

**`examples/weighted-routing.yaml`:** Complete rewrite — new YAML schema + scorer pipeline docs
**`examples/policy-config.yaml`:** Update comments for scorer config
**`README.md`:** Remove misleading demo (lines 172-192), update weighted description (#230)

#### Behavioral Contracts
- BC-17-1: Each scorer returns [0,1] for every instance
- BC-17-2: Weight normalization: `[3,2,2]` ≡ `[0.43,0.29,0.29]`
- BC-17-3: Non-weighted policies produce byte-identical output
- BC-17-4: NaN/Inf/negative weights rejected at config time
- BC-17-5: `weighted` with `load-balance:1` ≈ `least-loaded` distribution

#### Test Updates Required
- `sim/routing_test.go`: Rewrite WeightedScoring tests for scorer pipeline
- `sim/bundle_test.go`: Update RoutingConfig YAML parsing tests
- `sim/cluster/cluster_test.go` (lines 765-766): Update weight fields
- `sim/cluster/pending_requests_test.go` (lines 25-26): Update weight fields
- `sim/cluster/cluster_trace_test.go` (lines 102-103): Update weight fields
- Golden dataset: Regenerate `weighted` baselines (non-weighted unchanged)

#### Prerequisite
- Existing macro plan: `docs/plans/2026-02-19-weighted-scoring-macro-plan.md`
- Existing design doc: `docs/plans/2026-02-19-weighted-scoring-evolution-design.md`
- Needs micro plan per prworkflow.md before implementation

---

### Worktree 2: #236 — Roofline Validation

**Branch:** `fix/236-roofline-validation`
**PR:** #258 — awaiting merge
**Scope:** ~50 LOC non-test (small)

#### What was done (5 files, 6 commits)

**`sim/model_hardware_config.go`:**
- Added `invalidPositiveFloat()` helper (rejects `<= 0`, NaN, Inf)
- Added `ValidateRooflineConfig(ModelConfig, HardwareCalib) error` — validates 9 fields: `NumHeads`, `NumLayers`, `HiddenDim`, `BytesPerParam`, `TFlopsPeak`, `BwPeakTBs`, `BwEffConstant`, `MfuPrefill`, `MfuDecode`

**`sim/simulator.go`:**
- `NewSimulator`: validates `TP > 0` when `Roofline == true`, calls `ValidateRooflineConfig`

**`sim/roofline_step.go`:**
- Sorted map keys before float accumulation in `calculateMemoryAccessBytes()`
- Added precondition doc comment on `rooflineStepTime`

**`sim/roofline_step_test.go` (new):**
- Determinism test (100 iterations) + component-sum conservation invariant

**`sim/model_hardware_config_test.go`:**
- 7 new tests: table-driven model field validation, hardware field validation, NaN/Inf, valid config, NewSimulator integration (roofline + non-roofline)

#### Behavioral Contracts
- Zero/NaN/Inf model config fields → clear error, not silent corruption
- Zero/NaN/Inf hardware config fields → clear error listing all invalid fields
- TP=0 in roofline mode → clear error
- Non-roofline mode unaffected
- Sorted map iteration produces deterministic float accumulation

#### Discovered Issues
- #254 — Dead code in `calculateTransformerFlops` (duplicate attention block)
- #257 — Missing unit tests for roofline computation functions

---

### Worktree 3: #237 — Silent Dropped Requests Counter

**Branch:** `fix/237-silent-drop-counter`
**Scope:** ~20 LOC non-test (small)

#### Key Changes

**`sim/cluster/metrics.go`:**
- Lines 250-259 (TTFT loop): Add `droppedCount` counter, increment on `!ok`, log warning after loop (mirror `SLOAttainment` pattern at lines 296-327)
- Lines 261-270 (E2E loop): Same pattern — counter + warning

**`sim/workload/servegen.go`:**
- Line 207: Add `skippedRows` counter, increment on `len(record) < 4`, log warning after loop

#### Behavioral Contracts
- Dropped requests produce `logrus.Warnf` with count (not silent)
- Skipped CSV rows produce `logrus.Warnf` with count
- Existing behavior unchanged (still skip/drop, just with visibility)

#### Tests
- Behavioral: metrics with missing request IDs → warning logged, metrics still computed for present requests
- Behavioral: malformed CSV → warning logged, valid rows still parsed

---

### Worktree 4: #240 — Stale Comments Cleanup

**Branch:** `fix/240-stale-comments`
**Scope:** ~5 LOC non-test (trivial)

#### Key Changes

**`sim/priority.go` line 26:**
- Delete stale comment: `// Full SLO class integration (using TenantState) is deferred to a future PR.`
- Replace with: `// SLO class integration uses the SLOClass field on Request (set by workload generator).`

**`sim/queue.go` line 4:**
- Delete stale TODO: `// TODO: Requests need to be re-queued on preemption.`
- Preemption re-queuing is already implemented via `PreemptionEvent` in `sim/event.go`

**`sim/workload/spec.go` lines 19, 30:**
- No change needed — `AggregateRate` and `RateFraction` are validated to be `> 0` at load time, so bare `float64` is correct (zero is always an error, not ambiguous)

#### Tests
- No new tests needed (comment-only changes)
- Run `go test ./...` to verify no regressions

---

## Execution Order

```
Phase 0:  #234 (sequential)
          ├── micro plan → implement → PR → merge → ff main
          │
Phase 1:  ┌─── PR17 (worktree 1) ───────────────────────┐
(parallel) │    micro plan → implement → PR → merge       │
          ├─── #236 (worktree 2) ──────────────┐         │
          │    micro plan → implement → PR      │         │
          ├─── #237 (worktree 3) ──────────┐   │         │
          │    micro plan → implement → PR  │   │         │
          ├─── #240 (worktree 4) ──────┐   │   │         │
          │    micro plan → impl → PR  │   │   │         │
          └────────────────────────────┴───┴───┴─────────┘
```

## Verification

After all 5 PRs merge:
```bash
git fetch upstream main && git merge --ff-only upstream/main
go build ./...
golangci-lint run ./...
go test ./...
```

Additionally for PR17:
```bash
# Verify new scorer pipeline works
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy weighted \
  --routing-scorers "queue-depth:2,kv-utilization:2,load-balance:1"

# Verify non-weighted policies unchanged
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --routing-policy least-loaded
```

For #236:
```bash
# Verify roofline mode with valid config
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json --hardware H100 --tp 1
```
