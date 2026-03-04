# KV Capacity Auto-Calculation Macro Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Auto-calculate `total_kv_blocks` from model and hardware config in roofline mode, matching the llm-d-benchmark `capacity_planner.py` reference.

**Architecture:** Pure configuration-time function in `sim/latency/` that derives KV block count from existing model/hardware config. CLI integration in `cmd/root.go` between roofline config loading and validation. 2-PR series.

**Design Doc:** `docs/plans/2026-02-25-kv-capacity-auto-calculation-design.md`
**Issue:** #432

---

## A) Executive Summary

BLIS's roofline mode already has all information needed to derive KV cache capacity (model architecture + GPU specs), yet requires a manual `--total-kv-blocks` flag or defaults to an unrealistic 1M blocks. This 2-PR feature adds auto-calculation matching the llm-d-benchmark reference formula. PR 1 adds GPU memory capacity to the hardware config data. PR 2 implements the calculation function and CLI integration. No new interfaces, events, or module boundaries — this is a configuration-time enhancement below the extension threshold.

**Key milestones:**
- PR 1: GPU memory field in `HardwareCalib` + `hardware_config.json` data
- PR 2: Calculation function + CLI integration + validation against empirical `defaults.yaml`

---

## B) Repository Recon Summary

**Integration points (confirmed by inspection):**

| Component | File:Line | Relevance |
|---|---|---|
| `HardwareCalib` struct | `sim/model_hardware_config.go:16-28` | Needs `MemoryGiB` field (PR 1) |
| `hardware_config.json` | `hardware_config.json:1-22` | Needs `MemoryGiB` data for H100, A100-SXM (PR 1) |
| `GetHWConfig()` | `sim/latency/config.go:34-51` | Returns `HardwareCalib` — auto-gains `MemoryGiB` |
| `GetModelConfig()` | `sim/latency/config.go:73-126` | Returns `*ModelConfig`, discards raw HF config — PR 2 needs raw fields |
| `parseHFConfig()` | `sim/latency/config.go:54-71` | Returns `*HFConfig` with `Raw map[string]any` — PR 2 extracts MoE/tied-embedding indicators |
| `HFConfig` accessor methods | `sim/latency/config.go:128-227` | `GetBool`, `GetInt`, `GetString` — reusable for MoE/tied extraction |
| Roofline config loading | `cmd/root.go:178-198` | Where `roofline=true` is set, `modelConfig`/`hwConfig` populated |
| `total-kv-blocks` flag | `cmd/root.go:602` | Default 1M; `cmd.Flags().Changed()` at line 174 for R18 |
| `totalKVBlocks` validation | `cmd/root.go:281-283` | Must be > 0 — auto-calc inserts between lines 198-281 |
| `KVCacheConfig` constructor | `sim/config.go:13-27` | Receives `totalKVBlocks` — unchanged |
| `ValidateRooflineConfig()` | `sim/latency/config.go:237-272` | Validates latency fields — does NOT validate `MemoryGiB` (per Decision 4) |
| `defaults.yaml` empirical values | `defaults.yaml` | Llama-3.1-8B/H100/TP=2: 132,139 blocks (validation target) |

**GPU naming mismatch:** `defaults.yaml` uses `"A100-80"` while `hardware_config.json` uses `"A100-SXM"` for the same GPU (`cmd/default_config.go:41-50` vs `hardware_config.json:12`). PR 1 must reconcile (BC-14).

**Non-zero `HardwareCalib` construction sites (need `MemoryGiB` update in PR 1):**
1. `sim/latency/roofline_test.go:182-191` — `testHardwareCalib()` helper
2. `sim/latency/config_test.go:123` — valid config test
3. `sim/latency/config_test.go:175` — NaN/Inf test
4. `sim/latency/config_test.go:201` — valid config test
5. `sim/latency/config_test.go:215-218` — zero NumHeads test
6. `sim/latency/config_test.go:236-239` — zero TP test

**Zero-value sites (~40+ test files):** Safe to ignore — zero `MemoryGiB` means no auto-calculation (blackbox mode).

---

## C) High-Level Objectives + Non-Goals + Model Scoping

### Objectives

1. Auto-calculate `total_kv_blocks` in roofline mode matching `capacity_planner.py`
2. Maintain explicit `--total-kv-blocks` override precedence (R18)
3. Leave blackbox mode completely unchanged
4. Validate within 10% of empirical values (dense) / 20% (MoE)
5. Log calculation details for user diagnosis

### Non-Goals

- Pipeline parallelism, data parallelism modeling
- Dynamic KV cache resizing during simulation
- Quantized weight-aware memory estimation
- CPU tier auto-sizing
- KV cache dtype decoupling (`--kv-cache-dtype` flag)

### Compatibility Constraints

- Existing `--total-kv-blocks` flag behavior unchanged
- Existing `hardware_config.json` files without `MemoryGiB` continue to work (graceful fallback)
- Blackbox mode unchanged (KV-CAP-2)
- Golden dataset (`testdata/goldendataset.json`) unaffected — uses blackbox mode

### Model Scoping

| Component | Modeled | Simplified | Omitted | Justification |
|---|---|---|---|---|
| GPU memory budget | Total GiB × 0.9 utilization × TP GPUs | Fixed 0.9 `gpu_mem_util` | — | Matches reference; configurable deferred |
| Model weight memory | Architecture-based: layers + embeddings + lm_head | From `config.json` parameters, not safetensors | — | No external API needed; within 10% for dense |
| Activation memory | — | Fixed constants: 5.5/8.0 GiB | Per-model profiling | Matches reference; low sensitivity (±1 GiB = ±0.9% at TP=2) |
| Non-torch overhead | — | Fixed: 0.15/0.6 GiB per GPU | Per-driver profiling | Matches reference; stable across CUDA versions |
| MoE expert weights | Expert multiplication + router | Shared experts omitted | — | No shared-expert models in `defaults.yaml` |
| Quantized weights | — | — | Yes | Over-estimates conservatively; `--total-kv-blocks` escape hatch |
| CUDA graph memory | — | — | Yes (0 GiB) | Reference returns 0 |
| PyTorch fragmentation | — | — | Yes | Covered by 10% tolerance band |

---

## D) Concept Model (under 80 lines)

### Building Blocks

**1. GPU Memory Field (data extension)**
- Responsibility: Store GPU memory capacity in hardware calibration data
- OBSERVES: `hardware_config.json` data
- CONTROLS: `MemoryGiB` value in `HardwareCalib`
- OWNS: Nothing (immutable configuration data)
- INVARIANTS: `MemoryGiB > 0` when used for auto-calculation (KV-CAP-7)
- EVENTS: None
- EXTENSION FRICTION: 1 file (add entry to `hardware_config.json`)

**2. KV Capacity Calculator (pure function)**
- Responsibility: Derive `total_kv_blocks` from model architecture + GPU specs
- OBSERVES: `ModelConfig`, `HardwareCalib.MemoryGiB`, TP, block size, MoE indicators, `tie_word_embeddings`
- CONTROLS: Integer block count (returned) and diagnostic log output
- OWNS: Nothing (pure function, no mutable state)
- INVARIANTS: KV-CAP-1 through KV-CAP-9 (see design doc Section 4)
- EVENTS: None (configuration-time, not simulation-time)
- EXTENSION FRICTION: 0 files for new GPU types (just add JSON entry); 1 file for formula changes

### Interaction Model

```
hardware_config.json ──[MemoryGiB]──► GetHWConfig() ──► HardwareCalib
                                                              │
model_configs/config.json ──► parseHFConfig() ──► HFConfig   │
                                │                             │
                    ┌───────────┤                             │
                    │           ▼                             ▼
              MoE/tied    GetModelConfig()          CalculateKVBlocks()
              indicators    │                             │
                    │       ▼                             ▼
                    └──► ModelConfig ──────────────► total_kv_blocks (int64)
                                                         │
                                    cmd/root.go: if roofline && !Changed("total-kv-blocks")
                                                         │
                                                         ▼
                                                  KVCacheConfig.TotalKVBlocks
```

### System Invariants

All existing INV-1 through INV-8 preserved. See design doc Section 4 for downstream impact analysis. No new simulation-time state, events, or clock interactions.

### Real-System Correspondence

| BLIS concept | vLLM | Reference (`capacity_planner.py`) |
|---|---|---|
| Auto-calculated `total_kv_blocks` | `Worker.determine_num_available_blocks()` | `total_kv_cache_blocks()` |
| `MemoryGiB` | `torch.cuda.mem_get_info()` | `gpu_mem` parameter |
| Architecture-based weight count | `model.state_dict()` sizes | `model_params_by_dtype()` via HF API |

---

## E) Architectural Risk Register

| Decision | Assumption | Validation | Cost if Wrong | Gate |
|---|---|---|---|---|
| Architecture-based weight counting | Matches HF param counts within 10% for dense models | Unit test: Llama-3.1-8B/H100/TP=2 within 10% of 132,139 | PR 2 rework (formula fix) | Before PR 2 merge |
| Fixed activation constants (5.5/8.0 GiB) | Stable across vLLM versions | Sensitivity analysis shows ±3 GiB = ±2.6% at TP=2 | PR 2 constant update (1 line) | Before PR 2 merge |
| MoE detection from HF config fields | Union of 5 indicator fields covers all architectures | Mixtral-8x7B within 20% of empirical 58,377 | PR 2 detection fix (1 function) | Before PR 2 merge |
| A100-80 / A100-SXM naming | Alias resolves correctly | BC-14 test in PR 1 | PR 1 rework | Before PR 1 merge |

All risks have cost < 3 PRs. No mandatory spike required.

---

## F) Architectural Evolution

**Current → Target:**

The architecture change is minimal. No new modules, interfaces, or events are introduced.

1. `HardwareCalib` gains a `MemoryGiB` field — a data-only extension to an existing config type. The field is consumed only by the new calculation function, not by the existing roofline latency model.

2. A new exported pure function is added to `sim/latency/` for KV block capacity calculation. It accepts the existing `ModelConfig` and `HardwareCalib` types plus scalar parameters (TP, block size) and MoE/tied-embedding indicators extracted from the raw HF config. It returns `(int64, error)`.

3. `cmd/root.go` gains ~15 lines of CLI integration between the roofline config loading block (line 198) and the `totalKVBlocks` validation (line 281). This follows the existing `cmd.Flags().Changed()` R18 pattern already used for the blackbox path at line 174.

4. `GetModelConfig()` is not modified. Instead, the CLI layer calls `parseHFConfig()` (already exported via its return type `*HFConfig`) to access raw fields for MoE/tied-embedding extraction, sharing the same parsed config instance used by `GetModelConfig()`. This avoids modifying `GetModelConfig()`'s signature and its callers.

**What remains unchanged:** All simulation-time code, all existing interfaces, blackbox mode, golden dataset, KV store construction path, latency model, batch formation, routing, scheduling.

---

## G) Frozen Interface Reference

No interfaces are frozen by this plan. The calculation is a standalone pure function, not an interface implementation. Pre-existing frozen interfaces this plan depends on:

- `sim.KVStore` (frozen, `sim/kv_store.go`) — unchanged; receives `TotalKVBlocks` via `KVCacheConfig`
- `sim.LatencyModel` (frozen, `sim/latency_model.go`) — unchanged; `NewLatencyModel` factory unaffected

---

## H) Cross-Cutting Infrastructure Plan

| Item | Assigned PR | Details |
|---|---|---|
| Test infrastructure | PR 2 | Reuse existing `sim/latency/` test patterns. No new shared helpers needed. |
| CLAUDE.md update | PR 1 | Add `MemoryGiB` to `HardwareCalib` description. PR 2: add auto-calculation to "Key Data Flow" and CLI flag descriptions. |
| CI pipeline | — | No changes. Existing `go test ./...` and `golangci-lint` cover new code. |
| External dependencies | — | None. Pure Go, no new imports. |
| Interface freeze | — | N/A — no new interfaces introduced. |
| Golden dataset | — | Unaffected (blackbox mode, KV-CAP-2). |

---

## I) PR Plan

### PR 1: Add GPU Memory Capacity to Hardware Config

**Tier 1: Human Review Summary**

- **Title:** feat(latency): add GPU memory capacity to hardware calibration data
- **Building Block Change:** GPU Memory Field — extends `HardwareCalib` with `MemoryGiB`
- **Extension Type:** Data extension (below extension threshold — no new interfaces, no new events)
- **Motivation:** The KV capacity calculation needs GPU memory capacity. This is the prerequisite data.
- **Scope:**
  - In: `MemoryGiB` field on `HardwareCalib`, H100/A100 values in `hardware_config.json`, A100 naming alias
  - Out: Calculation function, CLI integration, auto-calculation logic
- **Behavioral Guarantees:**
  - BC-14: GPU name resolution — `"A100-80"` in `defaults.yaml` resolves to `"A100-SXM"` in `hardware_config.json`
  - All existing roofline tests pass unchanged (zero `MemoryGiB` has no effect on latency model)
- **Risks:** A100 naming mismatch — mitigated by adding alias entry or lookup mapping in `GetHWConfig()`
- **Cross-Cutting:** CLAUDE.md `HardwareCalib` description updated
- **Validation Gate:** None required (risk cost = 1 PR)

**Tier 2: Implementation Guide**

- **Architectural Impact:** Adds one `float64` field to `HardwareCalib`. No behavioral change to existing code paths.
- **API Surface Changes:** `HardwareCalib.MemoryGiB float64` (new field). No new methods or interfaces.
- **CLI Changes:** None.
- **Test Categories:** Unit (field present after parsing), regression (existing roofline tests pass).
- **Documentation Updates:** CLAUDE.md `HardwareCalib` description.
- **Extension Friction:** Adding a new GPU: 1 file (`hardware_config.json`). Same as current.
- **Parallel Development:** PR 2 depends on this. No parallel workstreams.
- **Why independently reviewable:** Pure data extension. Existing tests pass. No dead code — field will be consumed by PR 2, but is meaningful as documentation of GPU capability even without PR 2.
- **Why no dead code:** The `MemoryGiB` field is part of the hardware specification data. It is queryable via `GetHWConfig()` immediately after merge. The A100 naming resolution is testable immediately.
- **Files touched:** `sim/model_hardware_config.go` (1 field), `hardware_config.json` (2 entries), `sim/latency/roofline_test.go` (~1 line per fixture), `sim/latency/config_test.go` (~4 fixtures). ~5 files total.
- **Estimated size:** ~30 LOC changes + ~30 LOC tests

---

### PR 2: Auto-Calculate KV Blocks in Roofline Mode (depends on PR 1)

**Tier 1: Human Review Summary**

- **Title:** feat(latency): auto-calculate total_kv_blocks from model and hardware config
- **Building Block Change:** KV Capacity Calculator — new pure function + CLI integration
- **Extension Type:** Configuration enhancement (below extension threshold)
- **Motivation:** Eliminate the unrealistic 1M-block default in roofline mode by deriving KV capacity from the model/hardware config the user has already provided.
- **Scope:**
  - In: Calculation function, MoE detection, tied-embedding handling, model weight computation, CLI integration, validation against empirical values
  - Out: Quantized model support, CPU tier auto-sizing, `--kv-cache-dtype` flag, activation memory configurability
- **Behavioral Guarantees:** BC-1 through BC-25 (see design doc Section 11). Key contracts:
  - BC-1: Roofline mode + no explicit `--total-kv-blocks` → auto-calculated value
  - BC-2: Explicit `--total-kv-blocks` always wins (R18)
  - BC-3: Blackbox mode unchanged
  - BC-4: Within 10% of empirical for dense, 20% for MoE
  - BC-7: Zero/negative denominators return error, never panic (R11)
  - BC-8: Model weights exceeding GPU memory → error
  - BC-11: MoE MLP weights multiplied by `num_local_experts`
  - BC-12: `tie_word_embeddings: true` → `lm_head` omitted
  - BC-19: Non-SwiGLU activation → error
  - BC-23: `num_kv_heads % TP != 0` (when `num_kv_heads >= TP`) → error
- **Risks:**
  1. Formula accuracy — mitigated by validation against empirical `defaults.yaml` values
  2. HF config field extraction — mitigated by table-driven tests with real model configs
- **Cross-Cutting:** CLAUDE.md updated (auto-calculation in "Key Data Flow", CLI flag notes)
- **Validation Gate:** Formula accuracy verified before merge (empirical comparison)

**Tier 2: Implementation Guide**

- **Architectural Impact:** New exported function in `sim/latency/`. ~15 lines of CLI integration in `cmd/root.go`. No structural changes.
- **API Surface Changes:**
  - New exported function accepting `ModelConfig`, `HardwareCalib`, TP, block size, and MoE/tied-embedding indicators. Returns `(int64, error)`.
  - New helper to extract MoE indicators and `tie_word_embeddings` from `*HFConfig`. Returns a small struct with boolean `IsMoE`, `TieWordEmbeddings`, integer `NumLocalExperts`, string `HiddenAct`.
- **CLI Changes:** No new flags. Auto-calculation activates implicitly when roofline mode is active and `--total-kv-blocks` is not explicitly provided. Info-level log message reports calculated value with intermediates.
- **Test Categories:**
  - Unit: Calculation function with known inputs (Llama-3.1-8B, Llama-2-70B, Mixtral-8x7B, Qwen2.5-3B)
  - Unit: MoE detection (positive/negative cases, edge cases)
  - Unit: Model weight computation (dense, MoE, tied embeddings)
  - Unit: Edge cases (zero TP, zero GPU memory, overflow, floor-zero, num_kv_heads%TP)
  - Unit: Monotonicity (TP increase → more blocks)
  - Invariant: Purity (same inputs → same output)
  - Integration: CLI integration with roofline mode
  - Golden: Empirical validation against `defaults.yaml` values
- **Documentation Updates:** CLAUDE.md auto-calculation description, "Key Data Flow" section.
- **Extension Friction:** New GPU type: 0 code changes (just `hardware_config.json`). New activation constant: 1 line. New MoE indicator field: 1 line in detection function.
- **Parallel Development:** No further PRs depend on this. Feature is complete after PR 2.
- **Why independently reviewable:** Self-contained calculation + integration. All contracts testable.
- **Why no dead code:** Every function is called from CLI path or tests. Calculation function invoked at roofline config loading time.
- **Files touched:** `sim/latency/kv_capacity.go` (new, ~200 LOC), `sim/latency/kv_capacity_test.go` (new, ~400 LOC), `cmd/root.go` (~15 lines), CLAUDE.md (~5 lines). ~4 files total.
- **Estimated size:** ~200 LOC implementation + ~400 LOC tests

---

## J) Dependency DAG

```
PR 1: GPU memory field
  │
  ▼
PR 2: Auto-calculate KV blocks (depends on PR 1)
```

No parallelism possible — strictly sequential. PR 1 is small (~60 LOC total) and can be reviewed quickly.

**Merge sequence:** PR 1 → PR 2. No interface freeze needed. No validation gates between PRs.

---

## K) Design Bug Prevention Checklist

| Failure Mode | Prevention |
|---|---|
| Scaffolding creep | PR 1's `MemoryGiB` field is queryable immediately via `GetHWConfig()`. PR 2's function is called from CLI. No unused code. |
| Documentation drift | Each PR updates CLAUDE.md in the same commit. |
| Test infrastructure duplication | Reuse `sim/latency/` test patterns. No new shared helpers. |
| Golden dataset staleness | Blackbox golden tests unaffected (KV-CAP-2). No roofline golden tests exist. |
| Interface over-specification | No new interfaces. Pure function only. |
| Type catalog trap | No Go struct definitions in this macro plan. Module contracts described behaviorally. |
| Fidelity for its own sake | Every modeled component traces to the "realistic KV capacity" analysis question. |
| Shotgun surgery | `HardwareCalib` has a canonical `NewModelHardwareConfig()` constructor (`sim/config.go:71-84`). Non-zero construction sites audited (6 sites in PR 1). |
| Config mixing concerns | `MemoryGiB` belongs with `HardwareCalib` (GPU specification). Calculation inputs grouped by source. |

**Regression surfaces:**
- All existing `go test ./...` must pass after each PR
- `golangci-lint run ./...` must pass with zero new issues
- Golden dataset unchanged

**Backward compatibility:**
- Hardware config JSON files without `MemoryGiB` continue to work (field defaults to 0, auto-calculation skipped with warning)
- Explicit `--total-kv-blocks` always takes precedence
- Blackbox mode completely unchanged
