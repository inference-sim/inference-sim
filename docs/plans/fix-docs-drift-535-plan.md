# Fix Documentation Drift from PR #474 and #531 — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix stale documentation across 8 files so that user-facing docs accurately describe the KV cache auto-calculation feature and ITL counting behavior.

**The problem today:** PR #474 (KV capacity auto-calculation) and PR #531 (phantom ITL fix) merged without updating user-facing documentation. Users reading the docs get wrong information: `MemoryGiB` is described as "reserved for future" when it's actively used, the `--total-kv-blocks` resolution process omits auto-calculation, roofline.md claims MoE models aren't supported (crossmodel handles them), the README file tree is missing `kv_capacity.go`, and the golden dataset still references the old `simulation_worker` binary name.

**What this PR adds:**
1. Fixes all "reserved for future" references to `MemoryGiB` — now accurately describes its use in KV auto-calculation
2. Rewrites the `--total-kv-blocks` resolution process to document the full 4-layer priority chain (CLI flag > auto-calculation > defaults.yaml > hardcoded default)
3. Updates roofline.md MoE statement and adds KV auto-calculation documentation
4. Adds `kv_capacity.go` to README file tree, adds missing example files, fixes `ParseHFConfig` capitalization
5. Updates golden dataset `blis-cmd` field from `simulation_worker` to `blis`

**Why this matters:** BLIS docs are the primary onboarding path. Stale docs cause users to misunderstand KV block behavior, leading to misconfigured simulations and incorrect capacity plans.

**Architecture:** Pure documentation changes across 8 files: `CLAUDE.md`, `docs/concepts/roofline.md`, `docs/reference/configuration.md`, `docs/guide/latency-models.md`, `docs/guide/kv-cache.md`, `docs/concepts/glossary.md`, `README.md`, `testdata/goldendataset.json`. No code changes.

**Source:** GitHub issue #535

**Closes:** Fixes #535

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes documentation drift caused by PR #474 (KV capacity auto-calculation) and PR #531 (phantom ITL fix). Eight files contain stale content: two say `MemoryGiB` is "reserved for future" when the feature shipped, the configuration reference omits auto-calculation from its resolution process, roofline.md incorrectly claims MoE models aren't supported, the README file tree is missing `kv_capacity.go`, the glossary ITL definition omits the N-1 count, and the golden dataset references the old binary name.

Adjacent blocks: `cmd/root.go` (source of truth for resolution logic), `sim/latency/kv_capacity.go` (auto-calculation implementation), `sim/simulator.go` (ITL counting). No code changes — docs only.

Deviations from issue #535 documented in Section D below.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: MemoryGiB Accuracy
- GIVEN a user reading `docs/concepts/roofline.md` or `CLAUDE.md`
- WHEN they look up `MemoryGiB` in the hardware config table or file tree
- THEN the description MUST state it is actively used by KV capacity auto-calculation, not "reserved for future"
- MECHANISM: Direct text replacement in both locations

BC-2: Resolution Process Completeness
- GIVEN a user reading `docs/reference/configuration.md` Resolution Process
- WHEN they need to understand how `--total-kv-blocks` is determined
- THEN the documentation MUST describe all 4 resolution layers: (1) explicit CLI flag always wins, (2) auto-calculation from model architecture + GPU memory for roofline/crossmodel, (3) defaults.yaml per-model value, (4) hardcoded 1,000,000 fallback
- MECHANISM: Rewrite Resolution Process section

BC-3: MoE Documentation Accuracy
- GIVEN a user reading `docs/concepts/roofline.md`
- WHEN they encounter the introductory paragraph
- THEN it MUST NOT claim MoE models are unsupported; it MUST direct users to crossmodel mode for MoE
- MECHANISM: Replace "except Mixture-of-Expert (MoE) models currently" with guidance to use crossmodel

BC-4: README File Tree Completeness
- GIVEN a contributor reading the README project structure
- WHEN they look at `sim/latency/` entries
- THEN `kv_capacity.go` MUST be listed with an accurate description
- MECHANISM: Add file entry to README tree

BC-5: Golden Dataset Binary Name
- GIVEN a user or contributor reading `testdata/goldendataset.json`
- WHEN they examine the `blis-cmd` field
- THEN it MUST reference `./blis`, not `./simulation_worker`
- MECHANISM: String replacement in all 5 entries

BC-6: KV Auto-Calculation in User Guides
- GIVEN a user reading `docs/guide/latency-models.md` or `docs/guide/kv-cache.md`
- WHEN they look up KV block configuration
- THEN they MUST find a note that roofline/crossmodel modes auto-calculate `--total-kv-blocks` from GPU memory when the flag is not explicitly set
- MECHANISM: Add notes to both guide pages

BC-7: Configuration Footnote Accuracy
- GIVEN a user reading the `--total-kv-blocks` footnote in `docs/reference/configuration.md` or `docs/guide/kv-cache.md`
- WHEN they need to understand the default value behavior
- THEN the footnote MUST mention auto-calculation as a source alongside defaults.yaml
- MECHANISM: Expand footnote text

BC-8: ITL Count Accuracy in Glossary
- GIVEN a user reading `docs/concepts/glossary.md`
- WHEN they look up the ITL definition
- THEN it MUST state that N output tokens produce N-1 ITL entries
- MECHANISM: Add sentence to glossary ITL definition

**Negative Contracts:**

NC-1: No Stale "Reserved for Future"
- GIVEN the complete documentation set after this PR
- WHEN searching for "reserved for future" in docs referencing MemoryGiB or KV capacity
- THEN zero matches MUST exist

NC-2: No `simulation_worker` in Golden Dataset
- GIVEN `testdata/goldendataset.json`
- WHEN searching for "simulation_worker"
- THEN zero matches MUST exist

### C) Component Interaction

No new components. This PR modifies documentation only.

```
docs/concepts/roofline.md ──────┐
docs/reference/configuration.md ├── All describe --total-kv-blocks behavior
docs/guide/latency-models.md   ├── sourced from cmd/root.go:160-416
docs/guide/kv-cache.md ────────┘
CLAUDE.md ─── File tree + MemoryGiB description
README.md ─── File tree
testdata/goldendataset.json ─── Binary name
```

Extension friction: N/A (docs only).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| H5: Regenerate golden dataset with `./blis` binary name | String replacement of `blis-cmd` field only | SIMPLIFICATION: The `blis-cmd` field is metadata for reproducibility, not functional. Full regeneration would change metric values (PR #531 already regenerated). Only the binary name string needs fixing. |
| M4: Fix misleading double-log in cmd/root.go | Deferred — docs PR only | DEFERRAL: Code changes should go in a separate PR to keep this one docs-only and Small-tier eligible. |
| L2: Add ITL count note to glossary/core-engine | Included as clarification | ADDITION: Low-effort improvement that prevents future confusion about N vs N-1 ITL entries. |
| M3: README missing `sim/internal/` | Included | ADDITION: Found during audit, easy to fix alongside other README tree updates. |
| M3: README missing 3 example files | Included | ADDITION: Same as above. |
| L3: Add KV auto-calculation section to roofline.md | Not included — MemoryGiB table entry updated instead | SIMPLIFICATION: A full section with formula source and architecture support matrix is scope creep for a docs drift fix. The table entry fix (Task 2) and the user guide admonition (Task 4) together give users the information they need. A dedicated section can be added when roofline.md is next revised. |

### E) Review Guide

**The tricky part:** BC-2 (Resolution Process rewrite). The 4-layer priority chain must be technically accurate — get the ordering wrong and users will misconfigure their KV blocks. Cross-check against `cmd/root.go:160-416`.

**What to scrutinize:** The rewritten Resolution Process in `docs/reference/configuration.md`. Verify the priority ordering matches the actual code execution flow.

**What's safe to skim:** BC-5 (golden dataset binary name replacement) — mechanical find/replace. BC-4 (README tree) — straightforward additions.

**Known debt:** M4 (misleading double-log in `cmd/root.go`) deferred to a separate code PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `CLAUDE.md:230,244` — Fix MemoryGiB description, fix ParseHFConfig capitalization
- `docs/concepts/roofline.md:3,105` — Fix MoE statement, fix MemoryGiB table entry, add KV auto-calc note
- `docs/reference/configuration.md:43,330-346` — Expand footnote, rewrite Resolution Process
- `docs/guide/latency-models.md:101,128` — Add KV auto-calculation note to roofline and crossmodel sections
- `docs/guide/kv-cache.md:17-20` — Update footnote
- `docs/concepts/glossary.md:63-65` — Clarify ITL count
- `README.md:187-233` — Add kv_capacity.go, sim/internal/, 3 example files
- `testdata/goldendataset.json` — Replace `simulation_worker` with `blis` (5 entries)

**Key decisions:** No code changes. M4 deferred. Golden dataset is string-replaced, not regenerated.

### G) Task Breakdown

---

### Task 1: Fix MemoryGiB and ParseHFConfig in CLAUDE.md

**Contracts Implemented:** BC-1 (partial), NC-1 (partial)

**Files:**
- Modify: `CLAUDE.md:230,244`

**Step 1: Fix MemoryGiB description**

In `CLAUDE.md` line 230, replace:
```
│   ├── model_hardware_config.go # ModelConfig, HardwareCalib structs (config types stay in sim/); HardwareCalib includes MemoryGiB (reserved for future KV capacity auto-calculation)
```
with:
```
│   ├── model_hardware_config.go # ModelConfig, HardwareCalib structs (config types stay in sim/); HardwareCalib includes MemoryGiB (used by KV capacity auto-calculation in roofline/crossmodel modes)
```

**Step 2: Fix ParseHFConfig capitalization**

In `CLAUDE.md` line 244, replace:
```
│   ├── config.go              # HFConfig, GetHWConfig(), GetModelConfig(), ValidateRooflineConfig(), parseHWConfig(), parseHFConfig()
```
with:
```
│   ├── config.go              # HFConfig, GetHWConfig(), GetModelConfig(), ValidateRooflineConfig(), parseHWConfig(), ParseHFConfig()
```

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(CLAUDE.md): fix stale MemoryGiB description and ParseHFConfig capitalization

- Update MemoryGiB from 'reserved for future' to 'used by KV capacity auto-calculation' (BC-1)
- Fix parseHFConfig -> ParseHFConfig (exported in PR #474)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Fix roofline.md — MoE statement, MemoryGiB table, KV auto-calc note

**Contracts Implemented:** BC-1 (partial), BC-3, NC-1 (partial)

**Files:**
- Modify: `docs/concepts/roofline.md:3,105`

**Step 1: Fix MoE statement in opening paragraph**

In `docs/concepts/roofline.md` line 3, replace:
```
This document describes the analytical approach used to estimate the GPU latency for a single inference step using a roofline model. This requires no training, and works off-the-shelf for any Huggingface LLM whose `config.json` is saved under `model_configs/`, except Mixture-of-Expert (MoE) models currently.
```
with:
```
This document describes the analytical approach used to estimate the GPU latency for a single inference step using a roofline model. This requires no training, and works off-the-shelf for any Huggingface LLM whose `config.json` is saved under `model_configs/`. For Mixture-of-Experts (MoE) models, use `--latency-model crossmodel` instead — see [Cross-Model Mode](../guide/latency-models.md#cross-model-mode-physics-informed).
```

**Step 2: Fix MemoryGiB table entry**

In `docs/concepts/roofline.md` line 105, replace:
```
| `MemoryGiB` | GPU memory capacity in GiB (reserved for future KV capacity auto-calculation) |
```
with:
```
| `MemoryGiB` | GPU memory capacity in GiB. Used by `CalculateKVBlocks` to auto-derive `--total-kv-blocks` when roofline or crossmodel mode is active and the flag is not explicitly set. |
```

**Step 3: Commit**

```bash
git add docs/concepts/roofline.md
git commit -m "docs(roofline): fix MoE statement and stale MemoryGiB description

- Remove 'except MoE models currently' — crossmodel handles MoE (BC-3)
- Update MemoryGiB from 'reserved for future' to active description (BC-1)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Rewrite configuration.md Resolution Process and footnote

**Contracts Implemented:** BC-2, BC-7

**Files:**
- Modify: `docs/reference/configuration.md:43,330-346`

**Step 1: Expand the --total-kv-blocks footnote**

In `docs/reference/configuration.md` line 43, replace:
```
\* The CLI default is 1,000,000 but `defaults.yaml` overrides this per model when coefficients are loaded. For example, `llama-3.1-8b/H100/TP=2` uses 132,139 blocks. The override only applies if the user did not explicitly set `--total-kv-blocks`.
```
with:
```
\* The effective value of `--total-kv-blocks` depends on the latency backend — see [Resolution Process](#resolution-process) for the full priority chain. In blackbox mode, `defaults.yaml` overrides the 1,000,000 CLI default per model (e.g., `llama-3.1-8b/H100/TP=2` uses 132,139 blocks). In roofline or crossmodel mode, the value is auto-calculated from model architecture and GPU memory via `CalculateKVBlocks`, which supersedes the `defaults.yaml` value. Explicit `--total-kv-blocks` always takes precedence.
```

**Step 2: Rewrite the Resolution Process section**

In `docs/reference/configuration.md`, replace lines 330-346:
```markdown
### Resolution Process

When BLIS starts:

1. If `--latency-model roofline` is set:
   - Auto-resolve model config: check `model_configs/` for existing `config.json`, fetch from HuggingFace on miss (set `HF_TOKEN` for gated models)
   - Auto-resolve hardware config from bundled `hardware_config.json`
   - Load alpha coefficients and `total_kv_blocks` from `defaults.yaml` (beta coefficients are replaced by roofline computation)
   - `--model-config-folder` and `--hardware-config` override auto-resolution when explicitly set
2. If `--alpha-coeffs` and `--beta-coeffs` are not explicitly provided on the CLI and no roofline config is provided:
   - Look up the model in `defaults.yaml` using `--model`, `--hardware`, `--tp`, `--vllm-version`
   - Load alpha/beta coefficients and `total_kv_blocks` from the matching entry
   - Override `--total-kv-blocks` only if the user did not explicitly set it
3. If coefficients are still all-zero (no defaults found) but `--model-config-folder` and `--hardware-config` are provided:
   - Enable roofline mode (implicit activation)
4. If coefficients were explicitly provided via CLI (including explicit zeros):
   - Use them directly, no `defaults.yaml` lookup
```
with:
```markdown
### Resolution Process

When BLIS starts, it resolves latency coefficients and KV block counts through a layered process. Explicit CLI flags always take precedence (R18).

**Latency coefficient resolution:**

1. If `--latency-model roofline` or `--latency-model crossmodel` is set:
   - Auto-resolve model config: check `model_configs/` for existing `config.json`, fetch from HuggingFace on miss (set `HF_TOKEN` for gated models)
   - Auto-resolve hardware config from bundled `hardware_config.json`
   - For roofline: load alpha coefficients and per-model KV blocks from `defaults.yaml` (beta coefficients are replaced by analytical computation). Warns if no per-model KV blocks found
   - For crossmodel: load global alpha + beta coefficients from `crossmodel_defaults` in `defaults.yaml`, and per-model KV blocks if available
   - `--model-config-folder` and `--hardware-config` override auto-resolution when explicitly set
2. If `--alpha-coeffs` and `--beta-coeffs` are not explicitly provided on the CLI and no analytical backend is selected:
   - Look up the model in `defaults.yaml` using `--model`, `--hardware`, `--tp`, `--vllm-version`
   - Load alpha/beta coefficients from the matching entry
3. If coefficients are still all-zero (no defaults found) but `--model-config-folder` and `--hardware-config` are provided:
   - Enable roofline mode (implicit activation)
4. If coefficients were explicitly provided via CLI (including explicit zeros):
   - Use them directly, no `defaults.yaml` lookup

**`--total-kv-blocks` resolution** (highest priority wins):

1. **Explicit CLI flag** — if `--total-kv-blocks` is set, that value is used regardless of backend
2. **Auto-calculation** (roofline/crossmodel only) — when `MemoryGiB > 0` in the hardware config, `CalculateKVBlocks` derives the block count from model architecture and GPU memory, superseding the `defaults.yaml` value. Three failure modes: (a) if `MemoryGiB` is missing from `hardware_config.json`, BLIS warns and falls back to the `defaults.yaml` value (layer 3) or hardcoded default (layer 4); (b) if model architecture params cannot be extracted from `config.json`, BLIS exits with an error; (c) if the calculation itself fails (e.g., unsupported activation function), BLIS exits with an error. Only the `MemoryGiB`-missing case is a graceful fallback — other failures are fatal. Auto-calculation currently requires SwiGLU-family activations (`silu`, `swiglu`, `geglu`); models with other activations (e.g., Falcon's `gelu`) should set `--total-kv-blocks` explicitly
3. **`defaults.yaml`** — per-model block count loaded for the model/GPU/TP combination (e.g., 132,139 for llama-3.1-8b/H100/TP=2). For roofline/crossmodel with `MemoryGiB > 0`, this value is superseded by auto-calculation (layer 2). It remains the effective value only for blackbox mode or when `MemoryGiB` is unavailable in the hardware config
4. **Hardcoded default** — 1,000,000 (CLI flag default, used only when no other source provides a value)
```

**Step 3: Commit**

```bash
git add docs/reference/configuration.md
git commit -m "docs(configuration): rewrite Resolution Process for crossmodel + KV auto-calculation

- Expand --total-kv-blocks footnote with full priority chain (BC-7)
- Add crossmodel to latency coefficient resolution (was roofline-only)
- Add dedicated --total-kv-blocks resolution section with 4-layer priority (BC-2)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Add KV auto-calculation notes to user guides

**Contracts Implemented:** BC-6

**Files:**
- Modify: `docs/guide/latency-models.md:101,128`
- Modify: `docs/guide/kv-cache.md:20`

**Step 1: Add auto-calculation note to latency-models.md roofline section**

In `docs/guide/latency-models.md`, after line 101 (after "For capacity planning, simulate both configurations."), insert:

```markdown

!!! note "Automatic KV block calculation"
    When using roofline or crossmodel mode, `--total-kv-blocks` is automatically derived from model architecture and GPU memory if not explicitly set. The auto-calculated value accounts for TP (KV heads are sharded across ranks; total GPU memory scales with GPU count). Override with `--total-kv-blocks <N>` for non-standard deployments. The auto-calculation uses reference constants (90% GPU utilization, standard activation/overhead budgets matching the llm-d-benchmark capacity planner) and requires SwiGLU-family activations.
```

**Step 1b: Add matching note to crossmodel section**

In `docs/guide/latency-models.md`, after line 128 (after the dense model prefill limitation warning), insert:

```markdown

!!! note "Automatic KV block calculation"
    Like roofline mode, crossmodel auto-derives `--total-kv-blocks` from model architecture and GPU memory when the flag is not set. Override with `--total-kv-blocks <N>` for non-standard deployments. The auto-calculation uses reference constants (90% GPU utilization, standard activation/overhead budgets matching the llm-d-benchmark capacity planner) and requires SwiGLU-family activations (`silu`, `swiglu`, `geglu`).
```

**Step 2: Update kv-cache.md footnote**

In `docs/guide/kv-cache.md` line 20, replace:
```
*The CLI default is 1,000,000 but `defaults.yaml` overrides this per model. For LLaMA 3.1 8B / H100 / TP=2: 132,139 blocks.
```
with:
```
*In blackbox mode, `defaults.yaml` overrides the 1,000,000 CLI default per model (e.g., LLaMA 3.1 8B / H100 / TP=2: 132,139 blocks). In roofline or crossmodel mode, the block count is auto-calculated from model architecture and GPU memory, superseding the `defaults.yaml` value. Explicit `--total-kv-blocks` always wins. See [Configuration Reference](../reference/configuration.md#resolution-process).
```

**Step 3: Commit**

```bash
git add docs/guide/latency-models.md docs/guide/kv-cache.md
git commit -m "docs(guide): add KV auto-calculation notes to latency-models and kv-cache

- Add admonition to roofline section noting auto-calculated --total-kv-blocks (BC-6)
- Add admonition to crossmodel section noting auto-calculated --total-kv-blocks (BC-6)
- Update kv-cache.md footnote with full resolution chain (BC-6)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Clarify ITL count in glossary

**Contracts Implemented:** BC-8

**Files:**
- Modify: `docs/concepts/glossary.md:63-65`

**Step 1: Add ITL count clarification**

In `docs/concepts/glossary.md`, replace lines 63-65:
```markdown
### ITL (Inter-Token Latency)

The observed time between consecutive decode steps for a single request. ITL varies with batch composition changes between steps. Mean ITL is reported as TPOT (Time Per Output Token).
```
with:
```markdown
### ITL (Inter-Token Latency)

The observed time between consecutive decode steps for a single request. A request generating N output tokens produces N-1 ITL entries (the number of inter-token gaps). ITL varies with batch composition changes between steps. Mean ITL is reported as TPOT (Time Per Output Token).
```

**Step 2: Commit**

```bash
git add docs/concepts/glossary.md
git commit -m "docs(glossary): clarify ITL count is N-1 for N output tokens

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Update README file tree

**Contracts Implemented:** BC-4

**Files:**
- Modify: `README.md:187-233`

**Step 1: Add kv_capacity.go to sim/latency/ tree**

In `README.md`, after line 191 (`│   ├── config.go ...`), insert:
```
│   ├── kv_capacity.go     # KV cache block auto-calculation from model architecture + GPU memory
```

**Step 2: Add sim/internal/ to sim/ tree**

In `README.md`, after line 182 (`│   └── model_hardware_config.go ...`), replace `└──` with `├──` and add:
```
│   └── internal/           # Shared internal packages (hash, testutil, util)
```

Note: Line 182 currently uses `└──` which implies it's the last entry. Change it to `├──` since `internal/` follows it.

**Step 3: Add 3 missing example files**

In `README.md`, after line 233 (`│   └── inference-perf-shared-prefix.yaml`), adjust the tree entries. The current last entry uses `└──`. The 3 missing files are `regression_workload_cache_warmup.yaml`, `regression_workload_load_spikes.yaml`, `regression_workload_multiturn.yaml`. Append these after the existing entries (the README uses chronological ordering, not alphabetical), adjusting `└──` on the current last entry to `├──` and using `└──` on the final new entry.

**Step 4: Commit**

```bash
git add README.md
git commit -m "docs(README): add kv_capacity.go, sim/internal/, missing example files to tree

- Add kv_capacity.go to sim/latency/ section (BC-4)
- Add sim/internal/ directory with hash, testutil, util
- Add 3 regression workload example files

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Fix golden dataset binary name

**Contracts Implemented:** BC-5, NC-2

**Files:**
- Modify: `testdata/goldendataset.json` (5 entries)

**Step 1: Replace simulation_worker with blis**

In `testdata/goldendataset.json`, replace all 5 occurrences of `"./simulation_worker run` with `"./blis run`.

**Step 2: Verify no simulation_worker references remain**

Run: `grep -c "simulation_worker" testdata/goldendataset.json`
Expected: `0`

**Step 3: Run golden dataset tests to verify no functional change**

Run: `go test ./sim/... -run TestGolden -v`
Expected: PASS (the `blis-cmd` field is metadata, not used by test assertions)

**Step 4: Commit**

```bash
git add testdata/goldendataset.json
git commit -m "fix(golden): update blis-cmd from simulation_worker to blis

- Replace all 5 occurrences of ./simulation_worker with ./blis (BC-5, NC-2)
- Binary was renamed in workload unification (W0-2)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Final verification

**Step 1: Verify NC-1 — no stale "reserved for future" for MemoryGiB**

Run: `grep -rn "reserved for future" docs/ CLAUDE.md README.md`
Expected: Zero matches referencing MemoryGiB or KV capacity. (Matches in archived plans or unrelated contexts are acceptable.)

**Step 2: Verify NC-2 — no simulation_worker in golden dataset**

Run: `grep "simulation_worker" testdata/goldendataset.json`
Expected: Zero matches.

**Step 3: Build and test**

Run: `go build ./... && go test ./... -count=1`
Expected: Build succeeds, all tests pass.

**Step 4: Lint**

Run: `golangci-lint run ./...`
Expected: No new issues.

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|-------------|
| BC-1 | Task 1, 2 | Manual | grep for "reserved for future" + MemoryGiB |
| BC-2 | Task 3 | Manual | Read Resolution Process, cross-check cmd/root.go |
| BC-3 | Task 2 | Manual | Read roofline.md opening paragraph |
| BC-4 | Task 6 | Manual | Read README tree, verify kv_capacity.go present |
| BC-5 | Task 7 | Automated | `grep -c simulation_worker testdata/goldendataset.json` → 0 |
| BC-6 | Task 4 | Manual | Read latency-models.md and kv-cache.md |
| BC-7 | Task 3 | Manual | Read configuration.md footnote |
| BC-8 | Task 5 | Manual | Read glossary ITL definition, verify N-1 sentence present |
| NC-1 | Task 8 | Automated | `grep -rn "reserved for future" docs/ CLAUDE.md` → 0 MemoryGiB matches |
| NC-2 | Task 7, 8 | Automated | `grep simulation_worker testdata/goldendataset.json` → 0 |

No golden dataset regeneration needed — only the `blis-cmd` metadata string changes.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Resolution Process description doesn't match cmd/root.go | Medium | High | Cross-check every layer against actual code lines | Task 3 |
| MkDocs admonition syntax error | Low | Low | MkDocs build will catch during CI | Task 4 |
| Accidentally changing golden metric values | Low | High | Only replace `simulation_worker` → `blis`, nothing else | Task 7 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (M4 code change deferred)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] CLAUDE.md updated (Task 1)
- [x] No stale references left in CLAUDE.md after fixes
- [x] Documentation DRY: source-of-truth for `--total-kv-blocks` is `cmd/root.go`; all working copies updated
- [x] Deviation log reviewed — M4 deferred, all others addressed
- [x] Each task produces complete changes
- [x] Task dependencies correctly ordered (independent, can run in any order)
- [x] All contracts mapped to tasks
- [x] Golden dataset: only metadata string change, no metric regeneration needed

**Antipattern rules:** N/A for docs-only PR (no new code). R15 (stale references) is the primary rule being addressed.

---

## Appendix: File-Level Details

Omitted — all changes are documented inline in the task steps above. No new types, interfaces, or functions.
