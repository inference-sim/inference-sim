# Parameter Resolution Reference Doc Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a comprehensive parameter resolution section to `docs/reference/configuration.md` documenting precedence chains, unit gotchas, and common confusion scenarios.
**Source:** GitHub issue #676
**Closes:** Fixes #676

## Behavioral Contracts

BC-1: Precedence Chain Completeness
- GIVEN a user reading `docs/reference/configuration.md`
- WHEN they look up any parameter category (latency, KV cache, workload, routing, batch)
- THEN they find a numbered precedence chain showing all resolution layers from highest to lowest priority
- MECHANISM: Add a "Parameter Resolution by Category" subsection with per-category chains

BC-2: Unit Gotchas Table
- GIVEN a user or experiment author working with BLIS output JSON
- WHEN they need to interpret metric field units
- THEN they find a table listing known unit inconsistencies (field name, actual unit, historical context)
- MECHANISM: Add a "Known Unit Gotchas" subsection with a table

BC-3: Common Confusion Scenarios
- GIVEN a user who encounters unexpected parameter behavior
- WHEN they check the configuration reference
- THEN they find documented scenarios with correct resolution explanations (e.g., capacity estimates using wrong defaults, `--rate` vs `aggregate_rate`)
- MECHANISM: Add a "Common Pitfalls" subsection with scenario descriptions

BC-4: Cross-Reference Discoverability
- GIVEN a user reading workloads guide or experiment standards
- WHEN they encounter parameter precedence questions
- THEN they find a cross-reference pointing to the parameter resolution section in configuration.md
- MECHANISM: Add admonition callouts in `docs/guide/workloads.md` and `docs/contributing/standards/experiments.md`

## Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| `scheduling_delay_ms` is in μs (ticks) | Documents it as **fixed to ms** by BC-14, with note about stale hypothesis scripts | CORRECTION — code was fixed in the fix-audit-bugs PR; issue text is outdated |
| Edit `docs/reference/configuration.md` — add Parameter Resolution section | Expands existing "Configuration Precedence" and adds subsections inline | SIMPLIFICATION — the file already has a basic precedence section; expanding it is cleaner than creating a separate section |

## Tasks

### Task 1: Expand parameter resolution section in configuration.md (BC-1, BC-2, BC-3)

**Files:** modify `docs/reference/configuration.md`

**Impl:**

After the existing "Configuration Precedence" section (line 17), add three new subsections:

**1. "Parameter Resolution by Category"** — per-category precedence chains for:
- **Latency coefficients**: CLI `--alpha/beta-coeffs` → `defaults.yaml` models[] entry → error (blackbox) or analytical (roofline/crossmodel/trained-roofline)
- **Hardware/TP**: CLI `--hardware`/`--tp` → `defaults.yaml` defaults[] entry → error
- **KV cache blocks**: (already documented in Resolution Process — add cross-reference)
- **Workload parameters**: `--workload-spec` YAML → CLI distribution flags → preset from `defaults.yaml` → hardcoded defaults. Note: `--seed`, `--horizon`, `--num-requests` override YAML even when `--workload-spec` is set
- **Routing/admission/scheduling**: CLI flags → `--policy-config` YAML bundle → hardcoded defaults
- **Batch formation**: CLI flags → hardcoded defaults (no YAML override path)

**2. "Known Unit Gotchas"** — table with:
- `scheduling_delay_ms`: now in ms (fixed by BC-14); old hypothesis scripts divide by 1000 unnecessarily
- `scheduling_delay_p99_ms` (aggregate): always was in ms
- Simulation clock: ticks = microseconds (1 tick = 1μs)
- `--horizon`: in ticks (microseconds), not seconds
- `--admission-latency` / `--routing-latency`: in microseconds
- `think_time_us` in workload YAML: microseconds
- `aggregate_rate` / `--rate`: requests per second

**3. "Common Pitfalls"** — scenarios:
- **Capacity estimate mismatch (#390)**: CLI defaults (512/512 tokens) vs workload YAML (256/128) → capacity estimates off by ~1.5x. Resolution: always derive capacity from the actual workload spec, not CLI defaults.
- **`--rate` does NOT override workload-spec YAML**: `--rate` only applies in CLI distribution mode. When `--workload-spec` is set, use `aggregate_rate` in the YAML.
- **`aggregate_rate` override for inference-perf specs**: When converting inference-perf specs, per-stage rates override user-specified `aggregate_rate` (with warning). See `sim/workload/generator.go`.
- **`--total-kv-blocks` phantom default**: CLI default is 1,000,000 but `defaults.yaml` or auto-calculation almost always overrides it. The 1M value is a fallback, not a realistic default.
- **`enable_multi_turn_chat` semantic mismatch**: inference-perf's `enable_multi_turn_chat` vs BLIS's `multi_turn.single_session` — similar concept, different semantics (#517).

**Verify:** Documentation renders correctly — no broken links or formatting issues.
**Lint:** N/A (docs only)
**Commit:** `docs(config): add per-category parameter resolution chains and unit gotchas (BC-1, BC-2, BC-3)`

### Task 2: Add cross-references from workloads.md and experiments.md (BC-4)

**Files:** modify `docs/guide/workloads.md`, modify `docs/contributing/standards/experiments.md`

**Impl:**

In `docs/guide/workloads.md`, after the "Estimating Capacity" warning admonition (line 287), add:

```markdown
!!! tip "Parameter resolution reference"
    For the complete precedence chain of how CLI flags, workload-spec YAML, and `defaults.yaml` interact, see [Parameter Resolution by Category](../reference/configuration.md#parameter-resolution-by-category) in the Configuration Reference.
```

In `docs/contributing/standards/experiments.md`, after the replay validation caveat paragraph (around line 67), add:

```markdown
!!! tip "Parameter resolution"
    Experiment setup errors often stem from parameter precedence confusion (e.g., CLI defaults overriding intended workload values). See [Common Pitfalls](../../reference/configuration.md#common-pitfalls) in the Configuration Reference for documented scenarios.
```

**Verify:** Cross-reference links resolve correctly.
**Lint:** N/A (docs only)
**Commit:** `docs(workloads,experiments): add parameter resolution cross-references (BC-4)`

### Task 3: Fix stale scheduling_delay_ms warning in results.md

**Files:** modify `docs/guide/results.md`

**Impl:**

Update the stale warning at line 33-34 from claiming `scheduling_delay_ms` is in ticks to noting it's in milliseconds (fixed by BC-14). This is discovered pre-existing documentation debt — fixing it here is in-scope because the unit gotchas table in Task 1 references this field.

**Verify:** Warning text is accurate.
**Lint:** N/A (docs only)
**Commit:** `docs(results): fix stale scheduling_delay_ms unit warning`

## Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (docs-only, no code changes)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] CLAUDE.md update not needed (no new files/packages/CLI flags)
- [x] Documentation DRY: configuration.md is the canonical source for parameter resolution; cross-references point to it
- [x] Deviation log reviewed — scheduling_delay_ms correction documented
- [x] Each task produces working content
- [x] Task dependencies correctly ordered (Task 2 depends on Task 1 for anchor links)
- [x] All contracts mapped to tasks (BC-1,2,3→Task 1; BC-4→Task 2)
