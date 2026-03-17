# docs: expand configuration.md with observe/replay/calibrate flags — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add reference documentation for `blis observe`, `blis replay`, `blis calibrate`, `blis convert`, and `blis compose` CLI flags to the configuration reference page.
**Source:** GitHub issue #717 (child of #715 documentation umbrella)
**Closes:** Fixes #717

## Behavioral Contracts

BC-1: Observe flags documented
- GIVEN a user reads `docs/reference/configuration.md`
- WHEN they look for `blis observe` flags
- THEN they find all 27 flags organized into Required, Workload Input, Optional, and Distribution Synthesis groups, each with name, type, default, and description

BC-2: Replay flags documented
- GIVEN a user reads `docs/reference/configuration.md`
- WHEN they look for `blis replay` flags
- THEN they find the shared sim-config flags (via cross-reference to existing sections) plus replay-specific flags (`--trace-header`, `--trace-data`), with a note that `--results-path` writes `[]SimResult` JSON (not `MetricsOutput` JSON as in `blis run`)

BC-3: Calibrate flags documented
- GIVEN a user reads `docs/reference/configuration.md`
- WHEN they look for `blis calibrate` flags
- THEN they find all 7 flags with name, type, default, and description

BC-4: Convert and compose flags documented
- GIVEN a user reads `docs/reference/configuration.md`
- WHEN they look for `blis convert` and `blis compose` flags
- THEN they find flags for each convert subcommand (preset, servegen, infperf) and the compose command

## Tasks

### Task 1: Add observe command flags section (BC-1)

**Files:** modify `docs/reference/configuration.md`

**Impl:**

Add after the existing `## CLI Flag Summary by Sub-Config` section (end of file), a new top-level section `## blis observe` with four flag tables:

- **Required** (4 flags): `--server-url`, `--model`, `--trace-header`, `--trace-data`
- **Workload Input** (2 flags): `--workload-spec`, `--rate`
- **Optional** (7 flags): `--api-key`, `--server-type`, `--max-concurrency`, `--warmup-requests`, `--no-streaming`, `--seed`, `--horizon`, `--num-requests`
- **Distribution Synthesis** (14 flags): `--prompt-tokens` through `--prefix-tokens`, `--api-format`, `--unconstrained-output`, `--rtt-ms`

Source: `cmd/observe_cmd.go:87-122`

**Verify:** `mkdocs build` (if available) or visual inspection of markdown table formatting
**Commit:** `docs(config): add blis observe flags (BC-1)`

### Task 2: Add replay command flags section (BC-2)

**Files:** modify `docs/reference/configuration.md`

**Impl:**

Add `## blis replay` section after the observe section. Note that replay shares all sim-config flags with `blis run` (via `registerSimConfigFlags`) and adds two trace-specific flags. Include:

- Prose noting shared flags with cross-reference to existing sections
- **Replay-specific** table (2 flags): `--trace-header`, `--trace-data`
- A callout noting the `--results-path` semantic difference: `blis run` writes `MetricsOutput` JSON; `blis replay` writes `[]SimResult` JSON (request_id, ttft_us, e2e_us, input_tokens, output_tokens) for `blis calibrate` consumption.

Source: `cmd/replay.go:701-708`

**Verify:** Visual inspection of markdown
**Commit:** `docs(config): add blis replay flags (BC-2)`

### Task 3: Add calibrate command flags section (BC-3)

**Files:** modify `docs/reference/configuration.md`

**Impl:**

Add `## blis calibrate` section with a single table of 7 flags:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--trace-header` | string | "" | Path to TraceV2 header YAML file (from blis observe; required) |
| `--trace-data` | string | "" | Path to TraceV2 data CSV file (from blis observe; required) |
| `--sim-results` | string | "" | Path to SimResult JSON file (from blis replay --results-path; required) |
| `--report` | string | "" | Path to write calibration report JSON (required) |
| `--warmup-requests` | int | -1 | Number of initial requests to exclude (default: from trace header warm_up_requests; pass 0 to include all) |
| `--network-rtt-us` | int64 | -1 | Network RTT in microseconds added to sim-side latencies (default: from trace header network.measured_rtt_ms) |
| `--network-bandwidth-mbps` | float64 | 0 | Network bandwidth in Mbps for upload/download delay calculation (0 = no delay) |

Source: `cmd/calibrate.go:156-164`

**Verify:** Visual inspection of markdown
**Commit:** `docs(config): add blis calibrate flags (BC-3)`

### Task 4: Add convert and compose command flags section (BC-4)

**Files:** modify `docs/reference/configuration.md`

**Impl:**

Add `## blis convert` section with subsections for each subcommand:

- **`blis convert preset`**: `--name`, `--rate`, `--num-requests`, `--defaults-filepath`
- **`blis convert servegen`**: `--path`
- **`blis convert infperf`**: `--spec`

Add `## blis compose` section:
- `--from` (string array, repeatable): Path to v2 WorkloadSpec YAML file

Source: `cmd/convert.go:118-127`, `cmd/compose.go:39`

**Verify:** Visual inspection of markdown
**Commit:** `docs(config): add blis convert and compose flags (BC-4)`

## Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] CLAUDE.md: no update needed (no new files/packages, no new CLI flags, no file organization changes)
- [x] No stale references
- [x] Documentation DRY: this PR does not modify any canonical source in the source-of-truth map
- [x] Deviation log: No deviations — issue #717 asked to "consider" convert/compose; we include them (ADDITION: minimal effort, completes the reference page)
- [x] Task dependencies correctly ordered (independent sections, order is logical not required)
- [x] All contracts mapped to tasks
