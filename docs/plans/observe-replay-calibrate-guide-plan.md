# Observe / Replay / Calibrate Guide — Implementation Plan

**Goal:** Create an end-to-end user guide for the observe/replay/calibrate pipeline with worked examples and complete flag documentation.
**Source:** [#716](https://github.com/inference-sim/inference-sim/issues/716) (parent: #715)
**Closes:** `Fixes #716`

## Deviation Log

| Item | Source says | Plan says | Reason |
|------|-----------|-----------|--------|
| Observe flag count | 27 flags | 26 flags | CORRECTION — `cmd/observe_cmd.go` registers exactly 26 flags (lines 89-120). Source issue inherited an incorrect count. |

## Behavioral Contracts

BC-1: Guide page renders in MkDocs
- GIVEN the guide markdown file exists at `docs/guide/observe-replay-calibrate.md`
- WHEN a user builds the MkDocs site with `mkdocs build`
- THEN the page renders without errors and appears in the User Guide nav section

BC-2: All 26 observe flags documented
- GIVEN a user reads the observe section
- WHEN they look for any observe CLI flag
- THEN they find its name, type, default value, and description in a categorized table

BC-3: All 7 calibrate flags documented
- GIVEN a user reads the calibrate section
- WHEN they look for any calibrate CLI flag
- THEN they find its name, type, default value, and description

BC-4: Replay behavior documented
- GIVEN a user reads the replay section
- WHEN they want to understand how replay differs from `blis run`
- THEN they find an explanation of TraceV2 input, horizon auto-computation, and SimResult output

BC-5: End-to-end worked example
- GIVEN a user reads the worked example section
- WHEN they follow the commands in sequence
- THEN they understand the full pipeline: workload spec → observe → replay → calibrate → interpret

BC-6: Guide index updated
- GIVEN a user visits the User Guide index page
- WHEN they scan the guides table
- THEN they see an entry for "Observe / Replay / Calibrate" with a description of when to use it

## Tasks

### Task 1: Create guide page (BC-1, BC-2, BC-3, BC-4, BC-5)

**Files:** create `docs/guide/observe-replay-calibrate.md`

**Content structure:**
1. Title + one-sentence intro + quick example
2. Pipeline overview (3-step: observe → replay → calibrate)
3. `blis observe` section with 4 flag tables (required, workload input, optional, distribution synthesis) + examples for workload-spec mode, rate mode, chat API, streaming
4. `blis replay` section with key flags + how it differs from `blis run`
5. `blis calibrate` section with 7 flags (noting sentinel defaults: `--warmup-requests -1` and `--network-rtt-us -1` mean "use trace header value") + report JSON interpretation
6. Worked example: end-to-end walkthrough
7. Tips & troubleshooting

**Verify:** `mkdocs build` succeeds (if mkdocs installed), otherwise visual inspection of markdown
**Commit:** `docs(guide): add observe/replay/calibrate pipeline guide (BC-1 through BC-5)`

### Task 2: Update MkDocs nav and guide index (BC-1, BC-6)

**Files:** modify `mkdocs.yml`, modify `docs/guide/index.md`

**Impl:**
- Add `Observe / Replay / Calibrate: guide/observe-replay-calibrate.md` to `mkdocs.yml` nav under User Guide, after "Metrics & Results"
- Add row to guide index table: "Observe / Replay / Calibrate" | "Validating simulator accuracy against real servers"
- Add reading path: **Calibration:** Latency Models → Workload Specifications → Observe / Replay / Calibrate → Metrics & Results

**Verify:** MkDocs build succeeds, nav entry visible
**Commit:** `docs(nav): add observe-replay-calibrate to User Guide nav (BC-1, BC-6)`

## Sanity Checklist

- [x] No unnecessary abstractions — pure docs change
- [x] No feature creep — exactly what #716 requests
- [x] No unexercised flags or interfaces — N/A (docs only)
- [x] No partial implementations — all acceptance criteria covered
- [x] No breaking changes — additive only
- [x] CLAUDE.md — no update needed (no new files/packages, no CLI flags, no architecture changes)
- [x] Documentation DRY — this creates a new page, doesn't modify canonical sources
- [x] Task dependencies correctly ordered — Task 1 creates the file, Task 2 links it
