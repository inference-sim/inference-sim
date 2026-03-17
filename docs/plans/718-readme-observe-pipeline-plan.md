# 718: Update README with observe/replay/calibrate pipeline

**Goal:** Update README.md to cover the observe/replay/calibrate pipeline and fix the stale project structure tree.
**Source:** [#718](https://github.com/inference-sim/inference-sim/issues/718) (parent: #715)
**Closes:** #718
**Tier:** Small (docs-only, single file, no behavioral changes)
**Clarifications:** None needed — issue is unambiguous.

## Behavioral Contracts

**BC-1: All six blis commands shown in Usage section**
- GIVEN the README Usage section
- WHEN a user reads it
- THEN they see examples for: `blis run`, `blis observe`, `blis replay`, `blis calibrate`, `blis convert` (including `inference-perf`), and `blis compose`

**BC-2: Project structure tree matches reference doc**
- GIVEN the README project structure tree
- WHEN compared against `docs/reference/project-structure.md`
- THEN every file that appears in the reference doc is present in the README tree, with accurate descriptions, in the same ordering

## Tasks

### Task 1: Add observe/replay/calibrate usage examples

Add three new subsections to the README Usage section (after "Blackbox mode", before "Convert workload formats"):

1. **Observe real server latency** — `blis observe` with basic and chat format examples (matching CLAUDE.md)
2. **Replay traces through simulator** — `blis replay` example
3. **Calibrate simulator accuracy** — `blis calibrate` example

Also add `inference-perf` to the existing Convert section.

**Test:** Visual inspection — all 6 commands present in Usage.

### Task 2: Sync project structure tree with reference doc

Sync the entire README project structure tree with `docs/reference/project-structure.md`, preserving the reference doc's ordering:

**cmd/ section:**
- Add `replay.go` entry (with description from reference doc)
- Add `calibrate.go` entry (with description from reference doc)
- Add `observe_cmd.go` entry (with description from reference doc)
- Fix `observe.go` description: change "observe-predict-calibrate" to match reference doc description
- Match ordering from reference doc: `root.go`, `replay.go`, `calibrate.go`, `observe.go`, `observe_cmd.go`, `convert.go`, `compose.go`, `hfconfig.go`, `default_config.go`

**sim/latency/ section:**
- Add `trained_roofline.go` entry (present in reference, missing in README)

**sim/workload/ section:**
- Add `session.go` entry (present in reference, missing in README)
- Update `convert.go` description to include `ComposeSpecs` (matching reference doc)

**Test:** Diff README tree against reference doc — all entries match across all directories.

## Sanity Checklist

- [ ] No behavioral code changes
- [ ] No new files created (README.md edit only)
- [ ] Examples match CLAUDE.md usage section
- [ ] Tree matches `docs/reference/project-structure.md`
