# Docs: Update --roofline → --latency-model roofline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update all documentation references from the removed `--roofline` flag to `--latency-model roofline`.

**The problem today:** PR #475 replaced `--roofline` with `--latency-model roofline` in all Go code, but deferred 7 user-facing doc files + 2 root-level files. The MkDocs site will show stale CLI examples after #475 merges.

**What this PR adds:** Consistent `--latency-model roofline` references across all documentation.

**Architecture:** Pure find-and-replace across .md files with contextual adjustments for flag tables and flow descriptions.

**Source:** GitHub issue #476

**Closes:** Fixes #476

**Behavioral Contracts:**
- BC-1: Zero `--roofline` CLI flag references remain in user-facing docs after merge
- BC-2: Historical plan files annotated as superseded, not rewritten
- BC-3: MkDocs build succeeds (`mkdocs build --strict` equivalent — no broken links)

---

## Tasks

### Task 1: User-facing MkDocs docs (6 files)
Files: `docs/reference/configuration.md`, `docs/guide/latency-models.md`, `docs/reference/models.md`, `docs/concepts/roofline.md`, `docs/getting-started/quickstart.md`

### Task 2: Root-level docs (3 files)
Files: `README.md`, `CONTRIBUTING.md`, `CLAUDE.md`

### Task 3: Historical plan annotation (1 file)
File: `docs/plans/2026-02-25-roofline-flag-design.md`

### Task 4: Verification
`grep -r '\-\-roofline' docs/guide/ docs/getting-started/ docs/reference/ docs/concepts/ README.md CONTRIBUTING.md CLAUDE.md` returns 0 matches.
