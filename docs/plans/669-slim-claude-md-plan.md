# Slim CLAUDE.md Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce CLAUDE.md from ~416 lines to ~210 lines by extracting reference material to dedicated docs, replacing each section with a 1-2 line pointer.

**The problem today:** CLAUDE.md is loaded into every Claude Code session's context window. ~210 lines are reference material (File Organization tree, Latency Estimation details, Antipattern Prevention table, Key Data Flow diagram) that is rarely needed during a given task. Every session pays this context tax.

**What this PR adds:**
1. Replaces the File Organization tree (~150 lines) with a pointer to a new `docs/reference/project-structure.md`
2. Replaces the Latency Estimation section (~26 lines) with a pointer to the existing `docs/guide/latency-models.md`
3. Replaces the Antipattern Prevention table (~25 lines) with a pointer to the canonical `docs/contributing/standards/rules.md`
4. Replaces the Key Data Flow diagram (~8 lines) with a pointer to the existing `docs/concepts/architecture.md`

**Why this matters:** Reduces always-loaded context by ~200 lines per session, freeing token budget for actual task work. No information is lost — just moved from always-loaded to read-on-demand.

**Architecture:** Pure documentation change. CLAUDE.md is edited in place. One new file (`docs/reference/project-structure.md`) is created. The source-of-truth map in `docs/contributing/standards/principles.md` is updated to reflect the new canonical location of the file organization tree. MkDocs nav in `mkdocs.yml` is updated to include the new page.

**Source:** GitHub issue #669

**Closes:** Fixes #669

---

## Part 1: Design

### A. Behavioral Contracts

**BC-1: File Organization pointer replacement**
- GIVEN CLAUDE.md contains the File Organization tree (lines 201-350)
- WHEN the tree is replaced with a pointer
- THEN CLAUDE.md contains a 2-line pointer to `docs/reference/project-structure.md` and the full tree exists in that file

**BC-2: Latency Estimation pointer replacement**
- GIVEN CLAUDE.md contains the Latency Estimation section (lines 352-377)
- WHEN the section is replaced with a pointer
- THEN CLAUDE.md contains a 1-line pointer to `docs/guide/latency-models.md` with a note about the four modes

**BC-3: Antipattern Prevention table replacement**
- GIVEN CLAUDE.md contains the full R1-R23 table (lines 125-155)
- WHEN the table is replaced with a pointer
- THEN CLAUDE.md retains the canonical source header and pointer but removes the duplicated table. The "Code Review Standards" section (line 89) no longer says "below" and instead references `docs/contributing/standards/rules.md` directly

**BC-4: Key Data Flow pointer replacement**
- GIVEN CLAUDE.md contains the Key Data Flow diagram (lines 379-386)
- WHEN the diagram is replaced with a pointer
- THEN CLAUDE.md contains a 1-line pointer to `docs/concepts/architecture.md`

**BC-5: Source-of-truth map consistency**
- GIVEN the source-of-truth map tracks CLAUDE.md as canonical for file organization (line 98 of `docs/contributing/standards/principles.md`)
- WHEN the tree moves to `docs/reference/project-structure.md`
- THEN the map row changes canonical source from `CLAUDE.md (File Organization tree)` to `docs/reference/project-structure.md`, and working copies become `CLAUDE.md (pointer), README.md (Project Structure tree)`

**BC-6: MkDocs navigation**
- GIVEN the new file `docs/reference/project-structure.md` exists
- WHEN the MkDocs site is built
- THEN the project structure page appears under the Reference section in navigation

### B. Component Interaction

No code components involved. Only documentation files.

### C. Risk Assessment

- **Low risk**: Pure documentation movement. No code changes. No test changes.
- **MkDocs build risk**: New page must be in nav or it will be orphaned. Mitigated by updating mkdocs.yml.

### D. Deviation Log

- Issue suggests moving tree to `docs/reference/project-structure.md` or inline in `docs/reference/configuration.md`. Chose dedicated file for clarity.
- Issue says "Move Latency Estimation section to `docs/reference/latency-models.md`". Since `docs/guide/latency-models.md` already exists with comprehensive coverage of all 4 modes, the pointer targets the existing file instead of creating a new reference file.
- Issue says "Replace the Key Data Flow with a one-line pointer" without specifying target. Chose `docs/concepts/architecture.md` since it covers the request processing pipeline.
- README.md line 147 says "see CLAUDE.md" for the authoritative file organization. After this PR, the canonical source moves to `docs/reference/project-structure.md`. Updating README.md is deferred — it was not in the issue scope and the README's Project Structure tree remains a valid (if not authoritative) working copy.

### E. Test Strategy

No tests needed — documentation-only change. Verification via line count and content presence checks.

---

## Part 2: Tasks

### Task 1: Create `docs/reference/project-structure.md` with the File Organization tree

**Test (verification):**
```bash
# Verify file exists and contains the tree
test -f docs/reference/project-structure.md && grep -c "inference-sim/" docs/reference/project-structure.md
# Expected: file exists, grep count > 0
```

**Implementation:**
1. Create `docs/reference/project-structure.md` with the full File Organization tree content from CLAUDE.md (lines 201-350), using title "# Project Structure" and preserving the existing intro sentence
2. Update `mkdocs.yml` to add the new page under the Reference section nav with label "Project Structure: reference/project-structure.md"
3. Fix stale `mkdocs.yml` nav labels while editing: "Antipattern Rules (R1-R20)" → "Antipattern Rules (R1-R23)", "System Invariants (INV-1-8)" → "System Invariants (INV-1-INV-11)"

**Commit:** `docs(reference): add project-structure.md with file organization tree`

---

### Task 2: Replace CLAUDE.md sections with pointers

**Test (verification):**
```bash
# Verify CLAUDE.md line count is reduced
wc -l CLAUDE.md
# Expected: ~210 lines (down from 416)

# Verify pointers exist
grep -c "docs/reference/project-structure.md" CLAUDE.md
grep -c "docs/guide/latency-models.md" CLAUDE.md
grep -c "docs/contributing/standards/rules.md" CLAUDE.md
grep -c "docs/concepts/architecture.md" CLAUDE.md
# Expected: each returns >= 1
```

**Implementation:**
1. Replace the File Organization section (lines 201-350) with a 2-line pointer
2. Replace the Latency Estimation section (lines 352-377) with a 2-line pointer
3. Replace the Key Data Flow section (lines 379-386) with a 1-line pointer
4. Replace the Antipattern Prevention table (lines 131-155) with a 1-line pointer, keeping the canonical source header
5. Update the "Code Review Standards" section (line 89) to reference `docs/contributing/standards/rules.md` instead of "below"

**Commit:** `docs(claude-md): replace 4 reference sections with pointers (~200 lines removed)`

---

### Task 3: Update source-of-truth map

**Test (verification):**
```bash
# Verify map entry updated
grep "project-structure" docs/contributing/standards/principles.md
# Expected: new canonical location appears
```

**Implementation:**
1. In `docs/contributing/standards/principles.md` line 98, update the "File organization and architecture" row: canonical source from `CLAUDE.md (File Organization tree)` → `docs/reference/project-structure.md`; working copies from `README.md (Project Structure tree)` → `CLAUDE.md (pointer), README.md (Project Structure tree)`
2. Fix stale invariant count in same file line 93: `System invariants (INV-1–INV-8)` → `System invariants (INV-1–INV-11)`

**Commit:** `docs(standards): update source-of-truth map for project structure relocation`

---

## Part 3: Appendix

### J. Files Modified

| File | Action | Description |
|------|--------|-------------|
| `CLAUDE.md` | Edit | Replace 4 sections with pointers (~200 lines removed) |
| `docs/reference/project-structure.md` | Create | New file with File Organization tree |
| `mkdocs.yml` | Edit | Add project-structure.md to Reference nav; fix stale R1-R20→R1-R23 and INV-1-8→INV-1-11 labels |
| `docs/contributing/standards/principles.md` | Edit | Update source-of-truth map entry + fix stale INV-1-8→INV-1-11 count |
