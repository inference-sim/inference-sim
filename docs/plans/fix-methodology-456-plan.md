# Fix Strategy Evolution Methodology Page Issues — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix four post-merge issues in the Strategy Evolution methodology documentation: premature principles page, broken Mermaid line breaks, compressed overview diagram, and missing skills provenance.

**The problem today:** PR #455 shipped with `\n` literals in Mermaid nodes (should be `<br/>`), a horizontal overview diagram that compresses fonts, the principles catalog exposed before the simulator stabilizes, and no information about where to get the referenced Claude Code skills.

**What this PR adds:**
1. Removes principles from nav (keeps file as draft)
2. Fixes Mermaid `\n` → `<br/>` across all diagrams
3. Switches overview to top-down layout
4. Adds skills provenance section

**Architecture:** Edits to 3 existing files: `mkdocs.yml`, `docs/methodology/strategy-evolution.md`, `docs/methodology/index.md`. No new files.

**Source:** GitHub issue #456

**Closes:** Fixes #456

---

## Behavioral Contracts

BC-1: Mermaid Line Breaks Render Correctly
- GIVEN a Mermaid flowchart node with multi-line text
- WHEN the site is built and viewed
- THEN the text MUST display on separate lines (not as literal `\n`)
- MECHANISM: Use `<br/>` instead of `\n` in Mermaid node labels

BC-2: Overview Diagram Readable
- GIVEN the Phase 1-5 overview flowchart
- WHEN rendered in a browser
- THEN all node text MUST be readable without zooming
- MECHANISM: `flowchart TD` (top-down) instead of `flowchart LR` (left-right)

BC-3: Principles Not in Nav
- GIVEN the MkDocs site navigation
- WHEN a user browses the Methodology section
- THEN they MUST see only "Strategy Evolution" (not "Discovered Principles")

BC-4: Skills Provenance
- GIVEN a reader who wants to use the skills referenced in the inventory
- WHEN they read the Skills and Tools Inventory section
- THEN they MUST find information about where to install/obtain the skills

---

## Task Breakdown

### Task 1: Fix Mermaid line breaks and overview layout

**Files:** `docs/methodology/strategy-evolution.md`

**Step 1:** Replace all `\n` with `<br/>` in Mermaid node labels. Change overview from `flowchart LR` to `flowchart TD`.

**Step 2:** Verify `mkdocs build --strict` passes.

**Step 3:** Commit.

### Task 2: Remove principles from nav, update index

**Files:** `mkdocs.yml`, `docs/methodology/index.md`

**Step 1:** Remove `Discovered Principles: methodology/principles.md` from mkdocs.yml nav.

**Step 2:** Update index.md to mark principles as "coming soon."

**Step 3:** Update the cross-reference in strategy-evolution.md Phase 5 section (currently links to principles.md).

**Step 4:** Commit.

### Task 3: Add skills provenance section

**Files:** `docs/methodology/strategy-evolution.md`

**Step 1:** Add a provenance note below the skills inventory table explaining where the skills come from.

**Step 2:** Commit.

### Task 4: Final verification

Verify build, nav structure, cross-refs.
