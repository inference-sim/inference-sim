# Documentation Restructuring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Restructure the MkDocs Material documentation site from a filesystem-mirrored layout into an audience-oriented information architecture with dedicated Getting Started, User Guide, Concepts, Reference, and Contributing sections — and write the missing user-facing guide content.

**The problem today:** The docs nav has 6 flat top-level sections that map to directories (`Design/`, `Standards/`, `Process/`, `Templates/`). A capacity planner or inference platform engineer who wants to use BLIS sees contributor-internal content (antipattern rules, convergence protocols) at the same level as user documentation. There are no tutorials, no how-to guides, and no task-oriented content like "Your First Capacity Planning Run" or "Comparing Routing Policies." The Getting Started section is 3 lines of shell commands. The hypothesis experimentation workflow — one of BLIS's most powerful features — is invisible to users.

**What this PR adds:**
1. **Audience-oriented nav** — reorganized into Getting Started (new users), User Guide (active users), Concepts (understanding), Reference (lookup), and Contributing (developers), with Material theme tabs for top-level navigation
2. **User-facing guide content** — 8 new guide pages covering capacity planning, routing policies, KV cache tuning, roofline mode, workload specifications, cluster simulation, interpreting results, and hypothesis-driven experimentation
3. **Getting Started section** — "What is BLIS?", installation, quickstart, and a capacity planning tutorial that walks through a complete end-to-end workflow
4. **Material theme improvements** — navigation tabs, section indexes, footer navigation, and better use of admonitions

**Why this matters:** Documentation is the primary interface for new users. Without audience-oriented organization and guided content, adoption requires reading source code or contributor-internal documents. This restructuring makes BLIS accessible to its target audience (capacity planners, inference platform engineers, researchers) without reducing contributor documentation quality.

**Architecture:** Docs-only PR. Files move from `docs/design/` → `docs/concepts/`, `docs/design/configuration.md` → `docs/reference/configuration.md`, `docs/process/` + `docs/standards/` + `docs/templates/` → under `docs/contributing/`. New user-facing pages created in `docs/getting-started/` and `docs/guide/`. `mkdocs.yml` rewritten with new nav structure. `CONTRIBUTING.md` updated with new paths (sed hack removed). No Go code changes.

**Source:** GitHub issue #449

**Closes:** Fixes #449

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR restructures the BLIS documentation site from a 6-section flat nav mirroring the filesystem into a 5-section audience-oriented architecture. Existing design documentation moves under Concepts. Configuration and model references move under Reference. All contributor-facing content (process, standards, templates, extension recipes) consolidates under Contributing. Eight new user-facing guide pages are written. The Material theme is configured with navigation tabs, section indexes, and footer navigation.

No Go code is modified. No existing content is deleted — only moved and cross-references updated. The `docs/plans/` directory remains excluded via `exclude_docs`.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Nav Structure
- GIVEN the restructured `mkdocs.yml`
- WHEN a user visits the documentation site
- THEN they see 5 top-level tabs: Getting Started, User Guide, Concepts, Reference, Contributing
- MECHANISM: `navigation.tabs` in Material theme + restructured `nav:` section

BC-2: Getting Started Journey
- GIVEN a new user who has never used BLIS
- WHEN they navigate to Getting Started
- THEN they find pages in reading order: What is BLIS → Installation → Quick Start → Tutorial
- MECHANISM: Section with `index.md` + 4 ordered pages

BC-3: User Guide Coverage
- GIVEN a user wanting to accomplish a specific task
- WHEN they navigate to User Guide
- THEN they find a section index plus 7 task-oriented pages: routing policies, KV cache & memory management, roofline mode, workload specifications, cluster simulation, interpreting results, and hypothesis-driven experimentation
- MECHANISM: 8 files total (1 index.md + 7 guide pages) with practical examples and CLI commands

BC-4: Contributor Content Consolidation
- GIVEN contributor-facing content currently scattered across 4 top-level sections
- WHEN the docs are restructured
- THEN all contributor content (process, standards, templates, extension recipes) is accessible under a single Contributing section
- MECHANISM: File moves into `docs/contributing/` subdirectories + updated nav

BC-5: Cross-Reference Integrity
- GIVEN files that move to new paths
- WHEN any page contains a relative link to a moved file
- THEN the link resolves correctly
- MECHANISM: All relative links updated in moved files; `mkdocs serve` produces no broken-link warnings

BC-6: Material Theme Features
- GIVEN the current theme configuration
- WHEN `navigation.tabs` and `navigation.indexes` are enabled
- THEN top-level sections render as tabs and section landing pages use `index.md` files
- MECHANISM: Theme feature flags in `mkdocs.yml`

BC-7: Existing Content Preservation
- GIVEN all existing documentation content
- WHEN files are moved to new paths
- THEN no content is deleted or truncated — only moved and cross-references updated
- MECHANISM: `git mv` for file moves; content diffs show only link updates

**Negative Contracts:**

BC-8: No Broken Links
- GIVEN the restructured documentation
- WHEN `mkdocs serve` builds the site
- THEN there MUST be zero broken internal links or missing pages
- MECHANISM: Serve-time validation + manual verification

BC-9: No Content Duplication
- GIVEN contributor docs that appear in both old and new locations
- WHEN the restructuring is complete
- THEN each piece of content exists in exactly one location
- MECHANISM: File moves (not copies); old paths cease to exist

BC-10: No Scope Creep
- GIVEN this is a docs-only PR
- WHEN the implementation is complete
- THEN zero Go source files are modified and `go test ./...` produces identical output
- MECHANISM: Only `docs/`, `mkdocs.yml`, `CONTRIBUTING.md`, `CLAUDE.md`, `README.md`, `.github/workflows/`, `.github/PULL_REQUEST_TEMPLATE.md`, `.claude/skills/`, and `hypotheses/README.md` are touched. `sim/doc.go` comment left as known debt (see deviation log).

BC-11: Versioned Documentation
- GIVEN the mike versioning provider configured in mkdocs.yml
- WHEN a push to main occurs, `latest` docs are deployed; when a version tag (v*) is pushed, the corresponding major.minor docs are deployed
- THEN the site header shows a version selector allowing users to switch between `latest` and released versions (e.g., `0.6`, `0.7`)
- MECHANISM: `extra.version.provider: mike` in mkdocs.yml + GitHub Actions workflow with two triggers (main push + tag push)

### C) Component Interaction

```
mkdocs.yml (nav definition)
    │
    ├── docs/index.md (home — slimmed to hero + nav guide)
    ├── docs/getting-started/
    │   ├── index.md (What is BLIS?)
    │   ├── installation.md
    │   ├── quickstart.md
    │   └── tutorial.md (capacity planning tutorial)
    ├── docs/guide/
    │   ├── index.md (User Guide overview)
    │   ├── routing.md
    │   ├── kv-cache.md
    │   ├── roofline.md
    │   ├── workloads.md
    │   ├── cluster.md
    │   ├── results.md
    │   └── experimentation.md
    ├── docs/concepts/
    │   ├── index.md (from design/README.md, adapted)
    │   ├── glossary.md (from design/concepts.md)
    │   ├── architecture.md (from design/architecture.md)
    │   ├── core-engine.md (from design/core-engine.md)
    │   ├── roofline.md (from design/roofline.md)
    │   └── diagrams/ (from design/diagrams/)
    ├── docs/reference/
    │   ├── index.md (reference overview)
    │   ├── configuration.md (from design/configuration.md)
    │   ├── models.md (extracted from index.md)
    │   └── workload-spec.md (new — schema reference)
    └── docs/contributing/
        ├── index.md (from CONTRIBUTING.md content)
        ├── extension-recipes.md (from docs/extension-recipes.md)
        ├── pr-workflow.md (from process/)
        ├── design-process.md (from process/design.md)
        ├── macro-planning.md (from process/macro-plan.md)
        ├── hypothesis.md (from process/hypothesis.md)
        ├── convergence.md (from process/convergence.md)
        ├── standards/
        │   ├── rules.md (from standards/)
        │   ├── invariants.md (from standards/)
        │   ├── principles.md (from standards/)
        │   └── experiments.md (from standards/)
        └── templates/
            ├── design-guidelines.md (from templates/)
            ├── macro-plan.md (from templates/)
            ├── micro-plan.md (from templates/)
            └── hypothesis.md (from templates/)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue #449 proposes "Workload Spec Schema" as separate Reference page | Creates `reference/workload-spec.md` as a new page extracting schema docs from `configuration.md` workload section | ADDITION: Schema reference warrants its own page for direct linking |
| Issue #449 proposes flat Contributing with Standards/ and Templates/ subsections | Keeps process docs at contributing root level, nests standards/ and templates/ | SIMPLIFICATION: Process docs are used frequently enough to be top-level in Contributing |
| Issue #449 does not list "Macro Planning" under Contributing | Moves `process/macro-plan.md` → `contributing/macro-planning.md` | ADDITION: All 5 process docs move together; omitting one would be inconsistent |
| Issue #449 lists "breadcrumbs" as missing feature | Plan does not enable `navigation.path` (breadcrumbs) | DEFERRAL: Breadcrumbs require Material Insiders (paid) — not available in open-source MkDocs Material. Can revisit if the project subscribes |
| Plan says "No Go code changes" (BC-10) | `sim/doc.go:34` has comment referencing `docs/extension-recipes.md` → stale after move | DEFERRAL: Updating a Go comment is zero-risk but technically modifies a Go source file. Left as known debt; the comment is developer guidance, not functional code |
| Plan scope is docs-only | `.claude/skills/` (19 refs), `hypotheses/README.md` (5 refs), `README.md` (4 refs), `.github/PULL_REQUEST_TEMPLATE.md` (2 refs) also have stale paths | ADDITION: Added to Task 7 scope. These are non-MkDocs files but actively used by contributors and agents |
| Issue #449 says "Create" docs.yml workflow | `.github/workflows/docs.yml` already exists with `mkdocs gh-deploy` + sed hack | CORRECTION: Task 8 must REPLACE existing workflow, preserving PR validation job |

### E) Review Guide

**The tricky part:** Cross-reference updates after file moves. Every `[link](../path.md)` in moved files needs its relative path recalculated. The most error-prone files are the design docs that heavily cross-reference each other and the standards.

**What to scrutinize:** BC-5 (cross-reference integrity) and BC-8 (no broken links). Run `mkdocs serve` and check for warnings. Also verify the new user-facing content is technically accurate against the actual CLI behavior.

**What's safe to skim:** The file moves themselves (mechanical git mv operations) and the Material theme configuration (well-documented features).

**Known debt:** CONTRIBUTING.md at repo root and `docs/contributing/index.md` will have overlapping content (contributor quick start + standards overview). This PR creates `docs/contributing/index.md` as the docs site landing page for contributors, and updates CONTRIBUTING.md to link to the docs site rather than duplicating content. The sed hack is removed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create (new content):**
- `docs/getting-started/index.md` — What is BLIS?
- `docs/getting-started/installation.md`
- `docs/getting-started/quickstart.md`
- `docs/getting-started/tutorial.md` — Capacity planning tutorial
- `docs/guide/index.md` — User Guide overview
- `docs/guide/routing.md` — Routing policies guide
- `docs/guide/kv-cache.md` — KV cache & memory management guide
- `docs/guide/roofline.md` — Roofline mode guide
- `docs/guide/workloads.md` — Workload specifications guide
- `docs/guide/cluster.md` — Cluster simulation guide
- `docs/guide/results.md` — Interpreting results guide
- `docs/guide/experimentation.md` — Hypothesis-driven experimentation guide
- `docs/reference/index.md` — Reference overview
- `docs/reference/models.md` — Supported models catalog
- `docs/reference/workload-spec.md` — Workload spec schema reference
- `docs/contributing/index.md` — Contributing landing page

**Files moved in Task 1 then adapted (not new):**
- `docs/concepts/index.md` — Moved from `design/README.md` via `git mv`, adapted in Task 6

**Files to move (git mv):**
- `docs/design/concepts.md` → `docs/concepts/glossary.md`
- `docs/design/architecture.md` → `docs/concepts/architecture.md`
- `docs/design/core-engine.md` → `docs/concepts/core-engine.md`
- `docs/design/roofline.md` → `docs/concepts/roofline.md`
- `docs/design/diagrams/` → `docs/concepts/diagrams/`
- `docs/design/configuration.md` → `docs/reference/configuration.md`
- `docs/extension-recipes.md` → `docs/contributing/extension-recipes.md`
- `docs/process/pr-workflow.md` → `docs/contributing/pr-workflow.md`
- `docs/process/design.md` → `docs/contributing/design-process.md`
- `docs/process/macro-plan.md` → `docs/contributing/macro-planning.md`
- `docs/process/hypothesis.md` → `docs/contributing/hypothesis.md`
- `docs/process/convergence.md` → `docs/contributing/convergence.md`
- `docs/standards/rules.md` → `docs/contributing/standards/rules.md`
- `docs/standards/invariants.md` → `docs/contributing/standards/invariants.md`
- `docs/standards/principles.md` → `docs/contributing/standards/principles.md`
- `docs/standards/experiments.md` → `docs/contributing/standards/experiments.md`
- `docs/templates/design-guidelines.md` → `docs/contributing/templates/design-guidelines.md`
- `docs/templates/macro-plan.md` → `docs/contributing/templates/macro-plan.md`
- `docs/templates/micro-plan.md` → `docs/contributing/templates/micro-plan.md`
- `docs/templates/hypothesis.md` → `docs/contributing/templates/hypothesis.md`

**Files to modify:**
- `mkdocs.yml` — Complete nav rewrite + theme features + exclude_docs
- `docs/index.md` — Slim to hero + navigation guide (extract models to reference/), update all internal links
- All moved files — Update relative cross-references (see Task 3 mapping table)
- `CLAUDE.md` — Update ALL ~28 `docs/` path references (File Organization tree, governance paths, AND inline references throughout)
- `CONTRIBUTING.md` — Update all ~32 `docs/` path references, remove sed hack from Quick Start
- `README.md` (root) — Update 4 `docs/` path references (core-engine, roofline, invariants, design/)
- `.claude/skills/` — Update 19 path references across 5 files (convergence-review SKILL/prompts, hypothesis-experiment SKILL/prompts)
- `hypotheses/README.md` — Update 5 `docs/` path references
- `.github/PULL_REQUEST_TEMPLATE.md` — Update 2 `docs/standards/` path references
- `.github/workflows/docs.yml` — REPLACE existing workflow with mike-based versioned deployment (preserving PR validation job)
- `docs/contributing/standards/principles.md` — Update source-of-truth map (13+ absolute path references)
- `docs/contributing/templates/micro-plan.md` — Update sanity checklist path references

**Key decisions:**
- Use `git mv` for all moves to preserve history
- Section index pages (`index.md`) use `navigation.indexes` feature
- No symlinks — direct file locations for MkDocs compatibility
- `sim/doc.go:34` comment left as known debt (see deviation log)
- `hypotheses/*/FINDINGS.md` files (37 refs across 29 files) left as historical records — these are experiment snapshots that should not be retroactively modified
- `docs/plans/` files (159 refs across 13 files) left as-is — excluded from MkDocs and historical

### G) Task Breakdown

---

### Task 1: Create directory structure and move existing files

**Contracts Implemented:** BC-4, BC-7, BC-9

**Files:**
- Create: `docs/getting-started/`, `docs/guide/`, `docs/concepts/`, `docs/reference/`, `docs/contributing/standards/`, `docs/contributing/templates/`
- Move: All files listed in Section F

**Step 1: Create directories**

```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/docs-restructure-449
mkdir -p docs/getting-started docs/guide docs/concepts docs/reference docs/contributing/standards docs/contributing/templates
```

**Step 2: Move files with git mv**

```bash
# Concepts (from design/)
git mv docs/design/concepts.md docs/concepts/glossary.md
git mv docs/design/architecture.md docs/concepts/architecture.md
git mv docs/design/core-engine.md docs/concepts/core-engine.md
git mv docs/design/roofline.md docs/concepts/roofline.md
git mv docs/design/diagrams docs/concepts/diagrams

# Reference (from design/)
git mv docs/design/configuration.md docs/reference/configuration.md

# Contributing (from process/)
git mv docs/process/pr-workflow.md docs/contributing/pr-workflow.md
git mv docs/process/design.md docs/contributing/design-process.md
git mv docs/process/macro-plan.md docs/contributing/macro-planning.md
git mv docs/process/hypothesis.md docs/contributing/hypothesis.md
git mv docs/process/convergence.md docs/contributing/convergence.md

# Contributing (from extension-recipes)
git mv docs/extension-recipes.md docs/contributing/extension-recipes.md

# Contributing standards (from standards/)
git mv docs/standards/rules.md docs/contributing/standards/rules.md
git mv docs/standards/invariants.md docs/contributing/standards/invariants.md
git mv docs/standards/principles.md docs/contributing/standards/principles.md
git mv docs/standards/experiments.md docs/contributing/standards/experiments.md

# Contributing templates (from templates/)
git mv docs/templates/design-guidelines.md docs/contributing/templates/design-guidelines.md
git mv docs/templates/macro-plan.md docs/contributing/templates/macro-plan.md
git mv docs/templates/micro-plan.md docs/contributing/templates/micro-plan.md
git mv docs/templates/hypothesis.md docs/contributing/templates/hypothesis.md
```

**Step 3: Move design/README.md to concepts/index.md (preserve history)**

```bash
git mv docs/design/README.md docs/concepts/index.md
```

**Step 4: Remove emptied directories**

```bash
rmdir docs/design 2>/dev/null || true
rmdir docs/process 2>/dev/null || true
rmdir docs/standards 2>/dev/null || true
rmdir docs/templates 2>/dev/null || true
```

**Step 5: Commit file moves**

```bash
git add -A docs/
git commit -m "docs: move files to audience-oriented directory structure (BC-4, BC-7, BC-9)

- Move design docs to concepts/ (glossary, architecture, core-engine, roofline)
- Move configuration reference to reference/
- Move process docs to contributing/ (pr-workflow, design, macro-plan, hypothesis, convergence)
- Move standards to contributing/standards/
- Move templates to contributing/templates/
- Move extension recipes to contributing/
- Create new directory structure: getting-started/, guide/, concepts/, reference/, contributing/

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Update mkdocs.yml with new nav and theme features

**Contracts Implemented:** BC-1, BC-6

**Files:**
- Modify: `mkdocs.yml`

**Step 1: Rewrite mkdocs.yml**

Replace the entire `mkdocs.yml` with the new audience-oriented nav and enhanced theme configuration. Key changes:
- Add `navigation.tabs` for top-level tab navigation
- Add `navigation.indexes` for section index pages
- Add `navigation.footer` for next/previous navigation
- Restructure `nav:` to match the new file layout
- Update `exclude_docs` — MkDocs default exclusion `/templates/` only matches the TOP-LEVEL templates directory. Since templates move to `docs/contributing/templates/` (nested), the default exclusion no longer applies. The `!/contributing/templates/` negation is harmless but technically unnecessary. Keep `plans/` exclusion:
  ```yaml
  exclude_docs: |
    plans/
  ```
  Note: The old `!/templates/` negation is no longer needed and can be removed.

**Step 2: Verify mkdocs builds (non-strict — new content files don't exist yet)**

```bash
pip install mkdocs-material==9.7.3 2>/dev/null
mkdocs build 2>&1 | head -50
```

Expected: Build succeeds. Warnings about missing nav targets (getting-started/, guide/, reference/ files) are expected — those are created in Tasks 4-6. No errors for moved files.

**Step 3: Commit**

```bash
git add mkdocs.yml
git commit -m "docs: restructure mkdocs.yml with audience-oriented nav and Material tabs (BC-1, BC-6)

- Enable navigation.tabs, navigation.indexes, navigation.footer
- Reorganize nav into 5 sections: Getting Started, User Guide, Concepts, Reference, Contributing
- Update all paths for moved files

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Update cross-references in moved files

**Contracts Implemented:** BC-5, BC-8

**Files:**
- Modify: All files moved in Task 1 (update relative links)

**Step 1: Identify all cross-references**

For each moved file, find all relative links (`](../path)` or `](path.md)`) and update them to the correct relative path from the new location.

**Comprehensive cross-reference mapping (tricky cases where relative depth changes):**

Files staying in same directory (relative links between them unchanged):
- `concepts/architecture.md` ↔ `concepts/core-engine.md` ↔ `concepts/glossary.md` ↔ `concepts/roofline.md` — all same directory, links stay same
- `contributing/standards/rules.md` ↔ `contributing/standards/invariants.md` ↔ `contributing/standards/principles.md` — same directory
- `contributing/pr-workflow.md` ↔ `contributing/convergence.md` ↔ `contributing/hypothesis.md` — same directory

Files where relative depth changes (most error-prone):

> **IMPORTANT for implementer:** The Count column below is APPROXIMATE (estimated, not grep-verified). Use `grep` + `replace_all` to find ALL occurrences when implementing. Do NOT use the counts as verification checksums — always grep the actual file. The mapping shows which patterns to search for and what to replace them with.

**concepts/ → reference/ (configuration.md moved to different dir):**
| From file | Old link pattern | New link | Count (approx) |
|-----------|----------|----------|-------|
| `concepts/glossary.md` | `configuration.md#...` | `../reference/configuration.md#...` | 5 |
| `concepts/architecture.md` | `configuration.md#...` | `../reference/configuration.md#...` | 1 |
| `concepts/core-engine.md` | `configuration.md#...` | `../reference/configuration.md#...` | 1 |
| `concepts/roofline.md` | `configuration.md#...` | `../reference/configuration.md#...` | 1 |
| `concepts/index.md` | `configuration.md` | `../reference/configuration.md` | 2 |
| `reference/configuration.md` | `architecture.md` | `../concepts/architecture.md` | 1 |
| `reference/configuration.md` | `core-engine.md` | `../concepts/core-engine.md` | 1 |
| `reference/configuration.md` | `roofline.md` | `../concepts/roofline.md` | 1 |

**concepts/ → contributing/ (standards + extension-recipes moved under contributing/):**
| From file | Old link pattern | New link | Count (approx) |
|-----------|----------|----------|-------|
| `concepts/architecture.md` | `../standards/invariants.md` | `../contributing/standards/invariants.md` | 1 |
| `concepts/architecture.md` | `../extension-recipes.md` | `../contributing/extension-recipes.md` | 2 |
| `concepts/core-engine.md` | `../standards/invariants.md` | `../contributing/standards/invariants.md` | 1 |
| `concepts/core-engine.md` | `../extension-recipes.md` | `../contributing/extension-recipes.md` | 1 |
| `concepts/index.md` | `../extension-recipes.md` | `../contributing/extension-recipes.md` | 2 |
| `concepts/index.md` | `../standards/` | `../contributing/standards/` | 1 |
| `concepts/index.md` | `../process/` | `../contributing/` | 1 |

**concepts/index.md filename rename (concepts.md → glossary.md):**
| From file | Old link pattern | New link | Count (approx) |
|-----------|----------|----------|-------|
| `concepts/index.md` | `concepts.md` | `glossary.md` | 2 |

**contributing/ internal depth changes (process/ → contributing/, standards/ → contributing/standards/):**
| From file | Old link pattern | New link | Count (approx) |
|-----------|----------|----------|-------|
| `contributing/hypothesis.md` | `../standards/experiments.md` | `standards/experiments.md` | 9 |
| `contributing/hypothesis.md` | `../templates/hypothesis.md` | `templates/hypothesis.md` | 3 |
| `contributing/macro-planning.md` | `../templates/macro-plan.md` | `templates/macro-plan.md` | 2 |
| `contributing/macro-planning.md` | `../templates/design-guidelines.md` | `templates/design-guidelines.md` | 2 |
| `contributing/macro-planning.md` | `../standards/rules.md` | `standards/rules.md` | 1 |
| `contributing/design-process.md` | `../templates/design-guidelines.md` | `templates/design-guidelines.md` | 4 |
| `contributing/design-process.md` | `../standards/rules.md` | `standards/rules.md` | 1 |
| `contributing/design-process.md` | `../standards/invariants.md` | `standards/invariants.md` | 1 |
| `contributing/standards/rules.md` | `../process/pr-workflow.md` | `../pr-workflow.md` | 1 |
| `contributing/standards/experiments.md` | `../process/convergence.md` | `../convergence.md` | 2 |
| `contributing/standards/experiments.md` | `../process/hypothesis.md` | `../hypothesis.md` | 3 |

**Source-of-truth map in contributing/standards/principles.md (absolute path display text):**
All `docs/...` paths in the source-of-truth map table (lines 91-103) must be updated to new paths. These are NOT relative links — they are display text showing canonical file locations.

**Step 2: Run mkdocs build to verify (non-strict — Tasks 4-6 content files don't exist yet)**

```bash
mkdocs build 2>&1 | grep -i "warning\|error" | grep -v "getting-started\|guide/\|reference/"
```

Expected: No broken-link warnings for moved files. Warnings about missing getting-started/, guide/, reference/ nav targets are expected — those are created in Tasks 4-6.

**Step 3: Commit**

```bash
git add docs/
git commit -m "docs: update cross-references for moved files (BC-5, BC-8)

- Fix all relative links in concepts/, reference/, contributing/ files
- Verified via mkdocs build --strict

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Create Getting Started section content

**Contracts Implemented:** BC-2

**Files:**
- Create: `docs/getting-started/index.md`, `docs/getting-started/installation.md`, `docs/getting-started/quickstart.md`, `docs/getting-started/tutorial.md`

**Content outline:**

**index.md (What is BLIS?):**
- Problem statement: Why simulate inference serving?
- What BLIS does (capacity planning, policy optimization, performance prediction)
- Who should use BLIS (capacity planners, platform engineers, researchers)
- When NOT to use BLIS (not a benchmark, not a load generator)
- Quick feature overview with links to deeper docs

**installation.md:**
- Prerequisites (Go 1.21+)
- Building from source
- Verifying the build
- Optional: MkDocs for local docs

**quickstart.md:**
- Run your first simulation (single instance, default model)
- Read the output (TTFT, E2E, throughput)
- Try cluster mode (4 instances)
- Next steps links

**index.md additional content — DES framing (PP-6, PP-8):**
- Clarify BLIS is a time-advancing DES, not cycle-accurate (set expectations for GPU sim users)
- Explain deterministic replay as a superpower (INV-6: same seed → byte-identical output)
- Note homogeneous-instance assumption (all instances share identical SimConfig)

**tutorial.md (Capacity Planning Tutorial):**
- Scenario: "How many instances do I need for 1000 req/s?"
- Step 1: Baseline single-instance measurement (compute service capacity from beta coefficients: ~57.4 req/s for llama-3.1-8b/H100/TP=2)
- Step 2: Scale up with cluster mode — explain `excess = λ/k - μ` queue growth formula
- Step 3: Find the saturation point — distinguish 3 bottleneck types: compute (step time), memory (KV preemptions), queue (arrival > service)
- Step 4: Compare routing policies — reference `examples/routing-comparison.sh`
- Step 5: Interpret fitness scores — warn about normalization compression (38% TTFT improvement → 8% fitness diff)
- Step 6: Validate against SLO targets — "Will my deployment meet TTFT p99 < 200ms?"
- Complete walkthrough with actual CLI commands and expected output patterns
- DES insight: explain why saturation in DES means WaitQ growth (event-driven, not continuous)

**Step 1: Write all 4 files**

Create each file with substantive content (not stubs). Each should be self-contained and useful.

**Step 2: Verify builds**

```bash
mkdocs build --strict 2>&1 | grep -i "warning\|error"
```

**Step 3: Commit**

```bash
git add docs/getting-started/
git commit -m "docs: add Getting Started section with tutorial (BC-2)

- What is BLIS: problem statement, audience, feature overview
- Installation: prerequisites, build, verify
- Quick Start: first simulation, read output, cluster mode
- Tutorial: end-to-end capacity planning walkthrough

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Create User Guide content

**Contracts Implemented:** BC-3

**Files:**
- Create: `docs/guide/index.md`, `docs/guide/routing.md`, `docs/guide/kv-cache.md`, `docs/guide/roofline.md`, `docs/guide/workloads.md`, `docs/guide/cluster.md`, `docs/guide/results.md`, `docs/guide/experimentation.md`

**Content outline for each guide:**

**index.md:** Overview of all guides with one-sentence descriptions.

**routing.md (Routing Policies):**
- Available policies: round-robin, least-loaded, weighted, prefix-affinity, always-busiest (pathological)
- Weighted scoring explained with scorer composition
- Mapping to real systems: BLIS scorer → llm-d Endpoint Picker equivalents (PP-8)
- When to use which policy (decision matrix)
- Example: comparing policies with `examples/routing-comparison.sh`
- Signal freshness: three-tier model — Immediate (QueueDepth, BatchSize), Periodic (KVUtilization with `--snapshot-refresh-interval`), PendingRequests (instant injection) (PP-6, PP-7)
- Snapshot staleness safe zone: <5ms for kv-utilization, H29 findings (PP-9)
- Instance-level scheduling policies (FCFS, Priority-FCFS, SJF, ReversePriority) and their interaction with routing (PP-7)
- Scorer weight tuning methodology — reference H15, H23, H29 findings (PP-8)

**kv-cache.md (KV Cache & Memory Management):**
- Block allocation model — blocks as unit of KV cache, `--block-size-in-tokens` as tuning parameter (PP-7)
- Prefix caching and how it saves tokens (block-aligned hashing)
- Tiered caching (GPU + CPU offload): `--kv-cpu-blocks`, `--kv-offload-threshold`, transfer bandwidth/latency
- DroppedUnservable rejection: what happens when `ceil(inputTokens/blockSize) > TotalCapacity()` — minimum KV block requirements (PP-7)
- Tuning KV blocks for your workload — distribution MEDIAN drives KV pressure, not mean (H20 finding) (PP-9)
- Identifying the KV pressure cliff: sweep `--total-kv-blocks` to find preemption inflection point (PP-9)
- Chunked prefill for HOL blocking reduction — `--long-prefill-token-threshold`, H27 findings (PP-6)
- Batch formation coupling: `--max-num-running-reqs` and `--max-num-scheduled-tokens` as primary capacity knobs (PP-7)
- Livelock protection (R19): minimum blocks required, #386 fix (PP-6)

**roofline.md (Roofline Mode):**
- When to use roofline vs blackbox (roofline = zero alpha overhead; good for "can this model fit?" analysis, less accurate for tail latency under load) (PP-7)
- The --roofline flag (auto-fetch workflow)
- Manual configuration with --model-config-folder
- Adding support for new models
- TP degree impact on roofline accuracy (PP-8)
- Interpreting roofline vs blackbox differences — mean rankings equivalent, P99 diverges at high load (H19) (PP-7)

**workloads.md (Workload Specifications):**
- CLI mode vs workload-spec YAML
- Writing your first workload spec
- Arrival processes (poisson, gamma, weibull, constant) — DES implications: inter-arrival distributions directly determine event queue timing (PP-6)
- Token distributions (gaussian, exponential, pareto-lognormal) — distribution shape effects on memory pressure (PP-9)
- Multi-client workloads with SLO classes (critical, standard, sheddable, batch, background)
- Built-in presets and ServeGen traces
- inference-perf format compatibility (`InferencePerfSpec`) for reusing production workload definitions (PP-8)
- Estimating instance capacity for your workload: compute service rate from beta coefficients, match to `--rate` (PP-9)

**cluster.md (Cluster Simulation):**
- Single-instance vs cluster mode
- The admission → routing → scheduling pipeline
- Tensor parallelism: how `--tp` interacts with `--num-instances` (TP within node vs replication across nodes) (PP-8)
- Homogeneous-instance assumption: all instances share identical SimConfig — limitation for mixed-fleet modeling (PP-8)
- Scaling instances and understanding saturation — super-linear scaling at near-saturation (H7: 7.4x, not 2x) (PP-9)
- Admission control (token bucket rate limiting)
- Admission/routing latency: `--admission-latency` and `--routing-latency` model real network/processing overhead (PP-8)
- Decision tracing and counterfactual analysis — trace output format, `--summarize-trace`, regret interpretation (PP-9)
- Event ordering: `(timestamp, priority, seqID)` determines deterministic but non-FIFO ordering (PP-6)
- Work-conserving property (INV-8): BLIS never idles while requests wait (PP-6)

**results.md (Interpreting Results):**
- Understanding the JSON output — complete field-by-field guide
- Key metrics: TTFT, ITL, E2E, throughput, **scheduling_delay** (PP-9)
- Percentiles and what they tell you (P50 vs P99 divergence near saturation)
- Per-request results with --results-path — **gotcha: `scheduling_delay_ms` is in ticks (μs), not ms** (PP-7, PP-9)
- Anomaly counters: priority inversions, HOL blocking events, rejected requests, dropped unservable — what each means and when to worry (PP-9)
- KV cache metrics: preemption rate, cache hit rate, KV thrashing rate — thresholds and interpretation (PP-9)
- Per-SLO-class metrics: differential latency across SLO tiers (PP-9)
- Fitness evaluation: normalization behavior (`1/(1+x/1000)` compresses large differences), reference scales, valid metric keys, recommendation to examine raw metrics alongside (PP-9)
- Common patterns: saturation curves (λ/k > μ), tail latency spikes (P99 diverges at saturation), chunked prefill TTFT reduction but ITL insensitivity (H27/H28), snapshot staleness safe zone (H29), policy equivalence at low load (H23) (PP-9)
- Alpha overhead effect: non-blocking queueing delay inflates TTFT/E2E but doesn't advance sim clock — explains M/M/k divergence at high load (PP-6)

**experimentation.md (Hypothesis-Driven Experimentation):**
- Why hypothesis-driven experimentation? — DES deterministic replay makes controlled experiments impossible with real hardware (PP-6)
- Capacity planning validation framing: define deployment → define workload → define SLO → run → interpret (PP-8)
- The BLIS experiment workflow (design → implement → run → analyze → document)
- Using the `/hypothesis-test` skill for guided experiments
- Example: testing whether chunked prefill reduces TTFT (H27)
- Case studies from completed experiments: H7 (scaling), H29 (staleness), H27 (chunked prefill) (PP-8)
- The experiment harness (`hypotheses/lib/`)
- Findings documentation and the convergence review process
- Linking experiments to design decisions

**Step 1: Write all 8 files**

Each guide should include practical CLI examples, expected output patterns, and links to deeper reference/concept pages.

**Step 2: Verify builds**

```bash
mkdocs build --strict 2>&1 | grep -i "warning\|error"
```

**Step 3: Commit**

```bash
git add docs/guide/
git commit -m "docs: add User Guide with 8 task-oriented pages (BC-3)

- Routing policies: scorer composition, signal freshness, decision matrix
- KV cache: block allocation, prefix caching, tiered offload, chunked prefill
- Roofline mode: auto-fetch, manual config, new model support
- Workloads: YAML specs, arrival processes, distributions, presets
- Cluster simulation: pipeline, scaling, admission, tracing
- Interpreting results: metrics, percentiles, fitness evaluation
- Hypothesis experimentation: /hypothesis-test skill, experiment harness, findings

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Create section index pages and Reference content

**Contracts Implemented:** BC-1, BC-3

**Files:**
- Create: `docs/concepts/index.md`, `docs/reference/index.md`, `docs/reference/models.md`, `docs/reference/workload-spec.md`, `docs/contributing/index.md`

**Content outline:**

**concepts/index.md:** Already moved from `design/README.md` in Task 1 via `git mv`. Adapt content: update internal links, remove "For Contributors" section (now in contributing/), update reading order to reference new paths.

**reference/index.md:** Overview of reference materials — configuration, models, workload spec.

**reference/models.md:** Supported models catalog extracted from `index.md` (Dense, MoE, Quantized, Roofline-only sections). This is the single source of truth for model support.

**reference/workload-spec.md:** Complete workload-spec YAML schema reference — all fields, types, constraints, examples. Consolidates information from `configuration.md` workload section and `sim/workload/spec.go`.

**contributing/index.md:** Landing page for contributors, adapted from `CONTRIBUTING.md` content (Quick Start, Your First Contribution, link to PR workflow and standards).

**Step 1: Write all 5 files**

**Step 2: Update docs/index.md**

Slim the home page: keep hero section and feature list, remove the Supported Models tables (now in `reference/models.md`), update Documentation Guide table to point to new sections.

**Step 3: Verify builds**

```bash
mkdocs build --strict 2>&1 | grep -i "warning\|error"
```

**Step 4: Commit**

```bash
git add docs/concepts/index.md docs/reference/ docs/contributing/index.md docs/index.md
git commit -m "docs: add section indexes, models catalog, and workload spec reference (BC-1, BC-3)

- Concepts index: adapted from design/README.md with reading order
- Reference index + models catalog (extracted from index.md)
- Workload spec schema reference (new page)
- Contributing index: adapted from CONTRIBUTING.md
- Home page slimmed: models moved to reference/models.md

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Update all non-docs path references and final cross-reference pass

**Contracts Implemented:** BC-5, BC-8, BC-10

**Files:**
- Modify: `CLAUDE.md` — Update ALL ~28 `docs/` path references (File Organization tree rewrite, governance paths, AND inline references in Design Principles, BDD/TDD, PR Workflow, Key Invariants, Engineering Principles, Antipattern Prevention, Extension Recipes sections)
- Modify: `CONTRIBUTING.md` — Update all ~32 `docs/` path references (including tables at lines 354-379, inline skill references at lines 161/176/212/261/291) AND remove/update the sed hack in Quick Start section
- Modify: `docs/index.md` — Update Documentation Guide table links and Reading Order section links
- Modify: `README.md` (root) — Update 4 `docs/` path references (lines 419, 425, 509, 675)
- Modify: `.claude/skills/convergence-review/SKILL.md` — Update 1 path reference
- Modify: `.claude/skills/convergence-review/design-prompts.md` — Update 5 path references
- Modify: `.claude/skills/convergence-review/pr-prompts.md` — Update 3 path references
- Modify: `.claude/skills/hypothesis-experiment/SKILL.md` — Update 6 path references
- Modify: `.claude/skills/hypothesis-experiment/review-prompts.md` — Update 4 path references
- Modify: `hypotheses/README.md` — Update 5 `docs/` path references (lines 3, 5, 82)
- Modify: `.github/PULL_REQUEST_TEMPLATE.md` — Update 2 `docs/standards/` path references
- Modify: `docs/contributing/standards/principles.md` — Update source-of-truth map table (lines 91-103, ~13 absolute path references)
- Modify: `docs/contributing/templates/micro-plan.md` — Update sanity checklist path references (lines 534, 544)
- Modify: Any remaining broken cross-references found by grep verification

**Step 1: Update CLAUDE.md**

Update ALL `docs/` path references in CLAUDE.md. This includes:
- File Organization tree (entire docs/ subtree)
- Project Governance Documents section (13 doc path references)
- Design Principles: `docs/templates/design-guidelines.md` → `docs/contributing/templates/design-guidelines.md`
- Design Principles: `docs/templates/macro-plan.md` → `docs/contributing/templates/macro-plan.md`
- Design Principles: `docs/templates/micro-plan.md` → `docs/contributing/templates/micro-plan.md`
- BDD/TDD: `docs/standards/principles.md` → `docs/contributing/standards/principles.md`
- PR Workflow: `docs/process/pr-workflow.md` → `docs/contributing/pr-workflow.md`
- Key Invariants: `docs/standards/invariants.md` → `docs/contributing/standards/invariants.md`
- Engineering Principles: `docs/standards/principles.md` → `docs/contributing/standards/principles.md`
- Antipattern Prevention: `docs/standards/rules.md` → `docs/contributing/standards/rules.md`
- Extension Recipes: `docs/extension-recipes.md` → `docs/contributing/extension-recipes.md`

**Step 2: Update CONTRIBUTING.md (~32 path references)**

Use `grep -n` to find ALL `docs/` path references. The plan enumerates known ones but CONTRIBUTING.md has 32 lines with old paths — do NOT rely only on this list. Key patterns:
- `docs/process/pr-workflow.md` → `docs/contributing/pr-workflow.md` (~5 occurrences)
- `docs/templates/design-guidelines.md` → `docs/contributing/templates/design-guidelines.md` (~4 occurrences)
- `docs/templates/macro-plan.md` → `docs/contributing/templates/macro-plan.md` (~2 occurrences)
- `docs/templates/micro-plan.md` → `docs/contributing/templates/micro-plan.md` (~3 occurrences, including inline skill invocation examples)
- `docs/standards/principles.md` → `docs/contributing/standards/principles.md`
- `docs/standards/invariants.md` → `docs/contributing/standards/invariants.md`
- `docs/standards/rules.md` → `docs/contributing/standards/rules.md`
- `docs/standards/experiments.md` → `docs/contributing/standards/experiments.md`
- `docs/process/hypothesis.md` → `docs/contributing/hypothesis.md`
- `docs/process/design.md` → `docs/contributing/design-process.md`
- `docs/process/macro-plan.md` → `docs/contributing/macro-planning.md`
- `docs/process/convergence.md` → `docs/contributing/convergence.md`
- `docs/extension-recipes.md` → `docs/contributing/extension-recipes.md`
- `docs/design/` → `docs/concepts/` (line 376)
- Tables at lines 354-379 (Quick Reference), lines 372-379 (Document Reference)

Also update the Quick Start section: remove the `sed` hack (`sed 's|](docs/|](|g' CONTRIBUTING.md > docs/contributing.md`) and replace with a note that the Contributing section is served directly via `docs/contributing/index.md`.

**Step 2a: Update README.md (root), .claude/skills/, hypotheses/README.md, .github/ templates**

Use `grep -rn` to find and update ALL old `docs/` path references in these files:

```bash
# Find all old paths in non-docs files
grep -rn "docs/design/\|docs/process/\|docs/standards/\|docs/templates/\|docs/extension-recipes" \
  README.md .claude/skills/ hypotheses/README.md .github/PULL_REQUEST_TEMPLATE.md
```

Update each reference to the new path. For `.claude/skills/` files, these are path references that guide Claude Code agents — broken paths degrade AI-assisted workflow quality.

**Step 3: Update docs/index.md links**

Update all relative links in docs/index.md that point to old paths:
- `design/README.md` → `concepts/`
- `design/concepts.md` → `concepts/glossary.md`
- `design/core-engine.md` → `concepts/core-engine.md`
- `design/architecture.md` → `concepts/architecture.md`
- `design/configuration.md` → `reference/configuration.md`
- `standards/rules.md` → `contributing/standards/rules.md`
- `process/pr-workflow.md` → `contributing/pr-workflow.md`
- `templates/design-guidelines.md` → `contributing/templates/design-guidelines.md`
- `extension-recipes.md` → `contributing/extension-recipes.md`

**Step 3: Full cross-reference verification (comprehensive)**

```bash
OLD_PATHS="docs/design/\|docs/process/\|docs/standards/\|docs/templates/\|docs/extension-recipes"

# Search docs content (exclude plans/ which are historical)
grep -rn "$OLD_PATHS" docs/ --include="*.md" | grep -v "docs/plans/"

# Search root-level files
grep -n "$OLD_PATHS" CLAUDE.md CONTRIBUTING.md README.md

# Search agent skill files
grep -rn "$OLD_PATHS" .claude/skills/ --include="*.md"

# Search hypothesis README (but NOT individual FINDINGS.md which are historical)
grep -n "$OLD_PATHS" hypotheses/README.md

# Search GitHub templates
grep -rn "$OLD_PATHS" .github/ --include="*.md"

# Check for old extension-recipes links
grep -rn "extension-recipes.md" docs/ --include="*.md" | grep -v contributing | grep -v "docs/plans/"
```

ALL output from these greps must be zero lines (excluding docs/plans/ and hypotheses/*/FINDINGS.md which are intentionally historical).

Fix any stale references found.

**Step 5: Final mkdocs build**

```bash
mkdocs build --strict 2>&1
```

Expected: Clean build with no warnings.

**Step 6: Verify no Go code changes**

```bash
go test ./... 2>&1 | tail -15
```

Expected: All tests pass, identical to baseline.

**Step 7: Commit**

```bash
git add CLAUDE.md CONTRIBUTING.md README.md docs/ .claude/skills/ hypotheses/README.md .github/PULL_REQUEST_TEMPLATE.md
git commit -m "docs: update all path references across repo for docs restructure (BC-5, BC-8, BC-10)

- Update ~28 docs/ path references in CLAUDE.md (tree + inline)
- Update ~32 docs/ path references in CONTRIBUTING.md
- Update 4 docs/ path references in README.md
- Update 19 path references across 5 .claude/skills/ files
- Update 5 path references in hypotheses/README.md
- Update 2 path references in .github/PULL_REQUEST_TEMPLATE.md
- Update source-of-truth map in contributing/standards/principles.md
- Remove sed hack from CONTRIBUTING.md Quick Start
- Verified zero stale docs/ path references via comprehensive grep
- Verified clean mkdocs build --strict
- Verified go test ./... unchanged

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Replace docs workflow with mike-based versioned deployment

**Contracts Implemented:** BC-11

**Files:**
- Modify: `mkdocs.yml` — Add `extra.version` configuration
- Modify: `.github/workflows/docs.yml` — REPLACE existing `mkdocs gh-deploy` workflow with mike-based versioned deployment. **The file already exists** (65 lines) with `mkdocs gh-deploy --force` + sed hack. Must preserve the PR validation job.

**Step 1: Read existing `.github/workflows/docs.yml`**

Read the existing file first to understand what to preserve:
- **Preserve:** PR validation job (`pull_request` trigger + `mkdocs build --strict`)
- **Remove:** sed hack (`sed 's|](docs/|](|g' CONTRIBUTING.md > docs/contributing.md`) from both build and deploy jobs
- **Replace:** `mkdocs gh-deploy --force` with `mike deploy --push --update-aliases`
- **Add:** Tag push trigger for versioned releases

**Step 2: Add mike version config to mkdocs.yml**

Add to `mkdocs.yml`:
```yaml
extra:
  version:
    provider: mike
    default: latest
```

**Step 3: Rewrite `.github/workflows/docs.yml`**

Replace the workflow with three jobs:

1. **build** (on `pull_request` to main) — PR validation:
   ```bash
   mkdocs build --strict
   ```
   This preserves the existing PR validation gate. No sed hack needed.

2. **deploy** (on `push` to main) — Deploy as `latest`:
   ```bash
   mike deploy --push --update-aliases dev latest
   mike set-default --push latest  # idempotent, safe to run every time
   ```

3. **release** (on `push` tags `v*`) — Deploy versioned docs:
   ```bash
   VERSION=$(echo "${GITHUB_REF#refs/tags/v}" | cut -d. -f1-2)
   # Security: validate semver format
   if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+$ ]]; then
     echo "Invalid version: $VERSION"; exit 1
   fi
   mike deploy --push --update-aliases "$VERSION" stable
   ```

Security requirements (PP-10):
- Quote `"$VERSION"` in all mike commands to prevent word splitting
- Add regex validation for semver format before `mike deploy`
- Add concurrency group to prevent parallel gh-pages pushes:
  ```yaml
  concurrency:
    group: docs-deploy
    cancel-in-progress: false
  ```
- Pin `mike` version: `pip install mike==2.1.3`
- Use `actions/checkout@v4` with `fetch-depth: 0` (mike needs gh-pages history)
- Configure git credentials for the bot
- Verify `exclude_docs` works after build:
  ```bash
  test -z "$(find site/ -path '*/plans/*' 2>/dev/null)" || { echo "FAIL: plans/ leaked"; exit 1; }
  ```

**Step 4: Commit**

```bash
git add mkdocs.yml .github/workflows/docs.yml
git commit -m "docs: replace docs workflow with mike-based versioned deployment (BC-11)

- Configure mike version provider in mkdocs.yml
- Replace mkdocs gh-deploy with mike deploy for versioned docs
- Preserve PR validation job (mkdocs build --strict on pull_request)
- Remove sed hack from workflow (docs/contributing.md no longer generated)
- Add tag-push trigger for major.minor versioned releases
- Add version tag validation (semver regex)
- Add concurrency group to prevent parallel gh-pages pushes
- Version selector appears in site header

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|-------------|
| BC-1     | Task 2 | Build | `mkdocs build --strict` succeeds |
| BC-2     | Task 4 | Build | Getting Started pages render correctly |
| BC-3     | Task 5 | Build | User Guide pages render correctly |
| BC-4     | Task 1 | Build | `git status` shows only moves |
| BC-5     | Task 3, 7 | Build | `mkdocs build --strict` — zero broken link warnings |
| BC-6     | Task 2 | Build | Tabs visible in rendered site |
| BC-7     | Task 1 | Build | `git diff --stat` confirms moves not deletions |
| BC-8     | Task 7 | Build | Final `mkdocs build --strict` clean |
| BC-9     | Task 1 | Build | No duplicate files across old/new locations |
| BC-10    | Task 7 | Build | `go test ./...` unchanged; no Go files modified |
| BC-11    | Task 8 | Config | `mike` version provider in mkdocs.yml; workflow file valid YAML |

**No golden dataset updates needed.** This is a docs-only PR.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken cross-references after moves | High | Medium | Task 3 mapping table (rebuilt from grep) + Task 7 comprehensive grep verification + `mkdocs build --strict` |
| Stale paths in non-docs files | High | Medium | Task 7 explicitly scopes: CLAUDE.md, CONTRIBUTING.md, README.md, .claude/skills/, hypotheses/README.md, .github/ templates |
| CONTRIBUTING.md count underestimate | Medium | Medium | Plan specifies ~32 refs (verified by grep). Implementer must use grep, not plan enumeration |
| New content technically inaccurate | Medium | Medium | Content outlines enriched with domain-expert findings (PP-6 through PP-10); reference cmd/root.go, examples/, FINDINGS |
| Existing workflow destroyed by Task 8 | Medium | High | Task 8 explicitly reads existing file first, preserves PR validation job |
| Version tag injection | Low | Medium | Semver regex validation before mike deploy |
| Plans/ directory accidentally published | Low | High | Verify `exclude_docs: plans/` effective; CI step checks `find site/ -path '*/plans/*'` |
| Parallel gh-pages pushes corrupt branch | Low | Medium | Concurrency group on workflow |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (docs only)
- [x] No unexercised flags or interfaces (N/A — docs only)
- [x] No partial implementations
- [x] No breaking changes (file moves preserve content)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint (N/A — no Go code)
- [x] CLAUDE.md updated: File Organization tree, governance paths
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: canonical sources updated, working copies updated
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable output
- [x] Task dependencies correctly ordered (moves → nav → xrefs → content → indexes → final pass)
- [x] All contracts mapped to specific tasks

---

## Appendix: Content Guidelines for New Pages

### Writing Style for User Guide Pages

Each guide page should follow this structure:

1. **Opening paragraph** — What this page covers and when you'd need it
2. **Quick example** — A complete CLI command the reader can try immediately
3. **Concepts section** — Brief explanation of the underlying mechanism (link to Concepts for depth)
4. **How-to sections** — Task-oriented walkthroughs with CLI commands and expected output
5. **Tips and gotchas** — Admonition boxes for common mistakes
6. **Next steps** — Links to related guides and deeper reference

Use MkDocs Material admonitions liberally:
```markdown
!!! tip "When to use roofline mode"
    Use roofline when you don't have trained coefficients for your model/GPU combination.

!!! warning "Snapshot staleness"
    `kv-utilization` scores update on step boundaries (~9ms), not on routing decisions.
```

### Technical Accuracy Sources

- CLI flags: `cmd/root.go` (definitive source)
- Default values: `defaults.yaml`
- Workload spec schema: `sim/workload/spec.go`
- Example configs: `examples/` directory
- Experiment findings: `hypotheses/*/FINDINGS.md`
