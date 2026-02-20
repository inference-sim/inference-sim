# Documentation Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize all project governance documentation into a clear hierarchy (`docs/standards/`, `docs/process/`, `docs/templates/`) and eliminate rule duplication, so every rule, invariant, principle, and experiment standard has exactly one canonical home.

**The problem today:** Five problems:
1. **Rule duplication with drift** — antipattern rules are in CLAUDE.md (11), CONTRIBUTORS.md (15), PR template (8), and micro-plan Phase 8 (~30). These don't match.
2. **No structure** — process docs, templates, per-feature plans, and standards are all mixed in `docs/plans/`. A contributor can't tell which docs govern the project vs which are per-feature artifacts.
3. **Missing process docs** — PRs have a process doc (prworkflow.md) but design, macro-planning, and hypothesis experiments don't.
4. **Missing experiment standards** — hypothesis experiments have proven to be a powerful discovery mechanism (H3 alone produced 3 issues, 1 new rule, 1 new invariant) but there are no rigor standards.
5. **GitHub community gaps** — CONTRIBUTORS.md should be CONTRIBUTING.md per GitHub conventions; README has a placeholder license section.

**What this PR adds:**

*New directories:*
1. **`docs/standards/`** — rules.md, invariants.md, principles.md, experiments.md
2. **`docs/process/`** — pr-workflow.md (moved), design.md (new), macro-plan.md (new), hypothesis.md (new)
3. **`docs/templates/`** — micro-plan.md (moved), macro-plan.md (moved), design-guidelines.md (moved), hypothesis.md (new)

*File moves (with cross-reference updates):*
4. `docs/plans/prworkflow.md` → `docs/process/pr-workflow.md`
5. `docs/plans/prmicroplanprompt-v2.md` → `docs/templates/micro-plan.md`
6. `docs/plans/macroplanprompt.md` → `docs/templates/macro-plan.md`
7. `docs/plans/2026-02-18-design-guidelines.md` → `docs/templates/design-guidelines.md`

*File renames:*
8. `CONTRIBUTORS.md` → `CONTRIBUTING.md`

*Updated files:*
9. **CLAUDE.md** — replace inlined rules/invariants/principles with references; update all `docs/plans/` paths
10. **CONTRIBUTING.md** — replace duplicated checklist with references; update paths
11. **README.md** — fill in License section
12. **.github/PULL_REQUEST_TEMPLATE.md** — replace duplicated checklists with references
13. **docs/plans/prmicroplanprompt-v2.md** (at new location) — update Phase 8 to reference standards
14. **docs/plans/prworkflow.md** (at new location) — update internal cross-refs

**Why this matters:** Contributors (human and AI) can find any governance document by its type: standards = what rules apply, process = how to do the activity, templates = what the output looks like, plans = per-feature artifacts.

**Architecture:** Pure documentation refactoring — no Go code changes.

**Source:** Audit findings from hypothesis H3 session. Also addresses #282 (KV signal staleness), #283 (Router contract freshness), #284 (workload-spec seed independence).

**Closes:** Fixes #282, fixes #283

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR creates three new directories under `docs/` (standards, process, templates), moves governance documents out of the catch-all `docs/plans/`, consolidates duplicated rules into single-source-of-truth files, adds experiment standards, creates stub process docs for design/macro-plan/hypothesis activities, and updates all cross-references. Also renames CONTRIBUTORS.md → CONTRIBUTING.md and fills in the README license section.

No Go code changes. ~7 files created, ~4 files moved, ~6 files updated.

### B) Behavioral Contracts

**BC-1: Rule completeness** — `docs/standards/rules.md` MUST contain every unique antipattern rule from CLAUDE.md (11 numbered rules + principles prose), CONTRIBUTING.md (15-item checklist), PR template (8), and micro-plan Phase 8 (30 items), each appearing exactly once with number, evidence, and enforcement location. The union is 16 unique rules: R1-R11 from CLAUDE.md numbered rules, R12 "golden dataset regenerated" (in CONTRIBUTING.md/PR template but not CLAUDE.md rules), R13-R14 promoted from CLAUDE.md principles to rules (interfaces 2+ impls, no multi-module methods), R15 "stale PR references" (CONTRIBUTING.md only), R16 "signal freshness" (new from H3), plus "config grouped by module" promoted from principles (R17).

**BC-2: Invariant completeness** — `docs/standards/invariants.md` MUST contain every invariant from CLAUDE.md (6) plus INV-7 (signal freshness), each with formal statement and verification strategy.

**BC-3: Principle completeness** — `docs/standards/principles.md` MUST contain every engineering principle from CLAUDE.md, grouped by concern, with cross-references to enforcing rules.

**BC-4: Experiment standards** — `docs/standards/experiments.md` MUST define the deterministic/statistical taxonomy, 4 statistical subtypes, experiment design rules, findings classification, and the audit feedback loop.

**BC-5: File moves preserve content** — moved files (prworkflow, prmicroplanprompt, macroplanprompt, design-guidelines) MUST have identical content at their new locations (only internal cross-references updated).

**BC-6: All cross-references updated** — every `docs/plans/prworkflow.md` reference across the codebase MUST be updated to `docs/process/pr-workflow.md`. Same for all moved files. Verified by `grep`.

**BC-7: Layered content — no full duplication** — after this PR, `docs/standards/` is the canonical source for full rule definitions (evidence, enforcement, checks). CLAUDE.md retains a compact reference table (rule number + name + one-sentence summary) so AI agents have enough context without following file references. The PR template retains inline checklist items with parenthetical rule numbers. Full rule text (evidence, what-to-check, enforcement) appears ONLY in `docs/standards/rules.md`.

**BC-8: CONTRIBUTING.md rename** — `CONTRIBUTORS.md` MUST be renamed to `CONTRIBUTING.md`. All references updated.

**BC-9: README license** — README.md license section MUST reference Apache-2.0 with link to LICENSE file.

**NC-1: No information loss** — every rule, invariant, principle, and process step that exists today MUST exist after reorganization.

**NC-2: No broken links** — `grep -r 'docs/plans/prworkflow\|docs/plans/prmicroplan\|docs/plans/macroplan\|docs/plans/2026-02-18-design-guidelines\|CONTRIBUTORS.md'` across *.md files MUST return zero results after this PR, excluding: (a) redirect stubs, (b) this plan itself, (c) historical per-feature plans in `docs/plans/` which are frozen artifacts and not updated.

**NC-3: Frozen historical artifacts** — per-feature plans in `docs/plans/` (e.g., `pr12-architectural-predesign.md`, `2026-02-13-simplification-assessment-team-prompt.md`) are frozen artifacts. Their old path references are NOT updated — they record the state at the time they were written.

### C) Component Interaction

```
docs/
├── standards/           ← NEW (canonical rules, created in Tasks 1-4)
│   ├── rules.md
│   ├── invariants.md
│   ├── principles.md
│   └── experiments.md
├── process/             ← NEW (activity workflows, Task 5)
│   ├── pr-workflow.md       ← MOVED from docs/plans/prworkflow.md
│   ├── design.md            ← NEW stub
│   ├── macro-plan.md        ← NEW stub
│   └── hypothesis.md        ← NEW stub
├── templates/           ← NEW (artifact templates, Task 6)
│   ├── micro-plan.md        ← MOVED from docs/plans/prmicroplanprompt-v2.md
│   ├── macro-plan.md        ← MOVED from docs/plans/macroplanprompt.md
│   ├── design-guidelines.md ← MOVED from docs/plans/2026-02-18-design-guidelines.md
│   └── hypothesis.md        ← NEW
├── plans/               ← UNCHANGED (per-feature artifacts only)
├── pr-history.md
└── extension-recipes.md

CONTRIBUTING.md          ← RENAMED from CONTRIBUTORS.md
CLAUDE.md                ← UPDATED references
README.md                ← UPDATED license section
.github/PULL_REQUEST_TEMPLATE.md ← UPDATED references
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #281 requests CLI validation fix | Not in scope | DEFERRAL: #281 is a code fix, this PR is docs-only |
| #284 requests seed override | Not in scope | DEFERRAL: #284 is a code fix; documented in experiments.md ED-4 |
| Design guidelines Section 6 anti-patterns | Not merged into rules.md | SIMPLIFICATION: They serve a different purpose (historical evidence vs prescriptive rules). Cross-reference added. |

### E) Review Guide

1. **THE TRICKY PART:** Cross-reference updates — there are ~39 references to moved files across ~7 files. Missing one creates a broken link.
2. **WHAT TO SCRUTINIZE:** BC-6 (all cross-refs updated) and NC-2 (no broken links). Run the grep verification.
3. **WHAT'S SAFE TO SKIM:** Standards content (Tasks 1-4) — this is content extraction, not new logic.
4. **KNOWN DEBT:** `docs/plans/prmicroplanprompt-v1-deprecated.md` references old paths but is deprecated — not updating.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Create (7 files):**
- `docs/standards/rules.md` — 16 consolidated antipattern rules
- `docs/standards/invariants.md` — 7 system invariants
- `docs/standards/principles.md` — engineering principles by concern
- `docs/standards/experiments.md` — experiment taxonomy and rigor standards
- `docs/process/design.md` — design activity process (stub)
- `docs/process/macro-plan.md` — macro-plan activity process (stub)
- `docs/process/hypothesis.md` — hypothesis experiment process (stub)
- `docs/templates/hypothesis.md` — hypothesis experiment template

**Move (4 files, with cross-ref updates):**
- `docs/plans/prworkflow.md` → `docs/process/pr-workflow.md`
- `docs/plans/prmicroplanprompt-v2.md` → `docs/templates/micro-plan.md`
- `docs/plans/macroplanprompt.md` → `docs/templates/macro-plan.md`
- `docs/plans/2026-02-18-design-guidelines.md` → `docs/templates/design-guidelines.md`

**Rename (1 file):**
- `CONTRIBUTORS.md` → `CONTRIBUTING.md`

**Update (5 files):**
- `CLAUDE.md` — replace inlined content with references, update all paths
- `CONTRIBUTING.md` — replace duplicated checklist, update paths
- `README.md` — fill license section
- `.github/PULL_REQUEST_TEMPLATE.md` — reference standards by rule number
- Moved files — update internal cross-references

### G) Task Breakdown

---

### Task 1: Create `docs/standards/rules.md`

**Contracts:** BC-1, NC-1

**Files:** Create `docs/standards/rules.md`

**Step 1:** Create the file with all 16 rules (R1-R16). Each rule has: number, name, one-sentence summary, evidence, what to check, enforcement locations. Plus a quick-reference checklist at the bottom. Content as specified in the original plan (Rules 1-11 from CLAUDE.md, 12-15 from CONTRIBUTING.md, 16 from H3).

**Step 2:** Commit.
```bash
git add docs/standards/rules.md
git commit -m "docs(standards): create canonical antipattern rules R1-R16

Merges rules from CLAUDE.md (11), CONTRIBUTORS.md (4 extra), and H3
experiment findings (signal freshness). Single source of truth.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create `docs/standards/invariants.md`

**Contracts:** BC-2, NC-1

**Files:** Create `docs/standards/invariants.md`

**Step 1:** Create the file with all 7 invariants (INV-1 through INV-7). Each has: name, formal statement, verification strategy, evidence. Content as specified in the original plan.

**Step 2:** Commit.
```bash
git add docs/standards/invariants.md
git commit -m "docs(standards): create canonical system invariants INV-1 through INV-7

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Create `docs/standards/principles.md`

**Contracts:** BC-3, NC-1

**Files:** Create `docs/standards/principles.md`

**Step 1:** Create the file with engineering principles grouped by concern (separation of concerns, interface design, configuration, canonical constructors, output channels, error handling, BDD/TDD). Cross-references to enforcing rules. Content as specified in the original plan.

**Step 2:** Commit.
```bash
git add docs/standards/principles.md
git commit -m "docs(standards): create canonical engineering principles

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Create `docs/standards/experiments.md`

**Contracts:** BC-4, NC-1

**Files:** Create `docs/standards/experiments.md`

**Step 1:** Create the file with experiment taxonomy (deterministic/statistical), 4 statistical subtypes (dominance/monotonicity/equivalence/Pareto), experiment design rules (ED-1 through ED-4), findings classification table, and audit feedback loop. Content as specified in the original plan addition.

**Step 2:** Commit.
```bash
git add docs/standards/experiments.md
git commit -m "docs(standards): create experiment standards with taxonomy and rigor requirements

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Move process docs and create stubs

**Contracts:** BC-5, BC-6

**Files:**
- Move: `docs/plans/prworkflow.md` → `docs/process/pr-workflow.md`
- Create: `docs/process/design.md` (stub)
- Create: `docs/process/macro-plan.md` (stub)
- Create: `docs/process/hypothesis.md` (stub)

**Step 1:** Move prworkflow.md to new location. Update all internal cross-references within the file (references to prmicroplanprompt-v2.md → `../templates/micro-plan.md`, macroplanprompt.md → `../templates/macro-plan.md`, design-guidelines → `../templates/design-guidelines.md`).

**Step 1b:** Update Step 4.75 self-audit dimensions to reference rule numbers where they overlap with standards:
- Dimension 3 (Determinism) → reference R2 (sort map keys) and INV-6
- Dimension 7 (Test epistemology) → reference R7 (invariant tests alongside golden) and R12 (golden dataset regenerated)
- Dimension 8 (Construction sites) → reference R4 (construction site audit)
- Dimension 9 (Error paths) → reference R1 (no silent data loss) and R5 (transactional state mutation)
Keep the dimension descriptions human-readable — add rule numbers parenthetically, don't replace the text.

**Step 2:** Create stub process docs for the three activities that don't have process docs yet. Each stub should have: purpose, steps overview, quality gates, and references to the corresponding template. Mark as "Draft — to be expanded from experience."

**Step 3:** Commit.
```bash
git add docs/process/
git commit -m "docs(process): move pr-workflow and create design/macro-plan/hypothesis process stubs

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Move templates and create hypothesis template

**Contracts:** BC-5, BC-6

**Files:**
- Move: `docs/plans/prmicroplanprompt-v2.md` → `docs/templates/micro-plan.md`
- Move: `docs/plans/macroplanprompt.md` → `docs/templates/macro-plan.md`
- Move: `docs/plans/2026-02-18-design-guidelines.md` → `docs/templates/design-guidelines.md`
- Create: `docs/templates/hypothesis.md`

**Step 1:** Move the three existing template files. Update internal cross-references within each (e.g., prworkflow references inside prmicroplanprompt).

**Step 2:** Update Phase 8 of `docs/templates/micro-plan.md` to reference `docs/standards/rules.md` for antipattern rules, keeping only plan-specific checks inline.

**Step 3:** Create hypothesis experiment template with: header format, experiment classification, required sections (hypothesis, experiment design, predicted outcome, results, root cause, findings classification, audit).

**Step 4:** Commit.
```bash
git add docs/templates/
git commit -m "docs(templates): move micro-plan/macro-plan/design-guidelines, create hypothesis template

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Leave symlinks/redirects at old locations

**Contracts:** BC-6

**Files:** Create redirect stubs at old locations for backward compatibility.

**Step 1:** At each old location, leave a short markdown file that says "This document has moved to [new location]" with a link. This prevents broken bookmarks and gives people time to update their muscle memory.

Files to create:
- `docs/plans/prworkflow.md` → redirect to `../process/pr-workflow.md`
- `docs/plans/prmicroplanprompt-v2.md` → redirect to `../templates/micro-plan.md`
- `docs/plans/macroplanprompt.md` → redirect to `../templates/macro-plan.md`

The design guidelines file keeps its dated name as a redirect: `docs/plans/2026-02-18-design-guidelines.md` → redirect to `../templates/design-guidelines.md`.

**Step 2:** Commit.
```bash
git add docs/plans/prworkflow.md docs/plans/prmicroplanprompt-v2.md docs/plans/macroplanprompt.md docs/plans/2026-02-18-design-guidelines.md
git commit -m "docs(plans): leave redirect stubs at old locations for backward compatibility

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Update CLAUDE.md

**Contracts:** BC-6, BC-7

**Files:** Modify `CLAUDE.md`

**Step 1: Layered content approach for AI agent context.**

CLAUDE.md is loaded as system context for AI agents. Agents cannot follow file references at instruction-interpretation time. Therefore CLAUDE.md MUST retain enough inline content for agents to check rules during implementation.

**What stays inline (compact reference table):**
- "Antipattern Prevention" section: replace full rule text with a numbered table (R1-R17, each as: number + name + one-sentence summary). Add footer: "Full evidence, checks, and enforcement details: see `docs/standards/rules.md`"
- "Key Invariants" section: keep the 6 invariant statements inline (they're only 6 lines). Add INV-7 (signal freshness). Add footer pointing to `docs/standards/invariants.md` for verification strategies.
- "Engineering Principles" section: keep section headers + first sentence of each principle group. Move detailed bullet points to `docs/standards/principles.md`.
- "BDD/TDD Development" section: keep the 6 numbered practices. Move prohibited/required assertion pattern tables to `docs/standards/principles.md`.

**What moves out (evidence, enforcement details):**
- Per-rule evidence ("Issue #183 — ..."), enforcement locations, what-to-check guidance → `docs/standards/rules.md`
- Invariant verification strategies, code locations → `docs/standards/invariants.md`
- Detailed principle rationale → `docs/standards/principles.md`

**Step 2:** Update all `docs/plans/` paths to new locations:
- `docs/plans/prworkflow.md` → `docs/process/pr-workflow.md`
- `docs/plans/prmicroplanprompt-v2.md` → `docs/templates/micro-plan.md`
- `docs/plans/macroplanprompt.md` → `docs/templates/macro-plan.md`
- `docs/plans/2026-02-18-design-guidelines.md` → `docs/templates/design-guidelines.md`

**Step 3:** Update "Design Documents" section to reflect new directory structure. Add `docs/standards/` and `docs/process/` and `docs/templates/` to the file organization tree.

**Step 4:** Commit.
```bash
git add CLAUDE.md
git commit -m "docs(claude): replace inlined rules with references, update paths to new structure

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Rename CONTRIBUTORS.md → CONTRIBUTING.md and update

**Contracts:** BC-7, BC-8

**Files:**
- Rename: `CONTRIBUTORS.md` → `CONTRIBUTING.md`
- Update: references across codebase

**Step 1:** `git mv CONTRIBUTORS.md CONTRIBUTING.md`. Then create a redirect stub at the old location: `CONTRIBUTORS.md` → "This file has been renamed to [CONTRIBUTING.md](./CONTRIBUTING.md) per GitHub community standards."

**Step 2:** Replace duplicated "Engineering Principles" and "Antipattern Checklist" sections with references to `docs/standards/`. Keep getting-started content and extension recipes.

**Step 3:** Update all `docs/plans/` paths to new locations (same set as Task 8).

**Step 4:** Update `.github/PULL_REQUEST_TEMPLATE.md`:
- Reference `CONTRIBUTING.md` (not `CONTRIBUTORS.md`)
- Keep antipattern checklist items inline (contributors need to check items without opening another file) but add parenthetical rule numbers: e.g., `- [ ] No silent data loss (R1)`. Add link to `docs/standards/rules.md` for full details.
- Keep invariant checklist items inline with invariant IDs: e.g., `- [ ] Request conservation: injected == completed + queued + running (INV-1)`

**Step 5:** Commit.
```bash
git add CONTRIBUTING.md .github/PULL_REQUEST_TEMPLATE.md
git commit -m "docs: rename CONTRIBUTORS.md → CONTRIBUTING.md, reference standards

GitHub community standards require CONTRIBUTING.md. Replaces duplicated
rules with references to docs/standards/.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: Update README.md license section

**Contracts:** BC-9

**Files:** Modify `README.md`

**Step 1:** Replace the placeholder license section with:

```markdown
## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](./LICENSE) for details.
```

**Step 2:** Commit.
```bash
git add README.md
git commit -m "docs(readme): fill in Apache-2.0 license section

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Verify no broken cross-references

**Contracts:** NC-2

**Step 1:** Run verification greps:
```bash
# Old paths that should no longer exist in any .md file:
grep -r 'docs/plans/prworkflow\.md' --include='*.md' | grep -v 'redirect\|moved to\|consolidate-standards-plan'
grep -r 'docs/plans/prmicroplanprompt' --include='*.md' | grep -v 'redirect\|moved to\|deprecated\|consolidate-standards-plan'
grep -r 'docs/plans/macroplanprompt\.md' --include='*.md' | grep -v 'redirect\|moved to\|consolidate-standards-plan'
grep -r 'CONTRIBUTORS\.md' --include='*.md' | grep -v 'consolidate-standards-plan'
```

Each should return zero results (excluding redirect stubs and this plan itself).

**Step 2:** If any broken references found, fix them and amend the relevant commit.

**Step 3:** Final commit if any fixes needed.

---

### H) Test Strategy

| Contract | Task | Verification |
|----------|------|--------------|
| BC-1 | Task 1 | Count unique rules = 16. Union of all sources = 16. |
| BC-2 | Task 2 | Count invariants = 7. All 6 from CLAUDE.md + INV-7. |
| BC-4 | Task 4 | experiments.md has 2 types, 4 subtypes, ED-1–ED-4, findings table, audit step. |
| BC-5 | Tasks 5-6 | `diff` moved files — only cross-references changed, content identical. |
| BC-6 | Task 11 | Grep verification returns zero broken references. |
| BC-7 | Tasks 8-9 | `grep -c 'docs/standards/' CLAUDE.md` > 0. No rule text duplicated. |
| BC-8 | Task 9 | `ls CONTRIBUTING.md` exists. `ls CONTRIBUTORS.md` does not exist. |
| BC-9 | Task 10 | README.md contains "Apache License, Version 2.0". |
| NC-2 | Task 11 | Grep verification. |

No Go code changes — no `go test` needed. Verification is document review + grep.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Broken cross-references | High | Medium | Task 11 verification grep; redirect stubs at old locations |
| CLAUDE.md too lean for AI context | Low | Medium | Keep brief summaries, not just bare links |
| Per-feature plans reference old template paths | Medium | Low | Redirect stubs in Task 7 handle this gracefully |
| Git history disrupted by moves | Low | Low | Use `git mv` for rename tracking |

### J) Sanity Checklist

- [x] No Go code changes
- [x] No unnecessary abstractions
- [x] Every source rule mapped to destination
- [x] CLAUDE.md still readable as standalone (brief summaries, not just links)
- [x] Redirect stubs at old locations for backward compatibility
- [x] CONTRIBUTING.md follows GitHub conventions
- [x] README license section filled
- [x] Experiment standards include workload-spec seed caveat (ED-4, references #284)
- [ ] All cross-references verified (Task 11)
