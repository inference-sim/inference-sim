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
| BC-1 | Task 1 | Count unique rules = 17. Union of all sources = 17. |
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

---

## Appendix K: File-Level Implementation Details

### K.1: `docs/standards/rules.md`

```markdown
# BLIS Antipattern Rules

Every rule traces to a real bug, design failure, or hypothesis finding. Rules are enforced at three checkpoints:
- **PR template** — checklist before merge
- **Micro-plan Phase 8** — checklist before implementation
- **Self-audit Step 4.75** — deliberate critical thinking before commit

For the full process, see [docs/process/pr-workflow.md](../process/pr-workflow.md).

## Rules

### R1: No silent data loss

Every error path must either return an error, panic with context, or increment a counter. A `continue` or early `return` that silently drops a request, metric, or allocation is a correctness bug.

**Evidence:** Issue #183 — a KV allocation failure silently dropped a request. The golden test perpetuated the bug for months because it captured "499 completions" as the expected value.

**Check:** For every `continue` or early `return` in new code, verify the error is propagated, counted, or documented as safe.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 9.

---

### R2: Sort map keys before float accumulation

Go map iteration is non-deterministic. Any `for k, v := range someMap` that feeds a running sum (`total += v`) or determines output ordering must sort keys first. Unsorted iteration violates the determinism invariant (INV-6).

**Evidence:** Five sites iterated Go maps to accumulate floats or determine output ordering, violating determinism.

**Check:** For every `range` over a map, check if the loop body accumulates floats or produces ordered output. If so, sort keys first.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 3.

---

### R3: Validate ALL numeric CLI flags

Every numeric flag (`--rate`, `--fitness-weights`, `--kv-cpu-blocks`, etc.) must be validated for: zero, negative, NaN, Inf, and empty string. Missing validation causes infinite loops (Rate=0) or wrong results (NaN weights).

**Evidence:** `--rate 0` caused an infinite loop deep in the simulation. `--snapshot-refresh-interval` was added without validation (#281).

**Check:** For every new CLI flag, add validation in `cmd/root.go` with `logrus.Fatalf` for invalid values.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 6.

---

### R4: Construction site audit

Before adding a field to a struct, find every place that struct is constructed as a literal. If there are multiple sites, either add a canonical constructor or update every site. Missing a site causes silent field-zero bugs.

**Evidence:** Issue #181 — adding `InstanceID` to per-request metrics required changes in 4 files. Three construction sites for `RequestMetrics` existed, and one was missed initially.

**Check:** `grep 'StructName{' across the codebase`. List every site. Update all or refactor to canonical constructor.

**Enforced:** PR template, micro-plan Phase 0 + Phase 8, self-audit dimension 8.

---

### R5: Transactional state mutation

Any loop that allocates resources (blocks, slots, counters) must handle mid-loop failure by rolling back all mutations from previous iterations. A partial allocation that returns `false` without cleanup violates conservation invariants.

**Evidence:** KV block allocation (`AllocateKVBlocks`) had a mid-loop failure path that didn't roll back previously allocated blocks, violating KV conservation (INV-4).

**Check:** For every loop that mutates state, verify the failure path rolls back all mutations.

**Enforced:** PR template, micro-plan Phase 8, self-audit dimension 9.

---

### R6: No logrus.Fatalf in library code

The `sim/` package tree must never terminate the process — return errors so callers can handle them. Only `cmd/` may terminate. This enables embedding, testing, and adapters.

**Evidence:** Library code that called `logrus.Fatalf` prevented test isolation and made the simulator non-embeddable.

**Check:** `grep -r 'logrus.Fatal\|os.Exit' sim/` must return zero results.

**Enforced:** PR template, micro-plan Phase 8.

---

### R7: Invariant tests alongside golden tests

Golden tests (comparing against known-good output) are regression freezes, not correctness checks. If a bug exists when the golden values are captured, the golden test perpetuates the bug. Every subsystem that has golden tests must also have invariant tests that verify conservation laws, causality, and determinism.

**Evidence:** Issue #183 — the codellama golden dataset expected 499 completions because one request was silently dropped. A conservation invariant test would have caught it on day one.

**Check:** For every golden test, ask: "If this expected value were wrong, would any other test catch it?" If no, add an invariant test.

**Enforced:** PR template, micro-plan Phase 6 + Phase 8, self-audit dimension 7.

---

### R8: No exported mutable maps

Validation lookup maps (e.g., `validRoutingPolicies`) must be unexported. Expose through `IsValid*()` accessor functions. Exported maps allow callers to mutate global state, breaking encapsulation and enabling hard-to-trace bugs.

**Evidence:** Exported mutable maps were found during hardening audit — callers could silently add entries to validation maps.

**Check:** `grep -r 'var [A-Z].*map\[' sim/` must return zero mutable map results.

**Enforced:** PR template, micro-plan Phase 8.

---

### R9: Pointer types for YAML zero-value ambiguity

YAML config structs must use `*float64` (pointer) for fields where zero is a valid user-provided value, to distinguish "not set" (nil) from "set to zero" (0.0). Using bare `float64` causes silent misconfiguration when users intentionally set a value to zero.

**Evidence:** YAML fields with bare `float64` couldn't distinguish "user set this to 0" from "user didn't set this."

**Check:** For every new YAML config field where zero is meaningful, use a pointer type.

**Enforced:** Micro-plan Phase 8.

---

### R10: Strict YAML parsing

Use `yaml.KnownFields(true)` or equivalent strict parsing for all YAML config loading. Typos in field names must cause parse errors, not silent acceptance of malformed config.

**Evidence:** YAML typos in field names were silently accepted, producing default behavior instead of the user's intended configuration.

**Check:** Every `yaml.Unmarshal` or decoder usage must enable strict/known-fields mode.

**Enforced:** Micro-plan Phase 8.

---

### R11: Guard division in runtime computation

Any division where the denominator derives from runtime state (batch size, block count, request count, bandwidth) must guard against zero. CLI validation (R3) catches input zeros at the boundary; this rule catches intermediate zeros that arise during simulation.

**Evidence:** `utilization = usedBlocks / totalBlocks` when no blocks are configured; `avgLatency = sum / count` when count is zero.

**Check:** For every division, verify the denominator is either (a) guarded by an explicit zero check, or (b) proven non-zero by a documented invariant.

**Enforced:** Micro-plan Phase 8.

---

### R12: Golden dataset regenerated when output changes

When a PR changes output format, metrics, or default behavior, the golden dataset must be regenerated and the regeneration command documented. Golden tests that pass with stale expected values provide false confidence.

**Evidence:** Present in CONTRIBUTORS.md and PR template but not in CLAUDE.md's numbered rules — an inconsistency this consolidation resolves.

**Check:** If `go test ./sim/... -run Golden` fails after your changes, regenerate and document the command.

**Enforced:** PR template, micro-plan Phase 8.

---

### R13: Interfaces accommodate multiple implementations

New interfaces must accommodate at least two implementations (even if only one exists today). No methods that only make sense for one backend.

**Evidence:** `KVStore` interface has methods exposing block-level semantics. A distributed KV cache like LMCache thinks in tokens and layers, not blocks. The interface encodes vLLM's implementation model rather than an abstract behavioral contract.

**Check:** For every new interface, ask: "Could a second backend implement this without dummy methods?"

**Enforced:** Micro-plan Phase 8.

*Previously: principle in CLAUDE.md "Interface design" section. Promoted to numbered rule for checkability.*

---

### R14: No multi-module methods

No method should span multiple module responsibilities (scheduling + latency estimation + metrics in one function). Extract each concern into its module's interface.

**Evidence:** `Simulator.Step()` is 134 lines mixing scheduling, latency estimation, token generation, completion, and metrics. Impossible to swap the latency model without modifying this method.

**Check:** If a method touches >1 module's concern, extract each concern.

**Enforced:** Micro-plan Phase 8.

*Previously: principle in CLAUDE.md "Interface design" section. Promoted to numbered rule for checkability.*

---

### R15: Resolve stale PR references

After completing a PR, grep for references to that PR number (`planned for PR N`, `TODO.*PR N`) in the codebase. Resolve all stale references.

**Evidence:** Multiple stale comments referencing completed PRs accumulated over time, misleading future developers about what was implemented vs planned.

**Check:** `grep -rn 'planned for PR\|TODO.*PR' --include='*.go' --include='*.md'` for the current PR number.

**Enforced:** Micro-plan Phase 8.

---

### R16: Group configuration by module

Configuration parameters must be grouped by module — not added to a monolithic config struct mixing unrelated concerns. Each module's config should be independently specifiable and validatable.

**Evidence:** `SimConfig` combines hardware identity, model parameters, simulation parameters, and policy choices. Adding one autoscaling parameter requires understanding the entire struct.

**Check:** New config parameters go into the appropriate module's config group, not a catch-all struct.

**Enforced:** Micro-plan Phase 8.

*Previously: principle in CLAUDE.md "Configuration design" section. Promoted to numbered rule for checkability.*

---

### R17: Document signal freshness for routing inputs

Routing snapshot signals have different freshness guarantees due to DES event ordering. Scorer authors must understand which signals are synchronously fresh and which are stale. Any scorer intended for high-rate routing must either use a synchronously-fresh signal or be combined with one that does.

**Evidence:** H3 hypothesis experiment (#279) — kv-utilization scorer produced 200x worse distribution uniformity than queue-depth at rate=5000. See issues #282, #283.

**Freshness hierarchy:**
- **Tier 1 — Synchronously fresh (cluster-owned):** PendingRequests
- **Tier 2 — Stale within tick (instance-owned):** QueueDepth, BatchSize
- **Tier 3 — Stale across batch steps (instance-owned):** KVUtilization, CacheHitRate

**Check:** When writing a new scorer, identify which snapshot fields it reads and their freshness tier. If using only Tier 3 signals, document why or combine with a Tier 1 scorer.

**Enforced:** Design review, scorer implementation review.

---

## Quick Reference Checklist

For PR authors — check each rule before submitting:

- [ ] **R1:** No silent `continue`/`return` dropping data
- [ ] **R2:** Map keys sorted before float accumulation or ordered output
- [ ] **R3:** Every new CLI flag validated (zero, negative, NaN, Inf)
- [ ] **R4:** All struct construction sites audited for new fields
- [ ] **R5:** Resource allocation loops handle mid-loop failure with rollback
- [ ] **R6:** No `logrus.Fatalf` or `os.Exit` in `sim/` packages
- [ ] **R7:** Invariant tests alongside any golden tests
- [ ] **R8:** No exported mutable maps
- [ ] **R9:** `*float64` for YAML fields where zero is valid
- [ ] **R10:** YAML strict parsing (`KnownFields(true)`)
- [ ] **R11:** Division by runtime-derived denominators guarded
- [ ] **R12:** Golden dataset regenerated if output changed
- [ ] **R13:** New interfaces work for 2+ implementations
- [ ] **R14:** No method spans multiple module responsibilities
- [ ] **R15:** Stale PR references resolved
- [ ] **R16:** Config params grouped by module
- [ ] **R17:** Routing scorer signals documented for freshness tier
```

---

### K.2: `docs/standards/invariants.md`

```markdown
# BLIS System Invariants

Invariants are properties that must hold at all times during and after simulation. They are verified by invariant tests (see R7) and checked during self-audit (Step 4.75).

## INV-1: Request Conservation

**Statement:** `injected_requests == completed_requests + still_queued + still_running` at simulation end (all levels).

**Full pipeline:** `num_requests == injected_requests + rejected_requests` (from anomaly counters).

**Verification:** `sim/cluster/cluster_test.go` — conservation tests. Conservation fields (`still_queued`, `still_running`, `injected_requests`) are included in CLI JSON output.

**Evidence:** Issue #183 — a silently-dropped request violated conservation for months.

---

## INV-2: Request Lifecycle

**Statement:** Requests transition `queued → running → completed`. No invalid transitions. Requests not completed before horizon remain in current state.

**Verification:** State machine assertions in request processing code.

---

## INV-3: Clock Monotonicity

**Statement:** Simulation clock never decreases. Every event's timestamp ≥ the previous event's timestamp.

**Verification:** Clock is advanced in the event loop only via min-heap extraction, which guarantees non-decreasing order.

---

## INV-4: KV Cache Conservation

**Statement:** `allocated_blocks + free_blocks = total_blocks` at all times.

**Verification:** Checked after every allocation/deallocation. Transactional allocation with rollback on mid-loop failure (R5).

---

## INV-5: Causality

**Statement:** `arrival_time <= enqueue_time <= schedule_time <= completion_time` for every request.

**Verification:** Per-request metric timestamps recorded at each lifecycle stage. Invariant tests verify ordering for all completed requests.

---

## INV-6: Determinism

**Statement:** Same seed must produce byte-identical stdout across runs.

**Verification:** Run same configuration twice with same seed; diff stdout. Wall-clock timing goes to stderr (not stdout).

**Common violation sources:**
- Go map iteration feeding output ordering (R2)
- Floating-point accumulation order dependencies
- Wall-clock-dependent randomness (must use PartitionedRNG)
- Stateful scorers with non-deterministic internal state

---

## INV-7: Signal Freshness Hierarchy

**Statement:** Routing snapshot signals have tiered freshness due to DES event ordering. Cluster events at tick T drain before instance events at tick T.

| Signal | Owner | Freshness | Updated By |
|--------|-------|-----------|------------|
| PendingRequests | Cluster | Synchronous | `RoutingDecisionEvent.Execute()` |
| QueueDepth | Instance | Stale within tick | `QueuedEvent.Execute()` |
| BatchSize | Instance | Stale within tick | `StepEvent.Execute()` |
| KVUtilization | Instance | Stale across batch steps | `makeRunningBatch()` → `AllocateKVBlocks()` |
| CacheHitRate | Instance | Stale across batch steps | `makeRunningBatch()` |

**Design implication:** `EffectiveLoad()` = `QueueDepth + BatchSize + PendingRequests` compensates for Tier 2 staleness by including the Tier 1 PendingRequests term. KVUtilization has no analogous compensation.

**Verification:** H3 hypothesis experiment (`hypotheses/h3-signal-freshness/`).

**Evidence:** Issues #282, #283. At rate=5000, kv-utilization-only routing produces 200x worse distribution uniformity than queue-depth.
```

---

### K.3: `docs/standards/principles.md`

```markdown
# BLIS Engineering Principles

Principles guide design decisions. The [antipattern rules](rules.md) are specific, checkable manifestations of these principles. The [invariants](invariants.md) are properties that must always hold.

## Separation of Concerns

- `sim/` is a library — never call `os.Exit`, `logrus.Fatalf`, or terminate the process. Return errors. Only `cmd/` may terminate. *(Enforced by R6)*
- Cluster-level policies (admission, routing) receive `*RouterState` with global view. Instance-level policies (priority, scheduler) receive only local data. Never leak cluster state to instance-level code.
- Bridge types (`RouterState`, `RoutingSnapshot`) live in `sim/` to avoid import cycles.
- Unidirectional dependency: `cmd/ → sim/cluster/ → sim/` and `sim/cluster/ → sim/trace/`. `sim/` never imports subpackages.

## Interface Design

- Single-method interfaces where possible (`AdmissionPolicy`, `RoutingPolicy`, `PriorityPolicy`, `InstanceScheduler`).
- Query methods must be pure — no side effects, no state mutation, no destructive reads. Separate `Get()` and `Consume()` for query-and-clear.
- Factory functions must validate inputs: `IsValid*()` check + switch/case + panic on unknown.
- Interfaces defined by behavioral contract, not one implementation's data model. *(Enforced by R13)*
- Methods operate within a single module's responsibility. *(Enforced by R14)*

## Configuration Design

- Group configuration by module. *(Enforced by R16)*
- Each module's config independently specifiable and validatable.

## Canonical Constructors

- Every struct constructed in multiple places needs a canonical constructor. Struct literals appear in exactly one place. *(Enforced by R4)*
- Before adding a field, grep for ALL construction sites.

## Output Channel Separation

- **stdout** (deterministic): simulation results — metrics JSON, fitness scores, anomaly counters, KV cache metrics, per-SLO metrics, trace summaries. Use `fmt.Println`/`fmt.Printf`.
- **stderr** (diagnostic): configuration echoes, progress markers, warnings, errors. Use `logrus.*`, controlled by `--log`.
- Rule of thumb: if a user piping to a file would want to capture it, use `fmt`. If it's debugging context, use `logrus`.

## Error Handling Boundaries

| Layer | Strategy | Example |
|-------|----------|---------|
| CLI (`cmd/`) | `logrus.Fatalf` for user errors | Invalid `--rate` value |
| Library (`sim/`) | `panic()` for invariant violations | Unknown policy name in factory |
| Library (`sim/`) | `error` return for recoverable failures | File I/O, parse errors |
| Runtime (`sim/`) | `bool` return for expected conditions | KV allocation failure → preemption |

Never use `continue` in an error path without propagating, counting, or documenting why it's safe. *(Enforced by R1)*

## BDD/TDD Development

1. Write behavioral contracts first (GIVEN/WHEN/THEN)
2. Implement tests before code
3. Use table-driven tests
4. Test laws, not just values — invariant tests alongside golden tests *(Enforced by R7)*
5. Refactor survival test: "Would this test still pass if the implementation were completely rewritten but the behavior preserved?"
6. THEN clauses drive test quality — structural THEN → structural test

**Prohibited assertion patterns** (structural — break on refactor):
- Type assertions: `policy.(*ConcreteType)`
- Internal field access: `obj.internalField`
- Exact formula reproduction: `assert.Equal(score, 0.6*cache + 0.4*load)`

**Required assertion patterns** (behavioral — survive refactor):
- Observable output: `assert.Equal(policy.Compute(req, clock), 0.0)`
- Invariant verification: `assert.Equal(completed+queued+running, injected)`
- Ordering/ranking: `assert.True(scoreA > scoreB)`
```

---

### K.4: `docs/standards/experiments.md`

```markdown
# BLIS Experiment Standards

Hypothesis-driven experimentation is a first-class activity in BLIS — equal in rigor to implementation and design. Experiments serve three purposes:

1. **Validation** — confirm that implemented features work as designed (e.g., prefix-affinity produces 2.4x TTFT improvement for multi-turn chat)
2. **Discovery** — surface bugs, design gaps, and undocumented limitations (e.g., H3 revealed KV utilization signal staleness → 3 new issues, 1 new rule, 1 new invariant)
3. **Documentation** — each experiment becomes a reproducible artifact that helps users understand when to use which configuration

## Experiment Classification

Every hypothesis must be classified before designing the experiment. The classification determines rigor requirements.

### Type 1: Deterministic Experiments

**Definition:** Verify exact properties — invariants, conservation laws, error handling boundaries. Same seed = same result, guaranteed.

**Requirements:**
- Single seed sufficient (determinism is the point)
- Pass/fail is exact — the invariant holds or it doesn't
- Failure is ALWAYS a bug (never noise)
- No statistical analysis needed

**Examples:**
- H12: Request conservation holds across all policy configurations
- H13: Same seed produces byte-identical output
- H22: Zero KV blocks panics at CLI boundary, not deep in simulation

**Pass criteria:** The invariant holds for every configuration tested. One failure = bug.

---

### Type 2: Statistical Experiments

**Definition:** Compare metrics (TTFT, throughput, distribution uniformity) across configurations. Results vary by seed.

**Requirements:**
- **Minimum 3 seeds** (42, 123, 456) for each configuration
- **Effect size thresholds:**
  - **Significant:** >20% improvement consistent across ALL seeds
  - **Inconclusive:** <10% in any seed
  - **Equivalent:** within 5% across all seeds (for equivalence tests)
- **Directional consistency:** the predicted direction must hold across ALL seeds. One contradicting seed = hypothesis not confirmed
- **Report:** mean, min, max across seeds for primary metric. Include per-seed values for transparency.

**Subtypes:**

#### Dominance
A is strictly better than B on metric M.

- **Analysis:** Compare metric M for A vs B across all seeds. Compute ratio per seed.
- **Pass:** A beats B on M for all seeds, with >20% effect size in every seed.
- **Example:** H3 — queue-depth TTFT is 1.7-2.8x better than kv-utilization across 3 seeds.

#### Monotonicity
Increasing X should monotonically increase/decrease Y.

- **Analysis:** Run at ≥3 values of X. Verify Y changes monotonically.
- **Pass:** Y is strictly monotonic in X across all seeds. No inversions.
- **Example:** H8 — reducing total KV blocks increases preemption frequency. H9 — increasing prefix_length decreases TTFT.

#### Equivalence
A ≈ B within tolerance (baseline sanity checks).

- **Analysis:** Compare metric M for A vs B. Compute percentage difference per seed.
- **Pass:** |A - B| / max(A, B) < 5% across all seeds.
- **Example:** H4 — round-robin ≈ least-loaded for uniform workloads at low rates. H23 — all policies equivalent at near-zero load.

#### Pareto
No single configuration dominates all metrics simultaneously.

- **Analysis:** Run N configurations, measure multiple metrics. Identify Pareto-optimal set.
- **Pass:** At least 2 configurations are Pareto-optimal (each best on ≥1 metric).
- **Example:** H17 — different scorer weights optimize for different objectives (TTFT vs throughput).

---

## Experiment Design Rules

### ED-1: Controlled comparison
Vary exactly one dimension between configurations. Everything else held constant (same model, same instances, same workload, same seed). If the experiment requires varying multiple dimensions, decompose into separate sub-experiments.

### ED-2: Rate awareness
Many effects are rate-dependent (e.g., signal freshness only matters at high rates). When the hypothesis involves load-dependent behavior:
- Run at the target rate where the effect is expected
- Also run at a rate where the effect should vanish (to confirm the mechanism, not just the outcome)
- Document the rate-dependent transition point if observed

### ED-3: Precondition verification
Before comparing configurations, verify the experiment preconditions hold. Examples:
- Testing SJF vs FCFS? Verify queue depth exceeds batch size (otherwise both produce identical batches).
- Testing cache hit benefit? Verify KV blocks are large enough to hold the prefix (otherwise LRU eviction destroys it).

Document the precondition check in the experiment script (not just in prose).

### ED-4: Workload seed independence
**Known issue (#284):** Workload-spec YAMLs have their own `seed:` field independent of the CLI `--seed` flag. When using `--workload-spec`:
- Varying `--seed` alone changes the simulation RNG but NOT the workload (arrival times, token counts)
- For true multi-seed experiments with workload-spec, you must also vary the YAML seed
- For CLI-generated workloads (`--rate`, `--num-requests`), `--seed` controls everything

**Until #284 is resolved:** when using workload-spec, document whether the experiment varies workload seeds, CLI seeds, or both. If only CLI seeds are varied, note that the workload is identical across runs.

### ED-5: Reproducibility
Every experiment must be reproducible from its artifacts alone:
- `run.sh` must build the binary and run all variants
- Exact seed values documented
- Exact commit hash recorded (or the experiment is tied to a specific branch/PR)
- No manual steps between script invocation and results

---

## Findings Classification

Every experiment produces findings. Each finding MUST be classified:

| Finding Type | Definition | Action Required |
|-------------|------------|-----------------|
| **Confirmation** | The hypothesis holds; the system works as designed | Document in FINDINGS.md. No issues needed. |
| **Bug discovery** | The hypothesis failed due to a code defect | File GitHub issue. Fix in separate PR. |
| **New rule** | The experiment revealed a pattern that should be checked in all future PRs | Add to `docs/standards/rules.md` with evidence. |
| **New invariant** | The experiment revealed a property that must always hold | Add to `docs/standards/invariants.md`. |
| **Design limitation** | The system works as coded but has an undocumented behavioral limitation | Document in FINDINGS.md + file issue for design doc update. |
| **Surprise** | An unexpected result that doesn't fit other categories | Document in FINDINGS.md. May spawn new hypotheses. |

### The Audit Step

After analyzing results, EVERY experiment MUST audit findings against `docs/standards/`:

1. Do any findings reveal violations of existing rules or principles?
2. Do any findings suggest a new rule, invariant, or principle is needed?
3. Do any findings confirm that existing rules/invariants hold under new conditions?

This audit is what makes experiments a feedback loop into the standards. Example: H3 confirmed that the llm-d default config is robust (confirmation) AND revealed that KV utilization is stale at high rates (design limitation → new rule R17 + new invariant INV-7 + 3 issues).

---

## Experiment Artifacts

Each hypothesis experiment lives in `hypotheses/<name>/` with:

| File | Purpose |
|------|---------|
| `run.sh` | Self-contained script: builds binary, runs all variants, calls analyzer |
| `analyze.py` | Output parser producing formatted comparison tables |
| `FINDINGS.md` | Results, root cause analysis, findings classification, standards audit |
| `*.yaml` (optional) | Custom workload specs for this experiment |

Scripts must be reproducible — running `./run.sh` on the same commit produces deterministic output.
```

---

### K.5: `docs/process/design.md` (stub)

```markdown
# Design Process

> **Status:** Draft — to be expanded from experience.

This document describes the process for writing a BLIS design document. For the design document template itself, see [docs/templates/design-guidelines.md](../templates/design-guidelines.md).

## When a Design Doc is Needed

- New subsystem modules (new interface + integration)
- Backend swaps (alternative implementations requiring interface extraction)
- Architecture changes affecting module boundaries

**Not needed for:** Bug fixes, new policy templates behind existing interfaces, documentation changes.

## Steps

1. **Identify the extension type** — policy template, subsystem module, backend swap, or tier composition (see [design guidelines](../templates/design-guidelines.md) Section 5)
2. **Choose the design doc species** — decision record, specification, problem analysis, or system overview (Section 3.2)
3. **Complete the DES checklist** (Section 2.6) — model scoping, event design, state/statistics, V&V, randomness
4. **Write the design doc** per the template's required sections (Section 3.3): motivation, scope, modeling decisions, invariants, decisions with trade-offs, extension points, validation strategy
5. **Apply the staleness test** (Section 3.1) — would this content mislead if the implementation changes?
6. **Human review** — approve before macro/micro planning begins

## Quality Gates

- [ ] Extension type identified and correct recipe followed
- [ ] DES checklist from Section 2.6 completed
- [ ] No prohibited content (Section 3.4): no Go structs, no method implementations, no file:line references
- [ ] Every non-obvious decision has alternatives listed with rationale
- [ ] Validation strategy specified (which invariants? against what real-system data?)

## References

- Template: [docs/templates/design-guidelines.md](../templates/design-guidelines.md)
- Standards: [docs/standards/rules.md](../standards/rules.md), [docs/standards/invariants.md](../standards/invariants.md)
```

---

### K.6: `docs/process/macro-plan.md` (stub)

```markdown
# Macro Plan Process

> **Status:** Draft — to be expanded from experience.

This document describes the process for creating a macro-level implementation plan (multi-PR feature). For the macro plan template, see [docs/templates/macro-plan.md](../templates/macro-plan.md).

## When a Macro Plan is Needed

- Features spanning 2+ PRs
- Work requiring a dependency DAG between PRs
- Features touching multiple module boundaries

**Not needed for:** Single-PR features, bug fixes, documentation changes.

## Steps

1. **Design doc(s) as input** — read the relevant design doc(s) and/or GitHub issues
2. **Decompose into PRs** — each PR should be independently mergeable and testable
3. **Define the dependency DAG** — which PRs can be parallelized? Which must be sequential?
4. **Define module contracts per PR boundary** — what does each PR guarantee to the next?
5. **Identify frozen interfaces** — which interfaces are stable (can be developed against in parallel)?
6. **Identify flexible internals** — which implementation details may change during micro-planning?
7. **Human review** — approve before micro-planning begins for any PR in the plan

## Quality Gates

- [ ] Every PR in the plan is independently mergeable (no PR requires another PR's uncommitted code)
- [ ] Dependency DAG has no cycles
- [ ] Module contracts are testable with mocks (parallel development enabled)
- [ ] No Go struct definitions or method implementations (those belong in micro plans)
- [ ] Extension friction assessed for each new module boundary

## References

- Template: [docs/templates/macro-plan.md](../templates/macro-plan.md)
- Design guidelines: [docs/templates/design-guidelines.md](../templates/design-guidelines.md)
- Standards: [docs/standards/rules.md](../standards/rules.md)
```

---

### K.7: `docs/process/hypothesis.md` (stub)

```markdown
# Hypothesis Experiment Process

> **Status:** Draft — to be expanded from experience.

This document describes the process for running a hypothesis-driven experiment. For experiment standards (rigor, classification, analysis), see [docs/standards/experiments.md](../standards/experiments.md). For the experiment template, see [docs/templates/hypothesis.md](../templates/hypothesis.md).

## When to Run Experiments

- Validating that a new feature works as designed (post-PR confirmation)
- Testing intuitive claims about system behavior (from `docs/plans/research.md`)
- Investigating unexpected behavior observed during development
- Exploring design tradeoffs between configurations

## Steps

1. **Select or pose hypothesis** — from `docs/plans/research.md` or from a new observation
2. **Classify** — deterministic or statistical? If statistical, which subtype? (See [experiments.md](../standards/experiments.md))
3. **Create worktree** — `git worktree add .worktrees/hypothesis-<name> -b hypothesis/<name>`
4. **Design experiment** — controlled comparison (ED-1), rate awareness (ED-2), precondition verification (ED-3), seed strategy (ED-4)
5. **Implement** — create `hypotheses/<name>/run.sh`, `analyze.py`
6. **Run** — execute across required seeds; verify reproducibility (ED-5)
7. **Analyze** — produce comparison tables, compute effect sizes
8. **Classify findings** — confirmation, bug, new rule, new invariant, design limitation, or surprise
9. **Audit against standards** — check findings against `docs/standards/rules.md` and `docs/standards/invariants.md`
10. **Document** — write `FINDINGS.md` with results, root cause, classification, and audit
11. **File issues** — for any bugs, design limitations, or new rules/invariants discovered
12. **Commit and PR** — rebase on upstream/main, push, create PR

## Quality Gates

- [ ] Hypothesis classified (deterministic or statistical + subtype)
- [ ] Experiment design follows ED-1 through ED-5
- [ ] Results reproducible via `./run.sh`
- [ ] Findings classified per the findings table
- [ ] Standards audit completed
- [ ] Issues filed for all actionable findings

## References

- Standards: [docs/standards/experiments.md](../standards/experiments.md)
- Template: [docs/templates/hypothesis.md](../templates/hypothesis.md)
- Hypothesis catalog: [docs/plans/research.md](../plans/research.md)
- Example experiments: `hypotheses/h3-signal-freshness/`, `hypotheses/prefix-affinity/`
```

---

### K.8: `docs/templates/hypothesis.md`

```markdown
# Hypothesis Experiment Template

> **For Claude:** Use this template when creating a new hypothesis experiment in `hypotheses/<name>/`.

## FINDINGS.md Structure

Every experiment's `FINDINGS.md` MUST contain these sections:

```
# <Hypothesis Name>

**Status:** Confirmed | Refuted | Partially confirmed | Inconclusive
**Tier:** <tier number from research.md>
**Type:** Deterministic | Statistical (<subtype>)
**Date:** YYYY-MM-DD

## Hypothesis

> <Quoted hypothesis statement — intuitive claim about system behavior>

## Experiment Design

**Classification:** <Deterministic | Statistical/Dominance | Statistical/Monotonicity | Statistical/Equivalence | Statistical/Pareto>

**Configurations compared:**
- A: <description + exact CLI flags>
- B: <description + exact CLI flags>

**Controlled variables:** <what is held constant>
**Varied variable:** <what differs between A and B>
**Seeds:** <list of seeds used>
**Preconditions verified:** <what was checked before running>

## Results

<Comparison tables with per-seed values>

## Root Cause Analysis

<Why the results are what they are — trace through the code/architecture>

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| <finding 1> | Confirmation / Bug / New rule / etc. | <issue number or "documented here"> |

## Standards Audit

Findings checked against docs/standards/:
- [ ] Any violations of existing rules? <list or "none found">
- [ ] Any new rules needed? <list or "none">
- [ ] Any new invariants needed? <list or "none">
- [ ] Any existing rules/invariants confirmed? <list or "none">

## Implications for Users

<Practical guidance derived from this experiment>

## Reproducing

\`\`\`bash
cd hypotheses/<name>
./run.sh
\`\`\`
```

## run.sh Structure

```bash
#!/bin/bash
# <Hypothesis name>
# <One-line description>
# Usage: ./run.sh [--rebuild]

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BINARY="$REPO_ROOT/simulation_worker"

# Build if needed
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -x "$BINARY" ]]; then
    echo "Building simulation_worker..."
    (cd "$REPO_ROOT" && go build -o simulation_worker main.go)
fi

MODEL="meta-llama/llama-3.1-8b-instruct"

run_sim() {
    # Wrapper around the binary with common flags
    "$BINARY" run --model "$MODEL" --num-instances 4 \
        --log error --summarize-trace --trace-level decisions \
        "$@" 2>/dev/null
}

RESULTS_DIR=$(mktemp -d)
trap "rm -rf $RESULTS_DIR" EXIT

# ── Experiment sections ──────────────────────────────
# Each experiment: run configurations, save to temp files, analyze
# ─────────────────────────────────────────────────────
```

## analyze.py Structure

```python
#!/usr/bin/env python3
"""Analysis script for <hypothesis name>.

Parses BLIS multi-block output and produces comparison tables.
"""
import json, math, re, sys
from pathlib import Path

def parse_output(filepath):
    """Parse BLIS output → cluster metrics + distribution + cache hit rate."""
    content = Path(filepath).read_text()
    # Extract cluster JSON block
    # Extract target distribution from trace summary
    # Extract KV cache metrics
    # Return dict with ttft_mean, ttft_p99, throughput, dist, hit_rate, etc.
    ...
```
```

---

### K.9: CLAUDE.md Compact Rule Table (for Task 8)

The "Antipattern Prevention" section in CLAUDE.md should be replaced with:

```markdown
### Antipattern Prevention

17 rules, each tracing to a real bug. Full details (evidence, checks, enforcement): see [`docs/standards/rules.md`](docs/standards/rules.md).

| # | Rule | One-sentence summary |
|---|------|---------------------|
| R1 | No silent data loss | Every error path must return error, panic, or increment counter — never silently drop data |
| R2 | Sort map keys | Map iteration feeding float sums or output ordering must sort keys first (determinism) |
| R3 | Validate CLI flags | Every numeric flag validated for zero, negative, NaN, Inf |
| R4 | Construction site audit | Adding a struct field? Grep for ALL literal construction sites, update every one |
| R5 | Transactional mutation | Resource-allocating loops must rollback on mid-loop failure |
| R6 | No Fatalf in library | `sim/` never terminates the process — return errors to callers |
| R7 | Invariant tests | Every golden test needs a companion invariant test verifying a system law |
| R8 | No exported maps | Validation maps unexported; expose via `IsValid*()` accessors |
| R9 | YAML pointer types | Use `*float64` when zero is a valid user value |
| R10 | Strict YAML parsing | `yaml.KnownFields(true)` — typos must cause errors |
| R11 | Guard division | Runtime-derived denominators must be checked for zero |
| R12 | Golden regeneration | Regenerate and document golden dataset when output changes |
| R13 | Multi-impl interfaces | New interfaces must work for ≥2 backends |
| R14 | Single-module methods | No method spans scheduling + latency + metrics — extract concerns |
| R15 | Stale PR references | Grep for `planned for PR N` after completing PR N |
| R16 | Config by module | Group config parameters by module, not monolithic structs |
| R17 | Signal freshness | Document which routing signals are synchronously fresh vs stale |
```

---

### K.10: Redirect Stub Template (for Task 7)

Each redirect stub at old locations:

```markdown
> **This document has moved.**
>
> New location: [<new path>](<relative link>)
>
> This stub exists for backward compatibility. Please update your bookmarks.
```
