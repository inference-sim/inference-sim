# Human-First Contributing Docs Rewrite — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the Contributing section documentation so humans read it as a process guide with optional automation, not as an agent directive manual.

**The problem today:** The Contributing section serves two audiences — human contributors and Claude Code agents — but the balance is wrong. `pr-workflow.md` (1168 lines) is 70% agent directives with literal prompt strings. Templates (`micro-plan.md`, `macro-plan.md`) open as LLM system prompts. Manual alternatives are one-line afterthoughts. A human contributor visiting the docs site would be confused by content addressed to "You are operating inside a real repository."

**What this PR adds:**
1. **Human-first process docs** — Every step in `pr-workflow.md` and `hypothesis.md` reads as a human action, with tool automation as optional MkDocs admonition callouts
2. **Separated templates** — `micro-plan.md` and `macro-plan.md` become human-readable output format descriptions; agent preambles move to companion `-prompt.md` files
3. **Consistent skill references** — All contributing docs use the same admonition callout pattern (replacing the current mix of Prerequisites tables, inline invocations, and Quick Guide tables)

**Why this matters:** External contributors seeing LLM directives as the primary documentation will either be confused (humans) or skip the docs entirely. Making humans the primary audience makes the project more approachable while preserving all automation benefits for Claude Code users.

**Architecture:** Docs-only changes across `docs/contributing/`. No Go code, no test changes, no behavior changes to the process itself — only presentation. MkDocs admonition extension (already enabled) powers the skill callouts. Agent prompt files live alongside templates but are excluded from the MkDocs nav.

**Source:** GitHub issue #464

**Closes:** Fixes #464

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR rewrites the Contributing section documentation to be human-first. The process steps remain identical (worktree → plan → review → implement → review → audit → commit); only the presentation changes. The core principle follows `CONTRIBUTING.md` (#423), which already solved this problem — humans get a first-class workflow, Claude Code users get acceleration through skill callouts.

The rewrite affects 6 files directly (`pr-workflow.md`, `hypothesis.md`, `design-process.md`, `macro-planning.md`, `micro-plan.md`, `macro-plan.md`) plus supporting files (`contributing/index.md`, `CLAUDE.md`, `CONTRIBUTING.md`, convergence-review skill files). No code changes. No process semantic changes.

The rewrite also touches `CONTRIBUTING.md` (skill invocation paths) and `.claude/skills/convergence-review/pr-prompts.md` (template reference) to keep all consumers of the template paths consistent.

Adjacent documents unaffected: `convergence.md` (already audience-neutral), `standards/*.md` (already human-readable), `design-guidelines.md` (already technical reference), `extension-recipes.md` (already practical guide).

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Human-Readable Primary Path**
- GIVEN any step in `pr-workflow.md` or `hypothesis.md`
- WHEN a human contributor reads the step
- THEN the step describes a human action (e.g., "Write an implementation plan following the micro-plan template") with tool automation in an admonition callout
- MECHANISM: Each step heading names the human action; skill invocations appear only inside `!!! tip "Automation"` blocks

**BC-2: Empirical Rationale Preserved**
- GIVEN any "Why this step exists" rationale section in the current `pr-workflow.md` or `hypothesis.md`
- WHEN the rewrite is complete
- THEN every rationale section (with its PR/issue references) appears in the rewritten document
- MECHANISM: Rationale sections are copied verbatim or lightly edited for context; none are removed

**BC-3: Template/Prompt Separation**
- GIVEN the `micro-plan.md` and `macro-plan.md` template files
- WHEN a human visits the docs site Contributing > Templates section
- THEN they see a human-readable output format description with section explanations, not LLM directives ("You are operating inside a real repository")
- MECHANISM: Human-readable template stays at original path; agent preamble moves to `*-prompt.md` companion file

**BC-4: Consistent Skill Reference Pattern**
- GIVEN any skill reference across MkDocs-rendered contributing docs (`docs/contributing/*.md`)
- WHEN the rewrite is complete
- THEN every skill reference uses the MkDocs admonition pattern: `!!! tip "Automation"` followed by the skill invocation and a link to the Skills & Plugins page
- MECHANISM: Replace Prerequisites tables (3 docs), inline invocations (~20 instances), and Quick Guide tables (1 doc) with the standard admonition. Exception: `CONTRIBUTING.md` (GitHub-rendered) uses blockquote callouts instead of MkDocs admonitions.

**BC-5: First-Class Manual Path**
- GIVEN any contributing process doc (`pr-workflow.md`, `hypothesis.md`, `design-process.md`, `macro-planning.md`)
- WHEN a contributor without Claude Code reads the doc
- THEN they find a complete workflow they can follow using only standard git commands and manual review checklists
- MECHANISM: The primary step text IS the manual path; automation is additive, not the default

**BC-6: MkDocs Build Success**
- GIVEN all rewritten documentation files
- WHEN `mkdocs build --strict` runs
- THEN the build succeeds with zero errors and zero warnings
- MECHANISM: All cross-references updated, nav entries correct, admonition syntax valid

#### Negative Contracts

**BC-7: No Process Semantic Changes**
- GIVEN the current PR workflow (7 steps) and hypothesis workflow (11 steps)
- WHEN the rewrite is complete
- THEN the steps, their ordering, and their quality gates are identical — only the presentation changes
- MECHANISM: The rewrite preserves all steps, review perspectives, convergence protocol references, and quality checklists

**BC-8: No Content Loss**
- GIVEN the review perspective checklists in `pr-workflow.md` (10 perspectives × 2 gates) and `hypothesis.md` (5+5+10 perspectives)
- WHEN the rewrite is complete
- THEN every perspective checklist item is preserved
- MECHANISM: Perspectives are reorganized for readability but no checklist items are removed

### C) Component Interaction

```
docs/contributing/
├── pr-workflow.md          ← Major rewrite (1168→~600-700 lines)
├── hypothesis.md           ← Moderate rewrite (743→~500 lines)
├── design-process.md       ← Light touch (standardize callouts)
├── macro-planning.md       ← Light touch (standardize callouts)
├── convergence.md          ← No change
├── index.md                ← Minor updates
├── templates/
│   ├── micro-plan.md       ← Rewrite as human-readable template
│   ├── micro-plan-prompt.md ← NEW: agent preamble (moved from micro-plan.md)
│   ├── macro-plan.md       ← Rewrite as human-readable template
│   ├── macro-plan-prompt.md ← NEW: agent preamble (moved from macro-plan.md)
│   ├── design-guidelines.md ← No change
│   └── hypothesis.md       ← No change
├── standards/              ← No changes
└── extension-recipes.md    ← No change
```

Data flow: Human reads process doc → follows steps → optionally uses skill callouts for automation. Agent reads prompt file → generates artifact per template format.

New files: `micro-plan-prompt.md`, `macro-plan-prompt.md` (agent preamble content extracted from current templates).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Phase 2 template split: "Agent preamble in `.claude/skills/` or template frontmatter" | Agent preamble in `templates/*-prompt.md` companion files | SIMPLIFICATION: Companion files are simpler than skill modifications and keep agent content discoverable alongside templates. The `@docs/contributing/templates/micro-plan-prompt.md` reference pattern works unchanged with writing-plans skill. |
| Phase 3: "Cross-reference the new Skills & Plugins page (#462 issue 17)" | Cross-reference `docs/guide/skills-and-plugins.md` which already exists | CORRECTION: The Skills & Plugins page was already created in #462/#465. |
| N/A | All `writing-plans` invocations updated to reference `micro-plan-prompt.md` instead of `micro-plan.md` | ADDITION: After splitting templates, skill invocations in `pr-workflow.md`, `CONTRIBUTING.md`, and `.claude/skills/convergence-review/pr-prompts.md` must reference the prompt file (not the human-readable template). Without this, plan generation quality would silently degrade. |
| N/A | `CONTRIBUTING.md` added to scope | ADDITION: Issue #464 doesn't list CONTRIBUTING.md, but it has 8 references to `micro-plan.md`/`macro-plan.md` that must be audited for the template split, and its skill invocation paths should use the standard admonition pattern for consistency. |
| Issue says `pr-workflow.md` "consolidate 1168→~400 lines" | Plan targets ~600-700 lines | SCOPE_CHANGE: The ~400 line target in the issue was aspirational. Content preservation (BC-2, BC-8) requires ~350 lines for the 20 perspective checklists alone. Target revised to ~600-700 with content preservation as priority over line count. |
| Issue Phase 2 lists `templates/hypothesis.md` for "Light cleanup" | Plan marks it "No change" | DEFERRAL: `hypothesis.md` template opens with `> **For Claude:** Use this template...` but the rest is audience-neutral template structure. The one-line agent directive is minimal and not confusing to humans. Deferring to keep this PR focused on the high-impact rewrites. |
| N/A | `.claude/skills/convergence-review/design-prompts.md` added to scope | ADDITION: References `macro-plan.md` template by name. Agent-facing file that needs the prompt version path after the split. |

### E) Review Guide

**The tricky part:** `pr-workflow.md` rewrite — condensing 1168 lines to ~500 while preserving all review perspective checklists, rationale sections, and the workflow evolution appendix. The risk is accidentally dropping a perspective checklist item or rationale reference.

**What to scrutinize:** BC-2 (rationale preservation) and BC-8 (no content loss). Diff the review perspective sections line-by-line against the original.

**What's safe to skim:** `design-process.md` and `macro-planning.md` changes (light touch — only callout pattern standardization). MkDocs nav changes (mechanical).

**Known debt:** The `pr-workflow.md` Appendix: Workflow Evolution section (v1.0-v3.0) is ~60 lines of version history. The issue doesn't specify whether to preserve, condense, or remove it. Plan preserves it in a collapsed details block.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `docs/contributing/templates/micro-plan-prompt.md` — Agent preamble extracted from micro-plan.md
- `docs/contributing/templates/macro-plan-prompt.md` — Agent preamble extracted from macro-plan.md

**Files to modify:**
- `docs/contributing/pr-workflow.md` — Major rewrite
- `docs/contributing/hypothesis.md` — Moderate rewrite
- `docs/contributing/design-process.md` — Light touch
- `docs/contributing/macro-planning.md` — Light touch
- `docs/contributing/templates/micro-plan.md` — Rewrite as human-readable
- `docs/contributing/templates/macro-plan.md` — Rewrite as human-readable
- `docs/contributing/index.md` — Minor updates
- `CONTRIBUTING.md` — Update template references from `micro-plan.md` to `micro-plan-prompt.md` in skill paths; standardize skill callouts
- `.claude/skills/convergence-review/pr-prompts.md` — Update template reference from `micro-plan.md` to `micro-plan-prompt.md`; update canonical-source header
- `.claude/skills/convergence-review/design-prompts.md` — Update `macro-plan.md` reference to `macro-plan-prompt.md`
- `CLAUDE.md` — Update "Written per" refs (lines 54-55), file organization tree, governance docs templates section

**Key decisions:**
- Admonition pattern for ALL skill references (consistent visual treatment)
- Agent prompt files as companion `*-prompt.md` files (not `.claude/skills/`)
- Workflow Evolution appendix preserved in `<details>` block (not removed)
- Line target: `pr-workflow.md` ~600-700, `hypothesis.md` ~500 (content preservation over line count)

### G) Task Breakdown

---

### Task 1: Rewrite pr-workflow.md — Human-First Process Guide

**Contracts Implemented:** BC-1, BC-2, BC-4, BC-5, BC-7, BC-8

**Files:**
- Modify: `docs/contributing/pr-workflow.md`

**Step 1: Write the rewritten pr-workflow.md**

The rewrite follows these structural principles:
1. Remove the Prerequisites table (skill requirements) — skills referenced in callouts instead
2. Remove the Quick Reference table (skill invocations as primary) — steps ARE the reference
3. Remove Current Template Versions section — moved to contributing/index.md
4. Each step: human action as primary text, skill in `!!! tip "Automation"` admonition
5. Review perspectives: kept as subsections but introduced as "what to check" not "agent prompts"
6. Consolidate Examples A and B into a single "Example Walkthrough" section
7. Workflow Evolution appendix in a `<details>` collapsed block
8. PR Size Tiers table: kept (already human-readable)
9. All "Why this step exists" rationale: preserved verbatim
10. Tips for Success and Common Issues: kept, skill references updated to admonition pattern

The rewritten document structure:

```markdown
# PR Development Workflow

Status line, intro paragraph.

## Overview (ASCII diagram — kept)

## Step-by-Step Process

### Step 1: Create an Isolated Workspace
[Human action: create a git worktree]
!!! tip "Automation"
    `/superpowers:using-git-worktrees <name>`

### Step 2: Write an Implementation Plan
[Human action: write plan following micro-plan template]
!!! tip "Automation"
    `/superpowers:writing-plans ...`

### Step 2.5: Review the Plan
[Human action: review from 10 perspectives]
Perspective checklists (kept as-is)
Convergence protocol reference
!!! tip "Automation"
    `/pr-review-toolkit:review-pr` then `/convergence-review pr-plan <path>`

### Step 3: Human Review
[Kept as-is — already human-focused]

### Step 4: Implement the Plan
[Human action: implement TDD tasks from the plan]
!!! tip "Automation"
    `/superpowers:executing-plans @<plan-path>`

### Step 4.5: Review the Code
[Same pattern as Step 2.5]

### Step 4.75: Pre-Commit Self-Audit
[Kept as-is — already human-focused, no agent]

### Step 5: Commit, Push, and Create PR
[Human action: git add, commit, push, gh pr create]
!!! tip "Automation"
    `/commit-commands:commit-push-pr`

## Workflow Variants
## PR Size Tiers
## Tips for Success
## Common Issues and Solutions

<details><summary>Appendix: Workflow Evolution</summary>
[Full version history — preserved]
</details>
```

Write the complete rewritten file. Target ~600-700 lines (down from 1168). Content preservation (BC-2, BC-8) takes priority over line count — both sets of 10 perspectives alone require ~350 lines.

Key content transformations:
- **Step headers**: Remove skill names from headers (e.g., "Step 1: Create Isolated Worktree Using `using-git-worktrees` Skill" → "Step 1: Create an Isolated Workspace")
- **Prerequisites table**: Removed entirely; each skill referenced in its step's admonition
- **Quick Reference table**: Removed; steps themselves serve this purpose
- **Current Template Versions**: Moved to contributing/index.md (Task 6)
- **Review perspective prompts**: Changed from literal `/pr-review-toolkit:review-pr <exact prompt>` to human-readable checklists with the same content. The exact prompt text is preserved for each perspective but introduced as "Check for:" instead of "Prompt:"
- **Examples A and B**: Consolidated into one "Example Walkthrough" section showing the flow without literal skill commands
- **"For Claude:" directives**: Removed from inline text; agent-specific instructions only in admonition blocks
- **Headless Mode**: Kept (practical workaround), moved to Tips section
- **Skill Reference Quick Guide** (lines 904-918): Removed — skills are now referenced in admonition callouts per step, making this redundant

**Step 2: Verify the rewrite preserves all content**

Run these verification checks:

```bash
# IMPORTANT-1: Count review perspectives — BOTH gates must have all 10
# Step 2.5 (plan review): Perspectives 1-10
# Step 4.5 (code review): Perspectives 1-10
# These are DIFFERENT sets (e.g., Step 2.5 P3 = "Architecture Boundary" vs Step 4.5 P3 = "Test Behavioral Quality")
# Total should be 20 perspective headings
grep -c "#### Perspective" docs/contributing/pr-workflow.md
# Expected: 20

# IMPORTANT-3: Count ALL rationale sections (10 distinct "Why" sections in current doc)
# Enumerate each one and verify present:
# 1. "Why first?" (Step 1)
# 2. "Why two stages?" (Step 2.5)
# 3. "Why rounds with multiple perspectives?" (Step 2.5)
# 4. "Why this perspective exists:" (Step 2.5 P1)
# 5. "Why two stages?" (Step 4.5 — different rationale from Step 2.5)
# 6. "Why 10 perspectives in parallel?" (Step 4.5)
# 7. "Why not fix in-PR?" (Step 4.5 filing issues)
# 8. "Why a skill instead of prose?" (Step 4.5 verification)
# 9. "Why this step exists:" (Step 4.75)
# 10. "Why no agent?" (Step 4.75)
grep -c "Why " docs/contributing/pr-workflow.md
# Expected: ≥10

# Verify no broken internal links
grep -oP '\[.*?\]\(.*?\)' docs/contributing/pr-workflow.md | head -20
```

**CRITICAL: Both perspective sets must be preserved separately.** Step 2.5 and Step 4.5 have similar but NOT identical perspectives. Do NOT consolidate them into a single shared set.

**Gate-specific perspective differences to preserve (from convergence Round 3 PP-7/PP-8):**

- **P7 (vLLM/SGLang):** Step 2.5 reviews "this plan" and catches 4 items including "prefill/decode pipeline errors"; Step 4.5 reviews "this diff" and catches 3 items (no "prefill/decode pipeline errors"). Step 2.5 includes "missing scheduling features that real servers have" — Step 4.5 drops this. Preserve both versions separately.
- **P7 vs P8 scope boundary:** P7 focuses on **single-instance inference serving** (batching, KV eviction, prefill/decode, scheduling — maps to `sim/`). P8 focuses on **multi-instance cluster coordination** (routing, snapshots, scaling, admission — maps to `sim/cluster/`). Do not merge items between P7 and P8.
- **P8 (Distributed Platform):** Step 2.5 has richer qualifiers ("under high request rates", "between instances", "at scale") and catches "admission control failures"; Step 4.5 is abbreviated. Preserve the qualifier differences.

**Content-level verification (addresses convergence Round 1 finding from 6 perspectives):**
```bash
# Verify domain-specific terms survive in BOTH Step 2.5 and Step 4.5:
# DES terms:
# NOTE: grep -c counts LINES with matches, not individual matches.
# Run these on the CURRENT file first to establish baselines, then verify the rewrite
# meets or exceeds the baseline. Current baselines (measured):
# DES: 3 lines, vLLM: 2 lines, Distributed: 4 lines, Security: 6 lines
grep -c "event ordering\|clock monotonicity\|stale signal\|heap priority\|work-conserving" docs/contributing/pr-workflow.md
# Expected: ≥3 (baseline); ideally higher if rewrite puts each term on its own line

# vLLM/SGLang terms:
grep -c "batching semantics\|KV cache eviction\|chunked prefill\|preemption policy" docs/contributing/pr-workflow.md
# Expected: ≥2 (baseline); both gates must have at least one line

# Distributed platform terms:
grep -c "multi-instance coordination\|routing load imbalance\|stale snapshot\|admission control\|horizontal scaling\|prefix-affinity" docs/contributing/pr-workflow.md
# Expected: ≥4 (baseline); both gates must have at least one line

# Security/robustness terms:
grep -c "NaN\|degenerate\|resource exhaustion\|panic paths" docs/contributing/pr-workflow.md
# Expected: ≥6 (baseline); both gates must have at least one line

# "Catches:" summaries preserved:
grep -c "Catches:" docs/contributing/pr-workflow.md
# Expected: ≥20 (one per perspective)
```

**Writing-plans admonition must reference prompt file:**
All `writing-plans` invocations within `!!! tip "Automation"` admonitions must use `@docs/contributing/templates/micro-plan-prompt.md` (not `micro-plan.md`). The human primary text references the human-readable template; the agent admonition references the prompt file.

**Step 3: Commit**

```bash
git add docs/contributing/pr-workflow.md
git commit -m "docs(contributing): rewrite pr-workflow.md for human-first readability (BC-1, BC-2, BC-4, BC-5)

- Steps describe human actions; skills in admonition callouts
- All 20 review perspective checklists preserved
- All rationale sections preserved with PR/issue references
- Consolidated from 1168 to ~600-700 lines
- Workflow Evolution appendix in collapsed details block

Fixes #464 (Phase 1: pr-workflow.md)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Moderate Rewrite of hypothesis.md

**Contracts Implemented:** BC-1, BC-2, BC-4, BC-5, BC-7, BC-8

**Files:**
- Modify: `docs/contributing/hypothesis.md`

**Step 1: Rewrite hypothesis.md**

The hypothesis doc is already better balanced (55:45 human:agent). Changes needed:

1. **Prerequisites table** → Replace with inline admonition callouts per step (same pattern as Task 1)
2. **Quick Reference table** → Replace command column with human actions; keep as a summary table
3. **Step 0**: Already has manual alternative inline — promote manual to primary text
4. **Steps 2, 5, 8**: Change from "Primary mechanism (Claude Code): `/convergence-review ...`" to human action primary with automation admonition
5. **Step 3**: Rewrite "For Claude" directive ("Do not say 'I'll proceed unless you stop me'") as universal process guidance: "Wait for explicit approval before proceeding. Do not assume silence is consent."
6. **Step 7**: Keep as-is (already human-readable)
7. **Step 9**: Rewrite "For Claude" block ("This is NOT an agent pass. Stop, think critically...") as human-first: "Review each dimension yourself using critical thinking — do not delegate to automated tools." Agent-specific framing moved to admonition if needed.
8. **Step 10**: Same admonition pattern — human action primary
9. **Review perspective checklists**: Already human-readable (checklist format). Keep as-is.
10. **All other sections**: Already well-written for humans. Keep as-is.

Specific transformations:
- Remove Prerequisites table (3 rows)
- Quick Reference: change "Action" column from skill invocations to human actions
- Step 0: "Create a git worktree for the experiment" + automation admonition
- Steps 2, 5: "Review using the [N] perspectives below, applying the convergence protocol" + automation admonition
- Step 8: Same pattern
- Step 10: "Commit, push, and create a PR" + automation admonition

Target: ~500 lines (down from 743).

The heavy reduction comes from:
- Removing Prerequisites table (~10 lines)
- Condensing Quick Reference (~10 lines)
- Removing duplicate "Manual alternative:" lines (now primary text IS the manual path, ~20 lines)
- Condensing "Primary mechanism (Claude Code):" blocks into admonitions (~30 lines)

Most content is preserved — the review perspectives, issue taxonomy, promotion guidance, and parallel execution mode are already human-readable.

**Step 2: Verify content preservation**

```bash
# Count review perspectives (5 + 5 + 10 = 20)
grep -c "Reviewer\|Perspective" docs/contributing/hypothesis.md

# Count quality gate items
grep -c "\- \[ \]" docs/contributing/hypothesis.md
```

**Step 3: Commit**

```bash
git add docs/contributing/hypothesis.md
git commit -m "docs(contributing): rewrite hypothesis.md for human-first readability (BC-1, BC-4, BC-5)

- Manual workflow as primary text, skills in admonition callouts
- All 20 review perspective checklists preserved
- Prerequisites table replaced by inline callouts
- Condensed from 743 to ~500 lines

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Light Touch on design-process.md and macro-planning.md

**Contracts Implemented:** BC-4, BC-5

**Files:**
- Modify: `docs/contributing/design-process.md`
- Modify: `docs/contributing/macro-planning.md`

**Step 1: Update design-process.md**

Changes:
1. Replace Prerequisites table with admonition in Step 6:
   ```markdown
   ### Step 6: Convergence Review

   Run all 8 perspectives in parallel (or sequentially), applying the
   [convergence protocol](convergence.md). Fix CRITICAL and IMPORTANT
   findings, re-run until convergence.

   !!! tip "Automation"
       `/convergence-review design <design-doc-path>` dispatches all 8 perspectives
       and enforces convergence. See [Skills & Plugins](../guide/skills-and-plugins.md).
   ```
2. Remove the Prerequisites table at bottom
3. Step 6 text already says "Manual alternative: review against each perspective checklist below" — promote this to primary text

**Step 2: Update macro-planning.md**

Same pattern as design-process.md:
1. Replace Prerequisites table with admonition in Step 7
2. Promote manual review to primary text

**Step 3: Commit**

```bash
git add docs/contributing/design-process.md docs/contributing/macro-planning.md
git commit -m "docs(contributing): standardize skill callouts in design-process and macro-planning (BC-4)

- Replace Prerequisites tables with inline admonition callouts
- Manual review promoted to primary text

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Split micro-plan.md into Human Template + Agent Prompt

**Contracts Implemented:** BC-3

**Files:**
- Modify: `docs/contributing/templates/micro-plan.md`
- Create: `docs/contributing/templates/micro-plan-prompt.md`

**Step 1: Create the agent prompt file**

Move the current content of `micro-plan.md` (the full LLM system prompt) to `micro-plan-prompt.md`. This preserves the exact agent instructions for the writing-plans skill.

**Step 2: Rewrite micro-plan.md as human-readable template**

The human-readable version describes the output format — what sections to include, what goes in each section, and the quality criteria. Structure:

```markdown
# Micro Plan Template (Single-PR Implementation Plan)

This template defines the output format for a single-PR implementation plan.
Use this when planning any PR — from bug fixes to new features.

## Document Header

Every plan starts with:
- **Goal**: One sentence a non-contributor could understand
- **The problem today**: What's missing or broken (2-3 sentences)
- **What this PR adds**: Numbered list of concrete capabilities
- **Why this matters**: Connection to broader project vision
- **Architecture**: Technical approach (2-3 sentences)
- **Source**: Link to source of work (macro plan, issue, design doc)
- **Closes**: GitHub issue numbers (auto-closes on merge)

## Part 1: Design Validation (~120 lines)

### A) Executive Summary
[Description of what goes here — 5-10 lines, plain language]

### B) Behavioral Contracts
[Description: 3-15 named contracts in GIVEN/WHEN/THEN format]
[Quality gate: THEN clauses must describe observable behavior]

### C) Component Interaction
[Description: text diagram, API contracts, state changes]

### D) Deviation Log
[Description: table comparing micro plan vs source document]

### E) Review Guide
[Description: where to focus reviewer attention]

## Part 2: Executable Implementation

### F) Implementation Overview
[Description: files to create/modify, key decisions]

### G) Task Breakdown (6-12 tasks)
[Description: TDD format — test → fail → implement → pass → lint → commit]
[Each task: contracts implemented, files, complete code, exact commands]

### H) Test Strategy
[Description: contract-to-test mapping table]

### I) Risk Analysis
[Description: risks with likelihood/impact/mitigation]

## Part 3: Quality Assurance

### J) Sanity Checklist
[The full checklist — kept from current template]

## Appendix: File-Level Details
[Description: complete implementation for every file touched]
```

The human-readable version is ~200 lines (down from 670) and explains what each section is FOR, not how an LLM should produce it.

**Domain-specific guidance that must survive in the human template** (from DES/vLLM expert review):
- Test Strategy section (H) must include or cross-reference the invariant checklist from `docs/contributing/standards/invariants.md` (INV-1 through INV-8), not just say "contract-to-test mapping table"
- Appendix must include the "Event ordering" prompt (Priority? Timestamp? Secondary tie-breaking?) since this is DES-critical design guidance
- Sanity Checklist section (J) must preserve the full R1-R20 antipattern checklist (not condensed)

Include a note at the top:

```markdown
!!! note "For Claude Code users"
    The `writing-plans` skill generates plans from this template automatically.
    The agent prompt version is at
    [`micro-plan-prompt.md`](micro-plan-prompt.md).
```

**Step 3: Commit**

```bash
git add docs/contributing/templates/micro-plan.md docs/contributing/templates/micro-plan-prompt.md
git commit -m "docs(templates): split micro-plan into human template + agent prompt (BC-3)

- micro-plan.md: human-readable output format description (~200 lines)
- micro-plan-prompt.md: full agent preamble for writing-plans skill (670 lines)
- Docs site shows the human version; agents reference the prompt version

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Split macro-plan.md into Human Template + Agent Prompt

**Contracts Implemented:** BC-3

**Files:**
- Modify: `docs/contributing/templates/macro-plan.md`
- Create: `docs/contributing/templates/macro-plan-prompt.md`

**Step 1: Create the agent prompt file**

Move current `macro-plan.md` content to `macro-plan-prompt.md`.

**Step 2: Rewrite macro-plan.md as human-readable template**

Same pattern as Task 4. Structure:

```markdown
# Macro Plan Template (Multi-PR Feature Plan)

This template defines the output format for a macro-level implementation plan.
Use this when a feature spans 2+ PRs and requires a dependency DAG.

[Sections with descriptions of what goes in each — same approach as micro-plan]
```

Include the automation admonition note at top.

**Step 3: Commit**

```bash
git add docs/contributing/templates/macro-plan.md docs/contributing/templates/macro-plan-prompt.md
git commit -m "docs(templates): split macro-plan into human template + agent prompt (BC-3)

- macro-plan.md: human-readable output format description
- macro-plan-prompt.md: full agent preamble for macro planning skill

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Consistency Pass — Cross-References, Nav, CLAUDE.md, and Consumer Updates

**Contracts Implemented:** BC-4, BC-6

**Files:**
- Modify: `docs/contributing/index.md`
- Modify: `CONTRIBUTING.md`
- Modify: `.claude/skills/convergence-review/pr-prompts.md`
- Modify: `.claude/skills/convergence-review/design-prompts.md`
- Modify: `CLAUDE.md`

**Step 1: Update CONTRIBUTING.md (template path references)**

CONTRIBUTING.md has 8 template references. Enumerate ALL and specify disposition:

| Line | Content | Reference Type | Action |
|------|---------|---------------|--------|
| 161 | `@docs/contributing/templates/micro-plan.md` in writing-plans invocation | Skill `@` ref | Change to `micro-plan-prompt.md` |
| 176 | `@docs/contributing/templates/micro-plan.md` in writing-plans invocation | Skill `@` ref | Change to `micro-plan-prompt.md` |
| 195 | `following [docs/.../macro-plan.md](...)` | Human markdown link | **Keep as-is** — human readers need the human template |
| 212 | `following docs/.../micro-plan.md` in "Without Claude Code" | Human text ref | **Keep as-is** — human manual path |
| 219 | `[docs/.../macro-plan.md](...)` in "Without Claude Code" | Human markdown link | **Keep as-is** — human manual path |
| 291 | `via docs/.../micro-plan.md` in Subsystem Module recipe | Human text ref | **Keep as-is** — human extension recipe |
| 378 | `docs/contributing/templates/micro-plan.md` in Key References table | Human reference | **Keep as-is** — add note about prompt companion |
| 379 | `docs/contributing/templates/macro-plan.md` in Key References table | Human reference | **Keep as-is** — add note about prompt companion |

**Rule:** Only lines with skill `@` invocations (161, 176) change to prompt versions. All human-facing links (195, 212, 219, 291, 378, 379) stay pointing to the human-readable template.

Also standardize skill references in "Bug Fix / Small Change", "New Policy or Extension", and "New Feature" sections. **Note: CONTRIBUTING.md is rendered on GitHub (not MkDocs), so use GitHub-compatible formatting (blockquotes with bold labels) instead of MkDocs admonitions (`!!! tip`).** Example:

```markdown
> **Automation:** `/superpowers:using-git-worktrees fix-<name>` —
> see [Skills & Plugins](docs/guide/skills-and-plugins.md).
```

**Step 2: Update convergence-review pr-prompts.md (template reference)**

In `.claude/skills/convergence-review/pr-prompts.md` line 86, update:
```
Verify all sections from `docs/contributing/templates/micro-plan.md` are present
```
to:
```
Verify all sections from `docs/contributing/templates/micro-plan-prompt.md` are present
```

**Step 3: Update convergence-review design-prompts.md (template reference)**

In `.claude/skills/convergence-review/design-prompts.md` line 11, update:
```
Macro plan template: docs/contributing/templates/macro-plan.md
```
to:
```
Macro plan template: docs/contributing/templates/macro-plan-prompt.md
```

Also update the canonical-source header note in `pr-prompts.md` to acknowledge format divergence:
```
Canonical source: `docs/contributing/pr-workflow.md` (v3.0). After the human-first rewrite,
pr-workflow.md contains the same checklist content in human-readable format; this file
preserves the agent dispatch format. Content is aligned; format differs intentionally.
```

**Step 4: Update contributing/index.md**

Add the Current Template Versions info (moved from pr-workflow.md). Update the Templates table to note the template/prompt split:

```markdown
## Templates

| Template | Purpose | Agent Prompt |
|----------|---------|--------------|
| [Design Guidelines](templates/design-guidelines.md) | DES foundations, module architecture | N/A (reference material) |
| [Macro Plan](templates/macro-plan.md) | Multi-PR feature decomposition | [`macro-plan-prompt.md`](templates/macro-plan-prompt.md) |
| [Micro Plan](templates/micro-plan.md) | Single-PR implementation with TDD tasks | [`micro-plan-prompt.md`](templates/micro-plan-prompt.md) |
| [Hypothesis](templates/hypothesis.md) | Experiment FINDINGS.md structure | N/A (template is audience-neutral) |
```

**Step 5: Update CLAUDE.md**

Three update zones:

**(a) Design Principles section (lines 54-55):** Update "Written per" references to note prompt companions:
```markdown
- **Macro plans**: Written per `docs/contributing/templates/macro-plan.md` (human template; agent prompt: `macro-plan-prompt.md`)
- **Micro plans**: Written per `docs/contributing/templates/micro-plan.md` (human template; agent prompt: `micro-plan-prompt.md`)
```

**(b) File Organization tree:** Expand `templates/` to list files:
```
│   ├── templates/         # Artifact templates + agent prompts
│       ├── design-guidelines.md  # DES foundations, module architecture
│       ├── macro-plan.md         # Multi-PR template (human-readable)
│       ├── macro-plan-prompt.md  # Agent preamble for macro planning
│       ├── micro-plan.md         # Single-PR template (human-readable)
│       ├── micro-plan-prompt.md  # Agent preamble for writing-plans skill
│       └── hypothesis.md         # Experiment FINDINGS.md template
```

**(c) Project Governance Documents > Templates:** Reference prompt files:
```markdown
- `docs/contributing/templates/micro-plan.md`: Human-readable template for single-PR planning. **Agent prompt:** `micro-plan-prompt.md`
- `docs/contributing/templates/macro-plan.md`: Human-readable template for multi-PR planning. **Agent prompt:** `macro-plan-prompt.md`
```

**Step 6: Verify MkDocs build**

```bash
mkdocs build --strict 2>&1 | tail -20
```

Expected: Build succeeds with zero errors. Note: `--strict` does NOT warn about non-nav files when `nav:` is explicitly defined (MkDocs >= 1.4), so the prompt files will build without warnings.

**Step 7: Verify template path updates**

```bash
# Verify CONTRIBUTING.md skill invocations reference prompt files
grep "micro-plan-prompt.md" CONTRIBUTING.md
# Expected: 2 matches (lines 161, 176)

# Verify convergence-review skills reference prompt files
grep "micro-plan-prompt.md" .claude/skills/convergence-review/pr-prompts.md
grep "macro-plan-prompt.md" .claude/skills/convergence-review/design-prompts.md

# Verify human-facing refs still point to human templates
grep "micro-plan.md" CONTRIBUTING.md | grep -v "prompt"
# Expected: lines 212, 291, 378 still reference micro-plan.md (not prompt)
```

**Step 8: Commit**

```bash
git add docs/contributing/index.md CONTRIBUTING.md .claude/skills/convergence-review/pr-prompts.md .claude/skills/convergence-review/design-prompts.md CLAUDE.md
git commit -m "docs: update cross-references and template paths for prompt split (BC-4, BC-6)

- CONTRIBUTING.md: only skill @invocations → *-prompt.md; human links unchanged
- convergence-review pr-prompts.md + design-prompts.md: template refs → prompt versions
- contributing/index.md: template table with agent prompt column
- CLAUDE.md: 'Written per' refs, file org tree, governance docs updated
- MkDocs build verified clean

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 1 | Manual | Read each step in pr-workflow.md: human action is primary text |
| BC-2 | Task 1 | Grep | `grep -c "Why " pr-workflow.md` ≥ 10 — all rationale sections enumerated in Task 1 Step 2 |
| BC-3 | Tasks 4, 5 | Manual | micro-plan.md opens with human-readable description, not "You are operating" |
| BC-4 | Tasks 1-3 | Grep | `grep -c '!!! tip "Automation"' docs/contributing/*.md` — all skill refs use admonitions |
| BC-5 | Tasks 1-3 | Manual | Read each doc without Claude Code context — workflow is complete |
| BC-6 | Task 6 | Build | `mkdocs build --strict` — zero errors |
| BC-7 | Tasks 1, 2 | Manual | Step count and ordering matches original |
| BC-8 | Tasks 1, 2 | Grep | `grep -c "Perspective\|Reviewer" pr-workflow.md hypothesis.md` — counts match original |

No Go tests needed — this is a docs-only PR. No golden dataset changes.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Accidentally dropping a review perspective checklist item | Medium | High | Grep-based verification of perspective counts before and after | Tasks 1, 2 |
| Breaking cross-references between docs | Medium | Medium | `mkdocs build --strict` catches broken links | Task 6 |
| Agent prompt files not discoverable for writing-plans skill | Low | High | Admonition in human template links to prompt file; CLAUDE.md references both | Tasks 4, 5, 6 |
| pr-workflow.md rewrite too aggressive — loses nuance | Low | Medium | Preserve all rationale sections verbatim; review diff carefully | Task 1 |
| MkDocs admonition syntax errors | Low | Low | Build verification catches rendering issues | Task 6 |
| Template split breaks skill invocations (CRITICAL from review) | High | High | Task 6 explicitly updates CONTRIBUTING.md, pr-workflow.md admonitions, and convergence-review skill to reference `*-prompt.md` files | Tasks 1, 4, 5, 6 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — straightforward doc rewrites
- [x] No feature creep beyond PR scope — presentation only, no process changes
- [x] No unexercised flags or interfaces — N/A (docs-only)
- [x] No partial implementations — each task produces complete, reviewable docs
- [x] No breaking changes without explicit contract updates — no behavior changes
- [x] No hidden global state impact — N/A (docs-only)
- [x] All new code will pass golangci-lint — N/A (no Go code)
- [x] CLAUDE.md updated: file organization tree updated for new prompt files
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY: source-of-truth for PR workflow remains pr-workflow.md
- [x] Deviation log reviewed — all entries documented and justified (2 simplification/correction + 5 additions/scope changes)
- [x] Each task produces working, testable documentation
- [x] Task dependencies are correctly ordered (Tasks 1-3 independent; Tasks 4-5 independent; Task 6 depends on all)
- [x] All contracts mapped to specific tasks

**Antipattern rules:** N/A — docs-only PR. No Go code changes.

---

## Appendix: File-Level Implementation Details

### File: `docs/contributing/templates/micro-plan-prompt.md`

**Purpose:** Preserve the current `micro-plan.md` agent preamble content for the `writing-plans` skill.

**Content:** Exact copy of current `micro-plan.md` (670 lines). No modifications needed — the content is correct as an agent prompt.

### File: `docs/contributing/templates/macro-plan-prompt.md`

**Purpose:** Preserve the current `macro-plan.md` agent preamble content.

**Content:** Exact copy of current `macro-plan.md` (529 lines). No modifications needed.

### File: `docs/contributing/templates/micro-plan.md` (rewritten)

**Purpose:** Human-readable output format description for single-PR implementation plans.

**Key sections:**
- Document header format (with examples)
- Part 1: Design Validation (sections A-E with descriptions of what goes in each)
- Part 2: Executable Implementation (sections F-I with descriptions)
- Part 3: Quality Assurance (section J — full sanity checklist preserved)
- Appendix description
- Admonition linking to agent prompt companion file

Target: ~200 lines.

### File: `docs/contributing/templates/macro-plan.md` (rewritten)

**Purpose:** Human-readable output format description for multi-PR feature plans.

**Key sections:** Mirror the structure of the current template but with human-readable section descriptions instead of LLM directives.

Target: ~150 lines.

### File: `docs/contributing/pr-workflow.md` (rewritten)

**Purpose:** End-to-end PR development workflow guide, human-first.

**Key structural changes:**
1. Remove: Prerequisites table, Quick Reference table, Current Template Versions
2. Transform: Each step heading names a human action; skill in admonition
3. Keep: Overview ASCII diagram, all review perspectives, all rationale sections, PR Size Tiers, Tips, Common Issues
4. Condense: Merge Examples A and B into one walkthrough
5. Collapse: Workflow Evolution appendix into `<details>` block

Target: ~600-700 lines (content preservation over line count).

### File: `docs/contributing/hypothesis.md` (rewritten)

**Purpose:** End-to-end hypothesis experiment process, human-first.

**Key structural changes:**
1. Remove: Prerequisites table
2. Transform: Quick Reference actions to human descriptions; steps use admonition pattern
3. Keep: All review perspective checklists, quality gates, issue taxonomy, promotion guidance, parallel mode

Target: ~500 lines.
