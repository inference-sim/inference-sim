# pr-docs Gate for Convergence Review — Implementation Plan

**Goal:** Add a `pr-docs` gate type to the convergence-review skill with 7 documentation-specific review perspectives, eliminating wasted DES/vLLM/Performance/Security agent invocations for docs-only PRs.

**The problem today:** When a docs-only PR invokes `/convergence-review pr-code`, 4 of the 10 perspectives (DES Expert, vLLM Expert, Performance, Distributed Platform) return SUGGESTION-only findings with boilerplate "no Go code changes, nothing to review from this perspective" output. This wastes ~4 agent invocations per convergence round and obscures actionable findings with noise.

**What this PR adds:**
1. `pr-docs` gate type recognized by the convergence-review skill with diff-anchored state management (same as `pr-code`)
2. 7 docs-tailored perspectives in `pr-prompts.md` Section C: Substance & Accuracy, Cross-Document Consistency, Canonical Source Integrity, Completeness, Structural Validation (direct), Getting-Started Experience, DRY Compliance
3. PD-5 (Structural Validation) executes directly (no agent) — same pattern as PP-5 in `pr-plan` gates
4. Gate type table row with all 7 required columns populated

**Why this matters:** Docs-only PRs are a common pattern in BLIS (standards hardening, recipe updates, workflow revisions). The existing 10-perspective `pr-code` gate applies a wrong lens — DES and vLLM expertise is irrelevant for `.md` file changes. The `pr-docs` gate applies 7 targeted lenses that catch real docs bugs: stale cross-references, working-copy drift, count staleness, DRY violations, and contributor journey gaps.

**Architecture:** Pure skill-layer change. Two files edited: `SKILL.md` (gate type table + Phase A exception + frontmatter) and `pr-prompts.md` (Section C with 7 prompts). No Go code, no new interfaces, no CLI changes. The `pr-docs` gate inherits the diff-anchored state machine from `pr-code` with a different perspective set.

**Source:** GitHub issue [inference-sim/inference-sim#543](https://github.com/inference-sim/inference-sim/issues/543)

**Closes:** Fixes #543

**Behavioral Contracts:** See Part 1, Section B below.

---

## Phase 0: Component Context

**Building block modified:** `convergence-review` skill (`SKILL.md` + `pr-prompts.md`)

**Adjacent components:**
- `pr-workflow.md` — references `pr-code` gate; not modified in this PR (out of scope per issue AC)
- `design-prompts.md` — co-owned by convergence-review skill; not modified (no new design/macro-plan perspectives)
- `hypothesis-experiment/review-prompts.md` — not modified

**Invariants touched:** None (no Go code changes; the skill invariants SK-INV-1 through SK-INV-6 are not modified — they apply to all gates equally)

**Construction Site Audit:** N/A — no Go structs added or modified. The "construction sites" equivalent for skills are the places that enumerate valid gate types:
1. SKILL.md gate type table (Section "Gate Types") — must add `pr-docs` row
2. SKILL.md frontmatter description — must update count from 7 to 8 and add `pr-docs (7)` to the list
3. SKILL.md Phase A Step 3 exception — must extend to cover `pr-docs` alongside `pr-plan`
4. SKILL.md "Integration with Other Skills" — must add `pr-docs` invocation example

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds one new gate type (`pr-docs`) to an existing skill that already supports 7 gate types. The implementation is purely additive: one new row in the gate type table, one new section in the prompts file, and two minor updates to existing prose in SKILL.md (frontmatter description and Phase A exception). No state machine logic changes — `pr-docs` uses the existing diff-anchored path identically to `pr-code`, just with 7 different perspectives and PD-5 executing directly (same as PP-5 in `pr-plan`). The skill's Phase A dispatch loop, convergence protocol, and Phase B fix loop are unchanged.

**Deviation flags:** None. Issue spec is unambiguous.

### B) Behavioral Contracts

BC-1: Gate type recognized
- GIVEN the convergence-review skill is invoked with `pr-docs` as the gate type
- WHEN the skill parses and validates the gate argument
- THEN `pr-docs` is accepted as a valid gate type (not rejected with "unknown gate type" error)
- MECHANISM: The `pr-docs` row is present in the SKILL.md gate type table

BC-2: Docs-specific perspectives eliminate irrelevant agent invocations
- GIVEN a `pr-docs` gate is dispatched on a docs-only diff
- WHEN the 7 perspectives are applied
- THEN no DES Expert, vLLM Expert, Performance, Distributed Platform, or Security perspectives are invoked — only the 7 docs-tailored perspectives (Substance & Accuracy, Cross-Document Consistency, Canonical Source Integrity, Completeness, Structural Validation, Getting-Started Experience, DRY Compliance) run

BC-3: Diff-anchored state (same as pr-code)
- GIVEN a `pr-docs` review is initiated
- WHEN the state file is created
- THEN the gate is diff-anchored: state keyed by `pr-docs-<branch-name>`, commit anchored by `git rev-parse HEAD`, artifact is `git diff HEAD` output

BC-4: Gate table row complete with all 7 required columns
- GIVEN the SKILL.md gate table
- WHEN the `pr-docs` row is inspected
- THEN all 7 required columns are declared and non-empty: Gate, Anchor, Artifact, Triage Scope, Verification, Perspectives, Prompts Source

BC-5: PD-5 executes directly (no agent)
- GIVEN the Phase A dispatch for a `pr-docs` gate
- WHEN the agent prompt is assembled
- THEN the agent receives 6 prompts (PD-1 through PD-4, PD-6, PD-7); PD-5 Structural Validation is performed directly by the skill, not delegated to the agent — matching the PP-5 pattern for `pr-plan` gates

BC-6: All PD prompts use standard citation-requirement footer
- GIVEN pr-prompts.md Section C
- WHEN each of the 7 PD-n prompts is examined
- THEN each prompt (except PD-5, which is direct) ends with the standard citation-requirement footer: Severity/Location/Issue/Expected fields required, "Findings without a specific location will be DISCARDED as unverifiable", and the Report format — consistent with Section A and Section B prompts

### C) Component Interaction

```
User invocation
    │
    ▼
convergence-review SKILL.md
    │ parse gate type → "pr-docs"
    │ look up in gate table
    ▼
Gate type table (SKILL.md)
    │ Anchor: Diff → git diff HEAD
    │ Triage Scope: File-change heuristic
    │ Verification: Link check
    │ Perspectives: 7
    │ Prompts Source: pr-prompts.md Section C
    ▼
Phase A dispatch
    │ PD-5 → direct evaluation (no agent)
    │ PD-1,2,3,4,6,7 → assembled into single foreground agent call
    ▼
pr-prompts.md Section C
    │ PD-1: Substance & Accuracy
    │ PD-2: Cross-Document Consistency
    │ PD-3: Canonical Source Integrity
    │ PD-4: Completeness
    │ (PD-5 direct)
    │ PD-6: Getting-Started Experience
    │ PD-7: DRY Compliance
    ▼
State file: .claude/convergence-state/pr-docs-<branch>.json
    (same schema as pr-code, diff-anchored)
```

Boundary: SKILL.md is the dispatch contract. `pr-prompts.md` is the perspective content. The state machine in SKILL.md Phases A/B is unchanged — `pr-docs` uses the existing diff-anchored path.

### D) Deviation Log

**Verbatim Acceptance Criteria from issue #543:**
> - [ ] pr-docs recognized as valid gate type in convergence-review SKILL.md
> - [ ] 7 perspective prompts defined in pr-prompts.md Section C
> - [ ] Artifact = git diff (same as pr-code)
> - [ ] Documented in the gate type table

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue: "PD-5: Structural Validation (direct, no agent)" | SKILL.md Phase A exception extended from `pr-plan`-only to also cover `pr-docs` | CLARIFICATION: Issue implies this but doesn't specify the SKILL.md prose change explicitly. The PP-5 exception in SKILL.md must be extended for PD-5 to work correctly. |
| Issue: no mention of updating SKILL.md frontmatter description | Frontmatter updated: "7 gate types" → "8 gate types", `pr-docs (7)` added to gate list | CORRECTION: The existing frontmatter would be factually wrong after adding a gate — must update to maintain accuracy |
| Issue: no mention of integration example | Integration example added to SKILL.md "Integration with Other Skills" section | ADDITION: All 7 existing gate types have examples in that section; omitting `pr-docs` would leave the section incomplete |
| Issue: no mention of artifact context payload update | Phase A Step 3 artifact payload list updated from `(pr-code)` to `(pr-code, pr-docs)` | CORRECTION: Without this, the dispatch specification omits how to build the `pr-docs` artifact even though the gate table says "Diff" — a functional gap in the skill spec |
| Issue: "pr-docs recognized as valid gate type in convergence-review SKILL.md" | pr-workflow.md NOT updated | SCOPE_CHANGE: Issue AC covers only SKILL.md and pr-prompts.md. pr-workflow.md Step 4.5 automation tip currently shows only `pr-code`; updating it is a separate concern. A follow-up note is added to the PR description. |

### E) Review Guide

**Tricky part:** PD-5 must be marked "perform directly, no agent" consistently in both SKILL.md (Phase A exception) and pr-prompts.md (PD-5 section header annotation). If either is missing, the agent will receive PD-5 as a normal prompt and duplicate the direct evaluation.

**Scrutinize:** (1) Gate table row — all 7 columns populated correctly. (2) Phase A exception update — ensures `pr-docs` is listed alongside `pr-plan` for the PD-5/PP-5 direct evaluation. (3) Each of the 6 agent-dispatched PD prompts — standard citation footer present exactly.

**Safe to skim:** The PD prompt content itself — the perspectives are derived directly from the issue spec and follow the exact same structural pattern as Section B prompts.

**Known debt:** `pr-workflow.md` Step 4.5 still shows only `/convergence-review pr-code`; a future PR should add a note directing docs-only PRs to `/convergence-review pr-docs`.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Change |
|------|--------|
| `.claude/skills/convergence-review/SKILL.md` | (1) Frontmatter: 7→8 gate types, add `pr-docs (7)`. (2) Gate table: add `pr-docs` row with all 7 columns. (3) Phase A Step 3 exception: extend from `pr-plan`-only to include `pr-docs`. (4) Integration section: add `pr-docs` example. |
| `.claude/skills/convergence-review/pr-prompts.md` | (1) Header: update perspective count from 20 to 27. (2) Add Section C with PD-1 through PD-7 prompts. |

No dead code. No new files created. No Go changes.

### G) Task Breakdown

#### Task 1: Update SKILL.md — gate table, frontmatter, Phase A exception, integration example (BC-1, BC-3, BC-4, BC-5)

**Files:** modify `.claude/skills/convergence-review/SKILL.md`

**Pre-verify (read before edit):** Confirmed current state:
- Frontmatter line 3: `description: Dispatch parallel review perspectives and enforce convergence via persistent state file and automatic Phase A/B loop. Supports 7 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).`
- Gate table: 7 rows, last row ends at `h-findings`
- Phase A Step 3 exception (around line 193): `"For pr-plan gates, the agent receives 9 prompts (PP-1 through PP-4, PP-6 through PP-10)"`
- Integration section near bottom: 5 examples (`/convergence-review design`, `macro-plan`, `pr-plan`, `pr-code`, `h-design`, `h-code`, `h-findings`)

**Edit 1 — frontmatter description:**
```
Old: Supports 7 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).
New: Supports 8 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), pr-docs (7), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).
```

**Edit 2 — gate table row (insert after `pr-code` row, before `h-design`):**
```markdown
| `pr-docs` | Diff | `git diff HEAD` output | File-change heuristic | Link check | 7 | [pr-prompts.md](pr-prompts.md) Section C |
```
Rationale: `pr-docs` is a PR-family gate and uses `pr-prompts.md`. It belongs adjacent to `pr-code` in the table, not after the hypothesis-family gates which use a different prompts file.

**Edit 3 — Phase A Step 3 exception:**
```
Old: For `pr-plan` gates, the agent receives 9 prompts (PP-1 through PP-4, PP-6 through PP-10); PP-5 findings from direct evaluation are added to the findings array in step 4 alongside sections parsed from the agent's output.
New: For `pr-plan` and `pr-docs` gates, the agent receives a reduced set of prompts — `pr-plan` omits PP-5 (agent receives PP-1–PP-4, PP-6–PP-10); `pr-docs` omits PD-5 (agent receives PD-1–PD-4, PD-6–PD-7). Direct evaluation findings for PP-5 and PD-5 are added to the findings array in step 4 alongside sections parsed from the agent's output.
```

**Edit 4 — Phase A Step 3 artifact context payload (~line 211 of SKILL.md):**
```
Old: - Diff-anchored gates (`pr-code`): paste `git diff HEAD` output.
New: - Diff-anchored gates (`pr-code`, `pr-docs`): paste `git diff HEAD` output.
```

**Edit 5 — State file findings array fields table, perspective ID example (SKILL.md line 99):**
```
Old: | `perspective` | string | Perspective ID (e.g., "PC-1", "PP-3", "DD-2") |
New: | `perspective` | string | Perspective ID (e.g., "PC-1", "PP-3", "DD-2", "PD-1") |
```

**Edit 7 — Phase A Step 4a: extend direct-evaluation merge instruction to cover pr-docs/PD-5 (SKILL.md line 224):**
```
Old: For `pr-plan` gates, also include findings from the direct PP-5 evaluation in this same array.
New: For `pr-plan` gates, also include findings from the direct PP-5 evaluation in this same array. For `pr-docs` gates, also include findings from the direct PD-5 evaluation in this same array.
```

**Edit 6 — integration example, append note about docs-only scope (append new subsection at end of "Integration with Other Skills" section):**
The existing section has 4 subsections. Append:
```markdown
### From PR workflow — docs-only PRs (Step 4.5)

Use `pr-docs` instead of `pr-code` when the branch contains only documentation changes (.md files).
```
/convergence-review pr-docs
```
```
Note: Pre-verify count corrected — the integration section has 4 subsections (design process, macro planning, PR workflow, hypothesis experiment), not 5 as previously stated in the plan.

**Verify:**
```bash
# Confirm 8 gate rows in the gate type table (rows 29-36 after edit)
# Use awk to only count rows in the gate table section (between ## Gate Types and ## next section)
awk '/^## Gate Types/,/^---/' .claude/skills/convergence-review/SKILL.md | grep "^| \`" | wc -l
# Expected: 8

# Confirm pr-docs present
grep "pr-docs" .claude/skills/convergence-review/SKILL.md | wc -l
# Expected: >= 5 (frontmatter, table row, exception, artifact payload, integration)

# Confirm Phase A exception (Step 3) mentions both pr-plan and pr-docs
grep "pr-plan.*pr-docs\|pr-docs.*pr-plan" .claude/skills/convergence-review/SKILL.md
# Expected: 2 matches (Step 3 exception + Step 4a merge instruction)

# Confirm artifact payload updated
grep "Diff-anchored gates" .claude/skills/convergence-review/SKILL.md
# Expected line contains: (pr-code, pr-docs)

# Confirm line 99 perspective ID example updated
grep "PD-1" .claude/skills/convergence-review/SKILL.md
# Expected: 1 match (in the findings array fields table)
```

**Commit:** `feat(skills): add pr-docs gate type to SKILL.md table and infrastructure (BC-1,BC-3,BC-4,BC-5)`

---

#### Task 2: Add Section C to pr-prompts.md (BC-2, BC-5, BC-6)

**Files:** modify `.claude/skills/convergence-review/pr-prompts.md`

**Pre-verify:** Header line 3 currently: `"Contains exact prompts for the 20 PR review perspectives across plan review and code review gates."`

**Edit 1 — update header:**
```
Old: Contains exact prompts for the 20 PR review perspectives across plan review and code review gates.
New: Contains exact prompts for the 27 PR review perspectives across plan review, code review, and docs review gates.
```

**Edit 2 — append Section C after the last line of Section B (after PC-10):**

```markdown
---

## Section C: PR Docs Review (7 perspectives) — Step 4.5 (docs-only PRs)

### PD-1: Substance & Accuracy

```
Review this diff for substance and factual accuracy. Check for:
- Factual claims that are wrong (wrong issue numbers, wrong PR citations, wrong counts)
- File paths that don't exist or have been renamed
- Rule or invariant references that are stale (e.g., "R1-R20" when the current range is R1-R23)
- Count claims that don't match the actual count (e.g., "7 gate types" when there are 8)
- Procedure steps that describe behavior not matching the current implementation

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-2: Cross-Document Consistency

```
Review this diff for cross-document consistency. Check for:
- Working copies not updated when their canonical source was changed (check the source-of-truth map in docs/contributing/standards/principles.md)
- CLAUDE.md sections that reference file paths, rules, or counts that this diff changes
- Stale references in one document to content in another document that was modified
- Scope mismatch: did this change touch a canonical source but miss one or more of its working copies?

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-3: Canonical Source Integrity

```
Review this diff for canonical source integrity. Check for:
- A canonical source losing its "canonical" designation (e.g., a "canonical source" header removed)
- Contradictions introduced between two canonical sources (e.g., two files that should agree but now say different things)
- A working copy claiming authority it shouldn't have (e.g., "If this section diverges, THIS FILE is authoritative" when it should be the canonical source)
- A document that previously deferred to a canonical source now stating its own version of the truth

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-4: Completeness

```
Review this diff for completeness. Check for:
- Structured documents (tables, checklists, numbered lists) where a new entry was added but a parallel list elsewhere wasn't updated
- Count references that are now off by one (e.g., "8 perspectives" in the table of contents but only 7 in the list)
- Acceptance criteria or checklist items that are referenced but not fulfilled by the diff
- Link targets that are mentioned in the new/changed text but don't exist in the repo
- Section references (e.g., "see Section C") that now point to the wrong section number after a renumbering

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-5: Structural Validation (perform directly, no agent)

> **For Claude:** Perform these 4 checks directly. Do NOT dispatch an agent.

**Check 1 — Template Compliance:**
For each modified document, verify it follows its stated template or structure. If it's a table, all columns present. If it's a Gherkin-style contract, GIVEN/WHEN/THEN all present. If it's a checklist, all items have checkboxes.

**Check 2 — Internal Link Validity:**
For each internal link or file reference added or changed in the diff, verify the target file or section exists. Pay particular attention to links using the `[text](path)` format — check that the path resolves from the document's location.

**Check 3 — Source-of-Truth Map Consistency:**
Check `docs/contributing/standards/principles.md` for the source-of-truth map. For each canonical source modified in this diff, verify all listed working copies are also updated in this diff (or are provably unaffected).

**Check 4 — Parallel Structure Preservation:**
If the modified document uses parallel structure (e.g., all gate types have the same columns, all perspectives have the same footer), verify the new content preserves that parallel structure exactly.

### PD-6: Getting-Started Experience

```
Review this diff from the perspective of a new contributor. Simulate two journeys:
(1) A new contributor reading the docs to understand BLIS workflow for the first time
(2) An experienced contributor using updated docs to execute a specific workflow step

Check for:
- Instructions that are now inconsistent with each other (step A says one thing, step B contradicts it)
- A new contributor following instructions exactly and getting stuck because a prerequisite step was removed or changed
- Examples that illustrate the old behavior but haven't been updated to show the new behavior
- Terminology introduced without definition (jargon added without a "means X" explanation)

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### PD-7: DRY Compliance

```
Review this diff for DRY (Don't Repeat Yourself) violations. Focus on STRUCTURAL duplication — do not re-report cross-document sync issues already covered by PD-2 (Cross-Document Consistency). PD-2 checks "did this diff update all working copies?" — PD-7 checks "does this diff introduce new duplication or orphan a copy?"

Check for:
- Content duplicated in two or more places without a "canonical source" header on one of them (new duplication introduced by this diff)
- A fact stated in multiple documents that could diverge — is there a clear canonical source that others defer to? (structural question about the repo's doc architecture)
- New content added that is already stated elsewhere — should it be a link/reference instead of a copy?
- Content removed from one place but not removed from its copies, creating a shadow/orphaned version

DIFF:
<paste git diff output>

For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```
```

**Verify:**
```bash
# Count PD sections
grep "^### PD-" .claude/skills/convergence-review/pr-prompts.md | wc -l
# Expected: 7

# Confirm standard footer on agent-dispatched prompts (19 existing + 6 new PD prompts = 25)
grep -c "DISCARDED as unverifiable" .claude/skills/convergence-review/pr-prompts.md
# Expected: 25 (19 existing in A+B: 9 in Section A [PP-5 is direct, no footer] + 10 in Section B, plus 6 new PD prompts excluding PD-5 direct)

# Confirm Section C header present
grep "Section C" .claude/skills/convergence-review/pr-prompts.md
# Expected: 1 match
```

**Commit:** `feat(skills): add pr-prompts.md Section C — 7 pr-docs perspectives (BC-2,BC-5,BC-6)`

---

### H) Test Strategy

This PR has no Go code — the "tests" are structural verification checks (no unit tests possible for skill content).

| Contract | Task | Verification Type | Verification Step |
|----------|------|-------------------|-------------------|
| BC-1 (gate recognized) | Task 1 | Structural | `grep "pr-docs" SKILL.md` finds gate table row |
| BC-2 (docs perspectives only) | Task 2 | Structural | `grep "PD-" pr-prompts.md` shows 7 PD sections; no DES/vLLM/perf/security sections in Section C |
| BC-3 (diff-anchored) | Task 1 | Structural | Gate table row column 2 = "Diff", column 3 = "`git diff HEAD` output" |
| BC-4 (7-column table row) | Task 1 | Structural | Gate table row has 7 pipe-separated values |
| BC-5 (PD-5 direct) | Tasks 1+2 | Structural | SKILL.md exception includes `pr-docs`; PD-5 has "> For Claude: Perform directly" annotation |
| BC-6 (standard footer) | Task 2 | Structural | `grep -c "DISCARDED as unverifiable" pr-prompts.md` = 25 (19 existing + 6 new) |

No invariants (INV-1 through INV-12) are affected — no Go code changed.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| PD-5 exception update missed in SKILL.md | Low | High (agent would receive PD-5 as normal prompt, double-evaluating structural checks) | Verify step in Task 1 explicitly greps for `pr-plan.*pr-docs` in the exception | Task 1 |
| Gate table column count wrong (not 7) | Low | Medium (SK-INV-4 equivalent — structural inconsistency) | Post-edit verify: `grep "pr-docs" SKILL.md` and count pipes in the matching row | Task 1 |
| Section B `---` separator omitted before Section C | Low | Low (cosmetic) | Pre-verify that Section B ends with `---`; append separator before Section C header | Task 2 |
| Perspective count in pr-prompts.md header not updated | Low | Low (cosmetic inconsistency but technically a factual error) | Task 2 explicitly edits the header line | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — pure additive content, no new dispatch logic
- [x] No feature creep — strictly the 7 perspectives from the issue spec, no extras
- [x] No unexercised flags or interfaces — `pr-docs` is fully wired via existing gate table lookup
- [x] No partial implementations — both SKILL.md and pr-prompts.md are fully specified
- [x] No breaking changes — existing 7 gate types unchanged
- [x] No hidden global state impact — skill is stateless except for the convergence-state file, which is per-gate
- [x] All new code will pass golangci-lint — N/A (no Go code)
- [x] CLAUDE.md updated if needed — no new files, no CLI flags; CLAUDE.md unchanged
- [x] No stale references left in CLAUDE.md — not applicable
- [x] Documentation DRY — this PR modifies `.claude/skills/` which is not in the source-of-truth map for `docs/contributing/standards/` canonical sources. No working copies to update.
- [x] Deviation log reviewed — 4 entries, all justified
- [x] Each task produces complete, verifiable content
- [x] Task dependencies correctly ordered — Task 1 (SKILL.md infrastructure) can run in any order relative to Task 2 (prompts content). No dependency.
- [x] All contracts mapped to tasks — BC-1,3,4,5 → Task 1; BC-2,5,6 → Task 2
- [x] Construction site audit completed — no Go structs; SKILL.md enumeration sites all covered in Task 1

**Antipattern rules:** All R1-R23 are Go-specific (silent continue, map sorting, YAML typing, etc.) and not applicable to skill/markdown file edits. No Go code changed.

---

## Appendix: File-Level Implementation Details

### File: `.claude/skills/convergence-review/SKILL.md`

**Purpose:** The convergence-review skill dispatcher. Defines valid gate types, dispatch protocol, and state machine.

**Change 1 — Frontmatter (line 3):**
```
Before:
Supports 7 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).

After:
Supports 8 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), pr-docs (7), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).
```

**Change 2 — Gate table (insert after `pr-code` row, before `h-design`):**
```
| `pr-docs` | Diff | `git diff HEAD` output | File-change heuristic | Link check | 7 | [pr-prompts.md](pr-prompts.md) Section C |
```
`pr-docs` belongs adjacent to `pr-code` (PR family, same prompts file) — not after the hypothesis gates.

**Change 3 — Phase A Step 3 exception (current text):**
```
For `pr-plan` gates, the agent receives 9 prompts (PP-1 through PP-4, PP-6 through PP-10); PP-5 findings from direct evaluation are added to the findings array in step 4 alongside sections parsed from the agent's output. For all other gates, the agent receives all N prompts.
```
Replace with:
```
For `pr-plan` and `pr-docs` gates, the agent receives a reduced set of prompts — `pr-plan` omits PP-5 (agent receives PP-1–PP-4, PP-6–PP-10); `pr-docs` omits PD-5 (agent receives PD-1–PD-4, PD-6–PD-7). Direct evaluation findings for PP-5 and PD-5 are added to the findings array in step 4 alongside sections parsed from the agent's output. For all other gates, the agent receives all N prompts.
```

**Change 4 — Phase A Step 3 artifact context payload:**
```
Old: - Diff-anchored gates (`pr-code`): paste `git diff HEAD` output.
New: - Diff-anchored gates (`pr-code`, `pr-docs`): paste `git diff HEAD` output.
```

**Change 5 — Phase A Step 4a: extend direct-evaluation merge instruction (SKILL.md line 224):**
```
Old: For `pr-plan` gates, also include findings from the direct PP-5 evaluation in this same array.
New: For `pr-plan` gates, also include findings from the direct PP-5 evaluation in this same array. For `pr-docs` gates, also include findings from the direct PD-5 evaluation in this same array.
```

**Change 6 — State file findings array fields table, perspective ID example (line 99):**
```
Old: | `perspective` | string | Perspective ID (e.g., "PC-1", "PP-3", "DD-2") |
New: | `perspective` | string | Perspective ID (e.g., "PC-1", "PP-3", "DD-2", "PD-1") |
```

**Change 7 — Integration section (append new subsection at end):**
```markdown
### From PR workflow — docs-only PRs (Step 4.5)

Use `pr-docs` instead of `pr-code` when the branch contains only documentation changes (.md files).
```
/convergence-review pr-docs
```
```

---

### File: `.claude/skills/convergence-review/pr-prompts.md`

**Purpose:** Stores exact perspective prompts for all PR review gates. Sections A (plan), B (code), C (docs) map to `pr-plan`, `pr-code`, `pr-docs` gates.

**Change 1 — Header line 3:**
```
Before:
Contains exact prompts for the 20 PR review perspectives across plan review and code review gates.

After:
Contains exact prompts for the 27 PR review perspectives across plan review, code review, and docs review gates.
```

**Change 2 — Append Section C (full content in Task 2 above)**

The 7 perspectives follow the pattern of Section B exactly: each has a code-block prompt with a DIFF placeholder and the standard citation footer, except PD-5 which has the "perform directly, no agent" annotation matching PP-5.

**Key implementation notes:**
- PD-5 annotation must exactly match PP-5's format: `> **For Claude:** Perform these N checks directly. Do NOT dispatch an agent.`
- DIFF placeholder in each agent prompt must be: `` `<paste git diff output>` ``  (matching Section B)
- Section C separator line (`---`) before the `## Section C` header, matching the separator before `## Section B`
