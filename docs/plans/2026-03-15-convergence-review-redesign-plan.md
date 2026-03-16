# Convergence-Review Structural Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the convergence-review skill reliably enforce the fix-and-rerun loop by replacing prose instructions with a state-file-driven two-phase state machine.

**The problem today:** The convergence-review skill describes a mandatory re-run loop (dispatch perspectives, tally, fix, re-dispatch), but Claude frequently skips the re-run after fixing issues. The loop breaks because the algorithm is pseudocode in markdown with no persistent state, no structural enforcement, and contradictory re-invocation instructions. This was observed in #430, #390, and #413.

**What this PR adds:**
1. **State-file-driven loop** — a JSON state file at `.claude/convergence-state/` tracks round number, findings history, and convergence status, surviving context dilution and session boundaries.
2. **Two-phase execution** — Phase A (dispatch/tally/triage) and Phase B (fix/verify/rerun) with automatic transition. The loop only exits on convergence (0/0) or stall (round > 10).
3. **Confidence-tiered fix autonomy** — all CRITICALs get a fix plan presented to the user; IMPORTANT/SUGGESTION auto-fix. After the user responds to a CRITICAL fix plan, Phase B continues with remaining items.
4. **Gate-category-driven behavior** — a definitive gate-to-category table drives commit anchoring, triage scope, and verification gate selection. New gates are added by declaring their category.

**Why this matters:** The convergence protocol is the quality gate for all PRs, hypotheses, designs, and macro plans. If the skill doesn't enforce the loop, the protocol is aspirational, not real.

**Architecture:** This is a tooling-only PR. The primary change is a complete rewrite of `.claude/skills/convergence-review/SKILL.md` (229→~400 lines). Minor update to `docs/contributing/convergence.md` (one line replaced with expanded version). One line added to `.gitignore`. No Go code, no simulation changes.

**Source:** Design doc: `docs/plans/2026-03-15-convergence-review-redesign-design.md`. GitHub issue #430.

**Closes:** Fixes #430

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR rewrites the `convergence-review` skill's SKILL.md to replace its prose-described loop with a two-phase state machine backed by a persistent JSON state file. The skill is invoked by the PR workflow (Steps 2.5 and 4.5), hypothesis workflow (Steps 2, 5, 8), design process, and macro-plan process. It dispatches parallel review perspectives using prompt files (`pr-prompts.md`, `design-prompts.md`, `hypothesis-experiment/review-prompts.md`) which are **unchanged** by this PR.

Adjacent components: `docs/contributing/convergence.md` (canonical protocol rules, minor update), `.gitignore` (new entry), prompt files (unchanged).

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: State File Creation
- GIVEN the skill is invoked for a gate with no existing state file
- WHEN Phase A begins
- THEN a state file MUST be created at `.claude/convergence-state/<gate>-<artifact-id>.json` with round 1 and status `reviewing`
- MECHANISM: Phase A entry condition "no file exists" branch

BC-2: Automatic Phase B Entry
- GIVEN Phase A tallies in-scope CRITICAL > 0 or in-scope IMPORTANT > 0
- WHEN the tally is complete
- THEN the skill MUST enter Phase B immediately without pausing or asking for re-invocation
- MECHANISM: Phase A step 10 "Not converged" branch

BC-3: Automatic Phase A Re-entry
- GIVEN Phase B has fixed all items and updated the state file
- WHEN Phase B completes
- THEN the skill MUST re-enter Phase A immediately for the next round
- MECHANISM: Phase B step 8

BC-4: Convergence Exit
- GIVEN Phase A tallies in-scope CRITICAL == 0 and in-scope IMPORTANT == 0
- WHEN the convergence check passes
- THEN the skill MUST fix any remaining suggestions, delete the state file, and exit
- MECHANISM: Phase A step 10 "Converged" branch + Suggestion Cleanup section

BC-5: Stale State Invalidation
- GIVEN a state file exists with a commit hash that differs from current HEAD
- WHEN the skill is invoked (for file/diff-anchored gates)
- THEN the state MUST be reset to round 1 with a log message
- MECHANISM: Phase A entry condition "commit differs" branch (SK-INV-5). Uses full SHA (`git rev-parse HEAD`).

BC-6: CRITICAL Fix Plan with Resumption
- GIVEN a CRITICAL finding requires fixing in Phase B
- WHEN the skill processes the finding
- THEN the skill MUST present a fix plan and stop, waiting for user acknowledgment before applying. After the user responds (approve, alternative, file, or downgrade), Phase B MUST continue processing remaining items. The state file's `not-converged` status ensures the loop resumes even if the session is interrupted during the pause.
- MECHANISM: Phase B step 2 confidence-tiered autonomy

BC-7: Independent Tallying
- GIVEN reviewer agents return output with severity classifications
- WHEN the skill collects results
- THEN the skill MUST count findings independently from agent output text, never using agent-reported totals
- MECHANISM: Phase A step 4 + SK-INV-3

BC-8: Gate Table Completeness
- GIVEN the SKILL.md contains a gate-to-category table
- WHEN any gate type is referenced in the skill
- THEN every gate MUST appear in the table with all 7 columns (Gate, Anchor, Artifact, Triage Scope, Verification, Perspectives, Prompts Source)

BC-14: Suggestion Cleanup Verification Failure
- GIVEN suggestions are being fixed after convergence (0/0 CRITICAL/IMPORTANT)
- WHEN a suggestion fix causes the verification gate to fail
- THEN the skill MUST treat it as an unresolvable verification failure: ask the user whether to revert the suggestion fix and exit, or fix the verification issue and continue
- MECHANISM: Suggestion Cleanup step 4 verification gate

**Negative Contracts:**

BC-9: No Premature Exit
- GIVEN findings exist with in-scope CRITICAL > 0 or in-scope IMPORTANT > 0
- WHEN the skill is executing
- THEN the skill MUST NOT exit. The only exits are: converged (0/0), stalled (round > 10), or unresolvable verification failure
- MECHANISM: SK-INV-1

BC-10: No Re-invocation Section
- GIVEN the updated SKILL.md
- WHEN a reader looks for re-invocation instructions
- THEN the "After fixes / Re-invoke" section MUST NOT exist. The loop is automatic.

**Error Handling Contracts:**

BC-11: Corrupted State Recovery
- GIVEN a state file exists but cannot be parsed (corrupted JSON, missing fields, unsupported schema version)
- WHEN the skill loads state
- THEN the skill MUST log a warning, delete the corrupted file, and start fresh at round 1

BC-12: Empty Diff Handling
- GIVEN a diff-anchored gate (pr-code) is invoked
- WHEN `git diff HEAD` produces no output
- THEN the skill MUST emit a warning and skip dispatch (not produce vacuous reviews)
- NOTE: `git diff HEAD` includes staged and unstaged changes but not untracked files. Users should stage new files before invoking the skill. This is an accepted limitation.

BC-13: Consistency Check
- GIVEN the tally produces counts that conflict with the intended status
- WHEN the state file is about to be updated
- THEN the count-derived status MUST take precedence and a visible warning MUST be emitted
- MECHANISM: Phase A step 5 + SK-INV-4

### C) Component Interaction

```
User invokes /convergence-review <gate> [artifact] [--model]
        │
        ▼
┌─────────────────────────┐
│  SKILL.md (this PR)     │ ◄── reads prompts from:
│                         │     ├── pr-prompts.md (unchanged)
│  Phase A: Dispatch      │     ├── design-prompts.md (unchanged)
│  Phase B: Fix & Rerun   │     └── hypothesis-experiment/review-prompts.md (unchanged)
│                         │
│  Reads/writes state:    │
│  .claude/convergence-   │
│    state/<id>.json      │
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  convergence.md         │ ◄── canonical protocol (one line expanded: note about automation)
│  (docs/contributing/)   │
└─────────────────────────┘
```

State ownership: The state file is owned exclusively by the convergence-review skill. No other skill reads or writes it.

Extension friction: Adding a new gate type requires: (1) add row to gate table in SKILL.md with all 7 columns, (2) add perspective section to a prompts file. Two files, zero code changes.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Gate table has 5 columns (Gate, Anchor, Artifact, Triage Scope, Verification) | Gate table has 7 columns (adds Perspectives, Prompts Source) | ADDITION: The 2 extra columns are reference metadata carried over from the current skill's gate table. They improve implementer convenience without changing behavioral semantics. The 5 behavioral columns match the design doc exactly. |
| Design doc extension point says "all 5 columns" | Plan says "all 7 columns" | ADDITION: Consistent with the expanded table. The design doc's Section 6 should be updated in the same commit to say "all 7 columns." |
| Design doc uses `git rev-parse --short HEAD` | Plan uses `git rev-parse HEAD` (full SHA) | CORRECTION: Short hashes can collide in large repos. Full SHA eliminates false stale-state resumption. |
| Design doc triage parse-failure default: Fix-all | Plan triage parse-failure default: File-all | CORRECTION: Fix-all applies fixes to out-of-scope items, potentially changing unrelated code. File-all is safer — it tracks the findings as issues without applying risky fixes. |

### E) Review Guide

**The tricky part:** BC-6 (CRITICAL fix plan with resumption). The skill pauses output and waits for user input mid-Phase-B. After the user responds, the skill must continue Phase B from where it paused, not restart or exit. The state file's `not-converged` status ensures the loop resumes even across session boundaries — but within a session, the skill must simply continue processing remaining findings after each CRITICAL pause.

**What to scrutinize:** The Phase A/B transition language (BC-2 and BC-3). These are the contracts that prevent the loop from breaking. Verify the language is unambiguous and leaves no exit path.

**What's safe to skim:** The gate table content (carried over from the design doc with 2 metadata columns added). The convergence protocol section (copied from convergence.md).

**Known debt:** Prompt files reference "Task tool" in their dispatch pattern but SKILL.md uses "Agent tool" — terminology mismatch across the boundary. Prompt files are out of scope for this PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `.claude/skills/convergence-review/SKILL.md` — Complete rewrite (229→~400 lines)
- `docs/contributing/convergence.md` — One line replaced with expanded version (line 11)
- `.gitignore` — Add 1 entry

**Files to commit (already exist in worktree):**
- `docs/plans/2026-03-15-convergence-review-redesign-design.md` — Design doc
- `docs/plans/2026-03-15-convergence-review-redesign-plan.md` — This plan

**Key decisions:**
- No Go code. This is a markdown-only PR.
- Prompt files unchanged — dispatch mechanics are the same, only loop control changes.
- SKILL.md rewritten in one shot (not incrementally) to avoid fragile intermediate states.
- Commit anchor uses full SHA (`git rev-parse HEAD`) instead of short hash.
- Triage parse-failure defaults to File-all (safer than Fix-all for out-of-scope items).
- Round counter handling: Phase A step 2 increments the round ONLY when resuming from `not-converged` status. If resuming from `reviewing` status (interrupted mid-Phase-A), the round is NOT incremented — the same round is replayed. The increment is immediately persisted to the state file. Phase B transition banner shows "Round N+1" as preview text only.

### G) Task Breakdown

---

### Task 1: Add `.claude/convergence-state/` to `.gitignore`

**Contracts Implemented:** BC-1 (prerequisite — state directory must be gitignored)

**Files:**
- Modify: `.gitignore`

**Step 1: Add the gitignore entry**

In `.gitignore`, add after the last entry:

```
# Convergence review state (session-local, not committed)
.claude/convergence-state/
```

**Step 2: Verify**

Run: `grep -n "convergence-state" .gitignore`
Expected: Line with `.claude/convergence-state/`

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: gitignore convergence-review state directory

Prerequisite for #430: state files must be ignored before the skill writes them.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Update `docs/contributing/convergence.md`

**Contracts Implemented:** None directly (documentation update)

**Files:**
- Modify: `docs/contributing/convergence.md` (line 11)

**Step 1: Replace line 11 with expanded version**

Replace the existing executable-implementation callout (line 11) with:

```markdown
> **Executable implementation:** The `convergence-review` skill automates this protocol — dispatching perspectives, tallying findings independently, and enforcing the re-run gate via a persistent state file at `.claude/convergence-state/`. The skill automatically loops through Phase A (review/tally) and Phase B (fix/verify) until convergence or stall, with no manual re-invocation needed. Invoke with `/convergence-review <gate-type> [artifact-path] [--model opus|sonnet|haiku]` (default: `haiku`).
```

**Step 2: Verify**

Run: `grep -c "persistent state file" docs/contributing/convergence.md`
Expected: 1

**Step 3: Commit**

```bash
git add docs/contributing/convergence.md
git commit -m "docs(convergence): note state-file automation in protocol doc

Reference the persistent state file and automatic Phase A/B loop.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Write complete new SKILL.md

**Contracts Implemented:** BC-1 through BC-14 (all contracts)

**Files:**
- Rewrite: `.claude/skills/convergence-review/SKILL.md` (complete replacement)

**Step 1: Write the entire new SKILL.md**

Replace the entire contents of `.claude/skills/convergence-review/SKILL.md` with the complete new skill file. The file has these sections in order:

1. **Frontmatter** — Updated description mentioning state file and Phase A/B loop
2. **Title** — "Convergence Review Dispatcher"
3. **Model Selection** — Same as current, plus re-entry behavior (use state file value if `--model` omitted)
4. **Gate Types table** — 7 columns (Gate, Anchor, Artifact, Triage Scope, Verification, Perspectives, Prompts Source), 7 gates, plus anchor/triage/verification category descriptions
5. **State File** — Location, schema (illustrative), lifecycle, commit anchoring (full SHA), error handling, schema version, artifact ID collision note
6. **Convergence Protocol** — Hard rules, severity classification, behavioral invariants (SK-INV-1 through SK-INV-5)
7. **Phase A — Review and Tally** — Entry conditions (5 branches), empty-diff precondition, dispatch, independent tally, consistency check with visible warning, triage (gate-table-driven with batch prompt, parse-failure defaults to File-all), convergence check, state update (round incremented and immediately persisted), status banner, three-way branch
8. **Phase B — Fix and Rerun** — Confidence-tiered fix (CRITICALs: fix plan → stop → wait → user responds → continue remaining items; IMPORTANT/SUGGESTION: auto-fix), staging new files, gate-table-driven verification, state update, transition banner, immediate Phase A re-entry. No git commit. Multi-session caveat.
9. **Suggestion Cleanup on Convergence** — Fix suggestions, verify, delete state, exit
10. **Integration with Other Skills** — Usage examples for each workflow

**The complete SKILL.md structure** is specified in the Appendix below. The implementing agent should compose the full file from the structural specifications provided — each section lists the exact behavioral elements that must be present. Do not omit or improvise sections beyond what the Appendix specifies.

Key differences from current SKILL.md:
- Old "Dispatch Instructions" (Steps 1-5), "Collect and Tally", "Report", "Round Tracking", and "After fixes / Re-invoke" sections are **all deleted** (replaced by Phase A/B)
- New sections: State File, Behavioral Invariants, Phase A, Phase B, Suggestion Cleanup
- Gate table expanded from 4 to 7 columns
- Commit anchor uses full SHA (`git rev-parse HEAD`)
- Triage parse-failure defaults to File-all

**Step 2: Verify structural correctness**

Run each check:

```bash
# BC-8: Gate table has all 7 gates
grep -c "^\| \`" .claude/skills/convergence-review/SKILL.md
# Expected: 7

# BC-10: No re-invocation section
grep -c "Re-invoke" .claude/skills/convergence-review/SKILL.md
# Expected: 0

# SK-INV-1 through SK-INV-5 present
grep -c "SK-INV" .claude/skills/convergence-review/SKILL.md
# Expected: >= 5

# Phase A and Phase B sections present
grep -c "## Phase A" .claude/skills/convergence-review/SKILL.md
# Expected: 1
grep -c "## Phase B" .claude/skills/convergence-review/SKILL.md
# Expected: 1

# State file section present
grep -c "## State File" .claude/skills/convergence-review/SKILL.md
# Expected: 1

# Full SHA, not short
grep -c "rev-parse HEAD" .claude/skills/convergence-review/SKILL.md
# Expected: >= 1
grep -c "rev-parse --short" .claude/skills/convergence-review/SKILL.md
# Expected: 0

# File-all default for parse failure
grep -c "File-all" .claude/skills/convergence-review/SKILL.md
# Expected: >= 1
```

**Step 3: Commit**

```bash
git add .claude/skills/convergence-review/SKILL.md
git commit -m "feat(convergence-review): rewrite skill with state-file-driven loop (#430)

Replace prose-described convergence loop with two-phase state machine:
- Phase A: dispatch/tally/triage with persistent state file
- Phase B: confidence-tiered fix/verify/auto-rerun
- SK-INV-1-5: loop integrity, round monotonicity, tally independence,
  state-status consistency, stale invalidation
- Gate table expanded to 7 columns with category-driven behavior
- Full SHA commit anchoring (not short hash)
- Triage parse-failure defaults to File-all (safe for codebase)
- CRITICAL fix plan with 4 user options + resumption spec
- 'After fixes / Re-invoke' section deleted (S4 from #430)

Implements BC-1 through BC-14.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Update design doc extension point + commit docs

**Contracts Implemented:** Deviation log consistency

**Files:**
- Modify: `docs/plans/2026-03-15-convergence-review-redesign-design.md` (Section 6, extension point text)
- Commit: `docs/plans/2026-03-15-convergence-review-redesign-design.md`
- Commit: `docs/plans/2026-03-15-convergence-review-redesign-plan.md`

**Step 1: Update design doc for consistency with plan deviations**

In `docs/plans/2026-03-15-convergence-review-redesign-design.md`, make these updates:

1. **Section 1, gate-to-category table:** Add Perspectives and Prompts Source columns (matching the SKILL.md 7-column table).
2. **Section 1, line after table ("all 5 columns"):** Change to "all 7 columns."
3. **Section 1, commit anchoring text:** Change `git rev-parse --short HEAD` to `git rev-parse HEAD` (full SHA). Update all occurrences.
4. **Section 1, illustrative JSON:** Change `"commit": "a529ff4"` to a full 40-character SHA example (e.g., `"commit": "a529ff4e3b1c2d4e5f6a7b8c9d0e1f2a3b4c5d6e"`).
5. **Section 6, extension point text:** Change "all 5 columns" to "all 7 columns (Gate, Anchor, Artifact, Triage Scope, Verification, Perspectives, Prompts Source)."

**Step 2: Commit all docs**

```bash
git add docs/plans/2026-03-15-convergence-review-redesign-design.md
git add docs/plans/2026-03-15-convergence-review-redesign-plan.md
git commit -m "docs: design doc + implementation plan for convergence-review redesign (#430)

- Design doc: two-phase state machine, converged in 4 rounds of 8-perspective review
- Implementation plan: 4 tasks, 14 behavioral contracts
- Update design doc extension point to match 7-column gate table

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|-------------|
| BC-1 | Task 3 | Structural | State file section exists in SKILL.md |
| BC-2 | Task 3 | Structural | Phase A "Not converged → Phase B immediately" text exists |
| BC-3 | Task 3 | Structural | Phase B "Re-enter Phase A immediately" text exists |
| BC-4 | Task 3 | Structural | Suggestion cleanup section exists |
| BC-5 | Task 3 | Structural | Commit anchoring with `git rev-parse HEAD` (full SHA) |
| BC-6 | Task 3 | Structural | CRITICAL fix plan with 4 user options + "continue Phase B" resumption |
| BC-7 | Task 3 | Structural | SK-INV-3 present, "Never trust agent-reported totals" text |
| BC-8 | Task 3 | Structural | Gate table has 7 rows and 7 columns |
| BC-9 | Task 3 | Structural | SK-INV-1 present |
| BC-10 | Task 3 | Structural | "After fixes" / "Re-invoke" text absent |
| BC-11 | Task 3 | Structural | Error handling paragraph exists |
| BC-12 | Task 3 | Structural | Empty-diff precondition + untracked files note |
| BC-13 | Task 3 | Structural | SK-INV-4 with visible warning text exists |
| BC-14 | Task 3 | Structural | Suggestion cleanup verification failure handling exists |

This is a tooling-only PR (markdown files). Verification is structural — grep for required sections and text patterns. Task 3 Step 2 runs all structural checks.

No golden dataset updates needed. No Go code changes. No lint required.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| SKILL.md too large for context window | Medium | High | Keep to ~400 lines; Phase A/B self-contained | Task 3 |
| CRITICAL fix pause breaks loop | Medium | High | State file `not-converged` ensures resumption; explicit "continue Phase B" instruction | Task 3 |
| Untracked files missed by `git diff HEAD` | Low | Medium | Documented as accepted limitation; users should stage new files | Task 3 |
| Prompt files reference "Task tool" vs "Agent tool" | Low | Low | Out of scope; prompt files dispatch correctly regardless of terminology | N/A |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — markdown rewrite, not a framework
- [x] No feature creep beyond PR scope — prompt files unchanged, pr-workflow.md unchanged
- [x] No unexercised flags or interfaces — all gate table columns used by Phase A/B
- [x] No partial implementations — every Phase A step and Phase B step is complete
- [x] No breaking changes — `/convergence-review` invocation syntax unchanged
- [x] No hidden global state impact — state file is local and gitignored
- [x] CLAUDE.md update not needed — no new files/packages, no CLI flag changes
- [x] Documentation DRY — convergence.md updated, no other working copies affected
- [x] Deviation log reviewed — 4 deviations documented with rationale (2 additions, 2 corrections)
- [x] Each task produces working, testable output
- [x] Task dependencies correctly ordered (gitignore → convergence.md → SKILL.md → docs)
- [x] All contracts mapped to tasks
- [x] No golden dataset changes needed
- [x] Design doc extension point updated to match implementation (7 columns)

**Antipattern rules:** N/A for a markdown-only PR (no Go code). R1-R23 are Go-specific.

---

## Appendix: File-Level Implementation Details

### File: `.claude/skills/convergence-review/SKILL.md`

**Purpose:** The convergence-review skill — dispatches parallel review perspectives and enforces the convergence loop via persistent state.

**Complete content:** The implementing agent should write the file with these sections in order. Each section's content is fully specified in the design doc and the task descriptions above. The key behavioral elements that MUST be present:

1. **Frontmatter:** name=convergence-review, description mentioning state file + Phase A/B, argument-hint unchanged
2. **Model Selection:** `--model` flag with re-entry behavior (omitted = use state file value)
3. **Gate Table:** 7 columns, 7 gates, plus 3 category description blocks (anchor, triage, verification)
4. **State File:** Location `.claude/convergence-state/`, illustrative JSON schema with `schema_version`, lifecycle (5 items), commit anchoring using full SHA (`git rev-parse HEAD`), error handling (corrupted/unsupported → round 1), 24h staleness warning for context gates, artifact ID collision note
5. **Convergence Protocol:** 7 hard rules, severity table, SK-INV-1 through SK-INV-5 with visible warning on SK-INV-4 mismatch
6. **Phase A (10 steps):** parse args → load/create state (5 branches; round incremented and immediately persisted ONLY for `not-converged` resumption, NOT for `reviewing` resumption) → dispatch (empty-diff precondition) → tally independently → consistency check → triage (gate-table-driven, batch prompt, parse-failure defaults to File-all with warning) → convergence check → update state → banner → branch (converge/stall/Phase B)
7. **Phase B (8 steps):** list findings → confidence-tiered fix (CRITICALs: emit fix plan, STOP, wait for user response, then CONTINUE with remaining items; IMPORTANT/SUGGESTION: auto-fix) → fix in priority order → stage new files → verify per gate table → update state → banner → re-enter Phase A immediately
8. **Suggestion Cleanup:** fix suggestions → verify (if verification fails, ask user: revert suggestion fix or fix the issue) → delete state → exit
9. **Integration:** usage examples for design, macro-plan, pr-plan, pr-code, h-design, h-code, h-findings

### File: `docs/contributing/convergence.md`

**Purpose:** Canonical convergence protocol rules.

**Change:** Line 11 — replace the executable-implementation callout with expanded version mentioning persistent state file and automatic Phase A/B loop. One line replaced, content expanded. No protocol rule changes.

### File: `.gitignore`

**Purpose:** Git ignore patterns.

**Change:** Add `.claude/convergence-state/` entry at the end. One line added.

### File: `docs/plans/2026-03-15-convergence-review-redesign-design.md`

**Purpose:** Design doc for this PR.

**Change:** Update Section 6 extension point text from "all 5 columns" to "all 7 columns." Update gate-to-category table to add Perspectives and Prompts Source columns.
