# Express Lane Tier + Small PR Pre-Pass Skip Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce process overhead for trivial changes by adding an Express Lane tier that skips planning and review, and remove the redundant pre-pass stage for Small PRs.

**The problem today:** A 1-line typo fix goes through the full PR workflow (worktree → plan → review → implement → audit → commit), taking ~30 minutes of ceremony for a 30-second fix. Additionally, Small PRs run a holistic pre-pass that overlaps significantly with the first two convergence perspectives, adding review time without proportional safety benefit.

**What this PR adds:**
1. Express Lane tier — a new tier below Small for purely mechanical changes (≤3 lines), skipping plan, review, and human gate entirely
2. Pre-pass elimination for Small PRs — changes Small tier review from "single pre-pass" to "single convergence round (no pre-pass)"

**Why this matters:** Process right-sizing — matching ceremony to risk. The current workflow was designed for simulation engine changes where silent bugs corrupt results. Applying the same rigor to a typo fix creates friction without value.

**Architecture:** Docs-only change to `docs/contributing/pr-workflow.md`. The PR Size Tiers table gains a new row, the Small row's review columns change, the Rules section gains Express Lane constraints, and a few prose references to "two-stage" / "pre-pass" are qualified with tier-dependent notes.

**Source:** GitHub issue #673

**Closes:** Fixes #673

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds one row to the PR Size Tiers table (Express Lane) and modifies the Small row's review columns. It also updates surrounding prose that currently implies all tiers use the two-stage (pre-pass + convergence) structure. The changes are confined to `docs/contributing/pr-workflow.md` — no other canonical sources or working copies need updating because CONTRIBUTING.md and the convergence-review skill don't reference tier-level review details.

No deviations from the source issue.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Express Lane Tier Exists
- GIVEN a contributor reads the PR Size Tiers table
- WHEN they look for guidance on a ≤3-line mechanical change
- THEN they find an Express Lane tier with criteria, abbreviated process (implement → self-audit → commit), and no plan/review/human-gate requirements
- MECHANISM: New row in the PR Size Tiers table, before Small

BC-2: Express Lane Upgrade Rule
- GIVEN a contributor using the Express Lane tier
- WHEN their change grows beyond 3 lines, touches a canonical source, or introduces any behavioral change
- THEN the Rules section instructs them to stop and switch to Small tier
- MECHANISM: New bullet in the Rules section

BC-3: Small PR Pre-Pass Eliminated
- GIVEN a contributor with a Small-tier PR
- WHEN they reach Steps 2.5 and 4.5
- THEN the table tells them to run a single convergence round with no pre-pass (not "single pre-pass sufficient")
- MECHANISM: Updated text in the Small row's Plan Review and Code Review columns

BC-4: Medium/Large Unchanged
- GIVEN a contributor with a Medium or Large PR
- WHEN they reach Steps 2.5 and 4.5
- THEN the review process remains full two-stage (pre-pass + convergence)
- MECHANISM: Medium and Large rows remain unchanged

**Negative contracts:**

BC-5: Express Lane Cannot Be Used for Source-of-Truth Files
- GIVEN a change to any file listed in the source-of-truth map (rules.md, invariants.md, principles.md, etc.)
- WHEN the contributor checks Express Lane criteria
- THEN the criteria explicitly exclude changes to canonical sources
- MECHANISM: "not in any canonical source from the source-of-truth map" in criteria

BC-6: Prose References Qualified
- GIVEN a reader of the Overview, Example Walkthrough, or Step descriptions
- WHEN they encounter mentions of "two-stage" or "pre-pass" review
- THEN these references either note the tier-dependent behavior or describe the default (Medium/Large) process without implying it applies universally
- MECHANISM: Targeted edits to prose in the Overview section, Example Walkthrough, Step 2.5/4.5 body text, Step 4.75, and Mermaid diagram note

### C) Component Interaction

N/A — docs-only change with no code components.

### D) Deviation Log

No deviations from source document (issue #673).

The issue says "Update CLAUDE.md PR workflow summary if needed." After inspection, CLAUDE.md's PR Workflow section says only "Diligently follow the workflow in docs/contributing/pr-workflow.md" — it doesn't detail tiers or pre-pass behavior, so no CLAUDE.md update is needed.

### E) Review Guide

1. **The tricky part:** Making sure the Express Lane criteria are tight enough to prevent misuse (behavioral changes sneaking through) but loose enough to actually save time for genuine typo fixes.
2. **What to scrutinize:** BC-5 (source-of-truth exclusion) and BC-2 (upgrade rule) — these are the safety rails.
3. **What's safe to skim:** BC-4 (Medium/Large unchanged) and BC-6 (prose qualifications) — mechanical consistency edits.
4. **Known debt:** None.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `docs/contributing/pr-workflow.md` — all changes here (tiers table, rules, overview prose, example walkthrough)

**Key decisions:**
- Express Lane row goes ABOVE Small (ascending order of ceremony)
- Express Lane self-audit is reduced to 3 dimensions (correctness, determinism, consistency) per the issue, not the full 10
- The "Steps 1, 1.5, 2, 3, 4, 5 are always required" rule needs a carve-out for Express Lane
- Prose references to "two-stage" in Overview, Key Insights, Steps 2.5/4.5/4.75 body text, and Example Walkthrough need tier-aware qualification
- Mermaid diagram needs a note about Express Lane abbreviated path

### G) Task Breakdown

---

### Task 1: Add Express Lane Row and Update Small Row in Tiers Table (BC-1, BC-3)

**Contracts Implemented:** BC-1, BC-3

**Files:**
- Modify: `docs/contributing/pr-workflow.md:420-424` (PR Size Tiers table)

**Step 1: Edit the PR Size Tiers table**

Replace the existing table (lines 420-424) with a 4-row table:

| Tier | Criteria | Plan Review (Step 2.5) | Code Review (Step 4.5) | Self-Audit (Step 4.75) |
|------|----------|----------------------|----------------------|----------------------|
| **Express Lane** | ≤3 lines changed AND purely mechanical — limited to: typo fixes in prose/comments, whitespace/formatting, comment rewording, removal of commented-out code — AND no behavioral change AND not in any file from the [source-of-truth map](standards/principles.md#source-of-truth-map) | Skip | Skip | 3 dimensions (correctness, determinism, consistency) |
| **Small** | Docs-only with no process/workflow semantic changes (typo fixes, formatting, comment updates, link fixes), OR ≤3 files changed AND only mechanical changes (renames, formatting) AND no behavioral logic changes AND no new interfaces/types AND no new CLI flags | Single convergence round (no pre-pass) | Single convergence round (no pre-pass) | Full (all 10 dimensions) |
| **Medium** | 4–10 files changed, OR new policy template behind existing interface | Full two-stage (pre-pass + convergence) | Full two-stage (pre-pass + convergence) | Full (all 10 dimensions) |
| **Large** | >10 files, OR new interfaces/modules, OR architecture changes | Full two-stage (pre-pass + convergence) | Full two-stage (pre-pass + convergence) | Full (all 10 dimensions) |

**Verify:** Read back the file and confirm the table has 4 data rows with correct column values.

**Commit:** `docs(pr-workflow): add Express Lane tier and remove Small pre-pass (BC-1, BC-3)`

---

### Task 2: Update Rules Section for Express Lane (BC-2, BC-5)

**Contracts Implemented:** BC-2, BC-5

**Files:**
- Modify: `docs/contributing/pr-workflow.md:426-430` (Rules section)

**Step 1: Update the rules bullets**

Replace the "Steps 1, 1.5, 2, 3, 4, 5 are always required" bullet and add Express Lane rules:

```markdown
**Rules:**
- **Express Lane process:** Implement → self-audit (3 dimensions: correctness, determinism, consistency) → commit. No worktree, plan, convergence review, or human gate required. If the change grows beyond 3 lines, touches a source-of-truth file, or introduces any behavioral change, stop and switch to Small tier.
- **All other tiers require Steps 1–5** — worktree, source audit, plan, human review, execution, and commit apply to Small, Medium, and Large.
- **Self-audit is always full for Small and above** — the 10-dimension critical thinking check catches substance bugs that no automated review can. It costs 5 minutes and has caught 3+ real bugs in every PR where it was applied.
- **When in doubt, tier up** — if you're unsure whether a change is Express Lane or Small, use Small. If unsure between Small and Medium, use Medium.
- **Human reviewer can override** — if the human reviewer at Step 3 believes the tier is wrong, they can request a different tier.
```

**Verify:** Read back the rules section and confirm Express Lane process, upgrade rule, and source-of-truth exclusion are present.

**Commit:** `docs(pr-workflow): add Express Lane rules and upgrade safeguards (BC-2, BC-5)`

---

### Task 3: Qualify Prose References to Pre-Pass and Two-Stage (BC-6)

**Contracts Implemented:** BC-6

**Files:**
- Modify: `docs/contributing/pr-workflow.md:11-31` (Mermaid diagram — add note)
- Modify: `docs/contributing/pr-workflow.md:36-37` (Key insights)
- Modify: `docs/contributing/pr-workflow.md:103-106` (Step 2.5 two-stage section)
- Modify: `docs/contributing/pr-workflow.md:234-239` (Step 4.5 two-stage section)
- Modify: `docs/contributing/pr-workflow.md:357-360` (Step 4.75 opening)
- Modify: `docs/contributing/pr-workflow.md:444` (Example Walkthrough step 4)
- Modify: `docs/contributing/pr-workflow.md:447` (Example Walkthrough step 7)

**Step 1: Add Mermaid diagram note (F5)**

After the mermaid diagram closing tag (line 31), before the "Key insights:" line, add:

```markdown
> **Note:** The diagram above shows the default (Medium/Large) path. Express Lane PRs follow an abbreviated path: implement → self-audit (3 dimensions) → commit. Small PRs follow the full path but skip the pre-pass in Steps 2.5 and 4.5. See [PR Size Tiers](#pr-size-tiers).
```

**Step 2: Update Key Insights (S6: keep "10")**

Lines 36-37 currently say:
```
   - **Plan Review** (Step 2.5) — two-stage: holistic pre-pass, then 10 targeted perspectives. Catches design issues before implementation.
   - **Code Review** (Step 4.5) — two-stage: holistic pre-pass, then 10 targeted perspectives. Catches implementation issues before PR creation.
```

Replace with:
```
   - **Plan Review** (Step 2.5) — up to two stages depending on [PR size tier](#pr-size-tiers): holistic pre-pass (Medium/Large only), then 10 targeted perspectives. Catches design issues before implementation.
   - **Code Review** (Step 4.5) — up to two stages depending on [PR size tier](#pr-size-tiers): holistic pre-pass (Medium/Large only), then 10 targeted perspectives. Catches implementation issues before PR creation.
```

**Step 3: Qualify Step 2.5 body text (F4)**

Lines 103-106 currently say:
```
**Two-stage review:**

1. **Holistic pre-pass:** Do a single deep review to catch cross-cutting issues before the formal convergence protocol.
2. **Formal convergence:** Run all 10 perspectives below in parallel.
```

Replace with:
```
**Two-stage review (Medium/Large PRs; Small PRs skip the pre-pass — see [PR Size Tiers](#pr-size-tiers)):**

1. **Holistic pre-pass (Medium/Large only):** Do a single deep review to catch cross-cutting issues before the formal convergence protocol.
2. **Formal convergence:** Run all 10 perspectives below in parallel.
```

**Step 4: Qualify Step 4.5 body text (F4)**

Lines 234-239 currently say:
```
Review the implementation from 10 targeted perspectives, applying the [convergence protocol](convergence.md): zero CRITICAL + zero IMPORTANT = converged; fix and re-run entire round otherwise. Max 10 rounds. Same two-stage structure as Step 2.5 (holistic pre-pass, then formal convergence), but the 10 perspectives differ: plan review checks design soundness; code review checks implementation quality.

**Two-stage review:**

1. **Holistic pre-pass:** Single deep review to catch cross-cutting issues.
2. **Formal convergence:** Run all 10 perspectives below in parallel.
```

Replace with:
```
Review the implementation from 10 targeted perspectives, applying the [convergence protocol](convergence.md): zero CRITICAL + zero IMPORTANT = converged; fix and re-run entire round otherwise. Max 10 rounds. Same structure as Step 2.5 (two-stage for Medium/Large, convergence-only for Small), but the 10 perspectives differ: plan review checks design soundness; code review checks implementation quality.

**Two-stage review (Medium/Large PRs; Small PRs skip the pre-pass — see [PR Size Tiers](#pr-size-tiers)):**

1. **Holistic pre-pass (Medium/Large only):** Single deep review to catch cross-cutting issues.
2. **Formal convergence:** Run all 10 perspectives below in parallel.
```

**Step 5: Add Express Lane note to Step 4.75 (F1)**

The opening of Step 4.75 (line 357-360) currently says:
```
### Step 4.75: Pre-Commit Self-Audit

Stop, think critically, and answer each question below from your own reasoning. Do not delegate to automated tools — review each dimension yourself using critical thinking. Report all issues found. If you find zero issues, explain why you're confident for each dimension.
```

After the opening paragraph, add:
```
> **Express Lane PRs:** Check only dimensions 1 (logic bugs), 3 (determinism), and 4 (consistency). The full 10-dimension audit applies to Small and above.
```

**Step 6: Update Example Walkthrough steps 4 and 7 (F3)**

Line 444 currently says:
```
4. **Review plan:** Run all 10 perspectives (Stage 1 pre-pass, then Stage 2 convergence). Fix issues, re-run until converged.
```

Replace with:
```
4. **Review plan:** For Medium/Large PRs, run holistic pre-pass first. Then run all 10 perspectives in convergence rounds. Fix issues, re-run until converged. (Small PRs skip the pre-pass.)
```

Line 447 currently says:
```
7. **Review code:** Same two-stage review as plan. Fix issues, re-run until converged. Run verification gate.
```

Replace with:
```
7. **Review code:** Same review structure as plan (two-stage for Medium/Large, convergence-only for Small). Fix issues, re-run until converged. Run verification gate.
```

**Verify:** Read back all modified sections to confirm tier-qualified language appears in: (1) mermaid note, (2) key insights, (3) Step 2.5 body, (4) Step 4.5 body, (5) Step 4.75 note, (6) walkthrough steps 4 and 7.

**Commit:** `docs(pr-workflow): qualify all pre-pass/two-stage references as tier-dependent (BC-6)`

---

### Task 4: Update Version History (BC-4)

**Contracts Implemented:** BC-4 (confirms Medium/Large unchanged — version note documents scope)

**Files:**
- Modify: `docs/contributing/pr-workflow.md:543` (version history in appendix)

**Step 1: Add v4.2 entry**

After the v4.1 line, add:
```
**v4.2 (2026-03-17):** Added Express Lane tier for ≤3-line mechanical changes (implement → self-audit → commit, no plan/review/human gate). Eliminated pre-pass for Small PRs — single convergence round sufficient. Medium/Large unchanged (#673).
```

Also update the version in the document header (line 3) from `v4.1` to `v4.2` and the date to `2026-03-17`.

**Verify:** Read back header and version history to confirm v4.2 entry present with correct date.

**Commit:** `docs(pr-workflow): bump version to v4.2 with Express Lane changelog (BC-4)`

---

### H) Test Strategy

No Go tests — this is a docs-only PR. Verification is manual:

| Contract | Task | Verification |
|----------|------|-------------|
| BC-1 | Task 1 | Express Lane row visible in table with correct columns |
| BC-2 | Task 2 | Upgrade rule present in Rules section |
| BC-3 | Task 1 | Small row says "Single convergence round (no pre-pass)" |
| BC-4 | Task 4 | Medium/Large rows unchanged; version note confirms scope |
| BC-5 | Task 2 | Source-of-truth exclusion in criteria + rules |
| BC-6 | Task 3 | Mermaid note, Key Insights, Step 2.5/4.5 body, Step 4.75, Walkthrough steps 4+7 all qualified |

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Express Lane misused for behavioral changes | Low | Medium | Tight criteria (≤3 lines + mechanical + no canonical source) + upgrade rule |
| Small PRs miss cross-cutting issues without pre-pass | Low | Low | Pre-pass overlap with Perspectives 1-2 is documented; convergence round still catches issues |
| Stale prose references to "pre-pass for all" | Medium | Low | Task 3 explicitly qualifies all known references |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces — N/A (docs-only)
- [x] No partial implementations
- [x] No breaking changes without explicit contract updates
- [x] No hidden global state impact — N/A
- [x] All new code will pass golangci-lint — N/A (docs-only)
- [x] CLAUDE.md — checked, no update needed (doesn't reference tiers or pre-pass)
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY — checked source-of-truth map; pr-workflow.md is canonical for PR workflow; working copies (CONTRIBUTING.md, convergence-review/pr-prompts.md) don't reference tier details or pre-pass
- [x] Deviation log reviewed — no deviations
- [x] Each task produces working documentation
- [x] Task dependencies correctly ordered (1→2→3→4)
- [x] All contracts mapped to tasks

---

## Appendix: File-Level Implementation Details

**File: `docs/contributing/pr-workflow.md`**

This is the only file modified. All changes are within the existing document structure:

1. **Line 3** — version bump: `v4.1 → v4.2`
2. **After line 31** — Mermaid diagram: add Express Lane / Small note
3. **Lines 36-37** — Key Insights: qualify "two-stage" with tier reference, keep "10"
4. **Lines 103-106** — Step 2.5: qualify "Two-stage review" heading and pre-pass as Medium/Large only
5. **Lines 234-239** — Step 4.5: qualify "Two-stage review" heading and pre-pass as Medium/Large only
6. **Lines 357-360** — Step 4.75: add Express Lane 3-dimension note
7. **Lines 420-424** — PR Size Tiers table: add Express Lane row, update Small row
8. **Lines 426-430** — Rules section: rewrite for Express Lane carve-out
9. **Line 444** — Example Walkthrough step 4: qualify pre-pass reference
10. **Line 447** — Example Walkthrough step 7: qualify "two-stage" reference
11. **After line 543** — Version history: add v4.2 entry
