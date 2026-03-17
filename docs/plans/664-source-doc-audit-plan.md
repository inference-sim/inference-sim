# Source Document Audit Step — Implementation Plan

**Goal:** Add Step 1.5 (Source Document Audit) to the PR workflow so ambiguities in source documents are resolved before planning begins.
**Source:** [GitHub issue #664](https://github.com/inference-sim/inference-sim/issues/664)
**Closes:** Fixes #664

## Behavioral Contracts

BC-1: Step 1.5 Exists in Workflow
- GIVEN a reader opens `docs/contributing/pr-workflow.md`
- WHEN they read the Step-by-Step Process section
- THEN they see Step 1.5 (Source Document Audit) between Step 1 (Create Isolated Workspace) and Step 2 (Write Implementation Plan), describing three named audit checks — (1) ambiguous requirements, (2) contradictions with existing invariants or standards, (3) missing information needed for planning — and specifying `CLARIFICATION` deviation log entries as output

BC-2: Mermaid Flowchart Shows Step 1.5
- GIVEN a reader views the Overview flowchart in `pr-workflow.md`
- WHEN the Mermaid diagram renders
- THEN Step 1.5 appears between S1 and S2 in the flow, with the label "Step 1.5: Source Document Audit"

BC-3: CLARIFICATION Is a Valid Deviation Log Reason
- GIVEN a contributor writes a micro plan deviation log
- WHEN they need to record a clarification resolved during the source document audit
- THEN the micro-plan template (both human and agent-prompt versions) lists `CLARIFICATION` as a valid reason alongside `SIMPLIFICATION / CORRECTION / DEFERRAL / ADDITION / SCOPE_CHANGE`

BC-4: Example Walkthrough Includes Step 1.5
- GIVEN a reader follows the Example Walkthrough section
- WHEN they read the numbered steps
- THEN Step 1.5 (audit source document) appears between workspace creation and plan writing

BC-5: Workflow Version Updated
- GIVEN the workflow has been modified
- WHEN a reader checks the version header
- THEN the version is incremented and the date is updated

## Part 1: Design Validation

### A) Executive Summary

This PR adds a new lightweight step (1.5) to the PR workflow that requires contributors to audit their source document for ambiguities, contradictions, and missing information before writing a micro plan. The output is a list of resolved clarifications recorded in the deviation log. This is a docs-only change affecting three files: the PR workflow, the micro-plan template, and the micro-plan agent prompt.

### B) Behavioral Contracts

See above.

### C) Component Interaction

No code components. Three documentation files:

```
pr-workflow.md ──references──> micro-plan.md (Deviation Log section)
                                  │
                              micro-plan-prompt.md (agent companion)
```

The workflow references the deviation log format defined in the template. Both the human template and agent prompt must list the same valid reasons.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Update CLAUDE.md working copy of the PR workflow summary if needed" | No CLAUDE.md change needed — the PR Workflow section only says "follow docs/contributing/pr-workflow.md" without enumerating steps | SIMPLIFICATION |
| (not mentioned in issue) | Also updates `micro-plan-prompt.md` (agent companion) to add `CLARIFICATION` — ensures human template and agent prompt stay in sync | ADDITION |

### E) Review Guide

- **Scrutinize:** The Mermaid flowchart syntax (easy to break). The exact wording of Step 1.5 — is it actionable without being overly prescriptive?
- **Safe to skim:** Version bump, example walkthrough update (mechanical).
- **Known debt:** None.

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify:
- `docs/contributing/pr-workflow.md` — add Step 1.5, update Mermaid, update Example Walkthrough, bump version
- `docs/contributing/templates/micro-plan.md` — add `CLARIFICATION` to Deviation Log reasons
- `docs/contributing/templates/micro-plan-prompt.md` — add `CLARIFICATION` to Deviation Log reasons

### G) Task Breakdown

#### Task 1: Add Step 1.5 to pr-workflow.md (BC-1, BC-2, BC-4, BC-5)

**Files:** modify `docs/contributing/pr-workflow.md`

**Changes:**

1. Update the Mermaid flowchart (BC-2). Change line 22 from:
   ```
   S1 --> S2 --> S25 --> S3 --> S4 --> S45 --> S475 --> S5
   ```
   to:
   ```
   S1 --> S15 --> S2 --> S25 --> S3 --> S4 --> S45 --> S475 --> S5
   ```
   Add the node definition after S1's definition (after line 13):
   ```
   S15["Step 1.5: Source Document Audit"]
   ```

2. Add Step 1.5 section after Step 1's `---` separator (BC-1):
   ```markdown
   ### Step 1.5: Audit the Source Document

   Before writing the plan, scan the source document (design doc, macro plan section, GitHub issue, or feature request) for:

   1. **Ambiguous requirements** — flag and resolve with the document author before planning
   2. **Contradictions with existing invariants or standards** — check against `standards/invariants.md` and `standards/rules.md`
   3. **Missing information needed for planning** — which extension type? which invariants affected? which construction sites?

   **Output:** A list of resolved clarifications, appended to the micro plan's Deviation Log with reason `CLARIFICATION`.

   > **CLARIFICATION vs CORRECTION:** Use `CLARIFICATION` when the source was ambiguous or incomplete and you chose an interpretation. Use `CORRECTION` when the source was factually wrong about existing code or behavior.

   If the source document is unambiguous and complete, skip this step — but note in the plan that no clarifications were needed.
   ```

3. Update the Example Walkthrough to insert step 1.5 (BC-4):
   - After "Create worktree:" add: "**Audit source:** Scan source document for ambiguities, contradictions, missing info. Record clarifications."
   - Renumber remaining steps.

4. Bump version to v4.1 and update date (BC-5).

5. Update the PR Size Tiers "Rules" bullet (line 410) from:
   `Steps 1, 2, 3, 4, 5 are always required`
   to:
   `Steps 1, 1.5, 2, 3, 4, 5 are always required`
   Update the full line from:
   `- **Steps 1, 2, 3, 4, 5 are always required** — worktree, plan, human review, execution, and commit apply to all tiers.`
   to:
   `- **Steps 1, 1.5, 2, 3, 4, 5 are always required** — worktree, source audit, plan, human review, execution, and commit apply to all tiers.`

6. Verify the "Key insights" section (lines 32-37) still reads correctly. No update needed — Step 1.5 is a sub-step of the planning phase, not a new quality assurance stage. The two key insights (worktree isolation, three-stage QA) remain accurate.

7. Add a changelog entry to the Appendix: Workflow Evolution (after the v4.0 entry, before `</details>`):
   ```
   **v4.1 (2026-03-17):** Added Step 1.5 (Source Document Audit) between Steps 1 and 2 — structured pre-audit of source documents for ambiguities, contradictions, and missing information before planning begins. Added `CLARIFICATION` as a Deviation Log reason (#664).
   ```

**Verify:** Visual inspection — Mermaid renders, step numbering is correct.
**Lint:** N/A (markdown only)
**Commit:** `docs(pr-workflow): add Step 1.5 Source Document Audit (BC-1, BC-2, BC-4, BC-5)`

#### Task 2: Add CLARIFICATION to micro-plan.md (BC-3)

**Files:** modify `docs/contributing/templates/micro-plan.md`

**Changes:**

1. In the Deviation Log table (Part 1, Section D), update the Reason column:
   - From: `SIMPLIFICATION / CORRECTION / DEFERRAL / ADDITION / SCOPE_CHANGE`
   - To: `CLARIFICATION / SIMPLIFICATION / CORRECTION / DEFERRAL / ADDITION / SCOPE_CHANGE`

**Verify:** Visual inspection.
**Lint:** N/A (markdown only)
**Commit:** `docs(micro-plan): add CLARIFICATION as valid Deviation Log reason (BC-3)`

#### Task 3: Add CLARIFICATION to micro-plan-prompt.md (BC-3)

**Files:** modify `docs/contributing/templates/micro-plan-prompt.md`

**Changes:**

1. In the deviation log reasons list (lines 255-260), add a new bullet as the **first entry** (after line 255 "Categories of deviation:", before line 256 "SIMPLIFICATION"):
   ```
   - CLARIFICATION: Ambiguity in source document resolved during Step 1.5 audit
   ```
   This is consistent with Task 2 which places CLARIFICATION first in the reason list.
   Line 535 (`Deviation log reviewed`) does not enumerate reasons — no change needed there.

**Verify:** Visual inspection.
**Lint:** N/A (markdown only)
**Commit:** `docs(micro-plan-prompt): add CLARIFICATION as valid Deviation Log reason (BC-3)`

### H) Test Strategy

No code tests. Verification is by visual inspection that:
- Mermaid flowchart renders with the new node
- Step 1.5 appears in correct position
- CLARIFICATION appears in both template files
- Example walkthrough includes the new step

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Mermaid syntax error breaks flowchart | Medium | Medium | Careful editing of flowchart, verify renders |
| Step numbering confusion (1.5 vs integer steps) | Low | Low | Issue explicitly requests "Step 1.5"; consistent with existing "Step 2.5", "Step 4.5" pattern |

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] CLAUDE.md — no update needed (PR Workflow section just references the file)
- [x] Documentation DRY — both micro-plan.md and micro-plan-prompt.md updated for CLARIFICATION
- [x] Deviation log reviewed — one SIMPLIFICATION, one ADDITION

---

## Appendix: File-Level Implementation Details

**File: `docs/contributing/pr-workflow.md`**

- **Purpose:** Main PR workflow document. Add Step 1.5, update Mermaid, update Size Tiers rules, update Example Walkthrough, bump version, add changelog.
- **Key edits:** Lines 3 (version), 13-14 (Mermaid node), 22 (Mermaid chain), 63-64 (new Step 1.5 section insertion point), 410 (Size Tiers rules), 424-431 (Example Walkthrough renumbering), 524 (changelog entry).

**File: `docs/contributing/templates/micro-plan.md`**

- **Purpose:** Human-readable micro-plan template. Add CLARIFICATION to Deviation Log reason list.
- **Key edits:** Line 151 (Deviation Log table Reason column).

**File: `docs/contributing/templates/micro-plan-prompt.md`**

- **Purpose:** Agent prompt companion for micro-plan generation. Add CLARIFICATION to deviation log reasons.
- **Key edits:** After line 258 (add new bullet for CLARIFICATION reason).
