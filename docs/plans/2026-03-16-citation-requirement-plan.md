# Citation Requirement for Convergence-Review Perspective Prompts

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Closes:** #666
**Parent issue:** #665
**PR tier:** Medium (3 files, semantic workflow change — not purely mechanical)

---

## Executive Summary

Review agents currently produce findings with only a severity level and vague description (e.g., "IMPORTANT: there might be an issue with KV conservation"). These findings are unverifiable without re-reading the entire artifact. This PR updates the footer of every perspective prompt in three review-prompt files to require a specific location citation for each finding. Findings without citations are explicitly marked as DISCARDED.

**Scope:** Three skill files only — no Go code changes, no CLI changes, no test changes.

---

## Part 1: Design

### A. Behavioral Contracts

**BC-1 (PR Plan prompts):** GIVEN the convergence-review skill dispatches any of the 9 PR plan perspectives (PP-1 through PP-4, PP-6 through PP-10 in pr-prompts.md — PP-5 is a direct Claude check, not agent-dispatched), WHEN the agent produces a finding, THEN the finding MUST include: severity, exact location (section heading + line for plan docs), specific issue, and expected correct behavior. Findings without a location are marked DISCARDED.

**BC-2 (PR Code prompts):** GIVEN the convergence-review skill dispatches any of the 10 PR code perspectives (PC-1 through PC-10 in pr-prompts.md), WHEN the agent produces a finding, THEN the finding MUST include: severity, exact location (file:line for code), specific issue, and expected correct behavior. Findings without a location are marked DISCARDED.

**BC-3 (Design/Macro prompts):** GIVEN the convergence-review skill dispatches any of the 16 design/macro-plan perspectives (DD-1 through DD-8, MP-1 through MP-8 in design-prompts.md), WHEN the agent produces a finding, THEN the finding MUST include the four required fields. Findings without a location are marked DISCARDED.

**BC-4 (Hypothesis prompts):** GIVEN the hypothesis-experiment skill dispatches any of the 20 review perspectives (DR-1 through DR-5, CR-1 through CR-5, FR-1 through FR-10 in review-prompts.md), WHEN the agent produces a finding, THEN the finding MUST include the four required fields. Findings without a location are marked DISCARDED.

**BC-5 (Format consistency):** The new footer is identical across all 55 prompt instances (19 in pr-prompts.md + 16 in design-prompts.md + 20 in review-prompts.md). The only change is footer text — all existing prompt body content is untouched.

### B. Old Footer (to be replaced everywhere)

```
Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### C. New Footer (to replace all instances)

```
For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### D. Instance Counts (verified by grep)

| File | Instances | Notes |
|------|-----------|-------|
| `.claude/skills/convergence-review/pr-prompts.md` | 19 | PP-5 has no footer (direct Claude check, not dispatched) |
| `.claude/skills/convergence-review/design-prompts.md` | 16 | DD-1–8 (8) + MP-1–8 (8) |
| `.claude/skills/hypothesis-experiment/review-prompts.md` | 20 | DR-1–5 (5) + CR-1–5 (5) + FR-1–10 (10) |
| **Total** | **55** | |

> **Note on issue count:** Issue #666 states "20 prompt footers (10 plan + 10 code)" for pr-prompts.md, but the actual file has 19 because PP-5 ("Structural Validation — perform directly, no agent") explicitly instructs Claude to perform checks directly without agent dispatch. It has no prompt-footer block. This is correct behavior — PP-5 is not an agent-dispatch prompt.

### E. Risks

- **Risk:** The new footer text inside a code-block (```) might confuse agents reading the skill file — they might think the footer IS their prompt. **Mitigation:** The existing prompts already use code blocks this way; agents handle it correctly.
- **Risk:** One prompt in review-prompts.md (line 417) may end the file without a trailing newline after the footer. **Mitigation:** Task 3 verifies the file ends correctly after replacement.

---

## Part 2: Implementation Tasks

### Task 1: Update pr-prompts.md (19 footer replacements)

**Files:**
- Modify: `.claude/skills/convergence-review/pr-prompts.md`

**Step 1: Verify current state**

```bash
cd .worktrees/issue-666-citations
grep -c "Rate each finding as CRITICAL" .claude/skills/convergence-review/pr-prompts.md
```
Expected output: `19`

**Step 2: Apply the replacement**

Use Edit tool with `replace_all: true`. The `old_string` and `new_string` are the interior footer lines only — do NOT include the surrounding triple-backtick fences (` ``` `), which stay in place.

```
old_string (exact 2 lines — interior of code block, no backticks):
Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.

new_string (exact 7 lines — interior of code block, no backticks):
For each finding, you MUST provide:
- Severity: CRITICAL, IMPORTANT, or SUGGESTION
- Location: exact file:line (for code) or section heading + line (for docs/plans)
- Issue: what is wrong (specific, not vague)
- Expected: what the correct behavior should be

Findings without a specific location will be DISCARDED as unverifiable.

Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.
```

**Step 3: Verify the replacement**

```bash
grep -c "Rate each finding" .claude/skills/convergence-review/pr-prompts.md
```
Expected: `0`

```bash
grep -c "DISCARDED as unverifiable" .claude/skills/convergence-review/pr-prompts.md
```
Expected: `19`

```bash
grep -c "Location:" .claude/skills/convergence-review/pr-prompts.md
```
Expected: `19`

**Step 4: Verify PP-5 (no agent-dispatch) is untouched**

PP-5 starts at line 78. Confirm it still reads: `> **For Claude:** Perform these 4 checks directly. Do NOT dispatch an agent.` with no footer block.

```bash
grep -A3 "PP-5: Structural Validation" .claude/skills/convergence-review/pr-prompts.md
```
Expected output includes: `Perform these 4 checks directly. Do NOT dispatch an agent.`

**Step 5: Commit**

```bash
cd .worktrees/issue-666-citations
git add .claude/skills/convergence-review/pr-prompts.md
git commit -m "feat(convergence-review): require file:line citations in PR perspective prompts

- Implement BC-1: 9 PR plan perspectives (PP-1-4, PP-6-10) now require
  severity + location + issue + expected for each finding
- Implement BC-2: 10 PR code perspectives (PC-1-10) now require
  severity + location (file:line) + issue + expected for each finding
- Findings without a specific location are marked DISCARDED as unverifiable
- PP-5 (direct Claude check, no agent dispatch) unchanged

Part of #666"
```

---

### Task 2: Update design-prompts.md (16 footer replacements)

**Files:**
- Modify: `.claude/skills/convergence-review/design-prompts.md`

**Step 1: Verify current state**

```bash
grep -c "Rate each finding as CRITICAL" .claude/skills/convergence-review/design-prompts.md
```
Expected: `16`

**Step 2: Apply the replacement**

Same `replace_all: true` edit — identical old/new strings as Task 1.

**Step 3: Verify**

```bash
grep -c "Rate each finding" .claude/skills/convergence-review/design-prompts.md
```
Expected: `0`

```bash
grep -c "DISCARDED as unverifiable" .claude/skills/convergence-review/design-prompts.md
```
Expected: `16`

**Step 4: Commit**

```bash
git add .claude/skills/convergence-review/design-prompts.md
git commit -m "feat(convergence-review): require file:line citations in design/macro perspective prompts

- Implement BC-3: 8 design document perspectives (DD-1-8) now require
  severity + location + issue + expected for each finding
- Implement BC-3: 8 macro plan perspectives (MP-1-8) now require
  severity + location (section heading + line) + issue + expected
- Findings without a specific location are marked DISCARDED as unverifiable

Part of #666"
```

---

### Task 3: Update hypothesis-experiment/review-prompts.md (20 footer replacements)

**Files:**
- Modify: `.claude/skills/hypothesis-experiment/review-prompts.md`

**Step 1: Verify current state**

```bash
grep -c "Rate each finding as CRITICAL" .claude/skills/hypothesis-experiment/review-prompts.md
```
Expected: `20`

**Step 2: Apply the replacement**

Same `replace_all: true` edit — identical old/new strings as Tasks 1 and 2.

**Step 3: Verify**

```bash
grep -c "Rate each finding" .claude/skills/hypothesis-experiment/review-prompts.md
```
Expected: `0`

```bash
grep -c "DISCARDED as unverifiable" .claude/skills/hypothesis-experiment/review-prompts.md
```
Expected: `20`

**Step 4: Verify file ends cleanly**

The last prompt (FR-10) ends at approximately line 419. Check the file ends with the footer and a trailing newline:

```bash
tail -5 .claude/skills/hypothesis-experiment/review-prompts.md
```
Expected: last line is `Report: (1) numbered list of findings with severity and location, (2) total CRITICAL count, (3) total IMPORTANT count.` followed by ` ``` ` and a newline.

**Step 5: Commit**

```bash
git add .claude/skills/hypothesis-experiment/review-prompts.md
git commit -m "feat(hypothesis-experiment): require file:line citations in review perspective prompts

- Implement BC-4: 20 hypothesis review perspectives (DR-1-5, CR-1-5, FR-1-10)
  now require severity + location + issue + expected for each finding
- Findings without a specific location are marked DISCARDED as unverifiable
- Consistent with convergence-review updates in same PR

Closes #666"
```

---

## Part 3: Self-Audit Checklist (Step 4.75)

Completing the 10-dimension self-audit at execution time (not pre-answered — must be done after implementation):

1. **Logic bugs:** Is the new footer text correct? Does it accidentally break the prompt structure (e.g., markdown leaking out of code blocks)?
2. **Design bugs:** Does the "DISCARDED" enforcement actually work? Enforcement is prompt-level only — there is no programmatic filter in the triage pipeline. The footer instructs agents to omit locationless findings; the convergence-review skill's human tally step provides the backstop. This is the correct design given the constraints.
3. **Determinism:** Not applicable (no Go code).
4. **Consistency:** Are all 55 instances identical? No variation in wording across PR/design/hypothesis files?
5. **Documentation:** Does this PR need to update `docs/contributing/pr-workflow.md`? Check: pr-workflow.md describes perspectives narratively; it does NOT contain the prompt footers. No update needed.
6. **Edge cases:** PP-5 (no footer) — already addressed in Task 1 Step 4.
7. **Test epistemology:** Grep-based verification is correct — the count check is a definitive assertion that all instances were replaced.
8. **Construction site uniqueness:** N/A (no struct fields added).
9. **Error path completeness:** N/A (no code paths).
10. **Documentation DRY:** The canonical source note at the top of each file (`pr-workflow.md`, `hypothesis.md`, etc.) does not need updating — it describes content authority, not footer format.

---

## Appendix: File Locations

All files are in the project root's `.claude/skills/` directory:

```
.claude/skills/
├── convergence-review/
│   ├── pr-prompts.md        ← 19 footer instances
│   └── design-prompts.md    ← 16 footer instances
└── hypothesis-experiment/
    └── review-prompts.md    ← 20 footer instances
```

These files are skill configuration tracked in the git repository. They are not Go source files, but they are committed alongside the Go codebase. The `.gitignore` only ignores `.claude/worktrees/` and other runtime artifacts — not `.claude/skills/`.

To verify they are tracked:
```bash
git ls-files .claude/skills/convergence-review/pr-prompts.md
```
Expected: `.claude/skills/convergence-review/pr-prompts.md`

---

## Deviation Log

| Deviation | Reason | Approved |
|-----------|--------|---------|
| Issue says "20 footers" in pr-prompts.md; actual is 19 | PP-5 has no agent-dispatch footer (correct behavior — it's a direct Claude check). Issue count was an estimate. | No approval needed — pure fact correction. |
| No changes to pr-workflow.md or hypothesis.md | Those docs describe perspectives narratively, not with exact prompt footer text. Only the prompt files need updating. | N/A |
