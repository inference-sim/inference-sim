# Convergence-Review Structured JSON Findings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make convergence-review findings durable by writing per-perspective findings to the state file, enabling audit trails, cross-round fix verification, and session crash recovery.

**The problem today:** Convergence findings exist only in conversation context. If the session crashes, context compacts, or the human wants to audit later, the evidence is gone. Phase B reads findings from conversation context, which is fragile. Round N+1 cannot mechanically verify that Round N's CRITICAL/IMPORTANT items were actually addressed — it relies on the agent asserting they were fixed.

**What this PR adds:**
1. **Findings producer/consumer contract** — the state file's `findings` array becomes a required, well-defined structure with producer contract (Phase A MUST write all fields) and consumer contract (Phase B MUST read from state file). Each finding records perspective ID, severity, location (file:line), description, and disposition (fix/filed/downgraded/resolved).
2. **Phase A persistence** — the tally step accumulates findings in memory through steps 4-7, then persists them atomically in step 8. Findings survive context compaction and session crashes.
3. **Phase B state-driven fixing** — Phase B reads findings from the state file (not conversation context), making fix lists deterministic across session boundaries.
4. **Fix verification gate** — before dispatching Round N+1's perspectives, Phase A checks that each Round N `fix`-disposition finding shows evidence of being addressed (referenced file appears dirty in `git status`, or finding was explicitly marked `resolved` in Phase B). This is a best-effort heuristic — the next Phase A round is the real verification.

**Why this matters:** This closes the reliability gap between the state-file-driven loop (#683) and the actual findings data. The state file tracks *that* we're in round 3, but not *what* was found or fixed. This PR makes the state file the single source of truth for the entire convergence lifecycle.

**Architecture:** Pure skill modification — edits `.claude/skills/convergence-review/SKILL.md`. No Go code. The changes touch four sections: State File schema, Phase A steps 4/8, Phase B steps 1/6, and a new fix-verification substep in Phase A entry.

**Source:** GitHub issue #668

**Closes:** Fixes #668

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR modifies the convergence-review skill's SKILL.md to make per-perspective findings a first-class part of the state file. Today, the state file tracks round counts, severity tallies, and convergence status — but individual findings (what was found, where, and whether it was fixed) live only in conversation context. This PR establishes the `findings` array as a producer/consumer contract (Phase A writes, Phase B reads), adds explicit read/write instructions to both phases, and introduces a fix-verification pre-check between rounds.

The skill interacts with: the Phase A/B loop machinery (already in SKILL.md from #683), the perspective prompt files (pr-prompts.md, design-prompts.md, review-prompts.md — unchanged), and the `.claude/convergence-state/` directory (already gitignored).

No deviations from issue #668's scope. The deviation log below lists exclusions from the related (closed) issue #667, not deviations from #668 itself.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Findings Persistence**
- GIVEN a convergence review round completes Phase A tally
- WHEN findings are extracted from perspective agent outputs
- THEN the state file's history entry for that round MUST contain a `findings` array with one entry per finding, each having: `perspective` (string), `severity` (CRITICAL/IMPORTANT/SUGGESTION), `location` (string, file:line or section reference), `description` (string), and `disposition` (fix/filed/downgraded/resolved)
- MECHANISM: Phase A step 4 extracts findings into an in-memory array; steps 5-7 may update dispositions (triage); step 8 persists the final array atomically to the state file.

**BC-2: Phase B Reads From State**
- GIVEN Phase B is entered after a not-converged tally
- WHEN Phase B lists findings to fix (step 1)
- THEN Phase B MUST read the findings list from the state file's most recent history entry, not from conversation context
- MECHANISM: Phase B step 1 explicitly references the state file as the source.

**BC-3: Fix Verification Before Re-dispatch**
- GIVEN Round N produced CRITICAL or IMPORTANT findings with disposition `fix`
- WHEN Phase A is entered for Round N+1
- THEN before dispatching perspectives, the skill MUST check that each Round N `fix`-disposition finding has evidence of being addressed: either (a) the finding was marked `resolved` in Phase B, or (b) the file referenced in `location` appears dirty in `git status`. If `location` is not a parseable file path (e.g., "unknown", a section heading, or a context-anchored reference), the finding is assumed addressed if Phase B completed — file-based verification is skipped with a note.
- MECHANISM: New substep between state loading and perspective dispatch in Phase A. This is a best-effort heuristic — the `location` field records where the problem was found, not necessarily which file was edited to fix it. The next Phase A round (full re-review) is the authoritative verification.

**BC-4: Findings Producer/Consumer Contract**
- GIVEN the state file schema section in SKILL.md
- WHEN an implementer reads the schema
- THEN the `findings` array MUST be documented as a producer/consumer contract: Phase A MUST write all specified fields (producer), Phase B MUST read from them (consumer). The schema MUST list required fields and their types.
- MECHANISM: Replace "illustrative, not normative" with producer/consumer contract language and add field documentation table.

#### Negative Contracts

**BC-5: No Conversation Context Dependency**
- GIVEN Phase B is entered after a session crash and restart
- WHEN the skill resumes from the state file
- THEN Phase B MUST NOT require any conversation context from the previous session to determine what to fix
- MECHANISM: All finding data is in the state file.

**BC-6: Fix Verification Skips Non-Fix Dispositions**
- GIVEN a Round N finding with disposition `filed`, `downgraded`, or `resolved`
- WHEN fix verification runs before Round N+1
- THEN the finding MUST be excluded from the verification check (it was intentionally deferred, demoted, or already confirmed fixed)
- MECHANISM: Verification only checks findings with disposition `fix`.

#### Error Handling Contracts

**BC-7: Missing Location Graceful Handling**
- GIVEN a perspective agent produces a finding without a parseable file:line location
- WHEN the tally step writes findings to the state file
- THEN the finding MUST still be recorded with `location` set to the best available reference (section heading, "unknown", or the raw text), and a warning MUST be emitted
- MECHANISM: Location is best-effort, not a hard gate for recording.

**BC-8: Backward Compatibility for Pre-#668 State Files**
- GIVEN a state file written before this PR (history entries may lack a `findings` array)
- WHEN Phase B step 1 attempts to read the `findings` array
- THEN if the array is missing or null, Phase B MUST treat it as an empty findings list and emit a warning: `"History entry for round N has no findings array (pre-#668 state file). Proceeding with empty fix list."`
- MECHANISM: Null-check on `findings` array in Phase B step 1 before filtering.

### C) Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│                 convergence-review SKILL.md              │
│                                                         │
│  Phase A                    Phase B                     │
│  ┌─────────────────┐       ┌─────────────────┐         │
│  │ 1. Load state   │       │ 1. Read findings │←── BC-2 │
│  │ 1a. Fix verify  │←BC-3  │    from state    │         │
│  │ 2. Dispatch     │       │ 2. Fix items     │         │
│  │ 3. Collect      │       │ 3. Stage files   │         │
│  │ 4. Tally+Write  │←BC-1  │ 4. Verify gate   │         │
│  │ 5. Triage       │       │ 5. Update state  │←── BC-2 │
│  │ 6. Converge?    │       │ 6. Re-enter A    │         │
│  └────────┬────────┘       └─────────────────┘         │
│           │                         ↑                   │
│           └── not-converged ────────┘                   │
│                                                         │
│  State File (.claude/convergence-state/<id>.json)       │
│  ┌──────────────────────────────────────────┐           │
│  │ history[N].findings[] ← NEW (normative)  │           │
│  │   .perspective, .severity, .location,    │           │
│  │   .description, .disposition             │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

**State ownership:** The state file is owned by the skill. No other skill reads or writes it.

**Extension friction:** Adding a new field to the findings schema requires editing 1 file (SKILL.md). Low friction.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Add canary result to structured convergence report (if #TBD JSON report is implemented)" | Omitted | SCOPE_CHANGE: Canary injection (#667) was closed as wontfix. No canary result field needed. |
| "Create `.claude/skills/convergence-review/canaries/`" | Omitted | SCOPE_CHANGE: Same — #667 closed. |

### E) Review Guide

1. **The tricky part:** BC-3 (fix verification) — the mechanism for detecting "was this finding addressed?" is heuristic. Checking file modification time is fragile (a file could be modified for unrelated reasons). The plan uses file modification as evidence, not proof — the next Phase A round is the real verification. The fix-verification gate is a fast pre-check, not a substitute for re-review.
2. **What to scrutinize:** The wording of Phase B step 1 (BC-2) — it must be unambiguous that findings come from state, not context. Also the schema field definitions (BC-4) — are the types and constraints clear enough for an LLM to follow?
3. **What's safe to skim:** The deviation log (only #667 omission). The component diagram (straightforward).
4. **Known debt:** The `location` field is best-effort (BC-7). Perspective agents don't always produce clean file:line references, especially for doc/plan gates. A future PR could add structured location parsing.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `.claude/skills/convergence-review/SKILL.md` — all changes in this single file

**Key decisions:**
- Schema becomes normative; field types documented inline with the JSON example
- Fix verification is a soft gate (warning + proceed) not a hard gate (block dispatch) — the next Phase A round is the real check
- `disposition` field uses string enum: `fix`, `filed`, `downgraded`, `resolved`

### G) Task Breakdown

#### Task 1: Make State File Schema Normative (BC-4)

**Contracts Implemented:** BC-4

**Files:**
- Modify: `.claude/skills/convergence-review/SKILL.md:66-93`

**Step 1: Edit the schema section**

Context: The schema is currently labeled "illustrative, not normative." We make it normative and document the findings array fields.

In `.claude/skills/convergence-review/SKILL.md`, replace line 66:

```
**Schema (illustrative, not normative — exact field names owned by implementation):**
```

with:

```
**Schema (producer/consumer contract — Phase A MUST write these fields, Phase B MUST read from them):**
```

Then, update the sample findings in the JSON block (lines 85-86) to include the `location` field:

```json
        {"perspective": "PC-1", "severity": "CRITICAL", "location": "sim/kv/cache.go:142", "disposition": "fix", "description": "Missing zero-guard on denominator"},
        {"perspective": "PC-3", "severity": "IMPORTANT", "location": "sim/config.go:88", "disposition": "filed", "issue": "#692", "description": "Exported mutable map"}
```

Then, after the JSON block (after the closing fence), add a field documentation table:

```markdown
**Findings array fields (required per finding):**

| Field | Type | Description |
|-------|------|-------------|
| `perspective` | string | Perspective ID (e.g., "PC-1", "PP-3", "DD-2") |
| `severity` | string | One of: `CRITICAL`, `IMPORTANT`, `SUGGESTION` |
| `location` | string | File:line reference, section heading, or "unknown" if not parseable |
| `description` | string | What is wrong (specific, not vague) |
| `disposition` | string | One of: `fix` (in-scope, to be fixed), `filed` (GitHub issue created), `downgraded` (demoted to SUGGESTION), `resolved` (confirmed fixed in Phase B). Initial value at Phase A extraction is always `fix`; transitions to other values happen in triage (step 6) or Phase B. |
| `issue` | string (optional) | GitHub issue number (e.g., `"#692"`). Present only when `disposition == "filed"`. |
```

**Step 2: Verify the edit**

Read the modified section to confirm the schema label and table are correct.

**Step 3: Commit**

```bash
git add .claude/skills/convergence-review/SKILL.md
git commit -m "feat(convergence-review): make state file findings schema normative (BC-4)

- Change schema label from 'illustrative' to 'normative'
- Add findings array field documentation table
- Document disposition enum: fix/filed/downgraded/resolved

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Update Phase A Tally to Write Findings (BC-1, BC-7)

**Contracts Implemented:** BC-1, BC-7

**Files:**
- Modify: `.claude/skills/convergence-review/SKILL.md:175` (Phase A step 4)
- Modify: `.claude/skills/convergence-review/SKILL.md:194` (Phase A step 8)

**Step 1: Expand Phase A step 4**

Context: Step 4 currently says "Collect and tally independently." We add explicit instructions to build a findings array for the state file.

Replace line 175:

```
4. **Collect and tally independently.** Read each agent's output. Extract findings with severity. Count CRITICAL and IMPORTANT yourself. **Never trust agent-reported totals** (per #390).
```

with:

```
4. **Collect, extract, and tally independently.** For each agent's output:
   a. Extract individual findings. For each finding, record: `perspective` (agent ID), `severity`, `location` (file:line or best available reference), `description`.
   b. If a finding lacks a parseable location, set `location` to `"unknown"` and emit a warning: `"Finding from <perspective> has no location — recorded as 'unknown'."` (BC-7)
   c. Set initial `disposition` to `"fix"` for all findings. (Triage in step 6 may update some to `"filed"`.)
   d. Count CRITICAL and IMPORTANT yourself. **Never trust agent-reported totals** (per #390).
   e. Build the round's `findings` array in memory from all extracted findings across all perspectives. This array is NOT yet written to disk — it is persisted atomically in step 8 after triage (step 6) may have updated dispositions.
   f. If a perspective agent produces no parseable findings, record zero findings for that perspective (not an error).
```

**Step 2: Update Phase A step 8**

Replace line 194:

```
8. **Update state file:** Append round entry to history, set status, write `updated_at`. For file/diff-anchored gates, record full SHA (`git rev-parse HEAD`).
```

with:

```
8. **Update state file:** Append round entry to history — including the complete `findings` array built in step 4 — set status, write `updated_at`. For file/diff-anchored gates, record full SHA (`git rev-parse HEAD`). The findings array MUST be persisted before proceeding to step 9 (BC-1).
```

**Step 3: Verify the edits**

Read both modified sections.

**Step 4: Commit**

```bash
git add .claude/skills/convergence-review/SKILL.md
git commit -m "feat(convergence-review): write per-perspective findings to state file (BC-1, BC-7)

- Expand Phase A step 4 with finding extraction protocol
- Handle missing locations gracefully (BC-7: set to 'unknown' + warning)
- Ensure findings array persisted in step 8 before proceeding

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Update Phase B to Read From State File (BC-2, BC-5, BC-8)

**Contracts Implemented:** BC-2, BC-5, BC-8

**Files:**
- Modify: `.claude/skills/convergence-review/SKILL.md:209` (Phase B step 1)
- Modify: `.claude/skills/convergence-review/SKILL.md:223` (Phase B step 6)

**Step 1: Update Phase B step 1**

Context: Step 1 currently says "List all findings to fix, grouped by priority." It's ambiguous where findings come from. Make it explicit.

Replace line 209:

```
1. **List all findings to fix**, grouped by priority.
```

with:

```
1. **Read findings from the state file.** Load the most recent history entry's `findings` array. If the array is missing or null (pre-#668 state file), treat as empty and emit a warning (BC-8). Filter to findings with `disposition == "fix"`. Group by severity (CRITICAL first, then IMPORTANT, then SUGGESTION). This is the authoritative fix list — do not rely on conversation context for what to fix (BC-2, BC-5).
```

**Step 2: Update Phase B step 6**

Context: Step 6 says "Record fixes in history." We clarify that disposition is updated per-finding.

Replace line 223:

```
6. **Update state file:** Record fixes in history (with disposition for each finding), set status to `not-converged`.
```

with:

```
6. **Update state file:** **Mutate in-place** the current round's history entry (do NOT append a new entry — SK-INV-2 round monotonicity). For each finding that was fixed, update its `disposition` from `"fix"` to `"resolved"`. For findings downgraded by the user during CRITICAL review, update `disposition` to `"downgraded"`. Persist the state file with status `not-converged`.
```

**Step 2a: Fix pre-existing Phase B step numbering gap**

The current SKILL.md Phase B steps are numbered 1, 2, 4, 5, 6, 7, 8 (step 3 is missing). Renumber to consecutive 1-7. Apply these replacements AFTER the step 1 and step 6 content replacements above (so the content is already updated):

| Current step number | New step number | Content starts with |
|---|---|---|
| `4. **Stage any new files**` | `3. **Stage any new files**` |
| `5. **Run verification gate**` | `4. **Run verification gate**` |
| `6. **Update state file:** **Mutate in-place**` | `5. **Update state file:** **Mutate in-place**` |
| `7. **Emit transition banner:**` | `6. **Emit transition banner:**` |
| `8. **Re-enter Phase A immediately.**` | `7. **Re-enter Phase A immediately.**` |

Note: Steps 1 and 2 keep their numbers. The new total is 7 steps (was 8 with the gap).

**Step 3: Verify the edits**

Read both modified sections. Confirm step numbering is consecutive 1-7.

**Step 4: Commit**

```bash
git add .claude/skills/convergence-review/SKILL.md
git commit -m "feat(convergence-review): Phase B reads findings from state file (BC-2, BC-5)

- Phase B step 1 explicitly reads from state file findings array
- Phase B step 5 (was 6) mutates disposition in-place (fix → resolved)
- Fix pre-existing step numbering gap (1,2,4-8 → consecutive 1-7)
- Ensures session crash recovery without conversation context

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Add Fix Verification Gate (BC-3, BC-6)

**Contracts Implemented:** BC-3, BC-6

**Files:**
- Modify: `.claude/skills/convergence-review/SKILL.md:155-166` (Phase A entry conditions, between state loading and dispatch)

**Step 1: Add fix verification substep**

Context: After loading state (step 2) and before dispatching perspectives (step 3), we add a verification substep for Round N+1 that checks Round N's findings were addressed.

After the existing step 2 block (after line 165, before step 3 at line 167), insert:

Also rename the section header from "**Entry conditions (checked in order):**" to "**Entry conditions and pre-dispatch checks (checked in order):**" since step 2a is a pre-dispatch action, not an entry condition.

```markdown
2a. **Fix verification (Round 2+ only).** If this is Round 2 or later and the previous round has a history entry:
   - Load the previous round's `findings` array.
   - For each finding with `disposition == "fix"`:
     - Extract the file path from `location` (if parseable as a file path).
     - If `location` is not a parseable file path (e.g., `"unknown"`, a section heading, or a context reference): skip file-based verification for this finding with a note: `"Skipping file verification for <perspective> finding — no file path in location."`
     - If `location` is a file path: check if the file appears dirty in `git status`. If the file is NOT dirty and the finding was not explicitly marked `resolved`: emit a warning: `"WARNING: Round N finding not yet addressed: <perspective> <severity>: <description truncated to 80 chars>"`
   - Findings with `disposition` of `filed`, `downgraded`, or `resolved` are skipped (BC-6).
   - **This is a soft gate:** warnings are emitted but dispatch proceeds regardless. The next Phase A round is the real verification — this substep catches obvious oversights early.
   - **Note:** The `location` field records where the problem was found, not necessarily which file was edited to fix it. A finding citing `sim/config.go:88` might be fixed by editing `sim/kv/cache.go`. This heuristic catches the common case; the full re-review catches the rest.
```

**Step 2: Verify the edit**

Read the modified Phase A entry section.

**Step 3: Commit**

```bash
git add .claude/skills/convergence-review/SKILL.md
git commit -m "feat(convergence-review): add fix verification gate between rounds (BC-3, BC-6)

- New Phase A step 2a checks previous round's findings were addressed
- Soft gate: warns but does not block dispatch
- Skips filed/downgraded/resolved findings (BC-6)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Add SK-INV-6 for Findings Persistence

**Contracts Implemented:** BC-1 (strengthening via invariant)

**Files:**
- Modify: `.claude/skills/convergence-review/SKILL.md:141-149` (Behavioral Invariants section)

**Step 1: Add SK-INV-6**

Context: The existing SK-INV-1 through SK-INV-5 cover loop integrity, round monotonicity, tally independence, state-status consistency, and stale invalidation. Add SK-INV-6 for findings persistence.

After SK-INV-5 (line 149), add:

```markdown
- **SK-INV-6 Findings persistence:** Every Phase A tally that writes a history entry (step 8) MUST include a `findings` array with all extracted findings. Phase B MUST read its fix list exclusively from the state file's findings array, never from conversation context. A session crash after Phase A step 8 (state file write) MUST NOT lose any finding data. A crash during steps 4-7 (before step 8) results in an incomplete round — the `reviewing` status triggers re-dispatch on resume, which is the correct recovery path.
```

**Step 2: Verify the edit**

Read the behavioral invariants section.

**Step 3: Commit**

```bash
git add .claude/skills/convergence-review/SKILL.md
git commit -m "feat(convergence-review): add SK-INV-6 findings persistence invariant

- Phase A must persist findings array in every history entry
- Phase B must read from state file, not conversation context
- Session crash must not lose finding data

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Verification |
|----------|------|-----------|--------------|
| BC-1 | Task 2 | Manual | Read SKILL.md Phase A step 4 — confirms findings write instruction |
| BC-2 | Task 3 | Manual | Read SKILL.md Phase B step 1 — confirms state file read instruction |
| BC-3 | Task 4 | Manual | Read SKILL.md Phase A step 2a — confirms fix verification gate |
| BC-4 | Task 1 | Manual | Read SKILL.md schema section — confirms normative label + field table |
| BC-5 | Task 3 | Manual | Read SKILL.md Phase B step 1 — confirms no conversation context dependency |
| BC-6 | Task 4 | Manual | Read SKILL.md Phase A step 2a — confirms filed/downgraded/resolved skip |
| BC-7 | Task 2 | Manual | Read SKILL.md Phase A step 4b — confirms unknown location handling |
| BC-8 | Task 3 | Manual | Read SKILL.md Phase B step 1 — confirms null-check for missing findings array |

This is a skill-only PR. There are no Go tests. Verification is by reading the modified SKILL.md and confirming the instructions are unambiguous and internally consistent. The real "test" is the next convergence review invocation — if the skill executor follows the updated instructions, findings will be persisted.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Fix verification heuristic produces false positives (file modified for unrelated reasons) | Medium | Low | Soft gate — warnings only, does not block. Next Phase A round is the real check. |
| LLM executor ignores new findings-write instructions | Low | Medium | SK-INV-6 invariant makes it a named, checkable property. Same enforcement as SK-INV-1 through SK-INV-5. |
| Schema version bump needed | Low | Low | schema_version stays at 1 — the findings array was already in the illustrative schema. No breaking change. Pre-#668 state files with missing `findings` arrays are handled by BC-8 (treat as empty array + warning). |
| State file grows large with many findings across many rounds | Low | Low | State files are deleted on convergence. Typical convergence is 1-3 rounds. Max 10 rounds × ~20 findings = ~200 entries — trivially small JSON. |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — pure skill text edits
- [x] No feature creep beyond PR scope — no canary injection (#667 closed)
- [x] No unexercised flags or interfaces — no new code
- [x] No partial implementations — all 4 behavioral changes complete
- [x] No breaking changes — schema_version stays at 1, findings array was already illustrative
- [x] No hidden global state impact — state file already existed
- [x] CLAUDE.md — no update needed (no new files/packages/CLI flags)
- [x] No stale references — verified issue #668 dependency on #430/#683 resolved
- [x] Documentation DRY — SKILL.md is the canonical source for convergence-review; no working copies elsewhere
- [x] Deviation log reviewed — only #667 omission (justified: wontfix)
- [x] Each task produces working, testable skill text
- [x] Task dependencies correctly ordered (1→2→3→4→5, linear)
- [x] All contracts mapped to tasks

**Antipattern rules:** Not applicable (no Go code). R1-R23 are code-level rules.

---

## Appendix: File-Level Implementation Details

### File: `.claude/skills/convergence-review/SKILL.md`

**Purpose:** The convergence-review skill definition. All changes are in this file.

**Changes by section:**

1. **Line 66 (Schema label):** `"illustrative, not normative"` → producer/consumer contract framing
2. **Lines 85-86 (JSON example):** Add `location` field to sample finding entries
3. **After line 93 (Schema field docs):** New table documenting findings array fields with disposition lifecycle
4. **Lines 141-149 (Behavioral Invariants):** Add SK-INV-6 (crash-timing-precise)
5. **Line 155 (Phase A section header):** Rename to "Entry conditions and pre-dispatch checks"
6. **Between lines 165-167 (Phase A entry):** New step 2a (fix verification gate with dirty-check and unparseable-location handling)
7. **Line 175 (Phase A step 4):** Expand with finding extraction protocol (in-memory accumulation, triage note, no-findings handling)
8. **Line 194 (Phase A step 8):** Clarify atomic findings array persistence
9. **Line 209 (Phase B step 1):** Explicit state file read with BC-8 null-check for pre-#668 state files
10. **Phase B step numbering:** Fix pre-existing gap (1,2,4,5,6,7,8 → 1,2,3,4,5,6,7)
11. **Line 223 (Phase B step 6):** In-place mutation of history entry with per-finding disposition update

No new files created. No files deleted.
