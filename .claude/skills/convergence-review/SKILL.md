---
name: convergence-review
description: Dispatch parallel review perspectives and enforce convergence via persistent state file and automatic Phase A/B loop. Supports 7 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10). Automatically loops through Phase A (review/tally) and Phase B (fix/verify) until 0 CRITICAL + 0 IMPORTANT or round limit.
argument-hint: <gate-type> [artifact-path] [--model opus|sonnet|haiku]
---

# Convergence Review Dispatcher

Dispatch parallel review perspectives for gate **$0** and enforce convergence via a two-phase state machine with persistent state.

## Model Selection

The skill accepts an optional `--model` flag that selects the review model. Valid values: `opus`, `sonnet`, `haiku`. Default: **`haiku`**. Invalid values are rejected with an error message.

```
/convergence-review pr-code                    # uses haiku (default)
/convergence-review pr-plan plan.md --model sonnet
/convergence-review h-findings FINDINGS.md --model opus
```

`REVIEW_MODEL` stores the short name. The Agent tool resolves these to the appropriate model version (e.g., `opus` → `claude-opus-4-6`).

**On re-entry (resuming from a state file):** If `--model` is explicitly provided, use it and update the state file. If omitted, use the value from the state file (not the default `haiku`).

## Gate Types

| Gate | Anchor | Artifact | Triage Scope | Verification | Perspectives | Prompts Source |
|------|--------|----------|-------------|-------------|-------------|----------------|
| `design` | File | Design doc at `$1` | Disabled | Link check | 8 | [design-prompts.md](design-prompts.md) Section A |
| `macro-plan` | File | Macro plan at `$1` | Disabled | Link check | 8 | [design-prompts.md](design-prompts.md) Section B |
| `pr-plan` | File | Micro plan at `$1` | Disabled | Link check | 10 | [pr-prompts.md](pr-prompts.md) Section A |
| `pr-code` | Diff | `git diff HEAD` output | File-change heuristic | Build/test/lint | 10 | [pr-prompts.md](pr-prompts.md) Section B |
| `h-design` | Context | Conversation context | Disabled | None | 5 | `.claude/skills/hypothesis-experiment/review-prompts.md` Section A |
| `h-code` | File | `run.sh` + `analyze.py` at `$1` | Change-recommending | Build/test/lint | 5 | `.claude/skills/hypothesis-experiment/review-prompts.md` Section B |
| `h-findings` | File | FINDINGS.md at `$1` | Change-recommending | Link check | 10 | `.claude/skills/hypothesis-experiment/review-prompts.md` Section C |

When a new gate type is added, it MUST be added to this table with all 7 columns declared.

**Anchor categories:**
- **File-anchored:** Artifact is a file path. State keyed by `<gate>-<basename-without-extension>`. Commit anchor: `git rev-parse HEAD` (full SHA).
- **Diff-anchored:** Artifact is `git diff HEAD`. State keyed by `<gate>-<branch-name>`. Commit anchor: `git rev-parse HEAD` (full SHA).
- **Context-anchored:** Artifact is conversation context. State keyed by `<gate>-<branch-name>`. No commit anchor (null). Staleness based on `updated_at` timestamp (24-hour warning).

**Triage scope categories:**
- **Disabled:** All findings are in-scope. Document and context reviews naturally reference files across the codebase.
- **File-change heuristic:** If a finding references a file not in `git diff HEAD`, flag it as potentially out-of-scope. Note: `git diff HEAD` includes staged and unstaged changes but not untracked files — stage new files before invoking.
- **Change-recommending:** If a finding recommends *changing* a file outside the experiment directory, flag it. Mere *references* are not flagged. Examples: "sim/kv/cache.go should add a nil check" → flagged. "sim/kv/cache.go exhibits O(n²) behavior" → not flagged.

**Verification categories:**
- **Build/test/lint:** Run the project's CI verification (`go build ./...`, `go test ./... -count=1`, `golangci-lint run ./...`).
- **Link check:** Verify referenced files exist, check for broken internal links.
- **None:** No verification gate (context-based gates with no file artifact).

---

## State File

**Location:** `.claude/convergence-state/<gate>-<artifact-id>.json`

The state file persists convergence state across the Phase A/B loop and across session boundaries. It tracks the current round, model, findings history, and convergence status.

The artifact ID is derived from the gate's anchor category (see gate table). Behavioral requirement: no collisions across concurrent gates, human-readable names.

**Artifact ID collision between workflow steps:** The PR workflow invokes `pr-code` at both Step 2.5 and Step 4.5 on the same branch. The commit anchor disambiguates: code is committed between steps, so the stale-commit check resets the state. **Users must commit between successive convergence passes on the same branch.** If Step 4.5 is invoked before committing, stale state from Step 2.5 (possibly `converged`) would cause premature exit.

**Schema (illustrative, not normative — exact field names owned by implementation):**

```json
{
  "gate": "pr-code",
  "artifact_id": "pr-code-pr19-routing",
  "commit": "a529ff4e3b1c2d4e5f6a7b8c9d0e1f2a3b4c5d6e",
  "schema_version": 1,
  "round": 2,
  "max_rounds": 10,
  "model": "haiku",
  "history": [
    {
      "round": 1,
      "critical": 2,
      "important": 5,
      "suggestion": 3,
      "in_scope": {"critical": 2, "important": 3},
      "findings": [
        {"perspective": "PC-1", "severity": "CRITICAL", "disposition": "fix", "description": "..."},
        {"perspective": "PC-3", "severity": "IMPORTANT", "disposition": "filed", "issue": "#692", "description": "..."}
      ]
    }
  ],
  "status": "not-converged",
  "updated_at": "2026-03-15T10:30:00Z"
}
```

**Status values:** `reviewing` (Phase A in progress), `not-converged` (entering Phase B), `converged` (done), `stalled` (round > max_rounds).

**Schema version handling:** If `schema_version` is higher than the skill's supported version, treat as corrupted — delete and start fresh at round 1.

**Lifecycle:**
1. **Created** on first invocation (round 1, status `reviewing`).
2. **Updated** after each Phase A tally (history appended, status set).
3. **Deleted** on convergence (after suggestion cleanup) — no stale state.
4. **Deleted** on stall if user chooses abort. Reset round counter if user chooses continue.

**Commit anchoring** follows the anchor category from the gate table:
- **File/diff-anchored gates:** Record `git rev-parse HEAD` (full SHA). On re-entry, if stored commit differs from current HEAD → stale. Reset to round 1 with log message. (Commit amend also changes HEAD and triggers reset — conservative but correct.)
- **Context-anchored gates:** Record no commit (`null`). If `updated_at` is older than 24 hours, emit staleness warning and ask user: continue or reset.

**Error handling:** If a state file exists but fails to parse (corrupted JSON, missing required fields, `schema_version` higher than supported), treat as "no file exists" — log a warning, delete the file, start fresh at round 1.

**Gitignore:** `.claude/convergence-state/` is gitignored (session-local state, not committed).

---

## Convergence Protocol (non-negotiable)

> **Canonical source:** [docs/contributing/convergence.md](../../../docs/contributing/convergence.md). The rules below are a self-contained copy for skill execution; convergence.md is authoritative if they diverge.

These rules are identical across all gates. No exceptions. No shortcuts.

### Hard Rules

1. **Zero means zero.** One CRITICAL from one reviewer = not converged.
2. **Re-run is mandatory.** After fixing issues, Phase B always re-enters Phase A. The state machine enforces this — no manual re-invocation needed.
3. **SUGGESTION items do not block.** Only CRITICAL and IMPORTANT count for convergence. Suggestions are fixed but don't trigger re-runs.
4. **Independent tallying.** Read each agent's output. Count findings yourself. Agents have fabricated "0 CRITICAL, 0 IMPORTANT" when actual output contained 3 CRITICAL + 18 IMPORTANT (#390). **Never use agent-reported totals.**
5. **No partial re-runs.** Re-run ALL perspectives, not just the ones that found issues. Fixes can introduce new issues in other perspectives.
6. **Agent timeout = 5 minutes.** If an agent exceeds this, check its output and restart. If it fails, perform that review directly.
7. **Max 10 rounds per gate.** If still not converged, stall and ask user.

### Severity Classification

| Severity | Definition | Blocks? |
|----------|-----------|---------|
| **CRITICAL** | Must fix. Missing control experiment, status contradicted by data, silent data loss, cross-document contradiction. | Yes |
| **IMPORTANT** | Should fix. Fixing would change a conclusion, metric, or user guidance. | Yes |
| **SUGGESTION** | Cosmetic. Off-by-one line citation, style consistency, terminology nit. | No |

**When in doubt:** If fixing it would change any conclusion → IMPORTANT. If only readability → SUGGESTION.

### Behavioral Invariants

The skill MUST maintain these properties (SK-INV = skill invariant, separate from DES invariants INV-1 through INV-11):

- **SK-INV-1 Loop integrity:** The skill MUST never exit with status `not-converged`. The only exits are `converged` (0/0 in-scope), `stalled` (round > 10), or unresolvable verification failure (user decision).
- **SK-INV-2 Round monotonicity:** The round counter MUST never decrease within a single state file lifetime (stale-commit reset creates a new state file).
- **SK-INV-3 Tally independence:** The skill MUST count findings from agent output independently. It MUST never use agent-reported totals as the convergence input.
- **SK-INV-4 State-status consistency:** If `in_scope.critical + in_scope.important > 0`, status MUST be `not-converged` or `stalled`. If both are 0, status MUST be `converged`. If the consistency check detects a mismatch, the count-derived status takes precedence and a **visible warning** is emitted (e.g., "SK-INV-4 mismatch: counts show 2 IMPORTANT but status was converged — correcting to not-converged").
- **SK-INV-5 Stale invalidation:** If the stored commit differs from current HEAD (for file/diff-anchored gates), the state MUST be reset to round 1.

---

## Phase A — Review and Tally

**Entry conditions (checked in order):**

1. **Parse args:** Strip `--model <value>` if present, validate. Remaining tokens → gate type and artifact path.

2. **Load or create state:**
   - **No file exists (or corrupted/unsupported version):** Create new state (round 1, status `reviewing`). Proceed to dispatch.
   - **File exists, commit matches HEAD (or context-anchored gate), status `not-converged`:** Increment round (immediately persist) and proceed to dispatch.
   - **File exists, commit matches HEAD (or context-anchored gate), status `reviewing`:** Resume Phase A from dispatch WITHOUT incrementing round (interrupted mid-Phase-A replay). Proceed to dispatch.
   - **File exists, commit differs from HEAD (file/diff-anchored gates):** Stale. Log: `"State from commit <old> invalidated by HEAD <new>. Starting Round 1."` Delete old file, create fresh state. Proceed to dispatch.
   - **File exists, status `converged`:** Emit `"Gate <gate> already converged in Round N. Nothing to do."` Exit.
   - **File exists, status `stalled`:** Emit `"Gate <gate> stalled after N rounds."` Ask user: abort (delete state, exit) or reset (reset round counter, proceed).
   - **Context-anchored gate, `updated_at` older than 24 hours:** Emit staleness warning, ask user: continue or reset.

3. **Dispatch all N perspectives** simultaneously as background agents. Model from state file.
   - **Exception:** The structural validation perspective in PR plan reviews is performed directly (not delegated to an agent) because it requires full conversation context.
   - **Empty-diff precondition (diff-anchored gates only):** If `git diff HEAD` produces no output, emit warning ("No changes detected since last commit — nothing to review") and skip dispatch. Stage new files or commit before invoking.

4. **Collect and tally independently.** Read each agent's output. Extract findings with severity. Count CRITICAL and IMPORTANT yourself. **Never trust agent-reported totals** (per #390).

5. **State file consistency check:** Before recording results, verify in-scope counts match the intended status. If inconsistent, count-derived status takes precedence and a **visible warning** is emitted (SK-INV-4).

6. **Triage.** All findings default to **Fix** (in-scope). Phase B proceeds automatically.
   - **Out-of-scope detection** follows the gate table's "Triage Scope" column.
   - When out-of-scope findings are flagged, present as a **single batch prompt**:
     ```
     N findings reference files outside the current scope:
     1. [IMPORTANT] PC-3: tiered.go — pre-existing race condition
     2. [IMPORTANT] PC-9: config.go — exported mutable map
     Default: Fix all. Enter numbers to file as issues instead (e.g., "1,2"):
     ```
   - CRITICAL items get warning: `"(CRITICAL — presumed in-scope; file only if certain this is pre-existing)"`
   - **If user response cannot be parsed, default to File-all** for flagged items and emit a warning.
   - Items marked **File:** filed as GitHub issues immediately (following `docs/contributing/pr-workflow.md` conventions), recorded in state history with disposition `filed` and issue number, excluded from convergence check.

7. **Convergence check:** `in_scope.critical == 0 AND in_scope.important == 0`.

8. **Update state file:** Append round entry to history, set status, write `updated_at`. For file/diff-anchored gates, record full SHA (`git rev-parse HEAD`).

9. **Emit status banner** (exact formatting is illustrative — implementation owns the rendering).

10. **Branch:**
    - **Converged:** Fix suggestions if any (see Suggestion Cleanup below), run verification, delete state file, exit.
    - **Stalled:** Keep state file, present round-over-round trend, ask user: abort or reset.
    - **Not converged:** Enter Phase B immediately. No exit, no pause, no re-invocation needed.

---

## Phase B — Fix and Rerun

Phase B is entered automatically from Phase A when in-scope CRITICAL or IMPORTANT findings exist. **No manual re-invocation needed.**

1. **List all findings to fix**, grouped by priority.

2. **Fix with confidence-tiered autonomy:**
   - **All CRITICAL fixes:** Emit the proposed changes as output text, then **stop and wait for the user's next message** before applying. The user may: (a) approve — apply the fix, (b) provide an alternative fix — apply the user's version, (c) file as issue — record with disposition `filed` and exclude from convergence, (d) downgrade to SUGGESTION — remove from blocking count. **After the user responds, continue Phase B with the remaining items.** The state file's `not-converged` status ensures the loop resumes even if the session is interrupted during a CRITICAL fix pause.
   - **IMPORTANT and SUGGESTION fixes:** Auto-fix without pause. The next Phase A round catches any semantic regressions.

3. **Fix all items** in priority order: CRITICAL first, then IMPORTANT, then SUGGESTION.

4. **Stage any new files** created by fixes so they are visible to the next Phase A round's diff.

5. **Run verification gate** per the gate table's "Verification" column:
   - **Build/test/lint:** Run the project's CI verification. If any fail, fix before proceeding.
   - **Link check:** Verify referenced files exist, check for broken internal links.
   - **None:** No verification gate.

6. **Update state file:** Record fixes in history (with disposition for each finding), set status to `not-converged`.

7. **Emit transition banner:**
   ```
   ═══════════════════════════════════════════════════
     ROUND N FIXES COMPLETE — M items resolved
     Re-entering Phase A for Round N+1...
   ═══════════════════════════════════════════════════
   ```

8. **Re-enter Phase A immediately.** No pause, no user prompt, no manual re-invocation. This is the structural enforcement that prevents the loop from breaking.

**No git commit in Phase B.** Fixes accumulate as working tree changes. The PR workflow's Step 5 handles the single commit. The state file's `history` array is the audit trail.

**Multi-session caveat:** If the session ends mid-convergence, the working tree preserves fixes on disk and the state file records history. On session resumption, the skill re-reads the state file and continues from the last recorded status. The next Phase A round catches any incomplete fixes.

**Only exits from the loop:**
- Phase A: `converged` (0/0 in-scope)
- Phase A: `stalled` (round > 10)
- Phase B: verification gate failure that can't be resolved (ask user)

---

## Suggestion Cleanup on Convergence

When Phase A tallies 0 CRITICAL, 0 IMPORTANT, but N > 0 SUGGESTION:

1. Emit the CONVERGED banner.
2. List suggestion items.
3. Fix all suggestions.
4. Run verification gate (per gate table). If verification fails after a suggestion fix, ask the user: revert the suggestion fix and exit clean, or fix the verification issue and continue.
5. Delete state file.
6. Exit skill.

No re-run triggered. Suggestions are cosmetic by definition.

---

## Integration with Other Skills

### From design process
```
/convergence-review design docs/plans/archive/<design-doc>.md
```

### From macro planning process
```
/convergence-review macro-plan docs/plans/<macro-plan>.md
```

### From PR workflow (Steps 2.5 and 4.5)
```
/convergence-review pr-plan docs/plans/pr<N>-<name>-plan.md
/convergence-review pr-code
```

### From hypothesis-experiment skill (Steps 2, 5, 8)
```
/convergence-review h-design
/convergence-review h-code hypotheses/h-<name>/
/convergence-review h-findings hypotheses/h-<name>/FINDINGS.md
```
