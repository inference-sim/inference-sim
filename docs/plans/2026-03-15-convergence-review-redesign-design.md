# Convergence-Review Skill: Structural Redesign

**Species:** Specification
**Status:** Draft
**Decision Status:** Proposed
**Closes:** #430
**Author:** Claude + Sri
**Date:** 2026-03-15

---

## Motivation

Users need reliable automated convergence enforcement for review gates. Today, the `convergence-review` skill describes a mandatory re-run loop (dispatch perspectives, tally, fix, re-dispatch), but Claude frequently skips the re-run after fixing issues. This prevents the skill from achieving its purpose: guaranteeing that all CRITICAL and IMPORTANT findings are resolved before proceeding.

Four root causes identified in #430:

1. **No structural loop enforcement** — the algorithm is pseudocode in markdown; the LLM treats it as guidance, not control flow. Evidence: in #430, Claude exited after applying fixes without re-running reviewers. In #390, agents fabricated "0 CRITICAL, 0 IMPORTANT" and the skill accepted without verification.
2. **No persistent state** — no round counter, state file, or continuation detection across invocations. Without state, a re-invoked skill cannot distinguish "first round" from "re-run after fixes." This also causes context dilution: a single round generates thousands of tokens, and the loop instruction is buried by the time fixes are applied. A state file acts as a compact structural reminder.
3. **Ambiguous re-invocation** — the "After fixes" section suggests manual re-invocation, contradicting the algorithm's automatic loop. Two contradictory instructions in the same skill.
4. **Completion bias** — after fixing issues, the LLM biases toward reporting success rather than re-running reviewers. In #413, 4 rounds were needed; without structural enforcement, the skill would have exited after Round 1 fixes.

## Scope

**In scope:** Restructuring the convergence-review skill (SKILL.md) with state-file-driven loop enforcement, triage, and status banners. Minor update to convergence.md.

**Out of scope:** Changes to perspective prompt files (pr-prompts.md, design-prompts.md, hypothesis-experiment/review-prompts.md). Changes to the hypothesis-experiment skill. Multi-agent orchestration changes. Changes to pr-workflow.md.

**Deferred:** Cross-session state merge for parallel reviews. Undo for filed issues. Finding-level IDs for cross-round tracking.

## Modeling Decisions

| Aspect | Treatment | What is lost |
|--------|-----------|-------------|
| Loop enforcement | Modeled — two-phase state machine with persistent state | N/A (core feature) |
| State persistence | Modeled — JSON file on disk, chosen over alternatives (conversation context markers, environment variables, structured prompt injection) for cross-session durability and human inspectability | In-memory speed; atomic writes (partial write = corruption, handled by fallback to round 1) |
| Out-of-scope detection | Simplified — file-change heuristic for code/experiment gates; disabled for document/context gates | Semantic analysis of whether a finding is truly in-scope; false positives for experiment gates that reference simulator source files |
| Fix correctness | Simplified — verification gate catches build/lint failures; no independent semantic validation of fix quality (relies on next-round LLM reviewers to catch semantic regressions) | Automated semantic validation of fix quality |
| Fix classification | Simplified — conservative heuristic (all CRITICALs treated as semantic, requiring fix plan); boundary between mechanical and semantic is LLM judgment for IMPORTANT/SUGGESTION | Intent-aware fix routing with structural enforcement of the mechanical/semantic boundary |
| Concurrent reviews | Omitted — undefined behavior if two sessions review the same artifact | Lock-free concurrent convergence |
| Fix-oscillation detection | Omitted — max 10 rounds is the only circuit breaker | Early detection of A-breaks-B-breaks-A cycles |

## DES Checklist

This design modifies a Claude Code skill, not a simulation module. All DES checklist items from design guidelines Section 2.6 are N/A:

- No new events, state variables, or clock interactions
- No randomness sources
- No simulation invariants affected (INV-1 through INV-11 unchanged)
- No files in `sim/`, `cmd/`, or `testdata/` modified

The SK-INV namespace (Section 5) is for skill-layer behavioral invariants, separate from the DES INV namespace.

## Design

### Overview

Replace the prose-described loop with a **two-phase state-machine** backed by a **persistent state file**:

- **Phase A (Review):** Dispatch all perspectives, collect results, tally findings independently, triage. Three outcomes: Converged (0/0 in-scope critical/important), Not-converged (enter Phase B), Stalled (round > 10).
- **Phase B (Fix-and-Rerun):** Fix items by priority (with confidence-tiered autonomy), run verification gate, update state, re-enter Phase A immediately. No manual re-invocation. No exit except convergence or stall.

```
┌─────────────────────────────────────────────────────────┐
│                    Skill Invocation                      │
│                                                          │
│  Parse args → Load/create state → Check commit anchor    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │    Phase A: Review   │◄──────────────────┐
              │                      │                    │
              │  1. Dispatch N       │                    │
              │     perspectives     │                    │
              │  2. Collect & tally  │                    │
              │  3. Triage findings  │                    │
              │  4. Emit banner      │                    │
              └──────────┬───────────┘                    │
                         │                                │
              ┌──────────┼───────────┐                    │
              │          │           │                    │
              ▼          ▼           ▼                    │
         Converged    Stalled   Not-converged             │
           │            │           │                     │
           ▼            ▼           ▼                     │
      Fix SUGGs    Ask user    ┌──────────────┐           │
      Verify       (abort/     │ Phase B: Fix │           │
      Delete state  reset)     │              │           │
      Exit                     │ 1. Fix CRITs │           │
                               │    (fix plan)│           │
                               │ 2. Fix IMPs  │           │
                               │ 3. Fix SUGGs │           │
                               │ 4. Stage new │           │
                               │    files     │           │
                               │ 5. Verify    │           │
                               │ 6. Update    │           │
                               │    state     │           │
                               │ 7. Banner    │           │
                               └──────┬───────┘           │
                                      │                   │
                                      └───────────────────┘
```

### 1. State File Schema and Lifecycle

**Location:** `.claude/convergence-state/<gate>-<artifact-id>.json`

The state file tracks: which gate is being reviewed, the current round, the outcome of each past round (counts by severity, in-scope counts, any filed issues, per-finding disposition and description), and whether the reviewed artifact has changed since the last round. The artifact ID must be unique across concurrent gates and human-readable.

**Artifact ID derivation rules:**
- **File-anchored gates** (artifact is a file path): derive ID from the file basename without extension, prefixed by gate name.
- **Diff-anchored gates** (artifact is a git diff): derive ID from the current branch name, prefixed by gate name.
- **Context-anchored gates** (artifact is conversation context): derive ID from the branch name, prefixed by gate name.

The micro plan owns the exact derivation algorithm. The behavioral requirement is: no collisions across concurrent gates, human-readable names.

**Gate-to-category table (definitive):**

| Gate | Anchor | Artifact | Triage scope | Verification |
|------|--------|----------|-------------|-------------|
| `design` | File | Design doc at `$1` | Disabled | Link check |
| `macro-plan` | File | Macro plan at `$1` | Disabled | Link check |
| `pr-plan` | File | Micro plan at `$1` | Disabled | Link check |
| `pr-code` | Diff | `git diff HEAD` output | File-change heuristic | Build/test/lint |
| `h-design` | Context | Conversation context | Disabled | None |
| `h-code` | File | `run.sh` + `analyze.py` at `$1` | Change-recommending heuristic | Build/test/lint |
| `h-findings` | File | FINDINGS.md at `$1` | Change-recommending heuristic | Link check |

When a new gate type is added, it must be added to this table with all 5 columns declared.

**Artifact ID collision between workflow steps:** The PR workflow invokes `pr-code` at both Step 2.5 (plan review) and Step 4.5 (code review), both on the same branch. This produces the same artifact ID. The commit anchor disambiguates: code is committed between Steps 2.5 and 4.5, so the stale-commit check resets the state. **Users must commit between successive convergence passes on the same branch.** If Step 4.5 is invoked before committing, the stale state from Step 2.5 (possibly with `converged` status) would cause premature exit.

**State file contents (illustrative, not normative — micro plan owns exact field names):**

```json
{
  "gate": "pr-code",
  "artifact_id": "pr-code-pr19-routing",
  "commit": "a529ff4",
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

Note: `disposition` field on each finding (`fix`, `filed`, or future values like `defer`, `downgrade`) replaces the separate `filed` array from earlier drafts. This supports future triage dispositions without schema changes.

**Status values:** `reviewing` (Phase A in progress), `not-converged` (entering Phase B), `converged` (done), `stalled` (round > max_rounds).

**Schema version handling:** If a state file's `schema_version` is higher than the skill's supported version, treat it as corrupted — delete and start fresh at round 1. This prevents partial parsing of unknown schema formats.

**Lifecycle:**
1. **Created** on first invocation (round 1, status `reviewing`).
2. **Updated** after each Phase A tally (history appended, status set).
3. **Deleted** on convergence (after suggestion cleanup) — no stale state.
4. **Deleted** on stall if user chooses abort. Reset round counter if user chooses continue.
5. **Stalled files persist** until user decides — they document why the gate didn't converge.

**Commit anchoring** follows the anchor category from the gate table:
- **File-anchored and diff-anchored gates** record `git rev-parse --short HEAD`. On re-entry, if stored commit differs from current HEAD, the state is stale — reset to round 1 with a log message. (Note: commit amend also changes HEAD and triggers reset. This is conservative but correct — the reviewed artifact has changed.)
- **Context-anchored gates** record no commit (`null`). State validity relies on session continuity and `updated_at` timestamp. Cross-session re-entry loads the state as-is — the user is responsible for ensuring the design context is still relevant.

**Error handling:** If a state file exists but fails to parse (corrupted JSON, missing required fields, unsupported schema version), treat it as "no file exists" — log a warning, delete the corrupted file, start fresh at round 1.

**Gitignore:** Add `.claude/convergence-state/` to `.gitignore` (session-local state, not committed).

### 2. Phase A — Review and Tally

**Entry conditions (checked in order):**

1. **Parse args:** The skill accepts an optional `--model` flag that selects the review model (valid values: `opus`, `sonnet`, `haiku`; invalid values rejected). The remaining tokens specify the gate type and artifact path.

2. **Load or create state:**
   - **No file exists (or corrupted/unsupported version):** Create new state (round 1, status `reviewing`). Proceed to dispatch.
   - **File exists, commit matches HEAD (or context-anchored gate):** Resume. If status is `not-converged`, increment round. If status is `reviewing` (interrupted mid-Phase-A), resume Phase A from dispatch without incrementing round. For context-anchored gates, if `updated_at` is older than 24 hours, emit a staleness warning ("State file is N hours old — design context may have changed") and ask user whether to continue or reset. Proceed to dispatch.
   - **File exists, commit differs from HEAD:** Stale. Log: `"State from commit <old> invalidated by HEAD <new>. Starting Round 1."` Delete old file, create fresh state. Proceed to dispatch.
   - **File exists, status `converged`:** Emit `"Gate <gate> already converged in Round N. Nothing to do."` Exit.
   - **File exists, status `stalled`:** Emit `"Gate <gate> stalled after N rounds."` Ask user: abort (delete state, exit) or reset (reset round counter, proceed).

3. **`--model` handling on re-entry:** If `--model` is explicitly provided, use it and update state file. If omitted, use the value from state file (not the default `haiku`). First invocation without `--model` defaults to `haiku`.

4. **Dispatch all N perspectives** simultaneously as background agents. Model from state. Exception: the structural validation perspective in PR plan reviews is performed directly (not delegated to an agent) because it requires full conversation context.
   - **Empty-diff precondition** (diff-anchored gates only): If `git diff HEAD` produces no output, emit a warning ("No changes detected since last commit — nothing to review") and skip dispatch. This prevents vacuous reviewer output and false convergence. The user should commit their changes or check working tree state.

5. **Collect and tally** independently. Read each agent's output, extract findings with severity, count CRITICAL and IMPORTANT yourself. Never trust agent-reported totals (per #390).

6. **State file consistency check:** Before recording results, verify: if tallied counts show `in_scope.critical + in_scope.important > 0`, status must be `not-converged`. If both are 0, status must be `converged`. If the check detects inconsistency, the count-derived status takes precedence **and a visible warning is emitted** (e.g., "SK-INV-4 mismatch: counts show 2 IMPORTANT but status was converged — correcting to not-converged"). The warning surfaces systematic tally bugs that would otherwise be silently corrected every round.

7. **Triage:**
   - All findings default to **Fix** (in-scope). Phase B proceeds automatically.
   - **Out-of-scope detection** follows the gate table's "Triage scope" column:
     - **Disabled** (document gates, context gates): All findings are in-scope. Document reviews naturally reference files across the codebase.
     - **File-change heuristic** (diff-anchored gates): If a finding references a file not in `git diff HEAD`, flag it.
     - **Change-recommending heuristic** (experiment gates): If a finding recommends *changing* a file outside the experiment directory, flag it. Mere *references* to simulator source files are not flagged — experiment findings routinely reference source code. Examples: "sim/kv/cache.go should add a nil check" → flagged (recommends change). "sim/kv/cache.go exhibits O(n^2) behavior under this workload" → not flagged (observation only).
   - When out-of-scope findings are flagged, present them as a **single batch prompt** (not N sequential prompts):
     ```
     2 findings reference files outside the current diff:
     1. [IMPORTANT] PC-3: tiered.go — pre-existing race condition
     2. [IMPORTANT] PC-9: config.go — exported mutable map
     Default: Fix all. Enter numbers to file as issues instead (e.g., "1,2"):
     ```
   - CRITICAL items in the batch get a warning: `"(CRITICAL — presumed in-scope; file only if certain this is pre-existing)"`
   - If the user's response cannot be parsed (e.g., invalid format), default to Fix-all and emit a warning. Do not block the loop on triage parse failure.
   - Items marked **File** are filed as GitHub issues immediately (following the project's issue-filing conventions in `docs/contributing/pr-workflow.md`), recorded in state history with disposition `filed` and issue number, and excluded from convergence check.

8. **Convergence check** uses post-triage in-scope counts: `in_scope.critical == 0 AND in_scope.important == 0`.

9. **Update state file:** Append round entry to `history[]`, set status, write `updated_at`. For diff/file-anchored gates, record current HEAD.

10. **Emit status banner** (exact formatting is illustrative — micro plan owns the rendering).

11. **Branch:**
    - **Converged:** Fix suggestions (if any, see Section 4), run verification, delete state file, exit.
    - **Stalled:** Keep state file, present round-over-round trend (are the same findings recurring or new ones each round?), ask user: abort or reset.
    - **Not converged:** Enter Phase B immediately. No exit, no pause.

### 3. Phase B — Fix and Rerun

Phase B is entered automatically from Phase A when in-scope CRITICAL or IMPORTANT findings exist.

1. **List all findings to fix**, grouped by priority.

2. **Fix with confidence-tiered autonomy:**
   - **All CRITICAL fixes** get a fix plan: emit the proposed changes as output text, then **stop and wait for the user's next message** before applying (this is the only way to implement "acknowledgment" in the Claude Code execution model). The user may: (a) approve — apply the fix, (b) provide an alternative fix — apply the user's version, (c) file as issue — record with disposition `filed` and exclude from convergence, (d) downgrade to SUGGESTION — remove from blocking count. CRITICALs are high-stakes by definition and the LLM could apply a subtly wrong semantic fix that passes build/test/lint.
   - **IMPORTANT and SUGGESTION fixes**: auto-fix without pause. These are lower-risk. The next Phase A round catches any semantic regressions introduced by auto-fixes.
   - The boundary between "mechanical" and "semantic" within IMPORTANT/SUGGESTION is LLM judgment, not structural enforcement. This is an accepted limitation — see Modeling Decisions table.
   - **Rejected alternatives for CRITICAL fix autonomy:** (a) Auto-fix all CRITICALs with rollback on verification failure — rejected because verification only catches syntactic errors, not semantic regressions. (b) Per-finding user choice for all severities — rejected as too slow; IMPORTANT/SUGGESTION are lower-risk and the next round catches regressions. (c) Nature-based classification (mechanical vs. semantic) — rejected because the boundary is not structurally enforceable; the conservative "all CRITICALs pause" is simpler and safer.

3. **Fix all items** in priority order: CRITICAL first, then IMPORTANT, then SUGGESTION.

4. **Stage any new files** created by fixes so they are visible to the next Phase A round's diff. Do not stage modifications to existing tracked files (those are visible automatically).

5. **Run verification gate** per the gate table's "Verification" column:
   - **Build/test/lint** (code gates): run the project's CI verification. If any fail, fix before proceeding.
   - **Link check** (document gates): verify referenced files exist, check for broken internal links.
   - **None** (context gates): no verification gate.

6. **Update state file:** Record fixes in history (with disposition `fix` for each finding), set status to `not-converged`.

7. **Emit transition banner.**

8. **Re-enter Phase A immediately.** No pause, no user prompt, no manual re-invocation.

**No git commit in Phase B.** Fixes accumulate as working tree changes. The PR workflow's Step 5 handles the single commit for the entire PR. The state file's `history` array is the audit trail for what was fixed in each round.

**Multi-session caveat:** If the session ends mid-convergence, the working tree preserves all fixes on disk, and the state file records round history. On session resumption, the skill re-reads the state file and continues from the last recorded status. However, the LLM has no memory of specific fixes applied in the previous session — it relies on the state file history and the current diff. If a previous session's fix was partial (file saved incompletely), the next Phase A round will catch it.

**Only exits from the loop:**
- Phase A: `converged` (0/0 in-scope)
- Phase A: `stalled` (round > 10)
- Phase B: verification gate failure that can't be resolved (ask user)

### 4. Suggestion Cleanup on Convergence

When Phase A tallies 0 CRITICAL, 0 IMPORTANT, but N > 0 SUGGESTION:

1. Emit the CONVERGED banner.
2. List suggestion items.
3. Fix all suggestions.
4. Run verification gate (per gate table).
5. Delete state file.
6. Exit skill.

No re-run triggered. Suggestions are cosmetic by definition.

### 5. Behavioral Invariants

The redesigned skill must maintain these properties (SK-INV = skill invariant, separate from DES invariants INV-1 through INV-11):

- **SK-INV-1 Loop integrity:** The skill must never exit with status `not-converged`. The only exits are `converged` (0/0 in-scope), `stalled` (round > max_rounds), or unresolvable verification failure (user decision).
- **SK-INV-2 Round monotonicity:** The round counter in the state file must never decrease (except on stale-commit reset, which creates a new state file).
- **SK-INV-3 Tally independence:** The skill must count findings from agent output independently. It must never use agent-reported totals as the convergence input.
- **SK-INV-4 State-status consistency:** If `in_scope.critical + in_scope.important > 0`, status must be `not-converged` or `stalled`. If both are 0, status must be `converged`. If the consistency check detects a mismatch, the count-derived status takes precedence.
- **SK-INV-5 Stale invalidation:** If the stored commit differs from current HEAD (for file/diff-anchored gates), the state must be reset to round 1. No resumption from stale state.

### 6. Extension Points

- **New gate types:** Add a row to the gate-to-category table (Section 1) with all 5 columns, add a new section to the appropriate prompts file, and add it to the gate table in SKILL.md. State schema, Phase A/B logic, and convergence protocol are gate-agnostic. Perspective set changes mid-convergence require a state reset (analogous to commit anchor invalidation).
- **New triage dispositions:** Currently Fix or File. Future options: Defer (track but don't fix this round), Downgrade (reclassify severity). The per-finding `disposition` field in the state schema accommodates new values without schema changes.
- **Alternative verification gates:** Currently Go-specific for code gates. Future: per-gate verification commands, or a project-level configuration. This extension point is deferred to a future design.
- **Custom perspective sets:** Currently fixed per gate type. Future: allow users to specify a subset of perspectives. This extension point is deferred to a future design.

### 7. Files Changed

**Modified:**
- **`.claude/skills/convergence-review/SKILL.md`** — Major rewrite. Pseudocode loop replaced with Phase A/B state machine. State file read/write instructions. Triage step. Status banners. "After fixes / Re-invoke" section deleted (S4).
- **`docs/contributing/convergence.md`** — Minor update. Add note that `convergence-review` skill now enforces the loop automatically via state files. No protocol rule changes.
- **`.gitignore`** — Add `.claude/convergence-state/` entry.

**Created:**
- **`.claude/convergence-state/`** — Runtime directory created on first use by the skill (not a source change).

**Unchanged:**
- `pr-prompts.md`, `design-prompts.md`, `hypothesis-experiment/review-prompts.md` — perspective prompts untouched.
- `docs/contributing/pr-workflow.md` — already says "run convergence-review" at Steps 2.5 and 4.5; the skill now handles the loop internally.

### 8. Design Decisions

| Decision | Choice | Status | Rationale | What breaks if wrong |
|----------|--------|--------|-----------|---------------------|
| State location | `.claude/convergence-state/` | Proposed | Gitignored, alongside other Claude state | State files committed accidentally; need .gitignore entry |
| State format | JSON file on disk | Proposed | Cross-session durable, human-inspectable, simple to parse | Partial writes cause corruption (mitigated by fallback to round 1) |
| State keying | `<gate>-<artifact-id>.json` | Proposed | Human-readable filenames, no collisions | Collisions between reviews of different artifacts with same basename |
| Schema version | Include `schema_version` field; unknown version → delete and restart | Proposed | Forward-compatible migration; clean failure on version mismatch | Extra field in every state file (negligible cost) |
| Commit anchoring | HEAD hash for file/diff gates | Proposed | Invalidates stale state after code changes | Commit amend mid-convergence resets round history (conservative but safe) |
| Commit skip | Null for context-anchored gates | Proposed | No file to anchor; rule-based, not per-gate special-case | Stale context-gate state loaded in new session (accepted risk) |
| Model persistence | Stored in state, overridable; round count not reset on model change | Proposed | Consistent across rounds; explicit `--model` overrides stored value | Model upgrade mid-convergence finds new issues invisible to cheaper model (accepted — artifact unchanged, round budget shared) |
| Max rounds | 10 (match convergence.md) | Proposed | Protocol parity, no divergence from canonical source | 10 rounds may be excessive for simple reviews; accepted since convergence.md is authoritative |
| Fix autonomy | All CRITICALs get fix plan (user may approve, provide alternative, file, or downgrade); IMPORTANT/SUGGESTION auto-fix | Proposed | Conservative: CRITICALs are high-stakes. Alternatives rejected: all-auto-fix (no semantic safety), per-finding-choice (too slow), nature-based-classification (not structurally enforceable) | Slight slowdown for CRITICAL fixes; if most CRITICALs are mechanical, the pause is unnecessary but safe. Misclassified semantic IMPORTANT auto-fixed (mitigated by next Phase A round) |
| Triage default | All findings default to Fix | Proposed | Auto-proceeds for majority case | False-positive findings get auto-fixed (mitigated by verification gate + next round) |
| Triage batching | Single batch prompt for all out-of-scope findings | Proposed | One prompt, not N sequential prompts | User must parse multiple items at once (acceptable trade-off for speed) |
| CRITICAL filing | Requires explicit user confirmation in batch prompt | Proposed | CRITICALs presumed in-scope | Out-of-scope CRITICAL auto-fixed unnecessarily (low risk — CRITICALs are rarely out-of-scope) |
| Suggestions | Fixed but don't block or trigger rerun | Proposed | Prevents accumulation across rounds | Suggestion fix introduces new issue (caught by verification gate) |
| Phase B commits | None (no auto-commit) | Proposed | PR workflow Step 5 handles single commit; state history is audit trail | Multi-session: working tree state unknown on resumption (mitigated by state file + next Phase A catching issues) |
| Verification gate | Gate-table-driven | Proposed | Build/test/lint for code; link check for docs; none for context | Document verification too shallow (accepted — document reviews are about content, not links) |
| Stalled UX | User choice with round-over-round trend | Proposed | Trend info helps user decide reset vs abort | User overloaded with history data (mitigated by summary format) |
| Re-invocation section | Deleted from SKILL.md | Proposed | S4 from issue — removes contradictory escape hatch | No way to manually trigger "just one more round" (accepted — the loop handles this) |

### 9. Verification and Validation

**Success criteria:**
1. Given a review round with CRITICAL findings, the skill must always re-enter Phase A after Phase B without manual re-invocation. Observable: state file `history` length increments after each Phase A dispatch, and no `not-converged` status persists at session end without a subsequent Phase A entry.
2. The skill must never exit with status `not-converged` unless round > max_rounds or user explicitly aborts.
3. State file must be consistent with actual convergence status at all times (SK-INV-4).

**Verification plan:**
- **Invariant trace-through:** After implementation, manually trace through each Phase A branch (converged, stalled, not-converged, reviewing-interrupted) and each Phase B step. Specific transitions to verify: (a) Phase A not-converged at round 9 → Phase B → Phase A round 10 → stall. (b) Phase A converged with 3 suggestions → suggestion cleanup → delete state. (c) Corrupted state file → fallback to round 1.
- **State file parsing:** Test corrupted JSON, missing fields, stale commit, unsupported schema version — confirm graceful fallback to round 1 in all cases.

**Validation plan (regression against historical failures):**
- **#390 replay:** Inject mock agent output where agent claims "0 CRITICAL" but output text contains 2 CRITICAL findings. Pass: skill's independent tally reads 2, not 0. SK-INV-3 holds.
- **#413 replay:** Run convergence against a known-bad artifact requiring multiple rounds. Pass: state file `history` has >1 entry, final entry shows `in_scope.critical == 0 && in_scope.important == 0`, and the Phase A/B loop executed without manual re-invocation. (Exact round count may vary due to LLM non-determinism; the key property is multi-round automatic convergence.)
- **#430 scenario:** Run convergence, verify IMPORTANT findings exist after Phase A, confirm Phase B entered automatically. Pass: state file shows `not-converged` after Phase A, then a subsequent Phase A entry in the same session. Fail: session ends with `not-converged` and no subsequent Phase A.
- **Mid-Phase-B interruption:** Start convergence, interrupt session after Phase B fixes are partially applied (e.g., 2 of 4 files fixed). Resume in new session. Pass: skill reads state file, enters Phase A, reviewers catch the 2 unfixed items. This validates the multi-session caveat claim.

Note: since skill behavior is non-deterministic (LLM responses vary), these replays validate the state machine transitions and state file integrity, not exact output. The falsification conditions below address the non-determinism risk.

**Falsification conditions:**
- If the LLM ignores the Phase B re-entry instruction despite reading the state file showing `not-converged`, the state-machine approach has failed. Fallback: external orchestrator or hard-coded tool that enforces the loop outside the skill's markdown.
- If the LLM writes incorrect status to the state file (e.g., `converged` with non-zero blocking counts), SK-INV-4's consistency check should catch it. If that check is also ignored, the fundamental assumption that LLMs follow explicit data-driven instructions is violated.
