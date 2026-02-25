---
name: convergence-review
description: Dispatch parallel review perspectives and enforce convergence (zero CRITICAL + zero IMPORTANT). Supports 7 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).
argument-hint: <gate-type> [artifact-path]
---

# Convergence Review Dispatcher

Dispatch parallel review perspectives for gate **$0** and enforce convergence.

## Gate Types

| Gate | Perspectives | Artifact | Prompts Source |
|------|-------------|----------|----------------|
| `design` | 8 | Design doc at `$1` | [design-prompts.md](design-prompts.md) Section A |
| `macro-plan` | 8 | Macro plan at `$1` | [design-prompts.md](design-prompts.md) Section B |
| `pr-plan` | 10 | Micro plan at `$1` | [pr-prompts.md](pr-prompts.md) Section A |
| `pr-code` | 10 | Current git diff | [pr-prompts.md](pr-prompts.md) Section B |
| `h-design` | 5 | Design from conversation context | `.claude/skills/hypothesis-experiment/review-prompts.md` Section A |
| `h-code` | 5 | `run.sh` + `analyze.py` at `$1` | `.claude/skills/hypothesis-experiment/review-prompts.md` Section B |
| `h-findings` | 10 | FINDINGS.md at `$1` | `.claude/skills/hypothesis-experiment/review-prompts.md` Section C |

---

## Convergence State Management

### State Directory

State files live in `.claude/convergence-state/`, keyed by gate type and artifact identity. This supports concurrent sessions (different terminals reviewing different artifacts) and sequential sessions without collision.

**State directory layout:**

```
.claude/convergence-state/
  pr-code--feature-scorer-framework.json
  design--docs-plans-archive-kv-tiered-design.md.json
  pr-plan--docs-plans-pr19-scorer-plan.md.json
```

**Filename:** `{gate}--{artifact-id-with-slashes-replaced-by-dashes}.json`

**Artifact identity by gate type:**

| Gate type | Artifact ID | Example |
|-----------|------------|---------|
| `pr-code` | Branch name | `feature/scorer-framework` |
| `pr-plan` | Micro-plan file path | `docs/plans/pr19-scorer-plan.md` |
| `macro-plan` | Macro-plan file path | `docs/plans/weighted-scoring-macro-plan.md` |
| `design` | Design doc file path | `docs/plans/archive/kv-tiered-design.md` |
| `h-design` | Hypothesis doc file path | `hypotheses/h-cross-model/HYPOTHESIS.md` |
| `h-code` | Branch name | `hypothesis/h-cross-model` |
| `h-findings` | FINDINGS.md file path | `hypotheses/h-cross-model/FINDINGS.md` |

For diff-based gates (`pr-code`, `h-code`), the branch name is used — the diff content changes between rounds but the branch is stable.

**State file schema:**

```json
{
  "gate": "pr-code",
  "artifact_id": "feature/scorer-framework",
  "round": 2,
  "max_rounds": 10,
  "history": [
    {"round": 1, "critical": 2, "important": 5, "suggestions": 3, "status": "fixed"}
  ],
  "status": "awaiting-rerun",
  "updated_at": "2026-02-25T14:30:00Z"
}
```

**Status values:** `"awaiting-rerun"` | `"converged"` | `"stalled"`

### State File Operations

On invocation, determine the artifact ID from the gate type:
- `pr-code` / `h-code`: Run `git branch --show-current` to get the branch name
- All other gates: Use the `$1` argument (artifact file path)

Then derive the state file path:
```
.claude/convergence-state/{gate}--{artifact_id with / replaced by -}.json
```

Read the state file. If it does not exist or `status` is `"converged"` or `"stalled"`, enter **Phase A**. If `status` is `"awaiting-rerun"`, enter **Phase B**.

---

## Two-Phase Convergence Protocol (non-negotiable)

> **Canonical source:** [docs/process/hypothesis.md — Universal Convergence Protocol](../../../docs/process/hypothesis.md#universal-convergence-protocol). The rules below are a self-contained copy for skill execution; hypothesis.md is authoritative if they diverge.

The convergence loop is **automatic and self-driven**. The user invokes `/convergence-review <gate> [artifact]` once. The skill manages the loop internally via state files. There is no manual re-invocation between rounds.

### Phase A: Review

Phase A dispatches all perspectives, collects results, tallies findings, and produces exactly one of three outcomes. **Phase A never applies fixes.**

```
1. Determine round number:
   - If state file exists and status is "awaiting-rerun": round = state.round + 1
   - Otherwise: round = 1, create state file with max_rounds = 10

2. Dispatch ALL perspectives in parallel (background Task agents, model=haiku)
3. Wait for all to complete (5 min timeout per agent)
4. Read each agent's output INDEPENDENTLY
5. Tally CRITICAL and IMPORTANT counts YOURSELF (do NOT trust agent totals)

6. Determine outcome:
   a. CONVERGED (total_critical == 0 AND total_important == 0):
      - Update state file: status = "converged"
      - Delete state file (cleanup)
      - Emit CONVERGED status banner
      - Proceed to next workflow step

   b. NOT CONVERGED, rounds remaining (round < max_rounds):
      - Update state file: round = current, status = "awaiting-rerun",
        append round to history with counts and status = "pending-fix"
      - Emit NOT CONVERGED status banner
      - Enter Phase B immediately

   c. NOT CONVERGED, round limit reached (round >= max_rounds):
      - Update state file: status = "stalled",
        append round to history with counts
      - Emit STALLED status banner with remaining issues
      - Use AskUserQuestion with options:
        (a) Increase limit and continue
        (b) Accept remaining issues and proceed
        (c) Abort
```

### Phase B: Fix-and-Rerun

Phase B is only entered from Phase A outcome (b). It applies all CRITICAL and IMPORTANT fixes, then immediately re-enters Phase A. **Phase B has no exit path other than invoking Phase A.**

```
1. Fix all CRITICAL items first, then all IMPORTANT items
2. Run tests/verification as appropriate for the gate type
3. Update state file: mark current round's history entry status = "fixed"
4. Re-enter Phase A (next round)
```

**Why this works:** Each phase is short enough that its exit instruction is never far from the current position in context. Phase A cannot skip re-dispatch because it never fixes anything. Phase B cannot skip re-review because its only exit is invoking Phase A. The state file is the structural enforcement — a concrete artifact on disk that determines behavior, not a prose instruction buried in context.

### Hard Rules

1. **Zero means zero.** One CRITICAL from one reviewer = not converged.
2. **Re-run is mandatory and automatic.** After fixing issues, Phase B MUST re-enter Phase A. There is no manual re-invocation. There is no "fixes were trivial, skip re-review."
3. **SUGGESTION items do not block.** Only CRITICAL and IMPORTANT count.
4. **Independent tallying.** Read each agent's output file. Count findings yourself. Agents have fabricated "0 CRITICAL, 0 IMPORTANT" when actual output contained 3 CRITICAL + 18 IMPORTANT (#390).
5. **No partial re-runs.** Re-run ALL perspectives, not just the ones that found issues. Fixes can introduce new issues in other perspectives.
6. **Agent timeout = 5 minutes.** If an agent exceeds this, check its output and restart. If it fails, perform that review directly.
7. **Max 10 rounds per gate (default).** If still not converged, enter stall protocol (AskUserQuestion). This is a safety net (R19 — circuit breaker), not an expected outcome.

### Severity Classification

When reviewing agent output, verify severity assignments:

| Severity | Definition | Blocks? |
|----------|-----------|---------|
| **CRITICAL** | Must fix. Missing control experiment, status contradicted by data, silent data loss, cross-document contradiction. | Yes |
| **IMPORTANT** | Should fix. Fixing would change a conclusion, metric, or user guidance. Sub-threshold effect, stale text, undocumented confound. | Yes |
| **SUGGESTION** | Cosmetic. Off-by-one line citation, style consistency, terminology nit. Fixing only improves readability. | No |

**When in doubt:** If fixing it would change any conclusion → IMPORTANT. If only readability → SUGGESTION.

---

## Round-Boundary Status Banners

After every Phase A verdict, emit a visible status banner. This ensures the user sees the round status and expects the next round (or knows the review is complete).

### Converged Banner

```
╔══════════════════════════════════════════════════════════════╗
║  CONVERGED — Round N/M — Gate: <gate-type>                  ║
║  0 CRITICAL | 0 IMPORTANT | K SUGGESTION                   ║
╠══════════════════════════════════════════════════════════════╣
║  Round History:                                             ║
║    Round 1: 2 CRITICAL, 5 IMPORTANT — fixed                ║
║    Round 2: 0 CRITICAL, 1 IMPORTANT — fixed                ║
║    Round 3: 0 CRITICAL, 0 IMPORTANT — CONVERGED            ║
╚══════════════════════════════════════════════════════════════╝
```

### Not Converged Banner (continuing to Phase B)

```
╔══════════════════════════════════════════════════════════════╗
║  NOT CONVERGED — Round N/M — Gate: <gate-type>              ║
║  X CRITICAL | Y IMPORTANT | Z SUGGESTION                   ║
║  Entering Phase B: fixing issues, then re-reviewing...      ║
╠══════════════════════════════════════════════════════════════╣
║  Round History:                                             ║
║    Round 1: X CRITICAL, Y IMPORTANT — fixing now            ║
╚══════════════════════════════════════════════════════════════╝
```

### Stalled Banner (round limit reached)

```
╔══════════════════════════════════════════════════════════════╗
║  STALLED — Round N/M (limit reached) — Gate: <gate-type>   ║
║  X CRITICAL | Y IMPORTANT remaining                        ║
║  Human decision required.                                   ║
╠══════════════════════════════════════════════════════════════╣
║  Round History:                                             ║
║    Round 1: 3 CRITICAL, 7 IMPORTANT — fixed                ║
║    Round 2: 1 CRITICAL, 2 IMPORTANT — fixed                ║
║    ...                                                      ║
║    Round 5: X CRITICAL, Y IMPORTANT — stalled              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Dispatch Instructions

### Step 1: Load Perspective Prompts

Based on gate type `$0`, load the correct prompts file:

- **`design` or `macro-plan`**: Read [design-prompts.md](design-prompts.md) for the matching section
- **`pr-plan` or `pr-code`**: Read [pr-prompts.md](pr-prompts.md) for the matching section
- **`h-design`, `h-code`, or `h-findings`**: Read `.claude/skills/hypothesis-experiment/review-prompts.md` for the matching section

### Step 2: Prepare Context

Each perspective agent needs the artifact being reviewed:

| Gate | What to include in each agent's prompt |
|------|----------------------------------------|
| `design` | The design document contents (read `$1`) |
| `macro-plan` | The macro plan contents (read `$1`) |
| `pr-plan` | The micro plan file contents (read `$1`) |
| `pr-code` | The current `git diff` output |
| `h-design` | The hypothesis sentence, classification, and experiment design (from conversation) |
| `h-code` | The `run.sh` and `analyze.py` file contents |
| `h-findings` | The `FINDINGS.md` file contents + `run.sh` path for cross-reference |

### Step 3: Dispatch All Perspectives

Launch all N perspectives simultaneously as background Task agents:

```
For each perspective P in the gate's perspective set:
    Task(
        subagent_type = "general-purpose",
        model = "haiku",
        run_in_background = True,
        prompt = "<perspective prompt from prompts file>\n\n<artifact content>"
    )
```

**Why haiku?** Fast (~2-3 min), thorough reviews with accurate severity classification. Haiku produces consistent CRITICAL/IMPORTANT/SUGGESTION output. Using opus/sonnet for 10 parallel reviewers is unnecessarily expensive.

**Exception — Perspective 5 in PR plan reviews (Structural Validation):** Perform this check directly (no agent). It requires structural validation of the plan (task dependencies, template completeness) that benefits from your full conversation context.

### Step 4: Collect and Tally

After all agents complete:

1. **Read each output file** using the Read tool or TaskOutput
2. **For each agent**, extract:
   - List of findings with severity
   - Count of CRITICAL findings
   - Count of IMPORTANT findings
3. **Independently verify** the counts match the findings listed
4. **Aggregate** across all perspectives

### Step 5: Report and Transition

Present results in this format, then follow the Phase A outcome rules:

```
## Round N Results — Gate: <gate-type>

| Perspective | CRITICAL | IMPORTANT | SUGGESTION |
|-------------|----------|-----------|------------|
| P1: <name>  | 0        | 1         | 2          |
| P2: <name>  | 0        | 0         | 1          |
| ...         | ...      | ...       | ...        |
| **TOTAL**   | **0**    | **1**     | **3**      |
```

Then emit the appropriate status banner and follow Phase A's outcome logic:
- If converged → clean up state file, report success, proceed
- If not converged, rounds remaining → enter Phase B immediately
- If stalled → AskUserQuestion for human decision

---

## Integration with Other Skills

### From design process
Review a design document before macro/micro planning:
```
/convergence-review design docs/plans/archive/<design-doc>.md
```

### From macro planning process
Review a macro plan before micro-planning any PR:
```
/convergence-review macro-plan docs/plans/<macro-plan>.md
```

### From PR workflow
The PR workflow calls this skill at Steps 2.5 and 4.5:
```
/convergence-review pr-plan docs/plans/pr<N>-<name>-plan.md
/convergence-review pr-code
```

### From hypothesis-experiment skill
The hypothesis-experiment skill calls this skill at Steps 2, 5, and 8:
```
/convergence-review h-design
/convergence-review h-code hypotheses/h-<name>/
/convergence-review h-findings hypotheses/h-<name>/FINDINGS.md
```
