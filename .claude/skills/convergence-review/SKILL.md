---
name: convergence-review
description: Dispatch parallel review perspectives and enforce convergence (zero CRITICAL + zero IMPORTANT). Supports 7 gate types — design doc (8), macro plan (8), PR plan (10), PR code (10), hypothesis design (5), hypothesis code (5), hypothesis FINDINGS (10).
argument-hint: <gate-type> [artifact-path] [--model opus|sonnet|haiku]
---

# Convergence Review Dispatcher

Dispatch parallel review perspectives for gate **$0** and enforce convergence.

## Model Selection

The `--model` flag controls which model runs the perspective agents:

```
/convergence-review pr-code                    # uses haiku (default)
/convergence-review pr-plan plan.md --model sonnet
/convergence-review h-findings FINDINGS.md --model opus
```

**Parsing order:** Strip `--model <value>` from `$ARGS` first, then parse remaining tokens as `$0` (gate type) and `$1` (artifact path).

**Validation:** If `--model` value is not one of `opus`, `sonnet`, `haiku`, report the error to the user and stop. If `--model` is present but has no value, report the error and stop.

Valid values: `opus`, `sonnet`, `haiku`. Default: **`haiku`**.

`REVIEW_MODEL` stores the short name (`opus`, `sonnet`, or `haiku`). The Task tool resolves these to the appropriate model version (e.g., `opus` → `claude-opus-4-6`). No explicit model-ID mapping is needed in this skill.

Store the resolved model in a variable `REVIEW_MODEL` and use it in all Task dispatches.

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

## Convergence Protocol (non-negotiable)

> **Canonical source:** [docs/process/hypothesis.md — Universal Convergence Protocol](../../../docs/process/hypothesis.md#universal-convergence-protocol). The rules below are a self-contained copy for skill execution; hypothesis.md is authoritative if they diverge.

These rules are identical across all gates. No exceptions. No shortcuts.

### The Algorithm

```
round = 1
while round <= 10:
    1. Dispatch ALL perspectives in parallel (background Task agents, model=REVIEW_MODEL)
    2. Wait for all to complete (5 min timeout per agent)
    3. Read each agent's output INDEPENDENTLY
    4. Tally CRITICAL and IMPORTANT counts YOURSELF (do NOT trust agent totals)
    5. Report round results to user

    if total_critical == 0 AND total_important == 0:
        CONVERGED — report success, proceed to next workflow step
        break
    else:
        Report all findings with severity
        Fix all CRITICAL and IMPORTANT items
        round += 1
        RE-RUN ENTIRE ROUND (go to step 1)

if round > 10:
    SUSPEND — document remaining issues as future work
```

### Hard Rules

1. **Zero means zero.** One CRITICAL from one reviewer = not converged.
2. **Re-run is mandatory.** After fixing issues, you MUST re-run the entire round. You may NOT skip, propose alternatives, or rationalize fixes were trivial.
3. **SUGGESTION items do not block.** Only CRITICAL and IMPORTANT count.
4. **Independent tallying.** Read each agent's output file. Count findings yourself. Agents have fabricated "0 CRITICAL, 0 IMPORTANT" when actual output contained 3 CRITICAL + 18 IMPORTANT (#390).
5. **No partial re-runs.** Re-run ALL perspectives, not just the ones that found issues. Fixes can introduce new issues in other perspectives.
6. **Agent timeout = 5 minutes.** If an agent exceeds this, check its output and restart. If it fails, perform that review directly.
7. **Max 10 rounds per gate.** If still not converged, suspend the experiment/PR.

### Severity Classification

When reviewing agent output, verify severity assignments:

| Severity | Definition | Blocks? |
|----------|-----------|---------|
| **CRITICAL** | Must fix. Missing control experiment, status contradicted by data, silent data loss, cross-document contradiction. | Yes |
| **IMPORTANT** | Should fix. Fixing would change a conclusion, metric, or user guidance. Sub-threshold effect, stale text, undocumented confound. | Yes |
| **SUGGESTION** | Cosmetic. Off-by-one line citation, style consistency, terminology nit. Fixing only improves readability. | No |

**When in doubt:** If fixing it would change any conclusion → IMPORTANT. If only readability → SUGGESTION.

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
        model = REVIEW_MODEL,       # from --model flag, default "haiku"
        run_in_background = True,
        prompt = "<perspective prompt from prompts file>\n\n<artifact content>"
    )
```

**Default model: haiku.** Fast (~2-3 min per agent), thorough reviews with accurate severity classification. Haiku produces consistent CRITICAL/IMPORTANT/SUGGESTION output and has been validated across 10+ PRs. Use `--model sonnet` for higher review quality, or `--model opus` for maximum depth (note: opus costs ~15x more per agent; for 10 parallel reviewers this adds up quickly).

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

### Step 5: Report

Present results in this format:

```
## Round N Results — Gate: <gate-type>

| Perspective | CRITICAL | IMPORTANT | SUGGESTION |
|-------------|----------|-----------|------------|
| P1: <name>  | 0        | 1         | 2          |
| P2: <name>  | 0        | 0         | 1          |
| ...         | ...      | ...       | ...        |
| **TOTAL**   | **0**    | **1**     | **3**      |

### Convergence: NOT CONVERGED (1 IMPORTANT remaining)

### Findings requiring action:
1. [IMPORTANT] P1: <finding description>
```

If converged:
```
### Convergence: CONVERGED in Round N

All perspectives report 0 CRITICAL and 0 IMPORTANT findings.
Proceed to the next workflow step.
```

---

## Round Tracking

Track rounds across invocations. If this is a re-run after fixes:

```
## Round History
- Round 1: 2 CRITICAL, 5 IMPORTANT — fixed
- Round 2: 0 CRITICAL, 1 IMPORTANT — fixed
- Round 3: 0 CRITICAL, 0 IMPORTANT — CONVERGED
```

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

### After fixes
Re-invoke with the same arguments to start the next round:
```
/convergence-review pr-code  (Round 2 after fixes)
```
