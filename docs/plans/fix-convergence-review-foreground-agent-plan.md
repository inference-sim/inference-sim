# Fix: Convergence Review — Single Foreground Agent Dispatch

**Goal:** Replace the N-parallel-background-agents dispatch in Phase A with a single foreground agent that receives the artifact once and writes all perspective sections in one response.

**Source:** Conversation — user-reported "No task found with ID" errors when running `/convergence-review`; agreed fix: single foreground agent to eliminate background task ID failures and reduce token cost (artifact sent once instead of N times).

**Closes:** N/A (no tracked issue)

**PR size tier:** Small (3 files, execution-only change, no new interfaces/types, no new CLI flags)

---

## Behavioral Contracts

### BC-1: Single foreground agent replaces N background agents

GIVEN the skill is at Phase A Step 3 for any gate type
WHEN N perspective prompts are assembled
THEN the skill invokes exactly one foreground agent (no `run_in_background`) with all N prompts and the artifact in a single call, waits for it to complete, and uses its output as the source for all N perspectives

### BC-2: Artifact sent once

GIVEN a diff-anchored gate with a large diff
WHEN the single agent is assembled
THEN the artifact content appears exactly once in the prompt, not once per perspective

### BC-3: Perspective independence preserved

GIVEN the single agent receives all N prompts
WHEN the prompt is assembled
THEN each perspective prompt is clearly delimited by a `## [<ID>] <Name>` header and the agent is instructed to treat each section as independent (complete it fully before reading the next prompt)

### BC-4: Tally reads sections, not separate agents

GIVEN the single agent returns output
WHEN the skill tallies findings
THEN it parses the output by section header `## [<ID>]` to extract per-perspective findings, and counts independently (never trusts agent-reported totals)

### BC-5: PP-5 exception preserved and tallied

GIVEN a `pr-plan` gate
WHEN the single agent is invoked
THEN it receives only PP-1 through PP-4 and PP-6 through PP-10 (9 prompts); PP-5 (structural validation) is still performed directly by the skill, not delegated; and PP-5 findings are added to the tally findings array in step 4 alongside findings parsed from the agent's output sections

### BC-6: No state file schema change

GIVEN a convergence round completes
WHEN findings are written to the state file
THEN the schema is unchanged — `perspective` field still records the perspective ID (e.g., "PP-1"), `severity`, `location`, `description`, `disposition` are unchanged

### BC-7: Missing section header is a warning, not clean evidence

GIVEN the single agent's output is missing a `## [<ID>]` section header for one or more expected perspectives
WHEN the skill parses the output
THEN it emits a WARNING for each missing section and records 0 findings for that perspective — it does NOT treat the absence as convergence-clean evidence

---

## Tasks

### Task 1: Update SKILL.md — Phase A Steps 3 and 4 (single edit, single commit)

**Why combined:** Steps 3 and 4 must be updated together. Committing Step 3 alone leaves the file semantically inconsistent (Step 3 says "single agent" while Step 4 still says "for each agent's output").

**Test first:** Read SKILL.md. Verify Step 3 contains "simultaneously as background agents" and `run_in_background`, and Step 4a contains "For each agent's output".

**Change 1 — Replace Step 3:**

Old text (exact):
```
3. **Dispatch all N perspectives** simultaneously as background agents. Model from state file.
   - **Exception:** The structural validation perspective in PR plan reviews is performed directly (not delegated to an agent) because it requires full conversation context.
   - **Context payload per gate type:**
     - File-anchored gates (`design`, `macro-plan`, `pr-plan`, `h-code`, `h-findings`): Pass the file contents of `$1` to each agent.
     - Diff-anchored gates (`pr-code`): Pass `git diff HEAD` output to each agent.
     - Context-anchored gates (`h-design`): Pass the hypothesis sentence, classification, and experiment design from the current conversation context.
   - **Empty-diff precondition (diff-anchored gates only):** If `git diff HEAD` produces no output, emit warning ("No changes detected since last commit — nothing to review") and skip dispatch. Stage new files or commit before invoking.
```

New text:
```
3. **Assemble and dispatch a single foreground agent** with all N perspective prompts and the artifact. Model from state file.

   **Why single agent:** Background task IDs are session-local and can become invalid mid-session (producing "No task found with ID" errors). A single foreground call is reliable and sends the artifact once instead of N times (lower token cost for large diffs/files).

   - **Exception:** The structural validation perspective in PR plan reviews (PP-5) is performed directly by the skill — not delegated. For `pr-plan` gates, the agent receives 9 prompts (PP-1 through PP-4, PP-6 through PP-10); PP-5 findings from direct evaluation are added to the findings array in step 4 alongside sections parsed from the agent's output. For all other gates, the agent receives all N prompts.

   - **Empty-diff precondition (diff-anchored gates only):** If `git diff HEAD` produces no output, emit warning ("No changes detected since last commit — nothing to review") and skip dispatch. Stage new files or commit before invoking.

   - **Prompt assembly:** Build a single prompt in this order:
     1. Opening instruction:
        ```
        You are performing a multi-perspective review. Each section below is an independent review perspective. Complete each section fully before moving to the next. Do not let earlier sections influence later ones — treat each as if starting fresh.
        ```
     2. Artifact block (once):
        ```
        ## ARTIFACT

        <artifact content — diff, file contents, or context per gate type>
        ```
        Context payload per gate type:
        - File-anchored gates (`design`, `macro-plan`, `pr-plan`, `h-code`, `h-findings`): paste the file contents of `$1`.
        - Diff-anchored gates (`pr-code`): paste `git diff HEAD` output.
        - Context-anchored gates (`h-design`): paste the hypothesis sentence, classification, and experiment design from the current conversation context.
     3. Perspective sections (one per prompt, in ID order):
        ```
        ## [<ID>] <Perspective Name>

        <full prompt text for this perspective>
        ```

   - **Invoke:** Call the Agent tool with `run_in_background=False` (foreground), the assembled prompt, and `model=REVIEW_MODEL` where `REVIEW_MODEL` defaults to `"haiku"` when `--model` is not specified (see Model Selection section). Pass the model value explicitly — never omit it — so the Agent tool never inherits the session's active model (which may be `opus`).

   - **Missing section handling:** After the agent completes, collect all `## [<ID>]` headers that appear in the output. For each expected perspective ID that is absent, emit `"WARNING: No section found for <ID> in agent output — recording 0 findings for this perspective"` and proceed. Do NOT treat a missing section as convergence-clean evidence.
```

**Change 2 — Replace Step 4 preamble and step 4a only:**

Old text (exact):
```
4. **Collect, extract, and tally independently.** For each agent's output:
   a. Extract individual findings. For each finding, record: `perspective` (agent ID), `severity`, `location` (file:line or best available reference), `description`.
```

New text:
```
4. **Collect, extract, and tally independently.** Parse the single agent's output by section header:
   a. Split the output on `## [<ID>]` headers (lines matching `^## \[`) to isolate each perspective's section. For each section, extract individual findings. For each finding, record: `perspective` (the ID from the section header, e.g., "PP-1"), `severity`, `location` (file:line or best available reference), `description`. For `pr-plan` gates, also include findings from the direct PP-5 evaluation in this same array.
```

**Verify after both changes:**
- The word "background" no longer appears in Steps 3 or 4 (only in the "Why single agent" rationale)
- Steps 4b through 4f are unchanged
- The PP-5 exception text explicitly mentions feeding PP-5 findings into step 4
- Missing section handling is present in Step 3

**Commit:** `docs: update SKILL.md Phase A steps 3 and 4 to single foreground agent dispatch`

---

### Task 2: Update pr-prompts.md dispatch pattern header

**Test first:** Read the dispatch pattern block at the top of pr-prompts.md. Verify it references `run_in_background=True`.

**Change:** Replace the dispatch pattern block.

Old text (exact):
```
**Dispatch pattern:** Launch each perspective as a parallel Task agent:
```
Task(subagent_type="general-purpose", model=REVIEW_MODEL, run_in_background=True,
     prompt="<prompt from below>\n\n<artifact content>")
```
Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).
```

New text:
```
**Dispatch pattern:** All perspectives are passed to a single foreground agent in one call. The artifact is sent once; each perspective appears as a `## [<ID>] <Name>` section. See SKILL.md Phase A Step 3 for the full assembly specification. Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).
```

**Verify:** No `run_in_background` anywhere in pr-prompts.md.

**Commit:** `docs: update pr-prompts.md dispatch pattern to single foreground agent`

---

### Task 3: Update design-prompts.md dispatch pattern header

Identical operation to Task 2 on a different file.

**Test first:** Read the dispatch pattern block at the top of design-prompts.md. Verify it references `run_in_background=True`.

**Change:** Same replacement as Task 2:

Old text (exact):
```
**Dispatch pattern:** Launch each perspective as a parallel Task agent:
```
Task(subagent_type="general-purpose", model=REVIEW_MODEL, run_in_background=True,
     prompt="<prompt from below>\n\n<artifact content>")
```
Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).
```

New text:
```
**Dispatch pattern:** All perspectives are passed to a single foreground agent in one call. The artifact is sent once; each perspective appears as a `## [<ID>] <Name>` section. See SKILL.md Phase A Step 3 for the full assembly specification. Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).
```

**Verify:** No `run_in_background` anywhere in design-prompts.md.

**Commit:** `docs: update design-prompts.md dispatch pattern to single foreground agent`

---

## Sanity Checklist

- [ ] `run_in_background` does not appear in any of the 3 changed files (except in the "Why single agent" rationale note in SKILL.md)
- [ ] PP-5 exception text explicitly states PP-5 findings feed into the step 4 tally array
- [ ] Missing section handler is present in new Step 3 (WARNING + 0 findings, not convergence-clean)
- [ ] Artifact appears in the assembled prompt exactly once (described in Step 3)
- [ ] Section header format `## [<ID>] <Name>` is consistent between Step 3 (assembly) and Step 4 (parsing uses `^## \[`)
- [ ] The Agent tool call in new Step 3 explicitly passes `model=REVIEW_MODEL` (not omitted, not hardcoded)
- [ ] Steps 3 and 4 are updated in a single SKILL.md commit (never committed separately)
- [ ] State file schema is unchanged (no new fields, no removed fields)
- [ ] No changes to the perspective prompts themselves in pr-prompts.md or design-prompts.md
- [ ] No changes to SKILL.md Phase B, state file schema, convergence rules, or triage logic
