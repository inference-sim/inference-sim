# PR760: Add /quick-review Bob Command for Token-Efficient Multi-Perspective Reviews

**Goal:** Create `.bob/commands/quick-review.md` that provides convergence-review quality perspectives with sequential execution (no parallel agents, no convergence loop).

**The problem today:** Bob users have two review options: (1) `/review` command with 4 general perspectives (functionality, security, maintainability, performance), or (2) request maintainers run `convergence-review` with 5-10 domain-specific perspectives but at high token cost (parallel dispatch) and with convergence loop overhead. There's no middle ground for developers who want domain-specific review quality during iterative development without the token cost and convergence commitment.

**What this PR adds:**
1. **New Bob command file** — `.bob/commands/quick-review.md` that Bob auto-discovers and makes available as `/quick-review <gate-type> [artifact-path]`
2. **Sequential perspective execution** — All N perspectives for a gate run in a single session (not parallel agents), reducing token cost by ~1/N while maintaining same quality
3. **Reuse of convergence-review perspectives** — Reads and applies the exact same 43 perspective prompts from `.claude/skills/convergence-review/` files, ensuring consistent quality standards
4. **Findings integration** — Reports all findings via `submit_review_findings` tool to Bob Findings panel, with no convergence loop (single pass, exit)

**Why this matters:** This enables Bob users to get domain-specific review feedback (DES Expert, vLLM Expert, Distributed Platform Expert perspectives) during development without requiring maintainer intervention or paying the token cost of parallel dispatch. It bridges the gap between general `/review` and formal `convergence-review` gates.

**Architecture:** This PR creates a Bob command file at `.bob/commands/quick-review.md` that Bob automatically discovers. The command reads perspective prompts from `.claude/skills/convergence-review/*.md` files (pr-prompts.md, design-prompts.md, hypothesis-experiment/review-prompts.md), executes them sequentially in a single session, extracts findings from each perspective's output, and reports via the `submit_review_findings` tool. No BLIS code changes.

**Source:** GitHub issue #760

**Closes:** #760

---

## Part 1: Design Validation

### A. Behavioral Contracts

**BC-1: Command Invocation with Fuzzy Aliases**
- GIVEN user invokes `/quick-review [gate-type] [artifact-path]`
- WHEN no arguments provided: default to `pr-code` gate (review current git diff)
- WHEN gate-type provided: normalize using fuzzy matching
  - **Exact aliases** (case-insensitive):
    - `plan` → `pr-plan`
    - `code` or `implementation` → `pr-code`
    - `findings` or `discovery` → `h-findings`
  - **Fuzzy matching** for unrecognized inputs (using Levenshtein distance):
    - Use normalized Levenshtein distance (0-100% similarity) to find closest canonical gate type
    - If similarity > 70%: suggest closest match and ask for confirmation
    - If similarity ≤ 70%: show error with all valid gates and aliases
    - Empty/null input: treat as 0% similarity, show error with valid gates
  - **Canonical names** accepted as-is: `pr-plan`, `pr-code`, `design`, `macro-plan`, `h-design`, `h-code`, `h-findings`
- WHEN artifact-path omitted: use sensible default per gate type
  - `pr-code`: `git diff HEAD` (default)
  - `pr-plan`: search for `docs/plans/pr*-plan.md` matching current branch
  - `design`: search for `docs/plans/*-design.md` in current directory
  - `macro-plan`: search for `docs/plans/*-plan.md` (non-PR plans)
  - `h-code`: use current directory if in `hypotheses/h-*/`
  - `h-findings`: search for `FINDINGS.md` in current directory
  - `h-design`: use conversation context (no file needed)
- WHEN both provided: use specified gate-type and artifact-path
- THEN Bob loads appropriate perspective prompts and artifact, executes review sequentially

**BC-2: Sequential Perspective Execution**
- GIVEN a gate type with N perspectives
- WHEN review executes
- THEN all N perspectives run in single session (not parallel agents)
- AND findings accumulate across perspectives
- AND token cost is ~1/N of parallel dispatch (shared context reduces per-perspective overhead compared to N independent agent sessions)

**BC-3: Perspective Prompt Reuse**
- GIVEN perspective prompts exist in `.claude/skills/convergence-review/` files as markdown code blocks
- WHEN `/quick-review` executes
- THEN it reads perspective prompts by extracting text between ``` delimiters in the prompt files
- AND applies the exact same prompt text as convergence-review to the artifact
- AND maintains same quality standards

**BC-4: Findings Reporting**
- GIVEN review completes with findings
- WHEN all perspectives finish
- THEN Bob calls `submit_review_findings` with all collected findings
- AND each finding includes required fields: category, type, severity, title, message, path, line, issueScope
- AND each finding may include optional fields: column, endLine, endColumn, suggestion, issue (for filed disposition)

**BC-5: No Convergence Loop**
- GIVEN review completes
- WHEN findings are reported
- THEN command exits (no Phase B, no fix loop, no state file)

**BC-6: Artifact Loading with Defaults**
- GIVEN gate type determines artifact type
- WHEN pr-code: load `git diff HEAD` (always, artifact-path ignored)
- WHEN file-based gates without artifact-path: search for default file
  - pr-plan: find `docs/plans/pr*-plan.md` matching current branch name (exact substring match: branch `pr760-quick-review` matches `pr760-quick-review-plan.md`)
  - design: find `docs/plans/*-design.md` in current directory
  - macro-plan: find `docs/plans/*-plan.md` (excluding pr* patterns)
  - h-code: use current directory if path contains `hypotheses/h-*/`
  - h-findings: find `FINDINGS.md` in current directory or `hypotheses/h-*/`
  - If multiple matches found: use most recently modified file
- WHEN file-based gates with artifact-path: load specified file
- WHEN h-design: use conversation context (no file needed)
- THEN artifact content passed to each perspective

**BC-7: Error Handling with Fuzzy Suggestions**
- GIVEN invalid gate type OR missing artifact file OR empty git diff OR default file search fails
- WHEN command executes
- THEN clear error message shown with:
  - **Unrecognized gate with high similarity (>70%)**: "Did you mean '<closest-match>'? Valid gates: pr-plan, pr-code, design, macro-plan, h-design, h-code, h-findings. Aliases: plan, code, implementation, findings, discovery"
  - **Unrecognized gate with low similarity (≤70%)**: "Invalid gate type '<input>'. Valid gates: pr-plan, pr-code, design, macro-plan, h-design, h-code, h-findings. Aliases: plan, code, implementation, findings, discovery"
  - **Empty or null input**: "No gate type provided. Valid gates: pr-plan, pr-code, design, macro-plan, h-design, h-code, h-findings. Aliases: plan, code, implementation, findings, discovery"
  - **Missing file**: show searched paths and suggest correct path
  - **Empty diff**: suggest staging changes or committing
  - **Default search failed**: show search pattern and suggest explicit path
- AND command exits gracefully

### B. Component Interaction

```
User invokes: /quick-review pr-code
     ↓
Bob Command Handler
     ↓
Parse args → gate-type, artifact-path
     ↓
Load artifact (git diff / file / context)
     ↓
Read perspective prompts from .claude/skills/convergence-review/
     ↓
FOR EACH perspective in gate:
  ├─ Apply perspective prompt to artifact
  ├─ Extract findings (severity, location, issue, expected)
  └─ Accumulate in findings list
     ↓
Call submit_review_findings(findings)
     ↓
Exit
```

**Key interactions:**
1. Command parser → gate type validator
2. Artifact loader → file system / git
3. Prompt reader → `.claude/skills/convergence-review/*.md` files
4. Perspective executor → sequential loop (not parallel agents)
5. Finding extractor → structured finding objects
6. Reporter → `submit_review_findings` tool

### C. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Perspective prompts change format | High - parsing breaks | Use robust parsing; test against current prompts |
| Sequential execution too slow | Medium - user experience | Acceptable trade-off for token savings; document expected time |
| Finding extraction inconsistent | High - quality degradation | Follow exact prompt format; validate against convergence-review output |
| Artifact loading fails | Medium - command unusable | Comprehensive error handling with clear messages |
| Prompt file paths change | Medium - command breaks | Use relative paths from repo root; document dependency |

### D. Extension Points

This PR adds a new Bob command. Future extensions:
- Add `--gate-type` flag to `/review` to use quick-review perspectives
- Add progress indicator for long reviews (10 perspectives)
- Cache perspective prompts to avoid repeated file reads
- Add `--perspective` flag to run single perspective

### E. Deviation Log

| Item | Source Says | Plan Does | Reason |
|------|------------|-----------|--------|
| Model selection | Issue doesn't specify | No `--model` flag | CLARIFICATION: Bob doesn't expose model flags (user confirmed) |
| Prompt source | Issue says "reuse prompts" | Read from `.claude/skills/convergence-review/*.md` | CLARIFICATION: Implementation detail - Bob reads perspective prompts from repository skill files by extracting markdown code blocks |

---

## Part 2: Executable Tasks

### F. Test Strategy

| Contract | Test Type | Verification Method |
|----------|-----------|---------------------|
| BC-1 | Manual | Review syntax examples in docs; test invocation with valid/invalid gates |
| BC-2 | Manual | Verify single session execution (no parallel agents spawned) |
| BC-3 | Manual | Cross-reference perspective text with actual prompt files; verify exact match |
| BC-4 | Manual | Verify findings appear in Bob Findings panel with all required fields |
| BC-5 | Manual | Verify no loop, immediate exit after findings reported |
| BC-6 | Manual | Test each gate type with appropriate artifact |
| BC-7 | Manual | Test error cases: invalid gate, missing file, empty diff |

### G. Task Breakdown

**Task 1: Create Bob command file**

Create `.bob/commands/quick-review.md` with frontmatter and command prompt:
1. **Frontmatter** — YAML with `description` and `argument-hint` fields
2. **Command Prompt** — Instructions for Bob to execute the quick-review workflow
   - Parse `$1` (gate-type, default: `pr-code`) and `$2` (artifact-path, default: per gate type)
     - Note: Bob uses `$1`, `$2`, etc. to access command arguments (standard Bob syntax per documentation)
   - Normalize gate-type with fuzzy matching:
     - Exact aliases (case-insensitive): `plan` → `pr-plan`, `code`/`implementation` → `pr-code`, `findings`/`discovery` → `h-findings`
     - Fuzzy match unrecognized inputs using Levenshtein distance: if similarity >70%, suggest closest match; else show error with valid gates
     - Empty/null input: treat as 0% similarity, show error with valid gates
   - Apply default resolution logic:
     - No args: use `pr-code` gate with `git diff HEAD`
     - Gate only: search for default artifact per gate type (use most recently modified if multiple matches)
     - Both provided: use as specified
   - Load artifact based on gate type (git diff for pr-code, file for others, context for h-design)
   - Read perspective prompts from `.bob/prompts/review-perspectives.md`
     - Extract prompts by finding text between triple-backtick delimiters (```...```) in the prompt file
     - Each perspective is a markdown code block in the source file
   - Execute each perspective sequentially (not parallel)
   - Extract findings from each perspective's output
   - Accumulate findings across all perspectives
   - Call `submit_review_findings` with complete findings array
   - Exit (no Phase B, no state file)

**Files:**
- Create: `.bob/prompts/review-perspectives.md` (all 43 perspective prompts)
- Create: `.bob/commands/quick-review.md` (command file)

**Test:** Review command file format against Bob documentation; verify frontmatter syntax and argument placeholders

**Commit:**
```bash
git add .bob/prompts/review-perspectives.md .bob/commands/quick-review.md
git commit -m "feat: add /quick-review Bob command (BC-1 through BC-6)

- Create self-contained perspective prompts file
- Create Bob command file with frontmatter
- Define sequential perspective execution
- Integrate with submit_review_findings tool
- Support all 7 gate types (43 perspectives)

Closes #760"
```

**Task 2: Create implementation guide**

Create `docs/bob-commands/quick-review-implementation.md` with these sections:
1. **Overview** — What the command does, how it differs from `/review` and `/convergence-review`
2. **Bob Command Format** — Explain frontmatter fields and command prompt structure
3. **Gate Types** — Table of 7 gates with perspective counts and artifact types
4. **Perspective Sources** — Map gate types to prompt files with line ranges
5. **Sequential Execution** — Algorithm for reading prompts and executing perspectives
6. **Finding Extraction** — Required and optional fields for `submit_review_findings`
7. **Error Handling** — Error cases and messages
8. **Usage Examples** — One example per gate type (7 total)
9. **Comparison Table** — `/review` vs `/quick-review` vs `/convergence-review`

**Files:**
- Create: `docs/bob-commands/quick-review-implementation.md`

**Test:** Cross-reference with actual prompt files; verify all 43 perspectives mapped correctly

**Commit:**
```bash
git add docs/bob-commands/quick-review-implementation.md
git commit -m "docs: add implementation guide for /quick-review (BC-2, BC-3, BC-4)

- Document Bob command format
- Map perspectives to source files
- Define sequential execution algorithm
- Provide usage examples

Ref: #760"
```

---

## Part 3: Sanity Checklist

### J. Pre-Commit Verification

- [ ] All documentation files created and complete
- [ ] Perspective mapping verified against actual prompt files
- [ ] Algorithm description is implementable
- [ ] Examples cover all gate types
- [ ] Error cases documented
- [ ] Issue #760 updated with implementation plan
- [ ] No code changes (documentation only)
- [ ] All files use consistent formatting
- [ ] Cross-references between docs are correct
- [ ] Deviation log complete

---

## Appendix: File-Level Details

### .bob/commands/quick-review.md

**Purpose:** Bob command file that defines `/quick-review` behavior

**Structure:**
1. Frontmatter (YAML) — `description` and `argument-hint` fields
2. Command Prompt — Instructions for Bob to execute the workflow

**Frontmatter fields:**
- `description`: "Execute multi-perspective review using convergence-review perspectives (sequential, no loop)"
- `argument-hint`: "<gate-type> [artifact-path]"

**Command prompt sections:**
1. Role definition — Bob as domain-specific reviewer
2. Argument parsing with defaults, aliases, and fuzzy matching:
   - Extract gate-type from `$1` (default: `pr-code`) using Bob's standard `$1` syntax
   - Normalize exact aliases (case-insensitive): `plan` → `pr-plan`, `code`/`implementation` → `pr-code`, `findings`/`discovery` → `h-findings`
   - Apply fuzzy matching for unrecognized inputs using Levenshtein distance: if similarity >70%, suggest closest match; else error with valid gates
   - Handle empty/null input: treat as 0% similarity, show error with valid gates
   - Extract artifact-path from `$2` (default: per gate type)
   - Apply default resolution: no args → pr-code + git diff, gate only → search for default artifact (use most recently modified if multiple matches)
3. Gate validation — Check normalized gate-type is one of 7 valid options
4. Artifact loading — Load based on gate type (git diff / file / context) with default search if path not provided
5. Perspective reading — Read from `.claude/skills/convergence-review/*.md` files by extracting text between triple-backtick delimiters (```...```)
6. Sequential execution — Loop through perspectives, apply to artifact
7. Finding extraction — Parse output for severity, location, description
8. Accumulation — Build findings array
9. Reporting — Call `submit_review_findings` with findings
10. Exit — No Phase B, no state file

**Line count estimate:** ~200 lines

### docs/bob-commands/quick-review-implementation.md

**Purpose:** Implementation guide for understanding and maintaining the command

**Sections:**
1. Overview — What the command does, how it differs from other review commands
2. Bob Command Format — Explain frontmatter and command prompt structure
3. Gate Types — Table mapping gate types to perspective counts and artifacts
4. Perspective Sources — Map gate types to prompt files (pr-prompts.md, design-prompts.md, review-prompts.md) with line ranges
5. Sequential Execution — Algorithm for reading prompts and executing perspectives
6. Finding Extraction — Required fields (category, type, severity, title, message, path, line, issueScope) and optional fields (column, endLine, endColumn, suggestion, issue)
7. Error Handling — Error cases (invalid gate, missing file, empty diff) with messages
8. Usage Examples — 7 examples (one per gate type) with expected output
9. Comparison Table — `/review` vs `/quick-review` vs `/convergence-review` (tool, perspectives, execution, loop, output, token cost, use case)

**Line count estimate:** ~300 lines
