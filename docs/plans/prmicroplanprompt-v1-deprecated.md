⚠️ **DEPRECATED - Use prmicroplanprompt-v2.md instead**

**This version is kept for reference only.**

**Active version:** `prmicroplanprompt-v2.md`

**Migration date:** 2026-02-14

**Reason for deprecation:** v2 merges design rigor (behavioral contracts, architecture validation) with executable task breakdown (TDD format, complete code examples) for use with the `executing-plans` skill. This enables a single planning stage that produces both human-reviewable design and agent-executable implementation steps.

**What v2 adds:**
- Executable task breakdown with TDD steps (test → fail → implement → pass → lint → commit)
- Complete code in task steps (no placeholders)
- Batch structure for checkpoint reviews
- Direct compatibility with `executing-plans` and `subagent-driven-development` skills
- Required document header for skill integration

**To migrate:** Use `prmicroplanprompt-v2.md` with the workflow in `prworkflow.md`

---

# ORIGINAL CONTENT BELOW (v1 - deprecated)

---

You are operating inside a real repository with full code access.

You are tasked with producing a PR-SPECIFIC MICRO-DESIGN PLAN for
PR <X> from the approved Macro Plan.

This is not a restatement of the macro plan.
This is deep implementation planning.

No coding yet.
Only design.

Use behavior driven development methodology.

This plan has TWO AUDIENCES:
1) A human reviewer who thinks in building blocks, invariants, and contracts
2) Claude Code executing the implementation

Structure the output accordingly: human-reviewable sections first,
implementation details in an appendix.

======================================================================
TEAM-BASED PLANNING PROCESS
======================================================================

You MUST use an agent team to produce this plan. Multiple independent
perspectives catch what a single agent misses. The team produces the
same output as before, but with higher quality through specialization
and adversarial review.

TEAM STRUCTURE (3 teammates + lead):

  "codebase-analyst"    — Reads code, finds facts
  "plan-designer"       — Designs contracts + architecture
  "plan-reviewer"       — Challenges everything, designs tests

  Lead (you): Coordinates via shared task list, synthesizes final document

TASK DEPENDENCIES:

  Phase 0, 3 (analyst)         → no dependencies
  Phase 1, 2, 4, 5 (designer)  → depends on Phase 0, 3
  Phase 6b, 7 Part A (reviewer)  → depends on Phase 0, 3
  Phase 6, 7 Part B+C (reviewer) → depends on designer (all phases)
  Phase 8 (lead)               → depends on all above

Create tasks in the shared task list with these dependencies.
The framework handles ordering — do not prescribe spawn sequence.

GROUND RULES FOR ALL TEAMMATES:
- Every behavioral claim about existing code MUST cite file:line
- Anti-hallucination: if you cannot cite it, mark it UNVERIFIED
- "Should" is banned — use "MUST" or "MUST NOT"
- Read the actual code, not just CLAUDE.md or macro plan
- If you disagree with another teammate's finding, say so and cite evidence

======================================================================
TEAMMATE: CODEBASE ANALYST
======================================================================

Role: Deep code reader. Produces the factual foundation that all
other teammates build on. Read-only — no design decisions.

Produces: Phase 0 (Component Context) + Phase 3 (Deviation Log)

Instructions:

Read the macro plan entry for PR <X> to understand the intended scope.
Then read ALL files the PR will touch and their adjacent files.

SCOPE CHECK: If the PR touches more than 8 files or spans more than
2 packages, flag it as SCOPE-WARNING in your deliverable. The lead
MUST decide whether to split the PR before proceeding.

PHASE 0 — COMPONENT CONTEXT

Identify this PR's place in the concept model from the macro plan:

1) Which building block is being added or modified?
2) What are the adjacent blocks it interacts with?
3) What invariants from the concept model does this PR touch?
4) What state ownership changes (if any)?

Then inspect the relevant parts of the repository.

List confirmed facts (with file:line citations):
- Current type definitions, method signatures, struct fields
- Current call sites that will be affected
- Current test coverage of the area being changed
- Adjacent code that might be affected

Flag anything from the macro plan that doesn't match current code as
a DEVIATION — these must be resolved before implementation begins.

PHASE 3 — DEVIATION LOG

Compare the macro plan entry for this PR against the actual codebase.

For each difference:

| Macro Plan Says | Code Actually Does | Reason |
|-----------------|-------------------|--------|

Categories of deviation:
- SIMPLIFICATION: Macro plan specified more than needed at this stage
- CORRECTION: Macro plan was wrong about existing code or behavior
- DEFERRAL: Feature moved to a later PR (explain why)
- ADDITION: Something the macro plan missed

If there are zero deviations, state "No deviations from macro plan."

Deliverable: Send your complete analysis to the team lead.
Include: all file:line citations, confirmed facts, deviations found.

======================================================================
TEAMMATE: PLAN DESIGNER
======================================================================

Role: Architect. Uses codebase-analyst's findings to design contracts,
component interactions, and implementation approach. This is the
creative core of the plan.

Produces: Phase 1 (Behavioral Contracts) + Phase 2 (Component
Interaction) + Phase 4 (Implementation Plan) + Phase 5
(Exercisability Proof)

Instructions:

Build on the codebase-analyst's confirmed facts and file:line
citations as your foundation. Do NOT re-read files the analyst
already covered unless you need to verify a specific detail.

PHASE 1 — BEHAVIORAL CONTRACTS (Human-Reviewable)

This is the most important section of the plan. It defines what this
PR guarantees. Use named contracts (BC-1, BC-2, ...) that can be
referenced in tests, reviews, and future PRs.

For each contract, use this format:

  BC-N: <Name>
  - GIVEN <precondition>
  - WHEN <action>
  - THEN <observable outcome>
  - MECHANISM: <one sentence explaining how> (optional but recommended)

Group contracts into:

1) Behavioral Contracts (what MUST happen)
   - Normal operation
   - Edge cases
   - Backward compatibility

2) Negative Contracts (what MUST NOT happen)
   - Invariant violations this PR could cause
   - Cross-boundary state leaks
   - Performance regressions

3) Error Handling Contracts
   - What happens on invalid input
   - What happens on resource exhaustion
   - Panic vs error return vs log-and-continue (be explicit)

TARGET: 3-15 contracts per PR. Pure refactoring PRs with no new
behavior may have as few as 3. More than 15 means the PR may be
too large.

No vague wording. "Should" is banned — use "MUST" or "MUST NOT."

PHASE 2 — COMPONENT INTERACTION (Human-Reviewable)

Describe this PR's building block and how it connects to the system.
This is the "box-and-arrow" view, NOT the file-level view.

1) Component Diagram (text-based)
   - This PR's component and its responsibility
   - Adjacent components (existing or new)
   - Data flow direction between them
   - What crosses each boundary (types, not implementations)

2) API Contracts
   - New interfaces or types (signature + one-line semantics)
   - Method preconditions and postconditions
   - Failure modes and how callers handle them

3) State Changes
   - New mutable state and its owner
   - State lifecycle (created when, destroyed when, accessed by whom)

TARGET: under 40 lines. Infrastructure PRs that introduce multiple
interacting types may go up to 60 lines with justification.
Beyond 60 lines, the PR scope is likely too broad.

PHASE 4 — IMPLEMENTATION PLAN

Start with key decisions and alternatives considered (3-5 lines).

Design-time assertions (confirm all before proceeding):
- No dead code introduced
- All new codepaths are exercisable (proven in Phase 5)
- No unused abstractions or speculative scaffolding

If you detect dead code risk, redesign before proceeding.

Then decompose into discrete, ordered tasks. Each IT-N entry
replaces a traditional file list — the task graph IS the
implementation plan, not a supplement to it.

  IT-N: <Short imperative description>
  - Files: <files touched (new or modified)>
  - Depends on: <IT-M, IT-K, ... or "nothing">
  - Contracts: <BC-N, BC-M, ... that this task implements or enables>
  - Verification: <command to run after completing this task>
  - Parallel: <yes/no — can run concurrently with non-dependent siblings?>

Ordering rules:
1) Types and interfaces before implementations
2) Implementations before tests (unless TDD — then invert)
3) Core logic before call-site updates
4) Tests for a contract adjacent to the task implementing it

Group tasks into BATCHES of 2-4. Align batch boundaries with
logical component boundaries (e.g., "types + constructors" then
"call-site migration" then "tests"). Each batch ends with a
verification checkpoint:

  Batch N checkpoint:
  - Build: go build ./...
  - Tests: go test ./... -run <relevant tests>
  - Contracts verified: BC-1, BC-3

TARGET: 4-12 tasks per PR. Fewer than 4 means the PR is trivial.
More than 12 means the PR scope may be too large.

PHASE 5 — EXERCISABILITY PROOF

Show how every new codepath is exercised. Acceptable exercise methods:

1) CLI commands (preferred for user-facing features)
   - Exact command
   - Expected observable behavior

2) Tests (valid for internal refactors and library code)
   - Test name and what contract it verifies
   - Why CLI exercise is not applicable

3) Existing test passthrough (valid for pure refactors)
   - Which existing tests exercise the refactored codepath
   - Why no new tests are needed

If a codepath cannot be exercised by ANY of these methods, it is
dead code. Redesign.

Deliverable: Send your complete design to the team lead.
Include: all contracts (BC-1..BC-N), component diagram, implementation
plan (task graph), exercisability proof.

======================================================================
TEAMMATE: PLAN REVIEWER
======================================================================

Role: Devil's advocate + test architect. Challenges the plan for
completeness, finds missing edge cases, maps contracts to tests,
identifies risks. This teammate ensures the plan survives expert review.

Produces: Phase 6 (Test Strategy) + Phase 6b (Verification Protocol)
+ Phase 7 (Risk Analysis & Review Guide)

Instructions:

This role has two groups of tasks with different dependencies:

From codebase-analyst's report (no dependency on plan-designer):
- Draft the verification protocol (Phase 6b)
- Identify risk areas and draft Phase 7 Part A risk entries

From plan-designer's output (after designer completes all phases):
- Map contracts to tests (Phase 6)
- Draft the review guide (Phase 7 Part B) — requires seeing contracts
  to identify which are hardest to verify
- Challenge contracts and task graph (Phase 7 Part C)
- Spot-check the designer's file:line citations in contracts and task
  graph — verify at least 3 against the actual codebase. Flag any
  that don't match as CITATION-MISMATCH.

PHASE 6 — TEST STRATEGY

Map contracts to tests:

| Contract | Test Type | Test Name / Description |
|----------|-----------|------------------------|
| BC-1     | Unit      | TestFoo_GivenX_ThenY   |
| BC-2     | Golden    | TestFoo_GoldenEquiv    |
| ...      | ...       | ...                    |

Test types:
- Unit: specific function behavior
- Integration: cross-component or CLI-level
- Golden: regression against known-good output
- Failure: error paths, panics, edge cases
- Benchmark: performance-sensitive paths (optional)

Additional requirements:
- Shared test infrastructure: use existing helpers (e.g.,
  sim/internal/testutil or equivalent). If new helpers are needed,
  add them to the shared package — not locally in the test file.
- Golden dataset: if this PR changes output format or adds new
  metrics, document how golden data should be updated.
- Lint: `golangci-lint run ./...` must pass with zero new issues.

For EACH contract, verify:
- Is the contract testable as written?
- Are there missing edge cases the contract doesn't cover?
- Is the contract's MECHANISM actually how the code works? (cite
  file:line to confirm or refute)

PHASE 6b — VERIFICATION PROTOCOL

Define the verification sequence implementing teammates MUST follow.
Correctness is checked incrementally, not just at the end.

Three levels:
1) Per-task: command to run after each IT-N (typically `go build`
   or a targeted test). Specify which contract is likely violated
   on failure.
2) Per-batch: full test suite for completed contracts + lint check
   (`golangci-lint run ./...`). List which contracts are now verified.
3) Final: `go build ./...` + `go test ./...` + `golangci-lint run
   ./...` + golden dataset regression (if applicable) + CLI exercise
   commands from Phase 5.

Failure rule: on ANY verification failure, the implementing teammate
MUST stop and diagnose before proceeding. Never skip a failing
checkpoint.

PHASE 7 — RISK ANALYSIS & REVIEW GUIDE

PART A: Risks

For each risk:
- Risk description
- Likelihood (low/medium/high)
- Impact (low/medium/high)
- Mitigation (specific test or design choice)

Actively look for:
- Golden test breakage (event ordering, RNG sequence, metric values)
- Invariant violations (clock monotonicity, KV conservation, etc.)
- Missing error handling (what if input is nil? what if queue is empty?)
- Performance regressions on the hot path
- Contracts that are incomplete or untestable
- Dead code the plan-designer didn't notice

PART B: Review Guide (for the human reviewer)

In 5-10 lines, tell the reviewer:

1) THE TRICKY PART: What's the most subtle or error-prone aspect?
2) WHAT TO SCRUTINIZE: Which contract(s) are hardest to verify?
3) WHAT'S SAFE TO SKIM: Which parts are mechanical/boilerplate?
4) KNOWN DEBT: Any pre-existing issues encountered but not fixed?

This section exists because human attention is scarce.
Direct it to where it matters most.

PART C: Challenges to Plan Designer

List specific challenges to the plan-designer's work:
- Contracts you believe are incomplete or incorrect (cite evidence)
- Missing negative contracts
- Exercisability gaps
- Implementation decisions you disagree with (cite alternatives)
- Task graph issues: circular dependencies, missing contract coverage,
  incorrect parallelization claims, insufficient verification commands

The lead MUST address each challenge in the final document — either
by accepting the reviewer's feedback or explaining why it's rejected.

Deliverable: Send your complete review to the team lead.
Include: test strategy table, verification protocol, risk analysis,
review guide, and specific challenges to the plan-designer's output.

======================================================================
LEAD: SYNTHESIS
======================================================================

After all three teammates report:

1) Resolve conflicts between teammates. If the reviewer challenged a
   contract, either amend the contract or document why the challenge
   is rejected. Every challenge must be addressed.

2) Complete Phase 8 — DESIGN SANITY CHECKLIST:

   Before implementation, verify:
   - [ ] No unnecessary abstractions.
   - [ ] No feature creep beyond PR scope.
   - [ ] No unexercised flags or interfaces.
   - [ ] No partial implementations.
   - [ ] No breaking changes without explicit contract updates.
   - [ ] No hidden global state impact.
   - [ ] All new code will pass golangci-lint.
   - [ ] Shared test helpers used from existing shared test package
         (not duplicated locally).
   - [ ] CLAUDE.md updated if: new files/packages added, file
         organization changed, plan milestone completed, new CLI
         flags added.
   - [ ] No stale references left in CLAUDE.md.
   - [ ] Deviation log reviewed — no unresolved deviations.
   - [ ] All reviewer challenges addressed (accepted or rejected
         with justification).
   - [ ] Task graph has no circular dependencies.
   - [ ] Every contract (BC-N) is covered by at least one task (IT-N).
   - [ ] Every task has a verification command.
   - [ ] Parallelization claims are correct (no shared-state conflicts
         between tasks marked parallel).
   - [ ] Batch checkpoints verify all contracts completed so far.

3) Assemble the final document in the output format below.

4) Write the document to:
   docs/plans/pr<X>-micro-plan.md

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

--- PART 1: Human Review (target: under 120 lines) ---

A) Executive Summary       — 5-10 lines; building block, adjacent
                             blocks, DEVIATION/SCOPE-WARNING flags
B) Behavioral Contracts    — Phase 1 (BC-1..BC-N)
C) Component Interaction   — Phase 2
D) Deviation Log           — Phase 3
E) Review Guide            — Phase 7, Part B

--- PART 2: Implementation Reference ---

F) Implementation Plan     — Phase 4 (key decisions + task graph)
G) Exercisability Proof    — Phase 5
H) Test Strategy           — Phase 6
I) Verification Protocol   — Phase 6b
J) Risk Analysis           — Phase 7, Part A
K) Reviewer Challenges     — Phase 7, Part C + lead resolutions
L) Sanity Checklist        — Phase 8

--- PART 3: Execution Details ---

M) Commit Strategy: one commit per batch (default); reference contract
   IDs in messages; final commit includes CLAUDE.md updates.

--- APPENDIX ---

N) File-level reference: exact signatures, struct fields, constructor
   params, event logic, metric rules, behavioral subtleties with
   file:line citations. No length limit. This is REFERENCE MATERIAL —
   execution order is defined by the task graph in Section F.

======================================================================

Quality bar:

- Must survive expert review and systems-level scrutiny.
- Must eliminate dead code and reduce implementation bug risk.
- Must remain within macro plan scope (deviations logged).
- Must pass golangci-lint with zero new issues.
- All reviewer challenges addressed in the final document.
- Task graph complete: every contract mapped, dependencies acyclic.
- Verification catches regressions incrementally, not just at the end.

======================================================================

Lint: `golangci-lint run ./...` (version pinned in CI). All new code
must pass with zero issues. Do not fix pre-existing lint in unrelated
code (scope creep).

Think carefully. Inspect deeply. Design defensively.
Challenge each other's work. Direct the reviewer's attention wisely.
