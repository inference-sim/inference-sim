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

TEAM STRUCTURE (3 agents + lead):

  Agent 1: "codebase-analyst"    — Reads code, finds facts
  Agent 2: "plan-designer"       — Designs contracts + architecture
  Agent 3: "plan-reviewer"       — Challenges everything, designs tests

  Lead (you): Coordinates, synthesizes final document

EXECUTION ORDER:

  1. Spawn "codebase-analyst" (FIRST, solo — others depend on it)
  2. Wait for codebase-analyst to complete
  3. Spawn "plan-designer" and "plan-reviewer" (PARALLEL — reviewer
     starts from codebase analysis while designer works on contracts)
  4. When plan-designer finishes, send its contracts to plan-reviewer
     (reviewer maps contracts to tests and challenges them)
  5. Wait for both to complete
  6. Synthesize all outputs into the final micro plan document

GROUND RULES FOR ALL AGENTS:
- Every behavioral claim about existing code MUST cite file:line
- Anti-hallucination: if you cannot cite it, mark it UNVERIFIED
- "Should" is banned — use "MUST" or "MUST NOT"
- Read the actual code, not just CLAUDE.md or macro plan
- If you disagree with another agent's finding, say so and cite evidence

======================================================================
AGENT 1: CODEBASE ANALYST
======================================================================

Role: Deep code reader. Produces the factual foundation that all
other agents build on. Read-only — no design decisions.

Produces: Phase 0 (Component Context) + Phase 3 (Deviation Log)

Instructions:

Read the macro plan entry for PR <X> to understand the intended scope.
Then read ALL files the PR will touch and their adjacent files.

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
AGENT 2: PLAN DESIGNER
======================================================================

Role: Architect. Uses codebase-analyst's findings to design contracts,
component interactions, and implementation approach. This is the
creative core of the plan.

Produces: Phase 1 (Behavioral Contracts) + Phase 2 (Component
Interaction) + Phase 4 (Implementation Summary) + Phase 5
(Exercisability Proof)

Instructions:

Wait for codebase-analyst's report. Use its confirmed facts and
file:line citations as your foundation. Do NOT re-read files the
analyst already covered unless you need to verify a specific detail.

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

PHASE 4 — IMPLEMENTATION SUMMARY

A concise overview of the implementation approach:

1) Files to modify (with one-line description of each change)
2) New files to create (justify each — prefer modifying existing files)
3) Key implementation decisions and alternatives considered

Explicitly confirm (these are design-time assertions):
- No dead code introduced
- All new codepaths are exercisable (proven in Phase 5)
- No unused abstractions
- No speculative scaffolding

If you detect dead code risk, redesign before proceeding.

KEEP THIS SECTION UNDER 30 LINES.
Detailed file-level plans go in the Appendix.

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
summary, exercisability proof.

======================================================================
AGENT 3: PLAN REVIEWER
======================================================================

Role: Devil's advocate + test architect. Challenges the plan for
completeness, finds missing edge cases, maps contracts to tests,
identifies risks. This agent ensures the plan survives expert review.

Produces: Phase 6 (Test Strategy) + Phase 7 (Risk Analysis & Review
Guide)

Instructions:

Start by reading the codebase-analyst's report. Begin designing the
test infrastructure and identifying risk areas while the plan-designer
works. When the plan-designer's contracts arrive, map them to tests
and challenge them.

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

The lead MUST address each challenge in the final document — either
by accepting the reviewer's feedback or explaining why it's rejected.

Deliverable: Send your complete review to the team lead.
Include: test strategy table, risk analysis, review guide, and
specific challenges to the plan-designer's output.

======================================================================
LEAD: SYNTHESIS
======================================================================

After all three agents report:

1) Resolve conflicts between agents. If the reviewer challenged a
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

3) Assemble the final document in the output format below.

4) Write the document to:
   docs/plans/pr<X>-micro-plan.md

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

--- PART 1: Human Review (target: under 120 lines total) ---

A) Executive Summary (5-10 lines, include: which building block from
   the concept model, adjacent blocks, and any DEVIATION flags)
B) Behavioral Contracts (Phase 1)
C) Component Interaction (Phase 2)
D) Deviation Log (Phase 3)
E) Review Guide (Phase 7, Part B)

--- PART 2: Implementation Reference ---

F) Implementation Summary (Phase 4)
G) Exercisability Proof (Phase 5)
H) Test Strategy (Phase 6)
I) Risk Analysis (Phase 7, Part A)
J) Reviewer Challenges + Resolutions
K) Sanity Checklist (Phase 8)

--- APPENDIX: File-Level Details (for execution, not review) ---

L) Detailed file changes, exact method signatures, RNG call sequences,
   data structure layouts, and other implementation specifics.

   This section has no length limit. It should contain everything
   needed to implement the PR without further codebase exploration.

   Include:
   - Exact function signatures with doc comments
   - Constructor parameter lists
   - Struct field definitions
   - Event execution logic
   - Metric aggregation rules
   - Any behavioral subtlety (e.g., horizon boundary semantics,
     append-slice behavior to preserve) with file:line citations

======================================================================

Quality bar:

- Must survive expert review.
- Must survive systems-level scrutiny.
- Must eliminate dead code.
- Must reduce risk of implementation bugs.
- Must remain strictly within the scope defined in the Macro Plan
  (deviations must be logged and justified).
- Must pass golangci-lint with zero new issues.
- Every reviewer challenge must be addressed in the final document.

======================================================================
LINTING REQUIREMENTS
======================================================================

This project uses golangci-lint for static analysis.
Version is pinned in CI (see .github/workflows/ci.yml).

Local verification (run before submitting PR):
```bash
golangci-lint run ./...
```

Rules:
1. All NEW code must pass lint with zero issues.
2. Do not fix pre-existing lint issues in unrelated code (scope creep).
3. If a lint rule seems wrong, document why and discuss before disabling.

======================================================================

Think carefully.
Inspect deeply.
Design defensively.
Challenge each other's work.
Direct the reviewer's attention wisely.
