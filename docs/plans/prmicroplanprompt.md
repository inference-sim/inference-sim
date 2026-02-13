You are operating inside a real repository with full code access.

You are tasked with producing a PR-SPECIFIC MICRO-DESIGN PLAN.

This is for PR <X> from the approved Macro Plan.

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
PHASE 0 — COMPONENT CONTEXT
======================================================================

Identify this PR's place in the concept model from the macro plan:

1) Which building block is being added or modified?
2) What are the adjacent blocks it interacts with?
3) What invariants from the concept model does this PR touch?
4) What state ownership changes (if any)?

Then inspect ONLY the relevant parts of the repository.

List confirmed facts (with file:line citations).
Flag anything from the macro plan that doesn't match current code as
a DEVIATION — these must be resolved before implementation begins.

======================================================================
PHASE 1 — BEHAVIORAL CONTRACTS (Human-Reviewable)
======================================================================

This is the most important section of the plan. It defines what this PR
guarantees. Use named contracts (BC-1, BC-2, ...) that can be referenced
in tests, reviews, and future PRs.

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

TARGET: 3-15 contracts per PR. Pure refactoring PRs with no new behavior
may have as few as 3. More than 15 means the PR may be too large.

No vague wording. "Should" is banned — use "MUST" or "MUST NOT."

======================================================================
PHASE 2 — COMPONENT INTERACTION (Human-Reviewable)
======================================================================

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

======================================================================
PHASE 3 — DEVIATION LOG
======================================================================

Compare this micro plan against the macro plan for this PR.

For each difference:

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|

Categories of deviation:
- SIMPLIFICATION: Macro plan specified more than needed at this stage
- CORRECTION: Macro plan was wrong about existing code or behavior
- DEFERRAL: Feature moved to a later PR (explain why)
- ADDITION: Something the macro plan missed

If there are zero deviations, state "No deviations from macro plan."

======================================================================
PHASE 4 — IMPLEMENTATION SUMMARY
======================================================================

A concise overview of the implementation approach:

1) Files to modify (with one-line description of each change)
2) New files to create (justify each — prefer modifying existing files)
3) Key implementation decisions and alternatives considered

Explicitly confirm (these are design-time assertions, verified in Phase 8):
- No dead code introduced
- All new codepaths are exercisable (proven in Phase 5)
- No unused abstractions
- No speculative scaffolding

If you detect dead code risk → redesign before proceeding.

KEEP THIS SECTION UNDER 30 LINES.
Detailed file-level plans go in the Appendix.

======================================================================
PHASE 5 — EXERCISABILITY PROOF
======================================================================

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

======================================================================
PHASE 6 — TEST STRATEGY
======================================================================

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
- Shared test infrastructure: use existing helpers (e.g., sim/internal/testutil
  or equivalent shared package). If new helpers are needed, add them to the
  shared package — not locally in the test file.
- Golden dataset: if this PR changes output format or adds new metrics,
  document how golden data should be updated.
- Lint: `golangci-lint run ./...` must pass with zero new issues.

======================================================================
PHASE 7 — RISK ANALYSIS & REVIEW GUIDE
======================================================================

PART A: Risks

For each risk:
- Risk description
- Likelihood (low/medium/high)
- Impact (low/medium/high)
- Mitigation (specific test or design choice)

PART B: Review Guide (for the human reviewer)

In 5-10 lines, tell the reviewer:

1) THE TRICKY PART: What's the most subtle or error-prone aspect?
2) WHAT TO SCRUTINIZE: Which contract(s) are hardest to verify?
3) WHAT'S SAFE TO SKIM: Which parts are mechanical/boilerplate?
4) KNOWN DEBT: Any pre-existing issues encountered but not fixed?

This section exists because human attention is scarce.
Direct it to where it matters most.

======================================================================
PHASE 8 — DESIGN SANITY CHECKLIST
======================================================================

Before implementation, verify:

- [ ] No unnecessary abstractions.
- [ ] No feature creep beyond PR scope.
- [ ] No unexercised flags or interfaces.
- [ ] No partial implementations.
- [ ] No breaking changes without explicit contract updates.
- [ ] No hidden global state impact.
- [ ] All new code will pass golangci-lint.
- [ ] Shared test helpers used from existing shared test package (not duplicated locally).
- [ ] CLAUDE.md updated if: new files/packages added, file organization
      changed, plan milestone completed, new CLI flags added.
- [ ] No stale references left in CLAUDE.md.
- [ ] Deviation log reviewed — no unresolved deviations.

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

--- PART 1: Human Review (target: under 120 lines total) ---

A) Executive Summary (5-10 lines, include: which building block from the
   concept model, adjacent blocks, and any DEVIATION flags from Phase 0)
B) Behavioral Contracts (Phase 1)
C) Component Interaction (Phase 2)
D) Deviation Log (Phase 3)
E) Review Guide (Phase 7, Part B)

--- PART 2: Implementation Reference ---

F) Implementation Summary (Phase 4)
G) Exercisability Proof (Phase 5)
H) Test Strategy (Phase 6)
I) Risk Analysis (Phase 7, Part A)
J) Sanity Checklist (Phase 8)

--- APPENDIX: File-Level Details (for execution, not review) ---

K) Detailed file changes, exact method signatures, RNG call sequences,
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
Direct the reviewer's attention wisely.
