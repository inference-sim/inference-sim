You are operating inside a real repository with full code access.

You are tasked with producing a PR-SPECIFIC MICRO-DESIGN PLAN.

This is for PR <X> from the approved Macro Plan.

This is not a restatement of the macro plan.
This is deep implementation planning.

No coding yet.
Only design.

Use behavior driven development methodology for each PR. 

======================================================================
PHASE 0 — TARGETED RECON
======================================================================

Re-inspect ONLY the parts of the repository that this PR touches.

Identify:

- Exact files/modules impacted
- Current behavior of those codepaths
- Relevant invariants
- Data flow across boundaries
- CLI flag parsing paths
- Config interactions
- Concurrency assumptions (if any)

List confirmed facts.

======================================================================
PHASE 1 — CONTRACT EXPANSION
======================================================================

Expand this PR’s contracts into precise guarantees.

Define:

1) Behavioral Contracts
   - What MUST happen
   - What MUST NOT happen
   - Edge case behavior
   - Error handling guarantees
   - Backward compatibility expectations

2) API Contracts
   - Exact interface/type changes
   - Method semantics
   - Input/output invariants
   - Thread-safety guarantees (if relevant)
   - Failure modes

Be explicit.

No vague wording.

======================================================================
PHASE 2 — IMPLEMENTATION PLAN
======================================================================

Provide:

- Files to modify
- New files (if absolutely necessary)
- Code movement (if refactoring is enabling)
- Control flow changes
- Validation logic
- Error handling additions
- Logging/observability hooks

Explicitly confirm:

- No dead code introduced
- All new codepaths exercised via CLI
- No unused abstractions introduced
- No speculative scaffolding
- All new code passes `golangci-lint`

If you detect dead code risk → redesign before proceeding.

======================================================================
PHASE 3 — CLI EXERCISE PROOF
======================================================================

Show EXACT commands that:

- Trigger every new codepath
- Demonstrate expected output
- Validate feature behavior

If full coverage cannot be demonstrated via CLI:
→ The design is incomplete.

======================================================================
PHASE 4 — TEST MATRIX
======================================================================

Define:

- Unit tests (specific functions)
- Integration tests (CLI level)
- Regression tests
- Failure mode tests
- Golden tests (if applicable)
- Concurrency tests (if applicable)
- Lint verification (golangci-lint)

Each test must map to a specific contract.

Lint requirement:
- Run `golangci-lint run ./...` on all new/modified packages
- All new code must pass with zero issues
- Pre-existing issues in untouched code may be noted but not fixed in this PR

======================================================================
PHASE 5 — RISK ANALYSIS
======================================================================

Identify:

- Invariant break risks
- Performance risks
- Backward compatibility risks
- Hidden coupling risks
- Observability gaps

Explain how tests or design prevent each.

======================================================================
PHASE 6 — DESIGN SANITY CHECKLIST
======================================================================

Before implementation, verify:

- No unnecessary abstractions.
- No feature creep beyond PR scope.
- No unexercised flags.
- No partial implementations.
- No breaking changes without explicit contract updates.
- No hidden global state impact.
- All new code will pass golangci-lint.

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

A) Executive summary
B) Targeted Recon Summary  
C) Expanded Contracts  
D) Detailed Implementation Plan  
E) CLI Exercise Proof  
F) Test Matrix  
G) Risk Analysis  
H) Sanity Checklist  

======================================================================

Quality bar:

- Must survive expert Go review.
- Must survive systems-level scrutiny.
- Must eliminate dead code.
- Must reduce risk of implementation bugs.
- Must remain strictly within the scope defined in the Macro Plan.
- Must pass golangci-lint with zero new issues.

======================================================================
LINTING REQUIREMENTS
======================================================================

This project uses golangci-lint for static analysis. The same linter
configuration runs locally and in CI.

Local verification (run before submitting PR):
```bash
golangci-lint run ./...
```

CI verification (.github/workflows/ci.yml):
- Uses golangci/golangci-lint-action@v6
- Same version as local development

Rules:
1. All NEW code must pass lint with zero issues.
2. Do not fix pre-existing lint issues in unrelated code (scope creep).
3. If a lint rule seems wrong, document why and discuss before disabling.

======================================================================

Think carefully.
Inspect deeply.
Design defensively.
