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

Each test must map to a specific contract.

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

Think carefully.
Inspect deeply.
Design defensively.
