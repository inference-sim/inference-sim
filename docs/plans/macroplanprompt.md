You are operating inside a real repository with full code access.

You are tasked with producing a MACRO-LEVEL DESIGN PLAN for a:

“Major Feature Expansion with Architectural Changes.”

This is a program-level plan that defines objectives, architectural evolution, and an ordered PR series.

This is NOT implementation planning.
This is NOT a micro-level design.
This is NOT speculative architecture.

You MUST inspect the real codebase before proposing changes.

======================================================================
PHASE 0 — REPOSITORY RECON (MANDATORY)
======================================================================

Before proposing anything:

1) Identify and summarize:
   - Top-level packages/modules and responsibilities
   - Core data structures and interfaces
   - Key invariants and assumptions encoded in the system
   - CLI entrypoints and current flag surface
   - Configuration flow
   - Existing extension points
   - Areas of tight coupling or fragility

2) Clearly separate:
   - Confirmed facts (from inspection)
   - Inferred behavior (explicitly labeled)
   - Open uncertainties

3) Identify architectural constraints that must not be violated.

No invented abstractions.
No imagined extension points.
Everything must be grounded in code inspection.

======================================================================
PHASE 1 — HIGH-LEVEL OBJECTIVES
======================================================================

Define:

- 3–7 crisp objectives
- Explicit non-goals
- Compatibility constraints
- Performance constraints
- Backward compatibility guarantees
- Operational/CLI stability expectations

Be precise.

======================================================================
PHASE 2 — PROPOSED ARCHITECTURAL EVOLUTION
======================================================================

Only after recon:

- Describe how the architecture must evolve.
- Specify what changes structurally.
- Identify new extension points (if any).
- Identify refactors that are strictly enabling (no behavior change).
- Explicitly describe what remains unchanged.

Highlight risks and invariants.

No premature generalization.

======================================================================
PHASE 3 — ORDERED PR SERIES (PR0 … PRN)
======================================================================

Design an incremental, community-consumable PR sequence.

For EACH PR include:

- Title
- Motivation / User Value
- Scope (In / Out)
- Architectural Impact
- Behavioral Guarantees (high-level)
- API Surface Changes (if any)
- CLI Invocation Examples (exact commands)
- Test Categories (high-level)
- README Changes (concise)
- Risks + Mitigations
- Why this PR is independently reviewable
- Why it introduces no dead code

Constraints:

- Each PR must deliver one cohesive feature.
- Each PR must be usable immediately after merge.
- No speculative scaffolding.
- No unused interfaces.
- No flags that aren’t exercised.
- If code cannot be triggered via CLI → redesign.

======================================================================
PHASE 4 — DEPENDENCY DAG & PARALLELISM
======================================================================

Provide:

- A PR dependency graph (partial order).
- Parallelizable workstreams.
- Merge sequencing guidance.
- Integration risk notes.

Maximize safe parallelism.

======================================================================
PHASE 5 — DESIGN BUG PREVENTION
======================================================================

Include:

- Invariants that must never be broken.
- Regression surfaces.
- Observability/testing hooks required.
- Backward compatibility enforcement.
- Common architectural failure modes and how this plan prevents them.

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

A) Executive Summary  
B) Repository Recon Summary  
C) High-Level Objectives + Non-Goals  
D) Architectural Evolution  
E) PR Plan (PR0…PRN)  
F) Dependency DAG  
G) Design Bug Prevention Checklist  

======================================================================

Quality bar:

- Grounded in real code.
- No hallucinated modules.
- No dead code.
- No bloated PRs.
- Must withstand Go expert and systems-level review.
- Must be realistic and implementable.

Think deeply before answering.
Inspect before designing.
