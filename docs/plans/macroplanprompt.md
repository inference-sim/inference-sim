You are operating inside a real repository with full code access.

You are tasked with producing a MACRO-LEVEL DESIGN PLAN for a:

"Major Feature Expansion with Architectural Changes."

This is a program-level plan that defines objectives, a concept model,
architectural evolution, and an ordered PR series.

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
   - Confirmed facts (from inspection — cite file:line for every claim)
   - Inferred behavior (explicitly labeled as inference)
   - Open uncertainties

3) Identify architectural constraints that must not be violated.

No invented abstractions.
No imagined extension points.
Everything must be grounded in code inspection with source references.

ANTI-HALLUCINATION RULE: For every behavioral claim about existing code,
provide a file:line citation. If you cannot cite it, mark it as
"UNVERIFIED" and do not rely on it in subsequent phases.

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
PHASE 2 — CONCEPT MODEL
======================================================================

Before diving into architecture, define the system at the level a human
would explain it on a whiteboard:

1) Building Blocks (3-7 named components)
   - Name and one-sentence responsibility for each
   - Ownership: what mutable state does each block own exclusively?
   - No building block may have more than one core responsibility

2) Interaction Model
   - Who calls whom (directional arrows)
   - Data flow between blocks (what crosses each boundary)
   - Ownership transfer rules (when does data change owners?)

3) System Invariants
   - What must ALWAYS hold (e.g., "clock never decreases")
   - What must NEVER happen (e.g., "no cross-instance state mutation")
   - Causality constraints (ordering guarantees)

4) Extension Points
   - Where do new behaviors plug in? (interface name + responsibility)
   - What is the default behavior for each extension point?

5) State Ownership Map
   - For every piece of mutable state: exactly one owner
   - Shared state must be explicitly identified and justified

THE CONCEPT MODEL MUST FIT IN UNDER 60 LINES.
If it doesn't, the design is too complex — simplify before proceeding.

Every PR in Phase 6 must map to adding or modifying a specific building
block from this model. If a PR cannot be described as a building block
change, redesign the PR or the model.

======================================================================
PHASE 3 — ARCHITECTURAL RISK REGISTER
======================================================================

For every non-obvious architectural decision in the concept model:

| Decision | Assumption | Validation Method | Cost if Wrong | Gate |
|----------|------------|-------------------|---------------|------|

- DECISION: The choice being made
- ASSUMPTION: What must be true for this to work
- VALIDATION: How to test cheaply (mock study, prototype, analysis, spike)
- COST IF WRONG: What breaks — count the affected PRs
- GATE: When validation must complete (before which PR)

Example row:
| Shared-clock event loop | O(N) scan per event is fast for N≤16 |
  Benchmark N=16, 10K events | PR 3 rework | Before PR 3 merge |

MANDATORY VALIDATION RULE:
If cost-of-being-wrong ≥ 3 PRs of rework, validation is MANDATORY.
The plan must include a spike/mock study PR or pre-PR validation step.

For each validation gate, specify:
- Exact success criteria (not "looks good" — measurable outcomes)
- Abort plan (what changes if validation fails)

======================================================================
PHASE 4 — PROPOSED ARCHITECTURAL EVOLUTION
======================================================================

Only after the concept model and risk register:

- Describe how the architecture evolves FROM current TO concept model.
- Map each structural change to a concept model building block.
- Identify refactors that are strictly enabling (no behavior change).
- Explicitly describe what remains unchanged.
- For each new extension point: what is the default implementation and
  when does the first non-default implementation arrive?

Highlight risks and invariants.

No premature generalization.
No extension point without a concrete non-default implementation planned.

======================================================================
PHASE 5 — CROSS-CUTTING INFRASTRUCTURE
======================================================================

Plan ONCE for the entire PR series. Each item must be assigned to a
specific PR (defined in Phase 6) or handled as a standalone preparatory
PR. Phases 5 and 6 are co-developed: sketch the PR series first, then
assign cross-cutting items, then finalize both.

1) Shared Test Infrastructure
   - First: identify existing shared test packages in the codebase.
     Build on them rather than duplicating or replacing them.
   - New test helper packages, shared fixtures, golden dataset types
   - Which PR creates them? Which PRs consume them?
   - How do golden datasets evolve as the system grows?

2) Documentation Maintenance
   - CLAUDE.md update triggers: new packages, new files, changed file
     organization, completed plan milestones, new CLI flags
   - Who updates CLAUDE.md? (The PR that causes the change.)
   - README update triggers and ownership

3) CI Pipeline Changes
   - New test packages to add to CI
   - New linter rules or build steps
   - Performance regression benchmarks

4) Dependency Management
   - New external dependencies (justify each one)
   - Version pinning strategy

No item may be left as "address when needed."
This applies to cross-cutting infrastructure (test helpers, CI, docs),
not to feature packages which are detailed in Phase 6.

======================================================================
PHASE 6 — ORDERED PR SERIES (PR0 … PRN)
======================================================================

Design an incremental, independently reviewable and mergeable PR sequence.

For EACH PR, provide TWO TIERS:

--- TIER 1: Human Review Summary (target 15 lines, max 25) ---

- Title
- Building Block Change: Which concept model block is added/modified?
- Motivation: Why does this PR exist? (1-2 sentences)
- Scope: In / Out (bullet points)
- Behavioral Guarantees: What MUST hold after this PR merges?
  (Use named contracts: BC-1, BC-2, etc.)
- Risks: Top 1-2 risks and how they're mitigated
- Cross-Cutting: Which shared infra does this PR create or consume?
- Validation Gate: Does this PR depend on a risk register validation?

--- TIER 2: Implementation Guide (for micro-planning) ---

- Architectural Impact (what changes structurally)
- API Surface Changes (new types, interfaces, methods)
- CLI Changes (new flags, changed behavior)
- Test Categories (unit, integration, regression, golden)
- Documentation Updates (CLAUDE.md, README)
- Why this PR is independently reviewable
- Why it introduces no dead code

Constraints:

- Each PR must deliver one cohesive building block change.
- Each PR must be exercisable immediately after merge.
  "Exercisable" means: via CLI, OR via tests that demonstrate the
  new behavior. Internal refactors exercised by passing existing tests
  are valid. Scaffolding exercised only by future PRs is NOT valid.
- No speculative scaffolding.
- No unused interfaces.
- No flags that aren't exercised.

======================================================================
PHASE 7 — DEPENDENCY DAG & PARALLELISM
======================================================================

Provide:

- A PR dependency graph (partial order).
- Parallelizable workstreams.
- Merge sequencing guidance.
- Validation gate placement (from risk register).
- Integration risk notes.

Maximize safe parallelism.

======================================================================
PHASE 8 — DESIGN BUG PREVENTION
======================================================================

Include:

- Invariants that must never be broken (reference concept model).
- Regression surfaces (which existing tests must keep passing).
- Cross-PR state migration risks (data format changes across PRs).
- Backward compatibility enforcement.
- Common architectural failure modes and how this plan prevents them:
  - Scaffolding creep (dead code introduced "for later")
  - Documentation drift (CLAUDE.md diverges from reality)
  - Test infrastructure duplication (helpers copied across packages)
  - Golden dataset staleness (regression baselines not updated)
  - Interface over-specification (freezing APIs too early)

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

A) Executive Summary (under 15 lines — synthesize the elevator pitch:
   what is being built, why, how many PRs, key milestones)
B) Repository Recon Summary
C) High-Level Objectives + Non-Goals
D) Concept Model (under 60 lines — the "whiteboard picture")
E) Architectural Risk Register
F) Architectural Evolution (current → target, mapped to concept model)
G) Cross-Cutting Infrastructure Plan
H) PR Plan (PR0…PRN, Tier 1 + Tier 2 per PR)
I) Dependency DAG
J) Design Bug Prevention Checklist

CONTEXT BUDGET RULE:
Sections A, C, and D are the human-review core and must be concise.
H-Tier-1 summaries should target 15 lines each (max 25).
All other sections are reference material consulted on demand.
The plan should be structured so a human can review the core sections
(A + C + D + all H-Tier-1 summaries) without needing to read the rest.

======================================================================

Quality bar:

- Grounded in real code with file:line citations.
- No hallucinated modules or behaviors.
- No dead code.
- No bloated PRs.
- Must withstand expert review.
- Must be realistic and implementable.
- Concept model must be simple enough to explain verbally in 2 minutes.

======================================================================
LIVING DOCUMENT PROTOCOL
======================================================================

This plan will evolve. When updating:

1) Add a dated revision note at the top explaining what changed and why.
2) If a risk register validation fails, document the finding and the
   resulting plan changes explicitly.
3) Never silently change a PR's behavioral guarantees — if contracts
   change, note the old contract, new contract, and reason.
4) Track completed PRs by marking their status in the PR plan section.

======================================================================

Think deeply before answering.
Inspect before designing.
Validate before committing.
