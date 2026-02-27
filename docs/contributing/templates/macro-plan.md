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
PREREQUISITE — DESIGN GUIDELINES
======================================================================

Before writing a macro plan, read and internalize:

- `docs/contributing/templates/design-guidelines.md` — BLIS design guidelines
  covering DES foundations, module architecture, and extension framework.

The macro plan must be consistent with these guidelines. Specifically:
- Building blocks must use the MODULE CONTRACT TEMPLATE from Section 4.3
  (observes / controls / owns / invariants / events / extension friction)
- New events must be classified as EXOGENOUS or ENDOGENOUS (Section 2.2)
- Modeling decisions must apply the SIX SCOPING CRITERIA (Section 2.1)
- PRs must identify their EXTENSION TYPE (Section 5.1): policy template,
  subsystem module, backend swap, or tier composition
- Building blocks should map to REAL SYSTEM COMPONENTS (Section 4.4)

If no design doc exists for the feature being planned, one must be
created (per the guidelines) before or alongside the macro plan.

======================================================================
ABSTRACTION LEVEL RULE (NON-NEGOTIABLE)
======================================================================

The macro plan describes WHAT to build and in WHAT ORDER, not HOW to
implement each piece. Enforce these boundaries strictly:

ALLOWED in macro plan:
  - Behavioral descriptions of module contracts (prose)
  - Frozen interface signatures (Go code ONLY for interfaces whose
    freeze PR has already merged — these are facts, not aspirations)
  - File path inventories (which packages, which files, per PR)
  - LOC estimates per PR (sizing heuristics)
  - CLI flag names and configuration surface area
  - Brief YAML/config examples
  - Architecture diagrams (text-based)

PROHIBITED in macro plan:
  - Method implementations (belongs in micro plan)
  - Struct field lists (belongs in micro plan)
  - Pre-freeze interface signatures in Go syntax (describe behaviorally:
    "a single-method interface that selects a target instance given
    request metadata and per-instance snapshots")
  - Factory function code (belongs in micro plan)
  - Test code (belongs in micro plan)

THE TEST: Is this content a FACT about merged code, or an ASPIRATION
about code to be written? Facts are allowed. Aspirations must be
described behaviorally, not as Go code.

WHY THIS MATTERS: The original macro plan had ~200 lines of Go code
including full method implementations (TokenBucket.Admit(),
PartitionedRNG.ForSubsystem()). These diverged during micro-planning
and became misleading. Behavioral descriptions survive intact because
they describe WHAT, not HOW.

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
   - Existing extension points (map to design guidelines Section 4.2)
   - Areas of tight coupling or fragility
   - Current module boundaries vs. target module map (guidelines 4.2)

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
PHASE 1 — HIGH-LEVEL OBJECTIVES AND MODEL SCOPING
======================================================================

Define:

- 3-7 crisp objectives
- Explicit non-goals
- Compatibility constraints
- Performance constraints
- Backward compatibility guarantees
- Operational/CLI stability expectations

Be precise.

MODEL SCOPING (required — applies Banks et al. criteria):

For this feature expansion, answer:

1) What ANALYSIS QUESTIONS does this feature help answer?
   (e.g., "What is the optimal routing policy for heterogeneous
   hardware?" or "How does autoscaling latency affect tail TTFT?")

2) What must be MODELED to answer those questions?

3) What can be SIMPLIFIED without affecting the analysis?
   (e.g., "model scaling latency as fixed delay, not warmup curve")

4) What can be OMITTED entirely?
   (e.g., "network partitions between instances — out of scope")

Present as a table:

| Component | Modeled | Simplified | Omitted | Justification |
|-----------|---------|------------|---------|---------------|
| (example) Scaling latency | -- | Fixed delay, not warmup curve | -- | Same steady-state throughput; warmup matters only for sub-minute scale-up |
| (example) Network partitions | -- | -- | Yes | Not needed for routing policy comparison; add if modeling failure recovery |

For each "Simplified" entry, state what real-system behavior is lost
and under what conditions it would matter in the Justification column.
This is the fidelity trade-off record — it prevents "fidelity for its
own sake" and enables future refinement with clear upgrade paths.

======================================================================
PHASE 2 — CONCEPT MODEL
======================================================================

Before diving into architecture, define the system at the level a human
would explain it on a whiteboard:

1) Building Blocks (3-7 named components)

   For EACH building block, provide the MODULE CONTRACT:
   - Name and one-sentence responsibility
   - OBSERVES: what state does this module read? (its inputs)
   - CONTROLS: what decisions does this module make? (its outputs)
   - OWNS: what mutable state does it exclusively manage?
   - INVARIANTS: what must always hold for this module?
   - EVENTS: what events does it produce or consume?
     Classify each as EXOGENOUS (driven by external input) or
     ENDOGENOUS (driven by internal state transitions).
   - EXTENSION FRICTION: how many files must change to add one more
     variant? (Reference targets from design guidelines Section 4.5)

   No building block may have more than one core responsibility.

2) Interaction Model
   - Who calls whom (directional arrows)
   - Data flow between blocks (what crosses each boundary)
   - Ownership transfer rules (when does data change owners?)

3) System Invariants
   - What must ALWAYS hold (e.g., "clock never decreases")
   - What must NEVER happen (e.g., "no cross-instance state mutation")
   - Causality constraints (ordering guarantees)
   - DES-specific: state vs. statistics separation — which data is
     simulation state (evolves the system) vs. derived statistics
     (output for analysis)?

4) Extension Points
   - Where do new behaviors plug in? (behavioral description of
     interface contract + responsibility)
   - What is the default behavior for each extension point?
   - What is the FIRST non-default implementation planned?

5) State Ownership Map
   - For every piece of mutable state: exactly one owner
   - Shared state must be explicitly identified and justified

6) Real-System Correspondence
   - Map each building block to the real inference system component(s)
     it models. Use a table:

   | Building Block | llm-d | vLLM | SGLang | Other |
   |----------------|-------|------|--------|-------|

   - BLIS models an extensible distributed inference platform, not any
     single system. The table ensures the architecture stays grounded
     in real systems while remaining general enough to express
     behaviors from multiple targets.

THE CONCEPT MODEL MUST FIT IN UNDER 80 LINES.
(Increased from 60 to accommodate module contracts and real-system
correspondence. If it exceeds 80, the design is too complex —
simplify before proceeding.)

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
| Shared-clock event loop | O(N) scan per event is fast for N<=16 |
  Benchmark N=16, 10K events | PR 3 rework | Before PR 3 merge |

MANDATORY VALIDATION RULE:
If cost-of-being-wrong >= 3 PRs of rework, validation is MANDATORY.
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

ABSTRACTION LEVEL CHECK: This section describes the evolution
BEHAVIORALLY. Do NOT include Go code here. Describe what each
module will do and what contract it will satisfy, not how it will
be implemented. Interface signatures appear only in the output
sections (Section G, defined in the Output Format section below)
and only for already-frozen interfaces.

FIDELITY TRADE-OFFS: For each architectural simplification, state:
- What real-system behavior is being approximated
- What analysis questions the approximation still answers correctly
- Under what conditions the approximation breaks down
- What the upgrade path looks like (which future PR refines this)

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
   - INVARIANT TESTS: which system invariants must have companion tests?
     (Golden tests alone are insufficient — see design guidelines 6.3)

2) Documentation Maintenance
   - CLAUDE.md update triggers: new packages, new files, changed file
     organization, completed plan milestones, new CLI flags
   - Who updates CLAUDE.md? (The PR that causes the change.)
   - README update triggers and ownership
   - Design guidelines compliance: does this feature expansion require
     updating the target module map in the design guidelines?

3) CI Pipeline Changes
   - New test packages to add to CI
   - New linter rules or build steps
   - Performance regression benchmarks

4) Dependency Management
   - New external dependencies (justify each one)
   - Version pinning strategy

5) Interface Freeze Schedule
   - Which PR freezes which interface?
   - What must be validated before freezing?
   - After freeze: parallel development of templates/implementations
     can proceed independently

No item may be left as "address when needed."
This applies to cross-cutting infrastructure (test helpers, CI, docs),
not to feature packages which are detailed in Phase 6.

======================================================================
PHASE 6 — ORDERED PR SERIES (PR0 ... PRN)
======================================================================

Design an incremental, independently reviewable and mergeable PR sequence.

For EACH PR, provide TWO TIERS:

--- TIER 1: Human Review Summary (target 15 lines, max 25) ---

- Title
- Building Block Change: Which concept model block is added/modified?
- Extension Type (from design guidelines Section 5.1):
    - policy template: new algorithm behind existing interface
    - subsystem module: new interface + new events
    - backend swap: alternative implementation, requires interface extraction
    - tier composition: decorator/wrapper over existing module
- Motivation: Why does this PR exist? (1-2 sentences)
- Scope: In / Out (bullet points)
- Behavioral Guarantees: What MUST hold after this PR merges?
  (Use named contracts: BC-1, BC-2, etc.)
- Risks: Top 1-2 risks and how they're mitigated
- Cross-Cutting: Which shared infra does this PR create or consume?
- Validation Gate: Does this PR depend on a risk register validation?

--- TIER 2: Implementation Guide (for micro-planning) ---

- Architectural Impact (what changes structurally)
- API Surface Changes (new types, interfaces, methods — described
  BEHAVIORALLY, not as Go code. E.g., "New single-method interface
  for latency estimation, replacing hardcoded Step() logic")
- CLI Changes (new flags, changed behavior)
- Test Categories (unit, integration, regression, golden, invariant)
- Documentation Updates (CLAUDE.md, README, design guidelines if needed)
- Extension Friction: how many files to add one more variant of the
  new type/interface? Compare against reference targets (guidelines 4.5)
- Parallel Development: after this PR merges, what can proceed
  independently? (e.g., "multiple routing policies can be developed
  in parallel after the interface freeze in this PR")
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
- Each PR must identify its extension type and follow the corresponding
  recipe from the design guidelines (Section 5.2-5.5).

======================================================================
PHASE 7 — DEPENDENCY DAG & PARALLELISM
======================================================================

Provide:

- A PR dependency graph (partial order).
- Parallelizable workstreams.
- Merge sequencing guidance.
- Validation gate placement (from risk register).
- Interface freeze points (from Phase 5, item 5: Interface Freeze
  Schedule) — mark which PRs unlock parallel development of multiple
  implementations.
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

Common architectural failure modes and how this plan prevents them:

  General:
  - Scaffolding creep (dead code introduced "for later").
    Prevention: every struct field, method, and flag must be exercised
    by the end of the PR that introduces it.
  - Documentation drift (CLAUDE.md diverges from reality).
    Prevention: the PR that causes the change updates CLAUDE.md in the
    same commit. No deferred documentation.
  - Test infrastructure duplication (helpers copied across packages).
    Prevention: shared test packages created in an early PR, consumed
    by all subsequent PRs (specified in Phase 5, item 1).
  - Golden dataset staleness (regression baselines not updated).
    Prevention: every PR that changes output format includes a golden
    dataset regeneration step with a verification command.
  - Interface over-specification (freezing APIs too early).
    Prevention: interfaces are frozen only after at least one
    non-default implementation is designed (even if not yet built).
    The freeze PR must demonstrate the interface accommodates two
    implementations.

  DES-specific (from design guidelines Section 6):
  - The Type Catalog trap: macro plan includes Go struct definitions
    that diverge from implementation. Prevention: describe modules
    behaviorally, not as Go types.
  - Fidelity for its own sake: modeling components that don't affect
    any analysis question. Prevention: every component must trace to
    a modeling decision in Phase 1.
  - Golden tests without invariant tests: characterization tests that
    encode bugs as expected values. Prevention: every subsystem with
    golden tests must have companion invariant tests.
  - Mixing exogenous and endogenous: tight coupling between workload
    generation and simulation logic that prevents replay experiments.
    Prevention: exogenous inputs must be separable from endogenous
    simulation logic.
  - Interface leaking implementation: interfaces that encode one
    backend's data model instead of the abstract behavioral contract.
    Prevention: interfaces must accommodate at least two backends
    (even if only one is implemented initially).

  Module architecture (from design guidelines Section 6.2):
  - Shotgun surgery: multiple construction sites for the same type.
    Prevention: canonical constructors for types constructed in >1 place.
  - Destructive reads: methods that both query and clear state.
    Prevention: separate Get() and Consume() methods.
  - Monolith methods: single methods containing logic for multiple
    modules. Prevention: each module's logic callable through its
    own interface.
  - Config mixing concerns: single config struct combining unrelated
    parameters. Prevention: group configuration by module.

======================================================================
OUTPUT FORMAT (STRICT)
======================================================================

A) Executive Summary (under 15 lines — synthesize the elevator pitch:
   what is being built, why, how many PRs, key milestones)
B) Repository Recon Summary
C) High-Level Objectives + Non-Goals + Model Scoping Table
D) Concept Model (under 80 lines — building blocks with module
   contracts, interactions, invariants, extension points, real-system
   correspondence)
E) Architectural Risk Register
F) Architectural Evolution (current -> target, mapped to concept model,
   described BEHAVIORALLY — no Go code)
G) Frozen Interface Reference (ONLY for interfaces whose freeze PR
   has already merged — Go signatures with per-PR annotations.
   Include both interfaces frozen BY this plan's PRs and pre-existing
   frozen interfaces that this plan depends on. Omit entirely if no
   interfaces are frozen yet.)
H) Cross-Cutting Infrastructure Plan
I) PR Plan (PR0...PRN, Tier 1 + Tier 2 per PR)
J) Dependency DAG
K) Design Bug Prevention Checklist

CONTEXT BUDGET RULE:
Sections A, C, and D are the human-review core and must be concise.
I-Tier-1 summaries should target 15 lines each (max 25).
All other sections are reference material consulted on demand.
The plan should be structured so a human can review the core sections
(A + C + D + all I-Tier-1 summaries) without needing to read the rest.

ABSTRACTION LEVEL CHECK (final gate):
Before submitting, verify:
- Section F contains ZERO lines of Go code
- Section G contains ONLY frozen interface signatures (merged code)
- Sections A-E and H-K contain ZERO Go code
- All pre-freeze interfaces are described behaviorally
- All module contracts use the template from Phase 2, not Go structs

======================================================================

Quality bar:

- Grounded in real code with file:line citations.
- No hallucinated modules or behaviors.
- No dead code.
- No bloated PRs.
- Must withstand expert review.
- Must be realistic and implementable.
- Concept model must be simple enough to explain verbally in 2 minutes.
- Consistent with design guidelines (docs/contributing/templates/design-guidelines.md).
- Every building block has a module contract.
- Every PR has an extension type.
- Every modeling decision traces to an analysis question.

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
5) After each PR merges, check: does the concept model still accurately
   describe the system? If not, update it. A stale concept model is
   worse than no concept model.

======================================================================

Think deeply before answering.
Inspect before designing.
Validate before committing.
Scope before modeling.
Describe behavior, not implementation.
