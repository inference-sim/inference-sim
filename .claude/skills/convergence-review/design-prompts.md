# Design & Macro Plan Review Perspective Prompts

Reference file for the convergence-review skill. Contains exact prompts for design document review (8 perspectives) and macro plan review (8 perspectives).

**Canonical sources:** `docs/contributing/design-process.md` (Design Review Perspectives section) and `docs/contributing/macro-planning.md` (Macro Plan Review Perspectives section). The perspective names and checklist items below are expanded versions of the process doc checklists, with full prompt context for agent dispatch. If perspective names diverge, the process docs are authoritative.

**Related documents:**
- Design doc process: `docs/contributing/design-process.md`
- Design guidelines: `docs/contributing/templates/design-guidelines.md`
- Macro plan process: `docs/contributing/macro-planning.md`
- Macro plan template: `docs/contributing/templates/macro-plan-prompt.md`

**Dispatch pattern:** Launch each perspective as a parallel Task agent:
```
Task(subagent_type="general-purpose", model=REVIEW_MODEL, run_in_background=True,
     prompt="<prompt from below>\n\n<artifact content>")
```
Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).

---

## Section A: Design Document Review (8 perspectives) — `design` gate

### DD-1: Motivation & Scoping

```
You are reviewing a BLIS design document. BLIS is a discrete-event simulator for LLM inference serving systems.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Motivation & Scoping (design guidelines Section 2.1)
- Are the analysis questions clear and specific?
- Is the modeling decisions table complete (modeled / simplified / omitted)?
- Does every "simplified" entry state what real-system behavior is lost?
- Has each component been evaluated against the six model scoping criteria (Banks et al., design guidelines Section 2.1)?
- What is the simplest version that answers the same questions?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-2: DES Foundations

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: DES Foundations (design guidelines Section 2)
- Is the DES design review checklist (Section 2.6) completed with all 10 questions answered?
- Are new events classified as exogenous or endogenous? (Section 2.2)
- Do new events specify priority constants for (timestamp, priority, seqID) ordering?
- Is state vs. statistics separation maintained? (Section 2.3) Does any module mix state mutation and metric computation?
- Are new randomness sources declared with PartitionedRNG subsystem names? (Section 2.5)
- Is verification strategy specified (which invariants)? Is validation strategy specified (against what real data)? (Section 2.4)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-3: Module Contract Completeness

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Module Contract Completeness (design guidelines Section 4.3)
- Does every new or modified module have all 6 contract aspects?
  (observes / controls / owns / invariants / events / extension friction)
- Are invariants named (INV-N) and cross-referenced with existing invariants (INV-1 through INV-8)?
- Is the extension friction count specified and within reference targets (Section 4.5)?
- Are behavioral contracts testable? Could someone write a GIVEN/WHEN/THEN test from the contract alone?
- Are failure modes specified? What happens under overload, misconfiguration, degenerate inputs?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-4: Extension Framework Fit

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Extension Framework Fit (design guidelines Section 5)
- Is the extension type correctly identified? (policy template, subsystem module, backend swap, tier composition)
- Is the correct extension recipe followed? (Section 5.2-5.5)
- Is the no-op default specified (existing behavior unchanged when extension not configured)?
- Is parallel development path described?
- Does the design maintain separation of concerns? (sim/ is a library, cluster/ sees global state, instances see only local data)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-5: Prohibited Content

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Prohibited Content (design guidelines Section 3.4)
Design docs describe WHAT modules do and WHY, never HOW they're implemented. Check for:
- Go struct definitions with field lists (prohibited — belongs in micro plans)
- Method implementations (prohibited — belongs in micro plans)
- File paths with line numbers (prohibited — design docs should be durable)
- Interface signatures in Go syntax for pre-freeze interfaces (prohibited — describe behaviorally instead)

Apply the staleness test (Section 3.1): Would any content in this doc mislead if the implementation changes?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-6: Trade-off Quality

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Trade-off Quality
- Does every non-obvious decision have alternatives listed with rationale?
- For each decision: what breaks if it's wrong?
- Is there a Decision Status column (Proposed / Implemented / Superseded)?
- Are real-system correspondences documented? Does each building block map to real components (llm-d, vLLM, SGLang)?
- Are there assumptions about LLM inference serving that this design gets wrong?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-7: Validation Strategy

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Validation Strategy
- How will correctness be verified? Which specific invariants will be tested?
- How will fidelity be validated? Against what real-system data or analytical baselines?
- Are both verification and validation addressed (not just one)?
- Are hypothesis experiments planned to validate key design claims?
- Are there measurable success criteria (not "looks correct" — quantitative thresholds)?
- What would falsify the design's assumptions? How would you detect it?
- Does the design enable future validation against real vLLM/SGLang deployments?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-8: Staleness Resistance

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Staleness Resistance (design guidelines Section 3.1)
- Apply the staleness test to every section: Would this content mislead if the implementation changes during micro-planning?
- Is content described behaviorally (what crosses a boundary and why) rather than structurally (how the boundary is implemented)?
- Are multi-instance coordination implications considered? (Routing, load balancing, stale snapshot propagation)
- Does the design handle horizontal scaling and instance heterogeneity?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section B: Macro Plan Review (8 perspectives) — `macro-plan` gate

### MP-1: Objective Clarity

```
You are reviewing a BLIS macro-level implementation plan (multi-PR feature).

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Objective Clarity (macro-plan template Phase 1)
- Are 3-7 crisp objectives defined?
- Are non-goals explicitly listed?
- Is the model scoping table present (modeled / simplified / omitted / justification)?
- Are analysis questions specific enough to drive component selection?
- For each "Simplified" entry, is the lost real-system behavior documented?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-2: Concept Model Quality

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Concept Model Quality (macro-plan template Phase 2)
- Is the concept model under 80 lines?
- Does every building block have all 6 module contract aspects? (observes, controls, owns, invariants, events, extension friction)
- Is real-system correspondence documented (llm-d / vLLM / SGLang mapping table)?
- Is the state ownership map complete (exactly one owner per mutable state)?
- Are behavioral contracts testable with mocks? (Enables parallel development)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-3: PR Decomposition

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: PR Decomposition Quality (macro-plan template Phase 6)
- Is each PR independently mergeable? (No PR requires another PR's uncommitted code)
- Does the dependency DAG have no cycles?
- Can module contracts be tested with mocks (parallel development enabled)?
- Does each PR identify its extension type? (policy template, subsystem module, backend swap, tier composition)
- Is each PR exercisable immediately after merge? (Via CLI or tests demonstrating new behavior)
- Are parallelizable workstreams identified? Is safe parallelism maximized?
- Are interface freeze points marked? (Which PRs unlock parallel development?)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-4: Abstraction Level

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Abstraction Level Compliance (macro-plan template Abstraction Level Rule)
The macro plan describes WHAT to build and in WHAT ORDER, not HOW. Check:
- Zero Go code in Sections A-F and H-K (only Section G may have frozen interface signatures)?
- Are all pre-freeze interfaces described behaviorally, not as Go code?
- Is every code snippet a FACT about merged code, not an ASPIRATION about unwritten code?
- Are module contracts using the template from Phase 2, not Go structs?
- Check the concept model (<80 lines?). Does it use behavioral descriptions or implementation details?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-5: Risk Register

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Risk Register Completeness (macro-plan template Phase 3)
For each entry, verify:
- DECISION: Is the choice clearly stated?
- ASSUMPTION: What must be true for this to work?
- VALIDATION: How to test cheaply? (Mock study, prototype, analysis, spike)
- COST IF WRONG: How many PRs of rework? (Count affected PRs)
- GATE: When must validation complete? (Before which PR)

MANDATORY VALIDATION RULE: If cost-of-being-wrong >= 3 PRs, validation is MANDATORY.
- Are success criteria measurable (not "looks good")?
- Are abort plans specified (what changes if validation fails)?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-6: Cross-Cutting Infrastructure

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Cross-Cutting Infrastructure (macro-plan template Phase 5)
Verify Phase 5 is complete:
1. Shared Test Infrastructure — which PR creates it? Which consume it? Are invariant tests planned?
2. Documentation Maintenance — CLAUDE.md update triggers? README updates? Design guidelines updates?
3. CI Pipeline — new test packages added? New linter rules? Performance benchmarks?
4. Dependency Management — new external deps justified? Version pinning?
5. Interface Freeze Schedule — which PR freezes which interface? What must be validated before freezing?

Check that NO item is left as "address when needed." Every cross-cutting concern must be assigned to a specific PR.

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-7: Extension Friction

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Extension Friction (design guidelines Section 4.5)
- For each new module boundary, is the touch-point count for adding one more variant specified?
- Are touch-point counts within reference targets from design guidelines Section 4.5?
- If friction exceeds targets, is this acknowledged and justified?
- Does each building block map correctly to real inference system components? (llm-d, vLLM, SGLang)
- Are batching, KV cache, scheduling, and routing semantics accurate?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-8: Design Bug Prevention

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Design Bug Prevention (macro-plan template Phase 8)
Check that the plan prevents these failure modes:

General:
- Scaffolding creep prevented (every struct/method/flag exercised by end of introducing PR)?
- Documentation drift prevented (CLAUDE.md updated in the same PR that causes the change)?
- Test infrastructure duplication prevented (shared packages created early)?
- Golden dataset staleness prevented (regeneration steps included)?
- Interface over-specification prevented (frozen only after 2+ implementations designed)?

DES-specific:
- Type catalog trap prevented (behavioral descriptions, not Go structs)?
- Fidelity for its own sake prevented (every component traces to an analysis question)?
- Golden without invariant prevented (companion invariant tests for golden tests)?
- Exogenous/endogenous mixing prevented (separable workload inputs)?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```
