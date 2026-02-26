# Design & Macro Plan Review Perspective Prompts

Reference file for the convergence-review skill. Contains exact prompts for design document review (8 perspectives) and macro plan review (8 perspectives).

**Canonical sources:** `docs/process/design.md` (Design Review Perspectives section) and `docs/process/macro-plan.md` (Macro Plan Review Perspectives section). The perspective names and checklist items below are expanded versions of the process doc checklists, with full prompt context for agent dispatch. If perspective names diverge, the process docs are authoritative.

**Related documents:**
- Design doc process: `docs/process/design.md`
- Design guidelines: `docs/templates/design-guidelines.md`
- Macro plan process: `docs/process/macro-plan.md`
- Macro plan template: `docs/templates/macro-plan.md`

**Dispatch pattern:** Launch each perspective as a parallel Task agent:
```
Task(subagent_type="general-purpose", model=REVIEW_MODEL, run_in_background=True,
     prompt="<prompt from below>\n\n<artifact content>")
```
Model selection is controlled by the `--model` flag in the convergence-review skill (default: `haiku`).

---

## Section A: Design Document Review (8 perspectives) — `design` gate

### DD-1: DES Foundations Compliance

```
You are reviewing a BLIS design document. BLIS is a discrete-event simulator for LLM inference serving systems.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: DES Foundations (design guidelines Section 2)
Check the DES design review checklist (Section 2.6):
- What analysis questions does this design help answer? (Model scoping 2.1)
- What is modeled, simplified, and deliberately omitted? Is there a table?
- Are new events minimal and atomic? Classified as exogenous or endogenous? (2.2)
- Are new events assigned priority constants for (timestamp, priority, seqID) ordering?
- Is state vs. statistics separation maintained? (2.3) Does any module mix state mutation and metric computation?
- Is verification strategy specified (which invariants)? Is validation strategy specified (against what real data)? (2.4)
- If new randomness is introduced, does it flow through PartitionedRNG with a named subsystem? (2.5)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-2: Module Architecture

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Module Architecture
- Does each module have a complete behavioral contract? (observes / controls / owns / invariants / events / extension friction)
- Does the design fit BLIS's two-layer architecture? (domain-agnostic kernel + domain-specific modules)
- Is the extension type identified? (policy template, subsystem module, backend swap, tier composition)
- Is the correct extension recipe followed? (design guidelines Section 5)
- Does the design maintain separation of concerns? (sim/ is a library, cluster/ sees global state, instances see only local data)
- Is extension friction assessed? How many files to add one more variant?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-3: Prohibited Content

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Prohibited Content (design guidelines Section 3.4)
Design docs describe WHAT modules do and WHY, never HOW they're implemented. Check for:
- Go struct definitions (prohibited — belongs in micro plans)
- Method implementations (prohibited — belongs in micro plans)
- file:line references to current code (prohibited — design docs should be durable)
- Factory function code (prohibited)
- Test code (prohibited)
- Interface signatures for unmerged code (prohibited — describe behaviorally instead)

The staleness test (Section 3.1): Would any content in this doc mislead if the implementation changes?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-4: Invariant & Contract Completeness

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Invariant and Contract Completeness
- Are ALL system invariants that this design affects identified? Check against docs/standards/invariants.md (INV-1 through INV-8).
- Are new invariants introduced? Are they precisely stated and verifiable?
- Are behavioral contracts testable? Could someone write a GIVEN/WHEN/THEN test from the contract alone?
- Are failure modes specified? What happens under overload, misconfiguration, degenerate inputs?
- Does each non-obvious decision have alternatives listed with rationale?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-5: Modeling Decisions

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Modeling Decisions and Fidelity Trade-offs
Apply the six scoping criteria (Banks et al., design guidelines Section 2.1):
1. Will including this component significantly affect accuracy for target analysis questions?
2. What accuracy level is actually required?
3. Can data requirements be satisfied (alpha/beta coefficients, hardware specs, traces)?
4. What is the cost of inclusion (complexity, maintenance, config surface)?
5. What breaks if we omit it? (<5% impact → defer)
6. What is the simplest version that answers the same questions?

For each simplification: is the real-system behavior loss documented? Under what conditions would it matter?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-6: vLLM/SGLang Domain Fit

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: vLLM/SGLang Domain Fit
- Does the design match real continuous-batching server behavior?
- Are KV cache semantics consistent with vLLM's implementation?
- Are scheduling and preemption policies realistic?
- Is the real-system correspondence table present? Does each building block map to a real component?
- Are there assumptions about LLM inference serving that this design gets wrong?
- Does the design enable future validation against real vLLM/SGLang data?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-7: Distributed Platform Implications

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Distributed Platform Implications (llm-d, KServe, vLLM multi-node)
- Are multi-instance coordination implications considered?
- Does the design handle routing, load balancing, and instance heterogeneity?
- Are stale snapshot propagation risks identified?
- Is horizontal scaling behavior addressed?
- Are admission control implications at cluster scale considered?
- Does the design enable prefix-affinity routing across instances?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### DD-8: Validation Strategy

```
You are reviewing a BLIS design document.

DESIGN DOCUMENT:
<paste design doc>

YOUR FOCUS: Validation Strategy
- How will correctness be verified? Which specific invariants will be tested?
- How will fidelity be validated? Against what real-system data or analytical baselines?
- Are hypothesis experiments planned to validate key design claims?
- Are there measurable success criteria (not "looks correct" — quantitative thresholds)?
- Is sensitivity analysis planned for key parameters?
- What would falsify the design's assumptions? How would you detect it?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

---

## Section B: Macro Plan Review (8 perspectives) — `macro-plan` gate

### MP-1: PR Decomposition

```
You are reviewing a BLIS macro-level implementation plan (multi-PR feature).

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: PR Decomposition Quality
- Is each PR independently mergeable? (No PR requires another PR's uncommitted code)
- Is each PR exercisable immediately after merge? (Via CLI or tests demonstrating new behavior)
- Is there speculative scaffolding? (Unused interfaces, flags, or types "for later")
- Does each PR deliver one cohesive building block change?
- Are PR scope boundaries clean? (No PR does too much or too little)
- Does each PR identify its extension type? (policy template, subsystem module, backend swap, tier composition)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-2: Dependency DAG

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Dependency DAG Correctness
- Does the dependency graph have any cycles? (Must be a DAG)
- Are parallelizable workstreams identified? Is safe parallelism maximized?
- Is merge sequencing guidance clear?
- Are validation gates placed correctly? (From risk register — before the PR that depends on the assumption)
- Are interface freeze points marked? (Which PRs unlock parallel development?)
- Are there implicit dependencies not shown in the DAG? (e.g., shared test helpers)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-3: Module Contracts

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Module Contract Completeness
For each building block / PR, verify the module contract is complete:
- OBSERVES: What state does the module read? (Inputs)
- CONTROLS: What decisions does the module make? (Outputs)
- OWNS: What mutable state does it exclusively manage?
- INVARIANTS: What must always hold for this module?
- EVENTS: What events does it produce or consume? Classified as exogenous or endogenous?
- EXTENSION FRICTION: How many files to add one more variant?

Are behavioral guarantees (BC-1, BC-2, etc.) testable with mocks? (Enables parallel development)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-4: Abstraction Level

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Abstraction Level Compliance
The macro plan describes WHAT to build and in WHAT ORDER, not HOW. Check:
- Does the plan contain Go struct definitions? (PROHIBITED — belongs in micro plans)
- Does the plan contain method implementations? (PROHIBITED)
- Does the plan contain pre-freeze interface signatures as Go code? (PROHIBITED — describe behaviorally)
- Are frozen interfaces (already merged code) correctly distinguished from aspirational ones?

THE TEST: Is content a FACT about merged code, or an ASPIRATION about code to be written?
- Facts (frozen interfaces, merged code references): Go code allowed
- Aspirations (planned interfaces, future modules): Must be behavioral descriptions only

Check the concept model (<80 lines?). Does it use behavioral descriptions or implementation details?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-5: Risk Register

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Risk Register Completeness
For each entry in the architectural risk register, verify:
- DECISION: Is the choice clearly stated?
- ASSUMPTION: What must be true for this to work?
- VALIDATION: How to test cheaply? (Mock study, prototype, analysis, spike)
- COST IF WRONG: How many PRs of rework? (Count affected PRs)
- GATE: When must validation complete? (Before which PR)

MANDATORY VALIDATION RULE: If cost-of-being-wrong >= 3 PRs, validation is MANDATORY.
- Is there a spike/validation PR or pre-PR validation step?
- Are success criteria measurable (not "looks good")?
- Is there an abort plan (what changes if validation fails)?

Are there unidentified risks? (Missing entries for non-obvious architectural decisions)

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-6: DES Expert

```
You are reviewing a BLIS macro-level implementation plan as a DES expert.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: DES Design Implications
- Is model scoping justified? (Banks et al. criteria from design guidelines Section 2.1)
- Are new events correctly classified (exogenous vs endogenous)?
- Is state vs. statistics separation maintained across the PR series?
- Are there event-ordering implications for the proposed architectural changes?
- Does the concept model accurately describe the DES components and their interactions?
- Are there DES-specific failure modes in the design bug prevention checklist?
  - Type catalog trap (Go structs that diverge from implementation)
  - Fidelity for its own sake (components that don't affect analysis questions)
  - Golden tests without invariant tests
  - Mixing exogenous and endogenous inputs

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-7: vLLM/SGLang Expert

```
You are reviewing a BLIS macro-level implementation plan as a vLLM/SGLang expert.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Real-System Accuracy
- Is the real-system correspondence table present and accurate?
- Does each building block map correctly to real inference system components?
  (Check against: llm-d, vLLM, SGLang, and other systems listed)
- Are batching, KV cache, scheduling, and routing semantics accurate?
- Are there assumptions about LLM serving that the plan gets wrong?
- For each "Simplified" modeling decision: is the lost real-system behavior documented?
- Would the planned features enable meaningful comparison with real vLLM deployments?

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```

### MP-8: Cross-Cutting Completeness

```
You are reviewing a BLIS macro-level implementation plan.

MACRO PLAN:
<paste macro plan>

YOUR FOCUS: Cross-Cutting Infrastructure
Verify Phase 5 (cross-cutting) is complete:
1. Shared Test Infrastructure — which PR creates it? Which consume it? Are invariant tests planned?
2. Documentation Maintenance — CLAUDE.md update triggers? README updates? Design guidelines updates?
3. CI Pipeline — new test packages added? New linter rules? Performance benchmarks?
4. Dependency Management — new external deps justified? Version pinning?
5. Interface Freeze Schedule — which PR freezes which interface? What must be validated before freezing?

Check that NO item is left as "address when needed." Every cross-cutting concern must be assigned to a specific PR.

Also check the design bug prevention checklist (Phase 8):
- Scaffolding creep prevention
- Documentation drift prevention
- Test infrastructure duplication prevention
- Golden dataset staleness prevention
- Interface over-specification prevention

Rate each finding as CRITICAL, IMPORTANT, or SUGGESTION.
Report: (1) numbered list of findings with severity, (2) total CRITICAL count, (3) total IMPORTANT count.
```
