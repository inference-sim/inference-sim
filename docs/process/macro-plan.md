# Macro Plan Process

**Status:** Active (v1.0 — updated 2026-02-26)

This document describes the process for creating a macro-level implementation plan (multi-PR feature). For the macro plan template, see [docs/templates/macro-plan.md](../templates/macro-plan.md).

## When a Macro Plan is Needed

- Features spanning 2+ PRs
- Work requiring a dependency DAG between PRs
- Features touching multiple module boundaries

**Not needed for:** Single-PR features, bug fixes, documentation changes.

## Steps

1. **Design doc(s) as input** — read the relevant design doc(s) and/or GitHub issues
2. **Decompose into PRs** — each PR should be independently mergeable and testable
3. **Define the dependency DAG** — which PRs can be parallelized? Which must be sequential?
4. **Define module contracts per PR boundary** — what does each PR guarantee to the next?
5. **Identify frozen interfaces** — which interfaces are stable (can be developed against in parallel)?
6. **Identify flexible internals** — which implementation details may change during micro-planning?
7. **Convergence review** — `/convergence-review macro-plan <plan-path>` dispatches 8 parallel perspectives and enforces convergence (see [docs/process/convergence.md](convergence.md)). Manual alternative: review against each perspective checklist below.
8. **Human review** — approve before micro-planning begins for any PR in the plan

## Macro Plan Review Perspectives (8)

For each perspective, check every item. Classify findings as CRITICAL / IMPORTANT / SUGGESTION per the [convergence protocol](convergence.md). Section references below refer to [design-guidelines.md](../templates/design-guidelines.md) and [macro-plan template](../templates/macro-plan.md) unless otherwise noted.

**Perspective 1 — Objective Clarity:**
- Are 3-7 crisp objectives defined?
- Are non-goals explicitly listed?
- Is the model scoping table present (modeled / simplified / omitted / justification)?
- Are analysis questions specific enough to drive component selection?

**Perspective 2 — Concept Model Quality:**
- Is the concept model under 80 lines?
- Does every building block have all 6 module contract aspects (observes, controls, owns, invariants, events, extension friction)?
- Is real-system correspondence documented (llm-d / vLLM / SGLang mapping table)?
- Is the state ownership map complete (exactly one owner per mutable state)?

**Perspective 3 — PR Decomposition:**
- Is every PR independently mergeable and testable?
- Does the dependency DAG have no cycles?
- Can module contracts be tested with mocks (parallel development enabled)?
- Does each PR identify its extension type (policy template / subsystem module / backend swap / tier composition)?

**Perspective 4 — Abstraction Level:**
- Zero Go code in Sections A-F and H-K (only Section G may have frozen interface signatures)?
- Are all pre-freeze interfaces described behaviorally, not as Go code?
- Is every code snippet a FACT about merged code, not an ASPIRATION about unwritten code?
- Are module contracts using the template from Phase 2, not Go structs?

**Perspective 5 — Risk Register:**
- Does every non-obvious architectural decision have a risk entry?
- For decisions with cost-of-being-wrong >= 3 PRs, is validation MANDATORY with a specific gate?
- Does each validation gate have exact success criteria (not "looks good")?
- Are abort plans specified (what changes if validation fails)?

**Perspective 6 — Cross-Cutting Infrastructure:**
- Are test infrastructure, documentation, and CI changes each assigned to a specific PR?
- Is the interface freeze schedule documented (which PR freezes which interface)?
- Is CLAUDE.md update ownership clear (the PR that causes the change updates it)?
- Are no items left as "address when needed"?

**Perspective 7 — Extension Friction:**
- For each new module boundary, is the touch-point count for adding one more variant specified?
- Are touch-point counts within reference targets from design guidelines Section 4.5?
- If friction exceeds targets, is this acknowledged and justified?

**Perspective 8 — Design Bug Prevention:**
- Is scaffolding creep prevented (every struct/method/flag exercised by end of introducing PR)?
- Is documentation drift prevented (CLAUDE.md updated in the same PR that causes the change)?
- Is test infrastructure duplication prevented (shared packages created early)?
- Is golden dataset staleness prevented (regeneration steps included)?
- Are DES-specific anti-patterns addressed (type catalog, fidelity for its own sake, golden without invariant)?

## Quality Gates

- [ ] Every PR in the plan is independently mergeable (no PR requires another PR's uncommitted code)
- [ ] Dependency DAG has no cycles
- [ ] Module contracts are testable with mocks (parallel development enabled)
- [ ] No Go struct definitions or method implementations (those belong in micro plans)
- [ ] Extension friction assessed for each new module boundary

## Prerequisites

| Skill | Purpose | Manual Alternative |
|-------|---------|--------------------|
| `convergence-review` | Dispatch parallel review perspectives (Step 7) | Review against each perspective checklist above |

## References

- Template: [docs/templates/macro-plan.md](../templates/macro-plan.md)
- Design guidelines: [docs/templates/design-guidelines.md](../templates/design-guidelines.md)
- Convergence protocol: [docs/process/convergence.md](convergence.md)
- Standards: [docs/standards/rules.md](../standards/rules.md)
