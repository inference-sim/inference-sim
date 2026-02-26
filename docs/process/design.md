# Design Process

**Status:** Active (v1.0 — updated 2026-02-26)

This document describes the process for writing a BLIS design document. For the design document template itself, see [docs/templates/design-guidelines.md](../templates/design-guidelines.md).

## When a Design Doc is Needed

- New subsystem modules (new interface + integration)
- Backend swaps (alternative implementations requiring interface extraction)
- Architecture changes affecting module boundaries

**Not needed for:** Bug fixes, new policy templates behind existing interfaces, documentation changes.

## Steps

1. **Identify the extension type** — policy template, subsystem module, backend swap, or tier composition (see [design guidelines](../templates/design-guidelines.md) Section 5)
2. **Choose the design doc species** — decision record, specification, problem analysis, or system overview (Section 3.2)
3. **Complete the DES checklist** (Section 2.6) — model scoping, event design, state/statistics, V&V, randomness
4. **Write the design doc** per the template's required sections (Section 3.3): motivation, scope, modeling decisions, invariants, decisions with trade-offs, extension points, validation strategy
5. **Apply the staleness test** (Section 3.1) — would this content mislead if the implementation changes?
6. **Convergence review** — `/convergence-review design <design-doc-path>` dispatches 8 parallel perspectives and enforces convergence (see [docs/process/convergence.md](convergence.md)). Manual alternative: review against each perspective checklist below.
7. **Human review** — approve before macro/micro planning begins

## Design Review Perspectives (8)

For each perspective, check every item. Classify findings as CRITICAL / IMPORTANT / SUGGESTION per the [convergence protocol](convergence.md).

**Perspective 1 — Motivation & Scoping:**
- Are the analysis questions clear and specific?
- Is the modeling decisions table complete (modeled / simplified / omitted)?
- Does every "simplified" entry state what real-system behavior is lost?
- Has each component been evaluated against the six model scoping criteria (Section 2.1)?

**Perspective 2 — DES Foundations:**
- Is the DES design review checklist (Section 2.6) completed with all 10 questions answered?
- Are new events classified as exogenous or endogenous?
- Do new events specify priority constants for tie-breaking?
- Is state/statistics separation maintained (Section 2.3)?
- Are new randomness sources declared with PartitionedRNG subsystem names?

**Perspective 3 — Module Contract Completeness:**
- Does every new or modified module have all 6 contract aspects (observes, controls, owns, invariants, events, extension friction)?
- Are invariants named (INV-N) and cross-referenced with existing invariants?
- Is the extension friction count specified and within reference targets (Section 4.5)?

**Perspective 4 — Extension Framework Fit:**
- Is the extension type correctly identified (policy template / subsystem module / backend swap / tier composition)?
- Is the correct recipe from Section 5 followed?
- Is the no-op default specified (existing behavior unchanged when extension not configured)?
- Is parallel development path described?

**Perspective 5 — Prohibited Content:**
- Any Go struct definitions with field lists? (Prohibited — Section 3.4)
- Any method implementations? (Prohibited)
- Any file paths with line numbers? (Prohibited)
- Any interface signatures in Go syntax for pre-freeze interfaces? (Prohibited)

**Perspective 6 — Trade-off Quality:**
- Does every non-obvious decision have alternatives listed with rationale?
- For each decision: what breaks if it's wrong?
- Is there a Decision Status column (Proposed / Implemented / Superseded)?

**Perspective 7 — Validation Strategy:**
- How will correctness be verified? (Which invariants?)
- How will fidelity be validated? (Against what real-system data?)
- Are both verification and validation addressed (not just one)?

**Perspective 8 — Staleness Resistance:**
- Apply the staleness test (Section 3.1) to every section
- Would any content mislead if the implementation changes during micro-planning?
- Is content described behaviorally (what crosses a boundary and why) rather than structurally (how the boundary is implemented)?

## Quality Gates

- [ ] Extension type identified and correct recipe followed
- [ ] DES checklist from Section 2.6 completed
- [ ] No prohibited content (Section 3.4): no Go structs, no method implementations, no file:line references
- [ ] Every non-obvious decision has alternatives listed with rationale
- [ ] Validation strategy specified (which invariants? against what real-system data?)

## Prerequisites

| Skill | Purpose | Manual Alternative |
|-------|---------|--------------------|
| `convergence-review` | Dispatch parallel review perspectives (Step 6) | Review against each perspective checklist above |

## References

- Template: [docs/templates/design-guidelines.md](../templates/design-guidelines.md)
- Convergence protocol: [docs/process/convergence.md](convergence.md)
- Standards: [docs/standards/rules.md](../standards/rules.md), [docs/standards/invariants.md](../standards/invariants.md)
