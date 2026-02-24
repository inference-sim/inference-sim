# Design Process

> **Status:** Draft — to be expanded from experience.

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
6. **Convergence review** — `/convergence-review design <design-doc-path>` dispatches 8 parallel perspectives and enforces convergence (see [hypothesis.md — Universal Convergence Protocol](hypothesis.md#universal-convergence-protocol)). Manual alternative: review against the quality gates below.
7. **Human review** — approve before macro/micro planning begins

## Quality Gates

- [ ] Extension type identified and correct recipe followed
- [ ] DES checklist from Section 2.6 completed
- [ ] No prohibited content (Section 3.4): no Go structs, no method implementations, no file:line references
- [ ] Every non-obvious decision has alternatives listed with rationale
- [ ] Validation strategy specified (which invariants? against what real-system data?)

## Prerequisites

| Skill | Purpose | Manual Alternative |
|-------|---------|--------------------|
| `convergence-review` | Dispatch parallel review perspectives (Step 6) | Review against quality gates manually |

## References

- Template: [docs/templates/design-guidelines.md](../templates/design-guidelines.md)
- Standards: [docs/standards/rules.md](../standards/rules.md), [docs/standards/invariants.md](../standards/invariants.md)
