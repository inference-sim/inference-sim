# Macro Plan Process

> **Status:** Draft — to be expanded from experience.

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
7. **Convergence review** — `/convergence-review macro-plan <plan-path>` dispatches 8 parallel perspectives and enforces convergence (see [docs/process/convergence.md](convergence.md)). Manual alternative: review against the quality gates below.
8. **Human review** — approve before micro-planning begins for any PR in the plan

## Quality Gates

- [ ] Every PR in the plan is independently mergeable (no PR requires another PR's uncommitted code)
- [ ] Dependency DAG has no cycles
- [ ] Module contracts are testable with mocks (parallel development enabled)
- [ ] No Go struct definitions or method implementations (those belong in micro plans)
- [ ] Extension friction assessed for each new module boundary

## Prerequisites

| Skill | Purpose | Manual Alternative |
|-------|---------|--------------------|
| `convergence-review` | Dispatch parallel review perspectives (Step 7) | Review against quality gates manually |

## References

- Template: [docs/templates/macro-plan.md](../templates/macro-plan.md)
- Design guidelines: [docs/templates/design-guidelines.md](../templates/design-guidelines.md)
- Standards: [docs/standards/rules.md](../standards/rules.md)
