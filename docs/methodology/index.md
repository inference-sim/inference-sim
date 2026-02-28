# Methodology

Reusable research methodologies developed and validated through BLIS experiments. These approaches are domain-agnostic — while they were refined on LLM inference serving, the techniques apply to any system with configurable policies and a simulation or benchmark harness.

## Methodology Pages

| Page | Description |
|------|-------------|
| [Strategy Evolution](strategy-evolution.md) | Iterative policy discovery through simulation: hypothesis-bundle-driven methodology with multi-judge review, convergence-gated verification, Bayesian parameter optimization, and cumulative principle extraction |
| [Hypothesis Bundles in Practice](hypothesis-bundles.md) | Detailed examples of hypothesis bundles from PR #452 (scheduling) and PR #447 (routing), prediction error analysis, bundle sizing, and writing guidelines |

## When to Use Strategy Evolution

Strategy Evolution is the right approach when:

- Your system has **multiple interacting policy layers** (routing, scheduling, memory, admission) where interactions produce non-obvious emergent behaviors
- The **optimal configuration cannot be derived analytically** because layer interactions are too complex
- You have a **deterministic simulator or benchmark** that accepts parameterized configuration and produces machine-parseable metrics
- You need **defensible results** — not just "it works" but "here's why it works, here's the evidence, and here are the principles"

## Relationship to Other Processes

Strategy Evolution integrates existing BLIS processes:

- **[Hypothesis Experiments](../contributing/hypothesis.md)** — Each strategy iteration is formulated as a hypothesis bundle. The hypothesis experiment framework provides the per-arm workflow (experiment design standards, review gates, convergence protocol).
- **[Convergence Protocol](../contributing/convergence.md)** — Three convergence-gated review stages per iteration: Design Review (5 perspectives), Code Review (5 perspectives), FINDINGS Review (10 perspectives).
- **[PR Workflow](../contributing/pr-workflow.md)** — Implementation of winning strategies follows the standard PR workflow.
