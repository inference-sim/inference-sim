# Methodology

Reusable research methodologies developed and validated through BLIS experiments. These approaches are domain-agnostic — while they were refined on LLM inference serving, the techniques apply to any system with configurable policies and a simulation or benchmark harness.

## Methodology Pages

| Page | Description |
|------|-------------|
| [Strategy Evolution](strategy-evolution.md) | Iterative policy discovery through simulation: 5-phase methodology with multi-judge review, Bayesian parameter optimization, and cumulative principle extraction |
| [Discovered Principles](principles.md) | Catalog of 30 principles discovered across strategy evolution experiments, with source iteration and experimental evidence for each. |

## When to Use Strategy Evolution

Strategy Evolution is the right approach when:

- Your system has **multiple interacting policy layers** (routing, scheduling, memory, admission) where interactions produce non-obvious emergent behaviors
- The **optimal configuration cannot be derived analytically** because layer interactions are too complex
- You have a **deterministic simulator or benchmark** that accepts parameterized configuration and produces machine-parseable metrics
- You need **defensible results** — not just "it works" but "here's why it works, here's the evidence, and here are the principles"

## Relationship to Other Processes

Strategy Evolution builds on existing BLIS processes:

- **[Hypothesis Experiments](../contributing/hypothesis.md)** — Strategy Evolution's Phase 3 uses the hypothesis experiment framework for structured measurement
- **[Convergence Protocol](../contributing/convergence.md)** — Multi-judge review in Phase 2 follows the convergence protocol
- **[PR Workflow](../contributing/pr-workflow.md)** — Implementation of winning strategies follows the standard PR workflow
