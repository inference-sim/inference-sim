---
name: Hypothesis Proposal
about: Propose a new hypothesis experiment for BLIS
title: "hypothesis: "
labels: hypothesis
assignees: ''
---

## Hypothesis

> State the hypothesis as an intuitive, behavioral claim. Use the sentence pattern for the chosen family (see `docs/standards/experiments.md`).

## Classification

- **Family:** <!-- Workload/arrival | Scheduler invariants | Performance-regime | Structural model | Robustness/failure-mode | Cross-policy comparative -->
- **VV&UQ:** <!-- Verification | Validation | UQ -->
- **Type:** <!-- Deterministic | Statistical (Dominance/Monotonicity/Equivalence/Pareto) -->

## Diagnostic value

> If this hypothesis fails, what would it indicate? Which code path or design assumption would be worth investigating?

## Proposed experiment design

- **Configurations to compare:** <!-- A vs B, varying exactly one dimension -->
- **Primary metric:** <!-- TTFT p99, throughput, preemption count, etc. -->
- **Workload:** <!-- What arrival process, token distributions, rate? -->
- **Seeds:** <!-- 42, 123, 456 for statistical; single seed for deterministic -->

## Coverage

- **Does this fill a family gap?** <!-- Check hypotheses/README.md coverage table -->
- **Related hypotheses:** <!-- Any existing H1-H26 or filed issues this builds on? -->
