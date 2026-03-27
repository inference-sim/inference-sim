# H17: Multi-Scorer Pareto Frontier

**Status**: Reclassified to Statistical/Dominance
**Date**: 2026-02-22

## Hypothesis

> Multi-scorer weights should produce a Pareto frontier: no single configuration dominates all metrics. Different weight combinations optimize for different objectives -- cache-heavy weights maximize locality (good TTFT), load-balance weights maximize fairness (good tail latency). No single weight combination should be best on ALL metrics simultaneously.

**Refuted if:** A single weight configuration dominates all others on every metric (TTFT, E2E, throughput) within a single workload, across all 3 seeds.
