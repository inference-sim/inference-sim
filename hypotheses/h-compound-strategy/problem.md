# Strategy Evolution: Compound Strategy (Iteration 4)

**Iteration:** 4 (compound: all three confirmed levers)
**Date:** 2026-03-10
**Branch:** `hypothesis-playground`

---

## Phase 1: Problem Framing

### Goal

Test whether combining all three confirmed mechanisms produces super-additive improvement per S4. The three levers operate at different layers:

1. **StaticClassWeight** (S17): Queue ordering layer — class-aware priority
2. **SLOGatedAdmission** (S19): Traffic management layer — non-zero-sum cluster health
3. **PriorityPreemption** (S21): Batch composition layer — immediate critical scheduling

**Predicted interaction:** Admission reduces total queue depth → fewer sheddable requests in the batch → priority preemption triggers less often BUT each preemption is more impactful (shorter queue for preempted requests to re-enter). The compound could be super-additive or sub-additive depending on which effect dominates.

### Baselines and Configurations

| ID | Priority | Admission | Preemption | Purpose |
|----|----------|-----------|------------|---------|
| B2 | StaticClassWeight | AlwaysAdmit | Off | Iteration 1 baseline |
| T2 | StaticClassWeight | SLOGatedAdmission(100) | Off | Admission only (Iter 2) |
| T3 | StaticClassWeight | AlwaysAdmit | Margin=5.0 | Preemption only (Iter 3) |
| **T4** | **StaticClassWeight** | **SLOGatedAdmission(100)** | **Margin=5.0** | **Full compound** |

The full comparison matrix allows computing:
- T4 vs B2 = total compound effect
- T4 vs T2 = marginal value of adding preemption to admission
- T4 vs T3 = marginal value of adding admission to preemption
- Super-additivity: T4 improvement > (T2 improvement + T3 improvement)

### Hypothesis Bundle (4 arms)

**H-main (Dominance):** T4 reduces critical TTFT P99 by >25% over B2 at 120%, exceeding either T2 or T3 alone.

**H-super-additivity (S4):** `(B2 - T4) > (B2 - T2) + (B2 - T3)` — compound exceeds sum of parts.
*If sub-additive: admission reduces the queue depth that preemption needs to be effective (the mechanisms partially substitute).*

**H-cluster-health:** T4 produces the best cluster-wide TTFT P99 (better than T2, T3, or B2) because admission reduces load AND preemption accelerates critical.

**H-control-negative:** With uniform SLO, T4 produces <5% difference from B2 (no false triggers from either mechanism).
