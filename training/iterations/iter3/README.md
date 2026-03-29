# Iteration 3: Design Summary

## Overview

Iteration 3 addresses iter2's regression (136% → 134%) by simplifying the model and adding a targeted improvement for TP=4 experiments.

## Key Changes

### 1. Simplification (Removed β₇ and β₈)

**Rationale**: Both iter2 coefficients remained at initial values throughout 51 optimization trials:
- β₇ (very long context) = 1.0 (initial value) → reasoning TTFT still ~100%
- β₈ (per-request decode) = 30μs (initial value) → β₁ inflation persists

**Impact**: Reduces model from 9 to 8 Beta terms, eliminating parameter bloat (12 params → 11 params for 15 experiments)

### 2. Targeted Improvement (Added NEW β₇: TP-Dependent Prefill Communication)

**Rationale**: TP=4 experiments show asymmetric errors:
- High TTFT (42-90%) but excellent E2E (8-11%)
- Indicates prefill underestimation while decode is accurate
- Current β₃ captures decode TP comm correctly (stable 0.394)
- Missing: Prefill TP comm, which scales with prompt_tokens × TP × num_layers

**Formula**:
```
β₇ × TP × num_layers × (prompt_tokens / 1000.0) microseconds
```

**Expected coefficient**: β₇ ~ 5-15 μs per (TP × layer × K-token)

## Files Delivered

1. **iter3-HYPOTHESIS.md** - Complete hypothesis bundle with H-main (mandatory) + 4 additional hypotheses
2. **iteration_manifest.yaml** - Iteration metadata and reasoning
3. **coefficient_bounds.yaml** - Bounds and warm-start initial values
4. **sim/latency/evolved_model.go** - Implemented backend (8 Beta terms)

## Expected Outcomes

**Primary target**: Overall loss <100% (from 136% in iter2)
- TTFT RMSE <50% (from 72.75%)
- E2E RMSE <50% (from 63.44%)
- TP=4 TTFT improvement: 42-90% → <35%

**Secondary predictions**:
- Reasoning experiments: Stay ~100% TTFT (confirming β₇ was ineffective)
- Scout experiments: Stay 160-200% combined (confirming β₈ was ineffective)
- TP=1 experiments: No change (<2 pp TTFT difference)
- TP=2 experiments: Moderate improvement (5-10 pp TTFT)

## Validation Checklist

- [x] H-main hypothesis exists with quantitative prediction, causal mechanism, diagnostic clause
- [x] Additional hypotheses (H-simplification, H-boundary, H-scout-reasoning, H-coefficient-norm) included
- [x] iteration_manifest.yaml declares backend="evolved" and lists modified files
- [x] coefficient_bounds.yaml has bounds AND initial values for all alpha/beta
- [x] Initial values warm-started from iter2 optimal (α from iter2, β₀-β₆ from iter2, new β₇=0.00001)
- [x] sim/latency/evolved_model.go compiles (`go build -o blis main.go` ✅)
- [x] StepTime() has physics comments for each basis function
- [x] QueueingTime/OutputTokenProcessingTime/PostDecodeFixedOverhead unchanged from iter2
- [x] All features are workload-agnostic (no workload labels)
- [x] Backend name stays "evolved" (not "evolved_iter3")

## Design Philosophy

This iteration follows the key principle from iter2 findings:

> "Adding terms without validation leads to parameter bloat"

Instead of adding more formulas to fix reasoning/Scout failures, iter3:
1. Removes ineffective terms (simplification)
2. Adds one well-grounded term (TP prefill comm) with clear physics justification
3. Leaves reasoning/Scout for investigation (profiling, end-to-end testing) not formula additions

This disciplined approach should restore the 33% improvement trajectory seen in iter0→iter1, rather than continuing the stagnation of iter1→iter2.
