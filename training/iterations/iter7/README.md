# Iteration 7: Clean Data Retraining with Decode Overhead Decoupling

## Executive Summary

**Critical Discovery**: Iter6 post-analysis revealed original reasoning experiments were collected from **overloaded servers** (85% failure rate, 259-second timeouts). Journey trace analysis showed 97-99% of training data consisted of timeout requests—no physics-based model can fit this.

**Fresh Data Available**: Three reasoning-lite experiments collected on 2026-03-30 show roofline baseline improved from 99% → 53% avg TTFT error (range: 15-92%), confirming data quality issue rather than model deficiency.

**Iter7 Strategy**: Retrain on clean dataset (exclude 3 corrupted reasoning, include 3 reasoning-lite, total still 15 experiments) while addressing decode coefficient destabilization from iter6.

## Changes from Iter6

### 1. Training Data Swap
- **Excluded**: 3 corrupted reasoning experiments with 99% TTFT error (20260217-170634-llama-2-7b-tp1-reasoning, 48-llama-4-scout-17b-16e-tp2-reasoning-2, 66-qwen2-5-7b-instruct-tp1-reasoning-1-1)
- **Included**: 3 reasoning-lite experiments with 15-92% TTFT roofline baseline (48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1, 66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1, 67-llama-2-7b-hf-tp1-reasoning-lite-1-1)
- **Total**: Still 15 experiments (12 original + 3 reasoning-lite)

### 2. Model Architecture
- **Added β₇**: Decode per-request overhead (5-15ms expected) in StepTime
  - Physics: Output processing, TP coordination, result aggregation per decode request
  - Purpose: Decouple framework overhead from compute/memory efficiency (β₁/β₄)
  - Expected to stabilize β₁ → 1.00-1.15 and β₄ → 0.75-0.90 (revert to iter3 ranges)

### 3. Coefficient Initialization
- **Alpha reversion**: Warm-start from iter4 (NOT iter6) with tight bounds
  - α₀: 1.5ms (was 4.07ms in iter6)
  - α₁: 125μs (was 351μs in iter6, 6× inflated)
  - α₂: 36μs (was 216μs in iter6, 6× inflated)
  - Bounds: [0.0, 0.0002] for α₁, [0.0, 0.0001] for α₂ (prevent inflation)

- **Beta reversion**: β₁/β₄ from iter3 (NOT iter6/iter5) for decode stabilization
  - β₁: 1.037 (was 1.851 in iter6, 78% destabilized)
  - β₄: 0.796 (was 1.451 in iter6, 82% destabilized)
  - β₄ constrained [0.4, 1.0] (cannot exceed theoretical compute time)
  - Other Beta from iter6 (stable): β₀=0.164, β₂=0.270, β₃=0.000620, β₅=0.00431, β₆=0.0215
  - β₇: 0.010 (10ms initial, NEW)

## Hypothesis Bundle

### H-main: Clean Data Enables Coefficient Stabilization
- **Prediction**: Overall loss 161.69% → **<80%**
  - TTFT RMSE: 69.47% → **<40%** (removing 99% errors)
  - E2E RMSE: 92.22% → **<50%** (stabilizing β₁/β₄)
  - All coefficients physically plausible

- **Mechanism**: Removing 3 corrupted reasoning experiments (597% combined loss) eliminates:
  - Alpha inflation driver (no need to absorb 100-200ms gap via per-token overhead)
  - Decode destabilization driver (no bimodal 50ms vs 259s distribution preventing convergence)
  - 99% outliers that prevented iter3/4/5/6 convergence

- **Diagnostic**: If loss >100%, reasoning-lite data still has issues OR β₇ mandatory

### H-decode-overhead: Decode Needs Per-Request Overhead
- **Prediction**: E2E RMSE 92.22% → **<60%**, β₁ → 1.00-1.15, β₄ → 0.75-0.90, β₇ = 5-15ms

- **Mechanism**: vLLM decode phase has fixed overhead beyond memory/compute:
  - Output processing (sampling, stop condition, streaming): 1-5ms per step
  - TP coordination (synchronization barriers): 1-3ms per step
  - KV cache write-back: Similar to β₃ but during decode

- **Diagnostic**: If β₇ <3ms, decode overhead negligible, different root cause

### H-alpha-reversion: Prevent Inflation via Tight Bounds
- **Prediction**: α₁ 351μs → **<150μs**, α₂ 216μs → **<50μs**, minimal loss impact (<5%)

- **Mechanism**: Iter6 inflation occurred because optimizer used per-token overhead to help reasoning (insufficient, but reduced error). With clean data, inflation unnecessary.

- **Diagnostic**: If α₁ >200μs recurs, reasoning-lite still problematic OR β₆ insufficient

### H-error-pattern: Workload Redistribution
- **Prediction**: Error redistributes from reasoning-dominated (4 at 99%) to architecture-dominated:
  - Reasoning-lite: 99% → **30-60%** (3 experiments)
  - Scout MoE: 87-99% → **50-70%** (4 experiments)
  - Mistral TP=2: 91% → **60-80%** (1 experiment)
  - Short-context: 11-46% → **10-35%** (7 experiments, slight improvement)

- **Diagnostic**: If reasoning-lite >70%, check traces; if Scout >80%, MoE overhead term needed

### H-boundary: Decode Overhead Fixed, Not Scaled
- **Prediction**: β₇ = 5-15ms, decode_time = β₇ + per_token × num_tokens

- **Mechanism**: Decode overhead is fixed per request (output processing, TP coordination), not scaled by output length. At num_tokens=1, overhead dominates; at num_tokens=100, per-token dominates.

- **Diagnostic**: If β₇ >20ms, absorbing scheduler overhead, adjust β₆

## Expected Outcomes

### Success Criteria
- ✅ **Overall loss <80%** (primary goal, 81.69pp improvement from iter6)
- ✅ **TTFT RMSE <40%** (29.47pp improvement)
- ✅ **E2E RMSE <50%** (42.22pp improvement)
- ✅ **All coefficients physically plausible** (Alpha within 10% of iter4, Beta within expected ranges)
- ✅ **No experiment with TTFT >90%** (reasoning-lite replaces 99% outliers)
- ✅ **β₁/β₄ within iter3 ranges** (1.00-1.15 and 0.75-0.90 respectively)

### Per-Experiment Predictions
1. **Reasoning-lite** (3 experiments): 99% → 30-60% TTFT (69-39pp improvement, biggest)
2. **Scout codegen**: 98% → 50-70% TTFT (28-48pp)
3. **Scout roleplay**: 87% → 50-70% TTFT (17-37pp)
4. **Mistral TP=2 general-lite**: 91% → 60-80% TTFT (11-31pp)
5. **Short-context codegen/roleplay** (7 experiments): 11-46% → 10-35% TTFT (slight improvement)

### Failure Modes
- ❌ If loss >100%: reasoning-lite data quality issue OR β₇ mandatory but insufficient
- ❌ If α₁ >200μs or α₂ >100μs: bounds too loose OR reasoning-lite problematic OR β₆ insufficient
- ❌ If β₁ >1.5 or β₄ >1.2: β₇ needed but insufficient OR different decode physics issue
- ❌ If reasoning-lite >70% TTFT: workload-specific overhead still required despite clean data

## Files Generated

1. **iter7-HYPOTHESIS.md**: Full hypothesis bundle with 5 testable predictions (H-main, H-decode-overhead, H-alpha-reversion, H-error-pattern, H-boundary)
2. **iteration_manifest.yaml**: Backend="evolved", 8 beta coefficients (added β₇)
3. **coefficient_bounds.yaml**: Alpha from iter4 (tight bounds), Beta from iter3/iter6, β₇ initial 10ms
4. **sim/latency/evolved_model.go**: Added β₇ decode per-request overhead term in StepTime

## Next Steps (Agent 2 - Orchestration)

1. **Verify training data**: Ensure 3 reasoning-lite experiments available, exclude 3 corrupted reasoning
2. **Run inner loop**: 150-200 trials expected (8-coefficient model, data distribution change)
3. **Monitor convergence**: Watch β₁/β₄ stabilization (should converge toward iter3 ranges: 1.037, 0.796)
4. **Check Alpha bounds**: Ensure α₁ <200μs, α₂ <100μs throughout optimization (tight bounds prevent inflation)

## Context for Agent 3 (Analysis)

When validating hypotheses:
- **H-main baseline**: Iter6 loss 161.69% (TTFT 69.47%, E2E 92.22%) on mixed clean/corrupt data
- **H-main prediction**: <80% overall (removing 597% combined loss from 3 corrupt experiments)
- **H-decode-overhead baseline**: Iter6 β₁=1.851, β₄=1.451 (destabilized 78-82% from iter3)
- **H-decode-overhead prediction**: β₁ → 1.00-1.15, β₄ → 0.75-0.90, β₇ = 5-15ms
- **H-alpha-reversion baseline**: Iter6 α₁=351μs, α₂=216μs (6-10× inflated)
- **H-alpha-reversion prediction**: α₁ <150μs, α₂ <50μs (return to iter4 physical values)
- **Key comparison**: Reasoning-lite (roofline 15-92% TTFT) vs original reasoning (roofline 99% TTFT)

If H-main fails (loss >100%), immediately investigate:
1. Check reasoning-lite traces for hidden quality issues (failure rates, latency distributions)
2. Compare roofline baseline (15-92%) vs evolved performance (should be better, not worse)
3. If evolved performance matches roofline, model capacity may be saturated (need more terms)
4. If H-decode-overhead also fails (β₇ <3ms), different decode physics issue (not framework overhead)
