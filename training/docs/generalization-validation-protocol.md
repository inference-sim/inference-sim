# Generalization Validation Protocol

**Challenge**: With only 15 experiments, we must validate that the trained latency model generalizes to unseen models, workloads, and configurations without overfitting to the training distribution.

**Solution**: Three-tier validation framework combining cross-validation (uses existing data), synthetic probing (physics consistency), and analytical checks (causality bounds).

## Tier 1: Cross-Validation (Ground Truth Required)

**Mandatory checks using the 15 training experiments:**

### CV-1: Leave-One-Model-Out (LOMO)
**Test**: Model architecture generalization
**Method**:
- Train on 5 models (all experiments for those models), test on 6th held-out model (all its experiments)
- Rotate through all 6 models as holdout
**Critical test case**: Hold out Llama-4-Scout-17B-16E (MoE) to test dense→MoE transfer
**Pass criteria**:
- MAPE on held-out model < 15% for TTFT and E2E
- MoE holdout: MAPE < 20% (more lenient since it's the only MoE in training)
**Failure diagnosis**: If MoE fails but dense models pass → need MoE-specific basis function (expert routing overhead)

### CV-2: Leave-One-Workload-Out (LOWO)
**Test**: Workload distribution generalization (validates workload-agnostic constraint)
**Method**:
- Train on 3 workload types (all experiments with those workloads), test on 4th held-out workload (all its experiments)
- Rotate through all 4 workloads as holdout
**Pass criteria**:
- MAPE should be **nearly identical** across all 4 workload holdouts (variance < 3% MAPE)
- If one workload degrades significantly, it indicates basis functions are memorizing workload-specific batch distributions
**Failure diagnosis**:
- High variance → basis functions not truly workload-agnostic, likely depending on batch distribution quirks
- Specific workload fails (e.g., general-lite) → missing basis function for that batch composition pattern

### CV-3: Leave-One-TP-Out (LOTO)
**Test**: Tensor parallelism scaling generalization
**Method**:
- Train on TP∈{1,2}, test on TP=4
- Train on TP∈{1,4}, test on TP=2 (interpolation test)
- Train on TP∈{2,4}, test on TP=1 (tests if model can predict no-communication case)
**Pass criteria**:
- TP=4 extrapolation: MAPE < 18%
- TP=2 interpolation: MAPE < 12%
- TP=1 (no comm): MAPE < 10%
**Failure diagnosis**:
- TP=4 extrapolation fails → TP communication overhead basis function doesn't scale correctly
- TP=1 fails → model depends on TP-specific terms when TP=1 should have zero communication cost

### CV-4: Stratified K-Fold (K=5)
**Test**: Overall generalization across all dimensions simultaneously
**Method**:
- Partition 15 experiments into 5 folds of 3 experiments each
- Stratify by (model size, workload type, TP config) to ensure each fold has diversity
- Train on 4 folds (12 experiments), test on 1 fold (3 experiments)
- Repeat 5 times, aggregate MAPE across all test folds
**Pass criteria**:
- Mean test MAPE < 12% for TTFT and E2E
- Std dev of test MAPE across folds < 4% (consistency check)
**Failure diagnosis**: High fold-to-fold variance → model is sensitive to specific experiment combinations, likely overfitting

### CV-5: vLLM Config Sensitivity
**Test**: Robustness to scheduling parameter variation
**Method**:
- Group experiments by (model, workload, TP), identify experiments that differ only in vLLM config (max_num_seqs, max_num_batched_tokens, max_model_len)
- Hold out high/low config extremes, train on middle configs
**Pass criteria**:
- Config extrapolation: MAPE < 15%
**Failure diagnosis**: Config extrapolation fails → batch formation dynamics not captured, need scheduler-aware basis function

## Tier 2: Synthetic Probing (No Ground Truth)

**Mandatory checks using synthetic batch compositions not seen in training:**

### SP-1: Extreme Batch Composition
**Test**: Boundary behavior for extreme prefill/decode ratios
**Method**: Generate synthetic batches for each training model at TP=1:
- **All-prefill batch**: batch_size=32, all requests prefill=512 tokens, decode=0
- **All-decode batch**: batch_size=128, all requests prefill=0, decode=1 token
- **Heterogeneous contexts**: batch_size=16, context_lengths=[10, 50, 100, 500, 1000, 2000, 4000, 8000] × 2 requests each
**Pass criteria** (no ground truth, physics checks only):
1. **Stability**: No NaN, no negative latencies, no infinite values
2. **Relative ordering**:
   - All-prefill per-token latency > all-decode per-token latency (prefill is O(n²), decode is O(n))
   - Heterogeneous batch latency ≈ max(individual request latencies) since vLLM processes in lockstep
3. **Monotonicity**:
   - Doubling batch_size (with same per-request tokens) should increase total latency by < 2× (amortization)
   - Doubling prefill tokens should increase latency by ≈ 4× (quadratic attention)
**Failure diagnosis**: Violations indicate basis functions have runaway extrapolation or missing saturation terms

### SP-2: Sequence Length Extrapolation
**Test**: Graceful degradation outside training sequence length distribution
**Method**: If training data has seq_len ∈ [128, 2048], test on:
- **Very short**: 10-token prefill, 5-token decode
- **Very long**: 8192-token prefill, 1024-token decode
**Pass criteria**:
1. **Stability**: No NaN, no negative latencies
2. **Scaling consistency**:
   - Short sequences: Latency should not go below minimal framework overhead (α₀ term should dominate)
   - Long sequences: Latency should scale polynomially (not exponentially) with length
3. **Cross-model consistency**: Relative ranking of models by latency should be preserved at extreme lengths
**Failure diagnosis**: Exponential blowup → attention basis function has incorrect exponent; negative values → missing lower bounds

### SP-3: Unseen Model Size Interpolation/Extrapolation
**Test**: Scaling laws for untested model sizes
**Method**: Test on models NOT in training set but available in HuggingFace:
- **Interpolation**: Llama-2-13B (between training 7B and 34B)
- **Extrapolation (small)**: Llama-2-1B (below training range)
- **Extrapolation (large)**: Llama-2-405B (above training range if we have 70B)
**Pass criteria** (no ground truth, compare against roofline/trained_roofline):
1. **Relative agreement**: Evolved model predictions should track roofline within 30% for interpolated sizes
2. **Monotonicity**: Larger models should have higher latency (more FLOPs, more memory bandwidth)
3. **Scaling law consistency**: log(latency) should be approximately linear in log(model_params) for compute-bound workloads
**Failure diagnosis**: Large discrepancies → basis functions depend on specific model sizes seen in training, not parameterized by architecture features

### SP-4: Unseen MoE Configurations
**Test**: MoE architecture transfer (if only 1 MoE in training)
**Method**: Test on Mixtral-8x7B (different expert count, routing strategy)
**Pass criteria**:
1. **Relative to dense**: Mixtral-8x7B should have lower latency than dense 56B model (sparse activation)
2. **Routing overhead**: Should show constant overhead vs dense model of similar active parameter count
**Failure diagnosis**: If MoE predictions are nonsensical → MoE basis function is overfit to Llama-4-Scout specifics

## Tier 3: Analytical Consistency (Physics-Based Invariants)

**Mandatory checks that predictions must satisfy known physical laws:**

### AC-1: Causality Bounds
**Test**: Predictions cannot violate hardware speed-of-light limits
**Method**: For each experiment, compute theoretical lower bound latency:
```
min_latency = max(
  FLOPs_total / GPU_TFLOPS_peak,  # compute-bound lower bound
  Bytes_total / GPU_BW_TB_s        # memory-bound lower bound
)
```
**Pass criteria**:
- **MANDATORY**: `predicted_latency >= 0.8 × min_latency` for all experiments
- If predicted latency is faster than 80% of theoretical peak → model has unphysical predictions
**Failure diagnosis**: Model predicting super-peak performance → missing overhead terms or negative coefficients

### AC-2: TP Scaling Consistency
**Test**: Tensor parallelism must show communication overhead
**Method**: For each (model, workload) pair, compare predictions at TP=1 vs TP=4
**Pass criteria**:
1. **Upper bound**: `latency(TP=4) ≤ latency(TP=1)` (TP should never increase latency by more than communication overhead)
2. **Realistic speedup**: `1.5 < latency(TP=1) / latency(TP=4) < 3.5` (not ideal 4× speedup due to comm overhead)
3. **Communication term presence**: `latency(TP=4) - latency(TP=1)/4 > 0` (must have positive communication overhead)
**Failure diagnosis**:
- TP=4 slower than TP=1 → communication overhead term is too large or has wrong sign
- TP=4 achieves 4× speedup → missing communication overhead basis function

### AC-3: Batch Size Amortization
**Test**: Per-request overhead decreases with batch size (kernel launch amortization)
**Method**: For fixed (model, workload, TP), vary batch_size from 1 to 128
**Pass criteria**:
1. **Sublinear scaling**: `latency(batch=128) / latency(batch=1) < 64` (better than linear)
2. **Monotonicity**: `d(latency)/d(batch_size) > 0` (latency increases with batch size)
3. **Diminishing returns**: `d²(latency)/d(batch_size)² < 0` (concave curve, amortization saturates)
**Failure diagnosis**:
- Linear scaling → missing batch amortization term
- Super-linear scaling → contention effects dominate, may need memory pressure term

### AC-4: Prefill/Decode Asymmetry
**Test**: Prefill (O(n²) attention) must be slower per-token than decode (O(n) attention)
**Method**: For same model, compare:
- Batch A: 1 request, prefill=1000 tokens
- Batch B: 1 request, prefill=10 tokens, decode=990 tokens (same total)
**Pass criteria**:
1. **Prefill dominance**: `latency(Batch A) > 5 × latency(Batch B)` (prefill is quadratic)
2. **Per-token asymmetry**: `latency(Batch A) / 1000 > 2 × latency(Batch B) / 1000`
**Failure diagnosis**: Prefill not significantly slower → attention basis function missing quadratic term

### AC-5: Model Size Monotonicity
**Test**: Larger models must have higher latency (more compute, more memory)
**Method**: For same (workload, TP, batch_size), compare predictions across model sizes
**Pass criteria**:
1. **Monotonicity**: `latency(7B) < latency(13B) < latency(34B) < latency(70B)`
2. **Scaling law**: `log(latency) ≈ a × log(num_params) + b` (approximately linear on log-log plot)
**Failure diagnosis**: Non-monotonic → basis functions not properly parameterized by model architecture

### AC-6: Hardware Portability (Future)
**Test**: Predictions scale correctly with hardware specs (when A100/L40S data available)
**Method**: For same (model, workload, TP, batch), compare predictions on H100 vs A100
**Pass criteria**:
1. **Compute-bound ratio**: For prefill-heavy workloads, `latency_A100 / latency_H100 ≈ TFLOPS_H100 / TFLOPS_A100 = 1989/312 ≈ 6.4×`
2. **Memory-bound ratio**: For decode-heavy workloads, `latency_A100 / latency_H100 ≈ BW_H100 / BW_A100 = 3.35/2.0 ≈ 1.7×`
**Failure diagnosis**: Wrong ratios → basis functions use hardcoded GPU constants instead of hardware_config.json parameters

## Validation Schedule

**During training** (after each outer loop iteration):
1. Run **CV-4 (stratified k-fold)** to get primary generalization metric
2. If CV-4 passes, run **CV-1 (LOMO)** to check model-specific generalization
3. If CV-1 passes, run **CV-2 (LOWO)** to validate workload-agnostic constraint

**After training convergence**:
1. Run ALL Tier 1 cross-validation checks (CV-1 through CV-5)
2. Run ALL Tier 2 synthetic probing checks (SP-1 through SP-4)
3. Run ALL Tier 3 analytical consistency checks (AC-1 through AC-6)

**Acceptance criteria for final model**:
- **MANDATORY**: All AC checks pass (Tier 3) — physics violations are unacceptable
- **PRIMARY**: CV-4 (k-fold) MAPE < 12% (Tier 1)
- **SECONDARY**: All CV checks pass with specified thresholds (Tier 1)
- **TERTIARY**: All SP checks pass stability and monotonicity tests (Tier 2)

**If validation fails**:
- Tier 3 failure → agent must add missing physics terms (e.g., communication overhead, causality bounds)
- Tier 1 failure (specific holdout) → agent must analyze which basis function is memorizing training distribution
- Tier 2 failure → agent must add robustness terms (e.g., saturation for extreme batch sizes)

## Post-Deployment Monitoring (Future Work)

**After deployment in production**:
1. **Prediction error tracking**: Track actual vs predicted latencies for real workloads, flag outliers (MAPE > 20%)
2. **Incremental retraining**: When new ground truth available (new model, new workload), add to training set and re-run inner loop only (beta recalibration without changing basis functions)
3. **Basis function stability**: If beta coefficients drift significantly (> 50% change), may need outer loop refinement to add new basis function
