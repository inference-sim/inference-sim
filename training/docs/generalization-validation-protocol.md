# Generalization Validation Protocol

**Challenge**: With only 15 experiments, we must validate that the trained latency model generalizes to unseen models, workloads, and configurations without overfitting to the training distribution.

**Solution**: Two-tier validation framework combining cross-validation (uses existing data) and analytical checks (physics constraints).

**Implementation**: Validation tests are run using the pre-implemented `inner_loop_optimize.py` script with different data subsets. The script accepts `--data-dir` to specify which experiments to use for training vs testing.

## Tier 1: Cross-Validation (Ground Truth Required)

**Three clean holdout tests using the 15 training experiments:**

### CV-1: Leave-One-Model-Out (LOMO)
**Test**: Dense→MoE architectural generalization

**Method**:
- Training: 11 dense model experiments (Llama-2-7B, Llama-3.1-70B, Mistral-Nemo-12B, Qwen2.5-7B, Yi-34B)
- Test: 4 MoE experiments (Llama-4-Scout-17B-16E)

**Pass criteria**: MAPE < 20% (TTFT and E2E)
- Lenient threshold since only 1 MoE architecture in training data

**Failure diagnosis**: If fails → Need MoE-specific basis function (expert routing overhead, load imbalance)

### CV-2: Leave-One-Workload-Out (LOWO)
**Test**: Workload-agnostic constraint validation

**Method**:
- Training: codegen (4) + reasoning (3) = 7 experiments
- Test: roleplay (3) + general (5) = 8 experiments

**Pass criteria**:
- Mean MAPE < 15% (TTFT and E2E) across test set
- **Critical**: roleplay MAPE and general MAPE should be similar (variance < 3%)
  - If similar → basis functions depend on batch composition, not workload-specific distributions
  - If divergent → basis functions memorizing workload patterns

**Failure diagnosis**:
- High variance between roleplay and general → Basis functions not truly workload-agnostic
- Both fail uniformly → Missing basis function for batch composition patterns not seen in codegen/reasoning

### CV-3: Leave-One-TP-Out (LOTO)
**Test**: TP communication overhead interpolation

**Method**:
- Training: TP=1 (7) + TP=4 (2) = 9 experiments
- Test: TP=2 (6) experiments

**Pass criteria**: MAPE < 15% (TTFT and E2E)
- Tests if TP communication overhead basis function has correct functional form (interpolates between extremes)

**Failure diagnosis**: If fails → TP basis function has wrong functional form (e.g., linear when should be logarithmic, or missing interaction terms)

## Tier 2: Analytical Consistency (Physics-Based Invariants)

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
1. Run **AC checks (AC-1 through AC-6)** — fast physics validation, catches unphysical predictions immediately
2. Optionally run **one CV check** (rotate between CV-1, CV-2, CV-3) to track generalization trend

**After training convergence**:
1. Run **ALL CV checks (CV-1, CV-2, CV-3)** — comprehensive holdout validation
2. Run **ALL AC checks (AC-1 through AC-6)** — final physics validation

**Acceptance criteria for final model**:
- **MANDATORY**: All AC checks pass (Tier 2) — physics violations are unacceptable
- **PRIMARY**: All CV checks pass with specified thresholds (Tier 1)
  - CV-1 (MoE): MAPE < 20%
  - CV-2 (workload): Mean MAPE < 15%, roleplay vs general variance < 3%
  - CV-3 (TP=2): MAPE < 15%

**If validation fails**:
- **AC failure** → Agent must add missing physics terms (communication overhead, causality bounds, etc.)
- **CV-1 (MoE) fails** → Need MoE-specific basis function (expert routing overhead, load imbalance)
- **CV-2 (workload) fails** → Basis functions memorizing workload distributions, violates workload-agnostic constraint
- **CV-3 (TP=2) fails** → TP basis function has wrong functional form, doesn't interpolate between TP=1 and TP=4

## Running Cross-Validation Tests

**Using the pre-implemented inner loop script:**

### CV-1: Leave-One-Model-Out (Dense→MoE)

```bash
# Step 1: Create training subset (dense models only)
mkdir -p trainval_data_cv1_train
cp -r trainval_data/{llama-2-7b,llama-3.1-70b,mistral-nemo-12b,qwen2.5-7b,yi-34b}* trainval_data_cv1_train/

# Step 2: Create test subset (MoE only)
mkdir -p trainval_data_cv1_test
cp -r trainval_data/llama-4-scout-17b-16e* trainval_data_cv1_test/

# Step 3: Generate manifest and Go code with outer loop agent
# (Agent runs on dense models only)

# Step 4: Train on dense subset
cd training/
python inner_loop_optimize.py --data-dir trainval_data_cv1_train

# Step 5: Evaluate on MoE holdout (frozen basis functions, refit coefficients)
python inner_loop_optimize.py --data-dir trainval_data_cv1_test \
  --manifest iteration_manifest_cv1.yaml  # Uses same basis functions

# Step 6: Check: MAPE < 20% on MoE test set?
```

### CV-2: Leave-One-Workload-Out

```bash
# Training: codegen + reasoning
mkdir -p trainval_data_cv2_train
cp -r trainval_data/*codegen* trainval_data/*reasoning* trainval_data_cv2_train/

# Test: roleplay + general
mkdir -p trainval_data_cv2_test
cp -r trainval_data/*roleplay* trainval_data/*general* trainval_data_cv2_test/

# Train and evaluate
python inner_loop_optimize.py --data-dir trainval_data_cv2_train
python inner_loop_optimize.py --data-dir trainval_data_cv2_test

# Check: Mean MAPE < 15%? Roleplay vs general variance < 3%?
```

### CV-3: Leave-One-TP-Out

```bash
# Training: TP=1 + TP=4
mkdir -p trainval_data_cv3_train
cp -r trainval_data/*tp1* trainval_data/*tp4* trainval_data_cv3_train/

# Test: TP=2
mkdir -p trainval_data_cv3_test
cp -r trainval_data/*tp2* trainval_data_cv3_test/

# Train and evaluate
python inner_loop_optimize.py --data-dir trainval_data_cv3_train
python inner_loop_optimize.py --data-dir trainval_data_cv3_test

# Check: MAPE < 15% on TP=2?
```

**Note**: In all CV tests, **basis functions are frozen** from the main training run. Only the coefficients (α, β) are refit on the holdout training set using `inner_loop_optimize.py`.

---

## Post-Deployment Monitoring (Future Work)

**After deployment in production**:
1. **Prediction error tracking**: Track actual vs predicted latencies for real workloads, flag outliers (MAPE > 20%)
2. **Incremental retraining**: When new ground truth available (new model, new workload), add to training set and re-run inner loop only (beta recalibration without changing basis functions)
3. **Basis function stability**: If beta coefficients drift significantly (> 50% change), may need outer loop refinement to add new basis function
