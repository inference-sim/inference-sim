# Generalization Validation Protocol

**Challenge**: With only 15 experiments, we must validate that the trained latency model generalizes to unseen models, workloads, and configurations without overfitting to the training distribution.

**Solution**: Two-tier validation framework combining cross-validation (uses existing data) and analytical checks (physics constraints).

**Implementation**: Validation tests are run using the pre-implemented `inner_loop_optimize.py` script with different data subsets. The script accepts `--data-dir` to specify which experiments to use for training vs testing.

## Tier 1: Cross-Validation (Ground Truth Required)

**Three clean holdout tests using the 15 training experiments:**

### CV-1: Leave-Yi-and-Mistral-Out (Dense Family Generalization)
**Test**: Dense model family generalization (Llama/Qwen → Yi/Mistral)

**Method**:
- Training: 12 experiments (6 Llama + 2 Qwen + 4 Scout MoE)
  - Llama: Llama-2-7B (4 workloads) + Llama-3.1-70B (2 workloads)
  - Qwen: Qwen2.5-7B (2 workloads)
  - Scout: Llama-4-Scout-17B-16E MoE (4 workloads) — ensures MoE-specific terms (β₅, β₈) are trainable
- Test: 3 experiments (1 Yi + 2 Mistral-Nemo)
  - Yi-34B (1 workload)
  - Mistral-Nemo-12B (2 workloads)

**Pass criteria**: Mean MAPE < 15% (TTFT and E2E) across 3 test experiments
- Standard threshold (same as CV-2/CV-3) — fair test since all coefficients trainable

**Failure diagnosis**: If fails → Model overfitting to Llama/Qwen architectural specifics; not learning transferable dense model principles (compute/memory/TP patterns)

**Rationale for design**:
- **Why include Scout in training?** The evolved model has MoE-specific terms (β₅ for gating, β₈ for routing). If we trained only on dense models, these terms would be **unidentifiable** (MoE basis functions return 0 for dense models → no gradient signal). Including Scout ensures all coefficients are trainable.
- **Why test on Yi+Mistral?** Tests whether the model learned general dense model principles (compute/memory/TP scaling) that transfer to unseen dense architectures, rather than memorizing Llama/Qwen specifics.
- **Why not test on Scout?** Testing dense→MoE generalization with untrainable MoE terms would be a flawed test (guaranteed to fail). CV-1 tests dense→dense generalization with all terms trainable.

### CV-2: Leave-One-Workload-Out (LOWO)
**Test**: Workload-agnostic constraint validation

**Method**:
- Training: codegen (4) + reasoning-lite (3) = 7 experiments
- Test: roleplay (3) + general-lite (5) = 8 experiments

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
  - CV-1 (Yi+Mistral): Mean MAPE < 15% across 3 test experiments
  - CV-2 (workload): Mean MAPE < 15%, roleplay vs general variance < 3%
  - CV-3 (TP=2): MAPE < 15%

**If validation fails**:
- **AC failure** → Agent must add missing physics terms (communication overhead, causality bounds, etc.)
- **CV-1 (Yi+Mistral) fails** → Model overfitting to Llama/Qwen specifics; not learning transferable compute/memory/TP patterns
- **CV-2 (workload) fails** → Basis functions memorizing workload distributions, violates workload-agnostic constraint
- **CV-3 (TP=2) fails** → TP basis function has wrong functional form, doesn't interpolate between TP=1 and TP=4

## Running Cross-Validation Tests

**Using the automated CV test runner:**

All CV tests are implemented in `training/scripts/run_cv_tests.py`, which handles:
- Creating train/test data splits (deterministic, never randomized)
- Training on train set (1000 trials, seed=42)
- Evaluating on test set with trained coefficients
- Computing MAPE metrics and checking pass/fail criteria
- Generating JSON results + markdown reports

### Run All Tests

```bash
cd training

# Run all three CV tests (recommended after iteration converges)
python scripts/run_cv_tests.py --iteration 9 --cv-test all

# Or run individual tests
python scripts/run_cv_tests.py --iteration 9 --cv-test CV-1
python scripts/run_cv_tests.py --iteration 9 --cv-test CV-2
python scripts/run_cv_tests.py --iteration 9 --cv-test CV-3
```

**Runtime**: ~33 minutes per test (1000 trials), ~100 minutes total for all 3 tests.

### Outputs

```
training/cv_results/
├── cv-1_results.json        # CV-1 detailed results
├── cv-1_report.md            # CV-1 human-readable report (with pass/fail)
├── cv-2_results.json
├── cv-2_report.md
├── cv-3_results.json
└── cv-3_report.md
```

### Interpreting Results

**Check the reports:**
```bash
cat cv_results/cv-1_report.md  # Look for "Status: ✅ PASS" or "❌ FAIL"
cat cv_results/cv-2_report.md
cat cv_results/cv-3_report.md
```

**Pass criteria:**
- **CV-1**: Mean MAPE < 15% across Yi + Mistral (3 experiments)
- **CV-2**: Mean MAPE < 15% AND roleplay/general variance < 3%
- **CV-3**: Mean MAPE < 15% across TP=2 experiments

**Example successful result (CV-1):**
```
Status: ✅ PASS
Train loss: 155.0%
Test TTFT MAPE: 14.2% ← Under 15% threshold
Test E2E MAPE: 13.8%  ← Under 15% threshold
```

**Example failure (CV-1):**
```
Status: ❌ FAIL
Train loss: 155.0%
Test TTFT MAPE: 22.5% ← Over 15% threshold
Test E2E MAPE: 19.3%  ← Over 15% threshold

Interpretation: Model overfitting to Llama/Qwen; add regularization or more diverse training data
```

### Advanced Options

```bash
# Use fewer trials for faster testing (less accurate)
python scripts/run_cv_tests.py --iteration 9 --cv-test CV-1 --n-trials 500

# Specify custom data directory
python scripts/run_cv_tests.py --iteration 9 --cv-test all --data-dir trainval_data

# Custom output directory
python scripts/run_cv_tests.py --iteration 9 --cv-test all --output-dir my_cv_results
```

**Note**: The script uses **fixed deterministic data splits** (never randomized) with seed=42 for reproducibility. All results are fully reproducible across runs.

---

## Post-Deployment Monitoring (Future Work)

**After deployment in production**:
1. **Prediction error tracking**: Track actual vs predicted latencies for real workloads, flag outliers (MAPE > 20%)
2. **Incremental retraining**: When new ground truth available (new model, new workload), add to training set and re-run inner loop only (beta recalibration without changing basis functions)
3. **Basis function stability**: If beta coefficients drift significantly (> 50% change), may need outer loop refinement to add new basis function
