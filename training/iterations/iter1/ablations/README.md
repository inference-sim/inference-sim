# Iteration 1: Ablation Experiments

This directory contains ablation experiments to validate hypothesis predictions about basis function importance.

## Overview

**Ablation experiments** test whether a basis function is **critical**, **moderate**, or **redundant** by:
1. Removing the term (forcing its coefficient to 0)
2. Re-optimizing the model without that term
3. Measuring performance degradation compared to the full model

## Scripts

### `run_ablation.py`
Runs a single ablation experiment by temporarily swapping bounds files.

**Usage**:
```bash
cd ablations/
python run_ablation.py --iteration 1 --ablation chunking --n-trials 50
```

**Arguments**:
- `--iteration N`: Iteration number
- `--ablation NAME`: Which term to ablate (`chunking`, `tp_comm`, `kv_mgmt`)
- `--n-trials N`: Number of optimization trials (default: 50)

**Process**:
1. Backs up original `coefficient_bounds.yaml`
2. Installs ablation-specific bounds (forces target coefficient to 0)
3. Runs `inner_loop_optimize.py` from training root
4. Saves results to `ablation_no_{NAME}_results.json`
5. Restores original bounds

---

### `compare_ablation.py`
Compares ablation results to full model baseline.

**Usage**:
```bash
cd ablations/
python compare_ablation.py \
  --baseline ../inner_loop_results.json \
  --ablation ablation_no_chunking_results.json \
  --output ablation_no_chunking_comparison.json
```

**Output**: JSON with delta metrics, verdict (CRITICAL/MODERATE/REDUNDANT), and recommendation (KEEP/REMOVE).

---

### `check_ablations_status.py`
Monitors ablation progress and generates summary when all complete.

**Usage**:
```bash
cd ablations/
python check_ablations_status.py
```

**Process**:
1. Checks if all 3 ablation results exist
2. If all complete:
   - Runs `compare_ablation.py` for each
   - Generates `ABLATION-SUMMARY.md`
   - Reports next steps for updating HYPOTHESIS-validation and FINDINGS

---

## Files

### Bounds Files (Input)
- `ablation_no_chunking_bounds.yaml` - Forces β₅ to [0.0, 0.0]
- `ablation_no_tp_comm_bounds.yaml` - Forces β₃ to [0.0, 0.0]
- `ablation_no_kv_mgmt_bounds.yaml` - Forces β₄ to [0.0, 0.0]

### Results Files (Output)
- `ablation_no_chunking_results.json` - Optimization results without β₅
- `ablation_no_tp_comm_results.json` - Optimization results without β₃
- `ablation_no_kv_mgmt_results.json` - Optimization results without β₄

### Comparison Files (Output)
- `ablation_no_chunking_comparison.json` - Delta metrics for β₅ ablation
- `ablation_no_tp_comm_comparison.json` - Delta metrics for β₃ ablation
- `ablation_no_kv_mgmt_comparison.json` - Delta metrics for β₄ ablation

### Summary (Output)
- `ABLATION-SUMMARY.md` - Comprehensive analysis of all ablations with verdicts and recommendations

---

## Hypotheses Being Tested

### H-ablation-chunking: Prefill Chunking Term Importance
**Prediction**: Removing β₅ (chunking) will increase TTFT RMSE by >15%

### H-ablation-tp-comm: TP Communication Term Importance
**Prediction**: Removing β₃ (TP communication) will increase overall loss by >10% for TP>1

### H-ablation-kv-mgmt: KV Management Term Importance
**Prediction**: Removing β₄ (KV management) will increase E2E RMSE by >10%

---

## Interpretation

### Verdicts
- **CRITICAL**: Δ Overall > 10% OR Δ TTFT RMSE > 15% OR Δ E2E RMSE > 10%
  - Term captures essential overhead that other terms cannot compensate for
  - **Recommendation**: KEEP for iter2

- **MODERATE**: Δ Overall 5-10% OR Δ TTFT RMSE 5-15% OR Δ E2E RMSE 5-10%
  - Term captures real overhead but partial compensation occurs
  - **Recommendation**: KEEP for iter2 (may need feature extraction investigation)

- **REDUNDANT**: Δ Overall < 5% AND Δ TTFT RMSE < 5% AND Δ E2E RMSE < 5%
  - Term overhead is negligible or fully absorbed by other terms (e.g., α₀)
  - **Recommendation**: REMOVE for iter2

---

## Status (Current Run)

**Started**: 2026-03-28 (3 ablations running in parallel)

**Progress**:
- Chunking ablation: Running (task ID: bee6a03)
- TP comm ablation: Running (task ID: b223e6a)
- KV mgmt ablation: Running (task ID: bc39660)

**ETA**: ~50-60 minutes per ablation (50 trials × ~65 sec/trial)

To check status:
```bash
cd ablations/
python check_ablations_status.py
```

---

## Next Steps After Completion

1. **Review**: Read `ABLATION-SUMMARY.md` for verdicts and evidence
2. **Update Hypothesis Validation**: Replace "⚠️ INCONCLUSIVE" verdicts in `../iter1-HYPOTHESIS-validation.md` with actual ablation results
3. **Update Findings**: Incorporate ablation recommendations into `../iter1-FINDINGS.md` basis function changes section
4. **Plan iter2**: Use ablation verdicts to decide which terms to keep/remove for next iteration
