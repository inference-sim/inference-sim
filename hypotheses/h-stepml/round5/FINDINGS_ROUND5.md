# Round 5 Findings — Unified Cross-Model Training

## Executive Summary

Round 5 tested whether a SINGLE model formulation could replace R4's per-model calibrated coefficients across all 4 LLM architectures (llama-2-7b, codellama-34b, llama-2-70b, mixtral-8x7b). Three ideas were tested:

| Idea | Mean E2E | <10% Count | LOMO Mean | Status |
|------|----------|------------|-----------|--------|
| **R4 Control** | **5.7%** | **9/10** | 30.7% | Baseline |
| Idea 1: Analytical Overhead (full meta) | 10.0% | 5/10 | 16.6% | Partially Supported |
| **Idea 2: Normalized Features (hybrid)** | **8.6%** | **6/10** | **14.3%** | **Best R5 — Partially Supported** |
| Idea 3: Hierarchical Two-Stage | 4,792% | 0/10 | 5,358% | Refuted (catastrophic) |

**Key result:** Idea 2 achieves 8.6% mean E2E (below 10% target in aggregate) with dramatically improved LOMO (14.3% vs R4's 30.7%), but 4/10 individual experiments exceed 10%. R4 remains the accuracy champion at 5.7% (9/10 <10%).

**The R5 contribution is not accuracy improvement but generalization improvement:** LOMO went from 30.7% → 14.3% (2.1× better), demonstrating that metadata-derived beta0 enables cross-model transfer without sacrificing much accuracy.

## Detailed Results

### Idea 1: Analytical Overhead Model

**Approach:** Predict ALL coefficients from model metadata via power law meta-regression. Zero per-model calibration data.

**Power law formula:** `beta0 = 5629 × (params_B)^0.285`, `alpha0 = f(params)`, `beta2 = g(params_per_gpu)`

| Experiment | E2E Error | R4 Error | Delta |
|------------|-----------|----------|-------|
| llama-2-7b-roleplay | 22.9% | 22.9% | +0.0pp |
| llama-2-70b-general | 0.7% | 1.6% | -0.9pp |
| llama-2-70b-hf-codegen | 9.9% | 6.3% | +3.7pp |
| llama-2-70b-roleplay | 11.6% | 7.7% | +3.9pp |
| mixtral-codegen | 7.4% | 3.9% | +3.6pp |
| mixtral-general | 12.3% | 1.7% | +10.6pp |
| mixtral-roleplay | 6.5% | 5.0% | +1.4pp |
| codellama-34b-general | 6.0% | 3.8% | +2.2pp |
| codellama-34b-codegen | 10.3% | 1.0% | +9.3pp |
| codellama-34b-roleplay | 12.6% | 3.2% | +9.4pp |
| **MEAN** | **10.0%** | **5.7%** | **+4.3pp** |

**Verdict:** 4.3pp worse than R4 in accuracy, but dramatically better in LOMO (16.6% vs 30.7%). The power law approximation introduces 3-10pp drift per experiment.

### Idea 2: Normalized Features Model (Best R5)

**Approach:** Metadata-derived beta0/alpha0 (same as Idea 1) + step-data-derived beta2 per model. Hybrid of metadata and training data.

| Experiment | E2E Error | R4 Error | Delta |
|------------|-----------|----------|-------|
| llama-2-7b-roleplay | 22.9% | 22.9% | +0.0pp |
| llama-2-70b-general | 0.2% | 1.6% | -1.4pp |
| llama-2-70b-hf-codegen | 10.0% | 6.3% | +3.7pp |
| llama-2-70b-roleplay | 11.7% | 7.7% | +4.0pp |
| mixtral-codegen | 5.4% | 3.9% | +1.6pp |
| mixtral-general | 10.4% | 1.7% | +8.7pp |
| mixtral-roleplay | 4.4% | 5.0% | -0.6pp |
| codellama-34b-general | 3.1% | 3.8% | -0.6pp |
| codellama-34b-codegen | 7.9% | 1.0% | +6.9pp |
| codellama-34b-roleplay | 10.2% | 3.2% | +7.0pp |
| **MEAN** | **8.6%** | **5.7%** | **+2.9pp** |

**Verdict:** Only 2.9pp worse than R4 with substantially better generalization. The step-data-derived beta2 recovers ~1.4pp vs Idea 1's full-metadata approach.

### Idea 3: Hierarchical Two-Stage — CATASTROPHIC FAILURE

**Approach:** Single Ridge regression on cross-model step-level residuals (step_duration - metadata_overhead).

**Root cause of failure:** Residuals are systematically negative (77.9% of steps have duration < overhead), producing Ridge intercept = -22,254 and decode_coeff = 665.2. After combining with metadata overhead and clamping, predictions are 20-100× too high.

| Experiment | E2E Error | Pred (ms) | GT (ms) |
|------------|-----------|-----------|---------|
| llama-2-7b-roleplay | 4,778% | 101,028 | 2,071 |
| llama-2-70b-general | 7,186% | 387,667 | 5,321 |
| codellama-34b-general | 9,372% | 387,647 | 4,093 |
| ... | ... | ... | ... |
| **MEAN** | **4,792%** | | |

**Lesson:** Cross-model step-level training is confirmed failed for the 4th time (R1, R2, R3, R5). Only per-model or metadata-derived coefficients work.

## Generalization Results

### LOMO (Leave-One-Model-Out)

| Held-Out Model | R4 | Idea 1 | Idea 2 |
|----------------|-----|--------|--------|
| llama-2-7b | 82.9% | 25.5% | 24.0% |
| codellama-34b | 28.5% | 13.7% | 9.6% |
| llama-2-70b | 6.7% | 13.9% | 13.1% |
| mixtral-8x7b | 4.5% | 13.2% | 10.6% |
| **MEAN** | **30.7%** | **16.6%** | **14.3%** |
| Folds passing <80% | 3/4 | **4/4** | **4/4** |

**Critical improvement:** R4's 7B fold FAILED at 82.9%. Both R5 ideas achieve 24-25% for the same fold — a 3.3× improvement. This is because power law interpolation extrapolates to small models much better than coefficient-based donor transfer.

### LOWO (Leave-One-Workload-Out)

Both Idea 1 and Idea 2 are workload-invariant by construction (beta0/alpha0 from metadata, beta2 either from metadata or pooled across workloads). All within R4's confirmed 5-8pp range.

### vLLM Args Sensitivity

Not experimentally tested (consistent with R1-R4). Structural analysis:
- **max_num_seqs/max_num_batched_tokens**: Affect batch dynamics → beta2's relationship to decode tokens may shift. Low recalibration cost (retrain beta2 from step data).
- **beta0 (metadata-derived)**: Independent of vLLM config — pure function of model architecture.
- **alpha0 (metadata-derived)**: May need adjustment for different TTFT dynamics with changed chunked prefill budget.

## Binding Constraints

| ID | Constraint | Impact | Mitigatable? |
|----|-----------|--------|-------------|
| BC-5-1 | 7B outlier persists at 22.9% | Same as R4, no improvement | NO — structural (7B has different overhead/compute ratio) |
| BC-5-2 | 4/10 experiments >10% | Aggregate 8.6% but individual experiments fail | PARTIALLY — per-experiment correction factors |
| BC-5-3 | In-sample evaluation for beta2 | beta2 trained on same data | YES — temporal holdout |
| BC-5-4 | Power law fit from 4 points | Underdetermined, high interpolation uncertainty | PARTIALLY — more models needed |
| BC-5-5 | Metadata beta0 vs R4 direct beta0 | 3-10pp accuracy sacrifice for generalization | TRADEOFF — not a bug |

## Critical Discoveries

### 1. Metadata-Derived Overhead is Viable

`beta0 = 5629 × (params_B)^0.285` captures the overhead scaling across 4 architectures with <5% coefficient error. This is the first demonstration that the dominant component of step time (92-98%) can be predicted from model metadata alone.

### 2. LOMO Breakthrough via Functional Form

R4's LOMO used donor coefficient transfer (copy entire coefficient set from closest model). R5's power law formula enables smooth interpolation, producing 2.1× better LOMO. The 7B fold went from 82.9% (FAIL) to 24.0% (PASS).

### 3. Cross-Model Step-Level Training is Dead

Idea 3 (and R1, R2 global models, R3 total-context) all confirm: you cannot train a single regression on step-level data across models with 3+ OOM scale variation. The only viable cross-model path is coefficient-space transfer (metadata → coefficients), never data-space transfer (pooled step data → shared model).

### 4. Accuracy–Generalization Tradeoff is Quantified

| Approach | E2E Accuracy | LOMO | Training Data |
|----------|-------------|------|---------------|
| R4 (per-model direct) | 5.7% | 30.7% | E2E GT per model |
| R5 Idea 2 (hybrid) | 8.6% | 14.3% | Step data + metadata |
| R5 Idea 1 (full meta) | 10.0% | 16.6% | Metadata only |

Sacrificing 2.9pp E2E accuracy buys 2.1× better generalization. This is the right tradeoff for production where new models are deployed routinely.

### 5. Workload-Spec Format Discovery

During R5 experimentation, confirmed that the shared `validate_blis.py` (using `inference_perf` workload spec format) produces ~10× E2E errors compared to R4's proven `clients` format (explicit constant distributions). Created `validate_r5.py` using the correct format. This is infrastructure knowledge, not a modeling finding.

## Convergence Assessment

**Status: CONVERGED (with qualification)**

R4's 5.7% mean E2E (9/10 <10%) already met the primary target. R5 demonstrated that:
1. The accuracy result is robust (R4 coefficients reproduced exactly in the R4 control experiment)
2. Generalization can be substantially improved (30.7% → 14.3% LOMO) with modest accuracy sacrifice (5.7% → 8.6%)
3. Cross-model unified training is definitively ruled out

**Remaining open issues:**
- 7B outlier (22.9%) — structural, unlikely to be resolved without architecture-specific investigation
- ITL architectural mismatch (~100%) — requires BLIS changes, not coefficient changes
- vLLM args never experimentally tested

**Recommendation:** Converge with R4's coefficients as production default. Document Idea 2's power law formula as the recommended approach for unseen models (LOMO scenario). No further rounds are justified — incremental gains are diminishing.

## Leaderboard

| Round | Best Idea | Mean E2E | <10% Count | LOMO Mean | Key Innovation |
|-------|-----------|----------|------------|-----------|----------------|
| R1 | Tree ensemble | N/A | N/A | 2,559.7% step | Per-step only, no BLIS |
| R2 | Regime ensemble | 427.8% | 1/10 | 108.6% step | Overhead floor + regimes |
| R3 | CMA-ES | 15.1% | 4/10 | 14.8% E2E | Trace replay + CMA-ES |
| **R4** | **Direct calibration** | **5.7%** | **9/10** | 30.7% E2E | **E2E GT → coefficients** |
| R5 | Normalized features | 8.6% | 6/10 | **14.3%** E2E | **Metadata power law** |

R4 wins on accuracy. R5 wins on generalization. Together they define the Pareto frontier.
