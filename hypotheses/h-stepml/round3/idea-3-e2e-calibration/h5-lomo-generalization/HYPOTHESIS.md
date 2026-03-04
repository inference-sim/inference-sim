# Sub-Hypothesis H5: Leave-One-Model-Out (LOMO) Generalization for CMA-ES

## Claim

CMA-ES-optimized coefficients from one model group, when applied to another model group's experiments, achieve <50% E2E error — demonstrating partial cross-model transfer of the calibration.

## Rationale

Per-model training is mandatory (R1+R2 confirmed). However, the degree of cross-model transfer matters for practical deployment: if a new LLM is deployed without ground-truth traces for calibration, can the CMA-ES coefficients from the most similar existing model provide a reasonable starting point? This tests whether CMA-ES captures generic simulation dynamics (overhead floor, scheduling overhead) that transfer across models, or whether it overfits to model-specific batch behavior.

## Prior Round Context

- **R1 LOMO per-step:** 2,559.7% (catastrophic — models differ by 3+ OOM in step-time scale)
- **R2 LOMO per-step:** 108.6% (23.6x improvement due to regime structure)
- **R3 H1 per-model E2E:** 7B=22.2%, 70B=8.3%, 34B=7.9%, Mixtral=25.9%

## Method

1. Load all 5 CMA-ES-optimized artifacts from H1
2. For each model group as held-out:
   - Apply each other model's artifact to the held-out model's experiments
   - Run BLIS with trace replay for all held-out model's experiments
   - Record E2E/TTFT/ITL errors
3. Report NxN cross-model transfer matrix (4 models × 4 models)
4. For each held-out model: identify which donor model transfers best
5. Compute mean LOMO E2E (best donor per held-out model)

## Refutation Criteria

- **Supported:** Mean best-donor LOMO E2E < 50% — partial transfer exists
- **Refuted:** All cross-model transfers > 80% — CMA-ES coefficients are fully model-specific with no transferable knowledge

## Diagnostics

- 4×4 cross-model E2E transfer matrix
- Best donor model for each held-out target
- Analysis of which CMA-ES parameters transfer (overhead floor, scheduling time) vs which don't (step-time coefficients)
- Comparison: LOMO E2E vs in-distribution E2E per model
