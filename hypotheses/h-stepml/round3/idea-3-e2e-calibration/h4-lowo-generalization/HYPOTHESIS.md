# Sub-Hypothesis H4: Leave-One-Workload-Out (LOWO) Generalization for CMA-ES

## Claim

CMA-ES-optimized coefficients per model, when evaluated on individual workloads, achieve <20% E2E error on all 3 workloads per model (not just the mean). For models with 3 workloads, a reduced-budget CMA-ES (50 evaluations) optimized on 2 workloads generalizes to the held-out 3rd with <25% E2E.

## Rationale

The Round 3 CMA-ES (H1) optimized per-model across ALL workloads simultaneously. The per-workload breakdown was not reported separately. If the optimizer overfits to one dominant workload (e.g., general) at the expense of others, LOWO error will be much worse than the aggregate. Additionally, a true LOWO test — optimizing on 2 workloads and evaluating on the 3rd — tests whether the CMA-ES calibration captures model-intrinsic properties rather than workload-specific dynamics.

## Prior Round Context

- **R2 LOWO per-step:** 117.4% (regime ensemble)
- **R3 H1 CMA-ES aggregate:** 15.1% E2E across all workloads
- **Mixtral:** 25.9% E2E aggregate — worst model group

## Method

### Part A: Per-Workload Breakdown of Existing CMA-ES Results
1. Load H1 per-experiment results
2. Group by model × workload
3. Report individual workload E2E/TTFT/ITL errors
4. Identify whether any workloads are systematically worse

### Part B: True LOWO (Reduced-Budget CMA-ES)
1. For each model group with 3 workloads (70B, 34B, Mixtral — NOT 7B which has 1):
   - 3 folds: optimize CMA-ES on 2 workloads, evaluate on held-out 3rd
   - Budget: 50 evaluations per fold (reduced from 150 for feasibility)
   - Same parameter vector and bounds as H1
2. Compare LOWO E2E vs in-distribution E2E per fold

## Refutation Criteria

- **Part A supported:** All workloads within 2x of aggregate E2E (e.g., aggregate 15.1% → all workloads <30%)
- **Part B supported:** Mean LOWO E2E < 25% across folds
- **Refuted:** LOWO E2E > 50% for any fold — CMA-ES overfits to workload-specific patterns

## Diagnostics

- Per-model × per-workload E2E/TTFT/ITL error table
- LOWO fold results: train workloads vs held-out workload error
- Convergence speed: does 50 evals suffice for LOWO?
- Parameter stability: how much do coefficients vary across folds?
