# H3: Leave-One-Model-Out (LOMO) Cross-Validation

**Status:** REFUTED (mean best-donor E2E 30.7% > target 20%)

## Key Numbers

| Metric | Result | Target | Verdict |
|--------|--------|--------|---------|
| Mean best-donor E2E | 30.7% | <20% | FAIL |
| Mean best-donor ITL | 158.1% | <30% | FAIL |

## Method

For each of 4 normalized models, train the direct calibration (H1 approach)
on the remaining 3 models' data, then evaluate each donor model's coefficients
on the held-out model's experiments. Report the best donor per fold.

## LOMO Transfer Matrix (E2E %)

| Holdout \ Donor | codellama-34b | llama-2-70b | llama-2-7b | mixtral-8x7b |
|-----------------|---------------|-------------|------------|--------------|
| codellama-34b | - | 31.1 | 33.3 | **28.5** |
| llama-2-70b | 20.4 | - | 47.0 | **6.7** |
| llama-2-7b | **82.9** | 137.3 | - | 137.6 |
| mixtral-8x7b | 20.2 | **4.5** | 46.8 | - |

Best donors: mixtral for codellama (28.5%), mixtral for 70b (6.7%),
codellama for 7b (82.9%), 70b for mixtral (4.5%).

## Key Findings

1. **Large models transfer well to each other.** llama-2-70b and
   mixtral-8x7b-v0-1 have similar overhead floors (~18-19ms) and transfer
   bidirectionally with <7% E2E error. codellama-34b transfers reasonably
   to/from large models (20-28%).

2. **Small models are untransferable.** llama-2-7b has fundamentally different
   batch dynamics (avg batch size 12 vs 33-46) and a much smaller overhead
   floor (9.7ms vs 14-19ms). No donor achieves <80% E2E error on 7B.

3. **The overhead floor drives transferability.** Models with similar overhead
   floors (beta0) transfer well; models with different floors do not. This is
   because beta0 dominates the step time prediction (92-98% of total).

4. **ITL transfer is even worse** (158.1% mean) because ITL error is structural
   and model-independent -- it reflects the fundamental BLIS vs lifecycle
   measurement gap.

## Comparison to R3

| Metric | R3 LOMO | H3 (this) |
|--------|---------|-----------|
| Mean best-donor E2E | 14.8% | 30.7% |

R3's CMA-ES-based LOMO performed better (14.8%) because CMA-ES was E2E-optimized
without the overhead floor constraint. The direct calibration approach produces
per-model coefficients that are too model-specific to transfer.

## Implications

The poor LOMO result indicates that the direct calibration approach is inherently
model-specific. The overhead floor (beta0) must be calibrated per model, which
limits zero-shot generalization. For new models without E2E ground truth, the
approach cannot be used directly -- a model size heuristic (e.g., beta0 scaling
with parameter count) would be needed.
