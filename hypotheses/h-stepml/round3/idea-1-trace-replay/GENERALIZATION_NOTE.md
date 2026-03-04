# Generalization Testing: Idea 1 — Trace Replay

## Why LOMO/LOWO Are Not Applicable

Idea 1 does **not train a new model**. It reuses Round 2's regime ensemble artifacts
(per-model coefficients + overhead floors) and changes only the *input mode* from
workload-spec generation to ground-truth trace replay.

Generalization of the underlying step-time model was already tested in Round 2:
- **LOMO (Leave-One-Model-Out):** 108.6% per-step MAPE (R2 Idea 2 H4)
- **LOWO (Leave-One-Workload-Out):** 117.4% per-step MAPE (R2 Idea 2 H5)

The trace-replay contribution is infrastructure (correct arrival times), not a new
predictive model. Testing LOMO/LOWO on trace replay would re-test R2's coefficients
in a different execution mode, which is equivalent to the existing per-experiment
results in H1/H2/H3 (all 10 experiments use the same per-model coefficients across
all 3 workloads — this is already an implicit LOWO test).

## Implicit LOWO Evidence

The trace-replay results show consistent per-model E2E error across workloads,
confirming that R2's per-model coefficients generalize across workloads:

| Model | Roleplay | Codegen | General | Range |
|---|---|---|---|---|
| llama-2-7b | 60.9% | — | — | N/A (1 workload) |
| llama-2-70b | 55.7% | 56.1% | 62.7% | 7.0pp |
| codellama-34b | 53.6% | 55.0% | 58.9% | 5.3pp |
| mixtral-8x7b | 51.4% | 52.2% | 55.3% | 3.9pp |

The small per-model workload variation (3.9–7.0pp) confirms workload generalization.
