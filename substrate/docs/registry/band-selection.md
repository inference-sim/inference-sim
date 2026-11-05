# Choosing between the kernel's two step-time band edges

`blis-latency-kernel` reports a band rather than a point estimate. `Overlap` sums the max
over resources within each layer and then sums layers; `NoOverlap` sums every resource.
Which edge a consumer should use is a model-selection question, and this document records
how it was answered.

## Evidence

NVIDIA's FPM dataset (`huggingface.co/datasets/nvidia/aisimulate-fpm-dataset`, Apache-2.0)
measures one synchronized whole-forward iteration at a known batch and KV-token count. It
prices the same quantity the band brackets, with no scheduler in between, which makes it
the right evidence for this choice — and the wrong evidence for fitting a coefficient,
because it constrains a composition rather than a constant.

Reproduce with `scripts/select_overlap_band.py`. The band CSV comes from the kernel
repository, which owns composition; this repository only selects between its columns.

| cell | n | Overlap | NoOverlap | winner |
|---|---|---|---|---|
| MiniMax-M2.7 h200 pure-tp4 | 45 | 9.32% | **8.33%** | NoOverlap |
| MiniMax-M2.7 h200 pure-tp2 | 41 | **6.17%** | 9.89% | Overlap |
| MiniMax-M2.7 h200 tep4 | 45 | 18.35% | **7.88%** | NoOverlap |
| MiniMax-M2.7 h200 tep2 | 43 | 15.19% | **9.60%** | NoOverlap |
| GLM-5.2-NVFP4 b200 tep8 | 45 | 23.71% | **15.05%** | NoOverlap |
| **pooled** | **219** | 14.70% | **10.16%** | **NoOverlap** |

Signed means matter more than absolute ones here, because a one-sided error is a missing
term rather than scatter:

| edge | signed mean |
|---|---|
| Overlap | **−13.45%** |
| NoOverlap | −3.44% |

`NoOverlap` is closer on 158 of 219 points and is nearly unbiased. `Overlap`'s −13.45%
is almost exactly the −13.10% one-sided TPOT deficit BLIS shows end-to-end on the
InferenceX corpus, which is independent corroboration: two different datasets, measured
through different paths, agree on the magnitude of the same defect.

## Physical reason

vLLM serves decode in PIECEWISE cudagraph mode, where attention executes eagerly between
captured segments. Per-layer overlap is therefore structurally limited, so the pessimistic
edge being closer is the expected outcome rather than a coincidence.

## The dissent, and why it is a regime boundary

`pure-tp2` holds the whole model on two GPUs with no expert-parallel sharding, so the
expert weight read is four times the `tp4` case and dominates a single resource. Where one
resource holds most of a step, per-stage max IS the right composition, and `Overlap` wins.
Even there `NoOverlap` is closer at batch ≥ 128.

The principled fix is not a global switch but a concentration-aware choice inside the
kernel, which is the only place that holds `StepEstimate.PerResource`. Five cells is not
enough evidence to propose that form; it is enough to reject the optimistic edge as a
universal default.

## Separation of evidence

- **Fitting** — AISimulate operator tables (Apache-2.0), per-kernel.
- **Model selection** — NVIDIA FPM (Apache-2.0), whole-forward. Used for fitting nothing.
- **Evaluation** — SemiAnalysis InferenceX. Never used for fitting or selection.

`select_overlap_band.py` refuses any path under an `inferencex` or `semianalysis`
directory, so the separation is enforced by the tool rather than by convention.
