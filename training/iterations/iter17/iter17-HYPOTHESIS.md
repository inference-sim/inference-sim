# Iteration 17: Scout MoE Fix + Full 15-Experiment Training

## Context

Iteration 16 achieved a 38.5× improvement in loss (2319% → 60.19%) by adopting the
trained-roofline architecture. Post-mortem analysis revealed **two compounding defects**
that prevented iter16 from reaching its true potential:

**Defect 1 — Scout MoE fix absent during optimization**: The #877 interleaved
architecture fix was merged to the training branch 24 minutes *after* iter16 was
committed — iter16 results at `0b41c913`/`faf97f11` (15:24–15:32), Scout MoE fix at
`fcad4689` (15:56) on 2026-04-01. Optimization itself completed earlier (~13:42 per
`inner_loop_results.json` timestamp), so the fix was absent for the entire run. Iter16's optimizer searched with wrong Scout FLOPs for 4/15
experiments, forcing β₁ to collapse to 0.20 and α₂ to inflate to 45.7µs as
compensating artifacts.

**Defect 2 — Only 9/15 experiments used during optimization**: Six experiments using
gated Meta Llama models (`meta-llama/Llama-2-7b-hf` × 4, `meta-llama/Llama-3.1-70B-Instruct` × 2)
silently failed during iter16 optimization because `HF_TOKEN` was unset and
`model_configs/` lacked their `config.json` files. The optimizer fit coefficients to
9 experiments (Scout × 4, Mistral × 2, Qwen × 2, Yi × 1) while 6 Llama experiments
were silently excluded — a 40% reduction in training coverage.

Iter17 fixes both defects:
1. #877 is present in the compiled binary during every trial evaluation
2. NousResearch public mirror configs pre-cached before optimization starts
   (see `training/trainval_data/README.md` → "Option 2: Use NousResearch public mirrors")

No architectural changes to `evolved_model.go` — only re-optimization with correct
data.

---

## H-main: Re-Optimization With Correct Scout FLOPs Strictly Reduces Loss Below 60.19%

**Prediction**: Overall loss will be **strictly less than iter16's 60.19%**, with:
- Scout mean TTFT APE < 39% (below iter16's best Scout experiment at 39.6%)
- β₁ (prefill correction) recovering from 0.20 toward the physically plausible range 0.4–0.9
- α₂ (output token cost) retreating from 45.7µs toward 1–10µs range

**Causal Mechanism**: In iter16, the optimizer fit 4 Scout experiments against inflated
prefill FLOPs (all 48 layers treated as full MoE experts). With #877, Scout's 24 MoE
layers use `MoEExpertFFNDim × kEff` and 24 dense layers use `DenseIntermediateDim`
— significantly lower total FLOPs per prefill step. The prefill basis function values
for Scout decrease, allowing the optimizer to find a β₁ that is physically meaningful
(~0.5–0.9) rather than a compensating discount (~0.20). Since β₁ governs all 15
prefill predictions (not just Scout), correcting it improves the entire dataset fit.

**Diagnostic Clause**: If loss remains ≥ 60.19%, investigate:
- Whether the compiled `blis` binary actually reflects #877 (check `git log --oneline`
  for `fcad4689` in the binary's source)
- Whether β₁ is still ≤ 0.25 (indicates optimizer still sees inflated Scout prefill FLOPs)
- Whether the 4 Scout experiments still dominate residual error

---

## H-scout: All Four Scout Experiments Improve Over iter16

**Prediction**: All four Scout experiments will have lower TTFT APE than their iter16
values:
- Scout reasoning-lite: < 72.5% TTFT APE (iter16: 72.5%)
- Scout codegen: < 47.1% TTFT APE (iter16: 47.1%)
- Scout general-lite: < 41.1% TTFT APE (iter16: 41.1%)
- Scout roleplay: < 39.6% TTFT APE (iter16: 39.6%)

**Causal Mechanism**: Iter16's coefficients were optimized against wrong Scout FLOPs;
the optimizer minimized a distorted loss surface. With correct FLOPs, the same 7-term
formula can fit Scout's actual prefill/decode times without the compensating distortions
(collapsed β₁, inflated α₂). The fix does not change the formula expressiveness — only
the numerical values the formula operates on.

**Diagnostic Clause**: If Scout experiments do *not* improve uniformly, it indicates the
remaining Scout error is not FLOPs-related (e.g., expert routing efficiency at inference
time differs from the kEff model, or FP8 MFU on H100 differs from modeled value). In
that case, iter18 should investigate MoE-specific correction terms.

---

## H-dense-stable: Dense Model Predictions Are Unaffected or Marginally Better

**Prediction**: Dense model experiments (11/15) will have TTFT and E2E APE within ±5
percentage points of their iter16 values. The 10 best-predicted experiments (all dense)
should remain approximately stable.

**Causal Mechanism**: #877 only modifies FLOPs calculations for models with
`InterleaveMoELayerStep > 0`. Dense models (Llama-2, Llama-3.1, Mistral, Qwen, Yi)
are unaffected by the fix. The expected coefficient shifts (β₁ recovering, α₂ retreating)
will have second-order effects on dense predictions — minor re-fitting at most.

**Diagnostic Clause**: If dense model APE increases by > 10 percentage points, the
coefficient adjustments from Scout correction are harming dense predictions. Consider
adding a MoE-specific correction term (β₈ × isMoE) to decouple Scout and dense fitting.

---

## H-cv: All Three Cross-Validation Tests Pass at <15% MAPE

**Prediction**: After main optimization converges, all three CV holdout tests will pass:
- **CV-1** (Leave-Yi-and-Mistral-Out): TTFT MAPE < 15% and E2E MAPE < 15% on Yi + 2×Mistral
- **CV-2** (Leave-One-Workload-Out): TTFT MAPE < 15%, E2E MAPE < 15%, workload variance < 3%
- **CV-3** (Leave-One-TP-Out): TTFT MAPE < 15% and E2E MAPE < 15% on 6 TP=2 experiments

**Causal Mechanism**: The trained-roofline architecture's basis functions (`max(compute, memory)`)
are physics-derived, not data-derived — they express hardware laws that generalize by design:
- CV-1 (dense family): Compute/memory/weight-load basis functions are architecture-agnostic;
  they transfer from Llama/Qwen to Yi/Mistral because all dense transformers obey the same
  roofline physics
- CV-2 (workload): Basis functions depend only on token counts and batch composition, not on
  semantic workload type; codegen vs roleplay is irrelevant to FLOPs and bandwidth
- CV-3 (TP scaling): The β₄ TP communication term was zero in the trained-roofline prior but
  became 0.396 in iter16 — a non-zero signal that should enable TP=1+TP=4 → TP=2 interpolation

**Why iter17 should improve CV results over iter16**: In iter16, the CV tests were run against
coefficients fitted without the 6 Llama experiments. For CV-1 in particular, the Llama family
was in the TEST set but also absent from the MAIN training signal. With iter17 training on all
15 experiments, the coefficients better represent the full distribution, which may improve
generalization to Yi/Mistral holdouts.

**Diagnostic Clause**:
- CV-1 fails: Scout MoE terms and dense terms are coupled in a way that confounds Yi/Mistral
  predictions → iter18 should investigate per-family corrections
- CV-2 fails (high workload variance): The β₇ per-step constant or α₂ per-token term is
  absorbing workload-specific variance → reduce α₂ bounds or add regularization
- CV-3 fails: β₄ TP correction still has wrong functional form; verify `T_tp` basis function
  computation in `evolved_model.go`

---

## H-full-dataset: Training on All 15 Experiments Tightens Dense Model Fit

**Prediction**: Dense model TTFT APE will decrease for at least 3 of the 6 previously-
unseen Llama experiments, relative to iter16's already-good results (7–34.5% TTFT APE).
The 4 Llama-2 experiments and 2 Llama-3.1 experiments were never seen by the iter16
optimizer — their iter16 numbers reflect generalization, not fit.

**Causal Mechanism**: In iter16, the 9 training experiments were Scout (×4), Mistral (×2),
Qwen (×2), Yi (×1). The optimizer had no Llama-family examples to fit — Llama predictions
at 7–34.5% were emergent generalization. Including all 6 Llama experiments in the
optimizer's objective function gives it direct gradient signal for the Llama model family,
which should reduce their error further.

However, the effect may be modest: iter16's Llama results already generalize well
(Llama-3.1-70B at 7–11% TTFT APE), suggesting the coefficient space for dense models
is already well-identified from Mistral/Qwen/Yi. The main lift is expected for
Llama-2-7B reasoning-lite (34.5% TTFT APE — highest among dense models) where explicit
training signal may help.

**Diagnostic Clause**: If Llama APE increases when moved from eval-only to training, it
indicates the Llama-family gradient conflicts with Scout gradient — the optimizer is
making a Scout vs Llama trade-off. Investigate whether a per-family β₁ correction is
needed.

---

## H-convergence: Patience-Based Stopping Terminates Before 500 Trials

**Prediction**: The optimizer will stop before 500 trials (budget: 1000) due to the
`--patience 100` convergence criterion, with the best trial found before trial 300.

**Causal Mechanism**: The iter16 warm start (loss 60.19%) places the optimizer near a
local minimum already. The TPE sampler with 10 parallel jobs will rapidly explore the
neighborhood of iter16 coefficients. Once the corrected Scout FLOPs reveal the true
minimum, TPE should converge quickly — the architecture is proven and the warm-start
region is close to optimal.

**Note on warm-start mechanics**: Because n_jobs=10, `enqueue_trial` is skipped (Optuna
parallel constraint). The warm-start values in `coefficient_bounds.yaml` are used only
as documentation of the starting point; the optimizer begins with random TPE exploration.
Despite this, iter16 found its best trial at #957 of 1705 — the patience stop at 100
stale trials will yield an equivalent or better result in fewer total trials.

**Diagnostic Clause**: If patience stop triggers before trial 100, suspect the warm start
was too close to a local (not global) minimum and the optimizer could not escape it.
Run with `--patience 200` or a cold start in iter18.
