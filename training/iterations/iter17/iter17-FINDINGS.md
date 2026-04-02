# Iteration 17: Findings and Principles

## Summary

Iteration 17 attempted to improve on iter16's 60.19% loss by training on all 15 experiments
(vs iter16's silent 9/15) with the corrected Scout MoE simulator. **The result: no improvement
over iter16.** The best new coefficients found (CMA-ES, 65.37%) are worse than iter16's
coefficients re-evaluated on the same 15-experiment corrected simulator (60.19%).

**Key discovery**: The `#877` Scout MoE fix was already effectively applied during iter16's
evaluation — `interleave_moe_layer_step=1` was already present in Scout's `config.json` before
the commit. Both iter16 and iter17 evaluate identically on this dimension. The real difference
between iterations was training set size (9 → 15 experiments), which made the optimization
harder without finding a better minimum.

**Status**: ⚠️ No improvement. Iter16 coefficients (60.19%) remain the best available. Scout
MoE is the dominant error source (42–73% TTFT APE) and requires architectural changes in iter18.

---

## Infrastructure Issues Encountered

Iter17 required multiple restarts due to setup problems discovered during execution:

1. **Wrong worktree branch**: The worktree was created from `main` instead of `training`,
   missing the training Python scripts. Fixed by `git reset --hard fcad4689`.

2. **Missing trainval_data in worktree**: Experiment data is gitignored and lives only in the
   main repo's `training/trainval_data/`. Fixed by passing
   `--data-dir /path/to/main/repo/training/trainval_data`.

3. **Missing Llama model configs**: The worktree's `model_configs/` only had configs tracked
   by git. Llama-2 and Llama-3.1 configs were absent, causing 6/15 experiments to silently fail
   in early optimization runs. Fixed by copying configs from the main repo's `model_configs/`.

4. **Trial timeout (120s) too short for 15 experiments**: Each trial evaluates 15 experiments
   sequentially (~73s total). The default 120s timeout was too tight with variance. Fixed by
   setting `--timeout 200`.

5. **Corrupted optimization run**: One run had inconsistent experiment counts — the first 276
   trials saw 9/15 experiments, and later trials saw 15/15 after configs were added mid-run.
   The SQLite study DB was deleted and the run restarted clean.

6. **Warm-start incompatibility with n_jobs > 1**: The optimizer's `enqueue_trial` call was
   incorrectly gated on `n_jobs <= 1`. Fixed by removing the restriction — `enqueue_trial`
   before `study.optimize()` is safe regardless of n_jobs.

7. **Missing `cmaes` Python package**: CMA-ES sampler requires `pip install cmaes`. Fixed.

---

## Search Strategy

Multiple optimization approaches were attempted before concluding:

| Run | Sampler | n_jobs | Trials | Best Loss | Outcome |
|-----|---------|--------|--------|-----------|---------|
| TPE warm-start | TPE | 15 | 60 | 60.19 | Plateaued at warm-start, never improved |
| CMA-ES sequential | CMA-ES | 1 | 11 | 84.38 | Stopped to switch to parallel |
| CMA-ES parallel | CMA-ES | 11 | 188 | **65.37** | Converged to different local minimum |

**TPE finding**: With iter16's coefficients as warm-start, TPE never beat 60.19 in 60 trials.
The optimizer started exactly at a local minimum it had already thoroughly explored in iter16.

**CMA-ES finding**: Starting from the same x0, CMA-ES found a *different* local minimum at
65.37. It did not find the 60.19 basin. This confirms the 15-experiment loss landscape has
multiple stable local minima, with iter16's point being deeper.

---

## Results vs Iter16

### Overall Loss Comparison (15 experiments, corrected simulator)

| Metric | Iter16 coefficients | Iter17 CMA-ES best | Change |
|--------|--------------------|--------------------|--------|
| Overall loss | **60.19%** | 65.37% | +5.18% worse |
| TTFT RMSE | 31.36% | 32.31% | worse |
| E2E RMSE | 28.83% | 33.06% | worse |

**Iter16 coefficients remain the best for the corrected 15-experiment simulator.**

### Per-Experiment Results (iter17 CMA-ES best, sorted by TTFT APE)

| Experiment | Model | TP | TTFT APE | E2E APE | Status |
|---|---|---|---|---|---|
| Scout reasoning-lite | Scout-17B-16E | 2 | 73.4% | 61.0% | ❌ Worst |
| Scout codegen | Scout-17B-16E | 2 | 48.0% | 47.4% | ⚠️ |
| Scout general-lite | Scout-17B-16E | 2 | 45.3% | 56.7% | ⚠️ |
| Scout roleplay | Scout-17B-16E | 2 | 42.5% | 40.4% | ⚠️ |
| Llama-2 roleplay | Llama-2-7B | 1 | 34.5% | 46.4% | ⚠️ |
| Llama-2 reasoning-lite | Llama-2-7B | 1 | 28.0% | 21.4% | ⚠️ |
| Mistral general-lite | Mistral-Nemo-12B | 2 | 28.1% | 18.1% | ⚠️ |
| Llama-2 codegen | Llama-2-7B | 1 | 22.6% | 36.2% | ⚠️ |
| Mistral codegen | Mistral-Nemo-12B | 1 | 17.6% | 32.1% | ⚠️ |
| Llama-3.1 general-lite | Llama-3.1-70B | 4 | 13.3% | 1.3% | ✅ |
| Qwen roleplay | Qwen2.5-7B | 1 | 12.5% | 8.9% | ✅ |
| Llama-3.1 codegen | Llama-3.1-70B | 4 | 8.5% | 6.4% | ✅ |
| Llama-2 general | Llama-2-7B | 1 | 8.6% | 10.5% | ✅ |
| Yi-34B general-lite | Yi-34B | 2 | 5.7% | 8.5% | ✅ |
| Qwen reasoning-lite | Qwen2.5-7B | 1 | 4.0% | 0.9% | ✅ |

**Scout dominates error**: All 4 Scout experiments occupy the top 4 worst positions (42–73%
TTFT APE). Dense models are mostly good (1–35% TTFT APE).

---

## Coefficient Analysis

### Best Coefficients (CMA-ES trial 178)

| Coeff | Description | Iter17 | Iter16 | Change | Assessment |
|-------|-------------|--------|--------|--------|------------|
| **α₀** | QueueingTime (µs) | 16,361 | 15,569 | +5% | Stable |
| **α₁** | PostDecodeFixedOverhead (µs) | 6,270 | 815 | **+669%** | ❌ Exploded |
| **α₂** | OutputTokenProcessingTime (µs) | 20.4 | 45.7 | -55% | ✅ Retreated |
| **β₁** | Prefill correction | 0.164 | 0.201 | -18% | Still collapsed |
| **β₂** | Decode correction | 2.321 | 1.617 | +44% | Higher |
| **β₃** | Weight loading correction | 1.075 | 1.360 | -21% | Lower |
| **β₄** | TP communication | 0.364 | 0.396 | -8% | Stable |
| **β₅** | Per-layer overhead (µs/layer) | 52.2 | 62.2 | -16% | Lower |
| **β₆** | Per-request scheduling (µs/req) | 122.4 | 2.94 | **+4066%** | ❌ Exploded |
| **β₇** | Per-step constant (µs/step) | 170.7 | 169.4 | +1% | Stable |

**α₁ explosion (+669%)**: PostDecodeFixedOverhead jumped from 815µs to 6,270µs (6.27ms per
request). This is 3.4× higher than the trained-roofline prior (1.85ms). Physical interpretation:
the optimizer is compensating for the increased training complexity (15 experiments including
high-latency Llama-2 workloads) by inflating per-request overhead.

**β₆ explosion (+4066%)**: Per-request scheduling jumped from 2.94µs to 122µs. Combined with
the high α₁, the optimizer is allocating large fixed overheads to fit the diverse workloads.

**β₁ still collapsed**: Prefill correction remains at 0.164 (cf. iter16's 0.201). Neither TPE
nor CMA-ES was able to recover β₁ to the physically plausible range (0.5–0.9).

---

## Critical Finding: #877 Fix Was Already Applied in Iter16

The iter17 hypothesis assumed #877 changed the loss landscape for the evolved model. This was
incorrect. Re-evaluating iter16's coefficients on the iter17 simulator (with `#877` fix applied)
gives **exactly** 60.19% — the same as iter16's original score.

Investigation revealed: `interleave_moe_layer_step: 1` was already present in Scout's
`config.json` before the `#877` commit (`fcad4689`). The `evolved_model.go` code to use this
field was added in iter16 (commit `0b41c913`). Both the code and the config field were already
in place during iter16's evaluation; `#877` added parsing support that was already effectively
active because the config field existed.

**Implication**: Iter17's real difference from iter16 was only the training set (9 → 15
experiments). The #877 fix was a red herring for the loss improvement hypothesis.

---

## Comparison to Previous Iterations

| Iteration | Loss | TTFT RMSE | E2E RMSE | Key Change |
|-----------|------|-----------|----------|------------|
| iter8 | 155.4% | 64.0% | 91.4% | Previous best pre-iter16 |
| iter16 | **60.19%** | 31.36% | 28.83% | Trained-roofline architecture |
| **iter17 (CMA-ES)** | 65.37% | 32.31% | 33.06% | 15-exp training, different basin |

---

## Recommendations for Iter18

### Priority 1: MoE-Specific Architecture Term (Scout accounts for all top-4 worst results)

The 7-term formula cannot simultaneously fit dense models (3–35% APE) and Scout MoE (42–73%
APE). Scout's expert routing, FP8 quantization, and EP=TP parallelism produce fundamentally
different latency characteristics. Two approaches:

1. **β₈ × isMoE**: Add a binary MoE correction term. Simple, conservative.
2. **Separate MoE/dense coefficients**: Maintain two coefficient sets, selected by model type.
   More expressive but doubles search space.

### Priority 2: Accept Iter16 Coefficients as Iter17 Output

The iter16 coefficients (60.19% on 15 experiments with corrected simulator) remain the best
available. Since iter17 found nothing better, the practical outcome of iter17 is: **verified
that iter16's coefficients generalize correctly to all 15 experiments** when evaluated with
the current simulator. Iter17 added no new trained coefficients.

### Priority 3: Fix Search Infrastructure for Future Iterations

Three improvements for iter18:
1. Create a symlink `training/trainval_data_actual` → main repo trainval_data, or document
   the `--data-dir` absolute path requirement in the iteration template.
2. Set default timeout to 200s for 15-experiment evaluations.
3. Add `--sampler cmaes` as a standard option in iteration manifests.

---

## Meta-Learning: What Iter17 Taught Us

1. **The #877 fix was a false hypothesis**: We assumed it changed the loss landscape, but the
   config field was already present. Always verify simulator behavior with a direct A/B test
   before attributing differences to code changes.

2. **More experiments ≠ better optimization**: Expanding from 9 to 15 experiments made the
   optimization harder without finding a better global minimum. The 15-experiment landscape
   has multiple stable basins; iter16's point happens to be the deepest one found.

3. **CMA-ES explores differently but not necessarily better**: CMA-ES with warm-start from
   iter16 converged to a different local minimum (65.37%) rather than the iter16 basin (60.19%).
   This suggests the basins are separated in coefficient space and CMA-ES's initial step size
   (sigma0=0.3) overshot the iter16 basin.

4. **Scout MoE is the architectural limit**: 7 generations of coefficient tuning have not
   meaningfully reduced Scout error below 40% TTFT APE. The formula needs an architectural
   extension, not better coefficients.
