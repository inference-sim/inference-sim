# Result: BLIS on blis-latency-kernel, against AISimulate

Worktree-scoped experiment. In the headline comparison latency comes exclusively from
`blis-latency-kernel`. BLIS's roofline and trained-physics backends are constructed only by
`cmd/estimatorscore`, which scores them as additional columns on the subset they are
calibrated for; they supply nothing to the figures below.

> **Reading order.** This document is chronological: it records what was measured when, including
> figures that a later measurement superseded. The live numbers are under
> [Current standing](#current-standing). The Headline below is the FIRST scored run and is
> superseded -- it predates the InferenceX protocol match, which changed both the point count
> (238 to 192) and AISimulate's own re-anchored figure (7.15% to 8.87%). It is kept because the
> path from it to the current number is the evidence that no step was tuned.

## Headline (superseded -- see [Current standing](#current-standing))

On the 238 points measured under vLLM -- the engine BLIS models:

| Model | n | MAPE | median | worst |
|---|---|---|---|---|
| BLIS + blis-latency-kernel | 238 | **12.01%** | — | — |
| AISimulate, same points | 238 | **7.15%** | 4.19% | 75.12% |

Excluding the four sweeps whose MEASURED curve is non-monotone in concurrency:

| Model | n | MAPE | median | worst |
|---|---|---|---|---|
| BLIS + blis-latency-kernel | 213 | **11.54%** | 8.74% | 67.07% |
| AISimulate, same points | 213 | **6.68%** | 4.09% | 75.12% |

**AISimulate is ahead.** The goal was to beat it; this does not.

These figures are HIGHER than the 11.26% and 10.72% this document reported earlier, and the
earlier numbers were wrong. They were taken before the harness had a convergence criterion,
and part of their apparent advantage was measurement noise flattering the result. See
"Making the measurement converge" below. Both corrections are recorded rather than quietly
replaced, because a number that moved when the method was fixed is exactly the kind of thing
a reviewer needs to see.

What the thesis did establish: the kernel alone scores 13.67% on the full 447 points with the
resident batch assumed equal to client concurrency. Letting BLIS's scheduler decide the
resident batch moves the vLLM subset to 12.01%. Part of the gap closed, and a larger part
remains.

Note also that the 9.41% named in the goal is AISimulate's figure over all 447 points,
including the sglang and trt arms BLIS does not model. On the vLLM subset AISimulate scores
7.15%, and 6.68% on the monotone part of it. That is the real bar.

## Making the measurement converge

Before any modelling conclusion could be trusted, the harness had to stop reporting a number
that depended on itself. It did:

| Completions per point | MAPE (monotone vLLM) |
|---|---|
| 20 | 9.92% |
| 30 | 11.20% |
| 40 | 9.89% |
| 60 | 10.72% |
| 100 | 10.64% |

A 1.3-point swing with no trend -- larger than several of the modelling effects this document
reports, and enough to make any comparison between two variants meaningless. Two causes, both
properties of the harness rather than of the deployment:

**A closed-loop pool does not start in its steady state.** At t=0 all N users submit at once,
so the first requests see a resident batch that is still filling and a cold KV cache. Their
inter-token latency was pooled with the settled requests', so the mean depended on what
fraction of the run was transient.

**A fixed completion budget covers fewer pool cycles as concurrency rises.** At concurrency 32
a 40-completion budget is barely one cycle, and the mean then depends on where the run
stopped. Measured on `minimax-m2.5-b200-fp8-vllm-tp4` at c=32: 9746, 8186, 9397, 9490 us at
20, 40, 100, 200 completions, while c=8 had converged by 40.

The criterion now: each point gathers `max(24, 4 x concurrency)` completions and discards the
first half. With it, the score is monotone and tight in the budget -- 11.31%, 11.54%, 11.68%
at 2, 4 and 6 cycles -- so a 0.4-point modelling effect is now resolvable where before it was
buried in noise. The criterion is set from the shape of the transient, applies identically to
every point, and is not a tuning knob: it made the reported score WORSE.

## The comparison is verified, not asserted

AISimulate's error here is recomputed from its own published per-point relatives,
re-anchored to its own lowest concurrency exactly as BLIS is. Reproduced independently
outside the Go harness: 7.15% on the same 238 points. Both sides are the same quantity --
a normalised curve divided by the measured normalised curve.

## Where the error is, measured

By concurrency:

| Concurrency | n | BLIS | AISimulate |
|---|---|---|---|
| 4 | 43 | 0.65% | 0.37% |
| 8 | 43 | 7.80% | 5.66% |
| 16 | 43 | 13.76% | 7.27% |
| 32 | 40 | 15.70% | 6.35% |
| 64 | 38 | 18.03% | 11.31% |
| 128 | 15 | 13.74% | **18.72%** |
| 256 | 6 | 4.68% | 4.14% |
| 512 | 3 | 20.02% | **28.10%** |
| 1024 | 2 | 39.13% | 16.58% |
| 2048 | 1 | 59.47% | 15.91% |

BLIS beats AISimulate at 128 and 512 and loses badly at 1024 and 2048 (three points
total). The bulk of the deficit is the 16-64 band, which is 121 of 238 points.

By chip family:

| Family | n | BLIS | AISimulate |
|---|---|---|---|
| hopper | 133 | 10.60% | 4.84% |
| blackwell | 105 | 12.36% | 10.08% |

The ten worst sweeps are dominated by one model:

| Scenario | Workload | n | BLIS | AISimulate |
|---|---|---|---|---|
| minimax-m2.5-h200-fp8-vllm-tp4 | 1k8k | 5 | 33.36% | 12.69% |
| minimax-m2.5-b300-fp4-vllm-tp2-ep4-dp2 | 1k1k | 4 | 32.71% | 4.80% |
| minimax-m2.5-h200-fp8-vllm-tp4 | 1k1k | 9 | 23.57% | 13.43% |
| minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2 | 1k1k | 3 | 20.32% | 13.02% |
| minimax-m2.5-b200-fp8-vllm-tp2 | 1k8k | 5 | 18.86% | 4.03% |

Seven of the ten worst are minimax-m2.5. On a deployment BLIS fits well --
`gpt-oss-120b-h200-fp4-vllm-tp4` at 1k1k -- it beats AISimulate at four of five points
(0.74% against 5.05%, 0.49% against 4.25%), so the machinery is capable and the deficit is
localised rather than systemic.

## What is ruled out as the cause

**Not insufficient simulation.** Mean ITL varies under 2% between 30 and 200 completed
sessions per point (`minimax-m2.5-h200-fp8-vllm-tp4`, c=16: 9633/9671/9823 us; c=32:
13778/13438/13578 us). The runs are in steady state and deterministic at a fixed seed.

**Not a metric mismatch.** AISimulate's 7.15% is reproduced independently from its own data.

**Not KV capacity being wrong.** All 39 deployments derive a budget from the kernel's memory
methods after the upstream `/expertTensorShards` fix, and preemption is active.

**Partly unfittable data.** `minimax-m2.5-h200-fp8-vllm-tp4` at 1k1k has a measured TPOT that
FALLS from 3.1848 at c=16 to 2.2677 at c=32. No monotone model fits that transition; it is one
of the four non-monotone transitions in the corpus. This arm alone contributes 9 of the 238
points and 23.57% error.

## What remains untested

The 209 non-vLLM points (sglang, trt) are out of scope for a BLIS fidelity claim and are not
included above. Running them would produce a number, not evidence.

## The ceiling on what this comparison can show

Two bugs were found and fixed after the first score. One was serious; the second finding is
a limit rather than a bug, and it bounds the whole exercise.

**Fixed: data parallelism was not scaling the batch caps.** vLLM runs `dp` independent
EngineCores, each with its own `max_num_seqs` and token budget, splitting requests disjointly
across them -- the same rule `KVBudget` already applied to KV blocks. The harness set a single
core's caps. On the two `ep4-dp2` sweeps, which begin at concurrency 256 against a stated
`max_num_seqs` of 256, the resident batch saturated at the very first point and the predicted
curve went FLAT: 1.002 and 1.003 across a four-fold concurrency rise, against a measurement
that doubled. At concurrency 512 the error fell from 59.47% to 0.84% once dp was applied.

Aggregate effect was small -- 11.38% to 11.26% -- because only 3 of 238 points sit above
`256 x dp`. A serious bug on few points.

**Measured, not assumed: the snapshot publishes no engine settings, and it barely matters.**
Its topology records carry `framework`, `precision`, `serving`, `spec_method` and the five
parallelism widths. They do NOT carry `max_num_seqs`, `max_num_batched_tokens`, `block_size`,
`gpu_memory_utilization`, `cudagraph_mode`, `scheduling_policy`, or the chunked-prefill and
prefix-caching flags. The scenario files default them.

A first draft of this document argued that this bounds what the comparison can show, because
the defaults were justified for a step-time ratio ("a constant cancels") and that
justification fails with a scheduler in the loop. The first half is right; the conclusion was
not, and `cmd/sensitivity` settles it by measurement on the 213 monotone vLLM points:

| Setting | MAPE | delta |
|---|---|---|
| scenario values | 11.20% | — |
| max_num_seqs x4 | 11.03% | -0.17 |
| max_num_seqs /2 | 11.56% | +0.36 |
| token budget x2 | 11.24% | +0.03 |
| KV bound removed entirely | 11.21% | +0.00 |

Against a 4.5-point gap, the largest of these is 0.36. The resident batch in this corpus is
set by the offered concurrency and the work per request rather than by the caps, so the caps
barely register. The unstated settings are a footnote, and the gap is a modelling gap.

The dp fix stands for a different reason: its 3 points sat AT the cap, where it binds
absolutely rather than marginally.

## What was ruled out as the cause of the minimax deficit

`ExpertsTouched(tokens, E, k, local) = local * (1 - ((E-k)/E)^tokens)` is identical for
minimax (E=256, k=8) and gpt-oss (E=128, k=4) at every batch size -- 0.031 at one token,
0.398 at 16, 0.983 at 128 -- because the two models' E/k ratios coincide. So the expert-count
term cannot explain a minimax-specific error, and the hypothesis that it did is rejected.

## Standing

The goal was to beat AISimulate. On the vLLM subset AISimulate scores 7.15% (6.68% on the
monotone subset), not the 9.41% quoted in the goal, which is its figure over all 447 points
including the sglang and trt arms BLIS does not model. BLIS scores 11.26% and 10.72%. The
gap is real and is not closed.

## Where the residual actually is, decomposed

The per-resource breakdown at decode (context 1024, microseconds per step) isolates it:

| batch | minimax SM | minimax HBM | gpt-oss SM | gpt-oss HBM |
|---|---|---|---|---|
| 1 | 793 | 1,107 | 507 | 566 |
| 8 | 827 | 2,900 | 545 | 1,310 |
| 32 | 942 | 6,786 | 673 | 2,936 |
| 64 | 1,096 | 9,043 | 844 | 3,907 |
| 256 | 2,020 | 11,143 | 1,870 | 5,049 |

Both are HBM-bound and the overlap-to-serialized ratio is similar (1.26-1.47 on both), so the
choice of band edge is not the minimax-specific cause.

The cause is the composition of that HBM term. For minimax at tp=4 with expert parallelism
off -- 62 layers, 256 experts, three matrices at 1536x3072, fp8 -- expert weights are

| batch | experts touched | expert GiB | KV GiB | expert share |
|---|---|---|---|---|
| 1 | 8.0 | 1.63 | 0.030 | 98.2% |
| 8 | 57.4 | 11.73 | 0.242 | 98.0% |
| 32 | 163.3 | 33.37 | 0.969 | 97.2% |
| 64 | 222.4 | 45.46 | 1.938 | 95.9% |
| 256 | 255.9 | 52.30 | 7.750 | 87.1% |

So 87-98% of minimax's decode HBM traffic is expert weights, and the term grows 32x between
batch 1 and 256 as `ExpertsTouched` saturates toward all 256 local experts. That is exactly
the shape of the observed error: a systematic over-prediction that GROWS with batch, on the
model whose step is most dominated by this one term.

`ExpertsTouched` itself is not the culprit -- it is identical for minimax (E=256, k=8) and
gpt-oss (E=128, k=4) at every batch size. The remaining candidate is the RATE the expert bytes
are charged at, which is a calibration question answerable against AISimulate's own MoE
measurement parquets rather than by adjusting a number.

That is the open lead. It is checkable, which is the bar for pursuing it.

## The expert-rate hypothesis, tested and rejected

The residual's signature -- a systematic over-prediction growing with batch, concentrated on
the model whose decode step is 87-98% expert-weight traffic -- pointed at the rate those bytes
are charged at. That is checkable against NVIDIA's own MoE measurements rather than adjustable,
which is why it was the lead worth pursuing.

`b200_sxm/moe/trtllm/1.3.0rc20/moe_perf.parquet` holds 218,457 measured MoE latencies, and
6,804 of them are at minimax-m2.5's exact geometry: hidden 3072, inter 1536, 256 experts,
top_k 8. Filtering further to the deployment actually being scored -- `moe_tp_size` 4 with
`moe_ep_size` 1, nvfp4, balanced routing -- leaves 27 points per kernel implementation.

The sweep carries two implementations, which matters and nearly produced a wrong conclusion:
`moe_torch_flow_min_latency` and `moe_torch_flow_cutlass` differ by roughly 5x at every token
count. A latency-sensitive engine picks the faster one, so that is the comparison.

Predicted over measured, where the prediction is the kernel's own form
`max(flops/(peak x eff), touched_expert_bytes/bandwidth)`:

| tokens | 1 | 8 | 32 | 64 | 80 | 3072 | 8192 | 16384 |
|---|---|---|---|---|---|---|---|---|
| vs min_latency | 0.12 | 0.65 | 0.76 | 0.99 | 1.03 | 0.53 | 0.37 | 0.39 |
| vs cutlass | 0.03 | 0.13 | 0.15 | 0.22 | 0.23 | 0.14 | 0.09 | 0.10 |

Geometric mean against the faster kernel: **0.625**. Against the slower: 0.153.

**The hypothesis is rejected.** The MoE term is UNDER-predicted at almost every token count --
it crosses 1.0 only briefly around 64-80 tokens and falls back to 0.37-0.53 at large batches.
An under-predicted term cannot produce the observed over-prediction, and its shape in the batch
is wrong for the residual too: the error grows with batch while this ratio falls.

So the expert rate is not the cause. Two things follow. The residual must come from a term that
grows with batch and is charged too heavily -- attention, the collectives, or the per-layer
composition rather than the MoE GEMM. And separately, the MoE term being 0.625 of measurement
is a real calibration gap in its own right, in the direction that would make predictions
optimistic; it is not what this experiment is chasing, but it belongs in the record.

This is the second hypothesis rejected by data rather than by argument, after `ExpertsTouched`.
Both rejections cost little and prevented a change that would have been justified by a story
instead of a measurement.

## A confirmed over-prediction in decode attention

With the MoE rate rejected, the search narrowed to a term that GROWS with batch and is charged
too heavily. Decode attention is the other batch-scaling term in the per-resource breakdown,
and NVIDIA's generation-attention sweep can test it directly.

`h200_sxm/attention/trtllm/1.3.0rc20/generation_attention_perf.parquet`, filtered to
minimax-m2.5's per-rank attention geometry at tp=4 -- 12 query heads, 2 KV heads, head
dimension 128, full attention, fp8 KV cache -- gives 164 measured points.

One column reading had to be corrected before the comparison meant anything. `isl` is 1 for
every generation row; the context length is carried by `step`, which runs 1 to 131071. A first
pass used `isl`, which made the predicted KV read about 2000x too small and collapsed the
prediction onto the floor, giving 0.754. Checking that latency actually tracks `step` -- 5.78 us
at step 1 rising to 140.75 us at step 131071, at fixed batch 8 -- settled it. The first check of
that ("latency is flat in step") looked only at steps 1 to 63, where the floor dominates and it
IS flat; extending the range corrected it.

Predicted over measured, where the prediction is the registry's own form
`floor + kv_bytes/rate` with the committed h200 constants (floor 11.0 us, rate 2.784e6 B/us):

| batch | 1 | 8 | 32 | 128 | 256 | 512 | 1024 | 2048 |
|---|---|---|---|---|---|---|---|---|
| geo-mean ratio | 1.372 | 1.548 | 1.563 | 1.506 | 1.195 | 1.027 | 0.680 | 0.432 |

Restricted to the regime the corpus actually runs -- context 512 to 8191, batch 4 to 256, which
is where 121 of the 213 scored points sit -- 26 measured points give a geometric mean of
**1.294**, and the ratio stays between 1.20 and 1.39 across every batch size in that band.

**So decode attention is over-predicted by about 30% in the regime being scored.** That is the
right sign and the right place: it is charged too heavily, it scales with batch, and its excess
is largest in the 8-128 band where the residual is worst.

This is a calibration finding about `blis-registry`, not about BLIS or the adapter. Acting on
it means re-fitting `attention_decode_floor` and `attention_decode_rate` against this sweep,
which is a change to a committed coefficient set outside this worktree, and the fit would have
to hold across every part rather than being tuned on the arm that exposed it. It is recorded
here for that decision rather than taken.

Worth noting what it does NOT explain: the ratio falls below 1.0 by batch 1024, so the
over-prediction is not uniform, and the three worst points in the corpus sit at concurrency
1024 and 2048 where attention is UNDER-predicted. More than one term is in play.

## Correction: the attention over-prediction is not family-wide

The 1.29x decode-attention over-prediction recorded earlier was measured on ONE geometry --
12 query heads, 2 KV heads, head dimension 128, which is minimax-m2.5's per-rank shape at tp=4
-- on 26 points. `blis-registry/scripts/validate_against_aisimulate_tables.py` now prices the
committed coefficients against every measured shape in the family and finds:

| population | n | predicted / measured |
|---|---|---|
| h200 gqa, corpus regime, minimax per-rank geometry | 26 | **1.294** |
| h200 gqa, corpus regime, every geometry | 3,003 | **0.795** |

Both figures are correct and they answer different questions. Across the family the committed
form is 20% too CHEAP; on one shape it is 29% too expensive. A coefficient scoped to a part
applies to every geometry on it, so the family-wide figure is the one that describes the
coefficient, and the per-geometry figure describes the residual at the shape minimax happens to
run.

The consequence matters more than either number: a form that is 0.80 on average and 1.29 on one
shape is NOT uniformly mis-scaled. Its error depends on geometry. That is scatter, not bias,
and it is precisely what the error decomposition predicted -- BLIS and AISimulate carry
identical mean log-error and differ only in spread. A re-fit of the floor and rate moves the
average and cannot fix the geometry dependence, which is why "recalibrate attention" would not
have closed the gap and why the decomposition was worth doing before the re-fit.

Two filtering details had to be right for the two figures to be comparable at all, and both
were wrong in a first version of the gate:

  - The KV cache dtype halves or doubles bytes per token. Pooling fp8 and bfloat16 rows gave
    1.156 where the fp8 rows alone give 1.294. The corpus runs fp8 throughout, so the check
    filters to it.
  - The collection must be pinned. Globbing every framework and version gave 66,148 rows where
    the committed fit used 40,367, and a different fit with them.

## What the sliding-window split changed, measured

The per-kind swa coefficients are better calibrated in isolation -- a windowed kernel sustains
0.26 of peak bandwidth on h200 against 0.58 for full attention, and the naive form is 32x to
245x wrong on windowed rows across six parts. Dispatching on the kind made the END-TO-END shape
score WORSE:

| | before | after |
|---|---|---|
| all 447 points | 13.67% | 13.86% |
| gpt-oss only (the only model with swa layers) | 11.65% | 12.09% |

147 points moved, 85 worse and 62 better. The mechanism is visible: the swa law makes windowed
layers more expensive, gpt-oss was already over-predicting on 79 of its 190 points, and the
change raised the prediction on 146 of them.

So a more accurate per-kernel coefficient moved the aggregate the wrong way. That is a finding
about gpt-oss, not an argument against the coefficient: something else in its step is already
too expensive, and under-charging the windowed layers was partly cancelling it. The
coefficients stay, because they are measured and the cancellation was accidental; the
cancelling term is the thing to find.

## Current standing

Measured under the protocol InferenceX actually used, which `APPLES-TO-APPLES.md` documents and
`REPRODUCE.md` shows how to re-run:

| Model | n | MAPE | median | worst |
|---|---|---|---|---|
| BLIS + blis-latency-kernel | 192 | **10.41%** | 7.59% | 76.72% |
| AISimulate, same points | 192 | **8.87%** | 5.68% | 75.12% |

Monotone subset only: 9.92% against 8.32% on 171 points.

By chip family:

| Family | n | BLIS | AISimulate |
|---|---|---|---|
| Blackwell | 84 | **11.91%** | 12.61% |
| Hopper | 108 | 9.24% | 5.96% |

**AISimulate is ahead overall and this kernel is ahead on Blackwell.** That split is the most
useful diagnostic here: AISimulate's error more than doubles between the families while this
kernel's moves by two and a half points. A table-interpolating model is limited by how densely
the part was collected; a closed-form one is limited by its forms, which do not know which part
they are on.

### How the figure moved

| Stage | vLLM subset |
|---|---|
| Kernel alone, resident batch assumed equal to concurrency | 13.67% (447 pts) |
| BLIS scheduler deciding the resident batch | 14.98% |
| Routed-expert term composing as a sum | 10.39% |
| Warm-up and request budget matched to InferenceX's protocol | **10.41%** |

The last step is behaviour-neutral to two decimal places, which is the useful result: it confirms
the harness's earlier convergence criterion was already converged, so no figure in this project
rests on a budget this project chose. The protocol is now the measured one rather than a defended
one.

### What remains

About 1.5 points, and it is not closable by recalibration. `cmd/kernelscore` reports the
decomposition directly (`harness.Logs`), in log space, where a ratio's bias and scatter separate:

| model | n | mean log | sd log | bias | floor |
|---|---|---|---|---|---|
| BLIS + blis-latency-kernel | 192 | +0.0165 | 0.1442 | +1.67% | **10.23%** |
| AISimulate, same points | 192 | +0.0462 | 0.1122 | +4.73% | 7.73% |

`mean log` is the bias a single rescaling could remove; `sd log` is the scatter it could not;
`floor` is the mean magnitude surviving perfect bias removal.

This kernel's bias is 2.8x SMALLER than AISimulate's -- +0.0165 against +0.0462 -- and its
scatter is 1.29x larger. The whole of the 1.5-point deficit is scatter: removing this project's
bias perfectly would reach 10.23%, still above AISimulate's 8.87%, while removing AISimulate's
would take it to 7.73%. There is more recalibration headroom in the baseline than in this kernel,
and none of it is available to this kernel by recalibration.

Scatter is what a closed-form model trades away against a 1.87-million-row interpolation over 14
operator families.

> An earlier version of this section reported "near-identical mean log-error -- +0.0461 against
> +0.0462 -- and differ only in spread, 0.1677 against 0.1119", concluding the two models carry
> the same bias. The AISimulate figure was right and this project's was not. Those numbers were
> prose with no command behind them, which is how the error survived; they are now computed by
> the scorer and reproduced by `REPRODUCE.md`. The qualitative conclusion -- the gap is scatter,
> not bias -- is unchanged and in fact strengthened.

One regression is unexplained and is recorded rather than buried: on the absolute-ITL corpus the
two Granite arms improved sharply and Kimi-K3 improved, while Nemotron-3-Ultra worsened from
86.8% to 117.6%. Three hypotheses about it were tested against NVIDIA's tables and all three were
wrong, including one that scored better and was reverted because it was right for MoE and wrong
for dense GEMM. A change justified on MoE tables should not have made Nemotron worse.

## Reporting the sign, and what it changed

Until this point every figure here was an absolute MAPE. Magnitude answers how wrong a model is;
it cannot answer which way, and the two imply different fixes. A consistent sign is a
miscalibrated coefficient and is removable by rescaling. A sign that turns with batch size is a
missing mechanism and is not.

`cmd/kernelscore` now reports both. Errors are kept SIGNED at the point of computation and
absoluted only at each aggregation (`harness.Abs`), so every previously published absolute figure
is unchanged by construction -- the headline reproduces at 10.41% against 8.87% on 192 points.

| model | n | mean | median | mean abs | over | p10 | p90 |
|---|---|---|---|---|---|---|---|
| BLIS + blis-latency-kernel | 192 | +2.71% | **+0.78%** | 10.41% | **53%** | -11.61% | +19.60% |
| AISimulate, same points | 192 | +5.39% | +4.02% | **8.87%** | 71% | -6.83% | +19.77% |

Positive means the model predicted a LARGER relative rise in time per output token than was
measured. This kernel is nearly unbiased -- a median of +0.78% and a coin-flip 53% of points over
-- while AISimulate over-predicts on 71% of points with a median of +4.02%. Read with the log-space
decomposition above, the ranking by absolute MAPE conceals that this kernel already wins the part
of the error that is cheap to fix and loses the part that is not.

### A hypothesis the sign killed

The per-concurrency magnitudes rise monotonically -- 4.79% at concurrency 8 to 12.76% at 128 --
which reads as a mechanism that compounds with batch size, and the obvious candidate was missing
contention: a `max` over resources that under-states a step the more requests share it.

The sign refutes it. Missing contention would show a consistent NEGATIVE sign that deepens with
concurrency. Instead:

| conc | n | BLIS signed (magnitude) | AISimulate signed (magnitude) |
|---|---|---|---|
| 8 | 43 | +2.40% (4.79%) | +4.19% (5.66%) |
| 16 | 43 | +3.38% (10.38%) | +2.66% (7.27%) |
| 32 | 40 | +5.38% (11.73%) | +0.43% (6.35%) |
| 64 | 38 | +2.14% (12.36%) | +8.56% (11.31%) |
| 128 | 14 | +0.25% (12.76%) | +16.50% (20.06%) |
| 256 | 4 | -4.37% (7.66%) | +6.21% (6.21%) |
| 1024 | 2 | -21.73% (21.73%) | +15.55% (16.58%) |
| 2048 | 1 | -47.87% (47.87%) | +15.91% (15.91%) |

The signed mean does not grow with concurrency; it wanders and then turns negative at the top.
Magnitude grows while direction does not, which is scatter increasing, not bias accumulating. An
earlier version of this document presented the magnitude column alone and drew the
batch-composition conclusion from it; that conclusion was unsupported and the sign is what showed
it.

The three points above concurrency 512 deserve their own note: -21.73% and -47.87% are large
under-predictions on two and one point respectively, the opposite sign from the rest of the curve,
and AISimulate is better there (+15.55%, +15.91%). That is a real weakness on 3 of 192 points, and
it is not the same phenomenon as the 16-64 band.

## Prefill attention: the right defect, the wrong regime

One structural defect was found, measured, and deliberately NOT fixed. It is recorded because the
measurement is reusable and the reasoning is the point.

The kernel prices prefill attention as `floor + causal_flops / (peak * eff(tokens) * work_scale)`,
where `eff` is the dense-GEMM ramp `eps_max * m / (m + m_half)` keyed on the step's total scheduled
tokens. That ramp is fitted on MATMUL ROWS (`gemm_eps_max_bf16`, `gemm_m_half_bf16`), so one
request of 2048 tokens and eight of 256 receive identical attention efficiency despite entirely
different attention shapes.

AISimulate's own interpolation design states the governing fact: curvature is a property of the
AXIS, not the table -- context attention is quadratic along sequence and roughly linear along
batch and heads, and confining a sqrt transform to the sequence axis moved their measured interior
error from 9.44% to 2.00%.

Tested against NVIDIA's context-attention parquets by
`blis-registry/scripts/probe_attention_prefill_axis.py`, which refits `(floor, work_scale)` per arm
on the same rows over the same grids as the shipping fitter and scores on held-out shapes:

| key | h200 | h100 | b200 | b300 | gb200 | l40s |
|---|---|---|---|---|---|---|
| `b*isl` (current) | — | — | — | — | — | — |
| `isl` only | **-5.2%** | **-7.6%** | **-5.7%** | **-5.9%** | **-5.7%** | **-10.6%** |
| `sqrt(b)*isl` | -0.6% | -1.0% | -0.7% | -0.8% | -0.7% | -1.6% |
| `b*sqrt(isl)` | +6.2% | +9.4% | +7.3% | +8.0% | +7.6% | +8.3% |

`isl` alone wins on every part, with `work_scale` barely moving (0.48 to 0.48 on h200), so this is
a better key rather than a rescaling. The script guards itself: the current key fitted on all rows
must reproduce the registry's committed h200 entry -- n=55,096, floor 26.5us, work_scale 0.48 -- and
it prints `GUARD FAILED` and exits non-zero if it cannot, because a harness that cannot reproduce
the published fit from the published data is not measuring the same quantity.

The mechanism is measurable directly. At fixed sequence length and head count, measured latency
scales as `batch^0.730` (isl=512), `batch^0.843` (2048) and `batch^0.943` (8192): sub-linear,
approaching linear as sequences grow. `causal_flops` already scales linearly in batch, so putting
batch in the efficiency key as well makes efficiency RISE with batch and partially cancels the
over-count -- an accidental approximation of sub-linearity with the wrong functional form,
confounded across two axes.

Neither key fixes the trend, which is why no change was made. Signed residuals by batch, h200:

| batch | `b*isl` (current) | `isl` only |
|---|---|---|
| 1 | +28.2% | +24.4% |
| 16 | -8.6% | -7.3% |
| 64 | -50.7% | -46.0% |
| 256 | -80.2% | -73.7% |

Both fail from +28% over-prediction at batch 1 to -80% under at batch 256. `isl` only reduces the
trend; no single scalar key can remove it, because the model needs a batch term with its own
exponent.

### Why it was not fixed

Instrumenting `StepTime` over the whole vLLM corpus -- 8,186,921 simulated steps -- gives the
distribution of prefill requests per step:

| prefill requests in the step | steps | share |
|---|---|---|
| 0 | 8,050,278 | 98.33% |
| 1 | 109,115 | 1.33% |
| 2 or more | 27,528 | 0.336% |
| 16 or more | 54 | 0.00066% |

The defect can only bite where two or more prefills share a step, which is 0.336% of steps, and
the large residuals need batch 16 or more, which is 0.00066%. Weighting the measured per-batch
improvement by these frequencies bounds the effect on step time across the run at **-0.0388%**,
and that is generous because attention is only part of a step.

A single 2,877-token prefill already fills the token budget, so vLLM's scheduler essentially never
co-schedules prefills. The parquet sweeps batch 1 to 256 uniformly; real serving traffic does not.

The finding stands and the fix is not worth making here. It would matter for a deployment that
batches prefills -- a long-context ingestion workload, or an engine with a much larger token
budget -- and the measurement above is what a future change should be judged against.

Since 98.33% of steps are pure decode, that is where essentially all of the 10.41% must live.

## Four estimators on the Hopper subset

The comparison above asks how this kernel compares with NVIDIA's simulator. It does not ask what
the kernel earned over what BLIS shipped before. `cmd/estimatorscore` answers that by scoring four
estimators on one subset, with everything except the forward-pass model held fixed.

24 sweeps, 104 points, framework vLLM, chips h100 and h200:

| estimator | n | mean | median | mean abs | over | p10 | p90 |
|---|---|---|---|---|---|---|---|
| blis-latency-kernel | 104 | +1.71% | **-0.46%** | **8.87%** | 44% | -10.76% | +16.91% |
| AISimulate | 104 | +1.75% | +2.02% | **6.04%** | 66% | -7.13% | +10.45% |
| roofline | 104 | +161.54% | +135.30% | 161.54% | 100% | +53.12% | +279.13% |
| trained-physics | 104 | -38.93% | -39.04% | 38.93% | 0% | -67.20% | -12.59% |

The kernel is **18x better than roofline and 4.4x better than trained-physics** on identical
points. That is the justification for the kernel, measured rather than asserted.

### What is held fixed

Step time, and nothing else:

- **KV blocks come from the kernel for every arm.** BLIS normally sizes KV with
  `latency.CalculateKVBlocks`, which sits outside the `sim.LatencyModel` seam. Letting each arm
  size its own pool would give each a different resident batch and the result would mix admission
  behaviour into a step-time comparison.
- **The host per-token cost is the kernel's for every arm**, 45.9 us/token. The metric is mean
  inter-token latency, so an arm with a different per-token host cost would differ for reasons
  unrelated to step time. Zeroing it -- the first attempt -- would have made every analytic arm
  45.9 us/token cheaper than the kernel per token.
- Workload, seed, warm-up discard, request budget, batch caps and dp scaling are the harness's,
  untouched.

Audited rather than assumed: at every concurrency on `gpt-oss-120b-h200-fp4-vllm-tp4`, all three
arms report an identical KV pool of 817,357 blocks, complete exactly `12 x concurrency` requests
and discard exactly `2 x concurrency`.

### The two analytic arms fail structurally, in opposite directions

Roofline is over on 100% of points and trained-physics under on 0% -- saturated signs, which
indicate a missing term rather than a mis-set coefficient. The predicted curves show it, against a
measurement that rises 2.36x:

| concurrency | measured | kernel | roofline | trained-physics |
|---|---|---|---|---|
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 8 | 1.1565 | 1.1329 | 1.7947 | 1.0027 |
| 16 | 1.3275 | 1.3579 | 3.1124 | 1.0082 |
| 32 | 1.8010 | 1.6830 | 4.9277 | 1.0191 |
| 64 | 2.3641 | 2.0878 | **6.7031** | **1.0408** |

The absolute levels explain both. Roofline anchors at 1,086 us against the kernel's 3,336 and
reaches 7,283: it is nearly free at low batch, having no fixed overhead to speak of, so
normalising to its own anchor inflates every later point. Trained-physics runs 17,743 us to
18,466 us -- a 1.04x rise -- because its per-layer and per-step constants dominate and concurrency
barely moves it.

### What this comparison does not show

- **Neither analytic arm was calibrated for these deployments.** Trained-physics is a single
  global fit (11 beta + 3 alpha, iter29, loss 34.57%) transcribed from `defaults.yaml`, with no
  pipeline on `main` that can re-derive it, and it is not scoped per chip. Roofline reads
  `mfuPrefill`/`mfuDecode` per chip but has never seen mxfp4 gpt-oss or fp8 minimax. This is a fair
  measurement of what BLIS shipped, not an indictment of those models in the regime they were
  fitted for.
- **Hopper only, and that is forced rather than chosen.** `hardware_config.json` carries H100,
  H200, A100-SXM, A100-80 and L40S; `blis-registry` has no `roofline-b200.yaml` or
  `roofline-b300.yaml`. Scoring roofline on Blackwell would mean inventing `mfu` values. The
  kernel-against-AISimulate headline covers every chip and is unaffected.
- **These 104 points are not the headline's 108.** The subset is Hopper sweeps where all three arms
  scored every point. `gpt-oss-120b-h100-fp4-vllm-tp2 1k8k` was dropped for all arms because
  trained-physics completed no requests at concurrency 4 within the horizon, consistent with step
  times roughly 3x the kernel's starving the run.
- **The KV pool never binds on this subset.** 817,357 blocks of 16 tokens is about 13 million
  tokens, roughly 6,400 concurrent 2k-token sequences, against a stated `max_num_seqs` of 256. The
  resident batch is governed by the sequence cap and the token budget, not by memory. This holds
  for all four arms equally so the ranking is unaffected, but the subset does not exercise
  KV-pressure behaviour at all.
- **Activation precision is resolved, not assumed.** `gpt-oss-120b` (mxfp4) and `minimax-m2.5`
  (fp8) declare no `torch_dtype` or `dtype`, so `GetModelConfigFromHF` returns a zero
  `BytesPerParam` and both analytic arms refuse to construct. vLLM resolves exactly this case: with
  no dtype in the config it reads the safetensors weight metadata and otherwise falls back to the
  platform's first supported dtype, which on any device of capability 80 or above is bfloat16
  (`transformers_utils/model_arch_config_convertor.py` `get_torch_dtype`, `config/model.py`
  `_resolve_auto_dtype`, `platforms/cuda.py` `supported_dtypes`). The arms use 2 bytes because that
  is what the engine being modelled runs.

### A defect in BLIS's own KV sizing, found by this audit

Verifying that the shared KV number is not merely shared but CORRECT required deriving it a second
way, through `latency.CalculateKVBlocks`. The two paths disagree, model-specifically:

| deployment | kernel | legacy path | ratio |
|---|---|---|---|
| minimax-m2.5 h200 tp4 | 153,471 | 150,788 | 1.018 |
| minimax-m2.5 h200 tp8 | 419,483 | 417,795 | 1.004 |
| gpt-oss-120b h200 tp4 | 817,357 | 509,966 | **1.603** |
| gpt-oss-120b h200 tp8 | 1,733,706 | 1,429,655 | **1.213** |
| gpt-oss-120b h100 tp2 | 159,297 | **error** | — |

The legacy path sizes `gpt-oss-120b`'s weights at 217 GiB. The config's own arithmetic -- 36 layers
x 128 experts x 3 matrices of 2880x2880, plus attention and embeddings, 116.5B parameters -- gives
217.0 GiB at 2 bytes per parameter and 54.3 GiB at 0.5, identifying the assumed precision as bf16
on a model served at mxfp4. At tp=2 the over-count exceeds the device budget and KV sizing returns
an error rather than a number.

The kernel is right: 58.6 GiB of whole-model weights at tp4, 56.5 at tp2. Filed upstream as
**inference-sim#1852**, a sibling of #1563 which covers nvfp4 but not mxfp4 and does not state the
hard-failure consequence.

This does not affect any figure in this document: every arm draws its KV budget from the kernel's
memory methods, and no scoring path calls `CalculateKVBlocks`. The defect was visible only because
the audit derived the same quantity twice.
