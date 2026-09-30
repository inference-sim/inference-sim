# Result: BLIS on blis-latency-kernel, against AISimulate

Worktree-scoped experiment. Latency comes exclusively from `blis-latency-kernel`; no
roofline or trained-physics model is constructed.

## Headline

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
