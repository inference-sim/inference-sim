# Result: BLIS on blis-latency-kernel, against AISimulate

Worktree-scoped experiment. Latency comes exclusively from `blis-latency-kernel`; no
roofline or trained-physics model is constructed.

## Headline

On the 238 points measured under vLLM -- the engine BLIS models:

| Model | n | MAPE | median | worst |
|---|---|---|---|---|
| BLIS + blis-latency-kernel | 238 | **11.26%** | 7.82% | 66.14% |
| AISimulate, same points | 238 | **7.15%** | 4.19% | 75.12% |

Excluding the four sweeps whose MEASURED curve is non-monotone in concurrency:

| Model | n | MAPE | median |
|---|---|---|---|
| BLIS + blis-latency-kernel | 213 | **10.72%** | 7.51% |
| AISimulate, same points | 213 | **6.68%** | 4.09% |

Both figures are reported because the exclusion criterion, while defensible, moves the
comparison. It is a property of the measurement alone -- time per output token cannot fall
when a fixed deployment is given more concurrent work -- it never reads a prediction, and it
IMPROVES AISimulate's score (7.15% to 6.68%) as well as ours, so it raises the bar rather
than lowering it.

**AISimulate is still ahead.** The goal was to beat 9.41%; this does not.

What did improve, and it is the thesis working: the kernel alone scores 13.67% on the full
447 points with resident batch assumed equal to client concurrency. Letting BLIS's scheduler
decide the resident batch moves the vLLM subset to 11.38%. The comparable AISimulate figure
on that same subset is 7.15%, not 9.41% -- the 9.41% is over all 447 points, and the vLLM
subset is one AISimulate happens to fit better than average.

So the honest statement is: the scheduler closed part of the gap, and a gap remains.

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

**The limit: the snapshot publishes no engine settings.** Its topology records carry
`framework`, `precision`, `serving`, `spec_method` and the five parallelism widths. They do
NOT carry `max_num_seqs`, `max_num_batched_tokens`, `block_size`,
`gpu_memory_utilization`, `cudagraph_mode`, `scheduling_policy`, or the chunked-prefill and
prefix-caching flags.

The scenario files generated for the earlier step-time comparison say so, and justify
defaulting them on the grounds that "the comparison scores a ratio across concurrency at a
fixed deployment, so a constant cancels". That reasoning is correct for a step-time ratio and
WRONG once a scheduler is in the loop: `max_num_seqs` and the token budget bound the resident
batch, and the resident batch is what sets the shape of the curve. A default that cancelled
before is load-bearing now.

So part of the residual is attributable to engine settings this project has to guess. The
honest consequence: this comparison cannot be driven to AISimulate's figure by any amount of
correct modelling, because AISimulate replays the engine that produced the measurements and
therefore knows the settings, while BLIS is given a guess. Closing the remaining gap by
adjusting those guesses until the 238 points improve would be fitting to the evaluation set.

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
