# Accuracy

<p class="lede">The kernel is scored two ways: by the shape of its response to load, against
NVIDIA's AISimulate accuracy snapshot, and by its absolute inter-token latency, against
published benchmark reports. On the first it is somewhat behind AISimulate on the same
points; on the second, the corpus is small and the kernel is too fast throughout. Both say
the same thing about the kernel's direction: its step time grows more slowly with load than
the measured one. A third comparison, against measured forward passes, decides which of its
two estimates to use.</p>

The first two sets of figures were last computed for version 0.1 of the kernel, by
`cmd/shape` and `cmd/score`. Both compare against measurement corpora extracted from the
publishers' own releases, which are not redistributed here
([Commands](../reference/commands.md)).

## Whole-step time

NVIDIA's FPM dataset, published alongside AISimulate, times single forward passes at a known
batch and context, with no scheduler in between. That is the quantity the kernel prices, so
it is the right evidence for choosing between Overlap and NoOverlap. Over 219 points, on two
models, two chips and five parallel layouts:

| Estimate | Mean signed error | Mean absolute error |
|---|---|---|
| Overlap | −13.5% | 14.7% |
| NoOverlap | −3.4% | 10.2% |

NoOverlap is the closer of the two at 158 of the 219 points, and at four of the five layouts.
The exception, MiniMax-M2.7 at pure tensor-parallel width 2, is the one layout where a single
resource holds most of the step. Both
signed errors are negative: on this data the measured step is slower than either estimate.
blis-registry records the comparison in
[`docs/band-selection.md`](https://github.com/inference-sim/blis-registry/blob/main/docs/band-selection.md),
with a script that reproduces it from the output of `cmd/bandprobe`.

## Shape: how step time grows with load

NVIDIA's AISimulate publishes an accuracy snapshot of vLLM, SGLang and TensorRT-LLM
deployments measured at several client concurrencies. It does not disclose absolute
latencies: every measurement is normalized to its sweep's lowest concurrency. So the snapshot
can score how step time *grows* as a deployment is loaded, but not its level.

`cmd/shape` computes the same ratio from the kernel's step times, at a fixed deployment, and
compares. Over 204 sweeps and 1,066 points:

| Model | Mean absolute error | Median | Worst |
|---|---|---|---|
| This kernel | 11.3% | 5.7% | 408% |
| AISimulate, same points | 8.2% | 4.1% | 432% |

AISimulate reports 10.1% over the whole snapshot of 1,135 points for the same quantity.

Of the 862 points other than each sweep's first, where both curves are 1 by construction, the
kernel's curve rises more slowly than the measured one at 619 and faster at 243. One quantity
the comparison has to assume is the batch: the snapshot states the client concurrency but not
how many requests the engine actually held at once, and the comparison takes the two to be
equal. Where the engine held more, the true curve is steeper than the one priced; where it
held fewer, flatter.

## Level: inter-token latency

Published benchmark reports give absolute inter-token latency at several concurrencies. At
steady-state decode one step produces one token per request, so the interval between tokens
is one step plus the host's per-token work: nearly a direct reading of a step time.
`cmd/score` compares against every point a step-time model can be held to:

| Deployment | Points | Mean absolute error | Median | Worst |
|---|---|---|---|---|
| Kimi-K3 on H100 | 3 | 46.2% | 39.2% | 67.1% |
| Nemotron-3-Ultra on H100 | 3 | 18.5% | 15.7% | 36.6% |
| All in scope | 6 | 32.4% | 36.6% | 67.1% |

Six points on two models is too few to generalize from, and they are reported as they are.
The kernel predicts too short a step at every one, by more as concurrency grows: from 3.3% to
36.6% short on Nemotron-3-Ultra between concurrency 8 and 32, and from 32.3% to 67.1% on
Kimi-K3. That agrees in direction with the shape comparison. The Nemotron-3-Ultra report
states how many requests the engine held, and the comparison uses it, so there the shortfall
is not the batch: something that grows with the batch is missing from the kernel or priced too
low. The Kimi-K3 report gives its effective concurrency instead, which the comparison takes
as the number of requests held.

## What these comparisons cannot show

The shape comparison is blind to a constant factor: a kernel uniformly too fast by half scores
perfectly on it. It is also nearly blind to any term that scales in proportion to the batch,
since such a term cancels in the ratio. The level comparison sees both, but on very few
points. Neither measures the step time the kernel computes directly; both go through the
serving engine's scheduling, which the kernel does not model.

## Where the error is known to come from

Some of it is recorded in the source. The routed experts' arithmetic is charged at more than
its share under tensor parallelism, a known over-charge that is kept because removing it makes
the score worse: it offsets a term not yet identified. That and the other notes are listed on
[Known divergences](../reference/divergences.md).
