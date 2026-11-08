# Accuracy

<p class="lede">End-to-end accuracy, a simulation compared with a measured benchmark, is
computed by accuracy-dashboard's scoring harness alone, which publishes every figure. This page
keeps the one comparison that is not end to end: measured forward passes, which decide which of
the kernel's two estimates to use.</p>

The end-to-end commands that used to live here (cmd/shape and cmd/score) are retired, with
the figures they printed: a second scorer would sooner or later disagree with the dashboard
about what a figure means (design §2.2, COV-1).

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
[band selection](https://github.com/inference-sim/blis-substrate/blob/main/docs/registry/band-selection.md),
with a script that reproduces it from the output of `kerneltests/cmd/bandprobe`.

## Where the error is known to come from

Some of it is recorded in the source. The routed experts' arithmetic is charged at more than
its share under tensor parallelism, a known over-charge that is kept because removing it makes
the score worse: it offsets a term not yet identified. That and the other notes are listed on
[Known divergences](../reference/divergences.md).
