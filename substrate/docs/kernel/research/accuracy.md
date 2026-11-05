# Accuracy

<p class="lede">The cost model prices one engine step. The evidence for that price is a
comparison with measured forward passes, which time one step at a known batch and context
with no scheduler in between. It also decides which of the cost model's two estimates to use.
How accurate a whole simulation is, against measured end-to-end benchmarks, is computed in one
place: accuracy-dashboard's scoring harness, which the dashboard publishes.</p>

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
The registry records the comparison in
[Choosing the overlap band](../../registry/band-selection.md), with a script that reproduces it
from the output of `bandprobe` ([Commands](../reference/commands.md)).

## Where the error is known to come from

Some of it is recorded in the source. The routed experts' arithmetic is charged at more than
its share under tensor parallelism, a known over-charge that is kept because removing it makes
the measured accuracy worse: it offsets a term not yet identified. That and the other notes
are listed on [Known divergences](../reference/divergences.md).
