# How the laws are tested

<p class="lede">A cost model can be wrong in ways that still produce plausible numbers, so
its tests are written to catch exactly that. Few of them compare against a fixed number.
Most compare against the engine's own source, against an exact relation the answer must
satisfy, or against a deployment that differs in one setting.</p>

## Against the engine's source

Where the kernel reproduces an exact rule of vLLM, a test transcribes that rule from vLLM
v0.31.0 and checks the kernel against the transcription, over a grid of inputs rather than at
one point. The rules checked this way include how ranks are grouped for each kind of
parallelism, the size of a KV cache page, how many tokens of a context each decode-context
rank holds, and how a prefill is split across prefill-context ranks.

The transcription is the oracle, so it is kept as close to vLLM's code as Go allows, and
cites the lines it follows.

## Against exact relations

Where no closed form exists to compare against, a test changes one input and checks how the
output must respond. Three kinds recur:

**Inflate one coefficient.** Multiply the fixed cost of one collective, on one group size, by
ten, and a step's time rises exactly when the step launches that collective, by exactly nine
times its fixed cost for each launch. This asks which collectives a layout runs without
reading anything inside the kernel, and it is how the decode-context combine and the
prefill-context gathers are tested.

**Laws that hold exactly.** Growing a decode's context must add exactly the bytes the KV cache
grew by, divided by the attention rate, to the memory time. Three expert-parallel widths of
the same model must satisfy an exact linear relation in the number of experts the fullest
rank holds. Tripling every memory coefficient must move the fixed memory and no step time.

**Properties over random inputs.** The KV page arithmetic is checked against vLLM's over
20,000 random model geometries, and the efficiency ramp is checked over a sweep of sizes to be
monotone, bounded and exact at its half point.

## Against a deployment that starts

Most refusals are tested beside a deployment that differs only in the setting at fault, and
does start. A test that checked only for an error would pass when the kernel refused for some
other reason.

## Breaking a law on purpose

A test that has never failed has not been shown to test anything. The practice here is to
check a new law by breaking it on purpose (deleting a term, changing a comparison, swapping a
divisor) and confirming that a test fails, and to name in the commit the changes it was
checked against.

## The documentation

The Go on these pages is taken from `example_test.go`, whose examples Go runs and checks, and
a test checks the output shown beside each example against that example's. The figure on
[Pricing a step](../concepts/step.md), the list of [known divergences](../reference/divergences.md)
and the [pins](../reference/sources.md) are generated from the code, and a test fails when a
committed copy differs from what the generator would write. The table of assumptions on
[Provenance, Evidence, Resolved](../reference/provenance.md) is checked against the
assumptions the code can record.
