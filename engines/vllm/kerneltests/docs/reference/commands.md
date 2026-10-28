# Commands

<p class="lede">Five commands live in blis-vllm's <code>kerneltests/cmd/</code>: they open a
kernel from a scenario's vLLM engine block, which blis-vllm decodes, so they live with the engine
model. Two score the kernel against published measurements; three print what the kernel
computes, for checking it or for use by another tool. Each runs with
<code>go run ./kerneltests/cmd/&lt;name&gt;</code> from blis-vllm's root, after
<code>kerneltests/scripts/fetch-testdata.sh</code>.</p>

Every command takes `-catalog` and `-registry` to read a catalog or registry other than the
fetched one; `BLIS_CATALOG` and `BLIS_REGISTRY` set the same defaults for all of them.

## Scoring

These two compare against measurement corpora that are not distributed with this repository,
because they are other publishers' data. They read them from `-data`, or from the directory
`BLIS_MEASUREMENTS` names, and say so when they are missing. [Accuracy](../research/accuracy.md)
describes what they measure.

`score`
:   Inter-token latency against published benchmark runs: the absolute error per point, per
    model and in aggregate, for every point a step-time model can be held to, with the points
    it leaves out and why. `-verbose` prints every point.

`shape`
:   The shape of the kernel's response to concurrency against NVIDIA's AISimulate accuracy
    snapshot: the ratio of step times across concurrency at a fixed deployment, each side
    normalized to its own lowest concurrency, beside AISimulate's own error on the same
    points. `-testdata` names the scenario directory, `testdata/aisimulate` by default.

## Inspecting

`worked-table`
:   A worked table: a fixed list of batch shapes, each priced on a deployment that
    SemiAnalysis's InferenceX benchmark measured, with its time on HBM, SM and the GPU
    interconnect, its Overlap estimate, its bottleneck, and the ratio of NoOverlap to
    Overlap. A document that quotes the table is then checked against this implementation
    rather than against a second one. `-format markdown` prints it as Markdown.

`bandprobe`
:   Prices one step per line of CSV on standard input and prints both estimates, so a fitter
    outside this repository can compare both with measured step times without reimplementing
    the composition. `-scenario` names the deployment. Each input line is
    `batch,prefill_tokens,context_tokens_per_sequence`.

`overlap-probe`
:   Prints Overlap and NoOverlap side by side over a concurrency sweep, each normalized to its
    value at the lowest concurrency, to show whether overlap changes the shape of the response
    or only its level. `-context` sets the context per request.
