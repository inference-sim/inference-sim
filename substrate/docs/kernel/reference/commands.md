# Commands

<p class="lede">Three commands live in blis-vllm's <code>kerneltests/cmd/</code>. They open a
kernel from a scenario's vLLM engine block, which blis-vllm decodes, so they live with the engine
model. Each prints what the kernel computes, for checking it or for use by another tool. Each
runs with <code>go run ./kerneltests/cmd/&lt;name&gt;</code> from blis-vllm's root.</p>

Every command takes `-catalog` and `-registry`, the catalog and registry directories to read;
`BLIS_CATALOG` and `BLIS_REGISTRY` set the same defaults for all of them.

None of them compares the kernel with measured end-to-end benchmarks. That comparison is
accuracy-dashboard's scoring harness, the one end-to-end scorer in BLIS.

## Inspecting

`worked-table`
:   A worked table: a fixed list of batch shapes, each priced on a deployment that
    SemiAnalysis's InferenceX benchmark measured, with its time on HBM, SM and the GPU
    interconnect, its Overlap estimate, its bottleneck, and the ratio of NoOverlap to
    Overlap. A document that quotes the table is then checked against this implementation
    rather than against a second one. `-format markdown` prints it as Markdown.

`bandprobe`
:   Prices one step per line of CSV on standard input and prints both estimates, so a fitter
    can compare both with measured step times without reimplementing
    the composition. `-scenario` names the deployment. Each input line is
    `batch,prefill_tokens,context_tokens_per_sequence`.

`overlap-probe`
:   Prints Overlap and NoOverlap side by side over a concurrency sweep, each normalized to its
    value at the lowest concurrency, to show whether overlap changes the shape of the response
    or only its level. `-context` sets the context per request.
