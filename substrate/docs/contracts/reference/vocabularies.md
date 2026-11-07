# Vocabularies

<p class="lede">Every enumerated key in a document accepts only the values listed here; an
unrecognized value is an error. The kernel's own enumerations are listed at the end. The
lists are generated from the Go constants, so they are the values the code accepts.</p>

These vocabularies are fixed by the schema and do not change with an engine release.
Values that do, such as backend names and cache dtypes, are listed with the
[rules pack](rules.md) for each release.

## Coefficients

### Units

The unit of a coefficient's `value`.

--8<-- "docs/contracts/generated/enum/vocab.Unit.md"

### Methods

How a coefficient's value was obtained. This is the field that keeps a measurement
distinguishable from an estimate, so it is never inferred. The methods are ranked from most to
least trusted (design 9.2), and a lookup that finds more than one entry uses the one with the
smallest rank number.

--8<-- "docs/contracts/generated/enum/vocab.Method.md"

`measured` (rank 1)
:   From a measurement of the thing it prices. The entry cites it.

`calibrated` (rank 2)
:   Back-solved through the cost model from end-to-end [measured runs](measurement.md), and
    checked against held-out ones. The entry cites its runs as sources of kind
    `measured_case` and names the held-out metrics in `validated`. It is not `derived`,
    which is a catalog [provenance](#provenances): a figure computed from other catalog
    facts.

`transferred` (rank 3)
:   A value measured in another scope and carried to this one by a stated rule. The entry names
    the scope in `transferred_from` and the rule in `transfer_rule`.

`literature` (rank 4)
:   From a published result that was not independently reproduced.

`vendor_spec` (rank 5)
:   A datasheet figure: nominal, not achieved.

`roofline` (rank 6)
:   Derived from a datasheet peak by the roofline form, before any measurement exists.

`assumed` (rank 7)
:   A placeholder, or an estimate reached by reasoning. A careful estimate is still assumed;
    the reasoning goes in `rationale`.

`not_charged`
:   Deliberately zero, so that "no cost" can be told apart from "cost unknown". It has no rank.

`measured`, `calibrated`, `literature` and `vendor_spec` count as evidenced. That is a
statement about the basis of a number, not its quality: a vendor figure is evidenced but nominal.

A price reports the least trusted effective rank among the coefficients it used. A lookup
outside the range an entry's fit covered ranks no better than `transferred`, and an entry whose
interval is unbounded or wider than half its value ranks as `assumed`.

### Rate bases

What a rate in bytes per microsecond measures. SUB-5 bounds a rate by the chip's datasheet
memory bandwidth only where the rate is a memory rate.

--8<-- "docs/contracts/generated/enum/vocab.RateBasis.md"

`memory`
:   A rate at which a kernel reads or writes device memory. It cannot exceed the datasheet
    memory bandwidth. This is the default.

`work`
:   A rate over a quantity of work counted in bytes, such as a record counted once per query
    and key pair. It may exceed the memory bandwidth, and the entry's `rationale` says why.

### Measurement times

What a measured number includes (design VAL-9).

--8<-- "docs/contracts/generated/enum/vocab.MeasurementTime.md"

`device`
:   Device time only.

`device_and_host`
:   Device time and the host issue time.

### Measurement executions

How the kernels ran when the number was measured.

--8<-- "docs/contracts/generated/enum/vocab.MeasurementExecution.md"

`captured`
:   From a captured graph.

`eager`
:   Launched one by one.

### Fit dimensions

The inputs of a priced step that a fit's data can have covered (design VAL-10). A lookup with
an input outside the recorded range is extrapolated.

--8<-- "docs/contracts/generated/enum/vocab.FitDim.md"

### Scope keys

The dimensions along which a coefficient's validity is bounded. A coefficient's `scope`
uses these as its keys.

--8<-- "docs/contracts/generated/enum/vocab.ScopeKey.md"

### Source kinds

--8<-- "docs/contracts/generated/enum/vocab.SourceKind.md"

### Source roles

--8<-- "docs/contracts/generated/enum/vocab.SourceRole.md"

## Catalog facts

### Provenances

Where a chip's or fabric's figures come from. `vendor_spec` is a datasheet figure;
`derived` is one computed from datasheet figures by a step the datasheet does not state.

--8<-- "docs/contracts/generated/enum/vocab.Provenance.md"

## Model graphs

### Primitives

A node's `op`.

--8<-- "docs/contracts/generated/enum/model.Op.md"

### Dtypes

A graph's `weight_dtype`, and a node's `weight_dtype` and `state_dtype`. Only formats that
change a byte count or a peak rate appear. `nvfp4`, `mxfp4` and `int4` are three different
4-bit formats, not spellings of one.

--8<-- "docs/contracts/generated/enum/model.DType.md"

### Attention kinds

An `Attention` node's `kind`. `mla` and `sparse_mla` store one latent vector per token,
which is why their `n_kv` is 1. `sparse_mla` is attention by a sparse kernel over an
explicit set of positions, bounded by a top-k (`index_topk`), a compressed stream
(`compress_ratio`), a sliding window (`window`), or a combination: DeepSeek-V4.1's
window-only layers run the same sparse FlashMLA kernel as its compressed ones.

--8<-- "docs/contracts/generated/enum/model.AttentionKind.md"

### Recurrent kinds

A `RecurrentUpdate` node's `recurrent_kind`.

--8<-- "docs/contracts/generated/enum/model.RecurrentKind.md"

### Emit conditions

A node's `emit`. `""`, the empty value, means the node always exists, and is what an
omitted `emit` reads as.

--8<-- "docs/contracts/generated/enum/model.EmitCondition.md"

`tensor_parallel`
:   Present when the tensor-parallel width exceeds 1.

`expert_parallel`
:   Present when the expert-parallel width exceeds 1.

`tensor_parallel_unless_sp_moe`
:   Present when the tensor-parallel width exceeds 1 and the engine has not made the MoE
    input sequence-parallel. Under sequence-parallel MoE the reduction is replaced by a
    reduce-scatter and all-gather pair.

`sequence_parallel`
:   Present when the model's own sequence parallelism is on, as the rules pack decides it
    for `global.architecture`. DeepSeek-V4.1 then shards the residual stream by token and
    all-gathers each attention input, then reduce-scatters its output.

`tensor_parallel_unless_sequence_parallel`
:   Present when the tensor-parallel width exceeds 1 and the model's own sequence
    parallelism is off: the all-reduces sequence parallelism replaces.

The last two belong on collectives. Work runs in every layout, including tensor
parallelism one, where neither condition holds, so a node stated twice, once under each,
would vanish there. Work is stated once, with
[`under_sequence_parallel`](#sequence-parallel-splits) for how the mode divides it. A
non-collective node carrying either condition draws a warning.

A model that decides its own sequence parallelism never consults the engine-wide rule
`tensor_parallel_unless_sp_moe` names, so its decoder-layer reductions, the MoE's
included, use `tensor_parallel_unless_sequence_parallel`. DeepSeek-V4.1 passes its own flag
to its MoE, and vLLM's FusedMoE then skips the final tensor-parallel all-reduce. The
0.31.0 rules pack reports the engine-wide condition on such a model as an error.

The condition holds on DeepSeek-V4's and V4.1's mega-MoE backend too. That backend requires
expert parallelism, so it runs without the model's sequence parallelism only at tensor
parallelism one or pipeline parallelism above one. There its routed experts combine over
the expert-parallel group with no tensor-parallel all-reduce, and its shared expert, which
fuses into the kernel only under sequence parallelism or tensor parallelism one, runs
tensor-parallel and all-reduces its own output. So the layer still reduces one
hidden-width tensor across tensor-parallel ranks, as on the default backend.

`tensor_parallel_without_expert_parallel`
:   Present when the tensor-parallel width exceeds 1 and expert parallelism is off: a
    reduction an MoE layer needs only while its experts are sharded by tensor parallelism.

`expert_parallel_without_dispatch`
:   Present when expert parallelism is on, the tensor-parallel width exceeds 1, and the MoE
    layer dispatches no tokens: every rank holds every token, because the expert-parallel
    group is the tensor-parallel group, so the ranks' partial expert outputs combine by an
    all-reduce rather than an all-to-all.

A model whose MoE reduces a different tensor in each of those two modes states one
collective under each. Muse Spark is the case: under tensor parallelism it all-reduces its
`top_k` per-route rows before its post-expert norm, which is not linear in a partial sum;
under expert parallelism only the combined output, at its latent width.

### Modalities

A graph's `modality`. An omitted `modality` means `text_only`.

--8<-- "docs/contracts/generated/enum/model.Modality.md"

### Tensor-parallel layouts

A node's `tp`. An omitted `tp` leaves the layout to the cost model's inference. A `GEMM`
accepts `column`, `row` and `replicated`; an `Elementwise` or `Attention` node accepts
`sharded` and `replicated`. No other primitive accepts one.

--8<-- "docs/contracts/generated/enum/model.TPLayout.md"

`column`
:   The output width `n` splits across ranks.

`row`
:   The input width `k` splits across ranks. The partial sums are reduced by a collective
    the graph states separately.

`sharded`
:   The work splits across ranks with the heads or columns it operates on.

`replicated`
:   Every rank holds the whole weight and does the whole node for every token it holds.

### Sequence-parallel splits

A node's `under_sequence_parallel`: how it divides when the model's own sequence
parallelism is on. An omitted value means the node runs as `tp` states, over every token
the step schedules. It needs `global.architecture`, and a collective cannot carry it.

--8<-- "docs/contracts/generated/enum/model.SequenceParallelSplit.md"

`token_sharded`
:   The node runs on its rank's shard of the step's tokens, `ceil(tokens / tp)` rows,
    because vLLM's `sp_shard` pads the count to a multiple of `tp`. A `GEMM`,
    `Elementwise` or `Attention` node then holds its whole weight, whatever `tp` states:
    a column or row split needs every rank to hold the same rows. A `GroupedGEMM`'s
    experts stay divided as expert parallelism decides, and a `Gather`'s table as its
    `shard` states.

`tp` alone can't say this. Under DeepSeek-V4.1's sequence parallelism the residual stream
is sharded by token, and only attention runs on the gathered stream, between its
all-gather and its reduce-scatter. mHC, before the gather, and the fused q/kv
A-projection, after it, are both `replicated`, yet the first sees `1/tp` of the tokens and
the second all of them. Engram's `wkv` is column-split without the mode and replicated with
it, and runs at tensor parallelism one too. It is one node, `tp: column` with
`under_sequence_parallel: token_sharded`.

### Cache kinds

A cache's `kind`, which decides how it grows.

--8<-- "docs/contracts/generated/enum/model.CacheKind.md"

`kv`
:   Keys and values, or a latent: one entry per `tokens_per_entry` context tokens.

`index`
:   A sparse-attention indexer's key cache. It grows like `kv`.

`window`
:   A sliding-window ring, bounded by `window` tokens per request.

`state`
:   Fixed per-request state, `entries` entries for the request's lifetime.

### Gather residency

A `Gather` node's `residency`, and the engine setting that can change it, its
`residency_control`. An omitted `residency` means `device`; an omitted
`residency_control` means no setting moves the table.

--8<-- "docs/contracts/generated/enum/model.Residency.md"

--8<-- "docs/contracts/generated/enum/model.ResidencyControl.md"

`engram_cpu_offload`
:   vLLM's Engram `cpu_offload`. A deployment sets it with
    `engine.engram_config.cpu_offload`. The graph states the residency the engine gives the
    table when the deployment leaves it alone, which is a release fact: under 0.31.0 it is
    `host`, and that pack's `engram-residency-default` rule checks it.

### Gather shards

A `Gather` node's `shard`. An omitted `shard` means `replicated`.

--8<-- "docs/contracts/generated/enum/model.GatherShard.md"

`rows`
:   The rows split evenly across ranks, and each token's row is read on the rank that holds
    it. vLLM's vocabulary embedding works this way.

`heads`
:   The table is `rows_per_token` sub-tables, one per hash head, split across S shards.
    Each shard holds ceil(`rows_per_token` / S) of them and reads one row from each for
    every token. S is the tensor-parallel width, or that times the node's data-parallel
    replicas when they split the table rather than share one host copy, which the
    deployment's `engram_config` decides. DeepSeek-V4.1's Engram works this way.

### Context work

An `Elementwise` node's `context.per`, `context.phase` and `context.decode_extent`.

--8<-- "docs/contracts/generated/enum/model.ContextPer.md"

`query_token`
:   Once per scheduled token, each over the context.

`request`
:   Once per request per step, over its context, however many of its tokens are scheduled.

--8<-- "docs/contracts/generated/enum/model.Phase.md"

An omitted `phase` means every step; `prefill` means only steps that schedule the
request's prompt tokens.

--8<-- "docs/contracts/generated/enum/model.Extent.md"

An omitted `decode_extent` means the request's context; `max_model_len` means the engine's
`max_model_len`, for a kernel launched over a buffer padded to it.

### Drafting modes

A speculator's `drafting`. An omitted `drafting` means autoregressive.

--8<-- "docs/contracts/generated/enum/model.Drafting.md"

`parallel_block`
:   The draft stack runs once per step over `num_spec_tokens` positions per request, and
    each position predicts one draft token. DeepSeek-V4.1's DSpark drafts this way.

## Deployments

### Pool roles

--8<-- "docs/contracts/generated/enum/deployment.Role.md"

### Data-parallel load balancing

A pool's `parallel.data_parallel_load_balancing`. `internal`, the default, serves one
endpoint and the engine balances requests across every rank. `hybrid`
(`--data-parallel-hybrid-lb`) serves one endpoint per node, balancing across that node's
ranks. `external` (`--data-parallel-external-lb`, or
`--data-parallel-multi-port-external-lb`, which launches one such server per local rank)
serves one endpoint per rank, and the router chooses the rank. `Pack.ResolveDPLoadBalancing`
returns the mode that runs: `hybrid` with one rank per node runs `external`, and with
every rank on one node runs `internal`.

--8<-- "docs/contracts/generated/enum/deployment.DPLoadBalancing.md"

## Traces

### Trace modes

A [trace](trace.md) header's `mode`: how the trace was produced. A `converted` trace
records no BLIS deployment.

--8<-- "docs/contracts/generated/enum/trace.Mode.md"

### Trace statuses

A trace record's `status`.

--8<-- "docs/contracts/generated/enum/trace.Status.md"

### Replay scenarios

A replay's `scenario`.

--8<-- "docs/contracts/generated/enum/workload.ReplayScenario.md"

### Corpus formats

A replay corpus's `format`.

--8<-- "docs/contracts/generated/enum/workload.CorpusFormat.md"

### Corpus encodings

A replay corpus's `encoding`.

--8<-- "docs/contracts/generated/enum/workload.CorpusEncoding.md"

### Cache-bust modes

A replay's `cache_bust`.

--8<-- "docs/contracts/generated/enum/workload.CacheBust.md"

### Warmup modes

A replay's `warmup.mode`. `synthesized` sends, for every stream mid-flight at its lane's
snapshot, its last earlier request with one output token, and starts profiling when all
have returned. `cache_pressure` runs an accelerated warmup for a stated duration.

--8<-- "docs/contracts/generated/enum/workload.WarmupMode.md"

### Time units

A trace header's `time_unit`. Several members name the same unit, because the tools that
write traces spell it differently.

--8<-- "docs/contracts/generated/enum/trace.TimeUnit.md"

## Acceptance

### Thinking modes

A profile's and a workload's `thinking`: whether the target model ran with its reasoning
mode on.

--8<-- "docs/contracts/generated/enum/acceptance.Thinking.md"

## Measured runs

The vocabularies of a [measured run](measurement.md) and its licence registry.

### Verdicts

--8<-- "docs/contracts/generated/enum/measurement.Verdict.md"

### Resolutions

--8<-- "docs/contracts/generated/enum/measurement.Resolution.md"

### Measurement source kinds

--8<-- "docs/contracts/generated/enum/measurement.SourceKind.md"

### Input roles

--8<-- "docs/contracts/generated/enum/measurement.InputRole.md"

### Extraction methods

--8<-- "docs/contracts/generated/enum/measurement.ExtractionMethod.md"

### Ramp strategies

--8<-- "docs/contracts/generated/enum/measurement.RampStrategy.md"

### Metric formulas

`interactivity` is 1000 over an inter-token-latency statistic, in tokens per second per
user. `output_throughput`, `total_throughput` and `request_throughput` are rates over a
window, with a denominator rather than a statistic.

--8<-- "docs/contracts/generated/enum/measurement.Formula.md"

### Statistics

A percentile is written `pNN` or `pNN.N`, strictly between 0 and 100, with no leading or
trailing zeros: `p50`, `p99`, `p99.9`.

--8<-- "docs/contracts/generated/enum/measurement.Statistic.md"

### Populations

--8<-- "docs/contracts/generated/enum/measurement.Population.md"

### Denominators

--8<-- "docs/contracts/generated/enum/measurement.Denominator.md"

### Clocks

--8<-- "docs/contracts/generated/enum/measurement.Clock.md"

### Start events

--8<-- "docs/contracts/generated/enum/measurement.StartEvent.md"

### Stop events

--8<-- "docs/contracts/generated/enum/measurement.StopEvent.md"

### Output length sources

--8<-- "docs/contracts/generated/enum/measurement.OutputLenSource.md"

### Percentile methods

--8<-- "docs/contracts/generated/enum/measurement.PercentileMethod.md"

### Metric units

--8<-- "docs/contracts/generated/enum/measurement.MetricUnit.md"

### Provenance row verdicts

--8<-- "docs/contracts/generated/enum/measurement.RowVerdict.md"

### Materiality

--8<-- "docs/contracts/generated/enum/measurement.Materiality.md"

### Difference kinds

--8<-- "docs/contracts/generated/enum/measurement.DiffKind.md"

### Licence statuses

--8<-- "docs/contracts/generated/enum/measurement.LicenseStatus.md"

### Permitted uses

--8<-- "docs/contracts/generated/enum/measurement.Use.md"

### Router kinds

--8<-- "docs/contracts/generated/enum/router.Kind.md"

## Kernel

### Resources

What a `StepEstimate` attributes cost to.

--8<-- "docs/contracts/generated/enum/kernel.Resource.md"

### Directions

A tier transfer's direction.

--8<-- "docs/contracts/generated/enum/kernel.Direction.md"

Source: [`vocab`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/vocab), and the
`spec/` and `kernel` packages.
