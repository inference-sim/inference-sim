# Kernel interface

<p class="lede"><code>kernel.Kernel</code> is the interface a cost model implements and a
simulator calls. One value prices one pool of one deployment: how much memory a rank
holds, how long a forward pass over a batch takes, and what transfers and host work cost.
The cost model in <code>kernel/</code> is the implementation in use.</p>

Every method is a pure function of its arguments and the configuration the kernel was
constructed from: the same inputs give the same outputs, nothing is mutated, and calls are
safe from several goroutines at once. There is no `Init` method; a kernel comes from its
implementation's constructor, already resolved. Why the interface is drawn this way is the
subject of [Where the kernel stops](../design/kernel-boundary.md).

## Methods

| Method | Returns |
|---|---|
| **Memory** | |
| `FixedBytes() MemoryBreakdown` | Per-rank memory that does not depend on the requests: weights, activation peak, captured CUDA graphs, redundant experts, communication buffers. |
| `SequenceFixedBytes() int64` | Per-sequence memory that does not grow with length, such as recurrent state. Zero for a model with attention only. |
| `SequenceVariableBytes(tokens int) int64` | Per-sequence memory for a given token count, quantized to the engine's page size. |
| **Step time** | |
| `StepTime(Batch) StepEstimate` | The time of one forward pass over a batch. |
| **Transfers** | |
| `TierTime(tier, dir, bytes, queueDepth) time.Duration` | One transfer to or from an offload tier. |
| `PDTransferTime(tokens, from, to Placement) time.Duration` | Moving one request's KV between pools. |
| `PDTransferDemand(tokens, from, to Placement) TransferDemand` | The same handoff as payload, floor and links, for a caller arbitrating concurrent transfers. |
| **Host work** | |
| `AdmissionOverhead(promptTokens int) time.Duration` | CPU work before a request can be scheduled, such as tokenization. Excludes queueing. |
| `OutputTokenOverhead() time.Duration` | Host work per emitted token, such as detokenization and streaming. |
| `CompletionOverhead() time.Duration` | Fixed work when a request finishes. Counts toward end-to-end latency, not time to first token. |
| **Provenance** | |
| `Provenance() []CoefficientOrigin` | Every coefficient the kernel used, with its set, method and scope. |
| `Resolved() Resolution` | What the kernel resolved, including requests it overrode. |
| `Deployment() deployment.Pool` | The pool it prices, as the document stated it. |

`Deployment()` returns the request and `Resolved()` the resolution. For the parallel
widths or the all-reduce backend, read `Resolved()`. For settings that resolution does not
change, such as `max_num_seqs` and `block_size`, read the pool. The returned pool is a
shallow copy; treat it as read-only.

## Step estimates

A step is priced as a sequence of stages, one per layer, since each layer needs the
previous layer's output. Within a stage, work on different resources can overlap.

`StepEstimate` reports two bounds and a diagnosis:

`Overlap`
:   Perfect overlap within each stage: the slowest resource in each stage, summed over the
    stages.

`NoOverlap`
:   No overlap anywhere: every resource's work, summed.

`Bottleneck`
:   The resource that did the most work over the step, which answers "what should be
    relieved". It is not the resource that set `Overlap`; since `Overlap` sums per-stage
    maxima, no single resource need account for it.

`PerResource`
:   Each resource's total, for a caller that needs to see why.

`Stages`
:   For a pipeline-parallel deployment (`pp` > 1), the step split by stage in pipeline order,
    each bracketed the same way and including the send of its activations to the next stage.
    The stages' `Overlap` figures sum exactly to `Overlap`, and their `NoOverlap` figures to
    `NoOverlap`; a single-stage step reports none. A simulator needs the split because two
    batches in flight occupy two stages at once: a new batch can enter every slowest-stage
    time, while each batch's output takes the whole sum. `kerneltest.CheckStages` checks the
    law.

A configuration whose conclusion changes between the two bounds needs a measurement rather
than a prediction.

The resources are:

--8<-- "docs/contracts/generated/enum/kernel.Resource.md"

A collective appears under `sm` when the resolved backend is an SM-consuming kernel and
under a link otherwise. The resolved backend decides this, not the requested one.

## Batches

A `Batch` is what the simulator supplies for one step.

| Field | Meaning |
|---|---|
| `Reqs []ReqShape` | One entry per request in the batch. |
| `DecodeThreshold int` | Engine configuration that selects the attention kernel for each request. |
| `SMBudget int` | SMs available this step: the chip's count less any held by a concurrent offload fetch. |
| `PadTokens int` | Tokens this data-parallel rank pads its step with under vLLM's DP lockstep: every rank runs the group's largest token count, and a rank with nothing scheduled runs a dummy batch of pad alone. Pad tokens run the GEMMs, the MoE dispatch and combine and the expert paths, but never the logits, the LM head, the vocabulary all-gather or sampling. They are not requests. 0 means none; never negative. |

Each `ReqShape` carries the counts an engine's own scheduler reads:

| Field | Meaning |
|---|---|
| `Scheduled` | Tokens scheduled this step. Under speculative decoding a decoding request schedules the draft length plus one, so a count above 1 does not imply prefill. |
| `Computed` | Tokens already computed, as of batch formation. |
| `PromptLen` | The prompt length. With `Computed`, it tells the last chunk of a prefill split across steps apart from a decode. |
| `CachedTokens` | The prefix-cache hit, already subtracted from `Scheduled`. A count, not a rate. |

`Batch.Tokens()` sums `Scheduled`, and `Batch.PaddedTokens()` adds `PadTokens`: the count the layers run. `Batch.Validate()` refuses a negative count. `Batch.UniformDecode(width)` reports whether every
request schedules exactly `width` tokens. Some engine behavior, such as when dual-batch
overlap engages, uses one threshold for such a batch and another for any other.

## Resolution

A deployment states requests. The kernel's constructor decides what runs, and
`Resolution` reports it.

| Field | Meaning |
|---|---|
| `TensorParallelWidth`, `DataParallelWidth`, `ExpertParallelWidth` | The resolved layout. The expert width is `tp × pcp × dp` with expert parallelism on, and 1 with it off. |
| `AllReduceBackend` | The all-reduce that will run, which may differ from the request. |
| `AsyncScheduling` | The resolved value of the tri-state request. |
| `CascadeAttention` | Whether cascade attention is available after every condition is applied. |
| `SequenceParallelMoE` | Whether the MoE input is made sequence-parallel, replacing an all-reduce with a reduce-scatter and all-gather. |
| `Overrides []Override` | Each request the layout could not honor: the field, what was requested, what was resolved, and why. |

A width of zero means unset, not "no parallelism". `TensorParallel()`, `DataParallel()`
and `ExpertParallel()` return the widths with a floor of one, for a reader that may see a
zero value.

## Transfer demand

`PDTransferTime` prices a handoff as if it ran alone. A simulator running several at once
needs to know which links they share, and which part of each price sharing stretches.
`PDTransferDemand` answers both:

| Field | Meaning |
|---|---|
| `Bytes int64` | The payload. |
| `Floor time.Duration` | Time the transfer takes however much bandwidth it gets. Sharing a link does not lengthen it. |
| `Links []LinkUse` | Every link the payload crosses, each named once: a `LinkID` and the rate the link offers to a transfer alone on it. |

A `LinkID` is opaque. Two demands naming the same ID share that link while both are in
flight, and IDs are comparable across the kernels of one deployment.

The law is part of the contract: alone, a demand costs exactly what `PDTransferTime`
charges, `max(Floor, Bytes / slowest link rate)`, or `Floor` with no links.
`TransferDemand.AloneTime()` is that definition, including its conversion to a duration,
so an implementation that computes `PDTransferTime` from its own demand satisfies the law
by construction.

The kernel states the links; it never sees other transfers. How a shared link's bandwidth
is divided is the simulator's to decide, and the schema does not prescribe it.

Package `kernel/kerneltest` checks the law against any implementation, the way
`testing/fstest` checks a file system:

```go
func TestPDTransferLaw(t *testing.T) {
    k := newKernel(t) // the implementation under test
    kerneltest.TestPDTransfer(t, k, kerneltest.PDCases())
}
```

`PDCases` covers a same-node, a same-rack and a cross-rack handoff at token counts from 0 to
131,072. `CheckPDTransfer` checks one case and returns an error, for a caller with cases
of its own; `TestPDTransfer` fails when given no cases, since it would then check nothing.
`CheckLinksAgree` checks that two kernels of one deployment name the same links at the same
rates for one handoff, and that both demands are well formed. `CheckPDTransfer` requires
a well-formed demand (`TransferDemand.Check`, which also refuses a payload that crosses no
link, since an arbiter would then have nothing to share), the same demand from two calls,
and an exact match with `PDTransferTime`. `TestPDTransfer` also requires that more tokens
never move fewer bytes between one pair of placements; it does not require the floor to
stay fixed, since per-transfer overhead can grow with the pages a transfer moves.

Adding a method to `Kernel` breaks every other implementation, fakes included, until it
adds the method.

## Other types

| Type | Meaning |
|---|---|
| `MemoryBreakdown` | `Weights`, `ActivationPeak`, `CUDAGraph`, `EPLBRedundant`, `CommBuffers`, in bytes; `Total()` sums them. |
| `Direction` | `to_tier` (eviction) or `from_tier` (fetch). Priced separately because flash reads and writes differ. |
| `Placement` | A rank's `Node`, `Rack` and `Pool` (the pool's role). Comparing two placements chooses the link a transfer uses. |
| `CoefficientOrigin` | One row of the provenance trail: entry `Name`, `Set`, `Method`, `Scope`. |

Source: [`kernel`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/kernel).
