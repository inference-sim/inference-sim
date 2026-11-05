# Use the kernel from a simulator

<p class="lede">A discrete-event simulator decides which requests run in each step; the
kernel says how long each step takes. The simulator builds one kernel per pool before the run,
reads the pool's settings and memory from it once, and then calls it once per simulated
step.</p>

inference-sim adopts the kernel this way in pull request
[#1851](https://github.com/inference-sim/inference-sim/pull/1851), in the package
`sim/kernelmodel`. That pull request pins an earlier commit of this kernel, so a few of the
method names it calls differ from the ones here; what follows describes this version.

## Before the run

**One kernel per pool.** A colocated deployment has one pool and needs one kernel, from
`Open`. A disaggregated deployment has a prefill pool and a decode pool, each with its own
parallelism and engine settings, and needs one kernel for each, from `OpenPool`. Pricing both
pools with one kernel would charge the decode pool the prefill pool's parallelism, and nothing
would report the mistake.

**Configure the scheduler from the kernel.** `Deployment()` returns the pool as its document
states it, including the engine settings a scheduler sizes itself from: the block size, the
token budget per step (`max_num_batched_tokens`) and the limit on concurrent requests
(`max_num_seqs`). A setting the document leaves out is zero there, and the kernel prices
vLLM's default for it instead; `Provenance` then holds an entry for it from
`KernelAssumptionSet`, whose scope begins with the value used. A scheduler should take those
values too, so that it and the cost model describe the same engine. `Resolved()` returns the
parallel widths resolution settled; `Resolved().DataParallel()` is how many replicas schedule
their own requests.

**Size the KV cache.** The engine gives the KV cache whatever its memory budget leaves after
the fixed occupancy. Per rank, that is the GPU's memory times the deployment's
`gpu_memory_utilization`, less `FixedBytes().Total()`, divided into pages of
`SequenceVariableBytes(block_size)` bytes. The GPU's memory comes from the catalog's chip,
which `OpenInputs` returns beside the documents the kernel was built from.

## Each step

Build a `kernel.Batch` from the requests the scheduler formed:

| Field | From the simulator |
|---|---|
| `Scheduled` | The tokens the request runs this step, with any prefix-cache hits already removed |
| `Computed` | The tokens it had computed before this step |
| `PromptLen` | Its prompt length |
| `CachedTokens` | Not read by this kernel |
| `DecodeThreshold` | The engine's decode threshold; see [Price a batch from Go](go.md) |
| `SMBudget` | The SMs available this step, if fewer than the chip's; zero means all |
| `PadTokens` | Under data-parallel lockstep, the tokens this rank pads its step with to the group's largest count; zero means none. Pad tokens run the GEMMs, the MoE dispatch and the experts, but no attention and no LM head, and the capture ceiling is met at the padded count. A rank with nothing scheduled sends an empty `Reqs` and its pad. Do not add pad tokens as requests, which would charge them attention and logits |

Call `StepTime` and advance the clock by `NoOverlap`. inference-sim converts it to its
microsecond tick and never lets a step take less than one tick.

The kernel's other methods price the work around a step:

| Method | When |
|---|---|
| `AdmissionOverhead(promptTokens)` | When a request arrives, before it can be scheduled |
| `OutputTokenOverhead()` | For each token a request emits |
| `CompletionOverhead()` | When a request finishes |
| `PDTransferTime(tokens, from, to)` | When a request's KV cache moves from the prefill pool to the decode pool |
| `TierTime(tier, dir, bytes, depth)` | When KV blocks move to or from an offload tier |

## What stays with the simulator

Queueing, admission order, preemption and batch formation are the simulator's. They depend on
when requests arrive, and the kernel is a pure function of a batch. The kernel prices any
batch it is given and does not check that an engine would have formed it, so a simulator
must respect the limits it read from the kernel itself.
