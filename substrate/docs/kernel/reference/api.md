# API

<p class="lede">The package's exported names, what each is for, and what this kernel does
for each method of the interface it implements. The full Go documentation is on
<a href="https://pkg.go.dev/github.com/inference-sim/blis-substrate/kernel">pkg.go.dev</a>;
the interface itself is defined and documented in
<a href="https://pkg.go.dev/github.com/inference-sim/blis-substrate/contracts/kernel">contracts/kernel</a>.</p>

```go
import latencykernel "github.com/inference-sim/blis-substrate/kernel"
```

## Building a kernel

| Name | What it does |
|---|---|
| `Open(scenario, repos)` | Loads a scenario file and everything it names, and builds a kernel for its first pool. |
| `OpenPool(scenario, repos, i)` | The same for pool `i` of a disaggregated deployment. |
| `OpenInputs(scenario, repos, i)` | Loads the documents without building a kernel, for a caller that needs them too. |
| `New(inputs)` | Builds a kernel from documents already in memory. `Open` and `OpenPool` end here. |
| `(*Kernel).RecordSelected(data)` | Records the coefficients the kernel's lookups returned in the data's selected-entry report. `OpenPool` calls it. |
| `Inputs` | The documents a kernel is built from: scenario, deployment, pool index, model graph, chip, fabric, storage devices, coefficient sets, engine rules, `Operations`, the pool's lowered operations with the kernel that executes each (`plan.PoolPlan.Operations()`), which a coefficient that lists implementations is matched against, for the operations that read it, and `Implementations`, the same by lowered primitive when a caller cannot say which operation runs which ([VAL-8](provenance.md#kernel-implementations)). |
| `Repos` | The scenario directory and the data (`*load.Data`) a scenario's names resolve against, and the engine's part, which the caller supplies: `Rules`, the engine rules for a release, `RulesKind`, the engine kind those rules are for (empty means `vllm`; `Open` refuses a scenario whose engine block names another kind), and `EngineSettings`, which decodes a pool's engine block. |
| `RulesFrom(lookup)` | Adapts a lookup returning a concrete rules type, such as blis-vllm/rules' `Lookup`, to `Repos.Rules`. |
| `LoadBundle(path)` | **Deprecated:** use `contracts.LoadBundle`, which this calls. Reads a scenario and its deployment from one file of two YAML documents. |

[Price a batch from Go](../guides/go.md) shows each in use.

## Kernel methods

`*Kernel` implements `kernel.Kernel` from `contracts/kernel`. Every method is a pure function of
its arguments and safe to call from many goroutines at once.

| Method | What this kernel does |
|---|---|
| `StepTime(batch)` | Prices one forward pass. [Pricing a step](../concepts/step.md). |
| `FixedBytes()` | Memory per rank before any request. [Memory](../concepts/memory.md). |
| `SequenceVariableBytes(tokens)` | One request's KV cache at a length, in whole pages, and for a graph that states its caches in whole blocks of the engine's shared pool. |
| `SequenceFixedBytes()` | What one request holds whatever its length: recurrent state where it does not grow with length, and the window rings and fixed state caches a graph states. |
| `TierTime(tier, dir, bytes, depth)` | A transfer to or from an offload tier: the larger of the tier's base latency and the time to move the bytes at the slower of the tier's rate and the host link, that rate shared among the transfers queued. The tier's rates and latency are the registry's `offload_tier_<tier>_read_rate`, `_write_rate` and `_base_latency` for the scenario's chip, not catalog figures, and they are in the set `cost-model-offload-tiers`. A tier the scenario's sets carry no terms for prices as the largest representable duration, and `New` refuses a deployment whose offload block names such a tier. |
| `PDTransferTime(tokens, from, to)` | Moving one request's KV cache between pools: the whole request's cache, which decode-context parallelism does not divide, over NVLink within a node or an NVLink rack, and over the fabric otherwise. A graph that states its caches moves only the caches the target owns, each window ring clipped to cdiv(window, block) + 1 pages, and fixed state caches whole. It is `PDTransferDemand(...).AloneTime()`. |
| `PDTransferDemand(tokens, from, to)` | The same handoff separated into payload bytes, a fixed floor and the links it crosses, each with its capacity and the number of the transfer's own flows (`Streams`) that cross it, so a simulator running several handoffs can share the links between them. Link IDs come from the placement, so every kernel of a deployment names a link alike. Where the scenario states [switch groups](../concepts/switch-groups.md), a handoff between groups also names the two groups' uplinks. |
| `AdmissionOverhead(tokens)` | Host time before a request can be scheduled: a fixed part per request and a part per prompt token. |
| `AdmissionDemand(tokens)` | `AdmissionOverhead` as `Service` (the per-token part, held on one server of the admission queue) and `Delay` (the per-request part, waited out holding none). A simulator that queues requests queues them for `Service` only. |
| `OutputTokenOverhead()` | Host time per emitted token. |
| `CompletionOverhead()` | Host time when a request finishes. |
| `Provenance()` | Every coefficient in scope and every assumption, each with its method. [Provenance](provenance.md#provenance). |
| `Resolved()` | The configuration after resolution, with every override. [Resolved](provenance.md#resolved). |
| `Deployment()` | The pool as the document stated it. A deep copy: writing to it changes nothing in the kernel. |

And ten that are not on the interface:

| Method | What it does |
|---|---|
| `Evidence()` | How many of the values in `Provenance` rest on evidence, of how many, and which were assumed. [Evidence](provenance.md#evidence). |
| `SequenceAdmissionBytes()` | What the engine reserves for one request whatever its length when it admits it: `SequenceFixedBytes` with each window ring at vLLM's admission cap, which grows with `max_num_batched_tokens`, rather than its steady decode hold. vLLM's reported KV capacity sizes a request with it. [Memory](../concepts/memory.md#what-each-request-holds-regardless-of-length). |
| `SequenceBoundedBytes(inFlight)` | A request's bounded state with `inFlight` tokens scheduled but not settled: each window ring at the pages that many in-flight tokens hold, in whole pool blocks. At 1 + num_spec_tokens it is `SequenceFixedBytes`; at the engine's max in-flight tokens it is `SequenceAdmissionBytes`. |
| `SequenceBoundedBytesAt(computed, inFlight)` | The same for a request that has computed only `computed` tokens: a ring holds pages only for tokens it has seen, as vLLM allocates them. Past the window it equals `SequenceBoundedBytes(inFlight)`. |
| `PreSnapshotBytes()` | What a rank holds before vLLM measures its `gpu_memory_utilization` budget: the communicator reservation and the engine workspace. `FixedBytes` leaves it out of its total. [Memory](../concepts/memory.md). |
| `CaptureTokens()` | The largest step, in tokens, a captured CUDA graph covers: the stated `max_cudagraph_capture_size` or `cudagraph_capture_sizes` as vLLM resolves them, or its default. A larger step runs with no graph. Zero when nothing is captured. |
| `GraphModes()` | The CUDA graph mode a uniform decode step runs under and the one every other step runs under (`FULL`, `PIECEWISE` or `NONE`), after the engine's overrides. |
| `DecodeContextParallelWidth()` | How many ranks shard the decode KV cache. |
| `PrefillContextParallelWidth()` | How many ranks split a prefill. |
| `StepTimeInto(batch, per)` | `StepTime`, writing the per-resource breakdown into a map the caller reuses instead of allocating one. |

The two context-parallel widths are methods here only until `kernel.Resolution` in
`contracts/kernel` carries them.

## Constants

| Name | Value |
|---|---|
| `EngineBehaviourVersion` | The vLLM release whose behavior the kernel prices: `0.31.0`. |
| `EngineBehaviourCommit` | That release's commit. |
| `KernelAssumptionSet` | The set name on `Provenance` entries the kernel supplied itself: `blis-latency-kernel`. |
