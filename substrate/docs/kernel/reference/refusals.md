# What New refuses

<p class="lede">Every reason <code>New</code> returns an error instead of a kernel, grouped
by why. Each error message names the setting at fault and, where there is one, the setting to
change. The engine's reasons cite the vLLM v0.31.0 source in the code beside them.</p>

## The documents are malformed

- A scenario file has a key its document type does not define. `LoadBundle` decodes strictly,
  so `Open`, `OpenPool` and `OpenInputs` fail here.
- A document fails the contracts' field-level validation: a missing required field, a value
  out of range. Rule-level findings, which depend on the engine version, do not stop
  construction.
- The deployment has no pool at the requested index, or a tensor- or data-parallel width
  below one.
- No engine rules are given. Opening a scenario file (`Open`, `OpenPool`, `OpenInputs`) fails
  earlier, when `Repos.Rules` is unset or has no rules for the engine version it names.
- The engine rules cannot answer: a rules method panics while the kernel is built, as a
  forwarding wrapper around nothing does. The refusal names the panic. A rules method that
  panics later, inside a pricing call, is a defect in the caller's rules and panics there.

## The kernel implementations that run are malformed

- `Inputs.Operations` names a layer the graph (the target's, or the draft model's) does not have, a
  block-index scorer the layer does not run,
  or kernel implementations that differ from `Inputs.Implementations` when both are given.
- `Inputs.Implementations` lists a primitive with no implementation, or has an implementation with no library, a code hash that is not
  lower-case hexadecimal of at least 12 digits, a code hash with no variant, or two different code
  hashes for one library and variant. A pool runs one build of a kernel.
- A coefficient the cost model needs lists kernel implementations, the implementations of the
  operations that read it (the primitive it prices, narrowed to its kind where its name states
  one) are not all among them (or none is stated for that primitive), and no
  less trusted entry of its name covers the deployment. The error names each refused entry, the
  implementation it is valid for, and the primitive and what runs it.

## The engine would not start it

These are layouts vLLM v0.31.0 refuses at startup.

**Context parallelism**

- Prefill-context parallelism on a model with any layer whose attention is not latent, or
  with a recurrent layer.
- Prefill- and decode-context parallelism together on anything but DSA sparse attention.
- The `a2a` decode-context combine together with prefill-context parallelism.
- A decode-context combine backend the engine release does not accept.
- With decode-context parallelism on, a stated `cp_kv_cache_interleave_size` larger than the
  block size or not dividing it, unless a NIXL connector is configured. With a
  `MultiConnector`, whose child connectors a deployment cannot name, an interleave that does
  not fit, or one left unstated, is refused because the outcome cannot be known.

**Sparse attention (DeepSeek sparse attention, DSA: DeepSeek-V3.2, GLM-5)**

- A stated block size that is not a multiple of 64.
- A chip older than Hopper, where no sparse-attention backend runs.
- A cache data type no backend on the chip serves under the requested parallelism.
- Decode-context parallelism on FlashMLA's sparse backend with a 16-bit cache, with the
  `a2a` combine, with local and gathered head counts that pad to different kernel sizes, or,
  without prefill-context parallelism, with 32 or more attention heads per rank; or prefill-
  and decode-context parallelism there on a cache other than `fp8_ds_mla`.

**Cache data types**

- A cache data type vLLM does not accept, or a recurrent-state data type it does not accept.
- `fp8_inc`, which no CUDA attention backend serves.
- The packed `fp8_ds_mla` and `nvfp4_ds_mla` layouts on a model whose target stack holds any
  attention that is not sparse.
- Plain `nvfp4` on a model with latent attention.

**Block size and architecture**

- A block size a DeepSeek-V4-family model's sparse backends do not accept: DeepSeek-V4.1's
  run 64-token kernel blocks on SM90 and 128-token ones on SM100, and DeepSeek-V4's run
  256-token blocks on every part.
- A model graph that states facts by GPU generation, on a chip whose compute capability the
  catalog does not state or whose generation the graph does not describe.

**Experts**

- An expert-parallel width larger than the model's number of experts, which would leave some
  ranks with none.

**Attention heads**

- A tensor-parallel width that does not divide the KV heads of an attention whose graph says
  its projection cannot replicate a head across ranks (`kv_heads_not_replicated`), as Muse
  Spark's fused QKV and output-gate projection cannot: TP above 16 on its 16 KV heads.

## The prefill lane the engine reports

FlashInfer's trtllm-gen prefill is the one kernel with a
[pair measured on it](../../registry/reference/sets/cost-model-attention.md): on b200, b300,
gb200-nvl72 and gb300, `attention_prefill_*_trtllm_gen_kv_bf16` and `_kv_fp8` price the prefill of
the non-latent layers. The kernel does not work out from any engine setting whether an engine runs
that kernel, and it reads no engine setting for it: which kernel runs is the engine's to decide
and to report (design X2, SUB-6). The caller reports the kernel of each lowered operation
(`Inputs.Operations`, or `Inputs.Implementations` by primitive), and the lane prices the prefill
only when one pair lists exactly the implementation reported for `attention.prefill` (library,
variant and code hash, VAL-8). The pair that lists it prices; the cache dtype and the block size
do not choose it. Latent (MLA) layers keep the part's pooled pair, and decode is not priced from
the lane.

- The pair is chosen by the format of the cache the reported prefill operations read
  (`PrimitiveShape.Format`): the bf16 pair for a bf16 cache, the fp8 pair for an fp8 one. A
  kernel that serves both formats may be listed by both pairs and is not ambiguous. Prefill
  operations that read caches of two formats are refused, since the lane is priced from one pair
  for the layers of a kind. Operations that state no cache format leave the lane unpriced, with
  the assumption below. Reported only by primitive (`Inputs.Implementations`), the kernel cannot
  tell the format, and two pairs listing the reported implementation are refused.
- The pairs are read by the prefill of the non-latent layers alone. A pool that runs trtllm-gen
  for them and another kernel for the latent layers accepts a pair that lists the trtllm-gen
  kernel alone.
- With no pair listing it (the pairs list none, list another, or nothing was reported for the
  prefill primitive), the part's pooled pair prices the prefill and the assumption
  `trtllm_gen_prefill_lane` records why. A model with no non-latent layer has nothing for the
  lane to price.

The lookup binds each pair to the primitive its name prices. A pair valid for a decode kernel A
does not price a prefill that runs B, a pool that reports A for decode and nothing for prefill
does not admit it, and a prefill that runs two kernels is priced only by a pair that lists both.

The registry's pairs are `transferred` from vLLM 0.25.0 and list no implementation, because no
engine release publishes kernel hashes yet (`kernel_impls.json`; inference-sim/blis-registry#73).
So on the committed registry the lane is not priced and an engine's report changes no price. The
lane is dormant until an entry lists the hash of the prefill kernel and the engine reports it.

A setting that selects a kernel the registry has not measured is the engine model's to refuse:
it resolves the setting to the implementation that runs, reports it, and decides whether a
primitive whose reported implementation no entry lists may be priced from the pooled entry. The
kernel reads no setting and applies no engine default.

## The kernel cannot price it truthfully

- Pipeline parallelism beside data parallelism or context parallelism (`pcp` or `dcp`
  above one): the stage split prices one pipeline per pool. Pipeline parallelism on a model
  whose graph states its caches, whose sizes are per layer kind over the whole stack rather
  than per stage. A drafter that reads an auxiliary hidden state of a layer outside the
  target's stack.
- `dcp_q_replicate` where it would take effect: decode-context parallelism without
  prefill-context parallelism, on a latent-attention model. It trades a collective for a
  replicated projection the kernel cannot identify in the graph, so pricing only the saving
  would make it look free.
- A cache data type vLLM accepts whose page layout the kernel does not size: the `turboquant`
  and per-token-head formats.
- A quantization, CUDA graph mode or decode-context combine backend the kernel does not
  price.

## A coefficient is missing

- A required coefficient has no entry in the sets the scenario names, or none whose scope
  covers this chip, model and layout: the matrix-multiply efficiency for the served data
  type, the memory derate, the memory magnitudes, or a collective's floor, rate or transition
  rate at the width the layout needs.
- A collective's group width is not one the registry measured on the chip. The kernel does
  not borrow a narrower width's figures, because they would understate the wider group. Two
  cases are priced rather than refused: a group narrower than every measured width, at the
  narrowest, and a group wider than 16, the widest the registry's sweeps cover, at the
  chip's widest measured width.

Optional coefficients are not refused when missing; see
[Provenance, Evidence, Resolved](provenance.md#optional-coefficients-that-were-missing).

## In the source

The checks run in `New` and the functions it calls in
[`new.go`](https://github.com/inference-sim/blis-substrate/blob/main/kernel/new.go),
[`sparse_mla.go`](https://github.com/inference-sim/blis-substrate/blob/main/kernel/sparse_mla.go)
and [`internal/resolve`](https://github.com/inference-sim/blis-substrate/tree/main/kernel/internal/resolve),
and the tests in `refusals_test.go`, `sparse_mla_test.go` and `dcp_knobs_test.go` check most of
them.
