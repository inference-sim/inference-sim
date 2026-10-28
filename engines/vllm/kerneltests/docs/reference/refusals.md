# What New refuses

<p class="lede">Every reason <code>New</code> returns an error instead of a kernel, grouped
by why. Each error message names the setting at fault and, where there is one, the setting to
change. The engine's reasons cite the vLLM v0.31.0 source in the code beside them.</p>

## The documents are malformed

- A scenario file has a key its document type does not define. `LoadBundle` decodes strictly,
  so `Open`, `OpenPool` and `OpenInputs` fail here.
- A document fails blis-schemas' field-level validation: a missing required field, a value
  out of range. Rule-level findings, which depend on the engine version, do not stop
  construction.
- The deployment has no pool at the requested index, or a tensor- or data-parallel width
  below one.
- No engine rules are given. Opening a scenario file (`Open`, `OpenPool`, `OpenInputs`) fails
  earlier, when `Repos.Rules` is unset or has no rules for the engine version it names.
- The engine rules cannot answer: a rules method panics while the kernel is built, as a
  forwarding wrapper around nothing does. The refusal names the panic. A rules method that
  panics later, inside a pricing call, is a defect in the caller's rules and panics there.

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

## The kernel cannot price it truthfully

- Pipeline parallelism wider than one: the kernel has no stage split or transfer between
  stages.
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
[`new.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/new.go),
[`sparse_mla.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/sparse_mla.go)
and [`internal/resolve`](https://github.com/inference-sim/blis-latency-kernel/tree/main/internal/resolve),
and the tests in `refusals_test.go`, `sparse_mla_test.go` and `dcp_knobs_test.go` check most of
them.
