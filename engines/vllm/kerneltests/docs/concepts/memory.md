# Memory

<p class="lede">A GPU's memory holds two kinds of thing. Some is taken before any request
arrives: the model's weights, scratch space, captured CUDA graphs, communication buffers.
The rest is the KV cache, which grows with every request's context. The kernel reports the
first as one breakdown per GPU, and the second as a function of a request's length, so a
caller can work out how many requests fit.</p>

Everything here is per rank. A *rank* is one GPU's process in the deployment; a *pool* is the
set of GPUs that run one engine configuration, and every rank in it holds about the same.

## What is held before any request

`FixedBytes` returns five figures, kept separate because a caller deciding whether to add GPUs
or to raise the memory it gives the engine needs to know which one dominates.

Weights
:   Each rank holds its share of the model's parameters. The linear layers are at the width
    they are served in; the embedding table stays at the checkpoint's width, because the
    engine does not quantize it. Dense layers are divided across the tensor-parallel group. Experts are divided by expert
    parallelism into whole experts per rank; without it, every rank holds a slice of every
    expert. When the experts do not divide evenly, some ranks hold one more than the rest,
    and the kernel reports the fuller rank, because its memory runs out first.

Activations
:   Scratch space for the largest step the engine schedules:

    number of buffers × token budget per step × activation width × 2 bytes,

    with a floor beneath it. The number of buffers comes from NVIDIA's memory model, which
    states it separately for dense and mixture-of-experts models and by tensor-parallel
    width. The token budget is the engine's `max_num_batched_tokens`. Activations stay
    16-bit even when the weights are served in fp8 or fp4, because quantized matrix
    multiplies return their results at the input's width.

    The engine's profile run also samples once from dummy logits, one row per request,
    up to `max_num_seqs`. The peak is the larger of that pass and the forward. The pass
    holds the logits in the model dtype and an fp32 copy, over the whole vocabulary.

    A model whose own code allocates buffers at the token budget adds them here, because
    the registry's buffer count is generic and knows no model's buffers. vLLM's memory
    profiler counts everything allocated before the KV cache, so these buffers take room
    from the KV cache. DeepSeek-V4.1's profile run reserves the prefill gather workspaces at
    the context bound, and the engine keeps one per lane at the larger request: the
    attention's (4 × (`max_model_len` / compress ratio + window + token budget) × 512 BF16
    values, plus the top-k indices) or the indexer's (40 × `max_model_len` / compress ratio
    entries of 132 bytes), with DSpark's own in a second lane. The indexer also stands in
    512 MiB for its logits. Beside those sit the indexers' top-k buffer (one more for
    DSpark), the candidate-block buffer and, while speculating, DSpark's hidden-state buffer
    of token budget × 4 × hidden size. Model init also builds two rotary caches, plain and
    YaRN, over the model's whole 1M-position window, 256 MiB each whatever the context
    bound. At a 1M-token context and a 32,768-token budget all of these come to 7.9 GiB a
    rank, against 2.1 GiB at 131,072 and 8,192.

CUDA graphs
:   The memory a captured graph holds, for the capture mode the deployment runs in. vLLM
    subtracts its own estimate of it from the KV budget. That estimate grows with the
    largest graph in each family: the capture ceiling for the mixed graphs, and for the
    uniform-decode graphs the largest size that `max_num_seqs` requests of 1 + K tokens
    fill. Where the registry states a floor and a per-token cost for each family, the
    kernel charges floor + cost × ceiling for each one. Otherwise it charges the mode's
    flat figure.

EPLB replicas
:   The redundant copies of popular experts that expert-parallel load balancing (EPLB)
    adds, sized from the experts' weights. They are moved out of the weights figure rather
    than added to it, so the total counts them once. Zero when it is off.

Communication buffers
:   Zero. The collective library's communicator reservation and the engine's own workspace
    (CUDA context, cuBLAS, allocator slack) are reported by `PreSnapshotBytes` instead,
    outside the total. vLLM starts the device and the collective library before it measures
    the memory its `gpu_memory_utilization` budget is drawn from, so they take room beyond
    that budget, not from the KV cache. A deployment starts only if the budget plus
    `PreSnapshotBytes` fits the part.

The weights and the replicas are computed from the model graph. The other magnitudes come
from the registry: the activation and CUDA graph figures from its `cost-model-memory` set,
the communicator and workspace from `cost-model-primitives`. A kernel lists each registry
value in scope, with how it was obtained, in
[`Provenance`](../reference/provenance.md). For the
deployment on [Pricing a step](step.md):

<!-- example_test.go:memory: blis-vllm kerneltests/example_test.go -->
```go
// The memory example lives in blis-vllm's kerneltests/example_test.go
// (https://github.com/inference-sim/blis-vllm/blob/main/kerneltests/example_test.go),
// which decodes the vLLM engine block it opens the kernel from.
```

```text
weights       78.2 GiB
activations    1.1 GiB
CUDA graphs    0.8 GiB
comm buffers   0.0 GiB
EPLB           0.0 GiB
total         80.1 GiB per rank
before the budget  3.9 GiB
a 4,096-token request: 137 MiB of KV cache per rank
```

## What each request adds

`SequenceVariableBytes(tokens)` is the KV cache one request holds at a given length. Each
layer that keeps a cache stores, for every token,

\[
n_\text{tensors} \cdot \max\!\left(1, \left\lfloor \frac{n_\text{kv}}{\text{tp}}
\right\rfloor\right) \cdot d_\text{head} \cdot \text{bytes per element}
\]

where \(n_\text{tensors}\) is two for ordinary attention, a key and a value, and one for
latent attention (MLA), which caches a single compressed vector per token. Four details
change the answer by large factors:

- **Heads are not split.** Tensor parallelism divides the KV heads among the ranks, but a
  head is never split: a model with 8 KV heads holds one per rank at tp 8, and still one at
  tp 16. A latent cache has a single head, so no tensor-parallel width shrinks it.
- **The cache's data type is its own.** An fp8 cache holds one byte per element even for a
  model served in bf16. Some packed formats state their own size: the sparse-attention
  `fp8_ds_mla` layout takes 656 bytes per token per layer, scales included.
- **Memory comes in pages.** The cache is allocated in blocks of `block_size` tokens, so a
  request's length is rounded up to a whole number of blocks before it is charged.
- **Decode-context parallelism shards tokens.** It spreads a request's tokens across its
  group, so each rank holds a share of them. Prefill-context parallelism does not: every rank
  keeps the whole cache.

The example above is a latent cache: 576 one-byte elements per token in each of 61 layers is
35,136 bytes per token, and a 4,096-token request holds 137 MiB on every rank.

### When the graph states its caches

A model graph can say, per attention layer, which caches the layer owns and how each is laid
out: the bytes one stored record takes under each cache data type, how many context tokens a
record covers, a page size and page alignment of its own, and whether a sliding window or a
fixed entry count bounds it. A layer that states its caches is sized from them alone, and
every other layer keeps the formula above, so a graph opts in node by node.

- **Only an owner holds a cache.** A layer that reads another layer's cache adds read traffic
  and nothing resident. DeepSeek-V4.1-Flash owns its compressed and index caches on four of
  its 40 layers, so it holds about 1,790 bytes per token on H200, where one width times every layer
  gave 20,480.
- **A compressed record covers several tokens.** A cache that stores one record per two
  tokens grows at half its record size per token.
- **A window cache is a bounded ring.** It is counted in `SequenceFixedBytes`, at the pages
  a request holds in steady decode: the window less one, plus the tokens one decode step has
  in flight (one, plus the speculative tokens), rounded up to pages, plus one page.
- **Every cache is drawn from one shared pool.** vLLM groups the layers' caches into KV cache
  groups and gives every group's block the size of the widest group's, so a group of
  narrower pages wastes the rest of each block. The kernel groups the stated caches as the
  release does, through the rules pack, and sizes a request in whole pool blocks in every
  group. DeepSeek-V4.1-Flash on H200 has a 116,928-byte block: its compressed and index
  caches fill it, and its 43 window rings share eight groups of at most six, so a request
  holds 49 blocks in steady decode, 5.73 MB, where its caches' own pages sum to 5.00 MB. A
  rules pack that cannot group a graph's caches leaves each at its own page, and says so in
  `Provenance`.
- **The record follows the chip and the cache data type.** The graph is resolved for the
  chip's GPU generation before it is sized, and a record is looked up under the cache data
  type the attention backend actually runs: DeepSeek-V4.1 serves `fp8` as `fp8_ds_mla` on
  SM90 (584-byte records), and on SM100 as `nvfp4_ds_mla` (528-byte window records,
  288-byte compressed ones) where its mega attention serves the tensor-parallel width, or
  as `fp8_ds_mla` (528-byte records) where it does not.
- **The drafter's caches are held only while speculating.**

## What each request holds regardless of length

`SequenceFixedBytes` holds the state that does not grow with the context: the window rings and
fixed state caches a graph states (above), and the per-request state of recurrent layers
(Mamba-2, gated delta networks, Kimi delta attention). Each rank holds its tensor-parallel
share of that state, as it holds its share of the heads. How much recurrent state a request
holds depends on the engine's recurrent cache mode:

| Mode | State a request holds | Counted in |
|---|---|---|
| `none` | One page, plus one per speculative token | `SequenceFixedBytes` |
| `align` | Two pages, plus one per speculative token | `SequenceFixedBytes` |
| `all` | One page per block of its context, plus one per speculative token | `SequenceVariableBytes` |

vLLM uses `align` whenever prefix caching is on. For a model with no recurrent layers and no
stated bounded caches, `SequenceFixedBytes` is zero.

`SequenceAdmissionBytes` is the same state at what the engine reserves when it admits a
request rather than what the request holds in steady decode. vLLM caps a request's window
ring at the window less one, plus the tokens every step in flight can schedule (two steps of
`max_num_batched_tokens` under asynchronous scheduling), plus one page, and sizes its
reported KV capacity with that cap. On DeepSeek-V4.1-Flash at 8,192 batched tokens that is
517 pages in each window group against the six of steady decode, 484 MB a request, so it
moves with the token budget. The engine's own "GPU KV cache size" is the pool's blocks over a
request's blocks at `max_model_len` (this plus `SequenceVariableBytes(max_model_len)`), times
`max_model_len`.

## In the source

The breakdown is composed in `computeFixedBytes` and its magnitudes resolved in `liftMemory`,
both in [`new.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/new.go). The
KV and recurrent-state laws are in
[`internal/price/memory.go`](https://github.com/inference-sim/blis-latency-kernel/blob/main/internal/price/memory.go).
