# KV Cache & Memory Management

This guide covers KV cache allocation, prefix caching, tiered GPU+CPU offload, and chunked prefill — the memory subsystem that determines how many requests can run concurrently.

```bash
# Quick example: simulate with reduced KV blocks to observe preemptions
./blis run --model qwen/qwen3-14b \
  --total-kv-blocks 5000 --rate 50 --num-requests 200
```

## Block Allocation Model

KV cache is allocated in **blocks** of `--block-size-in-tokens` tokens (default: 16). Each request consumes `ceil(token_count / block_size)` blocks. Blocks are reference-counted and can be shared across requests via prefix caching.

| Flag | Default | Description |
|------|---------|-------------|
| `--total-kv-blocks` | Per-model* | Total GPU-tier KV blocks |
| `--block-size-in-tokens` | 16 | Tokens per block |

*For roofline and trained-physics modes, the block count is auto-calculated from model architecture and GPU memory. Explicit `--total-kv-blocks` always wins. See [Configuration Reference](../reference/configuration.md#resolution-process).

!!! tip "Block size affects prefix cache granularity"
    Prefix caching uses block-aligned hashing (`hash.ComputeBlockHashes`). Smaller block sizes increase cache hit granularity but also increase allocation overhead. Choose block size relative to your typical prefix lengths.

## Prefix Caching

When requests share common prefixes (e.g., system prompts in RAG), BLIS can reuse KV cache blocks from prior computations. This reduces prefill tokens and improves TTFT.

Prefix caching is automatic when using the `weighted` routing policy. The default profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) queries actual instance KV cache state to route requests to instances with cached prefix blocks:

```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --routing-policy weighted \
  --prefix-tokens 512 --rate 100 --num-requests 500
```

## Minimum KV Block Requirements

!!! danger "DroppedUnservable rejection"
    Requests are dropped as **unservable** (incrementing `DroppedUnservable`) in two cases:

    1. **MaxModelLen guard** — when `--max-model-len` is set, requests whose total sequence length (input + output budget) exceeds the context window are rejected before entering the queue. This mirrors vLLM's `--max-model-len` validation.
    2. **KV capacity guard** — when `ceil(inputTokens / blockSize) > TotalCapacity()`, the request physically cannot fit in GPU memory. This mirrors vLLM's pre-engine rejection path.

    Both guards fire at enqueue time, before the request enters the wait queue.

!!! info "Proactive MaxModelLen cap"
    When `--max-model-len` is set, a three-part enforcement matches vLLM's scheduler semantics: (1) `FormBatch` proactively clamps token scheduling to `maxModelLen - 1 - ProgressIndex`, (2) `executeBatchStep` skips decode when no tokens are allocated, and (3) `processCompletions` force-completes requests at the `maxModelLen - 1` boundary. Output per length-capped request: `maxModelLen - 1 - inputLen` tokens.

Compute the minimum blocks needed for your workload:

```
min_blocks = ceil(max_input_tokens / block_size)
```

For a workload with max 7,000 input tokens and block size 16: `ceil(7000/16) = 438` blocks minimum. Below this, requests are dropped. Below ~2x this threshold, cascading preemptions cause severe throughput degradation.

## Tiered Caching (GPU + CPU Offload)

BLIS models tiered KV cache with GPU→CPU offloading:

```bash
./blis run --model qwen/qwen3-14b \
  --kv-cpu-blocks 50000 \
  --kv-offload-threshold 0.9 \
  --kv-transfer-bandwidth 100.0 \
  --rate 100 --num-requests 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-cpu-blocks` | 0 | CPU-tier blocks (0 = disabled) |
| `--kv-offload-threshold` | 0.9 | GPU utilization fraction above which blocks offload to CPU |
| `--kv-transfer-bandwidth` | 100.0 | GPU→CPU transfer rate in blocks/tick |
| `--kv-transfer-base-latency` | 0 | Fixed per-transfer latency in ticks |

### Multi-Tier Offload Config Surface (`--kv-offload-config`)

The scalar flags above cover the single CPU tier. For vLLM's **multi-tier** offload
(CPU → disk / object store), BLIS captures the full config surface through one strict-YAML
file — `--kv-offload-config <path>` — with a single top-level `kv_offload:` block. This
mirrors `--lora-config` / `--saturation-config`: absent ⇒ the offload subsystem is inert and
output is byte-identical to a build without it.

```bash
./blis run --model qwen/qwen3-14b --kv-offload-config offload.yaml
```

```yaml
kv_offload:
  cpu_bytes_to_use: 17179869184     # required when the block is present
  block_size: 16                    # optional; default = GPU block size (mutually
                                    #   exclusive with blocks_per_chunk)
  # blocks_per_chunk: 1             # alternate encoding of block_size (default 1)
  eviction_policy: lru              # lru | arc  (default lru)
  offload_prompt_only: true         # vLLM DEFAULT (prompt-only). false => promptAndDecode:
                                    #   full decode blocks are offloaded and reused too (see below)
  # self_describing_kv_events: false
  # tokens_per_hash: 16             # default = GPU block size
  secondary_tiers:
    - type: fs                      # only "fs" is representable today; obj/p2p error loudly
      root_dir: /mnt/kv-cache
      n_read_threads: 16            # vLLM default 16
      n_write_threads: 16           # vLLM default 16
      locality: LOCAL               # LOCAL | REMOTE (optional)
      direct_io: true               # REQUIRED — BLIS makes vLLM's runtime O_DIRECT probe explicit
      device_class: nvme_gen4       # resolves read/write bandwidth + latency from defaults.yaml
      # read_bandwidth: 7000.0      # bytes/µs — overrides device_class (per-direction, required as a pair)
      # write_bandwidth: 5000.0
      # base_latency: 80.0          # µs
```

Defaults match vLLM knob-for-knob. Anything vLLM accepts either maps to a BLIS config or
fails **loudly** at startup — never silently ignored: `store_threshold >= 2` is rejected
(vLLM's `TieringOffloadingSpec` rejects it), and `obj`/`p2p`/`example` tier types are rejected
(no faithful BLIS mapping yet). `device_class` names resolve against the `kv_offload_devices:`
block shipped in `defaults.yaml` (bandwidth in bytes/µs, latency in µs); an explicit
`read_bandwidth`/`write_bandwidth`/`base_latency` triple overrides the class.

The resolved config is recorded in the exported trace header, so a `blis run --trace-output`
round-trips through `blis replay` (INV-13): on replay the header is authoritative and a config
the binary cannot reproduce fails loudly rather than silently degrading to single-tier.

The `offload_prompt_only` knob is an explicit policy over *what enters the tiers*. Modeling
vLLM's mechanism (`_calc_num_offloadable_tokens` + `storable_chunks`), a request's computed KV is
truncated to the prompt length when `true` (the default), then floor-divided into whole chunks — so
a chunk containing any decode token is never offloaded (a prompt of `1.5 ×` the chunk size offloads
exactly 1 chunk). With `offload_prompt_only: false` (vLLM's `promptAndDecode`), full decode blocks
are offloaded too; because BLIS already hashes every completed block prefix-consistently (for
`block_size > 1`), a later request on the same instance whose **input contains earlier output
tokens** (multi-turn / agentic workloads) reloads that decode KV from the tiers instead of
recomputing it. A reloaded prefix is billed as a **cache hit** — its tokens are dropped from the
prefill forward pass, so it lowers that request's prefill compute and TTFT rather than being
charged as a full recompute (#1699; the same correction applies to the legacy `--kv-cpu-blocks`
tier). Reuse is single-instance (offload tiers are per-instance and invisible to the router).

!!! note "Chunked-prefill limitation (#1706)"
    The reload credit is currently capped at **one prefill chunk**. When a reload extends the
    cached prefix past a request's per-step chunk boundary (set by `--long-prefill-token-threshold`
    / `--max-num-batched-tokens`), only the first chunk is billed as a hit; the reloaded remainder
    is re-billed as recompute on later steps. So for long-context prompts that are chunked (the main
    reason to offload KV), the realized TTFT benefit is roughly `chunk_size / prompt_length` — small
    at the default batched-token budget. Uncapped (prompt-fits-one-chunk) workloads get the full
    credit. Tracked in [#1706](https://github.com/inference-sim/inference-sim/issues/1706).

At `block_size == 1` decode blocks take a guarded allocation path that leaves them unhashed, so
decode-offload is inert there — a degenerate offload block size (real offload block sizes track
the GPU block size). With no `--kv-offload-config`, behavior is unchanged (INV-6).

### Enabling and Disabling Offload

Offload is **off by default**. There is one switch — the presence of `--kv-offload-config`:

```bash
# ENABLED — CPU staging tier
./blis run --model qwen/qwen3-14b --workload-spec wl.yaml \
  --total-kv-blocks 3000 --kv-offload-config offload_cpu.yaml

# DISABLED — omit the flag
./blis run --model qwen/qwen3-14b --workload-spec wl.yaml \
  --total-kv-blocks 3000
```

`--kv-offload-config` is mutually exclusive with the legacy scalar `--kv-cpu-blocks` tier.

### Sizing the CPU Tier

Blocks reach the CPU tier only by being evicted from the GPU tier, so a CPU tier that cannot
outgrow the GPU tier has nothing to hold.

#### Block capacity — what the tier actually holds

The tier is configured in **bytes**, but the cache uses it in **blocks**. BLIS converts once, at
startup, by integer division:

```
per_block_bytes = 2 (K+V) × layers × kv_heads × head_dim × dtype_bytes × block_size
block_capacity  = floor(cpu_bytes_to_use / per_block_bytes)
```

`block_capacity` is the number that matters: it is what LRU/ARC evicts against, and the only
figure comparable to the workload's working set. Bytes are not comparable to anything — two
models with the same `cpu_bytes_to_use` hold wildly different numbers of blocks.

##### `block_capacity` vs `--total-kv-blocks`

They are **the same unit for two different tiers**, and that is the whole point of computing
`block_capacity` — it is what makes the two tiers comparable:

| | Tier | How it is configured | Value for the example below |
|---|---|---|---|
| `--total-kv-blocks` | GPU (HBM) | directly, in blocks — or auto-derived from GPU memory when omitted | 3,000 |
| `block_capacity` | CPU (host RAM) | *indirectly*, in bytes via `cpu_bytes_to_use`; BLIS divides | 419,430 |

Both count the same thing — whole KV blocks of `block_size` tokens — so `block_capacity > total_kv_blocks`
is the condition for the CPU tier to be able to hold what the GPU evicts. The asymmetry is only in
the configured units: the GPU tier is set in blocks, the CPU tier in bytes (matching vLLM's knob
for each). Both scale with TP, so the comparison must be made at the intended TP.

For Qwen3-14B at TP=1, `2 × 40 layers × 8 KV heads × 128 head-dim × 2 B (bf16) × 16 tokens` =
**2,621,440 bytes/block**. So the 1 TiB tier in the example above is
`1099511627776 / 2621440` = **419,430 blocks**, against 3,000 GPU blocks and a working set of
`100 prefixes × 1024 tokens / 16` = 6,400 blocks — comfortably larger than both.

Because the division is truncating, the remainder is simply unused: at 2,621,440 bytes/block,
`cpu_bytes_to_use: 5000000` buys 1 block, not 1.9. Sizing from a target block count avoids this
entirely — `cpu_bytes_to_use = target_blocks × per_block_bytes`.

!!! warning "The capacity of a CPU tier must be larger than the default GPU tier"
    Check `cpu_bytes_to_use / per_block_bytes` against `--total-kv-blocks` before concluding anything from a run.

#### "Per TP rank" — both sides of that division are per-GPU

Under tensor parallelism the KV cache is **sharded across ranks**: each GPU holds the same
*logical* blocks (the same token ranges) but only its own slice of the KV heads. So each GPU
needs only its own slice of host memory to stage those blocks. BLIS therefore treats both sides
of the division as per-rank — `cpu_bytes_to_use` is the host budget for **one** GPU, and
`per_block_bytes` is one block's size on **one** rank (`KVBytesPerToken(model, tp) × block_size`,
divided by TP at [`sim/latency/kv_capacity.go:182`](https://github.com/inference-sim/inference-sim/blob/main/sim/latency/kv_capacity.go#L182)).
Dividing a per-rank budget by a per-rank block size yields the count of *logical* blocks the
replica can cache — which is exactly the unit the cache model needs, and the same unit as
`--total-kv-blocks`.

Two consequences, both easy to get backwards:

1. **The same `cpu_bytes_to_use` buys more blocks at higher TP.** `per_block_bytes` shrinks by a
   factor of TP, so block capacity grows by the same factor.
2. **The host RAM a deployment actually consumes is `TP × cpu_bytes_to_use`.** The value in the
   YAML is per-GPU, so a node-level memory budget must be divided by TP before it goes in the
   file — not entered whole.

Measured for Qwen3-14B (`--tp N`, verified by reading back the derived value):

| `--tp` | `per_block_bytes` | Blocks from a 16 GiB `cpu_bytes_to_use` | Host RAM used node-wide |
|---|---|---|---|
| 1 | 2,621,440 | 6,553 | 16 GiB |
| 2 | 1,310,720 | 13,107 | 32 GiB |
| 4 | 655,360 | 26,214 | 64 GiB |
| 8 | 327,680 | 52,428 | 128 GiB |

This matches vLLM, where `cpu_bytes_to_use` is likewise a per-worker budget.

!!! note "Exception: MLA models are not divided by TP"
    **MLA models** (DeepSeek-V2/V3, GLM-5.2, Kimi-K3) cache a single compressed latent of
    `kv_lora_rank + qk_rope_head_dim` per token per layer. That latent is *replicated* on every
    rank rather than sharded, so `per_block_bytes` is **not** divided by TP — block capacity is
    the same at TP=8 as at TP=1, while node-wide host RAM still scales with TP.

    Source: the MLA branch in
    [`sim/latency/kv_capacity.go:130-135`](https://github.com/inference-sim/inference-sim/blob/main/sim/latency/kv_capacity.go#L130-L135)
    ("replicated across TP ranks (NOT divided by TP), and independent of numKVHeads/headDim"),
    added in #1527. This mirrors vLLM, which likewise keeps the MLA latent unsharded; the
    architecture is from the [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434) §2.1.

!!! note "Known inaccuracy: GQA with fewer KV heads than ranks over-reports block capacity"
    When `kv_heads < tp` (e.g. 2 KV heads at TP=4) vLLM **replicates** KV heads across GPUs, so
    per-GPU KV bytes do not keep shrinking. BLIS divides by TP regardless, which *underestimates*
    per-GPU bytes and therefore *overestimates* CPU-tier block capacity — by `tp / kv_heads`, so
    2× in that example. The simulator models more cache than the hardware would have.

    This is **deliberate and documented** in
    [`sim/latency/kv_capacity.go:92-100`](https://github.com/inference-sim/inference-sim/blob/main/sim/latency/kv_capacity.go#L92-L100),
    which labels it "a known approximation (optimistic)" — but it is an accuracy defect rather
    than a modeling choice with a justification, and it is silent: nothing warns at startup that
    the configuration entered this regime. Treat capacity figures as optimistic for any
    `kv_heads < tp` run, and prefer `tp <= kv_heads` when the comparison matters. Worth raising
    upstream; not yet filed.

    When `kv_heads >= tp` the division is exact — and a `kv_heads` not evenly divisible by `tp`
    is rejected outright rather than approximated.

### Example: The Measured Effect of CPU Offload

#### The workload needs prefixes the GPU will evict

Offload only helps if blocks are **evicted** from the GPU and later requested again. A single
shared prefix (`--prefix-tokens N`) stays pinned hot on the GPU and is never evicted, so the
lower tiers have nothing to serve and the run is byte-identical with offload on or off. The
fixture below forces eviction with 100 distinct per-tenant prefixes:

```yaml
# wl_multitenant.yaml — 100 tenants, each with its own 1,024-token prefix
version: "2"
seed: 42
aggregate_rate: 4.0
num_requests: 600
cohorts:
  - id: tenants
    population: 100
    prefix_group: doc
    prefix_sharing: per_member     # 100 DISTINCT prefixes => GPU must evict
    prefix_length: 1024            # ADDITIVE on input_distribution
    rate_fraction: 1.0
    arrival: {process: poisson}
    input_distribution:  {type: constant, params: {value: 1280}}
    output_distribution: {type: constant, params: {value: 16}}
```

```yaml
# offload_cpu.yaml — one CPU tier, large enough to hold the whole working set
kv_offload:
  cpu_bytes_to_use: 1099511627776   # 1 TiB
  block_size: 16
  eviction_policy: lru
  offload_prompt_only: true
```

!!! warning "`prefix_length` is added to `input_distribution`"
    The generator samples `input_distribution` first, then prepends the prefix tokens to that slice:

    ```go
    // sim/workload/generator.go
    inputTokens = append(append([]sim.TokenID{}, prefix...), inputTokens...)
    ```

    So `input_distribution: 1280` with `prefix_length: 1024` gives 1,280 **unique** tokens per
    request and a **2,304**-token prompt — not a 1,280-token prompt of which 1,024 are shared.
    Therefore, 600 requests report `total_input_tokens: 1382400` (= 600 × 2,304), and the
    shareable fraction is `1024 / 2304` = 44%.

    Separately, `--workload-spec` supersedes `--prefix-tokens` / `--rate`: passing them alongside
    a spec is inert rather than an error, so the arrival rate comes from the spec's
    `aggregate_rate: 4.0`.

#### Results

Run the enabled and disabled commands from the previous section. `Cache Hit Rate` is printed to
stdout under `=== KV Cache Metrics ===`; add `--metrics-path m.json` for the full-precision
`cache_hit_rate` field.

| | `cache_hit_rate` | `ttft_mean_ms` | `e2e_mean_ms` | `responses_per_sec` |
|---|---|---|---|---|
| Offload **disabled** | 0.0772 | 52.550 | 250.303 | 4.1640 |
| Offload **enabled** (1 TiB CPU tier) | **0.3665** | **48.914** | **243.994** | 4.1641 |

Cache hit rate rises **4.7×** and mean TTFT falls **6.9%** (−3.6 ms). Throughput is unchanged
because this workload is arrival-bound — 4 req/s offered against an unsaturated instance. Offload
buys latency here; it buys *throughput* only under saturation, where spending fewer prefill tokens
per request lets more requests into each step.

Why the hit rate lands near 0.37: only the 1,024-token prefix is shareable and each tenant's first
request must miss, so the ceiling is `1024 × 500 / 1382400` = **37.0%**. The enabled run reaches
0.3665 — essentially every reuse the workload contains.

!!! tip "Large cohorts need `--lazy-generation`"
    Eager workload generation costs roughly `population × num_requests × 44 KB` in resident
    memory, so a 300-member cohort at 3,000 requests needs ~38 GB and is OOM-killed. Add
    `--lazy-generation` whenever that product exceeds a few tens of thousands.

## Chunked Prefill

Long prefill sequences can cause **head-of-line (HOL) blocking** — a 2,048-token prefill takes ~97ms on Qwen3-14B / H100 / TP=1 (roofline mode), blocking shorter requests from starting.

Chunked prefill splits long prefills into smaller chunks:

```bash
./blis run --model qwen/qwen3-14b \
  --long-prefill-token-threshold 256 \
  --rate 100 --num-requests 500
```

!!! info "Chunked prefill benefits TTFT, not ITL"
    With `--long-prefill-token-threshold=256`, short-request TTFT p99 improves by ~52% in bimodal workloads. But ITL is unaffected (<0.5%) because ~255 of ~256 ITL samples per request are decode-only steps. The benefit is in scheduling new requests, not in token generation speed.

## Batch Formation Parameters

KV cache pressure is directly coupled to batch formation:

| Flag | Default | Description |
|------|---------|-------------|
| `--max-num-seqs` | 256 | Maximum requests in the running batch (vLLM parity; deprecated alias `--max-num-running-reqs`) |
| `--max-num-batched-tokens` | 2048 | Token budget per step (vLLM parity; deprecated alias `--max-num-scheduled-tokens`) |

These are the primary capacity knobs — in vLLM terms, `max_num_seqs` and `max_num_batched_tokens`. Reducing them decreases KV cache pressure but also reduces throughput.

## Identifying the KV Pressure Cliff

Preemption rates spike non-linearly as KV blocks decrease past a threshold. The threshold depends on your workload's **median** token count (not mean or tail):

```bash
# Sweep KV blocks to find the cliff
for blocks in 100000 50000 20000 10000 5000 3000; do
  echo "=== blocks=$blocks ==="
  ./blis run --model qwen/qwen3-14b \
    --total-kv-blocks $blocks --rate 50 --num-requests 200 2>/dev/null \
    | grep -E "preemption_count|completed_requests"
done
```

!!! tip "Distribution median drives KV pressure"
    ParetoLogNormal distributions produce *fewer* preemptions than Gaussian despite similar means, because the Pareto component's median (~79 tokens) is much lower than Gaussian's median (~256 tokens). Short requests cycle faster, creating "breathing room" in the KV cache.

## Further Reading

- [Core Engine: KV Cache](../concepts/core-engine.md#kv-cache-management) — internal mechanics
- [Configuration Reference](../reference/configuration.md#kv-cache-configuration) — all KV cache flags
- [Metrics & Results](results.md) — understanding preemption rate, cache hit rate, KV thrashing
