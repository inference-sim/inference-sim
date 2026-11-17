# KV Cache & Memory Management

This guide covers KV cache allocation, prefix caching, tiered GPU+CPU offload, and chunked prefill — the memory subsystem that determines how many requests can run concurrently.

The examples use the canonical scenario from [Getting Started](../getting-started/quickstart.md); any scenario directory works.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios testdata/kernel/aisimulate \
  --rate 50 --num-requests 200
```

## Block Allocation Model

KV cache is allocated in **blocks** of the scenario pool's `engine.block_size` tokens. Each request consumes `ceil(token_count / block_size)` blocks. Blocks are reference-counted and can be shared across requests via prefix caching.

The GPU-tier block budget is not a flag. blis-latency-kernel computes it with its memory methods from the scenario — model, hardware, parallelism, quantization, `kv_cache_dtype`, `gpu_memory_utilization` and `block_size` — **per data-parallel rank**: a pool with `dp > 1` (MoE only) becomes one replica per rank, each with its own budget. A LoRA run's static adapter HBM reservation is set aside from that budget. To change the budget, change the scenario (for example its `gpu_memory_utilization` or `cache_dtype`).

!!! tip "Block size affects prefix cache granularity"
    Prefix caching uses block-aligned hashing (`hash.ComputeBlockHashes`). Smaller block sizes increase cache hit granularity but also increase allocation overhead. Choose block size relative to your typical prefix lengths.

## Prefix Caching

When requests share common prefixes (e.g., system prompts in RAG), BLIS can reuse KV cache blocks from prior computations. This reduces prefill tokens and improves TTFT.

Each instance reuses cached blocks on its own. Whether same-prefix requests land on the same
instance is the llm-d router's decision. Its built-in default configuration includes the
`prefix-cache-scorer` (weight 3, beside `queue-scorer` and `kv-cache-utilization-scorer` at 2),
which estimates each endpoint's cached prefix from the requests the router has sent it; see
[Routing](routing.md). `examples/llm-d/prefix-affinity.yaml` is an explicit prefix-aware
configuration:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node-prefix-affinity.yaml \
  --scenarios examples/llm-d/scenarios \
  --num-instances 4 \
  --prefix-tokens 512 --rate 100 --num-requests 500
```

The `precise-prefix-cache-producer`, which reads vLLM KV-cache events, runs: with it in the
router configuration each simulated engine publishes vLLM's KV events (a block stored, a block
removed, and a CPU-tier location event), the router receives them `--kv-event-delay` later and
indexes the blocks each engine can serve. [Routing](routing.md) gives the configuration and the cases that are
refused.

### Disabling reuse across requests

A deployment launched with vLLM's `--no-enable-prefix-caching` reuses nothing across
requests. A scenario states that with `enable_prefix_caching: false` on its pool's engine
block; the field is tri-state, and omitting it takes vLLM's own default, which is **on**.

The setting changes the work a prefill does, not only the memory it holds: with reuse on, a
matched prefix arrives as already-computed tokens and only the remainder is charged. A
request's own progress is unaffected either way — a chunked prefill resumes from where it
stopped, because that is not another request's block.

```yaml
pools:
  - role: colocated
    engine:
      enable_prefix_caching: false
```

A workload whose requests share no prefix is unaffected by the setting, because there is
nothing to reuse. That is the case for the AISimulate accuracy corpus, whose spec sets
`cached_prefix_tokens` to zero.

## Minimum KV Block Requirements

!!! danger "DroppedUnservable rejection"
    The run-wide `DroppedUnservable` count also includes empty prompts rejected by the gateway before router dispatch. The engine drops requests as **unservable** in two KV-related cases:

    1. **MaxModelLen guard** — requests whose total sequence length (input + output budget) exceeds the scenario's `max_model_len` are rejected before entering the queue. This mirrors vLLM's `--max-model-len` validation.
    2. **KV capacity guard** — when `ceil(inputTokens / blockSize) > TotalCapacity()`, the request physically cannot fit in GPU memory. This mirrors vLLM's pre-engine rejection path.

    Both engine guards fire at enqueue time, before the request enters the wait queue. The gateway's empty-prompt rejection happens before router flow control and shares the counter, but is not a KV-capacity failure.

!!! info "Proactive MaxModelLen cap"
    A three-part enforcement of the scenario's `max_model_len` matches vLLM's scheduler semantics: (1) `FormBatch` proactively clamps token scheduling to `maxModelLen - 1 - ProgressIndex`, (2) `executeBatchStep` skips decode when no tokens are allocated, and (3) `processCompletions` force-completes requests at the `maxModelLen - 1` boundary. Output per length-capped request: `maxModelLen - inputLen` tokens — `ProgressIndex` is BLIS's `num_computed_tokens` and lags the generated-token count by one (the first output token is charged to prefill), so stopping at `maxModelLen - 1` is exactly vLLM's `check_stop` (`num_tokens >= max_model_len`) and the boundary token is counted.

Compute the minimum blocks needed for your workload:

```
min_blocks = ceil(max_input_tokens / block_size)
```

For a workload with max 7,000 input tokens and block size 16: `ceil(7000/16) = 438` blocks minimum. Below this, requests are dropped. Below ~2x this threshold, cascading preemptions cause severe throughput degradation.

## Tiered Caching (GPU + CPU Offload)

The scenario deployment's `offload` block configures the offload tiers. It is the only source
of offload configuration; no flag sets it. It maps onto vLLM's OffloadingConnector:

```yaml
offload:
  connector: OffloadingConnector    # or SimpleCPUOffloadConnector; may be left out
  spec: CPUOffloadingSpec           # or TieringOffloadingSpec; may be left out
  eviction_policy: lru              # lru only (vLLM's arc is not simulated yet)
  tiers:
    - tier: cpu_dram                # the CPU tier (vLLM cpu_bytes_to_use); required, and first
      bytes: 17179869184
```

```bash
./blis run --scenario llama-offload.yaml \
  --scenarios my-scenarios \
  --rate 100 --num-requests 500
```

Here `my-scenarios/llama-offload.yaml` is a copy of the canonical scenario with the block above
added to its deployment.

Each tier names a catalog storage device (`<catalog>/devices/storage.yaml` in the catalog in
use: the embedded copy, or the directory `--catalog` / `BLIS_CATALOG` names), and the kernel
prices each tier transfer from that device (its `TierTime`). There are no transfer-rate flags. A
class the catalog does not define, or a missing or malformed table, is a hard error naming the
path. The rules:

- The `cpu_dram` tier is required, and it must be the first tier: every secondary tier sits
  behind the CPU tier, as in vLLM.
- The v0.33.0 engine models the `cpu_dram` tier only and refuses a secondary tier, such as
  `tier: nvme_gen4` (blis-vllm#63). A secondary tier needs `spec: TieringOffloadingSpec`, and
  its device must be one the catalog defines.
- An object store (`s3`) is refused: vLLM reaches it through its `obj` tier, which BLIS does
  not simulate.
- `SimpleCPUOffloadConnector` takes one `cpu_dram` tier and offloads every block.
- Another connector or spec, `prefetch_depth`, `arc` and a 0-byte tier are refused by name.

vLLM's defaults apply to what blis-schemas cannot state: the offload block size (one GPU
block per chunk), `offload_prompt_only: true` and `tokens_per_hash`. Under P/D disaggregation
each pool sizes the tiers by its own bytes per block and prices them with its own kernel. See
[Latency models](latency-models.md#kv-offload) for the mapping.

**Chunks (INV-K6).** The tiers hold vLLM's offload chunks: GPU blocks keyed by their last
block's hash. The CPU tier holds `cpu_bytes_to_use // (bytes per block × blocks_per_chunk)`
chunks, evicts whole chunks in LRU order, and every tier transfer moves one chunk. A load starts
at the chunk holding the GPU-cached boundary, covers whole chunks, and is skipped when it is
shorter than one chunk. A store offers whole prompt chunks only. A CPU budget under one chunk is
refused at startup. blis-schemas' `offload` block cannot yet state the width (the proposed field
is `offload.blocks_per_chunk`), so a scenario runs one GPU block per chunk.

The resolved config is recorded in the exported trace header, so a `blis run --trace-output`
round-trips through `blis replay` (INV-13): on replay the header is authoritative, a scenario
`offload` block may only restate it, and a config the binary cannot reproduce fails loudly
rather than silently degrading to single-tier.

**What enters the tiers.** Modeling vLLM's mechanism (`_calc_num_offloadable_tokens` +
`storable_chunks`), a request's computed KV is truncated to the prompt length under vLLM's
default `offload_prompt_only: true`, then floor-divided into whole chunks, so a chunk containing
any decode token is never offloaded (a prompt of `1.5 ×` the chunk size offloads exactly 1
chunk). The `SimpleCPUOffloadConnector` has no prompt-only clamp: full decode blocks are
offloaded too. Because BLIS hashes every completed block prefix-consistently (for
`block_size > 1`), a later request on the same instance whose **input contains earlier output
tokens** (multi-turn / agentic workloads) then reloads that decode KV from the tier instead of
recomputing it. A reloaded prefix is billed as a **cache hit**: its tokens are dropped from the
prefill forward pass, so it lowers that request's prefill compute and TTFT rather than being
charged as a full recompute (#1699). Reuse is single-instance (offload tiers are per-instance
and invisible to the router).

At `block_size == 1` decode blocks take a guarded allocation path that leaves them unhashed, so
decode-offload is inert there. With no `offload` block, behavior is unchanged (INV-6).

### Enabling and Disabling Offload

Offload is **off by default**. There is one switch: the presence of an `offload` block in the
scenario's deployment.

### Sizing the CPU Tier

The CPU tier is configured in **bytes**, but the cache uses it in **blocks** — the same unit as
the GPU tier. BLIS converts once, at startup:

```
block_capacity = floor(cpu_bytes_to_use / per_block_bytes)
```

`per_block_bytes` is derived from the model, so it differs per deployment. The rule that matters:
**`block_capacity` must comfortably exceed the GPU-tier budget.** A CPU tier no larger than the
GPU tier cannot serve anything the GPU evicted, and the run will show no benefit.

Offload also only helps if blocks are **evicted** from the GPU and later requested again — for
example a workload with many distinct per-tenant prefixes (`prefix_sharing: per_member`) whose
working set exceeds the GPU tier. Note that a workload spec's `prefix_length` is **added** to the
sampled `input_distribution` length, not carved out of it.

!!! note "`cpu_bytes_to_use` is per GPU, not per deployment"
    Under tensor parallelism the KV cache is sharded across ranks, so both sides of that division
    are per-rank quantities. The host memory a deployment actually consumes is `TP × cpu_bytes_to_use`.
    A node-level memory budget must therefore be divided by TP before it goes in the `cpu_dram` tier's `bytes`. This
    matches vLLM, where `cpu_bytes_to_use` is likewise a per-worker budget.

## Chunked Prefill

Long prefill sequences can cause **head-of-line (HOL) blocking** — a long prefill occupies a whole step, blocking shorter requests from starting.

Chunked prefill splits long prefills into smaller chunks. The threshold is the scenario's, stated in a pool's `engine` block (an engine setting has no flag, inference-sim#2021 B6):

```yaml
    engine:
      long_prefill_token_threshold: 256
```

!!! info "Chunked prefill benefits TTFT, not ITL"
    With `long_prefill_token_threshold: 256`, short-request TTFT p99 improves by ~52% in bimodal workloads. But ITL is unaffected (<0.5%) because ~255 of ~256 ITL samples per request are decode-only steps. The benefit is in scheduling new requests, not in token generation speed.

!!! note "A lone request is not chunked"
    As in vLLM v0.31.0 (#57951), the threshold applies only when the engine's scheduler holds another request, running or waiting. A request alone takes the whole `max_num_batched_tokens` budget, since there is nobody for it to starve.

## Batch Formation Parameters

KV cache pressure is directly coupled to batch formation. The scenario pool's `engine.max_num_seqs` (maximum requests in the running batch) and `engine.max_num_batched_tokens` (token budget per step) are the primary capacity knobs, with vLLM's meaning. Reducing them decreases KV cache pressure but also reduces throughput.

## Identifying the KV Pressure Cliff

Preemption rates spike non-linearly once the working set exceeds the KV budget. Sweep offered load against a fixed scenario to find it:

```bash
for rate in 5 10 20 40 80; do
  echo "=== rate=$rate ==="
  ./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
    --scenarios testdata/kernel/aisimulate \
    --rate $rate --num-requests 200 2>/dev/null \
    | grep -E "preemption_count|completed_requests"
done
```

!!! tip "Distribution median drives KV pressure"
    ParetoLogNormal distributions produce *fewer* preemptions than Gaussian despite similar means, because the Pareto component's median (~79 tokens) is much lower than Gaussian's median (~256 tokens). Short requests cycle faster, creating "breathing room" in the KV cache.

## Further Reading

- [Core Engine: The engine interface](../concepts/core-engine.md#the-engine-interface) — where the engine's KV cache sits
- [Configuration Reference](../reference/configuration.md#kv-cache-configuration) — KV cache configuration
- [Metrics & Results](results.md) — understanding preemption rate, cache hit rate, KV thrashing
