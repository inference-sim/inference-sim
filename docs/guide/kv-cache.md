# KV Cache & Memory Management

This guide covers KV cache allocation, prefix caching, tiered GPU+CPU offload, and chunked prefill — the memory subsystem that determines how many requests can run concurrently.

The examples use the canonical scenario from [Getting Started](../getting-started/quickstart.md); any scenario directory works.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 50 --num-requests 200
```

## Block Allocation Model

KV cache is allocated in **blocks** of the scenario pool's `engine.block_size` tokens. Each request consumes `ceil(token_count / block_size)` blocks. Blocks are reference-counted and can be shared across requests via prefix caching.

The GPU-tier block budget is not a flag. blis-latency-kernel computes it with its memory methods from the scenario — model, hardware, parallelism, quantization, `cache_dtype`, `gpu_memory_utilization` and `block_size` — **per data-parallel rank**: a pool with `dp > 1` (MoE only) becomes one replica per rank, each with its own budget. A LoRA run's static adapter HBM reservation is set aside from that budget. To change the budget, change the scenario (for example its `gpu_memory_utilization` or `cache_dtype`).

!!! tip "Block size affects prefix cache granularity"
    Prefix caching uses block-aligned hashing (`hash.ComputeBlockHashes`). Smaller block sizes increase cache hit granularity but also increase allocation overhead. Choose block size relative to your typical prefix lengths.

## Prefix Caching

When requests share common prefixes (e.g., system prompts in RAG), BLIS can reuse KV cache blocks from prior computations. This reduces prefill tokens and improves TTFT.

Prefix caching is automatic when using the `weighted` routing policy. The default profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) queries actual instance KV cache state to route requests to instances with cached prefix blocks:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --routing-policy weighted \
  --prefix-tokens 512 --rate 100 --num-requests 500
```

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
    Requests are dropped as **unservable** (incrementing `DroppedUnservable`) in two cases:

    1. **MaxModelLen guard** — requests whose total sequence length (input + output budget) exceeds the scenario's `max_model_len` are rejected before entering the queue. This mirrors vLLM's `--max-model-len` validation.
    2. **KV capacity guard** — when `ceil(inputTokens / blockSize) > TotalCapacity()`, the request physically cannot fit in GPU memory. This mirrors vLLM's pre-engine rejection path.

    Both guards fire at enqueue time, before the request enters the wait queue.

!!! info "Proactive MaxModelLen cap"
    A three-part enforcement of the scenario's `max_model_len` matches vLLM's scheduler semantics: (1) `FormBatch` proactively clamps token scheduling to `maxModelLen - 1 - ProgressIndex`, (2) `executeBatchStep` skips decode when no tokens are allocated, and (3) `processCompletions` force-completes requests at the `maxModelLen - 1` boundary. Output per length-capped request: `maxModelLen - inputLen` tokens — `ProgressIndex` is BLIS's `num_computed_tokens` and lags the generated-token count by one (the first output token is charged to prefill), so stopping at `maxModelLen - 1` is exactly vLLM's `check_stop` (`num_tokens >= max_model_len`) and the boundary token is counted.

Compute the minimum blocks needed for your workload:

```
min_blocks = ceil(max_input_tokens / block_size)
```

For a workload with max 7,000 input tokens and block size 16: `ceil(7000/16) = 438` blocks minimum. Below this, requests are dropped. Below ~2x this threshold, cascading preemptions cause severe throughput degradation.

## Tiered Caching (GPU + CPU Offload)

The legacy single CPU tier is enabled with `--kv-cpu-blocks`:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --kv-cpu-blocks 50000 \
  --kv-offload-threshold 0.9 \
  --rate 100 --num-requests 500
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kv-cpu-blocks` | 0 | CPU-tier blocks (0 = disabled) |
| `--kv-offload-threshold` | 0.9 | GPU utilization fraction above which blocks offload to CPU |

The GPU↔CPU transfer is priced by the kernel from the catalog's `cpu_dram` storage device (`<catalog>/devices/storage.yaml`). There are no transfer-rate flags. For a multi-tier hierarchy, use `--kv-offload-config` instead; the two are mutually exclusive.

### Multi-Tier Offload Config Surface (`--kv-offload-config`)

The scalar flags above cover the single CPU tier. For vLLM's **multi-tier** offload
(CPU → disk / object store), BLIS captures the full config surface through one strict-YAML
file — `--kv-offload-config <path>` — with a single top-level `kv_offload:` block. This
mirrors `--lora-config` / `--saturation-config`: absent ⇒ the offload subsystem is inert and
output is byte-identical to a build without it.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --kv-offload-config offload.yaml
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
      device_class: nvme_gen4       # REQUIRED — a catalog storage device; the kernel prices transfers
```

Defaults match vLLM knob-for-knob. Anything vLLM accepts either maps to a BLIS config or
fails **loudly** at startup — never silently ignored: `store_threshold >= 2` is rejected
(vLLM's `TieringOffloadingSpec` rejects it), and `obj`/`p2p`/`example` tier types are rejected
(no faithful BLIS mapping yet).

Every tier must name a `device_class` from the catalog's storage-device table
(`<catalog>/devices/storage.yaml`, located by `--catalog` / `BLIS_CATALOG`); the kernel prices
each tier transfer from that device (its `TierTime`). An explicit `read_bandwidth`,
`write_bandwidth` or `base_latency` on a tier is refused. A class the catalog does not define, or
a missing/malformed table, is a hard error naming the path. KV offload combined with P/D
disaggregation is refused.

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

At `block_size == 1` decode blocks take a guarded allocation path that leaves them unhashed, so
decode-offload is inert there — a degenerate offload block size (real offload block sizes track
the GPU block size). With no `--kv-offload-config`, behavior is unchanged (INV-6).

### Enabling and Disabling Offload

Offload is **off by default**. There is one switch — the presence of `--kv-offload-config`
(or, for the legacy tier, a non-zero `--kv-cpu-blocks`). The two are mutually exclusive.

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
    A node-level memory budget must therefore be divided by TP before it goes in the file. This
    matches vLLM, where `cpu_bytes_to_use` is likewise a per-worker budget.

## Chunked Prefill

Long prefill sequences can cause **head-of-line (HOL) blocking** — a long prefill occupies a whole step, blocking shorter requests from starting.

Chunked prefill splits long prefills into smaller chunks:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --long-prefill-token-threshold 256 \
  --rate 100 --num-requests 500
```

!!! info "Chunked prefill benefits TTFT, not ITL"
    With `--long-prefill-token-threshold=256`, short-request TTFT p99 improves by ~52% in bimodal workloads. But ITL is unaffected (<0.5%) because ~255 of ~256 ITL samples per request are decode-only steps. The benefit is in scheduling new requests, not in token generation speed.

## Batch Formation Parameters

KV cache pressure is directly coupled to batch formation. The scenario pool's `engine.max_num_seqs` (maximum requests in the running batch) and `engine.max_num_batched_tokens` (token budget per step) are the primary capacity knobs, with vLLM's meaning. Reducing them decreases KV cache pressure but also reduces throughput.

## Identifying the KV Pressure Cliff

Preemption rates spike non-linearly once the working set exceeds the KV budget. Sweep offered load against a fixed scenario to find it:

```bash
for rate in 5 10 20 40 80; do
  echo "=== rate=$rate ==="
  ./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
    --scenarios <dir of scenario files> --registry $PWD/blis-registry \
    --rate $rate --num-requests 200 2>/dev/null \
    | grep -E "preemption_count|completed_requests"
done
```

!!! tip "Distribution median drives KV pressure"
    ParetoLogNormal distributions produce *fewer* preemptions than Gaussian despite similar means, because the Pareto component's median (~79 tokens) is much lower than Gaussian's median (~256 tokens). Short requests cycle faster, creating "breathing room" in the KV cache.

## Further Reading

- [Core Engine: KV Cache](../concepts/core-engine.md#kv-cache-management) — internal mechanics
- [Configuration Reference](../reference/configuration.md#kv-cache-configuration) — all KV cache flags
- [Metrics & Results](results.md) — understanding preemption rate, cache hit rate, KV thrashing
