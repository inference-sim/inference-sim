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
are offloaded too; because BLIS already hashes every completed block prefix-consistently, a later
request on the same instance whose **input contains earlier output tokens** (multi-turn / agentic
workloads) reloads that decode KV from the tiers instead of recomputing it — so the cache hit-rate
reflects the policy. Reuse is single-instance (offload tiers are per-instance and invisible to the
router). With no `--kv-offload-config`, behavior is unchanged (INV-6).

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
