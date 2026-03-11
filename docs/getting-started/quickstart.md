# Quick Start

Run your first BLIS simulation in 30 seconds.

## Single-Instance Simulation

```bash
./blis run --model qwen/qwen3-14b
```

This runs 100 requests through a single inference instance using roofline mode (analytical estimation) for Qwen3 14B on an H100 GPU with TP=1. The model config is auto-fetched from HuggingFace on first use.

### Reading the Output

BLIS prints deterministic JSON to stdout (diagnostic logs go to stderr). Pipe to `jq` for formatting:

```bash
./blis run --model qwen/qwen3-14b | jq .
```

**Latency metrics** (all in milliseconds, with mean/p90/p95/p99 variants):

| Field | What It Measures |
|-------|-----------------|
| `ttft_mean_ms`, `ttft_p99_ms` | **Time to First Token** — arrival to first output token. Measures responsiveness for interactive use. |
| `e2e_mean_ms`, `e2e_p99_ms` | **End-to-End latency** — arrival to final token. Measures total request duration. |
| `itl_mean_ms`, `itl_p99_ms` | **Inter-Token Latency** — time between consecutive output tokens. Measures streaming smoothness. |
| `scheduling_delay_p99_ms` | Time from arrival to entering the running batch (includes queueing). |

**Throughput and accounting:**

| Field | What It Measures |
|-------|-----------------|
| `responses_per_sec` | Request throughput — completed requests per simulated second. |
| `tokens_per_sec` | Output token throughput. |
| `completed_requests` | Requests that finished within the simulation horizon. |
| `injected_requests` | Total requests that entered the simulator (= completed + still_queued + still_running + dropped). |
| `total_input_tokens`, `total_output_tokens` | Aggregate token counts across all completed requests. |

**Health indicators:**

| Field | What It Measures |
|-------|-----------------|
| `preemption_count` | KV cache evictions — indicates GPU memory pressure. Non-zero means the batch is too large for available KV blocks. |
| `dropped_unservable` | Requests rejected because input tokens exceed KV cache capacity or `--max-model-len`. |
| `kv_allocation_failures` | Failed KV block allocations (triggers preemption). |
| `still_queued`, `still_running` | Requests not yet completed at simulation end — non-zero means the horizon was reached before all requests finished. |

## Cluster Mode

Scale to 4 instances with routing:

```bash
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 \
  --routing-policy weighted \
  --rate 100 --num-requests 500
```

This simulates a 4-instance cluster receiving 100 requests/second. The `weighted` routing policy uses the default scorer profile (`prefix-affinity:3, queue-depth:2, kv-utilization:2`) to distribute requests across instances.

## Try Different Configurations

```bash
# Higher traffic rate
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 500 --num-requests 2000

# With decision tracing (see where each request was routed)
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace

# With trained-roofline mode (recommended for new models, 7% MAPE)
./blis run --model qwen/qwen3-14b \
  --latency-model trained-roofline --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500

# With pure roofline mode (analytical, no learned corrections)
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500
```

> **Tip:** Roofline, trained-roofline, and cross-model modes auto-fetch model configs from HuggingFace. Set `HF_TOKEN` to access gated models (e.g., LLaMA) and avoid rate limits:
>
> ```bash
> export HF_TOKEN=your_token_here
> ```
>
> See [HuggingFace access tokens](https://huggingface.co/docs/hub/en/security-tokens) to create a token.

## What's Next

- **[Tutorial: Capacity Planning](tutorial.md)** — Full walkthrough: find the right instance count for your workload
- **[Routing Policies](../guide/routing.md)** — Understand and compare routing strategies
- **[Configuration Reference](../reference/configuration.md)** — Complete CLI flag reference
