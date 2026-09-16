# Quick Start

Run your first BLIS simulation in 30 seconds.

**Optional:** Set `HF_TOKEN` to access gated models (e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)) and avoid HuggingFace rate limits:

```bash
export HF_TOKEN=your_token_here
```

## Locate the model catalog

Every `blis run` / `blis replay` needs to be told where the **model catalog** is — a
directory holding one subdirectory per model, each with that model's HuggingFace
`config.json`. The repository's own `model_configs/` tree is such a catalog. There is
**no default and no search path**: supply `--catalog <path>` or set `BLIS_CATALOG`
(the flag wins when both are set), or the run is refused.

```bash
export BLIS_CATALOG=$PWD/model_configs   # or pass --catalog on every command
```

## Single-Instance Simulation

```bash
./blis run --model qwen/qwen3-14b --catalog model_configs
```

This runs 100 requests through a single inference instance using the default trained-physics latency model for Qwen3 14B on an H100 GPU with TP=1.

!!! note "First-run HuggingFace fetch"
    If the catalog has no entry for the model, BLIS auto-fetches its `config.json` from HuggingFace (~1 second for public models) into `<catalog>/<model>/`; later runs use that entry. For air-gapped environments, pre-populate `<catalog>/<model>/config.json` before the run.

### Reading the Output

BLIS prints diagnostic logs to stderr and results to stdout. You'll see log lines (prefixed with `INFO` or `WARN`) followed by a `=== Simulation Metrics ===` header and pretty-printed JSON:

**Latency metrics** (all in milliseconds, reported as mean/p90/p95/p99):

| Field | What It Measures |
|-------|-----------------|
| `ttft_mean_ms`, `ttft_p99_ms` | **Time to First Token** — how long until the first output token is generated. Lower is better for interactive use. |
| `e2e_mean_ms`, `e2e_p99_ms` | **End-to-End latency** — total time from request arrival to final output token. |
| `itl_mean_ms`, `itl_p99_ms` | **Inter-Token Latency** — time between consecutive output tokens. Lower means smoother streaming. |
| `scheduling_delay_p99_ms` | Wait time from request arrival until processing begins (includes any queueing). |

**Throughput:**

| Field | What It Measures |
|-------|-----------------|
| `responses_per_sec` | Completed requests per second. |
| `tokens_per_sec` | Output tokens generated per second. |
| `completed_requests` | How many requests finished within the simulation window. |
| `total_input_tokens`, `total_output_tokens` | Total tokens processed across all completed requests. |

**Health indicators:**

| Field | What It Measures |
|-------|-----------------|
| `preemption_count` | Number of times a running request was evicted to make room for others. Non-zero suggests the system is overloaded. |
| `dropped_unservable` | Requests rejected because they were too large for the configured memory or context length. |
| `still_queued`, `still_running` | Requests not yet completed when the simulation ended. Non-zero means the workload outlasted the simulation window. |

## Cluster Mode

Scale to 4 instances with routing:

```bash
./blis run \
  --model qwen/qwen3-14b \
  --num-instances 4 \
  --routing-policy weighted \
  --rate 100 --num-requests 500
```

This simulates a 4-instance cluster receiving 100 requests/second. The `weighted` routing policy uses the default scorer profile (`precise-prefix-cache:2, queue-depth:1, kv-utilization:1`) to distribute requests across instances.

!!! note "Multi-instance output format"
    In cluster mode, BLIS prints one JSON block per instance plus a cluster-level summary (5 blocks total for 4 instances). The cluster summary has `"instance_id": "cluster"`. If piping to `jq`, use `--slurp` to handle multiple JSON objects: `./blis run ... 2>/dev/null | jq --slurp '.[] | select(.instance_id == "cluster")'` to extract the cluster summary.

## Try Different Configurations

```bash
# Higher traffic rate
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 500 --num-requests 2000

# With decision tracing (see where each request was routed)
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace

# With trained-physics mode (recommended for new models)
./blis run --model qwen/qwen3-14b \
  --latency-model trained-physics --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500

# With pure roofline mode (analytical, no learned corrections)
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500
```

## What's Next

- **[Tutorial: Capacity Planning](tutorial.md)** — Full walkthrough: find the right instance count for your workload
- **[Routing Policies](../guide/routing.md)** — Understand and compare routing strategies
- **[Configuration Reference](../reference/configuration.md)** — Complete CLI flag reference
