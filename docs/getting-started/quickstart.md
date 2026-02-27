# Quick Start

Run your first BLIS simulation in 30 seconds.

## Single-Instance Simulation

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct
```

This runs 100 requests through a single inference instance using pre-trained coefficients for LLaMA 3.1 8B on an H100 GPU with TP=2.

### Reading the Output

The JSON output on stdout contains:

```json
{
  "ttft_mean_ms": 12.34,
  "ttft_p99_ms": 45.67,
  "e2e_mean_ms": 1234.56,
  "e2e_p99_ms": 2345.67,
  "itl_mean_ms": 8.91,
  "responses_per_sec": 5.67,
  "completed_requests": 100
}
```

| Metric | What It Measures |
|--------|-----------------|
| **TTFT** (Time To First Token) | Latency from request arrival to first output token — measures responsiveness |
| **E2E** (End-to-End) | Total latency from arrival to final token — measures total request duration |
| **ITL** (Inter-Token Latency) | Time between consecutive output tokens — measures streaming smoothness |
| **responses_per_sec** | Throughput — requests completed per second |

## Cluster Mode

Scale to 4 instances with routing:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 \
  --routing-policy weighted \
  --rate 100 --num-requests 500
```

This simulates a 4-instance cluster receiving 100 requests/second. The `weighted` routing policy uses the default scorer profile (`prefix-affinity:3, queue-depth:2, kv-utilization:2`) to distribute requests across instances.

## Try Different Configurations

```bash
# Higher traffic rate
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 500 --num-requests 2000

# With decision tracing (see where each request was routed)
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 100 --num-requests 500 \
  --trace-level decisions --summarize-trace

# With roofline mode (no pre-trained coefficients needed)
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2 \
  --num-instances 4 --rate 100 --num-requests 500
```

## What's Next

- **[Tutorial: Capacity Planning](tutorial.md)** — Full walkthrough: find the right instance count for your workload
- **[Routing Policies](../guide/routing.md)** — Understand and compare routing strategies
- **[Configuration Reference](../reference/configuration.md)** — Complete CLI flag reference
