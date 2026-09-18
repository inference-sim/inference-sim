# What is BLIS?

**BLIS** (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics, scheduling policies, and token generation — all without requiring real GPUs.

---

## Why Simulate Inference Serving?

Deploying LLM inference at scale requires answering capacity planning questions that are expensive to answer with real hardware:

- **How many instances do I need** to serve 1,000 requests/second at p99 TTFT < 200ms?
- **Which routing policy** minimizes tail latency for my workload mix?
- **How much KV cache memory** do I need before preemptions degrade throughput?
- **What happens at 2x traffic** — does latency degrade gracefully or catastrophically?

Running these experiments on real GPUs costs thousands of dollars and takes days. BLIS answers them in seconds on a laptop.

## Who Should Use BLIS

| Audience | Use Case |
|----------|----------|
| **Capacity planners** | Determine instance counts, GPU memory, and TP configurations before procurement |
| **Platform engineers** | Compare routing policies, tune scorer weights, evaluate admission control strategies |
| **Researchers** | Run controlled experiments on scheduling, batching, and caching algorithms |
| **Developers** | Validate new policies against existing ones before deploying to production |

## What BLIS Is Not

!!! warning "Setting expectations"
    - **Not a benchmark** — BLIS simulates serving behavior, it does not generate real GPU load
    - **Not primarily a load generator** — BLIS focuses on simulation. Real-mode traffic generation against OpenAI-compatible endpoints is available but experimental. For production load testing, use tools like `inference-perf` or `genai-perf`

## Key Features

See the [Home page feature list](../index.md#key-features) for the full capabilities catalog, including the workload specification DSL, metrics pipeline, latency model backends, and policy framework.

## Next Steps

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — Build BLIS from source
- :material-rocket-launch: **[Quick Start](quickstart.md)** — Run your first simulation in 30 seconds
- :material-chart-line: **[Tutorial: Capacity Planning](tutorial.md)** — End-to-end walkthrough
- :material-book-open-variant: **[User Guide](../guide/index.md)** — Task-oriented how-to guides

</div>
