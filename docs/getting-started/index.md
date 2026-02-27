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
    - **Not a load generator** — use tools like `inference-perf` or `genai-perf` for real traffic generation
    - **Not cycle-accurate** — BLIS uses trained regression coefficients or analytical roofline estimates, not GPU microarchitecture simulation. It trades cycle-accuracy for speed and configurability
    - **Homogeneous instances only** — all simulated instances share the same configuration (model, GPU, TP). Mixed-fleet modeling is not yet supported

## Key Capabilities

- **Deterministic execution** — same seed produces byte-identical output across runs, enabling controlled experiments
- **Two latency modes** — blackbox (data-driven coefficients) and roofline (analytical FLOPs/bandwidth)
- **Pluggable policies** — routing, admission, scheduling, and priority policies are all configurable
- **KV cache modeling** — block allocation, prefix caching, tiered GPU+CPU offload, chunked prefill
- **Multi-client workloads** — model heterogeneous traffic with SLO classes, arrival processes, and token distributions
- **Decision tracing** — log every routing decision with counterfactual regret analysis
- **Hypothesis experimentation** — built-in framework for rigorous, reproducible experiments

## Next Steps

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** — Build BLIS from source
- :material-rocket-launch: **[Quick Start](quickstart.md)** — Run your first simulation in 30 seconds
- :material-chart-line: **[Tutorial: Capacity Planning](tutorial.md)** — End-to-end walkthrough
- :material-book-open-variant: **[User Guide](../guide/index.md)** — Task-oriented how-to guides

</div>
