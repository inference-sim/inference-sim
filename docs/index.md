# BLIS — Blackbox Inference Simulator

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by pluggable latency models (data-driven coefficients, analytical roofline, or custom backends).

The simulator is CPU-only, deterministic, and designed for **capacity planning**, **policy optimization research**, and **performance prediction** across model/GPU/TP configurations without requiring real GPUs.

---

## Quick Start

```bash
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o blis main.go
./blis run --model meta-llama/llama-3.1-8b-instruct
```

---

## Key Features

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **Deterministic execution** — same seed produces byte-identical output across runs
- **KV-cache modeling** with prefix caching and tiered GPU+CPU offload
- **Chunked prefill and preemption-aware batch formation**
- **Pluggable latency models** — blackbox (data-driven coefficients) and roofline (analytical FLOPs/bandwidth), with an extensible interface for custom backends
- **Multi-instance cluster simulation** with shared-clock event loop
- **Pluggable routing policies** — round-robin, least-loaded, and composable weighted-scoring with prefix-affinity, queue-depth, and KV-utilization scorers
- **Admission control**, **priority policies**, and **instance schedulers** — each a pluggable policy axis
- **Canonical workload specification** — multi-client YAML DSL with Poisson/Gamma/Weibull/constant arrival processes, 5 distribution types, SLO classes (critical/standard/sheddable/batch/background), prefix groups, cohort dynamics, multimodal, reasoning multi-turn, and composable specs via `blis compose`
- **Rich metrics pipeline** — per-request, per-instance, and cluster-level metrics including TTFT/ITL/E2E distributions, KV cache diagnostics, anomaly detection (priority inversions, HOL blocking), SLO attainment, Jain fairness index, and multi-objective fitness evaluation
- **Decision tracing and counterfactual analysis** with top-k regret computation
- **Hypothesis experimentation framework** for rigorous, reproducible experiments

---

## Architecture Overview

```
Request Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion
                                            ↓              ↓
                                      KV Allocation   Latency Estimation
```

Admission and Routing apply in cluster mode (multi-instance). Single-instance mode skips directly to WaitQueue.

---

## Documentation Guide

| Section | What You'll Find |
|---------|-----------------|
| [Getting Started](getting-started/index.md) | What is BLIS, installation, quick start, capacity planning tutorial |
| [Concepts](concepts/index.md) | System architecture, core engine, glossary, roofline estimation |
| [User Guide](guide/index.md) | Task-oriented guides: routing, admission, scheduling, latency models, KV cache, workloads, cluster, metrics, experimentation |
| [Reference](reference/index.md) | Configuration reference, supported models, workload spec schema |
| [Contributing](contributing/index.md) | Extension recipes, PR workflow, standards, templates |

### Reading Order for Newcomers

1. **[What is BLIS?](getting-started/index.md)** — understand the problem BLIS solves
2. **[Quick Start](getting-started/quickstart.md)** — run your first simulation
3. **[Tutorial: Capacity Planning](getting-started/tutorial.md)** — end-to-end walkthrough
4. **[Glossary](concepts/glossary.md)** — learn BLIS-specific terminology
5. **[User Guide](guide/index.md)** — task-oriented how-to guides

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/inference-sim/inference-sim/blob/main/LICENSE) for details.
