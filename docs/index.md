# BLIS — Blackbox Inference Simulator

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients or analytical roofline estimates.

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
- **Two latency estimation modes**: blackbox (data-driven) and roofline (analytical)
- **Multi-instance cluster simulation** with shared-clock event loop
- **Pluggable routing policies**: round-robin, least-loaded, weighted-scoring, prefix-affinity
- **Admission control**, **priority policies**, and **instance schedulers**
- **ServeGen-informed workload generation** with multi-client traffic classes
- **Decision tracing and counterfactual analysis** with top-k regret computation
- **Fitness evaluation** with weighted multi-objective scoring
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
| [User Guide](guide/index.md) | Task-oriented guides: routing, KV cache, roofline, workloads, cluster, results, experimentation |
| [Concepts](concepts/index.md) | System architecture, core engine, glossary, roofline estimation |
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
