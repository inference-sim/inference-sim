# BLIS — Blackbox Inference Simulator

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients or analytical roofline estimates.

The simulator is CPU-only, deterministic, and designed for **capacity planning**, **policy optimization research**, and **performance prediction** across model/GPU/TP configurations without requiring real GPUs.

---

## Quick Start

```bash
# Build
git clone git@github.com:inference-sim/inference-sim.git
cd inference-sim
go build -o simulation_worker main.go

# Run with default model
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct
```

---

## Key Features

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **KV-cache modeling** with prefix caching, chunked prefill, and tiered GPU+CPU offload
- **Two latency estimation modes**: blackbox (data-driven) and roofline (analytical)
- **Multi-instance cluster simulation** with shared-clock event loop
- **Pluggable routing policies**: round-robin, least-loaded, weighted-scoring, prefix-affinity
- **Admission control**, **priority policies**, and **instance schedulers**
- **ServeGen-informed workload generation** with multi-client traffic classes
- **Decision tracing and counterfactual analysis** with top-k regret computation
- **Fitness evaluation** with weighted multi-objective scoring

---

## Architecture Overview

```
Request Arrival → Admission → Routing → WaitQueue → Batch Formation → Step Execution → Completion
                                            ↓              ↓
                                      KV Allocation   Latency Estimation
```

For detailed architecture documentation, see [Cluster Architecture](design/architecture.md) and [Core Engine](design/core-engine.md).

---

## Documentation Guide

| Section | What You'll Find |
|---------|-----------------|
| [Design](design/README.md) | System architecture, core engine, concepts glossary, configuration reference |
| [Standards](standards/rules.md) | Antipattern rules (R1-R20), system invariants (INV-1-8), engineering principles |
| [Process](process/pr-workflow.md) | PR workflow, design document process, hypothesis experiments |
| [Templates](templates/design-guidelines.md) | Design guidelines, macro/micro plan templates, hypothesis template |
| [Extension Recipes](extension-recipes.md) | Step-by-step guides for adding policies, scorers, KV tiers, and more |
| [Contributing](contributing.md) | Engineering standards, development workflow, and getting started |

### Reading Order for Newcomers

1. **[Concepts & Glossary](design/concepts.md)** — learn BLIS-specific terminology
2. **[Core Engine](design/core-engine.md)** — understand the DES architecture and single-instance simulation
3. **[Cluster Architecture](design/architecture.md)** — understand multi-instance orchestration
4. **[Configuration Reference](design/configuration.md)** — when running experiments
5. **[Extension Recipes](extension-recipes.md)** — when adding new policies or features

---

## Supported Models

### Dense Models
- LLaMA 3.x (8B, 70B variants)
- Qwen 2.5 (1.5B - 72B)
- Mistral (7B, Small 24B)
- Phi-4, CodeLlama, Granite

### MoE Models
- Mixtral 8x7B
- Additional MoE models supported via HuggingFace config.json

See [`defaults.yaml`](https://github.com/inference-sim/inference-sim/blob/main/defaults.yaml) for the full list of pre-trained model configurations.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/inference-sim/inference-sim/blob/main/LICENSE) for details.
