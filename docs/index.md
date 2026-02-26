# BLIS — Blackbox Inference Simulator

A discrete-event simulator for LLM inference serving systems. BLIS models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation — all driven by trained performance coefficients or analytical roofline estimates.

The simulator is CPU-only, deterministic, and designed for **capacity planning**, **policy optimization research**, and **performance prediction** across model/GPU/TP configurations without requiring real GPUs.

---

## Quick Start

```bash
# Build
git clone https://github.com/inference-sim/inference-sim.git
cd inference-sim
go build -o simulation_worker main.go

# Run with default model
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct
```

---

## Key Features

- **Discrete-event simulation** for prefill, decode, and request scheduling
- **Deterministic execution** — same seed produces byte-identical output across runs (INV-6)
- **KV-cache modeling** with prefix caching and tiered GPU+CPU offload
- **Chunked prefill and preemption-aware batch formation**
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

Admission and Routing apply in cluster mode (multi-instance). Single-instance mode skips directly to WaitQueue.

For detailed architecture documentation, see [Cluster Architecture](design/architecture.md) and [Core Engine](design/core-engine.md).

---

## Documentation Guide

| Section | What You'll Find |
|---------|-----------------|
| [Design](design/README.md) | System architecture, core engine, concepts glossary, configuration reference |
| [Standards](standards/rules.md) | Antipattern rules (R1-R20), system invariants (INV-1-8), engineering principles |
| [Process](process/pr-workflow.md) | PR workflow, design document process, macro planning, hypothesis experiments, convergence protocol |
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

All models below have pre-trained alpha/beta coefficients in `defaults.yaml` for blackbox mode. Models with a HuggingFace `config.json` in `model_configs/` additionally support roofline mode.

### Dense Models

| Model | Sizes |
|-------|-------|
| Meta LLaMA 3.1 | 8B |
| Meta LLaMA 3.3 | 70B |
| IBM Granite 3.1 | 8B |
| CodeLlama | 34B |
| Microsoft Phi-4 | 14B |
| Mistral Small (2501) | 24B |
| Mistral Small 3.1 (2503) | 24B |
| NVIDIA LLaMA 3.1 Nemotron | 70B |
| OpenAI GPT-OSS | 20B, 120B |
| Qwen 2.5 | 7B |
| SmolLM3 | 3B |

### MoE Models

| Model | Architecture |
|-------|-------------|
| LLaMA 4 Maverick | 17B, 128 experts |
| LLaMA 4 Scout | 17B, 16 experts |
| Mixtral | 8x7B |

### Quantized Variants

Red Hat AI (`redhatai/`) provides FP8, W4A16, and W8A8 quantized variants for many of the above models, including LLaMA 3.1/3.3/4, Mistral Small 3.1, Phi-4, and Qwen 2.5. See [`defaults.yaml`](https://github.com/inference-sim/inference-sim/blob/main/defaults.yaml) for the full list.

### Roofline-Only Models

Any model with a HuggingFace `config.json` can use roofline mode via `--model-config-folder`. Pre-packaged configs exist for additional architectures (Qwen 2.5 1.5B/3B, Qwen 3 14B, LLaMA 2 7B/70B) in `model_configs/`.

---

## License

This project is licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/inference-sim/inference-sim/blob/main/LICENSE) for details.
