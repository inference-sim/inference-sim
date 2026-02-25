# BLIS Design Documentation

This directory contains the public-facing design documentation for BLIS (Blackbox Inference Simulator). These pages explain BLIS's architecture, concepts, and configuration at the level needed to understand the system without reading source code.

## Design Pages

### Architecture & Internals

| Page | Description |
|------|-------------|
| [Cluster Architecture](architecture.md) | Multi-instance simulation: admission, routing, scorer composition, snapshot freshness, shared-clock event loop |
| [Core Engine](core-engine.md) | Single-instance DES engine: event queue, Step() phases, request lifecycle, batch formation, KV cache, latency models |
| [Concepts & Glossary](concepts.md) | Definitions of BLIS-specific terminology |
| [Configuration Reference](configuration.md) | All CLI flags, sub-config types, defaults.yaml behavior, workload modes |

### Foundational Documents

| Page | Description |
|------|-------------|
| [Simulation Approach](../approach.md) | Research-oriented overview: DES methodology, latency model mathematics, calibration process |
| [Roofline Estimation](../roofline.md) | Analytical GPU step time estimation without training data |

### For Contributors

| Page | Description |
|------|-------------|
| [Extension Recipes](../extension-recipes.md) | Step-by-step guides for adding policies, scorers, KV tiers, trace records, and metrics |
| [Engineering Standards](../standards/) | Antipattern rules (R1-R20), system invariants (INV-1 through INV-8), BDD/TDD principles |
| [Process Workflows](../process/) | PR workflow, design document process, hypothesis experiment protocol |

## Diagrams

| Diagram | Description |
|---------|-------------|
| ![Cluster Data Flow](diagrams/clusterdataflow.png) | End-to-end cluster pipeline: request arrival through metrics output |
| ![Request Lifecycle](diagrams/requestlifecycle.png) | Request state machine: states, transitions, and metric recording points |
| ![Event Processing](diagrams/eventprocessingloop.png) | DES event loop: min-heap queue, clock advancement, Step() decomposition |
| ![Scoring Pipeline](diagrams/scoringpipeline.png) | Weighted scorer composition: per-scorer normalization, weight multiplication, argmax selection |

## Reading Order

For newcomers to BLIS:

1. Start with **[Simulation Approach](../approach.md)** for the research motivation and latency model math
2. Read **[Concepts & Glossary](concepts.md)** to learn BLIS-specific terminology
3. Read **[Core Engine](core-engine.md)** to understand single-instance simulation
4. Read **[Cluster Architecture](architecture.md)** to understand multi-instance orchestration
5. Consult **[Configuration Reference](configuration.md)** when running experiments
6. See **[Extension Recipes](../extension-recipes.md)** when adding new policies or features
