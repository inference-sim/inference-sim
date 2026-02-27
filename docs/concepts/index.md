# Concepts

These pages explain BLIS's architecture and core mechanisms at the level needed to understand the system without reading source code. For task-oriented how-to guides, see the [User Guide](../guide/index.md).

## Concept Pages

| Page | Description |
|------|-------------|
| [Glossary](glossary.md) | Definitions of BLIS-specific terminology |
| [Cluster Architecture](architecture.md) | Multi-instance simulation: admission, routing, scorer composition, snapshot freshness, shared-clock event loop |
| [Core Engine](core-engine.md) | Single-instance DES engine: event queue, Step() phases, request lifecycle, batch formation, KV cache, latency models |
| [Roofline Estimation](roofline.md) | Analytical GPU step time estimation without training data |

## Diagrams

| Diagram | Description |
|---------|-------------|
| ![Cluster Data Flow](diagrams/clusterdataflow.png) | End-to-end cluster pipeline: request arrival through metrics output |
| ![Request Lifecycle](diagrams/requestlifecycle.png) | Request state machine: states, transitions, and metric recording points |
| ![Event Processing](diagrams/eventprocessingloop.png) | DES event loop: min-heap queue, clock advancement, Step() decomposition |
| ![Scoring Pipeline](diagrams/scoringpipeline.png) | Weighted scorer composition: per-scorer normalization, weight multiplication, argmax selection |

## Reading Order

For newcomers to BLIS:

1. Start with **[Glossary](glossary.md)** to learn BLIS-specific terminology
2. Read **[Core Engine](core-engine.md)** to understand the DES architecture and single-instance simulation
3. Read **[Cluster Architecture](architecture.md)** to understand multi-instance orchestration
4. Consult **[Configuration Reference](../reference/configuration.md)** when running experiments
5. See **[Extension Recipes](../contributing/extension-recipes.md)** when adding new policies or features
