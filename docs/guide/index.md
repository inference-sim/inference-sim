# User Guide

Task-oriented guides for using BLIS effectively. Each guide covers a specific feature with practical CLI examples and expected output.

## Guides

| Guide | When to Use |
|-------|-------------|
| [Routing Policies](routing.md) | Choosing and configuring how requests are distributed across instances |
| [KV Cache & Memory](kv-cache.md) | Tuning GPU/CPU memory allocation, prefix caching, and chunked prefill |
| [Roofline Mode](roofline.md) | Running simulations without pre-trained coefficients |
| [Workload Specifications](workloads.md) | Defining multi-client traffic patterns with YAML |
| [Cluster Simulation](cluster.md) | Running multi-instance simulations with admission and routing |
| [Interpreting Results](results.md) | Understanding JSON output, metrics, anomaly counters, and fitness scores |
| [Hypothesis Experimentation](experimentation.md) | Running rigorous, reproducible experiments with the `/hypothesis-test` skill |

## Reading Paths

**Capacity planning:** [Quick Start](../getting-started/quickstart.md) → [Tutorial](../getting-started/tutorial.md) → [Cluster Simulation](cluster.md) → [Interpreting Results](results.md)

**Routing optimization:** [Routing Policies](routing.md) → [Cluster Simulation](cluster.md) → [Interpreting Results](results.md)

**Memory tuning:** [KV Cache & Memory](kv-cache.md) → [Interpreting Results](results.md)

**New model evaluation:** [Roofline Mode](roofline.md) → [Workload Specifications](workloads.md) → [Interpreting Results](results.md)

**Research:** [Hypothesis Experimentation](experimentation.md) → [Interpreting Results](results.md)
