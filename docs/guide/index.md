# User Guide

Task-oriented guides for using BLIS effectively. Each guide covers a specific feature with practical CLI examples and expected output.

## Guides

| Guide | When to Use |
|-------|-------------|
| [Routing Policies](routing.md) | Choosing and configuring how requests are distributed across instances |
| [Admission Control](admission.md) | Rate-limiting and traffic shaping at the cluster gateway |
| [Scheduling & Priority](scheduling.md) | Controlling request processing order within each instance |
| [Latency Models](latency-models.md) | Choosing between blackbox (data-driven) and roofline (analytical) step time estimation |
| [KV Cache & Memory](kv-cache.md) | Tuning GPU/CPU memory allocation, prefix caching, and chunked prefill |
| [Workload Specifications](workloads.md) | Defining multi-client traffic patterns with YAML |
| [Cluster Simulation](cluster.md) | Running multi-instance simulations with the full pipeline |
| [Metrics & Results](results.md) | Understanding JSON output, metrics, anomaly counters, and fitness scores |
| [Hypothesis Experimentation](experimentation.md) | Running rigorous, reproducible experiments with the `/hypothesis-experiment` skill |
| [Skills & Plugins](skills-and-plugins.md) | Claude Code skills, plugin marketplaces, and workflow tooling |

## Reading Paths

**Capacity planning:** [Quick Start](../getting-started/quickstart.md) → [Tutorial](../getting-started/tutorial.md) → [Cluster Simulation](cluster.md) → [Metrics & Results](results.md)

**Routing optimization:** [Routing Policies](routing.md) → [Cluster Simulation](cluster.md) → [Metrics & Results](results.md)

**Memory tuning:** [KV Cache & Memory](kv-cache.md) → [Metrics & Results](results.md)

**New model evaluation:** [Latency Models](latency-models.md) → [Workload Specifications](workloads.md) → [Metrics & Results](results.md)

**Research:** [Hypothesis Experimentation](experimentation.md) → [Metrics & Results](results.md)
