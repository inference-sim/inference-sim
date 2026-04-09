---
date: 2026-04-09
draft: true
authors:
  - dipanwita
  - mert
  - jing
  - nick
  - michael
  - asser
  - vishakha
  - srini
  - fabio
categories:
  - Architecture
  - Deep Dives
---

# Building Trust: The Physics of High-Fidelity Inference Simulation

Imagine testing routing policies, autoscaling strategies, and hardware configurations without touching production. No risk. No downtime. Just answers. That is the promise of simulation — but only if it is accurate enough to trust.

A simple queueing model predicts 50ms time-to-first-token. Production measures 200ms. The difference reveals how much complexity hides beneath the surface. Capacity decisions are million-dollar bets—H100 vs A100, tensor parallelism 4 vs 8—and intuition fails at this scale.

<!-- more -->

## The Right Physics, Not Everything

Building a trustworthy simulator is not about modeling everything — it is about modeling the right physics. The batch dynamics that couple request latencies. The KV cache pressure that triggers preemption. The prefill-decode handoffs that trade network costs for throughput. Miss any of these, and your predictions diverge from reality.

BLIS achieves this through discrete-event simulation that models the actual physics — the mechanisms that determine latency in real systems: how requests couple through shared batch steps, how KV cache pressure triggers preemption cascades, how prefill-decode disaggregation trades network transfer costs for hardware specialization. Discrete-event simulation lets BLIS run orders of magnitude faster than real-time by jumping directly from event to event (request arrivals, batch completions, routing decisions) rather than ticking through every microsecond. The entire simulation runs on CPU — no GPUs required. Step times come from analytical roofline models (compute vs memory bottlenecks derived from model architecture and hardware specs) corrected with coefficients trained on real vLLM production traces. The result: evaluate hours of production traffic in seconds, with single-digit percent accuracy. Fast enough for rapid iteration, accurate enough to trust for production decisions.

This article shows what it takes to build that level of fidelity—from token generation physics to distributed orchestration. Let us follow a request's 50-millisecond journey through the system to see where every millisecond of that complexity lives.

## A Request's Journey: The Hidden Complexity

A user hits enter. Fifty milliseconds later, the first token appears. What happened in between? Three architectural layers working together: the inference engine (vLLM), the data plane (cluster orchestration), and the control plane (autoscaling). Model them all with fidelity, or your capacity decisions will be wrong.

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Engine (vLLM)"]
        Sched[Scheduling]
        KV[KV Cache]
        Batch[Batch Formation]
        Step[Step Execution]
        Sched --> KV --> Batch --> Step
    end

    subgraph Layer2["Layer 2: Data Plane"]
        Admit[Admission]
        Route[Routing]
        Flow[Flow Control]
        Admit --> Flow --> Route
    end

    subgraph Layer3["Layer 3: Control Plane"]
        Monitor[Monitor Metrics]
        Decide[Scale Decision]
        Actuate[Add/Remove Instances]
        Monitor --> Decide --> Actuate
    end

    Request([Request]) --> Layer2
    Layer2 --> Layer1
    Layer1 --> Response([Response])
    Layer3 -.-> Layer2
    Layer1 -.->|metrics| Layer3

    style Layer1 fill:#e1f5ff
    style Layer2 fill:#fff4e1
    style Layer3 fill:#ffe1e1
```

### Layer 1: The Engine (vLLM)

Most people misunderstand how vLLM works. It does not process requests individually—it processes batches in steps. One step equals one GPU forward pass. All requests in the batch execute together, and the step time is determined by the slowest operation.

Consider four requests: three generate single output tokens (2ms, memory-bound), while one processes a 512-token prompt (20ms, compute-bound). The step time is 20ms. All four wait, even though three could finish in 2ms alone. Ten decode requests do not take "10 × 2ms"—they take one 2ms batch step. Get this wrong, and throughput predictions are 5-10x off.

BLIS replicates vLLM's mechanisms exactly. Priority scheduling for critical requests. Block-level KV cache with prefix reuse and preemption. Continuous batching where requests join and leave mid-flight. No approximations.

For step timing, BLIS computes two bottlenecks: compute (FLOPs / GPU_TFLOPS) and memory (bytes / bandwidth). Step time is the maximum. A 512-token prefill on H100: compute-bound at 20ms. Decode reading KV cache: memory-bound at 2ms. BLIS applies learned corrections:

$$
t_{\text{step}} = \sum_{i} \beta_i \cdot \phi_i(\text{batch}, \text{model}, \text{hardware})
$$

where $\phi_i$ are roofline basis functions and $\beta_i$ are coefficients trained on real vLLM traces. Model architecture comes from HuggingFace, hardware specs from datasheets. CPU-only, microsecond-fast, single-digit percent accurate.

Batch composition evolves constantly. Small decode batch: 2ms per token. Long prompt joins: everyone waits 20ms. Prompt finishes: back to 2ms. Request completes: 1.8ms with smaller batch. Latencies couple through batching—not independent.

BLIS models this through discrete-event simulation: one step event per batch operation, membership updating after each completion. This is the foundation for accurate prediction. But a single vLLM instance is only part of production systems. Clusters add orchestration complexity.

### Layer 2: The Data Plane (Cluster Orchestration)

[To be written - admission, routing, signal staleness, P/D orchestration]

### Layer 3: The Control Plane (Autoscaling)

[To be written - WVA pipeline, feedback delays]

### The Complete Journey

[To be written - integration of all three layers with end-to-end trace]

## BLIS in Action: A Real Scenario

[To be written - PD disaggregation example with real numbers]

## From Modeling to Validation

[To be written - recap + tease validation article]

---

*This is the first article in a series on BLIS's architecture. Next: **Validating Against Ground Truth** - how BLIS achieves single-digit percent error on real workloads.*
