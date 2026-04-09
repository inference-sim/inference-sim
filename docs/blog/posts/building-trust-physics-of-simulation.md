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

# The Physics of High-Fidelity Inference Simulation

Every capacity decision in LLM inference carries real stakes. Choosing the wrong GPU type or tensor parallelism degree means overspending by millions or underdelivering on latency SLOs, while testing a new routing policy on live traffic risks cascading bugs across the entire fleet.

What does it take to build a simulator accurate enough to guide these decisions? The challenge lies in capturing the right mechanisms. Inference engines process batches in lockstep where all requests wait for the slowest operation, KV cache fills trigger preemptions, and a single long prompt stalls dozens of short decodes. When these couplings are not modeled, predictions diverge from reality - a back-of-the-envelope model might predict 50ms time-to-first-token while production measures 200ms.

<!-- more -->

## Building Fidelity from First Principles

[BLIS](https://github.com/inference-sim/inference-sim) (Blackbox Inference Simulator) models inference serving through discrete-event simulation, advancing from event to event rather than stepping through continuous time. This approach runs orders of magnitude faster than real-time, requires no GPUs, and evaluates hours of production traffic in seconds.

BLIS models the mechanisms that determine latency: continuous batching, KV cache pressure, prefill-decode competition, to predict production behavior accurately. This fidelity enables **capacity planning** and **configuration search**: determining instance count, GPU type, TP degree, routing weights, and admission thresholds. Without modeling these couplings, planners predict linear scaling where production saturates or miss SLO violations from batch interference.

By modeling production systems ([vLLM](https://github.com/vllm-project/vllm), [llm-d](https://llm-d.ai)) behavior, BLIS enables safe experimentation before deployment:

- **Routing policies** — Test new scorer combinations and weights
- **Admission control** — Explore saturation thresholds and flow control strategies
- **Capacity planning** — Compare model/GPU/TP configurations
- **Workload analysis** — Evaluate architecture changes against realistic traffic

Physics-based dynamics with learnable latency components generalize across model architectures and hardware while maintaining production fidelity - meaning you can test new configurations on a laptop in seconds without needing production infrastructure. This rapid iteration enables projects like [ADRS](https://sky.cs.berkeley.edu/project/adrs/) (AI-Driven Research Systems) to develop and validate new serving policies and algorithms through fast simulation loops before production deployment.

This article walks through what it takes to build that level of fidelity — from token batching physics to distributed orchestration, by following a request's end-to-end journey through the system to see where every millisecond of complexity originates.

## A Request's Journey: The Hidden Complexity

A user hits enter, and fifty milliseconds later the first token appears. What happened in between? Three architectural layers working together: the inference engine (vLLM), the data plane (cluster orchestration), and the control plane (autoscaling), all of which high-fidelity simulation must model.

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Engine (vLLM)"]
        Sched[Scheduling]
        KV[KV Cache]
        Batch[Batch Formation]
        Step[Forward Pass]
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
```

### Layer 1: The Engine (vLLM)

> **TL;DR:** Batched execution couples requests together - a heavy prompt in the batch slows down fast decodes running alongside it. BLIS models the full vLLM pipeline (continuous batching, request scheduling and preemption, KV cache pressure, chunked prefill) and predicts forward pass timing using a generalizable model that runs on CPUs without needing real GPUs.

The inference engine does not process requests individually. It processes them in continuously evolving batches. A **step** is one GPU forward pass that advances every request in the batch, either processing prompt tokens (prefill) or generating the next output token (decode). The slowest operation determines when the step completes.

Why does this matter? Consider a batch with three requests decoding single tokens (fast, memory-bound) and one request processing a 512-token prompt (slow, compute-bound). Everyone waits for the slowest. This is not an edge case - batch composition constantly shifts as new requests arrive and completed ones leave.

**What BLIS captures.** vLLM's complexity comes from continuous batching (requests join and leave mid-flight), mixed prefill-decode execution (fast decode waits for slow prefill), block-level KV cache management (prefix reuse, preemption, CPU offloading when GPU memory fills), and chunked prefill (breaking large prompts into smaller pieces). BLIS models all of these mechanisms because they determine when requests complete.

**How BLIS predicts step time without GPUs.** BLIS uses a trained model that combines physics-based basis functions with learned corrections:

$$
t_{\text{step}} = \sum_{i} \beta_i \cdot \phi_i(\text{batch}, \text{LLM}, \text{hardware})
$$

where $\phi_i$ are basis functions that capture computational physics (how batch size, sequence length, and LLM architectures affect compute and memory bandwidth), and $\beta_i$ are coefficients trained on real vLLM traces that correct for hardware-specific bottlenecks. This approach generalizes across LLM architectures, hardware configurations, and tensor parallelism degrees, enabling seamless experimentation with any model-GPU-TP combination without per-configuration calibration. Accurate forward pass predictions drive accurate end-to-end latency metrics.

### Layer 2: The Data Plane (Cluster Orchestration)

> **TL;DR:** Production clusters run multiple vLLM instances behind a routing gateway. BLIS models llm-d's GIE architecture: composable weighted routing, token bucket admission control, in-flight request tracking, configurable cache signal staleness, and prefill/decode disaggregation. Pluggable interfaces for admission policies, routing scorers, and disaggregation deciders enable algorithm discovery - test new serving policies without writing production code.

Production systems run multiple vLLM instances behind a gateway layer. BLIS models the data plane through pluggable interfaces for admission policies, saturation detectors, routing scorers, disaggregation deciders, so you can bring your own custom algorithms and test them against production workloads without writing production code or risking live traffic!

```mermaid
graph LR
    A[Request Arrives] --> B{Admission & Flow Control}
    B -->|Rejected| X[Drop]
    B -->|Queued| C[Routing]
    C --> D{Disaggregation?}
    D -->|No| E[vLLM Processing]
    D -->|Yes| F[Prefill Pool]
    F -->|KV Transfer| G[Decode Pool]
```

**Admission and flow control** determine whether requests enter the system and when they dispatch. BLIS models llm-d's GIE (Gateway Inference Engine) architecture: token bucket rate limiting prevents queue explosion during spikes, and a gateway queue holds requests when the cluster is saturated, releasing them only when capacity opens up. This late binding prevents pile-on where burst arrivals flood the same instance.

**Routing** assigns each request to an instance by scoring on weighted signals - prefix cache hits, queue depth, KV utilization. The challenge: burst arrivals cause all routing decisions to see the same stale state and pick the same "best" instance. BLIS models in-flight tracking (counting already-dispatched requests) and signal staleness (cache state queries a 2-second-old snapshot, matching llm-d's ZMQ propagation delay).

**Prefill/decode disaggregation** separates compute-bound prefill from memory-bound decode onto dedicated GPU pools, allowing each to be sized for its bottleneck. Requests process prefill first, then transfer their KV cache over the network to a decode instance. BLIS models the full pipeline: prefill routing, KV transfer, decode routing, and fair-share bandwidth contention when multiple transfers run concurrently.

### Layer 3: The Control Plane (Autoscaling)

> **TL;DR:** Real autoscaling experiments are expensive - feedback loops spanning minutes (HPA scrapes, pod scheduling, VM provisioning, model loading) require 30+ minutes and 10+ GPU replicas per test. BLIS models llm-d's WVA (Workload Variant Autoscaler) four-stage pipeline with pluggable Collector/Analyzer/Optimizer/Actuator interfaces, compressing experiments to seconds on a laptop.

Autoscaling dynamically adjusts the number of running instances to match demand, adding capacity during traffic spikes and removing it when load drops to avoid paying for idle GPUs. In production systems, this balance between meeting SLOs and controlling costs happens automatically through feedback loops spanning minutes—HPA scrapes, Kubernetes pod scheduling, VM provisioning, and model weight loading all contribute latency before a new replica begins serving. A single experiment needs 30+ minutes of real traffic across 10+ GPU replicas. Each configuration burns real GPU-hours.

**What BLIS captures.** BLIS models llm-d's WVA four-stage pipeline — Collect, Analyze, Optimize, Actuate, with pluggable interfaces. **Collector** observes per-replica metrics, **Analyzer** detects saturation and emits scaling signals, **Optimizer** decides which GPU types to add/remove respecting multi-model inventory constraints, and **Actuator** applies decisions with configurable delay.

**Why simulate?** BLIS compresses 30-minute experiments into seconds on a laptop, enabling rapid iteration without burning production GPU-hours. Researchers can sweep scaling thresholds to find optimal trigger points, compare analyzer strategies under identical workloads, and test multi-model scenarios where scaling one model's replicas steals GPUs from another. Each pluggable interface becomes a research hook - swap in a cost-aware Optimizer that prioritizes cheaper GPU types, or test an Analyzer that predicts load spikes from traffic patterns. The result: discover and validate better autoscaling policies before deployment, with full control over feedback delays, provisioning latencies, and actuation lags that determine real-world scaling behavior.

## BLIS in Action: A Real Scenario

Consider a production decision: should you enable precise prefix cache routing, which queries actual KV cache state but incurs 2-second staleness, or stick with simpler load-balanced routing that sees fresh queue depths? The trade-off matters—prefix-aware routing could reduce TTFT by 20-30% when prompts share prefixes, but stale cache signals might cause pile-on during bursts.

Testing this in production is expensive. You'd need to:
- Deploy both configurations to separate GPU pools
- Run 30+ minutes of representative traffic per config
- Control for workload variance (same arrival patterns, prompt distributions)
- Repeat across multiple load levels to understand behavior under saturation

With BLIS, the experiment takes seconds:

```bash
# Capture production workload characteristics
blis observe --server-url https://prod.api --model qwen/qwen3-14b \
  --workload chatbot --rate 50 --trace-header workload.yaml

# Compare routing policies
blis run --trace-header workload.yaml --model qwen/qwen3-14b \
  --routing-policy weighted \
  --routing-scorers "queue-depth:1,kv-utilization:1,load-balance:1"

blis run --trace-header workload.yaml --model qwen/qwen3-14b \
  --routing-policy weighted \
  --routing-scorers "precise-prefix-cache:2,queue-depth:1,kv-utilization:1" \
  --cache-signal-delay 2000000  # 2s staleness (µs)
```

The simulator reveals the trade-off space: prefix-aware routing wins at moderate load (20% TTFT improvement) but degrades during bursts when stale signals cause all requests with similar prefixes to pile onto the same instance. Load-balanced routing is more robust but misses optimization opportunities when cache hits are available.

This insight drives an informed decision: use prefix-aware routing with a lower staleness threshold (500ms instead of 2s) or implement hybrid strategies that fall back to load balancing during detected bursts. Testing these refinements in BLIS takes minutes. Rolling them out to production without simulation could take weeks of A/B testing across GPU pools.

**What this enables:**

- **Policy optimization** — Test 10 routing configurations in minutes, not weeks of production A/B tests
- **Capacity planning** — Determine instance count and GPU type for target throughput before provisioning
- **Architecture experiments** — Compare disaggregated vs. unified serving, different TP degrees, multi-model GPU sharing
- **Algorithm discovery** — Validate novel policies (cost-aware routing, predictive autoscaling) in a safe sandbox

When your simulator models engine physics (vLLM's batch-step execution), data plane coordination (routing with real-world staleness), and control plane delays (autoscaling feedback loops), you can make architectural decisions based on simulation. That's the unlock.

## From Modeling to Validation

We have covered building a high-fidelity simulator like BLIS requires careful modeling of the physics of engine-level batch scheduling and processing, data plane coordination, and control plane feedback loops. BLIS combines physics-based dynamics with learned corrections, running entirely on CPUs with pluggable interfaces for each component to enable extremely fast algorithm discovery and configuration search. 

But **how do we know this modeling is accurate?**

We have validated BLIS against production workloads running on real systems and compared its prediction accuracy to commercial inference simulators. The methodology and results, including cross-system accuracy benchmarks and what accuracy is achievable without per-configuration tuning, are detailed enough to deserve their own article.

**Coming Soon: Validating Against Ground Truth** — Quantifying BLIS accuracy on real workloads, how validation catches regressions before they ship, and why validation is a discipline, not a step.

---

*This is the first article in a series on BLIS's architecture. Next: **Validating Against Ground Truth** - quantifying accuracy on real workloads and the methodology behind it.*
