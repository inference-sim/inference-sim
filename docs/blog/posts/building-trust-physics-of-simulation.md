---
date: 2026-04-09
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

# The Physics of High-Fidelity Distributed Inference Platform Simulation

Production LLM inference platforms are distributed systems where routing policies, admission control, autoscaling, and engine-level scheduling all interact to determine latencies and throughput. How do you explore how different policies and configurations affect these KPIs before deploying to production? Testing a new routing scorer or autoscaling threshold on live traffic risks cascading bugs across the fleet, while building separate test environments burns GPU-hours and still cannot predict interactions between cluster-level policies and engine-level batch dynamics.

The answer is **end-to-end simulation**: model the entire distributed inference stack to explore how policies and configurations affect latencies and throughput for your workloads. *What does it take to build a simulator accurate enough to guide these decisions?* The challenge lies in capturing the right mechanisms across all three layers. At the engine level, batches process together — all requests wait for the slowest operation to finish, so KV cache fills trigger preemptions and long prompts stall short decodes. At the cluster level, routing policies operate on stale cache state, admission control gates overload, and prefill/decode disaggregation trades utilization for latency. At the control plane, autoscalers react to lagged metrics, creating oscillations. When these couplings are not modeled, predictions diverge: a back-of-the-envelope model might predict 50ms time-to-first-token while production measures 200ms.

<!-- more -->

## Building Fidelity from First Principles

[BLIS](https://github.com/inference-sim/inference-sim) (Blackbox Inference Simulator) models inference serving through discrete-event simulation, advancing from event to event rather than stepping through continuous time. This approach runs orders of magnitude faster than real-time, requires no GPUs, and evaluates hours of production traffic in seconds.

BLIS uses discrete-event simulation to model all three layers. This full-stack fidelity enables **capacity planning** (instance count, GPU type, TP degree) and **configuration search** (routing weights, admission thresholds). Without modeling distributed system couplings, planners predict linear scaling where production saturates, miss SLO violations from routing pile-on, or deploy autoscalers that oscillate.

By modeling production systems ([vLLM](https://github.com/vllm-project/vllm), [llm-d](https://llm-d.ai)) behavior, BLIS enables safe experimentation before deployment:

- **Routing policies** — Test new scorer combinations and weights
- **Admission control** — Explore saturation thresholds and flow control strategies
- **Capacity planning** — Compare model/GPU/TP configurations
- **Workload analysis** — Test how switching from TP=2 to TP=4 affects tail latency under production traffic patterns

Physics-based dynamics with learnable latency components generalize across model architectures and hardware while maintaining production fidelity. Test new configurations on a laptop in seconds without needing production infrastructure. This enables projects like [ADRS](https://sky.cs.berkeley.edu/project/adrs/) (AI-Driven Research Systems) to develop and validate serving policies through fast simulation loops.

This article walks through what it takes to build that level of fidelity — from token batching physics to distributed orchestration, by following a request's end-to-end journey through the system to see where every millisecond of complexity originates.

## A Request's Journey: The Hidden Complexity

A user hits enter, and fifty milliseconds later the first token appears. What happened in between? Three architectural layers working together: the inference engine (vLLM), the data plane (cluster orchestration), and the control plane (autoscaling), all of which high-fidelity simulation must model.

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Engine (vLLM)"]
        Sched[Scheduling]
        KV[KV Cache]
        Batch[Prefill+Decode Batch Formation]
        Step[Forward Pass]
        Sched --> KV --> Batch --> Step
    end

    subgraph Layer2["Layer 2: Data Plane"]
        Admit[Admission]
        Route[Routing]
        PD{P/D Split?}
        Admit --> Route
        Route --> PD
    end

    subgraph PrefillPool["Prefill Pool"]
        PF[Prefill Processing]
    end

    subgraph DecodePool["Decode Pool"]
        Dec[Decode Processing]
    end

    subgraph Layer3["Layer 3: Control Plane"]
        Monitor[Monitor Metrics]
        Decide[Scale Decision]
        Actuate[Add/Remove Instances]
        Monitor --> Decide --> Actuate
    end

    Request([Request]) --> Layer2
    PD -->|Unified| Layer1
    PD -->|Disaggregated| PrefillPool
    PrefillPool -->|KV Transfer| DecodePool
    Layer1 --> Response([Response])
    DecodePool --> Response
    Layer3 -.-> Layer2
    Layer1 -.->|metrics| Layer3
    PrefillPool -.->|metrics| Layer3
    DecodePool -.->|metrics| Layer3
```

### Layer 1: The Engine (vLLM)

> **TL;DR:** Batched execution couples requests together - a heavy prompt in the batch slows down fast decodes running alongside it. BLIS models the full vLLM pipeline (continuous batching, request scheduling and preemption, KV cache pressure, chunked prefill) and predicts forward pass timing using a generalizable model that runs on CPUs without needing real GPUs.

The inference engine does not process requests individually. It processes them in continuously evolving batches. A **step** is one GPU forward pass that advances every request in the batch, either processing prompt tokens (prefill) or generating the next output token (decode). The slowest operation determines when the step completes.

Why does this matter? Consider a batch with three requests decoding single tokens (fast, memory-bound) and one request processing a 512-token prompt (slow, compute-bound). Everyone waits for the slowest. This is not an edge case - batch composition constantly shifts as new requests arrive and completed ones leave.

**What BLIS captures.** vLLM's complexity: continuous batching (requests join and leave mid-flight), mixed prefill-decode execution, block-level KV cache management (prefix reuse, preemption, CPU offloading), and chunked prefill. BLIS models these mechanisms because they determine when requests complete.

**How BLIS predicts step time without GPUs.** BLIS combines physics-based latency models with learned corrections:

```mermaid
flowchart LR
    subgraph Inputs["📊 <b>Inputs</b>"]
        Batch["Batch State<br/>(size, sequences)"]
        LLM["Model Architecture<br/>(layers, params)"]
        HW["Hardware Specs<br/>(TFLOPs, bandwidth)"]
    end

    subgraph Physics["⚡ <b>Physics-Based Latency Model</b>"]
        Compute["Compute Bound<br/>Compute Operations / GPU Speed"]
        Memory["Memory Bound<br/>Bytes Transferred / GPU Bandwidth"]
    end

    subgraph Learned["🎯 <b>Learned Corrections</b>"]
        Coeffs["Trained on<br/>Real vLLM Traces"]
    end

    Batch --> Physics
    LLM --> Physics
    HW --> Physics

    Compute --> StepTime["⏱️ Predicted Step Time"]
    Memory --> StepTime
    Coeffs --> StepTime
```

This approach generalizes across LLM architectures, hardware configurations, and tensor parallelism degrees, enabling seamless experimentation with any model-GPU-TP combination without per-configuration calibration. Accurate forward pass predictions drive accurate end-to-end latency metrics.

### Layer 2: The Data Plane (Cluster Orchestration)

> **TL;DR:** Production clusters run multiple vLLM instances behind a routing gateway. BLIS models saturation-based admission control, composable weighted routing with in-flight tracking, configurable cache signal staleness, and prefill/decode disaggregation. Pluggable interfaces enable algorithm discovery—test new serving policies without writing production code.

```mermaid
graph LR
    A[Request Arrives] --> B{Admission Control}
    B -->|Rejected| X[Drop]
    B -->|Admitted| C[Routing]
    C --> D{Disaggregation?}
    D -->|No| E[Aggregated vLLM Processing]
    D -->|Yes| F[Prefill Pool]
    F -->|KV Transfer| G[Decode Pool]
```

**Admission control** determines whether requests enter the system. BLIS models saturation-based admit/reject decisions: when cluster load exceeds thresholds, incoming requests are rejected or queued rather than overwhelming instances. This prevents queue explosion during traffic spikes and avoids pile-on where burst arrivals flood the same "best" instance.

**Routing** assigns each request to an instance by scoring on weighted signals - prefix cache hits, queue depth, KV utilization. The challenge: burst arrivals cause all routing decisions to see the same stale state and pick the same "best" instance. BLIS models in-flight tracking (counting already-dispatched requests) and signal staleness (cache state queries a 2-second-old snapshot, matching llm-d's ZMQ propagation delay).

**Prefill/decode disaggregation** separates compute-bound prefill from memory-bound decode onto dedicated GPU pools, allowing each to be sized for its bottleneck. Requests process prefill first, then transfer their KV cache over the network to a decode instance. BLIS models the full pipeline: prefill routing, KV transfer, decode routing, and fair-share bandwidth contention when multiple transfers run concurrently.

### Layer 3: The Control Plane (Autoscaling)

> **TL;DR:** Real autoscaling experiments are expensive—feedback loops spanning minutes (HPA scrapes, pod scheduling, VM provisioning, model loading) require 30+ minutes and 10+ GPU replicas per test. BLIS models llm-d's WVA (Workload Variant Autoscaler) four-stage pipeline with pluggable Collector/Analyzer/Optimizer/Actuator interfaces, compressing experiments to seconds on a laptop.

Autoscaling dynamically adjusts instance count to match demand. In production, this happens through feedback loops where HPA scrapes, Kubernetes scheduling, VM provisioning, and model loading all add latency before a new replica serves traffic.

**What BLIS captures.** BLIS models llm-d's WVA four-stage pipeline — Collect, Analyze, Optimize, Actuate, with pluggable interfaces. **Collector** observes per-replica metrics, **Analyzer** detects saturation and emits scaling signals, **Optimizer** decides which GPU types to add/remove respecting multi-model inventory constraints, and **Actuator** applies decisions with configurable delay.

**What BLIS enables.** Researchers can sweep scaling thresholds, compare analyzer strategies under identical workloads, and test multi-model scenarios where scaling one model steals GPUs from another. Each pluggable interface becomes a research hook—swap in a cost-aware Optimizer or an Analyzer that predicts load spikes. The result: discover and validate better autoscaling policies before deployment, with full control over feedback delays and provisioning latencies.

## BLIS in Action: Simulating a Configuration Decision

Consider a configuration decision: you are deploying Qwen3-14B for chatbot workloads at 50 req/s with 8 instances. Does routing policy matter? What about hardware choice?

Testing this in production means provisioning separate GPU pools, running 30+ minutes of traffic per setup and burning GPU-hours to discover the answer. With BLIS, you can simulate these configurations in seconds on a laptop:

```bash
# Install and build BLIS
git clone https://github.com/inference-sim/inference-sim.git
go build -o blis main.go

# H100 with round-robin routing
./blis run --model qwen/qwen3-14b --workload chatbot --rate 50 \
  --num-instances 8 --tp 2 --hardware H100 --routing-policy round-robin

# H100 with prefix-aware routing
./blis run --model qwen/qwen3-14b --workload chatbot --rate 50 \
  --num-instances 8 --tp 2 --hardware H100 --routing-policy weighted \
  --routing-scorers "prefix-affinity:2,queue-depth:1"

# A100-80 with prefix-aware routing
./blis run --model qwen/qwen3-14b --workload chatbot --rate 50 \
  --num-instances 8 --tp 2 --hardware A100-80 --routing-policy weighted \
  --routing-scorers "prefix-affinity:2,queue-depth:1"
```

**Simulated Results:**

| Configuration | Predicted P99 TTFT | Key Finding |
|---------------|----------|-------------|
| H100 (round-robin) | 12.1ms | Baseline with naive routing |
| H100 (prefix-aware) | 11.3ms | **7% improvement** from KV cache reuse |
| A100-80 (prefix-aware) | 45.8ms | **4× slower than H100** — hardware choice dominates |

**What the simulation predicts:** Prefix-aware routing delivers measurable gains on H100, but hardware choice has far greater impact. These simulated predictions guide configuration decisions without provisioning real GPUs.

## From Modeling to Validation

We have covered what it takes to build a high-fidelity distributed platform simulator—modeling engine physics, data plane coordination, and control plane feedback loops. But **how do we know this modeling is accurate?**

We have validated BLIS against production workloads and compared its accuracy to commercial simulators. The methodology and results—cross-system benchmarks and achievable accuracy without per-configuration tuning—deserve their own article.

---

*Next in this series: **Validating Against Ground Truth** — Quantifying BLIS accuracy on real workloads, how validation catches regressions, and the methodology behind single-digit percent error.*
