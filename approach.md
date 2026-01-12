# BLIS Approach: Blackbox Inference Simulator for vLLM Internals

## 1. Architectural Overview

BLIS is a **Discrete Event Simulator (DES)** integrated with a lightweight, CPU-only latency regression model. Unlike real-time execution engines, BLIS bypasses tensor operations, advancing the simulation clock by calculating time-deltas between discrete system events such as request arrivals, scheduling ticks, and KV-cache state changes.

### Core Components

* **The Event Engine:** A priority-queue-based orchestrator that manages the global simulation clock and triggers system callbacks at precise timestamps.
* **The Virtual Scheduler:** A high-fidelity mirror of vLLM’s `Scheduler` class. It manages request state transitions across `Waiting`, `Running`, and `Preempted` queues.
* **The Virtual KV-Cache Manager:** Simulates **PagedAttention** block allocation. It tracks logical-to-physical mappings, reference counts for prefix sharing, and memory fragmentation without physical GPU memory allocation.
* **The Latency Model:** An analytical engine that predicts the duration of vLLM iterations based on batch composition (prefill/decode counts), model architecture, and hardware topology.

---

## 2. Capabilities

* **vLLM Feature Parity:** Supports Prefix Caching, Chunked Prefill, Continuous Batching, and Preemption (Swap/Recompute).
* **Configuration-Aware:** Directly ingests standard vLLM arguments such as `max-num-batched-tokens`, `max-num-seqs`, and `block-size`.
* **Workload Agnostic:** Scales across diverse arrival rates (e.g., Poisson distributions) and varying sequence length distributions.
* **Extensible Design:** Easily adapts to new hardware profiles, Tensor Parallelism (TP) degrees, or vLLM version updates via coefficient recalibration.

---

## 3. Key Simulation Pillars

### A. Scheduling & Memory Management

BLIS maintains a synchronized state between the Scheduler and the KV-Cache Manager to replicate vLLM's internal behavior:

* **KV-Cache Manager:** Implements **Hash-Deterministic Mapping** for Prefix Caching. It calculates prefix hits for incoming requests, allowing the scheduler to skip redundant computations. It manages a global block pool and tracks block lifecycles to signal memory pressure.
* **Scheduler:** Implements the core logic for **Continuous Batching** and **Chunked Prefill**. It executes the following loop until the simulation horizon is met or all requests are satisfied:



```python
while time < horizon and active_requests:
    # Identify the next batch of tokens based on memory/quota
    scheduled_batch = scheduler.schedule() 
    
    # Predict GPU execution time for this specific iteration
    step_time = gpu_latency_model.predict(scheduled_batch)
    
    # Advance request states and global clock
    scheduler.update_processing_indices(scheduled_batch)
    event_queue.push(current_time + step_time)
```

### B. GPU Latency Model

To achieve **CPU-only simulation**, BLIS replaces expensive kernel execution with a linear regression model that predicts the duration of a single GPU iteration ($L_{gpu}$):

$$L_{gpu} = \beta_0 + \beta_1 X + \beta_2 Y$$

**Where:**
* **$X$**: Total number of **uncached prefill tokens** in the scheduled batch.
* **$Y$**: Total number of **decode tokens** (number of sequences) in the scheduled batch.
* **$\beta_i$**: Learned coefficients derived from hardware-specific profiling.

### C. CPU & System Overhead Model

Total request latency includes non-GPU overheads such as tokenization, API serialization, and networking. These are modeled as:

$$L_{cpu} = \alpha_0 + \alpha_1 M + \alpha_2 N$$

**Where:**
* **$M$**: The input sequence length.
* **$N$**: The generated output length.

---

## 4. Metrics Capture

BLIS aggregates simulated GPU and CPU latencies to derive standard vLLM serving metrics:

| Metric | Calculation Logic |
| :--- | :--- |
| **TTFT** | $L_{cpu} + \sum (\text{prefill\_step\_times}) - \text{Arrival Time}$ |
| **ITL** | Observed $\Delta t$ between consecutive decode iterations |
| **E2E Latency** | $L_{cpu} + \sum (\text{prefill\_steps}) + \sum (\text{decode\_steps}) - \text{Arrival Time}$ |

---

## 5. Training & Calibration

BLIS uses a **Data-Driven Calibration** strategy to ensure simulation accuracy. This process is performed once per environment configuration (Model, GPU, TP, vLLM version):

1.  **Initialization:** Define baseline estimates for $\alpha$ and $\beta$ coefficients.
2.  **Profiling:** Execute a reference workload on a live vLLM instance to collect ground-truth (GT) Mean and P90 metrics.
3.  **Optimization:** Run BLIS iteratively using **Bayesian Optimization** to minimize the multi-objective loss function:



$$\text{Loss} = \sum_{m \in \{TTFT, ITL, E2E\}} (|GT_{mean\_m} - Sim_{mean\_m}| + |GT_{p90\_m} - Sim_{p90\_m}|)$$

4.  **Artifact Generation:** Optimal coefficients are stored in `coefficients.yaml` for production-grade inference.

---

## 6. Inference

During inference, BLIS loads the pre-calibrated coefficients for the target configuration. By executing the high-speed Discrete Event loop, it generates high-fidelity performance projections for novel workloads in seconds. This enables:
* **Capacity Planning:** Determining how many concurrent users a node can support.
* **Hyperparameter Tuning:** Finding the optimal `max_num_batched_tokens` or `block_size` without burning GPU hours.
* **Regression Testing:** Detecting how code changes in vLLM internals affect throughput and latency.