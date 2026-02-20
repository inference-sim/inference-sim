# Research Hypotheses for BLIS Inference Simulator

## Problem Statement

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation.

We recently validated a powerful research methodology: **pose an intuitive hypothesis about system behavior, design experiments to test it, and use hypothesis failures to surface bugs and design limitations.** Specifically, we hypothesized that "prefix-aware routing should outperform load-only routing for prefix-heavy workloads." Testing this hypothesis uncovered 3 bugs in the multi-turn workload generator, led to a new configurable `prefix_length` field, and produced documented example YAMLs with clear TTFT improvements (2.3× better mean, 3.1× better p99).

We want to generate 20 more testable hypotheses that:
1. Are **intuitive** — explainable to anyone without deep system knowledge
2. Are **testable** — can be validated with BLIS experiments (existing or near-future capabilities)
3. Are **documentable** — each becomes an example YAML + comparison script that users can run
4. **Achieve broad coverage** — spanning routing, scheduling, KV cache, workload patterns, admission control, tiered storage, multi-instance behavior, and latency modeling
5. Follow the pattern: **hypothesis → experiment design → predicted outcome → what bugs/limitations failure would surface**

### BLIS Capabilities (what can be tested today)

**Routing policies:** round-robin, least-loaded, weighted (composable scorer pipeline with prefix-affinity, queue-depth, kv-utilization, load-balance scorers), prefix-affinity (legacy), always-busiest (pathological)

**Scheduling:** FCFS, priority-FCFS, SJF (shortest-job-first), reverse-priority (pathological)

**Admission control:** always-admit, token-bucket, reject-all (pathological)

**KV cache:** Single-tier (GPU), tiered (GPU+CPU with offload/reload and transfer latency), prefix caching with block-level LRU eviction

**Workload generation:** Poisson/Gamma/Weibull arrivals, Gaussian/Exponential/ParetoLogNormal distributions, prefix groups with configurable prefix_length, multi-turn chat with context accumulation, multi-client with SLO classes (realtime/interactive/batch), multimodal tokens, reasoning with think time

**Metrics:** TTFT (mean/p90/p95/p99), E2E latency, ITL (inter-token latency), throughput (req/s, tok/s), scheduling delay, per-instance distribution, anomaly detection (priority inversions, HOL blocking), fitness evaluation with weighted multi-objective scoring

**Decision tracing:** Per-request routing decisions with counterfactual analysis (what-if top-k candidates)

**Cluster:** 1-64 instances, shared-clock event loop, online routing pipeline with admission/routing latency, pending request tracking

### What makes a good hypothesis

- **Intuitive claim:** "X should be better than Y for workload Z because of mechanism M"
- **Clear experiment:** Two configurations differing in exactly one dimension
- **Measurable outcome:** Specific metric (TTFT p99, throughput, distribution uniformity) that should differ
- **Diagnostic value:** If the hypothesis fails, it points to a specific code path or design assumption worth investigating
- **User value:** The experiment becomes a documented example that helps users understand when to use which configuration
