# Problem Statement: Optimal Joint Scheduling + KV Cache Strategy for LLM Inference

## Context

BLIS (Blackbox Inference Simulator) is a discrete-event simulator for LLM inference serving systems. It models multi-instance clusters with configurable admission control, request routing, KV-cache dynamics (including tiered GPU+CPU offloading), scheduling policies, and token generation.

The current default configuration (llm-d parity) uses:
- **Routing**: Weighted scoring with `prefix-affinity:3, queue-depth:2, kv-utilization:2`
- **Scheduling**: FCFS (first-come-first-served) at each instance
- **KV Cache**: Single-tier GPU (tiered GPU+CPU available but not default)
- **Priority**: Constant (all requests equal)
- **Admission**: Always-admit
- **Batch Formation**: vLLM-style FCFS with chunked prefill

## Problem

Current scheduling and KV caching policies in BLIS are designed independently — the router doesn't know what the scheduler will do with a request, the scheduler doesn't know why the router sent a request to this instance, and KV eviction is oblivious to both routing intent and scheduling priorities.

We need to discover **jointly optimized strategies** where routing, scheduling, KV cache management, and priority assignment work together as a coherent system to substantially beat the current baselines on mixed production workloads.

## Target Workload: Orthogonal SLO × Multi-Turn

**Design principle: SLO tiers and workload types are orthogonal.** All 3 tiers share the identical multi-turn prefix-heavy workload pattern. The SLO class is the ONLY differentiator — strategies cannot exploit token-length differences as a proxy for SLO.

- **All tiers**: prefix_group="system-prompt", prefix_length=512, input~Gaussian(256,100), output~Exponential(128), gamma arrival CV=2.0
- **Critical (20%)**: Premium customers — strict TTFT/E2E targets
- **Standard (50%)**: Normal customers — balanced latency/throughput
- **Sheddable (30%)**: Free tier / batch — best-effort
- **Near-saturation load**: 2000 req/s across 8 instances

## Constraints

1. **KV offloading must always be enabled** — every strategy uses tiered GPU+CPU KV cache
2. **Must be implementable in BLIS** — strategies must map to the simulator's extension points (new scorers, schedulers, priority policies, KV store variants)
3. **Must be defensible to vLLM/llm-d/distributed inference experts** — strategies grounded in real system mechanics, not simulator artifacts
4. **Must beat baselines on**: throughput (tokens/s), TTFT P99, E2E P99, and SLO-class fairness

## Key Findings From Prior Experiments

These findings constrain and inform strategy design:

1. **Queue-depth and kv-utilization scorers are redundant** when KV blocks are abundant — they track correlated signals
2. **Prefix-affinity is degenerate without load-balancing** — cold-start ties pile ALL requests onto instance_0
3. **Session stickiness inherently load-balances** for multi-turn workloads — cache-heavy scoring dominates
4. **Distribution MEDIAN drives KV pressure**, not mean or tail — ParetoLogNormal produces fewer preemptions than Gaussian
5. **Chunked prefill benefits TTFT, NOT ITL** — reduces HOL blocking for new requests
6. **Snapshot staleness safe zone is <5ms for kv-utilization** — composite scorers mitigate ~99% of staleness effect
7. **Horizontal scaling is super-linear** at near-saturation — TTFT P99 scales 7.4x for 4→8 instances
8. **LL tie-breaking has positional bias** — always sends to instance 0 when loads are equal
9. **Counterfactual regret is structurally zero** for score-based policies — chosen IS best by definition
10. **Combined pathological policies are super-additive** — interactions between bad routing + bad scheduling are worse than sum of parts
11. **Precise KV-aware routing (llm-d blog)** achieves 57x TTFT improvement over approximate prefix-aware routing by maintaining real-time global cache state

## What We're Looking For

Novel strategies that exploit one or more of these opportunities:
- **Cross-layer information sharing**: Router knows scheduler state, scheduler knows routing intent, KV eviction knows both
- **SLO-aware differentiation**: Different treatment for critical vs sheddable requests throughout the pipeline
- **Proactive KV management**: Anticipate cache pressure instead of reacting to it
- **Load-regime adaptation**: Different behavior at low load vs near-saturation vs overload
- **Cache-aware scheduling**: Schedule requests that share KV blocks together to maximize cache reuse
- **Deadline-driven urgency**: Requests approaching SLO deadlines get routing/scheduling priority

## Parameterized Strategy Design

Every strategy must be a **parameterized template** — the core mechanism defines WHAT the strategy does, while tunable parameters control HOW AGGRESSIVELY it does it. This separation enables:

1. **Bayesian optimization** of parameters for each target workload
2. **Fair comparison** — each strategy gets the benefit of parameter tuning, not just hand-picked values
3. **Expert defensibility** — "our mechanism is X; Bayesian search found these parameters optimal" is stronger than "we tried 0.9 and it worked"

Each strategy exposes a parameter vector, e.g.:
```yaml
parameters:
  routing_weights:     [w_prefix, w_queue, w_kv, w_custom]  # scorer weights
  offload_threshold:   [0.5, 1.0]   # range for GPU→CPU offload trigger
  priority_age_weight: [1e-7, 1e-4] # range for SLO aging factor
  prefill_threshold:   [64, 512]    # range for chunked prefill threshold
  # ... strategy-specific parameters
```

The optimization harness runs BLIS with different parameter settings, measures the multi-objective fitness (TTFT P99, throughput, E2E P99, fairness), and uses Bayesian optimization (or grid search for small parameter spaces) to find the Pareto-optimal operating point.

## Success Criteria

A winning strategy should demonstrate:
- **>15% TTFT P99 improvement** over the baseline on mixed production workloads
- **>5% throughput improvement** (harder to achieve with the same hardware model)
- **No SLO class starvation** — sheddable requests still complete, critical requests meet deadlines
- **Robust across seeds** — improvements consistent across 3+ random seeds
- **Mechanistically explained** — not just "it's faster" but "it's faster because X mechanism does Y"
- **Parameter sensitivity characterized** — show which parameters matter most and their optimal ranges
