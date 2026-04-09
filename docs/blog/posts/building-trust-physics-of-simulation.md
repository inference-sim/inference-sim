---
date: 2026-04-09
draft: true
authors:
  - dipanwita
categories:
  - Architecture
  - Deep Dive
---

# Building Trust: The Physics of High-Fidelity Inference Simulation

Imagine testing routing policies, autoscaling strategies, and hardware configurations without touching production. No risk. No downtime. Just answers. That's the promise of simulation—but only if it's accurate enough to trust.

A simple queueing model predicts 50ms time-to-first-token. Production measures 200ms. The difference reveals how much complexity hides beneath the surface. Capacity decisions are million-dollar bets—H100 vs A100, tensor parallelism 4 vs 8—and intuition fails at this scale.

<!-- more -->

## The Right Physics, Not Everything

Building a trustworthy simulator isn't about modeling everything—it's about modeling the right physics. The batch dynamics that couple request latencies. The KV cache pressure that triggers preemption. The prefill-decode handoffs that trade network costs for throughput. Miss any of these, and your predictions diverge from reality.

BLIS achieves this by modeling the actual physics—the mechanisms that determine latency in real systems: how requests couple through shared batch steps, how KV cache pressure triggers preemption cascades, how prefill-decode disaggregation trades network transfer costs for hardware specialization. The entire simulation runs on CPU—no GPUs required. Step times come from analytical roofline models (compute vs memory bottlenecks derived from model architecture and hardware specs) corrected with coefficients trained on real vLLM production traces. The result: microsecond-scale predictions with single-digit percent accuracy. Fast enough for rapid iteration, accurate enough to trust for production decisions.

This article shows what it takes to build that level of fidelity—from token generation physics to distributed orchestration. Let's follow a request's 50-millisecond journey through the system to see where every millisecond of that complexity lives.

## A Request's Journey: The Hidden Complexity

A user hits enter. Fifty milliseconds later, the first token appears. What happened in between? Three architectural layers working together: the inference engine (vLLM), the data plane (cluster orchestration), and the control plane (autoscaling). Model them all with fidelity, or your capacity decisions will be wrong.

### Layer 1: The Engine (vLLM)

Let's start at the engine level—the inference engine running on a single GPU instance. This is vLLM, the workhorse that schedules requests, manages memory, and generates tokens.

Here's the key thing most people miss: vLLM doesn't process requests one at a time. It processes batches in "steps." A step is one pass through the GPU—all requests in the batch go through together. Some are processing their prompts, others are generating tokens. The step takes as long as the slowest operation, and everyone waits.

Picture four requests in a batch. Three are generating single output tokens—fast, memory-bound operations taking maybe 2 milliseconds. But the fourth is crunching through a 512-token prompt—a compute-heavy operation taking 20 milliseconds. The step time is dominated by that one prompt. All four requests wait the full 20 milliseconds, even though three could finish in 2.

This is why simple per-request models fail. They calculate each request independently: time = α + β × tokens. But that's not reality. Ten requests generating tokens don't take "10 × single_request_time." They take one batch step that covers all ten. Get this wrong, and throughput predictions can be 5-10x off.

So what does BLIS replicate from vLLM to capture this dynamic?

Scheduling: Priority queues where critical requests go before batch jobs. KV cache: Block-level memory allocation where shared prompts get reused (massive speedup for RAG workloads), and running requests get evicted when memory fills. Continuous batching: Requests join and leave mid-flight as they complete. BLIS models all of this—not as approximations, but with exact vLLM semantics.

Now here's where the CPU-only simulation comes in. For each step, BLIS computes two bottlenecks. Compute time: FLOPs divided by GPU TFLOPS—for example, a 512-token prefill on an H100 takes about 20 milliseconds. Memory time: bytes divided by GPU bandwidth—a decode step reading KV cache takes about 2 milliseconds. The step time is the maximum of the two—whichever is slower. BLIS then applies learned corrections to account for kernel overhead, cache behavior, and other real-world effects. This is why BLIS is fast and accurate without GPUs: we're computing the same bottleneck analysis that vLLM's GPU experiences, using model architecture from HuggingFace configs and hardware specs from datasheets.

Watch what happens as batches evolve. Small batch, decode-only: fast, 2 milliseconds per token. New request joins with a long prompt: everyone waits 20 milliseconds while it processes. That request finishes and switches to decode: back to fast 2-millisecond steps. Another request completes and leaves the batch: even faster now, maybe 1.8 milliseconds. The batch size is constantly changing, and so is the step time.

You can't model inference with per-request equations. vLLM works in batches and steps, with request latencies coupled through that batching. BLIS captures this through discrete-event simulation—one step event per batch operation, with batch membership updating after each completion.

But a single vLLM instance running in isolation is only part of the story. In production, you have a cluster of instances—and that's where orchestration complexity enters.

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
