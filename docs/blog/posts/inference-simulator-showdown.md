---
date: 2026-04-03
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
  - Simulator Evaluation
  - Benchmarking
  - Capacity Planning
---

# Understanding LLM Inference Simulators: Lessons from 38 Production Experiments

We evaluated five LLM inference simulators across 38 production experiments to understand their accuracy, speed, and coverage trade-offs. This post shares what we learned about choosing simulators for capacity planning, configuration search, and algorithm discovery.

<!-- more -->

## Our Goals

Capacity planning for LLM inference is expensive. Running real experiments on production hardware takes days and costs thousands in GPU time. Simulators promise to accelerate this process — but how do they compare in practice?

We set out to answer practical questions: How accurate are simulator predictions across different models and hardware? Which simulators support the configurations teams actually deploy? What are the speed-accuracy trade-offs? When should you use analytical models versus discrete-event simulation?

The stakes matter. Deploying Mixtral-8x7B with the wrong GPU count — four instead of eight, or vice versa — can mean either blown SLO budgets or wasted capacity. At H100 cloud rates, mis-sizing a production cluster by 20-30 GPUs costs six figures annually.

Rather than declaring a "best" simulator, our goal was to understand the trade-offs each tool offers.

## Our Approach

### The Simulators

We evaluated five open-source simulators, each taking a different approach to modeling LLM inference:

**[Vidur](https://github.com/microsoft/vidur)** uses discrete-event simulation with profiled GPU kernel times to model vLLM's scheduling decisions. High fidelity to scheduler behavior, but requires profiling each model. Pre-profiled kernels available only for vLLM v0.

**[LLMServingSim](https://github.com/casys-kaist/LLMServingSim)** combines discrete-event simulation with fine-grained network modeling via [astra-sim](https://github.com/astra-sim/astra-sim).

**[LLM-Optimizer](https://github.com/bentoml/llm-optimizer)** uses analytical roofline modeling. Queries HuggingFace for model config, runs extremely fast roofline calculation. No request scheduling or queueing behavior.

**[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator)** uses operation-level profiling — breaks inference into GEMM, attention, communication ops, measures them on hardware, composes results. Dense models only.

**[BLIS](https://github.com/inference-sim/inference-sim)** combines discrete-event simulation with latency models using analytical physics-based basis functions and coefficient training. We evaluated two modes: Roofline (analytical baseline) and Evolved (learned coefficients).

### The Experiments

We ran **38 experiments** on production hardware using vLLM v0.15.1, then asked each simulator to predict the results. We selected widely-deployed models, common GPU types, and practical serving configurations that teams encounter in production.

The test matrix:

- **6 models** spanning dense and MoE architectures—Llama-3.1-8B, Qwen3-14B, CodeLlama-34B, Llama-2-70B (dense), and Mixtral-8x7B, Mixtral-8x22B (MoE).
- **3 GPU types:** H100, A100-80GB, L40S.
- Serving parameters swept:
    - tensor parallelism (1/2/4/8),
    - CPU offload (on/off),
    - LLM GPU memory utilization (0.90/0.95),
    - vLLM chunk size (1024/2048/4096).

For workloads, we used **[ServeGen](https://github.com/alibaba/ServeGen) multi-turn traces from production logs**, split into four categories:

- **General-Purpose:** Chatbot traffic with variable request patterns
- **Code Generation:** GitHub Copilot-style code assistant traffic
- **Role-Playing:** Conversational assistant traffic
- **Reasoning:** Tasks with extended thinking time

We tracked three latency metrics: **E2E Mean MAPE** (end-to-end latency), **TTFT Mean MAPE** (time to first token), and **ITL Mean MAPE** (inter-token latency).

> **What is MAPE?**
> Mean Absolute Percentage Error measures prediction accuracy: `|predicted - actual| / actual x 100`. A simulator with 15% MAPE means its predictions are off by 15% on average.

### Evaluation Methodology

We evaluated each simulator as a blackbox: off-the-shelf usage with publicly available documentation, no custom profiling, no internal modifications, no advantage to any tool. This reflects how platform engineers without simulator-specific expertise would use these tools in practice.

## Our Results

### Coverage: What Can Each Simulator Model?

Coverage emerged as the first major differentiator. Before evaluating accuracy, we discovered that most simulators couldn't run large portions of our test suite.

| Simulator | Experiments Covered | Coverage | Key Limitations |
|-----------|---------------------|----------|-----------------|
| **BLIS** (both variants) | 38/38 | 100% | None—full model, workload, GPU, and serving parameter support |
| **LLM-Optimizer** | 36/38 | 94.7% | No L40S hardware profiles (H100/A100 only); MoE approximated as dense; limited vLLM argument support |
| **AIConfigurator** | 19/38 | 50% | Dense models & H100 only|
| **Vidur** | 4/38 | 10.5% | Requires pre-built model profiles; only CodeLlama-34B & Llama-2-70B |
| **LLMServingSim** | 1/38 | 2.6% | Only 1 model with profiled coefficients matching our test set (Mixtral-8x7B); supports only 2 models total on H100; prohibitive runtime limits broader testing |

Coverage gaps compound. No MoE support eliminates Mixtral evaluations. H100-only restricts cost-optimization studies across GPU types.

### Accuracy: How Well Do Predictions Match Reality?

We compared simulators on their natively-supported experiments:

![BLIS vs LLM-Optimizer comparison](figures/sim_comparisons/blis_vs_llm_optimizer.png)

*Figure 1: Accuracy comparison between BLIS and LLM-Optimizer across 36 shared experiments on H100/A100-80GB with all 6 models. BLIS-Evolved achieved 11.79% E2E MAPE and 22.81% TTFT MAPE. Pure roofline models (BLIS-Roofline, LLM-Optimizer) miss queueing delays and TP communication overhead. LLM-Optimizer approximates MoE as dense and lacks L40S GPU support.*

![BLIS vs AIConfigurator comparison](figures/sim_comparisons/blis_vs_aiconfigurator.png)

*Figure 2: Accuracy comparison between BLIS and AIConfigurator across 19 shared experiments on H100 with dense models only (Qwen3-14B, CodeLlama-34B, Llama-2-70B, Llama-3.1-8B). AIConfigurator uses operation-level profiling but excludes MoE architectures and A100/L40S GPUs.*

![BLIS vs Vidur comparison](figures/sim_comparisons/blis_vs_vidur.png)

*Figure 3: Accuracy comparison between BLIS and Vidur across 4 shared experiments with 2 models (CodeLlama-34B, Llama-2-70B) on H100 only. Vidur requires pre-built model profiles and does not support MoE architectures.*

![BLIS vs LLMServingSim comparison](figures/sim_comparisons/blis_vs_llmservingsim.png)

*Figure 4: Accuracy comparison between BLIS and LLMServingSim on 1 shared experiment (Mixtral-8x7B TP4, 2000-request cluster workload). BLIS-Evolved achieved 1.34% E2E MAPE, LLMServingSim 76%, and BLIS-Roofline 97.69%. BLIS-Evolved is 72.8× more accurate than roofline and 56.7× better than LLMServingSim on this high-TP MoE configuration.*

Across their supported experiments, median E2E error ranged from 7.4% (BLIS-Evolved) to 619% (Vidur). TTFT error spanned 12.6% (BLIS-Evolved) to 29,783%. ITL error ranged 9.8% (BLIS-Evolved) to 259%.

**An important nuance:** BLIS-Evolved's E2E latency prediction is strong (11.79% MAPE), but TTFT accuracy is weaker (22.81% MAPE) - nearly 2× worse. For workloads where time-to-first-token dominates user experience, this gap matters.

### The Accuracy-Speed Frontier

Runtime determines whether a simulator is practical for your workflow.

| Simulator | Median Runtime (s) | Speedup vs. Real |
|-----------|-------------------|------------------|
| **LLM-Optimizer** | 0.1 | 22,166× |
| **BLIS-Evolved** | 0.8 | 1,479× |
| **BLIS-Roofline** | 1.6 | 766× |
| **AIConfigurator** | 3.5 | 349× |
| **Vidur** | 9.1 | 134× |
| **LLMServingSim** | 353.3 | 0.4× |

*Median runtime and speedup relative to real GPU experiments. LLM-Optimizer is 22,000× faster, BLIS-Evolved 1,500× faster, while LLMServingSim is actually slower than running real experiments. Practical impact? Simulating 100 experiments takes 10 seconds (LLM-Optimizer), 80 seconds (BLIS-Evolved), 15 minutes (Vidur), or 10 hours (LLMServingSim).*

![Speed vs Accuracy Pareto Frontier](figures/fig5_pareto.png)

*Figure 5: Speed vs accuracy Pareto frontier across all 38 experiments. LLM-Optimizer occupies the fast-but-rough corner (0.1s runtime, instant feedback, no queueing dynamics). LLMServingSim sits in the slow-but-detailed region (353s runtime, yet higher MAPE than BLIS-Evolved—fidelity alone does not guarantee accuracy). BLIS-Evolved hits the frontier elbow (0.8s runtime, 11.79% E2E MAPE), balancing accuracy and speed. Error bars show median runtime with interquartile range.*

## Lessons Learned

### Lesson 1: Coverage Determines Usability

Before worrying about accuracy, check whether a simulator supports your deployment. MoE models, diverse GPU types, and vLLM configurations eliminated 50-90% of experiments for most tools. If a simulator can't model your architecture, accuracy is irrelevant.

### Lesson 2: Speed-Accuracy Trade-Offs Are Real

Analytical simulators (LLM-Optimizer, BLIS-Roofline) offer instant feedback but miss queueing dynamics and communication overhead. Discrete-event simulators (BLIS-Evolved, Vidur, LLMServingSim) capture these effects but take longer.

A hybrid workflow emerged as practical: use fast analytical models for initial exploration (1,000 configs in 2 minutes with LLM-Optimizer), then validate promising candidates with discrete-event simulation (13 minutes for 1,000 configs with BLIS-Evolved).

### Lesson 3: Use Case Determines Requirements

**For capacity planning with tail latency SLOs:** You need discrete-event simulation to model queueing and output P90/P99 metrics. Analytical simulators only predict mean latency. Among discrete-event simulators, coverage and speed matter — you'll run many experiments.

**For rapid configuration search:** Analytical models (LLM-Optimizer, BLIS-Roofline) excel at pruning obviously bad configs. Use them first, validate finalists with discrete-event simulation.

**For AI-driven algorithm discovery:** Speed dominates. Training loops need sub-second simulation. LLM-Optimizer (0.1s) makes large-scale exploration feasible. BLIS-Evolved (0.8s) offers multi-instance modeling if you need it and can accept the slowdown.

### Lesson 4: Learned Coefficients Help

Comparing BLIS-Roofline (analytical) to BLIS-Evolved (learned coefficients) showed consistent accuracy improvements. Pure roofline models miss second-order effects — queueing delays, communication overhead, architecture-specific behavior. Training coefficients on real data captures these gaps without sacrificing generality.

### Lesson 5: No Universal "Best" Simulator

Each simulator offers different trade-offs. Fast but inaccurate wastes time on bad decisions. Accurate but slow limits exploration. High accuracy on limited coverage doesn't help if your architecture isn't supported.

The practical answer: combine multiple simulators. Use fast analytical models for exploration, discrete-event simulation for validation, and always test on your specific model/hardware/workload before trusting predictions.

## Practical Guidance by Use Case

Based on our experiments, here's what worked for different workflows:

### Capacity Planning & Configuration Search

**The challenge:** You need accurate predictions to size production clusters. A fast simulator with 50% error leads to wrong resource decisions — overprovision and waste budget, or underprovision and miss SLOs.

**What worked for us:**

- **For tail latency SLOs (P90, P99):** Use discrete-event simulators. Analytical models (LLM-Optimizer, AIConfigurator) only predict mean latency and cannot validate tail latency requirements. Among discrete-event options, BLIS-Evolved offered the best coverage (38/38 experiments) at practical speed (0.8s per run). Vidur provides high scheduler fidelity but requires per-model profiling.

- **For rapid configuration exploration:** Start with LLM-Optimizer (0.1s per config) to sweep 1,000 candidates in 2 minutes and eliminate obviously bad choices (wrong TP, insufficient memory). Then validate promising candidates with BLIS-Evolved (13 minutes for 1,000 configs). This hybrid workflow balances speed and accuracy.

- **When to avoid:** LLMServingSim's ~6 minute runtime makes iterative exploration impractical.

### AI-Driven Algorithm Discovery

**The challenge:** Training loops for discovering better serving algorithms ([ADRS](https://sky.cs.berkeley.edu/project/adrs/)) need millions of simulations. Speed dominates — even 1 second per simulation becomes a bottleneck.

**What worked for us:**

- **For single-instance algorithms:** LLM-Optimizer (0.1s per run) is 8× faster than any alternative. While it misses queueing dynamics, algorithm discovery often optimizes *relative* performance across policies rather than absolute accuracy. The speed advantage enables large-scale exploration that wouldn't be feasible otherwise.

- **For multi-instance algorithms:** If you need to explore routing policies across replicas, BLIS-Evolved (0.8s per run) supports multi-instance modeling. The 8× slowdown versus LLM-Optimizer is the trade-off for cluster-level features.

- **When to avoid:** AIConfigurator (3.5s), Vidur (9.1s), and LLMServingSim (353s) are too slow for training loops. Better suited for validating hand-designed systems.

## What We Didn't Test

This evaluation focused on single-instance vLLM serving accuracy. We did not test:

**Multi-instance cluster dynamics:** Real deployments use load balancing, request routing, and autoscaling across multiple instances. How simulators predict cluster-level behavior remains open.

**Cost modeling:** Capacity planning involves cost per token and total cost of ownership, not just latency. Cost prediction accuracy is future work.

**Production drift:** Models update, workloads shift, hardware changes. How quickly simulators become stale and how much re-profiling is needed remains unexplored.

Acknowledging these gaps clarifies scope. Single-instance latency prediction is the foundation, but production serving encompasses more.

## Final Thoughts

There is no universal best simulator — only the best simulator for your problem.

Our 38 experiments revealed massive variation in accuracy (1-76% error), speed (0.1s to 6 minutes), and coverage (10-100% of experiments). Each simulator makes different trade-offs optimized for different workflows.

The key insight: a hybrid approach combining multiple simulators handles most practical use cases. Use fast analytical simulators (BLIS-Roofline, LLM-Optimizer) for rapid exploration, then validate using slower, more-accurate simulators (BLIS-Evolved, AIConfigurator). Match the tool to the problem.

Marketing claims are not validation. Run the simulator on *your* model, *your* hardware, *your* workload, then compare against real measurements. Test before you trust.
