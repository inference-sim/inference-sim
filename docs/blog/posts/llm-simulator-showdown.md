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

# The LLM Simulator Showdown: Which Tool Actually Delivers?

Choosing the right LLM inference simulator can save weeks of experimentation and thousands in compute costs. But which one actually works? We tested five popular simulators head-to-head across 38 real-world experiments on production hardware.

<!-- more -->

## Table of Contents
1. [Why Simulator Choice Matters](#why-simulator-choice-matters)
2. [Meet the Contenders](#meet-the-contenders)
3. [How We Tested](#how-we-tested)
4. [Accuracy and Coverage: Who Gets It Right?](#accuracy-and-coverage)
5. [Speed vs. Accuracy: The Pareto Frontier](#speed-vs-accuracy-the-pareto-frontier)
6. [Which Simulator For Your Use Case](#which-simulator-for-your-use-case)
7. [The Bottom Line](#the-bottom-line)


<a name="why-simulator-choice-matters"></a>
## Why Simulator Choice Matters

Choosing the right LLM inference simulator can save weeks of experimentation and thousands in compute costs. But which one actually works?

Imagine you are deploying Mixtral-8x7B for your AI-powered coding assistant. Four GPUs or eight? Which hardware meets your SLO targets? Running real experiments could take days and cost thousands. A simulator promises answers in minutes — if you trust its predictions.

We tested five simulators across 38 production experiments. Accuracy ranged from 1% to 76% error on identical workloads. Speed varied from milliseconds to hours. Some tools couldn't run two-thirds of our experiments.

This guide breaks down which simulator to use for capacity planning, configuration search, and algorithm discovery — backed by hard data from 38 experiments across six models, four workload types, and three GPU types.

<a name="meet-the-contenders"></a>
## Meet the Contenders

Five popular simulators. Five completely different bets on how to predict LLM inference performance.

**[Vidur](https://github.com/microsoft/vidur)** replays your exact workload through discrete-event simulation, using profiled GPU kernel times to model vLLM's scheduling decisions. High fidelity to scheduler behavior, but you need to profile each model first.

**[LLMServingSim](https://github.com/casys-kaist/LLMServingSim)** takes a different path—discrete-event simulation with fine-grained network modeling via [astra-sim](https://github.com/astra-sim/astra-sim). The catch? 353 seconds per experiment.

**[LLM-Optimizer](https://github.com/bentoml/llm-optimizer)** takes the opposite approach: analytical modeling from first principles. Query HuggingFace for LLM config, run the roofline calculation, get an answer in 0.1 seconds. No request scheduling or queueing behavior captured.

**[AIConfigurator](https://github.com/ai-dynamo/aiconfigurator)** uses operation-level profiling—break inference into GEMM, attention, communication ops, measure them on hardware, then compose the results. Dense models only.

**[BLIS](https://github.com/inference-sim/inference-sim)** combines analytical roofline with discrete-event scheduling. Multiple latency modes available—we are comparing Roofline (analytical baseline) and Evolved (learned coefficients).

Let us answer the question : which approach actually delivers?

<a name="how-we-tested"></a>
## How We Tested

We ran **38 experiments** on production hardware using vLLM v0.15.1, then asked each simulator to predict the results.

The test matrix: **6 models** spanning dense and MoE architectures—Llama-3.1-8B, Qwen3-14B, CodeLlama-34B, Llama-2-70B (dense), and Mixtral-8x7B, Mixtral-8x22B (MoE). **3 GPU types:** H100, A100-80GB, L40S. Serving parameters swept: tensor parallelism (1/2/4/8), CPU offload (on/off), vLLM GPU memory utilization (0.90/0.95), vLLM chunk size (1024/2048/4096).

For workloads, we used **[ServeGen](https://github.com/alibaba/ServeGen) multi-turn traces from Alibaba production logs**, split into four categories:

- **General-Purpose:** Chatbot traffic with variable request patterns
- **Code Generation:** GitHub Copilot-style code assistant traffic
- **Role-Playing:** Conversational assistant traffic
- **Reasoning:** Tasks with extended thinking time

Every experiment tracked three latency metrics: **E2E Mean MAPE** (end-to-end latency), **TTFT Mean MAPE** (time to first token), and **ITL Mean MAPE** (inter-token latency), capturing whether a simulator gets the full user experience right, not just throughput.  

> **What is MAPE?**
> Mean Absolute Percentage Error measures prediction accuracy: `|predicted - actual| / actual x 100`. A simulator with 15% MAPE means its predictions are off by 15% on average. Lower is better, and even 30-40% error can derail capacity planning decisions.

<a name="accuracy-and-coverage"></a>
## Accuracy and Coverage: Who Gets It Right?

Picture this: you are deploying Mixtral-8x7B on A100 nodes with TP=4. You have read the accuracy benchmarks, picked your simulator, fired it up, and it tells you it does not support MoE models! Or A100s. Or diverse TP configurations. Before you ever question the predictions, coverage gaps have already made the decision for you.

Only **BLIS** (both variants) covers all 38 experiments. Since no other simulator runs the full test suite, we compare BLIS against each simulator on their supported experiments.

| Simulator | Experiments Covered | Coverage | Key Limitations |
|-----------|---------------------|----------|-----------------|
| **BLIS** (both variants) | 38/38 | 100% | None—full model, workload, GPU, and serving parameter support |
| **LLM-Optimizer** | 36/38 | 94.7% | No L40S hardware profiles (H100/A100 only); MoE approximated as dense; limited vLLM argument support |
| **AIConfigurator** | 19/38 | 50% | Dense models & H100 only|
| **Vidur** | 4/38 | 10.5% | Requires pre-built model profiles; only CodeLlama-34B & Llama-2-70B |
| **LLMServingSim** | 1/38 | 2.6% | Only 1 model with profiled coefficients matching our test set (Mixtral-8x7B); supports only 2 models total on H100; prohibitive runtime limits broader testing |

**Coverage gaps compound.** No MoE support eliminates Mixtral evaluations. H100-only means no cost-optimization across GPU types. Multiply those constraints and a simulator advertising "high accuracy" may only deliver it on a narrow slice of what production actually looks like.

**Head-to-head accuracy comparisons on shared experiments:**

![Figure 1: BLIS vs LLM-Optimizer](figures/sim_comparisons/blis_vs_llm_optimizer.png)

**Figure 1:** 36 shared experiments on H100/A100-80GB, all 6 models. BLIS-Evolved achieved 11.79% E2E MAPE and 22.81% TTFT MAPE. Pure roofline (BLIS-Roofline, LLM-Optimizer) misses queueing delays and TP communication overhead. LLM-Optimizer: MoE approximated as dense, no L40S GPU support.

![Figure 2: BLIS vs AIConfigurator](figures/sim_comparisons/blis_vs_aiconfigurator.png)

**Figure 2:** 19 shared experiments on H100, dense models only (Qwen3-14B, CodeLlama-34B, Llama-2-70B, Llama-3.1-8B). AIConfigurator uses operation-level profiling but excludes MoE and A100/L40S GPUs.

![Figure 3: BLIS vs Vidur](figures/sim_comparisons/blis_vs_vidur.png)

**Figure 3:** 4 shared experiments, 2 models (CodeLlama-34B, Llama-2-70B) only on H100. Vidur requires pre-built model profiles, does not support MoE.

![Figure 4: BLIS vs LLMServingSim](figures/sim_comparisons/blis_vs_llmservingsim.png)

**Figure 4:** 1 shared experiment (Mixtral-8x7B TP4, 2000-request cluster workload). BLIS-Evolved: 1.34% E2E MAPE. LLMServingSim: 76.00%. BLIS-Roofline: 97.69%. BLIS-Evolved 72.8× more accurate than roofline, 56.7× better than LLMServingSim on this high-TP MoE configuration.

**Accuracy varies dramatically across simulators.** Across their supported experiments, median E2E error ranges from 7.4% (BLIS-Evolved) to 619% (Vidur) - an 83× spread. TTFT error spans 12.6% to 29,783% (2,355× spread). ITL error ranges 9.8% to 259% (26× spread). These gaps are not rounding errors, they determine whether your capacity plan holds or collapses on launch day.

<a name="speed-vs-accuracy-the-pareto-frontier"></a>
## Speed vs. Accuracy: The Pareto Frontier

Accuracy tells you whether to trust a simulator. Speed tells you whether you can practically use it.

| Simulator | Median Runtime (s) | Speedup vs. Real |
|-----------|-------------------|------------------|
| **LLM-Optimizer** | 0.1 | 22,166× |
| **BLIS-Evolved** | 0.8 | 1,479× |
| **BLIS-Roofline** | 1.6 | 766× |
| **AIConfigurator** | 3.5 | 349× |
| **Vidur** | 9.1 | 134× |
| **LLMServingSim** | 353.3 | 0.4× |

Median runtime and speedup relative to real GPU experiments. LLM-Optimizer is 22,000× faster, BLIS-Evolved 1,500× faster, while LLMServingSim is actually slower than running real experiments. **Practical impact?** Simulating 100 experiments takes 10 seconds (LLM-Optimizer), 80 seconds (BLIS-Evolved), 15 minutes (Vidur), or 10 hours (LLMServingSim).

![Figure 5: Pareto Frontier](figures/fig5_pareto.png)

**Figure 5:** Speed vs accuracy Pareto frontier across all 38 experiments. LLM-Optimizer occupies the fast-but-rough corner (0.1s, instant feedback, no queueing). LLMServingSim sits in slow-but-detailed (353s, yet higher MAPE than BLIS-Evolved—fidelity alone does not guarantee accuracy). **BLIS-Evolved hits the frontier elbow** (0.8s, 11.79% E2E MAPE), balancing accuracy and speed. Median runtime with IQR error bars.

**Pick your point on the curve.** One deployment decision? BLIS-Evolved (0.8s) once. Searching 1,000 configs? 13 minutes with BLIS-Evolved vs 98 hours with LLMServingSim. Training an RL agent? Only LLM-Optimizer (0.1s) makes millions of episodes feasible.

<a name="which-simulator-for-your-use-case"></a>
## Which Simulator For Your Use Case

### Capacity Planning & Configuration Search

**Accuracy matters most.** A fast simulator with 50% error means wrong resource decisions—overprovision and waste budget, or underprovision and miss SLOs. BLIS-Evolved delivered 11.79% E2E error and 22.81% TTFT error across 38 experiments (Figure 1). Pure roofline models miss queueing delays and communication overhead—errors that compound when planning at scale.

**For capacity planning with SLO targets:** Use **BLIS-Evolved**. If your SLOs specify tail latency (P90 < 500ms, P99 < 1s), you need a discrete-event simulator that models queueing and outputs P90/P99 metrics. Analytical simulators (LLM-Optimizer, AIConfigurator) only predict mean latency and cannot validate tail latency SLOs. BLIS-Evolved covers all models/GPUs/workloads at 0.8s per run and supports vLLM arguments (chunk size, GPU memory utilization, CPU offload). Vidur provides scheduler-level detail and can be used if you have time to profile each model you want to plan for.

**For rapid config space exploration (mean latency only):** Use **LLM-Optimizer** first, then validate with BLIS-Evolved. At 0.1 seconds per config, LLM-Optimizer sweeps 1,000 candidates in 2 minutes to eliminate obviously bad configs (wrong TP, insufficient memory). But roofline accuracy degrades on high-parallelism and MoE workloads (Figure 1)—validate final candidates with BLIS-Evolved (0.8s per config, 13 minutes for 1,000 runs) before making resource commitments.

**LLMServingSim** At ~6 minutes per run, LLMServingSim is too slow for the iterative config exploration capacity planning requires.

### AI-Driven Algorithm Discovery

An exciting emerging area: using RL or automated search to *discover* better serving algorithms ([ADRS](https://sky.cs.berkeley.edu/project/adrs/)). The simulator becomes a training environment for exploring scheduling policies and batching strategies.

**Speed dominates.** Training loops need many simulations. At 353 seconds per run, LLMServingSim is impractical. Even Vidur at 9.1 seconds adds up fast. You need sub-second simulation for training to complete in reasonable time.

**LLM-Optimizer (0.1s) is the natural fit for single-instance simulation.** Fast enough for large-scale exploration when you are discovering scheduling policies or batching strategies for a single server. Misses queueing dynamics, but for discovery you are learning *relative* performance across policies, not absolute accuracy. Limited to single-instance—cannot model multi-instance features like routing.

**BLIS-Evolved (0.8s) offers a middle ground with multi-instance support.** Captures queueing effects and supports multi-instance features like routing policies across replicas. Tradeoff: 8× slower than LLM-Optimizer, but enables exploration of distributed serving algorithms.

**AIConfigurator and Vidur also support multi-instance scenarios** but runtime makes large-scale training impractical. Better for validating hand-designed multi-instance systems. LLMServingSim does not scale for training loops.

<a name="the-bottom-line"></a>
## The Bottom Line

There is no universal best simulator - only the best simulator *for your problem*.

Across 38 experiments, we measured a massive accuracy spread between simulators. We evaluated on three axes: **Accuracy** (can you trust it?), **Speed** (can you explore with it?), **Coverage** (can it model your deployment?). Fast but inaccurate wastes time. Accurate but slow limits exploration. Neither matters if coverage gaps block your architecture.

**BLIS-Evolved** hits the sweet spot: high accuracy, extensive coverage, moderate speed. 
**LLM-Optimizer** dominates speed at 0.1s for rapid exploration. **Vidur** provides scheduler-level fidelity for focused research.

Marketing claims are not validation. Run the simulator on *your* model, *your* hardware, *your* workload, then compare against real measurements. Test before you trust.
