# Step-Time Prediction Research Design

**Status:** Draft
**Date:** 2026-02-26
**Species:** Specification (new subsystem with precise behavioral requirements)
**Type:** Research methodology design
**Target venue:** Top-tier AI/systems conference (MLSys, OSDI, EuroSys, NSDI)

## Motivation

BLIS has two latency estimation modes: blackbox (alpha/beta regression) and roofline (analytical FLOPs/bandwidth). The blackbox model has known accuracy limitations — its step-time estimation uses only 2 features and ignores batch composition details, KV cache lengths, and architecture-specific compute patterns; its scheduling and preemption overhead methods return 0 (not modeled). This research aims to replace the blackbox model with a data-driven alternative covering all 5 LatencyModel methods. **The roofline model is unaffected by this research and will continue to serve users who provide hardware and model configuration files.** This design specifies the research pipeline, evaluation framework, and integration path for developing that blackbox replacement. The primary success metric is workload-level E2E mean error < 10%.

## Scope

**In scope:**
- Replacing the blackbox (alpha/beta regression) latency model with a data-driven alternative — **all 5 LatencyModel methods** are in scope: step-time estimation (primary target), queueing-time estimation, output token processing time, scheduling processing time, and preemption processing time
- Single-instance latency prediction (the full LatencyModel behavioral contract)
- Dense and MoE transformer architectures on H100 GPUs
- Statistical, analytical, ML-based, evolutionary (e.g., OpenEvolve, GEPA), and hybrid approaches
- Integration path back to BLIS via the policy template extension recipe (new algorithm behind existing LatencyModel interface)

**Not affected (preserved as-is):**
- The roofline latency model — this analytical FLOPs/bandwidth model is a separate, independent implementation that will continue to be available for users who provide hardware specs and model config. It is used only as a comparison baseline during this research. The roofline model's implementations of all 5 methods remain untouched.

**Explicitly out of scope:**
- Multi-instance (cluster-level) effects — latency estimation is an instance-local computation
- Non-H100 hardware (validation dimension only; training uses H100 data exclusively)
- vLLM versions other than v0.15.1 (robustness evaluation only, not training)

**Deferred:**
- Production deployment of the winning model (separate macro plan)
- Automated retraining pipeline (future work after initial research)

## Analysis Questions

This research design helps answer the following questions:

| # | Priority | Question | Why It Matters |
|---|----------|----------|----------------|
| AQ-1 | P1 | Can a data-driven latency model achieve <10% workload-level E2E mean error across all 16 experiments? | For each experiment (model × workload), the simulation produces one mean E2E latency. That value must be within 10% of ground truth. This is the primary go/no-go metric — it directly measures simulation fidelity at the level users care about. |
| AQ-2 | P1 | What per-step MAPE does the winning model achieve, and how does per-step error propagate to workload-level E2E mean? | Per-step accuracy is the foundational building block. The relationship between per-step MAPE and workload-level E2E mean error determines whether per-step improvements translate to simulation fidelity. |
| AQ-3 | P2 | How does per-step prediction error propagate to TTFT mean and ITL mean? | TTFT and ITL mean are P2 metrics. Per-step errors may compound or cancel differently for prefill-dominated (TTFT) vs. decode-dominated (ITL) paths. |
| AQ-4 | P4 | Which features are causally relevant to step duration vs. merely correlated with system load state? | Prevents overfitting to scheduler artifacts that won't generalize across policies |
| AQ-5 | P4 | Do dense and MoE architectures require separate models, or can a unified model capture both compute patterns? | Determines whether the blackbox replacement needs architecture-specific variants or a single unified model suffices |
| AQ-6 | P4 | What is the minimum calibration data needed for a new model/GPU combination? | Determines the retraining burden for users deploying BLIS with unseen configurations |

## Problem Statement

BLIS has two step-time estimation modes: blackbox (3-coefficient linear regression) and roofline (analytical FLOPs/bandwidth with MFU lookup). The blackbox model uses only 3 coefficients (intercept + prefill token weight + decode token weight) and ignores batch composition details, per-request KV cache lengths, and architecture-specific compute patterns. **This research replaces the blackbox model only. The roofline model is unaffected and continues to serve as an independent, analytically-grounded alternative for users with hardware/model configuration files.**

We have ~165K step-level ground-truth data points from instrumented vLLM (v0.15.1) with journey/step tracing and KV events, covering 4 model families (dense + MoE), 4 workload types, and multiple TP configurations on H100 GPUs.

**Algorithm scope:** Not restricted to ML. Research ideas may propose statistical, analytical, physics-informed, machine learning, evolutionary, or hybrid approaches. Evolutionary techniques are explicitly encouraged — including LLM-guided program synthesis tools such as OpenEvolve (MAP-Elites + LLMs for algorithm discovery; github.com/algorithmicsuperintelligence/openevolve) and GEPA (genetic-Pareto optimization with LLM-powered reflection; github.com/gepa-ai/gepa). These are particularly suited to this problem because: (a) the evaluation function (workload-level E2E mean error) is well-defined and cheap to compute, (b) the search space (prediction formulas, feature combinations, piecewise models) is expressible as code, and (c) evolutionary approaches can discover non-obvious functional forms that hand-designed regression or neural architectures might miss. Every idea must cite relevant prior work and justify its algorithmic choice with first-principles reasoning. Each approach must be novel or distinct from prior art in the systems/ML literature.

**Goal:** Design prediction algorithms to replace the blackbox latency model (all 5 LatencyModel methods) — achieving <10% workload-level E2E mean error while generalizing across models, workloads, hardware, and vLLM configurations. Step-time estimation is the primary research target; the other 4 methods (queueing, output token processing, scheduling, preemption) are secondary targets that contribute to E2E fidelity. The roofline model remains available as an independent option; this research does not modify it. Approaches may use any algorithmic family: statistical regression, analytical models, ML (tree ensembles, neural networks), evolutionary program synthesis (OpenEvolve, GEPA), or hybrids. Each approach must cite relevant literature and meet the rigor standards of a top-tier AI/systems venue (MLSys, OSDI, EuroSys, NSDI).

## Ground-Truth Data

### Source

20 experiments (16 main + 4 sweep) produced by a Tekton pipeline running the instrumented vLLM fork. The data collection infrastructure is documented in the companion data collection repository.

### Step-Level Schema (from `traces.json` step.BATCH_SUMMARY events)

#### Feature Categories

**Batch computation features** (causally related to step time — these determine FLOPs and memory access):

| Feature | Type | Description | Causal Role |
|---------|------|-------------|-------------|
| `step.duration_us` | int | **Ground truth target** — step execution time in microseconds. Defined as `ts_end_ns - ts_start_ns` (wall-clock duration of the model execution call in vLLM). Includes GPU compute, memory access, and kernel launch overhead but excludes scheduler overhead and Python-side batch preparation. | Target |
| `batch.prefill_tokens` | int | Prefill tokens in this step | Determines prefill FLOPs (linear in tokens) |
| `batch.decode_tokens` | int | Decode tokens in this step | Determines decode FLOPs (linear in tokens) |
| `batch.num_prefill_reqs` | int | Prefill request count | Determines attention kernel invocations |
| `batch.num_decode_reqs` | int | Decode request count | Determines attention kernel invocations |
| `batch.scheduled_tokens` | int | Total tokens scheduled | Sum of prefill + decode tokens |

**System state features** (correlated with step time but not causally deterministic — these reflect scheduler state, not batch computation):

| Feature | Type | Description | Causal Role |
|---------|------|-------------|-------------|
| `queue.running_depth` | int | Active batch size (requests) | Proxy for batch size (already captured above); may introduce spurious correlations with scheduling policy |
| `queue.waiting_depth` | int | Queue depth | No physical relationship to step execution time; correlated with arrival rate, not computation |
| `kv.usage_gpu_ratio` | float | GPU KV cache utilization | Global cache state, not per-batch computation; high utilization correlates with large batches but the causal path goes through batch composition |
| `kv.blocks_free_gpu` | int | Free GPU KV blocks | Same concern as kv_usage_gpu_ratio |
| `kv.blocks_total_gpu` | int | Total GPU KV blocks | Constant per experiment; useful only as experiment metadata |

**Step lifecycle features** (potential data leakage risk):

| Feature | Type | Description | Leakage Risk |
|---------|------|-------------|--------------|
| `batch.num_finished` | int | Requests finishing this step | Measured after step execution — this is an *output*, not an input. Leaks future information if used as a predictor. |
| `batch.num_preempted` | int | Preemptions this step | Same leakage concern as num_finished |
| `step.ts_start_ns` | int | Step start timestamp (nanoseconds) | Time metadata only |
| `step.ts_end_ns` | int | Step end timestamp (nanoseconds) | Encodes the target (duration = end - start). Must not be used as a feature. |
| `step.id` | int | Sequential step identifier | Monotonic counter; useful for ordering, not prediction |

#### Known Feature Gaps

The step-level schema above has two structural gaps that research ideas must address:

**Gap 1: No per-request KV cache lengths.** The aggregate features (`prefill_tokens`, `decode_tokens`) cannot distinguish batches with identical token counts but different KV length distributions. In vLLM, attention FLOPs for decode scale as O(batch_size * kv_len * head_dim * num_heads) per request, and `kv_len` varies dramatically in continuous batching. BLIS's own hypothesis H8 showed that using `max_kv_len` for all requests overestimated attention FLOPs by 12.96x at batch_size=16. Features like `mean_kv_len`, `max_kv_len`, `kv_len_variance`, or per-request KV length histograms are derivable from the per-request lifecycle data and should be considered.

**Gap 2: No MoE-specific features.** Mixtral-8x7B uses 8 experts with top-2 routing, meaning only ~25% of parameters are active per token. No features capture active expert count, expert load balance, or tokens-per-expert. Research ideas involving MoE must either (a) propose MoE-specific features derivable from the available data, (b) treat MoE as a separate modeling problem, or (c) exclude Mixtral from dense-architecture models and report separate MoE results.

**Gap 3: No prefix cache hit information.** With `prefix_caching=True`, cache hits reduce effective prefill tokens, but `prefill_tokens` may reflect the full input length rather than post-cache-hit count. Research ideas should clarify this semantics and consider a `cache_hit_ratio` feature if available.

#### Real-System Correspondence

| Prediction Concept | vLLM v0.15.1 | SGLang (approximate) | llm-d (approximate) |
|---|---|---|---|
| Prefill tokens | Scheduler output: number of prefill tokens | Equivalent scheduler output | Forwarded from vLLM/SGLang backend |
| Decode tokens | Scheduler output: total batched tokens minus prefill tokens | Equivalent scheduler output | Forwarded from backend |
| KV usage GPU ratio | Block manager GPU cache utilization query | Equivalent KV cache manager query | Instance-level metric |
| Step time target | Wall-clock duration of model executor call | Wall-clock of model forward pass | Instance-level step duration |
| Batch composition | Scheduler output: list of scheduled sequence groups | Equivalent batch representation | Backend-dependent |

**Note:** The training data is from vLLM v0.15.1 only. SGLang and llm-d mappings are approximate and provided for context on real-system correspondence. Transferability to other serving systems is an evaluation dimension, not a training requirement.

#### Sampling Bias Characterization

**Sampling note:** 10% of steps are traced (`step_tracing_sample_rate=0.1`). The ~165K sampled steps represent ~1.65M total steps.

**Bias risk:** The 10% sampling rate may introduce systematic bias if:
- Sampling is periodic (every 10th step) rather than random → correlated with batch formation cycles
- Sampling overhead affects the traced step's duration → measurement bias in the target variable
- Warm-up/cool-down phases are over/under-represented → non-stationary distribution

**Mitigation (required in Phase 0):** Before any modeling begins, the shared infrastructure must characterize the sampling distribution by comparing: (1) the distribution of `step_id` values in the sample vs. uniform random, (2) whether traced steps have systematically different durations than expected (detectable via autocorrelation analysis on consecutive step IDs), and (3) whether all 16 model×workload combinations have proportional representation.

### Model/GPU/TP Coverage

| Model | Architecture | TP | Workloads |
|-------|-------------|-----|-----------|
| Llama-2-7B | Dense | 1 | general, codegen, roleplay, reasoning |
| Llama-2-70B | Dense | 4 | general, codegen, roleplay, reasoning |
| Mixtral-8x7B-v0.1 | MoE (8 experts, top-2) | 2 | general, codegen, roleplay, reasoning |
| CodeLlama-34B | Dense | 2 | general, codegen, roleplay, reasoning |

All on H100 80GB, vLLM v0.15.1, `max_model_len=4096`, `max_num_batched_tokens=2048`, `max_num_seqs=128`, chunked prefill enabled, prefix caching enabled.

**MoE note:** Mixtral-8x7B has 46.7B total parameters but only ~12.9B are active per token (top-2 of 8 experts). This means: (a) GEMM shapes differ from dense models of similar total parameter count, (b) memory access patterns differ (loading different expert weights per token), and (c) the existing roofline model has no MoE-specific logic. Research ideas must address this explicitly.

**MoE validation requirement:** Because there is only one MoE model in the dataset (Mixtral-8x7B), MoE generalization cannot be tested via leave-one-model-out within the MoE category. Instead, MoE validation uses: (1) leave-one-workload-out within the Mixtral experiments (4 workloads provide 4-fold cross-validation), and (2) comparison of dense-trained vs. unified model predictions on MoE data to assess whether MoE-specific modeling is necessary. Success criterion for MoE: workload-level E2E mean error < 15% on all 4 Mixtral-8x7B experiments.

**vLLM version scoping:** All data is from vLLM v0.15.1. The vLLM scheduler has undergone significant changes across versions (v1 scheduler rewrite, FlashInfer integration, chunked prefill changes). Version robustness is an *evaluation dimension* (measuring which scheduler changes would invalidate predictions), not a training requirement. The model is scoped to v0.15.1 behavior.

### Additional Data Sources

- **KV events** (`kv_events.jsonl`): BlockStored, CacheStoreCommitted, TransferInitiated/Completed — useful for KV cache offloading research.
- **Per-request lifecycle** (`per_request_lifecycle_metrics.json`): Per-token timestamps, input/output token counts — useful for request-level validation.
- **MFU benchmarks** (`InferSim/bench_data/`): Kernel-level GEMM and attention MFU data by GPU/attention-config — useful for physics-informed features.

## Modeling Decisions

Per Section 2.1 of the Design Guidelines, what is modeled, simplified, and deliberately omitted. Each entry has been evaluated against the six Banks et al. model scoping criteria (design guidelines Section 2.1).

| Aspect | Treatment | Justification | Lost Behavior (if simplified/omitted) |
|--------|-----------|---------------|---------------------------------------|
| **Batch composition features** (prefill/decode tokens, request counts) | **Modeled** | Direct causal relationship to step FLOPs and memory access | — |
| **Per-request KV cache lengths** | **Modeled** (derived from per-request lifecycle data) | Attention FLOPs scale with per-request kv_len; aggregate tokens are insufficient (H8 evidence) | — |
| **Model architecture (dense vs. MoE)** | **Modeled** as experiment-level metadata and potential feature | Fundamentally different compute patterns require explicit handling | — |
| **Chunked prefill interaction** | **Modeled** implicitly (mixed batches are the norm with chunked prefill enabled) | With chunked prefill, nearly every step contains both prefill and decode tokens. The prefill/decode interaction is non-additive (BLIS H5). | — |
| **Hardware characteristics** | **Simplified** — H100-only for training; hardware generalization evaluated but not trained **(P5 — lower priority)** | Single-GPU-generation data; cross-hardware transfer is an evaluation dimension, not a training target. Hardware generalization is lower priority than E2E/TTFT/ITL mean accuracy. | Lost: GPU-specific memory bandwidth curves, cache hierarchy effects, SM occupancy differences across GPU generations. Predictions may not transfer to A100/H20 without recalibration of hardware-dependent coefficients. |
| **vLLM scheduler internals** (batch formation algorithm, preemption policy) | **Modeled** — `SchedulingProcessingTime()` and `PreemptionProcessingTime()` are in scope for calibration; currently return 0 | Step time depends on *what* is in the batch; scheduling/preemption overhead depends on *how* the batch was formed. Both contribute to E2E latency. Ground-truth traces contain timing data for scheduling iterations. | Partially recovered: scheduling overhead at high queue depths and preemption swap-in/out latency can be calibrated from traces. Still lost: scheduler algorithmic internals (priority sorting complexity, KV allocation search). |
| **Kernel-level scheduling** (CUDA stream management, kernel launch overhead, CUDA graph compilation) | **Omitted** | Below the abstraction level of per-step prediction; captured implicitly in the target variable | Lost: CUDA graph capture/replay transitions that cause occasional latency spikes, kernel launch overhead variance under high GPU utilization, stream synchronization costs. These effects are subsumed into the step duration measurement. |
| **Multi-instance effects** (routing, admission, snapshot staleness) | **Omitted** | Step time is instance-local; cluster effects are orthogonal (see Multi-Instance Scope section) | Lost: None for step-time prediction. Multi-instance effects influence *which* batches are formed, not *how long* a formed batch takes to execute. |
| **System state features** (running_depth, waiting_depth, kv_blocks_free) | **Simplified** — available but flagged as potentially spurious | These proxy for batch size and arrival rate; including them risks overfitting to scheduler policy rather than computation | Lost: Indirect signals about memory pressure (high kv_usage may correlate with slower memory access due to fragmentation). However, this correlation is mediated through batch composition, which is already modeled. |
| **Post-step features** (num_finished, num_preempted) | **Omitted from prediction** | These are step outputs, not inputs — using them as features creates data leakage | Lost: None — these are measured after step execution; omitting them is correctness, not a tradeoff. |
| **Temporal dynamics** (step-to-step correlation, warm-up transients) | **Simplified** — each step predicted independently | Modeling temporal dependencies adds complexity; independent prediction is the simplest approach that answers AQ-1 | Lost: CUDA cache warming effects (first few steps of a new model are slower), KV cache fragmentation accumulation over time, thermal throttling under sustained load. Research ideas proposing temporal features may recover some of this signal. |
| **Prefix cache hit effects** | **Simplified** — noted as feature gap, addressed per-idea | Cache hits reduce effective prefill tokens; semantics must be clarified during data loading | Lost: Distinction between cold-start prefill and cache-hit prefill. A cache hit reduces effective compute but `prefill_tokens` may report the full input length. |
| **Quantization (precision) effects** | **Simplified** — all training data uses the default vLLM precision (BF16 on H100); quantization is not varied **(P6 — lowest priority)** | Training data is from a single precision configuration; quantization effects are the lowest evaluation priority. Report BF16 results as the primary finding; FP8/INT4 transferability is informational only. | Lost: FP8 on H100 Hopper achieves ~2× the BF16 TFLOP/s; INT4/INT8 quantization changes both compute throughput and memory bandwidth characteristics. A model trained on BF16 data would need recalibration for quantized deployments. This is informational only, not a training variable or success gate. |

### Banks et al. Six Criteria Evaluation

Each modeling decision was evaluated against the six criteria from Banks et al. (design guidelines Section 2.1):

| Aspect | Can AQ be answered without it? | Does omitting change the answer? | Does it change other components? | Accuracy lost? | Complexity added? | Data available? |
|--------|------|------|------|------|------|------|
| Batch composition | No — these are the primary prediction inputs | Yes — predictions would be constant | No | Total | None (direct features) | Yes |
| Per-request KV lengths | Partially — aggregate tokens provide a proxy | Yes — 12.96× overestimate at batch_size=16 (H8) | No | High for heterogeneous batches | Moderate (need per-request derivation) | Yes (from lifecycle data) |
| Model architecture | Partially — a single model could learn both | Yes — MoE has 25% active parameters | No | Moderate to high for MoE | Low (experiment metadata) | Yes |
| Chunked prefill | Partially — token counts capture it | Yes — non-additive interaction (H5) | No | Moderate | None (already in batch features) | Yes |
| Hardware characteristics | Yes for H100-only | Yes for cross-hardware | No | None for H100; unknown for others | High (multi-GPU calibration) | H100 only |
| vLLM scheduler internals | Partially — scheduling/preemption overhead is measurable | Yes — 0-valued methods underpredict E2E at high preemption rates | No | Moderate (accumulates across preemption cycles) | Low (calibrated constants) to Moderate (data-driven) | Yes (timing data in traces) |
| Kernel-level scheduling | Yes — absorbed into target | Marginally — occasional spikes | No | Low (noise level) | Very high | No (need profiling) |
| System state features | Yes — proxied by batch composition | Marginally — fragmentation effects | No | Low | Low (direct features) | Yes |
| Temporal dynamics | Mostly — independent prediction sufficient | Marginally — warm-up effects | No | Low to moderate | High (sequence model) | Yes (step ordering) |
| Quantization effects | Yes for BF16-only | Yes for cross-precision | No | None for BF16; unknown for FP8/INT8 | High (multi-precision calibration) | BF16 only |

## DES Integration Path

The winning research model must integrate back into BLIS as a third LatencyModel implementation, replacing the blackbox model as the default for users without hardware/model config files. **The roofline model is completely unaffected — it remains the preferred option for users who can provide GPU specs and HuggingFace model configs.**

### Extension Type: Policy Template (New Latency Model Backend)

Per the Design Guidelines (Section 5.2), this is a **policy template** — a new algorithm behind the existing LatencyModel interface. The LatencyModel interface already exists and has two implementations (blackbox and roofline), so no interface extraction is needed (which distinguishes this from a backend swap, Section 5.4). The recipe:

1. The new implementation satisfies the existing 5-method LatencyModel behavioral contract
2. Registration via the existing latency model factory pattern (package-level registration at import time)
3. Selection via configuration: the StepML model becomes the new default for users without hardware/model config files; the roofline model remains the preferred option when `--roofline` is used
4. No changes to batch formation, scheduling, roofline, or other modules

### No-Op Default (Graceful Degradation)

When StepML model artifacts are absent (no trained weights file found at the configured path), the factory falls back to the existing blackbox model. This ensures:
- Existing users who upgrade BLIS see no behavior change until they explicitly provide StepML artifacts
- CI/CD tests continue to pass without requiring trained model files
- The migration path is opt-in: users must train a StepML model and configure its artifact path to activate it

### Factory Signature Compatibility

The existing factory accepts two parameters: calibration coefficients and model/hardware configuration. The StepML model requires different construction-time inputs: a trained model artifact path and model architecture metadata. Two integration approaches:

| Approach | Description | Tradeoff |
|----------|-------------|----------|
| **Extend existing config** | Add optional StepML fields to the existing configuration types. If present and artifact exists, construct StepML; otherwise fall back to blackbox. | Minimal disruption; config grows over time |
| **Separate factory** | Register a second factory for StepML alongside the existing one. Selection based on a mode flag (blackbox/stepml/roofline). | Cleaner separation; requires plumbing a mode selector |

**Decision deferred** to the production macro plan. The research phase validates that a model *can* satisfy the behavioral contract; the factory wiring is an integration detail.

### Go Parallel Development Path

During research (Python), the existing Go latency model code is unmodified. The Go integration PR can be developed independently after research concludes:
- **Research team** works in Python under `hypotheses/h-stepml/`
- **Integration team** (future) writes the Go implementation in `sim/latency/` once the winning model is selected
- No merge conflicts because the research artifacts and Go code are in non-overlapping directories

### Cluster-Level Configuration

The StepML model is instance-local (same as blackbox and roofline). In cluster mode, each instance independently constructs its own LatencyModel. No cluster-level configuration changes are needed for the research phase. For production integration, the model/hardware configuration types will likely need extension to include a model artifact path — this is an integration detail addressed in the production macro plan, not a cluster-level architectural change.

### LatencyModel Behavioral Contract Compatibility

The LatencyModel behavioral contract defines 5 methods. **All 5 methods are in scope** for the StepML replacement. The roofline model's implementations of all 5 methods remain untouched.

#### E2E Latency Decomposition Through the Simulator

A request's end-to-end latency in BLIS is the sum of delays contributed by each LatencyModel method plus emergent queueing time. Tracing through the simulator code:

```
Request lifecycle (sim/event.go + sim/simulator.go):

[1] ArrivalEvent.Execute()           → QueueingTime(req)              sim/event.go:31
    Adds arrival-to-queue delay. Currently: alpha0 + alpha1*inputLen

[2] QueuedEvent.Execute()            → (emergent wait in WaitQ)       sim/event.go:52-63
    Request waits until a StepEvent schedules it. Duration is emergent
    from simulation dynamics — depends on accuracy of ALL other components.

[3] VLLMBatchFormation.FormBatch()   → SchedulingProcessingTime()     sim/batch_formation.go:131
    Per-request scheduling overhead when admitted to batch.
    Currently: returns 0

[3b] preemptForTokens()              → PreemptionProcessingTime()     sim/batch_formation.go:160
     If KV pressure forces eviction. Currently: returns 0

[4] executeBatchStep()               → StepTime(batch)                sim/simulator.go:415
    GPU execution time for the formed batch. The primary target.
    Currently: beta0 + beta1*prefill + beta2*decode

[4b] executeBatchStep()              → ConsumePendingTransferLatency() sim/simulator.go:418
     KV cache CPU↔GPU transfer cost. 0 for single-tier.
     Handled by KV subsystem, not LatencyModel.

[5] executeBatchStep() per token     → OutputTokenProcessingTime()    sim/simulator.go:433,466
    Added to each decode token's ITL. Currently: constant alpha2

TTFT(req) = [1] + [2] + [3] + [4_first_step] + [5]
ITL(req)  = [4_per_decode_step] + [5]
E2E(req)  = [1] + [2] + [3] + Σ[4] + Σ[5] + optional([3b] + [4b])
```

**Key insight for research:** Improving only `StepTime` [4] leaves 4 other controllable delays at their current values (0 or fixed constants). Errors in [1], [3], and [5] compound into E2E error even if [4] is perfect. Conversely, calibrating [1]/[3]/[5] to compensate for systematic bias in [4] is a valid optimization strategy.

| Method (behavioral description) | Research Scope | Current Value | Integration Strategy |
|--------|---------------|---------------|---------------------|
| **Step-time estimation** [4]: given a batch of requests, return the predicted step duration in microseconds | **Primary target** — replaces the 2-feature, 3-coefficient linear regression | `beta0 + beta1*prefill + beta2*decode` | Data-driven prediction (blackbox replacement only; roofline is unaffected) |
| **Queueing-time estimation** [1]: given a request, estimate arrival-to-queue delay | **In scope** — currently uses 2 alpha coefficients | `alpha0 + alpha1*inputLen` | Data-driven or improved coefficient-based prediction |
| **Output token processing time** [5]: per-token post-processing overhead | **In scope** — currently uses 1 alpha coefficient | constant `alpha2` | Data-driven or improved coefficient-based prediction |
| **Scheduling processing time** [3]: per-request scheduling overhead | **In scope** — currently returns 0 (not modeled) | **0** | Data-driven prediction or calibrated constant; the current 0-return may underestimate scheduling overhead at high queue depths |
| **Preemption processing time** [3b]: per-preemption overhead | **In scope** — currently returns 0 (not modeled) | **0** | Data-driven prediction or calibrated constant; the current 0-return ignores swap-in/out latency |

**Feature availability at prediction time:** The step-time estimation method receives a batch of Request objects. Each Request provides fields relevant to step-time prediction: input token sequence, output token sequence, a progress index tracking how many tokens have been generated so far, and the number of new tokens to generate this step. The ground-truth data schema has aggregate batch features. Research models must demonstrate that their required features are derivable from the Request batch available at call time, or propose how to bridge this gap.

**Progress index as KV cache length proxy (Gap 1 bridge):** The progress index on each Request tracks the cumulative token progress: input tokens processed so far plus output tokens generated so far. During prefill, progress index < input length (still processing input tokens). Once prefill completes, progress index >= input length, and the excess equals output tokens generated. For a decode request, the progress index itself approximates the total KV cache length for that request (since it counts all tokens processed). This provides the per-request KV cache length signal identified in Gap 1 without requiring any interface changes. Research models requiring per-request KV lengths should use the progress index directly as the KV length proxy.

**Inference latency constraint:** The step-time estimation method is called once per scheduler step in the DES event loop. Prediction must complete in <1ms to avoid dominating simulation wall-clock time. **Analysis:** At 128 requests per batch, a linear model requires ~128 feature extractions + one matrix multiply (<1μs). A tree ensemble (100 trees) requires ~100 tree traversals (~10μs). A small neural network (2 hidden layers, 64 units) requires ~5K multiply-adds (~50μs). An evolved prediction function (typical output of OpenEvolve/GEPA) is usually a compact formula or piecewise function (<1μs). All are well within the 1ms budget. Only approaches requiring GPU inference or large (>10M parameter) networks per step are excluded. Note: evolutionary search (OpenEvolve, GEPA) is expensive during *training* (hundreds of LLM evaluations) but the *evolved artifact* is typically a simple, fast function — the training cost is amortized.

### Python-to-Go Integration Strategy

Research experimentation uses Python (scikit-learn, XGBoost, PyTorch). The winning model must eventually run in Go for BLIS integration. Three integration paths exist:

| Path | Description | Tradeoffs |
|------|-------------|-----------|
| **Go-native reimplementation** | Rewrite the prediction logic in Go | Best performance; most engineering effort; only viable for simple models (linear, tree ensemble with Go libraries) |
| **ONNX export + Go runtime** | Export trained model to ONNX, use Go ONNX runtime | Moderate performance; works for tree ensembles and neural networks; adds ONNX dependency |
| **Coefficient export** | Export learned coefficients/weights, implement prediction formula in Go | Best for parametric models (regression, piecewise linear); trivial integration; limited to models expressible as closed-form formulas |
| **Evolved code translation** | Translate evolved Python prediction code to Go directly | Best for evolutionary approaches (OpenEvolve, GEPA) that produce interpretable code; the evolved function is typically simple enough to port line-by-line; may require manual cleanup of Python idioms |

**Decision deferred:** The integration path depends on the winning model's complexity. Research ideas should note which path they would require. The macro plan for production deployment (separate from this research design) will specify the chosen path.

## Module Contract: StepML LatencyModel

Per the Design Guidelines (Section 4), the six-aspect behavioral contract for the new LatencyModel variant:

| Aspect | Specification |
|--------|--------------|
| **Observes (per call)** | For step-time estimation: batch of Request objects — per-request input length, output length, cumulative progress (KV cache length proxy), and per-step token count. For queueing-time estimation: a single Request with input length. For output/scheduling/preemption overheads: no per-call inputs (may use construction-time calibration). Scheduling-policy-derived fields (priority, SLO class, queue depth) are available but should not be read — step-time depends on batch computation, not scheduling state. |
| **Observes (construction time)** | Model architecture metadata (dense/MoE, layer count, hidden dimension); trained model artifact (weights or coefficients); optional hardware calibration parameters |
| **Controls** | All 5 latency estimates (integer microseconds): step time (primary), queueing time, output token processing time, scheduling processing time, preemption processing time |
| **State owned** | Trained model weights/coefficients (immutable after construction); model architecture config; hardware config. All state is set at construction time and never mutated. |
| **Invariants maintained** | INV-M-1: Step-time estimate is positive for any non-empty batch. INV-M-2: Step-time estimation is deterministic — same batch produces same estimate. INV-M-3: Step-time estimation is side-effect-free (pure function of batch + immutable state). **Performance requirement:** Prediction latency < 1ms at maximum batch size on any machine capable of running BLIS (INV-M-4). **Soft expectation:** Monotonicity in total tokens (INV-M-5). **P1 fidelity:** Bounded systematic bias |MSPE| < 5% (INV-M-6, connects per-step accuracy to E2E mean fidelity). |
| **Events produced/consumed** | None — step-time estimation is a synchronous query called by the event loop during step execution |
| **Extension friction** | ~3-4 files for the StepML integration: implementation file in `sim/latency/`, factory modification, config type extension for artifact path, and CLI flag addition. This exceeds the ~2 file reference target for "New latency model backend" (design guidelines Section 4.5) because the StepML model requires new construction-time inputs (artifact path) not present in the existing config types. This friction is acceptable because it is a one-time cost for a qualitatively different model type (data-driven with external artifacts vs. formula-based). Zero changes to batch formation, scheduling, routing, or KV cache modules. |

### All Five Methods In Scope

All 5 LatencyModel methods are targets for the StepML replacement. Step-time estimation is the primary research target (it has the highest impact on E2E mean latency). The other 4 methods contribute to TTFT, ITL, and E2E through queueing delay, per-token overhead, scheduling overhead, and preemption cost:

- **Queueing-time estimation** [1] affects TTFT directly (arrival-to-first-token delay includes queueing delay). Currently `alpha0 + alpha1*inputLen`. If this systematically over/underpredicts, all downstream latencies (TTFT, E2E) inherit the bias. Research ideas may propose data-driven improvements or retain the current implementation if the E2E mean error is dominated by step-time errors.
- **Output token processing time** [5] affects ITL (added to each decode step's inter-token latency). Currently constant `alpha2`. If the constant is wrong by X μs and a request generates N tokens, the E2E error accumulates X×N μs. May benefit from data-driven calibration that varies with model size, TP degree, or output position.
- **Scheduling processing time** [3] currently returns 0. In real vLLM, `scheduler.schedule()` performs queue scanning, KV block allocation, and priority sorting — measurable overhead that scales with queue depth and batch size. Ground-truth traces contain timing data for this.
- **Preemption processing time** [3b] currently returns 0. In real vLLM, preemption involves KV block release, request state bookkeeping, and re-queuing. Preemption events are observable in ground-truth traces.

**Three optimization strategies:**

1. **Per-component accuracy:** Improve each method independently to minimize its own prediction error. Straightforward but may miss compositional effects.
2. **End-to-end calibration:** Jointly optimize all 5 methods to minimize workload-level E2E mean error. Allows intentional over/underprediction of one component to compensate for systematic bias in another. For example, if StepTime systematically overpredicts by 5%, reducing OutputTokenProcessingTime can partially compensate.
3. **Error attribution first:** Measure which components contribute most to E2E error before optimizing. If QueueingTime contributes 60% of E2E error and StepTime contributes 30%, improving QueueingTime has higher ROI.

**Minimum viable scope:** A research idea that improves only step-time estimation while retaining the current implementations for the other 4 methods is acceptable — the success criterion is workload-level E2E mean accuracy, and step-time dominates the E2E budget. However, ideas that improve multiple methods or pursue end-to-end calibration may achieve better E2E fidelity with less per-component effort.

### Construction and Failure Behavior

- **Successful construction:** The factory creates a StepML LatencyModel when a valid model artifact exists at the configured path and the artifact format matches the expected version.
- **Artifact not found:** The factory falls back to constructing the blackbox LatencyModel (no-op default — existing behavior preserved). **Note:** This fallback is new behavior to be implemented during integration. The current factory does not implement fallback — it returns errors on invalid configuration. The StepML integration PR will add the three-way dispatch (roofline → stepml → blackbox) with explicit, logged fallback rather than silent degradation.
- **Artifact corrupted/incompatible:** The factory returns an error (surfaced to the CLI layer, which logs and exits per the error handling boundary: CLI → fatal, library → error).
- **Concurrency safety:** The LatencyModel is instantiated once per simulator instance and called sequentially within the single-threaded event loop. No concurrent access to the model's immutable state occurs. In cluster mode, each instance has its own LatencyModel — no sharing across instances.

## Invariant Analysis

### Existing BLIS Invariants (INV-1 through INV-8)

| Invariant | Impact on StepML Research | Action Required |
|-----------|--------------------------|-----------------|
| **INV-1 Request conservation** | Not affected — StepML predicts step time, does not create/destroy requests | None |
| **INV-2 Request lifecycle** | Not affected — lifecycle transitions are scheduler responsibility | None |
| **INV-3 Clock monotonicity** | StepTime must return positive values; negative estimates would violate clock monotonicity when added to current time | Enforce positive output (invariant M-1 below) |
| **INV-4 KV cache conservation** | Not affected — KV allocation is separate from step time prediction | None |
| **INV-5 Causality** | Step time estimates feed into completion time calculation; underestimation could create causality-violating reorderings in edge cases. Preserved because: INV-M-1 ensures positive step times, so completion_time > schedule_time always holds. Severe underestimation could cause a request to "complete" before requests that arrived much earlier, but this is a fidelity issue (wrong timing), not a causality violation (the event ordering is still consistent with the predicted timeline). | Monitor via end-to-end validation; Stage 1 error propagation analysis will detect systematic underestimation |
| **INV-6 Determinism** | StepML model must produce identical predictions for identical inputs across runs. Verification: INV-M-2 test runs the same simulation twice with the same seed and asserts byte-identical stdout. | Enforce deterministic inference (no random sampling at prediction time); no floating-point non-determinism from unordered reductions |
| **INV-7 Signal freshness** | Not affected — StepML is called within the step execution path, not via stale snapshots | None |
| **INV-8 Work-conserving** | Not affected — work conservation is scheduler responsibility | None |

### Research-Specific Invariants

| ID | Invariant | Verification |
|----|-----------|-------------|
| **INV-M-1** | Step-time estimate > 0 for all non-empty batches; returns 0 for empty batch (no-op) | Unit test with adversarial inputs: single token, max batch (128), all-prefill, all-decode, MoE vs. dense, single-token-prefill (minimum overhead) |
| **INV-M-2** | Prediction is deterministic: identical batch input produces identical output across calls and across simulation runs | Unit test + cross-run comparison (supports INV-6 system determinism) |
| **INV-M-3** | Prediction is side-effect-free: calling step-time estimation does not modify any Request in the batch, does not modify internal model state | Unit test comparing request state and model state before/after call |
| **INV-M-4** | Prediction latency < 1ms for batch sizes up to max_num_seqs (128) | Benchmark test with p99 latency measurement (not just mean) |
| **INV-M-5** | Monotonicity (soft expectation): for a fixed request count, increasing total new tokens (prefill or decode) should not decrease the predicted step time. This is a soft expectation, not a hard architectural constraint — ML models may violate it in edge cases. Falsification threshold: >5% of test pairs violate monotonicity (see Falsification Criteria). | Empirical test with incrementally larger batches; violations indicate model learned spurious correlations. If enforcement is needed, isotonic regression post-processing or monotonic feature constraints in gradient boosting can be applied. |
| **INV-M-6** | Bounded systematic bias (P1 fidelity): the mean signed percentage error (MSPE) magnitude must be < 5%. Systematic directional bias compounds across steps within a request, degrading E2E mean latency fidelity (the P1 metric). Random errors partially cancel; systematic bias does not. | Unit test on validation set: compute MSPE across all steps; assert |MSPE| < 5%. Per-model MSPE breakdown to detect architecture-specific bias. |

### Degenerate Input Handling (R20)

The StepML model must handle these degenerate inputs explicitly:

| Input | Expected Behavior | Rationale |
|-------|------------------|-----------|
| Empty batch (no requests) | Return 0 (no-op step). Note: an empty batch should never reach the LatencyModel in normal operation — batch formation always produces non-empty batches. This is a defensive guard. If reached, the 0 return does not advance the clock, but since no step event exists for empty batches, this cannot cause an infinite loop. | Consistent with INV-M-1 boundary condition; defensive guard against batch formation bugs |
| Single-token prefill (1 request, 1 new token) | Return positive estimate (minimum step overhead) | Minimum viable batch; must not return 0 |
| Maximum batch (128 decode requests) | Return positive estimate within reasonable range | Upper bound of normal operation |
| All-prefill batch (no decode requests) | Return positive estimate | Pure prefill step (first step of many requests) |
| All-decode batch (no prefill requests) | Return positive estimate | Pure decode step (common in steady state) |
| MoE model batch (if architecture-aware) | Return positive estimate; no division by zero on expert counts (R11) | Expert count and tokens-per-expert may be zero for some routing configurations |
| Batch with extreme KV length variance (one request at kv_len=1, another at kv_len=4096) | Return positive estimate; must not average the KV lengths in a way that loses the variance signal | Tests that the model handles heterogeneous KV lengths (Gap 1 motivation) |
| Batch with all-zero progress indices (all requests are new prefills) | Return positive estimate based on prefill tokens only | Common at simulation start or after a burst of new arrivals |
| Batch with a request having zero new tokens (progress index > 0 but no decode token this step) | Handle gracefully; this can occur during preemption/resume cycles | Edge case from continuous batching scheduler |
| Batch where total tokens exceed training distribution (e.g., 4096 prefill tokens in one request) | Return positive estimate; must not produce wildly extrapolated values | Out-of-distribution extrapolation; the model should degrade gracefully |
| Extremely large batch size (>128, if max_num_seqs is reconfigured) | Return positive estimate; prediction time still < 1ms | Future-proofing; the 128 limit is configurable |

## Multi-Instance Scope

**This research targets single-instance step-time prediction.** The rationale:

1. Step time is computed per-instance by the LatencyModel behavioral contract — it depends only on the batch composition at that instance, not on other instances' state.
2. Multi-instance effects (routing decisions, admission control, snapshot staleness) affect *which* requests reach an instance, not *how long* a step takes once the batch is formed.
3. Ground-truth data was collected from single-instance vLLM deployments.

**Routing snapshot boundary constraint:** The LatencyModel is called within the per-instance step execution path, which operates on the instance's local batch. It does not observe or depend on routing snapshots, which are cluster-level constructs with their own staleness properties (INV-7). This means the StepML model's predictions are independent of snapshot refresh intervals and routing policy choices — a desirable property that simplifies validation.

**Shared read-only state:** The per-instance LatencyModel may share read-only reference data (e.g., benchmark databases, normalization parameters) across instances via shared pointers in the config. This is safe because all shared data is immutable after construction. Any future model that maintains mutable per-instance state (e.g., online learning, adaptive calibration) must ensure separate instances receive independent copies.

**Instance heterogeneity:** The LatencyModel is constructed per-instance, so the architecture supports heterogeneity in principle. However, the current cluster deployment uses a single shared configuration for all instances; per-instance differentiation (different GPU types, TP degrees, or model configurations) would require extending the deployment config to support per-instance overrides. For this research, all training data is homogeneous (H100, single precision), so heterogeneity is deferred to future work.

**Cluster-level validation (deferred):** After the winning model is integrated into BLIS, a separate validation study should measure end-to-end cluster metric fidelity (TTFT p50/p99, throughput) to verify that per-step prediction errors don't compound across instances or interact with routing/scheduling policies.

## Decisions with Trade-offs

| ID | Decision | Status |
|----|----------|--------|
| D-1 | Train on 10%-sampled data with bias characterization | Proposed |
| D-2 | Research both unified and per-architecture models | Proposed |
| D-3 | Allow Python features with Go-feasibility tracking | Proposed |
| D-4 | Temporal split within experiments, stratified across | Proposed |
| D-5 | Research in Python, defer Go integration path selection | Proposed |
| D-6 | Treat blackbox as replaced (not demoted to fallback) | Proposed |
| D-7 | Report systematic bias (signed error) alongside MAPE | Proposed |
| D-8 | Resolve prefix cache semantics in Phase 0 infrastructure | Proposed |

### D-1: Train on 10%-sampled data vs. collect full traces

**Status:** Proposed
**Decision:** Use the existing 10%-sampled data with explicit bias characterization.
**Alternatives considered:** (a) Re-run all experiments with 100% tracing — prohibitively expensive (20 experiments × 4+ hours each); (b) Use the 10% sample without bias analysis — risks systematic error.
**Why this wins:** The 165K samples provide sufficient statistical power if the sampling is approximately random. The bias characterization (see Sampling Bias section) will detect non-random sampling.
**What breaks if wrong:** If the sampling is systematically biased (e.g., periodic, correlated with batch size), model performance on full traces may differ from validation set performance.

### D-2: Single unified model vs. per-architecture models

**Status:** Proposed
**Decision:** Research both approaches — each idea may propose either.
**Alternatives considered:** (a) Mandate unified model — may sacrifice MoE accuracy; (b) Mandate separate models — may miss transfer learning opportunities.
**Why this wins:** The research phase should explore both; the comparison leaderboard will determine which is better empirically.

### D-3: Feature engineering in Python vs. Go-compatible features only

**Status:** Proposed
**Decision:** Allow Python-only features during research; require Go-feasibility analysis per idea.
**Alternatives considered:** (a) Restrict to Go-computable features from the start — limits research exploration; (b) Allow arbitrary features without feasibility analysis — risks un-integratable winners.
**Why this wins:** Research should not be prematurely constrained, but integration feasibility must be tracked.

### D-4: Temporal vs. random data split

**Status:** Proposed
**Decision:** Use temporal ordering within each experiment to prevent temporal leakage, with stratification by model×workload across experiments.
**Alternatives considered:** (a) Pure random split — risks temporal leakage from autocorrelated consecutive steps; (b) Leave-one-experiment-out — too few experiments (16) for stable estimates.
**Why this wins:** Temporal ordering within experiments preserves the natural train→validate→test sequence while stratification ensures representation.

### D-5: Research language (Python) vs. deployment language (Go)

**Status:** Proposed
**Decision:** Conduct all research in Python; defer Go integration path selection to the production macro plan.
**Alternatives considered:** (a) Research directly in Go — limits ML library access; (b) Research in Python with Go prototype in parallel — doubles effort during exploratory phase.
**Why this wins:** Python has the best ML ecosystem (scikit-learn, XGBoost, PyTorch). The Go integration path depends on the winning model's complexity (see Python-to-Go Integration Strategy), which is unknown until research concludes.
**What breaks if wrong:** If the winning model is fundamentally difficult to port to Go (e.g., requires a custom CUDA kernel), the integration cost may be higher than expected. Mitigated by requiring each idea to document its Go integration path.

### D-6: Blackbox replacement vs. blackbox demotion

**Status:** Proposed
**Decision:** The StepML model replaces the blackbox as the default for users without hardware/model config. The blackbox remains available as a fallback (no-op default) but is no longer the recommended option.
**Alternatives considered:** (a) Keep blackbox as co-equal option — confusing for users; (b) Remove blackbox entirely — breaks backward compatibility; (c) Demote blackbox to "legacy" — same as our approach but with a deprecation timeline.
**Why this wins:** Users get a better default without losing the fallback. The roofline model is unaffected.
**What breaks if wrong:** If StepML's accuracy is model-specific (e.g., only works for Llama but not for future architectures), users may need to fall back to blackbox more often than expected.

### D-7: Report systematic bias (signed error) alongside MAPE

**Status:** Proposed
**Decision:** Require all ideas to report mean signed percentage error (MSPE) in addition to MAPE.
**Alternatives considered:** (a) Report only MAPE — simpler but hides directional bias; (b) Report full error distribution only — too much detail for leaderboard comparison.
**Why this wins:** A model with 8% MAPE but consistent -7% MSPE (systematic underestimation) would cause BLIS to underpredict step times, leading to optimistic TTFT and throughput estimates. Systematic bias is worse than random error for simulation fidelity because it doesn't cancel across steps.
**What breaks if wrong:** N/A — this is a reporting requirement, not a design decision.

### D-8: Resolve prefix cache semantics in Phase 0

**Status:** Proposed
**Decision:** During Phase 0 (shared infrastructure), characterize the `prefill_tokens` field's semantics: does it reflect pre-cache-hit or post-cache-hit token count? Also determine how prefix cache hits interact with the progress index — specifically, whether cached tokens contribute to the progress index differently than newly computed tokens.
**Alternatives considered:** (a) Defer to per-idea — each idea would need to independently investigate; (b) Assume post-cache-hit — may be wrong.
**Why this wins:** This is a shared infrastructure concern. A single investigation benefits all ideas and prevents inconsistent feature interpretation. The distinction matters because (a) new tokens only = measures actual GPU work per step, (b) total KV length including cached = measures memory bandwidth cost of attention, and (c) both as separate features = captures the compute/memory tradeoff that prefix caching introduces.
**What breaks if wrong:** If `prefill_tokens` counts pre-cache-hit tokens but the model assumes post-cache-hit, predictions will overestimate step time for cache-hit-heavy workloads.
**Phase 0 deliverable:** A characterization report documenting the exact semantics of `prefill_tokens` and `decode_tokens` in the presence of prefix caching, with evidence from the trace data (e.g., comparing `prefill_tokens` with `sum(input_tokens)` for requests known to have cache hits).

## Pipeline Architecture

Four phases, each using a specific skill:

### Phase 1: Ideation (`/research-ideas`)

Generate research ideas for step-time prediction algorithms (statistical, analytical, ML, or hybrid), iteratively reviewed by multiple LLM judges. Each idea must cite relevant prior work and meet top-tier conference rigor.

**problem.md scope:**
- Dense + MoE model step-time prediction
- KV cache dynamics as features (offloading, utilization)
- Single-instance step-time error reduction
- Literature-grounded algorithmic design with citations
- Must address evaluation dimensions: workload-level E2E mean fidelity (P1), TTFT/ITL mean fidelity (P2), tail behavior (P3), generalization (workloads, vLLM config, LLMs), ease of use, retraining story, vLLM version robustness, overhead, reproducibility, hardware generalization (P5, lower priority), quantization (P6, lowest)

**Context sources:**
- This repo (BLIS simulator, existing latency models, calibration infrastructure)
- `tektonc-data-collection/results/` (ground-truth data schema)
- `InferSim/bench_data/` (kernel benchmark data)
- Existing roofline hypotheses (`hypotheses/h-roofline/`)
- Targeted web search for academic literature (10 specific queries)

**Output:** `research.md` with 3+ ranked ideas, each with literature citations + LLM reviews.

### Phase 2: Hypothesis Selection

Extract top ideas from `research.md` and map each to a hypothesis family with 2-3 sub-hypotheses. Each idea's HYPOTHESIS.md must include the literature citations and algorithmic justification from `research.md`.

### Phase 3: Experimentation (`/hypothesis-test` adapted)

Per idea, scaffold and run:
1. **H-stepml-N.1: Feature Engineering** — which features matter?
2. **H-stepml-N.2: Model Training** — does the model achieve <10% MAPE?
3. **H-stepml-N.3: Generalization** — does it hold across configs?

### Phase 4: Comparison & Selection

Cross-idea leaderboard on held-out test set.

## Hypothesis Structure

### Directory Organization

Each research idea follows the standard BLIS hypothesis directory structure (per `docs/templates/hypothesis.md`), organized under `hypotheses/h-stepml/` (to be created during Phase 0 infrastructure setup). The structure consists of:

- **Shared infrastructure directory:** Common data loading, split computation, evaluation harness, and baseline implementations. All ideas import from here rather than reimplementing.
- **Per-idea directories:** Each idea contains three sub-hypotheses (h1-features, h2-model, h3-generalization), each following the standard hypothesis template with HYPOTHESIS.md, experiment scripts, analysis scripts, and FINDINGS.md.
- **Leaderboard:** A top-level index tracking all ideas' results across evaluation dimensions.

Exact filenames and module organization are determined during micro-planning (Phase 0 infrastructure plan), not prescribed here.

### Dependency Chain (per idea)

```
h1-features ──→ h2-model ──→ h3-generalization
```

Short-circuit rule: if h1 shows features yield per-step MAPE > 30% even with a strong baseline model (Ridge regression), skip h2 and h3 for that idea. **Justification and validation:** The naive mean baseline typically achieves ~40-60% per-step MAPE. The blackbox 2-feature model achieves ~20-25% per-step MAPE. A feature set that cannot reduce per-step MAPE below 30% with a linear model is performing worse than the trivially simple blackbox baseline — this strongly suggests the features lack causal signal rather than the model lacking capacity. The 30% threshold is validated in Phase 0 infrastructure by computing the blackbox baseline's workload-level E2E mean error and per-step MAPE on the training split. Threshold adjustment is bidirectional: if the blackbox baseline exceeds 25% per-step MAPE, the short-circuit threshold is raised to `blackbox_MAPE + 10%`; if the blackbox baseline achieves workload-level E2E mean error < 12% (already near the 10% target), the incremental value of a complex replacement diminishes — Phase 0 must report this finding and reassess whether the research is justified. The threshold is confirmed during the Phase 0 baseline computation before any idea begins h1.

## Evaluation Framework

### Primary Metrics

The primary success metric is **workload-level E2E mean error**, not per-step MAPE. For each of the 16 experiments (model × workload), the simulation produces one mean E2E latency value. That value must be within 10% of the ground-truth mean E2E.

| Metric | Target | Granularity | Source |
|--------|--------|-------------|--------|
| Workload-level E2E mean error | < 10% | One value per experiment (16 experiments) | `shared/evaluation.py` |
| Per-step MAPE (step_duration_us) | Diagnostic (no hard target) | Per-step | `shared/evaluation.py` |
| Pearson r (step-level) | > 0.95 | Per-step | `shared/evaluation.py` |

**Definition:** For experiment $e$, the workload-level E2E mean error is: `|predicted_mean_e2e(e) - observed_mean_e2e(e)| / observed_mean_e2e(e)`. The predicted mean E2E is reconstructed by summing predicted step times along each request's path and averaging across all requests in the experiment. The 10% target means this error must be < 10% for each experiment individually (not averaged across experiments).

### Diagnostic Breakdowns

| Dimension | Splits |
|-----------|--------|
| Per-model | Llama-7B, Llama-70B, Mixtral-8x7B, CodeLlama-34B |
| Per-architecture | Dense vs MoE |
| Per-workload | general, codegen, roleplay, reasoning |
| Per-phase | Prefill-heavy steps vs decode-heavy steps vs mixed batches (chunked prefill produces mixed batches nearly every step — the prefill/decode interaction is non-additive per BLIS hypothesis H5) |
| Per-load | Low QPS vs high QPS |

### Evaluation Dimensions (ordered by priority)

| Priority | Dimension | What We Measure |
|----------|-----------|----------------|
| **P1** | **Workload-level E2E mean fidelity** | For each experiment: `|predicted_mean_e2e - observed_mean_e2e| / observed_mean_e2e < 10%`. This is the primary go/no-go metric. |
| **P1** | **Per-step accuracy (diagnostic)** | Per-step MAPE and Pearson r on test set — reported as a diagnostic to understand error aggregation, but not a hard gate. No fixed per-step MAPE target. |
| **P2** | **TTFT mean fidelity** | TTFT mean prediction error (first-step prediction accuracy) |
| **P2** | **ITL mean fidelity** | ITL mean prediction error (decode step prediction accuracy) |
| **P3** | **Tail latency behavior** | p99 E2E and TTFT — no ranking inversions vs. baseline |
| **P4** | **Generalization: workloads** | MAPE variance across codegen, reasoning, roleplay, general |
| **P4** | **Generalization: vLLM config** | Sensitivity to max_num_seqs, max_num_batched_tokens changes |
| **P4** | **Generalization: LLMs** | Leave-one-model-out MAPE (dense + MoE) |
| **P4** | **Ease of use** | Number of inputs required, configuration complexity |
| **P4** | **Retraining story** | When retraining is needed, data requirements, training time |
| **P4** | **vLLM version robustness** | Which vLLM scheduler changes would invalidate the model |
| **P4** | **Overheads** | Data collection time, training time, inference latency of the model |
| **P4** | **Reproducibility** | Fixed seeds, deterministic training, documented dependencies |
| **P5** | **Generalization: hardware** | Transferability across GPU generations (H100 → A100 etc.) — training data is H100-only. This dimension measures *what would need to be re-measured/re-calibrated* for a new GPU, not actual cross-hardware training. Lower priority: report findings but do not block on cross-hardware accuracy. |
| **P6** | **Quantization transferability** | Impact of precision changes (BF16 → FP8/INT4) — lowest priority. Report BF16-only results as primary; quantization effects are informational only. |

### Baselines

Every `analyze.py` must compare against:
1. **Blackbox (being replaced):** 2-feature, 3-coefficient linear regression on cache-miss tokens (new tokens for prefill requests, i.e., `NumNewTokens` during prefill — which may be less than total prefill tokens due to chunked prefill or prefix cache hits) and decode tokens. Structurally incapable of capturing KV-length-dependent step time because it aggregates all tokens into two scalar sums — a batch with one long-context decode and one short-context decode produces identical predictions. This structural limitation is the primary motivation for replacing it. **The winning model must outperform this baseline.**
2. **Roofline (not being replaced — comparison only):** Analytical FLOPs/bandwidth model (when model config available). The roofline model remains an independent, unmodified option in BLIS. It is included here for informational comparison only. Note: the roofline model has no MoE-specific logic; for Mixtral-8x7B, roofline MAPE should be reported separately and interpreted accordingly.
3. **Naive mean:** Always predict mean `step_duration_us`

**Baseline fairness:** The blackbox baseline's beta coefficients should be re-trained on the same training split used by research models (not the defaults.yaml coefficients, which were fit on different data). This ensures a fair comparison.

### Statistical Rigor Requirements

Reporting follows the metrics priority order. Each idea must report all metrics, but evaluation weight follows P1 → P5.

- **(P1 — Workload-level E2E mean) Primary fidelity metric:** Each idea must report workload-level E2E mean error for each of the 16 experiments via synthetic trace replay (Stage 1). Target: < 10% on each experiment. This is the primary go/no-go signal. Report the count of experiments passing and failing the 10% threshold.
- **(P2 — TTFT/ITL mean) Component-level fidelity:** Report workload-level TTFT mean error and ITL mean error for each experiment. Target: < 15% on each experiment.
- **Per-step accuracy (diagnostic):** Report per-step MAPE with 95% confidence interval (bootstrap with 1000 resamples) and Pearson r. Per-step MAPE has no hard target — it is reported as a diagnostic to understand how per-step errors aggregate into workload-level means. The improvement over the best baseline must be statistically significant (Wilcoxon signed-rank test on paired per-experiment E2E mean errors, p < 0.05).
- **Systematic bias (signed error):** Report mean signed percentage error (MSPE) alongside MAPE. Per D-7, systematic bias is worse than random error for simulation fidelity. An MSPE magnitude >5% warrants investigation and explanation (e.g., the model systematically underestimates large-batch steps because the training distribution is skewed toward small batches).
- **Per-model breakdown:** Report MAPE separately for each of the 4 models. An idea that achieves <10% overall MAPE but >20% on MoE is not considered successful for MoE generalization. Per-architecture threshold: <15% MAPE for both dense and MoE independently (this is looser than the 10% overall target to allow for MoE's inherent complexity while still requiring meaningful improvement over baselines).
- **(P3 — Tail) Error distribution:** Report the distribution of per-step absolute percentage errors (histogram + percentiles: p50, p90, p99). A model with 8% MAPE but 50% p99 error is unreliable. Tail analysis is evaluated after mean metrics.
- **(P3 — Tail) Extreme error analysis (p99 interaction with simulation):** Steps with >50% prediction error can cause cascading simulation artifacts — an underestimated long step causes premature scheduling of the next batch, while an overestimated step delays completion events. Each idea must report: (a) the fraction of steps with >50% error, (b) whether those extreme errors correlate with specific batch compositions (e.g., first-step-of-prefill, maximum batch size), and (c) a qualitative assessment of whether the extreme errors are systematically biased in one direction (over vs. under). If >5% of steps have >50% error, the idea must discuss the simulation impact.
- **Monotonicity check:** Verify INV-M-5 empirically — plot predicted step time vs. total tokens for fixed batch compositions. Non-monotonic predictions indicate the model learned spurious correlations. Report the fraction of test-set pairs where adding tokens decreases predicted time.

### End-to-End Validation

Per-step MAPE does not directly indicate simulation fidelity. End-to-end validation occurs in two stages:

#### Workload-Level E2E Mean Error Definition

The P1 metric is **workload-level E2E mean error** — the error of means, computed per experiment:

```
For experiment e:
  predicted_mean_e2e(e) = mean over requests in e of (sum of predicted step times along request's path)
  observed_mean_e2e(e)  = mean over requests in e of (observed request-level E2E latency)
  E2E_mean_error(e)     = |predicted_mean_e2e(e) - observed_mean_e2e(e)| / observed_mean_e2e(e)
```

The target is `E2E_mean_error(e) < 10%` for **each** of the 16 experiments individually. This formulation benefits from cancellation — random per-step errors that are unbiased will partially cancel when averaged across requests. This is intentional: simulation users care about the mean E2E latency of a workload, not individual request accuracy. A model with zero systematic bias and moderate per-step variance will score well, which is the correct behavior for a simulation fidelity metric.

**Per-request MAPE** (mean of per-request absolute percentage errors) is reported separately as a diagnostic to detect cases where the workload-level mean is accurate but individual requests have large errors.

TTFT mean error and ITL mean error use the same workload-level formulation: `|predicted_mean_ttft(e) - observed_mean_ttft(e)| / observed_mean_ttft(e)`, target < 15%.

#### Stage 1: Lightweight Error Propagation Analysis (During Research)

Even before BLIS integration, each idea can estimate error propagation impact:

1. **Component-level error attribution (prerequisite):** Using the shared `e2e_decomposition.py` tool, measure each LatencyModel method's contribution to workload-level E2E mean error. For each of the 5 methods, replace the current prediction with ground-truth (if available) and measure the marginal improvement in E2E error. This reveals where the error budget is concentrated and which components have the highest ROI for improvement.
2. **Synthetic trace replay:** Using the per-request lifecycle data, reconstruct the sequence of steps for a complete request. Sum the predicted step times (and queueing time, output token processing time, scheduling time) along the request's path and compare to the observed request-level metrics (E2E latency, TTFT, ITL). This does not require BLIS integration — it uses Python and the ground-truth data.
3. **Acceptance criteria (ordered by metrics priority):**
   - **(P1 — E2E mean)** Workload-level E2E mean error < 10% on each experiment. This is the primary go/no-go criterion — it directly measures whether the model is accurate enough for simulation use.
   - **(P2 — TTFT/ITL mean)** Workload-level TTFT mean error < 15% and ITL mean error < 15% on each experiment. If E2E mean passes but TTFT mean fails, the model has systematic prefill-path or queueing-time errors.
   - **(P3 — Tail)** P99 E2E and P99 TTFT should not show ranking inversions across workloads compared to baselines. Tail metrics are evaluated after mean metrics pass.
4. **Cancellation analysis:** Compute the autocorrelation of per-step signed errors along each request's step sequence. If errors are positively autocorrelated (consecutive steps both over- or under-predicted), they compound. If negatively autocorrelated or uncorrelated, they partially cancel.

#### Stage 2: Full BLIS Integration Validation (Post-Research)

After the winning model is integrated into BLIS:

1. **Calibration pairs:** Use BLIS's existing calibration infrastructure to generate calibration pairs (predicted vs. observed) for TTFT, ITL, and E2E latency.
2. **Error propagation study:** Run BLIS with the winning StepML model vs. the blackbox model it replaces on the same workload traces. Compare end-to-end metric distributions (TTFT p50, p99; throughput; E2E latency) to quantify how per-step error propagates. The roofline model may also be included as an independent comparison point, but is not being replaced.
3. **Cluster-level validation:** In cluster mode, verify that per-step errors don't compound across instances or interact with routing/scheduling policies in unexpected ways.

**Integration gate (ordered by metrics priority):** The StepML model will not become the default in BLIS main branch until Stage 2 validation passes:
1. **(P1 — E2E mean)** Workload-level E2E mean error < 10% on all standard workload scenarios
2. **(P2 — TTFT/ITL mean)** Workload-level TTFT mean error < 15% and ITL mean error < 15%
3. **(P3 — Tail)** No P99 ranking inversions compared to the blackbox baseline on the standard workload suite
4. Throughput prediction error < 15%

This gate is enforced during the production integration macro plan, not during this research phase.

#### Falsification Criteria

The StepML approach should be abandoned (reverting to the blackbox model) if any of these are observed (ordered by metrics priority):

1. **(P1 — E2E mean)** **Workload-level E2E mean error > 10% on more than 25% of experiments** (4+ of 16) after all ideas have been evaluated — the model cannot achieve the primary fidelity target
2. **(P2 — TTFT/ITL)** **Workload-level TTFT mean or ITL mean error > 20%** on more than 25% of experiments — per-step errors propagate disproportionately into component-level metrics
3. **Systematic bias (|MSPE| > 10%)** that cannot be corrected with a simple bias term (the model has a fundamental directional error that compounds into E2E mean)
4. **(P3 — Tail)** **Extreme errors (>50% per-step error) on >10% of steps** (the model is unreliable for tail behavior)
5. **Non-monotonicity (INV-M-5 violations) on >5% of test pairs** (the model learned spurious correlations that would produce physically implausible simulation behavior)
6. **Request-feature insufficiency:** If the best model using only Request-derivable features (input tokens, output tokens, progress index, new tokens, plus experiment-level metadata) achieves workload-level E2E mean error > 15% on the majority of experiments despite trying multiple algorithmic approaches, the Request-only assumption is falsified and the LatencyModel interface may need extension

### Step-Level Validation Against Real System

Each idea validates its step-time predictions directly against the ground-truth step durations from instrumented vLLM. The `step.duration_us` field is the wall-clock duration of vLLM's model execution call, which is the authoritative measurement of real step time. No additional real-system benchmarking is required during the research phase because the ground-truth data already captures real execution times.

### MFU Sensitivity Analysis

The roofline baseline depends on MFU (Model FLOPs Utilization) values from `hardware_config.json`. Small changes in MFU can significantly affect roofline predictions. Each idea that uses physics-informed features (e.g., roofline-based features) must report sensitivity to MFU perturbation: ±10% MFU change → what MAPE change?

### Leaderboard Format (README.md)

Columns ordered by metrics priority (P1 → P5). Workload-level metrics are reported as "X/16 experiments passing threshold":

| Idea | Algorithm | Key Citations | E2E Mean <10% (P1) | TTFT Mean <15% (P2) | ITL Mean <15% (P2) | Per-step MAPE (diag) | Pearson r | P99 Inversions (P3) | Dense E2E Err | MoE E2E Err | p-value vs. baseline | Go Path |
|------|-----------|--------------|---------------------|---------------------|--------------------|--------------------|-----------|---------------------|---------------|-------------|---------------------|---------|

## Data Split Strategy

**One-time split, shared across all ideas:**

```python
# Within each experiment: temporal ordering (first 60% → train, next 20% → valid, last 20% → test)
# Across experiments: stratified by (model, workload_type) to ensure balanced representation
train (60%) / valid (20%) / test (20%)
```

### Temporal Leakage Prevention

Steps within a single experiment are temporally autocorrelated — consecutive steps share similar batch compositions because requests persist across steps in continuous batching. A random split would leak future batch composition information into the training set.

**Mitigation:** Within each of the 16 main experiments, steps are sorted by `step_id` (temporal order). The first 60% of each experiment's steps form the training set, the next 20% the validation set, and the final 20% the test set. This ensures the model is always evaluated on *future* steps relative to training data.

### Split Properties

- Stratification ensures each split contains data from all 16 model×workload combinations
- Temporal ordering within experiments prevents autocorrelation leakage
- Test set is NEVER seen during any idea's training or hyperparameter tuning
- The split is saved as indices (not data) so it's reproducible
- For generalization testing (h3), additional leave-one-model-out splits are created
- The 4 sweep experiments are held out entirely as an additional generalization test set

## Execution Plan

### Wave-Based Parallel Execution

```
Step 1: Build shared infrastructure
  └── data_loader.py, split.py, evaluation.py, baseline.py
  └── Parse all 20 experiments into unified dataset
  └── Compute baseline MAPE for reference
  └── e2e_decomposition.py: component-level error attribution
  │   └── Measure each LatencyModel method's contribution to E2E error
  │   └── Identify which component dominates the error budget
  └── lifecycle_kv_extractor.py: derive per-request KV lengths
      └── Extract per-request ProgressIndex at each step from request_metrics/
      └── Compute per-step kv_mean, kv_max, kv_sum from per-request data

Step 2: Run /research-ideas (sequential)
  └── problem.md with full context
  └── 3 iterations, multiple LLM judges
  └── Output: research.md with ranked ideas

Step 3: Scaffold hypotheses (parallel across ideas)
  └── Extract top 3-5 ideas
  └── Create directory structure
  └── Write HYPOTHESIS.md for each sub-hypothesis

Step 4 - Wave 1: Feature engineering (parallel across ideas)
  └── All h1-features run simultaneously
  └── Each idea extracts its unique features
  └── Short-circuit check: drop ideas with MAPE > 30%

Step 5 - Wave 2: Model training (parallel across surviving ideas)
  └── All h2-model run simultaneously
  └── Each uses its h1's best feature set
  └── Output: trained models + validation MAPE

Step 6 - Wave 3: Generalization (parallel)
  └── All h3-generalization run simultaneously
  └── Leave-one-model-out evaluation
  └── Cross-workload MAPE breakdown

Step 7: Leaderboard + selection (sequential)
  └── Compare all ideas on test set
  └── Evaluate on all dimensions (accuracy, ease of use, etc.)
  └── Select best approach, document findings
```

### Parallelism Model

- **Across ideas:** Fully parallel (independent data, independent models)
- **Within an idea:** Sequential dependency chain (h1 → h2 → h3)
- **Within a hypothesis:** Parallel configs/ablations within `run.sh`
- **Short-circuiting:** After Wave 1, drop ideas with feature MAPE > 30%

## Skill Invocation Specifications

Each skill invocation below specifies the key decision points and their rationale. The exact screen ordering may vary as skills evolve — these specifications capture the *intent* of each decision, not the exact UI flow.

---

### S1. `/research-ideas` — Phase 1: Ideation

**Purpose:** Generate 3+ research ideas for step-time prediction, iteratively reviewed by external LLM judges.

**Pre-requisite:** Write `problem.md` (content derived from this design doc — see problem.md Content section above) before invoking the skill.

#### Key Decision: Problem Source

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Problem definition source | Use pre-written `problem.md` | Content derived from this design doc's canonical sections |

#### problem.md Content

The `problem.md` file is written before skill invocation with content derived from this design document. It is a **reference template** — the authoritative content lives in the sections above (Problem Statement, Ground-Truth Data, Evaluation Framework, Baselines). The problem.md content should be regenerated from those sections at invocation time rather than maintained as a separate copy.

**Key content requirements for problem.md:**
- Problem statement: step-time prediction for LLM inference serving simulation
- Scope clarification: replacing the blackbox model only; roofline is NOT being replaced
- Available data: ~165K step-level observations, 4 models × 4 workloads, H100 GPUs, vLLM v0.15.1
- Algorithm scope: statistical, analytical, ML, or hybrid — not restricted to ML
- 10 evaluation dimensions (from the Evaluation Framework section above)
- Constraints: <1ms inference, features observable at step time, 10% sampling
- Baselines: blackbox (being replaced — must outperform), roofline (comparison only — NOT being replaced), naive mean
- Strong idea criteria: algorithmic design, feature representation, fitting methodology, generalization mechanism, related work positioning
- All 10 evaluation dimensions must be addressed per idea

**Data sources referenced:** This repository (BLIS latency models), the data collection results directory (ground-truth schema), kernel benchmark data, existing roofline hypotheses.

**Note:** The exact problem.md text is generated at invocation time from the canonical sections in this design document. This avoids maintaining a 150-line duplicate that could become stale.

#### Key Decision: Fresh Start vs. Existing Research

| Decision | Choice | Rationale |
|----------|--------|-----------|
| If existing research.md found | Start fresh | Clean start for this research campaign |

#### Key Decision: Background Sources

| Source | Selected | What It Provides |
|--------|----------|------------------|
| Current repository | Yes | BLIS architecture, existing latency models, MFU database, calibration infrastructure, roofline hypotheses |
| Other local repositories | Yes — data collection repo + InferSim | Ground-truth data schema, kernel benchmark data |
| Local documents | Yes — roofline design doc | Existing roofline model documentation |
| Web search (specific queries) | Yes — 10 targeted queries | Academic literature for conference-quality prior work |
| GitHub repositories | No | Local clone is more efficient than API scanning |
| Remote papers/URLs | No | Web search with specific queries is more effective |
| Skip background | No | Literature is essential for conference-quality ideas |

**Web search queries (12 targeted queries for academic literature):**
LLM inference serving latency prediction, Vidur step time estimation, roofline GPU kernel prediction, vLLM batch iteration modeling, DES inference serving, GPU kernel ML prediction, MoE inference cost model, splitwise/DistServe/Sarathi latency, SimLLM/LLMServingSim performance model, Bayesian GPU performance modeling, OpenEvolve evolutionary algorithm discovery LLM, GEPA genetic Pareto optimization code.

**Rationale for manual queries:** Targets the exact problem domain, known competing simulators, specific algorithmic families, and MoE challenges.

#### Key Decision: Review Configuration

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Number of LLM judges | All 3 available (Claude, GPT-4o, Gemini Flash) | Maximum review diversity |
| Number of iterations | 3 (recommended) | 3 ideas with iterative refinement; diminishing returns beyond 3 |

#### Expected Output

`research.md` containing:
- Problem statement + background context
- 3 research ideas, each with:
  - Feature engineering approach
  - Model architecture
  - Training methodology
  - Generalization strategy
  - Reviews from 3 LLM judges
- Executive summary ranking the ideas

---

### S2. `/hypothesis-test` — Phase 3: Experimentation

**Purpose:** Scaffold and run experiments for each research idea. Invoked once per idea (not once for all ideas).

**Pre-requisite:** Shared infrastructure (`hypotheses/h-stepml/shared/`) must be built first. Research ideas must be extracted from `research.md` and mapped to hypothesis directories.

**Adaptation:** The skill generates hypotheses for the whole project by default. We override this by providing pre-written `HYPOTHESIS.md` files and using the "pending" detection mechanism — the skill detects existing pending hypotheses and offers to test them.

#### Pre-invocation Setup (per idea)

Before invoking `/hypothesis-test` for Idea N, manually create:

```
hypotheses/h-stepml/idea-N-<name>/h1-features/HYPOTHESIS.md   # Status: Pending
hypotheses/h-stepml/idea-N-<name>/h2-model/HYPOTHESIS.md      # Status: Pending
hypotheses/h-stepml/idea-N-<name>/h3-generalization/HYPOTHESIS.md  # Status: Pending
```

Each `HYPOTHESIS.md` follows the project template with content derived from the research idea. It MUST include a Related Work section with citations from `research.md`. Example for h1-features:

```markdown
# Hypothesis: [Idea N] Feature Engineering

**Status:** Pending
**Family:** Performance-regime
**Type:** Type 2 (Statistical — Dominance)

## Claim

[Extracted from research.md idea N]: Feature set X (e.g., "physics-informed features
including compute intensity ratio, memory boundedness index, and prefill/decode token
interaction terms") achieves lower validation MAPE than raw batch features alone when
used with a baseline Ridge regression model.

## Refuted If

The engineered feature set achieves validation MAPE within 2 percentage points of
raw features (no statistically significant improvement at p < 0.05).

## Related Work

[Extracted from research.md idea N — required citations]:
- [Author et al., Venue Year]: How this work informs our feature design
- [Author et al., Venue Year]: What features they used and why ours differ
- ...

## Algorithmic Justification

[First-principles reasoning for why this feature set should work for step-time
prediction — e.g., "Compute intensity ratio captures the transition between
compute-bound and memory-bound regimes that determines which hardware resource
is the bottleneck for a given batch composition" (cf. Williams et al., roofline
model, 2009)]
```

#### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Project scope | Current directory, scoped to the specific idea directory | Prevents generating unrelated hypotheses |
| Hypothesis count | 3 per idea (pre-written) | h1-features → h2-model → h3-generalization dependency chain |
| Hypothesis selection | Select only our 3 pending HYPOTHESIS.md files; ignore any auto-generated ones | We control the hypothesis content via pre-written files |
| Execution mode | Sequential (within an idea) | h2 depends on h1's output; cross-idea parallelism handled by wave dispatch |
| Experiment approval | Review scaffolded code, then run all | Verify shared infrastructure usage before running |
| Commit | Commit all (including failed experiments) | Failed experiments provide diagnostic information |

**Verification checklist (before approving scaffolded experiments):**
- `run.sh` sources shared data loader, not custom parsing
- `run.sh` uses the shared train/valid/test split (not its own)
- `analyze.py` imports shared evaluation harness for MAPE computation
- `analyze.py` computes baseline comparisons (blackbox [must outperform], roofline [informational comparison], naive mean)
- `FINDINGS.md` template includes all evaluation dimensions + systematic bias (MSPE) + p99 error analysis
- No hardcoded absolute paths; uses relative paths from hypothesis directory

---

### S3. `convergence-review` — Review Gates

**Purpose:** Quality gates at three points per idea. Each gate ensures experiment rigor before proceeding.

#### Gate 1: Hypothesis Design Review (after writing HYPOTHESIS.md, before scaffolding)

```
/convergence-review h-design --model sonnet
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gate type | `h-design` | Hypothesis design review (5 perspectives) |
| Artifact path | (none — uses conversation context) | h-design reads from current conversation |
| Model | `sonnet` | Good balance of rigor and cost for design review. 5 perspectives × sonnet is reasonable. |

**Convergence criteria:** Zero CRITICAL, zero IMPORTANT findings across 5 perspectives.

**Perspectives evaluated:**
1. Hypothesis quality (testable, falsifiable, specific)
2. ED-1 through ED-6 rigor (controlled comparison, rate awareness, etc.)
3. Parameter calibration (realistic values from ground-truth data)
4. Control completeness (all confounds addressed)
5. DES fit (applicable to discrete-event simulation context)

#### Gate 2: Experiment Code Review (after scaffolding, before running)

```
/convergence-review h-code hypotheses/h-stepml/idea-N-<name>/h1-features --model sonnet
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gate type | `h-code` | Hypothesis code review (5 perspectives) |
| Artifact path | Path to the hypothesis directory being reviewed | Points to `run.sh` + `analyze.py` |
| Model | `sonnet` | Catches implementation bugs without opus cost |

**Convergence criteria:** Zero CRITICAL, zero IMPORTANT findings across 5 perspectives.

**Perspectives evaluated:**
1. Parser-output agreement (analyze.py correctly parses run.sh output)
2. CLI flags and commands (correct Python invocations, paths)
3. YAML/config fields (correct data loading, feature names)
4. Config diff (matches experiment design)
5. Seed/determinism (reproducible results)

#### Gate 3: Findings Review (after running, before leaderboard inclusion)

```
/convergence-review h-findings hypotheses/h-stepml/idea-N-<name>/h2-model/FINDINGS.md --model opus
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Gate type | `h-findings` | Hypothesis FINDINGS review (10 perspectives) |
| Artifact path | Path to the specific FINDINGS.md | The most important review — validates conclusions |
| Model | `opus` | 10 perspectives on findings quality warrants opus-level rigor. This is the gate that determines if an idea's results are trustworthy. |

**Convergence criteria:** Zero CRITICAL, zero IMPORTANT findings across 10 perspectives.

**Perspectives evaluated:**
1. Code verifier (results match what the code actually produces)
2. Experiment designer (design is sound)
3. Statistical rigor (MAPE computation, significance tests)
4. Control auditor (baselines compared correctly)
5. Standards compliance (BLIS experiment standards)
6. Substance/logic (conclusions follow from evidence)
7. DES mechanism (step-time modeling is physically meaningful)
8. Reproducibility (can be re-run independently)
9. Cross-experiment consistency (findings consistent with other ideas)
10. User guidance (actionable recommendations)

---

### S4. `dispatching-parallel-agents` — Wave Execution

**Purpose:** Run multiple ideas in parallel during each wave.

**Note:** This skill is a methodology guide, not an interactive workflow. It provides no screens or questions. The principled input is the task decomposition.

#### Wave 1 Dispatch: Feature Engineering

**Independence check:** All h1-features experiments are independent because:
- Each reads the same shared dataset (read-only)
- Each writes to its own `idea-N-<name>/h1-features/` directory (no shared write state)
- No idea's feature engineering depends on another idea's features

**Task definitions for parallel agents:**

| Agent | Directory | Task |
|-------|-----------|------|
| Agent 1 | `hypotheses/h-stepml/idea-1-<name>/h1-features/` | Run `./run.sh`, then `python3 analyze.py`. Report MAPE results. |
| Agent 2 | `hypotheses/h-stepml/idea-2-<name>/h1-features/` | Run `./run.sh`, then `python3 analyze.py`. Report MAPE results. |
| Agent 3 | `hypotheses/h-stepml/idea-3-<name>/h1-features/` | Run `./run.sh`, then `python3 analyze.py`. Report MAPE results. |

**Post-wave integration:** After all agents complete, the orchestrator:
1. Collects each agent's MAPE results
2. Applies short-circuit rule: drop ideas with MAPE > 30%
3. Updates leaderboard README.md
4. Proceeds to Wave 2 with surviving ideas

#### Wave 2 Dispatch: Model Training

Same pattern as Wave 1, but for `h2-model/` directories. Only surviving ideas participate.

#### Wave 3 Dispatch: Generalization

Same pattern, for `h3-generalization/` directories. Only ideas that passed Wave 2 (<10% MAPE on validation) participate.

---

### S5. `writing-plans` — Implementation Planning

**Purpose:** Create the detailed implementation plan for the shared infrastructure (Phase 0) before any skill is invoked.

**Invocation:** After this design doc is approved, invoke `/writing-plans` with this design doc as context.

**The skill asks one question after plan creation:**

| Question | Answer | Rationale |
|----------|--------|-----------|
| "Subagent-Driven or Parallel Session?" | **"Subagent-Driven (this session)"** | Shared infrastructure is a sequential dependency chain (data_loader → split → evaluation → baseline). Subagent-driven allows review between tasks and fast iteration in the current session. |

**Plan scope (what writing-plans should plan):**

The plan covers only the shared infrastructure (Step 1 of the execution plan). Subsequent steps (research-ideas, hypothesis-test, etc.) are orchestrated by this design doc's skill specifications, not by the implementation plan.

Tasks for the plan:
1. Parse all 20 experiments' step-level data → unified Parquet dataset
2. Verify `step.duration_us` semantics: confirm it equals `ts_end_ns - ts_start_ns` and characterize what it includes/excludes (D-8: resolve prefix cache semantics in this task too)
3. Characterize sampling distribution: check for periodic bias, measurement overhead, phase representation (see Sampling Bias Characterization)
4. Implement stratified train/valid/test split (60/20/20 by model×workload, temporal ordering within experiments)
5. Implement MAPE + Pearson r + MSPE + p99 error evaluation harness
6. Implement blackbox (being replaced) + roofline (informational comparison) + naive mean baselines
7. Verify baselines produce expected MAPE ranges; calibrate short-circuit threshold (if blackbox MAPE > 25%, adjust threshold to blackbox_MAPE + 10%)
8. Validate the temporal split: compare random vs. temporal split MAPE to confirm leakage prevention is effective
9. **E2E latency decomposition tool** (`e2e_decomposition.py`): Given ground-truth per-request lifecycle data and current LatencyModel predictions, compute the error contribution of each of the 5 LatencyModel methods to workload-level E2E mean error. This identifies which component dominates the error budget and should be prioritized. Implementation: reconstruct per-request E2E by summing predicted delays along the request path, then ablate each component (replace with ground-truth value) to measure its marginal error contribution.
10. **Per-request KV length extractor** (`lifecycle_kv_extractor.py`): Derive per-request KV cache lengths at each step from the per-request lifecycle data in `request_metrics/`. Output: per-step features (`kv_mean`, `kv_max`, `kv_sum`, `kv_std`) joined to the step-level dataset. This unlocks the primary feature gap identified in Round 1 and is prerequisite for any idea using per-request KV features.

---

### Skill Invocation Summary

| Phase | Skill | Invocations | Total Agent Cost |
|-------|-------|-------------|-----------------|
| 0. Infrastructure | `/writing-plans` | 1× | ~5 tasks × 1 agent |
| 1. Ideation | `/research-ideas` | 1× | 3 iterations × 3 judges = 9 review agents + 3 background agents |
| 2. Selection | Manual | 1× | 0 agents |
| 3a. Design review | `/convergence-review h-design` | 1× per idea (3-5×) | 5 perspectives × 3-5 ideas = 15-25 agents (sonnet) |
| 3b. Scaffolding | `/hypothesis-test` | 1× per idea (3-5×) | 3 hypotheses × 3-5 ideas = 9-15 scaffold agents |
| 3c. Code review | `/convergence-review h-code` | 3× per idea (9-15×) | 5 perspectives × 9-15 = 45-75 agents (sonnet) |
| 3d. Execution | `dispatching-parallel-agents` | 3 waves | 3-5 agents per wave × 3 waves = 9-15 agents |
| 3e. Findings review | `/convergence-review h-findings` | Up to 3× per idea | 10 perspectives × up to 15 = up to 150 agents (opus) |
| 4. Leaderboard | Manual | 1× | 0 agents |

## Validation Strategy

### Verification (Correctness)

| What | How |
|------|-----|
| Model predictions are positive for all valid inputs | Unit tests with adversarial batch compositions (INV-M-1) |
| Predictions are deterministic | Same-input-same-output test (INV-M-2) |
| Predictions are side-effect-free | Before/after Request state comparison (INV-M-3) |
| Prediction latency < 1ms | Benchmark test at max batch size with p99 measurement (INV-M-4) |
| Predictions are monotonic in total tokens | Incrementally larger batches test (INV-M-5) |
| Data loading preserves all records | Row count validation against source files |
| Train/valid/test split is temporally ordered | Assert all training step_ids < all validation step_ids within each experiment |
| Temporal split prevents leakage | Train a model on random split vs. temporal split; if random split MAPE is significantly lower (<2% difference), temporal leakage is minimal and the split is effective. If random is much lower (>5% difference), the temporal split is correctly preventing leakage. |
| Feature derivation from Request batch is correct | For each derived feature (e.g., mean KV length from progress indices), compare the Python research implementation's feature values against the values that would be computed from the Request fields. Document the exact mapping. |

### Validation (Fidelity)

| What | How |
|------|-----|
| Step-time prediction accuracy | MAPE + Pearson r on held-out test set (per-model breakdown) |
| Improvement over baselines is real | Paired statistical test (p < 0.05) against best baseline |
| Model doesn't overfit to sampling artifacts | Compare MAPE on 10%-sampled vs. full traces (if feasible) |
| Error doesn't compound in simulation | End-to-end validation with BLIS calibration infrastructure (deferred to post-research integration) |

### Reproducibility Contract

- All random seeds fixed and documented
- All Python dependencies pinned with exact versions
- Training data split saved as reproducible indices
- Trained model artifacts saved alongside experiment results
- Every `run.sh` is executable from a clean environment with documented setup steps

## DES Design Review Checklist

Per Section 2.6 of the Design Guidelines:

| Question | Answer |
|----------|--------|
| What analysis questions does this design help answer? | AQ-1 through AQ-6 (see Analysis Questions section), with AQ-1 (E2E mean fidelity) as the primary question |
| What is modeled, simplified, and deliberately omitted? | See Modeling Decisions table (with Banks et al. evaluation) |
| What events are introduced or modified? | None — step-time estimation is a synchronous query, not an event. No new events are introduced. |
| How do new events interact with existing tie-breaking rules? | N/A — no new events |
| What new state is introduced? Who owns it? | Trained model weights (immutable, owned by StepML LatencyModel instance). No mutable simulation state introduced. |
| What new metrics are derived? | Per-step prediction accuracy (MAPE, MSPE) — research-time only, not part of simulation output. |
| How will correctness be verified? | Invariants INV-M-1 through INV-M-5 (see Invariant Analysis). Unit tests. INV-6 (determinism) verified by INV-M-2: same batch → same prediction across runs (no random sampling at inference time). |
| How will fidelity be validated? | MAPE on held-out test set + lightweight error propagation analysis (Stage 1, during research) + full BLIS calibration (Stage 2, deferred). See Validation Strategy. |
| Does this introduce new randomness? | No — prediction is deterministic (supports INV-6). Training uses randomness but that's offline, not in the DES loop. All random seeds in training are fixed for reproducibility. |
| What is the simplest version that answers the same questions? | A re-trained blackbox model with more features (e.g., adding batch_size and mean_kv_len as 4th/5th coefficients). If this achieves <10% workload-level E2E mean error, more complex approaches are unnecessary. This is tested as the feature engineering baseline in h1-features. The roofline model is unaffected regardless of outcome. |
| How does this interact with KV cache offloading? | The StepML model predicts GPU-side step time. KV cache offload/reload latency is handled by the separate KV cache subsystem (transfer latency consumed via the KVStore interface). The StepML model does not predict transfer latency — it only predicts the model execution time for a formed batch. If offloading reduces effective GPU KV usage, the batch composition features capture this indirectly (fewer decode tokens if requests are offloaded). |
| Can step-time estimation cause livelock? | No. The step-time estimation is a pure query that returns immediately. It does not modify any queue or scheduling state. Even if the prediction is wildly incorrect (e.g., returns 1μs for a step that should take 10ms), the DES event loop will simply schedule the next event sooner — this causes incorrect simulation timing but not livelock, because the work-conserving invariant (INV-8) and request lifecycle (INV-2) are enforced by the scheduler, not the latency model. |

## Success Criteria

Ordered by priority (highest first):

### Priority 1: Workload-Level E2E Mean Fidelity
1. **Workload-level E2E mean error < 10%** on each of the 16 experiments individually — this is the primary go/no-go metric. For each experiment (model × workload), the predicted mean E2E latency must be within 10% of the ground-truth mean E2E latency.
2. The improvement over the best baseline is **statistically significant** (p < 0.05, paired test across experiments)
3. Per-step MAPE is reported as a **diagnostic metric** (no hard target, but informs understanding of how per-step errors aggregate)

### Priority 2: Per-Metric Accuracy (TTFT and ITL mean)
4. **Workload-level TTFT mean error < 15%** on each experiment — TTFT mean is reconstructed from first-step predictions
5. **Workload-level ITL mean error < 15%** on each experiment — ITL mean is reconstructed from decode step predictions
6. The winning model **outperforms the blackbox baseline** it replaces on all three workload-level mean metrics: E2E, TTFT, and ITL (roofline comparison is informational — roofline is not being replaced)

### Priority 3: Tail Latency Behavior
7. **Tail latencies** (p99 E2E, p99 TTFT) do not exhibit ranking inversions compared to the blackbox baseline — i.e., the StepML model should not produce a worse ordering of workload scenarios at the tail
8. Extreme errors (>50% per-step error) on <10% of steps

### Priority 4: Model Quality and Practicality
9. The winning model generalizes across **dense and MoE** architectures (workload-level E2E mean error < 15% on all MoE experiments)
10. The winning model's required features are **derivable from the Request batch** available at step-time estimation call time (LatencyModel behavioral contract compatibility demonstrated)
11. The retraining story is **documented and practical** (data requirements, training time, minimum calibration data)
12. Results are **reproducible** (fixed seeds, saved models, documented dependencies, reproducibility contract satisfied)
13. Prediction latency is **< 1ms** per step at maximum batch size (128 requests)
14. A **Go integration path** is identified for the winning model (coefficient export, ONNX, or reimplementation)

### Priority 5: Generalization (Lower Priority)
15. **Hardware generalization** is an evaluation dimension, not a success gate — report what would need recalibration for non-H100 GPUs, but do not block on cross-hardware accuracy
16. **Quantization generalization** is the lowest priority — report BF16-only results as the primary finding; FP8/INT4 transferability is informational only and does not affect the success/failure determination
