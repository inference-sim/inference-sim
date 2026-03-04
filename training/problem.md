# Problem Statement: Learned Latency Model for BLIS

**Species:** Specification
**Extension type:** Backend Swap (Phase B — interface already frozen)
**Strategy Evolution Phase 1 — Problem Framing**

_Date: 2026-03-01_
_Branch: training_
_Status: **Converged** (Round 5: 0 CRITICAL, 0 IMPORTANT across 8 perspectives) — awaiting human approval_

---

## 1. Goal and Analysis Questions

### 1a. Goal

Evolve a latency model for BLIS that accurately predicts per-request and aggregate latency metrics by learning from real vLLM serving traces. The model must generalize across model architectures, workload profiles, and load levels — not merely fit the training data.

The learned model is a **Backend Swap** (design guidelines Section 5.4, Phase B). Phase A (interface extraction) is already complete — the `LatencyModel` interface and `NewLatencyModel` factory exist with two implementations (Blackbox, Roofline). The learned model is a third backend, registered alongside existing ones. Existing backends are preserved; the learned model is activated only by explicit configuration. The existing `LatencyModel` interface is testable with mocks (it is already mocked in test suites).

### 1b. Analysis Questions

These are the user-facing questions this model enables BLIS to answer:

1. **"At what arrival rate does TTFT p99 exceed X ms for model Y on H100?"** — Capacity planning: sweeping arrival rate with the learned model should produce a realistic saturation curve, including the overload cliff.
2. **"What is the throughput ceiling for a new model given only its config.json?"** — Model selection: a user provides a HuggingFace config.json and gets a capacity estimate without dedicated profiling. **Caveat for MoE:** with N=1 MoE architecture in training, MoE-specific features are descriptive (they fit Mixtral) not predictive (they may not generalize to other MoE models like Mixtral-8x22B or DeepSeek-V2). Q2 for new MoE models should be treated with appropriate skepticism.
3. **"How much does switching from TP=2 to TP=4 improve latency for a 70B model?"** — TP sizing: the model should capture tensor-parallelism effects on step time.
4. **"What workload shape (input/output distribution) maximizes throughput for a given model?"** — Workload optimization: the model should capture how prefill vs decode token mix affects step time.
5. **"Will this workload cause preemptions and at what rate?"** — Operational planning: the model must produce step times that, when composed with BLIS's scheduler, generate realistic queue dynamics.

Each success criterion in Section 5 traces to at least one analysis question. Q1 and Q5 require aggregate-level fidelity. Q2 requires cross-model generalization. Q3 requires architecture-parameterized step time. Q4 requires accurate prefill/decode decomposition.

### 1c. Simplest Version

The simplest model that partially answers these questions is **per-model OLS regression** using the existing `beta0 + beta1*prefillTokens + beta2*decodeTokens` form, fitted on training data. This establishes whether the existing functional form is the bottleneck or the coefficient values are. If per-model OLS meets the success criteria, no structural improvement is needed — Iteration 0 tests this explicitly.

A single global linear regression (one beta set across all models) does NOT answer Q3 (TP sizing) because it conflates architecture effects — this defines the lower bound of useful complexity.

The complexity ladder: per-model OLS → +batch-size term → separate prefill/decode models → architecture-parameterized regression → hybrid roofline+learned. The Strategy Evolution iteration loop explores this ladder: we stop when marginal improvement is insufficient, not when the model is maximally complex.

## 2. Modeling Decisions

### 2a. Modeled / Simplified / Omitted

| Component | Decision | Status | What real-system behavior is captured or lost |
|-----------|----------|--------|-----------------------------------------------|
| **Batch token composition → step time** | Modeled | Active | Prefill tokens and decode tokens affect GPU kernel execution time differently (compute-bound vs memory-bound). |
| **Batch request count → step time** | Simplified (Iter 0) / Modeled (Iter 1+) | Proposed | At Iter 0, batch size effect is conflated with token counts. **Lost:** per-request memory access overhead that scales with request count independent of total tokens. Approach B adds a `batchSize` term. Training data batch range: 1-128 requests, 1-2048 tokens. Predictions outside this envelope are unvalidated extrapolation. |
| **Architecture features → step time** | Modeled | Active | Layer count, hidden dim, head geometry, FFN dim, MoE expert routing, TP degree. Enables cross-model generalization. |
| **Chunked prefill** | Modeled | Active | Mixed prefill+decode batches produce different step times than pure-decode. BLIS handles chunked prefill in batch formation; the latency model must produce accurate step times for mixed batches. |
| **Prefix cache hit/miss** | Modeled | Active | Cache-miss tokens (not total input tokens) drive prefill compute cost. The interface provides cache-miss count via request progress tracking. |
| **MoE sparse activation** | Modeled | Active | Mixtral activates 2 of 8 experts per token. Features: expert count, active experts per token. Without these, Mixtral compute overestimated ~4x. **N=1 caveat:** with only 1 MoE architecture in training, MoE features are descriptive (fit Mixtral) not predictive (generalize to other MoE). See Section 2b. |
| **Queueing overhead** | Simplified | Active | Per-request function of input token count. **Lost:** load-dependent queueing variation. The interface takes only a single request with no system state. BLIS's event queue models actual wait-for-scheduling separately; this captures only non-blocking preprocessing overhead. **What breaks if wrong:** if the minimum-interval extraction (Section 7a) misidentifies preprocessing overhead, BLIS double-counts delay. See Section 7a for extraction procedure. |
| **Output token processing** | Simplified | Active | Constant per model. Real variation is negligible. |
| **Inter-step scheduling overhead** | Modeled | Proposed (Iter 4) | **H30 (#480) proved this is the dominant error source.** Both crossmodel and per-model blackbox show identical -17% to -56% TTFT underprediction — attributable to BLIS scheduling `now + stepTime` with no gap (`simulator.go:427`). Real vLLM incurs 1-7ms/step: scheduler.schedule() CPU (0.5-2ms), prepare_model_input() (1-5ms), CUDA graph asymmetry, Python/GIL overhead. Over ~150 steps/request, the accumulated gap is 150-1050ms — consistent with observed TTFT underprediction. Modeled as `δ₀ + δ₁·batchSize` via `SchedulingProcessingTime()`, extracted from step-to-step timestamp gaps in BATCH_SUMMARY. See Section 7a for extraction procedure and Section 11f for the two-phase learning strategy. |
| **Preemption overhead** | Simplified | Active | Returns 0. Real overhead involves KV block deallocation + re-scheduling. **Sensitivity at 85% failure rate:** reasoning test-set experiments have high preemption rates. If each preemption costs ~100μs and there are ~4000 preemptions across an experiment, total overhead is ~0.4s — small relative to experiment duration (~1200s) but could cause localized latency spikes. If test-set experiments show systematic preemption-correlated step-time bias, this becomes a candidate for learning. |
| **GPU memory bandwidth contention** | Simplified | Active | Learned model captures effective bandwidth implicitly through data-derived coefficients. **Lost:** non-linear memory contention at extreme batch sizes. **Training envelope:** batches of 1-128 requests / 1-2048 tokens. Predictions outside this envelope are unvalidated extrapolation. |
| **Training data sampling** | Simplified | Active | Step-level training data is 10% probabilistic sample of real steps. **Lost if sampling is phase-correlated:** certain batch compositions may be under/over-represented, biasing coefficients. **Required validation:** compare sampled (prefill_tokens, decode_tokens, batch_size) distribution against the full step population estimated from journey trace step counts. |
| **Hardware generalization** | Omitted | Active | All data is H100 SXM. **Lost:** ability to predict on A100, L40, or other GPUs without retraining. |
| **Multi-instance routing effects** | Omitted | Active | All experiments are single-instance. **Note:** systematic step-time bias propagates through routing in cluster mode. H29 (#433): stale signals cause +242-548% TTFT p99 degradation. H7 (#401): horizontal scaling is super-linear — queue growth excess `lambda/k - mu` is highly sensitive to `mu`, so small step-time bias has amplified effects. Single-instance evaluation is necessary but not sufficient — a cluster-mode sanity check should follow (Section 11e). |
| **vLLM version / kernel variation** | Omitted | Active | All data is v0.15.1 with FLASH_ATTN. |
| **Arrival process** | Omitted | Active | We replay real arrival times. |

### 2b. MoE Feature Treatment

Mixtral-8x7B is the only MoE model in the dataset (1 of 4 architectures). The learned model includes MoE-specific features:

- `num_local_experts` (8 for Mixtral, 0 for dense models)
- `num_experts_per_tok` (2 for Mixtral, 0 for dense models)

The derived boolean `is_moe` is redundant when both continuous features are present and is omitted.

The effective compute per forward pass scales as `num_experts_per_tok / num_local_experts` relative to a dense model with the same intermediate dimension. **With N=1 MoE architecture, cross-validation fold 3 (validate on Mixtral) tests whether the model predicts Mixtral, but cannot distinguish "learned MoE physics" from "memorized Mixtral's offset."** The MoE features are descriptive for this Mixtral, not predictive for arbitrary MoE models. Q2 (config.json-only capacity estimate) should carry a warning for new MoE architectures.

## 3. Module Contract

**Module:** Learned Latency Model (Backend Swap of existing Latency Model)

| Aspect | Specification |
|--------|---------------|
| **Observes** | Batch composition (per-request cache-miss token counts, input/output lengths, progress state), model architecture (layer count, hidden dim, head geometry, FFN dim, MoE config, data type), server config (TP degree), hardware constants (GPU memory bandwidth, compute throughput, memory capacity). Does NOT observe KV cache state, queue depth, or routing state — these are system-level concerns outside the latency model. |
| **Controls** | Three actively learned estimates: batch step duration, request arrival-to-queue preprocessing delay, per-token output processing time. Two constant-zero estimates (may be learned in future iterations): scheduling overhead, preemption overhead. All in microsecond ticks. |
| **Owns** | Learned coefficients (immutable after construction) and architecture feature values (derived from config.json at construction time). No mutable state. No caches. |
| **Invariants** | INV-L1 (Non-negativity): all methods return ≥ 0; StepTime returns ≥ 1 for non-empty batches. INV-L2 (Per-dimension monotonicity): StepTime non-decreasing when prefill tokens increase (decode held constant) and non-decreasing when decode tokens increase (prefill held constant). INV-L3 (Finite outputs): no NaN, Inf, or overflow for any valid input. INV-L4 (Backward compatibility): all existing invariant tests INV-1 through INV-8 pass. |
| **Events** | None produced or consumed. Pure query module called synchronously during step execution. |
| **Extension friction** | Steady-state (adding another backend after this one): 2 files (implementation + factory branch), matching Section 4.5 reference target. First-time prerequisite cost: additional files for MoE config extension + architecture config population change + backend selection mechanism (see Section 3b). Prerequisites are one-time; subsequent backends cost 2 files. |

**Valid input definition** (for INV-L1/L3): A valid StepTime input is a non-nil slice of request pointers where each request has non-negative progress index, positive input token count, and non-negative new-token count. A valid QueueingTime input is a non-nil request pointer with positive input token count. Empty batch (`len(batch) == 0`): StepTime returns 0 (no work to do — consistent with existing blackbox behavior of returning intercept only, but clamped to 0 since no GPU step occurs).

**Key behavioral contracts** (GIVEN/WHEN/THEN):

- GIVEN a non-empty batch of valid requests, WHEN StepTime is called, THEN the result is >= 1 (INV-L1).
- GIVEN an empty batch, WHEN StepTime is called, THEN the result is 0 (no GPU work).
- GIVEN two batches identical except batch A has one more prefill token, WHEN StepTime is called on each, THEN result(A) >= result(B) (INV-L2, prefill dimension).
- GIVEN two batches identical except batch A has one more decode token, WHEN StepTime is called on each, THEN result(A) >= result(B) (INV-L2, decode dimension).
- GIVEN any valid input, WHEN any of the five methods is called, THEN the result is a finite non-negative integer — no NaN, Inf, or overflow (INV-L3).
- GIVEN the same batch and same model configuration, WHEN StepTime is called twice, THEN both results are identical (INV-6/INV-L3).

**Invariant notes:**
- **INV-1 (Request conservation), INV-2 (Request lifecycle):** Preserved by construction — the latency model does not inject, drop, or transition requests. Conservation and lifecycle are maintained by the simulator's event loop and batch formation, which are unchanged.
- **INV-L2 enforcement:** For linear/polynomial forms, enforced by constraining coefficients to be non-negative during training. For non-linear forms (if explored in later iterations), enforced by output-level monotonicity validation or constrained optimization. **Iteration 0 must include a data analysis step** that plots real step_duration vs total tokens and reports the monotonicity violation rate. **Threshold justification:** 5% is chosen as "clearly pathological" — if more than 1-in-20 real steps violate monotonicity, the linear assumption underlying INV-L2 is wrong and the invariant must be weakened. This is conservative: typical measurement noise in GPU step timing is <2%, so >5% violations indicate systematic non-monotonicity (e.g., cache effects at batch-size boundaries), not noise. The threshold may be adjusted after Iteration 0 data analysis establishes the actual violation rate. **If the 5% threshold is triggered,** the options are: (a) weaken to "monotonic in expectation" — enforce via training loss penalty rather than point-wise coefficient constraints; (b) restrict monotonicity to within-training-distribution inputs only, with extrapolation flagged as unvalidated; (c) drop INV-L2 entirely and add an extrapolation warning when batch composition deviates significantly from training. The choice depends on the data: if violations are concentrated at specific batch-size boundaries, option (b) is appropriate; if violations are uniformly distributed, option (a) or (c).

### 3a. Failure Modes

| Condition | Behavior | Mitigation | What breaks if frequent |
|-----------|----------|------------|------------------------|
| config.json missing required fields | `NewLatencyModel` returns error | Validate at construction with fallback field-name resolution | N/A — construction-time error |
| Coefficients produce negative step time | Clamp to `max(1, predicted)` | INV-L1 enforced at output; log warning | If clamping activates on >5% of steps, model is in extrapolation territory — all metrics unreliable. Flag as model failure. |
| Batch outside training distribution (>128 seqs or >2048 tokens) | Graceful extrapolation | Log warning when any feature exceeds training range max | Linear forms extrapolate monotonically. Non-linear forms may not — validate. |
| MoE fields absent for dense model | Treat as dense (experts=0, topk=0) | Expert routing term zeroes out | N/A — by design |
| Coefficients corrupted or missing | `NewLatencyModel` returns error | Validate for NaN/Inf at construction (R3) | N/A — construction-time error |
| QueueingTime extraction captures scheduler wait | BLIS double-counts delay → TTFT systematically overestimated | Use 5th percentile at lowest load (Section 7a); validate via first-step TTFT bias check (Section 11a) | If TTFT signed error is systematically positive across all models, re-extract alpha using alternative method (set to 0 and absorb into step time, or use SCHEDULED-to-FIRST_TOKEN interval instead) |

### 3b. No-op Default and Configuration Surface

When the learned model is not configured, existing behavior is unchanged. The `NewLatencyModel` factory currently dispatches on a boolean field. A new selection mechanism will be introduced — the specific choice (string enum, additional boolean, etc.) is deferred to micro-planning. **Backward compatibility:** existing `roofline: true` configurations must continue to work. The migration strategy (e.g., boolean maps to named backend) is a micro-planning concern.

**Structural prerequisites** (must be addressed before or in the first implementation PR):
- The model architecture configuration type currently lacks MoE routing parameters (expert count, active experts per token). These must be added to the architecture config, its parsing logic, and the canonical constructor. This touches the architecture config module and the latency config parser.
- The architecture configuration is currently populated only when roofline mode is selected. The learned backend requires it unconditionally. This is a cross-cutting change touching the CLI layer, the config resolution chain, and the latency factory.
- The current coefficient storage is an extensible numeric slice, sufficient through Iteration 1 (adding a batch-size coefficient does not require a format change). Architecture-parameterized forms (Iteration 2+) may need a richer coefficient format — evaluate at that point.

## 4. Alternative Approaches

| Approach | Description | Pros | Cons | Status |
|----------|-------------|------|------|--------|
| **A: Per-model OLS (existing form)** | Fit `beta0 + beta1*prefill + beta2*decode` per model | Minimal code change; fair baseline | No cross-model generalization; 6 coefficients per model | Proposed — Iter 0 |
| **B: Per-model OLS + batch size** | Add `beta3 * batchSize` | Captures known batch-size effect; 8 coefficients per model | Still per-model; marginal over A | Proposed — Iter 1 candidate |
| **C: Separate prefill/decode models** | Different forms for prefill-heavy vs decode-heavy steps | Captures compute-vs-memory-bound asymmetry | More complex; requires batch classification | Proposed — Iter 1-2 candidate |
| **D: Architecture-parameterized** | Coefficients derived from config.json features | Cross-model generalization; single coefficient set | Requires sufficient diversity (4 models may be tight) | Proposed — Iter 2+ candidate |
| **E: Hybrid roofline + residual** | Roofline base + learned correction | Physically grounded + data-driven | Two-stage; roofline errors could dominate | Proposed — if D plateaus |
| **F: Neural network / lookup** | Train NN, export as polynomial/lookup | Maximum expressiveness | Interpretability loss; overfitting risk with 4 models | Not recommended |

**What breaks if wrong:**
- **Functional form too simple:** BLIS systematically underestimates step time → overestimates capacity. Signed bias bounds catch this.
- **Functional form too complex:** Overfitting to 4 models → fails on validation/test. Cross-validation catches this.
- **QueueingTime minimum-interval extraction wrong:** If minimum QUEUED→SCHEDULED includes scheduling wait, BLIS double-counts delay → TTFT systematically overestimated.
- **Clamping activates frequently:** Model is in extrapolation → capacity estimates unreliable. Track clamping rate per experiment.
- **10% sampling biased:** Learned coefficients optimized for non-representative steps → step MAPE good on biased sample, bad on full population. Sampling bias verification (Section 2a) catches this.
- **Cross-model generalization (D) fails:** Fallback is per-architecture-family coefficient sets (dense vs MoE), selected by config.json. User identifies model family, not dedicated profiling.

## 5. Quantitative Success Criteria

### 5a. Step-Level (direct test of learned function)

| Metric | Target | Rationale | Traces to |
|--------|--------|-----------|-----------|
| Step duration MAPE (aggregate) | < 15% | Step time is the atomic unit; errors compound through simulation | Q1, Q4 |
| Step duration MAPE (stratified: prefill-heavy) | < 20% | Prefill-heavy steps drive TTFT; systematic prefill underestimation is dangerous | Q1, Q4 |
| Step duration MAPE (stratified: decode-heavy) | < 15% | Decode steps dominate E2E; most steps are decode-only | Q4 |
| Step duration Pearson r | > 0.90 | Must capture relative ordering of fast vs slow steps | Q3, Q4 |
| Residual bias (mean signed error) | < ±5% of mean | Systematic over/under-prediction shifts all downstream metrics | Q1, Q2 |

Stratification: "prefill-heavy" = steps where prefill_tokens > 50% of total tokens. "Decode-heavy" = all others. Reported separately to prevent decode-step accuracy from masking prefill-step bias.

### 5b. Request-Level (BLIS replay)

| Metric | Validate Target | Test Target | Rationale |
|--------|----------------|-------------|-----------|
| TTFT MAPE | < 20% | < 30% | TTFT = queueing + prefill; most user-visible |
| E2E MAPE | < 15% | < 25% | E2E dominated by decode steps |
| TTFT rank correlation | > 0.80 | > 0.70 | Ordering matters for SLO analysis |
| TTFT mean signed error | < ±10% | < ±15% | Prevents systematic capacity overestimation |
| E2E mean signed error | < ±10% | < ±15% | Same directional bias guard |

**Note:** TTFT and E2E predictions evaluate the *composition* of the learned latency model and BLIS's event-queue scheduler, not the latency model in isolation. If request-level metrics are poor but step-level metrics (Section 5a) are good, the discrepancy points to BLIS scheduler fidelity rather than latency model accuracy.

**H30 finding (#480):** Iter 3 analytical evaluation used `T_queue_obs` (real observed queue wait), masking scheduling model errors. At saturation, `T_queue_obs` ≈ 120s dominates TTFT — coefficient error becomes < 0.02% of total TTFT. The "1.5% test TTFT error" was an illusion. **BLIS replay is the first true end-to-end test** and should be treated as the authoritative evaluation, not a secondary check.

### 5f. Cascade-Aware Constraints (Iter 4+)

Queueing amplifies systematic step-time bias non-linearly. At ρ=0.85, a +5% step-time bias shifts ρ to 0.893, increasing expected queue depth by 47%. At ρ=0.95, the same +5% causes ρ=0.998 (effectively infinite queue). The following constraints prevent cascade divergence:

| Metric | Target | Rationale |
|--------|--------|-----------|
| Step-time signed bias (global) | < ±3% | Keeps ρ perturbation below 0.03 across training-set utilization range |
| Step-time signed bias (per stratum) | < ±5% | Prevents operating-point-specific cascade (low-batch, high-batch, prefill-heavy) |
| Queue-depth RE (BLIS vs real) | < 20% | Leading indicator — queue diverges BEFORE TTFT/E2E diverge |
| Capacity RE (throughput at saturation) | < ±10% | Prevents regime transition (ρ crossing 1.0 boundary) |

Strata: low-batch (dc ≤ 20), medium-batch (20 < dc ≤ 80), high-batch (dc > 80), prefill-heavy (pf > 50% of total tokens).

### 5c. Aggregate-Level (capacity planning fidelity)

| Metric | Validate Target | Test Target | Rationale |
|--------|----------------|-------------|-----------|
| TTFT p50 relative error | < 10% | < 20% | Central tendency for planning |
| TTFT p99 relative error | < 25% | < 40% | Tail latency; harder to predict |
| Throughput relative error | < 10% | < 15% | Must not over/under-estimate capacity |
| Preemption count direction | Correct sign | Correct sign | If real has preemptions, BLIS should too |
| Completion count relative error | Informational | Informational | Reported but NOT gated — see Section 5e for rationale |

### 5d. Cross-Model Generalization

Report all metrics per model separately. "Works across models" means: a single set of learned coefficients + architecture features from config.json produces all the above targets without per-model tuning. "Across models" means within the H100/vLLM-v0.15.1/FLASH_ATTN envelope — not arbitrary deployments.

**Falsification criteria:** Cross-model generalization fails for a model if it misses test-target thresholds on more than 1 of the 3 capacity-planning-critical metrics (throughput RE, TTFT p50 RE, TTFT p99 RE). Failing both throughput AND any TTFT metric is an automatic failure regardless of other metrics. If 2 or more test models fail, the approach is refuted and the fallback is per-architecture-family coefficients.

### 5e. Handling Failed Requests

Reasoning experiments have 0.1%-85% failure rates. Evaluation handles this:

- **Arrival stream:** All requests (including vLLM failures) are in the BLIS replay.
- **Request matching:** Requests matched by arrival index (Nth arrival in real = Nth in BLIS).
- **Latency metrics:** Computed only over requests completing in BOTH real and simulated runs (intersection set by arrival index).
- **Completion count:** Reported as informational, NOT gated. Rationale: BLIS and vLLM have fundamentally different failure mechanisms (BLIS: DroppedUnservable when KV blocks exhausted; vLLM: timeout, OOM, preemption cascade). Making completion count a gating metric would test simulator fidelity, not the latency model. The 30% divergence threshold for switching to step-level-only evaluation is provisional — may be adjusted after Iteration 0 establishes baseline divergence rates.
- **Confound acknowledgment:** Test-set reasoning experiments test the latency model AND BLIS's scheduler/preemption fidelity simultaneously. If completion counts diverge significantly, step-level metrics (Section 5a) are the authoritative evaluation of the latency model itself.

## 6. Hard Constraints

### 6a. Interface Constraint

The learned model implements the existing frozen `LatencyModel` interface: five methods returning microsecond estimates for batch step duration, per-request preprocessing delay, per-token output processing, scheduling overhead, and preemption overhead. Pure query contract — no events, no side effects, no system-state access. Simulation loop unchanged.

### 6b. Determinism

INV-6: same inputs → identical outputs. No random sampling, no hidden state, no floating-point order dependence (summation of features must use deterministic order, not map iteration). Verification: same seed produces byte-identical stdout.

### 6c. Coefficient Portability

Learned coefficients storable in `defaults.yaml` or similar. The current `BetaCoeffs []float64` slice format is extensible for Iterations 0-1 (adding coefficients). Architecture-parameterized forms (Iteration 2+) may need a richer format — evaluate at that point. No training data required at simulation time.

### 6d. Architecture Parameterization

The model accepts architecture features derivable from HuggingFace config.json:
- **Layer geometry:** layer count, hidden dimensions, attention configuration (heads, KV heads, GQA ratio)
- **FFN geometry:** intermediate dimensions, vocabulary size
- **MoE routing:** expert count, active experts per token (requires `ModelConfig` extension — see Section 3b)
- **Data types:** parameter precision (FP16/BF16 → bytes per parameter)
- **Server config:** tensor parallelism degree, maximum batched tokens

## 7. Real-System Correspondence

### 7a. Trace Field → BLIS Method Mapping

| BLIS Method | What it models | vLLM trace source | Measurement |
|-------------|---------------|-------------------|-------------|
| `StepTime(batch)` | GPU execution time for one scheduler step | `step.BATCH_SUMMARY → step.duration_us` | Includes kernel execution + NCCL all-reduce + kernel launch. Does NOT include scheduler CPU time. |
| `QueueingTime(req)` | Non-blocking preprocessing overhead at arrival | `journey.QUEUED → journey.SCHEDULED` minus scheduler wait time | **Extraction procedure:** the QUEUED→SCHEDULED interval conflates preprocessing AND wait-for-batch-slot. To isolate preprocessing: (1) compute per-model percentile distribution of QUEUED→SCHEDULED at rate=stage_0 (lowest load); (2) use the 5th percentile as the preprocessing estimate (minimum is noisy — single outlier; 5th percentile is robust); (3) model as `alpha0 + alpha1*inputTokens` fitted to these low-load intervals. **What breaks:** if even low-load intervals include non-trivial wait, alpha overestimates preprocessing → BLIS double-counts delay → TTFT systematically overestimated. Validate by checking TTFT bias sign. |
| `OutputTokenProcessingTime()` | Per-token post-processing overhead | Derived from journey timing residuals | Non-blocking; adds to client-perceived ITL. |
| `SchedulingProcessingTime()` | Inter-step scheduling overhead per step | `step.BATCH_SUMMARY → step.ts_start_ns, step.ts_end_ns` | **Extraction procedure (Iter 4):** the inter-step gap is `ts_start_ns[k+1] - ts_end_ns[k]` for consecutive steps. This captures scheduler CPU + prepare_model_input + CUDA launch. (1) Extract all consecutive step pairs from BATCH_SUMMARY (10% sample limits pair count). (2) Fit `δ₀ + δ₁·batchSize` via NNLS on the gap durations. (3) Apply as per-step overhead: BLIS schedules next step at `now + stepTime + δ(batchSize)`. **H30 evidence:** both crossmodel and blackbox produce identical TTFT underprediction, proving the gap is in the scheduling model, not coefficients. |
| `PreemptionProcessingTime()` | Overhead per preemption event | Not directly measurable | Returns 0. |

### 7b. BLIS Config Derivation from Experiment Data

| BLIS Parameter | Source | Derivation |
|---------------|--------|------------|
| `--model` | exp-config.yaml: `model` | Direct |
| `--tp` | exp-config.yaml: `tensor_parallelism` | Direct |
| `--max-num-running-reqs` | exp-config.yaml: `max_num_seqs` | Direct (128 for all experiments) |
| `--max-scheduled-tokens` | exp-config.yaml: `max_num_batched_tokens` | Direct (2048 for all experiments) |
| `--total-kv-blocks` | step traces: `kv.blocks_total_gpu` | From first BATCH_SUMMARY event |
| `--block-size` | kv_events: `block_size` in BlockStored | Always 16 in this dataset |
| Model architecture | training/model_configs/\*/config.json | Layer count, hidden dim, heads, etc. |
| Workload replay | per_request_lifecycle_metrics.json | arrival_time (normalized), input_tokens, output_tokens, prefix_group |

## 8. Prior Knowledge Inventory

_(Unchanged from Round 2 — see Sections 8a, 8b, 8c in previous revision. Content moved here for completeness.)_

### 8a. From BLIS Hypothesis Experiments

Step time ≈ `beta0 + beta1*prefillTokens + beta2*decodeTokens` at moderate load. Alpha overhead is non-blocking (H-Step-Quantum #329). Chunked prefill splits long prefills into ~256-token chunks (H27 #433). Prefix cache creates bimodal prefill cost (H-Cross-Model #398). KV pressure drives preemption which changes batch composition (H8, H20 #401). Distribution median drives KV pressure (H20).

### 8b. From Training Data Analysis

4 architectures (7B-70B dense + 8x7B MoE, TP=1/2/4). Reasoning profile hits saturation (85% fail on 7b, 33% on 70b, 0.1% on 34b). 10 prefix groups × 10 prompts. Step sampling = 10% probabilistic. `defaults.yaml` has existing coefficients for codellama (H100/TP=1) and mixtral (H100/TP=2) — different TP from training data for codellama. All experiments: chunked_prefill=True, prefix_caching=True, FLASH_ATTN, vLLM v0.15.1, H100 SXM.

### 8c. Known Limitations of Current Models

Blackbox: per-model, no generalization. Roofline: sums per-request FLOPs additively, no GPU occupancy effects. Neither accounts for batch request count. Alpha is constant. Scheduling/preemption overhead returns 0. No MoE awareness (~4x overestimate for Mixtral). **Note for three-way comparison:** Roofline will perform dramatically worse on Mixtral due to MoE blindness — the learned-vs-blackbox comparison is the meaningful one for Mixtral.

## 9. Named Invariants

**Existing BLIS invariants preserved:**

| Invariant | Requirement |
|-----------|-------------|
| **INV-1** (Request conservation) | Preserved by construction: latency model does not inject, drop, or transition requests. |
| **INV-2** (Request lifecycle) | Preserved by construction: lifecycle transitions managed by event loop and batch formation, unchanged. |
| **INV-3** (Clock monotonicity) | StepTime > 0 for non-empty batches ensures clock always advances. |
| **INV-5** (Causality) | QueueingTime ≥ 0 ensures arrival_time ≤ enqueue_time. |
| **INV-6** (Determinism) | Same inputs → identical outputs. Deterministic feature summation order (no map iteration). Verified: same seed → byte-identical stdout. |
| **INV-8** (Work-conserving) | StepTime ≥ 1 for non-empty batches prevents infinite zero-duration step loops. |

**New invariants:**

| Invariant | Requirement |
|-----------|-------------|
| **INV-L1** (Non-negativity) | All five methods return ≥ 0. StepTime returns ≥ 1 for non-empty batches. |
| **INV-L2** (Per-dimension monotonicity) | StepTime non-decreasing when prefill tokens increase (decode held constant), and non-decreasing when decode tokens increase (prefill held constant). For linear forms: enforced by non-negative coefficients. For non-linear forms: enforced by constrained optimization or output validation. |
| **INV-L3** (Finite outputs) | No NaN, Inf, or overflow for any valid input (see valid-input definition in Section 3). Enforced by coefficient validation + output clamping. |
| **INV-L4** (Backward compatibility) | All existing invariant tests INV-1 through INV-8 pass with the learned model. No behavioral regression. |

## 10. DES Checklist

| # | Question | Answer |
|---|----------|--------|
| 1 | What analysis questions? | Section 1b: capacity planning, model selection, TP sizing, workload optimization, preemption prediction. |
| 2 | Modeled / simplified / omitted? | Section 2a table (15 rows with status tracking). |
| 3 | Events introduced or modified? | None. Pure query module. Existing step-execution event calls StepTime with same interface. |
| 4 | Event interaction with tie-breaking? | N/A — no new events. Different step durations change completion timestamps and thus event ordering, but the (timestamp, priority, seqID) mechanism is structurally unaffected. Cascading effect: different step times → different completion times → different co-batching in subsequent steps → different simulation trajectory. |
| 5 | New state? Who owns it? | Learned coefficients + derived architecture features, owned by the learned model instance. Immutable after construction. |
| 6 | New metrics? Collection strategy? | Evaluation metrics (MAPE, Pearson r, relative error) computed on-demand in external Python scripts against BLIS's standard JSON output. No incremental collection inside the simulator. No new simulator-internal metrics. |
| 7 | New randomness? | No. Deterministic (Section 6b). No PartitionedRNG subsystem. **Note:** floating-point summation order must be deterministic (Section 9, INV-6) — feature computation must not iterate over maps. |
| 8 | Correctness verification? | INV-L1 through INV-L4 via behavioral unit tests. INV-1 through INV-8 regression suite. Same-seed determinism (INV-6). State/statistics separation: trivially maintained — no mutable state, no in-simulator metric computation. |
| 9 | Fidelity validation? | Three-level evaluation (Section 11): step, request, aggregate. Three-way baseline (Section 11c). **Confound separation:** when completion counts diverge >30% in reasoning experiments, step-level metrics are authoritative (Section 5e). |
| 10 | Simplest version? | Per-model OLS with existing functional form (Section 1c). Tested in Iteration 0. |

## 11. Evaluation Protocol

### 11a. Step-Level Evaluation (Analytical — Coefficients in Isolation)

0. **Sampling bias validation (prerequisite):** Compare the joint distribution of (prefill_tokens, decode_tokens, batch_size) between the 10% sampled steps and the full population estimated from journey trace step counts. Acceptance criterion: KS test p > 0.05 per marginal, or marginal mean difference < 10%. If sampling bias is detected, investigate the sampling mechanism before proceeding — biased training data invalidates all subsequent metrics.
1. Extract (batch_features, step_duration_us) pairs from BATCH_SUMMARY events
2. Apply learned StepTime function to batch_features
3. Compare predicted vs actual step_duration_us
4. Report: aggregate MAPE, stratified MAPE (prefill-heavy vs decode-heavy), Pearson r, signed bias
5. **Iteration 0 data analysis:** plot step_duration vs total_tokens. Report monotonicity violation rate. If > 5% of steps violate monotonicity, reconsider INV-L2 enforcement strategy.
6. **QueueingTime extraction validation:** For requests scheduled on the first step after arrival at the lowest-load experiment (near-zero queue wait), compare BLIS TTFT against real TTFT. If TTFT signed error is systematically positive (> +10%) for these first-step requests, the QueueingTime extraction has captured scheduler wait time — re-extract using alternative methods (see Section 3a failure mode table).

**Critical limitation (H30 #480):** Step-level evaluation and journey-level analytical evaluation use real observed data (batch compositions, queue wait times) as inputs. This tests the learned function in isolation but does NOT test system composition — BLIS's batch formation, queueing dynamics, and event loop interact non-linearly with the coefficients. A model that passes 11a with 5% MAPE can still produce 40% TTFT error in BLIS replay (see H30: Iter 3 achieved 5.3% train TTFT analytically but -23% to -56% in BLIS). **Sections 11a and 11b test different properties and are not substitutable.**

### 11b. BLIS Replay Evaluation (System Composition — Authoritative)

1. For each validate/test experiment:
   - Derive BLIS config per Section 7b
   - Set learned latency coefficients (β, α, δ)
   - Convert per_request_lifecycle_metrics.json → BLIS workload replay
   - **Determinism check:** Run twice with same seed, verify byte-identical stdout (INV-6)
   - **Invariant check:** Verify INV-1, INV-4, INV-5 pass
   - Run BLIS simulation
2. Compare per Section 5 (including 5f cascade-aware constraints), broken down per model
3. Track clamping rate (% of steps where INV-L1 clamp activated) — if > 5%, flag model failure
4. **Multi-level diagnostic:** For each experiment, report error at all four levels (step, queue, request, aggregate) to identify WHERE remaining error lives. If step-level is accurate but TTFT diverges, the error is in queueing dynamics (δ or batch formation). If queue depth matches but TTFT diverges, the error is in α/γ.

### 11c. Three-Way Baseline Comparison

Every iteration reports: (1) Blackbox (per-model alpha/beta), (2) Roofline (analytical), (3) Learned. **Note:** Roofline will be dramatically worse on Mixtral due to MoE blindness (~4x overestimate). The learned-vs-blackbox comparison is the meaningful one for Mixtral. Iteration 0 must report all three to bootstrap the ledger.

### 11d. Iteration Ledger

One row per iteration in `training/ledger.md`. Columns: Iter, Strategy, Form, Params, per-model Step MAPE (train), per-model Step MAPE (val), per-model TTFT p99 RE (val), per-model Throughput RE (val), Key insight, Prediction accuracy, Status. Test set metrics reported ONLY for the final model.

### 11e. Cluster-Mode Sanity Check (Post-Training)

After the final model is selected, run a small multi-instance BLIS replay and verify that cluster-mode latency distributions are qualitatively similar to single-instance. The specific instance count, experiment selection, and divergence threshold are micro-planning concerns. Purpose: detect systematic step-time bias that compounds through routing feedback.

**Why this matters:** H29 (#433) showed that even correct step times with stale KV signals cause +242-548% TTFT p99 degradation. H7 (#401) showed super-linear scaling effects where queue growth rate excess `lambda/k - mu` is highly sensitive to `mu` (which depends on step time) — a 5% step-time bias can shift the saturation curve significantly because the excess is a difference of two similar numbers. Single-instance evaluation cannot detect these amplification effects. This check is qualitative and not part of the formal success criteria, but should precede any production deployment claim.

### 11f. Two-Phase Learning Strategy (Iter 4+)

H30-H32 (#480) proved that analytical evaluation and BLIS replay test different properties. Coefficients fitted analytically (minimizing step-time error on real batch compositions) can cascade badly in the simulator (where BLIS forms its own batches and queueing amplifies bias). The two-phase strategy addresses this:

**Phase 1 — Analytical warm-start (convex, fast).** Fit all 8 parameters using established techniques:
- β₀-β₃: NNLS on step data (Block A) + journey constraints (Block B), unchanged from Iter 3
- α₀: minimum-wait journey extraction (5th percentile of QUEUED→SCHEDULED at lowest load)
- α₂/γ₁: NNLS on E2E residuals after subtracting α + prefill + decode predictions
- δ₀, δ₁: NNLS on inter-step timing gaps from BATCH_SUMMARY (`ts_start[k+1] - ts_end[k]`)

Phase 1 exploits convex structure and produces physics-interpretable starting values.

**Phase 2 — Simulator-in-the-loop refinement (blackbox, simulator-faithful).** Run BLIS replay with candidate coefficients and optimize a multi-signal loss:

1. **Observability signals** (from BLIS Metrics after each run):
   - Per-request TTFT/E2E (`RequestTTFTs`, `RequestE2Es`) — primary comparison target
   - Queue depth at every step (`NumWaitQRequests[]`) — cascade indicator
   - Batch size at every step (`NumRunningBatchRequests[]`) — batch formation validation
   - Per-step predicted step time (new instrumentation: `StepDurations[]`) — step accuracy at BLIS's own operating points
   - Throughput, preemption count, completion count — aggregate validation

2. **Multi-signal loss** (weighted composite):
   - L1 step-time (15%): squared signed bias per decode-token bin (prevents operating-point cascade)
   - L2 queue dynamics (25%): resampled queue-depth trajectory L2 norm (leading indicator)
   - L3 TTFT (30%): per-request MAPE matched by arrival index
   - L4 E2E (20%): per-request MAPE matched by arrival index
   - L5 throughput (10%): squared relative error

3. **Coordinate-wise refinement** (7 free parameters, practical budget):
   - **Stage A — δ₀ sweep** (most consequential, 1 param): Fix β, α from Phase 1. Grid search δ₀ ∈ [500, 5000] on 4 fast-subset experiments, validate on all 10. Record per-stage checkpoint.
   - **Stage B — α₀, α₂ refinement** (next most consequential, 2 params): Fix β, δ₀ from Stage A. Bayesian optimization (GP surrogate, 30 evaluations, random_state=42). Record checkpoint. Also run Stage B with δ₀=0 as causality control for H-gamma-decrease.
   - **Stage C — β joint refinement** (4 params, fine-tuning): Fix δ₀, α from Stages A-B. CMA-ES with σ₀=5%, 50 generations × population 10, random_state=42. Record checkpoint.
   - **Stage D — global polish** (all 7, optional): If Stages A-C meet targets, skip.

   Each stage records coefficients and evaluates all 10 training experiments (checkpoints enable per-stage ablation). Fast subset (Stage A): llama-2-7b-general (TP=1), llama-2-70b-general (TP=4), mixtral-8x7b-general (MoE), codellama-34b-codegen (cross-profile). Validation gate: if any non-subset experiment shows TTFT |RE| > 25%, re-run Stage A with all 10.

   **Architecture note:** δ₀ is implemented via a new `InterStepOverhead() int64` method on the `LatencyModel` interface (Phase C interface evolution), NOT by modifying the existing `SchedulingProcessingTime()`. The existing method continues its per-request role at `batch_formation.go:131`. `InterStepOverhead()` fires every step in both paths of `scheduleNextStep()`. See `iter4-bundle.md` Architecture section.

4. **Physics guard-rails**: All parameters bounded ±30% of Phase 1 warm-start (α₂ extended to [0, 1291] to allow physical value discovery). β ≥ 0, δ₀ ≥ 0. If any parameter moves > 30%, it is absorbing error from another source — investigate before accepting.

5. **Validation gate**: After Phase 2 converges, run H31 (reasoning generalization) and H32 (aggregate capacity planning) with refined coefficients. Compare against both Phase 1 warm-start and Iter 3 original.

## 12. Workflow

### 12a. Branch Strategy

- **`training` branch**: problem.md, split.py, schemas.py, ledger.md.
- **Iteration worktrees**: branched from `training`, merged back after FINDINGS review.

### 12b. Parallel Development Path

Three tracks, coupled only by coefficient format:
1. **Python training pipeline** (`training/`)
2. **Go implementation** (`sim/latency/`)
3. **BLIS replay harness** (`training/`)

### 12c. Iteration Loop

Per Strategy Evolution: generate candidate → hypothesis bundle → Design Review (5-perspective) → human gate → implement + Code Review → execute → compare predictions → FINDINGS + FINDINGS Review → self-audit → ledger → Bayesian optimization (if H-main confirmed) → principle extraction → iterate or stop.

### 12d. Stopping Criterion

Stop when two consecutive iterations produce < 2% improvement on ALL of: validation TTFT p99 relative error, validation throughput relative error, AND validation step MAPE. ("All" = AND: the process continues as long as any metric is still improving meaningfully.) Non-regression constraint: no validation metric may degrade > 5% from its previous best.

The final model is evaluated on the test set exactly once.

## 13. Scope Exclusions

| Exclusion | What breaks if relaxed |
|-----------|----------------------|
| **Hardware generalization** — H100 SXM only | Coefficient format gains GPU-type fields. Functional form likely unchanged. |
| **Multi-instance routing** — single-instance only | Latency model unchanged; cluster-mode testing exercises routing composition. |
| **Arrival process** — replay real arrivals | Would need BLIS workload generation; latency model unchanged. |
| **vLLM version** — v0.15.1 only | Different kernels → different step-time profiles. Re-training needed. |
| **Interface changes** — 5 methods fixed | Would unlock load-dependent QueueingTime. Currently the biggest modeling simplification. |
| **Online learning** — offline only | Coefficients static. Online adaptation improves accuracy under distribution shift. |
