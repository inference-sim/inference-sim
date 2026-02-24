# Tiered Accuracy Model with Server-Side Instrumentation

## Executive Summary

[idea3](idea3-tiered-accuracy-model.md) proposed the Tiered Accuracy Model (TAM) under a hard constraint: **no server-side data**. Every experiment was limited to aggregate accuracy comparisons against client-side GuideLLM metrics. This made most hypotheses testable only indirectly — we could measure "did MAPE improve?" but rarely "did the correction fix the right thing?"

This document lifts that constraint. We now have access to an [instrumented vLLM fork](https://github.com/inference-sim/vllm) that emits two classes of server-side telemetry via OpenTelemetry:

| Trace layer | Scope | What it captures |
|-------------|-------|-----------------|
| **Journey traces** (`--enable-journey-tracing`) | Per-request | Lifecycle events (QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED), monotonic timestamps, scheduler step numbers, phase (WAITING/PREFILL/DECODE), preemption counts, finish status |
| **Step traces** (`--step-tracing-sample-rate`) | Per-step | Running/waiting queue depths, prefill/decode request counts, token distribution (prefill tokens, decode tokens), KV cache utilization (usage ratio, total/free GPU blocks), requests finished/preempted, and optionally per-request snapshots (GPU blocks allocated, prefix cache hits, effective prompt length) |

**The seven hypotheses from idea3 are unchanged.** What changes is the *quality of evidence* — from aggregate MAPE comparisons to direct mechanism isolation. This document redesigns each experiment to exploit server-side data, identifies which hypotheses become fully testable (vs. remaining partially indirect), and specifies new experiments that were impossible under the client-only constraint.

**Key wins from instrumentation:**

| Hypothesis | idea3 testability | idea4 testability | What changes |
|-----------|-------------------|-------------------|--------------|
| H1 (bandwidth) | Aggregate TPOT only | Per-step memory-bound fraction via step traces | Can isolate decode-only steps and measure memory-bound bias directly |
| H2 (overhead) | Aggregate TPOT only | Per-step overhead = `SCHEDULED→FIRST_TOKEN` - predicted compute | Can measure overhead per step, validate fixed-vs-variable functional form |
| H3 (GEMM shapes) | GQA vs non-GQA comparison | Same + per-step validation of compute-bound steps | Marginal improvement — still no per-GEMM kernel timing |
| H4 (per-component roofline) | Synthetic + aggregate | Per-step bottleneck transition detection via token distribution | Can identify steps that cross the bottleneck boundary |
| H5 (mixed batch) | QPS as weak proxy | Per-step prefill/decode token ratio from step traces | **Full testability** — can bucket steps by exact prefill/decode ratio |
| H6 (MFU grid) | Full (simulator-internal) | Same | No change needed — already fully testable |
| H7 (MoE) | Blocked on ground truth | Still blocked on MoE ground truth + MFU data | Instrumentation helps when MoE experiments are run |

---

## HG1: perLayerOverhead Calibration via Grid Search with Train/Test Split

**Motivation:** H2b established `perLayerOverhead=100μs/layer` from analytical reasoning (vLLM block-table management, per-layer tensor dispatch). This value achieved 17.0% TPOT MAPE across 13 experiments, but H2b's criterion 2 failed — one experiment worsened by 7.7pp. A data-driven grid search can find the value that minimizes aggregate error, and a train/test split guards against overfitting.

**Why this belongs in idea4 (not idea3):** idea3's constraint is "zero fitted parameters." A grid search that selects the best-fitting value from ground truth data is, by definition, a fitted parameter — even if it's a single scalar. This places it at the boundary between Tier 0 (analytical) and Tier 1 (calibrated). By using a train/test split, we ensure the selected value generalizes beyond the training set.

### Experiment Design

**Train/test split:** Stratify the 13 experiments by model family (5 families) and workload type (3 types). Assign ~70% to train, ~30% to test, ensuring each model family appears in both splits:

| Split | Experiments | Rationale |
|-------|------------|-----------|
| **Train (9)** | llama2-7b-tp1-chatsweep, llama2-7b-tp2-codesweep, llama2-7b-tp4-chatsweep, codellama-34b-tp2-chatsweep, codellama-34b-tp2-codesweep, llama2-70b-tp4-chatsweep, qwen3-14b-tp1-codesweep, qwen3-14b-tp2-chatsweep, qwen2.5-7b-summarization | All 5 families represented; mix of chat/code/summarization |
| **Test (4)** | llama2-7b-tp1-codesweep, llama2-7b-tp2-chatsweep, llama2-7b-tp4-codesweep, llama2-70b-tp4-codesweep | llama2-7b at all 3 TP levels + llama2-70b; unseen workload-TP combos |

**Grid search procedure:**

*Phase 1 (coarse):* Sweep `perLayerOverhead` from 0 to 500μs in 25μs steps. For each value, run BLIS against the 9 train experiments with `bwEfficiencyFactor=0.82` (H1). Compute train-set TPOT MAPE. Identify the coarse optimum.

*Phase 2 (fine):* Sweep ±25μs around the coarse optimum in 5μs steps. Run against train experiments only. Identify the fine-grained optimum.

*Validation:* Run the fine-grained optimum against the 4 test experiments. Compare test-set TPOT MAPE against: (a) H2b default (100μs), (b) no overhead (0μs baseline).

### Analysis

1. **Sweep curve:** Plot TPOT MAPE vs. `perLayerOverhead` for train and test sets separately. If the curves have similar shape and minimum location, the parameter generalizes. If the test-set minimum diverges from the train-set minimum by >30μs, the parameter is model-family-dependent and a single global value is insufficient.

2. **Per-model-family optima:** For each of the 5 model families, find the value that minimizes family-specific TPOT MAPE on train data. If family optima span a >2× range (e.g., 7B wants 60μs, 70B wants 150μs), this motivates per-family `perLayerOverhead` values in `hardware_config_roofline_valid.json`.

3. **Sensitivity:** Measure the width of the "flat region" where TPOT MAPE is within 1pp of the minimum. A wide flat region (≥50μs) means the exact value doesn't matter — use the analytical 100μs. A narrow region (<20μs) means the parameter is sensitive and calibration matters.

4. **Signed error analysis:** At the optimum, check whether TPOT prediction is balanced (mean signed error near zero) or systematically biased. If the optimum trades underprediction for overprediction rather than centering predictions, it may be compensating for other modeling errors (H3-H5).

### Accept Criteria

1. **Generalization:** Test-set TPOT MAPE at the optimum is within 3pp of train-set TPOT MAPE (no overfitting).
2. **Improvement:** Test-set TPOT MAPE at the optimum is < 20% (Tier 0 target from idea3).
3. **Stability:** The train-set optimum and test-set optimum (independently computed) differ by ≤ 30μs.

**Overfitting guard:** If test-set MAPE exceeds train-set MAPE by >5pp, reject the grid-search value and fall back to H2b's analytical 100μs/layer.

### Relation to Other Hypotheses

- **Input:** Requires H1 (BwEfficiencyFactor=0.82) as baseline — all grid search runs include H1 correction.
- **Output:** The selected `perLayerOverhead` value becomes the default for all subsequent hypothesis experiments (H3-H7).
- **Supersedes:** H2b's analytical estimate (100μs) if the grid search finds a significantly better value that generalizes. Otherwise, confirms H2b.

### Implementation

Experiment scripts: [`hypotheses/h-roofline/h2c-overhead-grid-search/`](hypotheses/h-roofline/h2c-overhead-grid-search/)

- `run.sh`: Two-phase grid search with train/test split
- `analyze.py`: Sweep curves, per-family optima, sensitivity analysis, overfitting check

---

## Redesigned Experiments

### H1: Memory Bandwidth Efficiency

**idea3 limitation:** Could not isolate memory-bound steps. Could only compare aggregate TPOT with and without the 0.80 bandwidth factor.

**idea4 experiment — direct per-step validation:**

**Setup:** Run a decode-heavy workload (codesweep, low QPS) against the instrumented vLLM. Collect all step traces.

**Step classification:** From step traces, identify **pure decode steps** (prefill_tokens = 0, decode_tokens > 0). These are unambiguously memory-bound at small batch sizes. For each pure decode step:

1. Record `(batch_size, decode_tokens, kv_cache_usage_ratio)` from the step trace
2. Record `step_duration = next_step.ts - this_step.ts` from consecutive step trace timestamps (or from journey SCHEDULED events within the step)
3. Compute predicted step time with peak BW (current BLIS) and with 0.80 × peak BW (corrected)
4. Compute per-step residual: `measured_step_time - predicted_step_time`

**Analysis:**
- Plot residuals vs. batch size for both BW settings. With peak BW, residuals should be systematically positive (underestimate). With 0.80 × peak, residuals should center near zero for small-batch pure-decode steps.
- Regress: `residual = α × batch_size + β`. With the correct BW, β should be close to zero (no systematic offset for memory-bound steps).

**Accept criterion:** Per-step mean residual for pure-decode steps decreases by ≥ 50% when using 0.80 × peak BW. The correction should reduce |β| (intercept of residual regression) to < 0.5ms.

**Why this is better:** idea3 could only say "TPOT MAPE improved by X pp." idea4 can say "the per-step bias for memory-bound steps specifically was reduced from Y ms to Z ms, confirming the bandwidth mechanism."

---

### H2: Scheduling Overhead

**idea3 limitation:** Could not observe per-step overhead or validate whether it's truly constant. Could only validate that adding a fixed overhead improved aggregate MAPE.

**idea4 experiment — direct overhead measurement:**

**Setup:** Run a low-QPS workload (minimal queuing) with journey and step tracing at 100% sample rate.

**Per-step overhead extraction:** For each request, the journey trace provides:
- `t_scheduled` = `journey.SCHEDULED.ts.monotonic`
- `t_first_token` = `journey.FIRST_TOKEN.ts.monotonic` (for prefill steps)
- `t_queued` = `journey.QUEUED.ts.monotonic`

For decode steps, consecutive journey events within the same request give inter-token timing. But the more useful signal is:

**Step-level overhead isolation:** For each scheduler step:
1. From step traces, record batch composition `(prefill_count, decode_count, prefill_tokens, decode_tokens)`
2. Compute BLIS predicted GPU compute time for this step (using corrected BW from H1)
3. Measure actual step duration from consecutive step trace timestamps
4. `overhead = measured_step_duration - predicted_GPU_compute`

**Analysis:**
- Plot `overhead` vs. `batch_size`, `prefill_count`, `decode_count`, `total_tokens`
- Fit: `overhead_ms = a × batch_size + b × prefill_count + c × decode_count + d`
- Test H2's claim: is `overhead ≈ constant`? Or does it scale with batch size?
- Compare against InferSim's model (5ms decode, 30ms prefill) and BLIS's model (50μs)

**Accept criterion:**
1. The measured mean overhead per decode step is > 1ms (confirming BLIS's 50μs is too low)
2. The measured mean overhead per prefill step is > 5ms (confirming prefill overhead is larger)
3. One of: (a) R² < 0.3 for overhead ~ batch_size regression → confirms H2's constant model, or (b) R² ≥ 0.3 → reveals batch-dependent overhead, use the fitted linear model instead

**Why this is better:** idea3 had to sweep overhead in 1ms increments and pick the best aggregate MAPE. idea4 directly measures the overhead distribution and its functional form, resolving whether it's truly constant or batch-dependent — a question idea3 explicitly flagged as unanswerable from client data.

**New experiment — prefill vs. decode overhead decomposition:**

From journey traces, compute:
- `prefill_overhead = (journey.FIRST_TOKEN.ts - journey.SCHEDULED.ts) - predicted_prefill_compute`
- For decode: measure inter-token intervals from consecutive step traces minus predicted per-token compute

This decomposes into separate prefill and decode overhead constants, rather than a single blended value. InferSim uses two (30ms prefill, 5ms decode) — we can validate this split directly.

---

### H3: GEMM Shape Mismatch (Fused QKV)

**idea3 limitation:** Could only compare aggregate E2E MAPE on GQA vs. non-GQA models. Could not isolate GEMM-component error.

**idea4 experiment — improved but still indirect:**

Server-side traces don't provide per-kernel timing (that would require CUDA profiling, not OTEL). However, step traces improve the experiment in two ways:

1. **Cleaner step isolation:** We can filter to pure-prefill steps (decode_tokens = 0) and pure-decode steps (prefill_tokens = 0), removing the mixed-batch confound from H5.

2. **Compute-dominated step identification:** For large-batch pure-prefill steps, compute dominates memory. The GEMM shape correction should have maximal impact on these steps. From step traces, select steps where `prefill_tokens > 512` and `decode_tokens = 0`.

**Per-step test:**
1. For each compute-dominated step, compute predicted time with split Q/K/V/O and fused QKV+O
2. Compare per-step residuals: `measured - predicted`
3. The fused model should reduce per-step residual variance for GQA models more than for non-GQA models

**Accept criterion:** Same as idea3 (E2E MAPE ≥ 2pp better on GQA models), plus: per-step residual standard deviation decreases by ≥ 15% on compute-dominated prefill steps for GQA models.

**Why this is marginally better:** The step-level filtering removes noise from mixed batches, giving a cleaner signal. But the fundamental limitation (no per-kernel timing) persists — we still can't observe the GEMM MFU mismatch directly.

---

### H4: Per-Component Roofline

**idea3 limitation:** Mechanism validated only synthetically (Part A). Aggregate accuracy (Part B) couldn't identify steps at the bottleneck transition.

**idea4 experiment — transition detection from step traces:**

**Setup:** Run a codesweep workload (long outputs → growing KV caches → decode steps transition from compute-bound to memory-bound during generation).

**Transition detection:** For each step, the step trace provides `(batch_size, decode_tokens, kv_cache_usage_ratio)`. As a request generates more tokens, `kv_cache_usage_ratio` increases and the operational intensity of attention shifts from compute-bound to memory-bound.

**Per-step bottleneck classification:**
1. For each step, compute the per-component operational intensity:
   - `attn_OI = attention_FLOPs / kv_cache_bytes` (decreases as KV grows)
   - `mlp_OI = mlp_FLOPs / weight_bytes` (constant per batch size)
2. Classify: if `attn_OI < roofline_ridge_point` and `mlp_OI > roofline_ridge_point`, the step is in the "mixed bottleneck" regime
3. Compare residuals (measured - predicted) for mixed-bottleneck steps under aggregate vs. per-component roofline

**Analysis:**
- Bucket steps into three regimes: both-compute-bound, both-memory-bound, mixed-bottleneck
- The per-component roofline should specifically improve predictions in the mixed-bottleneck bucket
- Plot: residual distribution per bucket, for aggregate vs. per-component model

**Accept criterion:** Per-step MAPE in the mixed-bottleneck bucket improves by ≥ 10% when switching from aggregate to per-component roofline. The other two buckets should show < 2% change (confirming the fix targets the right regime).

**Why this is better:** idea3's Part A was purely synthetic. idea4 identifies *real* steps at the transition boundary and measures the correction's impact on exactly those steps.

---

### H5: Mixed-Batch Model

**idea3 limitation:** "Weakest testability" — no batch composition visibility. Used QPS as a proxy.

**idea4 experiment — full testability via step traces:**

This is the single biggest improvement from instrumentation. Step traces provide the exact per-step `(prefill_tokens, decode_tokens)` — the variables idea3 could not observe.

**Setup:** Run high-QPS workloads across all 14 ground truth experiments (to maximize mixed batching). Collect all step traces.

**Per-step mixed-batch analysis:**

1. For each step, compute `prefill_ratio = prefill_tokens / (prefill_tokens + decode_tokens)`
2. Compute predicted step time under:
   - (a) BLIS weighted-average model: `0.75*P + 0.25*D` / `0.35*P + 0.65*D` / weighted blend
   - (b) Additive model: `GEMM(total_batch) + PrefillAttn(prefill_tokens) + DecodeAttn(decode_tokens, kv_len)`
3. Compute per-step residual for each model

**Analysis:**
- Bucket steps by `prefill_ratio` into deciles (0-10%, 10-20%, ..., 90-100%)
- Plot mean residual per bucket for both models
- idea3 predicted: the weighted-average model should be worst near 50/50 prefill/decode ratio; the additive model should be uniformly better
- **Directly validate idea3's Part A prediction** (≥ 15% latency difference at 50/50) using real steps, not synthetic configurations

**Per-request validation via journey traces:**
- For requests that span mixed-batch steps, compute `journey.FIRST_TOKEN.ts - journey.SCHEDULED.ts` and compare against BLIS predicted prefill time. If the additive model is correct, TTFT prediction should improve when the request's prefill step was mixed.

**Accept criterion:**
1. Per-step MAPE for steps with `prefill_ratio ∈ [0.3, 0.7]` (the "mixing zone") improves by ≥ 20% under the additive model
2. The additive model's residuals do not correlate with `prefill_ratio` (R² < 0.1), confirming it correctly handles all mixing regimes
3. The weighted-average model's residuals correlate with `prefill_ratio` (R² > 0.3), confirming it systematically mistreats certain mixing regimes

**Why this is dramatically better:** idea3 rated this "weakest testability." idea4 has *direct access* to the per-step variable the hypothesis is about. The experiment goes from "QPS as a weak proxy for mixed batching" to "bucket steps by their exact prefill/decode composition."

---

### H6: MFU Grid-Boundary Smoothing

**idea3 testability:** Already full — Part A is simulator-internal, Part B uses aggregate comparison.

**idea4 experiment — same, with minor improvement:**

Step traces don't add much here since the MFU lookup is entirely simulator-internal. The one improvement: with per-step batch sizes from step traces, we can verify that BLIS's MFU lookup was queried with the *correct* batch size (matching the actual step), reducing the chance that a MAPE improvement from smoothing is actually compensating for a batch-size mismatch.

**Additional validation step:**
- For each step trace with `(batch_size, prefill_tokens, decode_tokens)`, verify that BLIS's internal StepConfig for the same step uses consistent batch sizes
- This is a sanity check, not a new hypothesis test

---

### H7: MoE Architecture Support

**idea3 limitation:** Blocked on MoE ground truth experiments. Only Part A (cross-simulator comparison) was feasible.

**idea4 experiment — instrumented MoE runs unlock Part B:**

When MoE ground truth experiments are collected with the instrumented vLLM:

**H7a (FLOPs overestimate):**
- Step traces from MoE models show the same `(prefill_tokens, decode_tokens)` per step
- Per-step residuals should show systematic overestimate (predicted > measured) for compute-dominated MoE steps
- After correcting MLP FLOPs by `topK / numExperts`, residuals should center near zero

**H7b (grouped GEMM efficiency):**
- Journey traces provide per-request TTFT and decode timing
- Compare per-step residuals before/after applying grouped GEMM MFU discount
- Steps with larger batch sizes (more tokens per expert, better utilization) should need smaller discounts

**H7c (expert weight loading):**
- Step traces provide KV cache utilization, which correlates with memory pressure
- At small batch sizes, expert weight loading should dominate → step time should be insensitive to batch size
- At large batch sizes, compute should dominate → step time should scale linearly with batch size
- Step traces let us observe this crossover directly by plotting step time vs. batch size for MoE models

**H7d (shared experts):**
- Per-step validation: shared expert contribution should be additive
- Compare step time predictions with/without shared expert term against measured step duration

**H7e (EP communication):**
- Journey traces on multi-node deployments show preemption events and scheduling delays
- Step traces show queue depths per step — communication latency manifests as increased gap between steps
- Compare predicted inter-step gap (including communication) against measured

**Accept criterion:** Same as idea3 Part B (E2E MAPE < 20% on MoE models) plus: per-step residuals for MoE models should have mean absolute value < 5ms after all H7 sub-corrections are applied.

---

## New Experiments Enabled by Instrumentation

These experiments were impossible under the client-only constraint.

### N1: Queuing Model Validation

**Motivation:** BLIS's DES produces predicted queue times that directly affect TTFT and E2E. Under the client-only constraint, we could not separate queuing error from compute error — both appear as TTFT bias.

**Experiment:**
1. From journey traces, extract per-request `queue_time = SCHEDULED.ts - QUEUED.ts`
2. From BLIS, extract predicted queue time for the same request (from the DES's event log)
3. Compute per-request queue time residual: `measured - predicted`

**Analysis:**
- Plot predicted vs. measured queue time (should be close to y = x)
- Identify regimes where BLIS's queuing model diverges (high load, bursty arrivals, after preemptions)
- This isolates queuing error from compute error — previously confounded in TTFT

**Accept criterion:** Queue time prediction Pearson r ≥ 0.85; per-request queue time MAPE < 30%.

### N2: Preemption Impact Quantification

**Motivation:** Preemptions are invisible from client-side data — they add latency that appears as "compute slowdown." Journey traces expose preemption count and timing.

**Experiment:**
1. From journey traces, identify requests with `num_preemptions > 0`
2. Compute: `preemption_penalty = Σ (SCHEDULED.ts[resume_i] - PREEMPTED.ts[i])` across all preemptions
3. Compare BLIS predicted latency (which includes modeled preemption) against measured latency with and without the preemption penalty

**Analysis:**
- What fraction of E2E error is attributable to preemption modeling vs. compute modeling?
- Does BLIS correctly predict *when* preemptions occur (same step number)?
- Is the preemption → recompute cost accurately modeled?

**Accept criterion:** For preempted requests, E2E MAPE improves by ≥ 5pp when preemption timing is accurately modeled.

### N3: Batch Size Distribution Validation

**Motivation:** BLIS's DES predicts what batch sizes will form at each step. This drives both compute predictions and memory predictions. Under client-only constraint, batch sizes were unobservable.

**Experiment:**
1. From step traces, extract `(step_number, batch_size = prefill_count + decode_count)`
2. From BLIS, extract predicted batch size per step
3. Compare distributions: CDF, mean, p95, autocorrelation

**Analysis:**
- If batch size distributions diverge, ALL per-step predictions inherit systematic error
- Common failure modes: BLIS over-batches (more aggressive scheduler) or under-batches (more conservative admission)
- This is a root-cause diagnostic — it tells you whether step-level errors come from *what batch was formed* vs. *how long that batch took*

**Accept criterion:** KL divergence between predicted and measured batch size distributions < 0.1 nats; mean batch size within 10%.

### N4: KV Cache Utilization Trajectory

**Motivation:** Step traces provide `kv_cache_usage_ratio` per step. BLIS's KV cache manager produces its own utilization trajectory. Divergence means the DES's memory pressure model is wrong.

**Experiment:**
1. From step traces, extract `(step_number, kv_cache_usage_ratio, free_gpu_blocks)`
2. From BLIS, extract predicted KV utilization per step
3. Plot both trajectories over time; compute per-step divergence

**Analysis:**
- If BLIS under-predicts KV usage, it will under-predict preemptions and over-predict batch sizes
- If BLIS over-predicts KV usage, it will over-predict preemptions and under-predict throughput
- This connects to N2 (preemption) and N3 (batch size) — KV utilization is the causal root

**Accept criterion:** Mean absolute KV utilization error < 5% across the simulation; the onset of KV pressure (first step where utilization > 80%) matches within ±10 steps.

### N5: Prefix Cache Hit Rate Validation

**Motivation:** Rich step traces (when `--step-tracing-rich-subsample-rate` is enabled) provide per-request `gpu_blocks_prefix_cache_hits`. BLIS models prefix caching — this validates the hit rate predictions.

**Experiment:**
1. From rich step traces, compute prefix cache hit rate = `Σ prefix_cache_hits / Σ gpu_blocks_allocated`
2. From BLIS, compute predicted prefix cache hit rate from the simulated PrefixCacheIndex
3. Compare over time: does BLIS correctly predict when cache hits occur?

**Analysis:**
- Prefix cache hits reduce effective prefill length (reported as `effective_prompt_length` in rich traces)
- If BLIS's hit rate is too high, it under-predicts prefill time; if too low, it over-predicts
- This is especially important for prefix-heavy workloads (shared system prompts)

**Accept criterion:** Predicted prefix hit rate within 10% of measured; effective prompt length correlation r ≥ 0.9.

---

## Revised Tier Framework

Server-side instrumentation changes the tier structure. The original tiers were organized by *calibration cost*. With instrumentation, we add a new dimension: *mechanism verifiability*.

### Tier 0: Zero-Config (unchanged)

Same seven corrections from idea3. No fitted parameters. Hardware physics and execution semantics only.

**What changes:** Every correction now has a per-step accept criterion in addition to the aggregate accept criterion. This means we can validate the *mechanism*, not just the *net effect*.

| Hypothesis | idea3 accept criterion | idea4 additional accept criterion |
|-----------|----------------------|----------------------------------|
| H1 | TPOT MAPE ≥3pp better | Per-step decode residual mean < 0.5ms |
| H2 | TPOT MAPE ≥3pp better on holdout | Measured overhead within [1ms, 30ms]; functional form validated (constant vs. linear) |
| H3 | E2E MAPE ≥2pp better on GQA | Per-step residual σ ≤ 15% lower on compute-dominated prefill steps (GQA) |
| H4 | Synthetic ≥10% + E2E ≥1pp | Per-step MAPE ≥10% better in mixed-bottleneck regime specifically |
| H5 | Synthetic ≥15% + E2E ≥2pp at high QPS | Per-step MAPE ≥20% better in [0.3, 0.7] prefill_ratio bucket; residual decorrelates from prefill_ratio |
| H6 | ≥80% fewer discontinuities | No change |
| H7 | Within 20% of InferSim (Part A) | Per-step residuals < 5ms on MoE steps (when data available) |

### Tier 1: One-Trace Calibration (improved)

idea3 fit: `residual_ms = a × outputTokens + b × promptTokens + d` (token counts as proxy for unobservable batch state).

**With instrumentation**, the Tier 1 fit can use server-side features:
```
residual_ms = a × batch_size + b × prefill_tokens + c × decode_tokens
            + d × kv_cache_usage + e × prefill_count + f
```

These features are directly observable from step traces, eliminating the proxy problem.

**Protocol:**
1. Run one experiment with full instrumentation (journey + step traces at 100%)
2. Join: match each BLIS predicted step to the corresponding step trace
3. Compute per-step residual: `measured_step_duration - BLIS_predicted_step_time`
4. Fit the residual model above on 70% of steps; validate on 30%
5. Cross-validate across experiments (train on 10, hold out 4)

**Overfitting guard:** Same as idea3 — validation MAPE > 2× training MAPE → reject, fall back to Tier 0.

**Staleness detection:** Same — residual MAPE > 30% on new trace → flag as stale.

### Tier 2: Hardware Characterization (unchanged)

Automated micro-benchmarks for per-access-pattern bandwidth. Independent of server-side traces.

---

## Experiment Execution Plan

### Phase 1: Instrumentation Validation (pre-hypothesis)

Before testing any hypothesis, validate that the instrumented vLLM produces consistent data:

1. **Trace completeness:** Run a short workload (100 requests). Verify every request has a complete journey (QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED) and at least one step trace per scheduler iteration.
2. **Client-server consistency:** For each request, verify `client_TTFT ≈ journey.FIRST_TOKEN.ts - journey.ARRIVED.ts + network_RTT`. Any systematic discrepancy indicates clock synchronization issues.
3. **Step trace consistency:** Verify `Σ (step.decode_tokens)` across all steps for a request ≈ `request.output_token_count`. Any mismatch indicates dropped step traces.
4. **Overhead baseline:** At low QPS (no queuing), measure `journey.SCHEDULED.ts - journey.QUEUED.ts`. This should be near zero — if not, there's a scheduling delay floor that sets a lower bound for H2's overhead constant.

### Phase 2: Hypothesis Testing (H1 → H2 → H5 → H4 → H3 → H6 → H7)

Execution order changed from idea3 (H1, H2, H3, H5, H4, H6, H7) to prioritize hypotheses that benefit most from instrumentation:

1. **H1 (bandwidth)** — first, because it's the largest expected impact and the corrected BW is needed as input for H2's overhead isolation
2. **H2 (overhead)** — second, because the per-step overhead measurement requires accurate compute prediction (from H1)
3. **H5 (mixed batch)** — third, because it sees the largest testability upgrade (from "weakest" to "full") and the per-step prefill/decode ratio is the key new observable
4. **H4 (per-component roofline)** — fourth, because bottleneck transition detection requires step-level token distribution data
5. **H3 (GEMM shapes)** — fifth, because it benefits least from instrumentation (no per-kernel data)
6. **H6 (MFU grid)** — sixth, unchanged priority (already fully testable)
7. **H7 (MoE)** — last, blocked on MoE ground truth

**Within each hypothesis:**
1. Run all 14 ground truth experiments with full instrumentation
2. Apply the correction
3. Validate against both idea3's aggregate accept criteria AND idea4's per-step accept criteria
4. Regression gate: if any previously-accepted hypothesis's per-step metric worsens by > 2pp, investigate interaction before accepting

### Phase 3: New Experiments (N1 → N3 → N4 → N2 → N5)

After hypothesis testing:

1. **N1 (queuing)** — foundational: separates queuing error from compute error
2. **N3 (batch size)** — second: validates the DES's core scheduling behavior
3. **N4 (KV utilization)** — third: validates memory pressure modeling (causal root of N2)
4. **N2 (preemption)** — fourth: requires N4's KV utilization trajectory as context
5. **N5 (prefix cache)** — fifth: requires rich sub-sampling; run last to avoid overhead impact on other experiments

---

## Limitations and Open Questions

1. **Instrumentation overhead:** Server-side tracing adds overhead to vLLM (OTEL span creation, event logging). At 100% step sampling, this could add 50-200μs per step. This is small relative to step time (5-100ms) but should be quantified. Run a no-tracing baseline and compare throughput.

2. **Step duration measurement:** Step traces don't include an explicit `step_duration` field. Duration must be inferred from consecutive step timestamps (difference between step N and step N+1). This assumes steps are sequential with no gap — which holds for vLLM's single-threaded scheduler loop but may not hold for async variants.

3. **CUDA kernel overlap:** OTEL events are CPU-side. GPU kernels may overlap with CPU scheduling. The measured "step duration" includes both GPU compute and CPU scheduling, which is what we want for H2, but means per-step compute time includes a small CPU-GPU overlap component that our compute model doesn't account for.

4. **No per-kernel timing:** Even with full instrumentation, we don't have per-GEMM or per-attention-kernel timing. H3 (GEMM shapes) and H7b (grouped GEMM MFU) remain testable only at the aggregate-per-step level, not the per-kernel level. CUDA profiling (nsys/ncu) would be needed for kernel-level validation, but is out of scope for this framework.

5. **MoE experiments still blocked:** Instrumentation helps *when* MoE ground truth is collected, but doesn't remove the blocking dependency on running MoE models. Collecting Mixtral-8x7b and DeepSeek-V3 ground truth with full instrumentation should be a priority.

6. **Sampling rate tradeoffs:** 100% step sampling may be too expensive for long-running production-scale experiments. The analysis scripts should be designed to work with partial samples (1-10%) and extrapolate, with confidence intervals that widen with lower sampling.

7. **Clock synchronization:** Journey traces use `ts.monotonic` (server-side high-precision clock). GuideLLM uses client-side wall clock. The join assumes negligible clock skew between client and server (valid when co-located on the same machine, questionable for remote benchmarks). The Phase 1 validation step checks for this.

---

## References

All references from [idea3](idea3-tiered-accuracy-model.md) apply, plus:

| Reference | Citation |
|-----------|----------|
| Instrumented vLLM fork | [github.com/inference-sim/vllm](https://github.com/inference-sim/vllm) |
| Journey tracing spec | [JOURNEY_TRACING.md](https://github.com/inference-sim/vllm/blob/main/JOURNEY_TRACING.md) |
| Step tracing spec | [STEP_TRACING.md](https://github.com/inference-sim/vllm/blob/main/STEP_TRACING.md) |
| OpenTelemetry | [opentelemetry.io](https://opentelemetry.io) — CNCF observability framework |
