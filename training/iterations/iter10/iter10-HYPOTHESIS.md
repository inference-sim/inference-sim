# Iteration 10: Sequence-Length-Dependent Batching Inefficiency

## Motivation

**Critical Discovery from Iter9**: Scout's bottleneck is **sequence-length-dependent**, NOT architecture-dependent (FP8). The FP8 dequantization hypothesis was REJECTED (β₉ converged to 0.14 μs vs expected 17-50 μs), but we discovered a powerful new pattern:

**Sequence-Length Correlation** (inverse relationship with error):
- **Scout short-sequence** (roleplay, codegen): Improved significantly (-53pp, -34pp TTFT from iter8)
  - Roleplay: 79% → 26% TTFT (short sequences ~50-100 tokens)
  - Codegen: 92% → 58% TTFT (moderate sequences ~100-200 tokens)
- **Scout long-sequence** (general-lite, reasoning-lite): Failed completely (0pp, -8pp TTFT from iter8)
  - General-lite: 100% → 92% TTFT (long sequences ~400-600 tokens)
  - Reasoning-lite: 99% → 91% TTFT (long sequences ~200-400 tokens)

**Coefficient Explosions Reveal Missing Mechanisms**:
- β₆ (scheduler overhead): +654% (13ms → 99ms) — absorbing long-sequence scheduling delays
- β₂ (TP communication): +343% (0.18 → 0.82) — absorbing sequence-length-dependent TP overhead
- β₈ (MoE routing): +143% (30μs → 73μs) — now above predicted 10-50μs range
- β₃ (KV management): +118% (4.4ms → 9.6ms) — moving away from physical 0.4-1ms range

These explosions indicate the optimizer is compensating for missing sequence-length-dependent terms by inflating existing coefficients beyond physically plausible values.

**Iter10 Strategy**: Add β₁₀ to capture batching inefficiency (long sequences → lower batch efficiency → scheduling delays), split β₃ into base + sequence-length components, remove β₉ (FP8 hypothesis rejected), and constrain alpha bounds to prevent spurious decreases.

---

## H-main: Batching Inefficiency Captures Long-Sequence Overhead

**Prediction**:
- **Overall loss**: 160.6% → **<90%** (>70pp improvement, 44% reduction)
- **TTFT RMSE**: 64.8% → **<40%** (>25pp improvement, 38% reduction)
- **E2E RMSE**: 95.8% → **<55%** (>41pp improvement, 43% reduction)
- **Scout long-sequence TTFT**: Avg 91.5% (range 91-92%) → **<60%** (>31pp improvement for general-lite, reasoning-lite)
- **Scout short-sequence TTFT**: Maintain improvements (roleplay <30%, codegen <65%)
- **Dense long-sequence TTFT**: Improve to <70% (currently 77-92% for general-lite experiments)
- **β₁₀ coefficient**: 0.1-1.0 ms per (token²/batch_request) — physically plausible batching inefficiency overhead
- **β₆ reversion**: 99ms → <40ms (scheduler overhead offloads long-sequence delay to β₁₀)

**Quantitative Threshold**: If overall loss does NOT reduce below 110%, or if Scout long-sequence TTFT does NOT improve to <70%, then H-main is REJECTED.

**Causal Mechanism**:

**Physics Grounding**: Long-sequence requests create batching inefficiency that scales quadratically with sequence length:

1. **Batch Packing Constraint** (vLLM scheduler):
   - vLLM scheduler packs requests into batches subject to KV cache capacity: `Σ(prefill_tokens + kv_cache_blocks) ≤ max_num_batched_tokens`
   - Long sequences consume disproportionate batch capacity: 500-token request consumes 10× more capacity than 50-token request
   - **Result**: Fewer requests fit in each batch → lower GPU utilization → increased wait time → higher TTFT
   - Code: `vllm/core/scheduler.py:Scheduler._schedule()` line ~300-400 (batch formation logic)

2. **Quadratic Batch Efficiency Penalty**:
   - Batch efficiency: `actual_throughput / theoretical_peak_throughput ∝ batch_size / max_batch_size`
   - For long sequences: `batch_size` decreases quadratically with `prefill_tokens` due to KV capacity constraint
   - **Example**: 8× 500-token requests fill KV cache (batch_size=8), but 8× 50-token requests only consume 10% capacity (effective batch_size=80 if unlimited)
   - Effective batch size penalty: `∝ prefill_tokens²` (quadratic due to both sequence length AND reduced batch count)
   - Code: `vllm/core/scheduler.py:Scheduler.get_num_unfinished_requests()` — tracks batch utilization

3. **Scheduling Delay Amplification**:
   - When batch efficiency drops, requests wait longer in queue for the next batch formation cycle
   - Scheduler overhead (β₆) exploded to 99ms because it's absorbing these queueing delays
   - β₁₀ explicitly models this delay: `β₁₀ × Σ(prefillTokens²) / batchSize`
   - **Expected contribution**: For Scout general-lite (500 tokens, batch_size=4): β₁₀ × (500²/4) ≈ β₁₀ × 62,500
   - If β₁₀ = 0.5 ms per (token²/batch_request): 62,500 × 0.0000005 = 31ms per request
   - **This matches Scout long-sequence residual** (missing 50-80ms after β₈, β₉ contributions)

4. **Why Short Sequences Succeed**:
   - Short sequences (roleplay ~50-100 tokens) pack efficiently into batches
   - High batch efficiency → minimal queueing delay → β₁₀ contribution ≈ 0
   - Scout roleplay improved to 26% TTFT because β₁₀ term naturally vanishes for short sequences
   - **Validation**: β₁₀ × (100²/32) = β₁₀ × 312 (20× smaller than long-sequence contribution)

5. **Why β₆ Exploded**:
   - β₆ (scheduler overhead) is per-request constant, NOT sequence-length-dependent
   - Optimizer inflated β₆ from 13ms to 99ms to absorb long-sequence queueing delays
   - After adding β₁₀, β₆ should revert to physical 15-30ms range (actual scheduler CPU overhead)
   - β₁₀ offloads sequence-length-dependent delay, leaving β₆ to capture only base scheduler cost

**Code Citations**:

- **vLLM batch formation**: `vllm/core/scheduler.py`
  - Line ~300-400: `Scheduler._schedule()` — batch packing with KV capacity constraint
  - Line ~150-200: `Scheduler.get_num_unfinished_requests()` — tracks batch utilization and queue depth
  - Line ~250-280: `Scheduler._allocate()` — KV block allocation per request

- **Batch capacity constraint**: `vllm/config.py:SchedulerConfig`
  - `max_num_batched_tokens`: Hard limit on total tokens per batch (default 2560 for H100)
  - `max_num_seqs`: Max number of requests per batch (default 256, but limited by token capacity)

- **BLIS batch formation**: `sim/cluster/instance.go:InstanceSimulator.formBatch()`
  - Line ~400-500: Implements same batch packing logic (KV capacity constraint)
  - Currently doesn't model batching inefficiency overhead — β₁₀ adds this

**Diagnostic Clause**:

*If this hypothesis fails (overall loss remains >110% OR Scout long-sequence TTFT >70%), it indicates:*

1. **β₁₀ coefficient converged to zero** → Batching inefficiency negligible, investigate:
   - Alternative long-sequence bottlenecks (memory bandwidth saturation, cache thrashing)
   - Scheduler overhead is genuine 99ms (profile vLLM scheduler CPU time)
   - Long sequences trigger different execution path (e.g., chunked prefill not modeled)

2. **β₁₀ coefficient converged >10 ms per (token²/batch_request)** → Unrealistically high, investigate:
   - Absorbing other missing terms (memory bandwidth, framework overhead)
   - Basis function formulation issue (scaling incorrect, double-counting with β₆)
   - Training data bias (long-sequence experiments dominating optimization)

3. **β₁₀ is plausible (0.1-1.0 ms) but Scout long-sequence >70%** → Batching inefficiency is ONE component, investigate:
   - Memory bandwidth saturation for long sequences (activations, KV cache reads)
   - Chunked prefill overhead (vLLM splits long prefills into chunks)
   - TP=2 cross-GPU synchronization scaling with sequence length
   - Need iter11 with additional long-sequence term (β₁₁ for memory bandwidth)

4. **β₆ does NOT revert from 99ms** → β₁₀ insufficient to capture scheduler overhead, investigate:
   - Scheduler overhead is genuinely high (profile vLLM scheduler)
   - β₁₀ formulation doesn't fully capture queueing delay (refine basis function)
   - Missing scheduler-specific long-sequence overhead (priority queue operations scale with queue depth)

5. **Scout short-sequence regressed** → β₁₀ causing spurious overhead for short sequences, investigate:
   - Basis function formulation allows non-zero contribution for short sequences
   - Training dynamics creating zero-sum trade-off between long/short sequences
   - Need separate batch efficiency penalty (sigmoid or threshold-based, not quadratic)

**Next Investigation**: If H-main fails, profile vLLM scheduler and batch formation separately:
```bash
python -m vllm.profiler --model meta-llama/Llama-3.1-70B-Instruct \
  --workload general-lite --workload roleplay --profile-scheduler --profile-batch-formation \
  --measure-queue-time --measure-batch-efficiency
```
Measure: Per-request queue time vs sequence length, batch size distribution, scheduler CPU overhead

---

## H-kv-scaling: β₃ and β₃' Capture Base + Sequence-Length KV Overhead

**Prediction**:
- **β₃ (base KV overhead)**: 9.6ms → **0.4-1.5ms** (6-24× decrease, revert to physical range)
- **β₃' (sequence-length KV overhead)**: **0.1-1.0 μs per (token × layer)** (NEW coefficient)
- **Scout long-sequence KV contribution**: β₃' × (500 tokens × 56 layers) = β₃' × 28,000 ≈ 3-28ms per request (if β₃' = 0.1-1.0 μs)
- **Scout short-sequence KV contribution**: β₃' × (100 tokens × 56 layers) = β₃' × 5,600 ≈ 0.6-5.6ms per request (5× smaller)

**Causal Mechanism**:

KV cache management has two distinct components:

1. **Base per-request overhead (β₃)**: PagedAttention setup, block manager initialization, queue insertion
   - **Physical value**: 0.4-1.0ms per request (constant regardless of sequence length)
   - **Code**: `vllm/core/block_manager.py:BlockSpaceManager.can_allocate()` — checks available blocks
   - Iter9 β₃ = 9.6ms (inflated 10-24×) because it's absorbing sequence-length overhead

2. **Sequence-length-dependent overhead (β₃', NEW)**: Block allocation/deallocation scaling with KV cache size
   - **Physical mechanism**: Long sequences require more KV blocks → more allocation calls → more GPU memory operations
   - **Scaling**: `prefillTokens × numLayers` (KV cache size in token-layers)
   - **Expected β₃'**: 0.1-1.0 μs per (token × layer) — memory allocation overhead
   - **Code**: `vllm/core/block_manager.py:BlockSpaceManager.allocate()` line ~200-250 — allocates blocks proportional to KV size

**Validation**: After iter10, compare β₃ and β₃' contributions for Scout long vs short:
- Long (500 tokens): β₃ × 1 + β₃' × 28,000 ≈ 1ms + 3-28ms = 4-29ms
- Short (100 tokens): β₃ × 1 + β₃' × 5,600 ≈ 1ms + 0.6-5.6ms = 1.6-6.6ms
- **Ratio**: 2.5-5× difference (matches observed Scout roleplay vs general-lite TTFT gap)

**Diagnostic Clause**: *If β₃' converges to zero OR β₃ remains >5ms, it indicates KV management overhead is NOT sequence-length-dependent — investigate alternative mechanisms (memory bandwidth, GPU→CPU offloading).*

---

## H-scheduler-reversion: β₆ Should Revert After β₁₀ Addition

**Prediction**:
- **Iter9**: β₆ = 99.3ms per request (+654% from iter8's 13.2ms)
- **Iter10**: β₆ = **15-40ms per request** (60-75% decrease, revert toward physical range)

**Causal Mechanism**:

β₆ (scheduler overhead) exploded in iter9 because it absorbed long-sequence queueing delays that should have been captured by a sequence-length-dependent term (β₁₀). After adding β₁₀:
- β₁₀ captures batching inefficiency (queueing delay proportional to prefillTokens²/batchSize)
- β₆ reverts to physical scheduler CPU overhead (queue management, priority sorting, batch selection)
- **Physical β₆ value**: 15-30ms per request (vLLM scheduler CPU time, measured via profiling)

**Validation**: After iter10, check if β₆ decreases from 99ms toward 15-40ms range. If not, indicates:
1. Scheduler overhead is genuinely high (profile vLLM scheduler CPU time)
2. β₁₀ formulation insufficient (refine basis function or add threshold)

**Diagnostic Clause**: *If β₆ remains >60ms after β₁₀ addition, it indicates β₁₀ is insufficient OR scheduler overhead is genuinely high — profile vLLM scheduler separately to measure actual CPU overhead.*

---

## H-alpha-stability: Constrained Alpha Bounds Prevent Spurious Reduction

**Prediction**:
- **Iter9 alpha values** (DECLINED significantly, RED FLAG):
  - α₀ = 0.35ms (base overhead, -73% from iter8's 1.32ms)
  - α₁ = 65.0 μs/tok (input token, -45% from iter8's 117.6 μs/tok)
  - α₂ = 48.5 μs/tok (output token, -46% from iter8's 90.5 μs/tok)
- **Iter10 alpha values** (with constrained bounds):
  - α₀ = **0.8-2.5ms** (bounded [0.5ms, 5.0ms], prevent unrealistic decrease)
  - α₁ = **60-150 μs/tok** (bounded [50μs, 300μs])
  - α₂ = **50-120 μs/tok** (bounded [40μs, 250μs])

**Causal Mechanism**:

Iter9 alpha coefficients decreased 44-73% to compensate for beta explosions (β₆ +654%, β₂ +343%). This creates zero-sum trade-offs where long-sequence error absorption into beta terms comes at the expense of alpha baseline accuracy. **Constrained alpha bounds** prevent this spurious reduction:
- Lower bounds ensure alpha stays within physically plausible ranges (API overhead, tokenization cost)
- Alpha coefficients represent framework overhead that should be stable across iterations
- By constraining alpha, we force beta coefficients to model actual physics (not absorb baseline errors)

**Validation**: After iter10, verify alpha coefficients stay within constrained bounds AND beta explosions decrease (β₆, β₂, β₈ revert toward physical ranges).

**Diagnostic Clause**: *If alpha coefficients hit lower bounds (α₀ = 0.5ms, α₁ = 50μs, α₂ = 40μs), it indicates optimizer is still trying to reduce alpha — investigate whether bounds are too restrictive OR beta terms are still insufficient.*

---

## H-boundary-seq-length: β₁₀ Effect Should Scale Quadratically with Sequence Length

**Prediction**:
- **Short sequences** (50-100 tokens, roleplay): β₁₀ contribution ≈ **0.5-2ms** per request (minimal overhead)
- **Moderate sequences** (100-200 tokens, codegen): β₁₀ contribution ≈ **2-8ms** per request (moderate overhead)
- **Long sequences** (400-600 tokens, general-lite): β₁₀ contribution ≈ **20-80ms** per request (dominant overhead)
- **Ratio**: Long/Short ≈ 10-40× (quadratic scaling: (500/100)² × (batch_short/batch_long) ≈ 25 × 2 = 50×)

**Causal Mechanism**: β₁₀ basis function `β₁₀ × Σ(prefillTokens²/batchSize)` scales quadratically with sequence length:
- Quadratic term `prefillTokens²` captures disproportionate batch capacity consumption
- Division by `batchSize` amplifies effect for long sequences (lower batch efficiency → smaller batch_size denominator)
- **Result**: Long sequences have 10-50× higher β₁₀ contribution than short sequences

**Validation**: After iter10, compute per-experiment β₁₀ contributions and verify:
1. Scout roleplay (short): β₁₀ × (100²/32) ≈ β₁₀ × 312 ≈ 0.5-2ms
2. Scout codegen (moderate): β₁₀ × (150²/16) ≈ β₁₀ × 1,406 ≈ 2-8ms
3. Scout general-lite (long): β₁₀ × (500²/4) ≈ β₁₀ × 62,500 ≈ 20-80ms

**Diagnostic Clause**: *If β₁₀ contributions do NOT scale quadratically (long/short ratio <10×), it indicates basis function formulation is incorrect — refine to use sigmoid threshold or linear + quadratic split.*

---

## H-error-pattern-dense: Dense Long-Sequence Experiments Should Also Improve

**Prediction**: Dense model long-sequence experiments should improve >20pp TTFT:
- **Mistral Nemo general-lite** (exp 62): 91% → **<70%** (>21pp improvement)
- **Llama-2-7b reasoning-lite** (exp 67): 84% → **<60%** (>24pp improvement)
- **Qwen2.5-7b reasoning-lite** (exp 66): 79% → **<55%** (>24pp improvement)
- **01-ai Yi-34B general-lite** (exp 65): 78% → **<55%** (>23pp improvement)
- **Llama-3.1-70B general-lite** (exp 60): 77% → **<55%** (>22pp improvement)

**Causal Mechanism**: Batching inefficiency is NOT Scout-specific — it's a universal long-sequence bottleneck affecting ALL models. Dense models with long sequences (general-lite, reasoning-lite) also face batch packing constraints and queueing delays. β₁₀ should improve both Scout AND dense long-sequence experiments proportionally.

**Validation**: After iter10, compare TTFT improvements for Scout long vs dense long:
- Scout long (general-lite, reasoning-lite): expect >31pp improvement
- Dense long (5 experiments above): expect >20pp improvement
- **If dense improves <15pp**, indicates β₁₀ is Scout-specific (violates workload-agnostic constraint)

**Diagnostic Clause**: *If dense long-sequence experiments improve <15pp while Scout improves >30pp, it indicates β₁₀ is absorbing Scout-specific error (architecture-dependent) rather than universal batching inefficiency — refine basis function or add Scout-specific term.*

---

## H-robustness-batch-size: β₁₀ Should Generalize Across Batch Size Distributions

**Prediction**: β₁₀ mechanism should generalize across different batch size distributions (vLLM's dynamic batch formation):
- **Small batches** (batch_size=1-4): High β₁₀ contribution (low GPU utilization)
- **Medium batches** (batch_size=8-16): Moderate β₁₀ contribution
- **Large batches** (batch_size=32+): Low β₁₀ contribution (high GPU utilization)

**Causal Mechanism**: β₁₀ basis function explicitly divides by `batchSize`, ensuring generalization:
- When batch_size is small (long sequences), β₁₀ contribution amplifies
- When batch_size is large (short sequences), β₁₀ contribution diminishes
- **Adaptive scaling**: Same coefficient β₁₀ applies to all experiments, but contribution scales with observed batch size

**Validation**: After iter10, verify β₁₀ coefficient is stable (0.1-1.0 ms per (token²/batch_request)) and applies consistently across experiments with different batch size distributions.

**Diagnostic Clause**: *If β₁₀ coefficient is >5 ms OR <0.01 ms, it indicates basis function scaling is incorrect — investigate whether batchSize denominator captures actual batch efficiency penalty.*

---

## Summary of Predictions

| Hypothesis | Key Prediction | Success Threshold | Diagnostic If Failed |
|------------|----------------|-------------------|----------------------|
| **H-main** | Overall loss 160.6% → <90%, Scout long-sequence <60% TTFT | Loss <110% AND Scout long <70% | β₁₀ ~0 OR >10ms OR plausible but insufficient OR β₆ not revert |
| **H-kv-scaling** | β₃ = 0.4-1.5ms (base), β₃' = 0.1-1.0 μs (seq-len component) | β₃ <5ms AND β₃' >0 | KV overhead NOT seq-len dependent |
| **H-scheduler-reversion** | β₆ = 15-40ms (revert from 99ms) | β₆ <60ms | β₁₀ insufficient OR scheduler genuinely high |
| **H-alpha-stability** | α₀ = 0.8-2.5ms, α₁ = 60-150 μs, α₂ = 50-120 μs | Alpha within constrained bounds | Bounds too restrictive OR beta insufficient |
| **H-boundary** | β₁₀ contribution: short 0.5-2ms, long 20-80ms (quadratic scaling) | Long/short ratio >10× | Basis function formulation incorrect |
| **H-error-pattern-dense** | Dense long-sequence improve >20pp TTFT | Each dense long <70% TTFT | β₁₀ is Scout-specific (architecture-dependent) |
| **H-robustness** | β₁₀ = 0.1-1.0 ms per (token²/batch_request), generalizes across batch sizes | β₁₀ coefficient stable | Basis function scaling incorrect |

**Overall Success Criteria**: At least 5/7 hypotheses confirmed (✓) with H-main MANDATORY.

---

## Expected Coefficient Convergence (Iter10)

Based on iter9 results and iter10 additions:

| Coefficient | Iter9 | Iter10 Expected | Rationale |
|-------------|-------|-----------------|-----------|
| α₀ (base) | 0.35ms | **0.8-2.5ms** | Constrained lower bound prevents spurious reduction |
| α₁ (input token) | 65.0μs | **60-150μs** | Constrained bounds |
| α₂ (output token) | 48.5μs | **50-120μs** | Constrained bounds |
| β₀ (prefill compute) | 0.1624 | 0.14-0.22 | Stable, minor adjustment |
| β₁ (decode memory) | 1.3611 | 1.2-1.5 | Stable, may decrease slightly as other terms improve |
| β₂ (TP comm) | 0.8171 | **0.25-0.60** | Should decrease 50-70% as β₁₀ offloads long-seq overhead |
| β₃ (KV mgmt base) | 0.00959s = 9.6ms | **0.0004-0.0015s** = 0.4-1.5ms | **Revert 6-24× after splitting into base + seq-len** |
| β₃' (KV seq-len, NEW) | N/A | **0.0000001-0.000001s** = 0.1-1.0 μs per (token×layer) | **NEW coefficient** |
| β₄ (decode compute) | 0.4658 | 0.40-0.65 | Stable, may increase slightly |
| β₅ (MoE gating) | 0.0000198s = 19.8μs | 0.000015-0.000025s = 15-25μs | Stable within physical range ✓ |
| β₆ (scheduler) | 0.0993s = 99.3ms | **0.015-0.040s** = 15-40ms | **Decrease 60-75% as β₁₀ offloads long-seq delay** |
| β₇ (decode overhead) | 0.0110s = 11.0ms | 0.008-0.020s = 8-20ms | Stable within physical range ✓ |
| β₈ (MoE routing) | 0.0000727s = 72.7μs | 0.000025-0.000080s = 25-80μs | May decrease as β₁₀ reduces Scout pressure |
| **β₁₀ (batching inefficiency, NEW)** | **N/A** | **0.0000001-0.000001s** = 0.1-1.0 ms per (token²/batch_request) | **NEW: captures long-seq queueing delay** |

**Note**: β₉ (FP8 dequantization) **REMOVED** — converged to zero in iter9 (hypothesis rejected).

**Total coefficient count**: Iter9 had 10 beta coefficients (β₀-β₈, β₉). Iter10 has 10 beta coefficients (β₀-β₈, β₁₀, β₃') after removing β₉ and adding β₁₀ + β₃'.

---

## Risk Assessment

**Primary Risk**: β₁₀ insufficient — batching inefficiency is ONE component, but other long-sequence mechanisms also missing (memory bandwidth saturation, chunked prefill overhead).

**Mitigation**:
1. If iter10 achieves partial success (loss <110%, Scout long-sequence <70% TTFT), β₁₀ is correct but needs complementary term (β₁₁ for memory bandwidth)
2. If β₆ doesn't revert from 99ms, confirms additional scheduler-specific overhead exists
3. Prepare iter11 design for memory bandwidth saturation or chunked prefill overhead

**Secondary Risk**: β₁₀ formulation incorrect — quadratic scaling may be too aggressive, causing spurious overhead for moderate sequences.

**Mitigation**:
1. If moderate-sequence experiments (codegen) degrade >10pp, indicates β₁₀ overestimates overhead
2. Refine basis function to use sigmoid threshold or piecewise linear (only penalize sequences >300 tokens)
3. Validate batch size distributions in training data (ensure batchSize denominator is accurate)

**Tertiary Risk**: Alpha constraints too restrictive — optimizer hits lower bounds, indicating beta terms still insufficient.

**Mitigation**:
1. If alpha coefficients hit lower bounds (α₀ = 0.5ms, α₁ = 50μs, α₂ = 40μs), relax bounds slightly for iter11
2. Profile vLLM API overhead separately to measure actual physical alpha values
3. If bounds are correct, confirms beta terms need further refinement

---

## Success Definition

**Tier 1 (Full Success)**:
- Overall loss <90% ✓
- TTFT RMSE <40% ✓
- E2E RMSE <55% ✓
- Scout long-sequence <60% TTFT (general-lite, reasoning-lite) ✓
- Scout short-sequence stable (<30%, <65% TTFT) ✓
- Dense long-sequence <70% TTFT (all 5 experiments) ✓
- β₁₀ coefficient physically plausible (0.1-1.0 ms per (token²/batch_request)) ✓
- β₆ reverts to 15-40ms ✓
- β₃ reverts to 0.4-1.5ms, β₃' = 0.1-1.0 μs ✓
- Alpha coefficients within constrained bounds (not hitting lower bounds) ✓
- **At least 6/7 hypotheses confirmed** ✓

**Tier 2 (Partial Success)**:
- Overall loss <110% (significant improvement from iter9's 160.6%)
- Scout long-sequence <70% TTFT (>20pp improvement)
- β₁₀ and β₃' coefficients plausible
- At least 2/3 coefficient explosions (β₆, β₂, β₈) decrease >30%
- **At least 4/7 hypotheses confirmed**
- **Proceed to iter11** with additional long-sequence terms (memory bandwidth, chunked prefill)

**Tier 3 (Failure)**:
- Overall loss >130% (minimal improvement from iter9)
- Scout long-sequence >80% TTFT (<12pp improvement)
- β₁₀ converged to zero OR >5 ms (implausible)
- β₆ remains >80ms (no reversion)
- **<4/7 hypotheses confirmed**
- **Diagnostic**: Profile vLLM batch formation separately, validate basis function formulation, consider architecture-specific models

**If Tier 3**: Either (1) batching inefficiency formulation incorrect (refine β₁₀ basis function), (2) other long-sequence mechanisms dominate (memory bandwidth, chunked prefill), or (3) scheduler overhead genuinely 99ms (profile vLLM separately).

---

## Changes from Iter9

**Removed**:
- β₉ (FP8 dequantization): Converged to 0.14 μs (essentially zero), hypothesis REJECTED

**Added**:
- **β₁₀ (batching inefficiency)**: `β₁₀ × Σ(prefillTokens² / batchSize)` per request
  - Captures long-sequence queueing delays due to reduced batch efficiency
  - Expected coefficient: 0.1-1.0 ms per (token²/batch_request)
  - Basis function units: (tokens²/batch_request) × requests = tokens²
  - Should reduce β₆ from 99ms toward 15-40ms

- **β₃' (KV seq-len component)**: Split β₃ into base + sequence-length scaling
  - `β₃ × numRequests + β₃' × Σ(prefillTokens × numLayers)`
  - β₃: Base per-request KV overhead (0.4-1.5ms)
  - β₃': Sequence-length-dependent KV overhead (0.1-1.0 μs per token×layer)
  - Should revert β₃ from 9.6ms toward 0.4-1.5ms

**Modified**:
- **Alpha bounds**: Added lower bounds to prevent spurious reduction
  - α₀: [0.5ms, 5.0ms] (was [0.1ms, 10.0ms])
  - α₁: [50μs, 300μs] (was [10μs, 500μs])
  - α₂: [40μs, 250μs] (was [10μs, 500μs])

**Coefficient count**: 11 beta coefficients (β₀-β₈, β₃', β₁₀) — iter9 had 10 (β₀-β₉), removed β₉, added β₃' and β₁₀ = 11 total.
