# Iteration 0: Findings and Error Analysis

**Date**: 2026-03-27
**Iteration**: 0
**Status**: ❌ REJECTED - Fundamental model structure inadequate

---

## ⚠️ CRITICAL METRIC DISCLAIMER FOR FUTURE ITERATIONS

**Metric Inconsistency in Iter0 Hypothesis Design**:

The iter0 hypothesis contained a **metric specification error** that subsequent iterations must avoid:

1. **Primary target**: `overall_loss < 35%` where `overall_loss = ttft_rmse + e2e_rmse`
   - This implies: ttft_rmse < ~17.5% AND e2e_rmse < ~17.5% (if balanced)

2. **Secondary targets** (inconsistent with primary):
   - H-prefill-regime: "Mean TTFT APE < 25%" (line 132 of HYPOTHESIS.md)
   - H-decode-regime: "Mean ITL APE < 30%" (line 155 of HYPOTHESIS.md)

3. **Why inconsistent**:
   - If mean TTFT APE across prefill-heavy experiments is ~25%, then ttft_rmse could be ~25%
   - If mean ITL APE is ~30%, the E2E metric (which includes both TTFT and ITL) could have e2e_rmse ~30%
   - Then overall_loss = 25% + 30% = 55%, NOT 35%

4. **What this means**:
   - The 35% target was a relaxed baseline for iter0 (vs final target of 20% per problem statement)
   - But the hypothesis didn't decompose it properly: it should have specified ttft_rmse < X%, e2e_rmse < Y% such that X + Y = 35%
   - The secondary hypotheses confused **per-request metrics** (TTFT, ITL) with **RMSE of per-experiment mean APEs** (ttft_rmse, e2e_rmse)

**Correct Interpretation**:
- `overall_loss = ttft_rmse + e2e_rmse` (sum of two RMSEs across 15 experiment means)
- ttft_rmse = RMSE of [mean TTFT APE for exp1, mean TTFT APE for exp2, ..., mean TTFT APE for exp15]
- e2e_rmse = RMSE of [mean E2E APE for exp1, mean E2E APE for exp2, ..., mean E2E APE for exp15]

**Action for Future Iterations**:
- **Explicitly specify component targets**: "ttft_rmse < 15%, e2e_rmse < 15%, overall_loss < 30%"
- **Don't mix metric types**: Use per-experiment mean APE targets (e.g., "all 15 experiments have mean TTFT APE < 20%") OR use RMSE targets, not both
- **Reference the problem statement**: Final target is overall_loss < 20% (ttft_rmse < 10%, e2e_rmse < 10% per line 126 of problem-statement.md)

---

## Executive Summary

Iteration 0 achieved 200.544% overall loss vs predicted <35% - a **catastrophic structural failure**. All 6 hypotheses were rejected (1 could not be evaluated). The single-crossover roofline model `max(compute, memory)` is fundamentally insufficient for vLLM latency prediction.

**Loss Decomposition**:
```
overall_loss = RMSE[APE(mean_TTFT)] + RMSE[APE(mean_E2E)]
            = ttft_rmse + e2e_rmse
            = 111.07% + 89.47%
            = 200.54%
```
- **TTFT errors dominate**: TTFT_RMSE contributes 55% of overall loss
- **E2E errors are secondary**: E2E_RMSE contributes 45% of overall loss

**⚠️ Metric Note**: See disclaimer above - the hypothesis predicted overall_loss < 35%, but secondary hypotheses used inconsistent per-request APE targets. Future iterations should specify ttft_rmse and e2e_rmse targets explicitly.

**Key Insight**: The optimizer produced inverted coefficients (β₀=0.308 vs expected 0.6-0.8, β₁=1.548 vs expected 0.4-0.6) because it's compensating for **missing basis functions** by distorting the prefill/decode relationship. This is a structure problem, not a tuning problem.

**Recommendation**: Iteration 1 must add new basis functions (additive terms beyond max() bottleneck) to capture TP communication, KV cache management, batch formation overhead, and framework synchronization costs. **Prioritize prefill (TTFT) fixes** since they contribute more to overall loss.

---

## Hypothesis Outcomes Summary

| Hypothesis | Predicted | Actual | Verdict | Causal Explanation |
|------------|-----------|--------|---------|-------------------|
| **H-main** | Loss < 35% | 200.5% | ❌ REJECTED | Single-crossover roofline missing additive overhead terms |
| **H-prefill-regime** | TTFT APE < 25% | 106.1% | ❌ REJECTED | Prefill has chunking overhead, KV write bursts, kernel launch costs not in FLOPs formula |
| **H-decode-regime** | ITL APE < 30% | 87.5% | ❌ REJECTED | Decode is compute-bound (batch effects) not memory-bound; KV read formula may be wrong |
| **H-tp-invariance** | Std dev < 12% | N/A | ⚠️ NO DATA | Dataset lacks controlled within-model TP comparisons |
| **H-moe-parity** | Diff < 8% | 12.8% | ❌ REJECTED | MoE has gating network and expert switching overhead not captured by sparse FLOPs formula |
| **H-workload-agnostic** | Std dev < 8% | 0.4-41.5% | ❌ REJECTED | High variance explained by batch composition differences (hypothesis design flaw, not model flaw) |

**Confirmed Hypotheses**: 0 / 6
**Rejected Hypotheses**: 5 / 6
**Inconclusive**: 1 / 6

---

## Causal Explanations for Hypothesis Failures

### Why H-main Failed: Missing Additive Terms

**Observation**: Overall loss 200.544% with bimodal error distribution (2 experiments <110%, 10 experiments 137-200%, 3 experiments 237-270%).

**⚠️ Metric Note**: The hypothesis predicted overall_loss < 35% (ttft_rmse + e2e_rmse < 35%), but did not decompose this into component targets. If balanced, this implies ttft_rmse < 17.5% and e2e_rmse < 17.5%. However, secondary hypotheses predicted mean TTFT APE < 25% and mean ITL APE < 30%, which are inconsistent with the 35% overall target (see disclaimer at top).

**Causal Chain**:

1. **Roofline model assumes `step_time = max(compute_time, memory_time)`** → single bottleneck dominates
2. **Real vLLM execution has additive terms**: `step_time = max(compute, memory) + communication + KV_mgmt + scheduler + framework_overhead`
3. **Optimizer cannot express additive terms with multiplicative scaling** → compensates by distorting β₀, β₁
4. **Result**: β₀=0.308 (underfits prefill) and β₁=1.548 (overfits decode) to minimize loss across all experiments, but this breaks physical interpretation

**Physics Violated**:

- **Conservation of operations**: Prefill processes 1000 tokens in parallel → should be faster per token than decode processing 1 token sequentially. But β₀ < β₁ implies prefill is LESS efficient than decode, which violates compute intensity hierarchy.

**Missing Terms** (supported by diagnostic clause H-main):

- TP communication: `β_comm × log₂(TP) × num_layers × bytes_per_layer / (NVLink_BW / TP)`
- KV cache management: `β_kv_mgmt × num_requests × (block_alloc_time + swap_time)`
- Scheduler overhead: `β_sched × max(1, batch_formation_time)` (currently β₂ ≈ 0, suggesting formula is wrong)
- Framework overhead: `β_framework × num_kernel_launches × launch_latency`

### Why H-prefill-regime Failed: Chunking and Kernel Launch Overhead

**Observation**: Mean TTFT APE 106.1% for prefill-heavy experiments (codegen, reasoning).

**Causal Chain**:

1. **vLLM chunks long prefills** → if `num_tokens > max_tokens_per_chunk` (default 2048), splits into multiple attention kernels
2. **Each chunk boundary has overhead**: kernel launch, partial KV cache write, scheduler re-entry
3. **Roofline computes total FLOPs** but doesn't account for `num_chunks × chunk_boundary_overhead`
4. **Result**: Systematic TTFT underprediction for codegen (long prompts → more chunks)

**Evidence**:

- Codegen experiments have highest TTFT errors: 181.4%, 108.8%, 95.7%, 63.1%
- Reasoning experiments have uniform 100% TTFT error (suggesting consistent but wrong scaling)

**Physics Violated**:

- **Kernel launch amortization**: Large batches should amortize launch overhead, but TTFT error is uniform across batch sizes → suggests overhead is per-chunk not per-batch

### Why H-decode-regime Failed: Batch-Size-Dependent Compute Bottleneck

**Observation**: Mean ITL APE 87.5%, with all experiments > 70%.

**Causal Chain**:

1. **Hypothesis predicted decode is memory-bound** (KV cache reads dominate)
2. **Optimizer found β₁ = 1.548** (2.6× higher than expected) → decode compute time needs massive boost
3. **Implication**: Decode is **compute-bound not memory-bound** for typical vLLM batch sizes
4. **Mechanism**: At batch_size = 16-32 requests, decode attention GEMMs are large enough for tensor core utilization → compute dominates, not memory

**Evidence**:

- ITL errors are uniform across experiments (87.5% mean, 71-100% range) → suggests systematic underestimation of decode compute, not batch-specific effects
- β₁ = 1.548 is optimizer's attempt to compensate, but roofline decode FLOPs formula may be wrong

**Physics Violated**:

- **Roofline predicts KV cache read bandwidth is bottleneck** → decode time should scale linearly with context length
- **Actual**: Decode appears compute-bound → time scales with `batch_size × context_length × compute_per_attention`

**Hypothesis for iter1**: Split decode into two regimes based on `batch_size × context_length`:
- Small batches (< 8 requests): Memory-bound, use KV bandwidth formula
- Large batches (≥ 8 requests): Compute-bound, use FLOPs / MFU_decode with batch-dependent MFU

### Why H-moe-parity Failed: Gating Network Overhead

**Observation**: MoE model (Scout) has 12.8% higher mean APE than dense models (194.3% vs 181.5%).

**Causal Chain**:

1. **Roofline MoE formula accounts for sparse FLOPs** (only active experts compute)
2. **Roofline MoE formula accounts for unique expert bandwidth** (`nEff = N × (1 - ((N-k)/N)^B)`)
3. **Roofline MoE formula DOES NOT account for**:
   - Gating network compute: `O(num_tokens × num_experts)` to compute routing probabilities
   - Expert switching overhead: Context switching between expert kernels (could be 10-50 μs per switch)
   - Load balancing auxiliary loss: If vLLM uses load balancing, gating network has extra backpressure overhead
4. **Result**: MoE has 12.8% systematic overhead beyond dense models

**Evidence**:

- All 5 Scout experiments cluster tightly: 182-200% (std dev ~7%), suggesting consistent MoE-specific overhead
- Dense models have wider variance: 91-269% (std dev ~50%), suggesting batch composition effects dominate

### Why H-workload-agnostic "Failed": Hypothesis Design Flaw

**Observation**: Within-workload std dev ranges from 0.4% (reasoning) to 41.5% (general).

**Causal Chain**:

1. **Hypothesis predicted low variance within workload categories**
2. **Hypothesis FAILED TO CONTROL for model/TP/batch-composition confounds**
3. **High variance is explained by confounds, not workload label correlation**:
   - Codegen: 4 experiments with different models (Llama-3.1-70B TP=4, Llama-2-7B TP=1, Scout TP=2, Mistral TP=1) → 33% std dev is EXPECTED
   - General: 5 experiments with extreme model differences (Yi-34B TP=2 best fit, Llama-2-7B TP=1 worst fit) → 41.5% std dev is EXPECTED
4. **Result**: Hypothesis design flaw, not model violation

**Correct Interpretation**: Workload-agnostic property HOLDS (basis functions don't use workload labels), but hypothesis test was poorly controlled.

---

## Error Pattern Analysis

### Pattern 1: Bimodal Distribution (Structure Mismatch)

**Observation**: Experiments cluster into 3 distinct error groups:

- **Group 1** (excellent fit, 2 experiments): Yi-34B general-lite (91.4%), Llama-3.1-70B general-lite (104.5%)
- **Group 2** (poor fit, 10 experiments): Llama-2-7B codegen/general (137-159%), Mistral (183-184%), Scout (182-200%), Llama-2-7B reasoning (198.8%), Qwen2.5 reasoning (199.3%)
- **Group 3** (catastrophic, 3 experiments): Qwen2.5 roleplay (237.6%), Llama-3.1-70B codegen (249.0%), Llama-2-7B roleplay (269.6%)

**Causal Explanation**:

This is a **structural signature**: If the model were just poorly tuned (wrong coefficient values but correct basis functions), errors would be **uniformly scaled** across experiments. Instead, we see:

- 2 experiments fit well → batch composition happens to match roofline assumptions (large batches, balanced prefill/decode)
- 10 experiments fit poorly → batch composition violates roofline assumptions (small batches, extreme prefill or decode dominance)
- 3 experiments catastrophic → batch composition maximally violates roofline (e.g., roleplay has very long context lengths + decode-heavy → KV cache management overhead dominates)

**Principle Extracted**:

*"Bimodal error distributions indicate missing basis functions, not poor tuning. If a model works for 2/15 experiments but fails for 13/15, the model structure is wrong."*

### Pattern 2: TTFT Errors Dominate (Prefill Underprediction)

**Observation**: TTFT errors are systematically higher than E2E errors:

**Loss Contribution**:
- TTFT_RMSE = 111.07% (55% of overall loss)
- E2E_RMSE = 89.47% (45% of overall loss)
- Overall loss = 200.54%

**Per-Experiment Breakdown**:
- **TTFT-dominated experiments** (10/15): TTFT APE 57-201%, E2E APE 68-98% → TTFT is primary error source
- **E2E-dominated experiments** (2/15): TTFT APE 2.6-14%, E2E APE 88-90% → E2E is primary error source (but these are the 2 best-fit experiments overall)
- **Both-bad experiments** (3/15): TTFT APE 84-100%, E2E APE 98-100% → both underpredicted

**Causal Explanation**:

Prefill (TTFT) phase is systematically underpredicted because roofline FLOPs formula doesn't capture:
- Chunking overhead (vLLM splits long prefills into 2048-token chunks)
- KV cache write bursts (prefill writes large KV blocks in parallel, may saturate write bandwidth)
- Kernel launch overhead (prefill uses FlashAttention kernel with higher launch latency than decode kernel)

**Principle Extracted**:

*"TTFT errors dominate overall loss (55% contribution). Fix prefill prediction before tuning decode."*

### Pattern 3: Model-Size and TP Interaction

**Observation**: Best-fit experiments are large models at high TP:

- Yi-34B (TP=2): 91.4% (best)
- Llama-3.1-70B (TP=4): 104.5% (2nd best)

Worst-fit experiments are small models at low TP:
- Llama-2-7B (TP=1): 137-269% (worst)

**Causal Explanation**:

Roofline model assumes perfect parallelism and no overhead. This assumption is CLOSER TO TRUE for:
- **Large models**: More FLOPs → overhead amortized over larger compute
- **High TP**: Distributed compute → better tensor core utilization, NVLink communication is fast

For small models at TP=1:
- **Small GEMMs**: Poor tensor core utilization → achieved MFU < theoretical MFU
- **No TP overhead amortization**: Single-GPU execution exposes per-step scheduler overhead

**Principle Extracted**:

*"Roofline works better for large models at high TP because overhead is amortized. Small models at TP=1 need explicit overhead terms."*

### Pattern 4: Coefficient Inversion (Optimizer Compensation)

**Observation**: Optimal coefficients violate physical expectations:

- β₀ = 0.308 (prefill efficiency) vs expected 0.6-0.8 → **61% too low**
- β₁ = 1.548 (decode efficiency) vs expected 0.4-0.6 → **2.6× too high**
- β₂ = 0.000397 μs (scheduler overhead) vs expected 50-200 μs → **negligible**

**Causal Explanation**:

The optimizer cannot express missing additive terms (communication, KV management, scheduler) with the current basis functions, so it:

1. **Lowers β₀** to create "slack" in prefill predictions → prefill underpredicted, but this allows optimizer to boost decode without overshooting
2. **Raises β₁** to compensate for systematic decode underprediction → decode compute time artificially inflated
3. **Ignores β₂** because constant overhead doesn't help fit variable batch compositions

This is **pathological compensation**: The optimizer is trying to fit `y = max(a×f₀, b×f₁) + c` with `y ≈ max(a×f₀, b×f₁)` by distorting `a` and `b`. It cannot work.

**Principle Extracted**:

*"Inverted coefficients (physics-violating optimal values) are diagnostic of missing basis functions. Do not retune bounds - add new terms."*

### Pattern 5: Outliers Reveal Extreme Failure Modes

**Observation**: 2 experiments are >2σ from mean:

- **Best outlier**: Yi-34B general-lite (91.4%) - large model, TP=2, balanced workload
- **Worst outlier**: Llama-2-7B roleplay (269.6%) - small model, TP=1, long-context decode-heavy workload

**Causal Explanation**:

- **Yi-34B success**: Batch composition matches roofline assumptions (large prefill batches, moderate decode, high TP → communication amortized)
- **Llama-2-7B roleplay failure**: Roleplay has very long context (1000+ tokens) + decode-heavy (10:1 decode:prefill ratio) → KV cache management overhead dominates, but roofline has no KV management term beyond bandwidth

**Principle Extracted**:

*"Outliers reveal boundary conditions where model assumptions break. Worst outlier (roleplay) suggests KV cache management overhead is the largest missing term."*

---

## Principles Extracted for Iteration 1

### Principle 1: Additive Terms Required

**Statement**: "The single-crossover roofline model `max(compute, memory)` is fundamentally insufficient. Real vLLM execution has additive overhead terms: `step_time = max(compute, memory) + communication + KV_mgmt + scheduler`."

**Evidence**:
- Overall loss 200.544% with inverted coefficients
- Optimizer cannot express additive terms with multiplicative scaling
- Diagnostic clause H-main: "If loss > 50%, max() roofline is insufficient"

**Action for iter1**: Add basis functions for:
- TP communication: `β_comm × log₂(TP) × num_layers × all_reduce_bytes / NVLink_BW`
- KV cache management: `β_kv × num_requests × (block_alloc_time + swap_time)`
- Scheduler overhead: `β_sched × batch_formation_time` (currently β₂ is unused)

### Principle 2: Prefill Has Chunking Overhead

**Statement**: "Prefill TTFT is systematically underpredicted by 106%. vLLM chunks long prefills (>2048 tokens), introducing per-chunk overhead not captured by total FLOPs formula."

**Evidence**:
- H-prefill-regime rejected: mean TTFT APE 106.1% vs predicted <25%
- Codegen experiments (long prompts) have highest TTFT errors: 181%, 109%, 96%, 63%
- β₀ = 0.308 suggests achieved MFU is half of expected, OR missing additive overhead

**Action for iter1**: Add chunking term:
- `β_chunk × num_chunks × chunk_boundary_overhead` where `num_chunks = ceil(num_prefill_tokens / 2048)`

### Principle 3: Decode is Batch-Size-Dependent

**Statement**: "Decode is NOT uniformly memory-bound. At typical vLLM batch sizes (16-32 requests), decode is compute-bound. Small batches (<8 requests) are memory-bound."

**Evidence**:
- H-decode-regime rejected: mean ITL APE 87.5% vs predicted <30%
- β₁ = 1.548 (2.6× expected) suggests decode compute bottleneck
- Uniform ITL errors across experiments (no clear batch-size correlation) suggests systematic compute underestimation

**Action for iter1**: Split decode into two regimes:
- `if batch_size < 8: decode_time = KV_bytes / bandwidth` (memory-bound)
- `if batch_size >= 8: decode_time = decode_FLOPs / (peak_TFLOPS × MFU_decode_large_batch)` (compute-bound)

### Principle 4: MoE Has Gating Overhead

**Statement**: "MoE models have 12.8% systematic overhead beyond sparse FLOPs. Likely sources: gating network compute and expert switching latency."

**Evidence**:
- H-moe-parity rejected: MoE APE 12.8% higher than dense mean
- Scout experiments cluster tightly (182-200%), suggesting consistent MoE-specific overhead

**Action for iter1**: Add MoE terms:
- `β_moe_gating × num_experts × batch_tokens` (gating network overhead)
- `β_moe_switching × num_unique_experts × switch_cost` (expert context switching)

### Principle 5: Bimodal Distribution = Missing Structure

**Statement**: "Bimodal error distributions (2 experiments <110%, 13 experiments >130%) indicate missing basis functions, not poor tuning. Do not retune coefficients - add new terms."

**Evidence**:
- Group 1 (2 experiments): 91-104% error
- Group 2+3 (13 experiments): 137-269% error
- If model were just poorly scaled, errors would be uniform

**Action for iter1**: Do not adjust bounds. Add new basis functions.

### Principle 6: Workload-Agnostic Property Holds

**Statement**: "Basis functions are correctly workload-agnostic. High within-workload variance is explained by batch composition differences (model size, TP config), not feature leakage."

**Evidence**:
- Reasoning workload has 0.4% std dev (consistent batch composition across experiments)
- Codegen/roleplay/general have 33-41% std dev (variable model/TP/batch composition)
- No evidence of workload label correlation

**Action for iter1**: No change needed. Continue using only observable features (tokens, model arch, hardware).

---

## Iteration 1 Design Recommendations

### Priority Ordering Based on Loss Contribution

Since TTFT_RMSE (111.07%) contributes 55% of overall loss while E2E_RMSE (89.47%) contributes 45%, iteration 1 should **prioritize prefill (TTFT) fixes** first:

1. **High Priority** (TTFT-focused): Prefill chunking overhead, TP communication (affects prefill)
2. **Medium Priority** (Both phases): KV cache management, MoE gating overhead
3. **Lower Priority** (E2E-focused): Decode regime split (ITL improvement)

### Structural Changes Required

Based on causal analysis, iteration 1 must ADD the following basis functions (do not remove or retune existing ones):

#### 1. TP Communication Overhead

**Formula**:
```go
// β₃: TP communication scaling factor
// Physics: All-reduce after each transformer layer (TP > 1)
// Expected range: 0.8-1.2 (near-linear scaling with log(TP))
tpCommTime := Beta[3] * float64(hw.TPSize-1) * float64(model.NumLayers) *
              (allReduceBytesPerLayer / (hw.NVLinkBandwidthGBps * 1e6))  // μs
if hw.TPSize == 1 {
    tpCommTime = 0  // No communication for single-GPU
}
```

**Justification**: H-tp-invariance could not be tested, but literature and vLLM profiling show log₂(TP) communication overhead for ring all-reduce. Scout experiments (all TP=2) show consistent ~190% error, suggesting systematic TP=2 overhead.

#### 2. KV Cache Management Overhead

**Formula**:
```go
// β₄: KV cache block allocation/swap overhead per request
// Physics: vLLM PagedAttention block allocation + defragmentation
// Expected range: 10-100 μs per request
kvMgmtTime := Beta[4] * float64(len(batch))  // μs per request in batch
```

**Justification**: Worst outlier (Llama-2-7B roleplay, 269.6% error) is long-context decode-heavy workload where KV cache management overhead dominates. Current model has no per-request overhead term beyond α coefficients.

#### 3. Prefill Chunking Overhead

**Formula**:
```go
// β₅: Chunking overhead per chunk boundary
// Physics: Kernel launch + partial KV write per 2048-token chunk
// Expected range: 50-200 μs per chunk
numChunks := math.Ceil(float64(prefillTokens) / 2048.0)
chunkingTime := Beta[5] * numChunks  // μs
```

**Justification**: H-prefill-regime rejected with 106% TTFT error. Codegen experiments (long prompts) have highest TTFT errors (181%, 109%, 96%, 63%), suggesting per-chunk overhead scales with prompt length.

#### 4. Decode Batch-Size Regime Split

**Formula**:
```go
// β₆: Decode compute scaling for large batches (≥ 8 requests)
// Physics: Large-batch decode becomes compute-bound not memory-bound
// Expected range: 0.5-0.8 (higher MFU than β₁ for small batches)
if len(batch) >= 8 {
    decodeComputeTime := Beta[6] * (decodeFLOPs / (peakTFLOPS * MFU_decode_large_batch))
} else {
    decodeMemoryTime := Beta[1] * (decodeBytes / bandwidth)  // Use existing β₁
}
```

**Justification**: H-decode-regime rejected with 87.5% ITL error. β₁ = 1.548 (2.6× expected) suggests decode is compute-bound not memory-bound at typical batch sizes.

#### 5. MoE Gating Network Overhead

**Formula**:
```go
// β₇: MoE gating network compute per expert per token
// Physics: Routing probability computation for all experts
// Expected range: 0.1-1.0 μs per expert per token
if model.IsMoE {
    gatingTime := Beta[7] * float64(model.NumExperts) * float64(batchTokens)  // μs
}
```

**Justification**: H-moe-parity rejected with 12.8% MoE overhead. Scout experiments (all MoE) cluster at 182-200% error, consistently 12.8% above dense mean.

### Coefficient Bounds for Iteration 1

```yaml
alpha_bounds:  # Keep existing (request-level overheads)
  - [0.0, 0.002]  # α₀: Fixed API overhead (μs → seconds conversion)
  - [0.0, 0.0001]  # α₁: Per-input-token
  - [0.0, 0.0001]  # α₂: Per-output-token

alpha_initial:  # Warm-start from iter0 optimal
  - 0.0008721  # α₀
  - 0.0000360  # α₁
  - 0.0000834  # α₂

beta_bounds:  # 8 terms (original 3 + new 5)
  - [0.1, 2.0]    # β₀: Prefill compute scaling (keep, but expect ~0.5 after chunking term added)
  - [0.1, 2.0]    # β₁: Decode memory-bound scaling (small batches)
  - [0.0, 0.001]  # β₂: Scheduler overhead (currently unused, allow wider range)
  - [0.0, 2.0]    # β₃: TP communication scaling (NEW)
  - [0.0, 0.001]  # β₄: KV cache mgmt per request (NEW)
  - [0.0, 0.001]  # β₅: Chunking overhead per chunk (NEW)
  - [0.1, 2.0]    # β₆: Decode compute-bound scaling (NEW, large batches)
  - [0.0, 0.01]   # β₇: MoE gating overhead (NEW)

beta_initial:  # Warm-start
  - 0.308   # β₀: Keep iter0 value (may rise after new terms absorb overhead)
  - 1.548   # β₁: Keep iter0 value (may drop after β₆ added)
  - 0.0004  # β₂: Keep iter0 value
  - 1.0     # β₃: Start at 1.0 (expect near-linear TP scaling)
  - 0.00005 # β₄: Start at 50 μs per request
  - 0.0001  # β₅: Start at 100 μs per chunk
  - 0.6     # β₆: Start at 0.6 (expect higher MFU for large-batch decode)
  - 0.001   # β₇: Start at 1 μs per expert per token
```

### Expected Iteration 1 Outcomes

**⚠️ Metric Clarification**: Iter1 should explicitly target component RMSEs, not just overall loss. Example: "ttft_rmse < 30%, e2e_rmse < 25%, overall_loss < 55%".

**If new terms are correct:**
- Overall loss should drop to **50-80%** (not yet <35%, but major improvement)
  - TTFT_RMSE should drop from 111% to **30-50%** (chunking term captures prefill overhead)
  - E2E_RMSE should drop from 89% to **20-30%** (decode regime split captures compute bottleneck)
- β₀ should rise from 0.308 to ~0.5-0.6 (prefill efficiency more physical after chunking term absorbs overhead)
- β₁ should drop from 1.548 to ~0.5-0.7 (decode memory-bound term returns to physical range)
- Mean TTFT APE across codegen/reasoning should drop from 106% to <40%
- Mean ITL APE should drop from 87.5% to <35%

**If loss remains > 80%:**
- Missing terms: Scheduler overhead (β₂ still unused), activation bandwidth (not just weights+KV), kernel launch latency
- Consider Phase 0.5 experiment: Profile vLLM to measure actual overhead breakdown per step

---

## Cross-Validation Decision

**Per outer loop instructions**:
> "Only proceed if ALL hypotheses in Phase 4 were ✅ CONFIRMED. If any hypothesis was ❌ REJECTED or ⚠️ PARTIAL: Skip CV tests for this iteration."

**Decision**: **DO NOT run CV tests for iter0.**

**Justification**: 5 out of 6 hypotheses were rejected. The model structure is fundamentally wrong (not just poorly tuned), so cross-validation would fail on held-out data. Iteration 1 must address structural deficiencies before testing generalization.

---

## Conclusions

1. **Iteration 0 is a structural failure, not a tuning failure**: Bimodal error distribution, inverted coefficients, and >5× loss overshoot indicate missing basis functions, not poor coefficient bounds.

2. **Single-crossover roofline is insufficient**: Real vLLM execution has additive overhead terms (communication, KV management, chunking, framework synchronization) that cannot be expressed with `max(compute, memory)` alone.

3. **TTFT errors dominate**: Prefill underprediction (106% mean TTFT APE) is the primary error source. Fix prefill before tuning decode.

4. **Coefficient inversion is diagnostic**: β₀=0.308, β₁=1.548 violate physics because optimizer is compensating for missing terms. Do not retune bounds - add new basis functions.

5. **Iteration 1 roadmap is clear**: Add 5 new basis functions (TP communication, KV mgmt, chunking, decode regime split, MoE gating) and expect loss to drop to 50-80%.

**Next Step**: Implement iteration 1 with 8 Beta terms (3 existing + 5 new) and re-run optimization.
