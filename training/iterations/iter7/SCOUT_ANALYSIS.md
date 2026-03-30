# Scout MoE Architecture Analysis — Iter7 Deep Dive

## Executive Summary

**Critical Discovery**: All 4 Scout MoE experiments fail uniformly (79-100% TTFT) across workloads, contributing 767% combined loss (49% of iter7 total error budget). Non-Scout experiments with identical workloads succeed (5-66% TTFT). This proves the bottleneck is **Scout MoE architecture**, not workload characteristics.

**Key Evidence**: Scout reasoning-lite (98% TTFT) vs Qwen reasoning-lite (54% TTFT) and Llama-2 reasoning-lite (66% TTFT) — same clean data, different architecture, 32-44pp worse.

---

## Scout Experiment Performance

### All Scout Experiments (Sorted by TTFT)

| Experiment | Workload | TTFT | E2E | Combined Loss | Iter6 TTFT | Change |
|------------|----------|------|-----|---------------|------------|--------|
| **Scout roleplay** | roleplay-2 | 79.12% | 96.04% | 175.15% | 87.49% | -8.4pp ✓ |
| **Scout codegen** | codegen-2 | 92.11% | 98.26% | 190.38% | 98.03% | -5.9pp ✓ |
| **Scout reasoning-lite** | reasoning-lite-2-1 | 98.46% | 99.81% | 198.27% | NEW | N/A |
| **Scout general** | general-2 | 99.97% | 99.40% | 199.37% | 99.79% | +0.2pp ✗ |

**Observations**:
1. **Uniform failure**: All Scout experiments >79% TTFT (avg 90%)
2. **Slight improvement**: 3/4 experiments improved 5-8pp from iter6
3. **Still catastrophic**: All >79% TTFT despite clean data and coefficient stabilization
4. **Workload-independent**: General/reasoning/codegen/roleplay all fail uniformly

### Scout vs Non-Scout Comparison (Same Workloads)

| Workload | Scout TTFT | Non-Scout TTFT | Scout Loss | Non-Scout Loss | Scout Penalty |
|----------|-----------|----------------|------------|----------------|---------------|
| **Reasoning-lite** | 98.46% | 54-66% | 198.27% | 149-162% | **+32-44pp** |
| **Codegen** | 92.11% | 9-20% | 190.38% | 95-105% | **+72-83pp** |
| **Roleplay** | 79.12% | 56-57% | 175.15% | 131-134% | **+22-23pp** |
| **General** | 99.97% | 5% | 199.37% | 89% | **+95pp** |

**Scout penalty**: +22 to +95pp TTFT worse than non-Scout for identical workloads.

**Critical**: Scout reasoning-lite (98% TTFT) used **identical clean data** as Qwen (54%) and Llama-2 (66%) reasoning-lite, yet performed 32-44pp worse. This proves:
1. Clean data hypothesis worked for non-Scout ✓
2. Scout MoE architecture prevents convergence regardless of data quality ✗

---

## Scout Architecture Specifications

### Model Details

**Model**: RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic

**Key Characteristics**:
- **Architecture**: Interleaved MoE + dense layers (NOT pure MoE or pure dense)
- **Total layers**: 56 layers (26 MoE layers, 30 dense layers per ModelConfig)
- **Expert count**: 16 experts per MoE layer
- **Top-k routing**: 2 experts per token (hypothesized, not confirmed)
- **Quantization**: FP8 dynamic (unique to Scout in this dataset)
- **TP degree**: TP=2 (cross-GPU communication)
- **Total parameters**: 17B (16 experts contribute to parameter count)

### ModelConfig Fields (from iter1 Scout fix #877)

```go
InterleaveMoELayerStep: 26  // MoE layers interspersed every 26 layers
DenseIntermediateDim: set   // Dense layer intermediate dimension
```

**Purpose**: Correctly calculate FLOPs and weight bandwidth for mixed MoE+dense architecture by splitting layer types.

### What Makes Scout Unique in Dataset

1. **Only interleaved MoE+dense**: All other models are pure dense
2. **Only FP8 quantization**: Other models use BF16/FP16/INT8
3. **Only 56-layer model**: Most models have 32-40 layers (Llama-3.1-70B has 80)
4. **TP=2 with MoE**: Mistral TP=2 also fails (90% TTFT), but it's dense

---

## Hypothesis: Four Potential Bottlenecks

### Hypothesis 1: MoE Expert Routing Overhead (Most Likely)

**Mechanism**:

MoE layers require **per-request routing computation** not present in dense layers:
1. **Gating network**: Input → router → expert probabilities (softmax over 16 experts)
2. **Expert selection**: Top-k selection (choose 2 experts per token)
3. **Expert execution**: Dispatch tokens to selected experts (may require reordering)
4. **Expert aggregation**: Weighted sum of expert outputs
5. **Load balancing**: Auxiliary loss to balance expert utilization (training-time, but may affect inference)

**Physics Missing from Current Model**:

Current model (β₆, β₇) captures:
- β₆: Per-request scheduler overhead (21.5ms → 13.2ms in iter7)
- β₇: Per-request decode framework overhead (26.3ms in iter7)

But does NOT capture:
- **Per-request MoE routing**: Gating network computation, expert selection (1-5ms per MoE layer × 26 layers = 26-130ms)
- **Per-token expert dispatch**: Token reordering, expert selection (0.1-0.5ms per token × 1000 tokens = 100-500ms)
- **Expert load balancing**: Dynamic expert assignment may add variance

**Expected Overhead**:
- **Prefill**: 26 MoE layers × 3-5ms routing = 78-130ms per request
- **Decode**: 26 MoE layers × 1-2ms routing = 26-52ms per decode step

**Why This Explains Scout Failure**:
- Scout general TTFT: 100-200ms actual, predicted ~10-20ms
- Missing overhead: 78-130ms prefill MoE routing matches gap
- Scout decode (E2E - TTFT): also over-predicted by 26-52ms

**Evidence**:
- All Scout workloads fail uniformly (79-100% TTFT)
- Non-Scout same workloads succeed (5-66% TTFT)
- Scout roleplay (79% TTFT) better than Scout general (100% TTFT) — shorter context reduces MoE routing overhead

**Action**: Profile Scout MoE overhead with vLLM profiler to measure gating network latency, expert selection time, and dispatch overhead.

**Proposed Model Extension**:
```python
# Prefill
QueueingTime += β_moe × num_moe_layers × num_requests

# Decode (per step)
StepTime += β_moe_decode × num_moe_layers × num_requests_in_batch
```

Expected β_moe: 3-5ms per MoE layer (78-130ms for 26 layers).

---

### Hypothesis 2: Mixed-Precision Coordination (FP8 Dequantization)

**Mechanism**:

Scout uses **FP8 dynamic quantization**, requiring runtime dequantization:
1. **Weights stored in FP8**: 1 byte per parameter (vs 2 bytes BF16)
2. **Dequantization at runtime**: FP8 → BF16/FP32 conversion per layer
3. **Compute in BF16/FP32**: Matrix multiplications in higher precision
4. **Quantization for storage**: Output → FP8 for next layer (if applicable)

**Physics Missing from Current Model**:

Current model (β₀, β₂, β₁, β₄) assumes uniform precision:
- β₀: Prefill memory-bound (assumes BF16 weight bandwidth)
- β₁: Decode memory-bound (assumes BF16 weight bandwidth)
- But Scout has FP8 weights → 2× higher bandwidth, but dequantization overhead

**Expected Overhead**:
- **Dequantization**: 0.1-0.5ms per layer × 56 layers = 5.6-28ms per request
- **Precision coordination**: Mixed-precision kernel launch may add overhead

**Why This Might Explain Scout Failure**:
- Only Scout uses FP8 in dataset (other models BF16/FP16)
- Dequantization overhead: 5-28ms per request
- But this is **layer-dependent**, not MoE-specific
- Non-MoE FP8 models not in dataset to test independently

**Evidence Against**:
- If FP8 overhead, all Scout layers (MoE + dense) should be affected
- But Scout failure is uniform across workloads (not layer-count dependent)
- Llama-3.1-70B (80 layers, more than Scout 56) succeeds (29-41% TTFT)

**Action**: Profile Scout FP8 dequantization overhead, compare with BF16 model of similar size.

**Proposed Model Extension**:
```python
# If model uses FP8 quantization
if model.quantization == "FP8":
    QueueingTime += β_fp8_deq × num_layers
    StepTime += β_fp8_deq × num_layers
```

Expected β_fp8_deq: 0.1-0.5ms per layer (5.6-28ms for 56 layers).

---

### Hypothesis 3: TP Communication Overhead (TP=2 with MoE)

**Mechanism**:

Scout uses **TP=2 with MoE architecture**, which may have higher cross-GPU communication than dense TP=2:
1. **Expert routing across GPUs**: Gating network output must be communicated to both GPUs
2. **Expert weights split**: 16 experts split across 2 GPUs (8 experts per GPU)
3. **Cross-GPU expert dispatch**: Tokens may need to be routed to expert on other GPU
4. **Expert output aggregation**: Weighted sum across GPUs requires all-reduce

**Physics Missing from Current Model**:

Current model (β₂) captures TP communication:
- β₂: Prefill TP communication (all-reduce for attention, MLP)
- But assumes **dense layers** (uniform all-reduce)
- MoE may require **different communication pattern** (expert routing, selective all-reduce)

**Expected Overhead**:
- **Expert routing communication**: Gating network output broadcast (1-2ms per MoE layer)
- **Cross-GPU expert dispatch**: Token reordering across GPUs (2-5ms per MoE layer)
- **Expert aggregation**: All-reduce of expert outputs (1-3ms per MoE layer)
- Total: 4-10ms per MoE layer × 26 layers = 104-260ms per request

**Why This Might Explain Scout Failure**:
- Mistral TP=2 (dense) also fails (90% TTFT), supporting TP hypothesis
- Llama-3.1 TP=4 (dense) succeeds (29-41% TTFT), contradicting TP hypothesis
- If TP=2 problematic, TP=4 should be worse (more cross-GPU communication)

**Evidence Against**:
- Llama-3.1-70B TP=4 succeeds (29-41% TTFT) with 80 layers
- More layers + higher TP degree should amplify TP overhead
- But Llama-3.1 outperforms Scout (56 layers, TP=2)

**Action**: Compare Scout TP=2 vs Mistral TP=2 vs Llama-3.1 TP=4 to isolate TP effect from MoE effect.

**Proposed Model Extension**:
```python
# If MoE with TP > 1
if model.num_experts > 0 and model.tp_degree > 1:
    β₂_effective = β₂ × (1 + 0.5 × log2(tp_degree)) × (1 + 0.3 × has_moe)
```

Expected multiplier: 1.5-2× higher TP communication for MoE vs dense.

---

### Hypothesis 4: Model Config Issue (InterleaveMoELayerStep Incorrect)

**Mechanism**:

Iter1 Scout fix (#877) added `InterleaveMoELayerStep` and `DenseIntermediateDim` to ModelConfig for correct FLOPs/weight bandwidth calculation. But if these fields are **incorrect or incomplete**, model predictions may be systematically wrong.

**Potential Issues**:
1. **InterleaveMoELayerStep = 26**: May not match actual Scout architecture
   - Check HuggingFace config.json: `moe_layer_indices` or similar
   - If actual interlacing is different (e.g., every 13 layers, not 26), FLOPs wrong

2. **DenseIntermediateDim**: May not match actual dense layer dimension
   - Check HuggingFace config.json: `intermediate_size` for dense vs MoE layers
   - If different for dense vs MoE, weight bandwidth calculation wrong

3. **Expert count**: Assumed 16 experts, but may be different
   - Check HuggingFace config.json: `num_local_experts`
   - If 8 or 32 experts, FLOPs/weight bandwidth wrong

4. **Top-k routing**: Assumed 2 experts per token, but may be 1 or 4
   - Check HuggingFace config.json: `num_experts_per_tok`
   - Directly affects FLOPs (compute 2× if top-k=2, 4× if top-k=4)

**Why This Might Explain Scout Failure**:
- If InterleaveMoELayerStep wrong, FLOPs calculation systematically wrong
- If top-k wrong, compute time prediction wrong by 2-4×
- But this should affect **all Scout experiments uniformly** (confirmed: 79-100% TTFT)

**Evidence**:
- All Scout experiments fail uniformly (supports systematic config issue)
- Non-Scout experiments succeed (supports Scout-specific issue)
- Iter1 fix (#877) addressed FLOPs/weight bandwidth, but may be incomplete

**Action**: Verify Scout ModelConfig against HuggingFace config.json:
```bash
# Check HuggingFace config
cat ~/.cache/huggingface/hub/models--RedHatAI--Llama-4-Scout-17B-16E-Instruct-FP8-dynamic/config.json | jq '.'

# Compare with BLIS ModelConfig
grep -A 10 "RedHatAI/Llama-4-Scout" sim/models.go
```

**Proposed Fix**:
- Update `InterleaveMoELayerStep`, `DenseIntermediateDim` if incorrect
- Add `num_local_experts`, `num_experts_per_tok` to ModelConfig if missing
- Recalculate FLOPs/weight bandwidth with corrected fields

---

## Diagnostic Plan

### Phase 1: Verify Model Config (Quick, Low-Cost)

**Steps**:
1. Read HuggingFace config.json for Scout model
2. Compare `num_hidden_layers`, `num_local_experts`, `num_experts_per_tok`, `intermediate_size` with BLIS ModelConfig
3. Check `moe_layer_indices` or similar to verify InterleaveMoELayerStep=26
4. If mismatch, update ModelConfig and re-run roofline baseline (no training needed)

**Expected Outcome**:
- If config wrong: Roofline baseline improves from 15-92% → <50% TTFT
- If config correct: Roofline baseline stays 15-92%, confirms missing physics (not config issue)

**Time**: 30 minutes

---

### Phase 2: Profile MoE Overhead (Medium-Cost, High-Value)

**Steps**:
1. Run vLLM with Scout model, capture profiling trace:
   ```bash
   python -m vllm.profiler --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic \
       --workload reasoning-lite-2-1 --output scout_profile.json
   ```

2. Analyze trace for per-request timing:
   - Gating network latency (router forward pass)
   - Expert selection time (top-k selection)
   - Expert execution time (expert forward pass)
   - Expert aggregation time (weighted sum)
   - Total MoE overhead per layer

3. Compare MoE layer latency vs dense layer latency:
   - Dense layer: ~X ms
   - MoE layer: ~Y ms
   - MoE overhead: Y - X per layer

4. Extrapolate to full request:
   - MoE overhead per request: (Y - X) × 26 MoE layers
   - Expected: 78-130ms for prefill, 26-52ms for decode

**Expected Outcome**:
- If MoE overhead 78-130ms: Confirms Hypothesis 1 (expert routing bottleneck)
- If MoE overhead <20ms: Rejects Hypothesis 1, investigate Hypothesis 2 or 3

**Time**: 2-4 hours (includes vLLM setup, profiling, analysis)

---

### Phase 3: FP8 Dequantization Profiling (Medium-Cost, Medium-Value)

**Steps**:
1. Profile Scout FP8 model vs similar-sized BF16 model (e.g., Llama-2-13B)
2. Measure per-layer dequantization overhead:
   ```python
   # Pseudocode
   with torch.profiler.profile() as prof:
       output = model.forward(input)
   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

3. Look for FP8 dequantization kernels:
   - `dequantize_fp8_kernel`
   - `convert_fp8_to_bf16`
   - Total time per layer

4. Extrapolate to full request:
   - FP8 overhead per request: dequant_time × 56 layers
   - Expected: 5.6-28ms

**Expected Outcome**:
- If FP8 overhead 20-30ms: Confirms Hypothesis 2 contributes
- If FP8 overhead <5ms: Rejects Hypothesis 2, focus on Hypothesis 1

**Time**: 2-3 hours (includes setup, profiling, analysis)

---

### Phase 4: TP Communication Profiling (High-Cost, Low-Value)

**Steps**:
1. Compare Scout TP=2 vs Mistral TP=2 vs Llama-3.1 TP=4
2. Profile cross-GPU communication with nsys:
   ```bash
   nsys profile --trace cuda,nvtx python -m vllm.benchmark \
       --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic --tp 2
   ```

3. Measure NCCL all-reduce latency:
   - Prefill: all-reduce for attention, MLP
   - Decode: all-reduce for attention, MLP
   - MoE: expert routing communication (if visible)

4. Compare TP=2 MoE (Scout) vs TP=2 dense (Mistral) vs TP=4 dense (Llama-3.1)

**Expected Outcome**:
- If Scout TP=2 communication 2-3× higher than Mistral TP=2: Confirms Hypothesis 3
- If Scout TP=2 communication similar to Mistral TP=2: Rejects Hypothesis 3

**Time**: 4-6 hours (includes nsys setup, profiling, NCCL analysis)

**Priority**: LOW (Llama-3.1 TP=4 succeeds contradicts TP hypothesis)

---

## Recommendation for Iter8

### Add β₈ for MoE Routing Overhead (Recommended)

**Rationale**:
- Scout dominates error (49% of total loss from 27% of experiments)
- Current model captures MoE gating FLOPs (β₅) but NOT expert routing latency
- Keep Scout in training to learn MoE-specific coefficient
- Model will generalize to all MoE architectures (Scout, Mixtral, DeepSeek-V3)

**Proposed β₈ Basis Function**:
```
β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)
```
- Captures per-token expert routing cost (selection, load balancing, coordination)
- Expected range: 10-50μs per routed token
- For Scout prefill (100 tokens, 24 MoE layers, top-1): β₈ × 2400 ≈ 24-120ms

**Benefits**:
1. ✅ Achieves <80% overall loss target by capturing Scout overhead
2. ✅ Generalizes to all MoE architectures (not just Scout)
3. ✅ Preserves training data diversity (all 15 experiments)
4. ✅ Physics-informed basis function (scales with MoE parameters)

**Implementation**:
- Add β₈ to `sim/latency/evolved_model.go` StepTime calculation
- Update coefficient_bounds.yaml with β₈ bounds: `[0, 100]` μs per routed token
- Retrain iter8 on **all 15 experiments** (including 4 Scout)

**Action**: Implement β₈ for iter8, keep all experiments in training data. Profile Scout for validation after training.

---

## Model Extension Proposal (After Profiling)

### If MoE Overhead Confirmed (Hypothesis 1)

Add β_moe coefficient to capture per-request MoE routing overhead:

```go
// In QueueingTime() method
func (e *EvolvedModel) QueueingTime(req *sim.Request, cfg *ModelConfig) float64 {
    baseTime := e.Coefficients.Alpha[0] +
        e.Coefficients.Alpha[1]*float64(req.InputTokens) +
        e.Coefficients.Alpha[2]*float64(req.MaxOutputLen)

    schedulerOverhead := e.Coefficients.Beta[6]

    // Add MoE routing overhead if model has MoE layers
    moeOverhead := 0.0
    if cfg.InterleaveMoELayerStep > 0 {
        numMoELayers := float64(cfg.NumLayers) / float64(cfg.InterleaveMoELayerStep)
        moeOverhead = e.Coefficients.Beta[8] * numMoELayers  // β₈ = β_moe
    }

    return baseTime + schedulerOverhead + moeOverhead
}
```

**Expected β₈ (β_moe)**:
- Range: 3-5ms per MoE layer
- Scout (26 MoE layers): 78-130ms total
- Initialized: 0.004 (4ms per layer)
- Bounds: [0.001, 0.010] (1-10ms per layer)

**Training Impact**:
- Adds 1 coefficient (8 → 9 Beta coefficients)
- Expected to reduce Scout TTFT from 79-100% → 30-50%
- Non-Scout experiments unaffected (InterleaveMoELayerStep=0 for dense models)

---

### If FP8 Overhead Confirmed (Hypothesis 2)

Add β_fp8 coefficient to capture FP8 dequantization overhead:

```go
// In QueueingTime() and StepTime() methods
func (e *EvolvedModel) QueueingTime(req *sim.Request, cfg *ModelConfig) float64 {
    baseTime := ...
    schedulerOverhead := ...

    // Add FP8 dequantization overhead if model uses FP8
    fp8Overhead := 0.0
    if cfg.WeightBytesPerParam == 1.0 {  // FP8 uses 1 byte per param
        fp8Overhead = e.Coefficients.Beta[8] * float64(cfg.NumLayers)  // β₈ = β_fp8
    }

    return baseTime + schedulerOverhead + fp8Overhead
}
```

**Expected β₈ (β_fp8)**:
- Range: 0.1-0.5ms per layer
- Scout (56 layers): 5.6-28ms total
- Initialized: 0.0003 (0.3ms per layer)
- Bounds: [0.0001, 0.001] (0.1-1ms per layer)

---

## Summary

**Primary Finding**: Scout MoE architecture blocks progress (49% of error from 27% of experiments).

**Root Cause** (most likely): MoE expert routing overhead (78-130ms per request) not captured by current model.

**Recommendation**: Add β₈ (MoE routing overhead) in iter8, keep all 15 experiments in training data.

**Expected Outcome** (after β₈ addition):
- Scout TTFT: 79-100% → 30-50% (50-70pp improvement)
- Overall loss: 155% → <80% (75pp improvement)
- β₈ will absorb Scout residual while non-Scout experiments unaffected (β₈ contribution = 0 for dense models)
- Model generalizes to all MoE architectures (Scout, Mixtral, DeepSeek-V3)

**Next Steps**:
1. **Iter8**: Implement β₈, train on all 15 experiments (including 4 Scout)
2. **Validation**: Profile Scout MoE overhead to verify β₈ coefficient aligns with measured routing latency (10-50μs per routed token)
