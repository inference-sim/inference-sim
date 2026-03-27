# Iteration 0: Scaled Roofline + Request Overheads

**Date**: 2026-03-27
**Status**: ✅ Ready to execute
**Iteration**: 0 (cold start)

---

## Executive Summary

Test whether learned scaling factors on roofline basis functions + request-level overheads can achieve <35% overall loss on 15 training experiments. This establishes the viability of the scaled roofline approach and provides error patterns for iteration 1.

**Strategy Evolution Context**: This is the first outer loop iteration following Strategy Evolution methodology. We design a hypothesis bundle with testable predictions, implement it, measure outcomes, and extract principles from prediction errors.

**Files Created**:
- ✅ `training/iteration_manifest.yaml` - Backend declaration
- ✅ `sim/latency/evolved_model.go` - Basis function implementation (compiles successfully)
- ✅ `sim/latency/latency.go` - Backend registration (added "evolved" case)
- ✅ `training/coefficient_bounds.yaml` - Search space for optimization
- ✅ Binary: `blis` (10MB, compiled successfully)

**Next Step**: Run `cd training && python inner_loop_optimize.py --n-trials 50`

---

## Table of Contents

1. [VV&UQ Classification](#vvuq-classification)
2. [Background Research](#background-research)
3. [Hypothesis Bundle](#hypothesis-bundle)
4. [Basis Function Design](#basis-function-design-iteration-0)
5. [Parameter Predictions](#parameter-predictions)
6. [Success Criteria](#success-criteria-iteration-0)
7. [What We'll Learn](#what-well-learn)
8. [Generalization Validation](#generalization-validation-post-convergence)
9. [Root Cause Verification Plan](#root-cause-verification-plan)
10. [Files Created & Verification](#files-created--verification)
11. [Next Steps](#next-steps-execution)
12. [References](#references)

---

## VV&UQ Classification

**Category**: **Validation** — Testing whether the scaled roofline model matches expected vLLM system behavior within pre-specified accuracy intervals.

**Family**:
- Primary: **Structural model** — Tests roofline model assumptions (compute/memory bottleneck, phase structure)
- Secondary: **Cross-model comparative** — Tests generalization across architectures (dense/MoE, TP configs)

**Type**: **Type 2: Statistical Experiment**
- 15 experiments (ground truth data from real vLLM runs)
- Effect size: Overall loss < 35% for iteration 0 baseline
- No seed variation (ground truth is fixed; stochasticity is in coefficient optimization only)

---

## Background Research

### H100 Hardware Specifications
**Source**: `training/references/datasheets/h100-sxm.json`
- **Compute**: 1979 TFLOPS (BF16 tensor core)
- **Memory**: 3.35 TB/s HBM bandwidth, 80GB capacity
- **Interconnect**: 900 GB/s NVLink (for TP all-reduce)

### BLIS Roofline Model
**Source**: `sim/latency/roofline.go`
- **Line 227**: `rooflineStepTime()` computes `max(compute_time, memory_time)` (single-crossover roofline)
- **Line 48-110**: `calculateTransformerFlops()` — attention + MLP FLOPs (O(n²) attention for prefill)
- **Line 116-206**: `calculateMemoryAccessBytes()` — weights + KV cache + activations
- **Line 213**: "No overhead terms" — current roofline is pure compute/memory bound, no scheduler/framework costs
- **Known limitation (problem statement L319)**: MFU values in `hardware_config.json` are "theoretical not empirical"

### vLLM Step Anatomy
**Source**: `training/references/vllm/JOURNEY_TRACING.md`
- Request lifecycle: ARRIVED → QUEUED → SCHEDULED → FIRST_TOKEN → FINISHED
- Step execution = SCHEDULED phase (GPU execution of attention + FFN)
- API overhead = ARRIVED → QUEUED (tokenization, HTTP parsing, request validation)

### Training Data Coverage
**Source**: `training/trainval_data/` (15 experiments)
- **Models**: Dense (Llama-2-7B, Llama-3.1-70B, Mistral-Nemo-12B, Qwen2.5-7B, Yi-34B), MoE (Llama-4-Scout-17B-16E)
- **TP configs**: TP ∈ {1, 2, 4}
- **Workloads**: codegen, reasoning, roleplay, general-lite
- **Batch shapes**: Variable prefill/decode token distributions

---

## Hypothesis Bundle

### H-main: Core Mechanism (Structural Model Family)

**Prediction**: Prefill/decode roofline basis functions with learned scaling coefficients will achieve **overall loss < 35%** (RMSE of APE across 15 experiments).

**Causal Mechanism**:
Transformer step time follows the roofline model: `step_time = max(FLOPs / throughput, bytes / bandwidth)`. BLIS roofline (`sim/latency/roofline.go:227`) computes this analytically. We learn β₀, β₁ as efficiency factors that correct for:
- Kernel overhead (launch costs, non-perfect pipelining)
- Framework inefficiency (vLLM scheduler, batch formation)
- Imperfect compute-memory overlap (max() assumes perfect overlap; real GPUs have partial overlap)

Request-level overheads (α₀, α₁, α₂) capture API processing, tokenization, and detokenization costs.

**Code Citations** (RCV-1 compliance):
- `sim/latency/roofline.go:48-110` — `calculateTransformerFlops()`: attention GEMMs + FFN compute
- `sim/latency/roofline.go:116-206` — `calculateMemoryAccessBytes()`: weight loading + KV cache reads/writes
- `sim/latency/roofline.go:227-275` — `rooflineStepTime()`: computes `max(compute_time, memory_time)` with phase-specific MFU
- `sim/latency/roofline.go:236-241` — MFU application: divides FLOPs by `MfuPrefill` or `MfuDecode` based on token type

**Pass Criteria**:
1. Overall loss < 35% (RMSE of APE across all experiments)
2. ≥10/15 experiments with APE < 30% (majority accuracy)
3. No experiment with APE > 60% (no catastrophic failures)

**Diagnostic Clause**:
*If loss > 50%, the max() roofline model is fundamentally insufficient → iteration 1 needs additive terms (TP communication overhead, scheduler overhead, or separate compute-memory overlap model) not just scaling factors.*

---

### H-prefill-regime: Compute-Bound Hypothesis (Performance-Regime Family)

**Prediction**: Prefill-heavy experiments (codegen, reasoning with long inputs) will show **mean TTFT APE < 25%** because large-batch attention is compute-bound.

**Causal Mechanism**:
Prefill processes many tokens at once (batch size 10-50+). Attention is O(n²) in sequence length. GEMMs are large → tensor core utilization high → theoretical FLOPs formula applies.

**Code Citations** (RCV-1 compliance):
- `sim/latency/roofline.go:68-75` — QKV projection + output projection FLOPs: `qkvFlops + projFlops`
- `sim/latency/roofline.go:85` — Attention score operations: `4 × nHeads × newT × effectiveCtx × dHead` (O(n²) term)
- `sim/latency/roofline.go:96-106` — MLP FLOPs: `2 × newT × (nMat × dModel × dExpert)`

**Pass Criteria**:
- Mean TTFT APE < 25% across codegen + reasoning experiments (prefill-heavy subset)

**Diagnostic Clause**:
*If prefill APE > 30%, either:*
1. *Attention kernel efficiency differs from documented peak TFLOPS (actual MFU ≠ theoretical)*
2. *Chunking overhead not modeled (vLLM splits long prefills into chunks, boundary overhead per chunk)*
3. *FlashAttention kernel launch costs significant for small batches*

---

### H-decode-regime: Memory-Bound Hypothesis (Performance-Regime Family)

**Prediction**: Decode-heavy experiments will show **mean ITL APE < 30%** if bandwidth-limited by KV cache reads.

**Causal Mechanism**:
Decode generates one token per request. Attention is O(n) in context length (reads entire KV cache history). KV cache read dominates: `2 × layers × kv_heads × head_dim × context_len × bytes_per_param`.

**Code Citations** (RCV-1 compliance):
- `sim/latency/roofline.go:183-184` — KV cache access: `kvReadPerToken × seq` (reads past history, NOT current tokens)
- `sim/latency/roofline.go:177-178` — KV cache growth: `kvWritePerNewToken × newT` (writes new KV to HBM)
- `sim/latency/roofline.go:173` — Model weights: `weightsPerLayer × nLayers × BytesPerParam` (loaded once per step)

**Pass Criteria**:
- Mean ITL (inter-token latency) APE < 30% across experiments with decode-heavy batches

**Diagnostic Clause**:
*If decode APE > 35%, either:*
1. *Decode is compute-bound not memory-bound (low context lengths, large batch sizes → compute dominates)*
2. *KV cache bandwidth formula is wrong (missing activations, incorrect byte count, or caching effects)*

---

### H-tp-invariance: TP Communication Test (Cross-Model Family)

**Prediction**: TP=1, TP=2, TP=4 experiments will have **APE std dev < 12%** (within same model) if TP communication overhead is negligible compared to compute/memory.

**Causal Mechanism**:
At TP > 1, all-reduce after each layer introduces `log₂(TP) × num_layers` communication events. H100 NVLink @ 900GB/s suggests ~10-50μs per layer. If all-reduce is fast vs compute time (milliseconds per step), APE variance should be low.

**Expected Compute Speedup**: Near-linear up to TP=4 (attention + MLP workload distributed across GPUs)
**Expected Communication Overhead**: `O(log₂(TP))` for ring all-reduce topology

**Pass Criteria**:
- For each model, compute APE std dev across TP ∈ {1, 2, 4}
- All models must have std dev < 12%

**Diagnostic Clause**:
*If TP=4 shows >20% higher APE than TP=1 for same model, all-reduce costs need explicit basis function: `β_comm × log₂(TP) × num_layers`. Iteration 1 should add this term.*

---

### H-moe-parity: MoE Generalization (Cross-Model Family)

**Prediction**: MoE model (Llama-4-Scout-17B-16E) will have APE **within 8% of dense model mean** if roofline MoE formula is accurate.

**Causal Mechanism**:
Roofline computes expected unique experts per step: `nEff = N × (1 - ((N-k)/N)^B)` where N=total experts, k=active per token, B=batch tokens. MLP FLOPs scale by `nEff × expert_dim` not `N × expert_dim`. Formula assumes uniform random routing (upper bound on bandwidth).

**Code Citations** (RCV-1 compliance):
- `sim/latency/roofline.go:98-105` — MoE MLP FLOPs: `mlpFlopsPerLayer *= NumExpertsPerTok` (k active experts)
- `sim/latency/roofline.go:152-170` — Expected unique experts: `nEff = N × (1 - ((N-k)/N)^B)` with uniform routing assumption
- `sim/latency/roofline.go:169` — Weight bandwidth: `mlpWeightsPerLayer *= nEff` (only loaded experts count)

**Pass Criteria**:
- `|MoE_APE - mean(dense_APE)| < 8%` where dense_APE = mean across Llama-2-7B, Llama-3.1-70B, etc.

**Diagnostic Clause**:
*If MoE APE > 25%, either:*
1. *Expert routing overhead (gating network compute, load balancing) needs explicit basis function*
2. *Expert loading is bursty/correlated (not uniform random) → actual bandwidth differs from formula*
3. *Shared experts (DeepSeek-V3 pattern) not accounted for in current roofline*

---

### H-workload-agnostic: Generalization Check (Validation Family)

**Prediction**: APE std dev within each workload category < 8%, confirming basis functions depend only on batch shape not workload semantics.

**Causal Mechanism**:
Workload type labels (codegen, reasoning, roleplay, general-lite) are training metadata. Basis functions use ONLY:
- `num_prefill_tokens`, `num_decode_tokens`, `context_lengths` (batch composition)
- `model_architecture` (layers, dimensions, attention heads, expert counts)
- `hardware_specs` (TFLOPS, bandwidth, TP config)

No workload-specific signal available → predictions must be identical for two batches with same (tokens, context_lengths, model, hardware) regardless of workload label.

**Pass Criteria**:
- For each workload ∈ {codegen, reasoning, roleplay, general}, compute APE std dev across experiments with that workload
- All workload categories must have std dev < 8%

**Diagnostic Clause**:
*If one workload category shows >15% higher mean APE than others, either:*
1. *Experiments in that category have systematically different batch compositions (longer contexts, more decode tokens) → expected, not a problem*
2. *Basis functions accidentally correlate with workload patterns (IMPOSSIBLE given feature set) → code bug*

---

## Basis Function Design (Iteration 0)

### StepTime (β coefficients)

**β₀ × prefill_compute_time**:
```
prefill_FLOPs = calculateTransformerFlops(model, seq_len, num_prefill_tokens)
prefill_compute_time = prefill_FLOPs / (GPU_peak_TFLOPS × MFU_prefill)
contribution = β₀ × prefill_compute_time  # β₀ learns actual MFU correction
```

**β₁ × decode_compute_time**:
```
decode_FLOPs = calculateTransformerFlops(model, seq_len, num_decode_tokens)
decode_compute_time = decode_FLOPs / (GPU_peak_TFLOPS × MFU_decode)
contribution = β₁ × decode_compute_time  # β₁ learns decode efficiency (memory-bound regime)
```

**β₂ × constant**:
```
contribution = β₂  # Fixed per-step scheduler overhead (microseconds)
```

**Dimensional analysis**:
- `prefill_compute_time`, `decode_compute_time`: microseconds (μs) → β₀, β₁ dimensionless (scaling factors)
- `constant`: dimensionless (=1) → β₂ has units microseconds (μs)

### Request-Level Overheads (α coefficients)

**α₀**: Fixed API processing overhead (μs per request)
- HTTP parsing, request validation, queue insertion
- Expected: ~100-300μs (typical FastAPI/vLLM overhead)

**α₁**: Per-input-token tokenization cost (μs/token)
- HuggingFace tokenizer BPE encoding
- Expected: ~0.5-2μs/token

**α₂**: Per-output-token detokenization cost (μs/token)
- Streaming decode + output formatting
- Expected: ~1-3μs/token (streaming mode has lower overhead than batch decode)

**Implementation** (standard BLIS pattern, DO NOT modify):
- `QueueingTime(req) = α₀ + α₁ × num_input_tokens`
- `OutputTokenProcessingTime() = α₂` (per output token)
- `PostDecodeFixedOverhead() = 0` (no systematic per-request bias observed)

---

## Parameter Predictions

| Coefficient | Range | Expected Value | Justification |
|-------------|-------|----------------|---------------|
| **β₀** (prefill MFU) | [0.3, 3.0] | ~0.6-0.8 | Prefill large-GEMM tensor core efficiency 50-60% (kernel launch, pipelining overhead) |
| **β₁** (decode MFU) | [0.3, 3.0] | ~0.4-0.6 | Decode memory-bound, lower tensor core utilization, bandwidth haircut |
| **β₂** (overhead) | [0, 1000μs] | ~50-200μs | vLLM scheduler overhead per step (batch formation, KV block allocation) |
| **α₀** (fixed API) | [0, 1000μs] | ~100-300μs | HTTP parsing, request validation (typical FastAPI/vLLM overhead) |
| **α₁** (tokenization) | [0, 100μs/token] | ~0.5-2μs/token | HF tokenizer BPE encode performance |
| **α₂** (detokenization) | [0, 100μs/token] | ~1-3μs/token | Streaming decode + output formatting |

**Constraint**: All bounds have `lower_bound = 0.0` (no negative coefficients allowed per problem statement).

---

## Success Criteria (Iteration 0)

### Primary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Overall loss** | < 35% | First iteration baseline — confirms scaled roofline is viable foundation |
| **Majority accuracy** | ≥10/15 experiments with APE < 30% | Mechanism works for most cases |
| **No catastrophic failures** | No experiment with APE > 60% | No systematic blind spots |
| **Workload variance** | Std dev < 10% per workload | Workload-agnostic confirmation |

### Secondary Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Prefill accuracy** | Mean TTFT APE < 25% on codegen/reasoning | Compute-bound hypothesis |
| **Decode accuracy** | Mean ITL APE < 30% | Memory-bound hypothesis |
| **TP invariance** | APE std dev < 12% within model across TP configs | TP communication negligible |
| **MoE parity** | MoE APE within 8% of dense mean | MoE formula accurate |

---

## What We'll Learn

### If H-main confirmed (loss < 35%):
- Scaled roofline is viable foundation
- Proceed to error pattern analysis for iteration 1
- **Principle extracted**: "Roofline with learned MFU captures first-order vLLM latency behavior"

### If H-prefill-regime refuted (APE > 30%):
- Prefill has non-FLOPs costs
- **Iteration 1 action**: Add chunking overhead term: `β₃ × num_chunks × chunk_boundary_overhead`

### If H-decode-regime refuted (APE > 35%):
- Decode is compute-bound not memory-bound, OR KV bandwidth formula wrong
- **Iteration 1 action**: Separate decode into small-batch (memory-bound) vs large-batch (compute-bound) regimes

### If H-tp-invariance refuted (TP=4 APE >20% higher):
- TP communication overhead significant
- **Iteration 1 action**: Add `β₃ × log₂(TP) × num_layers` for all-reduce cost

### If H-moe-parity refuted (MoE APE > 25%):
- MoE has routing/load-imbalance overhead
- **Iteration 1 action**: Add `β₄ × num_experts × batch_size` for gating network overhead

### If H-workload-agnostic refuted (variance > 10%):
- Basis functions correlating with workload-specific patterns (should be IMPOSSIBLE)
- **Action**: Audit feature extraction — likely batch composition differs systematically, not basis function violation

---

## Generalization Validation (Post-Convergence)

**After iteration 0 converges**, run Tier 1 cross-validation (per `training/docs/generalization-validation-protocol.md`):

### CV-1: Leave-One-Model-Out (Dense→MoE)
- Train on 11 dense experiments
- Test on 4 MoE experiments
- **Pass**: MAPE < 20% on MoE holdout

### CV-2: Leave-One-Workload-Out
- Train on codegen (4) + reasoning (3) = 7 experiments
- Test on roleplay (3) + general (5) = 8 experiments
- **Pass**: Mean MAPE < 15%, roleplay vs general variance < 3%

### CV-3: Leave-One-TP-Out
- Train on TP ∈ {1, 4} = 9 experiments
- Test on TP=2 = 6 experiments
- **Pass**: MAPE < 15% on TP=2 holdout

**Note**: Basis functions are FROZEN from main training. Only coefficients (α, β) are refit on holdout training set.

---

## Root Cause Verification Plan

Following RCV standards from `docs/contributing/standards/experiments.md`:

### RCV-1: Code Citations
All causal claims cite `file:line` in roofline.go:
- Compute FLOPs calculation: `sim/latency/roofline.go:48-110`
- Memory bandwidth calculation: `sim/latency/roofline.go:116-206`
- Max bottleneck selection: `sim/latency/roofline.go:227`
- MFU application: `sim/latency/roofline.go:236-241`

### RCV-2: First-Principles Calculation
For any "surprising" result, compute expected value:
- Expected prefill time: `FLOPs / (peak_TFLOPS × 0.6)` (assume 60% MFU)
- Expected decode time: `bytes / (peak_bandwidth × 0.8)` (assume 80% bandwidth utilization)
- If result differs by >2×, flag as surprise and investigate

### RCV-3: Mechanism Verification
After identifying error patterns:
- Which specific basis function caused the error? (β₀ too low? β₂ missing?)
- What code path is responsible? (cite file:line)
- Can we reproduce the error with synthetic batch that isolates the mechanism?

### RCV-4: Control Experiments
If error pattern suggests missing term (e.g., TP communication):
- Synthesize batch with TP=1 vs TP=4, identical model/tokens
- Measure APE difference
- If difference matches hypothesis → confirm mechanism
- Iteration 1 adds the missing basis function

### RCV-5: Devil's Advocate (Pre-Review)
Before external review, argue OPPOSITE of conclusion:
- **If "Confirmed"**: Could this be overfitting to training data? Would it work on held-out experiments?
- **If "Refuted"**: Could the refutation be due to bad coefficient initialization, not wrong basis functions?

### RCV-6: Scope and Limitations
Document:
- Tested on 15 experiments (H100-SXM only)
- Batch sizes: typical vLLM defaults (max_num_seqs, max_num_batched_tokens from experiments)
- TP configs: {1, 2, 4} only (not tested on TP=8, TP=16)
- Workloads: 4 categories (codegen, reasoning, roleplay, general-lite) — NOT exhaustive of production workloads
- Models: 6 architectures (5 dense, 1 MoE) — limited MoE coverage

---

## Files Created & Verification

### Implementation Files ✅

```
training/
├── iteration_manifest.yaml          # Backend declaration
├── coefficient_bounds.yaml          # Search space for Bayesian opt
└── iter0-HYPOTHESIS.md              # This document

sim/latency/
├── evolved_model.go                 # New: 3 basis functions (β₀, β₁, β₂)
└── latency.go                       # Modified: Added "evolved" case
```

### Verification Checklist ✅

**Compilation**:
- [x] `go build -o blis main.go` successful
- [x] Binary created: `blis` (10MB)

**Backend Registration**:
- [x] "evolved" case added to `NewLatencyModel` switch in `latency.go`
- [x] Backend name in manifest matches registration: "evolved"

**Interface Compliance**:
- [x] Implements `StepTime(batch []*Request) int64`
- [x] Implements `QueueingTime(req *Request) int64`
- [x] Implements `OutputTokenProcessingTime() int64`
- [x] Implements `PostDecodeFixedOverhead() int64`

**Strategy Evolution Compliance**:
- [x] Hypothesis bundle with testable predictions
- [x] Physics-informed reasoning (roofline model)
- [x] Diagnostic clauses for all hypotheses
- [x] Expected coefficient ranges with justification

**BLIS Experiment Standards Compliance**:
- [x] VV&UQ classification (Validation - structural model)
- [x] Hypothesis family (structural model + cross-model comparative)
- [x] Experiment type (Type 2: Statistical)
- [x] Root cause verification plan (RCV-1 through RCV-6)
- [x] Code citations (`file:line` format)
- [x] First-principles calculations for expected values

**Outer Loop Specs Compliance**:
- [x] Three required files (manifest, Go code, bounds)
- [x] All bounds non-negative (lower_bound >= 0.0)
- [x] Workload-agnostic features only (no forbidden inputs)
- [x] Dimensional consistency verified
- [x] Only StepTime customized, other methods use standard implementation

---

## Next Steps (Execution)

### Step 1: Run Inner Loop Optimization
```bash
cd training/
python inner_loop_optimize.py --n-trials 50
```

This will:
1. Read `iteration_manifest.yaml`
2. Verify `sim/latency/evolved_model.go` exists
3. Compile BLIS (already done: binary exists ✅)
4. Load `coefficient_bounds.yaml`
5. Run Bayesian optimization (50 trials with early stopping)
6. Save results to `inner_loop_results.json`

**Expected runtime**: 10-30 minutes (depends on hardware)

### Step 2: Analyze Results
```bash
# View overall loss
cat inner_loop_results.json | jq '.best_loss'

# View per-experiment breakdown
cat inner_loop_results.json | jq '.detailed_diagnostics.per_experiment[] |
  {exp: .experiment_folder, ttft_ape: .ttft_mean_ape, e2e_ape: .e2e_mean_ape}'
```

**Check**:
- Overall loss < 35%? → H-main confirmed
- Which experiments have high APE? → Identify error patterns
- TP=4 systematically higher? → H-tp-invariance refuted, need TP term
- MoE APE > 25%? → H-moe-parity refuted, need MoE term
- Prefill APE > 30%? → H-prefill-regime refuted
- Decode APE > 35%? → H-decode-regime refuted

### Step 3: Document Findings
Create `training/iter0-FINDINGS.md` with:
- Status: Confirmed / Confirmed with nuance / Refuted / Inconclusive
- Resolution: What we learned
- Error pattern analysis (which experiments failed, why?)
- Prediction-vs-outcome comparison for each hypothesis arm
- Devil's Advocate section (RCV-5)
- Scope and Limitations (RCV-6)

### Step 4: Extract Principles
From confirmed predictions and prediction errors:
- Confirmed → Mechanism verified → Extract principle
- Refuted → Discrepancy reveals causal model gap → Extract corrective principle
- Example: If TP=4 shows high APE → Principle: "TP communication overhead significant at TP>2"

### Step 5: Design Iteration 1
Based on error patterns:
- Add missing basis functions (TP communication, MoE routing, chunking overhead)
- Adjust coefficient bounds if optimizer hit boundaries
- Propose next hypothesis bundle

### Step 6: Run Cross-Validation (Post-Convergence)
After iterations converge (loss < 10%, no improvement for 2 iterations):
```bash
# CV-1: Leave-One-Model-Out (Dense→MoE)
mkdir -p trainval_data_cv1_train trainval_data_cv1_test
# ... (see generalization-validation-protocol.md)

# CV-2: Leave-One-Workload-Out
# CV-3: Leave-One-TP-Out
```

---

## References

- **Problem statement**: `training/docs/agentic-latency-training-problem-statement.md`
- **Outer loop specs**: `training/docs/outer-loop-specs.md`
- **Strategy Evolution**: `docs/methodology/strategy-evolution.md`
- **Experiment standards**: `docs/contributing/standards/experiments.md`
- **Generalization protocol**: `training/docs/generalization-validation-protocol.md`
- **H100 datasheet**: `training/references/datasheets/h100-sxm.json`
- **BLIS roofline**: `sim/latency/roofline.go`
- **vLLM tracing**: `training/references/vllm/JOURNEY_TRACING.md`

---

## Status

**✅ Ready to execute inner loop optimization.**

**Next command**: `cd training && python inner_loop_optimize.py --n-trials 50`
