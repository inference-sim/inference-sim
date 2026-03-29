# Iteration 2: Findings and Principles

## Summary

Iteration 2 FAILED to improve upon iter1, with overall loss INCREASING from 134.54% to 136.19% (+1.64%). The hypothesis that adding β₇ (very long context overhead) and β₈ (per-request decode overhead) would reduce loss to <80% was REJECTED. Both new coefficients remained at their initial values throughout 51 optimization trials, indicating they provided no predictive value. Reasoning experiments still have catastrophic 99.97-99.99% TTFT errors, and Scout experiments still have 168-197% combined loss, confirming that the root causes are not missing overhead terms but rather incorrect formulas and unresolved simulator bugs.

**Key takeaways**:
- Adding more terms without validating their functional form leads to parameter bloat
- Coefficients that don't move during optimization should be removed
- Reasoning and Scout failures require investigation (profiling, debugging), not formula additions

---

## Error Analysis

### Systematic Patterns

**High-error experiments (APE > 50% combined)**:

1. **Reasoning experiments (4 experiments: 100% TTFT failures)**:
   - 20260217-170634-llama-2-7b-tp1-reasoning: TTFT=99.97%, E2E=93.11%, combined=193.09%
   - 66-qwen2-5-7b-instruct-tp1-reasoning-1-1: TTFT=99.99%, E2E=96.42%, combined=196.42%
   - 48-llama-4-scout-17b-16e-tp2-reasoning-2: TTFT=99.99%, E2E=95.05%, combined=195.04%
   - **Pattern**: ALL reasoning workloads fail catastrophically, regardless of model (Llama-2, Qwen2.5, Scout) or TP (1, 2)
   - **Why**: The β₇ formula `(prompt_tokens - 4096) / 1000 × num_layers` is structurally incorrect. It assumes linear overhead scaling with excess tokens and layers, but real vLLM overhead for long contexts is likely:
     - Non-linear (quadratic attention memory bandwidth saturation)
     - Batch-size-dependent (preemption triggers at different thresholds based on available KV cache)
     - KV cache hit-rate-dependent (prefix cache effectiveness varies with prompt uniqueness)

2. **Scout MoE experiments (4 experiments: 168-197% combined loss)**:
   - 17-llama-4-scout-17b-16e-tp2-general-2: TTFT=99.98%, E2E=97.59%, combined=197.58%
   - 20-llama-4-scout-17b-16e-tp2-codegen-2: TTFT=93.45%, E2E=90.92%, combined=184.37%
   - 21-llama-4-scout-17b-16e-tp2-roleplay-2: TTFT=87.50%, E2E=80.73%, combined=168.22%
   - 48-llama-4-scout-17b-16e-tp2-reasoning-2: TTFT=99.99%, E2E=95.05%, combined=195.04%
   - **Pattern**: ALL Scout experiments fail catastrophically across ALL workloads (general, codegen, roleplay, reasoning)
   - **Why**: The simulator bugs identified in iter1 (interleaved MoE layers, intermediate_size_mlp parsing, MoE gating FLOPs) were either not fully fixed OR new bugs were introduced. The systematic 90-100% errors across all workloads indicate a fundamental prefill calculation bug, not a data quality issue.

3. **TP=2 non-Scout experiments (2 experiments: 63-125% combined loss)**:
   - 62-mistral-nemo-12b-tp2-general-lite-2-1: TTFT=71.74%, E2E=53.36%, combined=125.10%
   - 65-01-ai-yi-34b-tp2-general-lite-2-1: TTFT=58.61%, E2E=4.03%, combined=62.64%
   - **Pattern**: TP=2 non-Scout experiments have high TTFT errors (59-72%) but variable E2E errors (4-53%)
   - **Why**: TP=2 prefill overhead may not be captured correctly by β₃ (TP communication) alone. All-reduce overhead in prefill differs from decode (larger activations, less pipelining).

4. **TP=4 large model experiments (2 experiments: 53-98% combined loss)**:
   - 60-llama-3-1-70b-tp4-general-lite-4-1: TTFT=89.80%, E2E=8.52%, combined=98.32%
   - 61-llama-3-1-70b-tp4-codegen-4-1: TTFT=41.94%, E2E=11.01%, combined=52.94%
   - **Pattern**: TTFT errors are high (42-90%) but E2E errors are EXCELLENT (8-11%)
   - **Why**: Large model prefill overhead (70B parameters, TP=4 communication) is not captured by β₀ alone. Decode predictions are excellent, suggesting the latency model works well for decode on large models but underestimates prefill time.

**Low-error experiments (APE < 50% combined)**:

1. **Mistral codegen (1 experiment: 19.42% combined loss)**:
   - 63-mistral-nemo-12b-tp1-codegen-1-1: TTFT=11.41%, E2E=8.01%, combined=19.42%
   - **Why**: Mistral-Nemo (12B) with TP=1 codegen workload hits the "sweet spot" - medium model size, no TP communication overhead, codegen workload has predictable batch composition

2. **Qwen2.5 roleplay (1 experiment: 46.99% combined loss)**:
   - 64-qwen2-5-7b-instruct-tp1-roleplay-1-1: TTFT=1.83%, E2E=45.16%, combined=46.99%
   - **Why**: Excellent TTFT prediction (1.83%) but moderate E2E error (45.16%). Roleplay workload may have variable output lengths causing E2E variance.

**Only 2/15 experiments achieved <50% combined loss**, compared to iter1's 7/15. This confirms iter2 made predictions WORSE.

---

### Error Correlations

**✅ Confirmed correlations with LOW error**:
- **TP=1, non-reasoning, non-Scout, medium model size (7-12B)**: Best predictions (Mistral codegen: 19.42%, Qwen roleplay: 46.99%)
- **Decode-heavy workloads**: Experiments with low TTFT proportion (Qwen roleplay: TTFT=1.83%) have better combined loss
- **Small model + TP=1**: Llama-2-7b-tp1-codegen (82.43%) is higher than expected, but Mistral/Qwen TP=1 are excellent, suggesting model-specific effects

**❌ Rejected correlations (expected to cause errors but don't)**:
- **Workload type (codegen vs roleplay vs general)**: No clear pattern - Scout codegen (184%) fails while Mistral codegen (19%) succeeds. This confirms the model's workload-agnostic design is correct, but model architecture and TP dominate errors.
- **Batch size regime**: The sigmoid interpolation for batch_size=8 transition doesn't show clear error spikes, suggesting the smooth transition worked or is irrelevant compared to larger error sources.

**✅ Confirmed correlations with HIGH error**:
- **Reasoning workload**: 100% of reasoning experiments have catastrophic TTFT failures (99.97-99.99%)
- **Scout MoE architecture**: 100% of Scout experiments have catastrophic failures (168-197% combined loss)
- **TP=2**: 7/8 TP=2 experiments have >60% combined loss (6 are Scout, 1 is Mistral with 125%)
- **Large model TP=4 prefill**: Both TP=4 experiments have high TTFT errors (42-90%) despite excellent E2E errors (8-11%)

---

### Root Cause Hypotheses

**Principle 1: β₇ formula is structurally incorrect for reasoning overhead**

**Evidence**:
- β₇=1.0 did not move from initial value during 51 optimization trials
- All 4 reasoning experiments have 99.97-99.99% TTFT errors WITH β₇ present
- The formula `(prompt_tokens - 4096) / 1000 × num_layers` assumes linear overhead scaling, but real vLLM long-context overhead is:
  - Quadratic in sequence length (attention memory bandwidth: O(n²) intermediate matrices)
  - Batch-size-dependent (preemption triggers vary with available KV cache)
  - KV cache hit-rate-dependent (unique prompts have lower prefix cache effectiveness)

**Mechanism**:

The β₇ formula is based on the assumption that overhead scales linearly with excess tokens beyond 4096. However, vLLM's actual long-context behavior:

1. **Attention memory bandwidth saturation**: For sequence length n > 4096, the attention mechanism must materialize O(n²) intermediate matrices (query × key). For n=8192, this is 4× more memory bandwidth than n=4096, not 2×. The linear formula underpredicts by 2×.

2. **KV cache preemption**: vLLM's PagedAttention triggers preemption when KV cache fills. For long sequences (>4096 tokens), preemption happens earlier in prefill, forcing recomputation. The overhead is step-function-like (0 overhead until preemption, then +50-100% overhead), not linear.

3. **Prefix cache miss rate**: Reasoning prompts (CoT chains) are highly unique. If prompt A is "Think step-by-step about quantum physics" and prompt B is "Think step-by-step about economics", they share only ~10 tokens. The formula assumes 4096 tokens are "cached" and only excess tokens pay overhead, but in practice the entire 8000+ token prompt is processed fresh.

**Action**: Next iteration must profile vLLM reasoning experiments to measure actual prefill time breakdown:
- Attention bandwidth time (via nsys profiling)
- KV cache preemption frequency (via vLLM logs)
- Prefix cache hit rates (via vLLM metrics)

Then design a formula that captures the TRUE overhead:
- Quadratic attention term: `(prompt_tokens / 4096)² × scaling_factor`
- Preemption term: `step_function(kv_cache_utilization > 0.9) × recomputation_cost`
- Cache miss term: `(1 - prefix_cache_hit_rate) × prompt_tokens`

---

**Principle 2: Scout fixes applied but experiments still fail - model structure inadequate**

**Evidence**:
- All 4 Scout experiments have 168-197% combined loss in iter2
- **CRITICAL TIMELINE**: Scout bugs (InterleaveMoELayerStep, DenseIntermediateDim, split FLOPs/bandwidth) were fixed on March 28, 11:52 PM - 11:59 PM
- Unit tests pass: `TestScoutInterleavedArchitecture_EndToEnd` validates FLOPs calculation
- Iter2 optimization ran March 29, 5:36 AM (6 hours AFTER fixes, binary recompiled with fixes)
- Scout failures are systematic across ALL workloads (general, codegen, roleplay, reasoning)
- β₆ (MoE gating)=0.008 unchanged from iter1

**Mechanism**:

The identified Scout bugs WERE fixed and tests pass, yet real experiments still fail at 168-197% combined loss. Three possible explanations:

1. **Tests validate FLOPs but not end-to-end latency prediction**:
   - `TestScoutInterleavedArchitecture_EndToEnd` only checks FLOPs calculation matches expected values
   - Does NOT validate that predicted latency matches real vLLM latency for Scout experiments
   - FLOPs may be correct, but how coefficients (β₀, β₁, β₅, β₆) are applied to Scout's mixed architecture may be wrong

2. **Additional unfixed bugs in coefficient application**:
   - FLOPs calculation splits MoE vs dense layers correctly (test passes)
   - But latency model may apply β₀ (prefill MFU) uniformly to all layers instead of per-layer-type
   - Or β₆ (MoE gating) may be applied to all 48 layers instead of only 24 MoE layers
   - These bugs wouldn't be caught by FLOPs-only tests

3. **Fundamental model structure incompatibility**:
   - Current basis functions assume homogeneous layers (all MoE OR all dense)
   - Scout has 24 MoE + 24 dense interleaved, requiring per-layer-type efficiency factors
   - Single β₀ (prefill MFU) cannot represent different MFU for MoE layers (lower due to routing overhead) vs dense layers (higher)
   - May need layer-type-specific basis functions: β₀_dense, β₀_moe, β₅_dense, β₅_moe

**Action**: Next iteration must:
1. **Add end-to-end latency validation test for Scout**: Extend tests beyond FLOPs to validate full latency prediction:
   ```go
   func TestScoutLatencyPrediction_EndToEnd(t *testing.T) {
       // Use trained coefficients from iter2
       coeffs := loadCoefficients("iter2")

       // Predict latency for Scout experiment 20 (codegen-2)
       predicted := predictLatency(scoutConfig, workload, coeffs)
       actual := loadGroundTruth("20-llama-4-scout-17b-16e-tp2-codegen-2")

       ttftAPE := abs(predicted.TTFT - actual.TTFT) / actual.TTFT * 100
       e2eAPE := abs(predicted.E2E - actual.E2E) / actual.E2E * 100

       // Should be <30% APE, not 93% TTFT / 91% E2E as in iter2
       assert.Less(t, ttftAPE, 30.0)
       assert.Less(t, e2eAPE, 30.0)
   }
   ```
   This will pinpoint WHERE in the latency calculation Scout diverges from reality.

2. **Add Scout-specific basis functions if tests reveal coefficient application issues**:
   - Split β₀ (prefill MFU) into β₀_dense and β₀_moe with separate scaling per layer type
   - Split β₅ (decode compute-bound) into β₅_dense and β₅_moe
   - This allows the optimizer to learn different efficiencies for MoE vs dense layers

3. **Alternative: Exclude Scout from training set if model structure cannot represent it**:
   - Train on Llama/Qwen/Mistral (dense models only), test on Scout
   - If test loss is still >100%, accept that current basis functions cannot model Scout
   - Document Scout as "unsupported architecture" until per-layer-type basis functions are added

---

**Principle 3: β₈ (per-request decode overhead) is too small to matter**

**Evidence**:
- β₈=3e-05 (30μs) did not move from initial value during 51 optimization trials
- At 30μs/request, β₈ contributes only 0.12-0.48ms for 4-16 request batches
- Typical decode step times are 10-1000ms, so β₈ is <0.1% of step time
- β₁=1.553 remains unchanged from iter1, showing β₈ failed to normalize the inflated decode MFU

**Mechanism**:

The β₈ term was designed to capture per-request overhead (scheduler work, attention state setup, kernel launch) that was hypothesized to be inflating β₁. However:

1. **Magnitude mismatch**: Real per-request overhead in vLLM is likely larger:
   - Scheduler per-request work: 5-20μs × 16 requests = 80-320μs
   - Attention state setup (KV block tables): 10-50μs × 16 requests = 160-800μs
   - Kernel launch overhead: 10-30μs × 16 requests = 160-480μs
   - **Total: 400-1600μs (0.4-1.6ms), not 30μs**

   The initial value of 30μs was set too low, causing β₈ to be irrelevant.

2. **Wrong functional form**: The flat per-request term `num_decode_requests × β₈` may be incorrect:
   - Real per-request overhead likely scales with TP (synchronization overhead per request increases with TP)
   - Real per-request overhead may scale with batch size (scheduler complexity is O(n log n) for priority queue operations)
   - Real per-request overhead may already be captured by β₂ (constant scheduler overhead) or β₃ (TP communication)

3. **β₁ inflation has a different cause**: The β₁=1.553 inflation may not be due to missing per-request overhead. Alternative hypotheses:
   - Decode FLOPs formula is wrong (undercounting FLOPs by 35%)
   - Memory bandwidth formula is wrong (undercounting KV cache reads)
   - Decode MFU calculation uses wrong roofline assumptions (peak memory bandwidth vs achieved bandwidth)

**Action**: Next iteration must:
1. **Remove β₈**: The ablation hypothesis predicted removing β₈ would degrade loss by >5%, but the evidence (β₈ didn't move, β₁ unchanged) suggests removing it will have NO impact. Simplify the model by removing ineffective terms.

2. **Investigate β₁ inflation root cause**: Instead of adding per-request overhead terms, investigate why β₁>1:
   - Audit decode FLOPs calculation: `2 × batch_size × output_tokens × hidden_dim²` - is this correct?
   - Audit memory bandwidth calculation: `2 × bytes_per_param × weights + 2 × batch_size × KV_cache_size` - is this correct?
   - Audit decode MFU formula: `decode_time = max(flops_time / β₅, bandwidth_time / β₁)` - should this be a sum, not max?

3. **Alternative hypothesis**: β₁>1 may indicate the decode roofline model is fundamentally wrong. Real GPU execution may not follow the max(compute, memory) roofline - it may be a mix of compute and memory at all batch sizes. Consider replacing discrete regimes with a single mixed term: `decode_time = β₁_compute × flops_time + β₁_memory × bandwidth_time`.

---

**Principle 4: TP=4 large model prefill overhead requires dedicated term**

**Evidence**:
- Both TP=4 experiments (Llama-3.1-70B) have high TTFT errors (42-90%) but excellent E2E errors (8-11%)
- This asymmetry suggests prefill is underestimated while decode is accurate
- β₃ (TP communication)=0.394 unchanged from iter1, suggesting TP formula is correct for decode but insufficient for prefill

**Mechanism**:

Large model (70B parameters) with TP=4 has significantly higher prefill overhead than small models:

1. **All-reduce communication overhead in prefill**: Prefill processes L tokens in parallel, requiring L × all-reduce operations. For TP=4, each all-reduce communicates `hidden_dim / TP = 8192 / 4 = 2048` elements, but prefill has L=512-2048 tokens, so total communication is `L × 2048 × 4 bytes = 4-16MB per layer`. This is 4-16× more communication than decode (which processes 1 token).

2. **Tensor parallelism overhead scales with model size**: 70B model has 80 layers vs 7B model's 32 layers. The all-reduce overhead is proportional to num_layers, but β₃ may not capture the layerwise scaling correctly.

3. **Prefill activation memory bandwidth**: Large models have large activations (batch_size × L × hidden_dim). For TP=4, each GPU must write activations to HBM, then read them back for the next layer. This memory bandwidth overhead is not captured by β₀ (which only captures compute MFU).

**Action**: Next iteration must:
1. **Add TP-dependent prefill overhead term**: Add β₉ that captures TP communication in prefill:
   - Formula: `β₉ × TP × num_layers × prompt_tokens × hidden_dim / bandwidth`
   - This scales with TP (more communication), num_layers (more all-reduces), and prompt_tokens (more parallel tokens)

2. **Alternative: Split β₀ by TP**: Instead of a global prefill MFU (β₀), use TP-dependent prefill MFU:
   - β₀_tp1 for TP=1 (no communication overhead)
   - β₀_tp2 for TP=2 (moderate communication overhead)
   - β₀_tp4 for TP=4 (high communication overhead)
   - This allows the optimizer to learn different prefill efficiencies per TP config

---

**Principle 5: Adding terms without validation leads to parameter bloat**

**Evidence**:
- β₇ and β₈ both stayed at initial values for 51 optimization trials
- Overall loss INCREASED from 134.54% to 136.19% (+1.64%)
- The new terms added complexity (9 Beta terms vs 8 in iter1) without predictive value

**Mechanism**:

The iteration design philosophy was "identify failure modes, add terms to capture them". However:

1. **Functional form matters more than term count**: Adding β₇ and β₈ with incorrect functional forms provides no predictive value. The optimizer cannot learn good coefficients for structurally wrong formulas.

2. **Overfitting risk**: Adding more terms increases the risk of overfitting to the training set. With 9 Beta terms and 3 Alpha terms (12 parameters total) for 15 experiments, the model has 0.8 parameters per experiment. This is close to the overfitting threshold.

3. **Optimizer exploration**: Bayesian optimization with 51 trials must explore a 12-dimensional space (3 Alpha + 9 Beta). With more dimensions, the optimizer needs more trials to converge. Early convergence (51/250 trials) suggests the optimizer found a local minimum and didn't fully explore the space.

**Action**: Next iteration must:
1. **Remove ineffective terms**: Remove β₇ and β₈ (back to 8 Beta terms from iter1)
2. **Validate functional forms before adding**: Before adding new terms, validate the functional form matches vLLM behavior:
   - Profile vLLM to measure actual overhead
   - Plot overhead vs proposed formula input (e.g., prompt_tokens, batch_size, TP)
   - Ensure formula captures the relationship (linear, quadratic, step function, etc.)

3. **Increase optimization trials**: With 8 Beta + 3 Alpha = 11 parameters, use 500-1000 trials to ensure full exploration of the space. Iter2's 51 trials with early convergence suggests under-exploration.

---

## Coefficient Analysis

**Alpha [α₀, α₁, α₂]** from `best_params.alpha`: [0.00116, 4.25e-05, 9.57e-05]

- α₀ = 1.16ms: Fixed API processing overhead (unchanged from iter1)
- α₁ = 42.5μs/token: Per-input-token tokenization (unchanged from iter1)
- α₂ = 95.7μs/token: Per-output-token detokenization (unchanged from iter1)

**Physical interpretation**: All Alpha coefficients are IDENTICAL to iter1, suggesting:
- The request-level overhead model is stable and correct
- The optimization focused on Beta (step-level) terms, but found no improvement

**Beta [β₀, β₁, ..., β₈]** from `best_params.beta`: [0.203, 1.553, 0.00012, 0.394, 0.00037, 0.651, 0.008, 1.0, 3e-05]

- β₀ = 0.203: Prefill compute MFU - **unchanged from iter1, still 2× below physical range (0.40-0.55)**
  - Physical interpretation: 20% prefill MFU implies H100 is only achieving 20% of peak FLOPs during prefill. Real H100 achieves 40-60% for large matmuls.
  - This suggests β₀ is compensating for missing prefill overhead (TP communication, activation bandwidth, KV cache writes)

- β₁ = 1.553: Decode memory-bound MFU - **unchanged from iter1, still 1.7× above physical range (0.60-0.90)**
  - Physical interpretation: 155% memory bandwidth efficiency is physically impossible (cannot read more data than peak HBM bandwidth)
  - This suggests β₁ is compensating for undercounted memory bandwidth (missing KV cache overhead, missing activation reads)

- β₂ = 0.12μs: Constant scheduler overhead - **unchanged from iter1, 40× below expected range (5-50μs)**
  - Physical interpretation: 0.12μs is negligible compared to step times (10-1000ms)
  - This suggests constant scheduler overhead is genuinely negligible OR is captured by other terms (β₈ per-request, β₄ KV mgmt)

- β₃ = 0.394: TP communication overhead - **unchanged from iter1, within expected range (0.30-0.50)**
  - Physical interpretation: TP communication adds ~39% overhead to base step time
  - This is consistent with all-reduce latency for TP=2,4 configs
  - β₃ appears correct for decode but may be insufficient for prefill (see Principle 4)

- β₄ = 0.37μs: KV management overhead per request - **unchanged from iter1, within expected range (0-50μs)**
  - Physical interpretation: 0.37μs per request for KV cache management (block allocation, eviction checks)
  - Iter1 ablation showed +30.28% E2E degradation when removed, confirming this is CRITICAL despite small magnitude
  - β₄ is the most important additive overhead term

- β₅ = 0.651: Decode compute-bound MFU - **unchanged from iter1, within physical range (0.50-0.70)**
  - Physical interpretation: 65% compute MFU for large-batch decode (compute-bound regime)
  - This is physically plausible for H100 running decode matmuls (smaller than prefill, less efficient)

- β₆ = 0.008: MoE gating overhead - **unchanged from iter1, within expected range (0.005-0.015)**
  - Physical interpretation: MoE gating adds ~0.8% overhead per step
  - This is consistent with expert routing logic (topk, load balancing)
  - However, Scout experiments still fail (168-197%), suggesting β₆ is correct but Scout bugs remain

- β₇ = 1.0: Very long context prefill overhead - **NEW in iter2, did not move from initial value**
  - Physical interpretation: At β₇=1.0, the formula `(prompt_tokens - 4096) / 1000 × num_layers` scales linearly
  - However, reasoning experiments still fail (99.97-99.99% TTFT), so β₇ is structurally incorrect
  - **Should be removed in iter3**

- β₈ = 3e-05 (30μs): Per-request decode overhead - **NEW in iter2, did not move from initial value**
  - Physical interpretation: 30μs per request is too small to matter (0.48ms for 16 requests vs 10-1000ms step times)
  - β₁ remains inflated at 1.553, showing β₈ failed to normalize it
  - **Should be removed in iter3**

**Redundant terms**:
- β₇: Stayed at initial value, reasoning experiments still fail → REMOVE
- β₈: Stayed at initial value, β₁ unchanged → REMOVE

**Missing physics**:
- TP-dependent prefill overhead: TP=4 experiments have high TTFT errors (42-90%) despite excellent E2E errors (8-11%)
- Quadratic long-context attention overhead: Reasoning experiments have 99.97-99.99% TTFT errors
- Activation memory bandwidth in prefill: Large models may have activation bandwidth bottlenecks not captured by β₀

---

## Recommendations for iter3

### Priority 1: Critical Issues (must address)

1. **Add end-to-end Scout latency validation beyond FLOPs tests**
   - **Evidence**: All Scout experiments fail with 168-197% combined loss across all workloads
   - **CRITICAL**: FLOPs tests already pass (TestScoutInterleavedArchitecture_EndToEnd), but real experiments fail
   - **Timeline**: Scout bugs fixed March 28, iter2 ran March 29 with fixes applied - failures persist
   - **Action**: Add end-to-end latency prediction validation:
     ```go
     // Test end-to-end latency prediction for Scout (not just FLOPs)
     func TestScoutLatencyPrediction_Experiment20(t *testing.T) {
         // Load trained coefficients from iter2
         coeffs := loadIterationCoefficients(2)

         // Load Scout experiment 20 (codegen-2)
         exp := loadExperiment("20-llama-4-scout-17b-16e-tp2-codegen-2")

         // Predict latencies using full latency model (not just FLOPs)
         predicted := predictExperimentLatencies(exp, coeffs)

         // Compare against ground truth
         ttftAPE := computeAPE(predicted.MeanTTFT, exp.GroundTruth.MeanTTFT)
         e2eAPE := computeAPE(predicted.MeanE2E, exp.GroundTruth.MeanE2E)

         // Currently: TTFT=93.45%, E2E=90.92% (catastrophic)
         // After proper fix: Should be <30%
         t.Logf("Scout TTFT APE: %.2f%% (target: <30%%)", ttftAPE)
         t.Logf("Scout E2E APE: %.2f%% (target: <30%%)", e2eAPE)

         if ttftAPE > 30.0 || e2eAPE > 30.0 {
             t.Errorf("Scout latency prediction still fails after bug fixes")
             // This will reveal WHERE in latency calculation Scout diverges
         }
     }
     ```
   - **Expected outcome**: Test will FAIL (Scout still at 90-100% APE), revealing that FLOPs fixes are insufficient and coefficient application has bugs

2. **Profile reasoning experiments to measure true long-context overhead**
   - **Evidence**: Reasoning experiments have 99.97-99.99% TTFT errors, β₇ formula is structurally incorrect
   - **Action**: Profile vLLM with nsys for reasoning workloads:
     ```bash
     # Profile reasoning experiment
     nsys profile -o reasoning_profile --trace=cuda,nvtx \
       python -m vllm.entrypoints.openai.api_server \
       --model meta-llama/Llama-2-7b-hf \
       --max-model-len 8192

     # Analyze attention kernel time vs sequence length
     nsys stats reasoning_profile.nsys-rep --report cuda_gpu_kern_sum
     ```
   - **Measure**: Attention kernel time for L=512, 1024, 2048, 4096, 8192 tokens
   - **Expected finding**: Attention time scales O(L²) not O(L), revealing quadratic overhead
   - **Design new formula**: `β₇ × (prompt_tokens / 4096)² × scaling_factor` instead of linear

3. **Remove β₇ and β₈ (ineffective terms)**
   - **Evidence**: Both stayed at initial values for 51 trials, no loss improvement
   - **Action**: Revert to iter1's 8 Beta terms (remove β₇ and β₈)
   - **Expected outcome**: Loss should remain ~136% (no change) or potentially improve (simpler model, less overfitting risk)

---

### Priority 2: Improvements (address after Priority 1)

1. **Add TP-dependent prefill overhead term (β₉)**
   - **Evidence**: TP=4 experiments have high TTFT errors (42-90%) but excellent E2E errors (8-11%)
   - **Action**: Add β₉ term for TP communication in prefill:
     ```
     β₉ × TP × num_layers × prompt_tokens × hidden_dim / network_bandwidth
     ```
   - **Bounds**: [0, 0.5] (allow up to 50% prefill overhead from TP communication)
   - **Expected outcome**: TP=4 TTFT errors drop from 42-90% to <30%

2. **Investigate β₁ inflation root cause**
   - **Evidence**: β₁=1.553 (155% memory bandwidth efficiency is impossible), β₈ failed to normalize it
   - **Action**: Audit decode roofline calculations:
     - Verify decode FLOPs formula: `2 × batch_size × output_tokens × hidden_dim × (4 × hidden_dim + 2 × MLP_dim)`
     - Verify memory bandwidth formula: `bytes_per_param × (weights + 2 × batch_size × KV_cache_size_per_token)`
     - Test alternative hypothesis: Replace `max(flops_time / β₅, bandwidth_time / β₁)` with `β₁_compute × flops_time + β₁_memory × bandwidth_time` (mixed compute+memory, not regimes)
   - **Expected outcome**: Either find undercounting in formulas OR discover roofline model is wrong for decode

3. **Increase optimization trials to 500-1000**
   - **Evidence**: Iter2 converged early at 51/250 trials, may have found local minimum
   - **Action**: Set `n_trials=1000` and disable early stopping
   - **Expected outcome**: More thorough exploration of parameter space, potentially better coefficients

---

### Priority 3: Refinements (nice-to-have)

1. **Split β₀ (prefill MFU) by TP config**
   - **Evidence**: TP=4 has high TTFT errors (42-90%), TP=1 has moderate errors (12-48%)
   - **Action**: Use TP-dependent prefill MFU: β₀_tp1, β₀_tp2, β₀_tp4
   - **Bounds**: β₀_tp1 ∈ [0.4, 0.7], β₀_tp2 ∈ [0.3, 0.6], β₀_tp4 ∈ [0.2, 0.5] (decreasing MFU with increasing TP)
   - **Expected outcome**: TP=4 TTFT errors drop, β₀ coefficients become physically plausible

2. **Test hypothesis: Remove sigmoid interpolation, revert to discrete regime split**
   - **Evidence**: Sigmoid smoothness is inconclusive, overall loss increased with sigmoid
   - **Action**: Revert to iter1's discrete `if batch_size < 8` split
   - **Expected outcome**: Either loss improves (sigmoid was harmful) or no change (sigmoid was neutral)

---

## Basis Function Changes for iter3

### Proposed changes:

**Remove**:
- β₇ (very long context overhead) - stayed at initial value, reasoning experiments still fail
- β₈ (per-request decode overhead) - stayed at initial value, β₁ unchanged

**Add**:
- β₇ (NEW): Quadratic long-context attention overhead - `β₇ × (prompt_tokens / 4096)² × num_layers`
  - Captures O(L²) attention memory bandwidth scaling
  - Bounds: [0, 2.0] (allow up to 2× prefill overhead for very long contexts)
  - Target: Reduce reasoning experiment TTFT errors from 99.97-99.99% to <60%

- β₈ (NEW): TP-dependent prefill communication overhead - `β₈ × TP × num_layers × prompt_tokens`
  - Captures TP all-reduce overhead in prefill (scales with TP, layers, and tokens)
  - Bounds: [0, 0.001] (μs per TP×layer×token)
  - Target: Reduce TP=4 TTFT errors from 42-90% to <30%

**Modify**:
- Revert sigmoid interpolation to discrete regime split (iter1 approach)
- Consider splitting β₀ into TP-dependent variants (β₀_tp1, β₀_tp2, β₀_tp4) if simple β₈ addition doesn't fix TP=4 prefill

---

## Bounds Adjustments for iter3

**Tighten** (converged coefficients):
- α₀: [0.0008, 0.0015] (tighten around 0.00116)
- α₁: [0.00003, 0.00006] (tighten around 0.0000425)
- α₂: [0.00007, 0.00012] (tighten around 0.0000957)
- β₃: [0.3, 0.5] (tighten around 0.394, confirmed correct for decode)
- β₄: [0.0002, 0.0005] (tighten around 0.00037, CRITICAL term despite small magnitude)
- β₅: [0.5, 0.8] (tighten around 0.651, physically plausible)
- β₆: [0.005, 0.015] (tighten around 0.008, correct for MoE gating)

**Expand** (to encourage movement):
- β₀: [0.1, 0.8] (expand from [0.1, 0.8] to allow larger range, currently stuck at 0.203)
- β₁: [0.3, 2.0] (expand from [0.3, 1.5] to allow optimizer to explore higher values, currently 1.553)
- β₂: [0, 0.0001] (keep unchanged, appears genuinely negligible)

**New terms**:
- β₇ (quadratic long-context): [0, 2.0] (allow up to 2× prefill overhead)
- β₈ (TP-dependent prefill): [0, 0.001] (μs per TP×layer×token)

---

## Expected Iteration 3 Outcomes

**Overall loss target**: <100% (from 136.19% in iter2)
- TTFT RMSE target: <50% (from 72.75% in iter2)
- E2E RMSE target: <50% (from 63.44% in iter2)

**Per-experiment targets**:
- Reasoning experiments: TTFT <60% (from 99.97-99.99%) after adding quadratic β₇
- Scout experiments: Combined loss <80% (from 168-197%) after unit test debugging
- Dense non-reasoning experiments: Maintain or improve current best (Mistral codegen: 19.42%)

**Coefficient targets**:
- β₀: 0.40-0.60 (up from 0.203, toward physical range)
- β₁: 0.80-1.20 (still above ideal 0.6-0.9, but reduced from 1.553)
- β₂: 0-50μs (allow exploration, currently 0.12μs)
- β₃: 0.35-0.45 (stable from iter2's 0.394)
- β₄: 0.3-0.5μs (stable from iter2's 0.37μs, CRITICAL term)
- β₅: 0.60-0.75 (stable from iter2's 0.651)
- β₆: 0.006-0.012 (stable from iter2's 0.008)
- β₇: 0.5-1.5 (NEW, quadratic long-context overhead)
- β₈: 0.0002-0.0008 (NEW, TP-dependent prefill overhead)

**If iter3 loss remains >100%**: Consider fundamental model redesign - current additive overhead approach may be insufficient. Explore:
- Neural network latency model (replace handcrafted formulas with learned features)
- Per-model coefficients (accept that different models need different coefficients)
- Data quality audit (verify ground-truth vLLM experiments are correct)

**If Scout experiments still fail**: Accept that Scout's interleaved MoE+dense architecture cannot be represented with current basis functions. Either:
- Exclude Scout from training set (train on dense models only, report Scout as unsupported)
- Implement per-layer-type basis functions (β₀_moe, β₀_dense, β₅_moe, β₅_dense)
