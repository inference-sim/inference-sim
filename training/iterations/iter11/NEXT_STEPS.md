# Iteration 11: Action Items for Iter12

## Status: Basis Functions Validated ✅, Model Still Broken ❌

After comprehensive audit and unit test validation, the β₁₀ and β₃' basis functions are **CORRECT**. The catastrophic loss (4084%) is due to other coefficients being out of range, particularly β₆ (scheduler overhead).

---

## Completed ✅

1. ✅ Audited β₁₀ basis function implementation → **CORRECT**
2. ✅ Audited β₃' basis function implementation → **CORRECT**
3. ✅ Ran unit tests (`TestBeta10BatchingInefficiency`, `TestBeta3PrimeKVSeqLen`) → **PASS (0% error)**
4. ✅ Fixed YAML comment errors in `coefficient_bounds.yaml` (ms → μs)
5. ✅ Created detailed audit report (`BASIS_FUNCTION_AUDIT.md`)
6. ✅ Created corrected findings (`CORRECTED_FINDINGS.md`)

---

## Critical Finding: β₆ Is The Real Problem

**β₆ (Scheduler Overhead)**: 59.3ms (expected 15-40ms) → **48-295% TOO HIGH**

This is the primary culprit for the catastrophic loss. Three possible explanations:

### Hypothesis A: Expected Range Is Wrong

**Claim**: β₆ = 59ms is actually correct; the expected range of 15-40ms is wrong.

**Evidence needed**:
- Profile vLLM scheduler CPU time on actual workloads
- Measure batch formation + KV block allocation time
- Compare against Scout long-sequence vs short-sequence experiments

**Action**:
```bash
# Add profiling to vLLM scheduler
python profile_vllm_scheduler.py \
  --model mistralai/Mistral-Nemo-Instruct-2407 \
  --workload general-lite \
  --profile-points "scheduler_start,batch_formation,kv_alloc,scheduler_end"
```

### Hypothesis B: β₁₀ and β₆ Are Competing

**Claim**: β₁₀ and β₆ both try to explain queueing delays, causing the optimizer to get confused.

**Evidence**:
- β₁₀ captures: Long sequences → low batch efficiency → queueing delays
- β₆ captures: Scheduler overhead per request (should include queueing?)
- Both converge to reasonable values (β₁₀ = 0.95μs, β₆ = 59ms)
- But overall loss is still catastrophic (optimizer stuck in local minimum)

**Possible solution**: Split β₆ into two terms:
- β₆ₐ: Scheduler CPU overhead (fixed per request, expected 15-40ms)
- β₆ᵦ: Queueing delay component (scales with batch efficiency, captured by β₁₀?)

**Action**: Try removing β₁₀ temporarily and see if β₆ stays at 59ms or increases further.

### Hypothesis C: Missing Complementary Term

**Claim**: β₆ is absorbing variance from a missing term (e.g., memory bandwidth saturation).

**Evidence needed**:
- Profile GPU memory bandwidth during long-sequence batches
- Check if GPU memory bandwidth saturates during Scout general-lite (500 tokens)
- Measure if memory bandwidth explains the 59ms vs 15-40ms gap

**Possible missing terms**:
- β₁₁: Memory bandwidth saturation (long sequences → high memory traffic)
- β₁₂: Chunked prefill overhead (vLLM splits long prefills into chunks)
- β₁₃: GPU→CPU KV cache offloading (when GPU memory is full)

**Action**: Add β₁₁ for memory bandwidth saturation and retry optimization.

---

## Action Plan for Iter12

### Phase 1: Validate Expected Ranges (1-2 days)

**Goal**: Determine if β₆ = 59ms is correct or wrong.

**Tasks**:
1. Profile vLLM scheduler overhead on 5 representative experiments:
   - Scout general-lite (500 tokens) → Expect high overhead
   - Scout roleplay (100 tokens) → Expect low overhead
   - Mistral Nemo general-lite (500 tokens) → Dense model comparison
   - Llama-2-7b reasoning-lite (200 tokens) → Mid-range sequence length
   - Measure: batch_formation_time + kv_block_alloc_time

2. Profile KV cache management overhead (β₃):
   - Measure PagedAttention block manager initialization time
   - Measure block allocation time per request (base overhead)
   - Measure block allocation scaling with sequence length
   - Verify if 0.2ms (base) + 0.25μs/token-layer (seq-len) matches profile

3. Profile decode per-request overhead (β₇):
   - Measure output processing time per decode request
   - Measure TP coordination overhead
   - Verify if 5ms (iter11) or 8-20ms (expected) is correct

**Deliverables**:
- Profiling report with measured values for β₆, β₃, β₇
- Updated expected ranges based on profiling (not physics estimates)
- Decision: Keep β₆ range as 15-40ms OR expand to 50-80ms

### Phase 2A: If β₆ = 59ms Is Correct (Update Ranges)

**Hypothesis**: The expected range of 15-40ms was wrong; 59ms is physically accurate.

**Actions**:
1. Update `coefficient_bounds.yaml`:
   ```yaml
   # β₆: Scheduler overhead (profiling shows 50-80ms, not 15-40ms)
   - [0.040, 0.090]  # [40ms, 90ms] — expanded based on profiling
   ```

2. Update β₃ and β₇ ranges if profiling shows different values

3. Retry iter12 with updated expected ranges (warm-start from iter9)

**Expected outcome**: Loss should improve significantly if ranges were the only problem.

### Phase 2B: If β₆ = 59ms Is Wrong (Split or Add Term)

**Hypothesis**: β₆ is absorbing variance from missing term or competing with β₁₀.

**Option 1: Split β₆**

Split scheduler overhead into CPU vs queueing components:
- β₆ₐ (index 7): Scheduler CPU overhead (fixed, 15-40ms)
- β₆ᵦ (index 11): Queueing delay per request (variable, explained by batch efficiency?)

**Option 2: Add β₁₁ (Memory Bandwidth Saturation)**

Add complementary term for long-sequence memory traffic:
- β₁₁ (index 11): Memory bandwidth saturation overhead
- Basis function: `Σ(prefillTokens × kvCacheSize / memoryBandwidth)`
- Expected: 0.01-0.1 μs per byte transferred

**Option 3: Remove β₁₀ Temporarily**

Test if β₁₀ and β₆ are interfering:
1. Remove β₁₀ from iter12 configuration
2. See if β₆ increases further (would confirm competition)
3. If β₆ stays at 59ms, suggests β₁₀ is not the issue

**Action**: Based on Phase 1 profiling, choose one option and implement for iter12.

### Phase 3: Retry Optimization (Iter12)

**Configuration**:
- Warm-start from iter9 (NOT iter10 or iter11)
- Use validated expected ranges from Phase 1 profiling
- Either: updated β₆ range, OR split β₆, OR add β₁₁
- Keep β₁₀ and β₃' as-is (they're correct!)

**Success criteria**:
- Overall loss: **<110%** (31% improvement from iter9's 160%)
- TTFT RMSE: **<50%** (23% improvement from iter9's 64.8%)
- E2E RMSE: **<65%** (32% improvement from iter9's 95.8%)
- β₆: Within updated expected range (whether 15-40ms or 50-80ms)
- At least 8/11 coefficients within expected ranges

---

## Alternative: Simplify Model First

If Phase 1 profiling is too time-consuming, consider:

### Simplified Iter12: Remove β₁₀ and β₃'

**Rationale**: Revert to iter9 model (before adding β₁₀ and β₃') to establish baseline.

**Hypothesis**: Maybe the model is over-parameterized; iter9 was stuck at 160% loss but at least stable.

**Actions**:
1. Remove β₁₀ and β₃' from model
2. Revert β₃ to single term (no split)
3. Retry optimization with iter9 architecture
4. Compare result to iter9's 160% loss

**Expected outcome**: If loss stays at 160%, confirms β₁₀ and β₃' weren't helping (but also weren't hurting). If loss improves below 160%, suggests model was over-parameterized.

---

## Don't Waste Compute

**Before starting iter12 optimization**:
1. ✅ Run profiling (Phase 1) to validate expected ranges
2. ✅ Run unit tests to validate any new basis functions
3. ✅ Manually calculate expected contributions for new terms
4. ✅ Document hypothesis with success criteria BEFORE training

**Never again**:
- ❌ Train for 11 hours without validating basis functions first
- ❌ Accept hypothesis diagnoses without code audit
- ❌ Rely on physics estimates without profiling validation
- ❌ Change YAML comments without fixing actual bugs

---

## Timeline Estimate

**Phase 1 (Profiling)**: 1-2 days
- Set up vLLM profiling instrumentation
- Run profiling on 5 experiments
- Analyze results and update expected ranges

**Phase 2 (Design Iter12)**: 1 day
- Based on profiling, choose: update ranges, split β₆, or add β₁₁
- Write unit tests for any new basis functions
- Update coefficient bounds YAML

**Phase 3 (Optimize Iter12)**: 11 hours (500 trials)
- Run optimization with validated configuration
- Monitor convergence
- Analyze results

**Total**: 2-3 days for Phase 1-2, then 11 hours for optimization.

**ROI**: 2-3 days of profiling prevents another wasted 11-hour training run.

---

## Summary

✅ **Do**: Profile vLLM to validate β₆, β₃, β₇ expected ranges
✅ **Do**: Base iter12 on profiling data, not physics estimates
✅ **Do**: Run unit tests before any training
❌ **Don't**: Modify β₁₀ or β₃' implementations (they're correct!)
❌ **Don't**: Train without validated expected ranges
❌ **Don't**: Accept hypothesis diagnoses without code audit

**Next immediate action**: Run Phase 1 profiling to determine if β₆ = 59ms is correct or wrong.
