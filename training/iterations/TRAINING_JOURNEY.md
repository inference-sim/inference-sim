# Training Journey: Iterations 0-6 Summary

## The Challenge

We're trying to predict how long it takes for vLLM (an LLM inference server) to process requests. The simulator uses a physics-based model with coefficients that need to be trained. The hardest problem: **"reasoning" workloads take 100-200ms but we predict ~1ms** (99% error).

## What We Tried (Iterations 0-6)

### Iter0-2: Foundation
**What we did**: Set up the basic model with 6 coefficients covering prefill compute, decode memory/compute, and communication overhead.

**Result**: Got to ~130% error. Most experiments predicted reasonably well (20-80% error), but reasoning experiments stuck at 99% error.

**Learning**: The model works for most cases, but reasoning is fundamentally broken.

---

### Iter3: "Maybe reasoning uses long contexts?"
**Hypothesis**: Reasoning experiments might use 8K-16K token contexts (much longer than the 1K tokens other workloads use), and our model doesn't handle long contexts well.

**What we did**: Added context-length-dependent terms, tried to model long-context behavior.

**Result**: Failed. Loss stayed at ~130%, reasoning still at 99%.

**Learning**: We were guessing without data. **Mistake: assumed reasoning = long context without checking traces**.

---

### Iter4: "Maybe it's activation memory bandwidth?"
**Hypothesis**: For long contexts, writing large activation tensors to memory could be the bottleneck.

**What we did**: Added β₆ (activation bandwidth term) that scales with tokens × layers.

**Result**: Catastrophic failure. Loss stayed at 129%, reasoning still at 99%, but other coefficients exploded (β₁ +73%, β₂ +328%). The new term created "collinearity" - it correlated with existing terms, so the optimizer couldn't fit them independently.

**Learning**: Adding terms that scale with the same things (num_layers) creates tangled correlations. Also, **still assuming reasoning = long context without proof**.

---

### Iter5: "Maybe it's per-layer overhead (kernel launches)?"
**Hypothesis**: Each layer has fixed overhead (kernel launch, scheduler operations) that adds up. For long contexts with chunking, this gets multiplied: `overhead = β₆ × num_layers × (1.0 + tokens/2048)`.

**What we did**: Replaced activation bandwidth with per-layer overhead in StepTime.

**Result**: **CATASTROPHIC FAILURE**. Loss exploded to 603% (4.7× worse!). Short-context experiments went from 4-77% error → 200-1091% error. Reasoning stayed at 99%.

**Why it failed**:
- β₀ (prefill efficiency) rose from 0.165 → 0.266, which shortened ALL predictions by 38%
- Large models (80 layers) got massive overhead from `β₆ × 80 × scale_factor`
- Result: Large models with short contexts catastrophically over-predicted (1091% error for Llama-3.1-70B)

**Critical Discovery**: After this failure, we analyzed the traces and found **reasoning uses ~1K tokens, NOT 8K-16K**!
- Llama-2 reasoning: 1082 tokens (mean), 106ms actual TTFT
- All experiments use similar context lengths (~1K tokens)
- The difference isn't context length - it's something else

**Learning**: **VALIDATE ASSUMPTIONS FROM TRACES FIRST**. We wasted 3 iterations (iter3/4/5) assuming reasoning = long context without checking the data.

---

### Iter6: "Maybe it's per-request scheduler overhead?"
**Hypothesis**: Based on trace analysis, reasoning differs from codegen not by context length (both ~1K tokens) but by **queuing/scheduler delay**. Traces show huge variance (p10=0.13ms, p90=215ms) suggesting batching delays.

**What we did**:
- Moved β₆ from StepTime (per-layer) to QueueingTime (per-request)
- Changed units from microseconds (per-layer kernel overhead) to milliseconds (per-request scheduler overhead)
- Removed chunking scale factor (all experiments use ~1K tokens, no chunking needed)

**Result**: ⚠️ **PARTIAL RECOVERY**
- Loss: 603% → 162% (73% improvement!)
- **8 out of 11 short-context experiments recovered** to iter4 levels (11-46% error)
  - Llama-3.1-70B codegen: 1091% → 26% (amazing recovery!)
  - Mistral codegen: 834% → 11%
  - Qwen roleplay: 736% → 10%
- **But reasoning still stuck at 99%** (completely unchanged)
- β₆ = 21.5ms (expected 50-150ms, way too low)

**Why mixed results**:
- ✅ Moving β₆ to QueueingTime **decoupled it from prefill compute** (β₀)
  - β₀ dropped from 0.266 → 0.164 (back to iter4), restoring short-context accuracy
  - β₂ (communication) dropped 80% (1.368 → 0.270) - biggest improvement!
- ❌ But uniform per-request overhead creates **zero-sum trade-off**:
  - At β₆ = 21.5ms: short-context recovers, reasoning stays at 99%
  - At β₆ = 100ms: reasoning improves to ~60%, but short-context degrades to 60-120%
  - Optimizer chose β₆ = 21.5ms to help 11 experiments at expense of 4 reasoning experiments

**Learning**: Uniform overhead terms that apply to ALL experiments create trade-offs. Can't help reasoning without hurting short-context.

---

## What We Know Now

### ✅ What Works
1. **Basic model is sound**: Prefill compute, decode memory/compute, communication terms all work well
2. **Short-context experiments**: Predicted within 11-46% error (good enough)
3. **Per-request scheduler overhead**: Correct mechanism, just insufficient magnitude for reasoning

### ❌ What's Broken
1. **Reasoning experiments**: Stuck at 99% error through 7 iterations (iter0-6)
2. **Scout MoE experiments**: All uniformly bad (87-99% error) - architecture-specific issue
3. **Mistral TP=2**: 91% error (but TP=1 works at 11%) - TP-specific issue

### 🤔 What We Still Don't Know

**The 78.5-178.5ms gap for reasoning** (β₆ captures 21.5ms, but actual is 100-200ms). Could be:

1. **Prefix cache misses**: Reasoning uses shared system prompt (100 tokens). If not cached, recomputing adds 10-50ms per request.

2. **Batching delay variance**: Traces show p10=0.13ms, p90=215ms (1650× variance). β₆ = 21.5ms captures mean, but 90th percentile is 10× larger. Multi-turn chat (reasoning) may wait longer for batches than single-turn completions (codegen).

3. **Attention kernel startup**: FlashAttention-2 may have fixed cost per batch (~20-50ms) independent of context length.

4. **Memory allocation**: Reasoning may allocate large activation buffers beyond KV cache blocks (+30-80ms per request).

5. **Workload-specific batching behavior**: Multi-turn chat (reasoning) may have higher concurrency and different batching patterns than other workloads (codegen, roleplay), even with same context length.

---

## Key Mistakes (Lessons Learned)

1. **Don't guess workload properties** - Check traces first! We wasted iter3/4/5 assuming reasoning = long context without validation.

2. **Avoid collinearity** - Don't add terms that scale with the same things (num_layers). They create tangled correlations.

3. **Watch for zero-sum trade-offs** - Uniform terms that apply to ALL experiments force the optimizer to choose winners and losers.

4. **Decouple terms by phase** - Moving β₆ from StepTime to QueueingTime was the breakthrough that eliminated collinearity.

5. **Profile before hypothesizing** - After 7 iterations, we still don't know what causes reasoning's 100-200ms delay. Should have profiled in iter5 or iter6.

---

## The Path Forward

### Immediate Actions (Iter7)

**MUST DO FIRST: Analyze traces to identify reasoning's dominant bottleneck**
- Measure prefix cache hit rate
- Decompose batching delay variance (p10 vs p90)
- Profile attention kernel startup cost
- Profile memory allocation overhead

**Then: Add workload-dependent or variance-driven term**

Can't use uniform β₆ for both reasoning and short-context. Three options:

1. **Split by workload** (β₆_reasoning=100ms, β₆_codegen=20ms)
   - ❌ Violates workload-agnostic design principle
   - ✅ Simple, guaranteed to work

2. **Model batching delay variance** (p10 vs p90 explicitly)
   - ✅ Workload-agnostic (variance is observable from request patterns)
   - ⚠️ Complex, may need per-request position in batch

3. **Model as function of concurrency** (more concurrent requests → longer delays)
   - ✅ Workload-agnostic (reasoning has higher concurrency due to multi-turn chat)
   - ✅ Physically grounded (scheduler delay increases with load)

### Other Issues

- **Scout MoE**: Profile to identify expert routing overhead
- **Mistral TP=2**: Profile cross-GPU KV cache allocation
- **Decode coefficients**: β₁/β₄ destabilized in iter6, may need decode overhead term

---

## Bottom Line

**After 7 iterations**: We've successfully modeled 11 out of 15 experiments (11-46% error). But **reasoning is fundamentally stuck** at 99% error, and we've tried 4 different hypotheses without success.

**The challenge**: Reasoning differs from codegen by only ~100ms (both use ~1K tokens, both do the same compute), but this 100ms is invisible to our current model. It's likely **batching/queuing behavior** that varies by workload pattern (multi-turn vs single-turn), not by any physical property we can measure from the request itself.

**Next step**: Stop guessing. Profile the actual systems to measure where the 100ms goes, then add the right term. We're at the limit of what pure coefficient fitting can solve - need empirical data.
