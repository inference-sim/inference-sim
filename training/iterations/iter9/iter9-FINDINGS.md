# Iteration 9: Findings and Principles

## Summary

**What worked?**
- Partial coefficient reversion: β₅ (MoE gating) and β₇ (decode overhead) both reverted toward physical ranges (-52%, -58%), no longer absorbing spurious Scout error.
- Short-sequence Scout improvement: Scout roleplay (-53pp) and Scout codegen (-34pp) showed significant TTFT improvements.

**What didn't work?**
- FP8 hypothesis FALSE: β₉ converged to essentially zero (0.14 μs vs expected 17-50 μs), proving FP8 dequantization is NOT Scout's bottleneck.
- Long-sequence Scout failure: Scout general-lite (92% TTFT) and Scout reasoning-lite (91% TTFT) showed minimal/zero improvement, contradicting the architecture-specific (FP8) hypothesis.
- Overall loss WORSENED: 160.6% (+5.25pp from iter8's 155.35%).

**What did we learn?**
- **Critical Discovery**: Scout's bottleneck is **sequence-length-dependent**, NOT architecture-dependent (FP8). Short-sequence Scout workloads improved significantly, while long-sequence workloads failed completely. This reveals a batching, scheduling, or memory-related bottleneck that scales inversely with sequence length.
- **Coefficient explosions** (β₆ +654%, β₂ +343%, β₈ +143%) indicate the model is absorbing Scout long-sequence error into scheduler overhead, TP communication, and MoE routing terms.

---

## Error Analysis

### Systematic Patterns

**High-error experiments** (TTFT APE > 50%):
- **Scout general-lite** (exp 17): TTFT=92%, E2E=100% — NEW DATA, longest sequence (~400-600 tokens), worst Scout performer
- **Scout reasoning-lite** (exp 48): TTFT=91%, E2E=100% — Long sequence (~200-400 tokens), minimal improvement from iter8 (-8pp)
- **Mistral Nemo general-lite** (exp 62): TTFT=91%, E2E=99% — NEW WORKLOAD (no iter8 baseline), long sequence
- **Llama-2-7b reasoning-lite** (exp 67): TTFT=84%, E2E=99% — Long sequence, stable from iter8
- **Qwen2.5-7b reasoning-lite** (exp 66): TTFT=79%, E2E=98% — Long sequence, stable from iter8
- **01-ai Yi-34B general-lite** (exp 65): TTFT=78%, E2E=98% — Long sequence, stable from iter8
- **Llama-3.1-70B general-lite** (exp 60): TTFT=77%, E2E=98% — Long sequence, stable from iter8
- **Llama-2-7b codegen** (exp 7): TTFT=76%, E2E=97% — Moderate sequence, stable from iter8
- **Scout codegen** (exp 20): TTFT=58%, E2E=98% — Moderate sequence, improved significantly from iter8 (-34pp)

**Pattern 1: Long-sequence workloads (general-lite, reasoning-lite) dominate high-error experiments**
- General-lite: 6/15 experiments, avg TTFT=87%
- Reasoning-lite: 3/15 experiments, avg TTFT=85%
- These workloads have longest sequences (~200-600 tokens) and highest errors

**Pattern 2: Scout long-sequence vs short-sequence divergence**
- **Long-sequence Scout** (general-lite 92%, reasoning-lite 91%): FAILED
- **Short-sequence Scout** (roleplay 26%, codegen 58%): SUCCEEDED (improved -53pp, -34pp from iter8)
- This proves Scout's bottleneck is **sequence-length-dependent**, not architecture-dependent (FP8)

**Low-error experiments** (TTFT APE < 30%):
- **Llama-2-7b roleplay**: TTFT=21%, E2E=89% — Short sequence, stable from iter8
- **Scout roleplay** (exp 21): TTFT=26%, E2E=94% — Short sequence, improved massively from iter8 (-53pp)
- **Llama-2-7b general**: TTFT=27%, E2E=92% — Moderate sequence, stable from iter8
- **Mistral Nemo codegen**: TTFT=26%, E2E=93% — Moderate sequence, stable from iter8
- **Llama-3.1-70B codegen**: TTFT=28%, E2E=94% — Moderate sequence, stable from iter8
- **Qwen2.5-7b roleplay**: TTFT=8%, E2E=88% — Short sequence, stable from iter8 (BEST experiment overall)

**Pattern 3: Short-sequence workloads (roleplay) and moderate-sequence workloads (codegen) perform best**
- Roleplay: Shortest sequences (~50-100 tokens), lowest TTFT errors (8-26%)
- Codegen: Moderate sequences (~100-200 tokens), moderate TTFT errors (26-76%)

**Error correlations**:
- ✅ **CONFIRMED: Sequence length correlates STRONGLY with TTFT error**
  - Short-sequence workloads (roleplay): 8-26% TTFT error (BEST)
  - Moderate-sequence workloads (codegen): 26-76% TTFT error (MODERATE)
  - Long-sequence workloads (general-lite, reasoning-lite): 77-92% TTFT error (WORST)
- ❌ **REJECTED: FP8 architecture correlates with TTFT error**
  - Scout FP8 experiments show MIXED results (26-92% TTFT), not uniform high error
  - Scout short-sequence (roleplay 26%, codegen 58%) performs better than dense long-sequence (general-lite 77-92%)
- ❌ **REJECTED: Model size correlates with TTFT error**
  - Large models (Llama-3.1-70B, Yi-34B) show mixed results (28-78% TTFT)
  - Small models (Llama-2-7b, Qwen2.5-7b) show mixed results (8-84% TTFT)
  - No clear size correlation

### Root Cause Hypotheses

**Principle 1: Scout's bottleneck is sequence-length-dependent, NOT architecture-dependent (FP8)**

- **Evidence**:
  - Scout short-sequence (roleplay 26% TTFT, codegen 58% TTFT) improved significantly (-53pp, -34pp from iter8)
  - Scout long-sequence (general-lite 92% TTFT, reasoning-lite 91% TTFT) failed to improve (0pp, -8pp from iter8)
  - β₉ (FP8 dequantization) converged to zero (0.14 μs vs expected 17-50 μs)
  - **Inverse correlation**: Longer sequences → worse performance (opposite of FP8 hypothesis prediction)

- **Mechanism**: Long-sequence Scout requests face bottlenecks that short-sequence requests avoid:
  1. **Batching inefficiency**: Long sequences don't fit well in batches (fewer requests per batch → delayed processing → increased wait time → higher TTFT)
  2. **Scheduler overhead**: β₆ exploded (+654%, 13ms → 99ms), suggesting scheduler struggles to schedule long-sequence requests efficiently
  3. **Memory bandwidth saturation**: Long sequences require more memory bandwidth (activations, KV cache), potentially saturating GPU HBM bandwidth
  4. **KV cache management**: β₃ doubled (+118%, 4.4ms → 9.6ms), suggesting KV allocation/eviction overhead scales with sequence length

- **Action**: **iter10 MUST add sequence-length-dependent basis function(s)**:
  - **Option 1**: Add β₁₀ for batching inefficiency (per-request overhead scaling with sequence length): `β₁₀ × (prefillTokens^2 / batchSize)`
  - **Option 2**: Add β₁₀ for memory bandwidth saturation: `β₁₀ × max(0, prefillTokens - threshold) × numLayers` (overhead kicks in above threshold)
  - **Option 3**: Refine β₃ (KV mgmt) to include sequence-length dependency: `β₃ × (1 + prefillTokens / threshold)`
  - **Priority**: Investigate batching inefficiency first (most likely root cause given inverse correlation)

**Principle 2: Coefficient explosions reveal missing mechanisms (β₆ scheduler, β₂ TP comm, β₈ MoE routing)**

- **Evidence**:
  - β₆ (scheduler overhead): +654% (13ms → 99ms) — 7.5× increase, now 99ms per request
  - β₂ (TP communication): +343% (0.18 → 0.82) — 4.5× increase, dimensionless factor
  - β₈ (MoE routing): +143% (30μs → 73μs) — 2.4× increase, now ABOVE predicted 10-50μs range
  - These explosions absorbed the error that β₉ (FP8) was supposed to capture

- **Mechanism**: When a hypothesis is WRONG (β₉ = 0, FP8 not the bottleneck), the optimizer compensates by inflating OTHER coefficients to absorb residual error. The specific coefficients that inflated (β₆, β₂, β₈) reveal which mechanisms are ACTUALLY involved:
  1. **β₆ (scheduler)**: Scheduler overhead is genuinely higher for Scout long-sequence requests. May involve:
     - Scheduling delay due to priority/fairness policies
     - Difficulty scheduling long-sequence requests (lower batch efficiency → more scheduler invocations)
     - Overhead from preemption/reordering for long-running requests
  2. **β₂ (TP communication)**: TP overhead is genuinely higher than initially modeled. May involve:
     - Cross-GPU synchronization for Scout TP=2 MoE expert routing
     - All-reduce latency scaling with model size and TP degree
     - TP communication overhead for long sequences (more activations to transfer)
  3. **β₈ (MoE routing)**: MoE routing overhead is genuinely higher (73μs vs 30μs iter8, above 10-50μs predicted). May involve:
     - Expert routing overhead scales with sequence length (more tokens → more routing decisions)
     - TP=2 cross-GPU expert dispatch overhead (not fully captured by β₂)
     - Scout's 26 MoE layers × 16 experts = high routing complexity

- **Action**: **iter10 should investigate whether these explosions are real or spurious**:
  - **β₆ (scheduler)**: Profile vLLM scheduler overhead for long vs short sequences. If real, refine β₆ basis function to include sequence-length dependency.
  - **β₂ (TP comm)**: Profile TP all-reduce latency for long sequences. If real, refine β₂ basis function to include sequence-length scaling.
  - **β₈ (MoE routing)**: Validate Scout model config (InterleaveMoELayerStep=26, NumExpertsPerTok). If config correct, accept 73μs as physical value (update predicted range to 10-80μs).

**Principle 3: Partial coefficient reversion (2/3) indicates error redistribution, not resolution**

- **Evidence**:
  - β₅ (MoE gating): 41μs → 20μs (-52%), now within physical 10-20μs range ✓
  - β₇ (decode overhead): 26ms → 11ms (-58%), now within physical 10-20ms range ✓
  - β₃ (KV mgmt): 4.4ms → 9.6ms (+118%), inflated further (moving AWAY from physical 0.4-1ms range)

- **Mechanism**: The optimizer redistributed error from β₅ and β₇ to β₃, β₆, β₂, β₈. This is partial progress — β₅ and β₇ are now physically plausible, no longer absorbing spurious Scout error. However, β₃'s inflation (+118%) indicates a missing sequence-length-dependent KV mechanism:
  1. **β₃ (KV mgmt) inflation**: β₃ is per-request overhead. Long-sequence Scout requests (general-lite, reasoning-lite) have much larger KV caches (~400-600 tokens × 56 layers × key/value). KV management overhead (PagedAttention block allocation, eviction, GPU↔CPU offloading) may scale with KV cache size, not just number of requests.
  2. **Sequence-length dependency**: β₃ should scale with `prefillTokens` or `kvCacheBlocks`, not just constant per-request overhead.

- **Action**: **iter10 should refine β₃ basis function to include sequence-length dependency**:
  - **Current**: `β₃ × numRequests` (constant per-request overhead)
  - **Proposed**: `β₃ × (numRequests + β₃' × totalPrefillTokens)` (add sequence-length scaling component)
  - This allows β₃ to capture both base KV overhead (PagedAttention setup) and sequence-length-dependent overhead (block allocation scaling with KV size)

**Principle 4: Workload-agnostic constraint validated for non-Scout, violated for Scout**

- **Evidence**:
  - **Non-Scout dense models** (11 experiments): TTFT errors scale with sequence length across ALL workload types (general-lite, reasoning-lite, codegen, roleplay). No workload-specific patterns — only sequence-length patterns.
  - **Scout MoE+FP8 models** (4 experiments): TTFT errors show WORKLOAD-SPECIFIC patterns within same sequence-length category:
    - Scout general-lite (long, 92% TTFT) vs Scout reasoning-lite (long, 91% TTFT): Similar errors (sequence-length dominates)
    - But Scout codegen (moderate, 58% TTFT) improved significantly (-34pp) while dense codegen (76% TTFT) did not — workload-specific behavior?

- **Mechanism**: The workload-agnostic constraint assumes batch composition (prompt length distribution, decode length distribution) fully determines latency, regardless of workload labels. For dense models, this holds — general-lite and reasoning-lite (both long sequences) show similar errors. For Scout, the constraint may be violated if:
  1. **Workload-specific batching patterns**: General-lite vs reasoning-lite may have different batching characteristics (arrival patterns, batch sizes) beyond just sequence length.
  2. **Expert routing patterns**: If general-lite vs reasoning-lite trigger different expert routing patterns (e.g., general-lite uses more experts per token), this could cause workload-specific overhead.
  3. **Data collection artifacts**: New exp17 (general-lite) may have collection issues (e.g., cold cache, different server state).

- **Action**: **iter10 should investigate Scout workload-specific patterns**:
  - Validate Scout general-lite data collection quality (compare request-level latencies, check for outliers)
  - Profile Scout expert routing patterns (general-lite vs reasoning-lite — do they trigger different expert selections?)
  - If patterns are real, consider separate model for Scout (violates workload-agnostic constraint)

---

## Coefficient Analysis

**Alpha [α₀, α₁, α₂]** from `best_params.alpha`: [Fixed API overhead, per-input-token, per-output-token]

- **Iter9 optimal values**:
  - α₀ = 0.35ms (base overhead, -73.4% from iter8's 1.32ms)
  - α₁ = 65.0 μs/tok (input token, -44.7% from iter8's 117.6 μs/tok)
  - α₂ = 48.5 μs/tok (output token, -46.4% from iter8's 90.5 μs/tok)

- **Physical interpretation**: All three alpha coefficients DECREASED significantly (44-73%), suggesting the model is compensating for beta coefficient explosions by reducing alpha baseline overhead. This is a **RED FLAG** — alpha coefficients should be stable (represent physical API/tokenization overhead), not trade off against beta coefficients.

- **Analysis**:
  - **α₀ decrease** (-73%): Base API overhead decreased from 1.32ms to 0.35ms. This is suspiciously low — vLLM API overhead (HTTP parsing, request validation, queue insertion) is typically 1-3ms, not 0.35ms.
  - **α₁ and α₂ decrease** (-45%, -46%): Per-token overhead decreased from 118μs and 91μs to 65μs and 49μs. This is also low — tokenization + tensor copy overhead is typically 50-150μs per token.
  - **Trade-off dynamic**: The optimizer reduced alpha (baseline overhead) to compensate for beta explosions (β₆ +654%, β₂ +343%). This creates a zero-sum trade-off where Scout error is absorbed into beta terms at the expense of alpha terms, degrading non-Scout experiments.

- **Recommendation**: **iter10 should constrain alpha coefficients to prevent spurious reduction**. Add alpha bounds to keep α₀ ≥ 0.5ms, α₁ ≥ 50μs, α₂ ≥ 40μs (prevent unrealistic decreases).

**Beta [β₀, β₁, ..., β₉]** from `best_params.beta`: [Step-level basis functions]

- **β₀ (prefill compute factor)**: 0.1624 (dimensionless, -15.1% from iter8's 0.1912)
  - **Physical interpretation**: Prefill compute efficiency factor. Lower value = higher efficiency (less time per FLOP). Decrease suggests model is compensating for other overheads by assuming higher prefill efficiency.
  - **Concern**: Prefill compute should be stable (hardware-dependent). Decrease may indicate spurious trade-off.

- **β₁ (decode memory factor)**: 1.3611 (dimensionless, +22.9% from iter8's 1.1076)
  - **Physical interpretation**: Decode memory bandwidth efficiency factor. Higher value = lower efficiency (more time per byte). Increase suggests decode memory bottleneck is more severe than initially modeled.
  - **Plausible**: Decode is memory-bound, and β₁ > 1 indicates decode latency exceeds theoretical memory bandwidth limit (due to cache misses, memory controller overhead).

- **β₂ (TP communication factor)**: 0.8171 (dimensionless, +342.6% from iter8's 0.1846)
  - **Physical interpretation**: TP all-reduce efficiency factor. Higher value = higher overhead. 4.5× increase suggests TP communication overhead is much larger than initially modeled.
  - **EXPLOSION ALERT**: This is a **coefficient explosion** (+343%). Likely absorbing Scout long-sequence overhead (TP=2 cross-GPU coordination for Scout general-lite, reasoning-lite).
  - **Action**: Profile TP all-reduce latency for long sequences. If real, refine β₂ basis function to include sequence-length scaling.

- **β₃ (KV management overhead)**: 0.00959 seconds = 9.59ms per request (+117.7% from iter8's 4.40ms)
  - **Physical interpretation**: Per-request KV cache management overhead (PagedAttention block allocation, eviction, GPU↔CPU offloading). Doubled from iter8.
  - **INFLATION ALERT**: This is a **coefficient inflation** (+118%). Moving AWAY from physical 0.4-1ms range, suggesting missing sequence-length dependency.
  - **Action**: Refine β₃ basis function to include sequence-length scaling: `β₃ × (numRequests + β₃' × totalPrefillTokens)`.

- **β₄ (decode compute factor)**: 0.4658 (dimensionless, -34.7% from iter8's 0.7132)
  - **Physical interpretation**: Decode compute efficiency factor. Lower value = higher efficiency. Decrease suggests model is compensating for decode memory bottleneck (β₁ +22.9%) by assuming higher decode compute efficiency.
  - **Plausible**: Decode is memory-bound, not compute-bound, so decode compute factor decreasing is consistent.

- **β₅ (MoE gating overhead)**: 0.0000198 seconds = 19.8 μs per token (-51.9% from iter8's 41.1 μs)
  - **Physical interpretation**: Per-token MoE gating network overhead (expert selection FLOPs). Decreased from 41μs to 20μs, now within physical 10-20μs range ✓
  - **REVERSION SUCCESS**: This is a **coefficient reversion** (-52%), no longer absorbing spurious Scout error. Now physically plausible.

- **β₆ (scheduler overhead)**: 0.0993 seconds = 99.3ms per request (+654.4% from iter8's 13.2ms)
  - **Physical interpretation**: Per-request scheduler overhead (queue management, priority sorting, batch formation). Exploded 7.5× from iter8.
  - **EXPLOSION ALERT**: This is a **coefficient explosion** (+654%). Likely absorbing Scout long-sequence scheduling overhead (scheduler struggles to schedule long-sequence requests efficiently due to low batch efficiency).
  - **Action**: Profile vLLM scheduler overhead for long vs short sequences. If real, refine β₆ basis function to include sequence-length dependency or batch-efficiency penalty.

- **β₇ (decode overhead)**: 0.0110 seconds = 11.0ms per request (-58.0% from iter8's 26.3ms)
  - **Physical interpretation**: Per-request decode overhead (output processing, TP coordination, KV writeback). Decreased from 26ms to 11ms, now within physical 10-20ms range ✓
  - **REVERSION SUCCESS**: This is a **coefficient reversion** (-58%), no longer absorbing spurious Scout error. Now physically plausible.

- **β₈ (MoE routing overhead)**: 0.0000727 seconds = 72.7 μs per routed token (+142.6% from iter8's 30.0 μs)
  - **Physical interpretation**: Per-routed-token MoE expert routing overhead (expert dispatch, load balancing, aggregation). Increased 2.4× from iter8, now ABOVE predicted 10-50μs range.
  - **INFLATION ALERT**: This is a **coefficient inflation** (+143%). Now 73μs vs iter8's 30μs, above predicted 10-50μs range.
  - **Analysis**: Could be REAL (Scout MoE routing overhead higher than initially estimated) OR absorbing other Scout overhead (cross-GPU expert dispatch for TP=2).
  - **Action**: Validate Scout model config (InterleaveMoELayerStep=26, NumExpertsPerTok). If config correct, accept 73μs as physical value and update predicted range to 10-80μs.

- **β₉ (FP8 dequantization overhead)**: 0.0000001365 seconds = 0.1365 μs per token per layer (NEW, essentially ZERO)
  - **Physical interpretation**: Per-token per-layer FP8 dequantization overhead (weight conversion FP8 → FP16/BF16, mixed-precision coordination, scale management). Expected 17-50 μs, actual 0.14 μs (124-366× too small).
  - **REJECTION**: This is **coefficient rejection** — β₉ is essentially zero, meaning FP8 dequantization overhead is negligible or already captured by roofline MFU.
  - **Action**: Remove β₉ term from iter10 (mechanism not active). Focus on sequence-length-dependent terms instead.

**Redundant terms**: β₉ should be removed (zero coefficient, mechanism not active).

**Missing physics**:
1. **Sequence-length-dependent batching inefficiency**: Long sequences → lower batch efficiency → increased scheduler overhead + wait time
2. **Sequence-length-dependent KV management**: β₃ should scale with prefill tokens, not just number of requests
3. **Sequence-length-dependent TP communication**: β₂ may need to scale with sequence length (more activations to transfer)

---

## Recommendations for iter10

### Priority 1: Critical Issues — Add Sequence-Length-Dependent Basis Function(s)

**Action 1: Add β₁₀ for batching inefficiency (HIGHEST PRIORITY)**

**Rationale**: Scout short-sequence (roleplay, codegen) improved significantly (-53pp, -34pp), while Scout long-sequence (general-lite, reasoning-lite) failed completely (0pp, -8pp). This inverse correlation strongly suggests **batching inefficiency** as the root cause:
- Long sequences → fewer requests fit in batch → lower GPU utilization → increased wait time → higher TTFT
- Short sequences → more requests fit in batch → higher GPU utilization → reduced wait time → lower TTFT

**Proposed basis function**:
```
β₁₀ × Σ(prefillTokens_i^2 / batchSize)
```
Where:
- `prefillTokens_i^2`: Quadratic penalty (long sequences have disproportionate impact on batch efficiency)
- `batchSize`: Actual batch size at step execution time (normalizer)
- This term captures the overhead of scheduling long-sequence requests that don't fit well in batches

**Expected β₁₀ coefficient**: 0.1-1.0 μs per (token²/batch) (calibrate against Scout long-sequence overhead ~40-60ms per request)

**Validation**: After iter10, check if:
- Scout long-sequence TTFT improves to <50% (general-lite, reasoning-lite)
- β₆ (scheduler overhead) decreases from 99ms toward 20-30ms (offloads batching inefficiency to β₁₀)
- Coefficient is physically plausible (0.1-1.0 μs range)

**Action 2: Refine β₃ (KV management) to include sequence-length dependency**

**Rationale**: β₃ doubled from 4.4ms to 9.6ms (+118%), moving AWAY from physical 0.4-1ms range. This suggests β₃ is absorbing sequence-length-dependent KV overhead (long sequences have larger KV caches → more block allocation/eviction overhead).

**Proposed basis function refinement**:
```
β₃ × numRequests + β₃' × Σ(prefillTokens_i × numLayers)
```
Where:
- `β₃ × numRequests`: Base per-request KV overhead (PagedAttention setup, ~0.4-1ms)
- `β₃' × Σ(prefillTokens_i × numLayers)`: Sequence-length-dependent KV overhead (block allocation scaling with KV cache size)

**Expected coefficients**:
- β₃ = 0.4-1.0ms per request (base overhead)
- β₃' = 0.1-1.0 μs per (token × layer) (KV block allocation overhead)

**Validation**: After iter10, check if:
- β₃ reverts to 0.4-1ms range (base overhead)
- β₃' captures sequence-length-dependent overhead (0.1-1.0 μs per token×layer)
- Long-sequence experiments improve (KV overhead now modeled correctly)

**Action 3: Remove β₉ (FP8 dequantization) — mechanism not active**

**Rationale**: β₉ converged to zero (0.14 μs vs expected 17-50 μs), proving FP8 dequantization is NOT Scout's bottleneck. Remove this term to reduce optimization dimensionality.

**Implementation**: Delete β₉ term from `sim/latency/evolved_model.go`, update `coefficient_bounds.yaml` to remove β₉ bounds.

### Priority 2: Improvements — Investigate and Constrain Coefficient Explosions

**Action 4: Profile β₆ (scheduler overhead) for long vs short sequences**

**Rationale**: β₆ exploded from 13ms to 99ms (+654%), suggesting scheduler struggles with long-sequence requests. Profile vLLM scheduler to determine if this is:
1. **Real overhead**: Scheduler genuinely spends 99ms per request on queue management, priority sorting, batch formation for long sequences.
2. **Spurious absorption**: β₆ is absorbing other long-sequence overhead (batching inefficiency, memory pressure).

**Profiling approach**:
```bash
# Profile vLLM scheduler overhead for Scout general-lite vs Scout roleplay
vllm_profiler --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic \
  --workload general-lite --workload roleplay --profile-scheduler
```

**Outcome**:
- If real: Refine β₆ basis function to include sequence-length dependency or batch-efficiency penalty.
- If spurious: β₆ will decrease after adding β₁₀ (batching inefficiency term offloads this overhead).

**Action 5: Profile β₂ (TP communication) for long vs short sequences**

**Rationale**: β₂ exploded from 0.18 to 0.82 (+343%), suggesting TP communication overhead is larger than initially modeled. Profile TP all-reduce latency to determine if this is:
1. **Real overhead**: TP all-reduce genuinely takes longer for long sequences (more activations to transfer).
2. **Spurious absorption**: β₂ is absorbing Scout TP=2 cross-GPU expert routing overhead (not captured by β₈).

**Profiling approach**:
```bash
# Profile TP all-reduce latency for long vs short sequences
torch_profiler --model meta-llama/Llama-3.1-70B-Instruct --tp 4 \
  --seq-lengths 100,200,400,600 --profile-all-reduce
```

**Outcome**:
- If real: Refine β₂ basis function to include sequence-length scaling: `β₂ × (1 + prefillTokens / threshold)`.
- If spurious: β₂ will decrease after fixing Scout bottleneck (batching inefficiency).

**Action 6: Validate Scout model config and β₈ (MoE routing) inflation**

**Rationale**: β₈ increased from 30μs to 73μs (+143%), now ABOVE predicted 10-50μs range. Validate Scout model config to ensure basis function is correct:
- `num_local_experts` = 16 ✓
- `num_experts_per_tok` = 1 or 2 (affects routed token count)
- `interleave_moe_layer_step` = 26 ✓ (26 MoE layers, 30 dense layers)

**Validation approach**:
```bash
# Check Scout HuggingFace config
curl https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic/resolve/main/config.json \
  | jq '.num_local_experts, .num_experts_per_tok, .moe_layer_indices'
```

**Outcome**:
- If config correct: Accept 73μs as physical value, update predicted range to 10-80μs.
- If config wrong: Fix basis function and re-run iter9 (β₈ may have been underestimating routed tokens).

**Action 7: Constrain alpha coefficients to prevent spurious reduction**

**Rationale**: All three alpha coefficients decreased significantly (44-73%), likely compensating for beta explosions. This creates zero-sum trade-offs. Constrain alpha bounds to keep:
- α₀ ≥ 0.5ms (API overhead lower bound)
- α₁ ≥ 50 μs/tok (tokenization overhead lower bound)
- α₂ ≥ 40 μs/tok (output processing overhead lower bound)

**Implementation**: Update `coefficient_bounds.yaml` to add alpha lower bounds.

### Priority 3: Refinements — Data Quality and Workload-Agnostic Validation

**Action 8: Validate Scout general-lite (exp17) data collection quality**

**Rationale**: New exp17 (general-lite-2-1) still shows 92% TTFT (worst Scout experiment), despite replacing saturated data. Investigate:
1. Request-level latency distribution (check for outliers, cold cache effects)
2. Server state during collection (CPU/GPU utilization, memory pressure)
3. Workload intensity (compare to Scout reasoning-lite — similar sequence length but 91% TTFT)

**Validation approach**:
```bash
# Analyze exp17 request-level latencies
python scripts/analyze_experiment_latencies.py \
  --experiment trainval_data/17-llama-4-scout-17b-16e-tp2-general-lite-2-1 \
  --compare trainval_data/48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1
```

**Outcome**:
- If outliers found: Re-collect exp17 under controlled conditions.
- If no outliers: Confirms general-lite workload has unique bottleneck (investigate workload-specific patterns).

**Action 9: Profile Scout expert routing patterns (general-lite vs reasoning-lite)**

**Rationale**: Scout general-lite (92% TTFT) and Scout reasoning-lite (91% TTFT) both failed despite similar sequence lengths. If workload-specific patterns exist (e.g., general-lite triggers more experts per token), this violates the workload-agnostic constraint.

**Profiling approach**:
```bash
# Profile Scout expert routing patterns
vllm_profiler --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-FP8-dynamic \
  --workload general-lite --workload reasoning-lite --profile-expert-routing \
  --output expert_routing_patterns.json
```

**Outcome**:
- If patterns differ: Violates workload-agnostic constraint — consider separate model for Scout.
- If patterns same: Confirms batching inefficiency as root cause (both workloads fail due to long sequences).

**Action 10: Consider architecture-specific models if Scout continues to fail**

**Rationale**: If iter10 (with β₁₀ batching inefficiency term) fails to improve Scout long-sequence experiments below 60% TTFT, consider splitting:
- **Dense model**: Trained on 11 non-Scout experiments (excludes FP8 + MoE complexity)
- **Scout model**: Trained on 4 Scout experiments + future MoE+FP8 models (allows Scout-specific tuning)

**Contingency**: Only pursue if iter10 fails (loss >120%, Scout long-sequence >70% TTFT).

---

## Basis Function Changes for iter10

**Remove**:
- **β₉ (FP8 dequantization)**: Converged to zero, mechanism not active. Remove term.

**Add**:
- **β₁₀ (batching inefficiency)**: `β₁₀ × Σ(prefillTokens_i^2 / batchSize)` — captures overhead of long sequences that don't fit well in batches.

**Modify**:
- **β₃ (KV management)**: Split into base overhead + sequence-length-dependent component:
  - `β₃ × numRequests + β₃' × Σ(prefillTokens_i × numLayers)`
  - This adds one new coefficient (β₃') while refining β₃ semantics.

**Total coefficient count**: 10 → 10 (remove β₉, add β₁₀, add β₃' — net 10 coefficients)

---

## Bounds Adjustments for iter10

**Alpha bounds** (add lower bounds to prevent spurious reduction):
- α₀: [0.5ms, 5.0ms] (was [0.1ms, 10.0ms]) — prevent unrealistic decrease
- α₁: [50μs, 300μs] (was [10μs, 500μs]) — prevent unrealistic decrease
- α₂: [40μs, 250μs] (was [10μs, 500μs]) — prevent unrealistic decrease

**Beta bounds** (adjust based on iter9 learnings):
- β₀ (prefill compute): [0.10, 0.30] (tighten around 0.16 optimal)
- β₁ (decode memory): [1.0, 1.5] (tighten around 1.36 optimal)
- β₂ (TP comm): [0.15, 1.2] (expand upper bound, converged to 0.82)
- β₃ (KV mgmt base): [0.0004, 0.002] seconds = [0.4ms, 2.0ms] (target physical range)
- β₃' (KV mgmt seq-len): [0.0000001, 0.000001] seconds = [0.1μs, 1.0μs] per (token×layer)
- β₄ (decode compute): [0.30, 0.80] (tighten around 0.47 optimal)
- β₅ (MoE gating): [0.00001, 0.00003] seconds = [10μs, 30μs] (tighten around 20μs optimal)
- β₆ (scheduler): [0.010, 0.040] seconds = [10ms, 40ms] (reduce upper bound, expect decrease after β₁₀ addition)
- β₇ (decode overhead): [0.005, 0.025] seconds = [5ms, 25ms] (tighten around 11ms optimal)
- β₈ (MoE routing): [0.00001, 0.00010] seconds = [10μs, 100μs] (expand upper bound, converged to 73μs)
- β₁₀ (batching inefficiency): [0.0000001, 0.000001] seconds = [0.1μs, 1.0μs] per (token²/batch) (NEW)

**Remove**: β₉ bounds (FP8 dequantization removed)

---

## Expected Iter10 Outcomes

**Success Criteria (Tier 1)**:
- Overall loss <80% (TTFT RMSE <40%, E2E RMSE <50%)
- Scout long-sequence (general-lite, reasoning-lite) <50% TTFT (currently 92%, 91%)
- Scout short-sequence (roleplay, codegen) remain <30%, <60% TTFT (currently 26%, 58%)
- β₁₀ (batching inefficiency) = 0.1-1.0 μs per (token²/batch) (physically plausible)
- β₆ (scheduler) decreases from 99ms to 20-30ms (batching inefficiency offloaded to β₁₀)
- β₃ and β₃' both physically plausible (β₃ = 0.4-1ms, β₃' = 0.1-1.0 μs)
- All coefficients within physical ranges (no explosions)

**Success Criteria (Tier 2 — Partial)**:
- Overall loss <100% (significant improvement from iter9's 161%)
- Scout long-sequence <70% TTFT (>20pp improvement)
- β₁₀ and β₃' coefficients plausible
- At least 2/3 coefficient explosions (β₆, β₂, β₈) decrease
- **Proceed to iter11** with additional Scout terms if needed

**Failure Criteria (Tier 3)**:
- Overall loss >130% (minimal improvement)
- Scout long-sequence >80% TTFT (<10pp improvement)
- β₁₀ converged to zero (batching inefficiency hypothesis rejected)
- **Diagnostic**: Consider architecture-specific models (separate model for Scout)
