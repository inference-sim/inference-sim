# Iteration 16: Dimensionality Reduction and Stable Basin Recovery

## H-main: Dimensionality Reduction Enables Optimizer Convergence

**Prediction**: Overall loss will decrease from 6538% (iter15) to **<1500%** (≥4× improvement), with:
- TTFT RMSE: 2099% → <700%
- E2E RMSE: 4439% → <1000%
- At least 10/15 experiments achieve TTFT APE < 100% (vs 1/15 in iter15)
- At least 3/15 experiments achieve TTFT APE < 50% (vs 1/15 in iter15)

**Causal Mechanism**:

Iter15 failed catastrophically (loss 6538%, 182% worse than iter14) because it attempted to solve three problems simultaneously with a 10-dimensional cold-start search:
1. Decode amplification (β₁: 5-15×, β₄: 3-8×)
2. MoE non-compute overhead (β₈, NEW)
3. Dense prefill batching (β₉, NEW)

The optimizer **rejected the strategy** — 5/10 coefficients collapsed to effectively zero:
- β₃ (KV mgmt) = 0.001 (1500× too small)
- β₆ (scheduler) = 0.042 (1000× too small)
- β₇ (decode overhead) = 0.016 (1000× too small)
- β₈ (MoE non-compute) = 0.000037 (270,000× too small)
- β₉ (prefill batching) = 7.2e-07 (2,800,000× too small)

**Why iter16 will succeed**:

1. **5D search space** (remove collapsed β₃, β₆, β₇, β₈, β₉):
   - Sample efficiency: 1000 trials in 5D = 200 trials/dimension (same as iter15's 2000 trials in 10D)
   - Optimization surface: Fewer local minima, faster convergence
   - Physics validity: All 5 remaining coefficients (β₀, β₁, β₂, β₄, β₅) were USED by iter15 (not rejected)

2. **Warm-start from iter9** (161% loss, same 15-experiment dataset):
   - Iter9 was the LAST STABLE state before cascade (iter10: 161% → 4267%, 26× worse)
   - Iter9 coefficients are physically plausible (no explosions yet)
   - Iter15 ignored iter9, compared cold-start to iter7 (different dataset) → strategic error
   - Starting in proven basin (161% loss) vs random basin (6538% loss) → 40× difference

3. **Confirmed mechanisms**:
   - **β₁, β₄ decode amplification**: Iter15 showed β₁=6.4, β₄=6.5 helps decode-heavy workloads (reasoning-lite E2E: 75-180% vs iter14's 100% timeout)
   - **β₅ MoE gating**: Iter14-15 showed β₅=20-50 captures gating overhead (iter14 fixed layer multiplier bug)
   - **β₂ TP comm**: Stable across iterations (0.15-0.25 range)
   - **β₀ prefill**: Though roofline-based formula has issues, β₀ scaling is necessary (iter15 β₀=0.092 provided 11× scale-down)

4. **Remove wrong physics**:
   - **β₈ (MoE non-compute)** was fundamentally wrong: MoE routing overhead is NOT a separate per-token term (already captured by β₅ gating or negligible)
   - **β₉ (prefill batching)** was fundamentally wrong: Dense overestimation is NOT due to batch heterogeneity (dense roleplay has WORSE errors than dense codegen, opposite of hypothesis)
   - **β₃, β₆, β₇** were negligible: Fixed overheads (0.4-100ms) are tiny compared to GPU execution time (seconds) → optimizer correctly pushed to zero

**Code Citations**:
- **Iter15 optimizer rejection**: `training/iterations/iter15/inner_loop_results.json` → β₃,₆,₇,₈,₉ all collapsed
- **Iter9 stability**: `training/iterations/iter9/inner_loop_results.json` → 161% loss, physically plausible coefficients
- **Decode amplification**: `training/iterations/iter15/iter15-FINDINGS.md` L159-163 → reasoning-lite E2E improved with β₁=6.4, β₄=6.5
- **5D vs 10D curse**: iter15 findings L368-391 → 2000 trials in 10D insufficient

**Diagnostic Clause**: *If this fails (loss remains >2000%), it indicates:*
1. *Roofline basis functions are fundamentally broken (not just dimensional curse) → Need empirical prefill model (iter17)*
2. *Warm-start from iter9 is incompatible with current amplified decode bounds → Re-tune bounds or try different initialization*
3. *5 remaining coefficients insufficient to model prefill/decode/MoE → Add NEW physically-grounded terms (not β₈,₉)*

---

## H-ablation-terms: Which Collapsed Terms Can Be Safely Removed?

**Prediction**: All 5 collapsed terms (β₃, β₆, β₇, β₈, β₉) can be removed without increasing loss, because iter15 optimizer already pushed them to effectively zero.

**Causal Mechanism**:

The optimizer's coefficient values reveal which terms capture real physics:
- **β₃ = 0.001** (expected 0.4-1.5ms): KV management overhead is 1500× too small → either negligible or already captured by other terms (β₀ prefill includes KV write, β₁ decode includes KV read)
- **β₆ = 0.042** (expected 40-100ms): Scheduler overhead is 1000× too small → vLLM scheduler overhead is fast (<1ms per request), not 40-100ms as initially hypothesized
- **β₇ = 0.016** (expected 15-30ms): Decode per-request overhead is 1000× too small → output processing/TP coordination are fast, not 15-30ms
- **β₈ = 0.000037** (expected 10-40 μs): MoE non-compute overhead is 270,000× too small → routing latency is negligible or already in β₅ (gating FLOPs capture routing cost)
- **β₉ = 7.2e-07** (expected 0.5-2.0 μs): Prefill batching penalty is 2,800,000× too small → batch heterogeneity does NOT explain dense overestimation (wrong hypothesis)

**Evidence**:
- Iter15 used 2000 trials with 10D search space → optimizer had sufficient exploration to find these terms if they mattered
- Optimizer converged without errors (no timeouts, numerical stability achieved) → but chose to reject 5 terms
- The 5 USED terms (β₀, β₁, β₂, β₄, β₅) all converged to physically plausible values (β₀=0.092, β₁=6.4, β₂=0.21, β₄=6.5, β₅=33.6)

**Diagnostic Clause**: *If removing terms increases loss, it indicates:*
- *Iter15 optimizer failed to converge → Need more than 2000 trials in 10D, or better initialization*
- *Collapsed terms were victims of coefficient interactions → Try removing terms incrementally (iter16 removes all 5 at once)*

---

## H-boundary: Where Should Decode Amplification Apply?

**Prediction**:
- **Decode-heavy workloads** (output tokens >> input tokens): E2E APE will be <200% (decode amplification prevents underestimation)
- **Prefill-heavy workloads** (input tokens >> output tokens): TTFT APE will remain >500% (decode amplification doesn't fix prefill errors)
- **Balanced workloads** (input ≈ output tokens): Both TTFT and E2E APE will be in 200-500% range

**Causal Mechanism**:

Iter15 showed decode amplification (β₁=6.4, β₄=6.5) has **directional correctness** but **limited scope**:

**Where it works** (reasoning-lite experiments):
- **Reasoning-lite**: 512 input tokens, 256-512 output tokens → decode is 60-80% of E2E latency
- **E2E APE results**: 75-180% (iter15) vs 100% timeout (iter14) → decode amplification prevents underestimation ✓
- **Why**: Long output sequences mean many decode steps → β₁, β₄ amplification accumulates → prevents predicted latency from being too small

**Where it fails** (dense roleplay experiments):
- **Roleplay**: 64 input tokens, 128 output tokens → prefill is still significant (40-60% of E2E)
- **TTFT APE results**: 4124-4151% (catastrophically high) ❌
- **E2E APE results**: 9711-11357% (even worse than TTFT) ❌
- **Why**: TTFT measures prefill latency (not decode), so decode amplification can't help. E2E errors are worse because both prefill AND decode are wrong (prefill underestimated 40×, decode overcorrected?).

**Where it partially works** (Scout balanced experiments):
- **Scout reasoning-lite (exp_48)**: 512 input, 256-512 output → balanced
- **TTFT APE**: 30%, **E2E APE**: 27% → BOTH low (unique success) ✓
- **Why**: Balanced workload means prefill and decode errors can partially cancel out

**Diagnostic Clause**: *If decode-heavy workloads have E2E APE >300%, it indicates:*
- *β₁, β₄ amplification is too strong (overcorrecting decode) → Reduce bounds to 3-10× instead of 5-15×*
- *Decode functional form is wrong (not just roofline × constant) → Add batch-size dependence or logarithmic scaling*

---

## H-error-pattern: Which Experiments Should Improve Most?

**Prediction**:
1. **Dense small-model experiments** (Llama-2-7b, Qwen-7b, Mistral-12b) will improve 2-4× in TTFT APE (currently 1300-4000%)
2. **Scout MoE experiments** will improve 1.5-2× in TTFT APE (currently 708-1634%)
3. **Large model TP=4 experiments** (Llama-3.1-70B, Yi-34B) will improve 1.5-3× in TTFT APE (currently 280-956%)

**Reasoning**:
- **Dense small models**: Most affected by 10D dimensional curse + cold-start. Warm-start from iter9 + 5D will help most because iter9 performed well on these experiments (roleplay 21-26% TTFT, codegen 76% TTFT).
- **Scout MoE**: Removing β₈ (wrong MoE hypothesis) will eliminate interference. Iter9 had β₅ (gating) at reasonable value (20μs), should help.
- **Large models**: TP communication (β₂) and massive parameter counts make these experiments sensitive to initialization. Warm-start will help but less dramatically.

**Evidence from iter9**:
- Dense roleplay: 8-26% TTFT (excellent) ✓
- Dense codegen: 26-76% TTFT (moderate) ✓
- Scout short-sequence: 26-58% TTFT (good) ✓
- Large model codegen: 28% TTFT (good) ✓

**Diagnostic Clause**: *If dense small-model experiments don't improve to <1000% TTFT, it indicates:*
- *Warm-start from iter9 is corrupted or incompatible with current bounds → Try iter7 warm-start instead*
- *Roofline prefill formula is so broken that no amount of scaling/simplification helps → Must replace with empirical model (iter17)*

---

## H-robustness: Will Coefficients Stay in Physical Ranges?

**Prediction**: All 5 remaining coefficients (β₀, β₁, β₂, β₄, β₅) will stay within their physical bounds, with no explosions or collapses:
- β₀: 0.05-0.25 (prefill MFU scaling)
- β₁: 5.0-15.0 (decode memory MFU, amplified)
- β₂: 0.15-0.25 (TP communication)
- β₄: 3.0-8.0 (decode compute MFU, amplified)
- β₅: 20-50 (MoE gating efficiency)

**Causal Mechanism**:

Iter15's coefficient explosions/collapses occurred because:
1. **Wrong physics hypotheses** (β₈, β₉) forced optimizer to compensate via other coefficients
2. **10D search space** made optimization inefficient, causing random walk in high-error regions
3. **Cold-start initialization** started optimizer in bad basin (random uniform sampling)

Iter16 fixes all three issues:
1. **Removed wrong terms**: β₃, β₆, β₇, β₈, β₉ all eliminated → no interference
2. **5D search space**: Optimizer can efficiently explore 5D surface with 1000 trials (200 trials/dim)
3. **Warm-start from iter9**: Optimizer starts near physically plausible basin (β₀=0.162, β₁=1.361, β₂=0.817, β₄=0.466, β₅=0.020)

**Evidence**:
- Iter15 used coefficients β₀=0.092, β₁=6.4, β₂=0.21, β₄=6.5, β₅=33.6 → all within bounds ✓
- Iter9 had β₀=0.162, β₁=1.361, β₂=0.817, β₄=0.466, β₅=0.020 → all within iter9's bounds ✓
- No coefficient explosions expected because we removed the problematic terms

**Diagnostic Clause**: *If coefficients explode or collapse, it indicates:*
- *Bounds are mismatched to warm-start initialization → Widen bounds to allow more exploration*
- *Coefficient interactions exist even in 5D → Add regularization penalty (L2 on coefficient magnitudes) to prevent explosions*
- *Warm-start basin is unstable → Try cold-start with physics-based midpoints instead*
