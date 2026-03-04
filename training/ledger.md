# Strategy Evolution Ledger — Learned Latency Model

| Iter | Strategy | Form | Params | llama-2-7b Step MAPE | llama-2-70b Step MAPE | mixtral-8x7b Step MAPE | codellama-34b Step MAPE | Key Insight | H-main Status |
|------|----------|------|--------|-----|-----|-----|-----|-------------|---------------|
| 0 | Per-model OLS | β₀+β₁·pf+β₂·dc | 3/model | 120.7% (wt:37.6%, R²=0.91) | 102.6% (wt:42.0%, R²=0.75) | 24.1% (wt:20.0%, R²=0.39) | 122.3% (wt:53.8%, R²=0.54) | Negative intercept → catastrophic short-step overprediction for dense models. Per-token cost is NOT constant — depends on batch utilization. Mixtral only model where form works. D4: batch-size r<0.1 (not a confound). D2: 42-79% monotonicity violations (noise floor). D6: strong autocorrelation (0.43-0.83) except Mixtral. | REFUTED (3/4 models) |
| 1* | Log-transform (EXPLORATORY) | log(dur)=β₀+β₁·log(pf+1)+β₂·log(dc+1) | 3/model | 54.0% | 70.1% | 20.4% | 81.0% | *Exploratory — no review gates.* Log drops MAPE 32-67pp. D4 batch r=0.61 in log space. 0-200us band is irreducible noise floor. | EXPLORATORY |
| 2* | Separate pf/dc (EXPLORATORY) | decode: log(dc); prefill: log(pf,dc,bs) | 2-5/model | 52.3% | 68.3% | 19.7% | 80.3% | *Exploratory — no review gates.* Only 0.7-1.7pp over Iter 1. TDE 0.8-14.5%. Near stopping criterion. | EXPLORATORY |
| 1 | Cross-model Huber+interaction | log(dur)=β·X, 3 arch feats, interaction, Huber, DES-wt | 8 (cross-model) | TDE=158.4% | TDE=7.3% | TDE=91.9% | TDE=25.3% | **REFUTED.** Cross-model DRAMATICALLY worse than Iter 0 per-model linear (TDE 0-10%). VIF>2000 for dc/batch (collinearity crisis). DES-weighting HURTS (concentrates on large steps). LOMO 67-313% (no generalization). Architecture features memorize 4 model means, don't generalize. **Key finding: TDE is the right metric, not MAPE. Per-model linear OLS with 0-10% TDE is the practical optimum.** | REFUTED |
| 2 | Journey-constrained NNLS + separate α | β₀+β₁·pf+β₂·dc (NNLS, β≥0), direct steps + journey constraints, separate α_ttft/α_e2e | 3β + 3α_ttft + 3α_e2e /model | TRAIN: TTFT=18.2%, E2E=1.9%; TEST: TTFT=1.5%, E2E=8.6% | TRAIN: TTFT=5.4%, E2E=1.1%; TEST: TTFT=1.4%, E2E=6.3% | TRAIN: TTFT=4.6%, E2E=1.4% | TRAIN: TTFT=3.8%, E2E=0.6%; TEST: TTFT=1.6%, E2E=10.9% | **H-main CONFIRMED.** Train: TTFT 3.8-18.2%, E2E 0.6-1.9% (all pass 25%/20%). Test: TTFT 1.4-1.6%, E2E 6.3-10.9%. Separate α (fit from TTFT residuals, not E2E) fixes 34pp TTFT regression. Journey constraints essential (5K-12K% without). Physics features structurally correct but need >4 envs. **Key: journey step indices → single NNLS; separate α_ttft/α_e2e prevents decode error leaking into TTFT.** | CONFIRMED |
| 3 | Physics-first global β/α/γ | β₀·L+β₁·dc·kv_dim+β₂·(pf+dc)·MoE+β₃·I(TP>1), α₀+α₁·ν_in, γ₀+γ₁·ν_out | 4β + 2α + 2γ (8 total, ALL global) | TRAIN: TTFT=17.6%, E2E=10.6%; TEST: TTFT=1.5%, E2E=7.4% | TRAIN: TTFT=11.8%, E2E=8.0%; TEST: TTFT=1.6%, E2E=5.8% | TRAIN: TTFT=16.4%, E2E=8.9% | TRAIN: TTFT=5.3%, E2E=10.0%; TEST: TTFT=1.8%, E2E=9.0% | **PARTIAL.** 8 truly global parameters (no per-model fitting). TTFT 5.3-17.6% (all pass 25% gate). E2E 8.0-10.6% (vs Iter 2's 0.6-1.9%). γ₁=861 µs/tok (170× real detokenization) — diagnostic: global β has ~0.86ms/tok average decode error. NNLS zeros compute feature in all 12 designs tested — KV bandwidth dominates. LOMO: codellama/mixtral generalize (3.8-4.1% TTFT held-out); 7b degrades (17.3%). TTFT signed bias reveals the model-level residual: codellama/70b underpredicted (-5 to -9%), 7b/mixtral overpredicted (+7 to +13%). **Key: 4 envs insufficient for accurate global β; need >8 models to separate compute from KV bandwidth and fit TP continuously.** | PARTIAL (TTFT pass, E2E degraded) |
| H30-H32 | BLIS replay diagnostic | Iter 3 coefficients through BLIS simulator | same 8 | BLIS TTFT RE: -17% to -56% (both BB and CM identical) | — | — | — | **DIAGNOSTIC.** Both backends produce same TTFT underprediction → error is in BLIS scheduling model (zero inter-step overhead), not coefficients. Analytical "1.5% test TTFT" was T_queue_obs masking. Saturation: +22% capacity overestimate → regime transition (ρ 1.24→1.02, 2000× TTFT). γ₁ accidentally compensates. **Key: dominant error is missing δ (inter-step overhead), not β/α inaccuracy.** | DIAGNOSTIC |

## Iteration 3 Full Results

### Strategy: Physics-First Truly Global β/α/γ

**Goal:** 8 truly global parameters (no per-model fitting) that predict step time, TTFT, and E2E for any model from its config.json alone.

**Feature selection rationale (GPU/vLLM ops):**

| Feature | β coefficient | Physical mechanism | GPU operation |
|---------|--------------|-------------------|---------------|
| `L` (num_layers) | 116.1 µs/layer | Per-layer CUDA kernel dispatch overhead. Each transformer layer dispatches 4-6 kernels (QKV proj, attention, FFN, layernorm). Also captures sequential weight loading — weights read layer-by-layer from HBM. | `cudaLaunchKernel` × 4-6 per layer |
| `dc × L × kv_heads × head_dim / TP` | 1226.9 µs/kv_unit | KV cache bandwidth. Each decode token computes attention by reading its accumulated K/V cache entries from HBM. Decode is MEMORY-BANDWIDTH BOUND on H100 (arithmetic intensity ~4 FLOPs/byte, well below roofline knee at ~295 FLOPs/byte). | `flash_attn` KV read from HBM |
| `(pf+dc) × I(MoE)` | 19.9 µs/MoE_token | MoE expert routing + dispatch/gather. Per-token: softmax top-k selection + all-to-all expert dispatch. NOT captured by compute feature because MoE sparsity means only k/N of FFN weights are active. | `moe_align_block_size` + expert scatter/gather |
| `I(TP > 1)` | 9445.2 µs/step | Fixed TP synchronization barrier. NCCL all-reduce after attention + FFN in each layer has a fixed latency component (~100-300 µs × L layers per step). Per-STEP, not per-token. | `ncclAllReduce` per layer × L |

**Why compute (FLOPs proxy) is excluded:** NNLS consistently zeros it across all 12 feature sets tested. On H100 with FLASH_ATTN, prefill compute overlaps with memory access through GPU pipelining. The per-layer feature (L) captures the depth-dependent cost, and KV bandwidth captures the per-token memory cost. With only 4 models, compute and L×H are collinear — the system can't separate them.

### Fitted Coefficients (all global — no per-model fitting)

**Global β (4 coefficients, step time):**

| Feature | Scale | β value | Unit |
|---------|-------|---------|------|
| L | per_model_constant | 116.110 | µs/layer |
| dc × kv_dim × 1e-6 | per_step_variable | 1226.868 | µs/kv_unit |
| (pf+dc) × I(MoE) | per_step_variable | 19.943 | µs/MoE_tok |
| I(TP > 1) | per_model_constant | 9445.157 | µs/step |

**Global α (2 coefficients, pre-scheduling):**

| Coeff | Value | Physical meaning |
|-------|-------|-----------------|
| α₀ | 13,732 µs | Fixed CPU overhead: tokenization setup, request validation, HTTP parsing. Model-independent (same SentencePiece 32K tokenizer for all 4 models). |
| α₁ | 0.000 µs/tok | Per-token tokenization cost below noise floor at this scale. |

**Global γ (2 coefficients, post-completion):**

| Coeff | Value | Physical meaning | Diagnostic |
|-------|-------|-----------------|------------|
| γ₀ | 0 µs | No fixed post-completion overhead | |
| γ₁ | 860.6 µs/tok | **170× real detokenization cost (~5 µs/tok)**. This is a β diagnostic: the global β has ~0.86ms/tok average decode error across models, and γ absorbs it. | ✗ Compensating β |

### Results Across All Splits (truly global — 8 parameters total)

| Split | Model | TTFT MAPE | E2E MAPE | TTFT bias | E2E bias | n |
|-------|-------|----------|---------|-----------|----------|---|
| **TRAIN** | llama-2-7b | **17.6%** | 10.6% | +7.4% | -1.6% | 33,000 |
| **TRAIN** | llama-2-70b | **11.8%** | 8.0% | -9.4% | +7.1% | 33,000 |
| **TRAIN** | mixtral-8x7b | **16.4%** | 8.9% | +12.9% | -0.4% | 32,658 |
| **TRAIN** | codellama-34b | **5.3%** | 10.0% | -5.0% | +9.8% | 16,800 |
| **VALIDATE** | codellama-34b | 2.4% | 13.5% | -0.8% | +13.4% | 16,200 |
| **VALIDATE** | mixtral-8x7b | **0.5%** | 8.5% | 0.0% | -6.0% | 4,800 |
| **TEST** | llama-2-7b | **1.5%** | 7.4% | -0.1% | -5.4% | 2,353 |
| **TEST** | llama-2-70b | **1.6%** | 5.8% | -0.0% | -4.3% | 4,800 |
| **TEST** | codellama-34b | **1.8%** | 9.0% | -0.0% | -6.5% | 4,800 |

**TTFT signed bias pattern reveals model-level β residual:** codellama/70b are underpredicted (-5 to -9%), while 7b/mixtral are overpredicted (+7 to +13%). The global α₀=13.7ms splits the difference. With per-model α, each model's bias zeroes out — but that's per-model fitting in disguise.

### Per-model α/γ ablation (for comparison only)

When α and γ are fitted per-model (NOT the primary result — shown to quantify β residual absorption):

| Model | α₀ per-model | γ₁ per-model | TTFT MAPE | E2E MAPE |
|-------|-------------|-------------|----------|---------|
| codellama-34b | 15,389 µs | 7.15 µs/tok | 3.9% | 4.8% |
| llama-2-70b | 19,340 µs | 0 (γ₀=62ms) | 6.8% | 6.1% |
| llama-2-7b | 12,069 µs | 1,206 µs/tok | 13.0% | 8.7% |
| mixtral-8x7b | 8,895 µs | 1,480 µs/tok | 10.8% | 8.0% |

Per-model α/γ improves train TTFT by ~4pp average (from 12.8% to 8.6%) and E2E by ~2pp (from 9.4% to 6.9%). The improvement is NOT because α/γ capture model-specific CPU behavior — it's because they absorb model-specific β errors. Evidence: α₀ ranges 8.9-19.3ms (2.2× variation for identical tokenizer), γ₁ ranges 7-1480 µs/tok (211× variation for identical detokenizer).

### LOMO (Leave-One-Model-Out)

| Held-out | Train TTFT | Train E2E | Best test TTFT | Best test E2E |
|----------|-----------|-----------|---------------|--------------|
| codellama-34b | 3.8% | 8.7% | 1.7% (test) | 9.2% (test) |
| llama-2-70b | 7.1% | 18.0% | 1.4% (test) | 4.2% (test) |
| llama-2-7b | 17.3% | 1.9% | 1.5% (test) | 6.7% (test) |
| mixtral-8x7b | 4.1% | 2.1% | 1.1% (val) | 12.5% (val) |

### Feature Search Summary (12 designs tested)

| Feature Set | Avg Train MAPE | Best Model | Worst Model |
|------------|---------------|------------|-------------|
| H: handoff_full (6 feat) | **7.5%** | codellama 3.9% TTFT | 7b 13.0% TTFT |
| G: handoff (4 feat) | 10.2% | codellama 4.5% TTFT | 7b 20.4% TTFT |
| K: decoder-centric | 12.6% | codellama 2.8% TTFT | 7b 30.1% TTFT |
| A-F: compute-based | 16-19% | — | mixtral 34-42% TTFT |

### Key Insights

1. **KV bandwidth dominates cross-model decode cost** — β₁ (KV read) is the only consistently non-zero per-token feature across all 12 designs. Physical: decode is memory-bandwidth-bound on H100.
2. **Compute/FLOPs feature is redundant** — NNLS zeros it in every design. Prefill compute overlaps with memory access (GPU pipelining). With 4 models, L captures the depth-dependent cost that includes compute.
3. **TP synchronization is a discrete mode switch** — 9.4ms per-step overhead for TP>1. Not proportional to TP degree (TP=2 and TP=4 both pay ~9.4ms). Physical: NCCL barrier latency is dominated by round-trip, not data volume at these scales.
4. **4 environments limit cross-model β** — LOMO shows codellama/mixtral generalize (3.8-4.1% TTFT) but 7b degrades to 17-35% when held out. 7b is the outlier: TP=1 (no TP sync), MHA-32 (vs GQA-8 for all others), smallest model (unique weight/compute balance).
5. **γ₁=861 µs/tok is a β diagnostic** — real detokenization is ~5 µs/tok. The 170× inflation measures the average β decode error across all 4 models. Only with per-model γ does codellama's γ₁=7.15 approach the physical value, confirming the β is accurate for codellama but not for the other 3.
6. **α and γ MUST be global** — per-model α/γ disguises per-model fitting as "CPU overhead." The 2.2× α₀ variation and 211× γ₁ variation across models with identical tokenizers proves these are β residual absorbers, not physical measurements. Honest reporting requires global α/γ.

### Comparison with Iter 2 Per-Model

| Metric | Iter 2 (per-model β/α) | Iter 3 (global β/α/γ) | Delta |
|--------|---------------------|-------------------|-------|
| Train TTFT (worst) | 18.2% (7b) | 17.6% (7b) | -0.6pp ≈ |
| Train TTFT (best) | 3.8% (codellama) | 5.3% (codellama) | +1.5pp |
| Train E2E (worst) | 1.9% (7b) | 10.6% (7b) | +8.7pp ✗ |
| Train E2E (best) | 0.6% (codellama) | 8.0% (70b) | +7.4pp ✗ |
| Test TTFT range | 1.4-1.6% | 1.5-1.8% | ≈ |
| Test E2E range | 6.3-10.9% | 5.8-9.0% | ≈ |
| Parameters | 9/model (27 total) | 8 total (truly global) | -19 fewer |

**Trade-off:** Truly global model trades ~8pp E2E train accuracy for zero per-model calibration. 8 parameters vs 27. Test performance is comparable (T_queue dominates reasoning profiles).

### Path to Iter 4
- ~~**More environments** (>8 models)~~ — deprioritized. H30-H32 proved the dominant error is inter-step overhead, not β feature design. More models help β but won't close the -17% to -56% TTFT gap.
- **Inter-step scheduling overhead (δ)** — the #1 priority. See H30-H32 diagnostics below and Iter 4 strategy.
- **Non-linear batch-size interaction** — revisit after δ closes the TTFT gap. May become identifiable once the dominant error is removed.

---

## H30-H32 Diagnostics: BLIS Replay vs Real vLLM (#480)

_Date: 2026-03-03. Branch: training. 16 experiments (10 train, 3 validate, 3 test), three-way comparison (crossmodel, per-model blackbox, real vLLM)._

### Summary

| Finding | Evidence | Implication |
|---------|----------|-------------|
| **F1: T_queue_obs masking** | Iter 3 analytical TTFT uses `T_queue_obs` (real queue wait). At saturation, T_queue ≈ 120s → coefficient error < 0.02% of TTFT. The "1.5% test TTFT" was an illusion. | Analytical eval (11a) tests coefficients in isolation; BLIS replay (11b) tests system composition. Not substitutable. |
| **F2: Zero inter-step overhead is dominant** | Both crossmodel AND blackbox show identical -17% to -56% TTFT underprediction on all 10 training experiments. Difference between backends: 0-5pp (noise). | Error is in BLIS scheduling model (`SchedulingProcessingTime() → 0`), not coefficients. Real vLLM: 1-7ms/step (scheduler CPU + prepare_model_input + CUDA graph). |
| **F3: Regime transition at saturation** | codellama-34b reasoning: 3.93 rps predicted vs 3.22 rps real (+22%). ρ shifts from 1.24 → 1.02 — phase transition from unstable to stable queueing. TTFT: 120,171ms real → 64ms BLIS (2000×). | NOT proportional amplification. Capacity estimates near saturation are exponentially sensitive to µ. Confirms H7, H29. |
| **F4: γ₁ compensates accidentally** | CM α₂=860.6 µs/tok adds ~213ms/request to E2E. For 70b: CM E2E +0.6% vs BB -6%. For 7b: CM E2E -14% vs BB -22%. | γ₁ absorbs both β decode error AND missing inter-step overhead. Will decrease once δ is introduced. |
| **F5: enable_multi_turn_chat bug** | `ExpandInferencePerfSpec` interprets flag as context accumulation; real inference-perf uses it for chat template format. Real data: constant input tokens. | BLIS bug — set false for all inference-perf replay. |
| **F6: Sub-saturation throughput is tautological** | All 10 training experiments: throughput ±2.5%. But at sub-saturation, all requests complete → throughput ≈ arrival rate. | Meaningful throughput test only at saturation. ±2.5% validates completion, not step accuracy. |

### Quantitative Results (TTFT Mean, ms)

| Model | Profile | Blackbox | CrossModel | Real vLLM | BB RE | CM RE |
|-------|---------|----------|------------|-----------|-------|-------|
| llama-2-7b | general | 27.0 | 27.3 | 44.8 | -40% | -39% |
| llama-2-7b | codegen | 22.8 | 22.7 | 28.0 | -19% | -19% |
| llama-2-7b | roleplay | 22.5 | 22.0 | 27.1 | -17% | -19% |
| llama-2-70b | general | 49.3 | 45.4 | 103.0 | -52% | -56% |
| llama-2-70b | codegen | 46.7 | 44.1 | 55.6 | -16% | -21% |
| llama-2-70b | roleplay | 46.8 | 43.9 | 54.8 | -15% | -20% |
| mixtral-8x7b | general | 47.4 | 51.1 | 68.9 | -31% | -26% |
| mixtral-8x7b | codegen | 46.2 | 47.5 | 58.9 | -22% | -19% |
| mixtral-8x7b | roleplay | 46.7 | 50.0 | 60.5 | -23% | -17% |
| codellama-34b | general | 39.5 | 39.7 | 51.6 | -23% | -23% |

Mean TTFT |RE|: BB=25.8%, CM=25.9%. Difference between backends: 0.1pp. The error is in the scheduling model.

### Inter-Step Overhead Components (from H30 vLLM analysis)

| Component | Estimated Range | Per-Request Impact (150 steps) |
|-----------|----------------|-------------------------------|
| scheduler.schedule() CPU | 0.5-2ms/step | 75-300ms |
| prepare_model_input() | 1-5ms/step | 150-750ms |
| CUDA graph capture/replay asymmetry | 0.01-50ms/step (bimodal) | variable |
| Python GIL + update_from_output() | 0.1-0.5ms/step | 15-75ms |
| HTTP/ASGI overhead (TTFT only) | 1-5ms/request (once) | 1-5ms |

Total per-request: ~240-1130ms. Observed TTFT gap: ~7-58ms (at sub-saturation, 10-16K requests, moderate load). The gap grows with load as higher queue depth increases step count per request.

---

## Iter 4 Strategy: Two-Phase Simulator-Faithful Learning

### Motivation

Iter 3 proved that 8 global physics coefficients (β, α, γ) produce excellent analytical accuracy (TTFT 5-18%, E2E 8-11%). H30-H32 proved that analytical accuracy ≠ simulator accuracy — the dominant error is missing inter-step overhead (δ), not coefficient inaccuracy. Iter 4 adds δ and uses simulator-in-the-loop refinement to ensure coefficients are faithful when composed with BLIS's scheduler.

### Parameter Space (8 parameters, all global)

| Param | Role | Warm-Start Source | Physical Meaning |
|-------|------|-------------------|-----------------|
| β₀ | Per-layer overhead | Iter 3 NNLS (116.1 µs/layer) | CUDA kernel dispatch per transformer layer |
| β₁ | KV bandwidth | Iter 3 NNLS (1226.9 µs/kv_unit) | Decode KV cache HBM read |
| β₂ | MoE dispatch | Iter 3 NNLS (19.9 µs/MoE_tok) | Expert routing + scatter/gather |
| β₃ | TP synchronization | Iter 3 NNLS (9445.2 µs/step) | NCCL all-reduce barrier |
| α₀ | Pre-scheduling | Iter 3 min-wait (13,732 µs) | HTTP parsing, tokenization |
| α₂ | Output processing | Iter 3 E2E residual (860.6 µs/tok) | β decode error absorber (diagnostic) |
| **δ₀** | **Fixed scheduling overhead** | **NEW: measured 8-18ms/step (model-dependent)** | **scheduler.schedule() + prepare_model_input() + Python/GIL** |

### Two-Phase Strategy

**Phase 1 (analytical warm-start):** NNLS/convex optimization on training data. Unchanged for β, α. New for δ: fit from BATCH_SUMMARY inter-step timing gaps.

**Phase 2 (simulator-in-the-loop refinement):** Coordinate-wise blackbox optimization using BLIS replay with multi-signal loss (step accuracy 15%, queue dynamics 25%, TTFT 30%, E2E 20%, throughput 10%). Four stages in order of consequence:
1. **δ₀ sweep** (grid search [5000, 25000], 4 fast experiments) — highest leverage
2. **α₀, α₂ refinement** (Bayesian optimization, 30 evals) — next highest
3. **β₀-β₃ fine-tuning** (CMA-ES, σ₀=5%, 50 generations) — already close
4. **Global polish** (optional, all 7, CMA-ES σ₀=3%) — only if gates still fail

Physics guard-rails: all params bounded ±30% of warm-start, β ≥ 0, δ ≥ 0.

### Predicted Impact

| Metric | Pre-δ (H30 actual) | Post-δ Target | Rationale |
|--------|---------------------|---------------|-----------|
| TTFT mean RE | -17% to -56% | < ±15% | δ closes the inter-step gap (~240-1130ms/req) |
| E2E mean RE | ±6-15% | < ±10% | γ₁ decreases as δ absorbs part of its load |
| Throughput RE (saturation) | +22% | < ±10% | δ lowers µ → ρ increases → closer to real saturation |
| γ₁ value | 860.6 µs/tok | ~200-600 µs/tok | Predicted decrease as δ takes over inter-step compensation |

### Hypothesis Bundle

See `training/iter4-bundle.md` for the full bundle (H-main + 6 arms).

---

## Iteration 2 Full Results

### Training Procedure (fully reproducible)

**Step-time model:** `Δt_k = β₀ + β₁·T_k^pf + β₂·T_k^dec` where T^pf = cache-miss prefill tokens, T^dc = decode tokens per step.

**Data sources:**
- `iter2_journeys.csv`: 150,858 journeys with `scheduler.step` indices (extracted by `iter2_extract.py` from OTEL `vllm.scheduler` scope spans)
- `iter0_steps.csv`: 122,752 sampled steps (10% BATCH_SUMMARY)

**Batch reconstruction:** For each experiment, reconstruct (T_k^pf, T_k^dc) for every step from journey step indices. Per-request: prefill steps = [step_scheduled, step_first_token], contributes chunk_size tokens per full step + remainder on last. Decode steps = (step_first_token, step_finished], contributes 1 token each. Chunk sizes from max(prefill_tokens): 7b=956, 70b=2046, mixtral=989, codellama=577.

**NNLS fitting (per environment):**
- Block A (direct steps, ~88K train): row = [1, pf, dc], target = duration_us
- Block B (journey intervals, ~231K train): row = [N_steps, Σpf, Σdc], target = interval_us
- Weight: w_journey = sqrt(N_direct / N_journey) applied to Block B
- Solve: `β = argmin_{β≥0} ||X·β - y||²` via scipy.optimize.nnls

**Separate alpha fitting (critical for TTFT):**
- α_ttft fit from TTFT residuals: `R = T_prefill_obs - T_prefill_pred`, features [1, ν_in, 1]
- α_e2e fit from E2E residuals: `R = E2E_obs - T_queue - T_prefill_pred - T_decode_pred`, features [1, ν_in, ν_out+1]
- Both via NNLS (α ≥ 0)

**Prediction formulas:**
```
TTFT_pred(r) = (α_ttft₀ + α_ttft₁·ν_in + α_ttft₂) + T_queue_obs(r) + T_prefill_pred(r)
E2E_pred(r)  = (α_e2e₀ + α_e2e₁·ν_in + α_e2e₂·(ν_out+1)) + T_queue_obs(r) + T_prefill_pred(r) + T_decode_pred(r)

where T_prefill_pred = Σ_{k∈prefill_steps} (β₀ + β₁·T_k^pf + β₂·T_k^dc)
      T_decode_pred  = Σ_{k∈decode_steps}  (β₀ + β₁·T_k^pf + β₂·T_k^dc)
      T_queue_obs    = (scheduled_ns - queued_ns) / 1000  [observed, not predicted]
```

### Fitted Coefficients (per environment)

| Model | β₀ (µs) | β₁ (µs/tok) | β₂ (µs/tok) | α_ttft₀ (µs) | α_e2e₀ (µs) | α_e2e₁ (µs/tok) | α_e2e₂ (µs/tok) |
|-------|---------|-------------|-------------|-------------|------------|----------------|----------------|
| llama-2-7b | 6,179 | 2.065 | 121.342 | 9,680 | 0 | 22.755 | 0 |
| llama-2-70b | 16,536 | 3.622 | 42.500 | 17,660 | 17,731 | 0 | 0 |
| mixtral-8x7b | 17,560 | 5.028 | 18.132 | 16,081 | 16,741 | 0 | 0 |
| codellama-34b | 13,549 | 2.020 | 35.910 | 14,986 | 0 | 4.170 | 50.8 |

**β₀ interpretation:** Absorbs ~6-18ms inter-step scheduler overhead from journey constraints. Direct-step-only physics model confirms GPU kernels have ~0 fixed overhead. For BLIS StepTime integration, β₀ includes this overhead which BLIS doesn't model separately.

### Results Across All Splits

| Split | Model | TTFT MAPE | E2E MAPE | n requests |
|-------|-------|----------|---------|------------|
| **TRAIN** | llama-2-7b | **18.2%** | **1.9%** | 33,000 |
| **TRAIN** | llama-2-70b | **5.4%** | **1.1%** | 33,000 |
| **TRAIN** | mixtral-8x7b | **4.6%** | **1.4%** | 32,658 |
| **TRAIN** | codellama-34b | **3.8%** | **0.6%** | 16,800 |
| **VALIDATE** | codellama-34b | **6.3%** | **0.7%** | 16,200 |
| **VALIDATE** | mixtral-8x7b | **1.0%** | 9.1% | 4,800 |
| **TEST** | llama-2-7b | **1.5%** | 8.6% | 2,353 |
| **TEST** | llama-2-70b | **1.4%** | 6.3% | 4,800 |
| **TEST** | codellama-34b | **1.6%** | 10.9% | 4,800 |

**Train:** TTFT 3.8-18.2%, E2E 0.6-1.9% — all 4 environments pass 25%/20% gates.
**Test:** TTFT 1.4-1.6% (better than train — reasoning profiles have long T_queue that dominates). E2E 6.3-10.9% (higher than train — reasoning has different load/preemption dynamics).

### Ablation Comparison (TRAIN, shared α for ablations)
| Config | llama-2-7b | llama-2-70b | mixtral-8x7b | codellama-34b |
|--------|-----------|------------|-------------|--------------|
| **Journey-NNLS + sep α** TTFT | **18.2%** | **5.4%** | **4.6%** | **3.8%** |
| Journey-NNLS + shared α TTFT | 35.1% | 5.5% | 4.6% | 37.8% |
| Step-NNLS TTFT | 11,227% | 5,511% | 11,807% | 36.4% |
| Step-OLS TTFT | 11,229% | 10,326% | 11,807% | 13.9% |
| **Journey-NNLS E2E** | **1.9%** | **1.1%** | **1.4%** | **0.6%** |
| Step-NNLS E2E | 19.6% | 1.5% | 3.6% | 1.5% |

### Hypothesis Assessment
| Hypothesis | Status | Evidence |
|------------|--------|----------|
| H-main (TTFT<25%, E2E<20%) | **CONFIRMED** | Train: TTFT 3.8-18.2%, E2E 0.6-1.9%. Test: TTFT 1.4-1.6%, E2E 6.3-10.9%. All within targets. |
| H-NNLS (NNLS > OLS) | **CONFIRMED** | Journey-NNLS vs Step-OLS: massive TTFT improvement (1K-11Kpp). |
| H-journey (journey > step-only) | **CONFIRMED** | Journey constraints reduce TTFT by 5,500-11,800pp. Essential for 3/4 models. |
| H-robustness (cross-profile) | **CONFIRMED** | codellama val: TTFT 6.3% vs train 3.8% (+2.5pp). E2E 0.7% vs 0.6% (+0.1pp). |
| Separate α vs shared α | **CONFIRMED** | Separate α fixes 16-34pp TTFT regression with 0pp E2E sacrifice. |

### Key Insights
1. **Journey step indices enable exact batch reconstruction** for all ~1.2M steps, collapsing Continuous EM to single NNLS
2. **NNLS (β≥0) eliminates negative intercepts** that caused Iter 0's 100-120% step MAPE
3. **Journey constraints are transformative** — without them, TTFT MAPE is 5K-12K%
4. **Separate α_ttft/α_e2e is critical** — shared α leaks decode prediction errors into TTFT (16-34pp worse)
5. **β₀ absorbs inter-step scheduler overhead** (~6-18ms) — real GPU kernels have ~0 fixed overhead
6. **Test TTFT (1.4-1.6%) is better than train (3.8-18.2%)** — reasoning profiles have long T_queue that dominates, making model contribution a small fraction
7. **Physics-informed features** (compute, KV bandwidth, TP comm) are structurally correct but need >4 environments for cross-model accuracy

### Physics-Informed Global β (single vector, all environments, direct steps only)
| Feature | β value | Physical meaning |
|---------|---------|-----------------|
| (pf+dc)×L (shared compute) | 0.010 µs/(tok·layer) | Forward-pass compute per token per layer |
| dc×L×kv_heads/1e3 (KV bandwidth) | 61.0 µs/unit | Decode KV cache read cost (dominant) |
| (pf+dc)×H×(TP-1)/1e3 (TP comm) | 0.004 µs/unit | TP all-reduce overhead |

### Path to Iter 3
- More training environments for global physics β convergence
- Extend features: TP all-reduce with hidden_dim scaling, MoE dispatch/gather
- Consider `Δt = β · f(batch_shape, model_features)` with physics-informed f for cross-model single β

---

## Iteration 0 Full Results

### Fitted Coefficients
| Model | β₀ | β₁ (prefill) | β₂ (decode) | Train MAPE | Weighted MAPE | R² | Pearson r | Null MAPE |
|-------|-----|---------------|-------------|------------|---------------|-----|-----------|-----------|
| llama-2-7b | -815.5 | 1.075 | 77.104 | 120.7% | 37.6% | 0.911 | 0.956 | 407.1% |
| llama-2-70b | -1376.8 | 2.245 | 66.573 | 102.6% | 42.0% | 0.753 | 0.868 | 416.0% |
| mixtral-8x7b | 67.5 | 1.392 | 7.162 | 24.1% | 20.0% | 0.390 | 0.624 | 60.1% |
| codellama-34b | -2206.3 | 4.121 | 83.297 | 122.3% | 53.8% | 0.536 | 0.733 | 497.2% |

### Error by Duration Band (llama-2-7b, representative)
| Band | n | MAPE | MAE | Bias | R² |
|------|---|------|-----|------|-----|
| 0-200us | 28,553 | 654% | 775us | +628% | -404 |
| 200-1000us | 8,867 | 407% | 1108us | +348% | -48 |
| 1000-5000us | 1,958 | 37% | 818us | +24% | 0.55 |
| 5000-50000us | 2,690 | 11% | 691us | +5% | -0.02 |

### Diagnostic Results
| Diag | Gate | Result |
|------|------|--------|
| D-integrity | ±5% step count | PASS (all 1.000) |
| D0 | Trivial MAPE <5% | FAIL (18-33%) → irreducible GPU noise floor |
| D2 | Monotonicity <5% | FAIL (42-79%) → GPU timing noise at fine granularity |
| D3 | Alpha extraction | Computed (α₀=121-240, α₁=0-0.15) |
| D4 | Batch-size r<0.3 | PASS (r=0.004-0.100) → batch-size NOT a confound |
| D5 | Heteroscedasticity | FAIL for 3/4 (r=0.34-0.60) → need weighted/log |
| D6 | Autocorrelation | FAIL for 3/4 (lag-1=0.43-0.83) → Mixtral only exception |
| D7 | Signed bias <±5% | PASS (3/4), FAIL llama-2-7b (+10.4%) |
| D8 | Prefill semantics | PASS — prefill+decode==scheduled confirms cache-miss tokens |

### Validation (codellama-34b cross-profile)
| Experiment | MAPE | Bias |
|------------|------|------|
| codegen | 162.2% | +98.6% |
| roleplay | 93.6% | -84.4% |
| combined | 127.4% | +3.3% |
| H-robustness: +5.2pp degradation → INCONCLUSIVE |

### Root Cause Analysis
The negative intercept (β₀ = -815 to -2206 for dense models) reveals the fundamental problem: **the linear form assumes per-token cost is constant across batch sizes, but GPU step execution has a strongly nonlinear batch-utilization curve**. Small decode-only batches (10-20 tokens) cost ~10us/token; large batches (200+ tokens) cost ~30-80us/token. OLS finds the best line through both regimes, resulting in:
- Slope (β₂) tuned to large-batch steps → overpredicts small batches
- Negative intercept (β₀) compensating → but clamp to 1us still overpredicts massively

Mixtral is the exception because its MoE sparse activation changes the compute profile: each expert processes fewer effective FLOPs per token, producing a more uniform per-token cost across batch sizes. β₀=67 (positive) confirms the form works when per-token cost is approximately constant.

### Implications for Iteration 1
1. **D4 says batch-size is not a simple additive confound** (r<0.1). Adding β₃·batchSize won't fix this.
2. **The nonlinearity is multiplicative**: per-token cost DEPENDS on batch utilization. Need interaction terms or separate models.
3. **Candidate approaches for Iter 1**:
   - (a) Log-transform: `log(duration) = β₀ + β₁·log(prefill+1) + β₂·log(decode+1)` — captures sub-linear scaling
   - (b) Separate models: different coefficients for small-batch (<50 tokens) vs large-batch (≥50 tokens) regimes
   - (c) Interaction term: `β₀ + β₁·prefill + β₂·decode + β₃·decode·log(batch_size+1)` — captures utilization-dependent per-token cost
4. **Duration-weighted MAPE** (37-54%) is the relevant metric for DES, not raw MAPE (100-120%)
5. **Mixtral coefficients are usable as-is** for capacity planning (24% MAPE, 0% total duration error)
