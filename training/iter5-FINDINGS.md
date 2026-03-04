# Iter 5 FINDINGS — Replay-Calibrated Joint Coefficient Optimization

_Date: 2026-03-04_
_Branch: `iter5-decomposed`_
_Strategy Evolution Phase 3: Results_

---

## Summary

H-ablation (Iter 3 β + replay-optimized α₀/δ₀/δ₁) achieves **13.3% train TTFT MAE** — a 40% relative improvement over the 22.1% Iter 3 baseline, with 8 truly cross-model parameters (4β + α₀ + δ₀ + δ₁ = 7 optimized, γ dropped). H-main (corrected β) fails for llama-2-70b at high load. Both arms reach near-zero TTFT error on validation.

## Fitted Coefficients (H-ablation — winner)

**β (step time, Iter 3 — unchanged):**

| Feature | Value | Physical mechanism |
|---------|-------|--------------------|
| β₀ (per-layer) | 116.1 µs/layer | CUDA kernel dispatch per transformer layer |
| β₁ (KV bandwidth) | 1,226.9 µs/kv_unit | HBM reads for decode attention |
| β₂ (MoE routing) | 19.9 µs/MoE_token | Expert selection + dispatch overhead |
| β₃ (TP sync) | 9,445.2 µs/step | NCCL all-reduce barrier |

**Replay-optimized overhead:**

| Parameter | Value | Physical interpretation |
|-----------|-------|----------------------|
| α₀ | **24,171 µs** | Per-request pipeline overhead (scheduling + input prep + one-cycle GPU pipeline delay) |
| δ₀ | **-3,542 µs** | Per-step correction (negative: removes β's absorbed overhead) |
| δ₁ | **-252.1 µs/req** | Per-request-in-batch correction (negative: larger batches had more absorbed overhead) |

**Effective per-step time:**
```
T_step = β·features(pf, dc, model) + δ₀ + δ₁ · batch_size
```
At batch_size=50: T_step = β·features - 3,542 - 12,605 = β·features - 16,147µs

**TTFT prediction (for a single-step prefill):**
```
TTFT_pred = α₀ + T_step_prefill + queue_wait
         = 24,171 + β·features - 3,542 - 252·batch_size + queue_wait
```

## Prediction vs Outcome

### H-main (corrected β + α₀=19041, δ₀=492)

| Metric | Predicted | Observed | Status |
|--------|-----------|----------|--------|
| Train TTFT MAE | <15% | **25.1%** | **REFUTED** |
| Train E2E MAE | <20% | 26.2% | REFUTED |
| Val TTFT MAE | — | **0.5%** | Excellent |
| β₁(KV) change | ~400-600 | 563.3 | Confirmed |
| β₃(TP) change | ~2000-4000 | 9,705.3 | **REFUTED** (stayed same) |
| δ₀ | ~5000-10000 | 492 | **REFUTED** (near zero) |

**Root cause of failure:** The corrected β removed too much overhead for codellama/70b (negative overheads of -8840 and -7020µs in Phase A), making β₁(KV) drop from 1227 to 563. This caused llama-2-70b general profile to cross the saturation boundary (103ms real → 244ms simulated, +137%), a regime transition error.

### H-ablation (Iter 3 β + α₀=24171, δ₀=-3542, δ₁=-252) — **WINNER**

| Metric | Predicted | Observed | Status |
|--------|-----------|----------|--------|
| Train TTFT MAE | <18% | **13.3%** | **CONFIRMED** (beat target) |
| Train E2E MAE | <20% | 26.6% | **REFUTED** |
| Val TTFT MAE | — | **0.6%** | Excellent |
| δ₁ sign | Negative | -252.1 | Confirmed |
| δ₀ sign | Positive | **-3,542** | **REFUTED** (negative, not positive) |
| Improvement over H-main | H-ablation worse | H-ablation **better by 12pp** | REFUTED (ablation wins) |

**Key finding:** The optimizer found that SUBTRACTING time per step (via negative δ) works better than adding it. This means Iter 3 β already overpredicts due to Block B entanglement, and the optimal strategy is to correct the overprediction rather than add more overhead.

### H-control (Iter 3 baseline)

| Metric | Predicted | Observed | Status |
|--------|-----------|----------|--------|
| Train TTFT MAE | ~21% | **22.1%** | Confirmed (within ±2%) |
| Train E2E MAE | ~18% | 18.4% | Confirmed |

### H-robustness (reasoning test set)

All arms fail catastrophically on reasoning experiments (>100s real TTFT → 30-50ms simulated). This is expected: reasoning profiles are at saturation (>33% failure rate) where BLIS's simulator dynamics diverge from vLLM's. None of the coefficient sets can predict the saturation regime.

## Results Table

| Arm | Train TTFT MAE | Train E2E MAE | Val TTFT MAE | Val E2E MAE |
|-----|---------------|---------------|-------------|-------------|
| **H-ablation** | **13.3%** | 26.6% | **0.6%** | 8.1% |
| H-main | 25.1% | 26.2% | 0.5% | N/A |
| H-control | 22.1% | 18.4% | 15.9% | 14.6% |

### Per-Model Train TTFT MAE

| Model | H-ablation | H-main | H-control |
|-------|-----------|--------|-----------|
| mixtral-8x7b | **7.8%** | 21.0% | 11.9% |
| codellama-34b | **8.1%** | 9.1% | 21.7% |
| llama-2-7b | **13.2%** | 11.9% | 25.3% |
| llama-2-70b | 20.5% | **47.7%** | 29.4% |

H-ablation wins 3/4 models on TTFT. llama-2-70b general profile remains the hardest case (-48.4% TTFT bias at high load).

## E2E Degradation Analysis

H-ablation's E2E MAE (26.6%) is worse than Iter 3's (18.4%). The negative δ removes ~16ms per step at batch_size=50, which accumulates over ~250 decode steps to -4000ms. This over-corrects E2E even though TTFT improves. A future iteration should consider separate δ for prefill-dominated vs decode-dominated steps, or a γ term for E2E calibration.

## Cross-Arm Comparison

**Prediction:** H-main beats H-ablation by >3pp TTFT.
**Outcome:** H-ablation beats H-main by **12pp TTFT** (13.3% vs 25.1%).

**Conclusion:** Correcting β (H-main) is NOT the right approach. The Iter 3 β, despite being "wrong" at the step level, produces better BLIS replay results because its entangled overhead partially models the pipeline delay. The optimal strategy is to keep the entangled β and fine-tune with (α, δ) that correct the residual.

## Principles Extracted

1. **P31: Entanglement can be feature, not bug.** Iter 3 β absorbed inter-step overhead via Block B journey constraints. This "incorrect" step-level prediction paradoxically produces better journey-level predictions because the absorbed overhead approximates the pipeline delay.

2. **P32: Negative δ is physically meaningful.** When β already contains overhead, the optimal δ is negative — it removes the excess. The optimizer discovered this without being told about the entanglement.

3. **P33: α₀ absorbs the pipeline factor.** The 2.0× pipeline delay (from Investigation 2) manifests as α₀=24ms — roughly one extra step cycle added per request as a fixed overhead.

4. **P34: TTFT and E2E optimization are partially adversarial.** Negative δ that improves TTFT worsens E2E because the correction accumulates over many decode steps. A future iteration needs separate correction terms for the prefill-dominated TTFT path vs the decode-dominated E2E path.

## Iter 5c: Joint 7-Param Non-Negative Optimization

After the E2E degradation finding, a follow-up run jointly optimized all 7 parameters (β₀-β₃, α₀, γ₁, δ₀) with non-negativity constraints. γ₁ (OutputTokenProcessingTime per output token) serves as the "phase-aware" mechanism: it adds to E2E without affecting TTFT (at TTFT measurement, output_tokens=0).

**Method:** Nelder-Mead on 4 roleplay experiments (multi-objective: 0.5×TTFT_MAPE + 0.5×E2E_MAPE).

**Best result (eval 156, 9.88% combined loss):**

| Parameter | Value | Iter 3 | Change | Physical meaning |
|-----------|-------|--------|--------|------------------|
| β₀ (per-layer) | **137** µs | 116 | +18% | CUDA kernel dispatch per layer |
| β₁ (KV bandwidth) | **2,643** µs | 1,227 | +115% | KV cache HBM read — DOUBLED (key finding) |
| β₂ (MoE routing) | **29** µs | 20 | +45% | Expert routing per MoE token |
| β₃ (TP sync) | **5,119** µs | 9,445 | **-46%** | NCCL barrier — HALVED (was most inflated) |
| α₀ (request overhead) | **21,128** µs | 13,732 | +54% | Pipeline delay + preprocessing |
| γ₁ (output processing) | **1,431** µs/tok | 861 | +66% | Per-output-token E2E correction |
| δ₀ (step overhead) | **178** µs | 0 | new | Small positive per-step overhead |

**All coefficients non-negative.** The optimizer found that:
- β₁(KV) should be LARGER than Iter 3 (not smaller as the wall-clock prototype suggested)
- β₃(TP) should be ~half of Iter 3 (the TP sync was the most inflated by Block B)
- δ₀ collapsed to ~178µs (nearly zero — most overhead captured by α₀ and γ₁)
- γ₁=1,431µs/tok independently tunes E2E without touching TTFT

**Not yet fully evaluated** on all experiments (optimizer loss on 4 roleplay only). Full evaluation on all 16 experiments pending in next iteration.

## Ledger Entry

| Iter | Strategy | Form | Params | Train TTFT MAE | Train E2E MAE | Val TTFT MAE | Key Insight | Status |
|------|----------|------|--------|---------------|---------------|-------------|-------------|--------|
| 5 | Replay-calibrated (α₀, δ₀, δ₁) | Iter 3 β + NM-optimized overhead | 7 global | **13.3%** | 26.6% | **0.6%** | Negative δ corrects β entanglement; α₀=24ms absorbs pipeline delay; correcting β (H-main) fails at high load | H-ablation CONFIRMED, H-main REFUTED |
| 5c | Joint 7-param non-neg | NM on (β,α₀,γ₁,δ₀), all ≥ 0 | 7 global | ~10%* | ~10%* | TBD | β₁(KV) doubles, β₃(TP) halves; γ₁ independently tunes E2E; δ₀→0 | PROMISING (optimizer loss 9.88%, full eval pending) |

*Optimizer subset only (4 roleplay experiments). Full evaluation needed.
