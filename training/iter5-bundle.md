# Iter 5 Hypothesis Bundle — Replay-Calibrated Joint Coefficient Optimization

_Date: 2026-03-04_
_Branch: `iter5-decomposed`_
_Strategy Evolution Phase 2 — Hypothesis Bundle Design_
_Status: **APPROVED** (human gate passed)_

---

## Strategy Summary

Improve BLIS replay TTFT accuracy from Iter 3's 21.1% MAE to <15% by jointly optimizing latency coefficients using BLIS replay error as the objective function. Two competing arms test whether refitting β (correcting the Block B entanglement) outperforms simply adding δ to the existing Iter 3 β.

## Prior Knowledge (from Iter 5 Prototype)

1. Iter 3 β₁(KV) and β₃(TP) are inflated ~2.5-3.5× by Block B journey constraints absorbing inter-step overhead
2. Wall-clock-fitted β alone fails cross-model (MoE β₂ 6× too high → queueing explosion)
3. Universal 2.0× pipeline factor: `first_token_ns` always falls AFTER `ts_start[S+1]` (one-cycle delay)
4. `step.duration_us` = scheduler CPU time (~200-2000µs), not GPU time (~5-20ms)
5. Iter 3 BLIS replay: 21.1% TTFT MAE, 17.9% E2E MAE (12 non-reasoning experiments)
6. Per-model TTFT bias: mixtral +2.4%, codellama -17 to -22%, llama-2-7b -20 to -35%, llama-2-70b -20 to -48%

## Hypothesis Arms

### H-main: Corrected β + Replay-Optimized (α₀, δ₀)

**Claim:** Refitting β from overhead-corrected step-level wall clock (Phase A), then optimizing (α₀, δ₀) against BLIS replay TTFT error (Phase B), will achieve **<15% TTFT MAE** across 10 training experiments (non-reasoning).

**Phase A — Corrected β fitting:**
- Target: `T_gpu_est = T_wall - T_overhead_median[model]`
- `T_wall` from consecutive-pair step timestamps (8,692 train pairs)
- `T_overhead_median` from per-model consecutive-pair residuals after subtracting Iter 3 β prediction (measured in prototype)
- Fit via NNLS (β ≥ 0) against architecture features [L, dc·kv_dim, (pf+dc)·MoE, I(TP>1)]

**Phase B — Replay optimization:**
- Fix β from Phase A
- Optimize (α₀, δ₀) via Nelder-Mead
- Loss: mean TTFT MAPE across 4 representative experiments (one per model, roleplay profile)
- Bounds: α₀ ∈ [0, 50000], δ₀ ∈ [0, 30000]

**Predicted outcome:**
- β₁(KV) ~400-600 (between wall-clock 355 and Iter 3 1227)
- β₃(TP) ~2000-4000 (between wall-clock 3770 and Iter 3 9445)
- δ₀ ≈ 5000-10000µs (one step's worth of pipeline overhead)
- α₀ ≈ 5000-15000µs (request preprocessing)
- TTFT MAE <15% train, improved TTFT bias toward zero

**Diagnostic clause:** If TTFT MAE increases despite δ > 0, the corrected β features don't separate GPU compute from overhead — the correction overshot.

### H-ablation-fixedβ: Iter 3 β + Replay-Optimized (α₀, δ₀, δ₁)

**Claim:** Keeping Iter 3 β and searching (α₀, δ₀, δ₁·batch_size) via replay will achieve **<18% TTFT MAE** — better than baseline but worse than H-main.

**Mechanism:** The batch-size-dependent δ term compensates for the fact that Iter 3 β absorbed different amounts of overhead at different batch compositions.

**Predicted outcome:**
- δ₀ ≈ 2000-5000µs
- δ₁ < 0 (overhead decreases with batch size, because Iter 3 β already absorbed more at high batch)
- α₀ ≈ 5000-15000µs (similar to H-main)
- TTFT MAE 16-18% (3-5pp improvement over baseline)

**Diagnostic clause:** If this arm MATCHES or BEATS H-main, correcting β doesn't matter — the entanglement is benign and pragmatic compensation is sufficient.

### H-control: Iter 3 Baseline (δ = 0)

**Claim:** Iter 3 coefficients with no optimization produce 21 ± 2% TTFT MAE, consistent with H30-H32 diagnostic.

**Purpose:** Anchors the comparison. Confirms the replay harness produces consistent results.

### H-robustness: Test-Set Evaluation

**Claim:** The winning configuration maintains <25% TTFT MAE on 3 test experiments (reasoning profiles). These are saturated workloads — predictions are inherently harder but should still improve over Iter 3.

## Experiment Design

**Optimizer setup:**
- Training loss: mean of per-experiment TTFT MAPE (4 roleplay experiments, one per model)
- Full evaluation: all 12 non-reasoning experiments + 3 reasoning test experiments
- Nelder-Mead with initial simplex scale = 20% of warm-start values
- Max evaluations: 200 (budget)
- Seed: 42 (deterministic)

**Evaluation metrics per experiment:**
- TTFT: mean_ms, p50, p99
- E2E: mean_ms, p50, p99
- Throughput: requests/sec
- Compared against per-experiment ground truth

## Success Criteria

| Arm | TTFT MAE (train) | E2E MAE (train) | Pass condition |
|-----|-----------------|----------------|----------------|
| H-main | <15% | <20% | Both must pass |
| H-ablation | <18% | <20% | Both must pass |
| H-control | ~21% | ~18% | Anchor |

**Cross-arm:** H-main beats H-ablation by >3pp TTFT → correcting β matters.
