# Iteration 28: Findings — TPE Cross-Check (No Improvement)

## Summary

TPE cross-check of iter27's CMA-ES result over 162 trials. Best loss: **34.6564%** vs
iter27 baseline of 34.61% (+0.05 points — no meaningful change). Patience=150 fired at
trial 150; iter27 warm-start (trial 0) remained best throughout.

## What Happened

162 trials ran over 27 minutes (10.1s/trial avg, 15 parallel jobs). TPE's random burn-in
(trials 1–25) and subsequent directed search (trials 26–162) found nothing better than the
warm-start. The closest competitor was trial 155 at 34.719% — 0.06 points worse.

4 trials failed (penalty loss applied, excluded from best).

## Cross-Check Conclusion

This is a **meaningful negative result**. Unlike the prior diagnostic run (1 trial), 162
trials with a full TPE directed-search phase robustly confirm that iter27's coefficient
values are a local optimum within these bounds. CMA-ES and TPE agree — the point is stable
across two qualitatively different optimization methods.

Progress beyond 34.61% likely requires either:
1. **Wider bounds** — the optimum may sit outside the current ±30% window
2. **Structural model change** — the Llama-4-Scout TTFT issue (59% APE) appears fundamental
   and is not addressable by tuning existing basis functions

## Dominant Hard Case (Unchanged)

Llama-4-Scout reasoning-lite: 67.6% combined loss (59.4% TTFT APE, 8.2% E2E APE).
This single experiment accounts for ~45% of total loss and was unchanged across all 162
trials. Structural modeling work is needed here.

## Full Training Journey

| Iter | Loss | Key change |
|---|---|---|
| 16 | 60.19% | Trained-roofline baseline |
| 20 | 40.58% | β₈·nMoELayers breakthrough |
| 21 | 39.86% | Prefill compute-only split |
| 22 | 39.42% | β₂ decode correction |
| 23 | 39.24% | 3D joint (β₁ₐ,β₂,β₈) |
| 24 | 39.18% | Decode memory-only split |
| 25 | 39.18% | β₈ moeScaling (arch-aware) |
| 26 | 37.42% | T_tp activated, β₄=0.41, β₅=49.6 |
| 27 | 34.61% | CMA-ES joint 6-param |
| **28** | **34.66%** | **TPE cross-check — no improvement (162 trials, patience=150)** |
