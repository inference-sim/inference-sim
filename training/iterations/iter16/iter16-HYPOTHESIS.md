# Iteration 16: Adopt Trained-Roofline Architecture + MoE/Quantization Fixes

## Context

Iterations 0-14 of the evolved model made two structural errors that no amount of coefficient tuning could fix:

1. **Missing memory-bandwidth floor**: Decode basis function used `FLOPs / (peakFlops × MfuDecode)` (compute-only), dropping the `max(compute, memory)` roofline crossover. At batch_size=1, this underestimates decode step time by ~13× (0.23ms predicted vs 3.08ms roofline).

2. **Per-request overhead accumulation**: β₃ (KV mgmt) and β₇ (decode overhead) were applied per-request per-step in StepTime, accumulating O(N×B) over N decode steps × B batch size. These terms inflated to compensate for the missing memory floor, causing 97-99% of predicted step time to be phantom overhead.

The `trained-roofline` backend (already in the codebase) avoids both errors and achieves 7% MAPE on 13 experiments. Iter15 adopts its architecture with three enhancements for the current 15-experiment dataset.

## H-main: Architecture Adoption Reduces Loss Below 100%

**Prediction**: Overall loss will decrease from iter8's 155% to **<100%**, with:
- E2E RMSE dropping from 91% to **<60%** (structural fix eliminates O(N×B) accumulation)
- TTFT RMSE remaining below **<70%** (same roofline basis, no regression)

**Causal Mechanism**: The trained-roofline formula `β₁·max(T_pf_compute, T_pf_kv) + β₂·max(T_dc_compute, T_dc_kv) + β₃·T_weight + β₅·L + β₆·batchSize + β₇`:
- Uses `max(compute, memory)` for decode, providing the correct memory-bandwidth floor (~3-7ms per step instead of ~0.2ms)
- Places per-request overhead (β₆) in µs scale (48.8µs/req), not ms scale (26.3ms/req)
- Uses PostDecodeFixedOverhead (α₁) for per-request completion overhead, applied once per request (not N×B times)
- The optimizer has physically correct basis functions to work with, so coefficients converge to plausible values

**Code Citations**:
- Roofline crossover: `trained_roofline.go:161-162` (`math.Max(tPfCompute, tPfKv)`)
- PostDecodeFixedOverhead: `sim/simulator.go:486` (applied once in `recordRequestCompletion`)
- Memory bandwidth: `roofline.go:334` (`totalMemoryS = (weightBytes + totalDynamicBytes) / peakBW`)

**Diagnostic Clause**: If overall loss remains >100%, investigate:
- Whether the optimizer found coefficients near the trained-roofline priors (β₁≈0.77, β₂≈1.13)
- Whether Scout MoE experiments dominate error (interleaved MoE fix may need validation)
- Whether reasoning-lite experiments still timeout (would indicate PostDecodeFixedOverhead not used correctly)

## H-moe-fix: Interleaved MoE Split Improves Scout Predictions

**Prediction**: Scout experiments (4/15) will have mean TTFT APE **<50%** and mean E2E APE **<50%**, compared to iter8's 79-100% TTFT APE for Scout.

**Causal Mechanism**: The #877 fix correctly splits Scout's 48 layers into 24 MoE + 24 dense:
- MoE layers use `MoEExpertFFNDim` (per-expert dim) × `kEff` (top-k routing)
- Dense layers use `DenseIntermediateDim` (different from MoE expert dim)
- FP8 models (Scout) use `EffectiveWeightBytesPerParam = 1.0` and `TFlopsFP8` peak compute
- Without this: all 48 layers treated identically → wrong FLOPs and weight bytes for Scout

**Diagnostic Clause**: If Scout APE remains >70%, it indicates the MoE layer count or FFN dimension mapping is incorrect for Scout's architecture. Check InterleaveMoELayerStep semantics against HuggingFace config.json.

## H-reasoning-lite: No More 100% Timeout Errors

**Prediction**: All three reasoning-lite experiments will have finite E2E predictions (APE < 500%, not exactly 100% as in iter8/iter14).

**Causal Mechanism**: The 100% APE in iter8 was caused by β₇ × numDecodeRequests × 1448 decode steps accumulating to 100+ seconds, causing all requests to timeout within the simulation horizon. With β₇ removed from StepTime and per-request overhead in PostDecodeFixedOverhead (α₁ ≈ 1.85ms, applied once), the accumulated decode time drops from ~222 seconds to ~8 seconds (roofline-scale), well within any reasonable simulation horizon.

**Diagnostic Clause**: If reasoning-lite experiments still return exactly 100% APE, it indicates a different numerical issue (not the accumulation bug). Check whether the DES simulation horizon is too short for 1448-token outputs at the predicted step time.

## H-warm-start: Trained-Roofline Priors Accelerate Convergence

**Prediction**: The optimizer will converge within **1000 trials** (vs 2000), with best coefficients within **±50% of the trained-roofline fitted values** for β₁-β₃.

**Causal Mechanism**: Warm-starting from the trained-roofline's 7% MAPE coefficients places the optimizer near a known-good minimum. The dataset shifted (reasoning-lite, Scout general-lite, #877 FLOPs), so coefficients will adjust, but the architecture is proven and the starting point is close.

**Diagnostic Clause**: If coefficients diverge >2× from priors (e.g., β₂ > 2.5 or β₁ < 0.3), the dataset shift is larger than expected and cold-start exploration may be needed in iter16.
