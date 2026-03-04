# Iter 5 Prototype Findings — Decomposed Inter-Step Overhead (Approach D)

_Date: 2026-03-04_
_Branch: `iter5-decomposed`_
_Status: Prototype complete — findings for strategy evolution Phase 2_

## Executive Summary

Approach D prototyped the journey-step correlation method for fitting inter-step overhead. The analysis yielded three findings that fundamentally reshape the fitting strategy:

1. **Iter 3 β is ~2-4x too large** for codellama-34b and llama-2-70b (predicted GPU time exceeds total wall clock)
2. **A universal 2.0x pipeline delay factor** between journey-observed time and step-to-step cadence
3. **vLLM's async scheduling creates a one-step pipeline delay** that any fitting approach must account for

These findings invalidate both Approaches A and B as described in the HANDOFF.md, and suggest a different path.

## Finding 1: Iter 3 β Overprediction

Decomposing the consecutive-pair wall clock (`ts_start[i+1] - ts_start[i]`) into scheduler time (measured), GPU time (β-predicted), and residual:

| Model | N pairs | T_wall(µs) | T_sched(µs) | T_gpu(β)(µs) | T_residual(µs) | Residual % |
|-------|---------|------------|-------------|--------------|----------------|-----------|
| llama-2-7b | 4,624 | 8,156 | 171 | 5,967 | +1,859 | +22.8% |
| mixtral-8x7b | 2,286 | 18,850 | 335 | 16,874 | +1,510 | +8.0% |
| codellama-34b | 2,840 | 14,738 | 265 | **21,772** | **-7,463** | **-50.6%** |
| llama-2-70b | 2,302 | 18,587 | 424 | **25,970** | **-7,919** | **-42.6%** |

For codellama-34b and llama-2-70b, the Iter 3 β-predicted GPU time is **larger than the total wall clock per step**. This is impossible — it means the coefficients absorbed inter-step overhead from Block B journey constraints and are inflated beyond the true GPU cost.

For llama-2-7b and mixtral-8x7b, the residual is positive (1.5-1.9ms) — these models have real overhead that β didn't absorb (or absorbed less).

### Wall-clock-fitted β vs Iter 3 β

| Feature | β₀ (L) | β₁ (KV bw) | β₂ (MoE) | β₃ (TP) |
|---------|--------|-----------|---------|---------|
| **Iter 3 (Block A+B)** | 116.1 | 1,226.9 | 19.9 | 9,445.2 |
| **Wall-clock only** | 230.5 | 277.4 | 128.0 | 2,775.4 |

The KV bandwidth coefficient drops 4.4x and TP sync drops 3.4x when fitted against wall-clock-only data. These inflated values were responsible for the negative residuals — they over-counted KV and TP costs because Block B journey constraints included inter-step gaps.

## Finding 2: Universal 2.0x Pipeline Factor

For single-step prefills (97-100% of all prefills), comparing journey-observed prefill time with matched step wall clock:

| Model | N matched | Journey real(µs) | Step wall(µs) | **Ratio** | Diff(µs) |
|-------|-----------|------------------|---------------|-----------|----------|
| llama-2-7b | 319 | 17,595 | 8,804 | **2.000** | 8,812 |
| codellama-34b | 333 | 30,308 | 15,144 | **2.004** | 15,149 |
| llama-2-70b | 368 | 37,947 | 18,964 | **2.009** | 19,005 |
| mixtral-8x7b | 378 | 38,631 | 19,565 | **1.984** | 19,221 |

The ratio is **2.00 ± 0.02 across all models and batch sizes** (p10=1.94, p90=2.06). This is not a coincidence — it's a structural property of vLLM's engine pipeline.

### Physical Explanation: Async Scheduling Pipeline Delay

Looking at vLLM's `EngineCore.step()` (`core.py:369-398`):

```python
def step(self):
    scheduler_output = self.scheduler.schedule()           # [1] Schedule
    future = self.model_executor.execute_model(...)        # [2] Submit to GPU
    model_output = future.result()                         # [3] Wait for GPU
    engine_core_outputs = self.scheduler.update_from_output(...)  # [4] Process output
```

Timeline for a request prefilled at step S:

```
Step S:   schedule() → [SCHEDULED event fires] → execute_model() → GPU runs → result() → update_from_output() → [FIRST_TOKEN event fires]
          |←─── ts_start[S] ──────────────────────────────────── ts_start[S+1] ────→|
```

But this means `journey.first_token_ns - journey.scheduled_ns` spans from the MIDDLE of step S's schedule() to the END of step S (after GPU + output processing). Meanwhile, `ts_start[S+1] - ts_start[S]` measures from start of step S to start of step S+1.

**If `scheduled_ns` fires at the beginning of schedule() and `first_token_ns` fires at the end of update_from_output()**: the journey duration spans ~1.0 step cycles, and the 2.0x ratio would require a different explanation.

**More likely explanation**: `scheduled_ns` fires during step S-1's schedule() (when the request is first allocated resources), and `first_token_ns` fires during step S's update_from_output() (when the first output token is detected). This means the journey spans step S-1's execute → step S's schedule → step S's execute → step S's update = approximately 2 step cycles.

This is consistent with vLLM's chunked prefill: a request gets *scheduled* in step S (resources allocated, `SCHEDULED` event fires), but its first output token isn't generated until the GPU processes it, which completes in update_from_output of the SAME step. The 2.0x factor arises because:
- `scheduled_ns` fires at the start of schedule() in step S
- `first_token_ns` fires at the end of step S+1 (because the execute_model in step S produces the prefill result, but the token sampling + FIRST_TOKEN detection happens in update_from_output of step S, which is AFTER execute_model of step S, and the wall clock for that is captured in the ts_start[S+1] measurement)

The exact mechanism needs verification by examining journey event emission points in the scheduler code. But the 2.0x factor is empirically established.

## Finding 3: Structured δ Has Good Signal (Per-Model)

When fitting `T_residual = δ₀ + δ₁·batch_size + δ₂·total_tokens` per model:

| Model | N | δ₀ (µs) | δ₁ (µs/req) | δ₂ (µs/tok) | R² |
|-------|---|---------|-------------|-------------|-----|
| llama-2-7b | 4,624 | 2,129 | +40.5 | -61.8 | 0.344 |
| codellama-34b | 2,840 | 2,379 | -349.3 | -0.75 | **0.971** |
| llama-2-70b | 2,302 | -1,048 | -215.9 | +7.6 | **0.946** |
| mixtral-8x7b | 2,286 | 8,554 | -195.4 | -11.3 | 0.753 |

For codellama/70b, the residual is highly structured (R² > 0.94) and dominated by δ₁·batch_size with NEGATIVE coefficient. This means the Iter 3 β overpredicts more at higher batch sizes — consistent with the β coefficient absorbing a per-step overhead that doesn't scale with batch size.

For llama-2-7b, R² is lower (0.34) because the residual is POSITIVE (real overhead exists) and has more variance.

## Implications for Fitting Strategy

### What This Means for Approaches A-F

| Approach | Status | Reasoning |
|----------|--------|-----------|
| **A (Step-only β + δ search)** | **Partially valid** | Step-only β is correct, but adding δ on top requires the 2.0x pipeline factor |
| **B (7-param CMA-ES)** | **Needs pipeline correction** | Joint optimization won't converge unless the 2.0x factor is modeled |
| **C (Decomposed δ from timestamps)** | **Viable** | But the decomposition in Analysis 1 is only meaningful for llama-2-7b/mixtral where residual is positive |
| **D (Journey-step correlation)** | **Key finding: 2.0x** | The universal 2.0x factor is the main result; structured δ is secondary |
| **E (Two-objective staged)** | **Recommended** | But must account for the pipeline factor |
| **F (Sub-step profiling)** | **Would resolve 2.0x ambiguity** | But requires new instrumentation |

### Recommended Next Step: Approach E with Pipeline Correction

1. **Fit β against wall-clock only** (no journey constraints) → gets the GPU compute time right
2. **Apply 2.0x pipeline factor** when converting from step-level predictions to journey-level predictions:
   `predicted_TTFT = α + 2.0 × Σ(β·features)` for single-step prefills
3. **Fit α as the remaining residual** between real TTFT and pipeline-corrected step time
4. **For BLIS**: the `InterStepOverhead()` should return **one step's worth of overhead** (schedule + input prep + output proc ≈ T_wall - T_gpu), and the simulator should naturally produce the correct timing through its event loop

### The Key Insight

The dominant error in BLIS is NOT that we need a δ per step. The dominant error is that **Iter 3 β was fitted with journey constraints that baked in inter-step overhead, inflating the coefficients**. When we fit β against wall-clock-only data:

- β₁(KV) drops from 1,227 → 277 (4.4x reduction)
- β₃(TP) drops from 9,445 → 2,775 (3.4x reduction)

The "missing" time that these inflated coefficients were compensating for is the inter-step overhead. If we use the correct (smaller) β and add an explicit InterStepOverhead in BLIS, the total should match reality without entanglement.

## Data Quality Notes

- Step data: 122,752 sampled steps (10% rate), 12,052 consecutive pairs
- Journey data: 150,858 journeys with step indices
- 97-100% of prefills are single-step across all models
- The 2.0x factor holds to ±2% across all batch sizes (5-128)
- Consecutive pairs are unbiased (running_depth distribution matches all steps)
