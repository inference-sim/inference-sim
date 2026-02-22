# H-Phase-Structure: Latency Model Phase Linearity Validation

**Status:** Confirmed
**Resolution:** Clean confirmation — TTFT is perfectly linear in input tokens and decode time is perfectly linear in output tokens. Adjusted R² = 1.000000 across all seeds for both phases. Measured slopes match analytical predictions from alpha/beta coefficients within 0.01%.
**Family:** Structural model
**VV&UQ:** Validation
**Tier:** Tier 1 (foundational — grounds all latency-based experiments)
**Type:** Statistical (Monotonicity — linearity is strict monotonicity with constant slope)
**Date:** 2026-02-21
**Rounds:** 1

## Hypothesis

> Component: latency model. Assumption: prefill cost is proportional to prompt token count and decode cost is proportional to generated token count. Observable: TTFT should be linear in input_tokens (R² > 0.95 for linear fit) with output held constant, and (E2E − TTFT) should be linear in output_tokens (R² > 0.95) with input held constant.

## Experiment Design

**Classification:** Statistical / Monotonicity (linearity = constant slope)

**Two sub-experiments:**

**Experiment A — TTFT vs Input Tokens:**
- Fixed output = 128 tokens (constant distribution)
- Input ∈ {64, 128, 256, 512, 1024} (constant distribution at each level)
- Single instance, `--max-num-running-reqs 1`, rate = 0.01 req/s, 20 requests per level
- Measure mean TTFT at each level, fit linear regression, report R²

**Experiment B — Decode Time (E2E − TTFT) vs Output Tokens:**
- Fixed input = 256 tokens (constant distribution)
- Output ∈ {64, 128, 256, 512, 1024} (constant distribution at each level)
- Same single-instance, no-queueing configuration
- Measure mean (E2E − TTFT) at each level, fit linear regression, report R²

**Configurations compared:**
- 5 input levels (Exp A) and 5 output levels (Exp B), all else held constant
- Common flags: `--model meta-llama/llama-3.1-8b-instruct --num-instances 1 --max-num-running-reqs 1 --scheduler fcfs --admission-policy always-admit --total-kv-blocks 1000000`

**Controlled variables:** Model (llama-3.1-8b, H100, TP=2), KV blocks (1M — no pressure), scheduler (FCFS), admission (always-admit), rate (0.01 req/s — zero queueing)

**Varied variable:** Input token count (Exp A) or output token count (Exp B)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- Calibration: E2E for (input=1, output=128) = 1126.241 ms (matches H-MMK calibration exactly)
- Rate 0.01 req/s → P(queueing) ≈ 1.1% for Exp A (service ~1.1s), ~8.5% for Exp B at output=1024 (service ~8.9s). Exp B is immune: E2E-TTFT cancels scheduling delay.
- Constant distributions → all requests at a given level are identical

**Reference:** `hypotheses/h-mmk-validation/run.sh` (ED-6 config diff: this experiment uses constant distributions on both input and output; H-MMK used constant input + exponential output. Rate reduced to 0.01. Single instance only.)

## Results

### Experiment A: TTFT vs Input Tokens

#### Raw TTFT (includes scheduling/queueing delay)

| Input | seed=42 | seed=123 | seed=456 | Mean |
|------:|--------:|---------:|---------:|-----:|
| 64 | 11.672 | 11.672 | 11.672 | 11.672 |
| 128 | 13.027 | **42.389** | 13.027 | 22.814 |
| 256 | 15.739 | 15.739 | 15.739 | 15.739 |
| 512 | 21.161 | 21.161 | 21.161 | 21.161 |
| 1024 | 32.008 | 32.008 | 32.008 | 32.008 |

| Seed | Slope (ms/tok) | Intercept (ms) | R² | Pass |
|-----:|---------------:|---------------:|---:|-----:|
| 42 | 0.021183 | 10.316 | 1.000000 | YES |
| 123 | 0.008234 | 21.327 | 0.065713 | NO |
| 456 | 0.021183 | 10.316 | 1.000000 | YES |

Seed 123 fails due to one Poisson arrival event at input=128 that queued behind the previous request (TTFT=42.389 vs expected 13.027). This is queueing noise, not model non-linearity.

#### Adjusted TTFT (scheduling delay subtracted — pure latency model)

| Input | seed=42 | seed=123 | seed=456 | Mean |
|------:|--------:|---------:|---------:|-----:|
| 64 | 9.846 | 9.846 | 9.846 | 9.846 |
| 128 | 10.977 | 10.977 | 10.977 | 10.977 |
| 256 | 13.239 | 13.239 | 13.239 | 13.239 |
| 512 | 17.762 | 17.762 | 17.762 | 17.762 |
| 1024 | 26.810 | 26.810 | 26.810 | 26.810 |

| Seed | Slope (ms/tok) | Intercept (ms) | R² | Pass |
|-----:|---------------:|---------------:|---:|-----:|
| 42 | 0.017671 | 8.715 | **1.000000** | YES |
| 123 | 0.017671 | 8.715 | **1.000000** | YES |
| 456 | 0.017671 | 8.715 | **1.000000** | YES |

**Analytical prediction:** slope = β₁ = 17.67 μs/tok = 0.01767 ms/tok; intercept = β₀ + α₂ = 8716.0 μs = 8.716 ms
**Measured:** slope = 0.017671 ms/tok (error: <0.01%); intercept = 8.715 ms (0.01% error due to int64 truncation in `getStepTime()`)

### Experiment B: Decode Time (E2E − TTFT) vs Output Tokens

| Output | seed=42 | seed=123 | seed=456 | Mean |
|-------:|--------:|---------:|---------:|-----:|
| 64 | 557.952 | 557.952 | 557.952 | 557.952 |
| 128 | 1115.904 | 1115.904 | 1115.904 | 1115.904 |
| 256 | 2231.808 | 2231.808 | 2231.808 | 2231.808 |
| 512 | 4463.616 | 4463.616 | 4463.616 | 4463.616 |
| 1024 | 8927.232 | 8927.232 | 8927.232 | 8927.232 |

| Seed | Slope (ms/tok) | Intercept (ms) | R² | Pass |
|-----:|---------------:|---------------:|---:|-----:|
| 42 | 8.718000 | -0.000 | **1.000000** | YES |
| 123 | 8.718000 | -0.000 | **1.000000** | YES |
| 456 | 8.718000 | -0.000 | **1.000000** | YES |

**Analytical prediction:** slope = β₀ + β₂ + α₂ = 6910.42 + 2.84 + 1805.54 = 8718.80 μs/tok = 8.719 ms/tok
**Measured:** slope = 8.718 ms/tok (error: <0.01% — int64 truncation: `int64(6913.26)` = 6913, `int64(1805.54)` = 1805, total 8718 vs 8718.80); intercept = -0.000 ms

## Root Cause Analysis

### Why the latency model is perfectly linear (RCV-1)

The DES step time formula is `stepTime = beta0 + beta1*cacheMissTokens + beta2*decodeTokens` (`sim/simulator.go:379-381`). With `max-num-running-reqs=1`:
- Prefill step: cacheMissTokens = input_tokens, decodeTokens = 0 → stepTime = β₀ + β₁×input
- Decode step: cacheMissTokens = 0, decodeTokens = 1 → stepTime = β₀ + β₂

Both are exactly linear in the token count. The TTFT additionally includes `getOutputTokenProcessingTime()` which returns α₂ = 1805.54 μs (`sim/simulator.go:631`), a constant. Each ITL entry adds `currStepAdvance + getOutputTokenProcessingTime()` (`sim/simulator.go:627,659`).

### Why raw TTFT has queueing noise (RCV-3)

TTFT is defined as `now + currStepAdvance + alpha2 - ArrivalTime` (`sim/simulator.go:631`). The `(now - ArrivalTime)` component includes both:
1. **Alpha model delay:** α₀ + α₁×input = deterministic, linear in input
2. **Actual queue wait:** stochastic, depends on Poisson inter-arrival timing

At rate=0.01 with Poisson arrivals, P(next request arrives during ~1.1s service) ≈ 1 - e^(-0.01×1.13) ≈ 1.1%. Seed 123 hit this at input=128, adding ~29.4ms mean queueing delay. Subtracting the per-request scheduling delay (`sim/metrics.go:158`) removes both components, leaving only the step time + α₂.

### Why E2E - TTFT is immune to queueing noise (RCV-3)

Decode time = E2E - TTFT = Σ(ITL). Each ITL = stepAdvance + α₂. The scheduling delay appears in both E2E and TTFT and cancels in the subtraction. This is why Experiment B shows zero seed-to-seed variation.

### Control experiment confirmation (RCV-4)

Seeds 42 and 456 had zero queueing (R² = 1.000000 for raw TTFT). This serves as a natural control — the same configuration with different Poisson realizations that happened to avoid queueing shows perfect linearity, confirming that the non-linearity in seed 123 is purely from queueing, not from the latency model.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**
The R² = 1.000000 results are suspiciously perfect — this could indicate that the DES is too simple rather than that the model is correct. A real LLM serving system has non-linear effects (attention scaling as O(n²), memory bandwidth saturation, batch scheduling overhead) that the alpha/beta linear model ignores by design. Confirming linearity at the model level says nothing about whether the model accurately predicts real system behavior. This experiment validates internal consistency, not external validity.

**Counter-counter:** True, but internal consistency is the prerequisite. If the DES didn't even implement its own coefficients correctly (e.g., off-by-one in token counting, integer truncation), no amount of coefficient fitting would produce accurate predictions. This experiment confirms the math is correct; coefficient accuracy is a separate validation question.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| TTFT is perfectly linear in input tokens (R² = 1.000000, slope matches β₁) | Confirmation | Documented here |
| Decode time is perfectly linear in output tokens (R² = 1.000000, slope matches β₀+β₂+α₂) | Confirmation | Documented here |
| Raw TTFT contaminated by Poisson queueing noise at rate=0.01 | Design limitation | Documented here — users should use adjusted TTFT for phase analysis |
| Scheduling delay subtraction cleanly isolates latency model from queueing effects | Confirmation | Documented here — useful technique for future experiments |
| Analytical slope predictions match measured slopes within 0.01% | Confirmation | Validates analytical prediction formulas in `docs/standards/experiments.md` |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found**
- [x] Any new rules needed? **None** — the queueing noise finding is a methodological insight, not a new rule
- [x] Any new invariants needed? **None** — phase structure linearity is already documented in `docs/standards/experiments.md` (cross-validation section)
- [x] Any existing rules/invariants confirmed? **Yes** — INV-6 (determinism) confirmed: adjusted TTFT identical across all seeds. Phase structure cross-validation from `docs/standards/experiments.md` confirmed with R² > 0.95 (actually = 1.0)

## Scope and Limitations (RCV-6)

- **Operating point tested:** Single instance, max-running-reqs=1, rate=0.01 req/s, FCFS scheduler, always-admit, 1M KV blocks (no pressure), llama-3.1-8b/H100/TP=2
- **Parameters findings depend on:** max-num-running-reqs=1 (eliminates batching effects), constant distributions (eliminates stochastic variation in token counts), low rate (minimizes queueing)
- **What was NOT tested:**
  - Batched execution (max-running-reqs > 1): with multiple requests in a batch, the per-request share of step time is non-trivial. The linear relationship may not hold per-request in batched mode.
  - Variable distributions (Gaussian, exponential): the per-request relationship is still linear, but averaging across requests with different token counts would show the mean relationship.
  - Roofline mode: analytical FLOPs/bandwidth estimation may have different non-linearities (e.g., attention quadratic scaling). This is hypothesis H19.
  - Very large token counts (>1024): the linear beta model may break down at extreme sequence lengths due to memory bandwidth saturation not modeled by the coefficients.
- **Generalizability:** The phase structure linearity holds for any alpha/beta coefficient set and any token count range where the coefficient model is used. This is a **structural claim** about the code (a linear combination of inputs produces linear output), not an empirical generalization from testing multiple coefficient sets. The claim follows from the code at `sim/simulator.go:379-381` being a 3-term linear combination.
- **Uncertainty quantification:** UQ not applicable — the relationship is deterministic under controlled conditions. The reported R² = 1.000000 is a display artifact of `:.6f` formatting; the true value is approximately 0.9999999 due to sub-microsecond residuals from `int64()` truncation in `getStepTime()`. The only macroscopic uncertainty is the queueing noise in raw TTFT, which is fully explained by Poisson arrival statistics.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Adjusted TTFT R² | 1.000000 (all 3 seeds) | High — deterministic given constant distributions |
| Decode R² | 1.000000 (all 3 seeds) | High — deterministic given constant distributions |
| TTFT slope vs β₁ | 0.0% error | High — exact match |
| Decode slope vs β₀+β₂+α₂ | 0.0% error | High — exact match |
| Sample size | 3 seeds × 5 levels × 20 requests = 300 requests per experiment | Adequate (constant distributions make >1 request per level redundant) |
| Mechanism | Linear beta coefficient model (sim/simulator.go:378-380) | High — code-traced, analytically predicted, experimentally confirmed |

## Implications for Users

1. **TTFT prediction:** For single-request inference, TTFT ≈ (α₀ + β₀ + α₂) + (α₁ + β₁) × input_tokens. For llama-3.1-8b on H100/TP=2: TTFT ≈ 10.317 ms + 0.02118 ms × input_tokens.

2. **Decode time prediction:** Total decode time ≈ output_tokens × (β₀ + β₂ + α₂). For llama-3.1-8b on H100/TP=2: decode ≈ 8.719 ms × output_tokens.

3. **E2E prediction:** E2E ≈ TTFT + decode ≈ 10.317 + 0.02118 × input + 8.719 × output (milliseconds). At the calibration point (input=1, output=128): 10.317 + 0.021 + 1116.0 = 1126.3 ms — matches the 1126.241 ms calibration.

4. **Methodological note:** When analyzing TTFT in experiments with queueing, subtract the per-request scheduling delay (`scheduling_delay_ms / 1000` in the per-request JSON) to isolate the latency model from queueing effects.

## Reproducing

```bash
cd hypotheses/h-phase-structure
./run.sh
```
