# H-Step-Quantum: Step-Time Quantum vs DES-to-M/M/1 Wait-Time Divergence

**Status:** Refuted
**Resolution:** Refuted -- wrong mental model. The DES-to-M/M/1 divergence is NOT primarily caused by discrete step-time quantization. Scaling beta coefficients down by 100x increases the percentage divergence from 47-78% to 98-99%, because the alpha model overhead (which does not block the server) increasingly dominates the calibrated service time. The root cause of the original divergence is the split between "server-blocking time" (step time only) and "request E2E time" (steps + alpha overhead), which makes the effective DES utilization lower than the nominal M/M/1 rho.
**Family:** Structural model
**VV&UQ:** Validation
**Tier:** Tier 1 (grounds other experiments)
**Type:** Statistical (Monotonicity)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> Reducing the DES step-time quantum (by scaling beta coefficients) should proportionally reduce the DES-to-M/M/1 mean wait time divergence. Specifically: at rho=0.7, the W_q error (currently ~60% with ~6.9ms steps) should scale linearly with step_time / mean_service_time, approaching 0% as step time -> 0.

## Experiment Design

**Classification:** Statistical / Monotonicity

**Configurations compared:**
- A (baseline): beta=[6910.42, 17.67, 2.84], alpha=[1601.35, 3.51, 1805.54], step_time ~6913 us
- B (10x smaller): beta=[691.042, 1.767, 0.284], alpha=[1601.35, 3.51, 1805.54], step_time ~691 us
- C (100x smaller): beta=[69.104, 0.177, 0.028], alpha=[1601.35, 3.51, 1805.54], step_time ~69 us

**All three configurations use identical:**
- `--model meta-llama/llama-3.1-8b-instruct --num-instances 1 --max-num-running-reqs 1`
- `--scheduler fcfs --admission-policy always-admit --total-kv-blocks 1000000`
- Workload: Poisson arrivals, constant input=1, exponential output mean=128
- Seeds: 42, 123, 456

**Controlled variables:** Alpha coefficients, input/output token distribution, admission policy, scheduler, instances, KV blocks
**Varied variable:** Beta coefficients (scaling factor: 1.0, 0.1, 0.01)
**Seeds:** 42, 123, 456
**Preconditions verified:**
- Calibration at each scale: 10 requests, constant output=128, very low rate (0.01 req/s)
- 1.0x: service_time=1126.2 ms, mu=0.888 req/s (matches H-MMK calibration)
- 0.1x: service_time=323.6 ms, mu=3.090 req/s
- 0.01x: service_time=243.3 ms, mu=4.109 req/s
- Arrival rates correctly computed as lambda = rho * mu for each scale

**Config diff against H-MMK (ED-6):**
- Added: `--beta-coeffs` and `--alpha-coeffs` explicit flags (H-MMK used defaults, which produce the 1.0x values)
- Removed: Sub-experiments 2 and 3 (cluster mode with routing)
- Changed: Added rho=0.9 (H-MMK used 0.85)
- Identical: All other flags, workload parameters, seed set

## Results

### Per-Scale W_q Comparison (3-seed average, 2000 requests/run)

| Scale | Step (us) | Svc (ms) | rho=0.3 err | rho=0.5 err | rho=0.7 err | rho=0.9 err |
|-------|-----------|----------|-------------|-------------|-------------|-------------|
| 1.0x  | 6913      | 1126.2   | -47.2%      | -50.5%      | -59.6%      | -78.3%      |
| 0.1x  | 691       | 323.6    | -93.6%      | -95.6%      | -97.2%      | -99.0%      |
| 0.01x | 69        | 243.3    | -98.4%      | -99.3%      | -99.7%      | -99.9%      |

**All errors are negative** (DES W_q < M/M/1 analytical). Error **increases** as beta decreases -- the opposite of the prediction.

**Monotonicity check:** 0/4 utilization levels show decreasing error with decreasing step quantum. The hypothesis is refuted at all operating points.

### Service Time Composition

| Scale | Service (ms) | Step total (ms) | Alpha total (ms) | Step fraction |
|-------|-------------|-----------------|-------------------|---------------|
| 1.0x  | 1126.2      | 891.8           | 232.7             | 79.2%         |
| 0.1x  | 323.6       | 89.2            | 232.7             | 27.6%         |
| 0.01x | 243.3       | 8.9             | 232.7             | 3.7%          |

As beta decreases, the alpha overhead (constant at 232.7 ms) increasingly dominates the calibrated service time. At 0.01x, 96.3% of the "service time" is alpha overhead.

### Conservation: INV-1 OK across all 36 runs (3 scales x 4 rhos x 3 seeds)

### Per-Seed Consistency

| Scale | rho | seed 42 | seed 123 | seed 456 | CV |
|-------|-----|---------|----------|----------|-----|
| 1.0x  | 0.7 | 979.7   | 1037.3   | 1170.1   | 7.5% |
| 0.1x  | 0.7 | 20.0    | 20.0     | 22.9     | 6.7% |
| 0.01x | 0.7 | 1.8     | 1.8      | 1.9      | 1.9% |

Results are consistent across seeds (CV < 11% for all configs).

## Root Cause Analysis

### The DES has two distinct "service time" concepts (RCV-1, RCV-3)

The M/M/1 analytical model uses a single service time mu = 1/E[S] where E[S] is the calibrated E2E time at zero load. But the BLIS DES has **two different time components** that contribute to E2E:

1. **Server-blocking time (step time):** Determined by `StepTime()` (`sim/latency_model.go:35-50`). This is the only component that advances the simulation clock and blocks the server from processing other requests. For a single request with output=128: step_total = prefill_step + 128 * decode_step.

2. **Alpha overhead (non-blocking):** Two components:
   - `QueueingTime()` (`sim/latency_model.go:53-57`): alpha0 + alpha1*inputLen = 1604.86 us. Applied as a delay between ArrivalEvent and QueuedEvent (`sim/event.go:31-35`). This delays enqueuing but does not block the server.
   - `OutputTokenProcessingTime()` (`sim/latency_model.go:60-62`): alpha2 = 1805.54 us per token. Added to each decode step's ITL (`sim/simulator.go:560`) and to TTFT/E2E metrics, but does NOT advance the simulation clock or delay the next step (`sim/simulator.go:618`: next step at `now + currStepAdvance`, not `now + currStepAdvance + outputProcessingTime`).

### Why reducing beta increases divergence (RCV-3)

The M/M/1 analytical W_q depends on **total service time** (1126 ms at 1.0x). The DES W_q depends only on **server-blocking time** (892 ms at 1.0x, 8.9 ms at 0.01x). The effective DES utilization is:

```
rho_eff = lambda * step_total_s   (not lambda * calibrated_service_s)
```

At nominal rho=0.7:
- 1.0x: rho_eff = 0.622 * 0.892 = 0.554 (server utilization)
- 0.1x: rho_eff = 2.163 * 0.089 = 0.193
- 0.01x: rho_eff = 2.877 * 0.009 = 0.026

At 0.01x, the DES server is idle 97.4% of the time despite the nominal M/M/1 rho being 0.7. Requests find an empty queue on almost every arrival, so W_q approaches the minimum (alpha0 queueing delay of ~1.6 ms).

Reducing beta shrinks step_total while leaving alpha_total constant. This **widens** the gap between nominal rho (calibrated from E2E) and effective rho (from step time only), making the DES look less loaded and the M/M/1 comparison worse.

### The H-MMK 47-71% divergence explained (RCV-2)

At the 1.0x baseline:
- Step fraction = 79.2%, so the effective utilization is ~79.2% of nominal
- At nominal rho=0.7: lambda = 0.622 req/s, mu_eff = 1/step_total = 1.121 req/s
- rho_eff = lambda / mu_eff = 0.622 / 1.121 = 0.555
- M/M/1 W_q at rho_eff=0.555 with mu_eff=1.121: rho_eff/(mu_eff*(1-rho_eff)) = 0.555/(1.121*0.445) = 1112 ms
- DES W_q observed: 1062 ms → error vs corrected M/M/1: -4.5%

This first-principles calculation shows that **most of the H-MMK divergence disappears** when using step_total as the service time. The alpha=0 control experiment confirms this: DES W_q at rho=0.7 is 2074 ms vs M/M/1 prediction 2081 ms (-0.3% error). The 47% minimum divergence at rho=0.3 in H-MMK is almost entirely from calibrating mu using E2E (which includes non-blocking alpha overhead) instead of step_total.

### Control experiment: alpha=0 (RCV-4) -- Round 2

To confirm the mechanism, a control experiment was run with alpha=[0, 0, 0] (no alpha overhead) while keeping beta at 1.0x. This eliminates the blocking/non-blocking split, making the calibrated service time equal to the step total (E2E = 891.8 ms = step_total, mu = 1.121 req/s).

| rho | W_q M/M/1 (ms) | W_q DES alpha=0 (ms) | Error | Original error (alpha != 0) |
|-----|----------------|----------------------|-------|---------------------------|
| 0.3 | 382.20         | 356.89               | -6.6% | -47.2%                    |
| 0.5 | 891.79         | 853.08               | -4.3% | -50.5%                    |
| 0.7 | 2080.85        | 2074.04              | -0.3% | -59.6%                    |
| 0.9 | 8026.13        | 6908.47              | -13.9%| -78.3%                    |

**The control confirms the mechanism.** Removing alpha overhead reduces the divergence from 47-78% to 0.3-14%. At rho=0.5 and rho=0.7, the DES is within the 5% M/M/1 tolerance. The remaining residual divergence at rho=0.3 (6.6%) and rho=0.9 (13.9%) is consistent with the discrete step effect and exponential service time approximation (M/G/1 correction).

**Interpretation:** The alpha overhead IS the dominant source of DES-to-M/M/1 divergence. With alpha=0, the DES closely matches M/M/1 across the utilization range, with residual error from discrete step quantization growing modestly at high rho. The original H-MMK "discrete step processing" mechanism was a secondary effect, not the primary one.

## Devil's Advocate (RCV-5)

**If this is "Refuted," argue why it might be Confirmed:**
The original hypothesis claimed step-time quantum causes divergence. At the 1.0x scale, the step fraction is 79.2%, meaning step quantization IS the dominant contributor to the server-blocking time. The divergence at 1.0x (47-78%) is partly due to discrete steps -- the experiment just didn't isolate this effect properly because reducing beta also changes the alpha/beta ratio. A better test would hold the alpha/beta ratio constant (scale both equally).

**Counter-argument (why Refuted is correct):**
Even at 1.0x, the first-principles calculation shows that 20% of the service time is non-blocking alpha overhead, which alone accounts for most of the low-rho divergence (47%). The discrete step effect contributes a second-order correction. The hypothesis's prediction was clearly wrong: divergence increases 2x (from 47% to 98%) instead of decreasing to ~5%. The mental model was wrong about the primary mechanism.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| DES has split service time: server-blocking (beta/step) vs non-blocking (alpha overhead) | **Design limitation** | File `--label design` issue for documentation and potential DES calibration guidance |
| Alpha overhead (output processing, queueing delay) does not advance simulation clock | **Confirmation** | Documented here -- this is by design, matching vLLM's architecture where post-processing is non-blocking |
| H-MMK's 47-71% divergence is primarily explained by the alpha/beta split, not discrete step quantization | **Surprise** | Overturns the proposed mechanism in H-MMK FINDINGS.md. File update to H-MMK documentation. |
| DES effective utilization = lambda * step_total, not lambda * E2E_total | **New insight** | Document as guidance for users comparing DES to analytical models |
| Conservation (INV-1) holds across all 36 runs | **Confirmation** | Documented here |
| Reducing step quantum increases M/M/1 divergence (opposite of prediction) | **Refutation** | Documented here -- the hypothesis is wrong |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **No.** The alpha overhead design is intentional (models vLLM post-processing).
- [x] Any new rules needed? **Yes.** Proposed: "When comparing DES to analytical queueing models, use step_total as the service time (not E2E), since alpha overhead does not block the server."
- [x] Any new invariants needed? **No.**
- [x] Any existing rules/invariants confirmed? **INV-1** (conservation) confirmed across all 36 runs.

## Scope and Limitations (RCV-6)

- **Operating point tested:** k=1, rho = {0.3, 0.5, 0.7, 0.9}, seeds = {42, 123, 456}, 2000 requests/run, beta scales = {1.0, 0.1, 0.01}, constant input=1, exponential output mean=128
- **Parameters findings depend on:** `--max-num-running-reqs 1` (single-server-per-instance). With batch sizes > 1, the alpha/beta split has different dynamics -- multiple requests share the same step time.
- **What was NOT tested:**
  - Batch sizes > 1 (typical inference serving uses 32-256)
  - Non-zero alpha scaling (what happens if we also scale alpha proportionally?)
  - Roofline mode (different step time model)
  - Different input/output token distributions
  - Multi-instance (k > 1) configurations
- **Generalizability:** The alpha/beta split finding generalizes to ALL BLIS configurations using the blackbox latency model, since the architecture (alpha overhead not advancing the clock) is fundamental. However, the quantitative impact varies with the alpha/beta ratio, which depends on input/output token counts and model coefficients.
- **Uncertainty quantification:** The effect is robust across seeds (CV < 11%). The direction (reducing beta increases divergence) is consistent across all 12 (rho, scale) combinations. No UQ needed for the qualitative finding.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Divergence direction (DES < M/M/1) | Consistent across all 36 runs | **High** -- same direction, growing with rho |
| Divergence increases with decreasing beta | 0/4 rhos show decreasing error | **High** -- opposite of prediction in all cases |
| Root cause: alpha/beta split | Step fraction ranges from 79.2% to 3.7% | **High** -- computed analytically AND confirmed by DES data |
| First-principles rho_eff calculation | Matches DES behavior qualitatively and quantitatively | **High** -- confirmed by alpha=0 control (divergence drops from 47-78% to 0.3-14%) |
| Conservation (INV-1) | Holds in all 36 runs | **High** |

## Implications for Users

1. **Do NOT calibrate M/M/1/M/M/k comparisons using E2E at zero load.** The calibrated E2E includes alpha overhead that does not block the DES server. Instead, compute the effective service time from beta coefficients directly: `step_total = (beta0 + beta1*input) + output_tokens * (beta0 + beta2)` for a single request.

2. **The effective DES utilization is lower than you think.** For the default llama-3.1-8b coefficients with input=1, output=128: `rho_eff = 0.792 * rho_nominal`. At nominal rho=0.7, the DES server is only at 55.4% utilization.

3. **The alpha overhead models post-processing and network delays** that in real vLLM are non-blocking (happen after the step completes and the server is ready for the next batch). This is architecturally correct -- the mismatch is with M/M/1, not with vLLM.

4. **For analytical model comparison, use step_total as the service time.** This gives: mu_eff = 1000 / step_total_ms. For the H-MMK experiment at 1.0x: mu_eff = 1000/891.8 = 1.121 req/s (vs calibrated mu = 0.888 req/s).

## Reproducing

```
cd hypotheses/h-step-quantum
./run.sh
```
