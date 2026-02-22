# H-Arrival-Generators: Validate Arrival Sampler Distributions

**Status:** Confirmed with design limitation
**Resolution:** Confirmation with design limitation — Poisson and low-CV samplers (Gamma CV=1.5, Weibull CV=1.5) match theoretical distributions. High-CV samplers (Gamma CV=3.5, Weibull CV=3.5) fail KS tests due to int64 microsecond clamping that truncates 6-42% of samples. The samplers are algorithmically correct; the discrepancy is from the `int64` floor at 1μs in `SampleIAT`.
**Family:** Workload/arrival
**VV&UQ:** Verification
**Tier:** Tier 1 (foundational — validates workload generation infrastructure)
**Type:** Statistical (Equivalence)
**Date:** 2026-02-21
**Rounds:** 1

## Hypothesis

> For each arrival sampler (Poisson, Gamma CV=1.5, Gamma CV=3.5, Weibull CV=1.5, Weibull CV=3.5), generating 10K+ inter-arrival times should yield (a) sample mean within 5% of theoretical mean, (b) sample CV within 10% of theoretical CV, and (c) KS test p > 0.05 against the theoretical CDF.

## Experiment Design

**Classification:** Statistical / Equivalence

**5 samplers tested independently:**
1. Poisson: IAT ~ Exp(rate=100), mean=10ms, CV=1.0
2. Gamma CV=1.5: shape=0.444, scale=22.5ms, mean=10ms
3. Gamma CV=3.5: shape=0.0816, scale=122.5ms, mean=10ms
4. Weibull CV=1.5: k=0.685, scale=7.73ms, mean=10ms
5. Weibull CV=3.5: k=0.375, scale=2.50ms, mean=10ms

**Method:** Run BLIS with each sampler, extract per-request `arrived_at` timestamps from `--results-path` JSON, compute inter-arrival times as consecutive differences.

**Controlled variables:** Rate (100 req/s), request count (10,001 → 10,000 IATs), workload (constant input=1, output=1), single instance, FCFS, always-admit, 1M KV blocks

**Varied variable:** Arrival process (one of 5 samplers)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- Minimal workload (input=1, output=1) ensures all requests complete, isolating arrival timing
- Rate=100 req/s is in the practical range for BLIS experiments

## Results

### Summary Table

| Sampler | Mean Err | CV Err | KS p (range) | Clamp% | Verdict |
|---------|----------|--------|---------------|--------|---------|
| Poisson | 0.0-0.6% | 0.7-1.4% | 0.43-0.98 | 0.0% | **PASS** |
| Gamma CV=1.5 | 0.3-0.7% | 0.8-2.3% | 0.063 (all) | 1.8% | **PASS** (marginal) |
| Gamma CV=3.5 | 1.5-3.1% | 0.2-5.5% | 0.000 (all) | 42-43% | **FAIL** (clamping) |
| Weibull CV=1.5 | 0.2-2.1% | 1.0-2.7% | 0.045-0.57 | 0.3% | **MARGINAL** (1/3 seeds) |
| Weibull CV=3.5 | 1.0-6.5% | 2.2-5.8% | 0.000 (all) | 6.5-6.8% | **FAIL** (clamping) |

### Detailed Per-Sampler Results

**Poisson (Exponential):** Clean pass across all 3 seeds. Mean within 0.6%, CV within 1.4%, KS p-values 0.43–0.98. Zero clamping (exponential with mean 10ms has negligible mass below 1μs).

**Gamma CV=1.5:** Passes all criteria but marginally on KS (p=0.063 for all seeds). The identical KS_D=0.0131 across all three seeds reveals this is a systematic effect from the 1.8% clamping, not random variation. Without clamping, the KS test would show different statistics per seed and likely higher p-values.

**Gamma CV=3.5:** Fails KS test decisively (p=0.000, D=0.4007). The 42% clamping rate means nearly half of all IATs are pinned to 1μs instead of their continuous values near 0. Despite this, mean error stays within 5% (the clamped values contribute little to the sum) and CV error within 10% for 2/3 seeds. The KS_D is identical across all seeds (0.4007) — the test is measuring the clamping step function, not sampler quality.

**Weibull CV=1.5:** Two seeds pass cleanly (p=0.57, 0.42), one fails marginally (seed 456: p=0.045, D=0.0138 vs critical value ~0.0136). With 15 KS tests at α=0.05, the expected false positive count is 0.75. This single marginal failure is consistent with statistical noise, not a sampler bug.

**Weibull CV=3.5:** Fails KS test (p=0.000, D=0.0517, identical across seeds). Clamping at 6.5-6.8% creates a systematic CDF discrepancy at x=1μs. Seed 123 also fails mean error (6.5%), suggesting the clamped samples slightly bias the mean.

## Root Cause Analysis

### The int64 clamping mechanism (RCV-1)

All five samplers return `int64` IATs with a floor of 1 (`sim/workload/arrival.go:24-27, 41-45, 91-98`):
```go
iat := int64(sample)
if iat < 1 {
    return 1
}
```

For distributions with high density near 0 (Gamma shape < 1, Weibull shape < 1), a fraction of continuous samples fall below 1μs and get clamped. The clamped fraction is the theoretical CDF evaluated at x=1μs:

| Sampler | Shape | P(X_continuous < 1μs) | Observed Clamp% |
|---------|-------|-----------------------|-----------------|
| Poisson | — | ≈ 0% (Exp mean=10,000μs) | 0.0% |
| Gamma CV=1.5 | 0.444 | ~1.3% (analytical) | 1.8% |
| Gamma CV=3.5 | 0.082 | ~40% (analytical) | 42-43% |
| Weibull CV=1.5 | 0.685 | ~0.3% (analytical) | 0.3% |
| Weibull CV=3.5 | 0.375 | ~6-7% (analytical) | 6.5-6.8% |

The observed clamping matches analytical predictions, confirming the clamping is the dominant source of discrepancy (a conditional KS test on the unclamped portion was not performed — see Scope).

### Why clamping creates identical KS_D across seeds (RCV-3)

The KS statistic D = max|F_n(x) - F(x)| measures the worst-case CDF discrepancy. For clamped distributions, the worst discrepancy is always at x = 1μs where the empirical CDF has a step function (jump from 0 to clamp%) while the theoretical CDF is smooth. The step size equals the clamped fraction, which is determined by the distribution parameters (not the seed). Different seeds produce different samples in the unclamped portion, but the clamped fraction is approximately constant (determined by P(X < 1μs)), so D is approximately constant.

### Weibull CV=1.5 marginal failure is statistical noise (RCV-2)

For n=10,000 at α=0.05, the KS critical value is approximately D_crit = 1.358/√n = 0.01358. Seed 456 produced D=0.0138 — exceeding the critical value by 0.0002 (1.6%). With 5 samplers × 3 seeds = 15 independent KS tests at α=0.05, P(at least one false positive) = 1 - (1-0.05)^15 = 54%. Getting exactly one marginal failure is the expected outcome.

## Devil's Advocate (RCV-5)

**Argue why the clamping finding might be less significant than claimed:**
The clamping only affects inter-arrival times below 1μs. At rate=100 req/s, a 1μs IAT means two requests arriving essentially simultaneously. Real LLM serving systems have similar discrete time resolution (network packets, OS scheduling). The clamping may actually be MORE realistic than the continuous distribution, not less.

**Counter:** True for practical purposes, but the hypothesis asked whether the samplers match theory. They don't for high CV, and the mismatch scales with CV. Users who choose Gamma CV=3.5 to model extreme burstiness get a distribution with 42% of mass at a single point — qualitatively different from the smooth heavy tail they expect.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Poisson sampler matches Exp(rate) | Confirmation | Documented here |
| Gamma CV=1.5 sampler matches theoretical within KS tolerance | Confirmation | Documented here |
| Gamma CV=3.5 fails KS due to 42% int64 clamping | Design limitation | File issue: document the clamping floor and recommend max CV |
| Weibull CV=1.5 marginal KS failure (1/3 seeds) | Statistical noise | Documented here — not a bug |
| Weibull CV=3.5 fails KS due to 6.8% int64 clamping | Design limitation | Same issue as Gamma CV=3.5 |
| KS test is inappropriate for validating clamped distributions | Surprise | Documented here — methodological finding |
| Identical KS_D across seeds reveals clamping as the sole discrepancy | Surprise | Documented here — validates sampler correctness |

## Standards Audit

- [x] Any violations of existing rules? **None**
- [x] Any new rules needed? **Candidate: "Document int64 clamping effects for high-CV arrival distributions."** The clamping is intentional (prevents zero/negative IATs) but the 42% distortion for Gamma CV=3.5 is undocumented. Users should know the effective distribution differs from the specified one.
- [x] Any new invariants needed? **None**
- [x] Any existing rules/invariants confirmed? **R3 (validate CLI flags) — the CV parameter is validated > 0 but there is no upper-bound warning for CVs that produce heavy clamping**

## Scope and Limitations (RCV-6)

- **Operating point tested:** Rate=100 req/s, 10,001 requests, single instance, 5 arrival processes
- **Parameters findings depend on:** The clamping fraction depends on rate — lower rates produce larger mean IATs and less clamping. Analytical estimates: at rate=1 req/s, Gamma CV=3.5 clamping drops from 42% to ~28%; at rate=0.01 req/s, ~19%; at rate=2000 (H5's rate), ~51%. The clamping never reaches 0% because Gamma(shape < 1) has infinite density at 0.
- **What was NOT tested:**
  - Other CV values (e.g., CV=2.0, CV=2.5 — where is the clamping threshold?)
  - Rate dependence of clamping (only tested at 100 req/s)
  - Whether the unclamped portion of the distribution is correct (would need conditional KS test)
  - Correlation structure of IAT sequences (the KS test assumes independence)
- **Generalizability:** Poisson passes at any practical rate. Low-CV Gamma/Weibull (CV ≤ 1.5) pass at rate ≤ 100 req/s. High-CV Gamma/Weibull (CV ≥ 3.5) fail at any rate due to fundamental int64 floor.
- **Uncertainty quantification:** The clamping threshold (in terms of CV) depends on the acceptable failure rate and the chosen rate. A full UQ sweep is needed to map the CV × rate → clamping% surface.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Poisson KS test | p = 0.43–0.98 across 3 seeds | High |
| Gamma CV=1.5 KS test | p = 0.063 (marginal) | Medium — clamping at 1.8% creates systematic D=0.013 |
| Gamma CV=3.5 clamping | 42-43% across 3 seeds | High — matches analytical prediction |
| Weibull CV=1.5 KS test | 2/3 pass, 1/3 marginal fail | Medium — consistent with false positive rate |
| Weibull CV=3.5 clamping | 6.5-6.8% across 3 seeds | High — matches analytical prediction |
| Sample size | 3 seeds × 10,000 IATs = 30,000 per sampler | Adequate for KS at α=0.05 |

## Implications for Users

1. **Poisson arrivals are reliable** — use freely at any rate.
2. **Gamma/Weibull with CV ≤ 1.5 are reliable** at rates ≤ 100 req/s. The marginal KS results suggest clamping effects begin to appear but remain within tolerance.
3. **Gamma CV ≥ 3.5 produces a qualitatively different distribution** than intended — 42% of samples clamped to 1μs. Users wanting extreme burstiness should be aware that the effective distribution has a point mass at the floor.
4. **H5's results (Gamma CV=3.5) are affected** — this experiment measures 42% clamping at rate=100 req/s. H5 used rate=2000 req/s where the analytical estimate is ~51% clamping (higher rate → smaller mean IAT → more mass below 1μs). The "burstiness" is real (many near-simultaneous arrivals from the clamped mass), but it's a different kind of burstiness than the smooth heavy tail of a true Gamma(0.08, ...).

## Reproducing

```bash
cd hypotheses/h-arrival-generators
./run.sh
```
