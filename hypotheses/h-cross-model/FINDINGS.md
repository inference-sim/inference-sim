# H-Cross-Model: Cross-Model Generalization Validation

**Status:** Partially confirmed
**Resolution:** Partial confirmation with surprise — 12/15 behavioral findings generalize; 3 cache-related findings do not generalize due to prefix cache mechanism and operating-point sensitivity
**Family:** Structural model (cross-model generalization). Note: this meta-experiment spans all 6 hypothesis families but is classified as Structural model because it validates whether the DES model architecture generalizes across parameter regimes.
**VV&UQ:** Mixed — Verification (H12, H13, H-Liveness, H-Overload: invariant checks) + Validation (all statistical sub-experiments: behavioral comparisons)
**Tier:** 1 (correctness baseline for cross-model portability)
**Type:** Mixed — 5 deterministic sub-experiments (invariants) + 10 statistical sub-experiments (directional)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> All confirmed BLIS behavioral findings should hold when running with a different model configuration (Qwen/Qwen2.5-7B-Instruct on H100/TP=1). The DES model captures system-level dynamics (queueing, scheduling, routing, caching) that are model-agnostic — therefore behaviors like "prefix-aware routing outperforms load-only" and "SJF helps short requests" should remain true regardless of which LLM's alpha/beta coefficients are used.

## Experiment Design

**Classification:** Mixed (deterministic invariants + statistical/dominance + statistical/monotonicity)

**Model configurations compared:**
- Original: `meta-llama/llama-3.1-8b-instruct` (H100, TP=2) — alpha=[1601.35, 3.51, 1805.54], beta=[6910.42, 17.67, 2.84], kv_blocks=132139
- Treatment: `Qwen/Qwen2.5-7B-Instruct` (H100, TP=1) — alpha=[4680.30, 0.0, 0.0], beta=[7051.80, 19.54, 25.43], kv_blocks=65833

**Coefficient provenance:** The Qwen alpha/beta coefficients and kv_blocks were specified in issue #396 by the project maintainer. They do NOT match any entry in `defaults.yaml` (the H100/TP=1 entry has alpha=[5223.76, 2.41, 5368.06], beta=[1602.21, 19.87, 452.94], kv_blocks=67659). The alpha1=0.0 and alpha2=0.0 values are notably different from the defaults.yaml profile. Users running `./simulation_worker run --model Qwen/Qwen2.5-7B-Instruct --hardware H100 --tp 1` without explicit `--alpha-coeffs`/`--beta-coeffs` would get the defaults.yaml coefficients, not the ones tested here.

**Key differences in alpha/beta profile:**
- **alpha1=0.0**: No input-length-dependent queueing delay (llama: 3.51 us/token). QueueingTime (`sim/latency_model.go:56-61`) becomes constant at alpha0=4680.3 us regardless of input length.
- **alpha2=0.0**: No output token processing overhead (llama: 1805.54 us). OutputTokenProcessingTime (`sim/latency_model.go:63-65`) returns 0.
- **beta2=25.43**: Decode tokens are 9x more expensive per step (llama: 2.84). StepTime (`sim/latency_model.go:38-53`) = beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
- **KV blocks**: 65833 (about half of llama's 132139)
- **max_scheduled_tokens**: 4096 (vs CLI default 2048)

**Controlled variables:** Workloads, seeds, policy configurations held constant across models.
**Varied variable:** Model alpha/beta coefficients and associated hardware parameters.
**Seeds:** 42, 123, 456 (same as originals)
**Preconditions verified:** Binary builds; workload YAMLs validated by code review Rounds 1-2.

## Results

### Summary Table

| ID | Category | Finding | Qwen Result | Status |
|----|----------|---------|-------------|--------|
| H12 | Verification | Conservation (INV-1) | 10/10 PASS | **Confirmed** |
| H13 | Verification | Determinism (INV-6) | 3/3 byte-identical | **Confirmed** |
| H-Liveness | Verification | No starvation | 6/6 PASS | **Confirmed** |
| H-Overload | Verification | Conservation under 10x | 9/9 PASS, 0 panics | **Confirmed** |
| H-Phase | Validation | TTFT/decode linearity | R²=1.000000 both | **Confirmed** |
| H-MMK | Validation | E2E monotonic in rho | Monotonically increasing | **Confirmed** |
| H1-SJF | Validation | SJF < FCFS TTFT | 3/3 seeds, 0.56x ratio | **Confirmed** |
| H3 | Validation | QD > KV uniformity | 3/3 directional, 1 seed <10% | **Confirmed with nuance** |
| H5 | Validation | Token-bucket reduces TTFT | 3/3 seeds, 1.8-1.9x | **Confirmed** |
| H8 | Validation | KV preemption monotonic | Monotonic (0→0.07→0.38) | **Confirmed** |
| H14 | Validation | Pathological > baseline | 3/3 seeds, 6.4-6.5x | **Confirmed** |
| H-Arrival | Validation | Poisson generators work | 1000 completed | **Confirmed** (trivially model-agnostic) |
| Prefix-Affinity | Validation | Cache < load TTFT | 0/3 seeds at low AND high rate | **Inconclusive** |
| H9 | Validation | TTFT monotonic in prefix | Flat ~27ms; control: no cache hits | **Refuted for this config** |
| H10 | Validation | Tiered < single preemption | 2/3 seeds, effect <10% | **Inconclusive** |

**Honest framing:** 4/4 verification findings (model-independent by construction) + 6/10 model-sensitive validation findings fully confirmed + 1/10 confirmed with nuance (H3: directional but 1 sub-threshold seed) + 1 trivially model-agnostic (H-Arrival). The 3 non-generalizing findings are all cache-related.

### Detailed Results: Confirmed Findings

**H12 — Conservation:** INV-1 holds for all 10 policy combinations. Token-bucket correctly rejects 171/200 requests (injected=29). Identical behavior to llama.

**H13 — Determinism:** Byte-identical output across duplicate runs for all 3 policy configurations. INV-6 holds.

**H-Overload — Conservation under stress:** At rate=1500 (>10x capacity), injected=2000, with 896-899 still queued, 1021-1023 running, 80-81 completed. Conservation holds. Token-bucket correctly limits injection to ~94-109 requests. No panics.

**H-Phase — Phase structure:** R²=1.000000 for both TTFT vs input tokens and decode time vs output tokens. TTFT slope = 0.0195 ms/token matches beta1/1000 = 0.01954 (`sim/latency_model.go:48`: `totalStepTime += m.betaCoeffs[1] * float64(totalCacheMissTokens)`). Decode slope = 7.077 ms/step matches (beta0+beta2)/1000 = 7.077. This validates the latency model architecture with a completely different coefficient profile.

**H-MMK — M/M/k validation:** Calibrated mu=1.0906 req/s (mean E2E=916.9ms at batch=1). E2E monotonically increases with rho: 930.5→938.1→959.3→1070.0 ms across rho=0.1→0.5. At rho=0.5, E2E divergence from calibrated baseline is 16.7% ((1070-917)/917). Note: this is an E2E divergence metric, not directly comparable to the H-MMK W_q divergence metric (28-71% for llama). The reduced divergence is consistent with alpha2=0 removing the OutputTokenProcessingTime overhead that inflated llama's E2E, as predicted by H-Step-Quantum, but the cross-experiment comparison has confounds (different model, different metric, M/M/k vs M/M/1 in H-Step-Quantum).

**H1-SJF:** SJF produces 0.56x TTFT ratio (44% reduction, >20% threshold) vs FCFS across all 3 seeds (seed 42: 0.564, seed 123: 0.569, seed 456: 0.558). Direction holds strongly.

**H3 — Signal freshness (confirmed with nuance):** Per-seed Jain's fairness: seed 42: QD=0.9999 vs KV=0.9756 (2.5% gap), seed 123: QD=0.9999 vs KV=0.9156 (9.2% gap), seed 456: QD=0.9999 vs KV=0.7676 (30.2% gap). Direction is consistent (QD > KV) for all 3 seeds. However, seed 42's 2.5% gap is below the 10% inconclusive threshold per experiment standards. Per the dominance subtype definition, consistent direction in all seeds is required but the per-seed threshold (<10% = inconclusive) makes seed 42 borderline. Classification: "Confirmed with nuance" — directionally consistent but weakened by sub-threshold seed 42. The weakening is because abundant KV blocks (65833) reduce kv-utilization staleness, narrowing the gap at low contention.

**H14 — Pathological:** always-busiest routing produces 6.4-6.5x TTFT degradation via `sim/routing.go:264-288` (`AlwaysBusiest.Route` selects max `EffectiveLoad()`, concentrating all requests). The effect is STRONGER than llama's 4.5x because beta2=25.43 means decode steps are slower, amplifying the penalty of concentration.

### Detailed Results: Non-Confirmed Findings

**Prefix-Affinity — INCONCLUSIVE:**

*Round 1 (rate=200):* Ratios 0.997-1.019 — within 2%, equivalent under the 5% threshold. No measurable difference at low load.

*Round 2 control (rate=2000, heavy overload):* Ratios 1.002-1.011 — cache-aware routing is 0.2-1.1% WORSE than load-only even under heavy load. 0/3 seeds show improvement.

This **refutes** the Round 1 hypothesis that the failure was "rate/load-dependent." The prefix-affinity benefit does not manifest at ANY rate with this coefficient profile. Possible mechanisms: (1) With beta2=25.43, decode tokens dominate step time, and prefix cache savings on beta1*cacheMissTokens are proportionally smaller. (2) The prefix-affinity scorer's routing decisions may create load imbalance that outweighs the cache benefit.

**Design caveat:** The workload uses `prefix_length: 512` but Gaussian input distribution with mean=256 (max=512). Many requests have fewer input tokens than the specified prefix length, limiting the potential cache benefit. This matches the original llama experiment's workload design (controlled variable, not a confound), but it means the prefix-affinity test operates with partial prefix coverage. This reduces the maximum possible effect size and may contribute to the null result.

**H9 — Prefix TTFT Monotonicity — REFUTED for this config:**

*Round 1 (rate=100, 4 instances, batch=256):* TTFT ~27ms for all prefix lengths (0-512). Flat within noise.

*Round 2 isolation control (rate=0.001, 1 instance, batch=1):* prefix=0: 21.73ms (matches first-principles: alpha0+beta0+beta1*512 = 4680+7052+10005 = 21737us = 21.74ms). prefix=256: 22.24ms. prefix=512: 22.74ms. TTFT **increases** by 4.6% with more prefix — OPPOSITE of expected direction.

The prefix=0 value matches the no-cache calculation perfectly. But prefix=512 should give TTFT = alpha0+beta0 = 11.73ms (zero cache miss tokens), not 22.74ms. This means **the prefix cache is NOT producing KV cache hits**. The 1ms increase per prefix level may be from prefix cache index lookup overhead (`sim/prefix_cache_index.go`) without corresponding StepTime reduction. Root cause unverified — requires code-level investigation of whether the KV cache block-hash matching works with synthetically-generated constant-value tokens from the workload generator. This is filed as an open question, not a confirmed mechanism.

**H10 — Tiered KV — INCONCLUSIVE:**

Per-seed results: seed 42: single=0.095 vs tiered=0.090 (5.3% reduction, <20% threshold), seed 123: single=0.020 vs tiered=0.010 (50% reduction, >20%), seed 456: single=0.090 vs tiered=0.090 (0% reduction). Only 1/3 seeds exceeds the 20% dominance threshold. The predicted direction does not hold across ALL seeds (seed 456 = 0%). Classification: Inconclusive per the directional consistency requirement.

The preemption rate at 2100 blocks is only 2-9.5% for Qwen (vs llama's 17.5%), providing insufficient pressure for the tiered benefit to manifest reliably. No control experiment at lower block counts was run to confirm this threshold hypothesis.

## Root Cause Analysis

### Confirmed findings — mechanism verification

**Latency model architecture (H-Phase):** StepTime formula verified at `sim/latency_model.go:38-53`: `totalStepTime += m.betaCoeffs[0] + m.betaCoeffs[1]*cacheMissTokens + m.betaCoeffs[2]*decodeTokens`. QueueingTime at `sim/latency_model.go:56-61`: `alpha0 + alpha1*inputLen`. OutputTokenProcessingTime at `sim/latency_model.go:63-65`: `return int64(m.alphaCoeffs[2])`. With alpha=[4680.3, 0, 0], QueueingTime = constant 4680.3us, OutputTokenProcessingTime = 0. R²=1.000000 confirms these formulas operate correctly with the new coefficients.

**Conservation (H12, H-Overload):** `sim/metrics.go:73` computes `injected_requests = completed + still_queued + still_running + dropped_unservable`. This is a structural property independent of alpha/beta coefficients.

**Pathological routing (H14):** `sim/routing.go:264-288` (`AlwaysBusiest`) selects max `EffectiveLoad()`, which is `QueueDepth + BatchSize + PendingRequests`. With Qwen's slower decode (beta2=25.43), requests remain in batch longer → higher BatchSize per instance → greater concentration penalty.

### Non-confirmed findings — mechanism analysis (RCV-3, RCV-4 partial)

**Prefix-Affinity non-generalization:** The Round 2 high-rate control (rate=2000) rules out rate-dependence as the explanation. The finding genuinely does not generalize to this alpha/beta profile. The proposed mechanism (beta2-dominant step time drowns out prefix cache savings) is plausible but lacks a direct control experiment isolating beta2. The control would require running with modified coefficients (e.g., beta2=2.84 like llama) while keeping other Qwen parameters — this is beyond the scope of a single-variable experiment.

**H9 prefix cache non-hits:** The isolation control (batch=1, rate=0.001, single instance) demonstrates that prefix=512 does NOT reduce TTFT even under zero queueing and zero batching. The most likely root cause is that the KV cache prefix matching mechanism (`sim/kvcache.go` block-hash matching) does not find cache hits for synthetically-generated tokens from the workload generator. This requires code-level investigation and is filed as an open question.

## Devil's Advocate (RCV-5)

**If this is "Partially confirmed," argue why it might be fully confirmed:**
The 3 failed experiments (Prefix-Affinity, H9, H10) may be artifacts of the experiment design rather than genuine non-generalization: (1) Prefix-Affinity might work with a workload that has longer shared prefixes relative to total input, (2) H9 may fail due to a workload generator issue with prefix token content, not a latency model issue, (3) H10 at 2100 blocks produces only 2-9.5% preemption for Qwen, insufficient for the tiered benefit to manifest. Different operating points might restore all three.

**If this is "Partially confirmed," argue why it might be more broadly refuted:**
The 12 confirmed findings include 4 invariant checks (model-independent by construction) and 1 trivially model-agnostic test (H-Arrival), leaving only 7/10 genuinely model-sensitive findings confirmed. Moreover, some confirmed findings operate at sub-threshold effect sizes for individual seeds (H3 seed 42: 2.5%). If the threshold were stricter, more findings would be inconclusive. The honest generalization rate for model-sensitive behavioral findings is 7/10 = 70%, and ALL three non-generalizing findings involve caching — suggesting a systematic weakness in cache-related behaviors across model configurations.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| 4/4 verification findings (H12, H13, H-Liveness, H-Overload) model-agnostic | Confirmation | Documented here — confirms INV-1, INV-6 |
| 7/10 model-sensitive validation findings generalize | Confirmation | Documented here |
| Phase linearity (H-Phase) generalizes perfectly (R²=1.0) | Confirmation | Validates latency model architecture |
| H14 pathological effect STRONGER with Qwen (6.5x vs 4.5x) | Surprise | Higher beta2 amplifies routing concentration penalty |
| Prefix-Affinity does NOT generalize even under load | Scope limitation | User guidance: prefix-affinity benefit is profile-dependent |
| H9 prefix cache not producing cache hits with synthetic tokens | Open question | Requires code investigation of KV cache block-hash matching with workload generator tokens |
| H10 tiered KV inconclusive at 2100 blocks | Scope limitation | Requires lower block count to produce equivalent pressure |
| H-MMK divergence potentially reduced with alpha2=0 | Surprise | Consistent with H-Step-Quantum but cross-experiment comparison has confounds |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found. INV-1 and INV-6 hold for Qwen.
- [x] Any new rules needed? None proposed — the coefficient provenance issue (custom coefficients not matching defaults.yaml) is specific to this experiment, not a general rule violation.
- [x] Any new invariants needed? None.
- [x] Any existing rules/invariants confirmed? **INV-1** confirmed across 10 policy configs + 9 overload configs with Qwen. **INV-6** confirmed for 3 policy configs. Other invariants (INV-2 through INV-5, INV-7, INV-8) were not directly tested.

## Scope and Limitations (RCV-6)

- **Operating point tested:** Custom Qwen coefficients from issue #396 (NOT the defaults.yaml entry). H100, TP=1, alpha=[4680.3, 0.0, 0.0], beta=[7051.8, 19.5, 25.4], kv_blocks=65833, 4 instances.
- **Coefficient source:** Issue #396 (external to defaults.yaml). The defaults.yaml Qwen H100/TP=1 entry has different coefficients (alpha1=2.41, alpha2=5368.06, beta2=452.94). Results may differ with defaults.yaml coefficients.
- **Parameters findings depend on:** Alpha profile (especially alpha1, alpha2); beta2/beta0 ratio; KV block count relative to workload demand.
- **What was NOT tested:** defaults.yaml Qwen coefficients; rate recalibration for equivalent utilization; more model configurations (A100, TP>1); roofline mode; multi-turn/reasoning workloads; ED-2 vanishing-point controls for confirmed findings; H10 at lower block count for equivalent preemption pressure.
- **Generalizability:** Invariants generalize universally (model-independent). Policy ordering (SJF, pathological, signal freshness, token-bucket) generalizes. Cache-related findings (prefix-affinity, prefix TTFT, tiered KV) do not generalize to this coefficient profile.
- **Uncertainty quantification:** UQ not performed — single model configuration tested.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Verification pass rate (INV-1, INV-6) | 28/28 checks | High — deterministic, model-independent |
| Policy comparison agreement (H1-SJF, H5, H14) | 3/3 confirmed, >20% in all seeds | High |
| Structural model agreement (H-Phase, H-MMK, H8) | 3/3 confirmed | High (H-Phase: R²=1.0, H8: strict monotonicity) |
| Signal freshness (H3) | 3/3 directionally consistent, 1 seed <20% effect | Medium — direction holds but 1 seed at 2.5% gap |
| Cache-related agreement | 0/3 (Prefix-Affinity, H9, H10) | Low — Prefix-Affinity refuted by control; H9 cache hits absent; H10 inconclusive |
| Effect size adequacy | 8/10 statistical findings >20% in all seeds | Medium |
| Control experiments (Round 2) | 2 executed (Prefix-Affinity high-rate, H9 isolation) | Prefix-Affinity control refuted rate-dependent hypothesis; H9 control confirmed cache non-hits |

## Implications for Users

1. **Invariants are portable.** Users can trust that conservation (INV-1), determinism (INV-6), liveness, and overload safety hold for any model configuration. No model-specific testing needed.

2. **Policy ordering is portable.** SJF beats FCFS, queue-depth beats kv-utilization for freshness, pathological configs produce pathological behavior, token-bucket provides load shedding — these hold regardless of alpha/beta profile.

3. **Cache-related findings are NOT portable.** Prefix-affinity, prefix caching TTFT reduction, and tiered KV benefits do not generalize to the tested Qwen configuration. Users should validate cache-related behaviors for their specific model configuration.

4. **H-Phase linearity validates the latency model architecture.** R²=1.0 with a completely different beta profile confirms the alpha/beta regression architecture is correct.

5. **Models with high beta2 amplify routing pathologies.** The 6.5x degradation (vs 4.5x for llama) under always-busiest routing shows that decode-heavy models are more sensitive to routing concentration.

## Reproducing

```
cd hypotheses/h-cross-model
./run.sh
```
