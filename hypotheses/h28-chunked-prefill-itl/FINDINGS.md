# H28: Chunked Prefill ITL Impact — FINDINGS

**Status:** REFUTED
**Resolution**: Refuted — wrong mental model
**Family**: Performance-regime
**VV&UQ**: Sensitivity analysis
**Type**: Statistical/Dominance
**Date**: 2026-02-25
**Rounds**: 1

**Hypothesis:** Enabling chunked prefill (`--long-prefill-token-threshold=512`) improves mean ITL by >15% for concurrent decode requests when large-input (2048-token) prefills are present, at the cost of >20% higher TTFT for those large-input requests compared to disabled chunking (threshold=0).

**Refuted if:** Mean ITL improvement for concurrent decode requests is <10% or TTFT increase for large-input requests is <10% when comparing threshold=512 vs threshold=0 under mixed prefill-decode workload with 4+ concurrent requests.

## Experiment Design

### Workload
- **Mixed workload**: 2 client types
  - Long-input client: 2048 constant input tokens, 256 constant output tokens (50% of rate)
  - Short-input client: 128 constant input tokens, 256 constant output tokens (50% of rate)
- **Rate**: 120 req/s aggregate across 4 instances (~79% utilization)
- **Requests**: 200 per run, 3 seeds (42, 123, 456)

### Configurations
| Parameter | Config A | Config B |
|-----------|----------|----------|
| `--long-prefill-token-threshold` | 0 (disabled) | 512 (enabled) |
| All other parameters | Identical | Identical |

### Mechanism
- **Config A (threshold=0):** A 2048-token request prefills in one step. Step time = beta0 + beta1 * 2048 = 6910 + 17.67 * 2048 = ~43,100 us. During this ~43ms step, all other requests on the same instance wait, inflating their ITL.
- **Config B (threshold=512):** A 2048-token request prefills over ~4 steps of 512 tokens each. Step time per chunk = beta0 + beta1 * 512 = 6910 + 17.67 * 512 = ~15,957 us. Between chunks, decode requests can be scheduled, reducing their ITL.

### Controls
- Routing policy: least-loaded (distributes evenly, no confound from prefix affinity)
- Scheduler: FCFS (no priority confound)
- Admission: always-admit (no rejection confound)
- KV blocks: defaults (abundant, no KV pressure)
- Constant token distributions: eliminates variance from token count randomness

## Results

### Conservation (INV-1)

| Config | Seed | Status | Details |
|--------|------|--------|---------|
| A (threshold=0) | 42 | PASS | injected=200, completed=200, queued=0, running=0 |
| A (threshold=0) | 123 | PASS | injected=200, completed=200, queued=0, running=0 |
| A (threshold=0) | 456 | PASS | injected=200, completed=200, queued=0, running=0 |
| B (threshold=512) | 42 | PASS | injected=200, completed=200, queued=0, running=0 |
| B (threshold=512) | 123 | PASS | injected=200, completed=200, queued=0, running=0 |
| B (threshold=512) | 456 | PASS | injected=200, completed=200, queued=0, running=0 |

All 6 runs pass INV-1 conservation. All 200 requests complete in every run.

### Primary Metrics (Averaged over 3 seeds)

#### Long-Input Requests (2048 tokens)

| Metric | Config A | Config B | Change |
|--------|----------|----------|--------|
| TTFT Mean (ms) | 96.42 | 120.26 | +24.7% |
| TTFT P50 (ms) | 97.80 | 112.77 | +15.3% |
| TTFT P99 (ms) | 168.32 | 209.04 | +24.2% |
| ITL Mean (ms) | 10.68 | 10.67 | -0.1% |
| E2E Mean (ms) | 2820.00 | 2842.12 | +0.8% |

#### Short-Input Requests (128 tokens)

| Metric | Config A | Config B | Change |
|--------|----------|----------|--------|
| ITL Mean (ms) | 10.63 | 10.68 | +0.5% |
| ITL P50 (ms) | 10.43 | 10.52 | +0.9% |
| ITL P99 (ms) | 13.04 | 13.09 | +0.4% |
| TTFT Mean (ms) | 38.68 | 33.53 | -13.3% |
| TTFT P99 (ms) | 46.83 | 33.13 | -29.2% |
| E2E Mean (ms) | 2748.28 | 2756.51 | +0.3% |

### Per-Seed Consistency

| Seed | ITL Improvement (short) | TTFT Increase (long) |
|------|------------------------|---------------------|
| 42 | -0.5% | +24.8% |
| 123 | -0.4% | +23.2% |
| 456 | -0.6% | +26.2% |

All three seeds show consistent results: negligible ITL change (-0.4% to -0.6%) and significant TTFT increase (+23.2% to +26.2%).

#### Short-Input TTFT (unexpected finding)
| Seed | Config A Mean (ms) | Config B Mean (ms) | Change |
|------|-------------------|-------------------|--------|
| 42   | 39.21             | 33.89             | -13.6% |
| 123  | 38.45             | 33.42             | -13.1% |
| 456  | 38.38             | 33.28             | -13.3% |

All three seeds show consistent short-input TTFT improvement, confirming the convoy-effect breaking mechanism described in H27.

## Verdict

**REFUTED.** ITL improvement for short-input (concurrent decode) requests is -0.5% — not only below the 10% refutation threshold but actually slightly negative (chunking marginally worsens ITL). The TTFT cost for long-input requests (+24.7%) exceeds the 20% threshold, confirming that chunking does increase TTFT, but the hypothesized ITL benefit is completely absent.

The hypothesis predicted a tradeoff: ITL improvement at the cost of TTFT. Only the cost side materialized. The benefit side (ITL improvement) is zero.

## Mechanism Analysis

The hypothesis was based on a flawed model of how ITL interacts with prefill step time in BLIS's discrete-event simulation.

### Why ITL Does Not Improve

The predicted mechanism was: long prefill steps (~43ms) block decode token generation, inflating ITL. Chunking into ~16ms steps would let decode requests interleave more frequently, reducing ITL.

This prediction fails because **BLIS's batch formation processes all running requests in each step simultaneously**. In the VLLMBatchFormation model:

1. **Decode tokens advance per-step, not per-wall-clock-time.** Each step advances all running decode requests by one token. Whether the step takes 43ms or 16ms, every decode request in the batch gets exactly one decode token per step.

2. **ITL = step time, not step count.** ITL is measured as the wall-clock time between successive token emissions. With threshold=0, a batch containing one 2048-token prefill and several decode requests produces a ~43ms step — each decode request's ITL for that step is ~43ms. With threshold=512, the step is ~16ms — ITL is ~16ms for that step. However, **the number of such "inflated" steps is tiny** relative to the total decode steps (256 output tokens per request). A request needing 256 decode steps will encounter at most 1-2 steps with a long-input prefill co-batched. The mean ITL is dominated by the ~255 "normal" decode-only steps (~8.7ms each).

3. **At 79% utilization with 4 instances, batch sizes are small.** With ~30 req/s per instance and step times of ~10-43ms, the average batch has only a few concurrent requests. The probability that a decode request is co-batched with a long-input prefill in any given step is low, limiting the ITL inflation window.

4. **Queueing dynamics dominate.** The observed ITL (~10.6ms) reflects the average step time across all batch compositions, not just the rare steps with large prefills. Chunking changes the step time distribution (fewer ~43ms outliers, more ~16ms steps) but the mean step time across the full simulation is slightly higher for Config B because chunking adds 3 extra beta0 overheads per long request (4 steps x 6910 us vs 1 step x 6910 us = +20.7ms per long request). This extra overhead is the probable cause of the observed -0.5% ITL worsening — it marginally increases average step time across the simulation.

### Why TTFT Increases

TTFT for long-input requests increases by ~24.7% because chunking splits a single-step prefill into ~4 steps:
- Config A: One step of ~43ms completes the prefill. TTFT = queueing delay + ~43ms.
- Config B: Four steps of ~16ms each, with other requests potentially interleaved between chunks. TTFT = queueing delay + 4 * ~16ms + inter-chunk scheduling delays = queueing delay + ~64ms + overhead.

The TTFT cost is structural: chunking trades one long step for multiple shorter steps, but the total prefill compute increases (4 * beta0 overhead vs 1 * beta0 overhead) and inter-chunk scheduling adds latency.

The observed +24.7% TTFT increase is less than the raw ~48% prefill-time increase because TTFT includes queueing delay (alpha0 + alpha1 x inputLen = 8.8ms for 2048-token requests), which is identical in both configs and dilutes the percentage difference. The blended TTFT = queueing + prefill, so the percentage increase is (64ms - 43ms) / (8.8ms + 43ms) = ~40%, further diluted by scheduling dynamics.

### Short-Input TTFT Improvement

An unexpected finding: short-input requests show a TTFT improvement of -13.3% (mean) and -29.2% (P99) with chunking. This occurs because chunking breaks the "convoy effect" — without chunking, short-input requests arriving during a ~43ms long-prefill step must wait for the entire step to complete before being scheduled. With chunking, the ~16ms step windows allow short-input requests to be scheduled sooner.

### Summary

Chunked prefill in BLIS produces a one-sided tradeoff: it costs long-input requests ~25% higher TTFT while providing zero ITL benefit to decode requests. The benefit appears only in TTFT for short-input requests (-13.3%), which was not the hypothesized mechanism. The core prediction — that step time reduction improves ITL — is incorrect because ITL is dominated by the decode-only steps that comprise >99% of each request's lifetime.

## Cross-Experiment References

- H27 (chunked prefill TTFT): Companion experiment confirming that chunked prefill benefits TTFT (52% P99 improvement for short requests). H28's unexpected -13.3% short-request TTFT finding is consistent with H27 but smaller due to threshold=512 (vs H27's 256) and different workload parameters (128-token vs 64-token short requests).
- H11 (token budget): Related batch formation investigation. H11 found ITL p99 increases 5.8x with larger maxScheduledTokens while ITL mean is unchanged — the same "decode steps dominate mean ITL" insight that drives H28's refutation.
- H-Phase-Structure (linearity): Chunked prefill creates piecewise-linear TTFT behavior (TTFT depends on ceil(input/threshold) step boundaries), modifying the perfect linearity found by H-Phase-Structure at threshold=0.

## Issues Filed

None. The results reflect correct simulator behavior — chunked prefill is working as designed. The hypothesis was based on an incorrect mental model of ITL dynamics.

**Devil's Advocate (RCV-5)**: "The experiment uses only 200 requests at 79% utilization with 4 instances. At higher utilization (>95%) or with fewer instances where batching is more intense, prefill-decode co-batching frequency increases dramatically. With shorter output tokens (e.g., 16 instead of 256), the few inflated prefill steps would represent ~6% of decode lifetime rather than ~0.4%, potentially crossing the 10% ITL improvement threshold."

**Scope and Limitations (RCV-6)**:
- Operating point: 79% utilization, 4 instances, least-loaded routing, FCFS scheduler
- Only one threshold value tested (512). Threshold 256 would create more chunks.
- Only one output token count (256). Shorter output = higher relative weight of prefill-inflated steps.
- Only one load level. Higher load = more co-batching = potentially measurable ITL effect.
- 200 requests, ~100 per client type — adequate for the observed effect sizes given large margins.
- Constant token distributions eliminate variance effects.

**Standards Audit**:
- [ ] No violations found
- [ ] No new rules needed
- [ ] No new invariants needed
- [x] INV-1 conservation confirmed (all 6 runs)
- [x] INV-8 work-conserving confirmed under chunked prefill

**Evidence Quality**:
| Metric | Value | Confidence |
|--------|-------|-----------|
| ITL improvement (short) | -0.5% (range: -0.4% to -0.6%) | HIGH — decisively below 10% threshold |
| TTFT increase (long) | +24.7% (range: +23.2% to +26.2%) | HIGH — tight cross-seed spread |
| Short TTFT improvement | -13.3% mean (unexpected) | MODERATE — not the primary metric, per-seed data not shown in main table |
| Mechanism confidence | Code-verified at batch_formation.go:84,116 | HIGH |

**Findings Classification**:
| Finding | Type | Action |
|---------|------|--------|
| ITL unaffected by chunked prefill (-0.5%) | Refutation (wrong mental model) | Document: decode steps dominate ITL |
| Short-request TTFT improves -13.3% | Surprise | Cross-reference H27 |
| Long-request TTFT degrades +24.7% | Confirmation (expected cost) | Document tradeoff |
| Decode steps dominate mean ITL | Insight | Promote to MEMORY.md DES behavior note |

**Implications for Users**:
- Do NOT enable chunked prefill expecting ITL improvement — the benefit is TTFT for short requests, not ITL
- Chunked prefill's actual benefit: reduces scheduling latency (TTFT) for short requests co-batched with long prefills (see H27: 52% P99 improvement)
- The cost is ~25% higher TTFT for long-input requests from per-chunk beta0 overhead
- ITL is dominated by decode-phase step time (~255 decode steps vs 1-2 prefill-co-batched steps per request)

**Reproducing**:
```
cd hypotheses/h28-chunked-prefill-itl
./run.sh
```
