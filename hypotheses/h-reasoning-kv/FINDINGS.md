# H-Reasoning-KV: Reasoning Context Accumulation Under KV Pressure

**Status:** Refuted (primary), Confirmed (supporting)
**Resolution:** Refuted — wrong mental model. Demand heterogeneity does NOT shift the preemption cliff because the cliff is driven by peak *concurrent* block demand, not per-request peak demand. The mean demand per request (72 blocks) is identical for both workloads, so concurrency-driven pressure is the same. However, context accumulation produces 63.8% prefix cache hits — a major positive finding for capacity planning.
**Family:** Performance-regime (scaling laws)
**VV&UQ:** Validation (cliff comparison) + Verification (#386, INV-1)
**Tier:** New (not in original research.md tiers)
**Type:** Statistical (Monotonicity) + Deterministic (invariant checks)
**Date:** 2026-02-23
**Rounds:** 1

## Hypothesis

> Under constrained KV capacity, multi-turn reasoning workloads with context accumulation trigger the preemption cliff at a block count proportional to their peak per-request demand (120 blocks for round 4), while standard workloads with uniform per-request demand (72 blocks) trigger it at a proportionally lower block count. The cliff shift ratio should be approximately 1.1-1.3x.

## Experiment Design

**Classification:** Statistical/Monotonicity (preemption rate vs block count)

**Configurations compared:**
- Workload A (reasoning): 2 sessions/s, 5 rounds with context accumulation, constant input=128, output=256
- Workload B (standard-throughput-matched): 10 req/s, constant input=896, output=256. Matches ~10 req/s effective rate of reasoning.
- Workload C (standard-session-matched): 2 req/s, constant input=896, output=256. Matches session arrival rate.

All workloads: `--model meta-llama/llama-3.1-8b-instruct --num-instances 1 --max-num-running-reqs 32 --max-num-scheduled-tokens 2048 --block-size-in-tokens 16 --horizon 100000000`

**KV sweep:** 5000, 3000, 2000, 1500, 1200, 1000, 800, 600, 400, 100 blocks
**Seeds:** 42, 123, 456
**Total runs:** 90 (10 blocks × 3 seeds × 3 workloads)
**Preconditions verified:** Context accumulation produces expected token pattern [128, 512, 896, 1280, 1664] — PASS. 123 sessions detected.

**Config diff (ED-6):** Novel experiment — no prior reference. Reasoning workload YAML is new. Standard workloads use constant distributions for isolation.

## Results

### 1. Preemption Cliff Detection (primary metric)

5% preemption rate threshold crossing (3-seed average):

| Workload | Cliff (blocks) | Per-seed cliffs |
|----------|----------------|-----------------|
| Reasoning | 2651 | [2541, 2805, 2606] |
| Std (throughput-matched) | 2482 | [2554, 2932, 1961] |
| Std (session-matched) | 710 | [549, 780, 800] |

**Cliff shift ratio (reasoning / std-throughput):** 1.09x
**Per-seed ratios:** [0.99, 0.96, 1.33]
**Detection criterion (pre-committed):** "Cliff shift detected" requires >=20% difference in ALL 3 seeds. Result: NOT DETECTED (two seeds show <10%).

**Statistical power caveat:** With 3 seeds and per-seed ratio SD ≈ 0.21, the 95% CI for the mean ratio is [0.74, 1.44] — the experiment cannot distinguish "no effect" from "moderate effect." The "not detected" conclusion rests primarily on the analytical argument (identical mean demand → identical cliff), not on experimental power alone. A 10-seed replication with finer block granularity (every 100 blocks near the cliff) would be needed for definitive statistical confirmation.

**Cliff characterization note:** The 5% preemption rate threshold is a heuristic. The data shows a gradual onset region (~1500-3000 blocks for reasoning/throughput-matched) rather than a sharp cliff. Compare H8's finer granularity (100-block steps) at a different operating point (4 instances, rate=2000).

### 2. Per-Round TTFT (blocks=5000, reasoning workload)

| Round | Input tokens | TTFT mean (ms) | Growth from Round 0 |
|-------|-------------|----------------|---------------------|
| 0 | 128 | 16.63 | — |
| 1 | 512 | 17.77 | +6.8% |
| 2 | 896 | 19.36 | +16.4% |
| 3 | 1280 | 21.12 | +27.0% |
| 4 | 1664 | 21.89 | +31.6% |

**Spearman rho(round, TTFT):** +1.000 for all 3 seeds (perfect monotonicity).

**Predicted vs actual:** Predicted 3.6x growth (from alpha0 + alpha1 × inputLen). Actual: 1.32x (31.6%). The 63.8% cache hit rate means round-4 requests with 1664 input tokens have ~1060 tokens cached, so effective new tokens ≈ 604 — explaining the attenuated TTFT growth.

### 3. Prefix Cache Hit Rate (blocks=5000)

| Workload | Cache Hit Rate |
|----------|---------------|
| Reasoning | **0.6384** |
| Std (throughput-matched) | 0.0000 |
| Std (session-matched) | 0.0000 |

Consistent across all 3 seeds (0.6379, 0.6369, 0.6405).

### 4. #386 Verification (blocks=100)

| Workload | Avg dropped | Per-seed |
|----------|------------|----------|
| Reasoning | **76.0** | [75, 74, 79] |
| Std (throughput-matched) | 0.0 | [0, 0, 0] |
| Std (session-matched) | 0.0 | [0, 0, 0] |

Reasoning drops rounds 3-4 (1280 and 1664 tokens need 80 and 104 blocks > 100 total). Standard requests (896 tokens = 56 blocks) fit in 100 blocks. #386 fix works as designed.

### 5. Conservation Invariant (INV-1)

**90/90 checks PASS.** `injected == completed + queued + running + dropped_unservable` holds for all configurations including blocks=100 with drops.

### 6. Preemption Dynamics (observation)

The standard-throughput-matched workload exhibits extreme non-monotonic preemption behavior:

| Blocks | Reasoning preemption rate | Std-throughput preemption rate |
|--------|--------------------------|-------------------------------|
| 2000 | 0.164 | 0.284 |
| 1500 | 0.257 | **160.5** |
| 1000 | **170.7** | **217.4** |
| 400 | 88.4 | **4602.3** |

Rates >1.0 indicate cascading preemptions — each completed request was preempted multiple times on average. At blocks=400 for standard-throughput-matched, 4602 preemptions per completed request indicates severe thrashing. This is the cascading preemption phenomenon from #349.

**Monotonicity (Spearman rho for blocks → preemption rate):**
- Session-matched: rho=-0.952 (near-perfect negative monotonicity — well-behaved)
- Reasoning: rho=-0.838 (good monotonicity despite cascade onset)
- Throughput-matched: rho=-0.483 (poor — cascading preemptions create non-monotonic behavior)

## Root Cause Analysis

### Why the cliff shift is negligible (RCV-1, RCV-3)

The hypothesis assumed per-request peak demand (120 blocks for round 4) would shift the cliff. But the cliff is driven by **concurrent** block demand, not individual request demand.

At steady state with rate=2 sessions/s:
- ~17.7 concurrent requests (Little's law, see research.md)
- Mean block demand per request: (24+48+72+96+120)/5 = 72 blocks (reasoning) = 72 blocks (standard)
- Total concurrent demand: 17.7 × 72 ≈ 1274 blocks for BOTH workloads

The mean demand is identical by design (we matched total token throughput). The variance creates only a ~100-block fluctuation when multiple round-4 requests coincide, which is <10% of the cliff point (~2500 blocks). This matches the observed 1.09x ratio.

**Code path (RCV-1):** Preemption occurs in `sim/batch_formation.go` `VLLMBatchFormation.FormBatch()` when `AllocateKVBlocks` fails. The allocation path in `sim/kvcache.go:AllocateKVBlocks` (line ~130-180) checks `freeBlocks >= blocksNeeded`. The cliff occurs when concurrent requests' total block demand exceeds total capacity — this depends on concurrent count × mean demand, not max single-request demand.

### Why prefix cache hits are 63.8% (RCV-1)

Context accumulation in `sim/workload/reasoning.go:54-58` creates round N's input by prepending all prior rounds' input + output tokens. These are exact token ID copies (line 55: `inputTokens = append(append([]int{}, contextPrefix...), newInputTokens...)`).

When the KV cache processes round N, `AllocateKVBlocks` in `sim/kvcache.go` hashes each block of 16 tokens. If the prior round's blocks are still in the LRU cache, the hash lookup succeeds (cache hit). With think_time=5s between rounds and rate=2 sessions/s, only ~10 other requests arrive between rounds, churning ~720 blocks out of 5000 total (14.4%). The LRU retains most prior-round blocks.

First-principles cache hit estimate (including round-0 dilution):
- Total tokens across all 5 rounds: 128+512+896+1280+1664 = 4480
- Cacheable tokens (if all prior-round blocks survive LRU): 0+384+768+1152+1536 = 3840
- Expected token-level hit rate: 3840/4480 = 85.7%
- LRU eviction correction: ~14.4% block churn between rounds → expected rate ≈ 85.7% × (1-0.144) = 73.3%
- Observed block-level rate: 63.8%

The residual gap (73.3% predicted vs 63.8% observed) is an **open question**. Possible causes: multi-session LRU contention patterns under interleaved arrivals, block-vs-token-level metric differences, or batch formation ordering effects. The cross-seed stability (0.6369-0.6405, σ=0.002) confirms this is a systematic effect, not noise. No control experiment disabling prefix matching was run (token-hash collision is theoretically impossible, making this unnecessary).

### Why TTFT growth is 32% not 360% (RCV-2)

TTFT has three components with different caching sensitivity:

1. **QueueingTime (alpha model)** — `latency_model.go:58-59`: uses `len(req.InputTokens)` (FULL input, NOT reduced by caching). Round 0: 1601+3.51×128=2050 μs. Round 4: 1601+3.51×1664=7442 μs. **3.63x growth — caching has NO effect here.**

2. **StepTime (beta model)** — `latency_model.go:42-43`: uses `NumNewTokens` (reduced by caching via `batch_formation.go:109-110`). Round 0: 6910+17.67×128=9172 μs. Round 4 with caching (602 new tokens): 6910+17.67×602=17,543 μs. **1.91x growth — significantly attenuated from the 3.96x without caching** (6910+17.67×1664=36,323 μs).

3. **OutputTokenProcessingTime + scheduling overhead** — constant 1806 μs. **1.0x growth.**

The composite TTFT growth is an average of these three channels, weighted by their contribution. The constant alpha0 (1601 μs) and constant OutputTokenProcessingTime (1806 μs) dilute the proportional growth. The observed 1.32x is consistent with the beta-channel attenuation being the primary mechanism, diluted by the two constant-overhead channels.

## Devil's Advocate (RCV-5)

**Argue why the cliff shift might actually be real:**
The 1.33x ratio in seed 456 suggests the effect exists but is masked by noise in the other seeds. With only 3 seeds and a wide cliff detection range (2541-2805 for reasoning), the signal-to-noise ratio is too low. Running with 10 seeds or using a finer block granularity (every 50 blocks near the cliff) might reveal a consistent 10-15% shift.

**Argue why the 63.8% cache hit rate might be misleading:**
The cache hit rate is inflated by the generous KV capacity (5000 blocks). At constrained block counts (800-1000), LRU eviction between rounds would destroy cached blocks, and the cache hit rate would drop to near zero. The practical benefit of reasoning prefix caching only exists when KV is over-provisioned — exactly the regime where it matters least.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Cliff shift not detected (1.09x, below threshold) | Refuted — wrong mental model | Documented here. Capacity planning: mean demand drives cliff, not per-request peak. |
| TTFT monotonicity confirmed (rho=+1.000) | Confirmation | Documented here. Context accumulation correctly increases TTFT across rounds. |
| 63.8% prefix cache hit rate for reasoning workloads | **Surprise** | Document as user guidance. Reasoning workloads benefit significantly from prefix caching when KV is adequate. |
| #386 correctly drops oversized reasoning requests | Confirmation | Validates #386 fix under reasoning workload (first test of this path). |
| INV-1 holds across all 90 configurations | Confirmation | Extends INV-1 validation to reasoning workloads + dropped_unservable. |
| Cascading preemptions (rate >100x) at constrained KV | Known phenomenon | Reproduces #349 at new operating points. No new issue needed. |
| `sim/workload/reasoning.go` generates correct token accumulation | Confirmation | First end-to-end validation of reasoning workload generation. |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found.** R19 livelock protection verified via #386.
- [x] Any new rules needed? **No.** Existing rules cover all observed behavior.
- [x] Any new invariants needed? **No.** INV-1 updated formula (with dropped_unservable) holds.
- [x] Any existing rules/invariants confirmed? **INV-1** confirmed under reasoning workload + drops. **R19** confirmed via #386 verification.

## Scope and Limitations (RCV-6)

- **Operating point tested:** rate=2 sessions/s, 5 rounds, constant input=128/output=256, single instance, blocks 100-5000
- **Parameters findings depend on:** think_time=5s (affects cache hit rate via LRU churn), constant distributions (eliminates variance confound)
- **Known confounds:**
  - Arrival pattern differs between reasoning (bursty: 5 rounds/session with 5s gaps) and throughput-matched (uniform Poisson). This confound strengthens the refutation: bursty arrivals should create MORE demand spikes, yet no cliff shift was detected.
  - `num_requests: 500` produces ~123 sessions with session truncation (some sessions incomplete). Round distribution is not uniform: 123 round-0, 75 round-4. This does not affect cliff analysis but is relevant for per-round TTFT interpretation.
  - Session-matched comparison (rate=2 req/s) isolates *throughput* as the dominant cliff driver. Reasoning-vs-throughput-matched comparison tests demand heterogeneity *with* the arrival pattern confound.
- **What was NOT tested:**
  - Variable input/output distributions (real workloads have variance)
  - Multi-instance cluster (routing interaction with reasoning workload)
  - Tiered KV cache (CPU offload interaction)
  - Roofline latency mode
  - High utilization (rate > 10 sessions/s)
  - Mixed reasoning + non-reasoning clients
  - Different think_time values (5s is generous for LRU retention)
  - Deconfounded heterogeneity test (Poisson at 10 req/s with randomized per-request sizes from [24,48,72,96,120] blocks)
- **Generalizability:** The 63.8% cache hit rate is specific to this think_time, rate, and KV capacity (5000 blocks). Higher rates, shorter think_times, or constrained KV would increase LRU churn and reduce cache hits. The cliff non-shift finding likely generalizes: any workload with the same mean per-request block demand will have the same cliff location regardless of per-request variance.
- **Uncertainty quantification:** Cliff location has wide seed-to-seed variance (±200 blocks). Cache hit rate is stable (±0.002 across seeds). More seeds or finer block granularity needed for UQ on cliff location.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Cliff shift ratio | 1.09x | Medium — wide variance across seeds (0.96-1.33) |
| TTFT monotonicity | rho=+1.000 | High — perfect monotonicity across all seeds |
| Cache hit rate | 0.6384 | High — stable across seeds (±0.002) |
| #386 dropped count | 76 avg | High — consistent mechanism (rounds 3-4 exceed capacity) |
| Conservation | 90/90 | High — deterministic check |

## Implications for Users

1. **Capacity planning for reasoning workloads:** Provision KV blocks based on **mean** per-request demand × expected concurrency, not peak per-request demand. The quadratic growth in per-request demand does NOT create a cliff shift versus standard workloads with equivalent throughput.

2. **Prefix caching is highly effective for reasoning/multi-turn** (at adequate KV capacity): 63.8% cache hit rate at 5000 blocks with rate=2 sessions/s and 5s think time. This reduces effective prefill tokens by ~2/3, significantly improving TTFT for later rounds. Cache hit rate will decrease with higher rates, shorter think times, or constrained KV capacity.

3. **TTFT growth is modest with caching** (at this operating point): Despite 13x input growth (128→1664 tokens), TTFT grows only 32% thanks to prefix caching. Under constrained KV (where cache eviction is high), growth could approach the analytical 3.6x prediction.

4. **#386 protects against oversized reasoning requests:** Late-round requests in very constrained KV will be cleanly dropped rather than causing livelock. Monitor `dropped_unservable` in production.

## Reproducing

```
cd hypotheses/h-reasoning-kv
./run.sh
```
