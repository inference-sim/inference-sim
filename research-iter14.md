# Iteration 14: Predictive TTFT-Budget Admission Control

## Problem
Current best compound (iter 13): `pa:4,qd:3` + SLO-gated admission + SLO-class priority = **120ms TTFT P99** at rate=2000 (55% better than RR). But the admission gate is crude — it rejects based on max QueueDepth > threshold, discarding 50% of requests. Many rejected sheddable requests COULD have been served within their SLO budget (their prefill is fast due to cache hits). And some admitted requests miss their SLO anyway (long prefill on a busy instance).

## Prior Art Summary (Iterations 1-13)
- PA:4,QD:3 is Bayesian-optimal routing ratio (iter 13)
- KV-utilization scorer is counterproductive (iter 8)
- SLO-differentiated routing fragments cache affinity (iter 5)
- Admission control is the breakthrough 3rd lever at high load (iter 11: +47% vs RR)
- Priority scheduling helps only when admission creates mixed queues (iter 3 vs 11)
- Compute floor: ~11ms with cache, ~20ms without (beta0 + beta1 × tokens)

---

# Idea 1: PredictiveSLOAdmission

## Algorithm

The admission controller estimates each request's **best-case TTFT** across all instances, then admits only if the estimated TTFT is within the request's SLO budget. This is physics-informed: it uses the same beta coefficients that drive the simulator's latency model.

### Parameterized Template (6 tunable parameters)

```go
type PredictiveSLOConfig struct {
    // Per-SLO-class TTFT budgets in microseconds.
    // Requests whose best-case estimated TTFT exceeds their budget are rejected.
    BudgetCritical  float64  // default: 200000 (200ms)
    BudgetStandard  float64  // default: 500000 (500ms)
    BudgetSheddable float64  // default: 300000 (300ms)

    // Headroom multiplier on budget for admission decisions.
    // 1.0 = exact budget, 1.5 = admit if estimated TTFT < 1.5x budget
    Headroom float64  // default: 1.0, range: [0.5, 3.0]

    // Average step time for queue wait estimation (microseconds).
    // This is the per-step latency that each queued request ahead of you costs.
    AvgStepTime float64  // default: 7000 (7ms), range: [5000, 20000]

    // Beta coefficients for prefill time estimation
    Beta0 float64  // step overhead (default: 6910.42)
    Beta1 float64  // per-cache-miss-token cost (default: 17.67)
}
```

### Core Algorithm

```go
func (p *PredictiveSLOAdmission) Admit(req *Request, state *RouterState) (bool, string) {
    // Critical: ALWAYS admit (never shed latency-sensitive traffic)
    if req.SLOClass == "critical" {
        return true, ""
    }

    // Compute this request's token count
    totalTokens := len(req.InputTokens)
    blockHashes := p.prefixIdx.ComputeBlockHashes(req.InputTokens)

    // Find the BEST-CASE estimated TTFT across all instances
    // = the instance where this request would have the lowest TTFT
    bestEstimatedTTFT := math.MaxFloat64

    for _, snap := range state.Snapshots {
        // Queue wait: how long this request waits for requests ahead
        queueWait := float64(snap.QueueDepth) * p.config.AvgStepTime

        // Prefill time: depends on cache match at this instance
        cacheMatch := p.prefixIdx.MatchLength(blockHashes, snap.ID)
        cacheMissTokens := totalTokens - cacheMatch * p.prefixIdx.BlockSize()
        if cacheMissTokens < 0 { cacheMissTokens = 0 }
        prefillTime := p.config.Beta0 + p.config.Beta1 * float64(cacheMissTokens)

        estimatedTTFT := queueWait + prefillTime
        if estimatedTTFT < bestEstimatedTTFT {
            bestEstimatedTTFT = estimatedTTFT
        }
    }

    // Get this request's SLO budget
    budget := p.budgetFor(req.SLOClass)

    // Admission decision: admit if best-case TTFT is within budget
    if bestEstimatedTTFT <= budget * p.config.Headroom {
        return true, fmt.Sprintf("predictive-admit (est=%.0fμs <= budget=%.0fμs)",
            bestEstimatedTTFT, budget)
    }

    // Reject: estimated TTFT exceeds budget even on the best instance
    return false, fmt.Sprintf("predictive-reject[%s] (est=%.0fμs > budget=%.0fμs)",
        req.SLOClass, bestEstimatedTTFT, budget)
}
```

### Why This Differs From SLOGatedAdmission (iter 11)

| Aspect | SLOGated (iter 11) | Predictive (iter 14) |
|--------|-------------------|---------------------|
| Signal | maxQueueDepth (cluster-wide scalar) | Per-instance (queueWait + prefillTime) |
| Cache awareness | None | Uses PrefixCacheIndex match length |
| Per-request adaptation | No — same threshold for all requests of a class | Yes — estimates THIS request's TTFT based on ITS tokens |
| SLO budget | Implicit (threshold is a proxy) | Explicit (budget in μs per SLO class) |
| Physics | None | Beta coefficients drive estimation |

## Hypotheses

### H14-1: Predictive admission admits 30-40% more requests than SLOGated at the SAME critical TTFT P99

**Mechanism**: A sheddable request with a 512-token prefix match has estimated prefill = beta0 + beta1×0 = 6910μs ≈ 7ms (full cache hit). SLOGated rejects it because maxQueueDepth > threshold. Predictive admits it because estimatedTTFT = 7ms + queueWait << budget.

**Quantitative prediction**: At rate=2000, SLOGated completed 750/1500 (50%). Predictive should complete 1050-1200/1500 (70-80%) at the same ~120ms critical P99.

**Control**: SLOGated compound (iter 11/13) at same rate, seed, workload.

### H14-2: Positive feedback loop — cache-hit-favoring admission increases effective cache hit rate

**Mechanism**: Predictive admits cache-hit requests (low estimated TTFT) and rejects cache-miss requests (high estimated TTFT). This means the SERVED workload has higher cache hit rate than the ARRIVING workload → lower average prefill → more capacity → even more requests admitted.

**Prediction**: Steady-state admitted cache hit rate > 60% (vs ~33% for uniform admission at equal SLO fractions).

**Measurement**: Count cache-hit vs cache-miss among admitted requests.

### H14-3: Predictive + pa:4,qd:3 + SLOClassPriority achieves TTFT P99 < 100ms at rate=2000

**Mechanism**: The positive feedback loop (H14-2) increases effective cache hit rate → shorter average prefill → shorter queue waits → lower P99 for ALL admitted requests.

**Prediction**: If cache hit rate rises to 60%+ and average prefill drops from 15ms to 10ms, P99 should drop from 120ms to 80-100ms.

**Control**: iter-13 compound (120ms) at same rate.

### H14-4: Throughput (admitted × completed) matches or exceeds SLOGated

**Mechanism**: By admitting cache-hit sheddable requests (which complete quickly) and rejecting cache-miss requests (which would consume long prefill time), the system processes MORE requests per unit time.

**Prediction**: Total completed tokens/sec ≥ 95% of iter-13.

## Component Isolation Matrix (5 configs)

| # | Config | Admission | Routing | Scheduler | Priority |
|---|--------|-----------|---------|-----------|----------|
| 1 | Baseline | always-admit | pa:3,qd:2,kv:2 | fcfs | constant |
| 2 | SLOGated (iter 11) | slo-gated | pa:4,qd:3 | priority-fcfs | slo-class |
| 3 | Predictive (this) | predictive-slo | pa:4,qd:3 | priority-fcfs | slo-class |
| 4 | Predictive-only | predictive-slo | pa:3,qd:2,kv:2 | fcfs | constant |
| 5 | RR | always-admit | round-robin | fcfs | constant |

Test at rate=2000 with 8 instances, orthogonal mixed-SLO workload, 3 seeds.

## Bayesian Optimization Space

After verifying default parameters, optimize:
1. `BudgetSheddable` (100000-500000 μs): controls how aggressively sheddable is rejected
2. `Headroom` (0.5-3.0): tolerance on TTFT estimation
3. `AvgStepTime` (5000-20000 μs): queue wait calibration
4. `pa_weight` (2-5): routing PA weight
5. `qd_weight` (2-5): routing QD weight

5 parameters × 30 calls × 3 seeds = 90 BLIS runs.

---

## Reviews for Idea 1

### GPT-4o Review (Iter 14 Idea 1)

**Verdict: REVISE — address statistical power, calibration circularity, and fairness metric before implementing.**

**1. Baseline reliability (120ms P99).** The iter-13 result rests on P99 of ~250 completed requests per seed (750 total across 3 seeds). P99 of 250 samples is the 2.5th-worst value -- effectively one observation separating P99 from P98. At this sample size, the 95% confidence interval on the P99 spans roughly +/-30% (bootstrap estimate for heavy-tailed latency distributions). A 120ms baseline could plausibly be 85-155ms. Recommendation: run 2000+ requests per seed (6000+ completed across seeds) so P99 is determined by ~20 observations per seed, reducing CI to +/-8%. Without this, any claimed improvement under 25% is indistinguishable from noise.

**2. Budget calibration circularity.** `BudgetSheddable` controls the admission/rejection boundary, but the TTFT distribution of admitted requests depends on which requests are admitted. This is a textbook feedback loop: tight budget rejects borderline requests, lowering queue depth, making the remaining estimates look good -- until you loosen the budget and queue depth climbs back. The Bayesian optimization over BudgetSheddable (100k-500k us) will find a local optimum, but it is workload-specific. More concerning: the "positive feedback loop" in H14-2 is the same circularity reframed as a feature. The system selects for cache-hit requests, inflating the effective cache hit rate, which is tautological -- you rejected the cache-miss requests by construction. This needs a fairness-aware metric (point 5) to have meaning.

**3. Queue wait under PriorityFCFS.** The estimate `queueWait = QueueDepth * AvgStepTime` assumes each queued request costs exactly one step cycle of delay. Under PriorityFCFS, a critical request inserted behind 5 sheddable requests jumps ahead of all of them. Its actual wait is ~0 (next batch). Conversely, a sheddable request behind 2 critical requests waits longer than `QueueDepth * AvgStepTime` because it is deferred. Since this controller only admits non-critical requests (critical always-admit), the estimate systematically UNDERESTIMATES queue wait for the very requests it is trying to protect. The fix is straightforward: count only same-or-higher-priority requests in the queue, or use per-priority queue depths from the snapshot. Without this, the controller admits requests that will actually miss their budget.

**4. O(N x B) cost is acceptable.** `ComputeBlockHashes` is O(B) where B = input_tokens / blockSize (typically 32-64 blocks for 512-1024 tokens). `MatchLength` per instance is O(B) hash lookups. For N=8 instances: 8 x 64 = 512 hash lookups per admission decision. At 2000 req/s that is ~1M lookups/s -- trivial for an in-memory map. The real concern is that the prefix cache index is already computed by the routing scorer (`routing_prefix_scorer.go:34`). Doing it twice (admission + routing) is wasteful. Share the `PrefixCacheIndex` instance and cache `ComputeBlockHashes` per request.

**5. Fairness comparison.** Comparing P99 of completed requests between Predictive (80% completion) and SLOGated (50% completion) is misleading -- Predictive cherry-picks easy requests. The correct comparison metric is **goodput**: requests completing within their SLO budget per unit time. A system that completes 600/1500 within SLO at rate=2000 is better than one completing 400/750 within SLO, even if the second has lower P99. Add a `goodput = completed_within_slo / total_arriving` column to the comparison matrix. Without this, H14-1's "30-40% more completions at same P99" claim is unfalsifiable -- of course P99 stays low when you selectively admit the fastest requests.

### Gemini Review (Iter 14 Idea 1)

**1. Is predictive TTFT estimation feasible at the llm-d EPP layer?**

Partially. The EPP already maintains a router-side prefix cache index (the "prefix store" in llm-d's Endpoint Picker) and receives per-instance health signals including queue depth from periodic health probes. Those two signals cover cacheMatch and queueWait. The hard part is beta coefficients. In llm-d today, the EPP has no model of per-token latency -- it scores endpoints by load and cache affinity, not by predicted latency. Injecting beta coefficients requires either (a) a ConfigMap per model/GPU/TP triple that the EPP reads at startup, or (b) a sidecar calibration loop that periodically regresses observed TTFT against token counts (online learning). Option (b) is more operationally viable but adds a feedback delay of minutes. The AvgStepTime parameter is essentially a proxy for system throughput and changes with batch size, making it the least stable input. Verdict: feasible with engineering investment, but the calibration pipeline for beta is the gap that does not exist in llm-d today.

**2. Mapping to Gateway API Inference Extension (GAIE)**

GAIE defines InferenceModel with `criticality: Critical | Sheddable` (two tiers, not three). The predictive admission maps cleanly to an EPP filter that runs BEFORE endpoint selection: if the model's criticality is Sheddable and estimated best-case TTFT exceeds a budget annotation on the InferenceModel CR, return 429. Critical requests bypass the filter entirely, matching the algorithm's `if req.SLOClass == "critical" { return true }`. This is NOT an ESE (Endpoint Selection Extension) -- ESEs pick endpoints, not gate admission. It is a pre-filter in the EPP's request pipeline, analogous to how llm-d's flow control currently operates. The three-tier SLO (critical/standard/sheddable) would require a GAIE extension or mapping standard into the "Standard" bucket via an annotation. The GAIE criticality field is intentionally minimal, so budget parameters would live in annotations or a CRD sidecar.

**3. Cache-hit favoritism and cold-start unfairness**

This is a real production concern and the review document underestimates it. Under sustained high load, predictive admission creates a two-class system: warm prefix groups (admitted, reinforcing their cache warmth via H14-2's positive feedback loop) and cold prefix groups (rejected, never warming up). In production, this means a new tenant deploying a new system prompt is perpetually shed until load drops. Mitigations: (a) a minimum admission rate per prefix group (fairness floor), (b) a "cold start bonus" that treats the first N requests of a new prefix group as critical regardless of SLO class, or (c) periodic cache-warming slots reserved for novel prefixes. Without one of these, the strategy is unsuitable for multi-tenant serving. BLIS could test this by adding a "new prefix group" arrival mid-simulation and measuring its admission rate versus established groups.

**4. Operational complexity**

Six parameters is too many for production. Beta0 and Beta1 change per model, per GPU, per TP, and per vLLM version (kernel updates shift coefficients). AvgStepTime changes with load. In practice, operators would need to run BLIS offline to calibrate each deployment, then inject those parameters via ConfigMap. This is viable for large-scale operators (10+ model deployments with SRE teams) but not for the typical Kubernetes user. The path to adoption is auto-calibration: the EPP observes actual TTFT and queue depth over a window and regresses beta online. This reduces user-facing parameters to just the SLO budgets (3 values) and headroom (1 value), which are business decisions, not system tuning.

**5. KubeCon framing for SIG-LLM**

Title: "Predictive SLO Admission: When Your Load Balancer Knows Physics." The hook: current llm-d admission is binary (admit/reject based on queue depth), blind to the fact that a cache-hit request completes 3x faster than a cache-miss request at the same queue depth. The result: by using the same prefix-cache index that already exists in the EPP plus a lightweight latency model, you admit 40% more requests at the same P99 TTFT. The demo: BLIS simulation showing the positive feedback loop (cache-hit admission -> higher effective cache rate -> more capacity -> more admissions) compared to queue-depth-only gating. The call to action: contribute the latency estimation module to llm-d's EPP as an optional admission filter, with auto-calibration via observed TTFT regression. Frame it as "completing the loop" -- llm-d already has the cache index and health probes, this just teaches the EPP to multiply.
