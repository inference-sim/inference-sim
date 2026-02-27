# Iteration 14: Predictive TTFT-Budget Admission Control

## The Genius Insight

Our iter-11 compound rejects 50% of requests using a crude queue-depth threshold. This is wasteful — many rejected sheddable requests COULD have been served within their SLO budget, and some admitted requests end up missing their budget anyway.

**The breakthrough**: Replace reactive admission (reject when queue > threshold) with PREDICTIVE admission (reject when estimated TTFT > SLO budget for this specific request). The admission controller uses the SAME physics (beta coefficients, cache state) that drives the simulator to PREDICT each request's TTFT before deciding whether to admit it.

## Strategy Template: PredictiveSLOAdmission

```
PredictiveSLOAdmission.Admit(req, state):
  // Step 1: Estimate this request's BEST-CASE TTFT
  //   = min over all instances of (queue_wait + prefill_time)
  bestTTFT = infinity
  for each instance snap in state.Snapshots:
    // Queue wait: QueueDepth × avg_step_time
    queueWait = snap.QueueDepth * PARAM_avg_step_time

    // Prefill time: depends on cache match
    // (uses PrefixCacheIndex for approximate cache state)
    cacheMatch = PrefixCacheIndex.MatchLength(req.blockHashes, snap.ID)
    cacheMissTokens = len(req.InputTokens) - cacheMatch * blockSize
    if cacheMissTokens < 0: cacheMissTokens = 0
    prefillTime = beta0 + beta1 * cacheMissTokens

    instanceTTFT = queueWait + prefillTime
    if instanceTTFT < bestTTFT:
      bestTTFT = instanceTTFT

  // Step 2: Compare estimated TTFT to SLO budget
  budget = PARAM_slo_budgets[req.SLOClass]

  if bestTTFT <= budget * PARAM_headroom:
    return true   // admit — can meet SLO

  // Step 3: SLO-class-dependent rejection
  if req.SLOClass == "critical":
    return true    // always admit critical, even if estimated miss
  if req.SLOClass == "standard":
    return bestTTFT <= budget * PARAM_headroom * 2  // 2x tolerance
  return false     // sheddable: reject if can't meet budget
```

## Parameters (6 total, suitable for Bayesian optimization)
1. `avg_step_time` (μs): estimated per-step time for queue wait calculation (5000-20000)
2. `slo_budget_critical` (μs): TTFT budget for critical (50000-200000)
3. `slo_budget_standard` (μs): for standard (100000-500000)
4. `slo_budget_sheddable` (μs): for sheddable (200000-1000000)
5. `headroom` (float): multiplier on budget for admission tolerance (0.5-2.0)
6. `pa_weight` / `qd_weight` ratio (routing, from Bayesian iter 13)

## Why This Is Genius

1. **Physics-informed admission**: Uses beta coefficients + cache state to PREDICT TTFT, not just count queue depth
2. **Per-request adaptation**: A request with a 4096-token cache hit (11ms prefill) can be admitted even when a request with 0 cache (82ms prefill) would be rejected — because the physics differ
3. **SLO-budget-aware**: Different SLO classes have different TTFT budgets. A 200ms budget for standard allows admission at higher queue depths than a 50ms budget for critical
4. **Self-calibrating**: The estimation uses the same beta coefficients that drive the simulator, so predictions are grounded in the same physics
5. **Reduces unnecessary rejection**: Instead of blanket 50% shedding, only rejects requests that CANNOT meet their SLO — higher throughput at the same TTFT

## Hypotheses

H14-1: PredictiveSLO admits 30-40% more requests than SLOGated while achieving the SAME critical TTFT P99
- Mechanism: cache-hit sheddable requests are admitted (low prefill → can meet budget) while cache-miss sheddable are rejected
- Prediction: 70-80% completion rate (vs iter-11's 50%) at same 120ms critical P99

H14-2: The per-request cache-aware admission creates a POSITIVE FEEDBACK LOOP
- Mechanism: admitting cache-hit requests and rejecting cache-miss requests increases the effective cache hit rate of the served workload → lower average TTFT → more capacity → even more requests admitted
- Prediction: steady-state cache hit rate of admitted requests > 80% (vs ~33% for uniform admission)

H14-3: PredictiveSLO + pa:4,qd:3 + SLOClassPriority achieves TTFT P99 < 100ms at rate=2000
- Mechanism: predictive admission + optimized routing + priority scheduling compound
- Control: iter-11 compound (120ms) and iter-13 Bayesian-optimized (120ms)
