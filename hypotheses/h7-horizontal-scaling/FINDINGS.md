# H7: Horizontal Scaling

**Status:** Confirmed with nuance
**Resolution:** Confirmation with surprise
**Family:** Performance-regime (scaling laws)
**VV&UQ:** Validation
**Tier:** 2 (behavioral comparison)
**Type:** Statistical (Monotonicity)
**Date:** 2026-02-23
**Rounds:** 2

## Hypothesis

> "Increasing instances from 4 to 8 should roughly halve TTFT p99 for saturated workloads."

The intuition is that if a workload saturates k instances (causing long queues and high utilization), adding more instances absorbs the excess load. With least-loaded routing distributing requests evenly, TTFT p99 should decrease because per-instance queue depths drop proportionally.

## Experiment Design

**Classification:** Statistical / Monotonicity (sweep 3 values)

**Configurations compared:**
- A (2 inst): `--num-instances 2 --rate 1000 --num-requests 500 --routing-policy least-loaded --scheduler fcfs --priority-policy constant --admission-policy always-admit --log error --summarize-trace --trace-level decisions`
- B (4 inst): Same with `--num-instances 4`
- C (8 inst): Same with `--num-instances 8`
- Control: Same configurations at `--rate 100` (sub-saturation)

**Controlled variables:** Model (llama-3.1-8b-instruct), rate (1000 for main, 100 for control), num-requests (500), routing (least-loaded), scheduler (fcfs), priority (constant), admission (always-admit), KV blocks (132139 from defaults.yaml for this model/GPU/TP -- no KV pressure)

**Varied variable:** `--num-instances` (2, 4, 8)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- Saturation confirmed: At rate=1000, TTFT p99 at 4 instances (917.70 ms) is >> TTFT p99 at rate=100 (39.42 ms), confirming ~23x amplification from queue buildup
- No KV pressure: 0 preemptions across all runs (132139 blocks from defaults.yaml is ample)
- INV-1 conservation: passes for all 18 runs

**Reference:** `hypotheses/h16-gamma-vs-poisson/run.sh` (same rate-sizing rationale, same cluster configuration except instance count is varied)

## Results

### Experiment 1: Saturating sweep (rate=1000, 500 requests)

| Inst | Seed | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) | Throughput |
|------|------|-----------------|----------------|----------------|---------------|------------|
| 2 | 42 | 1181.74 | 2269.06 | 7235.61 | 12049.60 | 42.49 |
| 4 | 42 | 483.12 | 917.70 | 5849.20 | 10659.70 | 48.91 |
| 8 | 42 | 144.02 | 260.48 | 5164.72 | 9976.63 | 52.67 |
| 2 | 123 | 1127.49 | 2289.38 | 6653.74 | 12313.39 | 43.47 |
| 4 | 123 | 455.43 | 881.67 | 5318.51 | 10973.84 | 49.49 |
| 8 | 123 | 130.39 | 265.83 | 4661.38 | 10322.30 | 53.02 |
| 2 | 456 | 1161.28 | 2300.50 | 6985.98 | 11504.06 | 42.49 |
| 4 | 456 | 473.00 | 1014.16 | 5619.68 | 9949.93 | 49.19 |
| 8 | 456 | 139.95 | 308.64 | 4946.50 | 9276.61 | 52.97 |

**Scaling ratios:**

| Seed | TTFT p99 4/2 | TTFT p99 8/4 | Throughput 4/2 | Throughput 8/4 |
|------|--------------|--------------|----------------|----------------|
| 42 | 0.404x | 0.284x | 1.151x | 1.077x |
| 123 | 0.385x | 0.302x | 1.138x | 1.071x |
| 456 | 0.441x | 0.304x | 1.158x | 1.077x |
| **Avg** | **0.410x** | **0.297x** | **1.149x** | **1.075x** |

### Control: Sub-saturation (rate=100, 500 requests)

| Inst | Seed | TTFT mean (ms) | TTFT p99 (ms) | E2E mean (ms) | E2E p99 (ms) | Throughput |
|------|------|-----------------|----------------|----------------|---------------|------------|
| 2 | 42 | 27.74 | 44.76 | 6005.06 | 11213.58 | 37.22 |
| 4 | 42 | 25.41 | 39.42 | 5319.58 | 10304.35 | 38.29 |
| 8 | 42 | 25.13 | 39.37 | 5003.94 | 9828.23 | 38.83 |
| 2 | 123 | 27.25 | 46.78 | 5469.20 | 11462.52 | 38.22 |
| 4 | 123 | 25.44 | 40.46 | 4810.25 | 10570.78 | 40.06 |
| 8 | 123 | 24.90 | 37.89 | 4510.31 | 10193.82 | 41.01 |
| 2 | 456 | 28.06 | 48.78 | 5790.63 | 10758.05 | 36.53 |
| 4 | 456 | 25.51 | 43.60 | 5108.98 | 9698.45 | 37.34 |
| 8 | 456 | 25.13 | 38.82 | 4793.05 | 9180.56 | 37.68 |

**Control scaling ratios:**

| Seed | TTFT p99 8/4 |
|------|--------------|
| 42 | 0.999x |
| 123 | 0.936x |
| 456 | 0.891x |
| **Avg** | **0.942x** |

At sub-saturation, the TTFT p99 scaling effect essentially vanishes (0.942x vs 0.297x under saturation), confirming the mechanism depends on queue buildup.

## Root Cause Analysis

The TTFT p99 improvement from horizontal scaling is **better** than the predicted "roughly halve" -- the 8-vs-4 ratio averages 0.297x (70% reduction), not 0.5x (50%).

**Mechanism: Queue depth reduction via load distribution**

1. **Least-loaded routing** (`sim/routing.go:107-125`) distributes requests to the instance with the lowest `EffectiveLoad()` (`sim/routing.go:23-25`), which sums `QueueDepth + BatchSize + PendingRequests`. This achieves near-uniform distribution across instances (Join-Shortest-Queue approximation).

2. **TTFT is dominated by queue waiting time under saturation.** The alpha coefficients (`sim/latency_model.go:56-61`) add ~3.4ms of queueing delay per request (alpha0=1601 + alpha1*512=~3398 us). But the real TTFT driver under saturation is scheduling_delay -- how long a request waits in the queue before being scheduled. With k instances processing at ~57.4 req/s each (step time ~17.4ms from beta coefficients: 6910 + 17.67*512 + 2.84*512 = 17411 us), the effective arrival rate per instance is rate/k.

3. **First-principles calculation** (RCV-2): At rate=1000 req/s with 4 instances, per-instance rate is ~250 req/s against a capacity of ~57.4 req/s (rho=4.35). With 8 instances, per-instance rate is ~125 req/s (rho=2.18). Both are above 1.0, so both accumulate infinite queues, but at different rates. The 8-instance queue grows at (125-57.4)=67.6 req/s vs (250-57.4)=192.6 req/s per instance. The growth rate ratio is 67.6/192.6 = 0.351x, which is reasonably close to the observed 0.297x TTFT p99 ratio (15% discrepancy). The qualitative mechanism -- super-linear improvement from the non-linear dependence of queue growth rate on (lambda/k - mu) -- is confirmed, though the simple fluid model overestimates the ratio slightly (predicts less improvement than observed), likely because batching effects further reduce effective queueing at lower per-instance rates.

4. **E2E does not halve** because E2E = TTFT + decode time, and decode time (~5000ms for 512 output tokens at ~10ms/decode-step) is constant regardless of instance count. The TTFT improvement (from ~1000ms to ~280ms) is a small fraction of total E2E (~5500ms). E2E p99 ratio 8/4 is approximately 0.93-0.95x -- modest because decode dominates.

5. **Throughput appears to scale only 1.075x** (not 2x) because throughput = completed/sim_time, and all 500 requests complete in every configuration. The difference is in sim_time: 8 instances finish the same 500 requests faster (shorter queues mean the last request completes sooner). True capacity scaling would be visible with a longer horizon or infinite request stream.

**Control experiment validates the mechanism** (RCV-4): At rate=100 (sub-saturation, rho=0.44 at 4 instances), queues never build up significantly, so TTFT p99 is ~40ms regardless of instance count (0.942x ratio). This confirms the scaling benefit comes from queue depth reduction, not from some other effect of the `--num-instances` flag.

## Devil's Advocate (RCV-5)

**Arguing this might be Refuted:**
The "roughly halve" prediction was too conservative -- the actual improvement is 0.297x (better than 0.5x). One could argue the hypothesis is technically refuted because the prediction was wrong in magnitude. Furthermore, the throughput scaling (1.075x) is far from the ideal 2x, which challenges the "roughly proportional" part of the hypothesis. The fixed request count (500) means we're not actually measuring steady-state scaling -- we're measuring time-to-complete-fixed-batch, which conflates queue draining with steady-state throughput.

**Arguing this is Confirmed:**
The core prediction -- that TTFT p99 decreases significantly when doubling instances under saturation -- holds strongly across all seeds. The 0.297x ratio is better than the predicted 0.5x, which is the "right kind of wrong" -- the system scales super-linearly for TTFT because queue growth rates are proportional to (lambda/k - mu), not just lambda/k. The monotonicity across all 3 values (2 < 4 < 8) and all 3 seeds is unambiguous.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| TTFT p99 decreases monotonically with instance count under saturation | Confirmation | documented here |
| 8-vs-4 TTFT p99 ratio is 0.297x (better than 0.5x halving) | Surprise | documented here -- first-principles explains: queue growth rate ratio (67.6/192.6=0.351x) approximates the improvement |
| E2E is insensitive to horizontal scaling (decode dominates) | Design limitation | documented here |
| Throughput scaling appears sub-linear due to fixed request count | Design limitation | documented here |
| Sub-saturation control confirms queue-depth is the mechanism | Confirmation | documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found. All runs deterministic (INV-6), conserving (INV-1), causal (INV-5).
- [x] Any new rules needed? None -- the findings are consistent with existing queueing theory expectations.
- [x] Any new invariants needed? None -- the monotonicity of TTFT with respect to instance count under saturation is a property of queueing theory, not a system invariant.
- [x] Any existing rules/invariants confirmed? INV-1 (conservation) confirmed across all 18 runs. INV-8 (work-conserving) implicitly confirmed -- all 500 requests complete in every configuration.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 2/4/8 instances, rate=1000 (saturating) and rate=100 (sub-saturation), 500 requests, least-loaded routing, fcfs scheduler, Poisson arrivals, Gaussian input/output distributions (default mean 512/512), seeds 42/123/456, KV blocks 132139 (from defaults.yaml)
- **Parameters findings depend on:** Saturation requires rate >> capacity. Least-loaded routing ensures even distribution. Default KV blocks ensure no preemption confound.
- **What was NOT tested:** Instance counts > 8 (16, 32); intermediate saturation levels (rho near 1.0); non-least-loaded routing (round-robin would distribute unevenly); KV-constrained regimes; workloads with heterogeneous request sizes; weighted routing.
- **Generalizability:** The monotonicity finding generalizes to any configuration where (a) the workload saturates the cluster and (b) routing distributes load approximately evenly. With round-robin routing, cyclic assignment may create uneven queues that weaken the scaling benefit. With very heterogeneous request sizes, JSQ (least-loaded) may not achieve even load.
- **Uncertainty quantification:** Effect is consistent across all 3 seeds with tight ranges (8/4 ratio: 0.284x-0.304x). The tight spread suggests high confidence. UQ not formally performed (no confidence intervals computed due to n=3 seeds), but the consistency is strong evidence.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT p99 8/4 ratio (sat.) | 0.297x avg (range 0.284x-0.304x) | High -- consistent across all seeds, tight range |
| TTFT p99 monotonicity | 3/3 seeds strictly decreasing | High -- unanimous |
| Throughput monotonicity | 3/3 seeds strictly increasing | High -- unanimous |
| Sub-saturation control | 0.942x avg (effect vanishes) | High -- validates mechanism |
| Sample size | 3 seeds x 5 configs = 15 data points | Medium -- 3 seeds is standard for this project but limits formal statistical testing |
| Mechanism | Queue growth rate reduction via load distribution | High -- first-principles calculation approximates observation (0.351x predicted vs 0.297x observed, 15% discrepancy) and control confirms |

## Implications for Users

1. **Horizontal scaling is highly effective for TTFT under saturation.** Doubling instances can reduce TTFT p99 by ~70% (not just 50%) because per-instance queue growth rates decrease super-linearly.

2. **E2E latency benefits less from scaling** because decode time (proportional to output length) dominates and is independent of instance count. Users optimizing for E2E should consider reducing output length or improving per-instance decode speed.

3. **Use least-loaded routing** for horizontal scaling benefits. Round-robin may not distribute load evenly enough to achieve the full scaling benefit.

4. **Scaling is only effective under saturation.** At sub-saturation (rho < 1 per instance), adding instances has negligible TTFT impact because queues don't build up.

5. **Capacity planning rule of thumb:** If TTFT p99 is unacceptable, estimate rho = arrival_rate / (num_instances * per_instance_capacity). If rho > 1, adding instances will reduce TTFT roughly proportional to the reduction in (rho - 1).

## Reproducing

```
cd hypotheses/h7-horizontal-scaling
./run.sh
```
