# H1-SJF: SJF Scheduling Reduces TTFT by 94% for Short Requests in Bimodal 32:1024 Workloads

**Status:** Confirmed
**Resolution:** Clean confirmation (for bimodal 32:1024 workload at rate=3000)
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Tier:** 2
**Type:** Statistical (Dominance)
**Date:** 2026-02-21
**Rounds:** 2

## Hypothesis

> SJF scheduling should reduce TTFT for mixed-length workloads. If short requests get stuck behind long ones (head-of-line blocking), scheduling short jobs first should reduce average wait time -- the classic SJF result from operating systems.

## Experiment Design

**Classification:** Statistical/Dominance

**Configurations compared:**
- A (FCFS): `--scheduler fcfs --num-instances 4 --workload-spec mixed-workload.yaml --seed <seed>`
- B (SJF): `--scheduler sjf --num-instances 4 --workload-spec mixed-workload.yaml --seed <seed>`

**Controlled variables:** model (llama-3.1-8b-instruct), num-instances (4), routing (round-robin, default), priority (constant, default), admission (always-admit, default), workload (same YAML), KV cache (default 1M blocks)

**Varied variable:** scheduler (fcfs vs sjf)

**Seeds:** 42, 123, 456

**Seed override note:** The workload YAML has `seed: 42` but this is overridden by the CLI `--seed` flag per the CLI flag precedence rule (R18). Seeds 123 and 456 correctly produce different arrival patterns, as confirmed by differing latency values across seeds.

**Preconditions verified:**
- Queue depth must exceed batch size for SJF reordering to matter. At rate=3000 req/s with 4 instances and 1000 total requests, queues build up substantially. The ~1200ms mean TTFT under FCFS confirms significant queueing. Under SJF, interactive scheduling delay drops from ~1.2M us to ~22K us, confirming the queue was deep enough for reordering to take effect.
- All 1000 requests completed under both schedulers (no starvation within the experiment horizon).

**Workload:**
- 50% "short-jobs" (interactive SLO class): 32 input tokens, 64 output tokens, Poisson arrival
- 50% "long-jobs" (batch SLO class): 1024 input tokens, 128 output tokens, Poisson arrival
- Aggregate rate: 3000 req/s, 1000 total requests
- Constant distributions for input/output to isolate the scheduling effect (ED-1)

## Results

### Aggregate Cluster Metrics

| Metric | Seed | FCFS | SJF | Change |
|--------|------|------|-----|--------|
| TTFT mean (ms) | 42 | 1269.39 | 683.10 | -46.2% |
| TTFT mean (ms) | 123 | 1234.79 | 672.10 | -45.6% |
| TTFT mean (ms) | 456 | 1164.65 | 617.30 | -47.0% |
| TTFT P99 (ms) | 42 | 2541.30 | 2538.12 | -0.1% |
| TTFT P99 (ms) | 123 | 2486.17 | 2483.89 | -0.1% |
| TTFT P99 (ms) | 456 | 2398.66 | 2384.17 | -0.6% |
| E2E mean (ms) | 42 | 3260.50 | 3140.54 | -3.7% |
| E2E mean (ms) | 123 | 3238.05 | 3130.00 | -3.3% |
| E2E mean (ms) | 456 | 3114.26 | 3016.82 | -3.1% |
| Completed | all | 1000 | 1000 | 0% |

### Per-SLO-Class TTFT (the key metric)

| SLO Class | Seed | FCFS TTFT mean (ms) | SJF TTFT mean (ms) | Change |
|-----------|------|---------------------|---------------------|--------|
| interactive | 42 | 1271.19 | 65.66 | **-94.8%** |
| interactive | 123 | 1214.28 | 66.16 | **-94.6%** |
| interactive | 456 | 1135.77 | 65.41 | **-94.2%** |
| batch | 42 | 1267.53 | 1320.62 | +4.2% |
| batch | 123 | 1256.31 | 1307.84 | +4.1% |
| batch | 456 | 1197.75 | 1249.73 | +4.3% |

### Per-SLO-Class Scheduling Delay (mechanism verification)

| SLO Class | Seed | FCFS delay (us) | SJF delay (us) | Change |
|-----------|------|-----------------|-----------------|--------|
| interactive | 42 | 1,228,241 | 22,178 | **-98.2%** |
| interactive | 123 | 1,170,758 | 22,306 | **-98.1%** |
| interactive | 456 | 1,092,465 | 21,816 | **-98.0%** |
| batch | 42 | 1,202,785 | 1,256,019 | +4.4% |
| batch | 123 | 1,191,760 | 1,243,190 | +4.3% |
| batch | 456 | 1,133,131 | 1,185,180 | +4.6% |

### Per-SLO-Class E2E

| SLO Class | Seed | FCFS E2E (ms) | SJF E2E (ms) | Change |
|-----------|------|---------------|---------------|--------|
| interactive | 42 | 2966.59 | 2716.45 | -8.4% |
| interactive | 123 | 2949.77 | 2725.40 | -7.6% |
| interactive | 456 | 2834.85 | 2640.16 | -6.9% |
| batch | 42 | 3563.97 | 3578.43 | +0.4% |
| batch | 123 | 3540.51 | 3554.50 | +0.4% |
| batch | 456 | 3434.44 | 3448.44 | +0.4% |

## Root Cause Analysis

The SJF scheduling effect is the classic shortest-job-first result from operating systems theory, confirmed to work correctly in the BLIS discrete-event simulator.

**Note on SJF sort key:** BLIS's SJF implementation sorts by `len(InputTokens)` only, not by estimated total service time (input + output). For this workload, the input-only sort gives the same ordering as total-service-time sort because short input correlates with short output. For workloads where short-input requests have very long outputs, the behavior would differ from textbook SJF.

### Mechanism (traced through code)

1. **Queue reordering** (`sim/simulator.go:481-483`): Before each step's batch formation, the simulator calls `sim.scheduler.OrderQueue(reqs, now)` on the wait queue. Under SJF (`sim/scheduler.go:46-57`), this sorts requests by `len(reqs[i].InputTokens)` ascending, putting 32-token requests ahead of 1024-token requests.

2. **Batch formation** (`sim/simulator.go:makeRunningBatch`, line 355+): `makeRunningBatch()` dequeues from the front of the reordered queue. Under SJF, short requests (32 input tokens) are dequeued first, getting scheduled into the running batch before long requests.

3. **Scheduling delay reduction**: Under FCFS, short requests arriving after long requests must wait for those long requests to be scheduled first. Under SJF, short requests jump ahead. The scheduling delay for interactive requests drops from ~1.2M us (FCFS) to ~22K us (SJF) -- a 98% reduction.

4. **TTFT reduction follows from scheduling delay**: TTFT = scheduling_delay + prefill_time. Since prefill time for 32 tokens is small (~6.9ms from beta coefficients: 6910 + 17.67*32 = ~7.5K us), the TTFT is dominated by scheduling delay under FCFS. Under SJF, scheduling delay drops to ~22ms, so TTFT drops to ~66ms.

5. **First-principles reconciliation**: The predicted TTFT for interactive requests under SJF (scheduling_delay ~22ms + prefill ~7.5ms = ~30ms) differs from the observed ~66ms. The gap is likely from batch-level step execution: the request waits for the current batch step to complete before being included in the next `makeRunningBatch` call. This adds up to one step time (~7-12ms) of additional delay, plus the alpha model's queueing overhead.

6. **Batch requests pay slightly more**: Long requests (batch SLO class) have their scheduling delay increase by ~4.3% under SJF because they are now consistently moved behind short requests in the queue. This is the expected tradeoff -- SJF helps short jobs at the cost of slightly delaying long ones.

### Direction explained (RCV-3)

SJF reduces short-request TTFT because it eliminates head-of-line blocking. Under FCFS, a short request arriving behind N long requests must wait for all N to be scheduled. Under SJF, the short request immediately jumps to the front. The 32:1 ratio of input token counts (1024 vs 32) makes this effect extreme -- short requests are always prioritized.

The aggregate TTFT improvement (~46%) is a weighted average: short requests improve ~95% while long requests degrade ~4%. Since both classes are 50% of traffic, the net is strongly positive.

### Control experiment (RCV-4)

The FCFS configuration is the control -- it disables the SJF reordering mechanism (the scheduler is a no-op). The SJF configuration enables only the queue reordering. All other variables (routing, admission, priority, workload, instances) are held constant. The scheduling delay comparison directly isolates the mechanism.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**

1. The 94% improvement is tested only at the extreme 32:1 input token ratio. At a 2:1 ratio (256:512), the service time difference is roughly 1.7:1, and the scheduling delay reduction would be much smaller -- possibly within noise.
2. The aggregate E2E improvement is only 3-4%, meaning SJF is a zero-sum redistribution: short requests gain what long requests lose. Under load where E2E matters more than TTFT, SJF provides no system-wide benefit.
3. Under least-loaded routing (instead of round-robin), short requests might naturally be routed to less-loaded instances, partially absorbing the SJF benefit and reducing the observed effect.
4. The experiment uses constant distributions. With real workloads using Gaussian or ParetoLogNormal distributions, overlapping tails would prevent clean class separation -- SJF would reorder within-class, not just between-class.

**Counter to the counter:**
The 94% effect at the tested operating point is so large that even a 10x reduction in effect size would leave a meaningful 9.4% improvement. And the mechanism is verified: SJF reorders by input tokens, which is the dominant latency component. The directional finding holds regardless of magnitude.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| SJF reduces interactive TTFT by 94-95% for bimodal 32:1024 workloads | Confirmation | Documented here |
| SJF increases batch scheduling delay by ~4.3% (expected tradeoff) | Confirmation | Documented here |
| Aggregate TTFT drops 46% under SJF for 50/50 short/long mix | Confirmation | Documented here |
| The scheduling delay metric directly isolates the mechanism (98% reduction for short requests) | Confirmation | Documented here |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None -- the SJF starvation risk is already documented in `sim/scheduler.go:43` warning comment
- [x] Any new invariants needed? None -- INV-1 (conservation) holds (1000 completed in both configs)
- [x] Any existing rules/invariants confirmed? INV-1 confirmed (1000 completed = 1000 injected for all seeds). INV-8 (work-conserving) implicitly confirmed -- no idle instances while work waits.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, rate=3000 req/s, 1000 requests, round-robin routing, constant priority, always-admit, 1M KV blocks, bimodal workload (32 vs 1024 input tokens)
- **Parameters findings depend on:** (1) Queue depth must be large enough for reordering to matter. At very low rates, SJF and FCFS produce identical results. (2) The 32:1 token ratio maximizes the separation. Smaller ratios would show smaller effects. (3) Constant distributions isolate the scheduling variable.
- **What was NOT tested:** (a) Continuous input distributions (e.g., Gaussian) where the SJF benefit would be smaller. (b) Very low rate where queues never build up. (c) Different instance counts. (d) Weighted or least-loaded routing (may partially absorb the scheduling effect). (e) SJF under sustained load to observe starvation of long requests.
- **Generalizability:** The directional finding (SJF helps short requests at the cost of long ones) generalizes to any mixed-length workload with sufficient queueing. The magnitude (94%) is specific to the 32:1024 bimodal configuration.
- **Uncertainty quantification:** 3 seeds tested. Effect is consistent across all seeds (94.2-94.8% TTFT reduction for interactive). Standard deviation of effect size is <0.3 percentage points. High confidence in the directional finding.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| Interactive TTFT reduction | 94.2-94.8% across 3 seeds | High -- consistent across all seeds, >20% threshold |
| Batch TTFT increase | 4.1-4.3% across 3 seeds | High -- consistent direction and magnitude |
| Mechanism (scheduling delay) | 98.0-98.2% reduction for interactive | High -- directly isolates the scheduling reordering effect |
| Sample size | 3 seeds x 2 configs x 1000 requests | Adequate for dominance test |
| Conservation (INV-1) | 1000 completed in all 6 runs | High -- no data loss |

## Implications for Users

1. **Use SJF when short-request latency matters**: For workloads mixing short interactive queries with long batch jobs, SJF scheduling dramatically reduces TTFT for short requests (94%+ in this experiment) with minimal impact on long requests (<5% increase).

2. **Combine with SLO-class routing**: For production deployments, consider pairing SJF with per-SLO-class routing so that short interactive requests are both routed to less-loaded instances AND scheduled first within each instance.

3. **Monitor for starvation under sustained load**: SJF can cause starvation for long requests under sustained high load. The experiment ran 1000 requests (finite), so starvation was not observed. Under sustained load, consider adding aging or a hybrid scheduler.

4. **Effect depends on load level**: SJF provides no benefit when queues are empty (low load). The benefit increases with queue depth.

5. **Magnitude is configuration-specific**: The 94% figure is specific to the extreme 32:1024 bimodal configuration. Smaller input token ratios (e.g., 128:512) would produce smaller effects. The directional finding (SJF helps short requests) generalizes, but the magnitude depends on the input token ratio.

## Reproducing

```
cd hypotheses/h1-sjf-scheduling
./run.sh
```
