# H24: Combined Pathological Anomalies

**Status:** Confirmed
**Resolution:** Clean confirmation
**Family:** Robustness/failure-mode
**VV&UQ:** Verification (anomaly detectors should fire)
**Tier:** 2 (diagnostic validation)
**Type:** Statistical/Dominance
**Date:** 2026-02-22
**Rounds:** 1

## Hypothesis

> "Combining always-busiest routing with inverted-slo scheduling should produce maximum measurable anomalies."

## Experiment Design

**Classification:** Statistical/Dominance — pathological configuration should be strictly worse on anomaly counters and latency metrics.

**Configurations compared:**
- A (Normal): `--routing-policy least-loaded --scheduler priority-fcfs --priority-policy slo-based --num-instances 4`
- B (Pathological): `--routing-policy always-busiest --scheduler priority-fcfs --priority-policy inverted-slo --num-instances 4`

**WARNING:** H14 BUG 3 established that `inverted-slo` + `reverse-priority` cancels out (double inversion = identity). Per issue #295, we use `inverted-slo` + `priority-fcfs` for a true single inversion.

**Controlled variables:** 4 instances, rate=2000, 500 requests, mixed-SLO workload (33% realtime, 34% interactive, 33% batch), `--trace-level decisions --summarize-trace`, `--scheduler priority-fcfs`

**Varied variable:** Routing policy (least-loaded vs always-busiest) and priority policy (slo-based vs inverted-slo)

**Seeds:** 42, 123, 456

**Preconditions verified:**
- `always-busiest` valid routing policy (`sim/bundle.go:62`)
- `inverted-slo` valid priority policy (`sim/bundle.go:63`)
- Priority inversion detector NOT suppressed for `inverted-slo` (only suppressed for `constant`/`""`, `sim/cluster/metrics.go:172-173`)
- `--scheduler priority-fcfs` (NOT `reverse-priority`) to avoid H14 BUG 3 double-inversion cancellation

## Results

### Experiment 1: Normal vs Pathological (3 seeds)

| Seed | Config | TTFT Mean (ms) | TTFT P99 (ms) | E2E P99 (ms) | HOL | Inversions | StdDev | Distribution |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 42 | normal | 489.2 | 1,078.0 | 14,806.0 | 0 | 1,596 | 1.9 | [125, 127, 126, 122] |
| 42 | pathological | 2,370.8 | 5,166.4 | 18,211.9 | 1 | 9,963 | 216.5 | [500] |
| | **Effect** | **4.8x** | **4.8x** | **1.2x** | | **6.2x** | | |
| 123 | normal | 425.9 | 924.5 | 17,368.0 | 0 | 1,854 | 0.7 | [124, 125, 125, 126] |
| 123 | pathological | 2,103.2 | 4,547.9 | 20,749.0 | 1 | 10,701 | 216.5 | [500] |
| | **Effect** | **4.9x** | **4.9x** | **1.2x** | | **5.8x** | | |
| 456 | normal | 440.5 | 886.2 | 15,600.8 | 0 | 1,732 | 0.7 | [126, 124, 125, 125] |
| 456 | pathological | 1,960.3 | 4,468.4 | 18,920.5 | 1 | 10,339 | 216.5 | [500] |
| | **Effect** | **4.5x** | **5.0x** | **1.2x** | | **6.0x** | | |

**Summary:**
- **Pathological HOL blocking > 0 in all seeds:** Yes (HOL=1 in all)
- **Pathological inversions > 0 in all seeds:** Yes (9,963-10,701)
- **Normal HOL blocking == 0 in all seeds:** Yes
- **Average TTFT P99 degradation:** 4.9x
- **Distribution stddev** confirms total imbalance: all 500 requests go to one instance (stddev=216.5 vs 0.7-1.9 for balanced)

**Note on normal inversions:** The normal configuration shows 1,596-1,854 "inversions." These are within-SLO-class false positives from the 2x threshold heuristic in `detectPriorityInversions()` (`sim/cluster/metrics.go:220`): batch requests with high variance in input size (exponential mean=1024) naturally produce E2E > 2x for earlier large requests vs later small ones. This is documented in H14 BUG 2. The key signal is the 6x increase in the pathological config.

### Experiment 2: Decomposed (seed 42)

| Configuration | TTFT P99 (ms) | E2E P99 (ms) | HOL | Inversions | StdDev | Distribution |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Normal (all correct) | 1,078.0 | 14,806.0 | 0 | 1,596 | 1.9 | [125, 127, 126, 122] |
| Routing-only pathological | 4,848.4 | 18,573.0 | 1 | 2,859 | 216.5 | [500] |
| Scheduling-only pathological | 1,270.2 | 14,672.3 | 0 | 2,158 | 2.2 | [124, 128, 126, 122] |
| All pathological | 5,166.4 | 18,211.9 | 1 | 9,963 | 216.5 | [500] |

**Attribution (TTFT P99 delta from normal):**
- Routing only: +3,770.4 ms (349.8%)
- Scheduling only: +192.2 ms (17.8%)
- Combined: +4,088.4 ms (379.3%)
- **Super-additivity:** +125.8 ms (combined > sum of parts)

**Key finding:** Routing (`always-busiest`) is the dominant contributor at ~95% of the total TTFT degradation. Scheduling (`inverted-slo`) contributes only ~5% independently. However, the combined configuration shows a small super-additive effect (+125.8 ms) and a dramatically amplified inversion count: 9,963 combined vs 2,859 (routing-only) + 2,158 (sched-only) = 5,017 expected additive. The 5,000 extra inversions arise from the interaction of queue concentration (routing) with priority distortion (scheduling).

### Experiment 3: Per-SLO Class Impact (seed 42)

| SLO Class | Normal TTFT P99 (ms) | Patho TTFT P99 (ms) | TTFT Ratio | Normal E2E P99 (ms) | Patho E2E P99 (ms) | E2E Ratio |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| batch | 1,131.8 | 5,172.2 | 4.6x | 21,560.2 | 25,016.6 | 1.2x |
| interactive | 1,020.0 | 5,135.7 | 5.0x | 5,761.2 | 9,529.5 | 1.7x |
| realtime | 1,068.6 | 5,066.5 | 4.7x | 2,007.9 | 5,595.7 | 2.8x |

**Key finding:** Realtime class has the highest E2E P99 degradation ratio (2.8x) despite having the smallest absolute values. This is expected: `inverted-slo` deprioritizes older requests, and realtime requests (small, fast) are starved behind the queue of requests that `always-busiest` concentrated on one instance. Batch requests show minimal E2E degradation (1.2x) because their long service time dominates over queueing delay. TTFT degradation is nearly uniform across classes (~4.6-5.0x) because TTFT is dominated by wait-queue time, which is inflated equally for all classes when all 500 requests queue on one instance.

## Root Cause Analysis

### Mechanism 1: Queue Concentration via `always-busiest` (RCV-1)

**Location:** `sim/routing.go:270-288`

`AlwaysBusiest.Route()` selects the instance with the highest `EffectiveLoad()`. At initialization, all instances have load=0. The first request goes to `instance_0` (tie-breaking by index). After one request, `instance_0` has load=1, making it the "busiest." All subsequent requests pile onto `instance_0`. Result: distribution = [500, 0, 0, 0].

**Direction:** This creates a single-instance bottleneck. 500 requests queue behind each other instead of spreading across 4 instances. TTFT increases because queue depth is ~4x higher.

**First-principles TTFT estimate (RCV-2):** With 4 instances and balanced routing, each instance gets ~125 requests. With 1 instance getting all 500, average queue depth is ~4x higher. TTFT is dominated by wait-queue time (time from arrival to first scheduling). For a batch of requests arriving at rate lambda on a single server with service time S, mean wait time scales roughly as E[W] ~ (lambda * S) / (1 - rho). With rho going from rho_balanced (spread over 4 servers) to rho_concentrated (all on 1), the effective utilization on the single instance is ~4x higher. The observed 4.9x TTFT P99 degradation is consistent with this: the P99 includes additional queueing effects at high utilization where Little's law gives disproportionately worse tail latency.

**Control experiment:** Routing-only pathological (Experiment 2) isolates this mechanism. It produces HOL=1, distribution=[500], and TTFT P99=4,848.4 ms — confirming that routing alone causes ~95% of the degradation.

### Mechanism 2: Priority Inversion via `inverted-slo` (RCV-1)

**Location:** `sim/priority.go:37-48`

`InvertedSLO.Compute()` returns `BaseScore - AgeWeight * age`. Newer requests get higher priority scores. When used with `priority-fcfs` scheduler (`sim/scheduler.go:27-38`, sorts by `reqs[i].Priority > reqs[j].Priority` i.e. highest priority first), newer requests jump ahead of older ones. This is anti-FCFS: the last request to arrive is served first.

**Direction:** Older requests are starved, increasing tail latency. The effect is modest on its own (+192.2 ms TTFT P99, 17.8%) because with 4 instances and load-balanced routing, each instance's queue is shallow (~125 requests) and the inversion window is limited.

**Control experiment:** Scheduling-only pathological (Experiment 2) isolates this mechanism. It produces TTFT P99=1,270.2 ms (vs 1,078.0 ms normal), confirming a modest but measurable independent effect.

### Mechanism 3: Super-Additive Interaction (RCV-3)

When both pathological policies combine:
1. `always-busiest` creates a single deep queue of 500 requests on one instance
2. `inverted-slo` + `priority-fcfs` continuously reorders this deep queue, putting the newest arrivals first

The deep queue (from mechanism 1) amplifies the inversion effect (mechanism 2) because there are more requests to reorder.

**First-principles calculation (RCV-2):** The detector at `sim/cluster/metrics.go:218-224` counts all pairs (i,j) where earlier request i has E2E > 2x later request j, within the same SLO class. With n requests per instance and k instances:
- Balanced (k=4, n=125): Total pairs examined = 4 x C(125,2) = 4 x 7,750 = 31,000 pairs across 4 instances
- Concentrated (k=1, n=500): Total pairs examined = 1 x C(500,2) = 124,750 pairs on 1 instance
- Ratio: 124,750 / 31,000 = 4.02x more candidate pairs

The inversion count depends on the fraction of pairs that exceed 2x, not just pair count. But inverted-slo increases this fraction by systematically placing newer (shorter-E2E) requests before older (longer-E2E) ones. The combined effect: more pairs (from concentration) x higher inversion rate (from priority distortion) = super-additive.

**Evidence:** Combined inversions (9,963) > routing inversions (2,859) + scheduling inversions (2,158) = 5,017. The excess 4,946 inversions represent the interaction effect. The 2.0x ratio (9,963 / 5,017) is consistent with the ~4x candidate-pair increase modulated by the inversion rate.

**Control experiment (RCV-4):** To confirm this is truly super-additive and not an artifact, one would need to run always-busiest routing with constant priority (no inversion) and verify inversions match the routing-only count. The routing-only experiment (always-busiest + slo-based) with 2,859 inversions vs normal (slo-based, balanced routing) with 1,596 inversions shows that queue concentration alone increases within-class inversions by ~1.8x.

## Devil's Advocate (RCV-5)

**If this is "Confirmed," argue why it might be Refuted:**

The HOL blocking count is only 1 in all seeds. One could argue this represents a single detection event (a threshold crossing in the detector) rather than continuous blocking. The detector uses a 2x-mean threshold (`sim/cluster/metrics.go:231-234`), meaning HOL=1 counts the number of instances exceeding the threshold, not the duration or severity. With only 1 busy instance out of 4, the count can only be 0 or 1. A more granular metric (e.g., cumulative excess queue-seconds) might reveal less dramatic blocking than the binary detection suggests.

**If this is "Refuted," argue why it might be Confirmed:**

The TTFT P99 degradation is 4.5-5.0x across all 3 seeds with zero overlap between normal and pathological ranges. The distribution is maximally imbalanced ([500] vs [~125 each]). Priority inversions increase 6x. These effects are large, consistent, and mechanistically explained. The hypothesis is directionally correct regardless of the specific HOL counter value.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| Combined pathological produces HOL blocking, priority inversions, 4.9x TTFT degradation | Confirmation | Documented here |
| Routing (`always-busiest`) is the dominant contributor (~95% of TTFT degradation) | Confirmation | Documented here; consistent with H14 |
| Priority inversion effect is modest alone (~18%) but amplified by queue concentration | Confirmation with nuance | Documented here |
| Super-additive interaction between routing and scheduling pathologies | Confirmation | Documented here |
| Realtime class has highest E2E degradation ratio (2.8x) due to priority starvation | Confirmation | Documented here |
| Normal config shows ~1,600-1,854 "inversions" (false positives from 2x threshold) | Known issue | H14 BUG 2 documented |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? None found
- [x] Any new rules needed? None — H14 already documented the HOL detector fix (#291) and priority inversion false positive issue (#292)
- [x] Any new invariants needed? None
- [x] Any existing rules/invariants confirmed? R20 (degenerate detector inputs) — the HOL detector now correctly handles all-traffic-on-one-instance after H14's fix

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, rate=2000, 500 requests, mixed-SLO workload (33% realtime / 34% interactive / 33% batch), default KV blocks (5000)
- **Parameters findings depend on:** Multiple instances (always-busiest has no effect at 1 instance), mixed SLO classes (priority inversion is meaningless with homogeneous workload)
- **What was NOT tested:** Different instance counts (2, 8, 16), different rates (low load where queueing is minimal), different workload compositions, KV-constrained scenarios, weighted routing policies
- **Generalizability:** The 4.9x TTFT degradation is specific to 4 instances (1/N utilization). At N instances, always-busiest wastes (N-1)/N capacity, so TTFT degradation should scale roughly linearly with N. The super-additive interaction should also scale with N (deeper single queue = more inversion pairs).
- **Uncertainty quantification:** 3 seeds tested, all consistent (TTFT ratio range: 4.5-5.0x). Narrow variance suggests the result is robust at this operating point. UQ for different rates or instance counts not performed.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT P99 degradation | 4.9x (avg across 3 seeds) | High — consistent across seeds, mechanistically explained |
| HOL blocking detection | 1 in all pathological runs, 0 in all normal | High — binary but correct |
| Priority inversions | 6x increase (pathological vs normal) | Medium — includes baseline false positives (H14 BUG 2) |
| Super-additivity | +125.8 ms TTFT, +4,946 inversions | Medium — single seed (42), would benefit from multi-seed decomposition |
| Sample size | 3 seeds x 2 configs x 500 requests = 3,000 | Adequate for dominance test |
| Mechanism | Queue concentration + priority distortion | High — decomposition experiment confirms independent and interaction effects |

## Implications for Users

1. **Never use `always-busiest` routing in production** — it defeats the purpose of multi-instance deployment by concentrating all load on one instance. TTFT degrades proportionally to instance count.
2. **`inverted-slo` has a modest standalone effect** — at balanced load across 4 instances, it increases TTFT P99 by only ~18%. The effect is amplified when combined with load imbalance.
3. **Anomaly detectors work correctly** — HOL blocking and priority inversion counters fire as expected with pathological configurations. The HOL detector now correctly handles the all-traffic-on-one-instance case (fixed after H14 BUG 1).
4. **Realtime workloads are most vulnerable** to priority inversion because their short service time means queueing delay dominates E2E. Batch workloads are relatively insensitive.

## Reproducing

```
cd hypotheses/h24-combined-pathological
./run.sh
```
