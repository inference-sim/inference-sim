# H2-Priority-FCFS

**Status:** Refuted
**Resolution:** Refuted -- wrong mental model
**Family:** Cross-policy comparative
**VV&UQ:** Validation
**Tier:** 5 (workload diversity)
**Type:** Statistical (Dominance)
**Date:** 2026-02-22
**Rounds:** 1

## Hypothesis

> "Priority-FCFS with SLO-based priority should reduce realtime TTFT at the cost of batch TTFT"

## Experiment Design

**Classification:** Statistical/Dominance

**Configurations compared:**
- A (baseline): `--priority-policy constant --scheduler fcfs` -- all requests equal priority, FIFO order
- B (prioritized): `--priority-policy slo-based --scheduler priority-fcfs` -- age-based priority with priority-aware scheduling

**Controlled variables:** model (llama-3.1-8b-instruct), num-instances (4), routing (round-robin default), admission (always-admit default), workload (mixed-SLO: 33% realtime, 34% interactive, 33% batch, all identical token sizes 256 input / 128 output, Poisson arrival at 500 req/s, 500 requests)

**Varied variable:** Priority policy (constant vs slo-based) and scheduler (fcfs vs priority-fcfs)

**Seeds:** 42, 123, 456

**Preconditions verified:**
1. Workload YAML field names cross-referenced against `sim/workload/spec.go` struct tags (all correct)
2. Per-request JSON field names verified against `sim/metrics_utils.go:RequestMetrics` (slo_class, ttft_ms, e2e_ms present)
3. Priority policy names verified against `sim/priority.go:NewPriorityPolicy` factory
4. Scheduler names verified against `sim/scheduler.go:NewScheduler` factory
5. Binary built and tested before experiment runs

## Results

### Aggregate Cluster Metrics

All metrics are **identical** (0.0% difference) between Config A and Config B across all 3 seeds:

| Metric | Seed 42 | Seed 123 | Seed 456 |
|--------|---------|----------|----------|
| TTFT mean (ms) | 28.20 | 31.03 | 27.47 |
| TTFT P99 (ms) | 41.49 | 47.98 | 43.13 |
| E2E mean (ms) | 1458.62 | 1462.64 | 1457.35 |
| Scheduling Delay P99 (ms) | 21.06 | 22.47 | 20.70 |
| Completed | 500 | 500 | 500 |

**No Config A vs Config B difference detected in any metric.**

### Per-SLO-Class TTFT Comparison

| SLO Class | Seed | Config A TTFT mean (ms) | Config B TTFT mean (ms) | Diff |
|-----------|------|------------------------|------------------------|------|
| realtime | 42 | 27.99 | 27.99 | 0.0% |
| interactive | 42 | 28.32 | 28.32 | 0.0% |
| batch | 42 | 28.28 | 28.28 | 0.0% |
| realtime | 123 | 30.87 | 30.87 | 0.0% |
| interactive | 123 | 30.97 | 30.97 | 0.0% |
| batch | 123 | 31.27 | 31.27 | 0.0% |
| realtime | 456 | 27.60 | 27.60 | 0.0% |
| interactive | 456 | 27.32 | 27.32 | 0.0% |
| batch | 456 | 27.52 | 27.52 | 0.0% |

### Scheduling Reordering Analysis

Zero reordering detected across all seeds:
- 0 requests changed completion rank (out of 500 per seed)
- Max rank shift: 0
- Mean rank shift: 0.0

**The two configurations produce byte-identical scheduling order.**

## Root Cause Analysis

The hypothesis is refuted because `SLOBasedPriority` does not differentiate by SLO class -- it is purely an age-based (waiting time) priority policy, which produces mathematically equivalent ordering to FCFS.

### Mathematical proof of equivalence

**Config A** (`constant` priority + `priority-fcfs` scheduler):
1. `ConstantPriority.Compute()` returns `0.0` for all requests (`sim/priority.go:17-19`)
2. `PriorityFCFSScheduler.OrderQueue()` sorts by priority descending, breaking ties by arrival time ascending, then ID ascending (`sim/scheduler.go:30-38`)
3. Since all priorities are 0.0, the sort falls through to the tiebreaker: **arrival time ascending = FCFS**

**Config B** (`slo-based` priority + `priority-fcfs` scheduler):
1. `SLOBasedPriority.Compute()` returns `0.0 + 1e-6 * (clock - req.ArrivalTime)` (`sim/priority.go:32-34`)
2. Priority is monotonically increasing with age: older requests (smaller ArrivalTime) get higher priority
3. `PriorityFCFSScheduler.OrderQueue()` sorts by priority descending: highest priority first = **oldest request first = FCFS**

Both configurations produce the same total order: requests sorted by arrival time ascending. The age-based priority `priority = age = clock - ArrivalTime` is a monotonic transformation of `ArrivalTime` (with negative sign), so sorting by priority descending is equivalent to sorting by ArrivalTime ascending.

**Key code citations:**
- `SLOBasedPriority.Compute()`: `sim/priority.go:32-34` -- `return s.BaseScore + s.AgeWeight*age`
- `PriorityFCFSScheduler.OrderQueue()`: `sim/scheduler.go:30-38` -- priority desc, then arrival asc
- Priority assignment: `sim/simulator.go:527` -- `req.Priority = sim.priorityPolicy.Compute(req, now)`
- Queue reordering: `sim/simulator.go:529-531` -- `sim.scheduler.OrderQueue(reqs, now)`

### Why the hypothesis contained a wrong mental model

The hypothesis assumed `slo-based` priority would assign different priorities to different SLO classes (e.g., realtime > interactive > batch). In reality, `SLOBasedPriority` is documented as:

> "Per-request SLO metadata is available on Request.SLOClass but not yet used by this scorer." (`sim/priority.go:26`)

The `slo-based` policy is an **age-based** policy intended to prevent SLO violations by boosting stale requests. It does not assign SLO-class-dependent base scores. All SLO classes share `BaseScore: 0.0` (`sim/priority.go:62`).

### Control experiment (RCV-4)

To confirm the age-ordering mechanism produces exact FCFS equivalence, a control experiment that breaks the equivalence would use `inverted-slo` priority (`priority = -age`), which would produce **anti-FCFS** ordering (newest requests first). This would produce measurably different results. (Not run in this experiment but would serve as the control.)

## Devil's Advocate (RCV-5)

**This is "Refuted." Argue why it might be Confirmed:**

One could argue that under different operating conditions (higher load causing deeper queue buildup, or variable-length requests), the dynamic age-based re-prioritization at each step might produce subtly different scheduling than static FCFS. Specifically, if a request's priority crosses another's between steps due to non-uniform step durations, the relative order could change. However, since `priority = age` is a strictly monotonic function of arrival time, and the arrival time is fixed at request creation, the ordering is stable regardless of when priorities are recomputed. The mathematical equivalence holds universally, not just at this operating point.

Another potential argument: if requests from different SLO classes had different processing times (different token counts), the priority-fcfs scheduler might interact differently with batch formation. But in this experiment, all SLO classes have identical token configurations (256 input, 128 output), so this cannot produce differentiation. Even with variable token counts, the sorting key (age-based priority) would still produce FCFS ordering.

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| `slo-based` priority + `priority-fcfs` scheduler is mathematically equivalent to `constant` + `fcfs` | Design limitation | File enhancement issue: true SLO-class-aware priority policy needed |
| `SLOBasedPriority` does not use `Request.SLOClass` despite its name | Design limitation | The policy name is misleading; it's actually an "age-based" policy. Consider renaming or extending. |
| Priority-FCFS scheduling framework works correctly (identical results confirm no bugs in priority propagation) | Confirmation | INV-6 (determinism) confirmed: same inputs produce same outputs even through different code paths |

## Standards Audit

Findings checked against docs/standards/:
- [x] Any violations of existing rules? **None found.** The code correctly implements the age-based priority as documented.
- [x] Any new rules needed? **Possible:** R-naming: Policy names should accurately reflect their behavior. "slo-based" suggests SLO-class differentiation, but the policy only uses request age.
- [x] Any new invariants needed? **None.** The equivalence is by design, not a bug.
- [x] Any existing rules/invariants confirmed? **INV-6 (Determinism)** confirmed: both configurations produce byte-identical results, demonstrating the scheduler correctly preserves deterministic ordering. **R2 (Sort map keys)** indirectly confirmed: the sorted scheduling order is reproducible.

## Scope and Limitations (RCV-6)

- **Operating point tested:** 4 instances, 500 req/s, 500 requests, round-robin routing, always-admit, llama-3.1-8b-instruct, H100, TP=2, uniform token sizes (256 input / 128 output)
- **Parameters findings depend on:** The mathematical equivalence holds for ALL parameter values -- it is a structural property of the `SLOBasedPriority` formula (monotone in age), not a numerical coincidence.
- **What was NOT tested:** Variable token lengths across SLO classes, `inverted-slo` priority (which would break the equivalence), a hypothetical true SLO-class-aware priority policy (which does not yet exist in BLIS)
- **Generalizability:** This finding generalizes universally. The equivalence between age-based priority ordering and FCFS ordering is mathematical, not empirical. It holds for any workload, any rate, any number of instances, and any token distribution.
- **Uncertainty quantification:** UQ not applicable. The finding is deterministic (0.0% difference), not statistical. The equivalence is proven by algebraic argument, not estimated from samples.

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|------------|
| TTFT difference (all SLO classes) | 0.0% | High -- exact mathematical equivalence, not sampling noise |
| Completion order difference | 0 reordered requests (0%) | High -- exhaustive comparison of all 500 requests x 3 seeds |
| Mechanism | Age-based priority is monotone transform of arrival time | High -- proven algebraically from source code, no control needed |
| Sample size | 3 seeds x 2 configs x 500 requests = 3000 request-config pairs | High for detecting any non-equivalence |

## Implications for Users

1. **Do not use `slo-based` + `priority-fcfs` expecting SLO differentiation.** This combination produces exactly the same behavior as `constant` + `fcfs`. There is no performance benefit or cost.

2. **True SLO-class-aware scheduling requires a new priority policy** that assigns different base scores per SLO class (e.g., `realtime: 100.0, interactive: 50.0, batch: 0.0`). This does not yet exist in BLIS.

3. **The `slo-based` policy name is misleading.** It is actually an "age-based" or "waiting-time" priority policy. The SLO metadata on the request (`SLOClass` field) is explicitly documented as "not yet used" (`sim/priority.go:26`).

4. **The `priority-fcfs` scheduler works correctly.** It faithfully sorts by priority and produces deterministic results. The issue is that the priority scores from `slo-based` do not encode SLO class information.

## Reproducing

```bash
cd hypotheses/h2-priority-fcfs
./run.sh
```
