# H14: Pathological Templates — Anomaly Detection Validation

**Status:** Partially confirmed
**Tier:** 2 (high diagnostic value)
**Type:** Statistical/Dominance
**Date:** 2026-02-20

## Hypothesis

> The pathological policies (`always-busiest`, `reverse-priority`, `inverted-slo`) exist specifically to test anomaly detection. `always-busiest` should produce HOL blocking (routes to most loaded instance). `reverse-priority` should produce priority inversions. If anomaly counters don't detect these, the detection logic has a bug.

## Experiment Design

**Classification:** Statistical/Dominance — pathological should be strictly worse on every dimension.

**Configurations compared:**
- A (Normal): `--routing-policy least-loaded --scheduler priority-fcfs --priority-policy slo-based`
- B (Pathological): `--routing-policy always-busiest --scheduler reverse-priority --priority-policy inverted-slo`

**Controlled variables:** 4 instances, rate=2000, 500 requests, mixed-SLO workload (33% realtime, 34% interactive, 33% batch), `--trace-level decisions --summarize-trace`

**Varied variable:** Policy configuration (normal vs pathological)

**Seeds:** 42, 123, 456 (CLI `--seed`). Note: all seeds produce identical output due to fixed workload-spec YAML seed (known issue #284).

**Preconditions verified:**
- `always-busiest`, `reverse-priority`, `inverted-slo` are valid CLI values (confirmed by running each)
- `detectPriorityInversions` is NOT suppressed for `inverted-slo` (only suppressed for `constant` and `""` per `metrics.go:171-173`)

## Results

### Experiment 1: Normal vs Pathological (3 seeds)

All seeds produce byte-identical output (ED-4: YAML seed fixed, CLI `--seed` has no effect on workload-spec generation).

| Config | TTFT Mean | TTFT P99 | HOL | Inversions | StdDev | Distribution |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| Normal | 489.2ms | 1,078.0ms | 0 | 7,463 | 1.9 | [125, 127, 126, 122] |
| Pathological | 2,256.4ms | 4,848.4ms | 0 | 16,138 | 216.5 | [500, 0, 0, 0] |
| **Effect** | **4.6x worse** | **4.5x worse** | **—** | **2.2x more** | **114x worse** | |

### Experiment 2: Decomposed (seed 42)

Isolates routing vs scheduling contribution by varying one dimension at a time.

| Configuration | TTFT P99 | HOL | Inversions | StdDev | Distribution |
|:---|:---:|:---:|:---:|:---:|:---|
| Normal (all correct) | 1,078.0ms | 0 | 7,463 | 1.9 | [125, 127, 126, 122] |
| Pathological routing only | 4,848.4ms | 0 | 16,138 | 216.5 | [500, 0, 0, 0] |
| Pathological scheduling only | 1,078.0ms | 0 | 7,463 | 1.9 | [125, 127, 126, 122] |
| All pathological | 4,848.4ms | 0 | 16,138 | 216.5 | [500, 0, 0, 0] |

**Key observation:** Pathological scheduling (reverse-priority + inverted-slo) produces output **identical to normal**. The entire pathological effect comes from `always-busiest` routing.

## Root Cause Analysis

### BUG 1: HOL Blocking Detector Blind Spot (severity: medium)

**Location:** `sim/cluster/metrics.go:212-247`

`always-busiest` routes ALL 500 requests to `instance_0`. The other 3 instances receive 0 requests, so their `NumWaitQRequests` slice is empty. The detector at line 221 skips instances with empty samples:

```go
if len(m.NumWaitQRequests) == 0 {
    continue  // instances_1,2,3 are skipped
}
```

This leaves `avgDepths` with only 1 entry (instance_0). The guard at line 233 then exits:

```go
if len(avgDepths) < 2 {
    return 0 // need at least 2 instances with samples
}
```

**Result:** The most extreme HOL blocking scenario possible (all traffic on 1 instance) produces HOL=0 because the detector can't compare against empty instances.

**Fix:** Include zero-traffic instances in the comparison. An instance receiving 0 requests while a sibling receives 500 IS the definition of HOL blocking. Replace the `len(m.NumWaitQRequests) == 0` skip with a zero-value default:

```go
for _, m := range perInstance {
    avg := 0.0
    if len(m.NumWaitQRequests) > 0 {
        sum := 0
        for _, d := range m.NumWaitQRequests {
            sum += d
        }
        avg = float64(sum) / float64(len(m.NumWaitQRequests))
    }
    avgDepths = append(avgDepths, avg)
    totalAvg += avg
}
```

### BUG 2: Priority Inversion Detector False Positives (severity: medium)

**Location:** `sim/cluster/metrics.go:164-207`

The **normal** configuration produces 7,463 priority inversions despite using correct `slo-based` priority + `priority-fcfs` scheduling. The detector uses a simple heuristic: count all pairs where an earlier-arriving request has E2E > 2× a later-arriving request's E2E.

For the mixed-SLO workload, this heuristic conflates two effects:
1. **Priority unfairness** (what we want to detect): lower-priority requests completing before higher-priority ones
2. **Workload heterogeneity** (false positives): a batch request (mean 1024 input, 512 output tokens) naturally has ~10-20× longer E2E than a realtime request (mean 64 input, 32 output tokens). An earlier batch request will always have E2E > 2× a later realtime request — not because of unfair scheduling, but because batch requests are intrinsically larger.

The 2× threshold is far too lenient for mixed-SLO workloads with 16× variation in request size. The detector should either:
- Compare within the same SLO class only (apples to apples)
- Use a per-class-normalized threshold
- Compare priority-weighted E2E instead of raw E2E

### BUG 3: Double Inversion Cancellation — `inverted-slo` + `reverse-priority` ≡ Normal (severity: medium)

`reverse-priority` + `inverted-slo` produce output **byte-identical** to `priority-fcfs` + `slo-based` at ALL configurations tested — including 1 instance where queues are deep. The root cause is NOT queue depth. It's a mathematical cancellation:

**Normal scheduling order:**
- `SLOBasedPriority`: priority = `0 + 1e-6 × age` → older requests get HIGHER priority
- `PriorityFCFS`: sorts HIGH priority first → **older requests dequeued first** (= FCFS)

**"Pathological" scheduling order:**
- `InvertedSLO`: priority = `0 - 1e-6 × age` → older requests get LOWER priority
- `ReversePriority`: sorts LOW priority first → **older requests dequeued first** (= FCFS)

The two inversions cancel: `(-age, ascending)` produces the same ordering as `(+age, descending)`. The "pathological" combination is mathematically equivalent to normal scheduling.

**Experiment 3 confirms this:** At 1 instance (deep queues), double inversion produces 1.00× on all metrics vs normal. But **single** inversions produce a measurable effect:

| Configuration | TTFT P99 | E2E P99 | Inversions |
|:---|:---:|:---:|:---:|
| Normal (slo-based + priority-fcfs) | 4,848ms | 18,573ms | 16,138 |
| Double inversion (inverted + reverse) | 4,848ms | 18,573ms | 16,138 |
| Inverted priority only (inverted + priority-fcfs) | 5,166ms | 18,212ms | 37,497 |
| Reverse scheduler only (slo-based + reverse) | 5,166ms | 18,212ms | 37,498 |

Single inversions show +7% TTFT p99 and 2.3× more inversions. Both single inversions produce near-identical results (both are a single flip of the ordering).

**Fix:** The research.md H14 spec and H24 spec should use single inversions, not double. For example, `inverted-slo` + `priority-fcfs` (inverted priority with correct scheduler) is actually pathological, whereas the intended `inverted-slo` + `reverse-priority` is accidentally equivalent to normal.

### FINDING 4: CLI `--seed` Doesn't Affect Workload-Spec Runs

All 3 seeds (42, 123, 456) produce byte-identical output. This confirms issue #284: the workload-spec YAML has its own `seed:` field that controls request generation, and `--seed` only controls simulation-level RNG. For workload-spec experiments, the simulation RNG is effectively unused because all randomness is in the pre-generated request stream.

**Impact on H14:** We effectively have 1 seed, not 3. The results are deterministic and reproducible, but the multi-seed statistical rigor is not achieved. This affects ALL workload-spec hypotheses (H5, H16, H18, and any experiment using `--workload-spec`).

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| `always-busiest` produces 4.5× worse TTFT | Confirmation | Documented here |
| `always-busiest` routes ALL traffic to 1 instance (stddev=216.5) | Confirmation | Documented here |
| HOL blocking detector returns 0 for extreme single-instance concentration | Bug discovery | #291 |
| Priority inversion detector false-positive rate: 7,463 for correct scheduling | Bug discovery | #292 |
| `inverted-slo` + `reverse-priority` double inversion cancels out to FCFS-equivalent | Bug discovery | #295 |
| `--seed` has no effect on workload-spec runs | Confirmation of #284 | Already tracked in #284 |

## Standards Audit

Findings checked against docs/standards/:

- [x] **Any violations of existing rules?** No rule violations in experiment code. However, the HOL blocking detector bug could be classified as a violation of R1 (no silent data loss) — the detector silently returns 0 instead of detecting extreme imbalance.
- [x] **Any new rules needed?** Consider: "Anomaly detectors must handle the degenerate case where one instance receives all traffic" — subsumed under the HOL blocking fix.
- [x] **Any new invariants needed?** None.
- [x] **Any existing rules/invariants confirmed?** INV-6 (determinism) confirmed — byte-identical output across seeds. INV-1 (conservation) confirmed — 500 injected = 500 completed for all configs.

## Implications for Users

1. **`always-busiest` works as intended** for load imbalance testing — it produces extreme concentration (all traffic on 1 instance) and 4.5× worse TTFT.

2. **Do not rely on HOL blocking counter for extreme imbalance** — the detector has a blind spot when all traffic goes to one instance. A nonzero HOL count means real blocking, but a zero count does NOT mean balanced distribution. Always check the target distribution directly.

3. **Do not rely on priority inversion counter for mixed-SLO workloads** — the current heuristic produces thousands of false positives when request sizes vary significantly. A high inversion count may reflect workload heterogeneity, not scheduling unfairness.

4. **Scheduling pathological templates require high queue depth** — `reverse-priority` and `inverted-slo` have no observable effect when queues are short. Use rate >> 5000 with 1-2 instances to observe scheduling effects.

5. **Multi-seed experiments with `--workload-spec` require varying the YAML seed** (issue #284) — CLI `--seed` alone produces identical results.

## Reproducing

```bash
cd hypotheses/h14-pathological-templates
./run.sh           # ~1 minute, all experiments
./run.sh --rebuild # rebuild binary first
```

Requires: Go 1.24+, Python 3
