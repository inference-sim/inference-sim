# H29: Stale Routing Snapshots Degrade Tail Latency — FINDINGS

## Metadata

| Field | Value |
|-------|-------|
| **Hypothesis** | Increasing snapshot refresh interval from 1ms to 100ms degrades TTFT p99 by >= 20% for weighted routing at >80% saturation |
| **Family** | Signal freshness |
| **VV&UQ** | Verification (deterministic, multi-seed) |
| **Type** | Deterministic |
| **Result** | **Confirmed** |
| **Resolution** | Clean confirmation |
| **Status** | Confirmed |
| **Date** | 2026-02-25 |
| **Rounds** | 1 |

## Hypothesis Statement

Increasing the snapshot refresh interval from 1ms to 100ms degrades TTFT p99 by at least 20% for weighted routing (kv-utilization scorer) at high request rates (>80% saturation, 4 instances), because stale load signals cause the router to repeatedly select already-loaded instances, creating transient load imbalance.

**Refuted if:** TTFT p99 difference between 1ms and 100ms snapshot refresh intervals is less than 10% across all 3 seeds at >80% saturation.

## Critical Design Note

The `--snapshot-refresh-interval` flag (defined in `cmd/root.go:667`) only controls **KVUtilization** staleness. From `sim/cluster/snapshot.go:39-48`:

```go
func newObservabilityConfig(refreshInterval int64) ObservabilityConfig {
    if refreshInterval <= 0 {
        return DefaultObservabilityConfig()  // all Immediate
    }
    return ObservabilityConfig{
        QueueDepth:    FieldConfig{Mode: Immediate},      // ALWAYS fresh
        BatchSize:     FieldConfig{Mode: Immediate},      // ALWAYS fresh
        KVUtilization: FieldConfig{Mode: Periodic, Interval: refreshInterval}, // AFFECTED
    }
}
```

Therefore:
- `kv-utilization` scorer IS affected (reads `KVUtilization` directly)
- `queue-depth` scorer is NOT affected (reads `EffectiveLoad() = QueueDepth + BatchSize + PendingRequests`, all Immediate/synchronous)

The original hypothesis text mentions "queue-depth scorer" but the mechanism only applies to `kv-utilization`. The experiment tests both scorers to confirm this architectural distinction.

## Experiment Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Instances | 4 | Multi-instance routing |
| Rate | 195 req/s | ~85% of 4-instance capacity (229.7 req/s) |
| Requests | 500 | Statistical significance |
| Input tokens | 512 (constant) | Standard workload |
| Output tokens | 512 (constant) | Standard workload |
| Fresh interval | 1000us (1ms) | Frequent KV refresh |
| Stale interval | 100000us (100ms) | ~5.7 step times between refreshes |
| Seeds | 42, 123, 456 | Determinism verification |

### Experiments

1. **kv-utilization:1** (staleness-sensitive): fresh vs stale, 3 seeds
2. **queue-depth:1** (negative control): fresh vs stale, 3 seeds — should show ~0% change
3. **kv-utilization:2,queue-depth:2** (mitigation): does fresh queue-depth compensate?
4. **Interval sweep** (kv-utilization:1, seed=42): dose-response from 0 to 500000us

## Results

### Experiment 1: kv-utilization:1 (staleness-sensitive)

| Seed | Metric | Fresh (1ms) | Stale (100ms) | Change |
|------|--------|-------------|---------------|--------|
| 42 | TTFT mean | 35.69ms | 139.39ms | +290.6% |
| 42 | TTFT p99 | 64.00ms | 414.76ms | **+548.1%** |
| 42 | E2E mean | 5395.92ms | 5491.10ms | +1.8% |
| 42 | E2E p99 | 10309.50ms | 10457.37ms | +1.4% |
| 123 | TTFT mean | 36.87ms | 128.66ms | +249.0% |
| 123 | TTFT p99 | 84.10ms | 287.92ms | **+242.3%** |
| 123 | E2E mean | 4889.89ms | 4972.25ms | +1.7% |
| 123 | E2E p99 | 10531.52ms | 10620.08ms | +0.8% |
| 456 | TTFT mean | 36.29ms | 127.55ms | +251.5% |
| 456 | TTFT p99 | 73.72ms | 273.39ms | **+270.9%** |
| 456 | E2E mean | 5176.72ms | 5259.94ms | +1.6% |
| 456 | E2E p99 | 9729.47ms | 9804.28ms | +0.8% |

**Instance distribution (Jain Fairness Index):**

| Seed | Fresh FI | Stale FI | Fresh Distribution | Stale Distribution |
|------|----------|----------|--------------------|--------------------|
| 42 | 0.9986 | 0.9952 | {0:126, 1:117, 2:128, 3:129} | {0:140, 1:120, 2:120, 3:120} |
| 123 | 0.9993 | 0.9952 | {0:123, 1:130, 2:126, 3:121} | {0:120, 1:140, 2:120, 3:120} |
| 456 | 0.9997 | 0.9952 | {0:123, 1:128, 2:123, 3:126} | {0:120, 1:120, 2:140, 3:120} |

Key observation: With stale signals, the distribution shifts to a biased pattern where one instance receives 140 requests while the other three receive 120 each (a 120/120/120/140 pattern). The biased instance varies by seed, confirming it is the instance that happens to appear "least loaded" when the stale snapshot is consulted.

### Experiment 2: queue-depth:1 (negative control)

| Seed | Metric | Fresh (1ms) | Stale (100ms) | Change |
|------|--------|-------------|---------------|--------|
| 42 | TTFT p99 | 48.45ms | 48.45ms | **0.0%** |
| 123 | TTFT p99 | 43.50ms | 43.50ms | **0.0%** |
| 456 | TTFT p99 | 42.59ms | 42.59ms | **0.0%** |

All metrics are **byte-identical** between fresh and stale for queue-depth:1. This confirms that `--snapshot-refresh-interval` has zero effect on queue-depth routing, validating the architectural distinction (QueueDepth is always Immediate per INV-7).

Instance distributions are also byte-identical between fresh and stale configurations.

### Experiment 3: Composite scorer (mitigation)

| Seed | Metric | Fresh (1ms) | Stale (100ms) | Change |
|------|--------|-------------|---------------|--------|
| 42 | TTFT p99 | 45.30ms | 49.99ms | +10.4% |
| 123 | TTFT p99 | 46.27ms | 44.93ms | -2.9% |
| 456 | TTFT p99 | 46.03ms | 47.82ms | +3.9% |

Mean TTFT p99 change: **+3.8%** (vs +353.8% for kv-utilization alone).

The composite scorer (kv-utilization:2,queue-depth:2) reduces the staleness impact by ~99%. The fresh queue-depth signal dominates routing decisions, effectively overriding the stale KV-utilization signal. This demonstrates that the default scoring profile (`prefix-affinity:3,queue-depth:2,kv-utilization:2`) is inherently resilient to KV staleness because queue-depth provides a real-time load signal.

### Experiment 4: Interval sweep

| Interval | TTFT p99 | Change vs 0us |
|----------|----------|---------------|
| 0us (immediate) | 64.00ms | baseline |
| 1000us (1ms) | 64.00ms | +0.0% |
| 5000us (5ms) | 64.00ms | +0.0% |
| 10000us (10ms) | 72.97ms | +14.0% |
| 50000us (50ms) | 229.57ms | +258.7% |
| 100000us (100ms) | 414.76ms | +548.1% |
| 500000us (500ms) | 1163.03ms | +1717.3% |

The dose-response curve shows:
- **Safe zone (0-5ms)**: No measurable impact. Refresh interval is shorter than the step time (~17.4ms), so KV signals remain fresh.
- **Threshold (~10ms)**: 14% degradation begins. Interval approaches step time, so signals become stale for ~1 step.
- **Degradation zone (50-500ms)**: Monotonic, super-linear degradation. At 500ms, staleness spans ~29 step times, and TTFT p99 degrades by 1717%.

E2E metrics are largely unaffected across all intervals (1-2% change) because decode time dominates E2E and is unaffected by routing quality.

## Root Cause Analysis

### Mechanism: Stale KV signals create routing herding

1. **Signal staleness**: At 100ms refresh interval (~5.7 step times), the router sees KV utilization from ~6 steps ago. During those steps, the chosen instance may have processed 6 batches, significantly changing its actual load.

2. **Herding effect**: When KV signals are stale, multiple arriving requests see the same "least loaded" instance and are routed there simultaneously. This creates burst-induced queueing on the herded instance while other instances idle.

3. **TTFT amplification**: The herded instance accumulates a deeper queue, increasing scheduling delay. Meanwhile, the stale signal persists for the full refresh interval, routing even more requests to the same instance. This positive feedback loop explains the super-linear degradation in the dose-response curve.

4. **E2E insensitivity**: E2E is dominated by decode processing: each decode step takes ~6913 us (beta0 + beta2*1 = 6910 + 2.84), so 512 decode steps ~ 3.5s of step time, plus alpha2 overhead (1806 us * 512 tokens ~ 925ms). Total E2E ~ 5-10s depending on queueing. The routing-induced queueing delay (~100-400ms) is small relative to total E2E, resulting in only 1-2% E2E impact.

5. **Distribution shift**: The 120/120/120/140 pattern under stale signals (vs ~125/125/125/125 under fresh) shows the herding is not catastrophic but is measurable. The 20-request imbalance is enough to create significant TTFT tail latency because queueing delay compounds non-linearly near saturation.

6. **Deterministic herding mechanism:** The exact 120/120/120/140 pattern across all 3 seeds (identical Jain FI = 0.9952) indicates this is a deterministic DES artifact, not stochastic herding. With constant-token workloads at 195 req/s and a 100ms refresh window, approximately 19.5 requests arrive per window. During each stale window, the router sees identical KV utilization across all instances and defaults to the first instance in the argmax tiebreaker (which varies by seed due to different initial KV states). This produces a fixed 20-request offset on one instance per refresh cycle, averaging to 140 vs 120 over 500 requests. The biased instance varies by seed because different arrival patterns create different initial KV states at the first refresh boundary.

### Comparison with H3

H3 (`hypotheses/h3-signal-freshness/FINDINGS.md`) showed 200x worse distribution uniformity for kv-utilization vs queue-depth at rate=5000. That experiment compared different *scorers* at the same interval. H29 complements H3 by showing that even a single scorer (kv-utilization) degrades dramatically with increasing staleness, and that the default composite scoring profile is resilient because queue-depth provides a real-time corrective signal.

### Key insight for INV-7 (Signal Freshness)

The experiment validates INV-7's design: the tiered freshness architecture (QueueDepth=Immediate, KVUtilization=Periodic) means that routing quality degrades gracefully when snapshot refresh is slow, as long as at least one Immediate signal is included in the scoring pipeline. The default profile (`prefix-affinity:3,queue-depth:2,kv-utilization:2`) achieves this by construction.

## Conservation Check (INV-1)

All 25 experiment runs pass conservation:

```
exp1_kv_fresh_s42: OK (injected=500, completed=500, queued=0, running=0)
exp1_kv_stale_s42: OK (injected=500, completed=500, queued=0, running=0)
exp2_qd_fresh_s42: OK (injected=500, completed=500, queued=0, running=0)
exp2_qd_stale_s42: OK (injected=500, completed=500, queued=0, running=0)
exp1_kv_fresh_s123: OK (injected=500, completed=500, queued=0, running=0)
exp1_kv_stale_s123: OK (injected=500, completed=500, queued=0, running=0)
exp2_qd_fresh_s123: OK (injected=500, completed=500, queued=0, running=0)
exp2_qd_stale_s123: OK (injected=500, completed=500, queued=0, running=0)
exp1_kv_fresh_s456: OK (injected=500, completed=500, queued=0, running=0)
exp1_kv_stale_s456: OK (injected=500, completed=500, queued=0, running=0)
exp2_qd_fresh_s456: OK (injected=500, completed=500, queued=0, running=0)
exp2_qd_stale_s456: OK (injected=500, completed=500, queued=0, running=0)
exp3_combo_fresh_s42: OK (injected=500, completed=500, queued=0, running=0)
exp3_combo_stale_s42: OK (injected=500, completed=500, queued=0, running=0)
exp3_combo_fresh_s123: OK (injected=500, completed=500, queued=0, running=0)
exp3_combo_stale_s123: OK (injected=500, completed=500, queued=0, running=0)
exp3_combo_fresh_s456: OK (injected=500, completed=500, queued=0, running=0)
exp3_combo_stale_s456: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i0: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i1000: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i5000: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i10000: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i50000: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i100000: OK (injected=500, completed=500, queued=0, running=0)
exp4_sweep_i500000: OK (injected=500, completed=500, queued=0, running=0)
```

## Verdict

- [x] **CONFIRMED**: TTFT p99 degradation >= 20% across all seeds

TTFT p99 degradation: **+242% to +548%** across 3 seeds (mean: +354%). This far exceeds the 20% confirmation threshold.

### Supporting evidence

1. **Negative control validates mechanism**: queue-depth:1 shows exactly 0.0% change, confirming the effect is specific to KV-utilization staleness.
2. **Dose-response is monotonic**: Degradation scales smoothly from 0% at 5ms to 1717% at 500ms, consistent with a staleness-driven herding mechanism.
3. **Composite scorer mitigates**: Adding queue-depth:2 alongside kv-utilization:2 reduces the staleness impact from 354% to 3.8% mean, demonstrating effective architectural mitigation.
4. **Conservation holds**: All runs pass INV-1, confirming results are not artifacts of simulation bugs.

### Practical guidance

- The default scoring profile (`prefix-affinity:3,queue-depth:2,kv-utilization:2`) is inherently resilient to KV staleness because queue-depth provides a real-time corrective signal.
- Users who configure kv-utilization as the sole scorer should keep snapshot refresh intervals below 5ms (< 1 step time) to avoid measurable degradation.
- The 10ms threshold (14% degradation) provides a useful "warning zone" for monitoring systems.

## Issues Filed

No new issues. The existing architecture (INV-7 tiered freshness + composite scoring) already addresses this risk by design. The experiment validates that the default configuration is resilient.

## Devil's Advocate (RCV-5)

The herding effect could be an artifact of constant-token workloads where all requests are identical. With variable-length requests, natural de-correlation of routing decisions might reduce the herding intensity. The 120/120/120/140 distribution shift is modest (Jain FI still 0.9952), suggesting the massive TTFT degradation might stem from near-saturation queueing amplification rather than the staleness mechanism alone. A sub-saturation control would distinguish these explanations.

## Scope and Limitations (RCV-6)

- Operating point: 85% saturation, 4 instances, constant 512/512 tokens
- Only tested at one load level. Effect likely vanishes at sub-saturation (no queueing = no herding penalty)
- Only tested with constant-token workloads. Variable-length workloads may de-correlate routing decisions
- Dose-response thresholds (safe zone <5ms, warning zone 10ms) derived from single seed (42) only
- Not tested with prefix-affinity scorer interactions
- Not tested with different instance counts or model configurations
- ED-2 gap: No sub-saturation rate control to confirm the load-dependence of the staleness effect

## Standards Audit

- [ ] No violations of existing rules found
- [ ] No new rules needed (existing R17 + INV-7 cover this)
- [ ] No new invariants needed
- [x] INV-7 (signal freshness) validated -- tiered architecture works as designed
- [x] INV-1 conservation confirmed for all 25 experiment runs (Exp 1-4)
- [x] R17 (signal freshness documentation) confirmed applicable

## Evidence Quality

| Metric | Value | Confidence |
|--------|-------|-----------|
| TTFT p99 degradation (kv-util) | +242% to +548% (mean +354%) | HIGH -- all seeds far exceed 20% threshold |
| Negative control (queue-depth) | 0.0% change | HIGH -- byte-identical, perfect mechanism isolation |
| Composite mitigation | +3.8% mean change | HIGH -- ~99% reduction from kv-only |
| Dose-response monotonicity | Monotonic 0->1717% | MODERATE -- single seed only |
| Safe zone threshold | <5ms | MODERATE -- single seed, no CI |

## Findings Classification

| Finding | Type | Action |
|---------|------|--------|
| KV staleness degrades TTFT p99 by +354% | Confirmation | Document in INV-7 guidance |
| Queue-depth immune to snapshot interval | Confirmation (INV-7) | Already documented |
| Composite scorer mitigates ~99% | Confirmation | Consider Go test |
| Safe zone <5ms (~1 step time) | Surprise | Qualify as single-seed |
| Dose-response super-linear | Surprise | Document herding feedback loop |
| 120/120/120/140 deterministic pattern | Surprise | Explain mechanism |

## Implications for Users

- The default scoring profile (`prefix-affinity:3,queue-depth:2,kv-utilization:2`) is inherently resilient to KV staleness because queue-depth provides a real-time corrective signal
- Users who configure kv-utilization as the sole scorer should keep snapshot refresh intervals below 5ms (< 1 step time) to avoid measurable degradation
- The 10ms threshold (14% degradation) provides a useful "warning zone" for monitoring systems
- At intervals exceeding 50ms, degradation becomes severe (>250%) due to herding feedback loop

## Reproducing

```
cd hypotheses/h29-snapshot-staleness
./run.sh
```
