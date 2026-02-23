# H26: Admission Latency Causal Ordering — FINDINGS

## Metadata

| Field | Value |
|-------|-------|
| **Hypothesis** | Adding admission latency should delay E2E by exactly that amount under low load |
| **Family** | Structural model |
| **VV&UQ** | Verification (deterministic) |
| **Type** | Deterministic |
| **Result** | **Confirmed** |
| **Round** | 1 |

## Hypothesis Statement

Under low load (no queuing), configuring `--admission-latency L` should increase both TTFT and E2E by exactly `L` microseconds. This validates the cluster event pipeline's causal ordering: Arrival -> Admission (+latency) -> Routing -> Queue -> Batch -> Step.

## Precondition Analysis

**Q: Is TTFT measured from request arrival or from queue entry?**

Both TTFT and E2E are measured from `req.ArrivalTime`, which is set at workload generation time (before the admission pipeline). This means both metrics include admission latency.

Code trace:
- `sim/cluster/workload.go:54`: `ArrivalTime: currentTime` — set at generation
- `sim/cluster/cluster.go:120`: `ClusterArrivalEvent{time: req.ArrivalTime}` — arrival event at original time
- `sim/cluster/cluster_event.go:89`: `time: e.time + cs.admissionLatency` — admission delays by L
- `sim/cluster/cluster_event.go:130`: `time: e.time + cs.routingLatency` — routing delays further
- `sim/cluster/cluster_event.go:186`: `inst.InjectRequestOnline(e.request, e.time)` — injects at delayed time
- `sim/simulator.go:564`: `FirstTokenTime = now + currStepAdvance + OutputTokenProcessingTime() - req.ArrivalTime` — TTFT from original ArrivalTime
- `sim/simulator.go:499-500`: `lat = req.FirstTokenTime + itlSum` — E2E = TTFT + decode time
- `sim/simulator.go:460`: `SchedulingDelay = now + scheduledDelay - req.ArrivalTime` — also from original ArrivalTime

**Conclusion**: `ArrivalTime` is never modified after generation. All metrics (TTFT, E2E, SchedulingDelay) are computed relative to the original arrival time. Admission latency creates a gap between arrival and instance injection that is captured in all three metrics.

## Experiment Design

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Rate | 10 req/s | Low load, avoids queuing confounds |
| Requests | 50 | Sufficient for mean comparison |
| Instances | 4 | Multi-instance cluster mode |
| Input tokens | 128 (constant) | Eliminates variance |
| Output tokens | 32 (constant) | Eliminates variance |
| Routing | least-loaded | Balanced distribution |
| Seed | 42 | Deterministic |
| Arrival | Poisson | Standard arrival process |

**Configs:**
- A: `--admission-latency 0` (baseline)
- B: `--admission-latency 10000` (10ms)
- C: `--admission-latency 50000` (50ms, linearity check)

**Control variable**: Only `--admission-latency` differs. All other parameters identical.

## Results

### Aggregate Metrics

| Config | TTFT mean (ms) | E2E mean (ms) | Completed |
|--------|----------------|---------------|-----------|
| A (latency=0) | 13.3742 | 292.5819 | 50 |
| B (latency=10ms) | 23.3742 | 302.5819 | 50 |
| C (latency=50ms) | 63.3742 | 342.5819 | 50 |

### Deltas

| Config | TTFT delta (ms) | E2E delta (ms) | Expected (ms) | TTFT match | E2E match |
|--------|-----------------|----------------|---------------|------------|-----------|
| B (10ms) | 10.0000 | 10.0000 | 10.0000 | PASS | PASS |
| C (50ms) | 50.0000 | 50.0000 | 50.0000 | PASS | PASS |

### Scheduling Delay (per-request mean, in ticks/us)

| Config | Mean sched delay (us) | Delta (us) | Expected (us) | Match |
|--------|----------------------|------------|----------------|-------|
| A | 2396.9 | — | — | — |
| B | 12396.9 | 10000.0 | 10000 | PASS |
| C | 52396.9 | 50000.0 | 50000 | PASS |

### Linearity Check

E2E delta ratio C/B = 50.0000 / 10.0000 = **5.0000** (expected: 5.0). PASS.

## Root Cause Analysis

The admission latency is injected as a timestamp offset in the cluster event pipeline:

1. `ClusterArrivalEvent.Execute()` (`cluster_event.go:85-93`): Creates `AdmissionDecisionEvent` with `time = arrival_time + admissionLatency`
2. `AdmissionDecisionEvent.Execute()` (`cluster_event.go:109-135`): If admitted, creates `RoutingDecisionEvent` with `time = admission_time + routingLatency`
3. `RoutingDecisionEvent.Execute()` (`cluster_event.go:148-193`): Calls `InjectRequestOnline(req, e.time)` — request enters instance at `arrival_time + admissionLatency + routingLatency`

Since `req.ArrivalTime` is set at generation time and never modified, the time gap between arrival and instance injection is exactly `admissionLatency + routingLatency`. All metrics (TTFT, E2E, SchedulingDelay) subtract `ArrivalTime` from the measurement timestamp, so the admission latency appears as an exact additive offset.

**Why exact (not approximate)?** Under low load with constant token lengths:
- No queuing interference (requests are processed immediately)
- Constant service time (same tokens for every request)
- Deterministic seed ensures identical arrival patterns
- Only the admission latency offset varies between configs

## Standards Audit

| Standard | Status |
|----------|--------|
| ED-1 (clear hypothesis) | Met — behavioral prediction with metric and direction |
| ED-2 (control variable) | Met — only admission-latency varies |
| ED-3 (parameter calibration) | Met — low rate avoids queuing |
| ED-4 (family classification) | Met — structural model |
| ED-5 (reproducibility) | Met — deterministic with seed=42 |
| ED-6 (config diff) | N/A — no reference experiment |
| RCV-1 (code citation) | Met — all causal claims cite file:line |
| RCV-2 (first principles) | Met — expected delta = admission latency exactly |
| RCV-3 (mechanism + direction) | Met — timestamp offset mechanism explains additive delta |
| RCV-4 (control experiment) | Met — Config A (latency=0) is the control |

## Findings Classification

| Finding | Type | Resolution |
|---------|------|------------|
| Admission latency adds exact offset to TTFT, E2E, and SchedulingDelay | Confirmation | Clean confirmation — event pipeline causal ordering works as designed |
| Both TTFT and E2E include admission delay (measured from ArrivalTime, not queue entry) | Confirmation | Validates measurement point: ArrivalTime = original generation time |
| Linearity holds (50ms/10ms ratio = 5.0) | Confirmation | Linear relationship: no overhead or interaction effects |

## Evidence Quality

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Precision | Exact | Delta matches to 4+ decimal places |
| Sample size | 50 per config | Sufficient for deterministic verification |
| Controls | 3 configs | Baseline + 2 treatment levels + linearity check |
| Confounds | None identified | Low load, constant tokens, deterministic seed |
| Reproducibility | Deterministic | `./run.sh` reproduces exactly |

## Promotion Assessment

This is a deterministic hypothesis with exact verification. Suitable for promotion to Go test suite:
- Test: run cluster sim with admission-latency=0 and admission-latency=L, verify E2E delta = L/1000 ms
- Would provide regression protection for the event pipeline's causal ordering

## Issues to File

None required. This is a clean confirmation with no bugs, design limitations, or surprises discovered.
