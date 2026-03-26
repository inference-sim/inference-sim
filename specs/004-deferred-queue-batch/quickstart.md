# Quickstart: Deferred Queue for Batch and Background Requests

## What This Feature Does

Batch and Background SLO-class requests are automatically held in a deferred queue when the cluster is busy with real-time traffic. They are promoted and processed normally as soon as the cluster becomes idle. This enables "spot/batch" workload semantics: lower-priority work eventually completes without interfering with real-time traffic.

## Zero Configuration Required

The deferred queue activates automatically based on `SLOClass`. No new YAML fields or CLI flags are needed.

## Scenario 1: Mixed Workload — Batch Waits, Real-Time Proceeds

```yaml
# workload.yaml
requests:
  - id: critical_0
    slo_class: critical
    arrival_time_us: 0
  - id: batch_0
    slo_class: batch
    arrival_time_us: 100
  - id: critical_1
    slo_class: critical
    arrival_time_us: 200
```

```bash
./blis run --model qwen/qwen3-14b --num-instances 1
```

Expected behavior:
- `critical_0` arrives at t=0, admitted immediately.
- `batch_0` arrives at t=100; cluster is busy → deferred.
- `critical_1` arrives at t=200, admitted immediately.
- After `critical_0` and `critical_1` complete, cluster becomes idle → `batch_0` is promoted and processed.
- All three requests appear as completed in metrics output.

## Scenario 2: Deferred Requests at Horizon

If the simulation horizon is reached before the cluster becomes idle and deferred requests are promoted:

```bash
./blis run --model qwen/qwen3-14b --num-instances 1 --horizon 1000
```

Sample output (Anomaly Counters):
```
=== Anomaly Counters ===
Deferred (horizon-interrupted): 5
```

These 5 requests are accounted for in the conservation equation:
`injected == completed + deferred_horizon_interrupted`

## Scenario 3: Background Requests with Tier-Shed Enabled

When `tier-shed` admission is active and the cluster is overloaded:

```yaml
deployment:
  admission_policy: tier-shed
  tier_shed_threshold: 5
  tier_shed_min_priority: 3
```

- Background requests (priority 0): deferred when cluster is busy (never shed by tier-shed policy).
- Sheddable requests (priority 2): shed by tier-shed under overload.
- The pre-admission deferral intercept fires BEFORE `tier-shed` checks — Background requests bypass tier-shed entirely when the cluster is busy.

## Verifying Request Conservation

Run with `--json` output and check:

```bash
./blis run --model qwen/qwen3-14b --json | jq '.deferred_horizon_interrupted'
```

The sum `completed + shed + dropped + deferred_horizon_interrupted` must equal `injected`.

## What Appears in the Output

A new line in the "=== Anomaly Counters ===" block (only printed when non-zero):

```
Deferred (horizon-interrupted): N
```

This count is also available in the JSON metrics output as `deferred_horizon_interrupted`.
