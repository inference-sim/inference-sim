# Scheduling & Priority

Routing decides **which instance** receives a request. Scheduling decides **what order** requests are processed within an instance. These are independent policy axes — you can combine any router configuration with any scheduler.

Each instance's engine orders its requests by its pool's `engine.scheduling_policy` in the scenario, as vLLM's `--scheduling-policy` does: `fcfs` (the default) or `priority`. No flag or policy-bundle key sets it; `--scheduler` and `--preemption-policy` were removed in v0.18.0. On vLLM v0.32.0 priority scheduling (`scheduling_policy: priority`, SLO priority overrides) is refused for now: the host does not yet carry request priority across the engine contract. The priority sections below describe what `priority` selects.

```yaml
# In each pool of the scenario's deployment:
    engine:
      scheduling_policy: fcfs   # or priority (refused on 0.32.0 for now)
```

## How Priority Works

!!! note "Not yet available on 0.32.0"
    Priority scheduling is refused on 0.32.0 for now.

Request priority is **static** — set once when the request enters the instance, and never changed.

At enqueue time (`EnqueueRequest` and `EnqueueDecodeSubRequest`), the simulator converts the request's `SLOClass` to a vLLM-convention priority value via `SLOPriorityMap.InvertForVLLM(SLOClass)`:

```
critical   → Priority 0.0  (most urgent)
standard   → Priority 1.0
batch      → Priority 5.0
sheddable  → Priority 6.0
background → Priority 7.0  (least urgent)
```

This follows **vLLM's convention: lower integer = more urgent**. The llm-d router (admission, flow control, routing) uses the llm-d convention (higher = more urgent), with priorities from its InferenceObjectives (see [Admission Control](admission.md)); the inversion happens at the cluster→instance dispatch boundary.

SLO class is set in your workload spec:

```yaml
clients:
  - slo_class: critical
    rate: 10
  - slo_class: standard
    rate: 50
```

Custom priority values can be overridden via the policy bundle:

```yaml
priority:
  slo_priorities:
    batch: 0   # make batch non-sheddable
```

## Scheduling Policies

Each policy orders the wait queue every step, before batch formation, through the `InstanceScheduler` interface's `OrderQueue` method.

| `scheduling_policy` | Queue order | Preemption victim | Notes |
|---------------------|-------------|-------------------|-------|
| `fcfs` | First-Come-First-Served. No reordering: requests are processed in arrival order. | The tail of the running batch. | Default. Fair and predictable. |
| `priority` | Sort by priority **ascending** (lower value = more urgent, vLLM convention), then by arrival time ascending within the same priority. Ties broken by request ID for determinism. | The least-urgent running request. | Useful when `SLOClass` is set in the workload spec. Without SLO classes, all requests get Priority=1.0 (standard) and this degrades to FCFS by arrival tiebreak. Refused on 0.32.0 for now. |

Both policies use `sort.SliceStable` for deterministic ordering (INV-6).

## How Scheduling and Priority Interact

The scheduling policy and priority are composed at each simulation step:

1. **Priority is already set** (static — set at enqueue via `SLOPriorityMap.InvertForVLLM`, not recomputed per step).
2. **Queue reordering**: Call `InstanceScheduler.OrderQueue()` on the wait queue.
3. **Batch formation**: Dequeue requests from the front of the reordered queue into the running batch.

Common combinations:

| Combination | Effective Behavior |
|-------------|-------------------|
| `priority` + mixed SLO classes | Critical requests scheduled first, background last. |
| `priority` + uniform SLO class | All priorities equal — degrades to FCFS by arrival time. |
| `fcfs` + any SLO class | FCFS by arrival time (priority computed but reordering skipped). |

### Preemption and Re-enqueueing

BLIS models vLLM's two-queue architecture (WaitQ + RunningBatch). When a request is preempted from the running batch due to KV cache pressure:

- The request is placed at the **front** of the WaitQ (not the back).
- Its progress is reset to zero (recompute mode, matching vLLM's recompute preemption).
- On the next step, the scheduler reorders the full queue including the preempted request.

This means preempted requests get implicit priority over fresh arrivals in FCFS mode. With `priority` and mixed SLO classes, the preempted request's static priority determines its position relative to other waiting requests.

Under `scheduling_policy: fcfs` (the default), the tail of the running batch is evicted. Under `scheduling_policy: priority`, which is refused on 0.32.0 for now, the least-urgent running request is evicted — the one with the **highest Priority value** (vLLM convention: background=7 is least urgent and evicted first). Among equal-priority requests, the most recently arrived is evicted first. This matches vLLM's `--scheduling-policy priority` preemption behavior (vLLM v0.31.0 `vllm/v1/core/sched/scheduler.py:765,777`). One field sets both the queue order and the preemption victim, as in vLLM: `priority` is BLIS's `priority-fcfs` queueing with `priority` preemption, and `fcfs` is `fcfs` for both. In a prefill/decode deployment each pool takes its own engine's policy.

## When to Use Which

| Workload | Recommended Configuration | Why |
|----------|--------------------------|-----|
| Uniform traffic, no SLO differentiation | `scheduling_policy: fcfs` (default) | No reordering needed. All requests are equivalent. |
| Mixed SLO classes (critical vs background) | `scheduling_policy: priority` with `slo_class` in the workload spec (refused on 0.32.0 for now) | Critical requests get Priority=0, scheduled before background (Priority=7). |
| Low load (< ~10 req/s) | Either | Batch sizes are small enough that both policies pick the same requests. At low load, both produce equivalent results within ~5%. |

## Further Reading

- [Routing](routing.md) — the upstream decision (which instance)
- [Cluster Simulation](cluster.md) — the full request pipeline from arrival to completion
- [Core Engine: Scheduling Policies](../concepts/core-engine.md#scheduling-policies) — implementation details and DES mechanics
