# Scheduling & Priority

Routing decides **which instance** receives a request. Scheduling decides **what order** requests are processed within an instance. These are independent policy axes — you can combine any routing policy with any scheduler.

```bash
# Priority-FCFS scheduling with SLO-class-based priority in a 4-instance cluster
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500 \
  --scheduler priority-fcfs
# SLO class is set in the workload spec (slo_class: critical/standard/batch/etc.)
```

## How Priority Works

Request priority is **static** — set once when the request enters the instance, and never changed.

At enqueue time (`EnqueueRequest` and `EnqueueDecodeSubRequest`), the simulator converts the request's `SLOClass` to a vLLM-convention priority value via `SLOPriorityMap.InvertForVLLM(SLOClass)`:

```
critical   → Priority 0.0  (most urgent)
standard   → Priority 1.0
batch      → Priority 5.0
sheddable  → Priority 6.0
background → Priority 7.0  (least urgent)
```

This follows **vLLM's convention: lower integer = more urgent**. The cluster layer (admission, routing, gateway queue) continues to use the llm-d convention (higher = more urgent); the inversion happens at the cluster→instance dispatch boundary.

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
admission:
  slo_priorities:
    batch: 0   # make batch non-sheddable
```

## Available Schedulers

Each scheduler implements the `InstanceScheduler` interface: a single `OrderQueue` method called every step to reorder the wait queue before batch formation.

| Scheduler | Flag Value | Strategy | Notes |
|-----------|-----------|----------|-------|
| **FCFS** | `--scheduler fcfs` | First-Come-First-Served. No reordering — requests are processed in arrival order. | Default. Fair and predictable. |
| **Priority-FCFS** | `--scheduler priority-fcfs` | Sort by priority **ascending** (lower value = more urgent, vLLM convention), then by arrival time ascending within the same priority. Ties broken by request ID for determinism. | Useful when `SLOClass` is set in the workload spec. Without SLO classes, all requests get Priority=1.0 (standard) and this degrades to FCFS by arrival tiebreak. |
| **SJF** | `--scheduler sjf` | Shortest Job First. Sort by input token count ascending, then by arrival time, then by ID. | Optimizes TTFT for short requests but can starve long ones under sustained load. Ignores `Request.Priority` entirely. |
| **Reverse-priority** | `--scheduler reverse-priority` | Sort by priority **descending** (highest value = least urgent scheduled first). | Pathological template for testing only — deliberately causes priority inversions. |

All schedulers use `sort.SliceStable` for deterministic ordering (INV-6).

## How Scheduling and Priority Interact

The scheduler and priority are composed at each simulation step:

1. **Priority is already set** (static — set at enqueue via `SLOPriorityMap.InvertForVLLM`, not recomputed per step).
2. **Queue reordering**: Call `InstanceScheduler.OrderQueue()` on the wait queue.
3. **Batch formation**: Dequeue requests from the front of the reordered queue into the running batch.

Common combinations:

| Combination | Effective Behavior |
|-------------|-------------------|
| `priority-fcfs` + mixed SLO classes | Critical requests scheduled first, background last. |
| `priority-fcfs` + uniform SLO class | All priorities equal — degrades to FCFS by arrival time. |
| `sjf` + any SLO class | SJF by input length (priority ignored). |
| `fcfs` + any SLO class | FCFS by arrival time (priority computed but reordering skipped). |

### Preemption and Re-enqueueing

BLIS models vLLM's two-queue architecture (WaitQ + RunningBatch). When a request is preempted from the running batch due to KV cache pressure:

- The request is placed at the **front** of the WaitQ (not the back).
- Its progress is reset to zero (recompute mode, matching vLLM's recompute preemption).
- On the next step, the scheduler reorders the full queue including the preempted request.

This means preempted requests get implicit priority over fresh arrivals in FCFS mode. With `priority-fcfs` and mixed SLO classes, the preempted request's static priority determines its position relative to other waiting requests.

By default (`--preemption-policy fcfs`), the tail of the running batch is evicted. When `--preemption-policy priority` is set, the least-urgent running request is evicted — the one with the **highest Priority value** (vLLM convention: background=7 is least urgent and evicted first). Among equal-priority requests, the most recently arrived is evicted first. This matches vLLM's `--scheduling-policy priority` preemption behavior (`scheduler.py:1086`).

## When to Use Which

| Workload | Recommended Configuration | Why |
|----------|--------------------------|-----|
| Uniform traffic, no SLO differentiation | `--scheduler fcfs` (default) | No reordering needed. All requests are equivalent. |
| Mixed SLO classes (critical vs background) | `--scheduler priority-fcfs` with `slo_class` in workload spec | Critical requests get Priority=0, scheduled before background (Priority=7). |
| Latency-sensitive short requests | `--scheduler sjf` | Short prompts get processed first. Watch for starvation of long requests under sustained load. |
| Low load (< ~10 req/s) | Any | Batch sizes are small enough that all schedulers pick the same requests. At low load, all schedulers produce equivalent results within ~5%. |

!!! tip "SJF starvation risk"
    Under sustained high load, SJF can indefinitely delay long-prompt requests as short ones keep arriving. BLIS does not currently implement aging or starvation guards for SJF. If your workload has a mix of short and long prompts at high utilization, prefer `--scheduler priority-fcfs` with SLO classes instead.

## Further Reading

- [Routing Policies](routing.md) — the upstream decision (which instance)
- [Cluster Simulation](cluster.md) — the full request pipeline from arrival to completion
- [Core Engine: Scheduling Policies](../concepts/core-engine.md#scheduling-policies) — implementation details and DES mechanics
