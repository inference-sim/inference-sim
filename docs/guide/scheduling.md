# Scheduling & Priority

Routing decides **which instance** receives a request. Scheduling decides **what order** requests are processed within an instance. These are independent policy axes -- you can combine any routing policy with any scheduler and any priority policy.

```bash
# Priority-FCFS scheduling with age-based priority in a 4-instance cluster
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 100 --num-requests 500 \
  --scheduler priority-fcfs --priority-policy slo-based
```

## Available Schedulers

Each scheduler implements the `InstanceScheduler` interface: a single `OrderQueue` method called every step to reorder the wait queue before batch formation.

| Scheduler | Flag Value | Strategy | Notes |
|-----------|-----------|----------|-------|
| **FCFS** | `--scheduler fcfs` | First-Come-First-Served. No reordering -- requests are processed in arrival order. | Default. Fair and predictable. |
| **Priority-FCFS** | `--scheduler priority-fcfs` | Sort by priority descending, then by arrival time ascending within the same priority. Ties broken by request ID for determinism. | Requires a non-constant priority policy to be useful. With `--priority-policy constant`, all scores are equal and this degrades to FCFS. |
| **SJF** | `--scheduler sjf` | Shortest Job First. Sort by input token count ascending, then by arrival time, then by ID. | Optimizes TTFT for short requests but can starve long ones under sustained load. Ignores priority scores entirely. |
| **Reverse-priority** | `--scheduler reverse-priority` | Lowest priority first. Sort by priority ascending, then by arrival time, then by ID. | Pathological template for testing only. Causes priority inversions by design. |

All schedulers use `sort.SliceStable` for deterministic ordering (INV-6).

## Priority Policies

Each priority policy implements the `PriorityPolicy` interface: a single `Compute` method that returns a `float64` score for a request at a given clock tick. Higher scores mean higher priority.

| Policy | Flag Value | Formula | Notes |
|--------|-----------|---------|-------|
| **Constant** | `--priority-policy constant` | Returns `0.0` for all requests. | Default. No differentiation -- all requests have equal priority. |
| **SLO-based** | `--priority-policy slo-based` | `BaseScore + AgeWeight * (clock - arrival_time)` | Favors older requests. With default `AgeWeight=1e-6`, a request waiting 1 second (1,000,000 ticks) gets +1.0 priority. Despite the name, does NOT currently use per-request SLO metadata (`Request.SLOClass` is available but unused by this policy). |
| **Inverted-SLO** | `--priority-policy inverted-slo` | `BaseScore - AgeWeight * (clock - arrival_time)` | Starves older requests -- newer requests get higher priority. Pathological template for testing only. |

Priority scores are recomputed every step before the scheduler orders the queue. This means SLO-based priority naturally ages: a request that has been waiting longer will have a higher score at the next step.

## How Scheduling and Priority Interact

The scheduler and priority policy are independent modules that compose at step time. The simulator calls them in sequence:

1. **Priority assignment**: For each queued request, call `PriorityPolicy.Compute()` and store the result on `Request.Priority`.
2. **Queue reordering**: Call `InstanceScheduler.OrderQueue()` on the wait queue.
3. **Batch formation**: Dequeue requests from the front of the reordered queue into the running batch.

This means certain combinations are degenerate:

| Combination | Effective Behavior |
|-------------|-------------------|
| `priority-fcfs` + `constant` | FCFS (all priority scores are 0.0, so the descending sort changes nothing) |
| `sjf` + any priority policy | SJF (priority scores are computed but ignored -- SJF sorts by input token count) |
| `fcfs` + `slo-based` | FCFS (priority scores are computed but FCFS does not reorder) |
| **`priority-fcfs` + `slo-based`** | **The useful combination**: older requests float to the top of the queue |

### Preemption and Re-enqueueing

BLIS models vLLM's two-queue architecture (WaitQ + RunningBatch), simplifying vLLM's three-queue model (waiting / running / swapped). When a request is preempted from the running batch due to KV cache pressure:

- The request is placed at the **front** of the WaitQ (not the back).
- Its progress is reset to zero (recompute mode, matching vLLM's recompute preemption).
- On the next step, the scheduler reorders the full queue including the preempted request.

This means preempted requests get implicit priority over fresh arrivals in FCFS mode. With priority-fcfs + slo-based, the preempted request's age still determines its position relative to other waiting requests.

## When to Use Which

| Workload | Recommended Configuration | Why |
|----------|--------------------------|-----|
| Uniform traffic, no SLO differentiation | `--scheduler fcfs` (default) | No reordering needed. All requests are equivalent. |
| Mixed SLO classes needing fairness | `--scheduler priority-fcfs --priority-policy slo-based` | Older requests float up, preventing starvation of any class. |
| Latency-sensitive short requests | `--scheduler sjf` | Short prompts get processed first. Watch for starvation of long requests under sustained load. |
| Low load (< ~10 req/s) | Any | Batch sizes are small enough that all schedulers pick the same requests. At low load, all four schedulers produce equivalent results within ~5%. |

!!! tip "SJF starvation risk"
    Under sustained high load, SJF can indefinitely delay long-prompt requests as short ones keep arriving. BLIS does not currently implement aging or starvation guards for SJF. If your workload has a mix of short and long prompts at high utilization, prefer `priority-fcfs` + `slo-based` instead.

## Further Reading

- [Routing Policies](routing.md) -- the upstream decision (which instance)
- [Cluster Simulation](cluster.md) -- the full request pipeline from arrival to completion
- [Core Engine: Scheduling Policies](../concepts/core-engine.md#scheduling-policies) -- implementation details and DES mechanics
