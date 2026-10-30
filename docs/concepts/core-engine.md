# Core Engine

This page describes BLIS's single-instance discrete event simulation engine. For multi-instance cluster orchestration, see [Cluster Architecture](architecture.md).

> **Canonical sources:** System invariants (INV-1 through INV-19, plus INV-A, INV-A2, INV-W3, the LoRA family INV-L1-INV-L6, PD disaggregation INV-PD-* and pool/transfer INV-P2-*) are defined in [`docs/contributing/standards/invariants.md`](../contributing/standards/invariants.md). If invariant descriptions here diverge, `invariants.md` is authoritative.

## Overview

Each BLIS instance pairs a host-side driver (`sim.EngineDriver`, `sim/engine_driver.go`) with an
engine it reaches only through blis-schemas' `engine.Engine` interface. The driver owns the
clock and the event queue, the client timeouts and the metrics; the engine makes every
scheduling, KV-cache and output decision and prices its own steps. The engine is
[blis-vllm](https://github.com/inference-sim/blis-vllm), built at one site,
`cmd/engine_build.go`, for the scenario's `engine_version: "0.32.0"`: blis-vllm's translated
vLLM v0.32.0 engine. A scenario with no `engine_version` is refused. The core loop is:

```
while events remain and clock < horizon:
    event = pop earliest event from queue
    advance clock to event.timestamp
    execute event (may produce new events)
```

## The engine interface

The driver turns its events into calls on `engine.Engine`:

| Driver event | Engine calls | Effect |
|---|---|---|
| arrival (`driverArrivalEvent`) | — | prices the request's front-end time (rendering, tokenization) and schedules its hand-off; it does not step the engine |
| queued (`driverQueuedEvent`) | `Add` | the request enters the engine's waiting queue; with no batch in flight, a step is queued at that instant (`driverArrivalStep`), after every arrival of the instant |
| step end (`driverStepEnd`) | `Complete` | at the step's end, the engine applies the oldest in-flight batch's output: emitted tokens, first tokens, finishes, KV events |
| wake (`driverWakeEvent`) | — | the engine asked to be stepped at an instant (`NextWake`) |
| timeout (`driverTimeoutEvent`) | `Finish` | a client deadline ends the request |

Every event but the arrival and the queued request then steps the engine (`kick`): it calls
`Schedule`, then `StepTime`, for as many batches as the engine admits while the GPU has a free
slot. The engine forms each batch as vLLM's scheduler does and says how long it runs.

**Clock monotonicity (INV-3):** the simulation clock never decreases; events are processed in
(timestamp, priority, seqID) order.

`InFlightLimit` bounds the batches in flight (vLLM's `max_concurrent_batches`, for example 2 under
async scheduling on one pipeline stage), and `Metrics`/`Snapshot` give the engine's `vllm:*` series and state for the
router's scrapes and the report. Batch formation, preemption, prefix caching, speculative
decoding, data parallelism and the KV cache are the engine's: see blis-vllm's
[README](https://github.com/inference-sim/blis-vllm#readme) and its
[feature matrix](https://github.com/inference-sim/blis-vllm/blob/main/docs/feature-matrix.md).

## Request Lifecycle

Requests follow a linear state machine with one exception (preemption):

```mermaid
stateDiagram-v2
    [*] --> Arrival
    Arrival --> Queued : host queueing delay
    Arrival --> DroppedUnservable : exceeds MaxModelLen or KV capacity

    state "Queued" as Queued
    note right of Queued
        EnqueueTime recorded
        Scheduler orders queue
    end note

    Queued --> Running : BatchFormation selects

    state "Running" as Running
    note right of Running
        ScheduleTime recorded
        TTFT at prefill→decode boundary
        KV blocks allocated
        ITL recorded per decode step
    end note

    Running --> Completed
    Running --> Queued : Preemption (KV eviction, re-enqueue front)

    state "Completed" as Completed
    note right of Completed
        CompletionTime recorded
        KV blocks released
        E2E = CompletionTime − ArrivalTime
    end note

    Completed --> RequestLeft
    RequestLeft --> [*]
    DroppedUnservable --> [*]
```

> **INV-5 (Causality):** ArrivalTime ≤ EnqueueTime ≤ ScheduleTime ≤ CompletionTime

### States

| State | Description |
|-------|-------------|
| **Queued** | In wait queue, not yet in running batch. Ordered by scheduling policy. |
| **Running** | In running batch, actively being processed. Progressing through prefill or decode. |
| **Completed** | All output tokens generated. KV blocks released. Metrics recorded. |

### Key Timestamps

These are the conceptual timestamps in a request's lifecycle. Some are stored as struct fields (e.g., `ArrivalTime`, `FirstTokenTime`), while others are computed at metric recording time.

| Timestamp | When Recorded | Used For |
|-----------|---------------|----------|
| Arrival time | Request creation | E2E and scheduling delay baseline |
| Enqueue time | After host queueing delay | Conceptual start of wait queue residence |
| Schedule time | Batch formation selects request | Scheduling delay = time in wait queue |
| First token time | End of prefill phase | TTFT = FirstTokenTime (stored on Request) |
| Completion time | All tokens generated | E2E = FirstTokenTime + sum(ITLs) |

**Causality invariant (INV-5):** `arrival_time <= enqueue_time <= schedule_time <= completion_time`

### Preemption

When KV cache pressure forces eviction, running requests are preempted:
1. Request is removed from the running batch. By default (`--preemption-policy fcfs`), the tail of the batch is evicted. `--preemption-policy priority` is refused on 0.32.0 for now.
2. Its KV blocks are freed
3. It is re-enqueued at the front of the wait queue (not the back)
4. A debug log is emitted and `PreemptionCount` is incremented

Preempted requests reset to the beginning of prefill (ProgressIndex = 0) and their KV blocks are freed. However, the freed blocks' prefix hashes are preserved in the KV cache's free list — when the request is re-scheduled, prefix caching may find these blocks (if not yet evicted by LRU), reducing recomputation. This matches vLLM's "recompute" preemption mode.

### Dropped Requests

Requests are dropped as unservable at enqueue time (incrementing `DroppedUnservable`) via two guards:

1. **MaxModelLen guard** — requests whose total sequence length exceeds the context window are rejected. When the request declares an output budget (`MaxOutputLen > 0`), the check is `input + budget > maxModelLen`. Otherwise, input length alone is checked (vLLM defaults `max_tokens` to `max_model_len - seq_len`; the engine's runtime stop handles output growth).
2. **KV capacity guard** — requests whose input tokens require more KV blocks than the total cache capacity are rejected. This prevents livelock where the simulator would endlessly preempt and re-enqueue a request that can never fit.

Both guards fire before the request enters the wait queue, mirroring vLLM's pre-engine rejection. Past admission, the engine caps a request at `max_model_len` as vLLM's `check_stop` does (`num_tokens >= max_model_len`): a length-capped request emits `MaxModelLen - len(InputTokens)` output tokens.

## Latency Models

Step time and host overheads come from **blis-latency-kernel**. The engine asks the kernel to
price each batch it forms (blis-vllm's kernel adapter, `compat/kernelmodel`), from the model
graph, chip and coefficients the scenario names; the 0.32.0 engine is priced with the kernel's
0.31.0 tuning, and a run says so on stderr. See the
[Latency Models guide](../guide/latency-models.md).

Host overheads are charged around the GPU step, never on it:

- **Queueing time** (tokenization, preprocessing) delays request enqueue but does not block
  the server; the simulation clock is not advanced by it.
- **Output token processing time** (detokenization, streaming) is added to per-request
  ITL/TTFT metrics but does not block the next step.
- **Post-decode fixed overhead** is charged once per request at completion.

This matches vLLM, where CPU post-processing runs concurrently with GPU execution.

## Scheduling Policies

Each instance's engine orders its waiting requests by the pool's engine block `scheduling_policy`, as vLLM does. Priority scheduling and `--preemption-policy priority` are refused on 0.32.0 for now. See [Scheduling & Priority](../guide/scheduling.md).

## Metrics

BLIS records per-request and aggregate metrics throughout the simulation.

### Per-Request Metrics

| Metric | Definition |
|--------|------------|
| **TTFT** | Time from arrival to first token: includes queueing delay, prefill step times, and per-token output processing overhead |
| **E2E** | `FirstTokenTime + sum(ITLs)`, where each ITL includes step time + per-token output processing |
| **ITL** | Observed time between consecutive decode steps (includes per-token output processing) |
| **Scheduling Delay** | Time from request arrival to entering the running batch (includes host queueing overhead + wait queue residence) |

### Aggregate Metrics

| Metric | Aggregation |
|--------|-------------|
| TTFT, E2E, ITL distributions | Mean, p90, p95, p99 in JSON output. Cluster-internal `Distribution` type also computes p50, min, max for fitness evaluation. |
| Throughput | Output tokens per second, requests per second |
| Preemption count | Total KV cache evictions |
| KV allocation failures | Failed block allocations |
| Dropped unservable | Requests too large for cache |

### Conservation Invariant (INV-1)

At simulation end, for a **single-instance** run: `injected_requests == completed_requests + still_queued + still_running + dropped_unservable + timed_out`

That five-term form is the single-instance specialisation, not the general rule. The canonical form is an eleven-term cluster equation: a cluster run adds six buckets for the embedded llm-d router's outcomes — routing rejections, flow-control queue depth and capacity rejections, in-flight eviction, TTL expiry, and encode-pool routing. Asserting the five-term form against cluster output is incomplete — see canonical [INV-1](../contributing/standards/invariants.md) for the full equation and the shared helpers that implement it.

This is the fundamental accounting invariant that ensures no requests are silently lost.

## Determinism

BLIS guarantees deterministic output: the same input at the same revisions produces a byte-identical `Result` document outside its `runtime` block (INV-6).

Key mechanisms:
- **Partitioned RNG:** Each subsystem (workload generation, scheduling, etc.) uses an independent random stream derived from the seed, so adding randomness to one subsystem doesn't perturb others
- **Sorted map iteration:** Where map iteration order affects output, keys are sorted first
- **Stdout/stderr separation:** Deterministic results go to stdout; wall-clock timing and diagnostics go to stderr via logrus
