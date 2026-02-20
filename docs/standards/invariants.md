# BLIS System Invariants

Invariants are properties that must hold at all times during and after simulation. They are verified by invariant tests (see R7) and checked during self-audit (Step 4.75).

## INV-1: Request Conservation

**Statement:** `injected_requests == completed_requests + still_queued + still_running` at simulation end (all levels).

**Full pipeline:** `num_requests == injected_requests + rejected_requests` (from anomaly counters).

**Verification:** `sim/cluster/cluster_test.go` — conservation tests. Conservation fields (`still_queued`, `still_running`, `injected_requests`) are included in CLI JSON output.

**Evidence:** Issue #183 — a silently-dropped request violated conservation for months.

**Experimental validation:** H12 confirmed conservation across 10 policy configurations (67 invariant checks) — including round-robin, least-loaded, weighted (multiple scorer configs), SJF, priority-FCFS, token-bucket admission, and always-busiest. H8 confirmed conservation under extreme KV pressure (15 configurations). Full preemption-path validation is blocked by the panic bug (#293).

---

## INV-2: Request Lifecycle

**Statement:** Requests transition `queued -> running -> completed`. No invalid transitions. Requests not completed before horizon remain in current state.

**Verification:** State machine assertions in request processing code.

---

## INV-3: Clock Monotonicity

**Statement:** Simulation clock never decreases. Every event's timestamp >= the previous event's timestamp.

**Verification:** Clock is advanced in the event loop only via min-heap extraction, which guarantees non-decreasing order.

---

## INV-4: KV Cache Conservation

**Statement:** `allocated_blocks + free_blocks = total_blocks` at all times.

**Verification:** Checked after every allocation/deallocation. Transactional allocation with rollback on mid-loop failure (R5).

**Operational note (H8):** KV cache pressure exhibits a sharp cliff, not gradual degradation. In H8's workload, performance was identical above ~2200 blocks and collapsed below it (4.7x TTFT P99 increase with just 4.5% fewer blocks). Below ~1000 blocks, the preempt-requeue cycle can livelock (see R19). Capacity planning formula: `threshold ≈ rate / num_instances × (input_tokens + output_tokens) / block_size`.

---

## INV-5: Causality

**Statement:** `arrival_time <= enqueue_time <= schedule_time <= completion_time` for every request.

**Verification:** Per-request metric timestamps recorded at each lifecycle stage. Invariant tests verify ordering for all completed requests.

---

## INV-6: Determinism

**Statement:** Same seed must produce byte-identical stdout across runs.

**Verification:** Run same configuration twice with same seed; diff stdout. Wall-clock timing goes to stderr (not stdout).

**Common violation sources:**
- Go map iteration feeding output ordering (R2)
- Floating-point accumulation order dependencies
- Wall-clock-dependent randomness (must use PartitionedRNG)
- Stateful scorers with non-deterministic internal state

---

## INV-7: Signal Freshness Hierarchy

**Statement:** Routing snapshot signals have tiered freshness due to DES event ordering. Cluster events at tick T drain before instance events at tick T.

| Signal | Owner | Freshness | Updated By |
|--------|-------|-----------|------------|
| PendingRequests | Cluster | Synchronous | `RoutingDecisionEvent.Execute()` |
| QueueDepth | Instance | Stale within tick | `QueuedEvent.Execute()` |
| BatchSize | Instance | Stale within tick | `StepEvent.Execute()` |
| KVUtilization | Instance | Stale across batch steps | `makeRunningBatch()` -> `AllocateKVBlocks()` |
| CacheHitRate | Instance | Stale across batch steps | `makeRunningBatch()` |

**Design implication:** `EffectiveLoad()` = `QueueDepth + BatchSize + PendingRequests` compensates for Tier 2 staleness by including the Tier 1 PendingRequests term. KVUtilization has no analogous compensation.

**Verification:** H3 hypothesis experiment (`hypotheses/h3-signal-freshness/`).

**Evidence:** Issues #282, #283. At rate=5000, kv-utilization-only routing produces 200x worse distribution uniformity than queue-depth.
