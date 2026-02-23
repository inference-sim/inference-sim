# Mock Study Findings

> **Note (v2.3):** This document was written before the v2.3 macro plan restructuring. References to "PR 6" for InstanceSnapshot and Run() restructuring now correspond to PR 4 in the v2.3 plan. See `2026-02-11-macro-implementation-plan-v2.md` for current PR mapping.

**Date:** 2026-02-13
**Status:** Complete
**Context:** Post-PR3 checkpoint per macro implementation plan v2.2

## Summary

Four hand-coded routing policies (round-robin, least-loaded, random, KV-aware) were tested against the ClusterSimulator API. The study revealed one **critical architectural issue** and four **observable gaps** that must be addressed before Phase 2 interface design.

## Critical Finding: Online Routing Required

**The current "batch-dispatch-then-run" architecture is incompatible with load-aware routing policies.**

`ClusterSimulator.Run()` dispatches ALL requests to instances before starting the event loop. This means:
- At dispatch time, all instances have zero queue depth, zero batch size, zero KV usage
- Least-loaded and KV-aware policies see identical state everywhere
- Tie-breaking routes **all** requests to instance 0
- Under contention (Scenario 1), this causes 3.5× worse throughput vs round-robin

```
Scenario 1: Tight KV (128 blocks, 2 instances, 200 reqs @ 50/s)
Policy          |  Completed |   MeanTTFT |    P99TTFT |    MeanE2E | Throughput
round-robin     |        191 |    18.13ms |   349.50ms |   172.39ms |    46.34/s
least-loaded    |         57 |  1395.74ms |  3035.01ms |  1605.51ms |    13.29/s
random          |        183 |    74.36ms |   675.11ms |   225.69ms |    43.10/s
kv-aware        |         57 |  1395.74ms |  3035.01ms |  1605.51ms |    13.29/s
```

**Implication for PR 6:** `ClusterSimulator.Run()` must be restructured so that routing decisions happen **during** the event loop, not before it. Each request's arrival event should trigger a routing decision using the current simulation state. This is the most important change the mock study surfaces.

**Proposed architecture change:**
1. Request generation produces `(request, arrivalTime)` pairs but does NOT assign instances
2. Arrival events are injected into a cluster-level event queue (or a scheduler)
3. When an arrival event fires, the routing policy is called with current `InstanceSnapshot`s
4. The request is then injected into the chosen instance

This aligns with real load balancer behavior — routing happens at arrival time, not in batch.

## Observable Gaps

| # | Observable Needed | Current Access Path | Proposed API |
|---|---|---|---|
| 1 | Queue depth | `inst.sim.WaitQ.Len()` | `InstanceSnapshot.QueueDepth` |
| 2 | Running batch size | `len(inst.sim.RunningBatch.Requests)` | `InstanceSnapshot.BatchSize` |
| 3 | KV utilization | `inst.sim.KVCache.{TotalBlocks,UsedBlockCnt}` | `InstanceSnapshot.KVUtilization` |
| 4 | Prefix cache state | `inst.sim.KVCache.HashToBlock` | `InstanceSnapshot.CacheHitRate` |

All four gaps require reaching through `InstanceSimulator.sim` (unexported field). This works within `package cluster` (same-package access) but will **not** work for `sim/policy/` — a separate package. The `InstanceSnapshot` abstraction is therefore **essential**, not optional.

## Awkward Patterns

1. **Package boundary problem:** `InstanceSimulator.sim` is unexported. Policies in `sim/policy/` cannot access simulator internals. `InstanceSnapshot` must be constructed by `ClusterSimulator` and passed to policies.

2. **Missing WaitQueue.Len():** Had to add as a prerequisite — the simplest observable was absent. (Fixed in this study.)

3. **No snapshot semantics:** Policies observe mutable state that can change between observation and decision. `InstanceSnapshot` provides point-in-time consistency.

4. **No cache hit rate tracking:** KV cache tracks block allocation but not hit/miss counts. Prefix-affinity routing (planned in PR 6) needs this.

## Under-Saturation Masks Policy Differences

```
Scenario 2: 8 instances, 500 reqs @ 100/s, 500 KV blocks
Policy          |  Completed |   MeanTTFT |    P99TTFT |    MeanE2E | Throughput
round-robin     |        500 |     2.77ms |     4.60ms |    74.88ms |    98.55/s
least-loaded    |        500 |     3.19ms |     4.85ms |    86.05ms |    98.46/s
random          |        500 |     2.87ms |     4.53ms |    75.70ms |    98.55/s
kv-aware        |        500 |     3.19ms |     4.85ms |    86.05ms |    98.46/s
```

With generous resources, all policies perform identically. Policy research requires workloads that create contention. This validates the plan's inclusion of workload generation enhancements (PR 10) and pathological templates (PR 9).

## Interface Adjustment Recommendations

### For ClusterSimulator (affects PR 6 architecture)
- **Restructure `Run()` to support online routing.** Requests must be routed at their arrival time, not pre-dispatched. This is a fundamental change to the event loop.
- Consider a cluster-level `ArrivalEvent` that triggers routing, then injects into the chosen instance.

### For PR 6 (RoutingPolicy + InstanceSnapshot)
- `InstanceSnapshot` must include at minimum: `QueueDepth`, `BatchSize`, `KVUtilization`, `FreeKVBlocks`, `InFlightRequests`
- Add `CacheHitRate` field (requires new tracking in `KVCacheState`)
- `InstanceSnapshot` should be constructed by `ClusterSimulator` before each routing decision (not by policy)
- The `RoutingPolicy.Route()` signature in the macro plan looks correct — it receives `[]InstanceSnapshot`

### For PR 4 (AdmissionPolicy)
- Admission decisions also need runtime state — total cluster queue depth, etc.
- `RouterState.Global` should include aggregate queue depth across instances

### For PR 9 (Pathological Templates)
- The `always-busiest` pathological template needs online routing to demonstrate HOL blocking — it won't work with pre-dispatch
- Test workloads must create contention (tight KV, high rate) to differentiate policies

### General
- `WaitQueue.Len()` added to `sim/queue.go` (done in this study)
- Consider `InstanceSimulator.Snapshot() InstanceSnapshot` method for PR 6

## Test Artifacts

All tests are in `sim/cluster/mock_study_test.go`:
- `TestMockStudy_RoundRobin_MatchesClusterSimulator` — harness validation (PASS)
- `TestMockStudy_RoutingPolicies_CompareMetrics` — 4-policy comparison under low load (PASS)
- `TestMockStudy_ObservableGaps` — documents API gaps (PASS)
- `TestMockStudy_HighLoad_PolicyDifferentiation` — 4-policy comparison under contention (PASS)
