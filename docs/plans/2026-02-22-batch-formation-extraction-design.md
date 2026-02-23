# Design: BatchFormation Interface Extraction

**Status:** Draft
**Date:** 2026-02-22
**Issue:** #242
**Species:** Specification (behavioral contract for a new interface boundary)
**Design Guidelines:** Section 4.2 (Target module map), Section 5.4 (Backend Swap recipe)

## Problem

`makeRunningBatch()` is a ~110-line private method on `Simulator` with no interface. It interleaves:
- Token budget management (batch formation)
- KV cache allocation (KV cache manager concern)
- Preemption / eviction (shared concern)
- Request state mutation (lifecycle concern)
- Metrics recording (statistics concern)
- Event scheduling (kernel concern)

This embodies vLLM's FCFS + chunked-prefill + preemption strategy. Alternative strategies cannot be swapped in:
- **Mooncake disaggregated prefill/decode** — prefill and decode run on separate instance pools with different batch formation rules
- **Speculative decoding** — multiple candidate tokens per step, different token budget accounting
- **SGLang tree-based scheduling** — RadixAttention-aware batch selection with different KV allocation patterns
- **Continuous batching without preemption** — simpler batch formation that just stops admitting when full

The design guidelines (Section 4.2) list batch formation as a "Target" module needing interface extraction, with a target extension friction of ~2 files.

## Scope

**Phase A only** (this PR): Extract `BatchFormation` interface from existing code. Move `makeRunningBatch()` logic behind the interface. Existing tests pass unchanged. No new strategies.

Phase B (future PRs): Implement alternative strategies behind the extracted interface.

## Approach: Separation of Decisions from Side Effects

The key design challenge: `makeRunningBatch()` mixes **decisions** (which requests to schedule, which to preempt) with **side effects** (KV allocation, event scheduling, metrics, state mutation).

**Decision:** Split the method into two parts:
1. **BatchFormation interface** — makes decisions about batch composition (what to schedule, what to preempt, token accounting)
2. **Simulator orchestration** — applies those decisions (KV allocation, events, metrics, state mutation)

This split keeps the interface focused on the scheduling strategy while the Simulator retains control of kernel concerns (event scheduling, metrics, state).

**Alternative considered:** Having the interface encompass all side effects (KV allocation, event scheduling, etc.). Rejected because:
- Different strategies need different KV allocation patterns — the interface would need to expose the full KVStore API
- Event scheduling is a kernel concern that shouldn't leak into policy modules
- Metrics recording is a statistics concern per DES principles

**Alternative considered:** Pure selection function `SelectBatch(candidates []*Request) []*Request`. Rejected because:
- Ignores KV cache constraints entirely — the strategy needs to know whether blocks can be allocated
- Preemption decisions depend on allocation results (evict → retry allocation → succeed/fail)
- Token budget interacts with chunked prefill and KV allocation — can't separate selection from budget

**Chosen approach:** The interface receives enough context to make allocation-aware decisions, but allocation itself happens inside the implementation (which receives the KVStore). The implementation returns a result struct describing what happened, and the Simulator applies the remaining side effects (events, metrics, state).

## Interface Contract

```
BatchFormation interface:
    FormBatch(ctx BatchContext) BatchResult
```

**BatchContext** (inputs — all read-only from the caller's perspective):
- Current running batch (requests continuing from previous step)
- Wait queue contents (ordered by scheduler)
- KV cache handle (for allocation queries and allocation)
- Token budget (max scheduled tokens)
- Max batch size (max running requests)
- Chunked prefill threshold
- Current simulation time
- Step count (for ScheduledStepIdx)
- Per-request computed token map (for tracking progress across steps)

**BatchResult** (outputs — decisions made):
- Updated running batch (continuing + newly scheduled requests)
- Preempted requests (to be re-queued)
- Newly scheduled requests (need ScheduledEvent + metrics recording)
- Whether preemption happened (flag for caller)
- Updated computed token map entries

## Implementation: VLLMBatchFormation

The existing `makeRunningBatch()` + `preempt()` logic, unchanged in behavior. Implements the vLLM FCFS + chunked-prefill + preemption strategy:

1. Process continuing requests: apply chunked prefill token limits, allocate KV blocks, preempt if needed
2. Dequeue new requests: check KV availability, apply cache hits, allocate blocks, respect batch size and token budget limits
3. Stop dequeuing if preemption happened (vLLM rule)

The implementation calls KVStore methods directly (AllocateKVBlocks, GetCachedBlocks, ReleaseKVBlocks) because KV allocation is integral to vLLM's batch formation decisions.

## Module Contract (per Section 4.3)

| Aspect | Contract |
|---|---|
| **Observes** | Running batch state, wait queue (ordered), KV cache state, token budget, batch size limit, chunked prefill threshold |
| **Controls** | Which requests run in next step, which are preempted, token allocation per request |
| **Owns** | No persistent state (stateless per step; all state is passed in via BatchContext) |
| **Invariants** | BC-1: Requests in result batch have KV blocks allocated. BC-2: Token budget not exceeded. BC-3: Batch size not exceeded. BC-4: Preempted requests have KV blocks released. |
| **Events** | None (the interface doesn't schedule events — that's the Simulator's job) |
| **Extension friction** | 2 files to add a new strategy (implementation file + factory registration) |

## Simulator Changes

**In `Step()`:** Replace `sim.makeRunningBatch(now)` with:
1. Build `BatchContext` from current simulator state
2. Call `sim.batchFormation.FormBatch(ctx)`
3. Apply result: update RunningBatch, create ScheduledEvents for newly scheduled requests, create PreemptionEvents for preempted requests, record metrics

**Remove from `Simulator`:** `makeRunningBatch()` method, `preempt()` method, `preemptionHappened` field (moved into BatchResult)

**Add to `Simulator`:** `batchFormation BatchFormation` field

**Add to `SimConfig`:** Nothing — the factory selects VLLMBatchFormation by default (the only implementation in Phase A)

## File Organization

**New file:** `sim/batch_formation.go` — interface, BatchContext, BatchResult, VLLMBatchFormation, factory

**Modified files:**
- `sim/simulator.go` — field changes, Step() orchestration, remove makeRunningBatch()/preempt()

**Unchanged:**
- `sim/kvcache.go`, `sim/kv_store.go` — KVStore interface unchanged
- `sim/cluster/` — delegates to Simulator, no changes needed
- `sim/latency_model.go` — independent module

## R13 Compliance (Multi-impl interfaces)

R13 requires new interfaces to work for >=2 backends. In Phase A, there is only one implementation (VLLMBatchFormation). This is the same pattern as the LatencyModel extraction (#241), which had two existing implementations. Here, the second implementation (e.g., disaggregated batching for PR14) is planned for Phase B.

Mitigation: The interface design is validated against three alternative strategies (disaggregated, speculative, continuous-without-preemption) to ensure it accommodates them without changes. The BatchContext/BatchResult pattern is strategy-agnostic — each strategy interprets the inputs differently and produces different outputs.

## Testing Strategy

1. **Behavioral equivalence:** All existing tests pass unchanged. The golden dataset test exercises Step() end-to-end. No golden dataset regeneration needed.
2. **Unit tests for VLLMBatchFormation:** Test batch formation logic in isolation — token budget exhaustion, preemption, KV allocation failure, cache hits, chunked prefill, batch size limits.
3. **Invariant tests:** BC-1 through BC-4 verified via behavioral assertions (not structural).
4. **Factory test:** Default factory returns VLLMBatchFormation (behavioral — verify FormBatch produces expected results, not type assertion).

## Complexity

Medium. ~200-250 lines of new code (interface + VLLMBatchFormation), ~150 lines removed from simulator.go (makeRunningBatch + preempt). Net simplification of simulator.go.
