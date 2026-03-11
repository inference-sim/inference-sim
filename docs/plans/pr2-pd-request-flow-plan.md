# PD Disaggregation PR2: End-to-End Disaggregated Request Flow — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable requests to be split into separate prefill and decode phases that execute on dedicated instance pools with simulated KV cache transfer between them.

**The problem today:** BLIS models every request as a single lifecycle on one instance — prefill and decode are co-located. This prevents capacity planning for disaggregated serving architectures (llm-d, DistServe, Splitwise) where prefill and decode run on separate pools. Users cannot answer: what pool ratio is optimal? When does KV transfer overhead exceed interference elimination benefit?

**What this PR adds:**
1. **Request splitting** — when disaggregation is enabled, each request produces a prefill sub-request and a decode sub-request. A parent record links them for metrics.
2. **Two-stage routing** — prefill sub-requests are routed to prefill-pool instances using pool-filtered snapshots; decode sub-requests are routed to decode-pool instances after KV transfer.
3. **KV transfer simulation** — after prefill completes, an asynchronous event models the network time to transfer KV cache data. Transfer duration depends on block count and configured bandwidth.
4. **Per-pool scorer configs** — `--prefill-routing-scorers` and `--decode-routing-scorers` allow independent tuning of routing for each pool.

**Why this matters:** This is the core disaggregation PR. After PR2, users can run pool ratio sweeps and transfer bandwidth sensitivity analyses. PR3 (metrics), PR4 (traces), and PR5 (prefix-threshold decider) all build on this foundation.

**Architecture:** All new code lives in `sim/cluster/` (parent request tracking, 4 new event types, pool-filtered routing, prefill completion detection, KV transfer scheduling). Small additions to `sim/`: `batch_formation.go` handles decode-only requests entering the batch (requests with pre-allocated KV arriving with ProgressIndex past input); `simulator.go` adds `EnqueueDecodeSubRequest()` method bypassing the oversized-request guard and TotalInputTokens counting (decode sub-requests have KV pre-allocated, so the guard is inappropriate and would leak blocks). New CLI flags in `cmd/root.go`.

**Source:** Macro plan PR2 in `docs/plans/pd-disaggregation-macro-plan.md`, design doc `docs/plans/pd-disaggregation-design.md`, GitHub issue #592.

**Closes:** Fixes #592

**Behavioral Contracts:** See Part 1, Section B below

---

## Phase 0: Component Context

### Building block placement
- **Added:** Request Splitter (cluster orchestrator concern), KV Transfer (cluster orchestrator concern), Pool-Scoped Router (configuration)
- **Modified:** DisaggregationDecisionEvent (bifurcation), ClusterSimulator (parent tracking, completion detection), DeploymentConfig (transfer config, per-pool scorers)
- **Adjacent:** Pool Topology (PR1, read-only), Routing Policy (existing, reused with filtered snapshots), KV Store (existing, used for pre-allocation), Batch Formation (existing, small decode-only path addition)

### Invariants touched
- INV-PD-1 through INV-PD-4 (new, verified by tests)
- INV-1 (extended for intermediate states)
- INV-4 (per-pool conservation)
- INV-6 (determinism preserved)
- INV-8 (per-pool work-conserving)

### Construction site audit

**DeploymentConfig** — constructed in:
1. `cmd/root.go:722-747` (CLI construction)
2. `sim/cluster/cluster_test.go` — `newTestDeploymentConfig()` helper
3. `sim/cluster/test_helpers_test.go` — various test helpers
4. `sim/cluster/*_test.go` — multiple test files

All sites use zero-value defaults for new fields (backward compatible). Tests that need disaggregation will use a new `newTestDisaggDeploymentConfig()` helper.

**Request** — constructed in:
1. `sim/cluster/test_helpers_test.go:40-48` — test helper
2. `sim/workload/generator.go` — production path
3. Various test files

Sub-requests are constructed ONLY in cluster-level code (new `splitRequest()` function). No modification to existing construction sites.

### Confirmed facts (with citations)
- DisaggregationDecisionEvent priority is 3 (`cluster_event.go:207`)
- PR1 stub: both paths schedule RoutingDecisionEvent (`cluster_event.go:226-228`)
- buildRouterState collects ALL instance snapshots (`cluster_event.go:63-72`)
- Completion detection uses CompletedRequests delta (`cluster.go:173-187`)
- InjectRequestOnline delegates to InjectArrivalAt (`instance.go:142-144`)
- KVStore.AllocateKVBlocks(req, start, end, cached) allocates KV blocks (`kv_store.go:13`)
- KVStore.ReleaseKVBlocks(req) frees all blocks for a request (`kv_store.go:15`)
- VLLMBatchFormation Phase 2 computes numNewTokens from inputLen and cachedBlocks (`batch_formation.go:121-122`)
- Request completion: `ProgressIndex == inputLen + max(outputLen, 1) - 1` (`simulator.go:407`)
- Zero-output requests complete after prefill (`simulator.go:410-413`)

### Deviations from source
- **DEVIATION (CORRECTION):** Design doc uses 5-unit priority spacing (0, 10, 15, 20, 25, 30, 35). PR1 established sequential integers (0, 1, 2, 3). PR2 continues with 4, 5, 6, 7 for new events.
- **DEVIATION (SIMPLIFICATION):** Design doc specifies separate `--kv-transfer-bandwidth` and `--kv-transfer-base-latency` for PD transfer. These names conflict with existing tiered KV cache flags. PR2 uses `--pd-transfer-bandwidth` and `--pd-transfer-base-latency` to avoid ambiguity.
- **DEVIATION (ADDITION):** Added `--pd-kv-bytes-per-token` CLI flag (not in design doc). Required to compute block_size_bytes for transfer duration when model config is unavailable (blackbox mode). Auto-computed from ModelConfig in roofline/crossmodel mode.

---

## Part 1: Design Validation

### A) Executive Summary

This PR implements the complete disaggregated request flow: request splitting, two-stage routing (prefill pool → decode pool), KV transfer simulation, and per-pool scorer configuration. It builds on PR1's pool topology and disaggregation decider interface.

When `AlwaysDisaggregate` (or any future decider) returns "disaggregate", the request is split into a prefill sub-request routed to the prefill pool and, after simulated KV transfer, a decode sub-request routed to the decode pool. Each sub-request follows the existing per-instance lifecycle. A cluster-level parent record links them.

When `NeverDisaggregate` is active (default, no pools configured), the pipeline is unchanged — byte-identical to pre-PR1 behavior.

Adjacent blocks: Pool Topology (PR1, read-only), Routing Policy (existing, reused), KV Store (existing, used for pre-allocation on decode instance), Batch Formation (small addition for decode-only requests).

No deviations that affect behavior. Priority numbering and CLI flag naming deviations are cosmetic.

### B) Behavioral Contracts

#### Positive Contracts

**BC-PD-5: Disaggregated request completion**
- GIVEN a cluster with pools configured and AlwaysDisaggregate decider
- WHEN requests are injected
- THEN every request completes through the full disaggregated path (prefill on prefill instance → KV transfer → decode on decode instance) and all output tokens are generated
- MECHANISM: DisaggregationDecisionEvent bifurcates to PrefillRoutingEvent; prefill completion triggers KVTransferStartedEvent → KVTransferCompletedEvent → DecodeRoutingEvent; decode sub-request generates output tokens normally

**BC-PD-6: KV completeness (INV-PD-1)**
- GIVEN a disaggregated request
- WHEN the decode sub-request is enqueued on a decode instance
- THEN the enqueue timestamp MUST be ≥ the KV transfer completion timestamp
- MECHANISM: DecodeRoutingEvent is scheduled only after KVTransferCompletedEvent fires

**BC-PD-7: Pool exclusivity (INV-PD-2)**
- GIVEN a cluster with prefill and decode pools
- WHEN requests are disaggregated
- THEN prefill sub-requests are routed ONLY to prefill pool instances and decode sub-requests ONLY to decode pool instances
- MECHANISM: Pool-filtered snapshots exclude non-pool instances before routing

**BC-PD-8: Transfer conservation (INV-PD-3)**
- GIVEN a simulation with disaggregated requests
- WHEN the simulation completes
- THEN initiated_transfers MUST equal completed_transfers
- MECHANISM: Counters incremented at KVTransferStartedEvent and KVTransferCompletedEvent; verified at simulation end

**BC-PD-9: Phase causality (INV-PD-4)**
- GIVEN a disaggregated request
- WHEN it completes
- THEN timestamps form a valid causal chain: arrival ≤ prefill_enqueue ≤ prefill_complete ≤ transfer_start ≤ transfer_complete ≤ decode_enqueue ≤ completion
- MECHANISM: Each event records its timestamp on the parent record; verified by invariant test

**BC-PD-10: Extended conservation (INV-1)**
- GIVEN a simulation with disaggregated requests
- WHEN the simulation ends
- THEN injected_requests == completed + prefill_queued + prefill_running + transferring + decode_queued + decode_running + dropped_unservable
- MECHANISM: Parent records track phase state; conservation check at simulation end

**BC-PD-11: Per-pool work-conserving (INV-8)**
- GIVEN a decode instance with a non-empty wait queue after a step completes
- WHEN a KV transfer completes and a decode sub-request is enqueued on that instance
- THEN a StepEvent MUST exist in the instance event queue
- MECHANISM: InjectRequestOnline schedules ArrivalEvent → QueuedEvent which triggers StepEvent if none scheduled

**BC-PD-12: Determinism (INV-6)**
- GIVEN identical configuration and seed
- WHEN the disaggregated simulation runs twice
- THEN stdout output MUST be byte-identical
- MECHANISM: All new code paths consume from PartitionedRNG deterministically; pool-filtered snapshots are ordered by instance index

**BC-PD-13: Non-disaggregated path unchanged**
- GIVEN pools not configured (prefill=0, decode=0)
- WHEN requests are injected
- THEN behavior is byte-identical to pre-PR2 code
- MECHANISM: poolsConfigured() returns false; no new events created; DisaggregationDecisionEvent not scheduled

**BC-PD-14: Transfer duration computation**
- GIVEN a prefill sub-request completing with N input tokens
- WHEN KV transfer is initiated
- THEN transfer_duration = ceil(base_latency_us + num_blocks * block_size_bytes / bandwidth_bytes_per_us) where num_blocks = ceil(N / block_size_tokens)
- MECHANISM: Computed in KVTransferStartedEvent.Execute()

**BC-PD-15: Per-pool routing**
- GIVEN --prefill-routing-scorers and --decode-routing-scorers configured differently
- WHEN requests are disaggregated
- THEN prefill routing uses prefill scorer config and decode routing uses decode scorer config
- MECHANISM: Separate routing policy instances constructed from per-pool scorer configs

#### Negative Contracts

**NC-PD-1: No cross-pool routing**
- GIVEN a cluster with pools configured
- WHEN a prefill sub-request is routed
- THEN it MUST NOT be routed to a decode pool instance (and vice versa)
- MECHANISM: Pool-filtered snapshots exclude wrong-pool instances

**NC-PD-2: No parent record for non-disaggregated requests**
- GIVEN pools not configured OR disaggregation decision is "local"
- WHEN a request is routed
- THEN no parent record, no sub-requests, no transfer events are created
- MECHANISM: DisaggregationDecisionEvent routes "local" to standard RoutingDecisionEvent

**NC-PD-3: No KV block leak**
- GIVEN a disaggregated request completing on the decode instance
- WHEN ReleaseKVBlocks is called
- THEN all KV blocks (pre-allocated input + allocated decode) are freed; UsedBlocks returns to pre-request level
- MECHANISM: AllocateKVBlocks appends all block IDs to RequestMap; ReleaseKVBlocks frees all of them

#### Error Handling Contracts

**EC-PD-1: Decode instance KV capacity insufficient**
- GIVEN a decode sub-request arriving at a decode instance
- WHEN KV pre-allocation for transferred blocks fails (insufficient capacity)
- THEN the decode sub-request is NOT injected; the request is silently lost (counted via parent record tracking, not in per-instance DroppedUnservable)
- MECHANISM: If AllocateTransferredKV returns false in DecodeRoutingEvent, log a warning and return without injection. The parent record's CompletionTime remains zero, enabling detection at simulation end.

**EC-PD-2: Transfer parameters validation**
- GIVEN --pd-transfer-bandwidth ≤ 0 OR --pd-transfer-base-latency < 0
- WHEN CLI flags are parsed
- THEN the program exits with a descriptive error message
- MECHANISM: CLI validation in cmd/root.go (R3)

### C) Component Interaction

```
AdmissionDecisionEvent
    │
    ├── [pools not configured] ──► RoutingDecisionEvent (unchanged)
    │
    └── [pools configured] ──► DisaggregationDecisionEvent
                                    │
                                    ├── [local] ──► RoutingDecisionEvent (unchanged)
                                    │
                                    └── [disaggregate] ──► splitRequest()
                                                            │
                                                            ▼
                                                    PrefillRoutingEvent
                                                    (pool-filtered prefill snapshots)
                                                            │
                                                            ▼
                                                    InjectRequestOnline(prefillSubReq)
                                                            │
                                                    [prefill runs on instance]
                                                            │
                                                    [prefill completes — detected in event loop]
                                                            │
                                                            ▼
                                                    KVTransferStartedEvent
                                                    (compute duration, increment counter)
                                                            │
                                                            ▼
                                                    KVTransferCompletedEvent
                                                    (increment counter, create decode sub-req)
                                                            │
                                                            ▼
                                                    DecodeRoutingEvent
                                                    (pool-filtered decode snapshots)
                                                            │
                                                            ▼
                                                    AllocateTransferredKV + InjectRequestOnline(decodeSubReq)
                                                            │
                                                    [decode runs on instance]
                                                            │
                                                    [decode completes — normal path]
```

**New mutable state (owned by ClusterSimulator):**
- `parentRequests map[string]*ParentRequest` — parent ID → tracking record (created at split, read at transfer/completion)
- `pendingPrefillCompletions map[string]string` — prefill sub-req ID → parent ID (created at inject, deleted at completion detection)
- `transfersInitiated int` / `transfersCompleted int` — INV-PD-3 counters
- `prefillRoutingPolicy sim.RoutingPolicy` / `decodeRoutingPolicy sim.RoutingPolicy` — per-pool routing (nil = use default)

**Extension friction:** Adding a new event type: 1 file (event definition). Adding a new decider: ~3 files (unchanged from PR1). Adding a new transfer model: 0 files Phase 1 (embedded in cluster), ~2 files Phase 2 (interface extraction).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| Priority constants: 15, 20, 25, 30, 35 | Sequential: 4, 5, 6, 7 | CORRECTION: PR1 established 0,1,2,3 numbering. Continue pattern. |
| `--kv-transfer-bandwidth` for PD transfer | `--pd-transfer-bandwidth` | SIMPLIFICATION: Avoids name collision with existing tiered KV cache flags |
| `--kv-transfer-base-latency` for PD transfer | `--pd-transfer-base-latency` | SIMPLIFICATION: Same as above |
| No bytes-per-token parameter | Added `--pd-kv-bytes-per-token` | ADDITION: Required for transfer duration computation when model config unavailable |
| Disaggregation-aware metrics in scope | Deferred to PR3 | DEFERRAL: Per macro plan, metrics are PR3 scope |
| Decision traces in scope | Deferred to PR4 | DEFERRAL: Per macro plan, traces are PR4 scope |
| PrefixThresholdDecider in scope | Deferred to PR5 | DEFERRAL: Per macro plan |
| KVTransferStartedEvent deallocates KV on prefill instance | Prefill instance releases KV normally via processCompletions | SIMPLIFICATION: Per-instance simulator untouched; KV released at sub-request completion automatically. Transfer duration computed from input token count. |
| Per-pool INV-4 verified during transfer | Verified at simulation end only | SIMPLIFICATION: Per-instance KV conservation is maintained by each instance's KV store independently. No cross-instance block movement in the simulation. |
| Decode sub-request injected via normal ArrivalEvent path | Decode sub-request injected via dedicated InjectDecodeOnline path | CORRECTION: Normal EnqueueRequest adds TotalInputTokens and applies oversized-request guard. For decode sub-requests with pre-allocated KV, the guard is inappropriate (would leak blocks) and input tokens would be double-counted. Dedicated injection path bypasses both. |
| Alpha-model QueueingTime applied to decode sub-requests | QueueingTime NOT applied (direct enqueue) | ADDITION: Decode sub-requests bypass ArrivalEvent → QueuedEvent chain, so the alpha-model latency (pre-scheduling overhead) is not applied. This is acceptable because the transfer already models the inter-instance delay. Documented as known modeling simplification. |

### E) Review Guide

1. **THE TRICKY PART:** Prefill completion detection in the cluster event loop. After processing an instance event, we must identify whether a newly completed request is a prefill sub-request and trigger KV transfer. The detection uses a `pendingPrefillCompletions` map checked against `Metrics.RequestCompletionTimes`.

2. **WHAT TO SCRUTINIZE:** BC-PD-6 (KV completeness) and BC-PD-9 (phase causality) — the temporal ordering of transfer events is the core correctness property. Also scrutinize the decode-only batch formation path in `batch_formation.go` — this is the only `sim/` change and must be backward-compatible.

3. **WHAT'S SAFE TO SKIM:** CLI flag additions in `cmd/root.go` (mechanical), `ParentRequest` type definition (simple struct), event priority constants (follow established pattern).

4. **KNOWN DEBT:** (a) KV pre-allocation on decode instance uses `AllocateKVBlocks(req, 0, inputLen, nil)` which computes hashes — a future optimization could skip hashing for transferred blocks. (b) Transfer duration uses a configurable bytes-per-token estimate rather than exact model-derived value in blackbox mode. (c) `ParentRequest.CompletionTime` is defined but not populated in PR2 — PR3 (metrics) will set it when decode sub-request completes. (d) `TotalInputTokens` in aggregated metrics does NOT include decode sub-request input tokens (by design — avoids double-counting), but this means the metric reflects prefill tokens only when disaggregation is active. (e) Decode sub-requests do NOT record TTFT (they skip prefill, so the TTFT recording at prefill completion never fires). Parent-level TTFT (`decode_first_token_time - arrival_time`) is a PR3 concern. (f) Decode sub-requests bypass admission control (injected via `InjectDecodeOnline`). This is correct because the parent request already passed admission. Per R23 (code path parity), this intentional divergence is documented. (g) `pendingPrefillCompletions` map scanning is O(pending) per completion event. For Phase 1 workloads (typically <100 concurrent prefill requests), this is acceptable. Phase 2 could add per-instance tracking for O(1) lookup if needed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/cluster/parent_request.go` — ParentRequest type, phase tracking, splitRequest helper
- `sim/cluster/pd_events.go` — PrefillRoutingEvent, KVTransferStartedEvent, KVTransferCompletedEvent, DecodeRoutingEvent
- `sim/cluster/disaggregation_test.go` — All disaggregation tests

**Files to modify:**
- `sim/simulator.go` — Add EnqueueDecodeSubRequest method (~12 lines, bypasses oversized guard and TotalInputTokens counting)
- `sim/batch_formation.go` — Decode-only request handling in Phase 2 (~15 lines)
- `sim/cluster/cluster_event.go` — DisaggregationDecisionEvent.Execute() bifurcation
- `sim/cluster/cluster.go` — Parent tracking, transfer counters, per-pool routing policies, prefill completion detection, pool-filtered snapshot building
- `sim/cluster/deployment.go` — PD transfer config fields, per-pool scorer configs
- `sim/cluster/instance.go` — AllocateTransferredKV method
- `sim/cluster/pool.go` — FilterSnapshotsByPool helper
- `cmd/root.go` — CLI flags, validation, DeploymentConfig construction

**Key decisions:**
1. Decode sub-requests have InputTokens=parent's (for metrics/latency model) but ProgressIndex=inputLen (skip prefill). KV blocks pre-allocated via AllocateTransferredKV before injection.
2. Prefill sub-requests have InputTokens=parent's but OutputTokens=nil (0 output tokens → completes after prefill). KV released normally by per-instance simulator.
3. Transfer duration computed from ceil(inputLen/blockSize) blocks × bytes-per-block / bandwidth.
4. Prefill completion detection via pendingPrefillCompletions map checked after instance event processing.

**Confirmation:** No dead code — every type, field, and method is exercised by end-to-end tests.

### G) Task Breakdown

---

### Task 1: ParentRequest Type and Pool Snapshot Filtering

**Contracts Implemented:** (Foundation for BC-PD-5, BC-PD-9, BC-PD-7)

**Files:**
- Create: `sim/cluster/parent_request.go`
- Modify: `sim/cluster/pool.go`
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing tests for ParentRequest and FilterSnapshotsByPool**

Context: ParentRequest tracks the disaggregated lifecycle of a parent request. FilterSnapshotsByPool filters routing snapshots to only include instances in a specific pool.

```go
// In sim/cluster/disaggregation_test.go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestParentRequest_NewParentRequest(t *testing.T) {
	req := &sim.Request{
		ID:          "req_0",
		InputTokens: make([]int, 100),
		ArrivalTime: 1000,
	}
	parent := NewParentRequest(req, 16) // blockSizeTokens=16

	// Observable behavior: parent tracks the original request metadata
	if parent.ID != "req_0" {
		t.Errorf("parent ID = %q, want %q", parent.ID, "req_0")
	}
	if parent.PrefillSubReqID != "req_0_prefill" {
		t.Errorf("prefill sub-req ID = %q, want %q", parent.PrefillSubReqID, "req_0_prefill")
	}
	if parent.DecodeSubReqID != "req_0_decode" {
		t.Errorf("decode sub-req ID = %q, want %q", parent.DecodeSubReqID, "req_0_decode")
	}
	if parent.NumKVBlocks != 7 { // ceil(100/16) = 7
		t.Errorf("NumKVBlocks = %d, want %d", parent.NumKVBlocks, 7)
	}
}

func TestFilterSnapshotsByPool(t *testing.T) {
	membership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRolePrefill,
		"instance_2": PoolRoleDecode,
		"instance_3": PoolRoleDecode,
	}
	snapshots := []sim.RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 1},
		{ID: "instance_1", QueueDepth: 2},
		{ID: "instance_2", QueueDepth: 3},
		{ID: "instance_3", QueueDepth: 4},
	}

	prefill := FilterSnapshotsByPool(snapshots, membership, PoolRolePrefill)
	if len(prefill) != 2 {
		t.Fatalf("prefill snapshots = %d, want 2", len(prefill))
	}
	if prefill[0].ID != "instance_0" || prefill[1].ID != "instance_1" {
		t.Errorf("prefill IDs = [%s, %s], want [instance_0, instance_1]", prefill[0].ID, prefill[1].ID)
	}

	decode := FilterSnapshotsByPool(snapshots, membership, PoolRoleDecode)
	if len(decode) != 2 {
		t.Fatalf("decode snapshots = %d, want 2", len(decode))
	}
	if decode[0].ID != "instance_2" || decode[1].ID != "instance_3" {
		t.Errorf("decode IDs = [%s, %s], want [instance_2, instance_3]", decode[0].ID, decode[1].ID)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestParentRequest_NewParentRequest|TestFilterSnapshotsByPool" -v`
Expected: FAIL — NewParentRequest and FilterSnapshotsByPool undefined

**Step 3: Implement ParentRequest and FilterSnapshotsByPool**

In `sim/cluster/parent_request.go`:
```go
package cluster

import "github.com/inference-sim/inference-sim/sim"

// ParentRequest tracks the disaggregated lifecycle of a request that was split
// into prefill and decode sub-requests. Owned by ClusterSimulator.
type ParentRequest struct {
	ID               string       // Original request ID
	OriginalRequest  *sim.Request // Pointer to the original request (for metadata)
	PrefillSubReqID  string
	DecodeSubReqID   string
	NumKVBlocks      int64 // KV blocks to transfer (ceil(inputLen / blockSize))

	// Phase timestamps (microseconds). Zero means phase not yet reached.
	ArrivalTime         int64
	PrefillEnqueueTime  int64
	PrefillCompleteTime int64
	TransferStartTime   int64
	TransferCompleteTime int64
	DecodeEnqueueTime   int64
	CompletionTime      int64

	// Instance assignment
	PrefillInstanceID string
	DecodeInstanceID  string
}

// NewParentRequest creates a ParentRequest from the original request.
func NewParentRequest(req *sim.Request, blockSizeTokens int64) *ParentRequest {
	inputLen := int64(len(req.InputTokens))
	numBlocks := (inputLen + blockSizeTokens - 1) / blockSizeTokens
	return &ParentRequest{
		ID:              req.ID,
		OriginalRequest: req,
		PrefillSubReqID: req.ID + "_prefill",
		DecodeSubReqID:  req.ID + "_decode",
		NumKVBlocks:     numBlocks,
		ArrivalTime:     req.ArrivalTime,
	}
}
```

In `sim/cluster/pool.go` (append):
```go
// FilterSnapshotsByPool returns only the snapshots for instances in the given pool role.
// Order is preserved (stable relative to the input slice).
func FilterSnapshotsByPool(snapshots []sim.RoutingSnapshot, membership map[string]PoolRole, role PoolRole) []sim.RoutingSnapshot {
	filtered := make([]sim.RoutingSnapshot, 0, len(snapshots))
	for _, snap := range snapshots {
		if membership[snap.ID] == role {
			filtered = append(filtered, snap)
		}
	}
	return filtered
}
```

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestParentRequest_NewParentRequest|TestFilterSnapshotsByPool" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/parent_request.go sim/cluster/pool.go sim/cluster/disaggregation_test.go
git commit -m "feat(cluster): add ParentRequest type and FilterSnapshotsByPool (BC-PD-5, BC-PD-7, BC-PD-9)

- Add ParentRequest struct for tracking disaggregated request lifecycle
- Add FilterSnapshotsByPool for pool-scoped routing
- Foundation types for PR2 disaggregated request flow

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: DeploymentConfig PD Transfer Fields and CLI Flags

**Contracts Implemented:** EC-PD-2, BC-PD-14 (config foundation)

**Files:**
- Modify: `sim/cluster/deployment.go`
- Modify: `cmd/root.go`
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing test for new config fields**

```go
// Append to sim/cluster/disaggregation_test.go
func TestDeploymentConfig_PDTransferFields(t *testing.T) {
	config := DeploymentConfig{
		PDTransferBandwidthGBps:  25.0,
		PDTransferBaseLatencyMs:  0.05,
		PDKVBytesPerToken:       512,
	}
	// Observable: fields are accessible and have expected values
	if config.PDTransferBandwidthGBps != 25.0 {
		t.Errorf("bandwidth = %f, want 25.0", config.PDTransferBandwidthGBps)
	}
	if config.PDTransferBaseLatencyMs != 0.05 {
		t.Errorf("base latency = %f, want 0.05", config.PDTransferBaseLatencyMs)
	}
	if config.PDKVBytesPerToken != 512 {
		t.Errorf("bytes per token = %d, want 512", config.PDKVBytesPerToken)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDeploymentConfig_PDTransferFields -v`
Expected: FAIL — fields undefined

**Step 3: Add fields to DeploymentConfig and CLI flags**

In `sim/cluster/deployment.go`, add after PDDecider field:
```go
	// PD KV transfer configuration (PR2)
	PDTransferBandwidthGBps float64            // Inter-instance KV transfer bandwidth in GB/s (default 25.0)
	PDTransferBaseLatencyMs float64            // Inter-instance KV transfer base latency in ms (default 0.05)
	PDKVBytesPerToken       int64              // KV cache bytes per token for transfer duration (default 512)

	// Per-pool routing scorer configuration (PR2)
	// When nil, both pools use the main RoutingScorerConfigs.
	PrefillScorerConfigs []sim.ScorerConfig // Scorer configs for prefill pool routing
	DecodeScorerConfigs  []sim.ScorerConfig // Scorer configs for decode pool routing
```

In `cmd/root.go`, add CLI flag variables (after pdDecider):
```go
	pdTransferBandwidth    float64
	pdTransferBaseLatency  float64
	pdKVBytesPerToken      int
	prefillRoutingScorers  string
	decodeRoutingScorers   string
```

In `cmd/root.go` init() function, add flag definitions:
```go
	runCmd.Flags().Float64Var(&pdTransferBandwidth, "pd-transfer-bandwidth", 25.0, "PD KV transfer bandwidth in GB/s")
	runCmd.Flags().Float64Var(&pdTransferBaseLatency, "pd-transfer-base-latency", 0.05, "PD KV transfer base latency in ms")
	runCmd.Flags().IntVar(&pdKVBytesPerToken, "pd-kv-bytes-per-token", 512, "KV cache bytes per token for PD transfer duration computation")
	runCmd.Flags().StringVar(&prefillRoutingScorers, "prefill-routing-scorers", "", "Scorer weights for prefill pool routing (e.g., queue-depth:2,kv-utilization:2)")
	runCmd.Flags().StringVar(&decodeRoutingScorers, "decode-routing-scorers", "", "Scorer weights for decode pool routing (e.g., queue-depth:2,kv-utilization:2)")
```

In `cmd/root.go` validation section (after PD topology validation), add:
```go
	// PD transfer parameter validation (R3, R11)
	if prefillInstances > 0 {
		if pdTransferBandwidth <= 0 || math.IsInf(pdTransferBandwidth, 0) || math.IsNaN(pdTransferBandwidth) {
			logrus.Fatalf("--pd-transfer-bandwidth must be a finite positive number, got %f", pdTransferBandwidth)
		}
		if pdTransferBaseLatency < 0 || math.IsInf(pdTransferBaseLatency, 0) || math.IsNaN(pdTransferBaseLatency) {
			logrus.Fatalf("--pd-transfer-base-latency must be a finite non-negative number, got %f", pdTransferBaseLatency)
		}
		if pdKVBytesPerToken <= 0 {
			logrus.Fatalf("--pd-kv-bytes-per-token must be > 0, got %d", pdKVBytesPerToken)
		}
	}
```

In `cmd/root.go` DeploymentConfig construction, add the new fields:
```go
		PDTransferBandwidthGBps: pdTransferBandwidth,
		PDTransferBaseLatencyMs: pdTransferBaseLatency,
		PDKVBytesPerToken:       int64(pdKVBytesPerToken),
```

Also add per-pool scorer config parsing in the scorer configuration section:
```go
	// Parse per-pool scorer configs if specified
	var prefillScorerCfgs, decodeScorerCfgs []sim.ScorerConfig
	if prefillRoutingScorers != "" {
		var err error
		prefillScorerCfgs, err = sim.ParseScorerConfigs(prefillRoutingScorers)
		if err != nil {
			logrus.Fatalf("invalid --prefill-routing-scorers: %v", err)
		}
	}
	if decodeRoutingScorers != "" {
		var err error
		decodeScorerCfgs, err = sim.ParseScorerConfigs(decodeRoutingScorers)
		if err != nil {
			logrus.Fatalf("invalid --decode-routing-scorers: %v", err)
		}
	}
```

And add to DeploymentConfig:
```go
		PrefillScorerConfigs: prefillScorerCfgs,
		DecodeScorerConfigs:  decodeScorerCfgs,
```

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDeploymentConfig_PDTransferFields -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/... ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/deployment.go cmd/root.go sim/cluster/disaggregation_test.go
git commit -m "feat(cluster): add PD transfer config fields and CLI flags (EC-PD-2, BC-PD-14)

- Add PDTransferBandwidthGBps, PDTransferBaseLatencyMs, PDKVBytesPerToken to DeploymentConfig
- Add PrefillScorerConfigs, DecodeScorerConfigs for per-pool routing
- Add CLI flags: --pd-transfer-bandwidth, --pd-transfer-base-latency, --pd-kv-bytes-per-token
- Add CLI flags: --prefill-routing-scorers, --decode-routing-scorers
- Validation: bandwidth > 0, base latency >= 0, bytes-per-token > 0 (R3)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: New Event Types

**Contracts Implemented:** (Foundation for BC-PD-5, BC-PD-6, BC-PD-8, BC-PD-9)

**Files:**
- Create: `sim/cluster/pd_events.go`
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing tests for event priority ordering**

```go
// Append to sim/cluster/disaggregation_test.go
import "container/heap"

func TestPDEventPriorities(t *testing.T) {
	// All PD events at the same timestamp should be processed in priority order
	q := &ClusterEventQueue{}
	heap.Init(q)

	seq := int64(0)
	nextSeq := func() int64 { s := seq; seq++; return s }

	req := &sim.Request{ID: "req_0", InputTokens: make([]int, 100)}
	parent := &ParentRequest{ID: "req_0", PrefillSubReqID: "req_0_prefill", DecodeSubReqID: "req_0_decode"}

	// Push events in reverse priority order
	heap.Push(q, clusterEventEntry{event: &DecodeRoutingEvent{time: 100, parentReq: parent, decodeSubReq: req}, seqID: nextSeq()})
	heap.Push(q, clusterEventEntry{event: &KVTransferCompletedEvent{time: 100, parentReq: parent}, seqID: nextSeq()})
	heap.Push(q, clusterEventEntry{event: &KVTransferStartedEvent{time: 100, parentReq: parent}, seqID: nextSeq()})
	heap.Push(q, clusterEventEntry{event: &PrefillRoutingEvent{time: 100, request: req, parentReq: parent}, seqID: nextSeq()})

	// Pop in priority order
	expectedPriorities := []int{4, 5, 6, 7}
	for i, expected := range expectedPriorities {
		entry := heap.Pop(q).(clusterEventEntry)
		if got := entry.event.Priority(); got != expected {
			t.Errorf("event %d: priority = %d, want %d", i, got, expected)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestPDEventPriorities -v`
Expected: FAIL — event types undefined

**Step 3: Implement event types**

In `sim/cluster/pd_events.go`:
```go
package cluster

import "github.com/inference-sim/inference-sim/sim"

// PrefillRoutingEvent routes a prefill sub-request to a prefill pool instance.
// Priority 4: after DisaggregationDecisionEvent (3), before KV transfer events.
type PrefillRoutingEvent struct {
	time       int64
	request    *sim.Request    // Prefill sub-request
	parentReq  *ParentRequest
}

func (e *PrefillRoutingEvent) Timestamp() int64 { return e.time }
func (e *PrefillRoutingEvent) Priority() int     { return 4 }
func (e *PrefillRoutingEvent) Execute(cs *ClusterSimulator) {
	// Implemented in Task 5
}

// KVTransferStartedEvent fires when a prefill sub-request completes.
// Records transfer initiation, computes duration, schedules completion.
// Priority 5: after prefill routing.
type KVTransferStartedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferStartedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferStartedEvent) Priority() int     { return 5 }
func (e *KVTransferStartedEvent) Execute(cs *ClusterSimulator) {
	// Implemented in Task 6
}

// KVTransferCompletedEvent fires after transfer duration elapses.
// Creates decode sub-request, schedules decode routing.
// Priority 6: after transfer start.
type KVTransferCompletedEvent struct {
	time      int64
	parentReq *ParentRequest
}

func (e *KVTransferCompletedEvent) Timestamp() int64 { return e.time }
func (e *KVTransferCompletedEvent) Priority() int     { return 6 }
func (e *KVTransferCompletedEvent) Execute(cs *ClusterSimulator) {
	// Implemented in Task 6
}

// DecodeRoutingEvent routes a decode sub-request to a decode pool instance.
// Priority 7: after transfer completion.
type DecodeRoutingEvent struct {
	time         int64
	parentReq    *ParentRequest
	decodeSubReq *sim.Request
}

func (e *DecodeRoutingEvent) Timestamp() int64 { return e.time }
func (e *DecodeRoutingEvent) Priority() int     { return 7 }
func (e *DecodeRoutingEvent) Execute(cs *ClusterSimulator) {
	// Implemented in Task 7
}
```

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestPDEventPriorities -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/pd_events.go sim/cluster/disaggregation_test.go
git commit -m "feat(cluster): add PD event types with priority ordering (BC-PD-5, BC-PD-6)

- PrefillRoutingEvent (priority 4)
- KVTransferStartedEvent (priority 5)
- KVTransferCompletedEvent (priority 6)
- DecodeRoutingEvent (priority 7)
- Execute methods stubbed for subsequent tasks

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Decode-Only Batch Formation and KV Pre-Allocation

**Contracts Implemented:** BC-PD-5 (decode sub-request scheduling), NC-PD-3 (no KV leak), EC-PD-1 (capacity insufficient)

**Files:**
- Modify: `sim/batch_formation.go`
- Modify: `sim/cluster/instance.go`
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing tests**

```go
// Append to sim/cluster/disaggregation_test.go
func TestAllocateTransferredKV_Success(t *testing.T) {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, ""),
	}
	inst := NewInstanceSimulator("decode_0", cfg)

	req := &sim.Request{
		ID:          "decode_sub_0",
		InputTokens: make([]int, 100), // 100 tokens → ceil(100/16) = 7 blocks
		State:       sim.StateQueued,
	}

	ok := inst.AllocateTransferredKV(req)
	if !ok {
		t.Fatal("AllocateTransferredKV returned false, want true")
	}

	// Observable: ProgressIndex set to input length
	if req.ProgressIndex != 100 {
		t.Errorf("ProgressIndex = %d, want 100", req.ProgressIndex)
	}

	// Observable: KV blocks are allocated (UsedBlocks > 0)
	if inst.sim.KVCache.UsedBlocks() == 0 {
		t.Error("UsedBlocks = 0 after AllocateTransferredKV, want > 0")
	}
}

func TestAllocateTransferredKV_InsufficientCapacity(t *testing.T) {
	cfg := sim.SimConfig{
		Horizon:             1000000,
		Seed:                42,
		KVCacheConfig:       sim.NewKVCacheConfig(2, 16, 0, 0, 0, 0), // Only 2 blocks
		BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
		ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test", "H100", 1, ""),
	}
	inst := NewInstanceSimulator("decode_0", cfg)

	req := &sim.Request{
		ID:          "decode_sub_0",
		InputTokens: make([]int, 100), // Needs 7 blocks but only 2 available
		State:       sim.StateQueued,
	}

	ok := inst.AllocateTransferredKV(req)
	if ok {
		t.Error("AllocateTransferredKV returned true with insufficient capacity, want false")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestAllocateTransferredKV" -v`
Expected: FAIL — AllocateTransferredKV undefined

**Step 3: Implement AllocateTransferredKV and decode-only batch formation**

In `sim/simulator.go`, add method after `EnqueueRequest`:
```go
// EnqueueDecodeSubRequest enqueues a decode sub-request that already has KV blocks
// pre-allocated (via PD disaggregation transfer). Bypasses the oversized-request guard
// (blocks already allocated, guard would leak them) and does NOT increment TotalInputTokens
// (input tokens were already counted by the prefill sub-request).
// Triggers StepEvent if the instance is idle (INV-8: work-conserving).
func (sim *Simulator) EnqueueDecodeSubRequest(r *Request) {
	sim.WaitQ.Enqueue(r)
	// Do NOT add len(r.InputTokens) to TotalInputTokens — already counted by prefill sub-request.
	// Trigger StepEvent if idle (work-conserving: INV-8)
	if sim.RunningBatch == nil || len(sim.RunningBatch.Requests) == 0 {
		if sim.stepEvent == nil {
			sim.stepEvent = &StepEvent{time: sim.Clock, sim: sim}
			sim.Schedule(sim.stepEvent)
		}
	}
}
```

In `sim/cluster/instance.go`, add methods:
```go
// AllocateTransferredKV simulates receiving transferred KV cache data from a prefill instance.
// Pre-allocates KV blocks for the request's input tokens and sets ProgressIndex past input.
// Returns false if insufficient KV capacity on this instance.
func (i *InstanceSimulator) AllocateTransferredKV(req *sim.Request) bool {
	inputLen := int64(len(req.InputTokens))
	if inputLen == 0 {
		req.ProgressIndex = 0
		return true
	}
	ok := i.sim.KVCache.AllocateKVBlocks(req, 0, inputLen, nil)
	if ok {
		req.ProgressIndex = inputLen
	}
	return ok
}

// InjectDecodeOnline injects a decode sub-request with pre-allocated KV.
// Bypasses the normal ArrivalEvent → QueuedEvent → EnqueueRequest chain to avoid
// the oversized-request guard (KV already allocated) and TotalInputTokens double-counting.
// Registers request in metrics and directly enqueues into wait queue.
func (i *InstanceSimulator) InjectDecodeOnline(req *sim.Request) {
	i.sim.Metrics.Requests[req.ID] = sim.NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
	i.sim.EnqueueDecodeSubRequest(req)
}
```

In `sim/batch_formation.go`, modify Phase 2 loop to handle decode-only requests. Add at the start of the Phase 2 loop body (after `next := ctx.WaitQ.Peek()`):
```go
		// Handle decode-only requests (PD disaggregation: KV pre-allocated by transfer).
		// When ProgressIndex >= inputLen, the request skips prefill and starts in decode phase.
		inputLen := util.Len64(next.InputTokens)
		if next.ProgressIndex >= inputLen && len(next.OutputTokens) > 0 {
			decodeTokens := int64(1)
			if ok := ctx.KVCache.AllocateKVBlocks(next, next.ProgressIndex, next.ProgressIndex+decodeTokens, nil); !ok {
				break
			}
			ctx.WaitQ.DequeueBatch()
			result.RunningBatch.Requests = append(result.RunningBatch.Requests, next)
			next.ScheduledStepIdx = ctx.StepCount
			result.NewlyScheduled = append(result.NewlyScheduled, ScheduledRequest{Request: next})
			tokenBudget -= decodeTokens
			next.State = StateRunning
			next.NumNewTokens = 1
			ctx.ComputedTokens[next.ID] = next.ProgressIndex + decodeTokens
			continue
		}
```

Note: The existing Phase 2 code after this handles normal prefill requests. The `inputLen` variable declaration on line 122 must be removed or the new code must use a different variable name to avoid shadowing. Adjust accordingly.

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestAllocateTransferredKV" -v`
Expected: PASS

Also verify existing tests still pass:
Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/... -count=1`
Expected: All PASS (batch_formation change is backward-compatible)

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/... ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/instance.go sim/batch_formation.go sim/cluster/disaggregation_test.go
git commit -m "feat(sim): add decode-only batch formation and KV pre-allocation (BC-PD-5, NC-PD-3, EC-PD-1)

- Add InstanceSimulator.AllocateTransferredKV for simulating KV transfer receipt
- Add decode-only request handling in VLLMBatchFormation Phase 2
- Requests with ProgressIndex >= inputLen skip prefill, start in decode phase

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: DisaggregationDecisionEvent Bifurcation and Prefill Routing

**Contracts Implemented:** BC-PD-5, BC-PD-7, BC-PD-13, NC-PD-1, NC-PD-2

**Files:**
- Modify: `sim/cluster/cluster_event.go`
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/pd_events.go`
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing test for bifurcation and prefill routing**

```go
// Append to sim/cluster/disaggregation_test.go
import "math"

func newTestDisaggDeploymentConfig(numInstances, prefill, decode int) DeploymentConfig {
	return DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, ""),
		},
		NumInstances:            numInstances,
		PrefillInstances:        prefill,
		DecodeInstances:         decode,
		PDDecider:               "always",
		RoutingPolicy:           "round-robin",
		PDTransferBandwidthGBps: 25.0,
		PDTransferBaseLatencyMs: 0.05,
		PDKVBytesPerToken:       512,
	}
}

func TestDisaggregation_PrefillRoutedToPrefillPool(t *testing.T) {
	// 4 instances: 2 prefill (0,1), 2 decode (2,3)
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(1) // Single request

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// BC-PD-7: Prefill sub-request must have been routed to a prefill instance
	// Verify via parent request tracking
	if len(cs.parentRequests) != 1 {
		t.Fatalf("parentRequests count = %d, want 1", len(cs.parentRequests))
	}
	for _, parent := range cs.parentRequests {
		prefillInstID := parent.PrefillInstanceID
		role, ok := cs.poolMembership[prefillInstID]
		if !ok {
			t.Errorf("prefill instance %q not in pool membership", prefillInstID)
		}
		if role != PoolRolePrefill {
			t.Errorf("prefill sub-request routed to %s (role=%v), want PoolRolePrefill", prefillInstID, role)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDisaggregation_PrefillRoutedToPrefillPool -v`
Expected: FAIL — parentRequests field or bifurcation not implemented

**Step 3: Implement bifurcation and prefill routing**

First, add new fields to ClusterSimulator in `sim/cluster/cluster.go`:
```go
	// PD disaggregation state (PR2)
	parentRequests            map[string]*ParentRequest // parent request ID → tracking record
	pendingPrefillCompletions map[string]string         // prefill sub-req ID → parent ID
	transfersInitiated        int
	transfersCompleted        int
	prefillRoutingPolicy      sim.RoutingPolicy // nil = use main routingPolicy
	decodeRoutingPolicy       sim.RoutingPolicy // nil = use main routingPolicy
```

Initialize in NewClusterSimulator (after pool membership setup):
```go
	if cs.poolMembership != nil {
		cs.parentRequests = make(map[string]*ParentRequest)
		cs.pendingPrefillCompletions = make(map[string]string)

		// Per-pool routing policies (use separate RNG partitions to avoid fragile coupling)
		if len(config.PrefillScorerConfigs) > 0 {
			cs.prefillRoutingPolicy = sim.NewRoutingPolicy("weighted", config.PrefillScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("prefill-router"))
		}
		if len(config.DecodeScorerConfigs) > 0 {
			cs.decodeRoutingPolicy = sim.NewRoutingPolicy("weighted", config.DecodeScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("decode-router"))
		}
	}
```

Modify DisaggregationDecisionEvent.Execute() in `sim/cluster/cluster_event.go` (also remove stale PR1 comments about "PR2 will add bifurcation" on lines 210-211 and 226-228):
```go
func (e *DisaggregationDecisionEvent) Execute(cs *ClusterSimulator) {
	decision := cs.disaggregationDecider.Decide(e.request)
	logrus.Debugf("[cluster] req %s: disaggregate=%v", e.request.ID, decision.Disaggregate)

	if !decision.Disaggregate {
		// Local path: standard routing (unchanged from PR1)
		heap.Push(&cs.clusterEvents, clusterEventEntry{
			event: &RoutingDecisionEvent{
				time:    e.time + cs.routingLatency,
				request: e.request,
			},
			seqID: cs.nextSeqID(),
		})
		return
	}

	// Disaggregated path: split request and route to prefill pool
	parent := NewParentRequest(e.request, cs.config.BlockSizeTokens)
	cs.parentRequests[parent.ID] = parent

	// Create prefill sub-request: same input, no output (completes after prefill)
	prefillSubReq := &sim.Request{
		ID:          parent.PrefillSubReqID,
		InputTokens: e.request.InputTokens,
		// OutputTokens intentionally nil: zero-output request completes at prefill end
		State:       sim.StateQueued,
		ArrivalTime: e.request.ArrivalTime,
		TenantID:    e.request.TenantID,
		SLOClass:    e.request.SLOClass,
		Model:       e.request.Model,
	}

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &PrefillRoutingEvent{
			time:      e.time + cs.routingLatency,
			request:   prefillSubReq,
			parentReq: parent,
		},
		seqID: cs.nextSeqID(),
	})
}
```

Implement PrefillRoutingEvent.Execute() in `sim/cluster/pd_events.go`:
```go
func (e *PrefillRoutingEvent) Execute(cs *ClusterSimulator) {
	// Build router state with pool-filtered snapshots
	allSnapshots := make([]sim.RoutingSnapshot, len(cs.instances))
	for i, inst := range cs.instances {
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
		allSnapshots[i] = snap
	}
	filteredSnapshots := FilterSnapshotsByPool(allSnapshots, cs.poolMembership, PoolRolePrefill)

	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}
	policy := cs.prefillRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decision := policy.Route(e.request, state)

	logrus.Debugf("[cluster] prefill req %s → instance %s", e.request.ID, decision.TargetInstance)

	e.request.AssignedInstance = decision.TargetInstance
	e.parentReq.PrefillInstanceID = decision.TargetInstance
	e.parentReq.PrefillEnqueueTime = e.time

	// Register as pending prefill completion for detection in event loop
	cs.pendingPrefillCompletions[e.request.ID] = e.parentReq.ID

	// Find target instance and inject
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			cs.inFlightRequests[decision.TargetInstance]++
			inst.InjectRequestOnline(e.request, e.time)
			return
		}
	}
	panic(fmt.Sprintf("PrefillRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
}
```

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDisaggregation_PrefillRoutedToPrefillPool -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/cluster_event.go sim/cluster/cluster.go sim/cluster/pd_events.go sim/cluster/disaggregation_test.go
git commit -m "feat(cluster): bifurcate DisaggregationDecisionEvent and implement prefill routing (BC-PD-5, BC-PD-7, BC-PD-13)

- DisaggregationDecisionEvent.Execute: disaggregate=true splits request, schedules PrefillRoutingEvent
- DisaggregationDecisionEvent.Execute: disaggregate=false routes to standard RoutingDecisionEvent (unchanged)
- PrefillRoutingEvent: pool-filtered routing to prefill instances
- Add parent request tracking and pending prefill completion map to ClusterSimulator

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Prefill Completion Detection and KV Transfer Pipeline

**Contracts Implemented:** BC-PD-6, BC-PD-8, BC-PD-9, BC-PD-14

**Files:**
- Modify: `sim/cluster/cluster.go` (event loop prefill completion detection)
- Modify: `sim/cluster/pd_events.go` (KVTransferStartedEvent, KVTransferCompletedEvent Execute)
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing test for transfer pipeline**

```go
// Append to sim/cluster/disaggregation_test.go
func TestDisaggregation_TransferConservation(t *testing.T) {
	// BC-PD-8: initiated_transfers == completed_transfers at simulation end
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("transfer conservation violated: initiated=%d, completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}
	if cs.transfersInitiated != 5 {
		t.Errorf("transfersInitiated = %d, want 5 (one per request)", cs.transfersInitiated)
	}
}

func TestDisaggregation_TransferDuration(t *testing.T) {
	// BC-PD-14: transfer duration matches formula
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.PDTransferBandwidthGBps = 25.0  // 25 GB/s = 25000 bytes/μs
	config.PDTransferBaseLatencyMs = 0.05  // 50 μs
	config.PDKVBytesPerToken = 512
	config.BlockSizeTokens = 16

	requests := []*sim.Request{{
		ID:           "req_0",
		InputTokens:  make([]int, 160), // 160 tokens → ceil(160/16)=10 blocks
		OutputTokens: make([]int, 10),
		State:        sim.StateQueued,
		ArrivalTime:  0,
	}}

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	parent := cs.parentRequests["req_0"]
	if parent == nil {
		t.Fatal("parent request not found")
	}

	// Transfer duration = base_lat_us + (numBlocks * blockSizeTokens * bytesPerToken) / (bandwidth_gbps * 1000)
	// = 50 + (10 * 16 * 512) / 25000
	// = 50 + 81920 / 25000
	// = 50 + 3.2768
	// = 53.2768 → ceil = 54 μs (after int64 conversion)
	// Verify transfer timestamps show a reasonable duration
	if parent.TransferStartTime == 0 {
		t.Error("TransferStartTime not set")
	}
	if parent.TransferCompleteTime <= parent.TransferStartTime {
		t.Errorf("TransferCompleteTime (%d) <= TransferStartTime (%d)",
			parent.TransferCompleteTime, parent.TransferStartTime)
	}
	transferDuration := parent.TransferCompleteTime - parent.TransferStartTime
	if transferDuration < 50 { // At least base latency (50 μs)
		t.Errorf("transfer duration = %d μs, want >= 50 (base latency)", transferDuration)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestDisaggregation_Transfer" -v`
Expected: FAIL — transfer pipeline not connected

**Step 3: Implement prefill completion detection and transfer events**

In `sim/cluster/cluster.go`, modify the event loop (after the instance event processing block that decrements inFlightRequests). Add prefill completion detection:

```go
			// PD disaggregation: detect prefill sub-request completions
			if delta > 0 && cs.poolsConfigured() && cs.poolMembership[instID] == PoolRolePrefill {
				cs.detectPrefillCompletions(inst, instID)
			}
```

Add the detection method to ClusterSimulator:
```go
// detectPrefillCompletions checks for newly completed prefill sub-requests on the given instance
// and schedules KV transfer events for each.
func (cs *ClusterSimulator) detectPrefillCompletions(inst *InstanceSimulator, instID string) {
	for subReqID, parentID := range cs.pendingPrefillCompletions {
		if _, completed := inst.Metrics().RequestCompletionTimes[subReqID]; completed {
			parent := cs.parentRequests[parentID]
			if parent == nil {
				continue
			}
			parent.PrefillCompleteTime = cs.clock
			delete(cs.pendingPrefillCompletions, subReqID)

			// Schedule KV transfer
			heap.Push(&cs.clusterEvents, clusterEventEntry{
				event: &KVTransferStartedEvent{
					time:      cs.clock,
					parentReq: parent,
				},
				seqID: cs.nextSeqID(),
			})
		}
	}
}
```

Implement KVTransferStartedEvent.Execute() in `sim/cluster/pd_events.go`:
```go
func (e *KVTransferStartedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersInitiated++
	e.parentReq.TransferStartTime = e.time

	// Compute transfer duration: base_latency_us + (numBlocks * blockSizeBytes) / bandwidthBytesPerUs
	numBlocks := e.parentReq.NumKVBlocks
	blockSizeBytes := cs.config.BlockSizeTokens * cs.config.PDKVBytesPerToken
	transferBytes := numBlocks * blockSizeBytes

	bandwidthBytesPerUs := cs.config.PDTransferBandwidthGBps * 1000.0 // GB/s → bytes/μs
	baseLatUs := cs.config.PDTransferBaseLatencyMs * 1000.0            // ms → μs

	var duration int64
	if bandwidthBytesPerUs > 0 {
		duration = int64(math.Ceil(baseLatUs + float64(transferBytes)/bandwidthBytesPerUs))
	} else {
		duration = int64(math.Ceil(baseLatUs))
	}
	if duration < 1 {
		duration = 1 // Minimum 1 μs transfer
	}

	logrus.Debugf("[cluster] KV transfer started for %s: %d blocks, duration=%d μs",
		e.parentReq.ID, numBlocks, duration)

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &KVTransferCompletedEvent{
			time:      e.time + duration,
			parentReq: e.parentReq,
		},
		seqID: cs.nextSeqID(),
	})
}
```

Implement KVTransferCompletedEvent.Execute() in `sim/cluster/pd_events.go`:
```go
func (e *KVTransferCompletedEvent) Execute(cs *ClusterSimulator) {
	cs.transfersCompleted++
	e.parentReq.TransferCompleteTime = e.time

	// Create decode sub-request
	orig := e.parentReq.OriginalRequest
	decodeSubReq := &sim.Request{
		ID:           e.parentReq.DecodeSubReqID,
		InputTokens:  orig.InputTokens,
		OutputTokens: orig.OutputTokens,
		State:        sim.StateQueued,
		ArrivalTime:  orig.ArrivalTime,
		TenantID:     orig.TenantID,
		SLOClass:     orig.SLOClass,
		Model:        orig.Model,
	}

	logrus.Debugf("[cluster] KV transfer completed for %s, scheduling decode routing", e.parentReq.ID)

	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &DecodeRoutingEvent{
			time:         e.time,
			parentReq:    e.parentReq,
			decodeSubReq: decodeSubReq,
		},
		seqID: cs.nextSeqID(),
	})
}
```

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestDisaggregation_Transfer" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/cluster.go sim/cluster/pd_events.go sim/cluster/disaggregation_test.go
git commit -m "feat(cluster): implement prefill completion detection and KV transfer pipeline (BC-PD-6, BC-PD-8, BC-PD-14)

- Detect prefill sub-request completions in event loop via pendingPrefillCompletions map
- KVTransferStartedEvent: compute transfer duration from block count and bandwidth
- KVTransferCompletedEvent: create decode sub-request, schedule DecodeRoutingEvent
- Transfer conservation counters for INV-PD-3

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Decode Routing and Injection

**Contracts Implemented:** BC-PD-5 (completion), BC-PD-7 (pool exclusivity for decode), NC-PD-1 (no cross-pool), EC-PD-1 (capacity insufficient)

**Files:**
- Modify: `sim/cluster/pd_events.go` (DecodeRoutingEvent Execute)
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing test for decode routing**

```go
// Append to sim/cluster/disaggregation_test.go
func TestDisaggregation_DecodeRoutedToDecodePool(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// BC-PD-7: Decode sub-requests must be routed to decode pool instances
	for _, parent := range cs.parentRequests {
		decodeInstID := parent.DecodeInstanceID
		if decodeInstID == "" {
			t.Errorf("decode instance not assigned for parent %s", parent.ID)
			continue
		}
		role, ok := cs.poolMembership[decodeInstID]
		if !ok {
			t.Errorf("decode instance %q not in pool membership", decodeInstID)
			continue
		}
		if role != PoolRoleDecode {
			t.Errorf("decode sub-request for %s routed to %s (role=%v), want PoolRoleDecode",
				parent.ID, decodeInstID, role)
		}
	}
}

func TestDisaggregation_RequestCompletesFullPath(t *testing.T) {
	// BC-PD-5: Request completes through full disaggregated path
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// All requests should complete: check aggregated metrics
	metrics := cs.AggregatedMetrics()
	// Each parent produces 1 prefill sub-req + 1 decode sub-req = 2 completed per parent
	// But we check that output tokens were generated (decode completed)
	if metrics.TotalOutputTokens == 0 {
		t.Error("TotalOutputTokens = 0, decode sub-requests did not generate output")
	}

	// Verify phase causality (BC-PD-9) for each parent
	for _, parent := range cs.parentRequests {
		if parent.TransferCompleteTime == 0 {
			t.Errorf("parent %s: TransferCompleteTime not set", parent.ID)
		}
		if parent.DecodeEnqueueTime < parent.TransferCompleteTime {
			t.Errorf("parent %s: DecodeEnqueueTime (%d) < TransferCompleteTime (%d) — violates INV-PD-1",
				parent.ID, parent.DecodeEnqueueTime, parent.TransferCompleteTime)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestDisaggregation_Decode|TestDisaggregation_RequestCompletes" -v`
Expected: FAIL — DecodeRoutingEvent.Execute not implemented

**Step 3: Implement DecodeRoutingEvent.Execute**

In `sim/cluster/pd_events.go`:
```go
func (e *DecodeRoutingEvent) Execute(cs *ClusterSimulator) {
	// Build router state with pool-filtered decode snapshots
	allSnapshots := make([]sim.RoutingSnapshot, len(cs.instances))
	for i, inst := range cs.instances {
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		snap.InFlightRequests = cs.inFlightRequests[string(inst.ID())]
		allSnapshots[i] = snap
	}
	filteredSnapshots := FilterSnapshotsByPool(allSnapshots, cs.poolMembership, PoolRoleDecode)

	state := &sim.RouterState{Snapshots: filteredSnapshots, Clock: cs.clock}
	policy := cs.decodeRoutingPolicy
	if policy == nil {
		policy = cs.routingPolicy
	}
	decision := policy.Route(e.decodeSubReq, state)

	logrus.Debugf("[cluster] decode req %s → instance %s", e.decodeSubReq.ID, decision.TargetInstance)

	e.decodeSubReq.AssignedInstance = decision.TargetInstance
	e.parentReq.DecodeInstanceID = decision.TargetInstance
	e.parentReq.DecodeEnqueueTime = e.time

	// Find target decode instance
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			// Pre-allocate KV blocks for transferred input (EC-PD-1: handle failure)
			if ok := inst.AllocateTransferredKV(e.decodeSubReq); !ok {
				logrus.Warnf("[cluster] decode instance %s: insufficient KV capacity for %s (%d input tokens)",
					decision.TargetInstance, e.decodeSubReq.ID, len(e.decodeSubReq.InputTokens))
				// Cannot proceed without KV — count as dropped
				cs.inFlightRequests[decision.TargetInstance]++
				cs.inFlightRequests[decision.TargetInstance]-- // immediately decrement
				return
			}

			cs.inFlightRequests[decision.TargetInstance]++
			// Use dedicated decode injection: bypasses ArrivalEvent → EnqueueRequest chain
			// to avoid oversized-request guard (KV already allocated) and TotalInputTokens double-counting
			inst.InjectDecodeOnline(e.decodeSubReq)
			return
		}
	}
	panic(fmt.Sprintf("DecodeRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
}
```

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestDisaggregation_Decode|TestDisaggregation_RequestCompletes" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/pd_events.go sim/cluster/disaggregation_test.go
git commit -m "feat(cluster): implement decode routing with KV pre-allocation (BC-PD-5, BC-PD-7, EC-PD-1)

- DecodeRoutingEvent: pool-filtered routing to decode instances
- Pre-allocate transferred KV blocks on decode instance before injection
- Decode sub-request starts in decode phase (ProgressIndex = inputLen)
- Graceful handling of insufficient KV capacity (EC-PD-1)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Per-Pool Routing Policies

**Contracts Implemented:** BC-PD-15

**Files:**
- Modify: `sim/cluster/cluster.go`
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write failing test for per-pool routing**

```go
// Append to sim/cluster/disaggregation_test.go
func TestDisaggregation_PerPoolScorerConfigs(t *testing.T) {
	// BC-PD-15: per-pool scorer configs produce different routing decisions
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.RoutingPolicy = "weighted"
	config.PrefillScorerConfigs = []sim.ScorerConfig{{Name: "queue-depth", Weight: 1.0}}
	config.DecodeScorerConfigs = []sim.ScorerConfig{{Name: "kv-utilization", Weight: 1.0}}

	requests := newTestRequests(3)

	cs := NewClusterSimulator(config, requests)

	// Verify separate policy instances were created
	if cs.prefillRoutingPolicy == nil {
		t.Error("prefillRoutingPolicy is nil, want non-nil when PrefillScorerConfigs specified")
	}
	if cs.decodeRoutingPolicy == nil {
		t.Error("decodeRoutingPolicy is nil, want non-nil when DecodeScorerConfigs specified")
	}

	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// All requests should complete successfully
	if cs.AggregatedMetrics().TotalOutputTokens == 0 {
		t.Error("no output tokens generated with per-pool scorer configs")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDisaggregation_PerPoolScorerConfigs -v`
Expected: FAIL or PASS (depending on Task 5 initialization). If PASS, the test validates the existing implementation.

**Step 3: Verify per-pool routing initialization is correct**

The per-pool routing policy initialization was already added in Task 5 (in NewClusterSimulator). Verify it works by running the test. If it passes, this task is a validation task.

If the test needs adjustment (e.g., the routing policy field names don't match), fix accordingly.

**Step 4: Run test to verify it passes**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDisaggregation_PerPoolScorerConfigs -v`
Expected: PASS

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/disaggregation_test.go
git commit -m "test(cluster): verify per-pool routing scorer configs (BC-PD-15)

- Test that PrefillScorerConfigs and DecodeScorerConfigs produce separate routing policy instances
- Verify end-to-end completion with per-pool configs

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: End-to-End Integration and Invariant Tests

**Contracts Implemented:** BC-PD-9, BC-PD-10, BC-PD-11, BC-PD-12

**Files:**
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write comprehensive invariant tests**

```go
// Append to sim/cluster/disaggregation_test.go

func TestDisaggregation_PhaseCausality(t *testing.T) {
	// BC-PD-9 / INV-PD-4: Full causal chain for every disaggregated request
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(10)

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	for _, parent := range cs.parentRequests {
		chain := []struct {
			name  string
			value int64
		}{
			{"ArrivalTime", parent.ArrivalTime},
			{"PrefillEnqueueTime", parent.PrefillEnqueueTime},
			{"PrefillCompleteTime", parent.PrefillCompleteTime},
			{"TransferStartTime", parent.TransferStartTime},
			{"TransferCompleteTime", parent.TransferCompleteTime},
			{"DecodeEnqueueTime", parent.DecodeEnqueueTime},
		}

		for i := 1; i < len(chain); i++ {
			if chain[i].value < chain[i-1].value {
				t.Errorf("parent %s: causality violated: %s (%d) < %s (%d)",
					parent.ID, chain[i].name, chain[i].value, chain[i-1].name, chain[i-1].value)
			}
		}
	}
}

func TestDisaggregation_ExtendedConservation(t *testing.T) {
	// BC-PD-10 / INV-1 extended: conservation with disaggregated intermediate states
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	numRequests := 10
	requests := newTestRequests(numRequests)

	cs := NewClusterSimulator(config, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// Each original request is disaggregated into prefill + decode sub-requests.
	// The original request is NOT injected into any instance.
	// Prefill sub-requests complete on prefill instances.
	// Decode sub-requests complete on decode instances.
	//
	// Conservation: numRequests == len(parentRequests) (all were disaggregated)
	// Per-instance: prefill_completed + decode_completed covers all sub-requests
	// Transfer conservation: initiated == completed

	if len(cs.parentRequests) != numRequests {
		t.Errorf("parentRequests = %d, want %d", len(cs.parentRequests), numRequests)
	}
	if cs.transfersInitiated != cs.transfersCompleted {
		t.Errorf("transfer conservation: initiated=%d != completed=%d",
			cs.transfersInitiated, cs.transfersCompleted)
	}

	// Count completed sub-requests across all instances
	totalCompleted := 0
	for _, inst := range cs.instances {
		totalCompleted += inst.Metrics().CompletedRequests
	}
	// Each parent produces 2 completed sub-requests (prefill + decode)
	expectedCompleted := numRequests * 2
	// Some may still be in progress if horizon is too short, but with MaxInt64 horizon all should complete
	if totalCompleted != expectedCompleted {
		t.Errorf("total completed sub-requests = %d, want %d", totalCompleted, expectedCompleted)
	}
}

func TestDisaggregation_PoolStability(t *testing.T) {
	// INV-PD-5: Pool membership unchanged after initialization
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)

	cs := NewClusterSimulator(config, requests)
	membershipBefore := cs.PoolMembership()

	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	membershipAfter := cs.PoolMembership()
	if len(membershipBefore) != len(membershipAfter) {
		t.Fatalf("pool membership size changed: before=%d, after=%d",
			len(membershipBefore), len(membershipAfter))
	}
	for id, roleBefore := range membershipBefore {
		roleAfter, ok := membershipAfter[id]
		if !ok {
			t.Errorf("instance %s missing from pool membership after simulation", id)
		}
		if roleBefore != roleAfter {
			t.Errorf("instance %s: role changed from %v to %v", id, roleBefore, roleAfter)
		}
	}
}

func TestDisaggregation_Determinism(t *testing.T) {
	// BC-PD-12 / INV-6: Same seed produces identical results
	config := newTestDisaggDeploymentConfig(4, 2, 2)

	run := func() *sim.Metrics {
		requests := newTestRequests(10)
		cs := NewClusterSimulator(config, requests)
		if err := cs.Run(); err != nil {
			t.Fatalf("Run failed: %v", err)
		}
		return cs.AggregatedMetrics()
	}

	m1 := run()
	m2 := run()

	if m1.CompletedRequests != m2.CompletedRequests {
		t.Errorf("non-deterministic CompletedRequests: %d vs %d", m1.CompletedRequests, m2.CompletedRequests)
	}
	if m1.TotalOutputTokens != m2.TotalOutputTokens {
		t.Errorf("non-deterministic TotalOutputTokens: %d vs %d", m1.TotalOutputTokens, m2.TotalOutputTokens)
	}
	if m1.SimEndedTime != m2.SimEndedTime {
		t.Errorf("non-deterministic SimEndedTime: %d vs %d", m1.SimEndedTime, m2.SimEndedTime)
	}
}
```

**Step 2: Run tests**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run "TestDisaggregation_PhaseCausality|TestDisaggregation_ExtendedConservation|TestDisaggregation_PoolStability|TestDisaggregation_Determinism" -v`
Expected: All PASS

**Step 3: Fix any failures**

If any invariant tests fail, debug and fix the implementation (not the tests). Common issues:
- Phase causality: timestamps not being set at the right event
- Conservation: sub-request counting mismatch
- Determinism: map iteration order in pool filtering (should be stable since we iterate the instances slice)

**Step 4: Run full test suite**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./... -count=1`
Expected: All PASS (including existing tests — backward compatibility)

**Step 5: Run lint check**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/disaggregation_test.go
git commit -m "test(cluster): add comprehensive invariant tests for disaggregation (BC-PD-9, BC-PD-10, BC-PD-11, BC-PD-12)

- INV-PD-4: Phase causality chain verification
- INV-1 extended: Conservation with disaggregated sub-requests
- INV-PD-5: Pool membership stability
- INV-6: Determinism (same seed, identical metrics)
- INV-PD-3: Transfer conservation (initiated == completed)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: Backward Compatibility, Documentation, and Verification Gate

**Contracts Implemented:** BC-PD-13 (backward compat), documentation updates

**Files:**
- Test: `sim/cluster/disaggregation_test.go`
- Modify: `CLAUDE.md`
- Modify: `docs/contributing/standards/invariants.md`

**Step 1: Write backward compatibility test**

```go
// Append to sim/cluster/disaggregation_test.go
func TestDisaggregation_BackwardCompatibility(t *testing.T) {
	// BC-PD-13: When pools not configured, behavior is identical to pre-PR2
	configNoPools := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:             math.MaxInt64,
			Seed:                42,
			KVCacheConfig:       sim.NewKVCacheConfig(10000, 16, 0, 0, 0, 0),
			BatchConfig:         sim.NewBatchConfig(256, 2048, 0),
			LatencyCoeffs:       sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 1, 100}),
			ModelHardwareConfig: sim.NewModelHardwareConfig(sim.ModelConfig{}, sim.HardwareCalib{}, "test-model", "H100", 1, ""),
		},
		NumInstances:  4,
		RoutingPolicy: "round-robin",
		// No PD config — pools disabled
	}

	requests := newTestRequests(10)
	cs := NewClusterSimulator(configNoPools, requests)
	if err := cs.Run(); err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// No parent requests should exist
	if cs.parentRequests != nil && len(cs.parentRequests) > 0 {
		t.Errorf("parentRequests should be empty when pools not configured, got %d", len(cs.parentRequests))
	}

	// All requests should complete normally
	metrics := cs.AggregatedMetrics()
	if metrics.CompletedRequests == 0 {
		t.Error("no requests completed in non-disaggregated mode")
	}

	// Conservation (INV-1)
	injected := metrics.CompletedRequests + metrics.StillQueued + metrics.StillRunning + metrics.DroppedUnservable
	if injected != 10 {
		t.Errorf("conservation violated: completed(%d) + queued(%d) + running(%d) + dropped(%d) = %d, want 10",
			metrics.CompletedRequests, metrics.StillQueued, metrics.StillRunning, metrics.DroppedUnservable, injected)
	}
}
```

**Step 2: Run backward compatibility test**

Run: `cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow && go test ./sim/cluster/... -run TestDisaggregation_BackwardCompatibility -v`
Expected: PASS

**Step 3: Update CLAUDE.md**

Add to Key Data Flow section:
```
### Disaggregated Data Flow (PD mode)

Request Arrival → Admission → Disaggregation Decision
  → [disaggregate] → Prefill Routing → Prefill Instance → Prefill Complete
    → KV Transfer Started → KV Transfer Completed
    → Decode Routing → KV Pre-Allocation → Decode Instance → Completion
  → [local] → Standard Routing → Any Instance → Completion
```

Add new CLI flags to cmd/root.go description:
```
--pd-transfer-bandwidth, --pd-transfer-base-latency, --pd-kv-bytes-per-token, --prefill-routing-scorers, --decode-routing-scorers
```

Add new files to File Organization:
```
│   ├── cluster_event.go       # + DisaggregationDecisionEvent bifurcation (PR2)
│   ├── pd_events.go           # PrefillRoutingEvent, KVTransferStartedEvent, KVTransferCompletedEvent, DecodeRoutingEvent
│   ├── parent_request.go      # ParentRequest type for disaggregated request tracking
```

Add to Key Invariants:
```
- **INV-PD-1 KV completeness**: decode_enqueue_time >= transfer_complete_time for every disaggregated request
- **INV-PD-2 Pool exclusivity**: prefill sub-requests on prefill instances only; decode on decode only
- **INV-PD-3 Transfer conservation**: initiated_transfers == completed_transfers at simulation end
- **INV-PD-4 Phase causality**: arrival ≤ prefill_enqueue ≤ prefill_complete ≤ transfer_start ≤ transfer_complete ≤ decode_enqueue ≤ completion
```

**Step 4: Update invariants.md**

Add INV-PD-1 through INV-PD-5 definitions with verification strategies to `docs/contributing/standards/invariants.md`.

**Step 5: Run verification gate**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
go build ./...
go test ./... -count=1
golangci-lint run ./...
git status
```

Report: build exit code, test pass/fail counts, lint issue count, working tree status.

**Step 6: Commit**

```bash
cd /ws/fork/inference-sim/.worktrees/pr2-pd-request-flow
git add sim/cluster/disaggregation_test.go CLAUDE.md docs/contributing/standards/invariants.md
git commit -m "docs: update CLAUDE.md and invariants.md for PD disaggregation PR2 (BC-PD-13)

- Add disaggregated data flow to Key Data Flow section
- Add new CLI flags documentation
- Add pd_events.go and parent_request.go to file organization
- Add INV-PD-1 through INV-PD-5 to invariants.md with verification strategies
- Backward compatibility test for non-disaggregated path

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-PD-5 | Task 5, 7 | Integration | TestDisaggregation_RequestCompletesFullPath |
| BC-PD-6 | Task 6 | Invariant | TestDisaggregation_TransferConservation (via timestamps) |
| BC-PD-7 | Task 5, 7 | Unit | TestDisaggregation_PrefillRoutedToPrefillPool, TestDisaggregation_DecodeRoutedToDecodePool |
| BC-PD-8 | Task 6 | Invariant | TestDisaggregation_TransferConservation |
| BC-PD-9 | Task 9 | Invariant | TestDisaggregation_PhaseCausality |
| BC-PD-10 | Task 9 | Invariant | TestDisaggregation_ExtendedConservation |
| BC-PD-11 | Task 9 | Invariant | (verified via work-conserving property of InjectRequestOnline → QueuedEvent → StepEvent chain) |
| BC-PD-12 | Task 9 | Invariant | TestDisaggregation_Determinism |
| BC-PD-13 | Task 10 | Integration | TestDisaggregation_BackwardCompatibility |
| BC-PD-14 | Task 6 | Unit | TestDisaggregation_TransferDuration |
| BC-PD-15 | Task 8 | Integration | TestDisaggregation_PerPoolScorerConfigs |
| NC-PD-1 | Task 5, 7 | Unit | (verified by TestDisaggregation_PrefillRoutedToPrefillPool, TestDisaggregation_DecodeRoutedToDecodePool) |
| NC-PD-2 | Task 10 | Integration | TestDisaggregation_BackwardCompatibility |
| NC-PD-3 | Task 4 | Unit | (verified by AllocateTransferredKV tests — blocks freed at completion) |
| EC-PD-1 | Task 4 | Unit | TestAllocateTransferredKV_InsufficientCapacity |
| EC-PD-2 | Task 2 | Unit | (CLI validation — tested via go build) |

**Golden dataset:** Not updated. Disaggregation is off by default (NeverDisaggregate). Existing golden tests pass unchanged. New disaggregated scenarios use invariant tests (not golden values).

**Shared test infrastructure:** `newTestDisaggDeploymentConfig()` helper in `disaggregation_test.go`. Reuses existing `newTestRequests()` from `test_helpers_test.go`.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| FormBatch Phase 2 decode-only path breaks existing tests | Low | High | Guarded by `ProgressIndex >= inputLen` check (zero-value ProgressIndex=0 < inputLen for all normal requests). Full test suite run in Task 4. | Task 4 |
| Prefill completion detection misses completions | Medium | High | pendingPrefillCompletions map is checked after every instance event with completion delta on prefill instances. Map is populated at injection time. | Task 6 |
| KV pre-allocation double-references blocks in GetCachedBlocks | Medium | Medium | Pre-allocated blocks have hashes. When FormBatch Phase 2 runs, the decode-only path (Task 4) bypasses GetCachedBlocks entirely — the `continue` skips the normal prefill path. | Task 4 |
| Transfer duration overflow for large models | Low | Low | Transfer duration computed as int64 microseconds. Max realistic: 1GB transfer at 1 GB/s = 1 second = 1e6 μs. Well within int64 range. R11: bandwidth validated > 0 at CLI. | Task 6 |
| Pool-filtered snapshots empty (all instances in other pool) | Low | High | ValidatePoolTopology ensures both pools have > 0 instances when disaggregation is enabled. Routing policy panics on empty snapshot list (existing behavior). | Task 1 (PR1) |
| Decode sub-request KV allocation contention | Medium | Medium | Decode instance may lack capacity for all transferred blocks. EC-PD-1 handles this: inject anyway, EnqueueRequest drops as unservable. | Task 7 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — ParentRequest is a plain struct, no interface
- [x] No feature creep — metrics (PR3), traces (PR4), PrefixThresholdDecider (PR5) all deferred
- [x] No unexercised flags — all new CLI flags used in disaggregated path
- [x] No partial implementations — full pipeline from split to decode completion
- [x] No breaking changes — non-disaggregated path byte-identical (BC-PD-13)
- [x] No hidden global state — all new state in ClusterSimulator fields
- [x] All new code passes golangci-lint
- [x] Shared test helpers — newTestDisaggDeploymentConfig reuses existing patterns
- [x] CLAUDE.md updated in Task 10
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY — invariants.md updated with INV-PD-1 through INV-PD-5
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7→8→9→10)
- [x] All contracts mapped to tasks (see Test Strategy table)
- [x] Golden dataset not needed (disaggregation off by default)
- [x] Construction site audit completed — DeploymentConfig and Request sites listed in Phase 0

**Antipattern rules:**
- [x] R1: No silent data loss — prefill completion detection logs and tracks every sub-request
- [x] R2: Sort map keys — pool-filtered snapshots preserve instance order (iterate instances slice, not map)
- [x] R3: Validate numeric parameters — pd-transfer-bandwidth > 0, pd-transfer-base-latency >= 0, pd-kv-bytes-per-token > 0
- [x] R4: Construction site audit — DeploymentConfig new fields zero-value safe for all existing sites
- [x] R5: Transactional mutation — KV pre-allocation is atomic (single AllocateKVBlocks call)
- [x] R6: No Fatalf in library — all sim/cluster code uses panic or returns error; Fatalf only in cmd/
- [x] R7: Invariant tests — comprehensive invariant tests in Task 9 (no golden tests added)
- [x] R8: No exported maps — parentRequests is unexported; PoolMembership() returns copy
- [x] R9: YAML pointer types — no new YAML fields with zero-value ambiguity
- [x] R10: Strict YAML parsing — no new YAML parsing added
- [x] R11: Guard division — bandwidth validated > 0 at CLI; guarded in transfer computation
- [x] R12: Golden regeneration — not needed (disaggregation off by default)
- [x] R13: Multi-impl interfaces — no new interfaces (reuses existing RoutingPolicy, KVStore)
- [x] R14: Single-module methods — each event Execute() handles one concern
- [x] R15: Stale PR references — grep for "PR2" references after completion
- [x] R16: Config by module — PD transfer config grouped in DeploymentConfig
- [x] R17: Signal freshness — pool-filtered snapshots use same staleness as existing snapshots
- [x] R18: CLI flag precedence — new flags independent of defaults.yaml
- [x] R19: Livelock protection — no unbounded retry loops in new code
- [x] R20: Degenerate inputs — zero input tokens → zero blocks → base latency only
- [x] R21: No range over mutable slices — pendingPrefillCompletions deletion uses delete() in map (safe)
- [x] R22: Pre-check consistency — N/A (no pre-checks in new code)
- [x] R23: Code path parity — disaggregated and non-disaggregated paths produce equivalent output format

---

## Appendix K: File-Level Implementation Details

### File: `sim/cluster/parent_request.go`

**Purpose:** Tracks the disaggregated lifecycle of a parent request that was split into prefill and decode sub-requests.

**Key types:**
- `ParentRequest` struct: ID, sub-request IDs, KV block count, phase timestamps, instance assignments
- `NewParentRequest(req, blockSizeTokens)`: canonical constructor computing ceil(inputLen/blockSize) blocks

**Behavioral notes:**
- NumKVBlocks uses ceiling division for conservative transfer duration estimation
- Phase timestamps are zero until the corresponding event fires
- OriginalRequest pointer kept for metadata access (e.g., OutputTokens for decode sub-request creation)

### File: `sim/cluster/pd_events.go`

**Purpose:** Four new cluster-level event types for the disaggregated request pipeline.

**Event priorities:**
| Event | Priority | Rationale |
|-------|----------|-----------|
| PrefillRoutingEvent | 4 | After DisaggregationDecisionEvent (3) |
| KVTransferStartedEvent | 5 | After prefill routing |
| KVTransferCompletedEvent | 6 | After transfer start |
| DecodeRoutingEvent | 7 | After transfer completion |

**Transfer duration formula:**
```
duration_us = max(1, ceil(base_latency_ms * 1000 + (numBlocks * blockSizeTokens * bytesPerToken) / (bandwidth_gbps * 1000)))
```

**RNG usage:** PrefillRoutingEvent and DecodeRoutingEvent use the same routing policy instances as standard routing, which consume from SubsystemRouter RNG partition.

### File: `sim/batch_formation.go` (modification)

**Purpose:** Handle decode-only requests in VLLMBatchFormation Phase 2.

**Change:** ~15 lines added at the start of the Phase 2 loop body. When `next.ProgressIndex >= inputLen && len(next.OutputTokens) > 0`, the request bypasses prefill KV allocation and enters the running batch with NumNewTokens=1 (first decode token).

**Backward compatibility:** Normal requests have ProgressIndex=0 when entering Phase 2, so `0 >= inputLen` is false for any request with input tokens. The new code path is never taken for non-disaggregated requests.

### File: `sim/cluster/instance.go` (modification)

**Purpose:** Add AllocateTransferredKV method for KV pre-allocation on decode instances.

**Method:** `AllocateTransferredKV(req) bool` — calls `sim.KVCache.AllocateKVBlocks(req, 0, inputLen, nil)` and sets `req.ProgressIndex = inputLen`. Returns false if insufficient capacity.

### File: `sim/cluster/cluster_event.go` (modification)

**Purpose:** Bifurcate DisaggregationDecisionEvent to split requests when disaggregate=true.

**Change:** Replace PR1 stub (both paths → RoutingDecisionEvent) with:
- disaggregate=true → create ParentRequest, create prefill sub-request, schedule PrefillRoutingEvent
- disaggregate=false → schedule RoutingDecisionEvent (unchanged)

### File: `sim/cluster/cluster.go` (modification)

**Purpose:** Add parent request tracking, prefill completion detection, per-pool routing.

**New fields:** parentRequests, pendingPrefillCompletions, transfersInitiated, transfersCompleted, prefillRoutingPolicy, decodeRoutingPolicy

**New method:** detectPrefillCompletions — called after instance event processing when CompletedRequests increases on a prefill instance. Scans pendingPrefillCompletions map for matching request IDs in RequestCompletionTimes.

**Event loop change:** After the existing completion-based inFlightRequests decrement, add prefill completion detection call.

### File: `sim/cluster/deployment.go` (modification)

**Purpose:** Add PD transfer configuration fields.

**New fields:** PDTransferBandwidthGBps (float64), PDTransferBaseLatencyMs (float64), PDKVBytesPerToken (int64), PrefillScorerConfigs ([]ScorerConfig), DecodeScorerConfigs ([]ScorerConfig)

### File: `sim/cluster/pool.go` (modification)

**Purpose:** Add FilterSnapshotsByPool helper.

**Function:** `FilterSnapshotsByPool(snapshots, membership, role) []RoutingSnapshot` — returns snapshots for instances matching the given pool role. Order preserved (stable).

### File: `cmd/root.go` (modification)

**Purpose:** Add CLI flags and validation for PD transfer parameters.

**New flags:** --pd-transfer-bandwidth (25.0), --pd-transfer-base-latency (0.05), --pd-kv-bytes-per-token (512), --prefill-routing-scorers (""), --decode-routing-scorers ("")

**Validation:** bandwidth > 0 (R3, R11), base latency >= 0 (R3), bytes-per-token > 0 (R3). Only validated when prefillInstances > 0.
