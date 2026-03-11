# PR3 Disaggregation-Aware Metrics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expose disaggregation-specific metrics — parent-level TTFT, KV transfer latency, and per-pool throughput — when a simulation runs with prefill/decode disaggregation enabled.

**The problem today:** When running a disaggregated simulation (separate prefill and decode pools), the existing metrics report only instance-level TTFT from the decode sub-request's perspective and aggregate throughput across all instances. Operators cannot see the transfer overhead, the true end-to-end TTFT from the original request's arrival, or how load is distributed between the prefill and decode pools. This makes it impossible to reason about transfer bottlenecks or pool imbalance.

**What this PR adds:**
1. **Parent TTFT distribution** — the elapsed time from each original request's arrival to its first decode token, reported as P50/P95/P99/mean/min/max in the `=== PD Metrics ===` output section.
2. **KV transfer duration distribution** — time spent transferring KV blocks from prefill to decode instance per request, showing transfer overhead.
3. **Per-pool throughput** — completed prefill sub-requests/sec and decode sub-requests/sec, so operators can identify pool bottlenecks.
4. **Load imbalance ratio** — max(prefill_rps, decode_rps) / min(prefill_rps, decode_rps); 1.0 = perfectly balanced; `inf (one pool idle)` = extreme imbalance (one pool has zero completions).
5. **Backward compatibility** — when disaggregation is disabled, the existing output is unchanged and `RawMetrics.PD` is nil.

**Why this matters:** PR3 closes the observability gap for PD disaggregation introduced in PR2, and unblocks PR4 (autoscaling) and PR5 (hypothesis experimentation) which both depend on actionable per-pool metrics.

**Architecture:** A new `CollectPDMetrics` function in `sim/cluster/pd_metrics.go` accepts `[]*ParentRequest`, the aggregated `*sim.Metrics`, a `map[string]PoolRole` pool membership map, and a `map[string]*sim.Metrics` per-instance metrics map. It returns `*PDMetrics` (nil when no disaggregation). `RawMetrics` gains a `PD *PDMetrics` field (nil-safe). Two new accessors on `ClusterSimulator` — `ParentRequests() []*ParentRequest` and `PerInstanceMetricsByID() map[string]*sim.Metrics` — bridge the cluster state to the metrics collector. `cmd/root.go` calls `CollectPDMetrics` and prints the `=== PD Metrics ===` section when PD is active.

**Source:** Issue #593 (PR3 in macro plan comment on issue #577), `docs/plans/pd-disaggregation-macro-plan.md`

**Closes:** Fixes #593

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

PR3 adds a new `pd_metrics.go` module to `sim/cluster/` that computes disaggregation-aware metrics from `ParentRequest` lifecycle timestamps and per-instance completion counts. It slots in after `CollectRawMetrics` in `cmd/root.go` and prints a new guarded output section.

**System position:** PR2 (merged) captures `ParentRequest` lifecycle data. PR3 reads that data to produce metrics. PR4 (future) will consume `PDMetrics` for autoscaling signals.

**Adjacent blocks:** `sim/cluster/cluster.go` (new accessors), `sim/cluster/metrics.go` (extends `RawMetrics`), `cmd/root.go` (output wiring), `sim/metrics.go` (source of `RequestTTFTs`).

**Key insight (DEVIATION flag):** Parent TTFT requires no new arithmetic. Because `KVTransferCompletedEvent.Execute` (pd_events.go:118-127) creates the decode sub-request with `ArrivalTime = orig.ArrivalTime`, the base simulator's TTFT recording at `sim/metrics.go` already computes `req.FirstTokenTime = now - req.ArrivalTime` relative to the original arrival. Therefore `aggregated.RequestTTFTs[parent.DecodeSubReqID]` equals parent-level TTFT directly — no additional calculation needed.

**DEVIATION:** `CompletionTime` field on `ParentRequest` is defined but never set in PR2. This PR does not set it (out of scope). Load imbalance uses throughput proxy (completions per pool / duration) rather than queue-depth time-series (which is not available post-simulation).

---

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: Parent TTFT accuracy**
- GIVEN a disaggregated request completed through the full prefill→transfer→decode path
- WHEN `CollectPDMetrics` is called with the completed `[]*ParentRequest` and aggregated metrics containing `RequestTTFTs`
- THEN `PDMetrics.ParentTTFT` contains the elapsed time from the original request's arrival to its first decode token, equal to the value stored in `aggregated.RequestTTFTs[parent.DecodeSubReqID]`
- MECHANISM: Decode sub-request is created with `ArrivalTime = orig.ArrivalTime` (pd_events.go:123), so the base simulator's TTFT measurement already captures parent-level elapsed time.

**BC-2: Transfer duration correctness**
- GIVEN a `ParentRequest` with both `TransferStartTime > 0` and `TransferCompleteTime > 0`
- WHEN `CollectPDMetrics` processes it
- THEN `PDMetrics.TransferDuration` contains the value `TransferCompleteTime - TransferStartTime` for that request
- MECHANISM: Direct timestamp subtraction from `ParentRequest` fields set by `KVTransferStartedEvent` and `KVTransferCompletedEvent`.

**BC-3: Transfer causality**
- GIVEN any completed `ParentRequest`
- WHEN its transfer timestamps are non-zero
- THEN `TransferCompleteTime >= TransferStartTime >= PrefillCompleteTime` (INV-PD-4 for transfer segment)
- MECHANISM: Events fire in priority order 5 → 6, with `TransferCompleteTime = TransferStartTime + duration` and `duration >= 1`.

**BC-4: Per-pool throughput**
- GIVEN a simulation with P prefill instances and D decode instances completing N disaggregated requests
- WHEN `CollectPDMetrics` is called with `poolMembership` and per-instance metrics
- THEN `PDMetrics.PrefillThroughput = sum(completions for prefill instances) / (simDurationUs / 1e6)` and `PDMetrics.DecodeThroughput = sum(completions for decode instances) / (simDurationUs / 1e6)`
- MECHANISM: Iterate `metricsByID`, sum `CompletedRequests` per pool role using `poolMembership`, divide by simulation duration in seconds.

**BC-5: Load imbalance ratio range**
- GIVEN both `PrefillThroughput > 0` and `DecodeThroughput > 0`
- WHEN `PDMetrics.LoadImbalanceRatio` is computed
- THEN `LoadImbalanceRatio >= 1.0` and equals 1.0 when both throughputs are equal
- SPECIAL CASES (BC-10): `math.MaxFloat64` when exactly one pool has zero throughput; `1.0` when both have zero throughput (no data)
- MECHANISM: `max(prefill, decode) / min(prefill, decode)`; numerator >= denominator always.

**BC-6: DisaggregatedCount accuracy**
- GIVEN N parent requests that completed the full prefill→transfer→decode path (i.e., `TransferCompleteTime > 0`)
- WHEN `CollectPDMetrics` is called
- THEN `PDMetrics.DisaggregatedCount == N`
- MECHANISM: Count parents where `TransferCompleteTime > 0` (confirms transfer occurred).

#### Negative Contracts

**BC-7: Nil when disaggregation disabled**
- GIVEN a simulation running without disaggregation (no prefill/decode pools configured)
- WHEN `CollectPDMetrics` is called with nil/empty parents and nil poolMembership
- THEN it returns `nil`
- MECHANISM: Early return when `len(parents) == 0`.

**BC-8: No mutation of input**
- GIVEN any inputs to `CollectPDMetrics`
- WHEN the function runs
- THEN the `parents` slice, `aggregated` metrics, `poolMembership` map, and `metricsByID` map are not modified
- MECHANISM: Read-only access to all input parameters.

**BC-9: RawMetrics backward compat**
- GIVEN a simulation without disaggregation where `rawMetrics.PD == nil`
- WHEN `cmd/root.go` processes results
- THEN no `=== PD Metrics ===` section is printed and all existing output sections (anomaly counters, KV cache metrics, per-SLO metrics, trace summary) are unchanged
- MECHANISM: `printPDMetrics` is guarded by `rawMetrics.PD != nil`.

#### Error Handling Contracts

**BC-10: Division-by-zero guard (R11)**
- GIVEN `LoadImbalanceRatio` calculation
- WHEN one pool has zero completions and the other has nonzero completions (one pool idle)
- THEN `LoadImbalanceRatio` is set to `math.MaxFloat64` (extreme imbalance sentinel — R11 guard, not divide-by-zero; printed as "inf (one pool idle)")
- WHEN both pools have zero completions (no disaggregated requests completed)
- THEN `LoadImbalanceRatio` is set to `1.0` (undefined/no data — neutral sentinel)
- RATIONALE: Returning `1.0` when one pool is completely idle would falsely indicate "perfectly balanced". `math.MaxFloat64` signals "effectively infinite" imbalance.
- MECHANISM: `if maxRPS == 0 { ratio = 1.0 } else if minRPS <= 0 { ratio = math.MaxFloat64 }` guard.

**BC-11: Missing TTFT data**
- GIVEN a `ParentRequest` where `aggregated.RequestTTFTs` does not contain the decode sub-req ID
- WHEN `CollectPDMetrics` processes it
- THEN that request is silently excluded from the TTFT distribution (no panic, no data corruption)
- MECHANISM: Map lookup returns 0.0 for missing key; values of 0.0 are excluded from the distribution to avoid misleading statistics.

---

### C) Component Interaction

```
cmd/root.go
  ├── cs.ParentRequests()          → []*ParentRequest     (new accessor, cluster.go)
  ├── cs.AggregatedMetrics()       → *sim.Metrics          (existing)
  ├── cs.PoolMembership()          → map[string]PoolRole   (existing, cluster.go:339)
  ├── cs.PerInstanceMetricsByID()  → map[string]*sim.Metrics (new accessor, cluster.go)
  └── CollectPDMetrics(...)        → *PDMetrics            (new, pd_metrics.go)
       └── written into rawMetrics.PD

RawMetrics.PD *PDMetrics           (new nullable field, metrics.go)
printPDMetrics(w, pd)              (new guarded print, root.go — nil-safe)
```

**API contracts:**
- `CollectPDMetrics(parents []*ParentRequest, aggregated *sim.Metrics, poolMembership map[string]PoolRole, metricsByID map[string]*sim.Metrics) *PDMetrics` — pure function, no side effects; returns nil when `len(parents) == 0`
- `(c *ClusterSimulator) ParentRequests() []*ParentRequest` — returns a snapshot of all tracked parent requests sorted by ID (R2: deterministic); panics if called before `Run()`
- `(c *ClusterSimulator) PerInstanceMetricsByID() map[string]*sim.Metrics` — returns map from instance ID string to its metrics; panics if called before `Run()`

**State changes:**
- `RawMetrics.PD *PDMetrics` — new nullable field; nil = disaggregation disabled; set once after simulation ends; not mutated after creation
- No mutable shared state added; `PDMetrics` is computed once and read-only thereafter

**Extension friction:**
- Adding one more `PDMetrics` field: touch `pd_metrics.go` (compute it) + `pd_metrics_test.go` (test it) + `root.go` (print it) = 3 files. Acceptable.

---

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Macro plan mentions JSON output extensions | Prints text section `=== PD Metrics ===` to stdout | Following existing pattern (`printKVCacheMetrics`, `printPerSLOMetrics`); JSON output refactor is out of scope for PR3 and would require `SaveResults` signature changes |
| Macro plan uses BC-PD-14, BC-PD-15, BC-PD-16 labels | This plan renames to BC-1 through BC-11 with plain descriptions | Micro-plan template requires self-contained BC numbering; macro-plan labels preserved in comments |
| `ParentRequest.CompletionTime` field exists | Not used in PR3 | `CompletionTime` is never set by PR2 event handlers; using it would return 0 (misleading). Computing parent E2E from `CompletionTime` is deferred to a future PR that sets this field |
| Load imbalance uses "queue depth" proxy | Uses throughput proxy (completions/sec per pool) | Queue depth time-series is not available post-simulation; throughput is the only meaningful per-pool load signal available |
| Per-pool throughput counts parent requests | Counts sub-request completions per pool | `CompletedRequests` in `sim.Metrics` counts sub-requests (prefill or decode) processed by each instance. One disaggregated parent = 1 prefill completion on a prefill instance + 1 decode completion on a decode instance. `PrefillThroughput` and `DecodeThroughput` are sub-request rates; they do not sum to cluster request rate. |
| Macro plan specifies "per-pool throughput AND latency" | Delivers per-pool throughput + `ParentTTFT` (aggregate) + `TransferDuration` | The macro plan's "per-pool latency" phrase is interpreted as parent-level aggregate TTFT (covering prefill + transfer + decode first token) and KV transfer duration distribution. Separate per-pool latency distributions (prefill-only TTFT and decode-only TTFT) are not produced in PR3; those require isolating the `_prefill` and `_decode` sub-request TTFT maps, which would require exposing additional `RequestTTFTs` state from `sim.Metrics` — deferred to a future PR. |

---

### E) Review Guide

**THE TRICKY PART:** The parent TTFT insight. Because the decode sub-request is created with `ArrivalTime = orig.ArrivalTime` (pd_events.go:123), `RequestTTFTs[decodeSubReqID]` already equals parent-level TTFT with no extra arithmetic. Reviewers should verify this is correct by tracing: `sim/metrics.go:397` computes `req.FirstTokenTime = now + currStepAdvance + ... - req.ArrivalTime`. If `req.ArrivalTime = orig.ArrivalTime` for the decode sub-req, this is relative to the original arrival. ✓

**WHAT TO SCRUTINIZE:** BC-1 (TTFT correctness), BC-7 (nil return for non-disaggregated), BC-10 (R11 division guard). Also check BC-9: that `printPDMetrics` is truly guarded and existing output sections are not affected.

**WHAT'S SAFE TO SKIM:** The `PerInstanceMetricsByID` accessor (straightforward map construction), the `printPDMetrics` function (mechanical print loop matching existing KV metrics pattern), and CLAUDE.md update.

**KNOWN DEBT:** `ParentRequest.CompletionTime` is never set by PR2. This PR leaves it unset. Parent E2E metrics will be added in a later PR when `CompletionTime` is populated.

---

## PART 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/cluster/pd_metrics.go` — `PDMetrics` struct + `CollectPDMetrics` pure function
- `sim/cluster/pd_metrics_test.go` — unit tests for `CollectPDMetrics` (direct construction, no ClusterSimulator dependency for unit tests)

**Files to modify:**
- `sim/cluster/metrics.go:87-110` — add `PD *PDMetrics` field to `RawMetrics`
- `sim/cluster/cluster.go` — add `ParentRequests()` and `PerInstanceMetricsByID()` accessors after line 367
- `cmd/root.go:809-876` — wire `CollectPDMetrics`, set `rawMetrics.PD`, add `printPDMetrics` call + function
- `CLAUDE.md` in worktree — add `pd_metrics.go` entry to File Organization tree

**Key decisions:**
1. `CollectPDMetrics` is a pure function (no ClusterSimulator receiver) — testable in isolation
2. `PDMetrics` uses `Distribution` (existing type) for TTFT and transfer — no new statistics types
3. `ParentRequests()` returns sorted-by-ID slice (R2: deterministic output)
4. `RawMetrics.PD *PDMetrics` — nullable pointer, nil = disabled (no need to export `PDMetrics{}` zero value)
5. TTFT values of 0.0 excluded from distribution (missing data guard, BC-11)

**R4 Construction site audit for `RawMetrics`:**
- `sim/cluster/metrics.go:119` — `&RawMetrics{...}` struct literal with named fields. Adding `PD` as a nil pointer field: existing literals remain valid (Go allows omitting fields). No update required.
- `sim/cluster/metrics_test.go:155,182,183,209,237,251` — test struct literals with named fields. Also remain valid. No update required.
- All sites confirmed: the new `PD *PDMetrics` field defaults to nil safely.

---

### G) Task Breakdown

---

### Task 1: PDMetrics struct + TTFT + transfer duration

**Contracts Implemented:** BC-1, BC-2, BC-3, BC-6, BC-7, BC-8, BC-11

**Files:**
- Create: `sim/cluster/pd_metrics.go`
- Create: `sim/cluster/pd_metrics_test.go`

**Step 1: Write failing tests for PDMetrics TTFT and transfer duration**

Context: We test `CollectPDMetrics` directly by constructing `ParentRequest` objects and synthetic `sim.Metrics`. No ClusterSimulator needed — pure unit test of the function.

In `sim/cluster/pd_metrics_test.go`:
```go
package cluster

import (
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildAggregatedWithTTFTs creates a minimal sim.Metrics with the given TTFT map.
func buildAggregatedWithTTFTs(ttfts map[string]float64, simEndedTime int64) *sim.Metrics {
	m := sim.NewMetrics()
	for k, v := range ttfts {
		m.RequestTTFTs[k] = v
	}
	m.SimEndedTime = simEndedTime
	return m
}

func TestCollectPDMetrics_NilWhenNoParents(t *testing.T) {
	// BC-7: returns nil for empty parents
	result := CollectPDMetrics(nil, sim.NewMetrics(), nil, nil)
	if result != nil {
		t.Errorf("CollectPDMetrics with nil parents = %v, want nil", result)
	}

	result = CollectPDMetrics([]*ParentRequest{}, sim.NewMetrics(), nil, nil)
	if result != nil {
		t.Errorf("CollectPDMetrics with empty parents = %v, want nil", result)
	}
}

func TestCollectPDMetrics_ParentTTFT(t *testing.T) {
	// BC-1: ParentTTFT distribution matches aggregated TTFT for decode sub-req IDs
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
		{ID: "req_1", DecodeSubReqID: "req_1_decode", TransferStartTime: 200, TransferCompleteTime: 350},
	}
	aggMetrics := buildAggregatedWithTTFTs(map[string]float64{
		"req_0_decode": 5000.0, // 5000 μs TTFT
		"req_1_decode": 7000.0, // 7000 μs TTFT
	}, 1_000_000)

	pd := CollectPDMetrics(parents, aggMetrics, nil, nil)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil, want non-nil PDMetrics")
	}
	if pd.ParentTTFT.Count != 2 {
		t.Errorf("ParentTTFT.Count = %d, want 2", pd.ParentTTFT.Count)
	}
	if pd.ParentTTFT.Min != 5000.0 {
		t.Errorf("ParentTTFT.Min = %.1f, want 5000.0", pd.ParentTTFT.Min)
	}
	if pd.ParentTTFT.Max != 7000.0 {
		t.Errorf("ParentTTFT.Max = %.1f, want 7000.0", pd.ParentTTFT.Max)
	}
}

func TestCollectPDMetrics_TTFTExcludesMissing(t *testing.T) {
	// BC-11: TTFT of 0.0 (missing key) is excluded from distribution
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
		{ID: "req_1", DecodeSubReqID: "req_1_decode", TransferStartTime: 200, TransferCompleteTime: 350},
	}
	// Only req_0_decode is in TTFT map; req_1_decode is missing
	aggMetrics := buildAggregatedWithTTFTs(map[string]float64{
		"req_0_decode": 5000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, aggMetrics, nil, nil)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil, want non-nil PDMetrics")
	}
	// Only 1 TTFT should be in the distribution
	if pd.ParentTTFT.Count != 1 {
		t.Errorf("ParentTTFT.Count = %d, want 1 (missing entries excluded)", pd.ParentTTFT.Count)
	}
}

func TestCollectPDMetrics_TransferDuration(t *testing.T) {
	// BC-2: TransferDuration = TransferCompleteTime - TransferStartTime per parent
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 1000, TransferCompleteTime: 1200},
		{ID: "req_1", DecodeSubReqID: "req_1_decode", TransferStartTime: 2000, TransferCompleteTime: 2500},
	}
	aggMetrics := buildAggregatedWithTTFTs(map[string]float64{
		"req_0_decode": 5000.0,
		"req_1_decode": 6000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, aggMetrics, nil, nil)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}
	if pd.TransferDuration.Count != 2 {
		t.Errorf("TransferDuration.Count = %d, want 2", pd.TransferDuration.Count)
	}
	// Durations: 200, 500 → min=200, max=500
	if pd.TransferDuration.Min != 200.0 {
		t.Errorf("TransferDuration.Min = %.1f, want 200.0", pd.TransferDuration.Min)
	}
	if pd.TransferDuration.Max != 500.0 {
		t.Errorf("TransferDuration.Max = %.1f, want 500.0", pd.TransferDuration.Max)
	}
}

func TestCollectPDMetrics_DisaggregatedCount(t *testing.T) {
	// BC-6: DisaggregatedCount = number of parents with TransferCompleteTime > 0
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
		{ID: "req_1", DecodeSubReqID: "req_1_decode", TransferStartTime: 0, TransferCompleteTime: 0}, // incomplete
	}
	aggMetrics := buildAggregatedWithTTFTs(map[string]float64{
		"req_0_decode": 5000.0,
	}, 1_000_000)

	pd := CollectPDMetrics(parents, aggMetrics, nil, nil)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}
	if pd.DisaggregatedCount != 1 {
		t.Errorf("DisaggregatedCount = %d, want 1 (only requests with completed transfer)", pd.DisaggregatedCount)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run TestCollectPDMetrics -v`
Expected: FAIL with `cannot refer to unexported name cluster.CollectPDMetrics` or `undefined: CollectPDMetrics`

**Step 3: Implement PDMetrics struct and CollectPDMetrics (TTFT + transfer + count)**

In `sim/cluster/pd_metrics.go`:
```go
package cluster

import (
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
)

// PDMetrics holds disaggregation-specific metrics collected after simulation.
// All duration values are in ticks (microseconds).
// Nil PDMetrics means disaggregation was not active during the simulation.
type PDMetrics struct {
	// DisaggregatedCount is the number of requests that completed the full
	// prefill→transfer→decode path (TransferCompleteTime > 0).
	DisaggregatedCount int

	// ParentTTFT is the distribution of time from original request arrival
	// to first decode token. Equal to aggregated.RequestTTFTs[parent.DecodeSubReqID]
	// because the decode sub-request is created with ArrivalTime = orig.ArrivalTime
	// (pd_events.go:123). Values are in ticks (microseconds).
	ParentTTFT Distribution

	// TransferDuration is the distribution of KV block transfer durations
	// (TransferCompleteTime - TransferStartTime) per request, in ticks (microseconds).
	TransferDuration Distribution

	// PrefillThroughput is completed prefill sub-requests per second.
	PrefillThroughput float64

	// DecodeThroughput is completed decode sub-requests per second.
	DecodeThroughput float64

	// LoadImbalanceRatio is max(PrefillThroughput, DecodeThroughput) /
	// min(PrefillThroughput, DecodeThroughput). A value of 1.0 means perfectly
	// balanced. Set to 1.0 when either pool has zero throughput (undefined).
	LoadImbalanceRatio float64
}

// CollectPDMetrics computes disaggregation-aware metrics from completed ParentRequests.
//
// Returns nil when len(parents) == 0 (disaggregation was not active or no requests
// completed the disaggregated path).
//
// Parameters:
//   - parents: snapshot of all tracked ParentRequests (from ClusterSimulator.ParentRequests())
//   - aggregated: aggregated sim.Metrics containing RequestTTFTs for decode sub-requests
//   - poolMembership: map from instance ID to PoolRole (from ClusterSimulator.PoolMembership())
//   - metricsByID: per-instance metrics keyed by instance ID string (from ClusterSimulator.PerInstanceMetricsByID())
func CollectPDMetrics(
	parents []*ParentRequest,
	aggregated *sim.Metrics,
	poolMembership map[string]PoolRole,
	metricsByID map[string]*sim.Metrics,
) *PDMetrics {
	if len(parents) == 0 {
		return nil
	}

	// Sort parents by ID for deterministic processing (R2).
	sorted := make([]*ParentRequest, len(parents))
	copy(sorted, parents)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].ID < sorted[j].ID
	})

	var (
		ttftValues     []float64
		transferValues []float64
		disaggCount    int
	)

	for _, p := range sorted {
		if p.TransferCompleteTime > 0 {
			disaggCount++
		}

		// Parent TTFT: use decode sub-req's TTFT from aggregated metrics.
		// The decode sub-request has ArrivalTime = orig.ArrivalTime (pd_events.go:123),
		// so RequestTTFTs[DecodeSubReqID] already equals parent-level TTFT.
		// Exclude 0.0 values (missing key returns 0 — avoids misleading statistics).
		if ttft := aggregated.RequestTTFTs[p.DecodeSubReqID]; ttft > 0 {
			ttftValues = append(ttftValues, ttft)
		}

		// Transfer duration: direct timestamp subtraction.
		if p.TransferStartTime > 0 && p.TransferCompleteTime >= p.TransferStartTime {
			transferValues = append(transferValues, float64(p.TransferCompleteTime-p.TransferStartTime))
		}
	}

	pd := &PDMetrics{
		DisaggregatedCount: disaggCount,
		ParentTTFT:         NewDistribution(ttftValues),
		TransferDuration:   NewDistribution(transferValues),
		LoadImbalanceRatio: 1.0, // default: balanced / undefined; overwritten by per-pool calc below
	}

	// Per-pool throughput (computed separately in Task 2).
	pd.PrefillThroughput, pd.DecodeThroughput, pd.LoadImbalanceRatio =
		collectPoolThroughput(poolMembership, metricsByID, aggregated.SimEndedTime)

	return pd
}

// collectPoolThroughput computes per-pool request throughput and load imbalance ratio.
// Returns (prefillRPS, decodeRPS, imbalanceRatio).
// Returns (0, 0, 1.0) when poolMembership or metricsByID are nil/empty.
func collectPoolThroughput(
	poolMembership map[string]PoolRole,
	metricsByID map[string]*sim.Metrics,
	simEndedTimeUs int64,
) (prefillRPS, decodeRPS, imbalanceRatio float64) {
	imbalanceRatio = 1.0
	if len(poolMembership) == 0 || len(metricsByID) == 0 || simEndedTimeUs <= 0 {
		return
	}

	var prefillCompleted, decodeCompleted int
	// Sort instance IDs for deterministic accumulation (R2).
	ids := make([]string, 0, len(metricsByID))
	for id := range metricsByID {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		m := metricsByID[id]
		switch poolMembership[id] {
		case PoolRolePrefill:
			prefillCompleted += m.CompletedRequests
		case PoolRoleDecode:
			decodeCompleted += m.CompletedRequests
		}
	}

	durationSec := float64(simEndedTimeUs) / 1e6
	prefillRPS = float64(prefillCompleted) / durationSec
	decodeRPS = float64(decodeCompleted) / durationSec

	// Compute load imbalance ratio (R11: guard division by zero).
	// math.MaxFloat64 signals "one pool completely idle" (extreme imbalance).
	// 1.0 when both pools have zero completions (ratio undefined, neutral sentinel).
	minRPS := prefillRPS
	maxRPS := decodeRPS
	if decodeRPS < prefillRPS {
		minRPS = decodeRPS
		maxRPS = prefillRPS
	}
	switch {
	case maxRPS == 0:
		imbalanceRatio = 1.0 // Both pools idle; ratio undefined — no data
	case minRPS <= 0:
		imbalanceRatio = math.MaxFloat64 // One pool completely idle = extreme imbalance
	default:
		imbalanceRatio = maxRPS / minRPS
	}
	return
}
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run TestCollectPDMetrics -v`
Expected: PASS (all 5 TestCollectPDMetrics_* tests pass)

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics add \
  sim/cluster/pd_metrics.go sim/cluster/pd_metrics_test.go
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics commit -m \
  "feat(cluster): add PDMetrics struct and CollectPDMetrics (BC-1,BC-2,BC-6,BC-7,BC-11)

- Add PDMetrics with ParentTTFT, TransferDuration, DisaggregatedCount
- Add CollectPDMetrics pure function computing PD metrics from ParentRequests
- Parent TTFT reads RequestTTFTs[DecodeSubReqID] directly (ArrivalTime trick)
- Per-pool throughput and load imbalance via collectPoolThroughput helper
- Guard: nil return when len(parents)==0; 0-TTFT exclusion for missing keys

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Per-pool throughput and load imbalance (integration invariant test)

**Contracts Implemented:** BC-4, BC-5, BC-10

**Files:**
- Modify: `sim/cluster/pd_metrics_test.go`

**Step 1: Write integration tests for per-pool throughput (validating Task 1 implementation)**

Context: Test `collectPoolThroughput` via `CollectPDMetrics` with a realistic pool membership + per-instance metrics fixture. These tests validate the `collectPoolThroughput` function already implemented in Task 1 Step 3, so they will **pass immediately** when run — this is intentional (tests written alongside implementation for a pure-helper function). We need to verify that prefill and decode completions are correctly attributed to pools.

Add to `sim/cluster/pd_metrics_test.go`:
```go
func TestCollectPDMetrics_PerPoolThroughput(t *testing.T) {
	// BC-4: PrefillThroughput = prefill completions / duration_sec
	//        DecodeThroughput  = decode completions / duration_sec
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
		{ID: "req_1", DecodeSubReqID: "req_1_decode", TransferStartTime: 200, TransferCompleteTime: 350},
	}

	poolMembership := map[string]PoolRole{
		"instance_0": PoolRolePrefill,
		"instance_1": PoolRoleDecode,
	}

	// instance_0 completed 2 prefill sub-requests; instance_1 completed 2 decode
	prefillMetrics := sim.NewMetrics()
	prefillMetrics.CompletedRequests = 2
	decodeMetrics := sim.NewMetrics()
	decodeMetrics.CompletedRequests = 2

	metricsByID := map[string]*sim.Metrics{
		"instance_0": prefillMetrics,
		"instance_1": decodeMetrics,
	}

	// simEndedTime = 2,000,000 μs = 2.0 seconds
	agg := buildAggregatedWithTTFTs(map[string]float64{
		"req_0_decode": 5000.0,
		"req_1_decode": 6000.0,
	}, 2_000_000)

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}

	// 2 completions / 2 seconds = 1.0 req/s each
	if pd.PrefillThroughput != 1.0 {
		t.Errorf("PrefillThroughput = %.4f, want 1.0", pd.PrefillThroughput)
	}
	if pd.DecodeThroughput != 1.0 {
		t.Errorf("DecodeThroughput = %.4f, want 1.0", pd.DecodeThroughput)
	}
}

func TestCollectPDMetrics_LoadImbalanceRatio_Balanced(t *testing.T) {
	// BC-5: LoadImbalanceRatio = 1.0 when throughputs are equal
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
	}
	poolMembership := map[string]PoolRole{"instance_0": PoolRolePrefill, "instance_1": PoolRoleDecode}
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 3
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 3
	metricsByID := map[string]*sim.Metrics{"instance_0": m1, "instance_1": m2}
	agg := buildAggregatedWithTTFTs(map[string]float64{"req_0_decode": 5000.0}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}
	if pd.LoadImbalanceRatio != 1.0 {
		t.Errorf("LoadImbalanceRatio = %.4f, want 1.0 for equal throughputs", pd.LoadImbalanceRatio)
	}
}

func TestCollectPDMetrics_LoadImbalanceRatio_Imbalanced(t *testing.T) {
	// BC-5: LoadImbalanceRatio >= 1.0 when throughputs differ
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
	}
	poolMembership := map[string]PoolRole{"instance_0": PoolRolePrefill, "instance_1": PoolRoleDecode}
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 4 // prefill: 4 completions → 4.0 rps
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 2 // decode: 2 completions → 2.0 rps
	metricsByID := map[string]*sim.Metrics{"instance_0": m1, "instance_1": m2}
	agg := buildAggregatedWithTTFTs(map[string]float64{"req_0_decode": 5000.0}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}
	if pd.LoadImbalanceRatio < 1.0 {
		t.Errorf("LoadImbalanceRatio = %.4f, want >= 1.0", pd.LoadImbalanceRatio)
	}
	// 4.0 / 2.0 = 2.0
	const wantRatio = 2.0
	const eps = 1e-9
	if pd.LoadImbalanceRatio < wantRatio-eps || pd.LoadImbalanceRatio > wantRatio+eps {
		t.Errorf("LoadImbalanceRatio = %.6f, want %.6f", pd.LoadImbalanceRatio, wantRatio)
	}
}

func TestCollectPDMetrics_LoadImbalanceRatio_ZeroMinGuard(t *testing.T) {
	// BC-10: LoadImbalanceRatio = math.MaxFloat64 when one pool idle (R11 guard).
	// One pool with completions + one pool with zero completions = extreme imbalance.
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
	}
	poolMembership := map[string]PoolRole{"instance_0": PoolRolePrefill, "instance_1": PoolRoleDecode}
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 3
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 0 // decode has zero completions → extreme imbalance
	metricsByID := map[string]*sim.Metrics{"instance_0": m1, "instance_1": m2}
	agg := buildAggregatedWithTTFTs(map[string]float64{"req_0_decode": 5000.0}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}
	if pd.LoadImbalanceRatio != math.MaxFloat64 {
		t.Errorf("LoadImbalanceRatio = %.4g, want math.MaxFloat64 when one pool idle (R11 guard)", pd.LoadImbalanceRatio)
	}
}

func TestCollectPDMetrics_LoadImbalanceRatio_BothZeroGuard(t *testing.T) {
	// BC-10: LoadImbalanceRatio = 1.0 when both pools have zero completions (no data).
	// Both zero means "no disaggregated requests completed" — neutral sentinel.
	parents := []*ParentRequest{
		{ID: "req_0", DecodeSubReqID: "req_0_decode", TransferStartTime: 100, TransferCompleteTime: 200},
	}
	poolMembership := map[string]PoolRole{"instance_0": PoolRolePrefill, "instance_1": PoolRoleDecode}
	m1 := sim.NewMetrics()
	m1.CompletedRequests = 0
	m2 := sim.NewMetrics()
	m2.CompletedRequests = 0
	metricsByID := map[string]*sim.Metrics{"instance_0": m1, "instance_1": m2}
	agg := buildAggregatedWithTTFTs(map[string]float64{"req_0_decode": 5000.0}, 1_000_000)

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Fatal("CollectPDMetrics returned nil")
	}
	if pd.LoadImbalanceRatio != 1.0 {
		t.Errorf("LoadImbalanceRatio = %.4f, want 1.0 when both pools idle (neutral sentinel)", pd.LoadImbalanceRatio)
	}
}
```

**Step 2: Run tests to verify they pass (already implemented in Task 1)**

Run: `go test ./sim/cluster/... -run TestCollectPDMetrics -v`
Expected: PASS (all 9 TestCollectPDMetrics_* tests pass — Tasks 1 + 2 combined)

**Step 3: No code changes needed**

The `collectPoolThroughput` function was already implemented in Task 1's Step 3. These tests validate the existing implementation.

**Step 4: Run full cluster test suite**

Run: `go test ./sim/cluster/... -v 2>&1 | tail -20`
Expected: all tests PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics add sim/cluster/pd_metrics_test.go
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics commit -m \
  "test(cluster): add per-pool throughput and load imbalance tests for PDMetrics (BC-4,BC-5,BC-10)

- Test balanced load imbalance ratio = 1.0
- Test imbalanced ratio = max/min
- Test R11 guard: one pool idle → math.MaxFloat64 (extreme imbalance)
- Test R11 guard: both pools idle → 1.0 (neutral/no-data sentinel)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: ClusterSimulator accessors (ParentRequests + PerInstanceMetricsByID)

**Contracts Implemented:** BC-8 (no mutation of input)

**Files:**
- Modify: `sim/cluster/cluster.go` (after line 367 — after `PerInstanceMetrics()`)
- Modify: `sim/cluster/disaggregation_test.go` (add accessor tests)

**Step 1: Write failing tests for new accessors**

Add to `sim/cluster/disaggregation_test.go`:
```go
func TestClusterSimulator_ParentRequests_ReturnsAllParents(t *testing.T) {
	// ClusterSimulator.ParentRequests() returns all tracked parent requests
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	// All 3 requests should be disaggregated (PDDecider="always")
	if len(parents) != 3 {
		t.Errorf("ParentRequests() len = %d, want 3", len(parents))
	}
	// Result must be sorted by ID for determinism
	for i := 1; i < len(parents); i++ {
		if parents[i].ID < parents[i-1].ID {
			t.Errorf("ParentRequests() not sorted: parents[%d].ID=%q < parents[%d].ID=%q",
				i, parents[i].ID, i-1, parents[i-1].ID)
		}
	}
}

func TestClusterSimulator_PerInstanceMetricsByID_ContainsAllInstances(t *testing.T) {
	// ClusterSimulator.PerInstanceMetricsByID() contains one entry per instance
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	metricsByID := cs.PerInstanceMetricsByID()
	if len(metricsByID) != 4 {
		t.Errorf("PerInstanceMetricsByID() len = %d, want 4", len(metricsByID))
	}
	// Instance IDs should match the known pattern
	for _, id := range []string{"instance_0", "instance_1", "instance_2", "instance_3"} {
		if _, ok := metricsByID[id]; !ok {
			t.Errorf("PerInstanceMetricsByID() missing instance %q", id)
		}
	}
}

func TestClusterSimulator_PDMetricsInvariant_PoolConservation(t *testing.T) {
	// BC-PD-15 invariant: sum of per-pool completions = total completions
	// (Prefill pool completions + decode pool completions = all instance completions summed)
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	metricsByID := cs.PerInstanceMetricsByID()
	poolMembership := cs.PoolMembership()

	var prefillTotal, decodeTotal, allTotal int
	for id, m := range metricsByID {
		allTotal += m.CompletedRequests
		switch poolMembership[id] {
		case PoolRolePrefill:
			prefillTotal += m.CompletedRequests
		case PoolRoleDecode:
			decodeTotal += m.CompletedRequests
		}
	}
	// Every instance is either prefill or decode in a fully-disaggregated config
	if prefillTotal+decodeTotal != allTotal {
		t.Errorf("pool conservation violated: prefill(%d) + decode(%d) != all(%d)",
			prefillTotal, decodeTotal, allTotal)
	}
}

func TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration(t *testing.T) {
	// BC-1 causality invariant: ParentTTFT.Mean >= TransferDuration.Mean
	// because TTFT is measured from original request arrival (before transfer starts).
	//
	// This test validates the mechanism in pd_events.go:123:
	//   decode_subreq.ArrivalTime = orig.ArrivalTime
	// For every disaggregated request:
	//   TTFT_r = firstTokenTime - orig.ArrivalTime
	//   TransferDuration_r = TransferCompleteTime - TransferStartTime
	// Since firstToken >= TransferCompleteTime >= TransferStartTime >= orig.ArrivalTime:
	//   TTFT_r >= TransferDuration_r for all r => mean(TTFT) >= mean(Transfer)
	//
	// If ArrivalTime were incorrectly set to decode enqueue time instead of original
	// arrival, TTFT would be shorter than transfer duration, violating this invariant.
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)
	mustRun(t, cs)

	parents := cs.ParentRequests()
	agg := cs.AggregatedMetrics()
	poolMembership := cs.PoolMembership()
	metricsByID := cs.PerInstanceMetricsByID()

	pd := CollectPDMetrics(parents, agg, poolMembership, metricsByID)
	if pd == nil {
		t.Skip("no PD metrics — disaggregation may not be active in test config")
	}
	if pd.DisaggregatedCount == 0 {
		t.Skip("no disaggregated requests completed — cannot test TTFT causality")
	}

	// ParentTTFT.Mean must be >= TransferDuration.Mean (causality from original arrival)
	if pd.ParentTTFT.Count > 0 && pd.TransferDuration.Count > 0 {
		if pd.ParentTTFT.Mean < pd.TransferDuration.Mean {
			t.Errorf("BC-1 causality violated: ParentTTFT.Mean (%.1f μs) < TransferDuration.Mean (%.1f μs): "+
				"parent TTFT must be measured from original arrival (pd_events.go:123), "+
				"not from decode enqueue time",
				pd.ParentTTFT.Mean, pd.TransferDuration.Mean)
		}
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./sim/cluster/... -run "TestClusterSimulator_ParentRequests|TestClusterSimulator_PerInstanceMetricsByID|TestClusterSimulator_PDMetricsInvariant|TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration" -v`
Expected: FAIL with `cs.ParentRequests undefined` and similar

**Step 3: Implement accessors in cluster.go**

In `sim/cluster/cluster.go`, add after the `PerInstanceMetrics()` function (after line 367):

```go
// ParentRequests returns a snapshot of all tracked ParentRequests, sorted by ID.
// Returns an empty slice when disaggregation is disabled or no requests were disaggregated.
// Panics if called before Run() has completed.
// The returned slice is a copy — callers may not modify ParentRequest values.
func (c *ClusterSimulator) ParentRequests() []*ParentRequest {
	if !c.hasRun {
		panic("ClusterSimulator.ParentRequests() called before Run()")
	}
	result := make([]*ParentRequest, 0, len(c.parentRequests))
	for _, pr := range c.parentRequests {
		result = append(result, pr)
	}
	// Sort by ID for deterministic output (R2).
	sort.Slice(result, func(i, j int) bool {
		return result[i].ID < result[j].ID
	})
	return result
}

// PerInstanceMetricsByID returns a map from instance ID string to its post-simulation metrics.
// Panics if called before Run() has completed.
func (c *ClusterSimulator) PerInstanceMetricsByID() map[string]*sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.PerInstanceMetricsByID() called before Run()")
	}
	result := make(map[string]*sim.Metrics, len(c.instances))
	for _, inst := range c.instances {
		result[string(inst.ID())] = inst.Metrics()
	}
	return result
}
```

Also add `"sort"` to the `cluster.go` import block if not already present. Check current imports:
```go
import (
    "container/heap"
    "fmt"
    "math"
    "sort"  // add if missing

    "github.com/inference-sim/inference-sim/sim"
    "github.com/inference-sim/inference-sim/sim/trace"
    "github.com/sirupsen/logrus"
)
```

**Step 4: Run tests to verify they pass**

Run: `go test ./sim/cluster/... -run "TestClusterSimulator_ParentRequests|TestClusterSimulator_PerInstanceMetricsByID|TestClusterSimulator_PDMetricsInvariant|TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration" -v`
Expected: PASS

**Step 5: Run full cluster test suite**

Run: `go test ./sim/cluster/... 2>&1 | tail -5`
Expected: All tests pass

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 7: Commit**

```bash
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics add \
  sim/cluster/cluster.go sim/cluster/disaggregation_test.go
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics commit -m \
  "feat(cluster): add ParentRequests() and PerInstanceMetricsByID() accessors

- ParentRequests(): sorted snapshot of disaggregated request lifecycle records
- PerInstanceMetricsByID(): map from instance ID to post-simulation metrics
- Add pool conservation invariant test (BC-PD-15)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: RawMetrics.PD field and CollectRawMetrics wiring

**Contracts Implemented:** BC-9 (backward compat — nil when no PD)

**Files:**
- Modify: `sim/cluster/metrics.go:87-110` — add `PD *PDMetrics` to `RawMetrics`
- Modify: `sim/cluster/metrics_test.go` — add test for nil PD field

**Step 1: Write failing test for RawMetrics.PD**

Add to `sim/cluster/metrics_test.go`:
```go
func TestCollectRawMetrics_PDFieldNilWhenNoDisagg(t *testing.T) {
	// BC-9: RawMetrics.PD is nil when disaggregation was not active
	// (CollectRawMetrics does not populate PD — that is done by cmd/root.go)
	agg := sim.NewMetrics()
	agg.SimEndedTime = 1_000_000
	raw := CollectRawMetrics(agg, nil, 0, "constant")
	if raw.PD != nil {
		t.Errorf("RawMetrics.PD = %v, want nil for non-disaggregated simulation", raw.PD)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestCollectRawMetrics_PDFieldNilWhenNoDisagg -v`
Expected: FAIL with `raw.PD undefined (type *RawMetrics has no field or method PD)` or compilation error

**Step 3: Add PD field to RawMetrics**

In `sim/cluster/metrics.go`, modify `RawMetrics` struct (lines 87-110):
```go
// RawMetrics holds cluster-level metrics aggregated after simulation.
type RawMetrics struct {
	// Latency distributions (in ticks)
	TTFT Distribution
	E2E  Distribution

	// Per-SLO-class distributions (PR10: keyed by SLOClass string)
	PerSLOClass map[string]*SLOMetrics

	// Throughput
	RequestsPerSec float64
	TokensPerSec   float64

	// Anomaly counters
	PriorityInversions int
	HOLBlockingEvents  int
	RejectedRequests   int
	DroppedUnservable  int

	// KV cache metrics (PR12)
	CacheHitRate    float64
	PreemptionRate  float64
	KVThrashingRate float64

	// PD disaggregation metrics (PR3). Nil when disaggregation is not active.
	PD *PDMetrics
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestCollectRawMetrics_PDFieldNilWhenNoDisagg -v`
Expected: PASS

**Step 5: Run full test suite to catch any construction-site breakage (R4)**

Run: `go test ./sim/cluster/... ./cmd/... 2>&1 | tail -5`
Expected: All tests pass. (All `&RawMetrics{...}` literals use named fields so the new PD field defaults to nil — no breakage.)

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/cluster/... ./cmd/...`
Expected: No new issues

**Step 7: Commit**

```bash
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics add \
  sim/cluster/metrics.go sim/cluster/metrics_test.go
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics commit -m \
  "feat(cluster): add PD *PDMetrics field to RawMetrics (BC-9)

- RawMetrics.PD is nil when disaggregation is inactive (backward compat)
- R4 audit: all existing RawMetrics literals use named fields, no update needed

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: cmd/root.go wiring and printPDMetrics output

**Contracts Implemented:** BC-9 (output section guarded by PD != nil)

**Files:**
- Modify: `cmd/root.go:809-876`

**Step 1: Write failing test for PD metrics output**

Add to `cmd/kv_metrics_output_test.go` (following the existing pattern in that file):
```go
func TestPrintPDMetrics_NilDoesNotPrint(t *testing.T) {
	// BC-9: printPDMetrics does not print when pd is nil
	var buf bytes.Buffer
	printPDMetrics(&buf, nil)
	if buf.Len() != 0 {
		t.Errorf("printPDMetrics(nil) printed %q, want empty output", buf.String())
	}
}

func TestPrintPDMetrics_PrintsSection(t *testing.T) {
	// BC-9: printPDMetrics prints the PD Metrics section header and key fields
	var buf bytes.Buffer
	pd := &cluster.PDMetrics{
		DisaggregatedCount: 5,
		PrefillThroughput:  2.5,
		DecodeThroughput:   2.0,
		LoadImbalanceRatio: 1.25,
		ParentTTFT:         cluster.Distribution{Mean: 5000, P50: 4800, P95: 7200, P99: 9100, Count: 5},
		TransferDuration:   cluster.Distribution{Mean: 300, P50: 280, P95: 450, P99: 500, Count: 5},
	}
	printPDMetrics(&buf, pd)

	output := buf.String()
	if !strings.Contains(output, "=== PD Metrics ===") {
		t.Errorf("output missing section header, got:\n%s", output)
	}
	if !strings.Contains(output, "Disaggregated Requests: 5") {
		t.Errorf("output missing disaggregated count, got:\n%s", output)
	}
	if !strings.Contains(output, "Prefill Throughput:") {
		t.Errorf("output missing prefill throughput, got:\n%s", output)
	}
	if !strings.Contains(output, "Load Imbalance Ratio:") {
		t.Errorf("output missing load imbalance ratio, got:\n%s", output)
	}
}
```

**Step 2: Run tests to verify they fail**

Run: `go test ./cmd/... -run "TestPrintPDMetrics" -v`
Expected: FAIL with `printPDMetrics undefined`

**Step 3: Add printPDMetrics function and wiring in cmd/root.go**

In `cmd/root.go`, after the `printKVCacheMetrics` function (around line 891), add:

```go
// printPDMetrics prints disaggregation-aware metrics to w when pd is non-nil.
// Skips silently when pd is nil (disaggregation was not active).
func printPDMetrics(w io.Writer, pd *cluster.PDMetrics) {
	if pd == nil {
		return
	}
	_, _ = fmt.Fprintln(w, "=== PD Metrics ===")
	_, _ = fmt.Fprintf(w, "Disaggregated Requests: %d\n", pd.DisaggregatedCount)
	_, _ = fmt.Fprintf(w, "Prefill Throughput: %.4f sub-req/s\n", pd.PrefillThroughput)
	_, _ = fmt.Fprintf(w, "Decode Throughput: %.4f sub-req/s\n", pd.DecodeThroughput)
	// math.MaxFloat64 sentinel = one pool completely idle (extreme imbalance).
	if pd.LoadImbalanceRatio >= math.MaxFloat64/2 {
		_, _ = fmt.Fprintf(w, "Load Imbalance Ratio: inf (one pool idle)\n")
	} else {
		_, _ = fmt.Fprintf(w, "Load Imbalance Ratio: %.4f\n", pd.LoadImbalanceRatio)
	}
	if pd.ParentTTFT.Count > 0 {
		_, _ = fmt.Fprintf(w, "Parent TTFT (μs): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
			pd.ParentTTFT.Mean, pd.ParentTTFT.P50, pd.ParentTTFT.P95, pd.ParentTTFT.P99)
	}
	if pd.TransferDuration.Count > 0 {
		_, _ = fmt.Fprintf(w, "KV Transfer Duration (μs): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
			pd.TransferDuration.Mean, pd.TransferDuration.P50, pd.TransferDuration.P95, pd.TransferDuration.P99)
	}
}
```

In `cmd/root.go`, after the `rawMetrics` is collected and `CollectRawMetrics` call (lines 809-815), add the PD metrics wiring. The change is to the block starting at line 809:

```go
// Collect RawMetrics and compute fitness (PR9)
rawMetrics := cluster.CollectRawMetrics(
    cs.AggregatedMetrics(),
    cs.PerInstanceMetrics(),
    cs.RejectedRequests(),
    priorityPolicy,
)

// Collect PD metrics when disaggregation was active (PR3).
rawMetrics.PD = cluster.CollectPDMetrics(
    cs.ParentRequests(),
    cs.AggregatedMetrics(),
    cs.PoolMembership(),
    cs.PerInstanceMetricsByID(),
)
```

Then add the `printPDMetrics` call after the KV cache metrics print (around line 849):

```go
// Print KV cache metrics if any nonzero (BC-1, BC-2)
printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

// Print PD disaggregation metrics if active (PR3)
printPDMetrics(os.Stdout, rawMetrics.PD)
```

**Step 4: Run tests to verify they pass**

Run: `go test ./cmd/... -run "TestPrintPDMetrics" -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `go test ./... 2>&1 | tail -10`
Expected: All tests pass

**Step 6: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 7: Commit**

```bash
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics add cmd/root.go cmd/kv_metrics_output_test.go
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics commit -m \
  "feat(cmd): wire CollectPDMetrics and printPDMetrics output section (BC-9)

- Call CollectPDMetrics after CollectRawMetrics; assign to rawMetrics.PD
- printPDMetrics prints '=== PD Metrics ===' section when PD != nil
- Section is guarded: non-disaggregated simulations produce no extra output

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: CLAUDE.md documentation update

**Contracts Implemented:** (documentation only)

**Files:**
- Modify: `CLAUDE.md` in worktree

**Step 1: No test needed (documentation update)**

**Step 2: Update CLAUDE.md File Organization tree**

In `CLAUDE.md`, in the `sim/cluster/` section of the File Organization tree, add `pd_metrics.go` entry after `pool.go`:

Find this existing entry:
```
│   ├── pool.go                # PoolRole type, ValidatePoolTopology(), BuildPoolMembership() for PD disaggregation pool topology
│   └── evaluation.go          # EvaluationResult wrapper (RawMetrics + FitnessResult + trace + summary)
```

Change to:
```
│   ├── pool.go                # PoolRole type, ValidatePoolTopology(), BuildPoolMembership() for PD disaggregation pool topology
│   ├── pd_metrics.go          # PDMetrics struct, CollectPDMetrics() — disaggregation-aware metrics (parent TTFT, transfer duration, per-pool throughput, load imbalance ratio)
│   └── evaluation.go          # EvaluationResult wrapper (RawMetrics + FitnessResult + trace + summary)
```

Also update the `cluster.go` entry to mention the new accessors:
Find:
```
│   ├── cluster.go             # ClusterSimulator orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation; Run() returns error
```
Change to:
```
│   ├── cluster.go             # ClusterSimulator orchestrates N instances with shared-clock event loop, online routing pipeline, and metrics aggregation; Run() returns error; ParentRequests(), PerInstanceMetricsByID() accessors added in PR3
```

And update the `metrics.go` entry to mention the new PD field:
Find:
```
│   ├── metrics.go             # RawMetrics, Distribution, FitnessResult, CollectRawMetrics (accepts priorityPolicy), ComputeFitness (returns (FitnessResult, error)), anomaly detection, ParseFitnessWeights with NaN/Inf validation, per-SLO-class metrics, JainFairnessIndex
```
Change to:
```
│   ├── metrics.go             # RawMetrics (with PD *PDMetrics field), Distribution, FitnessResult, CollectRawMetrics (accepts priorityPolicy), ComputeFitness (returns (FitnessResult, error)), anomaly detection, ParseFitnessWeights with NaN/Inf validation, per-SLO-class metrics, JainFairnessIndex
```

**Step 3: Verify build still passes**

Run: `go build ./... 2>&1`
Expected: No output (clean build)

**Step 4: Commit**

```bash
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics add CLAUDE.md
git -C /ws/fork/inference-sim/.worktrees/pr3-pd-metrics commit -m \
  "docs: update CLAUDE.md file organization for PR3 pd_metrics.go

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 (parent TTFT accuracy) | Task 1 | Unit | `TestCollectPDMetrics_ParentTTFT` |
| BC-2 (transfer duration) | Task 1 | Unit | `TestCollectPDMetrics_TransferDuration` |
| BC-3 (transfer causality) | Task 1 | Invariant | `TestCollectPDMetrics_TransferDuration` (min ≥ 0 via sort; implicit in NewDistribution) |
| BC-4 (per-pool throughput) | Task 2 | Unit | `TestCollectPDMetrics_PerPoolThroughput` |
| BC-5 (load imbalance ≥ 1.0) | Task 2 | Unit | `TestCollectPDMetrics_LoadImbalanceRatio_Balanced`, `_Imbalanced` |
| BC-6 (disaggregated count) | Task 1 | Unit | `TestCollectPDMetrics_DisaggregatedCount` |
| BC-7 (nil when no parents) | Task 1 | Unit | `TestCollectPDMetrics_NilWhenNoParents` |
| BC-8 (no mutation) | all | Implicit | Pure function — no shared state, no receiver mutation |
| BC-9 (backward compat) | Task 4+5 | Unit | `TestCollectRawMetrics_PDFieldNilWhenNoDisagg`, `TestPrintPDMetrics_NilDoesNotPrint` |
| BC-10 (division guard R11) | Task 2 | Unit | `TestCollectPDMetrics_LoadImbalanceRatio_ZeroMinGuard`, `TestCollectPDMetrics_LoadImbalanceRatio_BothZeroGuard` |
| BC-11 (missing TTFT excluded) | Task 1 | Unit | `TestCollectPDMetrics_TTFTExcludesMissing` |
| Pool conservation invariant | Task 3 | Invariant | `TestClusterSimulator_PDMetricsInvariant_PoolConservation` |
| BC-1 TTFT causality invariant | Task 3 | Invariant | `TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration` — asserts `ParentTTFT.Mean >= TransferDuration.Mean`; catches future breakage of pd_events.go:123 (decode sub-req `ArrivalTime = orig.ArrivalTime`) |
| Accessor correctness | Task 3 | Integration | `TestClusterSimulator_ParentRequests_ReturnsAllParents`, `TestClusterSimulator_PerInstanceMetricsByID_ContainsAllInstances` |
| Print output | Task 5 | Unit | `TestPrintPDMetrics_PrintsSection` |

**Golden dataset:** This PR adds a new output section (`=== PD Metrics ===`) that only prints when disaggregation is active. The existing golden dataset tests use single-instance mode (no disaggregation), so `testdata/goldendataset.json` does not need regeneration. Verify by running `go test ./... -run TestGolden` after all tasks complete — it should still pass without changes.

**Shared test infrastructure:** Use existing `newTestDisaggDeploymentConfig`, `newTestRequests`, and `mustRun` helpers from `sim/cluster/test_helpers_test.go` and `cluster_test.go`. The new `buildAggregatedWithTTFTs` helper in `pd_metrics_test.go` is scoped to that file (not duplicating anything in `sim/internal/testutil`).

---

### I) Risk Analysis

**Risk 1: TTFT identity assumption broken by future PR2 change**
- Likelihood: Low (PR2 is merged; `ArrivalTime = orig.ArrivalTime` is a stable contract)
- Impact: High (parent TTFT would be wrong)
- Mitigation: BC-1 explicitly documents the mechanism with file:line citation (pd_events.go:123); `TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration` (Task 3) runs end-to-end disaggregated simulation and asserts `ParentTTFT.Mean >= TransferDuration.Mean` — if ArrivalTime were changed to decode enqueue time, TTFT would be shorter than transfer duration, failing this invariant
- Task: Task 1 (implementation), Task 3 (invariant test)

**Risk 2: `RequestTTFTs` map returns 0.0 for decode sub-reqs that didn't generate output tokens**
- Likelihood: Medium (possible in partial-simulation or zero-output requests)
- Impact: Low (misleading statistics if included)
- Mitigation: BC-11: values of 0.0 are excluded from the TTFT distribution; `TestCollectPDMetrics_TTFTExcludesMissing` validates this
- Task: Task 1

**Risk 3: `sort` import not present in cluster.go (R2 compliance)**
- Likelihood: Low (cluster.go already uses sorting elsewhere)
- Impact: Low (compile error caught immediately)
- Mitigation: Step 3 of Task 3 checks and adds the import if missing
- Task: Task 3

**Risk 4: `cmd/root.go` calls `CollectPDMetrics` before `Run()` (would panic)**
- Likelihood: None (code sequence is deterministic: `cs.Run()` → `CollectRawMetrics` → `CollectPDMetrics`)
- Impact: N/A
- Mitigation: Not a risk; accessor panics are defensive and will never trigger in normal flow

**Risk 5: kv_metrics_output_test.go test file location**
- Likelihood: None (confirmed resolved)
- Impact: N/A
- Mitigation: File is `package cmd` (confirmed). Add `printPDMetrics` tests directly there.

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions (`PDMetrics` is a plain struct; `collectPoolThroughput` is a private helper)
- [x] No feature creep beyond PR3 scope (`CompletionTime`-based E2E deferred; no autoscaling signals)
- [x] No unexercised flags or interfaces (no new interfaces; `PD *PDMetrics` is nil-checked before use)
- [x] No partial implementations (all 6 fields of `PDMetrics` are populated in Task 1)
- [x] No breaking changes without explicit contract updates (`RawMetrics.PD` is new nullable field; existing literals safe)
- [x] No hidden global state impact (`CollectPDMetrics` is a pure function; no package-level variables)
- [x] All new code will pass golangci-lint (verified against existing patterns)
- [x] Shared test helpers used from existing package (disaggregation_test.go helpers)
- [x] CLAUDE.md updated (Task 6 adds `pd_metrics.go` to file tree)
- [x] No stale references (no "planned for PR N" text added)
- [x] Documentation DRY (only CLAUDE.md file tree updated; no duplication of canonical docs)
- [x] Deviation log reviewed — all deviations justified (JSON output, CompletionTime, load imbalance proxy)
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Tasks 1→2→3→4→5→6)
- [x] All contracts mapped to specific tasks (see Section H)
- [x] Golden dataset regeneration not needed (new section only prints for disaggregated runs; existing golden tests use single-instance mode)
- [x] Construction site audit completed (6 `&RawMetrics{...}` sites listed; all safe with nil pointer default)
- [x] Macro plan status: PR3 = this PR. When merged to `pd` branch, mark PR3 complete in macro plan.

**Antipattern rules:**
- [x] R1: No silent returns — `CollectPDMetrics` returns nil explicitly (documented)
- [x] R2: Map keys sorted — `sort.Slice` on parents by ID; `sort.Strings` on instance IDs in `collectPoolThroughput`
- [x] R3: No new numeric parameters requiring validation (all inputs are derived from simulation state)
- [x] R4: All `RawMetrics` literal sites audited (6 sites; all use named fields; nil default safe)
- [x] R5: No resource-allocating loops (pure computation, no KV/memory allocation)
- [x] R6: No `logrus.Fatalf` in `sim/cluster/` (new code is in `sim/cluster/` — no Fatalf used)
- [x] R7: Invariant tests `TestClusterSimulator_PDMetricsInvariant_PoolConservation` (pool conservation) and `TestCollectPDMetrics_ParentTTFT_IncludesTransferDuration` (BC-1 TTFT causality) alongside integration tests
- [x] R8: `PerInstanceMetricsByID()` returns a new map copy — not the internal map (R8 compliant)
- [x] R9: No YAML fields with float zero validity (no YAML in this PR)
- [x] R10: No YAML parsing in this PR
- [x] R11: Division guarded — `collectPoolThroughput` uses switch: `maxRPS==0→1.0`, `minRPS<=0→math.MaxFloat64`, else `maxRPS/minRPS`
- [x] R12: Golden dataset not affected (no output format change for non-disaggregated runs)
- [x] R13: No new interfaces (function, not interface)
- [x] R14: `CollectPDMetrics` is pure metrics computation — no scheduling/latency crossover
- [x] R15: No stale PR references added
- [x] R16: No config parameters added
- [x] R17: No routing signals modified
- [x] R18: No CLI flags or defaults.yaml interaction
- [x] R19: No retry loops
- [x] R20: `CollectPDMetrics` handles empty inputs (returns nil); `NewDistribution` handles empty slice
- [x] R21: No range over shrinking slices
- [x] R22: No capacity pre-checks
- [x] R23: No parallel code paths

---

## APPENDIX: File-Level Implementation Details

### File: `sim/cluster/pd_metrics.go`

**Purpose:** Pure function computing disaggregation-aware metrics from ParentRequest lifecycle timestamps and per-instance completion counts.

**Complete implementation:** (see Task 1, Step 3 above — the full file content is given there)

**Key implementation notes:**
- RNG: None (pure computation)
- Metrics: Input-only, no recording
- Event ordering: N/A (post-simulation)
- State mutation: None (pure function, no side effects)
- Error handling: No panics; nil return for empty parents; 0-value exclusion for missing TTFT

---

### File: `sim/cluster/pd_metrics_test.go`

**Purpose:** Unit and invariant tests for `CollectPDMetrics` — direct construction without ClusterSimulator dependency.

**Complete implementation:** (see Task 1 Step 1 + Task 2 Step 1 for full test code)

**Key notes:**
- Package: `cluster` (same package, can access types directly)
- Helper: `buildAggregatedWithTTFTs` constructs minimal `sim.Metrics` with given TTFT map
- All tests are independent (no shared state, no init())

---

### File: `sim/cluster/metrics.go` (modification)

**Purpose:** Add `PD *PDMetrics` nullable field to `RawMetrics`.

**Change:** Add one line to the `RawMetrics` struct after `KVThrashingRate`:
```go
// PD disaggregation metrics (PR3). Nil when disaggregation is not active.
PD *PDMetrics
```

**R4 audit:** All 6 struct literal construction sites listed in Section F use named fields — the new field defaults to nil without code changes.

---

### File: `sim/cluster/cluster.go` (modification)

**Purpose:** Add `ParentRequests()` and `PerInstanceMetricsByID()` public accessors.

**Location:** After `PerInstanceMetrics()` at line 367.

**Complete code:** (see Task 3, Step 3 above)

**Import:** Verify `"sort"` is present in the import block. It is used by `sort.Slice` in `ParentRequests()`. Looking at cluster.go imports (lines 3-11), `"sort"` is NOT currently imported. Add it.

---

### File: `cmd/root.go` (modification)

**Purpose:** Wire `CollectPDMetrics` call and `printPDMetrics` output function.

**Change 1:** After line 815 (after `CollectRawMetrics` call), add:
```go
// Collect PD metrics when disaggregation was active (PR3).
rawMetrics.PD = cluster.CollectPDMetrics(
    cs.ParentRequests(),
    cs.AggregatedMetrics(),
    cs.PoolMembership(),
    cs.PerInstanceMetricsByID(),
)
```

**Change 2:** After line 849 (`printKVCacheMetrics` call), add:
```go
// Print PD disaggregation metrics if active (PR3)
printPDMetrics(os.Stdout, rawMetrics.PD)
```

**Change 3:** Add `printPDMetrics` function after `printKVCacheMetrics` (around line 891):
```go
// printPDMetrics prints disaggregation-aware metrics to w when pd is non-nil.
func printPDMetrics(w io.Writer, pd *cluster.PDMetrics) {
    if pd == nil {
        return
    }
    _, _ = fmt.Fprintln(w, "=== PD Metrics ===")
    _, _ = fmt.Fprintf(w, "Disaggregated Requests: %d\n", pd.DisaggregatedCount)
    _, _ = fmt.Fprintf(w, "Prefill Throughput: %.4f sub-req/s\n", pd.PrefillThroughput)
    _, _ = fmt.Fprintf(w, "Decode Throughput: %.4f sub-req/s\n", pd.DecodeThroughput)
    // math.MaxFloat64 sentinel = one pool completely idle (extreme imbalance).
    if pd.LoadImbalanceRatio >= math.MaxFloat64/2 {
        _, _ = fmt.Fprintf(w, "Load Imbalance Ratio: inf (one pool idle)\n")
    } else {
        _, _ = fmt.Fprintf(w, "Load Imbalance Ratio: %.4f\n", pd.LoadImbalanceRatio)
    }
    if pd.ParentTTFT.Count > 0 {
        _, _ = fmt.Fprintf(w, "Parent TTFT (μs): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
            pd.ParentTTFT.Mean, pd.ParentTTFT.P50, pd.ParentTTFT.P95, pd.ParentTTFT.P99)
    }
    if pd.TransferDuration.Count > 0 {
        _, _ = fmt.Fprintf(w, "KV Transfer Duration (μs): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
            pd.TransferDuration.Mean, pd.TransferDuration.P50, pd.TransferDuration.P95, pd.TransferDuration.P99)
    }
}
```

**Note:** `cmd/kv_metrics_output_test.go` is `package cmd` (confirmed). Add `printPDMetrics` tests directly to this file alongside the existing `printKVCacheMetrics` and `printPerSLOMetrics` tests.

---

### File: `CLAUDE.md` (modification)

**Purpose:** Keep file organization tree accurate per the checklist requirement.

**Change:** Add `pd_metrics.go` entry between `pool.go` and `evaluation.go` in the `sim/cluster/` section (see Task 6, Step 2 for exact text).
