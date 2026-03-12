# PR4: Disaggregation Decision Trace Support

**Goal:** Add decision trace records for disaggregation decisions, KV transfer events, and per-pool routing choices in the PD disaggregation pipeline.

**The problem today:** When running a disaggregated simulation, the trace output only captures admission decisions and standard routing decisions. Disaggregation decisions, prefill-pool routing, decode-pool routing, and KV transfer events are invisible to trace analysis — users cannot audit why a request was disaggregated, which prefill instance was chosen, how long KV transfer took, or where decode was routed.

**What this PR adds:**
1. `DisaggregationRecord` — captures whether each request was disaggregated or routed locally, at what clock tick.
2. `PrefillRoutingRecord` — captures prefill pool routing decision with optional counterfactual analysis (top-k candidates, regret).
3. `DecodeRoutingRecord` — captures decode pool routing decision with optional counterfactual analysis.
4. `KVTransferRecord` — captures KV transfer start time, duration, block count, and both instance IDs.

**Why this matters:** Without disaggregation-aware traces, users cannot debug why requests take unexpected paths through the disaggregated pipeline or compare routing quality across prefill and decode pools. This PR is the observability layer for the disaggregated flow introduced in PR2.

**Architecture:** All new record types are pure data in `sim/trace/record.go` (no sim/ dependency). `SimulationTrace` in `sim/trace/trace.go` gains four new record slices and four `Record*` methods. Instrumentation calls are added to `DisaggregationDecisionEvent` in `sim/cluster/cluster_event.go` and to `PrefillRoutingEvent`, `DecodeRoutingEvent` in `sim/cluster/pd_events.go`. `KVTransferRecord` is emitted inside `DecodeRoutingEvent.Execute()` (after decode instance is assigned) so all fields are fully populated.

**Source:** Macro plan PR4 section ([#577 comment](https://github.com/inference-sim/inference-sim/issues/577#issuecomment-4032683694)), issue [#594](https://github.com/inference-sim/inference-sim/issues/594).

**Closes:** Fixes #594

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

**Building block modified:** Trace recording subsystem (trace extension, per macro plan PR4 classification).

**Adjacent blocks:**
- `sim/cluster/cluster_event.go` — `DisaggregationDecisionEvent` (caller)
- `sim/cluster/pd_events.go` — `PrefillRoutingEvent`, `KVTransferCompletedEvent`, `DecodeRoutingEvent` (callers)
- `sim/cluster/counterfactual.go` — `computeCounterfactual()` (reused for pool routing records)
- `sim/trace/` — modified (owned by this PR)

**Invariants touched:** None. This PR is pure observability — it only reads state produced by PR2 events. No simulation behavior changes.

**Construction site audit:**

`SimulationTrace` struct: constructed in exactly one place — `trace.NewSimulationTrace()` in `sim/trace/trace.go:39`. That canonical constructor is the only construction site (R4 compliant). This PR adds 4 new slice fields and updates the constructor initializer to include them.

No other struct has fields added by this PR.

---

## Part 1: Design Validation

### A) Executive Summary

PR4 is pure observability — it adds 4 new trace record types and wires them into the 4 PD event handlers introduced by PR1/PR2. No simulation behavior changes. No new CLI flags (controlled by existing `--trace-level` and `--counterfactual-k`).

The 4 new record types live in `sim/trace/record.go` (no dependencies). `SimulationTrace` gains 4 new slices and 4 `Record*` methods. Three event handlers in `sim/cluster/` call these methods when `cs.trace != nil`. `KVTransferRecord` is emitted from `DecodeRoutingEvent.Execute()` (not `KVTransferCompletedEvent`) so that `DecodeInstanceID` is populated before recording.

Counterfactual analysis (`computeCounterfactual`) is reused for `PrefillRoutingRecord` and `DecodeRoutingRecord` — pool-filtered snapshots are already local variables in both routing event handlers.

The existing `--trace-level decisions` and `--counterfactual-k` flags control the new records identically to existing records: `cs.trace == nil` when tracing is off (zero overhead), and `CounterfactualK > 0` activates counterfactual analysis for pool routing records.

Non-disaggregated simulations produce zero disaggregation-specific records (BC-PD-18) because the PD event handlers never fire when `cs.poolsConfigured() == false`.

### B) Behavioral Contracts

**BC-PD-17: Record coverage**
- GIVEN a disaggregated simulation (`--prefill-instances > 0 --decode-instances > 0 --pd-decider always`) with `--trace-level decisions`
- WHEN the simulation completes
- THEN the trace contains exactly one DisaggregationRecord, one PrefillRoutingRecord, one KVTransferRecord, and one DecodeRoutingRecord for each disaggregated request; every record has a non-empty ChosenInstance (for routing records) or non-zero TransferDuration (for KV records)

**BC-PD-18: Non-disaggregated trace isolation**
- GIVEN a simulation without pool configuration (`--prefill-instances=0 --decode-instances=0`) with `--trace-level decisions`
- WHEN the simulation completes
- THEN the trace contains zero DisaggregationRecords, zero PrefillRoutingRecords, zero DecodeRoutingRecords, and zero KVTransferRecords

**BC-PD-19: Counterfactual for pool routing**
- GIVEN a disaggregated simulation with `--trace-level decisions` and `--counterfactual-k 2`, and both pools have ≥ 2 instances
- WHEN the simulation completes
- THEN every PrefillRoutingRecord and DecodeRoutingRecord has Candidates slice with len ≤ 2 and Regret ≥ 0

**BC-TRACE-COMPAT: Existing trace records unchanged**
- GIVEN a non-disaggregated simulation with `--trace-level decisions`
- WHEN the simulation completes
- THEN admission records and routing records are present as before (byte-identical to pre-PR4)

### C) Component Interaction

```
cluster_event.go                sim/trace/                sim/cluster/
┌──────────────────┐            ┌─────────────────────┐
│ DisaggDecision   │──Record──► │ SimulationTrace      │
│  Event.Execute() │            │  .Disaggregations    │
└──────────────────┘            │  .PrefillRoutings    │
                                │  .DecodeRoutings     │
pd_events.go                    │  .KVTransfers        │
┌──────────────────┐            │  .RecordDisagg()     │
│ PrefillRouting   │──Record──► │  .RecordPrefill()    │
│  Event.Execute() │            │  .RecordDecode()     │
│                  │            │  .RecordKVTransfer() │
│ DecodeRouting    │──Record──► │                      │
│  Event.Execute() │──KVTransfer└─────────────────────┘
└──────────────────┘

counterfactual.go
┌──────────────────┐
│ computeCounter   │◄── called by PrefillRouting and DecodeRouting
│  factual()       │    with pool-filtered snapshots
└──────────────────┘

State ownership:
- Record types: owned by trace package (pure data, no pointers to sim/ types)
- Slices in SimulationTrace: appended by Record* methods, read by callers
- filteredSnapshots: local variables in routing event handlers, passed to computeCounterfactual
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| KVTransferRecord: "block count, duration, prefill instance, decode instance" | KVTransferRecord is recorded in DecodeRoutingEvent.Execute() (not KVTransferCompletedEvent) | CORRECTION: DecodeInstanceID is unknown until DecodeRoutingEvent fires. Recording in DecodeRoutingEvent guarantees all 4 fields are populated. |
| Trace records include counterfactual for per-pool routing | Counterfactual uses pool-filtered snapshots (not all-instances snapshots) | CLARIFICATION: Pool routing decisions are made with pool-filtered snapshots; counterfactual must use the same candidate set (comparing filtered pool members) to be meaningful. |

### E) Review Guide

**The tricky part:** `KVTransferRecord` is recorded in `DecodeRoutingEvent.Execute()`, not `KVTransferCompletedEvent.Execute()`. This is a deliberate choice (see Deviation Log) — `DecodeInstanceID` is set in `DecodeRoutingEvent.Execute()` and was not available earlier. Verify that `e.parentReq.TransferStartTime` and `e.parentReq.TransferCompleteTime` are both set before `DecodeRoutingEvent` fires (they are: `TransferStartTime` set in `KVTransferStartedEvent.Execute()`, `TransferCompleteTime` set in `KVTransferCompletedEvent.Execute()`).

**Scrutinize:** The pool-filtered snapshot slice passed to `computeCounterfactual` in pool routing events. It must use `filteredSnapshots` (pool-only), not the full `buildRouterState()` snapshots. Using the wrong snapshot set would produce counterfactual candidates from the wrong pool.

**Safe to skim:** Record field definitions (pure data), `NewSimulationTrace` update (trivial slice initialization), `RecordDisaggregation` call (no complexity).

**Known debt:** `DisaggregationDecision` has no `Reason` field today; `DisaggregationRecord.Disaggregate` is the only information available. PR5 (PrefixThresholdDecider) can optionally add a reason to `DisaggregationDecision` if needed.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/trace/record.go` — add 4 new record types
- `sim/trace/trace.go` — add 4 new slices to `SimulationTrace`, update `NewSimulationTrace`, add 4 `Record*` methods
- `sim/cluster/cluster_event.go` — instrument `DisaggregationDecisionEvent.Execute()`
- `sim/cluster/pd_events.go` — instrument `PrefillRoutingEvent.Execute()`, `DecodeRoutingEvent.Execute()`

**Files to create:**
- `sim/cluster/pd_traces_test.go` — new tests for BC-PD-17, BC-PD-18, BC-PD-19

**Existing test files updated:**
- `sim/trace/trace_test.go` — add unit tests for new record types and methods

**Key decisions:**
1. `KVTransferRecord` recorded in `DecodeRoutingEvent.Execute()` so all 4 fields are populated
2. Counterfactual for pool routing uses `filteredSnapshots` (pool-scoped, already a local variable)
3. All 4 new fields in `SimulationTrace` initialized as empty slices (never nil) so callers can always range over them safely

**No dead code:** All 4 new record types are emitted in disaggregated simulations exercised by tests. All 4 `Record*` methods are called from production event handlers.

### G) Task Breakdown

---

#### Task 1: Add trace record types and recording methods

**Contracts:** BC-PD-17, BC-PD-18, BC-PD-19 (data foundations)
**Files:** modify `sim/trace/record.go`, modify `sim/trace/trace.go`, modify `sim/trace/trace_test.go`

**Step 1: Write failing test**

Add to `sim/trace/trace_test.go`:

```go
func TestSimulationTrace_NewRecordTypes_Initialized(t *testing.T) {
    // GIVEN a new SimulationTrace
    tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})

    // WHEN checked immediately after creation
    // THEN all PD-specific slices are non-nil and empty (not nil)
    if tr.Disaggregations == nil {
        t.Error("Disaggregations slice is nil, want empty non-nil slice")
    }
    if tr.PrefillRoutings == nil {
        t.Error("PrefillRoutings slice is nil, want empty non-nil slice")
    }
    if tr.DecodeRoutings == nil {
        t.Error("DecodeRoutings slice is nil, want empty non-nil slice")
    }
    if tr.KVTransfers == nil {
        t.Error("KVTransfers slice is nil, want empty non-nil slice")
    }
    if len(tr.Disaggregations) != 0 || len(tr.PrefillRoutings) != 0 ||
        len(tr.DecodeRoutings) != 0 || len(tr.KVTransfers) != 0 {
        t.Error("expected all PD slices empty after construction")
    }
}

func TestSimulationTrace_RecordDisaggregation(t *testing.T) {
    tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
    tr.RecordDisaggregation(DisaggregationRecord{RequestID: "req_0", Clock: 100, Disaggregate: true})
    tr.RecordDisaggregation(DisaggregationRecord{RequestID: "req_1", Clock: 200, Disaggregate: false})

    if len(tr.Disaggregations) != 2 {
        t.Fatalf("expected 2 disaggregation records, got %d", len(tr.Disaggregations))
    }
    if !tr.Disaggregations[0].Disaggregate {
        t.Error("first record: Disaggregate should be true")
    }
    if tr.Disaggregations[1].Disaggregate {
        t.Error("second record: Disaggregate should be false")
    }
}

func TestSimulationTrace_RecordPrefillRouting(t *testing.T) {
    tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
    tr.RecordPrefillRouting(PrefillRoutingRecord{
        ParentRequestID: "req_0",
        Clock:           150,
        ChosenInstance:  "instance_0",
        Scores:          map[string]float64{"instance_0": 0.9, "instance_1": 0.7},
        Regret:          0.2,
    })

    if len(tr.PrefillRoutings) != 1 {
        t.Fatalf("expected 1 prefill routing record, got %d", len(tr.PrefillRoutings))
    }
    r := tr.PrefillRoutings[0]
    if r.ChosenInstance != "instance_0" {
        t.Errorf("ChosenInstance = %q, want %q", r.ChosenInstance, "instance_0")
    }
    if r.Regret != 0.2 {
        t.Errorf("Regret = %f, want 0.2", r.Regret)
    }
}

func TestSimulationTrace_RecordKVTransfer(t *testing.T) {
    tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
    tr.RecordKVTransfer(KVTransferRecord{
        ParentRequestID:   "req_0",
        TransferStartTime: 500,
        TransferDuration:  42,
        NumKVBlocks:       7,
        PrefillInstanceID: "instance_0",
        DecodeInstanceID:  "instance_2",
    })

    if len(tr.KVTransfers) != 1 {
        t.Fatalf("expected 1 KV transfer record, got %d", len(tr.KVTransfers))
    }
    r := tr.KVTransfers[0]
    if r.TransferDuration != 42 {
        t.Errorf("TransferDuration = %d, want 42", r.TransferDuration)
    }
    if r.NumKVBlocks != 7 {
        t.Errorf("NumKVBlocks = %d, want 7", r.NumKVBlocks)
    }
    if r.PrefillInstanceID != "instance_0" || r.DecodeInstanceID != "instance_2" {
        t.Errorf("instance IDs = (%q, %q), want (instance_0, instance_2)",
            r.PrefillInstanceID, r.DecodeInstanceID)
    }
}

func TestSimulationTrace_RecordDecodeRouting(t *testing.T) {
    tr := NewSimulationTrace(TraceConfig{Level: TraceLevelDecisions})
    tr.RecordDecodeRouting(DecodeRoutingRecord{
        ParentRequestID: "req_0",
        Clock:           600,
        ChosenInstance:  "instance_2",
        Candidates: []CandidateScore{
            {InstanceID: "instance_2", Score: 0.8},
            {InstanceID: "instance_3", Score: 0.6},
        },
        Regret: 0.0,
    })

    if len(tr.DecodeRoutings) != 1 {
        t.Fatalf("expected 1 decode routing record, got %d", len(tr.DecodeRoutings))
    }
    r := tr.DecodeRoutings[0]
    if r.ChosenInstance != "instance_2" {
        t.Errorf("ChosenInstance = %q, want %q", r.ChosenInstance, "instance_2")
    }
    if len(r.Candidates) != 2 {
        t.Errorf("Candidates len = %d, want 2", len(r.Candidates))
    }
}
```

**Step 2: Run test — expect failure**
```bash
cd /ws/fork/inference-sim/.worktrees/pr4-pd-traces
go test ./sim/trace/... -run TestSimulationTrace_NewRecordTypes 2>&1 | tail -5
# Expected: FAIL (undefined: DisaggregationRecord, PrefillRoutingRecord, etc.)
```

**Step 3: Implement**

`sim/trace/record.go` — append to file:
```go
// DisaggregationRecord captures a PD disaggregation decision.
type DisaggregationRecord struct {
	RequestID    string
	Clock        int64
	Disaggregate bool // true = routed to prefill pool; false = standard routing
}

// PrefillRoutingRecord captures a prefill pool routing decision with optional counterfactual analysis.
type PrefillRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	Scores          map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates      []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret          float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// DecodeRoutingRecord captures a decode pool routing decision with optional counterfactual analysis.
type DecodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	Scores          map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates      []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret          float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// KVTransferRecord captures a KV cache transfer event between prefill and decode instances.
type KVTransferRecord struct {
	ParentRequestID   string
	TransferStartTime int64 // microseconds (sim clock)
	TransferDuration  int64 // microseconds
	NumKVBlocks       int64
	PrefillInstanceID string
	DecodeInstanceID  string
}
```

`sim/trace/trace.go` — update `SimulationTrace` struct, `NewSimulationTrace`, add 4 methods:
```go
// SimulationTrace collects decision records during a cluster simulation.
type SimulationTrace struct {
	Config          TraceConfig
	Admissions      []AdmissionRecord
	Routings        []RoutingRecord
	Disaggregations []DisaggregationRecord
	PrefillRoutings []PrefillRoutingRecord
	DecodeRoutings  []DecodeRoutingRecord
	KVTransfers     []KVTransferRecord
}

// NewSimulationTrace creates a SimulationTrace ready for recording.
func NewSimulationTrace(config TraceConfig) *SimulationTrace {
	return &SimulationTrace{
		Config:          config,
		Admissions:      make([]AdmissionRecord, 0),
		Routings:        make([]RoutingRecord, 0),
		Disaggregations: make([]DisaggregationRecord, 0),
		PrefillRoutings: make([]PrefillRoutingRecord, 0),
		DecodeRoutings:  make([]DecodeRoutingRecord, 0),
		KVTransfers:     make([]KVTransferRecord, 0),
	}
}

// RecordDisaggregation appends a disaggregation decision record.
func (st *SimulationTrace) RecordDisaggregation(record DisaggregationRecord) {
	st.Disaggregations = append(st.Disaggregations, record)
}

// RecordPrefillRouting appends a prefill pool routing decision record.
func (st *SimulationTrace) RecordPrefillRouting(record PrefillRoutingRecord) {
	st.PrefillRoutings = append(st.PrefillRoutings, record)
}

// RecordDecodeRouting appends a decode pool routing decision record.
func (st *SimulationTrace) RecordDecodeRouting(record DecodeRoutingRecord) {
	st.DecodeRoutings = append(st.DecodeRoutings, record)
}

// RecordKVTransfer appends a KV transfer event record.
func (st *SimulationTrace) RecordKVTransfer(record KVTransferRecord) {
	st.KVTransfers = append(st.KVTransfers, record)
}
```

**Step 4: Run test — expect pass**
```bash
go test ./sim/trace/... -run "TestSimulationTrace_NewRecordTypes|TestSimulationTrace_RecordDisagg|TestSimulationTrace_RecordPrefill|TestSimulationTrace_RecordKVTransfer|TestSimulationTrace_RecordDecodeRouting" -v 2>&1 | tail -15
# Expected: PASS (all 5 tests)
```

**Step 5: Lint**
```bash
golangci-lint run ./sim/trace/... 2>&1 | tail -5
# Expected: no issues
```

**Step 6: Commit**
```bash
git add sim/trace/record.go sim/trace/trace.go sim/trace/trace_test.go
git commit -m "feat(trace): add PD disaggregation trace record types (BC-PD-17)

- Add DisaggregationRecord, PrefillRoutingRecord, DecodeRoutingRecord, KVTransferRecord
- Add Disaggregations, PrefillRoutings, DecodeRoutings, KVTransfers slices to SimulationTrace
- Add RecordDisaggregation, RecordPrefillRouting, RecordDecodeRouting, RecordKVTransfer methods
- Update NewSimulationTrace constructor to initialize all new slices

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Instrument DisaggregationDecisionEvent and write BC-PD-18 test

**Contracts:** BC-PD-17 (partial), BC-PD-18
**Files:** modify `sim/cluster/cluster_event.go`, create `sim/cluster/pd_traces_test.go`

**Step 1: Write failing test**

Create `sim/cluster/pd_traces_test.go`:

```go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// TestPDTrace_NonDisaggMode_NoDisaggRecords verifies BC-PD-18:
// when disaggregation is not configured, no PD-specific trace records are emitted.
func TestPDTrace_NonDisaggMode_NoDisaggRecords(t *testing.T) {
	// GIVEN non-disaggregated simulation with trace enabled
	config := DeploymentConfig{
		SimConfig: sim.SimConfig{
			Horizon:       10000000,
			Seed:          42,
			KVCacheConfig: sim.NewKVCacheConfig(100, 16, 0, 0, 0, 0),
			BatchConfig:   sim.NewBatchConfig(10, 2048, 0),
			LatencyCoeffs: sim.NewLatencyCoeffs([]float64{1000, 10, 5}, []float64{100, 50, 25}),
		},
		NumInstances: 4,
		TraceLevel:   "decisions",
	}
	requests := newTestRequests(5)
	cs := NewClusterSimulator(config, requests)

	// WHEN run
	mustRun(t, cs)

	// THEN no disaggregation-specific trace records
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace with trace-level decisions")
	}
	if len(tr.Disaggregations) != 0 {
		t.Errorf("expected 0 disaggregation records in non-disagg mode, got %d", len(tr.Disaggregations))
	}
	if len(tr.PrefillRoutings) != 0 {
		t.Errorf("expected 0 prefill routing records in non-disagg mode, got %d", len(tr.PrefillRoutings))
	}
	if len(tr.DecodeRoutings) != 0 {
		t.Errorf("expected 0 decode routing records in non-disagg mode, got %d", len(tr.DecodeRoutings))
	}
	if len(tr.KVTransfers) != 0 {
		t.Errorf("expected 0 KV transfer records in non-disagg mode, got %d", len(tr.KVTransfers))
	}
	// Existing admission/routing records still present (BC-TRACE-COMPAT)
	if len(tr.Admissions) != 5 {
		t.Errorf("expected 5 admission records, got %d", len(tr.Admissions))
	}
	if len(tr.Routings) != 5 {
		t.Errorf("expected 5 routing records, got %d", len(tr.Routings))
	}
}

// TestPDTrace_DisaggMode_DisaggDecisionRecorded verifies disaggregation decisions are recorded.
func TestPDTrace_DisaggMode_DisaggDecisionRecorded(t *testing.T) {
	// GIVEN disaggregated simulation with trace enabled
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)

	// WHEN run
	mustRun(t, cs)

	// THEN exactly 3 disaggregation records (one per request), all Disaggregate=true
	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}
	if len(tr.Disaggregations) != 3 {
		t.Errorf("expected 3 disaggregation records (one per request), got %d", len(tr.Disaggregations))
	}
	for i, r := range tr.Disaggregations {
		if !r.Disaggregate {
			t.Errorf("disaggregation[%d]: Disaggregate=false, want true (AlwaysDisaggregate)", i)
		}
		if r.RequestID == "" {
			t.Errorf("disaggregation[%d]: RequestID empty", i)
		}
	}
}
```

**Step 2: Run test — expect failure**
```bash
go test ./sim/cluster/... -run "TestPDTrace_NonDisaggMode|TestPDTrace_DisaggMode_DisaggDecision" 2>&1 | tail -5
# Expected: FAIL (Disaggregations is nil or 0 after run — not yet instrumented)
```

**Step 3: Implement**

First, update the stale `ClusterEvent.Priority()` comment in `sim/cluster/cluster_event.go` (line 17):

Replace:
```go
Priority() int // 0=Arrival, 1=Admission, 2=Routing, 3=Disaggregation
```
With:
```go
Priority() int // 0=Arrival, 1=Admission, 2=Routing, 3=Disaggregation, 4=PrefillRouting, 5=KVTransferStarted, 6=KVTransferCompleted, 7=DecodeRouting
```

Then, in `DisaggregationDecisionEvent.Execute()`, after calling `cs.disaggregationDecider.Decide(e.request)` and before the `if !decision.Disaggregate` branch:

```go
// Record disaggregation decision if tracing is enabled (BC-PD-17)
if cs.trace != nil {
    cs.trace.RecordDisaggregation(trace.DisaggregationRecord{
        RequestID:    e.request.ID,
        Clock:        cs.clock,
        Disaggregate: decision.Disaggregate,
    })
}
```

**Step 4: Run test — expect pass**
```bash
go test ./sim/cluster/... -run "TestPDTrace_NonDisaggMode|TestPDTrace_DisaggMode_DisaggDecision" -v 2>&1 | tail -10
# Expected: PASS
```

**Step 5: Lint**
```bash
golangci-lint run ./sim/cluster/... 2>&1 | tail -5
```

**Step 6: Commit**
```bash
git add sim/cluster/cluster_event.go sim/cluster/pd_traces_test.go
git commit -m "feat(cluster): instrument DisaggregationDecisionEvent with trace recording (BC-PD-18)

- Record DisaggregationRecord for every disaggregation decision when trace enabled
- Non-disaggregated mode: no disaggregation records (BC-PD-18)
- Update ClusterEvent.Priority() comment to include full PD priority range (4-7)
- Add TestPDTrace_NonDisaggMode_NoDisaggRecords and TestPDTrace_DisaggMode_DisaggDecisionRecorded

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Instrument PrefillRoutingEvent and DecodeRoutingEvent (with KVTransferRecord)

**Contracts:** BC-PD-17 (full coverage), BC-PD-19
**Files:** modify `sim/cluster/pd_events.go`, modify `sim/cluster/pd_traces_test.go`

**Step 1: Write failing test**

Add to `sim/cluster/pd_traces_test.go`:

```go
// TestPDTrace_DisaggMode_AllRecordTypesPresent verifies BC-PD-17:
// all 4 PD trace record types are emitted for each disaggregated request.
func TestPDTrace_DisaggMode_AllRecordTypesPresent(t *testing.T) {
	const numRequests = 5
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	requests := newTestRequests(numRequests)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// BC-PD-17: one record of each type per disaggregated request
	if len(tr.Disaggregations) != numRequests {
		t.Errorf("Disaggregations: expected %d, got %d", numRequests, len(tr.Disaggregations))
	}
	if len(tr.PrefillRoutings) != numRequests {
		t.Errorf("PrefillRoutings: expected %d, got %d", numRequests, len(tr.PrefillRoutings))
	}
	if len(tr.KVTransfers) != numRequests {
		t.Errorf("KVTransfers: expected %d, got %d", numRequests, len(tr.KVTransfers))
	}
	if len(tr.DecodeRoutings) != numRequests {
		t.Errorf("DecodeRoutings: expected %d, got %d", numRequests, len(tr.DecodeRoutings))
	}

	// Verify KVTransfer records have both instance IDs and non-zero duration
	for i, kv := range tr.KVTransfers {
		if kv.PrefillInstanceID == "" {
			t.Errorf("KVTransfers[%d]: PrefillInstanceID empty", i)
		}
		if kv.DecodeInstanceID == "" {
			t.Errorf("KVTransfers[%d]: DecodeInstanceID empty", i)
		}
		if kv.TransferDuration <= 0 {
			t.Errorf("KVTransfers[%d]: TransferDuration=%d, want > 0", i, kv.TransferDuration)
		}
		if kv.NumKVBlocks <= 0 {
			t.Errorf("KVTransfers[%d]: NumKVBlocks=%d, want > 0", i, kv.NumKVBlocks)
		}
	}

	// Verify per-pool routing records have non-empty chosen instances
	for i, r := range tr.PrefillRoutings {
		if r.ChosenInstance == "" {
			t.Errorf("PrefillRoutings[%d]: ChosenInstance empty", i)
		}
		if r.ParentRequestID == "" {
			t.Errorf("PrefillRoutings[%d]: ParentRequestID empty", i)
		}
	}
	for i, r := range tr.DecodeRoutings {
		if r.ChosenInstance == "" {
			t.Errorf("DecodeRoutings[%d]: ChosenInstance empty", i)
		}
		if r.ParentRequestID == "" {
			t.Errorf("DecodeRoutings[%d]: ParentRequestID empty", i)
		}
	}
}

// TestPDTrace_DisaggMode_Counterfactual verifies BC-PD-19:
// per-pool routing records have counterfactual candidates when k > 0.
func TestPDTrace_DisaggMode_Counterfactual(t *testing.T) {
	// GIVEN disaggregated simulation with k=2, 2 prefill + 2 decode instances
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	config.TraceLevel = "decisions"
	config.CounterfactualK = 2
	config.RoutingPolicy = "weighted"
	config.RoutingScorerConfigs = sim.DefaultScorerConfigs()
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests)

	mustRun(t, cs)

	tr := cs.Trace()
	if tr == nil {
		t.Fatal("expected non-nil trace")
	}

	// THEN prefill routing records have candidates (BC-PD-19)
	for i, r := range tr.PrefillRoutings {
		if len(r.Candidates) == 0 {
			t.Errorf("PrefillRoutings[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("PrefillRoutings[%d]: expected ≤2 candidates, got %d", i, len(r.Candidates))
		}
		if r.Regret < 0 {
			t.Errorf("PrefillRoutings[%d]: Regret=%f, want ≥0", i, r.Regret)
		}
	}

	// THEN decode routing records have candidates (BC-PD-19)
	for i, r := range tr.DecodeRoutings {
		if len(r.Candidates) == 0 {
			t.Errorf("DecodeRoutings[%d]: expected candidates with k=2, got none", i)
		}
		if len(r.Candidates) > 2 {
			t.Errorf("DecodeRoutings[%d]: expected ≤2 candidates, got %d", i, len(r.Candidates))
		}
		if r.Regret < 0 {
			t.Errorf("DecodeRoutings[%d]: Regret=%f, want ≥0", i, r.Regret)
		}
	}
}
```

**Step 2: Run tests — expect failure**
```bash
go test ./sim/cluster/... -run "TestPDTrace_DisaggMode_AllRecordTypes|TestPDTrace_DisaggMode_Counterfactual" 2>&1 | tail -5
# Expected: FAIL (PrefillRoutings, KVTransfers, DecodeRoutings empty — not yet instrumented)
```

**Step 3: Implement**

In `sim/cluster/pd_events.go`, `PrefillRoutingEvent.Execute()`, after `e.parentReq.PrefillEnqueueTime = e.time`:

```go
// Record prefill routing decision if tracing is enabled (BC-PD-17, BC-PD-19)
if cs.trace != nil {
    record := trace.PrefillRoutingRecord{
        ParentRequestID: e.parentReq.ID,
        Clock:           cs.clock,
        ChosenInstance:  decision.TargetInstance,
        Scores:          copyScores(decision.Scores),
    }
    if cs.trace.Config.CounterfactualK > 0 {
        record.Candidates, record.Regret = computeCounterfactual(
            decision.TargetInstance, decision.Scores,
            filteredSnapshots, cs.trace.Config.CounterfactualK,
        )
    }
    cs.trace.RecordPrefillRouting(record)
}
```

In `sim/cluster/pd_events.go`, `DecodeRoutingEvent.Execute()`: replace the entire `for _, inst := range cs.instances` loop (lines ~169-186) with the restructured version below. The trace recording happens **inside the successful KV allocation path** (after `AllocateTransferredKV` returns `true`), not before the loop. Recording before the allocation check would create orphan trace records for dropped requests (violates R1).

Note: the `trace` package must be imported in `pd_events.go`. Add to the import block:
```go
"github.com/inference-sim/inference-sim/sim/trace"
```

**Replace `for _, inst := range cs.instances` loop with:**

```go
// Find target decode instance
for _, inst := range cs.instances {
    if string(inst.ID()) == decision.TargetInstance {
        // Pre-allocate KV blocks for transferred input
        if ok := inst.AllocateTransferredKV(e.decodeSubReq); !ok {
            logrus.Warnf("[cluster] decode instance %s: insufficient KV capacity for %s (%d input tokens)",
                decision.TargetInstance, e.decodeSubReq.ID, len(e.decodeSubReq.InputTokens))
            // Cannot proceed without KV — request effectively dropped
            return
        }

        // Record KV transfer and decode routing after successful KV allocation (BC-PD-17, BC-PD-19)
        // Placement after AllocateTransferredKV ensures records only exist for requests that
        // complete the decode phase (R1: no orphan records for dropped requests).
        // KVTransferRecord is recorded here so DecodeInstanceID is fully populated.
        // Both TransferStartTime and TransferCompleteTime were set in earlier event handlers.
        if cs.trace != nil {
            cs.trace.RecordKVTransfer(trace.KVTransferRecord{
                ParentRequestID:   e.parentReq.ID,
                TransferStartTime: e.parentReq.TransferStartTime,
                TransferDuration:  e.parentReq.TransferCompleteTime - e.parentReq.TransferStartTime,
                NumKVBlocks:       e.parentReq.NumKVBlocks,
                PrefillInstanceID: e.parentReq.PrefillInstanceID,
                DecodeInstanceID:  e.parentReq.DecodeInstanceID,
            })
            decodeRecord := trace.DecodeRoutingRecord{
                ParentRequestID: e.parentReq.ID,
                Clock:           cs.clock,
                ChosenInstance:  decision.TargetInstance,
                Scores:          copyScores(decision.Scores),
            }
            if cs.trace.Config.CounterfactualK > 0 {
                decodeRecord.Candidates, decodeRecord.Regret = computeCounterfactual(
                    decision.TargetInstance, decision.Scores,
                    filteredSnapshots, cs.trace.Config.CounterfactualK,
                )
            }
            cs.trace.RecordDecodeRouting(decodeRecord)
        }

        cs.inFlightRequests[decision.TargetInstance]++
        inst.InjectDecodeOnline(e.decodeSubReq)
        return
    }
}
panic(fmt.Sprintf("DecodeRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
```

**Step 4: Run tests — expect pass**
```bash
go test ./sim/cluster/... -run "TestPDTrace" -v 2>&1 | tail -20
# Expected: all 4 TestPDTrace_* tests PASS
```

**Step 5: Lint**
```bash
golangci-lint run ./sim/cluster/... 2>&1 | tail -5
```

**Step 6: Commit**
```bash
git add sim/cluster/pd_events.go sim/cluster/pd_traces_test.go
git commit -m "feat(cluster): instrument PD event handlers with trace recording (BC-PD-17, BC-PD-19)

- Instrument PrefillRoutingEvent with PrefillRoutingRecord + counterfactual support
- Instrument DecodeRoutingEvent with DecodeRoutingRecord + counterfactual + KVTransferRecord
- KVTransferRecord recorded in DecodeRoutingEvent so DecodeInstanceID is fully populated
- Add TestPDTrace_DisaggMode_AllRecordTypesPresent (BC-PD-17)
- Add TestPDTrace_DisaggMode_Counterfactual (BC-PD-19)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Verify full test suite and update CLAUDE.md

**Contracts:** All (verification + documentation)
**Files:** modify `CLAUDE.md`

**Step 1: Run full test suite**
```bash
go test ./... -count=1 2>&1 | tail -20
# Expected: all tests pass
```

**Step 2: Run lint**
```bash
golangci-lint run ./... 2>&1 | tail -5
# Expected: no issues
```

**Step 3: Update CLAUDE.md**

In `CLAUDE.md`, update the `sim/trace/` section in the File Organization tree. Current text:
```
├── sim/trace/                 # Decision trace recording (PR13)
│   ├── trace.go               # TraceLevel, TraceConfig, SimulationTrace, NewSimulationTrace, recording methods
│   ├── record.go              # AdmissionRecord, RoutingRecord, CandidateScore (pure data types, no sim/ dependency)
│   └── summary.go             # TraceSummary, Summarize()
```

Replace with:
```
├── sim/trace/                 # Decision trace recording (PR13, extended in PR4)
│   ├── trace.go               # TraceLevel, TraceConfig, SimulationTrace, NewSimulationTrace, recording methods (RecordAdmission, RecordRouting, RecordDisaggregation, RecordPrefillRouting, RecordDecodeRouting, RecordKVTransfer)
│   ├── record.go              # AdmissionRecord, RoutingRecord, CandidateScore, DisaggregationRecord, PrefillRoutingRecord, DecodeRoutingRecord, KVTransferRecord (pure data types, no sim/ dependency)
│   └── summary.go             # TraceSummary, Summarize()
```

**Step 4: Commit**
```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md trace/ descriptions for PR4 PD trace records

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|---|---|---|---|
| BC-PD-17 (record types exist + initialized) | Task 1 | Unit | `TestSimulationTrace_NewRecordTypes_Initialized` |
| BC-PD-17 (RecordDisaggregation method) | Task 1 | Unit | `TestSimulationTrace_RecordDisaggregation` |
| BC-PD-17 (RecordPrefillRouting method) | Task 1 | Unit | `TestSimulationTrace_RecordPrefillRouting` |
| BC-PD-17 (RecordKVTransfer method) | Task 1 | Unit | `TestSimulationTrace_RecordKVTransfer` |
| BC-PD-17 (RecordDecodeRouting method) | Task 1 | Unit | `TestSimulationTrace_RecordDecodeRouting` |
| BC-PD-18 (non-disagg no PD records) | Task 2 | Integration | `TestPDTrace_NonDisaggMode_NoDisaggRecords` |
| BC-PD-17 (disagg decision recorded) | Task 2 | Integration | `TestPDTrace_DisaggMode_DisaggDecisionRecorded` |
| BC-PD-17 (all 4 types in disagg sim) | Task 3 | Integration | `TestPDTrace_DisaggMode_AllRecordTypesPresent` |
| BC-PD-19 (counterfactual in pool routing) | Task 3 | Integration | `TestPDTrace_DisaggMode_Counterfactual` |
| BC-TRACE-COMPAT | Task 2 | Regression | `TestPDTrace_NonDisaggMode_NoDisaggRecords` (checks Admissions/Routings still present) |

**Invariant coverage:**
- INV-6 (determinism): Trace recording adds no RNG calls, preserving determinism. Verified by existing `TestDisaggregation_Determinism`.
- BC-PD-17 is verified by counting records in integration tests (count = num requests = invariant law).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|---|---|---|---|---|
| `filteredSnapshots` used for counterfactual instead of `buildRouterState()` snapshots — wrong pool | Low | High | Code review: verify `filteredSnapshots` local var (not `cs.instances`) is passed to `computeCounterfactual` | Task 3 |
| `KVTransferRecord.TransferDuration` computed incorrectly (TransferCompleteTime = 0 if decode routing fires before complete) | Low | Medium | Unit test verifies TransferDuration > 0; sequence is deterministic in single-threaded DES | Task 3 |
| `pd_events.go` missing `trace` import after instrumentation | Low | Low | `golangci-lint run ./sim/cluster/...` in each task's lint step | Task 3 |
| Existing trace tests broken by `SimulationTrace` struct change | Low | Low | `go test ./sim/trace/...` runs after Task 1; `NewSimulationTrace` is the canonical constructor | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `PrefillRoutingRecord` and `DecodeRoutingRecord` are intentionally similar (parallel pool routing paths)
- [x] No feature creep — no trace summary statistics, no new CLI flags
- [x] No unexercised flags or interfaces — all 4 Record* methods called in tests
- [x] No partial implementations — all 4 record types fully implemented and instrumented
- [x] No breaking changes — `SimulationTrace` gains new fields; callers using `.Routings` and `.Admissions` are unaffected
- [x] No hidden global state — pure appends to slices owned by `SimulationTrace`
- [x] All new code will pass golangci-lint — no unexported structs with exported fields issue (all fields are exported for JSON serialization)
- [x] Shared test helpers used — `newTestDisaggDeploymentConfig`, `newTestRequests`, `mustRun` from `test_helpers_test.go`
- [x] CLAUDE.md updated — Task 4 updates sim/trace/ description
- [x] No stale references — checked (R15); `ClusterEvent.Priority()` comment updated in Task 2 to include PD priorities 4-7
- [x] Documentation DRY — CLAUDE.md `sim/trace/` section updated in same PR
- [x] Deviation log reviewed — deviation (KVTransfer in DecodeRoutingEvent) justified above
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered: Task 1 (types) → Task 2 (disagg event) → Task 3 (pool routing events) → Task 4 (docs)
- [x] All contracts mapped to tasks
- [x] No golden dataset changes (pure observability, no simulation output changes)
- [x] Construction site audit: `SimulationTrace{}` literal only in `NewSimulationTrace()` — updated in Task 1

**Antipattern rules:**
- [x] R1: No silent data loss — `RecordKVTransfer` called with all fields set before emit
- [x] R2: No map iteration — `Scores` map is copied (not iterated for output ordering)
- [x] R3: No new numeric parameters
- [x] R4: Construction site — `SimulationTrace{}` only in `NewSimulationTrace` — updated in Task 1
- [x] R6: No `logrus.Fatalf` — pure data slices, no process termination
- [x] R7: Integration tests act as invariant tests (count records = count requests is a conservation law)
- [x] R8: `Scores` maps in records are copies (`copyScores`) — not shared with routing policy
- [x] R11: No new division operations
- [x] R13: No new interfaces
- [x] R14: Each `Record*` method does one thing (append)
- [x] R15: No stale PR references — verified by grep
- [x] R23: PrefillRoutingRecord and DecodeRoutingRecord have identical fields (parallel paths, R23 parity)

---

## Appendix: File-Level Implementation Details

### File: `sim/trace/record.go`

**Purpose:** Pure data types for trace records. No dependencies on sim/ or cluster packages.

**Complete additions (append after line 33):**

```go
// DisaggregationRecord captures a PD disaggregation decision.
type DisaggregationRecord struct {
	RequestID    string
	Clock        int64
	Disaggregate bool // true = routed to prefill pool; false = standard routing
}

// PrefillRoutingRecord captures a prefill pool routing decision with optional counterfactual analysis.
type PrefillRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	Scores          map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates      []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret          float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// DecodeRoutingRecord captures a decode pool routing decision with optional counterfactual analysis.
type DecodeRoutingRecord struct {
	ParentRequestID string
	Clock           int64
	ChosenInstance  string
	Scores          map[string]float64 // from RoutingDecision.Scores (may be nil)
	Candidates      []CandidateScore   // top-k candidates sorted by score desc (nil if k=0)
	Regret          float64            // max(alternative scores) - score(chosen); 0 if chosen is best
}

// KVTransferRecord captures a KV cache transfer event between prefill and decode instances.
// Recorded in DecodeRoutingEvent.Execute() (not KVTransferCompletedEvent) so that
// DecodeInstanceID is fully populated when the record is written.
type KVTransferRecord struct {
	ParentRequestID   string
	TransferStartTime int64 // microseconds (sim clock when transfer started)
	TransferDuration  int64 // microseconds (TransferCompleteTime - TransferStartTime)
	NumKVBlocks       int64
	PrefillInstanceID string
	DecodeInstanceID  string
}
```

### File: `sim/trace/trace.go`

**Purpose:** SimulationTrace type with record slices and append methods.

**Updated SimulationTrace struct (replace lines 32-36):**

```go
// SimulationTrace collects decision records during a cluster simulation.
type SimulationTrace struct {
	Config          TraceConfig
	Admissions      []AdmissionRecord
	Routings        []RoutingRecord
	Disaggregations []DisaggregationRecord
	PrefillRoutings []PrefillRoutingRecord
	DecodeRoutings  []DecodeRoutingRecord
	KVTransfers     []KVTransferRecord
}
```

**Updated NewSimulationTrace (replace lines 39-45):**

```go
// NewSimulationTrace creates a SimulationTrace ready for recording.
func NewSimulationTrace(config TraceConfig) *SimulationTrace {
	return &SimulationTrace{
		Config:          config,
		Admissions:      make([]AdmissionRecord, 0),
		Routings:        make([]RoutingRecord, 0),
		Disaggregations: make([]DisaggregationRecord, 0),
		PrefillRoutings: make([]PrefillRoutingRecord, 0),
		DecodeRoutings:  make([]DecodeRoutingRecord, 0),
		KVTransfers:     make([]KVTransferRecord, 0),
	}
}
```

**Append to file after line 55 (after RecordRouting method):**

```go
// RecordDisaggregation appends a disaggregation decision record.
func (st *SimulationTrace) RecordDisaggregation(record DisaggregationRecord) {
	st.Disaggregations = append(st.Disaggregations, record)
}

// RecordPrefillRouting appends a prefill pool routing decision record.
func (st *SimulationTrace) RecordPrefillRouting(record PrefillRoutingRecord) {
	st.PrefillRoutings = append(st.PrefillRoutings, record)
}

// RecordDecodeRouting appends a decode pool routing decision record.
func (st *SimulationTrace) RecordDecodeRouting(record DecodeRoutingRecord) {
	st.DecodeRoutings = append(st.DecodeRoutings, record)
}

// RecordKVTransfer appends a KV cache transfer event record.
func (st *SimulationTrace) RecordKVTransfer(record KVTransferRecord) {
	st.KVTransfers = append(st.KVTransfers, record)
}
```

### File: `sim/cluster/cluster_event.go`

**Purpose:** Add DisaggregationRecord emission in DisaggregationDecisionEvent.Execute(). Also update the stale `ClusterEvent.Priority()` comment to include the full priority range.

**Priority comment update:** Update `ClusterEvent` interface comment (line 17 of `cluster_event.go`):

Replace:
```go
Priority() int // 0=Arrival, 1=Admission, 2=Routing, 3=Disaggregation
```
With:
```go
Priority() int // 0=Arrival, 1=Admission, 2=Routing, 3=Disaggregation, 4=PrefillRouting, 5=KVTransferStarted, 6=KVTransferCompleted, 7=DecodeRouting
```

**Insertion point:** In `DisaggregationDecisionEvent.Execute()`, after `logrus.Debugf("[cluster] req %s: disaggregate=%v", ...)` and before `if !decision.Disaggregate {`.

**Code to insert:**

```go
// Record disaggregation decision if tracing is enabled (BC-PD-17)
if cs.trace != nil {
    cs.trace.RecordDisaggregation(trace.DisaggregationRecord{
        RequestID:    e.request.ID,
        Clock:        cs.clock,
        Disaggregate: decision.Disaggregate,
    })
}
```

No new imports needed — `trace` package already imported in `cluster_event.go`.

### File: `sim/cluster/pd_events.go`

**Purpose:** Add PrefillRoutingRecord, DecodeRoutingRecord, and KVTransferRecord emission in PD event handlers.

**Import addition:** Add `"github.com/inference-sim/inference-sim/sim/trace"` to import block.

**PrefillRoutingEvent.Execute() insertion point:** After `e.parentReq.PrefillEnqueueTime = e.time` (line ~39) and before `cs.pendingPrefillCompletions[...]`.

**Code to insert in PrefillRoutingEvent.Execute():**

```go
// Record prefill routing decision if tracing is enabled (BC-PD-17, BC-PD-19)
if cs.trace != nil {
    record := trace.PrefillRoutingRecord{
        ParentRequestID: e.parentReq.ID,
        Clock:           cs.clock,
        ChosenInstance:  decision.TargetInstance,
        Scores:          copyScores(decision.Scores),
    }
    if cs.trace.Config.CounterfactualK > 0 {
        record.Candidates, record.Regret = computeCounterfactual(
            decision.TargetInstance, decision.Scores,
            filteredSnapshots, cs.trace.Config.CounterfactualK,
        )
    }
    cs.trace.RecordPrefillRouting(record)
}
```

**DecodeRoutingEvent.Execute() change:** Replace the entire `for _, inst := range cs.instances` loop with a restructured version that records traces **inside the successful KV allocation path** (after `AllocateTransferredKV` returns `true`). Recording before the KV allocation check would create orphan trace records for requests that failed KV allocation (violates R1).

**Replace the `for _, inst := range cs.instances` loop (lines ~169-186) with:**

```go
// Find target decode instance
for _, inst := range cs.instances {
    if string(inst.ID()) == decision.TargetInstance {
        // Pre-allocate KV blocks for transferred input
        if ok := inst.AllocateTransferredKV(e.decodeSubReq); !ok {
            logrus.Warnf("[cluster] decode instance %s: insufficient KV capacity for %s (%d input tokens)",
                decision.TargetInstance, e.decodeSubReq.ID, len(e.decodeSubReq.InputTokens))
            // Cannot proceed without KV — request effectively dropped
            return
        }

        // Record KV transfer and decode routing after successful KV allocation (BC-PD-17, BC-PD-19)
        // Placement after AllocateTransferredKV ensures records only exist for requests that
        // complete the decode phase (R1: no orphan records for dropped requests).
        // KVTransferRecord is recorded here so DecodeInstanceID is fully populated.
        // Both TransferStartTime and TransferCompleteTime were set in earlier event handlers.
        if cs.trace != nil {
            cs.trace.RecordKVTransfer(trace.KVTransferRecord{
                ParentRequestID:   e.parentReq.ID,
                TransferStartTime: e.parentReq.TransferStartTime,
                TransferDuration:  e.parentReq.TransferCompleteTime - e.parentReq.TransferStartTime,
                NumKVBlocks:       e.parentReq.NumKVBlocks,
                PrefillInstanceID: e.parentReq.PrefillInstanceID,
                DecodeInstanceID:  e.parentReq.DecodeInstanceID,
            })
            decodeRecord := trace.DecodeRoutingRecord{
                ParentRequestID: e.parentReq.ID,
                Clock:           cs.clock,
                ChosenInstance:  decision.TargetInstance,
                Scores:          copyScores(decision.Scores),
            }
            if cs.trace.Config.CounterfactualK > 0 {
                decodeRecord.Candidates, decodeRecord.Regret = computeCounterfactual(
                    decision.TargetInstance, decision.Scores,
                    filteredSnapshots, cs.trace.Config.CounterfactualK,
                )
            }
            cs.trace.RecordDecodeRouting(decodeRecord)
        }

        cs.inFlightRequests[decision.TargetInstance]++
        inst.InjectDecodeOnline(e.decodeSubReq)
        return
    }
}
panic(fmt.Sprintf("DecodeRoutingEvent: invalid TargetInstance %q", decision.TargetInstance))
```

### File: `sim/cluster/pd_traces_test.go`

**Purpose:** Integration tests for PD trace records.

**Key test setup:** Uses `newTestDisaggDeploymentConfig(4, 2, 2)` from `test_helpers_test.go` (already available in package). Sets `config.TraceLevel = "decisions"`.

**No new imports needed** beyond what's already in the package.
