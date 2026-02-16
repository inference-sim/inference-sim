# Routing Policies Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable intelligent request routing across instances via pluggable RoutingPolicy interface with four production-ready templates (RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity).

**Architecture:** Replaces hardcoded round-robin in `RoutingDecisionEvent.Execute` with pluggable policy selection. Routing types (`RoutingPolicy`, `RoutingSnapshot`, `RoutingDecision`) live in `sim/` package (mirroring `AdmissionPolicy` pattern) to avoid import cycles. `sim/cluster` converts its `InstanceSnapshot` to `sim.RoutingSnapshot` before calling Route(). Default policy (round-robin) preserves existing behavior for backward compatibility.

**Macro Plan Reference:** Phase 2, PR 6 (docs/plans/2026-02-11-macro-implementation-plan-v2.md:1155-1178)

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation (Human Review)

### A) Executive Summary

**Building Block:** RoutingPolicy interface and four template implementations (RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity) in the `sim/` package.

**Adjacent Blocks:**
- ClusterSimulator.Run() (orchestrates control plane)
- RoutingDecisionEvent.Execute() (invokes policy, converts InstanceSnapshot → RoutingSnapshot)
- InstanceSnapshot (provides observable state, converted to RoutingSnapshot for policy)
- DeploymentConfig (carries routing config from CLI)

**Deviations:** See Section D. Key deviation: types in `sim/` not `sim/policy/` to avoid import cycle (matches AdmissionPolicy pattern).

### B) Behavioral Contracts

#### Positive Contracts (Normal Operation)

**BC-1: RoutingPolicy Interface Contract**
- GIVEN a RoutingPolicy implementation
- WHEN Route() is called with a Request, RoutingSnapshots, and clock
- THEN it MUST return a RoutingDecision with TargetInstance set to a valid instance ID from the snapshot list
- MECHANISM: All templates implement Route() and select target from snapshots

**BC-2: RoundRobin Deterministic Ordering**
- GIVEN RoundRobin policy
- WHEN N requests are routed to M instances
- THEN request i is routed to snapshots[i mod M], ensuring even distribution
- MECHANISM: Counter increments on each Route() call, mod M for index

**BC-3: LeastLoaded Load-Based Selection**
- GIVEN LeastLoaded policy and instance snapshots
- WHEN Route() is called
- THEN it MUST select the instance with minimum (QueueDepth + BatchSize); ties broken by first occurrence in snapshot order (lowest index)
- MECHANISM: Iterate snapshots, track minimum load with strict `<` comparison (first-wins on tie)

**BC-4: WeightedScoring Multi-Factor Scoring**
- GIVEN WeightedScoring with cacheWeight and loadWeight parameters
- WHEN Route() is called
- THEN it MUST compute composite score = (1-KVUtilization)*cacheWeight + (1-normalizedLoad)*loadWeight and select highest-scoring instance; ties broken by first occurrence in snapshot order
- MECHANISM: Normalize load as load/maxLoad (0 when maxLoad==0), compute weighted sum, argmax with strict `>` (first-wins on tie)

**BC-5: PrefixAffinity Cache-Aware Routing**
- GIVEN PrefixAffinity policy and request with InputTokens
- WHEN Route() is called
- THEN it MUST compute prefix hash (matching KVCache format: pipe-delimited decimal strings), attempt to route to instance that previously processed matching prefix
- MECHANISM: Maintain map[prefixHash]string, lookup on Route(), fallback to LeastLoaded on miss

**BC-6: Backward Compatibility with Default Policy**
- GIVEN cluster with no --routing-policy flag (defaults to "round-robin")
- WHEN simulation runs
- THEN behavior MUST match PR 5 round-robin implementation exactly (identical metrics)
- MECHANISM: Default CLI flag value "round-robin"; RoundRobin produces same sequence as old counter-based logic. Verified by existing `TestClusterSimulator_SingleInstance_GoldenEquivalence` continuing to pass.

#### Negative Contracts (What MUST NOT Happen)

**BC-7: No Invalid Instance Selection**
- GIVEN any routing policy
- WHEN Route() returns a RoutingDecision
- THEN TargetInstance MUST NOT be an ID not present in the snapshots parameter
- MECHANISM: All policies select from snapshots; RoutingDecisionEvent.Execute panics if target not found

**BC-8: No State Leakage Across Calls**
- GIVEN PrefixAffinity policy maintaining prefix-to-instance map
- WHEN multiple simulations run sequentially
- THEN state from previous simulation MUST NOT affect routing in subsequent simulation
- MECHANISM: Policy instantiated per ClusterSimulator; no global state

**BC-9: No Observability Violations**
- GIVEN any routing policy
- WHEN Route() is called
- THEN it MUST NOT access InstanceSimulator internals directly; MUST use only RoutingSnapshot fields
- MECHANISM: Route() receives RoutingSnapshot (value type in sim/ package), not *InstanceSimulator

#### Error Handling Contracts

**BC-10: Empty Snapshots Handling**
- GIVEN any routing policy
- WHEN Route() is called with empty snapshots slice
- THEN it MUST panic with descriptive message
- MECHANISM: Explicit len(snapshots)==0 check at top of each Route() implementation

**BC-11: Unrecognized Policy Name**
- GIVEN NewRoutingPolicy() called with invalid policy name
- WHEN factory function executes
- THEN it MUST panic with message "unknown routing policy %q"
- MECHANISM: Switch statement with default case panic (mirrors sim.NewAdmissionPolicy pattern)

**BC-12: Invalid Weight Parameters**
- GIVEN WeightedScoring with cacheWeight < 0 or loadWeight < 0
- WHEN Route() is called
- THEN behavior is undefined (negative weights are nonsensical but non-fatal)
- MECHANISM: No validation (user error, degrades gracefully to poor routing)

### C) Component Interaction

#### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       ClusterSimulator                          │
│  - holds: routingPolicy sim.RoutingPolicy (new)                 │
│  - holds: snapshotProvider SnapshotProvider (existing)          │
│  - holds: clusterEvents ClusterEventQueue (existing)            │
└──────────────────┬────────────────────────────────────┬─────────┘
                   │                                    │
                   │ creates/owns                       │ calls Snapshot()
                   ▼                                    ▼
       ┌─────────────────────┐          ┌──────────────────────────┐
       │ RoutingDecisionEvent│          │   SnapshotProvider        │
       │  - request *Request │          │   (CachedSnapshotProvider)│
       │  - time int64       │          └──────────────────────────┘
       └──────────┬──────────┘
                  │ Execute(cs)
                  │ converts InstanceSnapshot → RoutingSnapshot
                  │ calls Route()
                  ▼
       ┌────────────────────────────────────┐
       │  sim.RoutingPolicy (interface)      │ ◄─── NEW (in sim/ package)
       │  Route(req, []RoutingSnapshot, clk) │
       └────────────────────────────────────┘
                  △
                  │ implements
      ┌───────────┴──────────┬────────────┬───────────┐
      │                      │            │           │
┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ RoundRobin  │  │ LeastLoaded  │  │ Weighted     │  │ PrefixAffinity│
│ (sim/)      │  │ (sim/)       │  │ Scoring      │  │ (sim/)        │
└─────────────┘  └──────────────┘  │ (sim/)       │  └──────────────┘
                                   └──────────────┘

Data flow:
  Request → ClusterArrivalEvent → AdmissionDecisionEvent
    → RoutingDecisionEvent.Execute()
    → convert []InstanceSnapshot → []sim.RoutingSnapshot
    → cs.routingPolicy.Route(req, routingSnapshots, clock)
    → sim.RoutingDecision{TargetInstance: "instance_0"}
    → find instance by ID → target.InjectRequestOnline(req, time)
```

#### API Contracts

**RoutingSnapshot struct** (`sim/routing.go`):
```go
type RoutingSnapshot struct {
    ID            string
    QueueDepth    int
    BatchSize     int
    KVUtilization float64
    FreeKVBlocks  int64
}
```
- Semantics: Lightweight view of instance state for routing decisions.
- Populated by `sim/cluster` from `cluster.InstanceSnapshot` in `RoutingDecisionEvent.Execute()`.

**RoutingPolicy interface** (`sim/routing.go`):
```go
type RoutingPolicy interface {
    Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision
}
```
- Preconditions: len(snapshots) > 0, req != nil
- Postconditions: RoutingDecision.TargetInstance ∈ {snap.ID | snap ∈ snapshots}
- Failure mode: panic on empty snapshots

**RoutingDecision struct** (`sim/routing.go`):
```go
type RoutingDecision struct {
    TargetInstance string
    Reason         string
    Scores         map[string]float64
}
```
- Semantics: TargetInstance is the authoritative routing target (must match a snapshot ID)

**NewRoutingPolicy factory** (`sim/routing.go`):
```go
func NewRoutingPolicy(name string, cacheWeight, loadWeight float64) RoutingPolicy
```
- Valid names: "", "round-robin", "least-loaded", "weighted", "prefix-affinity"
- Empty string defaults to "round-robin"
- Panics on unrecognized name

#### State Changes

**New mutable state in ClusterSimulator** (`sim/cluster/cluster.go`):
- `routingPolicy sim.RoutingPolicy` field (replaces `roundRobinCounter int`)
- Owner: ClusterSimulator
- Lifecycle: Created in NewClusterSimulator(), used throughout Run(), no explicit cleanup
- Accessed by: RoutingDecisionEvent.Execute()

**Removed field from ClusterSimulator:**
- `roundRobinCounter int` — replaced by RoundRobin policy's internal counter

**New mutable state in PrefixAffinity** (`sim/routing.go`):
- `prefixMap map[string]string` (prefix hash → last routed instance ID)
- Owner: PrefixAffinity instance
- Lifecycle: Created with policy, updated on each Route() call
- Accessed by: Route() method only (private to policy)

**New config fields in DeploymentConfig** (`sim/cluster/deployment.go`):
- `RoutingPolicy string` (CLI flag value)
- `RoutingCacheWeight float64` (for WeightedScoring)
- `RoutingLoadWeight float64` (for WeightedScoring)

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| RoutingPolicy in `sim/policy/routing.go` | RoutingPolicy in `sim/routing.go` | CORRECTION: `sim/policy/` → `sim/cluster/` → `sim/policy/` creates import cycle. Placing in `sim/` mirrors the `sim/admission.go` pattern (AdmissionPolicy was moved from `sim/policy/` to `sim/` in PR 5 for this exact reason). |
| Route() takes `[]cluster.InstanceSnapshot` | Route() takes `[]sim.RoutingSnapshot` | CORRECTION: RoutingSnapshot defined in `sim/` to avoid importing `sim/cluster/`. ClusterSimulator converts InstanceSnapshot → RoutingSnapshot at the call site. |
| RoutingDecision uses `cluster.InstanceID` | RoutingDecision uses `string` | SIMPLIFICATION: Avoids importing `sim/cluster/`. InstanceID is `type InstanceID string`, so string is equivalent and avoids coupling. |
| Route() takes PriorityDecision parameter | PR 6 Route() has no priority parameter | SIMPLIFICATION: PriorityPolicy is PR 7; interface will be extended in PR 8 when RouterState is added. All implementations are in-tree, so the breaking change in PR 8 is a single atomic commit. |
| Route() takes RouterState parameter | PR 6 Route() has no state parameter | SIMPLIFICATION: RouterState is PR 8; interface will be extended then. |

### E) Review Guide

**THE TRICKY PART:** The `InstanceSnapshot` → `RoutingSnapshot` conversion in `RoutingDecisionEvent.Execute`. Ensure all fields are mapped correctly and the string↔InstanceID boundary is handled properly.

**WHAT TO SCRUTINIZE:**
- BC-6 (backward compatibility): RoundRobin default produces identical metrics to PR 5. Primary guard: existing `TestClusterSimulator_SingleInstance_GoldenEquivalence` must still pass.
- BC-5 (PrefixAffinity): Prefix hash uses KVCache-compatible format (pipe-delimited decimal strings, NOT 4-byte binary encoding)
- BC-4 (WeightedScoring): Load normalization logic (avoid divide-by-zero on uniform load)

**WHAT'S SAFE TO SKIM:**
- RoundRobin and LeastLoaded are trivial (counter mod N, argmin over load)
- RoutingDecision struct definition (simple data carrier)
- CLI flag wiring (mechanical, mirrors admission flags)

**KNOWN DEBT:**
- RoutingPolicy interface will be extended in PR 8 with RouterState parameter. This is a backward-incompatible interface change, acceptable because all implementations are in-tree (single atomic commit in PR 8).
- RoutingSnapshot is a thin copy of InstanceSnapshot fields. If InstanceSnapshot gains new fields relevant to routing, RoutingSnapshot must be updated too.

---

## PART 2: Executable Implementation (Agent Execution)

### F) Implementation Overview

**Files to create:**
- `sim/routing.go` (~220 LOC): RoutingSnapshot, RoutingPolicy interface, RoutingDecision struct, four templates, factory
- `sim/routing_test.go` (~200 LOC): Unit tests for each template

**Files to modify:**
- `sim/cluster/cluster.go` (~20 LOC): Replace `roundRobinCounter` with `routingPolicy` field, initialize in constructor
- `sim/cluster/cluster_event.go` (~15 LOC): Replace hardcoded round-robin with snapshot conversion + policy.Route() call
- `sim/cluster/deployment.go` (~10 LOC): Add RoutingPolicy, RoutingCacheWeight, RoutingLoadWeight fields
- `cmd/root.go` (~25 LOC): Add CLI flags, pass config to NewClusterSimulator
- `CLAUDE.md` (~5 LOC): Update "Current Implementation Focus" to note PR 6 completion

**Key decisions:**
- RoutingPolicy lives in `sim/routing.go` (not `sim/policy/`) to avoid import cycle, mirroring AdmissionPolicy in `sim/admission.go`
- RoutingSnapshot is a lightweight value type in `sim/` — `sim/cluster` converts InstanceSnapshot → RoutingSnapshot at the routing call site
- PrefixAffinity uses pipe-delimited decimal string hash (matching KVCache's `kvcache.go:108-116` format exactly)
- WeightedScoring normalizes load as (QueueDepth + BatchSize) / maxLoad to [0,1] range
- Default CLI flag values: routingPolicy="round-robin", cacheWeight=0.6, loadWeight=0.4
- RoutingDecision.Scores populated only by WeightedScoring (optimization: other policies set to nil)

**Confirmation:**
- No dead code: All four templates exercisable via --routing-policy flag; `roundRobinCounter` field removed
- All paths exercisable: Each template has dedicated test; CLI integration test verifies flag wiring
- No scaffolding: Policy invoked in RoutingDecisionEvent.Execute immediately

### G) Task Breakdown

Tasks are grouped into 3 batches for checkpoint reviews:

**Batch 1 (Tasks 1-3): Core Infrastructure**
- Task 1: RoutingSnapshot, RoutingPolicy interface, RoutingDecision struct, factory, RoundRobin template
- Task 2: LeastLoaded template
- Task 3: ClusterSimulator integration (replace hardcoded routing, remove roundRobinCounter)

**Batch 2 (Tasks 4-5): Advanced Policies**
- Task 4: WeightedScoring template
- Task 5: PrefixAffinity template

**Batch 3 (Tasks 6-7): CLI and Integration**
- Task 6: CLI flags and DeploymentConfig
- Task 7: Integration tests and documentation

---

#### Task 1: RoutingSnapshot, RoutingPolicy Interface, and RoundRobin Template

**Contracts Implemented:** BC-1, BC-2, BC-6, BC-10, BC-11

**Files:**
- Create: `sim/routing.go`
- Create: `sim/routing_test.go`

**Step 1: Write failing test for RoutingPolicy interface and RoundRobin**

Context: Establish RoutingPolicy interface contract and implement simplest template (RoundRobin) for baseline.

In `sim/routing_test.go`:
```go
package sim

import (
	"fmt"
	"testing"
)

// TestRoutingPolicy_Interface_Contract verifies the RoutingPolicy interface contract.
func TestRoutingPolicy_Interface_Contract(t *testing.T) {
	// GIVEN a RoutingPolicy implementation (RoundRobin)
	policy := NewRoutingPolicy("round-robin", 0, 0)

	// WHEN Route() is called with valid inputs
	req := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5},
		{ID: "instance_1", QueueDepth: 3},
	}
	decision := policy.Route(req, snapshots, 1000)

	// THEN RoutingDecision must have TargetInstance set
	if decision.TargetInstance == "" {
		t.Errorf("Expected non-empty TargetInstance, got empty")
	}

	// THEN TargetInstance must be in snapshots
	found := false
	for _, snap := range snapshots {
		if snap.ID == decision.TargetInstance {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("TargetInstance %q not found in snapshots", decision.TargetInstance)
	}
}

// TestRoundRobin_DeterministicOrdering verifies BC-2.
func TestRoundRobin_DeterministicOrdering(t *testing.T) {
	// GIVEN RoundRobin policy
	policy := NewRoutingPolicy("round-robin", 0, 0)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0"},
		{ID: "instance_1"},
		{ID: "instance_2"},
	}

	// WHEN 6 requests are routed
	var targets []string
	for i := 0; i < 6; i++ {
		req := &Request{ID: fmt.Sprintf("req%d", i)}
		decision := policy.Route(req, snapshots, int64(i*1000))
		targets = append(targets, decision.TargetInstance)
	}

	// THEN requests distributed round-robin: 0, 1, 2, 0, 1, 2
	expected := []string{"instance_0", "instance_1", "instance_2", "instance_0", "instance_1", "instance_2"}
	for i, exp := range expected {
		if targets[i] != exp {
			t.Errorf("Request %d: expected %q, got %q", i, exp, targets[i])
		}
	}
}

// TestRoundRobin_EmptySnapshots_Panics verifies BC-10.
func TestRoundRobin_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("round-robin", 0, 0)
	req := &Request{ID: "req1"}
	policy.Route(req, []RoutingSnapshot{}, 1000) // should panic
}

// TestNewRoutingPolicy_UnknownName_Panics verifies BC-11.
func TestNewRoutingPolicy_UnknownName_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on unknown policy name, got none")
		}
	}()

	NewRoutingPolicy("invalid-policy", 0, 0) // should panic
}

// TestNewRoutingPolicy_DefaultName verifies empty string defaults to round-robin.
func TestNewRoutingPolicy_DefaultName(t *testing.T) {
	policy := NewRoutingPolicy("", 0, 0)
	if policy == nil {
		t.Errorf("Expected non-nil policy for empty string, got nil")
	}
	// Type assertion to verify it's RoundRobin
	if _, ok := policy.(*RoundRobin); !ok {
		t.Errorf("Expected RoundRobin for empty string, got %T", policy)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run "TestRoutingPolicy|TestRoundRobin|TestNewRoutingPolicy" -v`
Expected: FAIL (types/functions not defined)

**Step 3: Implement RoutingSnapshot, RoutingPolicy interface, and RoundRobin template**

Context: Define types in `sim/` package (no cluster imports needed). Implement RoundRobin with counter.

In `sim/routing.go`:
```go
package sim

import "fmt"

// RoutingSnapshot is a lightweight view of instance state for routing decisions.
// Populated by ClusterSimulator from cluster.InstanceSnapshot at routing time.
type RoutingSnapshot struct {
	ID            string
	QueueDepth    int
	BatchSize     int
	KVUtilization float64
	FreeKVBlocks  int64
}

// RoutingDecision encapsulates the routing decision for a request.
type RoutingDecision struct {
	TargetInstance string             // Instance ID to route to (must match a snapshot ID)
	Reason         string             // Human-readable explanation
	Scores         map[string]float64 // Per-instance scores (nil for policies without scoring)
}

// RoutingPolicy decides which instance should handle a request.
// Implementations receive request, instance snapshots, and current clock.
// This is a transitional interface for PR 6; PR 8 will extend with RouterState parameter.
type RoutingPolicy interface {
	Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision
}

// RoundRobin routes requests in round-robin order across instances.
type RoundRobin struct {
	counter int
}

// Route implements RoutingPolicy for RoundRobin.
func (rr *RoundRobin) Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision {
	if len(snapshots) == 0 {
		panic("RoundRobin.Route: empty snapshots")
	}
	target := snapshots[rr.counter%len(snapshots)]
	rr.counter++
	return RoutingDecision{
		TargetInstance: target.ID,
		Reason:         fmt.Sprintf("round-robin[%d]", rr.counter-1),
	}
}

// NewRoutingPolicy creates a routing policy by name.
// Valid names: "", "round-robin", "least-loaded", "weighted", "prefix-affinity".
// Empty string defaults to round-robin.
// For weighted scoring, cacheWeight and loadWeight configure the composite score.
// Panics on unrecognized names.
func NewRoutingPolicy(name string, cacheWeight, loadWeight float64) RoutingPolicy {
	switch name {
	case "", "round-robin":
		return &RoundRobin{}
	default:
		panic(fmt.Sprintf("unknown routing policy %q", name))
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run "TestRoutingPolicy|TestRoundRobin|TestNewRoutingPolicy" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/routing.go sim/routing_test.go
git commit -m "feat(sim): add RoutingPolicy interface and RoundRobin template (BC-1, BC-2, BC-6, BC-10, BC-11)

- Add RoutingSnapshot, RoutingPolicy, RoutingDecision types in sim/ package
- Implement RoundRobin template with deterministic counter-based selection
- Add NewRoutingPolicy factory with panic on unknown name
- Place in sim/ (not sim/policy/) to avoid import cycle, matching AdmissionPolicy pattern

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: LeastLoaded Template

**Contracts Implemented:** BC-3, BC-10

**Files:**
- Modify: `sim/routing.go`
- Modify: `sim/routing_test.go`

**Step 1: Write failing test for LeastLoaded**

Context: LeastLoaded selects instance with minimum (QueueDepth + BatchSize). Ties broken by first occurrence in snapshot order.

In `sim/routing_test.go`:
```go
// TestLeastLoaded_LoadBasedSelection verifies BC-3.
func TestLeastLoaded_LoadBasedSelection(t *testing.T) {
	// GIVEN LeastLoaded policy
	policy := NewRoutingPolicy("least-loaded", 0, 0)

	tests := []struct {
		name      string
		snapshots []RoutingSnapshot
		expected  string
	}{
		{
			name: "instance 1 has lowest load",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 10, BatchSize: 5},  // load=15
				{ID: "instance_1", QueueDepth: 3, BatchSize: 2},   // load=5 (min)
				{ID: "instance_2", QueueDepth: 7, BatchSize: 8},   // load=15
			},
			expected: "instance_1",
		},
		{
			name: "tie broken by first occurrence (lowest index)",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5},   // load=10 (tie, first)
				{ID: "instance_1", QueueDepth: 8, BatchSize: 2},   // load=10 (tie, second)
				{ID: "instance_2", QueueDepth: 3, BatchSize: 12},  // load=15
			},
			expected: "instance_0", // first occurrence wins tie
		},
		{
			name: "all instances equal load",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
			},
			expected: "instance_0", // first occurrence wins
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "req1"}
			decision := policy.Route(req, tt.snapshots, 1000)
			if decision.TargetInstance != tt.expected {
				t.Errorf("Expected %q, got %q", tt.expected, decision.TargetInstance)
			}
		})
	}
}

// TestLeastLoaded_EmptySnapshots_Panics verifies BC-10.
func TestLeastLoaded_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("Expected panic on empty snapshots, got none")
		}
	}()

	policy := NewRoutingPolicy("least-loaded", 0, 0)
	req := &Request{ID: "req1"}
	policy.Route(req, []RoutingSnapshot{}, 1000) // should panic
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestLeastLoaded -v`
Expected: FAIL with "unknown routing policy \"least-loaded\""

**Step 3: Implement LeastLoaded template**

Context: Iterate snapshots, compute load as QueueDepth + BatchSize, select argmin. Ties broken by first occurrence (strict `<` comparison).

In `sim/routing.go`:
```go
// LeastLoaded routes requests to the instance with minimum (QueueDepth + BatchSize).
// Ties are broken by first occurrence in snapshot order (lowest index).
type LeastLoaded struct{}

// Route implements RoutingPolicy for LeastLoaded.
func (ll *LeastLoaded) Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision {
	if len(snapshots) == 0 {
		panic("LeastLoaded.Route: empty snapshots")
	}

	minLoad := snapshots[0].QueueDepth + snapshots[0].BatchSize
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].QueueDepth + snapshots[i].BatchSize
		if load < minLoad {
			minLoad = load
			target = snapshots[i]
		}
	}

	return RoutingDecision{
		TargetInstance: target.ID,
		Reason:         fmt.Sprintf("least-loaded (load=%d)", minLoad),
	}
}
```

Also update NewRoutingPolicy factory to include `"least-loaded"` case returning `&LeastLoaded{}`.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestLeastLoaded -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/routing.go sim/routing_test.go
git commit -m "feat(sim): add LeastLoaded routing template (BC-3, BC-10)

- Implement LeastLoaded policy selecting instance with minimum (QueueDepth + BatchSize)
- Tie-breaking by first occurrence in snapshot order (lowest index)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: ClusterSimulator Integration

**Contracts Implemented:** BC-1, BC-6, BC-7, BC-9

**Files:**
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/cluster_event.go`
- Modify: `sim/cluster/deployment.go`
- Test: `sim/cluster/cluster_test.go`

**Step 1: Write failing test for policy integration**

Context: Verify ClusterSimulator invokes routing policy correctly and round-robin default preserves PR 5 behavior.

In `sim/cluster/cluster_test.go`:
```go
// TestClusterSimulator_RoutingPolicy_RoundRobinDefault verifies BC-6 (backward compatibility).
func TestClusterSimulator_RoutingPolicy_RoundRobinDefault(t *testing.T) {
	// GIVEN cluster with default routing policy (round-robin)
	config := newTestDeploymentConfig(3)
	config.RoutingPolicy = "round-robin" // explicit for clarity
	workload := newTestWorkload(10)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	// THEN requests are distributed evenly across instances
	counts := make(map[InstanceID]int)
	for _, inst := range cs.Instances() {
		counts[inst.ID()] = inst.Metrics().CompletedRequests
	}

	// With 10 requests and 3 instances, expect 4, 3, 3 or 3, 4, 3 or 3, 3, 4
	total := 0
	for _, count := range counts {
		total += count
		if count < 3 || count > 4 {
			t.Errorf("Expected 3-4 requests per instance, got %d", count)
		}
	}
	if total != 10 {
		t.Errorf("Expected 10 total completed requests, got %d", total)
	}
}

// TestClusterSimulator_RoutingPolicy_LeastLoaded verifies load-aware routing.
func TestClusterSimulator_RoutingPolicy_LeastLoaded(t *testing.T) {
	// GIVEN cluster with least-loaded routing policy
	config := newTestDeploymentConfig(2)
	config.RoutingPolicy = "least-loaded"
	workload := newTestWorkload(5)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	// THEN least-loaded policy was invoked (no panic, simulation completes)
	if cs.AggregatedMetrics().CompletedRequests == 0 {
		t.Errorf("Expected non-zero completed requests, got 0")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_RoutingPolicy -v`
Expected: FAIL (RoutingPolicy field not yet in ClusterSimulator)

**Step 3: Implement ClusterSimulator routing policy integration**

Context: Replace `roundRobinCounter` with `routingPolicy`, initialize in constructor, invoke in RoutingDecisionEvent.Execute with snapshot conversion.

In `sim/cluster/deployment.go`, add after line 30 (after TokenBucketRefillRate):
```go
	// Routing policy configuration (PR 6)
	RoutingPolicy      string  // "round-robin" (default), "least-loaded", "weighted", "prefix-affinity"
	RoutingCacheWeight float64 // for weighted scoring, default 0.6
	RoutingLoadWeight  float64 // for weighted scoring, default 0.4
```

In `sim/cluster/cluster.go`:
1. Remove `roundRobinCounter int` field from ClusterSimulator struct (line 32)
2. Add `routingPolicy sim.RoutingPolicy` field
3. In NewClusterSimulator, replace line 69 area with:
```go
	routingPolicy:    sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingCacheWeight, config.RoutingLoadWeight),
```

In `sim/cluster/cluster_event.go`, replace RoutingDecisionEvent.Execute (lines 113-117):
```go
// Execute routes the request using the configured routing policy and injects it.
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	// Convert InstanceSnapshots to RoutingSnapshots for the policy
	routingSnapshots := make([]sim.RoutingSnapshot, len(cs.instances))
	for i, inst := range cs.instances {
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		routingSnapshots[i] = sim.RoutingSnapshot{
			ID:            string(snap.ID),
			QueueDepth:    snap.QueueDepth,
			BatchSize:     snap.BatchSize,
			KVUtilization: snap.KVUtilization,
			FreeKVBlocks:  snap.FreeKVBlocks,
		}
	}

	// Invoke routing policy
	decision := cs.routingPolicy.Route(e.request, routingSnapshots, cs.clock)

	// Find target instance and inject request
	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			inst.InjectRequestOnline(e.request, e.time)
			return
		}
	}

	// Should never reach here (policy contract ensures valid target)
	panic(fmt.Sprintf("RoutingDecisionEvent: invalid TargetInstance %q", decision.TargetInstance))
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestClusterSimulator_RoutingPolicy -v`
Expected: PASS

**Step 5: Run full test suite to verify backward compatibility (BC-6)**

Run: `go test ./...`
Expected: ALL PASS (including existing `TestClusterSimulator_SingleInstance_GoldenEquivalence`)

**Step 6: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 7: Commit with contract reference**

```bash
git add sim/cluster/cluster.go sim/cluster/cluster_event.go sim/cluster/deployment.go sim/cluster/cluster_test.go
git commit -m "feat(cluster): integrate RoutingPolicy into ClusterSimulator (BC-1, BC-6, BC-7, BC-9)

- Replace roundRobinCounter field with routingPolicy sim.RoutingPolicy
- Convert InstanceSnapshot → RoutingSnapshot in RoutingDecisionEvent.Execute
- Add RoutingPolicy, RoutingCacheWeight, RoutingLoadWeight to DeploymentConfig
- Backward compat verified: existing golden dataset tests pass with round-robin default

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**CHECKPOINT REVIEW:** Batch 1 complete. Core infrastructure in place: RoutingPolicy interface in `sim/`, RoundRobin and LeastLoaded templates, ClusterSimulator integration with snapshot conversion. Verify BC-1 through BC-11, confirm all tests pass (including golden dataset), lint clean.

---

#### Task 4: WeightedScoring Template

**Contracts Implemented:** BC-4, BC-10, BC-12

**Files:**
- Modify: `sim/routing.go`
- Modify: `sim/routing_test.go`

**Step 1: Write failing test for WeightedScoring**

Context: WeightedScoring computes composite score = (1-KVUtil)*cacheWeight + (1-normalizedLoad)*loadWeight, selects argmax.

In `sim/routing_test.go`:
```go
// TestWeightedScoring_MultiFactor verifies BC-4.
func TestWeightedScoring_MultiFactor(t *testing.T) {
	// GIVEN WeightedScoring with cacheWeight=0.6, loadWeight=0.4
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	tests := []struct {
		name      string
		snapshots []RoutingSnapshot
		expected  string
		reason    string
	}{
		{
			name: "instance 1 wins on low KV utilization",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.8}, // load=10, norm=1.0, score = (1-0.8)*0.6 + (1-1.0)*0.4 = 0.12
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.2}, // load=10, norm=1.0, score = (1-0.2)*0.6 + (1-1.0)*0.4 = 0.48
			},
			expected: "instance_1",
			reason:   "equal load (both normalized to 1.0); instance_1 wins on lower KVUtilization",
		},
		{
			name: "instance 0 wins on low load",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 2, BatchSize: 2, KVUtilization: 0.5}, // load=4, norm=4/10=0.4, score = 0.5*0.6 + 0.6*0.4 = 0.3 + 0.24 = 0.54
				{ID: "instance_1", QueueDepth: 8, BatchSize: 2, KVUtilization: 0.5}, // load=10, norm=1.0, score = 0.5*0.6 + 0.0*0.4 = 0.3 + 0 = 0.3
			},
			expected: "instance_0",
			reason:   "equal KVUtilization; instance_0 wins on lower normalized load",
		},
		{
			name: "all equal scores, first occurrence wins",
			snapshots: []RoutingSnapshot{
				{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.5},
				{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.5},
			},
			expected: "instance_0",
			reason:   "tie broken by first occurrence in snapshot order",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &Request{ID: "req1"}
			decision := policy.Route(req, tt.snapshots, 1000)
			if decision.TargetInstance != tt.expected {
				t.Errorf("%s: expected %q, got %q", tt.reason, tt.expected, decision.TargetInstance)
			}
			// Verify Scores map is populated
			if decision.Scores == nil || len(decision.Scores) != len(tt.snapshots) {
				t.Errorf("Expected Scores map with %d entries, got %v", len(tt.snapshots), decision.Scores)
			}
		})
	}
}

// TestWeightedScoring_UniformLoad verifies divide-by-zero safety.
func TestWeightedScoring_UniformLoad(t *testing.T) {
	// GIVEN WeightedScoring policy
	policy := NewRoutingPolicy("weighted", 0.6, 0.4)

	// WHEN all instances have identical load (maxLoad == minLoad)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.3}, // load=10, norm=1.0, score = 0.7*0.6 + 0*0.4 = 0.42
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.7}, // load=10, norm=1.0, score = 0.3*0.6 + 0*0.4 = 0.18
	}

	req := &Request{ID: "req1"}
	decision := policy.Route(req, snapshots, 1000)

	// THEN decision is made based on cache score alone (load component cancels out)
	// instance_0 wins: (1-0.3)*0.6 + (1-1.0)*0.4 = 0.42 vs (1-0.7)*0.6 + (1-1.0)*0.4 = 0.18
	if decision.TargetInstance != "instance_0" {
		t.Errorf("Expected instance_0 to win on cache score alone, got %q", decision.TargetInstance)
	}
}

// TestWeightedScoring_NegativeWeights verifies BC-12 (undefined but non-fatal).
func TestWeightedScoring_NegativeWeights(t *testing.T) {
	// GIVEN WeightedScoring with negative weights (user error)
	policy := NewRoutingPolicy("weighted", -0.5, -0.5)

	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5, KVUtilization: 0.5},
	}

	req := &Request{ID: "req1"}
	// WHEN Route is called
	// THEN it does not panic (undefined but degrades gracefully)
	decision := policy.Route(req, snapshots, 1000)
	if decision.TargetInstance == "" {
		t.Errorf("Expected non-empty TargetInstance even with negative weights")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestWeightedScoring -v`
Expected: FAIL with "unknown routing policy \"weighted\""

**Step 3: Implement WeightedScoring template**

Context: Normalize load to [0,1] relative to max load (handle uniform load edge case). Compute composite score, select argmax. First occurrence wins on tie (strict `>` comparison).

In `sim/routing.go`:
```go
// WeightedScoring routes requests using a weighted combination of cache affinity and load balance.
// Score = (1 - KVUtilization) * cacheWeight + (1 - normalizedLoad) * loadWeight.
// Higher scores are preferred. Ties broken by first occurrence in snapshot order.
type WeightedScoring struct {
	cacheWeight float64
	loadWeight  float64
}

// Route implements RoutingPolicy for WeightedScoring.
func (ws *WeightedScoring) Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision {
	if len(snapshots) == 0 {
		panic("WeightedScoring.Route: empty snapshots")
	}

	// Compute loads and find max for normalization
	loads := make([]int, len(snapshots))
	maxLoad := 0
	for i, snap := range snapshots {
		loads[i] = snap.QueueDepth + snap.BatchSize
		if loads[i] > maxLoad {
			maxLoad = loads[i]
		}
	}

	// Compute scores
	scores := make(map[string]float64, len(snapshots))
	bestScore := -1.0
	bestIdx := 0

	for i, snap := range snapshots {
		// Normalize load to [0,1] (handle uniform load edge case: all zero → normalizedLoad = 0)
		normalizedLoad := 0.0
		if maxLoad > 0 {
			normalizedLoad = float64(loads[i]) / float64(maxLoad)
		}

		// Composite score
		cacheScore := (1.0 - snap.KVUtilization) * ws.cacheWeight
		loadScore := (1.0 - normalizedLoad) * ws.loadWeight
		score := cacheScore + loadScore
		scores[snap.ID] = score

		// Select argmax; first occurrence wins on tie (strict >)
		if score > bestScore {
			bestScore = score
			bestIdx = i
		}
	}

	return RoutingDecision{
		TargetInstance: snapshots[bestIdx].ID,
		Reason:         fmt.Sprintf("weighted-scoring (score=%.3f)", bestScore),
		Scores:         scores,
	}
}
```

Also update NewRoutingPolicy factory to include `"weighted"` case returning `&WeightedScoring{cacheWeight: cacheWeight, loadWeight: loadWeight}`.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestWeightedScoring -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/routing.go sim/routing_test.go
git commit -m "feat(sim): add WeightedScoring routing template (BC-4, BC-10, BC-12)

- Implement WeightedScoring with composite score (cache affinity + load balance)
- Normalize load to [0,1] relative to max (handle uniform load edge case)
- Populate Scores map in RoutingDecision for observability

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: PrefixAffinity Template

**Contracts Implemented:** BC-5, BC-8, BC-10

**Files:**
- Modify: `sim/routing.go`
- Modify: `sim/routing_test.go`

**Step 1: Write failing test for PrefixAffinity**

Context: PrefixAffinity maintains map[prefixHash]string (instance ID), routes to cached instance on hit, falls back to LeastLoaded on miss.

In `sim/routing_test.go`:
```go
// TestPrefixAffinity_CacheHit verifies BC-5 (cache-aware routing).
func TestPrefixAffinity_CacheHit(t *testing.T) {
	// GIVEN PrefixAffinity policy
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1},
	}

	// WHEN first request with prefix [1, 2, 3] is routed
	req1 := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	decision1 := policy.Route(req1, snapshots, 1000)
	firstTarget := decision1.TargetInstance

	// WHEN second request with same prefix is routed
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy.Route(req2, snapshots, 2000)

	// THEN second request routes to same instance (cache hit)
	if decision2.TargetInstance != firstTarget {
		t.Errorf("Expected cache hit routing to %q, got %q", firstTarget, decision2.TargetInstance)
	}
}

// TestPrefixAffinity_CacheMiss verifies fallback to LeastLoaded.
func TestPrefixAffinity_CacheMiss(t *testing.T) {
	// GIVEN PrefixAffinity policy
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 10, BatchSize: 5}, // load=15
		{ID: "instance_1", QueueDepth: 2, BatchSize: 1},  // load=3 (min)
	}

	// WHEN request with new prefix is routed (cache miss)
	req := &Request{ID: "req1", InputTokens: []int{7, 8, 9}}
	decision := policy.Route(req, snapshots, 1000)

	// THEN fallback to least-loaded selects instance_1
	if decision.TargetInstance != "instance_1" {
		t.Errorf("Expected fallback to least-loaded (instance_1), got %q", decision.TargetInstance)
	}
}

// TestPrefixAffinity_DifferentPrefixes verifies distinct hashing.
func TestPrefixAffinity_DifferentPrefixes(t *testing.T) {
	// GIVEN PrefixAffinity policy
	policy := NewRoutingPolicy("prefix-affinity", 0, 0)
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
	}

	// WHEN two requests with different prefixes are routed
	req1 := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	req2 := &Request{ID: "req2", InputTokens: []int{4, 5, 6}}

	decision1 := policy.Route(req1, snapshots, 1000)
	decision2 := policy.Route(req2, snapshots, 2000)

	// THEN both routing decisions are valid
	validIDs := map[string]bool{"instance_0": true, "instance_1": true}
	if !validIDs[decision1.TargetInstance] || !validIDs[decision2.TargetInstance] {
		t.Errorf("Invalid routing decisions: %q, %q", decision1.TargetInstance, decision2.TargetInstance)
	}
}

// TestPrefixAffinity_HashMatchesKVCache verifies hash format consistency.
func TestPrefixAffinity_HashMatchesKVCache(t *testing.T) {
	// PrefixAffinity uses pipe-delimited decimal string hash, matching KVCache format.
	// Verify two different token sequences produce different hashes.
	hash1 := computePrefixHash([]int{1, 2, 3})
	hash2 := computePrefixHash([]int{3, 2, 1})
	if hash1 == hash2 {
		t.Errorf("Expected different hashes for [1,2,3] and [3,2,1], got same: %s", hash1)
	}

	// Verify same tokens produce same hash (deterministic)
	hash3 := computePrefixHash([]int{1, 2, 3})
	if hash1 != hash3 {
		t.Errorf("Expected identical hash for same tokens, got %q and %q", hash1, hash3)
	}
}

// TestPrefixAffinity_NoStateLeak verifies BC-8 (no cross-simulation state leak).
func TestPrefixAffinity_NoStateLeak(t *testing.T) {
	// GIVEN two independent PrefixAffinity policy instances
	policy1 := NewRoutingPolicy("prefix-affinity", 0, 0)
	policy2 := NewRoutingPolicy("prefix-affinity", 0, 0)

	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 5},
		{ID: "instance_1", QueueDepth: 5, BatchSize: 5},
	}

	// WHEN policy1 routes a request and builds cache
	req1 := &Request{ID: "req1", InputTokens: []int{1, 2, 3}}
	policy1.Route(req1, snapshots, 1000)

	// WHEN policy2 routes same prefix — it's a cache miss (independent state)
	req2 := &Request{ID: "req2", InputTokens: []int{1, 2, 3}}
	decision2 := policy2.Route(req2, snapshots, 1000)

	// THEN policy2 has no cache state — falls back to LeastLoaded
	// Both instances have equal load, so first occurrence (instance_0) wins
	if decision2.TargetInstance != "instance_0" {
		t.Errorf("Expected independent state (fallback to least-loaded), got %q", decision2.TargetInstance)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestPrefixAffinity -v`
Expected: FAIL with "unknown routing policy \"prefix-affinity\""

**Step 3: Implement PrefixAffinity template**

Context: Hash InputTokens using pipe-delimited decimal string format (matching KVCache's `kvcache.go:108-116`). Maintain prefix-to-instance map. Fallback to LeastLoaded on miss.

In `sim/routing.go`:
```go
import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
)

// PrefixAffinity routes requests with matching prefixes to the same instance (cache-aware).
// On cache miss, falls back to LeastLoaded. Maintains prefix-to-instance mapping.
type PrefixAffinity struct {
	prefixMap map[string]string // prefix hash → instance ID
}

// Route implements RoutingPolicy for PrefixAffinity.
func (pa *PrefixAffinity) Route(req *Request, snapshots []RoutingSnapshot, clock int64) RoutingDecision {
	if len(snapshots) == 0 {
		panic("PrefixAffinity.Route: empty snapshots")
	}

	// Compute prefix hash (matching KVCache format: pipe-delimited decimal strings)
	prefixHash := computePrefixHash(req.InputTokens)

	// Check cache for existing mapping
	if targetID, found := pa.prefixMap[prefixHash]; found {
		// Verify target still in snapshots (instance may have been removed)
		for _, snap := range snapshots {
			if snap.ID == targetID {
				return RoutingDecision{
					TargetInstance: targetID,
					Reason:         "prefix-affinity (cache-hit)",
				}
			}
		}
	}

	// Cache miss or stale entry: fallback to LeastLoaded
	ll := &LeastLoaded{}
	decision := ll.Route(req, snapshots, clock)

	// Update cache with new mapping
	pa.prefixMap[prefixHash] = decision.TargetInstance

	return RoutingDecision{
		TargetInstance: decision.TargetInstance,
		Reason:         "prefix-affinity (cache-miss, fallback to least-loaded)",
	}
}

// computePrefixHash computes SHA256 hash of input tokens using pipe-delimited
// decimal string format, matching KVCache hash computation (kvcache.go:108-116).
func computePrefixHash(tokens []int) string {
	h := sha256.New()
	var b strings.Builder
	for i, token := range tokens {
		if i > 0 {
			b.WriteString("|")
		}
		b.WriteString(strconv.Itoa(token))
	}
	h.Write([]byte(b.String()))
	return hex.EncodeToString(h.Sum(nil))
}
```

Also update NewRoutingPolicy factory to include `"prefix-affinity"` case returning `&PrefixAffinity{prefixMap: make(map[string]string)}`.

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestPrefixAffinity -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit with contract reference**

```bash
git add sim/routing.go sim/routing_test.go
git commit -m "feat(sim): add PrefixAffinity routing template (BC-5, BC-8, BC-10)

- Implement PrefixAffinity with prefix-to-instance cache using SHA256 hash
- Use KVCache-compatible hash format (pipe-delimited decimal strings)
- Fallback to LeastLoaded on cache miss or stale entry

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**CHECKPOINT REVIEW:** Batch 2 complete. All four routing templates implemented: RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity. Verify BC-4, BC-5, BC-8, BC-10, BC-12. Confirm tests pass, lint clean.

---

#### Task 6: CLI Flags and DeploymentConfig

**Contracts Implemented:** BC-6, BC-11

**Files:**
- Modify: `cmd/root.go`

**Step 1: Add CLI flags for routing policy**

Context: Add --routing-policy, --routing-cache-weight, --routing-load-weight flags. Mirror admission flag pattern.

In `cmd/root.go`:

Add flag variables (after line 62, near other routing pipeline vars):
```go
	// Routing policy config (PR 6)
	routingPolicy      string  // Routing policy name
	routingCacheWeight float64 // Cache affinity weight for weighted scoring
	routingLoadWeight  float64 // Load balance weight for weighted scoring
```

Add flag definitions (after line 297, near admission flags):
```go
	// Routing policy config
	runCmd.Flags().StringVar(&routingPolicy, "routing-policy", "round-robin", "Routing policy: round-robin, least-loaded, weighted, prefix-affinity")
	runCmd.Flags().Float64Var(&routingCacheWeight, "routing-cache-weight", 0.6, "Cache affinity weight for weighted routing")
	runCmd.Flags().Float64Var(&routingLoadWeight, "routing-load-weight", 0.4, "Load balance weight for weighted routing")
```

Update DeploymentConfig construction (add after RoutingLatency line ~221):
```go
			RoutingPolicy:             routingPolicy,
			RoutingCacheWeight:        routingCacheWeight,
			RoutingLoadWeight:         routingLoadWeight,
```

**Step 2: Manual CLI test**

Run: `go build -o simulation_worker main.go`
Run: `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --max-prompts 5 --routing-policy least-loaded`
Expected: Simulation completes without error

Run: `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --max-prompts 5 --routing-policy weighted --routing-cache-weight 0.7 --routing-load-weight 0.3`
Expected: Simulation completes without error

Run: `./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --num-instances 2 --max-prompts 5 --routing-policy invalid-name`
Expected: Panic with "unknown routing policy \"invalid-name\""

**Step 3: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 4: Commit with contract reference**

```bash
git add cmd/root.go
git commit -m "feat(cmd): add routing policy CLI flags (BC-6, BC-11)

- Add --routing-policy flag (default \"round-robin\")
- Add --routing-cache-weight flag (default 0.6)
- Add --routing-load-weight flag (default 0.4)
- Wire flags to DeploymentConfig

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 7: Integration Tests and Documentation

**Contracts Implemented:** BC-6 (backward compatibility verification)

**Files:**
- Modify: `sim/cluster/cluster_test.go`
- Modify: `CLAUDE.md`

**Step 1: Write integration tests for all policies**

Context: Verify all four policies complete simulation without panic and that backward compat holds.

In `sim/cluster/cluster_test.go`:
```go
// TestClusterSimulator_AllRoutingPolicies_Smoke verifies all policies are exercisable.
func TestClusterSimulator_AllRoutingPolicies_Smoke(t *testing.T) {
	policies := []string{"round-robin", "least-loaded", "weighted", "prefix-affinity"}

	for _, policyName := range policies {
		t.Run(policyName, func(t *testing.T) {
			config := newTestDeploymentConfig(2)
			config.RoutingPolicy = policyName
			config.RoutingCacheWeight = 0.6
			config.RoutingLoadWeight = 0.4
			workload := newTestWorkload(5)

			cs := NewClusterSimulator(config, workload, "")
			cs.Run()

			// Verify simulation completes without panic
			if cs.AggregatedMetrics().CompletedRequests == 0 {
				t.Errorf("Policy %q: expected non-zero completed requests", policyName)
			}
		})
	}
}
```

**Step 2: Run full test suite**

Run: `go test ./...`
Expected: ALL PASS (including existing golden dataset tests verifying BC-6 backward compatibility)

**Step 3: Update CLAUDE.md**

Context: Update "Current Implementation Focus" to note PR 6 completion and add routing.go to file organization.

In `CLAUDE.md`, update the "Completed" line in "Current Implementation Focus":
```markdown
- **Completed:** PR1 (PartitionedRNG), PR2 (InstanceSimulator), PR3 (ClusterSimulator with shared-clock event loop, round-robin dispatch, metrics aggregation, golden dataset equivalence tests), PR4 (cluster control plane with online routing pipeline, SnapshotProvider, AdmissionPolicy with AlwaysAdmit + TokenBucket templates, cluster event queue), PR5 (architectural simplification: SimConfig struct, unified CLI path through ClusterSimulator, field privatization, AdmissionPolicy consolidated to `sim/admission.go`), **PR6 (RoutingPolicy interface in sim/routing.go with RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity templates; RoutingSnapshot bridge type)**
- **Next:** PR7 (priority+scheduler), then PR8+ (policy bundles, raw metrics, tiered KV cache, decision traces)
```

In the file organization section, add routing.go:
```markdown
│   ├── routing.go             # RoutingPolicy interface, RoutingSnapshot, RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity
```

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit with contract reference**

```bash
git add sim/cluster/cluster_test.go CLAUDE.md
git commit -m "test(cluster): add integration tests and update docs for routing policies (BC-6)

- Add smoke tests for all four routing policy templates
- Update CLAUDE.md to note PR 6 completion and routing.go in file organization
- BC-6 backward compat verified by existing golden dataset tests continuing to pass

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

**CHECKPOINT REVIEW:** Batch 3 complete. CLI flags wired, integration tests added, documentation updated. All contracts BC-1 through BC-12 implemented and tested. Verify full test suite passes, lint clean, CLAUDE.md accurate.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name / Description |
|----------|------|-----------|--------------------------|
| BC-1 | Task 1 | Unit | TestRoutingPolicy_Interface_Contract |
| BC-2 | Task 1 | Unit | TestRoundRobin_DeterministicOrdering |
| BC-3 | Task 2 | Unit | TestLeastLoaded_LoadBasedSelection |
| BC-4 | Task 4 | Unit | TestWeightedScoring_MultiFactor, TestWeightedScoring_UniformLoad |
| BC-5 | Task 5 | Unit | TestPrefixAffinity_CacheHit, TestPrefixAffinity_CacheMiss |
| BC-6 | Task 3,7 | Golden | TestClusterSimulator_SingleInstance_GoldenEquivalence (existing, must pass) |
| BC-7 | Task 3 | Integration | TestClusterSimulator_RoutingPolicy_RoundRobinDefault |
| BC-8 | Task 5 | Unit | TestPrefixAffinity_NoStateLeak |
| BC-9 | Task 3 | Design | Route() takes RoutingSnapshot (sim/ value type), not *InstanceSimulator |
| BC-10 | Tasks 1,2 | Failure | TestRoundRobin_EmptySnapshots_Panics, TestLeastLoaded_EmptySnapshots_Panics |
| BC-11 | Task 1 | Failure | TestNewRoutingPolicy_UnknownName_Panics |
| BC-12 | Task 4 | Unit | TestWeightedScoring_NegativeWeights |

**Golden dataset updates:** None needed. Routing does not change metrics format. BC-6 is verified by the existing `TestClusterSimulator_SingleInstance_GoldenEquivalence` continuing to pass unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| Import cycle between sim/ and sim/cluster/ | High (was in original plan) | Blocker | Fixed: RoutingPolicy in sim/routing.go, RoutingSnapshot bridge type | Task 1 |
| PrefixAffinity hash mismatch with KVCache | Medium | Functional correctness | Fixed: Use KVCache-compatible pipe-delimited format | Task 5 |
| Round-robin backward compatibility broken | Low | Golden dataset regression | Full test suite run in Task 3 Step 5 | Task 3 |
| Snapshot conversion overhead (O(N) per routing) | Low | Performance for 100+ instances | Acceptable for expected cluster sizes; ObservabilityConfig caching mitigates | Task 3 |

---

## PART 3: Quality Assurance

### J) Design Sanity Checklist

Pre-implementation verification:

- [x] No unnecessary abstractions. (RoutingPolicy interface is minimal; RoutingSnapshot is thin bridge)
- [x] No feature creep beyond PR scope. (Only routing; no priority, no scheduler, no RouterState)
- [x] No unexercised flags or interfaces. (All four templates exercisable via --routing-policy flag)
- [x] No partial implementations. (All templates complete and functional)
- [x] No breaking changes without explicit contract updates. (Backward compatible; round-robin default)
- [x] No hidden global state impact. (PrefixAffinity state is per-policy instance, owned by ClusterSimulator)
- [x] All new code will pass golangci-lint. (Verified at each task)
- [x] Shared test helpers used from existing shared test package where applicable.
- [x] CLAUDE.md updated. (Task 7 updates "Current Implementation Focus" and file organization)
- [x] No stale references left in CLAUDE.md. (File paths verified accurate)
- [x] Deviation log reviewed — all deviations justified (package location, RoutingSnapshot, simplified interface).
- [x] Each task produces working, testable code (no scaffolding). (Every task has test-first TDD cycle)
- [x] Task dependencies are correctly ordered. (Interface → templates → integration → CLI → docs)
- [x] All contracts are mapped to specific tasks. (Section H maps BC-1 through BC-12)
- [x] Golden dataset regeneration documented (if needed). (Not needed; routing doesn't change metrics format)
- [x] Dead code removed: `roundRobinCounter` field removed in Task 3.

---

## APPENDIX: File-Level Implementation Details

### File: `sim/routing.go`

**Purpose:** Defines RoutingSnapshot, RoutingPolicy interface, RoutingDecision struct, and four production-ready templates.

**Key Implementation Notes:**

**Package placement:** `sim/` (not `sim/policy/`) to avoid import cycle. Mirrors `sim/admission.go` pattern established in PR 5.

**RoutingSnapshot:** Lightweight value type with same fields as `cluster.InstanceSnapshot` (ID, QueueDepth, BatchSize, KVUtilization, FreeKVBlocks) but uses `string` for ID instead of `cluster.InstanceID`. Populated by `sim/cluster/cluster_event.go` from InstanceSnapshot at routing time.

**RNG usage:** None. Routing policies are deterministic (RoundRobin uses counter, others use snapshot state).

**Metrics:** RoutingDecision.Scores populated only by WeightedScoring (nil for other policies).

**State mutation:**
- RoundRobin: `counter` increments on each Route() call
- PrefixAffinity: `prefixMap` updated on each Route() call (cache miss inserts new entry)

**Error handling:** Panic on empty snapshots (BC-10), panic on unknown policy name in factory (BC-11).

**Hash computation:** `computePrefixHash` uses pipe-delimited decimal strings (`"1|2|3"`) matching `kvcache.go:108-116` format exactly. This ensures routing-level prefix matching aligns with instance-level KV cache hashing.

---

### File: `sim/routing_test.go`

**Purpose:** Unit tests for RoutingPolicy interface and all templates.

**Test Coverage:** See Section H for complete contract-to-test mapping.

---

### File: `sim/cluster/cluster.go`

**Purpose:** Replace `roundRobinCounter` with `routingPolicy` field, initialize in constructor.

**Changes:**
- Remove `roundRobinCounter int` field (line 32)
- Add `routingPolicy sim.RoutingPolicy` field
- Initialize routingPolicy in NewClusterSimulator using `sim.NewRoutingPolicy` factory

---

### File: `sim/cluster/cluster_event.go`

**Purpose:** Replace hardcoded round-robin with snapshot conversion and policy invocation.

**Changes:** Replace RoutingDecisionEvent.Execute (lines 113-117) with:
1. Collect InstanceSnapshots for all instances via snapshotProvider
2. Convert `[]InstanceSnapshot` → `[]sim.RoutingSnapshot` (field-by-field copy, string(ID) conversion)
3. Call `cs.routingPolicy.Route(req, routingSnapshots, cs.clock)`
4. Find target instance by ID and call InjectRequestOnline
5. Panic if target not found (defensive; should never happen per BC-7)

**Snapshot timing:** Snapshots collected at `cs.clock` (simulation time when event fires), request injected at `e.time` (event's scheduled timestamp). This is correct: routing sees the state at decision time, injection happens at the event's scheduled time (which may differ from cs.clock only by event processing order within same timestamp).

---

### File: `sim/cluster/deployment.go`

**Purpose:** Add routing policy configuration fields to DeploymentConfig.

**Changes:** After line 30, add RoutingPolicy, RoutingCacheWeight, RoutingLoadWeight fields.

---

### File: `cmd/root.go`

**Purpose:** Add CLI flags for routing policy configuration.

**Changes:**
- Add routingPolicy, routingCacheWeight, routingLoadWeight flag variables
- Add runCmd.Flags() definitions with defaults
- Wire flags to DeploymentConfig struct

---

### File: `CLAUDE.md`

**Purpose:** Update "Current Implementation Focus" to reflect PR 6 completion and add routing.go to file organization.

---

## EXECUTION HANDOFF

**Plan complete and saved to `docs/plans/pr6-routing-plan.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
