# PR8: RouterState and PolicyBundle Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce `RouterState` as a unified cluster-state parameter for policy interfaces, and `PolicyBundle` for YAML-based policy configuration via `--policy-config`.

**Architecture:** `RouterState` lives in `sim/` (bridge type pattern, avoids import cycles) and subsumes the separate `snapshots`/`clock` parameters currently passed to policies. `PolicyBundle` in `sim/bundle.go` parses YAML config files and merges with CLI flag defaults. After PR8, policy interfaces are frozen (additive changes only).

**Macro Plan Reference:** Phase 2b, PR 8 — `docs/plans/2026-02-11-macro-implementation-plan-v2.md:1211`

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

- **Building block:** RouterState (cluster-wide state for policies) + PolicyBundle (YAML config)
- **Adjacent blocks:** AdmissionPolicy, RoutingPolicy, PriorityPolicy (interfaces updated), ClusterSimulator (wiring), cmd/root.go (CLI)
- **DEVIATION flags:** (1) `sim/policy/` package does not exist; files go in `sim/` per established pattern. (2) PriorityPolicy and InstanceScheduler signatures unchanged — they operate at instance level, not cluster level. (3) TenantState/GlobalMetrics deferred to PR9+ — no TenantID on Request yet.

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: RouterState Construction**
- GIVEN a ClusterSimulator with N instances and a current clock
- WHEN a RouterState is constructed
- THEN it MUST contain exactly N RoutingSnapshots (one per instance) and the current clock value
- MECHANISM: Struct literal `&RouterState{Snapshots: snapshots, Clock: clock}` in `sim/router_state.go`

**BC-2: AdmissionPolicy Receives RouterState**
- GIVEN an AdmissionPolicy implementation (AlwaysAdmit or TokenBucket)
- WHEN `Admit(req, state)` is called with a valid RouterState
- THEN behavior MUST be identical to the previous `Admit(req, clock)` — state.Clock replaces the clock parameter
- MECHANISM: Interface signature change; implementations read `state.Clock`

**BC-3: RoutingPolicy Receives RouterState**
- GIVEN a RoutingPolicy implementation
- WHEN `Route(req, state)` is called with a valid RouterState
- THEN behavior MUST be identical to the previous `Route(req, snapshots, clock)` — state.Snapshots and state.Clock replace separate parameters
- MECHANISM: Interface signature change; implementations read `state.Snapshots` and `state.Clock`

**BC-4: Golden Dataset Equivalence**
- GIVEN the golden dataset test cases
- WHEN run through ClusterSimulator with default policies and RouterState wiring
- THEN all metrics MUST match exactly (bit-for-bit identical output)
- MECHANISM: RouterState is a transparent wrapper; no behavioral change with default policies

**BC-8: Admission Receives Full RouterState (Snapshots + Clock)**
- GIVEN an AdmissionDecisionEvent executing in the cluster pipeline
- WHEN the admission policy is invoked
- THEN the RouterState MUST contain both Snapshots (one per instance) and Clock — not just Clock
- MECHANISM: `AdmissionDecisionEvent.Execute` builds full RouterState with snapshot collection before calling `Admit`
- RATIONALE: Enables future admission policies that are load-aware (e.g., "admit only if at least one instance has <80% KV utilization")

**BC-9: Routing Priority Hint**
- GIVEN a RoutingDecision returned by a RoutingPolicy
- WHEN `RoutingDecision.Priority` is non-zero
- THEN `RoutingDecisionEvent.Execute` MUST set `req.Priority = decision.Priority` before injecting the request into the target instance
- WHEN `RoutingDecision.Priority` is zero (default)
- THEN `req.Priority` MUST NOT be modified (instance-level PriorityPolicy will compute it)
- MECHANISM: New `Priority float64` field on `RoutingDecision` struct; checked in `RoutingDecisionEvent.Execute`
- RATIONALE: Enables cluster-level priority assignment and joint admission+routing optimization strategies

**BC-5: PolicyBundle YAML Loading**
- GIVEN a valid YAML file with policy configuration
- WHEN `LoadPolicyBundle(path)` is called
- THEN it MUST return a PolicyBundle struct with all fields populated from YAML
- MECHANISM: `sim/bundle.go` with `yaml.v3` parsing

**BC-6: PolicyBundle CLI Override**
- GIVEN a PolicyBundle loaded from YAML AND CLI flags set
- WHEN the configuration is merged
- THEN CLI flags MUST override YAML values (CLI takes precedence)
- MECHANISM: Inline merge in `cmd/root.go` using Cobra's `cmd.Flags().Changed()` to detect explicit CLI flags

**BC-7: PolicyBundle Validation**
- GIVEN a PolicyBundle with an invalid policy name
- WHEN `Validate()` is called
- THEN it MUST return an error describing the invalid field
- MECHANISM: Validation against known policy names

#### Negative Contracts

**NC-1: No Interface Change for Instance-Level Policies**
- PriorityPolicy and InstanceScheduler signatures MUST NOT change — they operate per-instance, not at cluster level
- MECHANISM: Only AdmissionPolicy and RoutingPolicy are cluster-level

**NC-2: No TenantState in This PR**
- RouterState MUST NOT include TenantState or GlobalMetrics fields — no TenantID on Request yet
- MECHANISM: Deferred to PR9+; documented in deviation log

#### Error Handling Contracts

**EC-1: Invalid YAML File**
- GIVEN a path to a nonexistent or malformed YAML file
- WHEN `LoadPolicyBundle(path)` is called
- THEN it MUST return an error (not panic)
- MECHANISM: `os.ReadFile` + `yaml.Unmarshal` error propagation

**EC-2: Empty PolicyBundle Uses Defaults**
- GIVEN a PolicyBundle with zero-value fields (empty strings)
- WHEN applied to DeploymentConfig
- THEN zero-value fields MUST NOT override existing DeploymentConfig values
- MECHANISM: Only non-empty PolicyBundle fields are applied

### C) Component Interaction

```
                     cmd/root.go
                         │
            ┌────────────┼──────────────┐
            │            │              │
       --policy-config   │         CLI flags
            │            │         (override)
            ▼            │              │
      LoadPolicyBundle   │              │
            │            ▼              │
            └────► DeploymentConfig ◄───┘
                         │
                         ▼
                  ClusterSimulator
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
         Admission    Routing    Instance
          Policy      Policy     (unchanged)
              │          │
              └────┬─────┘
                   ▼
             *RouterState ─────────────────── shared: built once per
          (Snapshots + Clock)                 arrival pipeline, reused
                                              by admission and routing
```

**API Contracts:**
- `RouterState` — value type in `sim/`, fields: `Snapshots []RoutingSnapshot`, `Clock int64`
- `RoutingDecision` — add `Priority float64` field (zero = defer to instance-level PriorityPolicy)
- `AdmissionPolicy.Admit(req *Request, state *RouterState) (bool, string)`
- `RoutingPolicy.Route(req *Request, state *RouterState) RoutingDecision`
- `PolicyBundle` — struct in `sim/` with YAML tags, `LoadPolicyBundle(path) (*PolicyBundle, error)`, `Validate() error`

**State Changes:**
- `ClusterSimulator` builds `RouterState` once per arrival pipeline (snapshot collection + clock), reused by both admission and routing events
- `RoutingDecisionEvent.Execute` applies `decision.Priority` to request if non-zero
- `PolicyBundle` is stateless (loaded once, applied to config)

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| Files in `sim/policy/bundle.go` | Files in `sim/bundle.go` | `sim/policy/` does not exist; established pattern (PR4-7) keeps policy types in `sim/` |
| RouterState in `sim/cluster/router_state.go` | RouterState in `sim/router_state.go` | Import cycle: policy interfaces in `sim/` cannot import `sim/cluster/`; bridge type pattern |
| In Scope: TenantState, GlobalMetrics | Deferred to PR9+ | No TenantID on Request; multi-tenant requires workload changes beyond PR8 scope |
| PriorityPolicy gets RouterState parameter | PriorityPolicy unchanged | PriorityPolicy is called in Simulator.Step() (instance-level); threading cluster state there adds complexity with no current consumer |
| `AdmissionPolicy.Decide()` returns `AdmissionDecision` struct | Keep `Admit()` returning `(bool, string)` | SIMPLIFICATION: Current return type is sufficient; AdmissionDecision struct adds DELAY action not needed until autoscaler (PR11) |

### E) Review Guide

1. **THE TRICKY PART:** (a) `PrefixAffinity.Route` internally delegates to `LeastLoaded.Route` at routing.go:164 — this production call site must pass the received `state` through, not reconstruct one. (b) The shared RouterState for admission+routing means snapshot collection happens at admission time — verify staleness is acceptable.
2. **WHAT TO SCRUTINIZE:** BC-4 (golden equivalence) — key regression gate. BC-9 (priority hint) — verify zero-value doesn't mutate request. Also verify YAML merge logic (BC-6).
3. **WHAT'S SAFE TO SKIM:** The mechanical test updates (changing `policy.Route(req, snaps, clock)` → `policy.Route(req, state)`) — these are repetitive.
4. **KNOWN DEBT:** (a) routing.go:26 comment says "transitional interface for PR 6; PR 8 will extend with RouterState parameter" — remove. (b) priority.go:26 comment says "Full SLO class integration (using TenantState) is planned for PR8" — update to "PR9+". (c) PolicyBundle `Validate()` hardcodes valid policy names, duplicating the `New*` factory functions — acceptable for now but tech debt if new policy templates are added.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/router_state.go` — RouterState struct (~20 LOC)
- `sim/router_state_test.go` — RouterState tests (~40 LOC)
- `sim/bundle.go` — PolicyBundle struct + YAML loading (~90 LOC)
- `sim/bundle_test.go` — PolicyBundle tests (~120 LOC)

**Files to modify:**
- `sim/admission.go` — Change `Admit` signature to accept `*RouterState` (~10 LOC)
- `sim/admission_test.go` — Update test call sites (~30 LOC)
- `sim/routing.go` — Change `Route` signature to accept `*RouterState`, add `Priority` field to `RoutingDecision`, remove transitional comment (~45 LOC)
- `sim/routing_test.go` — Update test call sites + add priority hint test (~100 LOC)
- `sim/priority.go` — Update stale PR8 comment to say PR9+ (~1 LOC)
- `sim/cluster/cluster_event.go` — Build shared RouterState, pass to policies, apply priority hint (~25 LOC)
- `sim/cluster/cluster_test.go` — Minor updates if needed
- `cmd/root.go` — Add `--policy-config` flag, load and merge (~40 LOC)
- `CLAUDE.md` — Update completed PRs, file organization

**Key decisions:**
- RouterState is a bridge type in `sim/` (same pattern as RoutingSnapshot)
- RouterState built once per arrival pipeline (snapshots collected at admission time, reused for routing)
- `RoutingDecision.Priority` enables cluster-level priority assignment without changing PriorityPolicy interface
- PolicyBundle uses `gopkg.in/yaml.v3` (already a dependency)
- CLI flags override YAML values (non-empty CLI value wins)

**Confirmation:** No dead code — RouterState used by policy interfaces, Priority field exercisable via custom RoutingPolicy, PolicyBundle used by CLI.

### G) Task Breakdown

#### Task 1: RouterState Type

**Contracts Implemented:** BC-1

**Files:**
- Create: `sim/router_state.go`
- Create: `sim/router_state_test.go`

**Step 1: Write failing test**

```go
// sim/router_state_test.go
package sim

import "testing"

func TestRouterState_FieldAccess(t *testing.T) {
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 5, BatchSize: 2, KVUtilization: 0.3, FreeKVBlocks: 100},
		{ID: "instance_1", QueueDepth: 3, BatchSize: 1, KVUtilization: 0.5, FreeKVBlocks: 80},
	}
	state := &RouterState{
		Snapshots: snapshots,
		Clock:     42000,
	}

	if len(state.Snapshots) != 2 {
		t.Errorf("expected 2 snapshots, got %d", len(state.Snapshots))
	}
	if state.Clock != 42000 {
		t.Errorf("expected clock 42000, got %d", state.Clock)
	}
	if state.Snapshots[0].ID != "instance_0" {
		t.Errorf("expected first snapshot ID 'instance_0', got %q", state.Snapshots[0].ID)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestRouterState_FieldAccess -v`
Expected: FAIL — `RouterState` undefined

**Step 3: Implement**

```go
// sim/router_state.go
package sim

// RouterState provides cluster-wide state to policy interfaces.
// Built by ClusterSimulator before each policy invocation.
// This is a bridge type in sim/ (not sim/cluster/) to avoid import cycles —
// same pattern as RoutingSnapshot.
//
// USAGE BOUNDARY: Only constructed by ClusterSimulator's event handlers.
// Single-instance Simulator does not use RouterState — instance-level policies
// (PriorityPolicy, InstanceScheduler) receive parameters directly.
type RouterState struct {
	Snapshots []RoutingSnapshot // One per instance, same order as ClusterSimulator.instances
	Clock     int64             // Current simulation clock in microseconds
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestRouterState_FieldAccess -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/router_state.go sim/router_state_test.go
git commit -m "feat(sim): add RouterState bridge type (BC-1)

- RouterState holds Snapshots + Clock for policy interfaces
- Bridge type in sim/ to avoid import cycles (same pattern as RoutingSnapshot)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Update AdmissionPolicy Interface

**Contracts Implemented:** BC-2, BC-8

**Files:**
- Modify: `sim/admission.go`
- Modify: `sim/admission_test.go`
- Modify: `sim/cluster/cluster_event.go` (AdmissionDecisionEvent.Execute + helper)

**Step 1: Write failing test**

Update `sim/admission_test.go` — change all `Admit(req, clock)` calls to `Admit(req, &RouterState{Clock: clock})`:

```go
// sim/admission_test.go
package sim

import (
	"testing"
)

func TestAlwaysAdmit_AdmitsAll(t *testing.T) {
	policy := &AlwaysAdmit{}

	tests := []struct {
		name  string
		req   *Request
		clock int64
	}{
		{name: "empty request", req: &Request{ID: "r0", InputTokens: []int{}}, clock: 0},
		{name: "small request", req: &Request{ID: "r1", InputTokens: make([]int, 10)}, clock: 1000},
		{name: "large request", req: &Request{ID: "r2", InputTokens: make([]int, 10000)}, clock: 5_000_000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state := &RouterState{Clock: tt.clock}
			admitted, reason := policy.Admit(tt.req, state)
			if !admitted {
				t.Errorf("expected admitted=true, got false")
			}
			if reason != "" {
				t.Errorf("expected empty reason, got %q", reason)
			}
		})
	}
}

func TestTokenBucket_AdmitAndReject(t *testing.T) {
	t.Run("admits when tokens available", func(t *testing.T) {
		tb := NewTokenBucket(100, 10)
		req := &Request{ID: "r0", InputTokens: make([]int, 50)}
		admitted, reason := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("expected admission with sufficient tokens")
		}
		if reason != "" {
			t.Errorf("expected empty reason, got %q", reason)
		}
	})

	t.Run("rejects when tokens exhausted", func(t *testing.T) {
		tb := NewTokenBucket(10, 0)
		req := &Request{ID: "r0", InputTokens: make([]int, 10)}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		admitted, reason := tb.Admit(req, &RouterState{Clock: 0})
		if admitted {
			t.Fatal("expected rejection with exhausted tokens")
		}
		if reason != "insufficient tokens" {
			t.Errorf("expected reason %q, got %q", "insufficient tokens", reason)
		}
	})

	t.Run("refill restores tokens over time", func(t *testing.T) {
		tb := NewTokenBucket(100, 1_000_000)
		req := &Request{ID: "r0", InputTokens: make([]int, 100)}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("first request should be admitted")
		}

		admitted, _ = tb.Admit(req, &RouterState{Clock: 50})
		if admitted {
			t.Fatal("expected rejection: only 50 tokens refilled, need 100")
		}

		admitted, reason := tb.Admit(req, &RouterState{Clock: 150})
		if !admitted {
			t.Fatalf("expected admission after refill, reason: %s", reason)
		}
	})

	t.Run("capacity caps refill", func(t *testing.T) {
		tb := NewTokenBucket(10, 1_000_000)
		req := &Request{ID: "r0", InputTokens: make([]int, 5)}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("should admit")
		}

		admitted, _ = tb.Admit(req, &RouterState{Clock: 1_000_000})
		if !admitted {
			t.Fatal("should admit after refill")
		}

		admitted, _ = tb.Admit(req, &RouterState{Clock: 1_000_000})
		if !admitted {
			t.Fatal("should admit: 5 tokens remain")
		}

		admitted, _ = tb.Admit(req, &RouterState{Clock: 1_000_000})
		if admitted {
			t.Fatal("should reject: 0 tokens remain, no time elapsed for refill")
		}
	})

	t.Run("zero-cost request always admitted", func(t *testing.T) {
		tb := NewTokenBucket(0, 0)
		req := &Request{ID: "r0", InputTokens: []int{}}

		admitted, _ := tb.Admit(req, &RouterState{Clock: 0})
		if !admitted {
			t.Fatal("zero-cost request should always be admitted")
		}
	})
}

func TestNewAdmissionPolicy_ValidNames(t *testing.T) {
	t.Run("always-admit", func(t *testing.T) {
		p := NewAdmissionPolicy("always-admit", 0, 0)
		if _, ok := p.(*AlwaysAdmit); !ok {
			t.Errorf("expected *AlwaysAdmit, got %T", p)
		}
	})

	t.Run("empty string returns AlwaysAdmit", func(t *testing.T) {
		p := NewAdmissionPolicy("", 0, 0)
		if _, ok := p.(*AlwaysAdmit); !ok {
			t.Errorf("expected *AlwaysAdmit for empty string, got %T", p)
		}
	})

	t.Run("token-bucket", func(t *testing.T) {
		p := NewAdmissionPolicy("token-bucket", 100, 10)
		if _, ok := p.(*TokenBucket); !ok {
			t.Errorf("expected *TokenBucket, got %T", p)
		}
	})
}

func TestNewAdmissionPolicy_InvalidName_Panics(t *testing.T) {
	tests := []struct {
		name       string
		policyName string
	}{
		{"unknown name", "round-robin"},
		{"typo", "always_admit"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			defer func() {
				r := recover()
				if r == nil {
					t.Errorf("expected panic for policy name %q, got none", tt.policyName)
				}
			}()
			NewAdmissionPolicy(tt.policyName, 0, 0)
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestAlwaysAdmit -v`
Expected: FAIL — `Admit` has wrong signature

**Step 3: Implement**

In `sim/admission.go`, change interface and implementations:

```go
package sim

import "fmt"

// AdmissionPolicy decides whether a request is admitted for processing.
// Used by ClusterSimulator's online routing pipeline to gate incoming requests.
type AdmissionPolicy interface {
	Admit(req *Request, state *RouterState) (admitted bool, reason string)
}

type AlwaysAdmit struct{}

func (a *AlwaysAdmit) Admit(_ *Request, _ *RouterState) (bool, string) {
	return true, ""
}

type TokenBucket struct {
	capacity      float64
	refillRate    float64
	currentTokens float64
	lastRefill    int64
}

func NewTokenBucket(capacity, refillRate float64) *TokenBucket {
	return &TokenBucket{
		capacity:      capacity,
		refillRate:    refillRate,
		currentTokens: capacity,
	}
}

func (tb *TokenBucket) Admit(req *Request, state *RouterState) (bool, string) {
	clock := state.Clock
	elapsed := clock - tb.lastRefill
	if elapsed > 0 {
		refill := float64(elapsed) * tb.refillRate / 1e6
		tb.currentTokens = min(tb.capacity, tb.currentTokens+refill)
		tb.lastRefill = clock
	}
	cost := float64(len(req.InputTokens))
	if tb.currentTokens >= cost {
		tb.currentTokens -= cost
		return true, ""
	}
	return false, "insufficient tokens"
}

func NewAdmissionPolicy(name string, capacity, refillRate float64) AdmissionPolicy {
	switch name {
	case "", "always-admit":
		return &AlwaysAdmit{}
	case "token-bucket":
		return NewTokenBucket(capacity, refillRate)
	default:
		panic(fmt.Sprintf("unknown admission policy %q", name))
	}
}
```

In `sim/cluster/cluster_event.go`, add a shared helper to build RouterState (reused by admission and routing), and update `AdmissionDecisionEvent.Execute`:

```go
// buildRouterState constructs a RouterState from current cluster state.
// Called by both AdmissionDecisionEvent and RoutingDecisionEvent to provide
// cluster-wide context to policies. Snapshots are collected from the
// SnapshotProvider at the current clock time.
func buildRouterState(cs *ClusterSimulator) *sim.RouterState {
	snapshots := make([]sim.RoutingSnapshot, len(cs.instances))
	for i, inst := range cs.instances {
		snap := cs.snapshotProvider.Snapshot(inst.ID(), cs.clock)
		snapshots[i] = sim.RoutingSnapshot{
			ID:            string(snap.ID),
			QueueDepth:    snap.QueueDepth,
			BatchSize:     snap.BatchSize,
			KVUtilization: snap.KVUtilization,
			FreeKVBlocks:  snap.FreeKVBlocks,
		}
	}
	return &sim.RouterState{
		Snapshots: snapshots,
		Clock:     cs.clock,
	}
}

// In AdmissionDecisionEvent.Execute, change line 89 from:
//   admitted, _ := cs.admissionPolicy.Admit(e.request, cs.clock)
// To:
	state := buildRouterState(cs)
	admitted, _ := cs.admissionPolicy.Admit(e.request, state)
```

**Step 4: Run tests**

Run: `go test ./sim/... ./sim/cluster/... -v`
Expected: PASS (all tests)

**Step 5: Run lint**

Run: `golangci-lint run ./sim/... ./sim/cluster/...`

**Step 6: Commit**

```bash
git add sim/admission.go sim/admission_test.go sim/cluster/cluster_event.go
git commit -m "feat(sim): update AdmissionPolicy to accept *RouterState (BC-2)

- Change Admit(req, clock) → Admit(req, state) where state.Clock replaces clock
- Update AlwaysAdmit and TokenBucket implementations
- Update AdmissionDecisionEvent.Execute to build RouterState
- All existing tests updated and passing

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Update RoutingPolicy Interface

**Contracts Implemented:** BC-3, BC-9

**Files:**
- Modify: `sim/routing.go`
- Modify: `sim/routing_test.go`
- Modify: `sim/priority.go` (update stale comment)
- Modify: `sim/cluster/cluster_event.go` (RoutingDecisionEvent.Execute)

**Step 1: Write failing test**

Update the first test in `sim/routing_test.go` to use new signature:

```go
// Change all occurrences of:
//   policy.Route(req, snapshots, clock)
// To:
//   policy.Route(req, &RouterState{Snapshots: snapshots, Clock: clock})
```

The full test file is large (454 lines). The mechanical transformation is:
- Every `policy.Route(req, snapshots, clock)` → `policy.Route(req, &RouterState{Snapshots: snapshots, Clock: clock})`
- Every `policy.Route(req, snapshots, int64(N))` → `policy.Route(req, &RouterState{Snapshots: snapshots, Clock: int64(N)})`

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestRoutingPolicy -v`
Expected: FAIL — `Route` has wrong signature

**Step 3: Implement**

In `sim/routing.go`:

1. Add `Priority` field to `RoutingDecision`:
```go
type RoutingDecision struct {
	TargetInstance string             // Instance ID to route to (must match a snapshot ID)
	Reason         string             // Human-readable explanation
	Scores         map[string]float64 // Instance ID → composite score (nil for policies without scoring)
	Priority       float64            // Cluster-level priority hint; 0 = defer to instance PriorityPolicy
}
```

2. Change interface:
```go
type RoutingPolicy interface {
	Route(req *Request, state *RouterState) RoutingDecision
}
```

3. Remove the transitional comment on line 26.

4. Update each implementation to read `state.Snapshots` and `state.Clock`. Pattern for `RoundRobin`:
```go
func (rr *RoundRobin) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("RoundRobin.Route: empty snapshots")
	}
	// ... rest unchanged, uses snapshots local var
}
```

5. Same pattern for `LeastLoaded`, `WeightedScoring`.

6. **`PrefixAffinity` — critical detail:** The internal delegation to `LeastLoaded.Route` at line 164 must pass the received `state` through (not reconstruct):
```go
// In PrefixAffinity.Route:
func (pa *PrefixAffinity) Route(req *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	// ... prefix hash lookup ...

	// Cache miss: delegate to LeastLoaded, passing state through
	ll := &LeastLoaded{}
	decision := ll.Route(req, state)  // NOT ll.Route(req, snapshots, clock)

	pa.prefixMap[prefixHash] = decision.TargetInstance
	// ...
}
```

7. Update stale comment in `sim/priority.go:26` from "planned for PR8" to "planned for PR9+".

In `sim/cluster/cluster_event.go`, update `RoutingDecisionEvent.Execute` to use shared `buildRouterState` helper (from Task 2) and apply priority hint:

```go
func (e *RoutingDecisionEvent) Execute(cs *ClusterSimulator) {
	state := buildRouterState(cs)
	decision := cs.routingPolicy.Route(e.request, state)

	// BC-9: Apply cluster-level priority hint if set
	if decision.Priority != 0 {
		e.request.Priority = decision.Priority
	}

	for _, inst := range cs.instances {
		if string(inst.ID()) == decision.TargetInstance {
			inst.InjectRequestOnline(e.request, e.time)
			return
		}
	}
	panic(fmt.Sprintf("RoutingDecisionEvent: invalid TargetInstance %q", decision.TargetInstance))
}
```

Add a test for BC-9 in `sim/routing_test.go`:

```go
func TestRoutingDecision_PriorityHint_DefaultZero(t *testing.T) {
	// GIVEN any routing policy
	policy := NewRoutingPolicy("round-robin", 0, 0)
	state := &RouterState{
		Snapshots: []RoutingSnapshot{{ID: "instance_0"}},
		Clock:     1000,
	}

	// WHEN Route is called
	decision := policy.Route(&Request{ID: "req1"}, state)

	// THEN Priority is zero (defer to instance-level PriorityPolicy)
	if decision.Priority != 0 {
		t.Errorf("expected default Priority 0, got %f", decision.Priority)
	}
}
```

**Step 4: Run tests**

Run: `go test ./sim/... ./sim/cluster/... -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/... ./sim/cluster/...`

**Step 6: Commit**

```bash
git add sim/routing.go sim/routing_test.go sim/priority.go sim/cluster/cluster_event.go
git commit -m "feat(sim): update RoutingPolicy to accept *RouterState, add Priority hint (BC-3, BC-9)

- Change Route(req, snapshots, clock) → Route(req, state)
- Add Priority float64 field to RoutingDecision (cluster-level priority hint)
- Update RoundRobin, LeastLoaded, WeightedScoring, PrefixAffinity
- PrefixAffinity passes received state through to LeastLoaded fallback
- Update RoutingDecisionEvent.Execute: use buildRouterState, apply priority hint
- Remove transitional PR6 comment from routing.go
- Update stale PR8 comment in priority.go to PR9+
- All routing tests updated and passing

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Golden Dataset Equivalence + BC-8 Verification

**Contracts Implemented:** BC-4, BC-8

**Files:**
- Modify: `sim/cluster/cluster_event_test.go` (add BC-8 test)

**Step 1: Run golden dataset tests**

Run: `go test ./sim/... ./sim/cluster/... -v -run Golden`
Expected: PASS — all golden dataset test cases produce identical metrics

If any golden test fails, the RouterState wiring has introduced a behavioral change that must be investigated and fixed before proceeding.

**Step 2: Add BC-8 test — verify buildRouterState populates snapshots for admission**

```go
// In sim/cluster/cluster_event_test.go, add:

func TestBuildRouterState_PopulatesSnapshots(t *testing.T) {
	// GIVEN a ClusterSimulator with 2 instances
	config := testDeploymentConfig(2) // use existing test helper or inline
	cs := NewClusterSimulator(config, testWorkloadConfig(), "")

	// WHEN buildRouterState is called
	state := buildRouterState(cs)

	// THEN state must contain exactly 2 snapshots and the current clock
	if len(state.Snapshots) != 2 {
		t.Errorf("expected 2 snapshots, got %d", len(state.Snapshots))
	}
	if state.Clock != cs.Clock() {
		t.Errorf("expected clock %d, got %d", cs.Clock(), state.Clock)
	}
	// Verify each snapshot has a valid instance ID
	for i, snap := range state.Snapshots {
		if snap.ID == "" {
			t.Errorf("snapshot %d has empty ID", i)
		}
	}
}
```

Note: Adapt the test helper construction to match existing patterns in `cluster_test.go`. The key assertion is `len(state.Snapshots) == config.NumInstances`.

**Step 3: Run full test suite**

Run: `go test ./... -v`
Expected: PASS — all tests green

**Step 4: Commit** (if BC-8 test added)

```bash
git add sim/cluster/cluster_event_test.go
git commit -m "test(cluster): add BC-8 test verifying buildRouterState populates snapshots

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: PolicyBundle Type and YAML Loading

**Contracts Implemented:** BC-5, BC-7, EC-1, EC-2

**Files:**
- Create: `sim/bundle.go`
- Create: `sim/bundle_test.go`

**Step 1: Write failing tests**

```go
// sim/bundle_test.go
package sim

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadPolicyBundle_ValidYAML(t *testing.T) {
	yaml := `
admission:
  policy: token-bucket
  token_bucket_capacity: 5000
  token_bucket_refill_rate: 500
routing:
  policy: weighted
  cache_weight: 0.7
  load_weight: 0.3
priority:
  policy: slo-based
scheduler: priority-fcfs
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Admission.Policy != "token-bucket" {
		t.Errorf("expected admission policy 'token-bucket', got %q", bundle.Admission.Policy)
	}
	if bundle.Admission.TokenBucketCapacity != 5000 {
		t.Errorf("expected capacity 5000, got %f", bundle.Admission.TokenBucketCapacity)
	}
	if bundle.Routing.Policy != "weighted" {
		t.Errorf("expected routing policy 'weighted', got %q", bundle.Routing.Policy)
	}
	if bundle.Routing.CacheWeight != 0.7 {
		t.Errorf("expected cache weight 0.7, got %f", bundle.Routing.CacheWeight)
	}
	if bundle.Priority.Policy != "slo-based" {
		t.Errorf("expected priority policy 'slo-based', got %q", bundle.Priority.Policy)
	}
	if bundle.Scheduler != "priority-fcfs" {
		t.Errorf("expected scheduler 'priority-fcfs', got %q", bundle.Scheduler)
	}
}

func TestLoadPolicyBundle_EmptyFields(t *testing.T) {
	yaml := `
routing:
  policy: least-loaded
`
	path := writeTempYAML(t, yaml)
	bundle, err := LoadPolicyBundle(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Admission.Policy != "" {
		t.Errorf("expected empty admission policy, got %q", bundle.Admission.Policy)
	}
	if bundle.Routing.Policy != "least-loaded" {
		t.Errorf("expected 'least-loaded', got %q", bundle.Routing.Policy)
	}
	if bundle.Scheduler != "" {
		t.Errorf("expected empty scheduler, got %q", bundle.Scheduler)
	}
}

func TestLoadPolicyBundle_NonexistentFile(t *testing.T) {
	_, err := LoadPolicyBundle("/nonexistent/path.yaml")
	if err == nil {
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoadPolicyBundle_MalformedYAML(t *testing.T) {
	path := writeTempYAML(t, "{{invalid yaml")
	_, err := LoadPolicyBundle(path)
	if err == nil {
		t.Fatal("expected error for malformed YAML")
	}
}

func TestPolicyBundle_Validate_ValidPolicies(t *testing.T) {
	bundle := &PolicyBundle{
		Admission: AdmissionConfig{Policy: "token-bucket"},
		Routing:   RoutingConfig{Policy: "weighted"},
		Priority:  PriorityConfig{Policy: "slo-based"},
		Scheduler: "priority-fcfs",
	}
	if err := bundle.Validate(); err != nil {
		t.Errorf("expected no error, got: %v", err)
	}
}

func TestPolicyBundle_Validate_EmptyIsValid(t *testing.T) {
	bundle := &PolicyBundle{}
	if err := bundle.Validate(); err != nil {
		t.Errorf("empty bundle should be valid, got: %v", err)
	}
}

func TestPolicyBundle_Validate_InvalidPolicy(t *testing.T) {
	tests := []struct {
		name   string
		bundle PolicyBundle
	}{
		{"bad admission", PolicyBundle{Admission: AdmissionConfig{Policy: "invalid"}}},
		{"bad routing", PolicyBundle{Routing: RoutingConfig{Policy: "invalid"}}},
		{"bad priority", PolicyBundle{Priority: PriorityConfig{Policy: "invalid"}}},
		{"bad scheduler", PolicyBundle{Scheduler: "invalid"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := tt.bundle.Validate(); err == nil {
				t.Error("expected validation error")
			}
		})
	}
}

func writeTempYAML(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "policy.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestLoadPolicyBundle -v`
Expected: FAIL — `LoadPolicyBundle` undefined

**Step 3: Implement**

```go
// sim/bundle.go
package sim

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// PolicyBundle holds unified policy configuration, loadable from a YAML file.
// Zero-value fields mean "use default" — they do not override DeploymentConfig.
type PolicyBundle struct {
	Admission AdmissionConfig `yaml:"admission"`
	Routing   RoutingConfig   `yaml:"routing"`
	Priority  PriorityConfig  `yaml:"priority"`
	Scheduler string          `yaml:"scheduler"`
}

// AdmissionConfig holds admission policy configuration.
type AdmissionConfig struct {
	Policy                string  `yaml:"policy"`
	TokenBucketCapacity   float64 `yaml:"token_bucket_capacity"`
	TokenBucketRefillRate float64 `yaml:"token_bucket_refill_rate"`
}

// RoutingConfig holds routing policy configuration.
type RoutingConfig struct {
	Policy      string  `yaml:"policy"`
	CacheWeight float64 `yaml:"cache_weight"`
	LoadWeight  float64 `yaml:"load_weight"`
}

// PriorityConfig holds priority policy configuration.
type PriorityConfig struct {
	Policy string `yaml:"policy"`
}

// LoadPolicyBundle reads and parses a YAML policy configuration file.
func LoadPolicyBundle(path string) (*PolicyBundle, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading policy config: %w", err)
	}
	var bundle PolicyBundle
	if err := yaml.Unmarshal(data, &bundle); err != nil {
		return nil, fmt.Errorf("parsing policy config: %w", err)
	}
	return &bundle, nil
}

var (
	validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true}
	validRoutingPolicies   = map[string]bool{"": true, "round-robin": true, "least-loaded": true, "weighted": true, "prefix-affinity": true}
	validPriorityPolicies  = map[string]bool{"": true, "constant": true, "slo-based": true}
	validSchedulers        = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true}
)

// Validate checks that all policy names in the bundle are recognized.
func (b *PolicyBundle) Validate() error {
	if !validAdmissionPolicies[b.Admission.Policy] {
		return fmt.Errorf("unknown admission policy %q", b.Admission.Policy)
	}
	if !validRoutingPolicies[b.Routing.Policy] {
		return fmt.Errorf("unknown routing policy %q", b.Routing.Policy)
	}
	if !validPriorityPolicies[b.Priority.Policy] {
		return fmt.Errorf("unknown priority policy %q", b.Priority.Policy)
	}
	if !validSchedulers[b.Scheduler] {
		return fmt.Errorf("unknown scheduler %q", b.Scheduler)
	}
	return nil
}
```

**Step 4: Run tests**

Run: `go test ./sim/... -run "TestLoadPolicyBundle|TestPolicyBundle_Validate" -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./sim/...`

**Step 6: Commit**

```bash
git add sim/bundle.go sim/bundle_test.go
git commit -m "feat(sim): add PolicyBundle with YAML loading and validation (BC-5, BC-7, EC-1)

- PolicyBundle struct with YAML tags for admission, routing, priority, scheduler
- LoadPolicyBundle reads and parses YAML files
- Validate checks all policy names against known values
- Zero-value fields mean 'use default'

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: CLI Flag --policy-config

**Contracts Implemented:** BC-6, EC-2

**Files:**
- Modify: `cmd/root.go`

**Step 1: Write failing test** (manual verification — CLI integration)

Since cmd/ has no test files, this task is verified by running the CLI:

```bash
# Create test YAML
cat > /tmp/test-policy.yaml << 'EOF'
routing:
  policy: least-loaded
scheduler: sjf
EOF

# Run with --policy-config (should fail until implemented)
go run main.go run --model meta-llama/llama-3.1-8b-instruct --max-prompts 5 --policy-config /tmp/test-policy.yaml
```

**Step 2: Implement**

In `cmd/root.go`:

1. Add variable: `policyConfigPath string`
2. Add flag in `init()`: `runCmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")`
3. In the `Run` function (after workload validation, before building DeploymentConfig), add policy config loading:

```go
// Load policy bundle if specified
if policyConfigPath != "" {
	bundle, err := sim.LoadPolicyBundle(policyConfigPath)
	if err != nil {
		logrus.Fatalf("Failed to load policy config: %v", err)
	}
	if err := bundle.Validate(); err != nil {
		logrus.Fatalf("Invalid policy config: %v", err)
	}

	// Apply bundle values as defaults (CLI flags override)
	if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
		admissionPolicy = bundle.Admission.Policy
	}
	if bundle.Admission.TokenBucketCapacity != 0 && !cmd.Flags().Changed("token-bucket-capacity") {
		tokenBucketCapacity = bundle.Admission.TokenBucketCapacity
	}
	if bundle.Admission.TokenBucketRefillRate != 0 && !cmd.Flags().Changed("token-bucket-refill-rate") {
		tokenBucketRefillRate = bundle.Admission.TokenBucketRefillRate
	}
	if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
		routingPolicy = bundle.Routing.Policy
	}
	if bundle.Routing.CacheWeight != 0 && !cmd.Flags().Changed("routing-cache-weight") {
		routingCacheWeight = bundle.Routing.CacheWeight
	}
	if bundle.Routing.LoadWeight != 0 && !cmd.Flags().Changed("routing-load-weight") {
		routingLoadWeight = bundle.Routing.LoadWeight
	}
	if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
		priorityPolicy = bundle.Priority.Policy
	}
	if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
		scheduler = bundle.Scheduler
	}
}
```

**Step 3: Verify**

```bash
# Test 1: YAML config applies
go run main.go run --model meta-llama/llama-3.1-8b-instruct --max-prompts 5 --policy-config /tmp/test-policy.yaml --log info 2>&1 | head -20

# Test 2: CLI flag overrides YAML
go run main.go run --model meta-llama/llama-3.1-8b-instruct --max-prompts 5 --policy-config /tmp/test-policy.yaml --routing-policy round-robin --log info 2>&1 | head -20

# Test 3: Invalid YAML path
go run main.go run --model meta-llama/llama-3.1-8b-instruct --policy-config /nonexistent.yaml 2>&1 | head -5
# Expected: "Failed to load policy config: ..."
```

**Step 4: Run full test suite**

Run: `go test ./... -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./...`

**Step 6: Commit**

```bash
git add cmd/root.go
git commit -m "feat(cmd): add --policy-config flag for YAML policy configuration (BC-6, EC-2)

- Load PolicyBundle from YAML file when --policy-config is set
- Validate policy names on load
- CLI flags override YAML values (Changed() check)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 7: Update CLAUDE.md and Documentation

**Contracts Implemented:** (documentation)

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

1. In "Current Implementation Focus" → Completed list: add PR8
2. In "Next:" line: update to PR9+
3. In "File Organization": add `sim/router_state.go` and `sim/bundle.go`
4. In "Code Architecture" → Core Simulation Engine: add router_state.go and bundle.go entries
5. In `cmd/root.go` description: add `--policy-config`
6. Update revision subtitle in macro plan (16 PRs, 8 remaining)

**Step 2: Update macro plan**

In `docs/plans/2026-02-11-macro-implementation-plan-v2.md`:
1. Add `✅ COMPLETED` marker to PR8 heading
2. Update B.6 CLI flags: move PR6/PR7 flags to "already added", keep PR9 in "to be added"
3. Fix `sim/policy/` references to `sim/` in PR4-8 sections

**Step 3: Commit**

```bash
git add CLAUDE.md docs/plans/2026-02-11-macro-implementation-plan-v2.md
git commit -m "docs: update CLAUDE.md and macro plan for PR8 completion

- Mark PR8 as completed
- Add router_state.go, bundle.go to file organization
- Add --policy-config to CLI flags documentation
- Fix sim/policy/ references to sim/ (package doesn't exist)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestRouterState_FieldAccess` |
| BC-2 | Task 2 | Unit | `TestAlwaysAdmit_AdmitsAll`, `TestTokenBucket_AdmitAndReject` (updated) |
| BC-3 | Task 3 | Unit | All routing tests in `routing_test.go` (updated) |
| BC-4 | Task 4 | Golden | Existing golden dataset tests |
| BC-5 | Task 5 | Unit | `TestLoadPolicyBundle_ValidYAML`, `TestLoadPolicyBundle_EmptyFields` |
| BC-6 | Task 6 | Manual | CLI verification with YAML + flag override |
| BC-7 | Task 5 | Unit | `TestPolicyBundle_Validate_*` |
| BC-8 | Task 4 | Unit | `TestBuildRouterState_PopulatesSnapshots` |
| BC-9 | Task 3 | Unit | `TestRoutingDecision_PriorityHint_DefaultZero` |
| EC-1 | Task 5 | Unit | `TestLoadPolicyBundle_NonexistentFile`, `TestLoadPolicyBundle_MalformedYAML` |
| EC-2 | Task 6 | Manual | CLI verification with empty fields |
| NC-1 | Task 2-3 | Verify | PriorityPolicy/InstanceScheduler signatures unchanged |
| NC-2 | N/A | Verify | No TenantState fields in RouterState |

**Golden dataset:** No updates needed — RouterState is a transparent wrapper, no behavioral change.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Interface change breaks golden tests | Low | High | Task 4 explicitly verifies golden equivalence | Task 4 |
| yaml.v3 not available as dependency | Low | Medium | Check go.mod; if missing, use `go get gopkg.in/yaml.v3` | Task 5 |
| CLI flag override logic incorrect | Medium | Medium | Manual CLI testing with YAML + flag combinations | Task 6 |
| Test update misses a call site | Low | Medium | Compiler catches all interface mismatches | Task 2-3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (RouterState is minimal: Snapshots + Clock)
- [x] No feature creep beyond PR scope (TenantState/GlobalMetrics deferred)
- [x] No unexercised flags or interfaces (--policy-config exercised, RouterState used by all policies)
- [x] No partial implementations (all policy implementations updated)
- [x] No breaking changes without explicit contract updates (BC-2, BC-3 document the interface changes)
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from `sim/internal/testutil` (golden tests)
- [x] CLAUDE.md updated (Task 7)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7)
- [x] All contracts mapped to tasks

---

## Appendix: File-Level Implementation Details

### File: `sim/router_state.go`

**Purpose:** Bridge type providing cluster-wide state to policy interfaces.

```go
package sim

// RouterState provides cluster-wide state to policy interfaces.
// Built by ClusterSimulator before each policy invocation.
// This is a bridge type in sim/ (not sim/cluster/) to avoid import cycles —
// same pattern as RoutingSnapshot.
//
// USAGE BOUNDARY: Only constructed by ClusterSimulator's event handlers.
// Single-instance Simulator does not use RouterState — instance-level policies
// (PriorityPolicy, InstanceScheduler) receive parameters directly.
type RouterState struct {
	Snapshots []RoutingSnapshot // One per instance, same order as ClusterSimulator.instances
	Clock     int64             // Current simulation clock in microseconds
	// TODO(PR9): Add TenantState for SLO-class-aware admission/routing
	// TODO(PR9): Add GlobalMetrics for adaptive policies
}
```

**Key notes:**
- Value type, no pointer receiver methods needed
- No TenantState/GlobalMetrics fields yet (deferred to PR9+, additive under interface freeze)
- Constructed fresh for each arrival pipeline via `buildRouterState()` helper

### File: `sim/bundle.go`

**Purpose:** YAML-based policy configuration with validation.

See Task 5 Step 3 for complete implementation.

**Key notes:**
- Uses `gopkg.in/yaml.v3` for parsing
- Zero-value fields mean "use default" (don't override)
- Validation checks against hardcoded known-policy maps
- Error return (not panic) for I/O and parse errors

### File: `sim/admission.go` (modified)

**Key change:** `Admit(req *Request, clock int64)` → `Admit(req *Request, state *RouterState)`

All implementations extract `state.Clock` where they previously used `clock` parameter.

### File: `sim/routing.go` (modified)

**Key changes:**
1. `Route(req *Request, snapshots []RoutingSnapshot, clock int64)` → `Route(req *Request, state *RouterState)`
2. `RoutingDecision` gets new `Priority float64` field (zero = defer to instance-level PriorityPolicy)
3. All implementations extract `state.Snapshots` and `state.Clock`
4. `PrefixAffinity.Route` passes received `state` through to `LeastLoaded.Route` (not reconstructed)
5. Transitional comment on line 26 removed

### File: `sim/priority.go` (modified)

**Key change:** Comment on line 26 updated from "planned for PR8" to "planned for PR9+".

### File: `sim/cluster/cluster_event.go` (modified)

**Key changes:**
1. New `buildRouterState(cs)` helper — collects snapshots + clock into `*sim.RouterState`, shared by admission and routing
2. `AdmissionDecisionEvent.Execute` calls `buildRouterState(cs)` → full RouterState with snapshots (BC-8)
3. `RoutingDecisionEvent.Execute` calls `buildRouterState(cs)` → passes to routing policy, then applies `decision.Priority` if non-zero (BC-9)

### File: `cmd/root.go` (modified)

**Key change:** New `--policy-config` flag. When set, loads YAML and applies as defaults. CLI flags override via Cobra's `cmd.Flags().Changed()` check.
