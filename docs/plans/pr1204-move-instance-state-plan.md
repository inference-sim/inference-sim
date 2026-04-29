# Move InstanceState to sim Package — Implementation Plan

**Goal:** Move `InstanceState` type and constants from `sim/cluster/` to `sim/` so that `InstanceSnapshot.State` can be typed as `InstanceState` instead of `string`, enabling compile-time safety for ProgressHook consumers.
**Source:** [Issue #1204](https://github.com/inference-sim/inference-sim/issues/1204)
**Closes:** `Fixes #1204`

**Tier:** Medium by file count (7 production files), but purely mechanical renames — no design decisions, no behavioral changes. Using compact format.

## Behavioral Contracts

BC-1: Type availability at sim package level
- GIVEN a consumer imports `github.com/inference-sim/inference-sim/sim`
- WHEN they reference `sim.InstanceState`, `sim.InstanceStateActive`, etc.
- THEN the compiler resolves the type and all 6 constants without importing `sim/cluster`

BC-2: InstanceSnapshot.State typed as InstanceState
- GIVEN a ProgressHook implementation receives an InstanceSnapshot
- WHEN it reads the State field
- THEN the field type is `sim.InstanceState` (not `string`), enabling typed switch/comparison

BC-3: Zero behavioral regression
- GIVEN identical simulation inputs
- WHEN running before and after this refactor
- THEN simulation output is byte-identical (INV-6 preserved)

BC-4: No circular imports
- GIVEN the package dependency graph
- WHEN `sim/cluster/` uses `sim.InstanceState` constants
- THEN compilation succeeds (sim/cluster already imports sim/)

## Tasks

### Task 1: Create sim/instance_state.go (BC-1, BC-4)

**Files:** create `sim/instance_state.go`, test `sim/instance_state_test.go`

**Test:**
```go
// sim/instance_state_test.go
package sim

import "testing"

func TestInstanceState_Constants(t *testing.T) {
	states := []InstanceState{
		InstanceStateScheduling,
		InstanceStateLoading,
		InstanceStateWarmingUp,
		InstanceStateActive,
		InstanceStateDraining,
		InstanceStateTerminated,
	}
	for _, s := range states {
		if s == "" {
			t.Errorf("InstanceState constant is empty")
		}
		if !IsValidInstanceState(string(s)) {
			t.Errorf("IsValidInstanceState(%q) = false, want true", s)
		}
	}
	if IsValidInstanceState("bogus") {
		t.Error("IsValidInstanceState(bogus) = true, want false")
	}
}
```

**Impl:**
Move type declaration, 6 constants, `validInstanceStates` map, and `IsValidInstanceState` function from `sim/cluster/infra_node.go` to new `sim/instance_state.go`.

**Verify:** `go test ./sim/... -run TestInstanceState_Constants`
**Lint:** `golangci-lint run ./sim/...`
**Commit:** `refactor(sim): move InstanceState type to sim package (BC-1)`

### Task 2: Update sim/cluster references (BC-3, BC-4)

**Files:** modify `sim/cluster/infra_node.go`, `sim/cluster/instance.go`, `sim/cluster/cluster.go`, `sim/cluster/direct_actuator.go`, `sim/cluster/infra_lifecycle_event.go`

**Impl:**
- Remove `InstanceState` type, constants, validation map, and `IsValidInstanceState` from `infra_node.go`
- Replace all bare `InstanceState*` references in cluster package with `sim.InstanceState*`
- Replace `InstanceState` type references with `sim.InstanceState`
- Update `validInstanceTransitions` map type annotations
- Update `TransitionTo` parameter type

**Verify:** `go build ./... && go test ./sim/cluster/... -count=1`
**Lint:** `golangci-lint run ./sim/cluster/...`
**Commit:** `refactor(sim/cluster): use sim.InstanceState from parent package (BC-3, BC-4)`

### Task 3: Update InstanceSnapshot.State type (BC-2)

**Files:** modify `sim/progress_hook.go`, modify `sim/cluster/cluster.go` (cluster construction site), modify `sim/simulator.go` (single-instance construction site)

**Impl:**
- Change `InstanceSnapshot.State` from `string` to `InstanceState`
- In `sim/simulator.go:326`: change `State: "Active"` to `State: InstanceStateActive`
- In `sim/cluster/cluster.go:812`: change `State: string(inst.State)` to `State: inst.State` (no cast needed — type is now `sim.InstanceState`)
- Update comment on State field (remove "matches string constants" wording)

**Verify:** `go build ./... && go test ./... -count=1`
**Lint:** `golangci-lint run ./...`
**Commit:** `refactor(sim): type InstanceSnapshot.State as InstanceState (BC-2)`

## Sanity Checklist

- [ ] No circular imports (sim/cluster → sim is existing direction)
- [ ] All 6 constants preserve exact string values (byte-identical output)
- [ ] `validInstanceTransitions` map types updated
- [ ] `TransitionTo` method signature updated
- [ ] `IsValidInstanceState` moved (not duplicated)
- [ ] Test files in sim/cluster still compile (they use unqualified names within package)
- [ ] No stale references remain (grep confirms zero hits for old location)
