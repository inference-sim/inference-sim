# Per-Tier Shedding in ProgressSnapshot — Implementation Plan

**Goal:** Expose per-SLO-class shedding counts in `ProgressSnapshot` so live progress consumers can observe shedding rates per tier during simulation, not just at the end.

**Source:** https://github.com/inference-sim/inference-sim/issues/1309

**Closes:** `Fixes #1309`

## Context

Today `shedByTier map[string]int` on `ClusterSimulator` accumulates all shedding events (admission rejections + gateway queue shed victims + in-flight evictions) grouped by SLO class. It's reported once at simulation end via `cs.ShedByTier()` in `cmd/root.go` and `cmd/replay.go`.

`ProgressSnapshot` is BLIS's equivalent of Prometheus scraping — periodic emission of metrics during simulation. It already has aggregate counters (`GatewayQueueShed`, `RejectedRequests`) but lacks the per-tier breakdown.

This PR adds `ShedByTier map[string]int` to `ProgressSnapshot`, populated in `ClusterSimulator.maybeDeliverProgressSnapshot()`. The map is a snapshot copy of the accumulator at that point in time (cumulative, not delta).

## Behavioral Contracts

**BC-1: ShedByTier appears in progress snapshots when shedding occurs**
- GIVEN a simulation with tier-shed admission and sheddable requests that get rejected
- WHEN a progress snapshot is emitted after rejection events
- THEN `snapshot.ShedByTier` contains non-zero counts for the shed tiers, matching `cs.ShedByTier()` at that point in time

**BC-2: ShedByTier is nil when no shedding has occurred**
- GIVEN a simulation with always-admit policy (no rejections)
- WHEN progress snapshots are emitted
- THEN `snapshot.ShedByTier` is nil (zero-value map — no allocation for empty case)

**BC-3: Final snapshot ShedByTier matches post-simulation ShedByTier()**
- GIVEN any simulation that completes
- WHEN the final progress snapshot is emitted (IsFinal=true)
- THEN `snapshot.ShedByTier` equals `cs.ShedByTier()` (same counts for all tiers)

**BC-4: Determinism preserved (INV-6)**
- GIVEN two runs with identical seed and config
- WHEN one has a progress hook and one does not
- THEN both produce identical `ShedByTier()` at simulation end (hook doesn't affect simulation state)

## Tasks

### Task 1: Add ShedByTier field to ProgressSnapshot

**Test:** Verify the struct compiles with the new field and zero-value is nil map.

**Implementation:**
- File: `sim/progress_hook.go`
- Add `ShedByTier map[string]int` field to `ProgressSnapshot` struct in the cluster-mode section (after `GatewayQueueShed`)

### Task 2: Populate ShedByTier in maybeDeliverProgressSnapshot

**Test:** Write test `TestClusterSimulator_ProgressHook_ShedByTier` that sets up tier-shed admission, generates sheddable requests that get rejected, verifies final snapshot contains per-tier counts matching `cs.ShedByTier()` (BC-1, BC-3).

**Implementation:**
- File: `sim/cluster/cluster.go` in `maybeDeliverProgressSnapshot()`
- After building the `snap` struct literal, add:
  ```go
  if len(c.shedByTier) > 0 {
      snap.ShedByTier = make(map[string]int, len(c.shedByTier))
      for k, v := range c.shedByTier {
          snap.ShedByTier[k] = v
      }
  }
  ```
- This ensures: (a) nil when no shedding has occurred (BC-2), (b) defensive copy not reference (hook consumers cannot mutate internal state, preserving INV-6), (c) no allocation in non-shedding simulations
- Also update ProgressSnapshot doc comment in `sim/progress_hook.go` to mention `ShedByTier` as a map field (same treatment as `InstanceSnapshots` slice)

### Task 3: Test nil case — no shedding produces nil ShedByTier (BC-2)

**Test:** Write test `TestClusterSimulator_ProgressHook_ShedByTierNilWhenNoShedding` that runs always-admit config, verifies all snapshots have `ShedByTier == nil`.

### Task 4: Test determinism — hook presence doesn't affect ShedByTier counts (BC-4)

**Test:** Write test `TestClusterSimulator_ProgressHook_ShedByTierDeterminism` — run with and without hook, compare `cs.ShedByTier()` output. (Extends existing determinism test pattern.)

## Sanity Checklist

- [ ] No new interfaces or types — just a map field on existing struct
- [ ] No new CLI flags
- [ ] INV-6 (determinism) preserved — snapshot is read-only copy, no simulation state mutation
- [ ] INV-1 unaffected — we're reading the counter, not modifying request lifecycle
- [ ] Map copy prevents mutation by hook consumers (ProgressSnapshot doc says "read-only")
- [ ] Nil when empty avoids unnecessary allocation in non-shedding simulations
