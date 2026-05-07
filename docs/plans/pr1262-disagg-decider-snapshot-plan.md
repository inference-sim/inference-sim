# Thread Decode Snapshot Into DisaggregationDecider.Decide (GAP-2) — Implementation Plan

**Goal:** Pass the already-selected decode pod's `RoutingSnapshot` into `DisaggregationDecider.Decide`, so deciders can inspect per-pod state (cache presence, load) when choosing whether to disaggregate — mirroring llm-d's decider API.
**Source:** https://github.com/inference-sim/inference-sim/issues/1262 (parent tracking: #1260)
**Closes:** Fixes #1262

## Behavioral Contracts

**BC-1: Interface carries decode snapshot**
- GIVEN a `DisaggregationDecider` implementation
- WHEN the cluster calls `Decide(req, snap)`
- THEN the implementation receives both the request and the decode pod's `RoutingSnapshot` (or a zero-value snapshot in edge cases)

**BC-2: Snapshot-agnostic deciders are decision-invariant**
- GIVEN `NeverDisaggregate` or `AlwaysDisaggregate`
- WHEN `Decide(req, snap)` is called with any snapshot value (including zero value)
- THEN the returned `DisaggregationDecision` is identical to the pre-change behavior (never and always, respectively)

**BC-3: Cluster passes the decode pod's snapshot, not an arbitrary one**
- GIVEN a cluster where decode routing has selected `TargetInstance = X`
- WHEN `ClusterSimulator.executeDisaggregatedRouting` invokes the decider
- THEN the snapshot passed to `Decide` has `ID == X` (decode target, which may be a dedicated decode pod or a shared-role pod)

**BC-4: PrefixThresholdDecider decision preserved**
- GIVEN a `PrefixThresholdDecider` with existing threshold/block-size configuration
- WHEN `Decide(req, snap)` is called
- THEN the returned decision matches pre-change behavior for all existing tests (snapshot argument is accepted but ignored in this PR; GAP-3 will wire it in)

**BC-5: INV-9 oracle boundary preserved**
- GIVEN any decider implementation under this PR
- WHEN `Decide(req, snap)` runs
- THEN no implementation reads `req.OutputTokens` (snapshot threading does not introduce OutputTokens access)

## Tasks

### Task 1: Widen `DisaggregationDecider.Decide` interface + update all implementations (BC-1, BC-2, BC-4, BC-5)

**Files:** modify `sim/disaggregation.go`, `sim/disaggregation_test.go`

**Test (add to `sim/disaggregation_test.go`):**

```go
// TestDisaggregate_SnapshotPassthrough verifies BC-1/BC-2: all built-in
// deciders accept a RoutingSnapshot argument and produce identical decisions
// when called with a zero-value snapshot vs an arbitrary populated snapshot
// (snapshot-agnostic implementations must remain unaffected by snap content).
func TestDisaggregate_SnapshotPassthrough(t *testing.T) {
    req := &Request{ID: "req-1", InputTokens: make([]int, 100)}
    zero := RoutingSnapshot{}
    populated := RoutingSnapshot{ID: "decode_0", QueueDepth: 7, KVUtilization: 0.4}

    deciders := []DisaggregationDecider{
        &NeverDisaggregate{},
        &AlwaysDisaggregate{},
        NewPrefixThresholdDecider(512, 16),
    }
    for _, d := range deciders {
        gotZero := d.Decide(req, zero)
        gotPop := d.Decide(req, populated)
        if gotZero != gotPop {
            t.Errorf("%T: decision differs between zero and populated snapshot "+
                "(zero=%v, populated=%v); snapshot-agnostic deciders must be invariant",
                d, gotZero, gotPop)
        }
    }
}
```

Update all existing tests in `sim/disaggregation_test.go` that call `.Decide(req)` to call `.Decide(req, RoutingSnapshot{})`.

**Impl (in `sim/disaggregation.go`):**

```go
type DisaggregationDecider interface {
    Decide(req *Request, snap RoutingSnapshot) DisaggregationDecision
}

func (n *NeverDisaggregate) Decide(_ *Request, _ RoutingSnapshot) DisaggregationDecision {
    return DisaggregationDecision{Disaggregate: false}
}

func (a *AlwaysDisaggregate) Decide(_ *Request, _ RoutingSnapshot) DisaggregationDecision {
    return DisaggregationDecision{Disaggregate: true}
}

func (p *PrefixThresholdDecider) Decide(req *Request, _ RoutingSnapshot) DisaggregationDecision {
    // body unchanged — GAP-3 will consume snap.
    // ... (existing body)
}
```

Also update the interface docstring to document the snapshot argument and the zero-value contract.

**Verify:** `go test ./sim/ -run TestDisagg -v`
**Lint:** `golangci-lint run ./sim/...`

### Task 2: Pass decode snapshot at the call site (BC-3)

**Files:** modify `sim/cluster/cluster.go`

> **Note:** Originally targeted `sim/cluster/cluster_event.go` (`DisaggregationDecisionEvent.Execute`). GAP-1 (#1266) moved the disaggregation decision into `ClusterSimulator.executeDisaggregatedRouting` in `sim/cluster/cluster.go` — the snapshot-threading change was reapplied at the new call site on merge. GAP-5 (#1278) later extended `FilterSnapshotsByPool` to use `.Has(role)` set-membership so shared-role pods appear in the decode-filtered list; the ID-based snapshot lookup is unaffected.

**Test:** covered indirectly by existing cluster-level disaggregation tests (`TestPrefixThreshold_*`, `TestDisaggregation_*`) which exercise the full pipeline and will fail to compile without this change. No new test required — passing a zero-value snapshot would be invisible to behavior; the existing tests verify end-to-end wiring.

**Impl (in `sim/cluster/cluster.go`, `ClusterSimulator.executeDisaggregatedRouting`):**

After the line that computes `decodeDecision := policy.Route(...)`, find the snapshot whose `ID` equals `decodeDecision.TargetInstance` in `filteredSnapshots`. Pass it into `Decide`:

```go
var decodeSnap sim.RoutingSnapshot
for _, s := range filteredSnapshots {
    if s.ID == decodeDecision.TargetInstance {
        decodeSnap = s
        break
    }
}
// If no match (should not occur given R6 contract on Route), decodeSnap is zero-value.
disaggDecision := cs.disaggregationDecider.Decide(e.request, decodeSnap)
```

**Verify:** `go test ./... -count=1`
**Lint:** `golangci-lint run ./...`

### Task 3: Commit and push

**Commit:** `refactor(sim): thread decode snapshot into DisaggregationDecider.Decide (BC-1..BC-5)`

## Sanity Checklist

- [x] R1 (no silent continue): no error paths touched
- [x] R4 (canonical construction): no struct construction changes
- [x] R6 (routing target validity): call site falls back to zero-value snapshot when the snapshot ID is not found, avoiding a nil/panic path; matches the issue's "zero-value allowed" contract
- [x] R13 (single-module interface): the interface remains single-module (cluster → decider); the snapshot type is already part of the `sim` package
- [x] INV-9: no implementation reads `req.OutputTokens`
- [x] INV-1/4/5/6/7/8/10/11/12: unaffected (no event ordering, conservation, or causality changes)
- [x] CLAUDE.md "Recent Changes": not updated by this PR (the CLAUDE.md working copy mirrors the canonical source — this PR does not modify canonical standards; an entry may be added at merge time alongside GAP-3/GAP-1/GAP-4 for the overall PD parity series, but is not required for a single GAP-2 refactor)
