# Widen DisaggregationDecider.Decide Interface (GAP-2) — Implementation Plan

**Goal:** Widen `DisaggregationDecider.Decide` to receive the decode-pool `*RouterState` (not a single snapshot), and expand `DisaggregationDecision` with optional `DecodePodOverride` / `PrefillPodHint` fields so future joint D+P policies (e.g., GAP-3, GAP-4) can fully reconsider pod selection without another interface break. Built-in deciders (`NeverDisaggregate`, `AlwaysDisaggregate`, `PrefixThresholdDecider`) ignore `state` and leave overrides empty — parity with llm-d today, extensibility for tomorrow.
**Source:** https://github.com/inference-sim/inference-sim/issues/1262 (scope expanded per [#1262 comment 4399184134](https://github.com/inference-sim/inference-sim/issues/1262#issuecomment-4399184134); parent tracking: #1260)
**Closes:** Fixes #1262

## Behavioral Contracts

**BC-1: Interface carries the decode-pool RouterState**
- GIVEN a `DisaggregationDecider` implementation
- WHEN the cluster calls `Decide(req, state)`
- THEN the implementation receives the request and a non-nil `*RouterState` whose `Snapshots` contain every routable decode-pool instance

**BC-2: State-agnostic deciders are decision-invariant**
- GIVEN `NeverDisaggregate`, `AlwaysDisaggregate`, or `PrefixThresholdDecider`
- WHEN `Decide(req, state)` is called with any `*RouterState` value (including nil)
- THEN the returned `DisaggregationDecision` is identical to the pre-change behavior — and `DecodePodOverride` / `PrefillPodHint` are the empty string (built-in deciders never override)

**BC-3: Cluster passes the decode pool's RouterState**
- GIVEN a cluster with decode-pool instances D₁…Dₙ (either dedicated-decode or shared-role pods)
- WHEN `ClusterSimulator.executeDisaggregatedRouting` invokes the decider
- THEN the `state.Snapshots` passed to `Decide` contains exactly the ID set `{D₁…Dₙ}` (verified by `TestDisaggregation_DeciderReceivesDecodePoolState`)

**BC-4: DecodePodOverride retargets the decode pod**
- GIVEN a decider that returns `DisaggregationDecision{Disaggregate: false, DecodePodOverride: "decode_X"}`
- WHEN `executeDisaggregatedRouting` processes the decision
- THEN the request is injected into instance `decode_X` regardless of what the decode routing policy selected (verified by `TestDisaggregation_DecodePodOverrideReroutes`)

**BC-5: INV-9 oracle boundary preserved**
- GIVEN any decider implementation under this PR
- WHEN `Decide(req, state)` runs
- THEN no implementation reads `req.OutputTokens` (state threading does not introduce OutputTokens access)

## Tasks

### Task 1: Widen the interface + decision struct (`sim/disaggregation.go`)

**Files:** modify `sim/disaggregation.go`, `sim/disaggregation_test.go`

**Impl — `DisaggregationDecision` expanded:**

```go
type DisaggregationDecision struct {
    Disaggregate      bool
    DecodePodOverride string // empty = keep pre-selected decode pod
    PrefillPodHint    string // empty = normal prefill routing (GAP-4 will use this)
}
```

**Impl — Interface signature takes `*RouterState`:**

```go
type DisaggregationDecider interface {
    Decide(req *Request, state *RouterState) DisaggregationDecision
}
```

All three built-in implementations (`NeverDisaggregate`, `AlwaysDisaggregate`, `PrefixThresholdDecider`) ignore `state` and return zero-valued overrides. The `PrefixThresholdDecider` body is unchanged; issue #1263 (GAP-3) will replace the cluster-wide cache estimate with per-pod queries from `state.Snapshots`.

**Tests:**
- Rename `TestDisaggregationDecider_SnapshotAgnostic` → `TestDisaggregationDecider_StateAgnostic` (BC-2) — verifies nil vs populated `*RouterState` produces identical decisions across all built-ins.
- Add `TestDisaggregationDecider_BuiltinsReturnNoOverrides` — pins the invariant that every built-in leaves `DecodePodOverride` and `PrefillPodHint` empty (regression guard for future edits).
- Update every existing `.Decide(req)` / `.Decide(req, snap)` call site to `.Decide(req, (*RouterState)(nil))` for mechanical tests that do not exercise state.

**Verify:** `go test ./sim/ -run TestDisagg -v`
**Lint:** `golangci-lint run ./sim/...`

### Task 2: Thread state + apply override at the call site (BC-1, BC-3, BC-4)

**Files:** modify `sim/cluster/cluster.go`

**Impl (in `ClusterSimulator.executeDisaggregatedRouting`):**

```go
// state already exists (built at the top of the function to pick the decode pod).
disaggDecision := cs.disaggregationDecider.Decide(req, state)

// Apply optional retargeting from joint D+P policies.
if disaggDecision.DecodePodOverride != "" {
    decodeDecision.TargetInstance = disaggDecision.DecodePodOverride
}
```

The override must be a member of the decode-pool snapshot set; the downstream `decodeInst == nil` guard panics on a bad override, which is the correct contract violation for a test-only / future policy bug.

**Verify:** `go test ./sim/cluster/ -run TestDisaggregation -v && go test ./... -count=1`

### Task 3: Cluster-level behavior tests (BC-3, BC-4)

**Files:** modify `sim/cluster/disaggregation_test.go`

**Test 1 — `TestDisaggregation_DeciderReceivesDecodePoolState` (BC-3):** install a `recordingDecider` that captures every `(req, state)` invocation, run a 4-instance cluster (2 prefill + 2 decode), assert `state.Snapshots` contains exactly the decode-pool IDs on every call.

**Test 2 — `TestDisaggregation_DecodePodOverrideReroutes` (BC-4):** install an `overrideDecider` returning `{Disaggregate: false, DecodePodOverride: <specific decode pod>}`, run a 4-instance cluster, assert every routed request lands on the override target (not the round-robin pick).

### Task 4: Commit and push

**Commit:** `feat(sim): widen DisaggregationDecider — pass *RouterState + decision overrides (BC-1..BC-5)`

## Sanity Checklist

- [x] R1 (no silent continue): no error paths touched
- [x] R4 (canonical construction): `DisaggregationDecision` gains fields at a single site; zero-value defaults keep all callers forward-compatible
- [x] R6 (no `logrus.Fatalf` in sim/): no change
- [x] R13 (single-module interface): interface remains single-module (cluster → decider); `*RouterState` is already part of the `sim` package
- [x] INV-1/4/5/6/7/8/10/11/12: unaffected (no event ordering, conservation, or causality changes)
- [x] INV-9: no implementation reads `req.OutputTokens`; `state` carries snapshots and clock only
- [x] CLAUDE.md: no canonical-source edits; Change History entry may be added at merge time alongside GAP-3/GAP-4, not required for this PR
