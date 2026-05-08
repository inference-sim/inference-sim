# Rewrite PrefixThresholdDecider to Use Per-Pod KV Cache State (GAP-3) — Implementation Plan

**Goal:** Replace `PrefixThresholdDecider`'s cluster-wide LRU approximation with a per-pod KV cache query against the *pre-selected decode pod*, matching llm-d's `PrefixBasedPDDecider.disaggregate` (queries `endpoint.Get(PrefixCacheMatchInfoKey)`). Thread `cacheQueryFn` (already wired for the `precise-prefix-cache` scorer since PR #883) through the decider constructor, and add `RouterState.SelectedInstance` so the decider can identify which snapshot was pre-selected. Delete the now-redundant `DisaggregationObserver` / `ObserveRouting` machinery.

**Source:** https://github.com/inference-sim/inference-sim/issues/1263 (parent tracker: #1260; prereq: #1262 merged as PR #1265). Addresses @mtoslalibu's GAP-3 note on PR #1265 — chose "option (a) receive the selected pod ID alongside state" over "option (b) scan all snapshots" for direct llm-d parity; joint-D+P policies remain viable via `DisaggregationDecision.DecodePodOverride` (added in GAP-2) against the widened `state.Snapshots`.

**Closes:** Fixes #1263

## Scope & Non-Goals

**In scope:**
- `sim/disaggregation.go`: `PrefixThresholdDecider` body rewrite + constructor signature + deletion of observer interface/scratchpad.
- `sim/router_state.go`: additive `SelectedInstance` field.
- `sim/cluster/cluster.go`: pass `cs.cacheQueryFn` into `NewPrefixThresholdDecider`; populate `state.SelectedInstance` before `Decide`; remove `notifyDisaggregationObserver` helper.
- `sim/cluster/pd_events.go`, `sim/cluster/disaggregation_test.go`: remove observer call sites; rewrite one behavioral test that asserted observer wiring.
- `sim/disaggregation_test.go`: replace LRU-warming tests with per-pod cache-query tests using a stub `cacheQueryFn`.

**Out of scope:**
- Joint D+P policies that exploit `DecodePodOverride` based on cache affinity (option B from the review comment) — future work.
- Threshold-zero short-circuit (llm-d behavior #1250 fix 3) — `threshold=0` already means "disaggregate any non-empty request," which is the more natural semantics for the BLIS flag; not re-semanticing here.
- `RequestView` type for structural INV-9 enforcement (#1250 fix 5) — separate track.

## Behavioral Contracts

**BC-1: Decider queries the pre-selected pod's cache**
- GIVEN a `PrefixThresholdDecider` constructed with `cacheQuery` map
- WHEN the cluster calls `Decide(req, state)` with `state.SelectedInstance = "decode_X"`
- THEN the decider invokes `cacheQuery["decode_X"](req.InputTokens)` (and no other closure in the map) to obtain the cached block count used in the formula `nonCachedTokens = len(InputTokens) − cachedBlocks × blockSize`.

**BC-2: Decision formula preserved**
- GIVEN `cachedBlocks` returned by the per-pod query
- WHEN `Decide` computes `nonCachedTokens = len(InputTokens) − cachedBlocks × blockSize`
- THEN `Disaggregate = (nonCachedTokens > threshold)` — identical threshold semantics to pre-GAP-3 behavior (strict `>`, not `>=`).

**BC-3: Empty-input and missing-selection short-circuits**
- GIVEN `len(req.InputTokens) == 0`, OR `state == nil`, OR `state.SelectedInstance == ""`, OR `state.SelectedInstance` is not a key in the `cacheQuery` map
- WHEN `Decide` runs
- THEN the decision is `DisaggregationDecision{Disaggregate: false}` (conservative: if we can't locate the selected pod's cache, assume we don't need to disaggregate — matches llm-d's nil-endpoint guard at `prefix_based_pd_decider.go:108-111`).

**BC-4: Per-pod isolation — cache hits on pod A do not influence a decision targeting pod B**
- GIVEN two decode pods A and B where A's KV store contains the request's prefix blocks and B's does not
- WHEN `Decide` is called with `state.SelectedInstance = "B"`
- THEN `Disaggregate` reflects B's cold cache (i.e. `nonCachedTokens = len(InputTokens)`) and returns `true` when `len(InputTokens) > threshold`.

**BC-5: Cluster populates `state.SelectedInstance` with the decode routing policy's pick**
- GIVEN `executeDisaggregatedRouting` in `cluster.go`
- WHEN it calls `cs.disaggregationDecider.Decide(req, state)`
- THEN `state.SelectedInstance == decodeDecision.TargetInstance` at the moment of the call. (This is the observable wiring contract; `DecodePodOverride` may retarget AFTER the decider returns, but the decider itself sees the routing policy's pick.)

**BC-6: Observer machinery fully removed, no call sites remain**
- GIVEN the post-PR tree
- WHEN grep searches for `DisaggregationObserver`, `ObserveRouting`, or `notifyDisaggregationObserver`
- THEN no matches remain in production source (`sim/**/*.go` excluding `_test.go`).

**BC-7: INV-9 oracle boundary preserved**
- No implementation reads `req.OutputTokens`. The decider consumes only `req.InputTokens`, `state.SelectedInstance`, and the cache query map.

## Cross-path parity (run / replay / observe)

All changes live in `sim/` and `sim/cluster/` (the DES kernel and cluster layer). Applies uniformly to `blis run` and `blis replay`. `blis observe` dispatches to a real server and never invokes a `DisaggregationDecider` — out of scope.

## Tasks

### Task 1: Add `RouterState.SelectedInstance` field (BC-5 precondition)

**Files:** modify `sim/router_state.go`.

**Impl:**

```go
type RouterState struct {
    Snapshots        []RoutingSnapshot
    LoadingSnapshots []RoutingSnapshot
    Clock            int64
    // SelectedInstance is the instance ID pre-selected by the caller (typically
    // the decode-routing policy) before invoking a DisaggregationDecider.
    // Empty for contexts where no prior selection has been made (e.g., when
    // RoutingPolicy.Route itself builds the state). DisaggregationDecider
    // implementations may read this to identify which snapshot in Snapshots
    // was chosen upstream; a zero value must be tolerated (treat as "unknown").
    SelectedInstance string
}
```

**Test — `sim/disaggregation_test.go`:**
- Update existing `TestDisaggregationDecider_StateAgnostic` — the state-agnostic contract applies to `NeverDisaggregate` and `AlwaysDisaggregate` only after GAP-3. `PrefixThresholdDecider` is no longer state-agnostic. Split into two sub-tests: one for the two ignorant deciders (unchanged behavior), one for `PrefixThresholdDecider` with an explicit stub `cacheQuery` returning `0` (zero blocks → cold → decision depends on threshold).

**Verify:** `go build ./...` (compiles).

**Lint:** `golangci-lint run ./sim/...`.

### Task 2: Rewrite `PrefixThresholdDecider` (BC-1..BC-4, BC-7)

**Files:** modify `sim/disaggregation.go`, `sim/disaggregation_test.go`.

**Impl — struct and constructor:**

```go
type PrefixThresholdDecider struct {
    threshold  int
    blockSize  int
    cacheQuery map[string]func([]int) int // per-pod cache lookup (same map as precise-prefix-cache scorer)
}

// NewPrefixThresholdDecider creates a PrefixThresholdDecider.
// threshold must be >= 0, blockSize must be > 0.
// cacheQuery may be nil (all requests treated as cold); in production callers
// pass the cluster's cacheQueryFn map built from CachedSnapshotProvider.
func NewPrefixThresholdDecider(threshold, blockSize int, cacheQuery map[string]func([]int) int) *PrefixThresholdDecider {
    if threshold < 0 {
        panic(fmt.Sprintf("NewPrefixThresholdDecider: threshold must be >= 0, got %d", threshold))
    }
    if blockSize <= 0 {
        panic(fmt.Sprintf("NewPrefixThresholdDecider: blockSize must be > 0, got %d", blockSize))
    }
    return &PrefixThresholdDecider{threshold: threshold, blockSize: blockSize, cacheQuery: cacheQuery}
}
```

**Impl — `Decide` body:**

```go
func (p *PrefixThresholdDecider) Decide(req *Request, state *RouterState) DisaggregationDecision {
    if len(req.InputTokens) == 0 {
        return DisaggregationDecision{Disaggregate: false}
    }
    if state == nil || state.SelectedInstance == "" || p.cacheQuery == nil {
        return DisaggregationDecision{Disaggregate: false}
    }
    fn, ok := p.cacheQuery[state.SelectedInstance]
    if !ok || fn == nil {
        return DisaggregationDecision{Disaggregate: false}
    }
    cachedBlocks := fn(req.InputTokens)
    nonCachedTokens := len(req.InputTokens) - cachedBlocks*p.blockSize
    return DisaggregationDecision{Disaggregate: nonCachedTokens > p.threshold}
}
```

**Impl — deletions (in `sim/disaggregation.go`):**
- `globalVirtualInstance` constant (L92).
- `defaultDisaggLRUCapacity` constant (L96).
- `DisaggregationObserver` interface (L98-113) including its doc comment.
- Fields `idx`, `cachedHashes`, `cachedReqID` on `PrefixThresholdDecider`.
- `ObserveRouting` method (L174-194).
- Compile-time assertion `var _ DisaggregationObserver = (*PrefixThresholdDecider)(nil)` (L201).
- Doc comment line referencing the observer in the `DisaggregationDecider` interface docs (L39-40): "Stateful implementations may additionally implement DisaggregationObserver…".
- Doc comment line in `PrefixThresholdDecider` struct doc about the scratchpad and observer.

**Impl — `sim/disaggregation_test.go` test updates:**
- Delete `noopDisaggregationObserver` (L239-244).
- Delete `TestPrefixThresholdDecider_Interface` as currently written (L246-252) — rewrite to assert `DisaggregationDecider` only (R13 no longer applies because we're removing the observer interface entirely).
- Delete `TestPrefixThresholdDecider_CacheAware` and `TestPrefixThresholdDecider_CacheAware_SubRequestIDMismatch` (L329-413) — these tested the LRU-warming-via-ObserveRouting path which no longer exists.
- Update `TestNewPrefixThresholdDecider_PanicsOnNegativeThreshold` / `_PanicsOnZeroBlockSize` to pass `nil` as the third constructor argument.
- Update `TestDisaggregationDecider_INV9_OracleBoundary` to construct the decider with `nil` cacheQuery.
- Add `TestPrefixThresholdDecider_PerPodCacheQuery` (BC-1, BC-4): stub `cacheQuery := map[string]func([]int) int{"decode_A": func(t []int) int { return 40 }, "decode_B": func(t []int) int { return 0 }}` (A: 40×16=640 cached tokens; B: cold). Construct `NewPrefixThresholdDecider(512, 16, cacheQuery)`. Call `Decide(req-840-tokens, &RouterState{SelectedInstance: "decode_A"})` → expect `Disaggregate=false` (840−640=200 ≤ 512). Call with `SelectedInstance="decode_B"` → expect `Disaggregate=true` (840−0=840 > 512).
- Add `TestPrefixThresholdDecider_MissingSelection_ReturnsFalse` (BC-3): nil state, empty `SelectedInstance`, unknown `SelectedInstance`, and nil closure in the map → all return `Disaggregate=false`.
- Update `TestPrefixThresholdDecider_EmptyTokens`, `_AboveThreshold`, `_BelowOrAtThreshold`, `_ZeroThreshold` to pass a stub `cacheQuery` (or `nil`) — with nil or missing-selection they hit the BC-3 short-circuit. Rework these tests to use a stub returning `0` for the selected pod so the decision path actually exercises the formula with cachedBlocks=0.
- Keep `TestDisaggregationDecider_StateAgnostic` only for the two always-constant deciders (`NeverDisaggregate`, `AlwaysDisaggregate`). Remove `PrefixThresholdDecider` from that test (no longer state-agnostic — that's the whole point of this PR).

**Verify:** `go test ./sim/ -run TestPrefix -v && go test ./sim/ -run TestDisagg -v`.

**Lint:** `golangci-lint run ./sim/...`.

### Task 3: Wire cluster construction + populate `SelectedInstance` + remove observer helper (BC-5, BC-6)

**Files:** modify `sim/cluster/cluster.go`, `sim/cluster/pd_events.go`.

**Impl in `cluster.go` — reorder decider construction:**

The current switch (L236-241) constructs the decider at L238 *before* `cs.cacheQueryFn` is built at L351. Move the `case "prefix-threshold":` branch to after L351 while keeping the `default:` branch in place — or cleanest: move the whole switch. I'll move the whole switch block from L236-241 to a new location right after L351 (where `cs.cacheQueryFn` is guaranteed populated), and leave the pool-membership + parent-map setup at L233-245.

**Sketch:**

```go
// near L234 — keep pool membership + parent maps; drop decider wiring
if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 {
    cs.poolMembership = prePoolMembership
    cs.parentRequests = make(map[string]*ParentRequest)
    cs.pendingPrefillCompletions = make(map[string]string)
    cs.pendingDecodeCompletions = make(map[string]string)
    logrus.Infof("[cluster] PD disaggregation enabled: …")
    if config.SharedInstances > 0 && !reflect.DeepEqual(config.PrefillOverrides, config.DecodeOverrides) {
        logrus.Infof("[cluster] shared-role pods use DecodeOverrides …")
    }
}

// … line 351 populates cs.cacheQueryFn …
cs.cacheQueryFn = cs.snapshotProvider.BuildCacheQueryFn()

// NEW: decider construction moved here (needs cacheQueryFn)
if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 {
    switch config.PDDecider {
    case "prefix-threshold":
        cs.disaggregationDecider = sim.NewPrefixThresholdDecider(config.PDPrefixThreshold, int(config.BlockSizeTokens), cs.cacheQueryFn)
    default:
        cs.disaggregationDecider = sim.NewDisaggregationDecider(config.PDDecider)
    }
}
```

**Impl in `cluster.go` — populate `SelectedInstance` and remove notifier:**

```go
// executeDisaggregatedRouting — unchanged up to decodeDecision. Then:
state.SelectedInstance = decodeDecision.TargetInstance
disaggDecision := cs.disaggregationDecider.Decide(req, state)
```

Delete:
- `notifyDisaggregationObserver` helper (L1399-1410).
- Call sites at L1705 (standard routing path, post-GAP-1 unified path) and L1808 (executeDisaggregatedRouting non-disagg path).

**Impl in `pd_events.go`:**
- Delete L84 `cs.notifyDisaggregationObserver(e.request, decision.TargetInstance)` and any surrounding comment referencing it.

**Verify:** `go build ./... && go test ./sim/cluster/ -run TestDisagg -v`.

**Lint:** `golangci-lint run ./sim/cluster/...`.

### Task 4: Rewrite cluster-level behavioral test for per-pod cache (BC-1 e2e)

**Files:** modify `sim/cluster/disaggregation_test.go`.

**Impl:**

Rename `TestPrefixThreshold_ObserverWarmsCache` (L656-721) → `TestPrefixThreshold_PerPodCacheQuery`. The scenario shifts:

- Before: req1's prefill warmed a global LRU via `ObserveRouting`; req2 with overlapping prefix consulted the global LRU → not disaggregated.
- After: req1's prefill runs on decode pod X (via PD disaggregated path), X's actual KV cache stores the blocks; req2 arrives later, its decode routing policy picks pod X (sticky prefix-affinity from `precise-prefix-cache` scorer OR by configuration), the decider queries X's per-pod cache → finds the blocks → not disaggregated.

The test needs the test config to:
- Use 1 decode instance (so there's no routing ambiguity — req1 and req2 both land on the same pod), OR
- Use `precise-prefix-cache` scorer and `--cache-signal-delay=0` (oracle mode) to guarantee routing affinity.

Option "1 decode instance" is simpler. Update `newTestPrefixThresholdConfig` caller here to use `newTestDisaggDeploymentConfig(2, 1, 1)` — 1 prefill + 1 decode. (Other tests in the file use 2+2; leave them untouched.)

Also delete the error message references to `notifyDisaggregationObserver` wiring (L718).

**Keep other `TestPrefixThreshold_*` tests** (BelowThreshold, AboveThreshold) as-is — they don't depend on observer warming; they just exercise threshold bifurcation with unique tokens (no cache hits expected).

**Verify:** `go test ./sim/cluster/ -run TestPrefixThreshold -v`.

### Task 5: Full test + lint gate

**Commands:**

```
go build ./...
go test ./... -count=1
golangci-lint run ./...
```

**Pass criterion:** build clean; all packages pass; lint reports 0 issues.

### Task 6: Commit and push

**Commit message:**

```
refactor(sim): PrefixThresholdDecider queries per-pod KV cache (GAP-3)

- Replace cluster-wide LRU approximation with per-pod cacheQueryFn lookup
  against state.SelectedInstance, matching llm-d PrefixBasedPDDecider.
- Add RouterState.SelectedInstance; cluster populates it before Decide.
- Thread cs.cacheQueryFn through NewPrefixThresholdDecider; reorder
  construction to run after cacheQueryFn is built.
- Delete DisaggregationObserver interface, ObserveRouting, scratchpad
  fields, globalVirtualInstance constant, notifyDisaggregationObserver
  helper and its three call sites (BC-6).
- Rewrite observer-wiring tests to assert per-pod cache consultation.

Implements BC-1..BC-7. INV-9 preserved. Applies to blis run + blis replay.

Fixes #1263
Part of #1260

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Sanity Checklist

- [x] **R1 (no silent continue):** no new error paths; deletions remove observer calls cleanly.
- [x] **R3 (defensive validation):** constructor still validates threshold ≥ 0 and blockSize > 0; new third arg (cacheQuery) allowed to be nil (test mode).
- [x] **R4 (canonical construction):** struct fields shrink (`idx`, `cachedHashes`, `cachedReqID` removed, `cacheQuery` added); only one construction site in `sim/cluster/cluster.go`.
- [x] **R6 (no `logrus.Fatalf` in sim/):** no new error handling in library path.
- [x] **R13 (interface ≥2 backends):** `DisaggregationObserver` removed entirely, so R13 no longer applies; `DisaggregationDecider` keeps ≥3 backends.
- [x] **R17/R23 (signal freshness):** per-pod cache query inherits `CachedSnapshotProvider` staleness (controlled by `--cache-signal-delay`) — same tier as `precise-prefix-cache` scorer (documented in code comment on the new `cacheQuery` field).
- [x] **INV-1/3/4/5/6/7/8/10/11/12:** no event ordering, conservation, or causality changes. Same `Decide` return type; same event pipeline; same aggregation.
- [x] **INV-9:** `Decide` reads `req.InputTokens` and `state.SelectedInstance` only; no `req.OutputTokens` access.
- [x] **INV-6 determinism:** `cacheQuery` map iteration is not used for ordered output; per-pod query is a single keyed lookup (`map[key]` access). No new sort order introduced.
- [x] **Cross-path parity:** `blis run` + `blis replay` (DES). `blis observe` unaffected (no decider in HTTP path).
- [x] **CLAUDE.md / docs:** no canonical-source edits required. Change-history entry skipped (the Recent Changes log was removed in #1259).
