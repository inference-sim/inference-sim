# Remove Deferred Queue Dead Code — Implementation Plan

**Goal:** Remove all deferred queue infrastructure from `sim/cluster/` — the field, three methods, test file, stale comments, spec directory, and documentation references — since the gateway queue (#882) superseded this feature.

**Source:** [GitHub issue #1018](https://github.com/inference-sim/inference-sim/issues/1018) (Part D of #1011, GAIE-parity admission overhaul).

**Closes:** Fixes #1018.

**PR Size Tier:** Small — mechanical dead-code removal, no behavioral changes, no new interfaces or CLI flags.

**Source Document Audit:** No clarifications needed. The issue accurately describes all dead code locations (line numbers verified and updated). All items confirmed dead: `deferredQueue` is never populated, `isBusy()` and `promoteDeferred()` are `//nolint:unused`, `DeferredQueueLen()` is only called from tests (verified: grep shows no callers outside `cluster_deferred_test.go`), and `DeferredHorizonInterrupted` was never wired into `cmd/` or `metrics.go`.

**Note on #899:** `isBusy()`, `promoteDeferred()`, and `DeferredQueueLen()` were originally preserved for issue #899 (scheduling-tier deferral redesign). That feature is now moot — the gateway queue (`GatewayQueue` with saturation-gated dispatch, #882) handles flow control at the admission layer, which is the better architectural location. These methods are safe to remove unconditionally.

## Behavioral Contracts

BC-1: Dead code removal does not break existing behavior
- GIVEN the deferred queue code is unreachable (nothing populates `deferredQueue`, `isBusy()` and `promoteDeferred()` are unused, `DeferredQueueLen()` is only called from its own tests)
- WHEN all deferred queue code, tests, and the spec directory are removed
- THEN `go build ./...` succeeds, `go test ./...` passes with no regressions, and `golangci-lint run ./...` reports no new issues

BC-2: Documentation accuracy after removal
- GIVEN `CLAUDE.md` and `docs/contributing/standards/invariants.md` reference the deferred queue
- WHEN those references are removed or updated
- THEN the documentation accurately reflects the current codebase (no mentions of deferred queue infrastructure that no longer exists)

## Tasks

### Task 1: Remove deferred queue code from cluster.go (BC-1)

**Files:** modify `sim/cluster/cluster.go`

**What to remove:**
1. Line 38: `deferredQueue []*sim.Request` field from `ClusterSimulator` struct
2. Lines 1062–1079: `isBusy()` method (including `//nolint:unused` directive on line 1070)
3. Lines 1149–1165: `promoteDeferred()` method (including `//nolint:unused` directive on line 1156)
4. Lines 1168–1176: `DeferredQueueLen()` method
5. Lines 634–639: Replace the stale comment and warning that reference `isBusy()` and the deferred queue. The bookkeeping check itself is still valid — a terminated instance should have zero in-flight requests. Replace with:
   ```go
   // I1: a non-zero inFlightRequests at termination time indicates a bookkeeping bug —
   // a missing completion event or an early-termination race.
   if c.inFlightRequests[instID] != 0 {
       logrus.Warnf("[cluster] instance %s terminated with inFlightRequests=%d — bookkeeping bug",
           instID, c.inFlightRequests[instID])
   ```

**Verify:** `go build ./... && go test ./sim/cluster/... && golangci-lint run ./sim/cluster/...`
**Commit:** `refactor(cluster): remove deferred queue dead code (BC-1)`

### Task 2: Remove cluster_deferred_test.go and preserve TestBatchRequestsNotSerialized (BC-1)

**Files:** modify `sim/cluster/cluster_test.go` (add one test), delete `sim/cluster/cluster_deferred_test.go`

**Step 2a — Move `TestBatchRequestsNotSerialized` to `cluster_test.go`:** This test is a regression guard for issue #965 (batch requests should not be serialized). It does NOT use `DeferredQueueLen()` and is independent of the deferred queue. Move it (and its helper `newBatchTestRequests`) to `cluster_test.go` before deleting the file.

**Step 2b — Delete `cluster_deferred_test.go`:** The remaining four test functions (`TestAlwaysAdmit_BatchNotDeferred`, `TestTierShed_RejectsBatchUnderOverload`, `TestINV1_NoDeferredTerm`, `TestDeferredQueueInfraExists`) all call `DeferredQueueLen()` and test deferred-queue-specific behavior that no longer exists. Conservation is already tested by `TestConservation_*` tests in `cluster_test.go`.

**Verify:** `go test ./sim/cluster/... -run TestBatchRequestsNotSerialized`
**Commit:** `refactor(cluster): preserve regression test, remove cluster_deferred_test.go (BC-1)`

### Task 3: Remove specs/004-deferred-queue-batch/ directory (BC-1)

**Files:** delete `specs/004-deferred-queue-batch/` directory (8 files: spec.md, plan.md, tasks.md, research.md, data-model.md, quickstart.md, checklists/requirements.md, contracts/deferred-queue.md)

**Why safe:** This is the spec for the superseded feature. The feature was implemented, then the pre-admission intercept was removed in #1012. The spec describes behavior that no longer exists.

**Verify:** `go build ./...` (specs are not compiled — this is a sanity check)
**Commit:** `refactor: remove specs/004-deferred-queue-batch/ (BC-1)`

### Task 4: Update documentation references (BC-2)

**Files:** modify `CLAUDE.md`, modify `docs/contributing/standards/invariants.md`

**CLAUDE.md changes:**
1. Remove the "Recent Changes" bullet about "Remove deferred-queue pre-admission intercept (#1012/#1016)" — this is stale (describes a completed intermediate step, and the deferred queue is now fully removed).
2. Remove the bullet about "Phase 1B-2a: Deferred queue for batch/background requests (#810)" — the feature no longer exists.
3. In the bullet about "fix(workload): inference_perf SLOClass regression (#965)", remove the clause "Commit `8bc7a48c` introduced a deferred queue that serialized all `batch`-class requests; inference_perf workloads (used by all training experiments) had `SLOClass: "batch"` as a semantically-inert legacy label, inflating TTFT 6–100× after the deferred queue was added." — this historical context is no longer relevant since the deferred queue is gone. Keep the rest of the bullet (the SLOClass fix itself is still relevant).
4. **Do NOT modify** the bullet about "Precise prefix cache scoring (#883)". The phrase "including deferred NodePool instances" refers to dynamically-arriving instances via `NodeReadyEvent` (deferred placement), not the dead deferred queue. These are two different uses of "deferred" — the NodePool deferred placement feature is still active.

**invariants.md change:**
1. Line 15: Remove `sim/cluster/cluster_deferred_test.go` from INV-1 verification reference (file is being deleted in Task 2). Keep the reference to `sim/cluster/cluster_test.go`.

**Verify:** Grep for "deferred" in CLAUDE.md and invariants.md to confirm no stale references remain.
**Commit:** `docs: remove deferred queue references from CLAUDE.md and invariants.md (BC-2)`

## Sanity Checklist

- [ ] **R1 (no silent drops):** No behavioral code is removed — only dead code. No new `continue` or early `return` paths.
- [ ] **R4 (construction site):** `deferredQueue` field removal — checked `NewClusterSimulator` for any initialization. The field was nil-initialized (Go zero value), so no construction site update is needed.
- [ ] **INV-1 (conservation):** Conservation equation in CLAUDE.md does not include `deferred_horizon_interrupted` (it was removed from the equation in #1012). No change needed to INV-1 itself.
- [ ] **INV-6 (determinism):** No behavioral changes — determinism unaffected.
- [ ] **INV-8 (work-conserving):** Gateway queue maintains work-conserving property — no change.
- [ ] **Build + test + lint pass:** `go build ./... && go test ./... && golangci-lint run ./...`
