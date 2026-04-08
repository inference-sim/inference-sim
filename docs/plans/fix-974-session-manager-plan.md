# fix(workload): Session Conflation in Multi-Stage Multi-Turn Workloads

**Status:** Awaiting human review (Step 3)
**Closes:** Fixes #974
**PR size tier:** Medium (behavioral logic fix; compact format not applicable)
**The problem today:** Multi-stage + multi-turn workloads via `InferencePerfSpec` silently produce wrong session counts — the first client claims all sessions, all others get zero.
**What this PR adds:** Corrects the session-matching predicate in `GenerateWorkload` to use `req.ClientID == client.ID` and removes the now-redundant `claimedSessions` guard.
**Why this matters:** Accurate session generation is essential for closed-loop multi-turn calibration runs; the bug renders multi-stage + multi-turn workloads unusable.
**Architecture:** Pure fix within `sim/workload/` (library package). Touches `GenerateWorkload` in `generator.go` only; no new types, interfaces, or CLI flags.
**Source:** GitHub issue #974
**Behavioral Contracts:** Part 1 Section C below.

---

## Phase 0: Component Context

1. **Building block modified:** `GenerateWorkload` in `sim/workload/generator.go` — the function that converts a `WorkloadSpec` into round-0 requests + `SessionBlueprint` list.
2. **Adjacent blocks:** `ExpandInferencePerfSpec` (produces `ClientSpec` list with unique `clientID`), `GenerateReasoningRequests` (sets `req.ClientID`), `SessionManager` (consumes `SessionBlueprint` list).
3. **Invariants touched:** INV-6 (determinism) — fix must not introduce non-deterministic behavior. INV-11 (session completeness) — every session must reach a terminal state; a session that is never created (0 blueprints) violates this.
4. **Construction site audit:** No new struct fields added. `SessionBlueprint` is constructed at `generator.go:483-501` (closed-loop path) and `generator.go:607-622` (concurrency path) — both are unchanged. No audit action required.

---

## Part 1: Design Validation

### A) Executive Summary

`GenerateWorkload` builds `SessionBlueprint` objects for closed-loop multi-turn clients by scanning round-0 requests and matching them to their originating client. The matching predicate uses `(TenantID, SLOClass, Model)`. In multi-stage workloads (via `ExpandInferencePerfSpec`), every client across all stages shares the same `TenantID` (the `prefixGroup`, e.g. `"prompt-0"`) and the same `SLOClass` (`"standard"`), making the predicate non-unique. The first client to iterate (`stage-0-prompt-0-user-0`) sweeps all matching sessions; every subsequent client gets zero blueprints. The `claimedSessions` guard was intended to prevent double-claiming but fails because the first client claims everything before others iterate.

The fix replaces the predicate with `req.ClientID == client.ID`. `GenerateReasoningRequests` (reasoning.go:77) already sets `req.ClientID = clientID` uniquely per client, making this a provably correct 1:1 match. The now-redundant `claimedSessions` map is removed. A secondary instance of the broad predicate in the prefix-extraction loop (line 448) is also replaced for consistency.

### B) Root Cause Trace

```
ExpandInferencePerfSpec (multi-stage path, inference_perf.go:230-248)
  → stage clients: TenantID = prefixGroup = "prompt-0" for ALL stages
  → SLOClass = "standard" for ALL clients
  → clientID = "stage-{s}-prompt-{p}-user-{u}" (unique per client)

GenerateRequests (generator.go:167)
  → GenerateReasoningRequests(clientID=client.ID, ...)
      → req.ClientID = clientID   ← unique
      → req.TenantID = tenantID   ← shared across stages
      → req.SessionID = fmt.Sprintf("sess_%d", rng.Int63())  ← unique

GenerateWorkload (generator.go:464-474) — THE BUG
  → old predicate: (TenantID, SLOClass, Model) — but ExpandInferencePerfSpec
    never sets Model on ClientSpec, so Model="" for all clients; the effective
    predicate was (TenantID, SLOClass) only, making the conflation even easier
  → for client := stage-0-prompt-0-user-0:
      matches ALL reqs where TenantID="prompt-0", SLOClass="standard"
      → claims sessions from stage-0 AND stage-1 clients
  → for client := stage-0-prompt-0-user-1:
      all its sessions already claimed → sessionIDsForClient = {}
  → for clients stage-0-prompt-0-user-2 .. stage-1-prompt-0-user-2:
      same — zero blueprints
```

### C) Behavioral Contracts

**BC-1: One session per client (primary invariant)**
- GIVEN a multi-stage + multi-turn InferencePerfSpec with S stages × N prompts × M users
- WHEN `GenerateWorkload` is called
- THEN every client owns exactly one `SessionBlueprint` in `gw.Sessions` (no client receives more than one, no client receives zero)

**BC-2: Total session count equals total client count**
- GIVEN the same spec as BC-1
- WHEN `GenerateWorkload` is called
- THEN `len(gw.Sessions) == S × N × M`

**BC-3: Single-stage multi-turn unaffected (regression)**
- GIVEN an InferencePerfSpec with exactly 1 stage and `EnableMultiTurnChat=true`
- WHEN `GenerateWorkload` is called
- THEN every client still owns exactly one `SessionBlueprint` (no regression from change)

**BC-4: Prefix tokens extracted from correct client's request**
- GIVEN a multi-stage multi-turn spec with `SystemPromptLen > 0` and `PrefixGroup != ""`
- WHEN `GenerateWorkload` is called
- THEN each `SessionBlueprint.Prefix` has length equal to its client's `PrefixLength`,
  provided the round-0 request's `InputTokens` length is ≥ `PrefixLength`
  (extracted from that client's own round-0 request, not another client's)

### D) Component Interaction

```
ExpandInferencePerfSpec          GenerateRequests
        │                               │
        │ ClientSpec list               │ []*sim.Request (with ClientID set)
        └──────────────────────────────►│
                                        │
                                  GenerateWorkload
                                  ┌─────┴──────────────────┐
                                  │ for each closed-loop    │
                                  │ client:                 │
                                  │   scan reqs where       │
                                  │   req.ClientID==client  │ ← FIX: was (TenantID,SLOClass,Model)
                                  │   → SessionBlueprint    │
                                  └────────────┬────────────┘
                                               │ []SessionBlueprint (1 per client)
                                               ▼
                                        SessionManager
                                   (consumes blueprints,
                                    generates follow-up rounds)
```

State boundary: `GenerateWorkload` reads `req.ClientID` (set by `GenerateReasoningRequests`) and writes `SessionBlueprint.ClientID`. No cluster-level state accessed.

### E) Deviation Log

| # | Source says | Plan does | Reason |
|---|------------|-----------|--------|
| 1 | Issue #974 names two root cause candidates: `ExpandInferencePerfSpec` and `GenerateWorkload` | Fix is in `GenerateWorkload` only | Investigation confirmed `ExpandInferencePerfSpec` correctly generates unique `clientID` per stage (inference_perf.go:233); the bug is solely in the session-matching predicate in `GenerateWorkload`. |
| 2 | Issue #974 describes `blis observe` and `blis run` modes as both affected | Plan modifies only `sim/workload/generator.go` | Both modes share `GenerateWorkload`; fixing the shared code path fixes both. |

### F) Review Guide

Focus areas:
1. **BC-1/BC-2 correctness** — does `req.ClientID == client.ID` correctly 1:1 attribute sessions? Verify: `GenerateReasoningRequests` sets `req.ClientID = clientID` (reasoning.go:77) where `clientID` is passed from `GenerateRequests` as `client.ID`. ✅
2. **`claimedSessions` removal safety** — `SessionID` values are random int64-derived strings (reasoning.go:39), globally unique. `ClientID` is unique per client. No two sessions share a `ClientID`. The guard is redundant post-fix. ✅
3. **Secondary fix (prefix extraction)** — functionally safe for `InferencePerfSpec` (all clients in a `prefixGroup` have identical prefix tokens from `SystemPromptLen`). The fix makes the extraction more precise. ✅
4. **Single-stage path (BC-3)** — unaffected: single-stage clients each have unique `TenantID` combinations OR `claimedSessions` was rescuing them; the new code is correct by construction regardless.

---

## Part 2: Executable Implementation

### G) Task Breakdown

#### Task 1: Regression test for BC-1 and BC-2 — DONE

**Contracts:** BC-1, BC-2
**Files:** modify `sim/workload/generator_test.go`

**1. Write failing test:**
```go
// TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient (added at end of generator_test.go)
// GIVEN: InferencePerfSpec{2 stages, 1 prompt, 3 users} → 6 total clients
//   all sharing TenantID="prompt-0", SLOClass="standard" — the conflation trigger
// WHEN:  GenerateWorkload(ws, 120s horizon, 0)
// THEN:  len(gw.Sessions) == 6
//        for each client: sessionsByClient[client.ID] == 1
```

**2. Run to verify FAIL:** ✅ confirmed
```
stage-0-prompt-0-user-0: owns 6 sessions, want exactly 1
stage-0-prompt-0-user-{1,2}: owns 0 sessions, want exactly 1
stage-1-prompt-0-user-{0,1,2}: owns 0 sessions, want exactly 1
```

**3-5.** Fix applied in Task 2. Test now passes. Lint: no issues.

**Commit:** deferred — single commit with Task 2.

---

#### Task 2: Fix session-matching predicate — DONE

**Contracts:** BC-1, BC-2, BC-3, BC-4
**Files:** modify `sim/workload/generator.go`

**1. Change 1 — Remove `claimedSessions` (now redundant):**
Remove the variable declaration and its comment at generator.go:417-419 (pre-fix).

**2. Change 2 — Fix prefix-token extraction (BC-4):**
```go
// OLD:
if req.SessionID != "" && req.RoundIndex == 0 &&
    req.TenantID == client.TenantID && req.SLOClass == client.SLOClass {
// NEW:
if req.SessionID != "" && req.RoundIndex == 0 && req.ClientID == client.ID {
```

**3. Change 3 — Fix session-matching predicate (BC-1, BC-2):**
```go
// OLD:
for _, req := range reqs {
    if req.SessionID != "" && req.RoundIndex == 0 {
        if req.TenantID == client.TenantID && req.SLOClass == client.SLOClass && req.Model == client.Model {
            if owner, claimed := claimedSessions[req.SessionID]; !claimed || owner == client.ID {
                sessionIDsForClient[req.SessionID] = true
                closedLoopSessionIDs[req.SessionID] = true
                claimedSessions[req.SessionID] = client.ID
            }
        }
    }
}
// NEW:
for _, req := range reqs {
    if req.SessionID != "" && req.RoundIndex == 0 && req.ClientID == client.ID {
        sessionIDsForClient[req.SessionID] = true
        closedLoopSessionIDs[req.SessionID] = true
    }
}
```

**4. Verify PASS:**
```bash
go test ./sim/workload/ -run TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient -v
# PASS
go test ./sim/workload/ -count=1
# ok  github.com/inference-sim/inference-sim/sim/workload
```

**5. Lint:** `golangci-lint run ./sim/workload/...` — 0 issues.

**Commit:** `fix(workload): match sessions by ClientID in GenerateWorkload (#974)`

---

### H) Test Strategy

| Contract | Task | Test type | Test name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Invariant | `TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient` |
| BC-2 | Task 1 | Invariant | `TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient` |
| BC-3 | Existing | Regression | Existing single-stage tests in `inference_perf_test.go` and `generator_test.go` |
| BC-4 | Implicitly covered | Invariant | BC-4 correctness is structural: prefix extraction now uses same `ClientID` predicate, guaranteed by construction |

Notes:
- The new test verifies an invariant law ("one session per client") from first principles — not a golden value.
- BC-3 is verified by the full `go test ./sim/workload/` suite passing without regression.
- Golden datasets regenerated: all 15 experiments in both `roofline_goldendataset.json` and `trained_physics_iter29.json` use multi-stage multi-turn workloads affected by the bug. Correct session attribution changes `completed_requests` (3060→5130) and all latency values. Regenerated via `go test ./sim/cluster/ -update-golden` (added in same PR).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Single-stage regression: removing `claimedSessions` breaks single-stage multi-turn | Low | High | BC-3 covered by existing test suite. Note: single-stage + `NumUsersPerSystemPrompt > 1` also shared `TenantID = prefixGroup` across users (inference_perf.go:125-141), so `claimedSessions` was NOT a no-op there either — the original predicate was equally broken for that case. The ClientID fix is correct for both. | Task 2 |
| Prefix extraction returning wrong tokens for multi-stage | Low | Medium | All clients in a `prefixGroup` have identical prefix tokens (same `SystemPromptLen`); secondary fix is behavior-neutral in practice | Task 2 |
| Non-closed-loop clients accidentally affected | None | — | The fix is inside the `isClosedLoop(client)` guard; non-session clients are untouched | — |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code passes `golangci-lint`.
- [x] Shared test helpers used from existing package (no duplicated helpers).
- [x] CLAUDE.md: no new packages, no new CLI flags, no file organization changes — update not required.
- [x] No stale references in CLAUDE.md.
- [x] Documentation DRY: no canonical sources modified.
- [x] Deviation log reviewed — deviations are justified.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered (test proves bug exists → fix makes it pass).
- [x] All contracts mapped to specific tasks (see H above).
- [x] Golden datasets regenerated (see H above; all 15 experiments in both JSONs updated via `-update-golden`).
- [x] Construction site audit completed — no new struct fields.
- [x] Not part of a macro plan — no macro plan update needed.

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` dropping data.
- [x] R2: `sortedSessionIDs` still sorted before map iteration — unchanged.
- [x] R3: No new numeric parameters.
- [x] R4: No new struct fields — no construction sites to audit.
- [x] R5: No resource allocation loops modified.
- [x] R6: No `logrus.Fatalf` in `sim/` code.
- [x] R7: New test is an invariant test (not golden), satisfying the companion requirement.
- [x] R8: No new exported mutable maps.
- [x] R9: No YAML fields modified.
- [x] R10: No YAML parsing modified.
- [x] R11: No division introduced.
- [x] R12: Golden datasets affected and regenerated — all 15 experiments in `roofline_goldendataset.json` and `trained_physics_iter29.json` use multi-stage multi-turn workloads. Old values encoded the buggy session counts; new values reflect correct behavior.
- [x] R13: No new interfaces.
- [x] R14: No multi-concern methods modified.
- [x] R15: No stale PR references (plan references only `#974`, which is the current issue).
- [x] R16-R23: Not applicable to this fix.

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/generator.go`

**Purpose:** Contains `GenerateWorkload`, which produces `SessionBlueprint` objects for closed-loop multi-turn clients by scanning round-0 requests.

**Key implementation notes:**
- **State mutation:** Only local variables within `GenerateWorkload` are mutated. `claimedSessions` (removed) was a local `map[string]string`. `sessionIDsForClient` and `closedLoopSessionIDs` remain unchanged in purpose.
- **Error handling:** No new error paths introduced. The fix simplifies an existing loop.
- **RNG usage:** No RNG is involved in session matching — `ClientID` is a deterministic string set during `GenerateRequests`.
- **Line references (pre-fix):** `claimedSessions` declaration at lines 417-419; prefix predicate at lines 448-449; session-matching predicate at lines 465-474.

**Complete diff summary:**
```
- claimedSessions := make(map[string]string)  // removed
- if req.TenantID == client.TenantID && req.SLOClass == client.SLOClass {  // prefix loop
+ if req.ClientID == client.ID {  // prefix loop
- if req.TenantID == client.TenantID && req.SLOClass == client.SLOClass && req.Model == client.Model {  // session loop
-     if owner, claimed := claimedSessions[req.SessionID]; !claimed || owner == client.ID {
-         claimedSessions[req.SessionID] = client.ID
+ if req.ClientID == client.ID {  // session loop
```

### File: `sim/workload/generator_test.go`

**Purpose:** Adds regression test `TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient`.

**Test structure:**
- Builds a 2-stage × 1-prompt × 3-user `InferencePerfSpec` (6 total clients, all sharing `TenantID="prompt-0"`)
- Calls `GenerateWorkload` with a 120-second horizon
- Asserts `len(gw.Sessions) == 6` (BC-2)
- Asserts each client owns exactly 1 session by `ClientID` (BC-1)

**Why this test is sufficient:** It directly exercises the conflation trigger (shared `TenantID` across stages) and verifies the invariant (`one session per client`) from first principles — not a golden value.
