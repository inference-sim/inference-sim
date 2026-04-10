# hardening(workload): warn on zero-session client + single-stage multi-user session test

**Status:** Awaiting human review (Step 3)
**Closes:** Fixes #976, fixes #979 (T1-1 only)
**PR size tier:** Small (2 files, 1 warning line + 1 test function)
**The problem today:** (1) A closed-loop client that produces zero SessionBlueprints due to a future bug (e.g. `ClientID` not set on round-0 requests) fails silently — no log, no error. (2) The single-stage multi-user multi-turn scenario (same TenantID shared across users within a prompt group) has no regression test, despite being equally vulnerable to the bug fixed in #975.
**What this PR adds:** A `logrus.Warnf` after the session-matching loop when a closed-loop client gets zero sessions; and a regression test `TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient`.
**Why together:** Same file, complementary hardening — the test asserts the invariant holds, the warning makes it observable at runtime if it ever breaks.
**Architecture:** Pure `sim/workload/` changes. No new types, interfaces, or CLI flags.
**Source:** GitHub issues #976, #979 (T1-1)

---

## Phase 0: Component Context

1. **Building block modified:** `GenerateWorkload` in `sim/workload/generator.go` — the closed-loop session-matching inner loop (lines ~455-475 post-#975).
2. **Adjacent blocks:** `GenerateReasoningRequests` (sets `req.ClientID`), `SessionManager` (consumes blueprints). Neither is modified.
3. **Invariants touched:** None directly. The warning fires only when the invariant (one session per closed-loop client) is violated — it does not enforce it.
4. **Construction site audit:** No new struct fields. No construction sites to update.

---

## Part 1: Design Validation

### A) Executive Summary

After the session-matching loop in `GenerateWorkload`, each closed-loop client's `sessionIDsForClient` map holds the sessions it owns. With the `ClientID` fix from #975, this should always have exactly one entry. But if a future code path generates round-0 requests without setting `ClientID`, the map will be empty and the client will silently produce no blueprints — the simulation runs with wrong workload shape and no diagnostic.

The fix: emit `logrus.Warnf` immediately after the loop if `len(sessionIDsForClient) == 0`. This is a diagnostic guard for future regressions, not a behavior change for correct workloads.

The test: `TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient` uses 1 stage × 1 prompt × 4 users. All 4 clients share `TenantID="prompt-0"` and `SLOClass="standard"` — the same conflation trigger as the multi-stage bug in #974/#975. Asserts 4 sessions total, 1 per client.

### B) Root Cause Context (why these two go together)

Issue #976 guards against the failure mode that issue #974 was: a closed-loop client getting zero sessions. The test (T1-1) exercises the exact single-stage analog of the scenario we just fixed. Together they close the observability and test coverage gaps simultaneously.

### C) Behavioral Contracts

**BC-1: Warning emitted when closed-loop client has zero sessions**
- GIVEN a `GenerateWorkload` call where a closed-loop client's round-0 requests have `ClientID` unset (or mismatched)
- WHEN the session-matching loop completes with `len(sessionIDsForClient) == 0`
- THEN a `logrus.Warn` message is emitted containing the client ID
- AND execution continues (no panic, no error return — warning only)

**BC-2: No warning emitted for correct workloads**
- GIVEN a correct multi-stage or single-stage multi-turn workload (all `ClientID` fields set)
- WHEN `GenerateWorkload` runs
- THEN no warning is emitted for any closed-loop client

**BC-3: Single-stage multi-user → one session per client**
- GIVEN `InferencePerfSpec` with 1 stage, `NumUsersPerSystemPrompt=4`, `EnableMultiTurnChat=true`
- WHEN `GenerateWorkload(ws, horizon, 0)` is called
- THEN `len(gw.Sessions) == 4` (one per client)
- AND for every client `c`: exactly one `SessionBlueprint` has `ClientID == c.ID`

### D) Component Interaction

```
ExpandInferencePerfSpec
  → ClientSpec list (each with unique ClientID, shared TenantID per prompt group)

GenerateRequests
  → GenerateReasoningRequests sets req.ClientID = client.ID (correct path)
  → (future bug path: req.ClientID = "" → warning fires)

GenerateWorkload (this PR touches lines ~455-475)
  → session-matching loop: req.ClientID == client.ID
  → if len(sessionIDsForClient) == 0:
      logrus.Warnf(...)   ← NEW
  → else: create blueprints as before (UNCHANGED)
```

### E) Deviation Log

| # | Source says | Plan does | Reason |
|---|------------|-----------|--------|
| 1 | #976 suggests exact warning message | Plan uses that exact message | No deviation |
| 2 | #979 provides complete test implementation | Plan uses it verbatim | No deviation |
| 3 | #976 says warning fires when `ClientID` not set | Warning fires when `len(sessionIDsForClient) == 0` | These are equivalent for current call sites; the condition is more general and correct |

### F) Review Guide

1. **BC-1**: Is `logrus.Warnf` the right severity? Yes — this is a diagnostic for a degenerate state, not an error that stops execution. R6 prohibits `logrus.Fatalf` in `sim/` packages; `Warnf` is already used in `generator.go` (line ~56).
2. **BC-2**: Does the warning fire for any normal workload? No — `len(sessionIDsForClient)` is nonzero for all correct workloads since each client's requests have `ClientID` set by `GenerateReasoningRequests`.
3. **BC-3**: The test exercises the single-stage multi-user case. With 1 prompt group and 4 users, all clients share `TenantID="prompt-0"`. The old (pre-#975) predicate would have caused user-0 to claim all 4 sessions; the new ClientID predicate gives each user exactly 1. The test confirms the fix holds for this case.

---

## Part 2: Executable Implementation

### G) Task Breakdown

#### Task 1: Add warning for zero-session closed-loop client (BC-1, BC-2)

**File:** `sim/workload/generator.go`

**Change:** After the session-matching loop (after the closing `}` of `for _, req := range reqs { ... }`), add:

```go
// Warn if a closed-loop client produced no sessions. This indicates
// that round-0 requests for this client have ClientID unset or mismatched
// (e.g. a future code path that bypasses GenerateReasoningRequests).
// With the current implementation this should never fire.
if len(sessionIDsForClient) == 0 {
    logrus.Warnf("GenerateWorkload: closed-loop client %q produced no sessions — ClientID may not be set on round-0 requests", client.ID)
}
```

**Exact insertion point:** This goes between the closing `}` of the session-matching loop and the start of the `// Create a blueprint per session` comment. In the post-#975 code, this is approximately:

```go
        // Find all session IDs for this client in the generated requests.
        // Match by ClientID: ...
        sessionIDsForClient := make(map[string]bool)
        for _, req := range reqs {
            if req.SessionID != "" && req.RoundIndex == 0 && req.ClientID == client.ID {
                sessionIDsForClient[req.SessionID] = true
                closedLoopSessionIDs[req.SessionID] = true
            }
        }
        // ← INSERT WARNING HERE ←

        // Create a blueprint per session (R2: sort map keys for deterministic RNG draws)
        sortedSessionIDs := make([]string, 0, len(sessionIDsForClient))
```

**No import needed:** `logrus` is already imported in `generator.go`.

**Verify (no test needed for warning path — it's a diagnostic for future bugs):**
```bash
go build ./sim/workload/...
golangci-lint run ./sim/workload/...
```

---

#### Task 2: Add regression test for single-stage multi-user session count (BC-3)

**File:** `sim/workload/generator_test.go`

**Test to add** (append after `TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient`):

```go
// TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient is a
// regression test for the single-stage analog of #974.
//
// Single-stage workloads with NumUsersPerSystemPrompt > 1 have the same
// conflation trigger as multi-stage: all users in a prompt group share
// TenantID = prefixGroup (e.g. "prompt-0") and SLOClass = "standard".
// The old (TenantID, SLOClass, Model) predicate would cause user-0 to claim
// all sessions; the ClientID predicate (fixed in #975) gives each user exactly 1.
//
// Invariant: each client owns exactly one SessionBlueprint.
func TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient(t *testing.T) {
	// 1 stage × 1 prompt × 4 users = 4 clients.
	// All share TenantID="prompt-0", SLOClass="standard" — the conflation trigger.
	ipSpec := &InferencePerfSpec{
		Stages: []StageSpec{{Rate: 5.0, Duration: 60}},
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  1,
			NumUsersPerSystemPrompt: 4,
			SystemPromptLen:         10,
			QuestionLen:             20,
			OutputLen:               10,
			EnableMultiTurnChat:     true,
		},
	}
	ws, err := ExpandInferencePerfSpec(ipSpec, 42)
	if err != nil {
		t.Fatalf("ExpandInferencePerfSpec: %v", err)
	}
	ws.Version = "2"

	horizon := int64(60_000_000) // 60 seconds in µs
	gw, err := GenerateWorkload(ws, horizon, 0)
	if err != nil {
		t.Fatalf("GenerateWorkload: %v", err)
	}

	numClients := len(ws.Clients)
	if numClients != 4 {
		t.Fatalf("expected 4 clients from ExpandInferencePerfSpec, got %d — spec changed?", numClients)
	}
	if len(gw.Sessions) != numClients {
		t.Errorf("session count = %d, want %d (one per client)", len(gw.Sessions), numClients)
	}

	// Each client must own exactly one SessionBlueprint.
	sessionsByClient := make(map[string]int)
	for _, bp := range gw.Sessions {
		sessionsByClient[bp.ClientID]++
	}
	for _, c := range ws.Clients {
		if got := sessionsByClient[c.ID]; got != 1 {
			t.Errorf("client %q: owns %d sessions, want exactly 1", c.ID, got)
		}
	}
}
```

**Verify (expect PASS — the fix in #975 already handles this case):**
```bash
go test ./sim/workload/ -run TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient -v
# expected: PASS

go test ./sim/workload/ -count=1
# expected: ok
```

**Lint:**
```bash
golangci-lint run ./sim/workload/...
```

**Commit:** `hardening(workload): warn on zero-session closed-loop client; test single-stage multi-user (#976, #979)`

---

### H) Test Strategy

| Contract | Task | Test type | Test name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Defensive warning (no direct unit test — warning fires only on future bug condition) | — |
| BC-2 | Task 2 | Regression (negative: warning must NOT fire for correct workload) | `TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient` verifies correct path |
| BC-3 | Task 2 | Invariant | `TestGenerateWorkload_SingleStageMultiUserMultiTurn_OneSessionPerClient` |

**Note on BC-1 testability:** The warning fires only when `ClientID` is unset on round-0 requests — a state not reachable through the public API with current code. A direct unit test would require constructing pathological `[]*sim.Request` with `ClientID=""` for a client that `isClosedLoop()` returns true for. This is possible but adds test complexity for a pure observability addition. Given the warning is a guard against future bugs (not current behavior), omitting a direct test is acceptable. The warning's presence in code is verified by the build.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Warning fires spuriously on valid workloads | None | High (noise) | BC-2: `len(sessionIDsForClient) == 0` is unreachable with current ClientID-matching code for any valid spec |
| Test T1-1 reveals that single-stage multi-user was still broken | Low | High | The fix in #975 (ClientID predicate) already handles single-stage; test confirms this |
| `logrus.Warnf` in `sim/workload/` violates R6 | None | — | R6 prohibits `Fatalf`/`os.Exit`, not `Warnf`; `Warnf` is already used at line ~56 of generator.go |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions
- [x] No feature creep — warning is one line; test is one function
- [x] No new interfaces, types, or CLI flags
- [x] CLAUDE.md: no new packages, no file organization changes, no new CLI flags — no update required
- [x] All contracts mapped to tasks
- [x] No golden dataset regeneration needed — no simulation output changes
- [x] Construction site audit: no new struct fields
- [x] Not part of a macro plan

**Antipattern rules:**
- [x] R1: Warning does not suppress or drop data — it fires and continues
- [x] R2: No map iteration for output — `sessionIDsForClient` is only checked for length
- [x] R6: `logrus.Warnf` (not Fatalf) — permitted in `sim/` packages
- [x] R7: Test is an invariant test (one session per client), not a golden value
- [x] R15: Issue references #976 and #979 — both open

---

## Appendix: File-Level Implementation Details

### `sim/workload/generator.go`

**Insertion point** (post-#975 line numbers, approximately):

```
line 455:     // Find all session IDs for this client...
line 456:     sessionIDsForClient := make(map[string]bool)
line 457:     for _, req := range reqs {
line 458:         if req.SessionID != "" && req.RoundIndex == 0 && req.ClientID == client.ID {
line 459:             sessionIDsForClient[req.SessionID] = true
line 460:             closedLoopSessionIDs[req.SessionID] = true
line 461:         }
line 462:     }
line 463:     // ← INSERT 4-LINE WARNING BLOCK HERE ←
line 464:
line 465:     // Create a blueprint per session (R2: sort map keys...)
```

**No import change needed.** `logrus` is imported at the top of `generator.go`.

### `sim/workload/generator_test.go`

**Insertion point:** Append the new test function at the end of the file, after `TestGenerateWorkload_MultiStageMultiTurn_OneSessionPerClient` (the test added in #975).
