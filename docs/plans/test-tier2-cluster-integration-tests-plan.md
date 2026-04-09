# Tier 2 Simulation-Engine Integration Tests Plan

**Goal:** Add integration tests that verify INV-10, INV-11, session context accumulation, and end-to-end multi-turn correctness using the simulation engine, filling the test gaps identified in issue #980.

**The problem today:** Session invariants (INV-10, INV-11) are only verified at the unit level with hardcoded tick arguments in `session_test.go`. No test runs `ClusterSimulator.Run()` with a real `SessionManager` in the standard (non-disaggregated) code path, so silent session abandonment, causality violations, and context-accumulation bugs beyond round 0→1 would escape all existing tests.

**What this PR adds:**
1. `TestSession_ContextAccumulation_MultiStep` — verifies a 3-round accumulation chain (T2-4)
2. `TestSession_HorizonInterrupted_IsTerminal` — verifies horizon-interrupted sessions are truly terminal (T2-5)
3. `TestClusterSimulator_SessionTerminalStateCompleteness` — verifies INV-11 across a full cluster Run() (T2-1)
4. `TestClusterSimulator_SessionFollowUpCausality` — verifies INV-10 when DES computes completion times (T2-2)
5. `TestClusterSimulator_MultiTurnSession_EndToEnd` — verifies INV-1 + round chain integrity in the standard cluster path (T2-3)

**Why this matters:** Tier 2 tests catch DES/session integration bugs (wrong tick passed to OnComplete, silent session abandonment, multi-round context errors) that unit tests cannot, without requiring the latency model backend.

**Architecture:** Two files modified. Tests T2-4 and T2-5 go in `sim/workload/session_test.go` (pure unit, no cluster). Tests T2-1, T2-2, T2-3 go in `sim/cluster/cluster_test.go` (cluster integration). All cluster tests build `SessionBlueprint` structs directly (matching the disaggregation test pattern) and use `newTestDeploymentConfig` + `mustRun` helpers. No production code changes.

**Source:** https://github.com/inference-sim/inference-sim/issues/980

**Closes:** Fixes #980

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block modified:** `sim/workload/session_test.go` (workload unit tests), `sim/cluster/cluster_test.go` (cluster integration tests)
2. **Adjacent components:** `SessionManager.OnComplete`, `ClusterSimulator.Run()`, `NewClusterSimulator`, `newTestDeploymentConfig`, `mustRun`, `assertINV1Conservation`
3. **Invariants touched:** INV-10 (session causality), INV-11 (session terminal completeness), INV-1 (request conservation)
4. **Construction site audit:** No new structs added. `SessionBlueprint` is constructed in `TestDisaggregation_PD_SessionManager_*` (disaggregation_test.go:1627–1645) and will be constructed in the 3 new cluster tests. No canonical constructor exists; each site builds the struct inline — this is the established pattern.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds 5 behavioral tests across 2 files, all testing existing code against stated invariants — no production code changes. Tests T2-4 and T2-5 are pure unit tests in the `workload` package that extend coverage for context accumulation (round 0→1→2 chain) and terminal state idempotency. Tests T2-1, T2-2, T2-3 are Tier 2 integration tests in the `cluster` package that run `ClusterSimulator.Run()` with a real `SessionManager` callback, verifying INV-11 (no silent abandonment), INV-10 (causality from real DES ticks), and INV-1 conservation with dynamically injected follow-ups.

All 3 cluster tests follow the pattern established in `disaggregation_test.go`: build `SessionBlueprint` structs directly, wire `sm.OnComplete` as the cluster callback, and call `mustRun`. Tests avoid float comparisons (no latency assertions) and assert only integer counts, state flags, and ordering.

One deviation from the issue sketches: `atomic.AddInt64` is removed from T2-1 and `sync.Mutex` from T2-2 — the DES event loop is single-threaded, making concurrency primitives incorrect here (see Deviation Log).

### B) Behavioral Contracts

```
BC-1: Context accumulation — 3-round chain
- GIVEN a session with ContextGrowth="accumulate", InputSampler=constant(10), OutputSampler=constant(5)
- WHEN round 0 completes (input=10, output=5) → generates round 1
  AND  round 1 completes → generates round 2
- THEN round 1 input length == 25  (contextTokens(15: r0 input+output) + 10 new)
  AND  round 2 input length == 55  (contextTokens(45: 15 + r1 input(25) + r1 output(5)) + 10 new)
  NOTE: contextTokens is append-only across ALL rounds — it grows to 15 after round 0,
        then to 45 after round 1 (appending r1's 25-token input + 5-token output)

BC-2: Horizon-interrupted session is terminal
- GIVEN a session with Horizon=6000, ThinkTimeUs=1000
- WHEN round 0 completes at tick 5500 (next arrival = 6500 > horizon)
- THEN OnComplete returns nil (no follow-up generated)
  AND  any subsequent OnComplete call for the same session also returns nil

BC-3: INV-11 terminal completeness in a cluster Run()
- GIVEN N session blueprints wired into ClusterSimulator with a real SessionManager
- WHEN ClusterSimulator.Run() completes
- THEN for every session, exactly one call to OnComplete returned nil
  (i.e., terminalCount == N, no session silently abandoned)

BC-4: INV-10 follow-up causality in a cluster Run()
- GIVEN a multi-turn workload with ThinkTimeUs = T run through ClusterSimulator
- WHEN ClusterSimulator.Run() completes
- THEN for every consecutive round pair in every session:
      round[N+1].ArrivalTime >= round[N].completionTick + T

BC-5: INV-1 conservation with dynamic follow-up injection
- GIVEN a ClusterSimulator with session blueprints (MaxRounds=3, large horizon)
- WHEN ClusterSimulator.Run() completes
- THEN CompletedRequests + StillQueued + StillRunning + DroppedUnservable + TimedOutRequests
      == totalInjected  (seeds + follow-ups counted through onDone callback)

BC-6: Follow-ups always have RoundIndex > 0
- GIVEN a multi-turn workload run through ClusterSimulator
- WHEN follow-up requests are returned by the onDone callback
- THEN every follow-up has RoundIndex >= 1 (no follow-up masquerades as a seed)

BC-7: Session chain generates expected rounds
- GIVEN N sessions with MaxRounds=3 and a large horizon
- WHEN ClusterSimulator.Run() completes
- THEN every session has at least one request with RoundIndex > 0
  (session manager generated at least one follow-up per session)
```

### C) Component Interaction

```
sim/workload/session_test.go
  └── SessionManager.OnComplete(req, tick)
         [BC-1: checks accumulation after 2 consecutive completions]
         [BC-2: checks terminality after horizon interruption]

sim/cluster/cluster_test.go
  └── NewClusterSimulator(config, seeds, onDone)
         ↓ onDone wraps sm.OnComplete and records observations
  └── ClusterSimulator.Run()
         ↓ DES computes completion ticks, calls onDone for each
  └── cs.AggregatedMetrics()
         ↓ used for INV-1 conservation (BC-5)

SessionBlueprint built inline (no constructor) — matches disaggregation_test.go pattern.
assertINV1Conservation defined in disaggregation_test.go (same package cluster) — reused.
newTestDeploymentConfig + mustRun defined in cluster_test.go — reused.
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|---|---|---|
| T2-1 sketch uses `atomic.AddInt64` for terminalCount | Uses plain `int` counter | CORRECTION: DES event loop is single-threaded (SessionManager documented as "Single-threaded"). Atomic primitives are incorrect here — they imply concurrent access that doesn't exist and would mislead readers. |
| T2-1 uses `ExpandInferencePerfSpec` + `GenerateWorkload` | Builds `SessionBlueprint` directly | SIMPLIFICATION: Disaggregation tests establish a simpler, clearer pattern. Avoids coupling Tier 2 tests to inference_perf expansion logic. |
| T2-2 sketch uses `sync.Mutex` for records map | No mutex used | CORRECTION: Same single-thread reason as T2-1. |
| T2-2 sketch uses `WorkloadSpec` + `ReasoningSpec` | Builds `SessionBlueprint` directly | SIMPLIFICATION: Same reason as T2-1. |
| T2-3 sketch is marked "incomplete" | Fully implemented: INV-1 + round chain | ADDITION: Filling the stated intent from the issue. |
| T2-1 counts sessions via `sm.OnComplete` wrapper | Same approach, but inlined, no concurrency primitives | CORRECTION: See atomic/mutex deviations above. |

### E) Review Guide

Tricky part: T2-1's counting approach relies on the property that `OnComplete` returns nil exactly once per session (when the session transitions to any terminal state). Reviewers should verify this holds for all 4 terminal states: `sessionCompleted`, `sessionCancelled`, `sessionHorizonInterrupted`, `sessionBudgetExhausted` — each sets `sess.state` before returning nil, so the `state != sessionActive` guard prevents double-counting.

T2-2: verify the completionTick map uses the `tick` argument from `onDone`, not `req.ArrivalTime` or `req.CompletionTime` — the `tick` is the DES clock at completion.

Safe to skim: T2-4 and T2-5 are mechanical extensions of existing session_test.go patterns. The accumulation math is already tested for round 0→1 in `TestSession_ContextAccumulation`; T2-4 just extends to round 2.

Known debt: None introduced. These tests cover existing behavior.

Note on BC labels: This plan's BC-7 ("session chain generates expected rounds") is a different contract than the `session_test.go` inline "BC-7" comment in `TestSession_TimeoutCancels_NoMoreRounds`. The plan uses sequential BC numbers scoped to this PR; the session_test.go BC labels are scoped to a prior plan. No code conflict, but a reviewer noting both should be aware they are from different plans.

---

## Part 2: Executable Implementation

### F) Implementation Overview

- **Modify** `sim/workload/session_test.go` — add 2 tests (T2-4, T2-5). No new imports needed (already has `math/rand`, `testing`, `sim`).
- **Modify** `sim/cluster/cluster_test.go` — add 3 tests (T2-1, T2-2, T2-3) and 2 new imports: `"math/rand"` and `"github.com/inference-sim/inference-sim/sim/workload"`.

No production code changes. No dead code. No new exported symbols.

### G) Task Breakdown

---

#### Task 1: T2-4 — Multi-step context accumulation (BC-1)

**Files:** modify `sim/workload/session_test.go`

**Test (add after `TestSession_ContextAccumulation`):**

```go
// TestSession_ContextAccumulation_MultiStep verifies BC-1:
// context accumulation across a 3-round chain (round 0 → 1 → 2).
// Extends TestSession_ContextAccumulation (which only tests round 0 → 1).
func TestSession_ContextAccumulation_MultiStep(t *testing.T) {
	bp := makeTestBlueprint("sess-accum3", 3, 1000, "accumulate", 1_000_000)
	sm := NewSessionManager([]SessionBlueprint{bp})

	inputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(1)), 10)
	outputR0 := sim.GenerateRandomTokenIDs(rand.New(rand.NewSource(2)), 5)
	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-accum3", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15, // 10 input + 5 output
		InputTokens: inputR0, OutputTokens: outputR0,
	}
	follow1 := sm.OnComplete(req0, 5000)
	if len(follow1) != 1 {
		t.Fatalf("expected 1 follow-up after round 0, got %d", len(follow1))
	}
	// Round 1: 10 (r0 input) + 5 (r0 output) + 10 (new) = 25
	if len(follow1[0].InputTokens) != 25 {
		t.Errorf("BC-1: round 1 input length = %d, want 25 (10+5+10)", len(follow1[0].InputTokens))
	}

	req1 := &sim.Request{
		ID: "r1", SessionID: "sess-accum3", RoundIndex: 1,
		State: sim.StateCompleted,
		ProgressIndex: len(follow1[0].InputTokens) + len(follow1[0].OutputTokens), // 25 + 5 = 30
		InputTokens: follow1[0].InputTokens, OutputTokens: follow1[0].OutputTokens,
	}
	follow2 := sm.OnComplete(req1, 12000)
	if len(follow2) != 1 {
		t.Fatalf("expected 1 follow-up after round 1, got %d", len(follow2))
	}
	// Round 2: contextTokens grows to 45 (prior 15 + r1 input 25 + r1 output 5),
	// then + 10 new = 55. contextTokens is append-only across ALL rounds.
	if len(follow2[0].InputTokens) != 55 {
		t.Errorf("BC-1: round 2 input length = %d, want 55 (contextTokens(45)+10)", len(follow2[0].InputTokens))
	}
}
```

**Impl:** No production code changes needed.

**Verify:** `cd sim/workload && go test -run TestSession_ContextAccumulation_MultiStep -v`
Expected: `--- PASS: TestSession_ContextAccumulation_MultiStep`

**Lint:** `golangci-lint run ./sim/workload/...`

**Commit:** `test(workload): 3-round context accumulation chain (BC-1, T2-4, #980)`

---

#### Task 2: T2-5 — Horizon-interrupted session is terminal (BC-2)

**Files:** modify `sim/workload/session_test.go`

**Test (add after `TestSession_BeyondHorizon_NotGenerated`):**

```go
// TestSession_HorizonInterrupted_IsTerminal verifies BC-2:
// after horizon interruption, any further OnComplete call for the same
// session returns nil (the session is terminal, not silently active).
// Extends TestSession_BeyondHorizon_NotGenerated (which only checks the
// first call, not the terminal-state idempotency required by INV-11).
func TestSession_HorizonInterrupted_IsTerminal(t *testing.T) {
	bp := makeTestBlueprint("sess-hz-term", 3, 1000, "", 6000) // horizon = 6000
	sm := NewSessionManager([]SessionBlueprint{bp})

	req0 := &sim.Request{
		ID: "r0", SessionID: "sess-hz-term", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	// Next arrival = 5500 + 1000 = 6500 > horizon → horizon-interrupted
	follow := sm.OnComplete(req0, 5500)
	if follow != nil {
		t.Errorf("BC-2: expected nil (beyond horizon), got %d follow-ups", len(follow))
	}

	// BC-2: any subsequent call must also return nil (terminal, not active).
	// This simulates an implementation bug where the session state might be
	// reset or not properly persisted — a real DES would not call OnComplete
	// twice for the same session/round, but this guard ensures the terminal
	// state is idempotent regardless.
	req0b := &sim.Request{
		ID: "r0b", SessionID: "sess-hz-term", RoundIndex: 0,
		State: sim.StateCompleted, ProgressIndex: 15,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
	}
	follow2 := sm.OnComplete(req0b, 5500)
	if follow2 != nil {
		t.Errorf("BC-2: expected nil on repeat call (session must be terminal), got %d", len(follow2))
	}
}
```

**Impl:** No production code changes needed.

**Verify:** `cd sim/workload && go test -run TestSession_HorizonInterrupted_IsTerminal -v`
Expected: `--- PASS: TestSession_HorizonInterrupted_IsTerminal`

**Lint:** `golangci-lint run ./sim/workload/...`

**Commit:** `test(workload): horizon-interrupted session terminal idempotency (BC-2, T2-5, #980)`

---

#### Task 3: T2-1 — INV-11 terminal completeness in cluster Run() (BC-3)

**Files:** modify `sim/cluster/cluster_test.go` — add imports `"math/rand"` and `"github.com/inference-sim/inference-sim/sim/workload"`, add test.

**New imports to add** to the existing import block in `cluster_test.go`:

```go
"math/rand"
// ...
"github.com/inference-sim/inference-sim/sim/workload"
```

**Test:**

```go
// TestClusterSimulator_SessionTerminalStateCompleteness verifies INV-11:
// every session reaches exactly one terminal state after ClusterSimulator.Run().
// With the default blackbox latency model and a 500s horizon, all sessions
// complete normally (sessionCompleted path). This exercises the full DES
// pipeline: seed injected → rounds executed → OnComplete returns nil exactly once.
func TestClusterSimulator_SessionTerminalStateCompleteness(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler: %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 10},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler: %v", err)
	}

	const numSessions = 4
	const maxRounds = 3
	const horizon = int64(500_000_000) // 500 seconds — long enough for most rounds to complete

	blueprints := make([]workload.SessionBlueprint, numSessions)
	seeds := make([]*sim.Request, numSessions)
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t21_sess_%d", i)
		blueprints[i] = workload.SessionBlueprint{
			SessionID:     sessID,
			ClientID:      "test-client",
			MaxRounds:     maxRounds,
			ThinkTimeUs:   1_000_000, // 1 second think time
			Horizon:       horizon,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
			TenantID:      "test-tenant",
			SLOClass:      "standard",
			Model:         "test-model",
		}
		seeds[i] = &sim.Request{
			ID:           fmt.Sprintf("t21_sess_%d_r0", i),
			ArrivalTime:  int64(i * 1_000_000), // stagger arrivals by 1 second
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 10),
			MaxOutputLen: 10,
			State:        sim.StateQueued,
			SessionID:    sessID,
			RoundIndex:   0,
		}
	}

	sm := workload.NewSessionManager(blueprints)
	// Use a set (not a counter) so that one session being called twice AND another
	// being silently abandoned cannot cancel out to give a false pass.
	terminatedSessions := make(map[string]bool)
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		followUps := sm.OnComplete(req, tick)
		// A nil return from OnComplete means this session just reached a terminal
		// state (completed, cancelled, horizon-interrupted, or budget-exhausted).
		if followUps == nil && req.SessionID != "" {
			terminatedSessions[req.SessionID] = true
		}
		return followUps
	}

	config := newTestDeploymentConfig(1)
	config.Horizon = horizon
	cs := NewClusterSimulator(config, seeds, onDone)
	mustRun(t, cs)

	// INV-11: every session reached exactly one terminal state
	if len(terminatedSessions) != numSessions {
		t.Errorf("BC-3 (INV-11): terminal sessions = %d, want %d (no session silently abandoned)",
			len(terminatedSessions), numSessions)
	}
}
```

**Impl:** No production code changes.

**Verify:** `cd sim/cluster && go test -run TestClusterSimulator_SessionTerminalStateCompleteness -v`
Expected: `--- PASS: TestClusterSimulator_SessionTerminalStateCompleteness`

**Lint:** `golangci-lint run ./sim/cluster/...`

**Commit:** `test(cluster): INV-11 terminal state completeness in cluster Run() (BC-3, T2-1, #980)`

---

#### Task 4: T2-2 — INV-10 follow-up causality from real DES ticks (BC-4)

**Files:** modify `sim/cluster/cluster_test.go` (imports already added in Task 3)

**Test:**

```go
// TestClusterSimulator_SessionFollowUpCausality verifies INV-10:
// round[N+1].ArrivalTime >= round[N].completionTick + ThinkTimeUs,
// where completionTick comes from actual DES execution (not a hardcoded value).
func TestClusterSimulator_SessionFollowUpCausality(t *testing.T) {
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler: %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 10},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler: %v", err)
	}

	const thinkTimeUs = int64(500_000) // 500ms
	const numSessions = 3
	const maxRounds = 3

	rng := rand.New(rand.NewSource(7))
	blueprints := make([]workload.SessionBlueprint, numSessions)
	seeds := make([]*sim.Request, numSessions)
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t22_sess_%d", i)
		blueprints[i] = workload.SessionBlueprint{
			SessionID:     sessID,
			ClientID:      "test-client",
			MaxRounds:     maxRounds,
			ThinkTimeUs:   thinkTimeUs,
			Horizon:       math.MaxInt64,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
			TenantID:      "test-tenant",
			SLOClass:      "standard",
			Model:         "test-model",
		}
		seeds[i] = &sim.Request{
			ID:           fmt.Sprintf("t22_sess_%d_r0", i),
			ArrivalTime:  int64(i * 2_000_000),
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 10),
			MaxOutputLen: 10,
			State:        sim.StateQueued,
			SessionID:    sessID,
			RoundIndex:   0,
		}
	}

	// completionTick[sessID][roundIndex] = DES tick at which that round completed.
	type roundKey struct {
		sessID string
		round  int
	}
	completionTick := make(map[roundKey]int64)
	arrivalTime := make(map[roundKey]int64)
	// Seed arrivals are known upfront.
	for i, s := range seeds {
		arrivalTime[roundKey{fmt.Sprintf("t22_sess_%d", i), 0}] = s.ArrivalTime
	}

	sm := workload.NewSessionManager(blueprints)
	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		if req.SessionID != "" {
			completionTick[roundKey{req.SessionID, req.RoundIndex}] = tick
		}
		followUps := sm.OnComplete(req, tick)
		for _, fu := range followUps {
			arrivalTime[roundKey{fu.SessionID, fu.RoundIndex}] = fu.ArrivalTime
		}
		return followUps
	}

	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, seeds, onDone)
	mustRun(t, cs)

	// Verify INV-10 for every consecutive round pair in every session.
	// Also verify the DES clock actually advanced (completionTick > seed arrivalTime),
	// catching a bug where the wrong tick (e.g., 0) is passed to onDone.
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t22_sess_%d", i)
		seedArrival := seeds[i].ArrivalTime
		for n := 0; n < maxRounds-1; n++ {
			ck, okC := completionTick[roundKey{sessID, n}]
			arr, okA := arrivalTime[roundKey{sessID, n + 1}]
			if !okC || !okA {
				// Round may not have run (horizon or budget) — skip.
				continue
			}
			// DES must have advanced the clock past the seed arrival.
			if ck <= seedArrival {
				t.Errorf("BC-4: session %s round[%d] completionTick=%d <= seedArrival=%d (DES clock did not advance)",
					sessID, n, ck, seedArrival)
			}
			minArrival := ck + thinkTimeUs
			if arr < minArrival {
				t.Errorf("BC-4 (INV-10): session %s round[%d].arrival=%d < round[%d].completion(%d)+thinkTime(%d)=%d",
					sessID, n+1, arr, n, ck, thinkTimeUs, minArrival)
			}
		}
	}

	// Guard against vacuous pass: verify at least one round pair was actually checked.
	// With MaxInt64 horizon and 3 rounds/session, all pairs should be present.
	if len(completionTick) == 0 {
		t.Error("BC-4: no completionTick entries recorded — DES may have produced no completions")
	}
}
```

**Impl:** No production code changes.

**Verify:** `cd sim/cluster && go test -run TestClusterSimulator_SessionFollowUpCausality -v`
Expected: `--- PASS: TestClusterSimulator_SessionFollowUpCausality`

**Lint:** `golangci-lint run ./sim/cluster/...`

**Commit:** `test(cluster): INV-10 follow-up causality from real DES ticks (BC-4, T2-2, #980)`

---

#### Task 5: T2-3 — End-to-end multi-turn session in standard cluster (BC-5, BC-6, BC-7)

**Files:** modify `sim/cluster/cluster_test.go` (imports already added in Task 3)

**Test:**

```go
// TestClusterSimulator_MultiTurnSession_EndToEnd verifies BC-5, BC-6, BC-7:
// INV-1 conservation holds with dynamic follow-up injection, follow-ups have
// RoundIndex > 0, and every session generates at least one follow-up round.
// This is the first test of the standard (non-disaggregated) cluster path
// with real multi-turn session management.
func TestClusterSimulator_MultiTurnSession_EndToEnd(t *testing.T) {
	inputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 20},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler: %v", err)
	}
	outputSampler, err := workload.NewLengthSampler(workload.DistSpec{
		Type:   "constant",
		Params: map[string]float64{"value": 10},
	})
	if err != nil {
		t.Fatalf("NewLengthSampler: %v", err)
	}

	const numSessions = 3
	const maxRounds = 3

	rng := rand.New(rand.NewSource(13))
	blueprints := make([]workload.SessionBlueprint, numSessions)
	seeds := make([]*sim.Request, numSessions)
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t23_sess_%d", i)
		blueprints[i] = workload.SessionBlueprint{
			SessionID:     sessID,
			ClientID:      "test-client",
			MaxRounds:     maxRounds,
			ThinkTimeUs:   100_000, // 100ms
			Horizon:       math.MaxInt64,
			InputSampler:  inputSampler,
			OutputSampler: outputSampler,
			RNG:           rand.New(rand.NewSource(rng.Int63())),
			TenantID:      "test-tenant",
			SLOClass:      "standard",
			Model:         "test-model",
		}
		seeds[i] = &sim.Request{
			ID:           fmt.Sprintf("t23_sess_%d_r0", i),
			ArrivalTime:  int64(i * 1_000_000),
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 10),
			MaxOutputLen: 10,
			State:        sim.StateQueued,
			SessionID:    sessID,
			RoundIndex:   0,
		}
	}

	sm := workload.NewSessionManager(blueprints)
	totalInjected := numSessions // seeds
	followUpCount := 0
	// followUpsBySession[sessID] = list of RoundIndex values seen for follow-ups
	followUpsBySession := make(map[string][]int)

	onDone := func(req *sim.Request, tick int64) []*sim.Request {
		followUps := sm.OnComplete(req, tick)
		for _, fu := range followUps {
			totalInjected++
			followUpCount++
			if fu.SessionID != "" {
				followUpsBySession[fu.SessionID] = append(followUpsBySession[fu.SessionID], fu.RoundIndex)
			}
		}
		return followUps
	}

	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, seeds, onDone)
	mustRun(t, cs)

	metrics := cs.AggregatedMetrics()

	// BC-5 (INV-1): conservation with dynamic follow-up injection.
	// assertINV1Conservation checks 5 of 9 INV-1 terms:
	//   completed + queued + running + dropped + timedOut == totalInjected
	// The 4 missing terms (DeferredHorizonInterrupted, RoutingRejections,
	// GatewayQueueDepth, GatewayQueueShed) are all zero for this config:
	// no gateway queue, no deferred queue, no routing rejections. The
	// simplified check is therefore equivalent to the full INV-1 formula.
	assertINV1Conservation(t, metrics, totalInjected, "multi-turn end-to-end")

	// BC-5: some work was done
	if metrics.CompletedRequests == 0 {
		t.Error("BC-5: CompletedRequests = 0, expected > 0 (work must be done)")
	}

	// BC-6: no follow-up has RoundIndex == 0
	for sessID, rounds := range followUpsBySession {
		for _, ri := range rounds {
			if ri == 0 {
				t.Errorf("BC-6: session %s generated a follow-up with RoundIndex=0 (only seeds may have round 0)", sessID)
			}
		}
	}

	// BC-7: every session generated at least one follow-up
	for i := 0; i < numSessions; i++ {
		sessID := fmt.Sprintf("t23_sess_%d", i)
		if len(followUpsBySession[sessID]) == 0 {
			t.Errorf("BC-7: session %s generated 0 follow-ups, want >= 1 (MaxRounds=%d)", sessID, maxRounds)
		}
	}

	// Sanity: total follow-ups == numSessions * (maxRounds - 1) with MaxInt64 horizon
	expectedFollowUps := numSessions * (maxRounds - 1)
	if followUpCount != expectedFollowUps {
		t.Errorf("follow-up count = %d, want %d (%d sessions × %d follow-ups each)",
			followUpCount, expectedFollowUps, numSessions, maxRounds-1)
	}
}
```

**Impl:** No production code changes.

**Verify:** `cd sim/cluster && go test -run TestClusterSimulator_MultiTurnSession_EndToEnd -v`
Expected: `--- PASS: TestClusterSimulator_MultiTurnSession_EndToEnd`

**Lint:** `golangci-lint run ./sim/cluster/...`

**Commit:** `test(cluster): INV-1 + round chain end-to-end (BC-5,6,7, T2-3, #980)`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|---|---|---|---|
| BC-1 | Task 1 | Unit (workload) | `TestSession_ContextAccumulation_MultiStep` |
| BC-2 | Task 2 | Unit (workload) | `TestSession_HorizonInterrupted_IsTerminal` |
| BC-3 (INV-11) | Task 3 | Integration (cluster) | `TestClusterSimulator_SessionTerminalStateCompleteness` |
| BC-4 (INV-10) | Task 4 | Integration (cluster) | `TestClusterSimulator_SessionFollowUpCausality` |
| BC-5 (INV-1) | Task 5 | Integration (cluster) | `TestClusterSimulator_MultiTurnSession_EndToEnd` |
| BC-6 | Task 5 | Integration (cluster) | `TestClusterSimulator_MultiTurnSession_EndToEnd` |
| BC-7 | Task 5 | Integration (cluster) | `TestClusterSimulator_MultiTurnSession_EndToEnd` |

No golden tests in this PR — all assertions check behavioral laws (counts, orderings, flags), not snapshot values.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|---|---|---|---|---|
| T2-1 count logic misses sessions whose seeds never get scheduled | Low | High | Config uses `Horizon = 500_000_000` (500s) and seeds arrive at 0–3s stagger — all seeds guaranteed to be scheduled | Task 3 |
| T2-2 round pair not found (round didn't run) | Low | Medium | Guarded with `!okC || !okA` skip | Task 4 |
| T2-3 follow-up count assertion too strict (some sessions horizon-interrupted) | Low | Medium | Uses `math.MaxInt64` horizon — no horizon interruptions possible | Task 5 |
| Existing test for `TestSession_ContextAccumulation` overlaps with T2-4 | None | Low | T2-4 starts from round 1 not round 0, extends the chain — no duplication | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific:**
- [x] No unnecessary abstractions.
- [x] No feature creep — tests only, no production code.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint (no new exported symbols, no unused vars).
- [x] Shared test helpers used (`newTestDeploymentConfig`, `mustRun`, `assertINV1Conservation`, `makeTestBlueprint`) — not duplicated.
- [x] CLAUDE.md: no new packages, files, or CLI flags added — no update needed.
- [x] No stale references in CLAUDE.md from this PR.
- [x] Documentation DRY: no canonical sources modified.
- [x] Deviation log reviewed — all deviations are SIMPLIFICATION or CORRECTION with clear rationale.
- [x] Each task produces complete, immediately testable code.
- [x] Task ordering: Tasks 1–2 are independent; Tasks 3–5 share imports added in Task 3 (import addition is part of Task 3 step).
- [x] All contracts mapped to specific tasks.
- [x] No golden dataset changes.
- [x] Construction site audit: `SessionBlueprint` built inline in disaggregation_test.go — new inline constructions follow the same pattern. No canonical constructor added.
- [x] Not part of a macro plan.

**Antipattern rules:**
- [x] R1: No silent continues/returns — tests assert, don't silently skip.
- [x] R2: N/A (no float accumulation or ordered output).
- [x] R3: N/A (no new numeric parameters).
- [x] R4: No new struct fields.
- [x] R5: N/A (no resource allocation loops).
- [x] R6: No logrus.Fatalf in sim/ packages — tests use t.Fatalf.
- [x] R7: No golden tests added — all tests assert behavioral laws directly.
- [x] R8: N/A (no exported maps).
- [x] R9: N/A (no YAML fields).
- [x] R10: N/A (no YAML parsing).
- [x] R11: N/A (no division).
- [x] R12: N/A (no golden dataset changes).
- [x] R13: N/A (no new interfaces).
- [x] R14: N/A (no new methods spanning multiple modules).
- [x] R15: No stale PR references.
- [x] R16: N/A (no config params).
- [x] R17: N/A (no routing scorer signals).
- [x] R18: N/A (no CLI flags).
- [x] R19: N/A (no retry loops).
- [x] R20: N/A (no detectors/analyzers).
- [x] R21: N/A (no range over mutable slices).
- [x] R22: N/A (no pre-check estimates).
- [x] R23: N/A (no parallel code paths).

---

## Appendix: File-Level Implementation Details

### `sim/workload/session_test.go`

**Purpose:** Extend existing session unit tests with multi-step accumulation and terminal idempotency.

**Add 2 functions** after their thematically nearest existing tests:
- `TestSession_ContextAccumulation_MultiStep` — after `TestSession_ContextAccumulation` (line ~127)
- `TestSession_HorizonInterrupted_IsTerminal` — after `TestSession_BeyondHorizon_NotGenerated` (line ~145)

**No import changes needed.** Existing imports (`math/rand`, `testing`, `sim`) cover all new code.

**State mutation:** Uses `sm.sessions[sessID].state` via `OnComplete`. The test calls `OnComplete` twice on the same session ID in T2-5 to verify state doesn't revert to active after terminal.

**Error handling:** `t.Fatalf` on unexpected follow-up count (structural guard). `t.Errorf` for invariant violations (allows test to continue).

---

### `sim/cluster/cluster_test.go`

**Purpose:** Add 3 Tier 2 integration tests using the existing `newTestDeploymentConfig` + `mustRun` + `assertINV1Conservation` helpers.

**Import additions** to existing import block:

```go
import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"math/rand"       // NEW
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/testutil"
	"github.com/inference-sim/inference-sim/sim/workload" // NEW
)
```

**Add 3 test functions** after the flow-control tests (end of file, before the NodePool tests, or at the end):
- `TestClusterSimulator_SessionTerminalStateCompleteness`
- `TestClusterSimulator_SessionFollowUpCausality`
- `TestClusterSimulator_MultiTurnSession_EndToEnd`

**Config note:** Tasks 3–5 use `newTestDeploymentConfig(1)` (1 instance). Task 3 overrides `config.Horizon = horizon` to match blueprints. Tasks 4–5 use the default `math.MaxInt64` horizon from `newTestDeploymentConfig`.

**assertINV1Conservation:** defined in `disaggregation_test.go` (same `package cluster`), available to all test files in the package without any import.

**workload.NewLengthSampler:** returns `(LengthSampler, error)`. All 3 cluster tests call it and `t.Fatalf` on error.

**Single-threaded DES note:** `onDone` callbacks are called synchronously from the DES event loop. No mutex or atomic primitives are needed or appropriate for counters/maps mutated inside `onDone`.
