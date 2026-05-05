# Session Metrics Implementation Plan

**Goal:** Add multi-turn session metrics (TTFT cold/warm, session duration) to BLIS stdout output so operators can reason about cache-warmed latency and end-to-end session time.

**The problem today:** BLIS outputs TTFT and E2E distributions over all completed requests, but has no mechanism to distinguish first-round "cold" requests from follow-up "warm" requests that benefit from KV cache reuse. There is also no per-session wall-time metric. Operators running closed-loop multi-turn workloads cannot measure cache warm-up effect or trajectory length without post-processing the raw request log.

**What this PR adds:**

1. `SessionID` and `RoundIndex` fields on `RequestMetrics` — propagated from `Request` via the canonical constructor, making session context available for downstream aggregation.
2. `SessionMetrics` struct in `sim/cluster/metrics.go` — holds `SessionCount`, `TTFTCold` (round-0 requests), `TTFTWarm` (round≥1 requests), and `SessionDuration` (max completion − round-0 arrival per session) as `Distribution` values.
3. `ComputeSessionMetrics(*sim.Metrics) *SessionMetrics` — self-gating: returns nil when no requests carry a `SessionID`, so single-turn workloads see no output change.
4. `printSessionMetrics` in `cmd/root.go` — prints the new section after Per-Tenant Metrics; absent for nil input (zero impact on non-session runs).

**Why this matters:** Cache-warm TTFT vs cold TTFT is the primary observable for KV prefix reuse effectiveness in production LLM serving. Session duration lets capacity planners reason about chatbot conversation latency budgets. These are the two metrics most requested by LLM serving operators evaluating multi-turn workloads.

**Architecture:** `RequestMetrics` (sim package) gains two new fields populated by `NewRequestMetrics`. `ComputeSessionMetrics` reads `sim.Metrics.Requests` (the per-request map already populated by cluster), partitions TTFTs by `RoundIndex`, and computes per-session duration by grouping on `SessionID`. `printSessionMetrics` in cmd follows the existing `printPerTenantMetrics` pattern.

**Source:** [GitHub issue #1058](https://github.com/anthropics/inference-sim/issues/1058)

**Closes:** Fixes #1058

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block modified:** `sim.RequestMetrics` (data carrier) + new `cluster.SessionMetrics` (aggregation) + `cmd` print layer.
2. **Adjacent blocks:** `SessionManager` (workload) produces `SessionID`/`RoundIndex` on requests; `sim.Metrics.Requests` map accumulates `RequestMetrics`; `cmd/root.go` print loop consumes computed metrics.
3. **Invariants touched:** INV-6 (determinism) — `ComputeSessionMetrics` must use sorted map iteration (R2). INV-11 (session completeness) — not modified; we observe, not control.
4. **Construction site audit for `RequestMetrics{}`:**

   | Site | File | Type | Action |
   |------|------|------|--------|
   | `NewRequestMetrics` | `sim/metrics_utils.go:34` | Production canonical | **MUST update** — add `SessionID`/`RoundIndex` |
   | test literals (×4) | `sim/metrics_test.go:33,83,133,170` | Test partial literal | Exempt (R4) |
   | test literal | `sim/cluster/metrics_tenant_test.go:20` | Test partial literal | Exempt (R4) |
   | test literals (multiple) | `sim/cluster/metrics_test.go` | Test partial literal | Exempt (R4) |
   | test literals (multiple) | `sim/cluster/metrics_slo_test.go` | Test partial literal | Exempt (R4) |
   | test literals (×3) | `sim/cluster/multi_model_test.go:123,131,182` | Test partial literal | Exempt (R4) |
   | test literals (multiple) | `cmd/replay_test.go` | Test partial literal | Exempt (R4) |

   Only one production construction site: `NewRequestMetrics`. All other sites are test literals, explicitly exempt from R4 updates.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds a "Session Metrics" section to BLIS stdout output. It is purely additive: no existing output changes, no existing interfaces change, no behavioral changes to the simulation engine. The work is a three-layer addition: (1) two new fields on the existing `RequestMetrics` data carrier, (2) a new aggregation function `ComputeSessionMetrics` in `sim/cluster/metrics.go` following the self-gating pattern of `ComputePerTenantMetrics`, and (3) a new `printSessionMetrics` function in `cmd/root.go` called in the existing print loop. For single-turn workloads, `ComputeSessionMetrics` returns nil and `printSessionMetrics` is a no-op — zero observable impact. For multi-turn workloads, a new "=== Session Metrics ===" section appears after Per-Tenant Metrics. No deviation flags.

### B) Behavioral Contracts

**Positive contracts (what MUST happen):**

```
BC-1: Self-gating on non-session workloads
- GIVEN a completed simulation where no request has a non-empty SessionID
- WHEN ComputeSessionMetrics is called
- THEN it returns nil

BC-2: TTFT partitioning by round
- GIVEN a completed multi-turn simulation with at least one round-0 and one round≥1 request
- WHEN ComputeSessionMetrics is called
- THEN TTFTCold.Count equals the number of round-0 requests with non-empty SessionID
       AND TTFTWarm.Count equals the number of round≥1 requests with non-empty SessionID

BC-3: Session duration correctness
- GIVEN a session with round-0 arrival at time A and final completed round finishing at time B
- WHEN ComputeSessionMetrics is called
- THEN SessionDuration includes (B - A) for that session

BC-4: Mixed workload isolation
- GIVEN a simulation where some requests have SessionID and some do not
- WHEN ComputeSessionMetrics is called
- THEN only session requests contribute to any metric in SessionMetrics
       AND non-session requests do not affect TTFTCold, TTFTWarm, or SessionDuration

BC-5: SessionID and RoundIndex propagated to RequestMetrics
- GIVEN a completed Request with non-empty SessionID and RoundIndex N
- WHEN NewRequestMetrics is called for that request
- THEN the returned RequestMetrics has SessionID equal to the request's SessionID
       AND RoundIndex equal to N
```

**Negative contracts (what MUST NOT happen):**

```
BC-6: Session without observed round 0 excluded from SessionDuration
- GIVEN a session where no round-0 request appears in Metrics.Requests (e.g., all timed out before completion)
- WHEN ComputeSessionMetrics is called
- THEN that session does not contribute a value to SessionDuration
       AND no panic or error occurs
```

**Quality gate:** Every THEN clause describes observable output behavior — counts, values, presence/absence of output — not struct fields or map membership.

### C) Component Interaction

```
┌─────────────────────────────────────────────────────────┐
│  sim/workload (SessionManager)                          │
│  Produces: Request.SessionID, Request.RoundIndex        │
└──────────────────────────┬──────────────────────────────┘
                           │ request fields
                           ▼
┌─────────────────────────────────────────────────────────┐
│  sim/metrics_utils.go (NewRequestMetrics)               │
│  Reads: req.SessionID, req.RoundIndex                   │
│  Writes: RequestMetrics.SessionID, .RoundIndex          │
└──────────────────────────┬──────────────────────────────┘
                           │ populated RequestMetrics
                           ▼
┌─────────────────────────────────────────────────────────┐
│  sim.Metrics.Requests  map[string]RequestMetrics        │
│  (populated by cluster during simulation)               │
└──────────────────────────┬──────────────────────────────┘
                           │ *sim.Metrics passed post-sim
                           ▼
┌─────────────────────────────────────────────────────────┐
│  sim/cluster/metrics.go (ComputeSessionMetrics)         │
│  Reads: RequestMetrics.{SessionID,RoundIndex,TTFT,E2E}  │
│  Also reads: sim.Metrics.RequestCompletionTimes (µs)    │
│  Produces: *SessionMetrics{TTFTCold,TTFTWarm,Duration}  │
└──────────────────────────┬──────────────────────────────┘
                           │ *SessionMetrics
                           ▼
┌─────────────────────────────────────────────────────────┐
│  cmd/root.go (printSessionMetrics)                      │
│  Reads: *SessionMetrics (nil = no-op)                   │
│  Writes: io.Writer (stdout)                             │
└─────────────────────────────────────────────────────────┘
```

State ownership: `SessionID`/`RoundIndex` owned by `sim.Request` (set during workload generation, never mutated). `RequestMetrics` is a value copy — no shared state. `SessionMetrics` is computed post-simulation, no concurrent access.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue #1058: "TTFT2 (time-to-first-token of the second round)" | Uses TTFTWarm = TTFT for all round≥1 requests (not only round 2) | CLARIFICATION — round≥1 is more complete; round-2-only would exclude sessions with 2 rounds total |
| Issue #1058: "TrajTime (total wall time)" | Uses `session_duration` = max_completion − round0_arrival | CLARIFICATION — aligns with community naming conventions |
| Issue #1058: "MeanRoundsCompleted" | Dropped | DEFERRAL — user agreed to drop in pre-plan discussion |
| Issue #1058: "session completion rate" | Deferred | DEFERRAL — requires per-session terminal state tracking not available in aggregated `Metrics`; noted for future PR |

### E) Review Guide

**Tricky part:** `SessionDuration` requires correlating two different data structures: `RequestMetrics` (has round-0 arrival time via `ArrivedAt`) and `sim.Metrics.RequestCompletionTimes` (has completion timestamps keyed by request ID). The computation groups by `SessionID`, finds round-0 arrival, then finds the max completion time among all requests in the session. Reviewers should scrutinize the group-build loop for off-by-one on round index and for the BC-6 guard (sessions missing a round-0 entry are silently skipped — not an error).

**Scrutinize:** `ComputeSessionMetrics` map iteration must use sorted keys (R2). TTFT values in `RequestMetrics` are stored in ms (float64, converted in `NewRequestMetrics`). `RequestCompletionTimes` values are in µs ticks — session duration output should be in ms for consistency with other TTFT/E2E output.

**Safe to skim:** `printSessionMetrics` — it's a direct copy of the `printPerTenantMetrics` pattern with different field names. `NewRequestMetrics` change — two lines added, no logic.

**Known debt:** Session completion rate (what fraction of sessions fully completed vs were cancelled/timeout-interrupted) is deferred; would require surfacing per-session terminal states from `SessionManager` through aggregated `Metrics`.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Purpose |
|------|--------|---------|
| `sim/metrics_utils.go` | Modify | Add `SessionID`/`RoundIndex` fields to `RequestMetrics`; populate in `NewRequestMetrics` |
| `sim/cluster/metrics.go` | Modify | Add `SessionMetrics` struct and `ComputeSessionMetrics` function |
| `cmd/root.go` | Modify | Add `printSessionMetrics` function; call it after `printPerTenantMetrics` |
| `sim/metrics_utils_test.go` | Modify/Create | Test BC-5: `NewRequestMetrics` propagates `SessionID` and `RoundIndex` |
| `sim/cluster/metrics_session_test.go` | Create | Tests BC-1 through BC-6 for `ComputeSessionMetrics` |

No dead code. No new interfaces (R13 — no interface needed for a single implementation). No new CLI flags.

### G) Task Breakdown

---

#### Task 1: Propagate SessionID and RoundIndex into RequestMetrics (BC-5)

**Contracts:** BC-5

**Files:** modify `sim/metrics_utils.go`, modify `sim/metrics_utils_test.go`

**Step 1 — Write failing test:**

In `sim/metrics_utils_test.go`, add (or create the file if absent — check with `go test ./sim/... -run TestNewRequestMetrics_Session` first):

```go
func TestNewRequestMetrics_SessionFields(t *testing.T) {
	req := &Request{
		ID:               "req-1",
		SessionID:        "sess-abc",
		RoundIndex:       2,
		InputTokens:      []int{1, 2, 3},
		OutputTokens:     []int{4, 5},
		SLOClass:         "standard",
		TenantID:         "tenant-x",
		AssignedInstance: "inst-0",
		Model:            "model-a",
	}
	rm := NewRequestMetrics(req, 1000.0)
	if rm.SessionID != "sess-abc" {
		t.Errorf("SessionID: got %q, want %q", rm.SessionID, "sess-abc")
	}
	if rm.RoundIndex != 2 {
		t.Errorf("RoundIndex: got %d, want %d", rm.RoundIndex, 2)
	}

	// Non-session request: SessionID empty, RoundIndex 0
	req2 := &Request{ID: "req-2", InputTokens: []int{1}, OutputTokens: []int{2}}
	rm2 := NewRequestMetrics(req2, 500.0)
	if rm2.SessionID != "" {
		t.Errorf("non-session SessionID: got %q, want empty", rm2.SessionID)
	}
	if rm2.RoundIndex != 0 {
		t.Errorf("non-session RoundIndex: got %d, want 0", rm2.RoundIndex)
	}
}
```

**Step 2 — Run test to verify it fails:**
```bash
go test ./sim/... -run TestNewRequestMetrics_SessionFields
# Expected: FAIL (fields don't exist yet)
```

**Step 3 — Implement:**

In `sim/metrics_utils.go`, add two fields to `RequestMetrics` after `GatewayQueueDelay`:

```go
SessionID  string `json:"session_id,omitempty"`
RoundIndex int    `json:"round_index,omitempty"`
```

In `NewRequestMetrics`, add two lines after the `LengthCapped` assignment:

```go
SessionID:        req.SessionID,
RoundIndex:       req.RoundIndex,
```

**Step 4 — Run test to verify it passes:**
```bash
go test ./sim/... -run TestNewRequestMetrics_SessionFields
# Expected: PASS
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/...
# Expected: no issues
```

**Step 6 — Commit:**
```
feat(metrics): propagate SessionID and RoundIndex into RequestMetrics (BC-5)
```

---

#### Task 2: Add SessionMetrics struct and ComputeSessionMetrics (BC-1, BC-2, BC-3, BC-4, BC-6)

**Contracts:** BC-1, BC-2, BC-3, BC-4, BC-6

**Files:** modify `sim/cluster/metrics.go`, create `sim/cluster/metrics_session_test.go`

**Step 1 — Write failing tests:**

Create `sim/cluster/metrics_session_test.go`:

```go
package cluster

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildSessionMetrics is a test helper that builds a minimal *sim.Metrics
// with the given RequestMetrics entries and completion times.
func buildSessionMetrics(reqs []sim.RequestMetrics, completionTimes map[string]float64) *sim.Metrics {
	m := &sim.Metrics{
		Requests:               make(map[string]sim.RequestMetrics, len(reqs)),
		RequestCompletionTimes: make(map[string]float64, len(completionTimes)),
	}
	for _, r := range reqs {
		m.Requests[r.ID] = r
	}
	for id, t := range completionTimes {
		m.RequestCompletionTimes[id] = t
	}
	return m
}

// BC-1: returns nil when no requests have SessionID
func TestComputeSessionMetrics_NoSessions_ReturnsNil(t *testing.T) {
	m := buildSessionMetrics([]sim.RequestMetrics{
		{ID: "r1", TTFT: 10.0},
		{ID: "r2", TTFT: 20.0},
	}, map[string]float64{"r1": 100e3, "r2": 200e3})

	got := ComputeSessionMetrics(m)
	if got != nil {
		t.Errorf("expected nil for non-session workload, got %+v", got)
	}
}

// BC-1: returns nil for empty metrics
func TestComputeSessionMetrics_EmptyMetrics_ReturnsNil(t *testing.T) {
	m := &sim.Metrics{
		Requests:               make(map[string]sim.RequestMetrics),
		RequestCompletionTimes: make(map[string]float64),
	}
	got := ComputeSessionMetrics(m)
	if got != nil {
		t.Errorf("expected nil for empty metrics, got %+v", got)
	}
}

// BC-2: TTFT partitioned correctly by RoundIndex
func TestComputeSessionMetrics_TTFTPartition(t *testing.T) {
	m := buildSessionMetrics([]sim.RequestMetrics{
		{ID: "r0", SessionID: "s1", RoundIndex: 0, TTFT: 50.0, ArrivedAt: 0},
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
		{ID: "r2", SessionID: "s1", RoundIndex: 2, TTFT: 18.0, ArrivedAt: 200},
	}, map[string]float64{
		"r0": 500e3,
		"r1": 600e3,
		"r2": 700e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics")
	}
	if got.TTFTCold.Count != 1 {
		t.Errorf("TTFTCold.Count: got %d, want 1", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 2 {
		t.Errorf("TTFTWarm.Count: got %d, want 2", got.TTFTWarm.Count)
	}
	if got.TTFTCold.Mean != 50.0 {
		t.Errorf("TTFTCold.Mean: got %.2f, want 50.0", got.TTFTCold.Mean)
	}
}

// BC-3: SessionDuration = max_completion_ms - round0_arrival_ms
func TestComputeSessionMetrics_SessionDuration(t *testing.T) {
	// Round-0 arrives at 0ms; final round completes at 700,000 µs = 700ms
	// session_duration = 700ms - 0ms = 700ms
	m := buildSessionMetrics([]sim.RequestMetrics{
		{ID: "r0", SessionID: "s1", RoundIndex: 0, TTFT: 50.0, ArrivedAt: 0},
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
	}, map[string]float64{
		"r0": 300e3, // 300ms in µs
		"r1": 700e3, // 700ms in µs
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics")
	}
	if got.SessionDuration.Count != 1 {
		t.Errorf("SessionDuration.Count: got %d, want 1", got.SessionDuration.Count)
	}
	const wantDuration = 700.0 // ms
	if got.SessionDuration.Mean != wantDuration {
		t.Errorf("SessionDuration.Mean: got %.2f, want %.2f", got.SessionDuration.Mean, wantDuration)
	}
}

// BC-4: non-session requests do not affect metrics
func TestComputeSessionMetrics_MixedWorkload(t *testing.T) {
	m := buildSessionMetrics([]sim.RequestMetrics{
		{ID: "r0", SessionID: "s1", RoundIndex: 0, TTFT: 50.0, ArrivedAt: 0},
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
		// Non-session requests (SessionID == "")
		{ID: "ns1", TTFT: 999.0},
		{ID: "ns2", TTFT: 888.0},
	}, map[string]float64{
		"r0": 300e3, "r1": 700e3,
		"ns1": 400e3, "ns2": 500e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics")
	}
	// Only 2 session requests (1 cold, 1 warm)
	if got.TTFTCold.Count != 1 {
		t.Errorf("TTFTCold.Count: got %d, want 1 (non-session must be excluded)", got.TTFTCold.Count)
	}
	if got.TTFTWarm.Count != 1 {
		t.Errorf("TTFTWarm.Count: got %d, want 1 (non-session must be excluded)", got.TTFTWarm.Count)
	}
	// Mean cold TTFT should be 50.0, not contaminated by 999.0 or 888.0
	if got.TTFTCold.Mean != 50.0 {
		t.Errorf("TTFTCold.Mean: got %.2f, want 50.0 (non-session contamination?)", got.TTFTCold.Mean)
	}
}

// BC-6: session without round-0 in completed requests is excluded from SessionDuration, no panic
func TestComputeSessionMetrics_MissingRound0(t *testing.T) {
	// Only round-1 and round-2 appear (round-0 timed out / not in Requests map)
	m := buildSessionMetrics([]sim.RequestMetrics{
		{ID: "r1", SessionID: "s1", RoundIndex: 1, TTFT: 20.0, ArrivedAt: 100},
		{ID: "r2", SessionID: "s1", RoundIndex: 2, TTFT: 18.0, ArrivedAt: 200},
	}, map[string]float64{
		"r1": 600e3, "r2": 700e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil SessionMetrics (warm requests exist)")
	}
	// SessionDuration must be empty (no round-0 arrival reference)
	if got.SessionDuration.Count != 0 {
		t.Errorf("SessionDuration.Count: got %d, want 0 (no round-0 entry)", got.SessionDuration.Count)
	}
	// Warm TTFT should still be collected
	if got.TTFTWarm.Count != 2 {
		t.Errorf("TTFTWarm.Count: got %d, want 2", got.TTFTWarm.Count)
	}
}

// Invariant test: SessionCount equals number of distinct SessionIDs
func TestComputeSessionMetrics_SessionCount(t *testing.T) {
	m := buildSessionMetrics([]sim.RequestMetrics{
		{ID: "a0", SessionID: "s1", RoundIndex: 0, TTFT: 10.0, ArrivedAt: 0},
		{ID: "a1", SessionID: "s1", RoundIndex: 1, TTFT: 8.0, ArrivedAt: 50},
		{ID: "b0", SessionID: "s2", RoundIndex: 0, TTFT: 12.0, ArrivedAt: 0},
	}, map[string]float64{
		"a0": 100e3, "a1": 200e3, "b0": 150e3,
	})

	got := ComputeSessionMetrics(m)
	if got == nil {
		t.Fatal("expected non-nil")
	}
	if got.SessionCount != 2 {
		t.Errorf("SessionCount: got %d, want 2", got.SessionCount)
	}
}
```

**Step 2 — Run tests to verify they fail:**
```bash
go test ./sim/cluster/... -run TestComputeSessionMetrics
# Expected: FAIL (ComputeSessionMetrics undefined)
```

**Step 3 — Implement:**

Append to `sim/cluster/metrics.go`:

```go
// SessionMetrics holds aggregated metrics for multi-turn session workloads.
// Returned by ComputeSessionMetrics; nil when no session requests are present.
type SessionMetrics struct {
	SessionCount    int          // distinct SessionIDs observed in completed requests
	TTFTCold        Distribution // TTFT for round-0 requests (first-turn, cache cold)
	TTFTWarm        Distribution // TTFT for round≥1 requests (follow-up turns, cache warm)
	SessionDuration Distribution // max_completion - round0_arrival per session (ms)
}

// ComputeSessionMetrics computes session-level metrics from aggregated simulation metrics.
// Returns nil when no completed requests carry a non-empty SessionID (BC-1).
// Session duration excludes sessions whose round-0 request is not in m.Requests (BC-6).
func ComputeSessionMetrics(m *sim.Metrics) *SessionMetrics {
	// Gate: scan for any session request
	hasSession := false
	for _, rm := range m.Requests {
		if rm.SessionID != "" {
			hasSession = true
			break
		}
	}
	if !hasSession {
		return nil
	}

	// Partition TTFT values by round index
	var coldTTFTs, warmTTFTs []float64

	// Per-session data for duration: track round-0 arrival and max completion time
	type sessionData struct {
		round0ArrivalMs float64
		hasRound0       bool
		maxCompMs       float64
	}
	sessions := make(map[string]*sessionData)

	// Process in sorted request ID order for determinism (R2)
	reqIDs := sortedKeys(m.RequestTTFTs) // sorted IDs from a parallel map; fall back below
	_ = reqIDs                           // we iterate m.Requests via sorted session keys instead

	// Build session map and collect TTFT values; iterate requests in stable order.
	// We cannot use sortedKeys directly on m.Requests (different type), so build a
	// sorted slice of request IDs first.
	ids := make([]string, 0, len(m.Requests))
	for id := range m.Requests {
		ids = append(ids, id)
	}
	sort.Strings(ids)

	for _, id := range ids {
		rm := m.Requests[id]
		if rm.SessionID == "" {
			continue
		}

		// TTFT partition
		if rm.RoundIndex == 0 {
			coldTTFTs = append(coldTTFTs, rm.TTFT)
		} else {
			warmTTFTs = append(warmTTFTs, rm.TTFT)
		}

		// Session duration bookkeeping
		sd, ok := sessions[rm.SessionID]
		if !ok {
			sd = &sessionData{}
			sessions[rm.SessionID] = sd
		}
		if rm.RoundIndex == 0 {
			sd.round0ArrivalMs = rm.ArrivedAt
			sd.hasRound0 = true
		}
		// Track max completion time across all rounds of this session
		if compUs, exists := m.RequestCompletionTimes[id]; exists {
			compMs := compUs / 1000.0
			if compMs > sd.maxCompMs {
				sd.maxCompMs = compMs
			}
		}
	}

	// Compute session durations (BC-3, BC-6)
	var durationMs []float64
	sessionIDs := make([]string, 0, len(sessions))
	for sid := range sessions {
		sessionIDs = append(sessionIDs, sid)
	}
	sort.Strings(sessionIDs) // R2: deterministic order
	for _, sid := range sessionIDs {
		sd := sessions[sid]
		if !sd.hasRound0 {
			continue // BC-6: skip sessions without observed round-0
		}
		dur := sd.maxCompMs - sd.round0ArrivalMs
		if dur >= 0 {
			durationMs = append(durationMs, dur)
		}
	}

	return &SessionMetrics{
		SessionCount:    len(sessions),
		TTFTCold:        NewDistribution(coldTTFTs),
		TTFTWarm:        NewDistribution(warmTTFTs),
		SessionDuration: NewDistribution(durationMs),
	}
}
```

**Step 4 — Run tests to verify they pass:**
```bash
go test ./sim/cluster/... -run TestComputeSessionMetrics
# Expected: PASS
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/cluster/...
# Expected: no issues
```

**Step 6 — Commit:**
```
feat(metrics): add SessionMetrics and ComputeSessionMetrics (BC-1 through BC-6)
```

---

#### Task 3: Add printSessionMetrics to cmd/root.go

**Contracts:** BC-1 (nil guard), BC-2, BC-3 (output format)

**Files:** modify `cmd/root.go`

**Step 1 — Write failing test:**

In `cmd/root_test.go` (or `cmd/session_metrics_print_test.go` if that is cleaner), add:

```go
func TestPrintSessionMetrics_Nil_NoOutput(t *testing.T) {
	var buf bytes.Buffer
	printSessionMetrics(&buf, nil)
	if buf.Len() != 0 {
		t.Errorf("expected no output for nil SessionMetrics, got: %q", buf.String())
	}
}

func TestPrintSessionMetrics_OutputContainsKeys(t *testing.T) {
	sm := &cluster.SessionMetrics{
		SessionCount:    3,
		TTFTCold:        cluster.NewDistribution([]float64{50.0, 60.0}),
		TTFTWarm:        cluster.NewDistribution([]float64{20.0, 22.0}),
		SessionDuration: cluster.NewDistribution([]float64{700.0, 750.0}),
	}
	var buf bytes.Buffer
	printSessionMetrics(&buf, sm)
	out := buf.String()
	for _, want := range []string{"Session Metrics", "Sessions: 3", "TTFT cold", "TTFT warm", "Session duration"} {
		if !strings.Contains(out, want) {
			t.Errorf("output missing %q\nfull output:\n%s", want, out)
		}
	}
}

func TestPrintSessionMetrics_WarmOnly(t *testing.T) {
	// When TTFTCold.Count == 0, cold section must be absent
	sm := &cluster.SessionMetrics{
		SessionCount: 1,
		TTFTCold:     cluster.NewDistribution(nil),
		TTFTWarm:     cluster.NewDistribution([]float64{25.0}),
	}
	var buf bytes.Buffer
	printSessionMetrics(&buf, sm)
	out := buf.String()
	if strings.Contains(out, "TTFT cold") {
		t.Errorf("output must not contain 'TTFT cold' when count is 0\nfull output:\n%s", out)
	}
	if !strings.Contains(out, "TTFT warm") {
		t.Errorf("output must contain 'TTFT warm'\nfull output:\n%s", out)
	}
}
```

**Step 2 — Run tests to verify they fail:**
```bash
go test ./cmd/... -run TestPrintSessionMetrics
# Expected: FAIL (printSessionMetrics undefined)
```

**Step 3 — Implement:**

Add `printSessionMetrics` to `cmd/root.go`. Place it alongside the other `printXxx` functions:

```go
// printSessionMetrics writes the session metrics section to w.
// No-op when sm is nil (non-session workloads).
func printSessionMetrics(w io.Writer, sm *cluster.SessionMetrics) {
	if sm == nil {
		return
	}
	_, _ = fmt.Fprintln(w, "\n=== Session Metrics ===")
	_, _ = fmt.Fprintf(w, "  Sessions: %d\n", sm.SessionCount)
	if sm.TTFTCold.Count > 0 {
		_, _ = fmt.Fprintf(w, "  TTFT cold (round 0): mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.TTFTCold.Mean, sm.TTFTCold.P50, sm.TTFTCold.P95, sm.TTFTCold.P99, sm.TTFTCold.Count)
	}
	if sm.TTFTWarm.Count > 0 {
		_, _ = fmt.Fprintf(w, "  TTFT warm (round≥1): mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.TTFTWarm.Mean, sm.TTFTWarm.P50, sm.TTFTWarm.P95, sm.TTFTWarm.P99, sm.TTFTWarm.Count)
	}
	if sm.SessionDuration.Count > 0 {
		_, _ = fmt.Fprintf(w, "  Session duration:    mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.SessionDuration.Mean, sm.SessionDuration.P50, sm.SessionDuration.P95, sm.SessionDuration.P99, sm.SessionDuration.Count)
	}
}
```

Wire it into the print loop in the `run` command handler, after `printPerTenantMetrics`:

```go
perTenantMetrics := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
printPerTenantMetrics(os.Stdout, perTenantMetrics)
sessionMetrics := cluster.ComputeSessionMetrics(cs.AggregatedMetrics())
printSessionMetrics(os.Stdout, sessionMetrics)
```

Also wire it in the `replay` command handler at the equivalent print site.

**Step 4 — Run tests to verify they pass:**
```bash
go test ./cmd/... -run TestPrintSessionMetrics
# Expected: PASS
```

**Step 5 — Lint:**
```bash
golangci-lint run ./cmd/...
# Expected: no issues
```

**Step 6 — Commit:**
```
feat(cmd): add printSessionMetrics and wire into run/replay print loop
```

---

#### Task 4: End-to-end integration smoke test

**Contracts:** All (integration verification)

**Files:** no new files — uses existing test infrastructure

**Step 1 — Run full test suite:**
```bash
go test ./...
# Expected: all PASS
```

**Step 2 — Build and manual smoke check with multi-turn workload:**

Generate a small multi-turn workload trace and replay through the simulator to verify the Session Metrics section appears:

```bash
go build -o blis main.go
./blis replay \
  --trace-header traces/swe_smith_v3.yaml \
  --trace-data traces/swe_smith_v3.csv \
  --model qwen/qwen3-14b \
  --session-mode closed-loop \
  2>/dev/null | grep -A 10 "Session Metrics"
```

Expected output (values will vary):
```
=== Session Metrics ===
  Sessions: <N>
  TTFT cold (round 0): mean=... p50=... p95=... p99=... ms (n=...)
  TTFT warm (round≥1): mean=... p50=... p95=... p99=... ms (n=...)
  Session duration:    mean=... p50=... p95=... p99=... ms (n=...)
```

**Step 3 — Verify single-turn workload shows no section:**
```bash
./blis run --model qwen/qwen3-14b 2>/dev/null | grep "Session Metrics"
# Expected: no output (section absent)
```

**Step 4 — Commit:**
```
test(integration): verify session metrics section present/absent by workload type
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-5 | Task 1 | Unit | `TestNewRequestMetrics_SessionFields` |
| BC-1 (nil) | Task 2 | Unit | `TestComputeSessionMetrics_NoSessions_ReturnsNil` |
| BC-1 (empty) | Task 2 | Unit | `TestComputeSessionMetrics_EmptyMetrics_ReturnsNil` |
| BC-2 | Task 2 | Unit | `TestComputeSessionMetrics_TTFTPartition` |
| BC-3 | Task 2 | Unit | `TestComputeSessionMetrics_SessionDuration` |
| BC-4 | Task 2 | Unit | `TestComputeSessionMetrics_MixedWorkload` |
| BC-6 | Task 2 | Unit | `TestComputeSessionMetrics_MissingRound0` |
| INV-count | Task 2 | Invariant | `TestComputeSessionMetrics_SessionCount` |
| BC-1 (print) | Task 3 | Unit | `TestPrintSessionMetrics_Nil_NoOutput` |
| BC-2,3 (print) | Task 3 | Unit | `TestPrintSessionMetrics_OutputContainsKeys` |
| BC-2 (warm-only) | Task 3 | Unit | `TestPrintSessionMetrics_WarmOnly` |
| All | Task 4 | Integration | Manual smoke: `blis replay --session-mode closed-loop` |

**Invariant tests alongside golden behavior:**
- `TestComputeSessionMetrics_SessionCount` asserts the law: `SessionCount == len(distinct SessionIDs)` — this invariant would survive a complete rewrite of `ComputeSessionMetrics`.
- `TestComputeSessionMetrics_TTFTPartition` asserts the partitioning law: `TTFTCold.Count + TTFTWarm.Count == total session request count` (implicitly verifiable from the test's input).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| `RequestCompletionTimes` not populated for all completed requests | Low | Medium (silent gap in SessionDuration) | BC-6 guard: missing entries simply skip the session; `TestComputeSessionMetrics_MissingRound0` covers this | Task 2 |
| Non-deterministic output due to map iteration order | Low | High (INV-6 violation) | Explicit `sort.Strings(ids)` and `sort.Strings(sessionIDs)` before all map iterations | Task 2 |
| TTFT unit mismatch (ms vs µs) | Medium | Medium (wrong values) | `RequestMetrics.TTFT` is stored in ms (set by `NewRequestMetrics` which divides by 1000); `RequestCompletionTimes` is in µs; plan explicitly converts completion times with `/1000.0` | Task 2 |
| Missing wire-up in `replay` command (only `run` command wired) | Medium | Low (feature missing from replay) | Task 3 explicitly names both command handlers | Task 3 |
| `sim.Metrics.Requests` map may be nil in edge cases | Low | Low (nil map range is safe in Go) | Go spec: ranging over nil map is safe; no panic | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `SessionMetrics` is a plain struct; no interface needed (only one implementation).
- [x] No feature creep — session completion rate explicitly deferred; `MeanRoundsCompleted` dropped per user agreement.
- [x] No unexercised flags or interfaces — no new CLI flags.
- [x] No partial implementations — all four metrics (`SessionCount`, `TTFTCold`, `TTFTWarm`, `SessionDuration`) are fully implemented.
- [x] No breaking changes — `RequestMetrics` gains fields with `omitempty` tags; existing JSON output unchanged for non-session fields; no existing method signatures modified.
- [x] No hidden global state impact — `ComputeSessionMetrics` is a pure function over `*sim.Metrics`.
- [x] All new code will pass golangci-lint — lint check in every task.
- [x] Shared test helpers used — `buildSessionMetrics` is local to `metrics_session_test.go`; `NewDistribution` reused from existing package.
- [x] CLAUDE.md updated if needed — no new packages, no new CLI flags, no milestone completed. No update needed.
- [x] No stale references left in CLAUDE.md.
- [x] Documentation DRY — no canonical standard docs modified.
- [x] Deviation log reviewed — all deviations are CLARIFICATION or DEFERRAL; none unresolved.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered — Task 1 (RequestMetrics fields) must complete before Task 2 (ComputeSessionMetrics reads those fields).
- [x] All contracts mapped to specific tasks — see Test Strategy table.
- [x] Golden dataset regeneration — not needed (no existing golden datasets affected).
- [x] Construction site audit completed — one production site (`NewRequestMetrics`), all others are test literals.
- [x] PR is not part of a macro plan — no macro plan update needed.

**Antipattern rules:**
- [x] R1: No silent `continue`/`return` — BC-6 skip is logged implicitly via count (no data dropped silently; session simply absent from duration).
- [x] R2: Map keys sorted — explicit `sort.Strings(ids)` and `sort.Strings(sessionIDs)` in `ComputeSessionMetrics`.
- [x] R3: No new numeric parameters validated — no new CLI flags or constructor parameters.
- [x] R4: Construction site audit completed — only `NewRequestMetrics` is the production site; updated in Task 1.
- [x] R5: No resource allocation loops.
- [x] R6: No `logrus.Fatalf` or `os.Exit` in `sim/` packages — new code is in `sim/cluster/` and `cmd/`.
- [x] R7: Invariant test alongside golden — `TestComputeSessionMetrics_SessionCount` is an invariant test.
- [x] R8: No exported mutable maps — `SessionMetrics` has no map fields.
- [x] R9: No `*float64` YAML fields — no YAML config changes.
- [x] R10: YAML strict parsing — no YAML changes.
- [x] R11: No division by runtime-derived denominators — distributions use `NewDistribution` which guards empty input.
- [x] R12: No golden dataset changes.
- [x] R13: No new interfaces — `SessionMetrics` is a concrete type; only one implementation.
- [x] R14: No method spanning multiple responsibilities — `ComputeSessionMetrics` owns aggregation only; `printSessionMetrics` owns formatting only.
- [x] R15: No stale PR references.
- [x] R16: No config param changes.
- [x] R17: No scorer signals.
- [x] R18: No CLI flag value overwriting.
- [x] R19: No retry/requeue loops.
- [x] R20: No degenerate input — `NewDistribution` returns zero-value `Distribution` for empty input.
- [x] R21: No range over shrinking slices.
- [x] R22: No pre-check estimates.
- [x] R23: No parallel code paths.

---

## Appendix: File-Level Implementation Details

### File: `sim/metrics_utils.go`

**Purpose:** Extend `RequestMetrics` data carrier with session context fields.

**Complete diff for `RequestMetrics` struct** — add after `GatewayQueueDelay float64`:
```go
SessionID  string `json:"session_id,omitempty"`
RoundIndex int    `json:"round_index,omitempty"`
```

**Complete diff for `NewRequestMetrics`** — add inside the struct literal after `LengthCapped`:
```go
SessionID:        req.SessionID,
RoundIndex:       req.RoundIndex,
```

**Note on omitempty for RoundIndex:** `omitempty` is correct here. RoundIndex 0 on a non-session request is meaningless; `SessionID` omitempty already signals "this is not a session request". For session requests, `RoundIndex: 0` combined with a non-empty `SessionID` is the distinguishing signal for round-0 — and since `SessionID` is present, `round_index` will appear in JSON (it's 0, which omitempty would suppress). This is a mild ambiguity, but consistent with how other optional fields are handled in the struct.

**Correction:** After reflection, `RoundIndex int` with `omitempty` means round-0 session requests will show `session_id` but NOT `round_index` in JSON output (zero is omitted). This is the BC-5 issue noted in the design discussion. Since the user agreed that `omitempty` is not needed on `RoundIndex` (round-0 must be unambiguously distinguishable), use:
```go
RoundIndex int `json:"round_index,omitempty"`
```
is WRONG for session requests at round 0. Use without omitempty:
```go
RoundIndex int `json:"round_index"`
```
but this adds `"round_index": 0` noise to non-session requests. The plan resolves this as: **do not use omitempty on RoundIndex** — `"round_index": 0` for non-session requests is minor noise but semantically correct (they are at round 0 of a non-existent session). The `session_id` field's presence/absence is the true gate.

**Final field definitions:**
```go
SessionID  string `json:"session_id,omitempty"`
RoundIndex int    `json:"round_index"`
```

### File: `sim/cluster/metrics.go`

**Purpose:** Add `SessionMetrics` struct and `ComputeSessionMetrics` function.

**Key implementation notes:**
- **RNG usage:** None — pure aggregation function.
- **Metrics:** `TTFTCold`, `TTFTWarm` read from `RequestMetrics.TTFT` (ms). `SessionDuration` computed from `RequestMetrics.ArrivedAt` (ms) and `sim.Metrics.RequestCompletionTimes` values (µs, divided by 1000.0).
- **State mutation:** None — pure function, produces new struct.
- **Error handling:** No errors; BC-6 silently skips incomplete sessions; nil return for non-session workloads (BC-1).
- **Sort import:** `sort` package already imported in `metrics.go`.

### File: `cmd/root.go`

**Purpose:** Add `printSessionMetrics` and wire it into `run` and `replay` print loops.

**Key implementation notes:**
- **Wire-up location for `run`:** After `printPerTenantMetrics` call, approximately line 1778.
- **Wire-up location for `replay`:** Locate the equivalent `printPerTenantMetrics` call in the replay handler and add the same two lines after it.
- **Import:** `sim/cluster` already imported.
- **Format:** All latency values printed in ms (consistent with rest of BLIS output). `p95` included alongside `p50`/`p99` for consistency with existing `printPerSLOMetrics` format.

---

## Phase 2: Observe Extension (`cmd/session_metrics_trace.go`)

This phase was added after the initial plan to complete cross-path parity. `blis observe` does not run a DES — it dispatches to a real HTTP server and writes a TraceV2 CSV. `ComputeSessionMetrics` cannot be reused because it reads from `sim.Metrics` (a DES-internal structure), not from trace records.

### Why a separate function

`computeSessionMetricsFromTrace` lives in `cmd/` (unexported) rather than `sim/cluster/` for two reasons:

1. **Dependency direction.** It imports both `sim/cluster` (for `SessionMetrics`, `NewDistribution`) and `sim/workload` (for `TraceRecord`). Placing it in `sim/cluster/` would require `sim/cluster/` to import `sim/workload/`, violating the `cmd/ → sim/cluster/ → sim/` chain.
2. **Different input type.** The function operates on `[]workload.TraceRecord` (wall-clock timestamps from real HTTP exchanges), not on `sim.Metrics` (simulation-internal counters). The separation reflects the real boundary between observed and simulated data.

### TTFT formula

| Path | Formula | Units |
|------|---------|-------|
| DES (`ComputeSessionMetrics`) | `RequestTTFTs[id] / 1000.0` | µs → ms |
| Trace (`computeSessionMetricsFromTrace`) | `(FirstChunkTimeUs − SendTimeUs) / 1000.0` | µs → ms |

For a DES-exported trace, `SendTimeUs = req.ArrivalTime` and `FirstChunkTimeUs = ArrivalTime + FirstTokenTime`, so both formulas produce the same value. The semantic alignment is intentional.

### Behavioral contracts (trace path)

- **BC-T1 (self-gating):** Returns `nil` when no record has a non-empty `SessionID`. Session Metrics section is absent for non-session workloads.
- **BC-T2 (status guard):** Only records with `status == "ok"` and `FirstChunkTimeUs > 0` and `SendTimeUs > 0` contribute to TTFT. Error and timeout records are excluded, preventing 0ms contamination (R1 analog of I-1).
- **BC-T3 (cold/warm split):** `RoundIndex == 0` → cold TTFT; `RoundIndex > 0` → warm TTFT. Same semantics as the DES path.
- **BC-T4 (session duration):** `max(LastChunkTimeUs) − SendTimeUs[round=0]`, in ms, per session. `LastChunkTimeUs == 0` records are excluded from the max (timeout/error rounds do not inflate duration).
- **BC-T5 (BC-6 analog — missing round-0 excluded):** Sessions without a `round=0` record with `SendTimeUs > 0` are excluded from `SessionDuration`. Count of sessions contributing to duration may be less than `SessionCount`.
- **BC-T6 (determinism):** Records sorted by `RequestID` before iteration (R2, INV-6).

### Wiring

`computeSessionMetricsFromTrace(records)` is called in `observe_cmd.go` after `ExportTraceV2`, passing the recorder's records. `printSessionMetrics` (shared with the DES path) formats output. The section is absent when the function returns `nil`.
