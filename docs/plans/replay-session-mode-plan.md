# Closed-Loop Session Sequencing for Trace Replay — Implementation Plan

**Goal:** Enable `blis replay` to replay multi-turn sessions with load-adaptive timing, where turn N+1 arrives only after turn N completes plus think time.

**The problem today:** `blis replay` pre-bakes all arrival times from TraceV2 records. For multi-turn agentic workloads, this breaks the feedback loop: when turn N takes longer under load, turn N+1 still arrives at its pre-recorded timestamp rather than at `completion_time + think_time`.

**What this PR adds:**
1. A `SequenceSampler` that replays pre-recorded token counts in order (implements `LengthSampler`)
2. An optional `ThinkTimeSampler` field on `SessionBlueprint` for per-round think time variation
3. A `LoadTraceV2SessionBlueprints()` function that groups trace records by session and builds blueprints
4. CLI flags `--session-mode` and `--think-time-ms` on `blis replay` to activate closed-loop mode

**Why this matters:** Studying how serving policies (routing, admission, scheduling) affect session pacing requires closed-loop simulation. Without it, capacity planning for agentic workloads underestimates queueing delays.

**Architecture:** Extends existing `LengthSampler` interface with a new implementation (`SequenceSampler`), adds an optional sampler field to `SessionBlueprint`, and adds a new loader function in `replay.go`. CLI wiring in `cmd/replay.go`. No new interfaces or packages.

**Source:** [GitHub Issue #880](https://github.com/inference-sim/inference-sim/issues/880), implementation comment by @sriumcp.

**Closes:** `Fixes #880`

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Workload replay pipeline (`sim/workload/replay.go`) + Session management (`sim/workload/session.go`)
2. **Adjacent blocks:** `cmd/replay.go` (CLI), `sim/cluster/cluster.go` (accepts `onRequestDone` callback), `sim/workload/distribution.go` (LengthSampler interface)
3. **Invariants touched:** INV-6 (determinism), INV-10 (session causality), INV-11 (session completeness)
4. **Construction Site Audit:**
   - `SessionBlueprint{}` — constructed in `sim/workload/generator.go` (1 site) and `sim/workload/session_test.go` (1 helper). Adding `ThinkTimeSampler` field requires NO updates: it is a `LengthSampler` interface whose zero value is `nil`, which triggers the fallback to constant `ThinkTimeUs` (BC-4). Both existing sites remain correct without modification.
   - `SequenceSampler{}` — new type, 1 construction site in `LoadTraceV2SessionBlueprints()`

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds closed-loop session replay to `blis replay`. Today, all trace records are injected with their original timestamps regardless of simulated load. This PR introduces a `--session-mode closed-loop` flag that groups trace records by `SessionID`, creates `SessionBlueprint` objects from trace data (with `SequenceSampler` for deterministic token replay), and injects only round-0 requests. The existing `SessionManager` handles follow-up generation on completion, preserving INV-10 (session causality). Non-session records and `--session-mode fixed` (default) are unaffected.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: SequenceSampler Deterministic Replay
- GIVEN a SequenceSampler initialized with token counts [100, 200, 300]
- WHEN Sample() is called 3 times
- THEN it returns 100, 200, 300 in order
- MECHANISM: Internal index advances on each call

BC-2: SequenceSampler Wraps on Exhaustion
- GIVEN a SequenceSampler with 2 values [100, 200]
- WHEN Sample() is called 3 times
- THEN it returns 100, 200, 100 (wraps to beginning)

BC-3: ThinkTimeSampler Used When Present
- GIVEN a SessionBlueprint with ThinkTimeSampler set
- WHEN SessionManager.OnComplete generates a follow-up
- THEN the follow-up's arrival time = completion_tick + ThinkTimeSampler.Sample()
- MECHANISM: OnComplete checks if ThinkTimeSampler is non-nil; if so, samples from it instead of using constant ThinkTimeUs

BC-4: ThinkTimeSampler Absent Falls Back
- GIVEN a SessionBlueprint with ThinkTimeSampler = nil
- WHEN SessionManager.OnComplete generates a follow-up
- THEN the follow-up's arrival time = completion_tick + ThinkTimeUs (existing behavior)

BC-5: LoadTraceV2SessionBlueprints Groups by Session
- GIVEN a trace with records: (session=A, round=0), (session=A, round=1), (session=B, round=0)
- WHEN LoadTraceV2SessionBlueprints is called
- THEN it returns 2 blueprints (A with MaxRounds=2, B with MaxRounds=1) and 2 round-0 requests

BC-6: LoadTraceV2SessionBlueprints Token Sequence
- GIVEN a trace session A with round 0 (input=100, output=50) and round 1 (input=200, output=80)
- WHEN the blueprint's InputSampler is called twice
- THEN it returns 100 then 200

BC-7: Non-Session Records Pass Through
- GIVEN a trace with records having empty SessionID mixed with session records
- WHEN LoadTraceV2SessionBlueprints is called
- THEN non-session records are returned as regular requests alongside round-0 session requests

BC-8: CLI Default is Fixed Mode
- GIVEN `blis replay` invoked without `--session-mode`
- WHEN replay executes
- THEN all records are pre-injected with original timestamps (existing behavior unchanged)

BC-9: CLI Closed-Loop Mode Wires SessionManager
- GIVEN `blis replay --session-mode closed-loop`
- WHEN replay executes
- THEN only round-0 requests are injected and a SessionManager callback is passed to ClusterSimulator

BC-10: Think Time Override
- GIVEN `blis replay --session-mode closed-loop --think-time-ms 500`
- WHEN blueprints are created
- THEN all sessions use 500ms (500000 µs) constant think time instead of trace-derived gaps

**Negative contracts:**

BC-11: Invalid Session Mode Rejected
- GIVEN `blis replay --session-mode foobar`
- WHEN flag parsing completes
- THEN replay exits with fatal error naming valid options

BC-12: Closed-Loop Without Sessions Warns
- GIVEN a trace with no session records and `--session-mode closed-loop`
- WHEN replay loads the trace
- THEN it logs a warning and proceeds with zero blueprints (all records injected as regular requests)

### C) Component Interaction

```
cmd/replay.go
  │ --session-mode, --think-time-ms flags
  │
  ├─► [fixed mode] workload.LoadTraceV2Requests() ──► requests[] ──► ClusterSim(nil)
  │
  └─► [closed-loop] workload.LoadTraceV2SessionBlueprints()
        │
        ├──► round0Requests[] ──► ClusterSim(sessionMgr.OnComplete)
        └──► []SessionBlueprint ──► NewSessionManager()
               │
               └──► SessionBlueprint.InputSampler  = SequenceSampler (per-round input tokens)
                    SessionBlueprint.OutputSampler = SequenceSampler (per-round output tokens)
                    SessionBlueprint.ThinkTimeSampler = SequenceSampler (per-round gaps) or nil
```

**State ownership:** SequenceSampler owns its index counter. SessionBlueprint owns samplers. SessionManager owns session lifecycle state.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Add ThinkTimeSampler field to SessionBlueprint" | Uses `LengthSampler` interface for ThinkTimeSampler to reuse SequenceSampler | SIMPLIFICATION: avoids creating a separate sampler type for think time; LengthSampler returns int which maps to microseconds |
| Not specified | BC-2 wraps on exhaustion instead of panicking | ADDITION: defensive behavior if SessionManager somehow requests more rounds than trace has |
| Not specified | BC-12 warns on closed-loop with no sessions | ADDITION: user-friendly diagnostics |

### E) Review Guide

**Tricky part:** The `SequenceSampler` must be safe against exhaustion (BC-2). The think time derivation from trace gaps (inter-round arrival deltas) must handle edge cases (round 0 has no predecessor, negative gaps from clock skew).

**Scrutinize:** `LoadTraceV2SessionBlueprints` — correct grouping, sort by round, think time derivation, prefix token handling.

**Safe to skim:** CLI flag wiring in `cmd/replay.go` — mechanical plumbing.

**Known debt:** Context accumulation for trace replay sessions is not implemented (would require storing actual generated output from each round). All trace replay sessions use non-accumulating mode.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action |
|------|--------|
| `sim/workload/distribution.go` | Add `SequenceSampler` type |
| `sim/workload/distribution_test.go` | Add tests for `SequenceSampler` |
| `sim/workload/session.go` | Add `ThinkTimeSampler` field to `SessionBlueprint`, modify `OnComplete` |
| `sim/workload/session_test.go` | Add tests for `ThinkTimeSampler` behavior |
| `sim/workload/replay.go` | Add `LoadTraceV2SessionBlueprints()` function |
| `sim/workload/replay_test.go` | Add tests for blueprint loading |
| `cmd/replay.go` | Add `--session-mode` and `--think-time-ms` flags, wire closed-loop path |

No dead code. All new code is reachable from CLI flags.

### G) Task Breakdown

#### Task 1: SequenceSampler (BC-1, BC-2)

**Files:** modify `sim/workload/distribution.go`, create `sim/workload/distribution_test.go`

**Test:**
```go
// In sim/workload/distribution_test.go
func TestSequenceSampler_ReplayInOrder(t *testing.T) {
    s := &SequenceSampler{values: []int{100, 200, 300}}
    for i, want := range []int{100, 200, 300} {
        got := s.Sample(nil)
        if got != want {
            t.Errorf("call %d: got %d, want %d", i, got, want)
        }
    }
}

func TestSequenceSampler_WrapsOnExhaustion(t *testing.T) {
    s := &SequenceSampler{values: []int{10, 20}}
    _ = s.Sample(nil) // 10
    _ = s.Sample(nil) // 20
    got := s.Sample(nil)
    if got != 10 {
        t.Errorf("wrap: got %d, want 10", got)
    }
}

func TestSequenceSampler_SingleValue(t *testing.T) {
    s := &SequenceSampler{values: []int{42}}
    for i := 0; i < 5; i++ {
        got := s.Sample(nil)
        if got != 42 {
            t.Errorf("call %d: got %d, want 42", i, got)
        }
    }
}
```

**Impl:**
```go
// In sim/workload/distribution.go

// SequenceSampler replays a pre-recorded sequence of values in order.
// Used for trace replay where token counts are known per-round.
// Wraps to the beginning when the sequence is exhausted.
type SequenceSampler struct {
    values []int
    index  int
}

func (s *SequenceSampler) Sample(_ *rand.Rand) int {
    if len(s.values) == 0 {
        return 1 // defensive: empty sequence returns minimum token count
    }
    v := s.values[s.index%len(s.values)]
    s.index++
    return v
}
```

**Verify:** `go test ./sim/workload/... -run TestSequenceSampler`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `feat(workload): add SequenceSampler for trace replay (BC-1, BC-2)`

---

#### Task 2: ThinkTimeSampler in SessionBlueprint (BC-3, BC-4)

**Files:** modify `sim/workload/session.go`, modify `sim/workload/session_test.go`

**Test:**
```go
// In sim/workload/session_test.go

func TestSession_ThinkTimeSampler_UsedWhenPresent(t *testing.T) {
    bp := makeTestBlueprint("tts1", 3, 1000, "", 1_000_000)
    bp.ThinkTimeSampler = &SequenceSampler{values: []int{2000, 3000}}
    sm := NewSessionManager([]SessionBlueprint{bp})

    req0 := &sim.Request{
        ID: "r0", SessionID: "tts1", RoundIndex: 0,
        State: sim.StateCompleted, ProgressIndex: 15,
        InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
    }

    follow := sm.OnComplete(req0, 5000)
    if len(follow) != 1 {
        t.Fatalf("expected 1 follow-up, got %d", len(follow))
    }
    // Should use ThinkTimeSampler value (2000) not constant ThinkTimeUs (1000)
    if follow[0].ArrivalTime != 7000 {
        t.Errorf("BC-3: arrival = %d, want 7000 (5000 + 2000)", follow[0].ArrivalTime)
    }
}

func TestSession_ThinkTimeSampler_NilFallsBack(t *testing.T) {
    bp := makeTestBlueprint("tts2", 3, 1000, "", 1_000_000)
    // ThinkTimeSampler is nil by default
    sm := NewSessionManager([]SessionBlueprint{bp})

    req0 := &sim.Request{
        ID: "r0", SessionID: "tts2", RoundIndex: 0,
        State: sim.StateCompleted, ProgressIndex: 15,
        InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
    }

    follow := sm.OnComplete(req0, 5000)
    if len(follow) != 1 {
        t.Fatalf("expected 1 follow-up, got %d", len(follow))
    }
    if follow[0].ArrivalTime != 6000 {
        t.Errorf("BC-4: arrival = %d, want 6000 (5000 + 1000)", follow[0].ArrivalTime)
    }
}
```

**Impl:**

Add field to `SessionBlueprint`:
```go
ThinkTimeSampler LengthSampler // optional: per-round think time in µs; nil = use constant ThinkTimeUs
```

Modify `OnComplete` — replace the line:
```go
arrivalTime := tick + bp.ThinkTimeUs
```
with:
```go
var thinkTime int64
if bp.ThinkTimeSampler != nil {
    thinkTime = int64(bp.ThinkTimeSampler.Sample(bp.RNG))
} else {
    thinkTime = bp.ThinkTimeUs
}
arrivalTime := tick + thinkTime
```

**Verify:** `go test ./sim/workload/... -run TestSession`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `feat(workload): add ThinkTimeSampler to SessionBlueprint (BC-3, BC-4)`

---

#### Task 3: LoadTraceV2SessionBlueprints (BC-5, BC-6, BC-7)

**Files:** modify `sim/workload/replay.go`, create `sim/workload/replay_test.go`

**Test:**
```go
// In sim/workload/replay_test.go
package workload

import (
    "testing"
)

func TestLoadTraceV2SessionBlueprints_GroupsBySession(t *testing.T) {
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
            {RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
            {RequestID: 3, SessionID: "B", RoundIndex: 0, InputTokens: 150, OutputTokens: 60, ArrivalTimeUs: 1000},
        },
    }

    requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    // BC-5: 2 blueprints
    if len(blueprints) != 2 {
        t.Fatalf("BC-5: got %d blueprints, want 2", len(blueprints))
    }

    // BC-5: 2 round-0 requests
    if len(requests) != 2 {
        t.Fatalf("BC-5: got %d requests, want 2", len(requests))
    }

    // Find blueprint A
    var bpA *SessionBlueprint
    for i := range blueprints {
        if blueprints[i].SessionID == "A" {
            bpA = &blueprints[i]
            break
        }
    }
    if bpA == nil {
        t.Fatal("blueprint A not found")
    }
    if bpA.MaxRounds != 2 {
        t.Errorf("BC-5: session A MaxRounds = %d, want 2", bpA.MaxRounds)
    }

    // BC-6: input sampler replays trace values for follow-up rounds (round 0 is injected directly)
    got1 := bpA.InputSampler.Sample(nil)
    if got1 != 200 {
        t.Errorf("BC-6: input sampler first value = %d, want 200 (round 1 token count)", got1)
    }
}

func TestLoadTraceV2SessionBlueprints_NonSessionPassThrough(t *testing.T) {
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
            {RequestID: 2, SessionID: "A", RoundIndex: 0, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 1000},
        },
    }

    requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    // BC-7: 1 non-session + 1 round-0 session = 2 requests
    if len(requests) != 2 {
        t.Fatalf("BC-7: got %d requests, want 2", len(requests))
    }
    if len(blueprints) != 1 {
        t.Errorf("BC-7: got %d blueprints, want 1", len(blueprints))
    }
}

func TestLoadTraceV2SessionBlueprints_ThinkTimeFromTrace(t *testing.T) {
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
            {RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
            {RequestID: 3, SessionID: "A", RoundIndex: 2, InputTokens: 300, OutputTokens: 90, ArrivalTimeUs: 12000},
        },
    }

    _, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    bp := blueprints[0]
    // Think times derived from inter-round arrival gaps: [5000, 7000]
    // ThinkTimeSampler should replay these
    if bp.ThinkTimeSampler == nil {
        t.Fatal("expected ThinkTimeSampler to be set")
    }
    got1 := bp.ThinkTimeSampler.Sample(nil)
    got2 := bp.ThinkTimeSampler.Sample(nil)
    if got1 != 5000 || got2 != 7000 {
        t.Errorf("think times = [%d, %d], want [5000, 7000]", got1, got2)
    }
}

func TestLoadTraceV2SessionBlueprints_SingleRoundSession(t *testing.T) {
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
        },
    }

    requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 0, 0)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }
    if len(requests) != 1 {
        t.Fatalf("got %d requests, want 1", len(requests))
    }
    if len(blueprints) != 1 {
        t.Fatalf("got %d blueprints, want 1", len(blueprints))
    }
    bp := blueprints[0]
    if bp.MaxRounds != 1 {
        t.Errorf("MaxRounds = %d, want 1", bp.MaxRounds)
    }
    if bp.ThinkTimeSampler != nil {
        t.Error("expected nil ThinkTimeSampler for single-round session")
    }
}

func TestLoadTraceV2SessionBlueprints_OverrideThinkTime(t *testing.T) {
    trace := &TraceV2{
        Records: []TraceRecord{
            {RequestID: 1, SessionID: "A", RoundIndex: 0, InputTokens: 100, OutputTokens: 50, ArrivalTimeUs: 0},
            {RequestID: 2, SessionID: "A", RoundIndex: 1, InputTokens: 200, OutputTokens: 80, ArrivalTimeUs: 5000},
        },
    }

    _, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, 500_000, 0)
    if err != nil {
        t.Fatalf("unexpected error: %v", err)
    }

    bp := blueprints[0]
    // With override, ThinkTimeSampler should be nil and ThinkTimeUs = 500000
    if bp.ThinkTimeSampler != nil {
        t.Error("expected nil ThinkTimeSampler when override provided")
    }
    if bp.ThinkTimeUs != 500_000 {
        t.Errorf("ThinkTimeUs = %d, want 500000", bp.ThinkTimeUs)
    }
}
```

**Impl:**
```go
// In sim/workload/replay.go

// LoadTraceV2SessionBlueprints groups trace records by session and builds
// SessionBlueprints with SequenceSamplers for deterministic token replay.
// Returns round-0 requests (plus all non-session requests) for injection,
// and blueprints for the SessionManager.
//
// thinkTimeOverrideUs > 0: use constant think time for all sessions.
// thinkTimeOverrideUs == 0: derive per-round think time from trace arrival gaps.
// horizon: simulation horizon for blueprint (0 = math.MaxInt64).
func LoadTraceV2SessionBlueprints(trace *TraceV2, seed int64, thinkTimeOverrideUs int64, horizon int64) ([]*sim.Request, []SessionBlueprint, error) {
    if trace == nil || len(trace.Records) == 0 {
        return nil, nil, fmt.Errorf("empty trace")
    }
    if horizon <= 0 {
        horizon = math.MaxInt64
    }

    rng := rand.New(rand.NewSource(seed))

    // Generate shared prefix tokens per prefix group (same as LoadTraceV2Requests)
    prefixTokens := make(map[string][]int)
    for _, rec := range trace.Records {
        if rec.PrefixGroup != "" && rec.PrefixLength > 0 {
            if _, exists := prefixTokens[rec.PrefixGroup]; !exists {
                prefixTokens[rec.PrefixGroup] = sim.GenerateRandomTokenIDs(rng, rec.PrefixLength)
            }
        }
    }

    // Group records by session
    type sessionRounds struct {
        records []TraceRecord
    }
    sessionMap := make(map[string]*sessionRounds)
    var nonSessionRecords []TraceRecord
    // Use a slice to track insertion order for deterministic iteration
    var sessionOrder []string

    for _, rec := range trace.Records {
        if rec.SessionID == "" {
            nonSessionRecords = append(nonSessionRecords, rec)
            continue
        }
        sr, exists := sessionMap[rec.SessionID]
        if !exists {
            sr = &sessionRounds{}
            sessionMap[rec.SessionID] = sr
            sessionOrder = append(sessionOrder, rec.SessionID)
        }
        sr.records = append(sr.records, rec)
    }

    // Sort each session's records by RoundIndex and validate consecutive indices
    for sid, sr := range sessionMap {
        sort.Slice(sr.records, func(i, j int) bool {
            return sr.records[i].RoundIndex < sr.records[j].RoundIndex
        })
        for i, rec := range sr.records {
            if rec.RoundIndex != i {
                return nil, nil, fmt.Errorf("session %q has non-consecutive round indices (expected %d, got %d)", sid, i, rec.RoundIndex)
            }
        }
    }

    // Build blueprints and round-0 requests
    var requests []*sim.Request
    var blueprints []SessionBlueprint

    for _, sessionID := range sessionOrder {
        sr := sessionMap[sessionID]
        rounds := sr.records
        if len(rounds) == 0 {
            continue
        }

        // Build per-round token sequences
        inputSeq := make([]int, len(rounds))
        outputSeq := make([]int, len(rounds))
        for i, rec := range rounds {
            inputSeq[i] = rec.InputTokens
            outputSeq[i] = rec.OutputTokens
        }

        // Build think time sequence from inter-round arrival gaps
        var thinkTimeSampler LengthSampler
        var thinkTimeUs int64
        if thinkTimeOverrideUs > 0 {
            thinkTimeUs = thinkTimeOverrideUs
        } else if len(rounds) > 1 {
            thinkTimes := make([]int, len(rounds)-1)
            for i := 1; i < len(rounds); i++ {
                gap := rounds[i].ArrivalTimeUs - rounds[i-1].ArrivalTimeUs
                if gap < 0 {
                    gap = 0
                }
                thinkTimes[i-1] = int(gap)
            }
            thinkTimeSampler = &SequenceSampler{values: thinkTimes}
        }

        // Per-session RNG for deterministic token ID generation
        sessionRNG := rand.New(rand.NewSource(rng.Int63()))

        // Build round-0 request
        r0 := rounds[0]
        inputTokens := sim.GenerateRandomTokenIDs(sessionRNG, r0.InputTokens)
        if r0.PrefixGroup != "" {
            if prefix, ok := prefixTokens[r0.PrefixGroup]; ok {
                inputTokens = append(append([]int{}, prefix...), inputTokens...)
            }
        }
        outputTokens := sim.GenerateRandomTokenIDs(sessionRNG, r0.OutputTokens)

        // Determine prefix for blueprint
        var prefix []int
        if r0.PrefixGroup != "" {
            prefix = prefixTokens[r0.PrefixGroup]
        }

        req := &sim.Request{
            ID:              fmt.Sprintf("request_%d", r0.RequestID),
            ArrivalTime:     r0.ArrivalTimeUs,
            InputTokens:     inputTokens,
            OutputTokens:    outputTokens,
            MaxOutputLen:    len(outputTokens),
            State:           sim.StateQueued,
            TenantID:        r0.TenantID,
            SLOClass:        r0.SLOClass,
            SessionID:       sessionID,
            RoundIndex:      0,
            TextTokenCount:  r0.TextTokens,
            ImageTokenCount: r0.ImageTokens,
            AudioTokenCount: r0.AudioTokens,
            VideoTokenCount: r0.VideoTokens,
            ReasonRatio:     r0.ReasonRatio,
            Model:           r0.Model,
            Deadline:        r0.DeadlineUs,
            ClientID:        r0.ClientID,
            PrefixGroup:     r0.PrefixGroup,
            PrefixLength:    r0.PrefixLength,
            Streaming:       r0.Streaming,
        }
        requests = append(requests, req)

        bp := SessionBlueprint{
            SessionID:        sessionID,
            ClientID:         r0.ClientID,
            MaxRounds:        len(rounds),
            ThinkTimeUs:      thinkTimeUs,
            ThinkTimeSampler: thinkTimeSampler,
            Horizon:          horizon,
            InputSampler:     &SequenceSampler{values: inputSeq[1:]},  // skip round 0 (already injected)
            OutputSampler:    &SequenceSampler{values: outputSeq[1:]}, // skip round 0
            RNG:              sessionRNG,
            Prefix:           prefix,
            TenantID:         r0.TenantID,
            SLOClass:         r0.SLOClass,
            Model:            r0.Model,
        }
        blueprints = append(blueprints, bp)
    }

    // Build non-session requests (same as LoadTraceV2Requests)
    for _, rec := range nonSessionRecords {
        inputTokens := sim.GenerateRandomTokenIDs(rng, rec.InputTokens)
        if rec.PrefixGroup != "" {
            if prefix, ok := prefixTokens[rec.PrefixGroup]; ok {
                inputTokens = append(append([]int{}, prefix...), inputTokens...)
            }
        }
        outputTokens := sim.GenerateRandomTokenIDs(rng, rec.OutputTokens)
        req := &sim.Request{
            ID:              fmt.Sprintf("request_%d", rec.RequestID),
            ArrivalTime:     rec.ArrivalTimeUs,
            InputTokens:     inputTokens,
            OutputTokens:    outputTokens,
            MaxOutputLen:    len(outputTokens),
            State:           sim.StateQueued,
            TenantID:        rec.TenantID,
            SLOClass:        rec.SLOClass,
            SessionID:       rec.SessionID,
            RoundIndex:      rec.RoundIndex,
            TextTokenCount:  rec.TextTokens,
            ImageTokenCount: rec.ImageTokens,
            AudioTokenCount: rec.AudioTokens,
            VideoTokenCount: rec.VideoTokens,
            ReasonRatio:     rec.ReasonRatio,
            Model:           rec.Model,
            Deadline:        rec.DeadlineUs,
            ClientID:        rec.ClientID,
            PrefixGroup:     rec.PrefixGroup,
            PrefixLength:    rec.PrefixLength,
            Streaming:       rec.Streaming,
        }
        requests = append(requests, req)
    }

    return requests, blueprints, nil
}
```

**Verify:** `go test ./sim/workload/... -run TestLoadTraceV2SessionBlueprints`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `feat(workload): add LoadTraceV2SessionBlueprints for closed-loop replay (BC-5, BC-6, BC-7)`

---

#### Task 4: CLI Wiring (BC-8, BC-9, BC-10, BC-11, BC-12)

**Files:** modify `cmd/replay.go`

**Test:** Manual verification via build + flag parsing (CLI flags are validated at runtime).

**Impl:**

Add package-level vars:
```go
var (
    replaySessionMode string
    replayThinkTimeMs int
)
```

In `init()`, add flags:
```go
replayCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", `Session replay mode: "fixed" (pre-baked arrivals) or "closed-loop" (load-adaptive follow-ups)`)
replayCmd.Flags().IntVar(&replayThinkTimeMs, "think-time-ms", 0, "Override think time between session rounds in milliseconds (0 = derive from trace gaps; requires --session-mode closed-loop)")
```

In the `Run` function, after loading the trace and before building requests, replace the existing request-building block:
```go
// Validate session mode (BC-11)
if replaySessionMode != "fixed" && replaySessionMode != "closed-loop" {
    logrus.Fatalf("--session-mode must be \"fixed\" or \"closed-loop\", got %q", replaySessionMode)
}
if replayThinkTimeMs < 0 {
    logrus.Fatalf("--think-time-ms must be non-negative, got %d", replayThinkTimeMs)
}
if replayThinkTimeMs > 0 && replaySessionMode != "closed-loop" {
    logrus.Fatalf("--think-time-ms requires --session-mode closed-loop")
}

var requests []*sim.Request
var sessionMgr *workload.SessionManager

if replaySessionMode == "closed-loop" {
    thinkTimeUs := int64(replayThinkTimeMs) * 1000
    r0Requests, blueprints, err := workload.LoadTraceV2SessionBlueprints(traceData, seed, thinkTimeUs, replayHorizon)
    if err != nil {
        logrus.Fatalf("Failed to build session blueprints from trace: %v", err)
    }
    requests = r0Requests
    if len(blueprints) == 0 {
        logrus.Warnf("--session-mode closed-loop: no session records found in trace; all requests injected with fixed timing")
    } else {
        sessionMgr = workload.NewSessionManager(blueprints)
        logrus.Infof("Closed-loop mode: %d session blueprints, %d round-0 requests", len(blueprints), len(requests))
    }
} else {
    // BC-8: existing behavior
    var err error
    requests, err = workload.LoadTraceV2Requests(traceData, seed)
    if err != nil {
        logrus.Fatalf("Failed to build requests from trace: %v", err)
    }
    logrus.Infof("Built %d requests for replay", len(requests))
}
```

Update the `NewClusterSimulator` call to pass `onRequestDone`:
```go
var onRequestDone func(*sim.Request, int64) []*sim.Request
if sessionMgr != nil {
    onRequestDone = sessionMgr.OnComplete
}
cs := cluster.NewClusterSimulator(config, requests, onRequestDone)
```

**Verify:** `go build ./...`
**Lint:** `golangci-lint run ./cmd/...`
**Commit:** `feat(cmd): add --session-mode and --think-time-ms flags to blis replay (BC-8 through BC-12)`

---

### H) Test Strategy

| Contract | Test Type | Test Location |
|----------|-----------|---------------|
| BC-1, BC-2 | Unit (table-driven) | `distribution_test.go` |
| BC-3, BC-4 | Unit | `session_test.go` |
| BC-5, BC-6, BC-7 | Unit | `replay_test.go` |
| BC-8–BC-12 | Integration (build + flag validation) | Manual via `go build` + smoke test |

### I) Risk Analysis

| Risk | Mitigation |
|------|------------|
| SequenceSampler index overflow on very long sessions | Modulo wrapping (BC-2) prevents panic |
| Non-deterministic map iteration in session grouping | Use `sessionOrder` slice for deterministic iteration (INV-6) |
| Think time derivation from negative trace gaps | Clamp to 0 (defensive) |
| Single-round sessions create empty SequenceSampler | Single-round session: MaxRounds=1, no follow-ups generated, samplers never called |

---

## Sanity Checklist

- [x] R1: No silent `continue` or error swallowing
- [x] R2: Deterministic output (sorted session iteration, seeded RNG)
- [x] R4: SessionBlueprint construction sites checked (generator.go, session_test.go)
- [x] R8: No exported mutable maps
- [x] R13: Uses existing `LengthSampler` interface, no new interfaces
- [x] R14: Methods are single-module
- [x] INV-6: Determinism preserved (seeded RNG, ordered iteration)
- [x] INV-10: Session causality enforced by SessionManager
- [x] INV-11: Session completeness enforced by SessionManager

---

## Appendix: File-Level Implementation Details

| File | Lines Changed (est.) | Summary |
|------|---------------------|---------|
| `sim/workload/distribution.go` | +15 | Add `SequenceSampler` struct + `Sample()` method |
| `sim/workload/distribution_test.go` | +30 (new) | 3 test functions for SequenceSampler |
| `sim/workload/session.go` | +6 | Add `ThinkTimeSampler` field, modify `OnComplete` think time logic |
| `sim/workload/session_test.go` | +35 | 2 test functions for ThinkTimeSampler behavior |
| `sim/workload/replay.go` | +120 | Add `LoadTraceV2SessionBlueprints()` function |
| `sim/workload/replay_test.go` | +100 (new) | 5 test functions for blueprint loading |
| `cmd/replay.go` | +30 | Add flags, validation, closed-loop branch, wire SessionManager |

**Total estimated:** ~336 lines added/modified across 7 files.
