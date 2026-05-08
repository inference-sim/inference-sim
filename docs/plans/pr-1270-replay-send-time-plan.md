# fix(replay): Use send_time_us as DES injection time for observed traces

**Goal:** Fix systematic TTFT pessimism in calibration by using `send_time_us` (when set) as the DES injection time instead of `arrival_time_us`, aligning `blis replay` timing with the reference used by `blis calibrate`.
**Source:** https://github.com/inference-sim/inference-sim/issues/1270
**Closes:** Fixes #1270

## Behavioral Contracts

BC-1: Concurrency-mode injection time alignment
- GIVEN a trace record where `SendTimeUs > 0` (observed trace, concurrency mode)
- WHEN `LoadTraceV2Requests` builds a `sim.Request`
- THEN `req.ArrivalTime == rec.SendTimeUs` (not `rec.ArrivalTimeUs`)

BC-2: Rate-mode and zero-send-time backward compatibility
- GIVEN a trace record where `SendTimeUs == 0` (legacy or generated trace)
- WHEN `LoadTraceV2Requests` builds a `sim.Request`
- THEN `req.ArrivalTime == rec.ArrivalTimeUs` (no behavioral change)

BC-3: Session round-0 injection time alignment
- GIVEN a session trace record (round-0) where `SendTimeUs > 0`
- WHEN `LoadTraceV2SessionBlueprints` builds the initial `sim.Request`
- THEN `req.ArrivalTime == rec.SendTimeUs`

BC-4: Non-session record injection time alignment
- GIVEN a non-session trace record where `SendTimeUs > 0`
- WHEN `LoadTraceV2SessionBlueprints` appends it as a standalone request
- THEN `req.ArrivalTime == rec.SendTimeUs`

BC-5: Think-time gap derivation unaffected
- GIVEN a multi-round session trace where `SendTimeUs > ArrivalTimeUs` for round-0
- WHEN `LoadTraceV2SessionBlueprints` derives inter-round think times
- THEN think times are derived from `ArrivalTimeUs` differences (unchanged)

## Tasks

### Task 1: Add tests for concurrency-mode injection time (BC-1, BC-2, BC-3, BC-4, BC-5)

**Files:** modify `sim/workload/replay_test.go`

**Test:**

```go
// TestLoadTraceV2Requests_ConcurrencyModeUseSendTime verifies BC-1 and BC-2.
func TestLoadTraceV2Requests_ConcurrencyModeUseSendTime(t *testing.T) {
	tests := []struct {
		name          string
		sendTimeUs    int64
		arrivalTimeUs int64
		wantArrival   int64
	}{
		{
			name:          "non-zero send_time overrides arrival_time",
			sendTimeUs:    50000, // observed trace: slot wait caused send_time > arrival_time
			arrivalTimeUs: 0,
			wantArrival:   50000,
		},
		{
			name:          "send_time equals arrival_time: returns send_time unchanged",
			sendTimeUs:    100000,
			arrivalTimeUs: 100000,
			wantArrival:   100000,
		},
		{
			name:          "zero send_time falls back to arrival_time",
			sendTimeUs:    0,
			arrivalTimeUs: 200000,
			wantArrival:   200000,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			trace := &TraceV2{
				Records: []TraceRecord{
					{
						RequestID:     0,
						ArrivalTimeUs: tc.arrivalTimeUs,
						SendTimeUs:    tc.sendTimeUs,
						InputTokens:   50,
						OutputTokens:  25,
						Status:        "ok",
					},
				},
			}
			reqs, err := LoadTraceV2Requests(trace, 42)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(reqs) != 1 {
				t.Fatalf("got %d requests, want 1", len(reqs))
			}
			if reqs[0].ArrivalTime != tc.wantArrival {
				t.Errorf("ArrivalTime = %d, want %d", reqs[0].ArrivalTime, tc.wantArrival)
			}
		})
	}
}

// TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection verifies BC-3, BC-4, BC-5.
func TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection(t *testing.T) {
	// Session round-0 with send_time > arrival_time (BC-3)
	// Non-session record with send_time > arrival_time (BC-4)
	// Think-time gap still uses arrival_time_us deltas (BC-5)
	trace := &TraceV2{
		Records: []TraceRecord{
			// Session A: round-0 delayed by concurrency slot wait (50ms)
			{RequestID: 1, SessionID: "A", RoundIndex: 0,
				ArrivalTimeUs: 0, SendTimeUs: 50000,
				InputTokens: 100, OutputTokens: 50},
			// Session A: round-1 delayed by concurrency slot wait (30ms)
			{RequestID: 2, SessionID: "A", RoundIndex: 1,
				ArrivalTimeUs: 200000, SendTimeUs: 230000,
				InputTokens: 150, OutputTokens: 60},
			// Non-session record with send_time > arrival_time (BC-4)
			{RequestID: 3, SessionID: "",
				ArrivalTimeUs: 10000, SendTimeUs: 40000,
				InputTokens: 80, OutputTokens: 30},
		},
	}

	requests, blueprints, err := LoadTraceV2SessionBlueprints(trace, 42, nil, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Find the session round-0 request and the non-session request by ArrivalTime
	sessionArrival := int64(-1)
	nonSessionArrival := int64(-1)
	for _, r := range requests {
		if r.SessionID == "A" {
			sessionArrival = r.ArrivalTime
		} else if r.SessionID == "" {
			nonSessionArrival = r.ArrivalTime
		}
	}
	if sessionArrival == -1 {
		t.Fatal("session round-0 request not found")
	}
	if nonSessionArrival == -1 {
		t.Fatal("non-session request not found")
	}

	// BC-3: session round-0 uses send_time
	if sessionArrival != 50000 {
		t.Errorf("BC-3: session round-0 ArrivalTime = %d, want 50000 (send_time_us)", sessionArrival)
	}

	// BC-4: non-session request uses send_time
	if nonSessionArrival != 40000 {
		t.Errorf("BC-4: non-session ArrivalTime = %d, want 40000 (send_time_us)", nonSessionArrival)
	}

	// BC-5: think-time gap derived from ArrivalTimeUs differences (200000 - 0 = 200000)
	if len(blueprints) != 1 {
		t.Fatalf("expected 1 blueprint, got %d", len(blueprints))
	}
	bp := blueprints[0]
	if bp.ThinkTimeSampler == nil {
		t.Fatal("expected ThinkTimeSampler for multi-round session")
	}
	gotThinkTime := bp.ThinkTimeSampler.Sample(nil)
	if gotThinkTime != 200000 {
		t.Errorf("BC-5: think time = %d, want 200000 (from ArrivalTimeUs gap, not SendTimeUs gap)", gotThinkTime)
	}
}
```

**Verify:** `go test ./sim/workload/... -run TestLoadTraceV2Requests_ConcurrencyMode -count=1`
**Verify:** `go test ./sim/workload/... -run TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection -count=1`

Expected: FAIL (tests will fail before implementation because `ArrivalTimeUs` is used for injection in all paths)

**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `test(workload): add failing tests for concurrency-mode send_time injection (BC-1..5)`

---

### Task 2: Fix injection time in LoadTraceV2Requests and LoadTraceV2SessionBlueprints (BC-1..4)

**Files:** modify `sim/workload/replay.go`

**Impl:**

Add a package-level helper (used in 3 places in this file):

```go
// injectionTime returns the DES injection time for a trace record.
// For observed traces (concurrency mode), SendTimeUs > 0 and represents
// when the request was actually sent to the server — the correct reference
// for TTFT comparison against calibrate's send_time baseline.
// Falls back to ArrivalTimeUs for generated traces (SendTimeUs == 0).
func injectionTime(rec TraceRecord) int64 {
	if rec.SendTimeUs > 0 {
		return rec.SendTimeUs
	}
	return rec.ArrivalTimeUs
}
```

Then apply in three spots in `replay.go`:

1. In `LoadTraceV2Requests`, the `req := &sim.Request{...}` block (line ~48):
   ```go
   ArrivalTime: injectionTime(rec),
   ```

2. In `LoadTraceV2SessionBlueprints`, the round-0 `req := &sim.Request{...}` block (line ~203):
   ```go
   ArrivalTime: injectionTime(r0),
   ```

3. In `LoadTraceV2SessionBlueprints`, the non-session block `req := &sim.Request{...}` (line ~253):
   ```go
   ArrivalTime: injectionTime(rec),
   ```

Note: The think-time gap loop (`rounds[i].ArrivalTimeUs - rounds[i-1].ArrivalTimeUs`) is not modified — it correctly uses `ArrivalTimeUs` to preserve client pacing semantics (BC-5).

**Verify:** `go test ./sim/workload/... -run TestLoadTraceV2Requests_ConcurrencyMode -count=1`
**Verify:** `go test ./sim/workload/... -run TestLoadTraceV2SessionBlueprints_ConcurrencyModeInjection -count=1`

Expected: PASS

**Verify all:** `go test ./sim/workload/... -count=1`

Expected: all pass (regression check)

**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `fix(replay): use send_time_us as DES injection time for observed traces (BC-1..4)`

---

## Sanity Checklist

- [ ] R1: No silent `continue` — not applicable (no new error paths)
- [ ] R3: No new numeric parameters added — not applicable
- [ ] R4: No struct literal construction sites changed (no fields added, only which value is assigned to `ArrivalTime`)
- [ ] INV-3 Clock monotonicity: `SendTimeUs >= ArrivalTimeUs` for observed traces; sim-generated traces have `SendTimeUs == ArrivalTimeUs`. No new ordering violation.
- [ ] INV-5 Causality: `req.ArrivalTime` will now be `SendTimeUs` (≥ `ArrivalTimeUs`) — causality still holds since `send_time ≤ first_chunk_time ≤ last_chunk_time`
- [ ] INV-6 Determinism: helper is pure (no RNG, no map iteration) — determinism preserved
- [ ] BC-5 think-time gap: `ArrivalTimeUs` differences in the gap loop are not touched
- [ ] Backward compat: `SendTimeUs == 0` falls back to `ArrivalTimeUs` — old traces unaffected
