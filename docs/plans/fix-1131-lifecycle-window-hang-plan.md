# Fix Lifecycle Window Hang Implementation Plan

**Goal:** Prevent workload generation from hanging when lifecycle windows are used without an explicit horizon.
**Source:** [GitHub Issue #1131](https://github.com/inference-sim/inference-sim/issues/1131)
**Closes:** `Fixes #1131`

## Behavioral Contracts

**BC-1131-1: Early exit when past all lifecycle windows (standard client path)**
- GIVEN a client with `lifecycle.windows` and the default `horizon=math.MaxInt64`
- WHEN `currentTime` exceeds the last window's `EndUs`
- THEN the generator stops producing requests for that client in bounded time (no hang)

**BC-1131-2: Early exit when past all lifecycle windows (multi-session reasoning path)**
- GIVEN a multi-session reasoning client with `lifecycle.windows` and `horizon=math.MaxInt64`
- WHEN `currentTime` exceeds the last window's `EndUs`
- THEN the generator stops producing requests for that client in bounded time (no hang)

**BC-1131-3: Gap windows still work**
- GIVEN a client with multiple non-contiguous lifecycle windows (e.g., `[0, 1s)` and `[3s, 4s)`)
- WHEN generating requests with a horizon that covers all windows
- THEN requests are produced in both windows, with no requests in the gap between them

**BC-1131-4: Behavioral equivalence with explicit horizon**
- GIVEN a client with lifecycle windows ending at time T
- WHEN generating with `horizon=MaxInt64` (implicit) versus `horizon=2*T` (explicit, well beyond last window)
- THEN both produce the same requests (same count, same arrival times)

## Tasks

### Task 1: Add `lastWindowEndUs` helper and early-exit test (BC-1, BC-4)

**Files:** modify `sim/workload/generator.go`, modify `sim/workload/generator_test.go`

**Test:**

Add a table-driven test `TestGenerateRequests_LifecycleWindow_NoHang` that exercises:
- A single client with one lifecycle window `[0, 5_000_000)`, `horizon=math.MaxInt64`, `num_requests=0` (unlimited). Asserts: (a) the function returns without hanging (enforced by the test runner's timeout), (b) all returned requests have `ArrivalTime < 5_000_000`, (c) at least 1 request is produced.

```go
func TestGenerateRequests_LifecycleWindow_NoHang(t *testing.T) {
	// BC-1131-1: Generator must exit in bounded time when lifecycle windows
	// end well before MaxInt64 horizon. Before the fix, this hangs forever.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "phase1", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 256, "std_dev": 64, "min": 64, "max": 512}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "std_dev": 64, "min": 32, "max": 512}},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 5_000_000}},
			},
		}},
	}
	requests, err := GenerateRequests(spec, math.MaxInt64, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	for i, req := range requests {
		if req.ArrivalTime >= 5_000_000 {
			t.Errorf("request %d: ArrivalTime=%d outside window [0, 5000000)", i, req.ArrivalTime)
		}
	}
}
```

**Impl:**

Add a `lastWindowEndUs` helper function to `generator.go`:

```go
// lastWindowEndUs returns the maximum EndUs across all lifecycle windows.
func lastWindowEndUs(lifecycle *LifecycleSpec) int64 {
	var maxEnd int64
	for _, w := range lifecycle.Windows {
		if w.EndUs > maxEnd {
			maxEnd = w.EndUs
		}
	}
	return maxEnd
}
```

Then modify the standard client loop (line ~284) to break instead of continuing when past all windows:

```go
// Check lifecycle windows
if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
	if currentTime >= lastWindowEndUs(client.Lifecycle) {
		break
	}
	continue
}
```

And apply the same fix to the multi-session reasoning loop (line ~222):

```go
// Check lifecycle windows (bug fix: reasoning path was missing this)
if client.Lifecycle != nil && !isInActiveWindow(currentTime, client.Lifecycle) {
	if currentTime >= lastWindowEndUs(client.Lifecycle) {
		break
	}
	continue
}
```

**Verify:** `go test ./sim/workload/... -run TestGenerateRequests_LifecycleWindow_NoHang -timeout 30s`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `fix(workload): break generator loop when past last lifecycle window (BC-1, BC-2)`

### Task 2: Multi-window gap test (BC-3)

**Files:** modify `sim/workload/generator_test.go`

**Test:**

Add `TestGenerateRequests_LifecycleWindow_MultipleWindows` verifying requests appear in both windows but not in the gap:

```go
func TestGenerateRequests_LifecycleWindow_MultipleWindows(t *testing.T) {
	// BC-1131-3: Multiple non-contiguous windows must all produce requests,
	// with no requests in gaps between windows.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 50.0,
		Clients: []ClientSpec{{
			ID: "multi-win", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{
					{StartUs: 0, EndUs: 1_000_000},          // window 1: [0, 1s)
					{StartUs: 3_000_000, EndUs: 4_000_000},  // window 2: [3s, 4s)
				},
			},
		}},
	}
	requests, err := GenerateRequests(spec, math.MaxInt64, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests")
	}

	var inWindow1, inWindow2 int
	for i, req := range requests {
		arrTime := req.ArrivalTime
		inW1 := arrTime >= 0 && arrTime < 1_000_000
		inW2 := arrTime >= 3_000_000 && arrTime < 4_000_000
		if !inW1 && !inW2 {
			t.Errorf("request %d: ArrivalTime=%d outside all windows", i, req.ArrivalTime)
		}
		if inW1 {
			inWindow1++
		}
		if inW2 {
			inWindow2++
		}
	}
	if inWindow1 == 0 {
		t.Error("no requests in window 1 [0, 1s)")
	}
	if inWindow2 == 0 {
		t.Error("no requests in window 2 [3s, 4s)")
	}
}
```

Note: The `break` in Task 1 only fires when past ALL windows. For multiple windows, `lastWindowEndUs` returns the max end (4s in this case), so the loop keeps running through the gap (IAT samples skip past it via `continue`) and produces requests in window 2. The `break` only fires once `currentTime >= 4_000_000`.

**Impl:** No production code changes needed. This test validates that the Task 1 fix correctly handles multi-window gaps.

**Verify:** `go test ./sim/workload/... -run TestGenerateRequests_LifecycleWindow_MultipleWindows -timeout 30s`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `test(workload): add multi-window gap coverage for lifecycle early exit (BC-3)`

### Task 3: Behavioral equivalence test (BC-4)

**Files:** modify `sim/workload/generator_test.go`

**Test:**

Add `TestGenerateRequests_LifecycleWindow_EquivalentWithExplicitHorizon` verifying that implicit `MaxInt64` horizon and explicit large horizon produce identical results:

```go
func TestGenerateRequests_LifecycleWindow_EquivalentWithExplicitHorizon(t *testing.T) {
	// BC-1131-4: MaxInt64 horizon must produce the same requests as an explicit
	// horizon well beyond the last window. This proves the early-exit is
	// a pure optimization, not a behavioral change.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "equiv", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 256, "std_dev": 64, "min": 64, "max": 512}},
			OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "std_dev": 64, "min": 32, "max": 512}},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 5_000_000}},
			},
		}},
	}

	// Run with MaxInt64 (the fix path)
	r1, err := GenerateRequests(spec, math.MaxInt64, 0)
	if err != nil {
		t.Fatalf("MaxInt64 horizon: %v", err)
	}

	// Run with explicit horizon well beyond the window (10s > 5s window)
	r2, err := GenerateRequests(spec, 10_000_000, 0)
	if err != nil {
		t.Fatalf("explicit horizon: %v", err)
	}

	// Same count and arrival times (INV-6: determinism given same seed)
	if len(r1) != len(r2) {
		t.Fatalf("request count: MaxInt64=%d, explicit=%d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
	}
}
```

**Impl:** No production code changes needed.

**Verify:** `go test ./sim/workload/... -run TestGenerateRequests_LifecycleWindow_EquivalentWithExplicitHorizon -timeout 30s`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `test(workload): add equivalence test for lifecycle early exit (BC-4)`

### Task 4: Multi-session reasoning path hang test (BC-2)

**Files:** modify `sim/workload/generator_test.go`

**Test:**

Add `TestGenerateRequests_ReasoningMultiSession_LifecycleNoHang` covering the reasoning multi-session loop:

```go
func TestGenerateRequests_ReasoningMultiSession_LifecycleNoHang(t *testing.T) {
	// BC-1131-2: Multi-session reasoning path must also exit when past all windows.
	spec := &WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "reason-lc", TenantID: "t1", SLOClass: "standard", RateFraction: 1.0,
			Arrival:    ArrivalSpec{Process: "poisson"},
			InputDist:  DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
			OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
			Reasoning: &ReasoningSpec{
				ReasonRatioDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 0}},
				MultiTurn:       &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 50_000, ContextGrowth: "accumulate"},
			},
			Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 2_000_000}},
			},
		}},
	}
	requests, err := GenerateRequests(spec, math.MaxInt64, 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	for i, req := range requests {
		if req.ArrivalTime >= 2_000_000 {
			t.Errorf("request %d: ArrivalTime=%d outside window [0, 2000000)", i, req.ArrivalTime)
		}
	}
}
```

**Impl:** Already done in Task 1 (both loops were fixed together).

**Verify:** `go test ./sim/workload/... -run TestGenerateRequests_ReasoningMultiSession_LifecycleNoHang -timeout 30s`
**Lint:** `golangci-lint run ./sim/workload/...`
**Commit:** `test(workload): add reasoning multi-session lifecycle hang test (BC-2)`

## Sanity Checklist

- [x] **R1 (silent continue):** The `continue` is preserved for in-gap timestamps; `break` added only past the last window end. No data is silently discarded.
- [x] **R2 (determinism, INV-6):** No map iteration, no new non-determinism. BC-4 explicitly verifies deterministic equivalence.
- [x] **R4 (canonical constructors):** No new structs or struct fields. Only a new free function `lastWindowEndUs`.
- [x] **R7 (golden values):** No golden values — all tests assert behavioral invariants (arrival times within windows, bounded termination, equivalence).
- [x] **R12 (behavioral tests):** All four tests assert observable behavior (arrival time bounds, count > 0, equivalence). No structural assertions.
- [x] **INV-6 (determinism):** BC-4 test verifies identical output for same seed.
- [x] **Regression:** Existing lifecycle tests (`TestGenerateRequests_SingleSession_LifecycleWindowRoundSuppression`, `TestGenerateRequests_ReasoningClient_RespectsLifecycleWindows`) continue to pass unchanged.
- [x] **No new interfaces/types/CLI flags.**
- [x] **No documentation changes needed:** This is a bug fix for internal generator behavior, not a user-facing feature change. Lifecycle windows and horizon flag are already documented.
