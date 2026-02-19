# Phase 4: Invariant Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add standalone invariant tests that verify the five core simulation laws — request conservation, KV block conservation, causality, clock monotonicity, and determinism — independently of golden dataset values.

**The problem today:** The simulator's correctness depends almost entirely on golden dataset tests, which compare output against frozen "known-good" values. If a bug exists when the golden dataset is generated, the test encodes the bug as expected behavior. This actually happened: issue #183 found that 499 completions (instead of 500) was treated as correct because one request was silently dropped. Invariant tests derived from the *specification* would have caught this on day one.

**What this PR adds:**
1. **Request conservation tests** — verify `injected == completed + queued + running` after every simulation, including finite-horizon runs where not all requests complete
2. **Enhanced causality tests** — verify the full chain `ArrivalTime <= TTFT <= CompletionTime` and that all ITL values are non-negative, at both single-instance and cluster levels
3. **Clock monotonicity tests** — step-by-step instrumentation proving the clock never decreases, at both single-instance and cluster levels
4. **Byte-identical determinism test** — verify same seed produces identical JSON output via `SaveResults`, not just field-by-field integer comparison
5. **KV block conservation via KVStore interface** — verify `UsedBlocks() == 0` after full simulation through the KVStore interface (complementing the existing `KVCacheState`-specific `assertBlockConservation` helper)

**Why this matters:** These invariant tests form the safety net that prevents regression across PR11 (autoscaling), PR14 (P/D disaggregation), and all future extensions. They verify laws the system must satisfy regardless of output values.

**Architecture:** All tests live in existing test files (`sim/simulator_test.go`, `sim/cluster/cluster_test.go`, `sim/kvcache_test.go`). No production code changes. Tests use manual event-loop stepping (`HasPendingEvents` + `ProcessNextEvent`) to instrument clock monotonicity without modifying the simulator.

**Source:** GitHub issue #211, design doc `docs/plans/2026-02-18-hardening-antipattern-refactoring-design.md` Phase 4, relates to #199

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds standalone invariant tests for the five core simulation laws. No production code is modified — only test files are created or extended. The tests complement (not replace) the existing golden dataset tests, which already embed partial invariant checks from the Phase 2/3 hardening work.

The PR sits after the Phase 1-3 hardening fixes (which established helpers and correctness) and before PR11 (autoscaling). The invariant tests will catch regressions as the architecture evolves.

Adjacent blocks: `sim.Simulator` (event loop, metrics, queue, batch), `sim/cluster.ClusterSimulator` (shared clock, multi-instance), `sim.KVStore` (block accounting). No modifications to any of these — tests observe their public API.

No deviations flagged from Phase 0.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Request Conservation (Infinite Horizon)
- GIVEN a simulator with Horizon=MaxInt64 and N injected requests
- WHEN the simulation runs to completion
- THEN CompletedRequests == N AND WaitQ.Len() == 0 AND len(RunningBatch.Requests) == 0
- MECHANISM: Infinite horizon forces all requests through the full lifecycle

BC-2: Request Conservation (Finite Horizon)
- GIVEN a simulator with a finite Horizon and N injected requests spread across time
- WHEN the simulation ends (some requests may not complete)
- THEN `len(Metrics.Requests) == completed + WaitQ.Len() + len(RunningBatch.Requests)` (three-term equation)
- MECHANISM: Manually count queued + running after Run() returns; compare against registered requests

BC-3: Request Conservation (Cluster Level)
- GIVEN a ClusterSimulator with N instances and M total requests
- WHEN the cluster simulation runs to completion
- THEN sum of per-instance CompletedRequests == M (infinite horizon) AND aggregated CompletedRequests == M
- MECHANISM: Sum across all instances, compare with injection count

BC-4: Causality (Full Chain, Single-Instance)
- GIVEN a completed simulation with at least one completed request
- WHEN examining per-request metrics
- THEN for every completed request: ArrivalTime <= FirstTokenTime AND TTFT <= E2E AND all ITL values >= 0
- MECHANISM: Iterate Metrics.Requests cross-referenced with RequestTTFTs, RequestE2Es, AllITLs

BC-5: Causality (Cluster Level)
- GIVEN a cluster simulation with multiple instances
- WHEN examining per-instance metrics
- THEN the same causality chain holds for every completed request in every instance
- MECHANISM: Iterate each instance's Metrics independently

BC-6: Clock Monotonicity (Single-Instance)
- GIVEN a simulator with injected requests
- WHEN processing events one at a time via ProcessNextEvent()
- THEN the Clock value after each event MUST be >= the Clock value after the previous event
- MECHANISM: Manual event-loop stepping with clock recording between each step

BC-7: Clock Monotonicity (Cluster Level)
- GIVEN a cluster simulation
- WHEN the simulation completes
- THEN cluster.Clock() >= every instance's Clock()
- MECHANISM: Post-simulation comparison (strengthening existing test with multi-request workload)

BC-8: Determinism (Byte-Identical JSON)
- GIVEN two simulator runs with identical SimConfig and seed
- WHEN both runs save results via SaveResults to temp files
- THEN the two output files are byte-identical
- MECHANISM: Write to temp files, compare bytes

BC-9: Determinism (Cluster Level)
- GIVEN two cluster runs with identical config and seed
- WHEN both aggregate metrics
- THEN all integer metrics match exactly AND per-request metrics (sorted by ID, serialized to JSON) are byte-identical
- MECHANISM: Field-by-field integer comparison + sorted JSON comparison of request-level metrics

BC-10: KV Block Conservation (Post-Simulation via KVStore Interface)
- GIVEN a completed simulation (infinite horizon, all requests complete)
- WHEN checking block accounting
- THEN `KVCache.UsedBlocks() == 0` (all blocks released)
- MECHANISM: Already partially tested in golden dataset; this makes it standalone with multiple configs

**Negative contracts:**

NC-1: No Negative ITL
- GIVEN any simulation run
- WHEN examining inter-token latencies
- THEN no ITL value is negative (would imply time travel)
- MECHANISM: Check AllITLs slice after simulation

NC-2: No Clock Regression
- GIVEN any simulation event sequence
- WHEN processing events
- THEN `clock[i+1] >= clock[i]` for all consecutive events
- MECHANISM: BC-6 assertion applied exhaustively

### C) Component Interaction

```
┌────────────────────────────────────────────────────────┐
│  Invariant Test Suite (test files only)                 │
│                                                         │
│  sim/simulator_test.go:                                 │
│    BC-1, BC-2, BC-4, BC-6, BC-8, BC-10, NC-1, NC-2    │
│                                                         │
│  sim/cluster/cluster_test.go:                           │
│    BC-3, BC-5, BC-7, BC-9                              │
│                                                         │
│  (sim/kvcache_test.go: existing BC-10 coverage from     │
│    Phase 3 — assertBlockConservation helper)            │
└────────────────┬───────────────────────────────────────┘
                 │ observes (public API only)
                 ▼
┌────────────────────────────────────────────────────────┐
│  sim.Simulator         sim/cluster.ClusterSimulator     │
│  - Clock, WaitQ        - Clock(), Instances()           │
│  - RunningBatch        - AggregatedMetrics()            │
│  - Metrics.*           - Run()                          │
│  - KVCache (KVStore)                                    │
│  - ProcessNextEvent()                                   │
│  - HasPendingEvents()                                   │
└────────────────────────────────────────────────────────┘
```

No new APIs, types, or state. No state ownership changes. Extension friction: 0 files (pure test addition).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| 4f: "Regenerate golden dataset after fixes" | Not included | Golden dataset already passes on current code; Phases 1-3 were designed to avoid golden dataset changes. Regeneration is a no-op. |
| 4d: "Instrument event processing to record clock at each step" | Uses manual step-by-step loop, not callback instrumentation | Manual stepping via `HasPendingEvents()`/`ProcessNextEvent()` achieves the same coverage without production code changes. Callback instrumentation would add dead code in production. |
| Design doc suggests `sim/kvcache_test.go` for 4b | Block conservation for KVStore interface added to `sim/simulator_test.go` (post-simulation check); no changes to `kvcache_test.go` | Post-simulation `UsedBlocks() == 0` is a complementary check. The full `used + free == total` equation from 4b is already covered by Phase 3's `assertBlockConservation` helper in `kvcache_test.go` (unit-level allocate/release cycles). |
| Design doc 4a-4e are single-instance scoped | Micro plan adds cluster-level tests (BC-3, BC-5, BC-7, BC-9) | ADDITION: The design doc only explicitly mentions cluster-level for 4d (clock monotonicity). Cluster-level conservation, causality, and determinism are natural extensions that strengthen coverage. |
| Issue #199 mentions "state machine" invariant (queued → running → completed) | Not included | DEFERRAL: The design doc Phase 4 does not include state machine tests. Phase 6 (Modularity) introduces typed `RequestState` constants — state machine invariant tests are more appropriate after those constants land. |

### E) Review Guide

1. **THE TRICKY PART:** BC-2 (finite horizon conservation) — the three-term equation `completed + queued + running == injected` requires careful counting after `Run()` returns. The test must use a horizon that cuts off mid-simulation so some requests are genuinely still queued/running.
2. **WHAT TO SCRUTINIZE:** BC-6 (clock monotonicity) — the manual event loop must faithfully replicate `Simulator.Run()` without missing the horizon check or causing divergent behavior.
3. **WHAT'S SAFE TO SKIM:** BC-1, BC-3, BC-10 — straightforward post-simulation assertions, minimal logic.
4. **KNOWN DEBT:** The existing `TestSimulator_DeterministicWorkload` only compares integers. BC-8 supersedes it with byte-identical JSON comparison, but the old test is not removed (it provides faster feedback on obvious determinism breaks).

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/simulator_test.go` — add 6 test functions (BC-1 standalone, BC-2, BC-4 full chain, BC-6, BC-8, BC-10)
- `sim/cluster/cluster_test.go` — add 3 test functions (BC-3 standalone, BC-5, BC-7 strengthened, BC-9)

**Files unchanged:** No production code changes.

**Key decisions:**
- Clock monotonicity uses manual event-loop stepping, not production instrumentation
- Determinism uses temp file comparison via `SaveResults`, not in-memory JSON marshaling
- Finite horizon conservation uses manual injection with deterministic early/late request timing (not fragile horizon tuning)

**Confirmation:** No dead code — every test is independently runnable.

### G) Task Breakdown

---

### Task 1: Request Conservation — Standalone Single-Instance Tests

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write failing tests**

Context: The golden dataset test already checks `completed == injected` inline. We need standalone tests: one for infinite horizon (all complete) and one for finite horizon (three-term equation).

```go
// TestSimulator_RequestConservation_InfiniteHorizon_AllRequestsComplete verifies BC-1:
// GIVEN a simulator with Horizon=MaxInt64 and 50 injected requests
// WHEN the simulation runs to completion
// THEN CompletedRequests == 50 AND WaitQ is empty AND RunningBatch is empty.
func TestSimulator_RequestConservation_InfiniteHorizon_AllRequestsComplete(t *testing.T) {
	cfg := SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               99,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-conservation",
		GPU:                "H100",
		TP:                 1,
		GuideLLMConfig: &GuideLLMConfig{
			Rate: 10.0 / 1e6, MaxPrompts: 50,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		},
	}

	sim := NewSimulator(cfg)
	sim.Run()

	// Three-term equation: injected == completed + queued + running
	injected := len(sim.Metrics.Requests)
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}

	if completed+queued+running != injected {
		t.Errorf("request conservation violated: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
			completed, queued, running, completed+queued+running, injected)
	}

	// With infinite horizon, all should complete
	if completed != 50 {
		t.Errorf("infinite horizon: expected all 50 requests to complete, got %d", completed)
	}
	if queued != 0 {
		t.Errorf("infinite horizon: expected empty queue, got %d queued", queued)
	}
	if running != 0 {
		t.Errorf("infinite horizon: expected empty batch, got %d running", running)
	}
}

// TestSimulator_RequestConservation_FiniteHorizon_ThreeTermEquation verifies BC-2:
// GIVEN a simulator with a finite Horizon and requests injected at staggered times
//   (all arriving BEFORE the horizon, but late ones too large to complete)
// WHEN the simulation ends
// THEN completed + queued + running == injected (three-term conservation).
func TestSimulator_RequestConservation_FiniteHorizon_ThreeTermEquation(t *testing.T) {
	// CRITICAL: All injected requests MUST have ArrivalTime < Horizon.
	// Requests with ArrivalTime > Horizon have their ArrivalEvent never popped
	// from the event queue — they'd be in Metrics.Requests but NOT in
	// WaitQ/RunningBatch/completed, breaking the three-term equation.
	//
	// Strategy: inject early (small, fast) and late (large, slow) requests.
	// Early requests complete before the horizon. Late requests arrive before
	// the horizon but are too large to finish processing.
	cfg := SimConfig{
		Horizon:            500_000, // 0.5 seconds in ticks
		Seed:               42,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-conservation-finite",
		GPU:                "H100",
		TP:                 1,
	}

	sim := NewSimulator(cfg)

	// Inject 10 early requests (small, will complete before horizon)
	for i := 0; i < 10; i++ {
		sim.InjectArrival(&Request{
			ID:           fmt.Sprintf("early_%d", i),
			ArrivalTime:  int64(i * 10000), // 0 to 90,000 ticks (well before 500k horizon)
			InputTokens:  make([]int, 20),
			OutputTokens: make([]int, 5),
			State:        "queued",
		})
	}

	// Inject 5 late requests (large, arrive before horizon but won't complete)
	for i := 0; i < 5; i++ {
		sim.InjectArrival(&Request{
			ID:           fmt.Sprintf("late_%d", i),
			ArrivalTime:  int64(300_000 + i*40_000), // 300,000 to 460,000 (all < 500k horizon)
			InputTokens:  make([]int, 200),           // large prefill
			OutputTokens: make([]int, 100),           // many decode tokens
			State:        "queued",
		})
	}

	sim.Run()

	injected := len(sim.Metrics.Requests)
	completed := sim.Metrics.CompletedRequests
	queued := sim.WaitQ.Len()
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}

	if completed+queued+running != injected {
		t.Errorf("request conservation violated: completed(%d) + queued(%d) + running(%d) = %d, injected = %d",
			completed, queued, running, completed+queued+running, injected)
	}

	// Verify we actually tested the non-trivial case: some but not all completed
	if completed == injected {
		t.Fatalf("all %d requests completed — horizon too long, three-term case untested", injected)
	}
	if completed == 0 {
		t.Fatalf("no requests completed — horizon too short, test setup invalid")
	}
}
```

**Step 2: Run test to verify it passes (these are additive invariant tests, not TDD against new code)**

Run: `go test ./sim/... -run TestSimulator_RequestConservation -v`
Expected: PASS (invariants should hold on current code)

**Step 3: No implementation needed — pure test addition**

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add standalone request conservation invariant tests (BC-1, BC-2)

- Infinite horizon: completed == injected, queue and batch empty
- Finite horizon: completed + queued + running == injected (three-term)
- Independent of golden dataset values (issue #199)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Causality — Full Chain Tests

**Contracts Implemented:** BC-4, NC-1

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write tests**

Context: The existing golden dataset test only checks `E2E >= TTFT`. We need the full chain: `ArrivalTime <= FirstTokenTime`, plus `ITL >= 0` for all inter-token latencies.

```go
// TestSimulator_Causality_FullChain_ArrivalToCompletion verifies BC-4:
// GIVEN a completed simulation with multiple requests
// WHEN examining per-request timing metrics
// THEN for every completed request:
//   - ArrivalTime (in ms) <= TTFT (in ms)
//   - TTFT <= E2E
//   - All ITL values >= 0
func TestSimulator_Causality_FullChain_ArrivalToCompletion(t *testing.T) {
	cfg := SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               77,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-causality",
		GPU:                "H100",
		TP:                 1,
		GuideLLMConfig: &GuideLLMConfig{
			Rate: 5.0 / 1e6, MaxPrompts: 30,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		},
	}

	sim := NewSimulator(cfg)
	sim.Run()

	if sim.Metrics.CompletedRequests == 0 {
		t.Fatal("no completed requests — test setup invalid")
	}

	for id, rm := range sim.Metrics.Requests {
		ttft, hasTTFT := sim.Metrics.RequestTTFTs[id]
		e2e, hasE2E := sim.Metrics.RequestE2Es[id]

		if !hasTTFT || !hasE2E {
			continue // incomplete request (finite horizon) — skip
		}

		// ArrivedAt is in seconds (float64), TTFT is in milliseconds
		// Convert ArrivedAt to ms for comparison: arrivedAt_ms = arrivedAt_s * 1000
		arrivedAtMs := rm.ArrivedAt * 1000.0

		// Causality: arrival <= TTFT
		if arrivedAtMs > ttft {
			t.Errorf("causality violated for %s: ArrivedAt (%.2f ms) > TTFT (%.2f ms)", id, arrivedAtMs, ttft)
		}

		// Causality: TTFT <= E2E
		if ttft > e2e {
			t.Errorf("causality violated for %s: TTFT (%.2f ms) > E2E (%.2f ms)", id, ttft, e2e)
		}
	}

	// NC-1: All ITL values must be non-negative
	for i, itl := range sim.Metrics.AllITLs {
		if itl < 0 {
			t.Errorf("negative ITL at index %d: %d (time travel)", i, itl)
		}
	}
}
```

**Step 2: Run test**

Run: `go test ./sim/... -run TestSimulator_Causality_FullChain -v`
Expected: PASS

**Step 3: No implementation needed**

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add full causality chain invariant test (BC-4, NC-1)

- Verify ArrivalTime <= TTFT <= E2E for all completed requests
- Verify all ITL values >= 0 (no time travel)
- Independent of golden dataset values (issue #199)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Clock Monotonicity — Single-Instance Step-by-Step

**Contracts Implemented:** BC-6, NC-2

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write test**

Context: We manually drive the event loop via `HasPendingEvents()`/`ProcessNextEvent()` and record the clock after each event, verifying it never decreases.

```go
// TestSimulator_ClockMonotonicity_NeverDecreases verifies BC-6:
// GIVEN a simulator with injected requests
// WHEN processing events one at a time
// THEN the Clock value never decreases between consecutive events.
func TestSimulator_ClockMonotonicity_NeverDecreases(t *testing.T) {
	cfg := SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               55,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-monotonicity",
		GPU:                "H100",
		TP:                 1,
		GuideLLMConfig: &GuideLLMConfig{
			Rate: 10.0 / 1e6, MaxPrompts: 20,
			PromptTokens: 50, PromptTokensStdDev: 10, PromptTokensMin: 10, PromptTokensMax: 100,
			OutputTokens: 20, OutputTokensStdDev: 5, OutputTokensMin: 5, OutputTokensMax: 40,
		},
	}

	sim := NewSimulator(cfg)

	prevClock := int64(0)
	eventCount := 0
	for sim.HasPendingEvents() {
		sim.ProcessNextEvent()
		eventCount++

		if sim.Clock < prevClock {
			t.Fatalf("clock monotonicity violated at event %d: clock went from %d to %d",
				eventCount, prevClock, sim.Clock)
		}
		prevClock = sim.Clock

		if sim.Clock > sim.Horizon {
			break
		}
	}
	sim.Finalize()

	if eventCount == 0 {
		t.Fatal("no events processed — test setup invalid")
	}
	t.Logf("clock monotonicity held across %d events (final clock: %d)", eventCount, sim.Clock)
}
```

**Step 2: Run test**

Run: `go test ./sim/... -run TestSimulator_ClockMonotonicity -v`
Expected: PASS

**Step 3: No implementation needed**

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add clock monotonicity invariant test (BC-6, NC-2)

- Step-by-step event processing verifying clock never decreases
- Manual event loop mirrors Simulator.Run() without modification
- Issue #199

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Determinism — Byte-Identical JSON Output

**Contracts Implemented:** BC-8

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write test**

Context: Existing determinism tests compare individual integer fields. This test runs `SaveResults` to temp files and compares the normalized JSON, catching any non-determinism in serialization (e.g., map iteration order). Note: `SaveResults` embeds `time.Now()` for `sim_end_timestamp` and `simulation_duration_s`, which are wall-clock dependent. The test normalizes these fields by unmarshaling, zeroing them, and re-marshaling before comparison. A fixed `startTime` is passed to both calls to control `sim_start_timestamp`.

```go
// TestSimulator_Determinism_ByteIdenticalJSON verifies BC-8:
// GIVEN two simulator runs with identical config and seed
// WHEN both save results via SaveResults to temp files
// THEN the output files are identical after stripping wall-clock timestamps.
func TestSimulator_Determinism_ByteIdenticalJSON(t *testing.T) {
	cfg := SimConfig{
		Horizon:            math.MaxInt64,
		Seed:               42,
		TotalKVBlocks:      10000,
		BlockSizeTokens:    16,
		MaxRunningReqs:     256,
		MaxScheduledTokens: 2048,
		BetaCoeffs:         []float64{1000, 10, 5},
		AlphaCoeffs:        []float64{100, 1, 100},
		Model:              "test-determinism",
		GPU:                "H100",
		TP:                 1,
		GuideLLMConfig: &GuideLLMConfig{
			Rate: 5.0 / 1e6, MaxPrompts: 20,
			PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
			OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
		},
	}

	fixedTime := time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

	// Run 1
	sim1 := NewSimulator(cfg)
	sim1.Run()
	f1 := t.TempDir() + "/run1.json"
	sim1.Metrics.SaveResults("determinism-test", cfg.Horizon, cfg.TotalKVBlocks, fixedTime, f1)

	// Run 2
	sim2 := NewSimulator(cfg)
	sim2.Run()
	f2 := t.TempDir() + "/run2.json"
	sim2.Metrics.SaveResults("determinism-test", cfg.Horizon, cfg.TotalKVBlocks, fixedTime, f2)

	data1, err1 := os.ReadFile(f1)
	if err1 != nil {
		t.Fatalf("failed to read run1 output: %v", err1)
	}
	data2, err2 := os.ReadFile(f2)
	if err2 != nil {
		t.Fatalf("failed to read run2 output: %v", err2)
	}

	// Zero out the sim_end_timestamp which uses time.Now()
	var out1, out2 MetricsOutput
	if err := json.Unmarshal(data1, &out1); err != nil {
		t.Fatalf("failed to unmarshal run1: %v", err)
	}
	if err := json.Unmarshal(data2, &out2); err != nil {
		t.Fatalf("failed to unmarshal run2: %v", err)
	}
	out1.SimEndTimestamp = ""
	out2.SimEndTimestamp = ""
	out1.SimulationDurationSec = 0
	out2.SimulationDurationSec = 0

	norm1, _ := json.MarshalIndent(out1, "", "  ")
	norm2, _ := json.MarshalIndent(out2, "", "  ")

	if !bytes.Equal(norm1, norm2) {
		t.Error("determinism violation: normalized JSON differs between runs")
		// Find first difference for debugging
		lines1 := bytes.Split(norm1, []byte("\n"))
		lines2 := bytes.Split(norm2, []byte("\n"))
		maxLines := len(lines1)
		if len(lines2) > maxLines {
			maxLines = len(lines2)
		}
		for i := 0; i < maxLines; i++ {
			var l1, l2 []byte
			if i < len(lines1) {
				l1 = lines1[i]
			}
			if i < len(lines2) {
				l2 = lines2[i]
			}
			if !bytes.Equal(l1, l2) {
				t.Errorf("first difference at line %d:\n  run1: %s\n  run2: %s", i+1, l1, l2)
				break
			}
		}
	}
}
```

**Step 3: No implementation needed**

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add byte-identical JSON determinism test (BC-8)

- Save results via SaveResults, compare normalized JSON
- Strip wall-clock timestamps (sim_end_timestamp, simulation_duration)
- Catches serialization non-determinism (map iteration order)
- Issue #199

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: KV Block Conservation — Post-Simulation via KVStore Interface

**Contracts Implemented:** BC-10

**Files:**
- Modify: `sim/simulator_test.go`

**Step 1: Write test**

Context: The golden dataset test already checks `UsedBlocks() == 0` inline. This makes it a standalone test with table-driven configs (single-tier and tiered).

```go
// TestSimulator_KVBlockConservation_PostSimulation_ZeroLeak verifies BC-10:
// GIVEN a completed simulation with all requests finished (infinite horizon)
// WHEN checking KV block accounting via KVStore interface
// THEN UsedBlocks() == 0 (no leaked blocks).
func TestSimulator_KVBlockConservation_PostSimulation_ZeroLeak(t *testing.T) {
	tests := []struct {
		name        string
		kvCPUBlocks int64
	}{
		{"single-tier", 0},
		{"tiered-gpu-cpu", 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := SimConfig{
				Horizon:            math.MaxInt64,
				Seed:               42,
				TotalKVBlocks:      10000,
				BlockSizeTokens:    16,
				MaxRunningReqs:     256,
				MaxScheduledTokens: 2048,
				BetaCoeffs:         []float64{1000, 10, 5},
				AlphaCoeffs:        []float64{100, 1, 100},
				Model:              "test-kv-conservation",
				GPU:                "H100",
				TP:                 1,
				KVCPUBlocks:        tt.kvCPUBlocks,
				KVOffloadThreshold: 0.8,
				KVTransferBandwidth: 100.0,
				GuideLLMConfig: &GuideLLMConfig{
					Rate: 5.0 / 1e6, MaxPrompts: 20,
					PromptTokens: 100, PromptTokensStdDev: 20, PromptTokensMin: 10, PromptTokensMax: 200,
					OutputTokens: 50, OutputTokensStdDev: 10, OutputTokensMin: 10, OutputTokensMax: 100,
				},
			}

			sim := NewSimulator(cfg)
			sim.Run()

			if sim.KVCache.UsedBlocks() != 0 {
				t.Errorf("KV block leak: %d blocks still allocated after all requests completed (KVStore interface)",
					sim.KVCache.UsedBlocks())
			}

			// Conservation: remaining used should be zero, total capacity unchanged
			if sim.KVCache.TotalCapacity() != cfg.TotalKVBlocks {
				t.Errorf("TotalCapacity changed: got %d, want %d", sim.KVCache.TotalCapacity(), cfg.TotalKVBlocks)
			}
		})
	}
}
```

**Step 2: Run test**

Run: `go test ./sim/... -run TestSimulator_KVBlockConservation_PostSimulation -v`
Expected: PASS

**Step 3: No implementation needed**

**Step 4: Run lint**

Run: `golangci-lint run ./sim/...`

**Step 5: Commit**

```bash
git add sim/simulator_test.go
git commit -m "test(sim): add KV block conservation post-simulation test (BC-10)

- Verify UsedBlocks() == 0 after full simulation via KVStore interface
- Table-driven: single-tier and tiered configurations
- Independent of golden dataset values (issue #199)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Cluster-Level Invariants — Conservation, Causality, Clock, Determinism

**Contracts Implemented:** BC-3, BC-5, BC-7, BC-9

**Files:**
- Modify: `sim/cluster/cluster_test.go`

**Step 1: Write tests**

Context: The cluster layer needs its own invariant tests independent of the single-instance tests. BC-3 (conservation), BC-5 (causality), BC-7 (strengthened clock monotonicity), and BC-9 (byte-identical determinism).

```go
// TestClusterSimulator_RequestConservation_SumAcrossInstances verifies BC-3:
// GIVEN N=4 instances and 100 requests
// WHEN the cluster simulation completes (infinite horizon)
// THEN sum of per-instance CompletedRequests == 100 == aggregated CompletedRequests.
func TestClusterSimulator_RequestConservation_SumAcrossInstances(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	sumCompleted := 0
	for _, inst := range cs.Instances() {
		sumCompleted += inst.Metrics().CompletedRequests
	}

	agg := cs.AggregatedMetrics()

	// Conservation: sum of parts == whole
	if sumCompleted != agg.CompletedRequests {
		t.Errorf("conservation: sum of instance completions (%d) != aggregated (%d)",
			sumCompleted, agg.CompletedRequests)
	}

	// Conservation: injected == completed
	if agg.CompletedRequests != 100 {
		t.Errorf("conservation: aggregated completions (%d) != injected (100)",
			agg.CompletedRequests)
	}
}

// TestClusterSimulator_Causality_PerInstance verifies BC-5:
// GIVEN a cluster simulation with multiple instances
// WHEN examining per-instance metrics
// THEN for every completed request: TTFT <= E2E and all ITL >= 0.
func TestClusterSimulator_Causality_PerInstance(t *testing.T) {
	config := newTestDeploymentConfig(3)
	workload := newTestWorkload(50)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	totalChecked := 0
	for idx, inst := range cs.Instances() {
		m := inst.Metrics()
		for id, ttft := range m.RequestTTFTs {
			e2e, ok := m.RequestE2Es[id]
			if !ok {
				continue
			}
			if ttft > e2e {
				t.Errorf("causality violated: instance %d, request %s: TTFT (%.2f) > E2E (%.2f)",
					idx, id, ttft, e2e)
			}
			totalChecked++
		}

		for i, itl := range m.AllITLs {
			if itl < 0 {
				t.Errorf("negative ITL: instance %d, index %d: %d", idx, i, itl)
			}
		}
	}

	if totalChecked == 0 {
		t.Fatal("no completed requests checked — test setup invalid")
	}
}

// TestClusterSimulator_ClockMonotonicity_ClusterDominatesInstances verifies BC-7:
// GIVEN a cluster simulation with non-trivial workload
// WHEN the simulation completes
// THEN cluster.Clock() >= every instance's Clock().
func TestClusterSimulator_ClockMonotonicity_ClusterDominatesInstances(t *testing.T) {
	config := newTestDeploymentConfig(4)
	workload := newTestWorkload(100)

	cs := NewClusterSimulator(config, workload, "")
	cs.Run()

	for i, inst := range cs.Instances() {
		if cs.Clock() < inst.Clock() {
			t.Errorf("clock monotonicity violated: cluster clock (%d) < instance %d clock (%d)",
				cs.Clock(), i, inst.Clock())
		}
	}
}

// TestClusterSimulator_Determinism_ByteIdenticalAggregation verifies BC-9:
// GIVEN two cluster runs with identical config and seed
// WHEN both aggregate metrics and serialize to JSON
// THEN the serialized JSON is byte-identical.
func TestClusterSimulator_Determinism_ByteIdenticalAggregation(t *testing.T) {
	run := func() *sim.Metrics {
		config := newTestDeploymentConfig(3)
		workload := newTestWorkload(50)
		cs := NewClusterSimulator(config, workload, "")
		cs.Run()
		return cs.AggregatedMetrics()
	}

	m1 := run()
	m2 := run()

	// Compare integer fields
	if m1.CompletedRequests != m2.CompletedRequests {
		t.Errorf("determinism: CompletedRequests %d vs %d", m1.CompletedRequests, m2.CompletedRequests)
	}
	if m1.TotalInputTokens != m2.TotalInputTokens {
		t.Errorf("determinism: TotalInputTokens %d vs %d", m1.TotalInputTokens, m2.TotalInputTokens)
	}
	if m1.TotalOutputTokens != m2.TotalOutputTokens {
		t.Errorf("determinism: TotalOutputTokens %d vs %d", m1.TotalOutputTokens, m2.TotalOutputTokens)
	}
	if m1.SimEndedTime != m2.SimEndedTime {
		t.Errorf("determinism: SimEndedTime %d vs %d", m1.SimEndedTime, m2.SimEndedTime)
	}
	if m1.TTFTSum != m2.TTFTSum {
		t.Errorf("determinism: TTFTSum %d vs %d", m1.TTFTSum, m2.TTFTSum)
	}
	if m1.ITLSum != m2.ITLSum {
		t.Errorf("determinism: ITLSum %d vs %d", m1.ITLSum, m2.ITLSum)
	}

	// Compare per-request maps via JSON serialization (catches map ordering issues)
	j1, _ := json.Marshal(sortedRequestMetrics(m1.Requests))
	j2, _ := json.Marshal(sortedRequestMetrics(m2.Requests))
	if !bytes.Equal(j1, j2) {
		t.Error("determinism: per-request metrics JSON differs between runs")
	}
}

// sortedRequestMetrics returns RequestMetrics in sorted order for deterministic comparison.
func sortedRequestMetrics(m map[string]sim.RequestMetrics) []sim.RequestMetrics {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	result := make([]sim.RequestMetrics, len(keys))
	for i, k := range keys {
		result[i] = m[k]
	}
	return result
}
```

**Step 2: Run tests**

Run: `go test ./sim/cluster/... -run "TestClusterSimulator_(RequestConservation_Sum|Causality_PerInstance|ClockMonotonicity_ClusterDominates|Determinism_ByteIdentical)" -v`
Expected: PASS

**Step 3: No implementation needed**

**Step 4: Run lint**

Run: `golangci-lint run ./sim/cluster/...`

**Step 5: Commit**

```bash
git add sim/cluster/cluster_test.go
git commit -m "test(cluster): add cluster-level invariant tests (BC-3, BC-5, BC-7, BC-9)

- Request conservation: sum of instances == aggregated == injected
- Causality: TTFT <= E2E and ITL >= 0 per instance
- Clock monotonicity: cluster clock >= all instance clocks
- Determinism: byte-identical aggregated metrics JSON
- Issue #199

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Final Verification and Documentation Update

**Contracts Implemented:** All (verification)

**Files:**
- Run full test suite
- Update macro plan status if needed

**Step 1: Run full test suite**

Run: `go test ./... -v`
Expected: All tests pass, including new invariant tests

**Step 2: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 3: Verify test count**

Count new tests added:
- `sim/simulator_test.go`: 6 new tests (conservation infinite, conservation finite, causality full chain, clock monotonicity, determinism JSON, KV conservation)
- `sim/cluster/cluster_test.go`: 4 new tests (conservation sum, causality per-instance, clock monotonicity, determinism aggregation)

Total: ~9 new invariant test functions implementing 10 behavioral contracts.

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | 1 | Invariant | TestSimulator_RequestConservation_InfiniteHorizon_AllRequestsComplete |
| BC-2 | 1 | Invariant | TestSimulator_RequestConservation_FiniteHorizon_ThreeTermEquation |
| BC-3 | 6 | Invariant | TestClusterSimulator_RequestConservation_SumAcrossInstances |
| BC-4, NC-1 | 2 | Invariant | TestSimulator_Causality_FullChain_ArrivalToCompletion |
| BC-5 | 6 | Invariant | TestClusterSimulator_Causality_PerInstance |
| BC-6, NC-2 | 3 | Invariant | TestSimulator_ClockMonotonicity_NeverDecreases |
| BC-7 | 6 | Invariant | TestClusterSimulator_ClockMonotonicity_ClusterDominatesInstances |
| BC-8 | 4 | Invariant | TestSimulator_Determinism_ByteIdenticalJSON |
| BC-9 | 6 | Invariant | TestClusterSimulator_Determinism_ByteIdenticalAggregation |
| BC-10 | 5 | Invariant | TestSimulator_KVBlockConservation_PostSimulation_ZeroLeak |

**Golden dataset update:** Not needed — no production code changes.

**Shared test infrastructure:** Uses `newTestSimConfig()` from `sim/simulator_test.go` and `newTestDeploymentConfig`/`newTestWorkload` from `sim/cluster/cluster_test.go`.

---

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| SaveResults timestamp non-determinism breaks BC-8 | High | Low | Normalize timestamps before comparison | Task 4 |
| Finite horizon test doesn't actually leave requests pending | Low | Medium | Manual injection with late arrivals (after horizon) guarantees pending requests; fatal assertion if all complete | Task 1 |
| Clock monotonicity manual loop diverges from Run() behavior | Low | Medium | Loop mirrors Run() exactly: ProcessNextEvent + horizon check | Task 3 |
| Cluster determinism affected by goroutine scheduling | None | N/A | Cluster is single-threaded (shared-clock event loop) | Task 6 |

---

### E) Review Guide (expanded)

1. **THE TRICKY PART:** Task 4 (byte-identical determinism) — `SaveResults` embeds `time.Now()` for `sim_end_timestamp` and `simulation_duration_s`. These wall-clock values will always differ between runs. The test normalizes these fields before comparison. Verify the normalization is complete.

2. **WHAT TO SCRUTINIZE:** Task 1 (finite horizon conservation) — the horizon value (1,000,000 ticks) must be short enough that some requests remain pending but long enough that at least some complete. The test logs a warning if all requests complete (making the three-term test vacuous). Verify the warning is actionable.

3. **WHAT'S SAFE TO SKIM:** Tasks 5, 6 — straightforward post-simulation assertions with minimal logic.

4. **KNOWN DEBT:** The existing `TestSimulator_DeterministicWorkload` and `TestClusterSimulator_MultiInstance_Determinism` overlap with BC-8/BC-9 but use weaker assertions (field-by-field integers vs byte-identical JSON). They are not removed — they provide faster feedback and test a slightly different property.

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — pure test addition
- [x] No feature creep — tests only, no production code changes
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used from existing packages
- [x] CLAUDE.md — no updates needed (no new files/packages/flags)
- [x] Deviation log reviewed — golden regeneration deferred (already current), clock instrumentation via manual loop
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (independent tasks, but logical progression)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration: not needed
- [x] Construction site audit: N/A (no struct modifications)
- [x] No new CLI flags
- [x] No error paths added to production code
- [x] No map iteration in production code
- [x] Library code untouched
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes
- [x] No division operations added
- [x] Grepped for "Phase 4" / "PR Phase 4" references — none found requiring update
- [x] Macro plan status update: will update in Task 7 if needed

---

## Appendix: Additional Implementation Notes

### Import additions needed

**`sim/simulator_test.go`** needs:
```go
import (
    "bytes"
    "encoding/json"
    "os"
    "time"
    // existing imports remain
)
```

**`sim/cluster/cluster_test.go`** needs:
```go
import (
    "bytes"
    "encoding/json"
    "sort"
    // existing imports remain
)
```

### Key behavioral notes

- **Manual event loop (Task 3):** The loop `for sim.HasPendingEvents() { sim.ProcessNextEvent(); if sim.Clock > sim.Horizon { break } }` followed by `sim.Finalize()` is functionally identical to `sim.Run()`. This is the correct replication — do NOT add extra logic.

- **SaveResults wall-clock contamination (Task 4):** `SaveResults` writes `sim_end_timestamp` as `time.Now().Format(...)` and `simulation_duration_s` as `time.Since(startTime).Seconds()`. Both are wall-clock dependent. The test normalizes by unmarshaling to `MetricsOutput`, zeroing these two fields, and re-marshaling.

- **Finite horizon behavior (Task 1):** When `sim.Clock > sim.Horizon`, `Run()` breaks and calls `Finalize()`. Requests that arrived but haven't completed remain in WaitQ or RunningBatch. `Metrics.Requests` includes ALL injected requests (registered at injection time via `InjectArrival`), so `len(Metrics.Requests)` is the authoritative injection count. **Critical:** requests with `ArrivalTime > Horizon` have their ArrivalEvent never popped from the event queue — they're in `Metrics.Requests` but NOT in WaitQ/RunningBatch/completed. This would break the three-term equation. The test therefore ensures ALL injected requests have `ArrivalTime < Horizon` — late requests are large (long processing time) rather than late-arriving.

- **Cluster determinism (Task 6):** The `newTestWorkload` and `newTestDeploymentConfig` helpers use the same seed and config across both runs. The `ClusterSimulator` is single-threaded (shared-clock event loop), so there are no goroutine-ordering non-determinism concerns.
