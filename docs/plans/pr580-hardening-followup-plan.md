# PR #580: Hardening Follow-Up — PR #567 Review Findings

- **Goal:** Address validation gaps, missing observability counters, and test coverage gaps identified during PR #579's convergence review.
- **The problem today:** (1) `NewSimulator` accepts negative `MaxModelLen` from library callers bypassing the canonical constructor, silently disabling enforcement. (2) Malformed `rope_scaling` in HuggingFace configs silently falls through with no diagnostic. (3) Force-completed (length-capped) requests are indistinguishable from normal completions in structured output. (4) The BC-5 runtime cap test calls `processCompletions` directly and wouldn't catch a full event-loop regression. (5) INV-9 structural test doesn't scan cluster-level code that handles `*Request` in the routing pipeline.
- **What this PR adds:**
  1. Negative `MaxModelLen` validation in `NewSimulator` (defense-in-depth behind the canonical constructor's panic)
  2. `logrus.Warnf` diagnostics when `rope_scaling` is present but malformed in HF configs
  3. `LengthCappedRequests` metric counter across the 5-file pattern (metrics → output → cluster aggregation → CLI)
  4. End-to-end `sim.Run()` test for the BC-5 runtime length cap path
  5. INV-9 structural test extended to scan `sim/cluster/` control-plane files
  6. Negative `MaxOutputLen` validation in `EnqueueRequest` (R3 gap)
  7. `gemma3` model_type exclusion for rope_scaling (matches vLLM's exact `model_type == "gemma3"` check)
  8. Fixed `kvFeasibleMax` comment accuracy
- **Why this matters:** These are hardening items that close validation gaps, improve observability, and expand test coverage — reducing the risk of silent regressions in the MaxModelLen enforcement path.
- **Architecture:** Changes span `sim/simulator.go` (validation + counter), `sim/metrics.go` (field), `sim/metrics_utils.go` (JSON output), `sim/cluster/metrics.go` (aggregation), `sim/cluster/cluster.go` (aggregation), `cmd/root.go` (CLI output + rope_scaling), `sim/simulator_test.go` (tests).
- **Source:** [Issue #580](https://github.com/inference-sim/inference-sim/issues/580)
- **Closes:** `Fixes #580`
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** Simulator constructor validation, enqueue guard, metrics pipeline, rope_scaling config parser, INV-9 structural test.
2. **Adjacent blocks:** `sim/config.go` (canonical constructors already validate), `sim/cluster/cluster.go` (aggregation), `cmd/root.go` (CLI output and HF config parsing).
3. **Invariants touched:** INV-1 (conservation — LengthCappedRequests must be accounted for), INV-9 (oracle knowledge boundary — extended to cluster/).
4. **Construction Site Audit:**
   - `Metrics` struct: `NewMetrics()` is the only constructor. Adding `LengthCappedRequests int` field — update `NewMetrics()` (zero-value is correct).
   - `MetricsOutput` struct: used only via inline literal in `SaveResults()`. Adding `LengthCappedRequests int` field.
   - `RawMetrics` struct: used only via inline literal in `CollectRawMetrics()`. Adding `LengthCappedRequests int` field.
   - No new structs introduced.

---

## Part 1: Design Validation

### A) Executive Summary

This PR addresses 8 of 15 items from issue #580 (5 must-fix, 3 should-fix); 7 items deferred with justification. The changes are all hardening: validation gaps closed, observability improved, test coverage expanded. No new features, no behavioral changes to existing passing paths. The `LengthCappedRequests` counter follows the exact 5-file pattern established by `DroppedUnservable`. The rope_scaling improvements add warnings for malformed configs and extend the exclusion blacklist. The INV-9 test extension is a one-line addition.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Negative MaxModelLen rejection in NewSimulator
- GIVEN a SimConfig with MaxModelLen < 0
- WHEN NewSimulator is called
- THEN it returns a non-nil error containing "MaxModelLen"
- MECHANISM: Explicit check before the existing MaxModelLen > 0 block
```

```
BC-2: LengthCappedRequests counter incremented on force-completion
- GIVEN a request whose ProgressIndex >= MaxModelLen during processCompletions
- WHEN the runtime length cap fires (BC-5 path)
- THEN Metrics.LengthCappedRequests is incremented by 1
```

```
BC-3: LengthCappedRequests appears in JSON output
- GIVEN a simulation with LengthCappedRequests > 0
- WHEN SaveResults writes JSON
- THEN the output contains "length_capped_requests" with the correct count
```

```
BC-4: LengthCappedRequests aggregated across instances in cluster mode
- GIVEN a multi-instance simulation where some instances have length-capped requests
- WHEN cluster metrics are aggregated
- THEN the aggregated LengthCappedRequests equals the sum across all instances
```

```
BC-5: End-to-end runtime cap via sim.Run()
- GIVEN a simulator with MaxModelLen=100, a request with input=50, OutputTokens=200, MaxOutputLen=0
- WHEN sim.Run() completes
- THEN the request is force-completed, conservation holds, KV blocks are released
```

```
BC-6: INV-9 structural test covers cluster control-plane files
- GIVEN the INV-9 test scanning for OutputTokens references
- WHEN it runs
- THEN it scans cluster.go, cluster_event.go, snapshot.go, and counterfactual.go in addition to existing sim/ files
```

**Error handling contracts:**

```
BC-7: Negative MaxOutputLen warning and drop
- GIVEN a request with MaxOutputLen < 0
- WHEN EnqueueRequest is called
- THEN a warning is logged, DroppedUnservable is incremented, and the request is removed from Requests map
```

```
BC-8: Rope_scaling malformed config warnings
- GIVEN an HF config where rope_scaling exists but is not a JSON object, or factor is not a float64
- WHEN the rope_scaling block is processed
- THEN a logrus.Warnf is emitted (no crash, no silent fallthrough)
```

```
BC-9: gemma3 excluded from rope_scaling factor application via model_type check
- GIVEN an HF config where model_type == "gemma3" (exact match)
- WHEN the rope_scaling block is processed
- THEN the entire rope_scaling factor application is skipped (matching vLLM's model_type-level exclusion)
- MECHANISM: Check hfConfig.Raw["model_type"] == "gemma3" before entering the rope_scaling factor block
```

### C) Component Interaction

```
cmd/root.go (CLI layer)
  ├── rope_scaling parsing: adds warnings for malformed configs (BC-8)
  ├── rope_scaling blacklist: extends to gemma3, mrope (BC-9)
  ├── kvFeasibleMax comment fix
  └── Anomaly counters output: prints LengthCappedRequests (BC-3)

sim/simulator.go (core engine)
  ├── NewSimulator: negative MaxModelLen validation (BC-1)
  ├── EnqueueRequest: negative MaxOutputLen validation (BC-7)
  └── processCompletions: increments LengthCappedRequests (BC-2)

sim/metrics.go (metrics struct)
  └── LengthCappedRequests field + NewMetrics init (BC-2)

sim/metrics_utils.go (JSON output)
  └── MetricsOutput.LengthCappedRequests field (BC-3)

sim/cluster/cluster.go (aggregation)
  └── aggregateMetrics: sums LengthCappedRequests (BC-4)

sim/cluster/metrics.go (raw metrics)
  └── RawMetrics.LengthCappedRequests + CollectRawMetrics (BC-4)

sim/simulator_test.go (tests)
  ├── TestNewSimulator_NegativeMaxModelLen (BC-1)
  ├── TestProcessCompletions_RuntimeLengthCap_Counter (BC-2)
  ├── TestSaveResults_LengthCappedRequests_InJSON (BC-3)
  ├── TestSimulator_RuntimeLengthCap_E2E (BC-5)
  ├── TestINV9 extension (BC-6)
  └── TestEnqueueRequest_NegativeMaxOutputLen (BC-7)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| INV-9 paths `../cluster/` (issue text) | Uses `cluster/` prefix | CORRECTION: Go test CWD is `sim/`; `cluster/` resolves to `sim/cluster/`. Issue's `../cluster/` would resolve to non-existent directory |
| INV-9 scan adds `cluster_event.go` (not in issue) | Added | ADDITION: `cluster_event.go` defines routing event types that handle `*Request`; should be scanned for INV-9 |
| Add `mrope` to rope_scaling blacklist (should-fix) | Deferred | vLLM normalizes mrope→"default" and APPLIES the factor; blacklisting would produce opposite behavior. Needs deeper investigation |
| Add rope_scaling unit tests (should-fix) | Deferred | Requires extracting rope_scaling into a pure function — scope creep for a hardening PR |
| Change MaxModelLen type int → int64 (nice-to-have) | Deferred | Low-priority cleanup, would touch many sites |
| Add glossary entries (nice-to-have) | Deferred | Documentation-only, no code impact |
| Refine rope_scaling docs (nice-to-have) | Deferred | Documentation-only |
| Add cluster-mode MaxModelLen drop test (nice-to-have) | Deferred | Cluster-mode tests require more setup; the single-instance E2E test covers the core path |
| Add chunked prefill + MaxModelLen test (nice-to-have) | Deferred | Interaction testing — lower priority |

### E) Review Guide

**Scrutinize:** The `LengthCappedRequests` 5-file pattern — verify it mirrors `DroppedUnservable` exactly. The negative `MaxOutputLen` validation in `EnqueueRequest` — verify it doesn't break the existing `MaxOutputLen=0` path. The INV-9 test extension — verify the relative paths work from the test's CWD.

**Safe to skim:** Comment fix, blacklist extension (trivial string additions), BC-1 (one-line validation).

**Known debt:** rope_scaling is inline in `cmd/root.go` — extracting to a pure function would enable unit testing. Deferred per deviation log.

**Note:** INV-9 test paths use `cluster/` prefix (not `../cluster/`) because Go test CWD is `sim/` and `sim/cluster/` is a subdirectory.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/simulator.go` — Add negative MaxModelLen validation in NewSimulator, negative MaxOutputLen validation in EnqueueRequest, increment LengthCappedRequests in processCompletions
- `sim/metrics.go` — Add LengthCappedRequests field, update SaveResults output
- `sim/metrics_utils.go` — Add LengthCappedRequests to MetricsOutput
- `sim/cluster/cluster.go` — Add LengthCappedRequests to aggregateMetrics
- `sim/cluster/metrics.go` — Add LengthCappedRequests to RawMetrics and CollectRawMetrics
- `cmd/root.go` — Add rope_scaling warnings, extend blacklist, fix comment, add LengthCappedRequests to anomaly output
- `sim/simulator_test.go` — All new tests

**No new files. No dead code.**

### G) Task Breakdown

#### Task 1: Negative MaxModelLen validation in NewSimulator (BC-1)

**Contracts:** BC-1

**Test (sim/simulator_test.go):**
```go
// BC-1: Negative MaxModelLen rejected by NewSimulator
// Uses struct literal bypass (not canonical constructor) to simulate library caller
func TestNewSimulator_NegativeMaxModelLen_Error(t *testing.T) {
	cfg := newTestSimConfig()
	cfg.MaxModelLen = -5 // bypass canonical constructor's panic to test NewSimulator validation
	kvStore := MustNewKVCacheState(cfg.TotalKVBlocks, cfg.BlockSizeTokens)
	latencyModel, err := MustNewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		t.Fatalf("MustNewLatencyModel: %v", err)
	}
	_, err = NewSimulator(cfg, kvStore, latencyModel)
	if err == nil {
		t.Fatal("expected error for negative MaxModelLen")
	}
	if !strings.Contains(err.Error(), "MaxModelLen") {
		t.Errorf("error %q should mention MaxModelLen", err.Error())
	}
}
```

**Implementation (sim/simulator.go):** Add before line 104 (`if cfg.MaxModelLen > 0`):
```go
if cfg.MaxModelLen < 0 {
	return nil, fmt.Errorf("NewSimulator: MaxModelLen must be >= 0, got %d", cfg.MaxModelLen)
}
```

**Commands:**
```bash
cd .worktrees/pr580-hardening-followup
go test ./sim/... -run TestNewSimulator_NegativeMaxModelLen -count=1
golangci-lint run ./sim/...
```

---

#### Task 2: LengthCappedRequests metric counter — 5-file pattern (BC-2, BC-3, BC-4)

**Contracts:** BC-2, BC-3, BC-4

**Step 1: Add field to Metrics (sim/metrics.go):**
Add after `DroppedUnservable int`:
```go
LengthCappedRequests int // Requests force-completed by runtime MaxModelLen cap (BC-5 defense-in-depth)
```

Update `SaveResults` to include in `MetricsOutput` construction:
```go
LengthCappedRequests: m.LengthCappedRequests,
```

Update `InjectedRequests` computation — length-capped requests ARE completed (they go through `recordRequestCompletion`), so no change to INV-1 formula.

**Step 2: Add field to MetricsOutput (sim/metrics_utils.go):**
Add after `DroppedUnservable`:
```go
LengthCappedRequests    int              `json:"length_capped_requests"`
```

**Step 3: Increment counter in processCompletions (sim/simulator.go):**
In the BC-5 runtime cap block (after `logrus.Warnf`):
```go
sim.Metrics.LengthCappedRequests++
```

**Step 4: Aggregate in cluster (sim/cluster/cluster.go):**
Add after `merged.DroppedUnservable += m.DroppedUnservable`:
```go
merged.LengthCappedRequests += m.LengthCappedRequests
```

**Step 5: Add to RawMetrics and CollectRawMetrics (sim/cluster/metrics.go):**
Add field to `RawMetrics` after `DroppedUnservable`:
```go
LengthCappedRequests int
```
Add to `CollectRawMetrics` after `DroppedUnservable: aggregated.DroppedUnservable,`:
```go
LengthCappedRequests: aggregated.LengthCappedRequests,
```

**Step 6: Add CLI output (cmd/root.go):**
Update the anomaly counter condition and print:
```go
// Add LengthCappedRequests to the anomaly counter condition and output
if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 {
	// ... existing prints ...
	fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
}
```

**Tests (sim/simulator_test.go and sim/metrics_test.go):**

Test BC-2 (counter increment — extend existing `TestProcessCompletions_RuntimeLengthCap`):
Add to the existing test after the `req.State != StateCompleted` assertion:
```go
if sim.Metrics.LengthCappedRequests != 1 {
	t.Errorf("LengthCappedRequests = %d, want 1", sim.Metrics.LengthCappedRequests)
}
```

Test BC-3 (JSON output — new test following `TestSaveResults_DroppedUnservable_InJSON` pattern):
```go
func TestSaveResults_LengthCappedRequests_InJSON(t *testing.T) {
	m := NewMetrics()
	m.LengthCappedRequests = 3
	m.CompletedRequests = 3
	m.SimEndedTime = 1_000_000

	tmpFile := filepath.Join(t.TempDir(), "test_output.json")
	if err := m.SaveResults("test", 10_000_000, 100, tmpFile); err != nil {
		t.Fatalf("SaveResults returned error: %v", err)
	}

	data, err := os.ReadFile(tmpFile)
	if err != nil {
		t.Fatalf("reading output: %v", err)
	}

	var output MetricsOutput
	if err := json.Unmarshal(data, &output); err != nil {
		t.Fatalf("parsing JSON: %v", err)
	}

	if output.LengthCappedRequests != 3 {
		t.Errorf("LengthCappedRequests in JSON = %d, want 3", output.LengthCappedRequests)
	}
}
```

**Commands:**
```bash
go test ./sim/... -run "TestProcessCompletions_RuntimeLengthCap|TestSaveResults_LengthCapped" -count=1
go test ./sim/cluster/... -count=1
golangci-lint run ./sim/... ./sim/cluster/... ./cmd/...
```

---

#### Task 3: End-to-end runtime cap test via sim.Run() (BC-5)

**Contracts:** BC-5

**Test (sim/simulator_test.go):**
```go
// BC-5: End-to-end runtime length cap via sim.Run()
func TestSimulator_RuntimeLengthCap_E2E(t *testing.T) {
	// GIVEN a simulator with MaxModelLen=100
	cfg := SimConfig{
		Horizon:             10_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "", 100),
	}
	sim := mustNewSimulator(t, cfg)

	// AND a request that bypasses enqueue guard: input=50, MaxOutputLen=0, OutputTokens=200
	// MaxOutputLen=0 means input-only check (50 < 100), so enqueue succeeds.
	// But actual OutputTokens=200 means ProgressIndex will exceed MaxModelLen=100.
	req := &Request{
		ID:           "will_be_capped",
		InputTokens:  GenerateRandomTokenIDs(sim.WorkloadRNG(), 50),
		OutputTokens: GenerateRandomTokenIDs(sim.WorkloadRNG(), 200),
		ArrivalTime:  0,
		State:        StateQueued,
	}
	sim.InjectArrival(req)

	// WHEN sim.Run() completes
	sim.Run()

	// THEN the request completes (force-completed at MaxModelLen boundary)
	if sim.Metrics.CompletedRequests != 1 {
		t.Errorf("CompletedRequests = %d, want 1", sim.Metrics.CompletedRequests)
	}
	// AND LengthCappedRequests == 1
	if sim.Metrics.LengthCappedRequests != 1 {
		t.Errorf("LengthCappedRequests = %d, want 1", sim.Metrics.LengthCappedRequests)
	}
	// AND conservation holds (INV-1)
	total := sim.Metrics.CompletedRequests + sim.Metrics.StillQueued + sim.Metrics.StillRunning + sim.Metrics.DroppedUnservable
	if total != 1 {
		t.Errorf("INV-1: completed(%d)+queued(%d)+running(%d)+dropped(%d) = %d, want 1",
			sim.Metrics.CompletedRequests, sim.Metrics.StillQueued, sim.Metrics.StillRunning, sim.Metrics.DroppedUnservable, total)
	}
	// AND output was truncated: generated fewer tokens than the full 200 output
	if sim.Metrics.TotalOutputTokens >= 200 {
		t.Errorf("TotalOutputTokens = %d, want < 200 (force-completion should truncate output)", sim.Metrics.TotalOutputTokens)
	}
	if sim.Metrics.TotalOutputTokens == 0 {
		t.Error("TotalOutputTokens = 0, want > 0 (some decode work should have happened)")
	}
	// Regression anchor: exact count for this configuration (MaxModelLen=100 - input=50 = 50 decode steps)
	if sim.Metrics.TotalOutputTokens != 50 {
		t.Errorf("TotalOutputTokens = %d, want 50 (regression anchor)", sim.Metrics.TotalOutputTokens)
	}
	// AND KV blocks are released
	if sim.KVCache.UsedBlocks() != 0 {
		t.Errorf("UsedBlocks = %d, want 0 (KV blocks should be released after force-completion)", sim.KVCache.UsedBlocks())
	}
}
```

**Commands:**
```bash
go test ./sim/... -run TestSimulator_RuntimeLengthCap_E2E -count=1
```

---

#### Task 4: INV-9 structural test extension (BC-6)

**Contracts:** BC-6

**Implementation (sim/simulator_test.go):** Add cluster files to `controlPlaneFiles` list and refine the scan pattern to exclude metric-aggregate fields like `TotalOutputTokens`:
```go
controlPlaneFiles := []string{
	"admission.go",
	"routing.go",
	"routing_scorers.go",
	"routing_prefix_scorer.go",
	"scheduler.go",
	"priority.go",
	"cluster/cluster.go",
	"cluster/cluster_event.go",
	"cluster/snapshot.go",
	"cluster/counterfactual.go",
}
```

For cluster files that contain metrics aggregation (`TotalOutputTokens`), the scan must use line-level exclusion to avoid false positives:
```go
for _, filename := range controlPlaneFiles {
	data, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("failed to read %s: %v", filename, err)
	}
	for lineNum, line := range strings.Split(string(data), "\n") {
		// Remove known-safe metric names, then check for remaining OutputTokens references
		cleaned := strings.ReplaceAll(line, "TotalOutputTokens", "")
		if strings.Contains(cleaned, "OutputTokens") {
			t.Errorf("INV-9 violation: %s line %d references OutputTokens — control-plane code must not access oracle output length", filename, lineNum+1)
		}
	}
}
```
This replaces the existing simple `strings.Contains(content, "OutputTokens")` check for the full-file scan. The existing `EnqueueRequest`-specific function-body extraction for `simulator.go` remains unchanged.

**Commands:**
```bash
go test ./sim/... -run TestINV9_OracleKnowledgeBoundary -count=1
```

---

#### Task 5: Negative MaxOutputLen validation in EnqueueRequest (BC-7)

**Contracts:** BC-7

**Test (sim/simulator_test.go):**
```go
// BC-7: Negative MaxOutputLen → warning + dropped
func TestEnqueueRequest_NegativeMaxOutputLen_Dropped(t *testing.T) {
	cfg := SimConfig{
		Horizon:             1_000_000,
		Seed:                42,
		KVCacheConfig:       NewKVCacheConfig(1000, 16, 0, 0, 0, 0),
		BatchConfig:         NewBatchConfig(256, 2048, 0),
		LatencyCoeffs:       NewLatencyCoeffs([]float64{6910, 17.67, 2.84}, []float64{0, 0, 0}),
		ModelHardwareConfig: NewModelHardwareConfig(ModelConfig{}, HardwareCalib{}, "test", "H100", 1, "", 512),
	}
	sim := mustNewSimulator(t, cfg)

	req := &Request{
		ID:           "neg_budget",
		InputTokens:  make([]int, 100),
		OutputTokens: make([]int, 50),
		MaxOutputLen: -1,
		State:        StateQueued,
	}
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, 0)
	sim.EnqueueRequest(req)

	if sim.WaitQ.Len() != 0 {
		t.Errorf("WaitQ.Len() = %d, want 0 (negative MaxOutputLen should be dropped)", sim.WaitQ.Len())
	}
	if sim.Metrics.DroppedUnservable != 1 {
		t.Errorf("DroppedUnservable = %d, want 1", sim.Metrics.DroppedUnservable)
	}
}
```

**Implementation (sim/simulator.go):** Add before the MaxModelLen guard in `EnqueueRequest`:
```go
// Guard 0: Negative MaxOutputLen check (R3)
if r.MaxOutputLen < 0 {
	logrus.Warnf("dropping request %s: MaxOutputLen %d is negative (R3 validation gap)",
		r.ID, r.MaxOutputLen)
	sim.Metrics.DroppedUnservable++
	delete(sim.Metrics.Requests, r.ID)
	return
}
```

**Commands:**
```bash
go test ./sim/... -run TestEnqueueRequest_NegativeMaxOutputLen -count=1
golangci-lint run ./sim/...
```

---

#### Task 6: Rope_scaling improvements (BC-8, BC-9) + comment fix

**Contracts:** BC-8, BC-9

**Implementation (cmd/root.go):**

1. Add warnings for malformed rope_scaling (inside the `if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok` block):
```go
if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok {
	if ropeMap, ok := ropeScaling.(map[string]any); ok {
		// ... existing logic ...
	} else {
		logrus.Warnf("--latency-model: rope_scaling present but not a JSON object (type %T); ignoring", ropeScaling)
	}
}
```

2. Inside the factor extraction, add warning when factor is not a float64:
```go
if factor, ok := ropeMap["factor"].(float64); ok && factor > 1.0 {
	// ... existing logic ...
} else if _, hasKey := ropeMap["factor"]; hasKey {
	logrus.Warnf("--latency-model: rope_scaling.factor present but not a valid float64; ignoring")
}
```

3. Add gemma3 model_type exclusion (before the rope_scaling block, matching vLLM):
```go
// vLLM skips rope_scaling entirely for gemma3 models (config.py:1973):
// "gemma3's max_model_len (128K) is already scaled by RoPE scaling"
if modelType, ok := hfConfig.Raw["model_type"].(string); ok && modelType == "gemma3" {
	// Skip rope_scaling entirely — gemma3's max_position_embeddings is pre-scaled
} else if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok {
```
NOTE: mrope is NOT blacklisted — vLLM normalizes mrope→"default" and applies the factor. Blacklisting mrope would produce the opposite behavior. The issue's mrope item is deferred pending deeper investigation.

4. Fix kvFeasibleMax comment:
```go
kvFeasibleMax := int(totalKVBlocks * blockSizeTokens) // safe: totalKVBlocks * blockSizeTokens < maxModelLen (blocksNeeded > totalKVBlocks), fits in int
```

**Commands:**
```bash
go build ./...
golangci-lint run ./cmd/...
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | TestNewSimulator_NegativeMaxModelLen_Error |
| BC-2 | Task 2 | Unit | TestProcessCompletions_RuntimeLengthCap_Counter |
| BC-3 | Task 2 | Unit | TestSaveResults_LengthCappedRequests_InJSON |
| BC-5 | Task 3 | Integration | TestSimulator_RuntimeLengthCap_E2E |
| BC-6 | Task 4 | Structural | TestINV9_OracleKnowledgeBoundary (extended) |
| BC-7 | Task 5 | Unit | TestEnqueueRequest_NegativeMaxOutputLen_Dropped |
| BC-8, BC-9 | Task 6 | Manual | Verified by build + lint (rope_scaling is in cmd/) |

**Invariants verified:** INV-1 (conservation in BC-5 test), INV-9 (extended structural test).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| LengthCappedRequests breaks INV-1 conservation formula | Low | High | Length-capped requests go through `recordRequestCompletion` → counted in `CompletedRequests`. No formula change needed. Verified in BC-5 E2E test. | Task 3 |
| Negative MaxOutputLen validation breaks MaxOutputLen=0 path | Low | High | Guard is `< 0`, not `<= 0`. MaxOutputLen=0 means "no budget" (existing behavior). Tested by existing tests. | Task 5 |
| INV-9 cluster file paths wrong from test CWD | Low | Medium | Test runs in `sim/` directory. `cluster/cluster.go` resolves to `sim/cluster/cluster.go`. Line-level scan excludes `TotalOutputTokens` false positives. | Task 4 |
| rope_scaling warning triggers on valid configs | Low | Low | Warning only fires when type assertion fails (not a map, factor not float64). Valid configs always have a JSON object with float factor. | Task 6 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions
- [x] No feature creep beyond PR scope (nice-to-have items deferred)
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers used (mustNewSimulator, newTestSimConfig)
- [x] CLAUDE.md updated if needed — no new files/packages, file org unchanged
- [x] No stale references
- [x] Documentation DRY — no canonical sources modified
- [x] Deviation log reviewed — 9 entries (7 deferred + 2 corrections/additions) justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 2 before Task 3 — counter must exist)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration not needed (no output changes for existing paths)
- [x] Construction site audit completed (Metrics, MetricsOutput, RawMetrics)

**Antipattern rules:** R1 ✓ (no silent drops), R2 ✓ (no map iteration for output), R3 ✓ (validation added), R4 ✓ (all construction sites updated), R5 ✓ N/A, R6 ✓ (warnings in sim/, Fatalf only in cmd/), R7 ✓ (BC-5 E2E test is invariant-based), R8 ✓ N/A, R9 ✓ N/A, R10 ✓ N/A, R11 ✓ N/A, R12 ✓ N/A, R13 ✓ N/A, R14 ✓ N/A, R15 ✓ N/A, R16 ✓ N/A, R17 ✓ N/A, R18 ✓ N/A, R19 ✓ N/A, R20 ✓ N/A, R21 ✓ N/A, R22 ✓ N/A, R23 ✓ N/A.

---

## Appendix: File-Level Implementation Details

### File: `sim/simulator.go`

**Purpose:** Add negative MaxModelLen validation in NewSimulator, negative MaxOutputLen validation in EnqueueRequest, increment LengthCappedRequests counter in processCompletions.

**Changes:**

1. In `NewSimulator`, after `LongPrefillTokenThreshold` validation (line 103), before `if cfg.MaxModelLen > 0` (line 104):
```go
if cfg.MaxModelLen < 0 {
	return nil, fmt.Errorf("NewSimulator: MaxModelLen must be >= 0, got %d", cfg.MaxModelLen)
}
```

2. In `EnqueueRequest`, before the `// Guard 1: MaxModelLen check` block (line 260):
```go
// Guard 0: Negative MaxOutputLen check (R3)
if r.MaxOutputLen < 0 {
	logrus.Warnf("dropping request %s: MaxOutputLen %d is negative (R3 validation gap)",
		r.ID, r.MaxOutputLen)
	sim.Metrics.DroppedUnservable++
	delete(sim.Metrics.Requests, r.ID)
	return
}
```

3. In `processCompletions`, in the BC-5 runtime cap block (after the `logrus.Warnf`, before `req.State = StateCompleted`):
```go
sim.Metrics.LengthCappedRequests++
```

### File: `sim/metrics.go`

**Purpose:** Add LengthCappedRequests counter field and include in SaveResults output.

**Changes:**

1. Add field after `DroppedUnservable`:
```go
LengthCappedRequests int // Requests force-completed by runtime MaxModelLen cap (BC-5 defense-in-depth)
```

2. In `SaveResults`, add to MetricsOutput construction (after `DroppedUnservable`):
```go
LengthCappedRequests: m.LengthCappedRequests,
```

### File: `sim/metrics_utils.go`

**Purpose:** Add LengthCappedRequests to JSON output struct.

**Changes:** Add after `DroppedUnservable` in MetricsOutput:
```go
LengthCappedRequests    int              `json:"length_capped_requests"`
```

### File: `sim/cluster/cluster.go`

**Purpose:** Aggregate LengthCappedRequests across instances.

**Changes:** In `aggregateMetrics`, after `merged.DroppedUnservable += m.DroppedUnservable`:
```go
merged.LengthCappedRequests += m.LengthCappedRequests
```

### File: `sim/cluster/metrics.go`

**Purpose:** Add LengthCappedRequests to RawMetrics and collection.

**Changes:**

1. In `RawMetrics`, after `DroppedUnservable`:
```go
LengthCappedRequests int
```

2. In `CollectRawMetrics`, after `DroppedUnservable: aggregated.DroppedUnservable,`:
```go
LengthCappedRequests: aggregated.LengthCappedRequests,
```

### File: `cmd/root.go`

**Purpose:** Add rope_scaling warnings, extend blacklist, fix comment, add LengthCappedRequests to anomaly output.

**Changes:**

1. After `if ropeMap, ok := ropeScaling.(map[string]any); ok {`, add else clause:
```go
} else {
	logrus.Warnf("--latency-model: rope_scaling present but not a JSON object (type %T); ignoring", ropeScaling)
}
```

2. After factor extraction attempt, add warning for non-float factor:
```go
} else if _, hasKey := ropeMap["factor"]; hasKey {
	logrus.Warnf("--latency-model: rope_scaling.factor present but not a valid float64; ignoring")
}
```

3. Add gemma3 model_type exclusion (wrap the existing rope_scaling block):
```go
// vLLM skips rope_scaling for gemma3 (_get_and_verify_max_len): max_position_embeddings is pre-scaled
modelType, _ := hfConfig.Raw["model_type"].(string)
if modelType != "gemma3" {
	if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok {
		// ... existing rope_scaling logic unchanged ...
	}
}
```

4. Fix kvFeasibleMax comment:
```go
kvFeasibleMax := int(totalKVBlocks * blockSizeTokens) // safe: totalKVBlocks * blockSizeTokens < maxModelLen (blocksNeeded > totalKVBlocks), fits in int
```

5. Update anomaly counters output to include LengthCappedRequests:
```go
if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 {
	// ... existing prints ...
	fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
}
```

### File: `sim/simulator_test.go`

**Purpose:** All new tests for BC-1 through BC-7.

All test implementations as described in the task breakdown above.
