# PR1284: SimResult SLO/Model/ITL Fields Implementation Plan

**Status:** Draft — awaiting human review
**Closes:** Fixes #1284

---

## Part 1: Design Validation

### A. Executive Summary

`blis calibrate` can only produce workload-aggregate statistics because `SimResult` — the per-request handoff struct from `blis replay` — carries only five fields: RequestID, TTFT, E2E, InputTokens, OutputTokens. This PR adds three fields that `sim.RequestMetrics` already tracks but does not expose: `SLOClass`, `Model`, and `ITLMeanUs`. With those fields in `SimResult`, `PrepareCalibrationPairs` can bucket matched pairs by SLO class and model tag, and `blis calibrate` can log per-class MAPE lines — answering "is the simulator more accurate for `standard` requests than `batch` requests?"

The change is purely additive: the three new JSON fields use `omitempty` so existing consumers that do not know the fields simply never see them.

**PR size tier:** Medium (5 files changed, behavioral logic change in `PrepareCalibrationPairs`).

### B. Behavioral Contracts

**BC-1: New fields propagate through extractSimResults**
- GIVEN a completed replay request with SLOClass="standard", Model="qwen3-14b", and ITL=5.0ms
- WHEN `extractSimResults` serializes that request's `RequestMetrics`
- THEN the resulting `SimResult` has `SLOClass="standard"`, `Model="qwen3-14b"`, `ITLMeanUs=5000.0`

**BC-2: New fields are omitted when zero/empty (backward compatibility)**
- GIVEN a `SimResult` with empty `SLOClass` and `Model`, and `ITLMeanUs=0`
- WHEN the struct is JSON-marshaled
- THEN the JSON output does not contain `"slo_class"`, `"model"`, or `"itl_mean_us"` keys

**BC-3: Per-SLO breakdown buckets requests correctly**
- GIVEN SimResults with mixed SLO classes ("standard" and "batch"), each matched to real trace records
- WHEN `PrepareCalibrationPairs` processes these results
- THEN `pairs.BySLO["standard"]` contains only the TTFT and E2E pairs for "standard" requests, and `pairs.BySLO["batch"]` contains only the pairs for "batch" requests

**BC-4: Per-model breakdown buckets requests correctly**
- GIVEN SimResults with mixed model tags ("qwen3-14b" and "llama3-8b"), each matched to real records
- WHEN `PrepareCalibrationPairs` processes these results
- THEN `pairs.ByModel["qwen3-14b"]` and `pairs.ByModel["llama3-8b"]` each contain only the respective requests

**BC-5: Empty SLO class and empty model do not populate breakdown maps**
- GIVEN SimResults with empty `SLOClass` and `Model` fields
- WHEN `PrepareCalibrationPairs` processes them
- THEN `pairs.BySLO` and `pairs.ByModel` are both empty (no panic, no spurious key)

**BC-6: blis calibrate logs per-SLO MAPE in sorted order**
- GIVEN a calibration run where sim results contain requests from two SLO classes ("batch" and "standard")
- WHEN `blis calibrate` runs
- THEN per-SLO MAPE lines are printed for each SLO class in lexicographic order (determinism, R2)

### C. Component Interaction

```
cmd/replay.go
  extractSimResults()
    reads: sim.Metrics.Requests[id].{SLOClass, Model, ITL}
    writes: workload.SimResult.{SLOClass, Model, ITLMeanUs}
       (ITL ms → µs conversion: ITLMeanUs = rm.ITL * 1000)

sim/workload/calibrate.go
  SimResult: +SLOClass string, +Model string, +ITLMeanUs float64
  BreakdownPairs: TTFT LatencyPair, E2E LatencyPair  (new type)
  CalibrationPairs: +BySLO map[string]*BreakdownPairs
                    +ByModel map[string]*BreakdownPairs
  PrepareCalibrationPairs(): populates BySLO/ByModel when sr.SLOClass/sr.Model != ""
  MapePct(real, sim []float64) float64: new exported helper (MAPE as fraction; skips zero/NaN/Inf)

cmd/calibrate.go
  after existing quality log lines:
    if len(pairs.BySLO) > 0 → log per-SLO MAPE (sorted keys)
    if len(pairs.ByModel) > 0 → log per-model MAPE (sorted keys)
```

**Data flow note:** `BreakdownPairs` is a new unexported-to-the-consumer type. Callers (`cmd/calibrate.go`) access it via the `CalibrationPairs.BySLO` and `CalibrationPairs.ByModel` maps which are exported.

### D. Deviation Log

| # | Type | Source says | Plan does | Reason |
|---|------|-------------|-----------|--------|
| D-1 | CORRECTION | `BySLO map[string]*LatencyPair` | `BySLO map[string]*BreakdownPairs` where `BreakdownPairs` has `TTFT LatencyPair` and `E2E LatencyPair` | The issue's log code references `p.TTFT.Real` and `p.E2E.Real`, which are not fields of `LatencyPair` (which only has `Real []float64` and `Sim []float64`). A richer struct is needed to hold both dimensions per breakdown key. |
| D-2 | CORRECTION | `len(p.Real)` in log line | `len(p.TTFT.Real)` | After D-1 fix, `p` is `*BreakdownPairs` which has no `Real` field. `len(p.TTFT.Real)` is the correct request count for the SLO/model bucket. |
| D-3 | CORRECTION | `mape(p.TTFT.Real, p.TTFT.Sim)` — function named `mape` | Function named `mapePct` | Avoids shadowing or ambiguity with the inline MAPE computation in `ComputeCalibration`. The `Pct` suffix clarifies it returns a fraction (not a percentage), consistent with the `*100` in the format string. |
| D-4 | CLARIFICATION | Issue says add tests for `ByModel` in `sim/workload/calibrate_test.go` | Also add test for new `SLOClass`/`Model`/`ITL` field propagation in `cmd/replay_test.go` | The issue only specifies tests in `calibrate_test.go`, but `extractSimResults` has its own test file in `cmd/replay_test.go`. Adding a test there is required by BC-1 and ensures the unit is covered at the right layer. |

### E. Review Guide

Reviewers should focus on:
1. **BC-2 backward compatibility**: Confirm `omitempty` is present on all three new fields and JSON round-trip drops them when zero/empty.
2. **BC-1 unit conversion**: `rm.ITL` in `RequestMetrics` is in milliseconds; `ITLMeanUs` must be microseconds. Verify `* 1000` in `extractSimResults`.
3. **D-1 type correction**: Confirm `BreakdownPairs` has both `TTFT` and `E2E` sub-fields, not just one.
4. **Determinism (INV-6, R2)**: `sort.Strings(keys)` before iterating `BySLO`/`ByModel` in the log output.
5. **R4 construction sites**: `BreakdownPairs` has a single construction site in `PrepareCalibrationPairs`; no other code constructs it inline.

---

## Part 2: Executable Tasks

### F. Implementation Overview

4 implementation tasks + 1 verification task, strictly sequential (each builds on the previous).

| Task | Description | Files | BCs |
|------|-------------|-------|-----|
| T-1 | Extend `SimResult` with 3 new fields | `sim/workload/calibrate.go`, `cmd/replay_test.go` | BC-1, BC-2 |
| T-2 | Populate new fields in `extractSimResults` | `cmd/replay.go`, `cmd/replay_test.go` | BC-1 |
| T-3 | Add `BreakdownPairs`, extend `CalibrationPairs`, populate breakdowns | `sim/workload/calibrate.go`, `sim/workload/calibrate_test.go` | BC-3, BC-4, BC-5 |
| T-4 | Add `mapePct` helper + per-SLO/model log in `blis calibrate` | `sim/workload/calibrate.go`, `cmd/calibrate.go` | BC-6 |

---

### Task 1: Extend SimResult with SLOClass, Model, ITLMeanUs (BC-1, BC-2)

**Files:** modify `sim/workload/calibrate.go`; add test to `cmd/replay_test.go`

**Step 1 — Write failing test** (`cmd/replay_test.go`):

```go
func TestSimResult_NewFields_JSONOmitWhenEmpty(t *testing.T) {
    // BC-2: omitempty means empty SLOClass/Model and zero ITLMeanUs are omitted from JSON
    sr := workload.SimResult{
        RequestID:    1,
        TTFT:         100.0,
        E2E:          200.0,
        InputTokens:  10,
        OutputTokens: 5,
        // SLOClass, Model, ITLMeanUs intentionally zero/empty
    }
    data, err := json.Marshal(sr)
    if err != nil {
        t.Fatalf("json.Marshal: %v", err)
    }
    s := string(data)
    if strings.Contains(s, "slo_class") {
        t.Errorf("omitempty: slo_class should be absent, got: %s", s)
    }
    if strings.Contains(s, `"model"`) {
        t.Errorf("omitempty: model should be absent, got: %s", s)
    }
    if strings.Contains(s, "itl_mean_us") {
        t.Errorf("omitempty: itl_mean_us should be absent, got: %s", s)
    }

    // GIVEN non-empty fields: all three must round-trip
    sr2 := workload.SimResult{
        RequestID:    2,
        TTFT:         100.0,
        E2E:          200.0,
        InputTokens:  10,
        OutputTokens: 5,
        SLOClass:     "standard",
        Model:        "qwen3-14b",
        ITLMeanUs:    5000.0,
    }
    data2, err := json.Marshal(sr2)
    if err != nil {
        t.Fatalf("json.Marshal (non-empty): %v", err)
    }
    var got workload.SimResult
    if err := json.Unmarshal(data2, &got); err != nil {
        t.Fatalf("json.Unmarshal: %v", err)
    }
    if got.SLOClass != "standard" {
        t.Errorf("SLOClass round-trip: got %q, want %q", got.SLOClass, "standard")
    }
    if got.Model != "qwen3-14b" {
        t.Errorf("Model round-trip: got %q, want %q", got.Model, "qwen3-14b")
    }
    if got.ITLMeanUs != 5000.0 {
        t.Errorf("ITLMeanUs round-trip: got %f, want 5000.0", got.ITLMeanUs)
    }
}
```

**Step 2 — Run test to confirm failure:**
```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr1284-sim-result-slo-model-itl
go test ./cmd/... -run TestSimResult_NewFields_JSONOmitWhenEmpty
# Expected: compile error (SLOClass, Model, ITLMeanUs not yet on SimResult)
```

**Step 3 — Implement:** In `sim/workload/calibrate.go`, replace the `SimResult` struct (lines 85–91):

```go
// SimResult holds per-request sim output for calibration matching.
// TTFT and E2E are server-side latencies in microseconds (simulation ticks).
// SLOClass, Model, and ITLMeanUs are optional — omitted from JSON when zero/empty
// so existing consumers that do not know these fields are unaffected.
type SimResult struct {
    RequestID    int     `json:"request_id"`
    TTFT         float64 `json:"ttft_us"` // server-side TTFT in microseconds
    E2E          float64 `json:"e2e_us"`  // server-side E2E in microseconds
    InputTokens  int     `json:"input_tokens"`
    OutputTokens int     `json:"output_tokens"`
    SLOClass     string  `json:"slo_class,omitempty"`   // SLO tier (e.g., "standard", "batch"); empty if not set
    Model        string  `json:"model,omitempty"`       // model tag; empty if not set
    ITLMeanUs    float64 `json:"itl_mean_us,omitempty"` // mean ITL in microseconds (rm.ITL ms * 1000); 0 if not computed
}
```

**Step 4 — Run test to confirm pass:**
```bash
go test ./cmd/... -run TestSimResult_NewFields_JSONOmitWhenEmpty
# Expected: PASS
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/workload/... ./cmd/...
```

**Step 6 — Commit:**
```bash
git add sim/workload/calibrate.go cmd/replay_test.go
git commit -m "feat(calibrate): add SLOClass, Model, ITLMeanUs fields to SimResult (BC-1, BC-2)"
```

---

### Task 2: Populate new fields in extractSimResults (BC-1)

**Files:** modify `cmd/replay.go`; modify `cmd/replay_test.go`

**Step 1 — Write failing test** (add to `cmd/replay_test.go`):

```go
func TestExtractSimResults_PropagatesSLOClassModelITL(t *testing.T) {
    // GIVEN a Metrics struct with one completed request that has SLOClass, Model, and ITL set
    m := sim.NewMetrics()
    m.RequestTTFTs["request_0"] = 1000.0
    m.RequestE2Es["request_0"] = 5000.0
    m.Requests["request_0"] = sim.RequestMetrics{
        NumPrefillTokens: 100,
        NumDecodeTokens:  50,
        SLOClass:         "standard",
        Model:            "qwen3-14b",
        ITL:              5.0, // milliseconds
    }

    // WHEN extractSimResults is called
    results := extractSimResults(m)

    // THEN SLOClass, Model, and ITLMeanUs are populated correctly (BC-1)
    if len(results) != 1 {
        t.Fatalf("want 1 result, got %d", len(results))
    }
    r := results[0]
    if r.SLOClass != "standard" {
        t.Errorf("SLOClass: got %q, want %q", r.SLOClass, "standard")
    }
    if r.Model != "qwen3-14b" {
        t.Errorf("Model: got %q, want %q", r.Model, "qwen3-14b")
    }
    // ITL is 5.0ms in RequestMetrics → ITLMeanUs = 5000.0µs
    if r.ITLMeanUs != 5000.0 {
        t.Errorf("ITLMeanUs: got %f, want 5000.0 (5ms * 1000)", r.ITLMeanUs)
    }
}
```

**Step 2 — Run test to confirm failure:**
```bash
go test ./cmd/... -run TestExtractSimResults_PropagatesSLOClassModelITL
# Expected: FAIL (SLOClass=="" Model=="" ITLMeanUs==0)
```

**Step 3 — Implement:** In `cmd/replay.go`, find the `results = append(results, workload.SimResult{...})` block (lines 478–484). Replace:

```go
results = append(results, workload.SimResult{
    RequestID:    id,
    TTFT:         ttftUs,
    E2E:          e2eUs,
    InputTokens:  rm.NumPrefillTokens,
    OutputTokens: rm.NumDecodeTokens,
    SLOClass:     rm.SLOClass,
    Model:        rm.Model,
    ITLMeanUs:    rm.ITL * 1000, // rm.ITL is ms; ITLMeanUs is µs
})
```

**Step 4 — Run test to confirm pass:**
```bash
go test ./cmd/... -run TestExtractSimResults_PropagatesSLOClassModelITL
# Expected: PASS
```

Also run all existing extractSimResults tests to confirm no regression:
```bash
go test ./cmd/... -run TestExtractSimResults
# Expected: all PASS
```

**Step 5 — Lint:**
```bash
golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```bash
git add cmd/replay.go cmd/replay_test.go
git commit -m "feat(replay): populate SLOClass, Model, ITLMeanUs in extractSimResults (BC-1)"
```

---

### Task 3: Add BreakdownPairs type, extend CalibrationPairs, populate breakdowns (BC-3, BC-4, BC-5)

**Files:** modify `sim/workload/calibrate.go`; add tests to `sim/workload/calibrate_test.go`

**Step 1 — Write failing tests** (add to `sim/workload/calibrate_test.go`):

```go
func TestPrepareCalibrationPairs_SLOBreakdown(t *testing.T) {
    // GIVEN 4 matched requests with two SLO classes
    realRecords := []TraceRecord{
        {RequestID: 0, FirstChunkTimeUs: 500, LastChunkTimeUs: 1000, SendTimeUs: 0},
        {RequestID: 1, FirstChunkTimeUs: 1500, LastChunkTimeUs: 2000, SendTimeUs: 1000},
        {RequestID: 2, FirstChunkTimeUs: 2500, LastChunkTimeUs: 3000, SendTimeUs: 2000},
        {RequestID: 3, FirstChunkTimeUs: 3500, LastChunkTimeUs: 4000, SendTimeUs: 3000},
    }
    simResults := []SimResult{
        {RequestID: 0, TTFT: 450, E2E: 900, SLOClass: "standard"},
        {RequestID: 1, TTFT: 480, E2E: 950, SLOClass: "batch"},
        {RequestID: 2, TTFT: 460, E2E: 920, SLOClass: "standard"},
        {RequestID: 3, TTFT: 490, E2E: 960, SLOClass: "batch"},
    }

    // WHEN PrepareCalibrationPairs runs
    pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
    if err != nil {
        t.Fatal(err)
    }

    // THEN BySLO["standard"] contains 2 pairs, BySLO["batch"] contains 2 pairs (BC-3)
    if len(pairs.BySLO) != 2 {
        t.Fatalf("BySLO len: got %d, want 2", len(pairs.BySLO))
    }
    stdPairs, ok := pairs.BySLO["standard"]
    if !ok {
        t.Fatal("BySLO missing 'standard' key")
    }
    if len(stdPairs.TTFT.Real) != 2 {
        t.Errorf("BySLO[standard] TTFT count: got %d, want 2", len(stdPairs.TTFT.Real))
    }
    if len(stdPairs.E2E.Real) != 2 {
        t.Errorf("BySLO[standard] E2E count: got %d, want 2", len(stdPairs.E2E.Real))
    }
    batchPairs, ok := pairs.BySLO["batch"]
    if !ok {
        t.Fatal("BySLO missing 'batch' key")
    }
    if len(batchPairs.TTFT.Real) != 2 {
        t.Errorf("BySLO[batch] TTFT count: got %d, want 2", len(batchPairs.TTFT.Real))
    }
}

func TestPrepareCalibrationPairs_ModelBreakdown(t *testing.T) {
    // GIVEN 3 matched requests with two model tags
    realRecords := []TraceRecord{
        {RequestID: 0, FirstChunkTimeUs: 500, LastChunkTimeUs: 1000, SendTimeUs: 0},
        {RequestID: 1, FirstChunkTimeUs: 1500, LastChunkTimeUs: 2000, SendTimeUs: 1000},
        {RequestID: 2, FirstChunkTimeUs: 2500, LastChunkTimeUs: 3000, SendTimeUs: 2000},
    }
    simResults := []SimResult{
        {RequestID: 0, TTFT: 450, E2E: 900, Model: "qwen3-14b"},
        {RequestID: 1, TTFT: 480, E2E: 950, Model: "llama3-8b"},
        {RequestID: 2, TTFT: 460, E2E: 920, Model: "qwen3-14b"},
    }

    pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
    if err != nil {
        t.Fatal(err)
    }

    // THEN ByModel["qwen3-14b"] has 2 pairs, ByModel["llama3-8b"] has 1 pair (BC-4)
    if len(pairs.ByModel) != 2 {
        t.Fatalf("ByModel len: got %d, want 2", len(pairs.ByModel))
    }
    q, ok := pairs.ByModel["qwen3-14b"]
    if !ok {
        t.Fatal("ByModel missing 'qwen3-14b'")
    }
    if len(q.TTFT.Real) != 2 {
        t.Errorf("ByModel[qwen3-14b] TTFT count: got %d, want 2", len(q.TTFT.Real))
    }
    l, ok := pairs.ByModel["llama3-8b"]
    if !ok {
        t.Fatal("ByModel missing 'llama3-8b'")
    }
    if len(l.TTFT.Real) != 1 {
        t.Errorf("ByModel[llama3-8b] TTFT count: got %d, want 1", len(l.TTFT.Real))
    }
}

func TestPrepareCalibrationPairs_EmptySLOAndModel_NoBreakdown(t *testing.T) {
    // GIVEN requests with no SLO class and no model tag (BC-5)
    realRecords := []TraceRecord{
        {RequestID: 0, FirstChunkTimeUs: 500, LastChunkTimeUs: 1000, SendTimeUs: 0},
        {RequestID: 1, FirstChunkTimeUs: 1500, LastChunkTimeUs: 2000, SendTimeUs: 1000},
    }
    simResults := []SimResult{
        {RequestID: 0, TTFT: 450, E2E: 900, SLOClass: "", Model: ""},
        {RequestID: 1, TTFT: 480, E2E: 950, SLOClass: "", Model: ""},
    }

    pairs, _, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})
    if err != nil {
        t.Fatal(err)
    }

    // THEN BySLO and ByModel are both empty (no panic)
    if len(pairs.BySLO) != 0 {
        t.Errorf("BySLO should be empty for requests without SLO class, got len=%d", len(pairs.BySLO))
    }
    if len(pairs.ByModel) != 0 {
        t.Errorf("ByModel should be empty for requests without model tag, got len=%d", len(pairs.ByModel))
    }
}
```

**Step 2 — Run tests to confirm failure:**
```bash
go test ./sim/workload/... -run "TestPrepareCalibrationPairs_SLOBreakdown|TestPrepareCalibrationPairs_ModelBreakdown|TestPrepareCalibrationPairs_EmptySLOAndModel"
# Expected: compile error (BreakdownPairs, BySLO, ByModel do not exist yet)
```

**Step 3 — Implement:** In `sim/workload/calibrate.go`:

**3a. Add `BreakdownPairs` type** (after the `LatencyPair` type definition, around line 97):

```go
// BreakdownPairs holds matched real-vs-sim latency vectors for a single breakdown dimension
// (e.g., one SLO class or one model tag). Tracks TTFT and E2E separately.
type BreakdownPairs struct {
    TTFT LatencyPair
    E2E  LatencyPair
}
```

**3b. Extend `CalibrationPairs`** (add two fields after `ITLDropped`):

```go
type CalibrationPairs struct {
    TTFT               LatencyPair
    E2E                LatencyPair
    ITL                LatencyPair
    BySLO              map[string]*BreakdownPairs // keyed by SLOClass; only populated when SLOClass is non-empty
    ByModel            map[string]*BreakdownPairs // keyed by Model; only populated when Model is non-empty
    TokenMismatchCount int
    ExcludedWarmUp     int
    MatchedCount       int
    UnmatchedReal      int
    UnmatchedSim       int
    ITLDropped         int // Requests dropped from ITL due to clock skew (all negative deltas)
}
```

**3c. Initialize maps in `PrepareCalibrationPairs`** (replace `pairs := &CalibrationPairs{}`):

```go
pairs := &CalibrationPairs{
    BySLO:   make(map[string]*BreakdownPairs),
    ByModel: make(map[string]*BreakdownPairs),
}
```

**3d. Populate breakdowns per matched request** (add after `pairs.E2E.Sim = append(...)` in the match loop):

```go
// Per-SLO breakdown (only when SLOClass is set)
if sr.SLOClass != "" {
    bp, ok := pairs.BySLO[sr.SLOClass]
    if !ok {
        bp = &BreakdownPairs{}
        pairs.BySLO[sr.SLOClass] = bp
    }
    bp.TTFT.Real = append(bp.TTFT.Real, realTTFT)
    bp.TTFT.Sim = append(bp.TTFT.Sim, simTTFT)
    bp.E2E.Real = append(bp.E2E.Real, realE2E)
    bp.E2E.Sim = append(bp.E2E.Sim, simE2E)
}
// Per-model breakdown (only when Model is set)
if sr.Model != "" {
    bp, ok := pairs.ByModel[sr.Model]
    if !ok {
        bp = &BreakdownPairs{}
        pairs.ByModel[sr.Model] = bp
    }
    bp.TTFT.Real = append(bp.TTFT.Real, realTTFT)
    bp.TTFT.Sim = append(bp.TTFT.Sim, simTTFT)
    bp.E2E.Real = append(bp.E2E.Real, realE2E)
    bp.E2E.Sim = append(bp.E2E.Sim, simE2E)
}
```

**Step 4 — Run tests to confirm pass:**
```bash
go test ./sim/workload/... -run "TestPrepareCalibrationPairs_SLOBreakdown|TestPrepareCalibrationPairs_ModelBreakdown|TestPrepareCalibrationPairs_EmptySLOAndModel"
# Expected: PASS
```

Also run all existing calibrate tests:
```bash
go test ./sim/workload/... -run TestPrepareCalibrationPairs
# Expected: all PASS
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/workload/...
```

**Step 6 — Commit:**
```bash
git add sim/workload/calibrate.go sim/workload/calibrate_test.go
git commit -m "feat(calibrate): add BreakdownPairs, BySLO/ByModel to CalibrationPairs (BC-3, BC-4, BC-5)"
```

---

### Task 4: Add mapePct helper + per-SLO/model log output (BC-6)

**Files:** modify `sim/workload/calibrate.go`; modify `cmd/calibrate.go`

Note: BC-6 is a `logrus.Infof` output to stderr — not part of the deterministic stdout. No test for the exact log strings; the per-SLO/model MAPE computation is covered by BC-3/BC-4 tests in Task 3.

**Step 1 — Add `MapePct` helper** to `sim/workload/calibrate.go` (after `qualityRating`, in the helpers section):

```go
// MapePct computes mean absolute percentage error between real and sim slices.
// Pairs where real==0, NaN, or Inf are skipped. Returns 0 if no valid pairs.
// Returns a fraction (not a percentage) — multiply by 100 for display.
func MapePct(real, sim []float64) float64 {
    var sum float64
    count := 0
    for i := range real {
        if real[i] == 0 || math.IsNaN(real[i]) || math.IsInf(real[i], 0) {
            continue
        }
        err := math.Abs(real[i]-sim[i]) / real[i]
        if math.IsNaN(err) || math.IsInf(err, 0) {
            continue
        }
        sum += err
        count++
    }
    if count == 0 {
        return 0
    }
    return sum / float64(count)
}
```

**Step 2 — Add per-SLO/model log output** in `cmd/calibrate.go`, after the existing request-level quality log block (after the `itl` quality log line, before the closing `},`):

```go
// Per-SLO class breakdown (when data present)
if len(pairs.BySLO) > 0 {
    sloKeys := make([]string, 0, len(pairs.BySLO))
    for k := range pairs.BySLO {
        sloKeys = append(sloKeys, k)
    }
    sort.Strings(sloKeys) // deterministic output (R2, INV-6)
    logrus.Infof("Per-SLO-class calibration:")
    for _, slo := range sloKeys {
        p := pairs.BySLO[slo]
        if len(p.TTFT.Real) == 0 {
            continue
        }
        logrus.Infof("  SLO=%s: n=%d TTFT-MAPE=%.1f%% E2E-MAPE=%.1f%%",
            slo, len(p.TTFT.Real),
            workload.MapePct(p.TTFT.Real, p.TTFT.Sim)*100,
            workload.MapePct(p.E2E.Real, p.E2E.Sim)*100)
    }
}
// Per-model breakdown (when data present)
if len(pairs.ByModel) > 0 {
    modelKeys := make([]string, 0, len(pairs.ByModel))
    for k := range pairs.ByModel {
        modelKeys = append(modelKeys, k)
    }
    sort.Strings(modelKeys) // deterministic output (R2, INV-6)
    logrus.Infof("Per-model calibration:")
    for _, model := range modelKeys {
        p := pairs.ByModel[model]
        if len(p.TTFT.Real) == 0 {
            continue
        }
        logrus.Infof("  model=%s: n=%d TTFT-MAPE=%.1f%% E2E-MAPE=%.1f%%",
            model, len(p.TTFT.Real),
            workload.MapePct(p.TTFT.Real, p.TTFT.Sim)*100,
            workload.MapePct(p.E2E.Real, p.E2E.Sim)*100)
    }
}
```

**Note**: `MapePct` is exported so `cmd/calibrate.go` (package `cmd`) can call it from package `workload`. The `Pct` suffix signals it returns a fraction, not a percentage — consistent with the `*100` in the format strings.

**Step 3 — Run full test suite:**
```bash
go test ./sim/workload/... ./cmd/...
# Expected: all PASS
```

**Step 4 — Build:**
```bash
go build ./...
# Expected: exit 0
```

**Step 5 — Lint:**
```bash
golangci-lint run ./sim/workload/... ./cmd/...
# Expected: 0 issues
```

**Step 6 — Commit:**
```bash
git add sim/workload/calibrate.go cmd/calibrate.go
git commit -m "feat(calibrate): add MapePct helper (NaN-safe) + per-SLO/model MAPE log output (BC-6)"
```

---

## Part 3: Sanity Checklist

### J. Antipattern Prevention

- [ ] **R1 (no silent continue)**: All exclusion paths in `PrepareCalibrationPairs` that skip adding to `BySLO`/`ByModel` are guarded by explicit `!= ""` checks (not silent drops)
- [ ] **R2 (determinism)**: Both `BySLO` and `ByModel` keys are sorted before iterating in the log output (`sort.Strings`)
- [ ] **R4 (construction sites)**: `BreakdownPairs{}` is constructed in exactly one place: inline within `PrepareCalibrationPairs` at the `!ok` branch
- [ ] **R8 (exported mutable maps)**: `BySLO` and `ByModel` are exported fields of `CalibrationPairs` (needed for `cmd/calibrate.go` access). Acceptable — `CalibrationPairs` is a value-result type, not a shared singleton. Document that callers treat it as read-only.
- [ ] **R11 (division guard)**: `MapePct` skips pairs where `real[i] == 0` and guards `count == 0` before dividing
- [ ] **INV-6 (determinism)**: Breakdown log output is sorted; JSON output of `SimResult` is deterministic (field order is struct order, not map iteration)
- [ ] **BC-2 backward compatibility**: JSON marshal of `SimResult` with zero-value new fields produces identical output to before this PR (verified by `TestSimResult_NewFields_JSONOmitWhenEmpty`)

---

## Appendix: File-Level Details

### `sim/workload/calibrate.go`

| Change | Location | Description |
|--------|----------|-------------|
| `SimResult` +3 fields | lines 85–91 → lines 85–93 | Add `SLOClass`, `Model`, `ITLMeanUs` with `omitempty` |
| New type `BreakdownPairs` | after `LatencyPair` (line 97) | `TTFT LatencyPair`, `E2E LatencyPair` |
| `CalibrationPairs` +2 fields | line 100–110 | Add `BySLO`, `ByModel` as `map[string]*BreakdownPairs` |
| `PrepareCalibrationPairs` init | line 130 | Initialize maps in struct literal |
| `PrepareCalibrationPairs` loop | after line 180 | Populate `BySLO`, `ByModel` per matched pair |
| `MapePct` helper | after `qualityRating` (line 490) | Exported MAPE helper for cmd/ use |

### `cmd/replay.go`

| Change | Location | Description |
|--------|----------|-------------|
| `extractSimResults` append | lines 478–484 | Add `SLOClass`, `Model`, `ITLMeanUs: rm.ITL * 1000` |

### `cmd/calibrate.go`

| Change | Location | Description |
|--------|----------|-------------|
| Per-SLO/model log output | after line 213 | Two guarded log blocks with sorted key iteration |
| Import `sort` | import block | Must be added (not yet imported) |

### `sim/workload/calibrate_test.go`

| Test | BCs | Description |
|------|-----|-------------|
| `TestPrepareCalibrationPairs_SLOBreakdown` | BC-3 | Mixed SLO classes → correct buckets |
| `TestPrepareCalibrationPairs_ModelBreakdown` | BC-4 | Mixed model tags → correct buckets |
| `TestPrepareCalibrationPairs_EmptySLOAndModel_NoBreakdown` | BC-5 | Empty fields → empty maps, no panic |

### `cmd/replay_test.go`

| Test | BCs | Description |
|------|-----|-------------|
| `TestSimResult_NewFields_JSONOmitWhenEmpty` | BC-1, BC-2 | Omitempty + round-trip for new fields |
| `TestExtractSimResults_PropagatesSLOClassModelITL` | BC-1 | New fields flow from RequestMetrics to SimResult |
