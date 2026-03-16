# `blis calibrate` CLI Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `blis calibrate` command that compares real observed latencies (from `blis observe`) against simulator predictions (from `blis replay`) and produces a JSON calibration report.

**The problem today:** After running `blis observe` to capture real vLLM latencies and `blis replay` to produce simulated latencies for the same request trace, there is no CLI tool to compare the two. Users must write their own matching and statistical comparison code, making the observe/replay/calibrate loop (#652) incomplete.

**What this PR adds:**
1. **`blis calibrate` command** — reads a TraceV2 file (from `blis observe`) and a `SimResult` JSON file (from `blis replay`), matches requests by ID, applies network normalization, and writes a calibration report JSON.
2. **Report structure** — includes `trace_info` (matched/excluded/mismatched counts), `metrics.ttft` and `metrics.e2e` (MAPE, Pearson r, P50/P90/P95/P99, bias direction, quality rating), `config_match`, and `known_limitations`.
3. **Header-derived defaults** — `--warmup-requests` and `--network-rtt-us` fall back to values in the TraceV2 header when not explicitly set by the user (sentinel `-1` distinguishes "not set" from "set to 0").
4. **Summary stderr log** — reports matched pairs, TTFT MAPE, E2E MAPE, quality ratings to stderr after writing the report.

**Why this matters:** Closes the observe/replay/calibrate loop (#652). Users can now complete the full calibration cycle from a single command: `blis observe` → `blis replay` → `blis calibrate` → read JSON report. Downstream work (automated coefficient optimization) depends on this report format.

**Architecture:** The library code in `sim/workload/calibrate.go` is already complete (`PrepareCalibrationPairs`, `ComputeCalibration`, `BuildCalibrationReport`). This PR adds only `cmd/calibrate.go` (cobra command + I/O glue) and `cmd/calibrate_test.go` (6 unit tests). No new interfaces, no new module boundaries. Follows the same pattern as `cmd/replay.go` — package-level flag vars, `init()` registration with `rootCmd.AddCommand`.

**Source:** GitHub issue #658

**Closes:** Closes #658

**Behavioral Contracts:** See Part 1, Section B below

---

## PART 1: Design Validation

### A) Executive Summary

`blis calibrate` is the third step of the observe/replay/calibrate loop. It receives two files:
- A TraceV2 file (real observed timing from `blis observe`, YAML header + CSV data)
- A `[]SimResult` JSON file (simulated timing from `blis replay --results-path`)

It matches records by `RequestID`, applies optional network normalization, and writes a JSON calibration report. The library code is fully complete; this PR is pure CLI glue.

**System position:** `blis observe` → real trace → `blis replay` → sim results → **`blis calibrate`** → JSON report. Nothing depends on this PR's output format at the library level (report is a file, not a Go interface).

**Adjacent blocks:** `sim/workload/calibrate.go` (library, already merged), `sim/workload/tracev2.go` (`LoadTraceV2`, `TraceRecord`), `cmd/root.go` (cobra root for `AddCommand`).

**DEVIATION flags:** None. Issue spec matches current codebase exactly. `SimResult.TTFT`/`E2E` use `ttft_us`/`e2e_us` JSON tags (verified in `sim/workload/calibrate.go:51-57`). `LoadTraceV2` exists at `sim/workload/tracev2.go:155`. `PrepareCalibrationPairs`, `BuildCalibrationReport` exist at `sim/workload/calibrate.go:78,209`.

---

### B) Behavioral Contracts

**Positive contracts:**

**BC-1: Basic calibration report**
- GIVEN a valid TraceV2 file with N requests and a matching `[]SimResult` JSON with N entries
- WHEN `blis calibrate --trace-header H --trace-data D --sim-results S --report R` is run
- THEN a JSON file is written to R containing `trace_info.matched_pairs == N`, `metrics.ttft`, `metrics.e2e`, and `known_limitations` (non-empty)
- MECHANISM: `LoadTraceV2` → `json.Unmarshal([]SimResult)` → `PrepareCalibrationPairs` → `BuildCalibrationReport` → `json.MarshalIndent` → `os.WriteFile`

**BC-2: Warm-up exclusion from header**
- GIVEN a TraceV2 header with `warm_up_requests: 3` and 10 requests in the trace, and `--warmup-requests` flag not provided
- WHEN `blis calibrate` is run
- THEN the report shows `trace_info.warm_up_excluded == 3` and `trace_info.matched_pairs == 7`
- MECHANISM: sentinel `-1` default; when `calibrateWarmUpRequests == -1`, read `trace.Header.WarmUpRequests`

**BC-3: Warm-up exclusion from CLI flag overrides header**
- GIVEN a TraceV2 header with `warm_up_requests: 3` and 10 requests, and `--warmup-requests 0` explicitly set
- WHEN `blis calibrate` is run
- THEN the report shows `trace_info.warm_up_excluded == 0` and `trace_info.matched_pairs == 10` (flag overrides header)
- MECHANISM: `calibrateWarmUpRequests != -1` branch skips header fallback

**BC-4: Network RTT from header**
- GIVEN a TraceV2 header with `network.measured_rtt_ms: 1.0` and `--network-rtt-us` not provided
- WHEN `blis calibrate` is run
- THEN sim-side TTFT in the report is shifted by 1000 µs relative to server-side TTFT (RTT applied)
- MECHANISM: sentinel `-1` default; when `calibrateNetworkRTTUs == -1`, read `trace.Header.Network.MeasuredRTTMs * 1000`

**BC-5: Unmatched requests are counted, not errors**
- GIVEN a trace with request IDs [0,1,2] and sim results with IDs [0,1,3]
- WHEN `blis calibrate` is run
- THEN the command succeeds (exit 0), the report shows 2 matched pairs, and unmatched counts are non-zero; the report is valid JSON
- MECHANISM: `PrepareCalibrationPairs` returns `UnmatchedReal`/`UnmatchedSim` counts, not errors

**BC-6: Help text shows all flags**
- GIVEN `blis calibrate --help`
- WHEN executed
- THEN the output lists `--trace-header`, `--trace-data`, `--sim-results`, `--report`, `--warmup-requests`, `--network-rtt-us`, and `--network-bandwidth-mbps`
- MECHANISM: cobra `Flags()` registration in `init()`

**Negative contracts:**

**BC-7: Missing required flags fail with message**
- GIVEN any required flag (`--trace-header`, `--trace-data`, `--sim-results`, `--report`) is absent
- WHEN `blis calibrate` is run
- THEN the process exits non-zero with an error message identifying the missing flag; no partial output is written
- MECHANISM: explicit `logrus.Fatalf` check at the start of `Run`

**BC-8: Nonexistent input file fails with message**
- GIVEN `--trace-header` points to a file that does not exist
- WHEN `blis calibrate` is run
- THEN the process exits non-zero with a message containing the file path
- MECHANISM: `LoadTraceV2` returns an error, `logrus.Fatalf` propagates it

**Error handling contracts:**

**BC-9: Empty sim results file fails with message**
- GIVEN `--sim-results` points to a file containing `[]` (empty JSON array)
- WHEN `blis calibrate` is run
- THEN the process exits non-zero with a message indicating no sim results were found (cannot calibrate with zero data)
- MECHANISM: explicit length check after `json.Unmarshal`

**BC-10: Zero matched pairs fails with message**
- GIVEN a trace with request IDs [10,11,12] and sim results with IDs [0,1,2] (no overlap)
- WHEN `blis calibrate` is run
- THEN the process exits non-zero with a message indicating zero matching request IDs were found; no report file is written
- MECHANISM: `pairs.MatchedCount == 0` guard after `PrepareCalibrationPairs`, `logrus.Fatalf` (R1: no silent data loss)

**BC-11: Negative --network-rtt-us fails with message**
- GIVEN `--network-rtt-us -5000` is passed (not the sentinel -1)
- WHEN `blis calibrate` is run
- THEN the process exits non-zero with a message stating --network-rtt-us must be >= 0
- MECHANISM: explicit validation `calibrateNetworkRTTUs != -1 && calibrateNetworkRTTUs < 0` → `logrus.Fatalf` (R3)

---

### C) Component Interaction

```
cmd/calibrate.go
  │
  ├── reads TraceV2 ──────────────────────────── sim/workload/tracev2.go:LoadTraceV2
  │     (header YAML + data CSV → TraceV2{Header, []TraceRecord})
  │
  ├── reads SimResults ────────────────────────── cmd/calibrate.go (json.Unmarshal)
  │     (JSON file → []workload.SimResult)
  │
  ├── resolves config ─────────────────────────── cmd/calibrate.go (sentinel logic)
  │     (sentinel -1 → TraceV2.Header.WarmUpRequests / Network.MeasuredRTTMs)
  │
  ├── prepares pairs ──────────────────────────── sim/workload/calibrate.go:PrepareCalibrationPairs
  │     ([]TraceRecord + []SimResult + CalibrationConfig → *CalibrationPairs)
  │
  ├── builds report ───────────────────────────── sim/workload/calibrate.go:BuildCalibrationReport
  │     (*CalibrationPairs + ConfigMatchInfo → *CalibrationReport)
  │
  └── writes JSON ─────────────────────────────── cmd/calibrate.go (json.MarshalIndent + os.WriteFile)
        (*CalibrationReport → JSON file at --report path)
```

**API contracts (new types):** None. Only flag variables and cobra command (no exported types).

**State changes:** Package-level flag vars (unexported, mirroring replay.go pattern):
- `calibrateTraceHeaderPath string`
- `calibrateTraceDataPath string`
- `calibrateSimResultsPath string`
- `calibrateReportPath string`
- `calibrateWarmUpRequests int` (sentinel `-1`)
- `calibrateNetworkRTTUs int64` (sentinel `-1`)
- `calibrateNetworkBandwidthMbps float64`

**Extension friction:** Adding one more CLI flag requires 1 file change (`cmd/calibrate.go`). No interface changes, no type proliferation.

---

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| `--warmup-requests` default = 0, "use header if > 0" | sentinel `-1` default, fall back to header when `-1` | Cobra `int` cannot distinguish "not set" vs "set to 0" without sentinel. Issue spec implied this distinction; sentinel is the idiomatic solution. |
| `--network-rtt-us` default = 0, "use header if > 0" | sentinel `-1` default, fall back to header when `-1` | Same reason as above. |
| Issue mentions `ConfigMatchInfo{}` as empty with TODO | Implemented as empty struct, comment added | Confirmed correct per issue spec option (c). |
| Issue spec: `SimResult` JSON format with `RequestID`, `TTFT`, `E2E` | JSON tags are `request_id`, `ttft_us`, `e2e_us` (verified in calibrate.go:45-55) | CORRECTION: issue spec used Go field names in example; actual JSON tags differ. Using actual tags. |

---

### E) Review Guide

**The tricky part:** The sentinel `-1` logic for `--warmup-requests` and `--network-rtt-us`. The invariant is: "if the user explicitly passes 0, use 0; if they don't pass the flag at all, use the header value." The sentinel must be `-1`, not `0`, because the cobra default is what the user "didn't pass." Reviewers should verify BC-2 and BC-3 are both tested (header default vs CLI override).

**What to scrutinize:** BC-2/BC-3 warm-up precedence logic. BC-4 RTT unit conversion (`MeasuredRTTMs * 1000` → µs). BC-9 empty-results guard (prevents a confusing `ComputeCalibration` error about empty vectors).

**What's safe to skim:** The JSON read/write boilerplate. Flag registration. The stderr log summary (it's informational only).

**Known debt:**
- `ConfigMatchInfo{}` is empty — config matching deferred (#658 option c).
- `MetricComparison` struct in `sim/workload/calibrate.go` has no JSON tags — Go will serialize fields as PascalCase (`RealP50`, `SimP50`, `MAPE`, `PearsonR`, etc.) while the rest of the report uses snake_case (`matched_pairs`, `known_limitations`). This is a library-level inconsistency that this PR cannot fix (library is already merged). Future improvement: add `json:"real_p50"` tags to `MetricComparison` in a follow-up PR.
- Non-streaming `TraceRecord` entries (where `Streaming: false`) produce `realTTFT = FirstChunkTimeUs - SendTimeUs = E2E` (only 1 chunk). Comparing this against sim TTFT (server-side time-to-first-token) is semantically incorrect. Library `PrepareCalibrationPairs` does not filter by `Streaming`. For now, mixed streaming/non-streaming traces will show inflated TTFT MAPE. Best practice: use streaming-only traces for TTFT calibration.

---

## PART 2: Executable Implementation

### F) Implementation Overview

| File | Action |
|------|--------|
| `cmd/calibrate.go` | Create: cobra command, 7 flag vars, `Run` function (load → resolve → calibrate → write → log) |
| `cmd/calibrate_test.go` | Create: 6 tests (BC-1 basic, BC-2 warm-up from header, BC-3 warm-up from flag, BC-4 RTT from header, BC-5 unmatched, BC-11 negative RTT rejected) |

**Key decisions:**
- Flag vars are package-level (not struct fields) — consistent with `replay.go:26-30` pattern
- Tests use `t.TempDir()` and write synthetic fixture files — consistent with `replay_test.go:352` pattern
- Tests save/restore flag vars — consistent with replay end-to-end test pattern
- No new shared test helpers needed (all fixtures are tiny synthetic data)
- No golden dataset changes (this PR doesn't touch `sim/` output)

**Confirmation:** No dead code. All 7 flag vars are read in `Run`. All 6 test cases exercise distinct contracts. The cobra command is exercisable via `./blis calibrate --help` immediately after merge.

---

### G) Task Breakdown

#### Task 1: Cobra command skeleton + flag registration

**Contracts Implemented:** BC-6, BC-7

**Files:**
- Create: `cmd/calibrate.go`

**Step 1: Write failing test for flag registration**

Context: Verify all 7 flags are registered on `calibrateCmd`, mirroring the flag-registration test in `replay_test.go:18-66`.

```go
// In cmd/calibrate_test.go (create this file)
package cmd

import (
	"testing"
)

func TestCalibrateCmd_Flags_Registered(t *testing.T) {
	// GIVEN the calibrate command
	// WHEN we inspect its registered flags
	// THEN all 7 flags must be present
	flags := []string{
		"trace-header",
		"trace-data",
		"sim-results",
		"report",
		"warmup-requests",
		"network-rtt-us",
		"network-bandwidth-mbps",
	}
	for _, name := range flags {
		f := calibrateCmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("calibrateCmd missing flag --%s", name)
		}
	}
}
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/sri/Documents/Projects/inference-sim/.worktrees/pr20-calibrate-cli
go test ./cmd/... -run TestCalibrateCmd_Flags_Registered -v
```
Expected: FAIL with `undefined: calibrateCmd`

**Step 3: Implement the cobra command skeleton with flag registration**

Context: Create `cmd/calibrate.go` with the command struct, 7 package-level flag vars, `init()` registration, and a `Run` stub that calls `logrus.Fatalf` for any missing required flag. No calibration logic yet.

In `cmd/calibrate.go`:
```go
package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	calibrateTraceHeaderPath     string
	calibrateTraceDataPath       string
	calibrateSimResultsPath      string
	calibrateReportPath          string
	calibrateWarmUpRequests      int
	calibrateNetworkRTTUs        int64
	calibrateNetworkBandwidthMbps float64
)

var calibrateCmd = &cobra.Command{
	Use:   "calibrate",
	Short: "Compare real observed latencies against simulator predictions",
	Long: `Calibrate takes a TraceV2 file (from blis observe) and a SimResult JSON file
(from blis replay --results-path) and computes a calibration report comparing
real vs simulated TTFT and E2E latencies.

The report includes per-metric MAPE, Pearson r, percentile comparison, bias
direction, and a quality rating. Use --report to specify the output path.

Warm-up requests are excluded from comparison. By default, the warm-up count
is taken from the trace header (warm_up_requests field). Use --warmup-requests
to override. Pass --warmup-requests 0 to include all requests.

Network RTT and bandwidth adjustments shift sim-side latencies to client
perspective. By default, RTT is taken from the trace header
(network.measured_rtt_ms). Use --network-rtt-us to override in microseconds.

Example:
  blis calibrate --trace-header t.yaml --trace-data d.csv \
    --sim-results results.json --report calibration.json`,
	Run: func(cmd *cobra.Command, args []string) {
		if calibrateTraceHeaderPath == "" {
			logrus.Fatalf("--trace-header is required")
		}
		if calibrateTraceDataPath == "" {
			logrus.Fatalf("--trace-data is required")
		}
		if calibrateSimResultsPath == "" {
			logrus.Fatalf("--sim-results is required")
		}
		if calibrateReportPath == "" {
			logrus.Fatalf("--report is required")
		}
		// TODO: implement in Task 2
	},
}

func init() {
	calibrateCmd.Flags().StringVar(&calibrateTraceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateTraceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateSimResultsPath, "sim-results", "", "Path to SimResult JSON file (from blis replay --results-path; required)")
	calibrateCmd.Flags().StringVar(&calibrateReportPath, "report", "", "Path to write calibration report JSON (required)")
	calibrateCmd.Flags().IntVar(&calibrateWarmUpRequests, "warmup-requests", -1, "Number of initial requests to exclude (default: from trace header warm_up_requests; pass 0 to include all)")
	calibrateCmd.Flags().Int64Var(&calibrateNetworkRTTUs, "network-rtt-us", -1, "Network RTT in microseconds added to sim-side latencies (default: from trace header network.measured_rtt_ms)")
	calibrateCmd.Flags().Float64Var(&calibrateNetworkBandwidthMbps, "network-bandwidth-mbps", 0, "Network bandwidth in Mbps for upload/download delay calculation (default: 0 = no delay)")
	rootCmd.AddCommand(calibrateCmd)
}
```

**Step 4: Run test to verify it passes**

```bash
go test ./cmd/... -run TestCalibrateCmd_Flags_Registered -v
```
Expected: PASS

**Step 5: Run lint**

```bash
golangci-lint run ./cmd/...
```
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/calibrate.go cmd/calibrate_test.go
git commit -m "feat(cmd): add blis calibrate command skeleton with flag registration (BC-6, BC-7)

- Add calibrateCmd cobra command with 7 flag vars
- Register --trace-header, --trace-data, --sim-results, --report (required)
- Register --warmup-requests, --network-rtt-us, --network-bandwidth-mbps (optional)
- Sentinel -1 for warmup-requests and network-rtt-us (distinguishes unset from 0)
- Add TestCalibrateCmd_Flags_Registered

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: Implement Run function (load, calibrate, write)

**Contracts Implemented:** BC-1, BC-5, BC-8, BC-9, BC-10, BC-11

**Files:**
- Modify: `cmd/calibrate.go` (replace TODO stub with full implementation)

**Step 1: Write failing test for basic end-to-end calibration**

Context: Test the full command with synthetic TraceV2 + SimResult files. Mirroring `TestReplayCmd_EndToEnd_BlackboxMode` in replay_test.go (save/restore flag vars, temp dir).

```go
// In cmd/calibrate_test.go (add to existing file)
package cmd

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/inference-sim/sim/workload"
)

func TestCalibrateCmd_BasicReport_WritesMatchedPairs(t *testing.T) {
	// GIVEN a valid TraceV2 with 3 requests and matching SimResults
	dir := t.TempDir()

	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	simPath := filepath.Join(dir, "results.json")
	reportPath := filepath.Join(dir, "report.json")

	header := `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}

	// 3 requests with real timing: send=1000, first_chunk=5000, last_chunk=10000
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message\n" +
		"0,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,0,1000,5000,10000,5,ok,\n" +
		"1,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,100000,101000,105000,110000,5,ok,\n" +
		"2,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,200000,201000,205000,210000,5,ok,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	// SimResults matching request IDs 0, 1, 2
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 2, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	// Save and restore flag vars
	origHeader := calibrateTraceHeaderPath
	origData := calibrateTraceDataPath
	origSim := calibrateSimResultsPath
	origReport := calibrateReportPath
	origWarmUp := calibrateWarmUpRequests
	origRTT := calibrateNetworkRTTUs
	origBW := calibrateNetworkBandwidthMbps
	defer func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
	}()

	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1 // use header default (0)
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	// WHEN we invoke the command Run function directly
	calibrateCmd.Run(calibrateCmd, []string{})

	// THEN the report file is written
	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}

	// THEN it parses as valid JSON with matched_pairs == 3
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}
	if report.TraceInfo.MatchedPairs != 3 {
		t.Errorf("matched_pairs = %d, want 3", report.TraceInfo.MatchedPairs)
	}

	// THEN metrics are present (BC-1)
	if _, ok := report.Metrics["ttft"]; !ok {
		t.Error("report missing metrics.ttft")
	}
	if _, ok := report.Metrics["e2e"]; !ok {
		t.Error("report missing metrics.e2e")
	}

	// THEN known_limitations is non-empty (BC-1)
	if len(report.KnownLimitations) == 0 {
		t.Error("report.known_limitations should be non-empty")
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./cmd/... -run TestCalibrateCmd_BasicReport_WritesMatchedPairs -v
```
Expected: FAIL — the `Run` function currently just returns after validating flags (TODO stub).

**Step 3: Implement the Run function**

Replace the `// TODO: implement in Task 2` section in `cmd/calibrate.go:Run`:

```go
Run: func(cmd *cobra.Command, args []string) {
    if calibrateTraceHeaderPath == "" {
        logrus.Fatalf("--trace-header is required")
    }
    if calibrateTraceDataPath == "" {
        logrus.Fatalf("--trace-data is required")
    }
    if calibrateSimResultsPath == "" {
        logrus.Fatalf("--sim-results is required")
    }
    if calibrateReportPath == "" {
        logrus.Fatalf("--report is required")
    }

    // Step 1: Load TraceV2 (header + CSV data)
    trace, err := workload.LoadTraceV2(calibrateTraceHeaderPath, calibrateTraceDataPath)
    if err != nil {
        logrus.Fatalf("Failed to load TraceV2: %v", err)
    }

    // Step 2: Load SimResult JSON
    simData, err := os.ReadFile(calibrateSimResultsPath)
    if err != nil {
        logrus.Fatalf("Failed to read sim results from %s: %v", calibrateSimResultsPath, err)
    }
    var simResults []workload.SimResult
    if err := json.Unmarshal(simData, &simResults); err != nil {
        logrus.Fatalf("Failed to parse sim results JSON from %s: %v", calibrateSimResultsPath, err)
    }
    if len(simResults) == 0 {
        logrus.Fatalf("No sim results found in %s — cannot calibrate with empty data", calibrateSimResultsPath)
    }

    // Step 3: Resolve warm-up count (sentinel -1 → header fallback)
    warmUp := calibrateWarmUpRequests
    if warmUp == -1 {
        warmUp = trace.Header.WarmUpRequests
    }

    // Step 4: Resolve network RTT (sentinel -1 → header fallback)
    // Validate explicit user values: sentinel -1 means "use header"; any other negative = error (R3, BC-11)
    if calibrateNetworkRTTUs != -1 && calibrateNetworkRTTUs < 0 {
        logrus.Fatalf("--network-rtt-us must be >= 0 (or omit to use trace header), got %d", calibrateNetworkRTTUs)
    }
    var networkRTTUs int64
    if calibrateNetworkRTTUs == -1 {
        if trace.Header.Network != nil && trace.Header.Network.MeasuredRTTMs > 0 {
            networkRTTUs = int64(trace.Header.Network.MeasuredRTTMs * 1000)
        }
    } else {
        networkRTTUs = calibrateNetworkRTTUs
    }

    config := workload.CalibrationConfig{
        WarmUpRequests: warmUp,
        NetworkRTTUs:   networkRTTUs,
        BandwidthMbps:  calibrateNetworkBandwidthMbps,
    }

    // Step 5: Prepare calibration pairs
    pairs, err := workload.PrepareCalibrationPairs(trace.Records, simResults, &config)
    if err != nil {
        logrus.Fatalf("Failed to prepare calibration pairs: %v", err)
    }
    // Guard against zero matched pairs (R1: no silent data loss, BC-10)
    if pairs.MatchedCount == 0 {
        logrus.Fatalf("No matching request IDs found between trace and sim results — check that both files use the same request ID numbering")
    }

    // Step 6: Build report (empty ConfigMatchInfo — deferred, see TODO)
    // TODO: populate ConfigMatchInfo by comparing trace.Header.Server against sim config
    configMatch := workload.ConfigMatchInfo{}
    report, err := workload.BuildCalibrationReport(pairs, &configMatch)
    if err != nil {
        logrus.Fatalf("Failed to build calibration report: %v", err)
    }

    // Step 7: Write report JSON
    reportData, err := json.MarshalIndent(report, "", "  ")
    if err != nil {
        logrus.Fatalf("Failed to marshal calibration report: %v", err)
    }
    if err := os.WriteFile(calibrateReportPath, reportData, 0644); err != nil {
        logrus.Fatalf("Failed to write calibration report to %s: %v", calibrateReportPath, err)
    }

    // Step 8: Log summary to stderr
    logrus.Infof("Calibration report written to %s", calibrateReportPath)
    logrus.Infof("  Matched pairs: %d (warm-up excluded: %d, unmatched real: %d, unmatched sim: %d)",
        pairs.MatchedCount, pairs.ExcludedWarmUp, pairs.UnmatchedReal, pairs.UnmatchedSim)
    if ttft, ok := report.Metrics["ttft"]; ok {
        logrus.Infof("  TTFT: MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
            ttft.MAPE*100, ttft.PearsonR, ttft.Quality)
    }
    if e2e, ok := report.Metrics["e2e"]; ok {
        logrus.Infof("  E2E:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
            e2e.MAPE*100, e2e.PearsonR, e2e.Quality)
    }
},
```

Also add imports to the import block:
```go
import (
    "encoding/json"
    "os"

    "github.com/inference-sim/inference-sim/sim/workload"
    "github.com/sirupsen/logrus"
    "github.com/spf13/cobra"
)
```

**Step 4: Run test to verify it passes**

```bash
go test ./cmd/... -run TestCalibrateCmd_BasicReport_WritesMatchedPairs -v
```
Expected: PASS

**Step 5: Run lint**

```bash
golangci-lint run ./cmd/...
```
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/calibrate.go cmd/calibrate_test.go
git commit -m "feat(cmd): implement blis calibrate Run function (BC-1, BC-5, BC-8, BC-9, BC-10, BC-11)

- Load TraceV2 and SimResult JSON
- Resolve warmup/RTT from sentinel (-1) → header fallback
- Validate --network-rtt-us >= 0 when explicitly set (R3, BC-11)
- Guard zero matched pairs with logrus.Fatalf (R1, BC-10)
- PrepareCalibrationPairs + BuildCalibrationReport
- Write indented JSON report to --report path
- Log summary (matched pairs, MAPE, quality) to stderr
- Empty ConfigMatchInfo (deferred per issue spec option c)

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: Warm-up exclusion, RTT header-default, and negative RTT tests

**Contracts Implemented:** BC-2, BC-3, BC-4, BC-11

**Files:**
- Modify: `cmd/calibrate_test.go` (add 4 tests: BC-2, BC-3, BC-4, BC-11)

**Step 1: Write failing tests for warm-up precedence, RTT from header, and negative RTT validation**

Context: Four tests — BC-2 (header default warm-up), BC-3 (CLI flag overrides), BC-4 (RTT from header produces correct MAPE), BC-11 (negative explicit RTT causes Fatalf). Note: BC-11 uses `logrus.Fatal` which calls `os.Exit(1)` — this path is tested indirectly by inspecting that the report file is NOT written when negative RTT is passed. In test environments using `calibrateCmd.Run(...)` directly, `logrus.Fatalf` will call `os.Exit` and kill the test. Therefore BC-11 is verified by a build-check (the code path exists) rather than a runtime test. This is documented in the Test Strategy table.

```go
// In cmd/calibrate_test.go (add to existing file)

func TestCalibrateCmd_WarmUpFromHeader_ExcludesFirstN(t *testing.T) {
	// GIVEN a trace header with warm_up_requests=3 and 10 requests in data
	// AND --warmup-requests not set (sentinel -1)
	// WHEN blis calibrate is run
	// THEN report shows warm_up_excluded=3 and matched_pairs=7
	dir := t.TempDir()

	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	simPath := filepath.Join(dir, "results.json")
	reportPath := filepath.Join(dir, "report.json")

	// Header declares 3 warm-up requests
	header := `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 3
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}

	// 10 requests (IDs 0-9) with valid real timing
	var csvLines string
	csvLines = "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message\n"
	for i := 0; i < 10; i++ {
		base := int64(i) * 100000
		csvLines += fmt.Sprintf("%d,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,%d,%d,%d,%d,5,ok,\n",
			i, base, base+1000, base+5000, base+10000)
	}
	if err := os.WriteFile(dataPath, []byte(csvLines), 0644); err != nil {
		t.Fatal(err)
	}

	// SimResults for all 10 IDs
	simResults := make([]workload.SimResult, 10)
	for i := 0; i < 10; i++ {
		simResults[i] = workload.SimResult{RequestID: i, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5}
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	origHeader := calibrateTraceHeaderPath
	origData := calibrateTraceDataPath
	origSim := calibrateSimResultsPath
	origReport := calibrateReportPath
	origWarmUp := calibrateWarmUpRequests
	origRTT := calibrateNetworkRTTUs
	origBW := calibrateNetworkBandwidthMbps
	defer func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
	}()

	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1 // sentinel: use header
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}

	// BC-2: warm-up excluded comes from header
	if report.TraceInfo.WarmUpExcluded != 3 {
		t.Errorf("warm_up_excluded = %d, want 3 (from header)", report.TraceInfo.WarmUpExcluded)
	}
	if report.TraceInfo.MatchedPairs != 7 {
		t.Errorf("matched_pairs = %d, want 7", report.TraceInfo.MatchedPairs)
	}
}

func TestCalibrateCmd_WarmUpFlagOverridesHeader(t *testing.T) {
	// GIVEN a trace header with warm_up_requests=3 and 10 requests
	// AND --warmup-requests 0 explicitly set (override: include all)
	// WHEN blis calibrate is run
	// THEN report shows warm_up_excluded=0 and matched_pairs=10
	dir := t.TempDir()

	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	simPath := filepath.Join(dir, "results.json")
	reportPath := filepath.Join(dir, "report.json")

	header := `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 3
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}

	var csvLines string
	csvLines = "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message\n"
	for i := 0; i < 10; i++ {
		base := int64(i) * 100000
		csvLines += fmt.Sprintf("%d,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,%d,%d,%d,%d,5,ok,\n",
			i, base, base+1000, base+5000, base+10000)
	}
	if err := os.WriteFile(dataPath, []byte(csvLines), 0644); err != nil {
		t.Fatal(err)
	}

	simResults := make([]workload.SimResult, 10)
	for i := 0; i < 10; i++ {
		simResults[i] = workload.SimResult{RequestID: i, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5}
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	origHeader := calibrateTraceHeaderPath
	origData := calibrateTraceDataPath
	origSim := calibrateSimResultsPath
	origReport := calibrateReportPath
	origWarmUp := calibrateWarmUpRequests
	origRTT := calibrateNetworkRTTUs
	origBW := calibrateNetworkBandwidthMbps
	defer func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
	}()

	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = 0  // explicit override: include all
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}

	// BC-3: flag=0 overrides header warm_up_requests=3
	if report.TraceInfo.WarmUpExcluded != 0 {
		t.Errorf("warm_up_excluded = %d, want 0 (flag override)", report.TraceInfo.WarmUpExcluded)
	}
	if report.TraceInfo.MatchedPairs != 10 {
		t.Errorf("matched_pairs = %d, want 10", report.TraceInfo.MatchedPairs)
	}
}
```

Also add `TestCalibrateCmd_RTTFromHeader_AppliesCorrectly` to test BC-4 (RTT sentinel → header fallback). The trace header declares `network.measured_rtt_ms: 2.0`. With sim TTFT = 4000 µs and real TTFT constructed as exactly 4000 + 2000 = 6000 µs (send=1000, first_chunk=7000), we verify the report's `metrics.ttft` MAPE is 0.0 (perfect match when RTT is applied correctly):

```go
func TestCalibrateCmd_RTTFromHeader_AppliesCorrectly(t *testing.T) {
	// GIVEN a trace header with network.measured_rtt_ms=2.0 and --network-rtt-us not set
	// AND real TTFT constructed as simTTFT + 2000µs (exactly what the RTT should add)
	// WHEN blis calibrate is run
	// THEN MAPE is 0.0 (sim+RTT = real, perfect match) — verifying the header RTT was applied
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	simPath := filepath.Join(dir, "results.json")
	reportPath := filepath.Join(dir, "report.json")

	// Header with network RTT = 2ms
	header := `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
network:
  measured_rtt_ms: 2.0
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}

	// 5 requests: realTTFT = send_to_first_chunk = 6000µs (simTTFT=4000 + RTT=2000)
	// Use varied timing so Pearson r is computable (>= 3 points, non-constant)
	var csvLines string
	csvLines = "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message\n"
	// realTTFT[i] = simTTFT[i] + 2000. Use simTTFT values 3000, 4000, 5000, 6000, 7000.
	simTTFTs := []int64{3000, 4000, 5000, 6000, 7000}
	for i, st := range simTTFTs {
		send := int64(i*100000 + 1000)
		firstChunk := send + st + 2000 // real client sees simTTFT + RTT
		lastChunk := firstChunk + 5000
		csvLines += fmt.Sprintf("%d,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,%d,%d,%d,%d,5,ok,\n",
			i, int64(i)*100000, send, firstChunk, lastChunk)
	}
	if err := os.WriteFile(dataPath, []byte(csvLines), 0644); err != nil {
		t.Fatal(err)
	}

	// SimResults with varying TTFT matching the construction above
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 3000, E2E: 8000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 2, TTFT: 5000, E2E: 10000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 3, TTFT: 6000, E2E: 11000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 4, TTFT: 7000, E2E: 12000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	origHeader := calibrateTraceHeaderPath
	origData := calibrateTraceDataPath
	origSim := calibrateSimResultsPath
	origReport := calibrateReportPath
	origWarmUp := calibrateWarmUpRequests
	origRTT := calibrateNetworkRTTUs
	origBW := calibrateNetworkBandwidthMbps
	defer func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
	}()

	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1 // sentinel: use header (2ms = 2000µs)
	calibrateNetworkBandwidthMbps = 0

	calibrateCmd.Run(calibrateCmd, []string{})

	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}

	// BC-4: RTT was applied → sim TTFT + 2000µs = real TTFT → MAPE ≈ 0
	ttftMetric, ok := report.Metrics["ttft"]
	if !ok {
		t.Fatal("report missing metrics.ttft")
	}
	if ttftMetric.MAPE > 0.001 { // allow floating-point tolerance
		t.Errorf("TTFT MAPE = %.4f, want ~0.0 (RTT from header not applied)", ttftMetric.MAPE)
	}
}
```

Note: `fmt` import needed — add to existing `calibrate_test.go` import block.

**Step 2: Run tests to verify they fail**

```bash
go test ./cmd/... -run "TestCalibrateCmd_WarmUp|TestCalibrateCmd_RTT" -v
```
Expected: FAIL — `fmt` not imported yet in calibrate_test.go

**Step 3: Add `fmt` import to calibrate_test.go**

The test file import block must include `fmt`:
```go
import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "testing"

    "github.com/inference-sim/inference-sim/sim/workload"
)
```

**Step 4: Run tests to verify they pass**

```bash
go test ./cmd/... -run "TestCalibrateCmd_WarmUp|TestCalibrateCmd_RTT" -v
```
Expected: PASS (all 3 tests)

**Step 5: Run lint**

```bash
golangci-lint run ./cmd/...
```
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/calibrate_test.go
git commit -m "test(cmd): add warm-up and RTT header tests for blis calibrate (BC-2, BC-3, BC-4)

- TestCalibrateCmd_WarmUpFromHeader_ExcludesFirstN: sentinel -1 reads header
- TestCalibrateCmd_WarmUpFlagOverridesHeader: explicit 0 overrides header 3
- TestCalibrateCmd_RTTFromHeader_AppliesCorrectly: RTT sentinel reads header,
  MAPE=0 when sim+RTT=real (verifies unit conversion MeasuredRTTMs*1000)

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: Unmatched request test + CLAUDE.md update

**Contracts Implemented:** BC-5

**Files:**
- Modify: `cmd/calibrate_test.go` (add 1 test)
- Modify: `CLAUDE.md` (add calibrate.go to File Organization tree)

**Step 1: Write failing test for unmatched requests**

Context: Verify that when real trace has IDs [0,1,2] and sim results have IDs [0,1,3], the command still succeeds and reports 2 matched + 1 unmatched real + 1 unmatched sim.

```go
// In cmd/calibrate_test.go (add to existing file)

func TestCalibrateCmd_UnmatchedRequests_ReportSucceeds(t *testing.T) {
	// GIVEN a trace with IDs [0,1,2] and sim results with IDs [0,1,3]
	// WHEN blis calibrate is run
	// THEN it succeeds and the report shows 2 matched, 1 unmatched real, 1 unmatched sim
	dir := t.TempDir()

	headerPath := filepath.Join(dir, "trace.yaml")
	dataPath := filepath.Join(dir, "trace.csv")
	simPath := filepath.Join(dir, "results.json")
	reportPath := filepath.Join(dir, "report.json")

	header := `trace_version: 2
time_unit: microseconds
mode: real
warm_up_requests: 0
`
	if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
		t.Fatal(err)
	}

	// Real trace has IDs 0, 1, 2
	csvData := "request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message\n" +
		"0,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,0,1000,5000,10000,5,ok,\n" +
		"1,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,100000,101000,105000,110000,5,ok,\n" +
		"2,c1,t1,standard,s1,0,,true,10,5,10,0,0,0,0.0,,0,10,200000,201000,205000,210000,5,ok,\n"
	if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
		t.Fatal(err)
	}

	// Sim results have IDs 0, 1, 3 (ID 2 missing, extra ID 3)
	simResults := []workload.SimResult{
		{RequestID: 0, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 1, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
		{RequestID: 3, TTFT: 4000, E2E: 9000, InputTokens: 10, OutputTokens: 5},
	}
	simData, _ := json.Marshal(simResults)
	if err := os.WriteFile(simPath, simData, 0644); err != nil {
		t.Fatal(err)
	}

	origHeader := calibrateTraceHeaderPath
	origData := calibrateTraceDataPath
	origSim := calibrateSimResultsPath
	origReport := calibrateReportPath
	origWarmUp := calibrateWarmUpRequests
	origRTT := calibrateNetworkRTTUs
	origBW := calibrateNetworkBandwidthMbps
	defer func() {
		calibrateTraceHeaderPath = origHeader
		calibrateTraceDataPath = origData
		calibrateSimResultsPath = origSim
		calibrateReportPath = origReport
		calibrateWarmUpRequests = origWarmUp
		calibrateNetworkRTTUs = origRTT
		calibrateNetworkBandwidthMbps = origBW
	}()

	calibrateTraceHeaderPath = headerPath
	calibrateTraceDataPath = dataPath
	calibrateSimResultsPath = simPath
	calibrateReportPath = reportPath
	calibrateWarmUpRequests = -1
	calibrateNetworkRTTUs = -1
	calibrateNetworkBandwidthMbps = 0

	// WHEN the command is invoked (should not panic or logrus.Fatal)
	calibrateCmd.Run(calibrateCmd, []string{})

	// THEN the report is written and valid
	data, err := os.ReadFile(reportPath)
	if err != nil {
		t.Fatalf("report not written: %v", err)
	}
	var report workload.CalibrationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("report is not valid JSON: %v", err)
	}

	// THEN 2 matched pairs (IDs 0 and 1)
	if report.TraceInfo.MatchedPairs != 2 {
		t.Errorf("matched_pairs = %d, want 2", report.TraceInfo.MatchedPairs)
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./cmd/... -run TestCalibrateCmd_UnmatchedRequests_ReportSucceeds -v
```
Expected: PASS (this test should already pass after Task 2 — the library handles unmatched gracefully). If it PASSes, record that and proceed. If it FAILs, investigate.

**Step 3: Update CLAUDE.md File Organization**

In `CLAUDE.md`, add `calibrate.go` entry to the `cmd/` tree, after the `replay.go` entry:

Find:
```
│   ├── replay.go              # `blis replay` command: replays TraceV2 file through DES;
```

Add after it:
```
│   ├── calibrate.go           # `blis calibrate` command: compares real observed latencies (TraceV2) against sim predictions ([]SimResult JSON from blis replay); flags: --trace-header, --trace-data, --sim-results, --report (required), --warmup-requests (default: from header), --network-rtt-us (default: from header), --network-bandwidth-mbps; writes CalibrationReport JSON
```

**Step 4: Run all tests to verify no regressions**

```bash
go test ./... 2>&1
```
Expected: All packages PASS

**Step 5: Run lint**

```bash
golangci-lint run ./...
```
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/calibrate_test.go CLAUDE.md
git commit -m "test(cmd): add unmatched-request test and update CLAUDE.md (BC-5)

- TestCalibrateCmd_UnmatchedRequests_ReportSucceeds: verifies command succeeds
  with partial match (IDs [0,1,2] real vs [0,1,3] sim → 2 matched)
- Update CLAUDE.md cmd/ tree with calibrate.go entry

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 (basic report) | Task 2 | Integration | `TestCalibrateCmd_BasicReport_WritesMatchedPairs` |
| BC-2 (warmup from header) | Task 3 | Unit | `TestCalibrateCmd_WarmUpFromHeader_ExcludesFirstN` |
| BC-3 (warmup flag overrides header) | Task 3 | Unit | `TestCalibrateCmd_WarmUpFlagOverridesHeader` |
| BC-4 (RTT from header) | Task 3 | Integration | `TestCalibrateCmd_RTTFromHeader_AppliesCorrectly` — MAPE=0 when sim+RTT=real verifies header conversion |
| BC-5 (unmatched → success) | Task 4 | Integration | `TestCalibrateCmd_UnmatchedRequests_ReportSucceeds` |
| BC-6 (flags registered) | Task 1 | Unit | `TestCalibrateCmd_Flags_Registered` |
| BC-7 (missing required flags) | Task 1 | Implicit | Code path present and verifiable via `--help`; logrus.Fatalf paths require exit-handler override |
| BC-8, BC-9 (file errors) | Task 2 | Implicit | `logrus.Fatalf` paths; tested by inspecting code path completeness |
| BC-10 (zero matched pairs) | Task 2 | Implicit | Guard present in Run function; logrus.Fatalf path |
| BC-11 (negative RTT) | Task 3 | Implicit | Validation present in Run function; logrus.Fatalf path not runtime-testable without exit override |

**Golden dataset:** No changes. This PR adds no new `sim/` output metrics.

**Shared test infrastructure:** No new helpers needed. Tests use `t.TempDir()`, `os.WriteFile`, and direct `json.Marshal` (standard library, no shared infrastructure required).

**Invariant tests:** Not applicable. This PR is a CLI I/O layer with no simulation state. The library (`calibrate.go`) has its own contract (MAPE formula, Pearson r). Tests verify the CLI correctly wires library inputs/outputs — the behavioral contracts are the contracts.

---

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| User explicitly passes `--network-rtt-us -1` (the sentinel) | Low | Low | `-1` explicitly = "use header default" — documented behavior, not a bug | Task 2 |
| User passes `--network-rtt-us -5000` (negative, not sentinel) | Low | Medium | Explicit validation: `calibrateNetworkRTTUs != -1 && < 0 → logrus.Fatalf` (BC-11) | Task 2 |
| `MeasuredRTTMs * 1000` overflow for large RTT values | Very Low | Low | RTT values in practice are 0-100ms; 100ms → 100,000 µs well within int64 | Task 2 |
| Zero matched pairs → silent empty report | Low | Medium | Guard added: `pairs.MatchedCount == 0 → logrus.Fatalf` (BC-10, R1) | Task 2 |
| Race condition in tests (package-level flag vars) | Medium | Medium | Each test saves/restores all flag vars (same pattern as replay_test.go); no `t.Parallel()` | Tasks 2-4 |

---

## PART 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — pure I/O glue, no new types or interfaces
- [x] No feature creep — exactly the 7 flags and 4 tests specified in issue #658
- [x] No unexercised flags — all 7 flags read in `Run` (4 required + 3 optional)
- [x] No partial implementations — `Run` is fully implemented in Task 2
- [x] No breaking changes — new command only
- [x] No hidden global state — only adds 7 package-level flag vars in `cmd/`
- [x] All new code will pass golangci-lint — verified by lint step in each task
- [x] Shared test helpers used — `t.TempDir()` and stdlib; no duplication
- [x] CLAUDE.md updated — Task 4 adds `calibrate.go` to File Organization
- [x] No stale references left in CLAUDE.md
- [x] Documentation DRY — no canonical source files modified
- [x] Deviation log reviewed — 4 deviations, all justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered — Task 1 (skeleton) → Task 2 (Run) → Task 3 (warm-up tests) → Task 4 (unmatched test + docs)
- [x] All contracts mapped to tasks — see Test Strategy table
- [x] No golden dataset changes needed
- [x] Construction site audit — no new structs added; `ConfigMatchInfo{}` constructed in 1 place in `calibrate.go`, which is in the library (not this PR)
- [x] Not part of a macro plan (standalone issue)

**Antipattern rules:**
- [x] R1: No silent errors — all error paths call `logrus.Fatalf`
- [x] R2: N/A — no map iteration over floats
- [x] R3: `--warmup-requests` validated (sentinel `-1` or ≥0; negative non-sentinel silently treated as 0 by library — physically equivalent); `--network-rtt-us` validated (sentinel `-1` or ≥0; negative explicit values rejected with Fatalf, BC-11); `--network-bandwidth-mbps`: negative values silently disabled by library (≤0 → 0); physically meaningless and result is correct
- [x] R4: No new struct fields
- [x] R5: N/A — no resource allocation loops
- [x] R6: No `logrus.Fatalf` in `sim/` — only in `cmd/`
- [x] R7: N/A — no golden tests added
- [x] R8: No exported maps
- [x] R9: N/A — no YAML config fields
- [x] R10: N/A — no new YAML parsing
- [x] R11: N/A — no runtime-derived division
- [x] R12: N/A — no golden dataset changes
- [x] R13: N/A — no new interfaces
- [x] R14: `Run` function is single-concern (I/O glue only)
- [x] R15: No stale PR references in new code
- [x] R16: N/A — no new config structs
- [x] R17-R23: N/A — no routing, DES, or simulation logic

---

## APPENDIX: File-Level Implementation Details

### File: `cmd/calibrate.go`

**Purpose:** Cobra command implementing `blis calibrate`. Pure I/O glue: load TraceV2 + SimResult JSON, call library functions, write JSON report to disk.

**Complete Implementation:**

```go
package cmd

import (
	"encoding/json"
	"os"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	calibrateTraceHeaderPath      string
	calibrateTraceDataPath        string
	calibrateSimResultsPath       string
	calibrateReportPath           string
	calibrateWarmUpRequests       int
	calibrateNetworkRTTUs         int64
	calibrateNetworkBandwidthMbps float64
)

var calibrateCmd = &cobra.Command{
	Use:   "calibrate",
	Short: "Compare real observed latencies against simulator predictions",
	Long: `Calibrate takes a TraceV2 file (from blis observe) and a SimResult JSON file
(from blis replay --results-path) and computes a calibration report comparing
real vs simulated TTFT and E2E latencies.

The report includes per-metric MAPE, Pearson r, percentile comparison, bias
direction, and a quality rating. Use --report to specify the output path.

Warm-up requests are excluded from comparison. By default, the warm-up count
is taken from the trace header (warm_up_requests field). Use --warmup-requests
to override. Pass --warmup-requests 0 to include all requests.

Network RTT and bandwidth adjustments shift sim-side latencies to client
perspective. By default, RTT is taken from the trace header
(network.measured_rtt_ms). Use --network-rtt-us to override in microseconds.

Example:
  blis calibrate --trace-header t.yaml --trace-data d.csv \
    --sim-results results.json --report calibration.json`,
	Run: func(cmd *cobra.Command, args []string) {
		if calibrateTraceHeaderPath == "" {
			logrus.Fatalf("--trace-header is required")
		}
		if calibrateTraceDataPath == "" {
			logrus.Fatalf("--trace-data is required")
		}
		if calibrateSimResultsPath == "" {
			logrus.Fatalf("--sim-results is required")
		}
		if calibrateReportPath == "" {
			logrus.Fatalf("--report is required")
		}

		// Step 1: Load TraceV2 (header + CSV data)
		trace, err := workload.LoadTraceV2(calibrateTraceHeaderPath, calibrateTraceDataPath)
		if err != nil {
			logrus.Fatalf("Failed to load TraceV2: %v", err)
		}

		// Step 2: Load SimResult JSON
		simData, err := os.ReadFile(calibrateSimResultsPath)
		if err != nil {
			logrus.Fatalf("Failed to read sim results from %s: %v", calibrateSimResultsPath, err)
		}
		var simResults []workload.SimResult
		if err := json.Unmarshal(simData, &simResults); err != nil {
			logrus.Fatalf("Failed to parse sim results JSON from %s: %v", calibrateSimResultsPath, err)
		}
		if len(simResults) == 0 {
			logrus.Fatalf("No sim results found in %s — cannot calibrate with empty data", calibrateSimResultsPath)
		}

		// Step 3: Resolve warm-up count (sentinel -1 → header fallback)
		warmUp := calibrateWarmUpRequests
		if warmUp == -1 {
			warmUp = trace.Header.WarmUpRequests
		}

		// Step 4: Resolve network RTT (sentinel -1 → header fallback)
		// Reject explicit negative values (not the sentinel) — R3, BC-11
		if calibrateNetworkRTTUs != -1 && calibrateNetworkRTTUs < 0 {
			logrus.Fatalf("--network-rtt-us must be >= 0 (or omit to use trace header), got %d", calibrateNetworkRTTUs)
		}
		var networkRTTUs int64
		if calibrateNetworkRTTUs == -1 {
			if trace.Header.Network != nil && trace.Header.Network.MeasuredRTTMs > 0 {
				networkRTTUs = int64(trace.Header.Network.MeasuredRTTMs * 1000)
			}
		} else {
			networkRTTUs = calibrateNetworkRTTUs
		}

		config := workload.CalibrationConfig{
			WarmUpRequests: warmUp,
			NetworkRTTUs:   networkRTTUs,
			BandwidthMbps:  calibrateNetworkBandwidthMbps,
		}

		// Step 5: Prepare calibration pairs
		pairs, err := workload.PrepareCalibrationPairs(trace.Records, simResults, &config)
		if err != nil {
			logrus.Fatalf("Failed to prepare calibration pairs: %v", err)
		}
		// Guard against zero matched pairs (R1: no silent data loss, BC-10)
		if pairs.MatchedCount == 0 {
			logrus.Fatalf("No matching request IDs found between trace and sim results — check that both files use the same request ID numbering")
		}

		// Step 6: Build report (empty ConfigMatchInfo — see TODO)
		// TODO: populate ConfigMatchInfo by comparing trace.Header.Server against sim config (#658)
		configMatch := workload.ConfigMatchInfo{}
		report, err := workload.BuildCalibrationReport(pairs, &configMatch)
		if err != nil {
			logrus.Fatalf("Failed to build calibration report: %v", err)
		}

		// Step 7: Write report JSON
		reportData, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			logrus.Fatalf("Failed to marshal calibration report: %v", err)
		}
		if err := os.WriteFile(calibrateReportPath, reportData, 0644); err != nil {
			logrus.Fatalf("Failed to write calibration report to %s: %v", calibrateReportPath, err)
		}

		// Step 8: Log summary to stderr
		logrus.Infof("Calibration report written to %s", calibrateReportPath)
		logrus.Infof("  Matched pairs: %d (warm-up excluded: %d, unmatched real: %d, unmatched sim: %d)",
			pairs.MatchedCount, pairs.ExcludedWarmUp, pairs.UnmatchedReal, pairs.UnmatchedSim)
		if ttft, ok := report.Metrics["ttft"]; ok {
			logrus.Infof("  TTFT: MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
				ttft.MAPE*100, ttft.PearsonR, ttft.Quality)
		}
		if e2e, ok := report.Metrics["e2e"]; ok {
			logrus.Infof("  E2E:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
				e2e.MAPE*100, e2e.PearsonR, e2e.Quality)
		}
	},
}

func init() {
	calibrateCmd.Flags().StringVar(&calibrateTraceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateTraceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateSimResultsPath, "sim-results", "", "Path to SimResult JSON file (from blis replay --results-path; required)")
	calibrateCmd.Flags().StringVar(&calibrateReportPath, "report", "", "Path to write calibration report JSON (required)")
	calibrateCmd.Flags().IntVar(&calibrateWarmUpRequests, "warmup-requests", -1, "Number of initial requests to exclude (default: from trace header warm_up_requests; pass 0 to include all)")
	calibrateCmd.Flags().Int64Var(&calibrateNetworkRTTUs, "network-rtt-us", -1, "Network RTT in microseconds added to sim-side latencies (default: from trace header network.measured_rtt_ms)")
	calibrateCmd.Flags().Float64Var(&calibrateNetworkBandwidthMbps, "network-bandwidth-mbps", 0, "Network bandwidth in Mbps for upload/download delay calculation (default: 0 = no delay)")
	rootCmd.AddCommand(calibrateCmd)
}
```

**Key implementation notes:**
- `os.ReadFile` (stdlib) for reading the sim results file — consistent with `replay.go` pattern
- `json.MarshalIndent(report, "", "  ")` for human-readable output
- `logrus.Infof` for stderr summary — consistent with project output separation (stdout = results, stderr = diagnostics)
- Sentinel `-1` for `warmup-requests` and `network-rtt-us` uses two separate code paths (not shared helper) to keep the logic transparent

---

### File: `cmd/calibrate_test.go`

**Purpose:** 6 behavioral tests exercising the core contracts of the `blis calibrate` command. All tests use `t.TempDir()` + synthetic fixture files and save/restore package-level flag vars.

**Complete test file:** See Task 1-4 code blocks above. The complete assembled file has these tests:
1. `TestCalibrateCmd_Flags_Registered` (Task 1)
2. `TestCalibrateCmd_BasicReport_WritesMatchedPairs` (Task 2)
3. `TestCalibrateCmd_WarmUpFromHeader_ExcludesFirstN` (Task 3)
4. `TestCalibrateCmd_WarmUpFlagOverridesHeader` (Task 3)
5. `TestCalibrateCmd_RTTFromHeader_AppliesCorrectly` (Task 3)
6. `TestCalibrateCmd_UnmatchedRequests_ReportSucceeds` (Task 4)

**Import block:**
```go
import (
    "encoding/json"
    "fmt"
    "os"
    "path/filepath"
    "testing"

    "github.com/inference-sim/inference-sim/sim/workload"
)
```

**Note on `fmt` import:** Only used in Tasks 3 and 4 (the loop-based CSV generation). If the linter complains about unused `fmt` after Task 1 (before Tasks 3-4 add it), add a `_ = fmt.Sprintf` placeholder or delay adding the import until it's used.
