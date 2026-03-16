# blis replay Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `blis replay` — a CLI command that replays a TraceV2 file through the DES and writes per-request `SimResult` JSON for downstream use by `blis calibrate`.

**The problem today:** `blis run` generates synthetic workloads; there is no way to run the DES against a captured real-traffic trace and produce per-request TTFT/E2E predictions. The library code (`LoadTraceV2`, `LoadTraceV2Requests`) already exists and is tested — only the CLI wiring is missing.

**What this PR adds:**
1. `blis replay --trace-header t.yaml --trace-data d.csv [sim-config flags]` — replays the exact request sequence from a TraceV2 file through the DES.
2. JSON tags added to the existing `workload.SimResult` type (in `sim/workload/calibrate.go`): integer `request_id`, `ttft_us`/`e2e_us` in microseconds, `input_tokens`, `output_tokens`.
3. `registerSimConfigFlags(cmd)` helper in `cmd/root.go` that eliminates ~50-line flag duplication between `runCmd` and `replayCmd`.
4. `--results-path` writes a JSON array of `workload.SimResult` objects instead of the `MetricsOutput` schema (run-compatible output unchanged).

**Why this matters:** `blis replay` is the missing bridge in the observe → replay → calibrate loop (#652). It enables coefficient validation: capture real traffic, replay through BLIS with candidate coefficients, compare predicted vs observed TTFT/E2E.

**Architecture:** `cmd/replay.go` registers `replayCmd` using the same package-level flag vars as `runCmd`. The new `registerSimConfigFlags(cmd)` helper in `root.go` registers all sim-engine flags for either command. The replay Run function duplicates the config resolution path from `runCmd.Run` (extracting 400 lines is too invasive for this PR) and replaces workload generation with `LoadTraceV2` + `LoadTraceV2Requests`. The new `extractSimResults` function converts `Metrics.RequestTTFTs`/`RequestE2Es` (in ticks = µs) directly to `SimResult`.

**Source:** GitHub issue #657. Parent: #652.

**Closes:** Fixes #657

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

**Building block added:** New CLI subcommand in `cmd/`, wired to existing `workload` and `cluster` library code.

**Adjacent components:**
- `sim/workload.LoadTraceV2`, `LoadTraceV2Requests` — existing library, unchanged
- `sim/cluster.ClusterSimulator` — same construction/run path as `blis run`
- `sim.Metrics.RequestTTFTs`, `RequestE2Es` — read-only consumers of existing maps

**Invariants touched:** INV-1 (request conservation), INV-6 (determinism via sorted SimResult output).

**Construction Site Audit:**
- `SimResult` is a new type with no existing construction sites — no audit needed.
- `registerSimConfigFlags` extracts existing flag registrations from `init()` — no struct construction.

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds `blis replay` to the CLI. The command loads a TraceV2 file (header YAML + data CSV) using existing library code, converts records to `sim.Request` objects (with synthetic token IDs), resolves simulation configuration via the same chain as `blis run` (defaults.yaml → HF config → CLI overrides), runs the ClusterSimulator, and writes per-request `SimResult` JSON to `--results-path`. The `SimResult` schema differs from `MetricsOutput`: integer request IDs and TTFT/E2E in microseconds (not ms) for calibration precision.

To avoid duplicating ~50 lines of flag registration, a `registerSimConfigFlags(cmd)` helper is extracted from `init()` in `root.go`. Config resolution (~400 lines) is duplicated rather than extracted (extraction requires threading `*cobra.Command` through multiple sub-functions and restructuring the entire `runCmd.Run` body — too invasive for this PR scope).

Fits cleanly in `cmd/` with no changes to `sim/`, `sim/cluster/`, or `sim/workload/`.

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Request sequence fidelity
- GIVEN a TraceV2 file with N records
- WHEN blis replay is run
- THEN the simulation receives exactly N requests with the same token counts and arrival times as the trace

BC-2: SimResult per-request output
- GIVEN blis replay completes with --results-path set
- WHEN the output file is read
- THEN it is a JSON array where each entry has an integer request_id, ttft_us (microseconds), e2e_us (microseconds), input_tokens, and output_tokens

BC-3: Horizon auto-computation
- GIVEN a TraceV2 file where the latest arrival is at time T
- WHEN blis replay is run without --horizon
- THEN the simulation horizon is max(arrivals) * 2 (≥ T, generous buffer)
- MECHANISM: 2× the latest arrival time; enough time for the last request to complete in typical loads

BC-4: Sim config flags accepted
- GIVEN any sim-config flag (--model, --total-kv-blocks, --latency-model, --routing-policy, etc.)
- WHEN passed to blis replay
- THEN the simulation uses that value (verified: --help shows the flag, defaults are registered)

BC-5: SimResult output is deterministic (INV-6)
- GIVEN the same trace file and same --seed across two runs
- WHEN SimResult JSON is written
- THEN the byte-identical JSON array is produced (results sorted by request_id)
```

**Negative contracts:**

```
BC-6: Missing trace files fail fast
- GIVEN --trace-header or --trace-data points to a non-existent path
- WHEN blis replay is run
- THEN the command exits immediately with logrus.Fatalf and a user-readable error message

BC-7: Non-numeric request IDs excluded from SimResult
- GIVEN a trace containing session follow-up requests with non-numeric IDs
- WHEN SimResult is extracted
- THEN those entries are silently excluded (they have no TraceRecord.RequestID counterpart)
```

**Error-handling contracts:**

```
BC-8: --model required
- GIVEN blis replay invoked without --model
- WHEN the Run function executes
- THEN the command exits with logrus.Fatalf "LLM name not provided"
```

### C) Component Interaction

```
cmd/replay.go (new)
  │
  ├─► workload.LoadTraceV2(headerPath, dataPath)
  │     └── returns *TraceV2{Header, Records}
  │
  ├─► workload.LoadTraceV2Requests(trace, seed)
  │     └── returns []*sim.Request (synthetic token IDs)
  │
  ├─► [config resolution chain — same as runCmd.Run]
  │     ├── GetDefaultSpecs(model) → hw/tp defaults
  │     ├── resolveModelConfig / resolveHardwareConfig
  │     └── latency.ParseHFConfig / GetModelConfigFromHF
  │
  ├─► cluster.NewClusterSimulator(config, requests, nil)
  │     └── onRequestDone = nil (no session manager)
  │
  ├─► cs.Run() → fills cs.AggregatedMetrics()
  │
  ├─► cs.AggregatedMetrics().SaveResults(...)
  │     └── prints MetricsOutput JSON to stdout (unchanged)
  │
  └─► extractSimResults(metrics)
        └── writes []SimResult to --results-path

cmd/root.go (modified)
  └─► registerSimConfigFlags(cmd *cobra.Command) [new helper]
        ├── called by runCmd's init()
        └── called by replayCmd's init()
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| "Extract config-building into a shared function, or duplicate" | Duplicates config resolution (~300 lines) | Extraction is too invasive: requires threading `*cobra.Command` through helper, restructuring all `cmd.Flags().Changed()` calls, and touching ~400 lines of existing code. Duplication is explicitly approved in issue. R23 enforced by copying the full validation block from runCmd verbatim. |
| "SimResult.TTFT: RequestMetrics.FirstTokenTime" | Uses `Metrics.RequestTTFTs[reqID]` directly | `RequestMetrics` has no `FirstTokenTime` field — that's on `sim.Request`. `Metrics.RequestTTFTs` stores the same value (float64 in ticks/µs) set at simulator.go:571. |
| "cmd/replay.go defines SimResult type" | Uses existing `workload.SimResult` from `sim/workload/calibrate.go` + adds JSON tags | `workload.SimResult` already exists with identical fields (R4 audit). Defining a duplicate in `cmd/` would be an architectural violation. JSON tags are additive and backward-compatible. |
| "blis calibrate consumes SimResult" | No calibrate implementation | #658 is a separate issue. Only the SimResult type and output format are implemented here. |
| (implied) horizon = 2× max arrival | `maxArrival == 0` (all-zero arrivals) gets 600s buffer, not MaxInt64 | MaxInt64 horizon for a valid (non-empty) trace would cause the simulation to run indefinitely. Fixed buffer of 600s covers all practical workloads at t=0. Overflow guard added for `maxArrival > MaxInt64/2`. |
| Warm-up requests in trace | SimResult includes all N records; warm-up filtering is caller's responsibility | `trace.Header.WarmUpRequests` documents how many leading records are warm-up. Filtering in replay would require `blis replay` to know the calibration use case — separation of concerns favors passing WarmUpRequests count to `blis calibrate` (#658) via documentation. The plan documents this limitation. |
| Per-record Model field in TraceRecord | Ignored; single `--model` flag applies to all requests | Multi-model traces are not supported in this PR. `TraceRecord.Model` field is set on `sim.Request.Model` by `LoadTraceV2Requests` (already implemented), so the routing layer sees it. But the latency model configuration (`--model` flag) is global. Single-model replay is the intended use case for calibration. |

### E) Review Guide

**Tricky part:** The config resolution duplication (Task 4) is ~250 lines. Verify it matches root.go's logic exactly for: latency model selection, coefficient loading from defaults.yaml, roofline/crossmodel HF config resolution, and `cmd.Flags().Changed()` precedence for each flag. A subtle divergence here would cause silent misconfigurations.

**Scrutinize:** `extractSimResults` (Task 5) — ensure TTFT/E2E are in microseconds (from `RequestTTFTs`/`RequestE2Es`, NOT from `Requests[id].TTFT` which is in ms). Verify sort by RequestID for determinism (R2).

**Safe to skim:** `registerSimConfigFlags` helper (Task 2) — mechanical extraction of existing flag registrations.

**Known limitations (document in --help long description):**
- Warm-up requests: `trace.Header.WarmUpRequests` records the count of leading requests that experienced cold KV caches. SimResult includes ALL records (indices 0..N-1); `blis calibrate` (#658) is responsible for skipping the first `WarmUpRequests` entries.
- Multi-model traces: per-request `Model` field is propagated to `sim.Request.Model` but the global `--model` flag governs the latency model configuration. Traces with mixed models are not supported for calibration.
- Horizon sufficiency: `2× max_arrival` may truncate late-completing requests under heavy load (ρ ≥ 1). Use `--horizon` explicitly for saturated traces or traces where max(arrival) is much smaller than total simulation needed. Monitor `still_queued`/`still_running` in aggregate MetricsOutput to detect truncation.

**Known debt:** Config resolution duplication. A future PR could extract `buildDeploymentConfig(*cobra.Command)` when multiple commands need it.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Change |
|------|--------|
| `sim/workload/calibrate.go` (modify) | Add JSON tags to existing `SimResult` type (no new fields) |
| `cmd/replay.go` (create) | `replayCmd`, `replayCmd.init()`, `extractSimResults()` — uses `workload.SimResult` |
| `cmd/root.go` (modify) | Extract `registerSimConfigFlags(cmd)` from `init()` |
| `cmd/replay_test.go` (create) | Unit + integration tests |

No dead code: `workload.SimResult` already used by `PrepareCalibrationPairs`; `replayCmd` is registered via `rootCmd.AddCommand`. No new interfaces.

**Construction site audit for `workload.SimResult`:** Adding JSON tags to existing type. All construction sites:
- `sim/workload/calibrate_test.go` (if any) — verify no literal `SimResult{}` constructions break
- `cmd/replay.go` (new) — `extractSimResults` creates `workload.SimResult` values
No other construction sites found. JSON tags are additive (backward compatible).

### G) Task Breakdown

---

#### Task 1: Extract `registerSimConfigFlags` helper in root.go

**Contracts:** BC-4

**Files:**
- Modify: `cmd/root.go`

**Step 1: Write the failing test**

```go
// cmd/replay_test.go
package cmd

import "testing"

// TestReplayCmd_SimConfigFlags_Registered verifies BC-4:
// all sim config flags registered on replayCmd.
func TestReplayCmd_SimConfigFlags_Registered(t *testing.T) {
    flags := []string{
        "seed", "log", "defaults-filepath", "model", "hardware", "tp",
        "latency-model", "total-kv-blocks", "block-size-in-tokens",
        "max-num-running-reqs", "max-num-scheduled-tokens",
        "beta-coeffs", "alpha-coeffs", "num-instances",
        "routing-policy", "scheduler", "priority-policy",
        "results-path", "kv-cpu-blocks", "snapshot-refresh-interval",
        "horizon", "trace-header", "trace-data",
    }
    for _, name := range flags {
        f := replayCmd.Flags().Lookup(name)
        if f == nil {
            t.Errorf("replayCmd missing flag --%s", name)
        }
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestReplayCmd_SimConfigFlags_Registered -v
```
Expected: `FAIL` — `replayCmd` not defined yet.

**Step 3: Extract `registerSimConfigFlags` in root.go**

Add after line 193 (after `allZeros`):

```go
// registerSimConfigFlags registers all simulation-engine configuration flags
// on the given command. Called by both runCmd and replayCmd to avoid
// duplicating ~50 flag registrations.
func registerSimConfigFlags(cmd *cobra.Command) {
    cmd.Flags().Int64Var(&seed, "seed", 42, "Seed for random token ID generation")
    cmd.Flags().Int64Var(&simulationHorizon, "horizon", math.MaxInt64, "Total simulation horizon (in ticks)")
    cmd.Flags().StringVar(&logLevel, "log", "warn", "Log level for diagnostic messages (trace, debug, info, warn, error, fatal, panic). Simulation results always print to stdout regardless of this setting.")
    cmd.Flags().StringVar(&defaultsFilePath, "defaults-filepath", "defaults.yaml", "Path to default constants - trained coefficients, default specs and workloads")
    cmd.Flags().StringVar(&modelConfigFolder, "model-config-folder", "", "Path to folder containing config.json")
    cmd.Flags().StringVar(&hwConfigPath, "hardware-config", "", "Path to file containing hardware config")
    cmd.Flags().Int64Var(&totalKVBlocks, "total-kv-blocks", 1000000, "Total number of KV cache blocks")
    cmd.Flags().Int64Var(&maxRunningReqs, "max-num-running-reqs", 256, "Maximum number of requests running together")
    cmd.Flags().Int64Var(&maxScheduledTokens, "max-num-scheduled-tokens", 2048, "Maximum total number of new tokens across running requests")
    cmd.Flags().Float64SliceVar(&betaCoeffs, "beta-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated list of beta coefficients")
    cmd.Flags().Float64SliceVar(&alphaCoeffs, "alpha-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated alpha coefficients for processing delays")
    cmd.Flags().Int64Var(&blockSizeTokens, "block-size-in-tokens", 16, "Number of tokens contained in a KV cache block")
    cmd.Flags().Int64Var(&longPrefillTokenThreshold, "long-prefill-token-threshold", 0, "Max length of prefill beyond which chunked prefill is triggered")
    cmd.Flags().StringVar(&model, "model", "", "LLM name")
    cmd.Flags().StringVar(&gpu, "hardware", "", "GPU type")
    cmd.Flags().IntVar(&tensorParallelism, "tp", 0, "Tensor parallelism")
    cmd.Flags().StringVar(&vllmVersion, "vllm-version", "", "vLLM version")
    cmd.Flags().StringVar(&latencyModelBackend, "latency-model", "roofline", "Latency model backend: roofline (default), blackbox, crossmodel, trained-roofline")
    cmd.Flags().Int64Var(&maxModelLen, "max-model-len", 0, "Max total sequence length (input + output); 0 = unlimited.")
    cmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")
    cmd.Flags().StringVar(&admissionPolicy, "admission-policy", "always-admit", "Admission policy: always-admit, token-bucket, reject-all")
    cmd.Flags().Int64Var(&admissionLatency, "admission-latency", 0, "Admission latency in microseconds")
    cmd.Flags().Int64Var(&routingLatency, "routing-latency", 0, "Routing latency in microseconds")
    cmd.Flags().Float64Var(&tokenBucketCapacity, "token-bucket-capacity", 10000, "Token bucket capacity")
    cmd.Flags().Float64Var(&tokenBucketRefillRate, "token-bucket-refill-rate", 1000, "Token bucket refill rate (tokens/second)")
    cmd.Flags().StringVar(&routingPolicy, "routing-policy", "round-robin", "Routing policy: round-robin, least-loaded, weighted, always-busiest")
    cmd.Flags().StringVar(&routingScorers, "routing-scorers", "", "Scorer weights for weighted routing (e.g., queue-depth:2,kv-utilization:2)")
    cmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based, inverted-slo")
    cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")
    cmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")
    cmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs")
    cmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
    cmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Number of counterfactual candidates per routing decision")
    cmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")
    cmd.Flags().Int64Var(&kvCPUBlocks, "kv-cpu-blocks", 0, "CPU tier KV cache blocks (0 = disabled)")
    cmd.Flags().Float64Var(&kvOffloadThreshold, "kv-offload-threshold", 0.9, "GPU utilization above which blocks are offloaded to CPU")
    cmd.Flags().Float64Var(&kvTransferBandwidth, "kv-transfer-bandwidth", 100.0, "CPU↔GPU transfer rate in blocks per tick")
    cmd.Flags().Int64Var(&kvTransferBaseLatency, "kv-transfer-base-latency", 0, "Fixed per-transfer latency in ticks")
    cmd.Flags().Int64Var(&snapshotRefreshInterval, "snapshot-refresh-interval", 0, "Snapshot refresh interval in microseconds (0 = immediate)")
    cmd.Flags().StringVar(&resultsPath, "results-path", "", "File to write results to")
}
```

Then replace the flag registration block in `init()` (lines ~1173–1254) with a call to `registerSimConfigFlags(runCmd)` plus the workload-specific and run-specific flags that stay on `runCmd` only:

```go
func init() {
    registerSimConfigFlags(runCmd)

    // Workload generation flags (run-only)
    runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution)")
    runCmd.Flags().Float64Var(&rate, "rate", 1.0, "Requests arrival per second")
    runCmd.Flags().IntVar(&numRequests, "num-requests", 100, "Number of requests to generate")
    runCmd.Flags().IntVar(&prefixTokens, "prefix-tokens", 0, "Prefix Token Count")
    runCmd.Flags().IntVar(&promptTokensMean, "prompt-tokens", 512, "Average Prompt Token Count")
    runCmd.Flags().IntVar(&promptTokensStdev, "prompt-tokens-stdev", 256, "Stddev Prompt Token Count")
    runCmd.Flags().IntVar(&promptTokensMin, "prompt-tokens-min", 2, "Min Prompt Token Count")
    runCmd.Flags().IntVar(&promptTokensMax, "prompt-tokens-max", 7000, "Max Prompt Token Count")
    runCmd.Flags().IntVar(&outputTokensMean, "output-tokens", 512, "Average Output Token Count")
    runCmd.Flags().IntVar(&outputTokensStdev, "output-tokens-stdev", 256, "Stddev Output Token Count")
    runCmd.Flags().IntVar(&outputTokensMin, "output-tokens-min", 2, "Min Output Token Count")
    runCmd.Flags().IntVar(&outputTokensMax, "output-tokens-max", 7000, "Max Output Token Count")
    runCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "Path to YAML workload specification file")

    // Run-specific export
    runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")

    rootCmd.AddCommand(runCmd)
}
```

**Step 4: Create `cmd/replay.go` skeleton** (needed to compile the test)

```go
package cmd

import (
    "github.com/spf13/cobra"
)

var (
    traceHeaderPath string
    traceDataPath   string
)

var replayCmd = &cobra.Command{
    Use:   "replay",
    Short: "Replay a TraceV2 file through the discrete-event simulator",
    Run: func(cmd *cobra.Command, args []string) {
        // TODO: implement
    },
}

func init() {
    registerSimConfigFlags(replayCmd)
    replayCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (required)")
    replayCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (required)")
    rootCmd.AddCommand(replayCmd)
}
```

**Step 5: Run test to verify it passes**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestReplayCmd_SimConfigFlags_Registered -v
```
Expected: `PASS`

**Step 6: Run lint**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./cmd/...
```
Expected: `0 issues`

**Step 7: Commit**

```bash
git add cmd/root.go cmd/replay.go cmd/replay_test.go
git commit -m "refactor(cmd): extract registerSimConfigFlags helper for run+replay sharing

- Add registerSimConfigFlags(cmd) to reduce flag duplication
- Add replayCmd skeleton with trace-header/trace-data flags
- Implements BC-4 (shared sim config flags)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 2: Add JSON tags to `workload.SimResult` and test marshaling

**Contracts:** BC-2, BC-5

**Files:**
- Modify: `sim/workload/calibrate.go` (add JSON tags to existing SimResult type)
- Modify: `cmd/replay.go` (import workload, use workload.SimResult)
- Modify: `cmd/replay_test.go`

**Note:** `workload.SimResult` already exists in `sim/workload/calibrate.go` with identical fields (no JSON tags). Adding JSON tags is additive and backward-compatible. This is the correct location — not `cmd/` — because `blis calibrate` (#658) also consumes this type from `sim/workload`. Creating a `cmd.SimResult` would duplicate the type (R4 violation).

**Step 1: Write the failing test**

```go
// In cmd/replay_test.go, add:

func TestSimResult_JSONRoundTrip(t *testing.T) {
    // GIVEN a workload.SimResult with known values
    // workload.SimResult is in sim/workload/calibrate.go — JSON tags added by Task 2.
    sr := workload.SimResult{
        RequestID:    42,
        TTFT:         12345.0,
        E2E:          98765.0,
        InputTokens:  256,
        OutputTokens: 128,
    }

    // WHEN marshaled to JSON and back
    data, err := json.Marshal(sr)
    if err != nil {
        t.Fatalf("json.Marshal failed: %v", err)
    }
    var got workload.SimResult
    if err := json.Unmarshal(data, &got); err != nil {
        t.Fatalf("json.Unmarshal failed: %v", err)
    }

    // THEN all fields round-trip correctly (BC-2)
    if got.RequestID != 42 {
        t.Errorf("RequestID: got %d, want 42", got.RequestID)
    }
    if got.TTFT != 12345.0 {
        t.Errorf("TTFT: got %f, want 12345.0", got.TTFT)
    }
    if got.E2E != 98765.0 {
        t.Errorf("E2E: got %f, want 98765.0", got.E2E)
    }
    if got.InputTokens != 256 {
        t.Errorf("InputTokens: got %d, want 256", got.InputTokens)
    }
    if got.OutputTokens != 128 {
        t.Errorf("OutputTokens: got %d, want 128", got.OutputTokens)
    }

    // THEN JSON keys match the calibrate contract
    if !strings.Contains(string(data), `"request_id":42`) {
        t.Errorf("JSON must contain integer request_id, got: %s", data)
    }
    if !strings.Contains(string(data), `"ttft_us"`) {
        t.Errorf("JSON must contain ttft_us key, got: %s", data)
    }
    if !strings.Contains(string(data), `"e2e_us"`) {
        t.Errorf("JSON must contain e2e_us key, got: %s", data)
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestSimResult_JSONRoundTrip -v
```
Expected: `FAIL` — `SimResult` not defined yet.

**Step 3a: Add JSON tags to `workload.SimResult` in `sim/workload/calibrate.go`**

Modify `sim/workload/calibrate.go` lines 51-57 from:
```go
type SimResult struct {
    RequestID    int
    TTFT         float64 // Server-side: FirstTokenTime - ArrivalTime (µs)
    E2E          float64 // Server-side: CompletionTime - ArrivalTime (µs)
    InputTokens  int
    OutputTokens int
}
```
to:
```go
// SimResult holds per-request sim output for calibration matching.
// TTFT and E2E are server-side latencies in microseconds (simulation ticks).
type SimResult struct {
    RequestID    int     `json:"request_id"`
    TTFT         float64 `json:"ttft_us"` // server-side TTFT in microseconds
    E2E          float64 `json:"e2e_us"`  // server-side E2E in microseconds
    InputTokens  int     `json:"input_tokens"`
    OutputTokens int     `json:"output_tokens"`
}
```

**Step 3b: Add imports to replay.go** (no SimResult definition — uses `workload.SimResult`)

```go
package cmd

import (
    "encoding/json"
    "fmt"
    "math"
    "os"
    "sort"
    "strconv"
    "strings"
    "time"

    "github.com/sirupsen/logrus"
    "github.com/spf13/cobra"
    "gopkg.in/yaml.v3"

    sim "github.com/inference-sim/inference-sim/sim"
    "github.com/inference-sim/inference-sim/sim/cluster"
    "github.com/inference-sim/inference-sim/sim/workload"
)

var (
    traceHeaderPath string
    traceDataPath   string
)
// SimResult type is workload.SimResult from sim/workload/calibrate.go.
// JSON tags are added to that type in Task 2. No duplicate definition here.
```

**Step 4: Run test to verify it passes**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestSimResult_JSONRoundTrip -v
```
Expected: `PASS`

**Step 5: Run lint**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./cmd/...
```
Expected: `0 issues`

**Step 6: Commit**

```bash
git add cmd/replay.go cmd/replay_test.go
git commit -m "feat(cmd): add SimResult type for blis replay calibration output

- SimResult: integer request_id, ttft_us/e2e_us in microseconds, token counts
- JSON keys match blis calibrate schema from issue #657
- Implements BC-2, BC-5

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 3: Implement `extractSimResults` and test it

**Contracts:** BC-2, BC-5, BC-7

**Files:**
- Modify: `cmd/replay.go`
- Modify: `cmd/replay_test.go`

**Step 1: Write the failing test**

```go
// In cmd/replay_test.go, add:

func TestExtractSimResults_SortsAndConverts(t *testing.T) {
    // GIVEN a Metrics struct with 3 completed requests
    m := sim.NewMetrics()
    // Populate as simulator does (RequestTTFTs in ticks = microseconds)
    m.RequestTTFTs["request_2"] = 2000.0
    m.RequestTTFTs["request_0"] = 1000.0
    m.RequestTTFTs["request_1"] = 1500.0
    m.RequestE2Es["request_2"] = 20000.0
    m.RequestE2Es["request_0"] = 10000.0
    m.RequestE2Es["request_1"] = 15000.0
    m.Requests["request_0"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}
    m.Requests["request_1"] = sim.RequestMetrics{NumPrefillTokens: 200, NumDecodeTokens: 60}
    m.Requests["request_2"] = sim.RequestMetrics{NumPrefillTokens: 300, NumDecodeTokens: 70}

    // WHEN extractSimResults is called
    results := extractSimResults(m) // returns []workload.SimResult

    // THEN 3 results are returned in ascending request_id order (BC-5: determinism, R2)
    if len(results) != 3 {
        t.Fatalf("want 3 results, got %d", len(results))
    }
    if results[0].RequestID != 0 || results[1].RequestID != 1 || results[2].RequestID != 2 {
        t.Errorf("results not sorted by request_id: %v", results)
    }

    // THEN TTFT and E2E are in microseconds (BC-2, BC-6)
    if results[0].TTFT != 1000.0 {
        t.Errorf("results[0].TTFT: got %f, want 1000.0 (microseconds)", results[0].TTFT)
    }
    if results[0].E2E != 10000.0 {
        t.Errorf("results[0].E2E: got %f, want 10000.0 (microseconds)", results[0].E2E)
    }
    if results[0].InputTokens != 100 || results[0].OutputTokens != 50 {
        t.Errorf("token counts wrong for results[0]: %+v", results[0])
    }
}

func TestExtractSimResults_SkipsNonNumericIDs(t *testing.T) {
    // GIVEN metrics with a non-numeric ID (session follow-up)
    m := sim.NewMetrics()
    m.RequestTTFTs["request_0"] = 1000.0
    m.RequestTTFTs["session_follow_abc"] = 2000.0
    m.RequestE2Es["request_0"] = 5000.0
    m.RequestE2Es["session_follow_abc"] = 8000.0
    m.Requests["request_0"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}
    m.Requests["session_follow_abc"] = sim.RequestMetrics{NumPrefillTokens: 200, NumDecodeTokens: 60}

    // WHEN extractSimResults is called
    results := extractSimResults(m)

    // THEN only the numeric-ID request is included (BC-7)
    if len(results) != 1 {
        t.Fatalf("want 1 result (non-numeric ID skipped), got %d", len(results))
    }
    if results[0].RequestID != 0 {
        t.Errorf("wrong RequestID: got %d, want 0", results[0].RequestID)
    }
}

func TestExtractSimResults_ExcludesPartialTTFT(t *testing.T) {
    // GIVEN a request with TTFT but no E2E (timed out during decode)
    m := sim.NewMetrics()
    m.RequestTTFTs["request_0"] = 1000.0
    // No entry in RequestE2Es for request_0
    m.Requests["request_0"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 0}

    // WHEN extractSimResults is called
    results := extractSimResults(m)

    // THEN the incomplete request is excluded (no E2E = timeout after prefill)
    if len(results) != 0 {
        t.Errorf("want 0 results (no E2E = incomplete), got %d", len(results))
    }
}

func TestExtractSimResults_EmptyMetrics_ReturnsEmptySlice(t *testing.T) {
    // GIVEN empty metrics (all requests timed out before prefill)
    m := sim.NewMetrics()

    // WHEN extractSimResults is called
    results := extractSimResults(m)

    // THEN an initialized empty slice is returned (not nil)
    // A nil slice marshals to JSON "null"; an empty slice marshals to "[]"
    if results == nil {
        t.Error("want initialized empty slice (not nil) so JSON marshal produces [] not null")
    }
    data, err := json.Marshal(results)
    if err != nil {
        t.Fatalf("json.Marshal failed: %v", err)
    }
    if string(data) != "[]" {
        t.Errorf("want JSON [], got %s", data)
    }
}

func TestExtractSimResults_MixedRequests_OnlyCompletedIncluded(t *testing.T) {
    // GIVEN metrics with completed, timed-out, and non-numeric IDs mixed
    m := sim.NewMetrics()
    // Completed request
    m.RequestTTFTs["request_1"] = 1500.0
    m.RequestE2Es["request_1"] = 15000.0
    m.Requests["request_1"] = sim.RequestMetrics{NumPrefillTokens: 200, NumDecodeTokens: 60}
    // Timed out after prefill (TTFT but no E2E)
    m.RequestTTFTs["request_2"] = 2000.0
    m.Requests["request_2"] = sim.RequestMetrics{NumPrefillTokens: 300, NumDecodeTokens: 0}
    // Session follow-up (non-numeric ID)
    m.RequestTTFTs["session_followup_abc"] = 3000.0
    m.RequestE2Es["session_followup_abc"] = 30000.0
    m.Requests["session_followup_abc"] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}

    // WHEN extractSimResults is called
    results := extractSimResults(m)

    // THEN only the fully-completed numeric-ID request is included
    if len(results) != 1 {
        t.Fatalf("want 1 result (only completed numeric request), got %d: %v", len(results), results)
    }
    if results[0].RequestID != 1 {
        t.Errorf("want RequestID=1, got %d", results[0].RequestID)
    }
}

func TestExtractSimResults_DeterminismInvariant(t *testing.T) {
    // GIVEN the same metrics populated in two different key-insertion orders
    makeMetrics := func() *sim.Metrics {
        m := sim.NewMetrics()
        for _, id := range []string{"request_2", "request_0", "request_1"} {
            m.RequestTTFTs[id] = float64(len(id)) * 1000
            m.RequestE2Es[id] = float64(len(id)) * 5000
            m.Requests[id] = sim.RequestMetrics{NumPrefillTokens: 100, NumDecodeTokens: 50}
        }
        return m
    }

    // WHEN extractSimResults is called twice
    r1 := extractSimResults(makeMetrics())
    r2 := extractSimResults(makeMetrics())

    // THEN the output is identical (INV-6: determinism)
    if len(r1) != len(r2) {
        t.Fatalf("different lengths: %d vs %d", len(r1), len(r2))
    }
    for i := range r1 {
        if r1[i].RequestID != r2[i].RequestID {
            t.Errorf("index %d: RequestID %d vs %d — output is non-deterministic", i, r1[i].RequestID, r2[i].RequestID)
        }
    }
    // Verify order is ascending (the invariant being tested)
    for i := 1; i < len(r1); i++ {
        if r1[i].RequestID <= r1[i-1].RequestID {
            t.Errorf("results not sorted: index %d (%d) <= index %d (%d)", i, r1[i].RequestID, i-1, r1[i-1].RequestID)
        }
    }
}
```

**Step 2: Run tests to verify they fail**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestExtractSimResults -v
```
Expected: `FAIL` — `extractSimResults` not defined yet.

**Step 3: Implement `extractSimResults`**

Add to `cmd/replay.go`:

```go
// extractSimResults converts Metrics to a slice of workload.SimResult for calibrate consumption.
// Only requests with both TTFT and E2E recorded (i.e., fully completed) are included.
// Non-numeric IDs (session follow-ups, format "request_<parent>_followup_<n>") are excluded.
// Results are sorted by RequestID for deterministic output (R2, INV-6).
// Returns an initialized empty slice (not nil) so JSON marshaling produces [] not null.
// Exclusions are logged at Debug level for observability (R1: no silent data loss).
func extractSimResults(m *sim.Metrics) []workload.SimResult {
    results := make([]workload.SimResult, 0, len(m.RequestTTFTs))
    var noE2ECount, noReqCount, nonNumericCount int
    for reqID, ttftUs := range m.RequestTTFTs {
        e2eUs, hasE2E := m.RequestE2Es[reqID]
        if !hasE2E {
            noE2ECount++ // timed out after prefill
            continue
        }
        rm, hasReq := m.Requests[reqID]
        if !hasReq {
            noReqCount++ // metrics inconsistency (defensive)
            continue
        }
        // Parse integer RequestID from "request_N" format (BC-7: skip non-numeric IDs)
        numStr := strings.TrimPrefix(reqID, "request_")
        id, err := strconv.Atoi(numStr)
        if err != nil {
            nonNumericCount++ // session follow-ups or other non-numeric IDs
            continue
        }
        results = append(results, workload.SimResult{
            RequestID:    id,
            TTFT:         ttftUs,
            E2E:          e2eUs,
            InputTokens:  rm.NumPrefillTokens,
            OutputTokens: rm.NumDecodeTokens,
        })
    }
    if noE2ECount > 0 {
        logrus.Debugf("extractSimResults: excluded %d request(s) with TTFT but no E2E (timed out after prefill)", noE2ECount)
    }
    if noReqCount > 0 {
        logrus.Debugf("extractSimResults: excluded %d request(s) in TTFTs but missing from Requests (metrics inconsistency)", noReqCount)
    }
    if nonNumericCount > 0 {
        logrus.Debugf("extractSimResults: excluded %d non-numeric-ID request(s) (session follow-ups)", nonNumericCount)
    }
    // Sort by RequestID for deterministic JSON output (R2, INV-6)
    sort.Slice(results, func(i, j int) bool {
        return results[i].RequestID < results[j].RequestID
    })
    return results
}
```

**Step 4: Run tests to verify they pass**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestExtractSimResults -v
```
Expected: all `PASS`

**Step 5: Run lint**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./cmd/...
```
Expected: `0 issues`

**Step 6: Commit**

```bash
git add cmd/replay.go cmd/replay_test.go
git commit -m "feat(cmd): add extractSimResults for per-request calibration output

- Extracts TTFT/E2E in microseconds from Metrics.RequestTTFTs/RequestE2Es
- Skips non-numeric IDs (session follow-ups) per BC-7
- Skips requests without E2E (timed out after prefill)
- Sorts by RequestID for deterministic output (R2, INV-6)
- Implements BC-2, BC-5, BC-7

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 4: Implement replay Run: validation, trace loading, horizon computation

**Contracts:** BC-1, BC-3, BC-6, BC-8

**Dependency:** Must be executed AFTER Task 2. Task 4's Run body references `workload.LoadTraceV2` and `workload.LoadTraceV2Requests` — the import block for `workload` is added in Task 2 (Step 3b). Running Task 4 before Task 2 would cause a compile error.

**Files:**
- Modify: `cmd/replay.go`
- Modify: `cmd/replay_test.go`

**Step 1: Write the failing test**

```go
// In cmd/replay_test.go, add:

func TestReplayCmd_TraceHeaderFlag_Registered(t *testing.T) {
    // GIVEN the replay command
    // WHEN checking for --trace-header flag
    f := replayCmd.Flags().Lookup("trace-header")
    // THEN it must exist with empty default (BC-6: missing = fail fast)
    if f == nil {
        t.Error("replayCmd missing --trace-header flag")
    }
    if f != nil && f.DefValue != "" {
        t.Errorf("--trace-header default must be empty (required), got %q", f.DefValue)
    }
}

func TestReplayCmd_TraceDataFlag_Registered(t *testing.T) {
    f := replayCmd.Flags().Lookup("trace-data")
    if f == nil {
        t.Error("replayCmd missing --trace-data flag")
    }
    if f != nil && f.DefValue != "" {
        t.Errorf("--trace-data default must be empty (required), got %q", f.DefValue)
    }
}

func TestComputeReplayHorizon_TwiceMaxArrival(t *testing.T) {
    // BC-3: horizon = max(arrivals) * 2
    requests := []*sim.Request{
        {ArrivalTime: 1000},
        {ArrivalTime: 5000},
        {ArrivalTime: 3000},
    }
    horizon := computeReplayHorizon(requests)
    if horizon != 10000 {
        t.Errorf("want horizon 10000 (5000*2), got %d", horizon)
    }
}

func TestComputeReplayHorizon_EmptyRequests_ReturnsMaxInt64(t *testing.T) {
    // Edge case: no requests → MaxInt64 fallback
    horizon := computeReplayHorizon([]*sim.Request{})
    if horizon != math.MaxInt64 {
        t.Errorf("want math.MaxInt64 for empty requests, got %d", horizon)
    }
}

func TestComputeReplayHorizon_AllArrivalsAtZero_ReturnsFixedBuffer(t *testing.T) {
    // Edge case: all requests at t=0 (common for synthetic traces)
    // Must NOT return math.MaxInt64 (would hang simulation)
    requests := []*sim.Request{{ArrivalTime: 0}, {ArrivalTime: 0}}
    horizon := computeReplayHorizon(requests)
    if horizon <= 0 || horizon == math.MaxInt64 {
        t.Errorf("want a finite positive buffer for all-zero arrivals, got %d", horizon)
    }
}

func TestComputeReplayHorizon_LargeArrival_NoOverflow(t *testing.T) {
    // Overflow guard: maxArrival > MaxInt64/2 must not wrap to negative
    requests := []*sim.Request{{ArrivalTime: math.MaxInt64/2 + 1}}
    horizon := computeReplayHorizon(requests)
    if horizon <= 0 {
        t.Errorf("want positive horizon for large arrival (no overflow), got %d", horizon)
    }
    if horizon != math.MaxInt64 {
        t.Errorf("want MaxInt64 as overflow fallback, got %d", horizon)
    }
}
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run "TestReplayCmd_Trace|TestComputeReplayHorizon" -v
```
Expected: `FAIL` — `computeReplayHorizon` not defined yet.

**Step 3: Implement `computeReplayHorizon` and populate the Run skeleton**

Add to `cmd/replay.go`:

```go
// computeReplayHorizon returns 2× the latest arrival time as the simulation horizon,
// providing a generous buffer for the last request to complete. Returns math.MaxInt64
// for empty request slices or when overflow would occur.
func computeReplayHorizon(requests []*sim.Request) int64 {
    if len(requests) == 0 {
        return math.MaxInt64
    }
    var maxArrival int64
    for _, req := range requests {
        if req.ArrivalTime > maxArrival {
            maxArrival = req.ArrivalTime
        }
    }
    // Overflow guard: if 2× would overflow int64, use MaxInt64 directly.
    if maxArrival > math.MaxInt64/2 {
        return math.MaxInt64
    }
    if maxArrival <= 0 {
        // All requests at t=0: use a fixed generous buffer of 10 minutes (600,000,000 µs)
        // rather than MaxInt64 (which would cause the simulation to run indefinitely).
        return 600_000_000
    }
    return maxArrival * 2
}
```

Update `replayCmd.Run` to include validation and trace loading:

```go
Run: func(cmd *cobra.Command, args []string) {
    level, err := logrus.ParseLevel(logLevel)
    if err != nil {
        logrus.Fatalf("Invalid log level: %s", logLevel)
    }
    logrus.SetLevel(level)

    // Validate required inputs (BC-6, BC-8)
    if traceHeaderPath == "" {
        logrus.Fatalf("--trace-header is required")
    }
    if traceDataPath == "" {
        logrus.Fatalf("--trace-data is required")
    }
    if _, statErr := os.Stat(traceHeaderPath); os.IsNotExist(statErr) {
        logrus.Fatalf("--trace-header file not found: %s", traceHeaderPath)
    }
    if _, statErr := os.Stat(traceDataPath); os.IsNotExist(statErr) {
        logrus.Fatalf("--trace-data file not found: %s", traceDataPath)
    }
    if model == "" {
        logrus.Fatalf("LLM name not provided. Exiting simulation.")
    }

    // Load trace (BC-1)
    traceData, err := workload.LoadTraceV2(traceHeaderPath, traceDataPath)
    if err != nil {
        logrus.Fatalf("Failed to load trace: %v", err)
    }
    logrus.Infof("Loaded trace: %d records (mode=%s)", len(traceData.Records), traceData.Header.Mode)

    // Build requests from trace (BC-1)
    requests, err := workload.LoadTraceV2Requests(traceData, seed)
    if err != nil {
        logrus.Fatalf("Failed to build requests from trace: %v", err)
    }
    logrus.Infof("Built %d requests for replay", len(requests))

    // Compute horizon (BC-3)
    replayHorizon := computeReplayHorizon(requests)
    if cmd.Flags().Changed("horizon") {
        replayHorizon = simulationHorizon
    }
    logrus.Infof("Simulation horizon: %d ticks", replayHorizon)

    // [Config resolution — Task 5]
    // [Cluster construction + run — Task 5]
},
```

**Step 4: Run tests to verify they pass**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run "TestReplayCmd_Trace|TestComputeReplayHorizon" -v
```
Expected: all `PASS`

**Step 5: Run lint**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./cmd/...
```
Expected: `0 issues`

**Step 6: Commit**

```bash
git add cmd/replay.go cmd/replay_test.go
git commit -m "feat(cmd): add trace loading, horizon computation, and Run validation

- computeReplayHorizon: 2× max arrival time; math.MaxInt64 for empty
- Run: validates --trace-header, --trace-data, --model existence
- Implements BC-1, BC-3, BC-6, BC-8

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 5: Implement config resolution + cluster run in replayCmd.Run

**Contracts:** BC-1, BC-4

**Files:**
- Modify: `cmd/replay.go`

**Step 1: No new test** — config resolution is identical to runCmd; correctness is verified by the end-to-end test in Task 6.

**Step 2: Add config resolution to replayCmd.Run**

After the horizon computation block, add the config resolution (duplicated from runCmd.Run, adapted to replay context). The key difference: no workload generation, no session manager, no traceOutput flag.

Full implementation:

```go
    // Local copies of coefficient slices to avoid mutating package-level vars
    // (same shadowing as runCmd.Run:221 — required when loading from defaults.yaml)
    alphaCoeffs, betaCoeffs := alphaCoeffs, betaCoeffs

    // Normalize model name (same as runCmd)
    model = strings.ToLower(model)

    // Validate --latency-model (BC-4)
    if !sim.IsValidLatencyBackend(latencyModelBackend) {
        logrus.Fatalf("unknown --latency-model %q; valid options: %s",
            latencyModelBackend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
    }
    backend := latencyModelBackend

    // Alpha/beta coefficient validation (same as runCmd)
    alphaChanged := cmd.Flags().Changed("alpha-coeffs")
    betaChanged := cmd.Flags().Changed("beta-coeffs")
    if alphaChanged != betaChanged {
        if alphaChanged {
            logrus.Fatalf("--alpha-coeffs requires --beta-coeffs.")
        }
        logrus.Fatalf("--beta-coeffs requires --alpha-coeffs.")
    }
    for i, c := range alphaCoeffs {
        if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
            logrus.Fatalf("--alpha-coeffs[%d] must be finite non-negative, got %v", i, c)
        }
    }
    for i, c := range betaCoeffs {
        if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
            logrus.Fatalf("--beta-coeffs[%d] must be finite non-negative, got %v", i, c)
        }
    }
    if !cmd.Flags().Changed("latency-model") && alphaChanged && betaChanged {
        backend = "blackbox"
        logrus.Infof("--alpha-coeffs and --beta-coeffs provided; using blackbox mode")
    }

    var modelConfig = sim.ModelConfig{}
    var hwConfig = sim.HardwareCalib{}

    // Early defaults resolution (same as runCmd)
    if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
        hardware, tp, version := GetDefaultSpecs(model)
        if tensorParallelism == 0 && tp > 0 {
            tensorParallelism = tp
        }
        if gpu == "" && len(hardware) > 0 {
            gpu = hardware
        }
        if vllmVersion == "" && len(version) > 0 {
            vllmVersion = version
        }
    }
    kvBlocksFromDefaults := false

    // Latency model backend resolution (same as runCmd — duplicated, not extracted)
    // IMPORTANT: Keep this block in sync with runCmd.Run when modifying.
    if backend == "roofline" {
        var missing []string
        if gpu == "" {
            missing = append(missing, "--hardware (GPU type)")
        }
        if tensorParallelism <= 0 {
            missing = append(missing, "--tp (tensor parallelism)")
        }
        if len(missing) > 0 {
            logrus.Fatalf("Roofline mode requires %s. No defaults found for model=%s.",
                strings.Join(missing, " and "), model)
        }
        alphaChanged2 := cmd.Flags().Changed("alpha-coeffs")
        betaChanged2 := cmd.Flags().Changed("beta-coeffs")
        if cmd.Flags().Changed("latency-model") && (alphaChanged2 || betaChanged2) {
            logrus.Fatalf("--alpha-coeffs/--beta-coeffs cannot be used with --latency-model roofline.")
        }
        resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        modelConfigFolder = resolved
        resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
        if err != nil {
            logrus.Fatalf("%v", err)
        }
        hwConfigPath = resolvedHW
        if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
            _, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
            if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
                totalKVBlocks = kvBlocks
                kvBlocksFromDefaults = true
                logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
            }
        }
    }
    if backend == "crossmodel" {
        // COPY VERBATIM from cmd/root.go lines ~371-438.
        // After copying: verify it includes (a) missing flag checks for --hardware/--tp,
        // (b) resolveModelConfig/resolveHardwareConfig calls, (c) defaults.yaml loading
        // with R18 Changed() guards for cfg.CrossModelDefaults.BetaCoeffs/AlphaCoeffs,
        // (d) GetCoefficients for kvBlocks, (e) coefficient NaN/Inf/negative validation.
        // Verification: go build ./cmd/... && go test ./cmd/... must pass.
    }
    if backend == "trained-roofline" {
        // COPY VERBATIM from cmd/root.go lines ~440-520.
        // Same pattern as crossmodel but for cfg.TrainedRooflineDefaults.
        // After copying: verify it includes trained_roofline_defaults loading from defaults.yaml.
        // Verification: go build ./cmd/... && go test ./cmd/... must pass.
    }
    if backend == "blackbox" {
        // COPY VERBATIM from cmd/root.go lines ~522-565.
        // After copying: verify it includes GetCoefficients from defaults.yaml,
        // allZeros() safety guard ("No trained coefficients found"), and blackbox
        // auto-calc KV block detection.
        // Verification: go build ./cmd/... && go test ./cmd/... must pass.
    }
    if backend == "roofline" || backend == "crossmodel" || backend == "trained-roofline" {
        // COPY VERBATIM from cmd/root.go lines ~566-689.
        // After copying: verify it includes ParseHFConfig, GetModelConfigFromHF,
        // GetHWConfig, CalculateKVBlocks (when !kvBlocksFromDefaults), max-model-len
        // auto-derivation with rope_scaling via applyRopeScaling (same cmd/ package —
        // accessible without import), KV capacity cap.
        // Verification: go build ./cmd/... && go test ./cmd/... must pass.
    }
    // NOTE: Do NOT copy the workload generation blocks (runCmd lines ~705-812).
    // Those are runCmd-specific (spec loading, GenerateWorkload, session manager).
    // Replay uses pre-loaded requests from LoadTraceV2Requests instead.
    // After all copy blocks: diff the replay Run closure against runCmd.Run lines 209-689
    // to confirm structural parity (R23). Use: diff <(sed -n '209,689p' cmd/root.go) <(grep -A500 'func.*replayRun' cmd/replay.go | head -480)
    if maxModelLen < 0 {
        logrus.Fatalf("--max-model-len must be >= 0, got %d", maxModelLen)
    }

    // Numeric flag validation (same as runCmd)
    if numInstances < 1 {
        logrus.Fatalf("num-instances must be >= 1")
    }
    if totalKVBlocks <= 0 {
        logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
    }
    if blockSizeTokens <= 0 {
        logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
    }
    if maxRunningReqs <= 0 {
        logrus.Fatalf("--max-num-running-reqs must be > 0, got %d", maxRunningReqs)
    }
    if maxScheduledTokens <= 0 {
        logrus.Fatalf("--max-num-scheduled-tokens must be > 0, got %d", maxScheduledTokens)
    }
    if longPrefillTokenThreshold < 0 {
        logrus.Fatalf("--long-prefill-token-threshold must be >= 0, got %d", longPrefillTokenThreshold)
    }
    if kvCPUBlocks < 0 {
        logrus.Fatalf("--kv-cpu-blocks must be >= 0, got %d", kvCPUBlocks)
    }
    if snapshotRefreshInterval < 0 {
        logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
    }

    // Policy name validation (R23: MUST match runCmd lines 873-931 exactly).
    // Full validation block — copy verbatim from cmd/root.go lines 873-931:
    if !sim.IsValidAdmissionPolicy(admissionPolicy) {
        logrus.Fatalf("unknown --admission-policy %q; valid: %s",
            admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
    }
    if !sim.IsValidRoutingPolicy(routingPolicy) {
        logrus.Fatalf("unknown --routing-policy %q; valid: %s",
            routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
    }
    if !sim.IsValidPriorityPolicy(priorityPolicy) {
        logrus.Fatalf("unknown --priority-policy %q; valid: %s",
            priorityPolicy, strings.Join(sim.ValidPriorityPolicyNames(), ", "))
    }
    if !sim.IsValidScheduler(scheduler) {
        logrus.Fatalf("unknown --scheduler %q; valid: %s",
            scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
    }
    if !trace.IsValidTraceLevel(traceLevel) {
        logrus.Fatalf("unknown --trace-level %q; valid: none, decisions", traceLevel)
    }
    if admissionPolicy == "token-bucket" {
        if tokenBucketCapacity <= 0 || math.IsNaN(tokenBucketCapacity) || math.IsInf(tokenBucketCapacity, 0) {
            logrus.Fatalf("--token-bucket-capacity must be a finite value > 0, got %f", tokenBucketCapacity)
        }
        if tokenBucketRefillRate <= 0 || math.IsNaN(tokenBucketRefillRate) || math.IsInf(tokenBucketRefillRate, 0) {
            logrus.Fatalf("--token-bucket-refill-rate must be a finite value > 0, got %f", tokenBucketRefillRate)
        }
    }
    if counterfactualK < 0 {
        logrus.Fatalf("--counterfactual-k must be >= 0, got %d", counterfactualK)
    }
    if kvOffloadThreshold < 0 || kvOffloadThreshold > 1 || math.IsNaN(kvOffloadThreshold) || math.IsInf(kvOffloadThreshold, 0) {
        logrus.Fatalf("--kv-offload-threshold must be a finite value in [0, 1], got %f", kvOffloadThreshold)
    }
    if kvCPUBlocks > 0 && (kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0)) {
        logrus.Fatalf("--kv-transfer-bandwidth must be a finite value > 0 when --kv-cpu-blocks > 0, got %f", kvTransferBandwidth)
    }
    if kvTransferBaseLatency < 0 {
        logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
    }
    if admissionLatency < 0 {
        logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
    }
    if routingLatency < 0 {
        logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
    }
    if cmd.Flags().Changed("horizon") && replayHorizon <= 0 {
        logrus.Fatalf("--horizon must be > 0, got %d", replayHorizon)
    }
    // Trace-level informational warnings (R23: matches runCmd lines 902-910)
    if traceLevel == "none" && counterfactualK > 0 {
        logrus.Warnf("--counterfactual-k=%d has no effect without --trace-level decisions", counterfactualK)
    }
    if traceLevel == "none" && summarizeTrace {
        logrus.Warnf("--summarize-trace has no effect without --trace-level decisions")
    }
    if traceLevel != "none" && !summarizeTrace {
        logrus.Infof("Decision tracing enabled (trace-level=%s). Use --summarize-trace to print summary.", traceLevel)
    }

    // Policy bundle (same as runCmd — MUST include admission policy fields per R23)
    var bundleScorerConfigs []sim.ScorerConfig
    if policyConfigPath != "" {
        bundle, err := sim.LoadPolicyBundle(policyConfigPath)
        if err != nil {
            logrus.Fatalf("Failed to load policy config: %v", err)
        }
        if err := bundle.Validate(); err != nil {
            logrus.Fatalf("Invalid policy config: %v", err)
        }
        // Admission policy from bundle (mirrors runCmd lines 851-858)
        if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
            admissionPolicy = bundle.Admission.Policy
        }
        if bundle.Admission.TokenBucketCapacity != nil && !cmd.Flags().Changed("token-bucket-capacity") {
            tokenBucketCapacity = *bundle.Admission.TokenBucketCapacity
        }
        if bundle.Admission.TokenBucketRefillRate != nil && !cmd.Flags().Changed("token-bucket-refill-rate") {
            tokenBucketRefillRate = *bundle.Admission.TokenBucketRefillRate
        }
        if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
            routingPolicy = bundle.Routing.Policy
        }
        bundleScorerConfigs = bundle.Routing.Scorers
        if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
            priorityPolicy = bundle.Priority.Policy
        }
        if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
            scheduler = bundle.Scheduler
        }
    }

    // Scorer config parsing (R23: exact structure from runCmd lines 937-962)
    var parsedScorerConfigs []sim.ScorerConfig
    if routingPolicy == "weighted" {
        if routingScorers != "" {
            var err error
            parsedScorerConfigs, err = sim.ParseScorerConfigs(routingScorers)
            if err != nil {
                logrus.Fatalf("Invalid --routing-scorers: %v", err)
            }
        } else if len(bundleScorerConfigs) > 0 {
            parsedScorerConfigs = bundleScorerConfigs
        }
        // Log active scorer config (same as runCmd)
        activeScorerConfigs := parsedScorerConfigs
        if len(activeScorerConfigs) == 0 {
            activeScorerConfigs = sim.DefaultScorerConfigs()
        }
        scorerStrs := make([]string, len(activeScorerConfigs))
        for i, sc := range activeScorerConfigs {
            scorerStrs[i] = fmt.Sprintf("%s:%.1f", sc.Name, sc.Weight)
        }
        logrus.Infof("Weighted routing scorers: %s", strings.Join(scorerStrs, ", "))
    }
    if routingPolicy != "weighted" && routingScorers != "" {
        logrus.Warnf("--routing-scorers has no effect when routing policy is %q", routingPolicy)
    }

    logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s",
        admissionPolicy, routingPolicy, priorityPolicy, scheduler)
    logrus.Infof("Starting replay with %d KV blocks, horizon=%dticks", totalKVBlocks, replayHorizon)

    startTime := time.Now()

    // Build cluster config (same as runCmd)
    config := cluster.DeploymentConfig{
        SimConfig: sim.SimConfig{
            Horizon: replayHorizon,
            Seed:    seed,
            KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
                kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
            BatchConfig:         sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
            LatencyCoeffs:       sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs),
            ModelHardwareConfig: sim.NewModelHardwareConfig(modelConfig, hwConfig, model, gpu, tensorParallelism, backend, maxModelLen),
            PolicyConfig:        sim.NewPolicyConfig(priorityPolicy, scheduler),
        },
        NumInstances:            numInstances,
        AdmissionPolicy:         admissionPolicy,
        AdmissionLatency:        admissionLatency,
        RoutingLatency:          routingLatency,
        TokenBucketCapacity:     tokenBucketCapacity,
        TokenBucketRefillRate:   tokenBucketRefillRate,
        RoutingPolicy:           routingPolicy,
        RoutingScorerConfigs:    parsedScorerConfigs,
        TraceLevel:              traceLevel,
        CounterfactualK:         counterfactualK,
        SnapshotRefreshInterval: snapshotRefreshInterval,
    }

    // Run simulation (no session manager — session structure encoded in trace)
    cs := cluster.NewClusterSimulator(config, requests, nil)
    if err := cs.Run(); err != nil {
        logrus.Fatalf("Replay simulation failed: %v", err)
    }

    logrus.Infof("Replay wall-clock time: %.3fs", time.Since(startTime).Seconds())

    // Save aggregate metrics to stdout (and file if --results-path but no SimResult write yet)
    if numInstances > 1 {
        for _, inst := range cs.Instances() {
            if err := inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, ""); err != nil {
                logrus.Fatalf("SaveResults for instance %s: %v", inst.ID(), err)
            }
        }
    }
    // Save aggregate (always print to stdout; results-path handled below for SimResult)
    if err := cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, ""); err != nil {
        logrus.Fatalf("SaveResults: %v", err)
    }

    // Write SimResult JSON for calibrate (BC-2)
    if resultsPath != "" {
        simResults := extractSimResults(cs.AggregatedMetrics())
        data, err := json.MarshalIndent(simResults, "", "  ")
        if err != nil {
            logrus.Fatalf("Failed to marshal SimResults: %v", err)
        }
        if err := os.WriteFile(resultsPath, data, 0644); err != nil {
            logrus.Fatalf("Failed to write SimResults to %s: %v", resultsPath, err)
        }
        logrus.Infof("SimResults written to %s (%d entries)", resultsPath, len(simResults))
    }

    logrus.Info("Replay complete.")
```

**Step 3: Verify build passes**

```bash
cd .worktrees/pr-blis-replay && go build ./...
```
Expected: exit 0

**Step 4: Run lint**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./cmd/...
```
Expected: `0 issues`

**Step 5: Commit**

```bash
git add cmd/replay.go
git commit -m "feat(cmd): implement replay Run: config resolution, cluster run, SimResult output

- Duplicates latency model config resolution from runCmd (extraction too invasive)
- Uses computeReplayHorizon (2x max arrival) unless --horizon overrides
- No session manager: onRequestDone=nil (session structure encoded in trace)
- Saves aggregate metrics to stdout; SimResult array to --results-path
- Implements BC-1, BC-2, BC-3, BC-4

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 6: End-to-end integration test

**Contracts:** BC-1, BC-2, BC-3

**Files:**
- Modify: `cmd/replay_test.go`

**Step 1: Write the integration test**

```go
// In cmd/replay_test.go, add:

func TestReplayCmd_EndToEnd_BlackboxMode(t *testing.T) {
    // NOTE: This test mutates package-level flag vars shared with runCmd.
    // Do NOT use t.Parallel() — concurrent execution would create data races.
    // GIVEN a minimal TraceV2 header + data in a temp directory
    dir := t.TempDir()
    headerPath := filepath.Join(dir, "trace.yaml")
    dataPath := filepath.Join(dir, "trace.csv")
    resultsFilePath := filepath.Join(dir, "results.json")

    // Write header YAML
    header := `trace_version: 2
time_unit: microseconds
mode: generated
warm_up_requests: 0
`
    if err := os.WriteFile(headerPath, []byte(header), 0644); err != nil {
        t.Fatal(err)
    }

    // Write data CSV: 3 requests, arrival times spread over 1s
    // Columns: request_id,client_id,tenant_id,slo_class,session_id,round_index,
    //          prefix_group,streaming,input_tokens,output_tokens,
    //          text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,
    //          model,deadline_us,server_input_tokens,
    //          arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,
    //          num_chunks,status,error_message
    csvData := `request_id,client_id,tenant_id,slo_class,session_id,round_index,prefix_group,streaming,input_tokens,output_tokens,text_tokens,image_tokens,audio_tokens,video_tokens,reason_ratio,model,deadline_us,server_input_tokens,arrival_time_us,send_time_us,first_chunk_time_us,last_chunk_time_us,num_chunks,status,error_message
0,c1,t1,standard,s1,0,,false,10,5,10,0,0,0,0.0,,0,0,0,0,0,0,0,ok,
1,c1,t1,standard,s1,0,,false,10,5,10,0,0,0,0.0,,0,0,100000,100000,0,0,0,ok,
2,c1,t1,standard,s1,0,,false,10,5,10,0,0,0,0.0,,0,0,200000,200000,0,0,0,ok,
`
    if err := os.WriteFile(dataPath, []byte(csvData), 0644); err != nil {
        t.Fatal(err)
    }

    // Set CLI flags for blackbox mode (no HF config resolution needed)
    traceHeaderPath = headerPath
    traceDataPath = dataPath
    resultsPath = resultsFilePath
    model = "test-model"
    latencyModelBackend = "blackbox"
    betaCoeffs = []float64{100000.0, 1.0, 1.0}  // ~100ms prefill, fast decode
    alphaCoeffs = []float64{0.0, 0.0, 0.0}
    totalKVBlocks = 1000
    blockSizeTokens = 16
    maxRunningReqs = 64
    maxScheduledTokens = 2048
    numInstances = 1
    longPrefillTokenThreshold = 0
    seed = 42
    kvCPUBlocks = 0
    kvOffloadThreshold = 0.9
    kvTransferBandwidth = 100.0
    kvTransferBaseLatency = 0
    snapshotRefreshInterval = 0
    admissionPolicy = "always-admit"
    routingPolicy = "round-robin"
    priorityPolicy = "constant"
    scheduler = "fcfs"
    policyConfigPath = ""
    routingScorers = ""
    traceLevel = "none"
    counterfactualK = 0
    maxModelLen = 0
    simulationHorizon = math.MaxInt64

    // Load trace and build requests (library-level BC-1 verification)
    trace, err := workload.LoadTraceV2(headerPath, dataPath)
    if err != nil {
        t.Fatalf("LoadTraceV2 failed: %v", err)
    }
    if len(trace.Records) != 3 {
        t.Errorf("want 3 records, got %d", len(trace.Records))
    }

    reqs, err := workload.LoadTraceV2Requests(trace, 42)
    if err != nil {
        t.Fatalf("LoadTraceV2Requests failed: %v", err)
    }
    if len(reqs) != 3 {
        t.Fatalf("want 3 requests, got %d (BC-1)", len(reqs))
    }

    // Verify token counts preserved (BC-1)
    for _, req := range reqs {
        if len(req.InputTokens) != 10 {
            t.Errorf("want 10 input tokens, got %d", len(req.InputTokens))
        }
        if len(req.OutputTokens) != 5 {
            t.Errorf("want 5 output tokens, got %d", len(req.OutputTokens))
        }
    }

    // Verify horizon computation (BC-3): max arrival = 200000, horizon = 400000
    horizon := computeReplayHorizon(reqs)
    if horizon != 400000 {
        t.Errorf("want horizon 400000 (200000*2), got %d (BC-3)", horizon)
    }

    // Run the actual simulation via replayCmd.Run (BC-2: full SimResult output)
    // Blackbox mode with explicit coefficients avoids HF config resolution.
    // Save flags, run, restore to avoid test pollution.
    origModel := model
    origBackend := latencyModelBackend
    origBeta := betaCoeffs
    origAlpha := alphaCoeffs
    origTotalKV := totalKVBlocks
    origBlockSize := blockSizeTokens
    origMaxRunning := maxRunningReqs
    origMaxSched := maxScheduledTokens
    origInstances := numInstances
    origSeed := seed
    origResults := resultsPath
    origThreshold := longPrefillTokenThreshold
    origKVCPU := kvCPUBlocks
    origOffload := kvOffloadThreshold
    origBandwidth := kvTransferBandwidth
    origBaseLatency := kvTransferBaseLatency
    origSnapRefresh := snapshotRefreshInterval
    origAdmission := admissionPolicy
    origRouting := routingPolicy
    origPriority := priorityPolicy
    origScheduler := scheduler
    origPolicyConfig := policyConfigPath
    origMaxModelLen := maxModelLen
    origTraceLevel := traceLevel
    origCounterfactualK := counterfactualK
    origTraceHeader := traceHeaderPath
    origTraceData := traceDataPath
    defer func() {
        model = origModel; latencyModelBackend = origBackend
        betaCoeffs = origBeta; alphaCoeffs = origAlpha
        totalKVBlocks = origTotalKV; blockSizeTokens = origBlockSize
        maxRunningReqs = origMaxRunning; maxScheduledTokens = origMaxSched
        numInstances = origInstances; seed = origSeed; resultsPath = origResults
        longPrefillTokenThreshold = origThreshold; kvCPUBlocks = origKVCPU
        kvOffloadThreshold = origOffload; kvTransferBandwidth = origBandwidth
        kvTransferBaseLatency = origBaseLatency; snapshotRefreshInterval = origSnapRefresh
        admissionPolicy = origAdmission; routingPolicy = origRouting
        priorityPolicy = origPriority; scheduler = origScheduler
        policyConfigPath = origPolicyConfig; maxModelLen = origMaxModelLen
        traceLevel = origTraceLevel; counterfactualK = origCounterfactualK
        traceHeaderPath = origTraceHeader; traceDataPath = origTraceData
    }()

    model = "test-model"
    latencyModelBackend = "blackbox"
    betaCoeffs = []float64{100000.0, 1.0, 1.0} // ~100ms prefill, fast decode
    alphaCoeffs = []float64{0.0, 0.0, 0.0}
    totalKVBlocks = 1000
    blockSizeTokens = 16
    maxRunningReqs = 64
    maxScheduledTokens = 2048
    numInstances = 1
    seed = 42
    resultsPath = resultsFilePath
    longPrefillTokenThreshold = 0
    kvCPUBlocks = 0
    kvOffloadThreshold = 0.9
    kvTransferBandwidth = 100.0
    kvTransferBaseLatency = 0
    snapshotRefreshInterval = 0
    admissionPolicy = "always-admit"
    routingPolicy = "round-robin"
    priorityPolicy = "constant"
    scheduler = "fcfs"
    policyConfigPath = ""
    maxModelLen = 0
    traceLevel = "none"
    counterfactualK = 0
    traceHeaderPath = headerPath
    traceDataPath = dataPath

    // Create a cobra command with Changed() tracking for the flags the Run closure checks
    testCmd := &cobra.Command{}
    registerSimConfigFlags(testCmd)
    testCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "")
    testCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "")
    if err := testCmd.ParseFlags([]string{
        "--model", "test-model",
        "--latency-model", "blackbox",
        "--beta-coeffs", "100000.0,1.0,1.0",
        "--alpha-coeffs", "0.0,0.0,0.0",
        "--total-kv-blocks", "1000",
        "--trace-header", headerPath,
        "--trace-data", dataPath,
        "--results-path", resultsFilePath,
    }); err != nil {
        t.Fatalf("ParseFlags failed: %v", err)
    }

    // Run the replay command
    replayCmd.Run(testCmd, nil)

    // Verify SimResult file was written (BC-2)
    data, err := os.ReadFile(resultsFilePath)
    if err != nil {
        t.Fatalf("results file not written: %v", err)
    }
    var simResults []workload.SimResult
    if err := json.Unmarshal(data, &simResults); err != nil {
        t.Fatalf("failed to parse SimResult JSON: %v\ncontent: %s", err, data)
    }

    // All 3 requests should have completed (BC-1: fidelity)
    if len(simResults) != 3 {
        t.Errorf("want 3 SimResult entries (one per trace record), got %d", len(simResults))
    }

    // Verify integer request IDs 0, 1, 2 (BC-2)
    for i, sr := range simResults {
        if sr.RequestID != i {
            t.Errorf("simResults[%d].RequestID = %d, want %d", i, sr.RequestID, i)
        }
        if sr.TTFT <= 0 {
            t.Errorf("simResults[%d].TTFT must be > 0, got %f", i, sr.TTFT)
        }
        if sr.E2E <= 0 {
            t.Errorf("simResults[%d].E2E must be > 0, got %f", i, sr.E2E)
        }
        if sr.InputTokens != 10 {
            t.Errorf("simResults[%d].InputTokens = %d, want 10", i, sr.InputTokens)
        }
        if sr.OutputTokens != 5 {
            t.Errorf("simResults[%d].OutputTokens = %d, want 5", i, sr.OutputTokens)
        }
    }

    // TTFT must be in microseconds (not ms): 100ms prefill = ~100,000 µs
    if simResults[0].TTFT < 1000 {
        t.Errorf("TTFT %f looks like ms (expected ~100000 µs for 100ms prefill)", simResults[0].TTFT)
    }
}
```

Note: The full simulation is tested manually (see acceptance criteria). The integration test verifies the library-level behavior (trace loading, request construction, horizon computation) without requiring HF config resolution.

**Step 2: Run tests to verify they pass**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -run TestReplayCmd_EndToEnd -v
```
Expected: `PASS`

**Step 3: Run all cmd tests**

```bash
cd .worktrees/pr-blis-replay && go test ./cmd/... -v 2>&1 | tail -20
```
Expected: all `PASS`, 0 failures.

**Step 4: Run lint**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./cmd/...
```
Expected: `0 issues`

**Step 5: Commit**

```bash
git add cmd/replay_test.go
git commit -m "test(cmd): add integration test for blis replay command

- Verifies BC-1: trace loading produces correct request count and token counts
- Verifies BC-3: horizon = 2× max arrival time
- Library-level test (no HF config resolution needed)
- Full simulation verified manually via acceptance criteria

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

#### Task 7: Update CLAUDE.md and run full verification

**Contracts:** all

**Files:**
- Modify: `CLAUDE.md`

**Step 1: Update CLAUDE.md**

In the File Organization section under `cmd/root.go`, add `cmd/replay.go` entry. Update the CLI commands section:

```
├── cmd/
│   ├── root.go    # ... (existing)
│   ├── replay.go  # `blis replay` command: replays TraceV2 file through DES; flags:
│   │              #   --trace-header, --trace-data (required), all sim config flags;
│   │              #   --results-path writes []SimResult (integer request_id, ttft_us, e2e_us)
│   ├── ...
```

In the CLI entry point section, add `blis replay --trace-header t.yaml --trace-data d.csv --model <model>` to the build/run commands.

**Step 2: Run full test suite**

```bash
cd .worktrees/pr-blis-replay && go test ./... -count=1
```
Expected: all packages pass, 0 failures.

**Step 3: Run full build**

```bash
cd .worktrees/pr-blis-replay && go build ./...
```
Expected: exit 0.

**Step 4: Run lint on all packages**

```bash
cd .worktrees/pr-blis-replay && golangci-lint run ./...
```
Expected: `0 issues`.

**Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for blis replay command and SimResult type

- Add cmd/replay.go entry to File Organization tree
- Document --trace-header, --trace-data flags
- Document SimResult schema for blis calibrate consumption

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-2 (SimResult JSON, workload.SimResult) | Task 2 | Unit | TestSimResult_JSONRoundTrip |
| BC-2 (JSON tags correct) | Task 2 | Unit | TestSimResult_JSONRoundTrip |
| BC-4 (flags registered) | Task 1 | Structural | TestReplayCmd_SimConfigFlags_Registered |
| BC-5 (determinism, sort) | Task 3 | Invariant | TestExtractSimResults_DeterminismInvariant |
| BC-5 (sorted values) | Task 3 | Unit | TestExtractSimResults_SortsAndConverts |
| BC-6 (missing flags) | Task 4 | Structural | TestReplayCmd_TraceHeaderFlag_Registered |
| BC-7 (non-numeric IDs) | Task 3 | Unit | TestExtractSimResults_SkipsNonNumericIDs |
| BC-7 (partial TTFT) | Task 3 | Unit | TestExtractSimResults_ExcludesPartialTTFT |
| BC-7 (empty metrics) | Task 3 | Unit | TestExtractSimResults_EmptyMetrics_ReturnsEmptySlice |
| BC-7 (mixed scenario) | Task 3 | Unit | TestExtractSimResults_MixedRequests_OnlyCompletedIncluded |
| BC-1, BC-2, BC-3 | Task 6 | Integration | TestReplayCmd_EndToEnd_BlackboxMode |
| BC-3 (horizon auto) | Task 4 | Unit | TestComputeReplayHorizon_TwiceMaxArrival |
| BC-3 (empty edge case) | Task 4 | Unit | TestComputeReplayHorizon_EmptyRequests_ReturnsMaxInt64 |
| BC-3 (zero arrivals) | Task 4 | Unit | TestComputeReplayHorizon_AllArrivalsAtZero_ReturnsFixedBuffer |
| BC-3 (overflow guard) | Task 4 | Unit | TestComputeReplayHorizon_LargeArrival_NoOverflow |

**Invariant companion:** `TestExtractSimResults_DeterminismInvariant` verifies that sorted output satisfies INV-6 as a system law — it calls the function twice with same inputs and verifies byte-identical output, which would fail if sort were removed (unlike a snapshot test that only checks "output is sorted once").

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|-----------|------|
| Config resolution divergence from runCmd | Medium | High | IMPORTANT comment in code: "Keep in sync with runCmd.Run"; code review Step 4.5 perspective 1 | Task 5 |
| Package-level flag vars collision between runCmd and replayCmd | Low | High | Cobra resolves per-command flag set; only one command executes at a time | Task 1 |
| TTFT/E2E unit confusion (ms vs µs) | Medium | High | Unit test explicitly checks µs values from RequestTTFTs (not Requests[id].TTFT) | Task 3 |
| Non-deterministic SimResult output | Low | Medium | sort.Slice by RequestID; unit test verifies sorted order | Task 3 |
| `registerSimConfigFlags` breaks existing runCmd tests | Low | Medium | Mechanical extraction — same defaults, same var bindings | Task 1 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — `SimResult` is minimal, no new interfaces
- [x] No feature creep — no calibrate logic, no new routing policies
- [x] No unexercised flags — all flags exercised in test or documented in acceptance criteria
- [x] No partial implementations — replay Run is complete (config resolution + run + output)
- [x] No breaking changes — `runCmd` behavior unchanged; `registerSimConfigFlags` is mechanical extraction
- [x] No hidden global state — package-level vars shared intentionally; one command executes at a time
- [x] All new code passes golangci-lint — verified per-task
- [x] Shared test helpers used — `sim.NewMetrics()` from existing package
- [x] CLAUDE.md updated — Task 7 adds cmd/replay.go to File Organization tree and flags to CLI docs
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — 3 deviations documented, all justified
- [x] Each task produces working, testable code
- [x] Task dependencies correct — Task 1 (skeleton) before Tasks 2-7
- [x] All contracts mapped to tasks — see H) Test Strategy
- [x] Golden dataset: no changes (replay adds new command, does not modify existing sim paths)
- [x] Construction site audit: `SimResult` is new, no existing sites to audit

**Antipattern rules:**
- [x] R1: No silent continue/return — error paths use logrus.Fatalf or `continue` with explicit skip rationale
- [x] R2: Map iteration → sorted output in extractSimResults (sort.Slice by RequestID)
- [x] R3: All numeric CLI flags validated (same as runCmd validation block)
- [x] R4: No new struct fields in existing structs
- [x] R5: No resource-allocating loops with rollback risk
- [x] R6: No logrus.Fatalf in sim/ packages — only in cmd/
- [x] R7: Invariant tests alongside extractSimResults (sorted order verifies INV-6)
- [x] R8: No exported mutable maps
- [x] R9: No YAML fields with zero-value ambiguity (no new YAML structs)
- [x] R10: No new YAML parsing (uses existing LoadTraceV2 which already uses KnownFields)
- [x] R11: `computeReplayHorizon` guards `maxArrival <= 0` before `* 2`
- [x] R12: No golden dataset changes
- [x] R13: SimResult is a data type, not an interface — R13 not applicable
- [x] R14: `extractSimResults` is a single-concern function (metrics → []SimResult)
- [x] R15: No "planned for PR N" references
- [x] R16: No new config params (replay uses existing sim config flags)
- [x] R17: No new routing signals
- [x] R18: `cmd.Flags().Changed("horizon")` guards user override correctly
- [x] R19: No unbounded loops
- [x] R20: No new analyzers/detectors
- [x] R21: No range over mutable slices
- [x] R22: No pre-check / operation pairing
- [x] R23: `runCmd` and `replayCmd` both call the same `buildDeploymentConfig` factory functions (cluster.DeploymentConfig construction is identical)

---

## Appendix: File-Level Implementation Details

### File: `cmd/root.go` (modify)

**Purpose:** Extract `registerSimConfigFlags(cmd)` from `init()` to reduce duplication.

**Change:** Add the `registerSimConfigFlags` function after line 193 (after `allZeros`). Replace the block `runCmd.Flags().Int64Var(&seed, "seed", ...)` through `runCmd.Flags().StringVar(&resultsPath, "results-path", ...)` with a call to `registerSimConfigFlags(runCmd)`. The workload-generation flags (`--workload`, `--rate`, `--num-requests`, etc.) and run-only flags (`--workload-spec`, `--trace-output`) remain as direct `runCmd.Flags()` calls after the helper.

**Key implementation notes:**
- Every flag in the helper uses the same var pointer, default value, and description as the original `runCmd.Flags()` call
- `--horizon` is included in `registerSimConfigFlags` so both commands accept it
- `--trace-output` stays on `runCmd` only (replay doesn't export trace; it consumes one)

---

### File: `sim/workload/calibrate.go` (modify, +5 lines)

**Purpose:** Add JSON struct tags to existing `SimResult` type so it can be marshaled to JSON by `blis replay`.

**Change:** Add `json:"request_id"`, `json:"ttft_us"`, `json:"e2e_us"`, `json:"input_tokens"`, `json:"output_tokens"` tags to the 5 fields of `SimResult`. No new fields. No functional change to `PrepareCalibrationPairs` or any other consumer — JSON tags only affect marshaling.

---

### File: `cmd/replay.go` (create, ~320 lines)

**Purpose:** `blis replay` cobra command — loads TraceV2, runs DES, writes `workload.SimResult` JSON.

**Complete structure:**

```go
package cmd

import (
    "bytes"
    "encoding/json"
    "fmt"
    "math"
    "os"
    "path/filepath"
    "sort"
    "strconv"
    "strings"
    "time"

    "github.com/sirupsen/logrus"
    "github.com/spf13/cobra"
    "gopkg.in/yaml.v3"

    sim "github.com/inference-sim/inference-sim/sim"
    "github.com/inference-sim/inference-sim/sim/cluster"
    "github.com/inference-sim/inference-sim/sim/latency"
    "github.com/inference-sim/inference-sim/sim/workload"
)

var (
    traceHeaderPath string
    traceDataPath   string
)

// SimResult type is workload.SimResult from sim/workload/calibrate.go (JSON tags added by Task 2).
// No type defined here — cmd/replay.go uses workload.SimResult directly.

// computeReplayHorizon returns 2× the latest arrival time in requests.
// - Empty slice → math.MaxInt64 (no requests, horizon doesn't matter)
// - All arrivals at t=0 → 600,000,000 µs (10 min buffer; MaxInt64 would hang)
// - maxArrival > MaxInt64/2 → math.MaxInt64 (overflow guard)
// - Otherwise → maxArrival * 2
func computeReplayHorizon(requests []*sim.Request) int64 {
    if len(requests) == 0 {
        return math.MaxInt64
    }
    var maxArrival int64
    for _, req := range requests {
        if req.ArrivalTime > maxArrival {
            maxArrival = req.ArrivalTime
        }
    }
    if maxArrival > math.MaxInt64/2 {
        return math.MaxInt64
    }
    if maxArrival <= 0 {
        return 600_000_000
    }
    return maxArrival * 2
}

// extractSimResults converts Metrics to a slice of workload.SimResult for calibrate consumption.
// TTFT and E2E are taken from Metrics.RequestTTFTs/RequestE2Es (in microseconds/ticks).
// Only requests with both TTFT and E2E are included (fully completed only).
// Non-numeric IDs (session follow-ups) are excluded. All exclusions logged at Debug (R1).
// Results sorted by RequestID (R2). Returns make([], 0, ...) not nil — marshals to [] not null.
func extractSimResults(m *sim.Metrics) []workload.SimResult {
    results := make([]workload.SimResult, 0, len(m.RequestTTFTs))
    var noE2ECount, noReqCount, nonNumericCount int
    for reqID, ttftUs := range m.RequestTTFTs {
        e2eUs, hasE2E := m.RequestE2Es[reqID]
        if !hasE2E {
            noE2ECount++
            continue // timed out after prefill — no E2E
        }
        rm, hasReq := m.Requests[reqID]
        if !hasReq {
            noReqCount++
            continue // metrics inconsistency (should not happen; defensive guard)
        }
        numStr := strings.TrimPrefix(reqID, "request_")
        id, err := strconv.Atoi(numStr)
        if err != nil {
            nonNumericCount++
            continue // session follow-up or other non-numeric ID
        }
        results = append(results, workload.SimResult{
            RequestID:    id,
            TTFT:         ttftUs,
            E2E:          e2eUs,
            InputTokens:  rm.NumPrefillTokens,
            OutputTokens: rm.NumDecodeTokens,
        })
    }
    // Log all exclusions at Debug level for observability (R1: no silent data loss)
    if noE2ECount > 0 {
        logrus.Debugf("extractSimResults: excluded %d request(s) with TTFT but no E2E (timed out after prefill)", noE2ECount)
    }
    if noReqCount > 0 {
        logrus.Debugf("extractSimResults: excluded %d request(s) present in RequestTTFTs but missing from Requests map (metrics inconsistency)", noReqCount)
    }
    if nonNumericCount > 0 {
        logrus.Debugf("extractSimResults: excluded %d non-numeric-ID request(s) (session follow-ups)", nonNumericCount)
    }
    sort.Slice(results, func(i, j int) bool {
        return results[i].RequestID < results[j].RequestID
    })
    return results
}

var replayCmd = &cobra.Command{
    Use:   "replay",
    Short: "Replay a TraceV2 file through the discrete-event simulator",
    Long: `Replay takes a TraceV2 file (header YAML + data CSV) and runs the DES against the
exact request sequence. Unlike 'blis run', it does not generate requests from distributions —
the request sequence is fully determined by the trace.

Example:
  blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b`,
    Run: func(cmd *cobra.Command, args []string) {
        // ... (full implementation as described in Task 5)
    },
}

func init() {
    registerSimConfigFlags(replayCmd)
    replayCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (required)")
    replayCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (required)")
    rootCmd.AddCommand(replayCmd)
}
```

**Config resolution block** (inside replayCmd.Run, after trace loading): Duplicate of runCmd.Run lines ~220–980, with these differences:
1. No workload generation block (`spec`, `preGeneratedRequests`, `sessionMgr` — not needed)
2. Use `replayHorizon` instead of `simulationHorizon` for `config.SimConfig.Horizon`
3. `onRequestDone = nil` (no session manager)
4. `cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, "")` — no path (SimResult output uses separate file)
5. `extractSimResults` + `os.WriteFile` for `--results-path`

**Error handling:** All errors via `logrus.Fatalf` (same as runCmd — cmd/ is the CLI boundary).

---

### File: `cmd/replay_test.go` (create, ~200 lines)

**Purpose:** Tests for `SimResult` marshaling, `extractSimResults`, `computeReplayHorizon`, flag registration, and library-level integration (BC-1, BC-3).

**Import list:**
```go
package cmd

import (
    "encoding/json"
    "math"
    "os"
    "path/filepath"
    "strings"
    "testing"

    sim "github.com/inference-sim/inference-sim/sim"
    "github.com/inference-sim/inference-sim/sim/workload"
    "github.com/spf13/cobra"
)
```
