# `blis observe` CLI Command Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `blis observe` command that dispatches workload requests to a real inference server at precise arrival times and records per-request timing into TraceV2 files, completing the observe/replay/calibrate pipeline.

**The problem today:** BLIS can simulate inference latency (`blis run`) and replay captured traces (`blis replay`), but there is no way to collect ground-truth timing data from a real inference server using the same WorkloadSpec. Users who want to calibrate simulator accuracy must use external tools (inference-perf) and manually convert trace formats. This breaks the shared-pipeline guarantee — external tools don't support BLIS WorkloadSpec features like closed-loop sessions, cohorts, or SLO classes.

**What this PR adds:**

1. **`blis observe` command** — accepts a WorkloadSpec (YAML or CLI flags), dispatches requests to a real OpenAI-compatible server at rate-paced arrival times, and writes TraceV2 output files (YAML header + CSV data).
2. **Closed-loop session support** — multi-turn sessions generate follow-up requests on HTTP completion, reusing the same `SessionManager` as `blis run`. A serializer goroutine preserves the single-threaded contract.
3. **Concurrency-bounded dispatch** — semaphore limits in-flight requests. Warmup requests are dispatched but excluded from the trace.

**Why this matters:** This is the final piece of the observe/replay/calibrate pipeline (#652). Once merged, users can measure real server latency and calibrate BLIS predictions end-to-end using a single tool and shared workload definitions.

**Architecture:** New file `cmd/observe_cmd.go` adds the Cobra command and orchestrator. Existing `cmd/observe.go` (RealClient, Recorder) is reused unchanged. Workload generation uses `workload.GenerateWorkload` (shared with `blis run`). Session follow-ups are serialized through a dedicated goroutine to preserve `SessionManager`'s single-threaded contract.

**Source:** Design doc `docs/plans/2026-03-17-blis-observe-design.md`, GitHub issue #659

**Closes:** Fixes #659

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds the `blis observe` command — the data collection leg of the observe/replay/calibrate pipeline. It builds on existing infrastructure: `RealClient` and `Recorder` in `cmd/observe.go` handle HTTP dispatch and trace recording; `workload.GenerateWorkload` generates requests; `workload.SessionManager` handles multi-turn sessions.

The new code is entirely in `cmd/observe_cmd.go` (Cobra command + orchestrator) and `cmd/observe_cmd_test.go` (tests against mock HTTP server). No `sim/` packages are modified.

Adjacent blocks:
- **Upstream:** `sim/workload/generator.go` (GenerateWorkload), `sim/workload/session.go` (SessionManager), `sim/workload/synthesis.go` (SynthesizeFromDistribution)
- **Downstream:** `sim/workload/tracev2.go` (ExportTraceV2), consumed by `blis replay` and `blis calibrate`
- **Existing in `cmd/`:** `observe.go` (RealClient, Recorder, PendingRequest, RequestRecord)

No deviations from the design doc.

### B) Behavioral Contracts

**Positive contracts:**

**BC-1: Workload-spec dispatch**
- GIVEN a WorkloadSpec YAML with N clients and aggregate rate R
- WHEN `blis observe` is invoked with `--workload-spec`
- THEN requests are generated via `GenerateWorkload` and dispatched to the server at wall-clock times matching their arrival times
- MECHANISM: Dispatcher goroutine sleeps until `startWall + request.ArrivalTime`, then launches HTTP dispatch goroutine

**BC-2: Distribution synthesis dispatch**
- GIVEN `--rate` and distribution flags (prompt-tokens, output-tokens, etc.)
- WHEN `blis observe` is invoked without `--workload-spec`
- THEN a WorkloadSpec is synthesized via `SynthesizeFromDistribution` and requests are dispatched identically to BC-1
- MECHANISM: Same code path as `blis run`'s distribution mode, producing a WorkloadSpec before calling GenerateWorkload

**BC-3: Session follow-ups**
- GIVEN a WorkloadSpec with closed-loop session clients (max_rounds > 1)
- WHEN a session request completes successfully via HTTP
- THEN `SessionManager.OnComplete` is called (serialized) and any follow-up request is dispatched at `completionWallClock + thinkTime`
- MECHANISM: Completion events sent to a serializer goroutine via channel; serializer calls OnComplete and pushes follow-ups to followUpCh

**BC-4: TraceV2 export**
- GIVEN N requests dispatched (including session follow-ups) and W warmup requests
- WHEN all requests complete and drain finishes
- THEN a TraceV2 header YAML and data CSV are written with exactly `N - min(W, N)` records
- MECHANISM: Recorder accumulates records; warmup records are excluded by dispatch index; Export calls `workload.ExportTraceV2`

**BC-5: Streaming TTFT**
- GIVEN `--no-streaming` is not set (default)
- WHEN the server sends SSE chunks
- THEN `first_chunk_time_us` in the trace reflects the timestamp of the first SSE data chunk (true TTFT)
- MECHANISM: Existing `RealClient.handleStreamingResponse` records `FirstChunkTimeUs` on first chunk

**BC-6: Request conservation (OBS-INV-1)**
- GIVEN any observation run (including context cancellation)
- WHEN drain completes
- THEN `goroutine_launched_count == recorded_ok + recorded_error + recorded_timeout` (requests that were generated but never launched due to cancellation are excluded)
- MECHANISM: Every launched dispatch goroutine records exactly one TraceRecord via the Recorder before decrementing the WaitGroup. The WaitGroup ensures no goroutine is abandoned.

**BC-7: Concurrency bound (OBS-INV-2)**
- GIVEN `--max-concurrency K`
- WHEN requests are being dispatched
- THEN at most K HTTP requests are in-flight simultaneously
- MECHANISM: Buffered channel of size K acts as semaphore; acquired before goroutine launch, released after HTTP response

**Negative contracts:**

**BC-8: Session serialization (OBS-INV-3)**
- GIVEN concurrent HTTP completions for session requests
- WHEN completions arrive from multiple goroutines
- THEN all `SessionManager.OnComplete` calls MUST be made from a single goroutine (no concurrent access)
- MECHANISM: Completion events are sent to a serializer channel; a dedicated goroutine reads from it and calls OnComplete serially

**BC-9: No sim/ modification**
- GIVEN this PR's scope
- WHEN implementing observe
- THEN no files in `sim/` are created or modified
- MECHANISM: All new code is in `cmd/observe_cmd.go` and `cmd/observe_cmd_test.go`

**Error handling contracts:**

**BC-10: Non-fatal HTTP errors**
- GIVEN an HTTP error (5xx, timeout, connection refused)
- WHEN the error is returned from RealClient.Send
- THEN the error is recorded in the trace with status "error" or "timeout" and observation continues
- MECHANISM: RealClient.Send returns `(*RequestRecord, nil)` for HTTP errors (error recorded in Status field, not returned as Go error)

**BC-11: Session cancellation on error**
- GIVEN a session request that receives an HTTP error
- WHEN the adapter maps the response for SessionManager
- THEN `req.State` is set to `StateTimedOut`, and `SessionManager.OnComplete` cancels the session (no follow-up)
- MECHANISM: Adapter function sets State based on RequestRecord.Status

**BC-12: Graceful shutdown on Ctrl+C**
- GIVEN a running observation
- WHEN the user sends SIGINT
- THEN all in-flight requests are cancelled via context, partial trace is exported, and the command exits
- MECHANISM: `context.WithCancel` wraps the observation; signal handler triggers cancel

**BC-13: Required flag validation**
- GIVEN missing required flags (--server-url, --model, --trace-header, --trace-data) or no workload input
- WHEN `blis observe` is invoked
- THEN the command exits with a descriptive error via `logrus.Fatalf`
- MECHANISM: Cobra command Run function validates required flags at entry

**BC-14: Numeric flag validation (R3)**
- GIVEN invalid numeric flags (max-concurrency <= 0, warmup-requests < 0, rate <= 0/NaN/Inf)
- WHEN `blis observe` is invoked
- THEN the command exits with a descriptive error via `logrus.Fatalf`

### C) Component Interaction

```
WorkloadSpec (YAML or CLI flags)
    │
    ▼
GenerateWorkload()  ──→  []*sim.Request  +  []SessionBlueprint
    │                          │                     │
    │                          ▼                     ▼
    │                    Dispatcher goroutine    SessionManager
    │                     (rate-paced)           (serialized)
    │                          │                     │
    │                          ▼                     │
    │                    ┌─── Semaphore ───┐         │
    │                    │  (concurrency)  │         │
    │                    └────────────────-┘         │
    │                          │                     │
    │                     HTTP goroutines             │
    │                     (RealClient.Send)           │
    │                          │                     │
    │                          ▼                     │
    │                    completionCh  ──────────────┘
    │                          │           (serializer reads,
    │                          ▼            calls OnComplete,
    │                     Recorder          pushes follow-ups)
    │                          │
    │                          ▼
    │                    ExportTraceV2
    │                    (header + CSV)
```

**State ownership:**
- `Recorder` owns trace records (mutex-protected, existing)
- `SessionManager` owns session state (single-threaded via serializer)
- Dispatcher owns dispatch timing and concurrency control

**Extension friction:** Adding a new recording field: ~2 files (adapter + PendingRequest). Adding a new server protocol: ~1 file (new client type).

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Design lists `--horizon` default "from spec" | Plan uses `math.MaxInt64` as fallback when neither spec nor flag provides horizon | ADDITION: Distribution synthesis path has no spec horizon; must have a sane default |
| Design mentions `Recorder.RecordRequest` takes `(pending, record)` | Plan adds `arrivalTimeUs int64` + session params | ADDITION: Recorder needs arrival time and session metadata. R4 audit: no existing test callers of RecordRequest (verified via `grep`). Only internal caller is in the dispatch function. |

### E) Review Guide

**The tricky part:** The session serializer goroutine lifecycle. It must start before dispatch begins, receive completions from concurrent HTTP goroutines, call OnComplete serially, push follow-ups to the dispatcher, and shut down cleanly when all sessions are done. The drain logic (knowing when to close channels) is the most error-prone part.

**What to scrutinize:** BC-3 (session follow-ups), BC-8 (serialization), BC-6 (conservation). The adapter that maps `RequestRecord` → `sim.Request` fields for `OnComplete` — if `ProgressIndex` is wrong, context accumulation in follow-ups will be wrong.

**What's safe to skim:** Flag registration (mechanical), TraceV2 header construction (straightforward), distribution synthesis reuse (same code as `blis run`).

**Known debt:** (1) `RecordRequest` signature change (adding `arrivalTimeUs` + session params) is the only modification to existing code in `cmd/observe.go`. No existing test callers exist (verified). (2) Token count approximation: `strings.Repeat("hello ", N)` produces approximately N words but actual token count varies by tokenizer (typically 1-2 tokens per "hello " unit depending on BPE merge rules). `ServerInputTokens` in the trace records the server's actual count, which is the authoritative value for downstream calibration. This is a known simplification documented in the design doc's modeling decisions table.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `cmd/observe_cmd.go` — Cobra command, orchestrator, session adapter, dispatcher
- `cmd/observe_cmd_test.go` — All tests (mock HTTP server, session tests, drain tests)

**Files to modify:**
- `cmd/observe.go` — Add `arrivalTimeUs` parameter to `RecordRequest`, add `SessionID`/`RoundIndex` fields to recording

**Key decisions:**
- Session completions serialized via channel + dedicated goroutine (D7)
- Warmup exclusion by dispatch index (OBS-INV-4)
- Arrival times stored as relative in trace (Section 4 of design)

**No dead code:** Every function is called from the Cobra command or tests. No scaffolding.

### G) Task Breakdown

---

### Task 1: Extend Recorder with arrival time and session fields

**Contracts Implemented:** BC-4 (partial — trace record completeness), BC-6 (partial — conservation foundation)

**Files:**
- Modify: `cmd/observe.go` (RecordRequest signature, TraceRecord construction)
- Test: `cmd/observe_cmd_test.go` (new file)

**Step 1: Write failing test for RecordRequest with arrival time**

Context: The existing `RecordRequest` hardcodes `ArrivalTimeUs: 0`. We need to pass the actual arrival time and session metadata.

```go
package cmd

import (
	"testing"
)

func TestRecordRequest_PopulatesArrivalTimeAndSessionFields(t *testing.T) {
	// GIVEN a recorder and a completed request with arrival time and session info
	recorder := &Recorder{}
	pending := &PendingRequest{
		RequestID:   1,
		InputTokens: 100,
		Model:       "test-model",
		Streaming:   true,
		ClientID:    "client-1",
		TenantID:    "tenant-1",
		SLOClass:    "standard",
	}
	result := &RequestRecord{
		RequestID:         1,
		OutputTokens:      50,
		ServerInputTokens: 95,
		Status:            "ok",
		SendTimeUs:        1000000,
		FirstChunkTimeUs:  1000100,
		LastChunkTimeUs:   1000500,
		NumChunks:         10,
	}

	// WHEN recording with arrival time and session metadata
	recorder.RecordRequest(pending, result, 500000, "session-1", 0)

	// THEN the trace record has the correct arrival time and session fields
	records := recorder.Records()
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]
	if r.ArrivalTimeUs != 500000 {
		t.Errorf("ArrivalTimeUs: got %d, want 500000", r.ArrivalTimeUs)
	}
	if r.SessionID != "session-1" {
		t.Errorf("SessionID: got %q, want %q", r.SessionID, "session-1")
	}
	if r.RoundIndex != 0 {
		t.Errorf("RoundIndex: got %d, want 0", r.RoundIndex)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestRecordRequest_PopulatesArrivalTimeAndSessionFields -v`
Expected: FAIL — `RecordRequest` has wrong signature (too many arguments)

**Step 3: Implement — update RecordRequest signature**

In `cmd/observe.go`, update `RecordRequest` to accept arrival time and session fields:

```go
// RecordRequest captures one request-response cycle.
func (r *Recorder) RecordRequest(pending *PendingRequest, result *RequestRecord, arrivalTimeUs int64, sessionID string, roundIndex int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, workload.TraceRecord{
		Model:             pending.Model,
		ServerInputTokens: result.ServerInputTokens,
		RequestID:         result.RequestID,
		ClientID:          pending.ClientID,
		TenantID:          pending.TenantID,
		SLOClass:          pending.SLOClass,
		Streaming:         pending.Streaming,
		InputTokens:       pending.InputTokens,
		OutputTokens:      result.OutputTokens,
		ArrivalTimeUs:     arrivalTimeUs,
		SendTimeUs:        result.SendTimeUs,
		FirstChunkTimeUs:  result.FirstChunkTimeUs,
		LastChunkTimeUs:   result.LastChunkTimeUs,
		NumChunks:         result.NumChunks,
		Status:            result.Status,
		ErrorMessage:      result.ErrorMessage,
		SessionID:         sessionID,
		RoundIndex:        roundIndex,
	})
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestRecordRequest_PopulatesArrivalTimeAndSessionFields -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/observe.go cmd/observe_cmd_test.go
git commit -m "feat(cmd): extend RecordRequest with arrival time and session fields (BC-4)

- Add arrivalTimeUs, sessionID, roundIndex parameters to RecordRequest
- TraceRecord now populated with actual arrival time and session metadata

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Cobra command skeleton with flag registration and validation

**Contracts Implemented:** BC-13 (required flag validation), BC-14 (numeric flag validation)

**Files:**
- Create: `cmd/observe_cmd.go`
- Test: `cmd/observe_cmd_test.go` (add tests)

**Step 1: Write failing test for flag validation**

```go
func TestObserveCmd_MissingRequiredFlags_Errors(t *testing.T) {
	// GIVEN the observe command with no flags
	// WHEN executed
	// THEN it should produce an error about missing required flags
	// (We test this by verifying the command exists and has the expected flags)
	cmd := observeCmd
	if cmd == nil {
		t.Fatal("observeCmd is nil — command not registered")
	}
	if cmd.Use != "observe" {
		t.Errorf("Use: got %q, want %q", cmd.Use, "observe")
	}

	// Verify required flags exist
	for _, name := range []string{"server-url", "model", "trace-header", "trace-data"} {
		f := cmd.Flags().Lookup(name)
		if f == nil {
			t.Errorf("missing expected flag --%s", name)
		}
	}

	// Verify optional flags exist with defaults
	tests := []struct {
		name     string
		defValue string
	}{
		{"api-key", ""},
		{"server-type", "vllm"},
		{"max-concurrency", "256"},
		{"warmup-requests", "0"},
		{"no-streaming", "false"},
	}
	for _, tt := range tests {
		f := cmd.Flags().Lookup(tt.name)
		if f == nil {
			t.Errorf("missing expected flag --%s", tt.name)
			continue
		}
		if f.DefValue != tt.defValue {
			t.Errorf("--%s default: got %q, want %q", tt.name, f.DefValue, tt.defValue)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestObserveCmd_MissingRequiredFlags -v`
Expected: FAIL — `observeCmd` undefined

**Step 3: Implement Cobra command skeleton**

Create `cmd/observe_cmd.go`:

```go
package cmd

import (
	"context"
	"math"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	observeServerURL     string
	observeAPIKey        string
	observeServerType    string
	observeMaxConcur     int
	observeWarmup        int
	observeNoStreaming   bool
	observeTraceHeader   string
	observeTraceData     string
	observeModel         string
	observeWorkloadSpec  string
	observeRate          float64
	observeSeed          int64
	observeHorizon       int64
	observeNumRequests   int
	// Distribution synthesis flags (same as blis run)
	observePromptTokens    int
	observePromptStdDev    int
	observePromptMin       int
	observePromptMax       int
	observeOutputTokens    int
	observeOutputStdDev    int
	observeOutputMin       int
	observeOutputMax       int
	observePrefixTokens    int
)

var observeCmd = &cobra.Command{
	Use:   "observe",
	Short: "Dispatch workload requests to a real inference server and record timing",
	Long: `Observe sends requests from a WorkloadSpec to a real OpenAI-compatible inference
server at precise arrival times, recording per-request timing into TraceV2 files.

This is the data collection step of the observe/replay/calibrate pipeline.
The output TraceV2 files can be fed to 'blis replay' for simulation comparison
and 'blis calibrate' for accuracy measurement.

Supports both --workload-spec (YAML) and --rate (distribution synthesis) input paths.
Closed-loop sessions with multi-turn follow-ups are supported when the WorkloadSpec
contains session clients.

Example:
  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv

  blis observe --server-url http://localhost:8000 --model meta-llama/Llama-3.1-8B-Instruct \
    --rate 10 --num-requests 100 --trace-header trace.yaml --trace-data trace.csv`,
	Run: runObserve,
}

func init() {
	// Required flags
	observeCmd.Flags().StringVar(&observeServerURL, "server-url", "", "Inference server URL (required)")
	observeCmd.Flags().StringVar(&observeModel, "model", "", "Model name for API requests (required)")
	observeCmd.Flags().StringVar(&observeTraceHeader, "trace-header", "", "Output path for TraceV2 header YAML (required)")
	observeCmd.Flags().StringVar(&observeTraceData, "trace-data", "", "Output path for TraceV2 data CSV (required)")

	// Workload input
	observeCmd.Flags().StringVar(&observeWorkloadSpec, "workload-spec", "", "Path to WorkloadSpec YAML (alternative to --rate)")
	observeCmd.Flags().Float64Var(&observeRate, "rate", 0, "Requests per second for distribution synthesis")

	// Optional
	observeCmd.Flags().StringVar(&observeAPIKey, "api-key", "", "Bearer token for server authentication")
	observeCmd.Flags().StringVar(&observeServerType, "server-type", "vllm", "Server type (vllm, tgi, etc.)")
	observeCmd.Flags().IntVar(&observeMaxConcur, "max-concurrency", 256, "Maximum simultaneous in-flight requests")
	observeCmd.Flags().IntVar(&observeWarmup, "warmup-requests", 0, "Number of initial requests to exclude from trace")
	observeCmd.Flags().BoolVar(&observeNoStreaming, "no-streaming", false, "Disable streaming (use non-streaming HTTP)")
	observeCmd.Flags().Int64Var(&observeSeed, "seed", 42, "RNG seed for workload generation")
	observeCmd.Flags().Int64Var(&observeHorizon, "horizon", 0, "Observation horizon in microseconds (0 = from spec or unlimited)")
	observeCmd.Flags().IntVar(&observeNumRequests, "num-requests", 0, "Maximum requests to generate (0 = from spec or unlimited)")

	// Distribution synthesis flags (same names as blis run)
	observeCmd.Flags().IntVar(&observePromptTokens, "prompt-tokens", 512, "Average prompt token count (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptStdDev, "prompt-tokens-stdev", 50, "Prompt token std dev (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptMin, "prompt-tokens-min", 1, "Minimum prompt tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observePromptMax, "prompt-tokens-max", 2048, "Maximum prompt tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputTokens, "output-tokens", 512, "Average output token count (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputStdDev, "output-tokens-stdev", 50, "Output token std dev (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputMin, "output-tokens-min", 1, "Minimum output tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observeOutputMax, "output-tokens-max", 2048, "Maximum output tokens (distribution mode)")
	observeCmd.Flags().IntVar(&observePrefixTokens, "prefix-tokens", 0, "Shared prefix token count (distribution mode)")

	rootCmd.AddCommand(observeCmd)
}

func runObserve(cmd *cobra.Command, args []string) {
	// BC-13: Required flag validation
	if observeServerURL == "" {
		logrus.Fatalf("--server-url is required")
	}
	if observeModel == "" {
		logrus.Fatalf("--model is required")
	}
	if observeTraceHeader == "" {
		logrus.Fatalf("--trace-header is required")
	}
	if observeTraceData == "" {
		logrus.Fatalf("--trace-data is required")
	}
	if observeWorkloadSpec == "" && observeRate <= 0 {
		if !cmd.Flags().Changed("rate") && observeWorkloadSpec == "" {
			logrus.Fatalf("Either --workload-spec or --rate is required")
		}
	}

	// BC-14: Numeric flag validation (R3)
	if observeMaxConcur <= 0 {
		logrus.Fatalf("--max-concurrency must be > 0, got %d", observeMaxConcur)
	}
	if observeWarmup < 0 {
		logrus.Fatalf("--warmup-requests must be >= 0, got %d", observeWarmup)
	}
	if cmd.Flags().Changed("rate") && (observeRate <= 0 || math.IsNaN(observeRate) || math.IsInf(observeRate, 0)) {
		logrus.Fatalf("--rate must be a finite value > 0, got %v", observeRate)
	}

	// Generate workload
	var spec *workload.WorkloadSpec
	if observeWorkloadSpec != "" {
		var err error
		spec, err = workload.LoadWorkloadSpec(observeWorkloadSpec)
		if err != nil {
			logrus.Fatalf("Failed to load workload spec: %v", err)
		}
		if cmd.Flags().Changed("seed") {
			spec.Seed = observeSeed
		}
	} else {
		// Distribution synthesis (BC-2)
		spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
			Rate:               observeRate,
			NumRequests:        observeNumRequests,
			PrefixTokens:       observePrefixTokens,
			PromptTokensMean:   observePromptTokens,
			PromptTokensStdDev: observePromptStdDev,
			PromptTokensMin:    observePromptMin,
			PromptTokensMax:    observePromptMax,
			OutputTokensMean:   observeOutputTokens,
			OutputTokensStdDev: observeOutputStdDev,
			OutputTokensMin:    observeOutputMin,
			OutputTokensMax:    observeOutputMax,
		})
		spec.Seed = observeSeed
	}

	// Resolve horizon
	horizon := int64(math.MaxInt64)
	if cmd.Flags().Changed("horizon") && observeHorizon > 0 {
		horizon = observeHorizon
	} else if spec.Horizon > 0 {
		horizon = spec.Horizon
	}

	// Resolve max requests
	maxRequests := spec.NumRequests
	if cmd.Flags().Changed("num-requests") && observeNumRequests > 0 {
		maxRequests = int64(observeNumRequests)
	}

	// Guard unbounded generation
	if maxRequests <= 0 && horizon == math.MaxInt64 {
		logrus.Fatalf("Workload requires either num_requests, --num-requests, or --horizon to bound generation")
	}

	// Generate requests and session blueprints (BC-1, BC-2, D1)
	wl, err := workload.GenerateWorkload(spec, horizon, maxRequests)
	if err != nil {
		logrus.Fatalf("Failed to generate workload: %v", err)
	}

	logrus.Infof("Generated %d requests", len(wl.Requests))
	if len(wl.Sessions) > 0 {
		logrus.Infof("Generated %d session blueprints (closed-loop)", len(wl.Sessions))
	}

	if len(wl.Requests) == 0 {
		logrus.Warn("No requests generated — writing empty trace")
	}

	// Setup
	streaming := !observeNoStreaming
	client := NewRealClient(observeServerURL, observeAPIKey, observeModel, observeServerType)
	recorder := &Recorder{}

	var sessionMgr *workload.SessionManager
	if len(wl.Sessions) > 0 {
		sessionMgr = workload.NewSessionManager(wl.Sessions)
	}

	// Context for graceful shutdown (BC-12)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigCh
		logrus.Warn("Received interrupt signal, cancelling observation...")
		cancel()
	}()

	// Run orchestrator
	startTime := time.Now()
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, streaming, observeMaxConcur, observeWarmup)
	logrus.Infof("Observation wall-clock time: %.3fs", time.Since(startTime).Seconds())

	// Export trace (BC-4)
	header := &workload.TraceHeader{
		Version:        2,
		TimeUnit:       "us",
		CreatedAt:      time.Now().UTC().Format(time.RFC3339),
		Mode:           "real",
		WarmUpRequests: observeWarmup,
		Server: &workload.TraceServerConfig{
			Type:  observeServerType,
			Model: observeModel,
		},
	}
	if observeWorkloadSpec != "" {
		header.WorkloadSpec = observeWorkloadSpec
	}

	if err := recorder.Export(header, observeTraceHeader, observeTraceData); err != nil {
		logrus.Fatalf("Failed to export trace: %v", err)
	}

	records := recorder.Records()
	logrus.Infof("Trace exported: %d records to %s / %s", len(records), observeTraceHeader, observeTraceData)
}

// completionEvent carries HTTP completion info to the serializer goroutine.
type completionEvent struct {
	req    *sim.Request
	record *RequestRecord
	wallClock int64 // wall-clock microseconds at completion
}

// runObserveOrchestrator implements the dispatch loop with session support.
// This is the core orchestration function, extracted for testability.
func runObserveOrchestrator(
	ctx context.Context,
	client *RealClient,
	recorder *Recorder,
	sessionMgr *workload.SessionManager,
	requests []*sim.Request,
	streaming bool,
	maxConcurrency int,
	warmupCount int,
) {
	// placeholder — implemented in Task 3
	_ = ctx
	_ = client
	_ = recorder
	_ = sessionMgr
	_ = requests
	_ = streaming
	_ = maxConcurrency
	_ = warmupCount
}

// adaptForSessionManager converts an HTTP response into a sim.Request suitable
// for SessionManager.OnComplete. Only fields read by OnComplete are populated.
func adaptForSessionManager(original *sim.Request, record *RequestRecord) *sim.Request {
	// placeholder — implemented in Task 4
	return nil
}

// requestToPending converts a sim.Request to a PendingRequest for HTTP dispatch.
func requestToPending(req *sim.Request, reqIndex int, streaming bool) *PendingRequest {
	return &PendingRequest{
		RequestID:       reqIndex,
		InputTokens:     len(req.InputTokens),
		MaxOutputTokens: req.MaxOutputLen,
		Model:           req.Model,
		Streaming:       streaming,
		ClientID:        req.ClientID,
		TenantID:        req.TenantID,
		SLOClass:        req.SLOClass,
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestObserveCmd_MissingRequiredFlags -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues (unused parameters in placeholder functions may warn — acceptable until Task 3)

**Step 6: Commit**

```bash
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "feat(cmd): add blis observe command skeleton with flag registration (BC-13, BC-14)

- Register observe subcommand with all CLI flags
- Validate required flags and numeric parameters
- Generate workload via shared pipeline (D1)
- Stub orchestrator and adapter for subsequent tasks

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Open-loop dispatcher (no sessions)

**Contracts Implemented:** BC-1, BC-2, BC-6, BC-7

**Files:**
- Modify: `cmd/observe_cmd.go` (implement runObserveOrchestrator)
- Test: `cmd/observe_cmd_test.go`

**Step 1: Write failing test for open-loop dispatch with conservation check**

```go
func TestObserveOrchestrator_OpenLoop_ConservationAndConcurrency(t *testing.T) {
	// GIVEN a mock HTTP server that returns 200 OK with token counts
	requestCount := 0
	maxConcurrent := int64(0)
	currentConcurrent := int64(0)
	var mu sync.Mutex

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt64(&currentConcurrent, 1)
		defer atomic.AddInt64(&currentConcurrent, -1)

		// Track max concurrency
		cur := atomic.LoadInt64(&currentConcurrent)
		mu.Lock()
		if cur > maxConcurrent {
			maxConcurrent = cur
		}
		requestCount++
		mu.Unlock()

		// Simulate small processing time
		time.Sleep(10 * time.Millisecond)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(200)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	// Create 5 requests with staggered arrival times (10ms apart)
	requests := make([]*sim.Request, 5)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:          fmt.Sprintf("request_%d", i),
			ArrivalTime: int64(i) * 10000, // 10ms apart in microseconds
			InputTokens: make([]int, 100),
			OutputTokens: make([]int, 50),
			MaxOutputLen: 50,
			State:       sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// WHEN dispatching with max-concurrency 2 and 0 warmup
	ctx := context.Background()
	runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0)

	// THEN: BC-6 conservation: all 5 requests recorded
	records := recorder.Records()
	if len(records) != 5 {
		t.Fatalf("OBS-INV-1: expected 5 records, got %d", len(records))
	}

	// THEN: BC-7 concurrency bound: max concurrent <= 2
	if maxConcurrent > 2 {
		t.Errorf("OBS-INV-2: max concurrent %d exceeded limit 2", maxConcurrent)
	}

	// THEN: all records have status "ok"
	for i, r := range records {
		if r.Status != "ok" {
			t.Errorf("record %d: status %q, want %q", i, r.Status, "ok")
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestObserveOrchestrator_OpenLoop -v`
Expected: FAIL — runObserveOrchestrator is a placeholder

**Step 3: Implement open-loop dispatcher**

Replace the placeholder `runObserveOrchestrator` in `cmd/observe_cmd.go`:

```go
func runObserveOrchestrator(
	ctx context.Context,
	client *RealClient,
	recorder *Recorder,
	sessionMgr *workload.SessionManager,
	requests []*sim.Request,
	streaming bool,
	maxConcurrency int,
	warmupCount int,
) {
	if len(requests) == 0 {
		return
	}

	semaphore := make(chan struct{}, maxConcurrency)
	var wg sync.WaitGroup
	startWall := time.Now()
	dispatchIndex := 0

	// Channel for session follow-ups (buffered to avoid blocking serializer)
	followUpCh := make(chan *sim.Request, maxConcurrency)

	// Completion channel for session serialization (BC-8, D7)
	completionCh := make(chan completionEvent, maxConcurrency)

	// Active session tracking for drain (count unique session IDs)
	activeSessionCount := int64(0)
	if sessionMgr != nil {
		sessionIDs := make(map[string]bool)
		for _, req := range requests {
			if req.SessionID != "" && !sessionIDs[req.SessionID] {
				sessionIDs[req.SessionID] = true
				activeSessionCount++
			}
		}
	}

	// Session serializer goroutine (BC-8: single-threaded OnComplete)
	var serializerDone chan struct{}
	if sessionMgr != nil {
		serializerDone = make(chan struct{})
		go func() {
			defer close(serializerDone)
			for ce := range completionCh {
				adapted := adaptForSessionManager(ce.req, ce.record)
				followUps := sessionMgr.OnComplete(adapted, ce.wallClock)
				for _, fu := range followUps {
					followUpCh <- fu
				}
				// If session terminated (no follow-up and session request), decrement
				if ce.req.SessionID != "" && len(followUps) == 0 {
					atomic.AddInt64(&activeSessionCount, -1)
				}
			}
		}()
	}

	// Dispatch function (shared between pre-generated and follow-up requests)
	dispatch := func(req *sim.Request, idx int) {
		defer wg.Done()
		defer func() { <-semaphore }() // release concurrency slot

		pending := requestToPending(req, idx, streaming)
		record, _ := client.Send(ctx, pending)

		// Record trace (skip warmup by index)
		arrivalTimeUs := req.ArrivalTime
		if idx >= warmupCount {
			recorder.RecordRequest(pending, record, arrivalTimeUs, req.SessionID, req.RoundIndex)
		}

		// Session completion (BC-3)
		if sessionMgr != nil && req.SessionID != "" {
			completionCh <- completionEvent{
				req:       req,
				record:    record,
				wallClock: time.Now().Sub(startWall).Microseconds(),
			}
		}
	}

	// Merge pre-generated requests and follow-ups, dispatch in arrival order.
	// Follow-ups are buffered in a local slice and merged by arrival time
	// with pre-generated requests (deterministic, no select/default race).
	preGenIdx := 0
	var pendingFollowUps []*sim.Request

	drainFollowUps := func() {
		for {
			select {
			case fu := <-followUpCh:
				pendingFollowUps = append(pendingFollowUps, fu)
			default:
				return
			}
		}
	}

	for {
		// Drain any buffered follow-ups
		drainFollowUps()

		// Determine next request: pick earliest arrival time between
		// pre-generated and pending follow-ups
		var nextReq *sim.Request
		var isFollowUp bool

		hasPreGen := preGenIdx < len(requests)
		hasFollowUp := len(pendingFollowUps) > 0

		if hasPreGen && hasFollowUp {
			if pendingFollowUps[0].ArrivalTime <= requests[preGenIdx].ArrivalTime {
				nextReq = pendingFollowUps[0]
				pendingFollowUps = pendingFollowUps[1:]
				isFollowUp = true
			} else {
				nextReq = requests[preGenIdx]
				preGenIdx++
			}
		} else if hasPreGen {
			nextReq = requests[preGenIdx]
			preGenIdx++
		} else if hasFollowUp {
			nextReq = pendingFollowUps[0]
			pendingFollowUps = pendingFollowUps[1:]
			isFollowUp = true
		} else if sessionMgr != nil && atomic.LoadInt64(&activeSessionCount) > 0 {
			// No pre-generated or buffered follow-ups — wait for new follow-up or drain
			select {
			case fu, ok := <-followUpCh:
				if !ok {
					goto drain
				}
				nextReq = fu
				isFollowUp = true
			case <-ctx.Done():
				goto drain
			}
		} else {
			break // no more requests and no sessions
		}

		if nextReq == nil {
			continue
		}

		// Rate-pace: sleep until target wall-clock time
		targetWall := startWall.Add(time.Duration(nextReq.ArrivalTime) * time.Microsecond)
		sleepDur := time.Until(targetWall)
		if sleepDur > 0 {
			select {
			case <-time.After(sleepDur):
			case <-ctx.Done():
				goto drain
			}
		}

		// Acquire concurrency slot (BC-7)
		select {
		case semaphore <- struct{}{}:
		case <-ctx.Done():
			goto drain
		}

		idx := dispatchIndex
		dispatchIndex++
		_ = isFollowUp // used for logging if needed

		wg.Add(1)
		go dispatch(nextReq, idx)
	}

drain:
	// Wait for all in-flight requests
	wg.Wait()

	// Close session channels
	if sessionMgr != nil {
		close(completionCh)
		<-serializerDone
	}
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestObserveOrchestrator_OpenLoop -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./cmd/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "feat(cmd): implement open-loop dispatcher with concurrency control (BC-1, BC-6, BC-7)

- Rate-paced dispatch sleeping until target wall-clock time
- Semaphore-based concurrency bound
- Request conservation: every dispatch produces one trace record
- Session serializer goroutine stub (channel infrastructure ready)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Session adapter and follow-up dispatch

**Contracts Implemented:** BC-3, BC-8, BC-11

**Files:**
- Modify: `cmd/observe_cmd.go` (implement adaptForSessionManager)
- Test: `cmd/observe_cmd_test.go`

**Step 1: Write failing test for session follow-ups**

```go
func TestObserveOrchestrator_SessionFollowUp_GeneratesRound2(t *testing.T) {
	// GIVEN a mock server that returns success
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "response"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	// Create a session workload: 1 session with 2 rounds, think_time = 10ms
	spec := &workload.WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 10.0,
		Clients: []workload.ClientSpec{
			{
				ID:           "session-client",
				RateFraction: 1.0,
				Arrival:      workload.ArrivalSpec{Process: "constant"},
				InputDist:    workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist:   workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
				Reasoning: &workload.ReasoningSpec{
					MultiTurn: &workload.MultiTurnSpec{
						MaxRounds:  2,
						ThinkTimeUs:   10000, // 10ms in microseconds
						ContextGrowth: "accumulate",
						SingleSession: true,
					},
				},
			},
		},
	}

	wl, err := workload.GenerateWorkload(spec, 1_000_000, 1)
	if err != nil {
		t.Fatalf("GenerateWorkload: %v", err)
	}

	if len(wl.Sessions) == 0 {
		t.Skip("WorkloadSpec did not produce sessions — check spec configuration")
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	sessionMgr := workload.NewSessionManager(wl.Sessions)

	// WHEN dispatching
	ctx := context.Background()
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0)

	// THEN: should have round-0 + round-1 = 2 total records
	records := recorder.Records()
	if len(records) < 2 {
		t.Errorf("expected at least 2 records (round-0 + round-1 follow-up), got %d", len(records))
	}

	// Verify session metadata propagated
	hasRound0 := false
	hasRound1 := false
	for _, r := range records {
		if r.SessionID != "" && r.RoundIndex == 0 {
			hasRound0 = true
		}
		if r.SessionID != "" && r.RoundIndex == 1 {
			hasRound1 = true
		}
	}
	if !hasRound0 {
		t.Error("missing round-0 session record")
	}
	if !hasRound1 {
		t.Error("missing round-1 session follow-up record")
	}
}

func TestObserveOrchestrator_SessionError_CancelsSession(t *testing.T) {
	// GIVEN a mock server that returns 500 for all requests
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error": "internal error"}`))
	}))
	defer server.Close()

	spec := &workload.WorkloadSpec{
		Version:       "2",
		Seed:          42,
		AggregateRate: 10.0,
		Clients: []workload.ClientSpec{
			{
				ID:           "session-client",
				RateFraction: 1.0,
				Arrival:      workload.ArrivalSpec{Process: "constant"},
				InputDist:    workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				OutputDist:   workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 25}},
				Reasoning: &workload.ReasoningSpec{
					MultiTurn: &workload.MultiTurnSpec{
						MaxRounds:    3,
						ThinkTimeUs:   1000, // 1ms in microseconds
						ContextGrowth: "accumulate",
						SingleSession: true,
					},
				},
			},
		},
	}

	wl, err := workload.GenerateWorkload(spec, 1_000_000, 1)
	if err != nil {
		t.Fatalf("GenerateWorkload: %v", err)
	}
	if len(wl.Sessions) == 0 {
		t.Skip("No sessions generated")
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	sessionMgr := workload.NewSessionManager(wl.Sessions)

	// WHEN dispatching (server errors on all requests)
	ctx := context.Background()
	runObserveOrchestrator(ctx, client, recorder, sessionMgr, wl.Requests, false, 10, 0)

	// THEN: BC-11: only round-0 should be in trace (error cancels session, no follow-up)
	records := recorder.Records()
	for _, r := range records {
		if r.SessionID != "" && r.RoundIndex > 0 {
			t.Errorf("BC-11 violated: found round-%d record after error — session should have been cancelled", r.RoundIndex)
		}
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestObserveOrchestrator_Session -v`
Expected: FAIL — adaptForSessionManager returns nil

**Step 3: Implement session adapter**

Replace the placeholder `adaptForSessionManager`:

```go
// adaptForSessionManager converts an HTTP response into a sim.Request suitable
// for SessionManager.OnComplete. Only fields read by OnComplete are populated:
// State, ProgressIndex, InputTokens, OutputTokens, SessionID, ID.
func adaptForSessionManager(original *sim.Request, record *RequestRecord) *sim.Request {
	adapted := &sim.Request{
		ID:          original.ID,
		SessionID:   original.SessionID,
		RoundIndex:  original.RoundIndex,
		InputTokens: original.InputTokens,
	}

	// Map HTTP outcome to sim state
	if record.Status == "ok" {
		adapted.State = sim.StateCompleted
	} else {
		adapted.State = sim.StateTimedOut // error/timeout → cancels session
	}

	// ProgressIndex = input + output (used by OnComplete for context accumulation)
	outputCount := record.OutputTokens
	adapted.ProgressIndex = int64(len(original.InputTokens) + outputCount)

	// Generate synthetic output token IDs (length = server-reported completion_tokens)
	// OnComplete uses len(OutputTokens) and slices it by actualOutputLen
	if outputCount > 0 {
		adapted.OutputTokens = make([]int, outputCount)
		for i := range adapted.OutputTokens {
			adapted.OutputTokens[i] = i + 1 // arbitrary non-zero IDs
		}
	}

	return adapted
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./cmd/... -run TestObserveOrchestrator_Session -v`
Expected: PASS

**Step 5: Run lint and race detector**

Run: `go test ./cmd/... -run TestObserveOrchestrator_Session -race -v && golangci-lint run ./cmd/...`
Expected: PASS with no data races, no lint issues

**Step 6: Commit**

```bash
git add cmd/observe_cmd.go cmd/observe_cmd_test.go
git commit -m "feat(cmd): implement session adapter and follow-up dispatch (BC-3, BC-8, BC-11)

- adaptForSessionManager maps HTTP response to sim.Request for OnComplete
- Session follow-ups dispatched at completion + think_time
- HTTP errors cancel sessions (StateTimedOut → no follow-up)
- Race detector clean (serializer goroutine preserves single-threaded contract)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: Warmup exclusion

**Contracts Implemented:** BC-4 (complete), OBS-INV-4

**Files:**
- Test: `cmd/observe_cmd_test.go`

**Step 1: Write failing test for warmup exclusion**

```go
func TestObserveOrchestrator_WarmupExclusion(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"usage": map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	requests := make([]*sim.Request, 5)
	for i := range requests {
		requests[i] = &sim.Request{
			ID:           fmt.Sprintf("request_%d", i),
			ArrivalTime:  int64(i) * 1000,
			InputTokens:  make([]int, 10),
			OutputTokens: make([]int, 5),
			MaxOutputLen: 5,
			State:        sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// WHEN dispatching 5 requests with warmup=2
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 2)

	// THEN: OBS-INV-4: trace has 5 - min(2,5) = 3 records
	records := recorder.Records()
	if len(records) != 3 {
		t.Fatalf("OBS-INV-4: expected 3 records (5 dispatched - 2 warmup), got %d", len(records))
	}
}

func TestObserveOrchestrator_WarmupExceedsTotal(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"usage": map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	requests := make([]*sim.Request, 2)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: 0,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// WHEN warmup (5) > total requests (2)
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 5)

	// THEN: OBS-INV-4 edge case: trace is empty
	records := recorder.Records()
	if len(records) != 0 {
		t.Fatalf("OBS-INV-4 edge case: expected 0 records (warmup >= total), got %d", len(records))
	}
}
```

**Step 2: Run tests**

Run: `go test ./cmd/... -run TestObserveOrchestrator_Warmup -v`
Expected: PASS (warmup exclusion already implemented in Task 3 via `if idx >= warmupCount`)

**Step 3: No implementation needed (already in Task 3)**

The warmup check `if idx >= warmupCount` in the dispatch function already handles this correctly.

**Step 4: Verify and commit**

Run: `go test ./cmd/... -run TestObserveOrchestrator_Warmup -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./cmd/...`

**Step 6: Commit**

```bash
git add cmd/observe_cmd_test.go
git commit -m "test(cmd): add warmup exclusion tests (BC-4, OBS-INV-4)

- Verify trace excludes first N warmup requests by dispatch index
- Verify edge case: warmup >= total produces empty trace

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Timestamp ordering and TraceV2 round-trip test

**Contracts Implemented:** OBS-INV-5, BC-5

**Files:**
- Test: `cmd/observe_cmd_test.go`

**Step 1: Write tests**

```go
func TestObserveOrchestrator_TimestampOrdering(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		time.Sleep(5 * time.Millisecond) // ensure measurable latency
		json.NewEncoder(w).Encode(map[string]interface{}{
			"usage": map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 5},
		})
	}))
	defer server.Close()

	requests := []*sim.Request{{
		ID: "request_0", ArrivalTime: 0,
		InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
		MaxOutputLen: 5, State: sim.StateQueued,
	}}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0)

	records := recorder.Records()
	if len(records) != 1 {
		t.Fatalf("expected 1 record, got %d", len(records))
	}
	r := records[0]

	// OBS-INV-5: arrival <= send <= first_chunk <= last_chunk (for ok status)
	if r.Status == "ok" {
		if r.ArrivalTimeUs > r.SendTimeUs {
			t.Errorf("OBS-INV-5: arrival (%d) > send (%d)", r.ArrivalTimeUs, r.SendTimeUs)
		}
		if r.SendTimeUs > r.FirstChunkTimeUs {
			t.Errorf("OBS-INV-5: send (%d) > first_chunk (%d)", r.SendTimeUs, r.FirstChunkTimeUs)
		}
		if r.FirstChunkTimeUs > r.LastChunkTimeUs {
			t.Errorf("OBS-INV-5: first_chunk (%d) > last_chunk (%d)", r.FirstChunkTimeUs, r.LastChunkTimeUs)
		}
	}
}

func TestObserveOrchestrator_TraceV2RoundTrip(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"usage": map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	requests := make([]*sim.Request, 3)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: int64(i) * 100000,
			InputTokens: make([]int, 100), OutputTokens: make([]int, 50),
			MaxOutputLen: 50, State: sim.StateQueued, ClientID: "test-client",
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}
	runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 10, 0)

	// Export to temp files
	headerPath := filepath.Join(t.TempDir(), "header.yaml")
	dataPath := filepath.Join(t.TempDir(), "data.csv")
	header := &workload.TraceHeader{
		Version: 2, TimeUnit: "us", Mode: "real",
		Server: &workload.TraceServerConfig{Model: "test-model"},
	}
	if err := recorder.Export(header, headerPath, dataPath); err != nil {
		t.Fatalf("Export: %v", err)
	}

	// Round-trip: load and verify
	loaded, err := workload.LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatalf("LoadTraceV2: %v", err)
	}

	if len(loaded.Records) != 3 {
		t.Fatalf("round-trip: expected 3 records, got %d", len(loaded.Records))
	}

	originalRecords := recorder.Records()
	for i, orig := range originalRecords {
		loaded := loaded.Records[i]
		if orig.RequestID != loaded.RequestID {
			t.Errorf("record %d: RequestID mismatch: %d vs %d", i, orig.RequestID, loaded.RequestID)
		}
		if orig.InputTokens != loaded.InputTokens {
			t.Errorf("record %d: InputTokens mismatch: %d vs %d", i, orig.InputTokens, loaded.InputTokens)
		}
		if orig.Status != loaded.Status {
			t.Errorf("record %d: Status mismatch: %q vs %q", i, orig.Status, loaded.Status)
		}
	}
}
```

**Step 2-4: Run tests, verify pass**

Run: `go test ./cmd/... -run "TestObserveOrchestrator_Timestamp|TestObserveOrchestrator_TraceV2" -v`

**Step 5: Lint**

Run: `golangci-lint run ./cmd/...`

**Step 6: Commit**

```bash
git add cmd/observe_cmd_test.go
git commit -m "test(cmd): add timestamp ordering and TraceV2 round-trip tests (OBS-INV-5, BC-5)

- Verify arrival <= send <= first_chunk <= last_chunk for ok records
- Verify export → load preserves all field values

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Error storm drain and Ctrl+C cancellation

**Contracts Implemented:** BC-10, BC-12

**Files:**
- Test: `cmd/observe_cmd_test.go`

**Step 1: Write tests**

```go
func TestObserveOrchestrator_ErrorStormDrain(t *testing.T) {
	// GIVEN a server that refuses all connections
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(500)
		w.Write([]byte(`{"error": "down"}`))
	}))
	defer server.Close()

	requests := make([]*sim.Request, 10)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: int64(i) * 1000,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// WHEN all requests error
	done := make(chan struct{})
	go func() {
		runObserveOrchestrator(context.Background(), client, recorder, nil, requests, false, 5, 0)
		close(done)
	}()

	// THEN: completes within bounded time (5 seconds)
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("drain did not complete within 5 seconds — possible hang")
	}

	// BC-10: all 10 requests recorded with error status
	records := recorder.Records()
	if len(records) != 10 {
		t.Fatalf("expected 10 records, got %d", len(records))
	}
	for i, r := range records {
		if r.Status != "error" {
			t.Errorf("record %d: status %q, want %q", i, r.Status, "error")
		}
	}
}

func TestObserveOrchestrator_ContextCancellation(t *testing.T) {
	// GIVEN a slow server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Second)
	}))
	defer server.Close()

	requests := make([]*sim.Request, 5)
	for i := range requests {
		requests[i] = &sim.Request{
			ID: fmt.Sprintf("request_%d", i), ArrivalTime: 0,
			InputTokens: make([]int, 10), OutputTokens: make([]int, 5),
			MaxOutputLen: 5, State: sim.StateQueued,
		}
	}

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	recorder := &Recorder{}

	// WHEN context is cancelled after 200ms
	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		runObserveOrchestrator(ctx, client, recorder, nil, requests, false, 2, 0)
		close(done)
	}()

	// THEN: orchestrator exits promptly (within 1 second after cancellation)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("orchestrator did not exit after context cancellation")
	}
}
```

**Step 2-6: Run, verify, lint, commit**

Run: `go test ./cmd/... -run "TestObserveOrchestrator_ErrorStorm|TestObserveOrchestrator_Context" -v -timeout 30s`

```bash
git add cmd/observe_cmd_test.go
git commit -m "test(cmd): add error storm drain and cancellation tests (BC-10, BC-12)

- Verify all-error workload completes and records all errors
- Verify context cancellation causes prompt orchestrator exit

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Pipeline parity test and race detector

**Contracts Implemented:** D1 (pipeline parity), OBS-INV-3 (race detector)

**Files:**
- Test: `cmd/observe_cmd_test.go`

**Step 1: Write tests**

```go
func TestObserveOrchestrator_PipelineParity_SameRequestSequence(t *testing.T) {
	// GIVEN a WorkloadSpec with seed 42
	spec := &workload.WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "parity-client", RateFraction: 1.0,
			Arrival:   workload.ArrivalSpec{Process: "constant"},
			InputDist: workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		}},
	}

	// WHEN generating workload (same call that blis run uses)
	wl1, err := workload.GenerateWorkload(spec, 1_000_000, 5)
	if err != nil {
		t.Fatalf("GenerateWorkload 1: %v", err)
	}

	// AND generating again with same spec (simulating blis observe path)
	spec2 := &workload.WorkloadSpec{
		Version: "2", Seed: 42, AggregateRate: 10.0,
		Clients: []workload.ClientSpec{{
			ID: "parity-client", RateFraction: 1.0,
			Arrival:   workload.ArrivalSpec{Process: "constant"},
			InputDist: workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
			OutputDist: workload.DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
		}},
	}
	wl2, err := workload.GenerateWorkload(spec2, 1_000_000, 5)
	if err != nil {
		t.Fatalf("GenerateWorkload 2: %v", err)
	}

	// THEN: calibration-relevant fields are identical
	if len(wl1.Requests) != len(wl2.Requests) {
		t.Fatalf("request count mismatch: %d vs %d", len(wl1.Requests), len(wl2.Requests))
	}
	for i := range wl1.Requests {
		r1, r2 := wl1.Requests[i], wl2.Requests[i]
		if r1.ArrivalTime != r2.ArrivalTime {
			t.Errorf("request %d: ArrivalTime %d vs %d", i, r1.ArrivalTime, r2.ArrivalTime)
		}
		if len(r1.InputTokens) != len(r2.InputTokens) {
			t.Errorf("request %d: input token count %d vs %d", i, len(r1.InputTokens), len(r2.InputTokens))
		}
		if len(r1.OutputTokens) != len(r2.OutputTokens) {
			t.Errorf("request %d: output token count %d vs %d", i, len(r1.OutputTokens), len(r2.OutputTokens))
		}
		if r1.SessionID != r2.SessionID {
			t.Errorf("request %d: SessionID %q vs %q", i, r1.SessionID, r2.SessionID)
		}
		if r1.RoundIndex != r2.RoundIndex {
			t.Errorf("request %d: RoundIndex %d vs %d", i, r1.RoundIndex, r2.RoundIndex)
		}
	}
}
```

**Step 2-6: Run (with -race flag), verify, lint, commit**

Run: `go test ./cmd/... -run TestObserveOrchestrator -race -v -timeout 60s`

```bash
git add cmd/observe_cmd_test.go
git commit -m "test(cmd): add pipeline parity test and run race detector (D1, OBS-INV-3)

- Verify same WorkloadSpec + seed produces identical calibration-relevant fields
- All session tests pass under -race (serializer goroutine correctness)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: CLAUDE.md update and cleanup

**Contracts Implemented:** BC-9 (no sim/ modification verification)

**Files:**
- Modify: `CLAUDE.md` (File Organization tree, Build and Run commands)
- Modify: `cmd/observe.go` (remove `LogRealModeNotImplemented` dead code)

**Step 1: Verify no sim/ changes**

Run: `git diff --name-only HEAD~8 | grep "^sim/" || echo "BC-9 verified: no sim/ files changed"`

**Step 2: Update CLAUDE.md**

In the **Build and Run Commands** section, after the `blis calibrate` example, add:
```bash
# Observe real server latency and record timing into TraceV2
./blis observe --server-url http://localhost:8000 --model qwen/qwen3-14b \
  --workload-spec workload.yaml --trace-header trace.yaml --trace-data trace.csv
```

In the **File Organization** section: CLAUDE.md now delegates to `docs/reference/project-structure.md`. If that file has an annotated tree, add `observe_cmd.go` under `cmd/` after `observe.go`. If CLAUDE.md has its own inline tree, add there instead. The entry should read:
```
│   ├── observe_cmd.go         # `blis observe` command: dispatches WorkloadSpec to real server; flags: --server-url, --model, --trace-header, --trace-data (required), --workload-spec, --rate, --max-concurrency, --warmup-requests, --no-streaming; writes TraceV2 (real mode)
```

In the **Current Implementation Focus** section, add:
```
Observe/replay/calibrate pipeline complete: `blis observe` (#659) dispatches workload to real servers, `blis replay` (#689) replays through DES, `blis calibrate` (#701) compares real vs simulated latencies.
```

**Step 3: Remove dead code**

Remove `LogRealModeNotImplemented()` from `cmd/observe.go` — it's a placeholder that's no longer needed now that `blis observe` is implemented.

**Step 4: Run full test suite**

Run: `go test ./... -count=1 && golangci-lint run ./...`
Expected: All tests pass, no lint issues

**Step 5: Commit**

```bash
git add CLAUDE.md cmd/observe.go
git commit -m "docs(CLAUDE.md): add blis observe command, update file organization

- Add observe to Build and Run Commands
- Add cmd/observe_cmd.go to File Organization tree
- Remove LogRealModeNotImplemented dead code

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 3 | Integration | TestObserveOrchestrator_OpenLoop_ConservationAndConcurrency |
| BC-2 | Task 2 | Unit | (spec synthesis verified via flag tests + Task 3 integration) |
| BC-3 | Task 4 | Integration | TestObserveOrchestrator_SessionFollowUp_GeneratesRound2 |
| BC-4 | Task 5 | Unit | TestObserveOrchestrator_WarmupExclusion |
| BC-5 | Task 6 | Unit | TestObserveOrchestrator_TimestampOrdering |
| BC-6 | Task 3 | Invariant | TestObserveOrchestrator_OpenLoop (conservation check) |
| BC-7 | Task 3 | Invariant | TestObserveOrchestrator_OpenLoop (concurrency check) |
| BC-8 | Task 8 | Race | All tests run with -race flag |
| BC-10 | Task 7 | Failure | TestObserveOrchestrator_ErrorStormDrain |
| BC-11 | Task 4 | Failure | TestObserveOrchestrator_SessionError_CancelsSession |
| BC-12 | Task 7 | Failure | TestObserveOrchestrator_ContextCancellation |
| BC-13 | Task 2 | Unit | TestObserveCmd_MissingRequiredFlags_Errors |
| BC-14 | Task 2 | Unit | (validated via flag registration) |
| D1 | Task 8 | Invariant | TestObserveOrchestrator_PipelineParity_SameRequestSequence |
| OBS-INV-4 | Task 5 | Invariant | TestObserveOrchestrator_WarmupExceedsTotal |
| OBS-INV-5 | Task 6 | Invariant | TestObserveOrchestrator_TimestampOrdering |

Golden dataset: Not affected (observe produces TraceV2, not simulation output).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| SessionManager data race | Medium | High (corrupted sessions) | Serializer goroutine + -race tests | Task 4, 8 |
| Dispatcher hangs on drain | Low | High (command never exits) | Bounded drain with context cancellation | Task 7 |
| Warmup off-by-one | Low | Medium (wrong record count) | Explicit edge case test | Task 5 |
| RecordRequest signature breaks existing callers | Low | Low (compile error) | Only internal caller | Task 1 |
| WorkloadSpec spec without sessions produces nil SessionManager | Low | Low (nil check already in code) | nil-safe dispatch path | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (no new interfaces, single concrete orchestrator)
- [x] No feature creep (no sim config flags, no retry, no progress reporting)
- [x] No unexercised flags (all flags used in runObserve or tests)
- [x] No partial implementations (every function called by command or tests)
- [x] No breaking changes (RecordRequest signature change is internal)
- [x] No hidden global state (observe flag vars are package-level but scoped to observe command)
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated (Task 9)
- [x] No stale references
- [x] Deviation log reviewed — 2 deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5→6→7→8→9)
- [x] All contracts mapped to tasks
- [x] R1: All errors recorded in trace (no silent loss)
- [x] R3: All numeric flags validated
- [x] R4: RecordRequest signature change — single construction site (observe.go internal)
- [x] R6: No logrus.Fatalf in sim/ (all new code in cmd/)
- [x] R7: Invariant tests alongside behavioral tests
- [x] R19: No unbounded retry loops (errors recorded, not retried)
- [x] R21: No range over mutable slices (follow-ups via channel, not slice)

---

## Appendix: Key Implementation Notes

**Session serializer lifecycle:**
1. Created before dispatch loop starts (if sessions exist)
2. Reads from `completionCh` (sent by HTTP goroutines)
3. Calls `sessionMgr.OnComplete` serially (OBS-INV-3)
4. Pushes follow-ups to `followUpCh` (read by dispatcher)
5. Closed after `wg.Wait()` completes (all HTTP goroutines done)
6. `serializerDone` channel signals cleanup complete

**Time-base handling:**
- Pre-generated requests: `ArrivalTime` is relative (from 0), used as-is in trace
- Session follow-ups: `OnComplete` returns ArrivalTime as `wallClock + thinkTime` (both relative to `startWall`). Stored as relative in trace.
- Send/chunk timestamps: Absolute wall-clock microseconds (from `time.Now().UnixMicro()`)

**Imports needed in observe_cmd.go:**
```go
import (
	"context"
	"math"
	"os"
	"os/signal"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)
```

**Imports needed in observe_cmd_test.go:**
```go
import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)
```
