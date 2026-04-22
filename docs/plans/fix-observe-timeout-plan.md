# Fix Observe Streaming Timeout Silent Data Corruption — Implementation Plan

**Goal:** Fix `blis observe` to correctly mark timed-out streaming requests as `status=timeout` instead of silently recording them as `status=ok`, and expose the HTTP client timeout as a configurable CLI flag (`--timeout`).

**The problem today:** When a streaming request exceeds the 5-minute HTTP client timeout during `blis observe`, the SSE scanner encounters a deadline error but only logs a warning. The `record.Status` remains `"ok"` (its initial value), producing trace records with `status=ok`, `output_tokens=0`, and empty `finish_reason`. These corrupt downstream calibration and latency analysis because they are indistinguishable from successful zero-output responses. Additionally, the 5-minute timeout is hardcoded with no way to adjust it for workloads that legitimately take longer.

**What this PR adds:**
1. Streaming timeout detection: when `scanner.Err()` returns a deadline/timeout error, set `record.Status = "timeout"` and `record.ErrorMessage` to include the error details.
2. Non-streaming timeout detection: when `io.ReadAll` returns a deadline/timeout error, distinguish it from generic read errors by setting `record.Status = "timeout"`.
3. Configurable timeout: `--timeout` CLI flag (in seconds, default 300) to control the HTTP client timeout for `blis observe`.
4. Documentation updates for the new flag in the observe guide, CLAUDE.md, and CLI help text.

**Why this matters:** Silent timeouts corrupt sim2real calibration data. In the reported case, ~11% of requests were silently recorded as ok, inflating latency metrics and corrupting result comparisons.

**Architecture:** All changes are in `cmd/` (CLI layer). `cmd/observe.go` handles HTTP client construction and response parsing. `cmd/observe_cmd.go` registers CLI flags and constructs the client. No `sim/` changes needed.

**Source:** [Issue #1118](https://github.com/inference-sim/inference-sim/issues/1118)

**Closes:** `Fixes #1118`

**Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** `RealClient` HTTP client in `cmd/observe.go` — the observe pipeline's interface to real inference servers.
2. **Adjacent blocks:** `Recorder` (receives `RequestRecord`), `runObserveOrchestrator` (dispatches requests), `blis calibrate` (consumes trace output downstream).
3. **Invariants touched:** None of the 11 simulation invariants are affected. This is a data-fidelity fix in the observe pipeline (pre-simulation).
4. **Construction site audit:**
   - `RequestRecord` is constructed at one site: `observe.go:90-93` inside `Send()`.
   - `RealClient` is constructed at one site: `observe.go:41-48` inside `NewRealClient()`.
   - `http.Client` is constructed inline inside `NewRealClient()` at `observe.go:47`. The `--timeout` flag value must be threaded from `observe_cmd.go` → `NewRealClient()` via a new `WithHTTPTimeout` option.

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes a data-fidelity bug where HTTP client timeouts during SSE streaming are silently swallowed, producing `status=ok` records with `output_tokens=0`. The fix adds timeout detection in both streaming and non-streaming response handlers, and exposes the HTTP client timeout as a configurable CLI flag.

The change is entirely in `cmd/` (CLI layer). The streaming handler at `observe.go:330-332` currently only warns on `scanner.Err()`; this PR adds status/error-message updates. The non-streaming handler already handles read errors correctly but doesn't distinguish timeouts. A new `--timeout` flag threads through `NewRealClient` via the existing `RealClientOption` pattern.

### B) Behavioral Contracts

**Positive contracts:**

BC-1: Streaming timeout sets status
- GIVEN a streaming request to `blis observe`
- WHEN the HTTP client timeout fires during SSE scanning (scanner.Err() returns non-nil)
- THEN `record.Status` is `"timeout"` and `record.ErrorMessage` contains the error string
- MECHANISM: After `scanner.Err()` check in `handleStreamingResponse`, set status and error message fields.

BC-2: Non-streaming timeout sets status
- GIVEN a non-streaming request to `blis observe`
- WHEN the HTTP client timeout fires during `io.ReadAll` (returns a deadline-exceeded error)
- THEN `record.Status` is `"timeout"` and `record.ErrorMessage` contains the error string
- MECHANISM: In `handleNonStreamingResponse`, check if the `io.ReadAll` error is a timeout (via `os.IsTimeout` or `errors.Is(err, context.DeadlineExceeded)`) and set `"timeout"` instead of generic `"error"`.

BC-3: Configurable timeout flag
- GIVEN a user running `blis observe`
- WHEN `--timeout 600` is passed
- THEN the HTTP client uses a 600-second timeout instead of the default 300 seconds
- MECHANISM: `WithHTTPTimeout` option applied in `NewRealClient`, timeout value from CLI flag.

BC-4: Default timeout unchanged
- GIVEN a user running `blis observe` without `--timeout`
- WHEN the default is used
- THEN the HTTP client timeout is 300 seconds (5 minutes), matching the pre-fix behavior

BC-5: Timeout flag validation
- GIVEN a user running `blis observe`
- WHEN `--timeout` is set to 0, a negative value, or greater than 86400
- THEN `blis observe` exits with a fatal error message

BC-6: Partial timestamps preserved on timeout
- GIVEN a streaming request that received some SSE chunks before timeout
- WHEN the timeout fires
- THEN `record.FirstChunkTimeUs` and `record.LastChunkTimeUs` retain whatever values were captured before the timeout, and `record.NumChunks` reflects partial data received

**Negative contracts:**

BC-7: Successful requests unaffected
- GIVEN a request that completes successfully
- WHEN the response is fully received
- THEN `record.Status` remains `"ok"` (no change in behavior for non-timeout cases)

BC-8: HTTP-level timeout detection
- GIVEN a request where the HTTP round-trip itself exceeds the timeout (before any response headers arrive)
- WHEN `httpClient.Do` returns a timeout error
- THEN `record.Status` is set to `"timeout"` and `record.ErrorMessage` contains the timeout error (not generic `"error"`)

### C) Component Interaction

```
CLI flag (--timeout) ──► runObserve() ──► NewRealClient(..., WithHTTPTimeout(d))
                                                │
                                                ▼
                                          http.Client{Timeout: d}
                                                │
                                                ▼
                                            Send()
                                           /       \
                                    streaming    non-streaming
                                         │             │
                                         ▼             ▼
                              handleStreamingResponse  handleNonStreamingResponse
                              scanner.Err() != nil     io.ReadAll err != nil
                                         │             │
                                         ▼             ▼
                              status="timeout"     status="timeout" (if deadline)
                              errMsg=err.Error()   status="error" (if other)
                                         │             │
                                         └──────┬──────┘
                                                ▼
                                          RequestRecord
                                                │
                                                ▼
                                          Recorder ──► TraceV2 CSV
```

State ownership: `RequestRecord` is created and owned by `Send()`. Status is set to `"ok"` at construction and updated only within `Send()` and its handler sub-functions.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Issue suggests `"timeout"` or `"error"` for scanner errors | Uses `isTimeoutError()` helper to distinguish timeout from other errors in both streaming and non-streaming paths | CLARIFICATION — Scanner errors include non-timeout failures (buffer overflow, connection reset, TLS errors). The helper checks both `os.IsTimeout()` and `errors.Is(err, context.DeadlineExceeded)` for robust detection. |
| Issue mentions non-streaming path at line 233 "should distinguish timeout from other read errors" | Plan implements this distinction using Go's `os.IsTimeout()` | ADDITION — Improves non-streaming error fidelity as suggested. |
| Issue does not mention configurable timeout flag | Plan adds `--timeout` CLI flag | ADDITION — User requested this in conversation; prevents the bug from recurring with different workloads. |

### E) Review Guide

**Scrutinize:** The `handleStreamingResponse` fix (BC-1) — ensure status is set *after* the scanner loop so partial timestamps are preserved (BC-6). The `handleNonStreamingResponse` timeout detection (BC-2) — ensure `os.IsTimeout()` correctly identifies Go HTTP client deadline errors.

**Safe to skim:** CLI flag wiring (BC-3/4/5) follows the exact same pattern as `WithAPIFormat`. Documentation updates.

**Known debt:** None introduced. This PR reduces existing debt (the silent corruption).

---

## Part 2: Executable Implementation

### F) Implementation Overview

Files to modify:
- `cmd/observe.go` — Fix streaming handler, improve non-streaming handler, add `WithHTTPTimeout` option
- `cmd/observe_cmd.go` — Add `--timeout` flag, validate, wire to `NewRealClient`
- `cmd/observe_test.go` — Tests for BC-1 through BC-7
- `docs/guide/observe-replay-calibrate.md` — Document `--timeout` flag in Optional Flags table
- `CLAUDE.md` — Update observe examples and Recent Changes

No new files. No dead code. No new packages.

### G) Task Breakdown

#### Task 1: Fix streaming timeout status (BC-1, BC-6)

**Contracts:** BC-1 (streaming timeout sets status), BC-6 (partial timestamps preserved)

**Files:** modify `cmd/observe.go`, test `cmd/observe_test.go`

**Test:**

```go
// TestRealClient_Streaming_Timeout_SetsTimeoutStatus verifies BC-1: when the
// HTTP client timeout fires during SSE streaming, record.Status is "timeout"
// and record.ErrorMessage contains error details, not silent "ok".
func TestRealClient_Streaming_Timeout_SetsTimeoutStatus(t *testing.T) {
	// Server sends one SSE chunk then hangs until client timeout.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		w.Header().Set("Content-Type", "text/event-stream")
		// Send one chunk so FirstChunkTimeUs is set (BC-6)
		_, _ = fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n")
		flusher.Flush()
		// Hang until client gives up
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	// Use a very short timeout to make the test fast
	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithHTTPTimeout(200*time.Millisecond))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: true,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "timeout" {
		t.Errorf("Status = %q, want %q", record.Status, "timeout")
	}
	if record.ErrorMessage == "" {
		t.Error("ErrorMessage should contain timeout error details")
	}
	// BC-6: partial timestamps preserved
	if record.FirstChunkTimeUs == 0 {
		t.Error("FirstChunkTimeUs should be set from the chunk received before timeout")
	}
	if record.NumChunks != 1 {
		t.Errorf("NumChunks = %d, want 1 (one chunk before timeout)", record.NumChunks)
	}
}
```

**Impl:**

In `cmd/observe.go`:

1. Add `"os"` and `"errors"` to the import block.

2. Add the `isTimeoutError` helper and `WithHTTPTimeout` option after the existing `WithAPIFormat`:

```go
// WithHTTPTimeout sets the HTTP client timeout for requests.
func WithHTTPTimeout(d time.Duration) RealClientOption {
	return func(c *RealClient) { c.httpClient.Timeout = d }
}

// isTimeoutError returns true if err is a timeout or deadline-exceeded error.
func isTimeoutError(err error) bool {
	if os.IsTimeout(err) {
		return true
	}
	return errors.Is(err, context.DeadlineExceeded)
}
```

3. Change `handleStreamingResponse` at lines 330-332 from:

```go
if err := scanner.Err(); err != nil {
	logrus.Warnf("observe: request %d: SSE scanner error: %v", record.RequestID, err)
}
```

to:

```go
if err := scanner.Err(); err != nil {
	if isTimeoutError(err) {
		record.Status = "timeout"
		record.ErrorMessage = fmt.Sprintf("streaming timeout: %v", err)
	} else {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("streaming error: %v", err)
	}
	logrus.Warnf("observe: request %d: SSE scanner error: %v", record.RequestID, err)
}
```

**Verify:** `cd .worktrees/fix-observe-timeout && go test ./cmd/... -run TestRealClient_Streaming_Timeout`
**Lint:** `cd .worktrees/fix-observe-timeout && golangci-lint run ./cmd/...`
**Commit:** `fix(observe): set status=timeout on streaming scanner error (BC-1, BC-6)`

---

#### Task 2: Fix non-streaming timeout status (BC-2)

**Contracts:** BC-2 (non-streaming timeout sets status), BC-8 (HTTP-level timeout detection)

**Files:** modify `cmd/observe.go`, test `cmd/observe_test.go`

**Test:**

```go
// TestRealClient_NonStreaming_Timeout_SetsTimeoutStatus verifies BC-2: when the
// HTTP client timeout fires during response body read, record.Status is "timeout".
func TestRealClient_NonStreaming_Timeout_SetsTimeoutStatus(t *testing.T) {
	// Server sends partial body then hangs.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		// Write partial data and flush, then hang
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("expected http.Flusher")
		}
		_, _ = w.Write([]byte(`{"choices":[{"text":"hel`))
		flusher.Flush()
		time.Sleep(5 * time.Second)
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm",
		WithHTTPTimeout(200*time.Millisecond))
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 0, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "timeout" {
		t.Errorf("Status = %q, want %q", record.Status, "timeout")
	}
	if record.ErrorMessage == "" {
		t.Error("ErrorMessage should contain timeout error details")
	}
}
```

**Impl:**

In `cmd/observe.go`, change `handleNonStreamingResponse` at lines 240-243 from:

```go
if err != nil {
	record.Status = "error"
	record.ErrorMessage = fmt.Sprintf("read error: %v", err)
	return record, nil
}
```

to:

```go
if err != nil {
	if isTimeoutError(err) {
		record.Status = "timeout"
		record.ErrorMessage = fmt.Sprintf("read timeout: %v", err)
	} else {
		record.Status = "error"
		record.ErrorMessage = fmt.Sprintf("read error: %v", err)
	}
	return record, nil
}
```

(`isTimeoutError` and imports were added in Task 1.)

**Verify:** `cd .worktrees/fix-observe-timeout && go test ./cmd/... -run TestRealClient_NonStreaming_Timeout`
**Lint:** `cd .worktrees/fix-observe-timeout && golangci-lint run ./cmd/...`
**Commit:** `fix(observe): distinguish timeout from error in non-streaming handler (BC-2)`

---

#### Task 3: Add --timeout CLI flag (BC-3, BC-4, BC-5)

**Contracts:** BC-3 (configurable timeout), BC-4 (default unchanged), BC-5 (validation)

**Files:** modify `cmd/observe_cmd.go`, test `cmd/observe_test.go`

**Test:**

```go
// TestObserveTimeoutFlagDefault verifies BC-4: default timeout is 300 seconds.
func TestObserveTimeoutFlagDefault(t *testing.T) {
	if observeTimeout != 300 {
		t.Errorf("default observeTimeout = %d, want 300", observeTimeout)
	}
}
```

For BC-5 validation, add a table-driven test for the validation logic.

**Impl:**

In `cmd/observe_cmd.go`:

1. Add variable: `observeTimeout int` in the `var` block.
2. Register flag in `init()`: `observeCmd.Flags().IntVar(&observeTimeout, "timeout", 300, "HTTP request timeout in seconds (per request)")`.
3. Add validation in `runObserve()` after existing numeric validations:
   ```go
   if observeTimeout <= 0 || observeTimeout > 86400 {
       logrus.Fatalf("--timeout must be between 1 and 86400 (1 day), got %d", observeTimeout)
   }
   ```
4. Wire to client construction: change the `NewRealClient` call to include `WithHTTPTimeout(time.Duration(observeTimeout) * time.Second)`.

**Verify:** `cd .worktrees/fix-observe-timeout && go test ./cmd/... -run TestObserveTimeout`
**Lint:** `cd .worktrees/fix-observe-timeout && golangci-lint run ./cmd/...`
**Commit:** `feat(observe): add --timeout flag for configurable HTTP timeout (BC-3, BC-4, BC-5)`

---

#### Task 4: Verify successful requests unaffected (BC-7)

**Contracts:** BC-7 (successful requests unaffected)

**Files:** test `cmd/observe_test.go` (no new test needed — existing tests cover this)

**Verification:** Run all existing observe tests to confirm no regression.

**Verify:** `cd .worktrees/fix-observe-timeout && go test ./cmd/... -run TestRealClient -v`
**Lint:** `cd .worktrees/fix-observe-timeout && golangci-lint run ./cmd/...`
**Commit:** (no commit — verification only)

---

#### Task 5: Documentation updates

**Contracts:** All (documentation)

**Files:** modify `docs/guide/observe-replay-calibrate.md`, modify `CLAUDE.md`

**Impl:**

In `docs/guide/observe-replay-calibrate.md`, add `--timeout` to the Optional Flags table:

```markdown
| `--timeout` | `int` | `300` | HTTP request timeout in seconds (per request); increase for slow servers or large-prefill workloads |
```

In `CLAUDE.md`, update the `## Recent Changes` section to add a bullet for this fix. Also add `--timeout` to the observe command example.

**Verify:** Visual inspection of docs.
**Lint:** N/A (markdown)
**Commit:** `docs(observe): document --timeout flag and streaming timeout fix`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 1 | Unit | `TestRealClient_Streaming_Timeout_SetsTimeoutStatus` |
| BC-2 | Task 2 | Unit | `TestRealClient_NonStreaming_Timeout_SetsTimeoutStatus` |
| BC-3 | Task 3 | Unit | wired via `WithHTTPTimeout` option (tested indirectly by Tasks 1, 2) |
| BC-4 | Task 3 | Unit | `TestObserveTimeoutFlagDefault` |
| BC-5 | Task 3 | Unit | validated in `runObserve()` — tested via existing flag validation pattern |
| BC-6 | Task 1 | Unit | assertions within `TestRealClient_Streaming_Timeout_SetsTimeoutStatus` |
| BC-7 | Task 4 | Regression | All existing `TestRealClient_*` tests |
| BC-8 | Task 2 | Unit | `TestRealClient_HTTPLevel_Timeout_SetsTimeoutStatus` |

**Invariant tests:** N/A — this PR does not affect simulation invariants. The tests verify observable behavior (status field values) not internal structure.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| `os.IsTimeout()` doesn't match Go HTTP client deadline errors | Low | Medium | Go's `net/http` timeout wraps the error in a `*url.Error` which implements `Timeout() bool`, so `os.IsTimeout()` works. Test verifies with real httptest server. | Task 2 |
| Changing streaming error handling from "ok" to "timeout" breaks downstream tools that filter on status | Low | Low | This is the intended fix — downstream tools should not treat timed-out requests as successful. The status field already documents `"timeout"` as a valid value (observe.go:78). | Task 1 |
| Short timeout in tests causes flakiness | Medium | Low | Tests use 200ms timeout with servers that sleep 5s — 25x margin. | Tasks 1, 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions.
- [x] No feature creep beyond PR scope.
- [x] No unexercised flags or interfaces.
- [x] No partial implementations.
- [x] No breaking changes without explicit contract updates.
- [x] No hidden global state impact.
- [x] All new code will pass golangci-lint.
- [x] Shared test helpers used from existing shared test package (installLogHook, testLogHook).
- [x] CLAUDE.md updated: new `--timeout` flag documented.
- [x] No stale references left in CLAUDE.md.
- [x] Documentation DRY: No canonical source modified.
- [x] Deviation log reviewed — all deviations justified.
- [x] Each task produces working, testable code.
- [x] Task dependencies correctly ordered (Task 1 adds `WithHTTPTimeout`, Task 2 uses it, Task 3 adds CLI flag).
- [x] All contracts mapped to tasks.
- [x] Construction site audit completed — `RequestRecord{Status: "ok"}` at one site (observe.go:90), `RealClient` at one site (observe.go:41).
- [x] R1: No silent `continue`/`return` dropping data — this PR fixes one.
- [x] R3: `--timeout` validated > 0 in `runObserve()`.
- [x] R4: No new struct fields added. `WithHTTPTimeout` modifies existing `httpClient.Timeout`.

---

## Appendix: File-Level Implementation Details

### File: `cmd/observe.go`

**Purpose:** Fix timeout handling in streaming/non-streaming response handlers; add `WithHTTPTimeout` option.

**Changes:**

1. Add `WithHTTPTimeout` option function (follows `WithAPIFormat` pattern):
   ```go
   // WithHTTPTimeout sets the HTTP client timeout for requests.
   func WithHTTPTimeout(d time.Duration) RealClientOption {
       return func(c *RealClient) { c.httpClient.Timeout = d }
   }
   ```

2. Add `isTimeoutError` helper:
   ```go
   // isTimeoutError returns true if err is a timeout or deadline-exceeded error.
   func isTimeoutError(err error) bool {
       if os.IsTimeout(err) {
           return true
       }
       return errors.Is(err, context.DeadlineExceeded)
   }
   ```

3. Fix `Send()` httpClient.Do error (line 170-173) — distinguish timeout from other HTTP errors:
   ```go
   resp, err := c.httpClient.Do(httpReq)
   if err != nil {
       if isTimeoutError(err) {
           record.Status = "timeout"
           record.ErrorMessage = fmt.Sprintf("HTTP timeout: %v", err)
       } else {
           record.Status = "error"
           record.ErrorMessage = fmt.Sprintf("HTTP error: %v", err)
       }
       return record, nil
   }
   ```

4. Fix `handleStreamingResponse` (line 330-332) — set status on scanner error:
   ```go
   if err := scanner.Err(); err != nil {
       if isTimeoutError(err) {
           record.Status = "timeout"
           record.ErrorMessage = fmt.Sprintf("streaming timeout: %v", err)
       } else {
           record.Status = "error"
           record.ErrorMessage = fmt.Sprintf("streaming error: %v", err)
       }
       logrus.Warnf("observe: request %d: SSE scanner error: %v", record.RequestID, err)
   }
   ```

4. Fix `handleNonStreamingResponse` (line 240-243) — distinguish timeout:
   ```go
   if err != nil {
       if isTimeoutError(err) {
           record.Status = "timeout"
           record.ErrorMessage = fmt.Sprintf("read timeout: %v", err)
       } else {
           record.Status = "error"
           record.ErrorMessage = fmt.Sprintf("read error: %v", err)
       }
       return record, nil
   }
   ```

5. Add `"os"` and `"errors"` to imports.

### File: `cmd/observe_cmd.go`

**Purpose:** Add `--timeout` flag, validate, and wire to client construction.

**Changes:**

1. Add `observeTimeout int` variable.
2. Register flag: `observeCmd.Flags().IntVar(&observeTimeout, "timeout", 300, "HTTP request timeout in seconds (per request)")`.
3. Validate: `if observeTimeout <= 0 || observeTimeout > 86400 { logrus.Fatalf(...) }`.
4. Wire: add `WithHTTPTimeout(time.Duration(observeTimeout) * time.Second)` to the `NewRealClient` call.

### File: `cmd/observe_test.go`

**Purpose:** Add tests for BC-1, BC-2, BC-6.

**Changes:** Two new test functions as specified in Tasks 1 and 2.

### File: `docs/guide/observe-replay-calibrate.md`

**Purpose:** Document the `--timeout` flag.

**Changes:** Add row to Optional Flags table.

### File: `CLAUDE.md`

**Purpose:** Document the fix and new flag in Recent Changes.

**Changes:** Add bullet to Recent Changes section.
