# PR655: Fix observe.go Measurement Semantics Bugs

- **Goal:** Fix five measurement bugs in `cmd/observe.go` so `blis observe` produces accurate timing and token count data when communicating with real inference servers.
- **The problem today:** The `RealClient` HTTP client has a dummy prompt (~4 tokens regardless of `InputTokens`), collapses TTFT with E2E for non-streaming, ignores server-reported `prompt_tokens`, hardcodes `max_tokens: 2048`, and lacks per-chunk ITL tracking. These bugs make observed data unusable for simulator calibration.
- **What this PR adds:**
  1. Proportional prompt generation (~N tokens for N `InputTokens`)
  2. Accurate non-streaming TTFT via first-byte reader wrapper
  3. Server `prompt_tokens` extraction into `RequestRecord.ServerInputTokens`
  4. Per-request `MaxOutputTokens` flowing through to HTTP body
  5. TODO comment for deferred per-chunk ITL (Bug 5)
  6. `Model` field wiring from `PendingRequest` to `TraceRecord`
- **Why this matters:** `blis observe` is the bridge between real server measurements and simulator replay. Corrupted measurements make calibration impossible.
- **Architecture:** All changes in `cmd/observe.go` (HTTP client) and `cmd/observe_test.go`. Adds a small `firstByteReader` wrapper type (unexported, single-purpose). Wires existing `TraceRecord` fields from PR #653.
- **Source:** GitHub issue #655
- **Closes:** `Fixes #655`
- **Behavioral Contracts:** See Part 1, Section B.

---

## Phase 0: Component Context

1. **Building block:** `RealClient` HTTP client in `cmd/observe.go`
2. **Adjacent blocks:** `workload.TraceRecord` (output schema), `Recorder` (trace aggregation), OpenAI-compatible server (external)
3. **Invariants touched:** None of the core simulator invariants (INV-1 through INV-9). This is CLI-layer code.
4. **Construction site audit:**
   - `PendingRequest{}`: 4 existing sites in `cmd/observe_test.go` (lines 25, 65, 96, 117) + 7 new sites added by this PR's tests (Tasks 1-4). All existing sites omit the new fields (`MaxOutputTokens`, `Model`), which default to zero — this is intentional (backward-compat: 0 triggers default 2048; empty Model is fine).
   - `RequestRecord{}`: 2 existing sites — `cmd/observe.go:62` (canonical, `ServerInputTokens` zero-valued, set later by response parsing), `cmd/observe_test.go:118` (test mock, zero-valued fields fine for concurrency test) + 1 new site in Task 4 test (`ServerInputTokens: 42` explicitly set). All use named fields; new field zero-values are backward-compatible.

---

## Part 1: Design Validation

### A) Executive Summary

This PR fixes five measurement bugs in `cmd/observe.go`, the HTTP client that sends requests to real inference servers. The fixes are mechanical: replace a dummy prompt with proportional token repetition, wrap the response body reader to capture first-byte timing, extract `prompt_tokens` from server responses, plumb per-request `MaxOutputTokens` into the HTTP body, and wire `Model` through to `TraceRecord`. Bug 5 (per-chunk ITL) is deferred with a TODO comment. All changes are in `cmd/` (CLI layer, not `sim/` library).

### B) Behavioral Contracts

**Positive contracts:**

```
BC-1: Proportional Prompt
- GIVEN a PendingRequest with InputTokens = N (N > 0)
- WHEN Send() constructs the HTTP request body
- THEN the "prompt" field contains N repetitions of a common word
- MECHANISM: strings.Repeat("hello ", N) produces ~N tokens with most tokenizers (best-effort;
  actual token count varies by tokenizer). ServerInputTokens (BC-3) provides the ground truth.
```

```
BC-2: Non-Streaming HTTP First-Byte Timing
- GIVEN a non-streaming request to a server
- WHEN the server sends the response body
- THEN FirstChunkTimeUs captures when the first response byte is received by the client
  AND LastChunkTimeUs captures when the full body read completes
  AND FirstChunkTimeUs <= LastChunkTimeUs
- MECHANISM: firstByteReader wrapper captures timestamp after first Read() returns n > 0
- LIMITATION: For non-streaming HTTP, real servers send the entire response after generation
  completes, so FirstChunkTimeUs approximates "time server finished + network transfer started,"
  not "time first token was generated." True TTFT is only measurable in streaming mode.
```

```
BC-3: Server Input Token Extraction
- GIVEN a server response containing usage.prompt_tokens
- WHEN the response is parsed (streaming or non-streaming)
- THEN RequestRecord.ServerInputTokens equals the server-reported prompt_tokens value
  AND Recorder maps it to TraceRecord.ServerInputTokens
```

```
BC-4: Per-Request MaxOutputTokens
- GIVEN a PendingRequest with MaxOutputTokens = M
- WHEN Send() constructs the HTTP request body
- THEN the "max_tokens" field equals M if M > 0, or 2048 if M <= 0
```

```
BC-5: Model Field Wiring
- GIVEN a PendingRequest with Model = "test-model"
- WHEN Recorder.RecordRequest() maps the result
- THEN TraceRecord.Model equals "test-model"
```

**Error handling contracts:**

```
BC-6: Zero InputTokens Guard
- GIVEN a PendingRequest with InputTokens <= 0
- WHEN Send() constructs the prompt
- THEN the prompt contains at least one word (not empty)
```

### C) Component Interaction

```
PendingRequest (cmd/)
  ├── InputTokens  → prompt construction (BC-1, BC-6)
  ├── MaxOutputTokens → max_tokens field (BC-4)
  ├── Model → TraceRecord.Model (BC-5)
  └── Streaming → streaming/non-streaming path selection

RealClient.Send()
  ├── Constructs HTTP body with prompt, max_tokens, model
  ├── Sends to OpenAI-compatible server
  └── Delegates to:
      ├── handleNonStreamingResponse()
      │   └── firstByteReader wraps resp.Body (BC-2)
      │   └── Extracts prompt_tokens + completion_tokens (BC-3)
      └── handleStreamingResponse()
          └── Extracts prompt_tokens from final chunk usage (BC-3)

RequestRecord (cmd/)
  ├── ServerInputTokens ← server prompt_tokens (BC-3)
  ├── FirstChunkTimeUs ← firstByteReader or first SSE chunk
  └── LastChunkTimeUs ← after full read or last SSE chunk

Recorder.RecordRequest()
  └── Maps RequestRecord → workload.TraceRecord
      ├── ServerInputTokens (BC-3)
      └── Model (BC-5)
```

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| Bug 5 needs full per-chunk ITL | Adds TODO comment only | DEFERRAL — requires new schema support per issue |
| Issue mentions "verify via server's prompt_tokens" | Test uses mock server prompt_tokens | SIMPLIFICATION — real server validation is integration test scope |
| Issue scope is 5 bugs | Plan adds Model field wiring (item 6) | ADDITION — partially resolves PR #653 TODO; TraceRecord.Model/ServerInputTokens fields already exist |
| N/A | Plan depends on PR #653 being merged | DEPENDENCY — TraceRecord schema fields (Model, DeadlineUs, ServerInputTokens) from #653 already on main |
| Silent `continue` on malformed SSE chunks (pre-existing) | Task 4 adds `logrus.Debugf` for the error | ADDITION — partial R1 improvement; debug-level logging for pre-existing silent data path |

### E) Review Guide

**Tricky part:** The `firstByteReader` wrapper (BC-2) — captures timestamp after `Read()` returns with `n > 0`, correctly measuring "first byte received." Handles partial-read + error case safely (timestamp fires when data arrives regardless of error). The `n > 0` guard avoids capturing time on empty reads.

**Scrutinize:** The `RecordRequest` field mapping — ensure all new fields (`ServerInputTokens`, `Model`) are wired correctly. The TODO comment at line 198-202 should be updated to reflect which fields are now wired.

**Safe to skim:** Prompt generation (BC-1) is straightforward string repetition. max_tokens plumbing (BC-4) is a simple field pass-through.

**Known debt:**
- `DeadlineUs` wiring remains as TODO (PendingRequest doesn't carry deadline info yet).
- `MaxOutputTokens` and `Model` on `PendingRequest`: no production caller sets these yet — `blis observe` is still a placeholder (`LogRealModeNotImplemented`). These fields are wired for when the observe command is fully integrated (#659).
- Pre-existing: streaming SSE `json.Unmarshal` errors silently `continue`. This PR adds `logrus.Debugf` as a partial R1 improvement (see deviation log).
- R23 note: `handleNonStreamingResponse` and `handleStreamingResponse` both extract `prompt_tokens` — same pattern, confirmed parity.
- **Server compatibility gaps** tracked in #660 comment: `stream_options` for streaming usage, `/v1/chat/completions` endpoint, `finish_reason` extraction.
- **Robustness hardening** tracked in #679: `scanner.Err()`, unbounded `InputTokens`, `completion_tokens` type assertion debug log, non-streaming TTFT semantics documentation, prompt tokenization accuracy, error body limit.

---

## Part 2: Executable Implementation

### F) Implementation Overview

| File | Action | Purpose |
|------|--------|---------|
| `cmd/observe.go` | Modify | Add fields, fix prompt, add firstByteReader, extract prompt_tokens, wire Model |
| `cmd/observe_test.go` | Modify | Update existing tests, add new tests for all contracts |

**Key decisions:**
- `firstByteReader` is unexported and local to `cmd/` — no cross-package exposure
- Default `MaxOutputTokens = 2048` preserves backward compatibility
- `strings.Repeat("hello ", N)` chosen for tokenizer-agnostic proportionality

### G) Task Breakdown

#### Task 1: Add struct fields + MaxOutputTokens plumbing (BC-4, BC-6)

**Contracts:** BC-4, BC-6

**Test (write first):**

```go
// cmd/observe_test.go — new test
func TestRealClient_MaxOutputTokens_FlowsThrough(t *testing.T) {
    var capturedBody map[string]interface{}
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        _ = json.NewDecoder(r.Body).Decode(&capturedBody)
        resp := map[string]interface{}{
            "choices": []map[string]interface{}{{"text": "ok"}},
            "usage":   map[string]interface{}{"prompt_tokens": 10.0, "completion_tokens": 5.0},
        }
        w.Header().Set("Content-Type", "application/json")
        _ = json.NewEncoder(w).Encode(resp)
    }))
    defer server.Close()

    client := NewRealClient(server.URL, "", "test-model", "vllm")

    // Explicit MaxOutputTokens
    _, _ = client.Send(context.Background(), &PendingRequest{
        RequestID: 0, InputTokens: 10, MaxOutputTokens: 512,
    })
    if got := int(capturedBody["max_tokens"].(float64)); got != 512 {
        t.Errorf("max_tokens = %d, want 512", got)
    }

    // Zero MaxOutputTokens → default 2048
    _, _ = client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 10, MaxOutputTokens: 0,
    })
    if got := int(capturedBody["max_tokens"].(float64)); got != 2048 {
        t.Errorf("max_tokens = %d, want 2048 (default)", got)
    }
}
```

**Run:** `go test ./cmd/... -run TestRealClient_MaxOutputTokens -v` → FAIL (field doesn't exist)

**Implement:**

`cmd/observe.go` — Add `MaxOutputTokens int` and `Model string` to `PendingRequest`. Add `ServerInputTokens int` to `RequestRecord`. Update `Send()`:
```go
maxTokens := req.MaxOutputTokens
if maxTokens < 0 {
    logrus.Warnf("PendingRequest.MaxOutputTokens is negative (%d), using default 2048", maxTokens)
}
if maxTokens <= 0 {
    maxTokens = 2048
}
body := map[string]interface{}{
    "model":      c.modelName,
    "max_tokens": maxTokens,
    "stream":     req.Streaming,
}
```

**Run:** `go test ./cmd/... -run TestRealClient_MaxOutputTokens -v` → PASS

**Lint:** `golangci-lint run ./cmd/...`

**Commit:** `git add cmd/observe.go cmd/observe_test.go && git commit -m "feat(observe): add MaxOutputTokens, Model, ServerInputTokens fields (BC-4)"`

---

#### Task 2: Proportional prompt generation (BC-1, BC-6)

**Contracts:** BC-1, BC-6

**Test (write first):**

```go
// cmd/observe_test.go — new test
func TestRealClient_ProportionalPrompt(t *testing.T) {
    var capturedBody map[string]interface{}
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        _ = json.NewDecoder(r.Body).Decode(&capturedBody)
        resp := map[string]interface{}{
            "choices": []map[string]interface{}{{"text": "ok"}},
            "usage":   map[string]interface{}{"prompt_tokens": 50.0, "completion_tokens": 5.0},
        }
        w.Header().Set("Content-Type", "application/json")
        _ = json.NewEncoder(w).Encode(resp)
    }))
    defer server.Close()

    client := NewRealClient(server.URL, "", "test-model", "vllm")
    _, _ = client.Send(context.Background(), &PendingRequest{
        RequestID: 0, InputTokens: 50,
    })
    prompt, ok := capturedBody["prompt"].(string)
    if !ok {
        t.Fatal("prompt not found in request body")
    }
    // Count "hello " repetitions
    count := strings.Count(prompt, "hello ")
    if count != 50 {
        t.Errorf("prompt contains %d 'hello ' repetitions, want 50", count)
    }

    // BC-6: Zero InputTokens guard — prompt must not be empty
    _, _ = client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 0,
    })
    prompt, ok = capturedBody["prompt"].(string)
    if !ok || !strings.Contains(prompt, "hello") {
        t.Errorf("zero InputTokens should produce at least one 'hello', got %q", prompt)
    }

    // BC-6: Negative InputTokens — should also produce at least one "hello" (with warning log)
    _, _ = client.Send(context.Background(), &PendingRequest{
        RequestID: 2, InputTokens: -5,
    })
    prompt, ok = capturedBody["prompt"].(string)
    if !ok || !strings.Contains(prompt, "hello") {
        t.Errorf("negative InputTokens should produce at least one 'hello', got %q", prompt)
    }
    // Note: logrus.Warnf fires for negative values but we don't assert on log output
    // (testing log output is structural, not behavioral)
}
```

**Run:** `go test ./cmd/... -run TestRealClient_ProportionalPrompt -v` → FAIL

**Implement:**

`cmd/observe.go` — Replace prompt line:
```go
inputTokens := req.InputTokens
if inputTokens < 0 {
    logrus.Warnf("PendingRequest.InputTokens is negative (%d), using 1 for prompt generation", inputTokens)
}
if inputTokens <= 0 {
    inputTokens = 1
}
// Note: req.InputTokens is NOT mutated — the trace record preserves the original requested value.
// Only the local inputTokens variable is clamped for prompt construction.
// Note: for very large InputTokens (e.g., 128K), this creates a ~768KB string.
// Acceptable for observe mode's typical use; server-side tokenization is the bottleneck.
body["prompt"] = strings.Repeat("hello ", inputTokens)
```

**Run:** `go test ./cmd/... -run TestRealClient_ProportionalPrompt -v` → PASS

**Lint:** `golangci-lint run ./cmd/...`

**Commit:** `git add cmd/observe.go cmd/observe_test.go && git commit -m "fix(observe): generate proportional prompt for accurate token counts (BC-1, BC-6)"`

---

#### Task 3: Non-streaming TTFT separation via firstByteReader (BC-2)

**Contracts:** BC-2

**Test (write first):**

```go
// cmd/observe_test.go — new test
func TestRealClient_NonStreaming_TTFTBeforeE2E(t *testing.T) {
    server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        w.Header().Set("Content-Type", "application/json")
        // Write partial response, flush, sleep, write rest
        flusher, ok := w.(http.Flusher)
        if !ok {
            t.Fatal("expected http.Flusher")
        }
        // Start writing the response — first byte arrives now
        data := []byte(`{"choices":[{"text":"hello world"}],"usage":{"prompt_tokens":10,"completion_tokens":2}}`)
        _, _ = w.Write(data[:10])
        flusher.Flush()
        time.Sleep(50 * time.Millisecond)
        _, _ = w.Write(data[10:])
    }))
    defer server.Close()

    client := NewRealClient(server.URL, "", "test-model", "vllm")
    record, err := client.Send(context.Background(), &PendingRequest{
        RequestID: 0, InputTokens: 10, Streaming: false,
    })
    if err != nil {
        t.Fatal(err)
    }
    if record.FirstChunkTimeUs == 0 {
        t.Error("FirstChunkTimeUs not recorded")
    }
    if record.LastChunkTimeUs == 0 {
        t.Error("LastChunkTimeUs not recorded")
    }
    if record.FirstChunkTimeUs > record.LastChunkTimeUs {
        t.Errorf("FirstChunkTimeUs (%d) > LastChunkTimeUs (%d)", record.FirstChunkTimeUs, record.LastChunkTimeUs)
    }
    // With 50ms sleep, there should be measurable separation (10ms threshold = 5x margin)
    if record.LastChunkTimeUs-record.FirstChunkTimeUs < 10_000 {
        t.Errorf("expected >= 10ms separation, got %d us", record.LastChunkTimeUs-record.FirstChunkTimeUs)
    }
}
```

**Run:** `go test ./cmd/... -run TestRealClient_NonStreaming_TTFTBeforeE2E -v` → FAIL

**Implement:**

`cmd/observe.go` — Add `firstByteReader` type and update `handleNonStreamingResponse()`:

```go
// firstByteReader wraps an io.Reader and captures the timestamp when the first byte is received.
type firstByteReader struct {
    r             io.Reader
    firstReadTime int64 // UnixMicro of first successful Read (n > 0); 0 = no data yet
}

func (f *firstByteReader) Read(p []byte) (int, error) {
    n, err := f.r.Read(p)
    if f.firstReadTime == 0 && n > 0 {
        f.firstReadTime = time.Now().UnixMicro()
    }
    return n, err
}
```

Replace `handleNonStreamingResponse()` lines 118-143 entirely. Task 3 replaces the reading + timing logic (lines 119-128). The JSON parsing and token extraction (lines 130-142) remain unchanged — Task 4 will add `prompt_tokens` extraction alongside the existing `completion_tokens` extraction:
```go
func (c *RealClient) handleNonStreamingResponse(resp *http.Response, record *RequestRecord) (*RequestRecord, error) {
    fbr := &firstByteReader{r: resp.Body}
    bodyData, err := io.ReadAll(fbr)
    if err != nil {
        record.Status = "error"
        record.ErrorMessage = fmt.Sprintf("read error: %v", err)
        return record, nil
    }
    record.FirstChunkTimeUs = fbr.firstReadTime
    record.LastChunkTimeUs = time.Now().UnixMicro()
    record.NumChunks = 1

    // JSON parsing + token extraction unchanged from current code (lines 130-143)
    var result map[string]interface{}
    if err := json.Unmarshal(bodyData, &result); err != nil {
        record.Status = "error"
        record.ErrorMessage = fmt.Sprintf("JSON parse error: %v", err)
        return record, nil
    }
    if usage, ok := result["usage"].(map[string]interface{}); ok {
        if ct, ok := usage["completion_tokens"].(float64); ok {
            record.OutputTokens = int(ct)
        }
    }
    return record, nil
}
```

**Run:** `go test ./cmd/... -run TestRealClient_NonStreaming_TTFTBeforeE2E -v` → PASS

**Lint:** `golangci-lint run ./cmd/...`

**Commit:** `git add cmd/observe.go cmd/observe_test.go && git commit -m "fix(observe): separate non-streaming TTFT from E2E via firstByteReader (BC-2)"`

---

#### Task 4: Extract prompt_tokens + wire ServerInputTokens and Model (BC-3, BC-5)

**Contracts:** BC-3, BC-5

**Test (write first):**

```go
// cmd/observe_test.go — update existing tests
// In TestRealClient_NonStreaming_RecordsTokenCounts, add:
if record.ServerInputTokens != 100 {
    t.Errorf("ServerInputTokens = %d, want 100", record.ServerInputTokens)
}

// In TestRealClient_Streaming_RecordsFirstAndLastChunkTime, add:
if record.ServerInputTokens != 100 {
    t.Errorf("ServerInputTokens = %d, want 100", record.ServerInputTokens)
}

// New test for Recorder Model wiring
func TestRecorder_WiresModelAndServerInputTokens(t *testing.T) {
    rec := &Recorder{}
    rec.RecordRequest(
        &PendingRequest{RequestID: 0, ClientID: "c1", Model: "test-model"},
        &RequestRecord{RequestID: 0, Status: "ok", ServerInputTokens: 42},
    )
    records := rec.Records()
    if len(records) != 1 {
        t.Fatalf("got %d records, want 1", len(records))
    }
    if records[0].Model != "test-model" {
        t.Errorf("Model = %q, want %q", records[0].Model, "test-model")
    }
    if records[0].ServerInputTokens != 42 {
        t.Errorf("ServerInputTokens = %d, want 42", records[0].ServerInputTokens)
    }
}
```

**Run:** `go test ./cmd/... -run "TestRealClient_NonStreaming_RecordsTokenCounts|TestRealClient_Streaming|TestRecorder_WiresModel" -v` → FAIL

**Implement:**

`cmd/observe.go` — In `handleNonStreamingResponse()`, inside the existing `if usage, ok` block, add alongside `completion_tokens`:
```go
if pt, ok := usage["prompt_tokens"].(float64); ok {
    record.ServerInputTokens = int(pt)
} else if _, exists := usage["prompt_tokens"]; exists {
    logrus.Debugf("observe: prompt_tokens has unexpected type %T, expected float64", usage["prompt_tokens"])
}
```

In `handleStreamingResponse()`, inside the existing `if lastUsage != nil` block, add alongside `completion_tokens`:
```go
if pt, ok := lastUsage["prompt_tokens"].(float64); ok {
    record.ServerInputTokens = int(pt)
} else if _, exists := lastUsage["prompt_tokens"]; exists {
    logrus.Debugf("observe: prompt_tokens has unexpected type %T, expected float64", lastUsage["prompt_tokens"])
}
```

Also add a debug log for the pre-existing silent `continue` on JSON parse errors in the streaming chunk loop:
```go
var chunk map[string]interface{}
if err := json.Unmarshal([]byte(data), &chunk); err != nil {
    logrus.Debugf("observe: skipping malformed SSE chunk: %v", err)
    continue
}
```

In `RecordRequest()`, wire the new fields and update the TODO comment:
```go
func (r *Recorder) RecordRequest(pending *PendingRequest, result *RequestRecord) {
    r.mu.Lock()
    defer r.mu.Unlock()
    r.records = append(r.records, workload.TraceRecord{
        // TODO: populate DeadlineUs once PendingRequest carries deadline info (out of #655 scope)
        Model:             pending.Model,
        ServerInputTokens: result.ServerInputTokens,
        RequestID:         result.RequestID,
        // ... rest unchanged
    })
}
```

**Run:** `go test ./cmd/... -v` → PASS

**Lint:** `golangci-lint run ./cmd/...`

**Commit:** `git add cmd/observe.go cmd/observe_test.go && git commit -m "fix(observe): extract prompt_tokens, wire Model and ServerInputTokens (BC-3, BC-5)"`

---

#### Task 5: Add deferred ITL TODO comment (Bug 5)

**Contracts:** None (documentation only)

**Implement:**

`cmd/observe.go` — In `handleStreamingResponse()`, after the chunk loop:
```go
// TODO: Per-chunk ITL timestamps not yet recorded (#655 Bug 5, deferred).
// Only first/last chunk times are captured. Full ITL distribution requires
// storing each chunk timestamp, which needs new schema support.
```

**Lint:** `golangci-lint run ./cmd/...`

**Commit:** `git add cmd/observe.go && git commit -m "docs(observe): add TODO for deferred per-chunk ITL timestamps"`

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | TestRealClient_ProportionalPrompt |
| BC-2 | Task 3 | Unit | TestRealClient_NonStreaming_TTFTBeforeE2E |
| BC-3 | Task 4 | Unit | TestRealClient_NonStreaming_RecordsTokenCounts (updated) |
| BC-3 | Task 4 | Unit | TestRealClient_Streaming_RecordsFirstAndLastChunkTime (updated) |
| BC-3, BC-5 | Task 4 | Unit | TestRecorder_WiresModelAndServerInputTokens |
| BC-4 | Task 1 | Unit | TestRealClient_MaxOutputTokens_FlowsThrough |
| BC-6 | Task 2 | Unit | TestRealClient_ProportionalPrompt (zero + negative InputTokens cases) |

No golden dataset impact — this is CLI-layer code, not simulator core.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| `firstByteReader` timestamp fires before any data arrives | Low | Medium | Timestamp captured inside `Read()`, so it fires when data is available | Task 3 |
| Prompt trailing space affects token count | Low | Low | Servers tokenize independently; "hello " is standard | Task 2 |
| Zero InputTokens produces empty prompt | Low | Medium | Guard: `if inputTokens <= 0 { inputTokens = 1 }` | Task 2 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

**Plan-specific checks:**
- [x] No unnecessary abstractions — `firstByteReader` is minimal, single-purpose
- [x] No feature creep — Bug 5 deferred, no new schema
- [x] No unexercised flags or interfaces
- [x] No partial implementations
- [x] No breaking changes — MaxOutputTokens defaults to 2048
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] Shared test helpers: N/A (no shared test infra needed, all tests use httptest)
- [x] CLAUDE.md: no updates needed (no new files/packages, no new CLI flags)
- [x] No stale references in CLAUDE.md
- [x] Documentation DRY: N/A (no canonical sources modified)
- [x] Deviation log reviewed — Bug 5 deferral justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 adds fields, Tasks 2-4 use them)
- [x] All contracts mapped to tasks (see Test Strategy table)
- [x] Golden dataset: N/A (CLI-layer code, no simulator output changes)
- [x] Construction site audit completed — PendingRequest (4 sites), RequestRecord (2 sites), all backward-compatible
- [x] Not part of a macro plan

**Antipattern rules (R1-R23):**
- [x] R1: No silent data loss — prompt_tokens extracted or zero when absent. New `prompt_tokens` extraction sites add `logrus.Debugf` when the field exists but has unexpected type (consistent with SSE parse error debug logging). Pre-existing `completion_tokens` extraction retains its original silent pattern (not modified by this PR).
- [x] R2: N/A — no map iteration in new code
- [x] R3: MaxOutputTokens validated (<=0 → default 2048)
- [x] R4: Construction sites audited — all sites backward-compatible (new fields zero-valued)
- [x] R5: N/A — no resource-allocating loops
- [x] R6: No Fatalf in sim/ — all changes in cmd/
- [x] R7: N/A — no golden tests added
- [x] R8: N/A — no exported maps
- [x] R9: N/A — no YAML fields
- [x] R10: N/A — no YAML parsing
- [x] R11: N/A — no division operations
- [x] R12: N/A — no golden dataset changes
- [x] R13: N/A — no new interfaces
- [x] R14: N/A — no multi-concern methods
- [x] R15: N/A — no PR references to clean
- [x] R16: N/A — no config changes
- [x] R17: N/A — no routing signals
- [x] R18: N/A — no CLI flag defaults
- [x] R19: N/A — no retry loops
- [x] R20: N/A — no anomaly detectors
- [x] R21: N/A — no range over mutable slices
- [x] R22: N/A — no pre-checks
- [x] R23: Applicable — `handleNonStreamingResponse` and `handleStreamingResponse` both extract `prompt_tokens` using identical `usage["prompt_tokens"].(float64)` pattern; parity confirmed

---

## Appendix: File-Level Implementation Details

**File: `cmd/observe.go`**

- **Purpose:** HTTP client for `blis observe` mode
- **Changes:**
  - `PendingRequest`: add `MaxOutputTokens int`, `Model string`
  - `RequestRecord`: add `ServerInputTokens int`
  - `Send()`: use `req.MaxOutputTokens` (default 2048), generate proportional prompt
  - Add `firstByteReader` type (unexported)
  - `handleNonStreamingResponse()`: wrap body with `firstByteReader`, extract `prompt_tokens`
  - `handleStreamingResponse()`: extract `prompt_tokens` from `lastUsage`, add ITL TODO
  - `RecordRequest()`: wire `Model`, `ServerInputTokens`, update TODO comment

**File: `cmd/observe_test.go`**

- **Purpose:** Tests for HTTP client measurement accuracy
- **Changes:**
  - Update 2 existing tests to verify `ServerInputTokens`
  - Add `TestRealClient_MaxOutputTokens_FlowsThrough`
  - Add `TestRealClient_ProportionalPrompt`
  - Add `TestRealClient_NonStreaming_TTFTBeforeE2E`
  - Add `TestRecorder_WiresModelAndServerInputTokens`
