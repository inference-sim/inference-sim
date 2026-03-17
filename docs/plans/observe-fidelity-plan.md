# Observe Fidelity: Server Compatibility & Prefix Sharing

| Field | Value |
|-------|-------|
| **Goal** | Close the remaining workload fidelity gaps in `blis observe` so that real-server traces accurately reflect production behavior for calibration |
| **Problem** | `RealClient` hardcodes `/v1/completions`, never sets `stream_options`, doesn't extract `finish_reason`, biases output length via `max_tokens`, constructs prompts without shared prefixes, and leaves `measured_rtt_ms` unpopulated |
| **What this PR adds** | (1) Chat completions endpoint support, (2) `stream_options` for streaming usage, (3) `finish_reason` extraction + TraceV2 schema extension, (4) configurable `max_tokens` behavior, (5) deterministic shared prefix strings for prefix groups, (6) `--rtt-ms` flag for network RTT |
| **Why** | Without these fixes, `blis observe` traces have silent zero token counts (streaming), wrong prompt_tokens (no chat template), missing truncation signal, biased output distribution, no prefix cache activation, and no network normalization — all degrading calibration accuracy |
| **Architecture** | `cmd/` layer only — `observe.go` (RealClient), `observe_cmd.go` (CLI flags, orchestrator), `sim/workload/tracev2.go` (schema). No `sim/` library changes except the TraceV2 schema addition |
| **Source** | [#660](https://github.com/inference-sim/inference-sim/issues/660) and addendum comments |
| **Closes** | #660 |
| **Behavioral Contracts** | BC-1 through BC-7 below |

---

## Part 1: Design Validation

### A. Executive Summary

This PR adds six server-compatibility and workload-fidelity improvements to `blis observe`. The changes fall into three categories:

1. **API compatibility** (BC-1, BC-2, BC-3): Support `/v1/chat/completions` endpoint, set `stream_options.include_usage` for streaming, and extract `finish_reason` from responses. These ensure token counts and completion status are accurately captured.

2. **Output fidelity** (BC-4, BC-5): Make `max_tokens` behavior configurable (constrained vs unconstrained) and generate deterministic shared prefix strings so prefix groups activate the real server's KV cache.

3. **Calibration support** (BC-6, BC-7): Add `finish_reason` to the TraceV2 schema and populate `measured_rtt_ms` via a `--rtt-ms` CLI flag.

All changes are in the `cmd/` layer (observe command) and the TraceV2 schema (`sim/workload/tracev2.go`). No simulation engine changes.

### B. Behavioral Contracts

#### BC-1: Chat Completions Endpoint Support

**GIVEN** a `RealClient` configured with `apiFormat = "chat"`
**WHEN** `Send()` constructs the HTTP request
**THEN** the request targets `/v1/chat/completions` with a `"messages"` array containing `[{"role": "user", "content": <prompt>}]` instead of a `"prompt"` field
**MECHANISM** New `apiFormat` field on `RealClient` (default: `"completions"`). `Send()` branches on this field to choose endpoint path and body structure. `NewRealClient` accepts the format parameter; `observe_cmd.go` passes it from a new `--api-format` CLI flag.

#### BC-2: Streaming Usage Extraction via `stream_options`

**GIVEN** a streaming request (`Streaming = true`)
**WHEN** `Send()` constructs the request body
**THEN** the body includes `"stream_options": {"include_usage": true}`
**AND** the streaming response parser extracts `usage` from the final SSE chunk (already implemented) and `ServerInputTokens` / `OutputTokens` are non-zero when the server reports them
**AND** if after streaming completes `lastUsage` is nil and chunks were received, a warning is logged: `"request %d: server returned no usage in streaming response; token counts will be zero"`
**MECHANISM** Conditional insertion of `stream_options` in `Send()` when `body["stream"] = true`. After the streaming loop, check `lastUsage == nil && chunkCount > 0` and emit `logrus.Warnf` (R1 — no silent data loss when server ignores `stream_options`).

#### BC-3: `finish_reason` Extraction

**GIVEN** a completed HTTP response (streaming or non-streaming)
**WHEN** the response parser processes `choices[0].finish_reason`
**THEN** `RequestRecord.FinishReason` contains the server's value (`"stop"`, `"length"`, `"abort"`, `"tool_calls"`, `"content_filter"`, etc.)
**AND** if `finish_reason == "length"`, a warning is logged to stderr (output truncated at max_tokens)
**AND** if `finish_reason == "abort"`, a warning is logged to stderr (server preemption/memory pressure — trace data may be unreliable)
**MECHANISM** New `FinishReason string` field on `RequestRecord`. Non-streaming: extracted from `choices[0].finish_reason` in the parsed JSON. Streaming: extracted from `choices[0].finish_reason` in content chunks; **must check `len(choices) > 0` before accessing** because the usage-only chunk (sent when `stream_options.include_usage` is set) has `"choices": []`. The `finish_reason` appears in the last **content** chunk, not the usage-only chunk. Warning via `logrus.Warnf` (stderr, per output channel separation).

#### BC-4: Configurable `max_tokens` Behavior

**GIVEN** the `--unconstrained-output` flag is set
**WHEN** `Send()` constructs the request body
**THEN** for `apiFormat = "chat"`: `max_tokens` is omitted from the request body (server defaults to `max_model_len - prompt_len`)
**AND** for `apiFormat = "completions"`: `max_tokens` is set to `math.MaxInt32` (because `/v1/completions` defaults to only 16 tokens when `max_tokens` is absent)
**AND** when `--unconstrained-output` is NOT set (default), the current behavior is preserved: `max_tokens = MaxOutputTokens`
**NOTE** `--unconstrained-output` applies globally to all requests. It is designed for calibration scenarios where output length bias must be eliminated.
**MECHANISM** New boolean flag `--unconstrained-output` (default: false). In `Send()`, when `Unconstrained` is true: if `apiFormat == "chat"`, omit `max_tokens`; if `apiFormat == "completions"`, set `body["max_tokens"] = math.MaxInt32`.

#### BC-5: Deterministic Shared Prefix Strings for Prefix Groups

**GIVEN** a workload with requests in prefix group `"group-A"` with `prefix_length = 100`
**WHEN** prompts are constructed for dispatch
**THEN** all requests in `"group-A"` share an identical prompt prefix substring of `prefix_length` words (which approximates `prefix_length` tokens within tokenizer-dependent variation — typically 1 word ≈ 1.0-1.5 subword tokens)
**AND** requests in a different group `"group-B"` have a different prefix substring
**AND** the prefix is deterministic given the same seed and group name
**AND** the `server_input_tokens` field captures the actual tokenized count as ground truth
**NOTE** Prefix cache activation on vLLM requires `--enable-prefix-caching` on the server. Cache hits are aligned to block boundaries (typically 16 tokens). The word-based approximation produces byte-identical prefix strings across requests in the same group, which is sufficient for cache activation even though the exact token count may differ from `prefix_length`.
**MECHANISM** Before dispatch, extract prefix groups and their lengths from `workload.Clients` into a `map[string]int`. Call `buildPrefixStrings(groups, seed)` to generate a `map[string]string`. Use `hash/fnv` FNV-64 hash for deterministic per-group seed derivation: `fnv64(fmt.Sprintf("%d:%s", seed, groupName))`. In `requestToPending`, if request has a `PrefixGroup` in the map: the suffix filler length is `max(1, len(req.InputTokens) - prefixLength)` words (because `InputTokens` already includes prefix tokens from the generator). The prompt is `prefixString + strings.Repeat("word ", suffixLen)`. Prompt assembly happens in `requestToPending` (always populating `PendingRequest.Prompt`), not lazily in `Send()`. `Send()` uses `pending.Prompt` directly — no fallback prompt generation path.

#### BC-6: `finish_reason` in TraceV2 Schema

**GIVEN** a `TraceRecord` with `FinishReason` populated
**WHEN** the record is exported to CSV and reloaded
**THEN** the `finish_reason` column round-trips correctly (export → load → identical value)
**AND** empty `finish_reason` round-trips as empty string
**MECHANISM** Add `FinishReason string` field to `TraceRecord`, new column `finish_reason` appended after `error_message` in `traceV2Columns` (index 25, 26th column). Update `ExportTraceV2`, `parseTraceRecord`, and `traceV2Columns`. Backward compatibility: `parseTraceRecord` accepts both 25-column (pre-`finish_reason`) and 26-column CSV rows. When loading a 25-column row, `FinishReason` defaults to empty string. Implementation: change the column count guard from `len(row) < len(traceV2Columns)` to `len(row) < len(traceV2Columns)-1` and conditionally parse index 25.

#### BC-7: `measured_rtt_ms` Population via CLI Flag

**GIVEN** the user passes `--rtt-ms 2.5`
**WHEN** the TraceV2 header is constructed after dispatch completes
**THEN** `header.Network.MeasuredRTTMs == 2.5`
**AND** when `--rtt-ms` is not set, `MeasuredRTTMs` defaults to 0 (omitted from YAML)
**MECHANISM** New `--rtt-ms` float64 flag (default: 0). Wired into `TraceHeader.Network.MeasuredRTTMs` in `runObserve`.

### C. Component Interaction

```
┌─────────────────────────────────────────────────────────────┐
│ observe_cmd.go                                               │
│  --api-format flag ──► NewRealClient(apiFormat)              │
│  --unconstrained-output flag ──► PendingRequest.Unconstrained│
│  --rtt-ms flag ──► TraceHeader.Network.MeasuredRTTMs         │
│  prefixStrings = buildPrefixStrings(workload, seed)          │
│  requestToPending() uses prefixStrings for prompt assembly   │
└───────────┬─────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ observe.go (RealClient)                                      │
│  Send() ─► branches on apiFormat for endpoint + body         │
│  Send() ─► adds stream_options when streaming                │
│  Send() ─► omits max_tokens when Unconstrained               │
│  handleNonStreamingResponse() ─► extracts finish_reason      │
│  handleStreamingResponse() ─► extracts finish_reason         │
│  Recorder.RecordRequest() ─► wires FinishReason to TraceRecord│
└───────────┬─────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────┐
│ sim/workload/tracev2.go (TraceV2 schema)                     │
│  TraceRecord.FinishReason (new field, column 26)             │
│  ExportTraceV2 / parseTraceRecord updated                    │
└─────────────────────────────────────────────────────────────┘
```

**State changes:** `RealClient` gains `apiFormat` field (via functional option `WithAPIFormat`). `PendingRequest` gains `Unconstrained bool`, `Prompt string` (assembled in `requestToPending` — always populated, `Send()` uses it directly), and `DeadlineUs int64`. `RequestRecord` gains `FinishReason string`. `TraceRecord` gains `FinishReason string` (index 25, 26th column).

**R4 construction site audit:**
- `PendingRequest`: constructed in `requestToPending` only (`observe_cmd.go:474`). Test construction sites use named fields — new zero-value fields are backward-compatible.
- `RequestRecord`: constructed in `Send()` (observe.go:65-68), then populated by `handleNonStreamingResponse` or `handleStreamingResponse`. Single construction site.
- `TraceRecord`: constructed in `Recorder.RecordRequest` (`observe.go:249`) and `RequestsToTraceRecords` (`tracev2.go:356`). The latter produces simulation-generated records where `FinishReason` correctly defaults to empty string (no server to report it).
- `RealClient`: constructed in `NewRealClient` only. Functional options pattern: `NewRealClient(url, key, model, serverType string, opts ...RealClientOption)` — existing callers compile unchanged.

**Extension friction:** Adding one more API format (e.g., Ollama) requires: 1 new case in `Send()` body construction, 1 new case in endpoint selection. Low friction — same pattern as `apiFormat`.

### D. Deviation Log

| # | Source says | Plan does | Category | Rationale |
|---|-----------|----------|----------|-----------|
| D1 | #660 suggests `endpoint` or `apiFormat` field | Uses `--api-format` CLI flag mapping to `apiFormat` field | Naming | `apiFormat` is more descriptive — it controls both endpoint path AND body format |
| D2 | #660 suggests "large fixed cap" for unconstrained output | For chat: omits `max_tokens` (server defaults to `max_model_len - prompt_len`). For completions: sends `math.MaxInt32` (because completions defaults to only 16 tokens when absent) | Endpoint-aware | vLLM/SGLang `/v1/completions` defaults `max_tokens` to 16, not `max_model_len`. `/v1/chat/completions` defaults to `max_model_len - prompt_len` when both `max_tokens` and `max_completion_tokens` are absent |
| D3 | #660 suggests auto-measured RTT via HTTP ping | Uses `--rtt-ms` flag only | Scope reduction | Auto-ping adds network probing logic and error handling; a user-provided value is simpler, more predictable, and sufficient for calibration. Auto-measurement can be a follow-up |
| D4 | #660 mentions adding `FinishReason` to TraceRecord "similar to #653" | Adds as column 26 after `error_message` | Alignment | Follows the established pattern of appending new columns at the end |
| D5 | #660 mentions tokenizer-aware prompts | Uses deterministic word sequences for prefix sharing | Scope reduction | True tokenizer integration requires model-specific vocabulary files; deterministic word sequences give good-enough prefix overlap for KV cache activation |
| D6 | #660 Priority 1 (per-request timeout) | Not in this PR's scope — partially addressed in #659/#704 | Deferred | #659 added session-level timeout handling and `Status: "timeout"` recording, but per-request `context.WithDeadline` from `req.Deadline` is NOT yet implemented in the dispatch goroutine. Task 8 wires `DeadlineUs` to TraceRecord for recording purposes only. Per-request timeout enforcement remains a follow-up |
| D7 | #660 Priority 3 (sessions / closed-loop) | Not in scope — already implemented in #659/#704 | Already done | `SessionManager` integration with follow-up dispatch already works. See `observe_cmd.go:296-313` |
| D8 | Items from #660 addendum comments | BC-1 (chat), BC-2 (stream_options), BC-3 (finish_reason), BC-4 (max_tokens bias), BC-7 (measured_rtt_ms) come from addendum comments, not the original issue body | Scope expansion | Three addendum comments on #660 identified server-compatibility gaps during #655 and #659 design reviews |

### E. Review Guide for Human Reviewer

1. **BC-6 schema change** is the highest-risk item — it touches the TraceV2 CSV format which has strict round-trip expectations. Verify the column count updates in `traceV2Columns`, `ExportTraceV2`, and `parseTraceRecord` are consistent.
2. **BC-1 chat completions** changes the HTTP request body structure — verify the `messages` format matches the OpenAI API spec.
3. **BC-5 prefix sharing** — verify the prefix string generation is truly deterministic across runs with the same seed.
4. Check that `--api-format` validation rejects unknown values (not just `completions` and `chat`).

---

## Part 2: Executable Implementation

### F. Implementation Overview

8 tasks, ordered by dependency. Tasks 1-2 are schema/infrastructure. Tasks 3-7 are the six behavioral contracts. Task 8 is the CLI flag wiring.

### G. Task Breakdown

#### Task 1: Add `FinishReason` to TraceV2 schema (BC-6)

**Why first:** Other tasks (BC-3) depend on `TraceRecord.FinishReason` existing.

**Step 1 — Write failing test:**

In `sim/workload/tracev2_test.go`, add a test that constructs a `TraceRecord` with `FinishReason: "stop"`, exports to CSV, reloads, and asserts the field round-trips.

```go
func TestTraceV2_FinishReason_RoundTrip(t *testing.T) {
    header := &TraceHeader{Version: 1, TimeUnit: "us", Mode: "real"}
    records := []TraceRecord{{
        RequestID:    1,
        InputTokens:  10,
        OutputTokens: 5,
        ArrivalTimeUs: 1000,
        SendTimeUs:    2000,
        Status:        "ok",
        FinishReason:  "stop",
    }}

    headerPath := filepath.Join(t.TempDir(), "h.yaml")
    dataPath := filepath.Join(t.TempDir(), "d.csv")
    require.NoError(t, ExportTraceV2(header, records, headerPath, dataPath))

    tv2, err := LoadTraceV2(headerPath, dataPath)
    require.NoError(t, err)
    require.Len(t, tv2.Records, 1)
    assert.Equal(t, "stop", tv2.Records[0].FinishReason)
}
```

**Step 2 — Run test, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./sim/workload/... -run TestTraceV2_FinishReason_RoundTrip -count=1
```
Expected: compilation error (no `FinishReason` field on `TraceRecord`).

**Step 3 — Implement:**

In `sim/workload/tracev2.go`:
- Add `FinishReason string` field to `TraceRecord` struct (after `ErrorMessage`)
- Append `"finish_reason"` to `traceV2Columns` (becomes index 25, 26 total columns)
- In `ExportTraceV2`: append `rec.FinishReason` to the CSV row
- In `parseTraceRecord`: if `len(row) >= 26`, parse index 25 as `FinishReason`; if `len(row) == 25`, default `FinishReason` to `""`
- In `LoadTraceV2`: change column count guard from `len(row) < len(traceV2Columns)` to `len(row) < len(traceV2Columns)-1` (accept 25 or 26 columns)
- Also add a backward-compat test: construct a 25-column CSV row and verify it loads with `FinishReason == ""`

**Step 4 — Run test, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./sim/workload/... -run TestTraceV2_FinishReason_RoundTrip -count=1
```

**Step 5 — Fix all existing tracev2 tests:**

Existing tests in `tracev2_test.go` construct raw CSV rows with 25 columns. Update ALL test helpers and raw row constructions to include a 26th column for `finish_reason`. Run full test suite:
```bash
cd .worktrees/observe-fidelity && go test ./sim/workload/... -count=1
```

**Step 6 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./sim/workload/...
```

**Step 7 — Commit:**
```
feat(tracev2): add finish_reason column to TraceV2 schema

Implement BC-6: finish_reason string field round-trips through CSV
export/load. Column 26, appended after error_message.
```

---

#### Task 2: Add `FinishReason` to `RequestRecord` and `Recorder` (BC-3 infrastructure)

**Step 1 — Write failing test:**

In `cmd/observe_test.go`, add a test that verifies `FinishReason` flows from `RequestRecord` through `Recorder` to `TraceRecord`:

```go
func TestRecorder_WiresFinishReason(t *testing.T) {
    rec := &Recorder{}
    pending := &PendingRequest{RequestID: 1, InputTokens: 10, MaxOutputTokens: 5}
    result := &RequestRecord{
        RequestID:    1,
        Status:       "ok",
        FinishReason: "length",
        OutputTokens: 5,
    }
    rec.RecordRequest(pending, result, 1000, "", 0)
    records := rec.Records()
    require.Len(t, records, 1)
    assert.Equal(t, "length", records[0].FinishReason)
}
```

**Step 2 — Run test, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run TestRecorder_WiresFinishReason -count=1
```
Expected: compilation error (no `FinishReason` on `RequestRecord`).

**Step 3 — Implement:**

In `cmd/observe.go`:
- Add `FinishReason string` to `RequestRecord` struct
- Add `Prompt string` to `PendingRequest` struct (needed by Tasks 4-6 before Task 7 adds prefix logic)
- Modify `Send()` to use `pending.Prompt` directly instead of generating the prompt inline. Remove the `strings.Repeat("hello ", inputTokens)` line; `Send()` now assumes `Prompt` is always populated by the caller.
- In `Recorder.RecordRequest`, wire `result.FinishReason` → `TraceRecord.FinishReason`

In `cmd/observe_cmd.go`:
- Update `requestToPending` to always populate `Prompt`: `pending.Prompt = strings.Repeat("hello ", max(1, len(req.InputTokens)))` (prefix logic added in Task 7 will replace this for prefix-group requests)

**Step 3b — Update existing tests for Prompt field:**

Since `Send()` now uses `pending.Prompt` directly (no fallback), all existing `PendingRequest` construction sites in `cmd/observe_test.go` must include a `Prompt` field. Key updates:
- `TestRealClient_ProportionalPrompt`: Refactor to test `requestToPending` instead of `Send()`. The proportional prompt behavior has moved from `Send()` to `requestToPending`. Alternatively, populate `Prompt` in the test: `Prompt: strings.Repeat("hello ", 50)` and assert the prompt passes through to the request body.
- `TestRealClient_NonStreaming_RecordsTokenCounts`: Add `Prompt: strings.Repeat("hello ", 100)`
- `TestRealClient_Streaming_RecordsFirstAndLastChunkTime`: Add `Prompt: strings.Repeat("hello ", 100)`
- `TestRealClient_ServerError_RecordsError`: Add `Prompt: "hello "`
- `TestRealClient_MaxOutputTokens_FlowsThrough`: Add `Prompt: strings.Repeat("hello ", 10)`
- `TestRealClient_NonStreaming_TTFTBeforeE2E`: Add `Prompt: strings.Repeat("hello ", 10)`
- `TestRecorder_WiresModelAndServerInputTokens`: Add `Prompt: strings.Repeat("hello ", 10)`

Run full existing test suite after these updates:
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -count=1
```

**Step 4 — Run test, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run TestRecorder_WiresFinishReason -count=1
```

**Step 5 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```
feat(observe): wire FinishReason through RequestRecord to TraceRecord

Implement BC-3 infrastructure: RequestRecord.FinishReason flows
through Recorder to TraceRecord.FinishReason (BC-6 column).
```

---

#### Task 3: Extract `finish_reason` from responses (BC-3)

**Step 1 — Write failing tests:**

In `cmd/observe_test.go`, add two tests:

```go
func TestRealClient_NonStreaming_ExtractsFinishReason(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        json.NewEncoder(w).Encode(map[string]interface{}{
            "choices": []map[string]interface{}{{"text": "hi", "finish_reason": "stop"}},
            "usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 5},
        })
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm")
    record, err := client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 10, MaxOutputTokens: 5,
        Prompt: strings.Repeat("hello ", 10),
    })
    require.NoError(t, err)
    assert.Equal(t, "stop", record.FinishReason)
}

func TestRealClient_Streaming_ExtractsFinishReason(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        flusher, _ := w.(http.Flusher)
        // Content chunk with finish_reason
        fmt.Fprintf(w, "data: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"length\"}]}\n\n")
        flusher.Flush()
        // Usage-only chunk (empty choices — must not overwrite finish_reason)
        fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n")
        flusher.Flush()
        fmt.Fprintf(w, "data: [DONE]\n\n")
        flusher.Flush()
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm")
    record, err := client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 10, MaxOutputTokens: 5, Streaming: true,
        Prompt: strings.Repeat("hello ", 10),
    })
    require.NoError(t, err)
    assert.Equal(t, "length", record.FinishReason)
    assert.Equal(t, 5, record.OutputTokens, "usage from usage-only chunk")
}
```

**Note:** The streaming chat completions test (`TestRealClient_StreamingChat_ExtractsFinishReason`) is deferred to Task 5 because it requires `WithAPIFormat("chat")` which is introduced there.

**Step 2 — Run tests, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run "TestRealClient_.*_ExtractsFinishReason" -count=1
```
Expected: `FinishReason` is empty string (not extracted yet).

**Step 3 — Implement:**

In `cmd/observe.go`:
- `handleNonStreamingResponse`: After JSON parsing, extract `choices[0].finish_reason` and set `record.FinishReason`
- `handleStreamingResponse`: On each chunk, first check `len(choices) > 0` (usage-only chunks have empty choices — skip them). Then extract `choices[0].finish_reason` with null-safety: JSON `null` unmarshals to Go `nil` in `interface{}`, not empty string. Use type assertion: `if fr, ok := choice["finish_reason"].(string); ok && fr != "" { record.FinishReason = fr }`. Keep the last non-empty value. The `finish_reason` appears in the last **content** chunk, not the usage-only chunk.
- After the streaming loop: if `lastUsage == nil && chunkCount > 0`, log `logrus.Warnf("request %d: server returned no usage in streaming response; token counts will be zero", record.RequestID)` (BC-2 R1 compliance)
- After extraction: if `record.FinishReason == "length"`, log `logrus.Warnf("request %d truncated at max_tokens (finish_reason=length)", record.RequestID)`
- If `record.FinishReason == "abort"`, log `logrus.Warnf("request %d aborted by server (finish_reason=abort) — possible memory pressure", record.RequestID)`

**Step 4 — Run tests, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run "TestRealClient_.*_ExtractsFinishReason" -count=1
```

**Step 5 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```
feat(observe): extract finish_reason from streaming and non-streaming responses

Implement BC-3: parse choices[0].finish_reason from JSON response
(non-streaming) and final SSE chunk (streaming). Log warning when
finish_reason == "length" (output truncated at max_tokens).
```

---

#### Task 4: Add `stream_options` for streaming usage (BC-2)

**Step 1 — Write failing test:**

In `cmd/observe_test.go`, add a test that captures the request body sent to the mock server and verifies `stream_options` is present:

```go
func TestRealClient_Streaming_SetsStreamOptions(t *testing.T) {
    var capturedBody map[string]interface{}
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&capturedBody)
        // Return minimal SSE response
        fmt.Fprintf(w, "data: {\"choices\":[{\"text\":\"hi\",\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":1,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n")
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm")
    client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 1, MaxOutputTokens: 1, Streaming: true,
        Prompt: "hello ",
    })

    streamOpts, ok := capturedBody["stream_options"].(map[string]interface{})
    require.True(t, ok, "stream_options must be present for streaming requests")
    assert.Equal(t, true, streamOpts["include_usage"])
}
```

Also add a negative test: non-streaming requests should NOT have `stream_options`.

**Step 2 — Run test, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run TestRealClient_Streaming_SetsStreamOptions -count=1
```

**Step 3 — Implement:**

In `cmd/observe.go` `Send()`, after setting `body["stream"] = true`, add:
```go
if pending.Streaming {
    body["stream_options"] = map[string]interface{}{"include_usage": true}
}
```

**Step 4 — Run test, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run TestRealClient_Streaming_SetsStreamOptions -count=1
```

**Step 5 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```
feat(observe): set stream_options.include_usage for streaming requests

Implement BC-2: streaming requests include
stream_options: {include_usage: true} so vLLM/SGLang report token
counts in the final SSE chunk.
```

---

#### Task 5: Chat completions endpoint support (BC-1)

**Step 1 — Write failing test:**

In `cmd/observe_test.go`:

```go
func TestRealClient_ChatFormat_UsesMessagesEndpoint(t *testing.T) {
    var capturedBody map[string]interface{}
    var capturedPath string
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        capturedPath = r.URL.Path
        json.NewDecoder(r.Body).Decode(&capturedBody)
        // Return chat completions response
        json.NewEncoder(w).Encode(map[string]interface{}{
            "choices": []map[string]interface{}{{"message": map[string]string{"content": "hi"}, "finish_reason": "stop"}},
            "usage":   map[string]interface{}{"prompt_tokens": 10, "completion_tokens": 5},
        })
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm", WithAPIFormat("chat"))
    client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 10, MaxOutputTokens: 5, Prompt: strings.Repeat("hello ", 10),
    })

    assert.Equal(t, "/v1/chat/completions", capturedPath)
    msgs, ok := capturedBody["messages"].([]interface{})
    require.True(t, ok, "body must contain messages array")
    require.Len(t, msgs, 1)
    _, hasPrompt := capturedBody["prompt"]
    assert.False(t, hasPrompt, "chat format must not include prompt field")
}
```

Also test that default (`completions`) still uses `/v1/completions` with `prompt` field (regression guard).

Also add the deferred streaming chat test from Task 3 (now that `WithAPIFormat` exists):

```go
func TestRealClient_StreamingChat_ExtractsFinishReason(t *testing.T) {
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        flusher, _ := w.(http.Flusher)
        // Chat streaming: delta.content instead of text
        fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hi\"},\"finish_reason\":null}]}\n\n")
        flusher.Flush()
        fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n")
        flusher.Flush()
        // Usage-only chunk with empty choices
        fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":2}}\n\n")
        flusher.Flush()
        fmt.Fprintf(w, "data: [DONE]\n\n")
        flusher.Flush()
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm", WithAPIFormat("chat"))
    record, err := client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 10, MaxOutputTokens: 5, Streaming: true,
        Prompt: strings.Repeat("hello ", 10),
    })
    require.NoError(t, err)
    assert.Equal(t, "stop", record.FinishReason)
    assert.Equal(t, 2, record.OutputTokens)
}
```

**Step 2 — Run test, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run TestRealClient_ChatFormat -count=1
```

**Step 3 — Implement:**

In `cmd/observe.go`:
- Add `apiFormat string` field to `RealClient` struct (default: `"completions"`)
- Add functional option pattern:
  ```go
  type RealClientOption func(*RealClient)
  func WithAPIFormat(format string) RealClientOption {
      return func(c *RealClient) { c.apiFormat = format }
  }
  ```
- Update `NewRealClient` signature: `func NewRealClient(baseURL, apiKey, modelName, serverType string, opts ...RealClientOption) *RealClient` — existing callers with 4 args compile unchanged (variadic zero args)
- In `Send()`, branch on `c.apiFormat`:
  - `"completions"` (default): current behavior (`/v1/completions`, `body["prompt"] = prompt`)
  - `"chat"`: endpoint `/v1/chat/completions`, `body["messages"] = []map[string]string{{"role": "user", "content": prompt}}`

In `cmd/observe_cmd.go`:
- Add `--api-format` string flag (default: `"completions"`, usage: `API format: "completions" or "chat"`)
- Validate in `runObserve` (CLI layer, before calling `NewRealClient`): `if apiFormat != "completions" && apiFormat != "chat" { logrus.Fatalf("--api-format must be 'completions' or 'chat', got %q", apiFormat) }`
- Pass to `NewRealClient` via `WithAPIFormat(apiFormat)`

Also add a test for invalid `--api-format` rejection: `TestObserveCmd_InvalidAPIFormat_Fails`.

**Step 4 — Run test, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run TestRealClient_ChatFormat -count=1
```

**Step 5 — Run full test suite:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -count=1
```

**Step 6 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./cmd/...
```

**Step 7 — Commit:**
```
feat(observe): add chat completions endpoint support via --api-format

Implement BC-1: --api-format=chat uses /v1/chat/completions with
messages array. Default "completions" preserves existing behavior.
```

---

#### Task 6: Configurable `max_tokens` behavior (BC-4)

**Step 1 — Write failing test:**

In `cmd/observe_test.go`:

```go
func TestRealClient_UnconstrainedChat_OmitsMaxTokens(t *testing.T) {
    var capturedBody map[string]interface{}
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&capturedBody)
        json.NewEncoder(w).Encode(map[string]interface{}{
            "choices": []map[string]interface{}{{"message": map[string]string{"content": "hi"}, "finish_reason": "stop"}},
            "usage":   map[string]interface{}{"prompt_tokens": 1, "completion_tokens": 1},
        })
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm", WithAPIFormat("chat"))
    client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 1, MaxOutputTokens: 100, Unconstrained: true,
        Prompt: "hello ",
    })

    _, hasMaxTokens := capturedBody["max_tokens"]
    assert.False(t, hasMaxTokens, "unconstrained chat must omit max_tokens")
}

func TestRealClient_UnconstrainedCompletions_SendsLargeMaxTokens(t *testing.T) {
    var capturedBody map[string]interface{}
    srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
        json.NewDecoder(r.Body).Decode(&capturedBody)
        json.NewEncoder(w).Encode(map[string]interface{}{
            "choices": []map[string]interface{}{{"text": "hi", "finish_reason": "stop"}},
            "usage":   map[string]interface{}{"prompt_tokens": 1, "completion_tokens": 1},
        })
    }))
    defer srv.Close()

    client := NewRealClient(srv.URL, "", "test", "vllm") // default: completions
    client.Send(context.Background(), &PendingRequest{
        RequestID: 1, InputTokens: 1, MaxOutputTokens: 100, Unconstrained: true,
        Prompt: "hello ",
    })

    maxTokens, ok := capturedBody["max_tokens"].(float64) // JSON numbers are float64
    require.True(t, ok, "unconstrained completions must send max_tokens")
    assert.Equal(t, float64(math.MaxInt32), maxTokens, "completions defaults to 16 if absent; must send large value")
}
```

Also test that `Unconstrained: false` (default) still sends `max_tokens = MaxOutputTokens`.

**Step 2 — Run test, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run "TestRealClient_Unconstrained" -count=1
```

**Step 3 — Implement:**

In `cmd/observe.go`:
- Add `Unconstrained bool` to `PendingRequest`
- In `Send()`, replace unconditional `body["max_tokens"] = maxOutput` with:
  ```go
  if !pending.Unconstrained {
      body["max_tokens"] = maxOutput
  } else if c.apiFormat == "completions" {
      body["max_tokens"] = math.MaxInt32 // completions defaults to 16 if absent
  }
  // chat with unconstrained: omit max_tokens (server uses max_model_len - prompt_len)
  ```

In `cmd/observe_cmd.go`:
- Add `--unconstrained-output` bool flag (default: false)
- In `requestToPending`, set `Unconstrained` from the flag

**Step 4 — Run test, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run "TestRealClient_Unconstrained" -count=1
```

**Step 5 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./cmd/...
```

**Step 6 — Commit:**
```
feat(observe): add --unconstrained-output to omit max_tokens from requests

Implement BC-4: when set, max_tokens is omitted from the request body
so the server generates freely up to its max_model_len. Default
preserves existing constrained behavior.
```

---

#### Task 7: Deterministic shared prefix strings (BC-5)

**Step 1 — Write failing test:**

In `cmd/observe_cmd_test.go`:

```go
func TestBuildPrefixStrings_DeterministicAndDistinct(t *testing.T) {
    // Two groups with different names produce different prefixes
    prefixes1 := buildPrefixStrings(map[string]int{"group-A": 100, "group-B": 50}, 42)

    // Same seed + same groups = identical prefixes (determinism)
    prefixes2 := buildPrefixStrings(map[string]int{"group-A": 100, "group-B": 50}, 42)
    assert.Equal(t, prefixes1["group-A"], prefixes2["group-A"])
    assert.Equal(t, prefixes1["group-B"], prefixes2["group-B"])

    // Different groups have different prefixes
    assert.NotEqual(t, prefixes1["group-A"], prefixes1["group-B"])

    // Prefix length approximately matches target token count
    // (words ≈ tokens for simple vocabulary)
    wordsA := strings.Fields(prefixes1["group-A"])
    assert.InDelta(t, 100, len(wordsA), 5, "prefix word count should approximate token count")
}

func TestRequestToPending_PrependsPrefixString(t *testing.T) {
    prefixes := map[string]string{"shared": "alpha bravo charlie "}
    prefixLengths := map[string]int{"shared": 3}
    req := &sim.Request{InputTokens: []int{1, 2, 3, 4, 5}, PrefixGroup: "shared"}
    // With prefix, the prompt should start with the prefix string
    // Suffix filler = max(1, len(InputTokens) - prefixLength) = max(1, 5-3) = 2 words
    pending := requestToPending(req, 0, false, false, prefixes, prefixLengths)
    assert.True(t, strings.HasPrefix(pending.Prompt, "alpha bravo charlie"))
    // Total prompt should be prefix (3 words) + filler (2 words)
    words := strings.Fields(pending.Prompt)
    assert.Equal(t, 5, len(words), "total words = prefix + suffix filler")
}
```

**Step 2 — Run test, verify failure:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run "TestBuildPrefixStrings|TestRequestToPending_PrependsPrefixString" -count=1
```

**Step 3 — Implement:**

In `cmd/observe_cmd.go`:
- Add `buildPrefixStrings(groups map[string]int, seed int64) map[string]string`:
  - For each group (iterated in sorted key order for determinism), derive per-group seed via `hash/fnv` FNV-64: `fnv64(fmt.Sprintf("%d:%s", seed, groupName))`
  - Generate `prefixLength` words from a hardcoded vocabulary of 100 common English words (deterministic via per-group RNG)
  - If `prefixLength <= 0`, skip the group (no entry in map)
  - Return the concatenated string as the prefix
- Extract prefix groups from workload: iterate `workload.Clients`, build `map[string]int` of `PrefixGroup → PrefixLength` (same logic as `sim/workload/client.go:generatePrefixTokens`)
- Add `Prompt string` field to `PendingRequest`
- **Combined `requestToPending` signature** (final state after Tasks 7 and 8):
  `func requestToPending(req *sim.Request, reqIndex int, streaming bool, unconstrained bool, prefixes map[string]string, prefixLengths map[string]int) *PendingRequest`
  - **Always** populate `Prompt`:
    - If `req.PrefixGroup` has a matching prefix string: `suffixLen = max(1, len(req.InputTokens) - prefixLengths[req.PrefixGroup])` words. `Prompt = prefixString + strings.Repeat("word ", suffixLen)` (note: `InputTokens` already includes prefix tokens from the generator, so we subtract to avoid double-counting)
    - Otherwise: `Prompt = strings.Repeat("hello ", max(1, len(req.InputTokens)))`
  - Set `Unconstrained` from flag
  - Set `DeadlineUs` from `req.Deadline`
- **Remove prompt generation from `Send()`** — `Send()` uses `pending.Prompt` directly, no fallback. Single code path.
- **Update call site** at `observe_cmd.go:321`: change `requestToPending(req, idx, streaming)` to `requestToPending(req, idx, streaming, observeUnconstrainedOutput, prefixes, prefixLengths)`. The `prefixes` and `prefixLengths` maps are computed before `runObserveOrchestrator` and passed as parameters (or captured in closure scope).

**Step 4 — Run tests, verify pass:**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -run "TestBuildPrefixStrings|TestRequestToPending_PrependsPrefixString" -count=1
```

**Step 5 — Run full test suite (regression):**
```bash
cd .worktrees/observe-fidelity && go test ./cmd/... -count=1
```

**Step 6 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./cmd/...
```

**Step 7 — Commit:**
```
feat(observe): deterministic shared prefix strings for prefix groups

Implement BC-5: requests in the same prefix group share an identical
prompt prefix substring, enabling real server prefix cache activation.
Prefix content is deterministic given seed + group name.
```

---

#### Task 8: `--rtt-ms` CLI flag (BC-7) and DeadlineUs wiring

**Step 1 — Write failing test:**

In `cmd/observe_cmd_test.go`:

```go
func TestObserveCmd_RttMsFlag_PopulatesHeader(t *testing.T) {
    // Unit test: verify the wiring from rttMs variable to TraceHeader.
    // Integration test would require a full observe run; instead test the
    // construction directly.
    rttMs := 2.5
    header := workload.TraceHeader{
        Version:  1,
        TimeUnit: "us",
        Mode:     "real",
    }
    if rttMs > 0 {
        header.Network = &workload.TraceNetworkConfig{MeasuredRTTMs: rttMs}
    }
    require.NotNil(t, header.Network)
    assert.Equal(t, 2.5, header.Network.MeasuredRTTMs)

    // Zero rtt-ms means no Network config
    rttMs = 0
    header2 := workload.TraceHeader{Version: 1, TimeUnit: "us", Mode: "real"}
    if rttMs > 0 {
        header2.Network = &workload.TraceNetworkConfig{MeasuredRTTMs: rttMs}
    }
    assert.Nil(t, header2.Network)
}
```

**Step 2 — Run test, verify failure.**

**Step 3 — Implement:**

In `cmd/observe_cmd.go`:
- Add `--rtt-ms` float64 flag (default: 0)
- Validate in `runObserve`: `if rttMs < 0 || math.IsNaN(rttMs) || math.IsInf(rttMs, 0) { logrus.Fatalf("--rtt-ms must be a finite value >= 0, got %v", rttMs) }`
- In `runObserve`, after dispatch completes: if `rttMs > 0`, set `header.Network = &workload.TraceNetworkConfig{MeasuredRTTMs: rttMs}` (replacing the existing TODO). If `rttMs == 0`, leave `Network` nil (omitted from YAML)
- Also wire `DeadlineUs` in `Recorder.RecordRequest`: set `trace.DeadlineUs = pending.DeadlineUs` (the field already exists on `TraceRecord`; `PendingRequest` needs a `DeadlineUs int64` field; `requestToPending` sets it from `req.Deadline`)

**Step 4 — Run test, verify pass.**

**Step 5 — Run full test suite:**
```bash
cd .worktrees/observe-fidelity && go test ./... -count=1
```

**Step 6 — Lint:**
```bash
cd .worktrees/observe-fidelity && golangci-lint run ./...
```

**Step 7 — Commit:**
```
feat(observe): add --rtt-ms flag and wire DeadlineUs in TraceV2 output

Implement BC-7: --rtt-ms populates TraceHeader.Network.MeasuredRTTMs.
Also wires PendingRequest.DeadlineUs through to TraceRecord.DeadlineUs
(resolving TODO from #659).
```

---

### H. Test Strategy

| Contract | Test | Type | Invariant Verified |
|----------|------|------|--------------------|
| BC-1 | `TestRealClient_ChatFormat_UsesMessagesEndpoint` | Behavioral | Correct endpoint path and body format |
| BC-1 | `TestRealClient_DefaultFormat_UsesCompletions` | Regression | Default behavior preserved |
| BC-1 | `TestObserveCmd_InvalidAPIFormat_Fails` | Behavioral | Unknown format rejected |
| BC-2 | `TestRealClient_Streaming_SetsStreamOptions` | Behavioral | `stream_options.include_usage` present |
| BC-2 | `TestRealClient_NonStreaming_NoStreamOptions` | Behavioral | Non-streaming omits `stream_options` |
| BC-3 | `TestRealClient_NonStreaming_ExtractsFinishReason` | Behavioral | `finish_reason` extracted from JSON |
| BC-3 | `TestRealClient_Streaming_ExtractsFinishReason` | Behavioral | `finish_reason` extracted from last content chunk; usage-only chunk (empty choices) skipped |
| BC-3 | `TestRealClient_StreamingChat_ExtractsFinishReason` | Behavioral | `finish_reason` works with chat `delta` format |
| BC-4 | `TestRealClient_UnconstrainedOutput_OmitsMaxTokens` | Behavioral | `max_tokens` absent (chat) or MaxInt32 (completions) |
| BC-4 | `TestRealClient_ConstrainedOutput_IncludesMaxTokens` | Regression | Default behavior preserved |
| BC-5 | `TestBuildPrefixStrings_DeterministicAndDistinct` | Behavioral | Determinism + cross-group distinctness |
| BC-5 | `TestRequestToPending_PrependsPrefixString` | Behavioral | Prefix prepended, suffix filler = inputTokens - prefixLength |
| BC-6 | `TestTraceV2_FinishReason_RoundTrip` | Behavioral | CSV round-trip correctness (26 columns) |
| BC-6 | `TestTraceV2_BackwardCompat_25Columns` | Behavioral | 25-column CSV loads with FinishReason = "" |
| BC-7 | `TestObserveCmd_RttMsFlag_PopulatesHeader` | Behavioral | Flag → header wiring |

### I. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| TraceV2 schema change breaks existing trace files | Medium | High | New column appended at end; `parseTraceRecord` accepts both 25 and 26 columns via `len(row) < len(traceV2Columns)-1` guard. Add backward-compat test for 25-column loading |
| Chat completions response format differs from completions | Medium | Medium | `usage` field is identical in both formats. `choices` structure differs (`text` vs `message.content`; `text` vs `delta.content` for streaming) but we only extract `usage` and `finish_reason` (same path in both). Streaming parser must handle empty `choices[]` in usage-only chunks |
| Prefix word count ≠ actual server token count | High | Low | `server_input_tokens` captures ground truth; prefix sharing is about cache activation, not exact counts. Word count ≈ token count within tokenizer-dependent variance (1 word ≈ 1.0-1.5 tokens) |
| `--unconstrained-output` causes very long responses | Medium | Low | User opt-in only; global flag applies to all requests. For `/v1/completions` sends `math.MaxInt32` (not omitted — completions defaults to 16) |
| Server ignores `stream_options` — zero token counts | Medium | Medium | `stream_options.include_usage` is OpenAI API extension supported by vLLM ≥ 0.4 and SGLang ≥ 0.3. Unsupported servers: warning logged when `lastUsage == nil && chunkCount > 0` (R1) |
| Chat format introduces template overhead in `server_input_tokens` | High | Low | Expected and desirable — `server_input_tokens` reflects what the server actually tokenized, including chat template tokens. This improves calibration accuracy |

---

## Part 3: Quality Assurance

### J. Sanity Checklist

#### Plan-Specific Checks

- [x] All 7 behavioral contracts have GIVEN/WHEN/THEN/MECHANISM
- [x] THEN clauses describe observable behavior (no internal type names)
- [x] 8 tasks cover all 7 contracts
- [x] Task 1 (schema) must come before Task 2-3 (FinishReason wiring)
- [x] Task 5 (chat format) is independent of Tasks 3-4
- [x] Deviation log accounts for all differences from #660

#### Antipattern Rules (R1-R23)

- [x] **R1** (no silent data loss): `finish_reason` extraction logs warning on "length" and "abort"; streaming without usage logs warning when `lastUsage == nil`; errors propagate
- [x] **R3** (validate numeric params): `--rtt-ms` validated `>= 0`, not NaN/Inf; `--api-format` validated as enum in `runObserve` (CLI layer)
- [x] **R4** (construction sites): `PendingRequest` in `requestToPending` only; `RequestRecord` in `handleNonStreamingResponse`/`handleStreamingResponse` only; `TraceRecord` in `Recorder.RecordRequest` and `RequestsToTraceRecords` (latter defaults FinishReason to empty — correct for simulation traces). `RealClient` in `NewRealClient` only (functional options preserve backward compat)
- [x] **R6** (no logrus.Fatalf in sim/): TraceV2 schema is in `sim/workload/` but only adds a field — no new `logrus` calls
- [x] **R8** (no exported mutable maps): prefix strings map is local to orchestrator, not exported
- [x] **R10** (strict YAML): TraceV2 header already uses `KnownFields(true)`; no new YAML parsing
- [x] **R13** (interfaces for multiple impls): No new interfaces introduced
- [x] **R16** (config by module): New flags are observe-specific, grouped with existing observe flags
- [x] **R18** (CLI flag precedence): `--api-format`, `--unconstrained-output`, `--rtt-ms` have sensible defaults; no override conflicts

---

## Appendix: File-Level Implementation Details

### `sim/workload/tracev2.go`

- Add `FinishReason string` field to `TraceRecord` struct after `ErrorMessage`
- Append `"finish_reason"` to `traceV2Columns` slice (index 25, 26 total)
- `ExportTraceV2`: append `rec.FinishReason` to CSV row after `rec.ErrorMessage`
- `parseTraceRecord`: if `len(row) >= 26`, parse index 25 as `FinishReason`; if 25 columns, default to `""`
- `LoadTraceV2`: change guard from `len(row) < len(traceV2Columns)` to `len(row) < len(traceV2Columns)-1`
- Note: `RequestsToTraceRecords` (line 356) is a second `TraceRecord` construction site — `FinishReason` defaults to `""` (correct for simulation-generated records)

### `cmd/observe.go`

- Add `FinishReason string` to `RequestRecord`
- Add `apiFormat string` to `RealClient` (default `"completions"`)
- Add `type RealClientOption func(*RealClient)` and `WithAPIFormat(string) RealClientOption`
- Update `NewRealClient` signature to `(baseURL, apiKey, modelName, serverType string, opts ...RealClientOption)` — existing 4-arg callers compile unchanged
- `Send()`:
  - Use `pending.Prompt` directly for the prompt (no fallback generation — always populated by caller)
  - Branch on `c.apiFormat` for endpoint URL and body construction
  - Add `stream_options` when `pending.Streaming`
  - Endpoint-aware `max_tokens`: if unconstrained + chat → omit; if unconstrained + completions → `math.MaxInt32`; else normal
- `handleNonStreamingResponse`: extract `choices[0].finish_reason`
- `handleStreamingResponse`: extract `finish_reason` from content chunks only (check `len(choices) > 0` — skip usage-only chunks with empty choices). After loop: warn if `lastUsage == nil && chunkCount > 0`
- Warn on `finish_reason == "length"` and `finish_reason == "abort"`
- `Recorder.RecordRequest`: wire `FinishReason` and `DeadlineUs`

### `cmd/observe_cmd.go`

- New flags: `--api-format` (string, validated as enum), `--unconstrained-output` (bool), `--rtt-ms` (float64, validated >= 0, not NaN/Inf)
- `buildPrefixStrings(groups map[string]int, seed int64) map[string]string` — uses `hash/fnv` FNV-64 for per-group seed derivation, 100-word hardcoded vocabulary
- `requestToPending` combined final signature: `(req *sim.Request, reqIndex int, streaming bool, unconstrained bool, prefixes map[string]string, prefixLengths map[string]int) *PendingRequest` — always populates `Prompt`, sets `Unconstrained`, sets `DeadlineUs`
- Prefix groups extracted from `workload.Clients` before dispatch
- `runObserve`: wire `--rtt-ms` to `TraceHeader.Network.MeasuredRTTMs` (only when > 0)

### `cmd/observe_test.go`

- New tests: `TestRecorder_WiresFinishReason`, `TestRealClient_NonStreaming_ExtractsFinishReason`, `TestRealClient_Streaming_ExtractsFinishReason`, `TestRealClient_StreamingChat_ExtractsFinishReason`, `TestRealClient_Streaming_SetsStreamOptions`, `TestRealClient_ChatFormat_UsesMessagesEndpoint`, `TestRealClient_UnconstrainedOutput_OmitsMaxTokens`

### `cmd/observe_cmd_test.go`

- New tests: `TestBuildPrefixStrings_DeterministicAndDistinct`, `TestRequestToPending_PrependsPrefixString`, `TestObserveCmd_RttMsFlag_PopulatesHeader`, `TestObserveCmd_InvalidAPIFormat_Fails`

### `sim/workload/tracev2_test.go`

- New tests: `TestTraceV2_FinishReason_RoundTrip`, `TestTraceV2_BackwardCompat_25Columns`
- Update all 9 existing tests that construct `make([]string, 25)` to `make([]string, 26)`: `TestParseTraceRecord_InvalidInteger_ReturnsError`, `TestParseTraceRecord_InvalidDeadlineUs_ReturnsError`, `TestParseTraceRecord_InvalidServerInputTokens_ReturnsError`, `TestParseTraceRecord_NegativeDeadlineUs_ReturnsError`, `TestParseTraceRecord_NegativeInputTokens_ReturnsError`, `TestParseTraceRecord_NegativeOutputTokens_ReturnsError`, `TestParseTraceRecord_NegativeServerInputTokens_ReturnsError`, `TestParseTraceRecord_DeadlineBeforeArrival_ReturnsError`, `TestParseTraceRecord_InvalidReasonRatio_ReturnsError`
