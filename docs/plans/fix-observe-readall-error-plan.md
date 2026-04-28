# fix(observe): handle io.ReadAll error on non-200 HTTP response

**Goal:** Log a warning when `io.ReadAll` fails on a non-200 HTTP response body, so operators know the error message in the trace may be incomplete.
**Source:** [GitHub issue #1120](https://github.com/inference-sim/inference-sim/issues/1120)
**Closes:** Fixes #1120

## Behavioral Contracts

BC-1: ReadAll failure warning
- GIVEN a server returns a non-200 HTTP status code AND the response body read fails (e.g., connection reset mid-error-response)
- WHEN `RealClient.Send` processes the response
- THEN a `logrus.Warn` message is emitted containing "failed to read error response body" AND `record.Status` is `"error"` AND `record.ErrorMessage` contains the HTTP status code and a note that the body read failed
- WHEN the body read failure is specifically a timeout (isTimeoutError returns true)
- THEN `record.Status` is `"timeout"` (not `"error"`) AND `record.ErrorMessage` contains "body read timed out", consistent with success-path handlers

BC-2: ReadAll success preserves existing behavior
- GIVEN a server returns a non-200 HTTP status code AND the response body reads successfully
- WHEN `RealClient.Send` processes the response
- THEN `record.Status` is `"error"` AND `record.ErrorMessage` contains the HTTP status code and the full body text AND no warning is logged about body read failure

## Tasks

### Task 1: Add test for ReadAll failure on non-200 response (BC-1)

**Files:** modify `cmd/observe_test.go`

**Test:**
```go
// TestRealClient_ServerError_BodyReadFailure verifies BC-1: when a non-200
// response body read fails (e.g., connection reset), a warning is logged
// and the error message notes the incomplete body.
func TestRealClient_ServerError_BodyReadFailure(t *testing.T) {
	hook := installLogHook(t)

	// Server sends a non-200 status code then abruptly closes the connection
	// mid-body, causing io.ReadAll to return an error.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hijacker, ok := w.(http.Hijacker)
		if !ok {
			t.Fatal("expected http.Hijacker")
		}
		conn, buf, err := hijacker.Hijack()
		if err != nil {
			t.Fatal(err)
		}
		// Write a partial HTTP response with Content-Length mismatch to force read error.
		_, _ = buf.WriteString("HTTP/1.1 500 Internal Server Error\r\n")
		_, _ = buf.WriteString("Content-Length: 1000\r\n")
		_, _ = buf.WriteString("\r\n")
		_, _ = buf.WriteString("partial error bo")
		_ = buf.Flush()
		_ = conn.Close()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 3, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "error" {
		t.Errorf("status = %q, want error", record.Status)
	}
	if !strings.Contains(record.ErrorMessage, "HTTP 500") {
		t.Errorf("ErrorMessage should contain HTTP status code, got %q", record.ErrorMessage)
	}
	if !strings.Contains(record.ErrorMessage, "body read failed") {
		t.Errorf("ErrorMessage should note body read failure, got %q", record.ErrorMessage)
	}
	if !hook.hasEntry("failed to read error response body") {
		t.Error("expected warning about failed body read, but no matching log entry found")
	}
}
```

**Impl:** No implementation yet — test must fail first.

**Verify:** `cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork2-for-paralelworks/inference-sim/.worktrees/fix-observe-readall-error && go test ./cmd/... -run TestRealClient_ServerError_BodyReadFailure -v` (expect FAIL)
**Lint:** `cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork2-for-paralelworks/inference-sim/.worktrees/fix-observe-readall-error && golangci-lint run ./cmd/...`

### Task 2: Add test for ReadAll success on non-200 response (BC-2)

**Files:** modify `cmd/observe_test.go`

**Test:**
```go
// TestRealClient_ServerError_BodyReadSuccess verifies BC-2: when a non-200
// response body reads successfully, the full body is included in ErrorMessage
// and no body-read-failure warning is logged.
func TestRealClient_ServerError_BodyReadSuccess(t *testing.T) {
	hook := installLogHook(t)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		_, _ = fmt.Fprint(w, "upstream timeout")
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		RequestID: 4, InputTokens: 10, Streaming: false,
		Prompt: strings.Repeat("hello ", 10),
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.Status != "error" {
		t.Errorf("status = %q, want error", record.Status)
	}
	if !strings.Contains(record.ErrorMessage, "HTTP 502") {
		t.Errorf("ErrorMessage should contain HTTP 502, got %q", record.ErrorMessage)
	}
	if !strings.Contains(record.ErrorMessage, "upstream timeout") {
		t.Errorf("ErrorMessage should contain body text, got %q", record.ErrorMessage)
	}
	if hook.hasEntry("failed to read error response body") {
		t.Error("no body-read-failure warning expected when body reads successfully")
	}
}
```

**Impl:** No implementation yet — this test should already PASS against existing code (it verifies existing behavior is preserved). If it does pass, good. If not, investigate.

**Verify:** `cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork2-for-paralelworks/inference-sim/.worktrees/fix-observe-readall-error && go test ./cmd/... -run TestRealClient_ServerError_BodyReadSuccess -v` (expect PASS)
**Lint:** `cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork2-for-paralelworks/inference-sim/.worktrees/fix-observe-readall-error && golangci-lint run ./cmd/...`

### Task 3: Implement the fix (BC-1)

**Files:** modify `cmd/observe.go`

**Impl:** Change line 199 from:
```go
bodyData, _ := io.ReadAll(resp.Body)
record.Status = "error"
record.ErrorMessage = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyData))
return record, nil
```
to:
```go
bodyData, readErr := io.ReadAll(resp.Body)
record.Status = "error"
if readErr != nil {
	logrus.Warnf("observe: request %d: failed to read error response body: %v", record.RequestID, readErr)
	record.ErrorMessage = fmt.Sprintf("HTTP %d: (body read failed: %v)", resp.StatusCode, readErr)
} else {
	record.ErrorMessage = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(bodyData))
}
return record, nil
```

**Verify:** `cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork2-for-paralelworks/inference-sim/.worktrees/fix-observe-readall-error && go test ./cmd/... -run "TestRealClient_ServerError" -v` (expect ALL PASS including the new BodyReadFailure test)
**Lint:** `cd /Users/toslali/Desktop/work/ibm/projects/llm-inference/study/inference-llmd/blis-main-fork2-for-paralelworks/inference-sim/.worktrees/fix-observe-readall-error && golangci-lint run ./cmd/...`

## Sanity Checklist

- [ ] **R1 (silent continue):** No `continue` added — n/a.
- [ ] **R2 (determinism):** No map iteration or float accumulation — n/a.
- [ ] **R4 (construction sites):** No new struct fields — n/a.
- [ ] **R5 (error swallowing):** This PR *fixes* an error swallowing bug. The `readErr` is now logged and included in the error message.
- [ ] **R8 (exported mutable maps):** No maps — n/a.
- [ ] **R9 (YAML pointer types):** No YAML — n/a.
- [ ] **R10 (strict YAML):** No YAML — n/a.
- [ ] **INV-6 (determinism):** Warning goes to stderr via logrus. No stdout change.
- [ ] **No new CLI flags, types, or interfaces.**
- [ ] **Existing `TestRealClient_ServerError_RecordsError` still passes** (unchanged behavior for successful body reads).
