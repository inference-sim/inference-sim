# Fix: blis observe ignores per-client streaming flag — Implementation Plan

**Goal:** Make `blis observe` respect the per-request `Streaming` field from the workload spec instead of applying a single global boolean.
**Source:** [GitHub Issue #828](https://github.com/inference-sim/inference-sim/issues/828)
**Closes:** Fixes #828

## Behavioral Contracts

BC-1: Per-request streaming propagation
- GIVEN a workload spec with a client declaring `streaming: false`
- WHEN `blis observe` dispatches requests for that client (without `--no-streaming`)
- THEN each `PendingRequest.Streaming` equals `req.Streaming` from the generator

BC-2: Global override via --no-streaming
- GIVEN a workload spec with a client declaring `streaming: true`
- WHEN `blis observe` is invoked with `--no-streaming`
- THEN all `PendingRequest.Streaming` are `false` regardless of per-request values

BC-3: Default behavior preserved
- GIVEN a workload spec with clients having mixed streaming values
- WHEN `blis observe` is invoked without `--no-streaming`
- THEN each request's streaming flag matches its client's spec value

## Change Summary

Three functions touched in `cmd/observe_cmd.go`:

1. **`runObserveCmd`** (call site, line 224/266): Remove `streaming := !observeNoStreaming`. Pass `observeNoStreaming` directly to orchestrator.
2. **`runObserveOrchestrator`** (signature, line 311): Rename param `streaming bool` → `noStreaming bool`.
3. **`requestToPending`** (signature, line 528 + body line 557): Replace `streaming bool` param with `noStreaming bool`. Set `Streaming: req.Streaming && !noStreaming`.

All existing test call sites for `runObserveOrchestrator` pass `false` for the streaming param, which maps to `false` for `noStreaming` — no semantic change needed. Same for `requestToPending` tests where `false` was passed for streaming — but note the boolean semantics flip: old `streaming=false` meant "non-streaming", new `noStreaming=false` means "no override" (per-request value used). Existing prefix tests pass `req.Streaming=false` (zero value) so `false && !false = false` — same result.

## Tasks

### Task 1: Write test for per-request streaming propagation (BC-1, BC-2, BC-3)

**Files:** modify `cmd/observe_cmd_test.go`

**Test:**
```go
func TestRequestToPending_UsesPerRequestStreaming(t *testing.T) {
	streamingReq := &sim.Request{
		ID:          "stream-req",
		InputTokens: make([]int, 5),
		Streaming:   true,
	}
	nonStreamingReq := &sim.Request{
		ID:          "nostream-req",
		InputTokens: make([]int, 5),
		Streaming:   false,
	}

	// BC-1 / BC-3: without global override, per-request value propagates
	p1 := requestToPending(streamingReq, 0, false, false, nil, nil)
	if !p1.Streaming {
		t.Error("expected Streaming=true for streaming request when noStreaming=false")
	}
	p2 := requestToPending(nonStreamingReq, 1, false, false, nil, nil)
	if p2.Streaming {
		t.Error("expected Streaming=false for non-streaming request when noStreaming=false")
	}

	// BC-2: --no-streaming overrides per-request value to false
	p3 := requestToPending(streamingReq, 2, true, false, nil, nil)
	if p3.Streaming {
		t.Error("expected Streaming=false when noStreaming=true overrides req.Streaming=true")
	}
}
```

**Verify (expect fail):** `go test ./cmd/... -run TestRequestToPending_UsesPerRequestStreaming`

### Task 2: Implement the fix (BC-1, BC-2, BC-3)

**Files:** modify `cmd/observe_cmd.go`

**Changes:**
1. **Line 224**: Remove `streaming := !observeNoStreaming`
2. **Line 266**: Change `streaming` → `observeNoStreaming` in `runObserveOrchestrator` call
3. **Line 311**: Rename parameter `streaming bool` → `noStreaming bool` in `runObserveOrchestrator` signature
4. **Line 372**: Change `streaming` → `noStreaming` in `requestToPending` call
5. **Line 528**: Rename parameter `streaming` → `noStreaming` in `requestToPending` signature
6. **Line 557**: Change `Streaming: streaming` → `Streaming: req.Streaming && !noStreaming`

**Verify:** `go test ./cmd/... -run TestRequestToPending`
**Full test:** `go test ./cmd/...`
**Lint:** `golangci-lint run ./cmd/...`

## Sanity Checklist

- [x] R1 (silent continue): No new error paths
- [x] R2 (determinism): No map iteration or floating-point changes
- [x] R4 (construction sites): `requestToPending` — 1 call site (line 372); `runObserveOrchestrator` — 1 production call site (line 266), ~10 test call sites (all pass `false` — semantics preserved under rename)
- [x] R13 (behavioral contracts): THEN clauses describe observable `PendingRequest.Streaming` values
- [x] R14 (single-module): Change is self-contained in `cmd/observe_cmd.go`
- [x] R23 (code path parity): Fix brings `observe` path into alignment with `run` path for `Streaming`
- [x] INV-6 (determinism): No non-deterministic output changes
