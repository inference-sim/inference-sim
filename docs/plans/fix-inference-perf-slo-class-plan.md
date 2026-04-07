# Plan: Fix inference_perf SLOClass regression (#965)

**PR size:** Medium (behavioral change; "when in doubt, tier up" from Small)
**Closes:** #965
**Branch:** `fix-inference-perf-slo-class`

---

## Header

**Goal:** Fix TTFT regression caused by `inference_perf` workload generating `SLOClass: "batch"` requests that are serialized by the deferred queue introduced in commit `8bc7a48c` (#841). Restore pre-#841 latency behavior for all training and simulation experiments that use `inference_perf` workloads. Add tests to catch this class of regression in the future.

**Source:** Issue #965 (diagnosed in comments: `8bc7a48c` is the regression commit; controlled experiment confirms changing `SLOClass: "batch"` → `"standard"` in `inference_perf.go` restores correct TTFT exactly).

**Deviation Log:**
- No deviations from source. No ambiguities. No prior plan existed.

---

## Part 1: Design Validation

### A. Behavioral Contracts

**BC-1 (SLO class correctness):** `ExpandInferencePerfSpec` MUST generate clients with `SLOClass == "standard"`. No client produced by `ExpandInferencePerfSpec` SHALL have `SLOClass == "batch"` or `SLOClass == "background"`.

**BC-2 (No deferred-queue serialization):** A cluster simulation running requests with `SLOClass: "standard"` at a rate where requests overlap MUST NOT be serialized by the deferred queue. Specifically, when 10 requests arrive densely (every 10µs) in a single-instance cluster with the default blackbox config, mean TTFT MUST be less than 15ms (well below the ~100ms mean TTFT that full serialization would produce; see Section F).

**BC-3 (Batch SLO IS serialized — guard validity):** The counter-test: 10 requests with `SLOClass: "batch"` arriving densely in the same setup MUST produce mean TTFT ≥ 15ms, confirming the deferred queue does serialize batch requests and that BC-2's bound is a real discriminator.

**BC-4 (scenarios.go unchanged):** `SLOClass: "batch"` assignments in `scenarios.go` MUST remain unchanged. Those model explicitly batch/background traffic clients and are out of scope.

**BC-5 (No regression in existing tests):** All existing tests in `sim/workload/` and `sim/cluster/` MUST continue to pass.

### B. Component Interaction

Two files modified, one extended:
- `sim/workload/inference_perf.go`: 3-line value change (production)
- `sim/workload/inference_perf_test.go`: one new unit test (BC-1)
- `sim/cluster/cluster_deferred_test.go`: two new integration tests (BC-2, BC-3)

No new types, interfaces, CLI flags, or construction sites.

### C. Risks

- **Risk:** Other `inference_perf`-style callers relied on `"batch"` SLO class semantics. **Mitigation:** There are no such callers — the SLO class was a semantically inert label before `8bc7a48c`. All existing tests check generated request counts/distributions, not SLO class.
- **Risk:** BC-2/BC-3 tests are fragile due to latency config assumptions. **Mitigation:** Tests use `newTestDeploymentConfig(1)` (project-standard test helper) and assert on a bound of 15ms that is ~2.4× above the expected non-serialized TTFT (~6.2ms) and ~6.7× below the serialized TTFT (~100ms mean). See Section F for derivation. *(Deviation: initial draft used 5ms; corrected in Section F after finding alpha/beta coefficients were swapped in the original derivation.)*

### D. Known Follow-Up (Out of Scope)

`scenarios.go` also uses `SLOClass: "batch"` for `ScenarioBurstyTraffic` and `ScenarioPrefixHeavy` clients. These were written before `8bc7a48c` and will exhibit deferred-queue serialization when run through a cluster. A follow-up issue should be filed to audit `scenarios.go` SLO classes. This PR does not touch `scenarios.go`.

### E. Invariants Affected

- INV-1 (request conservation): unaffected.
- INV-9 (oracle knowledge boundary): unaffected — `SLOClass` is read at admission time, not by the execution engine.

### F. TTFT Bound Derivation (BC-2 / BC-3)

Default config from `newTestDeploymentConfig(1)` — `sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs)`:
- **BetaCoeffs = [1000, 10, 5]** → `stepTime = 1000 + 10×cacheMissTokens + 5×decodeTokens µs`
- **AlphaCoeffs = [100, 1, 100]** → `QueueingTime = 100 + 1×inputLen µs`
- Backend: `"blackbox"`

Requests from `newDeferredTestRequests(10, sloClass)`: 50 input tokens, 20 output tokens, arrivals at `t = i×10µs`.

**Non-serialized (standard):** All 10 requests arrive within 90µs and queue together. QueueingTime per request = 100+50 = 150µs. Batched prefill (10 requests × 50 tokens = 500 cacheMissTokens): stepTime = 1000+10×500 = 6000µs. Mean TTFT ≈ 150 + 6000 = **6150µs ≈ 6.2ms**.

**Serialized (batch):** Each request processes alone. QueueingTime=150µs, prefill step=1000+10×50=1500µs, 20 decode steps at 1000+5×1≈1005µs each = 20100µs. Full request duration ≈ 21750µs. Request i waits i×21750µs. Mean wait ≈ 4.5×21750 = 97875µs. Mean TTFT ≈ 150+1500+97875 ≈ **99500µs ≈ 100ms**.

**Bound: 15ms** separates the two cases with ~2.4× margin on the low side (6.2ms) and ~6.7× on the high side (100ms).

---

## Part 2: Executable Tasks

### Task 1: Unit test — BC-1 (RED → GREEN)

**Test file:** `sim/workload/inference_perf_test.go`

```go
// TestInferencePerfClients_SLOClass_IsStandard asserts BC-1: no client from
// ExpandInferencePerfSpec uses SLOClass "batch" or "background".
// Regression guard for issue #965 (commit 8bc7a48c deferred-queue interaction).
func TestInferencePerfClients_SLOClass_IsStandard(t *testing.T) {
	spec := &InferencePerfSpec{
		SharedPrefix: &SharedPrefixSpec{
			NumUniqueSystemPrompts:  2,
			NumUsersPerSystemPrompt: 3,
			SystemPromptLen:         100,
			QuestionLen:             200,
			OutputLen:               50,
			EnableMultiTurnChat:     false,
		},
		Stages: []StageSpec{
			{Rate: 5.0, Duration: 60},
		},
	}
	ws, err := ExpandInferencePerfSpec(spec, 42)
	if err != nil {
		t.Fatalf("ExpandInferencePerfSpec: %v", err)
	}
	if len(ws.Clients) == 0 {
		t.Fatal("expected at least one client")
	}
	for _, c := range ws.Clients {
		if c.SLOClass == "batch" || c.SLOClass == "background" {
			t.Errorf("client %q has SLOClass %q; inference_perf must use \"standard\" (regression: issue #965)",
				c.ID, c.SLOClass)
		}
	}
}
```

**Run (RED):** `go test ./sim/workload/... -run TestInferencePerfClients_SLOClass_IsStandard -v`
Expected: FAIL — current code assigns `"batch"`.

**Implement:** In `sim/workload/inference_perf.go`, change all 3 occurrences of `SLOClass: "batch"` to `SLOClass: "standard"` (lines 131, 183, 237).

**Run (GREEN):** `go test ./sim/workload/... -run TestInferencePerfClients_SLOClass_IsStandard -v`
Expected: PASS.

**Full package:** `go test ./sim/workload/...`
Expected: all pass (no regressions).

**Lint:** `golangci-lint run ./sim/workload/...`

**Commit:** `fix(workload): change inference_perf SLOClass from "batch" to "standard" (#965)`

---

### Task 2: Integration tests — BC-2 and BC-3

**Test file:** `sim/cluster/cluster_deferred_test.go` (append to existing file)

```go
// TestDeferredQueue_StandardSLONotSerialized asserts BC-2: standard-class requests
// produced by inference_perf workloads are NOT serialized by the deferred queue.
//
// Setup: 10 requests with SLOClass "standard" arriving every 10µs — dense enough
// that most requests are in-flight simultaneously. The deferred queue intercept fires
// only for "batch"/"background"; standard requests bypass it entirely.
//
// Bound derivation (see plan Section F):
//   Non-serialized mean TTFT ≈ 6ms (QueueingTime=150µs + batched prefill=6000µs).
//   Serialized mean TTFT ≈ 100ms (requests processed one-by-one, each ~21.75ms).
//   Bound 15ms: ~2.4× above non-serialized (6.2ms) and ~6.7× below serialized (100ms).
//
// Regression guard for issue #965.
func TestDeferredQueue_StandardSLONotSerialized(t *testing.T) {
	requests := newDeferredTestRequests(10, "standard")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 10 {
		t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
	}

	ttftMeanMs := float64(m.TTFTSum) / float64(m.CompletedRequests) / 1000.0
	const boundMs = 15.0
	if ttftMeanMs >= boundMs {
		t.Errorf("mean TTFT %.2fms >= bound %.1fms: standard requests are being serialized by the deferred queue (regression: issue #965)",
			ttftMeanMs, boundMs)
	}
}

// TestDeferredQueue_BatchSLOIsSerializedAboveBound asserts BC-3: batch-class requests
// ARE serialized by the deferred queue (guard validity check — confirms the bound in
// TestDeferredQueue_StandardSLONotSerialized is a real discriminator, not vacuously passing).
func TestDeferredQueue_BatchSLOIsSerializedAboveBound(t *testing.T) {
	requests := newDeferredTestRequests(10, "batch")
	cfg := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(cfg, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	if m.CompletedRequests != 10 {
		t.Fatalf("completed %d requests, want 10", m.CompletedRequests)
	}

	ttftMeanMs := float64(m.TTFTSum) / float64(m.CompletedRequests) / 1000.0
	const boundMs = 15.0
	if ttftMeanMs < boundMs {
		t.Errorf("mean TTFT %.2fms < bound %.1fms: batch requests are NOT being serialized — deferred queue may be broken",
			ttftMeanMs, boundMs)
	}
}
```

**Run (both GREEN):** `go test ./sim/cluster/... -run "TestDeferredQueue_StandardSLONotSerialized|TestDeferredQueue_BatchSLOIsSerializedAboveBound" -v`
Both tests should pass immediately (Task 1 already fixed the SLO class; the batch counter-test simply confirms existing deferred-queue behavior).

**Full package:** `go test ./sim/cluster/...`
Expected: all pass.

**Lint:** `golangci-lint run ./sim/cluster/...`

**Commit:** `test(cluster): assert standard SLO bypasses deferred queue; batch SLO is serialized (#965)`

---

### Task 3: Full verification

```bash
go build ./...
go test ./...
golangci-lint run ./...
```

All must pass with zero errors.

---

## Part 3: Test Strategy

| Contract | Test | Type | File |
|----------|------|------|------|
| BC-1: no batch/background SLO in inference_perf | `TestInferencePerfClients_SLOClass_IsStandard` | Unit | `sim/workload/inference_perf_test.go` |
| BC-2: standard SLO not serialized | `TestDeferredQueue_StandardSLONotSerialized` | Integration | `sim/cluster/cluster_deferred_test.go` |
| BC-3: batch SLO IS serialized (guard validity) | `TestDeferredQueue_BatchSLOIsSerializedAboveBound` | Integration | `sim/cluster/cluster_deferred_test.go` |
| BC-4: scenarios.go unchanged | No change → existing tests unaffected | — | — |
| BC-5: no regressions | `go test ./...` | — | — |

---

## Appendix

### Files Modified

| File | Change |
|------|--------|
| `sim/workload/inference_perf.go` | Lines 131, 183, 237: `SLOClass: "batch"` → `SLOClass: "standard"` |
| `sim/workload/inference_perf_test.go` | Append `TestInferencePerfClients_SLOClass_IsStandard` |
| `sim/cluster/cluster_deferred_test.go` | Append `TestDeferredQueue_StandardSLONotSerialized` + `TestDeferredQueue_BatchSLOIsSerializedAboveBound` |

### Sanity Checklist

- [ ] `go build ./...` passes
- [ ] `go test ./...` passes
- [ ] `golangci-lint run ./...` passes
- [ ] `scenarios.go` grep confirms no `inference_perf.go` lines were touched
- [ ] Issue #965 closed by this PR
