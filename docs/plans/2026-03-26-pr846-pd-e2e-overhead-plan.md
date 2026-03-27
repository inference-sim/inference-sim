# PR #846: PD E2E Missing PostDecodeFixedOverhead Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix PD disaggregated E2E latency to include `PostDecodeFixedOverhead` (≈1.85 ms for trained-roofline), making it consistent with non-PD E2E.

**Architecture:** Add a delegation bridge `sim.Simulator.PostDecodeFixedOverhead()` → `i.sim.PostDecodeFixedOverhead()` → `InstanceSimulator.PostDecodeOverhead()`, then update `detectDecodeCompletions` to add the overhead when stamping `parent.CompletionTime`. `projectPDMetrics()` is unchanged because it already computes `e2e = parent.CompletionTime - parent.ArrivalTime`.

**Tech Stack:** Go 1.22, `sim` and `sim/cluster` packages.

**Closes:** #846

---

## Step 1.5 – Source Document Audit

**Source:** GitHub issue #846.

**Clarifications resolved:**
- CLARIFICATION C1: Issue offers two fix approaches ("add overhead to projection" vs "read RequestCompletionTimes[dec]"). We use a third, cleaner approach: stamp the overhead directly in `detectDecodeCompletions` so `parent.CompletionTime` becomes the authoritative client-visible completion time. This keeps `projectPDMetrics()` unchanged.
- CLARIFICATION C2: `RequestCompletionTimes[dec]` (= `itlSum + overhead + orig.ArrivalTime`) cannot be used to set `parent.CompletionTime` because `itlSum` excludes the prefill+transfer+scheduling delay — it is not an absolute time.
- CORRECTION: Issue suggests modifying `projectPDMetrics()`; the correct site is `detectDecodeCompletions` (cluster.go:599) because that is where the authoritative cluster-clock completion is stamped.

No contradictions with existing invariants. No missing information.

---

## Behavioral Contracts

**BC-1:** `sim.Simulator.PostDecodeFixedOverhead()` returns the same value as the underlying `LatencyModel.PostDecodeFixedOverhead()`. For `blackbox`/`roofline`/`cross-model` configs this is 0; for `trained-roofline` it is `int64(alpha1)`.

**BC-2:** `InstanceSimulator.PostDecodeOverhead()` returns the same value as the inner `sim.Simulator.PostDecodeFixedOverhead()`.

**BC-3:** After `detectDecodeCompletions` runs, for every completed parent request, `parent.CompletionTime = clusterClock + decodeInstance.PostDecodeOverhead()`. When overhead is 0, `parent.CompletionTime` equals the cluster clock (existing behavior preserved). When overhead > 0, `parent.CompletionTime` is `clusterClock + overhead`. Note: `parent.CompletionTime` can exceed `c.clock` at the moment of stamping; this is intentional and does not violate INV-3 (clock monotonicity) because `CompletionTime` is a derived timestamp, not the clock itself. Parity with the non-PD path: `recordRequestCompletion` applies `PostDecodeFixedOverhead` only when `len(req.OutputTokens) > 0`. In PD disaggregation, `KVTransferCompletedEvent.Execute` always sets `decodeSubReq.OutputTokens = orig.OutputTokens`, so the decode sub-request inherits the output token slice. This makes the unconditional application in the PD path safe: if `orig.OutputTokens` is nil/empty, `PostDecodeFixedOverhead()` is still added, but for `trained-roofline` that means +1850µs even for zero-output requests — an acceptable approximation since zero-output PD requests are pathological.

**BC-4:** After `projectPDMetrics()`, `RequestE2Es[parentID] = float64(parent.CompletionTime - parent.ArrivalTime)`. Since BC-3 now includes the overhead in `parent.CompletionTime`, E2E automatically includes overhead for `trained-roofline`. For all other backends (overhead=0), behavior is byte-identical to before.

**BC-5 (Regression):** For non-PD clusters (no parent requests), all metric values are byte-identical to before this change.

---

## Part 2: TDD Tasks

### Task 1 — `sim.Simulator.PostDecodeFixedOverhead()`

**Files:**
- Modify: `sim/simulator.go` (add method after line 277, near `CurrentClock()` and `SimHorizon()`)
- Test: `sim/simulator_test.go`

**Step 1: Write the failing test**

Open `sim/simulator_test.go` and add:

```go
// fixedOverheadModel is a test-only LatencyModel stub with configurable PostDecodeFixedOverhead.
// Placed at package level so all test functions can use it.
type fixedOverheadModel struct {
	overhead int64
}

func (m *fixedOverheadModel) StepTime(batch []*Request) int64      { return 1 }
func (m *fixedOverheadModel) QueueingTime(req *Request) int64       { return 0 }
func (m *fixedOverheadModel) OutputTokenProcessingTime() int64      { return 0 }
func (m *fixedOverheadModel) PostDecodeFixedOverhead() int64        { return m.overhead }

// BC-1: PostDecodeFixedOverhead() delegates to underlying LatencyModel.
func TestSimulator_PostDecodeFixedOverhead_DelegatesToModel(t *testing.T) {
	tests := []struct {
		name     string
		overhead int64
	}{
		{"zero (blackbox/roofline)", 0},
		{"positive (trained-roofline)", 1850},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			cfg := newTestSimConfig() // uses existing helper (sim/simulator_test.go:221)
			kvStore := MustNewKVStoreFromConfig(cfg.KVCacheConfig)
			model := &fixedOverheadModel{overhead: tc.overhead}
			s, err := NewSimulator(cfg, kvStore, model)
			if err != nil {
				t.Fatalf("NewSimulator: %v", err)
			}
			if got := s.PostDecodeFixedOverhead(); got != tc.overhead {
				t.Errorf("PostDecodeFixedOverhead() = %d, want %d", got, tc.overhead)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

```bash
cd .worktrees/pr846-pd-e2e-overhead
go test ./sim/... -run TestSimulator_PostDecodeFixedOverhead_DelegatesToModel -v
```

Expected: FAIL — `s.PostDecodeFixedOverhead undefined`

**Step 3: Implement**

Add after line 280 in `sim/simulator.go` (after `SimHorizon()`):

```go
// PostDecodeFixedOverhead returns the latency model's fixed per-request post-decode
// overhead in microseconds. Used by the cluster layer to include overhead in
// parent.CompletionTime when disaggregated decode sub-requests complete.
// Returns 0 for all backends except trained-roofline (BC-1, issue #846).
func (sim *Simulator) PostDecodeFixedOverhead() int64 {
	return sim.latencyModel.PostDecodeFixedOverhead()
}
```

**Step 4: Run test to verify it passes**

```bash
go test ./sim/... -run TestSimulator_PostDecodeFixedOverhead_DelegatesToModel -v
```

Expected: PASS

**Step 5: Lint**

```bash
golangci-lint run ./sim/...
```

Expected: 0 issues

**Step 6: Commit**

```bash
git add sim/simulator.go sim/simulator_test.go
git commit -m "feat(sim): expose PostDecodeFixedOverhead() on Simulator for cluster access (BC-1, #846)"
```

---

### Task 2 — `InstanceSimulator.PostDecodeOverhead()`

**Files:**
- Modify: `sim/cluster/instance.go` (add method near other delegation methods)
- Test: `sim/cluster/instance_test.go`

**Step 1: Write the failing test**

Open `sim/cluster/instance_test.go` and add:

```go
// BC-2: PostDecodeOverhead() delegates to inner sim.Simulator.PostDecodeFixedOverhead().
// Uses blackbox config (overhead=0) to verify the delegation path exists and returns 0.
func TestInstanceSimulator_PostDecodeOverhead_DelegatesToSim(t *testing.T) {
	cfg := newTestSimConfig() // uses existing helper (sim/cluster/instance_test.go:14)
	inst := NewInstanceSimulator("instance_0", cfg)
	// blackbox model always returns 0
	if got := inst.PostDecodeOverhead(); got != 0 {
		t.Errorf("PostDecodeOverhead() = %d, want 0 for blackbox model", got)
	}
}
```

**Step 2: Run test to verify it fails**

```bash
go test ./sim/cluster/... -run TestInstanceSimulator_PostDecodeOverhead_DelegatesToSim -v
```

Expected: FAIL — `inst.PostDecodeOverhead undefined`

**Step 3: Implement**

Add after `SimHorizon()` delegation or near other forwarding methods in `sim/cluster/instance.go`:

```go
// PostDecodeOverhead returns the fixed per-request post-decode overhead (µs)
// from the instance's underlying latency model. Used by detectDecodeCompletions
// to stamp parent.CompletionTime with the correct client-visible completion time.
// Returns 0 for blackbox/roofline/cross-model; non-zero for trained-roofline (BC-2, #846).
func (i *InstanceSimulator) PostDecodeOverhead() int64 {
	return i.sim.PostDecodeFixedOverhead()
}
```

**Step 4: Run test to verify it passes**

```bash
go test ./sim/cluster/... -run TestInstanceSimulator_PostDecodeOverhead_DelegatesToSim -v
```

Expected: PASS

**Step 5: Lint**

```bash
golangci-lint run ./sim/cluster/...
```

Expected: 0 issues

**Step 6: Commit**

```bash
git add sim/cluster/instance.go sim/cluster/instance_test.go
git commit -m "feat(cluster): expose PostDecodeOverhead() on InstanceSimulator (BC-2, #846)"
```

---

### Task 3 — Fix `detectDecodeCompletions` + update `ParentRequest` comment

**Files:**
- Modify: `sim/cluster/cluster.go` (line 599)
- Modify: `sim/cluster/parent_request.go` (CompletionTime field comment)
- Test: `sim/cluster/disaggregation_test.go`

**Step 1: Write the failing tests**

Add to `sim/cluster/disaggregation_test.go`:

```go
// BC-3: parent.CompletionTime >= c.clock when decode completes.
// For blackbox (overhead=0): CompletionTime == clusterClock at completion tick.
// Law: CompletionTime is always >= all phase timestamps that precede it.
func TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 {
			continue // incomplete (horizon-interrupted)
		}
		if parent.CompletionTime < parent.DecodeEnqueueTime {
			t.Errorf("parent %s: CompletionTime (%d) < DecodeEnqueueTime (%d)",
				parent.ID, parent.CompletionTime, parent.DecodeEnqueueTime)
		}
		if parent.CompletionTime < parent.TransferCompleteTime {
			t.Errorf("parent %s: CompletionTime (%d) < TransferCompleteTime (%d)",
				parent.ID, parent.CompletionTime, parent.TransferCompleteTime)
		}
	}
}

// BC-4 + overhead=0 regression: With blackbox (overhead=0), E2E from aggregated
// metrics equals CompletionTime - ArrivalTime (unchanged from pre-fix behavior).
func TestDisaggregation_E2E_IncludesOverhead_ZeroOverheadRegression(t *testing.T) {
	config := newTestDisaggDeploymentConfig(4, 2, 2)
	requests := newTestRequests(3)
	cs := NewClusterSimulator(config, requests, nil)
	mustRun(t, cs)

	m := cs.AggregatedMetrics()
	for _, parent := range cs.parentRequests {
		if parent.CompletionTime == 0 || parent.DecodeInstanceID == "" {
			continue // dropped or incomplete
		}
		e2e, ok := m.RequestE2Es[parent.ID]
		if !ok {
			t.Errorf("parent %s: no RequestE2Es entry after projectPDMetrics", parent.ID)
			continue
		}
		wantE2E := float64(parent.CompletionTime - parent.ArrivalTime)
		if e2e != wantE2E {
			t.Errorf("parent %s: RequestE2Es = %.0f, want %.0f (CompletionTime-ArrivalTime)",
				parent.ID, e2e, wantE2E)
		}
		// Law: E2E >= TTFT (CompletionTime includes overhead, decode always finishes after first token)
		ttft, hasTTFT := m.RequestTTFTs[parent.ID]
		if hasTTFT && e2e < ttft {
			t.Errorf("parent %s: E2E (%.0f) < TTFT (%.0f) — causality violated", parent.ID, e2e, ttft)
		}
	}
}
```

**Step 2: Run tests to verify they fail**

```bash
go test ./sim/cluster/... -run "TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps|TestDisaggregation_E2E_IncludesOverhead_ZeroOverheadRegression" -v
```

Expected: both PASS (not yet testing the overhead path — these tests confirm existing invariants hold before the fix and will keep passing after).

> Note: these tests pass before the fix because overhead=0 with blackbox. They are *regression* guards — they will fail if the fix introduces a regression.

**Step 3: Implement the fix**

In `sim/cluster/cluster.go`, change line 599:

```go
// Before:
parent.CompletionTime = c.clock

// After:
parent.CompletionTime = c.clock + inst.PostDecodeOverhead()
```

The full updated phase-2 loop (lines 597–601) becomes:

```go
// Phase 2: process in deterministic order
for _, subReqID := range completedIDs {
    parent := c.parentRequests[c.pendingDecodeCompletions[subReqID]]
    // Include PostDecodeFixedOverhead so parent.CompletionTime represents the
    // client-visible completion time (matching non-PD E2E semantics, issue #846).
    parent.CompletionTime = c.clock + inst.PostDecodeOverhead()
    delete(c.pendingDecodeCompletions, subReqID)
    c.pdDecodeCompletedCount++
}
```

In `sim/cluster/parent_request.go`, update the `CompletionTime` field comment:

```go
// CompletionTime has two meanings depending on outcome:
//   - Successful decode: set by detectDecodeCompletions to
//     clusterClock + decodeInstance.PostDecodeOverhead() when the decode
//     sub-request finishes its last step. Includes PostDecodeFixedOverhead
//     so that projectPDMetrics() computes the same client-visible E2E as
//     non-PD recordRequestCompletion (issue #846).
//   - Dropped at decode KV allocation: set to the DecodeRoutingEvent time (the point
//     when the drop was detected). CompletionTime < actual decode time (which never
//     happened). Use DecodeInstanceID == "" to distinguish dropped requests.
CompletionTime int64
```

**Step 4: Run regression tests**

```bash
go test ./sim/cluster/... -run "TestDisaggregation_CompletionTime_GeqAllPriorPhaseTimestamps|TestDisaggregation_E2E_IncludesOverhead_ZeroOverheadRegression" -v
```

Expected: both PASS (blackbox overhead=0, so no numeric change)

**Step 5: Run full suite**

```bash
go test ./... -count=1
```

Expected: all packages PASS

**Step 6: Lint**

```bash
golangci-lint run ./sim/... ./sim/cluster/...
```

Expected: 0 issues

**Step 7: Commit**

```bash
git add sim/cluster/cluster.go sim/cluster/parent_request.go sim/cluster/disaggregation_test.go
git commit -m "fix(cluster): include PostDecodeFixedOverhead in PD parent.CompletionTime (BC-3, BC-4, #846)

detectDecodeCompletions now stamps parent.CompletionTime as c.clock + inst.PostDecodeOverhead()
instead of c.clock alone. For blackbox/roofline/cross-model (overhead=0) the value is
byte-identical. For trained-roofline (~1850µs overhead), PD E2E now matches non-PD E2E."
```

---

### Task 4 — Update `docs/contributing/standards/invariants.md` (INV-PD-6 addendum)

**Files:**
- Modify: `docs/contributing/standards/invariants.md`

**Step 1: Locate the INV-PD-6 entry**

```bash
grep -n "INV-PD-6\|INV-PD" docs/contributing/standards/invariants.md | head -10
```

**Step 2: Add INV-PD-6b (or extend INV-PD-6)**

Find the INV-PD-6 block and add after it:

```markdown
**INV-PD-6b — CompletionTime includes PostDecodeFixedOverhead:**
For all completed parent requests (`DecodeInstanceID != ""`), `parent.CompletionTime`
equals the cluster clock at decode completion plus the decode instance's
`PostDecodeFixedOverhead`. For backends where overhead is 0 (blackbox, roofline,
cross-model), this is identical to the raw clock tick. For `trained-roofline`, this
adds ≈1.85 ms, matching how `recordRequestCompletion` computes non-PD E2E.

**Verification:** `parent.CompletionTime >= c.clock` (trivially), and when overhead > 0:
`parent.CompletionTime - parent.ArrivalTime` equals the non-PD E2E for the same
request under the same latency model.

**Filed:** issue #846, fixed in this PR.
```

**Step 3: Commit**

```bash
git add docs/contributing/standards/invariants.md
git commit -m "docs(standards): add INV-PD-6b for CompletionTime PostDecodeFixedOverhead invariant (#846)"
```

---

## Part 3: Sanity Checklist

- [ ] `go build ./...` passes
- [ ] `go test ./... -count=1` — all packages green
- [ ] `golangci-lint run ./...` — 0 issues
- [ ] `parent.CompletionTime = c.clock + inst.PostDecodeOverhead()` is the only code change to non-test, non-doc files in `cluster.go`
- [ ] For blackbox (overhead=0): no numeric change in any output
- [ ] `PostDecodeFixedOverhead()` added to `sim.Simulator` (not to any interface — avoids interface bloat)
- [ ] `PostDecodeOverhead()` added to `InstanceSimulator` (delegates cleanly)
- [ ] `ParentRequest.CompletionTime` comment updated
- [ ] INV-PD-6b documented in `invariants.md`
- [ ] No new exported types or interfaces introduced
- [ ] No changes to `docs/contributing/pr-workflow.md` or `CLAUDE.md` (not a source-of-truth file for this change)

---

## Appendix

### Files Changed

| File | Change Type | Description |
|------|-------------|-------------|
| `sim/simulator.go` | Add method | `PostDecodeFixedOverhead() int64` at line ~281 |
| `sim/simulator_test.go` | Add test | `TestSimulator_PostDecodeFixedOverhead_DelegatesToModel` |
| `sim/cluster/instance.go` | Add method | `PostDecodeOverhead() int64` |
| `sim/cluster/instance_test.go` | Add test | `TestInstanceSimulator_PostDecodeOverhead_DelegatesToSim` |
| `sim/cluster/cluster.go` | Bug fix | `detectDecodeCompletions` line 599: `c.clock + inst.PostDecodeOverhead()` |
| `sim/cluster/parent_request.go` | Doc update | `CompletionTime` field comment |
| `sim/cluster/disaggregation_test.go` | Add tests | BC-3 + BC-4 regression invariants |
| `docs/contributing/standards/invariants.md` | Doc update | INV-PD-6b |
| `docs/plans/2026-03-26-pr846-pd-e2e-overhead-plan.md` | New file | This plan |

### Deviation Log

| ID | Type | Description |
|----|------|-------------|
| D1 | CORRECTION | Issue suggests modifying `projectPDMetrics()`; we fix `detectDecodeCompletions` instead. Reason: projectPDMetrics already computes `e2e = parent.CompletionTime - parent.ArrivalTime`; stamping the overhead at detection time is the single correct place and avoids requiring the overhead in the projection function. |
| D2 | CORRECTION | Issue offers "read RequestCompletionTimes[dec]" as an alternative; we rejected this because `RequestCompletionTimes[dec] = req.FirstTokenTime + itlSum + overhead + req.ArrivalTime` and for decode sub-requests `req.FirstTokenTime = 0` (TTFT recording never fires since ProgressIndex starts at inputLen), so it reduces to `itlSum + overhead + orig.ArrivalTime`. This excludes the scheduling delay of the decode sub-request and is not an absolute wall-clock time — using it would produce a wrong (too-small) E2E. |
| D3 | CLARIFICATION | Tests use blackbox (overhead=0) for cluster-level integration to avoid needing a full trained-roofline hardware config. The non-zero overhead path is tested at `sim.Simulator` level via a `fixedOverheadModel` stub. |
