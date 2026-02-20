# CLI Metric Observability Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make all computed simulation metrics visible in CLI output and make stdout deterministic by design.

**The problem today:** BLIS computes PreemptionRate, CacheHitRate, KVThrashingRate, and per-SLO-class distributions internally but never prints them to stdout — users can't observe KV cache behavior, prefix caching effectiveness, or per-class fairness from the CLI. Additionally, stdout includes wall-clock fields (`simulation_duration_s`, timestamps) that vary between runs, breaking the determinism invariant for automated comparison pipelines. Finally, there's no way to verify request conservation from CLI output when requests are still in-flight at simulation end.

**What this PR adds:**
1. **KV cache metrics section** — prints PreemptionRate, CacheHitRate, KVThrashingRate to stdout when any value is nonzero (same pattern as anomaly counters)
2. **Per-SLO metrics section** — prints per-class TTFT and E2E distributions when multiple SLO classes are present in the workload
3. **Deterministic stdout** — removes wall-clock fields (`simulation_duration_s`, `sim_start_timestamp`, `sim_end_timestamp`) from the JSON output and moves wall-clock timing to stderr via logrus
4. **Conservation counts** — adds `still_queued`, `still_running`, `injected_requests` to the cluster-level JSON output

**Why this matters:** These changes unblock 10+ hypothesis experiments (H1, H2, H8, H9, H10, H15, H18, H20) that need to observe KV cache behavior and per-SLO metrics. Deterministic stdout enables automated experiment comparison without fragile field filtering.

**Architecture:** Changes span two layers: `sim/metrics_utils.go` and `sim/metrics.go` (JSON output struct and SaveResults), and `cmd/root.go` (new output sections for KV cache and per-SLO metrics). The `sim/cluster/metrics.go` ComputePerSLODistributions() function already exists — we just need to call it from `cmd/root.go`.

**Source:** GitHub issues #271, #272, #273

**Closes:** Fixes #271, fixes #272, fixes #273

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR improves CLI output observability across three dimensions:

1. **Surface hidden metrics** (#271): KV cache metrics (preemption rate, cache hit rate, thrashing rate) and per-SLO-class distributions are already computed in `sim/cluster/metrics.go` but never printed from `cmd/root.go`. We add two new stdout sections following the existing anomaly counters pattern.

2. **Deterministic stdout** (#272): Three wall-clock fields in `MetricsOutput` (`simulation_duration_s`, `sim_start_timestamp`, `sim_end_timestamp`) are diagnostic, not simulation data. We remove them from the JSON struct and log wall-clock timing to stderr instead.

3. **Conservation visibility** (#273): Add `still_queued`, `still_running`, `injected_requests` fields to `MetricsOutput` so conservation can be verified from CLI output even with finite horizons.

**Adjacent blocks:** `sim/metrics.go` (SaveResults), `sim/metrics_utils.go` (MetricsOutput struct), `cmd/root.go` (output printing), `sim/cluster/metrics.go` (RawMetrics, ComputePerSLODistributions).

**No deviations from source issues.**

### B) Behavioral Contracts

**Positive Contracts:**

**BC-1: KV cache metrics printed when nonzero**
- GIVEN a cluster simulation that produces nonzero PreemptionRate, CacheHitRate, or KVThrashingRate
- WHEN the simulation completes and results are printed
- THEN stdout MUST contain a `=== KV Cache Metrics ===` section with PreemptionRate, CacheHitRate, and KVThrashingRate values
- MECHANISM: `cmd/root.go` checks `rawMetrics` fields after `CollectRawMetrics` and prints when any is nonzero

**BC-2: KV cache metrics omitted when all zero**
- GIVEN a cluster simulation where PreemptionRate, CacheHitRate, and KVThrashingRate are all zero
- WHEN the simulation completes
- THEN stdout MUST NOT contain a `=== KV Cache Metrics ===` section
- MECHANISM: Same conditional guard pattern as anomaly counters

**BC-3: Per-SLO metrics printed when multiple classes present**
- GIVEN a workload with multiple SLO classes (e.g., realtime + batch)
- WHEN the simulation completes
- THEN stdout MUST contain a `=== Per-SLO Metrics ===` section with per-class TTFT and E2E distributions
- MECHANISM: `cmd/root.go` calls `ComputePerSLODistributions`, prints when `len(result) > 1`

**BC-4: Per-SLO metrics omitted for single-class workloads**
- GIVEN a workload with zero or one SLO class
- WHEN the simulation completes
- THEN stdout MUST NOT contain a `=== Per-SLO Metrics ===` section

**BC-5: Wall-clock fields removed from JSON output**
- GIVEN any simulation run
- WHEN `SaveResults` prints JSON to stdout
- THEN the JSON MUST NOT contain `simulation_duration_s`, `sim_start_timestamp`, or `sim_end_timestamp` fields
- MECHANISM: Fields removed from `MetricsOutput` struct

**BC-6: Wall-clock timing available on stderr**
- GIVEN any simulation run with `--log info` or lower
- WHEN the simulation completes
- THEN stderr MUST contain a logrus message with the wall-clock duration
- MECHANISM: `logrus.Infof` in `cmd/root.go` after simulation completes

**BC-7: Deterministic stdout**
- GIVEN two runs with the same seed and configuration
- WHEN both complete
- THEN their stdout MUST be byte-identical
- MECHANISM: All wall-clock fields removed from stdout; only deterministic simulation-derived values remain

**BC-8: Conservation fields in cluster JSON**
- GIVEN a cluster simulation
- WHEN results are printed
- THEN the cluster-level JSON MUST contain `still_queued`, `still_running`, and `injected_requests` fields
- MECHANISM: New fields in `MetricsOutput`, populated from simulator state at end of run

**BC-9: Conservation identity holds**
- GIVEN any cluster simulation
- WHEN results are printed
- THEN `injected_requests` MUST equal `completed_requests + still_queued + still_running` (all levels)
- AND for full pipeline conservation: `num_requests == injected_requests + Rejected Requests` (from anomaly counters)
- MECHANISM: `injected_requests` computed in `SaveResults` as sum of completed + queued + running. Rejected count is separate (printed in anomaly counters section, not included in `injected_requests` because rejected requests never reach any instance).

**BC-10: Per-SLO metrics sorted deterministically**
- GIVEN a workload with multiple SLO classes
- WHEN per-SLO metrics are printed
- THEN SLO classes MUST appear in sorted alphabetical order
- MECHANISM: Sort map keys before iterating (antipattern rule 2)

**Negative Contracts:**

**BC-11: No library code changes**
- The `sim/` package MUST NOT call logrus.Fatalf or os.Exit (rule 6). Wall-clock logging happens only in `cmd/root.go`.

**BC-12: SaveResults signature change is backward-compatible at call sites**
- All existing call sites of `SaveResults` MUST be updated to match the new signature (startTime parameter removed). No call site may silently break.

### C) Component Interaction

```
cmd/root.go ──────────────────────────────────
  │
  │ calls SaveResults() (per-instance + cluster)
  │ calls CollectRawMetrics() → RawMetrics
  │ calls ComputePerSLODistributions() → map
  │ prints: KV Cache Metrics section
  │ prints: Per-SLO Metrics section
  │ logrus.Infof: wall-clock timing
  │
  ├──→ sim/metrics.go
  │     SaveResults() — signature simplified
  │     (startTime removed)
  │
  ├──→ sim/metrics_utils.go
  │     MetricsOutput — 3 fields removed,
  │     3 fields added
  │
  └──→ sim/cluster/metrics.go
        ComputePerSLODistributions() — already
        exists, now called from cmd/root.go
```

**State changes:** None. This PR only changes output formatting and adds read-only observation fields.

**Extension friction:** Adding one more metric to KV cache section: 1 file (`cmd/root.go` print statement). Low friction.

### D) Deviation Log

| Source Says | Micro Plan Does | Reason |
|-------------|-----------------|--------|
| #271: "Add to cluster JSON struct" as alternative | Separate sections on stdout | SIMPLIFICATION: Avoids JSON schema breakage for KV/SLO metrics; conservation fields DO go in JSON |
| #273: "Low priority" | Included in this PR | ADDITION: Small incremental cost since we're already modifying MetricsOutput |
| #273: `injected == completed + queued + running + rejected` | Per-instance: `injected = completed + queued + running`; Cluster: `injected = completed + queued + running + rejected` | CORRECTION: Rejected requests happen at cluster level only; per-instance conservation doesn't include rejected |

### E) Review Guide

**The tricky part:** BC-12 — `SaveResults` signature change. There are 9 call sites (4 in metrics_test.go, 2 in simulator_test.go, 2 in cmd/root.go, 1 in cmd/root_test.go). Missing one creates a build error (not silent), so the risk is compilation failure, not silent bug. Also: `TestSimulator_Determinism_ByteIdenticalJSON` body references the removed fields — not just the signature.

**What to scrutinize:** BC-7 (deterministic stdout) — verify no other wall-clock or non-deterministic data leaks into stdout. BC-10 (sorted SLO keys) — verify map iteration is sorted.

**What's safe to skim:** BC-1/BC-2/BC-3/BC-4 are mechanical print additions following the existing anomaly counter pattern.

**Known debt:** The `Requests` field on `MetricsOutput` (line 74 of metrics_utils.go) includes per-request `SLOClass` but only when writing to file. Per-SLO CLI output uses `ComputePerSLODistributions` which reads from `Metrics.Requests` map directly, not from `MetricsOutput`.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to modify:**
- `sim/metrics_utils.go` — Remove 3 wall-clock fields, add 3 conservation fields to `MetricsOutput`
- `sim/metrics.go` — Simplify `SaveResults` signature (remove `startTime`), populate conservation fields, remove wall-clock field population
- `cmd/root.go` — Add KV cache section, per-SLO section, wall-clock logrus message, populate conservation fields
- `sim/metrics_test.go` — Update 4 test call sites for new SaveResults signature
- `sim/simulator_test.go` — Update 2 test call sites for new SaveResults signature
- `cmd/root_test.go` — Update 1 test call site
- `cmd/root.go` — Pass still_queued/still_running from cluster to SaveResults

**Key decisions:**
- KV cache and per-SLO metrics are separate stdout sections (not added to JSON struct) — avoids JSON schema breakage for machine parsers
- Conservation fields ARE added to JSON struct — they're simulation-derived, deterministic, and fundamental to correctness verification
- `startTime` parameter removed from `SaveResults` entirely — wall-clock logging moves to caller (`cmd/root.go`)

### G) Task Breakdown

---

### Task 1: Remove wall-clock fields from MetricsOutput and simplify SaveResults

**Contracts Implemented:** BC-5, BC-12

**Files:**
- Modify: `sim/metrics_utils.go:49-75` (MetricsOutput struct)
- Modify: `sim/metrics.go:63-77` (SaveResults function)
- Modify: `sim/metrics_test.go` (4 call sites)
- Modify: `sim/simulator_test.go` (2 call sites)
- Modify: `cmd/root_test.go` (1 call site)
- Modify: `cmd/root.go:437,441` (2 call sites)

**Step 1: Write failing test for BC-5**

Context: We test that the JSON output from SaveResults does NOT contain wall-clock fields.

In `sim/metrics_test.go`, add:
```go
func TestSaveResults_NoWallClockFields(t *testing.T) {
	// GIVEN a Metrics struct with completed requests
	m := NewMetrics()
	m.CompletedRequests = 1
	m.SimEndedTime = 1_000_000
	m.TotalInputTokens = 100
	m.TotalOutputTokens = 100
	m.RequestTTFTs["req1"] = 10.0
	m.RequestE2Es["req1"] = 100.0
	m.AllITLs = []int64{10}
	m.RequestSchedulingDelays["req1"] = 5

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "results.json")

	// WHEN SaveResults writes output
	m.SaveResults("test", 1_000_000, 1000, outPath)

	// THEN the JSON must not contain wall-clock fields
	data, err := os.ReadFile(outPath)
	require.NoError(t, err)
	jsonStr := string(data)
	assert.NotContains(t, jsonStr, "simulation_duration_s")
	assert.NotContains(t, jsonStr, "sim_start_timestamp")
	assert.NotContains(t, jsonStr, "sim_end_timestamp")
	// But it must still contain simulation-derived fields
	assert.Contains(t, jsonStr, "vllm_estimated_duration_s")
	assert.Contains(t, jsonStr, "completed_requests")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestSaveResults_NoWallClockFields -v`
Expected: FAIL — compilation error (SaveResults signature mismatch)

**Step 3: Implement the changes**

In `sim/metrics_utils.go`, remove these three fields from `MetricsOutput`:
```go
// REMOVE these lines:
// SimStartTimestamp     string           `json:"sim_start_timestamp"`
// SimEndTimestamp       string           `json:"sim_end_timestamp"`
// SimulationDurationSec float64          `json:"simulation_duration_s"`
```

In `sim/metrics.go`, change `SaveResults` signature and remove wall-clock population:
```go
func (m *Metrics) SaveResults(instanceID string, horizon int64, totalBlocks int64, outputFilePath string) {
	vllmRuntime := float64(m.SimEndedTime) / float64(1e6)

	output := MetricsOutput{
		InstanceID:           instanceID,
		CompletedRequests:    m.CompletedRequests,
		TotalInputTokens:     int(m.TotalInputTokens),
		TotalOutputTokens:    int(m.TotalOutputTokens),
		VllmDurationSec:      vllmRuntime,
		KVAllocationFailures: m.KVAllocationFailures,
	}
	// ... rest unchanged
```

Update all 9 call sites to remove the `startTime` argument:
- `cmd/root.go:437` — `inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, "")`
- `cmd/root.go:441` — `cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, resultsPath)`
- `sim/metrics_test.go` — 4 sites: remove `time.Now()` arg; also remove unused `"time"` import. Note: existing file does not import testify — add `"github.com/stretchr/testify/assert"` and `"github.com/stretchr/testify/require"` for new tests
- `sim/simulator_test.go` — 2 sites: remove `fixedTime` arg; also remove `fixedTime` variable declaration (line 691) and unused `"time"` import. **Additionally**, update `TestSimulator_Determinism_ByteIdenticalJSON` body: remove lines that zero out `out1.SimEndTimestamp`, `out2.SimEndTimestamp`, `out1.SimulationDurationSec`, `out2.SimulationDurationSec` (lines ~722-725) — these fields no longer exist in MetricsOutput. The test can now compare the marshaled JSON directly without zeroing any fields.
- `cmd/root_test.go` — 1 site: remove `time.Now()` arg; also remove unused `"time"` import

**Import cleanup:** After removing `startTime time.Time` from SaveResults, also remove the unused `"time"` import from `sim/metrics.go` (the only uses were `time.Now()` and `time.Since()` in the removed wall-clock lines).

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestSaveResults_NoWallClockFields -v`
Expected: PASS

Run: `go test ./... -count=1`
Expected: All tests pass (updated call sites compile correctly)

**Step 5: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/metrics_utils.go sim/metrics.go sim/metrics_test.go sim/simulator_test.go cmd/root.go cmd/root_test.go
git commit -m "refactor(metrics): remove wall-clock fields from JSON output (BC-5, BC-12)

- Remove simulation_duration_s, sim_start_timestamp, sim_end_timestamp from MetricsOutput
- Simplify SaveResults signature (remove startTime parameter)
- Update all 7 call sites
- vllm_estimated_duration_s (sim-clock) retained

Fixes #272

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add wall-clock timing to stderr via logrus

**Contracts Implemented:** BC-6, BC-7

**Files:**
- Modify: `cmd/root.go` (add logrus.Infof after simulation completes)

**Step 1: Write failing test for BC-6**

Context: This is a logrus message on stderr, tested via integration. We verify BC-7 (deterministic stdout) instead — run the binary twice and compare stdout.

In `cmd/root_test.go`, add:
```go
func TestSaveResults_DeterministicOutput(t *testing.T) {
	// GIVEN a Metrics struct with completed requests
	m := sim.NewMetrics()
	m.CompletedRequests = 5
	m.SimEndedTime = 5_000_000
	m.TotalInputTokens = 500
	m.TotalOutputTokens = 500
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("req%d", i)
		m.RequestTTFTs[id] = float64(i * 10)
		m.RequestE2Es[id] = float64(i * 100)
		m.RequestSchedulingDelays[id] = int64(i * 5)
	}
	m.AllITLs = []int64{10, 20, 30, 40, 50}

	// WHEN SaveResults is called twice
	var buf1, buf2 bytes.Buffer
	old := os.Stdout
	r1, w1, _ := os.Pipe()
	os.Stdout = w1
	m.SaveResults("test", 5_000_000, 1000, "")
	w1.Close()
	io.Copy(&buf1, r1)
	os.Stdout = old

	r2, w2, _ := os.Pipe()
	os.Stdout = w2
	m.SaveResults("test", 5_000_000, 1000, "")
	w2.Close()
	io.Copy(&buf2, r2)
	os.Stdout = old

	// THEN both outputs must be byte-identical
	assert.Equal(t, buf1.String(), buf2.String(), "SaveResults stdout must be deterministic")
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestSaveResults_DeterministicOutput -v`
Expected: PASS (should already pass since we removed wall-clock fields in Task 1)

**Step 3: Add wall-clock logrus message**

In `cmd/root.go`, after the simulation run completes (after `cs.Run()`) and before `SaveResults` calls, add:
```go
		wallClockDuration := time.Since(startTime)
		logrus.Infof("Simulation wall-clock time: %.3fs", wallClockDuration.Seconds())
```

**Step 4: Run all tests**

Run: `go test ./... -count=1`
Expected: All pass

**Step 5: Run lint**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/root.go cmd/root_test.go
git commit -m "feat(cli): add wall-clock timing to stderr via logrus (BC-6, BC-7)

- Log wall-clock duration at Info level on stderr
- Available with --log info, invisible with --log error
- Stdout is now fully deterministic (same seed = byte-identical output)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add KV cache metrics section to CLI output

**Contracts Implemented:** BC-1, BC-2

**Files:**
- Modify: `cmd/root.go` (add KV cache section after anomaly counters)
- Test: `cmd/root_test.go` or integration test

**Step 1: Write failing test for BC-1**

Context: We test that KV cache metrics appear in stdout when nonzero. This is best tested as an integration test against the CLI output. We'll use the RawMetrics struct directly.

In a new test file `cmd/kv_metrics_output_test.go`:
```go
package cmd

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPrintKVCacheMetrics_Nonzero_PrintsSection(t *testing.T) {
	// GIVEN nonzero KV cache metrics
	var buf bytes.Buffer

	// WHEN we print to the buffer
	printKVCacheMetrics(&buf, 0.05, 0.75, 0.02)

	// THEN the output must contain the KV cache section
	output := buf.String()
	assert.Contains(t, output, "=== KV Cache Metrics ===")
	assert.Contains(t, output, "Preemption Rate:")
	assert.Contains(t, output, "Cache Hit Rate:")
	assert.Contains(t, output, "KV Thrashing Rate:")
}

func TestPrintKVCacheMetrics_AllZero_NoOutput(t *testing.T) {
	// GIVEN all-zero KV cache metrics
	var buf bytes.Buffer

	// WHEN we print to the buffer
	printKVCacheMetrics(&buf, 0, 0, 0)

	// THEN no output
	assert.Empty(t, buf.String())
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestPrintKVCacheMetrics -v`
Expected: FAIL — `printKVCacheMetrics` undefined

**Step 3: Implement**

In `cmd/root.go`, add a helper function and call it after anomaly counters:

```go
// printKVCacheMetrics prints KV cache metrics to w when any value is nonzero.
func printKVCacheMetrics(w io.Writer, preemptionRate, cacheHitRate, kvThrashingRate float64) {
	if preemptionRate == 0 && cacheHitRate == 0 && kvThrashingRate == 0 {
		return
	}
	fmt.Fprintln(w, "=== KV Cache Metrics ===")
	fmt.Fprintf(w, "Preemption Rate: %.4f\n", preemptionRate)
	fmt.Fprintf(w, "Cache Hit Rate: %.4f\n", cacheHitRate)
	fmt.Fprintf(w, "KV Thrashing Rate: %.4f\n", kvThrashingRate)
}
```

Add `"io"` to the imports in `cmd/root.go` (needed for `io.Writer` parameter type).

In the `runCmd` block after anomaly counters:
```go
		// Print KV cache metrics if any nonzero
		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)
```

**Step 4: Run test**

Run: `go test ./cmd/... -run TestPrintKVCacheMetrics -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./cmd/...`

**Step 6: Commit**

```bash
git add cmd/root.go cmd/kv_metrics_output_test.go
git commit -m "feat(cli): surface KV cache metrics in CLI output (BC-1, BC-2)

- Print PreemptionRate, CacheHitRate, KVThrashingRate when nonzero
- Follows anomaly counters pattern (conditional section)

Fixes #271 (partial)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add per-SLO metrics section to CLI output

**Contracts Implemented:** BC-3, BC-4, BC-10

**Files:**
- Modify: `cmd/root.go` (call ComputePerSLODistributions, print results)
- Test: `cmd/kv_metrics_output_test.go` (add per-SLO tests)

**Step 1: Write failing test for BC-3 and BC-10**

```go
func TestPrintPerSLOMetrics_MultipleClasses_PrintsSorted(t *testing.T) {
	// GIVEN per-SLO distributions with multiple classes
	var buf bytes.Buffer
	sloMetrics := map[string]*cluster.SLOMetrics{
		"batch": {
			TTFT: cluster.Distribution{Mean: 100, P99: 200, Count: 10},
			E2E:  cluster.Distribution{Mean: 500, P99: 800, Count: 10},
		},
		"realtime": {
			TTFT: cluster.Distribution{Mean: 50, P99: 80, Count: 5},
			E2E:  cluster.Distribution{Mean: 200, P99: 300, Count: 5},
		},
	}

	// WHEN we print per-SLO metrics
	printPerSLOMetrics(&buf, sloMetrics)

	// THEN output must contain the section and classes in sorted order
	output := buf.String()
	assert.Contains(t, output, "=== Per-SLO Metrics ===")
	// "batch" must appear before "realtime" (alphabetical)
	batchIdx := bytes.Index([]byte(output), []byte("batch"))
	realtimeIdx := bytes.Index([]byte(output), []byte("realtime"))
	assert.True(t, batchIdx < realtimeIdx, "SLO classes must be sorted alphabetically")
}

func TestPrintPerSLOMetrics_SingleClass_NoOutput(t *testing.T) {
	// GIVEN per-SLO distributions with only one class
	var buf bytes.Buffer
	sloMetrics := map[string]*cluster.SLOMetrics{
		"default": {
			TTFT: cluster.Distribution{Mean: 100, P99: 200, Count: 10},
			E2E:  cluster.Distribution{Mean: 500, P99: 800, Count: 10},
		},
	}

	// WHEN we print per-SLO metrics
	printPerSLOMetrics(&buf, sloMetrics)

	// THEN no output (single class = no differentiation)
	assert.Empty(t, buf.String())
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./cmd/... -run TestPrintPerSLOMetrics -v`
Expected: FAIL — `printPerSLOMetrics` undefined

**Step 3: Implement**

In `cmd/root.go`:
```go
// printPerSLOMetrics prints per-SLO-class latency distributions when multiple classes exist.
func printPerSLOMetrics(w io.Writer, sloMetrics map[string]*cluster.SLOMetrics) {
	if len(sloMetrics) <= 1 {
		return
	}
	fmt.Fprintln(w, "=== Per-SLO Metrics ===")
	// Sort keys for deterministic output (antipattern rule 2)
	keys := make([]string, 0, len(sloMetrics))
	for k := range sloMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, cls := range keys {
		m := sloMetrics[cls]
		fmt.Fprintf(w, "  %s:\n", cls)
		fmt.Fprintf(w, "    TTFT: mean=%.2f p99=%.2f (n=%d)\n", m.TTFT.Mean, m.TTFT.P99, m.TTFT.Count)
		fmt.Fprintf(w, "    E2E:  mean=%.2f p99=%.2f (n=%d)\n", m.E2E.Mean, m.E2E.P99, m.E2E.Count)
	}
}
```

In the `runCmd` block:
```go
		// Print per-SLO metrics if multiple SLO classes present
		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions)
```

**Step 4: Run test**

Run: `go test ./cmd/... -run TestPrintPerSLOMetrics -v`
Expected: PASS

**Step 5: Run lint**

Run: `golangci-lint run ./cmd/...`

**Step 6: Commit**

```bash
git add cmd/root.go cmd/kv_metrics_output_test.go
git commit -m "feat(cli): surface per-SLO-class metrics in CLI output (BC-3, BC-4, BC-10)

- Print per-class TTFT and E2E distributions when >1 SLO class
- Sort SLO class keys for deterministic output
- Calls existing ComputePerSLODistributions()

Fixes #271 (complete)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Add conservation fields to MetricsOutput and populate them

**Contracts Implemented:** BC-8, BC-9

**Files:**
- Modify: `sim/metrics_utils.go` (add 3 fields to MetricsOutput)
- Modify: `sim/metrics.go` (add StillQueued, StillRunning to Metrics; populate in SaveResults)
- Modify: `sim/simulator.go` (record end-of-run queue/batch state into Metrics)
- Modify: `sim/cluster/cluster.go` (aggregate StillQueued/StillRunning in aggregateMetrics())
- Modify: `cmd/root.go` (add rejected to cluster-level injected_requests after SaveResults)

**Step 1: Write failing test for BC-8 and BC-9**

In `sim/metrics_test.go`:
```go
func TestSaveResults_ConservationFields(t *testing.T) {
	// GIVEN a Metrics struct with completed and in-flight requests
	m := NewMetrics()
	m.CompletedRequests = 8
	m.SimEndedTime = 5_000_000
	m.TotalInputTokens = 500
	m.TotalOutputTokens = 500
	m.StillQueued = 1
	m.StillRunning = 1
	for i := 0; i < 8; i++ {
		id := fmt.Sprintf("req%d", i)
		m.RequestTTFTs[id] = float64(i * 10)
		m.RequestE2Es[id] = float64(i * 100)
		m.RequestSchedulingDelays[id] = int64(i * 5)
	}
	m.AllITLs = []int64{10, 20, 30, 40, 50, 60, 70, 80}

	tmpDir := t.TempDir()
	outPath := filepath.Join(tmpDir, "results.json")

	// WHEN SaveResults writes output
	m.SaveResults("test", 5_000_000, 1000, outPath)

	// THEN JSON must contain conservation fields
	data, err := os.ReadFile(outPath)
	require.NoError(t, err)

	var output MetricsOutput
	require.NoError(t, json.Unmarshal(data, &output))

	assert.Equal(t, 1, output.StillQueued)
	assert.Equal(t, 1, output.StillRunning)
	// Conservation identity: injected = completed + queued + running
	assert.Equal(t, output.InjectedRequests, output.CompletedRequests+output.StillQueued+output.StillRunning)
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestSaveResults_ConservationFields -v`
Expected: FAIL — `StillQueued` field undefined

**Step 3: Implement**

In `sim/metrics_utils.go`, add to `MetricsOutput`:
```go
	StillQueued       int `json:"still_queued"`
	StillRunning      int `json:"still_running"`
	InjectedRequests  int `json:"injected_requests"`
```

In `sim/metrics.go`, add to `Metrics`:
```go
	StillQueued  int // Requests still in wait queue at sim end
	StillRunning int // Requests still in running batch at sim end
```

In `sim/metrics.go` `SaveResults`, populate the fields:
```go
	output.StillQueued = m.StillQueued
	output.StillRunning = m.StillRunning
	output.InjectedRequests = m.CompletedRequests + m.StillQueued + m.StillRunning
```

In `sim/simulator.go`, in `Run()` just before the `sim.Finalize()` call (between the event loop and Finalize), record:
```go
	sim.Metrics.StillQueued = sim.WaitQ.Len()
	if sim.RunningBatch != nil {
		sim.Metrics.StillRunning = len(sim.RunningBatch.Requests)
	}
```

In `sim/cluster/cluster.go`, in `aggregateMetrics()`, add these lines alongside the other field merges:
```go
	merged.StillQueued += m.StillQueued
	merged.StillRunning += m.StillRunning
```

**Step 4: Run test**

Run: `go test ./sim/... -run TestSaveResults_ConservationFields -v`
Expected: PASS

Run: `go test ./... -count=1`
Expected: All pass

**Step 5: Run lint**

Run: `golangci-lint run ./...`

**Step 6: Commit**

```bash
git add sim/metrics_utils.go sim/metrics.go sim/simulator.go
git commit -m "feat(metrics): add conservation fields to JSON output (BC-8, BC-9)

- Add still_queued, still_running, injected_requests to MetricsOutput
- Record end-of-run queue/batch state in Simulator.Run()
- Conservation identity: injected = completed + queued + running

Fixes #273

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Update CLAUDE.md and verify full test suite

**Contracts Implemented:** Documentation update

**Files:**
- Modify: `CLAUDE.md` (document new output sections)

**Step 1: Update CLAUDE.md**

In the `cmd/root.go` description line in CLAUDE.md, add the new CLI output sections:
- Add mention of `=== KV Cache Metrics ===` section
- Add mention of `=== Per-SLO Metrics ===` section
- Note that stdout is now deterministic (wall-clock fields moved to stderr)
- Note conservation fields in JSON

**Step 2: Run full verification**

Run: `go build ./... && go test ./... -count=1 && golangci-lint run ./...`
Expected: All pass, no lint issues

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for new CLI output sections

- Document KV Cache Metrics and Per-SLO Metrics sections
- Note deterministic stdout (wall-clock fields on stderr)
- Note conservation fields in JSON output

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 3 | Unit | TestPrintKVCacheMetrics_Nonzero_PrintsSection |
| BC-2 | Task 3 | Unit | TestPrintKVCacheMetrics_AllZero_NoOutput |
| BC-3 | Task 4 | Unit | TestPrintPerSLOMetrics_MultipleClasses_PrintsSorted |
| BC-4 | Task 4 | Unit | TestPrintPerSLOMetrics_SingleClass_NoOutput |
| BC-5 | Task 1 | Unit | TestSaveResults_NoWallClockFields |
| BC-6 | Task 2 | — | Logrus stderr (verified manually with --log info) |
| BC-7 | Task 2 | Unit | TestSaveResults_DeterministicOutput |
| BC-8 | Task 5 | Unit | TestSaveResults_ConservationFields |
| BC-9 | Task 5 | Unit | TestSaveResults_ConservationFields (identity assertion) |
| BC-10 | Task 4 | Unit | TestPrintPerSLOMetrics_MultipleClasses_PrintsSorted |
| BC-11 | — | Build | No logrus.Fatalf in sim/ (verified by lint/grep) |
| BC-12 | Task 1 | Build | All call sites compile with new signature |

**Golden dataset:** This PR does not change simulation behavior — only output formatting. Golden dataset values are unchanged. The `MetricsOutput` JSON struct changes (fields removed/added) but the golden dataset test uses `MetricsOutput` for file output, which still includes request-level data. Verify golden tests still pass after Task 1.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `SaveResults` call site missed | Low | High (build error) | Grep for all call sites; compilation catches misses |
| Per-SLO map iteration non-deterministic | Medium | Medium | Sort keys before printing (BC-10) |
| Conservation fields wrong for single-instance mode | Low | Medium | Test with cluster mode (Task 5); single-instance has no rejected count |
| Wall-clock logrus message at wrong level | Low | Low | Use Infof (visible at info, hidden at warn/error) |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — helper functions are minimal (printKVCacheMetrics, printPerSLOMetrics)
- [x] No feature creep — strictly addresses #271, #272, #273
- [x] No unexercised flags or interfaces
- [x] No partial implementations — each task produces working code
- [x] No breaking changes without contract updates — BC-12 covers SaveResults signature change
- [x] No hidden global state impact
- [x] All new code will pass golangci-lint
- [x] CLAUDE.md updated (Task 6)
- [x] No stale references left in CLAUDE.md
- [x] Deviation log reviewed — no unresolved deviations
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (Task 1 before Task 2)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration not needed (output format changes don't affect golden values)
- [x] Construction site audit: MetricsOutput has 1 construction site (metrics.go:67). SaveResults has 9 call sites (all listed in Task 1). aggregateMetrics() updated for StillQueued/StillRunning.
- [x] No new CLI flags — no numeric validation needed
- [x] No silent `continue` in new code
- [x] Map iteration sorted in printPerSLOMetrics (BC-10)
- [x] No logrus.Fatalf in sim/ — wall-clock logging only in cmd/
- [x] No resource allocation loops
- [x] No exported mutable maps
- [x] No YAML config changes
- [x] No division operations in new code

---

## Appendix: File-Level Details

### File: `sim/metrics_utils.go`

**Changes:** Remove 3 fields, add 3 fields to MetricsOutput.

Remove:
```go
SimStartTimestamp     string           `json:"sim_start_timestamp"`
SimEndTimestamp       string           `json:"sim_end_timestamp"`
SimulationDurationSec float64          `json:"simulation_duration_s"`
```

Add (after CompletedRequests):
```go
StillQueued       int `json:"still_queued"`
StillRunning      int `json:"still_running"`
InjectedRequests  int `json:"injected_requests"`
```

### File: `sim/metrics.go`

**Changes:**
1. Add `StillQueued`, `StillRunning` fields to `Metrics` struct
2. Change `SaveResults` signature: remove `startTime time.Time` parameter
3. Remove wall-clock field population from `SaveResults`
4. Add conservation field population in `SaveResults`

### File: `sim/simulator.go`

**Changes:** At end of `Run()`, record queue/batch state:
```go
sim.Metrics.StillQueued = sim.WaitQ.Len()
if sim.RunningBatch != nil {
    sim.Metrics.StillRunning = len(sim.RunningBatch.Requests)
}
```

### File: `cmd/root.go`

**Changes:**
1. Add `wallClockDuration` logrus.Infof after simulation
2. Remove `startTime` from `SaveResults` calls
3. Add `printKVCacheMetrics` helper + call after anomaly counters
4. Add `printPerSLOMetrics` helper + call after KV cache metrics
5. Add `import "sort"` if not present

### File: `cmd/kv_metrics_output_test.go` (new)

**Purpose:** Tests for printKVCacheMetrics and printPerSLOMetrics helpers.
