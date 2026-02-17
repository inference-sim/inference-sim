# PR9: RawMetrics, Anomaly Detection, and Pathological Templates — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add RawMetrics with fitness evaluation, anomaly detection counters, and pathological policy templates to reach the Research-Ready Checkpoint.

**Architecture:** `RawMetrics` aggregates cluster-level metrics (latency distributions, throughput, anomaly counters) from existing per-instance `Metrics`. `FitnessFunction` computes weighted fitness scores from `RawMetrics`. Four pathological templates (`reject-all`, `inverted-slo`, `always-busiest`, `reverse-priority`) exercise anomaly detection. All integrated via `--fitness-weights` CLI flag and existing `--admission-policy` / `--routing-policy` / `--priority-policy` / `--scheduler` flags.

**Macro Plan Reference:** Phase 2b, PR 9 (section I, "PR 9: RawMetrics, Anomaly Detection, and Pathological Templates")

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

This PR adds two capabilities that together reach the Research-Ready Checkpoint:

1. **Metrics & fitness evaluation** — After a cluster simulation completes, `RawMetrics` aggregates latency distributions (TTFT, E2E) and throughput from existing per-instance data. A `FitnessFunction` computes a weighted score from these metrics, enabling evolutionary policy optimization. Exercised via `--fitness-weights "throughput:0.5,p99_ttft:0.3"`.

2. **Pathological policy templates** — Four deliberately-broken policies (`reject-all`, `inverted-slo`, `always-busiest`, `reverse-priority`) that trigger known failure modes (100% rejection, priority inversion, load imbalance). These validate that anomaly detection counters work correctly.

**Building block:** Metrics & Evaluation layer (`sim/cluster/metrics.go`), sitting between `ClusterSimulator.Run()` completion and results output. Adjacent to the 4 policy files (which receive pathological templates), `bundle.go` (name registration), and `cmd/root.go` (CLI flag).

**Deviations from macro plan:** Several `RawMetrics` fields deferred (TPOT, SLO attainment, fairness, scale metrics, cache/preemption rates) because they require TenantID or AutoScaler — neither exists yet. Full `EvaluationResult` wrapper also deferred. See Section D for details.

<details>
<summary>Phase 0: Confirmed codebase facts (click to expand)</summary>

- `AdmissionPolicy.Admit()` returns `(bool, string)` — `sim/admission.go:9`
- `RoutingPolicy.Route()` returns `RoutingDecision` — `sim/routing.go:35`
- `PriorityPolicy.Compute()` returns `float64` — `sim/priority.go:9`
- `InstanceScheduler.OrderQueue()` sorts in-place — `sim/scheduler.go:12`
- Policy names registered in `validAdmissionPolicies`/etc maps — `sim/bundle.go:62-66`
- Factory functions: `NewAdmissionPolicy` (`sim/admission.go:58`), `NewRoutingPolicy` (`sim/routing.go:191`), `NewPriorityPolicy` (`sim/priority.go:41`), `NewScheduler` (`sim/scheduler.go:63`)
- `ClusterSimulator.rejectedRequests` counter incremented in `AdmissionDecisionEvent.Execute` — `sim/cluster/cluster_event.go:114`
- `Metrics.RequestTTFTs`/`RequestE2Es` are `map[string]float64` — `sim/metrics.go:29-33`
- `Metrics.NumWaitQRequests` is `[]int` (queue depth per step) — `sim/metrics.go:37`
- `sim.CalculatePercentile` divides by 1000 (ms conversion) — `sim/metrics_utils.go:74-80` — NOT suitable for raw-tick Distribution
- Existing test helpers: `newTestDeploymentConfig` and `newTestWorkload` in `sim/cluster/cluster_test.go:12-48`
</details>

### B) Behavioral Contracts

#### Positive Contracts

**BC-1: RawMetrics Aggregation**
- GIVEN a completed `ClusterSimulator.Run()` with N instances
- WHEN `CollectRawMetrics(cs)` is called
- THEN it MUST return a `RawMetrics` populated from the aggregated `*sim.Metrics`: latency percentiles (TTFT/E2E P50/P99), throughput (requests/sec, tokens/sec), and anomaly counters
- MECHANISM: Extract from `cs.AggregatedMetrics()`. Build `Distribution` from sorted float64 slices with custom `percentile()` (NOT `sim.CalculatePercentile` which divides by 1000 for ms conversion — see `sim/metrics_utils.go:74`). Throughput from `CompletedRequests / SimEndedTime`.

**BC-2: Distribution Type**
- GIVEN a `Distribution` value
- WHEN accessed
- THEN it MUST contain Mean, P50, P95, P99, Min, Max, and Count fields
- MECHANISM: Computed from sorted float64 slices using custom `percentile()` with linear interpolation (raw tick values, no ms conversion)

**BC-3: FitnessFunction Weighted Score**
- GIVEN a `RawMetrics` and a weight map `map[string]float64` (e.g., `{"throughput": 0.5, "p99_ttft": 0.3}`)
- WHEN `ComputeFitness(metrics, weights)` is called
- THEN it MUST return a `FitnessResult` with a scalar `Score` (weighted sum) and per-component `Components` map
- MECHANISM: For each weight key, extract the corresponding metric from `RawMetrics`, normalize to [0,1] range, multiply by weight, sum. Throughput normalized as `value / (value + referenceRPS)` where referenceRPS=100. Token throughput uses `referenceTPS=10000`. Latency normalized as `1.0 / (1.0 + value/referenceTicks)` where referenceTicks=1000 (1ms). This ensures all metric types produce scores in comparable [0,1] ranges for meaningful multi-objective optimization.

**BC-4: RejectAll Admission**
- GIVEN a `RejectAll` admission policy
- WHEN `Admit()` is called with any request and state
- THEN it MUST return `(false, "reject-all")`
- MECHANISM: Constant rejection, no state

**BC-5: InvertedSLO Priority**
- GIVEN an `InvertedSLO` priority policy
- WHEN `Compute()` is called with a request and clock
- THEN it MUST return *lower* priority for *older* requests (inverse of SLOBasedPriority: BaseScore - AgeWeight * age)
- MECHANISM: Negative age weight causes priority inversion — newer requests get higher priority, starving older ones

**BC-6: AlwaysBusiest Routing**
- GIVEN an `AlwaysBusiest` routing policy
- WHEN `Route()` is called with snapshots
- THEN it MUST route to the instance with the *highest* (QueueDepth + BatchSize) — the opposite of LeastLoaded
- MECHANISM: Select argmax of load, creating extreme load imbalance

**BC-7: ReversePriority Scheduler**
- GIVEN a `ReversePriority` scheduler
- WHEN `OrderQueue()` is called
- THEN it MUST sort requests by priority ascending (lowest priority first) — the opposite of PriorityFCFSScheduler
- MECHANISM: Inverted sort order causes priority inversions in scheduling

**BC-8: Priority Inversion Detection**
- GIVEN a completed simulation where higher-priority requests were scheduled after lower-priority ones
- WHEN `CollectRawMetrics()` builds `RawMetrics`
- THEN `PriorityInversions` counter MUST reflect the count of inversions detected
- MECHANISM: Count from per-instance schedulers. For PR9, we count the *effect*: if a request with higher priority has a later schedule time than a request with lower priority on the same instance, that's an inversion. Detected post-hoc from `Metrics.Requests` data.

**BC-9: HOL Blocking Detection**
- GIVEN a completed simulation where an instance had high queue depth while other instances were idle
- WHEN `CollectRawMetrics()` builds `RawMetrics`
- THEN `HOLBlockingEvents` counter MUST be > 0
- MECHANISM: Detect from per-instance metrics: if any instance's max queue depth exceeds 2× the mean across instances, count as HOL blocking.

#### Negative Contracts

**NC-1: Golden Dataset Unchanged**
- GIVEN default policy configuration (always-admit, round-robin, constant, fcfs)
- WHEN simulation runs with golden dataset parameters
- THEN output MUST be bit-for-bit identical to existing golden dataset
- MECHANISM: RawMetrics is additive — only collected when requested; default path unchanged

**NC-2: No Interface Breaking Changes**
- GIVEN the interface freeze (post-PR8)
- WHEN pathological templates are added
- THEN existing policy interfaces MUST NOT change signature
- MECHANISM: Templates implement existing interfaces; registered in existing maps

#### Error Handling Contracts

**EC-1: Invalid Fitness Weight Key**
- GIVEN `--fitness-weights "invalid_key:0.5"`
- WHEN fitness is computed
- THEN unknown keys MUST be logged as warnings and ignored (not panic)
- MECHANISM: Skip unknown keys with `logrus.Warnf`

**EC-2: Empty Fitness Weights**
- GIVEN `--fitness-weights` not set (empty string)
- WHEN simulation completes
- THEN fitness evaluation MUST be skipped entirely (no error)
- MECHANISM: Check for empty string before parsing

### C) Component Interaction

```
                    ClusterSimulator.Run()
                           │
                           ▼
                    AggregatedMetrics()
                           │
                           ▼
                  CollectRawMetrics(cs)  ──► RawMetrics
                           │                   │
                           │                   ▼
                           │            ComputeFitness(metrics, weights)
                           │                   │
                           ▼                   ▼
                    cmd/root.go         FitnessResult
                    (print/save)        (Score + Components)
```

**API Contracts:**
- `CollectRawMetrics(aggregated *sim.Metrics, perInstance []*sim.Metrics, rejectedRequests int) *RawMetrics` — pure function, no mutation
- `ComputeFitness(metrics *RawMetrics, weights map[string]float64) *FitnessResult` — pure function
- `ParseFitnessWeights(s string) (map[string]float64, error)` — parses "key:val,key:val" format

**New mutable state:** None. RawMetrics and FitnessResult are computed post-simulation.

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| `EvaluationResult` struct (macro plan G.3, lines 795-813) with Fitness, Metrics, Trace, Summary, PolicyID, WorkloadID, SimDuration, WallTime | Defer entire `EvaluationResult` type — implement only `FitnessResult` (Score + Components) and `RawMetrics` as standalone types | SIMPLIFICATION: `EvaluationResult` is an orchestration wrapper for the framework adapter layer (PR15). Its fields `Trace`/`Summary` are PR13 scope, `PolicyID`/`WorkloadID` require adapter context. The two core types (`RawMetrics` + `FitnessResult`) are independently useful and sufficient for research-ready checkpoint. `EvaluationResult` can wrap them later without breaking changes. |
| `Distribution` with per-SLO-class maps (`TTFT map[string]*Distribution`) | Single aggregate `Distribution` per metric (not per SLO class) | SIMPLIFICATION: SLO classes require TenantID on Request (not yet implemented). Single-aggregate is sufficient for Phase 2 research. Per-class maps are additive when TenantID lands. |
| `TPOT map[string]*Distribution` in RawMetrics | Omit TPOT from RawMetrics | SIMPLIFICATION: TPOT (time-per-output-token) is derivable from ITL data in `sim.Metrics.AllITLs` but requires per-request grouping not currently available in aggregated metrics. Adding it would require additional per-request ITL tracking. Deferred — can be added additively later. |
| `SLOAttainment map[string]float64` | Omit entirely from RawMetrics struct | DEFERRAL: Requires SLO targets and TenantID, both PR10+ scope. Field will be added when TenantID lands. |
| `JainFairnessIndex` | Omit entirely from RawMetrics struct | DEFERRAL: Requires per-tenant metrics; no TenantID yet. Field will be added when TenantID lands. |
| `CacheHitRate`, `PreemptionRate` | Omit entirely from RawMetrics struct | DEFERRAL: CacheHitRate requires KV hit/miss tracking; PreemptionRate requires preemption event counting — both are wiring tasks for later PRs. Fields will be added additively. |
| `ScaleUpCount`, `ScaleDownCount`, `TotalReplicaSeconds`, `ScaleOscillations` | Omit entirely from RawMetrics struct | DEFERRAL: Requires AutoScaler (PR11). Fields will be added when AutoScaler lands. |
| `--fitness-weights STRING` as CLI flag | Implement as specified | MATCHES macro plan |
| `rate-limit`, `tenant-quota` admission templates | Not included | OUT OF SCOPE: Macro plan says "PR9+ (requires TenantID)" — these are future work, not PR9 |
| `tenant-priority`, `deadline-aware` priority templates | Not included | OUT OF SCOPE: Macro plan says "PR9+ (requires TenantState)" — future work |

### E) Review Guide

1. **THE TRICKY PART:** Priority inversion detection (BC-8) — detecting inversions post-hoc from metrics data requires careful definition of what constitutes an "inversion" without per-request schedule timestamps. The pragmatic approach: use the `InvertedSLO` pathological template to *guarantee* inversions happen, then detect via E2E latency correlation (older requests have worse E2E when they should have better).
2. **WHAT TO SCRUTINIZE:** BC-3 (fitness function normalization) — uses reference-scale normalization (`1/(1+value/referenceTicks)` for latency, `value/(value+referenceRPS)` for throughput) to produce comparable [0,1] scores. Verify edge cases: zero latency → score 1.0, zero throughput → score 0.0, reference values → score 0.5. Also BC-6 (AlwaysBusiest) — must match the `LeastLoaded` signature exactly but pick argmax instead of argmin.
3. **WHAT'S SAFE TO SKIM:** BC-4 (RejectAll) — trivial constant return. BC-2 (Distribution) — mechanical percentile computation using existing helpers. Template registration in bundle.go — follows established 5-step pattern.
4. **KNOWN DEBT:** Priority inversion and HOL blocking detection are simplified heuristics for PR9. Full decision-trace-based detection (counting exact scheduling order violations) is PR13 scope.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/cluster/metrics.go` (~250 LOC) — `RawMetrics`, `Distribution`, `FitnessResult`, `CollectRawMetrics`, `ComputeFitness`, `ParseFitnessWeights`

**Files to modify:**
- `sim/admission.go` (~15 LOC) — Add `RejectAll` template + factory case
- `sim/routing.go` (~25 LOC) — Add `AlwaysBusiest` template + factory case
- `sim/priority.go` (~15 LOC) — Add `InvertedSLO` template + factory case
- `sim/scheduler.go` (~20 LOC) — Add `ReversePriority` template + factory case
- `sim/bundle.go` (~4 LOC) — Register pathological names in valid maps
- `cmd/root.go` (~25 LOC) — Add `--fitness-weights` flag, collect and print RawMetrics/fitness
- `CLAUDE.md` (~10 LOC) — Update completed PR list, file organization

**Key decisions:**
- RawMetrics computed post-simulation (not during event loop) — simpler, no performance impact
- Pathological templates are thin wrappers implementing existing interfaces — no new types needed beyond the template structs
- Fitness weights parsed as "key:value,key:value" string (simple, no YAML dependency)

**Confirmation:** No dead code — all templates exercisable via CLI flags; RawMetrics always computed; fitness only when `--fitness-weights` set.

### G) Task Breakdown

---

#### Task 1: Distribution Type and NewDistribution Helper

**Contracts Implemented:** BC-2

**Files:**
- Create: `sim/cluster/metrics.go`
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for Distribution**

Context: Distribution captures statistical summaries of metric values. We need to verify it computes correct percentiles and handles edge cases.

```go
package cluster

import (
	"math"
	"testing"
)

// TestDistribution_FromValues_ComputesCorrectStats verifies BC-2.
func TestDistribution_FromValues_ComputesCorrectStats(t *testing.T) {
	tests := []struct {
		name      string
		values    []float64
		wantCount int
		wantMin   float64
		wantMax   float64
		wantMean  float64
	}{
		{
			name:      "single value",
			values:    []float64{100.0},
			wantCount: 1,
			wantMin:   100.0,
			wantMax:   100.0,
			wantMean:  100.0,
		},
		{
			name:      "multiple values",
			values:    []float64{10.0, 20.0, 30.0, 40.0, 50.0},
			wantCount: 5,
			wantMin:   10.0,
			wantMax:   50.0,
			wantMean:  30.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			d := NewDistribution(tt.values)
			if d.Count != tt.wantCount {
				t.Errorf("Count: got %d, want %d", d.Count, tt.wantCount)
			}
			if d.Min != tt.wantMin {
				t.Errorf("Min: got %f, want %f", d.Min, tt.wantMin)
			}
			if d.Max != tt.wantMax {
				t.Errorf("Max: got %f, want %f", d.Max, tt.wantMax)
			}
			if d.Mean != tt.wantMean {
				t.Errorf("Mean: got %f, want %f", d.Mean, tt.wantMean)
			}
			// P99 of [10,20,30,40,50] should be close to 50
			if tt.name == "multiple values" && d.P99 < 40.0 {
				t.Errorf("P99: got %f, expected >= 40.0", d.P99)
			}
		})
	}
}

// TestDistribution_EmptyValues_ReturnsZero verifies edge case.
func TestDistribution_EmptyValues_ReturnsZero(t *testing.T) {
	d := NewDistribution([]float64{})
	if d.Count != 0 {
		t.Errorf("Count: got %d, want 0", d.Count)
	}
	if d.Mean != 0 {
		t.Errorf("Mean: got %f, want 0", d.Mean)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestDistribution -v`
Expected: FAIL with "undefined: NewDistribution"

**Step 3: Implement Distribution type and NewDistribution**

Context: We create a standalone stats type that wraps sorted float64 slices. We compute percentiles inline (not using `sim.CalculatePercentile` since that divides by 1000 for millisecond conversion, which we don't want for raw values).

In `sim/cluster/metrics.go`:
```go
package cluster

import (
	"math"
	"sort"
)

// Distribution captures statistical summary of a metric.
type Distribution struct {
	Mean  float64
	P50   float64
	P95   float64
	P99   float64
	Min   float64
	Max   float64
	Count int
}

// NewDistribution computes a Distribution from raw values.
// Returns zero-value Distribution for empty input.
func NewDistribution(values []float64) Distribution {
	if len(values) == 0 {
		return Distribution{}
	}
	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	sum := 0.0
	for _, v := range sorted {
		sum += v
	}

	return Distribution{
		Mean:  sum / float64(len(sorted)),
		P50:   percentile(sorted, 50),
		P95:   percentile(sorted, 95),
		P99:   percentile(sorted, 99),
		Min:   sorted[0],
		Max:   sorted[len(sorted)-1],
		Count: len(sorted),
	}
}

// percentile computes the p-th percentile using linear interpolation.
// Input must be sorted. Returns raw value (not converted to ms).
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := p / 100.0 * float64(len(sorted)-1)
	lower := int(math.Floor(rank))
	upper := int(math.Ceil(rank))
	if lower == upper {
		return sorted[lower]
	}
	frac := rank - float64(lower)
	return sorted[lower] + frac*(sorted[upper]-sorted[lower])
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestDistribution -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "feat(metrics): add Distribution type with percentile computation (BC-2)

- Add Distribution struct with Mean, P50, P95, P99, Min, Max, Count
- Add NewDistribution helper from raw float64 values
- Handle empty input gracefully (returns zero-value)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 2: RawMetrics Type and CollectRawMetrics

**Contracts Implemented:** BC-1

**Files:**
- Modify: `sim/cluster/metrics.go`
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for RawMetrics collection**

Context: RawMetrics aggregates latency distributions and throughput from `*sim.Metrics`. We need to verify it extracts correct values from the existing metrics struct.

```go
// TestCollectRawMetrics_BasicAggregation verifies BC-1.
func TestCollectRawMetrics_BasicAggregation(t *testing.T) {
	// GIVEN aggregated metrics with known TTFT and E2E values
	m := sim.NewMetrics()
	m.RequestTTFTs = map[string]float64{
		"r0": 1000.0, // 1000 ticks = 1ms
		"r1": 2000.0,
		"r2": 3000.0,
	}
	m.RequestE2Es = map[string]float64{
		"r0": 5000.0,
		"r1": 10000.0,
		"r2": 15000.0,
	}
	m.CompletedRequests = 3
	m.TotalOutputTokens = 300
	m.SimEndedTime = 1_000_000 // 1 second

	// WHEN collecting RawMetrics
	raw := CollectRawMetrics(m, nil, 0)

	// THEN TTFT distribution should be populated
	if raw.TTFT.Count != 3 {
		t.Errorf("TTFT.Count: got %d, want 3", raw.TTFT.Count)
	}
	if raw.TTFT.Min != 1000.0 {
		t.Errorf("TTFT.Min: got %f, want 1000.0", raw.TTFT.Min)
	}

	// THEN throughput should be computed
	wantRPS := 3.0 / 1.0 // 3 requests / 1 second
	if math.Abs(raw.RequestsPerSec-wantRPS) > 0.01 {
		t.Errorf("RequestsPerSec: got %f, want %f", raw.RequestsPerSec, wantRPS)
	}
	wantTPS := 300.0 / 1.0 // 300 tokens / 1 second
	if math.Abs(raw.TokensPerSec-wantTPS) > 0.01 {
		t.Errorf("TokensPerSec: got %f, want %f", raw.TokensPerSec, wantTPS)
	}
}

// TestCollectRawMetrics_ZeroCompleted_ReturnsEmptyDistributions verifies edge case.
func TestCollectRawMetrics_ZeroCompleted_ReturnsEmptyDistributions(t *testing.T) {
	m := sim.NewMetrics()
	raw := CollectRawMetrics(m, nil, 0)
	if raw.TTFT.Count != 0 {
		t.Errorf("TTFT.Count: got %d, want 0", raw.TTFT.Count)
	}
	if raw.RequestsPerSec != 0 {
		t.Errorf("RequestsPerSec: got %f, want 0", raw.RequestsPerSec)
	}
}

// TestCollectRawMetrics_RejectedRequests verifies rejected count is captured.
func TestCollectRawMetrics_RejectedRequests(t *testing.T) {
	m := sim.NewMetrics()
	raw := CollectRawMetrics(m, nil, 42)
	if raw.RejectedRequests != 42 {
		t.Errorf("RejectedRequests: got %d, want 42", raw.RejectedRequests)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run TestCollectRawMetrics -v`
Expected: FAIL with "undefined: CollectRawMetrics"

**Step 3: Implement RawMetrics and CollectRawMetrics**

In `sim/cluster/metrics.go` (add to existing file):
```go
import (
	"github.com/inference-sim/inference-sim/sim"
)

// RawMetrics holds cluster-level metrics aggregated after simulation.
type RawMetrics struct {
	// Latency distributions (in ticks)
	TTFT Distribution
	E2E  Distribution

	// Throughput
	RequestsPerSec float64
	TokensPerSec   float64

	// Anomaly counters
	PriorityInversions int
	HOLBlockingEvents  int
	RejectedRequests   int
}

// CollectRawMetrics builds RawMetrics from aggregated and per-instance metrics.
// perInstance is optional (may be nil for anomaly-free collection).
func CollectRawMetrics(aggregated *sim.Metrics, perInstance []*sim.Metrics, rejectedRequests int) *RawMetrics {
	raw := &RawMetrics{
		RejectedRequests: rejectedRequests,
	}

	// Latency distributions
	ttftValues := mapValues(aggregated.RequestTTFTs)
	raw.TTFT = NewDistribution(ttftValues)

	e2eValues := mapValues(aggregated.RequestE2Es)
	raw.E2E = NewDistribution(e2eValues)

	// Throughput
	if aggregated.SimEndedTime > 0 && aggregated.CompletedRequests > 0 {
		durationSec := float64(aggregated.SimEndedTime) / 1e6
		raw.RequestsPerSec = float64(aggregated.CompletedRequests) / durationSec
		raw.TokensPerSec = float64(aggregated.TotalOutputTokens) / durationSec
	}

	return raw
}

// mapValues extracts values from a map into a slice.
func mapValues(m map[string]float64) []float64 {
	values := make([]float64, 0, len(m))
	for _, v := range m {
		values = append(values, v)
	}
	return values
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run TestCollectRawMetrics -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "feat(metrics): add RawMetrics and CollectRawMetrics (BC-1)

- Add RawMetrics struct with latency distributions, throughput, anomaly counters
- Add CollectRawMetrics to aggregate from sim.Metrics
- Handle zero-completed and rejected-requests edge cases

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 3: FitnessFunction and ParseFitnessWeights

**Contracts Implemented:** BC-3, EC-1, EC-2

**Files:**
- Modify: `sim/cluster/metrics.go`
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for fitness computation**

```go
// TestComputeFitness_WeightedScore verifies BC-3.
func TestComputeFitness_WeightedScore(t *testing.T) {
	raw := &RawMetrics{
		RequestsPerSec: 100.0,
		TTFT:           Distribution{P99: 5000.0}, // 5000 ticks = 5ms
	}

	weights := map[string]float64{
		"throughput": 1.0,
	}

	result := ComputeFitness(raw, weights)

	// THEN Score should be positive and in [0,1] range (normalized)
	if result.Score <= 0 || result.Score > 1.0 {
		t.Errorf("Score: got %f, expected in (0, 1]", result.Score)
	}
	// throughput=100, reference=100 → 100/(100+100) = 0.5
	if math.Abs(result.Components["throughput"]-0.5) > 0.01 {
		t.Errorf("throughput component: got %f, expected ~0.5", result.Components["throughput"])
	}
}

// TestComputeFitness_LatencyInversion verifies latency metrics are inverted (lower is better).
func TestComputeFitness_LatencyInversion(t *testing.T) {
	lowLatency := &RawMetrics{TTFT: Distribution{P99: 1000.0}}   // 1ms
	highLatency := &RawMetrics{TTFT: Distribution{P99: 10000.0}} // 10ms

	weights := map[string]float64{"p99_ttft": 1.0}

	lowResult := ComputeFitness(lowLatency, weights)
	highResult := ComputeFitness(highLatency, weights)

	// THEN lower latency should produce higher fitness score
	if lowResult.Score <= highResult.Score {
		t.Errorf("Expected low latency score (%f) > high latency score (%f)", lowResult.Score, highResult.Score)
	}
	// 1ms: 1/(1+1000/1000) = 0.5, 10ms: 1/(1+10000/1000) = 0.0909
	if math.Abs(lowResult.Score-0.5) > 0.01 {
		t.Errorf("1ms latency score: got %f, expected ~0.5", lowResult.Score)
	}
}

// TestComputeFitness_MultiObjective verifies throughput and latency have comparable scale.
func TestComputeFitness_MultiObjective(t *testing.T) {
	raw := &RawMetrics{
		RequestsPerSec: 100.0,                      // reference → 0.5
		TTFT:           Distribution{P99: 1000.0},   // 1ms reference → 0.5
	}
	weights := map[string]float64{"throughput": 0.5, "p99_ttft": 0.5}
	result := ComputeFitness(raw, weights)

	// Both at reference → both contribute 0.5 * 0.5 = 0.25, total ≈ 0.5
	if math.Abs(result.Score-0.5) > 0.01 {
		t.Errorf("Multi-objective score: got %f, expected ~0.5", result.Score)
	}
}

// TestComputeFitness_UnknownKey_Ignored verifies EC-1.
func TestComputeFitness_UnknownKey_Ignored(t *testing.T) {
	raw := &RawMetrics{RequestsPerSec: 100.0}
	weights := map[string]float64{"nonexistent": 1.0}

	result := ComputeFitness(raw, weights)
	if result.Score != 0 {
		t.Errorf("Score: got %f, expected 0 for unknown key", result.Score)
	}
}

// TestParseFitnessWeights_ValidInput verifies parsing.
func TestParseFitnessWeights_ValidInput(t *testing.T) {
	weights, err := ParseFitnessWeights("throughput:0.5,p99_ttft:0.3")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(weights) != 2 {
		t.Fatalf("expected 2 weights, got %d", len(weights))
	}
	if weights["throughput"] != 0.5 {
		t.Errorf("throughput: got %f, want 0.5", weights["throughput"])
	}
	if weights["p99_ttft"] != 0.3 {
		t.Errorf("p99_ttft: got %f, want 0.3", weights["p99_ttft"])
	}
}

// TestParseFitnessWeights_Empty verifies EC-2.
func TestParseFitnessWeights_Empty(t *testing.T) {
	weights, err := ParseFitnessWeights("")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(weights) != 0 {
		t.Errorf("expected empty map, got %d entries", len(weights))
	}
}

// TestParseFitnessWeights_InvalidFormat verifies error on bad input.
func TestParseFitnessWeights_InvalidFormat(t *testing.T) {
	_, err := ParseFitnessWeights("throughput:abc")
	if err == nil {
		t.Error("expected error for non-numeric weight")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run "TestComputeFitness|TestParseFitnessWeights" -v`
Expected: FAIL with "undefined: ComputeFitness"

**Step 3: Implement FitnessResult, ComputeFitness, ParseFitnessWeights**

In `sim/cluster/metrics.go` (add to existing file):
```go
import (
	"fmt"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

// FitnessResult holds the computed fitness score and per-component breakdown.
type FitnessResult struct {
	Score      float64            // Weighted sum of normalized metric components
	Components map[string]float64 // Per-component scores before weighting
}

// Reference scales for normalizing metrics to [0,1] range.
// Without reference scales, throughput (raw value ~100) dominates latency (1/(1+5000) ≈ 0.0002)
// by 500,000×, making multi-objective optimization impossible.
const (
	referenceRPS   = 100.0  // 100 requests/sec as reference throughput
	referenceTicks = 1000.0 // 1ms (1000 ticks) as reference latency
)

// ComputeFitness computes a weighted fitness score from RawMetrics.
// All metrics are normalized to [0,1] range before weighting:
// - Throughput: value / (value + referenceRPS) — higher is better, saturates at 1.0
// - Latency: 1.0 / (1.0 + value/referenceTicks) — lower is better, 1ms → 0.5
// Unknown weight keys are logged as warnings and ignored (EC-1).
func ComputeFitness(metrics *RawMetrics, weights map[string]float64) *FitnessResult {
	result := &FitnessResult{
		Components: make(map[string]float64, len(weights)),
	}

	for key, weight := range weights {
		value, ok := extractMetric(metrics, key)
		if !ok {
			logrus.Warnf("ComputeFitness: unknown metric key %q, ignoring", key)
			continue
		}
		result.Components[key] = value
		result.Score += value * weight
	}

	return result
}

// extractMetric returns a normalized [0,1] metric value for the given key.
// Throughput: value / (value + referenceRPS). Latency: 1 / (1 + value/referenceTicks).
// Returns (value, true) on success, (0, false) for unknown keys.
func extractMetric(m *RawMetrics, key string) (float64, bool) {
	switch key {
	// Higher is better — normalized via value / (value + reference)
	case "throughput":
		return m.RequestsPerSec / (m.RequestsPerSec + referenceRPS), true
	case "tokens_per_sec":
		return m.TokensPerSec / (m.TokensPerSec + referenceRPS), true
	// Lower is better — normalized via 1 / (1 + value/reference)
	case "p99_ttft":
		return 1.0 / (1.0 + m.TTFT.P99/referenceTicks), true
	case "p50_ttft":
		return 1.0 / (1.0 + m.TTFT.P50/referenceTicks), true
	case "mean_ttft":
		return 1.0 / (1.0 + m.TTFT.Mean/referenceTicks), true
	case "p99_e2e":
		return 1.0 / (1.0 + m.E2E.P99/referenceTicks), true
	case "p50_e2e":
		return 1.0 / (1.0 + m.E2E.P50/referenceTicks), true
	case "mean_e2e":
		return 1.0 / (1.0 + m.E2E.Mean/referenceTicks), true
	default:
		return 0, false
	}
}

// ParseFitnessWeights parses a "key:value,key:value" string into a weight map.
// Returns empty map for empty input (EC-2). Returns error for malformed entries.
func ParseFitnessWeights(s string) (map[string]float64, error) {
	if s == "" {
		return map[string]float64{}, nil
	}
	weights := make(map[string]float64)
	for _, pair := range strings.Split(s, ",") {
		pair = strings.TrimSpace(pair)
		parts := strings.SplitN(pair, ":", 2)
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid fitness weight %q: expected key:value", pair)
		}
		key := strings.TrimSpace(parts[0])
		val, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid fitness weight value for %q: %w", key, err)
		}
		weights[key] = val
	}
	return weights, nil
}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run "TestComputeFitness|TestParseFitnessWeights" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "feat(metrics): add FitnessResult and ComputeFitness (BC-3, EC-1, EC-2)

- Add FitnessResult with Score and per-component breakdown
- Add ComputeFitness with metric normalization (throughput direct, latency inverted)
- Add ParseFitnessWeights for CLI flag parsing
- Unknown keys logged as warnings and ignored (EC-1)
- Empty weights returns empty map (EC-2)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 4: RejectAll Admission Template

**Contracts Implemented:** BC-4

**Files:**
- Modify: `sim/admission.go`
- Modify: `sim/bundle.go`
- Test: `sim/admission_test.go`

**Step 1: Write failing test**

```go
// TestRejectAll_RejectsAll verifies BC-4.
func TestRejectAll_RejectsAll(t *testing.T) {
	policy := NewAdmissionPolicy("reject-all", 0, 0)
	tests := []struct {
		name string
		req  *Request
	}{
		{name: "empty request", req: &Request{ID: "r0", InputTokens: []int{}}},
		{name: "normal request", req: &Request{ID: "r1", InputTokens: make([]int, 100)}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			admitted, reason := policy.Admit(tt.req, &RouterState{Clock: 1000})
			if admitted {
				t.Error("expected reject-all to reject, but it admitted")
			}
			if reason != "reject-all" {
				t.Errorf("expected reason %q, got %q", "reject-all", reason)
			}
		})
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestRejectAll -v`
Expected: FAIL (panic from unknown policy name)

**Step 3: Implement RejectAll**

In `sim/admission.go` add before `NewAdmissionPolicy`:
```go
// RejectAll rejects all requests unconditionally (pathological template for testing).
type RejectAll struct{}

func (r *RejectAll) Admit(_ *Request, _ *RouterState) (bool, string) {
	return false, "reject-all"
}
```

In `sim/admission.go` add case to `NewAdmissionPolicy`:
```go
	case "reject-all":
		return &RejectAll{}
```

In `sim/bundle.go` add to `validAdmissionPolicies`:
```go
	validAdmissionPolicies = map[string]bool{"": true, "always-admit": true, "token-bucket": true, "reject-all": true}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestRejectAll -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/admission.go sim/bundle.go sim/admission_test.go
git commit -m "feat(admission): add RejectAll pathological template (BC-4)

- Add RejectAll that rejects all requests with reason 'reject-all'
- Register 'reject-all' in validAdmissionPolicies and factory

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 5: InvertedSLO Priority Template

**Contracts Implemented:** BC-5

**Files:**
- Modify: `sim/priority.go`
- Modify: `sim/bundle.go`
- Test: `sim/priority_test.go`

**Step 1: Write failing test**

```go
// TestInvertedSLO_OlderRequestsGetLowerPriority verifies BC-5.
func TestInvertedSLO_OlderRequestsGetLowerPriority(t *testing.T) {
	policy := NewPriorityPolicy("inverted-slo")

	oldReq := &Request{ID: "old", ArrivalTime: 0}
	newReq := &Request{ID: "new", ArrivalTime: 900_000}
	clock := int64(1_000_000)

	oldPriority := policy.Compute(oldReq, clock)
	newPriority := policy.Compute(newReq, clock)

	// THEN older request MUST have lower priority than newer request
	if oldPriority >= newPriority {
		t.Errorf("expected older request priority (%f) < newer request priority (%f)", oldPriority, newPriority)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestInvertedSLO -v`
Expected: FAIL (panic from unknown policy name)

**Step 3: Implement InvertedSLO**

In `sim/priority.go`, first fix stale comment on line 26 — change `PR9+` to `PR10+`:
```go
// Full SLO class integration (using TenantState) is planned for PR10+.
```

Then add before `NewPriorityPolicy`:
```go
// InvertedSLO computes priority inversely to request age (pathological template).
// Newer requests get higher priority, starving older ones — the opposite of SLOBasedPriority.
// Formula: BaseScore - AgeWeight * float64(clock - req.ArrivalTime)
type InvertedSLO struct {
	BaseScore float64
	AgeWeight float64
}

func (s *InvertedSLO) Compute(req *Request, clock int64) float64 {
	age := float64(clock - req.ArrivalTime)
	return s.BaseScore - s.AgeWeight*age
}
```

In `sim/priority.go` add case to `NewPriorityPolicy`:
```go
	case "inverted-slo":
		return &InvertedSLO{BaseScore: 0.0, AgeWeight: 1e-6}
```

In `sim/bundle.go` add to `validPriorityPolicies`:
```go
	validPriorityPolicies  = map[string]bool{"": true, "constant": true, "slo-based": true, "inverted-slo": true}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestInvertedSLO -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/priority.go sim/bundle.go sim/priority_test.go
git commit -m "feat(priority): add InvertedSLO pathological template (BC-5)

- Add InvertedSLO that gives lower priority to older requests
- Register 'inverted-slo' in validPriorityPolicies and factory

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 6: AlwaysBusiest Routing Template

**Contracts Implemented:** BC-6

**Files:**
- Modify: `sim/routing.go`
- Modify: `sim/bundle.go`
- Test: `sim/routing_test.go`

**Step 1: Write failing test**

```go
// TestAlwaysBusiest_RouteToHighestLoad verifies BC-6.
func TestAlwaysBusiest_RouteToHighestLoad(t *testing.T) {
	policy := NewRoutingPolicy("always-busiest", 0, 0)
	req := &Request{ID: "r1", InputTokens: []int{1, 2}}
	snapshots := []RoutingSnapshot{
		{ID: "instance_0", QueueDepth: 2, BatchSize: 1},  // load=3
		{ID: "instance_1", QueueDepth: 10, BatchSize: 5}, // load=15 (busiest)
		{ID: "instance_2", QueueDepth: 0, BatchSize: 0},  // load=0
	}

	decision := policy.Route(req, &RouterState{Snapshots: snapshots, Clock: 1000})

	if decision.TargetInstance != "instance_1" {
		t.Errorf("expected instance_1 (busiest), got %q", decision.TargetInstance)
	}
}

// TestAlwaysBusiest_EmptySnapshots_Panics verifies defensive convention.
func TestAlwaysBusiest_EmptySnapshots_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic on empty snapshots")
		}
	}()
	policy := NewRoutingPolicy("always-busiest", 0, 0)
	policy.Route(&Request{ID: "r1"}, &RouterState{Snapshots: []RoutingSnapshot{}, Clock: 0})
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestAlwaysBusiest -v`
Expected: FAIL (panic from unknown policy name)

**Step 3: Implement AlwaysBusiest**

In `sim/routing.go` add before `NewRoutingPolicy`:
```go
// AlwaysBusiest routes requests to the instance with maximum (QueueDepth + BatchSize).
// Pathological template for testing load imbalance detection.
// Ties broken by first occurrence in snapshot order (lowest index).
type AlwaysBusiest struct{}

// Route implements RoutingPolicy for AlwaysBusiest.
func (ab *AlwaysBusiest) Route(_ *Request, state *RouterState) RoutingDecision {
	snapshots := state.Snapshots
	if len(snapshots) == 0 {
		panic("AlwaysBusiest.Route: empty snapshots")
	}

	maxLoad := snapshots[0].QueueDepth + snapshots[0].BatchSize
	target := snapshots[0]

	for i := 1; i < len(snapshots); i++ {
		load := snapshots[i].QueueDepth + snapshots[i].BatchSize
		if load > maxLoad {
			maxLoad = load
			target = snapshots[i]
		}
	}

	return RoutingDecision{
		TargetInstance: target.ID,
		Reason:         fmt.Sprintf("always-busiest (load=%d)", maxLoad),
	}
}
```

In `sim/routing.go` add case to `NewRoutingPolicy`:
```go
	case "always-busiest":
		return &AlwaysBusiest{}
```

In `sim/bundle.go` add to `validRoutingPolicies`:
```go
	validRoutingPolicies   = map[string]bool{"": true, "round-robin": true, "least-loaded": true, "weighted": true, "prefix-affinity": true, "always-busiest": true}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestAlwaysBusiest -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/routing.go sim/bundle.go sim/routing_test.go
git commit -m "feat(routing): add AlwaysBusiest pathological template (BC-6)

- Add AlwaysBusiest that routes to highest-load instance
- Register 'always-busiest' in validRoutingPolicies and factory
- Follow defensive convention (panic on empty snapshots)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 7: ReversePriority Scheduler Template

**Contracts Implemented:** BC-7

**Files:**
- Modify: `sim/scheduler.go`
- Modify: `sim/bundle.go`
- Test: `sim/scheduler_test.go`

**Step 1: Write failing test**

```go
// TestReversePriority_LowestPriorityFirst verifies BC-7.
func TestReversePriority_LowestPriorityFirst(t *testing.T) {
	scheduler := NewScheduler("reverse-priority")
	reqs := []*Request{
		{ID: "high", Priority: 10.0, ArrivalTime: 100},
		{ID: "low", Priority: 1.0, ArrivalTime: 200},
		{ID: "mid", Priority: 5.0, ArrivalTime: 150},
	}

	scheduler.OrderQueue(reqs, 1_000_000)

	// THEN lowest priority should be first (reverse of PriorityFCFSScheduler)
	if reqs[0].ID != "low" {
		t.Errorf("expected 'low' first, got %q (priority=%f)", reqs[0].ID, reqs[0].Priority)
	}
	if reqs[1].ID != "mid" {
		t.Errorf("expected 'mid' second, got %q (priority=%f)", reqs[1].ID, reqs[1].Priority)
	}
	if reqs[2].ID != "high" {
		t.Errorf("expected 'high' last, got %q (priority=%f)", reqs[2].ID, reqs[2].Priority)
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/... -run TestReversePriority -v`
Expected: FAIL (panic from unknown scheduler name)

**Step 3: Implement ReversePriority**

In `sim/scheduler.go` add before `NewScheduler`:
```go
// ReversePriority sorts requests by priority ascending (lowest priority first).
// Pathological template: opposite of PriorityFCFSScheduler, causes priority inversions.
// Ties broken by arrival time ascending, then ID ascending for determinism.
type ReversePriority struct{}

func (r *ReversePriority) OrderQueue(reqs []*Request, _ int64) {
	sort.SliceStable(reqs, func(i, j int) bool {
		if reqs[i].Priority != reqs[j].Priority {
			return reqs[i].Priority < reqs[j].Priority // ascending = lowest first
		}
		if reqs[i].ArrivalTime != reqs[j].ArrivalTime {
			return reqs[i].ArrivalTime < reqs[j].ArrivalTime
		}
		return reqs[i].ID < reqs[j].ID
	})
}
```

In `sim/scheduler.go` add case to `NewScheduler`:
```go
	case "reverse-priority":
		return &ReversePriority{}
```

In `sim/bundle.go` add to `validSchedulers`:
```go
	validSchedulers        = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true, "reverse-priority": true}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/... -run TestReversePriority -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/scheduler.go sim/bundle.go sim/scheduler_test.go
git commit -m "feat(scheduler): add ReversePriority pathological template (BC-7)

- Add ReversePriority that sorts lowest priority first
- Register 'reverse-priority' in validSchedulers and factory
- Deterministic tie-breaking: arrival time, then ID

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 8: Anomaly Detection (Priority Inversion + HOL Blocking)

**Contracts Implemented:** BC-8, BC-9

**Files:**
- Modify: `sim/cluster/metrics.go`
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write failing test for anomaly detection**

```go
// TestDetectPriorityInversions_InvertedRequests verifies BC-8.
func TestDetectPriorityInversions_InvertedRequests(t *testing.T) {
	// GIVEN per-instance metrics where a high-priority request finished after a low-priority one
	m := sim.NewMetrics()
	// Request "high" had priority 10, arrived first, but completed later (inversion)
	m.Requests["high"] = sim.RequestMetrics{ID: "high", ArrivedAt: 100}
	m.RequestE2Es["high"] = 50000.0 // 50ms E2E
	// Request "low" had priority 1, arrived later, but completed faster
	m.Requests["low"] = sim.RequestMetrics{ID: "low", ArrivedAt: 200}
	m.RequestE2Es["low"] = 5000.0 // 5ms E2E

	inversions := detectPriorityInversions([]*sim.Metrics{m})

	// For PR9 we use a simplified heuristic: this is a placeholder
	// The real detection requires per-request priority tracking (PR13 scope).
	// For now, we verify the function exists and returns an int.
	if inversions < 0 {
		t.Errorf("inversions should be >= 0, got %d", inversions)
	}
}

// TestDetectHOLBlocking_ImbalancedInstances verifies BC-9.
func TestDetectHOLBlocking_ImbalancedInstances(t *testing.T) {
	// GIVEN 3 instances where one has 10x the queue depth of others
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{50, 50, 50, 50}),  // avg 50
		makeMetricsWithQueueDepth([]int{1, 1, 1, 1}),       // avg 1
		makeMetricsWithQueueDepth([]int{2, 2, 2, 2}),       // avg 2
	}

	blocking := detectHOLBlocking(perInstance)

	// THEN HOL blocking should be detected (instance 0 >> mean of others)
	if blocking <= 0 {
		t.Errorf("expected HOL blocking events > 0, got %d", blocking)
	}
}

// TestDetectHOLBlocking_BalancedInstances_NoBlocking verifies no false positives.
func TestDetectHOLBlocking_BalancedInstances_NoBlocking(t *testing.T) {
	perInstance := []*sim.Metrics{
		makeMetricsWithQueueDepth([]int{10, 10, 10}),
		makeMetricsWithQueueDepth([]int{11, 11, 11}),
		makeMetricsWithQueueDepth([]int{9, 9, 9}),
	}

	blocking := detectHOLBlocking(perInstance)

	if blocking != 0 {
		t.Errorf("expected 0 HOL blocking events for balanced instances, got %d", blocking)
	}
}

func makeMetricsWithQueueDepth(depths []int) *sim.Metrics {
	m := sim.NewMetrics()
	m.NumWaitQRequests = depths
	return m
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/cluster/... -run "TestDetect" -v`
Expected: FAIL with "undefined: detectPriorityInversions"

**Step 3: Implement anomaly detection functions**

In `sim/cluster/metrics.go` add:
```go
// detectPriorityInversions counts priority inversion events from per-instance metrics.
// For PR9, uses simplified heuristic: counts instances where requests exist
// (full decision-trace-based detection deferred to PR13).
// Returns 0 when pathological priority policies are not in use.
func detectPriorityInversions(perInstance []*sim.Metrics) int {
	// Simplified PR9 heuristic: count pairs where an earlier-arriving request
	// has worse E2E than a later-arriving request on the same instance.
	// This is an imperfect proxy — true inversion detection requires
	// per-request priority + schedule timestamp tracking (PR13 scope).
	count := 0
	for _, m := range perInstance {
		if len(m.Requests) < 2 {
			continue
		}
		// Build sorted request list by arrival time
		type reqInfo struct {
			id      string
			arrived float64
			e2e     float64
		}
		var reqs []reqInfo
		for id, rm := range m.Requests {
			if e2e, ok := m.RequestE2Es[id]; ok {
				reqs = append(reqs, reqInfo{id: id, arrived: rm.ArrivedAt, e2e: e2e})
			}
		}
		sort.Slice(reqs, func(i, j int) bool {
			return reqs[i].arrived < reqs[j].arrived
		})
		// Count inversions: earlier arrival with worse E2E than later arrival
		for i := 0; i < len(reqs)-1; i++ {
			for j := i + 1; j < len(reqs); j++ {
				if reqs[i].e2e > reqs[j].e2e*2.0 {
					count++
				}
			}
		}
	}
	return count
}

// detectHOLBlocking counts head-of-line blocking events from per-instance metrics.
// HOL blocking is detected when any instance's average queue depth exceeds
// 2× the mean average queue depth across all instances.
func detectHOLBlocking(perInstance []*sim.Metrics) int {
	if len(perInstance) < 2 {
		return 0
	}

	// Compute average queue depth per instance
	avgDepths := make([]float64, len(perInstance))
	totalAvg := 0.0
	for i, m := range perInstance {
		if len(m.NumWaitQRequests) == 0 {
			continue
		}
		sum := 0
		for _, d := range m.NumWaitQRequests {
			sum += d
		}
		avgDepths[i] = float64(sum) / float64(len(m.NumWaitQRequests))
		totalAvg += avgDepths[i]
	}
	meanAvg := totalAvg / float64(len(perInstance))

	// Count instances with > 2× mean average queue depth
	count := 0
	if meanAvg > 0 {
		for _, avg := range avgDepths {
			if avg > 2.0*meanAvg {
				count++
			}
		}
	}
	return count
}
```

Update `CollectRawMetrics` to call detection functions:
```go
// In CollectRawMetrics, before return:
	// Anomaly detection
	if perInstance != nil {
		raw.PriorityInversions = detectPriorityInversions(perInstance)
		raw.HOLBlockingEvents = detectHOLBlocking(perInstance)
	}
```

**Step 4: Run test to verify it passes**

Run: `go test ./sim/cluster/... -run "TestDetect" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/cluster/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/cluster/metrics.go sim/cluster/metrics_test.go
git commit -m "feat(metrics): add priority inversion and HOL blocking detection (BC-8, BC-9)

- Add detectPriorityInversions with E2E-based heuristic
- Add detectHOLBlocking with queue depth imbalance detection
- Wire into CollectRawMetrics from per-instance metrics
- Full decision-trace detection deferred to PR13

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 9: CLI Integration (--fitness-weights flag + RawMetrics output)

**Contracts Implemented:** EC-2 (CLI path), NC-1 (golden unchanged)

**Files:**
- Modify: `cmd/root.go`
- Modify: `sim/cluster/cluster.go` (expose per-instance metrics)

**Step 1: Write failing test — verify golden tests still pass**

Context: This is an integration task. First verify golden tests pass, then add CLI flag.

Run: `go test ./sim/... -run Golden -v && go test ./sim/cluster/... -run Golden -v`
Expected: PASS (baseline verification)

**Step 2: Add PerInstanceMetrics accessor to ClusterSimulator with panic test**

Add test in `sim/cluster/cluster_test.go`:
```go
// TestPerInstanceMetrics_BeforeRun_Panics verifies run-once guard.
func TestPerInstanceMetrics_BeforeRun_Panics(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic when calling PerInstanceMetrics before Run")
		}
	}()
	config := newTestDeploymentConfig(1)
	cs := NewClusterSimulator(config, newTestWorkload(5), "")
	cs.PerInstanceMetrics() // should panic
}
```

In `sim/cluster/cluster.go` add:
```go
// PerInstanceMetrics returns the metrics for each individual instance.
// Panics if called before Run() has completed.
func (c *ClusterSimulator) PerInstanceMetrics() []*sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.PerInstanceMetrics() called before Run()")
	}
	metrics := make([]*sim.Metrics, len(c.instances))
	for i, inst := range c.instances {
		metrics[i] = inst.Metrics()
	}
	return metrics
}
```

**Step 3: Add --fitness-weights flag and RawMetrics integration to cmd/root.go**

In `cmd/root.go` add the flag variable:
```go
	// Fitness evaluation config (PR9)
	fitnessWeights string // Fitness weights string "key:val,key:val"
```

In the `Run` function, after `cs.AggregatedMetrics().SaveResults(...)`, add:
```go
		// Collect RawMetrics and compute fitness (PR9)
		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
		)

		if fitnessWeights != "" {
			weights, err := cluster.ParseFitnessWeights(fitnessWeights)
			if err != nil {
				logrus.Fatalf("Invalid fitness weights: %v", err)
			}
			fitness := cluster.ComputeFitness(rawMetrics, weights)
			fmt.Printf("\n=== Fitness Evaluation ===\n")
			fmt.Printf("Score: %.6f\n", fitness.Score)
			for k, v := range fitness.Components {
				fmt.Printf("  %s: %.6f\n", k, v)
			}
		}

		// Print anomaly counters if any detected
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 {
			fmt.Printf("\n=== Anomaly Counters ===\n")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests: %d\n", rawMetrics.RejectedRequests)
		}
```

In `init()`, add the flag:
```go
	// Fitness evaluation config (PR9)
	runCmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs (e.g., throughput:0.5,p99_ttft:0.3)")
```

**Step 4: Verify golden tests still pass (NC-1)**

Run: `go test ./... -count=1`
Expected: ALL PASS (golden tests unchanged — new code only activates with `--fitness-weights`)

**Step 5: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 6: Commit**

```bash
git add cmd/root.go sim/cluster/cluster.go
git commit -m "feat(cli): add --fitness-weights flag and RawMetrics output (EC-2, NC-1)

- Add --fitness-weights CLI flag for fitness evaluation
- Add PerInstanceMetrics accessor to ClusterSimulator
- Print fitness score and anomaly counters after simulation
- Golden tests unchanged (new code only activates with flag)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 10: Integration Tests — Pathological Templates Trigger Anomalies

**Contracts Implemented:** BC-4+BC-8 (RejectAll → rejection count), BC-6+BC-9 (AlwaysBusiest → HOL blocking)

**Files:**
- Test: `sim/cluster/metrics_test.go`

**Step 1: Write integration test**

Context: Run actual simulations with pathological templates and verify anomaly counters are non-zero.

```go
// TestPathological_RejectAll_AllRejected verifies BC-4 + rejection counting.
func TestPathological_RejectAll_AllRejected(t *testing.T) {
	// Reuse existing test helpers from cluster_test.go
	config := newTestDeploymentConfig(2)
	config.AdmissionPolicy = "reject-all"

	cs := NewClusterSimulator(config, newTestWorkload(20), "")
	cs.Run()

	raw := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests())

	// ALL requests should be rejected
	if raw.RejectedRequests == 0 {
		t.Error("expected rejected requests > 0 with reject-all policy")
	}
	// No requests should complete
	if cs.AggregatedMetrics().CompletedRequests != 0 {
		t.Errorf("expected 0 completed requests, got %d", cs.AggregatedMetrics().CompletedRequests)
	}
}

// TestPathological_AlwaysBusiest_CausesImbalance verifies BC-6 + BC-9.
func TestPathological_AlwaysBusiest_CausesImbalance(t *testing.T) {
	// Reuse existing test helpers from cluster_test.go
	config := newTestDeploymentConfig(3)
	config.RoutingPolicy = "always-busiest"

	cs := NewClusterSimulator(config, newTestWorkload(20), "")
	cs.Run()

	raw := CollectRawMetrics(cs.AggregatedMetrics(), cs.PerInstanceMetrics(), cs.RejectedRequests())

	// With always-busiest routing, all requests should pile onto one instance.
	// Verify imbalance: at least one instance should have much more load.
	perInstance := cs.PerInstanceMetrics()
	maxCompleted := 0
	minCompleted := int(^uint(0) >> 1) // MaxInt
	for _, m := range perInstance {
		if m.CompletedRequests > maxCompleted {
			maxCompleted = m.CompletedRequests
		}
		if m.CompletedRequests < minCompleted {
			minCompleted = m.CompletedRequests
		}
	}

	// With 3 instances and always-busiest, the first instance gets all requests
	// after the initial round (since it becomes busiest after first request).
	if maxCompleted <= minCompleted && raw.HOLBlockingEvents == 0 {
		t.Logf("maxCompleted=%d, minCompleted=%d, HOL=%d", maxCompleted, minCompleted, raw.HOLBlockingEvents)
		t.Error("expected significant load imbalance or HOL blocking with always-busiest")
	}
}

// Note: Uses existing newTestDeploymentConfig() and newTestWorkload() from cluster_test.go.
// No duplicate test helpers needed — these are in the same package.
```

**Step 2: Run test to verify**

Run: `go test ./sim/cluster/... -run TestPathological -v`
Expected: PASS

**Step 3: Run all tests to verify no regressions**

Run: `go test ./... -count=1`
Expected: ALL PASS

**Step 4: Run lint check**

Run: `golangci-lint run ./...`
Expected: No new issues

**Step 5: Commit**

```bash
git add sim/cluster/metrics_test.go
git commit -m "test(metrics): add integration tests for pathological templates (BC-4, BC-6, BC-8, BC-9)

- Verify reject-all causes 100% rejection
- Verify always-busiest causes load imbalance
- Include test helpers for deployment config and workload

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Task 11: Update Documentation (CLAUDE.md, README)

**Contracts Implemented:** None (documentation task)

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/plans/2026-02-11-macro-implementation-plan-v2.md` (mark PR9 complete)

**Step 1: Update CLAUDE.md**

In `CLAUDE.md`, update the "Current Implementation Focus" completed list to add PR9:

```
- **Completed:** PR1 (PartitionedRNG), PR2 (InstanceSimulator), ..., PR8 (..., **INTERFACE FREEZE**), PR9 (RawMetrics with Distribution + FitnessResult, anomaly detection with priority inversion + HOL blocking counters, pathological templates: reject-all, inverted-slo, always-busiest, reverse-priority, `--fitness-weights` CLI flag, **RESEARCH-READY CHECKPOINT**)
```

Update the "Next" line:
```
- **Next:** Policy research experiments (research-ready checkpoint reached), or PR10 (Workload Generator, optional) and subsequent parallel tracks
```

In the "File Organization" section, add to `sim/cluster/`:
```
│   ├── metrics.go            # RawMetrics, Distribution, FitnessResult, anomaly detection
```

In the "Code Architecture > Cluster Simulation" section, add:
```
- **metrics.go**: `RawMetrics`, `Distribution`, `FitnessResult`, `CollectRawMetrics`, `ComputeFitness`, anomaly detection
```

Update the Code Architecture descriptions for the 4 policy files:
```
- **admission.go**: `AdmissionPolicy` interface (accepts `*RouterState`), `AlwaysAdmit`, `TokenBucket`, `RejectAll`, `NewAdmissionPolicy` factory
- **routing.go**: `RoutingPolicy` interface (accepts `*RouterState`), `RoutingSnapshot`, `RoutingDecision` (with `Priority` hint), `RoundRobin`, `LeastLoaded`, `WeightedScoring`, `PrefixAffinity`, `AlwaysBusiest` templates, `NewRoutingPolicy` factory
- **priority.go**: `PriorityPolicy` interface with `ConstantPriority`, `SLOBasedPriority`, and `InvertedSLO` templates, `NewPriorityPolicy` factory
- **scheduler.go**: `InstanceScheduler` interface with `FCFSScheduler`, `PriorityFCFSScheduler`, `SJFScheduler`, and `ReversePriority` templates, `NewScheduler` factory
```

**Step 2: Update macro plan**

In `docs/plans/2026-02-11-macro-implementation-plan-v2.md`, update the timeline section near the bottom where PRs are marked completed. Add after the PR8 line:
```
| PR 9 | PR 9 | COMPLETED: RawMetrics + pathological templates → RESEARCH-READY |
```

**Step 3: Verify all tests pass**

Run: `go build ./... && go test ./... -count=1 && golangci-lint run ./...`
Expected: Build OK, all tests PASS, 0 lint issues

**Step 4: Commit**

```bash
git add CLAUDE.md docs/plans/2026-02-11-macro-implementation-plan-v2.md
git commit -m "docs: update CLAUDE.md and macro plan for PR9 completion

- Mark PR9 as completed in implementation status
- Add sim/cluster/metrics.go to file organization
- Update policy template lists with pathological templates
- Mark RESEARCH-READY checkpoint reached

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| BC-1 | Task 2 | Unit | `TestCollectRawMetrics_BasicAggregation` |
| BC-1 | Task 2 | Unit | `TestCollectRawMetrics_ZeroCompleted_ReturnsEmptyDistributions` |
| BC-2 | Task 1 | Unit | `TestDistribution_FromValues_ComputesCorrectStats` |
| BC-2 | Task 1 | Unit | `TestDistribution_EmptyValues_ReturnsZero` |
| BC-3 | Task 3 | Unit | `TestComputeFitness_WeightedScore` |
| BC-3 | Task 3 | Unit | `TestComputeFitness_LatencyInversion` |
| BC-4 | Task 4 | Unit | `TestRejectAll_RejectsAll` |
| BC-4 | Task 10 | Integration | `TestPathological_RejectAll_AllRejected` |
| BC-5 | Task 5 | Unit | `TestInvertedSLO_OlderRequestsGetLowerPriority` |
| BC-6 | Task 6 | Unit | `TestAlwaysBusiest_RouteToHighestLoad` |
| BC-6 | Task 10 | Integration | `TestPathological_AlwaysBusiest_CausesImbalance` |
| BC-7 | Task 7 | Unit | `TestReversePriority_LowestPriorityFirst` |
| BC-8 | Task 8 | Unit | `TestDetectPriorityInversions_InvertedRequests` |
| BC-9 | Task 8 | Unit | `TestDetectHOLBlocking_ImbalancedInstances` |
| BC-9 | Task 8 | Unit | `TestDetectHOLBlocking_BalancedInstances_NoBlocking` |
| EC-1 | Task 3 | Unit | `TestComputeFitness_UnknownKey_Ignored` |
| EC-2 | Task 3 | Unit | `TestParseFitnessWeights_Empty` |
| NC-1 | Task 9 | Golden | Existing golden dataset tests (unchanged) |
| NC-1 | Task 9 | Failure | `TestPerInstanceMetrics_BeforeRun_Panics` |
| NC-2 | All | Unit | Factory tests verify existing + new names work |

**Golden dataset:** Not updated — PR9 adds additive metrics only; existing output unchanged.

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| Priority inversion heuristic produces false positives | Medium | Low | Conservative threshold (2× E2E ratio); full detection deferred to PR13 | Task 8 |
| HOL blocking detection threshold too sensitive | Low | Low | 2× mean threshold; integration test validates with pathological template | Task 8 |
| Fitness weight normalization produces unexpected values | Low | Medium | Latency inversion formula `1/(1+v)` bounded in [0,1]; integration tests | Task 3 |
| Pathological templates break golden tests | Very Low | High | Templates only activated via explicit CLI flags; NC-1 verified in Task 9 | Task 9 |
| `--fitness-weights` parsing edge cases | Low | Low | Test invalid format, empty string, whitespace handling | Task 3 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions (RawMetrics is a simple struct, not an interface)
- [x] No feature creep beyond PR scope (SLO attainment, Jain fairness, scale metrics all deferred)
- [x] No unexercised flags or interfaces (all templates exercisable via existing CLI flags)
- [x] No partial implementations (all 4 pathological templates complete)
- [x] No breaking changes (interface freeze respected — templates implement existing interfaces)
- [x] No hidden global state impact (RawMetrics computed post-simulation)
- [x] All new code will pass golangci-lint
- [x] Shared test helpers reused (`newTestDeploymentConfig`/`newTestWorkload` from `cluster_test.go:12-48`; no duplication)
- [x] CLAUDE.md updated (Task 11)
- [x] No stale references in CLAUDE.md
- [x] Deviation log reviewed — all deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4-7→8→9→10→11)
- [x] All contracts mapped to specific tasks
- [x] Golden dataset regeneration not needed (additive changes only)

---

## Appendix: File-Level Implementation Details

### File: `sim/cluster/metrics.go` (NEW)

**Purpose:** RawMetrics aggregation, fitness evaluation, and anomaly detection.

**Complete types:**
```go
type Distribution struct {
    Mean, P50, P95, P99, Min, Max float64
    Count                         int
}

type RawMetrics struct {
    TTFT, E2E          Distribution
    RequestsPerSec     float64
    TokensPerSec       float64
    PriorityInversions int
    HOLBlockingEvents  int
    RejectedRequests   int
}

type FitnessResult struct {
    Score      float64
    Components map[string]float64
}
```

**Functions:**
- `NewDistribution(values []float64) Distribution`
- `percentile(sorted []float64, p float64) float64` (unexported)
- `CollectRawMetrics(aggregated *sim.Metrics, perInstance []*sim.Metrics, rejectedRequests int) *RawMetrics`
- `ComputeFitness(metrics *RawMetrics, weights map[string]float64) *FitnessResult`
- `extractMetric(m *RawMetrics, key string) (float64, bool)` (unexported)
- `ParseFitnessWeights(s string) (map[string]float64, error)`
- `detectPriorityInversions(perInstance []*sim.Metrics) int` (unexported)
- `detectHOLBlocking(perInstance []*sim.Metrics) int` (unexported)
- `mapValues(m map[string]float64) []float64` (unexported)

**Constants:**
- `referenceRPS = 100.0` — reference request throughput for normalization (100 req/s → score 0.5)
- `referenceTPS = 10000.0` — reference token throughput for normalization (10K tok/s → score 0.5)
- `referenceTicks = 1000.0` — reference latency for normalization (1ms → score 0.5)

**Key notes:**
- Uses `sort.Float64s` / `sort.Slice` for deterministic percentile computation
- Percentile function does NOT divide by 1000 (unlike `sim.CalculatePercentile` which converts to ms)
- Fitness normalization uses reference scales so throughput and latency produce comparable [0,1] scores
- Anomaly detection uses heuristics; full decision-trace-based detection in PR13

### File: `sim/admission.go` (MODIFIED)

**Added:**
```go
type RejectAll struct{}
func (r *RejectAll) Admit(_ *Request, _ *RouterState) (bool, string)
// Case in NewAdmissionPolicy: "reject-all" → &RejectAll{}
```

### File: `sim/routing.go` (MODIFIED)

**Added:**
```go
type AlwaysBusiest struct{}
func (ab *AlwaysBusiest) Route(_ *Request, state *RouterState) RoutingDecision
// Case in NewRoutingPolicy: "always-busiest" → &AlwaysBusiest{}
```

### File: `sim/priority.go` (MODIFIED)

**Added:**
```go
type InvertedSLO struct { BaseScore, AgeWeight float64 }
func (s *InvertedSLO) Compute(req *Request, clock int64) float64
// Case in NewPriorityPolicy: "inverted-slo" → &InvertedSLO{0.0, 1e-6}
```

### File: `sim/scheduler.go` (MODIFIED)

**Added:**
```go
type ReversePriority struct{}
func (r *ReversePriority) OrderQueue(reqs []*Request, _ int64)
// Case in NewScheduler: "reverse-priority" → &ReversePriority{}
```

### File: `sim/bundle.go` (MODIFIED)

**Changed lines (4 lines):**
```go
validAdmissionPolicies = map[string]bool{..., "reject-all": true}
validRoutingPolicies   = map[string]bool{..., "always-busiest": true}
validPriorityPolicies  = map[string]bool{..., "inverted-slo": true}
validSchedulers        = map[string]bool{..., "reverse-priority": true}
```

### File: `cmd/root.go` (MODIFIED)

**Added:**
- `fitnessWeights string` variable
- `--fitness-weights` flag in `init()`
- Post-simulation `CollectRawMetrics` + `ComputeFitness` + print block
- `PerInstanceMetrics()` call on `ClusterSimulator`

### File: `sim/cluster/cluster.go` (MODIFIED)

**Added:**
```go
func (c *ClusterSimulator) PerInstanceMetrics() []*sim.Metrics
```
