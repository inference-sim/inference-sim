# PR10: ServeGen-Informed Workload Generator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a full observe-predict-calibrate loop with ServeGen-informed workload generation: heterogeneous client decomposition, bursty arrival processes, empirical distributions, real mode HTTP client, trace v2 format, calibration framework, multimodal/reasoning workloads, network model, and per-SLO-class metrics.

**Architecture:** New `sim/workload/` package handles pure request generation from YAML specs. `cmd/observe.go` handles real mode HTTP. `sim/workload/tracev2.go` handles trace format. `sim/workload/calibrate.go` handles real-vs-sim comparison. Import boundary: `sim/workload/` → `sim/` only (never `sim/cluster/`). `cmd/` orchestrates all three packages.

**Tech Stack:** Go 1.21, `gopkg.in/yaml.v3` (strict parsing), `gonum.org/v1/gonum/stat/distuv` (statistical distributions), `github.com/montanaflynn/stats` (percentiles/correlation)

**Macro Plan Reference:** Phase 3, PR 10 in `docs/plans/2026-02-11-macro-implementation-plan-v2.md`

**Design Document:** `docs/plans/2026-02-16-workload-generator-design.md` (authoritative reference)

**Behavioral Contracts:** See Part 1, Section B below

---

## Part 1: Design Validation

### A) Executive Summary

PR10 creates the `sim/workload/` package — a ServeGen-informed workload generator that models heterogeneous clients with skewed rates, bursty arrival processes (Gamma/Weibull/Poisson), and empirical token length distributions. It also adds a real mode HTTP client for observing actual inference servers, a trace v2 format for recording/replaying workloads, a calibration framework for comparing real vs simulated latencies, multimodal/reasoning workload categories, a per-client network latency model, and per-SLO-class metric extensions.

**Adjacent blocks:** `sim/` (Request struct extended with TenantID, SLOClass, and other metadata fields), `sim/cluster/` (metrics.go extended with per-SLO distributions and fairness index), `cmd/` (new flags and observe subcommand).

**No deviations flagged from Phase 0 inspection.** The macro plan's PR10 entry points to the design doc, and the design doc's architecture matches the current codebase.

### B) Behavioral Contracts

**Positive Contracts:**

BC-1: Deterministic Generation
- GIVEN a WorkloadSpec with seed=42 and two clients
- WHEN GenerateRequests is called twice with the same spec and horizon
- THEN both calls produce identical request sequences (same IDs, arrival times, token counts)
- MECHANISM: PartitionedRNG with SubsystemWorkloadGen constant; per-client RNG derived from client index

BC-2: Client Rate Proportionality
- GIVEN a spec with client A (rate_fraction=0.7) and client B (rate_fraction=0.3) and aggregate_rate=100
- WHEN 1000+ requests are generated over a sufficient horizon
- THEN client A produces ~70% of requests and client B ~30% (within 5% tolerance)
- MECHANISM: Per-client rate = aggregate_rate * normalized_fraction; per-client arrival sampler generates at that rate

BC-3: Bursty Arrivals
- GIVEN a GammaSampler with CV=3.5 and a PoissonSampler with the same mean rate
- WHEN 1000 inter-arrival times are sampled from each
- THEN the Gamma sampler's coefficient of variation is measurably higher than the Poisson's (CV > 2.0 vs ~1.0)
- MECHANISM: Gamma shape = 1/CV², scale = CV²/rate; higher CV → more variance → burstier

BC-4: Empirical PDF Faithfulness
- GIVEN an EmpiricalPDFSampler loaded from a ServeGen dataset JSON
- WHEN 10000 samples are drawn
- THEN the sample distribution's mean is within 5% of the PDF's theoretical mean
- MECHANISM: CDF constructed from PDF probabilities; inverse CDF sampling via binary search

BC-5: Backward Compatibility
- GIVEN existing CLI invocations (`--workload distribution` and `--workload traces`)
- WHEN run with no `--workload-spec` flag
- THEN behavior is identical to pre-PR10 (golden dataset tests pass unchanged)
- MECHANISM: New workload-spec path is additive; existing generation code untouched

BC-6: Trace v2 Round-Trip
- GIVEN a set of generated requests exported to trace v2 format (header YAML + data CSV)
- WHEN the trace is loaded back
- THEN every request has matching arrival time, token counts, client ID, tenant ID, and SLO class
- MECHANISM: CSV export writes all fields; CSV loader parses all columns with exact type preservation

BC-7: Calibration Statistics
- GIVEN paired real and sim latency values for 100+ requests
- WHEN ComputeCalibration is called
- THEN MAPE, Pearson r, and per-percentile errors are computed correctly (verified against hand-calculated values)
- MECHANISM: Standard statistical formulas; montanaflynn/stats for percentiles

BC-8: Multimodal Token Accounting
- GIVEN a multimodal request with text=100, image=200, audio=50 tokens
- WHEN the request is generated
- THEN len(InputTokens) == 350 AND TextTokenCount==100 AND ImageTokenCount==200 AND AudioTokenCount==50
- MECHANISM: Generator concatenates modality token slices; metadata fields set from individual counts

BC-9: Network Latency Adjustment
- GIVEN a request with NetworkSpec{RTTMs: 10.0} and server TTFT of 500µs
- WHEN client-perspective TTFT is computed
- THEN client_ttft == 500 + 10000 (10ms in µs) = 10500µs
- MECHANISM: ComputeClientLatency adds RTT (converted to µs) to server-side metrics

BC-10: Per-SLO-Class Distribution Segregation
- GIVEN requests with SLOClass "realtime" (N=50) and "batch" (N=150)
- WHEN per-SLO-class distributions are computed
- THEN the "realtime" distribution has Count==50 and "batch" has Count==150
- MECHANISM: Filter requests by SLOClass before computing NewDistribution

BC-11: Calibration Orchestration Normalization
- GIVEN a real trace with 100 requests (including 10 warm-up) where real TTFT includes 5ms RTT, and a sim run of the same requests where sim TTFT is pure server-side
- WHEN the calibrate subcommand runs with the trace header specifying network RTT=5ms
- THEN (a) real and sim are matched by request_id (not position), (b) sim TTFT has RTT added (client-perspective), (c) warm-up requests are excluded, (d) MAPE is computed on the 90 remaining matched pairs, (e) time units are verified as microseconds on both sides
- MECHANISM: `PrepareCalibrationPairs` function matches, normalizes, and filters before passing to `ComputeCalibration`

BC-12: Calibration Report Annotations
- GIVEN any calibration run
- WHEN the report is generated
- THEN it always includes: (a) known limitations (prefix cache divergence, speculative decoding), (b) config match section (which sim params matched trace header vs defaulted), (c) token count mismatch warning if real and sim token counts differ for any request
- MECHANISM: CalibrationReport struct includes KnownLimitations, ConfigMatch, and TokenMismatchCount fields

**Negative Contracts:**

NC-1: No Import Cycle
- `sim/` and `sim/cluster/` MUST NOT import `sim/workload/`
- MECHANISM: `sim/workload/` only imports `sim/` for Request struct and RNG helpers

NC-2: No Silent YAML Errors
- Unknown YAML keys MUST produce parse errors (not be silently ignored)
- MECHANISM: `yaml.KnownFields(true)` on decoder, matching `LoadPolicyBundle` pattern

NC-3: No Panic from User Input
- Invalid workload-spec YAML MUST produce descriptive errors, not panics
- MECHANISM: All user-facing validation returns errors; `logrus.Fatalf` at CLI level only

NC-4: No Global RNG
- `sim/workload/` MUST NOT call `math/rand` package-level functions (e.g., `rand.Intn()`)
- All RNG usage MUST go through `*rand.Rand` instances from PartitionedRNG
- MECHANISM: Code review; gonum distributions receive explicit `Src` field

**Error Handling Contracts:**

EC-1: Invalid Distribution Params
- GIVEN a DistSpec with negative mean or NaN alpha
- WHEN Validate() is called
- THEN a descriptive error is returned naming the invalid field and value
- MECHANISM: validateFloat pattern from sim/bundle.go applied to all DistSpec params

EC-2: Empty Client List
- GIVEN a WorkloadSpec with no clients and no servegen_data
- WHEN Validate() is called
- THEN error: "at least one client or servegen_data path required"

EC-3: Real Mode Connection Failure
- GIVEN --real-mode with an unreachable --server-url
- WHEN the first request is sent
- THEN error is logged with URL and connection details; request recorded with status="error"

EC-4: Zero Rate Guard
- GIVEN a client whose normalized rate fraction rounds to 0 (or AggregateRate=0)
- WHEN GenerateRequests is called
- THEN the client produces 0 requests (not an infinite loop or division by zero)
- MECHANISM: Guard `if clientRate <= 0 { continue }` in per-client loop

EC-5: Zero Horizon Guard
- GIVEN horizon=0
- WHEN GenerateRequests is called
- THEN returns empty slice immediately (no infinite loop)
- MECHANISM: Guard `if horizon <= 0 { return nil, nil }` at top of GenerateRequests

### C) Component Interaction

```
cmd/root.go ─── --workload-spec ───> sim/workload/spec.go (LoadWorkloadSpec)
     │                                      │
     │                                      ▼
     │                              sim/workload/generator.go (GenerateRequests)
     │                                      │
     │              ┌───────────────────────┤
     │              ▼                       ▼
     │     sim/workload/arrival.go   sim/workload/distribution.go
     │     (ArrivalSampler)          (LengthSampler)
     │              │                       │
     │              └───────────┬───────────┘
     │                          ▼
     │                   []*sim.Request ──────> ClusterSimulator.Run()
     │
     ├─── --real-mode ───> cmd/observe.go (LoadGenerator + RealClient)
     │                          │
     │                          ▼
     │                   sim/workload/tracev2.go (Export)
     │
     └─── calibrate ────> sim/workload/calibrate.go (ComputeCalibration)
```

**Import boundary (enforced):**
- `sim/workload/` → `sim/` (Request, PartitionedRNG, GenerateRandomTokenIDs)
- `cmd/` → `sim/workload/`, `sim/cluster/`, `sim/`
- `sim/cluster/` → `sim/` (unchanged — never imports sim/workload/)

**New types crossing boundaries:**
- `sim/workload.WorkloadSpec` — consumed by cmd/, passed to GenerateRequests
- `[]*sim.Request` — output of GenerateRequests, consumed by ClusterSimulator
- `sim/workload.CalibrationReport` — output of ComputeCalibration, serialized to JSON by cmd/

### D) Deviation Log

| Macro Plan Says | Micro Plan Does | Reason |
|-----------------|-----------------|--------|
| Design doc section 2 shows `tracev2.go` and `calibrate.go` in `sim/cluster/` | Placed in `sim/workload/` (matching macro plan) | `sim/cluster/` must not import `sim/workload/`. These files need workload types, so they belong in `sim/workload/`. Design doc section 2 is stale; section 13 (commit grouping) has the correct paths. |
| Design doc section 13 lists `sim/workload/recorder.go` as separate file | Recorder types colocated in `cmd/observe.go` | Small enough to colocate; avoids file proliferation |
| Design doc mentions hand-rolled Marsaglia-Tsang for Gamma | Use gonum's `distuv.Gamma` for Gamma sampler; hand-rolled for Poisson and Weibull | Marsaglia-Tsang is 30-40 lines with subtle numerical pitfalls; gonum's implementation is well-tested. Poisson and Weibull are trivial (1-line formulas). |
| No explicit mention | Add `gonum.org/v1/gonum` and `github.com/montanaflynn/stats` to go.mod | User approved these dependencies for statistical calculations |

### E) Review Guide

**The tricky part:** The EmpiricalPDF sampler must handle ServeGen's data format correctly — the dataset JSON stores PDFs as Python dict string representations (`{224: 0.0003, 225: 0.0007, ...}`) keyed by time windows. The loading code must parse these string-encoded dicts and build proper CDFs.

**What to scrutinize:** BC-1 (determinism) and BC-4 (empirical PDF faithfulness) — these are the hardest to get right. Verify that RNG partitioning produces truly isolated streams and that CDF construction handles edge cases (single-bin PDFs, non-normalized probabilities).

**What's safe to skim:** The YAML spec types (ClientSpec, ArrivalSpec, DistSpec) are straightforward structs. The CLI flag additions follow the exact pattern of every prior PR.

**Known debt:** Prometheus scraping (mentioned in design doc section 8) is explicitly out of scope. Multi-turn real mode conversation flow is included but only tested with mocks. Memory scaling: `GenerateRequests` materializes all requests into a `[]*sim.Request` slice before simulation starts — follows existing BLIS pattern (`sim/cluster/workload.go`) but could be a concern for very long horizons at high rates (e.g., 100 req/s × 1 hour = 360K requests × ~5KB each ≈ 2GB). Streaming/lazy generation deferred to future PR.

---

## Part 2: Executable Implementation

### F) Implementation Overview

**Files to create:**
- `sim/workload/spec.go` — WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, validation, YAML loading
- `sim/workload/arrival.go` — ArrivalSampler interface + Poisson, Gamma, Weibull
- `sim/workload/distribution.go` — LengthSampler interface + Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF
- `sim/workload/client.go` — ClientPool, rate normalization, prefix group management
- `sim/workload/generator.go` — GenerateRequests pipeline, ServeGen data loading
- `sim/workload/tracev2.go` — TraceHeader, TraceRecord, Export/Load functions
- `sim/workload/calibrate.go` — CalibrationReport, ComputeCalibration, MAPE, Pearson r
- `sim/workload/network.go` — ComputeClientLatency
- `sim/workload/multimodal.go` — multimodal token generation
- `sim/workload/reasoning.go` — reasoning ratio, multi-turn generation
- `sim/workload/scenarios.go` — built-in scenario presets
- `cmd/observe.go` — RealClient, LoadGenerator, Recorder

**Files to modify:**
- `sim/request.go` — Add TenantID, SLOClass, Streaming, SessionID, RoundIndex, modality counts, ReasonRatio
- `sim/rng.go` — Add SubsystemWorkloadGen constant
- `cmd/root.go` — Add --workload-spec flag, integrate with ClusterSimulator
- `sim/cluster/workload.go` — Add third generation path for workload-spec
- `sim/cluster/cluster.go` — Accept workload-spec requests
- `sim/cluster/metrics.go` — Per-SLO-class distributions, SLOAttainment, JainFairnessIndex
- `go.mod` / `go.sum` — Add gonum and montanaflynn/stats dependencies

**Key decisions:**
- Arrival samplers use hand-rolled math (simple formulas) rather than gonum distributions
- gonum/stat used for Pearson correlation in calibration
- montanaflynn/stats used for percentile calculations in calibration
- Real mode HTTP client tested with mock `httptest.Server`
- All new Request fields have zero-value defaults (backward compatible)

### G) Task Breakdown

---

#### Section 1: Core Workload Specification

### Task 1: Spec Types + YAML Loading + Validation

**Contracts Implemented:** NC-2, EC-1, EC-2

**Files:**
- Create: `sim/workload/spec.go`
- Test: `sim/workload/spec_test.go`

**Step 1: Write failing test for YAML loading and validation**

Context: We need WorkloadSpec to load from YAML with strict parsing (no unknown keys) and validate all fields.

```go
package workload

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadWorkloadSpec_ValidYAML_LoadsCorrectly(t *testing.T) {
	// GIVEN a valid workload spec YAML file
	dir := t.TempDir()
	path := filepath.Join(dir, "spec.yaml")
	yaml := `
version: "1"
seed: 42
category: language
aggregate_rate: 100.0
clients:
  - id: "client-a"
    tenant_id: "tenant-1"
    slo_class: "batch"
    rate_fraction: 0.7
    streaming: false
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 128
        min: 10
        max: 4096
    output_distribution:
      type: exponential
      params:
        mean: 256
`
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN the spec is loaded
	spec, err := LoadWorkloadSpec(path)

	// THEN it loads without error and fields are correct
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.Version != "1" {
		t.Errorf("version = %q, want %q", spec.Version, "1")
	}
	if spec.Seed != 42 {
		t.Errorf("seed = %d, want 42", spec.Seed)
	}
	if spec.AggregateRate != 100.0 {
		t.Errorf("aggregate_rate = %f, want 100.0", spec.AggregateRate)
	}
	if len(spec.Clients) != 1 {
		t.Fatalf("clients count = %d, want 1", len(spec.Clients))
	}
	c := spec.Clients[0]
	if c.ID != "client-a" || c.TenantID != "tenant-1" || c.SLOClass != "batch" {
		t.Errorf("client fields mismatch: id=%q tenant=%q slo=%q", c.ID, c.TenantID, c.SLOClass)
	}
	if c.RateFraction != 0.7 {
		t.Errorf("rate_fraction = %f, want 0.7", c.RateFraction)
	}
}

func TestLoadWorkloadSpec_UnknownKey_ReturnsError(t *testing.T) {
	// GIVEN a YAML with an unknown key (typo)
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.yaml")
	yaml := `
version: "1"
seed: 42
aggreate_rate: 100.0
`
	if err := os.WriteFile(path, []byte(yaml), 0644); err != nil {
		t.Fatal(err)
	}

	// WHEN loaded
	_, err := LoadWorkloadSpec(path)

	// THEN strict parsing rejects the unknown key
	if err == nil {
		t.Fatal("expected error for unknown key, got nil")
	}
}

func TestWorkloadSpec_Validate_EmptyClients_ReturnsError(t *testing.T) {
	// GIVEN a spec with no clients and no servegen_data
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
	}

	// WHEN validated
	err := spec.Validate()

	// THEN returns descriptive error
	if err == nil {
		t.Fatal("expected validation error for empty clients")
	}
}

func TestWorkloadSpec_Validate_InvalidArrivalProcess_ReturnsError(t *testing.T) {
	spec := &WorkloadSpec{
		Version:       "1",
		AggregateRate: 100.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "invalid"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 10, "min": 1, "max": 1000}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	err := spec.Validate()
	if err == nil {
		t.Fatal("expected error for invalid arrival process")
	}
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestLoadWorkloadSpec -v`
Expected: FAIL (package doesn't exist yet)

**Step 3: Implement spec types and YAML loading**

Context: Create the full spec type hierarchy following the design doc section 3, with LoadWorkloadSpec following the LoadPolicyBundle pattern.

Create `sim/workload/spec.go` with:
- All spec types: WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, NetworkSpec, LifecycleSpec, ActiveWindow, MultimodalSpec, ReasoningSpec, MultiTurnSpec, ServeGenDataSpec
- `LoadWorkloadSpec(path string) (*WorkloadSpec, error)` — strict YAML parsing
- `(*WorkloadSpec) Validate() error` — validates all fields, arrival process names, distribution types, parameter ranges
- Valid process names: "poisson", "gamma", "weibull"
- Valid distribution types: "gaussian", "exponential", "pareto_lognormal", "empirical"
- Valid categories: "language", "multimodal", "reasoning"
- Valid SLO classes: "", "realtime", "interactive", "batch"

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run TestLoadWorkloadSpec -v && go test ./sim/workload/... -run TestWorkloadSpec -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`
Expected: No new issues

**Step 6: Commit**

```bash
git add sim/workload/spec.go sim/workload/spec_test.go
git commit -m "feat(workload): add WorkloadSpec types with strict YAML loading and validation (BC-5, NC-2, EC-1, EC-2)

- Add WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec and supporting types
- LoadWorkloadSpec with KnownFields(true) strict parsing
- Validate() checks process names, distribution types, parameter ranges
- Follow LoadPolicyBundle pattern from sim/bundle.go

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Arrival Samplers

**Contracts Implemented:** BC-3

**Files:**
- Create: `sim/workload/arrival.go`
- Test: `sim/workload/arrival_test.go`

**Step 1: Write failing test for arrival samplers**

```go
package workload

import (
	"math"
	"math/rand"
	"testing"
)

func TestPoissonSampler_MeanIAT_MatchesRate(t *testing.T) {
	// GIVEN a Poisson sampler at 10 req/sec (0.00001 req/µs)
	rng := rand.New(rand.NewSource(42))
	sampler := NewArrivalSampler(ArrivalSpec{Process: "poisson"}, 10.0/1e6)

	// WHEN 10000 IATs are sampled
	n := 10000
	sum := int64(0)
	for i := 0; i < n; i++ {
		sum += sampler.SampleIAT(rng)
	}
	meanIAT := float64(sum) / float64(n)

	// THEN mean IAT ≈ 1/rate = 100000 µs (within 5%)
	expected := 1e6 / 10.0
	if math.Abs(meanIAT-expected)/expected > 0.05 {
		t.Errorf("mean IAT = %.0f µs, want ≈ %.0f µs (within 5%%)", meanIAT, expected)
	}
}

func TestGammaSampler_HighCV_ProducesBurstierArrivals(t *testing.T) {
	// GIVEN a Gamma sampler with CV=3.5 and a Poisson sampler at same rate
	rng1 := rand.New(rand.NewSource(42))
	rng2 := rand.New(rand.NewSource(42))
	cv := 3.5
	rate := 10.0 / 1e6 // 10 req/sec
	gamma := NewArrivalSampler(ArrivalSpec{Process: "gamma", CV: &cv}, rate)
	poisson := NewArrivalSampler(ArrivalSpec{Process: "poisson"}, rate)

	// WHEN 10000 IATs sampled from each
	n := 10000
	gammaIATs := make([]float64, n)
	poissonIATs := make([]float64, n)
	for i := 0; i < n; i++ {
		gammaIATs[i] = float64(gamma.SampleIAT(rng1))
		poissonIATs[i] = float64(poisson.SampleIAT(rng2))
	}

	// THEN Gamma CV > 2.0 and Poisson CV ≈ 1.0
	gammaCV := coefficientOfVariation(gammaIATs)
	poissonCV := coefficientOfVariation(poissonIATs)
	if gammaCV < 2.0 {
		t.Errorf("gamma CV = %.2f, want > 2.0", gammaCV)
	}
	if poissonCV < 0.8 || poissonCV > 1.2 {
		t.Errorf("poisson CV = %.2f, want ≈ 1.0", poissonCV)
	}
}

func TestGammaSampler_MeanAndVariance_MatchTheoretical(t *testing.T) {
	// Tighter test: verify both mean and variance, not just CV
	rng := rand.New(rand.NewSource(42))
	cv := 2.0
	rate := 10.0 / 1e6 // 10 req/sec
	sampler := NewArrivalSampler(ArrivalSpec{Process: "gamma", CV: &cv}, rate)

	n := 50000
	vals := make([]float64, n)
	for i := 0; i < n; i++ {
		vals[i] = float64(sampler.SampleIAT(rng))
	}
	// Theoretical: mean = 1/rate = 100000 µs, variance = mean² * CV² = 100000² * 4
	mean, variance := meanAndVariance(vals)
	expectedMean := 1e6 / 10.0
	expectedVar := expectedMean * expectedMean * cv * cv
	if math.Abs(mean-expectedMean)/expectedMean > 0.05 {
		t.Errorf("gamma mean = %.0f, want ≈ %.0f (within 5%%)", mean, expectedMean)
	}
	if math.Abs(variance-expectedVar)/expectedVar > 0.15 {
		t.Errorf("gamma variance = %.0f, want ≈ %.0f (within 15%%)", variance, expectedVar)
	}
}

func meanAndVariance(vals []float64) (float64, float64) {
	n := float64(len(vals))
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	mean := sum / n
	sumSq := 0.0
	for _, v := range vals {
		d := v - mean
		sumSq += d * d
	}
	return mean, sumSq / n
}

func TestWeibullSampler_MeanIAT_MatchesRate(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	cv := 1.5
	rate := 10.0 / 1e6
	sampler := NewArrivalSampler(ArrivalSpec{Process: "weibull", CV: &cv}, rate)

	n := 10000
	sum := int64(0)
	for i := 0; i < n; i++ {
		sum += sampler.SampleIAT(rng)
	}
	meanIAT := float64(sum) / float64(n)
	expected := 1e6 / 10.0
	// Weibull mean is scale * Gamma(1 + 1/shape) — within 10% is acceptable
	if math.Abs(meanIAT-expected)/expected > 0.10 {
		t.Errorf("weibull mean IAT = %.0f µs, want ≈ %.0f µs (within 10%%)", meanIAT, expected)
	}
}

// coefficientOfVariation computes std_dev / mean.
func coefficientOfVariation(vals []float64) float64 {
	n := float64(len(vals))
	sum := 0.0
	for _, v := range vals {
		sum += v
	}
	mean := sum / n
	sumSq := 0.0
	for _, v := range vals {
		d := v - mean
		sumSq += d * d
	}
	stdDev := math.Sqrt(sumSq / n)
	return stdDev / mean
}
```

**Step 2: Run test to verify it fails**

Run: `go test ./sim/workload/... -run TestPoissonSampler -v`
Expected: FAIL

**Step 3: Implement arrival samplers**

Create `sim/workload/arrival.go` with:
- `ArrivalSampler` interface: `SampleIAT(rng *rand.Rand) int64`
- `PoissonSampler`: `int64(rng.ExpFloat64() / rateMicros)`
- `GammaSampler`: Use `gonum.org/v1/gonum/stat/distuv.Gamma{Alpha: shape, Beta: 1/scale}` (gonum is already a dependency for calibration; Marsaglia-Tsang is 30-40 lines to hand-roll correctly vs 3 lines with gonum). Pass `rng` as the `Src` field — `*math/rand.Rand` implements `rand.Source` (has `Int63()` and `Seed()` methods). **Verify BC-1 determinism** with a test: two GammaSamplers from same seed must produce identical sequences. Guard: if shape < 0.01 (CV > 10), fall back to PoissonSampler and log warning.
- `WeibullSampler`: inverse CDF `scale * math.Pow(-math.Log(1-rng.Float64()), 1/shape)`
- `NewArrivalSampler(spec ArrivalSpec, ratePerMicrosecond float64) ArrivalSampler` factory
- Weibull shape derivation from CV: use **bisection method** on `CV² = Γ(1+2/k)/Γ(1+1/k)² - 1` with `math.Gamma`. Convergence tolerance: `|CV_computed - CV_target| < 0.001`. Range: k ∈ [0.1, 100].
- Gamma shape/scale from CV and rate: shape = 1/CV², scale = (CV²/rate) × 1e6 µs

**Step 4: Run test to verify it passes**

Run: `go test ./sim/workload/... -run "TestPoissonSampler|TestGammaSampler|TestWeibullSampler" -v`
Expected: PASS

**Step 5: Run lint check**

Run: `golangci-lint run ./sim/workload/...`

**Step 6: Commit**

```bash
git add sim/workload/arrival.go sim/workload/arrival_test.go
git commit -m "feat(workload): add ArrivalSampler with Poisson, Gamma, Weibull (BC-3)

- PoissonSampler: exponential IATs (CV=1)
- GammaSampler: gonum distuv.Gamma with CV guard (shape < 0.01 falls back to Poisson)
- WeibullSampler: inverse CDF method, CV-parameterized via bisection
- Factory: NewArrivalSampler(spec, rate)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Length Distribution Samplers

**Contracts Implemented:** BC-4

**Files:**
- Create: `sim/workload/distribution.go`
- Test: `sim/workload/distribution_test.go`

**Step 1: Write failing test for distribution samplers**

```go
package workload

import (
	"math"
	"math/rand"
	"testing"
)

func TestGaussianSampler_MeanMatchesParam(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type:   "gaussian",
		Params: map[string]float64{"mean": 512, "std_dev": 128, "min": 10, "max": 4096},
	})
	if err != nil {
		t.Fatal(err)
	}
	n := 10000
	sum := 0
	for i := 0; i < n; i++ {
		sum += s.Sample(rng)
	}
	mean := float64(sum) / float64(n)
	if math.Abs(mean-512)/512 > 0.05 {
		t.Errorf("gaussian mean = %.1f, want ≈ 512 (within 5%%)", mean)
	}
}

func TestExponentialSampler_MeanMatchesParam(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type:   "exponential",
		Params: map[string]float64{"mean": 256},
	})
	if err != nil {
		t.Fatal(err)
	}
	n := 10000
	sum := 0
	for i := 0; i < n; i++ {
		sum += s.Sample(rng)
	}
	mean := float64(sum) / float64(n)
	if math.Abs(mean-256)/256 > 0.05 {
		t.Errorf("exponential mean = %.1f, want ≈ 256 (within 5%%)", mean)
	}
}

func TestEmpiricalPDFSampler_ReproducesDistribution(t *testing.T) {
	// GIVEN a simple empirical PDF: {10: 0.5, 20: 0.5}
	rng := rand.New(rand.NewSource(42))
	pdf := map[int]float64{10: 0.5, 20: 0.5}
	s := NewEmpiricalPDFSampler(pdf)

	// WHEN 10000 samples drawn
	n := 10000
	counts := make(map[int]int)
	for i := 0; i < n; i++ {
		v := s.Sample(rng)
		counts[v]++
	}

	// THEN each value appears ~50% of the time (within 5%)
	frac10 := float64(counts[10]) / float64(n)
	if math.Abs(frac10-0.5) > 0.05 {
		t.Errorf("P(10) = %.3f, want ≈ 0.5", frac10)
	}
}

func TestParetoLogNormalSampler_ProducesPositiveValues(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	s, err := NewLengthSampler(DistSpec{
		Type: "pareto_lognormal",
		Params: map[string]float64{
			"alpha": 1.5, "xm": 50, "mu": 5.5, "sigma": 1.2, "mix_weight": 0.3,
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 1000; i++ {
		v := s.Sample(rng)
		if v < 1 {
			t.Errorf("sample %d: got %d, want >= 1", i, v)
			break
		}
	}
}

func TestEmpiricalPDFSampler_SingleBin_AlwaysReturnsThatValue(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	pdf := map[int]float64{42: 1.0}
	s := NewEmpiricalPDFSampler(pdf)
	for i := 0; i < 100; i++ {
		if v := s.Sample(rng); v != 42 {
			t.Errorf("sample %d: got %d, want 42", i, v)
		}
	}
}

func TestEmpiricalPDFSampler_NonNormalized_NormalizesAutomatically(t *testing.T) {
	// GIVEN probabilities that sum to 2.0 (not 1.0)
	rng := rand.New(rand.NewSource(42))
	pdf := map[int]float64{10: 1.0, 20: 1.0}
	s := NewEmpiricalPDFSampler(pdf)
	// THEN samples are still ~50/50
	counts := make(map[int]int)
	for i := 0; i < 10000; i++ {
		counts[s.Sample(rng)]++
	}
	frac := float64(counts[10]) / 10000.0
	if frac < 0.45 || frac > 0.55 {
		t.Errorf("P(10) = %.3f, want ≈ 0.5 (non-normalized input should auto-normalize)", frac)
	}
}

func TestNewEmpiricalPDFSampler_NegativeProbability_ReturnsError(t *testing.T) {
	pdf := map[int]float64{10: 0.5, 20: -0.3}
	_, err := NewEmpiricalPDFSamplerValidated(pdf)
	if err == nil {
		t.Fatal("expected error for negative probability, got nil")
	}
}

func TestNewLengthSampler_EmptyEmpiricalPDF_ReturnsError(t *testing.T) {
	_, err := NewLengthSampler(DistSpec{Type: "empirical", Params: map[string]float64{}})
	if err == nil {
		t.Fatal("expected error for empty empirical PDF")
	}
}

func TestNewLengthSampler_InvalidType_ReturnsError(t *testing.T) {
	_, err := NewLengthSampler(DistSpec{Type: "unknown"})
	if err == nil {
		t.Fatal("expected error for unknown distribution type")
	}
}
```

**Step 2: Run test → fail**
**Step 3: Implement** `sim/workload/distribution.go` with LengthSampler interface, GaussianSampler, ExponentialSampler, ParetoLogNormalSampler, EmpiricalPDFSampler, and factory.
**Step 4: Run test → pass**
**Step 5: Lint**
**Step 6: Commit**

```bash
git add sim/workload/distribution.go sim/workload/distribution_test.go
git commit -m "feat(workload): add LengthSampler with Gaussian, Exponential, ParetoLogNormal, EmpiricalPDF (BC-4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Request Extensions + RNG Subsystem + CLI Integration

**Contracts Implemented:** BC-5

**Files:**
- Modify: `sim/request.go`
- Modify: `sim/rng.go`
- Modify: `cmd/root.go`
- Modify: `sim/cluster/cluster.go`
- Modify: `sim/cluster/workload.go`
- Test: `sim/workload/integration_test.go`

**Step 1: Write failing test**

```go
package workload

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

func TestGenerateRequests_SingleClient_ProducesRequests(t *testing.T) {
	// GIVEN a minimal workload spec with one client
	spec := &WorkloadSpec{
		Version:       "1",
		Seed:          42,
		Category:      "language",
		AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID:           "c1",
			TenantID:     "t1",
			SLOClass:     "batch",
			RateFraction: 1.0,
			Arrival:      ArrivalSpec{Process: "poisson"},
			InputDist:    DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist:   DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(1e6) // 1 second

	// WHEN requests are generated
	requests, err := GenerateRequests(spec, horizon)

	// THEN
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected at least one request")
	}
	// Requests should have TenantID and SLOClass set
	for i, req := range requests {
		if req.TenantID != "t1" {
			t.Errorf("request %d: TenantID = %q, want %q", i, req.TenantID, "t1")
			break
		}
		if req.SLOClass != "batch" {
			t.Errorf("request %d: SLOClass = %q, want %q", i, req.SLOClass, "batch")
			break
		}
		if len(req.InputTokens) == 0 || len(req.OutputTokens) == 0 {
			t.Errorf("request %d: empty token slices", i)
			break
		}
	}
	// Verify sorted by arrival time
	for i := 1; i < len(requests); i++ {
		if requests[i].ArrivalTime < requests[i-1].ArrivalTime {
			t.Errorf("requests not sorted: [%d].ArrivalTime=%d < [%d].ArrivalTime=%d",
				i, requests[i].ArrivalTime, i-1, requests[i-1].ArrivalTime)
			break
		}
	}
}

func TestGenerateRequests_Deterministic_SameSeedSameOutput(t *testing.T) {
	// BC-1: same seed + spec = identical requests
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		Clients: []ClientSpec{{
			ID: "c1", RateFraction: 1.0,
			Arrival:   ArrivalSpec{Process: "poisson"},
			InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
			OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}},
		}},
	}
	horizon := int64(1e6)

	r1, _ := GenerateRequests(spec, horizon)
	r2, _ := GenerateRequests(spec, horizon)

	if len(r1) != len(r2) {
		t.Fatalf("different counts: %d vs %d", len(r1), len(r2))
	}
	for i := range r1 {
		if r1[i].ArrivalTime != r2[i].ArrivalTime {
			t.Errorf("request %d: arrival %d vs %d", i, r1[i].ArrivalTime, r2[i].ArrivalTime)
			break
		}
		if len(r1[i].InputTokens) != len(r2[i].InputTokens) {
			t.Errorf("request %d: input len %d vs %d", i, len(r1[i].InputTokens), len(r2[i].InputTokens))
			break
		}
	}
}

func TestGenerateRequests_TwoClients_RateProportional(t *testing.T) {
	// BC-2: client rate fractions produce proportional request counts
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "a", TenantID: "a", RateFraction: 0.7,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "b", TenantID: "b", RateFraction: 0.3,
				Arrival:   ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	requests, err := GenerateRequests(spec, 10e6) // 10 seconds for statistical stability
	if err != nil {
		t.Fatal(err)
	}
	countA := 0
	for _, r := range requests {
		if r.TenantID == "a" {
			countA++
		}
	}
	fracA := float64(countA) / float64(len(requests))
	if math.Abs(fracA-0.7) > 0.05 {
		t.Errorf("client A fraction = %.3f, want ≈ 0.7 (within 5%%)", fracA)
	}
}

func TestGenerateRequests_RateFractionNormalization_SumsToOne(t *testing.T) {
	// Rate fractions that don't sum to 1.0 should be auto-normalized
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 100.0,
		Clients: []ClientSpec{
			{ID: "a", TenantID: "a", RateFraction: 7.0, // not normalized
				Arrival: ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
			{ID: "b", TenantID: "b", RateFraction: 3.0, // not normalized
				Arrival: ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 100, "std_dev": 20, "min": 10, "max": 500}},
				OutputDist: DistSpec{Type: "exponential", Params: map[string]float64{"mean": 50}}},
		},
	}
	requests, err := GenerateRequests(spec, 10e6)
	if err != nil {
		t.Fatal(err)
	}
	// Even with raw fractions 7:3, normalization should produce 70%/30%
	countA := 0
	for _, r := range requests {
		if r.TenantID == "a" {
			countA++
		}
	}
	fracA := float64(countA) / float64(len(requests))
	if math.Abs(fracA-0.7) > 0.05 {
		t.Errorf("normalized fraction A = %.3f, want ≈ 0.7", fracA)
	}
}

// Verify new Request fields don't break existing usage
func TestRequestNewFields_ZeroValueDefault(t *testing.T) {
	req := &sim.Request{ID: "test", State: "queued"}
	if req.TenantID != "" || req.SLOClass != "" || req.SessionID != "" {
		t.Error("new fields should have zero-value defaults")
	}
}
```

**Step 2: Run test → fail**

**Step 3: Implement**

1. Add fields to `sim/request.go`: TenantID, SLOClass, Streaming, SessionID, RoundIndex, TextTokenCount, ImageTokenCount, AudioTokenCount, VideoTokenCount, ReasonRatio
2. Add `SubsystemWorkloadGen = "workload-gen"` to `sim/rng.go` (intentionally separate from existing `SubsystemWorkload` — the legacy constant uses master seed directly for backward compat; new constant gets an isolated XOR-derived stream. Existing code at `sim/cluster/workload.go:27` continues using `SubsystemWorkload` unchanged.)
3. Create `sim/workload/client.go` with rate normalization and prefix group management
4. Create `sim/workload/generator.go` with GenerateRequests pipeline
5. Add `--workload-spec` flag to `cmd/root.go` (`workloadSpecPath string` variable). **Precedence rule:** if `--workload-spec` is set, it takes precedence over `--workload`. Check `workloadSpecPath != ""` BEFORE the existing `workloadType` logic. Update help text: `"Workload type (...). Use --workload-spec for ServeGen-style YAML config (overrides --workload)."`
6. When `workloadSpecPath` is set: load spec via `workload.LoadWorkloadSpec`, validate, call `workload.GenerateRequests`, pass resulting `[]*sim.Request` to ClusterSimulator
7. Extend `sim/cluster/cluster.go`: add `preGeneratedRequests []*sim.Request` field to `ClusterSimulator` struct. Keep constructor signature unchanged. Add setter: `func (c *ClusterSimulator) SetPreGeneratedRequests(reqs []*sim.Request)`. Update validation: `if workload == nil && tracesPath == "" && preGeneratedRequests == nil { panic(...) }` — BUT cmd/ calls setter AFTER construction, so remove this panic and let `generateRequests()` handle the empty case.
8. Extend `sim/cluster/workload.go` `generateRequests()`: check `len(c.preGeneratedRequests) > 0` FIRST (highest precedence), then existing traces path, then distribution. Precedence: preGenerated → traces → distribution.

**Step 4: Run test → pass** (also run `go test ./...` to verify BC-5 — golden tests unchanged)
**Step 5: Lint**
**Step 6: Commit**

```bash
git add sim/request.go sim/rng.go sim/workload/client.go sim/workload/generator.go \
  sim/workload/integration_test.go cmd/root.go sim/cluster/cluster.go sim/cluster/workload.go
git commit -m "feat(workload): add GenerateRequests pipeline with client decomposition and CLI integration (BC-1, BC-2, BC-5)

- Add TenantID, SLOClass, and other metadata fields to Request
- Add SubsystemWorkloadGen RNG constant
- GenerateRequests: per-client arrival + length sampling, merge + sort
- --workload-spec CLI flag with cluster integration
- Golden dataset tests pass unchanged (backward compatible)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: ServeGen Native Data Loading

**Contracts Implemented:** BC-4 (empirical PDF from real data)

**Files:**
- Modify: `sim/workload/generator.go`
- Modify: `sim/workload/spec.go` (ServeGenDataSpec type)
- Test: `sim/workload/servegen_test.go`

**Step 1: Write failing test**

```go
func TestServeGenDataLoading_SyntheticDataset_ProducesClients(t *testing.T) {
	// GIVEN a synthetic ServeGen data directory
	dir := t.TempDir()
	// Create chunk-0-trace.csv: start_time, rate, cv, pattern, param1, param2
	traceCSV := "0,1.0,2.5,Gamma,0.16,6.25\n600,0.5,1.0,Weibull,1.0,2000000\n"
	os.WriteFile(filepath.Join(dir, "chunk-0-trace.csv"), []byte(traceCSV), 0644)
	// Create chunk-0-dataset.json: keyed by window start time, each has input_tokens and output_tokens as Python dict strings
	datasetJSON := `{"0": {"input_tokens": "{100: 0.5, 200: 0.5}", "output_tokens": "{50: 0.7, 100: 0.3}"}}`
	os.WriteFile(filepath.Join(dir, "chunk-0-dataset.json"), []byte(datasetJSON), 0644)

	// WHEN loaded as a WorkloadSpec
	spec := &WorkloadSpec{
		Version: "1", Seed: 42, Category: "language", AggregateRate: 10.0,
		ServeGenData: &ServeGenDataSpec{Path: dir},
	}
	requests, err := GenerateRequests(spec, 1e6)

	// THEN requests are generated from the empirical distributions
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(requests) == 0 {
		t.Fatal("expected requests from ServeGen data")
	}
	// Verify input token lengths come from the empirical PDF (100 or ~200 range)
	for _, req := range requests[:min(10, len(requests))] {
		l := len(req.InputTokens)
		if l < 50 || l > 300 {
			t.Errorf("input length %d outside expected range [50, 300] from empirical PDF", l)
		}
	}
}

func TestParseServeGenPDF_PythonDictString_ConvertsCorrectly(t *testing.T) {
	input := "{100: 0.5, 200: 0.3, 300: 0.2}"
	pdf, err := parseServeGenPDF(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(pdf) != 3 {
		t.Fatalf("expected 3 bins, got %d", len(pdf))
	}
	if pdf[100] != 0.5 || pdf[200] != 0.3 || pdf[300] != 0.2 {
		t.Errorf("unexpected PDF values: %v", pdf)
	}
}

func TestParseServeGenPDF_EdgeCases(t *testing.T) {
	tests := []struct {
		name  string
		input string
		bins  int
	}{
		{"scientific notation", "{100: 3e-4, 200: 9.997e-1}", 2},
		{"extra whitespace", "{ 100 : 0.5 , 200 : 0.5 }", 2},
		{"trailing comma", "{100: 0.5, 200: 0.5,}", 2},
		{"large dict", func() string {
			// Simulate 1000-bin PDF
			s := "{"
			for i := 0; i < 1000; i++ {
				if i > 0 { s += ", " }
				s += fmt.Sprintf("%d: 0.001", i)
			}
			return s + "}"
		}(), 1000},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pdf, err := parseServeGenPDF(tt.input)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(pdf) != tt.bins {
				t.Errorf("got %d bins, want %d", len(pdf), tt.bins)
			}
		})
	}
}

func TestParseServeGenPDF_EmptyDict_ReturnsError(t *testing.T) {
	_, err := parseServeGenPDF("{}")
	if err == nil {
		t.Fatal("expected error for empty dict")
	}
}
```

**Step 2: Run test → fail**
**Step 3: Implement** ServeGen data loading in `sim/workload/generator.go`:
- `parseServeGenPDF(s string) (map[int]float64, error)` — parse Python dict string `{key: val, ...}` into Go map
- `loadServeGenData(path string, spec *WorkloadSpec) error` — scan directory for `chunk-*-trace.csv` and `chunk-*-dataset.json`, create ClientSpec per chunk with arrival params from trace CSV and EmpiricalPDF from dataset JSON
- Handle optional `span_start`/`span_end` time window filtering
**Step 4: Run test → pass**
**Step 5: Lint**
**Step 6: Commit**

```bash
git commit -m "feat(workload): add native ServeGen data file loading (BC-4)

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

#### Section 2: Observe-Predict-Calibrate Pipeline

### Task 6: Trace v2 Types + Export

**Contracts Implemented:** BC-6

**Files:**
- Create: `sim/workload/tracev2.go`
- Test: `sim/workload/tracev2_test.go`

**Step 1: Write failing test for trace round-trip**

```go
func TestTraceV2_RoundTrip_PreservesAllFields(t *testing.T) {
	// GIVEN a trace header and records
	header := &TraceHeader{
		Version: 2, TimeUnit: "microseconds", Mode: "generated",
		WarmUpRequests: 5, WorkloadSpec: "test.yaml",
	}
	records := []TraceRecord{
		{RequestID: 0, ClientID: "c1", TenantID: "t1", SLOClass: "batch",
			InputTokens: 512, OutputTokens: 128, ArrivalTimeUs: 0},
		{RequestID: 1, ClientID: "c2", TenantID: "t2", SLOClass: "realtime",
			InputTokens: 256, OutputTokens: 64, ArrivalTimeUs: 100000},
	}

	// WHEN exported and loaded back
	dir := t.TempDir()
	headerPath := filepath.Join(dir, "header.yaml")
	dataPath := filepath.Join(dir, "data.csv")
	if err := ExportTraceV2(header, records, headerPath, dataPath); err != nil {
		t.Fatal(err)
	}
	loaded, err := LoadTraceV2(headerPath, dataPath)
	if err != nil {
		t.Fatal(err)
	}

	// THEN all fields match
	if loaded.Header.Version != 2 {
		t.Errorf("version = %d, want 2", loaded.Header.Version)
	}
	if len(loaded.Records) != 2 {
		t.Fatalf("records = %d, want 2", len(loaded.Records))
	}
	if loaded.Records[0].InputTokens != 512 {
		t.Errorf("record 0 input_tokens = %d, want 512", loaded.Records[0].InputTokens)
	}
}
```

**Step 2-6: Standard TDD cycle + commit**

---

### Task 7: Real Mode HTTP Client (Mock-Tested)

**Contracts Implemented:** EC-3

**Files:**
- Create: `cmd/observe.go`
- Test: `cmd/observe_test.go`

**Step 1: Write failing test with httptest mock server**

```go
func TestRealClient_NonStreaming_RecordsTokenCounts(t *testing.T) {
	// GIVEN a mock OpenAI-compatible server
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{{"text": "hello world"}},
			"usage":   map[string]interface{}{"prompt_tokens": 100, "completion_tokens": 50},
		})
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		InputTokens: 100, Streaming: false,
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.OutputTokens != 50 {
		t.Errorf("output_tokens = %d, want 50", record.OutputTokens)
	}
	if record.Status != "ok" {
		t.Errorf("status = %q, want ok", record.Status)
	}
}
```

Also add streaming test:

```go
func TestRealClient_Streaming_RecordsFirstAndLastChunkTime(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		flusher, _ := w.(http.Flusher)
		w.Header().Set("Content-Type", "text/event-stream")
		for i := 0; i < 5; i++ {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":\"tok\"}}]}\n\n")
			flusher.Flush()
			time.Sleep(10 * time.Millisecond)
		}
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{}}],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":5}}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	client := NewRealClient(server.URL, "", "test-model", "vllm")
	record, err := client.Send(context.Background(), &PendingRequest{
		InputTokens: 100, Streaming: true,
	})
	if err != nil {
		t.Fatal(err)
	}
	if record.OutputTokens != 5 {
		t.Errorf("output_tokens = %d, want 5", record.OutputTokens)
	}
	if record.NumChunks < 5 {
		t.Errorf("num_chunks = %d, want >= 5", record.NumChunks)
	}
	if record.FirstChunkTimeUs == 0 {
		t.Error("first_chunk_time not recorded")
	}
	if record.LastChunkTimeUs <= record.FirstChunkTimeUs {
		t.Error("last_chunk_time should be > first_chunk_time for streaming")
	}
}
```

**Step 2-6: TDD cycle. Implement RealClient, LoadGenerator (goroutine-based dispatch), Recorder, streaming SSE parsing. Add CLI flags to cmd/root.go: `--real-mode` (bool), `--server-url` (string), `--server-type` (string: "vllm", "sglang", "tgi"), `--server-config` (string: path to server YAML), `--api-type` (string: "chat" or "completion"), `--api-key` (string), `--trace-output` (string: directory path).**

---

### Task 8: Trace v2 Loading + Sim Replay

**Contracts Implemented:** BC-6

**Files:**
- Modify: `sim/workload/tracev2.go` (add LoadTraceV2Requests)
- Modify: `sim/workload/generator.go` (synthetic token generation from trace)
- Test: `sim/workload/replay_test.go`

**Step 1: Test** that loading a trace CSV produces correct `[]*sim.Request` with synthetic token IDs and prefix group sharing.
**Step 2-6: TDD cycle + commit**

---

### Task 9: Calibration Framework + Orchestration

**Contracts Implemented:** BC-7, BC-11, BC-12

**Files:**
- Create: `sim/workload/calibrate.go`
- Modify: `go.mod` (add gonum, montanaflynn/stats)
- Modify: `cmd/root.go` (add `calibrate` subcommand)
- Test: `sim/workload/calibrate_test.go`

**Step 1: Write failing tests**

```go
// --- Statistics layer tests ---

func TestComputeCalibration_PerfectMatch_ZeroMAPE(t *testing.T) {
	real := []float64{100, 200, 300, 400, 500}
	sim := []float64{100, 200, 300, 400, 500}

	report, err := ComputeCalibration(real, sim, "ttft")
	if err != nil {
		t.Fatal(err)
	}
	if report.MAPE != 0.0 {
		t.Errorf("MAPE = %f, want 0.0", report.MAPE)
	}
	if report.PearsonR != 1.0 {
		t.Errorf("PearsonR = %f, want 1.0", report.PearsonR)
	}
}

func TestComputeCalibration_KnownError_CorrectMAPE(t *testing.T) {
	real := []float64{100, 200, 300}
	sim := []float64{110, 220, 330}

	report, err := ComputeCalibration(real, sim, "e2e")
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(report.MAPE-0.10) > 0.001 {
		t.Errorf("MAPE = %f, want 0.10", report.MAPE)
	}
}

// --- Orchestration layer tests (BC-11) ---

func TestPrepareCalibrationPairs_MatchesByRequestID(t *testing.T) {
	// GIVEN real trace records and sim results with same request IDs but different order
	realRecords := []TraceRecord{
		{RequestID: 0, ArrivalTimeUs: 0, FirstChunkTimeUs: 500, LastChunkTimeUs: 1000, SendTimeUs: 10},
		{RequestID: 1, ArrivalTimeUs: 100000, FirstChunkTimeUs: 100800, LastChunkTimeUs: 101500, SendTimeUs: 100010},
	}
	simResults := []SimResult{
		{RequestID: 1, TTFT: 750, E2E: 1400},  // note: out of order
		{RequestID: 0, TTFT: 450, E2E: 950},
	}

	// WHEN pairs are prepared (no network adjustment, no warm-up)
	pairs, err := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
		WarmUpRequests: 0,
		NetworkRTTUs:   0,
	})
	if err != nil {
		t.Fatal(err)
	}

	// THEN pairs are matched by request_id, not position
	if len(pairs.TTFT.Real) != 2 {
		t.Fatalf("expected 2 pairs, got %d", len(pairs.TTFT.Real))
	}
	// Request 0: real TTFT = first_chunk - send = 500 - 10 = 490
	// Request 0: sim TTFT = 450
	if pairs.TTFT.Real[0] != 490 || pairs.TTFT.Sim[0] != 450 {
		t.Errorf("request 0: real=%.0f sim=%.0f, want 490/450", pairs.TTFT.Real[0], pairs.TTFT.Sim[0])
	}
}

func TestPrepareCalibrationPairs_AppliesNetworkAdjustment(t *testing.T) {
	// GIVEN sim TTFT of 500µs and network RTT of 5000µs (5ms)
	realRecords := []TraceRecord{
		{RequestID: 0, FirstChunkTimeUs: 6000, SendTimeUs: 100, LastChunkTimeUs: 7000},
	}
	simResults := []SimResult{
		{RequestID: 0, TTFT: 500, E2E: 900},
	}

	// WHEN network adjustment is applied
	pairs, _ := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
		NetworkRTTUs: 5000,
	})

	// THEN sim TTFT = 500 + 5000 = 5500 (client-perspective)
	if pairs.TTFT.Sim[0] != 5500 {
		t.Errorf("sim TTFT with network = %.0f, want 5500", pairs.TTFT.Sim[0])
	}
	// Real TTFT = 6000 - 100 = 5900
	if pairs.TTFT.Real[0] != 5900 {
		t.Errorf("real TTFT = %.0f, want 5900", pairs.TTFT.Real[0])
	}
}

func TestPrepareCalibrationPairs_ExcludesWarmUp(t *testing.T) {
	// GIVEN 5 requests with warm_up_requests=2
	realRecords := make([]TraceRecord, 5)
	simResults := make([]SimResult, 5)
	for i := 0; i < 5; i++ {
		realRecords[i] = TraceRecord{RequestID: i, FirstChunkTimeUs: int64(i*1000 + 500), SendTimeUs: int64(i * 1000), LastChunkTimeUs: int64(i*1000 + 1000)}
		simResults[i] = SimResult{RequestID: i, TTFT: 450, E2E: 900}
	}

	pairs, _ := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{
		WarmUpRequests: 2,
	})

	// THEN only requests 2,3,4 are included (warm-up 0,1 excluded)
	if len(pairs.TTFT.Real) != 3 {
		t.Errorf("expected 3 pairs after warm-up exclusion, got %d", len(pairs.TTFT.Real))
	}
}

func TestPrepareCalibrationPairs_UnmatchedRequests_ReportsCount(t *testing.T) {
	// GIVEN 3 real records but only 2 sim results (request 2 missing from sim)
	realRecords := []TraceRecord{
		{RequestID: 0, FirstChunkTimeUs: 500, SendTimeUs: 0, LastChunkTimeUs: 1000},
		{RequestID: 1, FirstChunkTimeUs: 1500, SendTimeUs: 1000, LastChunkTimeUs: 2000},
		{RequestID: 2, FirstChunkTimeUs: 2500, SendTimeUs: 2000, LastChunkTimeUs: 3000},
	}
	simResults := []SimResult{
		{RequestID: 0, TTFT: 450, E2E: 900},
		{RequestID: 1, TTFT: 480, E2E: 950},
		// request 2 missing — e.g., sim dropped it due to admission policy
	}

	pairs, _ := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})

	if pairs.MatchedCount != 2 {
		t.Errorf("matched = %d, want 2", pairs.MatchedCount)
	}
	if pairs.UnmatchedReal != 1 {
		t.Errorf("unmatched real = %d, want 1", pairs.UnmatchedReal)
	}
}

func TestPrepareCalibrationPairs_DetectsTokenMismatch(t *testing.T) {
	// GIVEN real reports 512 input tokens but sim used 500
	realRecords := []TraceRecord{
		{RequestID: 0, InputTokens: 512, OutputTokens: 128, FirstChunkTimeUs: 500, SendTimeUs: 0, LastChunkTimeUs: 1000},
	}
	simResults := []SimResult{
		{RequestID: 0, TTFT: 450, E2E: 900, InputTokens: 500, OutputTokens: 128},
	}

	pairs, _ := PrepareCalibrationPairs(realRecords, simResults, &CalibrationConfig{})

	// THEN token mismatch is counted
	if pairs.TokenMismatchCount != 1 {
		t.Errorf("token mismatch count = %d, want 1", pairs.TokenMismatchCount)
	}
}
```

```go
// BC-12: Verify CalibrationReport includes all required annotations
func TestBuildCalibrationReport_IncludesAllAnnotations(t *testing.T) {
	pairs := &CalibrationPairs{
		TTFT:               LatencyPair{Real: []float64{100, 200}, Sim: []float64{110, 210}},
		E2E:                LatencyPair{Real: []float64{500, 600}, Sim: []float64{520, 630}},
		TokenMismatchCount: 1,
		MatchedCount:       2,
		ExcludedWarmUp:     3,
	}
	report, err := BuildCalibrationReport(pairs, &ConfigMatchInfo{
		Matched:   []string{"max_num_seqs=256"},
		Defaulted: []string{"block_size (not in trace header)"},
	})
	if err != nil {
		t.Fatal(err)
	}
	// BC-12a: known limitations always present
	if len(report.KnownLimitations) == 0 {
		t.Error("KnownLimitations must not be empty")
	}
	// BC-12b: config match populated
	if len(report.ConfigMatch.Matched) != 1 || len(report.ConfigMatch.Defaulted) != 1 {
		t.Error("ConfigMatch not populated correctly")
	}
	// BC-12c: token mismatch reported
	if report.TraceInfo.TokenMismatches != 1 {
		t.Errorf("TokenMismatches = %d, want 1", report.TraceInfo.TokenMismatches)
	}
	// Verify metrics computed
	if report.Metrics["ttft"] == nil || report.Metrics["e2e"] == nil {
		t.Error("expected TTFT and E2E metric comparisons in report")
	}
}
```

**Step 2-6: TDD cycle. Implement two layers in `sim/workload/calibrate.go`:**

**Layer 1 — Statistics (`ComputeCalibration`):**
- MAPE, Pearson r, per-percentile errors, quality rating
- Guard edge cases: empty slices return error; MAPE skips requests where real==0; Pearson r returns 0 for N<3
- Add gonum and montanaflynn/stats to go.mod

**Layer 2 — Orchestration (`PrepareCalibrationPairs`):**
```go
// CalibrationConfig holds normalization parameters for preparing calibration pairs.
type CalibrationConfig struct {
    WarmUpRequests int     // Exclude first N requests
    NetworkRTTUs   int64   // RTT in microseconds to add to sim latencies
    BandwidthMbps  float64 // For upload/download delay (0 = infinite)
}

// SimResult holds per-request sim output for calibration matching.
// All latencies are server-side (no network), in microseconds.
// PrepareCalibrationPairs adds network adjustment before comparing with real.
type SimResult struct {
    RequestID    int
    TTFT         float64 // Server-side: FirstTokenTime - ArrivalTime (µs)
    E2E          float64 // Server-side: CompletionTime - ArrivalTime (µs)
    InputTokens  int
    OutputTokens int
}
// Real-side equivalents (computed in PrepareCalibrationPairs):
//   Real TTFT = FirstChunkTimeUs - SendTimeUs (client-side, includes network)
//   Real E2E  = LastChunkTimeUs - SendTimeUs (client-side, includes network)
// Both are "time from request dispatch to event" — PrepareCalibrationPairs
// normalizes sim to client-perspective by adding RTT + bandwidth delays.

// CalibrationPairs holds matched, normalized real-vs-sim latency vectors.
type CalibrationPairs struct {
    TTFT               LatencyPair
    E2E                LatencyPair
    TokenMismatchCount int
    ExcludedWarmUp     int
    MatchedCount       int
    UnmatchedReal      int // Real requests with no sim match
    UnmatchedSim       int // Sim requests with no real match
}

type LatencyPair struct {
    Real []float64
    Sim  []float64
}

// PrepareCalibrationPairs matches real trace records with sim results,
// applies network normalization, excludes warm-up, and detects token mismatches.
//
// Orchestration steps:
// 1. Index sim results by RequestID
// 2. For each real record with RequestID >= WarmUpRequests:
//    a. Find matching sim result by RequestID
//    b. Compute real TTFT = FirstChunkTimeUs - SendTimeUs
//    c. Compute real E2E = LastChunkTimeUs - SendTimeUs
//    d. Compute sim client TTFT = sim.TTFT + NetworkRTTUs + uploadDelay
//    e. Compute sim client E2E = sim.E2E + NetworkRTTUs + uploadDelay + downloadDelay
//    f. Check token count match (real vs sim InputTokens/OutputTokens)
//    g. Append to paired vectors
// 3. Return CalibrationPairs with match statistics
func PrepareCalibrationPairs(
    realRecords []TraceRecord,
    simResults []SimResult,
    config *CalibrationConfig,
) (*CalibrationPairs, error)
```

**`calibrate` subcommand in `cmd/root.go`** (cobra subcommand, same pattern as existing `run` command — `rootCmd.AddCommand(calibrateCmd)`):**
1. Load trace v2 header + CSV → real records
2. Extract server config from trace header → override DeploymentConfig fields
3. Load workload spec (from trace header's workload_spec path) → generate requests
4. Run ClusterSimulator with those requests → collect per-request sim metrics
5. Extract SimResult per request from sim Metrics maps
6. Call PrepareCalibrationPairs with network config from trace header
7. Call ComputeCalibration for each metric (TTFT, E2E)
8. Build CalibrationReport with known limitations, config match, token mismatch count
9. Output as JSON

**CalibrationReport includes (BC-12):**
```go
type CalibrationReport struct {
    TraceInfo struct {
        NumRequests    int
        WarmUpExcluded int
        MatchedPairs   int
        TokenMismatches int
    }
    Metrics          map[string]*MetricComparison // "ttft", "e2e"
    ConfigMatch      ConfigMatchInfo
    KnownLimitations []string // Always populated
}

type ConfigMatchInfo struct {
    Matched   []string // e.g., ["max_num_seqs=256", "block_size=16"]
    Defaulted []string // e.g., ["gpu_memory_utilization (not in trace header)"]
}
```

---

#### Section 3: Workload Extensions

### Task 10: Multimodal Workload Generation

**Contracts Implemented:** BC-8

**Files:**
- Create: `sim/workload/multimodal.go`
- Test: `sim/workload/multimodal_test.go`

**Step 1: Test** that multimodal requests have correct token accounting (text + image + audio + video = total input). Include edge cases: zero images (text-only multimodal), zero audio, zero video. Verify that a request with all modality counts=0 is valid (pure text).
**Step 2-6: TDD cycle + commit**

---

### Task 11: Reasoning + Multi-Turn + Network Model

**Contracts Implemented:** BC-9

**Files:**
- Create: `sim/workload/reasoning.go`
- Create: `sim/workload/network.go`
- Test: `sim/workload/reasoning_test.go`, `sim/workload/network_test.go`

**Step 1: Tests** for:
- ReasonRatio in [0,1], multi-turn session IDs, context growth
- Network latency: RTT adds to TTFT, bandwidth affects upload delay

```go
func TestComputeClientLatency_RTTOnly_AddsToTTFT(t *testing.T) {
	net := &NetworkSpec{RTTMs: 10.0} // 10ms = 10000µs
	serverTTFT := float64(500) // 500µs
	clientTTFT := ComputeClientTTFT(serverTTFT, net, 100) // 100 input tokens
	// RTT only (no bandwidth limit): client_ttft = server_ttft + rtt_us
	if clientTTFT != 10500 {
		t.Errorf("client TTFT = %.0f, want 10500", clientTTFT)
	}
}

func TestComputeClientLatency_WithBandwidth_AddsUploadDelay(t *testing.T) {
	net := &NetworkSpec{RTTMs: 0, BandwidthMbps: 100} // 100 Mbps
	serverTTFT := float64(500)
	inputTokens := 1000 // 1000 tokens × 4 bytes = 4000 bytes = 32000 bits
	clientTTFT := ComputeClientTTFT(serverTTFT, net, inputTokens)
	// upload_delay = 32000 bits / 100e6 bps = 0.00032s = 320µs
	expectedUpload := float64(32000) / float64(100e6) * 1e6 // 320µs
	expected := 500 + expectedUpload
	if math.Abs(clientTTFT-expected) > 1 {
		t.Errorf("client TTFT = %.0f, want ≈ %.0f (with upload delay)", clientTTFT, expected)
	}
}

func TestComputeClientLatency_NilNetwork_NoAdjustment(t *testing.T) {
	clientTTFT := ComputeClientTTFT(500, nil, 100)
	if clientTTFT != 500 {
		t.Errorf("nil network should not adjust: got %.0f, want 500", clientTTFT)
	}
}
```

**Step 2-6: TDD cycle + commit**

---

### Task 12: Metrics Extensions + Scenarios + Documentation

**Contracts Implemented:** BC-10

**Files:**
- Modify: `sim/cluster/metrics.go`
- Create: `sim/workload/scenarios.go`
- Create: `examples/servegen-language.yaml`
- Modify: `CLAUDE.md`
- Test: `sim/cluster/metrics_slo_test.go`

**Step 1: Write failing test**

```go
func TestPerSLOClassDistribution_SegregatesCorrectly(t *testing.T) {
	// GIVEN metrics with mixed SLO classes
	aggregated := sim.NewMetrics()
	// ... populate with requests having SLOClass "realtime" and "batch"

	// WHEN per-SLO distributions are computed
	sloDistributions := ComputePerSLODistributions(aggregated)

	// THEN each class contains only its requests
	if sloDistributions["realtime"].TTFT.Count != 50 {
		t.Errorf("realtime TTFT count = %d, want 50", sloDistributions["realtime"].TTFT.Count)
	}
}

func TestJainFairnessIndex_EqualThroughput_ReturnsOne(t *testing.T) {
	throughputs := map[string]float64{"t1": 100, "t2": 100, "t3": 100}
	jfi := JainFairnessIndex(throughputs)
	if math.Abs(jfi-1.0) > 0.001 {
		t.Errorf("JFI = %f, want 1.0", jfi)
	}
}
```

**Step 2-6: TDD cycle. Add per-SLO distributions, SLOAttainment, JainFairnessIndex to metrics.go. Create scenario presets. Create example YAML. Update CLAUDE.md.**

---

### H) Test Strategy

| Contract | Task | Test Type | Test Name |
|----------|------|-----------|-----------|
| NC-2 | Task 1 | Unit | TestLoadWorkloadSpec_UnknownKey_ReturnsError |
| EC-1 | Task 1 | Unit | TestWorkloadSpec_Validate_InvalidArrivalProcess_ReturnsError |
| EC-2 | Task 1 | Unit | TestWorkloadSpec_Validate_EmptyClients_ReturnsError |
| BC-3 | Task 2 | Unit | TestGammaSampler_HighCV_ProducesBurstierArrivals |
| BC-4 | Task 3 | Unit | TestEmpiricalPDFSampler_ReproducesDistribution |
| BC-1 | Task 4 | Unit | TestGenerateRequests_Deterministic_SameSeedSameOutput |
| BC-2 | Task 4 | Unit | TestGenerateRequests_TwoClients_RateProportional, TestGenerateRequests_RateFractionNormalization |
| BC-5 | Task 4 | Golden | Existing golden dataset tests (unchanged) |
| BC-6 | Task 6,8 | Unit | TestTraceV2_RoundTrip_PreservesAllFields |
| EC-3 | Task 7 | Unit | TestRealClient_NonStreaming_RecordsTokenCounts |
| BC-7 | Task 9 | Unit | TestComputeCalibration_KnownError_CorrectMAPE |
| BC-11 | Task 9 | Unit | TestPrepareCalibrationPairs_MatchesByRequestID, _AppliesNetworkAdjustment, _ExcludesWarmUp |
| BC-12 | Task 9 | Unit | TestBuildCalibrationReport_IncludesAllAnnotations, TestPrepareCalibrationPairs_DetectsTokenMismatch |
| BC-8 | Task 10 | Unit | TestMultimodal_TokenAccounting |
| BC-9 | Task 11 | Unit | TestNetworkLatency_RTTAddsToTTFT |
| BC-10 | Task 12 | Unit | TestPerSLOClassDistribution_SegregatesCorrectly |

**Golden dataset:** No updates needed — existing tests verify BC-5 (backward compatibility).

### I) Risk Analysis

| Risk | Likelihood | Impact | Mitigation | Task |
|------|-----------|--------|------------|------|
| ServeGen data format parsing breaks on edge cases | Medium | Medium | Test with real ServeGen data files from `results-old/` | Task 5 |
| Gamma sampler numerical instability for very small shape | Low | Medium | Guard against shape < 0.01; fall back to Poisson | Task 2 |
| Real mode HTTP client timeout handling | Medium | Low | Mock-tested; manual e2e test later | Task 7 |
| Import cycle sim/workload ↔ sim/cluster | Low | High | Strict boundary: workload only imports sim/ | All tasks |
| gonum/stats dependency version conflicts | Low | Low | Pin specific versions in go.mod | Task 9 |

---

## Part 3: Quality Assurance

### J) Sanity Checklist

- [x] No unnecessary abstractions — each sampler is a simple struct
- [x] No feature creep — scope matches design doc exactly
- [x] No unexercised flags — `--workload-spec` exercises full pipeline
- [x] No partial implementations — every file is testable on commit
- [x] No breaking changes — all Request fields are additive zero-value defaults
- [x] No hidden global state — all state in WorkloadSpec or passed explicitly
- [x] All new code will pass golangci-lint
- [x] No shared test helpers duplicated locally
- [x] CLAUDE.md updated with new package, files, CLI flags
- [x] No stale references
- [x] Deviation log reviewed — 5 deviations justified
- [x] Each task produces working, testable code
- [x] Task dependencies correctly ordered (1→2→3→4→5, 4→6→7→8, 6+7→9, 4→10, 4→11, 4→12)
- [x] All contracts mapped to tasks
- [x] Golden dataset regeneration not needed (backward compatible)

---

## Appendix: File-Level Implementation Details

### File: `sim/workload/spec.go`

**Purpose:** All spec types + YAML loading + validation. Entry point for workload configuration.

**Key types:** WorkloadSpec, ClientSpec, ArrivalSpec, DistSpec, NetworkSpec, LifecycleSpec, ActiveWindow, MultimodalSpec, ReasoningSpec, MultiTurnSpec, ServeGenDataSpec

**Key functions:**
```go
func LoadWorkloadSpec(path string) (*WorkloadSpec, error)  // Strict YAML, KnownFields(true)
func (s *WorkloadSpec) Validate() error                     // All field validation
```

**Validation rules:**
- Version must be "1"
- AggregateRate must be > 0
- At least one client OR servegen_data specified
- Each client: valid arrival process, valid distribution types, rate_fraction > 0
- DistSpec params: non-negative, finite (NaN/Inf rejected via validateFloat)
- CV pointer: nil means "not set" (Poisson default for gamma/weibull). Weibull CV range: [0.01, 10.4] (outside this, bisection cannot converge — reject with descriptive error). Gamma CV range: (0, 10] (CV > 10 falls back to Poisson with warning).

### File: `sim/workload/arrival.go`

**Purpose:** Inter-arrival time sampling for each client.

**Interface:** `ArrivalSampler` with `SampleIAT(rng *rand.Rand) int64`

**Implementations:**
- `PoissonSampler{rateMicros float64}` — `int64(rng.ExpFloat64() / s.rateMicros)`
- `GammaSampler{dist distuv.Gamma}` — uses `gonum.org/v1/gonum/stat/distuv.Gamma{Alpha: shape, Beta: 1/scale, Src: rng}`. 3 lines vs 30-40 for hand-rolled Marsaglia-Tsang. Guard: shape < 0.01 falls back to Poisson.
- `WeibullSampler{shape, scale float64}` — `int64(s.scale * math.Pow(-math.Log(1-rng.Float64()), 1.0/s.shape))`

**Parameter derivation (from CV and rate):**
- Gamma: shape = 1/CV², scale = (CV²/rate) * 1e6 (to microseconds)
- Weibull: shape (k) derived from CV via **bisection** on `CV² = Γ(1+2/k)/Γ(1+1/k)² - 1` using `math.Gamma`. Tolerance: `|CV_computed - CV_target| < 0.001`. Range: k ∈ [0.1, 100]. **Max 100 iterations** — if convergence fails, log warning and use closest k found. Scale from mean: `scale = mean / Γ(1 + 1/k)`

### File: `sim/workload/distribution.go`

**Purpose:** Token length sampling.

**Interface:** `LengthSampler` with `Sample(rng *rand.Rand) int`

**Implementations:**
- `GaussianSampler` — reuses `sim.GenerateLengthGauss` pattern
- `ExponentialSampler` — `max(1, int(rng.ExpFloat64() * mean))`
- `ParetoLogNormalSampler` — mixture: with prob mix_weight draw Pareto, else LogNormal
- `EmpiricalPDFSampler` — CDF + binary search via `sort.SearchFloat64s`

### File: `sim/workload/generator.go`

**Purpose:** Main generation pipeline.

**Key function:**
```go
func GenerateRequests(spec *WorkloadSpec, horizon int64) ([]*sim.Request, error)
```

**Algorithm:** Per design doc section 6. Creates PartitionedRNG with SubsystemWorkloadGen. Normalizes rate fractions. Per-client: compute rate, create ArrivalSampler + LengthSamplers, generate requests respecting lifecycle windows, set metadata fields. Merge all clients, sort by ArrivalTime, assign sequential IDs.

### File: `sim/workload/tracev2.go`

**Purpose:** Trace v2 format types, export, and loading.

**Types:** TraceHeader (YAML), TraceRecord (CSV row), TraceV2 (combined)
**Functions:** ExportTraceV2, LoadTraceV2, LoadTraceV2Requests (creates []*sim.Request with synthetic token IDs)
**CSV format note:** All timestamp columns (ArrivalTimeUs, SendTimeUs, FirstChunkTimeUs, LastChunkTimeUs) MUST use integer formatting (`%d`), not float. Float formatting of int64 microsecond values (e.g., epoch-based) loses precision beyond ~2^53.

### File: `sim/workload/calibrate.go`

**Purpose:** Real-vs-sim statistical comparison.

**Two layers:**
- **Statistics:** `ComputeCalibration(real, sim []float64, metricName string)` — MAPE, Pearson r, per-percentile errors, quality rating
- **Orchestration:** `PrepareCalibrationPairs(realRecords, simResults, config)` — matches by request_id, applies network RTT + bandwidth normalization to sim, excludes warm-up, detects token count mismatches
**Types:** CalibrationReport (with KnownLimitations, ConfigMatch, TokenMismatchCount), MetricComparison, CalibrationConfig, SimResult, CalibrationPairs, LatencyPair

### File: `cmd/observe.go`

**Purpose:** Real mode HTTP client for OpenAI-compatible servers.

**Types:** RealClient, LoadGenerator, PendingRequest, RequestRecord, Recorder
**Key patterns:** httptest.Server for mock testing; goroutine-based dispatch; SSE chunk parsing for streaming

### File: `sim/request.go` (modifications)

**New fields (all zero-value safe):**
```go
TenantID        string   // Client/tenant identifier
SLOClass        string   // "realtime", "interactive", "batch"
Streaming       bool     // Streaming response mode
SessionID       string   // Multi-turn session link
RoundIndex      int      // Round within session
TextTokenCount  int      // Text input tokens
ImageTokenCount int      // Image input tokens
AudioTokenCount int      // Audio input tokens
VideoTokenCount int      // Video input tokens
ReasonRatio     float64  // reason_tokens / total_output_tokens; reasoning tokens are PART OF OutputTokens, not additional
```

### File: `sim/cluster/metrics.go` (modifications)

**New functions:**
```go
func ComputePerSLODistributions(aggregated *sim.Metrics) map[string]*SLOMetrics
func SLOAttainment(requests map[string]RequestMetrics, targets map[string]float64) float64
func JainFairnessIndex(throughputs map[string]float64) float64
```

**JainFairnessIndex formula:** `(Σxi)² / (N * Σxi²)` where xi = per-tenant throughput
