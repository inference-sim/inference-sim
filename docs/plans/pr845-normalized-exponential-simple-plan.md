# Implementation Plan: Normalized Exponential Arrival Sampler (Ultra-Simplified)

**PR Number:** #845
**Goal:** Add normalized exponential arrival sampler for inference-perf fidelity
**Source:** GitHub issue #845
**Closes:** Fixes #845
**Tier:** Small (3 files, ~80 LOC total)

---

## Part 1: Design Validation

### A. Executive Summary

This PR adds support for inference-perf's "normalized exponential" arrival pattern, eliminating systematic 2% duration bias in training data.

**Problem:** inference-perf generates exactly N requests spanning exactly T seconds by normalizing exponential samples. BLIS uses Poisson which stops at horizon, generating ~2% fewer requests.

**Solution:** Direct port of inference-perf's algorithm (~15 lines of Python → ~25 lines of Go), bypassing the factory pattern entirely:
1. Add `CustomSampler` field to `ClientSpec` (programmatic use only)
2. Implement `NormalizedExponentialSampler` (no factory registration)
3. Construct directly in `ExpandInferencePerfSpec`

**Files modified:** 3 files (spec.go, arrival.go, inference_perf.go)
**Lines changed:** ~50 LOC production + ~30 LOC tests = ~80 LOC total

**Key simplification:** No factory changes, no YAML changes, no sentinel overflow, no signature changes.

### B. Behavioral Contracts

**BC-1: Normalized Distribution**
```gherkin
GIVEN a sampler with count=3000, duration=600s (600M µs), rate=5/s
WHEN all 3000 intervals are generated
THEN sum(intervals) ≈ 600,000,000 µs (within count microseconds for rounding)
  AND exactly 3000 intervals are returned
```

**BC-2: Request Generation Guarantees**
```gherkin
GIVEN a client with CustomSampler=NormalizedExponentialSampler(count=3000, duration=600s)
WHEN workload is generated with horizon >= duration
THEN exactly 3000 requests are generated for that client
  AND all 3000 requests arrive within the duration window
NOTE: If horizon < duration, some requests may be dropped by the horizon guard
```

**BC-3: Inference-Perf Auto-Application (Single-Stage Only)**
```gherkin
GIVEN an InferencePerfSpec with a single stage AND no multi-turn
WHEN expanded to WorkloadSpec
THEN all clients use NormalizedExponentialSampler via CustomSamplerFactory
  AND count is set from ceil(stage.Rate * stage.Duration / numClients)
  AND duration is set from stage.Duration in microseconds
NOTE: Multi-stage workloads continue to use Poisson arrival (see BC-4)
NOTE: Multi-turn workloads continue to use Poisson arrival (see BC-5)
```

**BC-4: Multi-Stage Asymmetry (Architectural Constraint)**
```gherkin
GIVEN an InferencePerfSpec with multiple stages
WHEN expanded to WorkloadSpec
THEN all clients use Poisson arrival (not NormalizedExponentialSampler)
RATIONALE: Multi-stage workloads use per-stage client cohorts with lifecycle windows.
  Each cohort is active only during its stage's window. NormalizedExponentialSampler
  pre-generates N intervals spanning the FULL duration, but multi-stage clients are
  only active for a SUBSET of that duration (their stage's window). This mismatch
  would waste intervals or require complex windowing logic. Poisson generates
  intervals incrementally during the active window, matching the architecture.
FUTURE WORK: Per-stage NormalizedExponentialSampler construction (each stage gets
  its own sampler with count/duration scoped to that stage).
```

**BC-5: Multi-Turn Asymmetry (Session Start Time Only)**
```gherkin
GIVEN an InferencePerfSpec with multi-turn enabled
WHEN expanded to WorkloadSpec
THEN all clients use Poisson arrival for session start times
RATIONALE: Multi-turn workloads with SingleSession=true use ONE arrival sample
  for the session start time. The rounds within the session are spaced by
  ThinkTimeUs (not sampled IATs). NormalizedExponentialSampler pre-generates N
  intervals, but only the first would be used — wasting N-1 intervals.
  Poisson generates the session start time on-demand without waste.
NOTE: The rounds within each session still occur at exact ThinkTimeUs intervals,
  maintaining inference-perf's round-robin cycling semantics.
```

### C. Component Interaction

**Generator Loop (add CustomSampler check):**
```go
// generator.go:~115 (new code):
var arrivalSampler ArrivalSampler
if client.CustomSampler != nil {
    arrivalSampler = client.CustomSampler
} else {
    arrivalSampler = NewArrivalSampler(client.Arrival, ratePerMicrosecond)
}

// Rest of loop unchanged:
for currentTime < horizon {
    iat := arrivalSampler.SampleIAT(clientRNG)
    if iat == 0 { break }  // Exhausted sampler signals stop
    currentTime += iat
    if currentTime >= horizon { break }
    // generate request...
}
```

**Sampler Design (direct port from inference-perf):**
1. Constructor takes `rng`, `count`, `durationUs`
2. Generate N exponential samples (float64)
3. Normalize so sum equals duration (float64 arithmetic)
4. Convert to int64 microseconds, floor each to >= 1
5. SampleIAT() returns intervals sequentially
6. Returns 0 when exhausted (signals loop to break)

### D. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Memory for large N | 8N bytes for int64 intervals. For N=10K: 80KB. Reasonable for training workloads. |
| RNG determinism | Pass clientRNG from generator. RNG consumed in constructor (all N draws), not incrementally like Poisson. This changes downstream RNG stream - acceptable, documented. |
| Zero IAT from truncation | Floor all intervals to >= 1 after int64 conversion. |
| Float64 precision | Follows inference-perf. Sum may differ from target by ~N microseconds due to rounding. Acceptable. |

### E. Direct Port from inference-perf

**inference-perf (Python):**
```python
intervals = self._rand.exponential(1 / self._rate, num_requests)
total_interval_time = np.sum(intervals)
scale_factor = self._duration / total_interval_time
normalized_intervals = intervals * scale_factor
```

**BLIS (Go equivalent):**
```go
raw := make([]float64, count)
for i := range raw {
    raw[i] = rng.ExpFloat64() / rate
}
sum := 0.0
for _, v := range raw { sum += v }
scaleFactor := float64(durationUs) / sum
intervals := make([]int64, count)
for i, v := range raw {
    iat := int64(v * scaleFactor)
    if iat < 1 { iat = 1 }  // Floor to minimum 1µs
    intervals[i] = iat
}
```

---

## Part 2: Implementation Tasks

### Task 1: Add CustomSampler field to ClientSpec
- **Test:** `TestClientSpec_CustomSampler` — Verify non-nil CustomSampler is used in generation, nil falls back to factory
- **Code:** Add field to `ClientSpec` in spec.go:
```go
type ClientSpec struct {
    // ... existing fields
    Arrival       ArrivalSpec
    // CustomSampler allows programmatic injection of arrival samplers
    // (bypassing the factory). Used by inference-perf expansion.
    // Not exposed in YAML (yaml:"-" tag).
    CustomSampler ArrivalSampler `yaml:"-"`
}
```
- **Code:** Update generator.go:~115 to check CustomSampler:
```go
var arrivalSampler ArrivalSampler
if client.CustomSampler != nil {
    arrivalSampler = client.CustomSampler
} else {
    arrivalSampler = NewArrivalSampler(client.Arrival, ratePerMicrosecond)
}
```
- **Verification:** `go test ./sim/workload -run TestClientSpec_CustomSampler -v`
- **Commit:** `feat(workload): add CustomSampler field to ClientSpec (BC-2)`

### Task 2: Implement NormalizedExponentialSampler
- **Test:** `TestNormalizedExponentialSampler_Normalized` — Create sampler (count=600, duration=60s, rate=10/s). Sample all 600 IATs. Verify: (1) count==600, (2) sum within 600µs of 60,000,000µs, (3) all IATs >= 1, (4) final SampleIAT returns 0.
- **Test:** `TestNormalizedExponentialSampler_Deterministic` — Two samplers with same seed produce identical intervals.
- **Code:** Add sampler to arrival.go:
```go
// NormalizedExponentialSampler generates exactly N inter-arrival times
// that sum to exactly the target duration. Direct port of inference-perf's
// normalized exponential algorithm.
type NormalizedExponentialSampler struct {
    intervals []int64
    index     int
}

// NewNormalizedExponentialSampler creates a sampler that pre-generates
// count intervals normalized to sum to durationUs microseconds.
// rate is requests per microsecond (used for distribution shape only).
func NewNormalizedExponentialSampler(rng *rand.Rand, count int64, durationUs int64) *NormalizedExponentialSampler {
    if count <= 0 {
        panic(fmt.Sprintf("NormalizedExponentialSampler: count must be positive, got %d", count))
    }
    if durationUs <= 0 {
        panic(fmt.Sprintf("NormalizedExponentialSampler: duration must be positive, got %d", durationUs))
    }

    // Generate exponential samples (float64)
    rate := float64(count) / float64(durationUs)  // Requests per microsecond
    raw := make([]float64, count)
    for i := range raw {
        raw[i] = rng.ExpFloat64() / rate
    }

    // Normalize to sum exactly to duration (float64 arithmetic)
    sum := 0.0
    for _, v := range raw {
        sum += v
    }
    scaleFactor := float64(durationUs) / sum

    // Convert to int64 microseconds with floor to >= 1
    intervals := make([]int64, count)
    for i, v := range raw {
        iat := int64(v * scaleFactor)
        if iat < 1 {
            iat = 1
        }
        intervals[i] = iat
    }

    return &NormalizedExponentialSampler{intervals: intervals}
}

// SampleIAT returns pre-generated intervals sequentially.
// Returns 0 when exhausted (signals caller to stop).
func (s *NormalizedExponentialSampler) SampleIAT(_ *rand.Rand) int64 {
    if s.index >= len(s.intervals) {
        return 0  // Exhausted
    }
    iat := s.intervals[s.index]
    s.index++
    return iat
}
```
- **Verification:** `go test ./sim/workload -run TestNormalizedExponentialSampler -v`
- **Commit:** `feat(workload): implement NormalizedExponentialSampler (BC-1)`

### Task 3: Integrate into inference-perf
- **Test:** `TestExpandInferencePerfSpec_UsesNormalizedExponential` — Expand spec with stage (rate=5, duration=600). Verify: (1) clients have non-nil CustomSampler, (2) sampler type is NormalizedExponentialSampler, (3) count matches expected per-client count.
- **Code:** Modify `ExpandInferencePerfSpec()` in inference_perf.go:
  - Single-stage path (~line 119-132): Create sampler and assign to CustomSampler
  - Multi-stage path (~line 163-168): Same for each stage's clients
  - Compute total stage requests: `totalStageRequests := int64(math.Round(stage.Rate * float64(stage.Duration)))`
  - Compute per-client count: `perClientCount := totalStageRequests / int64(numClientsPerStage)` with remainder distribution
  - Compute per-client duration: `stageDurationUs := stage.Duration * 1_000_000` (microseconds)
  - Create sampler: `sampler := workload.NewNormalizedExponentialSampler(clientRNG, perClientCount, stageDurationUs)`
  - Assign to ClientSpec: `CustomSampler: sampler`
- **Code:** For remainder distribution, first `remainder` clients get `perClientCount+1`, rest get `perClientCount`
- **Verification:** `go test ./sim/workload -run TestExpandInferencePerfSpec_UsesNormalizedExponential -v`
- **Commit:** `feat(workload): use NormalizedExponentialSampler for inference-perf stages (BC-3)`

### Task 4: Update generator to handle zero IAT as stop signal
- **Test:** `TestGenerateRequests_StopsOnZeroIAT` — Create client with mock sampler that returns 3 non-zero IATs then 0. Verify exactly 3 requests generated.
- **Code:** In generator.go:~252, after `iat := arrivalSampler.SampleIAT(clientRNG)`, add:
```go
if iat == 0 {
    break  // Sampler exhausted
}
```
- **Verification:** `go test ./sim/workload -run TestGenerateRequests_StopsOnZeroIAT -v`
- **Commit:** `feat(workload): handle zero IAT as sampler exhaustion signal (BC-2)`

---

## Part 3: Acceptance Checklist

- [ ] Task 1: CustomSampler field test passes
- [ ] Task 2: Normalization test passes (sum within count µs)
- [ ] Task 2: Determinism test passes
- [ ] Task 3: Inference-perf integration test passes
- [ ] Task 4: Zero IAT stop signal test passes
- [ ] All tests pass: `go test ./sim/workload/... -count=1`
- [ ] Lint passes: `golangci-lint run ./sim/workload/...`
- [ ] Build passes: `go build ./...`

---

## Appendix: Why This Is Simpler

### Eliminated Complexity

**vs. Previous Plan:**
- ❌ No factory signature change (was: 10+ call site updates)
- ❌ No YAML validation updates (was: 2 validation functions)
- ❌ No error return from factory (was: error handling at all call sites)
- ❌ No RNG parameter to factory (was: threading RNG through factory)
- ❌ No `math.MaxInt64` sentinel (was: int64 overflow infinite loop bug)
- ❌ No Count/Duration fields on ArrivalSpec (was: cross-field validation)

**What Remains:**
- ✅ CustomSampler field (1 field, 3 lines)
- ✅ NormalizedExponentialSampler (50 lines)
- ✅ Direct construction in inference_perf.go (10 lines)
- ✅ Zero IAT check in generator (1 line)

### File Impact Summary

| File | Changes | LOC |
|------|---------|-----|
| sim/workload/spec.go | Add CustomSampler field + comment | +4 |
| sim/workload/arrival.go | Add NormalizedExponentialSampler (~50 lines with validation) | +50 |
| sim/workload/generator.go | CustomSampler check + zero IAT break | +6 |
| sim/workload/inference_perf.go | Construct and assign sampler | +15 |
| sim/workload/*_test.go | 4 new tests | +30 |
| **TOTAL** | | **~105 LOC** |

Note: Slightly higher than initial 80 LOC estimate due to Task 4 (zero IAT handling), but still far simpler than the 140 LOC factory-based approach.

---

## Deviation Log

**From original plan:**
1. **SIMPLIFICATION:** Bypass factory pattern entirely. Original plan modified factory signature and all call sites. New plan uses CustomSampler field.
2. **SIMPLIFICATION:** Use 0 as exhaustion signal instead of `math.MaxInt64` sentinel. Eliminates overflow bug.
3. **SIMPLIFICATION:** No YAML exposure. CustomSampler is programmatic only, not a YAML feature.
4. **SIMPLIFICATION:** Direct RNG threading. Pass clientRNG to constructor, no factory parameter.

**From inference-perf source:**
- Go requires explicit loops vs NumPy vectorization
- Floor all intervals to >= 1µs (Go specific, inference-perf uses float seconds)
- Return 0 when exhausted (Go sentinel convention)

All complexity from the factory-based approach has been **eliminated**.
