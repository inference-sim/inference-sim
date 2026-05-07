# Workload Auto-Tuning Design

**Date:** 2026-05-07
**Status:** Proposed
**Author:** Claude (with user guidance)

## Overview

This design introduces a new `blis tune` subcommand that automatically adjusts workload specifications to avoid persistent saturation while hitting a target request count. The tool performs binary search over rate multipliers, detects saturation using rigorous statistical tests, and scales workload timelines to achieve precise request counts.

## Motivation

Capacity planning and performance experiments require workloads that stress systems without causing persistent saturation. Manual tuning is time-consuming and error-prone:

1. Users must iteratively adjust rate parameters and re-run simulations
2. Detecting saturation requires manual analysis of metrics
3. Scaling workloads to hit exact request counts involves tedious timeline arithmetic
4. Different workload formats (ServeGen, presets, inference-perf) have different rate control mechanisms

The `blis tune` subcommand automates this workflow, enabling rapid exploration of system capacity limits.

## Goals

1. **Automated rate tuning:** Find the maximum rate multiplier that avoids persistent saturation
2. **Precise request counts:** Scale workload timelines to generate exactly N requests
3. **Rigorous saturation detection:** Use the canonical definition from Discussion #1163
4. **Clean output:** Produce valid, ready-to-use WorkloadSpec YAML files
5. **Reusable components:** Expose saturation detection as a library function for other tools

## Non-Goals

1. Multi-objective optimization (e.g., maximizing throughput while meeting SLO targets)
2. Online auto-scaling during live workloads (this is offline capacity planning)
3. Support for all workload types initially (Phase 1 focuses on spike-based cohorts)
4. Interactive tuning with user feedback loops

## User Stories

### Story 1: ServeGen Workload Tuning
As a capacity planner, I want to tune a ServeGen workload (tau2) to find the maximum sustainable rate for my cluster configuration, so I can size my deployment accurately.

```bash
./blis tune --workload-spec agentic_tau2-workload.yaml \
  --rate-field spike.trace_rate \
  --num-requests 1000 \
  --model qwen/qwen3-14b \
  --num-instances 4 \
  --output tuned.yaml
```

### Story 2: Preset Workload Experimentation
As a researcher, I want to scale a chatbot workload to generate exactly 500 requests without saturating the system, so I can measure steady-state latency characteristics.

```bash
./blis convert preset --name chatbot --rate 10 | \
  ./blis tune --workload-spec - \
    --rate-field aggregate_rate \
    --num-requests 500 \
    --model meta-llama/Llama-2-70b-hf \
    --output experiment.yaml
```

### Story 3: Batch Experimentation
As a developer, I want to quickly test different cluster sizes against the same workload intensity, so I can evaluate scaling efficiency.

```bash
for N in 2 4 8; do
  ./blis tune --workload-spec base.yaml \
    --rate-field spike.trace_rate \
    --num-requests 1000 \
    --num-instances $N \
    --output tuned-n${N}.yaml
done
```

## Design

### CLI Interface

**Subcommand:** `blis tune`

**Required flags:**
- `--workload-spec <path>` - Input workload YAML file (or `-` for stdin)
- `--rate-field <field>` - Which rate field to scale (e.g., `spike.trace_rate`, `aggregate_rate`)
- `--num-requests <int>` - Target total request count after tuning
- `--model <name>` - Model configuration for simulation

**Optional flags:**
- `--output <path>` - Output tuned workload YAML (default: stdout)
- `--mu-min <float>` - Minimum rate multiplier for binary search (default: 0.1)
- `--mu-max <float>` - Maximum rate multiplier for binary search (default: 10.0)
- `--mu-tolerance <float>` - Convergence threshold for binary search (default: 0.05)
- `--max-iterations <int>` - Max binary search iterations (default: 20)
- All standard `blis run` simulation flags (--gpu, --tensor-parallelism, --num-instances, etc.)

**Rate field validation:**
The tool detects available rate fields in the workload and validates `--rate-field`:
- If `aggregate_rate > 0` exists → suggest `--rate-field aggregate_rate`
- If cohorts have `spike.trace_rate` → suggest `--rate-field spike.trace_rate`
- If requested field doesn't exist → error with available options
- If neither exists → error "no scalable rate field found"

**Error cases:**
- Baseline workload (mu=1.0) is already saturated → error with diagnostic info
- No non-saturating rate found in [mu_min, mu_max] → error suggesting lower --mu-max
- Workload type not supported (e.g., diurnal cohorts) → error with feature request
- Invalid rate field path → error with suggestions

### Algorithm: Two-Phase Auto-Tuning

#### Phase 1: Find Optimal Rate Multiplier

**Goal:** Find the largest `mu` such that `rate * mu` does not cause persistent saturation.

**Binary search with adaptive upper bound:**

1. **Initialize bounds:** `mu_low = mu_min`, `mu_high = mu_max`

2. **Test baseline:**
   - Run simulation with `mu = 1.0` (original workload rate)
   - If saturated → error "baseline workload already saturated"
   - If not saturated → proceed to step 3

3. **Find upper bound:**
   - Double mu iteratively: test `mu = 2, 4, 8, 16, ...`
   - Stop when saturation detected or `mu >= mu_max`
   - Set `mu_high = first_saturating_mu`
   - Set `mu_low = last_non_saturating_mu`

4. **Binary search:**
   ```
   while |mu_high - mu_low| > mu_tolerance:
       mu_mid = (mu_low + mu_high) / 2
       run simulation with rate * mu_mid
       if saturated:
           mu_high = mu_mid
       else:
           mu_low = mu_mid
   ```

5. **Result:** `optimal_mu = mu_low` (largest non-saturating multiplier)

**Convergence:**
- Stops when `|mu_high - mu_low| < mu_tolerance` (default 0.05 = 5% precision)
- Or after `max_iterations` (default 20)
- Typical runs converge in 5-8 iterations

**Per-iteration simulation:**
- Modify workload spec in-memory: scale rate field by `mu_trial`
- Call `cluster.RunSimulation()` with modified spec
- Extract `*sim.MetricsOutput`
- Call `workload.IsPersistentlySaturated(metrics)` → boolean

#### Phase 2: Scale Timeline to Target Request Count

**Goal:** Adjust workload duration to generate exactly `--num-requests` requests at the optimal rate.

1. **Calculate expected requests** at `optimal_mu`:
   ```go
   expectedCount := 0.0
   for each cohort:
       rate := cohort.spike.trace_rate * optimal_mu
       duration_sec := cohort.spike.duration_us / 1e6
       expectedCount += rate * duration_sec * cohort.population
   ```

2. **Compute scale factor:**
   ```go
   k := num_requests / expectedCount
   ```

3. **Scale timeline proportionally** (maintains relative timing structure):
   - For each cohort:
     - `new_start_time_us = start_time_us * k`
     - `new_duration_us = duration_us * k`
   - Global horizon (if set):
     - `new_horizon = horizon * k`
   - **Preserve rate intensity:** `trace_rate` remains at `original * optimal_mu`

4. **Verification:**
   - Recompute expected count with scaled timeline
   - Warn if `|new_expected - target| > 0.01 * target` (1% tolerance)

**Example:**
- Original: 3 cohorts at [0-10min, 13-23min, 26-36min], rate=100 req/s each
- Expected at optimal_mu=2.5: 3 × 2.5 × 100 req/s × 600s = 450,000 requests
- Target: 1,000,000 requests → `k = 1,000,000 / 450,000 ≈ 2.22`
- Scaled: [0-22.2min, 28.9-51.1min, 57.7-79.9min], rate=250 req/s each
- Proportional scaling preserves temporal gaps between spikes

### Saturation Detection

**Implementation:** `sim/workload/saturation.go`

**Core function:**
```go
func IsPersistentlySaturated(metrics *sim.MetricsOutput) bool
```

**Detection algorithm** (from Discussion #1163):

Persistent saturation occurs when **ALL** of these conditions hold:

#### 1. Positive Backlog Slope with 95% Confidence

- Reconstruct `active_requests(t)` timeseries from `metrics.Requests`:
  ```go
  for each request:
      arrival_time = arrived_at_ms * 1000  // convert to µs
      completion_time = arrival_time + e2e_ms * 1000
      for each sample time t in [0, horizon]:
          if arrival_time <= t < completion_time:
              active_requests[t] += 1
  ```
- Sample at 1-second intervals (configurable granularity)
- Fit linear regression: `active_requests(t) ≈ a + b*t`
- Compute 95% confidence interval for slope `b` using standard error
- **Criterion:** Lower bound of CI must be > 0

#### 2. Sustained Underdrain

- Divide timeline into windows of size = 5% of horizon (minimum 30 seconds)
- For each window, compute:
  ```go
  drain_ratio = num_completed_in_window / num_arrived_in_window
  ```
- **Criterion:** `drain_ratio < 1.0` for at least 50% of windows in the second half of the simulation
  - Second half check avoids false positives from initial ramp-up

#### 3. Substantial Final Backlog Growth

- Measure initial backlog after warmup: `initial = active_requests(t_10%)`
- Measure final backlog: `final = metrics.StillQueued + metrics.StillRunning`
- **Criterion:** `final > 3 * initial`

#### 4. Latency Drift (Optional Confirming Signal)

- Compute Pearson correlation between arrival order and E2E latency
- Positive correlation (r > 0.3) confirms saturation but is NOT required for verdict

**Final verdict:** Saturated if conditions 1, 2, AND 3 are TRUE.

**Edge cases:**
- Empty requests array → NOT saturated (trivial case)
- Horizon < 60s → Skip drain ratio check (insufficient data)
- All requests complete within horizon → NOT saturated regardless of slope
- Zero variance in timeseries → Cannot fit regression, NOT saturated

**Statistical details:**
- Use `gonum/stat` for linear regression and standard errors
- Confidence interval: `[b - 1.96*SE, b + 1.96*SE]` for 95% level
- P-value computation uses t-statistic: `t = b / SE` with n-2 degrees of freedom

### Timeline Scaling

**Implementation:** `sim/workload/scale.go`

**Scope (Phase 1):** Spike-based cohorts only

**Supported workload types:**
- Cohorts with `spike` specs containing `trace_rate` and `duration_us`
- Example: ServeGen workloads (tau2, appworld, reasoning)

**Unsupported initially (return error):**
- Diurnal cohorts (24-hour sinusoidal modulation)
- Drain specs (linear ramp-down)
- Aggregate_rate mode with client-level rate fractions
- Inference-perf stage-based workloads

**Core function:**
```go
func ScaleWorkloadTimeline(spec *WorkloadSpec, scaleFactor float64) error
```

**Validation:**
- Check all cohorts have `spike` spec: `cohort.Spike != nil`
- Reject if `scaleFactor < 0.01` (unreasonably small, would create sub-millisecond durations)
- Reject if `scaleFactor > 100` (unreasonably large, likely user error)

**Scaling rules:**
```go
for each cohort in spec.Cohorts:
    if cohort.Spike == nil:
        return error "unsupported lifecycle type"

    cohort.Spike.StartTimeUs = int64(float64(cohort.Spike.StartTimeUs) * scaleFactor)
    cohort.Spike.DurationUs = int64(float64(cohort.Spike.DurationUs) * scaleFactor)
    // trace_rate unchanged (already scaled by optimal_mu in Phase 1)

if spec.Horizon > 0:
    spec.Horizon = int64(float64(spec.Horizon) * scaleFactor)
```

**Invariants maintained:**
- Rate intensity: `trace_rate * optimal_mu` remains constant
- Temporal proportions: gaps between spikes scale by same factor as durations
- Relative ordering: cohorts maintain their sequence
- Population: cohort population unchanged

**Request count verification:**
```go
expectedCount := 0.0
for each cohort:
    rate := cohort.Spike.TraceRate  // already includes optimal_mu
    duration_sec := float64(cohort.Spike.DurationUs) / 1e6
    expectedCount += rate * duration_sec * float64(cohort.Population)

if abs(expectedCount - float64(targetRequests)) > 0.01 * float64(targetRequests):
    logrus.Warnf("Expected count %.0f differs from target %d by >1%%",
                 expectedCount, targetRequests)
```

### Output Format

**Tuned workload YAML:**
- Valid `WorkloadSpec` that can be used with `blis run` or `blis replay`
- Metadata comment header for traceability:

```yaml
# Auto-tuned by blis tune
# Original: agentic_tau2-workload.yaml
# Target requests: 1000
# Optimal rate multiplier: 2.75
# Timeline scale factor: 0.92
# Generated: 2026-05-07T10:30:00Z
# Command: blis tune --workload-spec agentic_tau2-workload.yaml --rate-field spike.trace_rate --num-requests 1000 --model qwen/qwen3-14b

version: "2"
seed: 300
category: reasoning
cohorts:
  - id: midnight-background-reasoning
    population: 15
    spike:
      start_time_us: 0
      duration_us: 552000000  # scaled from 600000000
      trace_rate: 244.90  # scaled from 89.06 by mu=2.75
    ...
```

**Diagnostic output (stderr):**
```
[tune] Loading workload spec: agentic_tau2-workload.yaml
[tune] Detected rate field: spike.trace_rate
[tune] Target requests: 1000 (from --num-requests)
[tune] Model: qwen/qwen3-14b, Instances: 1
[tune]
[tune] Phase 1: Finding optimal rate multiplier
[tune] Testing baseline (mu=1.0)... not saturated ✓
[tune] Finding upper bound... mu=2.0 not saturated, mu=4.0 saturated ✓
[tune] Binary search bounds: [2.0, 4.0]
[tune]   iter 1: testing mu=3.00... saturated
[tune]   iter 2: testing mu=2.50... not saturated
[tune]   iter 3: testing mu=2.75... not saturated
[tune]   iter 4: testing mu=2.875... saturated
[tune]   iter 5: testing mu=2.8125... saturated
[tune]   iter 6: testing mu=2.78125... not saturated
[tune] Converged: optimal_mu = 2.78 (6 iterations, tolerance=0.05)
[tune]
[tune] Phase 2: Scaling timeline to target request count
[tune] Expected requests at mu=2.78: 1087 (baseline duration)
[tune] Scale factor: 0.92 (to achieve 1000 requests)
[tune] Scaled timeline: horizon 552s -> 508s
[tune]
[tune] Done. Writing tuned workload to tuned.yaml
```

**Error messages:**
```
[tune] ERROR: Workload saturated at baseline (mu=1.0)
[tune]   Backlog slope: +12.5 req/s (95% CI: [10.2, 14.8])
[tune]   Drain ratio: 0.73 (< 1.0 in 80% of windows)
[tune]   Final backlog: 342 requests (initial: 23)
[tune]   → System cannot handle this workload at current capacity
[tune]   → Try reducing num-instances or request rate

[tune] ERROR: No non-saturating rate found in range [0.1, 10.0]
[tune]   All tested multipliers resulted in saturation
[tune]   → Try reducing --mu-max to narrow search range
[tune]   → Or increase cluster capacity (--num-instances)

[tune] ERROR: Rate field 'spike.trace_rate' not found in workload
[tune]   Available fields: aggregate_rate
[tune]   → Use: --rate-field aggregate_rate

[tune] ERROR: Workload type not supported
[tune]   Found diurnal lifecycle in cohort 'morning-background'
[tune]   → blis tune currently supports spike-based cohorts only
[tune]   → See docs/guide/workload-tuning.md for supported types
```

### Module Structure

**New files:**

1. **`sim/workload/saturation.go`**
   - Public API:
     - `IsPersistentlySaturated(metrics *MetricsOutput) bool`
   - Internal helpers:
     - `reconstructActiveRequests(requests []RequestMetrics, horizon int64) []int`
     - `computeBacklogSlope(timeseries []int) (slope, lowerCI, upperCI float64)`
     - `computeDrainRatios(requests []RequestMetrics, windowSize int64) []float64`
     - `computeLatencyCorrelation(requests []RequestMetrics) float64`

2. **`sim/workload/autotune.go`**
   - Public API:
     - `TuneWorkload(spec *WorkloadSpec, config TuneConfig) (*WorkloadSpec, error)`
   - Configuration:
     ```go
     type TuneConfig struct {
         RateField      string  // e.g., "spike.trace_rate"
         TargetRequests int
         MuMin          float64
         MuMax          float64
         MuTolerance    float64
         MaxIterations  int
         SimConfig      SimConfig  // cluster config for trial runs
     }
     ```
   - Internal helpers:
     - `findOptimalMu(spec *WorkloadSpec, config TuneConfig) (float64, error)`
     - `scaleRateField(spec *WorkloadSpec, field string, multiplier float64) error`
     - `scaleWorkloadTimeline(spec *WorkloadSpec, scaleFactor float64) error`
     - `detectRateField(spec *WorkloadSpec, requestedField string) error`
     - `calculateExpectedRequests(spec *WorkloadSpec) int`

3. **`cmd/autotune.go`**
   - Cobra command setup
   - Flag parsing and validation
   - Thin wrapper calling `workload.TuneWorkload()`
   - YAML output with metadata header
   - Diagnostic logging to stderr

**Integration points:**

- **Calls into existing code:**
  - `cluster.RunSimulation()` - Execute each trial simulation
  - `workload.LoadWorkloadSpec()` - Load input YAML
  - `workload.GenerateRequests()` - Generate request sequence for each trial
  - `gopkg.in/yaml.v3` - Marshal tuned spec to YAML

- **Dependency graph:**
  ```
  cmd/autotune.go
      ↓
  sim/workload/autotune.go
      ↓
  ├── sim/workload/saturation.go
  ├── sim/cluster (RunSimulation)
  └── sim (MetricsOutput, Request types)
  ```

- **Follows BLIS separation of concerns:**
  - `sim/workload` = library functions (no I/O, no os.Exit)
  - `cmd/` = CLI layer (I/O, error reporting, process exit)
  - Deterministic given same inputs (respects INV-6)

### Testing Strategy

**Unit tests:**

1. **`saturation_test.go`:**
   - Synthetic metrics with known saturation characteristics
   - Test each condition independently (slope, drain ratio, backlog growth)
   - Edge cases: empty metrics, single request, all requests complete
   - Table-driven tests for various backlog slope scenarios

2. **`autotune_test.go`:**
   - Mock simulation results for binary search convergence
   - Test timeline scaling with example cohorts (tau2-style)
   - Rate field detection with different workload types
   - Error cases: baseline saturated, no convergence, invalid scale factors

3. **`scale_test.go`:**
   - Verify request count math for scaled timelines
   - Proportional scaling preserves temporal ratios
   - Edge cases: single cohort, overlapping spikes, zero-duration spikes

**Integration tests:**

4. **`cmd/autotune_test.go`:**
   - End-to-end: tune a small workload (10-20 requests)
   - Verify output YAML is valid and parseable
   - Verify tuned workload produces expected request count
   - Verify tuned workload is not saturated when run

**Golden tests:**

5. **`testdata/autotune/`:**
   - Input workload: `tau2-mini.yaml` (3 cohorts, 100 requests baseline)
   - Expected output: `tau2-mini-tuned.golden.yaml`
   - Verify deterministic output (INV-6)

### Future Extensions

**Phase 2 enhancements** (out of scope for initial PR):

1. **Support aggregate_rate workloads:**
   - Scale `spec.AggregateRate` instead of `spike.trace_rate`
   - Scale `spec.Horizon` or `spec.NumRequests` in Phase 2

2. **Support diurnal cohorts:**
   - Scale the 24-hour cycle proportionally
   - Preserve peak-to-trough ratios

3. **Support drain specs:**
   - Scale `ramp_duration_us` proportionally

4. **Multi-objective optimization:**
   - Maximize throughput while keeping P99 E2E < SLO target
   - Requires fitness function: `f(mu) = throughput if P99 < target else 0`

5. **Configurable saturation thresholds:**
   - Add flags: `--saturation-confidence`, `--saturation-backlog-multiplier`
   - Allow users to tune sensitivity

6. **Incremental tuning:**
   - Start from previously tuned workload instead of baseline
   - Useful for iterative capacity planning

7. **Multi-instance awareness:**
   - Detect whether saturation is global or per-instance
   - Suggest cluster size adjustments

## Open Questions

1. **Window size for drain ratio:** Should we use fixed 5% of horizon, or adaptive based on arrival rate?
   - **Decision:** Fixed 5% with 30s minimum is simpler and sufficient for typical workloads (10min+ duration)

2. **Sampling rate for timeseries:** 1-second intervals may be too coarse for sub-second workloads.
   - **Decision:** Use 1s default, document limitation for sub-second horizons

3. **Rounding errors in timeline scaling:** Should we round timestamps to nearest microsecond?
   - **Decision:** Use float64 → int64 truncation, document potential ±1µs drift

4. **Metadata preservation:** Should we preserve user comments in the original YAML?
   - **Decision:** No (YAML round-tripping loses comments), add header comment instead

5. **Progress reporting:** Should long-running binary searches show progress bars?
   - **Decision:** No (adds dependency), use simple log lines to stderr

## Success Metrics

1. **Correctness:** Tuned workloads are never persistently saturated (validate with manual runs)
2. **Precision:** Final request count within 1% of target in >95% of cases
3. **Efficiency:** Binary search converges in <10 iterations for typical workloads
4. **Usability:** Users can tune tau2 workload in <60 seconds on a laptop
5. **Reusability:** Saturation detection function used by at least 2 other tools within 6 months

## References

- [Discussion #1163: Persistent Saturation Detection](https://github.com/inference-sim/inference-sim/discussions/1163)
- [BLIS Design Guidelines](../contributing/templates/design-guidelines.md)
- [W0-4: Workload Unification](../plans/archive/)
- [ServeGen Multi-Period Workloads](../../sim/workload/servegen.go)
