# ServeGen Temporal Parity Design

**Date:** 2026-04-24
**Issue:** #1124
**Author:** Claude Sonnet 4.5
**Status:** Proposed

## Executive Summary

Extend BLIS WorkloadSpec v2 to support per-window parameters (arrival patterns, token distributions, trace rates) enabling full ServeGen temporal parity. This allows BLIS to preserve the time-varying nature of ServeGen workloads instead of collapsing 14 days of traffic patterns into a single peak-rate client.

**Key changes:**
- Extend `ActiveWindow` struct with optional per-window fields
- Implement ServeGen-compatible proportional rate allocation
- Add post-hoc IAT rescaling for exact rate matching
- Converter produces 88 clients (one per chunk) with ~50-200 lifecycle windows each

**Parity achieved:**
- ✅ Per-window trace rates (proportional allocation)
- ✅ Per-window arrival parameters (CV preservation)
- ✅ Per-window token distributions (no averaging)
- ✅ Post-hoc IAT rescaling (exact rate matching)
- ✅ 88 clients = 88 ServeGen chunks (1:1 mapping)

## Problem Statement

### Current Behavior

`blis convert servegen` currently:
1. Reads all 203 active 10-minute windows from chunk-8-trace.csv
2. **Picks only the highest-rate window** (22.61 req/s at hour 55.3)
3. Creates one BLIS client running at this peak rate **for the entire simulation**
4. **Result:** 3.5x request inflation for single chunks, 100-500x for full folders

### ServeGen's Actual Semantics

Each chunk represents **one production client's 14-day behavior**:
- chunk-8-trace.csv contains 203 active 10-minute windows
- Each window has: timestamp, rate (weight), CV, pattern, shape, scale
- chunk-8-dataset.json contains per-window empirical token distributions
- At any timestamp, multiple chunks are active simultaneously

**Rate allocation (from `construct.py:190-207`):**
```python
for ts, window_group in windows_by_time.items():
    target_aggregate_rate = rate_fn[ts]  # User-specified

    # Sum trace rates as weights
    total_client_rate = sum(w.rate for w in window_group)

    # Allocate proportionally
    for window in window_group:
        client_target_rate = target_aggregate_rate * (window.rate / total_client_rate)

        # Sample IATs, then rescale to match target
        iats = sample_gamma(shape, scale, n)
        scale_factor = window_duration / sum(iats)
        iats *= scale_factor
```

**Key insight:** Trace "rate" is a **weight for proportional allocation**, not the actual rate. Shape/scale control **CV (burstiness)**, not rate. IATs are post-hoc scaled to achieve the allocated rate.

### Impact

- **Capacity planning:** Over-generation leads to incorrect resource sizing (up to 500x)
- **Research validity:** Published results may have inflated load
- **Temporal realism:** Loses diurnal patterns, traffic spikes, client lifecycle dynamics

## Design Goals

1. **Maximal ServeGen parity** - Preserve all temporal structure
2. **One client per chunk** - 88 BLIS clients = 88 ServeGen chunks
3. **Per-window everything** - Distributions, arrival params, rates vary per window
4. **Backward compatible** - Existing workloads without per-window params continue to work
5. **No WorkloadSpec v3** - Extend v2 via optional fields

## Detailed Design

### 1. WorkloadSpec Extensions

**Extend `ActiveWindow` struct (`sim/workload/spec.go`):**

```go
// ActiveWindow represents a period when a client is active.
type ActiveWindow struct {
    StartUs int64 `yaml:"start_us"`
    EndUs   int64 `yaml:"end_us"`

    // Per-window parameters for time-varying workloads (ServeGen compatibility).
    // When set, these override the client-level parameters during this window.
    TraceRate   *float64      `yaml:"trace_rate,omitempty"`            // Weight for proportional allocation
    Arrival     *ArrivalSpec  `yaml:"arrival,omitempty"`               // Arrival pattern (shape/scale for CV)
    InputDist   *DistSpec     `yaml:"input_distribution,omitempty"`    // Input token distribution
    OutputDist  *DistSpec     `yaml:"output_distribution,omitempty"`   // Output token distribution
}
```

**Fallback semantics:**
- If `window.Arrival == nil`, use `client.Arrival`
- If `window.InputDist == nil`, use `client.InputDist`
- If `window.OutputDist == nil`, use `client.OutputDist`
- If `window.TraceRate == nil`, use `client.RateFraction`

**Example YAML:**
```yaml
version: "2"
aggregate_rate: 150
clients:
  - id: "servegen-chunk-8"
    tenant_id: "chunk-8"
    rate_fraction: 1.0
    slo_class: "standard"
    streaming: true

    # Client-level defaults (used if window doesn't override)
    arrival:
      process: "poisson"
    input_distribution:
      type: "gaussian"
      params: {mean: 512, stddev: 128}
    output_distribution:
      type: "gaussian"
      params: {mean: 128, stddev: 32}

    lifecycle:
      windows:
        # Window 1: Hour 54
        - start_us: 194400000000
          end_us: 195000000000
          trace_rate: 9.84
          arrival:
            process: "gamma"
            shape: 1.257
            scale: 35157.0  # microseconds
          input_distribution:
            type: "empirical"
            params:
              "205": 8.493e-05
              "206": 0.003106
              # ... ~50 entries
          output_distribution:
            type: "empirical"
            params:
              "1": 0.0716
              "2": 0.5278
              # ... ~80 entries

        # Window 2: Hour 60 (different everything!)
        - start_us: 216000000000
          end_us: 216600000000
          trace_rate: 22.5
          arrival:
            process: "gamma"
            shape: 1.041
            scale: 42740.2
          input_distribution:
            type: "empirical"
            params:
              "205": 0.000102
              "206": 0.002696
              # ... different distribution
          output_distribution:
            type: "empirical"
            params:
              "1": 0.0683
              "2": 0.5266
              # ... different distribution
```

### 2. Request Generation with Proportional Allocation

**New function: `generateRequestsForWindow()` (`sim/workload/generator.go`):**

```go
func generateRequestsForWindow(
    client ClientSpec,
    window ActiveWindow,
    allClients []ClientSpec,
    aggregateRate float64,
    rng *rand.Rand,
) []*sim.Request {

    // Step 1: Determine parameters (with fallback to client-level)
    arrivalSpec := window.Arrival
    if arrivalSpec == nil {
        arrivalSpec = &client.Arrival
    }
    inputSpec := window.InputDist
    if inputSpec == nil {
        inputSpec = &client.InputDist
    }
    outputSpec := window.OutputDist
    if outputSpec == nil {
        outputSpec = &client.OutputDist
    }
    traceRate := client.RateFraction
    if window.TraceRate != nil {
        traceRate = *window.TraceRate
    }

    // Step 2: Compute proportional allocation (ServeGen semantics)
    // Find all co-active windows at this timestamp
    totalTraceRate := 0.0
    for _, otherClient := range allClients {
        if otherClient.Lifecycle == nil {
            totalTraceRate += otherClient.RateFraction
            continue
        }
        for _, otherWindow := range otherClient.Lifecycle.Windows {
            if otherWindow.StartUs < window.EndUs && window.StartUs < otherWindow.EndUs {
                rate := otherClient.RateFraction
                if otherWindow.TraceRate != nil {
                    rate = *otherWindow.TraceRate
                }
                totalTraceRate += rate
                break
            }
        }
    }

    if totalTraceRate == 0 {
        return nil
    }

    // Allocated rate for this window
    windowTargetRate := aggregateRate * (traceRate / totalTraceRate)
    windowDurationUs := window.EndUs - window.StartUs
    windowDurationSec := float64(windowDurationUs) / 1e6

    // Step 3: Sample IATs using shape/scale (for CV)
    expectedRequests := windowTargetRate * windowDurationSec
    numRequests := int(math.Ceil(expectedRequests))

    arrivalSampler := NewArrivalSampler(*arrivalSpec, windowTargetRate/1e6)
    iats := make([]int64, numRequests)
    for i := 0; i < numRequests; i++ {
        iats[i] = arrivalSampler.SampleIAT(rng)
    }

    // Step 4: Rescale IATs to match target rate (ServeGen parity!)
    sumIATs := int64(0)
    for _, iat := range iats {
        sumIATs += iat
    }
    if sumIATs == 0 {
        return nil
    }

    scaleFactor := float64(windowDurationUs) / float64(sumIATs)
    for i := range iats {
        iats[i] = int64(float64(iats[i]) * scaleFactor)
    }

    // Step 5: Generate requests with window-specific distributions
    inputSampler := NewLengthSampler(*inputSpec, rng)
    outputSampler := NewLengthSampler(*outputSpec, rng)

    requests := make([]*sim.Request, 0, numRequests)
    currentTime := window.StartUs

    for i := 0; i < numRequests; i++ {
        currentTime += iats[i]
        if currentTime >= window.EndUs {
            break
        }

        req := &sim.Request{
            ID:           fmt.Sprintf("%s-%d", client.ID, i),
            ArrivalTime:  currentTime,
            TenantID:     client.TenantID,
            SLOClass:     client.SLOClass,
            InputTokens:  inputSampler.Sample(rng),
            OutputTokens: outputSampler.Sample(rng),
            Streaming:    client.Streaming,
        }
        requests = append(requests, req)
    }

    return requests
}
```

**Integration:**
```go
func GenerateRequests(spec *WorkloadSpec, horizon int64, maxRequests int64) ([]*sim.Request, error) {
    // Check if any client has per-window parameters
    hasPerWindowParams := false
    for _, client := range spec.Clients {
        if client.Lifecycle != nil {
            for _, window := range client.Lifecycle.Windows {
                if window.TraceRate != nil || window.Arrival != nil {
                    hasPerWindowParams = true
                    break
                }
            }
        }
        if hasPerWindowParams {
            break
        }
    }

    // Route to appropriate generator
    if hasPerWindowParams {
        return generateTimeVaryingRequests(spec, horizon, maxRequests, rng)
    }
    return generateStaticRequests(spec, horizon, maxRequests, rng)
}
```

### 3. ServeGen Converter

**Rewrite `loadServeGenChunk()` (`sim/workload/servegen.go`):**

```go
func loadServeGenChunk(
    chunkID string,
    tracePath string,
    datasetPath string,
    sgConfig *ServeGenDataSpec,
) (*ClientSpec, error) {

    // Parse all trace rows (not just peak)
    rows, err := parseServeGenTrace(tracePath)
    if err != nil {
        return nil, err
    }

    // Load per-window distributions
    datasetByTimestamp, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
    if err != nil {
        return nil, err
    }

    // Build lifecycle windows
    windows := make([]ActiveWindow, 0)

    for _, row := range rows {
        // Filter by span
        if sgConfig.SpanStart > 0 && row.startTimeSec < float64(sgConfig.SpanStart) {
            continue
        }
        if sgConfig.SpanEnd > 0 && row.startTimeSec >= float64(sgConfig.SpanEnd) {
            continue
        }

        // Skip inactive windows
        if row.rate <= 0 {
            continue
        }

        // Get distributions for this timestamp
        dataset, ok := datasetByTimestamp[int(row.startTimeSec)]
        if !ok {
            logrus.Warnf("No dataset for chunk %s at t=%.0f, skipping", chunkID, row.startTimeSec)
            continue
        }

        // Build arrival spec
        arrivalSpec := ArrivalSpec{
            Process: strings.ToLower(row.pattern),
        }
        if row.cv > 0 {
            cv := row.cv
            arrivalSpec.CV = &cv
        }
        if row.shapeParam > 0 && row.scaleParam > 0 {
            shape := row.shapeParam
            scale := row.scaleParam * 1e6  // seconds → microseconds
            arrivalSpec.Shape = &shape
            arrivalSpec.Scale = &scale
        }

        // Build window
        window := ActiveWindow{
            StartUs:   int64(row.startTimeSec * 1e6),
            EndUs:     int64((row.startTimeSec + 600) * 1e6),  // 10 minutes
            TraceRate: &row.rate,
            Arrival:   &arrivalSpec,
            InputDist: &DistSpec{
                Type:   "empirical",
                Params: intMapToStringMap(dataset.inputPDF),
            },
            OutputDist: &DistSpec{
                Type:   "empirical",
                Params: intMapToStringMap(dataset.outputPDF),
            },
        }

        windows = append(windows, window)
    }

    if len(windows) == 0 {
        return nil, nil  // Inactive chunk
    }

    // Build ClientSpec
    client := &ClientSpec{
        ID:           fmt.Sprintf("servegen-chunk-%s", chunkID),
        TenantID:     fmt.Sprintf("chunk-%s", chunkID),
        RateFraction: 1.0,
        SLOClass:     "standard",
        Streaming:    true,

        Lifecycle: &LifecycleSpec{
            Windows: windows,
        },

        // Client-level defaults (unused)
        Arrival:    ArrivalSpec{Process: "poisson"},
        InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 512, "stddev": 128}},
        OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "stddev": 32}},
    }

    return client, nil
}
```

### 4. Testing Strategy

**Unit tests:**
- `TestPerWindowParameters` - Verify override semantics
- `TestProportionalAllocation` - Verify ServeGen rate allocation
- `TestIATRescaling` - Verify post-hoc scaling matches target rate

**Integration tests:**
- `TestServeGenConversion_TemporalPreservation` - Verify converter output
- Validate distributions vary across windows
- Validate 88 clients with multiple windows each

**End-to-end validation:**
- Convert m-mid folder
- Run simulation with 14-day horizon
- Verify INV-1 (request conservation)
- Verify multi-tenant (88 distinct tenant_ids)
- Compare request counts vs ServeGen (should match within 5%)

## Implementation Plan

**Phase 1: Core infrastructure (PR #1)**
- Extend `ActiveWindow` struct
- Add fallback logic in generator
- Tests for per-window override semantics

**Phase 2: Request generation (PR #2)**
- Implement `generateRequestsForWindow()`
- Add proportional allocation logic
- Add IAT rescaling
- Tests for allocation and rescaling

**Phase 3: ServeGen converter (PR #3)**
- Rewrite `loadServeGenChunk()`
- Add `loadServeGenDatasetAllWindows()`
- Integration tests with real ServeGen data

**Phase 4: Validation (PR #4)**
- End-to-end tests
- Performance benchmarks
- Documentation updates

## Performance Considerations

**Complexity:**
- Per-window computation: O(W × C) where W = avg windows/client, C = client count
- For m-mid: O(50 × 88) = 4,400 comparisons per window
- Acceptable for 14-day simulation

**File size:**
- m-mid: 88 chunks × 50 windows × 2KB = ~9 MB YAML
- Gzip compression: ~1 MB
- Acceptable for source control

**Memory:**
- Per-window distributions stored in ActiveWindow
- Estimated 50 MB RAM for m-mid workload
- Acceptable for modern machines

## Migration Path

✅ **Backward compatible** - existing workloads continue to work
✅ **No CLI flag changes**
✅ **No WorkloadSpec v3** - extends v2 via optional fields
✅ **Existing tests pass unchanged**

## Alternative Approaches Considered

### Option B: One client per window (rejected)
- 17,864 clients for m-mid
- 100+ MB YAML files
- Conceptual mismatch: one chunk ≠ one client
- **Why rejected:** Poor performance, huge files, loses chunk identity

### Option C: Time-sliced rate normalization (rejected)
- Pre-compute rate fractions per time slice
- ~1,500-3,000 clients
- **Why rejected:** Lossy (averages distributions), still missing IAT rescaling

## Open Questions

None - design is complete.

## Risks & Mitigations

**Risk:** 9 MB YAML files
**Mitigation:** Acceptable for modern systems, can gzip if needed

**Risk:** O(W × C) complexity
**Mitigation:** Only ~4,400 comparisons for m-mid, acceptable

**Risk:** Breaking existing workloads
**Mitigation:** Fully backward compatible via optional fields

## Success Criteria

✅ Convert m-mid folder: 88 clients, ~4,400 lifecycle windows
✅ Request count matches ServeGen within 5%
✅ Distributions vary per window (verified via tests)
✅ Rate allocation follows proportional semantics
✅ All existing tests pass
✅ INV-1 (request conservation) holds

## References

- Issue #1124: https://github.com/inference-sim/inference-sim/issues/1124
- ServeGen paper: https://arxiv.org/abs/2505.09999
- ServeGen construct.py: Lines 14-236 (proportional allocation + IAT rescaling)
- BLIS client.go: Lines 30-95 (lifecycle-aware rate normalization)
