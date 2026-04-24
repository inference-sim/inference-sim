# ServeGen Temporal Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable BLIS to preserve ServeGen's time-varying workload structure by extending `ActiveWindow` with per-window parameters and implementing proportional rate allocation with IAT rescaling.

**Architecture:** Three-layer approach: (1) extend WorkloadSpec v2 with optional per-window fields, (2) add time-varying request generator with ServeGen proportional allocation, (3) rewrite ServeGen converter to populate per-window parameters. Each layer builds on TDD with unit → integration → e2e test progression.

**Tech Stack:** Go 1.22+, gopkg.in/yaml.v3, gonum/stat (for gamma distribution), existing BLIS workload infrastructure

---

## File Structure

**Modified files:**
- `sim/workload/spec.go` - Extend `ActiveWindow` struct with per-window fields
- `sim/workload/generator.go` - Add time-varying request generation with IAT rescaling
- `sim/workload/servegen.go` - Rewrite converter to populate per-window parameters

**New test files:**
- `sim/workload/generator_perwindow_test.go` - Unit tests for per-window parameter logic
- `sim/workload/servegen_temporal_test.go` - Integration tests for temporal preservation

---

## Task 1: Extend ActiveWindow Struct

**Files:**
- Modify: `sim/workload/spec.go:173-176`
- Test: `sim/workload/spec_test.go`

- [ ] **Step 1: Write test for per-window YAML parsing**

```go
// In sim/workload/spec_test.go
func TestActiveWindow_PerWindowParameters(t *testing.T) {
	yamlData := `
version: "2"
aggregate_rate: 100
clients:
  - id: "test-client"
    rate_fraction: 1.0
    arrival:
      process: "poisson"
    input_distribution:
      type: "constant"
      params: {value: 100}
    output_distribution:
      type: "constant"
      params: {value: 50}
    lifecycle:
      windows:
        - start_us: 0
          end_us: 10000000
          trace_rate: 15.2
          arrival:
            process: "gamma"
            shape: 1.5
            scale: 50000
          input_distribution:
            type: "constant"
            params: {value: 200}
          output_distribution:
            type: "constant"
            params: {value: 75}
`
	var spec WorkloadSpec
	err := yaml.Unmarshal([]byte(yamlData), &spec)
	require.NoError(t, err)

	require.Len(t, spec.Clients, 1)
	require.NotNil(t, spec.Clients[0].Lifecycle)
	require.Len(t, spec.Clients[0].Lifecycle.Windows, 1)

	window := spec.Clients[0].Lifecycle.Windows[0]
	assert.Equal(t, int64(0), window.StartUs)
	assert.Equal(t, int64(10000000), window.EndUs)
	assert.NotNil(t, window.TraceRate)
	assert.Equal(t, 15.2, *window.TraceRate)
	assert.NotNil(t, window.Arrival)
	assert.Equal(t, "gamma", window.Arrival.Process)
	assert.NotNil(t, window.Arrival.Shape)
	assert.Equal(t, 1.5, *window.Arrival.Shape)
	assert.NotNil(t, window.InputDist)
	assert.Equal(t, "constant", window.InputDist.Type)
	assert.Equal(t, 200.0, window.InputDist.Params["value"])
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestActiveWindow_PerWindowParameters -v`
Expected: FAIL with "unknown field 'trace_rate'" or similar

- [ ] **Step 3: Extend ActiveWindow struct**

```go
// In sim/workload/spec.go (replace lines 173-176)
// ActiveWindow represents a period when a client is active.
type ActiveWindow struct {
	StartUs int64 `yaml:"start_us"`
	EndUs   int64 `yaml:"end_us"`

	// Per-window parameters for time-varying workloads (ServeGen compatibility).
	// When set, these override the client-level parameters during this window.
	TraceRate  *float64      `yaml:"trace_rate,omitempty"`            // Weight for proportional allocation
	Arrival    *ArrivalSpec  `yaml:"arrival,omitempty"`               // Arrival pattern (shape/scale for CV)
	InputDist  *DistSpec     `yaml:"input_distribution,omitempty"`    // Input token distribution
	OutputDist *DistSpec     `yaml:"output_distribution,omitempty"`   // Output token distribution
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestActiveWindow_PerWindowParameters -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sim/workload/spec.go sim/workload/spec_test.go
git commit -m "feat(workload): extend ActiveWindow with per-window parameters

Add optional per-window fields to ActiveWindow:
- TraceRate: weight for proportional allocation (ServeGen)
- Arrival: arrival pattern override (shape/scale for CV)
- InputDist/OutputDist: per-window token distributions

Backward compatible via pointer types (nil = use client-level).
Enables time-varying workload support for issue #1124.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Parameter Fallback Helper

**Files:**
- Modify: `sim/workload/generator.go` (add at end)
- Test: `sim/workload/generator_perwindow_test.go` (new file)

- [ ] **Step 1: Write test for parameter fallback logic**

```go
// Create sim/workload/generator_perwindow_test.go
package workload

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResolveWindowParameters(t *testing.T) {
	clientArrival := ArrivalSpec{Process: "poisson"}
	clientInputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 100}}
	clientOutputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 50}}
	clientRateFraction := 1.0

	client := ClientSpec{
		Arrival:      clientArrival,
		InputDist:    clientInputDist,
		OutputDist:   clientOutputDist,
		RateFraction: clientRateFraction,
	}

	t.Run("window overrides all parameters", func(t *testing.T) {
		windowArrival := ArrivalSpec{Process: "gamma", Shape: ptr(2.0), Scale: ptr(1000.0)}
		windowInputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 200}}
		windowOutputDist := DistSpec{Type: "constant", Params: map[string]float64{"value": 75}}
		windowTraceRate := 15.2

		window := ActiveWindow{
			StartUs:    0,
			EndUs:      10000000,
			TraceRate:  &windowTraceRate,
			Arrival:    &windowArrival,
			InputDist:  &windowInputDist,
			OutputDist: &windowOutputDist,
		}

		arrival, input, output, traceRate := resolveWindowParameters(client, window)

		assert.Equal(t, "gamma", arrival.Process)
		assert.Equal(t, 2.0, *arrival.Shape)
		assert.Equal(t, "constant", input.Type)
		assert.Equal(t, 200.0, input.Params["value"])
		assert.Equal(t, "constant", output.Type)
		assert.Equal(t, 75.0, output.Params["value"])
		assert.Equal(t, 15.2, traceRate)
	})

	t.Run("window fallback to client parameters", func(t *testing.T) {
		window := ActiveWindow{
			StartUs: 0,
			EndUs:   10000000,
			// No overrides - should use client-level
		}

		arrival, input, output, traceRate := resolveWindowParameters(client, window)

		assert.Equal(t, "poisson", arrival.Process)
		assert.Equal(t, "constant", input.Type)
		assert.Equal(t, 100.0, input.Params["value"])
		assert.Equal(t, "constant", output.Type)
		assert.Equal(t, 50.0, output.Params["value"])
		assert.Equal(t, 1.0, traceRate)
	})

	t.Run("partial overrides", func(t *testing.T) {
		windowTraceRate := 22.5
		window := ActiveWindow{
			StartUs:   0,
			EndUs:     10000000,
			TraceRate: &windowTraceRate,
			// No arrival/dist overrides - should use client-level
		}

		arrival, input, output, traceRate := resolveWindowParameters(client, window)

		assert.Equal(t, "poisson", arrival.Process)  // From client
		assert.Equal(t, 100.0, input.Params["value"])  // From client
		assert.Equal(t, 22.5, traceRate)  // From window
	})
}

func ptr[T any](v T) *T {
	return &v
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestResolveWindowParameters -v`
Expected: FAIL with "undefined: resolveWindowParameters"

- [ ] **Step 3: Implement resolveWindowParameters helper**

```go
// Add to sim/workload/generator.go (at end of file)

// resolveWindowParameters returns the effective parameters for a window,
// with fallback to client-level parameters when window doesn't override.
func resolveWindowParameters(client ClientSpec, window ActiveWindow) (
	arrival ArrivalSpec,
	inputDist DistSpec,
	outputDist DistSpec,
	traceRate float64,
) {
	// Arrival pattern
	if window.Arrival != nil {
		arrival = *window.Arrival
	} else {
		arrival = client.Arrival
	}

	// Input distribution
	if window.InputDist != nil {
		inputDist = *window.InputDist
	} else {
		inputDist = client.InputDist
	}

	// Output distribution
	if window.OutputDist != nil {
		outputDist = *window.OutputDist
	} else {
		outputDist = client.OutputDist
	}

	// Trace rate
	if window.TraceRate != nil {
		traceRate = *window.TraceRate
	} else {
		traceRate = client.RateFraction
	}

	return arrival, inputDist, outputDist, traceRate
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestResolveWindowParameters -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_perwindow_test.go
git commit -m "feat(workload): add resolveWindowParameters helper

Helper function for per-window parameter resolution with fallback
to client-level defaults. Supports TraceRate, Arrival, InputDist,
OutputDist overrides for time-varying workloads.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Implement IAT Rescaling

**Files:**
- Modify: `sim/workload/generator.go` (add function)
- Test: `sim/workload/generator_perwindow_test.go`

- [ ] **Step 1: Write test for IAT rescaling**

```go
// Add to sim/workload/generator_perwindow_test.go
func TestRescaleIATsToMatchDuration(t *testing.T) {
	t.Run("rescale to 10 second window", func(t *testing.T) {
		// IATs sum to 20 seconds (20,000,000 µs)
		iats := []int64{5000000, 5000000, 5000000, 5000000}  // 5s each
		targetDuration := int64(10000000)  // 10 seconds

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		// Sum should equal target duration
		sum := int64(0)
		for _, iat := range rescaled {
			sum += iat
		}
		assert.Equal(t, targetDuration, sum)

		// CV should be preserved (all equal → CV=0)
		for i := 1; i < len(rescaled); i++ {
			assert.Equal(t, rescaled[0], rescaled[i], "uniform scaling preserves equal values")
		}
	})

	t.Run("rescale preserves relative ratios", func(t *testing.T) {
		// Unequal IATs
		iats := []int64{1000000, 3000000, 6000000}  // 1s, 3s, 6s (sum=10s)
		targetDuration := int64(20000000)  // 20 seconds

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		// Check sum
		sum := int64(0)
		for _, iat := range rescaled {
			sum += iat
		}
		assert.Equal(t, targetDuration, sum)

		// Check ratios preserved: rescaled[1]/rescaled[0] should equal iats[1]/iats[0]
		originalRatio := float64(iats[1]) / float64(iats[0])
		rescaledRatio := float64(rescaled[1]) / float64(rescaled[0])
		assert.InDelta(t, originalRatio, rescaledRatio, 0.01, "ratio 3:1 preserved")
	})

	t.Run("zero sum IATs returns zeros", func(t *testing.T) {
		iats := []int64{0, 0, 0}
		targetDuration := int64(10000000)

		rescaled := rescaleIATsToMatchDuration(iats, targetDuration)

		for _, iat := range rescaled {
			assert.Equal(t, int64(0), iat)
		}
	})
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestRescaleIATsToMatchDuration -v`
Expected: FAIL with "undefined: rescaleIATsToMatchDuration"

- [ ] **Step 3: Implement rescaleIATsToMatchDuration**

```go
// Add to sim/workload/generator.go (before resolveWindowParameters)

// rescaleIATsToMatchDuration rescales inter-arrival times to sum exactly to
// targetDuration, preserving relative ratios (CV). Implements ServeGen's
// post-hoc IAT scaling for exact rate matching (construct.py:46-48).
func rescaleIATsToMatchDuration(iats []int64, targetDuration int64) []int64 {
	if len(iats) == 0 {
		return nil
	}

	// Compute sum
	sumIATs := int64(0)
	for _, iat := range iats {
		sumIATs += iat
	}

	// Guard against zero sum
	if sumIATs == 0 {
		return iats
	}

	// Scale factor
	scaleFactor := float64(targetDuration) / float64(sumIATs)

	// Rescale all IATs
	rescaled := make([]int64, len(iats))
	for i, iat := range iats {
		rescaled[i] = int64(float64(iat) * scaleFactor)
	}

	return rescaled
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestRescaleIATsToMatchDuration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_perwindow_test.go
git commit -m "feat(workload): add IAT rescaling for exact rate matching

Implement rescaleIATsToMatchDuration to post-hoc scale inter-arrival
times to sum exactly to target duration. Preserves CV (relative ratios)
while achieving exact rate matching. Matches ServeGen construct.py:46-48.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Implement Proportional Allocation

**Files:**
- Modify: `sim/workload/generator.go` (add function)
- Test: `sim/workload/generator_perwindow_test.go`

- [ ] **Step 1: Write test for proportional allocation**

```go
// Add to sim/workload/generator_perwindow_test.go
func TestComputeProportionalRate(t *testing.T) {
	t.Run("three co-active clients with trace rates", func(t *testing.T) {
		// ServeGen scenario from spec: chunk-2 (15.2), chunk-8 (22.5), chunk-20 (5.3)
		// At timestamp 0-10s, all three overlap
		clients := []ClientSpec{
			{ID: "chunk-2", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptr(15.2)}},
			}},
			{ID: "chunk-8", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptr(22.5)}},
			}},
			{ID: "chunk-20", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptr(5.3)}},
			}},
		}

		aggregateRate := 150.0
		window := clients[1].Lifecycle.Windows[0]  // chunk-8's window

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// Expected: 150 × (22.5 / (15.2+22.5+5.3)) = 150 × (22.5 / 43.0) = 78.49
		assert.InDelta(t, 78.49, allocatedRate, 0.01)
	})

	t.Run("non-overlapping windows", func(t *testing.T) {
		// chunk-2 active 0-10s, chunk-8 active 20-30s
		clients := []ClientSpec{
			{ID: "chunk-2", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptr(15.2)}},
			}},
			{ID: "chunk-8", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 20000000, EndUs: 30000000, TraceRate: ptr(22.5)}},
			}},
		}

		aggregateRate := 150.0
		window := clients[1].Lifecycle.Windows[0]  // chunk-8's window (20-30s)

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// Expected: 150 × (22.5 / 22.5) = 150 (only chunk-8 active)
		assert.InDelta(t, 150.0, allocatedRate, 0.01)
	})

	t.Run("always-on client (no lifecycle)", func(t *testing.T) {
		// chunk-8 has lifecycle, background client always-on
		clients := []ClientSpec{
			{ID: "background", RateFraction: 10.0, Lifecycle: nil},  // Always-on
			{ID: "chunk-8", RateFraction: 1.0, Lifecycle: &LifecycleSpec{
				Windows: []ActiveWindow{{StartUs: 0, EndUs: 10000000, TraceRate: ptr(22.5)}},
			}},
		}

		aggregateRate := 150.0
		window := clients[1].Lifecycle.Windows[0]

		allocatedRate := computeProportionalRate(clients[1], window, clients, aggregateRate)

		// Expected: 150 × (22.5 / (10.0+22.5)) = 150 × (22.5 / 32.5) = 103.85
		assert.InDelta(t, 103.85, allocatedRate, 0.01)
	})
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestComputeProportionalRate -v`
Expected: FAIL with "undefined: computeProportionalRate"

- [ ] **Step 3: Implement computeProportionalRate**

```go
// Add to sim/workload/generator.go (before rescaleIATsToMatchDuration)

// computeProportionalRate computes the allocated rate for a window using
// ServeGen's proportional allocation semantics (construct.py:190-207).
// Returns: target_aggregate_rate × (window_trace_rate / sum_of_co_active_trace_rates)
func computeProportionalRate(
	client ClientSpec,
	window ActiveWindow,
	allClients []ClientSpec,
	aggregateRate float64,
) float64 {
	// Get this window's trace rate
	_, _, _, traceRate := resolveWindowParameters(client, window)

	// Sum trace rates of all co-active windows
	totalTraceRate := 0.0

	for _, otherClient := range allClients {
		// Always-on clients (no lifecycle) contribute their rate_fraction
		if otherClient.Lifecycle == nil || len(otherClient.Lifecycle.Windows) == 0 {
			totalTraceRate += otherClient.RateFraction
			continue
		}

		// Check if any of otherClient's windows overlap with current window
		for _, otherWindow := range otherClient.Lifecycle.Windows {
			// Time-based overlap check
			if otherWindow.StartUs < window.EndUs && window.StartUs < otherWindow.EndUs {
				_, _, _, otherRate := resolveWindowParameters(otherClient, otherWindow)
				totalTraceRate += otherRate
				break  // Only count each client once per timestamp
			}
		}
	}

	if totalTraceRate == 0 {
		return 0
	}

	// Proportional allocation
	return aggregateRate * (traceRate / totalTraceRate)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestComputeProportionalRate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_perwindow_test.go
git commit -m "feat(workload): implement proportional rate allocation

Add computeProportionalRate for ServeGen-compatible rate allocation.
Computes: aggregate_rate × (window_trace_rate / sum_co_active_rates).
Handles overlapping windows and always-on clients correctly.
Matches ServeGen construct.py:190-207 semantics.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Implement Per-Window Request Generation

**Files:**
- Modify: `sim/workload/generator.go` (add generateRequestsForWindow function)
- Test: `sim/workload/generator_perwindow_test.go`

- [ ] **Step 1: Write test for per-window generation**

```go
// Add to sim/workload/generator_perwindow_test.go
func TestGenerateRequestsForWindow(t *testing.T) {
	t.Run("single window with proportional allocation", func(t *testing.T) {
		clients := []ClientSpec{
			{
				ID: "test-client",
				TenantID: "tenant-1",
				SLOClass: "standard",
				RateFraction: 1.0,
				Streaming: true,
				Arrival: ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs: 0,
							EndUs: 10000000,  // 10 seconds
							TraceRate: ptr(100.0),
							Arrival: &ArrivalSpec{Process: "gamma", Shape: ptr(2.0), Scale: ptr(50000.0)},
							InputDist: &DistSpec{Type: "constant", Params: map[string]float64{"value": 200}},
							OutputDist: &DistSpec{Type: "constant", Params: map[string]float64{"value": 75}},
						},
					},
				},
			},
		}

		aggregateRate := 100.0
		rng := rand.New(rand.NewSource(42))

		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, aggregateRate, rng)

		// Should generate requests
		assert.Greater(t, len(requests), 0, "should generate requests")

		// Check window-specific distributions were used
		for _, req := range requests {
			assert.Len(t, req.InputTokens, 200, "should use window's input dist (200 tokens)")
			assert.Len(t, req.OutputTokens, 75, "should use window's output dist (75 tokens)")
		}

		// Check all requests within window bounds
		for _, req := range requests {
			assert.GreaterOrEqual(t, req.ArrivalTime, window.StartUs)
			assert.Less(t, req.ArrivalTime, window.EndUs)
		}

		// Check metadata
		assert.Equal(t, "test-client", requests[0].ClientID)
		assert.Equal(t, "tenant-1", requests[0].TenantID)
		assert.Equal(t, "standard", requests[0].SLOClass)
		assert.True(t, requests[0].Streaming)
	})

	t.Run("IAT rescaling achieves target rate", func(t *testing.T) {
		clients := []ClientSpec{
			{
				ID: "test-client",
				RateFraction: 1.0,
				Arrival: ArrivalSpec{Process: "poisson"},
				InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 10}},
				Lifecycle: &LifecycleSpec{
					Windows: []ActiveWindow{
						{
							StartUs: 0,
							EndUs: 10000000,  // 10 seconds
							TraceRate: ptr(50.0),
						},
					},
				},
			},
		}

		aggregateRate := 50.0
		rng := rand.New(rand.NewSource(123))

		window := clients[0].Lifecycle.Windows[0]
		requests := generateRequestsForWindow(clients[0], window, clients, aggregateRate, rng)

		// Check achieved rate is close to target (50 req/s for 10s = 500 requests)
		windowDurationSec := float64(window.EndUs-window.StartUs) / 1e6
		expectedRequests := int(aggregateRate * windowDurationSec)

		// Allow ±5% tolerance due to rounding and edge effects
		assert.InDelta(t, expectedRequests, len(requests), float64(expectedRequests)*0.05)
	})
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestGenerateRequestsForWindow -v`
Expected: FAIL with "undefined: generateRequestsForWindow"

- [ ] **Step 3: Implement generateRequestsForWindow** (part 1: signature and setup)

```go
// Add to sim/workload/generator.go (before computeProportionalRate)

// generateRequestsForWindow generates requests for a single lifecycle window
// with ServeGen-compatible proportional rate allocation and IAT rescaling.
func generateRequestsForWindow(
	client ClientSpec,
	window ActiveWindow,
	allClients []ClientSpec,
	aggregateRate float64,
	rng *rand.Rand,
) []*sim.Request {
	// Step 1: Resolve parameters with fallback
	arrival, inputDist, outputDist, _ := resolveWindowParameters(client, window)

	// Step 2: Compute allocated rate for this window
	windowTargetRate := computeProportionalRate(client, window, allClients, aggregateRate)
	if windowTargetRate <= 0 {
		return nil
	}

	windowDurationUs := window.EndUs - window.StartUs
	windowDurationSec := float64(windowDurationUs) / 1e6

	// Step 3: Determine number of requests
	expectedRequests := windowTargetRate * windowDurationSec
	numRequests := int(math.Ceil(expectedRequests))
	if numRequests == 0 {
		return nil
	}

	// Step 4: Create samplers
	arrivalSampler := NewArrivalSampler(arrival, windowTargetRate/1e6)
	inputSampler, err := NewLengthSampler(inputDist)
	if err != nil {
		logrus.Warnf("generateRequestsForWindow: client %q input dist: %v", client.ID, err)
		return nil
	}
	outputSampler, err := NewLengthSampler(outputDist)
	if err != nil {
		logrus.Warnf("generateRequestsForWindow: client %q output dist: %v", client.ID, err)
		return nil
	}

	// Step 5: Sample IATs using shape/scale (for CV)
	iats := make([]int64, numRequests)
	for i := 0; i < numRequests; i++ {
		iats[i] = arrivalSampler.SampleIAT(rng)
	}

	// Step 6: Rescale IATs to match target duration (ServeGen parity)
	iats = rescaleIATsToMatchDuration(iats, windowDurationUs)

	// Step 7: Generate requests with window-specific distributions
	requests := make([]*sim.Request, 0, numRequests)
	currentTime := window.StartUs

	for i := 0; i < numRequests; i++ {
		currentTime += iats[i]

		// Stop if we exceed window boundary
		if currentTime >= window.EndUs {
			break
		}

		// Sample token lengths
		inputLen := inputSampler.Sample(rng)
		outputLen := outputSampler.Sample(rng)
		inputTokens := sim.GenerateRandomTokenIDs(rng, inputLen)
		outputTokens := sim.GenerateRandomTokenIDs(rng, outputLen)

		req := &sim.Request{
			ID:               "", // Assigned later in merge+sort
			ArrivalTime:      currentTime,
			InputTokens:      inputTokens,
			OutputTokens:     outputTokens,
			MaxOutputLen:     outputLen,
			State:            sim.StateQueued,
			TenantID:         client.TenantID,
			SLOClass:         client.SLOClass,
			Model:            client.Model,
			ClientID:         client.ID,
			Streaming:        client.Streaming,
			Deadline:         0,  // Set by caller if needed
		}
		requests = append(requests, req)
	}

	return requests
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestGenerateRequestsForWindow -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_perwindow_test.go
git commit -m "feat(workload): implement per-window request generation

Add generateRequestsForWindow with full ServeGen parity:
- Proportional rate allocation across co-active windows
- IAT rescaling for exact rate matching
- Per-window arrival parameters (CV preservation)
- Per-window token distributions

Matches ServeGen construct.py:175-280 semantics.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Integrate Time-Varying Generator

**Files:**
- Modify: `sim/workload/generator.go` (modify GenerateRequests function)
- Test: `sim/workload/generator_perwindow_test.go`

- [ ] **Step 1: Write integration test**

```go
// Add to sim/workload/generator_perwindow_test.go
func TestGenerateRequests_TimeVaryingWorkload(t *testing.T) {
	t.Run("detects per-window parameters and routes to time-varying generator", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version: "2",
			AggregateRate: 100,
			Seed: 42,
			Clients: []ClientSpec{
				{
					ID: "client-1",
					TenantID: "tenant-1",
					SLOClass: "standard",
					RateFraction: 1.0,
					Streaming: true,
					Arrival: ArrivalSpec{Process: "poisson"},
					InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					Lifecycle: &LifecycleSpec{
						Windows: []ActiveWindow{
							{
								StartUs: 0,
								EndUs: 10000000,
								TraceRate: ptr(100.0),  // Per-window parameter present
							},
						},
					},
				},
			},
		}

		requests, err := GenerateRequests(spec, 10000000, 0)
		require.NoError(t, err)

		// Should generate requests
		assert.Greater(t, len(requests), 0)

		// Check requests are within window
		for _, req := range requests {
			assert.GreaterOrEqual(t, req.ArrivalTime, int64(0))
			assert.Less(t, req.ArrivalTime, int64(10000000))
		}

		// Check IDs are assigned
		assert.NotEmpty(t, requests[0].ID)

		// Check requests are sorted by arrival time
		for i := 1; i < len(requests); i++ {
			assert.LessOrEqual(t, requests[i-1].ArrivalTime, requests[i].ArrivalTime)
		}
	})

	t.Run("falls back to static generator when no per-window params", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version: "2",
			AggregateRate: 100,
			Seed: 42,
			Clients: []ClientSpec{
				{
					ID: "client-1",
					TenantID: "tenant-1",
					RateFraction: 1.0,
					Arrival: ArrivalSpec{Process: "poisson"},
					InputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 100}},
					OutputDist: DistSpec{Type: "constant", Params: map[string]float64{"value": 50}},
					// No lifecycle windows - should use static generator
				},
			},
		}

		requests, err := GenerateRequests(spec, 10000000, 0)
		require.NoError(t, err)

		// Should generate requests using existing static generator
		assert.Greater(t, len(requests), 0)
	})
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestGenerateRequests_TimeVaryingWorkload -v`
Expected: FAIL (requests won't be generated yet because detection/routing not implemented)

- [ ] **Step 3: Add detection helper**

```go
// Add to sim/workload/generator.go (before GenerateRequests function)

// hasPerWindowParameters checks if any client has per-window parameter overrides.
func hasPerWindowParameters(clients []ClientSpec) bool {
	for _, client := range clients {
		if client.Lifecycle == nil {
			continue
		}
		for _, window := range client.Lifecycle.Windows {
			if window.TraceRate != nil || window.Arrival != nil ||
			   window.InputDist != nil || window.OutputDist != nil {
				return true
			}
		}
	}
	return false
}
```

- [ ] **Step 4: Add time-varying generator wrapper**

```go
// Add to sim/workload/generator.go (after hasPerWindowParameters)

// generateTimeVaryingRequests generates requests for workloads with per-window
// parameters, using ServeGen-compatible proportional allocation and IAT rescaling.
func generateTimeVaryingRequests(
	spec *WorkloadSpec,
	horizon int64,
	maxRequests int64,
	allClients []ClientSpec,
	rng *rand.Rand,
) ([]*sim.Request, error) {
	var allRequests []*sim.Request

	// Generate requests for each client's windows
	for i := range allClients {
		client := &allClients[i]

		if client.Lifecycle == nil || len(client.Lifecycle.Windows) == 0 {
			// Client has no lifecycle windows - skip
			// (Could extend to support always-on clients mixed with windowed clients)
			continue
		}

		// Create per-client RNG for determinism
		clientSeed := rng.Int63()
		clientRNG := newRandFromSeed(clientSeed)

		// Generate requests for each window
		for _, window := range client.Lifecycle.Windows {
			// Skip windows outside horizon
			if window.StartUs >= horizon {
				break
			}
			if window.EndUs <= 0 {
				continue
			}

			windowRequests := generateRequestsForWindow(
				*client, window, allClients, spec.AggregateRate, clientRNG,
			)
			allRequests = append(allRequests, windowRequests...)
		}
	}

	// Sort all requests by arrival time
	sort.Slice(allRequests, func(i, j int) bool {
		return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
	})

	// Apply maxRequests cap if specified
	if maxRequests > 0 && int64(len(allRequests)) > maxRequests {
		allRequests = allRequests[:maxRequests]
	}

	// Assign sequential IDs
	for i, req := range allRequests {
		req.ID = fmt.Sprintf("%d", i)
	}

	return allRequests, nil
}
```

- [ ] **Step 5: Modify GenerateRequests to route to time-varying generator**

```go
// In sim/workload/generator.go, find GenerateRequests function (around line 18)
// Add detection and routing logic after line 89 (after generatePrefixTokens)

// Add this block right after: prefixes := generatePrefixTokens(allClients, workloadRNG)

// Check if any client has per-window parameters
if hasPerWindowParameters(allClients) {
	// Route to time-varying generator
	return generateTimeVaryingRequests(spec, horizon, maxRequests, allClients, workloadRNG)
}

// Continue with existing static generator for backward compatibility
```

- [ ] **Step 6: Run test to verify it passes**

Run: `go test ./sim/workload -run TestGenerateRequests_TimeVaryingWorkload -v`
Expected: PASS

- [ ] **Step 7: Run all existing tests to ensure backward compatibility**

Run: `go test ./sim/workload -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add sim/workload/generator.go sim/workload/generator_perwindow_test.go
git commit -m "feat(workload): integrate time-varying request generator

Add hasPerWindowParameters detection and generateTimeVaryingRequests
wrapper. Routes to time-varying generator when per-window params present,
falls back to static generator for backward compatibility.

Full ServeGen parity achieved: proportional allocation, IAT rescaling,
per-window distributions, multi-window clients.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Rewrite ServeGen Converter (Part 1: Dataset Loader)

**Files:**
- Modify: `sim/workload/servegen.go` (add loadServeGenDatasetAllWindows)
- Test: `sim/workload/servegen_temporal_test.go` (new file)

- [ ] **Step 1: Write test for dataset loader**

```go
// Create sim/workload/servegen_temporal_test.go
package workload

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoadServeGenDatasetAllWindows(t *testing.T) {
	// Create temporary test dataset
	tmpDir := t.TempDir()
	datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")

	dataset := map[string]map[string]string{
		"0": {
			"input_tokens": "{100: 0.5, 200: 0.5}",
			"output_tokens": "{50: 0.7, 100: 0.3}",
		},
		"600": {
			"input_tokens": "{150: 0.4, 250: 0.6}",
			"output_tokens": "{75: 0.8, 150: 0.2}",
		},
		"1200": {
			"input_tokens": "{}",  // Empty window - should be skipped
			"output_tokens": "{}",
		},
	}

	data, err := json.Marshal(dataset)
	require.NoError(t, err)
	err = os.WriteFile(datasetPath, data, 0644)
	require.NoError(t, err)

	// Load all windows
	sgConfig := &ServeGenDataSpec{}
	result, err := loadServeGenDatasetAllWindows(datasetPath, sgConfig)
	require.NoError(t, err)

	// Should load 2 non-empty windows
	assert.Len(t, result, 2)

	// Check window 0
	assert.Contains(t, result, 0)
	assert.Len(t, result[0].inputPDF, 2)
	assert.InDelta(t, 0.5, result[0].inputPDF[100], 0.001)
	assert.InDelta(t, 0.5, result[0].inputPDF[200], 0.001)

	// Check window 600
	assert.Contains(t, result, 600)
	assert.Len(t, result[600].inputPDF, 2)
	assert.InDelta(t, 0.4, result[600].inputPDF[150], 0.001)

	// Empty window should be skipped
	assert.NotContains(t, result, 1200)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestLoadServeGenDatasetAllWindows -v`
Expected: FAIL with "undefined: loadServeGenDatasetAllWindows"

- [ ] **Step 3: Implement loadServeGenDatasetAllWindows**

```go
// Add to sim/workload/servegen.go (before loadServeGenChunk function)

// datasetWindow holds PDFs for a single timestamp.
type datasetWindow struct {
	inputPDF  map[int]float64
	outputPDF map[int]float64
}

// loadServeGenDatasetAllWindows loads per-window token distributions from
// a ServeGen dataset JSON file. Returns map[timestamp] -> {inputPDF, outputPDF}.
// Skips empty windows (represented as "{}" in JSON).
func loadServeGenDatasetAllWindows(path string, sgConfig *ServeGenDataSpec) (map[int]datasetWindow, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading dataset: %w", err)
	}

	// Parse JSON: map[timestamp_str] -> {input_tokens: "...", output_tokens: "..."}
	var raw map[string]map[string]string
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("parsing dataset JSON: %w", err)
	}

	result := make(map[int]datasetWindow)

	for tsStr, window := range raw {
		timestamp, err := strconv.Atoi(tsStr)
		if err != nil {
			logrus.Warnf("loadServeGenDatasetAllWindows: skipping non-numeric key %q", tsStr)
			continue
		}

		// Filter by span
		if sgConfig.SpanStart > 0 && timestamp < sgConfig.SpanStart {
			continue
		}
		if sgConfig.SpanEnd > 0 && timestamp >= sgConfig.SpanEnd {
			continue
		}

		// Parse PDFs
		inputPDFStr := window["input_tokens"]
		outputPDFStr := window["output_tokens"]

		// Skip empty windows
		if inputPDFStr == "" || inputPDFStr == "{}" ||
		   outputPDFStr == "" || outputPDFStr == "{}" {
			continue
		}

		inputPDF, err := parseServeGenPDF(inputPDFStr)
		if err != nil {
			return nil, fmt.Errorf("parsing input PDF at timestamp %d: %w", timestamp, err)
		}

		outputPDF, err := parseServeGenPDF(outputPDFStr)
		if err != nil {
			return nil, fmt.Errorf("parsing output PDF at timestamp %d: %w", timestamp, err)
		}

		result[timestamp] = datasetWindow{
			inputPDF:  inputPDF,
			outputPDF: outputPDF,
		}
	}

	return result, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestLoadServeGenDatasetAllWindows -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sim/workload/servegen.go sim/workload/servegen_temporal_test.go
git commit -m "feat(workload): add per-window dataset loader for ServeGen

Implement loadServeGenDatasetAllWindows to load token distributions
for all 10-minute windows from ServeGen dataset JSON. Skips empty
windows and applies span filtering. Returns map[timestamp]->PDFs.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 8: Rewrite ServeGen Converter (Part 2: Full Converter)

**Files:**
- Modify: `sim/workload/servegen.go` (rewrite loadServeGenChunk)
- Test: `sim/workload/servegen_temporal_test.go`

- [ ] **Step 1: Write test for full converter**

```go
// Add to sim/workload/servegen_temporal_test.go
func TestLoadServeGenChunk_TemporalPreservation(t *testing.T) {
	// Create temporary test data
	tmpDir := t.TempDir()

	// Write trace CSV
	tracePath := filepath.Join(tmpDir, "chunk-test-trace.csv")
	traceData := `0,0,0,,0,0
600,10.5,0.95,Gamma,1.1,0.04
1200,22.8,1.02,Gamma,0.96,0.05
1800,0,0,,0,0
2400,15.3,0.88,Gamma,1.3,0.03`

	err := os.WriteFile(tracePath, []byte(traceData), 0644)
	require.NoError(t, err)

	// Write dataset JSON
	datasetPath := filepath.Join(tmpDir, "chunk-test-dataset.json")
	dataset := map[string]map[string]string{
		"600": {
			"input_tokens": "{100: 0.6, 200: 0.4}",
			"output_tokens": "{50: 0.8, 100: 0.2}",
		},
		"1200": {
			"input_tokens": "{150: 0.5, 250: 0.5}",
			"output_tokens": "{75: 0.7, 150: 0.3}",
		},
		"2400": {
			"input_tokens": "{120: 0.7, 220: 0.3}",
			"output_tokens": "{60: 0.9, 120: 0.1}",
		},
	}
	datasetJSON, err := json.Marshal(dataset)
	require.NoError(t, err)
	err = os.WriteFile(datasetPath, datasetJSON, 0644)
	require.NoError(t, err)

	// Load chunk
	sgConfig := &ServeGenDataSpec{}
	client, err := loadServeGenChunk("test", tracePath, datasetPath, sgConfig)
	require.NoError(t, err)
	require.NotNil(t, client)

	// Check client metadata
	assert.Equal(t, "servegen-chunk-test", client.ID)
	assert.Equal(t, "chunk-test", client.TenantID)
	assert.Equal(t, 1.0, client.RateFraction)
	assert.Equal(t, "standard", client.SLOClass)
	assert.True(t, client.Streaming)

	// Check lifecycle windows
	require.NotNil(t, client.Lifecycle)
	require.Len(t, client.Lifecycle.Windows, 3, "should have 3 active windows (skip rate=0)")

	// Check window 1 (timestamp 600)
	w1 := client.Lifecycle.Windows[0]
	assert.Equal(t, int64(600*1e6), w1.StartUs)
	assert.Equal(t, int64(1200*1e6), w1.EndUs)
	require.NotNil(t, w1.TraceRate)
	assert.Equal(t, 10.5, *w1.TraceRate)
	require.NotNil(t, w1.Arrival)
	assert.Equal(t, "gamma", w1.Arrival.Process)
	assert.Equal(t, 1.1, *w1.Arrival.Shape)
	assert.Equal(t, 40000.0, *w1.Arrival.Scale)  // 0.04s * 1e6 = 40000µs
	require.NotNil(t, w1.InputDist)
	assert.Equal(t, "empirical", w1.InputDist.Type)
	assert.InDelta(t, 0.6, w1.InputDist.Params["100"], 0.001)

	// Check window 2 (timestamp 1200) - different distribution
	w2 := client.Lifecycle.Windows[1]
	assert.Equal(t, int64(1200*1e6), w2.StartUs)
	assert.NotNil(t, w2.TraceRate)
	assert.Equal(t, 22.8, *w2.TraceRate)
	assert.InDelta(t, 0.5, w2.InputDist.Params["150"], 0.001)  // Different dist

	// Check window 3 (timestamp 2400)
	w3 := client.Lifecycle.Windows[2]
	assert.Equal(t, int64(2400*1e6), w3.StartUs)
	assert.NotNil(t, w3.TraceRate)
	assert.Equal(t, 15.3, *w3.TraceRate)
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `go test ./sim/workload -run TestLoadServeGenChunk_TemporalPreservation -v`
Expected: FAIL (current loadServeGenChunk only returns single window with peak rate)

- [ ] **Step 3: Rewrite loadServeGenChunk**

```go
// In sim/workload/servegen.go, replace the entire loadServeGenChunk function
// (currently around lines 124-194) with:

func loadServeGenChunk(chunkID, tracePath, datasetPath string, sgConfig *ServeGenDataSpec) (*ClientSpec, error) {
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

		// Skip inactive windows (rate = 0)
		if row.rate <= 0 {
			continue
		}

		// Get distributions for this timestamp
		dataset, ok := datasetByTimestamp[int(row.startTimeSec)]
		if !ok {
			logrus.Warnf("No dataset for chunk %s at t=%.0f, skipping window", chunkID, row.startTimeSec)
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

		// Client-level defaults (unused since all windows have overrides)
		Arrival:    ArrivalSpec{Process: "poisson"},
		InputDist:  DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 512, "stddev": 128}},
		OutputDist: DistSpec{Type: "gaussian", Params: map[string]float64{"mean": 128, "stddev": 32}},
	}

	return client, nil
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `go test ./sim/workload -run TestLoadServeGenChunk_TemporalPreservation -v`
Expected: PASS

- [ ] **Step 5: Run all servegen tests**

Run: `go test ./sim/workload -run ServeGen -v`
Expected: All PASS (some existing tests may need updates if they check window count)

- [ ] **Step 6: Commit**

```bash
git add sim/workload/servegen.go sim/workload/servegen_temporal_test.go
git commit -m "feat(workload): rewrite ServeGen converter for temporal parity

Rewrite loadServeGenChunk to preserve all active windows:
- Parse all trace rows (not just peak)
- Populate per-window trace_rate, arrival, distributions
- Create one ClientSpec per chunk with multiple windows
- Skip inactive windows (rate=0)

Achieves full temporal preservation for issue #1124.
m-mid folder: 88 clients with ~50-200 windows each.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 9: End-to-End Integration Test

**Files:**
- Test: `sim/workload/servegen_temporal_test.go`

- [ ] **Step 1: Write end-to-end test with real ServeGen data**

```go
// Add to sim/workload/servegen_temporal_test.go
func TestServeGenConversion_E2E(t *testing.T) {
	// Skip if ServeGen data not available
	if _, err := os.Stat("../../ServeGen/data/language/m-mid/chunk-8-trace.csv"); os.IsNotExist(err) {
		t.Skip("ServeGen data not available")
	}

	t.Run("convert and generate from real chunk-8", func(t *testing.T) {
		spec := &WorkloadSpec{
			Version: "2",
			AggregateRate: 150,
			Seed: 42,
			ServeGenData: &ServeGenDataSpec{
				Path: "../../ServeGen/data/language/m-mid",
			},
		}

		// Load ServeGen data (populates spec.Clients)
		err := loadServeGenData(spec)
		require.NoError(t, err)

		// Find chunk-8
		var chunk8 *ClientSpec
		for i := range spec.Clients {
			if spec.Clients[i].ID == "servegen-chunk-8" {
				chunk8 = &spec.Clients[i]
				break
			}
		}
		require.NotNil(t, chunk8, "chunk-8 should be loaded")

		// Check chunk-8 has multiple windows
		require.NotNil(t, chunk8.Lifecycle)
		assert.Greater(t, len(chunk8.Lifecycle.Windows), 5, "chunk-8 should have multiple windows")

		// Check per-window parameters exist
		for i, window := range chunk8.Lifecycle.Windows {
			assert.NotNil(t, window.TraceRate, "window %d should have trace_rate", i)
			assert.NotNil(t, window.Arrival, "window %d should have arrival", i)
			assert.NotNil(t, window.InputDist, "window %d should have input dist", i)
			assert.NotNil(t, window.OutputDist, "window %d should have output dist", i)
		}

		// Check distributions vary across windows (first vs last)
		if len(chunk8.Lifecycle.Windows) >= 2 {
			w1 := chunk8.Lifecycle.Windows[0]
			w2 := chunk8.Lifecycle.Windows[len(chunk8.Lifecycle.Windows)-1]

			// Different trace rates
			assert.NotEqual(t, *w1.TraceRate, *w2.TraceRate, "windows should have different rates")

			// Different distributions
			assert.NotEqual(t, w1.InputDist.Params, w2.InputDist.Params, "windows should have different input dists")
		}

		// Generate requests (short horizon for test speed)
		requests, err := GenerateRequests(spec, 600000000, 0)  // 10 minutes
		require.NoError(t, err)

		// Should generate requests
		assert.Greater(t, len(requests), 0)

		// Check requests have correct metadata
		chunk8Requests := 0
		for _, req := range requests {
			if req.ClientID == "servegen-chunk-8" {
				chunk8Requests++
				assert.Equal(t, "chunk-8", req.TenantID)
				assert.Equal(t, "standard", req.SLOClass)
				assert.True(t, req.Streaming)
			}
		}
		assert.Greater(t, chunk8Requests, 0, "should generate requests from chunk-8")
	})

	t.Run("multi-chunk proportional allocation", func(t *testing.T) {
		// Test with small subset of chunks to verify proportional allocation
		tmpDir := t.TempDir()

		// Create chunk-2 (trace rate 15.2)
		chunk2Trace := `600,15.2,0.9,Gamma,1.2,0.04`
		chunk2Dataset := map[string]map[string]string{
			"600": {
				"input_tokens": "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
		}
		os.WriteFile(filepath.Join(tmpDir, "chunk-2-trace.csv"), []byte(chunk2Trace), 0644)
		d2, _ := json.Marshal(chunk2Dataset)
		os.WriteFile(filepath.Join(tmpDir, "chunk-2-dataset.json"), d2, 0644)

		// Create chunk-8 (trace rate 22.5)
		chunk8Trace := `600,22.5,1.0,Gamma,1.0,0.04`
		chunk8Dataset := map[string]map[string]string{
			"600": {
				"input_tokens": "{100: 1.0}",
				"output_tokens": "{50: 1.0}",
			},
		}
		os.WriteFile(filepath.Join(tmpDir, "chunk-8-trace.csv"), []byte(chunk8Trace), 0644)
		d8, _ := json.Marshal(chunk8Dataset)
		os.WriteFile(filepath.Join(tmpDir, "chunk-8-dataset.json"), d8, 0644)

		// Load spec
		spec := &WorkloadSpec{
			Version: "2",
			AggregateRate: 150,
			Seed: 42,
			ServeGenData: &ServeGenDataSpec{Path: tmpDir},
		}
		err := loadServeGenData(spec)
		require.NoError(t, err)

		// Generate requests
		requests, err := GenerateRequests(spec, 1200000000, 0)  // 20 minutes
		require.NoError(t, err)

		// Count requests per chunk
		counts := make(map[string]int)
		for _, req := range requests {
			counts[req.ClientID]++
		}

		// Check proportional allocation: 15.2:22.5 ≈ 40%:60%
		total := counts["servegen-chunk-2"] + counts["servegen-chunk-8"]
		chunk2Ratio := float64(counts["servegen-chunk-2"]) / float64(total)
		chunk8Ratio := float64(counts["servegen-chunk-8"]) / float64(total)

		expectedChunk2Ratio := 15.2 / (15.2 + 22.5)
		expectedChunk8Ratio := 22.5 / (15.2 + 22.5)

		assert.InDelta(t, expectedChunk2Ratio, chunk2Ratio, 0.05)
		assert.InDelta(t, expectedChunk8Ratio, chunk8Ratio, 0.05)
	})
}
```

- [ ] **Step 2: Run test to verify it passes**

Run: `go test ./sim/workload -run TestServeGenConversion_E2E -v`
Expected: PASS (or SKIP if ServeGen data not available)

- [ ] **Step 3: Commit**

```bash
git add sim/workload/servegen_temporal_test.go
git commit -m "test(workload): add end-to-end ServeGen temporal parity tests

Add comprehensive e2e tests for ServeGen converter:
- Real chunk-8 data loading with multiple windows
- Per-window parameter validation
- Distribution variance across windows
- Multi-chunk proportional rate allocation

Validates full temporal preservation for issue #1124.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Task 10: Run Full Test Suite and Fix Regressions

**Files:**
- Various test files may need updates

- [ ] **Step 1: Run full workload test suite**

Run: `go test ./sim/workload -v`
Expected: Most tests PASS, may have some failures in existing ServeGen tests

- [ ] **Step 2: Identify and fix test regressions**

Common issues:
- Tests checking window count (now multiple windows vs one)
- Tests checking rate values (now per-window vs client-level)
- Tests checking YAML structure

For each failing test:
```go
// Example fix: Update assertion to check first window instead of client-level
// OLD:
// assert.Equal(t, 22.61, client.RateFraction)

// NEW:
// assert.NotNil(t, client.Lifecycle)
// assert.Greater(t, len(client.Lifecycle.Windows), 0)
// assert.NotNil(t, client.Lifecycle.Windows[0].TraceRate)
```

- [ ] **Step 3: Run tests again until all pass**

Run: `go test ./sim/workload -v`
Expected: ALL PASS

- [ ] **Step 4: Run tests for other packages to ensure no breakage**

Run: `go test ./sim/... -v`
Expected: ALL PASS

- [ ] **Step 5: Commit fixes**

```bash
git add sim/workload/*_test.go
git commit -m "fix(test): update tests for per-window parameters

Update existing ServeGen tests to handle multiple windows with
per-window parameters instead of single peak-rate window.

All tests now pass with temporal preservation changes.

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Self-Review Checklist

After completing all tasks, verify:

- [ ] **Spec coverage**: All requirements from design spec implemented
  - ✅ ActiveWindow extended with per-window fields
  - ✅ Parameter fallback logic
  - ✅ IAT rescaling
  - ✅ Proportional allocation
  - ✅ Per-window request generation
  - ✅ Time-varying generator integration
  - ✅ ServeGen converter rewrite
  - ✅ Comprehensive tests

- [ ] **No placeholders**: All code complete, no TODOs
- [ ] **Type consistency**: All function signatures match across tasks
- [ ] **Tests pass**: `go test ./sim/workload -v` shows ALL PASS
- [ ] **Backward compatibility**: Existing workloads without per-window params still work

---

## Validation

After implementation, validate with real ServeGen data:

```bash
# Convert m-mid folder
cd /Users/dipanwitaguhathakurta/Downloads/inference-sim-package/inference-sim
./blis convert servegen --path ServeGen/data/language/m-mid > servegen-m-mid.yaml

# Check output
echo "Clients: $(grep -c 'id: "servegen-chunk' servegen-m-mid.yaml)"
echo "Windows (total): $(grep -c 'start_us:' servegen-m-mid.yaml)"
# Expected: 88 clients, ~4400 windows

# Generate requests (short horizon for validation)
./blis run --workload-spec servegen-m-mid.yaml --horizon 3600000000 --output results.json

# Check results
cat results.json | jq '{injected: .injected_requests, completed: .completed_requests}'
# Verify INV-1: injected == completed (for short horizon)
```

---

## Success Criteria

✅ `go test ./sim/workload -v` - ALL PASS
✅ Convert m-mid: 88 clients with ~4400 lifecycle windows
✅ Per-window trace_rate, arrival, distributions present
✅ Proportional allocation matches ServeGen semantics
✅ IAT rescaling achieves target rates
✅ Backward compatibility: existing tests pass

