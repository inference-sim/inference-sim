# Saturation Analyzer Extension Guide

## Overview

Saturation analyzers detect when your system is overloaded. They classify the system into three states:

- **UNSATURATED**: System handling load comfortably
- **TRANSIENT_BACKLOG**: Temporary bursts or near capacity
- **PERSISTENTLY_SATURATED**: System can't keep up, backlog growing

The `SaturationAnalyzer` interface lets you plug in different detection algorithms.

## Current Implementation: BacklogDriftAnalyzer

The default analyzer watches how the request backlog changes over time:

- **How it works**: Splits observation into time windows, tracks active requests, fits a trend line
- **Classification**:
  - Growing backlog → PERSISTENTLY_SATURATED
  - Spike but stable trend → TRANSIENT_BACKLOG
  - Flat or declining → UNSATURATED

**Configuration options**:
- `--saturation-window`: Time window size (default: 60s)
- `--saturation-min-windows`: Minimum windows needed (default: 5)
- `--saturation-peak-ratio`: Burst threshold (default: 2.0)
- `--saturation-peak-band`: Confidence band (default: 0.2)
- `--saturation-ci`: Statistical confidence (default: 0.95)

## Usage Examples

```bash
# Low load - should show UNSATURATED
./blis run --model qwen/qwen3-14b --rate 5 --num-requests 500 \
  --saturation-report sat.json

# Medium load - should show TRANSIENT_BACKLOG
# Note: Use --horizon to stop before requests drain
./blis run --model qwen/qwen3-14b --rate 100 --num-requests 10000 \
  --horizon 60000000 --saturation-report sat.json

# High load - should show PERSISTENTLY_SATURATED
./blis run --model qwen/qwen3-14b --rate 500 --num-requests 100000 \
  --horizon 120000000 --saturation-report sat.json
```

**Important**: For PERSISTENTLY_SATURATED detection, use `--horizon` to end observation while load is active. If you let all requests complete, the system drains and looks unsaturated at the end.

## Classification Progression

As you increase load, classifications progress:

| System Load | Classification | What You'll See |
|-------------|----------------|-----------------|
| < 50% capacity | UNSATURATED | Small, stable backlog |
| ≈ 100% capacity | TRANSIENT_BACKLOG | Bursts but recovers |
| >> capacity | PERSISTENTLY_SATURATED | Backlog keeps growing |

## Adding a New Analyzer

Want to detect saturation using different metrics (CPU, memory, latency)? Here's the process:

### 1. Create Your Analyzer

Create a new file `sim/workload/saturation_<name>.go`:

```go
package workload

type YourAnalyzer struct {
    // Your configuration
}

// Implement the SaturationAnalyzer interface
func (a *YourAnalyzer) Analyze(requests []*sim.Request, simEndUs int64) SaturationReport {
    // 1. Compute your metrics from requests
    // 2. Classify as UNSATURATED, TRANSIENT_BACKLOG, or PERSISTENTLY_SATURATED
    // 3. Return SaturationReport

    return SaturationReport{
        Classification: "...",
        Algorithm:      "your-analyzer",
        Note:           "Human-readable explanation",
        Recommendation: "Suggested action",
        AlgorithmData:  yourMetrics, // Your custom data structure
    }
}
```

### 2. Register in Factory

Add your analyzer to `sim/workload/saturation_analyzer.go`:

```go
func NewSaturationAnalyzer(algorithm string, config interface{}) (SaturationAnalyzer, error) {
    switch algorithm {
    case "backlog-drift":
        // ... existing code ...

    case "your-analyzer":  // Add this
        cfg, ok := config.(YourConfig)
        if !ok {
            return nil, fmt.Errorf("your-analyzer requires YourConfig, got %T", config)
        }
        return NewYourAnalyzer(cfg), nil

    default:
        return nil, fmt.Errorf("unknown saturation algorithm: %s", algorithm)
    }
}
```

### 3. Add CLI Flags (Optional)

If you need configuration flags, add them to `cmd/root.go`:

```go
// In variable declarations
var (
    saturationAlgorithm string  // Algorithm selector
    // ... your analyzer's config flags ...
)

// In flag registration
runCmd.Flags().StringVar(&saturationAlgorithm, "saturation-algorithm", "backlog-drift",
    "Saturation detection algorithm: backlog-drift, your-analyzer")
```

### 4. Write Tests

Test your analyzer in `sim/workload/saturation_<name>_test.go`:

```go
func TestYourAnalyzer_HighLoad(t *testing.T) {
    analyzer := NewYourAnalyzer(config)
    requests := /* create test requests */

    report := analyzer.Analyze(requests, simEndUs)

    if report.Classification != "PERSISTENTLY_SATURATED" {
        t.Errorf("Expected PERSISTENTLY_SATURATED, got %s", report.Classification)
    }
}
```

## Output Format

The `backlog-drift` analyzer (currently the only implemented analyzer) produces the following JSON report format:

```json
{
  "classification": "TRANSIENT_BACKLOG",
  "slope": -0.0000186,
  "slope_lower": -0.0000235,
  "slope_upper": -0.0000137,
  "initial_backlog": 0,
  "final_backlog": 0,
  "peak_in_flight": 468,
  "mean_in_flight": 211.5,
  "windows": [
    {
      "start_us": 0,
      "end_us": 60000000,
      "num_entered": 120,
      "num_left": 100,
      "active_start": 0,
      "active_end": 20,
      "delta_backlog": 20,
      "drain_ratio": 0.833
    }
  ],
  "note": "Peak/mean ratio (2.21) is borderline. Using positive slope as tiebreaker → TRANSIENT_BACKLOG.",
  "recommendation": "System experienced transient congestion (bursts without persistent growth). Monitor for recurring patterns. Consider increasing capacity if bursts are frequent."
}
```

**Key fields:**
- `classification`: One of `UNSATURATED`, `TRANSIENT_BACKLOG`, or `PERSISTENTLY_SATURATED`
- `slope`, `slope_lower`, `slope_upper`: Linear regression slope with 95% confidence interval (req/µs)
- `peak_in_flight`, `mean_in_flight`: Peak and mean active request counts across windows
- `windows`: Per-window metrics (see `WindowMetrics` for field definitions)
- `note`: Explanation of the classification decision
- `recommendation`: User-facing guidance for capacity planning

## When to Use Which Analyzer

| Analyzer Type | Best For | Example Use Case |
|---------------|----------|------------------|
| **backlog-drift** | Time-series workloads with sustained load | Production traffic monitoring |
| **utilization-based** | Short observations, resource limits | Quick capacity checks |
| **queue-depth** | Request-rate-focused workloads | API endpoint monitoring |
| **latency-based** | User-facing SLAs | Customer experience tracking |

## Design Principles

- **One method**: Implement just `Analyze()`
- **Config validated early**: Check parameters at creation time, not analysis time
- **Unified output**: Always return `SaturationReport`
- **Algorithm-specific data**: Put your custom metrics in `AlgorithmData` field
- **Forward compatible**: The `algorithm` field lets tools know how to parse `algorithm_data`

## Need Help?

Check existing implementations:
- `sim/workload/saturation.go` - BacklogDriftAnalyzer
- `sim/workload/saturation_analyzer.go` - Interface definition
- `sim/workload/saturation_test.go` - Example tests
