# Saturation Detector Extension Guide

## Overview

Post-hoc saturation **detectors** classify a completed run into one of three
levels:

- **STABLE**: system handling load comfortably.
- **BACKLOGGED**: queue building but not runaway (temporary bursts or near
  capacity).
- **OVERLOADED**: system can't keep up; backlog growing.

> **Not** the real-time flow-control `sim.SaturationDetector` (a `float64`
> admission-control signal). These post-hoc detectors live in `sim/saturation/`
> and implement the `saturation.Detector` interface for trace analysis. See the
> naming note at the top of `sim/saturation/detector.go`.

Every detector is **streaming** (#1515/#1516): it folds one reconstructed event
at a time into internal state and returns an evolving per-event verdict. There
is no batch `Analyze()`/`Classify()` path — the former post-hoc batch analysis
library in `sim/workload` (`AnalyzeBacklogDrift*`, `BacklogDriftReport`, the
`slope-based`/`drain-ratio` classifiers) was removed in #1547.

## The `Detector` interface

`sim/saturation/detector.go`:

```go
type Detector interface {
    Name() string       // stable identifier, also the trace/report key
    Observe(event Event) // fold one event into internal state
    Detect() Result      // current verdict from accumulated state
    Reset()             // return to initial state (reused across replay legs)
}
```

An `Event` is one of two types — an `Arrival` or a `Completion` — reconstructed
from completed request metrics (each request contributes an arrival at its
arrival time and a completion at `arrival + E2E`). A `Result` carries the
`Level`, a `Score`, a `Confidence`, and a free-form `Signals` map for
diagnostics.

The drive loop (`ReplayOneDetector` in `replay.go`) calls `Reset()` once, then
`Observe(e)` followed by `Detect()` for **every** event in the deterministic
`(timestamp, event-type, request-id)`-sorted stream, recording each verdict to
the sink. A detector may ignore an event type in `Observe` (the threshold
detector, for example, only accumulates completions); its verdict then simply
doesn't change on the events it ignores.

## Built-in detectors

| Name | Levels emitted | What it measures |
|---|---|---|
| **composite** | STABLE / BACKLOGGED / OVERLOADED | `max(rate_deficit, quartile-filtered latency_trend)` banded against a `1/√arrivals` noise floor, scaled by `sensitivity` (default 1.0). |
| **threshold** | STABLE / OVERLOADED | Mean E2E latency vs. a configurable threshold (default 5000ms). Binary — never emits BACKLOGGED. |
| **backlog-drift** | STABLE / BACKLOGGED / OVERLOADED | Online OLS slope of in-flight (`arrivals − completions`) over a trailing window, banded against `slope_k × noiseFloor` (default `slope_k` = 3.0). |

## Running detectors

```bash
# Single detector (streaming path, #1516)
./blis run --model qwen/qwen3-14b --detectors composite --saturation-report sat.json

# The whole roster, or a named subset, over ONE deterministic replay (bank, #1519)
./blis run --model qwen/qwen3-14b --detectors all --saturation-report sat.json
./blis run --model qwen/qwen3-14b --detectors composite,threshold --saturation-report sat.json
```

`--detectors` also works on `blis replay` (byte-identical to `run` for the same
trace, INV-13) and `blis observe` (same pipeline over real-server latencies).

### Tuning via `--saturation-config`

A strict-YAML file carries one optional block per detector. **Every detector has
at least one calibration knob**, and that is a correctness property rather than a
convenience: detector scores are comparable only when each detector has first been
calibrated to the same false-alarm rate, and a detector with no knob cannot be
moved onto that rate — it can only be disqualified. See `SaturationConfig` in
`sim/saturation/config.go`.

```yaml
# composite: the noise-floor multiplier. Larger => higher bar => fires less.
# 1.0 is the default and reproduces the historical unscaled floor exactly.
composite:
  sensitivity: 2.0

# threshold: the ThresholdDetector's single knob
threshold:
  threshold_ms: 8000

# backlog_drift: mirrors saturation.BacklogDriftConfig
backlog_drift:
  window_size_sec: 30      # whole seconds
  min_windows: 5
  peak_ratio: 2.0
  peak_ratio_band: 0.2
  confidence_ci: 0.95
  warmup_windows: 2
  tail_windows: 1
  saturated_drain_ratio: 0.95
  transient_drain_ratio: 0.98
  slope_k: 3.0             # BACKLOGGED/OVERLOADED boundary multiplier
```

**Calibrating a knob.** Every knob above trades sensitivity against false alarms,
and larger always means "fires less". To calibrate: run a workload you believe is
healthy, sweep the knob upward, and take the smallest value that produces no
alarm. Comparing two detectors is only meaningful once both have been calibrated
that way — otherwise you are comparing a strict detector against a lenient one.

Two bounds worth knowing:

- All multiplicative knobs are rejected below `1e-6`. A subnormal multiplier passes
  a naive "is it positive?" check but drives its product with the noise floor to
  zero, which decouples a detector's level from its score.
- `backlog_drift.slope_k <= 1` makes the BACKLOGGED band unsatisfiable (that band is
  `noiseFloor < slope <= slope_k×noiseFloor`), so the detector reports only STABLE
  and OVERLOADED. That is a legitimate "maximally severe" setting, but a sweep that
  crosses 1 is comparing a two-level detector against a three-level one.

Absent block = defaults; a partial block overrides only the fields it names; an
unknown key or out-of-range value errors naming the field. Block ownership is
enforced: a block whose owning detector is not among the selected `--detectors`
is a hard error (no silent drop), for both the single-detector and bank paths.

## Adding a new detector

### 1. Implement the `Detector` interface

Create `sim/saturation/<name>.go`:

```go
package saturation

type MyDetector struct {
    // streaming state, populated by Observe, read by Detect, cleared by Reset
}

func NewMyDetector( /* config */ ) Detector {
    return &MyDetector{ /* ... */ }
}

func (d *MyDetector) Name() string { return "my-detector" }

func (d *MyDetector) Observe(event Event) {
    // fold this event into internal state (you may ignore an event type)
}

func (d *MyDetector) Detect() Result {
    // classify from accumulated state; return STABLE on empty input (R20 — no panic)
    return Result{
        Level:      Stable,
        Score:      0,
        Confidence: 0,
        Signals:    map[string]float64{ /* diagnostics */ },
    }
}

func (d *MyDetector) Reset() {
    // return to initial state so the detector can be reused across replay legs
}
```

### 2. Register in the builder and the roster

Add a `case` to `buildDetector` in `sim/saturation/config.go`:

```go
case "my-detector":
    return NewMyDetector( /* resolved from cfg */ ), nil
```

If your detector is tunable, add a block type to `SaturationConfig`, resolve it
(mirror `resolveBacklogDriftConfig`: validate and return errors, never panic —
R6), and add **one row** to the `blockOwners()` table in
`sim/saturation/config.go`:

```go
{"my_detector", "my-detector", func(c SaturationConfig) bool { return c.MyDetector != nil }},
```

Both `checkBlockOwnership` (single-detector) and `checkBlockOwnershipSet` (bank)
derive from that one table, so a foreign block is rejected identically on both
paths and the two cannot drift apart. Do **not** hand-edit either function.

Validate the knob's value in the resolver, not only in the detector's
constructor: a constructor-side fallback would silently coerce a bad value
instead of reporting it (R1).

Add the name to `rosterOrder` in `sim/saturation/bank.go` so it is included in
`--detectors all`:

```go
var rosterOrder = []string{"composite", "threshold", "backlog-drift", "my-detector"}
```

The bank fans every event out to each selected detector in this fixed canonical
order, so CLI argument order never changes output (INV-6). Wire the name into
the CLI validation in `cmd` if a fresh single-name path is desired.

### 3. Write tests

Test the streaming contract in `sim/saturation/<name>_test.go`. Prefer
behavioral assertions over structural ones (BDD/TDD): assert that a growing
backlog eventually drives the verdict to OVERLOADED, that a draining backlog
returns to STABLE, and that zero events yields STABLE without panicking. Follow
the table-driven style used in `composite_test.go` / `threshold_test.go`.

## Output format

`--saturation-report` writes a single JSON object with a `final` detector→label
map (the last-window plurality verdict per detector, #1517) and a `trace` array
of per-event verdicts (one record per event, tagged by detector name). Map keys
are sorted so repeated runs are byte-identical (INV-6).

```json
{
  "final": { "composite": "STABLE" },
  "trace": [
    {
      "timestamp": 150000,
      "detector": "composite",
      "result": {
        "level": "STABLE",
        "score": 0.35,
        "confidence": 0.95,
        "signals": { "latency_trend": 0.12, "rate_deficit": 0.0 }
      }
    }
  ]
}
```

The same `final` map is also spliced onto the metrics JSON on stdout (a run
*without* `--detectors` stays byte-identical to the historical no-feature
output).

## Design principles

- **Four small methods**: `Name`/`Observe`/`Detect`/`Reset`. No batch path.
- **Streaming and causal**: consume events in order; never look ahead.
- **Empty input is STABLE, not a panic** (R20).
- **Config validated at build time**: `buildDetector` returns errors naming the
  offending field; the library boundary never panics on user config (R6).
- **Diagnostics in `Signals`**: put intermediate quantities (slopes, ratios,
  noise floors) there so the trace explains each verdict.

## Need help?

Reference implementations:

- `sim/saturation/detector.go` — the `Detector` interface + `Event`/`Result`.
- `sim/saturation/composite.go`, `threshold.go`, `backlog_drift.go` — the three
  built-in detectors.
- `sim/saturation/config.go` — `SaturationConfig`, `buildDetector`, the
  `blockOwners()` ownership table.
- `sim/saturation/bank.go` — the multi-detector `Bank` + `rosterOrder`.
- `sim/saturation/replay.go` — the single-detector drive loop and event
  reconstruction.
- `sim/saturation/reduce.go` — the final-label plurality reducer.
