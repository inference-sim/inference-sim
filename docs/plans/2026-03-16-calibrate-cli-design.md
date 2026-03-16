# Design: `blis calibrate` CLI Command

**Status:** Approved
**Date:** 2026-03-16
**Closes:** #658
**Type:** New CLI command (thin wrapper over existing library)

## Context

`blis calibrate` is the final step in the observe/replay/calibrate loop (#652). The library code in `sim/workload/calibrate.go` is already complete. This design covers only the CLI glue layer.

## Command

```
blis calibrate --trace-header <path> --trace-data <path> --sim-results <path> --report <path>
               [--warmup-requests N] [--network-rtt-us N] [--network-bandwidth-mbps N]
```

## Design Decisions

### Sentinel defaults for optional numeric flags

`--warmup-requests` defaults to `-1` (sentinel = "auto from header"). When `-1`, the command reads `trace.Header.WarmUpRequests`. When `0`, it excludes nothing. This lets the user override the header value while still defaulting to it. Same pattern for `--network-rtt-us` (sentinel `int64(-1)`).

Rationale: cobra int flags cannot distinguish "not set" from "set to 0" without extra boolean flags. The sentinel avoids flag proliferation.

### Empty ConfigMatchInfo

Config matching (comparing sim parameters to trace header) is deferred. The CLI does not know what sim config was used — it only sees the results file. `ConfigMatchInfo{}` is set empty with a TODO comment.

### Data flow

1. Load TraceV2: `workload.LoadTraceV2(headerPath, dataPath)`
2. Read `[]workload.SimResult` from JSON file
3. Resolve warmup/RTT from sentinel → header → zero fallback
4. `workload.PrepareCalibrationPairs(trace.Records, simResults, &config)`
5. `workload.BuildCalibrationReport(pairs, &ConfigMatchInfo{})`
6. Marshal indented JSON → write to `--report`
7. Log summary (matched pairs, TTFT MAPE, E2E MAPE, quality ratings) to stderr

## Files Changed

| File | What |
|------|------|
| `cmd/calibrate.go` (new) | cobra command + Run function |
| `cmd/calibrate_test.go` (new) | 4 unit tests |

## Architecture Notes

- No new interfaces or module boundaries
- Only `cmd/` package touched — stays within cmd → sim/workload dependency direction
- `logrus.Fatalf` for all error exits (consistent with other `cmd/` commands)
- Closes #658; parent feature #652
