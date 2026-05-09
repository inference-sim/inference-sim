# PR1283: Document Scheduling Delay as Inherent Calibration Gap

**Goal:** Add a `### Known Gap: Scheduling Delay` section to `docs/guide/observe-replay-calibrate.md` explaining that `SchedulingDelay` is a simulator-internal diagnostic that cannot be calibrated against real server observations, and what users should do when they see high scheduling delay.

**Source:** https://github.com/inference-sim/inference-sim/issues/1283

**Closes:** #1283

**Tier:** Small (docs-only, 1 file, no behavioral change)

---

## Behavioral Contracts

**BC-1: Scheduling delay explanation present**
GIVEN a user reads `docs/guide/observe-replay-calibrate.md`,
WHEN they look at the calibration section,
THEN they find a clear explanation that `SchedulingDelay` is simulator-internal, not observable from real servers, and therefore not a separately calibratable metric.

**BC-2: Root-cause guidance present**
GIVEN a user sees high scheduling delay (e.g., P99 > 500ms — workload-dependent) in `blis run` or `blis replay` output and poor calibration MAPE,
WHEN they consult the calibration guide,
THEN they find actionable guidance: scheduling delay is already factored into E2E/TTFT; if aggregate E2E calibrates well, the scheduling model is correct; if high scheduling delay coincides with high MAPE, the root cause is likely queue buildup and the user should enable `--flow-control` (which adds explicit queue depth gating to better model servers with admission backpressure) and re-calibrate.

**BC-3: No Go code changes**
GIVEN the PR is documentation-only,
WHEN `go test ./...` is run,
THEN all tests pass unchanged (0 new failures).

---

## Tasks

### Task 1: Add `### Known Gap: Scheduling Delay` to observe-replay-calibrate.md

**Location:** `docs/guide/observe-replay-calibrate.md`, after the `**\`known_limitations\`**` field definition (line 327) and its trailing blank line (line 328), before the `---` horizontal rule (line 329) that precedes `## Worked Example`. This places the new section as the final subsection under `## blis calibrate`, still inside that section, not after the separator.

**Content to add:**

```markdown
### Known Gap: Scheduling Delay

!!! note "Scheduling delay cannot be calibrated"
    `blis run` and `blis replay` report **scheduling delay** — the time a request
    waited in the simulator's internal queue before being selected for batch execution.
    This appears in per-request `RequestMetrics.SchedulingDelay` and in aggregate
    P99 output. `blis observe` **cannot** record scheduling delay because real inference
    servers do not expose per-request queue wait time through their HTTP APIs. A
    client sees only TTFT and E2E; the server never reveals how long a request queued
    before execution began. This gap is inherent and cannot be closed without
    server-side instrumentation outside BLIS's scope.

**What this means for calibration:**

- Scheduling delay is already **factored into E2E and TTFT**. A real server's TTFT includes queue wait implicitly — it cannot be decomposed from execution time externally. Mathematically, `SchedulingDelay = schedule_time − arrival_time` and `TTFT ≥ SchedulingDelay` always (since token generation happens after scheduling). `blis calibrate` compares E2E and TTFT end-to-end; a good MAPE on those metrics means the scheduling model is correct, even without a direct scheduling delay match.
- **Do not expect** `SchedulingDelay` to appear as a separately calibratable field in the calibration report. It is a simulator-internal diagnostic only.
- **When scheduling delay matters:** If `blis run` shows high scheduling delay (e.g., P99 > 500ms — exact thresholds are workload-dependent) **and** calibration MAPE is high, the root cause is likely queue buildup rather than the execution latency model. In this case, enable flow control, which adds explicit queue depth gating to better model servers with admission backpressure, then re-calibrate:

  ```bash
  ./blis run --model qwen/qwen3-14b --flow-control --saturation-detector utilization \
    --queue-depth-threshold 5 --kv-cache-util-threshold 0.8
  ```
```

**Steps:**
1. Read `docs/guide/observe-replay-calibrate.md` to confirm current line count and insertion point (after `known_limitations` description, before the `---` before `## Worked Example`).
2. Edit the file to insert the new section.
3. Verify no Go files are modified: `git diff --name-only` shows only the docs file.
4. Verify build still passes: `go build ./...`
5. Commit: `git add docs/guide/observe-replay-calibrate.md && git commit -m "docs(calibrate): document scheduling delay as inherent calibration gap (#1283)"`

---

## Sanity Checklist

- [ ] Section added after `Interpreting the Calibration Report`, before `## Worked Example`
- [ ] Uses MkDocs `!!! note` admonition syntax (consistent with other callouts in the file)
- [ ] Explains WHAT scheduling delay is (queue wait before batch execution)
- [ ] Explains WHY it cannot be calibrated (not exposed via HTTP API)
- [ ] Explains what to do instead (E2E/TTFT already incorporate it; `--flow-control` for high scheduling delay + high MAPE)
- [ ] No Go source files modified
- [ ] `go test ./...` passes (BC-3)
- [ ] Closes #1283 in commit message
