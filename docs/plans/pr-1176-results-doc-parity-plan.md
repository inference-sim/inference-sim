# results.md Doc Parity Fix — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all six doc parity gaps between `docs/guide/results.md` and the actual CLI/JSON output of `blis run`.
**Source:** Issue #602 + comprehensive audit comment (April 2026).
**Closes:** Fixes #602.

## Behavioral Contracts

**BC-1: Anomaly Counters table complete**
- GIVEN a user reads the Anomaly Counters table in results.md
- WHEN they observe any anomaly counter in `blis run` stdout
- THEN every counter the CLI can print has a corresponding row in the table with description and action

**BC-2: Primary Metrics JSON table complete**
- GIVEN a user opens the `--metrics-path` JSON file
- WHEN they read any top-level field
- THEN every field in `MetricsOutput` (sim/metrics_utils.go:57) has a row in the Primary Metrics table

**BC-3: Per-request fields documented**
- GIVEN a user opens the `--metrics-path` JSON file and reads the `requests[]` array
- WHEN they inspect any field on a request entry
- THEN every field in `RequestMetrics` (sim/metrics_utils.go:13) is listed in a per-request section

**BC-4: Session Metrics section exists**
- GIVEN a user runs `blis run` with a multi-turn workload
- WHEN the `=== Session Metrics ===` section prints
- THEN results.md has a matching section explaining each line

**BC-5: PD Metrics section exists**
- GIVEN a user runs `blis run` with PD disaggregation enabled
- WHEN the `=== PD Metrics ===` section prints
- THEN results.md has a matching section explaining each line

**BC-6: per_model JSON accuracy corrected**
- GIVEN a user reads the Per-Model Metrics section
- WHEN they look for per-model data in the `--metrics-path` JSON file
- THEN the docs correctly state that per-model/per-tenant/session/PD metrics appear on stdout only, not in the JSON file

## Tasks

### Task 1: Fix Anomaly Counters table (BC-1)

**Files:** modify `docs/guide/results.md`

**Impl:** Replace the existing Anomaly Counters table (5 rows) with the complete 10-row version:

```markdown
| Counter | What It Means | Action |
|---------|--------------|--------|
| **Priority Inversions** | A lower-priority request was scheduled before a higher-priority one | Check scheduler choice — use `priority-fcfs` for SLO workloads |
| **HOL Blocking Events** | A long prefill blocked shorter requests | Enable chunked prefill: `--long-prefill-token-threshold 256` |
| **Rejected Requests (Admission)** | Admission policy rejected the request at cluster ingress | Check token bucket capacity or admission policy |
| **Shed (tier)** | Per-SLO-class breakdown of admission rejections under overload (sub-items under Rejected Requests) | Adjust `slo_priorities` or admission thresholds |
| **Rejected Requests (Routing)** | No routable instances for the request's model — all instances are `Loading` or `Draining` | Increase `initial_nodes`, reduce `loading_delay.mean`, or stagger drain operations |
| **Dropped Unservable** | Request exceeds `--max-model-len` context window or needs more KV blocks than exist | Check `--max-model-len` setting; increase `--total-kv-blocks` or reduce max input tokens |
| **Timed Out Requests** | Request exceeded its deadline before completing | Increase client `--timeout` or reduce load |
| **Length-Capped Requests** | Request was force-completed when it reached `MaxModelLen` tokens during decode | Expected if workloads push against `--max-model-len`; set `--max-model-len 0` to disable |
| **Gateway Queue Depth (horizon)** | Requests still waiting in the gateway queue when the simulation ended | Increase `--max-gateway-queue-depth` or reduce arrival rate |
| **Gateway Queue Shed** | Requests shed from the gateway queue because it was full | Increase `--max-gateway-queue-depth` or enable `--flow-control` with a saturation detector |
```

**Verify:** Check against `cmd/root.go:1635-1660` — every `fmt.Printf` in the anomaly block has a table row.

**Lint:** `golangci-lint run ./...` — no Go changes, should be clean.

**Commit:**
```bash
git add docs/guide/results.md
git commit -m "docs(results): add missing anomaly counter rows (BC-1)

Timed Out Requests, Length-Capped Requests, Shed (tier) breakdown,
Gateway Queue Depth, and Gateway Queue Shed were printed by the CLI
but absent from the Anomaly Counters table. Fixes #602.

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Expand Primary Metrics JSON table (BC-2)

**Files:** modify `docs/guide/results.md`

**Impl:** Replace the existing 9-row Primary Metrics table with the complete version covering all `MetricsOutput` fields (sim/metrics_utils.go:57-87):

```markdown
| Field | Unit | Description |
|-------|------|-------------|
| `instance_id` | string | Instance identifier (`"cluster"` for aggregate output) |
| `completed_requests` | count | Requests that finished before horizon |
| `still_queued` | count | Requests in the wait queue at horizon end |
| `still_running` | count | Requests in the running batch at horizon end |
| `injected_requests` | count | Total requests that entered the simulator |
| `total_input_tokens` | count | Sum of input tokens across all completed requests |
| `total_output_tokens` | count | Sum of output tokens across all completed requests |
| `vllm_estimated_duration_s` | s | Estimated vLLM wall-clock duration for the workload |
| `responses_per_sec` | req/s | Throughput (completed requests / effective duration) |
| `tokens_per_sec` | tok/s | Output token throughput |
| `e2e_mean_ms` | ms | Mean End-to-End latency |
| `e2e_p90_ms` | ms | 90th percentile E2E |
| `e2e_p95_ms` | ms | 95th percentile E2E |
| `e2e_p99_ms` | ms | 99th percentile E2E — tail latency |
| `ttft_mean_ms` | ms | Mean Time To First Token |
| `ttft_p90_ms` | ms | 90th percentile TTFT |
| `ttft_p95_ms` | ms | 95th percentile TTFT |
| `ttft_p99_ms` | ms | 99th percentile TTFT — tail latency |
| `itl_mean_ms` | ms | Mean Inter-Token Latency — streaming smoothness |
| `itl_p90_ms` | ms | 90th percentile ITL |
| `itl_p95_ms` | ms | 95th percentile ITL |
| `itl_p99_ms` | ms | 99th percentile ITL |
| `scheduling_delay_p99_ms` | ms | 99th percentile scheduling delay — queue wait time |
| `kv_allocation_failures` | count | KV cache allocation failures (omitted when zero) |
| `preemption_count` | count | Total preemption events (integer; see also KV Cache Metrics for the rate) |
| `dropped_unservable` | count | Requests dropped at enqueue due to context limit violations |
| `length_capped_requests` | count | Requests force-completed at `MaxModelLen` |
| `timed_out_requests` | count | Requests that exceeded their deadline |
| `requests` | array | Per-request detail records (omitted when empty; see [Per-Request Fields](#per-request-fields)) |
```

**Verify:** Compare against struct in `sim/metrics_utils.go:57-87` — every `json:"..."` tag has a row.

**Commit:**
```bash
git add docs/guide/results.md
git commit -m "docs(results): expand Primary Metrics JSON table to all MetricsOutput fields (BC-2)

Previous table documented 9 of 25+ fields. Now covers tokens_per_sec,
preemption_count, kv_allocation_failures, all percentiles (p90/p95/p99
for E2E, TTFT, ITL), request counters, and requests[] array reference.

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Add Per-Request Fields section (BC-3)

**Files:** modify `docs/guide/results.md`

**Impl:** After the Primary Metrics table, add a new `### Per-Request Fields` subsection:

```markdown
### Per-Request Fields

When the `requests` array is non-empty, each entry contains:

| Field | Unit | Description |
|-------|------|-------------|
| `requestID` | string | Unique request identifier |
| `arrived_at` | float64 | Arrival timestamp in ticks |
| `num_prefill_tokens` | count | Input (prompt) token count |
| `num_decode_tokens` | count | Output token count |
| `ttft_ms` | ms | Time To First Token for this request |
| `itl_ms` | ms | Mean Inter-Token Latency for this request |
| `e2e_ms` | ms | End-to-End latency for this request |
| `scheduling_delay_ms` | ms | Time spent in the wait queue before first scheduling |
| `slo_class` | string | SLO class (`critical`, `standard`, `batch`, etc.) — omitted if empty |
| `tenant_id` | string | Tenant label — omitted if empty |
| `handled_by` | string | Instance ID that processed the request — omitted if empty |
| `model` | string | Model tag — omitted if empty |
| `length_capped` | bool | `true` if request was force-completed at `MaxModelLen` — omitted when `false` |
| `gateway_queue_delay_ms` | ms | Time spent in the gateway queue — omitted when flow control is disabled |
| `session_id` | string | Multi-turn session link — omitted for single-turn requests |
| `round_index` | int | Round within session (`0` = first turn) |
```

**Verify:** Compare against `RequestMetrics` struct in `sim/metrics_utils.go:13-30` — every `json:"..."` tag has a row.

**Commit:**
```bash
git add docs/guide/results.md
git commit -m "docs(results): add Per-Request Fields section for requests[] array (BC-3)

16 fields in RequestMetrics were completely undocumented.

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Add Session Metrics and PD Metrics sections + fix per_model JSON claim (BC-4, BC-5, BC-6)

**Files:** modify `docs/guide/results.md`

**Impl — Session Metrics section:** After the Per-Tenant Metrics section, add:

```markdown
## Session Metrics

When a workload contains multi-turn sessions (requests with `session_id`), BLIS prints a `=== Session Metrics ===` block:

| Line | What It Means |
|------|--------------|
| **Sessions** | Number of distinct multi-turn sessions in the workload |
| **TTFT cold (round 0)** | TTFT distribution for the first round of each session — no KV cache warmth |
| **TTFT warm (round≥1)** | TTFT distribution for follow-up rounds — benefits from cached context |
| **Session duration** | End-to-end duration from session start to last round completion |

Cold vs. warm TTFT split reveals prefix cache effectiveness: if `TTFT warm` ≈ `TTFT cold`, prefix caching is not activating (check `--cache-signal-delay` and scorer configuration).
```

**Impl — PD Metrics section:** After Session Metrics, add:

```markdown
## PD Disaggregation Metrics

When PD disaggregation is active (`--prefill-instances > 0`), BLIS prints a `=== PD Metrics ===` block:

| Field | What It Means |
|-------|--------------|
| **Disaggregated Requests** | Requests routed through the prefill→decode transfer path |
| **Dropped at Decode KV** | Requests whose transferred KV blocks could not be accepted by the decode instance |
| **Prefill Throughput** | Sub-request completion rate on prefill instances (sub-req/s) |
| **Decode Throughput** | Sub-request completion rate on decode instances (sub-req/s) |
| **Load Imbalance Ratio** | `max(prefill_load, decode_load) / min(...)` — 1.0 = perfectly balanced; `inf` = one pool idle |
| **Parent TTFT** | Client-visible TTFT (prefill TTFT + KV transfer duration + first decode step); distribution in microseconds |
| **KV Transfer Duration** | Time to transfer KV blocks between prefill and decode; distribution in microseconds |
| **Peak Concurrent Transfers** | Maximum simultaneous in-flight KV transfers (only with `--pd-transfer-contention`) |
| **Mean Transfer Queue Depth** | Average queue depth at the transfer bandwidth bottleneck (only with `--pd-transfer-contention`) |
```

**Impl — Fix per_model JSON accuracy:** In the Per-Model Metrics section, replace lines 74-83 (the false claim AND the orphaned per_model table that follows it):

Find (lines 74-83 in results.md, inclusive of the blank line and table):
```
When `--metrics-path` is set, the JSON output includes a `per_model` key (omitted when no requests carry model tags). Each entry has:

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `ttft` | Distribution | TTFT distribution (p50, p99, mean, count) |
| `e2e` | Distribution | E2E latency distribution |
| `throughput_rps` | float64 | Requests per second for this model |
| `tokens_per_sec` | float64 | Output tokens per second for this model |
| `total_requests` | int | Number of completed requests for this model |
```

Replace with (single corrected sentence, table removed):
```
Per-model metrics appear on stdout only. The `--metrics-path` JSON file (see [Primary Metrics](#primary-metrics-json-output)) contains only the aggregate `MetricsOutput` fields. The same applies to per-tenant, session, and PD metrics — all are stdout-only sections.
```

**Verify:** Read through `cmd/root.go:1677-1840` — confirm Session and PD sections match the print functions exactly.

**Commit:**
```bash
git add docs/guide/results.md
git commit -m "docs(results): add Session Metrics + PD Metrics sections; fix per_model JSON claim (BC-4, BC-5, BC-6)

- Session Metrics: cold/warm TTFT split, session duration
- PD Metrics: disaggregated count, transfer stats, load imbalance, contention
- Correct false claim that --metrics-path JSON contains per_model key;
  per-model/per-tenant/session/PD metrics are stdout-only

Co-Authored-By: Claude Sonnet 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Sanity Checklist

- [x] No process/workflow doc changes (Small tier)
- [x] All 6 gaps from issue #602 audit addressed
- [x] All anomaly counter `fmt.Printf` calls in `cmd/root.go:1635-1660` have table rows (BC-1)
- [x] All `json:"..."` tags in `MetricsOutput` (metrics_utils.go:57-87) have table rows (BC-2)
- [x] All `json:"..."` tags in `RequestMetrics` (metrics_utils.go:13-30) have table rows (BC-3)
- [x] Session Metrics section matches `printSessionMetrics` output (BC-4)
- [x] PD Metrics section matches `printPDMetrics` output including conditional contention fields (BC-5)
- [x] per_model JSON claim corrected (BC-6)
- [x] Deviation log: no source document contradictions
- [x] No Go code changes — no build/test/lint concerns
