# KV Cache Hit-Rate Calibration

This note documents how to validate BLIS's tiered KV-offload model against real vLLM
runs using the `observe → replay → calibrate` pipeline (issue #1583, part of the
multi-tier KV-offload epic #1585). It complements
[Observe / Replay / Calibrate](observe-replay-calibrate.md) and
[KV Cache & Memory](kv-cache.md).

## What gets compared

`blis calibrate` compares **TTFT**, **E2E**, and — as of #1583 — the **KV cache
hit-rate**:

- **Real hit-rate** is scraped from the vLLM server's Prometheus `/metrics` endpoint
  by `blis observe --scrape-kv-metrics` and recorded in the trace header
  (`observed_kv_metrics`).
- **Simulated hit-rate** is the DES aggregate `cache_hit_rate`, written to the
  `blis replay --metrics-path` MetricsOutput file.
- `blis calibrate --sim-metrics <that file>` reports the absolute error in percentage
  points and a pass/fail verdict against `--hit-rate-tolerance-pp` (default **5 pp**),
  plus a TTFT-MAPE verdict against `--ttft-mape-threshold` (default **0.15**).

BLIS exposes a single aggregate hit-rate (GPU tier plus offload misses), so the
comparison is on the **overall** hit-rate. Per-tier read/write time is recorded in the
header (`read_time_total` / `write_time_total`) as informational data for a manual
bandwidth cross-check — the simulator has no per-tier hit counters to compare against.

## Dependency on an unreleased vLLM

The tiered counters `vllm:kv_offload_tiering_block_hits` /
`vllm:kv_offload_tiering_block_queries` (and `..._read_time` / `..._write_time`) come
from vLLM PR #48798, which is **in no release tag**. Record the exact commit the
cluster ran with `--vllm-commit <sha>` — it is stored verbatim in the header so the
dependency is explicit and reproducible.

If the server exposes only released metrics, `--scrape-kv-metrics` falls back to the
GPU-only `vllm:gpu_prefix_cache_*` family and tags the observation
`source: gpu-prefix-cache-fallback`. This is a **weaker** validation (GPU tier only,
no offload tiers) and is never conflated with a tiered hit-rate — treat it as a
separate, weaker check.

## Minimal experiment set

Validate against the four canonical cache paths, over a small prefix-length sweep,
single replica, TP=1:

| Path | How to elicit it |
|------|------------------|
| **GPU hit** | Repeat a prompt whose prefix is still resident in the GPU KV cache. |
| **CPU hit** | Repeat a prefix that has been offloaded to the CPU staging tier but evicted from GPU. |
| **Storage hit** | Repeat a prefix that has cascaded to a secondary `fs` tier and been evicted from CPU. |
| **Full miss** | A unique prefix with no cached blocks anywhere. |

Sweep the shared prefix length (e.g. 256 / 1024 / 4096 tokens) so the hit-rate spans
its range. Keep the deployment to a single replica at TP=1 so the aggregate hit-rate
is unambiguous.

## Procedure

```bash
# 1. Observe the real server, scraping the per-tier counters into the trace header.
#    --vllm-commit records the pinned (unreleased) vLLM the counters require. observe
#    dispatches to a real server and does not itself model the offload tiers, so no
#    --kv-offload-config is passed here — the observed hit-rate is whatever the server
#    reports.
blis observe --server-url http://localhost:8000 --model <model> \
  --workload-spec sweep.yaml \
  --scrape-kv-metrics --vllm-commit <sha> \
  --trace-header t.yaml --trace-data t.csv

# 2. Replay the captured trace through the DES to produce the SIM-side hit-rate. Supply
#    --kv-offload-config matching the observed deployment — replay models the tiers and
#    derives the aggregate cache_hit_rate into --metrics-path. (A tiered observation
#    replayed WITHOUT an offload config is a hard error, never a silent GPU-only value.)
blis replay --trace-header t.yaml --trace-data t.csv --model <model> \
  --kv-offload-config offload.yaml \
  --results-path sim.json --metrics-path simagg.json

# 3. Compare. TTFT/E2E come from --sim-results; the hit-rate comes from --sim-metrics.
blis calibrate --trace-header t.yaml --trace-data t.csv \
  --sim-results sim.json --sim-metrics simagg.json \
  --hit-rate-tolerance-pp 5 --ttft-mape-threshold 0.15 \
  --report calibration.json
```

> **Note.** `blis observe` records the observed hit-rate and its source, but not the
> offload config — it is a black-box dispatcher against a real server. The operator
> supplies the deployment's `--kv-offload-config` on the **replay** step so the
> simulator reproduces the tiered behaviour. For a sim-generated trace
> (`blis run --kv-offload-config … --trace-output`) the config is instead recorded in
> the header and reproduced authoritatively on replay (INV-13).

The report's `hit_rate` block reports `real_hit_rate`, `sim_hit_rate`, `abs_error_pp`,
`tolerance_pp`, `within`, and `source`. Targets on the minimal set: **TTFT MAPE ≤ 15%**
and **hit-rate absolute error ≤ 5 pp**.

## Notes and caveats

- The counter delta spans the whole measured window, which includes any warm-up
  requests (they are excluded from TTFT/E2E calibration but not from the engine-level
  counter delta). Keep warm-up small, or prefer `--prewarm-duration` (which runs before
  the start scrape).
- A scrape miss (unreachable `/metrics`, no recognized counters, zero queries, or a
  counter reset mid-window) is **not fatal**: `observe` warns and omits
  `observed_kv_metrics`; the workload trace is still exported.
- The `read_time`/`write_time` counters are vLLM **device service time excluding queue
  wait** — they calibrate the offload tiers' *service* term, not the queueing term. Do
  not fit queue wait against them.
