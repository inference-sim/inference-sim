# Metrics & Results

`blis run` and `blis replay` print one document to stdout and nothing else: a blis-schemas
**Result** (`kind: Result`, schema reference: blis-schemas `docs/reference/result.md`). It is
JSON by default; `--format yaml` prints the same document as YAML; `--output <file>` writes
the same bytes to a file. The document is validated with blis-schemas' own validator before it
is printed, and a document that does not validate is a fatal error naming the problem.
Diagnostics go to stderr.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 200 --num-requests 1000 --output result.json

# The same, for a person: the text sections, rendered to stderr from the Result.
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios --registry $PWD/blis-registry \
  --num-instances 4 --rate 200 --num-requests 1000 --view summary

# Read a metric
jq '.metrics.ttft_p99_ms' result.json
```

## The Result document

| Section | What it holds |
|---|---|
| `input` | The resolved `scenario` and `deployment`, the run controls under `run` (`load`, `num_requests`, `seed`, `horizon_s`, `workload`), and a `--policy-config` bundle verbatim under `policy`. |
| `revisions` | The BLIS build, the blis-schemas, blis-latency-kernel and llm-d router releases it was built against, the engine model, and the catalog and registry git revisions (`-dirty` when the checkout has uncommitted content; `unknown` when it is not a git checkout). |
| `metric_defs`, `metrics` | Standard-named metrics over the measured `window`, each defined in blis-schemas' measurement vocabulary on the `simulated` clock (below). |
| `window` | The measured window: requests not in warm-up, first arrival to last completion (or a replay's explicit profiling window). A run in which no measured request completed states its zero rates over the simulated run (`simulation_start` to `simulation_end`). |
| `requests` | INV-1's ledger: `generated`, `rejected`, `injected` and the eleven terminal buckets, plus `length_capped`. `generated` is the cluster's own arrival count, so blis-schemas' check that `injected` equals the sum of the buckets is not a restatement of a sum. |
| `goodput` | With SLO targets: the targets, `goodput_rps`, `slo_attainment`, and per class `completed` and `met`. |
| `breakdowns` | `per_model`, `per_tenant`, `per_adapter` (the run's own metrics over each key's requests), `pd` (the prefill/decode split, in ms), `sessions`. |
| `engine` | What the engines reported: under `cluster` the step-interval series, preemptions, KV figures, cache hit rates, in-flight counts, the server series and the acceptance record, plus the whole-run figures under `whole_run`; under each pool role its prefix-cache record; under `instances` each instance's own figures. |
| `router` | The embedded router's requirements (`embed.Requirements`), sheds by SLO class, and the priority-inversion and head-of-line counters. |
| `kernel` | blis-latency-kernel's coefficient provenance per pool role. |
| `saturation` | The detectors' final labels under `final`, when `--detectors` ran. |
| `overrides` | Every setting the run simulated differently from its scenario, and what chose it. |
| `artifacts` | Files the run wrote, by path relative to `--output`'s directory and sha256: `trace_header`/`trace_data` (`--trace-output` on run), `metrics_file` (`--metrics-path`), `sim_results` (`replay --results-path`). |
| `runtime` | Wall-clock seconds, start time and host. The only block that depends on the host or the clock. |

### Standard metrics

| Name | Definition (`formula`) |
|---|---|
| `ttft_{mean,p50,p90,p95,p99}_ms` | arrival to first output token (`ttft`) |
| `itl_{mean,p50,p90,p99}_ms` | per request `(latency - ttft) / (output_tokens - 1)`, per **token** (`tpot_excl_first`) |
| `e2e_{mean,p50,p90,p95,p99}_ms` | arrival to last streamed token (`e2el`) |
| `interactivity_mean` | mean of `1000 / itl_ms` over requests (`user_tps`) |
| `interactivity_p50`, `interactivity_p90` | `1000 / itl_p50_ms`, `1000 / itl_p90_ms` (`interactivity`) |
| `output_tok_s` | output tokens over the window (`output_throughput`, `profiling_window`) |
| `output_tok_s_per_gpu` | `output_tok_s` over the GPUs the run's instances occupy (`per_gpu`) |
| `request_throughput` | completed requests over the window (`request_throughput`) |

A latency family is stated only when it has a sample: a run that completed nothing states no
latency rather than a zero. Percentiles are numpy-linear.

`itl_*` is always per token. Under speculative decoding one decode step emits several tokens, so
the per-decode-step interval (the legacy `itl_*` keys of `MetricsOutput`) is longer by the
accepted length; it is the engine's `step_interval_*`, never `itl_*`.

### Where each `MetricsOutput` field went

Every field of the JSON object run and replay printed before the Result has a home in the
document, or is listed here as dropped with the reason.
`TestResultFieldMap_EveryMetricsOutputFieldHasAHome` keeps this table complete.

<!-- result-field-map -->
| `MetricsOutput` field | Result home |
|---|---|
| `instance_id` | dropped: the document is the cluster's; each instance's own figures are keyed by id under `engine.instances` |
| `completed_requests` | `requests.completed` |
| `still_queued` | `requests.still_queued` |
| `still_running` | `requests.still_running` |
| `injected_requests` | `requests.injected` |
| `dropped_unservable` | `requests.dropped_unservable` |
| `timed_out_requests` | `requests.timed_out` |
| `length_capped_requests` | `requests.length_capped` |
| `total_input_tokens` | `engine.cluster.total_input_tokens` |
| `total_output_tokens` | `engine.cluster.total_output_tokens` |
| `vllm_estimated_duration_s` | `engine.cluster.simulated_duration_s` |
| `responses_per_sec` | `metrics.request_throughput` over the window; whole run `engine.cluster.whole_run.responses_per_sec` |
| `tokens_per_sec` | `metrics.output_tok_s` over the window; whole run `engine.cluster.whole_run.tokens_per_sec` |
| `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms` | `metrics.ttft_*_ms` over the window; whole run `engine.cluster.whole_run.ttft_*_ms` |
| `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms` | `metrics.e2e_*_ms` over the window; whole run `engine.cluster.whole_run.e2e_*_ms` |
| `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms` | per decode step, so `engine.cluster.step_interval_*_ms` (window) and `engine.cluster.whole_run.step_interval_*_ms`; the Result's `itl_*_ms` is per token |
| `scheduling_delay_p99_ms` | `engine.cluster.scheduling_delay_p99_ms` |
| `kv_allocation_failures` | `engine.cluster.kv_allocation_failures` |
| `preemption_count` | `engine.cluster.preemptions` |
| `cache_hit_rate` | `engine.cluster.cache_hit_rate` (window); `engine.cluster.cache_hit_rate_whole_run` |
| `requests` | `artifacts.metrics_file` (the deprecated file's per-request rows, by digest) |
| `saturation` | `saturation.final` |
| `goodput_rps`, `slo_attainment` | `goodput.goodput_rps`, `goodput.slo_attainment` |
| `per_class` | `goodput.per_class` (`completed`, `met`); the per-class latency summaries are `--view summary`'s Per-SLO section |
| `adapters` | `engine.cluster.adapters`, and `breakdowns.per_adapter` |
| `catalog` | `revisions.catalog` (the revision, `-dirty` when uncommitted); the path is dropped: where a catalog sits is a fact of the host, not of the result |
| `overrides` | `overrides` |
| `agentic` | `engine.cluster.agentic` |
| `pd` | `breakdowns.pd` |
| `server_series` | `engine.cluster.server_series` |
| `percentile_method` | each percentile's `metric_defs` entry (`percentile_method: numpy_linear`) |
| `ttft_p50_ms`, `e2e_p50_ms` | `metrics.ttft_p50_ms`, `metrics.e2e_p50_ms` |
| `request_itl_mean_ms`, `request_itl_p50_ms`, `request_itl_p90_ms`, `request_itl_p99_ms` | `metrics.itl_*_ms` |
| `interactivity_p50`, `interactivity_p90` | `metrics.interactivity_p50`, `metrics.interactivity_p90` |
| `output_tok_per_user_mean` | `metrics.interactivity_mean` |
| `step_interval_mean_ms`, `step_interval_p90_ms`, `step_interval_p99_ms` | `engine.cluster.step_interval_*_ms` (with `p50`, `p95`) |
| `output_throughput_tok_s` | `metrics.output_tok_s` |
| `measured_window_s` | `window.duration_s` |
| `measured_window_source` | `window.start` and `window.stop` (`first_arrival_after_warmup`/`last_completion` derived, `profiling_start`/`profiling_end` explicit) |
| `measured_completed_requests` | `window.requests` |
| `cancelled_requests`, `context_overflow_requests`, `failed_requests` | `engine.cluster.measured_cancelled_requests`, `..._context_overflow_requests`, `..._failed_requests` |
| `cpu_tier_hit_rate` | `engine.cluster.cpu_tier_hit_rate` |
| `gateway_queue_delay_mean_ms`, `gateway_queue_delay_p90_ms` | `engine.cluster.gateway_queue_delay_*_ms` |
| `in_flight_requests_mean`, `in_flight_requests_max` | `engine.cluster.in_flight_requests_*` |
| `kv_capacity_tokens_per_replica` | `engine.cluster.kv_capacity_tokens_per_replica` |
| `whole_run` | `engine.cluster.whole_run` |
| `kernel_provenance` | `kernel` |
| `blis_version`, `kernel_version`, `schemas_version` | `revisions.blis`, `revisions.blis_latency_kernel`, `revisions.blis_schemas` |
| `acceptance` | `engine.cluster.acceptance` |
| `prefix_cache` | `engine.<pool role>.prefix_cache` |
<!-- /result-field-map -->

The text sections below (anomaly counters, KV cache, per-SLO, per-model, per-tenant,
sessions, P/D, fitness, trace summary) are what `--view summary` renders to stderr.

## Deprecated results files

`--metrics-path` (run and replay) and `replay --results-path` keep the content documented below
for one release and log a deprecation pointing at `--output`; the Result references each by
digest under `artifacts`. They are then removed.

### `--metrics-path` file (deprecated)

The `--metrics-path` file contains:

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
| `ttft_mean_ms` | ms | Mean Time To First Token — average responsiveness |
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
| `saturation` | object | Post-hoc saturation detection result (omitted when `--post-hoc-detector none`; see [Saturation Detection](#saturation-detection)) |
| `catalog` | object | Model-catalog provenance: path, git revision, dirty flag (see [Catalog Provenance](#catalog-provenance)); the Result records the revision under `revisions.catalog` |
| `overrides` | array | Settings the run simulated differently from its scenario, each `{setting, scenario, run, source}`; absent when the run follows its scenario (see [Scenario Overrides](#scenario-overrides)); the Result carries the same list under `overrides` |
| `cache_hit_rate` | fraction | GPU prefix-cache hit rate: prompt hit tokens over prompt query tokens summed across instances, as vLLM counts them (#1904); the measured window's in the file, the whole run's under `whole_run` — **file only** |
| measured-window keys | — | `ttft_p50_ms`, `request_itl_*`, `interactivity_*`, `step_interval_*`, `output_throughput_tok_s`, … — **file only**; see [Measured-Window Aggregates](#measured-window-aggregates-metrics-path-file) |
| `kernel_provenance` | object | Per-pool coefficient evidence, assumptions and skipped release rules — **file only**; see [Kernel Provenance](#kernel-provenance) |
| `blis_version`, `kernel_version`, `schemas_version` | string | Build pins from the binary's build information (`"unknown"` when absent) — **file only** |
| `requests` | array | Per-request detail records (omitted when empty; see [Per-Request Fields](#per-request-fields)) |

### Measured-Window Aggregates (`--metrics-path` file)

The results file carries aiperf 0.12.0's request metrics (#1938) over the **measured window**:
requests whose `phase` is not `warmup` that completed (before the window's end, when one is
set). With no phase set, that is every completed request. One value per request;
percentiles are numpy-linear (`percentile_method: "numpy_linear"`).

| Field | Definition |
|---|---|
| `ttft_{mean,p50,p90,p95,p99}_ms` | arrival → first output token |
| `e2e_{mean,p50,p90,p95,p99}_ms` | arrival → last streamed token (aiperf's request latency; excludes the completion overhead the whole-run E2E carries) |
| `request_itl_{mean,p50,p90,p99}_ms` | per request `(latency − ttft) / (output_tokens − 1)`, requests with ≥ 2 output tokens: per **token** |
| `interactivity_p50`, `interactivity_p90` | `1000 / request_itl_p50_ms`, `1000 / request_itl_p90_ms` (tokens/s/user) |
| `output_tok_per_user_mean` | mean of `1000 / request_itl_ms` over requests with a non-zero ITL |
| `step_interval_{mean,p90,p99}_ms` | the per-decode-step series `itl_*` summarizes, over measured requests completing in the window: per **step** |
| `output_throughput_tok_s`, `measured_window_s`, `measured_window_source` | measured output tokens / window; `explicit` when the replay set the profiling window, `derived` (first measured arrival → last measured completion) otherwise |
| `measured_completed_requests`, `cancelled_requests`, `context_overflow_requests`, `failed_requests` | counts in the window (INV-M1): cancelled = in flight when it closed or completed after its end; failed = timed out, dropped, rejected or shed |
| `cpu_tier_hit_rate` | prompt tokens the CPU tier supplied / prompt tokens queried (measured admissions) |
| `whole_run` | the whole-run `ttft_*`, `e2e_*` and `cache_hit_rate`, for consumers of the old meaning (`blis calibrate`) |
| `gateway_queue_delay_{mean,p90}_ms` | flow-control queueing per measured request (0 without flow control) |
| `in_flight_requests_{mean,max}` | engine-side in-flight requests (dispatched, not completed), time-weighted over the window |
| `kv_capacity_tokens_per_replica` | GPU KV pool of the smallest instance, tokens |

In the file, `ttft_*`, `e2e_*` and `cache_hit_rate` take these measured-window values; the
top-level whole-run keys keep the whole-run figures (TTFT there also counts requests that emitted a first token but did
not complete), and the file repeats them under `whole_run`.
The legacy `itl_*` keys are **unchanged** for backward compatibility — they
remain per verification step under speculative decoding — and `completed_requests` stays the
run-wide INV-1 count. Invariants: INV-M1 … INV-M4.

**Comparing a hit rate with aiperf 0.12.0.** Compare aiperf's `server_metrics_export.json`
`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` "profiling" totals with
`whole_run.cache_hit_rate`, not with the measured-window `cache_hit_rate`. aiperf takes that
total as last-profiling-scrape minus the scrape before the first PROFILING-tagged scrape
(`server_metrics/accumulator.py:190-194, 387-401`), and in 0.12.0 a scrape at warmup start
already carries the PROFILING tag. So the baseline precedes warmup, and the exported total
counts warmup plus profiling. Real 0.12.0 exports confirm it: the profiling total equals the
warmup total plus the post-start slices at every level. Under agentx replay the warmup prefills
are cold, so that figure runs several points below the profiling window's own.

### Kernel Provenance

`kernel_provenance` is keyed by pool role (`colocated`, or `prefill` and `decode`), #1942:

```json
{
  "kernel_provenance": {
    "colocated": {
      "method_counts": {"measured": 92, "vendor_spec": 15, "assumed": 47, "not_charged": 1},
      "evidenced": 107, "total": 155, "evidence_source": "kernel",
      "fallback_or_assumed": [{"Name": "attention_decode_floor_mla", "Set": "cost-model-attention", "Method": "assumed", "Scope": "hardware: h100"}],
      "assumptions": [{"Name": "max_model_len", "Set": "blis-latency-kernel", "Method": "assumed", "Scope": "1048576: ..."}],
      "skipped_rules": ["all2all-backend-known", "quantization-known", "..."],
      "skipped_rule_findings": []
    }
  }
}
```

The entries are blis-latency-kernel's own `kernel.CoefficientOrigin` records (hence their
field-name keys); `evidenced`/`total` are its `Evidence()`. `fallback_or_assumed` lists the
registry entries copied from another scope or assumed, `assumptions` the values the kernel
supplied itself. `skipped_rules` are the scenario's release rules that neither the kernel nor
BLIS enforces, and `skipped_rule_findings` what those rules report when the pack is run
report-only. `evidence_source` is `kernel` when the counts are the kernel's `Evidence()`. A run
no kernel priced writes `"kernel_provenance": {}`.

### Scheduling Delay

Scheduling delay isolates the WaitQ wait time from compute time. High scheduling delay + low preemptions = **queue saturation** (add instances). Low scheduling delay + high TTFT = **compute saturation** (reduce batch size or use chunked prefill).

!!! warning "Per-request units"
    All per-request latency fields (`ttft_ms`, `e2e_ms`, `itl_ms`, `scheduling_delay_ms`) are in **milliseconds** — converted from internal ticks by dividing by 1,000. Aggregate metrics (`scheduling_delay_p99_ms`, etc.) are also in milliseconds. See [Known Unit Gotchas](../reference/configuration.md#known-unit-gotchas) for the full unit reference. Note: hypothesis scripts written before BC-14 may divide `scheduling_delay_ms` by 1,000 unnecessarily — that field is now already in ms.

### Scenario Overrides

A run reproduces its scenario unless a flag or a policy bundle chooses another value for a
setting the scenario states. Today those are the two settings vLLM's
`engine.scheduling_policy` decides: the waiting queue's order and the preemption victim
(vLLM `fcfs` is BLIS `--scheduler fcfs` with `--preemption-policy fcfs`; vLLM `priority` is
`--scheduler priority-fcfs` with `--preemption-policy priority`). An explicit `--scheduler` or
`--preemption-policy`, or a `--policy-config` bundle's `scheduler` or `preemption.policy`, that
differs from the scenario is kept, since exploring a policy against a fixed deployment is the
point, and the results file says so:

```json
{
  "overrides": [
    {"setting": "scheduler", "scenario": "priority-fcfs", "run": "sjf", "source": "--scheduler"}
  ]
}
```

In a prefill/decode deployment each pool runs its own engine's policy, so a run-level choice
is recorded once per pool whose engine stated otherwise, as `"scheduler (prefill pool)"` or
`"preemption_policy (decode pool)"`.

A warning on stderr says the same. The Result carries the same list under `overrides`.

### Catalog Provenance

The `catalog` field records **which model catalog produced the result** (#1732), so a results
file is attributable and an *experiment* is distinguishable from a *reproducible run*. It is
written by both `blis run --metrics-path` and `blis replay --metrics-path`:

```json
{
  "catalog": {
    "path": "/home/me/blis-catalog",
    "revision": "63a9a5010a3f0c1d9f6b2e4a7c8d5b1e2f3a4b5c",
    "dirty": false
  }
}
```

- `path` — the catalog **clone root** that was located by `--catalog` / `BLIS_CATALOG`,
  recorded **verbatim as supplied**: a relative value stays relative and an absolute value
  stays absolute, neither is rewritten. So the field is not categorically CWD-relative — a
  relative value is only interpretable alongside the working directory the run was launched
  from, which is a reason to prefer an absolute `--catalog` / `BLIS_CATALOG` when the
  results file needs to be attributable later.
- `revision` — the catalog's git commit, or `"unknown"` when the catalog is not a git
  checkout (a revision is never invented).
- `dirty` — `true` when the catalog **subtree** has uncommitted or untracked content, i.e.
  the run used a config the recorded revision does not contain. A catalog inside a larger
  repository is judged on its own subtree, so an unrelated edit elsewhere in that repository
  does not mark the run an experiment. `dirty` is always `false` alongside an unknown
  revision — there is no committed state to differ from.

!!! note "In the Result"
    The Result records the catalog's revision (and `-dirty`) under `revisions.catalog`: a
    revision is an input of the run, so two runs on catalogs at different revisions are
    different inputs (INV-6). It never records the path, a fact of the host.

### Saturation Detection

Post-hoc saturation analysis classifies a completed run as `STABLE`, `BACKLOGGED` or
`OVERLOADED`. It is distinct from the llm-d router's saturation detector, which gates admission
and flow control during the run. Select detectors with
`--detectors <name|comma-list|all>` (`composite`, `threshold`, `backlog-drift`, `peak-rate`),
tune them with `--saturation-config`, and write the per-event verdict trace with
`--saturation-report`:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 10 --num-requests 100 --detectors all --saturation-report sat.json
```

Detector mechanics and tuning knobs:
[Saturation analyzer](../contributing/saturation-analyzer-extension.md).

### Per-Request Fields

When the `requests` array is non-empty, each entry contains:

| Field | Unit | Description |
|-------|------|-------------|
| `requestID` | string | Unique request identifier |
| `arrived_at` | s | Arrival timestamp in simulation seconds (`ArrivalTime / 1,000,000`) |
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
| `gateway_queue_delay_ms` | ms | Time from the router enqueuing the request in flow control to its dispatch — omitted when flow control (`featureGates: [flowControl]`) is off |
| `session_id` | string | Multi-turn session link — omitted for single-turn requests |
| `round_index` | int | Round within session (`0` = first turn); always present, defaults to `0` for non-session requests |

## Anomaly Counters

When anomalies are detected, `--view summary` prints `=== Anomaly Counters ===`:

| Counter | What It Means | Action |
|---------|--------------|--------|
| **Priority Inversions** | A lower-priority request was scheduled before a higher-priority one | Check scheduler choice — use `priority-fcfs` for SLO workloads |
| **HOL Blocking Events** | A long prefill blocked shorter requests | Enable chunked prefill: state `long_prefill_token_threshold: 256` in the scenario's pool `engine` block |
| **Rejected Requests (Admission)** | The llm-d router denied admission: an admitter plugin, or the legacy admission controller shedding a sheddable request (priority < 0) from a saturated pool | Check the `InferenceObjective` priorities and the saturation detector's parameters; see [Admission Control](admission.md) |
| **Shed (tier)** | Per-SLO-class count of requests the router shed (admission denials and flow-control TTL expiries) — printed as indented sub-items beneath Rejected Requests (Admission) | Raise the class's `InferenceObjective` priority, or the detector's thresholds |
| **Rejected Requests (Routing)** | No endpoint could serve the request: no ready candidates (all instances `Loading` or `Draining`), or every filter eliminated all | Increase `initial_nodes`, reduce `loading_delay.mean`, stagger drain operations, or check the router configuration's filters |
| **Dropped Unservable** | Request exceeds the context window (the scenario's `max_model_len`) or needs more KV blocks than exist | Check the scenario's `max_model_len`; deploy more GPUs per instance or reduce max input tokens |
| **Timed Out Requests** | Request exceeded its client deadline before completing | Increase `--timeout` or reduce load |
| **Length-Capped Requests** | Request was force-completed when it reached `MaxModelLen` tokens during decode | Expected if workloads push against the scenario's `max_model_len` |
| **Gateway Queue Depth (horizon)** | Requests still waiting in the router's flow control when the simulation ended — printed only when `> 0` | Reduce arrival rate or increase cluster capacity |
| **Gateway Queue Rejected** | Requests flow control rejected because its queues were at capacity — printed only when `> 0` | Raise `flowControl.maxRequests` or the priority band's capacity in the EndpointPickerConfig |
| **Gateway Expired (TTL)** | Requests that waited in flow control past their queue-wait budget — printed only when `> 0` | Raise `flowControl.defaultRequestTTL`, or add capacity |
| **Encode Routing Rejections** | Requests rejected at an encode stage — printed only when `> 0` | — |

Each counter is one INV-1 bucket; see [Admission Control](admission.md) for how router outcomes
map onto them.

!!! note "Stdout-only vs JSON counters"
    **Dropped Unservable**, **Timed Out Requests**, and **Length-Capped Requests** also appear as fields in the `--metrics-path` JSON output (see [Primary Metrics](#primary-metrics-json-output): `dropped_unservable`, `timed_out_requests`, `length_capped_requests`).

    All other counters in this table — including **Shed (tier)**, **Gateway Queue Depth (horizon)**, and **Gateway Queue Shed** — are in the Result's `requests` ledger and `router.shed_by_class`, and do not appear in the `--metrics-path` file.

!!! note "blis replay anomaly block"
    `blis replay` produces a subset of this output: **Timed Out Requests**, **Gateway Queue Depth (horizon)**, and **Gateway Queue Shed** are currently missing from the `blis replay` anomaly block even when non-zero (tracked in issue #1184).

## KV Cache Metrics

When KV cache activity is nonzero, `--view summary` prints `=== KV Cache Metrics ===`:

| Metric | Meaning | Concern Threshold |
|--------|---------|-------------------|
| **Preemption Rate** | Ratio of preemption events to completed requests — a single request can be preempted more than once | > 5% indicates KV pressure |
| **Cache Hit Rate** | Fraction of blocks served from prefix cache | Higher is better — indicates prefix reuse |
| **KV Thrashing Rate** | Repeated preemption-reallocation cycles | > 0 indicates severe memory pressure |

## Per-SLO-Class Metrics

When multiple SLO classes are present in the workload, `--view summary` prints per-class TTFT and E2E distributions. This lets you verify that `critical` requests meet SLOs even when `batch` traffic is heavy.

## Per-Model Metrics

When instances serve different models (multi-model deployment), `--view summary` prints per-model TTFT mean/p99, E2E mean/p99, and throughput (req/s). This appears automatically when requests carry model tags. Output format:

```
=== Per-Model Metrics ===
  qwen/qwen3-14b:
    TTFT: p50=1234.56 p99=5678.90 (n=250)
    E2E:  p50=9876.54 p99=12345.67 (n=250)
    Throughput: 50.00 req/s, 6400.00 tok/s
```

These sections are `--view summary`'s rendering, on stderr; the Result carries the same splits under `breakdowns` (`per_model`, `per_tenant`, `sessions`, `pd`).

## Per-Tenant Metrics

When requests carry `tenant_id` labels, `--view summary` prints per-tenant request counts, total output tokens served, and a [Jain Fairness Index](https://en.wikipedia.org/wiki/Fairness_measure) over the token distribution. This section appears automatically and is omitted when no requests carry tenant labels (backward-compatible with legacy and untenanted workloads). A workload with a single named tenant shows the section with Jain=1.0 (trivially fair).

```
=== Per-Tenant Metrics ===
  alice: requests=50, tokens=12500
  bob: requests=50, tokens=12480
  Jain Fairness Index: 0.9999
```

Tenants are listed in lexicographic order. The Jain Fairness Index ranges from `1/N` (maximally unfair — one tenant receives everything) to `1.0` (perfectly fair — all tenants receive equal tokens). A balanced two-tenant workload produces a value ≥ 0.99.

To tag requests with tenant labels, set `tenant_id` in your workload spec cohort:

```yaml
cohorts:
  - name: alice-traffic
    tenant_id: alice
    ...
  - name: bob-traffic
    tenant_id: bob
    ...
```

## Session Metrics

When a workload contains multi-turn sessions (requests with `session_id`), `--view summary` prints a `=== Session Metrics ===` block:

| Line | What It Means |
|------|--------------|
| **Sessions** | Number of distinct multi-turn sessions in the workload |
| **TTFT cold (round 0)** | TTFT distribution for the first round of each session — no KV cache warmth from prior context |
| **TTFT warm (round≥1)** | TTFT distribution for follow-up rounds — benefits from cached context |
| **Session duration** | End-to-end duration from session start (first round arrival) to last round completion |

Cold vs. warm TTFT split reveals prefix cache effectiveness: if `TTFT warm` ≈ `TTFT cold`, prefix caching is not activating for continuations — check that the router configuration includes the `prefix-cache-scorer` and that its weight is not dominated by load scorers.

## PD Disaggregation Metrics

When PD disaggregation is active (`--prefill-instances > 0`), `--view summary` prints a `=== PD Metrics ===` block:

| Field | What It Means |
|-------|--------------|
| **Disaggregated Requests** | Requests that completed KV transfer through the prefill→decode path |
| **Dropped at Decode KV** | Requests whose transferred KV blocks could not be accepted by the decode instance |
| **Prefill Throughput** | Sub-request completion rate on prefill instances (sub-req/s) |
| **Decode Throughput** | Sub-request completion rate on decode instances (sub-req/s) |
| **Load Imbalance Ratio** | `max(prefill_load, decode_load) / min(...)` — `1.0` = perfectly balanced; `inf` = one pool has no completions |
| **Parent TTFT** | Client-visible TTFT (decode-sub-request scheduling delay + first decode step) — the arrival → first-token-emitted-by-decode span, including the decode-queue wait and exactly one output-token processing overhead (#1510); distribution in microseconds |
| **KV Transfer Duration** | Time to transfer KV blocks from prefill to decode instance, priced by blis-latency-kernel over the scenario's fabric; distribution in microseconds |

PD Disaggregation Metrics are produced by both `blis run` and `blis replay` for a disaggregated scenario run with `--prefill-instances`/`--decode-instances`.

## Fitness Evaluation

For automated multi-configuration comparison:

```bash
--fitness-weights "p99_ttft:3,mean_e2e:1,throughput:2"
```

Valid metric keys: `throughput`, `tokens_per_sec`, `p99_ttft`, `p50_ttft`, `mean_ttft`, `p99_e2e`, `p50_e2e`, `mean_e2e`.

### How Normalization Works

- **Latency metrics:** `1 / (1 + value/1000)` — lower latency → higher score. Reference: 1000 ticks = 1ms
- **Throughput metrics:** `value / (value + reference)` — higher throughput → higher score. References: RPS=100, TPS=10,000

!!! warning "Normalization compresses large differences"
    The `1/(1+x/1000)` function compresses large raw differences into small score differences. A 38% TTFT p99 improvement (39,000→64,000 ticks) maps to only 2-8% fitness score difference. Always examine raw metrics alongside fitness scores for meaningful comparison.

## Common Patterns

### Saturation Curves

As arrival rate increases past per-instance service capacity (μ ≈ 1/step_time), TTFT p99 grows super-linearly. The queue growth rate `excess = λ/k - μ` determines how quickly latency degrades.

### Tail Latency Spikes

P99 diverges from mean sharply near saturation. A workload at 90% capacity may show 2x mean TTFT but 10x P99 TTFT.

### Metrics Staleness Effects

The router scores endpoints from metrics it scrapes every `--refresh-metrics-interval` (an EPP
flag). Between scrapes it sees stale queue and KV-cache values, so load-only configurations can
send bursts to the same endpoint. Shorten the interval in the scenario's `router.epp_flags`, or add a scorer fed by the
router's own in-flight accounting (`active-request-scorer`), to reduce the effect.

### Configuration Equivalence at Low Load

Router configurations produce near-equivalent results at low utilization. Differentiation requires moderate-to-high load where queueing dynamics dominate.

### Host Overhead

blis-latency-kernel prices host work outside the GPU step: per-request queueing (tokenization
and preprocessing) delays enqueue, creating an event gap, but does not occupy the GPU;
per-output-token processing adds to TTFT/E2E metrics but does not affect step scheduling. So
simulated E2E exceeds a theoretical M/M/k E2E, most at high load. To compare with a
theoretical model, use `rho_eff = lambda × step_total`, not `lambda × E2E_total`.

## Per-Request Results

For detailed analysis, save per-request data:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios <dir of scenario files> --registry $PWD/blis-registry \
  --rate 100 --num-requests 500 --metrics-path metrics.json
```

Each request record includes TTFT, E2E, scheduling delay, and completion status.

## Further Reading

- [Configuration Reference](../reference/configuration.md#fitness-evaluation) — fitness weight syntax
- [Tutorial: Capacity Planning](../getting-started/tutorial.md) — applying results to capacity decisions
