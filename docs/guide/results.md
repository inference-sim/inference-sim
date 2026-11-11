# Metrics & Results

`blis run` and `blis replay` print one document to stdout and nothing else: a **Result**
(`kind: Result`, `schema_version: 2`), the document blis-substrate's `contracts/spec/result`
defines (`blis schema` prints its JSON Schema). It is JSON by default; `--format yaml` prints the
same document as YAML; `--output <file>` writes the same bytes to a file. The document is
checked with contracts' own validator before it is printed, and a document that does not
validate is a fatal error naming the problem. Diagnostics go to stderr.

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios \
  --num-instances 4 --rate 200 --num-requests 1000 --output result.json

# The same, for a person: the text sections, rendered to stderr from the Result.
./blis run --scenario llama-3.1-70b-instruct-h200-tp4-4node.yaml \
  --scenarios testdata/scenarios \
  --num-instances 4 --rate 200 --num-requests 1000 --view summary

# Read a metric: each is a figure, its value and the samples it is computed over
jq '.metrics.ttft_p99_ms.value' result.json
```

## The Result document

| Section | What it holds |
|---|---|
| `engine` | The engine the scenario names: `kind` (`vllm`) and the upstream `version` it runs (`0.32.0`). |
| `data` | The catalog and registry the run read (design §3.3, SIM-6): for each, `source` (`embedded`, the copy compiled into the pinned blis-substrate, or `path` with `--catalog`/`--registry`), its content `hash`, the `overlays` applied with their hashes, and the `combined_hash`; `released_data` (true only for a tagged build reading both embedded copies with no overlay); and `build`, the release tag or `unreleased`. |
| `selected_entries` | The catalog and registry entries the run read, each by identity and hash, and the `unread_files` the loader opened but no reader used (VAL-6). |
| `input` | The resolved `scenario` and `deployment`, and a `--policy-config` bundle verbatim under `policy`. |
| `frame` | The run's experimental frame (design §8.6, DEV-7): `kind` (`terminating`); `generator` (`workload`: a catalog preset as `shape`, a replayed trace as `trace`, pinned by sha256 and carrying its header, an agentic replay as `replay`; `load`, the offered load; `seed`, the seed the workload's draws used; `rng`, the random-number generator and how its streams are seeded); `initial_state` (every engine's caches start `empty`); `phases` (`warmup`, `measured` with the window's `start` and `stop` rules, `drain`); `stop` (`requests`, the request cap, and `horizon_s`, for a replay the horizon it ran to); `acceptance` (each condition, its threshold, what was observed and whether it was met); and `metrics` (the identifiers of `metrics`). |
| `revisions` | The `simulator` build, the `substrate`, `router` and `engine_module` (blis-vllm) versions it was built against. |
| `window` | The measured window, `start_us` to `end_us`: requests not in warm-up, first arrival to last completion (or a replay's explicit profiling window), with its `duration_s` and the `requests` completed in it. A run in which no measured request completed states its zero rates over the simulated run. |
| `metrics` | Standard metrics over the window, keyed by contracts/metrics registry identifier (below). Each is a figure: its `value` and the `samples` it is computed over. |
| `requests` | Each generated request's fate (DEV-9): `completed`, `timed_out`, `dropped` (by reason: `admission_rejected`, `routing_rejected`, `gateway_queue_rejected`, `gateway_expired`, `gateway_evicted`, `unservable`, `decode_kv_unavailable`), `in_flight_at_horizon`, `queued_at_horizon`, `not_started`, and `length_capped`. `generated` is the cluster's own count, so the validator's check that the fates sum to it is not a restatement of a sum. |
| `goodput` | With SLO targets: the targets, and per class `completed` and `met`; `goodput_rps` and `slo_attainment` are in `metrics`. Unlike every other rate, `goodput_rps` is over the whole simulated run (`reports.engine.cluster.simulated_duration_s`), not the measured window, as the results file always computed it. |
| `breakdowns` | `per_model`, `per_tenant`, `per_adapter` (the run's own metrics over each key's requests), `pd` (the prefill/decode split, in ms), `sessions`. |
| `reports.engine` | What the engines reported: under `cluster` the per-step and KV figures, preemptions, cache hit rates, in-flight counts, the server series and the acceptance record, plus the whole-run figures under `whole_run`; under each pool role its prefix-cache record; under `instances` each instance's own figures. |
| `reports.router` | The embedded router's requirements (`embed.Requirements`), sheds by SLO class, and the priority-inversion and head-of-line counters; with `--fitness-weights`, the score under `fitness`; with `--summarize-trace`, the decision-trace summary under `trace_summary`. |
| `reports.kernel` | The cost model's coefficient provenance per pool role. |
| `reports.saturation` | The detectors' final labels under `final`, when `--detectors` ran. |
| `overrides` | Every setting the run simulated differently from its scenario, and what chose it: a `--scheduler` or `--preemption-policy` against the scenario's stated policy, and every other command-line flag that changes what is simulated and was given a value other than its default (`--timeout`, `--num-instances`, `--prefix-tokens`, the token-length flags, the router timing flags, the speculative-decoding flags, ...), as `setting` (the flag), `run` (its value; a file flag by `sha256:<digest>`) and `source`. A workload spec run without `--seed` adds `workload_seed` (the spec's own request seed). Flags that select or tune a reported section without changing what is simulated (`--detectors`, `--saturation-config`, `--saturation-final-window`, `--server-series-interval-s`, `--fitness-weights`, `--trace-level`, `--summarize-trace`) are recorded the same way, because they change bytes of the document (SIM-10, #2062). Flags that change no byte of it (`--output`, `--format`, `--view`, `--log`, output paths, `--counterfactual-k`, `--lazy-generation`) are not recorded; `cmd/result_flags.go` lists each with the reason, and a test fails for a new flag listed in neither place. The run's `name` carries a digest of the overrides, so two runs that differ only in a flag get different names. |
| `artifacts` | Files the run wrote, by sha256 and by path relative to `--output`'s directory (as given, with no `--output`): `trace_header`/`trace_data` (`--trace-output` on run). |
| `runtime` | Wall-clock seconds, start time and host. The only block that depends on the host or the clock. |

### Standard metrics

Each identifier is a contracts/metrics registry entry in the pinned blis-substrate release.

| Identifier | Definition |
|---|---|
| `ttft_{mean,p50,p90,p95,p99}_ms` | arrival to first output token |
| `tpot_{mean,p50,p90,p99}_ms` | per request `(latency - ttft) / (output_tokens - 1)`: time per output token after the first |
| `e2e_{mean,p50,p90,p95,p99}_ms` | arrival to last streamed token |
| `step_interval_{mean,p50,p90,p95,p99}_ms` | the engines' per-decode-step interval, over measured requests completing in the window |
| `interactivity_mean` | mean of `1000 / tpot_ms` over requests |
| `interactivity_p50`, `interactivity_p90` | `1000 / tpot_p50_ms`, `1000 / tpot_p90_ms` |
| `output_tok_s` | output tokens over the window |
| `output_tok_s_per_gpu` | `output_tok_s` over the GPUs the run's instances occupy |
| `total_tokens_per_second_per_gpu` | (prompt plus output tokens per second of the measured window) over the GPUs the run's instances occupy: InferenceX's `tput_per_gpu`. Stated only when the pinned metric registry defines it |
| `output_tokens_per_second_per_decode_gpu` | output tokens per second over the GPUs of the decode pool's instances, for a disaggregated run: InferenceX's `output_tput_per_gpu`. Stated only when the pinned registry defines it |
| `request_throughput` | completed requests over the window |
| `goodput_rps`, `slo_attainment` | with SLO targets: requests meeting every target, per second of the simulated run, and as a fraction of completed |

A latency family is stated only when it has a sample: a run that completed nothing states no
latency rather than a zero. The TPOT family needs a request with two or more output tokens;
`interactivity_p50` and `interactivity_p90` additionally need that percentile of TPOT to be
positive (they are its reciprocal), and `interactivity_mean` needs a request with a positive
TPOT. A run can therefore state `interactivity_mean` without the percentiles. Percentiles are
numpy-linear.

No `itl_*` is reported. The registry's `itl` pools every gap between consecutive tokens, across
requests, which BLIS does not compute. Schema 1's `itl_*` was the per-request time per output
token, which schema 2 names `tpot_*`. Under speculative decoding one decode step emits several
tokens, so `step_interval_*` is longer than `tpot_*` by the accepted length.

A pool's `engine.stream_interval` (vLLM's `SchedulerConfig.stream_interval`) groups the tokens the
client is sent: the first output at once, then a group each time that many tokens have
accumulated, and the rest when the request finishes. It changes when the client sees tokens and
nothing the engine does, so the engine's steps, TTFT, each request's end-to-end latency, `tpot_*`
and `step_interval_*` are as without it (the last chunk arrives at the finish, and a client
computing `(latency - ttft) / (output tokens - 1)` gets the same figure). Only the gaps between the
client's chunks change: fewer, longer ones. The simulator records them on the request
(`Request.ClientITL`); the Result does not report them, since it reports no pooled `itl_*`.

### From schema 1 to schema 2

The Result moved from blis-schemas' schema 1 to contracts/spec/result's schema 2. Each schema 1
member has a schema 2 home, or is listed as dropped with the reason.

| Schema 1 | Schema 2 |
|---|---|
| `metric_defs` | dropped: every metric is a contracts/metrics registry identifier, defined once there |
| `metrics.<id>` (a number) | `metrics.<id>.value`, with `metrics.<id>.samples` |
| `metrics.itl_{mean,p50,p90,p99}_ms` | `metrics.tpot_*_ms` |
| `engine.cluster.step_interval_*_ms` | `metrics.step_interval_*_ms`, and still in `reports.engine.cluster` |
| `goodput.goodput_rps`, `goodput.slo_attainment` | `metrics.goodput_rps`, `metrics.slo_attainment` |
| `input.run.load` | `frame.generator.load` |
| `input.run.seed` | `frame.generator.seed` (the seed the workload's draws used; `--seed` when it is not the spec's is an `overrides` row) |
| `input.run.workload` | `frame.generator.workload` |
| `input.run.num_requests` | `frame.stop.requests`, the cap as given; a cap below `requests.generated` (closed-loop follow-ups) is an `overrides` row, `num_requests_cap` |
| `input.run.horizon_s` | `frame.stop.horizon_s` |
| `engine.frame.*` | `frame.*`: `stop_rule` is `stop`, `generator.random` is `generator.rng`, `phases.measured_start`/`measured_stop` are `phases.measured.start`/`stop`, `acceptance[].unfinished_at_stop` is the condition's `observed` |
| `engine` (the reports) | `reports.engine` |
| `router` | `reports.router` |
| `kernel` | `reports.kernel` |
| `saturation` | `reports.saturation` |
| `revisions.blis` | `revisions.simulator` |
| `revisions.blis_schemas`, `revisions.blis_latency_kernel` | `revisions.substrate` (one module holds both) |
| `revisions.llm_d_router` | `revisions.router` |
| `revisions.engine.kind`, `.upstream_version` | `engine.kind`, `engine.version` |
| `revisions.engine.release` | `revisions.engine_module` |
| `revisions.catalog`, `revisions.registry` | `data.catalog`, `data.registry`, by content hash and source instead of a git revision |
| `window.start`, `window.stop` | `frame.phases.measured.start`, `.stop`; the window itself is `window.start_us`, `window.end_us` |
| `requests.injected` | dropped: the fates of the generated requests sum to `requests.generated` |
| `requests.rejected` | `requests.dropped.admission_rejected` |
| `requests.routing_rejections`, `requests.encode_routing_rejections` | `requests.dropped.routing_rejected` |
| `requests.gateway_queue_rejected`, `gateway_expired`, `gateway_evicted` | `requests.dropped.gateway_queue_rejected`, `.gateway_expired`, `.gateway_evicted` |
| `requests.dropped_unservable` | `requests.dropped.unservable` |
| `requests.still_running` | `requests.in_flight_at_horizon` |
| `requests.still_queued`, `requests.gateway_queue_depth` | `requests.queued_at_horizon` |
| (none) | `requests.not_started`, `requests.dropped.decode_kv_unavailable`, `selected_entries`, `data.released_data`, `data.build` |

### Where each `MetricsOutput` field went

Every field of the JSON object run and replay printed before the Result has a home in the
document, or is listed here as dropped with the reason.
`TestResultFieldMap_EveryMetricsOutputFieldHasAHome` keeps this table complete.

<!-- result-field-map -->
| `MetricsOutput` field | Result home |
|---|---|
| `instance_id` | dropped: the document is the cluster's; in a run of more than one instance each instance's own figures are keyed by id under `reports.engine.instances` |
| `completed_requests` | `requests.completed` |
| `still_queued` | `requests.queued_at_horizon` |
| `still_running` | `requests.in_flight_at_horizon` |
| `injected_requests` | `requests.generated` (the fates sum to it) |
| `dropped_unservable` | `requests.dropped.unservable` |
| `timed_out_requests` | `requests.timed_out` |
| `length_capped_requests` | `requests.length_capped` |
| `total_input_tokens` | `reports.engine.cluster.total_input_tokens` |
| `total_output_tokens` | `reports.engine.cluster.total_output_tokens` |
| `vllm_estimated_duration_s` | `reports.engine.cluster.simulated_duration_s` |
| `responses_per_sec` | `metrics.request_throughput` over the window; whole run `reports.engine.cluster.whole_run.responses_per_sec` |
| `tokens_per_sec` | `metrics.output_tok_s` over the window; whole run `reports.engine.cluster.whole_run.tokens_per_sec` |
| `ttft_mean_ms`, `ttft_p90_ms`, `ttft_p95_ms`, `ttft_p99_ms` | `metrics.ttft_*_ms` over the window; whole run `reports.engine.cluster.whole_run.ttft_*_ms` |
| `e2e_mean_ms`, `e2e_p90_ms`, `e2e_p95_ms`, `e2e_p99_ms` | `metrics.e2e_*_ms` over the window; whole run `reports.engine.cluster.whole_run.e2e_*_ms` |
| `itl_mean_ms`, `itl_p90_ms`, `itl_p95_ms`, `itl_p99_ms` | per decode step, so `reports.engine.cluster.step_interval_*_ms` (window) and `reports.engine.cluster.whole_run.step_interval_*_ms`; the Result's `tpot_*_ms` is per token |
| `scheduling_delay_p99_ms` | `reports.engine.cluster.scheduling_delay_p99_ms` |
| `kv_allocation_failures` | `reports.engine.cluster.kv_allocation_failures` |
| `preemption_count` | `reports.engine.cluster.preemptions` |
| `cache_hit_rate` | `reports.engine.cluster.cache_hit_rate` (window); `reports.engine.cluster.cache_hit_rate_whole_run` |
| `requests` | `request_rows` (one row per generated request) |
| `saturation` | `reports.saturation.final` |
| `goodput_rps`, `slo_attainment` | `metrics.goodput_rps`, `metrics.slo_attainment` |
| `per_class` | `goodput.per_class` (`completed`, `met`); the per-class latency summaries are `--view summary`'s Per-SLO section |
| `adapters` | `reports.engine.cluster.adapters`, and `breakdowns.per_adapter` |
| `catalog` | `data.catalog` (its source, content hash and overlays); the path is dropped: where a catalog sits is a fact of the host, not of the result |
| `overrides` | `overrides` |
| `agentic` | `reports.engine.cluster.agentic` |
| `pd` | `breakdowns.pd` |
| `server_series` | `reports.engine.cluster.server_series` |
| `percentile_method` | the contracts/metrics registry entry of each percentile (numpy-linear) |
| `ttft_p50_ms`, `e2e_p50_ms` | `metrics.ttft_p50_ms`, `metrics.e2e_p50_ms` |
| `request_itl_mean_ms`, `request_itl_p50_ms`, `request_itl_p90_ms`, `request_itl_p99_ms` | `metrics.tpot_*_ms` |
| `interactivity_p50`, `interactivity_p90` | `metrics.interactivity_p50`, `metrics.interactivity_p90` |
| `output_tok_per_user_mean` | `metrics.interactivity_mean` |
| `step_interval_mean_ms`, `step_interval_p90_ms`, `step_interval_p99_ms` | `metrics.step_interval_*_ms` (with `p50`, `p95`) |
| `output_throughput_tok_s` | `metrics.output_tok_s` |
| `measured_window_s` | `window.duration_s` |
| `measured_window_source` | `frame.phases.measured.start` and `.stop` (`first_arrival_after_warmup`/`last_completion` derived, `profiling_start`/`profiling_end` explicit) |
| `measured_completed_requests` | `window.requests` |
| `cancelled_requests`, `context_overflow_requests`, `failed_requests` | `reports.engine.cluster.measured_cancelled_requests`, `..._context_overflow_requests`, `..._failed_requests` |
| `cpu_tier_hit_rate` | `reports.engine.cluster.cpu_tier_hit_rate` |
| `gateway_queue_delay_mean_ms`, `gateway_queue_delay_p90_ms` | `reports.engine.cluster.gateway_queue_delay_*_ms` |
| `in_flight_requests_mean`, `in_flight_requests_max` | `reports.engine.cluster.in_flight_requests_*` |
| `kv_capacity_tokens_per_replica` | `reports.engine.cluster.kv_capacity_tokens_per_replica` |
| `whole_run` | `reports.engine.cluster.whole_run` |
| `kernel_provenance` | `reports.kernel` |
| `blis_version`, `kernel_version`, `schemas_version` | `revisions.simulator`, `revisions.substrate` (one module holds the kernel and the schemas) |
| `acceptance` | `reports.engine.cluster.acceptance` |
| `prefix_cache` | `reports.engine.<pool role>.prefix_cache` |
<!-- /result-field-map -->

`frame.stop.requests` is the cap as given: `--num-requests` (or the spec's `num_requests`), 0
for none. When closed-loop session rounds generated more during the run (they are generated on
top of the cap), `requests.generated` exceeds it and the cap is an `overrides` row,
`num_requests_cap`. A run in which nothing arrived and no `--horizon` bounded simulated time has
no span to report over and is refused.

The text sections below (anomaly counters, KV cache, per-SLO, per-model, per-tenant,
sessions, P/D, fitness, trace summary) are what `--view summary` renders to stderr.

## Removed results files

`--metrics-path` and `--results-path` were removed in v0.16.0 (#2071): the Result document on
standard output, or in the `--output` file, carries every figure those files held. The
aggregates are the Result's `metrics` and the engine's report (see [Where each `MetricsOutput`
field went](#where-each-metricsoutput-field-went)); the per-request rows are `request_rows`
(see [Per-Request Rows](#per-request-rows)). Read the Result instead of a file:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios testdata/kernel/aisimulate \
  --rate 100 --num-requests 500 --output result.json
```

### Measured-Window Aggregates

The Result carries aiperf 0.12.0's request metrics (#1938) over the **measured window**:
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
| `gateway_queue_delay_{mean,p90}_ms` | flow-control queueing per measured request (0 without flow control) |
| `in_flight_requests_{mean,max}` | engine-side in-flight requests (dispatched, not completed), time-weighted over the window |
| `kv_capacity_tokens_per_replica` | GPU KV pool of the smallest instance, tokens |

In the Result, `ttft_*`, `e2e_*` and the engine report's `cache_hit_rate` take these
measured-window values; `cache_hit_rate_whole_run` keeps the whole-run figure. The per-step
series is the engine report's `step_interval_*`, and `requests.completed` is the run-wide INV-1
count. Invariants: INV-M1 … INV-M4.

**Comparing a hit rate with aiperf 0.12.0.** Compare aiperf's `server_metrics_export.json`
`vllm:prefix_cache_hits` / `vllm:prefix_cache_queries` "profiling" totals with the Result's
`reports.engine.cluster.cache_hit_rate_whole_run`, not with the measured-window `cache_hit_rate`. aiperf takes that
total as last-profiling-scrape minus the scrape before the first PROFILING-tagged scrape
(`server_metrics/accumulator.py:190-194, 387-401`), and in 0.12.0 a scrape at warmup start
already carries the PROFILING tag. So the baseline precedes warmup, and the exported total
counts warmup plus profiling. Real 0.12.0 exports confirm it: the profiling total equals the
warmup total plus the post-start slices at every level. Under agentx replay the warmup prefills
are cold, so that figure runs several points below the profiling window's own.

### Kernel Provenance

The Result's `reports.kernel` is keyed by pool role (`colocated`, or `prefill` and `decode`), #1942:

```json
{
  "reports": {
    "kernel": {
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
}
```

The entries are blis-latency-kernel's own `kernel.CoefficientOrigin` records (hence their
field-name keys); `evidenced`/`total` are its `Evidence()`. `fallback_or_assumed` lists the
registry entries copied from another scope or assumed, `assumptions` the values the kernel
supplied itself. `skipped_rules` are the scenario's release rules that neither the kernel nor
BLIS enforces, and `skipped_rule_findings` what those rules report when the pack is run
report-only. `evidence_source` is `kernel` when the counts are the kernel's `Evidence()`. A run
no kernel priced states an empty map.

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
point, and the Result says so:

```json
{
  "overrides": [
    {"setting": "scheduler", "input": "priority-fcfs", "run": "sjf", "source": "--scheduler"}
  ]
}
```

In a prefill/decode deployment each pool runs its own engine's policy, so a run-level choice
is recorded once per pool whose engine stated otherwise, as `"scheduler (prefill pool)"` or
`"preemption_policy (decode pool)"`.

A warning on stderr says the same.

### Catalog Provenance

The Result records **which model catalog produced it** (#1732) under `data.catalog`: its
source (`embedded` or `path`), content hash and overlays (SIM-6, X8). The data is an input of
the run, so two runs on different data are different inputs (INV-6). The Result never records
the path, a fact of the host.

### Saturation Detection

Post-hoc saturation analysis classifies a completed run as `STABLE`, `BACKLOGGED` or
`OVERLOADED`. It is distinct from the llm-d router's saturation detector, which gates admission
and flow control during the run. Select detectors with
`--detectors <name|comma-list|all>` (`composite`, `threshold`, `backlog-drift`, `peak-rate`),
tune them with `--saturation-config`, and write the per-event verdict trace with
`--saturation-report`:

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios testdata/kernel/aisimulate \
  --rate 10 --num-requests 100 --detectors all --saturation-report sat.json
```

Detector mechanics and tuning knobs:
[Saturation analyzer](../contributing/saturation-analyzer-extension.md).

### Per-Request Rows

`request_rows` holds one row per generated request (DEV-9), in creation order. A latency is
stated only where the request produced the sample: no token, no `ttft_us`; fewer than two
output tokens, no `tpot_us`; not completed, no `e2e_us`.

| Field | Unit | Description |
|-------|------|-------------|
| `id` | string | Request identifier, unique within the run (SIM-8) |
| `fate` | string | `completed`, `dropped`, `timed_out`, `in_flight_at_horizon`, `queued_at_horizon` or `not_started` |
| `drop_reason` | string | Why the request was dropped; present exactly when `fate` is `dropped` |
| `arrival_us` | µs | Arrival on the simulated clock |
| `ttft_us`, `tpot_us`, `e2e_us` | µs | Time to first token; time per output token, `(e2e - ttft) / (output tokens - 1)`; request latency |
| `scheduling_delay_us` | µs | Arrival at an engine to the first scheduled step |
| `gateway_queue_delay_us` | µs | Time in the router's flow-control queue — omitted without flow control |
| `completion_index` | count | 1-based position in the run's completion order, for a completed request |
| `prompt_tokens`, `output_tokens` | count | Tokens in the prompt, and tokens the engine emitted |
| `cached_local_tokens`, `cached_external_tokens` | count | Prompt tokens the device prefix cache, and the KV offload tier, supplied at first admission |
| `length_capped` | bool | The request completed at a length limit — omitted when `false` |
| `phase`, `instance`, `model`, `adapter`, `slo_class`, `tenant_id` | string | Benchmark phase, serving instance, and workload tags — omitted when empty |
| `session_id`, `round_index` | string, int | Session link and round within it (`0` = first turn) |
| `agentic` | object | The request's place in an agentic replay — omitted for any other workload |

## Anomaly Counters

When anomalies are detected, `--view summary` prints `=== Anomaly Counters ===`:

| Counter | What It Means | Action |
|---------|--------------|--------|
| **Priority Inversions** | A lower-priority request was scheduled before a higher-priority one | Check scheduler choice — use `priority-fcfs` for SLO workloads |
| **HOL Blocking Events** | A long prefill blocked shorter requests | Enable chunked prefill: state `long_prefill_token_threshold: 256` in the scenario's pool `engine` block |
| **Rejected Requests (Admission)** | The llm-d router denied admission: an admitter plugin, or the legacy admission controller shedding a sheddable request (priority < 0) from a saturated pool | Check the `InferenceObjective` priorities and the saturation detector's parameters; see [Admission Control](admission.md) |
| **Shed (tier)** | Per-SLO-class count of requests the router shed (admission denials and flow-control TTL expiries) — printed as indented sub-items beneath Rejected Requests (Admission) | Raise the class's `InferenceObjective` priority, or the detector's thresholds |
| **Rejected Requests (Routing)** | No endpoint could serve the request: no ready candidates (all instances `Loading` or `Draining`), or every filter eliminated all | Increase `initial_nodes`, reduce `loading_delay.mean`, stagger drain operations, or check the router configuration's filters |
| **Dropped Unservable** | Empty prompt rejected by the gateway, or a request exceeds the context window (the scenario's `max_model_len`) or needs more KV blocks than exist | Check for empty inputs first; otherwise check `max_model_len`, deploy more GPUs per instance, or reduce max input tokens |
| **Timed Out Requests** | Request exceeded its client deadline before completing | Increase `--timeout` or reduce load |
| **Length-Capped Requests** | Request was force-completed when it reached `MaxModelLen` tokens during decode | Expected if workloads push against the scenario's `max_model_len` |
| **Gateway Queue Depth (horizon)** | Requests still waiting in the router's flow control when the simulation ended — printed only when `> 0` | Reduce arrival rate or increase cluster capacity |
| **Gateway Queue Rejected** | Requests flow control rejected because its queues were at capacity — printed only when `> 0` | Raise `flowControl.maxRequests` or the priority band's capacity in the EndpointPickerConfig |
| **Gateway Expired (TTL)** | Requests that waited in flow control past their queue-wait budget — printed only when `> 0` | Raise `flowControl.defaultRequestTTL`, or add capacity |
| **Encode Routing Rejections** | Requests rejected at an encode stage — printed only when `> 0` | — |

Each counter is one INV-1 bucket; see [Admission Control](admission.md) for how router outcomes
map onto them.

!!! note "Where the counters are in the Result"
    Every INV-1 counter is in the Result's `requests` ledger; **Shed (tier)** is `router.shed_by_class`, and **Priority Inversions** and **HOL Blocking Events** are `router.priority_inversions` and `router.hol_blocking_events`. `blis run` and `blis replay` render this section from one shared path, so both print the same counters.

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

The score, its per-metric components and the weights are in the Result at `router.fitness`;
`--view summary` also prints them on stderr. `--summarize-trace` (with `--trace-level decisions`)
is likewise at `router.trace_summary`.

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

For detailed analysis, read the Result's `request_rows` ([Per-Request Rows](#per-request-rows)):

```bash
./blis run --scenario llama-3.1-70b-instruct-h200-fp8-vllm-tp4.yaml \
  --scenarios testdata/kernel/aisimulate \
  --rate 100 --num-requests 500 --output result.json
```

Each row includes TTFT, E2E, scheduling delay, and its fate.

## Further Reading

- [Configuration Reference](../reference/configuration.md#fitness-evaluation) — fitness weight syntax
- [Tutorial: Capacity Planning](../getting-started/tutorial.md) — applying results to capacity decisions
