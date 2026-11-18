# Result field map

Where each field of the MetricsOutput JSON lives in the Result document, or why it was dropped. Read by cmd/result_fieldmap_test.go.

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
| `request_itl_mean_ms` | `metrics.interactivity_of_mean_tpot` is `1000 /` it |
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
