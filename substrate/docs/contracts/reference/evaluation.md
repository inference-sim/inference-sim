# Evaluation run

<p class="lede">What a benchmark measured, one row per concurrency level, so that a
prediction for the same scenario can be compared with it. The fields follow what
benchmark harnesses already report.</p>

For a validation case, use a [measured run](measurement.md), which also records
provenance, licence terms and metric definitions.

```yaml
--8<-- "docs/contracts/examples/evaluation-run.yaml"
```

## Keys

<!-- fields: evaluation.Run -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `EvaluationRun`. |
| `name` | string | yes | The run's identity. |
| `scenario` | string | yes | The scenario that was measured. |
| `harness` | string | yes | The benchmark harness that produced the numbers. Harnesses compute percentiles differently, so a comparison needs to know which. |
| `harness_version` | string | no | Its version. |
| `engine_version` | string | yes | The engine release the run exercised. It may differ from the scenario's. |
| `date` | string | no | When the measurement was taken, as an ISO 8601 date. |
| `points` | list | yes | At least one point, each at a different concurrency. |

### `points`

Every field is as measured. Nothing is derived, so a reader can recompute any derived
quantity and check it.

<!-- fields: evaluation.Point -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `concurrency` | integer | yes | Simultaneous requests the harness kept in flight, at least 1. |
| `completed_requests` | integer | no | Requests finished at this level. |
| `output_tokens_per_sec` | number | no | Output throughput, tokens per second. |
| `input_tokens_per_sec` | number | no | Input throughput, tokens per second. |
| `ttft_ms` | number | no | Time to first token, ms. |
| `itl_ms` | number | no | Inter-token latency, ms. |
| `e2e_ms` | number | no | End-to-end latency, ms. |
| `mean_isl` | integer | no | Mean input length actually served. |
| `mean_osl` | integer | no | Mean output length actually served. |
| `kv_utilization` | number | no | Fraction of the KV pool in use, in [0, 1]. |
| `prefix_cache_hit_rate` | number | no | Fraction of prompt tokens served from cache, in [0, 1]. |
| `preemptions` | integer | no | Requests evicted and later recomputed. Nonzero means the engine ran out of KV cache space at this load. |

## Checks

A run is invalid if

- `kind` is not `EvaluationRun`, or `name`, `scenario`, `harness` or `engine_version` is
  empty;
- there are no points, a `concurrency` is below 1, or two points share a concurrency;
- any number is not finite;
- `output_tokens_per_sec`, `ttft_ms` or `itl_ms` is negative;
- `kv_utilization` or `prefix_cache_hit_rate` lies outside [0, 1];
- `preemptions` is negative.

A point with output throughput but neither `ttft_ms` nor `itl_ms` draws a warning: no
prediction can be compared with it.

Source: [`spec/evaluation`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/evaluation).
