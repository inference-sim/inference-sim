# Result

<p class="lede">What one BLIS run predicts: one configuration, at one load, with one seed.
Its metrics are defined in the measured-run vocabulary, so a prediction and a
<a href="measurement.md">measured run</a> of the same configuration compare by metric name
and definition.</p>

`blis run` and `blis replay` write a Result to stdout, JSON by default and YAML on request;
`--output` writes the same bytes. `result.Marshal` writes either encoding with the keys below,
and `LoadResult` (`result.Unmarshal`) reads either, refusing an unknown key: a document whose
first non-blank character is `{` is read as JSON, anything else as YAML. A Result is written
with `result.Marshal`, not `encoding/json`, which reads none of these keys and drops a pool's
engine block. The document embeds the
input it was produced from, resolved, and the revision of everything it read: given that input
at the same revisions, the simulator reproduces the document byte for byte outside `runtime`,
`rerun_of` and `revision_changes`.

```json
--8<-- "docs/contracts/examples/result.json"
```

## Keys

<!-- fields: result.Result -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `Result`. |
| `schema_version` | integer | yes | Always `1`. |
| `name` | string | yes | The run's identity, a lowercase slug. |
| `input` | mapping | yes | Everything the run was given, resolved, below. |
| `revisions` | mapping | yes | The versions of every component and data repository the run read, below. |
| `rerun_of` | mapping | no | The Result whose input this run was produced from. |
| `revision_changes` | list | no | Each component whose revision differs from `rerun_of`'s. Only with `rerun_of`. |
| `metric_defs` | mapping | yes | Each metric's [definition](measurement.md#metric_defs), keyed by metric name, on the `simulated` clock, with `formula`, `population`, `clock` and `statistic` (`denominator` for a throughput) all stated. |
| `window` | mapping | yes | The measured window the metrics are over. |
| `metrics` | mapping | yes | Metric name to predicted value, finite and not negative; every name has a `metric_defs` entry and every entry a value. |
| `requests` | mapping | yes | The request-conservation ledger. |
| `goodput` | mapping | no | SLO outcome, when the input states SLO targets. |
| `breakdowns` | mapping | no | The metrics split along the axes the run exercised. |
| `engine` | mapping | no | Reported by the inference-engine model, keyed by pool role. |
| `router` | mapping | no | Reported by the embedded router. |
| `kernel` | mapping | no | Reported by the latency kernel, keyed by pool role. |
| `saturation` | mapping | no | The post-hoc saturation report, when detectors ran. |
| `overrides` | list | no | Every setting simulated differently from `input`, and what chose it. |
| `artifacts` | mapping | no | Bulk outputs by reference, keyed by role, a lowercase slug (`per_request`, `trace`, ...). |
| `runtime` | mapping | no | The only part that depends on the host or the wall clock. |

## Standard metric names

A Result and a measured run use the same names for the same definitions. A metric outside
this table takes a name of the same form and its own `metric_defs` entry. A Result that uses a
name from this table states the `formula`, `statistic` and `denominator` the table fixes for
it; the validator refuses another.

| Name | Definition |
|---|---|
| `output_tok_s` | Output tokens over the measured window (`output_throughput`, `denominator: profiling_window`). |
| `output_tok_s_per_gpu` | `output_tok_s` over the GPUs the deployment's pools use (`denominator: per_gpu`). |
| `interactivity_mean` | Mean over requests of (output tokens − 1) / (E2E − TTFT), tokens/s/user (`user_tps`). |
| `interactivity_p50`, `interactivity_p90` | 1000 over the p50 and p90 of per-request ITL (`interactivity`). |
| `ttft_mean_ms`, `ttft_p50_ms`, `ttft_p90_ms`, `ttft_p99_ms` | Time to first token (`ttft`), statistic as named. |
| `itl_mean_ms`, `itl_p50_ms`, `itl_p90_ms`, `itl_p99_ms` | Per-request inter-token latency per token (`tpot_excl_first`). Never the per-decode-step interval. |
| `e2e_mean_ms`, `e2e_p50_ms`, `e2e_p90_ms`, `e2e_p99_ms` | Request latency (`e2el`). |
| `request_throughput` | Completed requests over the measured window (`request_throughput`, `denominator: profiling_window`). |

## `input`

<!-- fields: result.Input -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `scenario` | mapping | yes | The [scenario](scenario.md), resolved. |
| `deployment` | mapping | yes | The [deployment](deployment.md), resolved, checked against the scenario's cluster. |
| `policy` | mapping | no | The simulator's policy bundle, verbatim, when one was given. |
| `run` | mapping | yes | The run-level controls, below. |

### `input.run`

<!-- fields: result.Run -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `load` | mapping | yes | The offered [load](measurement.md): exactly one of `concurrency` and `request_rate`. |
| `num_requests` | integer | yes | The most requests the run generates. Positive; `requests.generated` is at most this, fewer when a horizon or a closed-loop workload ends generation first. |
| `seed` | integer | yes | The run's seed. |
| `horizon_s` | number | no | Simulated-time bound in seconds. Positive. |
| `warmup` | mapping | no | The leading part excluded from the measured window. |
| `workload` | mapping | no | The [workload binding](workload.md) the run used. |

<!-- fields: result.Warmup -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `requests` | integer | one of | Requests excluded from the measured window; fewer than `num_requests`. |
| `duration_s` | number | one of | Seconds excluded from the measured window. |

## `revisions`

<!-- fields: result.Revisions -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `blis` | string | yes | The simulator build. |
| `blis_schemas` | string | yes | The schema release the document was written against. |
| `blis_latency_kernel` | string | no | The latency kernel. |
| `llm_d_router` | string | no | The embedded router. |
| `engine` | mapping | no | The inference-engine model, below. |
| `catalog` | mapping | no | The catalog, by release, revision, or both. |
| `registry` | mapping | no | The coefficient registry, by release, revision, or both. |

<!-- fields: result.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | The engine model, for example `vllm`. |
| `upstream_version` | string | yes | The real engine release it models; equal to `input.scenario.engine_version`. |
| `release` | string | no | The engine model's own release. |

<!-- fields: result.DataRepo -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `release` | string | one of | A release tag. |
| `revision` | string | one of | A commit. |

## `rerun_of` and `revision_changes`

A Result produced from another Result's input names it, and lists every component whose
revision differs. Both are absent on a fresh run.

<!-- fields: result.Reference -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The other Result's `name`. |
| `sha256` | string | yes | The other Result document's digest. |

<!-- fields: result.RevisionChange -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `component` | string | yes | A key of `revisions`, listed once. |
| `from` | string | yes | The other Result's revision. |
| `to` | string | yes | This run's revision, as `revisions.<component>` states it (for a mapping, its `release` or `revision`; for `engine`, its `release` or `upstream_version`); differs from `from`. |

## `window`

<!-- fields: result.Window -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `start` | string | yes | What opens the window, for example `first_send_after_warmup`. |
| `stop` | string | yes | What closes it, for example `last_completion`. |
| `duration_s` | number | yes | Its length in simulated seconds; positive when a throughput is reported. |
| `requests` | integer | yes | Requests completed inside it; at most `requests.completed`. |

## `requests`

Every injected request ends in exactly one terminal bucket: `injected` equals the sum of
`completed` through `encode_routing_rejections`, and `generated` equals `injected` plus
`rejected`.

<!-- fields: result.Requests -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `generated` | integer | yes | Every request the workload produced. |
| `rejected` | integer | yes | Refused at admission, never injected. |
| `injected` | integer | yes | Entered the simulator. |
| `completed` | integer | yes | Finished. |
| `still_queued` | integer | yes | Waiting in an engine queue at the end. |
| `still_running` | integer | yes | Running at the end. |
| `dropped_unservable` | integer | yes | Dropped as unservable. |
| `timed_out` | integer | yes | Timed out. |
| `routing_rejections` | integer | yes | No endpoint could serve. |
| `gateway_queue_depth` | integer | yes | Waiting in the router's flow control at the end. |
| `gateway_queue_rejected` | integer | yes | Rejected by flow control at capacity. |
| `gateway_evicted` | integer | yes | Revoked in flight by the router. |
| `gateway_expired` | integer | yes | Waited past the queue-wait budget. |
| `encode_routing_rejections` | integer | yes | Rejected at an encode stage. |
| `length_capped` | integer | yes | Completed requests cut at the length limit; not a terminal bucket. |

## `goodput`

<!-- fields: result.Goodput -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `targets` | mapping | yes | SLO targets by class, as in a [trace header](workload.md), validated as there. |
| `goodput_rps` | number | yes | Requests per second that met their targets. |
| `slo_attainment` | number | yes | Fraction of requests that met their targets, in [0, 1]. |
| `per_class` | mapping | no | The outcome per class, below; every key is a class of `targets`. |

<!-- fields: result.ClassGoodput -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `completed` | integer | yes | Completed requests of the class. |
| `met` | integer | yes | Those that met the class's targets; at most `completed`. |

## `breakdowns`

Each slice maps a metric name, a lowercase slug, to a finite value that is not negative.
`per_model`, `per_tenant` and `per_adapter` split the Result's own metrics, so each of their
names has a `metric_defs` entry and its value is in that entry's units. `pd` and `sessions`
also carry quantities with no measurement formula (a KV-transfer time, turns per session),
named in the same `<quantity>_<statistic>_<unit>` form.

<!-- fields: result.Breakdowns -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `pd` | mapping | no | The prefill/decode split of a disaggregated run. |
| `per_model` | mapping | no | By model. |
| `per_tenant` | mapping | no | By tenant. |
| `per_adapter` | mapping | no | By LoRA adapter. |
| `sessions` | mapping | no | Over multi-turn sessions. |

## `overrides`

<!-- fields: result.Override -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `setting` | string | yes | The setting simulated differently. |
| `input` | string | no | The value `input` states. |
| `run` | string | no | The value the run simulated. |
| `source` | string | yes | What chose it, for example a command-line flag. |

## `artifacts`

<!-- fields: result.Artifact -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `ref` | string | yes | Path, relative to the Result document. |
| `sha256` | string | yes | The file's digest. |
| `rows` | integer | no | Row count, for a table. |

## `runtime`

Determinism and round-trip comparisons skip this block, and nothing else.

## Validation

`Validate(Bundle{Result: r})` checks the Result field by field, the input it embeds included.
Engine rules do not run on that embedded input: pass `&r.Input.Scenario` and
`&r.Input.Deployment` as the bundle's `Scenario` and `Deployment` to rule-check it. With
`Rules` set, a Result and no `Scenario`, the rule layer warns (`result-input-rules`).

<!-- fields: result.Runtime -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `wall_clock_s` | number | no | Wall-clock seconds the run took. |
| `started_utc` | string | no | When it started, RFC 3339 in UTC, ending `Z`. |
| `host` | mapping | no | Host facts, for example `os` and `arch`. |
