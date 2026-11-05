# Measured run

<p class="lede">A measured sweep recorded as a validation case: where every number came
from, under what terms, how each metric was defined, and how far the configuration BLIS
would simulate is from the one that ran. The verdict is computed from those facts, never
read from the file.</p>

A measured run supersedes the [evaluation run](evaluation.md) for scoring. Runs live in
blis-catalog under `measurements/<license_ref>/<name>.yaml`, beside the licence registry
`measurements/sources.yaml`.

```yaml
--8<-- "docs/contracts/examples/measured-run.yaml"
```

## Keys

<!-- fields: measurement.MeasuredRun -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `MeasuredRun`. |
| `name` | string | yes | The run's identity, a lowercase slug. In the catalog, also its file name. |
| `schema_version` | integer | yes | Always `1`. |
| `source` | mapping | yes | Where the numbers came from, below. |
| `model` | string | no | The catalog model that was served. |
| `engine_version` | string | no | The engine release that ran. |
| `cluster` | mapping | no | The [cluster](scenario.md) it ran on. Coefficients are not part of a measurement, so this is the cluster alone. |
| `deployment` | mapping | no | The [deployment](deployment.md) that ran, checked against `cluster` when both are present. |
| `router` | mapping | no | The router in front of the deployment, below. |
| `workload` | mapping | no | The [workload binding](workload.md) that reproduces the request stream. |
| `client` | mapping | yes | The load generator, below. |
| `metric_defs` | mapping | yes | Each metric's definition, keyed by metric name. |
| `points` | list | yes | One entry per load level, strictly increasing. The first is the anchor shape error is measured from. |
| `provenance` | list | no | One row per setting, comparing the sources that state it. |
| `diff` | list | no | Each difference between the measured configuration and BLIS's. |
| `verdict` | string | no | `EXACT`, `PARTIAL` or `NOT_REPRODUCIBLE`. If present, must equal the computed verdict. |
| `resolution` | string | with `EXACT` | `STRICT` or `RESOLVED`. If present, must equal the computed resolution. |
| `duplicate_of` | string | no | Another run with the same model, hardware, loads and values. |
| `supersedes` | string | no | A run this one replaces. |

### `source`

<!-- fields: measurement.Source -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | The publisher. One of the [source kinds](vocabularies.md#measured-runs). |
| `adapter` | mapping | yes | The extracting code: `name`, `version` and the `sha256` of its source. |
| `inputs` | list | yes | Every file read, each with its digest. At least one has role `report`. |
| `extraction` | mapping | yes | How values were read, below. |
| `retrieved_utc` | string | no | When the inputs were fetched, RFC 3339 in UTC. |
| `license_ref` | string | yes | The entry in `measurements/sources.yaml` whose terms govern the run. |

<!-- fields: measurement.Adapter -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The adapter's path or name. |
| `version` | string | yes | Its version. |
| `sha256` | string | yes | Digest of its source, 64 lowercase hex characters. |

<!-- fields: measurement.Input -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `role` | string | yes | What the file is to the run. One of the [input roles](vocabularies.md#measured-runs). |
| `url` | string | `url` or `path` | Where it was fetched from, http(s). |
| `repo` | string | no | The repository holding it. |
| `commit` | string | no | The commit it was read at. |
| `path` | string | `url` or `path` | Its path in the repository or the local store. |
| `git_blob` | string | no | Its git blob id. |
| `size` | integer | no | Its size in bytes. |
| `sha256` | string | yes | Its digest, 64 lowercase hex characters. |
| `fetched_utc` | string | no | When it was fetched, RFC 3339 in UTC. |
| `note` | string | no | Anything a reader needs, such as how an excerpt was cut. |

<!-- fields: measurement.Extraction -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `method` | string | yes | How values were read. Every method parses; none executes downloaded content. |
| `selectors` | list | no | What was read: literal names, columns, row ids. |
| `label_verified` | boolean | no | Whether each metric label was checked against the harness's definition. |

### `client`

<!-- fields: measurement.ClientSpec -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `harness` | string | no | The load generator. A run without one cannot be EXACT. |
| `harness_version` | string | no | Its version. |
| `image_digest` | string | no | The client image, pinned by `sha256:<64 hex>`. |
| `command` | list | no | The argv as recorded. |
| `seed` | integer | no | The client seed. 0 is a seed. |
| `num_prompts` | integer | no | Requests per level. Not negative. |
| `burstiness` | number | no | Gamma shape of the arrivals; 1 is Poisson. Positive. |
| `ramp_up` | mapping | no | A rate ramp: `strategy` (`linear` or `exponential`), `start`, `end`. |
| `num_warmups` | integer | no | Warmup requests, excluded from the metrics. |
| `ignore_eos` | boolean | no | Whether output ran to `max_tokens` regardless of EOS. |
| `endpoint` | string | no | The API endpoint, such as `completions` or `chat`. |
| `tokenizer` | mapping | no | The tokenizer the client counted with: `repo` (required), `revision`, `sha256`, `mode`. |
| `duration_s` | number | no | Measurement duration per level, seconds. |
| `grace_s` | number | no | Grace period after the duration, seconds. |

<!-- fields: measurement.RampUp -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `strategy` | string | yes | `linear` or `exponential`. |
| `start` | number | yes | Starting rate, requests per second. |
| `end` | number | yes | Final rate, requests per second. |

<!-- fields: measurement.Tokenizer -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `repo` | string | yes | The tokenizer's repository. |
| `revision` | string | no | The revision it was read at. |
| `sha256` | string | no | Digest of the tokenizer files. |
| `mode` | string | no | The tokenizer mode, when the engine selects among several. |

### `metric_defs`

Each definition is keyed by the metric name the points use. A field left out is unstated,
and any unstated `formula`, `population`, `clock`, or `statistic` (`denominator` for a
throughput) keeps the run from EXACT: a scorer cannot tell whether a BLIS metric is the
same quantity.

<!-- fields: measurement.MetricDef -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `formula` | string | no | The quantity, such as `ttft` or `output_throughput`. |
| `statistic` | string | no | `mean`, `median`, `std`, `max`, or a percentile `pNN`. Not for a throughput. |
| `population` | string | no | What the statistic is over: `per_request`, `pooled_chunks` or `pooled_tokens`. |
| `denominator` | string | no | What a throughput divides by. Only for a throughput. |
| `clock` | string | no | `client` or `server` timestamps; `simulated` only in a [Result](result.md). |
| `start_event` | string | no | When a latency starts. |
| `stop_event` | string | no | When a latency stops. |
| `output_len_source` | string | no | Where output token counts come from. |
| `reasoning_counted_as_content` | boolean | no | Whether reasoning tokens count as output. |
| `percentile_method` | string | with a percentile | How a percentile interpolates; `numpy_linear`. |
| `unit` | string | yes | `ms`, `tokens_per_s` or `requests_per_s`; must be the formula's. |
| `cite` | string | yes | Where the definition is read from: a harness file and line, or the report's text. |

### `points`

<!-- fields: measurement.Point -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | Unique within the run. |
| `load` | mapping | yes | `concurrency`, `request_rate`, or both. Every point uses the same ones. |
| `metrics` | mapping | yes | Measured values keyed by a name in `metric_defs`, each `value` finite and not negative. |
| `completed` | integer | no | Requests completed. |
| `failed` | integer | no | Requests failed. |
| `realized` | mapping | no | `mean_isl` and `mean_osl` actually served. |
| `kv_utilization` | number | no | Fraction of the KV pool in use, in [0, 1]. |
| `prefix_hit_rate` | number | no | Fraction of prompt tokens served from cache, in [0, 1]. |
| `preemptions` | integer | no | Requests evicted and recomputed. |
| `noise` | mapping | no | Run-to-run spread: `seeds` (at least 2) and `cv`. |

<!-- fields: measurement.Load -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `concurrency` | integer | `concurrency` or `request_rate` | The client's concurrency cap. |
| `request_rate` | number | `concurrency` or `request_rate` | Offered requests per second. An unbounded rate under a cap is the cap alone. |

<!-- fields: measurement.Value -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `value` | number | yes | The measured value, in the definition's unit. |
| `report_label` | string | no | The label the report gave it. |
| `raw_key` | string | no | The key, column or row it was read from. |
| `label_verified` | boolean | no | Whether the label was checked against the definition. |

<!-- fields: measurement.Realized -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `mean_isl` | number | no | Mean input length served. |
| `mean_osl` | number | no | Mean output length served. |

<!-- fields: measurement.Noise -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `seeds` | integer | yes | Seeds the spread was measured over, at least 2. |
| `cv` | number | yes | Coefficient of variation across them. |

### `provenance`

One row per setting, comparing the report, the script that produced it, the run's own
artifact, and the value BLIS would use.

<!-- fields: measurement.ProvenanceRow -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `setting` | string | yes | The setting, unique within the run. |
| `report` | mapping | no | The report's value. |
| `script` | mapping | no | The producing script's value. |
| `artifact` | mapping | no | The run artifact's value. |
| `blis` | mapping | no | The value BLIS uses. |
| `verdict` | string | yes | `MATCH` or `MISMATCH` (at least two values listed), `UNSTATED` (at most one of report, script, artifact), or `UNVERIFIABLE` (report and script, no artifact). |
| `resolved_by` | string | no | For `UNSTATED` only: the engine rule, at the pinned version, that fixes the value. |
| `material` | string | yes | `yes`, `no`, `unknown`, or `bounded`. |
| `bound` | string | with `bounded` | How far it can move a metric. |
| `note` | string | no | Anything a reader needs. |

<!-- fields: measurement.Observed -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `value` | string | yes | The value as that source states it. |
| `ref` | string | except under `blis` | Where the source states it. |
| `commit` | string | no | The commit the source was read at. |

### `diff`

<!-- fields: measurement.DiffEntry -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `path` | string | yes | The field that differs. |
| `measured` | string | no | The measured value. |
| `blis` | string | no | The value BLIS uses. |
| `kind` | string | yes | `not_expressible`, `not_modelled`, `conflict`, `unresolved`, or `approximated`. |
| `removable` | boolean | no | Whether a change in `owner_repo` would close it. |
| `owner_repo` | string | when removable | The repository that owns the change. |
| `material` | string | yes | `yes`, `no`, `unknown`, or `bounded`. |
| `bound` | string | with `bounded` | How far it can move a metric. |
| `issue` | string | no | The issue tracking it. |

### `router`

The router is described by kind; an endpoint picker is pinned by the digest of its
config, whose bytes the simulator reads.

<!-- fields: router.Router -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | `direct` or `epp`. |
| `epp` | mapping | with `epp` | The picker. Not allowed with `direct`. |

<!-- fields: router.EPP -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `version` | string | yes | The picker's version. |
| `image` | string | no | Its image. |
| `config` | mapping | yes | Its plugin config: `path` and `sha256`. |
| `flow_control` | string | no | A summary of the flow-control settings. |

<!-- fields: router.ConfigRef -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `path` | string | yes | The config file. |
| `sha256` | string | yes | Its digest, 64 lowercase hex characters. |

## Verdict

The verdict is computed; a stated one must match.

`NOT_REPRODUCIBLE`
:   BLIS cannot build the run: `model`, `engine_version`, `cluster`, `deployment` or
    `workload` is missing, or a `diff` entry is `not_expressible`.

`EXACT`
:   Otherwise, when the diff is empty, no provenance row is `MISMATCH` or `UNVERIFIABLE`,
    every `UNSTATED` row has `resolved_by`, every metric definition is complete, the client
    harness is recorded, and the licence permits scoring. Resolution is `RESOLVED` when a
    row was resolved by a rule and `STRICT` otherwise. Only an EXACT run may anchor
    calibration.

`PARTIAL`
:   Any other run. It is scored and reported apart from EXACT runs.

## Checks

A run is invalid if

- `kind` is not `MeasuredRun`, `name` is not a lowercase slug, or `schema_version` is not 1;
- a vocabulary value is not recognized, or a digest is not 64 lowercase hex characters;
- the source has no adapter name, version or digest, no inputs, no `report` input, or an
  input without a `url` or `path`;
- a timestamp is not RFC 3339 UTC, or the client image is not pinned by digest;
- the cluster, deployment, router or workload is invalid, or the deployment does not fit
  the cluster;
- a metric definition has no `unit` or `cite`, a unit that is not its formula's, a
  statistic on a throughput, a denominator on a latency, or a percentile without a method;
- there are no points, two share an `id`, a point states no load, points mix load modes or
  are not strictly increasing, or a metric has no definition or a negative or non-finite
  value;
- a provenance row's sources do not support its verdict, `resolved_by` is on a row that is
  not `UNSTATED`, or `bound` is missing for `bounded` or present otherwise;
- a removable difference names no `owner_repo`;
- a string carries a credential: a token, a private key, a Kubernetes Secret reference, or a
  kubeconfig path;
- a stated verdict or resolution is not the computed one.

## Licence registry

`measurements/sources.yaml` lists every source a run may cite. Only a `verified` entry
permits anything.

```yaml
--8<-- "docs/contracts/examples/measurement-sources.yaml"
```

<!-- fields: measurement.Registry -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `MeasurementSources`. |
| `sources` | list | yes | One entry per source, below. |

<!-- fields: measurement.License -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The key a run's `license_ref` names, a lowercase slug. |
| `title` | string | yes | What the source is. |
| `license` | string | when verified | The SPDX identifier the evidence states. |
| `status` | string | yes | `verified`, `unverified`, or `refused`. |
| `permits` | list | when verified | Uses the terms permit: `score`, `derive_coefficients`, `redistribute`. Empty unless verified. |
| `evidence` | list | when verified | Quoted statements of the terms. |
| `attribution` | string | no | What a use must carry. |
| `note` | string | unless verified | Why a source is unverified or refused. |

<!-- fields: measurement.Evidence -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `url` | string | yes | Where the terms are, http(s). |
| `quote` | string | yes | The text that states them. |
| `location` | string | yes | Where the quoted text is. |

A catalog run must cite a licence that permits `redistribute`, since a committed file is
redistributed; [`validate-measurements -local`](cli.md#validate-measurements) admits one
that permits only scoring, for output kept out of the repository.

Source: [`spec/measurement`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/measurement),
[`spec/router`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/router).
