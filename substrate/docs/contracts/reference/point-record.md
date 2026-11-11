# Point record

<p class="lede">What the scoring harness writes for one measured point: its identity and
ledger unit, its closeness grade, every substitution the grade rests on, the builds and data
that produced it, and each predicted figure beside the measured one under one metric
identifier.</p>

The dashboard's scoring harness writes it (design §11.7); config search and the substrate's
gates read it, so it is defined here (SUB-19). `LoadPointRecord` reads JSON or YAML and
refuses an unknown key in either; `Record.Validate` checks it.

```json
--8<-- "docs/contracts/examples/point-record.json"
```

## Keys

<!-- fields: pointrecord.Record -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `PointRecord`. |
| `schema_version` | integer | yes | Always `1`. |
| `point_id` | string | yes | SHA-256 of the canonical JSON of `identity`, 64 lowercase hex characters. |
| `unit_id` | string | yes | The ledger unit (design §10.2). |
| `identity` | mapping | yes | The point's identifying fields in its source's terms, below. |
| `grade` | string | yes | `G1` to `G5` (design §11.3). |
| `g5_reason` | mapping | with `G5` | Why the point is not scored, below. |
| `substitutions` | list | no | Every departure from an exact comparison, sorted by kind and subject. |
| `build` | mapping | yes | The engine build that measured the point. |
| `simulator` | mapping | yes | The simulator build that scored it, and the data it read. |
| `harness_digest` | string | yes | The scoring harness image, `sha256:<64 hex>` (X8). |
| `figures` | list | unless `G5` | Predicted and measured figures, sorted by metric. |
| `settings` | list | no | Each classified engine setting, as recorded and as given to BLIS, sorted by arg. |
| `resolved` | list | no | Engine outcomes both sides report, sorted by quantity. Never passed to BLIS. |
| `replay` | mapping | no | The load generator's inputs, for an agentic point. |
| `facets` | mapping | yes | Labels for display and grouping. |
| `notes` | list | no | Why a reason or substitution applies; fixed text, one per subject. |
| `kernel_provenance` | mapping | no | The cost model's report of the entries it priced, verbatim. |

<!-- fields: pointrecord.Identity -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `source` | string | yes | The data source, as the ledger names it. |
| `model` | string | yes | The source's model name. |
| `hardware` | string | yes | The source's chip name. |
| `precision` | string | no | The source's precision. |
| `framework` | string | yes | The engine the source ran. |
| `benchmark_type` | string | yes | The source's benchmark type. |
| `recipe` | string | yes | The source's recipe fingerprint, or `row:<id>`. |
| `conc` | integer | yes | Concurrency. At least 1. |
| `measured_date` | string | yes | `YYYY-MM-DD`. |

## Grades

A point takes the highest-numbered grade whose conditions apply. Each substitution sets a
floor: those of grade G2 (`coefficient_method`, `coefficient_range`, `execution_mode`,
`kernel_hash`, `value_class`, `estimated`, `interim_status`, `resolved_differs`,
`client_difference`, `later_record`, `router_differs`), `other_release` for G3, and
`changed_functions` and `nightly_or_custom` for G4. A record whose grade is below its
substitutions' floor is refused, and a G2, G3 or G4 record names the substitution that
makes it so (COV-4).

<!-- fields: pointrecord.Substitution -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | One of the conditions above. |
| `subject` | string | yes | What was substituted: a field, a coefficient, a library, a release; for `router_differs`, the router the run used. |
| `detail` | string | for `client_difference` and `resolved_differs`; else no | What stood in, or what differed. |
| `record` | string | for `later_record` | The record the value was taken from, with its revision, for example a file and its commit. |
| `date` | string | for `later_record` | The date of that record, `YYYY-MM-DD`. |
| `entry` | string | for `router_differs` | The router table entry used, by name. Its name begins with the router, which is `subject`; its key is the router, its mode and its session-affinity setting. |
| `policy` | string | for `router_differs` | The llm-d router policy BLIS simulated in place of the run's router. |
| `evidence` | string | for `router_differs` | The router table's evidence for the match, as the table entry cites it. |

Four conditions beside the earlier ones name what the point must say. `client_difference`, `later_record` and `resolved_differs` are in design §11.3 and §11.2; `router_differs` follows the design change in blis-design#1, which is a draft and has not merged:

`client_difference`
:   The simulated requests differ from the ones the run's client sent (COV-12). `subject`
    says which part, for example the send gap; `detail` says how.
`later_record`
:   A fact of the cluster the run's own record lacks was taken from the benchmark's later
    record of the same runner. `subject` is the field, `record` the record and its revision,
    `date` the record's date.
`router_differs`
:   The run was measured behind a router BLIS does not host, and the router table maps it to
    an llm-d policy. `subject` is the router the run used, `entry` the table entry (which fixes the mode),
    `policy` the policy BLIS used, which must be one llm-d's router offers, and
    `evidence` the table's evidence. The point is never G1, and is not presented as if llm-d
    had routed the run.
`resolved_differs`
:   A resolved value the run logged differs from the engine's resolution (design §11.2).
    `subject` is the setting; `detail` gives both values.

A key belongs to its own kind: a `policy` on a `later_record` is refused.

<!-- fields: pointrecord.G5Reason -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `code` | string | yes | One of design §11.4: `code_not_visible`, `router_unsupported`, `value_not_recorded`, `field_not_modelled`, `field_unclassified`, `engine_unsupported`, `topology_unsupported`, `not_in_catalog`, `field_not_in_release`, `value_conflict`, `build_unparseable`. Any other is refused (COV-5). |
| `subject` | string | no | What the reason is about. |

## Builds

<!-- fields: pointrecord.Build -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `release` | string | no | The engine release the run used, when it can be read. |
| `image` | string | yes | The image that measured the point. |
| `commit` | string | no | The upstream commit, 40 lowercase hex characters. |
| `source` | string | with `commit` | Where it was read: `image_label`, `label_by_tag`, `server_log_banner`, `image_tag` or `release_tag`. |
| `version` | string | no | The version the server log's banner printed. |

<!-- fields: pointrecord.Simulator -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `build` | string | yes | A release tag, or the commit of a branch or pull request. |
| `engine` | mapping | yes | The engine release the build runs, as a [Result's](result.md#engine). |
| `data` | mapping | yes | The data it read, as a [Result's](result.md#data). |

## Figures

<!-- fields: pointrecord.Figure -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `metric` | string | yes | A [metric identifier](metrics.md); both sides are computed under it. |
| `predicted` | number | yes | The simulated figure. |
| `measured` | number | yes | The measured figure. |
| `predicted_samples` | integer | yes | Samples behind the prediction. |
| `measured_samples` | integer | yes | Samples behind the measurement. |
| `interval` | mapping | no | The prediction's replication interval. |

<!-- fields: pointrecord.Interval -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `low` | number | yes | Lower bound; at most `predicted`. |
| `high` | number | yes | Upper bound; at least `predicted`. |
| `level` | number | yes | Confidence level, in (0, 1). |
| `replications` | integer | yes | At least 2. |

## Settings and outcomes

A value is as the run printed it: a string (a float keeps its printed text), an integer, a
boolean, or null.

<!-- fields: pointrecord.Setting -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `arg` | string | yes | The engine's own setting name. |
| `measured` | scalar | yes | As the run recorded it. |
| `source` | string | no | Where it was read. |
| `blis` | scalar | yes | As parsed back from the scenario handed to BLIS. |
| `field` | string | with `mapped` | The scenario field it was written to. |
| `match` | string | yes | `exact`, `mapped`, `inert`, `estimated`, `differs`, `absent_in_blis`, `unreadable` or `conflict`. |
| `rule` | string | with `estimated` | The engine's estimation rule. |
| `occurrences` | integer | no | Times the value was found, when at least 2. |
| `values` | list | with `conflict` | Every value found. |
| `delta` | scalar | no | BLIS minus measured. |
| `ratio` | string | no | BLIS over measured, four significant figures. |
| `note` | string | no | Why, in fixed text. |

<!-- fields: pointrecord.Resolved -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `quantity` | string | yes | The outcome, such as `kv_cache_tokens_per_replica`. |
| `measured` | scalar | yes | As the run reported it. |
| `blis` | scalar | yes | As BLIS reported it. |
| `occurrences` | integer | no | As for a setting. |
| `delta` | scalar | no | As for a setting. |
| `ratio` | string | no | As for a setting. |
| `match` | string | yes | As for a setting. |

<!-- fields: pointrecord.Replay -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `client` | string | yes | The load generator. |
| `client_version` | string | yes | Its version. |
| `scenario` | string | yes | Its scenario. |
| `replayed_as` | string | no | The BLIS replay scenario. |
| `concurrency` | integer | yes | Lanes. At least 1. |
| `duration_s` | number | no | Benchmark duration. |
| `trajectory_start` | mapping | no | The snapshot window. |
| `corpus` | string | yes | The corpus. |
| `corpus_sha256` | string | yes | Its digest. |
| `seed` | integer | yes | The seed. |

<!-- fields: pointrecord.TrajectoryStart -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `min_ratio` | number | no | Lower bound, a fraction of the span. |
| `max_ratio` | number | no | Upper bound. |

## Facets and notes

<!-- fields: pointrecord.Facets -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `sweep_id` | string | yes | Groups one measured curve: one deployment, at most one point per concurrency. |
| `scenario` | string | yes | The deployment's name. |
| `label` | string | no | The workload label, such as `8k1k` or `agentx`. |
| `workload` | string | no | The workload. |
| `model` | string | yes | The catalog model BLIS priced. |
| `gpu` | string | no | The chip. |
| `precision` | string | no | The precision. |
| `framework` | string | no | The engine. |
| `serving` | string | no | `aggregated` or `disaggregated`. |
| `spec_method` | string | no | The speculative method, or `none`. |
| `parallelism` | mapping | no | The layout. |

<!-- fields: pointrecord.Parallelism -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tp` | integer | no | Tensor-parallel width. |
| `pp` | integer | no | Pipeline-parallel width. |
| `dp` | integer | no | Data-parallel width. |
| `ep` | integer | no | Expert-parallel width. |

<!-- fields: pointrecord.Note -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `subject` | string | yes | The reason or substitution it explains. |
| `text` | string | yes | Fixed text. |

Source: [`spec/pointrecord`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/pointrecord).
