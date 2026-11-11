# Result

<p class="lede">What one BLIS run predicts: one configuration under one experimental frame.
It names each figure by its identifier in the <a href="metrics.md">metric registry</a>,
records the engine, data and options the run used, and gives every request it generated one
fate.</p>

`blis run` and `blis replay` write a Result to stdout, JSON by default and YAML on request.
`result.Marshal` writes either encoding with the keys below, and `LoadResult`
(`result.Unmarshal`) reads either, refusing an unknown key: a document whose first non-blank
character is `{` is read as JSON, anything else as YAML. A Result is written with
`result.Marshal`, not `encoding/json`, which reads none of these keys and drops a pool's
engine block. Given the same input, options and data, the simulator reproduces the document
byte for byte outside `runtime`, `rerun_of` and `revision_changes`.

A number is finite, with one exception: `grace_period_s`, in the replay definition a Result
records, may be infinite, and `.inf` waits indefinitely. YAML writes it `.inf` and JSON writes the
string `"inf"`; either reads back as infinity, in that field only ([Numbers are
finite](../concepts/contracts.md#numbers-are-finite)).

```json
--8<-- "docs/contracts/examples/result.json"
```

## Keys

<!-- fields: result.Result -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | Always `Result`. |
| `schema_version` | integer | yes | Always `2`. |
| `name` | string | yes | The run's identity, a lowercase slug. |
| `engine` | mapping | yes | The engine the run used, below. |
| `data` | mapping | yes | The catalog and registry the run read, below. |
| `selected_entries` | mapping | yes | The registry entries lookups returned, and the files no lookup read, below. |
| `input` | mapping | yes | Everything the run was given, resolved, below. |
| `frame` | mapping | yes | The experimental frame, below. |
| `revisions` | mapping | yes | The BLIS module versions the simulator was built from. |
| `rerun_of` | mapping | no | The Result whose input this run was produced from. |
| `revision_changes` | list | no | Each component whose revision differs from `rerun_of`'s. Only with `rerun_of`. |
| `window` | mapping | yes | The measured window as it fell in this run. |
| `metrics` | mapping | no | Figures keyed by metric identifier, each one of `frame.metrics`. Absent when the run failed an acceptance condition. |
| `requests` | mapping | yes | The count of each fate. |
| `request_rows` | list | no | One row per generated request, below. |
| `goodput` | mapping | no | SLO targets and each class's outcome, when the input states targets. |
| `breakdowns` | mapping | no | The metrics split along the axes the run exercised. |
| `reports` | mapping | no | Each component's own report, carried as reported. |
| `overrides` | list | no | Every setting simulated differently from `input`, and what chose it. |
| `resolutions` | list | no | The engine's own resolution of the settings it resolves, beside `overrides`. |
| `artifacts` | mapping | no | Bulk outputs by reference, keyed by role, a lowercase slug. |
| `runtime` | mapping | no | The only part that depends on the host or the wall clock. |

## `engine`

<!-- fields: result.Engine -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | The engine, for example `vllm`; equal to `input.scenario.engine.kind` when the scenario states it. |
| `version` | string | yes | The upstream release it models; equal to the release the scenario names. |

## `data`

How the catalog and the registry reached the run (design §3.3). A content hash lists every
data file as `<sha256 of its bytes>  <path relative to the directory>`, sorts the lines by
path, ends each with a line feed, and takes the SHA-256 of the result.

<!-- fields: result.Data -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `catalog` | mapping | yes | The catalog, below. |
| `registry` | mapping | yes | The registry, below. |
| `released_data` | boolean | yes | True only for a tagged simulator release, run on its substrate release's embedded data, with no overlay. Only such a run's figures are published. |
| `build` | string | yes | The simulator build: a release tag `vX.Y.Z`, or `unreleased`. |

<!-- fields: result.DataSource -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `source` | string | yes | `embedded` (compiled into the substrate) or `path` (a directory named on the command line or in the environment). |
| `hash` | string | yes | Content hash of the base data, 64 lowercase hex characters. |
| `overlays` | list | no | Each overlay applied, in order. |
| `combined_hash` | string | yes | Content hash after every overlay; equal to `hash` with none. |

<!-- fields: result.Overlay -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `hash` | string | yes | The overlay directory's content hash. |

## `selected_entries`

The selected-entry report (VAL-6). It covers every reader behind the engine's
environment, the acceptance and routing profile readers included. A candidate registry
whose new entries no lookup selects shows them here as unread.

<!-- fields: result.SelectedEntries -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `closeness_version` | integer | no | `1` when the recorder passed on everything each coefficient's entry states: its `kernel_impls`, `fit_range` and `ci95`, and the flags found from them. It does not say the conditions were decided. Under `1`, `extrapolated` is evaluated only for a coefficient that has a `fit_range`, `unidentified` only for one that has a `ci95`, and the kernel hash only for one that has `kernel_impls`; where that data is absent the entry did not state it, and an omitted flag is "not evaluated", not false. Absent when the recorder held only identities, when no coefficient was recorded, or in a report written before the data existed; then no coefficient carries the data, and nothing is said. A coefficient carrying the data under an absent version is refused. A reader of the older strict layout fails on the new keys. |
| `coefficients` | list | no | Coefficients a lookup returned, once each, sorted by set, name, method and scope. |
| `acceptance_profiles` | list | no | Acceptance profiles a lookup returned, sorted by set and key. |
| `routing_profiles` | list | no | Routing profiles a lookup returned, sorted by set and key. |
| `unread_files` | list of strings | no | Registry files, relative and with forward slashes, from which no lookup returned an entry. Sorted; none of them is a selected entry's file. |

<!-- fields: result.SelectedCoefficient -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `set` | string | yes | The coefficient set. |
| `name` | string | yes | The coefficient's name. |
| `method` | string | yes | One of the registry's [methods](vocabularies.md). |
| `scope` | mapping | yes | The entry's [scope](coefficients.md), as written. |
| `file` | string | yes | The registry file it was read from. |
| `kernel_impls` | list | no | The kernel implementations the entry states it is valid for, copied from the entry: `library`, `variant` and `code_hash`, as in the [coefficient entry](coefficients.md). A reader maps each library version a benchmark run recorded to a kernel hash and compares it with these (design §11.3). Absent when the entry states none, and the hash of such an entry cannot be compared: a primitive priced from a `transferred`, `assumed` or `not_charged` entry, which states none, has no hash to compare. These are the implementations the entry is valid for, not the one the engine lowered to; the two agree where an entry states them (VAL-8). |
| `fit_range` | list | no | The range of inputs the entry's fit covered (design VAL-10), copied from the entry: a `dim`, `low` and `high` per dimension. Absent when the entry records none, and the extrapolation condition cannot then be evaluated for it. |
| `extrapolated` | boolean | no | True when a step the run priced read the entry outside `fit_range`. The cost model sets it from the trust verdict of each priced step, and the report reads it when the Result is written, so it covers the steps priced up to then; nothing in the Result says the run had finished. Requires `fit_range`. |
| `ci95` | list | no | The entry's 95% interval `[low, high]`, copied from the entry. Absent when the entry states none, and the identifiability condition cannot then be evaluated for it. |
| `unidentified` | boolean | no | True when the entry's data do not determine it: its 95% interval is unbounded or wider than half its value (design VAL-10). Known when the cost model is built. Requires `ci95`. |

Where an entry does not record a kernel implementation, a fit range or an interval, the Result
says so by leaving the key out, and a grader treats the condition it would decide as not
evaluated rather than as met (design §11.3). In the committed registry most measured entries
state none of the three, so most coefficients are not evaluated until the registry records them.

<!-- fields: result.SelectedProfile -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `set` | string | yes | The profile set. |
| `key` | string | yes | The profile's key, as `Key.String` renders it. |
| `file` | string | yes | The registry file it was read from. |

## `input`

<!-- fields: result.Input -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `scenario` | mapping | yes | The [scenario](scenario.md), resolved. |
| `deployment` | mapping | yes | The [deployment](deployment.md), resolved, checked against the scenario's cluster. |
| `packaging` | mapping | unless the scenario states `gpus_per_node` | The cluster's [packaging](scenario.md#packaging) as the run resolved it, with the source of each value (SIM-10). |
| `policy` | mapping | no | The simulator's policy bundle, verbatim, when one was given. |
| `options` | list | no | Every option that changes a byte of the Result (SIM-10), sorted by name. |
| `workload_properties` | list | no | Each workload property and what supplied it (D8), sorted by name. |

<!-- fields: scenario.Packaging -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `gpus_per_node` | integer | yes | GPUs per node the run simulated, at least 1. Equal to the scenario's `gpus_per_node` when it states one. |
| `gpus_per_node_source` | string | yes | `stated` (the scenario, for a chip whose entry leaves it open) or `catalog` (the chip's entry). |
| `gpus_per_rack` | integer | no | GPUs per rack, a multiple of `gpus_per_node`. Omitted when the chip has no rack tier. |
| `gpus_per_rack_source` | string | with `gpus_per_rack` | `catalog`: only a chip's entry states a rack. |

<!-- fields: result.Option -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The flag without its dashes, or the environment variable. |
| `value` | string | yes | The value the run used, as text. |
| `source` | string | yes | `command_line`, `environment`, or `default`. |

A workload property is a quantity that depends on the request stream but that BLIS does not
simulate, such as speculative acceptance or expert routing imbalance (D8).

<!-- fields: result.WorkloadProperty -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `name` | string | yes | The property, a lowercase slug. |
| `values` | mapping | unless `default` | The property's figures, by name. Finite. |
| `source` | string | yes | What supplied the values, highest precedence first: `command_line`, `scenario`, `workload` (stated by or derived from it), `profile` (the substrate's profile for the workload), `default` (none stated). |
| `profile` | string | with `profile` | The profile, as set and key. |
| `step` | string | for a profile lookup that can borrow | How the lookup found its value (D8): `exact`, `same_drafter_other_k`, `other_drafter` or `other_target`. |
| `differed` | list of strings | with a borrowing step | Each key field in which a borrowed profile differs from the run's: `model`, `drafter`, `speculative_method` or `num_spec_tokens`. None for `exact`. |
| `k_rule` | string | with `same_drafter_other_k` | How the borrowed value was brought to the run's K: `interpolated`, `above_largest`, `below_smallest_rates` or `below_smallest_proportional`. |
| `profiles` | list of strings | with `step` | Every profile the value was read from, as set and key, the one that sets it first; an interpolation lists both it lies between. |

## `frame`

The experimental frame (design §8.6, DEV-7): what the run feeds in, when it measures, and
what it measures.

<!-- fields: result.Frame -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `kind` | string | yes | `terminating` (a fixed amount of work) or `steady_state`. |
| `generator` | mapping | yes | The input, below. |
| `initial_state` | mapping | yes | The engines' caches at the start, below. |
| `phases` | mapping | yes | Warm-up, measured window and drain, each by a rule. |
| `stop` | mapping | yes | The stop rule: a request count, a horizon, or both. |
| `acceptance` | list | no | The conditions under which the run counts for this frame. |
| `metrics` | list of strings | yes | The [metric identifiers](metrics.md) the frame measures, sorted. |

<!-- fields: result.Generator -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `workload` | mapping | one of these | The [workload binding](workload.md) the run used. |
| `load` | mapping | one of these | The offered [load](measurement.md): a concurrency or a request rate, not both. |
| `seed` | integer | yes | The scenario seed every random substream derives from. |
| `rng` | string | yes | The random generator and its version, for example `pcg64-sha256/1`. |

<!-- fields: result.InitialState -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `caches` | string | yes | `empty` or `warmed`. |
| `rule` | string | with `warmed` | How the caches were warmed. |

<!-- fields: result.Phases -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `warmup` | mapping | yes | The leading part excluded from measurement. |
| `measured` | mapping | yes | What opens and closes the measured window. |
| `drain` | mapping | yes | What the run does after the window closes. |

<!-- fields: result.WarmupPhase -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `rule` | string | yes | `none`, `requests`, `duration`, `marked` (the workload marks warm-up requests) or `mser5`. |
| `requests` | integer | with `requests` | Leading requests excluded. Positive. |
| `duration_s` | number | with `duration` | Leading simulated seconds excluded. Positive. |

<!-- fields: result.MeasuredPhase -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `start` | string | yes | `first_arrival_after_warmup`, `profiling_start` or `simulation_start`. |
| `stop` | string | yes | `last_completion`, `profiling_end` or `simulation_end`. |

<!-- fields: result.DrainPhase -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `rule` | string | yes | `none`, `until_complete` or `grace`. |
| `duration_s` | number | with `grace` | The grace period in simulated seconds. Positive. |

<!-- fields: result.StopRule -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `requests` | integer | one of these | The most requests the generator emits. |
| `horizon_s` | number | one of these | The simulated time, in seconds, at which the run stops. Positive. |

<!-- fields: result.AcceptanceCondition -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `condition` | string | yes | `achieved_concurrency_within` (a fraction of the target) or `unfinished_fraction_at_most`. |
| `threshold` | number | yes | The bound. Not negative. |
| `observed` | number | yes | What the run achieved. Not negative. |
| `met` | boolean | yes | Whether `observed` is at most `threshold`. A run with any condition unmet reports no `metrics`. |

## `revisions`

<!-- fields: result.Revisions -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `simulator` | string | yes | The inference-sim build: a tag, a pseudo-version or a commit. |
| `substrate` | string | yes | The blis-substrate version the build selected. |
| `router` | string | no | The llm-d router fork's version. |
| `engine_module` | string | no | The engine repository's module version. |

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
| `component` | string | yes | A key of `revisions`, or `catalog` or `registry` for the data's combined hash; listed once. |
| `from` | string | yes | The other Result's value. |
| `to` | string | yes | This run's value; differs from `from`. |

## `window`

<!-- fields: result.Window -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `start_us` | integer | yes | The window's start on the simulated clock, microseconds. |
| `end_us` | integer | yes | Its end; not before `start_us`. |
| `duration_s` | number | yes | `end_us - start_us` in seconds; positive when a rate is reported. |
| `requests` | integer | yes | Requests completed inside it; at most `requests.completed`. |

## `metrics`

Each key is an identifier from the [metric registry](metrics.md). The registry gives its
formula, statistic, population, window and unit; the Result gives the value.

<!-- fields: result.Figure -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `value` | number | yes | Finite and not negative; at most 1 for a fraction. |
| `samples` | integer | yes | The size of the population the statistic is over. Positive for a latency. |

## `requests`

Every request the generator created has exactly one fate (design §8.7, DEV-9). `generated`
is the generator's own count, and equals the sum of the fates.

<!-- fields: result.Requests -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `generated` | integer | yes | Requests the workload created, every planned session round included. |
| `completed` | integer | yes | Finished. |
| `dropped` | mapping | no | Dropped, by reason: `admission_rejected`, `routing_rejected`, `encode_routing_rejected`, `gateway_queue_rejected`, `gateway_evicted`, `gateway_expired`, `unservable`, `context_overflow`, `decode_kv_unavailable`. Each count positive. |
| `timed_out` | integer | yes | Cancelled by the client's timeout. |
| `in_flight_at_horizon` | integer | yes | Running when the run stopped. |
| `queued_at_horizon` | integer | yes | Waiting when the run stopped. |
| `not_started` | integer | yes | Planned, such as a later session round, but never issued. |
| `length_capped` | integer | yes | Completed requests that stopped at a length limit. Not a fate. |

## `request_rows`

One row per generated request, replacing the `--metrics-path` and `--results-path` files.
Times are on the simulated clock in microseconds. A latency is present only where the
request produced the sample: a request that emitted no token has no `ttft_us`; only a
completed request with at least two output tokens has `tpot_us`; only a completed request
has `e2e_us`. The rows' fates agree with `requests`.

<!-- fields: result.RequestRow -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | The request's identifier, never reused in a run. |
| `fate` | string | yes | `completed`, `dropped`, `timed_out`, `in_flight_at_horizon`, `queued_at_horizon` or `not_started`. |
| `drop_reason` | string | with `dropped` | A reason from `requests.dropped`. |
| `arrival_us` | integer | yes | Arrival at the client. |
| `ttft_us` | integer | when sampled | Time to first token. |
| `tpot_us` | number | when sampled | (E2E - TTFT) / (output tokens - 1). |
| `e2e_us` | integer | when sampled | Request latency; at least `ttft_us`. |
| `scheduling_delay_us` | integer | no | Arrival at an engine to the first scheduled step. |
| `gateway_queue_delay_us` | integer | no | Time in the router's flow-control queue. |
| `completion_index` | integer | completed | The 1-based position in the run's order of completion. |
| `prompt_tokens` | integer | yes | Prompt tokens. |
| `cached_local_tokens` | integer | no | Prompt tokens served from the engine's own cache. |
| `cached_external_tokens` | integer | no | Prompt tokens served from an external cache tier. |
| `computed_tokens` | integer | no | Prompt tokens the engine computed. |
| `output_tokens` | integer | yes | Tokens emitted. |
| `length_capped` | boolean | no | Completed at a length limit. |
| `phase` | string | no | `warmup`, `profile` or `grace`, when the workload marks phases. |
| `instance` | string | no | The engine instance that served it. |
| `model` | string | no | The model tag. |
| `adapter` | string | no | The LoRA adapter. |
| `slo_class` | string | no | The SLO class. |
| `tenant_id` | string | no | The tenant. |
| `session_id` | string | no | The session. |
| `round_index` | integer | no | The round within the session. |
| `agentic` | mapping | no | The request's place in an agentic replay, below. |

<!-- fields: result.AgenticTag -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `lane` | integer | yes | The concurrency lane. |
| `tree` | integer | yes | The tree instance. |
| `trace_id` | string | yes | The corpus trace the tree replays. |
| `stream` | string | yes | The conversation. |
| `kind` | string | yes | `root`, `subagent` or `flat`. |
| `stream_instance` | integer | yes | The stream's instance within the run. |
| `turn` | integer | yes | The turn within the stream. |
| `outcome` | string | no | How the load generator would record it. |

## `goodput`

The figures `goodput_rps` and `slo_attainment` are metrics; this block holds the targets
and each class's counts.

<!-- fields: result.Goodput -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `targets` | mapping | yes | SLO targets by class, as in a [trace header](workload.md). |
| `per_class` | mapping | no | The outcome per class; every key is a class of `targets`. |

<!-- fields: result.ClassGoodput -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `completed` | integer | yes | Completed requests of the class. |
| `met` | integer | yes | Those that met the class's targets; at most `completed`. |

## `breakdowns`

`per_model`, `per_tenant` and `per_adapter` restate reported metrics over each key's
requests, so each of their names is a key of `metrics`. `pd` and `sessions` name
quantities of their own.

<!-- fields: result.Breakdowns -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `pd` | mapping | no | The prefill/decode split of a disaggregated run. |
| `per_model` | mapping | no | By model. |
| `per_tenant` | mapping | no | By tenant. |
| `per_adapter` | mapping | no | By LoRA adapter. |
| `sessions` | mapping | no | Over multi-turn sessions. |

## `reports`

<!-- fields: result.Reports -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `engine` | mapping | no | Reported by the engine, keyed by pool role. |
| `router` | mapping | no | Reported by the embedded router. |
| `kernel` | mapping | no | Reported by the cost model, keyed by pool role. |
| `saturation` | mapping | no | The post-hoc saturation report, when detectors ran. |

A report's numbers are not figures under the registry. A consumer that ranks or compares a
number reads it from `metrics`.

## `overrides`

<!-- fields: result.Override -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `setting` | string | yes | The setting simulated differently. |
| `input` | string | no | The value `input` states. |
| `run` | string | no | The value the run simulated. |
| `source` | string | yes | What chose it, for example a command-line flag. |

## `resolutions`

<!-- fields: result.Resolution -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `setting` | string | yes | The setting. The name an `overrides` entry uses for the same setting. |
| `pool` | string | no | The pool that resolved it, below. Absent for the whole deployment, or a deployment with one pool. |
| `host` | string | no | The value the host gave the engine: the host's value when `overrides` names this setting, else the input's. Absent when the host gave none and the engine chose one. |
| `engine` | string | yes | The value the engine resolved to, and the run simulated. |
| `reason` | string | when it differs | Why `engine` differs from `host`. Only where they differ. |

Resolutions are listed once each, sorted by `setting` and then `pool`. A pool is named by its
role (`colocated`, `prefill` or `decode`), and, when several pools share a role, by the role and
the pool's position among those of that role, counted from zero (`decode-0`, `decode-1`);
`result.PoolLabels` gives the names. A scenario names no pool, so the position in `pools` is the
only identity a pool has. A prefill and a decode pool resolve the same setting differently, and
each states its own.

An override is what the host changed; a resolution is what the engine then did with the value
it was given. Design §11.2 grades a benchmark point G2 when a resolved value the run logged
differs from the engine's resolution of the value passed, and this list is where the engine's
resolution is read, beside the host's value. `result.ResolutionsOf` writes it from the cost
model's own `kernel.Resolution`, and `result.ResolutionsOfPool` for one pool of several: the
layout widths and modes, and each request the cost model declined.

The settings the cost model resolves, and what vLLM calls the request each answers:

| `setting` | `engine` value | The vLLM setting it answers |
|---|---|---|
| `tensor_parallel_width` | the width, an integer | `tensor_parallel_size` |
| `data_parallel_width` | the width, an integer | `data_parallel_size` |
| `expert_parallel_width` | the width, an integer | None: it is derived. With `enable_expert_parallel` on it is tensor, prefill-context and data-parallel width multiplied; with it off it is 1. |
| `all_reduce_backend` | `custom` or `nccl` | `disable_custom_all_reduce`. The custom reduction runs unless disabled, and is declined where the group spans nodes without multi-node NVLink or has a rank count the kernel does not support. |
| `async_scheduling` | `true` or `false` | `async_scheduling`, a tri-state request that the engine resolves. |
| `cascade_attention` | `true` or `false` | The inverse of `disable_cascade_attn`: cascade attention is on when the request does not disable it and every gate allows it. |
| `sequence_parallel_moe` | `true` or `false` | None: whether the MoE input is sequence-parallel, from the engine's rule or the model's own, so a request to change it is not a single setting. |

A declined request is recorded under the setting it answers: the cost model's `allreduce_backend`
request lands on `all_reduce_backend`. A request for another field is recorded under the name the
cost model gives it, which is the engine's own setting name (for example `quantization`). The
host column is empty for `async_scheduling` and `cascade_attention` unless the cost model
declines a request for them, which the kernel does not do today.

## `artifacts`

<!-- fields: result.Artifact -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `ref` | string | yes | Path, relative to the Result document. |
| `sha256` | string | yes | The file's digest. |
| `rows` | integer | no | Row count, for a table. |

## `runtime`

Determinism and round-trip comparisons skip this block, and nothing else.

<!-- fields: result.Runtime -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `wall_clock_s` | number | no | Wall-clock seconds the run took. |
| `started_utc` | string | no | When it started, RFC 3339 in UTC, ending `Z`. |
| `host` | mapping | no | Host facts, for example `os` and `arch`. |

## Validation

`Validate(Bundle{Result: r})` checks the Result field by field, the input it embeds
included. Engine rules do not run on that embedded input: pass `&r.Input.Scenario` and
`&r.Input.Deployment` as the bundle's `Scenario` and `Deployment` to rule-check it. With
`Rules` set, a Result and no `Scenario`, the rule layer warns (`result-input-rules`).

Source: [`spec/result`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/result).
