# Workload

<p class="lede">Traffic comes at two fidelities. A <strong>shape</strong> describes it as
distributions and lives in the catalog. A <strong>trace</strong> is a captured file of real
requests. A <strong>replay</strong> names a load generator's trace-replay scenario over a
pinned corpus. A scenario's <code>workload</code> key binds exactly one of the three.</p>

## Binding

The value of a scenario's `workload` key.

```yaml
workload:
  shape: chatbot
```

<!-- fields: workload.Binding -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `shape` | string | one of the three | A workload shape under `workloads/` in the catalog. |
| `trace` | mapping | one of the three | A reference to a captured trace, below. |
| `replay` | mapping | one of the three | A trace-replay scenario, below. |

A binding with more than one key, or with none, is invalid. To pose a question with no traffic,
omit `workload` from the scenario instead.

## Trace

```yaml
--8<-- "docs/contracts/examples/trace-binding.yaml:workload"
```

The rows of a trace can number in the millions, so they stay in the file that `data`
names and never appear in the scenario. The scenario holds the path, an optional digest
and row count, and the small header that says how to read the file.

<!-- fields: workload.TraceRef -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `data` | string | yes | Path to the per-request data CSV. |
| `sha256` | string | no | Hex digest of the data file, 64 characters. |
| `rows` | integer | no | Expected number of rows in the data file. Not negative. |
| `header` | mapping | yes | The trace header, below. |
| `token_blocks` | mapping | no | The per-request block-hash sidecar, below. Lets a replay reproduce prefix-cache hits exactly rather than approximate them from prefix groups. |
| `category` | string | no | The workload category that selects an [acceptance profile](acceptance.md), such as `coding`. |
| `thinking` | string | no | Whether the model reasons: `on` or `off`. Selects the profile with `category`. |
| `acceptance` | mapping | no | The trace's own measured acceptance, an [inline value](acceptance.md#inline-values). Outranks the registry's. |
| `routing` | mapping | no | The traffic's own measured routing imbalance, an [inline value](routing.md#inline-values). Outranks the registry's. |

### `token_blocks`

<!-- fields: workload.TokenBlocks -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `path` | string | yes | Path to the block-hash sidecar. |
| `sha256` | string | yes | Its digest, 64 lowercase hex characters. |

### `header`

The trace's own [header](trace.md#header), copied from the trace file. A scenario that binds
a trace states the header so the scenario says how to read the file without opening it.

## Replay

A replay states the run a load generator makes, not the requests it sends: a named
scenario, the harness release it reproduces, the corpus it reads, and every knob the
scenario fixes. The corpus stays in its repository. How sessions are rebuilt and
dispatched is the harness's behaviour, which a simulator implements; this is the contract
it is checked against. Today one scenario is transcribed: aiperf 0.12.0's
`inferencex-agentx-mvp`, agentic replay of SemiAnalysis's Weka Claude Code traces.

```yaml
--8<-- "docs/contracts/examples/replay-binding.yaml:workload"
```

A replay arm takes one of two forms. Inline, it states the whole definition beside the
lane count, as above. As a reference, it names a catalog replay workload and states only
the run's choices:

```yaml
--8<-- "docs/contracts/examples/replay-reference.yaml:workload"
```

<!-- fields: workload.Replay -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `workload` | string | no | A catalog replay workload under `workloads/`, by file name. Set, the binding may state only `concurrency`, `random_seed`, `benchmark_duration_s`, `grace_period_s`, `max_context_length` and `acceptance`, which override the entry's; `ValidateAgainstCatalog` loads the entry and `ResolveReplay` merges the two. |
| `concurrency` | integer | yes | Session-tree lanes, at least 1. Subagents run beside their parent without taking a lane. One value: a scenario run is not a sweep. It is the run's choice, so a catalog entry does not state it. |

The other keys are the definition's, below.

### The definition

A key the scenario locks may be omitted, and then takes the value the harness applies; a
stated value the scenario's validator would refuse is an error. Citations are to aiperf
0.12.0, `common/scenario/inferencex_agentx_mvp.py` and `common/scenario/validator.py`.

<!-- fields: workload.ReplayDefinition -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `scenario` | string | yes | One of the [replay scenarios](vocabularies.md#replay-scenarios). |
| `harness_version` | string | yes | The harness release the replay reproduces. `0.12.0` for `inferencex-agentx-mvp`. |
| `corpus` | mapping | yes | The traces, below. |
| `random_seed` | integer | no | Seeds lane snapshots and prompt content. Omitted, 42. (aiperf alone would draw a random seed.) |
| `benchmark_duration_s` | number | no | The sending window, at least 900. Omitted, 1800. |
| `grace_period_s` | number | no | How long requests in flight at the window's end may finish. Not negative; `.inf` waits indefinitely. Omitted, 30. |
| `max_context_length` | integer or `max_model_len` | no | Drops every trace whose recorded peak prompt plus output exceeds it. `max_model_len` is the deployment's resolved context length. Omitted, no filter. |
| `cache_bust` | string | no | Must be `first_turn_prefix`, its default: one marker per session tree. |
| `system_idle_gap_cap_s` | number | no | Must be 10, its default: when nothing is in flight, pending timers are pulled in so the system idles at most this long. |
| `trajectory_start` | mapping | no | The window of each trace's span a lane's snapshot is drawn from, below. Omitted, the whole span. |
| `warmup` | mapping | no | How the server is primed before profiling, below. Omitted, `synthesized`. |
| `split_flattened_agents` | boolean | no | Whether agent chains a capture flattened are split by hash-prefix detection. Omitted, true. |
| `streaming` | boolean | no | Must be true. |
| `ignore_eos` | boolean | no | Must be true: every request generates its recorded output length. |
| `ignore_trace_delays` | boolean | no | Must be false. |
| `synthesis_max_isl` | integer | no | Must be unset: the scenario forbids input truncation. |
| `trace_idle_gap_cap_s` | number | no | Positive, or unset. A per-trajectory idle watchdog (the InferenceX fork's `--trace-idle-gap-cap-seconds`); the scenario's default leaves it off. |
| `inter_turn_delay_cap_s` | number | no | Must be unset: per-turn delays are uncapped. |
| `category` | string | no | The workload category that selects an [acceptance profile](acceptance.md). |
| `thinking` | string | no | `on` or `off`. Selects the profile with `category`. |
| `acceptance` | mapping | no | The run's own acceptance, an [inline value](acceptance.md#inline-values). |
| `routing` | mapping | no | The traffic's own measured routing imbalance, an [inline value](routing.md#inline-values). Outranks the registry's. |

### Catalog replay workload

A catalog file under `workloads/` whose top-level key is `replay:` is a replay workload
rather than a shape. It holds a definition, with no lane count, and is named by its file.
It may carry `_comment` keys at its top level, as a shape may. `LoadCatalogWorkload` reads
either form, `LoadWorkload` reads shapes only, and `validate-catalog` validates both.

```yaml title="workloads/workload-replay.yaml"
--8<-- "docs/contracts/examples/workload-replay.yaml"
```

A scenario cannot bind a replay workload as `shape:`; `ValidateAgainstCatalog` says so.

### `corpus`

<!-- fields: workload.Corpus -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `loader` | string | yes | The harness loader or public-dataset alias, such as `semianalysis_cc_traces_weka_062126_256k`. One the scenario accepts. Each alias reads the repository aiperf's `plugins.yaml` fixes for it, and `source.huggingface` must name that one. The two rolling aliases, `semianalysis_cc_traces_weka_with_subagents` and its `_256k` variant, track whatever corpus is current and draw a warning. `weka_trace`, a local directory, cannot pin the corpus and is refused, and `weka_hf` may only read `semianalysisai/cc-traces-weka-062126`. |
| `format` | string | yes | One of the [corpus formats](vocabularies.md#corpus-formats). |
| `encoding` | string | yes | One of the [corpus encodings](vocabularies.md#corpus-encodings). |
| `source` | mapping | yes | Where it is published, below. |
| `sha256` | string | yes | Hex digest of the corpus file, 64 characters: its Hugging Face LFS object id. Required because aiperf reads the default branch, not `revision`, so only the digest ties a run to the bytes it read. |

### `corpus.source`

<!-- fields: workload.CorpusSource -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `huggingface` | string | yes | The dataset repository, `owner/name`. |
| `revision` | string | yes | The commit the corpus was observed at. A record, not a pin: aiperf 0.12.0's Weka loaders set no `hf_revision` and read the default branch (`dataset/loader/base_hf_dataset.py:28, 77`). |
| `file` | string | no | The corpus file's path in the repository. |
| `split` | string | no | The dataset split. |

### `trajectory_start`

Each bound is a fraction of a trace's span, from its first request's start to its last's,
in [0, 1]. The scenario fills [0, 1]; unlike its locks, aiperf honours a stated window.

<!-- fields: workload.TrajectoryStart -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `min_ratio` | number | no | Lower bound. Omitted, 0. |
| `max_ratio` | number | no | Upper bound, at least `min_ratio`. Omitted, 1. |

### `warmup`

<!-- fields: workload.Warmup -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `mode` | string | yes | One of the [warmup modes](vocabularies.md#warmup-modes). |
| `duration_s` | number | `cache_pressure`, with one of these | How long the accelerated cache-pressure warmup runs. No other mode takes it. |
| `requests_per_lane` | integer | `cache_pressure`, with one of these | Request-bounded cache-pressure warmup: each lane sends this many extra one-token warmup requests. Mutually exclusive with `duration_s`; no other mode takes it. |

## Shape

A catalog file under `workloads/`, named by its file; a `name` key in it is rejected. This is
the catalog's `chatbot`:

```yaml title="workloads/chatbot.yaml"
--8<-- "catalog/workloads/chatbot.yaml"
```

<!-- fields: workload.Shape -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `prefix_tokens` | integer | no | Prefix length shared across requests. Zero means none. |
| `prompt` | mapping | yes | Prompt length distribution, below. |
| `output` | mapping | yes | Output length distribution, below. |
| `category` | string | no | The workload category that selects an [acceptance profile](acceptance.md), such as `coding`. |
| `thinking` | string | no | Whether the model reasons: `on` or `off`. Selects the profile with `category`. |
| `acceptance` | mapping | no | The traffic's own measured acceptance, an [inline value](acceptance.md#inline-values). Outranks the registry's. |
| `routing` | mapping | no | The traffic's own measured routing imbalance, an [inline value](routing.md#inline-values). Outranks the registry's. |

### `prompt` and `output`

<!-- fields: workload.Distribution -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `tokens` | integer | yes | Mean length in tokens. Positive. |
| `tokens_stdev` | integer | no | Standard deviation. |
| `tokens_min` | integer | no | Lower bound. Zero means unset. |
| `tokens_max` | integer | no | Upper bound. Zero means unset. |

A file may also carry `_comment` and `_comment_*` keys, at the top level and inside
`prompt` and `output`. They are discarded when the file is loaded.

## Checks

A binding is invalid unless it sets exactly one of `shape`, `trace` and `replay`. A trace reference is
invalid if

- `data` is empty, `rows` is negative, or `sha256` is present and is not 64 hex digits;
- its header is not a valid [trace header](trace.md#checks).

A trace reference or a shape is also invalid if a stated `category` is malformed,
`thinking` is not `on` or `off`, or `acceptance` is not a valid
[inline value](acceptance.md#inline-values).

A replay is invalid if

- `scenario` or `harness_version` is empty or not transcribed, or `concurrency` is less
  than 1;
- `benchmark_duration_s` is not finite, negative, or below 900;
- `grace_period_s` is negative or NaN;
- `cache_bust` is other than `first_turn_prefix`, or `system_idle_gap_cap_s` other than 10;
- a `trajectory_start` bound lies outside [0, 1], or `min_ratio` exceeds `max_ratio`;
- `warmup.mode` is unknown; a `cache_pressure` warmup does not state exactly one of
  `duration_s` or `requests_per_lane`, or states a non-positive one; or either is stated for
  another mode;
- `streaming` or `ignore_eos` is false, `ignore_trace_delays` true, `synthesis_max_isl` or
  `inter_turn_delay_cap_s` is set, or `trace_idle_gap_cap_s` is non-positive;
- the corpus loader is empty or not one the scenario accepts, is `weka_trace`, or is
  `weka_hf` on another repository, or an alias on a repository other than its own; `format` or `encoding` is unknown; the repository is not
  `owner/name`; `revision` is empty; or `sha256` is missing or not 64 hex digits;
- its acceptance terms are invalid, as for a trace.

A replay that references a catalog workload is invalid if it states anything but the run's
choices, or `concurrency` is less than 1. `ValidateAgainstCatalog` reports a reference
whose entry is missing, is a shape, or merges to an invalid replay.

A shape is invalid if

- `prefix_tokens` is negative, or longer than the mean prompt;
- a mean `tokens` is less than 1;
- `tokens_stdev` or `tokens_min` is negative;
- `tokens_min` exceeds `tokens_max`, or the mean lies outside [`tokens_min`, `tokens_max`]
  where those are set.

A shape built in Go must also have a name; one loaded from a file always does, since the
loader takes it from the file name.

Source: [`spec/workload`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/workload).
