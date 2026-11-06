# Trace

<p class="lede">A request stream on disk: a YAML header and a CSV of one row per request.
<code>blis run --trace-output</code> and <code>blis replay</code> write one, <code>blis
convert</code> writes one from another tool's capture, and <code>blis replay</code> reads
one. A scenario binds a trace by path, with its header.</p>

`trace.ReadHeader` and `trace.ReadRecords` read the two files strictly; `trace.Read` reads
both and validates them together. `trace.WriteHeader` and `trace.WriteRecords` write them,
and return the writer's error.

```yaml title="trace header"
--8<-- "docs/contracts/examples/trace-header.yaml"
```

```text title="trace data"
--8<-- "docs/contracts/examples/trace-data.csv"
```

Version 4 is the format inference-sim wrote as version 3, without what only the removed
`blis observe` command wrote: the `real` mode; the header keys `created_at`,
`warm_up_requests`, `workload_spec`, `server`, `network` and `observed_kv_metrics`; and the
columns `vllm_priority`, `x_request_id`, `server_input_tokens`, `send_time_us`,
`num_chunks`, `error_message` and `finish_reason`. It adds the mode `converted`.

## Header

<!-- fields: trace.Header -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `trace_version` | integer | yes | Always `4`. |
| `time_unit` | string | yes | Unit of the data file's timestamp columns. One of the [time units](vocabularies.md#time-units). |
| `mode` | string | yes | `generated` (`blis run`), `replayed` (`blis replay`) or `converted` (`blis convert`, from another tool's capture). |
| `workload_seed` | integer | no | The workload generator's seed. A seed of 0 is a real seed, distinct from absence. |
| `session_context_growth` | string | no | `accumulate`: each session round's input is the previous round's input and output, then its new tokens. Omitted, rounds are drawn independently. |
| `goodput_slo_targets` | mapping | no | Latency targets per SLO class, keyed by class name, below. Class names are free. |
| `kv_offload` | mapping | no | The resolved KV-cache offload configuration of the run that wrote the trace, below. |
| `max_nodes_spanned` | integer | no | The most nodes one instance occupied, when more than one. A replay refuses a trace it cannot reproduce. |
| `router` | mapping | no | The llm-d router inputs the run routed with, below. |

A `generated` or `replayed` trace records a BLIS deployment, and a replay keeps its header
authoritative: it does not add offload tiers or other deployment options the run did not
have. A `converted` trace records none, so a replay takes them from the scenario and the
command line (`Header.RecordsDeployment`, inference-sim #2067).

### `goodput_slo_targets.<class>`

Thresholds in milliseconds. Zero on a dimension leaves it unconstrained, but a class must
constrain at least one.

<!-- fields: trace.SLODimTargets -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `ttft_ms` | number | no | Time to first token. Not negative. |
| `itl_ms` | number | no | Time per output token. Not negative. |
| `e2e_ms` | number | no | Request latency. Not negative. |

### `kv_offload`

The resolved numbers, so a replay reproduces the run whatever its host's defaults are.

<!-- fields: trace.KVOffload -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `cpu_bytes_to_use` | integer | yes | Bytes of host memory for the CPU tier. |
| `per_block_bytes` | integer | yes | Bytes per KV block. |
| `block_size` | integer | yes | Tokens per block. |
| `blocks_per_chunk` | integer | yes | Blocks per transfer chunk. |
| `tokens_per_hash` | integer | yes | Tokens per prefix hash. |
| `eviction_policy` | string | yes | The CPU tier's eviction policy. |
| `offload_prompt_only` | boolean | yes | Whether only prompt blocks are offloaded. |
| `self_describing_kv_events` | boolean | yes | Whether KV events name their medium. |
| `secondary_tiers` | list | no | Tiers below the CPU tier, below. |
| `simple_cpu_offload` | boolean | no | vLLM's SimpleCPUOffloadConnector. |
| `lazy_offload` | boolean | no | Its lazy mode. |

<!-- fields: trace.KVOffloadTier -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `type` | string | yes | The tier's type. |
| `root_dir` | string | yes | Its directory. |
| `n_read_threads` | integer | yes | Read threads. |
| `n_write_threads` | integer | yes | Write threads. |
| `locality` | string | no | Where the tier lives relative to the node. |
| `enable_kv_events` | boolean | yes | Whether it emits KV events. |
| `direct_io` | boolean | yes | Whether it uses direct I/O. |
| `device_class` | string | no | A label; the numbers below decide. |
| `read_bandwidth` | number | yes | Bytes per microsecond. |
| `write_bandwidth` | number | yes | Bytes per microsecond. |
| `base_latency` | number | yes | Microseconds. |
| `saturation_queue_depth` | integer | yes | Queue depth at saturation; 1 turns the ramp off. |
| `single_transfer_fraction` | number | yes | Fraction of bandwidth one transfer reaches; 1 turns the ramp off. |
| `capacity_bytes` | integer | no | Size; 0 is unbounded. |
| `latency_jitter_stddev` | number | yes | Standard deviation of latency jitter. |

### `router`

<!-- fields: trace.Router -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `epp_flags` | list of strings | yes | The router's flags, with its EndpointPickerConfig inlined as `--config-text`. |
| `manifests` | string | no | The Kubernetes manifests, as one YAML stream. |
| `pod_labels` | mapping | no | Extra labels on the engines' pods. |
| `gateway_latency_us` | integer | no | Gateway latency. |
| `scrape_jitter_us` | integer | no | Metrics-scrape jitter. |
| `kv_event_delay_us` | integer | no | KV-event delivery delay. |
| `seed` | integer | no | The router's seed. |

## Data

The first row names the columns; a reader finds them by name. The columns of `Columns` are
required; those of `OptionalColumns` (`slo_target_us`, `adapter`, `think_time_us`,
`input_tokens_reset`) are written only when some row carries one. Any other column is
refused.

<!-- fields: trace.Record -->

| Key | Type | Required | Meaning |
|---|---|---|---|
| `request_id` | integer | yes | Unique in the trace. |
| `client_id` | string | yes | The client that sent it; may be empty. |
| `tenant_id` | string | yes | The tenant; may be empty. |
| `slo_class` | string | yes | The SLO class; may be empty. |
| `session_id` | string | yes | The session; may be empty. |
| `round_index` | integer | yes | The round within the session, 0 first; each once per session. |
| `prefix_group` | string | yes | The shared prefix's group; may be empty. |
| `prefix_length` | integer | yes | Tokens of the group's prefix the prompt starts with. |
| `streaming` | boolean | yes | `true` or `false`. |
| `input_tokens` | integer | yes | Prompt tokens after the prefix; in an accumulating session, the round's new tokens. |
| `output_tokens` | integer | yes | Output tokens. |
| `text_tokens` | integer | yes | Text tokens of a multimodal prompt. |
| `image_tokens` | integer | yes | Image tokens. |
| `audio_tokens` | integer | yes | Audio tokens. |
| `video_tokens` | integer | yes | Video tokens. |
| `reason_ratio` | number | yes | The reasoning share of the output, in [0, 1]. |
| `model` | string | yes | The model tag; empty for the scenario's model. |
| `deadline_us` | integer | yes | Absolute deadline on the arrival origin; 0 is none; not before `arrival_time_us`. |
| `arrival_time_us` | integer | yes | Arrival. |
| `first_chunk_time_us` | integer | yes | First token delivered in the run that wrote the trace; 0 for none. |
| `last_chunk_time_us` | integer | yes | Last token delivered; not before the first. |
| `status` | string | yes | One of the [trace statuses](vocabularies.md#trace-statuses). |
| `slo_target_us` | integer | no | The request's TTFT target; 0 is none. |
| `adapter` | string | no | The LoRA adapter; empty for the base model. |
| `think_time_us` | integer | no | Recorded client think time before the round; an empty cell is not recorded, distinct from 0. |
| `input_tokens_reset` | integer | no | The recorded absolute input of a round whose context was compacted; empty on other rounds. |

## Checks

A header is invalid if `trace_version` is not 4; `time_unit` or `mode` is empty or not in
its vocabulary (`real` is refused with a pointer to `converted`); `session_context_growth`
is other than `accumulate`; an SLO class name is empty, a target negative or not finite, or a
class sets no target; an offload number is negative or not finite; `max_nodes_spanned` is
negative or 1; `router` has no flags; or a `converted` trace states `kv_offload`,
`max_nodes_spanned` or `router`.

A record is invalid if a count or a time is negative, `reason_ratio` lies outside [0, 1],
`deadline_us` precedes a non-zero `arrival_time_us`, the last chunk precedes the first, or
`status` is unknown. A trace is invalid if a `request_id` repeats, or a session repeats a
round.

Source: [`spec/trace`](https://github.com/inference-sim/blis-substrate/tree/main/contracts/spec/trace).
