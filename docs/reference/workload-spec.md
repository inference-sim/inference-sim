# Workload Spec Schema

Complete YAML schema reference for BLIS workload specifications (`--workload-spec`). For a guide-level introduction, see [Workload Specifications](../guide/workloads.md).

## Top-Level Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | No | Schema version (`"2"` recommended; `"1"` auto-upgraded) |
| `seed` | int64 | No | RNG seed (overridden by CLI `--seed` if set) |
| `category` | string | No | `language`, `multimodal`, `reasoning`, or empty |
| `aggregate_rate` | float64 | **Yes** | Total arrival rate in requests/second |
| `num_requests` | int64 | No | Total requests to generate (0 = unlimited, use horizon) |
| `horizon` | int64 | No | Simulation time limit in ticks (overridden by CLI `--horizon` if set) |
| `clients` | list | **Yes*** | Client specifications (see below) |
| `cohorts` | list | No | Cohort specifications with population dynamics (diurnal, spike, drain patterns) |
| `servegen_data` | object | No | Native ServeGen data file loading |
| `inference_perf` | object | No | inference-perf format compatibility |

*At least one `client`, `cohort`, or `servegen_data` is required.

## Client Specification

Each entry in the `clients` list defines a traffic source:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Client identifier (for metrics grouping) |
| `tenant_id` | string | No | Tenant identifier |
| `slo_class` | string | No | SLO tier: `critical`, `standard`, `sheddable`, `batch`, `background`, or empty |
| `model` | string | No | Model name override (for multi-model workloads) |
| `rate_fraction` | float64 | **Yes** | Fraction of `aggregate_rate` for this client (must be positive) |
| `arrival` | object | **Yes** | Arrival process configuration |
| `input_distribution` | object | **Yes** | Input token length distribution |
| `output_distribution` | object | **Yes** | Output token length distribution |
| `prefix_group` | string | No | Prefix group name (requests in same group share prefixes) |
| `prefix_length` | int | No | Shared prefix token count (additive to input_distribution) |
| `streaming` | bool | No | Whether to simulate streaming output |
| `network` | object | No | Client-side network characteristics |
| `lifecycle` | object | No | Activity window configuration |
| `multimodal` | object | No | Multimodal token generation |
| `reasoning` | object | No | Reasoning multi-turn behavior |

## Arrival Process

| Field | Type | Values | Description |
|-------|------|--------|-------------|
| `process` | string | `poisson`, `gamma`, `weibull`, `constant` | Inter-arrival time distribution |
| `cv` | *float64 | Required for `gamma` and `weibull` | Coefficient of variation (burstiness). CV > 1 = bursty, CV < 1 = regular |

## Distribution Specification

Used for `input_distribution` and `output_distribution`:

| Field | Type | Description |
|-------|------|-------------|
| `type` | string | `gaussian`, `exponential`, `pareto_lognormal`, `constant`, `empirical` |
| `params` | map | Type-specific parameters (see below) |
| `file` | string | Reserved for future use (file-based loading not yet implemented). Use inline `params` instead. |

### Distribution Parameters

| Type | Parameters |
|------|-----------|
| `gaussian` | `mean`, `std_dev`, `min`, `max` |
| `exponential` | `mean` |
| `pareto_lognormal` | `alpha`, `xm`, `mu`, `sigma`, `mix_weight` |
| `constant` | `value` |
| `empirical` | inline `params` map (key=token count, value=probability) |

## Network Specification

| Field | Type | Description |
|-------|------|-------------|
| `rtt_ms` | float64 | Round-trip time in milliseconds |
| `bandwidth_mbps` | float64 | Bandwidth in Mbps |

## Reasoning Specification

| Field | Type | Description |
|-------|------|-------------|
| `reason_ratio_distribution` | DistSpec | Distribution of reasoning-to-output ratio |
| `multi_turn` | object | Multi-turn conversation configuration |
| `multi_turn.max_rounds` | int | Maximum conversation rounds |
| `multi_turn.think_time_us` | int64 | User think time between rounds (microseconds) |
| `multi_turn.context_growth` | string | `accumulate` (prepend prior context) |

## Cohort Specification

Each entry in the `cohorts` list defines a population with lifecycle dynamics. Cohorts expand into individual clients with lifecycle windows derived from diurnal, spike, or drain patterns.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | No | Cohort identifier |
| `population` | int | **Yes** | Number of clients in this cohort (max 100,000) |
| `tenant_id` | string | No | Tenant identifier |
| `slo_class` | string | No | SLO tier: `critical`, `standard`, `sheddable`, `batch`, `background` |
| `model` | string | No | Model name override |
| `arrival` | object | **Yes** | Arrival process configuration (same as Client) |
| `input_distribution` | object | **Yes** | Input token length distribution |
| `output_distribution` | object | **Yes** | Output token length distribution |
| `prefix_group` | string | No | Prefix group name |
| `streaming` | bool | No | Whether to simulate streaming output |
| `rate_fraction` | float64 | **Yes** | Fraction of `aggregate_rate` for each client in this cohort |
| `diurnal` | object | No | Sinusoidal rate modulation (see below) |
| `spike` | object | No | Traffic spike configuration (see below) |
| `drain` | object | No | Linear ramp-down to zero (see below) |

### Diurnal Pattern

| Field | Type | Description |
|-------|------|-------------|
| `peak_hour` | int | Hour of peak traffic (0-23) |
| `peak_to_trough_ratio` | float64 | Ratio of peak to trough rate (≥ 1.0) |

### Spike Pattern

| Field | Type | Description |
|-------|------|-------------|
| `start_time_us` | int64 | Spike start time in microseconds |
| `duration_us` | int64 | Spike duration in microseconds |

### Drain Pattern

| Field | Type | Description |
|-------|------|-------------|
| `start_time_us` | int64 | Drain start time in microseconds |
| `ramp_duration_us` | int64 | Ramp-down duration in microseconds |

## Lifecycle Specification

Activity window configuration for clients (used in the `lifecycle` field of Client Specification). Cohort patterns (diurnal, spike, drain) are converted into lifecycle windows internally.

| Field | Type | Description |
|-------|------|-------------|
| `windows` | list | List of active time windows |

### Active Window

| Field | Type | Description |
|-------|------|-------------|
| `start_us` | int64 | Window start time in microseconds |
| `end_us` | int64 | Window end time in microseconds |

## Multimodal Specification

Configures multimodal request generation (used in the `multimodal` field of Client Specification). Each distribution follows the same [Distribution Specification](#distribution-specification) format.

| Field | Type | Description |
|-------|------|-------------|
| `text_distribution` | DistSpec | Text token distribution |
| `image_distribution` | DistSpec | Image token distribution |
| `image_count_distribution` | DistSpec | Number of images per request |
| `audio_distribution` | DistSpec | Audio token distribution |
| `audio_count_distribution` | DistSpec | Number of audio segments per request |
| `video_distribution` | DistSpec | Video token distribution |
| `video_count_distribution` | DistSpec | Number of video segments per request |

## ServeGen Data Specification

Native ServeGen data file loading (used in the `servegen_data` top-level field):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `path` | string | **Yes** | Path to ServeGen data directory (containing `chunk-*-trace.csv` and `dataset.json`) |
| `span_start` | int64 | No | Trace span start filter (microseconds) |
| `span_end` | int64 | No | Trace span end filter (microseconds) |

## InferencePerf Specification

inference-perf format compatibility (used in the `inference_perf` top-level field):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `stages` | list | **Yes** | Rate/duration stages for load patterns |
| `shared_prefix` | object | **Yes** | Shared prefix expansion configuration |

### Stage

| Field | Type | Description |
|-------|------|-------------|
| `rate` | float64 | Requests per second for this stage |
| `duration` | int64 | Stage duration in seconds (note: unlike other time fields which use microseconds, this field uses seconds) |

### Shared Prefix

| Field | Type | Description |
|-------|------|-------------|
| `num_unique_system_prompts` | int | Number of unique system prompts |
| `num_users_per_system_prompt` | int | Users per system prompt |
| `system_prompt_len` | int | System prompt length in tokens |
| `question_len` | int | Question length in tokens |
| `output_len` | int | Output length in tokens |
| `enable_multi_turn_chat` | bool | Enable multi-turn chat mode |

## Complete Example

```yaml
version: "2"
seed: 42
category: reasoning
aggregate_rate: 500.0
num_requests: 500

clients:
  - id: "multi-turn-chat"
    tenant_id: "chat-users"
    slo_class: "standard"
    rate_fraction: 1.0
    streaming: true
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128
        std_dev: 30
        min: 32
        max: 512
    output_distribution:
      type: gaussian
      params:
        mean: 64
        std_dev: 20
        min: 16
        max: 256
    reasoning:
      reason_ratio_distribution:
        type: gaussian
        params:
          mean: 0
          std_dev: 0
          min: 0
          max: 0
      multi_turn:
        max_rounds: 5
        think_time_us: 500000
        context_growth: accumulate
```

## Validation

BLIS validates workload specs with strict YAML parsing (`KnownFields(true)`) — typos in field names cause errors. Additional validation:

- `aggregate_rate` must be positive
- Each client's `rate_fraction` must be positive
- `arrival.process` must be one of the valid processes
- `cv` for gamma/weibull must be finite and positive
- Weibull `cv` must be in [0.01, 10.4]
- Distribution types must be recognized
- All numeric params must be finite (no NaN or Inf)
- At least one `client`, `cohort`, or `servegen_data` is required
- Cohort `population` must be positive and ≤ 100,000
