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
| `servegen_data` | object | No | Native ServeGen data file loading |
| `inference_perf` | object | No | inference-perf format compatibility |

*At least one `client` or `servegen_data` is required.

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
| `pareto_lognormal` | `mean`, `sigma`, `alpha` |
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
