# Workload Specifications

This guide covers how to define the traffic patterns BLIS simulates — from simple CLI flags to complex multi-client YAML workload specs.

```bash
# Quick example: workload-spec YAML
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --workload-spec examples/multiturn-chat-demo.yaml
```

## Workload Modes

BLIS supports four modes, in order of precedence:

| Mode | Flag | Best For |
|------|------|----------|
| **Workload-spec YAML** | `--workload-spec <path>` | Multi-client workloads with custom distributions |
| **CLI distribution** | `--rate`, `--num-requests`, `--prompt-tokens` | Quick single-client experiments |
| **Named presets** | `--workload chatbot` | Standard workload profiles |
| **CSV traces** | `--workload traces` | Replaying recorded production traffic |

## CLI Distribution Mode (Default)

The simplest way to generate traffic:

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct \
  --rate 100 --num-requests 500 \
  --prompt-tokens 512 --prompt-tokens-stdev 256 \
  --output-tokens 256 --output-tokens-stdev 128
```

## Writing a Workload-Spec YAML

For complex workloads, use a YAML spec:

```yaml
version: "2"
seed: 42
aggregate_rate: 100       # Total arrival rate (req/s)
num_requests: 1000

clients:
  - id: "interactive"
    rate_fraction: 0.6    # 60% of traffic
    slo_class: "critical"
    prefix_group: "chat"
    prefix_length: 512
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 128
        min: 2
        max: 4096
    output_distribution:
      type: exponential
      params:
        mean: 128

  - id: "batch"
    rate_fraction: 0.4
    slo_class: "batch"
    arrival:
      process: gamma
      cv: 2.0
    input_distribution:
      type: gaussian
      params:
        mean: 1024
        std_dev: 512
        min: 2
        max: 7000
    output_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 256
        min: 2
        max: 7000
```

## Arrival Processes

| Process | Behavior | DES Impact | Use When |
|---------|----------|-----------|----------|
| `poisson` | Memoryless, uniform inter-arrival times | Steady event stream | Default; matches typical web traffic |
| `gamma` | Bursty (CV > 1) or regular (CV < 1) inter-arrivals | Burst events create temporary overloads | Modeling real traffic with bursts |
| `weibull` | Shape-controlled inter-arrival times | Similar to gamma, different tail behavior | Specific traffic shape matching |
| `constant` | Fixed inter-arrival time (deterministic) | Perfectly regular event stream | Controlled experiments, debugging |

!!! info "DES implication"
    Arrival processes directly determine the timing of `ArrivalEvent` injections into the event queue. Gamma CV=3.5 produces 1.66x worse TTFT p99 at sub-saturation because burst events arrive before the prior burst drains.

## Token Distributions

| Type | Parameters | Behavior |
|------|-----------|----------|
| `gaussian` | `mean`, `std_dev`, `min`, `max` | Normal distribution, clamped to range |
| `exponential` | `mean` | Right-skewed, long tail |
| `pareto_lognormal` | `mean`, `sigma`, `alpha` | Heavy-tailed (models real production traffic) |
| `constant` | `value` | Fixed token count (useful for controlled experiments) |
| `empirical` | `file` | Sample from a recorded distribution |

## SLO Classes

Requests can be tagged with SLO classes for per-class metric tracking:

| Class | Intended Use |
|-------|-------------|
| `critical` | Latency-sensitive user-facing requests |
| `standard` | Normal priority |
| `sheddable` | Can be dropped under load |
| `batch` | Offline processing, latency-tolerant |
| `background` | Lowest priority |

## Estimating Capacity for Your Workload

!!! warning "CLI mode and YAML mode have different defaults"
    CLI mode uses `--prompt-tokens 512, --output-tokens 512` by default (step time ~17.4ms, capacity ~57 req/s per instance). YAML workloads define their own distributions — a YAML with mean=256/128 has step time ~11.8ms, capacity ~85 req/s. Don't reuse capacity estimates across modes.

## Built-in Presets and Examples

BLIS ships with example workload specs in `examples/`:

| File | Description |
|------|-------------|
| `multiturn-chat-demo.yaml` | Multi-turn chat with prefix-affinity routing |
| `prefix-affinity-demo.yaml` | Shared-prefix workload for cache testing |
| `servegen-language.yaml` | ServeGen-derived language workload |
| `inference-perf-shared-prefix.yaml` | inference-perf format compatibility |

## Further Reading

- [Workload Spec Schema](../reference/workload-spec.md) — complete field reference
- [Configuration Reference](../reference/configuration.md#workload-configuration) — all workload flags
