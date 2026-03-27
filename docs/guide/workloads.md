# Workload Specifications

This guide covers how to define the traffic patterns BLIS simulates — from simple CLI flags to complex multi-client YAML workload specs.

```bash
# Quick example: workload-spec YAML
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --workload-spec examples/multiturn-chat-demo.yaml
```

## Workload Modes

BLIS supports three modes, in order of precedence:

| Mode | Flag | Best For |
|------|------|----------|
| **Workload-spec YAML** | `--workload-spec <path>` | Multi-client workloads with custom distributions |
| **CLI distribution** | `--rate`, `--num-requests`, `--prompt-tokens` | Quick single-client experiments |
| **Named presets** | `--workload chatbot` | Standard workload profiles |

!!! note "Migration: `--workload traces` removed"
    The `--workload traces` and `--workload-traces-filepath` flags have been removed. They performed lossy statistical approximation (averaged token lengths, constant arrival) rather than faithful replay. For trace replay, use `--workload-spec` with a [TraceV2 YAML file](../reference/workload-spec.md) instead. To export simulation results as TraceV2, use `--trace-output <prefix>`.

## Modeling Real Workloads

This section maps common traffic patterns to YAML workload spec configurations. For schema details, see the [Workload Spec Schema](../reference/workload-spec.md).

### Interactive Chat

User-facing chat applications need low latency, memoryless arrivals (users arrive independently), and moderate token variance around a central prompt length.

```yaml
clients:
  - id: "chat-user"
    rate_fraction: 1.0
    slo_class: "critical"           # Latency-sensitive — tracked separately in metrics
    prefix_group: "system-prompt"   # Shared system prompt enables prefix caching
    prefix_length: 512              # 512 tokens of shared context prepended to each request
    arrival:
      process: poisson              # Memoryless: users arrive independently of each other
    input_distribution:
      type: gaussian                # Moderate variance around a typical prompt length
      params:
        mean: 256
        std_dev: 128
        min: 2
        max: 4096
    output_distribution:
      type: exponential             # Most replies short, occasional long answers
      params:
        mean: 128
```

Pair with prefix-affinity routing for cache reuse:

```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --workload-spec chat.yaml \
  --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:2"
```

### RAG with Shared Prefixes

Retrieval-augmented generation workloads share a common document context across requests. The `prefix_group` and `prefix_length` fields model this shared context, and prefix-affinity routing ensures requests with the same prefix hit cached KV blocks on the same instance.

```yaml
clients:
  - id: "rag-query"
    rate_fraction: 1.0
    slo_class: "standard"
    prefix_group: "doc-context"     # All requests share the retrieved document context
    prefix_length: 2048             # Large shared prefix (retrieved passages)
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 128                   # Short user queries appended after the prefix
        std_dev: 64
        min: 2
        max: 512
    output_distribution:
      type: exponential             # Short answers mostly, occasional long explanations
      params:
        mean: 64
```

Run with aggressive prefix-affinity to maximize cache reuse:

```bash
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --workload-spec rag.yaml \
  --routing-policy weighted --routing-scorers "prefix-affinity:3,queue-depth:1"
```

### Batch / Offline Processing

Non-interactive workloads (summarization, data extraction) tolerate latency and typically have higher token counts. Use `batch` or `background` SLO classes so per-class metrics track them separately from latency-sensitive traffic.

```yaml
clients:
  - id: "batch-summarize"
    rate_fraction: 1.0
    slo_class: "batch"              # Latency-tolerant — won't pollute critical-class metrics
    arrival:
      process: gamma                # Bursty job queue patterns (jobs submitted in waves)
      cv: 2.0                       # CV > 1 produces bursts; CV = 1 is Poisson-equivalent
    input_distribution:
      type: gaussian
      params:
        mean: 4096                  # Long documents for summarization
        std_dev: 1000
        min: 100
        max: 8192
    output_distribution:
      type: gaussian
      params:
        mean: 512
        std_dev: 150
        min: 10
        max: 2048
```

### Bursty Traffic

For flash sales or traffic spikes, use Gamma arrivals with high CV or cohort spike patterns:

```yaml
# Option 1: Sustained burstiness via Gamma CV=3.5
clients:
  - id: "bursty-client"
    rate_fraction: 1.0
    slo_class: "critical"
    arrival:
      process: gamma
      cv: 3.5                       # High CV produces sustained burst clusters
    input_distribution:
      type: exponential
      params:
        mean: 512
    output_distribution:
      type: exponential
      params:
        mean: 256
```

For time-bounded traffic spikes, use cohort `spike` patterns instead (see [Cohort Dynamics](#cohort-dynamics) below).

!!! info "DES impact of burstiness"
    Gamma CV=3.5 produces 1.66x worse TTFT p99 at sub-saturation because burst events arrive before the prior burst drains. The effect is load-duration dependent: visible at moderate load, drowned by queue growth at high overload.

### Multi-Turn Conversations

For multi-round chat with context accumulation (e.g., reasoning models), use the `reasoning` field:

```yaml
clients:
  - id: "reasoning-session"
    rate_fraction: 1.0
    slo_class: "standard"
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 128
        min: 2
        max: 2048
    output_distribution:
      type: gaussian
      params:
        mean: 256
        std_dev: 128
        min: 2
        max: 2048
    reasoning:
      reason_ratio_distribution:    # Fraction of output that is "reasoning" tokens
        type: constant
        params:
          value: 50                 # 50% reasoning ratio
      multi_turn:
        max_rounds: 4              # Up to 4 conversation rounds per session
        think_time_us: 5000000     # 5 seconds between rounds (user think time)
        context_growth: accumulate # Each round prepends full prior context
```

Context accumulation means round N sees all prior input+output tokens as prefix, creating growing KV cache pressure across rounds.

#### Open-loop vs closed-loop scheduling

By default, multi-turn sessions use **closed-loop** scheduling: each round's arrival time is set by the simulator *after* the prior round completes — `next_arrival = completion_time + think_time_us`. This means actual server latency propagates into inter-round spacing; under high load the session stretches to reflect real queuing delays. This is the behaviorally correct default for capacity planning.

Setting `closed_loop: false` switches to **open-loop** scheduling: all round arrival times are pre-stamped before simulation begins, using a `1 µs/output-token` heuristic as the expected completion time. Server latency does not affect inter-round spacing — useful for reproducing pre-defined workload traces (e.g., inference-perf-style exports) where you want arrival times fixed regardless of how the simulator performs.

## CLI Distribution Mode (Default)

The simplest way to generate traffic:

```bash
./blis run --model qwen/qwen3-14b \
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
    rate_fraction: 0.6    # 60% of traffic — models a dominant chat workload
    slo_class: "critical" # Latency-sensitive: per-class metrics tracked separately
    prefix_group: "chat"  # Shared system prompt — enables prefix cache reuse
    prefix_length: 512    # 512-token system prompt prepended to each request
    arrival:
      process: poisson    # Memoryless: independent user arrivals (typical for web traffic)
    input_distribution:
      type: gaussian      # Moderate variance around a mean prompt length
      params:
        mean: 256
        std_dev: 128
        min: 2
        max: 4096
    output_distribution:
      type: exponential   # Right-skewed: most replies short, occasional long ones
      params:
        mean: 128

  - id: "batch"
    rate_fraction: 0.4    # 40% of traffic — background processing share
    slo_class: "batch"    # Latency-tolerant: won't pollute critical-class TTFT metrics
    arrival:
      process: gamma      # Bursty: jobs submitted in waves from a job queue
      cv: 2.0             # CV > 1 produces clustered arrivals (CV=1 ≈ Poisson)
    input_distribution:
      type: gaussian
      params:
        mean: 1024        # Longer inputs typical for summarization/extraction
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
| `poisson` | Memoryless, exponentially distributed inter-arrival times | Steady event stream | Default; matches typical web traffic |
| `gamma` | Bursty (CV > 1) or regular (CV < 1) inter-arrivals | Burst events create temporary overloads | Modeling real traffic with bursts |
| `weibull` | Shape-controlled inter-arrival times | Similar to gamma, different tail behavior | Specific traffic shape matching |
| `constant` | Fixed inter-arrival time (deterministic) | Perfectly regular event stream | Controlled experiments, debugging |

!!! info "DES implication"
    Arrival processes directly determine the timing of `ArrivalEvent` injections into the event queue. Gamma CV=3.5 produces 1.66x worse TTFT p99 at sub-saturation because burst events arrive before the prior burst drains.

#### What the arrival process governs for multi-turn clients

The unit controlled by the arrival process differs by client type:

| Client type | Arrival process governs |
|---|---|
| Non-multi-turn | Inter-arrival time between individual requests |
| Multi-turn, `single_session: false` (default) | Inter-arrival time between **session starts** — one new session begins per IAT tick, producing `max_rounds` rounds per session |
| Multi-turn, `single_session: true` | Fires exactly once per client, determining when that client's single session starts |

Within any session, inter-round spacing is always controlled by `think_time_us`, never by the arrival process.

#### ClientSpec is a traffic source, not a single user

A `ClientSpec` represents a **traffic source** that emits sessions over time, not a single user. One client with `single_session: false` and a long horizon generates many sequential sessions — it models a stream of independent users arriving over time. `single_session: true` changes the semantics to one persistent user who has exactly one conversation.

This distinction matters for modeling concurrent users correctly:

- **Wrong** for 50 concurrent users: one client with `rate_fraction: 1.0` and `single_session: false` — this produces sessions sequentially from a single source, not 50 simultaneous sessions.
- **Right** for 50 concurrent users: 50 clients (or a cohort with `population: 50`) each with `single_session: true` — each client draws one independent IAT and starts its own session. Per-client RNGs are seeded independently so start times are staggered even when all clients share the same arrival spec.

## Token Distributions

| Type | Parameters | Behavior |
|------|-----------|----------|
| `gaussian` | `mean`, `std_dev`, `min`, `max` | Normal distribution, clamped to range |
| `exponential` | `mean` | Right-skewed, long tail |
| `pareto_lognormal` | `alpha`, `xm`, `mu`, `sigma`, `mix_weight` | Heavy-tailed (Pareto-LogNormal mixture) |
| `constant` | `value` | Fixed token count (useful for controlled experiments) |
| `empirical` | `params` | Inline key-value map (token count → probability) |

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
    CLI mode uses `--prompt-tokens 512, --output-tokens 512` by default. With the default roofline latency model (Qwen3-14B / H100 / TP=1), a saturated single instance handles ~17 req/s. YAML workloads define their own distributions — a YAML with shorter sequences (e.g., mean=256/128) will have higher per-instance throughput. Don't reuse capacity estimates across modes or models.

!!! tip "Parameter resolution reference"
    For the complete precedence chain of how CLI flags, workload-spec YAML, and `defaults.yaml` interact, see [Parameter Resolution by Category](../reference/configuration.md#parameter-resolution-by-category) in the Configuration Reference. See also [Common Pitfalls](../reference/configuration.md#common-pitfalls) for documented gotchas like `--rate` vs `aggregate_rate`.

## Multi-Client Composition

Use `blis compose` to merge multiple workload specs into a single spec:

```bash
# Merge a chat workload and a batch workload into one combined spec
./blis compose --from chat.yaml --from batch.yaml > combined.yaml
```

The compose operation:

- **Concatenates** all client lists from each input spec
- **Sums** aggregate rates (e.g., 60 req/s + 40 req/s = 100 req/s total)
- **Renormalizes** `rate_fraction` values proportionally: each client's merged fraction = `original_fraction * (spec_rate / total_rate)`, preserving absolute request rates

This lets you build complex mixed workloads from reusable single-purpose specs.

## Cohort Dynamics

Cohorts model populations of similar clients with time-varying traffic patterns. Instead of defining individual clients, you specify a population count and a traffic pattern. BLIS expands each cohort into individual `ClientSpec` entries.

Three traffic patterns are available:

| Pattern | Behavior | Use Case |
|---------|----------|----------|
| `diurnal` | Sinusoidal rate modulation over 24 hours (peak_hour, peak_to_trough_ratio) | Day/night traffic cycles |
| `spike` | Clients active only during `[start_time_us, start_time_us + duration_us)` | Flash sales, traffic bursts |
| `drain` | Linear ramp-down to zero rate over `ramp_duration_us` | Graceful shutdown, load shedding |

```yaml
version: "2"
aggregate_rate: 200
num_requests: 5000

cohorts:
  - id: "daytime-users"
    population: 50                  # Expands to 50 individual clients
    slo_class: "critical"
    rate_fraction: 0.7
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 256, std_dev: 100, min: 2, max: 2048 }
    output_distribution:
      type: exponential
      params: { mean: 128 }
    diurnal:
      peak_hour: 14                 # Peak at 2 PM
      peak_to_trough_ratio: 3.0    # 3x more traffic at peak vs trough

  - id: "flash-sale"
    population: 20
    slo_class: "standard"
    rate_fraction: 0.3
    arrival:
      process: gamma
      cv: 2.5
    input_distribution:
      type: gaussian
      params: { mean: 128, std_dev: 50, min: 2, max: 512 }
    output_distribution:
      type: exponential
      params: { mean: 64 }
    spike:
      start_time_us: 10000000      # Spike starts at 10 seconds
      duration_us: 5000000         # Lasts 5 seconds
```

### Multi-Turn Sessions

Cohorts support the same advanced client features as explicit clients. The fields below propagate to every expanded member client unchanged.

| Field | Type | Description |
|---|---|---|
| `reasoning` | object | Enables reasoning workload; requires `reason_ratio_distribution` (reasoning-to-output token ratio). `multi_turn.max_rounds` controls conversation depth |
| `closed_loop` | \*bool | `null` (omitted): `true` for multi-turn, `false` for all others. `true`: each round waits for the previous reply. `false`: all rounds pre-generated at open-loop arrival times |
| `timeout` | \*int64 | Per-request timeout in µs. `null`: 300 s default when closed-loop; no deadline when open-loop. `0` = no timeout |
| `prefix_length` | int | Shared prefix token count prepended to every request |
| `network` | object | Client-side network RTT and bandwidth simulation |
| `multimodal` | object | Mixed-modality token generation (text + image/audio/video) |

Example — 50 agentic clients, each running 10-round reasoning sessions with a 2-second think time between rounds:

```yaml
version: "2"
category: reasoning
aggregate_rate: 5.0
cohorts:
  - id: agents
    population: 50
    rate_fraction: 1.0
    arrival:
      process: poisson
    input_distribution:
      type: gaussian
      params: { mean: 512, std_dev: 128, min: 64, max: 2048 }
    output_distribution:
      type: exponential
      params: { mean: 256 }
    reasoning:
      reason_ratio_distribution:
        type: constant
        params: { value: 50 }     # 50% reasoning tokens (0-100 scale)
      multi_turn:
        max_rounds: 10
        think_time_us: 2000000    # 2 seconds between rounds
        context_growth: accumulate
    closed_loop: true
    timeout: 300000000            # 300-second per-request timeout
```

> **Note:** Do not combine `spike` (which creates a short lifecycle window) with multi-turn `reasoning` (which generates rounds seconds apart). Rounds whose arrival times fall outside the spike window are silently filtered by the lifecycle window filter. Use diurnal or drain patterns — or no lifecycle modifier — with multi-turn cohorts.

## Advanced Features

### Multimodal Requests

The `multimodal` field on a client generates requests with combined text, image, audio, and video tokens. Total input = text + (image tokens x image count) + (audio tokens x audio count) + (video tokens x video count).

```yaml
clients:
  - id: "vision-model"
    # ... arrival, rate_fraction, etc.
    multimodal:
      text_distribution:
        type: gaussian
        params: { mean: 128, std_dev: 50, min: 2, max: 512 }
      image_distribution:
        type: constant
        params: { value: 576 }        # Tokens per image (e.g., ViT patch count)
      image_count_distribution:
        type: constant
        params: { value: 1 }          # One image per request
```

Audio and video follow the same pattern with `audio_distribution`/`audio_count_distribution` and `video_distribution`/`video_count_distribution`.

### Reasoning (Multi-Turn with Context Accumulation)

The `reasoning` field generates multi-turn conversation sessions where each round can accumulate prior context. See the [Multi-Turn Conversations](#multi-turn-conversations) section above for a full example. Key fields:

- `reason_ratio_distribution`: fraction of output tokens that represent "reasoning" (sampled as integer percentage, divided by 100)
- `multi_turn.max_rounds`: number of conversation rounds per session
- `multi_turn.think_time_us`: inter-round delay (user think time, in microseconds)
- `multi_turn.context_growth`: controls how each round's input is constructed.
    - `"accumulate"`: round N's input = all prior rounds' inputs + all prior rounds' outputs + freshly sampled new user turn. Input length grows linearly with round index, creating expanding KV cache pressure and enabling prefix-affinity routing reuse across rounds — use this for realistic chat or reasoning sessions.
    - `""` (omit): each round uses only freshly sampled tokens. Input length is stationary across rounds; no cross-round prefix sharing. Use this for agent workloads that do not maintain conversation history.
- `multi_turn.single_session`: if `true`, each client creates exactly one session (useful for modeling persistent chat sessions like inference-perf's `enable_multi_turn_chat`). Default: `false` (multiple independent sessions per client)
- `closed_loop`: controls whether round scheduling adapts to actual server latency. `null` (omit): closed-loop by default for multi-turn clients — each subsequent round is generated after the prior round completes, scheduling its arrival at `completion_time + think_time_us`. `false`: open-loop — all round arrival times are pre-stamped before simulation using a `1 µs/output-token` heuristic; inter-round spacing does not adapt to server latency. Use `false` to reproduce inference-perf-style pre-generated workloads.

### Client-Side Network Latency

The `network` field adds client-perspective latency to server-side metrics. Useful for modeling geographically distributed users:

```yaml
clients:
  - id: "remote-user"
    # ... arrival, rate_fraction, etc.
    network:
      rtt_ms: 50                      # Round-trip time in milliseconds
      bandwidth_mbps: 100             # Link bandwidth (affects upload/download delay)
```

Client TTFT = server TTFT + RTT + upload delay. Client E2E = server E2E + RTT + upload delay + download delay. Upload/download delays are computed from token counts (4 bytes per token ID).

## Built-in Presets and Examples

### Named Presets from defaults.yaml

BLIS ships with preset workload profiles in `defaults.yaml`. Use them via the CLI or the convert command:

```bash
# Run directly with a named preset
./blis run --model qwen/qwen3-14b --workload chatbot

# Convert a preset to a v2 WorkloadSpec YAML for customization
./blis convert preset --name chatbot --rate 10 --num-requests 100 > chatbot.yaml
```

Available presets from `defaults.yaml`:

| Preset | Prompt Mean | Output Mean | Description |
|--------|-------------|-------------|-------------|
| `chatbot` | 256 | 256 | Interactive chat with moderate token lengths |
| `contentgen` | 1024 | 1024 | Content generation with balanced I/O |
| `summarization` | 4096 | 512 | Long-document summarization (high input, moderate output) |
| `multidoc` | 10240 | 1536 | Multi-document processing (very long inputs) |

### Scenario Presets (Programmatic)

The `sim/workload/scenarios.go` module provides scenario functions for common workload patterns. These are used internally by the simulator and in hypothesis experiments:

| Scenario | Function | Key Characteristics |
|----------|----------|-------------------|
| Bursty traffic | `ScenarioBurstyTraffic` | Gamma CV=3.5, exponential tokens, `batch` SLO |
| Unfair tenants | `ScenarioUnfairTenants` | 90% low-priority batch + 10% high-priority critical |
| Prefix-heavy | `ScenarioPrefixHeavy` | 80% shared-prefix + 20% unique, tests prefix caching |
| Mixed SLO | `ScenarioMixedSLO` | Equal mix of critical/standard/batch classes |

### Example Files

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
