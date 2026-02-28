# Latency Models

The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships two backends -- blackbox (data-driven) and roofline (analytical) -- and the pluggable architecture supports adding custom backends.

```bash
# Blackbox mode (default) — uses pre-trained coefficients
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 100 --num-requests 500

# Roofline mode — analytical estimation from model architecture
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2 \
  --num-instances 4 --rate 100 --num-requests 500
```

## Blackbox Mode (Default)

Blackbox mode uses trained regression coefficients from `defaults.yaml`, fit offline via Bayesian optimization against real vLLM measurements.

**Beta coefficients** `[beta0, beta1, beta2]` estimate GPU step time:

```
StepTime = beta0 + beta1 * cache_miss_tokens + beta2 * decode_tokens
```

- `beta0` -- fixed per-step overhead (microseconds)
- `beta1` -- cost per prefill token (cache miss)
- `beta2` -- cost per decode token

**Alpha coefficients** `[alpha0, alpha1, alpha2]` estimate CPU-side overhead:

```
QueueingTime           = alpha0 + alpha1 * input_length
OutputTokenProcessingTime = alpha2
```

Pre-trained coefficient sets exist in `defaults.yaml` for common model/GPU/TP combinations (e.g., `meta-llama/llama-3.1-8b-instruct` on H100 with TP=2).

!!! note "Alpha overhead is non-blocking"
    Alpha coefficients model CPU post-processing (tokenization, output serialization) that runs concurrently with GPU execution. Alpha time inflates TTFT and ITL metrics but does **not** block step scheduling -- the next batch step is scheduled at `now + stepTime` regardless of alpha overhead. This matches real vLLM's asynchronous post-processing pipeline.

## Roofline Mode (Analytical)

Roofline mode computes step time analytically from model architecture (FLOPs, parameter count) and hardware specifications (compute throughput, memory bandwidth). It does not require pre-trained coefficients, making it suitable for new models.

### The `--roofline` Flag

The simplest way to use roofline mode:

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2
```

This auto-resolves both required inputs:

1. **Model config** -- checks `model_configs/` for a cached `config.json`, fetches from HuggingFace on miss
2. **Hardware config** -- uses the bundled `hardware_config.json`

For gated models (e.g., LLaMA), set `HF_TOKEN`:

```bash
export HF_TOKEN=your_token_here
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2
```

### Manual Configuration

For full control, provide configs explicitly:

```bash
./blis run --model my-custom-model \
  --model-config-folder ./my-model-configs/ \
  --hardware-config ./my-hardware-config.json \
  --hardware H100 --tp 4
```

### Adding Support for New Models

Any model with a HuggingFace `config.json` can use roofline mode:

1. Download `config.json` from HuggingFace
2. Place it in `model_configs/<model-name>/config.json`
3. Run with `--roofline --hardware <GPU> --tp <N>`

Or let BLIS fetch it automatically with `--roofline`.

### Tensor Parallelism and Roofline

The `--tp` flag divides FLOPs and memory bandwidth across TP ranks:

- Higher TP reduces per-GPU step time (more parallelism)
- Higher TP reduces KV blocks per GPU (memory split across ranks)

When choosing between TP and replication (more instances): TP reduces per-request latency, replication increases throughput. For capacity planning, simulate both configurations.

## When to Use Which

| Aspect | Blackbox (default) | Roofline |
|--------|-------------------|----------|
| **When to use** | Model has pre-trained coefficients in `defaults.yaml` | New model, no coefficients available, quick estimation |
| **Data required** | `defaults.yaml` entry for model/GPU/TP | HuggingFace `config.json` + `hardware_config.json` |
| **Accuracy** | Higher tail accuracy -- trained on real vLLM measurements | Good mean accuracy -- analytical estimation |
| **Alpha overhead** | Full alpha modeling (queueing + output processing) | Alpha from coefficients; step time is analytical |

!!! tip "Roofline for model fit, blackbox for SLO validation"
    Use roofline for "can this model serve at this rate?" analysis -- mean latency rankings are equivalent across modes (H19 experiment confirmed weighted < RR = LL ordering is identical). Use blackbox with trained coefficients for tail latency (p99) estimation under load, where alpha overhead (~4.3ms/req) materially affects scheduling timelines and p99 results diverge between modes.

## Pluggable Architecture

The `LatencyModel` interface (defined in `sim/latency_model.go`) has five methods:

| Method | Purpose |
|--------|---------|
| `StepTime(batch)` | Duration of one batch step given the running batch |
| `QueueingTime(req)` | Arrival-to-queue delay for a request |
| `OutputTokenProcessingTime()` | Per-token post-processing time |
| `SchedulingProcessingTime()` | Scheduling overhead per request |
| `PreemptionProcessingTime()` | Preemption overhead per eviction |

All time estimates are in microseconds (ticks).

New backends register via the `NewLatencyModelFunc` variable in `sim/latency_model.go`. The `sim/latency/register.go` file uses `init()` to wire the factory, breaking the import cycle between `sim/` (interface owner) and `sim/latency/` (implementation). To add a custom backend, implement the five methods and register your factory via `init()` in a sub-package. See [Extension Recipes](../contributing/extension-recipes.md) for a step-by-step guide.

## Further Reading

- [Roofline Estimation](../concepts/roofline.md) -- the mathematical model behind roofline step time calculation
- [Configuration Reference](../reference/configuration.md#roofline-mode) -- all roofline-related CLI flags
