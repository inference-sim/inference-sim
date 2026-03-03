# Latency Models

The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships two backends -- blackbox (data-driven) and roofline (analytical) -- and the pluggable architecture supports adding custom backends.

```bash
# Blackbox mode (default) — uses pre-trained coefficients
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --num-instances 4 --rate 100 --num-requests 500

# Roofline mode — analytical estimation from model architecture
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model roofline --hardware H100 --tp 2 \
  --num-instances 4 --rate 100 --num-requests 500

# Cross-model mode — physics-informed estimation from config.json (MoE-aware)
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model crossmodel --hardware H100 --tp 2 \
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

### The `--latency-model roofline` Flag

The simplest way to use roofline mode:

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model roofline --hardware H100 --tp 2
```

This auto-resolves both required inputs:

1. **Model config** -- checks `model_configs/` for a cached `config.json`, fetches from HuggingFace on miss
2. **Hardware config** -- uses the bundled `hardware_config.json`

For gated models (e.g., LLaMA), set `HF_TOKEN`:

```bash
export HF_TOKEN=your_token_here
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model roofline --hardware H100 --tp 2
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
3. Run with `--latency-model roofline --hardware <GPU> --tp <N>`

Or let BLIS fetch it automatically with `--latency-model roofline`.

### Tensor Parallelism and Roofline

The `--tp` flag divides FLOPs and memory bandwidth across TP ranks:

- Higher TP reduces per-GPU step time (more parallelism)
- Higher TP reduces KV blocks per GPU (memory split across ranks)

When choosing between TP and replication (more instances): TP reduces per-request latency, replication increases throughput. For capacity planning, simulate both configurations.

## Cross-Model Mode (Physics-Informed)

Cross-model mode estimates step time using 4 globally-fitted physics coefficients that work across model architectures. Unlike blackbox (per-model coefficients) or roofline (no MoE awareness), cross-model uses architecture features from `config.json` to scale a single coefficient set.

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model crossmodel --hardware H100 --tp 2
```

**StepTime formula:**

```
stepTime = β₀ × numLayers           # Per-layer CUDA kernel dispatch
         + β₁ × dc × kvDimScaled    # KV cache bandwidth (decode only)
         + β₂ × (pf+dc) × isMoE    # MoE expert routing (Mixtral, etc.)
         + β₃ × isTP                # TP synchronization barrier
```

Where `kvDimScaled = numLayers × numKVHeads × headDim / TP × 1e-6`, `isMoE = 1.0` if the model has expert routing, and `isTP = 1.0` if TP > 1.

**Pre-trained coefficients** from real vLLM measurements across 4 architectures (7B-70B dense + 8x7B MoE) are stored in `crossmodel_defaults` in `defaults.yaml`. No per-model calibration needed.

**MoE support:** Cross-model correctly handles Mixture-of-Experts models. The `β₂` term captures the per-token routing and expert dispatch overhead. The `num_local_experts` and `num_experts_per_tok` fields are parsed directly from the HuggingFace config.json.

!!! warning "Dense model prefill limitation"
    For dense models (non-MoE), step time does not scale with prefill token count — prefill compute cost is absorbed into the per-layer overhead (β₀). A batch prefilling 1 token costs the same as 2048 tokens. This is a known approximation from the training methodology (prefill KV writes overlap with compute on H100). For prefill-heavy dense-model workloads, **blackbox mode with trained coefficients** provides more accurate estimates because its `β₁` term explicitly models per-prefill-token cost.

## When to Use Which

| Aspect | Blackbox (default) | Roofline | Cross-Model |
|--------|-------------------|----------|-------------|
| **When to use** | Model has per-model coefficients in `defaults.yaml` | Quick estimation, no training data | New model from config.json, MoE models |
| **Data required** | `defaults.yaml` entry for model/GPU/TP | HuggingFace `config.json` + `hardware_config.json` | HuggingFace `config.json` (global coefficients bundled) |
| **Accuracy** | Highest (trained on real per-model measurements) | Good mean (analytical FLOPs/bandwidth) | Good mean across architectures (4 global coefficients) |
| **MoE support** | Yes (if trained coefficients exist) | No (~4x overestimate) | Yes (explicit MoE term) |
| **Alpha overhead** | Full alpha modeling | Alpha from coefficients; step is analytical | Full alpha modeling (same as blackbox) |

!!! tip "Choosing the right mode"
    **Blackbox** for models with trained coefficients (highest accuracy). **Cross-model** for new models or MoE models without per-model coefficients (global physics coefficients + config.json features). **Roofline** for quick analytical estimates of dense models when no coefficients are available at all.

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
