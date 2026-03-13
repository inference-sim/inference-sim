# Latency Models

The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships four backends -- blackbox (data-driven), roofline (analytical), cross-model (physics-informed), and trained-roofline (roofline × learned corrections) -- and the pluggable architecture supports adding custom backends.

```bash
# Blackbox mode (default) — uses pre-trained per-model coefficients
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500

# Roofline mode — pure analytical estimation from model architecture
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500

# Cross-model mode — physics-informed with hand-engineered features
./blis run --model qwen/qwen3-14b \
  --latency-model crossmodel --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500

# Trained-roofline mode — roofline basis functions × learned corrections (7% MAPE)
./blis run --model qwen/qwen3-14b \
  --latency-model trained-roofline --hardware H100 --tp 1 \
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

All alpha and beta coefficients must be non-negative. Negative values are rejected at construction time (INV-5: causality). Pre-trained coefficient sets exist in `defaults.yaml` for common model/GPU/TP combinations (e.g., `qwen/qwen3-14b` on H100 with TP=1).

!!! note "Alpha overhead is non-blocking"
    Alpha coefficients model CPU post-processing (tokenization, output serialization) that runs concurrently with GPU execution. Alpha time inflates TTFT and ITL metrics but does **not** block step scheduling -- the next batch step is scheduled at `now + stepTime` regardless of alpha overhead. This matches real vLLM's asynchronous post-processing pipeline.

## Roofline Mode (Analytical)

Roofline mode computes step time analytically from model architecture (FLOPs, parameter count) and hardware specifications (compute throughput, memory bandwidth). It does not require pre-trained coefficients, making it suitable for new models.

### The `--latency-model roofline` Flag

The simplest way to use roofline mode:

```bash
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1
```

This auto-resolves both required inputs:

1. **Model config** -- checks `model_configs/` for a cached `config.json`, fetches from HuggingFace on miss
2. **Hardware config** -- uses the bundled `hardware_config.json`

For gated models (e.g., LLaMA), set `HF_TOKEN`:

```bash
export HF_TOKEN=your_token_here
./blis run --model qwen/qwen3-14b \
  --latency-model roofline --hardware H100 --tp 1
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

The `--tp` flag divides FLOPs and memory traffic across TP ranks:

- Higher TP reduces per-GPU step time (more parallelism)
- Higher TP reduces KV blocks per GPU (memory split across ranks)

When choosing between TP and replication (more instances): TP reduces per-request latency, replication increases throughput. For capacity planning, simulate both configurations.

!!! note "Automatic KV block calculation"
    When using roofline or crossmodel mode, `--total-kv-blocks` is automatically derived from model architecture and GPU memory if not explicitly set. The auto-calculated value accounts for TP (KV heads are sharded across ranks; total GPU memory scales with GPU count). Override with `--total-kv-blocks <N>` for non-standard deployments. The auto-calculation uses reference constants (90% GPU utilization, standard activation/overhead budgets matching the llm-d-benchmark capacity planner) and requires SwiGLU-family activations.

!!! note "Automatic MaxModelLen derivation"
    When using roofline or crossmodel mode and `--max-model-len` is not explicitly set, BLIS auto-derives it from `max_position_embeddings` in the HuggingFace `config.json`. For models with `rope_scaling`, the scaling factor is applied based on vLLM's blacklist approach: types `linear`, `dynamic`, `yarn`, `default`, and `mrope` apply the factor; types `su`, `longrope`, and `llama3` are excluded (these encode the full context in `max_position_embeddings`). For `yarn`, `original_max_position_embeddings` is used as the base when present. `gemma3` models skip `rope_scaling` entirely (`max_position_embeddings` is pre-scaled). The derived value is then capped at the KV-feasible maximum (`total_kv_blocks * block_size`) to prevent context windows from exceeding GPU memory capacity. Override with `--max-model-len <N>` when needed.

## Cross-Model Mode (Physics-Informed)

Cross-model mode estimates step time using 7 globally-fitted coefficients (4 beta for step time + 3 alpha for CPU overhead) that work across model architectures. Unlike blackbox (per-model coefficients) or roofline (no MoE awareness), cross-model uses architecture features from `config.json` to scale a single coefficient set.

```bash
./blis run --model qwen/qwen3-14b \
  --latency-model crossmodel --hardware H100 --tp 1
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

**MoE support:** Cross-model correctly handles Mixture-of-Experts models. The `β₂` term captures the per-token routing and expert dispatch overhead, activated when `num_local_experts > 1` in the model's HuggingFace config.json (single-expert models are dense-equivalent). The MoE indicator is binary (MoE vs dense); the specific active expert count (`num_experts_per_tok`) is parsed for future refinement but not yet used in the formula.

!!! warning "Dense model prefill limitation"
    For dense models (non-MoE), step time does not scale with prefill token count — prefill compute cost is absorbed into the per-layer overhead (β₀). A batch prefilling 1 token costs the same as 2048 tokens. This is a known approximation from the training methodology (prefill KV writes overlap with compute on H100). For prefill-heavy dense-model workloads, **blackbox mode with trained coefficients** provides more accurate estimates because its `β₁` term explicitly models per-prefill-token cost.

!!! note "Automatic KV block calculation"
    Like roofline mode, crossmodel auto-derives `--total-kv-blocks` from model architecture and GPU memory when the flag is not set. Override with `--total-kv-blocks <N>` for non-standard deployments. The auto-calculation uses reference constants (90% GPU utilization, standard activation/overhead budgets matching the llm-d-benchmark capacity planner) and requires SwiGLU-family activations (`silu`, `swiglu`, `geglu`).

## Trained-Roofline Mode (Recommended for New Models)

Trained-roofline mode applies **learned correction factors** to analytical roofline basis functions, combining the physical grounding of roofline with the accuracy of data-driven fitting. Coefficients are fitted from 137K real vLLM requests across 4 architectures (Llama-2-7b, Llama-2-70b, Mixtral-8x7B, CodeLlama-34b) via non-negative least squares regression.

```bash
./blis run --model qwen/qwen3-14b \
  --latency-model trained-roofline --hardware H100 --tp 1
```

Same auto-fetch chain as roofline and crossmodel (HuggingFace config + hardware config resolution).

**StepTime formula** (7 terms):

```
StepTime = β₁ × max(T_pf_compute, T_pf_kv)    # prefill roofline bottleneck × correction
         + β₂ × max(T_dc_compute, T_dc_kv)    # decode roofline bottleneck × correction
         + β₃ × T_weight                       # weight loading × correction
         + β₄ × T_tp                           # TP communication × correction
         + β₅ × L                              # per-layer overhead (µs/layer)
         + β₆ × batch_size                     # per-request scheduling (µs/req)
         + β₇                                  # per-step fixed overhead (µs)
```

Where each basis function (T_pf_compute, T_pf_kv, etc.) is a full analytical roofline calculation from model architecture + hardware specs. β₁-β₄ are dimensionless correction factors (near 1.0 = roofline is accurate). β₅-β₇ capture overhead not in the roofline model.

**Key differences from roofline mode:**

- **No MFU scaling** -- β₁ and β₂ ARE the MFU corrections. Applying `MfuPrefill`/`MfuDecode` would double-count.
- **3-matrix SwiGLU** -- uses `6 × d × d_ff` for FFN FLOPs (gate + up + down) vs roofline's 2-matrix convention.
- **MoE-aware weight loading** -- `min(N, max(k, B×k))` effective experts, not all N.

**Alpha model** (3 coefficients):

- `α₀` = API processing overhead (constant, added to TTFT via `QueueingTime`)
- `α₁` = Fixed per-request post-decode overhead (added to E2E via `PostDecodeFixedOverhead`)
- `α₂` = Per-output-token detokenization cost (added to ITL via `OutputTokenProcessingTime`)

**Pre-trained coefficients** (7% MAPE on GPU combined step time, test split) are stored in `trained_roofline_defaults` in `defaults.yaml`. No per-model calibration needed -- the roofline basis functions handle architecture-specific scaling.

!!! note "TTFT accuracy caveat"
    The "7% MAPE" headline applies to GPU combined step time. The alpha model has higher error: α₀ (pre-queueing) has 93% MAPE because it's a single constant for a highly variable real-world quantity. TTFT predictions have higher error than GPU step time predictions. For TTFT-sensitive analysis, consider calibrating α₀ per-deployment.

!!! note "Chunked prefill limitation"
    Trained-roofline was fitted on single-step prefill data. When used with `--long-prefill-token-threshold > 0` (chunked prefill), the attention FLOPs formula uses `len(InputTokens)` (total prompt) as context for each chunk, overestimating early-chunk step times. For chunked-prefill workloads, pure roofline mode may be more accurate until coefficients are refit on chunked data.

## When to Use Which

| Aspect | Blackbox (default) | Roofline | Cross-Model | Trained-Roofline |
|--------|-------------------|----------|-------------|------------------|
| **When to use** | Model has per-model coefficients in `defaults.yaml` | Quick analytical estimate | Hand-engineered physics features | **Recommended** for new models (best accuracy without per-model training) |
| **Data required** | `defaults.yaml` entry for model/GPU/TP | HF `config.json` + `--hardware` + `--tp` | HF `config.json` + `--hardware` + `--tp` | HF `config.json` + `--hardware` + `--tp` (global coefficients bundled) |
| **GPU step time accuracy** | Highest (per-model) | Good (analytical) | Good (7 global params) | **7% MAPE** (10 global params, roofline × corrections) |
| **MoE support** | If trained | No (dense only) | Yes (binary indicator) | Yes (per-expert FLOPs + effective expert count) |
| **Alpha model** | α₀ + α₁·inputLen | Same as blackbox | Same as blackbox | α₀ (constant), α₁ (post-decode fixed), α₂ (per-token) |
| **PostDecodeFixedOverhead** | 0 | 0 | 0 | α₁ (~1.85ms) |

!!! tip "Choosing the right mode"
    **Trained-roofline** is the recommended default for any model with a HuggingFace `config.json` (7% MAPE GPU combined, MoE-aware, no per-model calibration needed). **Blackbox** for models with per-model coefficients in `defaults.yaml` (slightly higher accuracy due to per-model fitting). **Cross-model** for backward compatibility with existing crossmodel workflows. **Roofline** for pure analytical estimates when no learned corrections are desired.

## Pluggable Architecture

The `LatencyModel` interface (defined in `sim/latency_model.go`) has four methods:

| Method | Purpose |
|--------|---------|
| `StepTime(batch)` | Duration of one batch step given the running batch |
| `QueueingTime(req)` | Arrival-to-queue delay for a request |
| `OutputTokenProcessingTime()` | Per-token post-processing time |
| `PostDecodeFixedOverhead()` | Fixed per-request overhead at completion (0 for blackbox/roofline/crossmodel) |

All time estimates are in microseconds (ticks).

New backends register via the `NewLatencyModelFunc` variable in `sim/latency_model.go`. The `sim/latency/register.go` file uses `init()` to wire the factory, breaking the import cycle between `sim/` (interface owner) and `sim/latency/` (implementation). To add a custom backend, implement the four methods and register your factory via `init()` in a sub-package. See [Extension Recipes](../contributing/extension-recipes.md) for a step-by-step guide.

## Further Reading

- [Roofline Estimation](../concepts/roofline.md) -- the mathematical model behind roofline step time calculation
- [Configuration Reference](../reference/configuration.md#roofline-mode) -- all roofline-related CLI flags
