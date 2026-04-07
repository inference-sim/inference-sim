# Latency Models

The `LatencyModel` interface determines how BLIS estimates GPU step time for each batch iteration. BLIS ships five backends -- roofline (default, analytical), blackbox (data-driven), cross-model (physics-informed), trained-roofline (roofline × learned corrections), and trained-physics (roofline × learned corrections) -- and the pluggable architecture supports adding custom backends.

```bash
# Roofline mode (default) — analytical estimation from model architecture
./blis run --model qwen/qwen3-14b \
  --num-instances 4 --rate 100 --num-requests 500

# Blackbox mode — uses pre-trained per-model coefficients
./blis run --model qwen/qwen3-14b \
  --latency-model blackbox \
  --num-instances 4 --rate 100 --num-requests 500

# Cross-model mode — physics-informed with hand-engineered features
./blis run --model qwen/qwen3-14b \
  --latency-model crossmodel --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500

# Trained-roofline mode — roofline basis functions × learned corrections (7% MAPE)
./blis run --model qwen/qwen3-14b \
  --latency-model trained-roofline --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500

# Trained-physics mode — roofline × architecture-aware basis functions × learned corrections
./blis run --model qwen/qwen3-14b \
  --latency-model trained-physics --hardware H100 --tp 1 \
  --num-instances 4 --rate 100 --num-requests 500
```

## Blackbox Mode

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

## Roofline Mode (Default)

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

**Supported hardware:** The bundled `hardware_config.json` includes specs for **H100** (80 GB HBM3, 989.5 TFLOPS BF16, 3.35 TB/s), **A100-SXM** (80 GB HBM2e, 312 TFLOPS BF16, 2.04 TB/s), and **A100-80** (alias for A100-SXM). To use a different GPU, add an entry to `hardware_config.json` with the required fields (`TFlopsPeak`, `BwPeakTBs`, `mfuPrefill`, `mfuDecode`, `MemoryGiB`) and reference it via `--hardware <name>`.

**Validated models:** Any dense or MoE transformer with a HuggingFace `config.json` works. The following have been validated end-to-end:

- [Llama-2-7B](https://huggingface.co/meta-llama/Llama-2-7b-hf) / [Llama-2-70B](https://huggingface.co/meta-llama/Llama-2-70b-hf)
- [Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B)
- [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) (MoE)
- [CodeLlama-34B](https://huggingface.co/codellama/CodeLlama-34b-Instruct-hf)

Set `HF_TOKEN` to access gated models (e.g., [Llama-2](https://huggingface.co/meta-llama/Llama-2-7b-hf)) and avoid rate limits:

```bash
export HF_TOKEN=your_token_here
./blis run --model meta-llama/llama-3.1-8b-instruct \
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

**MoE support:** Cross-model correctly handles Mixture-of-Experts models. The `β₂` term captures the per-token routing and expert dispatch overhead, activated when `num_local_experts > 0` in the model's HuggingFace config.json. The MoE indicator is binary (MoE vs dense); the specific active expert count (`num_experts_per_tok`) is parsed for future refinement but not yet used in the formula.

!!! warning "Dense model prefill limitation"
    For dense models (non-MoE), step time does not scale with prefill token count — prefill compute cost is absorbed into the per-layer overhead (β₀). A batch prefilling 1 token costs the same as 2048 tokens. This is a known approximation from the training methodology (prefill KV writes overlap with compute on H100). For prefill-heavy dense-model workloads, **blackbox mode with trained coefficients** provides more accurate estimates because its `β₁` term explicitly models per-prefill-token cost.

!!! note "Automatic KV block calculation"
    Like roofline mode, crossmodel auto-derives `--total-kv-blocks` from model architecture and GPU memory when the flag is not set. Override with `--total-kv-blocks <N>` for non-standard deployments. The auto-calculation uses reference constants (90% GPU utilization, standard activation/overhead budgets matching the llm-d-benchmark capacity planner) and requires SwiGLU-family activations (`silu`, `swiglu`, `geglu`).

## Trained-Roofline Mode

Trained-roofline mode applies **learned correction factors** to analytical roofline basis functions, combining the physical grounding of roofline with the accuracy of data-driven fitting. Coefficients are fitted from 137K real vLLM requests across 4 architectures (Llama-2-7b, Llama-2-70b, Mixtral-8x7B, CodeLlama-34b) via non-negative least squares regression.

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --latency-model trained-roofline --hardware H100 --tp 2
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

## Trained-Physics Mode (Recommended for New Models)

Trained-physics mode applies **learned correction factors** to analytical roofline basis functions, combining the physical grounding of roofline with the accuracy of data-driven fitting. Coefficients are fitted from real vLLM measurements and generalize across model architectures, workloads, and TP configurations.

```bash
./blis run --model qwen/qwen3-14b \
  --latency-model trained-physics --hardware H100 --tp 1
```

Same auto-fetch chain as roofline mode (HuggingFace config + hardware config resolution).

**StepTime formula** (10 beta coefficients in bundled defaults):

```
StepTime = β₁ₐ × T_pf_compute                  # prefill compute only
         + β₁ᵦ × T_pf_kv                       # prefill memory (typically ~0)
         + β₂ₐ × T_dc_compute                  # decode compute (typically ~0)
         + β₂ᵦ × T_dc_kv                       # decode memory only
         + β₃ × T_weight                       # weight loading × correction
         + β₄ × T_tp                           # TP communication × correction
         + β₅ × L                              # per-layer overhead (µs/layer)
         + β₆ × batch_size                     # per-request scheduling (µs/req)
         + β₇                                  # per-step fixed overhead (µs)
         + β₈ × nMoE                           # per-MoE-layer overhead (µs/layer)
```

The model supports 7-10 beta coefficients. Bundled defaults use 10 coefficients with prefill/decode split.

**Beta coefficients:**

- **β₁ₐ** (prefill compute, ~0.15): Corrects analytical FlashAttention + MLP FLOP estimates for kernel efficiency, memory access patterns.
- **β₁ᵦ** (prefill memory, ~0): Prefill KV cache write bandwidth correction (typically near zero).
- **β₂ₐ** (decode compute, ~0): Decode compute correction (typically near zero, decode is memory-bound).
- **β₂ᵦ** (decode memory, ~1.9): Corrects KV cache read bandwidth. Primary decode bottleneck.
- **β₃** (weight loading, ~1.4): Corrects model weight bandwidth for cache effects, prefetching, HBM contention.
- **β₄** (TP communication, ~0.75): Corrects tensor-parallel All-Reduce overhead.
- **β₅** (per-layer, ~32 µs/layer): Fixed overhead per transformer layer: kernel launch, CUDA graph, residual connections.
- **β₆** (per-request, ~4 µs/request): Scheduling overhead per request: queue management, attention mask construction.
- **β₇** (per-step, ~126 µs/step): Fixed overhead per step: CUDA synchronization, sampler invocation.
- **β₈** (MoE-layer, ~482 µs/layer): Per-MoE-layer overhead for router gating, token permutation. Architecture-aware: applies only to interleaved MoE architectures (InterleaveMoELayerStep > 0). Zero for uniform MoE and dense models.

**Alpha coefficients** (3 terms, API/framework overheads in µs):

- **α₀** (QueueingTime, ~15,563 µs): Fixed per-request API processing (HTTP parsing, request validation, queue insertion).
- **α₁** (PostDecodeFixedOverhead, ~777 µs): Fixed per-request post-decode overhead (detokenization setup, finish reason determination).
- **α₂** (OutputTokenProcessingTime, ~46 µs/token): Per-output-token overhead (streaming token transmission, incremental detokenization).

**Pre-trained coefficients** are stored in `trained_physics_coefficients` in `defaults.yaml`. No per-model calibration needed -- the model generalizes across architectures, workloads, and TP configurations.

### Generalization Scope

The trained-physics model is designed to generalize without per-model calibration:

**Supported hardware:**

- **H100-SXM** (80 GB HBM3, 989.5 TFLOPS BF16 / 1979 TFLOPS FP8, 3.35 TB/s)
- **A100-SXM** (80 GB HBM2e, 312 TFLOPS BF16, 2.04 TB/s)
- **A100-80** (alias for A100-SXM)

Coefficients are trained on H100 traces but the roofline basis functions automatically scale to A100's compute/bandwidth specifications via hardware config.

**Model architectures:**

- **Dense transformers** (Llama-2, Qwen3, GPT, etc.): Standard attention + MLP layers
- **Uniform MoE** (Mixtral): All layers are MoE with top-k expert routing
- **Interleaved MoE** (Scout): Alternating MoE and dense layers with architecture-specific β₈ overhead

The model automatically detects MoE configuration from `config.json` (`num_local_experts`, `num_experts_per_tok`, `interleave_moe_layer_step`) and adjusts basis functions accordingly.

**Workload types:**

- **Prefill-heavy** (large input, short output): Chatbot prompts, document Q&A
- **Decode-heavy** (small input, long output): Content generation, code completion
- **Mixed batches** (concurrent prefill/decode): Production serving with heterogeneous requests
- **TP configurations**: TP=1, TP=2, TP=4, TP=8 (All-Reduce overhead scales via β₄)

**Why "recommended" over trained-roofline:**

Trained-physics uses **14 basis functions** (prefill compute/memory split, decode compute/memory split, weight, TP, layer overhead, batch overhead, step overhead, MoE overhead, 3 alpha overheads) that capture more architectural detail than trained-roofline's **11 terms** (prefill/decode combined via max(), simpler overhead structure). The additional basis functions enable better generalization to unseen model architectures (especially interleaved MoE) and batch compositions (mixed prefill/decode). Both models achieve similar accuracy on H100 (trained-physics: good, trained-roofline: 7% MAPE), but trained-physics is more robust to architectural variations.

!!! note "MoE architecture detection"
    β₈ applies conditionally based on `InterleaveMoELayerStep` from the model's `config.json`: 0 = uniform MoE (β₈ skipped), 1 = alternating MoE/dense (β₈ × 24 layers for Scout's 48 total), 2 = every 3rd layer is MoE, etc. This prevents over-penalizing uniform MoE models like Mixtral where expert routing overhead is amortized across all layers.

## When to Use Which

| Aspect | Roofline (default) | Blackbox | Cross-Model | Trained-Roofline | Trained-Physics |
|--------|-------------------|----------|-------------|------------------|-----------------|
| **When to use** | Quick analytical estimate | Model has per-model coefficients in `defaults.yaml` | Hand-engineered physics features | Roofline × learned corrections (7% MAPE) | **Recommended** for new models (generalizes across architectures, workloads, TP) |
| **Data required** | HF `config.json` + `--hardware` + `--tp` | `defaults.yaml` entry for model/GPU/TP | HF `config.json` + `--hardware` + `--tp` | HF `config.json` + `--hardware` + `--tp` (global coefficients bundled) | HF `config.json` + `--hardware` + `--tp` (global coefficients bundled) |
| **GPU step time accuracy** | Good (analytical) | Highest (per-model) | Good (7 global params) | **7% MAPE** (10 global params, roofline × corrections) | Good (13 global params, physics-informed basis functions) |
| **MoE support** | Yes (per-expert FLOPs + effective expert count) | If trained | Yes (binary indicator) | Yes (per-expert FLOPs + effective expert count) | Yes (per-expert FLOPs + effective expert count + β₈ per-MoE-layer overhead) |
| **Alpha model** | Same as blackbox | α₀ + α₁·inputLen | Same as blackbox | α₀ (constant), α₁ (post-decode fixed), α₂ (per-token) | α₀ (constant), α₁ (post-decode fixed), α₂ (per-token) |
| **PostDecodeFixedOverhead** | 0 | 0 | 0 | α₁ (~1.85ms) | α₁ (~777µs) |

!!! tip "Choosing the right mode"
    **Trained-physics** is the recommended default for any model with a HuggingFace `config.json` (generalizes across architectures, workloads, and TP configurations without per-model calibration). **Blackbox** for models with per-model coefficients in `defaults.yaml` (highest accuracy due to per-model fitting). **Trained-roofline** is an alternative trained model (7% MAPE GPU combined). **Cross-model** for backward compatibility with existing crossmodel workflows. **Roofline** for pure analytical estimates when no learned corrections are desired.

!!! warning "Current limitations"
    All analytical latency models support tensor parallelism (TP). Data parallelism (DP) and expert parallelism (EP) scheduling overhead are not yet modeled. Quantized weight precision (GPTQ, AWQ, FP8, compressed-tensors) is auto-detected from `quantization_config`, model name conventions (e.g., `w4a16`, `FP8`), or `torch_dtype` fallback, and is used for weight bandwidth and KV capacity calculations. MFU calibration values are still derived from FP16/BF16 measurements.

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
