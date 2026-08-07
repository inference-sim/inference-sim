# Model Compatibility

BLIS supports **any transformer model with a HuggingFace `config.json`** — no per-model setup or calibration required. Both latency backends (roofline and trained-physics) generalize across architectures.

BLIS has been tested and accuracy validated across a variety of model families and sizes, including both dense transformers and MoE (Mixture-of-Experts) architectures.

The simulator auto-fetches `config.json` from HuggingFace on first use. For gated models, set `HF_TOKEN`. For offline environments, cache configs locally in `model_configs/`.

## Validated Architectures

The latency models have been validated against real vLLM measurements on:

- Qwen 2.5 1.5B/3B, Qwen 3 14B
- LLaMA 2 7B/70B
- CodeLlama 34B
- Mixtral 8x7B (MoE)

**Trained-physics** achieves 7% MAPE GPU combined step time across these architectures. Any other model with a HuggingFace `config.json` will work — it just hasn't been formally validated.

!!! note "Parallelism and quantization"
    The analytical latency models (roofline, trained-physics) model tensor parallelism (TP). Data parallelism (DP) and expert parallelism (EP) are not yet modeled. Quantized weight precision is auto-detected and used for weight bandwidth and KV capacity calculations. Supported formats: GPTQ, AWQ, FP8, and compressed-tensors (via `quantization_config`), plus model name conventions (e.g., `w4a16`, `FP8`).

!!! info "MFU Calibration (Updated March 2026)"
    Hardware MFU (Model FLOPs Utilization) values in `hardware_config.json` were recalibrated based on empirical measurements and roofline theory. The updated values (H100: prefill=0.45/decode=0.30, A100: prefill=0.38/decode=0.18, L40S: prefill=0.32/decode=0.08) reflect conservative estimates for capacity planning. For detailed justification including evidence from FlashAttention-3, NVIDIA MLPerf, and production deployments, see [Discussion #589](https://github.com/inference-sim/inference-sim/discussions/589). If you have existing capacity planning results, consider re-running simulations with the updated values for more accurate estimates.

## Attention & KV-Cache Shape (MLA, head_dim, dense-prefix MoE)

BLIS derives KV-cache block capacity and total model-weight bytes from the HuggingFace `config.json`. As of #1527 the shape model represents the modern MLA MoE family (DeepSeek-V2/V3, Kimi-K3, GLM-5.2 `glm_moe_dsa`):

- **Explicit `head_dim`.** When a config declares `head_dim` (common in modern MLA/GQA designs where it differs from `hidden_size / num_attention_heads` — e.g. GLM-5.2: `head_dim=192` while `6144/64=96`), it is used for KV-cache and weight sizing. Absent the key, BLIS falls back to `hidden/heads` (unchanged behavior). *Note:* the step-time (latency) models still use `hidden/heads`; `head_dim` currently affects capacity only.
- **MLA compressed-KV.** For Multi-head Latent Attention models (`kv_lora_rank` present), the KV cache stores a single compressed latent of `kv_lora_rank + qk_rope_head_dim` scalars per token per layer (e.g. DeepSeek `512 + 64 = 576`), **not** the standard MHA/GQA `2 × head_dim × num_kv_heads`. The latent is replicated across tensor-parallel ranks (not sharded), matching vLLM's MLA cache. This corrects both KV capacity and PD KV-transfer sizing for the whole MLA family.
- **Dense-prefix MoE (`first_k_dense_replace`).** MoE models that run their first *K* layers as dense MLP (e.g. GLM-5.2: 3 of 78 dense; DeepSeek-V2-Lite: 1 of 27) have their weight estimate split into *K* dense layers + remaining MoE layers, instead of counting every layer as MoE. This is a prefix split, distinct from the every-Nth `interleave_moe_layer_step` pattern.

!!! warning "Known approximations for FP8 / DSA models"
    - **Block-wise FP8** (`weight_block_size`, e.g. GLM-5.2-FP8) is treated as a flat `1.0` byte/param. The per-block scale overhead and the `modules_to_not_convert` set (layernorms, gates, indexer, `lm_head`, embeddings, MTP modules kept at bf16) are **not** modeled, giving a slightly **optimistic** (low) weight estimate.
    - The **DeepSeek sparse-attention (DSA) indexer** (`index_n_heads`, `index_topk`) contributes no weight or index-KV — a second-order optimistic gap.
    - **MLA attention weight projections** (`q_lora_rank`/`kv_lora_rank` down/up matrices) keep the standard dense-attention weight approximation; only the KV *footprint* uses the compressed-latent shape.
    - **Speculative decoding / MTP throughput** is not modeled (decode is one token per step); this is tracked separately in [#1528](https://github.com/inference-sim/inference-sim/issues/1528).

## Removed Backends

### Blackbox Backend (removed April 2026)

The `blackbox` latency backend used simple alpha/beta regression coefficients without hardware awareness. It has been removed in favor of `trained-physics`, which provides physics-informed estimation with better generalization across models and configurations.

**Migration:** Use `--latency-model trained-physics` (recommended) or `roofline`.
