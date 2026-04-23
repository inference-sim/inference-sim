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

## Removed Backends

### Blackbox Backend (removed April 2026)

The `blackbox` latency backend used simple alpha/beta regression coefficients without hardware awareness. It has been removed in favor of `trained-physics`, which provides physics-informed estimation with better generalization across models and configurations.

**Migration:** Use `--latency-model trained-physics` (recommended) or `roofline`.
