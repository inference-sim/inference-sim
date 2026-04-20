# Supported Models

BLIS supports any dense transformer model with a HuggingFace `config.json`. The default roofline mode auto-fetches configs on first use — no setup required. MoE (Mixture-of-Experts) models are also supported; Mixtral 8x7B has been validated end-to-end.

## Blackbox Coefficient Catalog

The models below have pre-trained alpha/beta coefficients in [`defaults.yaml`](https://github.com/inference-sim/inference-sim/blob/main/defaults.yaml) for [blackbox mode](../guide/latency-models.md#blackbox-mode) (`--latency-model blackbox`). Blackbox mode offers slightly higher accuracy for these specific model/GPU/TP combinations due to per-model fitting.

### Dense Models

| Model | Sizes |
|-------|-------|
| Meta LLaMA 3.1 | 8B |
| Meta LLaMA 3.3 | 70B |
| IBM Granite 3.1 | 8B |
| CodeLlama | 34B |
| Microsoft Phi-4 | 14B |
| Mistral Small (2501) | 24B |
| Mistral Small 3.1 (2503) | 24B |
| NVIDIA LLaMA 3.1 Nemotron | 70B |
| OpenAI GPT-OSS | 20B, 120B |
| Qwen 2.5 | 7B |

### MoE Models

| Model | Architecture |
|-------|-------------|
| LLaMA 4 Maverick (FP8) | 17B, 128 experts |
| LLaMA 4 Scout | 17B, 16 experts |
| Mixtral | 8x7B |

### Quantized Variants

Red Hat AI (`redhatai/`) provides FP8, W4A16, and W8A8 quantized variants for many of the above models, including LLaMA 3.1/3.3/4, Mistral Small 3.1, Phi-4, Qwen 2.5, and SmolLM3 3B (FP8 only). See [`defaults.yaml`](https://github.com/inference-sim/inference-sim/blob/main/defaults.yaml) for the full list.

## Validated Architectures

The analytical modes (roofline, trained-physics) have been validated against real vLLM measurements on these architectures:

- Qwen 2.5 1.5B/3B, Qwen 3 14B
- LLaMA 2 7B/70B
- CodeLlama 34B
- Mixtral 8x7B (MoE)

**Trained-physics** achieves 7% MAPE GPU combined step time across these architectures. Any other model with a HuggingFace `config.json` will work — it just hasn't been formally validated.

!!! note "Parallelism and quantization"
    The analytical latency models (roofline, trained-physics) model tensor parallelism (TP). Data parallelism (DP) and expert parallelism (EP) are not yet modeled. Quantized weight precision is auto-detected and used for weight bandwidth and KV capacity calculations in all analytical backends. Supported formats: GPTQ, AWQ, FP8, and compressed-tensors (via `quantization_config`), plus model name conventions (e.g., `w4a16`, `FP8`).

!!! info "MFU Calibration (Updated March 2026)"
    Hardware MFU (Model FLOPs Utilization) values in `hardware_config.json` were recalibrated based on empirical measurements and roofline theory. The updated values (H100: prefill=0.45/decode=0.30, A100: prefill=0.38/decode=0.18, L40S: prefill=0.32/decode=0.08) reflect conservative estimates for capacity planning. For detailed justification including evidence from FlashAttention-3, NVIDIA MLPerf, and production deployments, see [Discussion #589](https://github.com/inference-sim/inference-sim/discussions/589). If you have existing capacity planning results, consider re-running simulations with the updated values for more accurate estimates.

## Adding Blackbox Coefficients

To calibrate blackbox coefficients for a new model:

1. Run live vLLM profiling (see [Coefficient Calibration](configuration.md#coefficient-calibration))
2. Add the entry to `defaults.yaml`
