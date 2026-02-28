# Supported Models

All models below have pre-trained alpha/beta coefficients in `defaults.yaml` for blackbox mode. Models with a HuggingFace `config.json` in `model_configs/` additionally support [roofline mode](../guide/latency-models.md#roofline-mode-analytical).

## Dense Models

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

## MoE Models

| Model | Architecture |
|-------|-------------|
| LLaMA 4 Maverick (FP8) | 17B, 128 experts |
| LLaMA 4 Scout | 17B, 16 experts |
| Mixtral | 8x7B |

## Quantized Variants

Red Hat AI (`redhatai/`) provides FP8, W4A16, and W8A8 quantized variants for many of the above models, including LLaMA 3.1/3.3/4, Mistral Small 3.1, Phi-4, Qwen 2.5, and SmolLM3 3B (FP8 only). See [`defaults.yaml`](https://github.com/inference-sim/inference-sim/blob/main/defaults.yaml) for the full list.

## Roofline-Only Models

Any model with a HuggingFace `config.json` can use roofline mode via `--roofline` or `--model-config-folder`. The `--roofline` flag auto-fetches configs from HuggingFace on first use, caching them in `model_configs/`. Tested architectures include Qwen 2.5 1.5B/3B, Qwen 3 14B, and LLaMA 2 7B/70B.

## Adding a New Model

To add blackbox support for a new model:

1. Calibrate alpha/beta coefficients using live vLLM profiling (see [Configuration: Coefficient Calibration](configuration.md#coefficient-calibration))
2. Add the entry to `defaults.yaml`

To add roofline support:

1. Download the model's `config.json` from HuggingFace
2. Place in `model_configs/<model-name>/config.json`
3. Run with `--roofline --hardware <GPU> --tp <N>`

Or let BLIS auto-fetch it with `--roofline`.
