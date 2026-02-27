# Roofline Mode

This guide covers analytical step time estimation using the roofline model — BLIS's mode for running simulations without pre-trained coefficients.

```bash
# Quick example: roofline mode with auto-fetch
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2 \
  --num-instances 4 --rate 100 --num-requests 500
```

## When to Use Roofline vs Blackbox

| Mode | When to Use | Accuracy |
|------|------------|----------|
| **Blackbox** (default) | Model has pre-trained coefficients in `defaults.yaml` | Higher — trained on real vLLM measurements |
| **Roofline** | New model, no coefficients available, quick estimation | Lower tail accuracy — no alpha overhead modeling |

!!! tip "Roofline for model fit, blackbox for SLO validation"
    Use roofline for "can this model serve at this rate?" analysis (mean latency is equivalent). Use blackbox with trained coefficients for tail latency (p99) estimation under load, where alpha overhead (~4.3ms/req) materially affects results.

## The `--roofline` Flag

The simplest way to use roofline mode:

```bash
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2
```

This auto-resolves both required inputs:

1. **Model config** — checks `model_configs/` for a cached `config.json`, fetches from HuggingFace on miss
2. **Hardware config** — uses the bundled `hardware_config.json`

For gated models (e.g., LLaMA), set `HF_TOKEN`:

```bash
export HF_TOKEN=your_token_here
./blis run --model meta-llama/llama-3.1-8b-instruct \
  --roofline --hardware H100 --tp 2
```

## Manual Configuration

For full control, provide configs explicitly:

```bash
./blis run --model my-custom-model \
  --model-config-folder ./my-model-configs/ \
  --hardware-config ./my-hardware-config.json \
  --hardware H100 --tp 4
```

## Adding Support for New Models

Any model with a HuggingFace `config.json` can use roofline mode:

1. Download `config.json` from HuggingFace
2. Place it in `model_configs/<model-name>/config.json`
3. Run with `--roofline --hardware <GPU> --tp <N>`

Or let BLIS fetch it automatically with `--roofline`.

## Tensor Parallelism and Roofline

The `--tp` flag divides FLOPs and memory bandwidth across TP ranks:

- Higher TP → lower per-GPU step time (more parallelism)
- Higher TP → fewer KV blocks per GPU (memory split across ranks)

When choosing between TP and replication (more instances): TP reduces per-request latency, replication increases throughput. For capacity planning, simulate both configurations.

## Further Reading

- [Roofline Estimation](../concepts/roofline.md) — the mathematical model
- [Configuration Reference](../reference/configuration.md#roofline-mode) — all roofline flags
