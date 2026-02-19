# Attention Configuration Reference

Benchmark data is organized by attention shape `{num_attention_heads}-{num_key_value_heads}-{head_dim}`, **not by model name**. Multiple models share the same benchmark data if they have matching attention configurations.

## Current Benchmark Configs

After comprehensive benchmarking, all 6 common shapes are available:

```bash
# View available benchmark data
ls InferSim/bench_data/h100/mha/prefill/
# 28-4-128.csv   ← Qwen2-7B, Qwen2.5-7B
# 32-32-128.csv  ← Llama-2-7B, Llama-1-7B (MHA)
# 32-8-128.csv   ← Mistral-7B, Mixtral-8x7B
# 40-40-128.csv  ← Llama-2-13B, Qwen-14B (MHA)
# 56-8-128.csv   ← CodeLlama-34B
# 64-8-128.csv   ← Llama-2-70B, Qwen2-72B

ls InferSim/bench_data/h100/mha/decode/
# Each shape has 3 TP variants: {shape}-tp1.csv, {shape}-tp2.csv, {shape}-tp4.csv
```

```bash
# View configured models
cat config/benchmark_config.json | jq '.benchmark_configs[] | {name, nh: .num_attention_heads, nkv: .num_key_value_heads, dh: .head_dim}'
```

## Common Model Configurations

| Config | Architecture | Models | Notes |
|--------|--------------|--------|-------|
| **28-4-128** | GQA | Qwen2-7B, Qwen2.5-7B | 7B with GQA (7:1) |
| **32-32-128** | MHA | Llama-2-7B, Llama-1-7B | Standard 7B models |
| **32-8-128** | GQA | Mistral-7B, Mixtral-8x7B | 7B with GQA (4:1) |
| **40-40-128** | MHA | Llama-2-13B, Qwen-14B (v1) | 13B standard |
| **56-8-128** | GQA | CodeLlama-34B | 34B variant |
| **64-8-128** | GQA | Llama-2-70B, Qwen2-72B | 70B with GQA (8:1) |

## How to Look Up Any Model

### From HuggingFace
```bash
curl -s https://huggingface.co/meta-llama/Llama-2-7b-hf/raw/main/config.json | \
  jq '{num_attention_heads, num_key_value_heads, hidden_size}'
```

Or visit: `https://huggingface.co/<org>/<model>/blob/main/config.json`

### Calculate head_dim
```
head_dim = hidden_size / num_attention_heads
```

Example: Llama-2-7B has `hidden_size=4096, num_attention_heads=32`
- `head_dim = 4096 / 32 = 128`

## Adding a New Model

1. **Check if attention config already exists:**
   ```bash
   ls InferSim/bench_data/mha/prefill/h100/
   ```

2. **If config exists:** Just add model to `config/benchmark_config.json` - no benchmarking needed!

3. **If new config:** Add to config and run benchmarks:
   ```bash
   python scripts/openshift/generate_job.py --gpu H100 --model your-model --phase prefill
   python scripts/openshift/generate_job.py --gpu H100 --model your-model --phase decode --tp 1
   ```

## Why This Matters

**Reusing benchmark data saves GPU time:**
- Llama-3-8B with 32-32-128? Use existing Llama-2-7B data
- New 7B model with 32-32-128? Use existing data
- Only benchmark when attention shape is genuinely new

**When MFU differs:**
- Changing from MHA (32-32-128) to GQA (32-8-128) = different kernels = need new benchmarks
- Same shape, different model size = same MFU (kernel doesn't care about layer count)

## Architecture Types

- **MHA (Multi-Head Attention):** `nh == nkv` (e.g., 32-32-128)
  - Each Q head has its own KV head

- **GQA (Grouped-Query Attention):** `nkv < nh` (e.g., 64-8-128)
  - Multiple Q heads share KV heads
  - Used in: Llama-2-70B, Mistral, CodeLlama

- **MQA (Multi-Query Attention):** `nkv == 1` (e.g., 32-1-128)
  - All heads share one KV head
  - Used in: PaLM, StarCoder
