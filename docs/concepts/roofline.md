# Roofline Step Time Estimation Logic

This document describes the analytical approach used to estimate the GPU latency for a single inference step using a roofline model. Roofline is the default latency model in BLIS — it requires no training and works off-the-shelf for any Huggingface LLM whose `config.json` is saved under `model_configs/` (auto-fetched from HuggingFace on first use).

!!! tip "Trained-Physics: higher accuracy"
    For higher accuracy, use `--latency-model trained-physics` which applies learned correction factors to roofline basis functions with architecture-aware MoE scaling. See [Trained-Physics Mode](../guide/latency-models.md#trained-physics-mode-recommended-for-new-models).


!!! note "Scope: TP-only, quantized weight memory supported"
    The roofline model accounts for tensor parallelism (TP) but does not model data parallelism (DP) or expert parallelism (EP) scheduling overhead. Quantized weight precision is auto-detected from HuggingFace `quantization_config` (GPTQ, AWQ, FP8, compressed-tensors), model name conventions (e.g., `w4a16`, `FP8`), or `torch_dtype` fallback, and is used for weight bandwidth and KV capacity calculations. KV cache and activation memory continue to use the compute dtype (`BytesPerParam` from `torch_dtype`).

## 1. Why Roofline?

The Blackbox optimization approach outlined in [Configuration: Coefficient Calibration](../reference/configuration.md#coefficient-calibration) requires training over ground-truth workload metrics obtained by running live vLLM instances over GPUs. In practice, collecting this ground-truth data and training the GPU latency model for each LLM can take several hours. For a faster approximation of GPU latencies without actually ever running vLLM, we recommend the roofline approach. This technique only requires the Huggingface LLM config (`config.json`) and hardware specifications (e.g. GPU Peak TFLOPS/Peak BW from NVIDIA datasheets) in `hardware_config.json`. It allows BLIS to generalize across diverse LLMs, TP values and workloads, while preserving TTFT/ITL/E2E accuracy.

## 2. Core Methodology: The Roofline Model
The simulator predicts execution time by identifying whether a phase (prefill/decode) is **Compute-Bound** or **Memory-Bound**. 

For each phase:

$$\text{Phase Time} = \max\left( \frac{\text{Total FLOPS}}{\text{Peak Performance}}, \frac{\text{Total Bytes Transferred}}{\text{Memory Bandwidth}} \right)$$

As a result, overall step time is given by:

$$\text{Step Time} = \text{Prefill Phase Time} + \text{Decode Phase Time}$$

---

## 3. Calculation Phases

### A. FLOPs Calculation (`calculateTransformerFlops`)
We track two types of operations:
* **GEMM Ops:** Matrix multiplications for QKV projections, Attention Output, and MLP (SwiGLU) layers. Includes $QK^T$ and $AV$ operations.
* **Vector Ops:** Non-tensor core operations like Softmax, RoPE (rotary embeddings), and normalization.

### B. Memory Access (`calculateMemoryAccessBytes`)
We calculate the data movement between HBM (High Bandwidth Memory) and the processor:
* **Weights:** Static model parameters loaded once per layer.
* **KV Cache Growth:** Writing new Key/Value tokens to memory.
* **KV Cache Access:** Reading the history (past tokens) for attention.

---

## 3. Execution Pipeline
The final step time is the sum of independent phases and overheads:

1.  **Prefill Phase:** Calculated for the initial prompt processing chunk.
2.  **Decode Phase:** Calculated for generating new tokens (usually memory-bound).
3.  **Communication Overhead:** If using Tensor Parallelism ($TP > 1$), adds All-Reduce latency per layer.
4.  **Hardware Overheads:** Static kernel launch times and layer-by-layer overhead constants.

---

## 4. Key Performance Variables
* **MFU (Model Flops Utilization):** Scaled TFLOPs efficiency factors for Prefill vs. Decode.
* **TP Factor:** Divides compute and memory load across multiple GPUs.
* **Bandwidth Efficiency:** Real-world effective bandwidth versus theoretical peak bandwidth.

## 5. Onboarding a new LLM/GPU

### Automatic (recommended): `--latency-model roofline`

The simplest way to run roofline mode is with `--latency-model roofline`, which auto-resolves the model config:

```bash
./blis run --model qwen/qwen3-14b --latency-model roofline --hardware H100 --tp 1
```

The flag automatically:
1. Checks `model_configs/` for an existing `config.json` (previously fetched)
2. Fetches from HuggingFace on miss and writes into `model_configs/` (supports `HF_TOKEN` for gated models)

For models not in `defaults.yaml`, add an `hf_repo` entry mapping the BLIS model name to the case-sensitive HuggingFace repo path.

### Manual: explicit config paths

Alternatively, download the `config.json` manually:

* Download the `config.json` for the LLM of your choice into `model_configs/`. [This](https://huggingface.co/Qwen/Qwen3-14B/blob/main/config.json) is an example config.json for `Qwen/Qwen3-14B`. The recommended file structure is `model_configs/qwen3-14b/config.json`.

### Adding a new GPU

* Refer to NVIDIA datasheets for GPU specs (for example, [datasheet for H100](https://www.nvidia.com/en-us/data-center/h100/)) and add an entry to `hardware_config.json`:

```json
{
    "<GPU_name>": {
        "TFlopsPeak":  989.5,
        "BwPeakTBs":   3.35,
        "mfuPrefill":  0.45,
        "mfuDecode":   0.30,
        "MemoryGiB":   80.0
    }
}
```

| Field | Description |
|-------|-------------|
| `TFlopsPeak` | Peak BF16 TFLOPS from GPU datasheet |
| `BwPeakTBs` | Peak HBM bandwidth in TB/s from GPU datasheet |
| `mfuPrefill` | Model FLOPS Utilization for prefill phase (compute-bound) |
| `mfuDecode` | Model FLOPS Utilization for decode phase (memory-bound) |
| `MemoryGiB` | GPU memory capacity in GiB. Used by `CalculateKVBlocks` to auto-derive `--total-kv-blocks` when roofline or trained-physics mode is active and the flag is not explicitly set. |

> Note: The Peak TFLOPS and BW for a given GPU family might vary by GPU connectivity (e.g. SXM vs PCIe). We recommend a separate entry for each GPU connectivity type - e.g. A100-SXM, A100-PCIe etc in `hardware_config.json`.
