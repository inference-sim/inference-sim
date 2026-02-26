# Roofline Step Time Estimation Logic

This document describes the analytical approach used to estimate the GPU latency for a single inference step using a roofline model. This requires no training, and works off-the-shelf for any Huggingface LLM whose `config.json` is saved under `model_configs/`, except Mixture-of-Expert (MoE) models currently.


## 1. Why Roofline?

The Blackbox optimization approach outlined in [Configuration: Coefficient Calibration](configuration.md#coefficient-calibration) requires training over ground-truth workload metrics obtained by running live vLLM instances over GPUs. In practice, collecting this ground-truth data and training the GPU latency model for each LLM can take several hours. For a faster approximation of GPU latencies without actually ever running vLLM, we recommend the roofline approach. This technique only requires the Huggingface LLM config (`config.json`) and hardware specifications (e.g. GPU Peak TFLOPS/Peak BW from NVIDIA datasheets) in `hardware_config.json`. It allows BLIS to generalize across diverse LLMs, TP values and workloads, while preserving TTFT/ITL/E2E accuracy.

## 2. Core Methodology: The Roofline Model (v2)

The simulator predicts execution time by identifying whether a phase (prefill/decode) is **Compute-Bound** or **Memory-Bound**.

**v2** uses MFU (Model FLOPs Utilization) values looked up from benchmark data (`bench_data/`) instead of static constants. MFU varies by operation type, batch size, sequence length, and attention configuration, providing more accurate estimates across diverse workloads.

For each phase:

$$\text{Phase Time} = \max\left( \frac{\text{Total FLOPS}}{\text{Peak Performance} \times \text{MFU}}, \frac{\text{Total Bytes Transferred}}{\text{Effective Bandwidth}} \right)$$

For mixed batches (both prefill and decode requests), phases are combined via max (modeling chunked-prefill overlap):

$$\text{Step Time} = \max(\text{Prefill Phase Time}, \text{Decode Phase Time}) + \text{CPU Overhead}$$

For single-phase batches:

$$\text{Step Time} = \text{Phase Time} + \text{CPU Overhead}$$

---

## 3. MFU Database (`bench_data/`)

The v2 roofline model uses empirically measured MFU values from GPU micro-benchmarks:

- **`bench_data/mha/prefill/<gpu>/`**: Attention prefill MFU by attention config and sequence length
- **`bench_data/mha/decode/<gpu>/`**: Attention decode MFU by attention config, batch size, KV length, and TP
- **`bench_data/gemm/<gpu>/data.csv`**: GEMM MFU by matrix dimensions (M, K, N)

MFU values are interpolated between grid points:
- Prefill: linear interpolation on sequence length
- Decode: bilinear interpolation on (batch_size, kv_len) grid
- GEMM: linear interpolation on M dimension after matching (K, N)

If the model's attention configuration doesn't match any benchmark config, the nearest available config is selected using Euclidean distance.

---

## 4. Calculation Phases

### A. GEMM Projections (`computeTransformerGEMMTimes`)

For each GEMM projection (Q, K, V, O, Gate, Up, Down), time is:

$$\text{GEMM Time} = \max\left( \frac{2 \times M \times K \times N}{\text{Peak FLOPS} \times \text{MFU}(M,K,N)}, \frac{K \times N \times \text{bytes\_per\_param}}{\text{Effective BW}} \right)$$

The memory-bandwidth floor ensures GEMM time never drops below the weight-load time, which dominates at small batch sizes.

### B. Attention Core (`calculateAttentionCoreFLOPs`)

FLOPs for QK^T and AV matmuls, with MFU looked up per sequence length (prefill) or per batch_size/kv_len (decode).

### C. Memory Access (`calculateMemoryAccessBytes`)

Data movement between HBM and the processor:
* **Weights:** Static model parameters loaded once per step.
* **KV Cache Growth:** Writing new Key/Value tokens to memory.
* **KV Cache Access:** Reading cached tokens for attention.
* **Activations:** Intermediate activation tensors.

---

## 5. Key Performance Variables

* **MFU (Model FLOPs Utilization):** Looked up from benchmark data per operation type, varying by batch size, sequence length, and matrix dimensions.
* **TP Factor:** Divides compute and memory load across multiple GPUs. Applied as `1/tp` scaling to both compute and memory bandwidth terms.
* **Bandwidth Efficiency Factor (`bwEfficiencyFactor`):** Scales peak HBM bandwidth to reflect real-world effective bandwidth (typically 0.75-0.85).
* **Per-Layer CPU Overhead (`perLayerOverhead`):** Microseconds of CPU scheduling overhead per transformer layer, scaled by TP.

## 6. Onboarding a new LLM/GPU

### Automatic (recommended): `--roofline` flag

The simplest way to run roofline mode is with the `--roofline` flag, which auto-resolves the model config:

```bash
./simulation_worker run --model meta-llama/llama-3.1-8b-instruct --roofline --hardware H100 --tp 1
```

The flag automatically:
1. Checks `model_configs/` for an existing `config.json` (previously fetched)
2. Fetches from HuggingFace on miss and writes into `model_configs/` (supports `HF_TOKEN` for gated models)
3. Loads MFU benchmark data from `bench_data/`
4. Loads `total-kv-blocks` from `defaults.yaml` (if not set explicitly)

**Alpha coefficients:** Roofline mode uses **zero alpha coefficients** by default. The roofline model is a pure analytical estimator — alpha coefficients were trained jointly with beta coefficients for the blackbox regression model, and mixing trained alpha with analytical step-time estimation produces inconsistent results. To override this, pass `--alpha-coeffs` explicitly on the command line.

For models not in `defaults.yaml`, add an `hf_repo` entry mapping the BLIS model name to the case-sensitive HuggingFace repo path.

### Manual: explicit config paths

Alternatively, download the `config.json` manually:

* Download the `config.json` for the LLM of your choice into `model_configs/`. [This](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json) is an example config.json for `meta-llama/Llama-3.3-70B-Instruct`. The recommended file structure is `model_configs/llama-3.1-8b-instruct/config.json`.

### Adding a new GPU

* Refer to NVIDIA datasheets for GPU specs (for example, [datasheet for H100](https://www.nvidia.com/en-us/data-center/h100/)) and add an entry to `hardware_config.json`:

```json
{
    "<GPU_name>": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "bwEfficiencyFactor": 0.82,
        "perLayerOverhead": 100
    }
}
```

| Field | Description |
|-------|-------------|
| `TFlopsPeak` | Peak BF16 TFLOPS from GPU datasheet |
| `BwPeakTBs` | Peak HBM bandwidth in TB/s from GPU datasheet |
| `bwEfficiencyFactor` | Fraction of peak BW achieved in practice (0-1, optional, 0 = use raw peak) |
| `perLayerOverhead` | CPU scheduling overhead per transformer layer in microseconds |

> Note: The Peak TFLOPS and BW for a given GPU family might vary by GPU connectivity (e.g. SXM vs PCIe). We recommend a separate entry for each GPU connectivity type - e.g. A100-SXM, A100-PCIe etc in `hardware_config.json`.
