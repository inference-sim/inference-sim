# Roofline Step Time Estimation Logic

This document describes the analytical approach used to estimate the GPU latency for a single inference step using a roofline model. This requires no training, and works off-the-shelf for any Huggingface LLM whose `config.json` is saved under `model_configs/`, except Mixture-of-Expert (MoE) models currently.


## 1. Why Roofline?

The Blackbox optimization approach outlined in [approach.md](./approach.md) requires training over ground-truth workload metrics obtained by running live vLLM instances over GPUs. In practice, collecting this ground-truth data and training the GPU latency model for each LLM can take several hours. For a faster approximation of GPU latencies without actually ever running vLLM, we recommend the roofline approach. This technique only requires the Huggingface LLM config (`config.json`) and hardware specifications (e.g. GPU Peak TFLOPS/Peak BW from NVIDIA datasheets) in `hardware_config.json`. It allows BLIS to generalize across diverse LLMs, TP values and workloads, while preserving TTFT/ITL/E2E accuracy.

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

* Download the `config.json` for the LLM of your choice into `model_configs/`. [This](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct/blob/main/config.json) is an example config.json for `meta-llama/Llama-3.3-70B-Instruct`. The recommended file structure is `model_configs/llama-3.1-8b-instruct/config.json`.
* Refer to NVIDIA datasheets for GPU specs (for example, [datasheet for H100](https://www.nvidia.com/en-us/data-center/h100/)) and add an entry to `hardware_config.json` as follows:

```
<GPU_name>: {
		"TFlopsEff":        <Peak TFLOPS from datasheet>,      
		"BwEffTBs":         <Peak BW from datasheet>,
        "BwEffConstant":    0.72,
		"TOverheadMicros":  500.0,
		"perLayerOverhead": 20.0,
		"mfuPrefill":       0.65,
		"mfuDecode":        0.12,
		"allReduceLatency": 20.0
	}
```

> Note: The Peak TFLOPS and BW for a given GPU family might vary by GPU connectivity (e.g. SXM vs PCIe). We recommend a separate entry for each GPU connectivity type - e.g. A100-SXM, A100-PCIe etc in `hardware_config.json`. 