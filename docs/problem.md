# Problem Statement: Bridging the Accuracy Gap in LLM Inference Performance Prediction

## Context

Large Language Model (LLM) inference simulation is critical for:
- **Model-system co-design**: Predicting performance before training/deployment
- **Hardware planning**: Determining GPU requirements and parallelization strategies
- **Performance optimization**: Identifying compute vs memory bottlenecks

Current state-of-the-art simulators like **InferSim** (Alibaba, 2025) achieve 4-15% prediction error by using **MFU (Model FLOPs Utilization)** lookup tables from benchmarked kernels rather than theoretical peak performance. We want to modify the current roofline implementation in (`sim/roofline_step.go`) to accurately estimate vLLM step times using InferSim's benchmarked MFU values.

Here are some relevant sources:
* InferSim codebase (https://github.com/inference-sim/InferSim/tree/main)
* InferSim technical report (`docs/infersim_tech_report (1).pdf`)
* Benchmarked H100 InferSim coefficients (`bench_data`)
* vLLM CUDA codebase (https://github.com/vllm-project/vllm/tree/main/csrc)

## Current State: Roofline V1 Limitations

The existing Go implementation (`sim/roofline_step.go`) uses a **calibrated roofline model** with the following limitations:

### 1. **Fixed MFU Values**
```go
adjustedPrefillMFU := hwConfig.MfuPrefill * hwConfig.MfuPrefillMultiplier
adjustedDecodeMFU := hwConfig.MfuDecode * hwConfig.MfuDecodeMultiplier
```
- Single MFU value for all prefill operations (regardless of sequence length)
- Single MFU value for all decode operations (regardless of batch size or KV length)
- Does not account for shape-dependent kernel efficiency

### 2. **Empirical Calibration Factors**
```go
prefillEffBW := effBW * hwConfig.PrefillBwFactor
decodeEffBW := effBW * hwConfig.DecodeBwFactor
effectiveTpPrefill := math.Pow(tpFactor, hwConfig.TpScalingExponent)
```
- 15+ tunable parameters per hardware configuration
- Requires extensive vLLM profiling to calibrate
- Not generalizable across models or workload patterns

### 3. **Aggregate Computation**
- Lumps all GEMM operations into single FLOPs calculation
- Cannot distinguish between Q/K/V projections, MLP layers with different shapes
- Misses per-operation efficiency variations

### 4. **No Attention-Specific Modeling**
```go
decodeComputeS = (dGemmFlops / (peakFlops * adjustedDecodeMFU)) + (dVectorFlops / vectorPeak)
```
- Attention core operations use `vectorPeak` (artificial 10% of peak FLOPs)
- Doesn't reflect actual FlashAttention/FlashInfer MFU characteristics
- Ignores sequence length and batch size impact on attention efficiency