# Hardware Config Calibration Guide

This guide explains the calibration parameters in `hardware_config.json` for roofline mode.

## Quick Reference

```json
{
  "H100": {
    "TFlopsPeak": 989.5,              // Peak tensor TFLOPS
    "BwPeakTBs": 3.35,                // Peak HBM bandwidth (TB/s)
    "BwEffConstant": 0.72,            // Effective bandwidth efficiency
    "mfuPrefill": 0.65,               // Model FLOPs Utilization - prefill
    "mfuDecode": 0.12,                // Model FLOPs Utilization - decode
    "tpScalingExponent": 0.8,         // TP scaling (prefill, sublinear)
    "decodeTpScalingExponent": 1.0,   // TP scaling (decode, linear)
    "TOverheadMicros": 50.0,          // Per-step overhead (μs)
    "prefillOverheadMicros": 100.0,   // Per-prefill-request overhead (μs)
    "mixedPrefillOverheadMicros": 50.0, // Prefill overhead in mixed batch (μs)
    "vectorPeakFraction": 0.1,        // Non-tensor-core efficiency
    "allReduceLatency": 20.0,         // All-reduce per layer (μs)
    // ... see full config below
  }
}
```

---

## Core Parameters

### `TFlopsPeak` (required)
**Peak tensor core TFLOPS**

- H100 SXM5: **989.5** TFLOPS (FP16/BF16)
- A100 SXM: **312** TFLOPS (FP16/BF16)

Source: Vendor specifications for tensor core performance.

### `BwPeakTBs` (required)
**Peak HBM memory bandwidth (TB/s)**

- H100 SXM5: **3.35** TB/s
- A100 SXM: **2.039** TB/s

Source: Vendor specifications for HBM3/HBM2e bandwidth.

---

## Efficiency Parameters

### `BwEffConstant` (required)
**Effective bandwidth as fraction of peak**

- **Range:** 0.6 - 0.8
- **H100:** 0.72 (72% of 3.35 TB/s = 2.41 TB/s)
- **A100:** 0.70 (70% of 2.039 TB/s = 1.43 TB/s)

**Why < 1.0?** Memory access patterns, cache effects, and row buffer locality reduce effective bandwidth below theoretical peak.

**How to calibrate:**
1. Run memory bandwidth benchmark (e.g., STREAM)
2. Divide measured bandwidth by `BwPeakTBs`
3. Use result as `BwEffConstant`

### `mfuPrefill` (required)
**Model FLOPs Utilization during prefill phase**

- **Range:** 0.5 - 0.7
- **H100:** 0.65 (65% of peak)
- **A100:** 0.60 (60% of peak)

**Meaning:** Fraction of theoretical peak FLOPS achieved during prefill (compute-bound workload).

**Factors affecting MFU:**
- Kernel launch overhead
- Memory bottlenecks during weight loading
- Suboptimal tensor shapes
- Framework overhead

**How to calibrate:**
1. Run prefill-only benchmark
2. Measure achieved TFLOPS: `achieved = (model_flops) / (measured_time_s * 1e12)`
3. Calculate MFU: `mfuPrefill = achieved / TFlopsPeak`

### `mfuDecode` (required)
**Model FLOPs Utilization during decode phase**

- **Range:** 0.08 - 0.15
- **H100:** 0.12 (12% of peak)
- **A100:** 0.10 (10% of peak)

**Meaning:** Fraction of theoretical peak FLOPS during decode (memory-bound workload).

**Why so low?** Decode is heavily memory-bound:
- Small batch sizes (often 1 token per request)
- Memory bandwidth saturated by weight loading
- Low arithmetic intensity (~1-2 FLOPs/byte)

**How to calibrate:**
1. Run decode-only benchmark (single token generation)
2. Measure achieved TFLOPS
3. Calculate MFU: `mfuDecode = achieved / TFlopsPeak`

---

## Tensor Parallelism Scaling

### `tpScalingExponent` (optional, default: 0.8)
**TP scaling for prefill (compute-bound)**

- **Range:** 0.7 - 0.9
- **Default:** 0.8 (sublinear scaling)

**Meaning:** Effective TP speedup = `TP^tpScalingExponent`

**Examples:**
- TP=2 with exp=0.8 → 1.74x speedup (not 2x)
- TP=4 with exp=0.8 → 3.03x speedup (not 4x)

**Why sublinear?** Communication overhead (all-reduce) reduces efficiency at higher TP.

### `decodeTpScalingExponent` (optional, default: 1.0)
**TP scaling for decode (memory-bound)**

- **Range:** 0.95 - 1.0
- **Default:** 1.0 (linear scaling)

**Meaning:** Decode scales more linearly because it's memory-bound, and each GPU has 1/TP of weights to load.

---

## Overhead Parameters

### `TOverheadMicros` (optional, default: 50)
**Per-step fixed overhead (microseconds)**

- **H100:** 50 μs
- **A100:** 60 μs

**Includes:** Kernel launch, scheduler overhead, CUDA synchronization.

### `prefillOverheadMicros` (optional, default: 100)
**Per-prefill-request overhead in pure prefill steps**

- **H100:** 100 μs
- **A100:** 120 μs

**Includes:** KV cache allocation, memory copying, block assignment.

### `mixedPrefillOverheadMicros` (optional, default: 50)
**Per-prefill-request overhead in mixed batches**

- **H100:** 50 μs (lower than pure prefill)
- **A100:** 60 μs

**Why lower?** Some overhead amortized across decode requests in same batch.

### `allReduceLatency` (optional, default: 20)
**All-reduce latency per layer (microseconds)**

- **H100 NVLink:** 10-20 μs
- **A100 NVLink:** 20-30 μs
- **PCIe:** 50-100 μs (much higher)

**Applies when:** TP > 1

---

## Advanced Parameters

### `mfuPrefillMultiplier` (optional, default: 1.0)
**Additional MFU adjustment factor for prefill**

- **Range:** 0.8 - 1.2
- **Use:** Fine-tune prefill predictions after setting base `mfuPrefill`

**Example:** If predictions are 10% too fast, set to 0.9.

### `mfuDecodeMultiplier` (optional, default: 1.0)
**Additional MFU adjustment factor for decode**

- **Range:** 0.8 - 1.2
- **Use:** Fine-tune decode predictions after setting base `mfuDecode`

### `prefillBwFactor` (optional, default: 1.0)
**Bandwidth scaling for prefill**

- **Range:** 0.8 - 1.0
- **Use:** Reduce effective bandwidth if prefill has memory contention

**Example:** Set to 0.9 if prefill has 10% BW reduction due to KV cache writes.

### `decodeBwFactor` (optional, default: 1.0)
**Bandwidth scaling for decode**

- **Range:** 0.7 - 1.0
- **Use:** Reduce effective bandwidth for scattered KV cache reads

**Example:** Set to 0.8 if decode has scattered memory access penalties.

### `vectorPeakFraction` (optional, default: 0.1)
**Non-tensor-core operation efficiency**

- **Range:** 0.05 - 0.15
- **Default:** 0.1 (10% of tensor core peak)

**Applies to:** Softmax, RoPE, layer norm, activation functions.

**Why 10%?** These operations run on scalar/vector units, not tensor cores.

### `perLayerOverhead` (optional, default: 5)
**Per-transformer-layer overhead (microseconds)**

- **H100:** 5 μs
- **A100:** 6 μs

**Includes:** Layer-specific kernel launches, residual connections.

---

## Calibration Workflow

### Step 1: Measure Hardware Specs
```bash
# Use nvidia-smi or vendor docs
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm --format=csv
```

Set `TFlopsPeak` and `BwPeakTBs` from specifications.

### Step 2: Run Memory Benchmark
```bash
# Example: STREAM benchmark
# Measure effective bandwidth
```

Calculate `BwEffConstant = measured_bw / BwPeakTBs`

### Step 3: Profile Prefill
```bash
# Run large-batch prefill
./simulation_worker run --workload distribution \
  --prompt-tokens 2048 --output-tokens 1 \
  --rate 100 --num-requests 50
```

Measure TFLOPS, calculate `mfuPrefill`.

### Step 4: Profile Decode
```bash
# Run single-token decode
./simulation_worker run --workload distribution \
  --prompt-tokens 50 --output-tokens 500 \
  --rate 1 --num-requests 10
```

Measure TFLOPS, calculate `mfuDecode`.

### Step 5: Tune Multipliers
Compare predictions vs. ground truth:
- Too fast → reduce multipliers (< 1.0)
- Too slow → increase multipliers (> 1.0)

---

## Example Configs

### H100 SXM5 (Optimized)
```json
{
  "H100": {
    "TFlopsPeak": 989.5,
    "BwPeakTBs": 3.35,
    "BwEffConstant": 0.72,
    "TOverheadMicros": 50.0,
    "perLayerOverhead": 5.0,
    "mfuPrefill": 0.65,
    "mfuDecode": 0.12,
    "allReduceLatency": 20.0,
    "tpScalingExponent": 0.8,
    "decodeTpScalingExponent": 1.0,
    "mfuPrefillMultiplier": 1.0,
    "mfuDecodeMultiplier": 1.0,
    "prefillBwFactor": 1.0,
    "decodeBwFactor": 1.0,
    "vectorPeakFraction": 0.1,
    "prefillOverheadMicros": 100.0,
    "mixedPrefillOverheadMicros": 50.0
  }
}
```

### A100 SXM (Conservative)
```json
{
  "A100-SXM": {
    "TFlopsPeak": 312.0,
    "BwPeakTBs": 2.039,
    "BwEffConstant": 0.70,
    "TOverheadMicros": 60.0,
    "perLayerOverhead": 6.0,
    "mfuPrefill": 0.60,
    "mfuDecode": 0.10,
    "allReduceLatency": 25.0,
    "tpScalingExponent": 0.75,
    "decodeTpScalingExponent": 1.0,
    "mfuPrefillMultiplier": 1.0,
    "mfuDecodeMultiplier": 1.0,
    "prefillBwFactor": 1.0,
    "decodeBwFactor": 1.0,
    "vectorPeakFraction": 0.1,
    "prefillOverheadMicros": 120.0,
    "mixedPrefillOverheadMicros": 60.0
  }
}
```

---

## Troubleshooting

### Error: "BwEffConstant must be a valid positive number, got 0"
**Cause:** Missing or zero calibration fields in `hardware_config.json`

**Fix:** Add all required fields (see examples above)

### Predictions too fast (< ground truth)
**Possible causes:**
- MFU values too high
- Overhead values too low
- BwEffConstant too high

**Fix:** Reduce `mfuPrefillMultiplier` or `mfuDecodeMultiplier` by 10-20%

### Predictions too slow (> ground truth)
**Possible causes:**
- MFU values too low
- Overhead values too high
- BwEffConstant too low

**Fix:** Increase multipliers or check if hardware specs are correct

### Decode much slower than expected
**Check:**
- Is `mfuDecode` too low? (should be 0.08-0.15)
- Is `decodeBwFactor` < 1.0? (reduces effective bandwidth)
- Is decode truly memory-bound? (expected for small batch sizes)

---

## Further Reading

- `ROOFLINE_V0.6.1_INTEGRATION.md` - Integration overview
- `blis.md` - Roofline model technical details
- `docs/roofline_v2_design.md` - Roofline v2 MFU-based approach
- `sim/roofline_step.go` - Implementation source code
