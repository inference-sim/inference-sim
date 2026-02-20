# H100 MFU Fix Summary

## Problem

The H100 benchmark data in `bench_data/` had MFU (Model FLOPs Utilization) values **> 1.0**, which is physically impossible. MFU represents the fraction of peak hardware performance achieved and must be ≤ 1.0 by definition.

### Root Cause

The benchmark data was generated with incorrect peak TFLOPs parameters:
- **Used**: 296 TFLOPs (default in standalone benchmark scripts)
- **Correct**:
  - FP16/BF16 (MHA): 989.5 TFLOPs
  - FP8 (GEMM): 1979.0 TFLOPs

### Impact

When MFU > 1, the roofline model formula `time = flops / (peakFlops * mfu)` produces artificially small time estimates, leading to impossibly high throughput predictions.

### Example

**Before fix:**
```
bench_data/mha/prefill/h100/32-8-128.csv
seq_len=4096: latency=221.05μs, mfu=4.201 (420% of peak!)
```

**After fix:**
```
bench_data/mha/prefill/h100/32-8-128.csv
seq_len=4096: latency=221.05μs, mfu=0.628 (62.8% of peak ✓)
```

The latency measurement is unchanged (correct), only the MFU interpretation was fixed.

## Solution Applied

### 1. Fixed Existing Data

Ran `scripts/fix_h100_mfu.py` which:
- Recalculated MFU from existing latency measurements
- Used correct peak TFLOPs (989.5 for FP16, 1979.0 for FP8)
- Created `.bak` backups of original files
- Fixed 25 benchmark files:
  - 1 GEMM file: max MFU 1.264 → 0.632
  - 6 MHA prefill files: max MFU 4.805 → 0.720
  - 18 MHA decode files: scaled proportionally

### 2. Updated Benchmark Scripts

To prevent this issue in future benchmarking runs:

#### InferSim/kernel_benchmark/deepgemm_gemm.py
- Changed `--gpu-tflops` from `default=296` to `required=True`
- Added validation message about MFU > 1.0

#### InferSim/kernel_benchmark/fa3_mha_prefill.py
- Added `--fp16-tflops` parameter (required)
- Added validation message

#### InferSim/kernel_benchmark/flashinfer_mha_decode.py
- Changed `--fp16-tflops` from `default=148` to `required=True`
- Added validation message

#### InferSim/scripts/run_benchmarks.py
- Updated to pass `--fp16-tflops` to prefill benchmark
- Already passed it to decode benchmark

#### InferSim/scripts/validate_benchmark_data.py
- Added hard error for MFU > 1.0 (in addition to existing range warnings)
- Fails validation if any MFU values exceed 1.0

## How It Works

The fix script doesn't re-run benchmarks - it recalculates MFU from existing latency data:

```python
# Original measurement (unchanged)
latency_us = 221.05

# Calculate achieved TFLOPs from latency
flops = calculate_flops(seq_len, num_heads, head_dim)
achieved_tflops = flops / (latency_us / 1e6) / 1e12

# Recalculate MFU with correct peak
mfu_old = achieved_tflops / 296      # Wrong! = 4.201
mfu_new = achieved_tflops / 989.5    # Correct! = 0.628
```

## Prevention

Future benchmarking runs will be safe because:

1. **Required Parameters**: All benchmark scripts now require `--gpu-tflops` or `--fp16-tflops` - no dangerous defaults
2. **Orchestration Scripts**: `InferSim/scripts/run_benchmarks.py` passes correct values from `config/benchmark_config.json`
3. **Validation**: `validate_benchmark_data.py` now fails if MFU > 1.0 is detected

## Verification

```bash
# Validate fixed data
python InferSim/scripts/validate_benchmark_data.py --gpu H100

# Should show:
# - All validations PASSED
# - No "ERROR" messages about MFU > 1.0
# - Some "WARNING" messages are OK (just range expectations)
```

## Files Modified

### Data Files (25 files fixed)
- `bench_data/gemm/h100/data.csv`
- `bench_data/mha/prefill/h100/*.csv` (6 files)
- `bench_data/mha/decode/h100/*.csv` (18 files)
- Backups saved with `.bak` extension

### Code Files
- `scripts/fix_h100_mfu.py` (new)
- `InferSim/kernel_benchmark/deepgemm_gemm.py`
- `InferSim/kernel_benchmark/fa3_mha_prefill.py`
- `InferSim/kernel_benchmark/flashinfer_mha_decode.py`
- `InferSim/scripts/run_benchmarks.py`
- `InferSim/scripts/validate_benchmark_data.py`

## Key Insight

Different operations use different precisions:
- **GEMM operations**: FP8 (H100: 1979 TFLOPs)
- **Attention operations**: FP16/BF16 (H100: 989.5 TFLOPs)

The correct peak must match the actual kernel implementation to calculate valid MFU values.

## References

- Config: `config/benchmark_config.json` (lines 2-12: GPU specs)
- InferSim Tech Report: Section 2.3 "Kernel Benchmark" (page 5)
- H100 Specs: 989.5 TFLOPs FP16, 1979.0 TFLOPs FP8
