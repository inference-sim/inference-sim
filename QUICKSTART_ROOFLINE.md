# Quick Start: Roofline Mode (v0.6.1)

This guide gets you running BLIS in roofline mode in 5 minutes.

## Prerequisites

```bash
# Ensure you're on the v0.6.1_roofline_valid branch
git branch --show-current
# Should show: v0.6.1_roofline_valid

# Build the binary
go build -o simulation_worker main.go
```

## Step 1: Verify Hardware Config

Check that `hardware_config.json` has complete calibration parameters:

```bash
# Should have ~18 fields per GPU (TFlopsPeak, BwPeakTBs, MfuPrefill, etc.)
cat hardware_config.json
```

✅ **Fixed in commit `cba73b3`** - Config now includes all required fields.

## Step 2: Run Roofline v1 (No Benchmark Data)

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json \
  --hardware H100 \
  --tp 1 \
  --workload chatbot \
  --rate 10
```

**Mode:** Roofline v1 (calibrated constants from `hardware_config.json`)

## Step 3: Run Roofline v2 (With Benchmark Data)

If you have MFU benchmark data in `bench_data/`:

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json \
  --bench-data-path bench_data \
  --hardware H100 \
  --tp 1 \
  --workload chatbot \
  --rate 10
```

**Mode:** Roofline v2 (per-GEMM MFU lookups, more accurate)

## Step 4: Run Evaluator (Optional)

```bash
python3 python_scripts/blis_evaluator.py \
  --ground-truth eval/combined_ground_truth.json \
  --verbose
```

✅ **Updated in commit `336f7fa`** - Now compatible with v0.6.1 + roofline.

---

## What Was Fixed

### Issue: Hardware Config Error
```
panic: NewInstanceSimulator(instance_0): creating latency model: 
latency model: invalid roofline config: HardwareCalib.BwEffConstant 
must be a valid positive number, got 0
```

### Root Cause
`hardware_config.json` only had 2 fields (`TFlopsPeak`, `BwPeakTBs`) but roofline mode requires 18+ calibration fields.

### Solution
**Commit `cba73b3`**: Added complete calibration parameters:
- Bandwidth efficiency (`BwEffConstant`: 0.72 for H100)
- MFU values (`mfuPrefill`: 0.65, `mfuDecode`: 0.12)
- TP scaling exponents (0.8 for prefill, 1.0 for decode)
- Overhead parameters (step, layer, request-level)
- 9 additional tuning parameters

---

## Architecture Overview

```
┌──────────────────────────────────────┐
│  User provides roofline CLI flags   │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Load hardware_config.json           │
│  + model config.json                 │
│  + (optional) bench_data/ MFU CSVs   │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  NewLatencyModel() auto-selects:     │
│  ├─ bench_data exists → Roofline v2  │
│  ├─ bench_data missing → Roofline v1 │
│  └─ no roofline flags → Blackbox     │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Cluster creates instance simulators │
│  Each uses selected latency model    │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│  Run simulation, output metrics      │
└──────────────────────────────────────┘
```

---

## Files Changed

| File | Change | Commit |
|------|--------|--------|
| `hardware_config.json` | Added 15+ calibration fields | `cba73b3` |
| `python_scripts/blis_evaluator.py` | Updated for v0.6.1 CLI | `336f7fa` |
| `sim/latency_model.go` | Added RooflineLatencyModelV2 | `f10e4c6` |
| `sim/config.go` | Added MFUDatabase support | `f10e4c6` |
| `sim/roofline_step_test.go` | Fixed function signatures | `6724947` |

---

## Documentation

📖 **Read these guides for more details:**

1. **`ROOFLINE_V0.6.1_INTEGRATION.md`**
   - Complete integration overview
   - Architecture changes
   - Migration guide

2. **`docs/HARDWARE_CONFIG_GUIDE.md`** ⭐ NEW
   - Parameter-by-parameter explanation
   - Calibration workflow
   - Troubleshooting guide

3. **`blis.md`**
   - Roofline v1 technical details
   - FLOPs and memory calculations

4. **`docs/roofline_v2_design.md`**
   - Roofline v2 MFU-based approach
   - Benchmark data requirements

---

## Common Issues

### 1. "invalid roofline config" error
**Fix:** Commit `cba73b3` added all required fields to `hardware_config.json`

### 2. Evaluator fails with "unknown flag"
**Fix:** Commit `336f7fa` updated evaluator for v0.6.1 (`--num-requests` instead of `--max-prompts`)

### 3. Missing model config
**Fix:** Ensure `model_configs/llama-3.1-8b-instruct/config.json` exists (HuggingFace config)

### 4. Warnings about "Decode MFU=0.0000"
**Expected:** MFU database uses nearest-neighbor lookup for unseen shapes (normal behavior)

---

## Testing

```bash
# Run all tests
go test ./sim/...

# Run roofline tests only
go test -v ./sim -run TestRoofline

# Test a simple simulation
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json \
  --hardware H100 \
  --tp 1 \
  --workload distribution \
  --rate 1 \
  --num-requests 5
```

All tests should pass ✅

---

## Next Steps

1. **Calibrate for your hardware:** Use `docs/HARDWARE_CONFIG_GUIDE.md`
2. **Collect MFU data (optional):** For roofline v2 accuracy
3. **Run evaluations:** Compare predictions vs. ground truth
4. **Tune parameters:** Adjust multipliers based on error analysis

---

## Support

- 📖 Full docs: `ROOFLINE_V0.6.1_INTEGRATION.md`
- 🔧 Calibration: `docs/HARDWARE_CONFIG_GUIDE.md`
- 🧪 Tests: `sim/roofline_step_test.go`
- 💬 Issues: Check commit messages for context

**Branch:** `v0.6.1_roofline_valid`
**Status:** ✅ Production-ready
