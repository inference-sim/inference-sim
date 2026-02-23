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

## Step 1: Check Hardware Config

**One config file for everything:**

```bash
cat hardware_config.json
```

This config has all fields and works for **both** roofline v1 and v2:
- Roofline v2: Only uses `TFlopsPeak`, `BwPeakTBs`, + optional overheads
- Roofline v1: Uses all 18 calibration fields

✅ **One config, both modes** - no confusion!

## Step 2: Run Roofline v2 (Recommended)

**With MFU benchmark data** (more accurate):

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

**Mode:** Roofline v2 (per-GEMM MFU lookups from database)

## Step 3: Run Roofline v1 (Fallback)

**Without benchmark data** (uses calibrated constants from config):

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

**Mode:** Roofline v1 (uses MFU values from config)

## Step 4: Run Evaluator (Optional)

```bash
python3 python_scripts/blis_evaluator.py \
  --ground-truth eval/combined_ground_truth.json \
  --verbose
```

✅ **Updated in commit `336f7fa`** - Now compatible with v0.6.1 + roofline.

---

## What Was Fixed

### Original Issue: Missing Config Fields
```
panic: NewInstanceSimulator(instance_0): creating latency model:
latency model: invalid roofline config: HardwareCalib.BwEffConstant
must be a valid positive number, got 0
```

**Root Cause:** Config only had 2 fields, but roofline v1 needs 18
**Fix (commit `cba73b3`):** Added all calibration fields to `hardware_config.json`

**Result:** One config file with all fields that works for both v1 and v2:
- Roofline v2: Ignores MFU fields (gets from database)
- Roofline v1: Uses all fields including MFU values

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

## Key Files

| File | Purpose |
|------|---------|
| `hardware_config.json` | Hardware specs (works for both v1 and v2) |
| `sim/latency_model.go` | Latency model with roofline v1 & v2 |
| `sim/roofline_step_v2.go` | Roofline v2 (MFU-based) |
| `python_scripts/blis_evaluator.py` | Evaluator (v0.6.1 compatible) |

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
