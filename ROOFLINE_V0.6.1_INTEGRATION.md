# Roofline v0.6.1 Integration Summary

This document summarizes the integration of roofline v1 and v2 support into BLIS v0.6.1's cluster architecture.

## Branch: `v0.6.1_roofline_valid`

**Base:** v0.6.1 stable release (f23b1d3)
**Status:** ✅ All tests passing, fully integrated

---

## What Was Integrated

### Core Roofline Files Added

1. **`sim/roofline_step.go`** - Roofline v1 (calibrated constants)
   - Uses hardware calibration factors from `hardware_config.json`
   - Computes FLOPs and memory access analytically
   - Applies empirically-tuned MFU values

2. **`sim/roofline_step_v2.go`** - Roofline v2 (MFU-based)
   - Per-GEMM MFU lookups from benchmark data
   - Per-attention MFU lookups (prefill/decode)
   - More accurate than v1, requires benchmark data

3. **`sim/mfu_database.go`** - MFU Benchmark Database
   - Loads MFU CSVs from `bench_data/`
   - Provides MFU lookups for GEMMs and attention ops
   - Supports nearest-neighbor fallback for unseen shapes

4. **`sim/roofline_step_test.go`** - Comprehensive test suite
   - Tests FLOPs calculations, memory modeling
   - Validates roofline invariants (monotonicity, conservation)
   - All tests passing

---

## Architecture Changes

### 1. Latency Model Abstraction (Enhanced)

**File:** `sim/latency_model.go`

Added `RooflineLatencyModelV2` to existing `LatencyModel` interface:

```go
type LatencyModel interface {
    StepTime(batch []*Request) int64
    QueueingTime(req *Request) int64
    OutputTokenProcessingTime() int64
    SchedulingProcessingTime() int64
    PreemptionProcessingTime() int64
}
```

**Implementations:**
- `BlackboxLatencyModel` - Alpha/beta regression (existing)
- `RooflineLatencyModel` - Roofline v1 with calibrated constants
- `RooflineLatencyModelV2` - Roofline v2 with MFU database lookups (NEW)

### 2. Config Extension

**File:** `sim/config.go`

Extended `ModelHardwareConfig` to support MFU database:

```go
type ModelHardwareConfig struct {
    ModelConfig ModelConfig
    HWConfig    HardwareCalib
    MFUDatabase *MFUDatabase  // NEW: Optional for roofline v2
    Model       string
    GPU         string
    TP          int
    Roofline    bool
}
```

Added builder method:
```go
func (m ModelHardwareConfig) WithMFUDatabase(mfuDB *MFUDatabase) ModelHardwareConfig
```

### 3. Automatic Mode Selection

**File:** `sim/latency_model.go` - `NewLatencyModel()`

The system automatically selects the best available model:

```
┌─────────────────────────────────────┐
│ NewLatencyModel(coeffs, hwConfig)  │
└──────────────┬──────────────────────┘
               │
               ├─ hw.Roofline == true?
               │  │
               │  ├─ hw.MFUDatabase != nil?
               │  │  ├─ YES → RooflineLatencyModelV2 (best accuracy)
               │  │  └─ NO  → RooflineLatencyModel (v1)
               │  │
               │  └─ NO → BlackboxLatencyModel (regression)
               │
```

---

## CLI Usage

### Roofline Mode Auto-Detection

When you provide these flags, roofline mode is enabled:

```bash
./simulation_worker run \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json \
  --bench-data-path bench_data \
  --hardware H100 \
  --tp 1
```

**Automatic selection:**
- If `bench_data/` contains MFU CSVs → **Roofline v2**
- If `bench_data/` is missing/empty → **Roofline v1**
- If roofline flags omitted → **Blackbox mode** (alpha/beta)

### Example Commands

**Roofline v1** (calibrated constants):
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

**Roofline v2** (MFU-based, more accurate):
```bash
# Ensure bench_data/ exists with MFU CSVs
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

---

## Evaluator Updates

**File:** `python_scripts/blis_evaluator.py`

Updated for v0.6.1 compatibility:

### Changes:
1. ✅ `--max-prompts` → `--num-requests` (CLI change)
2. ✅ Added `--num-instances 1` for single-instance evaluation
3. ✅ Improved JSON parsing for cluster output format
4. ✅ Auto-detects roofline v1/v2 based on `bench_data/` presence

### Usage:
```bash
# Run evaluation against ground truth
python3 python_scripts/blis_evaluator.py \
  --ground-truth eval/combined_ground_truth.json \
  --verbose
```

---

## File Structure

```
sim/
├── latency_model.go           # LatencyModel interface + implementations
├── latency_model_test.go      # Tests for all latency models
├── roofline_step.go           # Roofline v1 implementation
├── roofline_step_v2.go        # Roofline v2 implementation (NEW)
├── roofline_step_test.go      # Comprehensive roofline tests
├── mfu_database.go            # MFU benchmark database loader (NEW)
├── config.go                  # Extended with MFUDatabase support
└── model_hardware_config.go   # Hardware calibration structs

cmd/
└── root.go                    # CLI integration, loads MFU database

python_scripts/
└── blis_evaluator.py          # Updated for v0.6.1
```

---

## Testing

### Run All Tests
```bash
go test ./sim/...
```

### Run Roofline Tests Only
```bash
go test -v ./sim -run TestRoofline
go test -v ./sim -run TestCalculate
```

### Build and Test Binary
```bash
go build -o simulation_worker main.go
./simulation_worker --help
```

**All tests passing ✅**

---

## Key Implementation Details

### 1. Roofline v1 Calibration

Defined in `hardware_config.json`:

```json
{
  "H100": {
    "TFlopsPeak": 989.0,
    "BwPeakTBs": 3.35,
    "MfuPrefill": 0.65,
    "MfuDecode": 0.12,
    "TpScalingExponent": 0.8,
    "PrefillBwFactor": 1.0,
    ...
  }
}
```

### 2. Roofline v2 MFU Database

Expected structure in `bench_data/`:

```
bench_data/
├── gemm_bench/
│   └── gemm_fp16.csv           # m,k,n,latency_us,mfu
├── mha_prefill/
│   └── {n_heads}-{n_kv_heads}-{head_dim}/
│       └── mha_prefill_fp16.csv  # seq_len,mfu
└── mha_decode/
    └── {n_heads}-{n_kv_heads}-{head_dim}/
        └── mha_decode_fp16_tp{tp}.csv  # batch,kv_len,mfu
```

### 3. Cluster Architecture Integration

The roofline models integrate seamlessly with v0.6.1's cluster architecture:

- Single-instance mode: Direct roofline calculation per instance
- Multi-instance mode: Each instance uses roofline independently
- Metrics aggregated at cluster level

---

## Migration Guide

### From Old Roofline Branch
1. Switch to `v0.6.1_roofline_valid` branch
2. Rebuild: `go build -o simulation_worker main.go`
3. Update `hardware_config.json` with new calibration fields (see `sim/roofline_step_test.go` for examples)
4. No code changes needed - auto-selects roofline v1/v2

### From v0.6.1 Main
1. Merge `v0.6.1_roofline_valid` into your branch
2. Ensure `hardware_config.json` has roofline calibration fields
3. Optionally add `bench_data/` for roofline v2
4. Use CLI flags: `--model-config-folder`, `--hardware-config`, `--bench-data-path`

---

## Performance Characteristics

### Roofline v1 (Calibrated)
- ✅ Fast (no database lookups)
- ✅ No benchmark data required
- ⚠️ Accuracy depends on calibration quality
- ⚠️ Fixed MFU values per phase

### Roofline v2 (MFU-based)
- ✅ High accuracy (per-GEMM MFU)
- ✅ Adapts to batch size and sequence length
- ⚠️ Requires benchmark data collection
- ⚠️ Slightly slower (database lookups)

### Blackbox (Regression)
- ✅ Fast
- ✅ Simple (just alpha/beta coefficients)
- ⚠️ Requires training data per model/GPU/TP
- ⚠️ Limited to interpolation range

---

## Commits

```
336f7fa feat: update blis_evaluator.py for v0.6.1 + roofline compatibility
6724947 fix: update roofline_step_test.go for v0.6.1 compatibility
f10e4c6 feat: integrate roofline v2 with cluster architecture
c117fdd feat: add InferSim as submodule and integrate roofline v2 improvements
... (65 commits rebased from roofline_valid)
f23b1d3 refactor: canonical sub-config constructors... (v0.6.1 base)
```

---

## Next Steps

1. **Benchmark Data Collection**: Generate MFU CSVs for your target hardware
2. **Calibration Tuning**: Adjust `hardware_config.json` for your workload
3. **Validation**: Run evaluator against ground truth data
4. **Production**: Deploy with roofline v1 or v2 based on accuracy needs

---

## Questions?

See existing documentation:
- `blis.md` - Original roofline v1 documentation
- `docs/roofline_v2_design.md` - Roofline v2 technical design
- `docs/roofline.md` - Roofline approach overview
- `sim/roofline_step_test.go` - Usage examples in tests
