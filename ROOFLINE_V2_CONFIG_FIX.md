# Roofline v2 Config Fix - User's Question Answered

## The Question

> "I don't understand - why do you need the other fields in hardware_config.json when roofline v2 only needs 2 fields?"

**You were absolutely right!** This was a design flaw that has now been fixed.

---

## The Problem

The original validation code required **ALL 18 calibration fields** regardless of which roofline mode you were using:

```
BwEffConstant, MfuPrefill, MfuDecode, TpScalingExponent,
DecodeTpScalingExponent, MfuPrefillMultiplier, MfuDecodeMultiplier,
PrefillBwFactor, DecodeBwFactor, VectorPeakFraction,
PrefillOverheadMicros, MixedPrefillOverheadMicros, ...
```

But **roofline v2 doesn't need most of these** because it gets MFU values from the benchmark database, not the config file!

---

## What Each Mode Actually Needs

### Roofline v1 (Calibrated) - WITHOUT MFU Database
**Needs all 18 fields:**
- `TFlopsPeak`, `BwPeakTBs` - Hardware specs
- `BwEffConstant` - Effective bandwidth (72% of peak)
- `MfuPrefill`, `MfuDecode` - Model FLOPs Utilization
- `TpScalingExponent`, `DecodeTpScalingExponent` - TP scaling
- `VectorPeakFraction` - Non-tensor core efficiency
- All the multipliers and factors for tuning

**Why?** These MFU and efficiency values are used directly in calculations.

### Roofline v2 (MFU-Based) - WITH MFU Database
**Only needs 6 fields:**
- `TFlopsPeak`, `BwPeakTBs` - Hardware specs ✅
- `TOverheadMicros` - Per-step overhead (optional, default: 50)
- `PrefillOverheadMicros` - Prefill overhead (optional, default: 100)
- `MixedPrefillOverheadMicros` - Mixed batch overhead (optional, default: 50)
- `AllReduceLatency` - TP communication (optional, default: 20)

**Why?** MFU values come from `bench_data/` database, not config!

---

## The Fix

### Code Changes (Commit `451379f`)

**1. Made validation conditional:**
```go
// Before: Required all fields unconditionally
func ValidateRooflineConfig(mc ModelConfig, hc HardwareCalib) error

// After: Conditional based on MFU database presence
func ValidateRooflineConfig(mc ModelConfig, hc HardwareCalib, hasMFUDatabase bool) error
```

**2. Validation logic:**
```go
// Core specs - always required
if invalidPositiveFloat(hc.TFlopsPeak) { ... }
if invalidPositiveFloat(hc.BwPeakTBs) { ... }

// Roofline v1-specific - only required when no MFU database
if !hasMFUDatabase {
    if invalidPositiveFloat(hc.BwEffConstant) { ... }
    if invalidPositiveFloat(hc.MfuPrefill) { ... }
    if invalidPositiveFloat(hc.MfuDecode) { ... }
}
```

**3. Call site update:**
```go
hasMFUDatabase := hw.MFUDatabase != nil
if err := ValidateRooflineConfig(hw.ModelConfig, hw.HWConfig, hasMFUDatabase); err != nil {
    return nil, fmt.Errorf("latency model: %w", err)
}
```

### New Config Files

**`hardware_config_v2.json`** - Minimal (6 fields) for roofline v2:
```json
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "TOverheadMicros": 50.0,
        "PrefillOverheadMicros": 100.0,
        "MixedPrefillOverheadMicros": 50.0,
        "AllReduceLatency": 20.0
    }
}
```

**`hardware_config.json`** - Full (18 fields) for roofline v1:
```json
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "BwEffConstant": 0.72,
        "MfuPrefill": 0.65,
        "MfuDecode": 0.12,
        ... // 13 more fields
    }
}
```

---

## Usage Examples

### Roofline v2 (Minimal Config)
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config_v2.json \
  --bench-data-path bench_data \
  --hardware H100 \
  --tp 1 \
  --workload chatbot
```
✅ Works with just 6 fields!

### Roofline v1 (Full Config)
```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config.json \
  --hardware H100 \
  --tp 1 \
  --workload chatbot
```
✅ Requires all 18 fields

---

## Verification

### Test v2 with Minimal Config
```bash
# Should work without errors
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config_v2.json \
  --bench-data-path bench_data \
  --hardware H100 \
  --tp 1 \
  --workload distribution \
  --rate 1 \
  --num-requests 5

# Output: Should show "Enabling roofline v2 mode" and complete successfully
```

---

## Summary

| Aspect | Before Fix | After Fix |
|--------|-----------|-----------|
| **Roofline v2 config** | Required 18 fields | Requires 6 fields |
| **Validation** | Unconditional | Conditional on MFU database |
| **User experience** | Confusing error for v2 | Clear separation of v1 vs v2 |
| **Config files** | One file for both | Two files: v1 (full), v2 (minimal) |

---

## Why This Matters

**Before:** Users with MFU benchmark data still had to provide MFU values in config (which were ignored anyway!)

**After:** Users can provide minimal config for v2, making it clear which fields are actually used.

**Result:** 
- ✅ Clearer separation of concerns
- ✅ Less confusing configuration
- ✅ Easier to get started with roofline v2
- ✅ Better error messages

---

## Credits

Thank you for asking this question! It identified a real design flaw that has now been properly fixed. The validation should have always been conditional based on which mode was being used.

**Commits:**
- `451379f` - Fix: conditional validation
- `dc1a3c7` - Docs: updated quickstart guide
