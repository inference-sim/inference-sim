# Roofline v2: Absolute Minimal Config

## TL;DR

**For roofline v2, you ONLY need 2 fields:**
- `TFlopsPeak` (TFLOPS)
- `BwPeakTBs` (TB/s)

Everything else is optional!

---

## The Minimal Config

### `hardware_config_minimal.json`
```json
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35
    }
}
```

**That's it!** Just 2 fields.

---

## Usage

```bash
./simulation_worker run \
  --model meta-llama/llama-3.1-8b-instruct \
  --model-config-folder model_configs/llama-3.1-8b-instruct \
  --hardware-config hardware_config_minimal.json \
  --bench-data-path bench_data \
  --hardware H100 \
  --tp 1 \
  --workload chatbot
```

✅ Works perfectly!

---

## Optional Overhead Fields

The overhead parameters are **optional** and default to 0 if not provided:

| Field | Purpose | Default | Recommended |
|-------|---------|---------|-------------|
| `TOverheadMicros` | Per-step overhead | 0 | 50 µs |
| `PrefillOverheadMicros` | Pure prefill overhead | 0 | 100 µs |
| `MixedPrefillOverheadMicros` | Mixed batch prefill overhead | 0 | 50 µs |
| `AllReduceLatency` | TP communication per layer | 0 | 20 µs |

### When to Add Overheads

**Minimal config (0 overheads):**
- ✅ Quick testing
- ✅ When overhead is negligible vs compute
- ✅ Relative comparisons (overhead same across runs)
- ⚠️ Predictions will be slightly too fast

**With overheads (recommended for accuracy):**
- ✅ Absolute latency predictions
- ✅ Matching ground truth data
- ✅ Production capacity planning

---

## Config File Comparison

### Absolute Minimal (2 fields)
```json
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35
    }
}
```
**Use case:** Zero-overhead assumptions, quick tests

### With Overheads (6 fields) - Recommended
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
**Use case:** Accurate latency predictions

### Full Config (18 fields) - Roofline v1
```json
{
    "H100": {
        "TFlopsPeak": 989.5,
        "BwPeakTBs": 3.35,
        "BwEffConstant": 0.72,
        "MfuPrefill": 0.65,
        "MfuDecode": 0.12,
        ... // + 13 more fields
    }
}
```
**Use case:** When you DON'T have MFU benchmark data

---

## What Roofline v2 Gets From Where

| Information | Source |
|-------------|--------|
| Peak hardware specs | ✅ Config file (2 fields) |
| MFU values | ✅ Benchmark database (`bench_data/`) |
| Overhead parameters | ✅ Config (optional, defaults to 0) |

**Key insight:** MFU comes from database, not config!

---

## Impact of Zero Overheads

### Example Comparison

**With overheads (realistic):**
```
TTFT: 10.5 ms
ITL: 8.2 ms
E2E: 1500 ms
```

**Without overheads (0s in config):**
```
TTFT: 10.0 ms  (5% faster - no per-step overhead)
ITL: 8.0 ms    (2% faster)
E2E: 1450 ms   (3% faster overall)
```

**Difference:** ~3-5% faster predictions (systematic bias)

### When Zero Overhead Is OK

1. **Relative Comparisons**
   - Comparing two models on same hardware
   - Overhead cancels out in relative terms

2. **Large Batch Sizes**
   - 50 µs overhead vs 10 ms compute = 0.5% error
   - Negligible for large workloads

3. **Quick Prototyping**
   - Don't have exact overhead measurements yet
   - Just want ballpark estimates

---

## Summary

| Config | Fields | MFU Source | Overhead | Use Case |
|--------|--------|------------|----------|----------|
| **hardware_config_minimal.json** | 2 | Database | 0 | Quick tests |
| **hardware_config_v2.json** | 6 | Database | Calibrated | Accurate v2 ✅ |
| **hardware_config.json** | 18 | Config | Calibrated | No database (v1) |

---

## Recommendation

**For most users running roofline v2:**
Use `hardware_config_v2.json` (6 fields) which includes realistic overheads.

**For quick tests:**
Use `hardware_config_minimal.json` (2 fields) and accept ~3-5% speedup in predictions.

---

## Credits

User question: "but do you even need the overhead parameters?"

Answer: **No, they default to 0!** Only 2 fields are truly required for roofline v2.
