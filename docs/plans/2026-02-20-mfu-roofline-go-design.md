# MFU-Based Roofline Model: Go Implementation Design

**Date:** 2026-02-20
**Status:** Approved
**Goal:** Replace calibrated roofline constants with MFU-based lookups using pre-computed H100 benchmark data

---

## Executive Summary

This design implements Phase 2 of the Roofline V2 plan: loading pre-computed MFU benchmark data in Go and using it for dynamic roofline predictions. The approach directly ports InferSim's proven lookup logic while adapting it to BLIS's per-step batch aggregation model.

**Key Decisions:**
- Direct Python port (Approach A) - prioritize correctness over optimization
- Aggregate decode requests, group prefill by seq_len buckets
- Nearest neighbor fallback for missing attention configs
- Error out if CSV files missing (no silent fallbacks)

---

## 1. Architecture Overview

### High-Level Flow

```
Simulator Startup
    ↓
Load ModelConfig from HF config.json (existing)
    ↓
Compute attention config: "{num_heads}-{num_kv_heads}-{head_dim}"
    ↓
Load MFU Database (bench_data/h100/*.csv)
    ↓
During simulation: roofline_step.go calls lookup functions
    ↓
Lookup uses attention config + operation params → returns MFU
    ↓
Formula: time = max(flops/(peak*mfu), bytes/bw)
```

### Key Components

1. **Model Config Mapper** (part of mfu_database.go)
   - Derives attention config from ModelConfig
   - Computes `head_dim = hidden_size / num_attention_heads`
   - Formats as: `"{num_heads}-{num_kv_heads}-{head_dim}"`
   - Example: Llama-2-7B (32, 32, 4096) → "32-32-128"

2. **MFU Database** (sim/mfu_database.go)
   - Loads all CSVs at startup into simple Go slices
   - Stores data indexed by attention config strings
   - Exposes lookup functions: GetAttnPrefillMFU(), GetAttnDecodeMFU(), GetGEMMmfu()
   - Implements nearest-neighbor fallback
   - ~250 lines total

3. **Integration Point** (sim/roofline_step.go)
   - Modified to use MFU lookups instead of static constants
   - Aggregates decode requests, groups prefill requests
   - GEMM projections use per-operation lookups
   - ~50 line modification

4. **Fallback Strategy**
   - Exact attention config missing → nearest neighbor + log info
   - CSV files missing → error out (fatal)
   - Never silently uses wrong data

### Data Flow Example (Decode)

```
ModelConfig: {NumHeads: 32, NumKVHeads: 32, HiddenDim: 4096}
    ↓
Compute: head_dim = 4096/32 = 128, attention_config = "32-32-128"
    ↓
Step with 10 decode requests, kv_len range [3800-4200], TP=2
    ↓
Aggregate: batch_size=10, kv_len=4096 (max)
    ↓
roofline_step.go: mfuDB.GetAttnDecodeMFU("32-32-128", 10, 4096, 2)
    ↓
Database looks up "bench_data/h100/mha/decode/32-32-128-tp2.csv"
    ↓
Finds row with batch_size=10, kv_len=4096 → returns mfu=0.003
    ↓
Computes time = total_flops / (989.5 TFLOPs * 0.003)
```

---

## 2. Data Structures

### CSV Row Representations

**MHA Prefill Row:** `dtype,seq_len,latency_us,mfu`
- Store: seq_len, mfu

**MHA Decode Row:** `dtype,kv_dtype,batch_size,kv_len,latency_us,mfu`
- Store: batch_size, kv_len, mfu

**GEMM Row:** `m,k,n,latency_us,mfu`
- Store: m, k, n, mfu

### Database Structure

```
MFUDatabase:
  - prefillData: map[string][]MHAPrefillRow  // key: "32-32-128"
  - decodeData:  map[string][]MHADecodeRow   // key: "32-32-128-tp2"
  - gemmData:    []GEMMRow                   // All GEMM data (112 rows)
  - attentionConfig: string                  // Model's config "32-32-128"
  - availableConfigs: []AttentionShape       // For nearest neighbor
```

### Initialization

At simulator startup:
1. Compute attention config from ModelConfig
2. Load all CSV files from bench_data/{gpu}/
3. Parse and store in simple slices (sorted for floor lookups)
4. If exact config missing, find nearest neighbor and log info
5. If no CSV files exist, error out

---

## 3. Lookup Logic

### Prefill Lookup Strategy
- Given seq_len, find largest benchmarked seq_len ≤ target (floor logic)
- Example: seq_len=3000 → finds row with seq_len=1024
- Returns MFU from that row

### Decode Lookup Strategy
- Given (batch_size, kv_len, tp), look up TP-specific file
- Find largest batch_size ≤ target AND largest kv_len ≤ target
- Return MFU from matching row
- Example: batch_size=20, kv_len=5000 → finds (batch_size=16, kv_len=4096)

### GEMM Lookup Strategy (matches InferSim exactly)
- **Stage 1:** Find smallest (k, n) pair where k ≥ target_k AND n ≥ target_n
  - Uses Euclidean distance to find closest match above threshold
- **Stage 2:** Within that (k, n) pair, find largest m ≤ target_m
- Returns MFU from that row

### Nearest Neighbor Fallback
- If exact attention config missing (e.g., "35-7-128"), compute distance to all available configs
- Use Euclidean distance on (num_heads, num_kv_heads, head_dim)
- Pick closest match, log info message to user
- Example: "Attention config 35-7-128 not found, using nearest: 32-8-128"

### Error Conditions
- CSV files missing → error out, don't start simulator
- GPU folder missing (e.g., bench_data/a100) → error out
- This forces users to have benchmark data before running

---

## 4. Integration with Roofline

### Current roofline_step.go Modification

Replace static MFU constants with dynamic lookups. The formula remains: `time = max(flops/(peak*mfu), bytes/bw)`

### Prefill Phase Changes

**Batching Strategy - Group by seq_len buckets:**
- Bucket prefill requests by seq_len (e.g., 512, 1024, 2048, 4096)
- For each bucket:
  1. Compute total FLOPs for all requests in bucket
  2. Lookup MFU for that seq_len
  3. Compute time = flops / (peak * mfu)
- Sum times across all buckets

**Rationale:** Different seq_lens have different MFU values. Prefill requests with similar seq_lens batch efficiently, but heterogeneous seq_lens require padding/masking which reduces efficiency.

### Decode Phase Changes

**Batching Strategy - Aggregate all decode requests:**
- Sum total batch_size across all decode requests
- Pick representative kv_len (use max kv_len in the batch)
- Single MFU lookup for (total_batch_size, max_kv_len, tp)
- Compute total FLOPs, single time calculation

**Rationale:** The decode benchmark data already includes batch_size dimension with measured MFU values. This reflects vLLM's actual batching behavior where decode requests batch efficiently.

### GEMM Projections

- Each QKV projection, O projection, MLP projection computes its own (m, k, n)
- m = batch_size (aggregated for decode, per-bucket for prefill)
- Lookup MFU per GEMM operation
- Each gets individual time = flops / (peak * mfu)

### Mixed Batch Handling

- Keep current logic for combining prefill and decode phases
- Weighted averaging or max() based on token distribution
- MFU lookups happen before the mixing logic

### Example Flow

```
Step with:
- 10 decode requests, kv_len range [3800-4200]
- 2 prefill requests: seq_len=512, seq_len=4096

Decode Phase:
  batch_size = 10, kv_len = 4096 (max)
  mfu = lookup("32-32-128", batch_size=10, kv_len=4096, tp=2) → 0.003
  time = total_decode_flops / (peak * 0.003)

Prefill Phase:
  Bucket 1: seq_len=512
    mfu = lookup("32-32-128", seq_len=512) → 0.70
    flops = 100T
    time1 = 100T / (peak * 0.70)

  Bucket 2: seq_len=4096
    mfu = lookup("32-32-128", seq_len=4096) → 0.85
    flops = 900T
    time2 = 900T / (peak * 0.85)

  total_prefill_time = time1 + time2

Step Time:
  Use existing mixed batch logic to combine prefill + decode phases
```

---

## 5. Error Handling and Logging

### Startup Errors (Fatal)
- Missing bench_data directory → exit with clear error message
- Missing GPU subfolder (e.g., bench_data/h100) → exit
- Corrupted CSV files (parsing errors) → exit
- Empty CSV files → exit

### Runtime Warnings (Non-Fatal, Log Info)
- Exact attention config not found → log which nearest config is being used
- TP-specific decode file missing → try TP=1 fallback, log warning
- Unusual lookup values (e.g., seq_len > all benchmarked values) → log info about extrapolation

### Logging Strategy
- **Startup:** Single log line showing loaded MFU database status
  - Example: "Loaded MFU database: H100, attention config 32-32-128, 6 prefill rows, 24 decode rows, 112 GEMM rows"
- **Nearest neighbor:** Info level
  - Example: "Attention config 35-7-128 not found, using nearest: 32-8-128"
- **Per-step lookups:** No logging (too verbose)
- **Summary stats:** Optional debug mode showing MFU distribution

### User Experience
- Clear error messages point to missing data
- Info logs help users understand fallback behavior
- Simulation never silently uses wrong data

---

## 6. Files to Create/Modify

### New Files
- `sim/mfu_database.go` (~250 lines)
  - MFUDatabase struct
  - CSV loading functions
  - Lookup functions (GetAttnPrefillMFU, GetAttnDecodeMFU, GetGEMMmfu)
  - Nearest neighbor logic

- `sim/mfu_database_test.go` (~150 lines)
  - Unit tests for CSV parsing
  - Lookup function tests
  - Edge case handling

### Modified Files
- `sim/roofline_step.go` (~50 line modification)
  - Pass MFUDatabase to rooflineStepTime()
  - Replace hwConfig.MfuPrefill with mfuDB.GetAttnPrefillMFU()
  - Replace hwConfig.MfuDecode with mfuDB.GetAttnDecodeMFU()
  - Add GEMM MFU lookups for projections
  - Implement prefill bucketing logic
  - Implement decode aggregation logic

- `sim/simulator.go` (~20 line modification)
  - Initialize MFUDatabase at startup
  - Pass to rooflineStepTime() calls

---

## 7. Implementation Approach

### Rationale for Approach A (Direct Python Port)

**Why not optimize?**
- Dataset is small (~500KB, 112 GEMM rows, ~20 MHA rows)
- Linear search over 20 items takes nanoseconds
- Binary search would save nothing meaningful
- Correctness matters more than premature optimization

**Benefits:**
- Easy to verify against InferSim reference implementation
- Minimal risk of algorithmic bugs
- Simple debugging (no complex indexing logic)
- Future-proof (easy to understand and maintain)

**What we're NOT doing:**
- No pre-built indexes or hash maps for lookups
- No binary search (rows are small)
- No lazy loading or caching
- No threading/concurrency (startup is fast enough)

---

## 8. Success Criteria

### Functional Requirements
- ✅ All CSV files load successfully at startup
- ✅ Lookups return valid MFU values (0 < mfu ≤ 1.0)
- ✅ Nearest neighbor finds reasonable matches
- ✅ Simulator produces predictions without crashing
- ✅ Predictions differ from calibrated baseline (validates MFU is used)

### Non-Functional Requirements
- ✅ Startup time < 100ms (loading CSVs is fast)
- ✅ Lookup time < 1μs per query (negligible overhead)
- ✅ Clear error messages for missing data
- ✅ Info logs help users debug configuration issues

### Validation Strategy
- Run simulator on Llama-2-7B with TP=1,2,4
- Verify MFU values are in expected ranges
- Compare output to InferSim's vLLM predictions (if available)
- Check that different models use different attention configs

---

## 9. Open Questions

None - design is approved.

---

## 10. References

- Macro plan: `docs/plans/2026-02-18-roofline-v2-infersim-mfu.md`
- InferSim reference: `InferSim/mfu/mfu.py`
- vLLM architecture: `vllm.md`
- Benchmark data: `bench_data/h100/`
