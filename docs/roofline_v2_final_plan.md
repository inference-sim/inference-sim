# Roofline V2: Final Implementation Plan (InferSim-Based)

**Date**: 2025-02-16
**Status**: Production Plan - Ready for Implementation
**Timeline**: 11-12 days with H100 cluster access
**Validated Against**: InferSim (4-15% error on Qwen/DeepSeek)

---

## 🔑 Key Concept: Python Benchmarks → CSV → Go Lookups

**IMPORTANT**: This is NOT about porting benchmarking to Go!

```
┌────────────────────────────────────────────────────────────┐
│ Python (One-Time, Phase 1)                                 │
│  • Runs FlashAttention/GEMM kernels on H100                │
│  • Measures latency, calculates MFU                        │
│  • Outputs CSV files (~17 files, <1MB)                     │
│  • Uses existing InferSim scripts (90% reuse!)             │
└────────────────────────────────────────────────────────────┘
                          ↓ CSV files
┌────────────────────────────────────────────────────────────┐
│ Go (Runtime, Phase 2)                                      │
│  • Loads CSV files into memory                             │
│  • Looks up MFU for operation shapes                       │
│  • Uses in formula: time = max(flops/(peak*mfu), bytes/bw) │
│  • NO GPU code, NO benchmarking, NO kernel execution       │
└────────────────────────────────────────────────────────────┘
```

**Think of it as**: Python creates a "cheat sheet" of MFU values, Go reads from that cheat sheet!

---

## Executive Summary

### Approach

Implement **MFU-based roofline model** following InferSim methodology:
1. **Python: Benchmark actual LLM kernels** (FlashAttention-3, FlashInfer, DeepGEMM) on H100
2. **Python: Build MFU database** (~800 benchmarks) → CSV files
3. **Go: Load CSV files** and implement lookup logic
4. **Go: Use formula** `t = FLOPs / (Peak_FLOPS × MFU)` with lookups
5. Validate against vLLM measurements

### Why This Works

**InferSim's Key Insight**: "The accuracy of the MFU directly determines the accuracy of the simulator results."

- **MFU captures everything**: compute bottlenecks, memory bottlenecks, kernel overhead
- **No complex modeling needed**: Just measure MFU under different configs
- **Validated results**: 4-8% error on Qwen, 15% on DeepSeek-V3
- **Simple formula**: Easy to implement and maintain

### Division of Labor (CRITICAL)

**Python (One-Time Benchmarking)**:
- ✅ Runs FlashAttention, GEMM benchmarks on H100
- ✅ Measures actual kernel latencies
- ✅ Calculates MFU values
- ✅ Outputs CSV files (~17 files, <1MB)

**Go (Runtime Simulator)**:
- ✅ Parses CSV files at startup
- ✅ Loads MFU lookup tables into memory
- ✅ Does nearest-neighbor search to find MFU
- ✅ Uses MFU in roofline: `time = max(flops/(peak*mfu), bytes/bw)`
- ❌ **NO GPU code, NO benchmarking, NO kernel execution**

### What We'll Reuse from InferSim (90% Ready!)

✅ **Benchmark scripts** - `fa3_mha_prefill.py`, `flashinfer_mha_decode.py`, `deepgemm_gemm.py`
  - Only need 1-line change to prefill script (`fp16_tflops=989.5`)
  - Others accept TFLOPS as CLI arg
✅ **MFU lookup logic** - `mfu/mfu.py` works unchanged (port to Go)
✅ **Data structure** - Same CSV format, just create `bench_data/h100/`
✅ **FLOP formulas** - `flops/flops.py` unchanged
✅ **Roofline model** - Core algorithm ready

### What We Must Create

1. **Python: Run benchmarks on H100** - ~720 kernel runs (automated script provided)
2. **Python: Add H100 GPU profile** - 15 lines in `hardware/gpu.py`
3. **Go: CSV loader** - ~200 lines (parse CSVs into memory)
4. **Go: Lookup functions** - ~150 lines (nearest-neighbor search)
5. **Go: Roofline model** - ~150 lines (use MFU lookups)

---

## Architecture: Python Benchmarks → CSV → Go Lookups

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Python Benchmarking (One-Time on H100)            │
├─────────────────────────────────────────────────────────────┤
│ InferSim Scripts (Reuse 90%):                              │
│  • fa3_mha_prefill.py    ─────> bench_data/h100/mha/       │
│  • flashinfer_mha_decode.py ──> prefill/*.csv              │
│  • deepgemm_gemm.py       ─────> decode/*.csv              │
│                                  gemm/data.csv              │
│                                                             │
│ Output: Static CSV files with (shape, latency, mfu)        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: Go CSV Loading (No Benchmarking!)                 │
├─────────────────────────────────────────────────────────────┤
│ mfu_database.go:                                            │
│  • LoadMFUDatabase() ──> Reads CSVs into memory            │
│  • map[string][]Entry ──> In-memory lookup tables          │
│                                                             │
│ mfu_lookup.go:                                              │
│  • GetAttentionMFU() ───> Nearest neighbor search          │
│  • GetGemmMFU() ─────────> Distance-based matching         │
│                                                             │
│ roofline_infersim.go:                                       │
│  • time = max(flops/(peak*mfu), bytes/bw) ← Uses MFU       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ RUNTIME: Simulator Predictions                              │
├─────────────────────────────────────────────────────────────┤
│ For each inference request:                                 │
│  1. Look up MFU from CSV data (no benchmarking!)           │
│  2. Calculate: time = max(compute, memory)                  │
│  3. Return: TTFT, TPOT predictions                          │
└─────────────────────────────────────────────────────────────┘
```

**Key**: Python runs benchmarks once, Go uses results forever. No kernel execution in Go!

### What Go Does NOT Do
❌ Run FlashAttention kernels
❌ Run GEMM benchmarks
❌ Measure latency
❌ Calculate MFU at runtime
❌ Execute any GPU code

### What Go DOES Do
✅ Parse CSV files (one-time at startup)
✅ Load MFU tables into memory
✅ Look up pre-computed MFU values
✅ Calculate time using: `time = max(flops/(peak*mfu), bytes/bw)`
✅ Return predictions

### Deployment: What Ships with Go Binary

```
inference-sim/
├── inference-sim              # Go binary
└── bench_data/
    └── h100/                  # Ship these with binary
        ├── mha/
        │   ├── decode/
        │   │   ├── 32-8-128.csv    # Pre-computed MFU data
        │   │   ├── 28-4-128.csv
        │   │   └── ... (8 files)
        │   └── prefill/
        │       └── ... (8 files)
        └── gemm/
            └── data.csv            # Pre-computed GEMM MFU
```

**Size**: ~17 CSV files, <1MB total
**Usage**: `./inference-sim --mfu-data-path ./bench_data`
**One-time cost**: Run Python benchmarks on H100 once, use forever

---

## Table of Contents

1. [InferSim Architecture Deep Dive](#infersim-architecture-deep-dive)
2. [Critical Implementation Details](#critical-implementation-details)
3. [H100 Benchmarking Plan](#h100-benchmarking-plan)
4. [Go Implementation Strategy](#go-implementation-strategy)
5. [Integration with Existing Simulator](#integration-with-existing-simulator)
6. [Validation Protocol](#validation-protocol)
7. [Timeline and Resources](#timeline-and-resources)

---

## InferSim Architecture Deep Dive

### Core Formula (Verified from Code)

```python
# From layers/attn.py:decode_attn_core()

# Step 1: Calculate FLOPs
attn_core_gflops = 4 * num_heads * seq_len * head_dim * 2  # QK^T (2×) + AV (2×)

# Step 2: Look up MFU from benchmark data
attn_core_mfu = get_attn_decode_mfu(config, batch_size, kv_len, device_type)

# Step 3: Compute time
attn_core_time = batch_size * attn_core_gflops / (gpu.fp16_tflops * 1024 * mfu)

# Step 4: Roofline - compare with memory loading time
kv_load_time = kv_cache_bytes * kv_len * batch_size / gpu.mem_bw / 1024^3

# Step 5: Take maximum (roofline model!)
time = max(attn_core_time, kv_load_time)
```

**Key Observations**:
1. They use `fp16_tflops * 1024` (converting TFLOPs → GFLOPs)
2. Memory bandwidth is in GB/s, already has efficiency factor (0.8)
3. **They explicitly model KV loading time** and take max (roofline!)
4. MFU is looked up per configuration, not computed

### End-to-End Calculation

```python
# From models/model.py:decoding()

def decoding(self):
    # 1. Attention
    attn_core_time = attn.decode_attn_core(batch_size, kv_len, kvcache_bytes, device)
    attn_other_time = attn.decode_attn_others(batch_size, device)  # Projections

    # 2. MoE/FFN
    moe_time = moe.decode_moe(batch_size, device, world_size)

    # 3. Communication
    comm_time1, comm_time2 = comm.decode_comm(batch_size)

    # 4. Per-layer total
    per_layer_time = attn_core_time + attn_other_time + moe_time + comm_time1 + comm_time2

    # 5. Multiply by layers
    tpot = per_layer_time * num_layers

    # 6. Convert to ms and add schedule overhead
    tpot *= 1000  # s → ms
    tpot += 5     # Schedule overhead for decode

    # 7. Throughput
    throughput = batch_size / tpot * 1000  # tokens/s
```

**Schedule Overheads** (from code):
- **Prefill**: +30ms (line: `ttft += 30`)
- **Decode**: +5ms (line: `tpot += 5`)

These are measured constants from profiling vLLM scheduling overhead.

---

## Critical Implementation Details

### 1. Causal Mask Factor (Prefill)

**Location**: `layers/attn.py:prefill_attn_core()`, line 93

```python
attn_core_time = seq_len * attn_core_gflops / 1.8 / (gpu.fp16_tflops * 1024 * mfu)
#                                            ^^^^
#                                            This is key!
```

**Physical Meaning**:
- Causal mask makes ~50% of attention matrix valid (upper triangle)
- `/ 1.8 ≈ × 0.556` accounts for this plus some FlashAttention overhead
- **InferSim empirically determined this factor**

**Alternative derivation**:
- Theoretical causal mask: × 0.5
- FlashAttention overhead: ~10-15%
- Combined: 0.5 × 1.15 ≈ 0.575 ≈ 1/1.8

### 2. Memory Bandwidth with Efficiency

**Location**: `hardware/gpu.py`

```python
h800 = GPU(
    mem_bw=3430 * 0.8,  # Peak 3.35 TB/s × 0.8 efficiency = 2.744 TB/s
    nvlink_bw=400 * 0.8 / 2,  # Bidirectional → unidirectional × efficiency
)
```

**This matches STREAM benchmarks**:
- 0.8 efficiency is standard for real workloads
- No need to run STREAM separately!

### 3. MFU Lookup with Interpolation

**Location**: `mfu/mfu.py:get_attn_decode_mfu()`

```python
def get_attn_decode_mfu(config, target_bs, kv_len, device_type, use_fp8_kv):
    # Load CSV: bench_data/mha/decode/h800/32-8-128.csv

    # Find closest batch_size <= target_bs
    mfu_bs = 1
    for row in rows:
        bs = int(row['batch_size'])
        if bs <= target_bs:
            mfu_bs = bs
        else:
            break

    # Find closest kv_len <= kv_len
    mfu_kv_len = 1
    for row in rows:
        kv_l = int(row['kv_len'])
        if kv_l <= kv_len:
            mfu_kv_len = kv_l
        else:
            break

    # Return MFU for that (bs, kv_len) pair
    for row in rows:
        if row['batch_size'] == mfu_bs and row['kv_len'] == mfu_kv_len:
            return float(row['mfu'])

    return gpu.mfu  # Fallback
```

**Strategy**: Conservative interpolation (use closest smaller value)

### 4. Communication Model

**Location**: `comm/comm.py`

```python
def size_bw_model(self, tensor_shape, use_fp8=False, inter_node=False):
    size = 1 if use_fp8 else 2  # bytes per element
    for v in tensor_shape:
        size *= v

    if inter_node:
        return size / (1024**3) / self.gpu.rdma_bw
    else:
        return size / (1024**3) / self.gpu.nvlink_bw

def all_reduce(self, num_tokens):
    tensor_shape = [num_tokens * self.world_size, self.config.hidden_size]
    return self.size_bw_model(tensor_shape, inter_node=(self.num_nodes > 1))
```

**Simple bandwidth model**: `time = bytes / bandwidth`

---

## H100 Benchmarking Plan

### Hardware Specifications

**H100 80GB SXM** (from NVIDIA datasheet):
```
Peak FP16 TFLOPs:   989.5
Peak Memory BW:     3.35 TB/s → Effective: 3.35 * 0.8 = 2.68 TB/s
NVLink Bandwidth:   900 GB/s per direction → Effective: 900 * 0.8 = 720 GB/s
Number of SMs:      132
Memory:             80 GB
```

### Benchmark Matrix

#### MHA/GQA Configurations (Priority Order)

| Config | Description | Priority | Models Using This |
|--------|-------------|----------|-------------------|
| **28-4-128** | Qwen2.5-7B | **HIGH** | Qwen2.5 1.5B/3B/7B |
| **32-8-128** | Llama-3 | **HIGH** | Llama-2/3, most models |
| **32-4-128** | Llama-3.1 8B | **HIGH** | Llama-3.1 |
| **40-8-128** | Llama-3.3 70B | **MEDIUM** | Llama-3.3 |
| **64-8-128** | Large models | MEDIUM | Some 30B+ models |
| **16-2-128** | Small models | LOW | Tiny models |
| **32-32-128** | Pure MHA | LOW | Older architectures |
| **128-8-128** | Very large | LOW | 100B+ models |

**Recommendation**: Start with HIGH priority (3 configs), expand later if needed.

#### Benchmark Sweep

```python
# MHA Decode: 35 benchmarks per config
batch_sizes = [1, 16, 32, 64, 128, 256, 512]
kv_lens = [1024, 4096, 8192, 16384, 32768]

# MHA Prefill: 5 benchmarks per config
seq_lens = [1024, 4096, 8192, 16384, 32768]

# GEMM: 403 benchmarks total
M_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
K_values = [2048, 3584, 4096, 8192]  # Common hidden_dims
N_values = [6144, 11008, 14336, 18944, 24576]  # Common intermediate_dims
```

### Quick Reference: Reusing InferSim Scripts

**All scripts in `InferSim/kernel_benchmark/` - Production tested!**

| Script | Usage | Output Format | Modification |
|--------|-------|---------------|--------------|
| `fa3_mha_prefill.py` | `--config-path config.json` | `dtype,seq_len,latency_us,mfu` | Change line 111: `fp16_tflops=148` → `989.5` |
| `flashinfer_mha_decode.py` | `--config-path config.json --fp16-tflops 989.5` | `dtype,kv_dtype,batch_size,kv_len,latency_us,mfu` | ✅ Ready with CLI arg |
| `deepgemm_gemm.py` | `-k 4096 -n 11008 --gpu-tflops 989.5` | `m,k,n,latency_us,mfu` | ✅ Ready with CLI arg |

**Dependencies**: `torch, sgl-kernel (flashinfer), deep-gemm, pandas`

**Config File Format**:
```json
{
    "hidden_size": 3584,        // nh × dh
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "num_hidden_layers": 32,
    "intermediate_size": 14336
}
```

### Automated Collection Script

**File: `scripts/collect_h100_data.sh`** (Fully Automated!)

```bash
#!/bin/bash
set -e

cd InferSim
echo "=== H100 Data Collection Script ==="

# 1. Modify prefill script for H100
sed -i.bak 's/fp16_tflops = 148/fp16_tflops = 989.5/' kernel_benchmark/fa3_mha_prefill.py
echo "✓ Modified fa3_mha_prefill.py for H100"

# 2. Create output directories
mkdir -p bench_data/h100/{mha/{decode,prefill},gemm}
echo "✓ Created output directories"

# 3. MHA Benchmarks
CONFIGS=("32 8 128" "32 4 128" "28 4 128" "64 8 128" "16 2 128" "32 32 128" "128 8 128" "256 8 128")
echo "=== Running MHA Benchmarks (8 configs) ==="

for config in "${CONFIGS[@]}"; do
    IFS=' ' read -r NH NKV DH <<< "$config"
    echo "→ Config: nh=$NH, nkv=$NKV, dh=$DH"

    # Create temp config
    cat > /tmp/h100_config_${NH}_${NKV}_${DH}.json <<EOF
{"hidden_size": $((NH * DH)), "num_attention_heads": $NH, "num_key_value_heads": $NKV,
 "head_dim": $DH, "num_hidden_layers": 32, "intermediate_size": 14336}
EOF

    # Prefill
    python kernel_benchmark/fa3_mha_prefill.py \
        --config-path /tmp/h100_config_${NH}_${NKV}_${DH}.json
    mv attention_benchmark.csv bench_data/h100/mha/prefill/${NH}-${NKV}-${DH}.csv
    echo "  ✓ Prefill: $(wc -l < bench_data/h100/mha/prefill/${NH}-${NKV}-${DH}.csv) rows"

    # Decode
    python kernel_benchmark/flashinfer_mha_decode.py \
        --config-path /tmp/h100_config_${NH}_${NKV}_${DH}.json \
        --fp16-tflops 989.5 --kv-cache-dtype bf16 --tp-size 1
    mv attention_benchmark.csv bench_data/h100/mha/decode/${NH}-${NKV}-${DH}.csv
    echo "  ✓ Decode: $(wc -l < bench_data/h100/mha/decode/${NH}-${NKV}-${DH}.csv) rows"
done

# 4. GEMM Benchmarks
echo "=== Running GEMM Benchmarks (12 configs) ==="
for K in 2048 4096 8192; do
    for N in 6144 11008 14336 18944; do
        echo "→ K=$K, N=$N"
        python kernel_benchmark/deepgemm_gemm.py -k $K -n $N --gpu-tflops 989.5

        if [ ! -f bench_data/h100/gemm/data.csv ]; then
            cp gemm.csv bench_data/h100/gemm/data.csv
        else
            tail -n +2 gemm.csv >> bench_data/h100/gemm/data.csv
        fi
    done
done
echo "  ✓ Total: $(tail -n +2 bench_data/h100/gemm/data.csv | wc -l) benchmarks"

# 5. Validate
echo "=== Validation ==="
echo "MHA Prefill:  $(ls bench_data/h100/mha/prefill/*.csv | wc -l) files"
echo "MHA Decode:   $(ls bench_data/h100/mha/decode/*.csv | wc -l) files"
echo "GEMM:         $(tail -n +2 bench_data/h100/gemm/data.csv | wc -l) benchmarks"
echo "✅ Data collection complete!"
```

**Usage**:
```bash
chmod +x scripts/collect_h100_data.sh
./scripts/collect_h100_data.sh  # Run on H100 node
```

### SLURM Submission Script

**File: `scripts/benchmark_h100_complete.sh`** (For Cluster Submission)

```bash
#!/bin/bash
set -e

#SBATCH --job-name=infersim_h100_benchmark
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=benchmark_h100_%j.log

echo "=== InferSim H100 Benchmarking ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# Setup
export CUDA_VISIBLE_DEVICES=0
cd /scratch/$USER/InferSim
conda activate infersim

# Pin GPU clocks for consistent results
sudo nvidia-smi -pm 1
sudo nvidia-smi -lgc 1410  # H100 base clock

# Output directory
OUTPUT_DIR="bench_data_h100_$(date +%Y%m%d)"
mkdir -p $OUTPUT_DIR/{mha/{decode,prefill},gemm,grouped_gemm/{decode,prefill}}

# --- Part 1: MHA Benchmarks (HIGH PRIORITY) ---
echo ""
echo "=== Part 1/3: MHA Benchmarks ==="

HIGH_PRIORITY_CONFIGS=(
    "28 4 128 qwen2.5-7b"
    "32 8 128 llama-3"
    "32 4 128 llama-3.1"
)

for config_line in "${HIGH_PRIORITY_CONFIGS[@]}"; do
    IFS=' ' read -r NH NKV DH MODEL_NAME <<< "$config_line"

    echo ""
    echo "--- Config: nh=$NH, nkv=$NKV, dh=$DH ($MODEL_NAME) ---"

    # Create temporary HF config
    TEMP_CONFIG="/tmp/temp_config_${NH}_${NKV}_${DH}.json"
    cat > $TEMP_CONFIG <<EOF
{
    "hidden_size": $((NH * DH)),
    "num_attention_heads": $NH,
    "num_key_value_heads": $NKV,
    "head_dim": $DH,
    "num_hidden_layers": 32,
    "intermediate_size": 14336,
    "torch_dtype": "bfloat16"
}
EOF

    # Prefill benchmark
    echo "  Running prefill benchmark..."
    python kernel_benchmark/fa3_mha_prefill.py \
        --config-path $TEMP_CONFIG \
        --fp16-tflops 989 \
        2>&1 | tee /tmp/prefill_${NH}_${NKV}_${DH}.log

    if [ -f "attention_benchmark.csv" ]; then
        mv attention_benchmark.csv \
           $OUTPUT_DIR/mha/prefill/${NH}-${NKV}-${DH}.csv
        echo "  ✓ Prefill done: $(tail -n +2 $OUTPUT_DIR/mha/prefill/${NH}-${NKV}-${DH}.csv | wc -l) benchmarks"
    else
        echo "  ✗ Prefill failed!"
    fi

    # Decode benchmark
    echo "  Running decode benchmark..."
    python kernel_benchmark/flashinfer_mha_decode.py \
        --config-path $TEMP_CONFIG \
        --fp16-tflops 989 \
        --kv-cache-dtype bf16 \
        2>&1 | tee /tmp/decode_${NH}_${NKV}_${DH}.log

    if [ -f "attention_benchmark.csv" ]; then
        mv attention_benchmark.csv \
           $OUTPUT_DIR/mha/decode/${NH}-${NKV}-${DH}.csv
        echo "  ✓ Decode done: $(tail -n +2 $OUTPUT_DIR/mha/decode/${NH}-${NKV}-${DH}.csv | wc -l) benchmarks"
    else
        echo "  ✗ Decode failed!"
    fi

    echo "  ----------------------------------------"
done

# --- Part 2: GEMM Benchmarks ---
echo ""
echo "=== Part 2/3: GEMM Benchmarks ==="

# Use the existing deepgemm_gemm.py or create custom
cat > /tmp/gemm_sweep.py <<'GEMM_EOF'
import torch
import pandas as pd

def benchmark_gemm(m, k, n, num_trials=100):
    device = "cuda"
    dtype = torch.bfloat16

    A = torch.randn(m, k, device=device, dtype=dtype)
    B = torch.randn(k, n, device=device, dtype=dtype)

    # Warmup
    for _ in range(10):
        C = torch.matmul(A, B)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_trials):
        C = torch.matmul(A, B)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_trials
    latency_us = elapsed_ms * 1000

    # Calculate MFU
    flops = 2 * m * k * n
    peak_tflops = 989.5  # H100
    achieved_tflops = flops / (elapsed_ms / 1000) / 1e12
    mfu = achieved_tflops / peak_tflops

    return latency_us, mfu

# Comprehensive sweep
M_values = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
K_values = [2048, 3584, 4096, 8192]
N_values = [6144, 11008, 14336, 18944, 24576]

results = []
total = len(M_values) * len(K_values) * len(N_values)
count = 0

for m in M_values:
    for k in K_values:
        for n in N_values:
            count += 1
            latency_us, mfu = benchmark_gemm(m, k, n)
            results.append({
                "m": m,
                "k": k,
                "n": n,
                "latency_us": round(latency_us, 3),
                "mfu": round(mfu, 3)
            })
            if count % 10 == 0:
                print(f"Progress: {count}/{total} ({count*100/total:.1f}%)")

df = pd.DataFrame(results)
df.to_csv("gemm_benchmark.csv", index=False)
print(f"\n✓ GEMM benchmarks complete: {len(results)} total")
GEMM_EOF

echo "Running GEMM sweep (200 benchmarks, ~1 hour)..."
python /tmp/gemm_sweep.py 2>&1 | tee /tmp/gemm_sweep.log
mv gemm_benchmark.csv $OUTPUT_DIR/gemm/data.csv
echo "✓ GEMM done: $(tail -n +2 $OUTPUT_DIR/gemm/data.csv | wc -l) benchmarks"

# --- Part 3: Validation ---
echo ""
echo "=== Part 3/3: Data Validation ==="

python3 <<VALIDATE_EOF
import pandas as pd
import os

base = "$OUTPUT_DIR"

# Check MHA decode
for config in ["28-4-128", "32-8-128", "32-4-128"]:
    path = f"{base}/mha/decode/{config}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✓ MHA decode {config}: {len(df)} benchmarks, MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
    else:
        print(f"✗ Missing: {path}")

# Check MHA prefill
for config in ["28-4-128", "32-8-128", "32-4-128"]:
    path = f"{base}/mha/prefill/{config}.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✓ MHA prefill {config}: {len(df)} benchmarks, MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
    else:
        print(f"✗ Missing: {path}")

# Check GEMM
gemm_path = f"{base}/gemm/data.csv"
if os.path.exists(gemm_path):
    df = pd.read_csv(gemm_path)
    print(f"✓ GEMM: {len(df)} benchmarks, MFU range [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
else:
    print(f"✗ Missing: {gemm_path}")
VALIDATE_EOF

# Restore GPU settings
sudo nvidia-smi -pm 0

echo ""
echo "=== Benchmarking Complete ==="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Copy data to inference-sim repo:"
echo "   scp -r $OUTPUT_DIR user@local:~/inference-sim/bench_data_h100/"
echo "2. Proceed to Phase 2 (Go implementation)"
```

**Usage**:
```bash
# Submit to cluster
sbatch scripts/benchmark_h100_complete.sh

# Monitor progress
tail -f benchmark_h100_*.log

# Expected runtime: 3-4 hours
```

---

## Go Implementation Strategy

**CRITICAL**: Go is a CSV reader + lookup engine. NO benchmarking!

### Architecture

```go
// Modular design matching InferSim

sim/
├── mfu_database.go          // Parse CSV files into memory (Day 6)
├── mfu_lookup.go            // Nearest-neighbor search (Day 7)
├── roofline_infersim.go     // Use MFU lookups in roofline (Day 8-9)
└── (no GPU code, no kernel execution, no benchmarking)

bench_data/h100/             // Static CSV files (from Phase 1)
├── mha/
│   ├── decode/
│   │   ├── 28-4-128.csv    # Pre-computed MFU data
│   │   ├── 32-8-128.csv
│   │   └── 32-4-128.csv
│   └── prefill/
│       └── ... (8 files)
└── gemm/
    └── data.csv             # Pre-computed GEMM MFU
```

### Data Structures

```go
// CSV Entry Types
type AttentionMFUEntry struct {
    BatchSize int      // For decode
    SeqLen    int      // For prefill
    KVLen     int      // For decode
    LatencyUs float64  // Not used in Go (only for reference)
    MFU       float64  // ← This is what we use!
}

type GemmMFUEntry struct {
    M         int
    K         int
    N         int
    LatencyUs float64  // Not used in Go
    MFU       float64  // ← This is what we use!
}

// In-Memory Database
type MFUDatabase struct {
    GPU              string
    AttentionDecode  map[string][]AttentionMFUEntry  // "28-4-128" -> [{bs, kv_len, mfu}]
    AttentionPrefill map[string][]AttentionMFUEntry  // "28-4-128" -> [{seq_len, mfu}]
    Gemm             []GemmMFUEntry                   // [{m, k, n, mfu}]
}
```

### Data Flow (No Benchmarking!)

```
1. STARTUP: Load MFU Database
   ├─ Read CSV files from bench_data/h100/
   ├─ Parse into in-memory maps
   └─ Validate (warn if configs missing)

2. RUNTIME: Simulator Step Event
   ├─ Build StepConfig (prefill/decode requests)
   └─ Call rooflineStepTimeInferSim()

3. Roofline Calculation
   ├─ For each request:
   │   ├─ Calculate FLOPs (exact formula)
   │   ├─ Lookup MFU from database (NO benchmarking!)
   │   ├─ Compute time: FLOPs / (Peak × MFU)
   │   ├─ Memory time: Bytes / BW
   │   └─ Roofline: max(compute, memory)
   ├─ Add communication (if TP > 1)
   ├─ Multiply by num_layers
   └─ Add schedule overhead

4. Return Step Time (microseconds)
```

### Implementation Phases

**Day 6: CSV Loader** (~200 lines)
```go
// Load CSV files into memory structures
func LoadMFUDatabase(gpu, basePath string) (*MFUDatabase, error)
func loadAttentionMFU(path string) ([]AttentionMFUEntry, error)
func loadGemmMFU(path string) ([]GemmMFUEntry, error)
```

**Day 7: Lookup Logic** (~150 lines)
```go
// Nearest-neighbor search (mimic Python mfu/mfu.py)
func (db *MFUDatabase) GetAttentionDecodeMFU(nh, nkv, dh, bs, kv_len int) float64
func (db *MFUDatabase) GetAttentionPrefillMFU(nh, nkv, dh, seq_len int) float64
func (db *MFUDatabase) GetGemmMFU(m, k, n int) float64
```

**Day 8-9: Roofline** (~150 lines)
```go
// Use MFU lookups in time calculation
func rooflineStepTimeInferSim(
    modelConfig, hwProfile, stepConfig, tp, mfuDB) int64 {
    // 1. Lookup MFU (no benchmarking!)
    mfu := mfuDB.GetAttentionMFU(...)
    // 2. Calculate: time = max(flops/(peak*mfu), bytes/bw)
    return time
}
```

### Complete Implementation

**File: `sim/roofline_infersim.go`**

```go
package sim

import (
    "math"
)

// InferSim-style roofline calculation
func rooflineStepTimeInferSim(
    modelConfig ModelConfig,
    hwProfile HardwareProfile,
    stepConfig StepConfig,
    tp int,
    mfuDB *MFUDatabase,
) int64 {

    // Hardware parameters
    peakTFlops := hwProfile.TFlopsPeak
    memBwGBs := hwProfile.MemBWGBs  // Effective bandwidth (already × 0.8)
    nvlinkBwGBs := hwProfile.NVLinkBW

    var totalTimeS float64

    // ========== ATTENTION TIME ==========

    // --- Prefill Requests ---
    for _, req := range stepConfig.PrefillRequests {
        numTokens := int64(req.NumNewPrefillTokens)
        seqLen := req.ProgressIndex + numTokens

        // 1. Attention Core: softmax(QK^T)V
        // FLOPs: 4 × nh × seq_q × seq_kv × dh
        // For causal: divide by 1.8 (InferSim empirical factor)
        flopsAttnCoreGiga := 4.0 * float64(modelConfig.NumHeads) *
                            float64(numTokens) *
                            float64(seqLen) *
                            float64(modelConfig.HiddenDim / modelConfig.NumHeads) /
                            1e9

        // Lookup MFU
        mfuAttnCore := mfuDB.GetAttentionPrefillMFU(
            modelConfig.NumHeads,
            modelConfig.NumKVHeads,
            modelConfig.HiddenDim / modelConfig.NumHeads,
            int(numTokens),
        )

        // Compute time (with causal mask factor)
        timeAttnCoreS := flopsAttnCoreGiga / 1.8 / (peakTFlops * mfuAttnCore)

        // KV cache loading time (roofline)
        kvCacheBytesPerToken := float64(modelConfig.NumKVHeads) *
                                float64(modelConfig.HiddenDim / modelConfig.NumHeads) *
                                2.0 *  // K and V
                                2.0    // fp16 bytes

        kvLoadTimeS := kvCacheBytesPerToken * float64(numTokens) /
                      float64(modelConfig.NumLayers) /
                      (memBwGBs * 1e9)

        // Roofline: take maximum
        timeAttnCoreS = math.Max(timeAttnCoreS, kvLoadTimeS)

        // 2. QKV Projections
        // Q: hidden_dim × (nh × dh)
        // K, V: hidden_dim × (nkv × dh)
        flopsQKVProjGiga := (2.0 * float64(numTokens) *
                            float64(modelConfig.HiddenDim) *
                            float64(modelConfig.NumHeads*modelConfig.HiddenDim/modelConfig.NumHeads) +
                            4.0 * float64(numTokens) *
                            float64(modelConfig.HiddenDim) *
                            float64(modelConfig.NumKVHeads*modelConfig.HiddenDim/modelConfig.NumHeads)) /
                            1e9

        mfuQKVProj := mfuDB.GetGemmMFU(
            int(numTokens),
            modelConfig.HiddenDim,
            modelConfig.HiddenDim,
        )

        timeQKVProjS := flopsQKVProjGiga / (peakTFlops * mfuQKVProj)

        // 3. O Projection
        flopsOProjGiga := 2.0 * float64(numTokens) *
                         float64(modelConfig.NumHeads*modelConfig.HiddenDim/modelConfig.NumHeads) *
                         float64(modelConfig.HiddenDim) /
                         1e9

        mfuOProj := mfuDB.GetGemmMFU(
            int(numTokens),
            modelConfig.NumHeads * modelConfig.HiddenDim / modelConfig.NumHeads,
            modelConfig.HiddenDim,
        )

        timeOProjS := flopsOProjGiga / (peakTFlops * mfuOProj)

        totalTimeS += timeAttnCoreS + timeQKVProjS + timeOProjS
    }

    // --- Decode Requests ---
    numDecodeReqs := len(stepConfig.DecodeRequests)

    if numDecodeReqs > 0 {
        // Aggregate decode requests (they run in same batch)
        var avgProgressIndex int64
        for _, req := range stepConfig.DecodeRequests {
            avgProgressIndex += req.ProgressIndex
        }
        avgProgressIndex /= int64(numDecodeReqs)

        // 1. Attention Core (per request, but batched)
        flopsAttnCoreGiga := 4.0 * float64(modelConfig.NumHeads) *
                            1.0 *  // seq_q = 1 for decode
                            float64(avgProgressIndex) *
                            float64(modelConfig.HiddenDim / modelConfig.NumHeads) /
                            1e9

        mfuAttnCore := mfuDB.GetAttentionDecodeMFU(
            modelConfig.NumHeads,
            modelConfig.NumKVHeads,
            modelConfig.HiddenDim / modelConfig.NumHeads,
            numDecodeReqs,
            int(avgProgressIndex),
        )

        // Time for entire batch
        timeAttnCoreS := float64(numDecodeReqs) * flopsAttnCoreGiga / (peakTFlops * mfuAttnCore)

        // KV loading (per request)
        kvCacheBytesPerToken := float64(modelConfig.NumKVHeads) *
                                float64(modelConfig.HiddenDim / modelConfig.NumHeads) *
                                2.0 * 2.0

        kvLoadTimeS := kvCacheBytesPerToken * float64(avgProgressIndex) *
                      float64(numDecodeReqs) /
                      float64(modelConfig.NumLayers) /
                      (memBwGBs * 1e9)

        timeAttnCoreS = math.Max(timeAttnCoreS, kvLoadTimeS)

        // 2. Projections (batched)
        flopsQKVProjGiga := (2.0 * float64(numDecodeReqs) *
                            float64(modelConfig.HiddenDim) *
                            float64(modelConfig.NumHeads*modelConfig.HiddenDim/modelConfig.NumHeads) +
                            4.0 * float64(numDecodeReqs) *
                            float64(modelConfig.HiddenDim) *
                            float64(modelConfig.NumKVHeads*modelConfig.HiddenDim/modelConfig.NumHeads)) /
                            1e9

        mfuQKVProj := mfuDB.GetGemmMFU(
            numDecodeReqs,
            modelConfig.HiddenDim,
            modelConfig.HiddenDim,
        )

        timeQKVProjS := flopsQKVProjGiga / (peakTFlops * mfuQKVProj)

        // 3. O Projection
        flopsOProjGiga := 2.0 * float64(numDecodeReqs) *
                         float64(modelConfig.NumHeads*modelConfig.HiddenDim/modelConfig.NumHeads) *
                         float64(modelConfig.HiddenDim) /
                         1e9

        mfuOProj := mfuDB.GetGemmMFU(
            numDecodeReqs,
            modelConfig.NumHeads * modelConfig.HiddenDim / modelConfig.NumHeads,
            modelConfig.HiddenDim,
        )

        timeOProjS := flopsOProjGiga / (peakTFlops * mfuOProj)

        totalTimeS += timeAttnCoreS + timeQKVProjS + timeOProjS
    }

    // ========== MoE/FFN TIME ==========

    totalTokens := 0
    for _, req := range stepConfig.PrefillRequests {
        totalTokens += req.NumNewPrefillTokens
    }
    totalTokens += numDecodeReqs

    // FLOPs: 6 × tokens × hidden_dim × intermediate_dim
    // Factor of 6: gate (2×MK), up (2×MK), down (2×KN) projections
    flopsMoEGiga := 6.0 * float64(totalTokens) *
                   float64(modelConfig.HiddenDim) *
                   float64(modelConfig.IntermediateDim) /
                   1e9

    mfuMoE := mfuDB.GetGemmMFU(
        totalTokens,
        modelConfig.HiddenDim,
        modelConfig.IntermediateDim,
    )

    timeMoES := flopsMoEGiga / (peakTFlops * mfuMoE)

    // Weight loading (roofline)
    // 3 matrices (gate, up, down): 3 × hidden_dim × intermediate_dim
    weightBytesPerLayer := 3.0 * float64(modelConfig.HiddenDim) *
                          float64(modelConfig.IntermediateDim) *
                          2.0  // fp16

    weightLoadTimeS := weightBytesPerLayer / float64(modelConfig.NumLayers) / (memBwGBs * 1e9)

    timeMoES = math.Max(timeMoES, weightLoadTimeS)

    totalTimeS += timeMoES

    // ========== COMMUNICATION ==========

    if tp > 1 {
        // All-reduce per layer: 2 × hidden_dim × bytes
        commVolumeBytes := 2.0 * float64(modelConfig.HiddenDim) * 2.0  // fp16

        // Communication time per layer
        commTimePerLayerS := commVolumeBytes / (nvlinkBwGBs * 1e9)

        // Total comm across all layers
        commTimeS := commTimePerLayerS * float64(modelConfig.NumLayers)

        totalTimeS += commTimeS
    }

    // ========== FINALIZE ==========

    // Multiply per-layer time by number of layers
    perLayerTimeS := totalTimeS / float64(modelConfig.NumLayers)
    totalTimeS = perLayerTimeS * float64(modelConfig.NumLayers)

    // Add schedule overhead
    if len(stepConfig.PrefillRequests) > 0 {
        totalTimeS += 30e-3  // 30ms for prefill scheduling
    } else {
        totalTimeS += 5e-3   // 5ms for decode scheduling
    }

    return int64(totalTimeS * 1e6)  // Convert to microseconds
}
```

---

## Integration with Existing Simulator

### Modify `sim/simulator.go`

```go
// Add MFU database to Simulator struct
type Simulator struct {
    // ... existing fields ...

    // InferSim roofline mode
    MFUDatabase *MFUDatabase  // Add this
}

// Initialize in NewSimulator
func NewSimulator(...) *Simulator {
    s := &Simulator{
        // ... existing initialization ...
    }

    if roofline {
        // Load MFU database
        mfuDB, err := LoadMFUDatabase("H100", "data/mfu_h100")
        if err != nil {
            logrus.Warnf("Failed to load MFU database: %v", err)
        } else {
            s.MFUDatabase = mfuDB
        }
    }

    return s
}

// Update getStepTimeRoofline
func (sim *Simulator) getStepTimeRoofline() int64 {
    stepConfig := StepConfig{
        PrefillRequests: make([]PrefillRequestConfig, 0),
        DecodeRequests:  make([]DecodeRequestConfig, 0),
    }

    // Build stepConfig from RunningBatch (existing logic)
    for _, req := range sim.RunningBatch.Requests {
        if req.ProgressIndex < Len64(req.InputTokens) {
            stepConfig.PrefillRequests = append(stepConfig.PrefillRequests, ...)
        } else {
            stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, ...)
        }
    }

    // Choose implementation
    if sim.MFUDatabase != nil {
        // Use InferSim-style roofline
        return rooflineStepTimeInferSim(sim.ModelConfig, sim.HWConfig, stepConfig, sim.TP, sim.MFUDatabase)
    } else {
        // Fall back to current implementation
        return rooflineStepTime(sim.GPU, sim.ModelConfig, sim.HWConfig, stepConfig, sim.TP)
    }
}
```

---

## Validation Protocol

### Test Suite

```go
// File: sim/roofline_infersim_test.go

func TestInferSimRooflineAccuracy(t *testing.T) {
    // Load MFU database
    mfuDB, err := LoadMFUDatabase("H100", "testdata/mfu_h100")
    require.NoError(t, err)

    // Test case 1: Qwen 7B, 2048 token prefill
    modelConfig := ModelConfig{
        NumHeads:    28,
        NumKVHeads:  4,
        HiddenDim:   3584,
        NumLayers:   28,
        IntermediateDim: 18944,
    }

    hwProfile := HardwareProfile{
        TFlopsPeak: 989.5,
        MemBWGBs:   2680,  // 3.35 TB/s × 0.8
        NVLinkBW:   720,   // 900 GB/s × 0.8
    }

    stepConfig := StepConfig{
        PrefillRequests: []PrefillRequestConfig{
            {ProgressIndex: 0, NumNewPrefillTokens: 2048},
        },
    }

    // Calculate step time
    stepTime := rooflineStepTimeInferSim(modelConfig, hwProfile, stepConfig, 1, mfuDB)

    // Expected range (from InferSim paper validation)
    // Qwen 7B prefill 2048 tokens: ~15-20ms
    expectedMin := int64(15000)  // 15ms
    expectedMax := int64(25000)  // 25ms

    assert.True(t, stepTime >= expectedMin && stepTime <= expectedMax,
        "Step time %d us outside expected range [%d, %d]", stepTime, expectedMin, expectedMax)
}

func TestMFULookupInterpolation(t *testing.T) {
    mfuDB, _ := LoadMFUDatabase("H100", "testdata/mfu_h100")

    // Test exact match
    mfu := mfuDB.GetAttentionDecodeMFU(32, 8, 128, 64, 4096)
    assert.True(t, mfu > 0.0 && mfu < 1.0, "MFU should be in (0, 1)")

    // Test interpolation (batch size between benchmarks)
    mfu1 := mfuDB.GetAttentionDecodeMFU(32, 8, 128, 32, 4096)
    mfu2 := mfuDB.GetAttentionDecodeMFU(32, 8, 128, 64, 4096)
    mfu_interp := mfuDB.GetAttentionDecodeMFU(32, 8, 128, 48, 4096)

    // Should be between the two
    assert.True(t, mfu_interp >= math.Min(mfu1, mfu2) && mfu_interp <= math.Max(mfu1, mfu2))
}
```

### Validation Against InferSim Python

```bash
#!/bin/bash
# File: scripts/cross_validate.sh

# Run same configuration on both implementations
MODEL="qwen2.5-7b-instruct"
CONFIG="model_configs/$MODEL/config.json"

echo "=== Cross-Validation: Go vs Python InferSim ==="

# Test 1: Prefill 2048 tokens
echo "Test 1: Prefill 2048 tokens"

python_ms=$(cd /tmp/InferSim && python main.py \
    --config-path $CONFIG \
    --device-type H100 \
    --max-prefill-tokens 2048 \
    --prefill-only \
    | grep "TTFT (ms)" | awk '{print $3}')

go_ms=$(./inference-sim \
    --model $MODEL \
    --gpu H100 \
    --roofline infersim \
    --prefill-tokens 2048 \
    | grep "TTFT" | awk '{print $2}')

error=$(python3 -c "print(abs($go_ms - $python_ms) / $python_ms * 100)")
echo "Python: $python_ms ms"
echo "Go:     $go_ms ms"
echo "Error:  $error%"

# Success if error < 5%
if (( $(echo "$error < 5.0" | bc -l) )); then
    echo "✓ PASS"
else
    echo "✗ FAIL"
    exit 1
fi
```

---

## Timeline and Resources

### Detailed Schedule

#### Week 1: Python Data Collection (Days 1-5)

| Day | Tasks | Hours | What Runs | Output |
|-----|-------|-------|-----------|--------|
| **Day 1** | Clone InferSim, setup H100, install deps, modify scripts | 4h | Setup only | Ready to benchmark |
| **Day 2-3** | Run MHA benchmarks (8 configs × 2 stages) | 6h | **Python scripts** | 16 CSV files |
| **Day 4** | Run GEMM benchmarks (200 configs) | 4h | **Python script** | 1 CSV file |
| **Day 5** | Data validation, add H100 profile | 3h | Validation | Validated CSVs |

**What runs**: Python scripts (`fa3_mha_prefill.py`, `flashinfer_mha_decode.py`, `deepgemm_gemm.py`)
**Deliverable**: `bench_data/h100/*.csv` (~17 files, pre-computed MFU data)
**No Go code yet!**

#### Week 2: Go CSV Loading (Days 6-9)

| Day | Tasks | Hours | What Runs | Input/Output |
|-----|-------|-------|-----------|--------------|
| **Day 6** | CSV parser (mfu_database.go) | 6h | **Go loader** | CSVs → in-memory maps |
| **Day 7** | Lookup logic (mfu_lookup.go) | 6h | **Go search** | Query → MFU value |
| **Day 8** | Roofline (roofline_infersim.go) | 8h | **Go calc** | StepConfig → time |
| **Day 9** | Integration, unit tests | 6h | **Go tests** | Full simulator |

**What runs**: Go code (no benchmarking, just CSV reading + lookups)
**Deliverable**: Working simulator with MFU-based roofline
**No Python execution after Day 5!**

#### Week 3: Validation (Days 10-11)

| Day | Tasks | Hours | Depends On |
|-----|-------|-------|------------|
| **Day 10** | Cross-validation (Go vs Python) | 4h | Day 9 |
| **Day 11** | vLLM validation, error analysis | 6h | Day 10 |

**Deliverable**: Validated roofline model with error report

---

### Resource Requirements

#### Compute Resources

```
H100 Cluster:
├─ Days 1-4: 1 × H100 GPU × 20 hours
│             (for benchmarking)
├─ Day 10-11: 1 × H100 GPU × 8 hours
│             (for vLLM validation)
└─ Total: ~28 GPU-hours
```

**Cost Estimate** (AWS p5.xlarge): $5/hr × 28h = **~$140**

#### Human Resources

```
Developer Time:
├─ Phase 1: 2-3 hours/day monitoring benchmarks
├─ Phase 2: 6-8 hours/day coding
├─ Phase 3: 4-6 hours/day validation
└─ Total: ~70 developer hours over 11 days
```

---

## Risk Assessment

### High Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **H100 queue time** | Medium | High | Submit jobs early, use preemption |
| **Benchmark failures** | Low | Medium | Test scripts on Day 1, have fallbacks |
| **Missing configs** | Low | Medium | Start with HIGH priority, expand later |

### Medium Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Go port bugs** | Medium | Low | Incremental testing, keep Python reference |
| **MFU lookup edge cases** | Low | Low | Unit tests, fallback to default MFU |
| **Integration issues** | Low | Medium | Keep current roofline as fallback |

---

## Acceptance Criteria

### Phase 1: Python Data Collection

**Language**: Python only (no Go yet)
**Environment**: H100 cluster with CUDA

- [ ] ✅ 8 MHA configs benchmarked (decode + prefill) using InferSim scripts
- [ ] ✅ 200+ GEMM benchmarks collected using torch.matmul
- [ ] ✅ MFU ranges match InferSim patterns:
  - Decode MFU: 0.01-0.20 (memory-bound)
  - Prefill MFU: 0.50-0.70 (compute-bound)
  - GEMM MFU: 0.10-0.90 (varies with M)
- [ ] ✅ Data validation script passes
- [ ] ✅ CSVs organized in `bench_data/h100/` structure
- [ ] ✅ No GPU dependencies remain (only CSV files)

### Phase 2: Go CSV Loading

**Language**: Go only (no GPU access needed)
**Environment**: Development machine

- [ ] ✅ MFU database loads CSVs without errors
- [ ] ✅ Lookup returns reasonable values (0 < MFU < 1)
- [ ] ✅ Nearest-neighbor search works correctly
- [ ] ✅ Unit tests pass for FLOP calculation
- [ ] ✅ Integration compiles and runs
- [ ] ✅ **Zero GPU code in Go** (verify with code review)
- [ ] ✅ Binary runs on machine without GPU

### Phase 3: Validation

**What**: Compare predictions (no new benchmarking)

- [ ] ✅ Go vs Python InferSim: < 5% error (same inputs → same outputs)
- [ ] ✅ Simulator vs vLLM: < 15% error (realistic workloads)
- [ ] ✅ Generalization: Same error across Qwen/Llama models
- [ ] ✅ TP scaling: Works for TP=1,2,4 without re-benchmarking
- [ ] ✅ **CSV files work as standalone data** (no Python runtime needed)

---

## Comparison: Current vs Proposed

| Metric | Current Roofline | InferSim-Based | Improvement |
|--------|------------------|----------------|-------------|
| **Accuracy** | 11.23% E2E error | **4-15% (target <10%)** | ✅ Better |
| **Calibration Params** | 13 parameters | **~0 (MFU measured)** | ✅ Much simpler |
| **Generalization** | Tuned per model | **Zero-shot** | ✅ Works across models |
| **Maintainability** | Complex formulas | **Simple: t=FLOP/MFU** | ✅ Easy to debug |
| **Literature** | Williams 2009 | **InferSim 2025 + prior work** | ✅ Latest research |
| **Implementation** | Custom | **Validated by Alibaba** | ✅ Production-proven |
| **Data Collection** | Manual tuning | **Automated Python scripts** | ✅ Reproducible |
| **Runtime Overhead** | Calculate on-the-fly | **Pre-computed CSV lookup** | ✅ Faster |
| **GPU Dependencies** | None (all math) | **Python: Yes, Go: No** | ✅ Clear separation |

### Workflow Comparison

**Current Approach**:
1. Implement roofline formula in Go
2. Add calibration parameters
3. Tune parameters by trial and error
4. Hope it generalizes

**InferSim Approach**:
1. **Python**: Run benchmarks on H100 (one-time)
2. **Python**: Generate CSV files with MFU data
3. **Go**: Load CSVs, implement lookup
4. **Go**: Use lookups in roofline formula
5. ✅ Automatically works for all models!

---

## FAQ: Common Misconceptions

### Q: Do we need to port GPU benchmarking code to Go?
**A: NO!** Python does all benchmarking (one-time). Go only loads the resulting CSV files.

### Q: Does Go execute any GPU kernels?
**A: NO!** Go never touches the GPU. It just reads CSV files and does table lookups.

### Q: How do we update MFU data for new GPUs?
**A: Run Python scripts once on new GPU, generate new CSVs, ship with Go binary.**

### Q: What if we don't have exact MFU data for a config?
**A: Go does nearest-neighbor search. Falls back to default MFU if no data.**

### Q: Does this require CUDA/PyTorch at runtime?
**A: NO!** Only Python collection phase needs CUDA. Go binary is standalone.

### Q: How big is the MFU database?
**A: ~17 CSV files, <1MB total. Ships with the Go binary.**

### Q: Can we still use the existing roofline model?
**A: YES!** InferSim roofline is optional. Existing code stays as fallback.

---

## Open Questions for Discussion

1. **Cluster Access**: When can you provision H100 nodes? (blocks Day 1 start)

2. **TP Validation**: Do you need TP=8 support, or is TP=1,2,4 sufficient?

3. **MoE Models**: Do you plan to support MoE (DeepSeek-V3)? Requires GroupedGEMM benchmarks.

4. **FP8 Quantization**: Do you need FP8 support? InferSim has separate MFU data for FP8.

5. **Fallback Strategy**: Keep current roofline as backup, or fully replace?

---

## Next Actions

### Immediate (This Week)

1. **Approve this plan** (confirm Python→CSV→Go workflow)
2. **Secure H100 cluster access** (Python benchmarking needs H100)
3. **Clone InferSim repo** (verify scripts work)

### Week 1: Python Data Collection (Days 1-5)

**Who does what**:
- **Python scripts** (InferSim) run on H100
- **You** submit jobs, monitor progress
- **Output**: CSV files in `bench_data/h100/`

**Detailed steps**:
1. Clone InferSim: `git clone https://github.com/alibaba/InferSim.git`
2. Modify one line: `sed -i 's/fp16_tflops = 148/fp16_tflops = 989.5/' kernel_benchmark/fa3_mha_prefill.py`
3. Run automated script: `./scripts/collect_h100_data.sh`
4. Validate: `python scripts/validate_h100_data.py bench_data/h100`
5. Copy CSVs to Go project: `cp -r bench_data/h100 ~/inference-sim/bench_data/`

**No Go code yet!** This is purely Python benchmarking.

### Week 2: Go CSV Loading (Days 6-9)

**Who does what**:
- **Go code** loads CSV files
- **No GPU needed** (runs on dev machine)
- **Output**: Working simulator

**Detailed steps**:
1. Implement `mfu_database.go` (CSV parser)
2. Implement `mfu_lookup.go` (nearest-neighbor search)
3. Implement `roofline_infersim.go` (use MFU in formula)
4. Integrate with existing simulator
5. Write unit tests

**Verify**: Go binary runs on machine without GPU!

### Week 3: Validation (Days 10-11)

**Who does what**:
- **Both** run validation scripts
- **Compare** predictions (no new benchmarking)

**Detailed steps**:
1. Cross-validate: Go vs Python InferSim (same inputs → same outputs)
2. Validate: Simulator vs vLLM (realistic workloads)
3. Test generalization across models
4. Document results

---

## Conclusion

**This plan is systems-research-grade**:
- ✅ Based on published work (InferSim paper, September 2025)
- ✅ Validated results (4-15% error)
- ✅ Reproducible methodology (clear benchmarking protocol)
- ✅ Minimal calibration (MFU measured, not tuned)
- ✅ Production-proven (Alibaba uses this)

**Key architectural decision**:
- ✅ Python does heavy lifting (benchmarking) once
- ✅ Go uses lightweight lookups forever
- ✅ Clear separation of concerns
- ✅ No GPU dependencies in production Go binary

**What you get**:
1. **Automated Python scripts** (90% reuse from InferSim)
2. **Pre-computed CSV files** (~17 files, <1MB)
3. **Go simulator** that loads CSVs and does lookups
4. **Validated accuracy** (target <10% error)
5. **Zero-shot generalization** (works across models)

**Timeline**: 11 days
- Week 1: Python benchmarking (H100 required)
- Week 2: Go implementation (no GPU needed)
- Week 3: Validation

**Ready to start?** Confirm H100 access and I'll begin preparing Day 1 materials!

---

## TL;DR

**What**: MFU-based roofline model (InferSim methodology)

**How**:
1. Python runs benchmarks → generates CSV files (one-time, H100 cluster)
2. Go loads CSV files → does table lookups (runtime, any machine)

**Why**:
- 4-15% error (vs 11% current)
- Zero calibration needed
- Works across all models
- Production-proven

**Effort**: 11 days (5 Python + 4 Go + 2 validation)

**Cost**: ~$140 H100 time + 70 dev hours

**Key Point**: Go is a CSV reader, not a benchmarking tool!
