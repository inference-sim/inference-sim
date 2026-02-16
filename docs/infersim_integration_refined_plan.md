# InferSim Integration Plan

**Date**: 2025-02-16
**Timeline**: 11 days with H100 access
**Goal**: Port InferSim's MFU-based roofline model to Go for H100

## Summary

InferSim has H20/H800 data but **NOT H100**. Good news: **90% is reusable!**

### What We Reuse (✅ Ready)
1. **Benchmark scripts** - `fa3_mha_prefill.py`, `flashinfer_mha_decode.py`, `deepgemm_gemm.py`
   - Only need 1-line change to prefill script (`fp16_tflops=989.5`)
   - Others accept TFLOPS as CLI arg
2. **MFU lookup logic** - `mfu/mfu.py` works unchanged
3. **Data structure** - Same CSV format, just create `bench_data/h100/`
4. **FLOP formulas** - `flops/flops.py` unchanged
5. **Roofline model** - Core algorithm ready

### What We Need (⚠️ Action Required)
1. **Python: Run benchmarks on H100** - ~720 kernel runs (one-time, automated)
2. **Python: Add H100 GPU profile** - 15 lines in `hardware/gpu.py`
3. **Go: Load CSV files** - ~200 lines (parse CSVs into memory)
4. **Go: Lookup functions** - ~150 lines (nearest-neighbor search)
5. **Go: Roofline model** - ~150 lines (use MFU lookups)

**Key Insight**: Python does all heavy lifting (benchmarking). Go is just a CSV reader that does table lookups. No benchmarking code in Go!

### Division of Labor

| Language | What It Does | Size |
|----------|--------------|------|
| **Python** | Benchmarks kernels on H100 (one-time) | InferSim scripts (reuse) |
| **Python** | Generates CSVs with MFU data | ~720 rows across 17 files |
| **Go** | Loads CSVs into memory | ~200 lines |
| **Go** | Looks up MFU for operation shape | ~150 lines |
| **Go** | Roofline calculation using lookups | ~150 lines |

**Total Go code**: ~500 lines, all data-loading/lookup logic

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

## Three Phases

```
Phase 1: Collect H100 MFU Data (5 days) - Python benchmarking
Phase 2: Port Lookup to Go (4 days) - Load CSVs, no benchmarking
Phase 3: Validation (2 days) - Compare predictions
```

**IMPORTANT**:
- ✅ **Python does benchmarking** (Phase 1, one-time on H100)
- ✅ **Go loads static CSV files** (Phase 2, no benchmarking code)
- ✅ **CSVs are the lookup table** (bench_data/h100/*.csv)

## Architecture: Python Benchmarks → CSV → Go Lookups

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: Python Benchmarking (One-Time on H100)            │
├─────────────────────────────────────────────────────────────┤
│ InferSim Scripts:                                           │
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

## InferSim Core Concepts

### Architecture
- **Roofline model**: `time = max(compute_time, memory_time)`
- **MFU lookup**: CSV database indexed by operation shape (batch_size, seq_len, hidden_dim, etc.)
- **Separate tracking**: Attention core vs projections, prefill vs decode
- **Schedule overhead**: 30ms prefill, 5ms decode (empirically measured)

### Key Formulas
```python
# Attention: time = max(compute, kv_cache_loading)
attn_time = max(flops / (peak_tflops * mfu), kv_bytes / mem_bw)

# FFN/MoE: time = max(compute, weight_loading)
ffn_time = max(flops / (peak_tflops * mfu), weight_bytes / mem_bw)

# E2E: (attn + ffn + comm) × layers + overhead
```

## What InferSim Provides (✅ Ready to Reuse!)

### 1. Production Benchmark Scripts
**Location**: `InferSim/kernel_benchmark/`

All scripts output CSV files compatible with InferSim's data structure:

| Script | Usage | Output Format | Notes |
|--------|-------|---------------|-------|
| `fa3_mha_prefill.py` | `--config-path config.json` | `dtype,seq_len,latency_us,mfu` | **Needs edit**: Line 111 change `fp16_tflops=148` → `989.5` |
| `flashinfer_mha_decode.py` | `--config-path config.json --fp16-tflops 989.5 [--kv-cache-dtype bf16]` | `dtype,kv_dtype,batch_size,kv_len,latency_us,mfu` | ✅ Ready with arg |
| `deepgemm_gemm.py` | `-k 4096 -n 11008 --gpu-tflops 989.5` | `m,k,n,latency_us,mfu` | ✅ Ready with arg |

**Dependencies**: `torch, sgl-kernel (flashinfer), deep-gemm, pandas`

### 2. MFU Lookup Logic (`mfu/mfu.py`)
**Functions**:
- `get_attn_decode_mfu(config, bs, kv_len, device_type, use_fp8_kv)` - line 7
- `get_attn_prefill_mfu(config, seq_len, device_type)` - line 56
- `get_gemm_mfu(device_type, m, k, n)` - line 165

**Strategy**:
- Decode: Find closest `(batch_size <= target, kv_len <= target)`
- Prefill: Find closest `seq_len <= target`
- GEMM: Euclidean distance in (K,N) space, then find closest M
- Fallback to `gpu.mfu` default if file missing

### 3. Data Structure (`bench_data/`)
```
bench_data/{device_type}/      # "h100" for H100
├── mha/
│   ├── decode/*.csv           # Named: {nh}-{nkv}-{dh}.csv (e.g., 32-8-128.csv)
│   └── prefill/*.csv          # Named: {nh}-{nkv}-{dh}.csv
└── gemm/data.csv              # Single file with all M×K×N configs
```

**Existing GPUs**: H20 (148 TFLOPS), H800 (989 TFLOPS), H200, GB200

### 4. Hardware Profiles (`hardware/gpu.py`)
**Need to add H100**:
```python
h100 = GPU(
    fp16_tflops=989.5,        # Slightly higher than H800
    fp8_tflops=1979,
    mfu=0.35,                 # Default fallback
    mem=80,
    mem_bw=3350 * 0.8,        # 2680 GB/s effective
    nvlink_bw=900 * 0.8 / 2,  # 360 GB/s unidirectional
    rdma_bw=50 * 0.8,
)
gpu_map = {"H20": h20, "H800": h800, "H100": h100, ...}
```

## Deployment: What Ships with Go Binary

The Go simulator needs these static data files (collected once):

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

## Quick Reference: Script Usage

### Existing Scripts (InferSim/kernel_benchmark/)

**1. MHA Prefill** (`fa3_mha_prefill.py`)
```bash
# BEFORE: Edit line 111: fp16_tflops = 148 → 989.5
python fa3_mha_prefill.py --config-path config.json
# Output: attention_benchmark.csv → move to bench_data/h100/mha/prefill/{nh}-{nkv}-{dh}.csv
```

**2. MHA Decode** (`flashinfer_mha_decode.py`)
```bash
python flashinfer_mha_decode.py \
    --config-path config.json \
    --fp16-tflops 989.5 \
    --kv-cache-dtype bf16 \
    --tp-size 1
# Output: attention_benchmark.csv → move to bench_data/h100/mha/decode/{nh}-{nkv}-{dh}.csv
```

**3. GEMM** (`deepgemm_gemm.py`)
```bash
python deepgemm_gemm.py -k 4096 -n 11008 --gpu-tflops 989.5
# Output: gemm.csv → append to bench_data/h100/gemm/data.csv
```

### Config File Format
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

## H100 Data Collection Requirements

| Category | Configs | Benchmarks/Config | Total |
|----------|---------|-------------------|-------|
| MHA Decode | 8 head configs | 35 (7 batch × 5 kv_len) | 280 |
| MHA Prefill | 8 head configs | 5 seq_lens | 40 |
| GEMM | 1 file | 403 (M×K×N) | 403 |
| **Total** | - | - | **~720** |

### Head Configs (nh-nkv-dh)
32-8-128, 32-4-128, 28-4-128, 64-8-128, 16-2-128, 32-32-128, 128-8-128, 256-8-128

### Sweep Parameters
- **MHA Decode**: batch=[1,16,32,64,128,256,512], kv_len=[1k,4k,8k,16k,32k]
- **MHA Prefill**: seq_len=[1k,4k,8k,16k,32k]
- **GEMM**: M=[16,32,64,128,256,512,1k,2k,4k,8k], K=[2k,4k,8k], N=[6k,11k,14k,19k]

## Phase 1: Collect H100 MFU Data (5 days)

### Day 1: Environment Setup
**Tasks**: Clone InferSim, setup H100 access, install deps, modify scripts
```bash
# 1. Clone InferSim repo
git clone https://github.com/alibaba/InferSim.git
cd InferSim

# 2. Create environment
conda create -n infersim python=3.10 -y
conda activate infersim

# 3. Install dependencies
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation
pip install sgl-kernel  # Provides FlashInfer
pip install pandas

# 4. Modify fa3_mha_prefill.py for H100
sed -i 's/fp16_tflops = 148/fp16_tflops = 989.5/' kernel_benchmark/fa3_mha_prefill.py

# 5. Create output directory
mkdir -p bench_data/h100/{mha/{decode,prefill},gemm}

# 6. Test installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "from sgl_kernel.flash_attn import flash_attn_varlen_func"
```

### Day 2-3: Run MHA Benchmarks
**Tasks**: Benchmark 8 head configs using existing scripts
```bash
#!/bin/bash
cd InferSim

# Head configs: 32-8-128, 32-4-128, 28-4-128, 64-8-128, 16-2-128, 32-32-128, 128-8-128, 256-8-128
CONFIGS=("32 8 128" "32 4 128" "28 4 128" "64 8 128" "16 2 128" "32 32 128" "128 8 128" "256 8 128")

for config in "${CONFIGS[@]}"; do
    IFS=' ' read -r NH NKV DH <<< "$config"
    echo "Benchmarking nh=$NH, nkv=$NKV, dh=$DH"

    # Create temp config.json
    cat > /tmp/temp_config_${NH}_${NKV}_${DH}.json <<EOF
{
    "hidden_size": $((NH * DH)),
    "num_attention_heads": $NH,
    "num_key_value_heads": $NKV,
    "head_dim": $DH,
    "num_hidden_layers": 32,
    "intermediate_size": 14336
}
EOF

    # PREFILL: Uses modified script (fp16_tflops=989.5)
    python kernel_benchmark/fa3_mha_prefill.py \
        --config-path /tmp/temp_config_${NH}_${NKV}_${DH}.json
    mv attention_benchmark.csv bench_data/h100/mha/prefill/${NH}-${NKV}-${DH}.csv

    # DECODE: Pass fp16-tflops as arg
    python kernel_benchmark/flashinfer_mha_decode.py \
        --config-path /tmp/temp_config_${NH}_${NKV}_${DH}.json \
        --fp16-tflops 989.5 \
        --kv-cache-dtype bf16 \
        --tp-size 1
    mv attention_benchmark.csv bench_data/h100/mha/decode/${NH}-${NKV}-${DH}.csv
done

echo "Complete: $(ls bench_data/h100/mha/{prefill,decode}/*.csv | wc -l) files"
```
**Time**: ~45 min per config × 8 = 6 hours compute + queue = 2 days

### Day 4: Run GEMM Benchmarks
**Tasks**: Sweep M×K×N using existing deepgemm_gemm.py
```bash
#!/bin/bash
cd InferSim

# Run for common LLM dimensions
# K = hidden_dim, N = intermediate_dim
for K in 2048 4096 8192; do
    for N in 6144 11008 14336 18944; do
        echo "Benchmarking K=$K, N=$N"
        python kernel_benchmark/deepgemm_gemm.py \
            -k $K \
            -n $N \
            --gpu-tflops 989.5  # Use fp16 TFLOPS

        # Append to master file
        if [ ! -f bench_data/h100/gemm/data.csv ]; then
            cp gemm.csv bench_data/h100/gemm/data.csv
        else
            tail -n +2 gemm.csv >> bench_data/h100/gemm/data.csv
        fi
    done
done

echo "Total GEMM benchmarks: $(tail -n +2 bench_data/h100/gemm/data.csv | wc -l)"
```
**Time**: ~120 benchmarks × 5s = 10 min per (K,N) × 12 = 2 hours + queue = 1 day

### Day 5: Validate & Integrate
**Tasks**: Validate data, add H100 profile, test MFU lookup
```bash
# 1. Validate data quality
python3 << 'EOF'
import pandas as pd
import os

# Check MHA data
for stage in ['decode', 'prefill']:
    path = f'bench_data/h100/mha/{stage}'
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    print(f"\n{stage.upper()}: {len(files)} configs")
    for f in files:
        df = pd.read_csv(f'{path}/{f}')
        print(f"  {f}: {len(df)} benchmarks, MFU range: [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")

# Check GEMM data
df = pd.read_csv('bench_data/h100/gemm/data.csv')
print(f"\nGEMM: {len(df)} benchmarks")
print(f"  MFU range: [{df['mfu'].min():.3f}, {df['mfu'].max():.3f}]")
EOF

# 2. Add H100 to hardware/gpu.py
cat >> hardware/gpu.py << 'EOF'

h100 = GPU(
    fp16_tflops=989.5,
    fp8_tflops=1979,
    mfu=0.35,
    mem=80,
    mem_bw=3350 * 0.8,
    nvlink_bw=900 * 0.8 / 2,
    rdma_bw=50 * 0.8,
)
EOF

# Update gpu_map
sed -i 's/gpu_map = {/gpu_map = {"H100": h100, /' hardware/gpu.py

# 3. Test MFU lookup
python3 << 'EOF'
import sys
sys.path.append('.')
from config.model_config import ModelConfig
from mfu.mfu import get_attn_decode_mfu, get_gemm_mfu

# Create test config (Qwen2.5-7B: 28-4-128)
class TestConfig:
    attn_type = "MHA/GQA"
    num_attention_heads = 28
    num_key_value_heads = 4
    head_dim = 128

config = TestConfig()
mfu = get_attn_decode_mfu(config, 64, 4096, "H100", False)
print(f"✅ MFU lookup test: decode MFU={mfu} (expected ~0.15-0.25)")

mfu = get_gemm_mfu("H100", 1024, 4096, 11008)
print(f"✅ GEMM lookup test: MFU={mfu} (expected ~0.50-0.80)")
EOF
```

**Deliverables**:
- `bench_data/h100/mha/{prefill,decode}/` - 16 CSV files
- `bench_data/h100/gemm/data.csv` - 403+ benchmarks
- Updated `hardware/gpu.py` with H100 profile
- Validated MFU lookup working

## Phase 2: Port Lookup to Go (4 days)

**What we're porting**: CSV loading + lookup logic (NOT benchmarking!)

### Day 6: CSV Loader (`sim/mfu_database.go`)
**Task**: Load pre-computed CSV files into Go memory structures

**Input**: Static CSV files from Phase 1
- `bench_data/h100/mha/decode/*.csv` - attention decode MFU
- `bench_data/h100/mha/prefill/*.csv` - attention prefill MFU
- `bench_data/h100/gemm/data.csv` - GEMM MFU

**Output**: In-memory lookup tables
```go
type MFUDatabase struct {
    AttentionDecode  map[string][]AttentionMFUEntry  // "28-4-128" -> [{bs, kv_len, mfu}]
    AttentionPrefill map[string][]AttentionMFUEntry  // "28-4-128" -> [{seq_len, mfu}]
    Gemm             []GemmMFUEntry                   // [{m, k, n, mfu}]
}
```

**No benchmarking** - just parsing CSV files!

### Day 7: Lookup Functions (`sim/mfu_lookup.go`)
**Task**: Implement nearest-neighbor lookup (mimic Python `mfu/mfu.py`)

**Functions**:
- `GetAttentionDecodeMFU(nh, nkv, dh, bs, kv_len)` - find closest (bs, kv_len)
- `GetAttentionPrefillMFU(nh, nkv, dh, seq_len)` - find closest seq_len
- `GetGemmMFU(m, k, n)` - find closest (k, n), then m
- **Fallback to defaults** if no data: decode=0.15, prefill=0.60, gemm=0.50

**Logic**: Same as Python InferSim (lines 7-199 of `mfu/mfu.py`)

### Day 8-9: Roofline Model (`sim/roofline_infersim.go`)
**Task**: Use MFU lookups in roofline calculation

**Algorithm**: For each inference step:
```
1. Look up MFU from database (no benchmarking!)
2. Compute time = max(flops/(peak_tflops*mfu), memory_bytes/bandwidth)
3. Sum across layers + add overhead
```

**Key point**: All MFU values come from CSV lookups, not runtime benchmarking!

## Phase 3: Validation (2 days)

### Day 10: Cross-Validation (Go vs Python InferSim)
**Test cases**: Qwen2.5-7B, Llama-3.1-8B for prefill/decode
**Target**: < 5% error between Go and Python implementations

### Day 11: vLLM Validation
**Test real H100 runs**: Compare simulator predictions vs actual vLLM timings
**Target**: < 15% error vs real measurements

## Success Criteria

| Metric | Target |
|--------|--------|
| Go vs Python InferSim | < 5% error |
| Simulator vs vLLM | < 15% error |
| Coverage | Qwen2.5-7B, Llama-3.1-8B, CodeLlama-34B |
| TP scaling | Test TP=1,2,4 |

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| H100 access delays | Start Phase 1 immediately, use spot instances |
| Missing dependencies | Test setup Day 1, document versions |
| MFU data quality | Validate Day 5, compare to H800 patterns |
| Go port complexity | Incremental port, keep Python reference |

## Timeline

| Phase | Days | Dependencies |
|-------|------|-------------|
| Phase 1: Collect Data | 5 | H100 access |
| Phase 2: Port to Go | 4 | Phase 1 complete |
| Phase 3: Validation | 2 | Phase 2 complete |
| **Total** | **11 days** | - |

## Deliverables

**Phase 1**: bench_data_h100/ with 16 MHA CSVs + 1 GEMM CSV
**Phase 2**: Go implementation (mfu_database.go, mfu_lookup.go, roofline_infersim.go)
**Phase 3**: Validation report with error analysis

## Automated Benchmark Script

**Save as `scripts/collect_h100_data.sh`**:
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

## Next Steps

1. **Secure H100 cluster access** (highest priority)
2. **Run automated collection script** on H100
3. **Validate data** using Python validation script (Day 5)
4. **Add H100 profile** to `hardware/gpu.py`
5. **Start Phase 2**: Port to Go
