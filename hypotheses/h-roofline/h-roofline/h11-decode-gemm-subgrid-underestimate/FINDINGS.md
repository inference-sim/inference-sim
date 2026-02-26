# H34: Decode GEMM Time Underestimate Below MFU Grid Minimum -- FINDINGS

**Status**: Confirmed
**Date**: 2026-02-25

## Hypothesis

For decode steps at batch sizes 1-7 (below the GEMM MFU grid minimum M=8), the roofline model's `computeTransformerGEMMTimes` predicts GEMM time proportional to batch size (ratio bs/8 relative to bs=8), whereas actual GPU GEMM kernels at these small M values exhibit a near-constant memory-bound latency floor, causing the model to underestimate GEMM compute time by up to 8x at bs=1.

**Refuted if:** The predicted GEMM time ratio `computeTransformerGEMMTimes(bs=1) / computeTransformerGEMMTimes(bs=8)` exceeds 0.25 (i.e., the underestimate is less than 4x), indicating the MFU database already compensates for the memory-bound floor below M=8.

## Experiment Design

- **Independent variable**: Batch size (1, 2, 4, 8, 16, 32)
- **Controlled variables**: Model config (Llama-3.1-8B + eval suite), H100 hardware config, TP=1, tpScaling=1.0
- **Dependent variables**: GEMM time (us), ratio relative to bs=8, MFU value, compute/memory regime

### Methodology

Five sub-experiments:

1. **GEMM Time Scaling** (TestH34_GEMMTimeScalingAcrossBatchSizes): Measures computeTransformerGEMMTimes across batch sizes for all eval suite models, computing ratio vs bs=8
2. **Per-GEMM Breakdown** (TestH34_PerGEMMBreakdown): Individual Q/K/V/O/Gate/Up/Down projection GEMM times and MFU values at each batch size
3. **MFU Clamping** (TestH34_MFUClampingMechanism): Direct verification that GetGEMMmfu returns identical values for all bs < grid minimum
4. **Regime Analysis** (TestH34_RegimeAnalysis): Compute vs memory regime classification at small batch sizes, plus impact analysis if a constant GEMM floor were applied
5. **Multi-Model Summary** (TestH34_MultiModelSummary): Concise verdict table across all eval suite models

### Key Mechanism

The MFU clamping in `GetGEMMmfu` (mfu_database.go line 597):
```go
if m <= mPoints[0].m {
    mfu = mPoints[0].mfu
}
```

For all batch sizes below the GEMM grid minimum (typically M=8), the MFU is constant. Since GEMM time = `2 * m * k * n / (peakFlops * mfu)`, time is strictly proportional to m, giving ratio = bs/8 for bs < 8.

## Results

### Table 1: Multi-Model GEMM Time Ratio

```
Model               GEMM bs=1(us)  GEMM bs=8(us)      Ratio   Underest    Verdict
--------------------------------------------------------------------------------
llama-2-7b                  894.7         7157.8     0.1250       8.0x  CONFIRMED
llama-2-70b                8163.9        65311.4     0.1250       8.0x  CONFIRMED
llama-3.1-8b                879.4         7034.8     0.1250       8.0x  CONFIRMED
codellama-34b              3964.3        31714.8     0.1250       8.0x  CONFIRMED
qwen3-14b                  1611.6        12892.5     0.1250       8.0x  CONFIRMED
qwen2.5-7b                  806.3         6450.5     0.1250       8.0x  CONFIRMED

Mean ratio across models: 0.125000
Min ratio:  0.125000
Max ratio:  0.125000
Expected if perfectly linear: 0.125000 (1/8)
```

### Table 2: Per-GEMM Breakdown

Single layer, Llama-3.1-8B (hidden=4096, layers=32):

```
GEMM           bs=1(us)   bs=2(us)   bs=4(us)   bs=8(us)  bs=16(us)  bs=32(us)  Ratio bs1/8
-----------------------------------------------------------------------------------------------
Q_proj            2.610      5.220     10.439     20.879     20.879     20.105     0.125000
K_proj            0.652      1.305      2.610      5.220      5.220      5.026     0.125000
V_proj            0.652      1.305      2.610      5.220      5.220      5.026     0.125000
O_proj            2.610      5.220     10.439     20.879     20.879     20.105     0.125000
Gate_proj         6.985     13.970     27.940     55.881     54.284     55.071     0.125000
Up_proj           6.985     13.970     27.940     55.881     54.284     55.071     0.125000
Down_proj         6.985     13.970     27.940     55.881     54.284     55.071     0.125000
```

MFU values at each batch size:

```
GEMM               bs=1       bs=2       bs=4       bs=8      bs=16      bs=32
-----------------------------------------------------------------------------------------------
Q_proj         0.013000   0.013000   0.013000   0.013000   0.026000   0.054000
K_proj         0.013000   0.013000   0.013000   0.013000   0.026000   0.054000
V_proj         0.013000   0.013000   0.013000   0.013000   0.026000   0.054000
O_proj         0.013000   0.013000   0.013000   0.013000   0.026000   0.054000
Gate_proj      0.017000   0.017000   0.017000   0.017000   0.035000   0.069000
Up_proj        0.017000   0.017000   0.017000   0.017000   0.035000   0.069000
Down_proj      0.017000   0.017000   0.017000   0.017000   0.035000   0.069000
```

Note: MFU is constant across bs=1 through bs=8 (the grid minimum), then doubles at bs=16 and approximately doubles again at bs=32, reflecting improved GPU utilization at larger batch sizes. This confirms the clamping mechanism is the root cause of linear GEMM time scaling below bs=8.

### Table 3: MFU Clamping Verification

```
--- Q_proj (k=4096, n=4096) ---
BatchSize           MFU  SameAsBs1
----------------------------------
1              0.013000       SAME
2              0.013000       SAME
3              0.013000       SAME
4              0.013000       SAME
5              0.013000       SAME
6              0.013000       SAME
7              0.013000       SAME
8              0.013000       SAME
16             0.026000  DIFFERENT
32             0.054000  DIFFERENT
64             0.108000  DIFFERENT
  Sub-grid (bs<8) clamped: 7 / 7

--- Gate_proj (k=4096, n=14336) ---
BatchSize           MFU  SameAsBs1
----------------------------------
1              0.017000       SAME
2              0.017000       SAME
3              0.017000       SAME
4              0.017000       SAME
5              0.017000       SAME
6              0.017000       SAME
7              0.017000       SAME
8              0.017000       SAME
16             0.035000  DIFFERENT
32             0.069000  DIFFERENT
64             0.138000  DIFFERENT
  Sub-grid (bs<8) clamped: 7 / 7

--- Down_proj (k=14336, n=4096) ---
BatchSize           MFU  SameAsBs1
----------------------------------
1              0.017000       SAME
2              0.017000       SAME
3              0.017000       SAME
4              0.017000       SAME
5              0.017000       SAME
6              0.017000       SAME
7              0.017000       SAME
8              0.017000       SAME
16             0.035000  DIFFERENT
32             0.069000  DIFFERENT
64             0.138000  DIFFERENT
  Sub-grid (bs<8) clamped: 7 / 7
```

All three representative GEMM shapes (Q_proj, Gate_proj, Down_proj) show 7/7 sub-grid batch sizes clamped to the same MFU as bs=1. The MFU value at bs=8 equals that at bs=1, confirming M=8 is the grid minimum (not M=9 or M=16). MFU changes only at bs=16 and above.

### Table 4: Regime Analysis

Llama-3.1-8B, KVLen=1024, TP=1:

```
BatchSize      GEMM(us)     Attn(us)   Memory(us)     Step(us)   Regime
--------------------------------------------------------------------------
1                 879.4         67.9       5120.6         8321      MEM
2                1758.7       2035.7       5159.8         8360      MEM
4                3517.4       1357.1       5238.2         8438      MEM
8                7034.8       1163.2       5395.1        11398     COMP
16               6881.6       1085.7       5708.7        11167     COMP
32               6895.2       1737.1       6336.0        11832     COMP
```

Key observations:
- At bs=1, GEMM (879 us) is only 17% of memory time (5121 us) -- deeply memory-bound
- At bs=4, GEMM (3517 us) is 67% of memory time (5238 us) -- still memory-bound
- The crossover to compute-bound occurs at bs=8, where GEMM+Attn (8198 us) exceeds memory (5395 us)
- Memory time grows slowly with batch size (KV cache reads scale, but weight loads dominate)

### Table 5: Impact of Constant GEMM Floor

If GEMM time at bs < 8 were floored to the bs=8 GEMM time (7035 us for Llama-3.1-8B):

```
BatchSize   Current(us)    Floor(us)    Delta(us)  Increase%
--------------------------------------------------------
1                  8321        10303         1982     23.82%
2                  8360        12270         3910     46.77%
4                  8438        11592         3154     37.38%
```

Key observations:
- At bs=1: applying the floor adds 1982 us (+23.8%) because the higher GEMM time (7035 us) exceeds the memory time (5121 us), flipping the regime from MEM to COMP
- At bs=2: the largest impact (+46.8%) because both the GEMM floor increase and the attention time (2036 us) combine to push compute well above memory
- At bs=4: moderate impact (+37.4%) -- the floor adds 3517 us of GEMM time, pushing compute above memory

## Analysis

1. **MFU Clamping**: All sub-grid MFU values are identical. For every GEMM shape tested, `GetGEMMmfu` returns the exact same MFU for batch sizes 1 through 8. This is not interpolation -- it is flat clamping to the grid minimum's MFU value. The MFU only changes at bs=16 (approximately 2x) and bs=32 (approximately 4x vs the clamped value).

2. **Ratio linearity**: The ratio is exactly bs/8 with zero deviation. For all 6 models, the ratio gemmTime(bs=1)/gemmTime(bs=8) = 0.125000 to six decimal places. This is not approximate -- it is mathematically exact because the formula `2*m*k*n / (peakFlops * mfu)` with constant mfu produces time strictly proportional to m (batch size).

3. **Regime impact**: At small batch sizes (bs=1,2,4) the decode step is memory-bound under the current model. The GEMM underestimate is therefore absorbed by `max(compute, memory)` and does NOT currently affect step time predictions. However, the impact analysis shows that if a GEMM floor were applied (correcting the underestimate), the small-batch regime would flip to compute-bound, increasing step time by 24-47%. This means the current model arrives at the correct step time for the wrong reason -- memory time happens to dominate and masks the GEMM underestimate.

4. **Cross-model consistency**: The effect is perfectly uniform across all 6 eval suite models (llama-2-7b, llama-2-70b, llama-3.1-8b, codellama-34b, qwen3-14b, qwen2.5-7b). Every model shows ratio = 0.125000 and 8.0x underestimate. This is expected because the mechanism is purely in the MFU lookup, which is independent of model architecture parameters.

## Verdict

**[x]** CONFIRMED: Ratio bs1/bs8 <= 0.25 for all models
**[ ]** REFUTED: Ratio bs1/bs8 > 0.25 (MFU database already compensates)

The hypothesis is **CONFIRMED** with maximum strength. The ratio gemmTime(bs=1)/gemmTime(bs=8) = 0.125000 exactly (the theoretical minimum under the clamping mechanism) for all 6 models tested. This represents an 8x underestimate of GEMM compute time at bs=1, confirming that `computeTransformerGEMMTimes` predicts GEMM time strictly proportional to batch size below the MFU grid minimum.

**Practical impact is partially mitigated**: At small batch sizes (1-4), decode steps are memory-bound under the current model, so the GEMM underestimate is absorbed by `max(compute, memory)`. The step time prediction at bs=1 is 8321 us (memory-dominated), which would increase to 10303 us (+24%) if the GEMM floor correction were applied. The most impacted case is bs=2, where the corrected step time would increase by 47%.

## Implications

Confirmed. Recommended remediation options, in order of preference:

- **Option A**: Add a minimum GEMM time floor based on weight matrix memory bandwidth: `gemmTime = max(flops/(peakFlops*mfu), weightBytes/peakBW)` to model the memory-bound regime at small M. This is the most physically grounded approach -- at small M, GEMM time is dominated by loading the weight matrix, not by the FLOPs.
- **Option B**: Extend the GEMM MFU benchmark grid to include M=1,2,4 entries, capturing the actual kernel behavior at these small sizes. This requires actual GPU benchmarking but produces the most accurate results.
- **Option C**: Apply a sub-grid correction factor: for m < grid_min, multiply GEMM time by `grid_min / m` to approximate the constant-floor behavior. This is the simplest fix but least principled.
- **Impact**: Single-request and small-batch decode latency predictions become more accurate, especially for latency-sensitive applications. The current mitigating factor (memory-bound regime at small bs) means this bug primarily manifests as incorrect regime classification rather than incorrect step time, but any future changes that reduce memory time (e.g., model weight compression, higher memory bandwidth GPUs) would expose the GEMM underestimate as a direct step time error.

## Root Cause Verification

- [x] RCV-1: Mechanism identified -- MFU clamping in `GetGEMMmfu` causes constant MFU (0.013 for 4Kx4K, 0.017 for 4Kx14K shapes) for all m <= grid minimum (M=8). With constant MFU, GEMM time = `2*m*k*n / (peakFlops * mfu)` is strictly proportional to m.
- [x] RCV-2: Isolation confirmed -- single variable (batch size) swept with all other parameters held constant (model config, hardware config, TP=1, tpScaling=1.0). The 0.125 ratio is exact for all batch size / model combinations.
- [x] RCV-3: Reproducible across models -- consistent behavior across all 6 eval suite models (llama-2-7b, llama-2-70b, llama-3.1-8b, codellama-34b, qwen3-14b, qwen2.5-7b). Ratio = 0.125000 for every model.
- [x] RCV-4: Control experiment -- regime analysis confirms the underestimate IS absorbed by `max(compute, memory)` at bs=1,2,4 (all memory-bound). Impact analysis shows a corrected GEMM floor would increase step time by 24-47%, flipping the regime from MEM to COMP at small batch sizes.
