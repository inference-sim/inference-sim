# Iterative Roofline Improvement with Claude

- We first gave Claude the `inference-sim` repo to understand and document in `blis.md`. Similarly it captured its understanding of vLLM GPU kernel runtimes in `vllm.md` and existing research paper techniques in `research.md`. 
- We also separately designed an evaluation script (also using Claude) `python_scripts/blis_evaluator.py` which:
    - Runs BLIS for a variety of models and workloads
    - Compares BLIS metrics (mean e2e etc.) against ground truth and returns MAPE.
- We also collected ground truth vLLM data for `Llama2-7b` for two workload profiles - `chat` and `code` with `TP=1`.
- Now we give Claude the following prompt:
```
I want you to modify ONLY @sim/roofline_step.go. Refer to @blis.md to understand what it does. Your goal is to minimize the e2e mean error to below 30% that print out when you do the following: go build -o simulation_worker main.go && python3 python_scripts/blis_evaluator.py --ground-truth eval/combined_ground_truth.json. Use @vllm.md as reference on how to mimic vllm’s roofline calculations, and @research.md for reference on similar papers. You can also look at the model configs under @model_configs for reference and @hardware_config.go for hardware parameters.
```
- Claude iteratively edits the roofline script, builds the Go sim and runs the evaluation script to see how the MAPE changes. If a code change negatively affects MAPE, Claude reverts changes. It is able to tune hyperparameters such as overheads, effective BW/FLOPs to minimize the MAPE.
- Then we collected data for higher TPs (TP=2 and TP=4) for only the `code` workload. We now asked Claude to generalize the roofline code across TPs.
```
Modify your code to generalize to higher TP values.
```
- **Testing**: For `llama2-7b`, we collected test data for `TP=2 and TP=4` for the `chat` workload. We also collected `TP=1` ground truth for `qwen2.5-7b-instruct` for a different workload (`summarization`.)
- We now evaluated the Claude-generated roofline BLIS on all the workloads.

```
RESULTS - Mean % Error across QPS points
================================================================================

dec17-testsumm-qwen7-1:
  ttft_mean_ms        :    42.53%
  ttft_p90_ms         :    38.21%
  itl_mean_ms         :    19.70%
  e2e_mean_ms         :    14.26%
  e2e_p90_ms          :    18.30%

jan30-llama2-7b-chatsweep-1:
  ttft_mean_ms        :     5.45%
  ttft_p90_ms         :     3.67%
  itl_mean_ms         :     3.25%
  e2e_mean_ms         :     4.77%
  e2e_p90_ms          :     5.49%

jan30-llama2-7b-chatsweep-2:
  ttft_mean_ms        :     7.02%
  ttft_p90_ms         :     6.97%
  itl_mean_ms         :     1.29%
  e2e_mean_ms         :     4.05%
  e2e_p90_ms          :     4.76%

jan30-llama2-7b-chatsweep-4:
  ttft_mean_ms        :    39.83%
  ttft_p90_ms         :    39.32%
  itl_mean_ms         :    13.33%
  e2e_mean_ms         :    13.40%
  e2e_p90_ms          :    13.48%

jan30-llama2-7b-codesweep-1:
  ttft_mean_ms        :    45.12%
  ttft_p90_ms         :    37.28%
  itl_mean_ms         :    24.01%
  e2e_mean_ms         :    11.98%
  e2e_p90_ms          :    14.47%

jan30-llama2-7b-codesweep-2:
  ttft_mean_ms        :    36.21%
  ttft_p90_ms         :    25.38%
  itl_mean_ms         :    21.01%
  e2e_mean_ms         :    11.66%
  e2e_p90_ms          :    15.39%

jan30-llama2-7b-codesweep-4:
  ttft_mean_ms        :    41.51%
  ttft_p90_ms         :    28.14%
  itl_mean_ms         :    33.60%
  e2e_mean_ms         :    18.50%
  e2e_p90_ms          :    20.98%
```


# Current Roofline Model Improvements

## Overview

Improved the BLIS roofline simulator to better match vLLM ground truth measurements, reducing average e2e error from **15.3% → 11.2%**.

## Key Changes

**TP Scaling:** Changed from linear (`/tp`) to sublinear (`tp^exponent`) to model NVLink communication overhead.

**Memory Access:** Differentiated KV cache access patterns—decode uses scattered access (0.80x), prefill uses sequential (0.92x).

**Mixed Batches:** Adaptive weighting based on token distribution instead of simply adding prefill + decode times.

**Overhead Model:** Batch-size-aware scheduling overhead that scales with decode ratio.

## New Calibration Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `tpScalingExponent` | 0.30 | Sublinear TP scaling for prefill |
| `decodeTpScalingExponent` | 1.15 | TP scaling for decode |
| `mfuPrefillMultiplier` | 2.32 | Prefill MFU adjustment |
| `mfuDecodeMultiplier` | 2.00 | Decode MFU adjustment |
| `prefillBwFactor` | 0.75 | Prefill bandwidth efficiency |
| `decodeBwFactor` | 0.96 | Decode bandwidth efficiency |
| `vectorPeakFraction` | 0.10 | Non-tensor core efficiency |
| `prefillOverheadMicros` | 500.0 | Pure prefill overhead |
| `mixedPrefillOverheadMicros` | 150.0 | Mixed batch overhead |


## Files Modified

- `hardware_config.json` - Added calibration parameters
- `sim/model_hardware_config.go` - Added struct fields
- `sim/roofline_step.go` - Improved roofline calculations
