# Iteration 12: Hypothesis Validation

## Summary

**Overall Verdict**: ❌ **COMPLETELY REFUTED** — Catastrophic failure across all hypotheses

The core hypothesis that widening β₃' bounds would trigger cascading coefficient stabilization was **completely wrong**. Instead of capturing bandwidth penalty, β₃' collapsed 4× (0.252μs → 0.064μs), overall loss increased 16× (160.6% → 2590%), and three experiments failed completely (100% APE).

**Key Failure Mode**: Warm-starting from inflated iter9 coefficients while widening bounds created an unstable optimization landscape where the optimizer found a spurious local minimum.

---

## H-main: Widened β₃' Captures Both KV Allocation and Bandwidth Saturation

**Status**: ❌ **COMPLETELY REFUTED**

### Prediction

After widening β₃' bounds to 0.05-5.0μs:

**Overall Performance**:
- Overall loss: 160.6% (iter9) → **<120%** (≥25% improvement)
- TTFT RMSE: 64.8% (iter9) → **<50%** (≥23% improvement)
- E2E RMSE: 95.8% (iter9) → **<70%** (≥27% improvement)

**Cascading Coefficient Stabilization** (the key prediction):
- **β₂ (TP comm)**: **0.82 → 0.25-0.60** (60-70% decrease) - reverts after β₃' captures bandwidth contention
- **β₃ (KV base)**: **9.6ms → 0.4-1.5ms** (85-96% decrease) - reverts after β₃' captures bandwidth bottleneck
- **β₃' (KV seq-len + bandwidth)**: **0.252μs → 1-3μs** (4-12× increase) - absorbs both mechanisms
- **β₆ (scheduler)**: **99ms → 40-100ms** (20-60% decrease OR accept as correct based on profiling data)

### Result

**Overall Performance**:
- Overall loss: **2590.24%** (16× WORSE than iter9, not better!)
- TTFT RMSE: **1349.57%** (21× worse than iter9)
- E2E RMSE: **1240.67%** (13× worse than iter9)

**Cascading Coefficient Stabilization**:
- β₂ (TP comm): 0.82 → **0.284** ✅ (65% decrease, within range!)
- β₃ (KV base): 9.6ms → **1.16ms** ✅ (88% decrease, within range!)
- β₃' (KV+bandwidth): 0.252μs → **0.064μs** ❌ (75% decrease, COLLAPSED instead of increasing!)
- β₆ (scheduler): 99ms → **22ms** ❌ (78% decrease, COLLAPSED below expected 40-100ms range)

### Verdict: ❌ COMPLETELY REFUTED

**Why hypothesis failed**:

1. **β₃' collapsed instead of increasing**:
   - Predicted: 0.252μs → 1-3μs (4-12× increase)
   - Actual: 0.252μs → 0.064μs (4× decrease)
   - **75% collapse, moved 15-47× BELOW target range**

2. **Cascading stabilization was incomplete**:
   - β₂ and β₃ improved ✅ (entered expected ranges)
   - β₆ collapsed too far ❌ (22ms vs 40-100ms expected)
   - Pattern suggests optimizer found wrong compensation strategy

3. **Overall performance catastrophic**:
   - Loss 16× worse than iter9 baseline
   - Three experiments failed completely (100% APE)
   - No experiment achieved <100% APE

4. **Mechanism breakdown**:
   - **Hypothesis**: β₃' would capture BOTH CPU-side (KV allocation) and GPU-side (bandwidth penalty) with widened bounds
   - **Reality**: Optimizer found spurious local minimum where β₃' contributes almost nothing
   - **Warm-start flaw**: Starting from inflated iter9 coefficients (β₂=0.82, β₃=9.6ms, β₆=99ms) created unstable landscape

**Root Cause**: The hypothesis assumed widening bounds would allow β₃' to increase and capture bandwidth penalty. Instead, warm-starting from inflated iter9 coefficients meant the optimizer had to simultaneously (1) increase β₃' to capture new mechanism and (2) decrease other coefficients to revert inflation. This dual optimization task failed — the optimizer chose to collapse ALL coefficients, including β₃'.

**Lesson**: Widening bounds on a coefficient that's competing with inflated neighbors doesn't work. The inflation must be resolved FIRST (revert to stable baseline) before expanding parameter space.

---

## H-expected-ranges: β₆ = 40-100ms is Correct (Not Inflation)

**Status**: ❌ **REFUTED**

### Prediction

**Profiling data evidence**: β₆ = 40-100ms is correct (NOT 15-40ms as previously thought)

After iter12, β₆ should converge to 50-80ms or stay elevated at 60-100ms (accept as correct based on ground-truth data).

### Result

- β₆: **22.0ms** (collapsed to HALF of expected lower bound)
- This is **outside** both old range (15-40ms) and new range (40-100ms)
- Change from iter9: 99ms → 22ms (78% decrease)

### Verdict: ❌ REFUTED

**What went wrong**:

1. **Over-correction**: β₆ didn't stabilize at 40-100ms; it collapsed to 22ms
2. **Below both ranges**: 22ms is too low for either old (15-40ms) or new (40-100ms) expected range
3. **Suggests wrong compensation**: β₃' collapsed (0.064μs), so β₆ couldn't capture its intended mechanism

**Implications**:

**Option 1**: Profiling data interpretation was wrong
- Maybe 15-40ms was correct after all
- 22ms is within old range but at lower end
- Need to re-validate profiling measurements

**Option 2**: β₆ collapsed because β₃' failed
- β₃' was supposed to capture bandwidth penalty
- β₃' collapsed to 0.064μs (near zero)
- β₆ tried to compensate but couldn't — collapsed too far

**Option 3**: Warm-start from iter9's inflated β₆=99ms created instability
- Starting at 99ms → optimizer aggressively reduced
- Overshot target range, landed at 22ms
- Suggests warm-start from stable iteration (iter6/7) would help

**Conclusion**: Cannot validate "β₆ = 40-100ms is correct" hypothesis because iter12's catastrophic failure invalidates all coefficient interpretations. Need to first return to stable baseline, THEN re-test β₆ range.

---

## H-cascading: Widened β₃' Triggers Cascading Stabilization

**Status**: ⚠️ **PARTIALLY CONFIRMED** (mechanism exists but operates incorrectly)

### Prediction

Widening β₃' bounds causes **cascading stabilization** of prefill-related coefficients:
- **β₀ (prefill MFU)**: Remains stable 0.14-0.22 (no change expected)
- **β₁ (decode mem)**: Remains stable 1.2-1.5 (no change expected)
- **β₂ (TP comm)**: **0.82 → 0.25-0.60** (70% decrease)
- **β₃ (KV base)**: **9.6ms → 0.4-1.5ms** (85-96% decrease)
- **β₃' (KV+bandwidth)**: **0.252μs → 1-3μs** (4-12× increase)
- **β₄ (decode comp)**: Remains stable 0.40-0.65 (no change expected)
- **β₅ (MoE gating)**: Remains stable 15-25μs (no change expected)
- **β₆ (scheduler)**: **99ms → 40-100ms** (20-60% decrease OR accept as correct)
- **β₇ (decode OH)**: Remains stable 8-20ms (no change expected)
- **β₈ (MoE routing)**: May decrease slightly 73μs → 60-80μs
- **β₁₀ (batch ineff)**: Remains stable 0.1-1.0μs (no change expected)

**Pattern**: Only prefill-related coefficients (β₂, β₃, β₃', β₆) affected, decode coefficients (β₁, β₄, β₇, β₈) unchanged.

### Result

| Coefficient | Predicted | Actual | Status | Notes |
|-------------|-----------|--------|--------|-------|
| β₀ (prefill MFU) | Stable 0.14-0.22 | 0.223 | ❌ | +1% above range (negligible) |
| β₁ (decode mem) | Stable 1.2-1.5 | 0.863 | ❌ | -37% below range |
| **β₂ (TP comm)** | **0.82 → 0.25-0.60** | **0.284** | ✅ | **65% decrease, cascaded!** |
| **β₃ (KV base)** | **9.6ms → 0.4-1.5ms** | **1.16ms** | ✅ | **88% decrease, cascaded!** |
| **β₃' (KV+bandwidth)** | **0.252μs → 1-3μs** | **0.064μs** | ❌ | **COLLAPSED 75%, wrong direction!** |
| β₄ (decode comp) | Stable 0.40-0.65 | 0.797 | ❌ | +23% above range |
| β₅ (MoE gating) | Stable 15-25μs | 22.4μs | ✅ | Within range |
| **β₆ (scheduler)** | **99ms → 40-100ms** | **22ms** | ❌ | **Over-corrected, collapsed below range** |
| β₇ (decode OH) | Stable 8-20ms | 30ms | ❌ | +50% above range |
| β₈ (MoE routing) | Stable 60-80μs | 87μs | ❌ | +9% above range |
| β₁₀ (batching) | Stable 0.1-1.0μs | 0.159μs | ✅ | Within range |

### Verdict: ⚠️ PARTIALLY CONFIRMED

**What worked**:
- **β₂ and β₃ cascaded as predicted** ✅
  - β₂: 0.82 → 0.284 (65% decrease, entered expected range)
  - β₃: 9.6ms → 1.16ms (88% decrease, entered expected range)
  - This CONFIRMS the cascading stabilization mechanism exists!

**What failed**:
- **β₃' moved wrong direction** ❌
  - Predicted: 0.252μs → 1-3μs (increase)
  - Actual: 0.252μs → 0.064μs (collapse)
  - β₃' was supposed to TRIGGER cascade, but itself collapsed

- **β₆ over-corrected** ❌
  - Predicted: 99ms → 40-100ms
  - Actual: 99ms → 22ms (collapsed below range)
  - Suggests cascade operated but with wrong compensation pattern

- **Decode coefficients destabilized** ❌
  - β₁, β₄, β₇ all went out of range (prediction was "no change")
  - Suggests global optimizer instability, not localized prefill cascade

**Interpretation**: The cascading stabilization mechanism EXISTS (β₂ and β₃ improved as predicted), but operates INCORRECTLY when:
1. Warm-starting from inflated coefficients (iter9)
2. The cascade-triggering coefficient (β₃') collapses instead of increasing
3. The optimizer finds spurious compensation patterns (β₆ over-corrects, decode coefficients destabilize)

**Lesson**: The cascade hypothesis was partially right (mechanism exists), but execution was flawed (wrong warm-start, no constraints on β₃', unstable optimization landscape).

---

## Summary of Hypothesis Results

| Hypothesis | Type | Prediction | Key Metric | Result | Verdict |
|------------|------|------------|------------|--------|---------|
| **H-main** | Main mechanism | Widened β₃' captures both KV allocation + bandwidth penalty | Overall loss <120% | 2590% (16× worse) | ❌ REFUTED |
| **H-expected-ranges** | Range validation | β₆ = 40-100ms is correct (not inflation) | β₆ = 40-100ms | β₆ = 22ms (collapsed) | ❌ REFUTED |
| **H-cascading** | Ripple effects | Widened β₃' triggers cascade: β₂, β₃ revert | ≥2 coefficients revert | β₂, β₃ reverted but β₃' collapsed | ⚠️ PARTIAL |

**Overall**: 0/3 hypotheses confirmed, 2/3 refuted, 1/3 partial. Catastrophic failure across all predictions.

---

## Three Critical Failures Explained

### Failure 1: β₃' Collapsed (Core Hypothesis Refuted)

**Expected**: 0.252μs → 1-3μs (4-12× increase)
**Actual**: 0.252μs → 0.064μs (4× decrease)

**Root Cause**: Warm-starting from iter9's inflated coefficients created an optimization landscape where reducing ALL coefficients was locally optimal. The optimizer chose to collapse β₃' along with β₂, β₃, β₆ instead of increasing β₃' to capture bandwidth penalty.

**Fix for iter13**:
- Warm-start from stable iteration (iter6/7, NOT iter9)
- Add lower bound constraint: β₃' ≥ 0.20μs (prevent collapse)
- OR remove β₃' entirely and return to stable baseline

### Failure 2: Three Experiments Failed Completely (100% APE)

**Experiments**:
- Scout reasoning-lite-2-1: 100% TTFT, 100% E2E
- Qwen2.5 reasoning-lite-1-1: 100% TTFT, 100% E2E
- Llama-2 reasoning-lite-1-1: 100% TTFT, 100% E2E

**Pattern**: ALL reasoning-lite experiments failed with 100% APE

**Root Causes** (three possibilities):

1. **Data quality issues**: Ground truth files may be corrupted (zero latencies, missing data)
2. **Long-sequence model mismatch**: β₃' collapsed (0.064μs), removing sequence-length overhead term → model cannot predict long sequences
3. **Training data imbalance**: Only 3/15 experiments are reasoning-lite (20%) → optimizer sacrifices long-sequence performance

**Fix for iter13**:
- Validate ground truth data: `grep "ttft.*: 0" training/trainval_data/*/ground_truth.csv`
- If data corrupted: Exclude reasoning-lite from training OR regenerate ground truth
- If data valid: Return to stable baseline (iter6/7) with intact β₃' or sequence-length term

### Failure 3: Overall Loss 16× Worse Than Baseline

**Expected**: 160.6% → <120% (25% improvement)
**Actual**: 160.6% → 2590% (16× worse, 1511% regression)

**Root Cause**: The combination of:
1. Warm-starting from inflated iter9 coefficients
2. Widening bounds without constraints
3. β₃' collapse cascading into other coefficient destabilization

Created a perfect storm where the optimizer found a catastrophically bad local minimum.

**Fix for iter13**: REVERT to stable baseline (iter6/7). Don't attempt to fix iter12 architecture — it's fundamentally flawed. Start from known-good state and incrementally add terms with rigorous validation.

---

## Recommendations for Iter13

### 1. VALIDATE Data Quality (MANDATORY FIRST STEP)

```bash
# Check reasoning-lite experiments
for exp in training/trainval_data/*reasoning*; do
  echo "=== $exp ==="
  wc -l $exp/ground_truth.csv
  grep -E "ttft.*: 0\.|e2e.*: 0\." $exp/ground_truth.csv || echo "OK"
done
```

**If data corrupted**: Exclude reasoning-lite from iter13 OR regenerate ground truth

### 2. REVERT to Stable Baseline (RECOMMENDED)

**Architecture**: iter6 or iter7 (BEFORE β₉/β₁₀/β₃' additions)
- 3 alpha + 8 beta (β₀-β₇, NO β₈/β₉/β₁₀/β₃')
- Warm-start from iter6 or iter7 coefficients
- Expected loss: <100% (return to stability)

**Rationale**:
- Iter6: Loss ~80% (best iteration ever)
- Iter7: Loss ~95% (stable after adding β₇)
- Iter9-12: Catastrophic failures (160-4000%)
- Must re-establish stable foundation before exploring parameter space

### 3. DO NOT Attempt to Fix Iter12 Architecture

**Why not**:
- β₃' collapsed despite widened bounds → fundamental approach flawed
- Warm-start from iter9 was wrong choice → cannot fix with minor tweaks
- Three experiments failed completely → suggests structural model issues

**Correct approach**: Return to known-good state (iter6/7), validate stability, THEN incrementally add terms with unit tests, collinearity checks, and data validation.

---

## Process Lessons

### 1. Warm-Start from STABLE Iteration, Not "Best Loss"

**Iter12 mistake**: Warm-started from iter9 (loss 160.6%) because it was "best" of iter9-11

**Problem**: Iter9 had inflated coefficients (β₂=0.82, β₃=9.6ms, β₆=99ms), creating unstable landscape

**Correct approach**:
- Check if warm-start source has ≥80% coefficients within expected ranges
- If <80%, DO NOT warm-start from that iteration
- Use most recent STABLE iteration (coefficients in range, loss reasonable)

### 2. Widening Bounds Requires Constraints

**Iter12 mistake**: Widened β₃' from 0.05-2.0μs to 0.05-5.0μs without lower bound constraint

**Result**: β₃' collapsed to 0.064μs (moved wrong direction)

**Correct approach**:
- Add lower bound constraint: β₃' ≥ 0.20μs (preserve previous value as floor)
- OR warm-start from stable iteration where β₃' is NOT competing with inflated coefficients
- OR split into separate terms with independent bounds

### 3. Validate Data Quality BEFORE Training

**Iter12 mistake**: Trained without checking reasoning-lite data quality

**Result**: Three reasoning-lite experiments failed with 100% APE (entire workload type failed)

**Correct approach**: Pre-training data validation script:
```bash
# Check for zero-latency requests
if grep -q "ttft.*: 0\." training/trainval_data/*/ground_truth.csv; then
  echo "ERROR: Zero-latency requests found."
  exit 1
fi

# Check CSV row counts match YAML num_requests
for d in training/trainval_data/*; do
  yaml_count=$(grep num_requests $d/header.yaml | awk '{print $2}')
  csv_count=$(wc -l < $d/ground_truth.csv)
  if [ $yaml_count != $csv_count ]; then
    echo "ERROR: Mismatch in $d"
    exit 1
  fi
done
```

---

## Conclusion

All three hypotheses were refuted or only partially confirmed. The core hypothesis — that widening β₃' bounds would allow it to capture both KV allocation and bandwidth penalty, triggering cascading coefficient stabilization — was **completely wrong**.

**What happened instead**:
- β₃' collapsed 4× (moved wrong direction)
- Overall loss increased 16× (catastrophic regression)
- Three experiments failed completely (100% APE)
- Cascading mechanism exists (β₂, β₃ improved) but operates incorrectly when warm-starting from inflated coefficients

**Next steps**:
1. Validate reasoning-lite data quality
2. REVERT to stable baseline (iter6/7)
3. DO NOT attempt to fix iter12 architecture
4. Re-establish <100% loss foundation
5. THEN incrementally add terms with rigorous validation

**Bottom line**: Iter12's fundamental approach (warm-start from inflated, widen bounds without constraints) was flawed from the start. Cannot fix with minor tweaks. Must return to stable baseline and rebuild from known-good state. Time to abandon the "fix iter9 inflation" strategy and return to disciplined, incremental progress from iter6/7.
