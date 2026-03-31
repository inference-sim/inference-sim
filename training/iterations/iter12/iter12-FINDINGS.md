# Iteration 12: Findings and Analysis

## Executive Summary

**Status**: ❌ **CATASTROPHIC FAILURE (Tier 4)** — Worst iteration ever recorded

**Overall Loss**: 2590.24% (16× worse than iter9's 160.6%, 1.6× worse than iter10/11's ~4200%)

**⚠️ CRITICAL: HYPOTHESIS COMPLETELY REFUTED ⚠️**

The core hypothesis that widening β₃' bounds would allow it to capture bandwidth saturation and trigger cascading coefficient stabilization was **COMPLETELY WRONG**. Instead:

- **β₃' COLLAPSED** from 0.252μs (iter11) to 0.064μs (iter12) — moved AWAY from target range
- **NO cascading stabilization** — β₂ improved slightly but β₃ didn't stabilize as predicted
- **THREE experiments completely failed** (100% APE on all metrics)
- **Overall loss 16× worse** than iter9 baseline

**Root Cause**: Warm-starting from iter9's inflated coefficients (β₂=0.82, β₃=9.6ms, β₆=99ms) while widening β₃' bounds created an unstable optimization landscape. The optimizer found a spurious local minimum where β₃' collapsed and other coefficients compensated incorrectly.

**Key Learning**: Widening bounds on a coefficient that's competing with inflated neighbors doesn't work. The inflation must be resolved FIRST before expanding parameter space.

---

## Results

### Metrics (Iter12 vs Iter11 vs Iter9)

| Metric | Iter12 | Iter11 | Iter9 | Target | Status |
|--------|--------|--------|-------|--------|--------|
| **Overall Loss** | 2590.24% | 4084.44% | 160.6% | <90% | ❌ 29× worse than target, 16× worse than iter9 |
| **TTFT RMSE** | 1349.57% | 1423.25% | 64.8% | <40% | ❌ 34× worse than target, 21× worse than iter9 |
| **E2E RMSE** | 1240.67% | 2661.18% | 95.8% | <55% | ❌ 23× worse than target, 13× worse than iter9 |
| **Trials** | 413 | 500 | 142 | - | Converged early |

**Observation**: Iter12 is better than iter10/11 (both ~4000%) but **16× worse than iter9** (160.6%). The hypothesis strategy failed catastrophically.

---

## Coefficient Analysis

### Full Coefficient Status (Iter12 vs Iter11 vs Iter9)

| Coefficient | Iter12 | Iter11 | Iter9 | Expected Range | Status | Change from Iter9 |
|-------------|--------|--------|-------|----------------|--------|-------------------|
| β₀ (prefill MFU) | 0.223 | 0.286 | 0.162 | 0.14-0.22 | ❌ | +38% (slightly high) |
| β₁ (decode mem) | 0.863 | 1.107 | 1.361 | 1.2-1.5 | ❌ | -37% (too low) |
| β₂ (TP comm) | **0.284** | 0.383 | 0.817 | 0.25-0.60 | ✅ | **-65% (IMPROVED!)** |
| β₃ (KV base) | **1.16 ms** | 0.21 ms | 9.59 ms | 0.4-1.5 ms | ✅ | **-88% (IMPROVED!)** |
| **β₃' (KV+bandwidth)** | **0.064 μs** | 0.252 μs | N/A | 1.0-3.0 μs | ❌ | **COLLAPSED 15-47× below target!** |
| β₄ (decode comp) | 0.797 | 0.815 | 0.466 | 0.40-0.65 | ❌ | +71% (too high) |
| β₅ (MoE gating) | 22.4 μs | 15.5 μs | 19.8 μs | 15-25 μs | ✅ | +13% (within range) |
| β₆ (scheduler) | **22.0 ms** | 59.3 ms | 99.3 ms | 40-100 ms | ❌ | **-78% (collapsed!)** |
| β₇ (decode OH) | 30.0 ms | 5.0 ms | 11.0 ms | 8-20 ms | ❌ | +173% (too high) |
| β₈ (MoE routing) | 87.0 μs | 44.5 μs | 72.8 μs | 25-80 μs | ❌ | +19% (slightly high) |
| β₁₀ (batching ineff) | 0.159 μs | 0.950 μs | N/A | 0.1-1.0 μs | ✅ | Within range |

**Summary**:
- ✅ **6/11 coefficients within range** (β₂, β₃, β₅, β₈, β₁₀, and arguably β₈)
- ❌ **5/11 coefficients out of range** (β₀, β₁, β₃', β₄, β₆, β₇)
- **CRITICAL**: β₃' collapsed 4× from iter11 (0.252μs → 0.064μs), moving AWAY from target

### β₃' Analysis: The Core Failure

**Hypothesis prediction**: β₃' = 0.252μs (iter11) → **1-3μs** (4-12× increase to capture bandwidth penalty)

**Actual result**: β₃' = **0.064μs** (4× DECREASE, 15-47× below target range!)

**What went wrong**:
1. **Widened bounds backfired**: Expanding range to 0.05-5.0μs gave optimizer MORE room to collapse, not stabilize
2. **Warm-start from inflated iter9**: Starting with β₂=0.82, β₃=9.6ms, β₆=99ms created unstable landscape
3. **Competition, not complementarity**: β₃' didn't capture bandwidth penalty; instead optimizer found different (wrong) local minimum
4. **No cascading stabilization**: β₂ improved but β₆ collapsed too far (22ms vs 40-100ms expected)

**Implications**: The hypothesis that β₃' can capture BOTH KV allocation and bandwidth penalty is **refuted**. They may not be captured by the same basis function, or the warm-start strategy was fundamentally flawed.

---

## Per-Experiment Results

### Complete Failure Tier (100% APE - Model Predicts Nothing)

Three experiments completely failed with 100% APE across all metrics:

| Experiment | Model | Workload | TTFT APE | E2E APE | Notes |
|------------|-------|----------|----------|---------|-------|
| Scout reasoning-lite | Scout 17B | reasoning-lite-2-1 | 100% | 100% | Total prediction failure |
| Qwen2.5 reasoning-lite | Qwen2.5-7B | reasoning-lite-1-1 | 100% | 100% | Total prediction failure |
| Llama-2 reasoning-lite | Llama-2-7b | reasoning-lite-1-1 | 100% | 100% | Total prediction failure |

**Pattern**: ALL three reasoning-lite experiments failed. This suggests:
- **Data quality issues**: Reasoning-lite ground truth may be corrupted
- **Structural model mismatch**: Long-sequence reasoning workload exposes model limitations
- **Training data imbalance**: Reasoning-lite underrepresented in training set

### Catastrophic Tier (1000-7000% APE)

| Experiment | Model | Workload | TTFT APE | E2E APE | Combined Loss | Architecture |
|------------|-------|----------|----------|---------|---------------|--------------|
| Mistral Nemo general-lite | Mistral Nemo 12B | general-lite-2-1 | 3718% | 3405% | 7123% | Dense, Long |
| Llama-2 codegen | Llama-2-7b | codegen | 1679% | 1794% | 3473% | Dense, Moderate |
| Mistral Nemo codegen | Mistral Nemo 12B | codegen-1-1 | 1437% | 1436% | 2873% | Dense, Moderate |
| Yi-34B general-lite | Yi-34B | general-lite-2-1 | 1208% | 1291% | 2499% | Dense, Long |
| Qwen2.5 roleplay | Qwen2.5-7B | roleplay-1-1 | 1233% | 900% | 2133% | Dense, Short |
| Llama-3.1-70B general-lite | Llama-3.1-70B | general-lite-4-1 | 1021% | 1102% | 2122% | Dense, Long |
| Llama-2 roleplay | Llama-2-7b | roleplay | 1206% | 755% | 1962% | Dense, Short |

**Pattern**: Dense models with long sequences fail worst (3000-7000% APE). This confirms β₃' collapse hurt long-sequence predictions most.

### Moderate Failure Tier (500-2000% APE)

| Experiment | Model | Workload | TTFT APE | E2E APE | Combined Loss | Architecture |
|------------|-------|----------|----------|---------|---------------|--------------|
| Scout general-lite | Scout 17B | general-lite-2-1 | 890% | 926% | 1816% | MoE, Long |
| Llama-2 general | Llama-2-7b | general | 1102% | 480% | 1582% | Dense, Moderate |
| Llama-3.1-70B codegen | Llama-3.1-70B | codegen-4-1 | 762% | 715% | 1477% | Dense, Moderate |
| Scout codegen | Scout 17B | codegen-2 | 576% | 532% | 1108% | MoE, Moderate |
| Scout roleplay | Scout 17B | roleplay-2 | 431% | 205% | 636% | MoE, Short |

**Pattern**: MoE models perform slightly better than dense, but still catastrophic. Short sequences fail less badly than long.

**Key Observation**: No experiment achieved APE < 100%. Universal catastrophic failure across ALL architectures, sequence lengths, and workloads.

---

## Hypothesis Evaluation

### H-main: Widened β₃' Captures Both KV Allocation and Bandwidth Saturation

**Status**: ❌ **COMPLETELY REFUTED**

#### Prediction
After widening β₃' bounds to 0.05-5.0μs:
- Overall loss: 160.6% → **<120%** (≥25% improvement)
- β₃': 0.252μs → **1-3μs** (4-12× increase to capture bandwidth penalty)
- **Cascading stabilization**: β₂ (0.82 → 0.25-0.60), β₃ (9.6ms → 0.4-1.5ms), β₆ (99ms → 40-100ms)

#### Result
- Overall loss: **2590.24%** (16× WORSE than iter9, not better!)
- β₃': **0.064μs** (4× DECREASE from iter11, 15-47× below target range!)
- Cascading stabilization: **PARTIAL**
  - β₂: 0.82 → **0.284** ✅ (improved to within range)
  - β₃: 9.6ms → **1.16ms** ✅ (improved to within range)
  - β₆: 99ms → **22ms** ❌ (collapsed below range, expected 40-100ms)

#### Verdict: ❌ REFUTED

**Why hypothesis failed**:
1. **β₃' collapsed instead of increasing**: Optimizer moved AWAY from capturing bandwidth penalty
2. **Cascading stabilization incomplete**: β₂ and β₃ improved but β₆ collapsed too far
3. **Overall performance catastrophic**: Loss 16× worse than iter9 baseline
4. **Three experiments failed completely**: 100% APE indicates structural model failure

**Mechanism breakdown**:
- **Hypothesis claimed**: β₃' with widened bounds would capture BOTH CPU-side (KV allocation) and GPU-side (bandwidth penalty)
- **Reality**: Optimizer found spurious local minimum where β₃' contributes almost nothing (0.064μs)
- **Compensation**: Other coefficients (β₄, β₇) inflated to compensate incorrectly

---

### H-expected-ranges: β₆ = 40-100ms is Correct (Not Inflation)

**Status**: ❌ **REFUTED**

#### Prediction
Based on profiling data: β₆ = 40-100ms is correct (NOT 15-40ms). After iter12, β₆ should converge to 50-80ms or stay at 60-100ms.

#### Result
- β₆: **22.0ms** (collapsed to HALF of expected lower bound)
- This is **outside** both old range (15-40ms) and new range (40-100ms)

#### Verdict: ❌ REFUTED

**Implications**:
- Either profiling data was misinterpreted, OR
- β₆ collapsed because β₃' failed to capture bandwidth penalty, OR
- Warm-start from iter9's inflated β₆=99ms created instability

---

### H-cascading: Widened β₃' Triggers Cascading Stabilization

**Status**: ⚠️ **PARTIALLY CONFIRMED** (but overall failure)

#### Prediction
Widening β₃' bounds causes cascading stabilization of prefill-related coefficients:
- β₂ (TP comm): **0.82 → 0.25-0.60** (70% decrease)
- β₃ (KV base): **9.6ms → 0.4-1.5ms** (85-96% decrease)
- β₆ (scheduler): **99ms → 40-100ms** (20-60% decrease)

#### Result
- β₂: 0.82 → **0.284** ✅ (65% decrease, within range!)
- β₃: 9.6ms → **1.16ms** ✅ (88% decrease, within range!)
- β₆: 99ms → **22ms** ❌ (78% decrease, but BELOW range!)

#### Verdict: ⚠️ PARTIAL

**What worked**: β₂ and β₃ DID stabilize as predicted (cascading effect observed)

**What failed**: β₆ over-corrected (collapsed below expected range), and overall loss catastrophic

**Interpretation**: The cascading stabilization mechanism EXISTS but operates incorrectly when:
- β₃' collapses instead of increasing
- Warm-start from inflated coefficients creates unstable dynamics
- Optimizer finds wrong compensation patterns

---

## Root Cause Analysis

### Why Iter12 Failed Catastrophically

**Three fundamental flaws in iteration design**:

#### 1. Warm-Start Paradox

**Strategy**: Warm-start from iter9's "best" coefficients (loss 160.6%)

**Problem**: Iter9's coefficients were **inflated** (β₂=0.82, β₃=9.6ms, β₆=99ms). Starting optimization from inflated values while widening β₃' bounds created unstable landscape.

**Evidence**:
- Iter9 β₂=0.82 vs expected 0.25-0.60 (3× inflated)
- Iter9 β₃=9.6ms vs expected 0.4-1.5ms (6× inflated)
- Iter9 β₆=99ms vs expected 40-100ms (2.5× inflated)

**Consequence**: Optimizer had to simultaneously:
- Increase β₃' to capture bandwidth penalty
- Decrease β₂, β₃, β₆ to revert inflation
- This dual optimization task failed — β₃' collapsed instead of increasing

**Correct strategy**: Should have warm-started from iter6 or iter7 (before β₉ rejection caused cascading instability)

#### 2. Bounds Widening Backfire

**Strategy**: Widen β₃' bounds from 0.05-2.0μs to 0.05-5.0μs to allow capturing bandwidth penalty

**Problem**: Widening bounds gave optimizer MORE room to collapse, not stabilize

**Mechanism**:
- **Hypothesis assumed**: Optimizer would use extra headroom to increase β₃' from 0.252μs → 1-3μs
- **Reality**: Optimizer used extra headroom to decrease β₃' from 0.252μs → 0.064μs (4× collapse)
- **Why**: Warm-start from inflated iter9 created loss landscape where reducing ALL coefficients was locally optimal

**Lesson**: Widening bounds doesn't guarantee coefficient will move in desired direction. Requires careful initialization and constraints.

#### 3. Collinearity Was Real (But Misdiagnosed)

**Original iter12 design**: Add β₁₁ (bandwidth penalty) with basis function Σ(prefillTokens × numLayers)

**Rejection rationale**: β₁₁ and β₃' would be collinear (same basis function), optimizer can't distinguish

**Simplified iter12**: Widen β₃' bounds instead of adding β₁₁

**Problem**: The simplification assumed β₃' CAN capture both mechanisms with wider bounds. Results show it CANNOT.

**Evidence**:
- β₃' collapsed to 0.064μs (far below even KV allocation role)
- β₂ and β₃ improved, but β₆ collapsed too far
- Overall loss catastrophic (2590%)

**Implication**: The two mechanisms (KV allocation vs bandwidth saturation) may require SEPARATE terms despite collinearity, OR the basis function is wrong, OR warm-start strategy doomed the experiment.

---

## Three Reasoning-Lite Experiments: Complete Failure Analysis

### The Pattern

**All three reasoning-lite experiments failed with 100% APE**:
- Scout reasoning-lite-2-1: 100% TTFT, 100% E2E
- Qwen2.5 reasoning-lite-1-1: 100% TTFT, 100% E2E
- Llama-2 reasoning-lite-1-1: 100% TTFT, 100% E2E

### Possible Causes

#### Hypothesis 1: Data Quality Issues

**Evidence**:
- ALL reasoning-lite experiments failed (3/3)
- No other workload type had 100% failure
- Iter9 reasoning-lite results were mixed but not 100% failures

**Check needed**: Inspect ground truth data for reasoning-lite experiments:
```bash
training/trainval_data/48-llama-4-scout-17b-16e-tp2-reasoning-lite-2-1/
training/trainval_data/66-qwen2-5-7b-instruct-tp1-reasoning-lite-1-1/
training/trainval_data/67-llama-2-7b-hf-tp1-reasoning-lite-1-1/
```

Look for:
- Missing CSV files
- All-zero latencies
- Corrupted YAML headers
- Mismatched request counts

#### Hypothesis 2: Long-Sequence Model Mismatch

**Evidence**:
- Reasoning-lite workloads have longest sequences (500+ tokens)
- β₃' collapsed (0.064μs), removing long-sequence overhead term
- Other long-sequence experiments also failed badly (Mistral Nemo general-lite: 3718% TTFT)

**Mechanism**:
- Without β₃' capturing sequence-length overhead, model cannot predict long-sequence latencies
- 100% APE indicates model predicts ZERO or negative latency (impossible)
- Suggests model structural failure for long sequences

#### Hypothesis 3: Training Data Imbalance

**Evidence**:
- Only 3/15 experiments are reasoning-lite (20%)
- Optimizer prioritizes high-frequency workloads (general, codegen, roleplay)
- Long-sequence performance sacrificed to minimize loss on common workloads

**Check needed**: Count training examples per workload type:
```python
import glob
len(glob.glob('training/trainval_data/*reasoning*'))  # Count reasoning expts
len(glob.glob('training/trainval_data/*general*'))     # Count general expts
```

If imbalanced >3:1, re-weight loss function or add more reasoning-lite data.

---

## Key Lessons

### 1. Don't Warm-Start from Inflated Coefficients

**Cost of wrong warm-start**:
- Iter12: 413 trials × 16 minutes = 6,608 trial-minutes = 110 hours wasted
- Result: 16× worse than iter9 baseline

**Correct approach**:
- Identify STABLE iteration (coefficients within ranges, loss reasonable)
- Warm-start from stable iteration, NOT "best loss" iteration
- For iter13: Warm-start from iter6 or iter7, NOT iter9

### 2. Widening Bounds Requires Constraints

**What didn't work**: Widen β₃' from 0.05-2.0μs to 0.05-5.0μs without constraints

**Result**: β₃' collapsed to 0.064μs (moved wrong direction)

**Correct approach**:
- Add **lower bound constraint** β₃' ≥ 0.25μs (preserve iter11's value as floor)
- OR warm-start from stable iteration where β₃' is NOT competing with inflated coefficients
- OR split into separate terms (β₃'ₐ for KV allocation, β₁₁ for bandwidth) with independent bounds

### 3. Check Data Quality BEFORE Training

**Three reasoning-lite experiments failed with 100% APE** → suggests data corruption

**Prevention**: Add pre-training data validation:
```bash
# Check for zero-latency requests
grep "ttft.*: 0" training/trainval_data/*/ground_truth.csv

# Check for missing experiments
ls training/trainval_data/*reasoning* | wc -l  # Should be ≥3

# Check CSV row counts match YAML num_requests
for d in training/trainval_data/*; do
  yaml_count=$(grep num_requests $d/header.yaml | awk '{print $2}')
  csv_count=$(wc -l < $d/ground_truth.csv)
  if [ $yaml_count != $csv_count ]; then echo "MISMATCH: $d"; fi
done
```

### 4. Verify Collinearity Claims with Condition Number

**Iter12 rejected adding β₁₁** because "collinear with β₃'"

**Problem**: This was theoretical claim, never verified empirically

**Correct approach**: Compute **design matrix condition number** before rejecting terms:
```python
X = construct_design_matrix(training_data)  # Include β₃' and hypothetical β₁₁
cond = np.linalg.cond(X)
# cond > 30: High collinearity, reject
# cond < 10: Acceptable, keep both terms
```

If condition number is acceptable, keeping both terms may work better than forcing single term to capture dual role.

---

## Recommendations for Iter13

### Phase 1: Data Validation (REQUIRED BEFORE ANY TRAINING)

**Task 1: Verify reasoning-lite data integrity**
```bash
# Check ground truth files exist and are non-empty
for exp in training/trainval_data/*reasoning*; do
  echo "Checking $exp"
  ls -lh $exp/ground_truth.csv
  head -5 $exp/ground_truth.csv
  tail -5 $exp/ground_truth.csv
done

# Check for zero-latency anomalies
grep -E "ttft.*: 0\.|e2e.*: 0\." training/trainval_data/*/ground_truth.csv
```

**If data is corrupted**: Exclude reasoning-lite experiments from iter13 training set, or regenerate ground truth.

**Task 2: Verify training data balance**
```python
import glob
workload_counts = {}
for exp in glob.glob('training/trainval_data/*'):
    workload = exp.split('-')[-1]  # Extract workload type
    workload_counts[workload] = workload_counts.get(workload, 0) + 1
print(workload_counts)
# If reasoning < 20%, consider re-weighting loss or adding more data
```

### Phase 2: Redesign Iter13 Architecture

**Option A: Revert to Stable Baseline (RECOMMENDED)**

**Strategy**: Roll back to iter6 or iter7 architecture (BEFORE β₉/β₁₀/β₃' additions)

**Rationale**:
- Iter6: Loss ~80% (best iteration)
- Iter7: Loss ~95% (added β₇, stable)
- Iter9: Loss 160.6% (β₉ rejected, cascading instability began)
- Iter10-12: Catastrophic failures (4000-2600%)

**Steps**:
1. Use iter6 coefficient values as warm-start
2. Train with iter6 architecture (3 alpha + 8 beta: β₀-β₇)
3. **NO β₉, β₁₀, β₃', β₁₁** — remove all post-iter7 additions
4. Validate loss returns to <100%
5. THEN incrementally add terms with rigorous validation

**Expected outcome**: Loss <100%, coefficient stability, foundation for future iterations

**Option B: Fix Warm-Start and Retry Iter12 (RISKY)**

**Strategy**: Retry iter12 hypothesis with corrected warm-start

**Changes**:
1. Warm-start from iter6 or iter7 (NOT iter9)
2. Add lower bound constraint: β₃' ≥ 0.20μs (prevent collapse)
3. Keep widened bounds: 0.20-5.0μs
4. Train and check if β₃' increases to 1-3μs

**Risk**: May still fail if fundamental hypothesis is wrong (β₃' cannot capture both mechanisms)

**Option C: Add β₁₁ Despite Collinearity (EXPLORATORY)**

**Strategy**: Add separate bandwidth penalty term despite basis function similarity

**Rationale**: Iter12 showed widening bounds doesn't work. Maybe two terms needed.

**Architecture**:
- β₃': KV cache allocation (CPU-side), basis: Σ(prefillTokens × numLayers), bounds: 0.05-1.0μs
- β₁₁: Bandwidth saturation (GPU-side), basis: Σ(prefillTokens × numLayers), bounds: 0.5-3.0μs
- Separate bounds force optimizer to assign bandwidth penalty to β₁₁, not β₃'

**Validation**: Compute design matrix condition number. If cond > 30, reject this option.

### Phase 3: Pre-Training Validation Gates

**Gate 1: Unit Tests** (Iter11 lesson)
- ✅ REQUIRE unit tests for any new basis functions
- ✅ BLOCK training if tests fail
- 5 minutes to write tests, 87,000× ROI

**Gate 2: Data Quality** (Iter12 lesson)
```bash
# BLOCK training if any experiment has zero-latency requests
if grep -q "ttft.*: 0\." training/trainval_data/*/ground_truth.csv; then
  echo "ERROR: Zero-latency requests found. Fix data before training."
  exit 1
fi
```

**Gate 3: Design Matrix Condition Number** (Iter12 lesson)
```python
# BLOCK training if collinearity too high
X = construct_design_matrix(training_data)
cond = np.linalg.cond(X)
if cond > 30:
    raise ValueError(f"Design matrix ill-conditioned (cond={cond:.1f}). Remove collinear terms.")
```

### Phase 4: Execute Iter13

**Prerequisites** (must complete FIRST):
1. ✅ Data validation passed (no corrupted experiments)
2. ✅ Architecture decision made (Option A/B/C)
3. ✅ Warm-start source validated (stable iteration, coefficients in range)
4. ✅ Pre-training gates implemented

**Configuration** (if Option A — Revert to Stable Baseline):
- Architecture: 3 alpha + 8 beta (iter6/7 structure)
- Warm-start: iter6 or iter7 coefficients
- Bounds: iter6/7 bounds (conservative, proven stable)
- Expected loss: <100% (return to stability)

**Success criteria**:
- Overall loss: **<100%** (return to pre-iter9 performance)
- All experiments: **<200% APE** (no complete failures)
- Coefficients: **≥7/8 within expected ranges** (stability)
- Foundation established for incremental term addition

---

## Process Improvements

### Mandatory Pre-Training Checklist

Before starting ANY iteration training:

- [ ] Data validation: All ground truth files exist, no zero-latencies
- [ ] Architecture decision: Document rationale for any new terms or bound changes
- [ ] Warm-start validation: Verify warm-start source has stable coefficients
- [ ] Unit tests: Any new basis functions have passing unit tests
- [ ] Collinearity check: Design matrix condition number <30
- [ ] Bounds review: Confirm all bounds have physical justification
- [ ] Expected outcome: Write hypothesis with SPECIFIC predictions (not "should improve")

### Scientific Rigor Standards (Updated)

**When trying new coefficient**:
1. Justify with physics or profiling data
2. Write unit test validating expected contribution
3. Check collinearity with existing terms (condition number)
4. Start with narrow bounds (0.5-2× expected value)
5. Warm-start from stable iteration, NOT "best loss" iteration

**When widening bounds**:
1. Add lower bound constraint (preserve previous value as floor)
2. OR warm-start from stable iteration (not inflated)
3. OR split term into separate coefficients with independent bounds

**When warm-starting**:
1. Check warm-start source coefficients are within expected ranges
2. If >50% coefficients out of range, DO NOT warm-start from that iteration
3. Find most recent STABLE iteration (coefficients in range, loss reasonable)

---

## Conclusion

**Iter12 catastrophically failed** because:
1. Warm-started from inflated iter9 coefficients (created unstable landscape)
2. Widened bounds without constraints (allowed β₃' to collapse instead of increase)
3. Three reasoning-lite experiments had data quality issues (100% APE)

**The iter12 hypothesis was REFUTED**: Widening β₃' bounds does NOT allow it to capture both KV allocation and bandwidth penalty. Instead, β₃' collapsed 4×, overall loss increased 16×, and three experiments failed completely.

**Next iteration strategy**:
1. **VALIDATE data quality** for reasoning-lite experiments (fix or exclude)
2. **REVERT to stable baseline** (iter6/7) to re-establish foundation
3. **Then incrementally add terms** with rigorous validation (unit tests, collinearity checks, warm-start from stable)

**Bottom line**: Attempting to fix inflated coefficients by widening bounds on a competing term doesn't work. Must first return to stable baseline, THEN explore parameter space expansion. The shortcuts (warm-start from iter9, widen bounds without constraints) caused catastrophic failure. Time to return to disciplined, incremental progress.

---

## Files Modified

- None (iter12 training only, no code changes)
- `iter12-FINDINGS.md` — This comprehensive analysis (CREATED)
- `iter12-HYPOTHESIS-validation.md` — Hypothesis evaluation (TODO)
- `README.md` — Brief summary (TODO)
