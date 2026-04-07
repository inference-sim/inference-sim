# Iter31: Offset-Subtracted Kernel-Lookup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add launch-overhead offset subtraction to the kernel-lookup model so that γ corrections operate on net compute time rather than raw measured time (which includes CUDA graph overhead).

**Architecture:** Store per-model overhead estimates (δ_gemm, δ_attn) in kernel_profile.yaml. At runtime, subtract these before applying γ multipliers. The offset is also a learnable parameter (passed via beta-coeffs) to allow the optimizer to find the true CUDA graph savings. This should bring step times into the right regime, preventing DES queueing blowup.

**Tech Stack:** Go 1.22+ (runtime changes only), Python 3.10+ (profile regeneration), same training harness.

**Hypothesis:** `training/iterations/iter31/iter31-HYPOTHESIS.md`

---

### Task 1: Add Overhead Fields to KernelProfile + Update YAML Schema

**Files:**
- Modify: `sim/latency/kernel_profile.go` (add GemmOverheadUs, AttnOverheadUs fields)
- Modify: `sim/latency/kernel_profile_test.go` (update test YAML + assertions)

**Step 1: Update KernelProfile struct**

In `sim/latency/kernel_profile.go`, add two fields after the existing tables:

```go
type KernelProfile struct {
    // ... existing fields ...
    AllReduce           Lookup1D  `yaml:"allreduce"`
    MoECompute          *Lookup1D `yaml:"moe_compute,omitempty"`

    // Per-layer launch overhead estimates (µs). Subtracted from lookup values
    // before applying γ corrections, modeling CUDA graph overhead elimination.
    // Warm-start: GemmOverheadUs ≈ gemm(m=1), AttnOverheadUs ≈ attn(b=1, smallest_ctx).
    GemmOverheadUs float64 `yaml:"gemm_overhead_us"`
    AttnOverheadUs float64 `yaml:"attn_overhead_us"`
}
```

**Step 2: Update test YAML in kernel_profile_test.go**

Add to `testProfileYAML`:
```yaml
gemm_overhead_us: 150.0
attn_overhead_us: 12.0
```

Update `TestLoadKernelProfile_ValidYAML` to assert:
```go
assert.InDelta(t, 150.0, p.GemmOverheadUs, 0.01)
assert.InDelta(t, 12.0, p.AttnOverheadUs, 0.01)
```

Also update the invalid-TP test YAML to include the new fields.

**Step 3: Run tests**

```bash
cd /Users/sri/Documents/Projects/inference-sim && go test ./sim/latency/... -run TestLoadKernelProfile -v
```

**Step 4: Commit**

```bash
git add sim/latency/kernel_profile.go sim/latency/kernel_profile_test.go
git commit -m "feat(latency): add gemm_overhead_us and attn_overhead_us to KernelProfile"
```

---

### Task 2: Update KernelLookupModel to Use Offset Subtraction

**Files:**
- Modify: `sim/latency/kernel_lookup.go` (add delta fields, update StepTime formula)
- Modify: `sim/latency/kernel_lookup_test.go` (update test YAML + add offset test)

**Step 1: Update struct and constructor**

In `kernel_lookup.go`, change the `KernelLookupModel` struct to add offset fields
and update the gamma/coefficient mapping:

```go
type KernelLookupModel struct {
    gamma [10]float64
    // New mapping:
    //   [0]=γ₁(gemm), [1]=γ₂(pf_attn), [2]=γ₃(dc_attn),
    //   [3]=γ₄(allreduce), [4]=γ₅(moe),
    //   [5]=δ_gemm(overhead), [6]=δ_attn(overhead),
    //   [7]=γ₆(per-layer), [8]=γ₇(per-req), [9]=γ₈(per-step)
    alpha [3]float64

    // Lookup tables (unchanged)
    gemm           Lookup1D
    contextAttn    Lookup2D
    generationAttn Lookup2D
    allreduce      Lookup1D
    moeCompute     *Lookup1D

    // Per-layer overhead warm-starts from profile (used when delta overrides are 0)
    gemmOverheadWarmStart float64
    attnOverheadWarmStart float64

    // Architecture
    numLayers      int
    numMoELayers   int
    numDenseLayers int
    allReduceUnits int
}
```

In `NewKernelLookupModel`, store the warm-start overheads from the profile:
```go
return &KernelLookupModel{
    // ... existing ...
    gemmOverheadWarmStart: profile.GemmOverheadUs,
    attnOverheadWarmStart: profile.AttnOverheadUs,
}, nil
```

**Step 2: Update StepTime formula**

The key change — compute effective deltas and subtract before γ multiply:

```go
func (m *KernelLookupModel) StepTime(batch []*sim.Request) int64 {
    // ... existing batch scanning code (unchanged) ...

    // Effective overhead: use beta[5]/[6] if non-zero, else warm-start from profile
    deltaGemm := m.gamma[5]
    if deltaGemm == 0 {
        deltaGemm = m.gemmOverheadWarmStart
    }
    deltaAttn := m.gamma[6]
    if deltaAttn == 0 {
        deltaAttn = m.attnOverheadWarmStart
    }

    // γ₁·max(0, T_gemm - δ_gemm)·L
    var tGemm float64
    if totalTokens > 0 {
        raw := m.gemm.Interp1D(totalTokens)
        tGemm = clampPositive(raw-deltaGemm) * L
    }

    // γ₂·max(0, T_pf_attn - δ_attn)·L
    var tPfAttn float64
    if numPrefillRequests > 0 {
        raw := m.contextAttn.Interp2D(numPrefillRequests, avgPrefillISL)
        tPfAttn = clampPositive(raw-deltaAttn) * L
    }

    // γ₃·max(0, T_dc_attn - δ_attn)·L
    var tDcAttn float64
    if totalDecodeTokens > 0 {
        raw := m.generationAttn.Interp2D(totalDecodeTokens, avgDecodeCtx)
        tDcAttn = clampPositive(raw-deltaAttn) * L
    }

    // Rest unchanged: allreduce, moe, additive overheads
    // ...

    stepTime := m.gamma[0]*tGemm +
        m.gamma[1]*tPfAttn +
        m.gamma[2]*tDcAttn +
        m.gamma[3]*tAllReduce +
        m.gamma[4]*tMoE +
        m.gamma[7]*L +
        m.gamma[8]*batchSize +
        m.gamma[9]

    return max(1, clampToInt64(stepTime))
}
```

**Step 3: Update test YAML and add offset-specific test**

Update `writeKernelProfile()` in kernel_lookup_test.go to include:
```yaml
gemm_overhead_us: 148.0
attn_overhead_us: 10.0
```

Add new test:
```go
func TestKernelLookupModel_OffsetSubtraction(t *testing.T) {
    // With overhead subtraction, step time should be LOWER than without
    // Profile has gemm values ~150µs/layer and overhead ~148µs
    // So net GEMM ≈ 2-7µs/layer (just the compute part)
    model, err := testKernelLookupModel(t)
    require.NoError(t, err)

    // Single decode request: totalTokens=1
    // gemm(1) ≈ 150µs, overhead=148µs → net=2µs/layer × 32 = 64µs
    // Plus additive: γ₇·L + γ₈·bs + γ₉ = 20*32 + 1*1 + 50 = 691µs
    // Total ≈ 755µs
    req := &sim.Request{
        InputTokens:   make([]int, 128),
        OutputTokens:  make([]int, 10),
        ProgressIndex: 128,
        NumNewTokens:  1,
    }
    stepTime := model.StepTime([]*sim.Request{req})
    // Should be much lower than without offset (which would give ~5000+µs)
    assert.Less(t, stepTime, int64(3000), "offset subtraction should reduce step time significantly")
    assert.Greater(t, stepTime, int64(100), "step time should still be positive")
}
```

**Step 4: Update warm-start gammas in test helper**

```go
// New mapping: γ₁=1(gemm), γ₂=1(pf_attn), γ₃=1(dc_attn), γ₄=1(allreduce),
// γ₅=0(moe), δ_gemm=0(use profile), δ_attn=0(use profile),
// γ₆=20(layer), γ₇=1(req), γ₈=50(step)
gamma := []float64{1, 1, 1, 1, 0, 0, 0, 20, 1, 50}
```

**Step 5: Run tests**

```bash
go test ./sim/latency/... -run TestKernelLookupModel -v
```

**Step 6: Commit**

```bash
git add sim/latency/kernel_lookup.go sim/latency/kernel_lookup_test.go
git commit -m "feat(latency): add offset subtraction to KernelLookupModel — subtract CUDA graph overhead before γ"
```

---

### Task 3: Update Profile Generator + Regenerate Profiles

**Files:**
- Modify: `training/scripts/generate_kernel_profile.py` (compute + store overheads)
- Modify: `training/kernel_profiles/*.yaml` (regenerated)

**Step 1: Add overhead computation to generate_profile()**

After querying the fused GEMM, compute the overhead estimate:
```python
# CUDA graph overhead estimate: gemm(m=1) is launch-overhead-dominated
gemm_overhead_us = fused_gemm[0]  # first entry in the grid (m=1)

# Attention overhead: smallest batch × smallest ISL
attn_overhead_us = ctx_attn[0][0]  # ISL=smallest, batch_size=smallest
```

Add to the profile dict:
```python
"gemm_overhead_us": round(gemm_overhead_us, 4),
"attn_overhead_us": round(attn_overhead_us, 4),
```

**Step 2: Regenerate all 15 profiles**

```bash
cd /Users/sri/Documents/Projects/inference-sim
python3 training/scripts/generate_kernel_profile.py \
    --from-exp-dir training/trainval_data/ \
    --output-dir training/kernel_profiles/
```

**Step 3: Verify profiles load**

```bash
go test ./sim/latency/... -run TestLoadKernelProfile -v
```

**Step 4: Commit**

```bash
git add training/scripts/generate_kernel_profile.py training/kernel_profiles/
git commit -m "feat(training): add overhead estimates to kernel profiles and regenerate all 15"
```

---

### Task 4: Fix Evaluation Environment + Run Warm-Start

**Files:**
- Modify: `training/run_blis_and_compute_loss.py` (ensure defaults.yaml + model configs accessible)

**Step 1: Fix defaults.yaml path**

The training runner runs from `training/` but BLIS expects `defaults.yaml` in CWD.
Add `--defaults-filepath` passthrough to `build_blis_command`:
```python
# In build_blis_command, add:
defaults_path = os.path.join(os.path.dirname(blis_binary), "defaults.yaml")
if os.path.exists(defaults_path):
    cmd.extend(["--defaults-filepath", defaults_path])
```

**Step 2: Build BLIS and run warm-start evaluation**

```bash
cd /Users/sri/Documents/Projects/inference-sim
go build -o blis main.go

cd training
python3 run_blis_and_compute_loss.py \
    --data-dir trainval_data --blis-binary ../blis \
    --latency-model kernel-lookup \
    --beta-coeffs "1.0,1.0,1.0,1.0,0.0,0.0,0.0,20.0,1.0,50.0" \
    --alpha-coeffs "0.0,0.0,0.0" \
    --kernel-profiles-dir kernel_profiles \
    --max-workers 8 --evaluate-per-experiment
```

The warm-start uses δ_gemm=0 and δ_attn=0 (beta[5]=beta[6]=0), which triggers the
profile warm-start path in Go. γ₁-γ₄=1.0 corrections on NET compute (after overhead
subtraction) should give dramatically better results than iter30.

**Step 3: Also run trained-physics baseline for comparison**

```bash
python3 run_blis_and_compute_loss.py \
    --data-dir trainval_data --blis-binary ../blis \
    --latency-model trained-physics \
    --beta-coeffs "0.152128,0.0,1.36252915,0.752037,32.09546717,4.41684444,126.024825,481.8613888,0.0,1.94710771" \
    --alpha-coeffs "15563.199579,777.3455,45.907545" \
    --max-workers 8 --evaluate-per-experiment
```

**Step 4: Record results in iter31-FINDINGS.md**

**Step 5: Commit**

```bash
git add training/ && git commit -m "feat(training): iter31 warm-start evaluation with offset subtraction"
```

---

### Task 5: Optimize Coefficients (if warm-start is promising)

**Files:**
- Create: `training/iterations/iter31/coefficient_bounds.yaml`
- Modify: training optimization scripts as needed

If the warm-start loss is < 100%, optimize all 10 coefficients using CMA-ES
or sequential golden section. Bounds:

```yaml
bounds:
  - [0.5, 2.0]   # γ₁: GEMM correction
  - [0.5, 2.0]   # γ₂: prefill attention correction
  - [0.5, 2.0]   # γ₃: decode attention correction
  - [0.0, 2.0]   # γ₄: AllReduce correction
  - [0.0, 5.0]   # γ₅: MoE correction
  - [0.0, 300.0]  # δ_gemm: GEMM overhead (µs)
  - [0.0, 300.0]  # δ_attn: attention overhead (µs)
  - [0.0, 100.0]  # γ₆: per-layer (µs)
  - [0.0, 20.0]   # γ₇: per-request (µs)
  - [0.0, 200.0]  # γ₈: per-step (µs)
```

Target: overall loss < 32% (beat trained-physics by > 2.5 points).

---

## Execution Order

```
Task 1 (profile schema) → Task 2 (Go formula) → Task 3 (regenerate profiles) → Task 4 (evaluate) → Task 5 (optimize)
```

All tasks are sequential. Tasks 1-3 are code changes (~30 min). Task 4 is evaluation (~10 min). Task 5 is optimization (may take multiple runs).
