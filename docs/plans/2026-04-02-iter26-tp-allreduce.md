# Iter26: Physics-Based T_tp (TP All-Reduce) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Activate the TP All-Reduce basis function (T_tp) with a physics-derived formula, replacing the current `tTp := 0.0` hardcode, then calibrate β₄ via golden section search.

**Architecture:** Two changes: (1) replace zero T_tp in `evolved_model.go` with `(2·numDenseLayers + numMoELayers) × totalTokens × d × 2 × 2 × (TP-1)/TP / bwHbmUs`, (2) run golden section search on β₄ with all other coefficients frozen at iter25 values.

**Tech Stack:** Go (latency model), Python (golden section search), Optuna/blis training pipeline

---

### Task 1: Add physics-based T_tp to evolved_model.go

**Files:**
- Modify: `sim/latency/evolved_model.go` (line 218–220)

**Step 1: Read and verify the current T_tp section**

```bash
grep -n "T_tp\|tTp\|Beta\[3\]" sim/latency/evolved_model.go
```
Expected: `tTp := 0.0` around line 220, `m.Beta[3]*tTp` in the stepTime sum.

**Step 2: Replace the T_tp computation**

Replace this block:
```go
// T_tp: TP communication time (µs)
// Currently zeroed (β₄=0 in trained-roofline fit; TP cost absorbed into β₅·L).
tTp := 0.0
```

With:
```go
// T_tp: TP All-Reduce communication time (µs)
//
// Each transformer layer performs All-Reduces over NVLink for the attention
// sublayers. Dense layers also All-Reduce their FFN; MoE layers use EP
// All-to-All instead (captured by β₈). We count All-Reduce "units" as:
//   dense layer → 2 units (attention + FFN)
//   MoE layer   → 1 unit  (attention only; FFN replaced by EP All-to-All)
//
// Volume per unit: totalTokens × hiddenDim × 2 bytes (BF16) × 2 (ring phases)
// Denominator: bwHbmUs normalises to µs; β₄ absorbs NVLink/HBM ratio (~0.27 on H100)
//
// Generalisation:
//   TP=1 → (TP-1)/TP = 0 → tTp = 0 (no communication)
//   Dense-only model → numMoELayers=0 → units = 2·numDenseLayers
//   Mixtral (all MoE) → numDenseLayers=0 → units = numMoELayers (half of dense equivalent)
var tTp float64
if m.tp > 1 {
	totalTokens := totalPrefillTokens + totalDecodeTokens
	allReduceUnits := float64(2*m.numDenseLayers + m.numMoELayers)
	tpFactor := float64(m.tp-1) / float64(m.tp)
	tTp = allReduceUnits * totalTokens * float64(m.hiddenDim) * 2.0 * 2.0 * tpFactor / m.bwHbmUs
}
```

**Step 3: Update the doc comment in the formula header** (lines 36–65)

In the "Where β₁/β₁ₐ..." comment, update the β₄ description from:
```
// β₄ is TP communication correction,
```
to:
```
// β₄ is TP All-Reduce correction (absorbs NVLink/HBM bandwidth ratio; ~0.27 on H100),
```

**Step 4: Build and test**

```bash
go build -o blis main.go
go test ./sim/latency/...
```
Expected: both pass.

**Step 5: Verify backward compat — TP=1 model unchanged**

```bash
python3 training/run_blis_and_compute_loss.py \
  --latency-model evolved \
  --alpha-coeffs "15561.959717498621,776.243476414174,45.910232684500556" \
  --beta-coeffs "0.138541,0.0,1.363060401466404,0.0,62.28932987355146,2.7976795228174027,169.36568163371626,427.3,0.0,1.2632" \
  --blis-binary blis \
  --data-dir training/trainval_data \
  --max-workers 4 2>/dev/null | python3.11 -c "import json,sys; d=json.load(sys.stdin); print(f'β₄=0 (TP=1 models unaffected): loss={d[\"overall_loss\"]:.4f}')"
```
Expected: same loss as iter25 (~39.18%) since TP=1 models have tTp=0 and β₄=0.

**Step 6: Commit**

```bash
git add sim/latency/evolved_model.go
git commit -m "feat(latency): iter26 — activate T_tp All-Reduce basis function

Replace tTp:=0 with physics-based TP All-Reduce formula:
  (2·numDenseLayers + numMoELayers) × totalTokens × d × 2B × 2phases × (TP-1)/TP / bwHBM

Dense layers: 2 All-Reduce units (attention + FFN)
MoE layers: 1 All-Reduce unit (attention only; FFN uses EP All-to-All)
TP=1: tTp=0 (backward compat)
β₄ absorbs NVLink/HBM bandwidth ratio

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Create iter26 training artifacts

**Files:**
- Create: `training/iterations/iter26/coefficient_bounds.yaml`
- Create: `training/iterations/iter26/iter26-HYPOTHESIS.md`
- Create: `training/iterations/iter26/iteration_manifest.yaml`

**Step 1: Create coefficient_bounds.yaml**

```yaml
# Iteration 26: Activate T_tp All-Reduce basis function
# β₄ is the only parameter being searched; all others frozen at iter25 best.
# β₄ bounds: [0.0, 0.5] — NVLink/HBM ≈ 0.27 on H100, bound gives slack.
# β₄ initial: 0.0 — reset from meaningless iter25 value (was fitting tTp=0).

alpha_bounds:
  - [100.0, 50000.0]
  - [0.0, 10000.0]
  - [0.0, 50.0]

alpha_initial:
  - 15561.959717498621
  - 776.243476414174
  - 45.910232684500556

beta_bounds:
  - [0.0, 3.0]      # β₁ₐ: prefill compute (frozen)
  - [0.0, 3.0]      # β₂ₐ: decode compute (frozen, =0)
  - [0.3, 3.0]      # β₃:  weight loading (frozen)
  - [0.0, 0.5]      # β₄:  TP All-Reduce correction ← SEARCH TARGET
  - [0.0, 200.0]    # β₅:  per-layer overhead (frozen)
  - [0.0, 500.0]    # β₆:  per-request (frozen)
  - [0.0, 500.0]    # β₇:  per-step constant (frozen)
  - [0.0, 600.0]    # β₈:  per-MoE-layer overhead (frozen)
  - [0.0, 3.0]      # β₁ᵦ: prefill memory (frozen, =0)
  - [0.0, 3.0]      # β₂ᵦ: decode memory (frozen)

beta_initial:
  - 0.138541                  # β₁ₐ (iter25)
  - 0.0                       # β₂ₐ (dropped)
  - 1.363060401466404         # β₃  (iter25)
  - 0.0                       # β₄  RESET — old value was meaningless
  - 62.28932987355146         # β₅  (iter25)
  - 2.7976795228174027        # β₆  (iter25)
  - 169.36568163371626        # β₇  (iter25)
  - 427.3                     # β₈  (iter25)
  - 0.0                       # β₁ᵦ (dropped)
  - 1.2632                    # β₂ᵦ (iter25)
```

**Step 2: Create iter26-HYPOTHESIS.md**

```markdown
# Iteration 26: Physics-Based T_tp (TP All-Reduce)

## Context

Iterations 20–25 reduced loss from 60.19% to 39.18% via β₈ (MoE overhead) and
prefill/decode roofline splits. Throughout, β₄·T_tp = 0 — the TP All-Reduce
communication was entirely ignored, with β₅·L absorbing any residual TP cost.

Iter26 activates T_tp with a physics-derived formula.

## H-main: β₄ Converges to ~0.27 (NVLink/HBM Ratio on H100)

**Prediction**: β₄ converges to approximately 0.25–0.35 (the NVLink bandwidth / HBM
bandwidth ratio for H100: 900 GB/s / 3.35 TB/s ≈ 0.27).

**Causal Mechanism**: T_tp is normalized by bwHbmUs, so β₄ absorbs the actual
interconnect/memory bandwidth ratio. If TP All-Reduce is the dominant communication
cost and NVLink bandwidth is the bottleneck, β₄ ≈ 0.27.

**Diagnostic Clause**: If β₄ converges near 0, TP communication is negligible vs
the other terms. If β₄ > 0.5, the T_tp formula undercounts All-Reduce volume or
β₅ has been over-compensating.

## H-beta5: β₅ May Decrease After T_tp Activation

**Prediction**: β₅ (per-layer overhead, currently 62.3 µs/layer) may decrease
slightly because it was previously absorbing TP communication overhead for
TP=2 and TP=4 experiments.

**Diagnostic Clause**: Run golden section on β₅ after β₄ to check for drift.

## H-loss: Loss Holds or Improves Marginally

**Prediction**: Overall loss stays within ±0.5% of 39.18%. The T_tp formula
has the right physics but the training data (all H100, fixed NVLink topology)
provides limited leverage to improve Scout/dense APE further via TP correction.
The main benefit is formula correctness for new hardware/TP configurations.
```

**Step 3: Create iteration_manifest.yaml**

```yaml
iteration: 26
latency_backend_name: "evolved"
modified_files:
  - "sim/latency/evolved_model.go"
reasoning: |
  Iteration 26: Activate physics-based T_tp (TP All-Reduce) basis function.

  Replace tTp:=0 with:
    (2·numDenseLayers + numMoELayers) × totalTokens × d × 2B × 2phases × (TP-1)/TP / bwHBM

  β₄ is searched via golden section (bounds [0.0, 0.5]).
  All other coefficients frozen at iter25 best values.

  Run Command:
  python3.11 training/scripts/run_golden_section_b4.py
  (see iter26 scripts — or run manually per iteration_manifest instructions)

timestamp: "2026-04-02T12:00:00.000000"
```

**Step 4: Commit artifacts**

```bash
git add training/iterations/iter26/
git commit -m "docs(training): iter26 hypothesis and bounds — T_tp All-Reduce activation"
```

---

### Task 3: Run golden section search on β₄

**Files:**
- No new files — inline script

**Step 1: Rebuild blis (ensure it has the T_tp change)**

```bash
cd /Users/sri/Documents/Projects/inference-sim && go build -o blis main.go
echo "Binary rebuilt at $(date)"
```

**Step 2: Run golden section on β₄**

From `training/` directory:
```python
# Run this inline or as a script from training/
import subprocess, json

ALPHA = "15561.959717498621,776.243476414174,45.910232684500556"
# β₁ₐ, β₂ₐ, β₃, β₄(pivot), β₅, β₆, β₇, β₈, β₁ᵦ, β₂ᵦ
BETA_FIXED = "0.138541,0.0,1.363060401466404,{b4},62.28932987355146,2.7976795228174027,169.36568163371626,427.3,0.0,1.2632"

def eval_b4(b4):
    beta = BETA_FIXED.format(b4=b4)
    r = subprocess.run(
        ["python3", "run_blis_and_compute_loss.py",
         "--latency-model", "evolved",
         "--alpha-coeffs", ALPHA,
         "--beta-coeffs", beta,
         "--blis-binary", "../blis",
         "--data-dir", "trainval_data",
         "--max-workers", "4"],
        capture_output=True, text=True, timeout=300)
    return json.loads(r.stdout)["overall_loss"]

phi = (1 + 5**0.5) / 2
a, b = 0.0, 0.5
c = b - (b - a) / phi
d = a + (b - a) / phi
cache = {}

def cached(x):
    x = round(x, 6)
    if x not in cache:
        cache[x] = eval_b4(x)
        print(f"  β₄={x:.6f} → loss={cache[x]:.4f}", flush=True)
    return cache[x]

print("Golden section: β₄ ∈ [0.0, 0.5]", flush=True)
while abs(b - a) > 0.003:
    if cached(c) < cached(d): b = d
    else: a = c
    c = b - (b - a) / phi
    d = a + (b - a) / phi

best_b4 = (a + b) / 2
best_loss = cached(round(best_b4, 6))
print(f"\nConverged: β₄={best_b4:.4f}, loss={best_loss:.4f}")
print(f"Baseline (iter25): 39.18%")
print(f"Improvement: {39.18 - best_loss:.4f}")
```

Expected output: β₄ converges near 0.27 (or near 0 if TP communication is negligible).

**Step 3: Record result and optionally search β₅**

If best_loss improved by > 0.1%, also run golden section on β₅ in [40, 90]:
```
β₅ search: same pattern with best_b4 fixed, β₅ as pivot
```

**Step 4: Verify with detailed per-experiment evaluation**

```bash
python3 run_blis_and_compute_loss.py \
  --latency-model evolved \
  --alpha-coeffs "15561.959717498621,776.243476414174,45.910232684500556" \
  --beta-coeffs "0.138541,0.0,1.363060401466404,<BEST_B4>,62.28932987355146,2.7976795228174027,169.36568163371626,427.3,0.0,1.2632" \
  --blis-binary ../blis \
  --data-dir trainval_data \
  --evaluate-per-experiment 2>/dev/null | python3.11 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Overall: {d[\"overall_loss\"]:.4f}%')
print(f'TTFT RMSE: {d[\"ttft_rmse\"]:.2f}%, E2E RMSE: {d[\"e2e_rmse\"]:.2f}%')
for e in sorted(d['per_experiment'], key=lambda x: x['ttft_mean_ape'], reverse=True)[:5]:
    print(f'  {e[\"experiment_folder\"].split(\"/\")[-1][:45]:45s} TTFT={e[\"ttft_mean_ape\"]:5.1f}% E2E={e[\"e2e_mean_ape\"]:5.1f}%')
"
```

---

### Task 4: Document and commit iter26 results

**Files:**
- Create: `training/iterations/iter26/inner_loop_results.json`
- Create: `training/iterations/iter26/iter26-FINDINGS.md`

**Step 1: Create inner_loop_results.json** with:
- `best_params.alpha`: same as iter25
- `best_params.beta`: iter25 values but `beta[3]` (β₄) = converged value
- `loss.overall_loss`: verified loss from Task 3 Step 4

**Step 2: Write iter26-FINDINGS.md** covering:
- Converged β₄ value and physical interpretation
- Loss before/after
- Whether β₄ converged near 0.27 (NVLink/HBM ratio) or near 0
- Whether β₅ shifted
- Full coordinate descent history table (iter16 through iter26)

**Step 3: Commit**

```bash
git add training/iterations/iter26/ sim/latency/evolved_model.go
git commit -m "feat(training+latency): iter26 — T_tp All-Reduce, β₄=<VALUE>, loss=<LOSS>%"
```

**Step 4: Push and create PR**

```bash
git push origin training
gh pr create --repo inference-sim/inference-sim \
  --head sriumcp:training --base training \
  --title "feat(latency+training): iter26 — physics-based T_tp All-Reduce" \
  --body "..."
```
