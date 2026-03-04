# Iter 4 Handoff Brief — Joint Coefficient Optimization

_Date: 2026-03-04_
_Branch: `training` (main work), `iter4-sil` (worktree at `.worktrees/iter4-sil/`)_

## What Was Accomplished

### Documents (committed to `training` branch)
- `training/problem.md` — Updated with H30-H32 findings, InterStepOverhead architecture, two-phase strategy, cascade-aware constraints
- `training/ledger.md` — H30-H32 diagnostics section, Iter 4 strategy, Stage A failure log
- `training/iter4-bundle.md` — Hypothesis bundle (Design Review CONVERGED Round 4: 0C+0I)

### Go Implementation (committed to `training` branch)
- `sim/latency_model.go` — New `InterStepOverhead() int64` method on LatencyModel interface
- `sim/latency/crossmodel.go` — CrossModel returns `interStepCoeffs[0]` when set, 0 otherwise
- `sim/latency/latency.go` — Blackbox and Roofline return 0
- `sim/config.go` — `InterStepCoeffs []float64` added to LatencyCoeffs, `NewLatencyCoeffs` updated (R4)
- `sim/simulator.go` — `scheduleNextStep()` applies `InterStepOverhead()` in both paths; `executeBatchStep()` records StepDurations/StepPrefillTokens/StepDecodeTokens
- `sim/metrics.go` — Three new per-step metric fields
- 25 files total updated, all tests pass (`go test ./...`)

### Python Scripts (committed)
- `training/iter4_warmstart.py` — Phase 1 δ extraction from BATCH_SUMMARY step gaps
- `training/iter4_optimize.py` — Phase 2 SIL optimization framework (needs revision per finding below)
- `training/iter4_grid.py` — Quick δ grid search (Stage A — proved the entanglement)
- `training/cmd/replay/main.go` — Parameterized replay binary (`--beta`, `--alpha`, `--delta`, `--seed`, `--emit-step-metrics`), unit bug fixed (ticks→ms)

### Phase 1 Diagnostic Results
Measured inter-step gaps from 8,690 consecutive BATCH_SUMMARY step pairs:

| Model | TP | Median δ (µs) | Batch-size corr |
|-------|-----|---------------|-----------------|
| llama-2-7b | 1 | 7,774 | +0.80 |
| codellama-34b | 2 | 14,275 | -0.47 |
| llama-2-70b | 4 | 17,439 | -0.18 |
| mixtral-8x7b | 2 | 18,291 | +0.62 |

## The Critical Finding

**Stage A grid search FAILED.** Adding δ on top of Iter 3 β/α produces millions-of-percent error because **β already implicitly absorbed inter-step overhead** through journey-constraint NNLS fitting (which uses wall-clock intervals that include real vLLM's inter-step gaps).

Evidence:
- mixtral-8x7b at δ=0: TTFT +2.4% (near-perfect) → adding ANY δ destroys it
- llama-2-7b at δ=0: TTFT -35.2% → needs help, but δ helps 7b while destroying mixtral

**Root cause:** Iter 3 used two NNLS blocks:
- Block A: step-level observations (GPU compute only, `step.duration_us`)
- Block B: journey intervals (wall-clock, includes inter-step gaps)

Block B pulled β toward values that implicitly include inter-step overhead. The amount absorbed varies by model because the relative magnitude of inter-step vs compute differs per architecture.

## The Path Forward

**Joint refit of all 7 parameters (β₀, β₁, β₂, β₃, α₀, α₂, δ₀) using BLIS replay as the objective.** Two approaches:

### Approach A: Step-Only β + Joint α/δ Search
1. Refit β using ONLY Block A (step-level data, no journey constraints) — isolates GPU compute
2. Analytically extract δ₀ per model from measured gaps (already done)
3. Search (α₀, α₂) with BLIS replay to close the remaining gap
4. Use per-model δ₀ or `δ₀ = δ_base + δ_tp · I(TP>1)` if global doesn't work

### Approach B: Full 7-Param CMA-ES
1. Warm-start from: step-only β (no journey constraints), measured α₀, measured δ₀
2. CMA-ES with σ₀=20% (wider than planned, since warm-start is less reliable without journey constraints)
3. Multi-signal loss on 10 training experiments
4. Physics guard-rails: β ≥ 0, δ₀ ≥ 0, all bounded

### Key Constraint
The Iter 3 β values (116.1, 1226.9, 19.9, 9445.2) are NOT good warm-starts for joint optimization because they include absorbed inter-step overhead. **Step-only β must be recomputed first.** The `iter3_physics.py` script has the code — just run Block A without Block B.

## Files to Read First
1. `training/iter4-bundle.md` — Full hypothesis bundle (7 arms, converged design)
2. `training/ledger.md` — Iteration history including Stage A failure
3. `training/problem.md` Section 11f — Two-phase strategy specification
4. `training/iter3_physics.py` — Has the NNLS fitting code (Block A + Block B separation)
5. `training/iter4_grid.py` — The grid search that proved the entanglement

## How to Build and Run
```bash
# Build replay binary
go build -o replay training/cmd/replay/main.go

# Run one experiment (no delta — baseline)
./replay --input training/replay_data/20260217-231439-llama-2-7b-tp1-general.json

# Run with delta
./replay --input training/replay_data/20260217-231439-llama-2-7b-tp1-general.json --delta 5000

# Run with custom coefficients
./replay --input training/replay_data/20260217-231439-llama-2-7b-tp1-general.json \
  --beta 116.1,1226.9,19.9,9445.2 --alpha 13732,0,860.6 --delta 10000

# Ground truth for comparison
cat training/replay_data/20260217-231439-llama-2-7b-tp1-general_ground_truth.json | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Real TTFT={d[\"ttft\"][\"mean_ms\"]:.1f}ms E2E={d[\"e2e\"][\"mean_ms\"]:.1f}ms')"
```

## Worktree Status
`.worktrees/iter4-sil/` exists on branch `iter4-sil`. Can be removed with `git worktree remove .worktrees/iter4-sil` once artifacts are confirmed copied. The `sim/cluster/cluster.go` diff in the worktree (step metrics aggregation) should be reviewed — may need to be applied to the main branch.
