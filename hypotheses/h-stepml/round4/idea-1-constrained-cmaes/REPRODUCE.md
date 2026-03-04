# Reproduction Guide: idea-1-constrained-cmaes (Round 4)

> Auto-generated after WP3 completed for this idea.

## Base State

- **Branch:** stepml-experiments
- **Commit:** 31f4eb9e0465abcdd0bf47aefac518b7a5292c55
- **Date:** 2026-03-02 16:03:00

## Step 1: Create isolated worktree

```bash
git worktree add ../reproduce-idea-1-constrained-cmaes 31f4eb9e0465abcdd0bf47aefac518b7a5292c55 --detach
cd ../reproduce-idea-1-constrained-cmaes
```

## Step 2: Copy experiment artifacts

```bash
# Copy this idea's artifacts (HYPOTHESIS.md, sub-hypothesis dirs, trained models)
cp -r hypotheses/h-stepml/round4/idea-1-constrained-cmaes/ hypotheses/h-stepml/round4/idea-1-constrained-cmaes/

# Copy shared infrastructure
cp -r hypotheses/h-stepml/shared/ hypotheses/h-stepml/shared/

# Copy problem.md (input context)
cp hypotheses/h-stepml/problem.md hypotheses/h-stepml/problem.md
```

## Step 3: Apply Go changes

This idea modified Go code. Apply the patch:

```bash
git apply hypotheses/h-stepml/round4/idea-1-constrained-cmaes/go_changes.patch
go build -o simulation_worker main.go
go test ./sim/latency/...
```

### Patch summary

Modified files:
```
cmd/root.go
sim/config.go
sim/latency/latency.go
```

The Go changes add StepML model path configuration and factory dispatch for the latency model.

## Step 4: Modified files

```
cmd/root.go
hypotheses/h-stepml/problem.md
sim/config.go
sim/latency/latency.go
```

## Step 5: Re-run experiments

```bash
# Activate Python environment
source .inference-sim-env/bin/activate

# Build BLIS binary first
go build -o simulation_worker main.go

# Run the CMA-ES experiment
cd hypotheses/h-stepml/round4/idea-1-constrained-cmaes
python3 run_experiment.py
```

Sub-hypothesis directories and their key files:

```
h1-constrained-cmaes/
  summary.json
  artifacts/codellama-34b_optimized.json
  artifacts/llama-2-70b_optimized.json
  artifacts/llama-2-7b_optimized.json
  artifacts/mixtral-8x7b-v0-1_optimized.json
h2-pareto-sweep/
  alpha_0.7/summary.json
  alpha_0.7/artifacts/
```

**Note:** H3 (LOMO) and H4 (LOWO) were not run — H1 triggered the short-circuit criterion (51.2% E2E >> 25% threshold).

Runtime: ~60 minutes for H1 + H2 (CMA-ES dominates).

## Step 6: Validate via BLIS

The `run_experiment.py` script handles all BLIS validation internally. Key results are in `h1-constrained-cmaes/summary.json` and `h2-pareto-sweep/alpha_*/summary.json`.

## Key Results

See `FINDINGS_SUMMARY.md` in this directory for the full results table.

**Headline result:** REFUTED. Constraining CMA-ES parameter bounds to "physical" ranges destroyed E2E accuracy (51.2% mean E2E vs R3's 15.1%), confirming that the parameters compensate for unmodeled simulation dynamics.
