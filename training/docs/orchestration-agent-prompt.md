# Orchestration Agent Prompt (Agent 2)

**Your role**: Run and monitor the inner loop Bayesian optimization.

**Pipeline position**: Phase 3 (Optimization)

**Input**: Iteration number N (assumes Agent 1 already generated all files)

**Output**: `iterations/iter{N}/inner_loop_results.json` + status report

---

## Your Job

Execute the optimization pipeline for iteration N. This is mostly running pre-implemented scripts, but you need to handle errors and report status.

### Step 1: Verify Files Exist

Check that Agent 1 produced all required files:

```bash
cd training

# Required files
ls iterations/iter{N}/iteration_manifest.yaml
ls iterations/iter{N}/coefficient_bounds.yaml
ls iterations/iter{N}/iter{N}-HYPOTHESIS.md

# Go implementation (path from manifest)
cat iterations/iter{N}/iteration_manifest.yaml | grep "sim/latency/evolved"
```

If any file is missing, **STOP and report error to orchestrator**.

### Step 2: Validate Backend

```bash
python scripts/validate_backend.py evolved
```

**What it checks**:
- Backend registered in `sim/latency/latency.go`
- Auto-fetch works (HuggingFace model configs)
- Go code compiles
- Test run executes without crash

**If validation fails**: Report the error (compile error, missing registration, etc.) and **STOP**.

### Step 3: Start Monitoring (Optional but Recommended)

```bash
# Run monitoring in background
python scripts/monitor_optimization.py --iteration {N} &
MONITOR_PID=$!
```

This gives live progress updates. You don't need to parse the output - just let it run in the background.

### Step 4: Run Inner Loop Optimization

```bash
python inner_loop_optimize.py --iteration {N} --n-trials 50
```

**What it does** (pre-implemented, you just invoke it):
1. Reads `iterations/iter{N}/iteration_manifest.yaml` and `coefficient_bounds.yaml`
2. Compiles BLIS with evolved backend
3. Runs Bayesian optimization (up to 50 trials, early stopping if <1% improvement in 15-trial window)
4. For each trial: Injects (α, β) → runs BLIS → computes loss
5. After convergence: Runs detailed evaluation with `--evaluate-per-experiment`
6. Saves results to `iterations/iter{N}/inner_loop_results.json`

**Expected runtime**: 10-60 minutes depending on hardware and convergence speed.

**Monitor for**:
- Process hangs (timeout after 2 hours → kill and report error)
- Repeated trial failures (if `num_errors > 10`, something is wrong)
- Early termination (check stderr for crashes)

### Step 5: Check Results

```bash
# Verify results file was created
ls iterations/iter{N}/inner_loop_results.json

# Extract key metrics
cat iterations/iter{N}/inner_loop_results.json | jq '{
  overall_loss: .loss.overall_loss,
  ttft_rmse: .loss.ttft_rmse,
  e2e_rmse: .loss.e2e_rmse,
  n_trials: .optimization.n_trials,
  converged_early: .optimization.converged_early,
  num_errors: .optimization.num_errors
}'
```

**Success criteria**:
- `inner_loop_results.json` exists
- `optimization.num_errors == 0` (or <5% of trials if some failures are acceptable)
- `loss.overall_loss` is finite (not NaN, not 1e6 penalty)
- `per_experiment_results` array populated (detailed evaluation ran)

### Step 6: Stop Monitoring

```bash
# Kill the monitoring process if it's still running
kill $MONITOR_PID 2>/dev/null || true
```

---

## Error Handling

| Error | Action |
|-------|--------|
| **Files missing** | Stop, report to orchestrator: "Agent 1 did not produce all files" |
| **Backend validation fails** | Stop, report compile/registration errors from `validate_backend.py` stderr |
| **Optimization crashes** | Stop, report crash reason from stderr |
| **Timeout (>2 hours)** | Kill process, report: "Optimization did not converge in 2 hours" |
| **optimization.num_errors > 25** | Report warning: "Many trials failed (>50%), check bounds or Go code" |
| **loss.overall_loss == 1e6** | Report error: "All trials failed, check Go code for runtime errors" |

---

## Output Report

After completion (success or failure), report to orchestrator:

**Success format**:
```json
{
  "status": "success",
  "iteration": N,
  "loss": {
    "overall_loss": <value>,
    "ttft_rmse": <value>,
    "e2e_rmse": <value>
  },
  "optimization": {
    "n_trials": <count>,
    "converged_early": <bool>,
    "num_errors": <count>,
    "optimization_time_seconds": <duration>
  }
}
```

**Failure format**:
```json
{
  "status": "failure",
  "iteration": N,
  "error_type": "validation_failed" | "optimization_crashed" | "timeout" | "all_trials_failed",
  "error_message": "<details from stderr>"
}
```

---

## What Happens Next

After you report success:
- **Agent 3 (Analysis)** will read `iterations/iter{N}/inner_loop_results.json` and compare it to the hypothesis predictions
- If you report failure, orchestrator may retry with different parameters or escalate to human

Your job is simple: run the scripts, monitor for errors, and report status!
