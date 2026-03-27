# Outer-Inner Loop Contract

## Overview

This document specifies the interface contract between the outer loop (agentic strategy evolution) and inner loop (automated Bayesian optimization script). The outer loop is an AI agent that generates basis function code. The inner loop is a pre-implemented Python script (`inner_loop_optimize.py`) that compiles BLIS and runs optimization.

**Key responsibilities:**
- **Outer loop (agent)**: Generates Go source code, declares backend name, specifies coefficient bounds
- **Inner loop (script)**: Reads manifest, compiles BLIS, runs Bayesian optimization (50-100 trials), returns optimal coefficients

**Interface mechanism**: The outer loop writes three files to disk. The inner loop script reads these files and executes automatically.

**Critical constraint**: Basis functions must be compiled into BLIS before optimization starts. Coefficients are injected at runtime via CLI flags (no recompilation per trial).

## Outer Loop Deliverables

The outer loop agent must provide three files per iteration:

### 1. Iteration Manifest (`iteration_manifest.yaml`)

Declares what was generated and why.

**Required fields:**
- `iteration`: Iteration number
- `latency_backend_name`: Name used in Go `Register()` call (e.g., `"evolved"`)
- `modified_files`: List of Go source files (e.g., `["sim/latency/evolved_model.go"]`)
- `reasoning`: Agent's explanation of changes
- `timestamp`: ISO 8601 timestamp

**Agent guarantees:**
- Backend name matches `Register()` call in Go code
- All declared files exist
- Go code compiles without errors
- Backend implements `LatencyModel` interface correctly

### 2. Go Source Code

Implements `LatencyModel` interface with agent-designed basis functions.

**Required methods:**
- `StepTime(modelConfig, hwConfig, batchState) float64` - basis function logic
- `QueueingTime(req) int64` - request-level API overhead
- `OutputTokenProcessingTime() int64` - per-token post-processing

**Backend registration:**
```go
func init() {
    Register("<latency_backend_name>", func(cfg LatencyConfig) (LatencyModel, error) {
        // Return model struct with Alpha and Beta populated from cfg
    })
}
```

### 3. Coefficient Bounds (`coefficient_bounds.yaml`)

Search space for Bayesian optimization.

**Required fields:**
- `alpha_bounds`: Exactly 3 bounds `[[min, max], [min, max], [min, max]]`
- `beta_bounds`: N bounds matching number of beta coefficients in `StepTime()`

**Agent responsibility:** Bounds must be physically plausible and justified.

## Inner Loop Responsibilities

### Phase 1: Setup (Once Per Outer Iteration)

1. Read `iteration_manifest.yaml` → get `latency_backend_name`
2. Verify all files in `modified_files` exist
3. Compile BLIS: `go build -o blis main.go`
4. Load `coefficient_bounds.yaml`

### Phase 2: Optimization (Up to 50-100 Trials)

1. Bayesian optimizer proposes candidate (α, β)
2. Run BLIS with coefficients injected via CLI:
   ```
   ./blis run --latency-model <backend_name> \
     --alpha-coeffs <α₀>,<α₁>,<α₂> \
     --beta-coeffs <β₁>,...,<βₙ> \
     --experiments trainval_data/
   ```
3. Parse loss from output: `RMSE[APE(TTFT)] + RMSE[APE(E2E)]`
4. Update surrogate model and repeat
5. **Convergence check**: Stop early if best loss hasn't improved >1% in last 50 trials

### Phase 3: Post-Convergence Evaluation

After finding optimal (α*, β*):
1. Run detailed evaluation: `--evaluate-per-experiment` flag
2. Return optimal coefficients, loss, and per-experiment diagnostics

## Coefficient Injection

BLIS supports runtime coefficient injection (no recompilation needed):

**CLI flags:**
- `--alpha-coeffs <float>,<float>,<float>` (3 values)
- `--beta-coeffs <float>,...,<float>` (N values)

**Example:**
```bash
./blis run --latency-model evolved \
  --alpha-coeffs 0.00032,0.000045,0.000038 \
  --beta-coeffs 0.00087,0.00124,0.000021
```

**Validation:** BLIS requires both flags together (enforced at CLI level).

## Example: Iteration 3

**Outer loop outputs:**

`iteration_manifest.yaml`:
```yaml
iteration: 3
latency_backend_name: "evolved"
modified_files: ["sim/latency/evolved_model.go"]
reasoning: "Added TP communication term after observing 15% underprediction at TP=4"
timestamp: "2026-03-27T14:30:00Z"
```

`coefficient_bounds.yaml`:
```yaml
alpha_bounds: [[0.0, 0.001], [0.0, 0.0001], [0.0, 0.0001]]
beta_bounds: [[0.0, 0.002], [0.0, 0.002], [0.0, 0.00005]]  # 3 betas
```

`sim/latency/evolved_model.go`: (agent-generated implementation)

**Inner loop:**
1. Reads manifest → `latency_backend_name = "evolved"`
2. Verifies `sim/latency/evolved_model.go` exists
3. Compiles BLIS
4. Runs 50 Bayesian optimization trials with `--latency-model evolved`

## Performance

- **One compilation per outer iteration**: ~5-10 seconds
- **50-100 evaluations per iteration**: No recompilation (fast)
- **Total overhead across ~5 iterations**: ~25-50 seconds

---

## For Outer Loop Agent Implementers

**If you are implementing the outer loop agent**, see **[outer-loop-specs.md](outer-loop-specs.md)** for the complete prompt specification. That document contains:
- Detailed input/output format requirements
- Go code templates with examples
- Error pattern analysis guidance
- Reasoning process guidelines
- Validation checklist

**If you are using the inner loop script**, see:
- CLI usage: `python inner_loop_optimize.py --help`
- Implementation: `training/inner_loop_optimize.py`
- Loss computation: `training/run_blis_and_compute_loss.py`
