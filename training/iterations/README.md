# Iteration Artifacts Directory

This directory contains all artifacts for each outer loop iteration, organized by iteration number.

## Directory Structure

```
iterations/
├── iter0/
│   ├── iter0-HYPOTHESIS.md                # Phase 1: Hypothesis design
│   ├── iteration_manifest.yaml            # Backend declaration
│   ├── coefficient_bounds.yaml            # Search space + initial values
│   ├── inner_loop_results.json            # Phase 3: Optimization results
│   ├── iter0-HYPOTHESIS-validation.md     # Phase 4: Hypothesis verdicts
│   └── iter0-FINDINGS.md                  # Phase 4: Error analysis + principles
├── iter1/
│   ├── iter1-HYPOTHESIS.md
│   ├── iteration_manifest.yaml
│   ├── coefficient_bounds.yaml
│   └── ...
└── ...
```

## File Descriptions

### Phase 1: Hypothesis Design
- **`iter{N}-HYPOTHESIS.md`**: Hypothesis bundle with predictions, causal mechanisms, code citations, and success criteria

### Phase 2: Implementation
- **`iteration_manifest.yaml`**: Declares backend name, modified Go files, reasoning, timestamp
- **`coefficient_bounds.yaml`**: Alpha/Beta bounds and initial values for Bayesian optimization
- **Go files**: `sim/latency/evolved_*.go` (not stored here - in project root)

### Phase 3: Optimization
- **`inner_loop_results.json`**: Best coefficients, loss, per-experiment diagnostics, error log

### Phase 4: Analysis
- **`iter{N}-HYPOTHESIS-validation.md`**: Verdict for each hypothesis (✅ CONFIRMED / ❌ REJECTED / ⚠️ PARTIAL)
- **`iter{N}-FINDINGS.md`**: Error pattern analysis, causal explanations, principles for next iteration

### Phase 5: Cross-Validation (if all hypotheses confirmed)
- **`cv1_results.json`**, **`cv1_report.md`**: Leave-One-Model-Out results
- **`cv2_results.json`**, **`cv2_report.md`**: Leave-One-Workload-Out results
- **`cv3_results.json`**, **`cv3_report.md`**: Leave-One-TP-Out results
- **`iter{N}-GENERALIZATION-FINDINGS.md`**: Analysis of all CV results with pass/fail verdicts per test, workload-agnostic constraint validation, and overall generalization assessment