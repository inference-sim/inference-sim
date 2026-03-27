# Agentic Latency Model Training — Problem Statement

## Executive Summary

Build a two-loop agentic system that automatically discovers and fits physics-informed latency models for BLIS. The outer loop (3-agent pipeline: Design → Orchestration → Analysis) proposes basis functions grounded in GPU architecture and vLLM internals, runs optimization, and extracts principles. The inner loop (Bayesian optimization) fits coefficients to achieve state-of-the-art prediction accuracy across model architectures, tensor parallelism configurations, and workload patterns.

**Agent Instructions**: See individual agent prompts: [`design-agent-prompt.md`](design-agent-prompt.md), [`orchestration-agent-prompt.md`](orchestration-agent-prompt.md), [`analysis-agent-prompt.md`](analysis-agent-prompt.md).

---

## Background

BLIS supports four latency backends, each with limitations:
- **Roofline**: Analytical (FLOPs/bandwidth bounds) but ignores overheads
- **Blackbox**: Empirical per-model coefficients, no generalization
- **Cross-model**: Linear regression with fixed basis functions
- **Trained-roofline**: Manual coefficient tuning, requires iterative feature engineering

**Opportunity**: Replace manual feature engineering with agentic reasoning that discovers optimal basis functions while Bayesian optimization fits coefficients.

---

## The Two-Loop Architecture

### Outer Loop: Agentic Strategy Evolution (3-Agent Pipeline)

**Role**: Evolve latency model structure through physics-informed reasoning using a sequential 3-agent pipeline.

**⚠️ CRITICAL: This follows the [Strategy Evolution](../../docs/methodology/strategy-evolution.md) methodology**, adapted for latency model training:
- **Hypothesis Bundle Design**: Agent 1 generates hypothesis bundle with **mandatory H-main** (mechanism claim + predictions)
- **Implementation & Optimization**: Agent 2 implements and optimizes (Bayesian inner loop)
- **Verification**: Agent 3 verifies predictions against results
- **Principle Extraction**: Agent 3 extracts principles from both confirmations AND prediction errors → guides next iteration

Each iteration commits to predictions BEFORE seeing results. Prediction errors are the primary learning signal.

**Pipeline** (following [Strategy Evolution](../../docs/methodology/strategy-evolution.md)):

1. **Agent 1: Design** ([`design-agent-prompt.md`](design-agent-prompt.md)) — **Hypothesis Bundle Design**
   - **Input**: Previous iteration findings (if N > 0), training data, references
   - **Process**: Research → hypothesis bundle design (with **mandatory H-main**) → Go implementation
   - **Output**: `iter{N}-HYPOTHESIS.md`, `iteration_manifest.yaml`, `coefficient_bounds.yaml`, `evolved_model.go`
   - **Key requirement**: Must generate H-main with (1) quantitative prediction, (2) causal mechanism, (3) diagnostic clause
   - **Follows**: [Hypothesis Bundles](../../docs/methodology/hypothesis-bundles.md) structure and examples

2. **Agent 2: Orchestration** ([`orchestration-agent-prompt.md`](orchestration-agent-prompt.md)) — **Implementation & Optimization**
   - **Input**: Iteration number N (assumes Agent 1 completed)
   - **Process**: Validate backend → run Bayesian optimization → monitor progress
   - **Output**: `inner_loop_results.json` + status report
   - **Role**: Executes the optimization loop that tests Agent 1's predictions

3. **Agent 3: Analysis** ([`analysis-agent-prompt.md`](analysis-agent-prompt.md)) — **Verification & Principle Extraction**
   - **Input**: Optimization results from Agent 2
   - **Process**: Verify predictions → extract principles from confirmations AND errors → run CV if warranted
   - **Output**: `iter{N}-HYPOTHESIS-validation.md`, `iter{N}-FINDINGS.md`, `iter{N}-GENERALIZATION-FINDINGS.md` (if CV ran), CV results
   - **Key focus**: Evaluate H-main first, use diagnostic clauses to direct investigation when predictions fail

**End-to-End Workflow** (Strategy Evolution):

Each iteration N follows this sequence:
1. **Agent 1** reads previous iterations (0 to N-1) → generates hypothesis bundle with **mandatory H-main** → implements basis functions in Go → produces 4 files
2. **Agent 2** validates backend → runs Bayesian optimization → produces `inner_loop_results.json`
3. **Agent 3** compares H-main + other predictions vs results → validates verdicts → extracts principles from both successes AND failures → produces validation + findings + CV results (if warranted)
4. If not converged, Agent 1 uses Agent 3's findings (especially diagnostic clauses from failed predictions) to design iter{N+1}

**Key insight from Strategy Evolution**: Prediction errors are as valuable as confirmations — they reveal gaps in our understanding of vLLM/GPU dynamics.

### Inner Loop: Bayesian Optimization (Pre-implemented)

**Role**: Find best-fitting coefficients for a fixed set of basis functions.

**Script**: `training/inner_loop_optimize.py` (pre-implemented)

**Process**:
1. Read manifest, bounds, Go code from `iterations/iter{N}/`
2. Compile BLIS with evolved backend
3. Run Bayesian optimization (up to 50 trials, early stopping)
4. For each trial: Inject coefficients → run BLIS → compute loss
5. Save results to `iterations/iter{N}/inner_loop_results.json`

**Loss Function**:
```
overall_loss = RMSE[APE(mean_TTFT_per_exp)] + RMSE[APE(mean_E2E_per_exp)]
```

**Output JSON Structure** (`iterations/iter{N}/inner_loop_results.json`):
```json
{
  "loss": {
    "overall_loss": float,  // Sum of ttft_rmse + e2e_rmse
    "ttft_rmse": float,     // RMSE[APE(mean_TTFT_per_exp)] across 15 experiments
    "e2e_rmse": float       // RMSE[APE(mean_E2E_per_exp)] across 15 experiments
  },
  "optimization": {
    "n_trials": int,
    "converged_early": bool,
    "num_errors": int
  },
  "best_params": {
    "alpha": [α₀, α₁, α₂],
    "beta": [β₀, β₁, ..., βₙ]
  },
  "per_experiment_results": [
    {
      "experiment_folder": str,
      "ttft_mean_ape": float,  // APE for this experiment
      "e2e_mean_ape": float
    },
    ...
  ]
}
```

**Invocation**:
```bash
cd training && python inner_loop_optimize.py --iteration N --n-trials 50
```

---

## Problem Definition

**Given**:
- **Training data**: 15 experiments covering Llama/Mistral/Qwen/Yi (dense) + Scout (MoE), TP ∈ {1,2,4}, 4 workload types
- **Evaluation**: `run_blis_and_compute_loss.py` computes RMSE[APE] on TTFT and E2E
- **Constraints**: Basis functions must be workload-agnostic (use only batch/model/hardware features)

**Produce**:
- **Request-level alpha [α₀, α₁, α₂]**: Fixed API overhead, per-input-token, per-output-token costs
- **Step-level beta [β₁, ..., βₙ]**: Coefficients for basis functions in `StepTime(batch)`
- **Basis functions** f₁, ..., fₙ that capture GPU compute/memory/communication operations

**Prediction Function**:
```
StepTime(batch) = Σᵢ βᵢ · fᵢ(batch, model, hardware)
QueueingTime(req) = α₀ + α₁ × num_input_tokens
OutputTokenProcessingTime() = α₂
```

**Target Accuracy**: MAPE < 10% on TTFT, ITL, E2E across all 15 experiments

**Hypothesis Bundle Requirements** (from [Strategy Evolution](../../docs/methodology/strategy-evolution.md)):
- **H-main is MANDATORY**: Every iteration must have a main hypothesis with (1) quantitative prediction, (2) causal mechanism, (3) diagnostic clause
- Additional hypotheses (H-ablation, H-boundary, etc.) recommended based on iteration complexity
- Predictions must be quantitative and falsifiable (not "will improve" but "TTFT RMSE from X% to <Y%")
- Diagnostic clauses guide investigation when predictions fail

---

## Generalization Requirements

The basis functions must generalize across:

1. **Model Architecture**: Dense (Llama, Mistral, Qwen) vs MoE (Scout)
2. **Tensor Parallelism**: TP ∈ {1, 2, 4} with communication overhead scaling
3. **Workload Patterns**: codegen/reasoning/roleplay/general-lite (representative sample) → must generalize to unseen workloads
4. **Batch Compositions**: Variable prefill/decode ratios, context lengths, batch sizes
5. **Hardware Platforms**: Parameterized by `hardware_config.json` (H100 now, A100/L40S future)

**CRITICAL**: Workload type labels are training metadata only. Basis functions must predict latency from batch shape (token counts, context lengths) alone. Two batches with identical (tokens, contexts, model, hardware) must receive identical predictions regardless of workload label.

---

## Integration with BLIS

### Backend Name: "evolved"

**MUST** use backend name `"evolved"` consistently:
- CLI: `--latency-model evolved`
- Go registration: `Register("evolved", ...)`
- Manifest: `latency_backend_name: "evolved"`

**Rationale**: The evolved backend evolves its structure across iterations while maintaining a stable interface. Version history lives in git commits, not backend names.

**Important**: The `sim/latency/evolved_model.go` file is **mutable** — each iteration replaces the previous implementation. Agent 1 should read the current implementation before modifying it (if N > 0). Previous versions persist in git history.

### Coefficient Injection

**Alpha coefficients [α₀, α₁, α₂]** (request-level):
- α₀ = Fixed API processing overhead (μs per request)
- α₁ = Per-input-token overhead (μs/token) — tokenization
- α₂ = Per-output-token overhead (μs/token) — detokenization

**Beta coefficients [β₁, ..., βₙ]** (step-level):
- Each βᵢ multiplies basis function fᵢ(batch, model, hardware)
- Number of terms n evolved by outer loop
- Units depend on what fᵢ returns (see dimensional consistency in agent instructions)

### LatencyModel Interface

```go
type LatencyModel interface {
    StepTime(batch []*Request) int64  // YOU CUSTOMIZE THIS
    QueueingTime(req *Request) int64  // Fixed: α₀ + α₁ × input_len
    OutputTokenProcessingTime() int64 // Fixed: α₂
    PostDecodeFixedOverhead() int64   // Fixed: 0
}
```

**YOU ONLY MODIFY `StepTime()`**. Other methods use standard implementations.

---

## Success Criteria

### Primary Metrics

1. **Prediction accuracy**: MAPE < 10% on TTFT, ITL, E2E across all 15 experiments
2. **Cross-validation**: MAPE < 15-20% on held-out test sets (CV-1/CV-2/CV-3)
3. **Workload generalization**: Model generalizes to unseen workload patterns without retraining
4. **Physics interpretability**: Each basis function corresponds to known GPU operation

### Secondary Metrics

5. **Sample efficiency**: Converge in ≤ 5 outer loop iterations
6. **Optimization efficiency**: Inner loop converges in ≤ 50 trials
7. **Coefficient stability**: β values stay within physically plausible ranges

### Qualitative Goals

8. **Hypothesis discipline**: Every iteration has H-main with quantitative prediction, causal mechanism, and diagnostic clause (Strategy Evolution Phase 2)
9. **Agent reasoning**: Cite specific error patterns when adding/removing basis functions
10. **Prediction-outcome cycle**: Agent 3 validates predictions, uses diagnostic clauses when they fail (Strategy Evolution Phase 4-5)
11. **Discovered physics**: Rediscover known effects (attention O(n²), TP communication log₂(TP))
12. **Robustness**: Graceful degradation on out-of-distribution inputs

---

## Termination Criteria

Training stops when **all** of the following conditions are met:

1. **Prediction accuracy**: Per-experiment APE < 10% on mean TTFT and mean E2E for all 15 experiments
2. **Residual structure**: Errors show white noise (no systematic patterns by model, TP, or workload)
3. **Physics completeness**: All known GPU/vLLM effects captured (compute, memory, communication, framework overhead)
4. **Convergence**: New basis functions don't reduce `loss.overall_loss` by >5%
5. **Cross-validation**: All CV tests pass (CV1/CV2/CV3 MAPE < 15-20%)

**Signal**: Agent 1 writes `CONVERGED` in `iteration_manifest.yaml` reasoning field.

**Typical trajectory**: Expect 3-5 outer loop iterations to converge.

---

## Deliverables

### Pre-implemented Components ✅

1. **Inner loop script** (`training/inner_loop_optimize.py`)
2. **Loss computation** (`training/run_blis_and_compute_loss.py`)
3. **Cross-validation** (`training/scripts/run_cv_tests.py`)
4. **Monitoring** (`training/scripts/monitor_optimization.py`)

### Agent Prompts ✅

5. **Agent 1: Design** ([`design-agent-prompt.md`](design-agent-prompt.md))
   - Hypothesis generation, Go implementation, bounds specification

6. **Agent 2: Orchestration** ([`orchestration-agent-prompt.md`](orchestration-agent-prompt.md))
   - Backend validation, optimization execution, monitoring

7. **Agent 3: Analysis** ([`analysis-agent-prompt.md`](analysis-agent-prompt.md))
   - Hypothesis validation, findings extraction, cross-validation
   
---

## References

### Internal Documentation

**Agent invocation** (start here):
- **Generic prompts for any iteration N**: [`agent-invocation-prompts.md`](agent-invocation-prompts.md) — Ready-to-use prompts for invoking all 3 agents

**Agent specifications** (detailed instructions):
- **Agent 1 (Design)**: [`design-agent-prompt.md`](design-agent-prompt.md) — Hypothesis generation, Go implementation, bounds specification
- **Agent 2 (Orchestration)**: [`orchestration-agent-prompt.md`](orchestration-agent-prompt.md) — Backend validation, optimization execution, monitoring
- **Agent 3 (Analysis)**: [`analysis-agent-prompt.md`](analysis-agent-prompt.md) — Hypothesis validation, findings extraction, cross-validation

**Supporting documentation**:
- **Iteration structure**: [`../iterations/README.md`](../iterations/README.md) — Directory layout and file descriptions
- **CV protocol**: [`generalization-validation-protocol.md`](generalization-validation-protocol.md) — Cross-validation test specifications
- **BLIS latency models**: `../../docs/guide/latency-models.md` — Existing latency backend documentation

### Reference Materials (`training/references/`)
- `vllm/` — vLLM architecture, source code, profiling
- `datasheets/` — GPU hardware specs (H100, A100, L40S)
- `InferSim/`, `LLMServingSim/` — Other inference simulators
- `llm-optimizer/`, `aiconfigurator/` — Performance optimization

### Methodology (REQUIRED READING)

**⚠️ All agents must follow the Strategy Evolution methodology:**

- **Strategy Evolution**: [`../../docs/methodology/strategy-evolution.md`](../../docs/methodology/strategy-evolution.md)
  - **Agent 1**: Hypothesis bundle design with mandatory H-main
  - **Agent 2**: Implementation and optimization
  - **Agent 3**: Prediction verification and principle extraction

- **Hypothesis Bundles**: [`../../docs/methodology/hypothesis-bundles.md`](../../docs/methodology/hypothesis-bundles.md)
  - Complete worked examples from PR #452 (scheduling) and PR #447 (routing)
  - Shows H-main structure, verdict criteria, diagnostic clause usage
  - Explains why prediction errors are the most valuable learning signal

---

## Quick Start

**See [`../QUICKSTART.md`](../QUICKSTART.md) for detailed step-by-step instructions.**

**TL;DR**: For iteration N:
1. **Agent 1**: Use prompt from [`agent-invocation-prompts.md`](agent-invocation-prompts.md) → generates 4 files
2. **Agent 2**: Run `python inner_loop_optimize.py --iteration N --n-trials 50` → generates results.json
3. **Agent 3**: Use prompt from [`agent-invocation-prompts.md`](agent-invocation-prompts.md) → generates 2-3 analysis documents
4. **Iterate**: If not converged, Agent 1 uses FINDINGS.md to design iter{N+1}

**Typical trajectory**: 3-5 iterations to convergence (~1-2 hours per iteration).
