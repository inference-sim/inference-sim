# Agentic Latency Model Training — Problem Statement

## Executive Summary

Build a two-loop agentic system that automatically discovers and fits physics-informed latency models for BLIS. The system must produce basis functions that generalize across model architectures (dense and MoE), tensor parallelism configurations, vLLM scheduling parameters, and workload patterns — achieving state-of-the-art prediction accuracy without manual feature engineering.

## Related Work & Methodology

**Architectural Discussion**: For detailed discussion of the two-loop architecture, coefficient injection mechanism, and integration with BLIS, see [GitHub Issue #4 Comment](https://github.com/inference-sim/training/issues/4#issuecomment-4056357828).

**Agentic Strategy Evolution**: This problem adapts the [Strategy Evolution](../../docs/methodology/strategy-evolution.md) methodology developed for discovering optimal BLIS configurations. Key principles adapted:
- **Hypothesis-driven iteration**: Each outer loop iteration formulates testable predictions (basis function effectiveness) before implementation
- **Physics-informed reasoning**: Agent proposes basis functions grounded in GPU architecture, vLLM internals, and transformer compute patterns
- **Principle extraction**: Prediction errors (when basis functions fail) reveal causal model corrections that constrain future iterations
- **Convergence criterion**: Stop when error patterns show white noise (no systematic structure remaining)

For Strategy Evolution details, see:
- [Strategy Evolution Overview](../../docs/methodology/strategy-evolution.md) — structured iterative search with hypothesis bundles
- [Hypothesis Bundles in Practice](../../docs/methodology/hypothesis-bundles.md) — worked examples with prediction-vs-outcome analysis

## Background

BLIS supports four latency backends, each with limitations:
- **Roofline**: Analytical (FLOPs/bandwidth bounds) but ignores overheads
- **Blackbox**: Empirical per-model coefficients, no cross-model/cross-TP generalization
- **Cross-model**: Linear regression with fixed basis functions, limited expressiveness
- **Trained-roofline**: Manual coefficient tuning, requires iterative feature engineering

**Opportunity**: Replace manual feature engineering with agentic reasoning that discovers optimal basis functions, while Bayesian optimization fits coefficients.

## Problem Definition

Given:
- **Training/validation data**: 15 ground-truth experiments in `training/trainval_data/`, covering:
  - **Models**: Llama-2-7B, Llama-3.1-70B, Mistral-Nemo-12B, Qwen2.5-7B, Yi-34B, Llama-4-Scout-17B-16E (MoE)
  - **Tensor parallelism**: TP ∈ {1, 2, 4}
  - **vLLM configurations**: `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`
  - **Workload types**: codegen, reasoning, roleplay, general-lite (representative sample — model must generalize to unseen workloads)
  - **Metrics**: Per-request TTFT and ITL (Inter-Token Latency) with complete batch shape traces
- **Evaluation infrastructure**: `run_blis_and_compute_loss.py` runs BLIS with specified latency model (e.g., `--latency-model evolved`) and outputs JSON with loss: `RMSE[APE(mean_TTFT_per_exp)] + RMSE[APE(mean_E2E_per_exp)]`
- **Physics constraints**: Basis functions must respect causality, dimensional analysis, and known GPU architecture properties
- **Workload-agnostic constraint**: Basis functions MUST NOT use workload type labels (codegen, reasoning, roleplay, general-lite are training metadata only). At inference time, the latency model only observes: batch composition (prefill/decode token counts, context lengths, batch size), model architecture features, and hardware specs. This ensures generalization to unseen workload distributions.

Produce:
- **Generalized basis functions** f₁(batch, model, hardware), ..., fₙ(batch, model, hardware) that work across all model types, TP configs, and workloads
  - **CRITICAL**: Basis functions can ONLY depend on observable batch characteristics (token counts, context lengths, batch size), model architecture (layers, dimensions, attention heads), and hardware specs (FLOPS, bandwidth, TP config)
  - **FORBIDDEN**: Workload type labels, model name strings, hardcoded GPU-specific constants, or any feature not available at inference time
- **Request-level alpha coefficients** [α₀, α₁, α₂] for per-request overheads:
  - α₀ = Fixed API processing overhead (µs per request)
  - α₁ = Per-input-token API processing overhead (µs/token) — tokenization, input validation
  - α₂ = Per-output-token post-decode overhead (µs/token) — detokenization, output formatting
- **vLLM step-level beta coefficients** [β₁, ..., βₙ]:
  - Each βᵢ multiplies a corresponding basis function fᵢ(batch, model, hardware)
  - The agent discovers which basis functions are needed and what they represent
  - Basis functions may be physics-informed (FLOPs, bandwidth, communication) or empirical (batch size, layer count, constant overhead)
  - The number of beta coefficients n is evolved by the outer loop — start with n=7 (trained_roofline baseline), agent may add/remove terms
- **Prediction function**:
  - `StepTime(batch) = Σᵢ βᵢ · fᵢ(batch, model, hardware)`
  - `QueueingTime(req) = α₀ + α₁ × num_input_tokens`
  - `OutputTokenProcessingTime() = α₂` (per output token)
- **Target accuracy**: When running `--evaluate-per-experiment`, MAPE < 10% on E2E, TTFT, and ITL (Inter-Token Latency) across all 15 experiments
- **Loss function**: `RMSE[APE(mean_TTFT_per_exp)] + RMSE[APE(mean_E2E_per_exp)]` where APE is computed per experiment, then RMSE across all experiments

## Terminology: Alpha vs Beta

**IMPORTANT**: In the evolved latency model, alpha and beta have specific meanings:

- **Alpha coefficients [α₀, α₁, α₂]**: Request-level overheads that apply to individual requests, independent of batch composition:
  - α₀ = Fixed API processing overhead (µs per request) — HTTP parsing, request validation, constant setup cost
  - α₁ = Per-input-token API processing overhead (µs/token) — tokenization, input validation that scales with prompt length
  - α₂ = Per-output-token post-decode overhead (µs/token) — detokenization, output formatting that scales with generation length

- **Beta coefficients [β₁, ..., βₙ]**: vLLM step-level coefficients that scale basis functions in `StepTime()`:
  - Each βᵢ multiplies a corresponding basis function fᵢ(batch, model, hardware)
  - The functional form of each fᵢ is discovered by the agent (physics-based, empirical, hybrid)
  - Example forms: compute time, memory bandwidth, batch size, layer count, constant overhead, interaction terms
  - The number of terms n is not fixed — the agent may propose adding or removing basis functions

The agentic training process evolves **both** alphas (by testing different request-level overhead formulas) and betas (by proposing new basis functions for step time).

## The Two-Loop Architecture

### Outer Loop: Agentic Strategy Evolution

**Responsibility**: Evolve the structure of the latency model by reasoning about physics and error patterns.

**Agent specification**: For the complete prompt specification for implementing the outer loop agent (including output format requirements, Go code templates, reasoning guidelines, and validation checklist), see **[outer-loop-specs.md](outer-loop-specs.md)**.

**Input** (iteration N):
- Minimum-loss (α, β) from previous inner loop (if N=0: no prior iteration, start from scratch with α=[0,0,0])
- Loss value and per-experiment APE (Absolute Percentage Error) from `run_blis_and_compute_loss.py --evaluate-per-experiment`: APE for TTFT, E2E, and ITL per experiment
- Current basis functions f₁, ..., fₙ (if N=0: no prior basis functions, propose novel structure)
- Access to `training/references/` folder and internet search for background research

**Agent reasoning process**:
1. **Background research** (iteration 0 or when proposing new basis function categories):
   - Review `training/references/` folder for documentation on vLLM internals, other simulators, GPU architecture
   - Search the internet for related work on inference performance modeling, roofline analysis, GPU profiling
   - Understand what operations occur during a vLLM step (attention kernels, FFN, all-reduce, KV cache access)
   - Study empirical findings from prior latency modeling research

2. **Error pattern analysis**: Examine systematic patterns in per-experiment APE
   - "TTFT APE correlates with input length → α₁ (per-input-token overhead) may need adjustment"
   - "TTFT shows consistent underprediction (negative bias) → α₀ (fixed API overhead) miscalibrated"
   - "E2E APE shows per-token bias → α₂ (output token processing) needs tuning"
   - "Step time consistently underpredicted at TP=4 → missing TP communication term in beta basis functions"
   - "Prefill-heavy batches show high APE → prefill basis function needs revision"
   - "MoE experiments show higher APE than dense → need MoE-specific basis function (expert routing overhead)"

2. **Physics-informed hypothesis generation**: Propose new/modified basis functions for StepTime

   **IMPORTANT**: The agent designs **only the `StepTime()` method**. Other LatencyModel methods use standard BLIS implementations:
   - `QueueingTime(req)` = `α₀ + α₁ × input_len` (DO NOT modify)
   - `OutputTokenProcessingTime()` = `α₂` (DO NOT modify; models per-token streaming detokenization)
   - `PostDecodeFixedOverhead()` = `0` (DO NOT modify unless systematic per-request bias observed)

   The agent reasons about what computational/memory/communication operations occur during a vLLM step and proposes basis functions that capture those costs. Each basis function returns a value (typically in microseconds or dimensionless), which gets multiplied by its corresponding beta coefficient.

   **Knowledge sources for hypothesis generation**:
   - `training/references/` folder: vLLM architecture docs, other simulator designs, GPU profiling studies
   - Internet search: Recent papers on inference performance modeling, transformer optimization, GPU memory hierarchies
   - BLIS codebase: Existing latency models as reference (but NOT as starting point)
   - Hardware configs: Note that `MfuPrefill`/`MfuDecode` in `hardware_config.json` are theoretical values, not empirical. Agent may propose treating them as learnable parameters or replacing them with beta coefficients.

   **Potential basis function categories** (agent explores and discovers what's needed):
   - **Compute-bound terms**: Operations limited by GPU FLOP/s throughput
   - **Memory-bound terms**: Operations limited by HBM bandwidth or cache hierarchy
   - **Communication terms**: TP collective operations (all-reduce, broadcast)
   - **Structural overheads**: Framework/scheduler costs (per-layer, per-request, per-step)
   - **Interaction terms**: Non-linear effects from batch composition or resource contention
   - **Bottleneck terms**: Modeling overlapping compute/memory operations

   **CRITICAL CONSTRAINT — Workload-agnostic features**:
   Basis functions MUST depend ONLY on features observable at inference time:
   - ✅ **Allowed**: Batch composition (num_prefill_tokens, num_decode_tokens per request), context lengths, batch size, model architecture (layers, dimensions, attention heads, MoE parameters), hardware specs (FLOPS, bandwidth, TP config)
   - ❌ **FORBIDDEN**: Workload type labels (codegen/reasoning/roleplay/general-lite), model name strings, hardcoded GPU-specific constants, any metadata not available at inference time

   Workload labels in training data are metadata for cross-validation only. If a basis function would behave differently on two batches with identical (tokens, context_lengths, model, hardware) but different workload labels, it violates the generalization requirement.

   The agent proposes specific functional forms based on residual analysis and domain knowledge, ensuring all terms use hardware parameters from `hardware_config.json` rather than hardcoded constants.

3. **Dimensional consistency check**: Ensure `Σᵢ βᵢ · fᵢ` produces time (µs)
   - If fᵢ returns time (µs), then βᵢ is dimensionless (scaling factor)
   - If fᵢ returns a count (layers, requests, etc.), then βᵢ has units µs/count
   - If fᵢ returns dimensionless (ratios, normalized values), then βᵢ has units µs
   - Agent must verify dimensional consistency for each proposed basis function

4. **Search bound specification**: Propose reasonable ranges for coefficients based on physical/empirical constraints

   The agent should specify search bounds for each coefficient based on:
   - **Physical constraints**: Hardware specs (e.g., β can't predict faster than hardware peak throughput)
   - **Empirical bounds**: Observed APE magnitudes in training data
   - **Dimensional analysis**: Units of the coefficient constrain plausible ranges
   - **Prior knowledge**: Framework overhead characteristics from similar systems

   **MANDATORY constraint**: All bounds must have `lower_bound >= 0.0` (no negative coefficients allowed).

   **Alpha bounds considerations**:
   - α₀ (fixed overhead): Based on API framework characteristics (HTTP parsing, request validation)
     - Typical range: [0, 1ms] = [0, 0.001]
     - Suggested initial: 0.0002 (~200μs, typical vLLM API overhead)
   - α₁ (per-input-token): Based on tokenizer performance benchmarks
     - Typical range: [0, 100μs/token] = [0, 0.0001]
     - Suggested initial: 0.000001 (~1μs/token tokenization)
   - α₂ (per-output-token): Based on detokenizer and output formatting costs in streaming mode
     - Typical range: [0, 100μs/token] = [0, 0.0001]
     - Suggested initial: 0.000002 (~2μs/token detokenization)

   **Beta bounds considerations**:
   - If βᵢ scales a time estimate (dimensionless): Allow deviation from analytical model, but not orders of magnitude
     - Example: Roofline predicts 1.0, allow [0.3, 3.0] to capture overhead/inefficiency
     - Suggested initial: 1.0 (start at theoretical prediction)
   - If βᵢ scales a count (µs per unit): Based on typical GPU/scheduler overhead ranges
     - Example: Per-layer overhead [0, 50μs] = [0, 0.00005]
     - Suggested initial: midpoint or physically motivated value
   - If βᵢ is a constant term (µs): Based on observed maximum step overhead in profiling data
     - Example: Fixed scheduler overhead [0, 1ms] = [0, 0.001]

   **Initial value recommendations** (optional but recommended):
   The agent should provide `alpha_initial` and `beta_initial` arrays in `coefficient_bounds.yaml` to warm-start Bayesian optimization with physically plausible starting points. This accelerates convergence compared to uniform sampling from bounds.

   The agent should justify each range based on the specific basis function and available evidence, not use universal defaults.

**Output**:
- Updated basis functions f₁', ..., fₘ' (may add/remove/modify terms)
- Initial α values [α₀, α₁, α₂] and search ranges (if iteration 0, start with α=[0, 0, 0])
- Initial β values [β₁, ..., βₙ] and search ranges for each basis function

**End-of-iteration recording**: After each outer loop iteration completes (i.e., AFTER inner loop has converged to optimal coefficients):
1. Run detailed evaluation: `run_blis_and_compute_loss.py` with `--evaluate-per-experiment` flag
2. Record in training ledger (append to `training/evolution_ledger.jsonl` or similar):
   - Iteration number
   - Basis functions (code/description)
   - Final coefficients (α*, β*)
   - Overall loss
   - Per-experiment metrics (TTFT APE, E2E APE, ITL APE)
   - Agent reasoning (what was changed and why)
3. This ledger enables:
   - Progress tracking across iterations
   - Rollback if later iteration degrades performance
   - Post-training analysis of which basis functions contributed most

**Termination criterion**: Agent determines no structural improvements possible when:
- Residuals appear as white noise (no systematic patterns)
- All known physics effects are captured
- Additional terms don't reduce loss below noise floor
- Generalization performance (cross-validation MAPE) stops improving

### Inner Loop: Bayesian Optimization (Pre-implemented Script)

**Responsibility**: Find the best-fitting (α, β) coefficients for a fixed set of basis functions.

**Implementation**: The inner loop is a **pre-implemented Python script** (`training/inner_loop_optimize.py`). The outer loop agent does NOT need to implement optimization logic—it only needs to generate the three required input files.

**How to invoke**:
```bash
cd training/
python inner_loop_optimize.py --n-trials 50
```

The script automatically:
1. Reads `iteration_manifest.yaml` (outer loop output)
2. Verifies Go source files exist
3. Compiles BLIS binary
4. Loads `coefficient_bounds.yaml`
5. Runs Bayesian optimization (up to 50-100 trials)
   - **Early stopping**: Stops if best loss hasn't improved by >1% in last 50 trials
   - Prevents wasteful trials after convergence plateau
6. Calls `run_blis_and_compute_loss.py` for each trial **WITHOUT** `--evaluate-per-experiment` (optimization mode)
7. After convergence, runs ONE final evaluation **WITH** `--evaluate-per-experiment` (diagnostic mode)
8. Saves results to `inner_loop_results.json`

**Input files (outer loop must provide)**:
1. `iteration_manifest.yaml` - declares backend name and modified files
2. Go source file(s) - implements basis functions (path declared in manifest)
3. `coefficient_bounds.yaml` - search space for (α, β)

**Output file (inner loop produces)**:
- `inner_loop_results.json` containing:
  - `best_alpha`: Optimal alpha coefficients
  - `best_beta`: Optimal beta coefficients
  - `best_loss`: Final loss value
  - `detailed_diagnostics`: Per-experiment APE for error pattern analysis
  - `num_errors`: Count of failed trials
  - `error_log`: Details of any crashed trials

**Termination criterion**: Bayesian optimizer stops after 50-100 trials (configurable via `--n-trials`).

**Post-convergence evaluation**: The script automatically runs `run_blis_and_compute_loss.py --evaluate-per-experiment` after finding optimal coefficients to generate per-experiment diagnostics for the outer loop's next iteration.

## Generalization Requirements

The basis functions must generalize across five dimensions:

### 1. Model Architecture (Dense vs MoE)

**Dense models** (Llama, Mistral, Qwen):
- Standard transformer attention scaling with sequence length and model dimensions
- FFN compute where all layers use full feed-forward dimension

**MoE models** (Llama-4-Scout-17B-16E):
- Sparse FFN activation: Only top-k experts per token (e.g., k=2 out of N=16)
- Expert routing overhead: Gating network selects which experts to activate
- Weight loading patterns depend on expert selection strategy and batch size
- Potential load imbalance if expert selection skewed

**Requirement**: Basis functions should capture MoE-specific computational patterns (sparse expert activation, routing overhead, potential load imbalance) if residuals show systematic differences between dense and MoE models. Terms should activate only when `num_local_experts > 1`.

### 2. Tensor Parallelism (TP ∈ {1, 2, 4})

TP introduces:
- **All-reduce collectives**: After each attention and FFN layer across all layers
- **Communication cost**: Data transfer volume scales with model dimensions and inversely with available bandwidth
- **Synchronization overhead**: Logarithmic scaling with TP degree (tree-reduce pattern)
- **Diminishing returns**: Communication overhead grows with TP, offsetting compute speedup

**Requirement**: Basis functions should include TP-dependent terms if residuals show systematic error correlated with TP configuration. Such terms should vanish when TP=1 (no inter-GPU communication) and scale appropriately with TP degree.

### 3. vLLM Scheduling Parameters

**max_num_seqs** (batch size limit):
- Smaller batch → higher per-token overhead (kernel launch amortization)
- Larger batch → memory pressure, potential TLB misses, reduced cache efficiency

**max_num_batched_tokens** (total token budget):
- Affects batch formation decisions (can't pack beyond token limit)
- Interacts with KV cache memory constraints

**max_model_len** (context window):
- Determines KV cache block allocation
- Affects memory fragmentation

**Requirement**: Basis functions should capture how scheduling parameters affect latency. This may include linear effects (per-request overhead scales with batch size) or non-linear effects (saturation, memory pressure) if residuals show such patterns.

### 4. Workload Patterns

**Training data includes four example workload types** (representative sample, not exhaustive):
- **Codegen**: Long output sequences (high decode token ratio)
- **Reasoning**: Moderate input/output (balanced prefill/decode)
- **Roleplay**: Variable turn lengths (prefix sharing effects)
- **General-lite**: Short interactions (high per-request overhead)

**CRITICAL GENERALIZATION REQUIREMENT**: These four workloads are only a small subset of real-world usage patterns. The trained latency model will be evaluated on:
- **Unseen workloads**: Document QA, creative writing, translation, summarization, etc.
- **Unseen distributions**: Different input/output length distributions, arrival patterns, batch compositions
- **Production scenarios**: Mixed workloads in the same batch, extreme outliers (single-token decode, 10K-token prefill)

**Workload-agnostic constraint** (see "Given" section above): Workload type labels are training metadata only, NOT inference-time features. Basis functions must predict latency purely from batch shape (token counts, context lengths), not from semantic workload categories. Two batches with identical (tokens, context_lengths, model, hardware) must receive identical latency predictions regardless of workload label.

### 5. Hardware Platform

**Current data**: H100-SXM GPUs. **Future experiments**: A100-80GB, L40S, and potentially other accelerators.

**Requirement**: Basis functions must be parameterized by hardware specifications from `hardware_config.json`, not hardcoded for specific GPUs. This enables cross-hardware generalization without retraining the entire model structure — only beta coefficients may need recalibration for new hardware.

Hardware specs available from BLIS `hardware_config.json`:
- `TFlopsPeak`: Peak compute throughput (TFLOPS)
- `BwPeakTBs`: HBM memory bandwidth (TB/s)
- `MfuPrefill` / `MfuDecode`: **Theoretical MFU estimates** (NOT empirical measurements)
  - These values are initial guesses and may not reflect actual achievable utilization
  - Strategy evolution can propose treating MFU as a learnable parameter rather than a fixed constant
  - Agent may suggest replacing fixed MFU with basis function coefficients (e.g., β that scales FLOPs/bandwidth ratios)
- Additional fields: `MemoryGiB`, `TFlopsFP8` (for FP8-capable accelerators)

The agent should propose basis functions that use these hardware parameters rather than GPU-specific constants. If MFU values appear unrealistic based on residual analysis, the agent may propose evolving them as part of the latency model rather than treating them as ground truth.

## Integration with BLIS

### Existing Infrastructure

1. **run_blis_and_compute_loss.py**: Loss evaluation function for Bayesian optimization
   - **Input**: `--latency-model <backend_name>` specifies which latency backend to use
   - **Process**:
     - Loads all experiments from `trainval_data/`
     - Runs BLIS binary on each experiment with specified latency model
     - Compares predicted vs observed TTFT and E2E latency
     - Computes APE per experiment on mean TTFT and mean E2E
     - Takes RMSE across all experiments
   - **Output**: JSON to stdout with `{"overall_loss": <value>, "ttft_rmse": <value>, "e2e_rmse": <value>, ...}`
     - `overall_loss = RMSE[APE(mean_TTFT_per_exp)] + RMSE[APE(mean_E2E_per_exp)]`
     - With `--evaluate-per-experiment`: JSON includes `per_experiment` array with detailed metrics for each experiment

2. **Latency model plugin system** (`sim/latency/`):
   - Backend registration: `sim/latency/register.go`
   - Factory pattern: `NewLatencyModel(coeffs, modelConfig, hwConfig)`
   - Interface:
     - `StepTime(batch []*Request) int64` — vLLM step execution time
     - `QueueingTime(req *Request) int64` — ARRIVED → QUEUED overhead
     - `OutputTokenProcessingTime() int64` — per-token detokenization
     - `PostDecodeFixedOverhead() int64` — fixed post-decode overhead (E2E only)

3. **Coefficient structure** (`sim/config.go`): `LatencyCoeffs` struct with two fields:
   - `BetaCoeffs []float64` — vLLM step-level coefficients (≥3 elements)
   - `AlphaCoeffs []float64` — request-level coefficients (≥3 elements)

   **AlphaCoeffs = [α₀, α₁, α₂]**: Request-level overheads (independent of batch)
     - α₀ = Fixed API processing overhead (microseconds per request)
     - α₁ = Per-input-token API overhead (microseconds/token)
     - α₂ = Per-output-token post-decode overhead (microseconds/token)
   - **Request latency components**:
     - `QueueingTime(req) = α₀ + α₁ × num_input_tokens`
     - `OutputTokenProcessingTime() = α₂` (applied per output token)
   - **BetaCoeffs = [β₁, ..., βₙ]**: vLLM step-level coefficients (multiplied by basis functions)
     - Number of terms n is not fixed — evolved by the outer loop
     - Each βᵢ corresponds to a basis function fᵢ(batch, model, hardware)
     - Units depend on what fᵢ returns (see dimensional consistency check in agent reasoning)

4. **Model metadata** (`model_configs/`, HuggingFace auto-fetch):
   - Architecture params: `num_hidden_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`
   - MoE detection: `num_local_experts`, `num_experts_per_tok`
   - Quantization: `quantization_config.bits`, `torch_dtype`

### Required Modifications

1. **Evolve latency backend code generation**:
   - Agent outputs Go code implementing new `LatencyModel` interface
   - Primary method: `StepTime(batch []*Request) int64` computes basis functions and applies beta coefficients
   - Helper methods: `QueueingTime()`, `OutputTokenProcessingTime()`, `PostDecodeFixedOverhead()` apply alpha coefficients
   - Coefficients loaded from LatencyCoeffs struct (passed to factory constructor)

2. **Coefficient injection mechanism**:
   - Training driver writes coefficients into evolved latency backend:
     - Generate Go code for `sim/latency/evolved.go` (or modify existing file)
     - Embed alpha coefficients [α₀, α₁, α₂] in the model struct
     - Embed beta coefficients [β₁, ..., βₙ] in the model struct
     - Implement `LatencyModel` interface methods:
       - `QueueingTime(req) = α₀ + α₁ × num_input_tokens`
       - `OutputTokenProcessingTime() = α₂`
       - `StepTime(batch) = Σᵢ βᵢ · fᵢ(batch, model, hardware)`
   - Recompile BLIS with updated latency backend
   - `run_blis_and_compute_loss.py --latency-model evolved` uses the new backend
   - Script returns JSON with loss and diagnostic metrics

3. **Bayesian optimization integration**:
   - Python driver using Optuna, Ax, or GPyOpt
   - Objective function wraps `run_blis_and_compute_loss.py`:
     1. Write (α, β) coefficients to evolved latency backend (Go code or config)
     2. Run BLIS with `--latency-model evolved` (NO `--evaluate-per-experiment` flag — inner loop needs only overall_loss)
     3. Parse JSON output to extract `overall_loss = RMSE[APE(TTFT)] + RMSE[APE(E2E)]`
     4. Return loss value to Bayesian optimizer

## Success Criteria

### Primary Metrics

1. **Prediction accuracy**: When running evaluation with `--evaluate-per-experiment` flag, all three error metrics (E2E, TTFT, and ITL) must be below 10% MAPE across all 15 experiments
2. **Cross-model generalization**: Trained on 80% of experiments, test on held-out 20% with MAPE < 15% on TTFT/ITL
3. **Workload generalization**: Model trained on {codegen, reasoning, roleplay, general-lite} should generalize to unseen workload patterns (document QA, creative writing, translation, etc.) without retraining — basis functions must be workload-agnostic
4. **Physics interpretability**: Each basis function corresponds to a known GPU operation (compute, memory, communication) or observable batch characteristic (batch size, token distribution)

### Secondary Metrics

4. **Sample efficiency**: Outer loop converges in ≤ 5 iterations (5 agent-proposed model structures)
5. **Optimization efficiency**: Inner loop converges in ≤ 50 Bayesian optimization iterations per structure
6. **Coefficient stability**: β values stay within physically plausible ranges (e.g., 0.3 < MFU < 0.9)

### Qualitative Goals

7. **Agent reasoning quality**: Agent explanations for adding/removing basis functions should cite specific error patterns (APE correlations, systematic biases)
8. **Discovered physics**: System should discover known effects from first principles (attention quadratic scaling with sequence length, TP communication overhead scaling with log₂(TP), batch formation amortization)
9. **Robustness**: Predictions should degrade gracefully for out-of-distribution inputs:
   - Untested TP=8 configuration
   - Novel MoE architectures (different expert counts, routing strategies)
   - **Unseen workload patterns** (e.g., trained on codegen/reasoning, tested on translation/summarization)
   - Extreme batch compositions (all-prefill, all-decode, highly heterogeneous context lengths)

## Deliverables

### Pre-implemented Components (Already Exist)

1. **Inner loop script** (`training/inner_loop_optimize.py`): ✅ **Already implemented**
   - Reads outer loop outputs (manifest, bounds, Go code)
   - Compiles BLIS, runs Bayesian optimization
   - Saves results to `inner_loop_results.json`

2. **Loss computation** (`training/run_blis_and_compute_loss.py`): ✅ **Already implemented**
   - Runs BLIS on all experiments
   - Computes `RMSE[APE(TTFT)] + RMSE[APE(E2E)]`
   - Returns JSON with loss metrics

### To Be Implemented

3. **Outer loop driver** (`training/train_latency_model.py`):
   - Calls Claude API with error pattern analysis prompt (see **[outer-loop-specs.md](outer-loop-specs.md)** for the complete agent prompt specification)
   - Receives agent-generated: Go code, manifest, bounds
   - Invokes `python inner_loop_optimize.py`
   - Reads `inner_loop_results.json` for next iteration
   - Manages evolution ledger

4. **Generated BLIS latency backend** (`sim/latency/evolved_model.go`):
   - **Agent-generated** each iteration
   - Implements `LatencyModel` interface
   - Basis functions with agent's reasoning in comments
   - Registered with backend name from manifest

5. **Training report** (`training/docs/training_report.md`):
   - Generated after training completes
   - Evolution history: Basis functions per iteration
   - Loss trajectory across iterations
   - Residual analysis plots
   - Final coefficient values with interpretation

6. **Evolution ledger** (`training/evolution_ledger.jsonl`):
   - JSON Lines: One entry per iteration
   - Records: iteration, basis functions, coefficients, loss, diagnostics, reasoning
   - Enables rollback and post-training analysis

## Resolved Design Decisions

1. **Cold start**: Do NOT start from existing backends (trained_roofline, etc.) — propose novel structure from first principles
2. **Loss function**: Single combined loss `RMSE[APE(TTFT)] + RMSE[APE(E2E)]` from `run_blis_and_compute_loss.py`
3. **Regularization**: Delegated to agent — decides whether to penalize complexity based on residual analysis
4. **Hardware portability**: Deferred until after H100 baseline established
5. **KV cache dynamics**: BLIS handles GPU/CPU tiering — basis functions can leverage if needed
6. **Batch granularity**: Agent decides level of detail (aggregate vs per-request heterogeneity)
7. **ITL definition**: Use ground truth ITL from `trainval_data` as-is

## Generalization Validation

**Challenge**: With only 15 experiments, validate generalization without overfitting.

**Solution**: Two-tier validation (cross-validation + analytical checks)

**Current Focus: Tier 1 (Cross-Validation)**

Three holdout tests validate that basis functions generalize. For each test, **basis functions are frozen** (from main training), but **coefficients (α, β) are refit** on the holdout training set via inner loop optimization, then evaluated on the holdout test set:

- **CV-1 (LOMO)**: Train on 11 dense, test on 4 MoE → Dense→MoE architectural transfer (MAPE < 20%)
- **CV-2 (LOWO)**: Train on codegen+reasoning (7), test on roleplay+general (8) → Workload-agnostic constraint (MAPE < 15%, variance < 3%)
- **CV-3 (LOTO)**: Train on TP∈{1,4} (9), test on TP=2 (6) → TP communication interpolation (MAPE < 15%)

**Future: Tier 2 (Physics Checks)**: Fast analytical checks (AC-1 through AC-6) ensuring predictions respect causality, TP scaling, batch amortization, prefill/decode asymmetry, model monotonicity. Deferred until Tier 1 validation is working.

**Full details**: See [`generalization-validation-protocol.md`](generalization-validation-protocol.md) for complete pass/fail criteria and failure diagnosis.

## References

### Internal Documentation (BLIS codebase)
- BLIS latency model documentation: `docs/guide/latency-models.md`
- Existing backends (reference only, NOT starting point): `sim/latency/roofline.go`, `sim/latency/blackbox.go`, `sim/latency/cross_model.go`
- Training data format: `training/trainval_data/README.md`
- Model config extraction: `sim/latency/config.go`

### Reference Materials (`training/references/` folder)
**The agent should consult this folder for background knowledge when proposing basis functions:**

Available subdirectories:
- `vllm/` — vLLM architecture documentation, source code references, performance profiling
- `InferSim/` — InferSim (another inference simulator) design and implementation
- `LLMServingSim/` — LLMServingSim design papers and documentation
- `llm-optimizer/` — Performance optimization techniques and tools
- `aiconfigurator/` — AI workload configuration and analysis
- `datasheets/` — GPU hardware specifications (H100, A100, L40S memory hierarchies, compute specs)

These materials provide:
- How vLLM implements continuous batching, PagedAttention, KV cache management
- How other simulators model inference latency (comparison baseline)
- Detailed GPU architecture specs for accurate roofline analysis
- Empirical performance characterization from related work

### External Resources (Internet search enabled)
**The agent may search for:**
- Recent papers on LLM inference performance modeling (2023-2026)
- GPU architecture documentation (H100, A100, L40S memory hierarchies, compute units)
- Tensor parallelism communication patterns (NCCL, all-reduce algorithms)
- Continuous batching and PagedAttention implementations
- FlashAttention and other attention kernel optimizations
- MoE routing algorithms and load balancing strategies

### Methodology References
- Bayesian optimization: Optuna documentation, Gaussian Process Optimization papers
- Physics-informed ML: Karniadakis et al., "Physics-Informed Machine Learning" (2021)
- Symbolic regression: Brunton & Kutz, "Data-Driven Science and Engineering" (2019)
