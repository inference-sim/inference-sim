# Agentic Latency Model Training — Problem Statement

## Executive Summary

Build a two-loop agentic system that automatically discovers and fits physics-informed latency models for BLIS. The system must produce basis functions that generalize across model architectures (dense and MoE), tensor parallelism configurations, vLLM scheduling parameters, and workload patterns — achieving state-of-the-art prediction accuracy without manual feature engineering.

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

**Responsibility**: Evolve the structure of the latency model by reasoning about physics and residual patterns.

**Input** (iteration N):
- Minimum-loss (α, β) from previous inner loop (if N=0: no prior iteration, start from scratch with α=[0,0,0])
- Loss value and residuals: `observed_latency - predicted_latency` per experiment
- Current basis functions f₁, ..., fₙ (if N=0: no prior basis functions, propose novel structure)
- Access to `training/references/` folder and internet search for background research

**Agent reasoning process**:
1. **Background research** (iteration 0 or when proposing new basis function categories):
   - Review `training/references/` folder for documentation on vLLM internals, other simulators, GPU architecture
   - Search the internet for related work on inference performance modeling, roofline analysis, GPU profiling
   - Understand what operations occur during a vLLM step (attention kernels, FFN, all-reduce, KV cache access)
   - Study empirical findings from prior latency modeling research

2. **Residual analysis**: Examine systematic error patterns
   - "TTFT residuals correlate with input length → α₁ (per-input-token overhead) may need adjustment"
   - "Queueing time shows constant bias → α₀ (fixed API overhead) miscalibrated"
   - "Output token processing shows per-token bias → α₂ needs tuning"
   - "Step time underprediction at TP=4 → missing TP communication term in beta basis functions"
   - "Prefill-heavy batches show bias → prefill basis function needs revision"
   - "MoE models deviate from dense → need MoE-specific basis function (expert routing overhead)"

2. **Physics-informed hypothesis generation**: Propose new/modified basis functions for StepTime

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
   - **Empirical bounds**: Observed residual magnitudes in training data
   - **Dimensional analysis**: Units of the coefficient constrain plausible ranges
   - **Prior knowledge**: Framework overhead characteristics from similar systems

   **Alpha bounds considerations**:
   - α₀ (fixed overhead): Based on API framework characteristics (HTTP parsing, request validation)
   - α₁ (per-input-token): Based on tokenizer performance benchmarks
   - α₂ (per-output-token): Based on detokenizer and output formatting costs

   **Beta bounds considerations**:
   - If βᵢ scales a time estimate (dimensionless): Allow deviation from analytical model, but not orders of magnitude
   - If βᵢ scales a count (µs per unit): Based on typical GPU/scheduler overhead ranges
   - If βᵢ is a constant term (µs): Based on observed maximum step overhead in profiling data

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
   - Per-experiment metrics (TTFT MAPE, E2E MAPE, residuals)
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

### Inner Loop: Bayesian Optimization

**Responsibility**: Find the best-fitting (α, β) coefficients for a fixed set of basis functions.

**CRITICAL**: The inner loop MUST call `run_blis_and_compute_loss.py` **WITHOUT** the `--evaluate-per-experiment` flag. This flag should ONLY be used after inner loop convergence for residual analysis. During optimization, only `overall_loss` is needed - per-experiment breakdown would slow down the Bayesian optimization loop unnecessarily.

**Input**:
- Basis functions f₁, ..., fₙ from outer loop (fixed code in evolved latency backend)
- Search ranges for α [α₀, α₁, α₂] and β [β₁, ..., βₙ]
- Loss function: `L(α, β) = run_blis_and_compute_loss(α, β)` which computes:
  - `RMSE[APE(mean_TTFT_per_exp)] + RMSE[APE(mean_E2E_per_exp)]`
  - APE computed per experiment on experiment's mean TTFT and E2E
  - RMSE taken across all experiments, then summed

**Optimization process**:
1. **Surrogate model**: Build Gaussian process over the loss surface L(α, β)
2. **Acquisition function**: Balance exploration (high uncertainty) vs exploitation (low predicted loss)
3. **Candidate selection**: Propose next (α, β) point to evaluate
4. **Loss evaluation**:
   - Write current (α, β) coefficients into evolved latency backend code/config
   - Call `run_blis_and_compute_loss.py --latency-model evolved` (NO `--evaluate-per-experiment` flag)
   - Script runs BLIS on all experiments in `trainval_data/` using the evolved backend
   - Computes per-experiment APE on mean TTFT and mean E2E, then RMSE across experiments
   - Returns JSON output with: `{"overall_loss": RMSE[APE(TTFT)] + RMSE[APE(E2E)], ...}` (minimal output for speed)
5. **Surrogate update**: Incorporate new loss observation into Gaussian process
6. **Repeat** until convergence

**Termination criterion**: `Δloss < ε` (e.g., improvement < 0.1% MAPE per iteration)

**Output**: Minimum-loss (α*, β*) and final loss value

**Post-convergence evaluation**: ONLY after inner loop converges to optimal (α*, β*), run a single detailed evaluation with `--evaluate-per-experiment` flag to generate diagnostics (residuals, latency breakdown, throughput) for outer loop residual analysis. This step happens ONCE per outer loop iteration, not during the optimization loop.

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

7. **Agent reasoning quality**: Agent explanations for adding/removing basis functions should cite specific residual patterns
8. **Discovered physics**: System should discover known effects from first principles (attention quadratic scaling with sequence length, TP communication overhead scaling with log₂(TP), batch formation amortization)
9. **Robustness**: Predictions should degrade gracefully for out-of-distribution inputs:
   - Untested TP=8 configuration
   - Novel MoE architectures (different expert counts, routing strategies)
   - **Unseen workload patterns** (e.g., trained on codegen/reasoning, tested on translation/summarization)
   - Extreme batch compositions (all-prefill, all-decode, highly heterogeneous context lengths)

## Deliverables

1. **Python training driver** (`training/train_latency_model.py`):
   - Outer loop: Calls Claude API with residual analysis prompt
   - Inner loop: Bayesian optimization over coefficient space using `run_blis_and_compute_loss.py` as objective function
   - Ledger management: After each outer loop iteration, runs `--evaluate-per-experiment` and records results
   - Outputs: Best (α, β) per iteration, final latency backend code, evolution ledger

2. **Generated BLIS latency backend** (`sim/latency/evolved_model.go`):
   - Implements `LatencyBackend` interface
   - Basis functions as documented helper methods
   - Coefficient values in embedded struct or external config

3. **Training report** (`training/docs/training_report.md`):
   - Evolution history: Basis functions proposed at each iteration
   - Loss trajectory: Overall loss reduction across outer loop iterations
   - Per-iteration metrics: Detailed breakdown from `--evaluate-per-experiment` runs
   - Residual analysis: Plots showing error patterns before/after each evolution
   - Final coefficient values with physical interpretation

3a. **Evolution ledger** (`training/evolution_ledger.jsonl`):
   - JSON Lines format: One entry per outer loop iteration
   - Contains: iteration number, basis functions, coefficients, loss, per-experiment metrics, agent reasoning
   - Enables progress tracking, rollback, and post-training analysis

4. **Validation results** (`training/validation_results/`):
   - Per-experiment error breakdown (TTFT, ITL, per-stage latencies)
   - Comparison against existing backends (roofline, blackbox, cross-model, trained-roofline)
   - Ablation study: Contribution of each basis function to final accuracy

## Resolved Design Decisions

1. **Cold start**: Do NOT start from existing backends (trained_roofline, etc.) — propose novel structure from first principles
2. **Loss function**: Single combined loss `RMSE[APE(TTFT)] + RMSE[APE(E2E)]` from `run_blis_and_compute_loss.py`
3. **Regularization**: Delegated to agent — decides whether to penalize complexity based on residual analysis
4. **Hardware portability**: Deferred until after H100 baseline established
5. **KV cache dynamics**: BLIS handles GPU/CPU tiering — basis functions can leverage if needed
6. **Batch granularity**: Agent decides level of detail (aggregate vs per-request heterogeneity)
7. **ITL definition**: Use ground truth ITL from `trainval_data` as-is

## Generalization Validation

**Challenge**: With only 15 experiments, we must validate that the trained model generalizes to unseen models, workloads, and configurations without overfitting.

**Full details**: See [`generalization-validation-protocol.md`](generalization-validation-protocol.md) for complete validation protocol with pass/fail criteria, failure diagnosis, and validation schedule.

## Next Steps

1. **Milestone 1**: Implement inner loop BO driver that can optimize coefficients for a fixed (manually specified) set of basis functions
2. **Milestone 2**: Build agent prompt that analyzes residuals and proposes new basis functions (manually review and implement)
3. **Milestone 3**: Automate code generation — agent outputs Go code, system compiles and validates
4. **Milestone 4**: Close the outer loop — full automated training from cold start to converged latency backend
5. **Milestone 5**: Validation on held-out test set and comparison against existing BLIS backends

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
