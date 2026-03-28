package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 2: Very long context overhead + per-request decode overhead + smooth regime transition.
//
// Hypothesis: Very long contexts (>4096 tokens) have additional prefill overhead beyond standard
// FLOPs accounting, and decode has per-request setup overhead not captured by memory/compute terms.
//
// Changes from iter1:
//   - **Removed** β₅ (chunking overhead) - ablation showed +1.06% loss (redundant)
//   - **Added** β₇ (very long context prefill overhead) - captures reasoning experiments
//   - **Added** β₈ (per-request decode overhead) - normalizes inflated β₁
//   - **Modified** decode regime split to continuous sigmoid interpolation
//
// Basis functions (StepTime) - 9 terms:
//   - β₀ × prefill_compute_time: Prefill FLOPs / (peak_TFLOPS × MFU_prefill)
//   - β₁ × decode_memory_time × memory_weight: Decode small-batch memory-bound (with interpolation)
//   - β₂ × constant: Fixed vLLM scheduler overhead per step (microseconds)
//   - β₃ × tp_comm_time: TP communication overhead (all-reduce per layer)
//   - β₄ × kv_mgmt_time: KV cache management overhead per request
//   - β₅ × decode_compute_time × compute_weight: Decode large-batch compute-bound (with interpolation)
//   - β₆ × moe_gating_time: MoE gating network overhead per expert per token
//   - β₇ × long_context_overhead: Very long context prefill overhead (>4096 tokens)
//   - β₈ × per_request_overhead: Per-request decode overhead (scheduler + attention state)
//
// Alpha coefficients (request-level, unchanged):
//   - α₀: Fixed API processing overhead (microseconds per request)
//   - α₁: Per-input-token tokenization (microseconds per token)
//   - α₂: Per-output-token detokenization (microseconds per token)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇, β₈] - step-level coefficients (9 terms)
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 2 basis functions (9 terms):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × constant: Fixed scheduler overhead (microseconds)
//   - beta[3] × tp_comm_time: TP communication overhead (all-reduce)
//   - beta[4] × kv_mgmt_time: KV cache management per request
//   - beta[5] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[6] × moe_gating_time: MoE gating network overhead
//   - beta[7] × long_context_overhead: Very long context prefill overhead (>4096 tokens)
//   - beta[8] × per_request_overhead: Per-request decode overhead
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication: all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - MoE gating: routing probability computation for all experts
//   - Very long context: >4096 tokens have additional overhead (attention memory bandwidth saturation,
//     KV recomputation, reduced prefix cache effectiveness)
//   - Per-request decode: scheduler iteration, attention state setup, kernel launch overhead per request
//
// Expected coefficients (iteration 2):
//   - β₀ ≈ 0.4-0.5 (prefill efficiency, should rise from iter1's 0.203)
//   - β₁ ≈ 0.6-0.9 (decode memory-bound, should drop from iter1's 1.553)
//   - β₂ ≈ 0.12μs (vLLM scheduler overhead per step, stable from iter1)
//   - β₃ ≈ 0.394 (TP communication scaling, stable from iter1)
//   - β₄ ≈ 0.37μs per request (KV block allocation, stable from iter1)
//   - β₅ ≈ 0.6-0.8 (decode compute-bound for large batches, formerly β₆)
//   - β₆ ≈ 0.008 (MoE gating overhead, formerly β₇)
//   - β₇ ≈ 0.5-2.0 (very long context overhead scaling, NEW)
//   - β₈ ≈ 10-50μs per request (per-request decode overhead, NEW)
//
// Alpha coefficients are NOT used in StepTime (only in QueueingTime and OutputTokenProcessingTime).
func (m *EvolvedModel) StepTime(batch []*sim.Request) int64 {
	if len(batch) == 0 {
		return 0
	}

	// Build StepConfig from batch (same pattern as RooflineLatencyModel)
	stepConfig := StepConfig{
		PrefillRequests: make([]PrefillRequestConfig, 0, len(batch)),
		DecodeRequests:  make([]DecodeRequestConfig, 0, len(batch)),
	}

	for _, req := range batch {
		if req.ProgressIndex < int64(len(req.InputTokens)) {
			// Prefill phase
			stepConfig.PrefillRequests = append(stepConfig.PrefillRequests, PrefillRequestConfig{
				ProgressIndex:       req.ProgressIndex,
				NumNewPrefillTokens: req.NumNewTokens,
			})
		} else if len(req.OutputTokens) > 0 {
			// Decode phase
			stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, DecodeRequestConfig{
				ProgressIndex:      req.ProgressIndex,
				NumNewDecodeTokens: req.NumNewTokens,
			})
		}
	}

	// Select compute throughput based on weight precision and hardware capability
	// (same logic as roofline.go:236-241)
	peakFlops := m.hwConfig.TFlopsPeak * 1e12
	if m.modelConfig.EffectiveWeightBytesPerParam() == 1.0 && m.hwConfig.TFlopsFP8 > 0 {
		peakFlops = m.hwConfig.TFlopsFP8 * 1e12
	}

	tpFactor := float64(m.tp)
	bytesPerParam := m.modelConfig.EffectiveWeightBytesPerParam()

	// ========================================
	// β₀ × prefill_compute_time
	// ========================================
	// Physics: Prefill is compute-bound (large GEMMs, O(n²) attention)
	// Expected range: 0.4-0.5 (should rise from iter1's 0.203 after β₇ absorbs long-context overhead)
	// Units: seconds (converted to μs below)
	// Code: vllm/worker/model_runner.py:_prepare_model_input() computes prefill FLOPs
	var prefillComputeTimeSeconds float64
	var totalPrefillTokens int64
	var maxPrefillPromptTokens int64 // Track longest prompt for very long context term
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)
		totalPrefillTokens += numTokens
		// Track max prompt length (ProgressIndex + NumNewPrefillTokens is total context at this step)
		promptLength := req.ProgressIndex + numTokens
		if promptLength > maxPrefillPromptTokens {
			maxPrefillPromptTokens = promptLength
		}
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, numTokens, true, true)
		// Divide by (peak_TFLOPS × MFU_prefill) to get time in seconds
		prefillComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuPrefill)
	}
	prefillTimeUs := prefillComputeTimeSeconds * 1e6 // Convert to microseconds
	prefillContribution := m.Beta[0] * prefillTimeUs

	// ========================================
	// β₁ × decode_memory_time × memory_weight (small-batch, with sigmoid interpolation)
	// β₅ × decode_compute_time × compute_weight (large-batch, with sigmoid interpolation)
	// ========================================
	// Physics: Decode transitions from memory-bound (small batches) to compute-bound (large batches)
	// Expected range: β₁ ≈ 0.6-0.9 (should drop from iter1's 1.553), β₅ ≈ 0.6-0.8 (stable from iter1's β₆=0.651)
	// Units: seconds (converted to μs below)
	// Code: vllm/attention/backends/flashinfer.py reads KV cache per decode token
	//
	// Iteration 2 change: Replace discrete if-else with continuous sigmoid interpolation
	// memory_weight(n) = 1 / (1 + exp((n - 8) / 2))  [centered at batch_size=8, slope=2]
	// compute_weight(n) = 1 - memory_weight(n)
	var decodeMemoryTimeSeconds float64
	var decodeComputeTimeSeconds float64
	batchSize := len(stepConfig.DecodeRequests)
	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, 1, true, true)
		decodeMemoryTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuDecode)
		// Large-batch compute-bound time uses slightly higher MFU (better tensor core utilization)
		largeBatchMFU := m.hwConfig.MfuDecode * 1.2
		if largeBatchMFU > 1.0 {
			largeBatchMFU = 1.0
		}
		decodeComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * largeBatchMFU)
	}

	// Sigmoid interpolation weights (smooth transition from memory-bound to compute-bound)
	memoryWeight := 1.0 / (1.0 + math.Exp((float64(batchSize)-8.0)/2.0))
	computeWeight := 1.0 - memoryWeight

	decodeMemoryTimeUs := decodeMemoryTimeSeconds * 1e6
	decodeComputeTimeUs := decodeComputeTimeSeconds * 1e6
	decodeMemoryContribution := m.Beta[1] * decodeMemoryTimeUs * memoryWeight
	decodeComputeContribution := m.Beta[5] * decodeComputeTimeUs * computeWeight // Note: β₅ is now compute-bound (formerly β₆)

	// ========================================
	// β₂ × constant (scheduler overhead)
	// ========================================
	// Physics: Fixed vLLM scheduler overhead (batch formation, KV block allocation, kernel launch)
	// Expected range: 0.12μs based on iter1 convergence
	// Units: microseconds
	// Code: vllm/core/scheduler.py:schedule() has per-step overhead
	schedulerOverhead := m.Beta[2]

	// ========================================
	// β₃ × tp_comm_time (TP communication)
	// ========================================
	// Physics: Ring all-reduce after each transformer layer (TP > 1)
	// Expected range: 0.394 (stable from iter1)
	// Units: seconds (converted to μs below)
	// Code: vllm/model_executor/layers/linear.py:ColumnParallelLinear.forward() calls all_reduce
	var tpCommTimeSeconds float64
	if m.tp > 1 {
		// All-reduce bytes per layer: 2 × hidden_dim × bytes_per_param (forward + backward activations)
		allReduceBytesPerLayer := 2.0 * float64(m.modelConfig.HiddenDim) * bytesPerParam
		// Ring all-reduce: (2 × (TP-1) / TP) × bytes / bandwidth
		// Simplified to (TP-1) / TP ≈ 1 for TP > 1
		tpCommFactor := float64(m.tp-1) / float64(m.tp)
		nvlinkBandwidthBytesPerSec := m.hwConfig.BwPeakTBs * 1e12 // Convert TB/s to bytes/s
		tpCommTimeSeconds = tpCommFactor * float64(m.modelConfig.NumLayers) * allReduceBytesPerLayer / nvlinkBandwidthBytesPerSec
	}
	tpCommTimeUs := tpCommTimeSeconds * 1e6 // Convert to microseconds
	tpCommContribution := m.Beta[3] * tpCommTimeUs

	// ========================================
	// β₄ × kv_mgmt_time (KV cache management)
	// ========================================
	// Physics: vLLM PagedAttention block allocation/deallocation per request
	// Expected range: 0.37μs per request (stable from iter1, CRITICAL per ablation)
	// Units: seconds per request
	// Code: vllm/core/block_manager.py:BlockSpaceManager.allocate() allocates blocks per request
	kvMgmtTimeSeconds := float64(len(batch)) // Number of requests in batch
	kvMgmtTimeUs := kvMgmtTimeSeconds * 1e6  // Convert to microseconds
	kvMgmtContribution := m.Beta[4] * kvMgmtTimeUs

	// ========================================
	// β₆ × moe_gating_time (MoE gating)
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 0.008 (stable from iter1's β₇)
	// Units: seconds (converted to μs below)
	// Code: vllm/model_executor/layers/fused_moe/fused_moe.py computes gating logits
	// Note: Index is now β₆ (formerly β₇ in iter1) due to removing β₅ chunking
	//
	// BUG FIX (2026-03-28): Previous calculation multiplied num_experts × tokens (dimensionless count)
	// by 1e6 and treated it as time. Correct calculation: compute gating FLOPs and convert to time.
	// Gating network: hidden_dim → num_experts linear projection per token
	// FLOPs: 2 × tokens × hidden_dim × num_experts
	var moeGatingTimeSeconds float64
	if m.modelConfig.NumLocalExperts > 1 {
		totalTokens := totalPrefillTokens + int64(len(stepConfig.DecodeRequests))
		// Gating FLOPs: 2 (multiply-add) × tokens × hidden_dim × num_experts
		gatingFlops := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)
		// Gating runs on tensor cores with lower efficiency than main compute (small GEMM)
		// Use conservative 30% MFU for gating network
		gatingEfficiency := 0.3
		moeGatingTimeSeconds = gatingFlops / tpFactor / (peakFlops * gatingEfficiency)
	}
	moeGatingTimeUs := moeGatingTimeSeconds * 1e6 // Convert to microseconds
	moeGatingContribution := m.Beta[6] * moeGatingTimeUs // Note: β₆ is now MoE gating (formerly β₇)

	// ========================================
	// β₇ × long_context_overhead (very long context prefill overhead) - NEW in iter2
	// ========================================
	// Physics: Prompts >4096 tokens have additional prefill overhead beyond standard FLOPs accounting:
	//   1. Attention memory bandwidth saturation (intermediate matrices spill to HBM)
	//   2. KV cache recomputation under memory pressure (vLLM preemption)
	//   3. Reduced prefix cache effectiveness (unique long prompts)
	// Expected range: 0.5-2.0 (allows 50-200% prefill overhead for very long contexts)
	// Units: dimensionless scaling factor
	// Code: vllm/attention/backends/flashinfer.py:prefill_with_paged_kv() shows memory bandwidth saturation
	// Observable proxy: max(0, prompt_tokens - 4096) / 1000 × num_layers
	// This naturally captures reasoning experiments (longest prompts) without workload labels
	var longContextOverhead float64
	if maxPrefillPromptTokens > 4096 {
		// Only activate for very long prompts (>4096 tokens)
		excessTokens := float64(maxPrefillPromptTokens - 4096)
		// Scale by num_layers (overhead compounds across layers) and normalize by 1000
		longContextOverhead = (excessTokens / 1000.0) * float64(m.modelConfig.NumLayers)
	}
	longContextContribution := m.Beta[7] * longContextOverhead

	// ========================================
	// β₈ × per_request_overhead (per-request decode overhead) - NEW in iter2
	// ========================================
	// Physics: Each active request in decode batch incurs fixed overhead:
	//   1. Scheduler per-request work (priority check, sequence state update) - ~5-20μs per request
	//   2. Attention state setup (query offsets, KV block tables, sequence lengths) - ~10-50μs per request
	//   3. Kernel launch overhead for small batches - ~10-30μs per request
	// Expected range: 0.00001-0.00005 seconds (10-50μs per request)
	// Units: seconds per request
	// Code: vllm/core/scheduler.py:_schedule_running() iterates all active requests
	//       vllm/attention/backends/flashinfer.py:begin_forward() prepares per-request KV block tables
	// This overhead should normalize β₁ from inflated 1.553 to physically plausible 0.6-0.9
	numDecodeRequests := float64(len(stepConfig.DecodeRequests))
	perRequestTimeSeconds := numDecodeRequests // Number of active decode requests
	perRequestTimeUs := perRequestTimeSeconds * 1e6 // Convert to microseconds
	perRequestContribution := m.Beta[8] * perRequestTimeUs

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 2: 9 terms total (removed β₅ chunking, added β₇ long context, β₈ per-request)
	totalTimeUs := prefillContribution + decodeMemoryContribution + schedulerOverhead +
		tpCommContribution + kvMgmtContribution + decodeComputeContribution +
		moeGatingContribution + longContextContribution + perRequestContribution

	return max(1, clampToInt64(totalTimeUs))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// **DO NOT MODIFY THIS METHOD.** Standard implementation: α₀ + α₁ × input_len
//
// Physics:
//   - α₀: Fixed API processing (HTTP parsing, request validation, queue insertion)
//   - α₁: Per-input-token tokenization (HuggingFace BPE encoding scales with input length)
func (m *EvolvedModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.Alpha[0]                                 // Fixed overhead (μs)
	totalProcessingTime += m.Alpha[1] * float64(len(req.InputTokens)) // Tokenization (μs/token)
	return clampToInt64(totalProcessingTime)
}

// OutputTokenProcessingTime returns per-output-token post-processing overhead.
// **DO NOT MODIFY THIS METHOD.** Standard implementation: α₂ (streaming detokenization)
//
// Physics:
//   - α₂: Per-output-token detokenization + output formatting in streaming mode
//   - Applied per output token during decode phase
func (m *EvolvedModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.Alpha[2])
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// **DO NOT MODIFY THIS METHOD.** Return 0 unless systematic per-request bias observed.
//
// Physics:
//   - Models constant overhead at request completion (e.g., response finalization, logging)
//   - Iteration 0/1/2: No systematic bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0 // No systematic per-request bias in current training data
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
//
// This function is meant to be integrated into the main NewLatencyModel switch statement
// in latency.go. The factory pattern follows the existing backends (roofline, blackbox, etc.).
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 2: 3 alpha, 9 beta)
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 9 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 9 elements for iteration 2, got %d", len(coeffs.BetaCoeffs))
	}

	// Validate hardware config (same checks as roofline)
	if hw.TP <= 0 {
		return nil, fmt.Errorf("evolved model: TP must be > 0, got %d", hw.TP)
	}
	if err := ValidateRooflineConfig(hw.ModelConfig, hw.HWConfig); err != nil {
		return nil, fmt.Errorf("evolved model: %w", err)
	}

	// Validate coefficients (no NaN, Inf, or negative values)
	if err := validateCoeffs("AlphaCoeffs", coeffs.AlphaCoeffs); err != nil {
		return nil, err
	}
	if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
		return nil, err
	}

	return &EvolvedModel{
		Alpha:       [3]float64{coeffs.AlphaCoeffs[0], coeffs.AlphaCoeffs[1], coeffs.AlphaCoeffs[2]},
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 2 expects 9)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
