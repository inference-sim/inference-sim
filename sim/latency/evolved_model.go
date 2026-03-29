package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 4: Activation memory bandwidth + continued simplification.
//
// Hypothesis: Iter3 showed β₀ = 0.169 far below physical MFU (0.40-0.55), and β₇ (TP prefill
// comm) was rejected by optimizer (coefficient ≈ 0), eliminating communication as the missing
// overhead. Iter4 addresses this by: (1) removing ineffective β₂/β₇ (continuing iter3's
// successful simplification pattern), (2) adding activation memory bandwidth term to capture
// prefill bottleneck and allow β₀ to rise toward physical plausibility.
//
// Changes from iter3:
//   - **Removed** β₂ (scheduler overhead) - coefficient 9.97e-05 ≈ 0, negligible
//   - **Removed** β₇ (TP prefill communication) - coefficient 2.78e-07 ≈ 0, rejected by optimizer
//   - **Added** β₆ (NEW: activation write bandwidth) - captures HBM writes during prefill
//   - **Renumbered** remaining terms: β₃→β₂, β₄→β₃, β₅→β₄, β₆→β₅
//
// Basis functions (StepTime) - 7 terms:
//   - β₀ × prefill_compute_time: Prefill FLOPs / (peak_TFLOPS × MFU_prefill)
//   - β₁ × decode_memory_time × memory_weight: Decode small-batch memory-bound (with interpolation)
//   - β₂ × tp_comm_time: TP communication overhead (all-reduce per layer, DECODE ONLY)
//   - β₃ × kv_mgmt_time: KV cache management overhead per request
//   - β₄ × decode_compute_time × compute_weight: Decode large-batch compute-bound (with interpolation)
//   - β₅ × moe_gating_time: MoE gating network overhead per expert per token
//   - β₆ × activation_bandwidth_time: Activation write bandwidth during prefill (NEW in iter4)
//
// Alpha coefficients (request-level, unchanged):
//   - α₀: Fixed API processing overhead (microseconds per request)
//   - α₁: Per-input-token tokenization (microseconds per token)
//   - α₂: Per-output-token detokenization (microseconds per token)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆] - step-level coefficients (7 terms)
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 4 basis functions (7 terms):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × kv_mgmt_time: KV cache management per request
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[5] × moe_gating_time: MoE gating network overhead
//   - beta[6] × activation_bandwidth_time: Activation write bandwidth during prefill (NEW)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - MoE gating: routing probability computation for all experts
//   - Activation bandwidth: HBM writes for residual connections, attention outputs, layer norms
//
// Expected coefficients (iteration 4):
//   - β₀ ≈ 0.25-0.35 (prefill MFU, should rise from iter3's 0.169 with activation term added)
//   - β₁ ≈ 1.00-1.10 (decode memory-bound, stable at 1.037 from iter3)
//   - β₂ ≈ 0.318 (TP communication scaling for decode, stable from iter3)
//   - β₃ ≈ 0.37μs per request (KV block allocation, stable from iter3)
//   - β₄ ≈ 0.60-0.70 (decode compute-bound, may decrease from 0.796 if was absorbing activation overhead)
//   - β₅ ≈ 0.008-0.010 (MoE gating, may decrease from 0.0117 if was absorbing activation overhead)
//   - β₆ ≈ 3.0-6.0 (NEW: activation bandwidth multiplier, captures HBM write overhead)
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
	// Expected range: 0.25-0.35 (should rise from iter3's 0.169 with activation bandwidth term added)
	// Units: seconds (converted to μs below)
	// Code: vllm/worker/model_runner.py:_prepare_model_input() computes prefill FLOPs
	var prefillComputeTimeSeconds float64
	var totalPrefillTokens int64
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)
		totalPrefillTokens += numTokens
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, numTokens, true, true)
		// Divide by (peak_TFLOPS × MFU_prefill) to get time in seconds
		prefillComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuPrefill)
	}
	prefillTimeUs := prefillComputeTimeSeconds * 1e6 // Convert to microseconds
	prefillContribution := m.Beta[0] * prefillTimeUs

	// ========================================
	// β₁ × decode_memory_time × memory_weight (small-batch, with sigmoid interpolation)
	// β₄ × decode_compute_time × compute_weight (large-batch, with sigmoid interpolation)
	// ========================================
	// Physics: Decode transitions from memory-bound (small batches) to compute-bound (large batches)
	// Expected range: β₁ ≈ 1.00-1.10 (stable at 1.037 from iter3), β₄ ≈ 0.60-0.70 (may decrease if was absorbing activation overhead)
	// Units: seconds (converted to μs below)
	// Code: vllm/attention/backends/flashinfer.py reads KV cache per decode token
	//
	// Sigmoid interpolation from iter2: Replace discrete if-else with continuous sigmoid
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
	decodeComputeContribution := m.Beta[4] * decodeComputeTimeUs * computeWeight

	// ========================================
	// β₂ × tp_comm_time (TP communication for DECODE)
	// ========================================
	// Physics: Ring all-reduce after each transformer layer (TP > 1), DECODE ONLY
	// Expected range: 0.318 (stable from iter3)
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
	tpCommContribution := m.Beta[2] * tpCommTimeUs

	// ========================================
	// β₃ × kv_mgmt_time (KV cache management)
	// ========================================
	// Physics: vLLM PagedAttention block allocation/deallocation per request
	// Expected range: 0.37μs per request (stable from iter3)
	// Units: seconds per request
	// Code: vllm/core/block_manager.py:BlockSpaceManager.allocate() allocates blocks per request
	kvMgmtTimeSeconds := float64(len(batch)) // Number of requests in batch
	kvMgmtTimeUs := kvMgmtTimeSeconds * 1e6  // Convert to microseconds
	kvMgmtContribution := m.Beta[3] * kvMgmtTimeUs

	// ========================================
	// β₅ × moe_gating_time (MoE gating)
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 0.008-0.010 (may decrease from 0.0117 if was absorbing activation overhead)
	// Units: seconds (converted to μs below)
	// Code: vllm/model_executor/layers/fused_moe/fused_moe.py computes gating logits
	//
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
	moeGatingContribution := m.Beta[5] * moeGatingTimeUs

	// ========================================
	// β₆ × activation_bandwidth_time (Activation write bandwidth) - NEW in iter4
	// ========================================
	// Physics: During prefill, each transformer layer writes large activation tensors to HBM:
	//   - Residual connections (full hidden_dim vectors after each sublayer)
	//   - Attention outputs (Q, K, V projections before attention)
	//   - Layer norm outputs (normalized activations before/after sublayers)
	//   - MLP intermediate activations (expanded dimensions, 4× or 8× hidden_dim)
	//
	// For long prompts (4K-16K tokens), these writes become bandwidth-limited and compete
	// with KV cache writes. This overhead is NOT captured by β₀ (prefill compute MFU).
	//
	// Expected range: β₆ ~ 3.0-6.0 (multiplier on theoretical write time)
	// Units: seconds (converted to μs below)
	// Code: vllm/worker/model_runner.py allocates activation buffers,
	//       vllm/model_executor/layers/attention.py writes attention outputs to HBM
	//
	// Formula: activation_bytes = num_prefill_tokens × hidden_dim × bytes_per_param × num_layers × k
	// where k ≈ 4-6 accounts for:
	//   - Residual connections (1×)
	//   - Attention QKV projections (3×)
	//   - Layer norms (1-2×)
	//   - Competing KV cache writes (overhead factor)
	var activationBandwidthTimeSeconds float64
	if totalPrefillTokens > 0 {
		// Activation writes per layer: tokens × hidden_dim × bytes_per_param × k_factor
		// k_factor = 4 accounts for residual (1×) + QKV (3×)
		kFactor := 4.0
		activationBytesPerLayer := float64(totalPrefillTokens) * float64(m.modelConfig.HiddenDim) * bytesPerParam * kFactor
		totalActivationBytes := activationBytesPerLayer * float64(m.modelConfig.NumLayers)
		// HBM bandwidth (bytes per second)
		hbmBandwidthBytesPerSec := m.hwConfig.BwPeakTBs * 1e12 // Convert TB/s to bytes/s
		// Theoretical write time (seconds)
		activationBandwidthTimeSeconds = totalActivationBytes / hbmBandwidthBytesPerSec
	}
	activationBandwidthTimeUs := activationBandwidthTimeSeconds * 1e6 // Convert to microseconds
	activationBandwidthContribution := m.Beta[6] * activationBandwidthTimeUs

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 4: 7 terms total (removed β₂ scheduler, β₇ TP prefill comm; added β₆ activation bandwidth)
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtContribution + decodeComputeContribution +
		moeGatingContribution + activationBandwidthContribution

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
//   - Iteration 0/1/2/3: No systematic bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0 // No systematic per-request bias in current training data
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
//
// This function is meant to be integrated into the main NewLatencyModel switch statement
// in latency.go. The factory pattern follows the existing backends (roofline, blackbox, etc.).
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 4: 3 alpha, 7 beta)
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 7 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 7 elements for iteration 4, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 4 expects 7)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
