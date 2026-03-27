package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 1: Additive overhead terms + prefill chunking.
//
// Hypothesis: Real vLLM execution has additive overhead terms beyond max(compute, memory).
// Model structure: prefill + decode + tp_comm + kv_mgmt + chunking + scheduler
//
// Basis functions (StepTime):
//   - β₀ × prefill_compute_time: Prefill FLOPs / (peak_TFLOPS × MFU_prefill)
//   - β₁ × decode_memory_time: Decode small-batch memory-bound regime
//   - β₂ × constant: Fixed vLLM scheduler overhead per step (microseconds)
//   - β₃ × tp_comm_time: TP communication overhead (all-reduce per layer)
//   - β₄ × kv_mgmt_time: KV cache management overhead per request
//   - β₅ × chunking_time: Prefill chunking overhead per chunk boundary
//   - β₆ × decode_compute_time: Decode large-batch compute-bound regime
//   - β₇ × moe_gating_time: MoE gating network overhead per expert per token
//
// Alpha coefficients (request-level, unchanged from iter0):
//   - α₀: Fixed API processing overhead (microseconds per request)
//   - α₁: Per-input-token tokenization (microseconds per token)
//   - α₂: Per-output-token detokenization (microseconds per token)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇] - step-level coefficients
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 1 basis functions (8 terms):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time: Decode small-batch memory-bound regime
//   - beta[2] × constant: Fixed scheduler overhead (microseconds)
//   - beta[3] × tp_comm_time: TP communication overhead (all-reduce)
//   - beta[4] × kv_mgmt_time: KV cache management per request
//   - beta[5] × chunking_time: Prefill chunking overhead per chunk
//   - beta[6] × decode_compute_time: Decode large-batch compute-bound regime
//   - beta[7] × moe_gating_time: MoE gating network overhead
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - TP communication: all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - Chunking: vLLM splits prefills > 2048 tokens, introducing per-chunk overhead
//   - MoE gating: routing probability computation for all experts
//
// Expected coefficients (iteration 1):
//   - β₀ ≈ 0.5-0.6 (prefill efficiency, should rise from iter0's 0.308)
//   - β₁ ≈ 0.5-0.7 (decode memory-bound, should drop from iter0's 1.548)
//   - β₂ ≈ 50-200μs (vLLM scheduler overhead per step)
//   - β₃ ≈ 0.8-1.2 (TP communication scaling, near-linear with TP)
//   - β₄ ≈ 10-100μs per request (KV block allocation)
//   - β₅ ≈ 50-200μs per chunk (kernel launch + KV write)
//   - β₆ ≈ 0.6-0.8 (decode compute-bound for large batches)
//   - β₇ ≈ 0.1-1.0μs per expert per token (gating network)
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
	// Expected range: 0.5-0.6 (should rise from iter0's 0.308 after new terms absorb overhead)
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
	// β₁ × decode_memory_time (small-batch)
	// ========================================
	// Physics: Decode is memory-bound (KV cache reads, O(n) attention) for small batches
	// Expected range: 0.5-0.7 (should drop from iter0's 1.548 after β₆ captures large-batch compute)
	// Units: seconds (converted to μs below)
	// Code: vllm/attention/backends/flashinfer.py reads KV cache per decode token
	var decodeMemoryTimeSeconds float64
	batchSize := len(stepConfig.DecodeRequests)
	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, 1, true, true)
		decodeMemoryTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuDecode)
	}
	decodeMemoryTimeUs := decodeMemoryTimeSeconds * 1e6 // Convert to microseconds
	decodeMemoryContribution := m.Beta[1] * decodeMemoryTimeUs

	// ========================================
	// β₂ × constant (scheduler overhead)
	// ========================================
	// Physics: Fixed vLLM scheduler overhead (batch formation, KV block allocation, kernel launch)
	// Expected range: 50-200μs based on vLLM profiling
	// Units: microseconds
	// Code: vllm/core/scheduler.py:schedule() has per-step overhead
	schedulerOverhead := m.Beta[2]

	// ========================================
	// β₃ × tp_comm_time (TP communication)
	// ========================================
	// Physics: Ring all-reduce after each transformer layer (TP > 1)
	// Expected range: 0.8-1.2 (near-linear scaling with TP)
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
	// Expected range: 10-100μs per request (0.00001-0.0001 seconds)
	// Units: seconds per request
	// Code: vllm/core/block_manager.py:BlockSpaceManager.allocate() allocates blocks per request
	kvMgmtTimeSeconds := float64(len(batch)) // Number of requests in batch
	kvMgmtTimeUs := kvMgmtTimeSeconds * 1e6  // Convert to microseconds
	kvMgmtContribution := m.Beta[4] * kvMgmtTimeUs

	// ========================================
	// β₅ × chunking_time (prefill chunking)
	// ========================================
	// Physics: vLLM splits long prefills into 2048-token chunks, introducing per-chunk overhead
	// Expected range: 50-200μs per chunk (0.00005-0.0002 seconds)
	// Units: seconds per chunk
	// Code: vllm/worker/model_runner.py:_prepare_model_input() chunks prefill when num_tokens > max_tokens_per_chunk
	const maxTokensPerChunk = 2048.0
	numChunks := math.Ceil(float64(totalPrefillTokens) / maxTokensPerChunk)
	if numChunks < 1 {
		numChunks = 1 // At least one chunk even if no prefill tokens
	}
	chunkingTimeUs := numChunks * 1e6 // Convert to microseconds (β₅ is in seconds/chunk)
	chunkingContribution := m.Beta[5] * chunkingTimeUs

	// ========================================
	// β₆ × decode_compute_time (large-batch)
	// ========================================
	// Physics: Decode becomes compute-bound for large batches (≥ 8 requests) due to tensor core utilization
	// Expected range: 0.6-0.8 (higher MFU than β₁ memory-bound regime)
	// Units: seconds (converted to μs below)
	// Code: For large batches, decode attention GEMMs are large enough for compute dominance
	var decodeComputeTimeSeconds float64
	if batchSize >= 8 {
		// Large-batch: compute-bound regime
		for _, req := range stepConfig.DecodeRequests {
			f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, 1, true, true)
			// Use higher MFU for large-batch decode (assuming better tensor core utilization)
			largeBatchMFU := m.hwConfig.MfuDecode * 1.2 // Assume 20% better utilization (placeholder, will be learned)
			if largeBatchMFU > 1.0 {
				largeBatchMFU = 1.0 // Cap at 100% MFU
			}
			decodeComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * largeBatchMFU)
		}
	}
	decodeComputeTimeUs := decodeComputeTimeSeconds * 1e6 // Convert to microseconds
	decodeComputeContribution := m.Beta[6] * decodeComputeTimeUs

	// ========================================
	// β₇ × moe_gating_time (MoE gating)
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 0.1-1.0μs per expert per token (0.0000001-0.000001 seconds)
	// Units: seconds
	// Code: vllm/model_executor/layers/fused_moe/fused_moe.py computes gating logits
	var moeGatingTimeSeconds float64
	if m.modelConfig.NumLocalExperts > 1 {
		// Gating network overhead: num_experts × batch_tokens
		totalTokens := totalPrefillTokens + int64(len(stepConfig.DecodeRequests))
		moeGatingTimeSeconds = float64(m.modelConfig.NumLocalExperts) * float64(totalTokens)
	}
	moeGatingTimeUs := moeGatingTimeSeconds * 1e6 // Convert to microseconds
	moeGatingContribution := m.Beta[7] * moeGatingTimeUs

	// ========================================
	// Total step time (additive model)
	// ========================================
	totalTimeUs := prefillContribution + decodeMemoryContribution + schedulerOverhead +
		tpCommContribution + kvMgmtContribution + chunkingContribution +
		decodeComputeContribution + moeGatingContribution

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
//   - Iteration 0/1: No systematic bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0 // No systematic per-request bias in current training data
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
//
// This function is meant to be integrated into the main NewLatencyModel switch statement
// in latency.go. The factory pattern follows the existing backends (roofline, blackbox, etc.).
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 8 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 8 elements for iteration 1, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 1 expects 8)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
