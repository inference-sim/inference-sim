package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 6: Per-request scheduler overhead (batch formation + KV block allocation).
//
// Hypothesis: Iter5 CATASTROPHICALLY FAILED (loss 603% vs target <110%, TTFT 519% vs <55%).
// Post-analysis trace investigation revealed iter3/4/5 were based on WRONG ASSUMPTIONS:
// all assumed reasoning = long context (8K-16K tokens), but traces show reasoning uses ~1K tokens
// (same as all workloads). The "1000× underestimation" was actually "100-200× underestimation
// for SHORT contexts". Critical discovery: Reasoning (1K tokens, 100-200ms TTFT) differs from
// codegen (1K tokens, 20-50ms TTFT) NOT by context length, but by QUEUING/SCHEDULER DELAY.
// Trace variance (p10=0.13ms, p90=215ms = 1650×) indicates batching delay, not prefill compute.
//
// Iter6 moves scheduler overhead from StepTime (per-layer kernel overhead) to QueueingTime
// (per-request scheduler overhead):
//   - **Redefined** β₆: Now models vLLM scheduler overhead (batch formation + KV block allocation)
//     as fixed cost per request (50-150ms), not per-layer cost (500-3000μs × num_layers)
//   - **Moved** β₆ from StepTime to QueueingTime (scheduler overhead is per-request, not per-step)
//   - **Removed** per-layer overhead term entirely from StepTime (keep only β₀-β₅)
//   - **Expected**: Reasoning improves (99% → 40-45% TTFT) by capturing scheduler overhead,
//     short-context recovers (200-1091% → 4-40% TTFT) by removing wrong per-layer term
//
// Changes from iter5:
//   - **Removed** β₆ per-layer overhead from StepTime (caused catastrophic over-prediction for short contexts)
//   - **Added** β₆ scheduler overhead to QueueingTime (captures batch formation + KV allocation delay)
//   - **Reverted** Alpha to iter4 (iter5 Alpha exploded to absorb TTFT error)
//   - **Constrained** β₀ bounds to [0.10, 0.35] (prevent over-rise like iter5)
//   - **Expected**: Overall loss 603% → 90-110%, TTFT RMSE 519% → 40-50%, E2E RMSE 84% → 55-60%
//
// Basis functions:
//   - StepTime (6 beta terms: β₀-β₅): Prefill/decode compute, memory, communication, KV management, MoE gating
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (7 total):
//   - β₀: Prefill compute MFU scaling (dimensionless, expected 0.15-0.25)
//   - β₁: Decode memory-bound MFU (dimensionless, expected 1.00-1.10)
//   - β₂: TP decode communication scaling (dimensionless, expected 0.30-0.35)
//   - β₃: KV cache management overhead (μs per request, expected 400-500μs)
//   - β₄: Decode compute-bound MFU (dimensionless, expected 0.75-0.85)
//   - β₅: MoE gating overhead (μs, expected 10-12μs)
//   - β₆: **NEW** Scheduler overhead per request (ms, expected 50-150ms) - used in QueueingTime, NOT StepTime
//
// Alpha coefficients (request-level, reverted to iter4):
//   - α₀: Fixed API processing overhead (ms per request, 1.5ms)
//   - α₁: Per-input-token tokenization (ms per token, 125μs)
//   - α₂: Per-output-token detokenization (ms per token, 36μs)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆] - β₀-β₅ for StepTime, β₆ for QueueingTime
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 6 basis functions (6 terms in StepTime, β₆ moved to QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × kv_mgmt_time: KV cache management per request
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[5] × moe_gating_time: MoE gating network overhead
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - MoE gating: routing probability computation for all experts
//
// Expected coefficients (iteration 6):
//   - β₀ ≈ 0.15-0.25 (prefill MFU, should drop from iter5's 0.266 with scheduler overhead decoupled)
//   - β₁ ≈ 1.00-1.10 (decode memory-bound, continue recovery from iter5's 1.449 toward iter3's 1.037)
//   - β₂ ≈ 0.30-0.35 (TP communication, needs to drop from stuck 1.36-1.37)
//   - β₃ ≈ 400-500μs per request (KV block allocation, should recover from iter5's collapsed 0.013μs)
//   - β₄ ≈ 0.75-0.85 (decode compute-bound, recover from iter5's implausible 0.620 to iter3's 0.796)
//   - β₅ ≈ 10-12μs (MoE gating, close at iter5's 14.85μs, finish convergence to iter3's 11.66μs)
//   - β₆ ≈ 50-150ms (scheduler overhead per request, NEW in iter6, moved to QueueingTime)
//
// Beta[6] is NOT used in StepTime (moved to QueueingTime in iter6).
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
	// Expected range: 0.15-0.25 (should drop from iter5's 0.266 with scheduler overhead decoupled)
	// Units: seconds (converted to μs below)
	// Code: vllm/worker/model_runner.py:_prepare_model_input() computes prefill FLOPs
	var prefillComputeTimeSeconds float64
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)
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
	// Expected range: β₁ ≈ 1.00-1.10 (continue recovery from iter5's 1.449 toward iter3's 1.037),
	//                 β₄ ≈ 0.75-0.85 (recover from iter5's implausible 0.620 to iter3's 0.796)
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
	// Expected range: 0.30-0.35 (needs to drop from stuck 1.36-1.37 in iter4/iter5)
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
	// Expected range: 400-500μs per request (should recover from iter5's collapsed 0.013μs)
	// Units: microseconds per request
	// Code: vllm/core/block_manager.py:BlockSpaceManager.allocate() allocates blocks per request
	kvMgmtTimeUs := float64(len(batch)) // Number of requests in batch, in microseconds
	kvMgmtContribution := m.Beta[3] * kvMgmtTimeUs

	// ========================================
	// β₅ × moe_gating_time (MoE gating)
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 10-12μs (close at iter5's 14.85μs, finish convergence to iter3's 11.66μs)
	// Units: microseconds
	// Code: vllm/model_executor/layers/fused_moe/fused_moe.py computes gating logits
	//
	// Gating network: hidden_dim → num_experts linear projection per token
	// FLOPs: 2 × tokens × hidden_dim × num_experts
	var moeGatingTimeSeconds float64
	if m.modelConfig.NumLocalExperts > 1 {
		var totalPrefillTokens int64
		for _, req := range stepConfig.PrefillRequests {
			totalPrefillTokens += int64(req.NumNewPrefillTokens)
		}
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
	// Total step time (additive model)
	// ========================================
	// Iteration 6: 6 terms in StepTime (β₀-β₅), β₆ moved to QueueingTime
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtContribution + decodeComputeContribution +
		moeGatingContribution

	return max(1, clampToInt64(totalTimeUs))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// **MODIFIED IN ITER6**: Added Beta[6] scheduler overhead per request.
//
// Physics:
//   - α₀: Fixed API processing (HTTP parsing, request validation, queue insertion)
//   - α₁: Per-input-token tokenization (HuggingFace BPE encoding scales with input length)
//   - β₆: **NEW IN ITER6** Scheduler overhead per request (batch formation + KV block allocation)
//
// Scheduler overhead (β₆) captures:
//   1. Batch formation: vLLM scheduler computes can-schedule decision (capacity check, priority ordering)
//   2. KV block allocation: PagedAttention allocates physical blocks from block manager
//   3. Priority queue processing: Sorts waiting requests and selects batch
//
// Expected values:
//   - α₀: ~1-5ms (API gateway overhead)
//   - α₁: ~50-200μs per token (tokenization)
//   - β₆: ~50-150ms per request (scheduler overhead, varies by workload concurrency)
//
// Reasoning workloads (multi-turn chat, high concurrency) experience higher scheduler overhead
// (~100-150ms) than codegen workloads (single-turn, lower concurrency, ~10-30ms). This explains
// why reasoning (1K tokens) has 100-200ms TTFT while codegen (1K tokens) has 20-50ms TTFT.
func (m *EvolvedModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	// Alpha coefficients (in milliseconds, need conversion to microseconds)
	totalProcessingTime += m.Alpha[0] * 1000.0                                 // Fixed overhead (α₀ in ms, convert to μs)
	totalProcessingTime += m.Alpha[1] * float64(len(req.InputTokens)) * 1000.0 // Tokenization (α₁ in ms/token, convert to μs)

	// Beta[6] scheduler overhead (in milliseconds, need conversion to microseconds)
	// This is the NEW term in iter6: per-request scheduler overhead for batch formation + KV allocation
	totalProcessingTime += m.Beta[6] * 1000.0 // Scheduler overhead (β₆ in ms, convert to μs)

	return clampToInt64(totalProcessingTime)
}

// OutputTokenProcessingTime returns per-output-token post-processing overhead.
// **DO NOT MODIFY THIS METHOD.** Standard implementation: α₂ (streaming detokenization)
//
// Physics:
//   - α₂: Per-output-token detokenization + output formatting in streaming mode
//   - Applied per output token during decode phase
func (m *EvolvedModel) OutputTokenProcessingTime() int64 {
	// α₂ is in milliseconds, convert to microseconds
	return clampToInt64(m.Alpha[2] * 1000.0)
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// **DO NOT MODIFY THIS METHOD.** Return 0 unless systematic per-request bias observed.
//
// Physics:
//   - Models constant overhead at request completion (e.g., response finalization, logging)
//   - Iteration 0/1/2/3/4/5/6: No systematic bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0 // No systematic per-request bias in current training data
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
//
// This function is meant to be integrated into the main NewLatencyModel switch statement
// in latency.go. The factory pattern follows the existing backends (roofline, blackbox, etc.).
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 6: 3 alpha, 7 beta)
	// Beta count unchanged from iter5 (still 7), but β₆ moved from StepTime to QueueingTime
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 7 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 7 elements for iteration 6, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 6 expects 7: β₀-β₆)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
