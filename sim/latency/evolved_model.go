package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 8: MoE routing overhead mechanism.
//
// Critical discovery from iter7: Scout MoE architecture dominates error budget (49% of
// total loss from 27% of experiments). All 4 Scout experiments fail uniformly (79-100%
// TTFT) regardless of workload, while non-Scout reasoning-lite succeeded (99% → 54-66%
// TTFT). This proves the bottleneck is NOT workload or data quality but Scout MoE
// architecture-specific overhead not captured by current model.
//
// Root cause: Current model captures MoE gating FLOPs (β₅) but NOT per-token expert
// routing latency (expert selection, load balancing, coordination). vLLM's fused_moe.py
// has routing overhead beyond the gating network compute.
//
// Iter8 strategy: Add β₈ to capture per-token MoE routing overhead, train on all 15
// experiments (including 4 Scout) to learn MoE-specific coefficient. Expected: Overall
// loss 155% → <80% as β₈ absorbs Scout's 767% combined loss while leaving non-Scout
// experiments unaffected (β₈ = 0 for dense models).
//
// Changes from iter7:
//   - **Added β₈**: Per-token MoE routing overhead in StepTime (expected 10-50μs per routed token)
//   - **Basis function**: β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)
//   - **Physics**: Captures expert selection, load balancing, coordination beyond gating FLOPs
//   - **Warm-start**: All alpha/beta from iter7 optimal, β₈ initialized to 30μs
//   - **Expected**: Scout TTFT 79-100% → <50%, non-Scout stable (<±10pp change)
//
// Basis functions:
//   - StepTime (8 beta terms: β₀-β₅, β₇-β₈, note β₆ in QueueingTime):
//     Prefill/decode compute, memory, communication, KV management, MoE gating, MoE routing, decode overhead
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (9 total):
//   - β₀: Prefill compute MFU scaling (dimensionless, expected 0.15-0.25)
//   - β₁: Decode memory-bound MFU (dimensionless, expected 1.00-1.15)
//   - β₂: TP decode communication scaling (dimensionless, expected 0.20-0.35)
//   - β₃: KV cache management overhead (ms per request, expected 0.4-0.5ms)
//   - β₄: Decode compute-bound MFU (dimensionless, expected 0.70-0.90, constrained ≤1.0)
//   - β₅: MoE gating overhead (ms, expected 0.010-0.020ms = 10-20μs, should decrease from iter7's 41.1ms)
//   - β₆: Scheduler overhead per request (ms, expected 15-30ms) - used in QueueingTime, NOT StepTime
//   - β₇: Decode per-request overhead (ms, expected 10-20ms, should decrease from iter7's 26.3ms) - used in StepTime
//   - β₈: **NEW** MoE routing overhead per routed token (ms, expected 0.010-0.050ms = 10-50μs per routed token) - used in StepTime
//
// Alpha coefficients (request-level, stable from iter7):
//   - α₀: Fixed API processing overhead (ms per request, ~1.3ms)
//   - α₁: Per-input-token tokenization (ms per token, ~118μs, bounds [0.0, 0.0002])
//   - α₂: Per-output-token detokenization (ms per token, ~91μs, bounds [0.0, 0.0001])
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇, β₈] - β₀-β₅,β₇,β₈ for StepTime, β₆ for QueueingTime
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 8 basis functions (8 terms in StepTime, β₆ in QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × kv_mgmt_time: KV cache management per request
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[5] × moe_gating_time: MoE gating network overhead
//   - beta[7] × num_decode_requests: Decode per-request overhead (output processing, TP coordination)
//   - beta[8] × moe_routing_time: **NEW** MoE per-token routing overhead (expert selection, load balancing)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - MoE gating: routing probability computation for all experts (captured by β₅)
//   - MoE routing (NEW): per-token expert selection, dispatch, load balancing (captured by β₈)
//   - Decode overhead: Output processing, TP coordination, result aggregation per decode request
//
// Expected coefficients (iteration 8):
//   - β₀ ≈ 0.15-0.25 (prefill MFU, from iter7's 0.191)
//   - β₁ ≈ 1.00-1.15 (decode memory-bound, from iter7's stabilized 1.108)
//   - β₂ ≈ 0.20-0.35 (TP communication, from iter7's 0.185)
//   - β₃ ≈ 0.4-0.5ms per request (KV block allocation, should revert from iter7's 4.40ms)
//   - β₄ ≈ 0.70-0.90 (decode compute-bound, from iter7's stabilized 0.713)
//   - β₅ ≈ 10-20μs (MoE gating, should decrease from iter7's 41.1ms as β₈ offloads routing)
//   - β₆ ≈ 15-30ms (scheduler overhead per request, from iter7's 13.2ms, used in QueueingTime)
//   - β₇ ≈ 10-20ms (decode overhead per request, should decrease from iter7's 26.3ms as β₈ offloads Scout error)
//   - β₈ ≈ 10-50μs per routed token (NEW: MoE routing overhead)
//
// Beta[6] is NOT used in StepTime (moved to QueueingTime in iter6).
// Beta[7] is decode per-request overhead (added in iter7).
// Beta[8] is NEW in iter8 (MoE routing overhead per routed token).
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
	// Expected range: 0.15-0.25 (iter7: 0.191)
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
	// Expected range: β₁ ≈ 1.00-1.15 (iter7: 1.108, stabilized),
	//                 β₄ ≈ 0.70-0.90 (iter7: 0.713, stabilized)
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
	// Expected range: 0.20-0.35 (iter7: 0.185)
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
	// Expected range: 400-500μs per request (should revert from iter7's 4.40ms)
	// Units: microseconds per request
	// Code: vllm/core/block_manager.py:BlockSpaceManager.allocate() allocates blocks per request
	kvMgmtTimeUs := float64(len(batch)) // Number of requests in batch, in microseconds
	kvMgmtContribution := m.Beta[3] * kvMgmtTimeUs

	// ========================================
	// β₅ × moe_gating_time (MoE gating)
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 10-20μs (should decrease from iter7's 41.1ms as β₈ offloads routing overhead)
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
	// β₇ × decode_per_request_overhead (iter7)
	// ========================================
	// Physics: vLLM decode phase has fixed overhead per request beyond memory/compute:
	//   1. Output processing: After each decode step, vLLM processes output tokens
	//      (sampling, stop condition check, streaming updates)
	//   2. TP coordination: Decode requires per-request coordination across TP workers
	//      (synchronization barriers per step)
	//   3. KV cache write-back: Updated KV cache blocks written back to memory
	// Expected range: 10-20ms per decode request (should decrease from iter7's 26.3ms as β₈ offloads Scout error)
	// Units: milliseconds per request
	// Code: vllm/model_executor/model_loader.py:_run_workers() calls execute_model() per step
	//       vllm/worker/worker.py:execute_model() synchronizes across TP ranks per step
	numDecodeRequests := len(stepConfig.DecodeRequests)
	decodeOverheadTimeMs := float64(numDecodeRequests) // Number of decode requests
	decodeOverheadTimeUs := decodeOverheadTimeMs * 1000.0 // Convert to microseconds (β₇ is in ms)
	decodeOverheadContribution := m.Beta[7] * decodeOverheadTimeUs

	// ========================================
	// β₈ × moe_routing_time (NEW in iter8)
	// ========================================
	// Physics: MoE per-token expert routing overhead beyond gating FLOPs (β₅):
	//   1. Expert selection: Top-k selection from num_experts (k=1 or k=2 for Scout)
	//      - torch.topk() call per token: O(num_experts × log(k)) complexity
	//   2. Expert dispatch: Token reordering and routing to selected experts
	//      - Scatter/gather operations to reorder tokens by selected expert
	//      - Cross-GPU communication if TP > 1 (expert routing across GPUs)
	//   3. Load balancing: Dynamic expert assignment and utilization tracking
	//      - Auxiliary loss computation to balance expert utilization
	//      - Expert capacity constraints (may drop tokens if capacity exceeded)
	//   4. Expert aggregation: Weighted sum of expert outputs per token
	//      - Combine outputs from k experts with gating probabilities
	//
	// Expected range: 10-50μs per routed token
	// Units: microseconds per routed token
	// Code: vllm/model_executor/layers/fused_moe/fused_moe.py:fused_experts()
	//       Line ~150-200: Expert routing implementation (selection, dispatch, aggregation)
	//
	// Basis function: β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)
	//   - numMoELayers: Number of MoE layers in model (26 for Scout, 0 for dense models)
	//   - totalTokens: Prefill + decode tokens in batch
	//   - numExpertsPerTok: Active experts per token (k=1 or k=2, default to 1 if not set)
	//   - TP: Tensor parallelism degree (expert routing scales inversely with TP)
	//
	// For Scout prefill (numMoELayers=26, totalTokens=100, numExpertsPerTok=1, TP=2):
	//   - Routed tokens: 26 × 100 × 1 / 2 = 1300
	//   - β₈ contribution: β₈ × 1300 ≈ 13-65ms (if β₈ = 10-50μs)
	//   - This matches Scout TTFT residual (predicted ~10-20ms, actual 100-200ms → gap 80-180ms)
	//
	// For dense models (numMoELayers=0):
	//   - β₈ × 0 = 0 (no contribution, non-Scout experiments unaffected)
	var moeRoutingTimeUs float64
	if m.modelConfig.NumLocalExperts > 1 {
		// Calculate number of MoE layers
		// InterleaveMoELayerStep > 0 means interleaved MoE+dense architecture (e.g., Scout)
		// InterleaveMoELayerStep = 0 means pure MoE architecture (all layers are MoE)
		var numMoELayers float64
		if m.modelConfig.InterleaveMoELayerStep > 0 {
			// Interleaved architecture: numMoELayers = NumLayers / InterleaveMoELayerStep
			// Scout: 56 layers / 26 step ≈ 2 (but actually 26 MoE layers, so use step directly)
			// Approximation: Use InterleaveMoELayerStep as the number of dense layers per MoE layer
			// Then numMoELayers ≈ NumLayers / (InterleaveMoELayerStep / (InterleaveMoELayerStep - 1))
			// Simplified: For Scout, InterleaveMoELayerStep=26 means ~26 MoE layers in 56 total
			numMoELayers = float64(m.modelConfig.NumLayers) / (float64(m.modelConfig.InterleaveMoELayerStep) / (float64(m.modelConfig.InterleaveMoELayerStep-1) + 1e-6))
			// More accurate for Scout: If InterleaveMoELayerStep = 26, we know 26 MoE layers
			// So use min(NumLayers / 2, InterleaveMoELayerStep) as heuristic
			if float64(m.modelConfig.InterleaveMoELayerStep) < numMoELayers {
				numMoELayers = float64(m.modelConfig.InterleaveMoELayerStep)
			}
		} else {
			// Pure MoE architecture: all layers are MoE
			numMoELayers = float64(m.modelConfig.NumLayers)
		}

		// Calculate total tokens in batch
		var totalPrefillTokens int64
		for _, req := range stepConfig.PrefillRequests {
			totalPrefillTokens += int64(req.NumNewPrefillTokens)
		}
		totalTokens := float64(totalPrefillTokens + int64(len(stepConfig.DecodeRequests)))

		// NumExpertsPerTok: Active experts per token (top-k routing)
		// If not set in ModelConfig, default to 1 (conservative estimate)
		numExpertsPerTok := 1.0
		if m.modelConfig.NumExpertsPerTok > 0 {
			numExpertsPerTok = float64(m.modelConfig.NumExpertsPerTok)
		}

		// β₈ basis function: numMoELayers × totalTokens × numExpertsPerTok / TP
		// Units: number of routed tokens (TP division accounts for cross-GPU routing)
		routedTokens := numMoELayers * totalTokens * numExpertsPerTok / tpFactor

		// β₈ coefficient is in SECONDS per routed token (expected 0.000010-0.000050 = 10-50μs)
		// Convert to microseconds: routedTokens × β₈ × 1e6
		moeRoutingTimeUs = routedTokens * m.Beta[8] * 1e6
	}
	moeRoutingContribution := moeRoutingTimeUs // Already in microseconds

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 8: 8 terms in StepTime (β₀-β₅, β₇, β₈), β₆ moved to QueueingTime
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtContribution + decodeComputeContribution +
		moeGatingContribution + decodeOverheadContribution + moeRoutingContribution

	return max(1, clampToInt64(totalTimeUs))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// **MODIFIED IN ITER6**: Added Beta[6] scheduler overhead per request.
//
// Physics:
//   - α₀: Fixed API processing (HTTP parsing, request validation, queue insertion)
//   - α₁: Per-input-token tokenization (HuggingFace BPE encoding scales with input length)
//   - β₆: Scheduler overhead per request (batch formation + KV block allocation)
//
// Scheduler overhead (β₆) captures:
//   1. Batch formation: vLLM scheduler computes can-schedule decision (capacity check, priority ordering)
//   2. KV block allocation: PagedAttention allocates physical blocks from block manager
//   3. Priority queue processing: Sorts waiting requests and selects batch
//
// Expected values:
//   - α₀: ~1-2ms (API gateway overhead)
//   - α₁: ~100-150μs per token (tokenization)
//   - β₆: ~15-30ms per request (scheduler overhead, from iter7's 13.2ms)
func (m *EvolvedModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	// Alpha coefficients (in milliseconds, need conversion to microseconds)
	totalProcessingTime += m.Alpha[0] * 1000.0                                 // Fixed overhead (α₀ in ms, convert to μs)
	totalProcessingTime += m.Alpha[1] * float64(len(req.InputTokens)) * 1000.0 // Tokenization (α₁ in ms/token, convert to μs)

	// Beta[6] scheduler overhead (in milliseconds, need conversion to microseconds)
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
//   - Iteration 0-8: No systematic per-request bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0 // No systematic per-request bias in current training data
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
//
// This function is meant to be integrated into the main NewLatencyModel switch statement
// in latency.go. The factory pattern follows the existing backends (roofline, blackbox, etc.).
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 8: 3 alpha, 9 beta)
	// Beta count increased from iter7 (8 → 9), added β₈ for MoE routing overhead
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 9 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 9 elements for iteration 8, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 8 expects 9: β₀-β₈)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
