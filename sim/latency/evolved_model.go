package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 9: FP8 dequantization overhead mechanism.
//
// Critical discovery from iter8: β₈ (MoE routing overhead) converged to 30μs per routed token
// (physically plausible, within predicted 10-50μs range) and contributes ~39ms per Scout prefill
// request. However, Scout TTFT errors remained COMPLETELY UNCHANGED (79-100% APE, 0pp improvement
// from iter7). Overall loss stayed at 155.35% (vs iter7's 155.37%). This proves β₈ captures a
// REAL mechanism but is INSUFFICIENT — Scout's bottleneck is 100-200ms, not 39ms.
//
// Root cause: Scout is the ONLY FP8 model (torch.float8_e4m3fn, all others FP16/BF16). FP8 dynamic
// quantization introduces per-token dequantization overhead: FP8 → FP16/BF16 conversion before
// matmul, mixed-precision coordination (FP8 weights × FP16 activations), and dynamic scale factor
// management. This overhead (17-50μs per token per layer) is NOT captured by current model's
// compute/memory/communication basis functions.
//
// Gap analysis:
//   - Roofline underestimates Scout by -99.88% MPE (missing 99.88ms overhead)
//   - β₈ contribution: 39ms (only 39% of the gap)
//   - Remaining gap: 61ms (61% of missing overhead)
//   - β₃, β₅, β₇ remain inflated (4.4ms, 41μs, 26ms vs physical 0.4-1ms, 10-20μs, 10-20ms)
//
// Iter9 strategy: Add β₉ to capture per-token FP8 dequantization overhead, train on all 15
// experiments (including updated exp17 with clean general-lite data), and validate that β₉
// absorbs Scout's remaining 61ms gap while leaving non-FP8 experiments unaffected (β₉ = 0
// for non-FP8 models with BytesPerParam = 2.0).
//
// Changes from iter8:
//   - **Added β₉**: Per-token FP8 dequantization overhead (expected 17-50μs per token per layer)
//   - **Basis function**: β₉ × (totalTokens × numLayers × isFP8) where isFP8 = (BytesPerParam == 1.0)
//   - **Physics**: FP8 → FP16/BF16 dequant (10-30μs) + mixed-precision coord (5-15μs) + scale mgmt (2-5μs)
//   - **Warm-start**: All alpha/beta from iter8 optimal, β₉ initialized to 35μs per token per layer
//   - **Expected**: Scout TTFT 92% → <40%, β₃/β₅/β₇ revert to physical ranges, non-FP8 stable (<±10pp)
//   - **Data**: NEW exp17 (general-lite-2-1, clean data from 2026-03-30, normal server conditions)
//
// Basis functions:
//   - StepTime (9 beta terms: β₀-β₅, β₇-β₉, note β₆ in QueueingTime):
//     Prefill/decode compute, memory, communication, KV mgmt, MoE gating, MoE routing, decode overhead, FP8 dequant
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (10 total):
//   - β₀: Prefill compute MFU scaling (dimensionless, expected 0.15-0.25)
//   - β₁: Decode memory-bound MFU (dimensionless, expected 1.00-1.15)
//   - β₂: TP decode communication scaling (dimensionless, expected 0.20-0.35)
//   - β₃: KV cache management overhead (ms per request, expected 0.4-1.0ms, should REVERT from iter8's 4.4ms)
//   - β₄: Decode compute-bound MFU (dimensionless, expected 0.70-0.90, constrained ≤1.0)
//   - β₅: MoE gating overhead (ms, expected 0.010-0.020ms = 10-20μs, should REVERT from iter8's 41.1μs)
//   - β₆: Scheduler overhead per request (ms, expected 15-30ms) - used in QueueingTime, NOT StepTime
//   - β₇: Decode per-request overhead (ms, expected 10-20ms, should REVERT from iter8's 26.3ms) - used in StepTime
//   - β₈: MoE routing overhead per routed token (ms, expected 0.025-0.035ms = 25-35μs, stable from iter8) - used in StepTime
//   - β₉: **NEW** FP8 dequantization overhead per token per layer (ms, expected 0.017-0.050ms = 17-50μs) - used in StepTime
//
// Alpha coefficients (request-level, stable from iter8):
//   - α₀: Fixed API processing overhead (ms per request, ~1.3ms)
//   - α₁: Per-input-token tokenization (ms per token, ~118μs, bounds [0.0, 0.0002])
//   - α₂: Per-output-token detokenization (ms per token, ~91μs, bounds [0.0, 0.00015])
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇, β₈, β₉] - β₀-β₅,β₇,β₈,β₉ for StepTime, β₆ for QueueingTime
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 9 basis functions (9 terms in StepTime, β₆ in QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × kv_mgmt_time: KV cache management per request
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[5] × moe_gating_time: MoE gating network overhead
//   - beta[7] × num_decode_requests: Decode per-request overhead (output processing, TP coordination)
//   - beta[8] × moe_routing_time: MoE per-token routing overhead (expert selection, load balancing)
//   - beta[9] × fp8_dequant_time: **NEW** FP8 dequantization overhead (FP8→FP16 conversion, mixed-precision coordination)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - MoE gating: routing probability computation for all experts (captured by β₅)
//   - MoE routing: per-token expert selection, dispatch, load balancing (captured by β₈)
//   - FP8 dequantization (NEW): per-token per-layer overhead for FP8 models (captured by β₉)
//   - Decode overhead: Output processing, TP coordination, result aggregation per decode request
//
// Expected coefficients (iteration 9):
//   - β₀ ≈ 0.15-0.25 (prefill MFU, from iter8's 0.1912)
//   - β₁ ≈ 1.00-1.15 (decode memory-bound, from iter8's stabilized 1.1076)
//   - β₂ ≈ 0.20-0.35 (TP communication, from iter8's 0.1846)
//   - β₃ ≈ 0.4-1.0ms per request (KV block allocation, should REVERT from iter8's 4.40ms)
//   - β₄ ≈ 0.70-0.90 (decode compute-bound, from iter8's stabilized 0.7132)
//   - β₅ ≈ 10-20μs (MoE gating, should REVERT from iter8's 41.1μs)
//   - β₆ ≈ 15-30ms (scheduler overhead per request, from iter8's 13.2ms, used in QueueingTime)
//   - β₇ ≈ 10-20ms (decode overhead per request, should REVERT from iter8's 26.3ms)
//   - β₈ ≈ 25-35μs per routed token (MoE routing overhead, stable from iter8's 30μs)
//   - β₉ ≈ 17-50μs per token per layer (NEW: FP8 dequantization overhead)
//
// Beta[6] is NOT used in StepTime (moved to QueueingTime in iter6).
// Beta[7] is decode per-request overhead (added in iter7).
// Beta[8] is MoE routing overhead per routed token (added in iter8, real but insufficient for Scout).
// Beta[9] is NEW in iter9 (FP8 dequantization overhead per token per layer).
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
	// β₉ × fp8_dequant_time (NEW in iter9)
	// ========================================
	// Physics: FP8 dynamic quantization introduces per-token per-layer dequantization overhead
	// (FP8 → FP16/BF16 conversion) not captured by compute/memory/communication terms:
	//   1. Weight dequantization (10-30μs per token per layer): FP8 weights → FP16/BF16 before matmul
	//      - torch.float8_e4m3fn → torch.float16 conversion per layer
	//      - Dynamic quantization: scales computed per-tensor (additional overhead vs static)
	//   2. Mixed-precision coordination (5-15μs per token per layer): FP8 weights × FP16 activations
	//      - Tensor core mixed-precision mode requires synchronization
	//      - Precision mismatch handling per operation
	//   3. Dynamic scale management (2-5μs per token per layer): Per-tensor scale factors per layer
	//      - Scale recomputation on every forward pass (dynamic quantization)
	//      - Cross-GPU scale synchronization when TP > 1
	//
	// Total overhead: 17-50μs per token per layer (10-30 + 5-15 + 2-5)
	//
	// Expected range: 17-50μs per token per layer
	// Units: microseconds per token per layer
	// Code: vllm/model_executor/layers/quantization/fp8.py:Fp8LinearMethod.apply() line ~100-150
	//       Dequantization happens BEFORE tensor cores (preprocessing step, not captured by MFU)
	//
	// Basis function: β₉ × (totalTokens × numLayers × isFP8)
	//   - totalTokens: Prefill + decode tokens in batch
	//   - numLayers: Total number of layers in model (all layers use FP8 weights for Scout)
	//   - isFP8: 1 if BytesPerParam == 1.0 (FP8), 0 if BytesPerParam == 2.0 (FP16/BF16)
	//
	// For Scout (FP8, 56 layers, ~100 prefill tokens, TP=2):
	//   - Total tokens: 100 prefill + decode
	//   - β₉ contribution: β₉ × (100 × 56 × 1) = β₉ × 5600 ≈ 95-280ms per prefill request
	//   - This matches Scout TTFT residual (missing 61ms after β₈, actual full gap 100ms)
	//
	// For non-FP8 models (BytesPerParam = 2.0 for FP16/BF16):
	//   - isFP8 = 0 → β₉ × (... × 0) = 0 (no contribution, non-FP8 experiments unaffected)
	var fp8DequantTimeUs float64
	// Check if model uses FP8 weights (BytesPerParam == 1.0)
	// EffectiveWeightBytesPerParam() returns weight precision (1.0 for FP8, 2.0 for FP16/BF16)
	if m.modelConfig.EffectiveWeightBytesPerParam() == 1.0 {
		// Calculate total tokens in batch
		var totalPrefillTokens int64
		for _, req := range stepConfig.PrefillRequests {
			totalPrefillTokens += int64(req.NumNewPrefillTokens)
		}
		totalTokens := float64(totalPrefillTokens + int64(len(stepConfig.DecodeRequests)))

		// All layers use FP8 weights for FP8 models (Scout: 56 layers)
		numLayers := float64(m.modelConfig.NumLayers)

		// β₉ basis function: totalTokens × numLayers (isFP8 = 1 implicit in this branch)
		// Units: number of (token × layer) operations requiring FP8 dequantization
		tokenLayerOps := totalTokens * numLayers

		// β₉ coefficient is in SECONDS per token per layer (expected 0.000017-0.000050 = 17-50μs)
		// Convert to microseconds: tokenLayerOps × β₉ × 1e6
		fp8DequantTimeUs = tokenLayerOps * m.Beta[9] * 1e6
	}
	// else: non-FP8 model, fp8DequantTimeUs = 0 (default)
	fp8DequantContribution := fp8DequantTimeUs // Already in microseconds

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 9: 9 terms in StepTime (β₀-β₅, β₇-β₉), β₆ moved to QueueingTime
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtContribution + decodeComputeContribution +
		moeGatingContribution + decodeOverheadContribution + moeRoutingContribution +
		fp8DequantContribution

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
	// Validate coefficient counts (iteration 9: 3 alpha, 10 beta)
	// Beta count increased from iter8 (9 → 10), added β₉ for FP8 dequantization overhead
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 10 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 10 elements for iteration 9, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 9 expects 10: β₀-β₉)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
