package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 12: Memory Bandwidth Saturation via Widened β₃' Bounds (SIMPLIFIED DESIGN)
//
// **DESIGN EVOLUTION**: Original iter12 hypothesis (split β₆ → β₆ₐ + β₁₁ queueing) was REJECTED
// after analyzing profiling data. Second hypothesis (add β₁₁ bandwidth penalty) was REJECTED due
// to collinearity with β₃' (identical basis functions). Final simplified design: widen β₃' bounds
// to capture BOTH KV allocation (CPU-side) and bandwidth saturation (GPU-side) in single term.
//
// **HYPOTHESIS**: Memory bandwidth saturation during prefill causes triple explosion
// - β₂ (TP comm): 0.82 vs 0.25-0.60 expected (3× high)
// - β₃ (KV mgmt): 9.6ms vs 0.4-1.5ms expected (6× high)
// - β₆ (scheduler): 99ms vs 15-40ms expected (2.5-6× high, but profiling shows 40-100ms is CORRECT)
// - **Single root cause**: HBM bandwidth saturation slows down ALL memory operations simultaneously
//
// Critical discovery from iter9: Scout's bottleneck is **sequence-length-dependent**, NOT
// architecture-dependent (FP8). The FP8 hypothesis was REJECTED — β₉ converged to 0.14 μs
// (essentially zero) vs expected 17-50 μs. However, a powerful new pattern emerged:
//
// Sequence-Length Correlation (inverse relationship with error):
//   - Scout short-sequence (roleplay, codegen): Improved significantly (-53pp, -34pp TTFT from iter8)
//     • Roleplay: 79% → 26% TTFT (short sequences ~50-100 tokens)
//     • Codegen: 92% → 58% TTFT (moderate sequences ~100-200 tokens)
//   - Scout long-sequence (general-lite, reasoning-lite): Failed completely (0pp, -8pp TTFT from iter8)
//     • General-lite: 100% → 92% TTFT (long sequences ~400-600 tokens)
//     • Reasoning-lite: 99% → 91% TTFT (long sequences ~200-400 tokens)
//
// **Iter12 Simplified Strategy**: Widen β₃' bounds to capture BOTH mechanisms in single term
//   - **β₃' basis function**: Σ(prefillTokens × numLayers) [captures CPU + GPU effects]
//   - **Original role**: CPU-side KV cache block allocation (PagedAttention)
//   - **Extended role**: ALSO captures GPU-side HBM bandwidth saturation penalty
//   - **Solution**: Widen bounds from [0.05-2.0μs] to [0.05-5.0μs] (2.5× increase)
//   - **Advantage**: Single term, no collinearity, simpler than adding separate β₁₁
//   - **Cascading effect**: Should trigger β₂, β₃ reversion to physical ranges
//
// Profiling data findings (Scout, Llama-3.1, Qwen2.5):
//   - Cold-start effect: First 4-10 requests have 2-7× higher TTFT, then stabilizes
//   - Steady-state TTFT: 25-120ms (weakly correlated with tokens)
//   - **β₆ = 60-100ms is CORRECT** (captures cold-start + batch formation), expected range was WRONG
//
// Changes from iter11:
//   - **Widen β₃' bounds**: [0.05-2.0μs] → [0.05-5.0μs] (to capture bandwidth penalty)
//   - **Keep β₁₀ unchanged**: Iter11 audit proved correct (0% error in unit tests)
//   - **Warm-start from iter9**: Use iter9's optimal coefficients (loss 160.6%), NOT iter10/11
//   - **Update β₆ expected range**: 15-40ms → 40-100ms (based on profiling data)
//   - **Expected**: Cascading stabilization: β₂ (0.82 → 0.25-0.60), β₃ (9.6ms → 0.4-1.5ms),
//     β₃' (0.252μs → 1-3μs), β₆ (99ms → 40-100ms or accept as correct), overall loss <120%
//
// Basis functions:
//   - StepTime (10 beta terms: β₀-β₅, β₇-β₈, β₁₀, β₃', note β₆ in QueueingTime):
//     Prefill/decode compute, memory, communication, KV mgmt (base + seq-len with bandwidth penalty),
//     MoE gating, MoE routing, decode overhead, batching inefficiency
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (11 total, indexed 0-10):
//   - β₀ (index 0): Prefill compute MFU scaling (dimensionless, expected 0.14-0.22, stable)
//   - β₁ (index 1): Decode memory-bound MFU (dimensionless, expected 1.2-1.5, stable)
//   - β₂ (index 2): TP decode communication scaling (dimensionless, expected 0.25-0.60, EXPECT REVERT from iter9's 0.82 via cascading)
//   - β₃ (index 3): KV cache management BASE overhead (seconds per request, expected 0.0004-0.0015s = 0.4-1.5ms, EXPECT REVERT from iter9's 9.6ms via cascading)
//   - β₃' (index 4): KV cache + bandwidth overhead (seconds per token×layer, WIDENED to 0.05-5.0μs to capture BOTH KV allocation + bandwidth saturation, from iter10, expanded in iter12)
//   - β₄ (index 5): Decode compute-bound MFU (dimensionless, expected 0.40-0.65, stable)
//   - β₅ (index 6): MoE gating overhead (seconds, expected 15-25μs, stable)
//   - β₆ (index 7): Scheduler overhead per request (seconds, UPDATED RANGE: 40-100ms from profiling data, OLD 15-40ms was WRONG) - used in QueueingTime
//   - β₇ (index 8): Decode per-request overhead (seconds, expected 8-20ms, stable) - used in StepTime
//   - β₈ (index 9): MoE routing overhead per routed token (seconds, expected 25-80μs, stable) - used in StepTime
//   - β₁₀ (index 10): Batching inefficiency overhead (seconds per token²/batch_request, expected 0.1-1.0μs, from iter10, proven correct) - used in StepTime
//
// Alpha coefficients (request-level, constrained in iter10):
//   - α₀: Fixed API processing overhead (seconds per request, constrained ≥ 0.5ms)
//   - α₁: Per-input-token tokenization (seconds per token, constrained ≥ 50μs)
//   - α₂: Per-output-token detokenization (seconds per token, constrained ≥ 40μs)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₃', β₄, β₅, β₆, β₇, β₈, β₁₀] - 11 coefficients total (iter12 simplified)
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 12 basis functions (10 terms in StepTime, β₆ in QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × num_requests: KV cache management BASE overhead (PagedAttention setup)
//   - beta[4] × Σ(prefillTokens × numLayers): KV + bandwidth overhead (block allocation + HBM saturation, DUAL MECHANISM)
//   - beta[5] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[6] × moe_gating_time: MoE gating network overhead
//   - beta[8] × num_decode_requests: Decode per-request overhead (output processing, TP coordination)
//   - beta[9] × moe_routing_time: MoE per-token routing overhead (expert selection, load balancing)
//   - beta[10] × Σ(prefillTokens² / batchSize): Batching inefficiency overhead (queueing delays for long sequences)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation (base overhead per request + seq-len scaling with bandwidth penalty)
//   - MoE gating: routing probability computation for all experts (captured by β₅)
//   - MoE routing: per-token expert selection, dispatch, load balancing (captured by β₈)
//   - Batching inefficiency: long sequences consume disproportionate batch capacity → queueing delays (captured by β₁₀)
//   - Memory bandwidth saturation (iter12): HBM bandwidth contention during prefill (captured by widened β₃', DUAL MECHANISM)
//   - Decode overhead: Output processing, TP coordination, result aggregation per decode request
//
// Expected coefficients (iteration 12):
//   - β₀ ≈ 0.14-0.22 (prefill MFU, stable)
//   - β₁ ≈ 1.2-1.5 (decode memory-bound, stable)
//   - β₂ ≈ 0.25-0.60 (TP communication, EXPECT REVERT from iter9's 0.82 via cascading after widened β₃')
//   - β₃ ≈ 0.4-1.5ms per request (KV base overhead, EXPECT REVERT from iter9's 9.6ms via cascading after widened β₃')
//   - β₃' ≈ 1.0-3.0μs per (token×layer) (KV + bandwidth, WIDENED to capture BOTH mechanisms, EXPECT 4-12× increase from iter11's 0.252μs)
//   - β₄ ≈ 0.40-0.65 (decode compute-bound, stable)
//   - β₅ ≈ 15-25μs (MoE gating, stable)
//   - β₆ ≈ 40-100ms (scheduler overhead, UPDATED RANGE from profiling data, OLD 15-40ms was WRONG, used in QueueingTime)
//   - β₇ ≈ 8-20ms (decode overhead per request, stable)
//   - β₈ ≈ 25-80μs per routed token (MoE routing, stable)
//   - β₁₀ ≈ 0.1-1.0 μs per (token²/batch_request) (batching inefficiency, proven correct)
//
// Beta[7] is β₆, used in QueueingTime (NOT StepTime).
// Beta[8] is β₇ (decode per-request overhead, added in iter7).
// Beta[9] is β₈ (MoE routing overhead per routed token, added in iter8).
// Beta[10] is β₁₀ (batching inefficiency, added in iter10, proven correct).
// Beta[4] is β₃' (KV + bandwidth, added in iter10, WIDENED in iter12 to capture dual mechanism).
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
	// Expected range: 0.14-0.22 (iter9: 0.162)
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
	// Expected range: β₁ ≈ 1.2-1.5 (iter9: 1.361),
	//                 β₄ ≈ 0.40-0.65 (iter9: 0.466, index 5 in Beta array)
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
	decodeComputeContribution := m.Beta[5] * decodeComputeTimeUs * computeWeight // β₄ is at index 5

	// ========================================
	// β₂ × tp_comm_time (TP communication for DECODE)
	// ========================================
	// Physics: Ring all-reduce after each transformer layer (TP > 1), DECODE ONLY
	// Expected range: 0.25-0.60 (should decrease from iter9's 0.82 after β₁₀ offloads long-seq overhead)
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
	// β₃ × num_requests (KV cache management BASE overhead)
	// ========================================
	// Physics: vLLM PagedAttention base per-request overhead (block manager initialization, queue insertion)
	// Expected range: 0.4-1.5ms per request (should revert from iter9's 9.6ms after β₃' split)
	// Units: seconds per request (converted to μs below)
	// Code: vllm/core/block_manager.py:BlockSpaceManager.can_allocate() checks available blocks
	kvMgmtBaseTimeSeconds := float64(len(batch)) * m.Beta[3] // β₃ at index 3
	kvMgmtBaseContribution := kvMgmtBaseTimeSeconds * 1e6    // Convert to microseconds

	// ========================================
	// β₃' × Σ(prefillTokens × numLayers) (KV cache + bandwidth overhead, DUAL MECHANISM in iter12)
	// ========================================
	// Physics: DUAL MECHANISM (both scale with tokens × layers, captured by single term)
	//   1. CPU-side: KV cache block allocation (PagedAttention) - 0.1-1.0μs per (token×layer)
	//   2. GPU-side: HBM bandwidth saturation penalty during prefill - adds 0.5-4.0μs per (token×layer)
	// Expected range (WIDENED in iter12): 0.05-5.0μs per (token×layer)
	// Expected value: 1.0-3.0μs (4-12× increase from iter11's 0.252μs to capture bandwidth effect)
	// Units: seconds per (token×layer) (converted to μs below)
	// Code: vllm/core/block_manager.py:BlockSpaceManager.allocate() (CPU-side)
	//       H100 HBM3 memory controller queuing during prefill (GPU-side, physical bottleneck)
	//
	// Expected contribution (with widened β₃' = 1-3μs):
	//   - Scout general-lite (500 tokens × 56 layers): β₃' × 28,000 = 2μs × 28,000 ≈ 56ms
	//   - Scout roleplay (100 tokens × 56 layers): β₃' × 5,600 = 2μs × 5,600 ≈ 11ms
	// Comparison to iter11 (β₃' = 0.252μs):
	//   - Iter11: β₃' × 28,000 ≈ 7ms (underestimating, captures KV allocation only)
	//   - Iter12: β₃' × 28,000 ≈ 56ms (8× larger, captures BOTH KV allocation + bandwidth penalty)
	var kvMgmtSeqLenTokenLayers float64
	for _, req := range batch {
		// For prefill requests: use NumNewPrefillTokens (tokens being processed this step)
		// For decode requests: KV cache already allocated, seq-len overhead = 0 for this step
		if req.ProgressIndex < int64(len(req.InputTokens)) {
			// Prefill phase: allocate KV blocks for new prefill tokens
			numPrefillTokens := float64(req.NumNewTokens)
			kvMgmtSeqLenTokenLayers += numPrefillTokens * float64(m.modelConfig.NumLayers)
		}
		// Decode phase: no new KV allocation overhead (blocks already allocated during prefill)
	}
	kvMgmtSeqLenTimeSeconds := kvMgmtSeqLenTokenLayers * m.Beta[4] // β₃' at index 4
	kvMgmtSeqLenContribution := kvMgmtSeqLenTimeSeconds * 1e6      // Convert to microseconds

	// ========================================
	// β₅ × moe_gating_time (MoE gating)
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 15-25μs (stable from iter9's 19.8μs ✓)
	// Units: seconds (converted to μs below)
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
	moeGatingContribution := m.Beta[6] * moeGatingTimeUs // β₅ at index 6

	// ========================================
	// β₇ × decode_per_request_overhead (iter7)
	// ========================================
	// Physics: vLLM decode phase has fixed overhead per request beyond memory/compute:
	//   1. Output processing: After each decode step, vLLM processes output tokens
	//      (sampling, stop condition check, streaming updates)
	//   2. TP coordination: Decode requires per-request coordination across TP workers
	//      (synchronization barriers per step)
	//   3. KV cache write-back: Updated KV cache blocks written back to memory
	// Expected range: 8-20ms per decode request (stable from iter9's 11ms ✓)
	// Units: seconds per request (converted to μs below)
	// Code: vllm/model_executor/model_loader.py:_run_workers() calls execute_model() per step
	//       vllm/worker/worker.py:execute_model() synchronizes across TP ranks per step
	numDecodeRequests := len(stepConfig.DecodeRequests)
	decodeOverheadTimeSeconds := float64(numDecodeRequests) * m.Beta[8] // β₇ at index 8
	decodeOverheadContribution := decodeOverheadTimeSeconds * 1e6        // Convert to microseconds

	// ========================================
	// β₈ × moe_routing_time (iter8)
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
	// Expected range: 25-80μs per routed token (may decrease from iter9's 73μs)
	// Units: seconds per routed token (converted to μs below)
	// Code: vllm/model_executor/layers/fused_moe/fused_moe.py:fused_experts()
	//       Line ~150-200: Expert routing implementation (selection, dispatch, aggregation)
	//
	// Basis function: β₈ × (numMoELayers × totalTokens × numExpertsPerTok / TP)
	//   - numMoELayers: Number of MoE layers in model (26 for Scout, 0 for dense models)
	//   - totalTokens: Prefill + decode tokens in batch
	//   - numExpertsPerTok: Active experts per token (k=1 or k=2, default to 1 if not set)
	//   - TP: Tensor parallelism degree (expert routing scales inversely with TP)
	var moeRoutingTimeUs float64
	if m.modelConfig.NumLocalExperts > 1 {
		// Calculate number of MoE layers
		var numMoELayers float64
		if m.modelConfig.InterleaveMoELayerStep > 0 {
			numMoELayers = float64(m.modelConfig.NumLayers) / (float64(m.modelConfig.InterleaveMoELayerStep) / (float64(m.modelConfig.InterleaveMoELayerStep-1) + 1e-6))
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
		numExpertsPerTok := 1.0
		if m.modelConfig.NumExpertsPerTok > 0 {
			numExpertsPerTok = float64(m.modelConfig.NumExpertsPerTok)
		}

		// β₈ basis function: numMoELayers × totalTokens × numExpertsPerTok / TP
		routedTokens := numMoELayers * totalTokens * numExpertsPerTok / tpFactor

		// β₈ coefficient is in SECONDS per routed token (expected 0.000025-0.000080 = 25-80μs)
		// Convert to microseconds: routedTokens × β₈ × 1e6
		moeRoutingTimeUs = routedTokens * m.Beta[9] * 1e6 // β₈ at index 9
	}
	moeRoutingContribution := moeRoutingTimeUs // Already in microseconds

	// ========================================
	// β₁₀ × Σ(prefillTokens² / batchSize) (Batching inefficiency overhead, NEW in iter10, CORRECTED in iter11)
	// ========================================
	// Physics: Long sequences consume disproportionate batch capacity, leading to queueing delays:
	//   1. Batch packing constraint: Σ(prefill_tokens + kv_cache_blocks) ≤ max_num_batched_tokens
	//      - Long sequences (500 tokens) consume 10× more capacity than short sequences (50 tokens)
	//      - Fewer requests fit in each batch → lower GPU utilization → increased wait time
	//   2. Quadratic penalty: prefillTokens² captures disproportionate impact on batch efficiency
	//      - Batch size penalty: long sequences → fewer requests per batch
	//      - Queueing amplification: low batch efficiency → longer queue waits
	//   3. Division by batchSize: Amplifies effect for long sequences (lower batch efficiency → smaller denominator)
	//
	// **CORRECTED from iter10**: Expected range: 0.1-1.0 **μs** per (token²/batch_request), NOT 0.1-1.0 ms!
	// Units: seconds per (token²/batch_request) (converted to μs below)
	// Code: vllm/core/scheduler.py:Scheduler._schedule() line ~300-400 (batch formation logic)
	//
	// Expected contribution (CORRECTED):
	//   - Scout general-lite (500 tokens, batch_size=4): β₁₀ × (500²/4) = 0.5μs × 62,500 ≈ 31.25ms
	//   - Scout roleplay (100 tokens, batch_size=32): β₁₀ × (100²/32) = 0.5μs × 312 ≈ 0.156ms
	//   - Ratio: 200× difference (quadratic scaling amplifies long-sequence overhead)
	//
	// **Iter10 analysis**: β₁₀ converged to 0.945μs, giving contributions of 59ms (long-seq) and 0.3ms (short-seq).
	// These are physically reasonable! The iter10 hypothesis expected 0.1-1.0 ms (1000× too large), leading to
	// erroneous conclusion of "formulation bug." The basis function implementation is CORRECT.
	var batchingInefficiencySum float64
	effectiveBatchSize := float64(len(batch))
	if effectiveBatchSize < 1.0 {
		effectiveBatchSize = 1.0 // Prevent division by zero
	}

	for _, req := range batch {
		// Only apply to prefill requests (batching inefficiency occurs during prefill scheduling)
		if req.ProgressIndex < int64(len(req.InputTokens)) {
			numPrefillTokens := float64(req.NumNewTokens)
			// Quadratic term: prefillTokens² / batchSize
			batchingInefficiencySum += (numPrefillTokens * numPrefillTokens) / effectiveBatchSize
		}
		// Decode requests: no batching inefficiency overhead (already scheduled)
	}

	batchingInefficiencyTimeSeconds := batchingInefficiencySum * m.Beta[10] // β₁₀ at index 10
	batchingInefficiencyContribution := batchingInefficiencyTimeSeconds * 1e6 // Convert to microseconds

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 12 simplified: 10 terms in StepTime (β₀-β₅, β₇-β₈, β₁₀, β₃'), β₆ in QueueingTime
	// β₃' captures BOTH KV allocation (CPU-side) and bandwidth saturation (GPU-side) via widened bounds
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtBaseContribution + kvMgmtSeqLenContribution +
		decodeComputeContribution + moeGatingContribution + decodeOverheadContribution +
		moeRoutingContribution + batchingInefficiencyContribution

	return max(1, clampToInt64(totalTimeUs))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// **MODIFIED IN ITER6**: Added Beta[7] scheduler overhead per request.
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
//   4. Cold-start overhead: First 4-10 requests have 2-7× higher TTFT (profiling data finding)
//
// Expected values (iter12, UPDATED based on profiling data):
//   - α₀: 0.8-2.5ms (constrained ≥ 0.5ms to prevent spurious reduction)
//   - α₁: 60-150μs per token (constrained ≥ 50μs)
//   - β₆: 40-100ms per request (UPDATED from old 15-40ms based on profiling data showing cold-start overhead)
func (m *EvolvedModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	// Alpha coefficients (in seconds, convert to microseconds)
	totalProcessingTime += m.Alpha[0] * 1e6                                  // Fixed overhead (α₀ in seconds, convert to μs)
	totalProcessingTime += m.Alpha[1] * float64(len(req.InputTokens)) * 1e6 // Tokenization (α₁ in seconds/token, convert to μs)

	// Beta[7] scheduler overhead (in seconds, convert to microseconds)
	totalProcessingTime += m.Beta[7] * 1e6 // Scheduler overhead (β₆ at index 7, in seconds, convert to μs)

	return clampToInt64(totalProcessingTime)
}

// OutputTokenProcessingTime returns per-output-token post-processing overhead.
// **DO NOT MODIFY THIS METHOD.** Standard implementation: α₂ (streaming detokenization)
//
// Physics:
//   - α₂: Per-output-token detokenization + output formatting in streaming mode
//   - Applied per output token during decode phase
//
// Expected value (iter10): 50-120μs per token (constrained ≥ 40μs)
func (m *EvolvedModel) OutputTokenProcessingTime() int64 {
	// α₂ is in seconds, convert to microseconds
	return clampToInt64(m.Alpha[2] * 1e6)
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// **DO NOT MODIFY THIS METHOD.** Return 0 unless systematic per-request bias observed.
//
// Physics:
//   - Models constant overhead at request completion (e.g., response finalization, logging)
//   - Iteration 0-10: No systematic per-request bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0 // No systematic per-request bias in current training data
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
//
// This function is meant to be integrated into the main NewLatencyModel switch statement
// in latency.go. The factory pattern follows the existing backends (roofline, blackbox, etc.).
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 12 simplified: 3 alpha, 11 beta)
	// Beta count: β₀-β₈ (9 coefficients) + β₃' (1 coefficient) + β₁₀ (1 coefficient) = 11 total
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 11 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 11 elements for iteration 12, got %d (expected β₀-β₈, β₃', β₁₀)", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 12 simplified expects 11: β₀-β₈, β₃', β₁₀)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
