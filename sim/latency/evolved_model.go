package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 15: Three-Axis Correction — Decode Amplification + MoE Non-Compute + Dense Batching
//
// **CONTEXT**: Iterations 10-14 failed catastrophically (loss 2000-4000%). Iter14's β₅ MoE gating
// fix was necessary but insufficient (loss improved only 2.8%: 2387% → 2319%). Root cause analysis
// of baseline simulator errors (`training/baseline_errors.json`) reveals the roofline model has
// THREE INDEPENDENT systematic errors:
//
// 1. **Decode underestimation** (-90% to -95% ITL MPE across ALL 13 experiments)
//    - Roofline assumes decode achieves theoretical MFU (0.10-0.20 memory, 0.60-0.80 compute)
//    - Reality: vLLM decode is 10-20× slower (small GEMMs, PagedAttention overhead, pointer chasing)
//    - Fix: Amplify β₁ (decode memory) and β₄ (decode compute) by 5-15× (new bounds: 5.0-15.0, 3.0-8.0)
//
// 2. **MoE underestimation** (-69% avg TTFT MPE for Scout experiments)
//    - Roofline computes expert FLOPs correctly, but MoE has non-compute overhead beyond expert execution
//    - Missing: Token routing (scatter/gather), load imbalance (stragglers), expert communication (all-to-all)
//    - Fix: Add β₈ (MoE non-compute latency) — per-token routing overhead (20-80 μs/token)
//
// 3. **Dense overestimation** (+820% avg TTFT MPE for dense models)
//    - Roofline assumes prefill batches achieve near-peak MFU (0.50-0.60 on H100)
//    - Reality: Mixed prefill/decode batches suffer from batch heterogeneity, memory layout mismatch, synchronization overhead
//    - Fix: Add β₉ (prefill batching penalty) — heterogeneity-induced MFU degradation (0.5-2.0 μs/token)
//
// **ITER15 STRATEGY**: Address all three errors simultaneously with cold-start optimization
//   1. Amplify β₁, β₄ bounds by 5-15× to match 10-20× decode slowdown (baseline shows -90% to -95% ITL MPE)
//   2. Add β₈ (MoE non-compute) to capture routing latency + load imbalance beyond FLOPs
//   3. Add β₉ (prefill batching penalty) to capture mixed-batch MFU degradation (heterogeneity factor)
//   4. Cold-start with random initialization (NOT warm-start from iter7/iter14) — dataset shifted
//   5. Increase optimization budget from 1000 to 2000 trials (10-dimensional search requires more exploration)
//
// **EXPECTED OUTCOME**: Overall loss 2319% → <300% (≥87% improvement), with all three error axes corrected
//
// Basis functions (iter15 architecture):
//   - StepTime (8 beta terms: β₀-β₅, β₇, β₈, β₉, note β₆ in QueueingTime):
//     Prefill compute, decode memory (AMPLIFIED), TP comm, KV base, decode compute (AMPLIFIED),
//     MoE gating (fixed in iter14), decode overhead, MoE non-compute (NEW), prefill batching penalty (NEW)
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (10 total, indexed 0-9):
//   - β₀ (index 0): Prefill compute MFU (dimensionless, expected 0.05-0.25, SCALE DOWN from iter7's 0.16-0.22)
//   - β₁ (index 1): Decode memory MFU (dimensionless, expected 5.0-15.0, 10× amplification from 1.0-1.15)
//   - β₂ (index 2): TP communication scaling (dimensionless, expected 0.15-0.25, unchanged)
//   - β₃ (index 3): KV management BASE overhead (seconds per request, expected 0.4-1.5ms, unchanged)
//   - β₄ (index 4): Decode compute MFU (dimensionless, expected 3.0-8.0, 5× amplification from 0.7-0.85)
//   - β₅ (index 5): MoE gating efficiency (dimensionless, expected 20-50, fixed in iter14)
//   - β₆ (index 6): Scheduler overhead per request (seconds, expected 40-100ms, unchanged) - used in QueueingTime
//   - β₇ (index 7): Decode per-request overhead (seconds, expected 15-30ms, unchanged) - used in StepTime
//   - β₈ (index 8): MoE non-compute latency (seconds per token, expected 10-40 μs/token, NEW)
//   - β₉ (index 9): Prefill batching penalty (seconds per token, expected 0.5-2.0 μs/token, NEW)
//
// Alpha coefficients (request-level, unchanged):
//   - α₀: Fixed API processing overhead (seconds per request, 0.5-2.5ms)
//   - α₁: Per-input-token tokenization (seconds per token, 50-150μs)
//   - α₂: Per-output-token detokenization (seconds per token, 40-120μs)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇, β₈, β₉] - 10 coefficients total (iter15: added β₈, β₉)
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 15 basis functions (8 terms in StepTime, β₆ in QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound (AMPLIFIED 10×)
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × num_requests: KV cache management BASE overhead (PagedAttention setup)
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound (AMPLIFIED 5×)
//   - beta[5] × moe_gating_time: MoE gating network overhead (fixed in iter14 with layer multiplier)
//   - beta[7] × num_decode_requests: Decode per-request overhead (output processing, TP coordination)
//   - beta[8] × moe_noncompute_time: MoE non-compute overhead (routing latency, load imbalance) - NEW
//   - beta[9] × prefill_batching_penalty: Prefill batching inefficiency from heterogeneity - NEW
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation (base overhead per request only)
//   - MoE gating: routing probability computation for all experts (fixed in iter14 with layer multiplier)
//   - MoE non-compute: token routing (scatter/gather), load imbalance (stragglers), expert communication
//   - Prefill batching: MFU degradation from mixing long prefills with short decodes (heterogeneity penalty)
//   - Decode overhead: Output processing, TP coordination, result aggregation per decode request
//
// Expected coefficients (iteration 15, three-axis correction):
//   - β₀ ≈ 0.05-0.25 (prefill MFU, SCALE DOWN from iter7's 0.16-0.22 to fix dense overestimation)
//   - β₁ ≈ 5.0-15.0 (decode memory-bound, 10× amplification from iter7's 1.108)
//   - β₂ ≈ 0.15-0.25 (TP communication, unchanged from iter7)
//   - β₃ ≈ 0.4-1.5ms per request (KV base overhead, unchanged from iter7)
//   - β₄ ≈ 3.0-8.0 (decode compute-bound, 5× amplification from iter7's 0.713)
//   - β₅ ≈ 20-50 (MoE gating, iter14 prediction range after layer multiplier fix)
//   - β₆ ≈ 40-100ms (scheduler overhead, unchanged from iter7, used in QueueingTime)
//   - β₇ ≈ 15-30ms (decode overhead per request, unchanged from iter7)
//   - β₈ ≈ 10-40 μs/token (MoE non-compute latency, NEW, reduced from initial 20-80)
//   - β₉ ≈ 0.5-2.0 μs/token (prefill batching penalty, NEW)
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
	// Expected range: 0.5-1.0 (widened from iter7's 0.16-0.22)
	// Units: seconds (converted to μs below)
	var prefillComputeTimeSeconds float64
	var totalPrefillTokens int64
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)
		totalPrefillTokens += numTokens
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, numTokens, true, true)
		prefillComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuPrefill)
	}
	prefillTimeUs := prefillComputeTimeSeconds * 1e6
	prefillContribution := m.Beta[0] * prefillTimeUs

	// ========================================
	// β₉ × prefill_batching_penalty (NEW in iter15)
	// ========================================
	// Physics: Mixed prefill/decode batches suffer MFU degradation from heterogeneity
	// Expected range: 0.5-2.0 μs/token (additive overhead on top of β₀ × prefill_time)
	// Units: seconds per token (converted to μs below)
	//
	// Heterogeneity factor: num_decode_tokens / max(1, num_prefill_tokens)
	//   - High for codegen (many 1-token decode requests mixed with large prefills) → large β₉ contribution
	//   - Low for roleplay/reasoning (fewer decode requests per batch) → small β₉ contribution
	//
	// Why this addresses dense overestimation:
	//   - Dense models baseline shows +820% avg TTFT MPE (roofline assumes 0.5-0.6 MFU)
	//   - Reality: Mixed batches don't achieve theoretical MFU due to kernel inefficiency
	//   - β₉ adds per-token penalty proportional to batch heterogeneity
	var prefillBatchingPenaltySeconds float64
	if totalPrefillTokens > 0 {
		numDecodeRequests := len(stepConfig.DecodeRequests)
		heterogeneityFactor := float64(numDecodeRequests) / math.Max(1.0, float64(totalPrefillTokens))
		prefillBatchingPenaltySeconds = float64(totalPrefillTokens) * heterogeneityFactor * m.Beta[9] // β₉ at index 9
	}
	prefillBatchingPenaltyUs := prefillBatchingPenaltySeconds * 1e6
	prefillBatchingContribution := prefillBatchingPenaltyUs

	// ========================================
	// β₁ × decode_memory_time × memory_weight (small-batch) — **AMPLIFIED 10× in iter15**
	// β₄ × decode_compute_time × compute_weight (large-batch) — **AMPLIFIED 5× in iter15**
	// ========================================
	// Physics: Decode transitions from memory-bound (small batches) to compute-bound (large batches)
	// Expected range: β₁ ≈ 5.0-15.0 (10× amplification from iter7's 1.108), β₄ ≈ 3.0-8.0 (5× amplification from iter7's 0.713)
	// Units: seconds (converted to μs below)
	// Sigmoid interpolation: memory_weight(n) = 1 / (1 + exp((n - 8) / 2))
	//
	// Why amplification is needed:
	//   - Baseline shows -90% to -95% ITL MPE across ALL 13 experiments (decode 10-20× slower than roofline)
	//   - Roofline assumes decode achieves theoretical MFU (0.10-0.20 memory, 0.60-0.80 compute)
	//   - Reality: Small GEMMs, PagedAttention overhead, pointer chasing, per-layer synchronization
	//   - β₁, β₄ now amplify roofline decode estimates by 5-15× to match reality
	var decodeMemoryTimeSeconds float64
	var decodeComputeTimeSeconds float64
	batchSize := len(stepConfig.DecodeRequests)
	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, 1, true, true)
		decodeMemoryTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuDecode)
		largeBatchMFU := m.hwConfig.MfuDecode * 1.2
		if largeBatchMFU > 1.0 {
			largeBatchMFU = 1.0
		}
		decodeComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * largeBatchMFU)
	}

	memoryWeight := 1.0 / (1.0 + math.Exp((float64(batchSize)-8.0)/2.0))
	computeWeight := 1.0 - memoryWeight

	decodeMemoryTimeUs := decodeMemoryTimeSeconds * 1e6
	decodeComputeTimeUs := decodeComputeTimeSeconds * 1e6
	decodeMemoryContribution := m.Beta[1] * decodeMemoryTimeUs * memoryWeight // β₁ at index 1 (AMPLIFIED)
	decodeComputeContribution := m.Beta[4] * decodeComputeTimeUs * computeWeight // β₄ at index 4 (AMPLIFIED)

	// ========================================
	// β₂ × tp_comm_time (TP communication for DECODE)
	// ========================================
	// Physics: Ring all-reduce after each transformer layer (TP > 1), DECODE ONLY
	// Expected range: 0.15-0.25 (unchanged from iter7)
	// Units: seconds (converted to μs below)
	var tpCommTimeSeconds float64
	if m.tp > 1 {
		allReduceBytesPerLayer := 2.0 * float64(m.modelConfig.HiddenDim) * bytesPerParam
		tpCommFactor := float64(m.tp-1) / float64(m.tp)
		nvlinkBandwidthBytesPerSec := m.hwConfig.BwPeakTBs * 1e12
		tpCommTimeSeconds = tpCommFactor * float64(m.modelConfig.NumLayers) * allReduceBytesPerLayer / nvlinkBandwidthBytesPerSec
	}
	tpCommTimeUs := tpCommTimeSeconds * 1e6
	tpCommContribution := m.Beta[2] * tpCommTimeUs

	// ========================================
	// β₃ × num_requests (KV cache management BASE overhead)
	// ========================================
	// Physics: vLLM PagedAttention base per-request overhead (block manager initialization, queue insertion)
	// Expected range: 0.4-1.5ms per request (unchanged from iter7)
	// Units: seconds per request (converted to μs below)
	kvMgmtBaseTimeSeconds := float64(len(batch)) * m.Beta[3] // β₃ at index 3
	kvMgmtBaseContribution := kvMgmtBaseTimeSeconds * 1e6

	// ========================================
	// β₅ × moe_gating_time (MoE gating) - **FIXED in iter14**
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 20-50 (fixed in iter14, unchanged in iter15)
	// Units: seconds (converted to μs below)
	//
	// **NOTE**: β₅ captures gating COMPUTE cost (FLOPs). β₈ (NEW in iter15) captures gating LATENCY cost (routing overhead).
	//
	// Gating network: hidden_dim → num_experts linear projection per token per layer
	// FLOPs per layer: 2 × tokens × hidden_dim × num_experts
	// Total FLOPs: (per-layer FLOPs) × numMoELayers  ← Fixed in iter14!
	var moeGatingTimeSeconds float64
	if m.modelConfig.NumLocalExperts > 1 {
		// Calculate numMoELayers (same pattern as iter14, lines 240-252)
		var numMoELayers float64
		if m.modelConfig.InterleaveMoELayerStep > 0 {
			// Handle Scout-style interleaved MoE+dense architectures
			// InterleaveMoELayerStep=4 means: 1 MoE layer every 4 layers (pattern: MoE, dense, dense, dense, ...)
			numMoELayers = float64(m.modelConfig.NumLayers) / (float64(m.modelConfig.InterleaveMoELayerStep) / (float64(m.modelConfig.InterleaveMoELayerStep-1) + 1e-6))
			if float64(m.modelConfig.InterleaveMoELayerStep) < numMoELayers {
				numMoELayers = float64(m.modelConfig.InterleaveMoELayerStep)
			}
		} else {
			// Pure MoE architecture: all layers are MoE
			numMoELayers = float64(m.modelConfig.NumLayers)
		}

		totalTokens := totalPrefillTokens + int64(len(stepConfig.DecodeRequests))

		// Compute gating FLOPs per layer
		gatingFlopsPerLayer := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)

		// Compute gating time per layer
		gatingEfficiency := 0.3 // Assume 30% MFU (same as iter14)
		gatingTimePerLayerSeconds := gatingFlopsPerLayer / tpFactor / (peakFlops * gatingEfficiency)

		// Multiply by numMoELayers to account for ALL MoE layers (fixed in iter14)
		moeGatingTimeSeconds = gatingTimePerLayerSeconds * numMoELayers
	}
	moeGatingTimeUs := moeGatingTimeSeconds * 1e6
	moeGatingContribution := m.Beta[5] * moeGatingTimeUs // β₅ at index 5

	// ========================================
	// β₈ × moe_noncompute_time (NEW in iter15)
	// ========================================
	// Physics: MoE non-compute overhead beyond expert FLOPs
	// Expected range: 20-80 μs/token (per-token routing latency + load imbalance + expert communication)
	// Units: seconds per token (converted to μs below)
	//
	// Why this is needed:
	//   - Scout baseline shows -69% avg TTFT MPE (underestimation by 2-3×)
	//   - β₅ (MoE gating) only captures gating network FLOPs (routing probability computation)
	//   - But MoE has non-compute overhead that's not captured by FLOPs:
	//     1. **Token routing**: Scatter/gather operations to dispatch tokens to selected experts
	//     2. **Load imbalance**: When some experts get >2× avg load, stragglers dominate latency
	//     3. **Expert communication**: All-to-all tensor routing in distributed MoE (TP > 1)
	//   - These are communication/synchronization-bound, not compute-bound
	//   - β₈ adds a per-token latency term independent of FLOPs
	//
	// Basis function: num_moe_layers × total_tokens × moe_routing_latency_per_token
	//   - Applies to both prefill and decode tokens (routing happens for all tokens in MoE layers)
	//   - Scales linearly with number of MoE layers (more layers → more routing overhead)
	var moeNonComputeTimeSeconds float64
	if m.modelConfig.NumLocalExperts > 1 {
		// Calculate numMoELayers (same logic as β₅)
		var numMoELayers float64
		if m.modelConfig.InterleaveMoELayerStep > 0 {
			numMoELayers = float64(m.modelConfig.NumLayers) / (float64(m.modelConfig.InterleaveMoELayerStep) / (float64(m.modelConfig.InterleaveMoELayerStep-1) + 1e-6))
			if float64(m.modelConfig.InterleaveMoELayerStep) < numMoELayers {
				numMoELayers = float64(m.modelConfig.InterleaveMoELayerStep)
			}
		} else {
			numMoELayers = float64(m.modelConfig.NumLayers)
		}

		totalTokens := totalPrefillTokens + int64(len(stepConfig.DecodeRequests))

		// Per-token non-compute latency × total tokens × num MoE layers
		// β₈ is in seconds per token, so total time = β₈ × tokens × layers
		moeNonComputeTimeSeconds = m.Beta[8] * float64(totalTokens) * numMoELayers // β₈ at index 8
	}
	moeNonComputeTimeUs := moeNonComputeTimeSeconds * 1e6
	moeNonComputeContribution := moeNonComputeTimeUs

	// ========================================
	// β₇ × decode_per_request_overhead (unchanged from iter7)
	// ========================================
	// Physics: vLLM decode phase has fixed overhead per request beyond memory/compute
	//   1. Output processing: sampling, stop condition check, streaming updates
	//   2. TP coordination: synchronization barriers per step
	//   3. KV cache write-back: updated blocks written to memory
	// Expected range: 15-30ms per decode request (unchanged from iter7)
	// Units: seconds per request (converted to μs below)
	numDecodeRequests := len(stepConfig.DecodeRequests)
	decodeOverheadTimeSeconds := float64(numDecodeRequests) * m.Beta[7] // β₇ at index 7
	decodeOverheadContribution := decodeOverheadTimeSeconds * 1e6

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 15: 8 terms in StepTime (β₀-β₅, β₇, β₈, β₉), β₆ in QueueingTime
	// Added in iter15: β₈ (MoE non-compute), β₉ (prefill batching)
	totalTimeUs := prefillContribution + prefillBatchingContribution +
		decodeMemoryContribution + tpCommContribution + kvMgmtBaseContribution +
		decodeComputeContribution + moeGatingContribution + moeNonComputeContribution +
		decodeOverheadContribution

	return max(1, clampToInt64(totalTimeUs))
}

// QueueingTime computes request-level overhead (ARRIVED → QUEUED).
// **PRESERVED FROM ITER7**: Beta[6] scheduler overhead per request.
//
// Physics:
//   - α₀: Fixed API processing (HTTP parsing, request validation, queue insertion)
//   - α₁: Per-input-token tokenization (HuggingFace BPE encoding scales with input length)
//   - β₆: Scheduler overhead per request (batch formation + KV block allocation)
//
// Scheduler overhead (β₆) captures:
//   1. Batch formation: vLLM scheduler computes can-schedule decision
//   2. KV block allocation: PagedAttention allocates physical blocks
//   3. Priority queue processing: Sorts waiting requests and selects batch
//   4. Cold-start overhead: First 4-10 requests have 2-7× higher TTFT (profiling data)
//
// Expected values (iter15, from iter7):
//   - α₀: 1.32ms (iter7, physically plausible, constrained ≥0.5ms)
//   - α₁: 118μs per token (iter7, physically plausible, constrained ≥50μs)
//   - β₆: 13.2ms per request (iter7, may increase to 40-100ms with profiling evidence)
func (m *EvolvedModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	// Alpha coefficients (in seconds, convert to microseconds)
	totalProcessingTime += m.Alpha[0] * 1e6                                  // Fixed overhead (α₀)
	totalProcessingTime += m.Alpha[1] * float64(len(req.InputTokens)) * 1e6 // Tokenization (α₁)

	// Beta[6] scheduler overhead (in seconds, convert to microseconds)
	totalProcessingTime += m.Beta[6] * 1e6 // Scheduler overhead (β₆ at index 6)

	return clampToInt64(totalProcessingTime)
}

// OutputTokenProcessingTime returns per-output-token post-processing overhead.
// **DO NOT MODIFY THIS METHOD.** Standard implementation: α₂ (streaming detokenization)
//
// Physics:
//   - α₂: Per-output-token detokenization + output formatting in streaming mode
//   - Applied per output token during decode phase
//
// Expected value (iter15, from iter7): 90.5μs per token (constrained ≥40μs)
func (m *EvolvedModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.Alpha[2] * 1e6)
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// **DO NOT MODIFY THIS METHOD.** Return 0 unless systematic per-request bias observed.
//
// Physics:
//   - Models constant overhead at request completion (e.g., response finalization, logging)
//   - Iteration 0-15: No systematic per-request bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 15: 3 alpha, 10 beta - added β₈, β₉)
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 10 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 10 elements for iteration 15, got %d (expected β₀-β₉)", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs[:10], // Use first 10 beta coefficients (iteration 15: β₀-β₉)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
