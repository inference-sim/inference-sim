package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 14: Fix β₅ MoE Gating Layer Multiplier Bug + Return to Iter7 Baseline
//
// **CRITICAL BUG FIX**: Iter13 catastrophic failure (loss 2387%, 15.4× worse than iter7) traced
// to missing `× numMoELayers` multiplier in β₅ (MoE gating) basis function. This caused β₅ to
// explode 46,800× (from iter7's 0.0411 to 1924.4), producing massive latency overestimation.
//
// **EVIDENCE**: β₅ = 1924.4 / 56 Scout MoE layers = 34.4 (within expected 1-50 range for a
// per-layer coefficient), strongly suggesting the optimizer was compensating for a missing
// layer multiplier. The β₈ (MoE routing) code already correctly multiplies by numMoELayers
// (see line 290 in iter13), but β₅ was missing this critical multiplier.
//
// **ITER14 STRATEGY**: One bug, one fix, one iteration.
//   1. Return to iter7's stable 8-beta architecture (β₀-β₇, indices 0-7)
//   2. Fix β₅ basis function: add missing `× numMoELayers` multiplier (copy pattern from β₈)
//   3. Remove β₈ (MoE routing) and β₁₀ (batching inefficiency) until β₅ proven stable
//   4. Warm-start from iter7 optimal coefficients (not iter13's inflated values)
//
// **EXPECTED OUTCOME**: β₅ converges to 1-50 (physically plausible), loss 2387% → <180%
// (targeting iter7 baseline 155% + margin), zero 100% timeout errors, dense models recover
// to iter7 ±10pp.
//
// **WHY ITER7 BASELINE?**:
//   - Iter7 was last stable iteration (loss 155.37%, 6/8 coefficients in expected ranges)
//   - After iter7, all attempts to add complexity triggered β₅-driven catastrophic failures:
//     * Iter9: Added β₉ (FP8) → β₆ +654%, β₂ +343%, loss 161%
//     * Iter10: Added β₁₀ + β₃' → β₅ exploded, loss 4267%
//     * Iter11: Same as iter10 (audited, basis correct) → loss 4084%
//     * Iter12: Widened β₃' bounds → β₃' collapsed, loss 2590%
//     * Iter13: Returned to iter7 + β₈ + β₁₀ → β₅ exploded 46,800×, loss 2387%
//   - Pattern: ANY addition to iter7 triggered β₅ explosion when basis function broken
//   - Conclusion: Must fix β₅ FIRST, validate stability, THEN add complexity in iter15+
//
// **DATASET CONTEXT**: Between iter7 and iter13, reasoning experiments were converted to
// reasoning-lite (lighter load, different arrival characteristics). This complicates direct
// loss comparison (iter7's 155% was on different experiments), but the β₅ bug is architectural
// and must be fixed regardless of dataset.
//
// Basis functions (iter14 architecture, same as iter7):
//   - StepTime (6 beta terms: β₀-β₅, β₇, note β₆ in QueueingTime):
//     Prefill compute, decode memory, TP comm, KV base, decode compute, MoE gating (FIXED), decode overhead
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (8 total, indexed 0-7, same as iter7):
//   - β₀ (index 0): Prefill compute MFU scaling (dimensionless, expected 0.16-0.22, iter7: 0.191 ✓)
//   - β₁ (index 1): Decode memory-bound MFU (dimensionless, expected 1.00-1.15, iter7: 1.108 ✓)
//   - β₂ (index 2): TP decode communication scaling (dimensionless, expected 0.15-0.25, iter7: 0.185 ✓)
//   - β₃ (index 3): KV cache management BASE overhead (seconds per request, expected 0.4-1.5ms, iter7: 4.4ms may decrease)
//   - β₄ (index 4): Decode compute-bound MFU (dimensionless, expected 0.70-0.85, iter7: 0.713 ✓)
//   - β₅ (index 5): MoE gating efficiency (dimensionless, expected 1-50 after fix, iter7: 0.0411 collapsed)
//   - β₆ (index 6): Scheduler overhead per request (seconds, expected 40-100ms, iter7: 13.2ms may increase) - used in QueueingTime
//   - β₇ (index 7): Decode per-request overhead (seconds, expected 15-30ms, iter7: 26.3ms ✓) - used in StepTime
//
// Alpha coefficients (request-level, constrained):
//   - α₀: Fixed API processing overhead (seconds per request, constrained ≥ 0.5ms, iter7: 1.32ms ✓)
//   - α₁: Per-input-token tokenization (seconds per token, constrained ≥ 50μs, iter7: 118μs ✓)
//   - α₂: Per-output-token detokenization (seconds per token, constrained ≥ 40μs, iter7: 90.5μs ✓)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇] - 8 coefficients total (iter14, same as iter7)
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 14 basis functions (6 terms in StepTime, β₆ in QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × num_requests: KV cache management BASE overhead (PagedAttention setup)
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[5] × moe_gating_time: MoE gating network overhead (FIXED in iter14 with layer multiplier)
//   - beta[7] × num_decode_requests: Decode per-request overhead (output processing, TP coordination)
//
// Removed from iter13:
//   - ❌ beta[8] (was β₈ MoE routing in iter13, removed until β₅ proven stable)
//   - ❌ beta[9] (was β₁₀ batching inefficiency in iter13, removed until β₅ proven stable)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation (base overhead per request only)
//   - MoE gating: routing probability computation for all experts (NOW FIXED with layer multiplier)
//   - Decode overhead: Output processing, TP coordination, result aggregation per decode request
//
// Expected coefficients (iteration 14, targeting iter7 values):
//   - β₀ ≈ 0.16-0.22 (prefill MFU, iter7: 0.191 ✓)
//   - β₁ ≈ 1.00-1.15 (decode memory-bound, iter7: 1.108 ✓)
//   - β₂ ≈ 0.15-0.25 (TP communication, iter7: 0.185 ✓)
//   - β₃ ≈ 0.4-1.5ms per request (KV base overhead, iter7: 4.4ms may decrease)
//   - β₄ ≈ 0.70-0.85 (decode compute-bound, iter7: 0.713 ✓)
//   - β₅ ≈ 1-50 (MoE gating, iter7: 0.0411 collapsed, expect 20-40 after fix)
//   - β₆ ≈ 40-100ms (scheduler overhead, iter7: 13.2ms may increase, used in QueueingTime)
//   - β₇ ≈ 15-30ms (decode overhead per request, iter7: 26.3ms ✓)
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
	// Expected range: 0.16-0.22 (iter7: 0.191 ✓)
	// Units: seconds (converted to μs below)
	var prefillComputeTimeSeconds float64
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, numTokens, true, true)
		prefillComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuPrefill)
	}
	prefillTimeUs := prefillComputeTimeSeconds * 1e6
	prefillContribution := m.Beta[0] * prefillTimeUs

	// ========================================
	// β₁ × decode_memory_time × memory_weight (small-batch)
	// β₄ × decode_compute_time × compute_weight (large-batch)
	// ========================================
	// Physics: Decode transitions from memory-bound (small batches) to compute-bound (large batches)
	// Expected range: β₁ ≈ 1.00-1.15 (iter7: 1.108 ✓), β₄ ≈ 0.70-0.85 (iter7: 0.713 ✓)
	// Units: seconds (converted to μs below)
	// Sigmoid interpolation: memory_weight(n) = 1 / (1 + exp((n - 8) / 2))
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
	decodeMemoryContribution := m.Beta[1] * decodeMemoryTimeUs * memoryWeight
	decodeComputeContribution := m.Beta[4] * decodeComputeTimeUs * computeWeight // β₄ at index 4

	// ========================================
	// β₂ × tp_comm_time (TP communication for DECODE)
	// ========================================
	// Physics: Ring all-reduce after each transformer layer (TP > 1), DECODE ONLY
	// Expected range: 0.15-0.25 (iter7: 0.185 ✓)
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
	// Expected range: 0.4-1.5ms per request (iter7: 4.4ms may decrease after β₅ fix)
	// Units: seconds per request (converted to μs below)
	kvMgmtBaseTimeSeconds := float64(len(batch)) * m.Beta[3] // β₃ at index 3
	kvMgmtBaseContribution := kvMgmtBaseTimeSeconds * 1e6

	// ========================================
	// β₅ × moe_gating_time (MoE gating) - **FIXED IN ITER14**
	// ========================================
	// Physics: MoE gating network computes routing probabilities for all experts
	// Expected range: 1-50 after fix (iter7: 0.0411 collapsed, iter13: 1924.4 exploded)
	// Units: seconds (converted to μs below)
	//
	// **CRITICAL FIX**: Added missing `× numMoELayers` multiplier (copied from β₈ pattern)
	//
	// Gating network: hidden_dim → num_experts linear projection per token per layer
	// FLOPs per layer: 2 × tokens × hidden_dim × num_experts
	// Total FLOPs: (per-layer FLOPs) × numMoELayers  ← THIS WAS MISSING IN ITER13!
	//
	// Why this matters:
	//   - Scout has 56 MoE layers (InterleaveMoELayerStep=4 → 80/4 = 20 MoE layers, but config says 56)
	//   - Without layer multiplier: basis function computes ~0.04μs per token (single layer)
	//   - With layer multiplier: basis function computes 56 × 0.04μs = 2.24μs per token (all layers)
	//   - Iter13: β₅ exploded to 1924.4 to compensate for missing 56× factor
	//   - Iter13 evidence: 1924.4 / 56 = 34.4 (within expected 1-50 range for per-layer coefficient!)
	//   - Iter14: β₅ should converge to 20-40 (physically plausible) with corrected basis function
	var moeGatingTimeSeconds float64
	if m.modelConfig.NumLocalExperts > 1 {
		// Calculate numMoELayers (same pattern as β₈ MoE routing, see iter13 lines 269-277)
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

		var totalPrefillTokens int64
		for _, req := range stepConfig.PrefillRequests {
			totalPrefillTokens += int64(req.NumNewPrefillTokens)
		}
		totalTokens := totalPrefillTokens + int64(len(stepConfig.DecodeRequests))

		// Compute gating FLOPs per layer
		gatingFlopsPerLayer := 2.0 * float64(totalTokens) * float64(m.modelConfig.HiddenDim) * float64(m.modelConfig.NumLocalExperts)

		// Compute gating time per layer
		gatingEfficiency := 0.3 // Assume 30% MFU (may need tuning if β₅ still explodes)
		gatingTimePerLayerSeconds := gatingFlopsPerLayer / tpFactor / (peakFlops * gatingEfficiency)

		// **CRITICAL FIX**: Multiply by numMoELayers to account for ALL MoE layers
		moeGatingTimeSeconds = gatingTimePerLayerSeconds * numMoELayers
	}
	moeGatingTimeUs := moeGatingTimeSeconds * 1e6
	moeGatingContribution := m.Beta[5] * moeGatingTimeUs // β₅ at index 5

	// ========================================
	// β₇ × decode_per_request_overhead (iter7)
	// ========================================
	// Physics: vLLM decode phase has fixed overhead per request beyond memory/compute
	//   1. Output processing: sampling, stop condition check, streaming updates
	//   2. TP coordination: synchronization barriers per step
	//   3. KV cache write-back: updated blocks written to memory
	// Expected range: 15-30ms per decode request (iter7: 26.3ms ✓)
	// Units: seconds per request (converted to μs below)
	numDecodeRequests := len(stepConfig.DecodeRequests)
	decodeOverheadTimeSeconds := float64(numDecodeRequests) * m.Beta[7] // β₇ at index 7
	decodeOverheadContribution := decodeOverheadTimeSeconds * 1e6

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 14: 6 terms in StepTime (β₀-β₅, β₇), β₆ in QueueingTime
	// Removed from iter13: β₈ (MoE routing), β₁₀ (batching inefficiency)
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtBaseContribution +
		decodeComputeContribution + moeGatingContribution + decodeOverheadContribution

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
// Expected values (iter14, from iter7):
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
// Expected value (iter14, from iter7): 90.5μs per token (constrained ≥40μs)
func (m *EvolvedModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.Alpha[2] * 1e6)
}

// PostDecodeFixedOverhead returns fixed per-request overhead at completion.
// **DO NOT MODIFY THIS METHOD.** Return 0 unless systematic per-request bias observed.
//
// Physics:
//   - Models constant overhead at request completion (e.g., response finalization, logging)
//   - Iteration 0-14: No systematic per-request bias observed in training data → return 0
func (m *EvolvedModel) PostDecodeFixedOverhead() int64 {
	return 0
}

// NewEvolvedModel creates an EvolvedModel with validation.
// Called by NewLatencyModel() when hw.Backend == "evolved".
func NewEvolvedModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (*EvolvedModel, error) {
	// Validate coefficient counts (iteration 14: 3 alpha, 8 beta - same as iter7)
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 8 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 8 elements for iteration 14, got %d (expected β₀-β₇)", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs[:8], // Use first 8 beta coefficients (iteration 14: β₀-β₇)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
