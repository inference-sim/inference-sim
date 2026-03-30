package latency

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 7: Clean data retraining with decode overhead decoupling.
//
// Critical discovery from iter6 post-analysis: Original reasoning experiments were from
// overloaded servers (85% failure rate, 259s timeout). Journey trace analysis revealed
// 97-99% of training data consisted of requests stuck in queue for 259 seconds before
// timeout—no physics-based model can fit this. Fresh reasoning-lite data collected on
// 2026-03-30 shows roofline baseline improved from 99% → 53% avg TTFT error (range: 15-92%),
// confirming data quality issue rather than model deficiency.
//
// Iter7 strategy: Retrain on clean dataset (exclude 3 corrupted reasoning experiments,
// include 3 fresh reasoning-lite experiments, total still 15) while addressing decode
// coefficient destabilization observed in iter6.
//
// Changes from iter6:
//   - **Training data**: Exclude 3 corrupted reasoning experiments (99% TTFT), use 3
//     reasoning-lite experiments (roofline 15-92% TTFT)
//   - **Added β₇**: Decode per-request overhead (5-15ms) in StepTime to decouple framework
//     overhead from compute/memory efficiency (β₁/β₄)
//   - **Alpha reversion**: Warm-start from iter4 Alpha (not iter6) with tight bounds to
//     prevent inflation now that corrupt data removed
//   - **Beta reversion**: β₁/β₄ warm-start from iter3 (1.037, 0.796) to restore stability;
//     other Beta from iter6 (stable)
//   - **Expected**: Overall loss 161.69% → <80%, TTFT RMSE 69.47% → <40%, E2E RMSE 92.22% → <50%
//
// Basis functions:
//   - StepTime (7 beta terms: β₀-β₆, note β₆ MOVED to QueueingTime, β₇ NEW in StepTime):
//     Prefill/decode compute, memory, communication, KV management, MoE gating, decode overhead
//   - QueueingTime (Alpha + β₆): API overhead (α₀, α₁) + scheduler overhead (β₆)
//
// Beta coefficients (8 total):
//   - β₀: Prefill compute MFU scaling (dimensionless, expected 0.15-0.25)
//   - β₁: Decode memory-bound MFU (dimensionless, expected 1.00-1.15)
//   - β₂: TP decode communication scaling (dimensionless, expected 0.30-0.35)
//   - β₃: KV cache management overhead (ms per request, expected 0.4-0.5ms)
//   - β₄: Decode compute-bound MFU (dimensionless, expected 0.75-0.90, constrained ≤1.0)
//   - β₅: MoE gating overhead (ms, expected 0.01-0.012ms = 10-12μs)
//   - β₆: Scheduler overhead per request (ms, expected 50-150ms) - used in QueueingTime, NOT StepTime
//   - β₇: **NEW** Decode per-request overhead (ms, expected 5-15ms) - used in StepTime for decode-phase overhead
//
// Alpha coefficients (request-level, reverted to iter4 with tight bounds):
//   - α₀: Fixed API processing overhead (ms per request, 1.5ms target)
//   - α₁: Per-input-token tokenization (ms per token, 125μs target, bounds [0.0, 0.0002] to prevent inflation)
//   - α₂: Per-output-token detokenization (ms per token, 36μs target, bounds [0.0, 0.0001] to prevent inflation)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂, β₃, β₄, β₅, β₆, β₇] - β₀-β₅,β₇ for StepTime, β₆ for QueueingTime
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, communication, and overhead costs during batch execution.
//
// Iteration 7 basis functions (7 terms in StepTime, β₆ in QueueingTime):
//   - beta[0] × prefill_compute_time: Prefill efficiency vs theoretical MFU
//   - beta[1] × decode_memory_time × memory_weight(batch_size): Decode small-batch memory-bound
//   - beta[2] × tp_comm_time: TP communication overhead (decode all-reduce)
//   - beta[3] × kv_mgmt_time: KV cache management per request
//   - beta[4] × decode_compute_time × compute_weight(batch_size): Decode large-batch compute-bound
//   - beta[5] × moe_gating_time: MoE gating network overhead
//   - beta[7] × num_decode_requests: **NEW** Decode per-request overhead (output processing, TP coordination)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode small-batch is O(n) attention, memory-bound, KV cache reads dominate
//   - Decode large-batch becomes compute-bound due to tensor core utilization
//   - Smooth transition via sigmoid interpolation: memory_weight(n) = 1/(1+exp((n-8)/2))
//   - TP communication (decode): all-reduce after each layer when TP > 1
//   - KV management: PagedAttention block allocation/deallocation per request
//   - MoE gating: routing probability computation for all experts
//   - Decode overhead (NEW): Output processing, TP coordination, result aggregation per decode request
//
// Expected coefficients (iteration 7):
//   - β₀ ≈ 0.15-0.25 (prefill MFU, from iter6's 0.164)
//   - β₁ ≈ 1.00-1.15 (decode memory-bound, revert to iter3's 1.037 range)
//   - β₂ ≈ 0.30-0.35 (TP communication, from iter6's stable 0.270)
//   - β₃ ≈ 0.4-0.5ms per request (KV block allocation, from iter6's recovered 0.620ms)
//   - β₄ ≈ 0.75-0.90 (decode compute-bound, revert to iter3's 0.796, constrained ≤1.0)
//   - β₅ ≈ 10-12μs (MoE gating, from iter6's improving 4.31ms)
//   - β₆ ≈ 50-150ms (scheduler overhead per request, from iter6's 21.5ms, used in QueueingTime)
//   - β₇ ≈ 5-15ms (decode overhead per request, NEW in iter7)
//
// Beta[6] is NOT used in StepTime (moved to QueueingTime in iter6).
// Beta[7] is NEW in iter7 (decode per-request overhead in StepTime).
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
	// β₇ × decode_per_request_overhead (NEW in iter7)
	// ========================================
	// Physics: vLLM decode phase has fixed overhead per request beyond memory/compute:
	//   1. Output processing: After each decode step, vLLM processes output tokens
	//      (sampling, stop condition check, streaming updates)
	//   2. TP coordination: Decode requires per-request coordination across TP workers
	//      (synchronization barriers per step)
	//   3. KV cache write-back: Updated KV cache blocks written back to memory
	//      (similar to β₃ for prefill, but during decode phase)
	// Expected range: 5-15ms per decode request
	// Units: milliseconds per request
	// Code: vllm/model_executor/model_loader.py:_run_workers() calls execute_model() per step
	//       vllm/worker/worker.py:execute_model() synchronizes across TP ranks per step
	//
	// This term decouples decode framework overhead from compute/memory efficiency (β₁/β₄).
	// Iter6 destabilization: β₁ = 1.851, β₄ = 1.451 (both increased to compensate for missing
	// fixed overhead). Adding β₇ should stabilize β₁ → 1.00-1.15, β₄ → 0.75-0.90.
	numDecodeRequests := len(stepConfig.DecodeRequests)
	decodeOverheadTimeMs := float64(numDecodeRequests) // Number of decode requests
	decodeOverheadTimeUs := decodeOverheadTimeMs * 1000.0 // Convert to microseconds (β₇ is in ms)
	decodeOverheadContribution := m.Beta[7] * decodeOverheadTimeUs

	// ========================================
	// Total step time (additive model)
	// ========================================
	// Iteration 7: 7 terms in StepTime (β₀-β₅, β₇), β₆ moved to QueueingTime
	totalTimeUs := prefillContribution + decodeMemoryContribution +
		tpCommContribution + kvMgmtContribution + decodeComputeContribution +
		moeGatingContribution + decodeOverheadContribution

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
	// Validate coefficient counts (iteration 7: 3 alpha, 8 beta)
	// Beta count increased from iter6 (7 → 8), added β₇ for decode per-request overhead
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if len(coeffs.BetaCoeffs) < 8 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 8 elements for iteration 7, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration 7 expects 8: β₀-β₇)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
