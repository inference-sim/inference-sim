package latency

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// EvolvedModel implements physics-informed latency model with learned efficiency factors.
// Iteration 0: Scaled roofline + request overheads.
//
// Hypothesis: Roofline model with learned MFU corrections (β₀, β₁) captures first-order
// vLLM latency behavior. Request-level overheads (α₀, α₁, α₂) capture API processing,
// tokenization, and detokenization costs.
//
// Basis functions (StepTime):
//   - β₀ × prefill_compute_time: Prefill FLOPs / (peak_TFLOPS × MFU_prefill)
//   - β₁ × decode_compute_time: Decode FLOPs / (peak_TFLOPS × MFU_decode)
//   - β₂ × constant: Fixed vLLM scheduler overhead per step (microseconds)
//
// Alpha coefficients (request-level):
//   - α₀: Fixed API processing overhead (microseconds per request)
//   - α₁: Per-input-token tokenization (microseconds per token)
//   - α₂: Per-output-token detokenization (microseconds per token)
type EvolvedModel struct {
	Alpha       [3]float64 // [α₀, α₁, α₂] - request-level overheads
	Beta        []float64  // [β₀, β₁, β₂] - step-level coefficients
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
}

// StepTime computes vLLM step execution time using agent-designed basis functions.
//
// **THIS IS THE ONLY METHOD YOU CUSTOMIZE.** Design basis functions that capture
// compute, memory, and communication costs during batch execution.
//
// Iteration 0 basis functions:
//   - beta[0] × prefill_compute_time: Learns actual prefill efficiency vs theoretical MFU
//   - beta[1] × decode_compute_time: Learns actual decode efficiency vs theoretical MFU
//   - beta[2] × constant: Fixed scheduler overhead (microseconds)
//
// Physics grounding:
//   - Prefill is O(n²) attention, compute-bound, large GEMMs → high tensor core utilization
//   - Decode is O(n) attention, memory-bound, KV cache reads dominate → lower utilization
//   - Scheduler overhead: batch formation, KV block allocation, CUDA kernel launch
//
// Expected coefficients (iteration 0):
//   - β₀ ≈ 0.6-0.8 (prefill kernel efficiency 50-60%)
//   - β₁ ≈ 0.4-0.6 (decode bandwidth-limited, lower utilization)
//   - β₂ ≈ 50-200μs (vLLM scheduler overhead per step)
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

	// β₀ × prefill_compute_time
	// Physics: Prefill is compute-bound (large GEMMs, O(n²) attention)
	// FLOPs formula from calculateTransformerFlops (roofline.go:48-110)
	var prefillComputeTimeSeconds float64
	for _, req := range stepConfig.PrefillRequests {
		numTokens := int64(req.NumNewPrefillTokens)
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, numTokens, true, true)
		// Divide by (peak_TFLOPS × MFU_prefill) to get time in seconds
		prefillComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuPrefill)
	}
	prefillTimeUs := prefillComputeTimeSeconds * 1e6 // Convert to microseconds
	prefillContribution := m.Beta[0] * prefillTimeUs

	// β₁ × decode_compute_time
	// Physics: Decode is memory-bound (KV cache reads, O(n) attention)
	// Uses MfuDecode which reflects lower utilization in memory-bound regime
	var decodeComputeTimeSeconds float64
	for _, req := range stepConfig.DecodeRequests {
		f := calculateTransformerFlops(m.modelConfig, req.ProgressIndex, 1, true, true)
		decodeComputeTimeSeconds += f["total"] / tpFactor / (peakFlops * m.hwConfig.MfuDecode)
	}
	decodeTimeUs := decodeComputeTimeSeconds * 1e6 // Convert to microseconds
	decodeContribution := m.Beta[1] * decodeTimeUs

	// β₂ × constant
	// Physics: Fixed vLLM scheduler overhead (batch formation, KV block allocation, kernel launch)
	// Expected range: 50-200μs based on vLLM profiling
	schedulerOverhead := m.Beta[2]

	// Total step time
	totalTimeUs := prefillContribution + decodeContribution + schedulerOverhead

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
	totalProcessingTime += m.Alpha[0]                             // Fixed overhead (μs)
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
//   - Iteration 0: No systematic bias observed in training data → return 0
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
	if len(coeffs.BetaCoeffs) < 3 {
		return nil, fmt.Errorf("evolved model: BetaCoeffs requires at least 3 elements, got %d", len(coeffs.BetaCoeffs))
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
		Beta:        coeffs.BetaCoeffs, // Use all beta coefficients (iteration may evolve this)
		modelConfig: hw.ModelConfig,
		hwConfig:    hw.HWConfig,
		tp:          hw.TP,
	}, nil
}
