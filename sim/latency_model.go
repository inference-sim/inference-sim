package sim

import (
	"fmt"
	"math"
)

// LatencyModel estimates execution times for the DES step loop.
// Two implementations exist: BlackboxLatencyModel (alpha/beta regression)
// and RooflineLatencyModel (analytical FLOPs/bandwidth with MFU lookups).
// All time estimates are in microseconds (ticks).
type LatencyModel interface {
	// StepTime estimates the duration of one batch step given the running batch.
	// Precondition: each request in batch has NumNewTokens set by BatchFormation.FormBatch().
	StepTime(batch []*Request) int64

	// QueueingTime estimates the arrival-to-queue delay for a request.
	QueueingTime(req *Request) int64

	// OutputTokenProcessingTime estimates per-token post-processing time.
	OutputTokenProcessingTime() int64

	// SchedulingProcessingTime estimates scheduling overhead per request.
	SchedulingProcessingTime() int64

	// PreemptionProcessingTime estimates preemption overhead per eviction.
	PreemptionProcessingTime() int64
}

// BlackboxLatencyModel estimates latency using trained alpha/beta regression coefficients.
// Beta coefficients estimate step time: beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
// Alpha coefficients estimate overheads: alpha0 + alpha1*inputLen (queueing), alpha2 (output processing).
type BlackboxLatencyModel struct {
	betaCoeffs  []float64
	alphaCoeffs []float64
}

func (m *BlackboxLatencyModel) StepTime(batch []*Request) int64 {
	var totalCacheMissTokens, totalDecodeTokens int64
	for _, req := range batch {
		if req.ProgressIndex < Len64(req.InputTokens) {
			// Prefill phase: NumNewTokens are cache-miss tokens
			totalCacheMissTokens += int64(req.NumNewTokens)
		} else if len(req.OutputTokens) > 0 {
			// Decode phase: NumNewTokens is 1 (set by FormBatch)
			totalDecodeTokens += int64(req.NumNewTokens)
		}
	}
	var totalStepTime float64
	totalStepTime += m.betaCoeffs[0]
	totalStepTime += m.betaCoeffs[1] * float64(totalCacheMissTokens)
	totalStepTime += m.betaCoeffs[2] * float64(totalDecodeTokens)
	return int64(totalStepTime)
}

func (m *BlackboxLatencyModel) QueueingTime(req *Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *BlackboxLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *BlackboxLatencyModel) SchedulingProcessingTime() int64 {
	return 0
}

func (m *BlackboxLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}

// RooflineLatencyModel estimates latency using MFU-based roofline model.
// Uses MFU database for per-GEMM and per-attention MFU lookups for improved accuracy.
type RooflineLatencyModel struct {
	modelConfig ModelConfig
	hwConfig    HardwareCalib
	tp          int
	mfuDB       *MFUDatabase
	gpu         string
	alphaCoeffs []float64
}

func (m *RooflineLatencyModel) StepTime(batch []*Request) int64 {
	stepConfig := StepConfig{
		PrefillRequests: make([]PrefillRequestConfig, 0, len(batch)),
		DecodeRequests:  make([]DecodeRequestConfig, 0, len(batch)),
	}
	for _, req := range batch {
		if req.ProgressIndex < Len64(req.InputTokens) {
			stepConfig.PrefillRequests = append(stepConfig.PrefillRequests, PrefillRequestConfig{
				ProgressIndex:       req.ProgressIndex,
				NumNewPrefillTokens: req.NumNewTokens,
			})
		} else if len(req.OutputTokens) > 0 {
			stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, DecodeRequestConfig{
				ProgressIndex:      req.ProgressIndex,
				NumNewDecodeTokens: req.NumNewTokens,
			})
		}
	}
	return rooflineStepTime(m.gpu, m.modelConfig, m.hwConfig, stepConfig, m.tp, m.mfuDB)
}

func (m *RooflineLatencyModel) QueueingTime(req *Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return int64(totalProcessingTime)
}

func (m *RooflineLatencyModel) OutputTokenProcessingTime() int64 {
	return int64(m.alphaCoeffs[2])
}

func (m *RooflineLatencyModel) SchedulingProcessingTime() int64 {
	return 0
}

func (m *RooflineLatencyModel) PreemptionProcessingTime() int64 {
	return 0
}

// validateCoeffs checks for NaN or Inf in a coefficient slice.
func validateCoeffs(name string, coeffs []float64) error {
	for i, c := range coeffs {
		if math.IsNaN(c) {
			return fmt.Errorf("latency model: %s[%d] is NaN", name, i)
		}
		if math.IsInf(c, 0) {
			return fmt.Errorf("latency model: %s[%d] is Inf", name, i)
		}
	}
	return nil
}

// NewLatencyModel creates the appropriate LatencyModel based on config.
// Returns RooflineLatencyModel if hw.Roofline is true (requires MFUDatabase),
// BlackboxLatencyModel otherwise (alpha/beta regression).
// Returns error if coefficient slices are too short, contain NaN/Inf, or roofline config validation fails.
func NewLatencyModel(coeffs LatencyCoeffs, hw ModelHardwareConfig) (LatencyModel, error) {
	// Both implementations index alphaCoeffs[0..2]; validate upfront.
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if err := validateCoeffs("AlphaCoeffs", coeffs.AlphaCoeffs); err != nil {
		return nil, err
	}
	if hw.Roofline {
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: roofline requires TP > 0, got %d", hw.TP)
		}
		if err := ValidateRooflineConfig(hw.ModelConfig, hw.HWConfig); err != nil {
			return nil, fmt.Errorf("latency model: %w", err)
		}
		if hw.MFUDatabase == nil {
			return nil, fmt.Errorf("latency model: roofline requires MFU database (--bench-data-path)")
		}
		return &RooflineLatencyModel{
			modelConfig: hw.ModelConfig,
			hwConfig:    hw.HWConfig,
			tp:          hw.TP,
			mfuDB:       hw.MFUDatabase,
			gpu:         hw.GPU,
			alphaCoeffs: coeffs.AlphaCoeffs,
		}, nil
	}
	// BlackboxLatencyModel indexes betaCoeffs[0..2]; validate upfront.
	if len(coeffs.BetaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: BetaCoeffs requires at least 3 elements, got %d", len(coeffs.BetaCoeffs))
	}
	if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
		return nil, err
	}
	return &BlackboxLatencyModel{
		betaCoeffs:  coeffs.BetaCoeffs,
		alphaCoeffs: coeffs.AlphaCoeffs,
	}, nil
}
