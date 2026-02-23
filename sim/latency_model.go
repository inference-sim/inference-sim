package sim

import "fmt"

// LatencyModel estimates execution times for the DES step loop.
// Two implementations exist: BlackboxLatencyModel (alpha/beta regression)
// and RooflineLatencyModel (analytical FLOPs/bandwidth).
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

// RooflineLatencyModel estimates latency using analytical FLOPs/bandwidth roofline model.
// Step time is computed via rooflineStepTime(); overhead estimates use alpha coefficients.
type RooflineLatencyModel struct {
	modelConfig ModelConfig
	hwConfig    HardwareCalib
	tp          int
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
		} else {
			stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, DecodeRequestConfig{
				ProgressIndex:      req.ProgressIndex,
				NumNewDecodeTokens: req.NumNewTokens,
			})
		}
	}
	return rooflineStepTime(m.modelConfig, m.hwConfig, stepConfig, m.tp)
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

// NewLatencyModel creates the appropriate LatencyModel based on SimConfig.
// Returns RooflineLatencyModel if cfg.Roofline is true, BlackboxLatencyModel otherwise.
// Returns error if coefficient slices are too short or roofline config validation fails.
func NewLatencyModel(cfg SimConfig) (LatencyModel, error) {
	// Both implementations index alphaCoeffs[0..2]; validate upfront.
	if len(cfg.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: AlphaCoeffs requires at least 3 elements, got %d", len(cfg.AlphaCoeffs))
	}
	if cfg.Roofline {
		if cfg.TP <= 0 {
			return nil, fmt.Errorf("latency model: roofline requires TP > 0, got %d", cfg.TP)
		}
		if err := ValidateRooflineConfig(cfg.ModelConfig, cfg.HWConfig); err != nil {
			return nil, fmt.Errorf("latency model: %w", err)
		}
		return &RooflineLatencyModel{
			modelConfig: cfg.ModelConfig,
			hwConfig:    cfg.HWConfig,
			tp:          cfg.TP,
			alphaCoeffs: cfg.AlphaCoeffs,
		}, nil
	}
	// BlackboxLatencyModel indexes betaCoeffs[0..2]; validate upfront.
	if len(cfg.BetaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: BetaCoeffs requires at least 3 elements, got %d", len(cfg.BetaCoeffs))
	}
	return &BlackboxLatencyModel{
		betaCoeffs:  cfg.BetaCoeffs,
		alphaCoeffs: cfg.AlphaCoeffs,
	}, nil
}
