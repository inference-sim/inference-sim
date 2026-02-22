package sim

// LatencyModel estimates execution times for the DES step loop.
// Two implementations exist: BlackboxLatencyModel (alpha/beta regression)
// and RooflineLatencyModel (analytical FLOPs/bandwidth).
// All time estimates are in microseconds (ticks).
type LatencyModel interface {
	// StepTime estimates the duration of one batch step given the running batch.
	// Precondition: each request in batch has NumNewTokens set by makeRunningBatch().
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
			// Decode phase: NumNewTokens is 1 (set by makeRunningBatch)
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
