package sim

import (
	"testing"
)

// TestBlackboxLatencyModel_StepTime_PrefillAndDecode verifies BC-1:
// StepTime produces beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
func TestBlackboxLatencyModel_StepTime_PrefillAndDecode(t *testing.T) {
	// GIVEN a blackbox model with known beta coefficients
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	// AND a batch with 1 prefill request (30 new tokens) and 1 decode request
	batch := []*Request{
		{
			InputTokens:   make([]int, 100),
			ProgressIndex: 50, // < len(InputTokens), so prefill
			NumNewTokens:  30,
		},
		{
			InputTokens:   make([]int, 50),
			OutputTokens:  make([]int, 20),
			ProgressIndex: 60, // >= len(InputTokens), so decode
			NumNewTokens:  1,
		},
	}

	// WHEN StepTime is called
	result := model.StepTime(batch)

	// THEN result = beta0 + beta1*30 + beta2*1 = 1000 + 300 + 5 = 1305
	expected := int64(1305)
	if result != expected {
		t.Errorf("StepTime = %d, want %d", result, expected)
	}
}

// TestBlackboxLatencyModel_StepTime_EmptyBatch verifies StepTime with no requests.
func TestBlackboxLatencyModel_StepTime_EmptyBatch(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	// WHEN StepTime is called with an empty batch
	result := model.StepTime([]*Request{})

	// THEN result = beta0 only = 1000
	if result != 1000 {
		t.Errorf("StepTime(empty) = %d, want 1000", result)
	}
}

// TestBlackboxLatencyModel_QueueingTime verifies BC-3:
// QueueingTime = alpha0 + alpha1 * len(InputTokens).
func TestBlackboxLatencyModel_QueueingTime(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	req := &Request{InputTokens: make([]int, 50)}

	// WHEN QueueingTime is called
	result := model.QueueingTime(req)

	// THEN result = alpha0 + alpha1*50 = 100 + 50 = 150
	if result != 150 {
		t.Errorf("QueueingTime = %d, want 150", result)
	}
}

// TestBlackboxLatencyModel_OutputTokenProcessingTime verifies the alpha2 overhead.
func TestBlackboxLatencyModel_OutputTokenProcessingTime(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 200},
	}

	result := model.OutputTokenProcessingTime()

	if result != 200 {
		t.Errorf("OutputTokenProcessingTime = %d, want 200", result)
	}
}

// TestBlackboxLatencyModel_PlaceholderOverheads verifies placeholders return 0.
func TestBlackboxLatencyModel_PlaceholderOverheads(t *testing.T) {
	model := &BlackboxLatencyModel{
		betaCoeffs:  []float64{1000, 10, 5},
		alphaCoeffs: []float64{100, 1, 100},
	}

	if model.SchedulingProcessingTime() != 0 {
		t.Errorf("SchedulingProcessingTime = %d, want 0", model.SchedulingProcessingTime())
	}
	if model.PreemptionProcessingTime() != 0 {
		t.Errorf("PreemptionProcessingTime = %d, want 0", model.PreemptionProcessingTime())
	}
}
