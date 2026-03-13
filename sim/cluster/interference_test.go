package cluster

import (
	"fmt"
	"math"
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// stubLatencyModel returns fixed values for testing the interference wrapper.
type stubLatencyModel struct {
	stepTime                int64
	queueingTime            int64
	outputTokenProcessing   int64
	postDecodeFixedOverhead int64
}

func (s *stubLatencyModel) StepTime(_ []*sim.Request) int64         { return s.stepTime }
func (s *stubLatencyModel) QueueingTime(_ *sim.Request) int64       { return s.queueingTime }
func (s *stubLatencyModel) OutputTokenProcessingTime() int64        { return s.outputTokenProcessing }
func (s *stubLatencyModel) PostDecodeFixedOverhead() int64          { return s.postDecodeFixedOverhead }

// makeBatch creates a batch of requests with the given prefill/decode composition.
// Prefill requests have ProgressIndex=0 with len(InputTokens)=10.
// Decode requests have ProgressIndex=10 with len(InputTokens)=10 (past prefill).
func makeBatch(prefillCount, decodeCount int) []*sim.Request {
	batch := make([]*sim.Request, 0, prefillCount+decodeCount)
	for i := 0; i < prefillCount; i++ {
		batch = append(batch, &sim.Request{
			ID:            fmt.Sprintf("prefill_%d", i),
			InputTokens:   make([]int, 10),
			ProgressIndex: 0,
		})
	}
	for i := 0; i < decodeCount; i++ {
		batch = append(batch, &sim.Request{
			ID:            fmt.Sprintf("decode_%d", i),
			InputTokens:   make([]int, 10),
			ProgressIndex: 10,
		})
	}
	return batch
}

func TestNewInterferenceLatencyModel_Validation(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	tests := []struct {
		name          string
		inner         sim.LatencyModel
		prefillFactor float64
		decodeFactor  float64
		wantErr       bool
	}{
		{name: "valid zero factors", inner: inner, prefillFactor: 0, decodeFactor: 0},
		{name: "valid positive factors", inner: inner, prefillFactor: 0.5, decodeFactor: 0.3},
		{name: "nil inner", inner: nil, prefillFactor: 0, decodeFactor: 0, wantErr: true},
		{name: "negative prefill factor", inner: inner, prefillFactor: -0.1, decodeFactor: 0, wantErr: true},
		{name: "negative decode factor", inner: inner, prefillFactor: 0, decodeFactor: -0.1, wantErr: true},
		{name: "NaN prefill factor", inner: inner, prefillFactor: math.NaN(), decodeFactor: 0, wantErr: true},
		{name: "Inf decode factor", inner: inner, prefillFactor: 0, decodeFactor: math.Inf(1), wantErr: true},
		{name: "NaN decode factor", inner: inner, prefillFactor: 0, decodeFactor: math.NaN(), wantErr: true},
		{name: "negative Inf prefill", inner: inner, prefillFactor: math.Inf(-1), decodeFactor: 0, wantErr: true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := NewInterferenceLatencyModel(tc.inner, tc.prefillFactor, tc.decodeFactor)
			if (err != nil) != tc.wantErr {
				t.Errorf("NewInterferenceLatencyModel() error = %v, wantErr %v", err, tc.wantErr)
			}
		})
	}
}

func TestInterferenceLatencyModel_StepTime(t *testing.T) {
	const baseStepTime int64 = 1000

	tests := []struct {
		name           string
		prefillFactor  float64
		decodeFactor   float64
		prefillCount   int
		decodeCount    int
		wantMultiplier float64
	}{
		{name: "zero factors mixed batch", prefillFactor: 0, decodeFactor: 0, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.0},
		{name: "all prefill", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 4, decodeCount: 0, wantMultiplier: 1.0},
		{name: "all decode", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 0, decodeCount: 4, wantMultiplier: 1.0},
		{name: "empty batch", prefillFactor: 0.5, decodeFactor: 0.5, prefillCount: 0, decodeCount: 0, wantMultiplier: 1.0},
		{name: "prefill majority", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 3, decodeCount: 1, wantMultiplier: 1.125},
		{name: "decode majority", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 1, decodeCount: 3, wantMultiplier: 1.075},
		{name: "tied batch uses max factor", prefillFactor: 0.5, decodeFactor: 0.3, prefillCount: 2, decodeCount: 2, wantMultiplier: 1.25},
		{name: "single prefill", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 1, decodeCount: 0, wantMultiplier: 1.0},
		{name: "single decode", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 0, decodeCount: 1, wantMultiplier: 1.0},
		{name: "even split factor 1.0", prefillFactor: 1.0, decodeFactor: 1.0, prefillCount: 5, decodeCount: 5, wantMultiplier: 1.5},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			inner := &stubLatencyModel{stepTime: baseStepTime}
			model, err := NewInterferenceLatencyModel(inner, tc.prefillFactor, tc.decodeFactor)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			batch := makeBatch(tc.prefillCount, tc.decodeCount)
			got := model.StepTime(batch)
			want := int64(math.Round(float64(baseStepTime) * tc.wantMultiplier))
			if got != want {
				t.Errorf("StepTime() = %d, want %d (multiplier %.4f)", got, want, tc.wantMultiplier)
			}
			if m := model.LastAppliedMultiplier(); math.Abs(m-tc.wantMultiplier) > 1e-9 {
				t.Errorf("LastAppliedMultiplier() = %f, want %f", m, tc.wantMultiplier)
			}
		})
	}
}

func TestInterferenceLatencyModel_INV_P2_3_MultiplierMonotonicity(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	factors := []float64{0, 0.1, 0.5, 1.0, 2.0, 5.0}
	compositions := [][2]int{
		{0, 0}, {1, 0}, {0, 1}, {1, 1},
		{3, 1}, {1, 3}, {5, 5}, {10, 1}, {1, 10},
	}
	for _, pf := range factors {
		for _, df := range factors {
			model, err := NewInterferenceLatencyModel(inner, pf, df)
			if err != nil {
				t.Fatalf("factor (%f, %f): %v", pf, df, err)
			}
			for _, comp := range compositions {
				batch := makeBatch(comp[0], comp[1])
				got := model.StepTime(batch)
				if got < inner.stepTime {
					t.Errorf("INV-P2-3 violated: factors=(%f,%f) comp=(%d,%d) StepTime=%d < base=%d",
						pf, df, comp[0], comp[1], got, inner.stepTime)
				}
				if model.LastAppliedMultiplier() < 1.0 {
					t.Errorf("INV-P2-3 violated: factors=(%f,%f) comp=(%d,%d) multiplier=%f < 1.0",
						pf, df, comp[0], comp[1], model.LastAppliedMultiplier())
				}
			}
		}
	}
}

func TestInterferenceLatencyModel_PassThrough(t *testing.T) {
	inner := &stubLatencyModel{
		stepTime:                1000,
		queueingTime:            500,
		outputTokenProcessing:   200,
		postDecodeFixedOverhead: 100,
	}
	model, err := NewInterferenceLatencyModel(inner, 0.5, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	req := &sim.Request{InputTokens: make([]int, 5)}
	if got := model.QueueingTime(req); got != 500 {
		t.Errorf("QueueingTime() = %d, want 500", got)
	}
	if got := model.OutputTokenProcessingTime(); got != 200 {
		t.Errorf("OutputTokenProcessingTime() = %d, want 200", got)
	}
	if got := model.PostDecodeFixedOverhead(); got != 100 {
		t.Errorf("PostDecodeFixedOverhead() = %d, want 100", got)
	}
}

func TestInterferenceLatencyModel_LastAppliedMultiplier_InitialValue(t *testing.T) {
	inner := &stubLatencyModel{stepTime: 1000}
	model, err := NewInterferenceLatencyModel(inner, 0.5, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := model.LastAppliedMultiplier(); got != 1.0 {
		t.Errorf("initial LastAppliedMultiplier() = %f, want 1.0", got)
	}
}
