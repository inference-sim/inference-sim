// Moved from inference-sim/sim/engine/helpers_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package engine

// Test helpers the engine's tests share with package sim's (each package keeps its own copy).

// fixedStepModel is a test-only LatencyModel stub that returns a fixed step time.
type fixedStepModel struct {
	stepTime int64
}

func (m *fixedStepModel) StepTime(batch []*Request) int64  { return m.stepTime }
func (m *fixedStepModel) QueueingTime(req *Request) int64  { return 0 }
func (m *fixedStepModel) OutputTokenProcessingTime() int64 { return 0 }
func (m *fixedStepModel) PostDecodeFixedOverhead() int64   { return 0 }

func float64Ptr(v float64) *float64 { return &v }
