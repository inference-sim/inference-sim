// Package latency provides latency model implementations for the BLIS simulator.
// The LatencyModel interface is defined in sim/ (parent package).
// This package provides RooflineLatencyModel (analytical FLOPs/bandwidth) and
// TrainedPhysicsModel (physics-informed basis functions with architecture-aware MoE scaling).
package latency

import (
	"fmt"
	"math"
	"strings"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// clampToInt64 converts a float64 to int64, clamping values that would cause
// undefined behavior in Go's float64→int64 conversion. Specifically:
//   - NaN → math.MaxInt64 (NaN comparisons are always false in IEEE 754)
//   - Values >= float64(math.MaxInt64) → math.MaxInt64 (float64 rounds MaxInt64 up by 1)
func clampToInt64(v float64) int64 {
	// NaN must be checked explicitly: NaN > X and NaN >= X are both false.
	// float64(math.MaxInt64) rounds to 9223372036854775808.0 (MaxInt64+1),
	// so >= catches the exact boundary that would overflow int64().
	if math.IsNaN(v) || v >= float64(math.MaxInt64) {
		return math.MaxInt64
	}
	return int64(v)
}

// RooflineLatencyModel estimates latency using analytical FLOPs/bandwidth roofline model.
// Step time is computed via rooflineStepTime(); overhead estimates use alpha coefficients.
type RooflineLatencyModel struct {
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
	alphaCoeffs []float64
}

func (m *RooflineLatencyModel) StepTime(batch []*sim.Request) int64 {
	stepConfig := StepConfig{
		PrefillRequests: make([]PrefillRequestConfig, 0, len(batch)),
		DecodeRequests:  make([]DecodeRequestConfig, 0, len(batch)),
	}
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
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
	return max(1, rooflineStepTime(m.modelConfig, m.hwConfig, stepConfig, m.tp))
}

func (m *RooflineLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return clampToInt64(totalProcessingTime)
}

func (m *RooflineLatencyModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.alphaCoeffs[2])
}

func (m *RooflineLatencyModel) PostDecodeFixedOverhead() int64 { return 0 }

// validateCoeffs checks for NaN, Inf, or negative values in a coefficient slice.
func validateCoeffs(name string, coeffs []float64) error {
	for i, c := range coeffs {
		if math.IsNaN(c) {
			return fmt.Errorf("latency model: %s[%d] is NaN", name, i)
		}
		if math.IsInf(c, 0) {
			return fmt.Errorf("latency model: %s[%d] is Inf", name, i)
		}
		if c < 0 {
			return fmt.Errorf("latency model: %s[%d] must be non-negative, got %f", name, i, c)
		}
	}
	return nil
}

// NewLatencyModel creates the appropriate LatencyModel based on config.
// Dispatches on hw.Backend: "" or "roofline" → RooflineLatencyModel,
// "trained-physics" → TrainedPhysicsModel.
// Returns error if coefficient slices are too short, contain NaN/Inf, or config validation fails.
func NewLatencyModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig) (sim.LatencyModel, error) {
	// All implementations index alphaCoeffs[0..2]; validate upfront.
	if len(coeffs.AlphaCoeffs) < 3 {
		return nil, fmt.Errorf("latency model: AlphaCoeffs requires at least 3 elements, got %d", len(coeffs.AlphaCoeffs))
	}
	if err := validateCoeffs("AlphaCoeffs", coeffs.AlphaCoeffs); err != nil {
		return nil, err
	}
	switch hw.Backend {
	case "", "roofline":
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: roofline requires TP > 0, got %d", hw.TP)
		}
		if err := ValidateRooflineConfig(hw.ModelConfig, hw.HWConfig); err != nil {
			return nil, fmt.Errorf("latency model: %w", err)
		}
		return &RooflineLatencyModel{
			modelConfig: hw.ModelConfig,
			hwConfig:    hw.HWConfig,
			tp:          hw.TP,
			alphaCoeffs: coeffs.AlphaCoeffs,
		}, nil
	case "trained-physics":
		// TrainedPhysicsModel: physics-informed roofline with architecture-aware MoE overhead.
		// Uses roofline basis functions with learned corrections and conditional β₈ scaling.
		// Trained coefficients from iteration 29 (loss: 34.57%).
		model, err := NewTrainedPhysicsModel(coeffs, hw)
		if err != nil {
			return nil, err
		}
		return model, nil
	default:
		return nil, fmt.Errorf("latency model: unknown backend %q; valid options: %s",
			hw.Backend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
	}
}
