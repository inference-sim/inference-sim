// Package latency provides latency model implementations for the BLIS simulator.
// The LatencyModel interface is defined in sim/ (parent package).
// This package provides BlackboxLatencyModel (alpha/beta regression),
// RooflineLatencyModel (analytical FLOPs/bandwidth),
// CrossModelLatencyModel (physics-informed cross-model step time),
// TrainedRooflineLatencyModel (roofline basis functions × learned corrections), and
// TrainedPhysicsModel (physics-informed basis functions with architecture-aware MoE scaling).
package latency

import (
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
	"github.com/sirupsen/logrus"
)

// Package-level sync.Once to emit deprecation warnings only once per process.
var (
	warnCrossmodelOnce  sync.Once
	warnTrainedRoofOnce sync.Once
	warnBlackboxOnce    sync.Once
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

// BlackboxLatencyModel estimates latency using trained alpha/beta regression coefficients.
// Beta coefficients estimate step time: beta0 + beta1*cacheMissTokens + beta2*decodeTokens.
// Alpha coefficients estimate overheads: alpha0 + alpha1*inputLen (queueing), alpha2 (output processing).
type BlackboxLatencyModel struct {
	betaCoeffs  []float64
	alphaCoeffs []float64
}

func (m *BlackboxLatencyModel) StepTime(batch []*sim.Request) int64 {
	var totalCacheMissTokens, totalDecodeTokens int64
	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
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
	return max(1, clampToInt64(totalStepTime))
}

func (m *BlackboxLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(len(req.InputTokens))
	return clampToInt64(totalProcessingTime)
}

func (m *BlackboxLatencyModel) OutputTokenProcessingTime() int64 {
	return clampToInt64(m.alphaCoeffs[2])
}

func (m *BlackboxLatencyModel) PostDecodeFixedOverhead() int64 { return 0 }

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
// Dispatches on hw.Backend: "" or "roofline" → RooflineLatencyModel, "crossmodel" → CrossModelLatencyModel,
// "blackbox" → BlackboxLatencyModel, "trained-roofline" → TrainedRooflineLatencyModel.
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
	case "crossmodel":
		// Emit deprecation warning (once per process)
		warnCrossmodelOnce.Do(func() {
			logrus.Warn("latency model \"crossmodel\" is deprecated and will be removed in a future version. Use --latency-model trained-physics instead.")
		})

		// Validate required fields BEFORE computing derived features (R11: guard division)
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires TP > 0, got %d", hw.TP)
		}
		if hw.ModelConfig.NumLayers <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires NumLayers > 0, got %d", hw.ModelConfig.NumLayers)
		}
		if hw.ModelConfig.NumHeads <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires NumHeads > 0, got %d", hw.ModelConfig.NumHeads)
		}
		if hw.ModelConfig.HiddenDim <= 0 {
			return nil, fmt.Errorf("latency model: crossmodel requires HiddenDim > 0, got %d", hw.ModelConfig.HiddenDim)
		}
		if len(coeffs.BetaCoeffs) < 4 {
			return nil, fmt.Errorf("latency model: crossmodel BetaCoeffs requires at least 4 elements, got %d", len(coeffs.BetaCoeffs))
		}
		if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
			return nil, err
		}
		// Compute architecture features at construction time (BC-10)
		headDim := float64(hw.ModelConfig.HiddenDim) / float64(hw.ModelConfig.NumHeads)
		numKVHeads := hw.ModelConfig.NumKVHeads
		if numKVHeads == 0 {
			numKVHeads = hw.ModelConfig.NumHeads // GQA fallback
		}
		kvDimScaled := (float64(hw.ModelConfig.NumLayers) * float64(numKVHeads) * headDim / float64(hw.TP)) * 1e-6
		var isMoE float64
		if hw.ModelConfig.NumLocalExperts > 0 {
			isMoE = 1.0
		}
		var isTP float64
		if hw.TP > 1 {
			isTP = 1.0
		}
		return &CrossModelLatencyModel{
			betaCoeffs:  coeffs.BetaCoeffs,
			alphaCoeffs: coeffs.AlphaCoeffs,
			numLayers:   hw.ModelConfig.NumLayers,
			kvDimScaled: kvDimScaled,
			isMoE:       isMoE,
			isTP:        isTP,
		}, nil
	case "trained-roofline":
		// Emit deprecation warning (once per process)
		warnTrainedRoofOnce.Do(func() {
			logrus.Warn("latency model \"trained-roofline\" is deprecated and will be removed in a future version. Use --latency-model trained-physics instead.")
		})

		// TrainedRooflineLatencyModel: roofline basis functions × learned corrections.
		// Requires model architecture (config.json) and hardware specs for basis functions.
		if hw.TP <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires TP > 0, got %d", hw.TP)
		}
		if hw.ModelConfig.NumLayers <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumLayers > 0, got %d", hw.ModelConfig.NumLayers)
		}
		if hw.ModelConfig.NumHeads <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumHeads > 0, got %d", hw.ModelConfig.NumHeads)
		}
		if hw.ModelConfig.HiddenDim <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires HiddenDim > 0, got %d", hw.ModelConfig.HiddenDim)
		}
		if hw.ModelConfig.IntermediateDim <= 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires IntermediateDim > 0, got %d", hw.ModelConfig.IntermediateDim)
		}
		if hw.ModelConfig.NumHeads%hw.TP != 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumHeads (%d) divisible by TP (%d)", hw.ModelConfig.NumHeads, hw.TP)
		}
		numKVHeadsTR := hw.ModelConfig.NumKVHeads
		if numKVHeadsTR == 0 {
			numKVHeadsTR = hw.ModelConfig.NumHeads // MHA fallback
		}
		if numKVHeadsTR%hw.TP != 0 {
			return nil, fmt.Errorf("latency model: trained-roofline requires NumKVHeads (%d) divisible by TP (%d)", numKVHeadsTR, hw.TP)
		}
		if invalidPositiveFloat(hw.HWConfig.TFlopsPeak) {
			return nil, fmt.Errorf("latency model: trained-roofline requires valid TFlopsPeak > 0, got %v", hw.HWConfig.TFlopsPeak)
		}
		if invalidPositiveFloat(hw.HWConfig.BwPeakTBs) {
			return nil, fmt.Errorf("latency model: trained-roofline requires valid BwPeakTBs > 0, got %v", hw.HWConfig.BwPeakTBs)
		}
		if len(coeffs.BetaCoeffs) < 7 {
			return nil, fmt.Errorf("latency model: trained-roofline BetaCoeffs requires at least 7 elements, got %d", len(coeffs.BetaCoeffs))
		}
		if err := validateCoeffs("BetaCoeffs", coeffs.BetaCoeffs); err != nil {
			return nil, err
		}
		headDimTR := hw.ModelConfig.HiddenDim / hw.ModelConfig.NumHeads
		// Defensive copy of coefficient slices to enforce the "frozen at construction" contract.
		// This prevents callers from mutating coefficients after construction.
		betaCopy := append([]float64(nil), coeffs.BetaCoeffs...)
		alphaCopy := append([]float64(nil), coeffs.AlphaCoeffs...)
		return &TrainedRooflineLatencyModel{
			betaCoeffs:  betaCopy,
			alphaCoeffs: alphaCopy,
			numLayers:   hw.ModelConfig.NumLayers,
			hiddenDim:   hw.ModelConfig.HiddenDim,
			numHeads:    hw.ModelConfig.NumHeads,
			headDim:     headDimTR,
			dKV:         numKVHeadsTR * headDimTR,
			dFF:         hw.ModelConfig.IntermediateDim,
			kEff:        max(1, hw.ModelConfig.NumExpertsPerTok), // matches training: k_eff = max(1, k)
			numExperts:  hw.ModelConfig.NumLocalExperts,
			isMoE:       hw.ModelConfig.NumLocalExperts > 0,
			tp:          hw.TP,
			flopsPeakUs: hw.HWConfig.TFlopsPeak * 1e6,
			bwHbmUs:     hw.HWConfig.BwPeakTBs * 1e6,
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
	case "blackbox":
		// Emit deprecation warning (once per process)
		warnBlackboxOnce.Do(func() {
			logrus.Warn("latency model \"blackbox\" is deprecated and will be removed in a future version. Use --latency-model trained-physics instead.")
		})

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
	default:
		return nil, fmt.Errorf("latency model: unknown backend %q; valid options: %s",
			hw.Backend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
	}
}
