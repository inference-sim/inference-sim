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

// Option customizes a LatencyModel at construction. Used to inject optional
// dependencies (currently the LoRA adapter-cost accessor) without churning the
// NewLatencyModel signature at its many call sites (R4). Both backends receive
// the same options, so an adapter effect applies identically (R23).
type Option func(*latencyOptions)

// latencyOptions accumulates the applied Options. Zero value ⇒ no adapter effect
// and verify width 1 (spec-decode off).
type latencyOptions struct {
	adapterCost sim.AdapterCost
	specTokens  int // num_speculative_tokens (K); 0 ⇒ verify width 1 ⇒ unchanged (#1528)
}

// WithAdapterCost supplies the LoRA per-step compute-overhead accessor. A nil
// accessor (or no option) leaves StepTime byte-identical to a pre-feature build
// (INV-6). The concrete accessor lives in sim/lora and reaches here as the
// sim.AdapterCost interface — sim/latency never imports sim/lora.
func WithAdapterCost(ac sim.AdapterCost) Option {
	return func(o *latencyOptions) { o.adapterCost = ac }
}

// WithSpeculativeDecode supplies K (num_speculative_tokens) for speculative decoding
// / MTP (#1528). Decode step-time cost then reflects a verify width of K+1 positions
// per forward pass — verifying drafts is not free. K<=0 (or no option) leaves StepTime
// byte-identical to a pre-feature build (INV-6/INV-BC-DP1).
func WithSpeculativeDecode(k int) Option {
	return func(o *latencyOptions) {
		if k > 0 {
			o.specTokens = k
		}
	}
}

// applyAdapterOverhead multiplies a base step time by the batch's LoRA
// compute-overhead factor (>= 1.0) from the accessor. It is the single shared
// application point so both backends behave identically (R23). A nil accessor —
// or a factor of 1.0 or below (a batch with no adapter ids normalizes to exactly
// 1.0; sub-unit values are rejected by the guard below) — returns base unchanged,
// preserving byte-identity for a no-adapter step (INV-6/INV-BC-DP1).
func applyAdapterOverhead(base int64, batch []*sim.Request, ac sim.AdapterCost) int64 {
	if ac == nil {
		return base
	}
	factor := ac.StepOverheadFactor(batch)
	// The AdapterCost contract guarantees a finite factor >= 1.0 (enforced in
	// sim/lora at construction), but this package is agnostic to that guarantee.
	// A non-finite factor (NaN/±Inf) is NOT caught by `factor <= 1.0` — NaN
	// comparisons are always false — and would reach clampToInt64 as NaN→MaxInt64,
	// stalling the simulation clock (INV-3). Defend the clock at this boundary
	// (R3/R20): treat any non-finite or <= 1.0 factor as "no overhead" and return
	// base unchanged, mirroring clampToInt64's clamp-don't-panic philosophy.
	if math.IsNaN(factor) || math.IsInf(factor, 0) || factor <= 1.0 {
		return base
	}
	return max(1, clampToInt64(float64(base)*factor))
}

// RooflineLatencyModel estimates latency using analytical FLOPs/bandwidth roofline model.
// Step time is computed via rooflineStepTime(); overhead estimates use alpha coefficients.
type RooflineLatencyModel struct {
	modelConfig sim.ModelConfig
	hwConfig    sim.HardwareCalib
	tp          int
	alphaCoeffs []float64
	// adapterCost supplies the per-step LoRA compute-overhead factor (#1467). nil
	// when the LoRA subsystem is inert, in which case StepTime is byte-identical to
	// a pre-feature build (INV-6). Set via WithAdapterCost at construction.
	adapterCost sim.AdapterCost
	// specTokens is num_speculative_tokens (K) for speculative decoding / MTP (#1528);
	// 0 when off. Decode FLOPs and dynamic KV bytes use K+1 new tokens per decode
	// request (the target verifies K drafts + 1 bonus in one forward pass). 0 ⇒ verify
	// width 1 ⇒ StepTime byte-identical to a pre-feature build (INV-6). Set via
	// WithSpeculativeDecode at construction.
	specTokens int
}

func (m *RooflineLatencyModel) StepTime(batch []*sim.Request) int64 {
	stepConfig := StepConfig{
		PrefillRequests: make([]PrefillRequestConfig, 0, len(batch)),
		DecodeRequests:  make([]DecodeRequestConfig, 0, len(batch)),
	}
	for _, req := range batch {
		if req.ProgressIndex < req.InputLen() {
			stepConfig.PrefillRequests = append(stepConfig.PrefillRequests, PrefillRequestConfig{
				ProgressIndex:       req.ProgressIndex,
				NumNewPrefillTokens: req.NumNewTokens,
			})
		} else if len(req.OutputTokens) > 0 {
			// NumNewDecodeTokens carries the verify WIDTH (K+1) — the COST quantity:
			// the target processes K+1 positions per decode forward pass under
			// speculative decoding / MTP (#1528). This is deliberately NOT
			// req.NumNewTokens, which under MTP is the accepted-token count g (throughput,
			// drives ProgressIndex) — sizing cost by g would model acceptance as free.
			// specTokens=0 (feature off) ⇒ width 1 ⇒ byte-identical to pre-feature (INV-6).
			stepConfig.DecodeRequests = append(stepConfig.DecodeRequests, DecodeRequestConfig{
				ProgressIndex:      req.ProgressIndex,
				NumNewDecodeTokens: m.specTokens + 1,
			})
		}
	}
	return applyAdapterOverhead(max(1, rooflineStepTime(m.modelConfig, m.hwConfig, stepConfig, m.tp)), batch, m.adapterCost)
}

func (m *RooflineLatencyModel) QueueingTime(req *sim.Request) int64 {
	var totalProcessingTime float64
	totalProcessingTime += m.alphaCoeffs[0]
	totalProcessingTime += m.alphaCoeffs[1] * float64(req.InputLen())
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
//
// Options inject optional dependencies; the same options are applied to whichever
// backend is selected, so an adapter-overhead accessor (WithAdapterCost) affects
// both identically (R23). No options ⇒ pre-feature behavior (INV-6).
//
// The inter-node interconnect topology (#1530) is NOT an Option here: it arrives on
// hw.NetworkTopology (set by sim.WithNetworkTopology at ModelHardwareConfig
// construction), alongside the other latency inputs TP/DP/MoECommBackend. Only the
// trained-physics backend reads it — the roofline backend models no communication
// term at all (rooflineStepTime uses TP purely as a sharding divisor), so there is
// no cross-node cost for it to scale. That asymmetry is deliberate, not R23 drift:
// giving roofline a cross-node cost would mean adding a comm term it does not have,
// which is a different change from re-scaling an existing one.
func NewLatencyModel(coeffs sim.LatencyCoeffs, hw sim.ModelHardwareConfig, opts ...Option) (sim.LatencyModel, error) {
	var o latencyOptions
	for _, opt := range opts {
		opt(&o)
	}
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
			adapterCost: o.adapterCost,
			specTokens:  o.specTokens,
		}, nil
	case "trained-physics":
		// TrainedPhysicsModel: physics-informed roofline with architecture-aware MoE overhead.
		// Uses roofline basis functions with learned corrections and conditional β₈ scaling.
		// Trained coefficients from iteration 29 (loss: 34.57%).
		model, err := NewTrainedPhysicsModel(coeffs, hw)
		if err != nil {
			return nil, err
		}
		model.adapterCost = o.adapterCost
		model.specTokens = o.specTokens
		return model, nil
	default:
		return nil, fmt.Errorf("latency model: unknown backend %q; valid options: %s",
			hw.Backend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
	}
}
