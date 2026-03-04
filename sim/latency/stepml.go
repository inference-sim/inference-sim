// Package latency — StepML latency model.
//
// StepMLLatencyModel is a data-driven LatencyModel that loads trained model
// artifacts from a JSON file. It supports linear coefficient models and
// (future) XGBoost tree ensembles.
//
// This is the lightweight research-phase implementation (WP0). Production
// quality with full INV-M-1 through INV-M-6 validation comes in WP5.
package latency

import (
	"encoding/json"
	"fmt"
	"math"
	"os"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/util"
)

// StepMLArtifact is the top-level JSON schema for a StepML model artifact.
type StepMLArtifact struct {
	Version                      int              `json:"version"`
	StepTime                     *LinearModel     `json:"step_time"`
	StepTimeRegimes              []RegimeEntry    `json:"step_time_regimes,omitempty"`
	QueueingTime                 *LinearModel     `json:"queueing_time,omitempty"`
	StepOverheadUs               float64          `json:"step_overhead_us"`
	StepOverheadPerReqUs         float64          `json:"step_overhead_per_req_us"`
	OutputTokenProcessingTimeUs  float64          `json:"output_token_processing_time_us"`
	SchedulingProcessingTimeUs   float64          `json:"scheduling_processing_time_us"`
	PreemptionProcessingTimeUs   float64          `json:"preemption_processing_time_us"`
}

// LinearModel represents a linear prediction model: intercept + Σ(coeff_i * feature_i).
// When OutputTransform is "expm1", the prediction is exponentiated: exp(linear) - 1.
// This supports log-linear models trained on log1p(target).
type LinearModel struct {
	ModelType           string             `json:"model_type"`
	Intercept           float64            `json:"intercept"`
	FeatureCoefficients map[string]float64 `json:"feature_coefficients"`
	OutputTransform     string             `json:"output_transform,omitempty"`
}

// RegimeEntry pairs a condition with a linear model for regime-based dispatch.
// Regimes are evaluated in order; the first matching condition wins.
// A nil Condition acts as a fallback (matches everything).
type RegimeEntry struct {
	Name      string          `json:"name"`
	Condition *RegimeCondition `json:"condition"`
	Model     LinearModel     `json:"model"`
}

// RegimeCondition is a simple threshold condition on a batch feature.
type RegimeCondition struct {
	Feature string  `json:"feature"`
	Op      string  `json:"op"` // "==", "<", "<=", ">", ">="
	Value   float64 `json:"value"`
}

// StepMLLatencyModel implements sim.LatencyModel using trained artifacts.
type StepMLLatencyModel struct {
	artifact StepMLArtifact
}

// LoadStepMLArtifact reads and validates a StepML JSON artifact from disk.
func LoadStepMLArtifact(path string) (*StepMLArtifact, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("stepml: read artifact %q: %w", path, err)
	}

	var art StepMLArtifact
	if err := json.Unmarshal(data, &art); err != nil {
		return nil, fmt.Errorf("stepml: parse artifact %q: %w", path, err)
	}

	if art.Version < 1 {
		return nil, fmt.Errorf("stepml: unsupported artifact version %d", art.Version)
	}

	hasRegimes := len(art.StepTimeRegimes) > 0
	hasSingleModel := art.StepTime != nil

	if !hasRegimes && !hasSingleModel {
		return nil, fmt.Errorf("stepml: artifact missing step_time model (provide step_time or step_time_regimes)")
	}

	if hasSingleModel {
		if art.StepTime.ModelType != "linear" {
			return nil, fmt.Errorf("stepml: unsupported step_time model_type %q (supported: linear)", art.StepTime.ModelType)
		}
		if err := validateLinearModel("step_time", art.StepTime); err != nil {
			return nil, err
		}
	}

	for i, r := range art.StepTimeRegimes {
		name := fmt.Sprintf("step_time_regimes[%d](%s)", i, r.Name)
		if err := validateLinearModel(name, &r.Model); err != nil {
			return nil, err
		}
		if r.Condition != nil {
			switch r.Condition.Op {
			case "==", "<", "<=", ">", ">=":
				// valid
			default:
				return nil, fmt.Errorf("stepml: %s has unsupported condition op %q", name, r.Condition.Op)
			}
		}
	}

	if art.QueueingTime != nil {
		if err := validateLinearModel("queueing_time", art.QueueingTime); err != nil {
			return nil, err
		}
	}

	return &art, nil
}

func validateLinearModel(name string, m *LinearModel) error {
	if math.IsNaN(m.Intercept) || math.IsInf(m.Intercept, 0) {
		return fmt.Errorf("stepml: %s intercept is NaN/Inf", name)
	}
	for feat, coeff := range m.FeatureCoefficients {
		if math.IsNaN(coeff) || math.IsInf(coeff, 0) {
			return fmt.Errorf("stepml: %s coefficient %q is NaN/Inf", name, feat)
		}
	}
	return nil
}

// NewStepMLLatencyModel creates a StepML LatencyModel from a loaded artifact.
func NewStepMLLatencyModel(art *StepMLArtifact) *StepMLLatencyModel {
	return &StepMLLatencyModel{artifact: *art}
}

// extractBatchFeatures computes aggregate features from a batch of requests.
// These features mirror the ground-truth step-level data schema.
func extractBatchFeatures(batch []*sim.Request) map[string]float64 {
	var prefillTokens, decodeTokens int64
	var numPrefillReqs, numDecodeReqs int64
	var kvSum, kvMax int64
	var kvCount int64
	var kvSumSq float64 // for kv_std computation

	for _, req := range batch {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			// Prefill phase
			prefillTokens += int64(req.NumNewTokens)
			numPrefillReqs++
		} else if len(req.OutputTokens) > 0 {
			// Decode phase
			decodeTokens += int64(req.NumNewTokens)
			numDecodeReqs++
		}

		// KV cache proxy features from ProgressIndex
		pi := req.ProgressIndex
		kvSum += pi
		kvSumSq += float64(pi) * float64(pi)
		kvCount++
		if pi > kvMax {
			kvMax = pi
		}
	}

	var kvMean, kvStd float64
	if kvCount > 0 {
		kvMean = float64(kvSum) / float64(kvCount)
		if kvCount > 1 {
			// Population std: sqrt(E[X^2] - E[X]^2)
			variance := kvSumSq/float64(kvCount) - kvMean*kvMean
			if variance > 0 {
				kvStd = math.Sqrt(variance)
			}
		}
	}

	pf := float64(prefillTokens)
	dt := float64(decodeTokens)
	ndr := float64(numDecodeReqs)

	return map[string]float64{
		"prefill_tokens":   pf,
		"decode_tokens":    dt,
		"num_prefill_reqs": float64(numPrefillReqs),
		"num_decode_reqs":  ndr,
		"scheduled_tokens": pf + dt,
		"kv_sum":           float64(kvSum),
		"kv_max":           float64(kvMax),
		"kv_mean":          kvMean,
		"kv_std":           kvStd,
		// Interaction features for regime-specific models
		"prefill_x_decode":  pf * dt,
		"decode_x_kv_mean":  ndr * kvMean,
		"prefill_sq":        pf * pf,
	}
}

// predictLinear evaluates intercept + Σ(coeff_i * features[i]).
// Features not present in the coefficient map are ignored (zero contribution).
// When OutputTransform is "expm1", applies exp(result) - 1 (inverse of log1p).
func predictLinear(m *LinearModel, features map[string]float64) float64 {
	result := m.Intercept
	for feat, coeff := range m.FeatureCoefficients {
		if val, ok := features[feat]; ok {
			result += coeff * val
		}
	}
	if m.OutputTransform == "expm1" {
		result = math.Expm1(result) // exp(result) - 1
	}
	return result
}

// StepTime estimates batch step execution time in microseconds.
//
// The model predicts GPU compute time from batch features. Since the training
// data only captures GPU forward pass time (step.duration_us), the full step
// cycle time also includes CPU-side overhead (scheduling, sync, memory management).
//
// The overhead is applied as a floor (max) rather than additive because:
//   - Small batches: overhead-dominated (~4ms); compute is negligible (~100µs)
//   - Large batches: compute-dominated (~6ms); overhead is negligible
//   - The transition is at the memory↔compute bound crossover point
//
// step_overhead_per_req_us is additive (per-request scheduling cost).
//
// When StepTimeRegimes is present, evaluates regimes in order; first match wins.
// Otherwise falls back to the single StepTime linear model.
func (s *StepMLLatencyModel) StepTime(batch []*sim.Request) int64 {
	features := extractBatchFeatures(batch)

	var prediction float64
	if len(s.artifact.StepTimeRegimes) > 0 {
		prediction = s.predictRegime(features)
	} else {
		prediction = predictLinear(s.artifact.StepTime, features)
	}

	// Apply overhead as a floor: step_cycle = max(overhead, compute).
	// This models the memory↔compute crossover: at small batches the overhead
	// floor dominates; at large batches, compute exceeds the floor naturally.
	//
	// Also cap at 5x overhead to prevent exponential blowup from the expm1
	// transform on outlier feature values. 5x covers the realistic range
	// (e.g., 4ms overhead → 20ms cap, well above any real step time).
	if s.artifact.StepOverheadUs > 0 {
		if prediction < s.artifact.StepOverheadUs {
			prediction = s.artifact.StepOverheadUs
		}
		maxStep := s.artifact.StepOverheadUs * 3
		if prediction > maxStep {
			prediction = maxStep
		}
	}

	// Per-request overhead is always additive (scheduling cost per request).
	if s.artifact.StepOverheadPerReqUs > 0 {
		batchSize := features["num_prefill_reqs"] + features["num_decode_reqs"]
		prediction += s.artifact.StepOverheadPerReqUs * batchSize
	}

	if prediction < 1 {
		prediction = 1 // INV-M-1: positive predictions
	}
	return int64(prediction)
}

// predictRegime evaluates regime conditions in order and uses the first matching model.
func (s *StepMLLatencyModel) predictRegime(features map[string]float64) float64 {
	for _, r := range s.artifact.StepTimeRegimes {
		if r.Condition == nil || evalCondition(r.Condition, features) {
			return predictLinear(&r.Model, features)
		}
	}
	// No regime matched and no fallback — use single model if available
	if s.artifact.StepTime != nil {
		return predictLinear(s.artifact.StepTime, features)
	}
	return 1 // safety fallback
}

// evalCondition checks a simple threshold condition against feature values.
func evalCondition(c *RegimeCondition, features map[string]float64) bool {
	val := features[c.Feature]
	switch c.Op {
	case "==":
		return val == c.Value
	case "<":
		return val < c.Value
	case "<=":
		return val <= c.Value
	case ">":
		return val > c.Value
	case ">=":
		return val >= c.Value
	default:
		return false
	}
}

// QueueingTime estimates arrival-to-queue delay for a request in microseconds.
func (s *StepMLLatencyModel) QueueingTime(req *sim.Request) int64 {
	if s.artifact.QueueingTime == nil {
		return 0
	}
	features := map[string]float64{
		"input_len": float64(len(req.InputTokens)),
	}
	prediction := predictLinear(s.artifact.QueueingTime, features)
	if prediction < 0 {
		prediction = 0
	}
	return int64(prediction)
}

// OutputTokenProcessingTime estimates per-token post-processing in microseconds.
func (s *StepMLLatencyModel) OutputTokenProcessingTime() int64 {
	v := s.artifact.OutputTokenProcessingTimeUs
	if v < 0 {
		return 0
	}
	return int64(v)
}

// SchedulingProcessingTime estimates scheduling overhead per request in microseconds.
func (s *StepMLLatencyModel) SchedulingProcessingTime() int64 {
	v := s.artifact.SchedulingProcessingTimeUs
	if v < 0 {
		return 0
	}
	return int64(v)
}

// PreemptionProcessingTime estimates preemption overhead per eviction in microseconds.
func (s *StepMLLatencyModel) PreemptionProcessingTime() int64 {
	v := s.artifact.PreemptionProcessingTimeUs
	if v < 0 {
		return 0
	}
	return int64(v)
}
