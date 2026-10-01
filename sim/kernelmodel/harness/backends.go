// backends.go constructs BLIS's roofline and trained-physics latency backends for the SAME
// deployment the kernel was opened from, so one subset can be scored by four estimators.
//
// # What is held fixed, and why that is the whole point
//
// A four-way comparison is only about step time if step time is the only thing that differs.
// Everything else is pinned:
//
//   - KV blocks come from the KERNEL's memory methods for every arm. BLIS normally sizes KV
//     with latency.CalculateKVBlocks, which sits outside the sim.LatencyModel seam; letting
//     each arm size its own pool would give each a different resident batch, and the result
//     would mix admission behaviour into a step-time comparison.
//   - The workload, seed, warm-up discard, request budget, batch caps and dp scaling are the
//     harness's, untouched.
//   - Per-token and per-request HOST costs are not step time and do not differ: the registry
//     carries host_output_token = 45.9 us/token, the same magnitude trained-physics fitted,
//     so inter-token latency is not shifted by the choice of arm.
//
// # Where these backends are calibrated
//
// Roofline reads mfuPrefill/mfuDecode from hardware_config.json, which carries H100, H200,
// A100-SXM, A100-80 and L40S. It has NO Blackwell entry, and blis-registry has no
// roofline-b200.yaml or roofline-b300.yaml either. That is why the four-estimator table is
// Hopper-only: on Blackwell there is no coefficient to load, and inventing one to fill a
// column would be the failure the registry's own rules forbid.
//
// Trained-physics is a single global fit (11 beta + 3 alpha, iter29, loss 34.57%) transcribed
// from defaults.yaml. It is not scoped per chip, so it RUNS anywhere but is calibrated for
// none of these deployments specifically. Reported, not hidden.
package harness

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	"github.com/inference-sim/inference-sim/sim/latency"
	"gopkg.in/yaml.v3"
)

// vLLMDefaultActivationBytes is the activation width vLLM resolves to when a model config
// declares no dtype, on every CUDA device of capability 80 or above: bfloat16, 2 bytes.
// See altModel for the resolution chain this reproduces.
const vLLMDefaultActivationBytes = 2.0

// Estimator names the latency model an arm scores with.
type Estimator string

const (
	// EstimatorKernel is blis-latency-kernel, the subject of this experiment.
	EstimatorKernel Estimator = "kernel"
	// EstimatorRoofline is BLIS's analytic FLOPs/bandwidth backend.
	EstimatorRoofline Estimator = "roofline"
	// EstimatorTrainedPhysics is BLIS's roofline-plus-learned-corrections backend.
	EstimatorTrainedPhysics Estimator = "trained-physics"
)

// AllEstimators is the comparison order used in reports.
var AllEstimators = []Estimator{EstimatorKernel, EstimatorRoofline, EstimatorTrainedPhysics}

// trainedPhysicsCoefficients is the shipped fit, read from defaults.yaml rather than
// restated: cmd/ holds the same struct but in package main, which a library cannot import.
type trainedPhysicsCoefficients struct {
	TrainedPhysics *struct {
		AlphaCoeffs []float64 `yaml:"alpha_coeffs"`
		BetaCoeffs  []float64 `yaml:"beta_coeffs"`
	} `yaml:"trained_physics_coefficients"`
}

// BackendPaths locates the inputs the analytic backends need. Catalog is the clone root, the
// same directory the kernel's Repos names.
type BackendPaths struct {
	Catalog  string
	HWConfig string // hardware_config.json
	Defaults string // defaults.yaml
}

// altModel builds a roofline or trained-physics model for the deployment d.
//
// The ModelConfig derivation mirrors cmd/root.go's analytic-backend path exactly --
// ParseHFConfig, GetModelConfigFromHF, then the weight-precision and KV-dtype fallbacks --
// because a model configured differently from how production configures it would be a
// strawman rather than a baseline.
// hostAlpha is the alpha vector every analytic arm is given: the HOST costs, taken from the
// kernel so that no arm differs from another on anything but step time.
//
// alphaCoeffs[0] and [1] price QueueingTime (a constant plus a per-prompt-token term) and
// alphaCoeffs[2] prices OutputTokenProcessingTime, per emitted token. Only [2] touches the
// metric this comparison scores -- mean inter-token latency -- and the registry already
// carries the same magnitude the trained-physics fit produced (host_output_token = 45.9
// us/token, labelled assumed, citing that fit as an order-of-magnitude anchor). Passing the
// kernel's own value keeps the term identical across arms rather than silently removing a cost
// the kernel charges: zeroing it would have made every analytic arm's ITL 45.9 us per token
// cheaper than the kernel's for reasons unrelated to step time.
//
// [0] and [1] stay zero: QueueingTime is arrival-path work that the closed-loop harness
// measures around, and the kernel's admission overhead is a function of prompt length rather
// than a two-term affine fit, so there is no value that would make the arms equal.
type hostCosts struct {
	perOutputTokenUs int64
}

func altModel(e Estimator, d kernelmodel.Deployment, p BackendPaths, h hostCosts) (sim.LatencyModel, error) {
	hostAlpha := []float64{0, 0, float64(h.perOutputTokenUs)}
	hfPath := filepath.Join(p.Catalog, "models", d.Model, "config.json")
	hf, err := latency.ParseHFConfig(hfPath)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", d.Model, err)
	}
	mc, err := latency.GetModelConfigFromHF(hf)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", d.Model, err)
	}
	// Weight precision: quantization_config first, then the engine's stated format. cmd/
	// infers from the MODEL NAME; here the scenario states the quantization explicitly, which
	// is strictly better information about the same quantity.
	if mc.WeightBytesPerParam == 0 && d.Quantization != "" {
		if b := latency.InferWeightBytesFromModelName(d.Quantization); b > 0 {
			mc.WeightBytesPerParam = b
		}
	}
	// Activation precision when the config states none.
	//
	// GetModelConfigFromHF reads BytesPerParam from "torch_dtype" or "dtype". The two models
	// in this corpus's vLLM subset -- gpt-oss-120b (mxfp4) and minimax-m2.5 (fp8) -- declare
	// NEITHER, because their configs describe only the quantized weight format. Both analytic
	// backends then refuse to construct, and so would cmd/: it carries a model-name fallback
	// for WEIGHT precision and none for activation precision.
	//
	// vLLM resolves this case itself, and this follows its rule rather than inventing one.
	// With no dtype in the config, vLLM reads the safetensors weight metadata and otherwise
	// falls back to the platform's first supported dtype
	// (transformers_utils/model_arch_config_convertor.py get_torch_dtype, then
	// config/model.py _resolve_auto_dtype). On any device of capability 80 or above --
	// Ampere, Hopper, Blackwell -- platforms/cuda.py supported_dtypes returns
	// [bfloat16, float16, float32], so the resolved activation dtype is bfloat16: 2 bytes.
	//
	// Two bytes is therefore what the engine being modelled would actually run, not a
	// convenience. It is applied only when the config is silent, so a model that states its
	// dtype keeps it, and the substitution is reported by Deployment.AssumedActivationBytes
	// so a reader of the table knows which arms rest on it.
	if mc.BytesPerParam == 0 {
		mc.BytesPerParam = vLLMDefaultActivationBytes
	}
	if b, ok := latency.KVCacheDtypeToBytes(d.CacheDType); ok && b > 0 {
		mc.KVBytesPerParam = b
	}

	hc, err := latency.GetHWConfig(p.HWConfig, hwKey(d.Hardware))
	if err != nil {
		return nil, fmt.Errorf("%s on %s: %w", e, d.Hardware, err)
	}

	hw := sim.ModelHardwareConfig{
		Backend:     string(e),
		TP:          d.TP,
		ModelConfig: *mc,
		HWConfig:    hc,
	}
	coeffs := sim.LatencyCoeffs{}
	switch e {
	case EstimatorRoofline:
		coeffs.AlphaCoeffs = hostAlpha
	case EstimatorTrainedPhysics:
		tp, err := readTrainedPhysics(p.Defaults)
		if err != nil {
			return nil, err
		}
		coeffs.AlphaCoeffs = hostAlpha
		coeffs.BetaCoeffs = tp.BetaCoeffs
	default:
		return nil, fmt.Errorf("altModel: %q is not an analytic backend", e)
	}
	return latency.NewLatencyModel(coeffs, hw)
}

// readTrainedPhysics reads the shipped beta/alpha arrays from defaults.yaml.
func readTrainedPhysics(path string) (struct {
	AlphaCoeffs []float64 `yaml:"alpha_coeffs"`
	BetaCoeffs  []float64 `yaml:"beta_coeffs"`
}, error) {
	var zero struct {
		AlphaCoeffs []float64 `yaml:"alpha_coeffs"`
		BetaCoeffs  []float64 `yaml:"beta_coeffs"`
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return zero, fmt.Errorf("trained-physics defaults: %w", err)
	}
	var cfg trainedPhysicsCoefficients
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return zero, fmt.Errorf("trained-physics defaults: %w", err)
	}
	if cfg.TrainedPhysics == nil || len(cfg.TrainedPhysics.BetaCoeffs) < 7 {
		return zero, fmt.Errorf(
			"trained-physics defaults at %s carry fewer than the 7 beta coefficients the "+
				"backend requires; it selects its formula by len(beta), so a short array "+
				"would silently choose a different model", path)
	}
	zero.AlphaCoeffs = cfg.TrainedPhysics.AlphaCoeffs
	zero.BetaCoeffs = cfg.TrainedPhysics.BetaCoeffs
	return zero, nil
}

// hwKey maps a catalog chip name to hardware_config.json's key.
//
// The two namespaces differ in case and punctuation: the catalog says "h200", the calibration
// file says "H200". Only the parts the file actually carries are mapped; anything else is
// reported as unsupported rather than guessed, which is what keeps a missing calibration from
// silently becoming a different chip's numbers.
func hwKey(chip string) string {
	switch chip {
	case "h100":
		return "H100"
	case "h200":
		return "H200"
	case "a100-sxm":
		return "A100-SXM"
	case "a100-80":
		return "A100-80"
	case "l40s":
		return "L40S"
	default:
		return chip // GetHWConfig reports the available keys when this misses
	}
}
