package cmd

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/saturation"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// distDefaults are the shared default values for distribution synthesis flags.
// Both runCmd and observeCmd use these constants so the two commands produce
// identical workload shapes when called with no explicit distribution flags.
// Changing a value here affects both commands simultaneously.
//
// Flags covered:
//
//	--prompt-tokens, --prompt-tokens-stdev, --prompt-tokens-min, --prompt-tokens-max
//	--output-tokens, --output-tokens-stdev, --output-tokens-min, --output-tokens-max
//
// NOT covered: --prefix-tokens (default 0 means "no shared prefix", a feature toggle,
// not a distribution shape parameter).
//
// --num-requests is intentionally NOT shared: run defaults to 100 (safe for quick sims),
// observe defaults to 0 (requires explicit bound — see observe_cmd.go).
const (
	defaultPromptMean  = 512
	defaultPromptStdev = 256
	defaultPromptMin   = 2
	defaultPromptMax   = 7000
	defaultOutputMean  = 512
	defaultOutputStdev = 256
	defaultOutputMin   = 2
	defaultOutputMax   = 7000
)

var (
	// CLI flags for vllm server configs
	seed                      int64     // Seed for random token generation
	simulationHorizon         int64     // Total simulation time (in ticks)
	logLevel                  string    // Log verbosity level
	totalKVBlocks             int64     // Total number of KV blocks available on GPU
	maxRunningReqs            int64     // Maximum number of requests in the Running batch
	maxScheduledTokens        int64     // Maximum total number of tokens across requests in the Running batch
	blockSizeTokens           int64     // Number of tokens per KV block
	betaCoeffs                []float64 // List of beta coeffs corresponding to step features
	alphaCoeffs               []float64 // List of alpha coeffs corresponding to pre, postprocessing delays
	defaultsFilePath          string    // Path to default constants - trained coefficients, default specs and workloads
	modelConfigFolder         string    // Path to folder containing config.json and model.json
	hwConfigPath              string    // Path to constants specific to hardware type (GPU)
	workloadType              string    // Workload type (chatbot, summarization, contentgen, multidoc, distribution)
	longPrefillTokenThreshold int64     // Max length of prefill beyond which chunked prefill is triggered
	rate                      float64   // Requests arrival per second
	numRequests               int       // Number of requests
	concurrency               int       // Number of concurrent virtual users (closed-loop)
	thinkTimeMs               int       // Think time between response and next request (ms)
	prefixTokens              int       // Prefix Token Count
	promptTokensMean          int       // Average Prompt Token Count
	promptTokensStdev         int       // Stdev Prompt Token Count
	promptTokensMin           int       // Min Prompt Token Count
	promptTokensMax           int       // Max Prompt Token Count
	outputTokensMean          int       // Average Output Token Count
	outputTokensStdev         int       // Stdev Output Token Count
	outputTokensMin           int       // Min Output Token Count
	outputTokensMax           int       // Max Output Token Count
	latencyModelBackend       string    // CLI --latency-model flag: selects latency model backend (Cobra-bound, NEVER mutated inside Run)
	maxModelLen               int64     // CLI --max-model-len: max total sequence length (input + output); 0 = unlimited
	// CLI flags for model, GPU, TP
	model                string // LLM name
	gpu                  string // GPU type
	tensorParallelism    int    // TP value
	dataParallelism      int    // DP value (MoE only; trained-physics backend only)
	enableExpertParallel bool   // EP mode (MoE only; trained-physics backend only)
	moeCommBackend       string // MoE all-to-all comm backend (MoE only; trained-physics backend only)

	// cluster config
	numInstances int // Number of instances in the cluster

	// online routing pipeline config
	admissionPolicy       string             // Admission policy name
	admissionLatency      int64              // Admission latency in microseconds
	routingLatency        int64              // Routing latency in microseconds
	tokenBucketCapacity   float64            // Token bucket capacity
	tokenBucketRefillRate float64            // Token bucket refill rate (tokens/second)
	tierShedThreshold     int                // Tier-shed overload threshold (0 = any load)
	tierShedMinPriority   int                // Tier-shed minimum admitted priority under overload
	tenantBudgets         map[string]float64 // Per-tenant fraction of total capacity (nil = no enforcement)
	sloPriorityOverrides  map[string]int     // SLO class → priority overrides (nil = GAIE defaults)
	sloTargetsMap         map[string]int64   // SLO class → TTFT target µs for slo-deadline ordering (nil = disabled)
	gaieQDThreshold       float64            // GAIE-legacy queue depth threshold per instance (default 5)
	gaieKVThreshold       float64            // GAIE-legacy KV cache utilization threshold (default 0.8)

	// routing policy config (PR 6, evolved in PR17)
	routingPolicy  string // Routing policy name
	routingScorers string // Comma-separated name:weight pairs for weighted routing

	// Scheduler and preemption config
	scheduler        string // Scheduler name
	preemptionPolicy string // Preemption victim selection policy

	// Policy bundle config
	policyConfigPath string // Path to YAML policy configuration file

	// Fitness evaluation config (PR9)
	fitnessWeights string // Fitness weights string "key:val,key:val"

	// Decision trace config (PR13)
	traceLevel      string // Trace verbosity level
	counterfactualK int    // Number of counterfactual candidates
	summarizeTrace  bool   // Print trace summary after simulation

	// Workload spec config (PR10)
	workloadSpecPath string // Path to YAML workload specification file
	lazyGeneration   bool   // --lazy-generation: stream requests from generator (alpha, #1441)

	// Tiered KV cache config (PR12)
	kvCPUBlocks             int64
	kvOffloadThreshold      float64
	kvTransferBandwidth     float64
	kvTransferBaseLatency   int64
	snapshotRefreshInterval int64
	cacheSignalDelay        int64
	gpuMemoryUtilization    float64

	// PD disaggregation config
	prefillInstances       int     // Number of instances dedicated to prefill
	decodeInstances        int     // Number of instances dedicated to decode
	prefillDecodeInstances int     // Number of shared-role instances (both prefill and decode), issue #1276
	pdDecider              string  // Disaggregation decider name
	pdTransferBandwidth    float64 // Inter-instance KV transfer bandwidth in GB/s
	pdTransferBaseLatency  float64 // Inter-instance KV transfer base latency in ms
	pdTransferContention   bool    // Enable fair-share bandwidth contention model
	pdPrefixThreshold      int     // Non-cached token threshold for prefix-threshold decider
	prefillRoutingScorers  string  // Scorer weights for prefill pool routing
	decodeRoutingScorers   string  // Scorer weights for decode pool routing

	// E/P/D disaggregation config (GAP-4, issue #1264)
	encodeInstances int    // Number of instances dedicated to encoding multimodal input (0 = disabled)
	encodeDecider   string // Encode decider name: "never" (default), "always", "multimodal"

	// Autoscaler config (Phase 1C)
	modelAutoscalerIntervalUs float64 // tick interval in μs; 0 = disabled

	// Flow control config (issue #882, GIE parity)
	flowControlEnabled              bool
	flowControlDetector             string
	flowControlDispatchOrder        string
	flowControlSLOTargets           string
	flowControlMaxQueueDepth        int
	flowControlQueueDepthThreshold  float64
	flowControlKVCacheUtilThreshold float64
	flowControlMaxConcurrency       int
	flowControlPerBandCapacity      int
	flowControlUsageLimitThreshold  float64
	flowControlFairnessPolicy       string
	flowControlRequestTTL           int64
	flowControlQueueShedding        bool
	flowControlDispatchTickInterval int64
	flowControlInFlightEviction     bool

	// Per-pool hardware override config
	prefillTP           int
	decodeTP            int
	prefillHardware     string
	decodeHardware      string
	prefillLatencyModel string
	decodeLatencyModel  string
	prefillMaxModelLen  int64
	decodeMaxModelLen   int64

	// per-request timeout override for blis run (seconds; negative = disabled, 0 is rejected)
	requestTimeoutSecs int

	// Goodput SLO targets (issue #1413). Each is "class=duration[,class=duration...]" using
	// Go duration syntax. Distinct from --slo-targets (dispatch ordering, µs); these gate
	// goodput emission. Precedence: CLI > trace header > workload spec.
	goodputSLOTTFT string
	goodputSLOITL  string
	goodputSLOE2E  string

	// output file paths
	metricsPath      string // File to write MetricsOutput JSON for blis run (--metrics-path)
	resultsPath      string // File to write []SimResult JSON for blis replay (--results-path)
	saturationReport string // File to write BacklogDriftReport JSON for saturation analysis (--saturation-report)

	// saturation analysis configuration
	saturationWindowSec  int     // Window size in seconds for backlog-drift analysis (--saturation-window)
	saturationMinWindows int     // Minimum number of windows required for classification (--saturation-min-windows)
	saturationPeakRatio  float64 // Peak/mean ratio threshold for transient backlog detection (--saturation-peak-ratio)
	saturationPeakBand   float64 // Confidence band around peak ratio threshold (--saturation-peak-band)
	saturationConfidence float64 // Confidence level for slope CI (--saturation-ci)

	// post-hoc backlog classifier policy (#1391, #1392)
	saturationClassifier     string  // Classifier name: "drain-ratio" (default) or "slope-based" (--saturation-classifier)
	saturationWarmupWindows  int     // Inject windows skipped as warmup (drain-ratio only) (--saturation-warmup-windows)
	saturationTailWindows    int     // Inject windows skipped as tail (drain-ratio only) (--saturation-tail-windows)
	saturationSaturatedRatio float64 // DrainRatio < this → PERSISTENTLY_SATURATED (--saturation-drain-ratio-saturated)
	saturationTransientRatio float64 // DrainRatio < this → TRANSIENT_BACKLOG (--saturation-drain-ratio-transient)

	// post-hoc saturation detector configuration (#1369)
	postHocDetector     string  // Post-hoc saturation detector: "composite", "threshold", "none" (--post-hoc-detector)
	saturationThreshold float64 // Threshold in ms for threshold detector (--saturation-threshold-ms)

	// trace export
	traceOutput string // File prefix for TraceV2 export (<prefix>.yaml + <prefix>.csv)
)

// registerSaturationFlags registers backlog-drift analysis flags on the given command.
// These flags control post-hoc saturation classification and are shared across run, replay, and observe.
//
// Two classifier families are supported (#1391, #1392):
//   - "drain-ratio" (default): mean per-window NumLeft/NumEntered → quantified ρ = 1/DrainRatio
//   - "slope-based": OLS regression CI on ActiveEnd over inject-phase windows
func registerSaturationFlags(cmd *cobra.Command) {
	cmd.Flags().IntVar(&saturationWindowSec, "saturation-window", 60, "Window size in seconds for backlog-drift analysis")
	cmd.Flags().IntVar(&saturationMinWindows, "saturation-min-windows", 5, "Minimum number of complete windows required for reliable classification")
	cmd.Flags().Float64Var(&saturationPeakRatio, "saturation-peak-ratio", 2.0, "Peak/mean in-flight ratio threshold for TRANSIENT_BACKLOG detection (slope-based classifier only)")
	cmd.Flags().Float64Var(&saturationPeakBand, "saturation-peak-band", 0.2, "Confidence band around peak-ratio threshold (slope-based classifier only)")
	cmd.Flags().Float64Var(&saturationConfidence, "saturation-ci", 0.95, "Confidence level for slope significance test (0.90, 0.95, or 0.99) (slope-based classifier only)")
	cmd.Flags().StringVar(&saturationClassifier, "saturation-classifier", "drain-ratio",
		"Backlog classifier: "+strings.Join(sim.ValidBacklogClassifierNames(), ", "))
	cmd.Flags().IntVar(&saturationWarmupWindows, "saturation-warmup-windows", 2, "Inject windows skipped as warmup (drain-ratio classifier only)")
	cmd.Flags().IntVar(&saturationTailWindows, "saturation-tail-windows", 1, "Inject windows skipped as tail boundary (drain-ratio classifier only)")
	cmd.Flags().Float64Var(&saturationSaturatedRatio, "saturation-drain-ratio-saturated", 0.95, "Mean DrainRatio threshold below which run is PERSISTENTLY_SATURATED (drain-ratio classifier only)")
	cmd.Flags().Float64Var(&saturationTransientRatio, "saturation-drain-ratio-transient", 0.98, "Mean DrainRatio threshold below which run is TRANSIENT_BACKLOG (drain-ratio classifier only)")
}

// applyRopeScaling applies rope_scaling factor to maxPosEmb if applicable.
// Returns the (possibly scaled) value and whether scaling was applied.
// modelType is the HuggingFace model_type string (empty if not present).
// ropeScaling is the raw rope_scaling value from config.json (nil if not present).
//
// Blacklist approach matching vLLM's _get_and_verify_max_len:
// Types "su", "longrope", "llama3" are excluded (these encode full context in max_position_embeddings).
// All other types (linear, dynamic, yarn, default, mrope, etc.) apply the factor.
// "mrope" is intentionally NOT excluded: vLLM normalizes mrope → "default" via patch_rope_scaling_dict
// and then applies the factor. BLIS reads raw JSON where mrope falls through the blacklist — same result.
// For "yarn", original_max_position_embeddings is used as base when present.
// gemma3 model_type skips rope_scaling entirely (max_position_embeddings is pre-scaled).
// Uses substring match to handle both "gemma3" (top-level) and "gemma3_text" (after text_config pivot).
func applyRopeScaling(maxPosEmb int, modelType string, ropeScaling any) (scaled int, applied bool) {
	// R3: degenerate base guard
	if maxPosEmb <= 0 {
		return maxPosEmb, false
	}

	// gemma3 model_type: skip rope_scaling entirely (BC-3).
	// Note: ParseHFConfig's text_config pivot overwrites model_type from "gemma3" to
	// "gemma3_text" for multimodal models. Use strings.Contains to match both variants,
	// aligning with vLLM's substring check ("gemma3" not in hf_config.model_type).
	if strings.Contains(modelType, "gemma3") {
		return maxPosEmb, false
	}

	// No rope_scaling present
	if ropeScaling == nil {
		return maxPosEmb, false
	}

	// rope_scaling must be a JSON object (map[string]any)
	ropeMap, ok := ropeScaling.(map[string]any)
	if !ok {
		return maxPosEmb, false
	}

	// Read type (some configs use "rope_type" instead of "type")
	ropeType, _ := ropeMap["type"].(string)
	if ropeType == "" {
		ropeType, _ = ropeMap["rope_type"].(string)
	}

	// Blacklist: these types already embed scaled context in max_position_embeddings
	if ropeType == "su" || ropeType == "longrope" || ropeType == "llama3" {
		return maxPosEmb, false
	}

	// Extract factor
	factor, ok := ropeMap["factor"].(float64)
	if !ok || factor <= 1.0 {
		return maxPosEmb, false
	}

	// NaN/Inf defense-in-depth (standard JSON can't produce these, but non-standard sources might)
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		return maxPosEmb, false
	}

	// For yarn, use original_max_position_embeddings as base if available (BC-4)
	base := maxPosEmb
	if ropeType == "yarn" {
		if orig, ok := ropeMap["original_max_position_embeddings"].(float64); ok && orig > 0 {
			// Overflow guard on original_max_position_embeddings
			if orig >= float64(math.MaxInt) {
				return maxPosEmb, false
			}
			base = int(orig)
		}
	}

	// Compute scaled value with overflow guard
	product := float64(base) * factor
	if product >= float64(math.MaxInt) || product < 0 {
		return maxPosEmb, false
	}

	return int(product), true
}

// rootCmd is the base command for the CLI
var rootCmd = &cobra.Command{
	Use:   "blis",
	Short: "BLIS — Blackbox Inference Simulator for LLM serving systems",
}

// validateDistributionParams checks token distribution bounds common to both the
// concurrency and distribution synthesis paths (R3). Returns a non-empty error
// string if any parameter violates a bound, empty string if all are valid.
// A stdev of 0 is always valid — it produces a constant (deterministic) distribution.
// Extracted for unit testability (R14).
func validateDistributionParams(promptMin, promptMax, outputMin, outputMax, promptStdev, outputStdev, promptMean, outputMean int) string {
	if promptMin < 1 {
		return fmt.Sprintf("--prompt-tokens-min must be >= 1, got %d", promptMin)
	}
	if promptMax < 1 {
		return fmt.Sprintf("--prompt-tokens-max must be >= 1, got %d", promptMax)
	}
	if outputMin < 1 {
		return fmt.Sprintf("--output-tokens-min must be >= 1, got %d", outputMin)
	}
	if outputMax < 1 {
		return fmt.Sprintf("--output-tokens-max must be >= 1, got %d", outputMax)
	}
	if promptStdev < 0 {
		return fmt.Sprintf("--prompt-tokens-stdev must be >= 0, got %d", promptStdev)
	}
	if outputStdev < 0 {
		return fmt.Sprintf("--output-tokens-stdev must be >= 0, got %d", outputStdev)
	}
	if promptMin > promptMax {
		return fmt.Sprintf("--prompt-tokens-min (%d) must be <= --prompt-tokens-max (%d)", promptMin, promptMax)
	}
	if outputMin > outputMax {
		return fmt.Sprintf("--output-tokens-min (%d) must be <= --output-tokens-max (%d)", outputMin, outputMax)
	}
	if promptMean > promptMax || promptMean < promptMin || promptStdev > promptMax || (promptStdev != 0 && promptStdev < promptMin) {
		return "prompt-tokens and prompt-tokens-stdev should be in range [prompt-tokens-min, prompt-tokens-max]"
	}
	if outputMean > outputMax || outputMean < outputMin || outputStdev > outputMax || (outputStdev != 0 && outputStdev < outputMin) {
		return "output-tokens and output-tokens-stdev should be in range [output-tokens-min, output-tokens-max]"
	}
	return ""
}

// allZeros reports whether all values in the coefficients slice are 0 (default).
func allZeros(values []float64) bool {
	for _, v := range values {
		if v != 0 {
			return false
		}
	}
	return true
}

// latencyResolution holds the resolved components from resolveLatencyConfig.
// Callers use these values to construct sim.SimConfig sub-configs.
// Package-level vars (totalKVBlocks, maxModelLen, model, gpu, tensorParallelism,
// modelConfigFolder, hwConfigPath) are mutated as side effects.
type latencyResolution struct {
	Backend     string            // resolved latency backend name
	ModelConfig sim.ModelConfig   // HF-derived model architecture config
	HWConfig    sim.HardwareCalib // hardware calibration config
	AlphaCoeffs []float64         // resolved alpha coefficients (local copy, not package-level)
	BetaCoeffs  []float64         // resolved beta coefficients (local copy, not package-level)
}

// resolveLatencyConfig resolves the latency backend configuration from CLI flags and
// defaults.yaml. It is called by both runCmd and replayCmd to ensure a single code path
// (R23: code path parity). This eliminates the R23 comment-sync markers in replay.go.
//
// What it does:
//   - Normalizes model name to lowercase
//   - Validates gpuMemoryUtilization and blockSizeTokens (used in KV auto-calc)
//   - Applies defaults.yaml for GPU and TP when not set via CLI
//   - Validates alpha/beta coefficients and auto-detects trained-physics mode when coefficients are provided
//   - For roofline/trained-physics: resolves model config folder and
//     hardware config, loads coefficients from defaults.yaml, auto-calculates
//     total-kv-blocks and max-model-len from the HF config
//
// Side effects (package-level vars mutated):
//
//	model, gpu, tensorParallelism, modelConfigFolder, hwConfigPath,
//	totalKVBlocks, maxModelLen
//
// Returns values that cannot be stored as package-level vars (local coeff copies,
// resolved modelConfig/hwConfig structs, backend string).
func resolveLatencyConfig(cmd *cobra.Command) latencyResolution {
	// Work with local copies of coefficient slices. The package-level alphaCoeffs/betaCoeffs
	// hold Cobra-registered CLI defaults; mutating them directly would corrupt Cobra's
	// default-value tracking and break subsequent cmd.Flags().Changed() checks.
	alpha := append([]float64(nil), alphaCoeffs...)
	beta := append([]float64(nil), betaCoeffs...)

	// Normalize model name for consistent lookups (defaults.yaml keys, hf_repo,
	// bundled model_configs/, coefficient matching all use lowercase).
	model = strings.ToLower(model)

	// Validate --latency-model flag
	if !sim.IsValidLatencyBackend(latencyModelBackend) {
		logrus.Fatalf("unknown --latency-model %q; valid options: %s",
			latencyModelBackend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
	}
	backend := latencyModelBackend

	// Alpha and beta coefficients must be provided together or not at all.
	alphaChanged := cmd.Flags().Changed("alpha-coeffs")
	betaChanged := cmd.Flags().Changed("beta-coeffs")
	if alphaChanged != betaChanged {
		if alphaChanged {
			logrus.Fatalf("--alpha-coeffs requires --beta-coeffs. Both coefficient sets are needed for coefficient-based estimation")
		}
		logrus.Fatalf("--beta-coeffs requires --alpha-coeffs. Both coefficient sets are needed for coefficient-based estimation")
	}
	for i, c := range alpha {
		if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
			logrus.Fatalf("--alpha-coeffs[%d] must be a finite non-negative number, got %v", i, c)
		}
	}
	for i, c := range beta {
		if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
			logrus.Fatalf("--beta-coeffs[%d] must be a finite non-negative number, got %v", i, c)
		}
	}
	if !cmd.Flags().Changed("latency-model") && alphaChanged && betaChanged {
		backend = "trained-physics"
		logrus.Infof("--alpha-coeffs and --beta-coeffs provided; using trained-physics mode")
	}

	// Validate flags consumed inside this function before any KV auto-calc.
	// gpuMemoryUtilization and blockSizeTokens are used in CalculateKVBlocks;
	// validate here so errors are caught before computation rather than silently
	// producing wrong results.
	if gpuMemoryUtilization <= 0 || gpuMemoryUtilization > 1.0 || math.IsNaN(gpuMemoryUtilization) || math.IsInf(gpuMemoryUtilization, 0) {
		logrus.Fatalf("--gpu-memory-utilization must be a finite value in (0, 1.0], got %f", gpuMemoryUtilization)
	}
	if blockSizeTokens <= 0 {
		logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
	}

	var modelConfig sim.ModelConfig
	var hwConfig sim.HardwareCalib

	// Early defaults resolution: load hardware/TP from defaults.yaml
	// when not explicitly set via CLI flags.
	if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
		hardware, tp := GetDefaultSpecs(model)
		if tensorParallelism == 0 && tp > 0 {
			logrus.Warnf("Finding default values of TP for model=%v", model)
			logrus.Warnf("Using default tp=%v", tp)
			tensorParallelism = tp
		}
		if gpu == "" && len(hardware) > 0 {
			logrus.Warnf("Finding default values of hardware for model=%v", model)
			logrus.Warnf("Using default GPU=%v", hardware)
			gpu = hardware
		}
	}

	// --latency-model roofline
	if backend == "roofline" {
		var missing []string
		if gpu == "" {
			missing = append(missing, "--hardware (GPU type)")
		}
		if tensorParallelism <= 0 {
			missing = append(missing, "--tp (tensor parallelism)")
		}
		if len(missing) > 0 {
			logrus.Fatalf("Roofline mode requires %s. No defaults found in defaults.yaml for model=%s. "+
				"Provide these flags explicitly, or use --latency-model trained-physics for coefficient-based estimation",
				strings.Join(missing, " and "), model)
		}
		// alphaChanged == betaChanged is guaranteed by the "both or neither" check above,
		// so checking betaChanged alone is sufficient to confirm both were provided.
		if cmd.Flags().Changed("latency-model") && betaChanged {
			logrus.Fatalf("--alpha-coeffs/--beta-coeffs cannot be used with --latency-model roofline. " +
				"Roofline computes step time analytically. " +
				"Use --latency-model trained-physics if you want coefficient-based estimation")
		}
		if modelConfigFolder != "" {
			logrus.Infof("--latency-model: explicit --model-config-folder takes precedence over auto-resolution")
		}
		if hwConfigPath != "" {
			logrus.Infof("--latency-model: explicit --hardware-config takes precedence over auto-resolution")
		}
		resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		modelConfigFolder = resolved
		resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		hwConfigPath = resolvedHW
	}

	// --latency-model trained-physics: physics-informed roofline with architecture-aware MoE overhead.
	// Uses trained_physics_coefficients from defaults.yaml (10-beta, 3-alpha).
	if backend == "trained-physics" {
		var missing []string
		if gpu == "" {
			missing = append(missing, "--hardware (GPU type)")
		}
		if tensorParallelism <= 0 {
			missing = append(missing, "--tp (tensor parallelism)")
		}
		if len(missing) > 0 {
			logrus.Fatalf("--latency-model trained-physics requires %s. No defaults found in defaults.yaml for model=%s. "+
				"Provide these flags explicitly", strings.Join(missing, " and "), model)
		}
		resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		modelConfigFolder = resolved
		resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		hwConfigPath = resolvedHW
		if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
			data, readErr := os.ReadFile(defaultsFilePath)
			if readErr != nil {
				logrus.Warnf("--latency-model trained-physics: failed to read %s: %v", defaultsFilePath, readErr)
			} else {
				var cfg Config
				decoder := yaml.NewDecoder(bytes.NewReader(data))
				decoder.KnownFields(true) // R10: strict YAML parsing
				if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
					logrus.Fatalf("--latency-model trained-physics: failed to parse %s: %v", defaultsFilePath, yamlErr)
				}
				if cfg.TrainedPhysicsDefaults != nil {
					if !cmd.Flags().Changed("beta-coeffs") {
						beta = cfg.TrainedPhysicsDefaults.BetaCoeffs
						logrus.Infof("--latency-model: loaded trained-physics beta coefficients from defaults.yaml")
					}
					if !cmd.Flags().Changed("alpha-coeffs") {
						alpha = cfg.TrainedPhysicsDefaults.AlphaCoeffs
						logrus.Infof("--latency-model: loaded trained-physics alpha coefficients from defaults.yaml")
					}
				}
			}
		}
		// Validate trained-physics coefficients: at least 7 beta required (8th+ optional).
		if !cmd.Flags().Changed("beta-coeffs") && (len(beta) < 7 || allZeros(beta)) {
			logrus.Fatalf("--latency-model trained-physics: no trained_physics_coefficients found in %s and no --beta-coeffs provided. "+
				"Add trained_physics_coefficients to defaults.yaml or provide --beta-coeffs explicitly", defaultsFilePath)
		}
		if allZeros(alpha) && !cmd.Flags().Changed("alpha-coeffs") {
			logrus.Warnf("--latency-model trained-physics: no trained-physics alpha coefficients found; " +
				"QueueingTime, PostDecodeFixedOverhead, and OutputTokenProcessingTime will use zero alpha (may underestimate TTFT/E2E)")
		}
	}

	// Analytical backends: parse HF config, extract model/hardware config, auto-calc KV blocks and max-model-len.
	if backend == "roofline" || backend == "trained-physics" {
		hfPath := filepath.Join(modelConfigFolder, "config.json")
		hfConfig, err := latency.ParseHFConfig(hfPath)
		if err != nil {
			logrus.Fatalf("Failed to parse HuggingFace config: %v", err)
		}
		mc, err := latency.GetModelConfigFromHF(hfConfig)
		if err != nil {
			logrus.Fatalf("Failed to load model config: %v", err)
		}
		modelConfig = *mc
		hc, err := latency.GetHWConfig(hwConfigPath, gpu)
		if err != nil {
			logrus.Fatalf("Failed to load hardware config: %v", err)
		}
		hwConfig = hc

		applyWeightPrecisionFallback(&modelConfig, model, hfConfig.Raw)

		if backend == "roofline" && modelConfig.IsMoE() {
			logrus.Infof("--latency-model: MoE model detected (%d experts, top_%d). "+
				"Roofline models per-expert FLOPs and active weights; dispatch overhead is not modeled",
				modelConfig.NumLocalExperts, modelConfig.NumExpertsPerTok)
		}

		// KV capacity auto-calculation. Precedence: (1) --total-kv-blocks CLI flag,
		// (2) auto-calculate from model architecture + GPU memory, (3) default value.
		if !cmd.Flags().Changed("total-kv-blocks") {
			kvParams, kvParamsErr := latency.ExtractKVCapacityParams(hfConfig)
			if kvParamsErr != nil {
				logrus.Warnf("--latency-model: could not extract KV capacity params: %v. "+
					"Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", kvParamsErr, totalKVBlocks)
			} else if hwConfig.MemoryGiB <= 0 {
				logrus.Warnf("--latency-model: GPU memory capacity not available in hardware config; "+
					"using current total-kv-blocks=%d. Add MemoryGiB to hardware_config.json or pass --total-kv-blocks explicitly", totalKVBlocks)
			} else {
				if kvParams.HiddenAct == "" {
					logrus.Infof("--latency-model: hidden_act not set in config.json; assuming SwiGLU (3-matrix MLP) for weight estimation")
				}
				autoBlocks, calcErr := latency.CalculateKVBlocks(modelConfig, hwConfig, tensorParallelism, dataParallelism, blockSizeTokens, gpuMemoryUtilization, kvParams)
				if calcErr != nil {
					logrus.Fatalf("--latency-model: KV capacity auto-calculation failed: %v", calcErr)
				}
				totalKVBlocks = autoBlocks
				logrus.Infof("--gpu-memory-utilization: %.2f used for KV block auto-calculation", gpuMemoryUtilization)
				logrus.Infof("--latency-model: auto-calculated total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, DP=%d, block_size=%d, MoE=%v)",
					totalKVBlocks, hwConfig.MemoryGiB, tensorParallelism, dataParallelism, blockSizeTokens, kvParams.IsMoE)
			}
		}

		// Auto-derive --max-model-len from HF config's max_position_embeddings.
		// Skipped when --max-model-len is explicitly set (R18: CLI flags take precedence).
		if !cmd.Flags().Changed("max-model-len") {
			maxPosEmb := hfConfig.MustGetInt("max_position_embeddings", 0)
			if maxPosEmb > 0 {
				maxModelLen = int64(maxPosEmb)
				modelType, _ := hfConfig.Raw["model_type"].(string)
				scaled, applied := applyRopeScaling(maxPosEmb, modelType, hfConfig.Raw["rope_scaling"])
				if applied {
					ropeType := ""
					factor := 0.0
					if ropeMap, ok := hfConfig.Raw["rope_scaling"].(map[string]any); ok {
						ropeType, _ = ropeMap["type"].(string)
						if ropeType == "" {
							ropeType, _ = ropeMap["rope_type"].(string)
						}
						factor, _ = ropeMap["factor"].(float64)
					}
					logrus.Infof("--latency-model: applying %s rope_scaling factor %.1f: %d → %d", ropeType, factor, maxPosEmb, scaled)
					maxModelLen = int64(scaled)
				} else if strings.Contains(modelType, "gemma3") {
					logrus.Infof("--latency-model: skipping rope_scaling for gemma3 (max_position_embeddings is pre-scaled)")
				} else if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok && ropeScaling != nil {
					if ropeMap, ok := ropeScaling.(map[string]any); ok {
						if _, hasKey := ropeMap["factor"]; hasKey {
							logrus.Warnf("--latency-model: rope_scaling.factor present but not applied (excluded type, invalid value, or overflow); using max_position_embeddings as-is")
						}
					} else {
						logrus.Warnf("--latency-model: rope_scaling present but not a JSON object (type %T); ignoring", ropeScaling)
					}
				}
				logrus.Infof("--latency-model: auto-derived max-model-len=%d from max_position_embeddings", maxModelLen)
			}
		}

		// Cap maxModelLen at KV-feasible maximum (matches vLLM's _maybe_limit_model_len).
		if maxModelLen > 0 && blockSizeTokens > 0 {
			blocksNeeded := maxModelLen / blockSizeTokens
			if maxModelLen%blockSizeTokens != 0 {
				blocksNeeded++
			}
			if blocksNeeded > totalKVBlocks {
				kvFeasibleMax := totalKVBlocks * blockSizeTokens
				logrus.Warnf("--latency-model: max-model-len %d exceeds KV capacity (%d blocks × %d tokens); capping to %d tokens",
					maxModelLen, totalKVBlocks, blockSizeTokens, kvFeasibleMax)
				maxModelLen = kvFeasibleMax
			}
		}
	}

	if maxModelLen < 0 {
		logrus.Fatalf("--max-model-len must be >= 0, got %d", maxModelLen)
	}

	// DP/EP validation (single shared path for run and replay → INV-13 parity).
	// Mirrors the construction-time panics in sim.NewModelHardwareConfig but emits
	// clean CLI errors. modelConfig.NumLocalExperts is populated for the analytical
	// backends above; it is 0 (dense-equivalent) when no model config was resolved.
	if dataParallelism < 1 {
		logrus.Fatalf("--dp must be >= 1, got %d", dataParallelism)
	}
	if (dataParallelism > 1 || enableExpertParallel) && backend != "trained-physics" {
		// INV BC-ROOFLINE: roofline does not model DP/EP step-time effects, so
		// accepting these flags there would imply unsupported latency semantics.
		logrus.Fatalf("--dp > 1 and --enable-expert-parallel require --latency-model trained-physics "+
			"(got --dp=%d, --enable-expert-parallel=%t, --latency-model %s). The roofline backend is "+
			"DP/EP-blind for step time.",
			dataParallelism, enableExpertParallel, backend)
	}
	if dataParallelism > 1 && !modelConfig.IsMoE() {
		// Dense DP is the router-replica mechanism, not a latency divisor. vLLM
		// online serving allows --data-parallel-size > 1 on dense models by
		// spinning up N independent engines with an internal LB — the BLIS
		// equivalent is N instances via --num-instances.
		logrus.Fatalf("--dp > 1 is only supported for MoE models (got --dp=%d for a dense model). "+
			"In vLLM online serving, --data-parallel-size on a dense model creates N independent engines; "+
			"the BLIS equivalent is --num-instances (router replicas), not --dp.",
			dataParallelism)
	}
	if enableExpertParallel && !modelConfig.IsMoE() {
		// vLLM fatally rejects --enable-expert-parallel on dense models
		// (config/model.py:_verify_with_expert_parallelism raises ValueError,
		// killing the process before serving). Match that signal so BLIS doesn't
		// simulate a deployment that cannot exist in production.
		logrus.Fatalf("--enable-expert-parallel requires a MoE model (got dense model %q with no experts); "+
			"vLLM fatally rejects this configuration.", model)
	}
	// --moe-comm-backend selects the MoE dispatch/combine cost model (trained-physics
	// only, DP>1). Validate the name and reject non-default values on other backends so
	// the flag never silently no-ops. Empty string defers to the model factory's default.
	if moeCommBackend != "" {
		if !latency.IsValidMoECommBackend(moeCommBackend) {
			logrus.Fatalf("--moe-comm-backend %q is not a recognized vLLM MoE all-to-all backend (valid: %s).",
				moeCommBackend, strings.Join(latency.ValidMoECommBackends, ", "))
		}
		if backend != "trained-physics" {
			logrus.Fatalf("--moe-comm-backend requires --latency-model trained-physics "+
				"(got --moe-comm-backend=%s, --latency-model=%s). The roofline backend does not model "+
				"MoE communication.", moeCommBackend, backend)
		}
		// The flag is harmless but inert unless the MoE dispatch/combine term is actually
		// charged (isMoE && DP>1). Warn (not fatal) on the no-op cases so a user does not
		// believe a backend choice is affecting a run where it cannot.
		if !modelConfig.IsMoE() {
			logrus.Warnf("--moe-comm-backend=%s has no effect on a dense model; "+
				"MoE dispatch/combine comm is only charged for MoE models.", moeCommBackend)
		} else if dataParallelism <= 1 {
			logrus.Warnf("--moe-comm-backend=%s has no effect at --dp=%d; "+
				"MoE dispatch/combine comm is only charged when DP > 1 (at DP=1 the MoE FFN "+
				"all-reduces over the TP group instead).", moeCommBackend, dataParallelism)
		}
	}

	return latencyResolution{
		Backend:     backend,
		ModelConfig: modelConfig,
		HWConfig:    hwConfig,
		AlphaCoeffs: alpha,
		BetaCoeffs:  beta,
	}
}

// resolvePolicies resolves admission/routing/priority/scheduler policy configuration
// from CLI flags and an optional policy bundle YAML file. It is called by both runCmd
// and replayCmd to ensure a single validation code path (R23: code path parity).
//
// Precondition: resolveLatencyConfig must be called first. gpuMemoryUtilization and
// blockSizeTokens are validated there (before KV auto-calc); resolvePolicies does not
// re-validate them. Calling resolvePolicies without a prior resolveLatencyConfig call
// would bypass those validations.
//
// Side effects: may write admissionPolicy, routingPolicy, scheduler,
// tokenBucketCapacity, tokenBucketRefillRate, tierShedThreshold, tierShedMinPriority,
// tenantBudgets package-level vars (from policy bundle).
//
// Returns the parsed scorer configs for weighted routing (caller uses these in
// DeploymentConfig.RoutingScorerConfigs) and the loaded policy bundle (nil if none).
// Per-pool scorer configs (PD disaggregation) are NOT handled here — they remain inline
// in runCmd.
func resolvePolicies(cmd *cobra.Command) ([]sim.ScorerConfig, *sim.PolicyBundle) {
	var bundleScorerConfigs []sim.ScorerConfig
	var loadedBundle *sim.PolicyBundle

	// Load policy bundle if specified (R18: CLI flags override YAML values)
	if policyConfigPath != "" {
		bundle, err := sim.LoadPolicyBundle(policyConfigPath)
		if err != nil {
			logrus.Fatalf("Failed to load policy config: %v", err)
		}
		if err := bundle.Validate(); err != nil {
			logrus.Fatalf("Invalid policy config: %v", err)
		}
		loadedBundle = bundle
		// Apply bundle values as defaults; CLI flags override via Changed().
		if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
			admissionPolicy = bundle.Admission.Policy
		}
		if bundle.Admission.TokenBucketCapacity != nil && !cmd.Flags().Changed("token-bucket-capacity") {
			tokenBucketCapacity = *bundle.Admission.TokenBucketCapacity
		}
		if bundle.Admission.TokenBucketRefillRate != nil && !cmd.Flags().Changed("token-bucket-refill-rate") {
			tokenBucketRefillRate = *bundle.Admission.TokenBucketRefillRate
		}
		if bundle.Admission.TierShedThreshold != nil {
			tierShedThreshold = *bundle.Admission.TierShedThreshold
		}
		if bundle.Admission.TierShedMinPriority != nil {
			tierShedMinPriority = *bundle.Admission.TierShedMinPriority
		} else if bundle.Admission.Policy == "tier-shed" && bundle.Admission.TierShedMinPriority == nil {
			tierShedMinPriority = 3 // default: protect Critical (4) and Standard (3)
		}
		if bundle.TenantBudgets != nil {
			tenantBudgets = bundle.TenantBudgets
		}
		if bundle.Admission.SLOPriorities != nil {
			sloPriorityOverrides = bundle.Admission.SLOPriorities
			// Validate at CLI boundary (R3) before reaching library-level panic in NewSLOPriorityMap.
			for k, v := range sloPriorityOverrides {
				if v < -100 || v > 100 {
					logrus.Fatalf("slo_priorities override for %q: value %d out of range [-100, 100]", k, v)
				}
			}
		}
		if bundle.Admission.SLOTargets != nil && sloTargetsMap == nil {
			sloTargetsMap = bundle.Admission.SLOTargets
		}
		if bundle.Admission.GAIEQDThreshold != nil {
			gaieQDThreshold = *bundle.Admission.GAIEQDThreshold
		}
		if bundle.Admission.GAIEKVThreshold != nil {
			gaieKVThreshold = *bundle.Admission.GAIEKVThreshold
		}
		if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
			routingPolicy = bundle.Routing.Policy
		}
		bundleScorerConfigs = bundle.Routing.Scorers
		if bundle.Priority.Policy != "" && bundle.Priority.Policy != "constant" {
			logrus.Warnf("bundle priority.policy=%q has no effect: --priority-policy was removed in PR #1216. "+
				"Request priority is now static (set at enqueue via SLOPriorityMap.InvertForVLLM). "+
				"Use SLOClass in your workload spec for tier-based priority.", bundle.Priority.Policy)
		}
		if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
			scheduler = bundle.Scheduler
		}
		if bundle.Preemption.Policy != "" && !cmd.Flags().Changed("preemption-policy") {
			preemptionPolicy = bundle.Preemption.Policy
		}
	}

	// Apply defaults for GAIE-legacy thresholds (not set via CLI flags, only via bundle).
	if gaieQDThreshold == 0 {
		gaieQDThreshold = 5
	}
	if gaieKVThreshold == 0 {
		gaieKVThreshold = 0.8
	}

	// Policy name validation (R3: validate at CLI boundary before passing to library)
	if admissionPolicy == "token-bucket" {
		if tokenBucketCapacity <= 0 || math.IsNaN(tokenBucketCapacity) || math.IsInf(tokenBucketCapacity, 0) {
			logrus.Fatalf("--token-bucket-capacity must be a finite value > 0, got %v", tokenBucketCapacity)
		}
		if tokenBucketRefillRate <= 0 || math.IsNaN(tokenBucketRefillRate) || math.IsInf(tokenBucketRefillRate, 0) {
			logrus.Fatalf("--token-bucket-refill-rate must be a finite value > 0, got %v", tokenBucketRefillRate)
		}
	}
	if admissionPolicy == "gaie-legacy" {
		if gaieQDThreshold <= 0 || math.IsNaN(gaieQDThreshold) || math.IsInf(gaieQDThreshold, 0) {
			logrus.Fatalf("gaie_qd_threshold must be > 0, got %v", gaieQDThreshold)
		}
		if gaieKVThreshold <= 0 || gaieKVThreshold > 1.0 || math.IsNaN(gaieKVThreshold) || math.IsInf(gaieKVThreshold, 0) {
			logrus.Fatalf("gaie_kv_threshold must be in (0, 1.0], got %v", gaieKVThreshold)
		}
	}
	if !sim.IsValidAdmissionPolicy(admissionPolicy) {
		logrus.Fatalf("Unknown admission policy %q. Valid: %s", admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
	}
	if !sim.IsValidRoutingPolicy(routingPolicy) {
		logrus.Fatalf("Unknown routing policy %q. Valid: %s", routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
	}
	if !sim.IsValidScheduler(scheduler) {
		logrus.Fatalf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
	}
	if !sim.IsValidPreemptionPolicy(preemptionPolicy) {
		logrus.Fatalf("Unknown preemption policy %q. Valid: %s", preemptionPolicy, strings.Join(sim.ValidPreemptionPolicyNames(), ", "))
	}
	if !trace.IsValidTraceLevel(traceLevel) {
		logrus.Fatalf("Unknown trace level %q. Valid: none, decisions", traceLevel)
	}
	if counterfactualK < 0 {
		logrus.Fatalf("--counterfactual-k must be >= 0, got %d", counterfactualK)
	}
	if traceLevel == "none" && counterfactualK > 0 {
		logrus.Warnf("--counterfactual-k=%d has no effect without --trace-level decisions", counterfactualK)
	}
	if traceLevel == "none" && summarizeTrace {
		logrus.Warnf("--summarize-trace has no effect without --trace-level decisions")
	}
	if traceLevel != "none" && !summarizeTrace {
		logrus.Infof("Decision tracing enabled (trace-level=%s). Use --summarize-trace to print summary.", traceLevel)
	}
	if kvCPUBlocks < 0 {
		logrus.Fatalf("--kv-cpu-blocks must be >= 0, got %d", kvCPUBlocks)
	}
	if kvOffloadThreshold < 0 || kvOffloadThreshold > 1 || math.IsNaN(kvOffloadThreshold) || math.IsInf(kvOffloadThreshold, 0) {
		logrus.Fatalf("--kv-offload-threshold must be a finite value in [0, 1], got %f", kvOffloadThreshold)
	}
	// Note: gpuMemoryUtilization and blockSizeTokens are validated in resolveLatencyConfig
	// (before KV auto-calc). Not repeated here to avoid double-validation.
	if kvCPUBlocks > 0 && (kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0)) {
		logrus.Fatalf("--kv-transfer-bandwidth must be a finite value > 0 when --kv-cpu-blocks > 0, got %f", kvTransferBandwidth)
	}
	if kvTransferBaseLatency < 0 {
		logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
	}
	if snapshotRefreshInterval < 0 {
		logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
	}
	if cacheSignalDelay < 0 {
		logrus.Fatalf("--cache-signal-delay must be >= 0, got %d", cacheSignalDelay)
	}
	if admissionLatency < 0 {
		logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
	}
	if routingLatency < 0 {
		logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
	}
	// Flow control validation (R3: validate at CLI boundary before passing to library)
	if flowControlEnabled {
		if !sim.IsValidSaturationDetector(flowControlDetector) {
			logrus.Fatalf("Unknown saturation detector %q. Valid: %s", flowControlDetector, strings.Join(sim.ValidSaturationDetectorNames(), ", "))
		}
		if flowControlDispatchOrder != "fifo" && flowControlDispatchOrder != "priority" && flowControlDispatchOrder != "slo-deadline" {
			logrus.Fatalf("--dispatch-order must be 'fifo', 'priority', or 'slo-deadline', got %q", flowControlDispatchOrder)
		}
		if flowControlMaxQueueDepth < 0 {
			logrus.Fatalf("--max-gateway-queue-depth must be >= 0, got %d", flowControlMaxQueueDepth)
		}
		if flowControlPerBandCapacity < 0 {
			logrus.Fatalf("--per-band-capacity must be >= 0, got %d", flowControlPerBandCapacity)
		}
		if flowControlUsageLimitThreshold <= 0 || flowControlUsageLimitThreshold > 1.0 {
			logrus.Fatalf("--usage-limit-threshold must be in (0, 1.0], got %v", flowControlUsageLimitThreshold)
		}
		if flowControlFairnessPolicy != "global-strict" && flowControlFairnessPolicy != "round-robin" {
			logrus.Fatalf("--fairness-policy must be 'global-strict' or 'round-robin', got %q", flowControlFairnessPolicy)
		}
		if flowControlUsageLimitThreshold < 1.0 && flowControlDispatchOrder == "fifo" {
			logrus.Warnf("--usage-limit-threshold < 1.0 with --dispatch-order fifo: HoL blocking uses priority-order iteration, FIFO semantics will not apply to gating decisions")
		}
		// Parse --slo-targets into map (e.g. "critical=100000,standard=500000")
		if flowControlSLOTargets != "" {
			sloTargetsMap = make(map[string]int64)
			for _, pair := range strings.Split(flowControlSLOTargets, ",") {
				parts := strings.SplitN(strings.TrimSpace(pair), "=", 2)
				if len(parts) != 2 {
					logrus.Fatalf("--slo-targets: invalid format %q (expected key=value)", pair)
				}
				key := strings.TrimSpace(parts[0])
				if key == "" {
					logrus.Fatalf("--slo-targets: empty key in pair %q (expected key=value)", pair)
				}
				v, parseErr := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
				if parseErr != nil || v <= 0 {
					logrus.Fatalf("--slo-targets: value for %q must be a positive integer (µs), got %s", key, strings.TrimSpace(parts[1]))
				}
				sloTargetsMap[key] = v
			}
		}
		if sloTargetsMap != nil && flowControlDispatchOrder != "slo-deadline" {
			logrus.Warnf("--slo-targets has no effect without --dispatch-order slo-deadline")
		}
		// Validate only parameters consumed by the selected detector
		switch flowControlDetector {
		case "utilization":
			if flowControlQueueDepthThreshold <= 0 || math.IsNaN(flowControlQueueDepthThreshold) || math.IsInf(flowControlQueueDepthThreshold, 0) {
				logrus.Fatalf("--queue-depth-threshold must be a finite value > 0, got %v", flowControlQueueDepthThreshold)
			}
			if flowControlKVCacheUtilThreshold <= 0 || math.IsNaN(flowControlKVCacheUtilThreshold) || math.IsInf(flowControlKVCacheUtilThreshold, 0) {
				logrus.Fatalf("--kv-cache-util-threshold must be a finite value > 0, got %v", flowControlKVCacheUtilThreshold)
			}
		case "concurrency":
			if flowControlMaxConcurrency <= 0 {
				logrus.Fatalf("--max-concurrency must be > 0, got %d", flowControlMaxConcurrency)
			}
		case "", "never":
			logrus.Warnf("--flow-control enabled but --saturation-detector is %q (pass-through); specify 'utilization' or 'concurrency' for actual gating", flowControlDetector)
		}
	}
	if flowControlRequestTTL < 0 {
		logrus.Fatalf("--request-ttl must be >= 0, got %d", flowControlRequestTTL)
	}
	if flowControlRequestTTL > 0 && !flowControlEnabled {
		logrus.Warnf("--request-ttl %d has no effect without --flow-control", flowControlRequestTTL)
	}
	if flowControlDispatchTickInterval < 0 {
		logrus.Fatalf("--dispatch-tick-interval must be >= 0, got %d", flowControlDispatchTickInterval)
	}
	if flowControlQueueShedding && !flowControlEnabled {
		logrus.Warnf("--queue-shedding has no effect without --flow-control")
	}
	if flowControlInFlightEviction && !flowControlEnabled {
		logrus.Warnf("--in-flight-eviction has no effect without --flow-control")
	}

	logrus.Infof("Policy config: admission=%s, routing=%s, scheduler=%s, preemption=%s",
		admissionPolicy, routingPolicy, scheduler, preemptionPolicy)

	// Parse scorer configuration for weighted routing
	var parsedScorerConfigs []sim.ScorerConfig
	if routingPolicy == "weighted" {
		if routingScorers != "" {
			var err error
			parsedScorerConfigs, err = sim.ParseScorerConfigs(routingScorers)
			if err != nil {
				logrus.Fatalf("Invalid --routing-scorers: %v", err)
			}
		} else if len(bundleScorerConfigs) > 0 {
			parsedScorerConfigs = bundleScorerConfigs
		}
		activeScorerConfigs := parsedScorerConfigs
		if len(activeScorerConfigs) == 0 {
			activeScorerConfigs = sim.DefaultScorerConfigs()
		}
		scorerStrs := make([]string, len(activeScorerConfigs))
		for i, sc := range activeScorerConfigs {
			scorerStrs[i] = fmt.Sprintf("%s:%.1f", sc.Name, sc.Weight)
		}
		logrus.Infof("Weighted routing scorers: %s", strings.Join(scorerStrs, ", "))
	}
	if routingPolicy != "weighted" && routingScorers != "" {
		logrus.Warnf("--routing-scorers has no effect when routing policy is %q (only applies to 'weighted')", routingPolicy)
	}
	if admissionPolicy == "token-bucket" {
		logrus.Infof("Token bucket: capacity=%.0f, refill-rate=%.0f", tokenBucketCapacity, tokenBucketRefillRate)
	}

	return parsedScorerConfigs, loadedBundle
}

// registerSimConfigFlags registers all simulation-engine configuration flags
// on the given command. Called by both runCmd and replayCmd to avoid
// duplicating ~50 flag registrations.
func registerSimConfigFlags(cmd *cobra.Command) {
	cmd.Flags().Int64Var(&seed, "seed", 42, "Seed for random request generation")
	cmd.Flags().Int64Var(&simulationHorizon, "horizon", math.MaxInt64, "Total simulation horizon (in ticks)")
	cmd.Flags().StringVar(&logLevel, "log", "warn", "Log level for diagnostic messages (trace, debug, info, warn, error, fatal, panic). Simulation results always print to stdout regardless of this setting.")
	cmd.Flags().StringVar(&defaultsFilePath, "defaults-filepath", "defaults.yaml", "Path to default constants - trained coefficients, default specs and workloads")
	cmd.Flags().StringVar(&modelConfigFolder, "model-config-folder", "", "Path to folder containing config.json")
	cmd.Flags().StringVar(&hwConfigPath, "hardware-config", "", "Path to file containing hardware config")

	// vLLM server configs
	cmd.Flags().Int64Var(&totalKVBlocks, "total-kv-blocks", 1000000, "Total number of KV cache blocks")
	cmd.Flags().Int64Var(&maxRunningReqs, "max-num-running-reqs", 256, "Maximum number of requests running together")
	cmd.Flags().Int64Var(&maxScheduledTokens, "max-num-scheduled-tokens", 2048, "Maximum total number of new tokens across running requests")
	cmd.Flags().Float64SliceVar(&betaCoeffs, "beta-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated list of beta coefficients")
	cmd.Flags().Float64SliceVar(&alphaCoeffs, "alpha-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated alpha coefficients (alpha0,alpha1) for processing delays")
	cmd.Flags().Int64Var(&blockSizeTokens, "block-size-in-tokens", 16, "Number of tokens contained in a KV cache block")
	cmd.Flags().Int64Var(&longPrefillTokenThreshold, "long-prefill-token-threshold", 0, "Max length of prefill beyond which chunked prefill is triggered")

	// BLIS model configs
	cmd.Flags().StringVar(&model, "model", "", "LLM name")
	cmd.Flags().StringVar(&gpu, "hardware", "", "GPU type")
	cmd.Flags().IntVar(&tensorParallelism, "tp", 0, "Tensor parallelism")
	cmd.Flags().IntVar(&dataParallelism, "dp", 1, "Data parallelism degree (MoE models only; --latency-model trained-physics only)")
	cmd.Flags().BoolVar(&enableExpertParallel, "enable-expert-parallel", false, "Enable expert parallelism for MoE models (mirrors vLLM --enable-expert-parallel; --latency-model trained-physics only)")
	cmd.Flags().StringVar(&moeCommBackend, "moe-comm-backend", "", "MoE all-to-all comm backend for dispatch/combine cost (mirrors vLLM VLLM_ALL2ALL_BACKEND: naive, allgather_reducescatter [default], pplx, deepep_high_throughput, deepep_low_latency, mori, flashinfer_all2allv; MoE + --latency-model trained-physics + --dp > 1)")
	cmd.Flags().StringVar(&latencyModelBackend, "latency-model", "trained-physics", "Latency model backend: trained-physics (default), roofline")
	cmd.Flags().Int64Var(&maxModelLen, "max-model-len", 0, "Max total sequence length (input + output); 0 = unlimited. Auto-derived from HF config for analytical backends when not set.")

	// Cluster config
	cmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

	// Online routing pipeline config
	cmd.Flags().StringVar(&admissionPolicy, "admission-policy", "always-admit", "Admission policy: "+strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
	cmd.Flags().Int64Var(&admissionLatency, "admission-latency", 0, "Admission latency in microseconds")
	cmd.Flags().Int64Var(&routingLatency, "routing-latency", 0, "Routing latency in microseconds")
	cmd.Flags().Float64Var(&tokenBucketCapacity, "token-bucket-capacity", 10000, "Token bucket capacity")
	cmd.Flags().Float64Var(&tokenBucketRefillRate, "token-bucket-refill-rate", 1000, "Token bucket refill rate (tokens/second)")

	// Routing policy config
	cmd.Flags().StringVar(&routingPolicy, "routing-policy", "round-robin", "Routing policy: round-robin, least-loaded, weighted, always-busiest")
	cmd.Flags().StringVar(&routingScorers, "routing-scorers", "", "Scorer weights for weighted routing (e.g., queue-depth:2,kv-utilization:2,load-balance:1). Default: precise-prefix-cache:2,queue-depth:1,kv-utilization:1")

	// Scheduler and preemption config
	cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")
	cmd.Flags().StringVar(&preemptionPolicy, "preemption-policy", "fcfs", "Preemption victim selection: fcfs (tail-of-batch), priority (least-urgent SLO tier)")

	// Policy bundle config
	cmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")

	// Fitness evaluation config (PR9)
	cmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs (e.g., throughput:0.5,p99_ttft:0.3)")

	// Decision trace config (PR13)
	cmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
	cmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Number of counterfactual candidates per routing decision")
	cmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")

	// Tiered KV cache (PR12)
	cmd.Flags().Int64Var(&kvCPUBlocks, "kv-cpu-blocks", 0, "CPU tier KV cache blocks (0 = disabled, single-tier mode). Typical: 1/3 of --total-kv-blocks")
	cmd.Flags().Float64Var(&kvOffloadThreshold, "kv-offload-threshold", 0.9, "GPU utilization (0-1) above which blocks are offloaded to CPU. Default: offload when GPU >90% full")
	cmd.Flags().Float64Var(&kvTransferBandwidth, "kv-transfer-bandwidth", 100.0, "CPU↔GPU transfer rate in blocks per tick. Higher = faster transfers")
	cmd.Flags().Int64Var(&kvTransferBaseLatency, "kv-transfer-base-latency", 0, "Fixed per-transfer latency in ticks for CPU↔GPU KV transfers (0 = no fixed cost)")
	cmd.Flags().Int64Var(&snapshotRefreshInterval, "snapshot-refresh-interval", 50000, "Prometheus snapshot refresh interval for all instance metrics in microseconds (0 = immediate/oracle mode, default 50ms = llm-d parity)")
	cmd.Flags().Int64Var(&cacheSignalDelay, "cache-signal-delay", cluster.DefaultCacheSignalDelay, "Propagation delay for prefix cache signals in microseconds. Only affects precise-prefix-cache and no-hit-lru scorers; no effect on other routing policies. Default 50ms. Set to 0 for oracle mode (live cache state).")
	cmd.Flags().Float64Var(&modelAutoscalerIntervalUs, "model-autoscaler-interval-us", 0, "Autoscaler tick interval in microseconds (0 = disabled). Overrides policy-config autoscaler.interval_us when non-zero.")
	cmd.Flags().Float64Var(&gpuMemoryUtilization, "gpu-memory-utilization", 0.9, "Fraction of GPU memory to use for KV cache, in the range (0, 1.0]. Default: 0.9 (90%)")

	// PD disaggregation config
	cmd.Flags().IntVar(&prefillInstances, "prefill-instances", 0, "Number of instances dedicated to prefill (0 = disabled)")
	cmd.Flags().IntVar(&decodeInstances, "decode-instances", 0, "Number of instances dedicated to decode (0 = disabled)")
	cmd.Flags().IntVar(&prefillDecodeInstances, "prefill-decode-instances", 0, "Number of shared-role instances serving both prefill and decode (llm-d 'prefill-decode'/'both' parity; 0 = disabled). Must satisfy --prefill-instances + --decode-instances + --prefill-decode-instances <= --num-instances.")
	cmd.Flags().StringVar(&pdDecider, "pd-decider", "never", "PD disaggregation decider: never (default), always, prefix-threshold")
	cmd.Flags().Float64Var(&pdTransferBandwidth, "pd-transfer-bandwidth", 25.0, "PD KV transfer bandwidth in GB/s (NIXL RDMA default)")
	cmd.Flags().Float64Var(&pdTransferBaseLatency, "pd-transfer-base-latency", 0.05, "PD KV transfer base latency in ms")
	cmd.Flags().BoolVar(&pdTransferContention, "pd-transfer-contention", false, "Enable fair-share bandwidth contention model for concurrent KV transfers (INV-P2-2)")
	cmd.Flags().IntVar(&pdPrefixThreshold, "pd-prefix-threshold", 16, "Non-cached token threshold for prefix-threshold decider (>= 0); disaggregate when non-cached tokens exceed this value. Default 16 matches llm-d's shipped P/D configs (deploy/config/pd-epp-config.yaml).")
	cmd.Flags().StringVar(&prefillRoutingScorers, "prefill-routing-scorers", "", "Scorer weights for prefill pool routing (e.g., queue-depth:2,kv-utilization:2)")
	cmd.Flags().StringVar(&decodeRoutingScorers, "decode-routing-scorers", "", "Scorer weights for decode pool routing (e.g., queue-depth:2,kv-utilization:2)")

	// E/P/D disaggregation (GAP-4, issue #1264). Registered on both run and replay.
	cmd.Flags().IntVar(&encodeInstances, "encode-instances", 0, "Number of instances dedicated to encoding multimodal input (0 = encode pool disabled, default)")
	cmd.Flags().StringVar(&encodeDecider, "encode-decider", "never", "Encode decider: never (default), always, multimodal")

	// Flow control config (issue #882, GIE parity)
	cmd.Flags().BoolVar(&flowControlEnabled, "flow-control", false, "Enable gateway queue with saturation-gated dispatch (GIE flow control)")
	cmd.Flags().StringVar(&flowControlDetector, "saturation-detector", "never", "Saturation detector: "+strings.Join(sim.ValidSaturationDetectorNames(), ", "))
	cmd.Flags().StringVar(&flowControlDispatchOrder, "dispatch-order", "fifo", "Gateway queue dispatch order: fifo, priority, slo-deadline")
	cmd.Flags().StringVar(&flowControlSLOTargets, "slo-targets", "", "Per-SLO-class TTFT targets in µs for slo-deadline ordering (e.g., critical=100000,standard=500000)")
	cmd.Flags().IntVar(&flowControlMaxQueueDepth, "max-gateway-queue-depth", 0, "Max gateway queue depth (0=unlimited)")
	cmd.Flags().Float64Var(&flowControlQueueDepthThreshold, "queue-depth-threshold", 5, "Queue depth threshold for utilization detector")
	cmd.Flags().Float64Var(&flowControlKVCacheUtilThreshold, "kv-cache-util-threshold", 0.8, "KV cache utilization threshold for utilization detector")
	cmd.Flags().IntVar(&flowControlMaxConcurrency, "max-concurrency", 100, "Max concurrency per instance for concurrency detector")
	cmd.Flags().IntVar(&flowControlPerBandCapacity, "per-band-capacity", 0, "Max requests per priority band when --flow-control is enabled (0=unlimited)")
	cmd.Flags().Float64Var(&flowControlUsageLimitThreshold, "usage-limit-threshold", 1.0, "Per-band saturation ceiling for HoL blocking (1.0=no HoL, <1.0 gates lower-priority bands earlier)")
	cmd.Flags().StringVar(&flowControlFairnessPolicy, "fairness-policy", "global-strict", "Intra-band dispatch fairness: global-strict, round-robin")
	cmd.Flags().Int64Var(&flowControlRequestTTL, "request-ttl", 0, "Gateway queue request TTL in microseconds (0=disabled). Requires --flow-control.")
	cmd.Flags().BoolVar(&flowControlQueueShedding, "queue-shedding", false, "Enable cross-band victim shedding when gateway queue is full (BLIS-extra, not in llm-d; default: reject)")
	cmd.Flags().Int64Var(&flowControlDispatchTickInterval, "dispatch-tick-interval", 1000, "Microseconds between periodic gateway dispatch ticks (0 = use default 1ms; llm-d parity)")
	cmd.Flags().BoolVar(&flowControlInFlightEviction, "in-flight-eviction", false, "Enable in-flight eviction of sheddable requests when saturated (BLIS-extra, not in llm-d; requires --flow-control)")

	// Per-pool hardware overrides
	cmd.Flags().IntVar(&prefillTP, "prefill-tp", 0, "Tensor parallelism degree for prefill pool instances (0 = use global --tensor-parallelism)")
	cmd.Flags().IntVar(&decodeTP, "decode-tp", 0, "Tensor parallelism degree for decode pool instances (0 = use global --tensor-parallelism)")
	cmd.Flags().StringVar(&prefillHardware, "prefill-hardware", "", "GPU type for prefill pool instances (\"\" = use global --gpu)")
	cmd.Flags().StringVar(&decodeHardware, "decode-hardware", "", "GPU type for decode pool instances (\"\" = use global --gpu)")
	cmd.Flags().StringVar(&prefillLatencyModel, "prefill-latency-model", "", "Latency model backend for prefill pool instances (\"\" = use global --latency-model)")
	cmd.Flags().StringVar(&decodeLatencyModel, "decode-latency-model", "", "Latency model backend for decode pool instances (\"\" = use global --latency-model)")
	cmd.Flags().Int64Var(&prefillMaxModelLen, "prefill-max-model-len", 0, "Max model length for prefill pool instances (0 = use global --max-model-len)")
	cmd.Flags().Int64Var(&decodeMaxModelLen, "decode-max-model-len", 0, "Max model length for decode pool instances (0 = use global --max-model-len)")

}

// applyTimeoutToSpec sets ClientSpec.Timeout and CohortSpec.Timeout on every entry in spec.
// timeoutSecs>0 converts to µs and sets a deadline; timeoutSecs<=0 sets an explicit *int64(0)
// (disabled). Explicit zero is required so computeDeadline does not fall back to the 300s
// session default. Callers must reject timeoutSecs==0 before calling (use negative to disable).
func applyTimeoutToSpec(spec *workload.WorkloadSpec, timeoutSecs int) {
	var us int64
	if timeoutSecs > 0 {
		us = int64(timeoutSecs) * 1_000_000
	}
	for i := range spec.Clients {
		t := us
		spec.Clients[i].Timeout = &t
	}
	for i := range spec.Cohorts {
		t := us
		spec.Cohorts[i].Timeout = &t
	}
}

// applyTimeoutToRequests re-applies timeout to already-generated requests and session
// blueprints. This corrects deadlines for inference_perf specs: spec.Clients is empty
// when applyTimeoutToSpec runs and is populated inside GenerateWorkload, so the initial
// deadlines are computed from nil Timeout. Safe to call for all spec types.
// timeoutSecs>0 sets deadline=ArrivalTime+timeout; timeoutSecs<=0 sets deadline=0 (disabled).
func applyTimeoutToRequests(wl *workload.GeneratedWorkload, timeoutSecs int) {
	var timeoutUs int64
	if timeoutSecs > 0 {
		timeoutUs = int64(timeoutSecs) * 1_000_000
	}
	for _, req := range wl.Requests {
		if timeoutUs == 0 {
			req.Deadline = 0
		} else {
			req.Deadline = req.ArrivalTime + timeoutUs
		}
	}
	for i := range wl.Sessions {
		t := timeoutUs
		wl.Sessions[i].Timeout = &t
	}
}

// runCmd executes the simulation using parameters from CLI flags
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Run the inference simulation",
	Run: func(cmd *cobra.Command, args []string) {
		// Set up logging
		level, err := logrus.ParseLevel(logLevel)
		if err != nil {
			logrus.Fatalf("Invalid log level: %s", logLevel)
		}
		logrus.SetLevel(level)

		if model == "" { // model not provided, exit
			logrus.Fatalf("LLM name not provided. Exiting simulation.")
		}

		// Resolve latency backend configuration (single code path shared with replayCmd).
		lr := resolveLatencyConfig(cmd)

		// PD disaggregation requires ModelConfig for KV transfer duration derivation.
		// Analytical backends populate ModelConfig from HF config.json.
		// When PD is enabled and ModelConfig is zero-valued, resolve and load it using the
		// same resolution as analytical backends (--model-config-folder → local bundled → HuggingFace fetch → error).
		if prefillInstances > 0 && lr.ModelConfig.NumHeads == 0 {
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("PD disaggregation requires model architecture for KV transfer sizing: %v", err)
			}
			hfPath := filepath.Join(resolved, "config.json")
			hfConfig, parseErr := latency.ParseHFConfig(hfPath)
			if parseErr != nil {
				logrus.Fatalf("PD disaggregation requires model architecture for KV transfer sizing, but failed to parse %s: %v", hfPath, parseErr)
			}
			mc, mcErr := latency.GetModelConfigFromHF(hfConfig)
			if mcErr != nil {
				logrus.Fatalf("PD disaggregation requires model architecture for KV transfer sizing, but failed to extract ModelConfig: %v", mcErr)
			}
			applyWeightPrecisionFallback(mc, model, hfConfig.Raw)
			if mc.BytesPerParam <= 0 {
				logrus.Fatalf("PD disaggregation: could not determine model precision (BytesPerParam=%v) from %s — ensure torch_dtype or dtype is present in config.json", mc.BytesPerParam, hfPath)
			}
			lr.ModelConfig = *mc
			logrus.Infof("PD disaggregation: loaded ModelConfig from %s for KV transfer derivation", hfPath)
		}

		// Per-pool hardware override vars. TotalKVBlocks is populated from per-pool KV
		// auto-calc in the analytical backend block below (when applicable). TP/GPU/Backend/MaxModelLen
		// are populated from CLI flags after PD validation. Both paths are no-ops when disaggregation
		// is disabled (prefillInstances == 0).
		var prefillOverrides, decodeOverrides cluster.PoolOverrides

		// Per-pool KV auto-calculation: when PD disaggregation is active and a pool
		// uses different TP or GPU hardware, compute per-pool KV blocks from model + hardware.
		// Only runs for analytical backends where hardware configs are available.
		if lr.Backend == "roofline" || lr.Backend == "trained-physics" {
			if prefillInstances > 0 {
				hfPath := filepath.Join(modelConfigFolder, "config.json")
				hfConfig, err := latency.ParseHFConfig(hfPath)
				if err != nil {
					logrus.Fatalf("Failed to parse HuggingFace config for per-pool KV calc: %v", err)
				}
				kvParamsPool, kvErrPool := latency.ExtractKVCapacityParams(hfConfig)
				if kvErrPool != nil {
					logrus.Warnf("per-pool KV auto-calculation skipped (could not extract model KV params: %v); both pools will use global total-kv-blocks=%d", kvErrPool, totalKVBlocks)
				} else {
					// Prefill pool auto-calc
					poolPrefillTP := tensorParallelism
					if cmd.Flags().Changed("prefill-tp") {
						poolPrefillTP = prefillTP
					}
					poolPrefillGPU := gpu
					if cmd.Flags().Changed("prefill-hardware") {
						poolPrefillGPU = prefillHardware
					}
					if poolPrefillTP != tensorParallelism || poolPrefillGPU != gpu {
						poolHC, hcErr := latency.GetHWConfig(hwConfigPath, poolPrefillGPU)
						if hcErr != nil {
							logrus.Warnf("--prefill-hardware: failed to load hardware config for GPU %q: %v; prefill pool will use global total-kv-blocks=%d", poolPrefillGPU, hcErr, totalKVBlocks)
						} else if poolHC.MemoryGiB <= 0 {
							logrus.Warnf("--prefill-hardware: GPU memory capacity not available for %q in hardware config; prefill pool will use global total-kv-blocks=%d", poolPrefillGPU, totalKVBlocks)
						} else {
							// Per-pool TP but GLOBAL dp: per-pool DP is out of scope (#1420);
							// --dp applies uniformly to all pools. Not a bug — see issue #1420.
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolPrefillTP, dataParallelism, blockSizeTokens, gpuMemoryUtilization, kvParamsPool)
							if calcErr != nil {
								logrus.Fatalf("--prefill-tp/--prefill-hardware: KV capacity auto-calculation failed for prefill pool: %v", calcErr)
							} else {
								prefillOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--prefill-tp/--prefill-hardware: auto-calculated prefill pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, DP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolPrefillTP, dataParallelism)
								if !cmd.Flags().Changed("prefill-max-model-len") {
									kvFeasibleMax := poolBlocks * int64(blockSizeTokens)
									if kvFeasibleMax < maxModelLen {
										prefillOverrides.MaxModelLen = &kvFeasibleMax
										logrus.Infof("--prefill-tp/--prefill-hardware: auto-capped prefill pool max-model-len=%d (pool KV capacity smaller than global)", kvFeasibleMax)
									}
								}
							}
						}
					}

					// Decode pool auto-calc
					poolDecodeTP := tensorParallelism
					if cmd.Flags().Changed("decode-tp") {
						poolDecodeTP = decodeTP
					}
					poolDecodeGPU := gpu
					if cmd.Flags().Changed("decode-hardware") {
						poolDecodeGPU = decodeHardware
					}
					if poolDecodeTP != tensorParallelism || poolDecodeGPU != gpu {
						poolHC, hcErr := latency.GetHWConfig(hwConfigPath, poolDecodeGPU)
						if hcErr != nil {
							logrus.Warnf("--decode-hardware: failed to load hardware config for GPU %q: %v; decode pool will use global total-kv-blocks=%d", poolDecodeGPU, hcErr, totalKVBlocks)
						} else if poolHC.MemoryGiB <= 0 {
							logrus.Warnf("--decode-hardware: GPU memory capacity not available for %q in hardware config; decode pool will use global total-kv-blocks=%d", poolDecodeGPU, totalKVBlocks)
						} else {
							// Per-pool TP, global dp (see prefill-pool note above; #1420).
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolDecodeTP, dataParallelism, blockSizeTokens, gpuMemoryUtilization, kvParamsPool)
							if calcErr != nil {
								logrus.Fatalf("--decode-tp/--decode-hardware: KV capacity auto-calculation failed for decode pool: %v", calcErr)
							} else {
								decodeOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--decode-tp/--decode-hardware: auto-calculated decode pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, DP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolDecodeTP, dataParallelism)
								if !cmd.Flags().Changed("decode-max-model-len") {
									kvFeasibleMax := poolBlocks * int64(blockSizeTokens)
									if kvFeasibleMax < maxModelLen {
										decodeOverrides.MaxModelLen = &kvFeasibleMax
										logrus.Infof("--decode-tp/--decode-hardware: auto-capped decode pool max-model-len=%d (pool KV capacity smaller than global)", kvFeasibleMax)
									}
								}
							}
						}
					}
				}
			}
		}

		// R3: Validate workload generation flags (before any synthesis path consumes them)
		if numRequests < 0 {
			logrus.Fatalf("--num-requests must be >= 0, got %d", numRequests)
		}
		if prefixTokens < 0 {
			logrus.Fatalf("--prefix-tokens must be >= 0, got %d", prefixTokens)
		}

		// R3: Validate concurrency flags
		if concurrency < 0 {
			logrus.Fatalf("--concurrency must be >= 0, got %d", concurrency)
		}
		if thinkTimeMs < 0 {
			logrus.Fatalf("--think-time-ms must be >= 0, got %d", thinkTimeMs)
		}
		// BC-1: --concurrency and --rate are mutually exclusive
		if concurrency > 0 && cmd.Flags().Changed("rate") {
			logrus.Fatalf("--concurrency and --rate are mutually exclusive; use one or the other")
		}

		// Workload configuration — all paths synthesize a v2 WorkloadSpec
		// and generate requests via workload.GenerateRequests (BC-10).
		var spec *workload.WorkloadSpec
		var preGeneratedRequests []*sim.Request
		var sessionMgr *workload.SessionManager

		if workloadSpecPath != "" {
			if concurrency > 0 {
				logrus.Fatalf("--concurrency cannot be used with --workload-spec; " +
					"define concurrency in the spec file using clients[].concurrency instead")
			}
			// --workload-spec takes precedence over --workload
			var err error
			spec, err = workload.LoadWorkloadSpec(workloadSpecPath)
			if err != nil {
				logrus.Fatalf("Failed to load workload spec: %v", err)
			}
			// Apply CLI --seed override (R18: CLI flag precedence)
			if cmd.Flags().Changed("seed") {
				logrus.Infof("CLI --seed %d overrides workload-spec seed %d", seed, spec.Seed)
				spec.Seed = seed
			} else {
				logrus.Infof("Using workload-spec seed %d (CLI --seed not specified)", spec.Seed)
			}
			if spec.Horizon > 0 && !cmd.Flags().Changed("horizon") {
				simulationHorizon = spec.Horizon
			}
		} else if concurrency > 0 {
			// Concurrency mode → synthesize v2 spec with closed-loop client.
			// In concurrency mode, --num-requests has no meaningful default.
			// If the user did not explicitly set it, leave it at 0 (unbounded) and
			// require --horizon to bound the run. The existing unbounded-generation
			// guard will fire with a clear message if neither is provided.
			// R3: Validate distribution token bounds (shared with distribution mode).
			if msg := validateDistributionParams(promptTokensMin, promptTokensMax, outputTokensMin, outputTokensMax,
				promptTokensStdev, outputTokensStdev, promptTokensMean, outputTokensMean); msg != "" {
				logrus.Fatalf("%s", msg)
			}
			concurrencyNumRequests := 0
			if cmd.Flags().Changed("num-requests") {
				concurrencyNumRequests = numRequests
			}
			spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
				Concurrency: concurrency, ThinkTimeMs: thinkTimeMs,
				NumRequests: concurrencyNumRequests, PrefixTokens: prefixTokens,
				PromptTokensMean: promptTokensMean, PromptTokensStdDev: promptTokensStdev,
				PromptTokensMin: promptTokensMin, PromptTokensMax: promptTokensMax,
				OutputTokensMean: outputTokensMean, OutputTokensStdDev: outputTokensStdev,
				OutputTokensMin: outputTokensMin, OutputTokensMax: outputTokensMax,
			})
			spec.Seed = seed
		} else if workloadType == "distribution" {
			// Distribution mode → synthesize v2 spec from CLI flags
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
			}
			// R3: Validate distribution token bounds (shared with concurrency mode).
			if msg := validateDistributionParams(promptTokensMin, promptTokensMax, outputTokensMin, outputTokensMax,
				promptTokensStdev, outputTokensStdev, promptTokensMean, outputTokensMean); msg != "" {
				logrus.Fatalf("%s", msg)
			}
			spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
				Rate: rate, NumRequests: numRequests, PrefixTokens: prefixTokens,
				PromptTokensMean: promptTokensMean, PromptTokensStdDev: promptTokensStdev,
				PromptTokensMin: promptTokensMin, PromptTokensMax: promptTokensMax,
				OutputTokensMean: outputTokensMean, OutputTokensStdDev: outputTokensStdev,
				OutputTokensMin: outputTokensMin, OutputTokensMax: outputTokensMax,
			})
			spec.Seed = seed
		} else {
			// Preset name (chatbot, summarization, etc.) → synthesize v2 spec
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
			}
			wl := loadPresetWorkload(defaultsFilePath, workloadType)
			if wl == nil {
				logrus.Fatalf("Undefined workload %q. Use one among (chatbot, summarization, contentgen, multidoc) or --workload-spec", workloadType)
			}
			spec = workload.SynthesizeFromPreset(workloadType, workload.PresetConfig{
				PrefixTokens:     wl.PrefixTokens,
				PromptTokensMean: wl.PromptTokensMean, PromptTokensStdev: wl.PromptTokensStdev,
				PromptTokensMin: wl.PromptTokensMin, PromptTokensMax: wl.PromptTokensMax,
				OutputTokensMean: wl.OutputTokensMean, OutputTokensStdev: wl.OutputTokensStdev,
				OutputTokensMin: wl.OutputTokensMin, OutputTokensMax: wl.OutputTokensMax,
			}, rate, numRequests)
			spec.Seed = seed
		}

		// Apply per-request timeout to all clients.
		// For synthesized specs, always apply (default 300s matches the session-client default).
		// For file-loaded specs, only apply when the flag is explicitly set.
		if requestTimeoutSecs == 0 {
			logrus.Fatalf("--timeout must be positive (seconds) or negative to disable; got 0")
		}
		// Pre-expand inference-perf / ServeGen specs (LAZY MODE ONLY) so
		// the timeout-application step below sees every client — including
		// those populated by expansion. Without this, --lazy-generation +
		// --workload-spec=inference_perf.yaml + an explicit --timeout
		// would build streaming states from clients whose Timeout is nil
		// (because expansion inside GenerateWorkloadLazy happens AFTER
		// applyTimeoutToSpec runs), producing the default 300 s deadline
		// instead of the user-requested value (PR #1453 self-review).
		//
		// Scoped to `lazyGeneration` so it does not run for eager-only
		// invocations. Running it unconditionally would clear the
		// spec.InferencePerf marker before GenerateWorkload's Validate,
		// which suppresses the mixed-slo_class check via
		// `s.InferencePerf == nil && s.ServeGenData == nil` in spec.go.
		// Today all inference-perf-expanded clients carry SLOClass="standard"
		// (uniform → check can't fire), so eager was accidentally safe.
		// But scoping here defends against a future ExpandInferencePerfSpec
		// change that emits mixed/empty slo_class from silently failing
		// eager runs that previously validated (PR #1453 review round 3).
		//
		// REMOVING THIS CALL re-introduces the lazy timeout bug silently.
		// The regression is covered by
		// TestGenerateWorkloadLazy_InferencePerf_TimeoutAppliedAfterPreExpand
		// at the library layer; the cmd-level smoke is covered by the
		// inference-perf byte-identity check verified during PR review.
		//
		// ExpandClientsAndCohorts is idempotent — the generators'
		// validateAndExpandSpec runs it again with no effect since both
		// branches guard on len(spec.Clients) == 0. (#1441)
		if lazyGeneration {
			if err := workload.ExpandClientsAndCohorts(spec); err != nil {
				logrus.Fatalf("Failed to expand workload spec: %v", err)
			}
		}
		if workloadSpecPath == "" || cmd.Flags().Changed("timeout") {
			applyTimeoutToSpec(spec, requestTimeoutSecs)
		}

		// Resolve maxRequests: spec.NumRequests as default, CLI --num-requests overrides
		maxRequests := spec.NumRequests
		if cmd.Flags().Changed("num-requests") {
			maxRequests = int64(numRequests)
		}

		// Guard against unbounded generation
		if maxRequests <= 0 && simulationHorizon == math.MaxInt64 {
			logrus.Fatalf("Workload requires either num_requests or --horizon to bound generation")
		}

		// Lazy generation path (#1441, alpha). Default off. When set, build
		// a streaming workload source instead of materializing the full
		// request slice. As of #1460 there is NO eager-fallback class — every
		// spec the eager generator accepts is streamed: multi-session reasoning
		// (#1458), concurrency clients (#1459), and time-varying / per-window
		// workloads (#1460).
		var wl *workload.GeneratedWorkload
		// lazyRequestSource is typed as the interface satisfied by
		// *workload.lazyRequestSource: Next() delivers requests to the
		// cluster; Err() surfaces any terminal sampler/generator error
		// recorded on a per-client state after the run completes, so
		// cmd can Fatalf and match the eager path's abort-on-invalid-spec
		// behavior (PR #1453 review round 3).
		var lazyRequestSource interface {
			Next() (*sim.Request, bool)
			Err() error
		}
		if lazyGeneration {
			src, sessions, followUpBudget, lazyErr := workload.GenerateWorkloadLazy(spec, simulationHorizon, maxRequests)
			// As of #1460 there is no ErrLazyUnsupported* fallback class — every
			// spec the eager generator accepts is streamed. Any error is a real
			// spec/validation failure → abort (matches eager's error handling).
			if lazyErr != nil {
				logrus.Fatalf("Failed to build lazy workload: %v", lazyErr)
			}
			lazyRequestSource = src
			wl = &workload.GeneratedWorkload{Sessions: sessions, FollowUpBudget: followUpBudget}
		}
		if wl == nil {
			var err error
			wl, err = workload.GenerateWorkload(spec, simulationHorizon, maxRequests)
			if err != nil {
				logrus.Fatalf("Failed to generate workload: %v", err)
			}
		}
		// Re-apply timeout to generated requests and session blueprints.
		// For inference_perf specs, spec.Clients was empty at applyTimeoutToSpec time
		// and populated inside GenerateWorkload — deadlines need correction here.
		// In lazy mode wl.Requests is nil (no-op for the request loop); session
		// blueprint Timeout pointers still need refresh.
		if workloadSpecPath == "" || cmd.Flags().Changed("timeout") {
			applyTimeoutToRequests(wl, requestTimeoutSecs)
		}
		preGeneratedRequests = wl.Requests
		if len(wl.Sessions) > 0 {
			sessionMgr = workload.NewSessionManager(wl.Sessions)
			if wl.FollowUpBudget >= 0 {
				sessionMgr.SetFollowUpBudget(wl.FollowUpBudget)
			}
			if lazyRequestSource != nil {
				logrus.Infof("Generated streaming source + %d session blueprints (closed-loop, lazy)", len(wl.Sessions))
			} else {
				logrus.Infof("Generated %d requests + %d session blueprints (closed-loop)", len(wl.Requests), len(wl.Sessions))
			}
		} else if lazyRequestSource != nil {
			logrus.Infof("Generated streaming workload source (lazy, #1441)")
		} else {
			logrus.Infof("Generated %d requests via unified workload pipeline", len(wl.Requests))
		}

		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}
		if totalKVBlocks <= 0 {
			logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
		}
		if maxRunningReqs <= 0 {
			logrus.Fatalf("--max-num-running-reqs must be > 0, got %d", maxRunningReqs)
		}
		if maxScheduledTokens <= 0 {
			logrus.Fatalf("--max-num-scheduled-tokens must be > 0, got %d", maxScheduledTokens)
		}
		if longPrefillTokenThreshold < 0 {
			logrus.Fatalf("--long-prefill-token-threshold must be >= 0, got %d", longPrefillTokenThreshold)
		}
		// Changed() guard: unlike peer flags (default always positive), --horizon defaults
		// to math.MaxInt64 which would fail <= 0. Only validate when user explicitly sets it.
		if cmd.Flags().Changed("horizon") && simulationHorizon <= 0 {
			logrus.Fatalf("--horizon must be > 0, got %d", simulationHorizon)
		}

		// Resolve policy configuration (single code path shared with replayCmd).
		// Per-pool scorer configs (PD disaggregation) remain inline below.
		parsedScorerConfigs, bundle := resolvePolicies(cmd)

		// Resolve autoscaler and node pool config from policy bundle, then apply CLI overrides.
		var (
			bundleAutoscalerIntervalUs           float64
			bundleScaleUpStabilizationWindowUs   float64
			bundleScaleDownStabilizationWindowUs float64
			bundleHPAScrapeDelayMean             float64
			bundleHPAScrapeDelayStddev           float64
			bundleAnalyzerCfg                    cluster.V2SaturationAnalyzerConfig
			bundleNodePools                      []cluster.NodePoolConfig
			bundleInstanceLifecycle              cluster.InstanceLifecycleConfig
		)
		if bundle != nil {
			if bundle.Autoscaler.IntervalUs > 0 {
				bundleAutoscalerIntervalUs = bundle.Autoscaler.IntervalUs
				bundleScaleUpStabilizationWindowUs = bundle.Autoscaler.ScaleUpStabilizationWindowUs
				bundleScaleDownStabilizationWindowUs = bundle.Autoscaler.ScaleDownStabilizationWindowUs
				bundleHPAScrapeDelayMean = bundle.Autoscaler.HPAScrapeDelay.Mean
				bundleHPAScrapeDelayStddev = bundle.Autoscaler.HPAScrapeDelay.Stddev
				bundleAnalyzerCfg = cluster.V2SaturationAnalyzerConfig{
					KvCacheThreshold:  bundle.Autoscaler.Analyzer.KVCacheThreshold,
					ScaleUpThreshold:  bundle.Autoscaler.Analyzer.ScaleUpThreshold,
					ScaleDownBoundary: bundle.Autoscaler.Analyzer.ScaleDownBoundary,
					AvgInputTokens:    bundle.Autoscaler.Analyzer.AvgInputTokens,
				}
			}
			for _, np := range bundle.NodePools {
				bundleNodePools = append(bundleNodePools, cluster.NodePoolConfig{
					Name:         np.Name,
					GPUType:      np.GPUType,
					GPUsPerNode:  np.GPUsPerNode,
					GPUMemoryGiB: np.GPUMemoryGiB,
					InitialNodes: np.InitialNodes,
					MinNodes:     np.MinNodes,
					MaxNodes:     np.MaxNodes,
					ProvisioningDelay: cluster.DelaySpec{
						Mean:   np.ProvisioningDelay.Mean,
						Stddev: np.ProvisioningDelay.Stddev,
					},
					CostPerHour: np.CostPerHour,
				})
			}
			bundleInstanceLifecycle = cluster.InstanceLifecycleConfig{
				LoadingDelay: cluster.DelaySpec{
					Mean:   bundle.InstanceLifecycle.LoadingDelay.Mean,
					Stddev: bundle.InstanceLifecycle.LoadingDelay.Stddev,
				},
				WarmStartInitialInstances: bundle.InstanceLifecycle.WarmStartInitialInstances,
			}
		}
		// CLI flag overrides bundle value when explicitly set.
		if cmd.Flags().Changed("model-autoscaler-interval-us") {
			bundleAutoscalerIntervalUs = modelAutoscalerIntervalUs
		}

		// PD disaggregation validation (R3: validate at CLI boundary)
		if prefillInstances < 0 {
			logrus.Fatalf("--prefill-instances must be >= 0, got %d", prefillInstances)
		}
		if decodeInstances < 0 {
			logrus.Fatalf("--decode-instances must be >= 0, got %d", decodeInstances)
		}
		if prefillDecodeInstances < 0 {
			logrus.Fatalf("--prefill-decode-instances must be >= 0, got %d", prefillDecodeInstances)
		}
		if !sim.IsValidDisaggregationDecider(pdDecider) {
			logrus.Fatalf("Unknown PD decider %q. Valid: %s", pdDecider, strings.Join(sim.ValidDisaggregationDeciderNames(), ", "))
		}
		if err := cluster.ValidatePoolTopology(prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances, numInstances); err != nil {
			logrus.Fatalf("Invalid PD pool topology: %v", err)
		}
		// PD transfer parameter validation (R3, R11)
		if prefillInstances > 0 {
			if pdTransferBandwidth <= 0 || math.IsInf(pdTransferBandwidth, 0) || math.IsNaN(pdTransferBandwidth) {
				logrus.Fatalf("--pd-transfer-bandwidth must be a finite positive number, got %f", pdTransferBandwidth)
			}
			if pdTransferBaseLatency < 0 || math.IsInf(pdTransferBaseLatency, 0) || math.IsNaN(pdTransferBaseLatency) {
				logrus.Fatalf("--pd-transfer-base-latency must be a finite non-negative number, got %f", pdTransferBaseLatency)
			}
		}
		if pdDecider == "prefix-threshold" && pdPrefixThreshold < 0 {
			logrus.Fatalf("--pd-prefix-threshold must be >= 0, got %d", pdPrefixThreshold)
		}
		if pdDecider != "prefix-threshold" && cmd.Flags().Changed("pd-prefix-threshold") {
			logrus.Warnf("--pd-prefix-threshold=%d is ignored when --pd-decider=%q (only applies to the prefix-threshold decider)", pdPrefixThreshold, pdDecider)
		}
		if pdDecider != "" && pdDecider != "never" && prefillInstances == 0 {
			logrus.Warnf("--pd-decider=%q has no effect because --prefill-instances=0 (disaggregation is disabled); set --prefill-instances and --decode-instances to enable", pdDecider)
		}

		// E/P/D disaggregation validation (GAP-4, issue #1264).
		if encodeInstances < 0 {
			logrus.Fatalf("--encode-instances must be >= 0, got %d", encodeInstances)
		}
		if !sim.IsValidEncodeDecider(encodeDecider) {
			logrus.Fatalf("Unknown encode decider %q. Valid: %s", encodeDecider, strings.Join(sim.ValidEncodeDeciderNames(), ", "))
		}
		if encodeDecider != "" && encodeDecider != "never" && encodeInstances == 0 {
			logrus.Fatalf("--encode-decider=%q requires --encode-instances > 0 (the encode pool is disabled)", encodeDecider)
		}
		if encodeInstances > 0 && (encodeDecider == "" || encodeDecider == "never") {
			logrus.Warnf("--encode-decider=%q has no effect because --encode-instances=%d but the decider never encodes; set --encode-decider=multimodal or always to activate the encode pool", encodeDecider, encodeInstances)
		}

		// Per-pool hardware override construction (R3): build PoolOverrides from CLI flags.
		// Pointer fields use cmd.Flags().Changed() to distinguish "not set" from "set to value".
		// Warns if per-pool flags are set but disaggregation is disabled.
		perPoolFlagsChanged := cmd.Flags().Changed("prefill-tp") || cmd.Flags().Changed("decode-tp") ||
			cmd.Flags().Changed("prefill-hardware") || cmd.Flags().Changed("decode-hardware") ||
			cmd.Flags().Changed("prefill-latency-model") || cmd.Flags().Changed("decode-latency-model") ||
			cmd.Flags().Changed("prefill-max-model-len") || cmd.Flags().Changed("decode-max-model-len")
		if perPoolFlagsChanged && prefillInstances == 0 {
			logrus.Warnf("per-pool hardware flags (--prefill-tp, --decode-tp, etc.) have no effect when --prefill-instances=0 (disaggregation is disabled)")
		}
		if prefillInstances > 0 {
			// Prefill pool overrides
			if cmd.Flags().Changed("prefill-tp") {
				if prefillTP <= 0 {
					logrus.Fatalf("--prefill-tp must be > 0, got %d", prefillTP)
				}
				tp := prefillTP
				prefillOverrides.TP = &tp
			}
			if cmd.Flags().Changed("prefill-hardware") {
				prefillOverrides.GPU = prefillHardware
			}
			if cmd.Flags().Changed("prefill-latency-model") {
				if !sim.IsValidLatencyBackend(prefillLatencyModel) {
					logrus.Fatalf("--prefill-latency-model %q is not a recognized backend; valid: %s",
						prefillLatencyModel, strings.Join(sim.ValidLatencyBackendNames(), ", "))
				}
				prefillOverrides.LatencyBackend = prefillLatencyModel
			}
			if cmd.Flags().Changed("prefill-max-model-len") {
				if prefillMaxModelLen <= 0 {
					logrus.Fatalf("--prefill-max-model-len must be > 0 when set, got %d", prefillMaxModelLen)
				}
				ml := prefillMaxModelLen
				prefillOverrides.MaxModelLen = &ml
			}
			// Decode pool overrides
			if cmd.Flags().Changed("decode-tp") {
				if decodeTP <= 0 {
					logrus.Fatalf("--decode-tp must be > 0, got %d", decodeTP)
				}
				tp := decodeTP
				decodeOverrides.TP = &tp
			}
			if cmd.Flags().Changed("decode-hardware") {
				decodeOverrides.GPU = decodeHardware
			}
			if cmd.Flags().Changed("decode-latency-model") {
				if !sim.IsValidLatencyBackend(decodeLatencyModel) {
					logrus.Fatalf("--decode-latency-model %q is not a recognized backend; valid: %s",
						decodeLatencyModel, strings.Join(sim.ValidLatencyBackendNames(), ", "))
				}
				decodeOverrides.LatencyBackend = decodeLatencyModel
			}
			if cmd.Flags().Changed("decode-max-model-len") {
				if decodeMaxModelLen <= 0 {
					logrus.Fatalf("--decode-max-model-len must be > 0 when set, got %d", decodeMaxModelLen)
				}
				ml := decodeMaxModelLen
				decodeOverrides.MaxModelLen = &ml
			}
		}

		// Parse per-pool scorer configs (PD disaggregation — not in resolvePolicies)
		var prefillScorerCfgs, decodeScorerCfgs []sim.ScorerConfig
		if prefillRoutingScorers != "" {
			var err error
			prefillScorerCfgs, err = sim.ParseScorerConfigs(prefillRoutingScorers)
			if err != nil {
				logrus.Fatalf("Invalid --prefill-routing-scorers: %v", err)
			}
		}
		if decodeRoutingScorers != "" {
			var err error
			decodeScorerCfgs, err = sim.ParseScorerConfigs(decodeRoutingScorers)
			if err != nil {
				logrus.Fatalf("Invalid --decode-routing-scorers: %v", err)
			}
		}
		// Log configuration after all config sources (CLI, workload spec, policy bundle) are resolved
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v",
			totalKVBlocks, simulationHorizon, lr.AlphaCoeffs, lr.BetaCoeffs)

		startTime := time.Now() // Get current time (start)

		// Unified cluster path (used for all values of numInstances).
		// INV-13 SYNC POINT: PD fields below must stay in sync with cmd/replay.go (replayCmd
		// DeploymentConfig literal). See docs/contributing/standards/invariants.md INV-13.
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: simulationHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:          sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:        sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
				ModelHardwareConfig:  sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, dataParallelism, enableExpertParallel, moeCommBackend, lr.Backend, maxModelLen),
				PolicyConfig:         sim.NewPolicyConfig(scheduler, preemptionPolicy),
				SLOPriorityOverrides: sloPriorityOverrides,
			},
			NumInstances:                    numInstances,
			AdmissionPolicy:                 admissionPolicy,
			AdmissionLatency:                admissionLatency,
			RoutingLatency:                  routingLatency,
			TokenBucketCapacity:             tokenBucketCapacity,
			TokenBucketRefillRate:           tokenBucketRefillRate,
			RoutingPolicy:                   routingPolicy,
			RoutingScorerConfigs:            parsedScorerConfigs,
			TraceLevel:                      traceLevel,
			CounterfactualK:                 counterfactualK,
			SnapshotRefreshInterval:         snapshotRefreshInterval,
			CacheSignalDelay:                cacheSignalDelay,
			PrefillInstances:                prefillInstances,
			DecodeInstances:                 decodeInstances,
			SharedInstances:                 prefillDecodeInstances,
			EncodeInstances:                 encodeInstances,
			EncodeDecider:                   encodeDecider,
			PDDecider:                       pdDecider,
			PDPrefixThreshold:               pdPrefixThreshold,
			PDTransferBandwidthGBps:         pdTransferBandwidth,
			PDTransferBaseLatencyMs:         pdTransferBaseLatency,
			PDTransferContention:            pdTransferContention,
			PrefillScorerConfigs:            prefillScorerCfgs,
			DecodeScorerConfigs:             decodeScorerCfgs,
			PrefillOverrides:                prefillOverrides,
			DecodeOverrides:                 decodeOverrides,
			TierShedThreshold:               tierShedThreshold,
			TierShedMinPriority:             tierShedMinPriority,
			GAIEQDThreshold:                 gaieQDThreshold,
			GAIEKVThreshold:                 gaieKVThreshold,
			TenantBudgets:                   tenantBudgets,
			FlowControlEnabled:              flowControlEnabled,
			FlowControlDetector:             flowControlDetector,
			FlowControlDispatchOrder:        flowControlDispatchOrder,
			FlowControlSLOTargets:           sloTargetsMap,
			FlowControlMaxQueueDepth:        flowControlMaxQueueDepth,
			FlowControlQueueDepthThreshold:  flowControlQueueDepthThreshold,
			FlowControlKVCacheUtilThreshold: flowControlKVCacheUtilThreshold,
			FlowControlMaxConcurrency:       flowControlMaxConcurrency,
			FlowControlPerBandCapacity:      flowControlPerBandCapacity,
			FlowControlUsageLimitThreshold:  flowControlUsageLimitThreshold,
			FlowControlFairnessPolicy:       flowControlFairnessPolicy,
			FlowControlRequestTTL:           flowControlRequestTTL,
			FlowControlQueueShedding:        flowControlQueueShedding,
			FlowControlDispatchTickInterval: flowControlDispatchTickInterval,
			FlowControlInFlightEviction:     flowControlInFlightEviction,
			ModelAutoscalerIntervalUs:       bundleAutoscalerIntervalUs,
			ScaleUpStabilizationWindowUs:    bundleScaleUpStabilizationWindowUs,
			ScaleDownStabilizationWindowUs:  bundleScaleDownStabilizationWindowUs,
			HPAScrapeDelay:                  cluster.DelaySpec{Mean: bundleHPAScrapeDelayMean, Stddev: bundleHPAScrapeDelayStddev},
			AutoscalerAnalyzerConfig:        bundleAnalyzerCfg,
			NodePools:                       bundleNodePools,
			InstanceLifecycle:               bundleInstanceLifecycle,
		}
		// Session callback installation (Constraint 3 fix):
		// Follow-up collection must be UNCONDITIONAL for saturation analysis correctness.
		// The TraceV2 export (lines 1582-1601) remains gated on --trace-output, but the
		// follow-up accumulation happens regardless so saturation analysis sees complete workloads.
		var followUpRequests []*sim.Request
		var onRequestDone func(*sim.Request, int64) []*sim.Request
		if sessionMgr != nil {
			// Always install callback to accumulate follow-ups (for saturation analysis + optional trace export)
			baseCb := sessionMgr.OnComplete
			onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
				followUps := baseCb(req, clock)
				followUpRequests = append(followUpRequests, followUps...)
				return followUps
			}
		}
		// RequestSource: streaming in lazy mode, eager-slice otherwise.
		// The workload package's lazy source satisfies cluster.RequestSource
		// via structural typing — both define the same Next() method.
		var clusterRequestSource cluster.RequestSource
		if lazyRequestSource != nil {
			clusterRequestSource = lazyRequestSource
		} else {
			clusterRequestSource = cluster.NewSliceRequestSource(preGeneratedRequests)
		}
		cs := cluster.NewClusterSimulator(config, clusterRequestSource, onRequestDone)

		// Arrival hook: capture trace-emission references at the cluster's
		// single arrival boundary so the trace exporter no longer relies on
		// the eager preGeneratedRequests + followUpRequests list assembly
		// (issue #1440). The hook fires once per fresh arrival in
		// clock-monotonic order — see ClusterArrivalEvent.Execute. We hold
		// pointers (not copies) so the request's final state (set by the
		// event loop) is visible at export time.
		//
		// Only install when --trace-output is set (BC-1: zero overhead when
		// trace is disabled). Saturation analysis continues to use the
		// preGeneratedRequests + followUpRequests path below — those slices
		// remain populated for that purpose only.
		//
		// Install/export coupling: traceArrivals is declared nil here and
		// assigned a non-nil empty slice ONLY inside the install branch.
		// A nil traceArrivals at the export site below with traceOutput
		// non-empty means the install branch was dropped — we fail loudly
		// rather than write a silent empty trace (R1).
		// The arrival hook captures fresh-arrival references at the single
		// cluster boundary. It powers trace export (#1440) and, in lazy mode
		// where preGeneratedRequests is nil, also feeds saturation analysis.
		// In eager mode without --trace-output, the hook stays uninstalled
		// (BC-1 zero overhead) and saturation falls back to the
		// preGeneratedRequests + followUpRequests path.
		var traceArrivals []*sim.Request
		arrivalHookNeeded := traceOutput != "" || (lazyRequestSource != nil && saturationReport != "")
		if arrivalHookNeeded {
			traceArrivals = make([]*sim.Request, 0)
			cs.SetArrivalHook(func(req *sim.Request) {
				traceArrivals = append(traceArrivals, req)
			})
		}
		if err := cs.Run(); err != nil {
			logrus.Fatalf("Simulation failed: %v", err)
		}

		// Surface any terminal sampler / generator error the lazy source
		// recorded on a per-client state during the run. Eager mode would
		// have hit logrus.Fatalf inside cmd on the same invalid spec;
		// without this check, lazy mode would exit 0 with reduced traffic
		// and misleading capacity numbers (PR #1453 review round 3).
		if lazyRequestSource != nil {
			if err := lazyRequestSource.Err(); err != nil {
				logrus.Fatalf("Lazy workload sampler failure: %v", err)
			}
		}

		// Wall-clock timing on stderr (BC-6); stdout remains deterministic (BC-7)
		logrus.Infof("Simulation wall-clock time: %.3fs", time.Since(startTime).Seconds())

		// Resolve goodput SLO targets early so the trace export and aggregate metrics
		// see the same merged map (#1413, BC-1). Run has no trace header; precedence
		// here is CLI > workload spec.
		cliTTFT, cliITL, cliE2E, gpErr := resolveGoodputCLIFlags(goodputSLOTTFT, goodputSLOITL, goodputSLOE2E)
		if gpErr != nil {
			logrus.Fatalf("%v", gpErr)
		}
		var specTargets map[string]workload.SLODimTargets
		if spec != nil {
			specTargets = spec.GoodputSLOTargets
		}
		goodputTargets := mergeGoodputTargets(cliTTFT, cliITL, cliE2E, nil, specTargets)

		// Assemble allRequests for saturation analysis (BC-12, issue #1298).
		// Trace export is now driven by the arrival hook above and no longer
		// shares this slice (issue #1440). allRequests is nil when
		// --saturation-report is not set.
		//
		// In lazy mode (#1441), preGeneratedRequests is nil — the arrival hook
		// captures every fresh arrival in clock-monotonic order (already sorted
		// by INV-3), so we use traceArrivals directly. In eager mode we keep
		// the existing append+sort path for backward compatibility.
		var allRequests []*sim.Request
		if saturationReport != "" {
			if lazyRequestSource != nil {
				// traceArrivals already contains every fresh arrival in
				// clock-monotonic order — no separate followUpRequests merge
				// or post-sort required (the cluster delivers them in arrival
				// order via the hook).
				//
				// SAFETY: allRequests aliases the same backing array as
				// traceArrivals. Both downstream consumers (trace export
				// below + saturation analysis) MUST be read-only of this
				// slice — neither appends, reorders, nor mutates element
				// contents. If a future consumer needs to mutate, copy
				// first: `allRequests = append([]*sim.Request(nil), traceArrivals...)`.
				allRequests = traceArrivals
			} else {
				allRequests = make([]*sim.Request, 0, len(preGeneratedRequests)+len(followUpRequests))
				allRequests = append(allRequests, preGeneratedRequests...)
				allRequests = append(allRequests, followUpRequests...)
				// Sort by arrival time so RequestIDs (array indices) are arrival-ordered
				sort.SliceStable(allRequests, func(i, j int) bool {
					return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
				})
			}
		}

		// Export trace if requested (BC-1, BC-7). Records are sourced from
		// the arrival hook (issue #1440) — already in clock-monotonic order
		// per INV-3, so no sort is required.
		if traceOutput != "" {
			// Install/export coupling guard (R1): traceArrivals is a non-nil
			// empty slice when SetArrivalHook ran above. A nil here means
			// the install branch was dropped or moved without updating this
			// site — refuse to write a silent empty trace.
			if traceArrivals == nil {
				logrus.Fatalf("Trace export: arrival hook was not installed but --trace-output=%q is set — install/export branches diverged (issue #1440)", traceOutput)
			}
			records := workload.RequestsToTraceRecords(traceArrivals)
			header := &workload.TraceHeader{
				Version:           3,
				TimeUnit:          "microseconds",
				Mode:              "generated",
				WorkloadSeed:      &spec.Seed,
				GoodputSLOTargets: goodputTargets, // #1413, BC-7: persist resolved targets for downstream replay/calibrate
			}
			if err := workload.ExportTraceV2(header, records, traceOutput+".yaml", traceOutput+".csv"); err != nil {
				logrus.Fatalf("Trace export failed: %v", err)
			}
			logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", traceOutput, traceOutput, len(records))
		}

		// Saturation analysis if requested (issue #1298, #1391, #1392)
		if saturationReport != "" {
			simEndUs := workload.ComputeSimEndUs(allRequests, config.Horizon)

			// Validate classifier name (CLI gate; library factory panics on unknown).
			if !sim.IsValidBacklogClassifier(saturationClassifier) {
				logrus.Fatalf("Unknown --saturation-classifier %q. Valid: %s",
					saturationClassifier, strings.Join(sim.ValidBacklogClassifierNames(), ", "))
			}
			classifier := workload.NewBacklogClassifier(saturationClassifier)

			// Build saturation analysis config from flags (or defaults if not set)
			cfg := workload.NewBacklogDriftConfig(
				time.Duration(saturationWindowSec)*time.Second,
				saturationMinWindows,
				saturationPeakRatio,
				saturationPeakBand,
				saturationConfidence,
				saturationWarmupWindows,
				saturationTailWindows,
				saturationSaturatedRatio,
				saturationTransientRatio,
			)
			report := workload.AnalyzeBacklogDriftWithClassifier(allRequests, simEndUs, cfg, classifier)
			if err := workload.WriteBacklogDriftReportJSON(saturationReport, report); err != nil {
				logrus.Fatalf("Failed to write saturation report: %v", err)
			}
			logrus.Infof("Saturation report written to %s (classification: %s)", saturationReport, report.Classification)
		}

		// Validate and instantiate post-hoc saturation detector from CLI flags (#1369, C3)
		if !saturation.ValidDetectorNames()[postHocDetector] {
			logrus.Fatalf("--post-hoc-detector %q not recognized. Valid: composite, threshold, none", postHocDetector)
		}

		// Validate saturation threshold for negative values (I4)
		if saturationThreshold < 0 {
			logrus.Fatalf("--saturation-threshold-ms must be non-negative, got %.2f", saturationThreshold)
		}

		var saturationDetector sim.BatchClassifier
		if postHocDetector != "none" {
			saturationDetector = saturation.NewDetector(postHocDetector, saturation.DetectorOpts{
				ThresholdMs: saturationThreshold,
			})
		}

		if numInstances > 1 {
			// Print per-instance metrics to stdout (multi-instance only)
			for _, inst := range cs.Instances() {
				if err := inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, "", saturationDetector); err != nil {
					logrus.Fatalf("SaveResults for instance %s: %v", inst.ID(), err)
				}
			}
		}
		// Build aggregate output, inject goodput, then emit (#1413).
		aggregated := cs.AggregatedMetrics()
		clusterOutput := aggregated.BuildOutput("cluster", saturationDetector)
		emitGoodput(&clusterOutput, aggregated, cs.InjectedByClass(),
			float64(aggregated.SimEndedTime)/1e6, goodputTargets)
		if err := aggregated.EmitOutput(clusterOutput, metricsPath); err != nil {
			logrus.Fatalf("SaveResults: %v", err)
		}

		// Collect RawMetrics and compute fitness (PR9)
		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			scheduler,
			cs.RoutingRejections(),
			cs.EncodeRoutingRejections(),
			cs.InjectedByClass(),
		)

		rawMetrics.PD = cluster.CollectPDMetrics(
			cs.ParentRequests(),
			cs.AggregatedMetrics(),
			cs.PoolMembership(),
			cs.PerInstanceMetricsByID(),
		)
		rawMetrics.ShedByTier = cs.ShedByTier()                     // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
		rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth()       // Issue #882: gateway queue depth at horizon
		rawMetrics.GatewayQueueShed = cs.GatewayQueueShed()         // Issue #882: gateway queue shed count
		rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected() // Issue #1190: gateway queue rejected count
		rawMetrics.GatewayEvicted = cs.GatewayEvicted()             // Phase 4: in-flight eviction count (#1228)
		rawMetrics.GatewayExpired = cs.GatewayExpired()             // Phase 6: TTL expiration count (#1193)

		if rawMetrics.PD != nil && config.PDTransferContention {
			rawMetrics.PD.PeakConcurrentTransfers = cs.PeakConcurrentTransfers()
			rawMetrics.PD.MeanTransferQueueDepth = cs.MeanTransferQueueDepth()
		}

		if fitnessWeights != "" {
			weights, err := cluster.ParseFitnessWeights(fitnessWeights)
			if err != nil {
				logrus.Fatalf("Invalid fitness weights: %v", err)
			}
			fitness, fitErr := cluster.ComputeFitness(rawMetrics, weights)
			if fitErr != nil {
				logrus.Fatalf("Fitness evaluation failed: %v", fitErr)
			}
			fmt.Println("=== Fitness Evaluation ===")
			fmt.Printf("Score: %.6f\n", fitness.Score)
			// Sort keys for deterministic output order
			componentKeys := make([]string, 0, len(fitness.Components))
			for k := range fitness.Components {
				componentKeys = append(componentKeys, k)
			}
			sort.Strings(componentKeys)
			for _, k := range componentKeys {
				fmt.Printf("  %s: %.6f\n", k, fitness.Components[k])
			}
		}

		// Print anomaly counters if any detected
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueShed > 0 || rawMetrics.GatewayQueueRejected > 0 || rawMetrics.GatewayEvicted > 0 || rawMetrics.GatewayExpired > 0 || rawMetrics.EncodeRoutingRejections > 0 || rawMetrics.TimedOutRequests > 0 {
			fmt.Println("=== Anomaly Counters ===")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests (Admission): %d\n", rawMetrics.RejectedRequests)
			if len(rawMetrics.ShedByTier) > 0 {
				tierKeys := make([]string, 0, len(rawMetrics.ShedByTier))
				for k := range rawMetrics.ShedByTier {
					tierKeys = append(tierKeys, k)
				}
				sort.Strings(tierKeys) // R2/INV-6: deterministic output order
				for _, tier := range tierKeys {
					fmt.Printf("  Shed (%s): %d\n", tier, rawMetrics.ShedByTier[tier])
				}
			}
			fmt.Printf("Rejected Requests (Routing): %d\n", rawMetrics.RoutingRejections)
			fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
			fmt.Printf("Timed Out Requests: %d\n", rawMetrics.TimedOutRequests)
			fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
			if rawMetrics.GatewayQueueDepth > 0 {
				fmt.Printf("Gateway Queue Depth (horizon): %d\n", rawMetrics.GatewayQueueDepth)
			}
			if rawMetrics.GatewayQueueShed > 0 {
				fmt.Printf("Gateway Queue Shed: %d\n", rawMetrics.GatewayQueueShed)
			}
			if rawMetrics.GatewayQueueRejected > 0 {
				fmt.Printf("Gateway Queue Rejected: %d\n", rawMetrics.GatewayQueueRejected)
			}
			if rawMetrics.GatewayEvicted > 0 {
				fmt.Printf("Gateway Evicted (in-flight): %d\n", rawMetrics.GatewayEvicted)
			}
			if rawMetrics.GatewayExpired > 0 {
				fmt.Printf("Gateway Expired (TTL): %d\n", rawMetrics.GatewayExpired)
			}
			if rawMetrics.EncodeRoutingRejections > 0 {
				fmt.Printf("Encode Routing Rejections: %d\n", rawMetrics.EncodeRoutingRejections)
			}
		}

		// Print KV cache metrics if any nonzero (BC-1, BC-2)
		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		// Print per-SLO metrics. With goodput targets configured, the section prints
		// even for a single class (#1413, BC-5). Without goodput, the legacy
		// suppression for ≤1 class is preserved.
		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions, len(goodputTargets) > 0)

		// Print per-model metrics if requests carry model tags (Phase 1A, FR-011)
		perModelMetrics := cluster.ComputePerModelMetrics(cs.AggregatedMetrics())
		printPerModelMetrics(os.Stdout, perModelMetrics)

		// Print per-tenant fairness metrics if any request carries a tenant label (Phase 1B-2b, FR-010)
		perTenantMetrics := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
		printPerTenantMetrics(os.Stdout, perTenantMetrics)

		// Print session metrics if any request carries a session label (#1058)
		sessionMetrics := cluster.ComputeSessionMetrics(cs.AggregatedMetrics())
		printSessionMetrics(os.Stdout, sessionMetrics)

		// Print PD disaggregation metrics if disaggregation was active (PR4)
		printPDMetrics(os.Stdout, rawMetrics.PD, config.PDTransferContention)

		// Build and print trace summary if requested (BC-9)
		if cs.Trace() != nil && summarizeTrace {
			traceSummary := trace.Summarize(cs.Trace())
			fmt.Println("=== Trace Summary ===")
			fmt.Printf("Total Decisions: %d\n", traceSummary.TotalDecisions)
			fmt.Printf("  Admitted: %d\n", traceSummary.AdmittedCount)
			fmt.Printf("  Rejected: %d\n", traceSummary.RejectedCount)
			fmt.Printf("Unique Targets: %d\n", traceSummary.UniqueTargets)
			if len(traceSummary.TargetDistribution) > 0 {
				fmt.Println("Target Distribution:")
				targetKeys := make([]string, 0, len(traceSummary.TargetDistribution))
				for k := range traceSummary.TargetDistribution {
					targetKeys = append(targetKeys, k)
				}
				sort.Strings(targetKeys)
				for _, k := range targetKeys {
					fmt.Printf("  %s: %d\n", k, traceSummary.TargetDistribution[k])
				}
			}
			fmt.Printf("Mean Regret: %.6f\n", traceSummary.MeanRegret)
			fmt.Printf("Max Regret: %.6f\n", traceSummary.MaxRegret)
		}

		logrus.Info("Simulation complete.")
	},
}

// printKVCacheMetrics prints KV cache metrics to w when any value is nonzero.
func printKVCacheMetrics(w io.Writer, preemptionRate, cacheHitRate, kvThrashingRate float64) {
	if preemptionRate == 0 && cacheHitRate == 0 && kvThrashingRate == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== KV Cache Metrics ===")
	_, _ = fmt.Fprintf(w, "Preemption Rate: %.4f\n", preemptionRate)
	_, _ = fmt.Fprintf(w, "Cache Hit Rate: %.4f\n", cacheHitRate)
	_, _ = fmt.Fprintf(w, "KV Thrashing Rate: %.4f\n", kvThrashingRate)
}

// printPerSLOMetrics prints per-SLO-class latency distributions. Without
// goodput targets configured, the section is suppressed for ≤1 class (the
// legacy no-spurious-section behavior). With goodput targets configured, the
// section prints even for a single class so operators see the dimensions
// gating their goodput score (#1413, BC-5).
func printPerSLOMetrics(w io.Writer, sloMetrics map[string]*cluster.SLOMetrics, goodputConfigured bool) {
	if len(sloMetrics) == 0 {
		return
	}
	if len(sloMetrics) == 1 && !goodputConfigured {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-SLO Metrics ===")
	keys := make([]string, 0, len(sloMetrics))
	for k := range sloMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, cls := range keys {
		m := sloMetrics[cls]
		if m == nil {
			continue
		}
		_, _ = fmt.Fprintf(w, "  %s:\n", cls)
		_, _ = fmt.Fprintf(w, "    TTFT: mean=%.2f p99=%.2f (n=%d)\n", m.TTFT.Mean, m.TTFT.P99, m.TTFT.Count)
		_, _ = fmt.Fprintf(w, "    ITL:  mean=%.2f p99=%.2f (n=%d)\n", m.ITL.Mean, m.ITL.P99, m.ITL.Count)
		_, _ = fmt.Fprintf(w, "    E2E:  mean=%.2f p99=%.2f (n=%d)\n", m.E2E.Mean, m.E2E.P99, m.E2E.Count)
	}
}

// printPerModelMetrics prints per-model TTFT, E2E, and throughput.
// Follows the same pattern as printPerSLOMetrics (R2: sorted keys).
// No-op when perModelMetrics is nil or empty.
func printPerModelMetrics(w io.Writer, perModelMetrics map[string]*cluster.ModelMetrics) {
	if len(perModelMetrics) == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-Model Metrics ===")
	keys := make([]string, 0, len(perModelMetrics))
	for k := range perModelMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, model := range keys {
		m := perModelMetrics[model]
		if m == nil {
			continue
		}
		_, _ = fmt.Fprintf(w, "  %s:\n", model)
		_, _ = fmt.Fprintf(w, "    TTFT: p50=%.2f p99=%.2f (n=%d)\n", m.TTFT.P50, m.TTFT.P99, m.TTFT.Count)
		_, _ = fmt.Fprintf(w, "    E2E:  p50=%.2f p99=%.2f (n=%d)\n", m.E2E.P50, m.E2E.P99, m.E2E.Count)
		_, _ = fmt.Fprintf(w, "    Throughput: %.2f req/s, %.2f tok/s\n", m.ThroughputRPS, m.ThroughputTokenSec)
	}
}

// printPerTenantMetrics prints per-tenant request counts, token totals, and Jain fairness index.
// Follows the same pattern as printPerModelMetrics (R2: sorted keys).
// No-op when perTenantMetrics is nil or empty.
func printPerTenantMetrics(w io.Writer, perTenantMetrics map[string]*cluster.TenantMetrics) {
	if len(perTenantMetrics) == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-Tenant Metrics ===")
	keys := make([]string, 0, len(perTenantMetrics))
	for k := range perTenantMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	tokenMap := make(map[string]float64, len(perTenantMetrics))
	for _, tid := range keys {
		tm := perTenantMetrics[tid]
		_, _ = fmt.Fprintf(w, "  %s: requests=%d, tokens=%d\n", tid, tm.CompletedRequests, tm.TotalTokensServed)
		tokenMap[tid] = float64(tm.TotalTokensServed)
	}
	jain := cluster.JainFairnessIndex(tokenMap)
	_, _ = fmt.Fprintf(w, "  Jain Fairness Index: %.4f\n", jain)
}

// printSessionMetrics writes the session metrics section to w.
// No-op when sm is nil (single-turn workloads produce no session output).
func printSessionMetrics(w io.Writer, sm *cluster.SessionMetrics) {
	if sm == nil {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Session Metrics ===")
	_, _ = fmt.Fprintf(w, "  Sessions: %d\n", sm.SessionCount)
	if sm.TTFTCold.Count > 0 {
		_, _ = fmt.Fprintf(w, "  TTFT cold (round 0): mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.TTFTCold.Mean, sm.TTFTCold.P50, sm.TTFTCold.P95, sm.TTFTCold.P99, sm.TTFTCold.Count)
	}
	if sm.TTFTWarm.Count > 0 {
		_, _ = fmt.Fprintf(w, "  TTFT warm (round≥1): mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.TTFTWarm.Mean, sm.TTFTWarm.P50, sm.TTFTWarm.P95, sm.TTFTWarm.P99, sm.TTFTWarm.Count)
	}
	if sm.SessionDuration.Count > 0 {
		_, _ = fmt.Fprintf(w, "  Session duration:    mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.SessionDuration.Mean, sm.SessionDuration.P50, sm.SessionDuration.P95, sm.SessionDuration.P99, sm.SessionDuration.Count)
	}
}

// printPDMetrics prints the PD disaggregation metrics section when disaggregation was active.
// No-op when pd is nil (disaggregation inactive). When contentionEnabled, also prints
// peak concurrent transfers and mean transfer queue depth.
func printPDMetrics(w io.Writer, pd *cluster.PDMetrics, contentionEnabled bool) {
	if pd == nil {
		return
	}
	_, _ = fmt.Fprintln(w, "=== PD Metrics ===")
	_, _ = fmt.Fprintf(w, "Disaggregated Requests: %d\n", pd.DisaggregatedCount)
	_, _ = fmt.Fprintf(w, "Dropped at Decode KV: %d\n", pd.DroppedAtDecodeKV)
	_, _ = fmt.Fprintf(w, "Prefill Throughput: %.4f sub-req/s\n", pd.PrefillThroughput)
	_, _ = fmt.Fprintf(w, "Decode Throughput: %.4f sub-req/s\n", pd.DecodeThroughput)
	if pd.LoadImbalanceRatio == math.MaxFloat64 {
		_, _ = fmt.Fprintf(w, "Load Imbalance Ratio: inf (one pool idle)\n")
	} else {
		_, _ = fmt.Fprintf(w, "Load Imbalance Ratio: %.4f\n", pd.LoadImbalanceRatio)
	}
	if pd.ParentTTFT.Count > 0 {
		_, _ = fmt.Fprintf(w, "Parent TTFT (us): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
			pd.ParentTTFT.Mean, pd.ParentTTFT.P50, pd.ParentTTFT.P95, pd.ParentTTFT.P99)
	}
	if pd.TransferDuration.Count > 0 {
		_, _ = fmt.Fprintf(w, "KV Transfer Duration (us): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
			pd.TransferDuration.Mean, pd.TransferDuration.P50, pd.TransferDuration.P95, pd.TransferDuration.P99)
	}
	if contentionEnabled {
		_, _ = fmt.Fprintf(w, "Peak Concurrent Transfers: %d\n", pd.PeakConcurrentTransfers)
		_, _ = fmt.Fprintf(w, "Mean Transfer Queue Depth: %.4f\n", pd.MeanTransferQueueDepth)
	}
}

// Execute runs the CLI root command
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

// init sets up CLI flags and subcommands
func init() {
	registerSimConfigFlags(runCmd)

	// Workload generation flags (run-only)
	runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution)")

	runCmd.Flags().Float64Var(&rate, "rate", 1.0, "Requests arrival per second")
	runCmd.Flags().IntVar(&numRequests, "num-requests", 100, "Number of requests to generate")
	runCmd.Flags().IntVar(&concurrency, "concurrency", 0, "Number of concurrent virtual users (closed-loop, mutually exclusive with --rate)")
	runCmd.Flags().IntVar(&thinkTimeMs, "think-time-ms", 0, "Think time in ms between response and next request (concurrency mode)")
	runCmd.Flags().IntVar(&prefixTokens, "prefix-tokens", 0, "Prefix Token Count")
	runCmd.Flags().IntVar(&promptTokensMean, "prompt-tokens", defaultPromptMean, "Average Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensStdev, "prompt-tokens-stdev", defaultPromptStdev, "Stddev Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMin, "prompt-tokens-min", defaultPromptMin, "Min Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMax, "prompt-tokens-max", defaultPromptMax, "Max Prompt Token Count")
	runCmd.Flags().IntVar(&outputTokensMean, "output-tokens", defaultOutputMean, "Average Output Token Count")
	runCmd.Flags().IntVar(&outputTokensStdev, "output-tokens-stdev", defaultOutputStdev, "Stddev Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMin, "output-tokens-min", defaultOutputMin, "Min Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMax, "output-tokens-max", defaultOutputMax, "Max Output Token Count")
	runCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "Path to YAML workload specification file (overrides --workload)")
	runCmd.Flags().BoolVar(&lazyGeneration, "lazy-generation", false, "Alpha (#1441): stream requests from the workload generator instead of pre-generating the full slice. Default off. Supports every workload class — single-shot, single- and multi-session reasoning (#1458), concurrency clients (#1459), and time-varying / per-window workloads (#1460); no eager fallback.")
	runCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "Per-request deadline in seconds (default 300s matches the session-client default in computeDeadline). Negative = disabled; 0 is rejected. Consistent with blis observe: both commands reject 0.")
	runCmd.Flags().StringVar(&goodputSLOTTFT, "slo-ttft", "", "Per-class TTFT goodput thresholds (e.g. \"critical=100ms,standard=500ms\"). Precedence: CLI > trace header > workload spec.")
	runCmd.Flags().StringVar(&goodputSLOITL, "slo-itl", "", "Per-class mean ITL goodput thresholds (e.g. \"critical=50ms,standard=150ms\").")
	runCmd.Flags().StringVar(&goodputSLOE2E, "slo-e2e", "", "Per-class E2E goodput thresholds (e.g. \"critical=5s,standard=30s\").")

	// Run-specific export
	runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")
	runCmd.Flags().StringVar(&metricsPath, "metrics-path", "", "File to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats). Use --results-path on blis replay for per-request SimResult JSON.")
	runCmd.Flags().StringVar(&saturationReport, "saturation-report", "", "File to write saturation analysis JSON (backlog-drift classification)")

	// Post-hoc saturation detector flags (#1369)
	runCmd.Flags().StringVar(&postHocDetector, "post-hoc-detector", "none", "Post-hoc saturation detector: composite, threshold, none")
	runCmd.Flags().Float64Var(&saturationThreshold, "saturation-threshold-ms", 5000.0, "Threshold in ms for threshold detector (default 5000ms)")

	registerSaturationFlags(runCmd)

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
