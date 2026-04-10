package cmd

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
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
	// CLI flags for model, GPU, TP, vllm version
	model             string // LLM name
	gpu               string // GPU type
	tensorParallelism int    // TP value
	vllmVersion       string // vllm version

	// cluster config
	numInstances int // Number of instances in the cluster

	// online routing pipeline config
	admissionPolicy       string             // Admission policy name
	admissionLatency      int64              // Admission latency in microseconds
	routingLatency        int64              // Routing latency in microseconds
	tokenBucketCapacity   float64            // Token bucket capacity
	tokenBucketRefillRate float64            // Token bucket refill rate (tokens/second)
	tierShedThreshold      int                // Tier-shed overload threshold (0 = any load)
	tierShedMinPriority    int                // Tier-shed minimum admitted priority under overload
	tenantBudgets          map[string]float64 // Per-tenant fraction of total capacity (nil = no enforcement)
	sloPriorityOverrides   map[string]int     // SLO class → priority overrides (nil = GAIE defaults)

	// routing policy config (PR 6, evolved in PR17)
	routingPolicy  string // Routing policy name
	routingScorers string // Comma-separated name:weight pairs for weighted routing

	// Priority and scheduler config (PR7)
	priorityPolicy string // Priority policy name
	scheduler      string // Scheduler name

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

	// Tiered KV cache config (PR12)
	kvCPUBlocks             int64
	kvOffloadThreshold      float64
	kvTransferBandwidth     float64
	kvTransferBaseLatency   int64
	snapshotRefreshInterval int64
	cacheSignalDelay        int64
	gpuMemoryUtilization    float64

	// PD disaggregation config
	prefillInstances      int     // Number of instances dedicated to prefill
	decodeInstances       int     // Number of instances dedicated to decode
	pdDecider             string  // Disaggregation decider name
	pdTransferBandwidth   float64 // Inter-instance KV transfer bandwidth in GB/s
	pdTransferBaseLatency float64 // Inter-instance KV transfer base latency in ms
	pdTransferContention     bool    // Enable fair-share bandwidth contention model
	pdPrefixThreshold       int    // Non-cached token threshold for prefix-threshold decider
	pdDirectDecodeThreshold int    // Input token threshold for direct-to-decode decider
	prefillRoutingScorers   string // Scorer weights for prefill pool routing
	decodeRoutingScorers  string  // Scorer weights for decode pool routing

	// Flow control config (issue #882, GIE parity)
	flowControlEnabled              bool
	flowControlDetector             string
	flowControlDispatchOrder        string
	flowControlMaxQueueDepth        int
	flowControlQueueDepthThreshold  float64
	flowControlKVCacheUtilThreshold float64
	flowControlMaxConcurrency       int

	// Per-pool hardware override config
	prefillTP             int
	decodeTP              int
	prefillHardware       string
	decodeHardware        string
	prefillLatencyModel   string
	decodeLatencyModel    string
	prefillMaxModelLen    int64
	decodeMaxModelLen     int64

	// output file paths
	metricsPath string // File to write MetricsOutput JSON for blis run (--metrics-path)
	resultsPath string // File to write []SimResult JSON for blis replay (--results-path)

	// trace export
	traceOutput string // File prefix for TraceV2 export (<prefix>.yaml + <prefix>.csv)
)

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
	Backend     string           // resolved latency backend name
	ModelConfig sim.ModelConfig  // HF-derived model architecture config (zero for blackbox)
	HWConfig    sim.HardwareCalib // hardware calibration config (zero for blackbox)
	AlphaCoeffs []float64        // resolved alpha coefficients (local copy, not package-level)
	BetaCoeffs  []float64        // resolved beta coefficients (local copy, not package-level)
}

// resolveLatencyConfig resolves the latency backend configuration from CLI flags and
// defaults.yaml. It is called by both runCmd and replayCmd to ensure a single code path
// (R23: code path parity). This eliminates the R23 comment-sync markers in replay.go.
//
// What it does:
//   - Normalizes model name to lowercase
//   - Validates gpuMemoryUtilization and blockSizeTokens (used in KV auto-calc)
//   - Applies defaults.yaml for GPU, TP, and vllmVersion when not set via CLI
//   - Validates alpha/beta coefficients and auto-detects blackbox mode
//   - For roofline/crossmodel/trained-roofline: resolves model config folder and
//     hardware config, loads coefficients from defaults.yaml, auto-calculates
//     total-kv-blocks and max-model-len from the HF config
//   - For blackbox: loads coefficients and KV blocks from defaults.yaml, then
//     attempts auto-calculation from cached model config as a best-effort fallback
//
// Side effects (package-level vars mutated):
//
//	model, gpu, tensorParallelism, vllmVersion, modelConfigFolder, hwConfigPath,
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
			logrus.Fatalf("--alpha-coeffs requires --beta-coeffs. Both coefficient sets are needed for blackbox mode")
		}
		logrus.Fatalf("--beta-coeffs requires --alpha-coeffs. Both coefficient sets are needed for blackbox mode")
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
		backend = "blackbox"
		logrus.Infof("--alpha-coeffs and --beta-coeffs provided; using blackbox mode")
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

	// Early defaults resolution: load hardware/TP/vllmVersion from defaults.yaml
	// when not explicitly set via CLI flags.
	if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
		hardware, tp, version := GetDefaultSpecs(model)
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
		if vllmVersion == "" && len(version) > 0 {
			logrus.Warnf("Finding default values of vLLM version for model=%v", model)
			logrus.Warnf("Using default vLLM version=%v", version)
			vllmVersion = version
		}
	}

	kvBlocksFromDefaults := false

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
			logrus.Fatalf("Roofline mode (the default) requires %s. No defaults found in defaults.yaml for model=%s. "+
				"Provide these flags explicitly, or use --latency-model blackbox for offline coefficient-based estimation",
				strings.Join(missing, " and "), model)
		}
		// alphaChanged == betaChanged is guaranteed by the "both or neither" check above,
		// so checking betaChanged alone is sufficient to confirm both were provided.
		if cmd.Flags().Changed("latency-model") && betaChanged {
			logrus.Fatalf("--alpha-coeffs/--beta-coeffs cannot be used with --latency-model roofline. " +
				"Roofline computes step time analytically. " +
				"Use --latency-model blackbox if you want coefficient-based estimation")
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
		if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
			_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
			if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
				totalKVBlocks = kvBlocks
				kvBlocksFromDefaults = true
				logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
			}
		}
	}

	// --latency-model crossmodel
	if backend == "crossmodel" {
		var missing []string
		if gpu == "" {
			missing = append(missing, "--hardware (GPU type)")
		}
		if tensorParallelism <= 0 {
			missing = append(missing, "--tp (tensor parallelism)")
		}
		if len(missing) > 0 {
			logrus.Fatalf("--latency-model crossmodel requires %s. No defaults found in defaults.yaml for model=%s. "+
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
				logrus.Warnf("--latency-model crossmodel: failed to read %s: %v", defaultsFilePath, readErr)
			} else {
				var cfg Config
				decoder := yaml.NewDecoder(bytes.NewReader(data))
				decoder.KnownFields(true) // R10: strict YAML parsing
				if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
					logrus.Fatalf("--latency-model crossmodel: failed to parse %s: %v", defaultsFilePath, yamlErr)
				}
				if cfg.CrossModelDefaults != nil {
					if !cmd.Flags().Changed("beta-coeffs") {
						beta = cfg.CrossModelDefaults.BetaCoeffs
						logrus.Infof("--latency-model: loaded crossmodel beta coefficients from defaults.yaml")
					}
					if !cmd.Flags().Changed("alpha-coeffs") {
						alpha = cfg.CrossModelDefaults.AlphaCoeffs
						logrus.Infof("--latency-model: loaded crossmodel alpha coefficients from defaults.yaml")
					}
				}
			}
			_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
			if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
				totalKVBlocks = kvBlocks
				kvBlocksFromDefaults = true
				logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
			}
		}
		if !cmd.Flags().Changed("beta-coeffs") && (len(beta) < 4 || allZeros(beta)) {
			logrus.Fatalf("--latency-model crossmodel: no crossmodel_defaults found in %s and no --beta-coeffs provided. "+
				"Add crossmodel_defaults to defaults.yaml or provide --beta-coeffs explicitly", defaultsFilePath)
		}
	}

	// --latency-model trained-roofline
	if backend == "trained-roofline" {
		var missing []string
		if gpu == "" {
			missing = append(missing, "--hardware (GPU type)")
		}
		if tensorParallelism <= 0 {
			missing = append(missing, "--tp (tensor parallelism)")
		}
		if len(missing) > 0 {
			logrus.Fatalf("--latency-model trained-roofline requires %s. No defaults found in defaults.yaml for model=%s. "+
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
				logrus.Warnf("--latency-model trained-roofline: failed to read %s: %v", defaultsFilePath, readErr)
			} else {
				var cfg Config
				decoder := yaml.NewDecoder(bytes.NewReader(data))
				decoder.KnownFields(true) // R10: strict YAML parsing
				if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
					logrus.Fatalf("--latency-model trained-roofline: failed to parse %s: %v", defaultsFilePath, yamlErr)
				}
				if cfg.TrainedRooflineDefaults != nil {
					if !cmd.Flags().Changed("beta-coeffs") {
						beta = cfg.TrainedRooflineDefaults.BetaCoeffs
						logrus.Infof("--latency-model: loaded trained-roofline beta coefficients from defaults.yaml")
					}
					if !cmd.Flags().Changed("alpha-coeffs") {
						alpha = cfg.TrainedRooflineDefaults.AlphaCoeffs
						logrus.Infof("--latency-model: loaded trained-roofline alpha coefficients from defaults.yaml")
					}
				}
			}
			_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
			if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
				totalKVBlocks = kvBlocks
				kvBlocksFromDefaults = true
				logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
			}
		}
		// Validate trained-roofline coefficients.
		// trained_roofline.go uses betaCoeffs[0..6] = exactly 7 coefficients.
		if !cmd.Flags().Changed("beta-coeffs") && (len(beta) < 7 || allZeros(beta)) {
			logrus.Fatalf("--latency-model trained-roofline: no trained_roofline_defaults found in %s and no --beta-coeffs provided.", defaultsFilePath)
		}
		if allZeros(alpha) && !cmd.Flags().Changed("alpha-coeffs") {
			logrus.Warnf("--latency-model trained-roofline: no trained alpha coefficients found; " +
				"QueueingTime, PostDecodeFixedOverhead, and OutputTokenProcessingTime will use zero alpha (may underestimate TTFT/E2E)")
		}
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
			_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
			if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
				totalKVBlocks = kvBlocks
				kvBlocksFromDefaults = true
				logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
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

	// --latency-model blackbox: load coefficients and KV blocks from defaults.yaml.
	if backend == "blackbox" && !cmd.Flags().Changed("alpha-coeffs") && !cmd.Flags().Changed("beta-coeffs") {
		newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
		alpha, beta = newAlpha, newBeta
		if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
			totalKVBlocks = kvBlocks
			kvBlocksFromDefaults = true
		}
	}
	// Blackbox: best-effort KV-block auto-calculation from cached model config when
	// neither CLI flag nor defaults.yaml provided a value. Falls through silently if
	// configs unavailable — totalKVBlocks validation catches 0.
	if backend == "blackbox" && !cmd.Flags().Changed("total-kv-blocks") && !kvBlocksFromDefaults {
		baseDir := filepath.Dir(defaultsFilePath)
		cachedDir, dirErr := bundledModelConfigDir(model, baseDir)
		if dirErr == nil {
			hfPath := filepath.Join(cachedDir, "config.json")
			if _, statErr := os.Stat(hfPath); statErr == nil {
				hfCfg, parseErr := latency.ParseHFConfig(hfPath)
				if parseErr == nil {
					mc, mcErr := latency.GetModelConfigFromHF(hfCfg)
					if mcErr == nil {
						applyWeightPrecisionFallback(mc, model, hfCfg.Raw)
					}
					resolvedHW, hwPathErr := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
					if mcErr == nil && hwPathErr == nil {
						hc, hcErr := latency.GetHWConfig(resolvedHW, gpu)
						if hcErr == nil && hc.MemoryGiB > 0 {
							kvParams, kvErr := latency.ExtractKVCapacityParams(hfCfg)
							if kvErr == nil {
								autoBlocks, calcErr := latency.CalculateKVBlocks(*mc, hc, tensorParallelism, blockSizeTokens, gpuMemoryUtilization, kvParams)
								if calcErr == nil {
									totalKVBlocks = autoBlocks
									logrus.Infof("--latency-model blackbox: auto-calculated total-kv-blocks=%d from cached model config", totalKVBlocks)
								}
							}
						}
					}
				}
			}
		}
	}
	if backend == "blackbox" && allZeros(alpha) && allZeros(beta) {
		logrus.Fatalf("No trained coefficients found for model=%s, GPU=%s, TP=%d. "+
			"Provide --alpha-coeffs/--beta-coeffs, use --latency-model roofline, crossmodel, trained-roofline, or trained-physics",
			model, gpu, tensorParallelism)
	}

	// Analytical backends: parse HF config, extract model/hardware config, auto-calc KV blocks and max-model-len.
	if backend == "roofline" || backend == "crossmodel" || backend == "trained-roofline" || backend == "trained-physics" {
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

		if backend == "trained-roofline" {
			warnTrainedRooflineQuantization(&modelConfig)
		}
		if backend == "roofline" && modelConfig.NumLocalExperts > 1 {
			logrus.Infof("--latency-model: MoE model detected (%d experts, top_%d). "+
				"Roofline models per-expert FLOPs and active weights; dispatch overhead is not modeled",
				modelConfig.NumLocalExperts, modelConfig.NumExpertsPerTok)
		}

		// KV capacity auto-calculation. Precedence: (1) --total-kv-blocks CLI flag,
		// (2) defaults.yaml match, (3) auto-calculate from model architecture + GPU memory.
		if !cmd.Flags().Changed("total-kv-blocks") && !kvBlocksFromDefaults {
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
				autoBlocks, calcErr := latency.CalculateKVBlocks(modelConfig, hwConfig, tensorParallelism, blockSizeTokens, gpuMemoryUtilization, kvParams)
				if calcErr != nil {
					logrus.Warnf("--latency-model: KV capacity auto-calculation failed: %v. "+
						"Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", calcErr, totalKVBlocks)
				} else {
					totalKVBlocks = autoBlocks
					logrus.Infof("--gpu-memory-utilization: %.2f used for KV block auto-calculation", gpuMemoryUtilization)
					logrus.Infof("--latency-model: auto-calculated total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, block_size=%d, MoE=%v)",
						totalKVBlocks, hwConfig.MemoryGiB, tensorParallelism, blockSizeTokens, kvParams.IsMoE)
				}
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
// Side effects: may write admissionPolicy, routingPolicy, priorityPolicy, scheduler,
// tokenBucketCapacity, tokenBucketRefillRate, tierShedThreshold, tierShedMinPriority,
// tenantBudgets package-level vars (from policy bundle).
//
// Returns the parsed scorer configs for weighted routing (caller uses these in
// DeploymentConfig.RoutingScorerConfigs). Per-pool scorer configs (PD disaggregation)
// are NOT handled here — they remain inline in runCmd.
func resolvePolicies(cmd *cobra.Command) []sim.ScorerConfig {
	var bundleScorerConfigs []sim.ScorerConfig

	// Load policy bundle if specified (R18: CLI flags override YAML values)
	if policyConfigPath != "" {
		bundle, err := sim.LoadPolicyBundle(policyConfigPath)
		if err != nil {
			logrus.Fatalf("Failed to load policy config: %v", err)
		}
		if err := bundle.Validate(); err != nil {
			logrus.Fatalf("Invalid policy config: %v", err)
		}
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
		}
		if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
			routingPolicy = bundle.Routing.Policy
		}
		bundleScorerConfigs = bundle.Routing.Scorers
		if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
			priorityPolicy = bundle.Priority.Policy
		}
		if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
			scheduler = bundle.Scheduler
		}
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
	if !sim.IsValidAdmissionPolicy(admissionPolicy) {
		logrus.Fatalf("Unknown admission policy %q. Valid: %s", admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
	}
	if !sim.IsValidRoutingPolicy(routingPolicy) {
		logrus.Fatalf("Unknown routing policy %q. Valid: %s", routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
	}
	if !sim.IsValidPriorityPolicy(priorityPolicy) {
		logrus.Fatalf("Unknown priority policy %q. Valid: %s", priorityPolicy, strings.Join(sim.ValidPriorityPolicyNames(), ", "))
	}
	if !sim.IsValidScheduler(scheduler) {
		logrus.Fatalf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
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
		if flowControlDispatchOrder != "fifo" && flowControlDispatchOrder != "priority" {
			logrus.Fatalf("--dispatch-order must be 'fifo' or 'priority', got %q", flowControlDispatchOrder)
		}
		if flowControlMaxQueueDepth < 0 {
			logrus.Fatalf("--max-gateway-queue-depth must be >= 0, got %d", flowControlMaxQueueDepth)
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

	logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s",
		admissionPolicy, routingPolicy, priorityPolicy, scheduler)

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

	return parsedScorerConfigs
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
	cmd.Flags().StringVar(&vllmVersion, "vllm-version", "", "vLLM version")
	cmd.Flags().StringVar(&latencyModelBackend, "latency-model", "roofline", "Latency model backend: roofline (default), blackbox, crossmodel, trained-roofline, trained-physics")
	cmd.Flags().Int64Var(&maxModelLen, "max-model-len", 0, "Max total sequence length (input + output); 0 = unlimited. Auto-derived from HF config for roofline/crossmodel when not set.")

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

	// Priority and scheduler config (PR7)
	cmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based, inverted-slo")
	cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")

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
	cmd.Flags().Int64Var(&snapshotRefreshInterval, "snapshot-refresh-interval", 0, "Prometheus snapshot refresh interval for all instance metrics in microseconds (0 = immediate)")
	cmd.Flags().Int64Var(&cacheSignalDelay, "cache-signal-delay", cluster.DefaultCacheSignalDelay, "Propagation delay for prefix cache signals in microseconds. Only affects precise-prefix-cache and no-hit-lru scorers; no effect on other routing policies. Default 2s matches production llm-d speculative TTL. Set to 0 for oracle mode (live cache state).")
	cmd.Flags().Float64Var(&gpuMemoryUtilization, "gpu-memory-utilization", 0.9, "Fraction of GPU memory to use for KV cache, in the range (0, 1.0]. Default: 0.9 (90%)")

	// PD disaggregation config
	cmd.Flags().IntVar(&prefillInstances, "prefill-instances", 0, "Number of instances dedicated to prefill (0 = disabled)")
	cmd.Flags().IntVar(&decodeInstances, "decode-instances", 0, "Number of instances dedicated to decode (0 = disabled)")
	cmd.Flags().StringVar(&pdDecider, "pd-decider", "never", "PD disaggregation decider: never (default), always, prefix-threshold, direct-to-decode")
	cmd.Flags().Float64Var(&pdTransferBandwidth, "pd-transfer-bandwidth", 25.0, "PD KV transfer bandwidth in GB/s (NIXL RDMA default)")
	cmd.Flags().Float64Var(&pdTransferBaseLatency, "pd-transfer-base-latency", 0.05, "PD KV transfer base latency in ms")
	cmd.Flags().BoolVar(&pdTransferContention, "pd-transfer-contention", false, "Enable fair-share bandwidth contention model for concurrent KV transfers (INV-P2-2)")
	cmd.Flags().IntVar(&pdPrefixThreshold, "pd-prefix-threshold", 512, "Non-cached token threshold for prefix-threshold decider (>= 0); disaggregate when non-cached tokens exceed this value")
	cmd.Flags().IntVar(&pdDirectDecodeThreshold, "pd-direct-decode-threshold", 256, "Input token threshold for direct-to-decode (>= 0): requests with fewer than threshold tokens go direct to decode; requests with >= threshold tokens are disaggregated")
	cmd.Flags().StringVar(&prefillRoutingScorers, "prefill-routing-scorers", "", "Scorer weights for prefill pool routing (e.g., queue-depth:2,kv-utilization:2)")
	cmd.Flags().StringVar(&decodeRoutingScorers, "decode-routing-scorers", "", "Scorer weights for decode pool routing (e.g., queue-depth:2,kv-utilization:2)")

	// Flow control config (issue #882, GIE parity)
	cmd.Flags().BoolVar(&flowControlEnabled, "flow-control", false, "Enable gateway queue with saturation-gated dispatch (GIE flow control)")
	cmd.Flags().StringVar(&flowControlDetector, "saturation-detector", "never", "Saturation detector: "+strings.Join(sim.ValidSaturationDetectorNames(), ", "))
	cmd.Flags().StringVar(&flowControlDispatchOrder, "dispatch-order", "fifo", "Gateway queue dispatch order: fifo, priority")
	cmd.Flags().IntVar(&flowControlMaxQueueDepth, "max-gateway-queue-depth", 0, "Max gateway queue depth (0=unlimited)")
	cmd.Flags().Float64Var(&flowControlQueueDepthThreshold, "queue-depth-threshold", 5, "Queue depth threshold for utilization detector")
	cmd.Flags().Float64Var(&flowControlKVCacheUtilThreshold, "kv-cache-util-threshold", 0.8, "KV cache utilization threshold for utilization detector")
	cmd.Flags().IntVar(&flowControlMaxConcurrency, "max-concurrency", 100, "Max concurrency per instance for concurrency detector")

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
		// Analytical backends populate ModelConfig from HF config.json; blackbox does not.
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
		if lr.Backend == "roofline" || lr.Backend == "crossmodel" || lr.Backend == "trained-roofline" || lr.Backend == "trained-physics" {
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
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolPrefillTP, blockSizeTokens, gpuMemoryUtilization, kvParamsPool)
							if calcErr != nil {
								logrus.Warnf("--prefill-tp/--prefill-hardware: KV capacity auto-calculation failed for prefill pool: %v; prefill pool will use global total-kv-blocks=%d", calcErr, totalKVBlocks)
							} else {
								prefillOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--prefill-tp/--prefill-hardware: auto-calculated prefill pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolPrefillTP)
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
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolDecodeTP, blockSizeTokens, gpuMemoryUtilization, kvParamsPool)
							if calcErr != nil {
								logrus.Warnf("--decode-tp/--decode-hardware: KV capacity auto-calculation failed for decode pool: %v; decode pool will use global total-kv-blocks=%d", calcErr, totalKVBlocks)
							} else {
								decodeOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--decode-tp/--decode-hardware: auto-calculated decode pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolDecodeTP)
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
			// R3: Standalone validation for distribution token bounds (BC-1, BC-2)
			if promptTokensMin < 1 {
				logrus.Fatalf("--prompt-tokens-min must be >= 1, got %d", promptTokensMin)
			}
			if promptTokensMax < 1 {
				logrus.Fatalf("--prompt-tokens-max must be >= 1, got %d", promptTokensMax)
			}
			if outputTokensMin < 1 {
				logrus.Fatalf("--output-tokens-min must be >= 1, got %d", outputTokensMin)
			}
			if outputTokensMax < 1 {
				logrus.Fatalf("--output-tokens-max must be >= 1, got %d", outputTokensMax)
			}
			if promptTokensStdev < 0 {
				logrus.Fatalf("--prompt-tokens-stdev must be >= 0, got %d", promptTokensStdev)
			}
			if outputTokensStdev < 0 {
				logrus.Fatalf("--output-tokens-stdev must be >= 0, got %d", outputTokensStdev)
			}
			if promptTokensMin > promptTokensMax {
				logrus.Fatalf("--prompt-tokens-min (%d) must be <= --prompt-tokens-max (%d)", promptTokensMin, promptTokensMax)
			}
			if outputTokensMin > outputTokensMax {
				logrus.Fatalf("--output-tokens-min (%d) must be <= --output-tokens-max (%d)", outputTokensMin, outputTokensMax)
			}
			if promptTokensMean > promptTokensMax || promptTokensMean < promptTokensMin || promptTokensStdev > promptTokensMax || promptTokensStdev < promptTokensMin {
				logrus.Fatalf("prompt-tokens and prompt-tokens-stdev should be in range [prompt-tokens-min, prompt-tokens-max]")
			}
			if outputTokensMean > outputTokensMax || outputTokensMean < outputTokensMin || outputTokensStdev > outputTokensMax || outputTokensStdev < outputTokensMin {
				logrus.Fatalf("output-tokens and output-tokens-stdev should be in range [output-tokens-min, output-tokens-max]")
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

		// Resolve maxRequests: spec.NumRequests as default, CLI --num-requests overrides
		maxRequests := spec.NumRequests
		if cmd.Flags().Changed("num-requests") {
			maxRequests = int64(numRequests)
		}

		// Guard against unbounded generation
		if maxRequests <= 0 && simulationHorizon == math.MaxInt64 {
			logrus.Fatalf("Workload requires either num_requests or --horizon to bound generation")
		}

		wl, err := workload.GenerateWorkload(spec, simulationHorizon, maxRequests)
		if err != nil {
			logrus.Fatalf("Failed to generate workload: %v", err)
		}
		preGeneratedRequests = wl.Requests
		if len(wl.Sessions) > 0 {
			sessionMgr = workload.NewSessionManager(wl.Sessions)
			if wl.FollowUpBudget >= 0 {
				sessionMgr.SetFollowUpBudget(wl.FollowUpBudget)
			}
			logrus.Infof("Generated %d requests + %d session blueprints (closed-loop)", len(wl.Requests), len(wl.Sessions))
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
		parsedScorerConfigs := resolvePolicies(cmd)

		// PD disaggregation validation (R3: validate at CLI boundary)
		if prefillInstances < 0 {
			logrus.Fatalf("--prefill-instances must be >= 0, got %d", prefillInstances)
		}
		if decodeInstances < 0 {
			logrus.Fatalf("--decode-instances must be >= 0, got %d", decodeInstances)
		}
		if !sim.IsValidDisaggregationDecider(pdDecider) {
			logrus.Fatalf("Unknown PD decider %q. Valid: %s", pdDecider, strings.Join(sim.ValidDisaggregationDeciderNames(), ", "))
		}
		if err := cluster.ValidatePoolTopology(prefillInstances, decodeInstances, numInstances); err != nil {
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
		if pdDecider == "direct-to-decode" && pdDirectDecodeThreshold < 0 {
			logrus.Fatalf("--pd-direct-decode-threshold must be >= 0, got %d", pdDirectDecodeThreshold)
		}
		if pdDecider == "direct-to-decode" && pdDirectDecodeThreshold == 0 {
			logrus.Warnf("--pd-direct-decode-threshold=0 means all non-empty requests will be disaggregated (equivalent to --pd-decider=always for non-empty inputs). Did you intend a non-zero threshold?")
		}
		if pdDecider != "direct-to-decode" && cmd.Flags().Changed("pd-direct-decode-threshold") {
			logrus.Warnf("--pd-direct-decode-threshold=%d is ignored when --pd-decider=%q (only applies to the direct-to-decode decider)", pdDirectDecodeThreshold, pdDecider)
		}
		if pdDecider != "" && pdDecider != "never" && prefillInstances == 0 {
			logrus.Warnf("--pd-decider=%q has no effect because --prefill-instances=0 (disaggregation is disabled); set --prefill-instances and --decode-instances to enable", pdDecider)
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

		// Unified cluster path (used for all values of numInstances)
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: simulationHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:         sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:       sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
				ModelHardwareConfig: sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, lr.Backend, maxModelLen),
				PolicyConfig:        sim.NewPolicyConfig(priorityPolicy, scheduler),
			},
			NumInstances:            numInstances,
			AdmissionPolicy:         admissionPolicy,
			AdmissionLatency:        admissionLatency,
			RoutingLatency:          routingLatency,
			TokenBucketCapacity:     tokenBucketCapacity,
			TokenBucketRefillRate:   tokenBucketRefillRate,
			RoutingPolicy:           routingPolicy,
			RoutingScorerConfigs:    parsedScorerConfigs,
			TraceLevel:              traceLevel,
			CounterfactualK:         counterfactualK,
			SnapshotRefreshInterval: snapshotRefreshInterval,
			CacheSignalDelay:        cacheSignalDelay,
			PrefillInstances:        prefillInstances,
			DecodeInstances:         decodeInstances,
			PDDecider:               pdDecider,
			PDPrefixThreshold:       pdPrefixThreshold,
			PDDirectDecodeThreshold: pdDirectDecodeThreshold,
			PDTransferBandwidthGBps: pdTransferBandwidth,
			PDTransferBaseLatencyMs: pdTransferBaseLatency,
			PDTransferContention:    pdTransferContention,
			PrefillScorerConfigs:    prefillScorerCfgs,
			DecodeScorerConfigs:     decodeScorerCfgs,
			PrefillOverrides:        prefillOverrides,
			DecodeOverrides:         decodeOverrides,
			TierShedThreshold:       tierShedThreshold,
			TierShedMinPriority:     tierShedMinPriority,
			TenantBudgets:           tenantBudgets,
			SLOPriorityOverrides:    sloPriorityOverrides,
			FlowControlEnabled:              flowControlEnabled,
			FlowControlDetector:             flowControlDetector,
			FlowControlDispatchOrder:        flowControlDispatchOrder,
			FlowControlMaxQueueDepth:        flowControlMaxQueueDepth,
			FlowControlQueueDepthThreshold:  flowControlQueueDepthThreshold,
			FlowControlKVCacheUtilThreshold: flowControlKVCacheUtilThreshold,
			FlowControlMaxConcurrency:       flowControlMaxConcurrency,
		}
		var followUpRequests []*sim.Request
		var onRequestDone func(*sim.Request, int64) []*sim.Request
		if sessionMgr != nil {
			baseCb := sessionMgr.OnComplete
			if traceOutput != "" {
				// Wrap callback to accumulate follow-up requests for trace export
				onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
					followUps := baseCb(req, clock)
					followUpRequests = append(followUpRequests, followUps...)
					return followUps
				}
			} else {
				onRequestDone = baseCb
			}
		}
		cs := cluster.NewClusterSimulator(config, preGeneratedRequests, onRequestDone)
		if err := cs.Run(); err != nil {
			logrus.Fatalf("Simulation failed: %v", err)
		}

		// Wall-clock timing on stderr (BC-6); stdout remains deterministic (BC-7)
		logrus.Infof("Simulation wall-clock time: %.3fs", time.Since(startTime).Seconds())

		// Export trace if requested (BC-1, BC-7)
		if traceOutput != "" {
			allRequests := make([]*sim.Request, 0, len(preGeneratedRequests)+len(followUpRequests))
			allRequests = append(allRequests, preGeneratedRequests...)
			allRequests = append(allRequests, followUpRequests...)
			// Sort by arrival time so RequestIDs (array indices) are arrival-ordered
			sort.SliceStable(allRequests, func(i, j int) bool {
				return allRequests[i].ArrivalTime < allRequests[j].ArrivalTime
			})
			records := workload.RequestsToTraceRecords(allRequests)
			header := &workload.TraceHeader{
				Version:      2,
				TimeUnit:     "microseconds",
				Mode:         "generated",
				WorkloadSeed: &spec.Seed,
			}
			if err := workload.ExportTraceV2(header, records, traceOutput+".yaml", traceOutput+".csv"); err != nil {
				logrus.Fatalf("Trace export failed: %v", err)
			}
			logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", traceOutput, traceOutput, len(records))
		}

		if numInstances > 1 {
			// Print per-instance metrics to stdout (multi-instance only)
			for _, inst := range cs.Instances() {
				if err := inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, ""); err != nil {
					logrus.Fatalf("SaveResults for instance %s: %v", inst.ID(), err)
				}
			}
		}
		// Save aggregated metrics (prints to stdout + saves to file if metricsPath set)
		if err := cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, metricsPath); err != nil {
			logrus.Fatalf("SaveResults: %v", err)
		}

		// Collect RawMetrics and compute fitness (PR9)
		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			priorityPolicy,
			cs.RoutingRejections(),
		)

		rawMetrics.PD = cluster.CollectPDMetrics(
			cs.ParentRequests(),
			cs.AggregatedMetrics(),
			cs.PoolMembership(),
			cs.PerInstanceMetricsByID(),
		)
		rawMetrics.ShedByTier = cs.ShedByTier()                             // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
		rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth()               // Issue #882: gateway queue depth at horizon
		rawMetrics.GatewayQueueShed = cs.GatewayQueueShed()                 // Issue #882: gateway queue shed count

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
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueShed > 0 {
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
			fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
			if rawMetrics.GatewayQueueDepth > 0 {
				fmt.Printf("Gateway Queue Depth (horizon): %d\n", rawMetrics.GatewayQueueDepth)
			}
			if rawMetrics.GatewayQueueShed > 0 {
				fmt.Printf("Gateway Queue Shed: %d\n", rawMetrics.GatewayQueueShed)
			}
		}

		// Print KV cache metrics if any nonzero (BC-1, BC-2)
		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		// Print per-SLO metrics if multiple SLO classes present (BC-3, BC-4, BC-10)
		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions)

		// Print per-model metrics if requests carry model tags (Phase 1A, FR-011)
		perModelMetrics := cluster.ComputePerModelMetrics(cs.AggregatedMetrics())
		printPerModelMetrics(os.Stdout, perModelMetrics)

		// Print per-tenant fairness metrics if any request carries a tenant label (Phase 1B-2b, FR-010)
		perTenantMetrics := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
		printPerTenantMetrics(os.Stdout, perTenantMetrics)

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

// printPerSLOMetrics prints per-SLO-class latency distributions when multiple classes exist.
func printPerSLOMetrics(w io.Writer, sloMetrics map[string]*cluster.SLOMetrics) {
	if len(sloMetrics) <= 1 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-SLO Metrics ===")
	// Sort keys for deterministic output (antipattern rule 2)
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

	// Run-specific export
	runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")
	runCmd.Flags().StringVar(&metricsPath, "metrics-path", "", "File to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats). Use --results-path on blis replay for per-request SimResult JSON.")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
