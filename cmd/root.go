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
	admissionPolicy       string  // Admission policy name
	admissionLatency      int64   // Admission latency in microseconds
	routingLatency        int64   // Routing latency in microseconds
	tokenBucketCapacity   float64 // Token bucket capacity
	tokenBucketRefillRate float64 // Token bucket refill rate (tokens/second)

	// routing policy config (PR 6, evolved in PR17)
	routingPolicy  string // Routing policy name
	routingScorers string // Comma-separated name:weight pairs for weighted routing

	// Priority and scheduler config (PR7)
	priorityPolicy string // Priority policy name
	scheduler      string // Scheduler name

	// Policy bundle config (PR8)
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

	// results file path
	resultsPath string // File to save BLIS results to

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
	cmd.Flags().StringVar(&latencyModelBackend, "latency-model", "roofline", "Latency model backend: roofline (default), blackbox, crossmodel, trained-roofline")
	cmd.Flags().Int64Var(&maxModelLen, "max-model-len", 0, "Max total sequence length (input + output); 0 = unlimited. Auto-derived from HF config for roofline/crossmodel when not set.")

	// Cluster config
	cmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

	// Online routing pipeline config
	cmd.Flags().StringVar(&admissionPolicy, "admission-policy", "always-admit", "Admission policy: always-admit, token-bucket, reject-all")
	cmd.Flags().Int64Var(&admissionLatency, "admission-latency", 0, "Admission latency in microseconds")
	cmd.Flags().Int64Var(&routingLatency, "routing-latency", 0, "Routing latency in microseconds")
	cmd.Flags().Float64Var(&tokenBucketCapacity, "token-bucket-capacity", 10000, "Token bucket capacity")
	cmd.Flags().Float64Var(&tokenBucketRefillRate, "token-bucket-refill-rate", 1000, "Token bucket refill rate (tokens/second)")

	// Routing policy config
	cmd.Flags().StringVar(&routingPolicy, "routing-policy", "round-robin", "Routing policy: round-robin, least-loaded, weighted, always-busiest")
	cmd.Flags().StringVar(&routingScorers, "routing-scorers", "", "Scorer weights for weighted routing (e.g., queue-depth:2,kv-utilization:2,load-balance:1). Default: prefix-affinity:3,queue-depth:2,kv-utilization:2")

	// Priority and scheduler config (PR7)
	cmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based, inverted-slo")
	cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")

	// Policy bundle config (PR8)
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

	// Results path
	cmd.Flags().StringVar(&resultsPath, "results-path", "", "File to save BLIS results to")
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

		// Load alpha/beta coeffs from coefficients.yaml
		alphaCoeffs, betaCoeffs := alphaCoeffs, betaCoeffs

		// Validate --latency-model flag
		if !sim.IsValidLatencyBackend(latencyModelBackend) {
			logrus.Fatalf("unknown --latency-model %q; valid options: %s",
				latencyModelBackend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
		}

		// Local copy of CLI-provided backend.
		// Default is "roofline" both at the CLI level (Cobra flag) and the
		// library level (factory case "", "roofline": dispatch).
		backend := latencyModelBackend

		// Alpha and beta coefficients must be provided together or not at all.
		// Alpha controls queueing time, beta controls step time — one without
		// the other produces nonsensical results.
		alphaChanged := cmd.Flags().Changed("alpha-coeffs")
		betaChanged := cmd.Flags().Changed("beta-coeffs")
		if alphaChanged != betaChanged {
			if alphaChanged {
				logrus.Fatalf("--alpha-coeffs requires --beta-coeffs. " +
					"Both coefficient sets are needed for blackbox mode")
			}
			logrus.Fatalf("--beta-coeffs requires --alpha-coeffs. " +
				"Both coefficient sets are needed for blackbox mode")
		}

		// Validate coefficient values for NaN/Inf/negative (R3: CLI flags AND library constructors).
		// Library-level validateCoeffs() also checks, but CLI-level gives user-friendly flag names.
		for i, c := range alphaCoeffs {
			if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
				logrus.Fatalf("--alpha-coeffs[%d] must be a finite non-negative number, got %v", i, c)
			}
		}
		for i, c := range betaCoeffs {
			if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
				logrus.Fatalf("--beta-coeffs[%d] must be a finite non-negative number, got %v", i, c)
			}
		}

		// If both alpha and beta coefficients are explicitly provided but
		// --latency-model was not, auto-switch to blackbox. Providing both
		// coefficient sets is a clear signal the user wants trained blackbox
		// estimation, not analytical roofline.
		if !cmd.Flags().Changed("latency-model") && alphaChanged && betaChanged {
			backend = "blackbox"
			logrus.Infof("--alpha-coeffs and --beta-coeffs provided; using blackbox mode")
		}

		var modelConfig = sim.ModelConfig{}
		var hwConfig = sim.HardwareCalib{}

		// Normalize model name to lowercase for consistent lookups. All defaults.yaml
		// keys, hf_repo lookups, bundled model_configs/ directories, and coefficient
		// matching use lowercase names. This runs unconditionally (even with explicit
		// --alpha-coeffs/--beta-coeffs) to ensure output metadata and logs use a
		// canonical form. Note: HuggingFace repo names are case-sensitive, so the
		// hf_repo mapping in defaults.yaml preserves the original casing for API calls.
		model = strings.ToLower(model)

		// Early defaults resolution: load hardware/TP/vllm-version from defaults.yaml
		// when not explicitly set via CLI flags. This runs BEFORE analytical backend
		// blocks so that the default roofline mode can find --hardware and --tp values
		// for models registered in defaults.yaml (e.g., qwen/qwen3-14b).
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

		// Track whether defaults.yaml provided KV blocks (used to skip auto-calc).
		// Precedence: (1) --total-kv-blocks CLI flag, (2) defaults.yaml, (3) auto-calc.
		kvBlocksFromDefaults := false

		// --latency-model roofline: auto-resolve model config and hardware config
		if backend == "roofline" {
			var missing []string
			if gpu == "" {
				missing = append(missing, "--hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				missing = append(missing, "--tp (tensor parallelism)")
			}
			if len(missing) > 0 {
				logrus.Fatalf("Roofline mode (the default) requires %s. "+
					"No defaults found in defaults.yaml for model=%s. "+
					"Provide these flags explicitly, or use --latency-model blackbox for offline coefficient-based estimation",
					strings.Join(missing, " and "), model)
			}

			// Hard error if user explicitly requested roofline AND provided coefficients.
			// Roofline computes step time analytically — beta coefficients are meaningless.
			// Note: alphaChanged == betaChanged is guaranteed by the "both or neither"
			// check above, so checking betaChanged implies both were provided.
			if cmd.Flags().Changed("latency-model") && betaChanged {
				logrus.Fatalf("--alpha-coeffs/--beta-coeffs cannot be used with --latency-model roofline. " +
					"Roofline computes step time analytically. " +
					"Use --latency-model blackbox if you want coefficient-based estimation")
			}

			// Log when explicit overrides interact with --latency-model roofline
			if modelConfigFolder != "" {
				logrus.Infof("--latency-model: explicit --model-config-folder takes precedence over auto-resolution")
			}
			if hwConfigPath != "" {
				logrus.Infof("--latency-model: explicit --hardware-config takes precedence over auto-resolution")
			}

			// Resolve model config folder (cache → HF fetch → bundled fallback)
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved

			// Resolve hardware config (explicit → bundled default)
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW

			// Load totalKVBlocks from defaults.yaml if available. Roofline does NOT
			// use alpha coefficients — QueueingTime and OutputTokenProcessingTime
			// are zero by design.
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				// vllmVersion already resolved by early defaults block (lines 248-265).
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
		}

		// --latency-model crossmodel: auto-resolve model config and load global coefficients
		if backend == "crossmodel" {
			var missing []string
			if gpu == "" {
				missing = append(missing, "--hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				missing = append(missing, "--tp (tensor parallelism)")
			}
			if len(missing) > 0 {
				logrus.Fatalf("--latency-model crossmodel requires %s. "+
					"No defaults found in defaults.yaml for model=%s. "+
					"Provide these flags explicitly", strings.Join(missing, " and "), model)
			}

			// Resolve model config folder (same auto-fetch chain as roofline)
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved

			// Resolve hardware config (validates --hardware flag; crossmodel doesn't use HWConfig
			// for step time but loading validates the GPU type for future use)
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW

			// Load crossmodel defaults from defaults.yaml (R18: use Flags().Changed, not allZeros)
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
							betaCoeffs = cfg.CrossModelDefaults.BetaCoeffs
							logrus.Infof("--latency-model: loaded crossmodel beta coefficients from defaults.yaml")
						}
						if !cmd.Flags().Changed("alpha-coeffs") {
							alphaCoeffs = cfg.CrossModelDefaults.AlphaCoeffs
							logrus.Infof("--latency-model: loaded crossmodel alpha coefficients from defaults.yaml")
						}
					}
				}
				// Also load KV blocks from per-model config if available.
				// vllmVersion already resolved by early defaults block (lines 248-265).
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
			// Validate crossmodel coefficients were loaded (catches missing file, missing section, etc.)
			if !cmd.Flags().Changed("beta-coeffs") && (len(betaCoeffs) < 4 || allZeros(betaCoeffs)) {
				logrus.Fatalf("--latency-model crossmodel: no crossmodel_defaults found in %s and no --beta-coeffs provided. "+
					"Add crossmodel_defaults to defaults.yaml or provide --beta-coeffs explicitly",
					defaultsFilePath)
			}
		}

		// --latency-model trained-roofline: auto-resolve model config and load global coefficients
		if backend == "trained-roofline" {
			var missing []string
			if gpu == "" {
				missing = append(missing, "--hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				missing = append(missing, "--tp (tensor parallelism)")
			}
			if len(missing) > 0 {
				logrus.Fatalf("--latency-model trained-roofline requires %s. "+
					"No defaults found in defaults.yaml for model=%s. "+
					"Provide these flags explicitly", strings.Join(missing, " and "), model)
			}

			// Resolve model config folder (same auto-fetch chain as roofline/crossmodel)
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved

			// Resolve hardware config
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW

			// Load trained-roofline defaults from defaults.yaml (R18: use Flags().Changed)
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
							betaCoeffs = cfg.TrainedRooflineDefaults.BetaCoeffs
							logrus.Infof("--latency-model: loaded trained-roofline beta coefficients from defaults.yaml")
						}
						if !cmd.Flags().Changed("alpha-coeffs") {
							alphaCoeffs = cfg.TrainedRooflineDefaults.AlphaCoeffs
							logrus.Infof("--latency-model: loaded trained-roofline alpha coefficients from defaults.yaml")
						}
					}
				}
				// Also load KV blocks from per-model config if available.
				// vllmVersion already resolved by early defaults block (lines 248-265).
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
			// Validate trained-roofline coefficients were loaded
			if !cmd.Flags().Changed("beta-coeffs") && (len(betaCoeffs) < 7 || allZeros(betaCoeffs)) {
				logrus.Fatalf("--latency-model trained-roofline: no trained_roofline_defaults found in %s and no --beta-coeffs provided. "+
					"Add trained_roofline_defaults to defaults.yaml or provide --beta-coeffs explicitly",
					defaultsFilePath)
			}
			// Warn if alpha coefficients are zero (user provided --beta-coeffs but not --alpha-coeffs)
			if allZeros(alphaCoeffs) && !cmd.Flags().Changed("alpha-coeffs") {
				logrus.Warnf("--latency-model trained-roofline: no trained alpha coefficients found; " +
					"QueueingTime, PostDecodeFixedOverhead, and OutputTokenProcessingTime will use zero alpha (may underestimate TTFT/E2E)")
			}
		}

		// Blackbox mode: load alpha/beta coefficients and KV blocks from defaults.yaml.
		// Skipped for analytical backends (roofline, crossmodel, trained-roofline) which
		// load their own coefficients in their respective blocks above.
		// Note: the old guard also checked len(modelConfigFolder)==0 && len(hwConfigPath)==0
		// as part of implicit roofline detection. With explicit --latency-model selection,
		// those guards are no longer needed — blackbox ignores model-config/hw-config paths.
		if backend == "blackbox" && !cmd.Flags().Changed("alpha-coeffs") && !cmd.Flags().Changed("beta-coeffs") {
			newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
			alphaCoeffs, betaCoeffs = newAlpha, newBeta
			if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
				totalKVBlocks = kvBlocks
				kvBlocksFromDefaults = true
			}
		}
		// Blackbox mode: auto-calculate KV blocks when neither CLI flag nor
		// defaults.yaml provided a value. Uses cached model config (no HF fetch)
		// and bundled hardware config. Best-effort — falls through silently if
		// configs are unavailable (totalKVBlocks validation at line ~770 catches 0).
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
									autoBlocks, calcErr := latency.CalculateKVBlocks(*mc, hc, tensorParallelism, blockSizeTokens, 0.9, kvParams)
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
		// Zero-coefficients safety guard: prevents silently running with zero step times
		// when blackbox mode has no trained coefficients (would produce meaningless results).
		if backend == "blackbox" && allZeros(alphaCoeffs) && allZeros(betaCoeffs) {
			logrus.Fatalf("No trained coefficients found for model=%s, GPU=%s, TP=%d. "+
				"Provide --alpha-coeffs/--beta-coeffs, use --latency-model roofline, crossmodel, or trained-roofline",
				model, gpu, tensorParallelism)
		}
		// Analytical backends (roofline, crossmodel): parse HFConfig once, use for
		// both model config extraction and KV capacity auto-calculation.
		if backend == "roofline" || backend == "crossmodel" || backend == "trained-roofline" {
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

			// Trained-roofline uses hardcoded FP16 bytesPerElement (matching its training
			// pipeline); quantized weight precision is not applied to step time estimates.
			if backend == "trained-roofline" {
				warnTrainedRooflineQuantization(&modelConfig)
			}

			// MoE informational note: roofline models per-routed-expert FLOPs (top_k active)
			// and all-expert weight bandwidth (E experts loaded from HBM per step).
			// Shared expert FLOPs/weights and gate/router weights are NOT modeled in
			// roofline step time (they are included in KV capacity weight estimation).
			// Remaining approximations: no expert-load-imbalance or MoE dispatch overhead.
			if backend == "roofline" && modelConfig.NumLocalExperts > 1 {
				logrus.Infof("--latency-model: MoE model detected (%d experts, top_%d). "+
					"Roofline models per-expert FLOPs and active weights; dispatch overhead is not modeled",
					modelConfig.NumLocalExperts, modelConfig.NumExpertsPerTok)
			}

			// KV capacity auto-calculation: derive total-kv-blocks from model + hardware.
			// Precedence: (1) --total-kv-blocks CLI flag, (2) defaults.yaml match,
			// (3) auto-calculate from model architecture + GPU memory.
			if !cmd.Flags().Changed("total-kv-blocks") && !kvBlocksFromDefaults {
				kvParams, kvParamsErr := latency.ExtractKVCapacityParams(hfConfig)
				if kvParamsErr != nil {
					logrus.Warnf("--latency-model: could not extract KV capacity params: %v. "+
						"Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", kvParamsErr, totalKVBlocks)
				} else if hwConfig.MemoryGiB <= 0 {
					logrus.Warnf("--latency-model: GPU memory capacity not available in hardware config; using current total-kv-blocks=%d. "+
						"Add MemoryGiB to hardware_config.json or pass --total-kv-blocks explicitly", totalKVBlocks)
				} else {
					if kvParams.HiddenAct == "" {
						logrus.Infof("--latency-model: hidden_act not set in config.json; assuming SwiGLU (3-matrix MLP) for weight estimation")
					}
					autoBlocks, calcErr := latency.CalculateKVBlocks(modelConfig, hwConfig, tensorParallelism, blockSizeTokens, 0.9, kvParams)
					if calcErr != nil {
						logrus.Warnf("--latency-model: KV capacity auto-calculation failed: %v. "+
							"Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", calcErr, totalKVBlocks)
					} else {
						totalKVBlocks = autoBlocks
						logrus.Infof("--latency-model: auto-calculated total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, block_size=%d, MoE=%v)",
							totalKVBlocks, hwConfig.MemoryGiB, tensorParallelism, blockSizeTokens, kvParams.IsMoE)
					}
				}
			}

			// Auto-derive --max-model-len from HF config when not explicitly set (R18).
			// Mirrors vLLM's _auto_fit_max_model_len(): uses max_position_embeddings,
			// applies rope_scaling factor for older models, then caps at KV-feasible max.
			if !cmd.Flags().Changed("max-model-len") {
				maxPosEmb := hfConfig.MustGetInt("max_position_embeddings", 0)
				if maxPosEmb > 0 {
					maxModelLen = int64(maxPosEmb)

					// Apply rope_scaling factor via pure function (extracted for testability).
					// See applyRopeScaling godoc for blacklist details and vLLM fidelity notes.
					modelType, _ := hfConfig.Raw["model_type"].(string)
					scaled, applied := applyRopeScaling(maxPosEmb, modelType, hfConfig.Raw["rope_scaling"])
					if applied {
						// Recover rope type and factor for diagnostic logging
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
						// Factor not applied — could be excluded type, missing/invalid factor, or overflow
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
			// Without this, auto-derived max_position_embeddings can exceed KV capacity
			// for models with large context windows on small GPU configs (e.g., 128K context, TP=1).
			// Overflow-safe KV feasible max: compare in block space to avoid int64 multiplication overflow.
			// Instead of computing totalKVBlocks * blockSizeTokens (which could overflow for extreme configs),
			// we check whether ceil(maxModelLen / blockSizeTokens) > totalKVBlocks — the same comparison
			// used in NewSimulator's startup validation.
			if maxModelLen > 0 && blockSizeTokens > 0 {
				blocksNeeded := maxModelLen / blockSizeTokens
				if maxModelLen%blockSizeTokens != 0 {
					blocksNeeded++
				}
				if blocksNeeded > totalKVBlocks {
					kvFeasibleMax := totalKVBlocks * blockSizeTokens // product bounded by maxModelLen (blocksNeeded > totalKVBlocks)
					logrus.Warnf("--latency-model: max-model-len %d exceeds KV capacity (%d blocks × %d tokens); capping to %d tokens",
						maxModelLen, totalKVBlocks, blockSizeTokens, kvFeasibleMax)
					maxModelLen = kvFeasibleMax
				}
			}
		}

		// R3: Validate --max-model-len
		if maxModelLen < 0 {
			logrus.Fatalf("--max-model-len must be >= 0, got %d", maxModelLen)
		}

		// R3: Validate workload generation flags (before any synthesis path consumes them)
		if numRequests < 0 {
			logrus.Fatalf("--num-requests must be >= 0, got %d", numRequests)
		}
		if prefixTokens < 0 {
			logrus.Fatalf("--prefix-tokens must be >= 0, got %d", prefixTokens)
		}

		// Workload configuration — all paths synthesize a v2 WorkloadSpec
		// and generate requests via workload.GenerateRequests (BC-10).
		var spec *workload.WorkloadSpec
		var preGeneratedRequests []*sim.Request
		var sessionMgr *workload.SessionManager

		if workloadSpecPath != "" {
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
		if blockSizeTokens <= 0 {
			logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
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

		// Load policy bundle if specified (BC-6: CLI flags override YAML values)
		var bundleScorerConfigs []sim.ScorerConfig // captured for use in weighted routing setup
		if policyConfigPath != "" {
			bundle, err := sim.LoadPolicyBundle(policyConfigPath)
			if err != nil {
				logrus.Fatalf("Failed to load policy config: %v", err)
			}
			if err := bundle.Validate(); err != nil {
				logrus.Fatalf("Invalid policy config: %v", err)
			}

			// Apply bundle values as defaults; CLI flags override via Changed().
			// Pointer fields (nil = not set in YAML) correctly distinguish "0.0" from "unset".
			if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
				admissionPolicy = bundle.Admission.Policy
			}
			if bundle.Admission.TokenBucketCapacity != nil && !cmd.Flags().Changed("token-bucket-capacity") {
				tokenBucketCapacity = *bundle.Admission.TokenBucketCapacity
			}
			if bundle.Admission.TokenBucketRefillRate != nil && !cmd.Flags().Changed("token-bucket-refill-rate") {
				tokenBucketRefillRate = *bundle.Admission.TokenBucketRefillRate
			}
			if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
				routingPolicy = bundle.Routing.Policy
			}
			// Capture scorer configs from YAML bundle (already validated; CLI --routing-scorers overrides)
			bundleScorerConfigs = bundle.Routing.Scorers
			if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
				priorityPolicy = bundle.Priority.Policy
			}
			if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
				scheduler = bundle.Scheduler
			}
		}

		// Validate policy names (catches CLI typos before they become panics)
		// Token bucket parameter validation (R3: validate when policy is selected)
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
		if kvCPUBlocks > 0 && (kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0)) {
			logrus.Fatalf("--kv-transfer-bandwidth must be a finite value > 0 when --kv-cpu-blocks > 0, got %f", kvTransferBandwidth)
		}
		if kvTransferBaseLatency < 0 {
			logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
		}
		if snapshotRefreshInterval < 0 {
			logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
		}
		if admissionLatency < 0 {
			logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
		}
		if routingLatency < 0 {
			logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
		}

		// Log active policy configuration so users can verify which policies are in effect
		logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s",
			admissionPolicy, routingPolicy, priorityPolicy, scheduler)
		// Parse and validate scorer configuration for weighted routing
		var parsedScorerConfigs []sim.ScorerConfig
		if routingPolicy == "weighted" {
			if routingScorers != "" {
				var err error
				parsedScorerConfigs, err = sim.ParseScorerConfigs(routingScorers)
				if err != nil {
					logrus.Fatalf("Invalid --routing-scorers: %v", err)
				}
			} else if len(bundleScorerConfigs) > 0 {
				// Use YAML bundle scorers directly (no string round-trip)
				parsedScorerConfigs = bundleScorerConfigs
			}
			// Log active scorer configuration
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
			logrus.Infof("Token bucket: capacity=%.0f, refill-rate=%.0f",
				tokenBucketCapacity, tokenBucketRefillRate)
		}

		// Log configuration after all config sources (CLI, workload spec, policy bundle) are resolved
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v",
			totalKVBlocks, simulationHorizon, alphaCoeffs, betaCoeffs)

		startTime := time.Now() // Get current time (start)

		// Unified cluster path (used for all values of numInstances)
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: simulationHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					nil, kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:         sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:       sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs),
				ModelHardwareConfig: sim.NewModelHardwareConfig(modelConfig, hwConfig, model, gpu, tensorParallelism, backend, maxModelLen),
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
				Version:  2,
				TimeUnit: "microseconds",
				Mode:     "generated",
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
		// Save aggregated metrics (prints to stdout + saves to file if resultsPath set)
		if err := cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, resultsPath); err != nil {
			logrus.Fatalf("SaveResults: %v", err)
		}

		// Collect RawMetrics and compute fitness (PR9)
		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			priorityPolicy,
		)

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
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 {
			fmt.Println("=== Anomaly Counters ===")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests: %d\n", rawMetrics.RejectedRequests)
			fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
			fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
		}

		// Print KV cache metrics if any nonzero (BC-1, BC-2)
		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		// Print per-SLO metrics if multiple SLO classes present (BC-3, BC-4, BC-10)
		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions)

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
	runCmd.Flags().IntVar(&prefixTokens, "prefix-tokens", 0, "Prefix Token Count")
	runCmd.Flags().IntVar(&promptTokensMean, "prompt-tokens", 512, "Average Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensStdev, "prompt-tokens-stdev", 256, "Stddev Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMin, "prompt-tokens-min", 2, "Min Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMax, "prompt-tokens-max", 7000, "Max Prompt Token Count")
	runCmd.Flags().IntVar(&outputTokensMean, "output-tokens", 512, "Average Output Token Count")
	runCmd.Flags().IntVar(&outputTokensStdev, "output-tokens-stdev", 256, "Stddev Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMin, "output-tokens-min", 2, "Min Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMax, "output-tokens-max", 7000, "Max Output Token Count")
	runCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "Path to YAML workload specification file (overrides --workload)")

	// Run-specific export
	runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
