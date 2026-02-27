package cmd

import (
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
	workloadType              string    // Workload type (chatbot, summarization, contentgen, multidoc, distribution, traces)
	tracesWorkloadFilePath    string    // Workload filepath for traces workload type.
	maxModelLength            int       // Max request length (input + output tokens) to be handled
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
	rooflineActive            bool      // Runtime state: whether roofline step time is active (set by --roofline flag OR implicit detection)
	rooflineFlag              bool      // CLI --roofline flag: auto-fetch HF config and resolve hardware config

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
)

// rootCmd is the base command for the CLI
var rootCmd = &cobra.Command{
	Use:   "blis",
	Short: "BLIS — Blackbox Inference Simulator for LLM serving systems",
}

// check if all values in the coefficients list is 0 (default)
func AllZeros(values []float64) bool {
	for _, v := range values {
		if v != 0 {
			return false
		}
	}
	return true
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

		// Default: Do not use Roofline estimates for step time
		rooflineActive = false

		var modelConfig = sim.ModelConfig{}
		var hwConfig = sim.HardwareCalib{}

		// Normalize model name to lowercase for consistent lookups. All defaults.yaml
		// keys, hf_repo lookups, bundled model_configs/ directories, and coefficient
		// matching use lowercase names. This runs unconditionally (even with explicit
		// --alpha-coeffs/--beta-coeffs) to ensure output metadata and logs use a
		// canonical form. Note: HuggingFace repo names are case-sensitive, so the
		// hf_repo mapping in defaults.yaml preserves the original casing for API calls.
		model = strings.ToLower(model)

		// --roofline flag: auto-resolve model config and hardware config
		if rooflineFlag {
			if gpu == "" {
				logrus.Fatalf("--roofline requires --hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				logrus.Fatalf("--roofline requires --tp > 0")
			}

			// Warn if user also provided explicit beta coefficients — roofline replaces
			// step time estimation. Alpha coefficients are still used for queueing time
			// and output token processing time.
			if !AllZeros(betaCoeffs) {
				logrus.Warnf("--roofline replaces --beta-coeffs with analytical step time estimation. " +
					"Alpha coefficients are still used for queueing time and output token processing")
			}

			// Log when explicit overrides interact with --roofline
			if modelConfigFolder != "" {
				logrus.Infof("--roofline: explicit --model-config-folder takes precedence over auto-resolution")
			}
			if hwConfigPath != "" {
				logrus.Infof("--roofline: explicit --hardware-config takes precedence over auto-resolution")
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

			// Explicitly activate roofline mode (design doc step 4).
			// Do NOT rely on downstream "all coefficients zero" heuristic.
			// Note (I10): When --roofline is set with explicit --alpha-coeffs/--beta-coeffs,
			// SimConfig will contain both non-zero coefficients and roofline=true. This is
			// intentional: rooflineActive controls the latency model factory, and the
			// AllZeros() check at the implicit-detection block is guarded by !rooflineActive.
			rooflineActive = true

			// Load alpha coefficients and totalKVBlocks from defaults.yaml.
			// Roofline replaces beta (step time) but still needs alpha
			// (queueing time, output token processing) and KV cache capacity.
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				if vllmVersion == "" {
					_, _, ver := GetDefaultSpecs(model)
					if len(ver) > 0 {
						vllmVersion = ver
					}
				}
				defAlpha, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if AllZeros(alphaCoeffs) && !AllZeros(defAlpha) {
					alphaCoeffs = defAlpha
					logrus.Infof("--roofline: loaded alpha coefficients from defaults.yaml for queueing time estimation")
				}
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					logrus.Infof("--roofline: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				} else if !cmd.Flags().Changed("total-kv-blocks") {
					logrus.Warnf("--roofline: no trained total-kv-blocks found for model=%s, GPU=%s, TP=%d; "+
						"using default %d. Consider setting --total-kv-blocks explicitly for accurate KV cache simulation",
						model, gpu, tensorParallelism, totalKVBlocks)
				}
				if AllZeros(alphaCoeffs) {
					logrus.Warnf("--roofline: no trained alpha coefficients found for model=%s, GPU=%s, TP=%d; "+
						"queueing time and output token processing time will use zero alpha (may underestimate TTFT/ITL)",
						model, gpu, tensorParallelism)
				}
			} else {
				logrus.Warnf("--roofline: defaults file %s not found; alpha coefficients and total-kv-blocks not loaded. "+
					"Queueing time estimation will use zero alpha coefficients", defaultsFilePath)
			}
		}

		if AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) && len(modelConfigFolder) == 0 && len(hwConfigPath) == 0 { // default all 0s
			// GPU, TP, vLLM version configuration
			hardware, tp, version := GetDefaultSpecs(model) // pick default config for tp, GPU, vllmVersion

			// if tp args are missing, fall back to default
			if tensorParallelism == 0 && tp > 0 {
				logrus.Warnf("Finding default values of TP for model=%v\n", model)
				logrus.Warnf("Using default tp=%v", tp)
				tensorParallelism = tp
			}

			// if hardware args are missing, fall back to default
			if gpu == "" && len(hardware) > 0 {
				logrus.Warnf("Finding default values of hardware for model=%v\n", model)
				logrus.Warnf("Using default GPU=%v", hardware)
				gpu = hardware
			}

			// if vllm-version args are missing, fall back to default
			if vllmVersion == "" && len(version) > 0 {
				logrus.Warnf("Finding default values of vLLM version for model=%v\n", model)
				logrus.Warnf("Using default vLLM version=%v", version)
				vllmVersion = version
			}

			newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
			alphaCoeffs, betaCoeffs = newAlpha, newBeta
			if !cmd.Flags().Changed("total-kv-blocks") {
				totalKVBlocks = kvBlocks
			}
		}
		// Load roofline model/hardware configs when roofline mode is active.
		// Two activation paths: (1) explicit --roofline flag, (2) implicit detection
		// when trained coefficients are all-zero and config paths are provided.
		if !rooflineActive && AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
			logrus.Warnf("Trying roofline approach for model=%v, TP=%v, GPU=%v, vllmVersion=%v\n", model, tensorParallelism, gpu, vllmVersion)
			if len(modelConfigFolder) > 0 && len(hwConfigPath) > 0 && len(gpu) > 0 && tensorParallelism > 0 {
				rooflineActive = true
			} else if len(modelConfigFolder) == 0 {
				logrus.Fatalf("Please provide model config folder containing config.json for model=%v\n", model)
			} else if len(hwConfigPath) == 0 {
				logrus.Fatalf("Please provide hardware config path (e.g. hardware_config.json)\n")
			}
		}
		if rooflineActive {
			hfPath := filepath.Join(modelConfigFolder, "config.json")
			mc, err := latency.GetModelConfig(hfPath)
			if err != nil {
				logrus.Fatalf("Failed to load model config: %v", err)
			}
			modelConfig = *mc
			hc, err := latency.GetHWConfig(hwConfigPath, gpu)
			if err != nil {
				logrus.Fatalf("Failed to load hardware config: %v", err)
			}
			hwConfig = hc

			// Warn about known roofline estimation limitations
			if modelConfig.BytesPerParam > 0 && modelConfig.BytesPerParam <= 1 {
				logrus.Warnf("--roofline: model reports %.0f byte(s)/param (possible quantization). "+
					"Roofline step time estimates may be inaccurate for quantized models",
					modelConfig.BytesPerParam)
			}
			// Check for MoE model indicators in the raw HF config
			if hfRawBytes, readErr := os.ReadFile(hfPath); readErr == nil {
				if strings.Contains(string(hfRawBytes), `"num_local_experts"`) {
					logrus.Warnf("--roofline: model appears to be MoE (Mixture-of-Experts). " +
						"Roofline estimation assumes dense transformers and may overestimate MoE latency")
				}
			}
		}

		// Workload configuration — all paths synthesize a v2 WorkloadSpec
		// and generate requests via workload.GenerateRequests (BC-10).
		var spec *workload.WorkloadSpec
		var preGeneratedRequests []*sim.Request

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
		} else if workloadType == "traces" {
			// CSV trace path → synthesize v2 spec (lossy conversion)
			if tracesWorkloadFilePath == "" {
				logrus.Fatalf("--workload-traces-filepath is required when using --workload traces")
			}
			logrus.Warn("--workload traces uses lossy CSV conversion (averaged token lengths, constant arrival). " +
				"For faithful trace replay, use --workload-spec with a trace v2 YAML file instead.")
			var err error
			spec, err = workload.SynthesizeFromCSVTrace(tracesWorkloadFilePath, simulationHorizon)
			if err != nil {
				logrus.Fatalf("Failed to convert CSV trace: %v", err)
			}
			spec.Seed = seed
		} else if workloadType == "distribution" {
			// Distribution mode → synthesize v2 spec from CLI flags
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
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
				PrefixTokens: wl.PrefixTokens,
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

		reqs, err := workload.GenerateRequests(spec, simulationHorizon, maxRequests)
		if err != nil {
			logrus.Fatalf("Failed to generate workload: %v", err)
		}
		preGeneratedRequests = reqs
		logrus.Infof("Generated %d requests via unified workload pipeline", len(reqs))

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
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:         sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:       sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs),
				ModelHardwareConfig: sim.NewModelHardwareConfig(modelConfig, hwConfig, model, gpu, tensorParallelism, rooflineActive),
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
		cs := cluster.NewClusterSimulator(config, preGeneratedRequests)
		if err := cs.Run(); err != nil {
			logrus.Fatalf("Simulation failed: %v", err)
		}

		// Wall-clock timing on stderr (BC-6); stdout remains deterministic (BC-7)
		logrus.Infof("Simulation wall-clock time: %.3fs", time.Since(startTime).Seconds())

		if numInstances > 1 {
			// Print per-instance metrics to stdout (multi-instance only)
			for _, inst := range cs.Instances() {
				inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, "")
			}
		}
		// Save aggregated metrics (prints to stdout + saves to file if resultsPath set)
		cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, resultsPath)

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
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.DroppedUnservable > 0 {
			fmt.Println("=== Anomaly Counters ===")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests: %d\n", rawMetrics.RejectedRequests)
			fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
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

	runCmd.Flags().Int64Var(&seed, "seed", 42, "Seed for random request generation")
	runCmd.Flags().Int64Var(&simulationHorizon, "horizon", math.MaxInt64, "Total simulation horizon (in ticks)")
	runCmd.Flags().StringVar(&logLevel, "log", "warn", "Log level for diagnostic messages (trace, debug, info, warn, error, fatal, panic). Simulation results always print to stdout regardless of this setting.")
	runCmd.Flags().StringVar(&defaultsFilePath, "defaults-filepath", "defaults.yaml", "Path to default constants - trained coefficients, default specs and workloads")
	runCmd.Flags().StringVar(&modelConfigFolder, "model-config-folder", "", "Path to folder containing config.json")
	runCmd.Flags().StringVar(&hwConfigPath, "hardware-config", "", "Path to file containing hardware config")
	runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution, traces)")
	runCmd.Flags().StringVar(&tracesWorkloadFilePath, "workload-traces-filepath", "", "Workload filepath for traces workload type.")

	// vLLM server configs
	runCmd.Flags().Int64Var(&totalKVBlocks, "total-kv-blocks", 1000000, "Total number of KV cache blocks")
	runCmd.Flags().Int64Var(&maxRunningReqs, "max-num-running-reqs", 256, "Maximum number of requests running together")
	runCmd.Flags().Int64Var(&maxScheduledTokens, "max-num-scheduled-tokens", 2048, "Maximum total number of new tokens across running requests")
	runCmd.Flags().Float64SliceVar(&betaCoeffs, "beta-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated list of beta coefficients")
	runCmd.Flags().Float64SliceVar(&alphaCoeffs, "alpha-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated alpha coefficients (alpha0,alpha1) for processing delays")
	runCmd.Flags().Int64Var(&blockSizeTokens, "block-size-in-tokens", 16, "Number of tokens contained in a KV cache block")
	runCmd.Flags().IntVar(&maxModelLength, "max-model-len", 2048, "Max request length (input + output tokens)")
	runCmd.Flags().Int64Var(&longPrefillTokenThreshold, "long-prefill-token-threshold", 0, "Max length of prefill beyond which chunked prefill is triggered")

	// BLIS model configs
	runCmd.Flags().StringVar(&model, "model", "", "LLM name")
	runCmd.Flags().StringVar(&gpu, "hardware", "", "GPU type")
	runCmd.Flags().IntVar(&tensorParallelism, "tp", 0, "Tensor parallelism")
	runCmd.Flags().StringVar(&vllmVersion, "vllm-version", "", "vLLM version")
	runCmd.Flags().BoolVar(&rooflineFlag, "roofline", false, "Enable roofline mode with auto-fetch of HuggingFace config.json and bundled hardware config")

	// GuideLLM-style distribution-based workload generation config
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

	// Cluster config
	runCmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

	// Online routing pipeline config
	runCmd.Flags().StringVar(&admissionPolicy, "admission-policy", "always-admit", "Admission policy: always-admit, token-bucket, reject-all")
	runCmd.Flags().Int64Var(&admissionLatency, "admission-latency", 0, "Admission latency in microseconds")
	runCmd.Flags().Int64Var(&routingLatency, "routing-latency", 0, "Routing latency in microseconds")
	runCmd.Flags().Float64Var(&tokenBucketCapacity, "token-bucket-capacity", 10000, "Token bucket capacity")
	runCmd.Flags().Float64Var(&tokenBucketRefillRate, "token-bucket-refill-rate", 1000, "Token bucket refill rate (tokens/second)")

	// Routing policy config
	runCmd.Flags().StringVar(&routingPolicy, "routing-policy", "round-robin", "Routing policy: round-robin, least-loaded, weighted, prefix-affinity, always-busiest")
	runCmd.Flags().StringVar(&routingScorers, "routing-scorers", "", "Scorer weights for weighted routing (e.g., queue-depth:2,kv-utilization:2,load-balance:1). Default: prefix-affinity:3,queue-depth:2,kv-utilization:2")

	// Priority and scheduler config (PR7)
	runCmd.Flags().StringVar(&priorityPolicy, "priority-policy", "constant", "Priority policy: constant, slo-based, inverted-slo")
	runCmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")

	// Policy bundle config (PR8)
	runCmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")

	// Fitness evaluation config (PR9)
	runCmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs (e.g., throughput:0.5,p99_ttft:0.3)")

	// Decision trace config (PR13)
	runCmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
	runCmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Number of counterfactual candidates per routing decision")
	runCmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")

	// Workload spec config (PR10)
	runCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "Path to YAML workload specification file (overrides --workload)")

	// Tiered KV cache (PR12)
	runCmd.Flags().Int64Var(&kvCPUBlocks, "kv-cpu-blocks", 0, "CPU tier KV cache blocks (0 = disabled, single-tier mode). Typical: 1/3 of --total-kv-blocks")
	runCmd.Flags().Float64Var(&kvOffloadThreshold, "kv-offload-threshold", 0.9, "GPU utilization (0-1) above which blocks are offloaded to CPU. Default: offload when GPU >90% full")
	runCmd.Flags().Float64Var(&kvTransferBandwidth, "kv-transfer-bandwidth", 100.0, "CPU↔GPU transfer rate in blocks per tick. Higher = faster transfers")
	runCmd.Flags().Int64Var(&kvTransferBaseLatency, "kv-transfer-base-latency", 0, "Fixed per-transfer latency in ticks for CPU↔GPU KV transfers (0 = no fixed cost)")
	runCmd.Flags().Int64Var(&snapshotRefreshInterval, "snapshot-refresh-interval", 0, "KV utilization snapshot refresh interval in microseconds (0 = immediate refresh every call)")

	// Results path
	runCmd.Flags().StringVar(&resultsPath, "results-path", "", "File to save BLIS results to")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
