package cmd

import (
	"fmt"
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
	maxPrompts                int       // Number of requests
	prefixTokens              int       // Prefix Token Count
	promptTokensMean          int       // Average Prompt Token Count
	promptTokensStdev         int       // Stdev Prompt Token Count
	promptTokensMin           int       // Min Prompt Token Count
	promptTokensMax           int       // Max Prompt Token Count
	outputTokensMean          int       // Average Output Token Count
	outputTokensStdev         int       // Stdev Output Token Count
	outputTokensMin           int       // Min Output Token Count
	outputTokensMax           int       // Max Output Token Count
	roofline                  bool      // Whether to use roofline stepTime or not

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

	// routing policy config (PR 6)
	routingPolicy      string  // Routing policy name
	routingCacheWeight float64 // Cache affinity weight for weighted scoring
	routingLoadWeight  float64 // Load balance weight for weighted scoring

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
	kvCPUBlocks         int64
	kvOffloadThreshold  float64
	kvTransferBandwidth float64

	// results file path
	resultsPath string // File to save BLIS results to
)

// rootCmd is the base command for the CLI
var rootCmd = &cobra.Command{
	Use:   "inference-sim",
	Short: "Discrete-event simulator for inference platforms",
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
		roofline = false

		var modelConfig = sim.ModelConfig{}
		var hwConfig = sim.HardwareCalib{}

		if AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) && len(modelConfigFolder) == 0 && len(hwConfigPath) == 0 { // default all 0s
			// convert model name to lowercase
			model = strings.ToLower(model)

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
			alphaCoeffs, betaCoeffs, totalKVBlocks = newAlpha, newBeta, kvBlocks
		}
		if AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) {
			logrus.Warnf("Trying roofline approach for model=%v, TP=%v, GPU=%v, vllmVersion=%v\n", model, tensorParallelism, gpu, vllmVersion)
			if len(modelConfigFolder) > 0 && len(hwConfigPath) > 0 && len(gpu) > 0 && tensorParallelism > 0 {
				roofline = true
				hfPath := filepath.Join(modelConfigFolder, "config.json")
				mc, err := sim.GetModelConfig(hfPath)
				if err != nil {
					logrus.Fatalf("Failed to load model config: %v", err)
				}
				modelConfig = *mc
				hc, err := sim.GetHWConfig(hwConfigPath, gpu)
				if err != nil {
					logrus.Fatalf("Failed to load hardware config: %v", err)
				}
				hwConfig = hc
			} else if len(modelConfigFolder) == 0 {
				logrus.Fatalf("Please provide model config folder containing config.json for model=%v\n", model)
			} else if len(hwConfigPath) == 0 {
				logrus.Fatalf("Please provide hardware config path (e.g. hardware_config.json)\n")
			}
		}

		// Workload configuration
		var guideLLMConfig *sim.GuideLLMConfig
		var preGeneratedRequests []*sim.Request

		// --workload-spec takes precedence over --workload if set
		if workloadSpecPath != "" {
			spec, err := workload.LoadWorkloadSpec(workloadSpecPath)
			if err != nil {
				logrus.Fatalf("Failed to load workload spec: %v", err)
			}
			if err := spec.Validate(); err != nil {
				logrus.Fatalf("Invalid workload spec: %v", err)
			}
			// Apply spec horizon as default; CLI --horizon flag overrides via Changed().
			if spec.Horizon > 0 && !cmd.Flags().Changed("horizon") {
				simulationHorizon = spec.Horizon
			}
			reqs, err := workload.GenerateRequests(spec, simulationHorizon)
			if err != nil {
				logrus.Fatalf("Failed to generate workload: %v", err)
			}
			preGeneratedRequests = reqs
			// Set a placeholder GuideLLMConfig to satisfy constructor validation.
			// The actual requests come from preGeneratedRequests.
			guideLLMConfig = &sim.GuideLLMConfig{Rate: spec.AggregateRate / 1e6}
			logrus.Infof("Generated %d requests from workload spec", len(reqs))
		} else if workloadType == "distribution" { // if workloadType distribution, use args.
			// Validate rate (prevents infinite loop: int64(1/0) → MinInt64 → unbounded loop)
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
			}
			// error handling for prompt and output lengths
			if promptTokensMean > promptTokensMax || promptTokensMean < promptTokensMin || promptTokensStdev > promptTokensMax || promptTokensStdev < promptTokensMin {
				logrus.Fatalf("prompt-tokens and prompt-tokens-stdev should be in range [prompt-tokens-min, prompt-tokens-max]")
			}
			if outputTokensMean > outputTokensMax || outputTokensMean < outputTokensMin || outputTokensStdev > outputTokensMax || outputTokensStdev < outputTokensMin {
				logrus.Fatalf("output-tokens and output-tokens-stdev should be in range [output-tokens-min, output-tokens-max]")
			}
			guideLLMConfig = &sim.GuideLLMConfig{Rate: rate / 1e6, MaxPrompts: maxPrompts,
				PrefixTokens: prefixTokens, PromptTokens: promptTokensMean,
				PromptTokensStdDev: promptTokensStdev, PromptTokensMin: promptTokensMin, PromptTokensMax: promptTokensMax,
				OutputTokens: outputTokensMean, OutputTokensStdDev: outputTokensStdev,
				OutputTokensMin: outputTokensMin, OutputTokensMax: outputTokensMax}
		} else if workloadType != "traces" { // use default workload types
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
			}
			guideLLMConfig = GetWorkloadConfig(defaultsFilePath, workloadType, rate/1e6, maxPrompts)
			if guideLLMConfig == nil {
				logrus.Fatalf("Undefined workload. Use one among (chatbot, summarization, contentgen, multidoc)")
			}
		} else { // read from CSV
			guideLLMConfig = nil
		}

		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}

		if workloadType == "traces" && tracesWorkloadFilePath == "" {
			logrus.Fatalf("--workload-traces-filepath is required when using --workload traces")
		}

		// Load policy bundle if specified (BC-6: CLI flags override YAML values)
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
			if bundle.Routing.CacheWeight != nil && !cmd.Flags().Changed("routing-cache-weight") {
				routingCacheWeight = *bundle.Routing.CacheWeight
			}
			if bundle.Routing.LoadWeight != nil && !cmd.Flags().Changed("routing-load-weight") {
				routingLoadWeight = *bundle.Routing.LoadWeight
			}
			if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
				priorityPolicy = bundle.Priority.Policy
			}
			if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
				scheduler = bundle.Scheduler
			}
		}

		// Validate policy names (catches CLI typos before they become panics)
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

		// Log active policy configuration so users can verify which policies are in effect
		logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s",
			admissionPolicy, routingPolicy, priorityPolicy, scheduler)
		if routingPolicy == "weighted" {
			if math.IsNaN(routingCacheWeight) || math.IsInf(routingCacheWeight, 0) ||
				math.IsNaN(routingLoadWeight) || math.IsInf(routingLoadWeight, 0) {
				logrus.Fatalf("Routing weights must be finite numbers, got cache=%v, load=%v",
					routingCacheWeight, routingLoadWeight)
			}
			if routingCacheWeight < 0 || routingLoadWeight < 0 {
				logrus.Fatalf("Routing weights must be non-negative, got cache=%.2f, load=%.2f",
					routingCacheWeight, routingLoadWeight)
			}
			weightSum := routingCacheWeight + routingLoadWeight
			if weightSum <= 0 {
				logrus.Fatalf("Routing weights must have a positive sum, got cache=%.2f + load=%.2f = %.2f",
					routingCacheWeight, routingLoadWeight, weightSum)
			}
			if math.Abs(weightSum-1.0) > 0.01 {
				logrus.Warnf("Routing weights sum to %.2f (expected 1.0); weights will be auto-normalized to cache=%.2f, load=%.2f",
					weightSum, routingCacheWeight/weightSum, routingLoadWeight/weightSum)
			}
			logrus.Infof("Weighted routing: cache-weight=%.2f, load-weight=%.2f",
				routingCacheWeight, routingLoadWeight)
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
			NumInstances:              numInstances,
			Horizon:                   simulationHorizon,
			Seed:                      seed,
			TotalKVBlocks:             totalKVBlocks,
			BlockSizeTokens:           blockSizeTokens,
			MaxRunningReqs:            maxRunningReqs,
			MaxScheduledTokens:        maxScheduledTokens,
			LongPrefillTokenThreshold: longPrefillTokenThreshold,
			BetaCoeffs:                betaCoeffs,
			AlphaCoeffs:               alphaCoeffs,
			ModelConfig:               modelConfig,
			HWConfig:                  hwConfig,
			Model:                     model,
			GPU:                       gpu,
			TP:                        tensorParallelism,
			Roofline:                  roofline,
			AdmissionPolicy:           admissionPolicy,
			AdmissionLatency:          admissionLatency,
			RoutingLatency:            routingLatency,
			TokenBucketCapacity:       tokenBucketCapacity,
			TokenBucketRefillRate:     tokenBucketRefillRate,
			RoutingPolicy:             routingPolicy,
			RoutingCacheWeight:        routingCacheWeight,
			RoutingLoadWeight:         routingLoadWeight,
			PriorityPolicy:           priorityPolicy,
			Scheduler:                scheduler,
			TraceLevel:               traceLevel,
			CounterfactualK:          counterfactualK,
			KVCPUBlocks:             kvCPUBlocks,
			KVOffloadThreshold:      kvOffloadThreshold,
			KVTransferBandwidth:     kvTransferBandwidth,
		}
		cs := cluster.NewClusterSimulator(config, guideLLMConfig, tracesWorkloadFilePath)
		if len(preGeneratedRequests) > 0 {
			cs.SetPreGeneratedRequests(preGeneratedRequests)
		}
		if err := cs.Run(); err != nil {
			logrus.Fatalf("Simulation failed: %v", err)
		}

		if numInstances > 1 {
			// Print per-instance metrics to stdout (multi-instance only)
			for _, inst := range cs.Instances() {
				inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, startTime, "")
			}
		}
		// Save aggregated metrics (prints to stdout + saves to file if resultsPath set)
		cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, startTime, resultsPath)

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
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 {
			fmt.Println("=== Anomaly Counters ===")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests: %d\n", rawMetrics.RejectedRequests)
		}

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

	// GuideLLM-style distribution-based workload generation config
	runCmd.Flags().Float64Var(&rate, "rate", 1.0, "Requests arrival per second")
	runCmd.Flags().IntVar(&maxPrompts, "max-prompts", 100, "Number of requests")
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
	runCmd.Flags().Float64Var(&routingCacheWeight, "routing-cache-weight", 0.6, "Cache affinity weight for weighted routing")
	runCmd.Flags().Float64Var(&routingLoadWeight, "routing-load-weight", 0.4, "Load balance weight for weighted routing")

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

	// Results path
	runCmd.Flags().StringVar(&resultsPath, "results-path", "", "File to save BLIS results to")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
