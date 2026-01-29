package cmd

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
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
				modelConfig = *sim.GetModelConfig(hfPath)
				hwConfig = sim.GetHWConfig(hwConfigPath, gpu)
			} else if len(modelConfigFolder) == 0 {
				logrus.Fatalf("Please provide model config folder containing config.json for model=%v\n", model)
			} else if len(hwConfigPath) == 0 {
				logrus.Fatalf("Please provide hardware config path (e.g. hardware_config.json)\n")
			}
		}

		// Log configuration
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v",
			totalKVBlocks, simulationHorizon, alphaCoeffs, betaCoeffs)

		// Workload configuration
		var guideLLMConfig *sim.GuideLLMConfig

		if workloadType == "distribution" { // if workloadType distribution, use args.
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
			guideLLMConfig = GetWorkloadConfig(defaultsFilePath, workloadType, rate/1e6, maxPrompts)
			if guideLLMConfig == nil {
				logrus.Fatalf("Undefined workload. Use one among (chatbot, summarization, contentgen, multidoc)")
			}
		} else { // read from CSV
			guideLLMConfig = nil
		}

		startTime := time.Now() // Get current time (start)
		// Initialize and run the simulator
		s := sim.NewSimulator(
			simulationHorizon,
			seed,
			totalKVBlocks,
			blockSizeTokens,
			maxRunningReqs,
			maxScheduledTokens,
			longPrefillTokenThreshold,
			betaCoeffs,
			alphaCoeffs,
			guideLLMConfig,
			modelConfig,
			hwConfig,
			model,
			gpu,
			tensorParallelism,
			roofline,
			tracesWorkloadFilePath,
		)
		s.Run()

		// Print and save results
		s.Metrics.SaveResults(s.Horizon, totalKVBlocks, startTime, resultsPath)

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
	runCmd.Flags().StringVar(&logLevel, "log", "warn", "Log level (trace, debug, info, warn, error, fatal, panic)")
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

	// Results path
	runCmd.Flags().StringVar(&resultsPath, "results-path", "", "File to save BLIS results to")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
