package cmd

import (
	"math"
	"os"
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
	gammaCoeffs               float64   // Gamma coeff corresponding to model generalization
	fToken                    int64     // FLOPS per token specific to the LLM
	coeffsFilePath            string    // Path to trained coefficients filepath for testing/inference
	workloadFilePath          string    // Path to GuideLLM preset workload definitions filepath
	workloadType              string    // GuideLLM preset workload type (chatbot, summarization, contentgen, multidoc)
	maxModelLength            int       // Max request length (input + output tokens) to be handled
	longPrefillTokenThreshold int64     // Max length of prefill beyond which chunked prefill is triggered
	rate                      float64   // Requests arrival per second
	maxPrompts                int       // Number of requests
	promptTokensMean          int       // Average Prompt Token Count
	promptTokensStdev         int       // Stdev Prompt Token Count
	promptTokensMin           int       // Min Prompt Token Count
	promptTokensMax           int       // Max Prompt Token Count
	outputTokensMean          int       // Average Output Token Count
	outputTokensStdev         int       // Stdev Output Token Count
	outputTokensMin           int       // Min Output Token Count
	outputTokensMax           int       // Max Output Token Count

	// CLI flags for model, GPU, TP, vllm version
	model             string // LLM name
	gpu               string // GPU type
	tensorParallelism int    // TP value
	vllmVersion       string // vllm version
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

		if AllZeros(alphaCoeffs) && AllZeros(betaCoeffs) && gammaCoeffs == 0 { // default all 0s
			// GPU, TP, vLLM version configuration
			hardware, tp, version := GetDefaultConfig(model) // pick default config for tp, GPU, vllmVersion

			// if any of (hardware, tp, vllm-version args missing, fall back to default for all)
			if (tensorParallelism == 0 && tp > 0) || (gpu == "" && len(hardware) > 0) || (vllmVersion == "" && len(version) > 0) {
				logrus.Warnf("All of (GPU, TP, vLLM version) args should be provided, otherwise provide only model name. Using default tp=%v, GPU=%v, vllmVersion=%v", tp, hardware, version)
				tensorParallelism = tp
				gpu = hardware
				vllmVersion = version
			}
			newAlpha, newBeta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, coeffsFilePath)
			alphaCoeffs, betaCoeffs, totalKVBlocks = newAlpha, newBeta, kvBlocks
		}
		if len(alphaCoeffs) == 0 || len(betaCoeffs) == 0 {
			logrus.Fatalf("Could not find coefficients for model=%v, TP=%v, GPU=%v, vllmVersion=%v\n", model, tensorParallelism, gpu, vllmVersion)
		}

		// Log configuration
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v, gammaCoeffs=%v",
			totalKVBlocks, simulationHorizon, alphaCoeffs, betaCoeffs, gammaCoeffs)

		// Workload configuration
		var guideLLMConfig *sim.GuideLLMConfig

		if workloadType == "custom" { // if workloadType custom, use args. If no args, use defaults
			// error handling for prompt and output lengths
			if promptTokensMean > promptTokensMax || promptTokensMean < promptTokensMin || promptTokensStdev > promptTokensMax || promptTokensStdev < promptTokensMin {
				logrus.Fatalf("prompt-tokens and prompt-tokens-stdev should be in range [prompt-tokens-min, prompt-tokens-max]")
			}
			if outputTokensMean > outputTokensMax || outputTokensMean < outputTokensMin || outputTokensStdev > outputTokensMax || outputTokensStdev < outputTokensMin {
				logrus.Fatalf("output-tokens and output-tokens-stdev should be in range [output-tokens-min, output-tokens-max]")
			}
			guideLLMConfig = &sim.GuideLLMConfig{Rate: rate / 1e6, MaxPrompts: maxPrompts, PromptTokens: promptTokensMean,
				PromptTokensStdDev: promptTokensStdev, PromptTokensMin: promptTokensMin, PromptTokensMax: promptTokensMax,
				OutputTokens: outputTokensMean, OutputTokensStdDev: outputTokensStdev,
				OutputTokensMin: outputTokensMin, OutputTokensMax: outputTokensMax}
		} else { // else use preset workload types
			guideLLMConfig = GetWorkloadConfig(workloadFilePath, workloadType, rate/1e6, maxPrompts)
			if guideLLMConfig == nil {
				logrus.Fatalf("Undefined workload. Use one among (chatbot, summarization, contentgen, multidoc)")
			}
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
			gammaCoeffs,
			fToken,
			guideLLMConfig,
		)
		s.Run()
		s.Metrics.Print(s.Horizon, totalKVBlocks, startTime)

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
	runCmd.Flags().StringVar(&coeffsFilePath, "coeffs-filepath", "coefficients.yaml", "Path to trained coefficients filepath for testing/inference")
	runCmd.Flags().StringVar(&workloadFilePath, "workloads-filepath", "workloads.yaml", "Path to GuideLLM preset workload definitions filepath")
	runCmd.Flags().StringVar(&workloadType, "workload", "custom", "GuideLLM preset workload type (chatbot, summarization, contentgen, multidoc)")

	// vLLM server configs
	runCmd.Flags().Int64Var(&totalKVBlocks, "total-kv-blocks", 0, "Total number of KV cache blocks")
	runCmd.Flags().Int64Var(&maxRunningReqs, "max-num-running-reqs", 256, "Maximum number of requests running together")
	runCmd.Flags().Int64Var(&maxScheduledTokens, "max-num-scheduled-tokens", 2048, "Maximum total number of new tokens across running requests")
	runCmd.Flags().Float64SliceVar(&betaCoeffs, "beta-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated list of beta coefficients")
	runCmd.Flags().Float64SliceVar(&alphaCoeffs, "alpha-coeffs", []float64{0.0, 0.0, 0.0}, "Comma-separated alpha coefficients (alpha0,alpha1,alpha2) for processing delays")
	runCmd.Flags().Float64Var(&gammaCoeffs, "gamma-coeffs", 0.0, "Gamma coefficient for model generalization")
	runCmd.Flags().Int64Var(&fToken, "flop-per-token", 1000000, "FLOPS per token specific to the LLM")
	runCmd.Flags().Int64Var(&blockSizeTokens, "block-size-in-tokens", 16, "Number of tokens contained in a KV cache block")
	runCmd.Flags().IntVar(&maxModelLength, "max-model-len", 2048, "Max request length (input + output tokens)")
	runCmd.Flags().Int64Var(&longPrefillTokenThreshold, "long-prefill-token-threshold", 0, "Max length of prefill beyond which chunked prefill is triggered")

	// BLIS model configs
	runCmd.Flags().StringVar(&model, "model", "", "LLM name")
	runCmd.Flags().StringVar(&gpu, "hardware", "", "GPU type")
	runCmd.Flags().IntVar(&tensorParallelism, "tp", 0, "Tensor parallelism")
	runCmd.Flags().StringVar(&vllmVersion, "vllm-version", "", "vLLM version")

	// GuideLLM request generation config
	runCmd.Flags().Float64Var(&rate, "rate", 1.0, "Requests arrival per second")
	runCmd.Flags().IntVar(&maxPrompts, "max-prompts", 100, "Number of requests")
	runCmd.Flags().IntVar(&promptTokensMean, "prompt-tokens", 512, "Average Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensStdev, "prompt-tokens-stdev", 256, "Stddev Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMin, "prompt-tokens-min", 2, "Min Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMax, "prompt-tokens-max", 7000, "Max Prompt Token Count")
	runCmd.Flags().IntVar(&outputTokensMean, "output-tokens", 512, "Average Output Token Count")
	runCmd.Flags().IntVar(&outputTokensStdev, "output-tokens-stdev", 256, "Stddev Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMin, "output-tokens-min", 2, "Min Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMax, "output-tokens-max", 7000, "Max Output Token Count")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
