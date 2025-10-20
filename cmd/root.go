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
	// CLI flags for simulation configuration
	totalKVBlocks             int       // Total number of KV blocks available on GPU
	simulationHorizon         int64     // Total simulation time (in ticks)
	rate                      float64   // Poisson arrival rate (requests per tick)
	logLevel                  string    // Log verbosity level
	maxRunningReqs            int64     // Maximum number of requests in the Running batch
	maxScheduledTokens        int       // Maximum total number of tokens across requests in the Running batch
	blockSizeTokens           int       // Number of tokens per KV block
	requestsFilePath          string    // Path to requests workload file path, default ShareGPT
	regressionCoeffs          []float64 // List of beta coeffs corresponding to features
	queuingCoeffs             []float64 // List of regression coeffs corresponding to features
	finishedCoeffs            []float64 // List of regression coeffs corresponding to features
	maxModelLength            int       // Max request length (input + output tokens) to be handled
	longPrefillTokenThreshold int       // Max length of prefill beyond which chunked prefill is triggered
	queuingDelay              int       // Delay between server hit and queued per request
	finishedDelay             int       // Delay between finished and server left per request
)

// rootCmd is the base command for the CLI
var rootCmd = &cobra.Command{
	Use:   "inference-sim",
	Short: "Discrete-event simulator for inference platforms",
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

		// Log configuration
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks, request rate=%.2f, regression coefficients=%v",
			totalKVBlocks, simulationHorizon, rate, regressionCoeffs)

		startTime := time.Now() // Get current time (start)

		requests := ProcessInputShareGPT(requestsFilePath)

		// Initialize and run the simulator
		s := sim.NewSimulator(
			simulationHorizon,
			totalKVBlocks,
			blockSizeTokens,
			maxRunningReqs,
			maxScheduledTokens,
			longPrefillTokenThreshold,
			queuingDelay,
			finishedDelay,
			regressionCoeffs,
			queuingCoeffs,
			finishedCoeffs,
			rate,
			requests,
		)
		s.GeneratePoissonArrivals(rate, simulationHorizon)
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
	runCmd.Flags().IntVar(&totalKVBlocks, "total-kv-blocks", 8000000, "Total number of KV cache blocks")
	runCmd.Flags().Int64Var(&simulationHorizon, "horizon", math.MaxInt64, "Total simulation horizon (in ticks)")
	runCmd.Flags().Float64Var(&rate, "rate", 0.02, "Poisson arrival rate (requests per tick)")
	runCmd.Flags().StringVar(&logLevel, "log", "error", "Log level (trace, debug, info, warn, error, fatal, panic)")
	runCmd.Flags().Int64Var(&maxRunningReqs, "max-num-running-reqs", 35, "Maximum number of requests running together")
	runCmd.Flags().IntVar(&maxScheduledTokens, "max-num-scheduled-tokens", 8192, "Maximum total number of new tokens across running requests")
	runCmd.Flags().Float64SliceVar(&regressionCoeffs, "regression-coeffs", []float64{1.0, 2.0}, "List of beta coefficients")
	runCmd.Flags().Float64SliceVar(&queuingCoeffs, "queuing-coeffs", []float64{1.0, 2.0}, "List of queuing coefficients")
	runCmd.Flags().Float64SliceVar(&finishedCoeffs, "finished-coeffs", []float64{1.0, 2.0}, "List of finished coefficients")
	runCmd.Flags().StringVar(&requestsFilePath, "requests-file-path", "ShareGPT_V3_tokenized.json", "Path to workload tokenized JSON file")
	runCmd.Flags().IntVar(&blockSizeTokens, "block-size-in-tokens", 16, "Number of tokens contained in a KV cache block")
	runCmd.Flags().IntVar(&maxModelLength, "max-model-len", 2048, "Max request length (input + output tokens)")
	runCmd.Flags().IntVar(&longPrefillTokenThreshold, "long-prefill-token-threshold", 0, "Max length of prefill beyond which chunked prefill is triggered")
	runCmd.Flags().IntVar(&queuingDelay, "queuing-delay", 0, "Delay between server hit and queued per request")
	runCmd.Flags().IntVar(&finishedDelay, "finished-delay", 0, "Delay between finished and server left per request")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
