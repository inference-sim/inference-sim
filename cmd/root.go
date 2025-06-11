package cmd

import (
	"os"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
)

var (
	// CLI flags for simulation configuration
	totalKVBlocks     int     // Total number of KV blocks available on GPU
	simulationHorizon int64   // Total simulation time (in ticks)
	rate              float64 // Poisson arrival rate (requests per tick)
	logLevel          string  // Log verbosity level
	seed              int64   // Random seed for reproducibility
	stepDuration      int64   // Duration of each forward pass step (in ticks)
	maxBatchSize      int64   // Maximum number of requests per batch
	maxGPUAllocation  int64   // Max number of KV blocks usable in one batch
	blockSizeInTokens int     // Number of tokens per KV block
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
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks, rate=%.2f, step=%dticks",
			totalKVBlocks, simulationHorizon, rate, stepDuration)

		// Initialize and run the simulator
		s := sim.NewSimulator(
			simulationHorizon,
			stepDuration,
			totalKVBlocks,
			blockSizeInTokens,
			maxBatchSize,
			maxGPUAllocation,
		)
		s.GeneratePoissonArrivals(rate, simulationHorizon, seed)
		s.Run()
		s.Metrics.Print(stepDuration)

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
	runCmd.Flags().IntVar(&totalKVBlocks, "kv", 8000000, "Total number of KV cache blocks")
	runCmd.Flags().Int64Var(&simulationHorizon, "horizon", 10000, "Total simulation horizon (in ticks)")
	runCmd.Flags().Float64Var(&rate, "rate", 0.4, "Poisson arrival rate (requests per tick)")
	runCmd.Flags().StringVar(&logLevel, "log", "info", "Log level (trace, debug, info, warn, error, fatal, panic)")
	runCmd.Flags().Int64Var(&stepDuration, "step", 100, "Forward pass step duration (in ticks)")
	runCmd.Flags().Int64Var(&maxBatchSize, "max-batch", 8, "Maximum batch size")
	runCmd.Flags().Int64Var(&maxGPUAllocation, "max-gpu", 6000000, "Maximum GPU KV block allocation")
	runCmd.Flags().IntVar(&blockSizeInTokens, "block size in tokens", 16, "Number of tokens contained in a KV cache block")

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}
