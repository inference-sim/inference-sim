// cmd/root.go
package cmd

import (
	"os"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
)

var (
	totalKVBlocks     int
	simulationHorizon int64
	rate              float64
	logLevel          string
	seed              int64
	stepDuration      int64
	maxBatchSize      int64
	maxGPUAllocation  int64
	blockSize         int
)

var rootCmd = &cobra.Command{
	Use:   "inference-sim",
	Short: "Discrete-event simulator for inference platforms",
}

var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Run the vLLM simulation",
	Run: func(cmd *cobra.Command, args []string) {
		level, err := logrus.ParseLevel(logLevel)
		if err != nil {
			logrus.Fatalf("Invalid log level: %s", logLevel)
		}
		logrus.SetLevel(level)
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dµs, rate=%.2f, step=%dµs",
			totalKVBlocks, simulationHorizon, rate, stepDuration)

		s := sim.NewSimulator(totalKVBlocks, simulationHorizon, stepDuration, maxBatchSize, maxGPUAllocation, blockSize)
		s.GeneratePoissonArrivals(rate, simulationHorizon, seed)
		s.Run()
		s.Metrics.Print()
		logrus.Info("Simulation complete.")
	},
}

func Execute() {
	if err := rootCmd.Execute(); err != nil {
		os.Exit(1)
	}
}

func init() {
	runCmd.Flags().IntVar(&totalKVBlocks, "kv", 64, "Total number of KV cache blocks")
	runCmd.Flags().Int64Var(&simulationHorizon, "horizon", 1000000, "Total simulation horizon in microseconds")
	runCmd.Flags().Float64Var(&rate, "rate", 0.4, "Poisson arrival rate (requests per µs)")
	runCmd.Flags().StringVar(&logLevel, "log", "info", "Log level (debug, info, warn, error)")
	runCmd.Flags().Int64Var(&stepDuration, "step", 1000, "Forward pass step duration in microseconds")
	runCmd.Flags().Int64Var(&maxBatchSize, "max-batch", 8, "Maximum batch size")
	runCmd.Flags().Int64Var(&maxGPUAllocation, "max-gpu", 64, "Maximum GPU KV block allocation")
	runCmd.Flags().IntVar(&blockSize, "block size", 16, "Number of tokens contained in a KV cache block")

	rootCmd.AddCommand(runCmd)
}
