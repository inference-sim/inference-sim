package cmd

import (
	"github.com/spf13/cobra"
)

var (
	traceHeaderPath string
	traceDataPath   string
)

var replayCmd = &cobra.Command{
	Use:   "replay",
	Short: "Replay a TraceV2 file through the discrete-event simulator",
	Run: func(cmd *cobra.Command, args []string) {
		// TODO: implement in Task 4-5
	},
}

func init() {
	registerSimConfigFlags(replayCmd)
	replayCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (required)")
	replayCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (required)")
	rootCmd.AddCommand(replayCmd)
}
