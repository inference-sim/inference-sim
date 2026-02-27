package cmd

import (
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/inference-sim/sim/workload"
)

var composeFromPaths []string

var composeCmd = &cobra.Command{
	Use:   "compose",
	Short: "Merge multiple v2 workload specs into one",
	Long:  "Load multiple v2 WorkloadSpec YAML files and merge their client lists. Output is written to stdout.",
	Run: func(cmd *cobra.Command, args []string) {
		if len(composeFromPaths) == 0 {
			logrus.Fatalf("at least one --from flag is required")
		}

		var specs []*workload.WorkloadSpec
		for _, path := range composeFromPaths {
			spec, err := workload.LoadWorkloadSpec(path)
			if err != nil {
				logrus.Fatalf("Failed to load spec %s: %v", path, err)
			}
			specs = append(specs, spec)
		}

		merged, err := workload.ComposeSpecs(specs)
		if err != nil {
			logrus.Fatalf("Compose failed: %v", err)
		}
		writeSpecToStdout(merged)
	},
}

func init() {
	composeCmd.Flags().StringArrayVar(&composeFromPaths, "from", nil, "Path to v2 WorkloadSpec YAML file (can be repeated)")
	_ = composeCmd.MarkFlagRequired("from")

	rootCmd.AddCommand(composeCmd)
}
