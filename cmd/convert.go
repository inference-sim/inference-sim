package cmd

import (
	"fmt"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim/workload"
)

var convertCmd = &cobra.Command{
	Use:   "convert",
	Short: "Convert external workload formats to v2 YAML spec",
	Long:  "Convert external workload formats (ServeGen, inference-perf, CSV traces, presets) to v2 WorkloadSpec YAML. Output is written to stdout for piping.",
}

// --- blis convert servegen ---

var serveGenPath string

var convertServeGenCmd = &cobra.Command{
	Use:   "servegen",
	Short: "Convert ServeGen data directory to v2 spec",
	Run: func(cmd *cobra.Command, args []string) {
		spec, err := workload.ConvertServeGen(serveGenPath)
		if err != nil {
			logrus.Fatalf("ServeGen conversion failed: %v", err)
		}
		writeSpecToStdout(spec)
	},
}

// --- blis convert csv-trace ---

var (
	csvTracePath    string
	csvTraceHorizon int64
)

var convertCSVTraceCmd = &cobra.Command{
	Use:   "csv-trace",
	Short: "Convert legacy CSV trace file to v2 spec",
	Run: func(cmd *cobra.Command, args []string) {
		spec, err := workload.ConvertCSVTrace(csvTracePath, csvTraceHorizon)
		if err != nil {
			logrus.Fatalf("CSV trace conversion failed: %v", err)
		}
		writeSpecToStdout(spec)
	},
}

// --- blis convert preset ---

var (
	presetName         string
	presetRate         float64
	presetNumRequests  int
	presetDefaultsPath string
)

var convertPresetCmd = &cobra.Command{
	Use:   "preset",
	Short: "Convert a named workload preset to v2 spec",
	Run: func(cmd *cobra.Command, args []string) {
		wl := loadPresetWorkload(presetDefaultsPath, presetName)
		if wl == nil {
			logrus.Fatalf("Unknown preset %q. Check defaults.yaml for available workloads.", presetName)
		}
		preset := workload.PresetConfig{
			PrefixTokens:      wl.PrefixTokens,
			PromptTokensMean:  wl.PromptTokensMean,
			PromptTokensStdev: wl.PromptTokensStdev,
			PromptTokensMin:   wl.PromptTokensMin,
			PromptTokensMax:   wl.PromptTokensMax,
			OutputTokensMean:  wl.OutputTokensMean,
			OutputTokensStdev: wl.OutputTokensStdev,
			OutputTokensMin:   wl.OutputTokensMin,
			OutputTokensMax:   wl.OutputTokensMax,
		}
		spec, err := workload.ConvertPreset(presetName, presetRate, presetNumRequests, preset)
		if err != nil {
			logrus.Fatalf("Preset conversion failed: %v", err)
		}
		writeSpecToStdout(spec)
	},
}

// --- blis convert inference-perf ---

var infPerfSpecPath string

var convertInfPerfCmd = &cobra.Command{
	Use:   "inference-perf",
	Short: "Convert inference-perf YAML spec to v2 spec",
	Run: func(cmd *cobra.Command, args []string) {
		spec, err := workload.ConvertInferencePerf(infPerfSpecPath)
		if err != nil {
			logrus.Fatalf("inference-perf conversion failed: %v", err)
		}
		writeSpecToStdout(spec)
	},
}

// writeSpecToStdout marshals a WorkloadSpec to YAML and writes to stdout.
func writeSpecToStdout(spec *workload.WorkloadSpec) {
	data, err := yaml.Marshal(spec)
	if err != nil {
		logrus.Fatalf("YAML marshal failed: %v", err)
	}
	fmt.Print(string(data))
}

// loadPresetWorkload loads a named preset from defaults.yaml.
// Returns nil if the preset is not found.
func loadPresetWorkload(defaultsPath, name string) *Workload {
	cfg := loadDefaultsConfig(defaultsPath)
	if wl, ok := cfg.Workloads[name]; ok {
		return &wl
	}
	return nil
}

func init() {
	convertServeGenCmd.Flags().StringVar(&serveGenPath, "path", "", "Path to ServeGen data directory")
	_ = convertServeGenCmd.MarkFlagRequired("path")

	convertCSVTraceCmd.Flags().StringVar(&csvTracePath, "file", "", "Path to CSV trace file")
	convertCSVTraceCmd.Flags().Int64Var(&csvTraceHorizon, "horizon", 0, "Horizon in microseconds (0 = no truncation)")
	_ = convertCSVTraceCmd.MarkFlagRequired("file")

	convertPresetCmd.Flags().StringVar(&presetName, "name", "", "Preset name (e.g., chatbot, summarization)")
	convertPresetCmd.Flags().Float64Var(&presetRate, "rate", 1.0, "Request rate in req/s")
	convertPresetCmd.Flags().IntVar(&presetNumRequests, "num-requests", 100, "Number of requests")
	convertPresetCmd.Flags().StringVar(&presetDefaultsPath, "defaults-filepath", "defaults.yaml", "Path to defaults.yaml")
	_ = convertPresetCmd.MarkFlagRequired("name")

	convertInfPerfCmd.Flags().StringVar(&infPerfSpecPath, "spec", "", "Path to inference-perf YAML spec")
	_ = convertInfPerfCmd.MarkFlagRequired("spec")

	convertCmd.AddCommand(convertServeGenCmd)
	convertCmd.AddCommand(convertCSVTraceCmd)
	convertCmd.AddCommand(convertPresetCmd)
	convertCmd.AddCommand(convertInfPerfCmd)

	rootCmd.AddCommand(convertCmd)
}
