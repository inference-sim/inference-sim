package cmd

import (
	"fmt"
	"math"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/inference-sim/sim/workload"
)

var convertCmd = &cobra.Command{
	Use:   "convert",
	Short: "Convert external workload formats to v2 YAML spec",
	Long:  "Convert external workload formats (ServeGen, inference-perf, presets) to v2 WorkloadSpec YAML. Output is written to stdout for piping.",
}

// --- blis convert servegen ---

var (
	serveGenPath              string
	serveGenWindowDurationSec int
	serveGenDrainTimeoutSec   int
	serveGenTimeFilter        string
)

var convertServeGenCmd = &cobra.Command{
	Use:   "servegen",
	Short: "Convert ServeGen data directory to v2 spec with multi-period cohorts",
	Run: func(cmd *cobra.Command, args []string) {
		// R3: validate numeric CLI flags
		if serveGenWindowDurationSec <= 0 {
			logrus.Fatalf("--window-duration-seconds must be > 0, got %d", serveGenWindowDurationSec)
		}
		if serveGenDrainTimeoutSec < 0 {
			logrus.Fatalf("--drain-timeout-seconds must be >= 0, got %d", serveGenDrainTimeoutSec)
		}

		// Validate --time flag if specified
		if serveGenTimeFilter != "" {
			validTimes := map[string]bool{"midnight": true, "morning": true, "afternoon": true}
			if !validTimes[serveGenTimeFilter] {
				logrus.Fatalf("--time must be one of: midnight, morning, afternoon (got %q)", serveGenTimeFilter)
			}
		}

		spec, err := workload.ConvertServeGen(serveGenPath, serveGenWindowDurationSec, serveGenDrainTimeoutSec)
		if err != nil {
			logrus.Fatalf("ServeGen conversion failed: %v", err)
		}

		// Filter cohorts by time period if --time is specified
		if serveGenTimeFilter != "" {
			spec = filterCohortsByPeriod(spec, serveGenTimeFilter)
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
		// R3: validate numeric CLI flags at the boundary
		if presetRate <= 0 || math.IsNaN(presetRate) || math.IsInf(presetRate, 0) {
			logrus.Fatalf("--rate must be a finite value > 0, got %v", presetRate)
		}
		if presetNumRequests <= 0 {
			logrus.Fatalf("--num-requests must be > 0, got %d", presetNumRequests)
		}
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

// filterCohortsByPeriod filters cohorts to only those matching the specified time period.
// Cohort IDs are expected to have format: "{period}-{slo_class}" (e.g., "midnight-critical").
func filterCohortsByPeriod(spec *workload.WorkloadSpec, period string) *workload.WorkloadSpec {
	// Copy the full spec to preserve all fields, then replace Cohorts
	filtered := &workload.WorkloadSpec{
		Version:       spec.Version,
		Seed:          spec.Seed,
		AggregateRate: spec.AggregateRate,
		Category:      spec.Category,
		Clients:       spec.Clients,
		Cohorts:       []workload.CohortSpec{}, // Will be populated below
		Horizon:       spec.Horizon,
		NumRequests:   spec.NumRequests,
		ServeGenData:  spec.ServeGenData,
		InferencePerf: spec.InferencePerf,
	}

	prefix := period + "-"
	for _, cohort := range spec.Cohorts {
		// Check if cohort ID starts with the period name
		if len(cohort.ID) >= len(prefix) && cohort.ID[:len(prefix)] == prefix {
			filtered.Cohorts = append(filtered.Cohorts, cohort)
		}
	}

	if len(filtered.Cohorts) == 0 {
		logrus.Fatalf("No cohorts found for period %q. Generated cohort IDs may not match expected format.", period)
	}

	logrus.Infof("Filtered to %d cohorts for period %q", len(filtered.Cohorts), period)
	return filtered
}

// writeSpecToStdout validates and marshals a WorkloadSpec to YAML on stdout.
func writeSpecToStdout(spec *workload.WorkloadSpec) {
	if err := spec.Validate(); err != nil {
		logrus.Fatalf("generated spec is invalid: %v", err)
	}
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
	convertServeGenCmd.Flags().IntVar(&serveGenWindowDurationSec, "window-duration-seconds", 600, "Duration of each time period in seconds")
	convertServeGenCmd.Flags().IntVar(&serveGenDrainTimeoutSec, "drain-timeout-seconds", 180, "Gap between periods where no new requests arrive")
	convertServeGenCmd.Flags().StringVar(&serveGenTimeFilter, "time", "", "Optional: filter to single period (midnight, morning, or afternoon)")
	_ = convertServeGenCmd.MarkFlagRequired("path")

	convertPresetCmd.Flags().StringVar(&presetName, "name", "", "Preset name (e.g., chatbot, summarization)")
	convertPresetCmd.Flags().Float64Var(&presetRate, "rate", 1.0, "Request rate in req/s")
	convertPresetCmd.Flags().IntVar(&presetNumRequests, "num-requests", 100, "Number of requests")
	convertPresetCmd.Flags().StringVar(&presetDefaultsPath, "defaults-filepath", "defaults.yaml", "Path to defaults.yaml")
	_ = convertPresetCmd.MarkFlagRequired("name")

	convertInfPerfCmd.Flags().StringVar(&infPerfSpecPath, "spec", "", "Path to inference-perf YAML spec")
	_ = convertInfPerfCmd.MarkFlagRequired("spec")

	convertCmd.AddCommand(convertServeGenCmd)
	convertCmd.AddCommand(convertPresetCmd)
	convertCmd.AddCommand(convertInfPerfCmd)

	rootCmd.AddCommand(convertCmd)
}
