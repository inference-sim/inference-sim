package cmd

import (
	"encoding/json"
	"os"

	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
)

var (
	calibrateTraceHeaderPath      string
	calibrateTraceDataPath        string
	calibrateSimResultsPath       string
	calibrateReportPath           string
	calibrateWarmUpRequests       int
	calibrateNetworkRTTUs         int64
	calibrateNetworkBandwidthMbps float64
)

var calibrateCmd = &cobra.Command{
	Use:   "calibrate",
	Short: "Compare real observed latencies against simulator predictions",
	Long: `Calibrate takes a TraceV2 file (from blis observe) and a SimResult JSON file
(from blis replay --results-path) and computes a calibration report comparing
real vs simulated TTFT and E2E latencies.

The report includes per-metric MAPE, Pearson r, percentile comparison, bias
direction, and a quality rating. Use --report to specify the output path.

Warm-up requests are excluded from comparison. By default, the warm-up count
is taken from the trace header (warm_up_requests field). Use --warmup-requests
to override. Pass --warmup-requests 0 to include all requests.

Network RTT and bandwidth adjustments shift sim-side latencies to client
perspective. By default, RTT is taken from the trace header
(network.measured_rtt_ms). Use --network-rtt-us to override in microseconds.

Example:
  blis calibrate --trace-header t.yaml --trace-data d.csv \
    --sim-results results.json --report calibration.json`,
	Run: func(cmd *cobra.Command, args []string) {
		if calibrateTraceHeaderPath == "" {
			logrus.Fatalf("--trace-header is required")
		}
		if calibrateTraceDataPath == "" {
			logrus.Fatalf("--trace-data is required")
		}
		if calibrateSimResultsPath == "" {
			logrus.Fatalf("--sim-results is required")
		}
		if calibrateReportPath == "" {
			logrus.Fatalf("--report is required")
		}

		// Step 1: Load TraceV2 (header + CSV data)
		trace, err := workload.LoadTraceV2(calibrateTraceHeaderPath, calibrateTraceDataPath)
		if err != nil {
			logrus.Fatalf("Failed to load TraceV2: %v", err)
		}

		// Step 2: Load SimResult JSON
		simData, err := os.ReadFile(calibrateSimResultsPath)
		if err != nil {
			logrus.Fatalf("Failed to read sim results from %s: %v", calibrateSimResultsPath, err)
		}
		var simResults []workload.SimResult
		if err := json.Unmarshal(simData, &simResults); err != nil {
			logrus.Fatalf("Failed to parse sim results JSON from %s: %v", calibrateSimResultsPath, err)
		}
		if len(simResults) == 0 {
			logrus.Fatalf("No sim results found in %s — cannot calibrate with empty data", calibrateSimResultsPath)
		}

		// Step 3: Resolve warm-up count (sentinel -1 → header fallback)
		warmUp := calibrateWarmUpRequests
		if warmUp == -1 {
			warmUp = trace.Header.WarmUpRequests
		}

		// Step 4: Resolve network RTT (sentinel -1 → header fallback)
		// Reject explicit negative values (not the sentinel) — R3, BC-11
		if calibrateNetworkRTTUs != -1 && calibrateNetworkRTTUs < 0 {
			logrus.Fatalf("--network-rtt-us must be >= 0 (or omit to use trace header), got %d", calibrateNetworkRTTUs)
		}
		var networkRTTUs int64
		if calibrateNetworkRTTUs == -1 {
			if trace.Header.Network != nil && trace.Header.Network.MeasuredRTTMs > 0 {
				networkRTTUs = int64(trace.Header.Network.MeasuredRTTMs * 1000)
			}
		} else {
			networkRTTUs = calibrateNetworkRTTUs
		}

		config := workload.CalibrationConfig{
			WarmUpRequests: warmUp,
			NetworkRTTUs:   networkRTTUs,
			BandwidthMbps:  calibrateNetworkBandwidthMbps,
		}

		// Step 5: Prepare calibration pairs
		pairs, err := workload.PrepareCalibrationPairs(trace.Records, simResults, &config)
		if err != nil {
			logrus.Fatalf("Failed to prepare calibration pairs: %v", err)
		}
		// Guard against zero matched pairs (R1: no silent data loss, BC-10)
		if pairs.MatchedCount == 0 {
			logrus.Fatalf("No matching request IDs found between trace and sim results — check that both files use the same request ID numbering")
		}

		// Step 6: Build report (empty ConfigMatchInfo — deferred, see TODO)
		// TODO: populate ConfigMatchInfo by comparing trace.Header.Server against sim config (#658)
		configMatch := workload.ConfigMatchInfo{}
		report, err := workload.BuildCalibrationReport(pairs, &configMatch)
		if err != nil {
			logrus.Fatalf("Failed to build calibration report: %v", err)
		}

		// Step 7: Write report JSON
		reportData, err := json.MarshalIndent(report, "", "  ")
		if err != nil {
			logrus.Fatalf("Failed to marshal calibration report: %v", err)
		}
		if err := os.WriteFile(calibrateReportPath, reportData, 0644); err != nil {
			logrus.Fatalf("Failed to write calibration report to %s: %v", calibrateReportPath, err)
		}

		// Step 8: Log summary to stderr
		logrus.Infof("Calibration report written to %s", calibrateReportPath)
		logrus.Infof("  Matched pairs: %d (warm-up excluded: %d, unmatched real: %d, unmatched sim: %d)",
			pairs.MatchedCount, pairs.ExcludedWarmUp, pairs.UnmatchedReal, pairs.UnmatchedSim)
		if ttft, ok := report.Metrics["ttft"]; ok {
			logrus.Infof("  TTFT: MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
				ttft.MAPE*100, ttft.PearsonR, ttft.Quality)
		}
		if e2e, ok := report.Metrics["e2e"]; ok {
			logrus.Infof("  E2E:  MAPE=%.1f%%, PearsonR=%.3f, quality=%s",
				e2e.MAPE*100, e2e.PearsonR, e2e.Quality)
		}
	},
}

func init() {
	calibrateCmd.Flags().StringVar(&calibrateTraceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateTraceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateSimResultsPath, "sim-results", "", "Path to SimResult JSON file (from blis replay --results-path; required)")
	calibrateCmd.Flags().StringVar(&calibrateReportPath, "report", "", "Path to write calibration report JSON (required)")
	calibrateCmd.Flags().IntVar(&calibrateWarmUpRequests, "warmup-requests", -1, "Number of initial requests to exclude (default: from trace header warm_up_requests; pass 0 to include all)")
	calibrateCmd.Flags().Int64Var(&calibrateNetworkRTTUs, "network-rtt-us", -1, "Network RTT in microseconds added to sim-side latencies (default: from trace header network.measured_rtt_ms)")
	calibrateCmd.Flags().Float64Var(&calibrateNetworkBandwidthMbps, "network-bandwidth-mbps", 0, "Network bandwidth in Mbps for upload/download delay calculation (default: 0 = no delay)")
	rootCmd.AddCommand(calibrateCmd)
}
