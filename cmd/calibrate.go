package cmd

import (
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
		// TODO: implement in Task 2
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
