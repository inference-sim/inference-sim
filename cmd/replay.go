package cmd

import (
	"math"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/workload"
)

var (
	traceHeaderPath string
	traceDataPath   string
)

var replayCmd = &cobra.Command{
	Use:   "replay",
	Short: "Replay a TraceV2 file through the discrete-event simulator",
	Run: func(cmd *cobra.Command, args []string) {
		level, err := logrus.ParseLevel(logLevel)
		if err != nil {
			logrus.Fatalf("Invalid log level: %s", logLevel)
		}
		logrus.SetLevel(level)

		// Validate required inputs (BC-6, BC-8)
		if traceHeaderPath == "" {
			logrus.Fatalf("--trace-header is required")
		}
		if traceDataPath == "" {
			logrus.Fatalf("--trace-data is required")
		}
		if _, statErr := os.Stat(traceHeaderPath); os.IsNotExist(statErr) {
			logrus.Fatalf("--trace-header file not found: %s", traceHeaderPath)
		}
		if _, statErr := os.Stat(traceDataPath); os.IsNotExist(statErr) {
			logrus.Fatalf("--trace-data file not found: %s", traceDataPath)
		}
		if model == "" {
			logrus.Fatalf("LLM name not provided. Exiting simulation.")
		}

		// Load trace (BC-1)
		traceData, err := workload.LoadTraceV2(traceHeaderPath, traceDataPath)
		if err != nil {
			logrus.Fatalf("Failed to load trace: %v", err)
		}
		logrus.Infof("Loaded trace: %d records (mode=%s)", len(traceData.Records), traceData.Header.Mode)

		// Build requests from trace (BC-1)
		requests, err := workload.LoadTraceV2Requests(traceData, seed)
		if err != nil {
			logrus.Fatalf("Failed to build requests from trace: %v", err)
		}
		logrus.Infof("Built %d requests for replay", len(requests))

		// Compute horizon (BC-3)
		replayHorizon := computeReplayHorizon(requests)
		if cmd.Flags().Changed("horizon") {
			replayHorizon = simulationHorizon
		}
		logrus.Infof("Simulation horizon: %d ticks", replayHorizon)

		// [Config resolution and cluster run — Task 5]
		_ = replayHorizon // used in Task 5
	},
}

func init() {
	registerSimConfigFlags(replayCmd)
	replayCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (required)")
	replayCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (required)")
	rootCmd.AddCommand(replayCmd)
}

// computeReplayHorizon returns the simulation horizon for a trace replay.
// - Empty slice → math.MaxInt64 (no requests, horizon doesn't matter)
// - maxArrival > MaxInt64/2 → math.MaxInt64 (overflow guard for 2×)
// - maxArrival <= 0 (all at t=0) → 600,000,000 µs (10 min buffer; MaxInt64 would hang)
// - Otherwise → maxArrival * 2 (generous buffer for last request to complete)
func computeReplayHorizon(requests []*sim.Request) int64 {
	if len(requests) == 0 {
		return math.MaxInt64
	}
	var maxArrival int64
	for _, req := range requests {
		if req.ArrivalTime > maxArrival {
			maxArrival = req.ArrivalTime
		}
	}
	// Overflow guard: if 2× would overflow int64, use MaxInt64 directly.
	if maxArrival > math.MaxInt64/2 {
		return math.MaxInt64
	}
	if maxArrival <= 0 {
		// All requests at t=0: use a fixed generous buffer of 10 minutes (600,000,000 µs)
		// rather than MaxInt64 (which would cause the simulation to run indefinitely).
		return 600_000_000
	}
	return maxArrival * 2
}

// extractSimResults converts Metrics to a slice of workload.SimResult for calibrate consumption.
// Only requests with both TTFT and E2E recorded (i.e., fully completed) are included.
// Non-numeric IDs (session follow-ups, format "request_<parent>_followup_<n>") are excluded.
// Results are sorted by RequestID for deterministic output (R2, INV-6).
// Returns an initialized empty slice (not nil) so JSON marshaling produces [] not null.
// Exclusions are logged at Debug level for observability (R1: no silent data loss).
func extractSimResults(m *sim.Metrics) []workload.SimResult {
	results := make([]workload.SimResult, 0, len(m.RequestTTFTs))
	var noE2ECount, noReqCount, nonNumericCount int
	for reqID, ttftUs := range m.RequestTTFTs {
		e2eUs, hasE2E := m.RequestE2Es[reqID]
		if !hasE2E {
			noE2ECount++ // timed out after prefill
			continue
		}
		rm, hasReq := m.Requests[reqID]
		if !hasReq {
			noReqCount++ // metrics inconsistency (defensive)
			continue
		}
		// Parse integer RequestID from "request_N" format (BC-7: skip non-numeric IDs)
		numStr := strings.TrimPrefix(reqID, "request_")
		id, err := strconv.Atoi(numStr)
		if err != nil {
			nonNumericCount++ // session follow-ups or other non-numeric IDs
			continue
		}
		results = append(results, workload.SimResult{
			RequestID:    id,
			TTFT:         ttftUs,
			E2E:          e2eUs,
			InputTokens:  rm.NumPrefillTokens,
			OutputTokens: rm.NumDecodeTokens,
		})
	}
	// Log all exclusions at Debug level for observability (R1: no silent data loss)
	if noE2ECount > 0 {
		logrus.Debugf("extractSimResults: excluded %d request(s) with TTFT but no E2E (timed out after prefill)", noE2ECount)
	}
	if noReqCount > 0 {
		logrus.Debugf("extractSimResults: excluded %d request(s) in TTFTs but missing from Requests (metrics inconsistency)", noReqCount)
	}
	if nonNumericCount > 0 {
		logrus.Debugf("extractSimResults: excluded %d non-numeric-ID request(s) (session follow-ups)", nonNumericCount)
	}
	// Sort by RequestID for deterministic JSON output (R2, INV-6)
	sort.Slice(results, func(i, j int) bool {
		return results[i].RequestID < results[j].RequestID
	})
	return results
}
