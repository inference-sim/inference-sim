package cmd

import (
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
		// TODO: implement in Task 4-5
	},
}

func init() {
	registerSimConfigFlags(replayCmd)
	replayCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (required)")
	replayCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (required)")
	rootCmd.AddCommand(replayCmd)
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
