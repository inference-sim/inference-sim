package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

var (
	traceHeaderPath string
	traceDataPath   string
)

var replayCmd = &cobra.Command{
	Use:   "replay",
	Short: "Replay a TraceV2 file through the discrete-event simulator",
	Long: `Replay takes a TraceV2 file (header YAML + data CSV) and runs the DES against the
exact request sequence captured in the trace. Unlike 'blis run', it does not generate
requests from distributions — the request sequence is fully determined by the trace.

Use --results-path to write per-request SimResult JSON (request_id, ttft_us, e2e_us,
input_tokens, output_tokens) for downstream consumption by blis calibrate.

Known limitations:
  - Warm-up requests: trace.Header.warm_up_requests is not filtered; blis calibrate
    is responsible for excluding the first N warm-up entries from calibration.
  - Multi-model traces: per-request Model field is propagated to the simulator, but
    the latency model configuration (--model flag) applies globally to all requests.
  - Horizon: --horizon defaults to 2x the latest arrival time. For heavy-load traces
    where requests queue past 2x max_arrival, pass --horizon explicitly and monitor
    still_queued/still_running in the aggregate metrics output.

Example:
  blis replay --trace-header t.yaml --trace-data d.csv --model qwen/qwen3-14b`,
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

		// Resolve latency backend configuration (single code path shared with runCmd).
		lr := resolveLatencyConfig(cmd)

		// Numeric flag validation (same as runCmd)
		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}
		if totalKVBlocks <= 0 {
			logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
		}
		if maxRunningReqs <= 0 {
			logrus.Fatalf("--max-num-running-reqs must be > 0, got %d", maxRunningReqs)
		}
		if maxScheduledTokens <= 0 {
			logrus.Fatalf("--max-num-scheduled-tokens must be > 0, got %d", maxScheduledTokens)
		}
		if longPrefillTokenThreshold < 0 {
			logrus.Fatalf("--long-prefill-token-threshold must be >= 0, got %d", longPrefillTokenThreshold)
		}
		if cmd.Flags().Changed("horizon") && replayHorizon <= 0 {
			logrus.Fatalf("--horizon must be > 0, got %d", replayHorizon)
		}

		// Warn on PD-disaggregation flags that replay does not support.
		// These flags are registered via registerSimConfigFlags (shared with runCmd) but
		// replay does not build a PD-disaggregated ClusterSimulator.
		if pdTransferContention {
			logrus.Warnf("[replay] --pd-transfer-contention is not applicable to blis replay (PD disaggregation is not supported); flag ignored")
		}

		// Resolve policy configuration (single code path shared with runCmd).
		parsedScorerConfigs := resolvePolicies(cmd)

		logrus.Infof("Starting replay with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v",
			totalKVBlocks, replayHorizon, lr.AlphaCoeffs, lr.BetaCoeffs)

		startTime := time.Now()

		// Build cluster config (same as runCmd, using replayHorizon instead of simulationHorizon)
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: replayHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:         sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:       sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
				ModelHardwareConfig: sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, lr.Backend, maxModelLen),
				PolicyConfig:        sim.NewPolicyConfig(priorityPolicy, scheduler),
			},
			NumInstances:            numInstances,
			AdmissionPolicy:         admissionPolicy,
			AdmissionLatency:        admissionLatency,
			RoutingLatency:          routingLatency,
			TokenBucketCapacity:     tokenBucketCapacity,
			TokenBucketRefillRate:   tokenBucketRefillRate,
			RoutingPolicy:           routingPolicy,
			RoutingScorerConfigs:    parsedScorerConfigs,
			TraceLevel:              traceLevel,
			CounterfactualK:         counterfactualK,
			SnapshotRefreshInterval: snapshotRefreshInterval,
		}

		// Run simulation — no session manager (onRequestDone=nil: session structure encoded in trace)
		cs := cluster.NewClusterSimulator(config, requests, nil)
		if err := cs.Run(); err != nil {
			logrus.Fatalf("Replay simulation failed: %v", err)
		}

		logrus.Infof("Replay wall-clock time: %.3fs", time.Since(startTime).Seconds())

		// Save aggregate metrics to stdout (same as runCmd)
		if numInstances > 1 {
			for _, inst := range cs.Instances() {
				if err := inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, ""); err != nil {
					logrus.Fatalf("SaveResults for instance %s: %v", inst.ID(), err)
				}
			}
		}
		// Save aggregate (always print to stdout; SimResult output uses separate file)
		if err := cs.AggregatedMetrics().SaveResults("cluster", config.Horizon, totalKVBlocks, ""); err != nil {
			logrus.Fatalf("SaveResults: %v", err)
		}

		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			priorityPolicy,
			cs.RoutingRejections(),
		)
		rawMetrics.ShedByTier = cs.ShedByTier()                             // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
		rawMetrics.DeferredHorizonInterrupted = cs.DeferredQueueLen()        // Phase 1B-1b: deferred queue horizon count (FR-006)

		// Print anomaly counters if any detected
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.DeferredHorizonInterrupted > 0 {
			fmt.Println("=== Anomaly Counters ===")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests (Admission): %d\n", rawMetrics.RejectedRequests)
			if len(rawMetrics.ShedByTier) > 0 {
				tierKeys := make([]string, 0, len(rawMetrics.ShedByTier))
				for k := range rawMetrics.ShedByTier {
					tierKeys = append(tierKeys, k)
				}
				sort.Strings(tierKeys) // R2/INV-6: deterministic output order
				for _, tier := range tierKeys {
					fmt.Printf("  Shed (%s): %d\n", tier, rawMetrics.ShedByTier[tier])
				}
			}
			fmt.Printf("Rejected Requests (Routing): %d\n", rawMetrics.RoutingRejections)
			fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
			fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
			if rawMetrics.DeferredHorizonInterrupted > 0 {
				fmt.Printf("Deferred (horizon-interrupted): %d\n", rawMetrics.DeferredHorizonInterrupted)
			}
		}

		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions)

		if cs.Trace() != nil && summarizeTrace {
			traceSummary := trace.Summarize(cs.Trace())
			fmt.Println("=== Trace Summary ===")
			fmt.Printf("Total Decisions: %d\n", traceSummary.TotalDecisions)
			fmt.Printf("  Admitted: %d\n", traceSummary.AdmittedCount)
			fmt.Printf("  Rejected: %d\n", traceSummary.RejectedCount)
			fmt.Printf("Unique Targets: %d\n", traceSummary.UniqueTargets)
			if len(traceSummary.TargetDistribution) > 0 {
				fmt.Println("Target Distribution:")
				targetKeys := make([]string, 0, len(traceSummary.TargetDistribution))
				for k := range traceSummary.TargetDistribution {
					targetKeys = append(targetKeys, k)
				}
				sort.Strings(targetKeys)
				for _, k := range targetKeys {
					fmt.Printf("  %s: %d\n", k, traceSummary.TargetDistribution[k])
				}
			}
			fmt.Printf("Mean Regret: %.6f\n", traceSummary.MeanRegret)
			fmt.Printf("Max Regret: %.6f\n", traceSummary.MaxRegret)
		}

		// Warn if --fitness-weights is set (not supported in replay mode per R1)
		if fitnessWeights != "" {
			logrus.Warnf("--fitness-weights has no effect in replay mode (fitness evaluation not supported for replay)")
		}

		// Write SimResult JSON for calibrate consumption (BC-2)
		if resultsPath != "" {
			simResults := extractSimResults(cs.AggregatedMetrics())
			data, err := json.MarshalIndent(simResults, "", "  ")
			if err != nil {
				logrus.Fatalf("Failed to marshal SimResults: %v", err)
			}
			if err := os.WriteFile(resultsPath, data, 0644); err != nil {
				logrus.Fatalf("Failed to write SimResults to %s: %v", resultsPath, err)
			}
			logrus.Infof("SimResults written to %s (%d entries)", resultsPath, len(simResults))
		}

		logrus.Info("Replay complete.")
	},
}

func init() {
	registerSimConfigFlags(replayCmd)
	replayCmd.Flags().StringVar(&traceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (required)")
	replayCmd.Flags().StringVar(&traceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (required)")
	replayCmd.Flags().StringVar(&resultsPath, "results-path", "", "File to write []SimResult JSON (request_id, ttft_us, e2e_us, input_tokens, output_tokens) for blis calibrate consumption.")
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
