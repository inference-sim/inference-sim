package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
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
	calibrateITLDataPath          string
	// Goodput SLO targets (#1413). CLI > trace header precedence in calibrate
	// (no workload spec on calibrate path).
	calibrateGoodputSLOTTFT string
	calibrateGoodputSLOITL  string
	calibrateGoodputSLOE2E  string
	// Adapter-cost fidelity comparison vs the Digital Twin (US5, #1470). When
	// --adapter-reference is set, calibrate runs a standalone per-config DT
	// comparison (BLIS aggregate vs DT reference) instead of the real-vs-sim flow.
	calibrateAdapterReference  string
	calibrateSimMetrics        string
	calibrateSimMetricsBlind   string
	calibrateAdapterMAPEThresh float64
	// KV hit-rate comparison (#1583). --sim-metrics (reused from adapter mode) supplies
	// the simulator's aggregate cache_hit_rate; the tolerances gate the pass/fail
	// verdicts (absolute pp for hit-rate, MAPE fraction for TTFT).
	calibrateHitRateTolerancePP float64
	calibrateTTFTMapeThreshold  float64
	// Throughput comparison (#1647). --throughput-tolerance-pct gates the within-tolerance
	// verdict on raw output-token throughput; --num-gpus gates the per-GPU normalization.
	calibrateThroughputTolerancePct float64
	calibrateNumGPUs                int
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
		// Adapter-cost fidelity comparison mode (US5, #1470): BLIS aggregate vs DT
		// reference, per config. Standalone — does not use the real-vs-sim flow.
		if calibrateAdapterReference != "" {
			runAdapterReferenceComparison()
			return
		}

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

		// Validate bandwidth (R3, R20): NaN/Inf bypass computeUploadDelay's ≤0 guard
		if math.IsNaN(calibrateNetworkBandwidthMbps) || math.IsInf(calibrateNetworkBandwidthMbps, 0) {
			logrus.Fatalf("--network-bandwidth-mbps must be a finite number, got %v", calibrateNetworkBandwidthMbps)
		}

		// Validate hit-rate comparison tolerances (R3): finite and non-negative.
		if math.IsNaN(calibrateHitRateTolerancePP) || math.IsInf(calibrateHitRateTolerancePP, 0) || calibrateHitRateTolerancePP < 0 {
			logrus.Fatalf("--hit-rate-tolerance-pp must be a finite number >= 0, got %v", calibrateHitRateTolerancePP)
		}
		if math.IsNaN(calibrateTTFTMapeThreshold) || math.IsInf(calibrateTTFTMapeThreshold, 0) || calibrateTTFTMapeThreshold < 0 {
			logrus.Fatalf("--ttft-mape-threshold must be a finite number >= 0, got %v", calibrateTTFTMapeThreshold)
		}

		// Validate throughput comparison flags (#1647, R3/R20).
		if math.IsNaN(calibrateThroughputTolerancePct) || math.IsInf(calibrateThroughputTolerancePct, 0) || calibrateThroughputTolerancePct < 0 {
			logrus.Fatalf("--throughput-tolerance-pct must be a finite number >= 0, got %v", calibrateThroughputTolerancePct)
		}
		if calibrateNumGPUs < 0 {
			logrus.Fatalf("--num-gpus must be >= 0, got %d", calibrateNumGPUs)
		}

		config := workload.CalibrationConfig{
			WarmUpRequests: warmUp,
			NetworkRTTUs:   networkRTTUs,
			BandwidthMbps:  calibrateNetworkBandwidthMbps,
		}

		// Step 4.5: Load ITL data if provided
		var itlRecords []workload.ITLRecord
		if calibrateITLDataPath != "" {
			itlRecords, err = workload.LoadITL(calibrateITLDataPath)
			if err != nil {
				logrus.Fatalf("Failed to load ITL data from %s: %v", calibrateITLDataPath, err)
			}
		}

		// Step 5: Prepare calibration pairs (with or without ITL)
		var pairs *workload.CalibrationPairs
		if len(itlRecords) > 0 {
			pairs, err = workload.PrepareCalibrationPairsWithITL(trace.Records, simResults, itlRecords, &config)
		} else {
			pairs, _, err = workload.PrepareCalibrationPairs(trace.Records, simResults, &config)
		}
		if err != nil {
			logrus.Fatalf("Failed to prepare calibration pairs: %v", err)
		}
		// Guard against zero matched pairs (R1: no silent data loss, BC-10)
		if pairs.MatchedCount == 0 {
			logrus.Fatalf("No matching request IDs found between trace and sim results — check that both files use the same request ID numbering")
		}
		// Additional guard for corrupt-timestamp edge case (#700): MatchedCount can
		// be non-zero while TTFT vectors are empty (all matched requests had negative
		// real latencies). BuildCalibrationReport would silently produce an empty metrics map.
		if len(pairs.TTFT.Real) == 0 {
			logrus.Fatalf("No valid latency pairs after filtering — all matched requests may have corrupt timing data (negative latencies)")
		}

		// Step 6: Build report (empty ConfigMatchInfo — deferred, see TODO)
		// TODO: populate ConfigMatchInfo by comparing trace.Header.Server against sim config (#658)
		configMatch := workload.ConfigMatchInfo{}
		report, err := workload.BuildCalibrationReport(pairs, &configMatch)
		if err != nil {
			logrus.Fatalf("Failed to build calibration report: %v", err)
		}

		// Step 6.5: Goodput comparison (#1413, BC-9). Resolve targets from CLI > trace header.
		cliTTFT, cliITL, cliE2E, gpErr := resolveGoodputCLIFlags(calibrateGoodputSLOTTFT, calibrateGoodputSLOITL, calibrateGoodputSLOE2E)
		if gpErr != nil {
			logrus.Fatalf("%v", gpErr)
		}
		goodputTargets := mergeGoodputTargets(cliTTFT, cliITL, cliE2E, trace.Header.GoodputSLOTargets, nil)

		// Build the matched-set shared by the goodput (#1413) and throughput (#1647)
		// comparisons: trace records whose RequestID has a matching SimResult AND that
		// survived warm-up exclusion. Constructed once here (cheap, already O(N)); it is
		// read-only for both consumers.
		simByID := make(map[int]workload.SimResult, len(simResults))
		for _, sr := range simResults {
			simByID[sr.RequestID] = sr
		}
		matched := make(map[int]bool, pairs.MatchedCount)
		for _, rec := range trace.Records {
			if rec.RequestID < warmUp {
				continue
			}
			if _, ok := simByID[rec.RequestID]; ok {
				matched[rec.RequestID] = true
			}
		}

		if len(goodputTargets) > 0 {
			// ITL index by RequestID (goodput-only).
			itlByRequest := make(map[int][]workload.ITLRecord)
			for _, r := range itlRecords {
				itlByRequest[r.RequestID] = append(itlByRequest[r.RequestID], r)
			}
			// Real-side runtime: span from earliest send to latest last-chunk over matched set.
			var firstSend, lastChunk int64
			firstSend = -1
			for _, rec := range trace.Records {
				if !matched[rec.RequestID] {
					continue
				}
				if firstSend == -1 || rec.SendTimeUs < firstSend {
					firstSend = rec.SendTimeUs
				}
				if rec.LastChunkTimeUs > lastChunk {
					lastChunk = rec.LastChunkTimeUs
				}
			}
			runtimeSec := 0.0
			if firstSend >= 0 && lastChunk > firstSend {
				runtimeSec = float64(lastChunk-firstSend) / 1e6
			}

			report.Goodput = workload.ComputeGoodputComparison(
				trace.Records, simByID, matched, itlByRequest, goodputTargets, runtimeSec,
			)
			if report.Goodput != nil && report.Goodput.SkippedITL {
				logrus.Warn("calibrate goodput: ITL gating skipped — ITL data missing on real or sim side; TTFT/E2E rows still computed.")
			}
		}

		// Step 6.7: KV cache hit-rate comparison (#1583, BC-6/BC-12). Compares the
		// real observed hit-rate (trace header, from observe --scrape-kv-metrics)
		// against the simulator's aggregate cache_hit_rate (--sim-metrics MetricsOutput
		// from replay --metrics-path). Skipped with a warning when either side is
		// missing — never a fabricated comparison (R1).
		if obs := trace.Header.ObservedKVMetrics; obs != nil {
			if calibrateSimMetrics == "" {
				logrus.Warnf("calibrate: trace header carries an observed KV hit-rate (source=%s, %.4f) but --sim-metrics was not provided; skipping the hit-rate comparison. Produce it with `blis replay --metrics-path <file>` and pass --sim-metrics <file>.", obs.Source, obs.HitRate)
			} else {
				simHR, hrErr := loadSimCacheHitRate(calibrateSimMetrics)
				if hrErr != nil {
					logrus.Fatalf("%v", hrErr)
				}
				if simHR == nil {
					logrus.Warnf("calibrate: --sim-metrics %s has no cache_hit_rate; skipping the hit-rate comparison. Regenerate it with `blis replay --metrics-path` (a bare stdout run does not carry cache_hit_rate).", calibrateSimMetrics)
				} else {
					report.HitRate = workload.ComputeHitRateComparison(obs.HitRate, *simHR, calibrateHitRateTolerancePP, obs.Source)
				}
			}
		}

		// Step 6.8: Throughput comparison (#1647). Real vs sim aggregate output-token
		// throughput and request-completion rate over the same matched set, both in the
		// client frame (config supplies the network normalization). Emitted when derivable
		// from the required inputs (nil otherwise, so no Inf/NaN reaches JSON marshaling).
		report.Throughput = workload.ComputeThroughputComparison(
			trace.Records, simByID, matched, &config, calibrateNumGPUs, calibrateThroughputTolerancePct,
		)
		if report.Throughput == nil {
			// Never silently drop the block when the operator asked for a verdict (R1):
			// distinguish "not derivable" from "feature absent".
			if calibrateThroughputTolerancePct > 0 || calibrateNumGPUs > 0 {
				logrus.Warnf("calibrate: throughput comparison skipped — not derivable over the matched set (no positive real+sim makespan or zero matched output tokens); --throughput-tolerance-pct/--num-gpus have no effect.")
			}
		} else {
			// Coverage guard: throughput normalizes matched output tokens by a makespan
			// over the SAME survivor set, so if the sim completed only a fraction of the
			// eligible real requests, real tok/s is deflated toward sim and the verdict can
			// spuriously PASS. Warn when matched coverage of the (warm-up-filtered) real set
			// is low, so a dropped-request gap is visible, not masked (R1).
			eligibleReal := pairs.MatchedCount + pairs.UnmatchedReal
			if eligibleReal > 0 {
				coverage := float64(pairs.MatchedCount) / float64(eligibleReal)
				if coverage < 0.9 {
					logrus.Warnf("calibrate: throughput matched only %d/%d (%.0f%%) of eligible real requests — the sim did not complete the rest, so real throughput is measured over the survivor subset and the comparison may understate a coverage gap.",
						pairs.MatchedCount, eligibleReal, coverage*100)
				}
			}
			// Validity boundary (open-loop only): the reconstructed sim makespan
			// (send + simE2E) is a physical timeline only for fixed-mode replay. A
			// delta-encoded / closed-loop corpus regenerates its own arrival schedule,
			// so the real send schedule is not the sim's — warn rather than silently
			// emit a verdict that violates the boundary.
			if trace.Header.SessionContextGrowth != "" {
				logrus.Warnf("calibrate: throughput validity boundary — trace session_context_growth=%q indicates a closed-loop/delta corpus; the throughput comparison is open-loop only and may not be meaningful here.", trace.Header.SessionContextGrowth)
			}
		}

		// TTFT-MAPE tolerance verdict (#1583, BC-7): recorded in the report (not just
		// logged) so automation consumers see it. Set before the write below.
		if ttft, ok := report.Metrics["ttft"]; ok {
			report.TTFTTolerance = &workload.ToleranceVerdict{
				MAPE:      ttft.RequestLevel.MAPE,
				Threshold: calibrateTTFTMapeThreshold,
				Within:    ttft.RequestLevel.MAPE <= calibrateTTFTMapeThreshold,
			}
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

		// Workload-level distribution statistics
		logrus.Infof("Workload-level aggregate metrics:")
		if ttft, ok := report.Metrics["ttft"]; ok {
			logrus.Infof("  TTFT: Real mean=%.0fµs, Sim mean=%.0fµs, Error=%+.0fµs (%.1f%%)",
				ttft.WorkloadLevel.RealMean, ttft.WorkloadLevel.SimMean, ttft.WorkloadLevel.MeanError, ttft.WorkloadLevel.MeanPercentError*100)
			logrus.Infof("        P50: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				ttft.WorkloadLevel.RealP50, ttft.WorkloadLevel.SimP50, ttft.WorkloadLevel.P50Error, ttft.WorkloadLevel.P50PercentError*100)
			logrus.Infof("        P90: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				ttft.WorkloadLevel.RealP90, ttft.WorkloadLevel.SimP90, ttft.WorkloadLevel.P90Error, ttft.WorkloadLevel.P90PercentError*100)
			logrus.Infof("        P95: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				ttft.WorkloadLevel.RealP95, ttft.WorkloadLevel.SimP95, ttft.WorkloadLevel.P95Error, ttft.WorkloadLevel.P95PercentError*100)
			logrus.Infof("        P99: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				ttft.WorkloadLevel.RealP99, ttft.WorkloadLevel.SimP99, ttft.WorkloadLevel.P99Error, ttft.WorkloadLevel.P99PercentError*100)
		}
		if e2e, ok := report.Metrics["e2e"]; ok {
			logrus.Infof("  E2E:  Real mean=%.0fµs, Sim mean=%.0fµs, Error=%+.0fµs (%.1f%%)",
				e2e.WorkloadLevel.RealMean, e2e.WorkloadLevel.SimMean, e2e.WorkloadLevel.MeanError, e2e.WorkloadLevel.MeanPercentError*100)
			logrus.Infof("        P50: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				e2e.WorkloadLevel.RealP50, e2e.WorkloadLevel.SimP50, e2e.WorkloadLevel.P50Error, e2e.WorkloadLevel.P50PercentError*100)
			logrus.Infof("        P90: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				e2e.WorkloadLevel.RealP90, e2e.WorkloadLevel.SimP90, e2e.WorkloadLevel.P90Error, e2e.WorkloadLevel.P90PercentError*100)
			logrus.Infof("        P95: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				e2e.WorkloadLevel.RealP95, e2e.WorkloadLevel.SimP95, e2e.WorkloadLevel.P95Error, e2e.WorkloadLevel.P95PercentError*100)
			logrus.Infof("        P99: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				e2e.WorkloadLevel.RealP99, e2e.WorkloadLevel.SimP99, e2e.WorkloadLevel.P99Error, e2e.WorkloadLevel.P99PercentError*100)
		}
		if itl, ok := report.Metrics["itl"]; ok {
			logrus.Infof("  ITL:  Real mean=%.0fµs, Sim mean=%.0fµs, Error=%+.0fµs (%.1f%%)",
				itl.WorkloadLevel.RealMean, itl.WorkloadLevel.SimMean, itl.WorkloadLevel.MeanError, itl.WorkloadLevel.MeanPercentError*100)
			logrus.Infof("        P50: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				itl.WorkloadLevel.RealP50, itl.WorkloadLevel.SimP50, itl.WorkloadLevel.P50Error, itl.WorkloadLevel.P50PercentError*100)
			logrus.Infof("        P90: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				itl.WorkloadLevel.RealP90, itl.WorkloadLevel.SimP90, itl.WorkloadLevel.P90Error, itl.WorkloadLevel.P90PercentError*100)
			logrus.Infof("        P95: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				itl.WorkloadLevel.RealP95, itl.WorkloadLevel.SimP95, itl.WorkloadLevel.P95Error, itl.WorkloadLevel.P95PercentError*100)
			logrus.Infof("        P99: Real=%.0fµs, Sim=%.0fµs, Error=%+.0fµs (%.1f%%)",
				itl.WorkloadLevel.RealP99, itl.WorkloadLevel.SimP99, itl.WorkloadLevel.P99Error, itl.WorkloadLevel.P99PercentError*100)
		}

		// Request-level prediction quality
		logrus.Infof("Request-level prediction quality:")
		if ttft, ok := report.Metrics["ttft"]; ok {
			logrus.Infof("  TTFT: MAPE=%.1f%%, PearsonR=%.3f, Bias=%s, Quality=%s",
				ttft.RequestLevel.MAPE*100, ttft.RequestLevel.PearsonR, ttft.RequestLevel.BiasDirection, ttft.RequestLevel.Quality)
		}
		if e2e, ok := report.Metrics["e2e"]; ok {
			logrus.Infof("  E2E:  MAPE=%.1f%%, PearsonR=%.3f, Bias=%s, Quality=%s",
				e2e.RequestLevel.MAPE*100, e2e.RequestLevel.PearsonR, e2e.RequestLevel.BiasDirection, e2e.RequestLevel.Quality)
		}
		if itl, ok := report.Metrics["itl"]; ok {
			logrus.Infof("  ITL:  MAPE=%.1f%%, PearsonR=%.3f, Bias=%s, Quality=%s",
				itl.RequestLevel.MAPE*100, itl.RequestLevel.PearsonR, itl.RequestLevel.BiasDirection, itl.RequestLevel.Quality)
		}

		// KV cache hit-rate verdict + TTFT tolerance verdict (#1583, BC-6/BC-7).
		if report.HitRate != nil {
			hr := report.HitRate
			verdict := "WITHIN"
			logFn := logrus.Infof
			if !hr.Within {
				verdict = "EXCEEDS"
				logFn = logrus.Warnf
			}
			logFn("KV hit-rate (source=%s): Real=%.4f Sim=%.4f AbsErr=%.2fpp (tolerance %.2fpp) [%s]",
				hr.Source, hr.RealHitRate, hr.SimHitRate, hr.AbsErrorPP, hr.TolerancePP, verdict)
		}
		if tv := report.TTFTTolerance; tv != nil {
			mapePct := tv.MAPE * 100
			threshPct := tv.Threshold * 100
			if tv.Within {
				logrus.Infof("TTFT MAPE=%.1f%% (threshold %.1f%%) [WITHIN]", mapePct, threshPct)
			} else {
				logrus.Warnf("TTFT MAPE=%.1f%% (threshold %.1f%%) [EXCEEDS]", mapePct, threshPct)
			}
		}

		// Throughput summary (#1647).
		if tp := report.Throughput; tp != nil {
			// When tp != nil, both real rates are guaranteed > 0 (matched > 0, real runtime > 0,
			// real output tokens > 0), so the signed relative errors below never divide by zero.
			logrus.Infof("Throughput: output tok/s Real=%.1f Sim=%.1f (%+.1f%%), req/s Real=%.2f Sim=%.2f (%+.1f%%)",
				tp.RealOutputTokensPerSec, tp.SimOutputTokensPerSec, tp.OutputTokensPerSecError/tp.RealOutputTokensPerSec*100,
				tp.RealRequestsPerSec, tp.SimRequestsPerSec, tp.RequestsPerSecError/tp.RealRequestsPerSec*100)
			if tp.RealOutputTokensPerSecPerGPU != nil {
				logrus.Infof("            output tok/s/GPU Real=%.1f Sim=%.1f (num_gpus=%d)",
					*tp.RealOutputTokensPerSecPerGPU, *tp.SimOutputTokensPerSecPerGPU, *tp.NumGPUs)
			}
			if tp.Within != nil {
				verdict := "WITHIN"
				logFn := logrus.Infof
				if !*tp.Within {
					verdict = "EXCEEDS"
					logFn = logrus.Warnf
				}
				logFn("Throughput output-token error=%.1f%% (tolerance %.1f%%) [%s]",
					tp.OutputTokensPerSecPercentError*100, *tp.TolerancePct, verdict)
			}
		}

		// Per-SLO class breakdown (when data present)
		if len(pairs.BySLO) > 0 {
			sloKeys := make([]string, 0, len(pairs.BySLO))
			for k := range pairs.BySLO {
				sloKeys = append(sloKeys, k)
			}
			sort.Strings(sloKeys) // deterministic stderr output (R2)
			logrus.Infof("Per-SLO-class calibration:")
			for _, slo := range sloKeys {
				p := pairs.BySLO[slo]
				if p == nil || len(p.TTFT.Real) == 0 {
					continue
				}
				logrus.Infof("  SLO=%s: n=%d TTFT-MAPE=%.1f%% E2E-MAPE=%.1f%%",
					slo, len(p.TTFT.Real),
					workload.MapePct(p.TTFT.Real, p.TTFT.Sim)*100,
					workload.MapePct(p.E2E.Real, p.E2E.Sim)*100)
			}
		}
		// Per-class goodput summary (#1413, BC-9)
		if report.Goodput != nil && len(report.Goodput.PerClass) > 0 {
			logrus.Infof("Per-SLO-class goodput (real vs sim):")
			classKeys := make([]string, 0, len(report.Goodput.PerClass))
			for k := range report.Goodput.PerClass {
				classKeys = append(classKeys, k)
			}
			sort.Strings(classKeys)
			for _, cls := range classKeys {
				gc := report.Goodput.PerClass[cls]
				logrus.Infof("  %s: count=%d real_attain=%.3f sim_attain=%.3f real_rps=%.3f sim_rps=%.3f",
					cls, gc.Count, gc.RealSLOAttainment, gc.SimSLOAttainment, gc.RealGoodputRPS, gc.SimGoodputRPS)
			}
		}

		// Per-model breakdown (when data present)
		if len(pairs.ByModel) > 0 {
			modelKeys := make([]string, 0, len(pairs.ByModel))
			for k := range pairs.ByModel {
				modelKeys = append(modelKeys, k)
			}
			sort.Strings(modelKeys) // deterministic stderr output (R2)
			logrus.Infof("Per-model calibration:")
			for _, model := range modelKeys {
				p := pairs.ByModel[model]
				if p == nil || len(p.TTFT.Real) == 0 {
					continue
				}
				logrus.Infof("  model=%s: n=%d TTFT-MAPE=%.1f%% E2E-MAPE=%.1f%%",
					model, len(p.TTFT.Real),
					workload.MapePct(p.TTFT.Real, p.TTFT.Sim)*100,
					workload.MapePct(p.E2E.Real, p.E2E.Sim)*100)
			}
		}
	},
}

// loadBLISAggregate reads a BLIS MetricsOutput JSON file (from blis run/replay
// --metrics-path) and distills the aggregate TTFT + output throughput needed for
// the DT comparison. Output throughput = total_output_tokens / estimated duration.
func loadBLISAggregate(path string) workload.BLISAggregate {
	data, err := os.ReadFile(path)
	if err != nil {
		logrus.Fatalf("Failed to read sim metrics from %s: %v", path, err)
	}
	var m sim.MetricsOutput
	if err := json.Unmarshal(data, &m); err != nil {
		logrus.Fatalf("Failed to parse sim metrics JSON from %s: %v", path, err)
	}
	if m.VllmDurationSec <= 0 {
		logrus.Fatalf("sim metrics %s: vllm_estimated_duration_s must be > 0 to compute throughput, got %v", path, m.VllmDurationSec)
	}
	return workload.BLISAggregate{
		TTFTMs:           m.TTFTMeanMs,
		OutputThroughput: float64(m.TotalOutputTokens) / m.VllmDurationSec,
	}
}

// loadSimCacheHitRate reads a BLIS MetricsOutput JSON (from `blis run/replay
// --metrics-path`) and returns its aggregate cache_hit_rate (#1583). A nil return
// with nil error means the file parsed but carries no cache_hit_rate (e.g. produced
// without --metrics-path, or by a pre-feature binary) — the caller warns and skips
// the hit-rate comparison rather than fabricating a value.
func loadSimCacheHitRate(path string) (*float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read --sim-metrics %s: %w", path, err)
	}
	var m sim.MetricsOutput
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("parse --sim-metrics %s: %w", path, err)
	}
	return m.CacheHitRate, nil
}

// runAdapterReferenceComparison implements the US5 (#1470) DT fidelity comparison:
// BLIS aggregate (from --sim-metrics) vs a committed DT reference (--adapter-reference),
// per config, on TTFT and output throughput. Writes the report JSON and logs a
// per-metric pass/fail against --adapter-mape-threshold. Configs exceeding the bound
// are reported honestly (never a silent pass); the mechanism ships regardless of the
// empirical outcome (design §15).
func runAdapterReferenceComparison() {
	if calibrateSimMetrics == "" {
		logrus.Fatalf("--sim-metrics is required with --adapter-reference (BLIS aggregate MetricsOutput from run/replay --metrics-path)")
	}
	if calibrateReportPath == "" {
		logrus.Fatalf("--report is required")
	}
	if math.IsNaN(calibrateAdapterMAPEThresh) || math.IsInf(calibrateAdapterMAPEThresh, 0) || calibrateAdapterMAPEThresh <= 0 {
		logrus.Fatalf("--adapter-mape-threshold must be a finite number > 0, got %v", calibrateAdapterMAPEThresh)
	}

	ref, err := workload.LoadDTReference(calibrateAdapterReference)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	aware := loadBLISAggregate(calibrateSimMetrics)
	var blind *workload.BLISAggregate
	if calibrateSimMetricsBlind != "" {
		b := loadBLISAggregate(calibrateSimMetricsBlind)
		blind = &b
	}

	report := workload.CompareAdapterReference(ref, aware, blind, calibrateAdapterMAPEThresh)

	reportData, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		logrus.Fatalf("Failed to marshal adapter-reference report: %v", err)
	}
	if err := os.WriteFile(calibrateReportPath, reportData, 0644); err != nil {
		logrus.Fatalf("Failed to write adapter-reference report to %s: %v", calibrateReportPath, err)
	}

	logrus.Infof("Adapter-reference comparison (%s) written to %s (MAPE threshold %.0f%%)",
		report.Model, calibrateReportPath, report.Threshold*100)
	for _, m := range report.Metrics {
		verdict := "WITHIN"
		if !m.Within {
			verdict = "EXCEEDS"
		}
		line := fmt.Sprintf("  %-11s BLIS=%.2f DT=%.2f MAPE=%.1f%% [%s]",
			m.Metric, m.BLISValue, m.DTValue, m.MAPE*100, verdict)
		if m.DeltaMAPE != nil {
			line += fmt.Sprintf(" delta-normalized MAPE=%.1f%%", *m.DeltaMAPE*100)
		}
		if m.Within {
			logrus.Info(line)
		} else {
			logrus.Warn(line)
		}
	}
	// Surface silent degradation (R1): the user asked for the delta-normalized
	// diagnostic (--sim-metrics-blind) but a zero blind denominator on either side
	// prevented computing it for a metric.
	if calibrateSimMetricsBlind != "" {
		for _, m := range report.Metrics {
			if m.DeltaMAPE == nil {
				logrus.Warnf("delta-normalized diagnostic skipped for %q: the DT reference's adapter_blind %s is zero or the BLIS blind aggregate is zero (aware/blind ratio undefined).", m.Metric, m.Metric)
			}
		}
	}
	if !report.AllWithin {
		logrus.Warnf("SC-007 not satisfied on all metrics for %s at MAPE<=%.0f%% — that dimension's fidelity claim is unsupported for this config (design §15 falsification path). The comparison mechanism succeeded; the bound was not met.",
			report.Model, report.Threshold*100)
	}
}

func init() {
	calibrateCmd.Flags().StringVar(&calibrateTraceHeaderPath, "trace-header", "", "Path to TraceV2 header YAML file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateTraceDataPath, "trace-data", "", "Path to TraceV2 data CSV file (from blis observe; required)")
	calibrateCmd.Flags().StringVar(&calibrateSimResultsPath, "sim-results", "", "Path to SimResult JSON file (from blis replay --results-path; required)")
	calibrateCmd.Flags().StringVar(&calibrateReportPath, "report", "", "Path to write calibration report JSON (required)")
	calibrateCmd.Flags().IntVar(&calibrateWarmUpRequests, "warmup-requests", -1, "Number of initial requests to exclude (default: from trace header warm_up_requests; pass 0 to include all)")
	calibrateCmd.Flags().Int64Var(&calibrateNetworkRTTUs, "network-rtt-us", -1, "Network RTT in microseconds added to sim-side latencies (default: from trace header network.measured_rtt_ms)")
	calibrateCmd.Flags().Float64Var(&calibrateNetworkBandwidthMbps, "network-bandwidth-mbps", 0, "Network bandwidth in Mbps for upload/download delay calculation (default: 0 = no delay)")
	calibrateCmd.Flags().StringVar(&calibrateITLDataPath, "itl-data", "", "Optional path to ITL CSV file (from blis observe --record-itl) to include ITL metric in calibration report")
	calibrateCmd.Flags().StringVar(&calibrateGoodputSLOTTFT, "slo-ttft", "", "Per-class TTFT goodput thresholds (e.g. \"critical=100ms,standard=500ms\"). Precedence: CLI > trace header.")
	calibrateCmd.Flags().StringVar(&calibrateGoodputSLOITL, "slo-itl", "", "Per-class mean ITL goodput thresholds. Skipped with a warning when ITL data is absent on either side.")
	calibrateCmd.Flags().StringVar(&calibrateGoodputSLOE2E, "slo-e2e", "", "Per-class E2E goodput thresholds (e.g. \"critical=5s,standard=30s\").")
	// Adapter-cost fidelity comparison vs the Digital Twin (US5, #1470).
	calibrateCmd.Flags().StringVar(&calibrateAdapterReference, "adapter-reference", "", "Path to a DT reference JSON (per-config adapter_aware/adapter_blind aggregates). Enables standalone BLIS-vs-DT fidelity comparison; --trace-header/--trace-data/--sim-results are not used in this mode.")
	calibrateCmd.Flags().StringVar(&calibrateSimMetrics, "sim-metrics", "", "Path to a BLIS aggregate MetricsOutput JSON (from blis run/replay --metrics-path). Required with --adapter-reference (adapter-aware run). In the standard flow it supplies the simulator's cache_hit_rate for the KV hit-rate comparison (#1583).")
	calibrateCmd.Flags().Float64Var(&calibrateHitRateTolerancePP, "hit-rate-tolerance-pp", 5.0, "Absolute-error band (percentage points) for the KV cache hit-rate comparison (#1583). Real (trace header observed) vs sim (--sim-metrics cache_hit_rate).")
	calibrateCmd.Flags().Float64Var(&calibrateTTFTMapeThreshold, "ttft-mape-threshold", 0.15, "TTFT MAPE threshold (fraction) for the informational TTFT tolerance verdict (#1583; issue target 0.15).")
	calibrateCmd.Flags().StringVar(&calibrateSimMetricsBlind, "sim-metrics-blind", "", "Optional BLIS MetricsOutput JSON for the adapter-blind baseline run; enables the delta-normalized (aware/blind) diagnostic that isolates the ported adapter physics.")
	calibrateCmd.Flags().Float64Var(&calibrateAdapterMAPEThresh, "adapter-mape-threshold", 0.20, "MAPE bound (fraction) for the adapter-reference comparison (SC-007 target 0.20).")
	calibrateCmd.Flags().Float64Var(&calibrateThroughputTolerancePct, "throughput-tolerance-pct", 0, "Within-tolerance verdict (percent) on real-vs-sim output-token throughput (#1647). Verdict emitted only when > 0.")
	calibrateCmd.Flags().IntVar(&calibrateNumGPUs, "num-gpus", 0, "GPU count (TP×PP×DP×instances) for per-GPU throughput normalization (#1647). Per-GPU fields emitted only when > 0. Operator-supplied since the trace header records only TP.")
	rootCmd.AddCommand(calibrateCmd)
}
