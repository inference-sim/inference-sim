package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

var (
	traceHeaderPath     string
	traceDataPath       string
	replayTraceOutput   string // File prefix for TraceV2 re-export (<prefix>.yaml + <prefix>.csv)
	replaySessionMode   string
	replayThinkTimeMs   int
	replayThinkTimeDist string // distribution spec for think time (e.g. "lognormal:mu=2.0,sigma=0.6,min=3s,max=30s")
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

		// Validate session mode flags (BC-11)
		if replaySessionMode != "fixed" && replaySessionMode != "closed-loop" {
			logrus.Fatalf("--session-mode must be \"fixed\" or \"closed-loop\", got %q", replaySessionMode)
		}
		if replayThinkTimeMs < 0 {
			logrus.Fatalf("--think-time-ms must be non-negative, got %d", replayThinkTimeMs)
		}
		if replayThinkTimeMs > 0 && replaySessionMode != "closed-loop" {
			logrus.Fatalf("--think-time-ms requires --session-mode closed-loop")
		}
		if replayThinkTimeDist != "" && replaySessionMode != "closed-loop" {
			logrus.Fatalf("--think-time-dist requires --session-mode closed-loop")
		}
		if cmd.Flags().Changed("think-time-ms") && cmd.Flags().Changed("think-time-dist") {
			logrus.Fatalf("--think-time-ms and --think-time-dist are mutually exclusive")
		}

		// Resolve think-time sampler: --think-time-dist takes the general distribution;
		// --think-time-ms is a convenience alias for constant:<N>ms.
		// Neither → nil (derive per-session think time from trace arrival gaps).
		var thinkTimeSampler workload.LengthSampler
		if cmd.Flags().Changed("think-time-dist") {
			var err error
			thinkTimeSampler, err = workload.ParseThinkTimeDist(replayThinkTimeDist)
			if err != nil {
				logrus.Fatalf("--think-time-dist: %v", err)
			}
		} else if replayThinkTimeMs > 0 {
			var err error
			thinkTimeSampler, err = workload.ParseThinkTimeDist(fmt.Sprintf("constant:value=%dms", replayThinkTimeMs))
			if err != nil {
				logrus.Fatalf("--think-time-ms: %v", err)
			}
		}

		// Build requests from trace — mode selects pre-baked vs closed-loop (BC-8, BC-9)
		var requests []*sim.Request
		var sessionMgr *workload.SessionManager
		if replaySessionMode == "closed-loop" {
			// Closed-loop: inject only round-0 requests; SessionManager drives follow-ups.
			// Compute the preliminary horizon from trace records directly (O(n)) so we can
			// call LoadTraceV2SessionBlueprints exactly once with correct parameters.
			replayHorizonPrelim := computeHorizonFromMaxArrival(maxInjectedArrivalTimeUs(traceData))
			if cmd.Flags().Changed("horizon") {
				replayHorizonPrelim = simulationHorizon
			}
			r0Requests, blueprints, bErr := workload.LoadTraceV2SessionBlueprints(traceData, seed, thinkTimeSampler, replayHorizonPrelim)
			if bErr != nil {
				logrus.Fatalf("Failed to build session blueprints from trace: %v", bErr)
			}
			requests = r0Requests
			if len(blueprints) == 0 {
				// BC-12: warning path — no automated unit test (integration-level only)
				logrus.Warnf("--session-mode closed-loop: no session records found in trace; all requests injected with fixed timing")
			} else {
				sessionMgr = workload.NewSessionManager(blueprints)
				logrus.Infof("Closed-loop mode: %d session blueprints, %d round-0 requests", len(blueprints), len(requests))
			}
		} else {
			// Fixed mode (default): pre-baked arrivals, existing behavior (BC-8)
			var bErr error
			requests, bErr = workload.LoadTraceV2Requests(traceData, seed)
			if bErr != nil {
				logrus.Fatalf("Failed to build requests from trace: %v", bErr)
			}
			logrus.Infof("Built %d requests for replay", len(requests))
		}

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

		// Resolve policy configuration (single code path shared with runCmd).
		// Autoscaler and node-pool configs are not supported in replay — fail fast
		// rather than silently producing divergent results (INV-13, Track B).
		parsedScorerConfigs, bundle := resolvePolicies(cmd)
		if cmd.Flags().Changed("model-autoscaler-interval-us") {
			logrus.Fatalf("--model-autoscaler-interval-us is not supported in blis replay; remove this flag or use blis run instead")
		}
		if bundle != nil {
			if bundle.Autoscaler.IntervalUs > 0 {
				logrus.Fatalf("blis replay does not support autoscaler config (policy bundle interval_us=%g); remove the autoscaler section from the policy bundle or use blis run instead", bundle.Autoscaler.IntervalUs)
			}
			if len(bundle.NodePools) > 0 {
				logrus.Fatalf("blis replay does not support node_pools config (%d pool(s) in policy bundle); remove the node_pools section from the policy bundle or use blis run instead", len(bundle.NodePools))
			}
		}

		// PD disaggregation validation (same as runCmd, R3) — INV-13 Track A.
		if prefillInstances < 0 {
			logrus.Fatalf("--prefill-instances must be >= 0, got %d", prefillInstances)
		}
		if decodeInstances < 0 {
			logrus.Fatalf("--decode-instances must be >= 0, got %d", decodeInstances)
		}
		if prefillDecodeInstances < 0 {
			logrus.Fatalf("--prefill-decode-instances must be >= 0, got %d", prefillDecodeInstances)
		}
		if !sim.IsValidDisaggregationDecider(pdDecider) {
			logrus.Fatalf("Unknown PD decider %q. Valid: %s", pdDecider, strings.Join(sim.ValidDisaggregationDeciderNames(), ", "))
		}
		if err := cluster.ValidatePoolTopology(prefillInstances, decodeInstances, prefillDecodeInstances, numInstances); err != nil {
			logrus.Fatalf("Invalid PD pool topology: %v", err)
		}
		if prefillInstances > 0 {
			if pdTransferBandwidth <= 0 || math.IsInf(pdTransferBandwidth, 0) || math.IsNaN(pdTransferBandwidth) {
				logrus.Fatalf("--pd-transfer-bandwidth must be a finite positive number, got %f", pdTransferBandwidth)
			}
			if pdTransferBaseLatency < 0 || math.IsInf(pdTransferBaseLatency, 0) || math.IsNaN(pdTransferBaseLatency) {
				logrus.Fatalf("--pd-transfer-base-latency must be a finite non-negative number, got %f", pdTransferBaseLatency)
			}
		}
		if pdDecider == "prefix-threshold" && pdPrefixThreshold < 0 {
			logrus.Fatalf("--pd-prefix-threshold must be >= 0, got %d", pdPrefixThreshold)
		}
		if pdDecider != "prefix-threshold" && cmd.Flags().Changed("pd-prefix-threshold") {
			logrus.Fatalf("--pd-prefix-threshold=%d has no effect when --pd-decider=%q (only applies to the prefix-threshold decider); remove the flag or set --pd-decider=prefix-threshold", pdPrefixThreshold, pdDecider)
		}
		if pdDecider != "" && pdDecider != "never" && prefillInstances == 0 {
			logrus.Fatalf("--pd-decider=%q has no effect because --prefill-instances=0 (disaggregation is disabled); set --prefill-instances > 0 and --decode-instances > 0, or omit --pd-decider", pdDecider)
		}

		// ModelConfig resolution for PD KV transfer sizing (same as runCmd, root.go:1045-1065).
		// When PD is active and an analytical backend is in use, the ModelConfig may need to
		// be loaded from the HF config to calculate per-pool KV block counts. If resolveLatencyConfig
		// already loaded it (roofline/trained-physics), lr.ModelConfig.NumHeads will be non-zero.
		if prefillInstances > 0 && lr.ModelConfig.NumHeads == 0 {
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("PD disaggregation requires model architecture for KV transfer sizing: %v", err)
			}
			hfPath := filepath.Join(resolved, "config.json")
			hfConfig, parseErr := latency.ParseHFConfig(hfPath)
			if parseErr != nil {
				logrus.Fatalf("PD disaggregation requires model architecture for KV transfer sizing, but failed to parse %s: %v", hfPath, parseErr)
			}
			mc, mcErr := latency.GetModelConfigFromHF(hfConfig)
			if mcErr != nil {
				logrus.Fatalf("PD disaggregation requires model architecture for KV transfer sizing, but failed to extract ModelConfig: %v", mcErr)
			}
			applyWeightPrecisionFallback(mc, model, hfConfig.Raw)
			if mc.BytesPerParam <= 0 {
				logrus.Fatalf("PD disaggregation: could not determine model precision (BytesPerParam=%v) from %s — ensure torch_dtype or dtype is present in config.json", mc.BytesPerParam, hfPath)
			}
			lr.ModelConfig = *mc
			logrus.Infof("PD disaggregation: loaded ModelConfig from %s for KV transfer derivation", hfPath)
		}

		// Per-pool hardware override construction (same as runCmd).
		var prefillOverrides, decodeOverrides cluster.PoolOverrides

		// Per-pool KV auto-calculation (same as runCmd, root.go:1076-1156).
		// When PD disaggregation is active and a pool uses different TP or GPU hardware,
		// compute per-pool KV blocks from model + hardware for analytical backends.
		if lr.Backend == "roofline" || lr.Backend == "trained-physics" {
			if prefillInstances > 0 {
				hfPath := filepath.Join(modelConfigFolder, "config.json")
				hfConfig, err := latency.ParseHFConfig(hfPath)
				if err != nil {
					logrus.Fatalf("Failed to parse HuggingFace config for per-pool KV calc: %v", err)
				}
				kvParamsPool, kvErrPool := latency.ExtractKVCapacityParams(hfConfig)
				if kvErrPool != nil {
					logrus.Warnf("per-pool KV auto-calculation skipped (could not extract model KV params: %v); both pools will use global total-kv-blocks=%d", kvErrPool, totalKVBlocks)
				} else {
					// Prefill pool auto-calc
					poolPrefillTP := tensorParallelism
					if cmd.Flags().Changed("prefill-tp") {
						poolPrefillTP = prefillTP
					}
					poolPrefillGPU := gpu
					if cmd.Flags().Changed("prefill-hardware") {
						poolPrefillGPU = prefillHardware
					}
					if poolPrefillTP != tensorParallelism || poolPrefillGPU != gpu {
						poolHC, hcErr := latency.GetHWConfig(hwConfigPath, poolPrefillGPU)
						if hcErr != nil {
							logrus.Warnf("--prefill-hardware: failed to load hardware config for GPU %q: %v; prefill pool will use global total-kv-blocks=%d", poolPrefillGPU, hcErr, totalKVBlocks)
						} else if poolHC.MemoryGiB <= 0 {
							logrus.Warnf("--prefill-hardware: GPU memory capacity not available for %q in hardware config; prefill pool will use global total-kv-blocks=%d", poolPrefillGPU, totalKVBlocks)
						} else {
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolPrefillTP, blockSizeTokens, gpuMemoryUtilization, kvParamsPool)
							if calcErr != nil {
								logrus.Fatalf("--prefill-tp/--prefill-hardware: KV capacity auto-calculation failed for prefill pool: %v", calcErr)
							} else {
								prefillOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--prefill-tp/--prefill-hardware: auto-calculated prefill pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolPrefillTP)
								if !cmd.Flags().Changed("prefill-max-model-len") {
									kvFeasibleMax := poolBlocks * int64(blockSizeTokens)
									if kvFeasibleMax < maxModelLen {
										prefillOverrides.MaxModelLen = &kvFeasibleMax
										logrus.Infof("--prefill-tp/--prefill-hardware: auto-capped prefill pool max-model-len=%d (pool KV capacity smaller than global)", kvFeasibleMax)
									}
								}
							}
						}
					}

					// Decode pool auto-calc
					poolDecodeTP := tensorParallelism
					if cmd.Flags().Changed("decode-tp") {
						poolDecodeTP = decodeTP
					}
					poolDecodeGPU := gpu
					if cmd.Flags().Changed("decode-hardware") {
						poolDecodeGPU = decodeHardware
					}
					if poolDecodeTP != tensorParallelism || poolDecodeGPU != gpu {
						poolHC, hcErr := latency.GetHWConfig(hwConfigPath, poolDecodeGPU)
						if hcErr != nil {
							logrus.Warnf("--decode-hardware: failed to load hardware config for GPU %q: %v; decode pool will use global total-kv-blocks=%d", poolDecodeGPU, hcErr, totalKVBlocks)
						} else if poolHC.MemoryGiB <= 0 {
							logrus.Warnf("--decode-hardware: GPU memory capacity not available for %q in hardware config; decode pool will use global total-kv-blocks=%d", poolDecodeGPU, totalKVBlocks)
						} else {
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolDecodeTP, blockSizeTokens, gpuMemoryUtilization, kvParamsPool)
							if calcErr != nil {
								logrus.Fatalf("--decode-tp/--decode-hardware: KV capacity auto-calculation failed for decode pool: %v", calcErr)
							} else {
								decodeOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--decode-tp/--decode-hardware: auto-calculated decode pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolDecodeTP)
								if !cmd.Flags().Changed("decode-max-model-len") {
									kvFeasibleMax := poolBlocks * int64(blockSizeTokens)
									if kvFeasibleMax < maxModelLen {
										decodeOverrides.MaxModelLen = &kvFeasibleMax
										logrus.Infof("--decode-tp/--decode-hardware: auto-capped decode pool max-model-len=%d (pool KV capacity smaller than global)", kvFeasibleMax)
									}
								}
							}
						}
					}
				}
			}
		}

		perPoolFlagsChanged := cmd.Flags().Changed("prefill-tp") || cmd.Flags().Changed("decode-tp") ||
			cmd.Flags().Changed("prefill-hardware") || cmd.Flags().Changed("decode-hardware") ||
			cmd.Flags().Changed("prefill-latency-model") || cmd.Flags().Changed("decode-latency-model") ||
			cmd.Flags().Changed("prefill-max-model-len") || cmd.Flags().Changed("decode-max-model-len")
		if perPoolFlagsChanged && prefillInstances == 0 {
			logrus.Fatalf("per-pool hardware flags (--prefill-tp, --decode-tp, etc.) have no effect when --prefill-instances=0 (disaggregation is disabled); either set --prefill-instances > 0 or remove the per-pool flags")
		}
		if prefillInstances > 0 {
			if cmd.Flags().Changed("prefill-tp") {
				if prefillTP <= 0 {
					logrus.Fatalf("--prefill-tp must be > 0, got %d", prefillTP)
				}
				tp := prefillTP
				prefillOverrides.TP = &tp
			}
			if cmd.Flags().Changed("prefill-hardware") {
				prefillOverrides.GPU = prefillHardware
			}
			if cmd.Flags().Changed("prefill-latency-model") {
				if !sim.IsValidLatencyBackend(prefillLatencyModel) {
					logrus.Fatalf("--prefill-latency-model %q is not a recognized backend; valid: %s",
						prefillLatencyModel, strings.Join(sim.ValidLatencyBackendNames(), ", "))
				}
				prefillOverrides.LatencyBackend = prefillLatencyModel
			}
			if cmd.Flags().Changed("prefill-max-model-len") {
				if prefillMaxModelLen <= 0 {
					logrus.Fatalf("--prefill-max-model-len must be > 0 when set, got %d", prefillMaxModelLen)
				}
				ml := prefillMaxModelLen
				prefillOverrides.MaxModelLen = &ml
			}
			if cmd.Flags().Changed("decode-tp") {
				if decodeTP <= 0 {
					logrus.Fatalf("--decode-tp must be > 0, got %d", decodeTP)
				}
				tp := decodeTP
				decodeOverrides.TP = &tp
			}
			if cmd.Flags().Changed("decode-hardware") {
				decodeOverrides.GPU = decodeHardware
			}
			if cmd.Flags().Changed("decode-latency-model") {
				if !sim.IsValidLatencyBackend(decodeLatencyModel) {
					logrus.Fatalf("--decode-latency-model %q is not a recognized backend; valid: %s",
						decodeLatencyModel, strings.Join(sim.ValidLatencyBackendNames(), ", "))
				}
				decodeOverrides.LatencyBackend = decodeLatencyModel
			}
			if cmd.Flags().Changed("decode-max-model-len") {
				if decodeMaxModelLen <= 0 {
					logrus.Fatalf("--decode-max-model-len must be > 0 when set, got %d", decodeMaxModelLen)
				}
				ml := decodeMaxModelLen
				decodeOverrides.MaxModelLen = &ml
			}
		}

		// Parse per-pool scorer configs (same as runCmd).
		var prefillScorerCfgs, decodeScorerCfgs []sim.ScorerConfig
		if prefillRoutingScorers != "" {
			var err error
			prefillScorerCfgs, err = sim.ParseScorerConfigs(prefillRoutingScorers)
			if err != nil {
				logrus.Fatalf("Invalid --prefill-routing-scorers: %v", err)
			}
		}
		if decodeRoutingScorers != "" {
			var err error
			decodeScorerCfgs, err = sim.ParseScorerConfigs(decodeRoutingScorers)
			if err != nil {
				logrus.Fatalf("Invalid --decode-routing-scorers: %v", err)
			}
		}

		logrus.Infof("Starting replay with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v",
			totalKVBlocks, replayHorizon, lr.AlphaCoeffs, lr.BetaCoeffs)

		startTime := time.Now()

		// Build cluster config (same as runCmd, using replayHorizon instead of simulationHorizon).
		// INV-13 SYNC POINT: PD fields below must stay in sync with cmd/root.go (runCmd
		// DeploymentConfig literal). See docs/contributing/standards/invariants.md INV-13.
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: replayHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:          sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:        sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
				ModelHardwareConfig:  sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, lr.Backend, maxModelLen),
				PolicyConfig:         sim.NewPolicyConfig(scheduler, preemptionPolicy),
				SLOPriorityOverrides: sloPriorityOverrides,
			},
			NumInstances:                    numInstances,
			AdmissionPolicy:                 admissionPolicy,
			AdmissionLatency:                admissionLatency,
			RoutingLatency:                  routingLatency,
			TokenBucketCapacity:             tokenBucketCapacity,
			TokenBucketRefillRate:           tokenBucketRefillRate,
			RoutingPolicy:                   routingPolicy,
			RoutingScorerConfigs:            parsedScorerConfigs,
			TraceLevel:                      traceLevel,
			CounterfactualK:                 counterfactualK,
			SnapshotRefreshInterval:         snapshotRefreshInterval,
			CacheSignalDelay:                cacheSignalDelay,
			PrefillInstances:                prefillInstances,
			DecodeInstances:                 decodeInstances,
			SharedInstances:                 prefillDecodeInstances,
			PDDecider:                       pdDecider,
			PDPrefixThreshold:               pdPrefixThreshold,
			PDTransferBandwidthGBps:         pdTransferBandwidth,
			PDTransferBaseLatencyMs:         pdTransferBaseLatency,
			PDTransferContention:            pdTransferContention,
			PrefillScorerConfigs:            prefillScorerCfgs,
			DecodeScorerConfigs:             decodeScorerCfgs,
			PrefillOverrides:                prefillOverrides,
			DecodeOverrides:                 decodeOverrides,
			FlowControlEnabled:              flowControlEnabled,
			FlowControlDetector:             flowControlDetector,
			FlowControlDispatchOrder:        flowControlDispatchOrder,
			FlowControlMaxQueueDepth:        flowControlMaxQueueDepth,
			FlowControlQueueDepthThreshold:  flowControlQueueDepthThreshold,
			FlowControlKVCacheUtilThreshold: flowControlKVCacheUtilThreshold,
			FlowControlMaxConcurrency:       flowControlMaxConcurrency,
			FlowControlPerBandCapacity:      flowControlPerBandCapacity,
			FlowControlUsageLimitThreshold:  flowControlUsageLimitThreshold,
			FlowControlFairnessPolicy:       flowControlFairnessPolicy,
			TierShedThreshold:               tierShedThreshold,
			TierShedMinPriority:             tierShedMinPriority,
			GAIEQDThreshold:                 gaieQDThreshold,
			GAIEKVThreshold:                 gaieKVThreshold,
			TenantBudgets:                   tenantBudgets,
		}

		// Run simulation — wire SessionManager for closed-loop, nil for fixed mode
		var onRequestDone func(*sim.Request, int64) []*sim.Request
		if sessionMgr != nil {
			onRequestDone = sessionMgr.OnComplete
		}
		cs := cluster.NewClusterSimulator(config, requests, onRequestDone)
		if err := cs.Run(); err != nil {
			logrus.Fatalf("Replay simulation failed: %v", err)
		}

		logrus.Infof("Replay wall-clock time: %.3fs", time.Since(startTime).Seconds())

		// Export trace if requested (BC-1, BC-2, BC-3)
		if replayTraceOutput != "" {
			records := workload.RequestsToTraceRecords(requests)
			header := &workload.TraceHeader{
				Version:  2,
				TimeUnit: "microseconds",
				Mode:     "replayed",
			}
			if err := workload.ExportTraceV2(header, records, replayTraceOutput+".yaml", replayTraceOutput+".csv"); err != nil {
				logrus.Fatalf("Trace export failed: %v", err)
			}
			logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", replayTraceOutput, replayTraceOutput, len(records))
		}

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
			scheduler,
			cs.RoutingRejections(),
		)
		// INV-13 SYNC POINT (metrics): keep in sync with cmd/root.go post-simulation block.
		rawMetrics.PD = cluster.CollectPDMetrics(
			cs.ParentRequests(),
			cs.AggregatedMetrics(),
			cs.PoolMembership(),
			cs.PerInstanceMetricsByID(),
		)
		rawMetrics.ShedByTier = cs.ShedByTier()               // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
		rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth() // Issue #882: gateway queue depth at horizon
		rawMetrics.GatewayQueueShed = cs.GatewayQueueShed()         // Issue #882: gateway queue shed count
		rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected() // Issue #1190: gateway queue rejected count
		rawMetrics.GatewayEvicted = cs.GatewayEvicted()              // Phase 4: in-flight eviction count (#1228)

		if rawMetrics.PD != nil && config.PDTransferContention {
			rawMetrics.PD.PeakConcurrentTransfers = cs.PeakConcurrentTransfers()
			rawMetrics.PD.MeanTransferQueueDepth = cs.MeanTransferQueueDepth()
		}

		// Print anomaly counters if any detected
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueShed > 0 || rawMetrics.GatewayQueueRejected > 0 || rawMetrics.GatewayEvicted > 0 || rawMetrics.TimedOutRequests > 0 {
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
			fmt.Printf("Timed Out Requests: %d\n", rawMetrics.TimedOutRequests)
			fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
			if rawMetrics.GatewayQueueDepth > 0 {
				fmt.Printf("Gateway Queue Depth (horizon): %d\n", rawMetrics.GatewayQueueDepth)
			}
			if rawMetrics.GatewayQueueShed > 0 {
				fmt.Printf("Gateway Queue Shed: %d\n", rawMetrics.GatewayQueueShed)
			}
			if rawMetrics.GatewayQueueRejected > 0 {
				fmt.Printf("Gateway Queue Rejected: %d\n", rawMetrics.GatewayQueueRejected)
			}
			if rawMetrics.GatewayEvicted > 0 {
				fmt.Printf("Gateway Evicted (in-flight): %d\n", rawMetrics.GatewayEvicted)
			}
		}

		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions)

		// Print per-model metrics if requests carry model tags (Phase 1A, FR-011)
		perModelMetrics := cluster.ComputePerModelMetrics(cs.AggregatedMetrics())
		printPerModelMetrics(os.Stdout, perModelMetrics)

		// Print per-tenant fairness metrics if any request carries a tenant label (Phase 1B-2b, FR-010)
		perTenantMetrics := cluster.ComputePerTenantMetrics(cs.AggregatedMetrics())
		printPerTenantMetrics(os.Stdout, perTenantMetrics)

		// Print session metrics if any request carries a session label (#1058)
		sessionMetrics := cluster.ComputeSessionMetrics(cs.AggregatedMetrics())
		printSessionMetrics(os.Stdout, sessionMetrics)

		printPDMetrics(os.Stdout, rawMetrics.PD, config.PDTransferContention)

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
	replayCmd.Flags().StringVar(&replayTraceOutput, "trace-output", "", "Export replay results as TraceV2 files (<prefix>.yaml + <prefix>.csv); header mode is \"replayed\"")
	replayCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", `Session replay mode: "fixed" (pre-baked arrivals from trace) or "closed-loop" (load-adaptive follow-ups via SessionManager)`)
	replayCmd.Flags().IntVar(&replayThinkTimeMs, "think-time-ms", 0, "Override think time between session rounds in milliseconds (0 = derive from trace inter-round arrival gaps; mutually exclusive with --think-time-dist; requires --session-mode closed-loop)")
	replayCmd.Flags().StringVar(&replayThinkTimeDist, "think-time-dist", "", `Think-time distribution spec for closed-loop replay (e.g. "lognormal:mu=2.0,sigma=0.6,min=3s,max=30s" or "constant:value=500ms"). Mutually exclusive with --think-time-ms. Requires --session-mode closed-loop.`)
	rootCmd.AddCommand(replayCmd)
}

// maxInjectedArrivalTimeUs returns the maximum ArrivalTimeUs among records that
// will be injected as initial requests in closed-loop mode: session round-0 records
// and all non-session records. Used to compute the preliminary horizon in O(n)
// without a full LoadTraceV2SessionBlueprints call.
func maxInjectedArrivalTimeUs(trace *workload.TraceV2) int64 {
	var max int64
	for _, rec := range trace.Records {
		if rec.SessionID != "" && rec.RoundIndex != 0 {
			continue // skip follow-up session rounds
		}
		if rec.ArrivalTimeUs > max {
			max = rec.ArrivalTimeUs
		}
	}
	return max
}

// computeHorizonFromMaxArrival maps a maximum arrival time to a simulation horizon.
// - maxArrival > MaxInt64/2 → math.MaxInt64 (overflow guard for 2×)
// - maxArrival <= 0 (all at t=0) → 600,000,000 µs (10 min buffer; MaxInt64 would hang)
// - Otherwise → maxArrival * 2 (generous buffer for last request to complete)
// Used by both the blueprint horizon (closed-loop path) and the simulation horizon so they
// always apply identical logic.
func computeHorizonFromMaxArrival(maxArrival int64) int64 {
	switch {
	case maxArrival > math.MaxInt64/2:
		return math.MaxInt64
	case maxArrival <= 0:
		return 600_000_000
	default:
		return maxArrival * 2
	}
}

// computeReplayHorizon returns the simulation horizon for a trace replay.
// - Empty slice → math.MaxInt64 (no requests, horizon doesn't matter)
// - Otherwise → delegated to computeHorizonFromMaxArrival
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
	return computeHorizonFromMaxArrival(maxArrival)
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
