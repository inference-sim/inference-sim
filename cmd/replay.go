package cmd

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
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
	replayMetricsPath   string // File to write aggregate MetricsOutput JSON (incl. cache_hit_rate, #1583); symmetric with `blis run --metrics-path`
	replaySessionMode   string
	replayThinkTimeMs   int
	replayThinkTimeDist string // distribution spec for think time (e.g. "lognormal:mu=2.0,sigma=0.6,min=3s,max=30s")
	// saturationReport is declared in root.go and shared across run, replay, observe

	replayConcurrentSessions int  // >0 enables fixed-pool closed-loop replay (N concurrent sessions)
	replayTotalSessions      int  // total sessions to replay when pooling (0 = corpus size)
	replayShuffleCorpus      bool // randomize corpus step order (pool mode only; seeded from --seed) (#1480)
)

// corpusShuffleSeedSalt is XORed with the master --seed to derive the corpus-shuffle
// RNG stream, keeping it independent of the per-session token and clone RNGs (#1480).
// Shared by `blis replay --shuffle-corpus` and `blis observe --shuffle-corpus` so the
// SAME --seed selects the SAME subset/order on both — required for calibration parity.
const corpusShuffleSeedSalt = 0x53485546 // "SHUF"

var replayCmd = &cobra.Command{
	Use:   "replay",
	Short: "Replay a TraceV2 file through the discrete-event simulator",
	Long: `Replay takes a TraceV2 file (header YAML + data CSV) and runs the DES against the
exact request sequence captured in the trace. Unlike 'blis run', it does not generate
requests from distributions — the request sequence is fully determined by the trace.

Use --results-path to write per-request SimResult JSON (request_id, ttft_us, e2e_us,
input_tokens, output_tokens, slo_class, model, itl_mean_us) for downstream consumption by blis calibrate.

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
		if replayConcurrentSessions < 0 {
			logrus.Fatalf("--concurrent-sessions must be >= 0, got %d", replayConcurrentSessions)
		}
		if replayTotalSessions < 0 {
			logrus.Fatalf("--total-sessions must be >= 0, got %d", replayTotalSessions)
		}
		// Auto-promote BEFORE the think-time-requires-closed-loop checks below:
		// --concurrent-sessions' help text promises it "implies closed-loop session
		// semantics", so a caller pairing it with --think-time-ms/--think-time-dist
		// but omitting --session-mode must not be fatal'd for missing a mode that
		// this very flag is about to supply.
		if replayConcurrentSessions > 0 && replaySessionMode != "closed-loop" {
			// Pool mode requires closed-loop; promote automatically with a notice.
			logrus.Infof("--concurrent-sessions set; forcing --session-mode closed-loop")
			replaySessionMode = "closed-loop"
		}
		// blis convert otel writes session_context_growth=accumulate and encodes
		// per-round input_tokens as DELTAS, which only reconstruct correctly via
		// the closed-loop accumulate-buffer path. Fixed-mode replay reads
		// input_tokens as absolute per-round counts and never consults
		// SessionContextGrowth, so it would silently misinterpret the deltas and
		// produce wrong-but-plausible-looking metrics. Fail fast instead. This
		// check runs after the auto-promote block above, so a pool run
		// (--concurrent-sessions > 0, already promoted to closed-loop) passes.
		if traceData.Header.SessionContextGrowth == "accumulate" && replaySessionMode != "closed-loop" {
			logrus.Fatalf("trace header has session_context_growth=accumulate (per-round input_tokens are deltas that only reconstruct correctly in closed-loop replay), but --session-mode is %q. Re-run with --session-mode closed-loop, or use --concurrent-sessions N for pooled replay.", replaySessionMode)
		}
		if replayTotalSessions > 0 && replayConcurrentSessions == 0 {
			logrus.Fatalf("--total-sessions requires --concurrent-sessions > 0")
		}
		// --shuffle-corpus is a pool-mode concept (it randomizes the pool's step
		// order). Plain closed-loop replay injects every session at its recorded
		// arrival, so there is no step order to randomize — fail loudly (R1), never
		// a silent no-op.
		if replayShuffleCorpus && replayConcurrentSessions == 0 {
			logrus.Fatalf("--shuffle-corpus requires --concurrent-sessions > 0")
		}
		// Pool mode + output artifacts: guard against silent truncation (R1).
		// Trace re-export in pool mode would contain only the initial wave of
		// round-0 requests (follow-ups/clones are never collected for export), so
		// reject it outright — mirroring how replay fails fast on unsupported
		// features (INV-13). Per-request results similarly exclude clone sessions
		// (non-numeric ids), so warn loudly rather than silently subsetting.
		if replayConcurrentSessions > 0 && replayTraceOutput != "" {
			logrus.Fatalf("--trace-output is not supported with --concurrent-sessions (pool mode): a re-export would capture only the initial %d-session wave, not all pooled sessions. Re-run without --trace-output.", replayConcurrentSessions)
		}
		if replayConcurrentSessions > 0 && resultsPath != "" {
			logrus.Warnf("--results-path with --concurrent-sessions (pool mode): per-request results cover only the original corpus sessions' round-0; duplicated (clone) sessions and follow-up rounds are excluded (non-numeric request ids).")
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
		var poolDriver *workload.SessionPoolDriver
		if replaySessionMode == "closed-loop" {
			// Closed-loop: inject only round-0 requests; SessionManager drives follow-ups.
			// Compute the preliminary horizon from trace records directly (O(n)) so we can
			// call LoadTraceV2SessionBlueprints exactly once with correct parameters.
			replayHorizonPrelim := computeHorizonFromMaxArrival(workload.MaxNormalizedInjectionTimeUs(traceData))
			if cmd.Flags().Changed("horizon") {
				replayHorizonPrelim = simulationHorizon
			}
			// Pool mode drains on session count, not wall-clock. Unless the user set an
			// explicit --horizon cap, run the blueprints with an unbounded horizon so no
			// session is horizon-interrupted mid-drain (INV-11 / conservation).
			if replayConcurrentSessions > 0 && !cmd.Flags().Changed("horizon") {
				replayHorizonPrelim = math.MaxInt64
			}
			r0Requests, blueprints, bErr := workload.LoadTraceV2SessionBlueprints(traceData, seed, thinkTimeSampler, replayHorizonPrelim)
			if bErr != nil {
				logrus.Fatalf("Failed to build session blueprints from trace: %v", bErr)
			}
			if len(blueprints) == 0 {
				logrus.Warnf("--session-mode closed-loop: no session records found in trace; all requests injected with fixed timing")
				requests = r0Requests
			} else if replayConcurrentSessions > 0 {
				// Pool mode requires a pure-session corpus: every record must belong to
				// a session (carry a session_id). LoadTraceV2SessionBlueprints returns one
				// round-0 request per session PLUS one per non-session (single-shot)
				// record, so a mixed trace yields more round-0 requests than blueprints.
				// Surface that here with an actionable message naming the offending count,
				// rather than letting BuildSessionPool fail with its internal
				// "count mismatch" wording (R1 — no opaque internal-invariant error).
				if nonSession := len(r0Requests) - len(blueprints); nonSession > 0 {
					logrus.Fatalf("--concurrent-sessions requires every trace record to belong to a session, but %d of %d records have no session_id. Pooled replay cannot mix session and non-session (single-shot) records; re-export the corpus so every row carries a session_id (e.g. via `blis convert otel`), or drop --concurrent-sessions to replay it in plain closed-loop mode.", nonSession, len(r0Requests))
				}
				// Optional seeded shuffle of the corpus step order (#1480). Drawn from
				// a distinct stream off the master seed (XOR salt) so it is reproducible
				// from --seed yet does not perturb the per-session token / clone RNGs.
				if replayShuffleCorpus {
					workload.ShuffleSessions(blueprints, r0Requests, rand.New(rand.NewSource(seed^corpusShuffleSeedSalt)))
				}
				driver, initial, pErr := workload.BuildSessionPool(blueprints, r0Requests, replayConcurrentSessions, replayTotalSessions, seed)
				if pErr != nil {
					logrus.Fatalf("Failed to build session pool: %v", pErr)
				}
				poolDriver = driver
				requests = initial
				logrus.Infof("Session-pool mode: pool=%d total=%d, %d round-0 requests injected initially",
					replayConcurrentSessions, driver.TotalSessions(), len(initial))
			} else {
				requests = r0Requests
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
		if replayConcurrentSessions > 0 && !cmd.Flags().Changed("horizon") {
			replayHorizon = math.MaxInt64 // self-draining pool; cluster horizon unbounded
		}
		logrus.Infof("Simulation horizon: %d ticks", replayHorizon)

		// LoRA control-plane (#1464): resolve ONCE (R4) so the KV auto-capacity path
		// (resolveLatencyConfig + per-pool calc) subtracts the same static HBM
		// reservation runCmd does (PR5 / INV-13 parity) and the SimConfig below reuses
		// it. Reservation is 0 when the subsystem is inert (INV-6). Set before
		// resolveLatencyConfig.
		loraCfg := resolveLoRAConfig(cmd)
		loraReservedBytesForKV = adapterReservedBytesFor(loraCfg)

		// KV-cache offload config (#1587, BC-G6): the trace header is authoritative on
		// replay (unlike --lora-config, which is flags-only). Reconstruct the recorded
		// config, Fatalf if this binary cannot reproduce it, and Fatalf on a genuine
		// flag/header conflict. Inert when the trace carries no offload config (BC-G5).
		// Observe traces (mode "real") may model the observed deployment's offload via
		// --kv-offload-config for the sim side of the #1583 hit-rate comparison; sim-
		// generated traces remain header-authoritative (INV-13).
		kvOffloadCfg := reconcileReplayKVOffload(cmd, traceData.Header.KVOffload, traceData.Header.Mode == "real")
		// #1590 (H1): parity with run — the multi-tier offload chain (from the trace
		// header or --kv-offload-config) and the legacy --kv-cpu-blocks single CPU tier
		// are distinct offload models; refuse both with a clean CLI error rather than
		// the library-level panic in NewKVStore.
		if kvOffloadCfg.IsEnabled() && kvCPUBlocks > 0 {
			logrus.Fatalf("--kv-cpu-blocks conflicts with the trace's kv_offload config (distinct KV-offload models); set only one")
		}
		// #1583 (BC-10): a tiered observed hit-rate in the header is only replayable
		// into a comparable sim number if this replay reproduces a tiered offload
		// config. Fail loudly rather than silently produce a GPU-only hit-rate.
		if err := validateObservedKVReplayable(traceData.Header.ObservedKVMetrics, kvOffloadCfg.IsEnabled()); err != nil {
			logrus.Fatalf("%v", err)
		}

		// Resolve latency backend configuration (single code path shared with runCmd).
		lr := resolveLatencyConfig(cmd)

		// #1583: derive PerBlockBytes for an offload config supplied by --kv-offload-config
		// (the observe-trace calibration path). A sim-generated trace header already
		// carries the run-computed value (>0), so this is skipped for run/replay parity;
		// a flag-supplied config has PerBlockBytes==0 until the model resolves. Mirrors
		// the derivation in cmd/root.go (runCmd) so run and replay agree.
		if kvOffloadCfg.IsEnabled() && kvOffloadCfg.PerBlockBytes == 0 {
			perTokenKVBytes, pbErr := latency.KVBytesPerToken(lr.ModelConfig, tensorParallelism)
			if pbErr != nil {
				logrus.Fatalf("kv_offload: cannot derive per_block_bytes from the model: %v", pbErr)
			}
			kvOffloadCfg.PerBlockBytes = int64(perTokenKVBytes * float64(kvOffloadCfg.BlockSize))
			if kvOffloadCfg.PerBlockBytes <= 0 {
				logrus.Fatalf("kv_offload: derived per_block_bytes must be > 0 (KVBytesPerToken=%v × block_size=%d)", perTokenKVBytes, kvOffloadCfg.BlockSize)
			}
		}

		// Numeric flag validation (same as runCmd)
		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}
		if totalKVBlocks <= 0 {
			logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
		}
		if maxNumSeqs <= 0 {
			logrus.Fatalf("--max-num-seqs must be > 0, got %d", maxNumSeqs)
		}
		if maxNumBatchedTokens <= 0 {
			logrus.Fatalf("--max-num-batched-tokens must be > 0, got %d", maxNumBatchedTokens)
		}
		if longPrefillTokenThreshold < 0 {
			logrus.Fatalf("--long-prefill-token-threshold must be >= 0, got %d", longPrefillTokenThreshold)
		}
		if cmd.Flags().Changed("horizon") && replayHorizon <= 0 {
			logrus.Fatalf("--horizon must be > 0, got %d", replayHorizon)
		}

		// E/P/D validation (GAP-4, issue #1264). Per INV-13 (run/replay parity,
		// PR #1305), encode pool flags are supported in replay the same way PD
		// disaggregation is: validate the decider name and the encode-instances /
		// encode-decider pairing, then wire them into DeploymentConfig below.
		if encodeInstances < 0 {
			logrus.Fatalf("--encode-instances must be >= 0, got %d", encodeInstances)
		}
		if !sim.IsValidEncodeDecider(encodeDecider) {
			logrus.Fatalf("Unknown encode decider %q. Valid: %s", encodeDecider, strings.Join(sim.ValidEncodeDeciderNames(), ", "))
		}
		if encodeDecider != "" && encodeDecider != "never" && encodeInstances == 0 {
			logrus.Fatalf("--encode-decider=%q requires --encode-instances > 0 (the encode pool is disabled)", encodeDecider)
		}
		if encodeInstances > 0 && (encodeDecider == "" || encodeDecider == "never") {
			logrus.Warnf("--encode-decider=%q has no effect because --encode-instances=%d but the decider never encodes; set --encode-decider=multimodal or always to activate the encode pool", encodeDecider, encodeInstances)
		}

		// Resolve policy configuration (single code path shared with runCmd).
		// Autoscaler and node-pool configs are not supported in replay — fail fast
		// rather than silently producing divergent results (INV-13, Track B).
		parsedScorerConfigs, bundle := resolvePolicies(cmd)
		if cmd.Flags().Changed("model-autoscaler-interval-us") {
			logrus.Fatalf("--model-autoscaler-interval-us is not supported in blis replay; remove this flag or use blis run instead")
		}
		var bundleInstanceLifecycle cluster.InstanceLifecycleConfig
		if bundle != nil {
			if bundle.Autoscaler.IntervalUs > 0 {
				logrus.Fatalf("blis replay does not support autoscaler config (policy bundle interval_us=%g); remove the autoscaler section from the policy bundle or use blis run instead", bundle.Autoscaler.IntervalUs)
			}
			if len(bundle.NodePools) > 0 {
				logrus.Fatalf("blis replay does not support node_pools config (%d pool(s) in policy bundle); remove the node_pools section from the policy bundle or use blis run instead", len(bundle.NodePools))
			}
			bundleInstanceLifecycle = cluster.InstanceLifecycleConfig{
				LoadingDelay: cluster.DelaySpec{
					Mean:   bundle.InstanceLifecycle.LoadingDelay.Mean,
					Stddev: bundle.InstanceLifecycle.LoadingDelay.Stddev,
				},
				WarmStartInitialInstances: bundle.InstanceLifecycle.WarmStartInitialInstances,
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
		if err := cluster.ValidatePoolTopology(prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances, numInstances); err != nil {
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

		// ModelConfig resolution for PD KV transfer sizing (same as runCmd).
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

		// Per-pool KV auto-calculation (same as runCmd).
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
							// Per-pool TP but GLOBAL dp: per-pool DP is out of scope (#1420);
							// --dp applies uniformly to all pools. Mirrors run (cmd/root.go).
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolPrefillTP, dataParallelism, blockSizeTokens, gpuMemoryUtilization, kvParamsPool,
								latency.WithAdapterReservedBytes(loraReservedBytesForKV))
							if calcErr != nil {
								logrus.Fatalf("--prefill-tp/--prefill-hardware: KV capacity auto-calculation failed for prefill pool: %v", calcErr)
							} else {
								prefillOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--prefill-tp/--prefill-hardware: auto-calculated prefill pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, DP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolPrefillTP, dataParallelism)
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
							// Per-pool TP, global dp (see prefill-pool note above; #1420).
							poolBlocks, calcErr := latency.CalculateKVBlocks(lr.ModelConfig, poolHC, poolDecodeTP, dataParallelism, blockSizeTokens, gpuMemoryUtilization, kvParamsPool,
								latency.WithAdapterReservedBytes(loraReservedBytesForKV))
							if calcErr != nil {
								logrus.Fatalf("--decode-tp/--decode-hardware: KV capacity auto-calculation failed for decode pool: %v", calcErr)
							} else {
								decodeOverrides.TotalKVBlocks = &poolBlocks
								logrus.Infof("--decode-tp/--decode-hardware: auto-calculated decode pool total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, DP=%d)",
									poolBlocks, poolHC.MemoryGiB, poolDecodeTP, dataParallelism)
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
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency,
					sim.WithKVOffload(kvOffloadCfg)),
				BatchConfig:          sim.NewBatchConfig(maxNumSeqs, maxNumBatchedTokens, longPrefillTokenThreshold),
				LatencyCoeffs:        sim.NewLatencyCoeffs(lr.BetaCoeffs, lr.AlphaCoeffs),
				ModelHardwareConfig:  sim.NewModelHardwareConfig(lr.ModelConfig, lr.HWConfig, model, gpu, tensorParallelism, dataParallelism, enableExpertParallel, moeCommBackend, lr.Backend, maxModelLen),
				PolicyConfig:         sim.NewPolicyConfig(scheduler, preemptionPolicy),
				LoRAConfig:           loraCfg,
				SpeculativeConfig:    resolveSpeculativeConfig(cmd),
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
			EncodeInstances:                 encodeInstances,
			EncodeDecider:                   encodeDecider,
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
			FlowControlSLOTargets:           sloTargetsMap,
			FlowControlMaxQueueDepth:        flowControlMaxQueueDepth,
			FlowControlQueueDepthThreshold:  flowControlQueueDepthThreshold,
			FlowControlKVCacheUtilThreshold: flowControlKVCacheUtilThreshold,
			FlowControlMaxConcurrency:       flowControlMaxConcurrency,
			FlowControlPerBandCapacity:      flowControlPerBandCapacity,
			FlowControlUsageLimitThreshold:  flowControlUsageLimitThreshold,
			FlowControlFairnessPolicy:       flowControlFairnessPolicy,
			FlowControlRequestTTL:           flowControlRequestTTL,
			FlowControlQueueShedding:        flowControlQueueShedding,
			FlowControlDispatchTickInterval: flowControlDispatchTickInterval,
			FlowControlInFlightEviction:     flowControlInFlightEviction,
			TierShedThreshold:               tierShedThreshold,
			TierShedMinPriority:             tierShedMinPriority,
			GAIEQDThreshold:                 gaieQDThreshold,
			GAIEKVThreshold:                 gaieKVThreshold,
			TenantBudgets:                   tenantBudgets,
			InstanceLifecycle:               bundleInstanceLifecycle,
		}

		// Run simulation — wire SessionManager for closed-loop, nil for fixed mode
		// Collect follow-ups for saturation analysis in closed-loop mode (BC-12, issue #1298)
		var followUpRequests []*sim.Request
		var onRequestDone func(*sim.Request, int64) []*sim.Request
		switch {
		case poolDriver != nil:
			baseCb := poolDriver.OnComplete
			onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
				followUps := baseCb(req, clock)
				followUpRequests = append(followUpRequests, followUps...)
				return followUps
			}
		case sessionMgr != nil:
			baseCb := sessionMgr.OnComplete
			onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
				followUps := baseCb(req, clock)
				followUpRequests = append(followUpRequests, followUps...)
				return followUps
			}
		}
		cs := cluster.NewClusterSimulator(config, cluster.NewSliceRequestSource(requests), onRequestDone)

		// Resolve the saturation tracer BEFORE the run so a bad flag / config /
		// report path fails fast (#1516 single detector, #1519 bank).
		satTracer, satErr := resolveSaturation()
		if satErr != nil {
			logrus.Fatalf("%v", satErr)
		}

		if err := cs.Run(); err != nil {
			logrus.Fatalf("Replay simulation failed: %v", err)
		}
		if poolDriver != nil {
			// KNOWN LIMITATION (follow-up): under an explicit --horizon hard cap that
			// truncates mid-drain, a refill pushed by OnComplete but discarded by the
			// cluster's horizon guard is still counted as started, so Unstarted() may
			// undercount dropped sessions and this warning may not fire. The
			// self-draining path (no --horizon) is exact. Tracked in #1483.
			if un := poolDriver.Unstarted(); un > 0 {
				logrus.Warnf("%d of %d pooled sessions never admitted — a --horizon cap truncated the drain, and/or admitted sessions were dropped before reaching an instance (routing/gateway rejection, which does not fire the per-instance completion hook). Omit --horizon to self-drain; check routing-rejection metrics for the second cause.",
					un, poolDriver.TotalSessions())
			}
		}

		logrus.Infof("Replay wall-clock time: %.3fs", time.Since(startTime).Seconds())

		// Resolve goodput SLO targets early so the re-export header carries them (#1413, BC-7).
		// Replay precedence: CLI > trace header. (No workload spec in replay path.)
		cliTTFT, cliITL, cliE2E, gpErr := resolveGoodputCLIFlags(goodputSLOTTFT, goodputSLOITL, goodputSLOE2E)
		if gpErr != nil {
			logrus.Fatalf("%v", gpErr)
		}
		goodputTargets := mergeGoodputTargets(cliTTFT, cliITL, cliE2E, traceData.Header.GoodputSLOTargets, nil)

		// Export trace if requested (BC-1, BC-2, BC-3)
		if replayTraceOutput != "" {
			records := workload.RequestsToTraceRecords(requests)
			header := &workload.TraceHeader{
				Version:           3,
				TimeUnit:          "microseconds",
				Mode:              "replayed",
				GoodputSLOTargets: goodputTargets,                   // #1413, BC-7
				KVOffload:         simToHeaderOffload(kvOffloadCfg), // #1587, BC-G6: carry the offload config forward
				// #1583, BC-4: preserve the real-side observed KV hit-rate verbatim.
				// It is a real-server observation replay cannot regenerate; carrying it
				// forward keeps it available to a downstream calibrate and never
				// silently drops it.
				ObservedKVMetrics: traceData.Header.ObservedKVMetrics,
			}
			if err := workload.ExportTraceV2(header, records, replayTraceOutput+".yaml", replayTraceOutput+".csv"); err != nil {
				logrus.Fatalf("Trace export failed: %v", err)
			}
			logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", replayTraceOutput, replayTraceOutput, len(records))
		}

		// Save aggregate metrics to stdout (same as runCmd). Per-instance output
		// carries no saturation field — the final label is a cluster-level verdict
		// emitted on the aggregate below (#1517).
		if numInstances > 1 {
			for _, inst := range cs.Instances() {
				if err := inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, ""); err != nil {
					logrus.Fatalf("SaveResults for instance %s: %v", inst.ID(), err)
				}
			}
		}
		// Save aggregate (always print to stdout; SimResult output uses separate file)
		// goodputTargets resolved above for trace re-export; reused here (#1413, BC-1, BC-4).
		// The saturation reducer runs BEFORE EmitOutput and mutates clusterOutput.Saturation
		// (#1517), mirroring goodput's build-then-mutate-then-emit pattern.
		aggregated := cs.AggregatedMetrics()
		clusterOutput := aggregated.BuildOutput("cluster")
		emitGoodput(&clusterOutput, aggregated, cs.InjectedByClass(),
			float64(aggregated.SimEndedTime)/1e6, goodputTargets)

		// Saturation (#1516 single detector / #1519 bank / #1517 final label): same
		// pipeline as run/observe, sim-derived input. run → replay of the same trace
		// is byte-identical (INV-13). Guard on the tracer so the common no-detector
		// path skips the O(n log n) sort + O(n) copy in CompletedRequestMetrics().
		if satTracer != nil {
			final, err := satTracer.run(aggregated.CompletedRequestMetrics())
			if err != nil {
				logrus.Fatalf("Saturation: %v", err)
			}
			if len(final) > 0 {
				clusterOutput.Saturation = final
			}
		}

		// --metrics-path (#1583): write the aggregate MetricsOutput JSON (which gains
		// the file-only cache_hit_rate) so `blis calibrate --sim-metrics` can read the
		// simulator's hit-rate. Empty path → stdout only, byte-identical to before
		// (BC-8). run and replay --metrics-path of the same trace produce identical
		// cache_hit_rate (INV-13).
		if err := aggregated.EmitOutput(clusterOutput, replayMetricsPath); err != nil {
			logrus.Fatalf("SaveResults: %v", err)
		}

		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			scheduler,
			cs.RoutingRejections(),
			cs.EncodeRoutingRejections(),
			cs.InjectedByClass(),
		)
		// INV-13 SYNC POINT (metrics): keep in sync with cmd/root.go post-simulation block.
		rawMetrics.PD = cluster.CollectPDMetrics(
			cs.ParentRequests(),
			cs.AggregatedMetrics(),
			cs.PoolMembership(),
			cs.PerInstanceMetricsByID(),
		)
		rawMetrics.ShedByTier = cs.ShedByTier()                     // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
		rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth()       // Issue #882: gateway queue depth at horizon
		rawMetrics.GatewayQueueShed = cs.GatewayQueueShed()         // Issue #882: gateway queue shed count
		rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected() // Issue #1190: gateway queue rejected count
		rawMetrics.GatewayEvicted = cs.GatewayEvicted()             // Phase 4: in-flight eviction count (#1228)
		rawMetrics.GatewayExpired = cs.GatewayExpired()             // Phase 6: TTL expiration count (#1193)

		if rawMetrics.PD != nil && config.PDTransferContention {
			rawMetrics.PD.PeakConcurrentTransfers = cs.PeakConcurrentTransfers()
			rawMetrics.PD.MeanTransferQueueDepth = cs.MeanTransferQueueDepth()
		}

		// Print anomaly counters if any detected
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueShed > 0 || rawMetrics.GatewayQueueRejected > 0 || rawMetrics.GatewayEvicted > 0 || rawMetrics.GatewayExpired > 0 || rawMetrics.EncodeRoutingRejections > 0 || rawMetrics.TimedOutRequests > 0 {
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
			if rawMetrics.GatewayExpired > 0 {
				fmt.Printf("Gateway Expired (TTL): %d\n", rawMetrics.GatewayExpired)
			}
			if rawMetrics.EncodeRoutingRejections > 0 {
				fmt.Printf("Encode Routing Rejections: %d\n", rawMetrics.EncodeRoutingRejections)
			}
		}

		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions, len(goodputTargets) > 0)

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
	replayCmd.Flags().StringVar(&resultsPath, "results-path", "", "File to write []SimResult JSON (request_id, ttft_us, e2e_us, input_tokens, output_tokens, slo_class, model, itl_mean_us) for blis calibrate consumption.")
	replayCmd.Flags().StringVar(&replayTraceOutput, "trace-output", "", "Export replay results as TraceV2 files (<prefix>.yaml + <prefix>.csv); header mode is \"replayed\"")
	replayCmd.Flags().StringVar(&replayMetricsPath, "metrics-path", "", "File to write aggregate MetricsOutput JSON (incl. cache_hit_rate for `blis calibrate --sim-metrics`, #1583). Symmetric with `blis run --metrics-path`; stdout is unaffected.")

	// Saturation trace flags (#1516): --detectors + --saturation-config + --saturation-report.
	registerDetectorFlags(replayCmd)

	replayCmd.Flags().StringVar(&replaySessionMode, "session-mode", "fixed", `Session replay mode: "fixed" (pre-baked arrivals from trace) or "closed-loop" (load-adaptive follow-ups via SessionManager)`)
	replayCmd.Flags().IntVar(&replayThinkTimeMs, "think-time-ms", 0, "Override think time between session rounds in milliseconds (0 = derive from trace inter-round arrival gaps; mutually exclusive with --think-time-dist; requires --session-mode closed-loop)")
	replayCmd.Flags().StringVar(&replayThinkTimeDist, "think-time-dist", "", `Think-time distribution spec for closed-loop replay (e.g. "lognormal:mu=2.0,sigma=0.6,min=3s,max=30s" or "constant:value=500ms"). Mutually exclusive with --think-time-ms. Requires --session-mode closed-loop.`)
	replayCmd.Flags().IntVar(&replayConcurrentSessions, "concurrent-sessions", 0, "Replay a fixed pool of N concurrent closed-loop sessions drawn from the trace corpus (0 = disabled). Implies closed-loop session semantics.")
	replayCmd.Flags().IntVar(&replayTotalSessions, "total-sessions", 0, "Total sessions to replay under --concurrent-sessions; duplicates the corpus (with cache-busting) to fill. 0 = replay each corpus session once.")
	replayCmd.Flags().BoolVar(&replayShuffleCorpus, "shuffle-corpus", false, "Randomize the corpus step/admission order (seeded from --seed for reproducibility). Requires --concurrent-sessions > 0. With --total-sessions < corpus this yields a random subset; every session still runs otherwise.")
	replayCmd.Flags().StringVar(&goodputSLOTTFT, "slo-ttft", "", "Per-class TTFT goodput thresholds (e.g. \"critical=100ms,standard=500ms\"). Precedence: CLI > trace header > workload spec.")
	replayCmd.Flags().StringVar(&goodputSLOITL, "slo-itl", "", "Per-class mean ITL goodput thresholds (e.g. \"critical=50ms,standard=150ms\").")
	replayCmd.Flags().StringVar(&goodputSLOE2E, "slo-e2e", "", "Per-class E2E goodput thresholds (e.g. \"critical=5s,standard=30s\").")
	// --lazy-generation: accepted for CLI symmetry with `blis run` (#1441),
	// but ignored — replay reads requests from a captured trace and never
	// invokes the workload generator. The flag binds to a throwaway local
	// so no global state is mutated. BC-9.
	var replayLazyGenerationIgnored bool
	replayCmd.Flags().BoolVar(&replayLazyGenerationIgnored, "lazy-generation", false, "Accepted for symmetry with `blis run` (#1441); has no effect on replay.")
	rootCmd.AddCommand(replayCmd)
}

// computeHorizonFromMaxArrival maps a maximum injection time to a simulation
// horizon. The argument is the largest injected-request tick on the sim clock:
// the max ArrivalTime for fixed mode (computeReplayHorizon) or the normalized
// injection for closed-loop mode (workload.MaxNormalizedInjectionTimeUs) — both
// already re-based onto the arrival origin (#1606). (The parameter is named
// maxArrival for historical reasons; it is generic over any int64 tick.)
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
			SLOClass:     rm.SLOClass,
			Model:        rm.Model,
			ITLMeanUs:    m.RequestITLs[reqID], // already in ticks (µs), same as TTFT/E2E; 0 if not computed
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
	if len(m.RequestITLs) == 0 {
		logrus.Debugf("extractSimResults: RequestITLs is empty (no completed requests); ITLMeanUs will be 0 for all entries")
	}
	// Sort by RequestID for deterministic JSON output (R2, INV-6)
	sort.Slice(results, func(i, j int) bool {
		return results[i].RequestID < results[j].RequestID
	})
	return results
}
