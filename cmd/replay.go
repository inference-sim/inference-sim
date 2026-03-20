package cmd

import (
	"bytes"
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
	"gopkg.in/yaml.v3"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/latency"
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

		// Local copies of coefficient slices to avoid mutating package-level vars
		// (same shadowing as runCmd.Run — required when loading from defaults.yaml)
		alphaCoeffs, betaCoeffs := alphaCoeffs, betaCoeffs

		// Normalize model name (same as runCmd)
		model = strings.ToLower(model)

		// Validate --latency-model (BC-4)
		if !sim.IsValidLatencyBackend(latencyModelBackend) {
			logrus.Fatalf("unknown --latency-model %q; valid options: %s",
				latencyModelBackend, strings.Join(sim.ValidLatencyBackendNames(), ", "))
		}
		backend := latencyModelBackend

		// Alpha/beta coefficient validation (same as runCmd)
		alphaChanged := cmd.Flags().Changed("alpha-coeffs")
		betaChanged := cmd.Flags().Changed("beta-coeffs")
		if alphaChanged != betaChanged {
			if alphaChanged {
				logrus.Fatalf("--alpha-coeffs requires --beta-coeffs. Both coefficient sets are needed for blackbox mode")
			}
			logrus.Fatalf("--beta-coeffs requires --alpha-coeffs. Both coefficient sets are needed for blackbox mode")
		}
		for i, c := range alphaCoeffs {
			if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
				logrus.Fatalf("--alpha-coeffs[%d] must be a finite non-negative number, got %v", i, c)
			}
		}
		for i, c := range betaCoeffs {
			if math.IsNaN(c) || math.IsInf(c, 0) || c < 0 {
				logrus.Fatalf("--beta-coeffs[%d] must be a finite non-negative number, got %v", i, c)
			}
		}
		if !cmd.Flags().Changed("latency-model") && alphaChanged && betaChanged {
			backend = "blackbox"
			logrus.Infof("--alpha-coeffs and --beta-coeffs provided; using blackbox mode")
		}

		var modelConfig = sim.ModelConfig{}
		var hwConfig = sim.HardwareCalib{}

		// Early defaults resolution (same as runCmd)
		if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
			hardware, tp, version := GetDefaultSpecs(model)
			if tensorParallelism == 0 && tp > 0 {
				logrus.Warnf("Finding default values of TP for model=%v", model)
				logrus.Warnf("Using default tp=%v", tp)
				tensorParallelism = tp
			}
			if gpu == "" && len(hardware) > 0 {
				logrus.Warnf("Finding default values of hardware for model=%v", model)
				logrus.Warnf("Using default GPU=%v", hardware)
				gpu = hardware
			}
			if vllmVersion == "" && len(version) > 0 {
				logrus.Warnf("Finding default values of vLLM version for model=%v", model)
				logrus.Warnf("Using default vLLM version=%v", version)
				vllmVersion = version
			}
		}
		kvBlocksFromDefaults := false

		// Latency model backend resolution
		// IMPORTANT: Keep this block in sync with runCmd.Run when modifying.
		if backend == "roofline" {
			var missing []string
			if gpu == "" {
				missing = append(missing, "--hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				missing = append(missing, "--tp (tensor parallelism)")
			}
			if len(missing) > 0 {
				logrus.Fatalf("Roofline mode (the default) requires %s. No defaults found in defaults.yaml for model=%s. Provide these flags explicitly, or use --latency-model blackbox for offline coefficient-based estimation",
					strings.Join(missing, " and "), model)
			}
			alphaChanged2 := cmd.Flags().Changed("alpha-coeffs")
			betaChanged2 := cmd.Flags().Changed("beta-coeffs")
			if cmd.Flags().Changed("latency-model") && (alphaChanged2 || betaChanged2) {
				logrus.Fatalf("--alpha-coeffs/--beta-coeffs cannot be used with --latency-model roofline. Roofline computes step time analytically. Use --latency-model blackbox if you want coefficient-based estimation")
			}
			if modelConfigFolder != "" {
				logrus.Infof("--latency-model: explicit --model-config-folder takes precedence over auto-resolution")
			}
			if hwConfigPath != "" {
				logrus.Infof("--latency-model: explicit --hardware-config takes precedence over auto-resolution")
			}
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
		}

		if backend == "crossmodel" {
			var missing []string
			if gpu == "" {
				missing = append(missing, "--hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				missing = append(missing, "--tp (tensor parallelism)")
			}
			if len(missing) > 0 {
				logrus.Fatalf("--latency-model crossmodel requires %s. No defaults found in defaults.yaml for model=%s. Provide these flags explicitly",
					strings.Join(missing, " and "), model)
			}
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				data, readErr := os.ReadFile(defaultsFilePath)
				if readErr != nil {
					logrus.Warnf("--latency-model crossmodel: failed to read %s: %v", defaultsFilePath, readErr)
				} else {
					var cfg Config
					decoder := yaml.NewDecoder(bytes.NewReader(data))
					decoder.KnownFields(true)
					if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
						logrus.Fatalf("--latency-model crossmodel: failed to parse %s: %v", defaultsFilePath, yamlErr)
					}
					if cfg.CrossModelDefaults != nil {
						if !cmd.Flags().Changed("beta-coeffs") {
							betaCoeffs = cfg.CrossModelDefaults.BetaCoeffs
							logrus.Infof("--latency-model: loaded crossmodel beta coefficients from defaults.yaml")
						}
						if !cmd.Flags().Changed("alpha-coeffs") {
							alphaCoeffs = cfg.CrossModelDefaults.AlphaCoeffs
							logrus.Infof("--latency-model: loaded crossmodel alpha coefficients from defaults.yaml")
						}
					}
				}
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
			if !cmd.Flags().Changed("beta-coeffs") && (len(betaCoeffs) < 4 || allZeros(betaCoeffs)) {
				logrus.Fatalf("--latency-model crossmodel: no crossmodel_defaults found in %s and no --beta-coeffs provided. Add crossmodel_defaults to defaults.yaml or provide --beta-coeffs explicitly", defaultsFilePath)
			}
		}

		if backend == "trained-roofline" {
			var missing []string
			if gpu == "" {
				missing = append(missing, "--hardware (GPU type)")
			}
			if tensorParallelism <= 0 {
				missing = append(missing, "--tp (tensor parallelism)")
			}
			if len(missing) > 0 {
				logrus.Fatalf("--latency-model trained-roofline requires %s. No defaults found in defaults.yaml for model=%s. Provide these flags explicitly",
					strings.Join(missing, " and "), model)
			}
			resolved, err := resolveModelConfig(model, modelConfigFolder, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			modelConfigFolder = resolved
			resolvedHW, err := resolveHardwareConfig(hwConfigPath, defaultsFilePath)
			if err != nil {
				logrus.Fatalf("%v", err)
			}
			hwConfigPath = resolvedHW
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				data, readErr := os.ReadFile(defaultsFilePath)
				if readErr != nil {
					logrus.Warnf("--latency-model trained-roofline: failed to read %s: %v", defaultsFilePath, readErr)
				} else {
					var cfg Config
					decoder := yaml.NewDecoder(bytes.NewReader(data))
					decoder.KnownFields(true)
					if yamlErr := decoder.Decode(&cfg); yamlErr != nil {
						logrus.Fatalf("--latency-model trained-roofline: failed to parse %s: %v", defaultsFilePath, yamlErr)
					}
					if cfg.TrainedRooflineDefaults != nil {
						if !cmd.Flags().Changed("beta-coeffs") {
							betaCoeffs = cfg.TrainedRooflineDefaults.BetaCoeffs
							logrus.Infof("--latency-model: loaded trained-roofline beta coefficients from defaults.yaml")
						}
						if !cmd.Flags().Changed("alpha-coeffs") {
							alphaCoeffs = cfg.TrainedRooflineDefaults.AlphaCoeffs
							logrus.Infof("--latency-model: loaded trained-roofline alpha coefficients from defaults.yaml")
						}
					}
				}
				_, _, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model: loaded total-kv-blocks=%d from defaults.yaml", kvBlocks)
				}
			}
			if !cmd.Flags().Changed("beta-coeffs") && (len(betaCoeffs) < 10 || allZeros(betaCoeffs)) {
				logrus.Fatalf("--latency-model trained-roofline: no trained_roofline_defaults found in %s and no --beta-coeffs provided.", defaultsFilePath)
			}
		}

		if backend == "blackbox" {
			if _, statErr := os.Stat(defaultsFilePath); statErr == nil {
				alpha, beta, kvBlocks := GetCoefficients(model, tensorParallelism, gpu, vllmVersion, defaultsFilePath)
				if !cmd.Flags().Changed("alpha-coeffs") && len(alpha) > 0 {
					alphaCoeffs = alpha
					logrus.Infof("--latency-model: loaded alpha coefficients from defaults.yaml")
				}
				if !cmd.Flags().Changed("beta-coeffs") && len(beta) > 0 {
					betaCoeffs = beta
					logrus.Infof("--latency-model: loaded beta coefficients from defaults.yaml")
				}
				if !cmd.Flags().Changed("total-kv-blocks") && kvBlocks > 0 {
					totalKVBlocks = kvBlocks
					kvBlocksFromDefaults = true
					logrus.Infof("--latency-model blackbox: auto-calculated total-kv-blocks=%d from cached model config", kvBlocks)
				}
			}
		}
		if backend == "blackbox" && allZeros(alphaCoeffs) && allZeros(betaCoeffs) {
			logrus.Fatalf("No trained coefficients found for model=%s, GPU=%s, TP=%d. Provide --alpha-coeffs/--beta-coeffs, use --latency-model roofline, crossmodel, or trained-roofline",
				model, gpu, tensorParallelism)
		}

		if backend == "roofline" || backend == "crossmodel" || backend == "trained-roofline" {
			hfPath := filepath.Join(modelConfigFolder, "config.json")
			hfConfig, err := latency.ParseHFConfig(hfPath)
			if err != nil {
				logrus.Fatalf("Failed to parse HuggingFace config: %v", err)
			}
			mc, err := latency.GetModelConfigFromHF(hfConfig)
			if err != nil {
				logrus.Fatalf("Failed to load model config: %v", err)
			}
			modelConfig = *mc
			hc, err := latency.GetHWConfig(hwConfigPath, gpu)
			if err != nil {
				logrus.Fatalf("Failed to load hardware config: %v", err)
			}
			hwConfig = hc

			applyWeightPrecisionFallback(&modelConfig, model, hfConfig.Raw)

			if backend == "trained-roofline" {
				warnTrainedRooflineQuantization(&modelConfig)
			}

			if backend == "roofline" && modelConfig.NumLocalExperts > 1 {
				logrus.Infof("--latency-model: MoE model detected (%d experts, top_%d). Roofline models per-expert FLOPs and active weights; dispatch overhead is not modeled",
					modelConfig.NumLocalExperts, modelConfig.NumExpertsPerTok)
			}

			if !cmd.Flags().Changed("total-kv-blocks") && !kvBlocksFromDefaults {
				kvParams, kvParamsErr := latency.ExtractKVCapacityParams(hfConfig)
				if kvParamsErr != nil {
					logrus.Warnf("--latency-model: could not extract KV capacity params: %v. Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", kvParamsErr, totalKVBlocks)
				} else if hwConfig.MemoryGiB <= 0 {
					logrus.Warnf("--latency-model: GPU memory capacity not available in hardware config; using current total-kv-blocks=%d. Add MemoryGiB to hardware_config.json or pass --total-kv-blocks explicitly", totalKVBlocks)
				} else {
					if kvParams.HiddenAct == "" {
						logrus.Infof("--latency-model: hidden_act not set in config.json; assuming SwiGLU (3-matrix MLP) for weight estimation")
					}
					autoBlocks, calcErr := latency.CalculateKVBlocks(modelConfig, hwConfig, tensorParallelism, blockSizeTokens, gpuMemoryUtilization, kvParams)
					if calcErr != nil {
						logrus.Warnf("--latency-model: KV capacity auto-calculation failed: %v. Using total-kv-blocks=%d. Set --total-kv-blocks explicitly to override", calcErr, totalKVBlocks)
					} else {
						totalKVBlocks = autoBlocks
						logrus.Infof("--gpu-memory-utilization: %.2f used for KV block auto-calculation", gpuMemoryUtilization)
						logrus.Infof("--latency-model: auto-calculated total-kv-blocks=%d (GPU=%.0f GiB, TP=%d, block_size=%d, MoE=%v)",
							totalKVBlocks, hwConfig.MemoryGiB, tensorParallelism, blockSizeTokens, kvParams.IsMoE)
					}
				}
			}

			if !cmd.Flags().Changed("max-model-len") {
				maxPosEmb := hfConfig.MustGetInt("max_position_embeddings", 0)
				if maxPosEmb > 0 {
					maxModelLen = int64(maxPosEmb)
					modelType, _ := hfConfig.Raw["model_type"].(string)
					scaled, applied := applyRopeScaling(maxPosEmb, modelType, hfConfig.Raw["rope_scaling"])
					if applied {
						ropeType := ""
						factor := 0.0
						if ropeMap, ok := hfConfig.Raw["rope_scaling"].(map[string]any); ok {
							ropeType, _ = ropeMap["type"].(string)
							if ropeType == "" {
								ropeType, _ = ropeMap["rope_type"].(string)
							}
							factor, _ = ropeMap["factor"].(float64)
						}
						logrus.Infof("--latency-model: applying %s rope_scaling factor %.1f: %d → %d", ropeType, factor, maxPosEmb, scaled)
						maxModelLen = int64(scaled)
					} else if strings.Contains(modelType, "gemma3") {
						logrus.Infof("--latency-model: skipping rope_scaling for gemma3 (max_position_embeddings is pre-scaled)")
					} else if ropeScaling, ok := hfConfig.Raw["rope_scaling"]; ok && ropeScaling != nil {
						if ropeMap, ok := ropeScaling.(map[string]any); ok {
							if _, hasKey := ropeMap["factor"]; hasKey {
								logrus.Warnf("--latency-model: rope_scaling.factor present but not applied (excluded type, invalid value, or overflow); using max_position_embeddings as-is")
							}
						} else {
							logrus.Warnf("--latency-model: rope_scaling present but not a JSON object (type %T); ignoring", ropeScaling)
						}
					}
					logrus.Infof("--latency-model: auto-derived max-model-len=%d from max_position_embeddings", maxModelLen)
				}
			}

			if maxModelLen > 0 && blockSizeTokens > 0 {
				blocksNeeded := maxModelLen / blockSizeTokens
				if maxModelLen%blockSizeTokens != 0 {
					blocksNeeded++
				}
				if blocksNeeded > totalKVBlocks {
					kvFeasibleMax := totalKVBlocks * blockSizeTokens
					logrus.Warnf("--latency-model: max-model-len %d exceeds KV capacity (%d blocks × %d tokens); capping to %d tokens",
						maxModelLen, totalKVBlocks, blockSizeTokens, kvFeasibleMax)
					maxModelLen = kvFeasibleMax
				}
			}
		}

		if maxModelLen < 0 {
			logrus.Fatalf("--max-model-len must be >= 0, got %d", maxModelLen)
		}

		// Numeric flag validation (same as runCmd)
		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}
		if totalKVBlocks <= 0 {
			logrus.Fatalf("--total-kv-blocks must be > 0, got %d", totalKVBlocks)
		}
		if blockSizeTokens <= 0 {
			logrus.Fatalf("--block-size-in-tokens must be > 0, got %d", blockSizeTokens)
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

		// Load policy bundle if specified (R23: same as runCmd)
		var bundleScorerConfigs []sim.ScorerConfig
		if policyConfigPath != "" {
			bundle, err := sim.LoadPolicyBundle(policyConfigPath)
			if err != nil {
				logrus.Fatalf("Failed to load policy config: %v", err)
			}
			if err := bundle.Validate(); err != nil {
				logrus.Fatalf("Invalid policy config: %v", err)
			}
			if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
				admissionPolicy = bundle.Admission.Policy
			}
			if bundle.Admission.TokenBucketCapacity != nil && !cmd.Flags().Changed("token-bucket-capacity") {
				tokenBucketCapacity = *bundle.Admission.TokenBucketCapacity
			}
			if bundle.Admission.TokenBucketRefillRate != nil && !cmd.Flags().Changed("token-bucket-refill-rate") {
				tokenBucketRefillRate = *bundle.Admission.TokenBucketRefillRate
			}
			if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
				routingPolicy = bundle.Routing.Policy
			}
			bundleScorerConfigs = bundle.Routing.Scorers
			if bundle.Priority.Policy != "" && !cmd.Flags().Changed("priority-policy") {
				priorityPolicy = bundle.Priority.Policy
			}
			if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
				scheduler = bundle.Scheduler
			}
		}

		// Policy name validation (R23: MUST match runCmd lines 941-999 exactly)
		if admissionPolicy == "token-bucket" {
			if tokenBucketCapacity <= 0 || math.IsNaN(tokenBucketCapacity) || math.IsInf(tokenBucketCapacity, 0) {
				logrus.Fatalf("--token-bucket-capacity must be a finite value > 0, got %v", tokenBucketCapacity)
			}
			if tokenBucketRefillRate <= 0 || math.IsNaN(tokenBucketRefillRate) || math.IsInf(tokenBucketRefillRate, 0) {
				logrus.Fatalf("--token-bucket-refill-rate must be a finite value > 0, got %v", tokenBucketRefillRate)
			}
		}
		if !sim.IsValidAdmissionPolicy(admissionPolicy) {
			logrus.Fatalf("Unknown admission policy %q. Valid: %s", admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
		}
		if !sim.IsValidRoutingPolicy(routingPolicy) {
			logrus.Fatalf("Unknown routing policy %q. Valid: %s", routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
		}
		if !sim.IsValidPriorityPolicy(priorityPolicy) {
			logrus.Fatalf("Unknown priority policy %q. Valid: %s", priorityPolicy, strings.Join(sim.ValidPriorityPolicyNames(), ", "))
		}
		if !sim.IsValidScheduler(scheduler) {
			logrus.Fatalf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
		}
		if !trace.IsValidTraceLevel(traceLevel) {
			logrus.Fatalf("Unknown trace level %q. Valid: none, decisions", traceLevel)
		}
		if counterfactualK < 0 {
			logrus.Fatalf("--counterfactual-k must be >= 0, got %d", counterfactualK)
		}
		if traceLevel == "none" && counterfactualK > 0 {
			logrus.Warnf("--counterfactual-k=%d has no effect without --trace-level decisions", counterfactualK)
		}
		if traceLevel == "none" && summarizeTrace {
			logrus.Warnf("--summarize-trace has no effect without --trace-level decisions")
		}
		if traceLevel != "none" && !summarizeTrace {
			logrus.Infof("Decision tracing enabled (trace-level=%s). Use --summarize-trace to print summary.", traceLevel)
		}
		if kvCPUBlocks < 0 {
			logrus.Fatalf("--kv-cpu-blocks must be >= 0, got %d", kvCPUBlocks)
		}
		if kvOffloadThreshold < 0 || kvOffloadThreshold > 1 || math.IsNaN(kvOffloadThreshold) || math.IsInf(kvOffloadThreshold, 0) {
			logrus.Fatalf("--kv-offload-threshold must be a finite value in [0, 1], got %f", kvOffloadThreshold)
		}
		if gpuMemoryUtilization <= 0 || gpuMemoryUtilization > 1.0 || math.IsNaN(gpuMemoryUtilization) || math.IsInf(gpuMemoryUtilization, 0) {
			logrus.Fatalf("--gpu-memory-utilization must be a finite value in (0, 1.0], got %f", gpuMemoryUtilization)
		}
		if kvCPUBlocks > 0 && (kvTransferBandwidth <= 0 || math.IsNaN(kvTransferBandwidth) || math.IsInf(kvTransferBandwidth, 0)) {
			logrus.Fatalf("--kv-transfer-bandwidth must be a finite value > 0 when --kv-cpu-blocks > 0, got %f", kvTransferBandwidth)
		}
		if kvTransferBaseLatency < 0 {
			logrus.Fatalf("--kv-transfer-base-latency must be >= 0, got %d", kvTransferBaseLatency)
		}
		if snapshotRefreshInterval < 0 {
			logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
		}
		if admissionLatency < 0 {
			logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
		}
		if routingLatency < 0 {
			logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
		}

		logrus.Infof("Policy config: admission=%s, routing=%s, priority=%s, scheduler=%s",
			admissionPolicy, routingPolicy, priorityPolicy, scheduler)

		// Parse scorer configuration for weighted routing (R23: exact structure from runCmd)
		var parsedScorerConfigs []sim.ScorerConfig
		if routingPolicy == "weighted" {
			if routingScorers != "" {
				var err error
				parsedScorerConfigs, err = sim.ParseScorerConfigs(routingScorers)
				if err != nil {
					logrus.Fatalf("Invalid --routing-scorers: %v", err)
				}
			} else if len(bundleScorerConfigs) > 0 {
				parsedScorerConfigs = bundleScorerConfigs
			}
			activeScorerConfigs := parsedScorerConfigs
			if len(activeScorerConfigs) == 0 {
				activeScorerConfigs = sim.DefaultScorerConfigs()
			}
			scorerStrs := make([]string, len(activeScorerConfigs))
			for i, sc := range activeScorerConfigs {
				scorerStrs[i] = fmt.Sprintf("%s:%.1f", sc.Name, sc.Weight)
			}
			logrus.Infof("Weighted routing scorers: %s", strings.Join(scorerStrs, ", "))
		}
		if routingPolicy != "weighted" && routingScorers != "" {
			logrus.Warnf("--routing-scorers has no effect when routing policy is %q (only applies to 'weighted')", routingPolicy)
		}
		if admissionPolicy == "token-bucket" {
			logrus.Infof("Token bucket: capacity=%.0f, refill-rate=%.0f", tokenBucketCapacity, tokenBucketRefillRate)
		}

		logrus.Infof("Starting replay with %d KV blocks, horizon=%dticks, alphaCoeffs=%v, betaCoeffs=%v",
			totalKVBlocks, replayHorizon, alphaCoeffs, betaCoeffs)

		startTime := time.Now()

		// Build cluster config (same as runCmd, using replayHorizon instead of simulationHorizon)
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: replayHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency),
				BatchConfig:         sim.NewBatchConfig(maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold),
				LatencyCoeffs:       sim.NewLatencyCoeffs(betaCoeffs, alphaCoeffs),
				ModelHardwareConfig: sim.NewModelHardwareConfig(modelConfig, hwConfig, model, gpu, tensorParallelism, backend, maxModelLen),
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

		// Collect RawMetrics (R23: same as runCmd — needed for anomaly/KV/SLO output)
		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			priorityPolicy,
			cs.RoutingRejections(),
		)

		// Print anomaly counters if any detected (R23: same as runCmd)
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 {
			fmt.Println("=== Anomaly Counters ===")
			fmt.Printf("Priority Inversions: %d\n", rawMetrics.PriorityInversions)
			fmt.Printf("HOL Blocking Events: %d\n", rawMetrics.HOLBlockingEvents)
			fmt.Printf("Rejected Requests: %d\n", rawMetrics.RejectedRequests)
			fmt.Printf("Dropped Unservable: %d\n", rawMetrics.DroppedUnservable)
			fmt.Printf("Length-Capped Requests: %d\n", rawMetrics.LengthCappedRequests)
		}

		// Print KV cache metrics if any nonzero (R23: same as runCmd)
		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		// Print per-SLO metrics if multiple SLO classes present (R23: same as runCmd)
		sloDistributions := cluster.ComputePerSLODistributions(cs.AggregatedMetrics())
		printPerSLOMetrics(os.Stdout, sloDistributions)

		// Build and print trace summary if requested (R23: same as runCmd)
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
	// Override --results-path description for replay: schema differs from blis run.
	// blis run writes MetricsOutput JSON; blis replay writes []SimResult JSON.
	replayCmd.Flags().Lookup("results-path").Usage = "File to write []SimResult JSON (request_id, ttft_us, e2e_us, input_tokens, output_tokens) for blis calibrate consumption. Note: blis run writes MetricsOutput JSON to this flag; blis replay writes SimResult JSON."
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
