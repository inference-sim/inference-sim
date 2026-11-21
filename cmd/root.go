package cmd

import (
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/check"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/deployment"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/measurement"
	tracespec "github.com/inference-sim/inference-sim/substrate/contracts/spec/trace"
	schemaworkload "github.com/inference-sim/inference-sim/substrate/contracts/spec/workload"
)

// distDefaults are the default values for distribution synthesis flags.
//
// Flags covered:
//
//	--prompt-tokens, --prompt-tokens-stdev, --prompt-tokens-min, --prompt-tokens-max
//	--output-tokens, --output-tokens-stdev, --output-tokens-min, --output-tokens-max
//
// NOT covered: --prefix-tokens (default 0 means "no shared prefix", a feature toggle,
// not a distribution shape parameter).
const (
	defaultPromptMean  = 512
	defaultPromptStdev = 256
	defaultPromptMin   = 2
	defaultPromptMax   = 7000
	defaultOutputMean  = 512
	defaultOutputStdev = 256
	defaultOutputMin   = 2
	defaultOutputMax   = 7000
)

var (
	// Run settings: CLI flags, and the engine settings adoptKernelDeployment takes from the
	// scenario and the kernel (KV blocks, block size, sequence/token caps, max_model_len)
	seed                      int64                  // Seed for random token generation
	simulationHorizon         int64                  // Total simulation time (in ticks)
	logLevel                  string                 // Log verbosity level
	totalKVBlocks             int64                  // KV blocks per rank, sized by the kernel
	sequenceState             kvSequenceState        // each resident sequence's bounded state in pool blocks, from the kernel (#1941)
	maxNumSeqs                int64                  // Maximum number of requests in the Running batch (vLLM: --max-num-seqs)
	maxNumBatchedTokens       int64                  // Maximum total number of tokens across requests in the Running batch (vLLM: --max-num-batched-tokens)
	noEnablePrefixCaching     bool                   // --no-enable-prefix-caching: disable cross-request GPU prefix reuse (vLLM parity, #1867)
	prefixHit                 kvPrefixHit            // how the KV cache groups cut a prefix hit, from the kernel's model
	blockSizeTokens           int64                  // Number of tokens per KV block
	defaultsFilePath          string                 // Path to default constants: the LoRA cost coefficients
	workloadType              string                 // Workload type (chatbot, summarization, contentgen, multidoc, distribution)
	longPrefillTokenThreshold int64                  // the scenario's engine.long_prefill_token_threshold (colocated run)
	rate                      float64                // Requests arrival per second
	numRequests               int                    // Number of requests
	concurrency               int                    // Number of concurrent virtual users (closed-loop)
	thinkTimeMs               int                    // Think time between response and next request (ms)
	prefixTokens              int                    // Prefix Token Count
	promptTokensMean          int                    // Average Prompt Token Count
	promptTokensStdev         int                    // Stdev Prompt Token Count
	promptTokensMin           int                    // Min Prompt Token Count
	promptTokensMax           int                    // Max Prompt Token Count
	outputTokensMean          int                    // Average Output Token Count
	outputTokensStdev         int                    // Stdev Output Token Count
	outputTokensMin           int                    // Min Output Token Count
	outputTokensMax           int                    // Max Output Token Count
	kernelScenario            string                 // CLI --scenario: scenario file name (required)
	kernelScenarioDir         string                 // CLI --scenarios: directory of scenario files (required)
	kernelDeploymentExperts   int                    // routed expert count from the model graph; 0 for a dense model
	kernelDeploymentTopK      int                    // routed experts per token from the model graph; 0 for a dense model
	kernelOpened              *engineModel           // the kernel adoptKernelDeployment opened; nil until it runs
	scenarioEngine            engineRef              // the engine the scenario selects (selectEngine); set by adoptKernelDeployment
	kernelPDOpened            *kernelPools           // the per-role kernels openKernelPools opened; nil for a colocated run
	scenarioScheduler         string                 // the scheduler the scenario's engine states ("" = none)
	scenarioOffload           *deployment.Offload    // the offload block of the run's first-served pool (prefill under P/D, else the colocated pool), the deployment's applied; nil when it states none (#1910)
	scenarioDecodeOffload     *deployment.Offload    // the decode pool's, under P/D (pool offload, else the deployment's)
	scenarioOverrides         []sim.ScenarioOverride // what the run took from other than its scenario (results file)
	maxModelLen               int64                  // the scenario's engine.max_model_len: max total sequence length (input + output)
	// The deployment: model, GPU, TP, DP and EP, all from the scenario
	model                string // LLM name
	gpu                  string // GPU type
	tensorParallelism    int    // TP value
	dataParallelism      int    // DP value (MoE only), from the kernel scenario
	enableExpertParallel bool   // EP mode (MoE only), from the kernel scenario

	// cluster config
	numInstances int // Number of instances in the cluster

	sloPriorityOverrides map[string]int // SLO class → engine priority overrides (nil = defaults)

	// The scheduler and preemption policy, from the scenario's engine.scheduling_policy
	scheduler        string // Scheduler name
	preemptionPolicy string // Preemption victim selection policy

	// Policy bundle config
	policyConfigPath string // Path to YAML policy configuration file

	// LoRA control-plane config (#1464). All optional; absence => subsystem inert (INV-6).
	loraConfigPath            string  // Path to YAML file with a top-level lora: block (adapter registry + capacity + coefficients)
	loraAdapterCapacity       int     // --lora-adapter-capacity (applied only when Changed; 0 is meaningful => adapters forbidden)
	loraLoadBaseLatencyUs     float64 // --lora-load-base-latency-us
	loraLoadBandwidthBytesUs  float64 // --lora-load-bandwidth-bytes-us
	loraFootprintBytesPerRank float64 // --lora-footprint-bytes-per-rank

	// Speculative decoding / MTP (#1528). The draft length K and method come from the
	// scenario's engine.speculative_config (set when the kernel deployment is adopted); K=0 =>
	// feature inert, output byte-identical (INV-6). The acceptance is resolved by
	// blis-schemas' precedence (#1939): registry profile < workload < scenario < CLI.
	numSpeculativeTokens  int     // K, from the scenario
	speculativeAcceptance float64 // --speculative-acceptance-rate (α ∈ [0,1]); see speculative_acceptance.go
	speculativeMethod     string  // method, from the scenario (informational label)

	// loraReservedBytesForKV carries the resolved static LoRA HBM reservation (bytes)
	// into the kernel's KV sizing (applyKernelLoRAReservation, and each P/D pool's
	// SettingsReserving). Set once per command RunE from the single resolveLoRAConfig
	// call; 0 when the subsystem is inert, leaving the kernel's KV budget unchanged (INV-6).
	loraReservedBytesForKV int64

	// Fitness evaluation config (PR9)
	fitnessWeights string // Fitness weights string "key:val,key:val"

	// Decision trace config (PR13)
	traceLevel      string // Trace verbosity level
	counterfactualK int    // Number of counterfactual candidates
	summarizeTrace  bool   // Print trace summary after simulation

	// Workload spec config (PR10)
	workloadSpecPath string // Path to YAML workload specification file

	dpLoadBalancing string // the scenario's parallel.data_parallel_load_balancing: internal | hybrid | external (#1909)

	// PD disaggregation config
	prefillInstances       int  // Number of instances dedicated to prefill
	decodeInstances        int  // Number of instances dedicated to decode
	prefillDecodeInstances int  // Number of shared-role instances (both prefill and decode), issue #1276
	pdTransferContention   bool // concurrent P/D handoffs share the links they cross (#1922)

	// E/P/D disaggregation config
	encodeInstances int // Number of instances dedicated to encoding multimodal input (0 = disabled)

	// Autoscaler config (Phase 1C)
	modelAutoscalerIntervalUs float64 // tick interval in μs; 0 = disabled

	// per-request timeout override for blis run (seconds; negative = disabled, 0 is rejected)
	requestTimeoutSecs int

	// Goodput SLO targets (issue #1413). Each is "class=duration[,class=duration...]" using
	// Go duration syntax. Distinct from --slo-targets (dispatch ordering, µs); these gate
	// goodput emission. Precedence: CLI > trace header > workload spec.
	goodputSLOTTFT string
	goodputSLOITL  string
	goodputSLOE2E  string

	// output file paths
	serverSeriesIntervalS float64 // --server-series-interval-s: bin width of the file-only server_series block; 0 = off
	// saturationReport (--saturation-report): per-event verdict trace file (#1516).
	// Declared in saturation.go alongside --detectors.

	// trace export
	traceOutput string // File prefix for TraceV2 export (<prefix>.yaml + <prefix>.csv)
)

// rootCmd is the base command for the CLI
var rootCmd = &cobra.Command{
	Use:   "blis",
	Short: "BLIS — Blackbox Inference Simulator for LLM serving systems",
}

// validateDistributionParams checks token distribution bounds common to both the
// concurrency and distribution synthesis paths (R3). Returns a non-empty error
// string if any parameter violates a bound, empty string if all are valid.
// A stdev of 0 is always valid — it produces a constant (deterministic) distribution.
// Extracted for unit testability (R14).
func validateDistributionParams(promptMin, promptMax, outputMin, outputMax, promptStdev, outputStdev, promptMean, outputMean int) string {
	if promptMin < 1 {
		return fmt.Sprintf("--prompt-tokens-min must be >= 1, got %d", promptMin)
	}
	if promptMax < 1 {
		return fmt.Sprintf("--prompt-tokens-max must be >= 1, got %d", promptMax)
	}
	if outputMin < 1 {
		return fmt.Sprintf("--output-tokens-min must be >= 1, got %d", outputMin)
	}
	if outputMax < 1 {
		return fmt.Sprintf("--output-tokens-max must be >= 1, got %d", outputMax)
	}
	if promptStdev < 0 {
		return fmt.Sprintf("--prompt-tokens-stdev must be >= 0, got %d", promptStdev)
	}
	if outputStdev < 0 {
		return fmt.Sprintf("--output-tokens-stdev must be >= 0, got %d", outputStdev)
	}
	if promptMin > promptMax {
		return fmt.Sprintf("--prompt-tokens-min (%d) must be <= --prompt-tokens-max (%d)", promptMin, promptMax)
	}
	if outputMin > outputMax {
		return fmt.Sprintf("--output-tokens-min (%d) must be <= --output-tokens-max (%d)", outputMin, outputMax)
	}
	if promptMean > promptMax || promptMean < promptMin || promptStdev > promptMax || (promptStdev != 0 && promptStdev < promptMin) {
		return "prompt-tokens and prompt-tokens-stdev should be in range [prompt-tokens-min, prompt-tokens-max]"
	}
	if outputMean > outputMax || outputMean < outputMin || outputStdev > outputMax || (outputStdev != 0 && outputStdev < outputMin) {
		return "output-tokens and output-tokens-stdev should be in range [output-tokens-min, output-tokens-max]"
	}
	return ""
}

// latencyResolution holds the resolved components from resolveLatencyConfig.
// Callers use these values to construct sim.SimConfig sub-configs.
type latencyResolution struct {
	// KernelModel is the blis-latency-kernel adapter for the scenario's first pool; the caller
	// puts it on SimConfig.LatencyModel.
	KernelModel latencyModel
	// ModelConfig carries only the model graph's expert geometry, for the MoE gates and the
	// ModelHardwareConfig boundary.
	ModelConfig modelConfig
}

// refuseDPAutoscaler refuses the model autoscaler on an MoE scenario at dp > 1
// (#2147): the engine is the whole data-parallel group behind one endpoint, and the
// autoscaler places single-role instances with no DP-group awareness, so whether it would add
// one rank or one whole group is undefined. Dense dp > 1 is refused earlier
// (resolveLatencyConfig).
func refuseDPAutoscaler(isMoE bool, dp int, autoscalerActive bool) error {
	if !isMoE || dp <= 1 || !autoscalerActive {
		return nil
	}
	return fmt.Errorf("scenario dp %d (MoE) is not supported with the model autoscaler (#2147): the "+
		"engine is the whole data-parallel group behind one endpoint, and the autoscaler places single-role "+
		"instances with no DP-group awareness. Use a scenario that states dp 1 with the autoscaler, or "+
		"disable the autoscaler", dp)
}

// adoptSchedulingPolicy takes the instance scheduler and preemption policy from the scenario's
// engine.scheduling_policy, vLLM's own setting: fcfs, or priority (served lowest priority value
// first, then by arrival -- BLIS's priority-fcfs). The scenario is the only source (#2037).
func adoptSchedulingPolicy(m *engineModel) {
	stated := enginePolicyOf(m)
	scenarioScheduler = stated.scheduler
	scheduler, preemptionPolicy = stated.scheduler, stated.preemption
}

// enginePolicyOf resolves a pool's engine.scheduling_policy (build.EnginePolicyOf); a policy
// BLIS does not simulate is fatal.
func enginePolicyOf(m *engineModel) enginePolicy {
	p, err := enginePolicyFor(m)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	return enginePolicy{scheduler: p.Scheduler, preemption: p.Preemption}
}

// enginePolicy is a pool's resolved queue order and preemption policy.
type enginePolicy struct{ scheduler, preemption string }

// adoptKernelDeployment resolves the deployment from the --scenario kernel scenario, and is
// fatal when --scenario, --scenarios or --registry is missing.
//
// The scenario is the committed record of what was deployed: it states the model, the
// hardware, and each pool's tensor- and data-parallel width. Accepting the same facts as
// flags too would mean writing precedence logic to decide between two sources, which is
// complexity carrying no information -- so no flag restates them, and the rest are derived.
//
// It runs before the deployment gates rather than inside resolveLatencyConfig because
// those gates refuse a run whose deployment nobody chose (NS-6), and here the scenario is
// who chose it. The gates still execute, on these values.
func adoptKernelDeployment(cmd *cobra.Command) {
	// A fresh adoption starts with no scenario facts recorded (tests adopt more than once).
	scenarioScheduler, scenarioOverrides = "", nil
	scenarioRouter, scenarioStatesRouter = routerInputs{}, false
	// The engine is selected, and a release this build does not run refused (LIF-6), before the
	// deployment is read: no other check runs on a scenario this build cannot run.
	if kernelScenario != "" && kernelScenarioDir != "" {
		ref, err := selectEngine(engineScenarioPath())
		if err != nil {
			logrus.Fatalf("%v", err)
		}
		scenarioEngine = ref
		if ref.Kind == engineKindSGLang {
			sglangKernels.reset()
			if err := adoptSGLangTwin(); err != nil {
				logrus.Fatalf("%v", err)
			}
		}
	}
	in := adoptInputs(cmd)
	a, err := adoptScenario(in)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if taken, err := takePDTopologyFromPools(cmd, a); err != nil {
		logrus.Fatalf("%v", err)
	} else if taken {
		in.PDTopology = true
	}
	shape := a.Shape
	// The scenario's offload blocks are the run's tier configuration (#1910): each pool's own, else
	// the deployment's.
	scenarioOffload, scenarioDecodeOffload = runOffloads(shape)
	adoptScenarioEngineKnobs(shape)
	scenarioShape = shape
	acceptanceRecord, poolSpeculation = nil, nil
	prefixCacheRecords = nil
	// The router's inputs are the deployment's too: its router block and pool labels.
	adoptScenarioRouter(shape)
	if err := openFirstPool(in, &a); err != nil {
		logrus.Fatalf("%v", err)
	}
	m, settings := a.Model, a.Settings
	kernelOpened = m
	adoptSchedulingPolicy(m)
	dep := a.Deployment
	model = strings.ToLower(dep.Model)
	gpu = dep.Hardware
	tensorParallelism = dep.TP
	dataParallelism = dep.DP
	enableExpertParallel = dep.ExpertParallel
	// The expert geometry, so every MoE gate and the ModelHardwareConfig boundary see a
	// routed model as routed. The kernel parses a model GRAPH rather than an HF
	// config, so ModelConfig would otherwise stay zero and read as dense -- and the
	// library boundary's "DP > 1 needs >= 2 experts" would panic on a model that has
	// 128 of them. Stating the counts satisfies that invariant with the model's own
	// numbers instead of waiving it.
	kernelDeploymentExperts = dep.Experts
	kernelDeploymentTopK = dep.ExpertsPerTok
	// The engine the kernel priced, so the simulated scheduler admits against the same caps,
	// pages and pool (the build site, adoptFirstPoolEngine).
	adoptFirstPoolEngine(m, settings)
	if scenarioEngine.Kind == engineKindSGLang {
		// The prefix-cache record is the host adapter's reading of vLLM's model runner and cache
		// layout; SGLang's radix cache is the engine's own and reports through its metrics.
		prefixCacheRecords = nil
	}
}

// resolveLatencyConfig builds the latency model for a run or a replay: the blis-latency-kernel
// adapter for the scenario's first pool, which adoptKernelDeployment opened and sized the run
// from. It is the one path both commands traverse (R23, INV-13).
//
// What it does:
//   - Normalizes the model name and records the catalog root for the results file's
//     provenance block (#1732, #1900)
//   - Carries the model graph's expert geometry onto ModelConfig, so the MoE gates and the
//     ModelHardwareConfig boundary see a routed model as routed
//   - Refuses data parallelism on a dense model and expert parallelism on a dense model, as
//     the placement and vLLM do
//
// Side effects: model (lowercased).
func resolveLatencyConfig(cmd *cobra.Command) latencyResolution {
	model = strings.ToLower(model)
	m := kernelOpened
	if m == nil {
		// adoptKernelDeployment has not run (a caller that resolves the latency model
		// alone): open the scenario's first pool exactly as adoption would.
		var err error
		m, err = openModel()
		if err != nil {
			logrus.Fatalf("%v", err)
		}
	}
	modelConfig := expertModelConfig()
	if err := checkParallelism(modelConfig.IsMoE()); err != nil {
		logrus.Fatalf("%v", err)
	}
	return latencyResolution{
		KernelModel: m,
		ModelConfig: modelConfig,
	}
}

// resolvePolicies resolves the engine-level policy configuration (instance scheduler,
// preemption, SLO priority mapping) and the decision-trace settings from CLI flags and an
// optional policy bundle YAML file. It is called by both runCmd and replayCmd to ensure a
// single validation code path (R23: code path parity). Admission, routing and flow control
// are the llm-d router's (see router.go).
//
// Precondition: adoptKernelDeployment must be called first.
//
// Returns the loaded policy bundle (nil if none).
func resolvePolicies(cmd *cobra.Command) *sim.PolicyBundle {
	var loadedBundle *sim.PolicyBundle

	// Load policy bundle if specified (R18: CLI flags override YAML values)
	if policyConfigPath != "" {
		bundle, err := sim.LoadPolicyBundle(policyConfigPath)
		if err != nil {
			logrus.Fatalf("Failed to load policy config: %v", err)
		}
		if err := bundle.Validate(); err != nil {
			logrus.Fatalf("Invalid policy config: %v", err)
		}
		loadedBundle = bundle
		if bundle.Priority.SLOPriorities != nil {
			sloPriorityOverrides = bundle.Priority.SLOPriorities
			// Validate at CLI boundary (R3) before reaching library-level panic in NewSLOPriorityMap.
			for k, v := range sloPriorityOverrides {
				if v < -100 || v > 100 {
					logrus.Fatalf("slo_priorities override for %q: value %d out of range [-100, 100]", k, v)
				}
			}
		}
		if bundle.Priority.Policy != "" && bundle.Priority.Policy != "constant" {
			logrus.Warnf("bundle priority.policy=%q has no effect: --priority-policy was removed in PR #1216. "+
				"Request priority is now static (set at enqueue via SLOPriorityMap.InvertForVLLM). "+
				"Use SLOClass in your workload spec for tier-based priority.", bundle.Priority.Policy)
		}
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
	logrus.Infof("Policy config: scheduler=%s, preemption=%s", scheduler, preemptionPolicy)
	return loadedBundle
}

// registerSimConfigFlags registers all simulation-engine configuration flags
// on the given command. Called by both runCmd and replayCmd to avoid
// duplicating ~50 flag registrations.
func registerSimConfigFlags(cmd *cobra.Command) {
	cmd.Flags().Int64Var(&seed, "seed", 42, "Seed for random request generation")
	cmd.Flags().Int64Var(&simulationHorizon, "horizon", math.MaxInt64, "Total simulation horizon (in ticks)")
	cmd.Flags().StringVar(&logLevel, "log", "warn", "Log level for diagnostic messages (trace, debug, info, warn, error, fatal, panic). Simulation results always print to stdout regardless of this setting.")
	cmd.Flags().StringVar(&defaultsFilePath, "defaults-filepath", defaultDefaultsPath, "Path to default constants: the LoRA cost coefficients (the copy compiled into the binary is used when this is unset and no defaults.yaml is in the working directory)")
	registerDataFlags(cmd)

	// Engine settings are the scenario's (#2021 B6, #2037): block size, sequence and token caps,
	// max_model_len, long_prefill_token_threshold, prefix caching, cache dtype, speculation, the
	// scheduling policy, SWA bounded replay, data-parallel load balancing and the offload tiers;
	// the KV budget is the kernel's. No flag restates them (removed_flags.go).
	registerFrontendFlags(cmd)
	cmd.Flags().Float64Var(&serverSeriesIntervalS, "server-series-interval-s", 0, "Bin width in seconds of the per-engine vLLM /metrics time series (KV usage, running/waiting, prefix-cache and prompt-token-by-source counters, generation tokens) written as server_series to the --metrics-path file only; 0 disables it")

	// The deployment: a blis-schemas scenario (model, hardware, fabric, pools, engines)
	// priced by blis-latency-kernel with the coefficient sets it names.
	cmd.Flags().StringVar(&kernelScenario, "scenario", "", "Scenario FILE NAME within --scenarios, e.g. gpt-oss-120b-h200-fp4-vllm-tp4.yaml (REQUIRED): the blis-schemas Scenario and Deployment that state the model, hardware, fabric, each pool's parallelism and engine settings, and the coefficient sets that price it")
	cmd.Flags().StringVar(&kernelScenarioDir, "scenarios", "", "Directory holding scenario YAML files (REQUIRED)")

	// Cluster config
	cmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

	// The llm-d router: its own inputs, and the environment around it (router.go).
	registerRouterFlags(cmd)

	// Policy bundle config
	cmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")

	// Fitness evaluation config (PR9)
	cmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs (e.g., throughput:0.5,p99_ttft:0.3)")

	// Decision trace config (PR13)
	cmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
	cmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Record the top-k endpoints the router scored, with their state, per routing decision")
	cmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")

	cmd.Flags().Float64Var(&modelAutoscalerIntervalUs, "model-autoscaler-interval-us", 0, "Autoscaler tick interval in microseconds (0 = disabled). Overrides policy-config autoscaler.interval_us when non-zero.")

	// PD disaggregation config
	cmd.Flags().IntVar(&prefillInstances, "prefill-instances", 0, "Number of instances dedicated to prefill (0 = disabled)")
	cmd.Flags().IntVar(&decodeInstances, "decode-instances", 0, "Number of instances dedicated to decode (0 = disabled)")
	cmd.Flags().IntVar(&prefillDecodeInstances, "prefill-decode-instances", 0, "Number of shared-role instances serving both prefill and decode (llm-d 'prefill-decode'/'both' parity; 0 = disabled). Must satisfy --prefill-instances + --decode-instances + --prefill-decode-instances <= --num-instances.")
	cmd.Flags().BoolVar(&pdTransferContention, "pd-transfer-contention", false, "Concurrent P/D KV handoffs share the fabric links they cross (#1922): each link's bandwidth is divided max-min fairly among the handoffs on it, from the demand the cost model states for it, and re-divided as handoffs start and finish. A handoff alone takes exactly that demand's alone time. Default off: every handoff is priced as if alone.")

	// E/P/D disaggregation. Registered on both run and replay.
	cmd.Flags().IntVar(&encodeInstances, "encode-instances", 0, "Number of instances dedicated to encoding multimodal input (0 = encode pool disabled, default)")

	// LoRA control-plane config (#1464). Registered on both run and replay (INV-13
	// parity). All optional; absence => subsystem inert (INV-6). The adapter registry
	// and per-rank step_overhead_tiers are config-file only (--lora-config); a scalar
	// flag cannot express a per-rank map. Scalar coefficient flags compose with and
	// override the file / defaults.yaml (R18: applied only when Changed).
	cmd.Flags().StringVar(&loraConfigPath, "lora-config", "", "Path to YAML file with a top-level lora: block (adapter registry, capacity, cost coefficients). The static adapter HBM reservation is set aside before the kernel sizes the KV pool. Absent => LoRA subsystem inert.")
	cmd.Flags().IntVar(&loraAdapterCapacity, "lora-adapter-capacity", 0, "Per-instance resident adapter slots (0 with adapters declared => error). Applied only when set.")
	cmd.Flags().Float64Var(&loraLoadBaseLatencyUs, "lora-load-base-latency-us", 0, "Cold adapter-load fixed latency in µs. Applied only when set; else --lora-config / defaults.yaml.")
	cmd.Flags().Float64Var(&loraLoadBandwidthBytesUs, "lora-load-bandwidth-bytes-us", 0, "Cold adapter-load bandwidth in bytes/µs (>0). Applied only when set; else --lora-config / defaults.yaml.")
	cmd.Flags().Float64Var(&loraFootprintBytesPerRank, "lora-footprint-bytes-per-rank", 0, "Adapter HBM footprint per rank unit in bytes (>0). Applied only when set; else --lora-config / defaults.yaml.")

	// Speculative decoding / MTP (#1528). Model-level; shared by run and replay so a
	// trace round-trips under identical flags (INV-13). Default off => byte-identical.
	registerAcceptanceFlags(cmd)
	registerRoutingFlags(cmd)
}

// resolveSpeculativeConfig builds the speculative-decoding / MTP config (#1528): K and the
// method from the scenario, and the acceptance resolved by blis-schemas' precedence (#1939,
// resolveAcceptance), which it records for the results file (acceptanceRecord). Shared by run
// and replay, so a trace round-trips under identical flags (INV-13). Fatal on an invalid or
// unresolvable config (CLI boundary, R6).
func resolveSpeculativeConfig(cmd *cobra.Command) speculativeConfig {
	acceptanceRecord, poolSpeculation = nil, nil
	if poolsDraftDifferently(kernelPDOpened) {
		// The pools draft differently (inference-sim#2125): each resolves its own acceptance.
		return resolvePoolSpeculation(cmd, kernelPDOpened)
	}
	c, rec := resolveSpeculation(cmd, kernelOpened, numSpeculativeTokens, speculativeMethod, "")
	acceptanceRecord = rec
	return c
}

// resolveSpeculation resolves the speculation of the pool m drafting k tokens with method: its
// config and, when it drafts, the acceptance record, logging what it resolved (what names the pool
// when the run has more than one speculation).
func resolveSpeculation(cmd *cobra.Command, m *engineModel, k int, method, what string) (speculativeConfig, *sim.AcceptanceRecord) {
	ar, speculating := resolveAcceptanceFor(cmd, m, k, method)
	alpha := 0.0
	if speculating {
		alpha = ar.Res.AcceptanceRate()
		if ar.CLIRate != nil {
			alpha = *ar.CLIRate
		}
	} else if cmd.Flags().Changed(acceptanceRateFlag) {
		alpha = speculativeAcceptance // refused below: a rate with nothing drafted
	}
	c := speculativeConfigOf(k, alpha, method)
	if !c.IsEnabled() {
		return c, nil
	}
	rec := acceptanceRecordOf(ar.Res, c)
	logAcceptance(rec, c.K)
	logrus.Infof("speculative decoding enabled%s: method=%q k=%d acceptance=%.3f (mean %.3f tokens/step, verify width %d, %d lookahead KV slots)",
		what, c.Method, c.K, c.Acceptance, c.EffectiveTokensPerStep(), c.VerifyWidth(), c.LookaheadTokens())
	if c.Acceptance == 0 {
		logrus.Warnf("speculative decoding: acceptance length 1 => no throughput gain but verify-width cost applies (net slowdown); this is usually not intended")
	}
	return c, rec
}

// applyTimeoutToSpec sets ClientSpec.Timeout and CohortSpec.Timeout on every entry in spec.
// timeoutSecs>0 converts to µs and sets a deadline; timeoutSecs<=0 sets an explicit *int64(0)
// (disabled). Explicit zero is required so computeDeadline does not fall back to the 300s
// session default. Callers must reject timeoutSecs==0 before calling (use negative to disable).
func applyTimeoutToSpec(spec *workload.WorkloadSpec, timeoutSecs int) {
	var us int64
	if timeoutSecs > 0 {
		us = int64(timeoutSecs) * 1_000_000
	}
	for i := range spec.Clients {
		t := us
		spec.Clients[i].Timeout = &t
	}
	for i := range spec.Cohorts {
		t := us
		spec.Cohorts[i].Timeout = &t
	}
}

// applyTimeoutToSessions re-applies the timeout to the session blueprints the generator built:
// for inference_perf specs, spec.Clients is empty when applyTimeoutToSpec runs and is populated
// inside the generator. timeoutSecs>0 sets the timeout; timeoutSecs<=0 disables it.
func applyTimeoutToSessions(wl *workload.GeneratedWorkload, timeoutSecs int) {
	var timeoutUs int64
	if timeoutSecs > 0 {
		timeoutUs = int64(timeoutSecs) * 1_000_000
	}
	for i := range wl.Sessions {
		t := timeoutUs
		wl.Sessions[i].Timeout = &t
	}
}

// runCmd executes the simulation using parameters from CLI flags
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Run the inference simulation",
	Run: func(cmd *cobra.Command, args []string) {
		// Set up logging
		level, err := logrus.ParseLevel(logLevel)
		if err != nil {
			logrus.Fatalf("Invalid log level: %s", logLevel)
		}
		logrus.SetLevel(level)
		validateRouterFlags()
		checkResultFlags()
		checkReplicationFlags(cmd)
		if replications != 1 {
			runReplications(cmd)
			return
		}
		defer applyFromResult("run", cmd)()
		snapshotResultFlags("run", cmd)

		// The deployment comes from the scenario, so that resolution runs BEFORE the
		// gates below: those refuse a run whose deployment nobody chose, and the scenario
		// is who chose it. Ordering, not an exemption -- the gates still run, on the
		// derived values.
		adoptKernelDeployment(cmd)

		if model == "" { // model not provided, exit
			logrus.Fatalf("LLM name not provided. Exiting simulation.")
		}

		// LoRA control-plane (#1464): resolve the config ONCE here (R4 single site) so
		// the kernel's KV sizing -- applyKernelLoRAReservation here and each P/D pool's
		// overrides below set the static HBM reservation aside (PR5) -- and the SimConfig
		// literal further down share one resolution. The reservation is 0 (KV
		// unaffected) when the subsystem is inert (INV-6).
		loraCfg := resolveLoRAConfig(cmd)
		loraReservedBytesForKV = adapterReservedBytesFor(loraCfg)
		applyKernelLoRAReservation()

		// KV-cache offload config (#1587, #1910): the scenario's offload block, resolved ONCE
		// (R4) and validated at the CLI boundary. Inert (zero value) when the scenario states
		// none (BC-G5). Recorded in the exported trace header below for run/replay parity
		// (BC-G6).
		kvOffloadCfg := resolveKVOffloadConfig()
		// Under P/D the decode pool's offload is its own (a prefill pool can offload while the decode
		// pool does not); kvOffloadCfg is the prefill pool's.
		decodeOffloadCfg := resolveDecodeKVOffloadConfig()

		// Resolve the latency model (single code path shared with replayCmd).
		lr := resolveLatencyConfig(cmd)

		// Per-pool engine overrides, filled from each P/D pool's own kernel
		// (openKernelPools, below) after PD validation. Empty when disaggregation is
		// disabled (prefillInstances == 0).
		var prefillOverrides, decodeOverrides poolOverrides

		// R3: Validate workload generation flags (before any synthesis path consumes them)
		if numRequests < 0 {
			logrus.Fatalf("--num-requests must be >= 0, got %d", numRequests)
		}
		if prefixTokens < 0 {
			logrus.Fatalf("--prefix-tokens must be >= 0, got %d", prefixTokens)
		}

		// R3: Validate concurrency flags
		if concurrency < 0 {
			logrus.Fatalf("--concurrency must be >= 0, got %d", concurrency)
		}
		if thinkTimeMs < 0 {
			logrus.Fatalf("--think-time-ms must be >= 0, got %d", thinkTimeMs)
		}
		// BC-1: --concurrency and --rate are mutually exclusive
		if concurrency > 0 && cmd.Flags().Changed("rate") {
			logrus.Fatalf("--concurrency and --rate are mutually exclusive; use one or the other")
		}

		// Workload configuration — all paths synthesize a v2 WorkloadSpec and stream requests
		// from workload.GenerateWorkloadLazy (BC-10).
		var spec *workload.WorkloadSpec
		var sessionMgr *workload.SessionManager

		if workloadSpecPath != "" {
			if concurrency > 0 {
				logrus.Fatalf("--concurrency cannot be used with --workload-spec; " +
					"define concurrency in the spec file using clients[].concurrency instead")
			}
			// --workload-spec takes precedence over --workload
			var err error
			spec, err = workload.LoadWorkloadSpec(workloadSpecPath)
			if err != nil {
				logrus.Fatalf("Failed to load workload spec: %v", err)
			}
			// Apply CLI --seed override (R18: CLI flag precedence)
			if cmd.Flags().Changed("seed") {
				logrus.Infof("CLI --seed %d overrides workload-spec seed %d", seed, spec.Seed)
				spec.Seed = seed
			} else {
				logrus.Infof("Using workload-spec seed %d (CLI --seed not specified)", spec.Seed)
			}
			if spec.Horizon > 0 && !cmd.Flags().Changed("horizon") {
				simulationHorizon = spec.Horizon
			}
		} else if concurrency > 0 {
			// Concurrency mode → synthesize v2 spec with closed-loop client.
			// In concurrency mode, --num-requests has no meaningful default.
			// If the user did not explicitly set it, leave it at 0 (unbounded) and
			// require --horizon to bound the run. The existing unbounded-generation
			// guard will fire with a clear message if neither is provided.
			// R3: Validate distribution token bounds (shared with distribution mode).
			if msg := validateDistributionParams(promptTokensMin, promptTokensMax, outputTokensMin, outputTokensMax,
				promptTokensStdev, outputTokensStdev, promptTokensMean, outputTokensMean); msg != "" {
				logrus.Fatalf("%s", msg)
			}
			concurrencyNumRequests := 0
			if cmd.Flags().Changed("num-requests") {
				concurrencyNumRequests = numRequests
			}
			spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
				Concurrency: concurrency, ThinkTimeMs: thinkTimeMs,
				NumRequests: concurrencyNumRequests, PrefixTokens: prefixTokens,
				PromptTokensMean: promptTokensMean, PromptTokensStdDev: promptTokensStdev,
				PromptTokensMin: promptTokensMin, PromptTokensMax: promptTokensMax,
				OutputTokensMean: outputTokensMean, OutputTokensStdDev: outputTokensStdev,
				OutputTokensMin: outputTokensMin, OutputTokensMax: outputTokensMax,
			})
			spec.Seed = seed
		} else if workloadType == "distribution" {
			// Distribution mode → synthesize v2 spec from CLI flags
			if err := check.PositiveFinite("--rate", rate); err != nil {
				logrus.Fatalf("%v", err)
			}
			// R3: Validate distribution token bounds (shared with concurrency mode).
			if msg := validateDistributionParams(promptTokensMin, promptTokensMax, outputTokensMin, outputTokensMax,
				promptTokensStdev, outputTokensStdev, promptTokensMean, outputTokensMean); msg != "" {
				logrus.Fatalf("%s", msg)
			}
			spec = workload.SynthesizeFromDistribution(workload.DistributionParams{
				Rate: rate, NumRequests: numRequests, PrefixTokens: prefixTokens,
				PromptTokensMean: promptTokensMean, PromptTokensStdDev: promptTokensStdev,
				PromptTokensMin: promptTokensMin, PromptTokensMax: promptTokensMax,
				OutputTokensMean: outputTokensMean, OutputTokensStdDev: outputTokensStdev,
				OutputTokensMin: outputTokensMin, OutputTokensMax: outputTokensMax,
			})
			spec.Seed = seed
		} else {
			// Preset name (chatbot, summarization, etc.) → synthesize v2 spec
			if err := check.PositiveFinite("--rate", rate); err != nil {
				logrus.Fatalf("%v", err)
			}
			// #1769: the preset is read from <catalog>/workloads/<name>.yaml, through the
			// same reader `blis convert preset` uses.
			wl, presetErr := loadPresetWorkload(workloadType)
			if presetErr != nil {
				logrus.Fatalf("--workload %q could not be resolved: %v\n"+
					"  (or supply the workload directly with --workload-spec)", workloadType, presetErr)
			}
			spec = workload.SynthesizeFromPreset(workloadType, wl.toPresetConfig(), rate, numRequests)
			spec.Seed = seed
			// The preset is the simulated workload, so its acceptance terms are the workload
			// layer (#1939).
			workloadAcceptance = wl.Acceptance
		}

		// Apply per-request timeout to all clients.
		// For synthesized specs, always apply (default 300s matches the session-client default).
		// For file-loaded specs, only apply when the flag is explicitly set.
		if requestTimeoutSecs == 0 {
			logrus.Fatalf("--timeout must be positive (seconds) or negative to disable; got 0")
		}
		// Pre-expand inference-perf / ServeGen specs so the timeout-application step below
		// sees every client, including those expansion populates. Without this, a
		// --workload-spec=inference_perf.yaml run with an explicit --timeout would build
		// streaming states from clients whose Timeout is nil (expansion inside
		// GenerateWorkloadLazy happens after applyTimeoutToSpec runs), producing the default
		// 300 s deadline instead of the requested one (PR #1453 self-review). Covered by
		// TestGenerateWorkloadLazy_InferencePerf_TimeoutAppliedAfterPreExpand. Expansion is
		// idempotent: the generator's validateAndExpandSpec runs it again with no effect, since
		// both branches guard on len(spec.Clients) == 0 (#1441).
		if err := workload.ExpandClientsAndCohorts(spec); err != nil {
			logrus.Fatalf("Failed to expand workload spec: %v", err)
		}
		if workloadSpecPath == "" || cmd.Flags().Changed("timeout") {
			applyTimeoutToSpec(spec, requestTimeoutSecs)
		}

		// Resolve maxRequests: spec.NumRequests as default, CLI --num-requests overrides
		maxRequests := spec.NumRequests
		if cmd.Flags().Changed("num-requests") {
			maxRequests = int64(numRequests)
		}

		// A phased workload is the sum of its phases' requests (#1926).
		if len(spec.Phases) > 0 {
			maxRequests = 0
			for _, ph := range spec.Phases {
				maxRequests += ph.NumRequests
			}
		}

		// Guard against unbounded generation
		if maxRequests <= 0 && simulationHorizon == math.MaxInt64 {
			logrus.Fatalf("Workload requires either num_requests or --horizon to bound generation")
		}

		// Requests are streamed from the workload generator (#1441): every workload class,
		// including multi-session reasoning (#1458), concurrency clients (#1459) and
		// time-varying / per-window workloads (#1460). The source's Err surfaces a terminal
		// sampler or generator error recorded during the run (PR #1453 review round 3).
		var (
			requestSource  requestStream
			sessions       []workload.SessionBlueprint
			followUpBudget int64 = -1
			phaseGate      *workload.PhaseGate
		)
		if len(spec.Phases) > 0 {
			// Drain-gated phases (#1926): the first phase is sent now, each later one when the
			// one before it drains. The count is the phases' own.
			if cmd.Flags().Changed("num-requests") {
				logrus.Fatalf("--num-requests cannot be used with a phased workload: each phase states its num_requests")
			}
			phased, gErr := workload.GeneratePhased(spec)
			if gErr != nil {
				logrus.Fatalf("Failed to build the phased workload: %v", gErr)
			}
			var initial []*sim.Request
			phaseGate, initial, gErr = workload.NewPhaseGate(spec.TracePhases(), phased)
			if gErr != nil {
				logrus.Fatalf("Failed to build the phased workload: %v", gErr)
			}
			maxRequests = int64(len(phased))
			requestSource = phasedSource{cluster.NewSliceRequestSource(initial)}
			logrus.Infof("Phased workload: %d phases, %d of %d requests sent at the start", len(spec.Phases), len(initial), len(phased))
		} else {
			var lazy requestStream
			var gErr error
			lazy, sessions, followUpBudget, gErr = workload.GenerateWorkloadLazy(spec, simulationHorizon, maxRequests)
			if gErr != nil {
				logrus.Fatalf("Failed to build the workload: %v", gErr)
			}
			requestSource = lazy
		}
		wl := &workload.GeneratedWorkload{Sessions: sessions, FollowUpBudget: followUpBudget}
		// Apply the timeout to the session blueprints, which expansion inside the generator may
		// have populated after applyTimeoutToSpec ran.
		if workloadSpecPath == "" || cmd.Flags().Changed("timeout") {
			applyTimeoutToSessions(wl, requestTimeoutSecs)
		}
		if len(wl.Sessions) > 0 {
			sessionMgr = workload.NewSessionManager(wl.Sessions)
			if wl.FollowUpBudget >= 0 {
				sessionMgr.SetFollowUpBudget(wl.FollowUpBudget)
			}
			logrus.Infof("Generated a streaming source and %d session blueprints (closed-loop)", len(wl.Sessions))
		} else {
			logrus.Infof("Generated a streaming workload source (#1441)")
		}

		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}
		// Changed() guard: unlike peer flags (default always positive), --horizon defaults
		// to math.MaxInt64 which would fail <= 0. Only validate when user explicitly sets it.
		if cmd.Flags().Changed("horizon") && simulationHorizon <= 0 {
			logrus.Fatalf("--horizon must be > 0, got %d", simulationHorizon)
		}

		// Resolve policy configuration (single code path shared with replayCmd).
		bundle := resolvePolicies(cmd)

		// Resolve autoscaler and node pool config from policy bundle, then apply CLI overrides.
		var (
			bundleAutoscalerIntervalUs           float64
			bundleScaleUpStabilizationWindowUs   float64
			bundleScaleDownStabilizationWindowUs float64
			bundleHPAScrapeDelayMean             float64
			bundleHPAScrapeDelayStddev           float64
			bundleAnalyzerCfg                    cluster.V2SaturationAnalyzerConfig
			bundleAnalyzerSel                    cluster.AnalyzerSelection
			bundleCollectorWindow                int
			bundleScaleToZeroIdleUs              float64
			bundleScaleFromZeroHoldUs            float64
			bundleScaleFromZeroPollUs            float64
			bundleNodePools                      []cluster.NodePoolConfig
			poolChips                            map[string]*poolChip
			bundleInstanceLifecycle              cluster.InstanceLifecycleConfig
		)
		if bundle != nil {
			if bundle.Autoscaler.IntervalUs > 0 {
				bundleAutoscalerIntervalUs = bundle.Autoscaler.IntervalUs
				bundleScaleToZeroIdleUs = bundle.Autoscaler.ScaleToZeroIdleUs
				bundleScaleFromZeroHoldUs = bundle.Autoscaler.ScaleFromZeroHoldUs
				bundleScaleFromZeroPollUs = bundle.Autoscaler.ScaleFromZeroPollUs
				bundleScaleUpStabilizationWindowUs = bundle.Autoscaler.ScaleUpStabilizationWindowUs
				bundleScaleDownStabilizationWindowUs = bundle.Autoscaler.ScaleDownStabilizationWindowUs
				bundleHPAScrapeDelayMean = bundle.Autoscaler.HPAScrapeDelay.Mean
				bundleHPAScrapeDelayStddev = bundle.Autoscaler.HPAScrapeDelay.Stddev
				bundleCollectorWindow = bundle.Autoscaler.CollectorWindowSize
				bundleAnalyzerSel = cluster.AnalyzerSelection{Kind: bundle.Autoscaler.Analyzer.Kind,
					Signal: bundle.Autoscaler.Analyzer.Signal, Target: bundle.Autoscaler.Analyzer.Target}
				if err := bundleAnalyzerSel.Validate(); err != nil {
					logrus.Fatalf("policy bundle: %v", err)
				}
				bundleAnalyzerCfg = cluster.V2SaturationAnalyzerConfig{
					KvCacheThreshold:  bundle.Autoscaler.Analyzer.KVCacheThreshold,
					ScaleUpThreshold:  bundle.Autoscaler.Analyzer.ScaleUpThreshold,
					ScaleDownBoundary: bundle.Autoscaler.Analyzer.ScaleDownBoundary,
					AvgInputTokens:    bundle.Autoscaler.Analyzer.AvgInputTokens,
					MinReplicas:       bundle.Autoscaler.Analyzer.MinReplicas,
				}
			}
			// Each pool's chip is the scenario's (cluster.pools, SIM-17); a pool of a chip other
			// than the scenario's own opens a variant of the scenario for it.
			poolChips = openPoolChips(bundle.NodePools)
			for _, np := range bundle.NodePools {
				bundleNodePools = append(bundleNodePools, cluster.NodePoolConfig{
					Name:         np.Name,
					GPUType:      np.GPUType,
					GPUsPerNode:  np.GPUsPerNode,
					GPUMemoryGiB: np.GPUMemoryGiB,
					InitialNodes: np.InitialNodes,
					MinNodes:     np.MinNodes,
					MaxNodes:     np.MaxNodes,
					ProvisioningDelay: cluster.DelaySpec{
						Mean:   np.ProvisioningDelay.Mean,
						Stddev: np.ProvisioningDelay.Stddev,
					},
					CostPerHour: np.CostPerHour,
				})
			}
			bundleInstanceLifecycle = cluster.InstanceLifecycleConfig{
				LoadingDelay: cluster.DelaySpec{
					Mean:   bundle.InstanceLifecycle.LoadingDelay.Mean,
					Stddev: bundle.InstanceLifecycle.LoadingDelay.Stddev,
				},
				WarmStartInitialInstances: bundle.InstanceLifecycle.WarmStartInitialInstances,
				WeightSource:              bundle.InstanceLifecycle.WeightSource,
				DrainPolicy:               bundle.InstanceLifecycle.DrainPolicy,
			}
			if p := bundle.InstanceLifecycle.DrainPolicy; p != "" && !cluster.IsValidDrainPolicy(p) {
				logrus.Fatalf("policy bundle instance_lifecycle.drain_policy %q is not one of IMMEDIATE, WAIT, REDIRECT", p)
			}
			requireWeightSource(bundleInstanceLifecycle.WeightSource, kernelOpened, poolChips)
		} else {
			// No bundle places instances in the scenario's pools: refuse pool chips it would ignore.
			openPoolChips(nil)
		}
		// CLI flag overrides bundle value when explicitly set.
		if cmd.Flags().Changed("model-autoscaler-interval-us") {
			bundleAutoscalerIntervalUs = modelAutoscalerIntervalUs
		}

		// PD disaggregation validation (R3: validate at CLI boundary)
		if prefillInstances < 0 {
			logrus.Fatalf("--prefill-instances must be >= 0, got %d", prefillInstances)
		}
		if decodeInstances < 0 {
			logrus.Fatalf("--decode-instances must be >= 0, got %d", decodeInstances)
		}
		if prefillDecodeInstances < 0 {
			logrus.Fatalf("--prefill-decode-instances must be >= 0, got %d", prefillDecodeInstances)
		}
		if err := cluster.ValidatePoolTopology(prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances, numInstances); err != nil {
			logrus.Fatalf("Invalid PD pool topology: %v", err)
		}
		if encodeInstances < 0 {
			logrus.Fatalf("--encode-instances must be >= 0, got %d", encodeInstances)
		}

		// Each P/D role's engine is its own pool's, from its own kernel.
		kernelPD := openKernelPools()
		if kernelPD != nil {
			prefillOverrides, decodeOverrides = kernelPD.overrides()
			if traceOutput != "" {
				if err := refuseUnrecordableOffload(kvOffloadCfg, decodeOffloadCfg); err != nil {
					logrus.Fatalf("--trace-output: %v", err)
				}
			}
		}

		// LoRA control-plane (#1464). loraCfg was resolved once at the top of RunE (for
		// the KV HBM reservation); here we cross-validate every workload adapter
		// reference against the declared registry (unknown id / base-model mismatch =>
		// Fatalf, never a silent no-op). With no adapters and no workload adapter
		// references this is inert (INV-6).
		if err := workload.ValidateAdapterReferences(spec, adapterRegistryOf(loraCfg)); err != nil {
			logrus.Fatalf("LoRA workload validation: %v", err)
		}

		startTime := time.Now() // Get current time (start)

		if err := refuseDPAutoscaler(lr.ModelConfig.IsMoE(), dataParallelism, bundleAutoscalerIntervalUs > 0); err != nil {
			logrus.Fatalf("%v", err)
		}

		// Log configuration after all config sources (CLI, workload spec, policy bundle) are
		// resolved, matching the equivalent line in cmd/replay.go.
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks", totalKVBlocks, simulationHorizon)

		// #1590 (H1): derive per_block_bytes for the offload tier chain -- the kernel's
		// per-rank KV byte size of one GPU block -- and record it on the resolved offload
		// config. It feeds the CPU-tier block capacity and transfer-job sizing, and
		// round-trips through the trace header (INV-13). Only when offload is enabled.
		sizeOffloadBlocks(&kvOffloadCfg, false)

		// The kernel prices every offload transfer: the secondary tiers through TierTime and
		// the legacy CPU tier as one whole per-block reload charge. Shared with replayCmd
		// through one helper (R23, INV-13).
		applyKernelOffloadPricing(&kvOffloadCfg)
		// A disaggregated run's pools each size and price the offload by their own kernel.
		if kernelPD != nil {
			kernelPD.applyOffload(kvOffloadCfg, decodeOffloadCfg, &prefillOverrides, &decodeOverrides)
		}

		// Unified cluster path (used for all values of numInstances).
		// INV-13 SYNC POINT: PD fields below must stay in sync with cmd/replay.go (replayCmd
		// DeploymentConfig literal). See .github/agent-prompts/standards/invariants.md INV-13.
		// The instance counts are final: every instance must fit
		// its pool, and the P/D handoff pricer places instances by these counts.
		requireKernelCapacity(kernelPD)
		requirePullPricing(kernelPD, pdTransferContention)
		scenarioTopology, topologyErr := scenarioEndpointTopology(kernelOpened, kernelPD, numInstances, prefillInstances, decodeInstances)
		if topologyErr != nil {
			logrus.Fatalf("scenario endpoint placement: %v", topologyErr)
		}
		ec := engineConfigFor(cmd, lr.ModelConfig, dataParallelism, kvOffloadCfg, loraCfg)
		// Pools that draft differently each advance by their own speculation (inference-sim#2125); as in
		// the other command (INV-13).
		adoptPoolSpeculation(&prefillOverrides, &decodeOverrides)
		runEngine := engineConfigOfRun(ec, simulationHorizon, seed, lr.KernelModel, sloPriorityOverrides)
		config := cluster.DeploymentConfig{
			InstanceConfig:                 instanceConfigOf(runEngine),
			NumInstances:                   numInstances,
			DPLoadBalancing:                dpLoadBalancing,
			TraceLevel:                     traceLevel,
			CounterfactualK:                counterfactualK,
			PrefillInstances:               prefillInstances,
			DecodeInstances:                decodeInstances,
			SharedInstances:                prefillDecodeInstances,
			EncodeInstances:                encodeInstances,
			InstanceEndpoint:               kernelPD.instanceEndpoint(),
			PeerEndpoint:                   peerEndpoint(kernelPD, kernelOpened),
			PeerMessageLatency:             peerMessageLatencyUs,
			PDTransferContention:           pdTransferContention,
			PDPrefillKVHold:                kernelPD.holdsPrefillKV(),
			PDPrefillSamplesFirstToken:     true,
			Roles:                          rolesOf(runEngine, prefillOverrides, decodeOverrides),
			ModelAutoscalerIntervalUs:      bundleAutoscalerIntervalUs,
			ScaleToZeroIdleUs:              bundleScaleToZeroIdleUs,
			ScaleFromZeroHoldUs:            bundleScaleFromZeroHoldUs,
			ScaleFromZeroPollUs:            bundleScaleFromZeroPollUs,
			ScaleUpStabilizationWindowUs:   bundleScaleUpStabilizationWindowUs,
			ScaleDownStabilizationWindowUs: bundleScaleDownStabilizationWindowUs,
			HPAScrapeDelay:                 cluster.DelaySpec{Mean: bundleHPAScrapeDelayMean, Stddev: bundleHPAScrapeDelayStddev},
			AutoscalerAnalyzerConfig:       bundleAnalyzerCfg,
			AutoscalerAnalyzer:             bundleAnalyzerSel,
			CollectorWindowSize:            bundleCollectorWindow,
			NodePools:                      bundleNodePools,
			ScenarioEndpointTopology:       cluster.StaticEndpointTopology(scenarioTopology),
			InstanceLifecycle:              bundleInstanceLifecycle,
			WeightLoadTime:                 weightLoader(kernelOpened, poolChips),
			ClusterAutoscaler:              clusterAutoscalerOf(bundle),
			NewEngine: withPoolChips(engineDriverFactory(scenarioEngine, simulationHorizon, seed, lr.KernelModel, ec.Speculative.Acceptance),
				poolChips, scenarioEngine, simulationHorizon, seed, ec.Speculative.Acceptance),
		}
		stateInstanceGPUs(&config)
		// Session callback installation (Constraint 3 fix):
		// Follow-up collection must be UNCONDITIONAL for saturation analysis correctness.
		// The TraceV2 export (lines 1582-1601) remains gated on --trace-output, but the
		// follow-up accumulation happens regardless so saturation analysis sees complete workloads.
		var followUpRequests []*sim.Request
		// Each follow-up's think time (arrival minus the clock of the round it follows), kept
		// so an accumulate session exports with the exact think its replay needs (#1914).
		thinkUsByReqID := make(map[string]int64)
		noteThink := func(followUps []*sim.Request, clock int64) {
			for _, fu := range followUps {
				if fu.RoundIndex > 0 {
					thinkUsByReqID[fu.ID] = fu.ArrivalTime - clock
				}
			}
		}
		var onRequestDone func(*sim.Request, int64) []*sim.Request
		var unservedHook func(*sim.Request, int64) []*sim.Request
		if sessionMgr != nil {
			// Always install callback to accumulate follow-ups (for saturation analysis + optional trace export)
			onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
				followUps := sessionMgr.OnComplete(req, clock)
				followUpRequests = append(followUpRequests, followUps...)
				noteThink(followUps, clock)
				return followUps
			}
			// Route pre-instance drops (gateway client-deadline expiry, admission/routing
			// rejection, shed/evict) through OnUnserved so a cut-short session is cancelled,
			// matching served completions (#2064 review, run/replay parity). OnUnserved
			// drop-marks a copy so an evicted request's StateCompleted tombstone is not
			// misread as a successful round.
			unservedHook = func(req *sim.Request, clock int64) []*sim.Request {
				followUps := sessionMgr.OnUnserved(req, clock)
				followUpRequests = append(followUpRequests, followUps...)
				noteThink(followUps, clock)
				return followUps
			}
		}
		cs, csErr := cluster.BuildClusterSimulator(config, requestSource, onRequestDone)
		if csErr != nil {
			// A startup refusal: the scenario names a combination the engine does not build.
			logrus.Fatalf("%v", csErr)
		}
		if unservedHook != nil {
			cs.SetUnservedHook(unservedHook)
		}
		if sessionMgr != nil {
			cs.SetPlannedRounds(sessionMgr.PlannedUnissued)
		}
		if phaseGate != nil {
			cs.SetGate(phaseGate)
		}
		router := resolveRouterInputs()
		gw := newGateway(router, model, adapterIDsOf(loraCfg)...)
		cs.SetGateway(gw)

		// Arrival hook: capture trace-emission references at the cluster's
		// single arrival boundary so the trace exporter no longer relies on
		// the eager preGeneratedRequests + followUpRequests list assembly
		// (issue #1440). The hook fires once per fresh arrival in
		// clock-monotonic order — see ClusterArrivalEvent.Execute. We hold
		// pointers (not copies) so the request's final state (set by the
		// event loop) is visible at export time.
		//
		// Only install when --trace-output is set (BC-1: zero overhead when
		// trace is disabled).
		//
		// Install/export coupling: traceArrivals is declared nil here and
		// assigned a non-nil empty slice ONLY inside the install branch.
		// A nil traceArrivals at the export site below with traceOutput
		// non-empty means the install branch was dropped — we fail loudly
		// rather than write a silent empty trace (R1).
		// The arrival hook captures fresh-arrival references at the single
		// cluster boundary. It powers trace export (#1440). As of #1516 the
		// saturation trace is streamed over BuildOutput's completed-request
		// metrics, not over the arrival list, so the hook is needed only for
		// --trace-output (BC-1 zero overhead otherwise).
		var traceArrivals []*sim.Request
		arrivalHookNeeded := traceOutput != ""
		if arrivalHookNeeded {
			traceArrivals = make([]*sim.Request, 0)
			cs.SetArrivalHook(func(req *sim.Request) {
				traceArrivals = append(traceArrivals, req)
			})
		}

		// Resolve the saturation tracer from --detectors /
		// --saturation-report BEFORE the run so an unknown name, bad config, or
		// unwritable report path fails fast (#1516 single detector, #1519 bank).
		satTracer, satErr := resolveSaturation()
		if satErr != nil {
			logrus.Fatalf("%v", satErr)
		}

		if err := cs.Run(); err != nil {
			logrus.Fatalf("Simulation failed: %v", err)
		}

		// Surface any terminal sampler / generator error the source recorded on a per-client
		// state during the run; without this check the run would exit 0 with reduced traffic
		// and misleading capacity numbers (PR #1453 review round 3).
		if err := requestSource.Err(); err != nil {
			logrus.Fatalf("Workload sampler failure: %v", err)
		}

		// Cancelled-session count (#2064, BC-5): parity with blis replay. A timed-out or
		// dropped round cancels its whole session, so every later round is never issued;
		// the aggregate timed-out-request count alone understates the lost work.
		if sessionMgr != nil {
			if cancelled := sessionMgr.CancelledSessions(); cancelled > 0 {
				logrus.Warnf("[run] %d session(s) were cancelled before serving every round (a round timed out or was dropped); later rounds of those sessions were never issued. Pass --timeout to lengthen or disable the deadline.",
					cancelled)
			}
		}

		// Wall-clock timing on stderr (BC-6); stdout remains deterministic (BC-7)
		logrus.Infof("Simulation wall-clock time: %.3fs", time.Since(startTime).Seconds())

		// Resolve goodput SLO targets early so the trace export and aggregate metrics
		// see the same merged map (#1413, BC-1). Run has no trace header; precedence
		// here is CLI > workload spec.
		cliTTFT, cliITL, cliE2E, gpErr := resolveGoodputCLIFlags(goodputSLOTTFT, goodputSLOITL, goodputSLOE2E)
		if gpErr != nil {
			logrus.Fatalf("%v", gpErr)
		}
		var specTargets map[string]workload.SLODimTargets
		if spec != nil {
			specTargets = spec.GoodputSLOTargets
		}
		goodputTargets := mergeGoodputTargets(cliTTFT, cliITL, cliE2E, nil, specTargets)

		// Export trace if requested (BC-1, BC-7). Records are sourced from
		// the arrival hook (issue #1440) — already in clock-monotonic order
		// per INV-3, so no sort is required.
		if traceOutput != "" {
			// Install/export coupling guard (R1): traceArrivals is a non-nil
			// empty slice when SetArrivalHook ran above. A nil here means
			// the install branch was dropped or moved without updating this
			// site — refuse to write a silent empty trace.
			if traceArrivals == nil {
				logrus.Fatalf("Trace export: arrival hook was not installed but --trace-output=%q is set — install/export branches diverged", traceOutput)
			}
			records := workload.RequestsToTraceRecords(traceArrivals)
			// #1914 (INV-13): accumulate sessions export as per-round deltas under a header that
			// says so; absolute rounds would replay without the reuse of the previous round's
			// context. The re-export is also what fixed-accumulate replay reads.
			var growth tracespec.ContextGrowth
			if allSessionsAccumulate(wl) {
				var rErr error
				records, rErr = workload.ReExportClosedLoopRecords(traceArrivals, thinkUsByReqID, "accumulate")
				if rErr != nil {
					logrus.Fatalf("Trace export of accumulate sessions: %v", rErr)
				}
				growth = tracespec.ContextGrowth("accumulate")
			}
			if growth == "" && workload.OpenLoopAccumulate(spec) {
				// Open-loop accumulate sessions export as deltas too: absolute rounds would
				// replay without the previous round's context (#1914).
				var rErr error
				records, rErr = workload.ReExportOpenLoopAccumulateRecords(traceArrivals)
				if rErr != nil {
					logrus.Fatalf("Trace export of open-loop accumulate sessions: %v", rErr)
				}
				growth = tracespec.ContextGrowth("accumulate")
			}
			header := &workload.TraceHeader{
				SessionContextGrowth: growth,
				Version:              workload.TraceVersion,
				TimeUnit:             "microseconds",
				Mode:                 "generated",
				WorkloadSeed:         &spec.Seed,
				GoodputSLOTargets:    goodputTargets,                   // #1413, BC-7: persist resolved targets for downstream replay
				KVOffload:            simToHeaderOffload(kvOffloadCfg), // #1587, BC-G6: nil when inert (omitted); round-trips resolved config
				// #2146: record multi-node placement so replay can refuse a trace whose
				// step times it cannot reproduce. 0/1 (no span) is omitted, so a run
				// without multi-node placement writes a byte-identical header (INV-6).
				MaxNodesSpanned: crossNodeSpanForTrace(cs.MaxNodesSpanned()),
				// INV-13: the router the run routed with, so a replay rebuilds it.
				Router: router.header(gw.Seed()),
				// #1926: the stages the run ran in, so a replay sends them drain-gated too.
				Phases: spec.TracePhases(),
				// #2115 (DEV-7): the workload's acceptance terms, so a replay resolves the
				// profile this run did. Nil, and so omitted, when the workload states none.
				Acceptance: workloadAcceptance,
			}
			if err := workload.ExportTraceV2(header, records, traceOutput+".yaml", traceOutput+".csv"); err != nil {
				logrus.Fatalf("Trace export failed: %v", err)
			}
			logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", traceOutput, traceOutput, len(records))
		}

		// The Result document (#1871): stdout is exactly one blis-schemas Result, and
		// --output the same bytes. The run's figures and --view summary come from the one path
		// replay shares (INV-13).
		// With no cap the run generated what arrived before its horizon. blis-schemas requires a
		// positive bound, so an uncapped run in which nothing arrived records 1, the smallest
		// bound.
		runNumRequests := max(cs.Arrivals(), 1)
		if maxRequests > 0 {
			runNumRequests = int(maxRequests)
		}
		run := runControls{
			Load:        runLoadOf(spec, arrivalTimes(cs.AggregatedMetrics())),
			NumRequests: runNumRequests,
			// --seed seeds the engines and the router; a workload spec run without --seed keeps
			// the spec's own seed for its requests, which the spec file's sha256 (overrides,
			// workload-spec) pins and the workload_seed override row states (#1871 S8).
			Seed:    seed,
			Binding: runWorkloadBinding(),
		}
		var artifacts []artifactFile
		if traceOutput != "" {
			artifacts = traceArtifacts(traceOutput, len(traceArrivals))
		}
		var extra []sim.ScenarioOverride
		if spec != nil && spec.Seed != seed {
			extra = append(extra, sim.ScenarioOverride{Setting: "workload_seed", Scenario: fmt.Sprint(seed),
				Run: fmt.Sprint(spec.Seed), Source: "--workload-spec (its seed: field; --seed not given)"})
		}
		emitRunResult(postRun{
			cs: cs, gateway: gw, run: run, goodputTargets: goodputTargets, satTracer: satTracer, extraOverrides: extra,
			artifacts:      artifacts,
			fitnessWeights: fitnessWeights, started: startTime,
			frame: runFrameGenerator(spec), numRequestsCap: int(max(maxRequests, 0)),
		})

		logrus.Info("Simulation complete.")
	},
}

// runFrameGenerator is what a run tells the experimental frame about its requests: the
// generator that made them, the workload's identity, and the seed its draws used.
func runFrameGenerator(spec *workload.WorkloadSpec) frameGenerator {
	g := frameGenerator{Seed: seed}
	if spec != nil {
		g.Seed = spec.Seed
	}
	switch {
	case workloadSpecPath != "":
		g.Kind = generatorWorkloadSpec
		if d, err := fileDigest(workloadSpecPath); err == nil {
			g.Source = "sha256:" + d
		} else {
			logrus.Fatalf("--workload-spec %s: %v", workloadSpecPath, err)
		}
	case concurrency > 0:
		g.Kind = generatorConcurrency
	case workloadType == "distribution":
		g.Kind = generatorDistribution
	default:
		g.Kind, g.Source = generatorPreset, workloadType
	}
	return g
}

// runLoadOf is the offered load a run generated: --concurrency, a workload spec's own clients
// or rate, or --rate. arrivals are the run's arrival times, for a spec that states no single
// rate.
func runLoadOf(spec *workload.WorkloadSpec, arrivals []int64) measurement.Load {
	switch {
	case workloadSpecPath != "":
		return specLoad(spec, arrivals)
	case concurrency > 0:
		return measurement.Load{Concurrency: concurrency}
	default:
		return measurement.Load{RequestRate: rate}
	}
}

// arrivalTimes are the arrival times, in ticks, of every request the run registered.
func arrivalTimes(m *sim.Metrics) []int64 {
	out := make([]int64, 0, len(m.Requests))
	for _, rm := range m.Requests {
		out = append(out, int64(math.Round(rm.ArrivedAt*1e6)))
	}
	return out
}

// runWorkloadBinding is the workload a run used as a blis-schemas binding: a named catalog
// preset is a shape; a --workload-spec file or the distribution flags are recorded by the
// load and the run's own workload, which no shape names, so nil.
func runWorkloadBinding() *schemaworkload.Binding {
	if workloadSpecPath == "" && concurrency == 0 && workloadType != "distribution" {
		return &schemaworkload.Binding{Shape: workloadType}
	}
	return nil
}

// printKVCacheMetrics prints KV cache metrics to w when any value is nonzero.
func printKVCacheMetrics(w io.Writer, preemptionRate, cacheHitRate, kvThrashingRate float64) {
	if preemptionRate == 0 && cacheHitRate == 0 && kvThrashingRate == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== KV Cache Metrics ===")
	_, _ = fmt.Fprintf(w, "Preemption Rate: %.4f\n", preemptionRate)
	_, _ = fmt.Fprintf(w, "Cache Hit Rate: %.4f\n", cacheHitRate)
	_, _ = fmt.Fprintf(w, "KV Thrashing Rate: %.4f\n", kvThrashingRate)
}

// printPerSLOMetrics prints per-SLO-class latency distributions. Without
// goodput targets configured, the section is suppressed for ≤1 class (the
// legacy no-spurious-section behavior). With goodput targets configured, the
// section prints even for a single class so operators see the dimensions
// gating their goodput score (#1413, BC-5).
func printPerSLOMetrics(w io.Writer, sloMetrics map[string]*cluster.SLOMetrics, goodputConfigured bool) {
	if len(sloMetrics) == 0 {
		return
	}
	if len(sloMetrics) == 1 && !goodputConfigured {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-SLO Metrics ===")
	keys := make([]string, 0, len(sloMetrics))
	for k := range sloMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, cls := range keys {
		m := sloMetrics[cls]
		if m == nil {
			continue
		}
		_, _ = fmt.Fprintf(w, "  %s:\n", cls)
		_, _ = fmt.Fprintf(w, "    TTFT: mean=%.2f p99=%.2f (n=%d)\n", m.TTFT.Mean, m.TTFT.P99, m.TTFT.Count)
		_, _ = fmt.Fprintf(w, "    ITL:  mean=%.2f p99=%.2f (n=%d)\n", m.ITL.Mean, m.ITL.P99, m.ITL.Count)
		_, _ = fmt.Fprintf(w, "    E2E:  mean=%.2f p99=%.2f (n=%d)\n", m.E2E.Mean, m.E2E.P99, m.E2E.Count)
	}
}

// printPerModelMetrics prints per-model TTFT, E2E, and throughput.
// Follows the same pattern as printPerSLOMetrics (R2: sorted keys).
// No-op when perModelMetrics is nil or empty.
func printPerModelMetrics(w io.Writer, perModelMetrics map[string]*cluster.ModelMetrics) {
	if len(perModelMetrics) == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-Model Metrics ===")
	keys := make([]string, 0, len(perModelMetrics))
	for k := range perModelMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, model := range keys {
		m := perModelMetrics[model]
		if m == nil {
			continue
		}
		_, _ = fmt.Fprintf(w, "  %s:\n", model)
		_, _ = fmt.Fprintf(w, "    TTFT: p50=%.2f p99=%.2f (n=%d)\n", m.TTFT.P50, m.TTFT.P99, m.TTFT.Count)
		_, _ = fmt.Fprintf(w, "    E2E:  p50=%.2f p99=%.2f (n=%d)\n", m.E2E.P50, m.E2E.P99, m.E2E.Count)
		_, _ = fmt.Fprintf(w, "    Throughput: %.2f req/s, %.2f tok/s\n", m.ThroughputRPS, m.ThroughputTokenSec)
	}
}

// printPerTenantMetrics prints per-tenant request counts, token totals, and Jain fairness index.
// Follows the same pattern as printPerModelMetrics (R2: sorted keys).
// No-op when perTenantMetrics is nil or empty.
func printPerTenantMetrics(w io.Writer, perTenantMetrics map[string]*cluster.TenantMetrics) {
	if len(perTenantMetrics) == 0 {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Per-Tenant Metrics ===")
	keys := make([]string, 0, len(perTenantMetrics))
	for k := range perTenantMetrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	tokenMap := make(map[string]float64, len(perTenantMetrics))
	for _, tid := range keys {
		tm := perTenantMetrics[tid]
		_, _ = fmt.Fprintf(w, "  %s: requests=%d, tokens=%d\n", tid, tm.CompletedRequests, tm.TotalTokensServed)
		tokenMap[tid] = float64(tm.TotalTokensServed)
	}
	jain := cluster.JainFairnessIndex(tokenMap)
	_, _ = fmt.Fprintf(w, "  Jain Fairness Index: %.4f\n", jain)
}

// printSessionMetrics writes the session metrics section to w.
// No-op when sm is nil (single-turn workloads produce no session output).
func printSessionMetrics(w io.Writer, sm *cluster.SessionMetrics) {
	if sm == nil {
		return
	}
	_, _ = fmt.Fprintln(w, "=== Session Metrics ===")
	_, _ = fmt.Fprintf(w, "  Sessions: %d\n", sm.SessionCount)
	if sm.TTFTCold.Count > 0 {
		_, _ = fmt.Fprintf(w, "  TTFT cold (round 0): mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.TTFTCold.Mean, sm.TTFTCold.P50, sm.TTFTCold.P95, sm.TTFTCold.P99, sm.TTFTCold.Count)
	}
	if sm.TTFTWarm.Count > 0 {
		_, _ = fmt.Fprintf(w, "  TTFT warm (round≥1): mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.TTFTWarm.Mean, sm.TTFTWarm.P50, sm.TTFTWarm.P95, sm.TTFTWarm.P99, sm.TTFTWarm.Count)
	}
	if sm.SessionDuration.Count > 0 {
		_, _ = fmt.Fprintf(w, "  Session duration:    mean=%.2f p50=%.2f p95=%.2f p99=%.2f ms (n=%d)\n",
			sm.SessionDuration.Mean, sm.SessionDuration.P50, sm.SessionDuration.P95, sm.SessionDuration.P99, sm.SessionDuration.Count)
	}
}

// printPDMetrics prints the PD disaggregation metrics section when disaggregation was active.
// No-op when pd is nil (disaggregation inactive).
func printPDMetrics(w io.Writer, pd *cluster.PDMetrics) {
	if pd == nil {
		return
	}
	_, _ = fmt.Fprintln(w, "=== PD Metrics ===")
	_, _ = fmt.Fprintf(w, "Disaggregated Requests: %d\n", pd.DisaggregatedCount)
	_, _ = fmt.Fprintf(w, "Dropped at Decode KV: %d\n", pd.DroppedAtDecodeKV)
	_, _ = fmt.Fprintf(w, "Prefill Throughput: %.4f sub-req/s\n", pd.PrefillThroughput)
	_, _ = fmt.Fprintf(w, "Decode Throughput: %.4f sub-req/s\n", pd.DecodeThroughput)
	if pd.LoadImbalanceRatio == math.MaxFloat64 {
		_, _ = fmt.Fprintf(w, "Load Imbalance Ratio: inf (one pool idle)\n")
	} else {
		_, _ = fmt.Fprintf(w, "Load Imbalance Ratio: %.4f\n", pd.LoadImbalanceRatio)
	}
	if pd.ParentTTFT.Count > 0 {
		_, _ = fmt.Fprintf(w, "Parent TTFT (us): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
			pd.ParentTTFT.Mean, pd.ParentTTFT.P50, pd.ParentTTFT.P95, pd.ParentTTFT.P99)
	}
	if pd.TransferDuration.Count > 0 {
		_, _ = fmt.Fprintf(w, "KV Transfer Duration (us): mean=%.1f p50=%.1f p95=%.1f p99=%.1f\n",
			pd.TransferDuration.Mean, pd.TransferDuration.P50, pd.TransferDuration.P95, pd.TransferDuration.P99)
	}
	// The lines below print only when their feature is active, so a run without contention,
	// hold or CPU-tier hits prints the section it always did.
	if pd.LinkPeakConcurrent != nil && pd.TransferQueueDelay.Count > 0 {
		_, _ = fmt.Fprintf(w, "KV Transfer Queueing Delay (us): mean=%.1f p50=%.1f p90=%.1f p99=%.1f\n",
			pd.TransferQueueDelay.Mean, pd.TransferQueueDelay.P50, pd.TransferQueueDelayP90, pd.TransferQueueDelay.P99)
		links := make([]string, 0, len(pd.LinkPeakConcurrent))
		for l := range pd.LinkPeakConcurrent {
			links = append(links, l)
		}
		sort.Strings(links)
		for _, l := range links {
			_, _ = fmt.Fprintf(w, "  Peak Concurrent Transfers (%s): %d\n", l, pd.LinkPeakConcurrent[l])
		}
	}
	if pd.PrefillKVHoldPeakTokens > 0 {
		_, _ = fmt.Fprintf(w, "Prefill KV Hold Peak (tokens): %d\n", pd.PrefillKVHoldPeakTokens)
	}
	for _, pool := range []string{"prefill", "decode"} {
		if r := pd.CPUTierHitRate[pool]; r > 0 {
			_, _ = fmt.Fprintf(w, "CPU Tier Hit Rate (%s pool): %.4f\n", pool, r)
		}
	}
}

// Execute runs the CLI root command
func Execute() {
	if err := rootCmd.Execute(); err != nil {
		var code exitCode
		if errors.As(err, &code) {
			os.Exit(int(code))
		}
		os.Exit(1)
	}
}

// init sets up CLI flags and subcommands
func init() {
	registerSimConfigFlags(runCmd)

	// Workload generation flags (run-only)
	runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution)")

	registerIntervalFlags(runCmd)
	runCmd.Flags().Float64Var(&rate, "rate", 1.0, "Requests arrival per second")
	runCmd.Flags().IntVar(&numRequests, "num-requests", 100, "Number of requests to generate")
	runCmd.Flags().IntVar(&concurrency, "concurrency", 0, "Number of concurrent virtual users (closed-loop, mutually exclusive with --rate)")
	runCmd.Flags().IntVar(&thinkTimeMs, "think-time-ms", 0, "Think time in ms between response and next request (concurrency mode)")
	runCmd.Flags().IntVar(&prefixTokens, "prefix-tokens", 0, "Prefix Token Count")
	runCmd.Flags().IntVar(&promptTokensMean, "prompt-tokens", defaultPromptMean, "Average Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensStdev, "prompt-tokens-stdev", defaultPromptStdev, "Stddev Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMin, "prompt-tokens-min", defaultPromptMin, "Min Prompt Token Count")
	runCmd.Flags().IntVar(&promptTokensMax, "prompt-tokens-max", defaultPromptMax, "Max Prompt Token Count")
	runCmd.Flags().IntVar(&outputTokensMean, "output-tokens", defaultOutputMean, "Average Output Token Count")
	runCmd.Flags().IntVar(&outputTokensStdev, "output-tokens-stdev", defaultOutputStdev, "Stddev Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMin, "output-tokens-min", defaultOutputMin, "Min Output Token Count")
	runCmd.Flags().IntVar(&outputTokensMax, "output-tokens-max", defaultOutputMax, "Max Output Token Count")
	runCmd.Flags().StringVar(&workloadSpecPath, "workload-spec", "", "Path to YAML workload specification file (overrides --workload)")
	runCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "Per-request deadline in seconds (default 300s matches the session-client default in computeDeadline). Negative = disabled; 0 is rejected.")
	runCmd.Flags().StringVar(&goodputSLOTTFT, "slo-ttft", "", "Per-class TTFT goodput thresholds (e.g. \"critical=100ms,standard=500ms\"). Precedence: CLI > trace header > workload spec.")
	runCmd.Flags().StringVar(&goodputSLOITL, "slo-itl", "", "Per-class mean ITL goodput thresholds (e.g. \"critical=50ms,standard=150ms\").")
	runCmd.Flags().StringVar(&goodputSLOE2E, "slo-e2e", "", "Per-class E2E goodput thresholds (e.g. \"critical=5s,standard=30s\").")

	// Run-specific export
	runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")

	// Saturation trace flags (#1516): --detectors + --saturation-report.
	registerDetectorFlags(runCmd)
	// The Result document's encoding, file and summary view (#1871).
	registerResultFlags(runCmd)

	refuseRemovedFlags(runCmd)

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}

// prefixCacheRecords are each opened pool's prefix-hit resolution, keyed by role, for the
// results file.
var prefixCacheRecords map[string]sim.PrefixCacheRecord

// serverSeriesIntervalUs is --server-series-interval-s in ticks (us); 0 when off. A negative
// or sub-microsecond width is refused.
func serverSeriesIntervalUs() int64 {
	if serverSeriesIntervalS < 0 {
		logrus.Fatalf("--server-series-interval-s must be >= 0, got %g", serverSeriesIntervalS)
	}
	us := int64(math.Round(serverSeriesIntervalS * 1e6))
	if serverSeriesIntervalS > 0 && us < 1 {
		logrus.Fatalf("--server-series-interval-s %g is below one microsecond", serverSeriesIntervalS)
	}
	return us
}

// allSessionsAccumulate reports whether the run has session blueprints and every one grows
// its context by accumulation, the case whose trace export must carry deltas (#1914).
func allSessionsAccumulate(wl *workload.GeneratedWorkload) bool {
	if wl == nil || len(wl.Sessions) == 0 {
		return false
	}
	for _, bp := range wl.Sessions {
		if bp.ContextGrowth != "accumulate" {
			return false
		}
	}
	return true
}

// requestStream is a request source that reports a terminal generator error after the run.
type requestStream interface {
	cluster.RequestSource
	Err() error
}

// phasedSource is the first requests of a phased workload; the drain gate sends the rest.
type phasedSource struct{ *cluster.SliceRequestSource }

// Err is nil: a phased workload is generated whole before the run.
func (phasedSource) Err() error { return nil }
