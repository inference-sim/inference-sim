package cmd

import (
	"fmt"
	"io"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/inference-sim/inference-sim/sim/workload"
)

// distDefaults are the shared default values for distribution synthesis flags.
// Both runCmd and observeCmd use these constants so the two commands produce
// identical workload shapes when called with no explicit distribution flags.
// Changing a value here affects both commands simultaneously.
//
// Flags covered:
//
//	--prompt-tokens, --prompt-tokens-stdev, --prompt-tokens-min, --prompt-tokens-max
//	--output-tokens, --output-tokens-stdev, --output-tokens-min, --output-tokens-max
//
// NOT covered: --prefix-tokens (default 0 means "no shared prefix", a feature toggle,
// not a distribution shape parameter).
//
// --num-requests is intentionally NOT shared: run defaults to 100 (safe for quick sims),
// observe defaults to 0 (requires explicit bound — see observe_cmd.go).
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
	sequenceState             sim.KVSequenceState    // each resident sequence's bounded state in pool blocks, from the kernel (#1941)
	maxNumSeqs                int64                  // Maximum number of requests in the Running batch (vLLM: --max-num-seqs)
	maxNumBatchedTokens       int64                  // Maximum total number of tokens across requests in the Running batch (vLLM: --max-num-batched-tokens)
	noEnablePrefixCaching     bool                   // --no-enable-prefix-caching: disable cross-request GPU prefix reuse (vLLM parity, #1867)
	swaBoundedReplay          optionalBool           // --swa-bounded-replay: overrides the resolved DeepSeek-V4.1 SWA bounded replay
	prefixHit                 sim.KVPrefixHit        // how the KV cache groups cut a prefix hit, from the kernel's model
	blockSizeTokens           int64                  // Number of tokens per KV block
	defaultsFilePath          string                 // Path to default constants: the LoRA cost coefficients
	catalogPath               string                 // --catalog: catalog clone root (models/, hardware/, networks/, ...). No default; BLIS_CATALOG is the fallback (#1731)
	resolvedCatalogRoot       string                 // Catalog ROOT this run read (side effect of resolveLatencyConfig); recorded as results-file provenance (#1732)
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
	kernelRegistry            string                 // CLI --registry: blis-registry clone root (required)
	kernelDeploymentExperts   int                    // routed expert count from the model graph; 0 for a dense model
	kernelDeploymentTopK      int                    // routed experts per token from the model graph; 0 for a dense model
	kernelOpened              *engineModel           // the kernel adoptKernelDeployment opened; nil until it runs
	kernelPDOpened            *kernelPools           // the per-role kernels openKernelPools opened; nil for a colocated run
	scenarioScheduler         string                 // the scheduler the scenario's engine states ("" = none)
	scenarioPreemption        string                 // the preemption policy the scenario's engine states ("" = none)
	schedulerSource           string                 // what chose the scheduler: "--scheduler", "--policy-config", "" = the scenario
	preemptionSource          string                 // likewise for the preemption policy
	scenarioDisaggregated     bool                   // the scenario states prefill and decode pools (each pool records its own overrides)
	scenarioOffload           *deployment.Offload    // the scenario deployment's offload block, nil when it states none (#1910)
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

	// Scheduler and preemption config
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
	// scenario's engine.speculative (set when the kernel deployment is adopted); K=0 =>
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
	lazyGeneration   bool   // --lazy-generation: stream requests from generator (alpha, #1441)

	// Tiered KV cache config (PR12)
	kvCPUBlocks           int64
	kvOffloadThreshold    float64
	kvTransferBandwidth   float64 // no flag sets it: the kernel prices the legacy CPU tier (KVTransferTicksPerBlock)
	kvTransferBaseLatency int64   // no flag sets it, for the same reason
	dpLoadBalancing       string  // --dp-load-balancing: internal | hybrid | external (#1909)

	// PD disaggregation config
	prefillInstances       int  // Number of instances dedicated to prefill
	decodeInstances        int  // Number of instances dedicated to decode
	prefillDecodeInstances int  // Number of shared-role instances (both prefill and decode), issue #1276
	pdTransferContention   bool // concurrent P/D handoffs share the links they cross (#1922)

	// E/P/D disaggregation config (GAP-4, issue #1264)
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
	metricsPath           string  // File to write MetricsOutput JSON for blis run (--metrics-path)
	serverSeriesIntervalS float64 // --server-series-interval-s: bin width of the file-only server_series block; 0 = off
	resultsPath           string  // File to write []SimResult JSON for blis replay (--results-path)
	// saturationReport (--saturation-report): per-event verdict trace file (#1516).
	// Declared in saturation.go alongside --detectors / --saturation-config.

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
	KernelModel sim.LatencyModel
	// ModelConfig carries only the model graph's expert geometry, for the MoE gates and the
	// ModelHardwareConfig boundary.
	ModelConfig sim.ModelConfig
}

// dpPlacementPlan describes how an MoE deployment whose scenario states dp N expands
// into real single-node engine replicas (#1531, DP-as-real-placement). The
// zero-expansion case (Active=false, Replicas=1, PerRankDP=dp) covers dp 1, dense models
// (dp>1 rejected earlier in resolveLatencyConfig), and any config planDPPlacement
// declines to expand.
type dpPlacementPlan struct {
	Active    bool // true ⇒ expand into Replicas engine replicas, each configured DP=1
	Replicas  int  // engine replicas per logical --num-instances (dp when Active, else 1)
	PerRankDP int  // DP to configure on each replica (1 when Active, else dp)
	// LockstepRanks is how many consecutive replicas form one vLLM DP group whose ranks
	// step in lockstep behind one endpoint (#1909): dp when Active, else 0.
	LockstepRanks int
}

// dpPlacementInstanceWarnThreshold: warn (not fatal) when DP-as-placement expands
// to more than this many engine replicas, so an accidentally large scenario dp (a typo)
// is surfaced before the run consumes a large amount of memory/time.
const dpPlacementInstanceWarnThreshold = 512

// planDPPlacement decides DP-as-real-placement expansion (for both `blis run`
// and `blis replay` — see resolveDPPlacement) and rejects placement combinations
// #1531 does not yet model. It is pure (no package state) so the decision is
// unit-testable independently of the command wiring that applies it. On error it
// returns the zero plan (Active=false, Replicas=0); callers MUST `logrus.Fatalf`
// on error and not use the plan.
//
// DP-as-placement applies to an MoE model with dp>1 and no autoscaler; each rank is then a
// TP engine sized per-rank (DP=1). vLLM data parallelism is N EngineCores, each holding a
// disjoint share of the requests, so the BLIS equivalent is N real instances, each DP=1 (a
// per-instance /dp factor would double-count). For an MoE model the N ranks are one DP
// group (#1909): they step in lockstep and sit behind one router endpoint, vLLM's internal
// DP load balancer assigning each request to a rank; LockstepRanks carries that grouping to
// the cluster.
//
// PD disaggregation and node pools are SUPPORTED alongside it (#1553, lifting #1531's
// rejections). Neither changes the plan the way the autoscaler would: the PD extension is
// the same per-replica transformation applied to EACH pool (a topology P+D+S+E ≤ total
// becomes P·N+D·N+S·N+E·N ≤ total·N, which preserves the topology inequality), and node
// pools place the N×M replicas through the existing tested per-instance placement path.
// So the plan is identical to the plain (non-PD, non-node-pool) active plan; the per-pool
// count expansion is carried out by applyDPPlacement, not decided here.
//
// Expert parallelism is ALSO allowed alongside it (#1548, lifting #1531's rejection). It
// reserves no extra GPUs — the expert-parallel group IS the N×TP GPUs this placement
// already takes — so the plan is unchanged by it; what EP changes is how experts map onto
// that group, which resolveDPPlacement carries into each replica's latency model as the
// logical EP-group DP width. epOn is therefore not a rejection reason, and is kept
// as a parameter only so the caller's decision and its diagnostics read from one place.
//
// Guarded combination that still fails fast (never silently mis-modeled):
//   - Autoscaler (#1553 DECISION): the semantics of dynamically scaling a dp-expanded
//     population are undefined (add one rank? one whole DP group of N?), and
//     DirectActuator.scaleUp places a single-role instance with no DP-group awareness.
//     Supporting it would ship an untested, ambiguous path — exactly what #1531 guarded
//     against. Rejected with a clear message stating the decision.
//
// Dense dp>1 is rejected earlier (resolveLatencyConfig, cmd/root.go), so here it
// is simply a no-op.
func planDPPlacement(isMoE bool, dp int, epOn, pdActive, autoscalerActive, nodePoolsActive bool) (dpPlacementPlan, error) {
	if !isMoE || dp <= 1 {
		// No expansion ⇒ nothing erases the config's own DP ⇒ no EP width to carry.
		return dpPlacementPlan{Active: false, Replicas: 1, PerRankDP: dp}, nil
	}
	if autoscalerActive {
		return dpPlacementPlan{}, fmt.Errorf("scenario dp > 1 (MoE) is not supported with the model autoscaler (#1553 " +
			"decision): DP-as-placement spawns a fixed set of dp engine replicas, and the semantics of " +
			"dynamically scaling that population (add one rank, or one whole DP group of dp?) are undefined — " +
			"the autoscaler places single-role instances with no DP-group awareness. " +
			"Use a scenario that states dp 1 with the autoscaler, or disable the autoscaler")
	}
	// pdActive / nodePoolsActive are no longer rejection reasons (#1553). They are kept as
	// parameters so a future combination-specific guard has one home.
	_ = pdActive
	_ = nodePoolsActive
	// Expert parallelism reserves no extra GPUs, so the plan is identical either way.
	_ = epOn
	// The dp replicas of one logical instance are one vLLM DP group: an MoE model at dp > 1
	// runs as DPEngineCoreProc ranks that step in lockstep (vllm/v1/engine/core.py:1386-1395
	// at v0.31.0), with or without expert parallelism (use_all2all is true at dp > 1,
	// vllm/config/parallel.py:738-743), behind one API endpoint (#1909).
	return dpPlacementPlan{Active: true, Replicas: dp, PerRankDP: 1, LockstepRanks: dp}, nil
}

// dpPlacementDeployment carries the deployment quantities DP-as-real-placement
// adjusts. It is both the input (pre-expansion) and the output (post-expansion) of
// applyDPPlacement.
//
// The four PD pool counts (#1553) are expanded by the same Replicas factor as
// NumInstances: a PD topology of P prefill + D decode + S shared + E encode instances
// (with P+D+S+E ≤ total) becomes P·N + D·N + S·N + E·N replicas of total·N. Scaling
// every term by the same N preserves ValidatePoolTopology's inequality
// (P·N+D·N+S·N+E·N ≤ total·N), so a topology that passed at dp 1 still passes after
// expansion. They are zero for a non-PD run, so the multiply is a strict no-op there.
type dpPlacementDeployment struct {
	NumInstances  int   // engine replicas (logical --num-instances on the way in)
	TotalKVBlocks int64 // KV blocks per instance: the kernel's per-rank budget, unchanged by placement
	MaxModelLen   int64 // engine max_model_len (0 = unset/unlimited)

	// PD pool counts (#1553), each scaled by Replicas when the plan is active. Zero for
	// a non-PD deployment.
	PrefillInstances int
	DecodeInstances  int
	SharedInstances  int
	EncodeInstances  int
}

// applyDPPlacement applies a DP-as-placement plan to the deployment quantities: it
// expands the instance count and the PD pool counts by the replica factor. It is pure
// (no package state) so the production arithmetic — the exact defect BC-2 guards
// against, no dp² double-count — is directly unit-testable rather than re-implemented
// in a test. On error the deployment is returned UNCHANGED.
//
// The kernel sizes KV blocks and max_model_len per rank, so both pass through unchanged. A
// non-Active plan is the identity.
func applyDPPlacement(plan dpPlacementPlan, dp int, dep dpPlacementDeployment) (dpPlacementDeployment, error) {
	if !plan.Active {
		return dep, nil
	}
	out := dep
	out.NumInstances = dep.NumInstances * plan.Replicas
	// Scale each PD pool count by the same replica factor (#1553). The topology
	// inequality P+D+S+E ≤ total is preserved because every term (and total) scales by
	// the same N; the pool counts are 0 for a non-PD run, so this is a no-op there.
	out.PrefillInstances = dep.PrefillInstances * plan.Replicas
	out.DecodeInstances = dep.DecodeInstances * plan.Replicas
	out.SharedInstances = dep.SharedInstances * plan.Replicas
	out.EncodeInstances = dep.EncodeInstances * plan.Replicas
	// A replica with no KV blocks would panic NewSimulator, so it is reported as a clean
	// error instead (R1). The kernel refuses a non-positive per-rank budget, so this is
	// defense in depth.
	if out.TotalKVBlocks <= 0 {
		return dep, fmt.Errorf("dp %d leaves %d KV blocks on each of %d engine replicas; "+
			"the scenario's per-rank KV budget must be positive",
			dp, out.TotalKVBlocks, plan.Replicas)
	}
	return out, nil
}

// resolveDPPlacement plans DP-as-real-placement (#1531) and applies it to the cmd/
// deployment vars, emitting the operator diagnostics. It is the ONE code path both
// `blis run` and `blis replay` traverse (R23), so INV-13 parity for MoE dp>1 is
// structural (#1556 lifted the former run-only guard in cmd/replay.go). KV blocks and
// max_model_len are per rank from the kernel, so only the instance and pool counts change.
//
// Like resolveLatencyConfig and resolvePolicies, it READS AND WRITES the package-level
// deployment vars itself rather than taking them as arguments — deliberately, so there is no
// per-command wiring for a future edit to get right in only one of the two command
// bodies. The pure decision (planDPPlacement) and the pure arithmetic
// (applyDPPlacement) stay separately unit-testable.
//
// Reads: dataParallelism, enableExpertParallel, tensorParallelism,
// numInstances, totalKVBlocks, maxModelLen, and the prefill/decode/prefillDecode/encode
// instance counts (as pre-expansion pool inputs).
//
// Side effects (package-level vars mutated, only when the plan is active and every
// guard passes): numInstances and the four PD pool counts (prefillInstances,
// decodeInstances, prefillDecodeInstances, encodeInstances) — each scaled by Replicas so
// a PD topology spawns its N per-rank replicas per pool (#1553). totalKVBlocks and
// maxModelLen are written back unchanged.
//
// The plan is decided by the caller (planDPPlacement) and passed in, so the one decision
// is made in one place while its application stays here at the single write site.
//
// On error NOTHING is mutated and the caller MUST `logrus.Fatalf` — the CLI boundary
// owns termination; this function only reports. A non-Active plan (dense model, or
// dp 1) mutates nothing, which is what makes the feature a byte-identical no-op
// (INV-6).
func resolveDPPlacement(lr latencyResolution, plan dpPlacementPlan) (dpPlacementPlan, error) {
	if !plan.Active {
		return plan, nil
	}
	logicalInstances := numInstances
	dep, err := applyDPPlacement(plan, dataParallelism, dpPlacementDeployment{
		NumInstances:     numInstances,
		TotalKVBlocks:    totalKVBlocks,
		MaxModelLen:      maxModelLen,
		PrefillInstances: prefillInstances,
		DecodeInstances:  decodeInstances,
		SharedInstances:  prefillDecodeInstances,
		EncodeInstances:  encodeInstances,
	})
	if err != nil {
		return dpPlacementPlan{}, err
	}
	// The scaled PD topology still satisfies P·N+D·N+S·N+E·N ≤ total·N by construction
	// (the pre-scale topology passed ValidatePoolTopology at the CLI boundary and every
	// term scales by the same N). Re-validate as defense in depth (#1553, BC-2): a future
	// change to the scaling arithmetic that broke the invariant must fail loudly here
	// rather than mis-place instances (R1).
	if dep.PrefillInstances > 0 || dep.DecodeInstances > 0 || dep.SharedInstances > 0 || dep.EncodeInstances > 0 {
		if verr := cluster.ValidatePoolTopology(dep.PrefillInstances, dep.DecodeInstances,
			dep.SharedInstances, dep.EncodeInstances, dep.NumInstances); verr != nil {
			return dpPlacementPlan{}, fmt.Errorf("DP-as-placement expanded the PD pool topology past the "+
				"cluster invariant (this should be impossible — every term scales by the same dp): %w", verr)
		}
	}
	numInstances, totalKVBlocks, maxModelLen = dep.NumInstances, dep.TotalKVBlocks, dep.MaxModelLen
	prefillInstances, decodeInstances = dep.PrefillInstances, dep.DecodeInstances
	prefillDecodeInstances, encodeInstances = dep.SharedInstances, dep.EncodeInstances
	logrus.Infof("[cluster] DP-as-placement: scenario dp %d (MoE) → %d single-node engine replicas per logical instance "+
		"(%d logical × %d = %d instances), each per-rank (DP=1, %d KV blocks/replica)",
		dataParallelism, plan.Replicas, logicalInstances, plan.Replicas, numInstances, totalKVBlocks)
	if numInstances > dpPlacementInstanceWarnThreshold {
		logrus.Warnf("[cluster] DP-as-placement is spawning %d engine replicas (--num-instances %d × scenario dp %d); "+
			"if the scenario's dp was a typo this will consume a large amount of memory and time", numInstances, logicalInstances, dataParallelism)
	}
	if err := checkHybridDPNodeLocal(plan); err != nil {
		return dpPlacementPlan{}, err
	}
	if enableExpertParallel {
		// EP-ON placement (#1548): the expert-parallel group is the whole N×TP GPU set this
		// placement already reserves. The kernel prices each replica's steps for the
		// scenario's own parallel layout, EP group included.
		logrus.Infof("[cluster] EP-as-placement: the scenario's expert parallelism over the TP·DP=%d×%d=%d GPU "+
			"expert-parallel group (no additional GPUs reserved)",
			tensorParallelism, dataParallelism, tensorParallelism*dataParallelism)
	}
	return plan, nil
}

// checkHybridDPNodeLocal refuses --dp-load-balancing hybrid for a colocated replica whose
// ranks span nodes. vLLM's hybrid balancer runs one front-end per node over that node's
// ranks (DPLBAsyncMPClient over engine_ranks_managed, vllm/v1/engine/core_client.py:1500-1520;
// api_server_count defaults to data_parallel_size_local, vllm/entrypoints/cli/serve.py:
// 112-120), so a replica across nodes is several endpoints, which BLIS does not model. On one
// node hybrid is the internal balancer.
func checkHybridDPNodeLocal(plan dpPlacementPlan) error {
	if dpLoadBalancing != cluster.DPLoadBalancingHybrid || !plan.Active || kernelOpened == nil ||
		prefillInstances > 0 || decodeInstances > 0 || prefillDecodeInstances > 0 || encodeInstances > 0 {
		return nil
	}
	dp := plan.Replicas
	for first := 0; first+dp <= numInstances; first += dp {
		if a, b := kernelOpened.PlacementOf(first), kernelOpened.PlacementOf(first+dp-1); a.Node != b.Node {
			return fmt.Errorf("--dp-load-balancing hybrid: the replica of ranks %d-%d spans nodes %d-%d, and a "+
				"hybrid front-end balances only its own node's ranks (one endpoint per node), which BLIS does "+
				"not model; use internal or external", first, first+dp-1, a.Node, b.Node)
		}
	}
	return nil
}

// offloadPerBlockBytes is the per-rank KV bytes of one block of blockSize tokens, which sizes
// the offload tiers and their transfer jobs: the kernel's own answer (SequenceVariableBytes:
// page-quantized, sharded by the layout and the cache dtype). Fatal when it is not positive:
// a zero-byte block would make every offload transfer free.
func offloadPerBlockBytes(lr latencyResolution, blockSize int64) int64 {
	b := kernelOpened.Kernel().SequenceVariableBytes(int(blockSize))
	if b <= 0 {
		logrus.Fatalf("kv_offload: the kernel prices a %d-token block at %d bytes; per_block_bytes must be > 0",
			blockSize, b)
	}
	return b
}

// adoptSchedulingPolicy takes the instance scheduler from the scenario's
// engine.scheduling_policy, vLLM's own setting: fcfs, or priority (served lowest priority value
// first, then by arrival -- BLIS's priority-fcfs). Precedence is an explicit --scheduler, then a
// --policy-config bundle (applied later, when --scheduler is unset), then the scenario, then
// the default. An explicit --scheduler that differs is a deliberate policy experiment and is
// kept, with a warning that the simulated engine no longer matches the scenario.
func adoptSchedulingPolicy(cmd *cobra.Command, m *engineModel) {
	stated := enginePolicyOf(m)
	scenarioScheduler, scenarioPreemption = stated.scheduler, stated.preemption
	if cmd.Flags().Changed("scheduler") {
		schedulerSource = "--scheduler"
		recordPolicyOverride("scheduler", "", scenarioScheduler, scheduler, schedulerSource)
	} else {
		scheduler = stated.scheduler
	}
	if cmd.Flags().Changed("preemption-policy") {
		preemptionSource = "--preemption-policy"
		recordPolicyOverride("preemption_policy", "", scenarioPreemption, preemptionPolicy, preemptionSource)
	} else {
		preemptionPolicy = stated.preemption
	}
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

// recordPolicyOverride notes, when the policy a run simulates for setting differs from the one
// its scenario states (scope names a pool, "" = the run), that source chose it: warned on
// stderr and recorded in the results file (scenarioOverrides), so a run that no longer
// reproduces its scenario says so in the file that reports it, not only in a log line. No-op
// when the two agree; a pool that states no policy runs vLLM's default, fcfs (enginePolicyOf).
func recordPolicyOverride(setting, scope, stated, run, source string) {
	// A disaggregated run records its overrides per pool (poolOverridesOf), where each pool's
	// own stated policy is known; a run-level record would repeat the first pool's.
	if stated == "" || run == stated || (scope == "" && scenarioDisaggregated) {
		return
	}
	name := setting
	if scope != "" {
		name = setting + " (" + scope + " pool)"
	}
	logrus.Warnf("%s sets %s %s, overriding scenario %q's engine.scheduling_policy (BLIS %s); "+
		"the results file records the override", source, name, run, kernelScenario, stated)
	scenarioOverrides = append(scenarioOverrides, sim.ScenarioOverride{
		Setting: name, Scenario: stated, Run: run, Source: source,
	})
}

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
	scenarioScheduler, scenarioPreemption, scenarioOverrides = "", "", nil
	schedulerSource, preemptionSource, scenarioDisaggregated = "", "", false
	scenarioRouter, scenarioStatesRouter = routerInputs{}, false
	in := adoptInputs(cmd)
	a, err := adoptScenario(in)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	shape := a.Shape
	// The scenario's offload block is the run's tier configuration (#1910); --kv-offload-config
	// may only restate it (resolveKVOffloadConfig).
	scenarioOffload = shape.Offload
	adoptScenarioEngineKnobs(cmd, shape)
	scenarioShape = shape
	acceptanceRecord = nil
	prefixCacheRecords = nil
	scenarioDisaggregated = a.Disaggregated
	// The router's inputs are the deployment's too: its router block and pool labels.
	adoptScenarioRouter(shape)
	if err := openFirstPool(in, &a); err != nil {
		logrus.Fatalf("%v", err)
	}
	m, settings := a.Model, a.Settings
	kernelOpened = m
	adoptSchedulingPolicy(cmd, m)
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
	// pages and pool. Per data-parallel RANK: a dp>1 MoE deployment runs one replica per rank
	// (DP-as-placement), each sized as one EngineCore, so the placement never divides it.
	totalKVBlocks = settings.KVBlocks
	sequenceState = settings.SequenceState
	prefixHit = prefixHitOf("first pool", m)
	blockSizeTokens = int64(settings.BlockSize)
	maxNumSeqs = int64(settings.MaxNumSeqs)
	maxNumBatchedTokens = int64(settings.MaxNumBatchedTokens)
	maxModelLen = int64(settings.MaxModelLen)
	requireWindowFits("first pool", settings)
	noEnablePrefixCaching = settings.PrefixCachingDisabled
	numSpeculativeTokens = settings.SpeculativeTokens
	speculativeMethod = settings.SpeculativeMethod
	logrus.Infof("deployment from scenario %s -- model %s, hardware %s, "+
		"tp %d, dp %d, expert-parallel %t; engine from the kernel -- %d KV blocks/rank of %d "+
		"tokens, max_num_seqs %d, max_num_batched_tokens %d, max_model_len %d (%s), prefix caching %t, "+
		"speculative tokens %d",
		kernelScenario, model, gpu,
		tensorParallelism, dataParallelism, enableExpertParallel,
		totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen, settings.MaxModelLenFrom,
		!noEnablePrefixCaching, numSpeculativeTokens)
	logSequenceFixedCharge("first pool", settings)
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
// Side effects: model (lowercased), resolvedCatalogRoot.
func resolveLatencyConfig(cmd *cobra.Command) latencyResolution {
	model = strings.ToLower(model)
	catalogRoot, err := resolveCatalogRoot()
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	resolvedCatalogRoot = catalogRoot
	m := kernelOpened
	if m == nil {
		// adoptKernelDeployment has not run (a caller that resolves the latency model
		// alone): open the scenario's first pool exactly as adoption would.
		m, err = openModel(catalogRoot)
		if err != nil {
			logrus.Fatalf("%v", err)
		}
	}
	// The expert geometry the graph stated. Nothing else of ModelConfig is populated: no HF
	// config is parsed, and a half-filled struct read as complete is how a dense model comes
	// to be treated as MoE.
	var modelConfig sim.ModelConfig
	modelConfig.NumLocalExperts = kernelDeploymentExperts
	modelConfig.NumExpertsPerTok = kernelDeploymentTopK

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
		if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
			scheduler = bundle.Scheduler
			schedulerSource = "--policy-config"
			recordPolicyOverride("scheduler", "", scenarioScheduler, scheduler, schedulerSource)
		}
		if bundle.Preemption.Policy != "" && !cmd.Flags().Changed("preemption-policy") {
			preemptionPolicy = bundle.Preemption.Policy
			preemptionSource = "--policy-config"
			recordPolicyOverride("preemption_policy", "", scenarioPreemption, preemptionPolicy, preemptionSource)
		}
	}

	checkPolicyNames(scheduler, preemptionPolicy)
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
	if !cluster.IsValidDPLoadBalancing(dpLoadBalancing) {
		logrus.Fatalf("--dp-load-balancing must be %s, %s or %s, got %q", cluster.DPLoadBalancingInternal,
			cluster.DPLoadBalancingHybrid, cluster.DPLoadBalancingExternal, dpLoadBalancing)
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
	registerCatalogFlag(cmd)

	// Engine settings are the scenario's (#2021 B6): block size, sequence and token caps,
	// max_model_len, long_prefill_token_threshold, prefix caching, cache dtype, speculation; the
	// KV budget is the kernel's. The flags below override the scenario only where stated.
	cmd.Flags().Var(&swaBoundedReplay, "swa-bounded-replay", "Override DeepSeek-V4.1's SWA bounded replay (true|false), over the scenario's engine.swa_bounded_replay. Unset, the pool states it or it resolves as vLLM v0.31.0 does: on under the V2 model runner without prefill context parallelism. false is also what engine builds before the feature run")
	registerFrontendFlags(cmd)
	cmd.Flags().Float64Var(&serverSeriesIntervalS, "server-series-interval-s", 0, "Bin width in seconds of the per-engine vLLM /metrics time series (KV usage, running/waiting, prefix-cache and prompt-token-by-source counters, generation tokens) written as server_series to the --metrics-path file only; 0 disables it")

	// The deployment: a blis-schemas scenario (model, hardware, fabric, pools, engines)
	// priced by blis-latency-kernel with the coefficient sets it names.
	cmd.Flags().StringVar(&kernelScenario, "scenario", "", "Scenario FILE NAME within --scenarios, e.g. gpt-oss-120b-h200-fp4-vllm-tp4.yaml (REQUIRED): the blis-schemas Scenario and Deployment that state the model, hardware, fabric, each pool's parallelism and engine settings, and the coefficient sets that price it")
	cmd.Flags().StringVar(&kernelScenarioDir, "scenarios", "", "Directory holding scenario YAML files (REQUIRED)")
	cmd.Flags().StringVar(&kernelRegistry, "registry", "", "blis-registry clone root, holding the coefficient sets a scenario names (REQUIRED)")

	// Cluster config
	cmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

	// The llm-d router: its own inputs, and the environment around it (router.go).
	registerRouterFlags(cmd)

	// Scheduler and preemption config
	cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")
	cmd.Flags().StringVar(&preemptionPolicy, "preemption-policy", "fcfs", "Preemption victim selection: fcfs (tail-of-batch), priority (least-urgent SLO tier)")

	// Policy bundle config
	cmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")

	// Fitness evaluation config (PR9)
	cmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs (e.g., throughput:0.5,p99_ttft:0.3)")

	// Decision trace config (PR13)
	cmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
	cmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Record the top-k endpoints the router scored, with their state, per routing decision")
	cmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")

	// Tiered KV cache (PR12)
	cmd.Flags().Int64Var(&kvCPUBlocks, "kv-cpu-blocks", 0, "CPU tier KV cache blocks (0 = disabled, single-tier mode). Typical: 1/3 of the kernel's per-rank GPU KV blocks")
	cmd.Flags().Float64Var(&kvOffloadThreshold, "kv-offload-threshold", 0.9, "GPU utilization (0-1) above which blocks are offloaded to CPU. Default: offload when GPU >90% full")
	cmd.Flags().StringVar(&dpLoadBalancing, "dp-load-balancing", cluster.DPLoadBalancingInternal,
		"How a dp>1 MoE replica's requests reach its ranks, as vLLM's serve flags choose: "+
			"internal (default; one endpoint per replica, vLLM's DP balancer picks the rank), "+
			"hybrid (--data-parallel-hybrid-lb; the same for a replica on one node), or "+
			"external (--data-parallel-external-lb / --data-parallel-multi-port-external-lb; "+
			"every rank is its own router endpoint). Ranks step in lockstep in every mode")
	cmd.Flags().Float64Var(&modelAutoscalerIntervalUs, "model-autoscaler-interval-us", 0, "Autoscaler tick interval in microseconds (0 = disabled). Overrides policy-config autoscaler.interval_us when non-zero.")

	// PD disaggregation config
	cmd.Flags().IntVar(&prefillInstances, "prefill-instances", 0, "Number of instances dedicated to prefill (0 = disabled)")
	cmd.Flags().IntVar(&decodeInstances, "decode-instances", 0, "Number of instances dedicated to decode (0 = disabled)")
	cmd.Flags().IntVar(&prefillDecodeInstances, "prefill-decode-instances", 0, "Number of shared-role instances serving both prefill and decode (llm-d 'prefill-decode'/'both' parity; 0 = disabled). Must satisfy --prefill-instances + --decode-instances + --prefill-decode-instances <= --num-instances.")
	cmd.Flags().BoolVar(&pdTransferContention, "pd-transfer-contention", false, "Concurrent P/D KV handoffs share the fabric links they cross (#1922): each link's bandwidth is divided max-min fairly among the handoffs on it, from the kernel's PDTransferDemand, and re-divided as handoffs start and finish. A handoff alone still takes exactly the kernel's PDTransferTime. Default off: every handoff is priced as if alone.")

	// E/P/D disaggregation (GAP-4, issue #1264). Registered on both run and replay.
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

	// KV-cache offload config surface (H5, #1587). One flag: a strict-YAML file with a
	// single top-level kv_offload: block (CPU tier + ordered secondary tiers, per-tier
	// device physics). Registered on run and replay (INV-13). Absent => the offload
	// subsystem is inert and output is byte-identical to a build without the feature
	// (BC-G5). On replay the trace header is authoritative; a passed flag must match the
	// header (see resolveKVOffloadConfig / the replay wiring). device_class names resolve
	// against the CATALOG's storage-device table, <catalog>/devices/storage.yaml (#1770),
	// read only when some tier actually names a class.
	cmd.Flags().StringVar(&kvOffloadConfigPath, "kv-offload-config", "", "Path to a YAML file with a top-level kv_offload: block (multi-tier KV-cache offload config: cpu_bytes_to_use, block_size/blocks_per_chunk, eviction_policy, offload_prompt_only, secondary_tiers[] with per-tier device_class/direct_io). Every secondary tier names a device_class from the catalog's storage-device table at <catalog>/"+catalogStorageDevicesRelPath+" (located by --catalog / "+catalogEnvVar+"); the kernel prices its transfers from that device, so a tier stating its own read_bandwidth/write_bandwidth/base_latency is refused. When the scenario states an offload block, that block is the configuration and this file may only restate it (a file resolving differently is refused). Absent, with no scenario block => offload subsystem inert. On replay the trace header is authoritative.")
}

// resolveSpeculativeConfig builds the speculative-decoding / MTP config (#1528): K and the
// method from the scenario, and the acceptance resolved by blis-schemas' precedence (#1939,
// resolveAcceptance), which it records for the results file (acceptanceRecord). Shared by run
// and replay, so a trace round-trips under identical flags (INV-13). Fatal on an invalid or
// unresolvable config (CLI boundary, R6).
func resolveSpeculativeConfig(cmd *cobra.Command) sim.SpeculativeConfig {
	acceptanceRecord = nil
	ar, speculating := resolveAcceptance(cmd, numSpeculativeTokens, speculativeMethod)
	alpha := 0.0
	if speculating {
		alpha = ar.Res.AcceptanceRate()
		if ar.CLIRate != nil {
			alpha = *ar.CLIRate
		}
	} else if cmd.Flags().Changed(acceptanceRateFlag) {
		alpha = speculativeAcceptance // refused below: a rate with nothing drafted
	}
	c := speculativeConfigOf(numSpeculativeTokens, alpha, speculativeMethod)
	if c.IsEnabled() {
		acceptanceRecord = acceptanceRecordOf(ar.Res, c)
		logAcceptance(acceptanceRecord, c.K)
		logrus.Infof("speculative decoding enabled: method=%q k=%d acceptance=%.3f (mean %.3f tokens/step, verify width %d, %d lookahead KV slots)",
			c.Method, c.K, c.Acceptance, c.EffectiveTokensPerStep(), c.VerifyWidth(), c.LookaheadTokens())
		if c.Acceptance == 0 {
			logrus.Warnf("speculative decoding: acceptance length 1 => no throughput gain but verify-width cost applies (net slowdown); this is usually not intended")
		}
	}
	return c
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

// applyTimeoutToRequests re-applies timeout to already-generated requests and session
// blueprints. This corrects deadlines for inference_perf specs: spec.Clients is empty
// when applyTimeoutToSpec runs and is populated inside GenerateWorkload, so the initial
// deadlines are computed from nil Timeout. Safe to call for all spec types.
// timeoutSecs>0 sets deadline=ArrivalTime+timeout; timeoutSecs<=0 sets deadline=0 (disabled).
func applyTimeoutToRequests(wl *workload.GeneratedWorkload, timeoutSecs int) {
	var timeoutUs int64
	if timeoutSecs > 0 {
		timeoutUs = int64(timeoutSecs) * 1_000_000
	}
	for _, req := range wl.Requests {
		if timeoutUs == 0 {
			req.Deadline = 0
		} else {
			req.Deadline = req.ArrivalTime + timeoutUs
		}
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

		// KV-cache offload config surface (#1587): resolve ONCE (R4), validated at the
		// CLI boundary. Inert (zero value) when --kv-offload-config is absent (BC-G5).
		// Recorded in the exported trace header below for run/replay parity (BC-G6).
		kvOffloadCfg := resolveKVOffloadConfig(cmd)
		// #1590 (H1): --kv-offload-config (multi-tier chain) and --kv-cpu-blocks (legacy
		// single CPU tier) are distinct offload models; setting both is ambiguous.
		if kvOffloadCfg.IsEnabled() && kvCPUBlocks > 0 {
			logrus.Fatalf("--kv-offload-config and --kv-cpu-blocks are mutually exclusive (distinct KV-offload models); set only one")
		}

		// Resolve the latency model (single code path shared with replayCmd).
		lr := resolveLatencyConfig(cmd)

		// Per-pool engine overrides, filled from each P/D pool's own kernel
		// (openKernelPools, below) after PD validation. Empty when disaggregation is
		// disabled (prefillInstances == 0).
		var prefillOverrides, decodeOverrides cluster.PoolOverrides

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

		// Workload configuration — all paths synthesize a v2 WorkloadSpec
		// and generate requests via workload.GenerateRequests (BC-10).
		var spec *workload.WorkloadSpec
		var preGeneratedRequests []*sim.Request
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
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
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
			if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
				logrus.Fatalf("--rate must be a finite value > 0, got %v", rate)
			}
			// #1769: the preset is read from <catalog>/workloads/<name>.yaml, through the
			// same reader `blis convert preset` and `blis observe --workload` use.
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
		// Pre-expand inference-perf / ServeGen specs (LAZY MODE ONLY) so
		// the timeout-application step below sees every client — including
		// those populated by expansion. Without this, --lazy-generation +
		// --workload-spec=inference_perf.yaml + an explicit --timeout
		// would build streaming states from clients whose Timeout is nil
		// (because expansion inside GenerateWorkloadLazy happens AFTER
		// applyTimeoutToSpec runs), producing the default 300 s deadline
		// instead of the user-requested value (PR #1453 self-review).
		//
		// Scoped to `lazyGeneration` so it does not run for eager-only
		// invocations. Running it unconditionally would clear the
		// spec.InferencePerf marker before GenerateWorkload's Validate,
		// which suppresses the mixed-slo_class check via
		// `s.InferencePerf == nil && s.ServeGenData == nil` in spec.go.
		// Today all inference-perf-expanded clients carry SLOClass="standard"
		// (uniform → check can't fire), so eager was accidentally safe.
		// But scoping here defends against a future ExpandInferencePerfSpec
		// change that emits mixed/empty slo_class from silently failing
		// eager runs that previously validated (PR #1453 review round 3).
		//
		// REMOVING THIS CALL re-introduces the lazy timeout bug silently.
		// The regression is covered by
		// TestGenerateWorkloadLazy_InferencePerf_TimeoutAppliedAfterPreExpand
		// at the library layer; the cmd-level smoke is covered by the
		// inference-perf byte-identity check verified during PR review.
		//
		// ExpandClientsAndCohorts is idempotent — the generators'
		// validateAndExpandSpec runs it again with no effect since both
		// branches guard on len(spec.Clients) == 0. (#1441)
		if lazyGeneration {
			if err := workload.ExpandClientsAndCohorts(spec); err != nil {
				logrus.Fatalf("Failed to expand workload spec: %v", err)
			}
		}
		if workloadSpecPath == "" || cmd.Flags().Changed("timeout") {
			applyTimeoutToSpec(spec, requestTimeoutSecs)
		}

		// Resolve maxRequests: spec.NumRequests as default, CLI --num-requests overrides
		maxRequests := spec.NumRequests
		if cmd.Flags().Changed("num-requests") {
			maxRequests = int64(numRequests)
		}

		// Guard against unbounded generation
		if maxRequests <= 0 && simulationHorizon == math.MaxInt64 {
			logrus.Fatalf("Workload requires either num_requests or --horizon to bound generation")
		}

		// Lazy generation path (#1441, alpha). Default off. When set, build
		// a streaming workload source instead of materializing the full
		// request slice. As of #1460 there is NO eager-fallback class — every
		// spec the eager generator accepts is streamed: multi-session reasoning
		// (#1458), concurrency clients (#1459), and time-varying / per-window
		// workloads (#1460).
		var wl *workload.GeneratedWorkload
		// lazyRequestSource is typed as the interface satisfied by
		// *workload.lazyRequestSource: Next() delivers requests to the
		// cluster; Err() surfaces any terminal sampler/generator error
		// recorded on a per-client state after the run completes, so
		// cmd can Fatalf and match the eager path's abort-on-invalid-spec
		// behavior (PR #1453 review round 3).
		var lazyRequestSource interface {
			Next() (*sim.Request, bool)
			Err() error
		}
		if lazyGeneration {
			src, sessions, followUpBudget, lazyErr := workload.GenerateWorkloadLazy(spec, simulationHorizon, maxRequests)
			// As of #1460 there is no ErrLazyUnsupported* fallback class — every
			// spec the eager generator accepts is streamed. Any error is a real
			// spec/validation failure → abort (matches eager's error handling).
			if lazyErr != nil {
				logrus.Fatalf("Failed to build lazy workload: %v", lazyErr)
			}
			lazyRequestSource = src
			wl = &workload.GeneratedWorkload{Sessions: sessions, FollowUpBudget: followUpBudget}
		}
		if wl == nil {
			var err error
			wl, err = workload.GenerateWorkload(spec, simulationHorizon, maxRequests)
			if err != nil {
				logrus.Fatalf("Failed to generate workload: %v", err)
			}
		}
		// Re-apply timeout to generated requests and session blueprints.
		// For inference_perf specs, spec.Clients was empty at applyTimeoutToSpec time
		// and populated inside GenerateWorkload — deadlines need correction here.
		// In lazy mode wl.Requests is nil (no-op for the request loop); session
		// blueprint Timeout pointers still need refresh.
		if workloadSpecPath == "" || cmd.Flags().Changed("timeout") {
			applyTimeoutToRequests(wl, requestTimeoutSecs)
		}
		preGeneratedRequests = wl.Requests
		if len(wl.Sessions) > 0 {
			sessionMgr = workload.NewSessionManager(wl.Sessions)
			if wl.FollowUpBudget >= 0 {
				sessionMgr.SetFollowUpBudget(wl.FollowUpBudget)
			}
			if lazyRequestSource != nil {
				logrus.Infof("Generated streaming source + %d session blueprints (closed-loop, lazy)", len(wl.Sessions))
			} else {
				logrus.Infof("Generated %d requests + %d session blueprints (closed-loop)", len(wl.Requests), len(wl.Sessions))
			}
		} else if lazyRequestSource != nil {
			logrus.Infof("Generated streaming workload source (lazy, #1441)")
		} else {
			logrus.Infof("Generated %d requests via unified workload pipeline", len(wl.Requests))
		}

		if numInstances < 1 {
			logrus.Fatalf("num-instances must be >= 1")
		}
		if totalKVBlocks <= 0 {
			logrus.Fatalf("scenario %q: the kernel sized %d KV blocks per rank; it must be > 0", kernelScenario, totalKVBlocks)
		}
		if maxNumSeqs <= 0 {
			logrus.Fatalf("scenario %q: engine max_num_seqs must be > 0, got %d", kernelScenario, maxNumSeqs)
		}
		if maxNumBatchedTokens <= 0 {
			logrus.Fatalf("scenario %q: engine max_num_batched_tokens must be > 0, got %d", kernelScenario, maxNumBatchedTokens)
		}
		// Changed() guard: unlike peer flags (default always positive), --horizon defaults
		// to math.MaxInt64 which would fail <= 0. Only validate when user explicitly sets it.
		if cmd.Flags().Changed("horizon") && simulationHorizon <= 0 {
			logrus.Fatalf("--horizon must be > 0, got %d", simulationHorizon)
		}

		// Resolve policy configuration (single code path shared with replayCmd).
		// Per-pool scorer configs (PD disaggregation) remain inline below.
		bundle := resolvePolicies(cmd)

		// Resolve autoscaler and node pool config from policy bundle, then apply CLI overrides.
		var (
			bundleAutoscalerIntervalUs           float64
			bundleScaleUpStabilizationWindowUs   float64
			bundleScaleDownStabilizationWindowUs float64
			bundleHPAScrapeDelayMean             float64
			bundleHPAScrapeDelayStddev           float64
			bundleAnalyzerCfg                    cluster.V2SaturationAnalyzerConfig
			bundleNodePools                      []cluster.NodePoolConfig
			bundleInstanceLifecycle              cluster.InstanceLifecycleConfig
		)
		if bundle != nil {
			if bundle.Autoscaler.IntervalUs > 0 {
				bundleAutoscalerIntervalUs = bundle.Autoscaler.IntervalUs
				bundleScaleUpStabilizationWindowUs = bundle.Autoscaler.ScaleUpStabilizationWindowUs
				bundleScaleDownStabilizationWindowUs = bundle.Autoscaler.ScaleDownStabilizationWindowUs
				bundleHPAScrapeDelayMean = bundle.Autoscaler.HPAScrapeDelay.Mean
				bundleHPAScrapeDelayStddev = bundle.Autoscaler.HPAScrapeDelay.Stddev
				bundleAnalyzerCfg = cluster.V2SaturationAnalyzerConfig{
					KvCacheThreshold:  bundle.Autoscaler.Analyzer.KVCacheThreshold,
					ScaleUpThreshold:  bundle.Autoscaler.Analyzer.ScaleUpThreshold,
					ScaleDownBoundary: bundle.Autoscaler.Analyzer.ScaleDownBoundary,
					AvgInputTokens:    bundle.Autoscaler.Analyzer.AvgInputTokens,
				}
			}
			for _, np := range bundle.NodePools {
				// Every instance is priced by the kernel for the scenario's chip, so a node
				// pool of another GPU would simulate hardware the kernel never priced.
				if !strings.EqualFold(np.GPUType, gpu) {
					logrus.Fatalf("policy bundle node pool %q is %q GPUs, but scenario %q runs on %q; every "+
						"instance is priced for the scenario's hardware, so node pools must match it (heterogeneous "+
						"fleets: #1925)",
						np.Name, np.GPUType, kernelScenario, gpu)
				}
				// The kernel sizes every instance's KV from the catalog chip's memory; a pool
				// stating other memory would describe a device the run does not simulate.
				if chip := kernelOpened.Deployment().DeviceMemoryGiB; np.GPUMemoryGiB > 0 &&
					math.Abs(np.GPUMemoryGiB-chip) > 1e-9 {
					logrus.Fatalf("policy bundle node pool %q states gpu_memory_gib %g, but the catalog "+
						"chip %q has %g GiB, from which the kernel sizes every instance's KV; state %g",
						np.Name, np.GPUMemoryGiB, gpu, chip, chip)
				}
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
			}
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
		kernelPD := openKernelPools(resolvedCatalogRoot)
		if kernelPD != nil {
			prefillOverrides, decodeOverrides = kernelPD.overrides()
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

		// DP-as-real-placement (#1531, #1553): on an MoE model, the scenario's dp N means N
		// independent single-node engine replicas (vLLM's internal DP EngineCores), not one
		// lumped instance. Expand to numInstances × N real replicas — reusing the existing
		// per-instance placement path — each configured per-rank (DP=1). The kernel already
		// sized KV blocks and max_model_len per rank, so only the instance and PD pool counts
		// change. PD disaggregation and node pools are SUPPORTED (#1553, each pool spawns its
		// own N per-rank replicas); the autoscaler fails fast (planDPPlacement).
		// resolveDPPlacement is the ONE code path run and replay share (R23), so INV-13
		// parity is structural (#1556). A no-op for dp 1 and dense models.
		//
		// The plan is decided HERE — after the policy bundle is parsed, so the autoscaler /
		// node-pool predicates are real; this is where the autoscaler rejection (#1553
		// decision) surfaces. resolveDPPlacement APPLIES the plan (the single write site for
		// numInstances and the four PD pool counts).
		dpPlan, dpErr := planDPPlacement(lr.ModelConfig.IsMoE(), dataParallelism, enableExpertParallel,
			prefillInstances > 0 || decodeInstances > 0 || prefillDecodeInstances > 0 || encodeInstances > 0,
			bundleAutoscalerIntervalUs > 0, len(bundleNodePools) > 0)
		if dpErr != nil {
			logrus.Fatalf("%v", dpErr)
		}
		dpPlan, dpErr = resolveDPPlacement(lr, dpPlan)
		if dpErr != nil {
			logrus.Fatalf("%v", dpErr)
		}

		// Log configuration after all config sources (CLI, workload spec, policy bundle)
		// AND the DP-as-placement adjustment are resolved, so the reported block count is
		// the per-replica one actually configured — matching the equivalent line in
		// cmd/replay.go, which also logs after the DP block (diagnostic parity for a
		// feature whose whole point is that the two commands agree).
		logrus.Infof("Starting simulation with %d KV blocks, horizon=%dticks", totalKVBlocks, simulationHorizon)

		// #1590 (H1): derive per_block_bytes for the offload tier chain -- the kernel's
		// per-rank KV byte size of one GPU block -- and record it on the resolved offload
		// config. It feeds the CPU-tier block capacity and transfer-job sizing, and
		// round-trips through the trace header (INV-13). Only when offload is enabled.
		if kvOffloadCfg.IsEnabled() {
			kvOffloadCfg.PerBlockBytes = offloadPerBlockBytes(lr, blockSizeTokens)
			if err := kvOffloadCfg.CheckFitsOneChunk(); err != nil {
				logrus.Fatalf("%v", err)
			}
		}

		// The kernel prices every offload transfer: the secondary tiers through TierTime and
		// the legacy CPU tier as one whole per-block reload charge. Shared with replayCmd
		// through one helper (R23, INV-13).
		kernelCPUTierTicks := applyKernelOffloadPricing(&kvOffloadCfg)
		// A disaggregated run's pools each size and price the offload by their own kernel.
		if kernelPD != nil {
			kernelPD.applyOffload(kvOffloadCfg, &prefillOverrides, &decodeOverrides)
		}

		// Unified cluster path (used for all values of numInstances).
		// INV-13 SYNC POINT: PD fields below must stay in sync with cmd/replay.go (replayCmd
		// DeploymentConfig literal). See docs/contributing/standards/invariants.md INV-13.
		// The instance counts are final (DP-as-placement applied): every instance must fit
		// its pool, and the P/D handoff pricer places instances by these counts.
		requireKernelCapacity(kernelPD)
		pdTransferTime, pdTransferDemand := kernelPD.handoffPricing(pdTransferContention)
		scenarioTopology, topologyErr := scenarioEndpointTopology(kernelOpened, kernelPD, numInstances, prefillInstances, decodeInstances)
		if topologyErr != nil {
			logrus.Fatalf("scenario endpoint placement: %v", topologyErr)
		}
		ec := engineConfigFor(cmd, lr.ModelConfig, dpPlan.PerRankDP, kvOffloadCfg, kernelCPUTierTicks, loraCfg)
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: simulationHorizon,
				Seed:    seed,
				// The engine's configuration comes from the build site (engineConfigFor), the same
				// for run and replay (INV-13). DP-as-placement (#1531): dpPlan.PerRankDP is the
				// per-replica DP -- 1 when the plan is active (each replica is one rank), else the
				// CLI dataParallelism unchanged; since #1556 replay wires the SAME dpPlan.PerRankDP
				// from the SAME resolveDPPlacement.
				KVCacheConfig:        ec.KVCache,
				BatchConfig:          ec.Batch,
				ModelHardwareConfig:  ec.ModelHardware,
				PolicyConfig:         ec.Policy,
				LoRAConfig:           ec.LoRA,
				SpeculativeConfig:    ec.Speculative,
				Frontend:             frontendConfigFromCLI(),
				SLOPriorityOverrides: sloPriorityOverrides,
				// The kernel prices every step.
				LatencyModel:           lr.KernelModel,
				ServerSeriesIntervalUs: serverSeriesIntervalUs(),
			},
			NumInstances:                   numInstances,
			DPLockstepRanks:                dpPlan.LockstepRanks,
			DPLoadBalancing:                dpLoadBalancing,
			TraceLevel:                     traceLevel,
			CounterfactualK:                counterfactualK,
			PrefillInstances:               prefillInstances,
			DecodeInstances:                decodeInstances,
			SharedInstances:                prefillDecodeInstances,
			EncodeInstances:                encodeInstances,
			PDTransferTime:                 pdTransferTime,
			PDTransferContention:           pdTransferContention,
			PDTransferDemand:               pdTransferDemand,
			PDPrefillKVHold:                kernelPD.holdsPrefillKV(),
			PrefillOverrides:               prefillOverrides,
			DecodeOverrides:                decodeOverrides,
			ModelAutoscalerIntervalUs:      bundleAutoscalerIntervalUs,
			ScaleUpStabilizationWindowUs:   bundleScaleUpStabilizationWindowUs,
			ScaleDownStabilizationWindowUs: bundleScaleDownStabilizationWindowUs,
			HPAScrapeDelay:                 cluster.DelaySpec{Mean: bundleHPAScrapeDelayMean, Stddev: bundleHPAScrapeDelayStddev},
			AutoscalerAnalyzerConfig:       bundleAnalyzerCfg,
			NodePools:                      bundleNodePools,
			ScenarioEndpointTopology:       cluster.StaticEndpointTopology(scenarioTopology),
			InstanceLifecycle:              bundleInstanceLifecycle,
		}
		// Session callback installation (Constraint 3 fix):
		// Follow-up collection must be UNCONDITIONAL for saturation analysis correctness.
		// The TraceV2 export (lines 1582-1601) remains gated on --trace-output, but the
		// follow-up accumulation happens regardless so saturation analysis sees complete workloads.
		var followUpRequests []*sim.Request
		var onRequestDone func(*sim.Request, int64) []*sim.Request
		if sessionMgr != nil {
			// Always install callback to accumulate follow-ups (for saturation analysis + optional trace export)
			baseCb := sessionMgr.OnComplete
			onRequestDone = func(req *sim.Request, clock int64) []*sim.Request {
				followUps := baseCb(req, clock)
				followUpRequests = append(followUpRequests, followUps...)
				return followUps
			}
		}
		// RequestSource: streaming in lazy mode, eager-slice otherwise.
		// The workload package's lazy source satisfies cluster.RequestSource
		// via structural typing — both define the same Next() method.
		var clusterRequestSource cluster.RequestSource
		if lazyRequestSource != nil {
			clusterRequestSource = lazyRequestSource
		} else {
			clusterRequestSource = cluster.NewSliceRequestSource(preGeneratedRequests)
		}
		cs := cluster.NewClusterSimulator(config, clusterRequestSource, onRequestDone)
		router := resolveRouterInputs()
		gw := newGateway(router, model)
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
		// trace is disabled). Saturation analysis continues to use the
		// preGeneratedRequests + followUpRequests path below — those slices
		// remain populated for that purpose only.
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

		// Resolve the saturation tracer from --detectors / --saturation-config /
		// --saturation-report BEFORE the run so an unknown name, bad config, or
		// unwritable report path fails fast (#1516 single detector, #1519 bank).
		satTracer, satErr := resolveSaturation()
		if satErr != nil {
			logrus.Fatalf("%v", satErr)
		}

		if err := cs.Run(); err != nil {
			logrus.Fatalf("Simulation failed: %v", err)
		}

		// Surface any terminal sampler / generator error the lazy source
		// recorded on a per-client state during the run. Eager mode would
		// have hit logrus.Fatalf inside cmd on the same invalid spec;
		// without this check, lazy mode would exit 0 with reduced traffic
		// and misleading capacity numbers (PR #1453 review round 3).
		if lazyRequestSource != nil {
			if err := lazyRequestSource.Err(); err != nil {
				logrus.Fatalf("Lazy workload sampler failure: %v", err)
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
				logrus.Fatalf("Trace export: arrival hook was not installed but --trace-output=%q is set — install/export branches diverged (issue #1440)", traceOutput)
			}
			records := workload.RequestsToTraceRecords(traceArrivals)
			header := &workload.TraceHeader{
				Version:           3,
				TimeUnit:          "microseconds",
				Mode:              "generated",
				WorkloadSeed:      &spec.Seed,
				GoodputSLOTargets: goodputTargets,                   // #1413, BC-7: persist resolved targets for downstream replay/calibrate
				KVOffload:         simToHeaderOffload(kvOffloadCfg), // #1587, BC-G6: nil when inert (omitted); round-trips resolved config
				// #1530: record multi-node placement so replay can refuse a trace whose
				// step times it cannot reproduce. 0/1 (no span) is omitted, so a run
				// without multi-node placement writes a byte-identical header (INV-6).
				MaxNodesSpanned: crossNodeSpanForTrace(cs.MaxNodesSpanned()),
				// INV-13: the router the run routed with, so a replay rebuilds it.
				Router: router.header(gw.Seed()),
			}
			if err := workload.ExportTraceV2(header, records, traceOutput+".yaml", traceOutput+".csv"); err != nil {
				logrus.Fatalf("Trace export failed: %v", err)
			}
			logrus.Infof("Trace exported: %s.yaml, %s.csv (%d records)", traceOutput, traceOutput, len(records))
		}

		if numInstances > 1 {
			// Print per-instance metrics to stdout (multi-instance only). Per-instance
			// output carries no saturation field — the final label is a cluster-level
			// verdict, emitted on the aggregate below (#1517).
			for _, inst := range cs.Instances() {
				if err := inst.Metrics().SaveResults(string(inst.ID()), config.Horizon, totalKVBlocks, ""); err != nil {
					logrus.Fatalf("SaveResults for instance %s: %v", inst.ID(), err)
				}
			}
		}
		// Build aggregate output, inject goodput, run the saturation reducer, then
		// emit (#1413 goodput / #1517 saturation share the build-then-mutate-then-emit
		// pattern). The saturation label must reach stdout, so the tracer runs BEFORE
		// EmitOutput and mutates clusterOutput.Saturation — sim/ builds the struct and
		// knows nothing about saturation; cmd/ sets the field.
		aggregated := cs.AggregatedMetrics()
		clusterOutput := aggregated.BuildOutput("cluster")
		emitGoodput(&clusterOutput, aggregated, cs.InjectedByClass(),
			float64(aggregated.SimEndedTime)/1e6, goodputTargets)

		// Saturation (#1516 single detector / #1519 bank / #1517 final label): stream
		// the selected detector(s) over the aggregate's completed-request metrics,
		// write the per-event verdict trace to --saturation-report (if given), and
		// splice the per-detector final label onto stdout. Same pipeline as
		// replay/observe; only the input slice differs (here it is sim-derived,
		// INV-13). Guard on the tracer so the common no-detector path skips the
		// O(n log n) sort + O(n) copy in CompletedRequestMetrics().
		if satTracer != nil {
			final, err := satTracer.run(aggregated.CompletedRequestMetrics())
			if err != nil {
				logrus.Fatalf("Saturation: %v", err)
			}
			if len(final) > 0 {
				clusterOutput.Saturation = final
			}
		}

		// Catalog provenance (#1732): file-only, so it is passed as an EmitOutput option
		// rather than mutated onto clusterOutput above — stdout must stay byte-identical
		// (INV-6). Same shared helper on the replay path (INV-13).
		// P/D metrics (#1922/#1934/#1940): stdout's PD section and the file's pd block.
		pdMetrics := cs.PDMetrics()
		if err := aggregated.EmitOutput(clusterOutput, metricsPath,
			append(resultsFileEmitOptions(metricsPath, resolvedCatalogRoot), sim.WithPD(pdMetrics.Output()), sim.WithServerSeries(cs.ServerSeries()))...); err != nil {
			logrus.Fatalf("SaveResults: %v", err)
		}

		// Collect RawMetrics and compute fitness (PR9)
		rawMetrics := cluster.CollectRawMetrics(
			cs.AggregatedMetrics(),
			cs.PerInstanceMetrics(),
			cs.RejectedRequests(),
			scheduler,
			cs.RoutingRejections(),
			cs.EncodeRoutingRejections(),
			cs.InjectedByClass(),
		)

		rawMetrics.PD = pdMetrics
		rawMetrics.ShedByTier = cs.ShedByTier()                     // Phase 1B-1a: tier-shed per-tier breakdown (SC-004)
		rawMetrics.GatewayQueueDepth = cs.GatewayQueueDepth()       // Issue #882: gateway queue depth at horizon
		rawMetrics.GatewayQueueRejected = cs.GatewayQueueRejected() // Issue #1190: gateway queue rejected count
		rawMetrics.GatewayEvicted = cs.GatewayEvicted()             // Phase 4: in-flight eviction count (#1228)
		rawMetrics.GatewayExpired = cs.GatewayExpired()             // Phase 6: TTL expiration count (#1193)

		if fitnessWeights != "" {
			weights, err := cluster.ParseFitnessWeights(fitnessWeights)
			if err != nil {
				logrus.Fatalf("Invalid fitness weights: %v", err)
			}
			fitness, fitErr := cluster.ComputeFitness(rawMetrics, weights)
			if fitErr != nil {
				logrus.Fatalf("Fitness evaluation failed: %v", fitErr)
			}
			fmt.Println("=== Fitness Evaluation ===")
			fmt.Printf("Score: %.6f\n", fitness.Score)
			// Sort keys for deterministic output order
			componentKeys := make([]string, 0, len(fitness.Components))
			for k := range fitness.Components {
				componentKeys = append(componentKeys, k)
			}
			sort.Strings(componentKeys)
			for _, k := range componentKeys {
				fmt.Printf("  %s: %.6f\n", k, fitness.Components[k])
			}
		}

		// Print anomaly counters if any detected
		if rawMetrics.PriorityInversions > 0 || rawMetrics.HOLBlockingEvents > 0 || rawMetrics.RejectedRequests > 0 || rawMetrics.RoutingRejections > 0 || rawMetrics.DroppedUnservable > 0 || rawMetrics.LengthCappedRequests > 0 || rawMetrics.GatewayQueueDepth > 0 || rawMetrics.GatewayQueueRejected > 0 || rawMetrics.GatewayEvicted > 0 || rawMetrics.GatewayExpired > 0 || rawMetrics.EncodeRoutingRejections > 0 || rawMetrics.TimedOutRequests > 0 {
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

		// Print KV cache metrics if any nonzero (BC-1, BC-2)
		printKVCacheMetrics(os.Stdout, rawMetrics.PreemptionRate, rawMetrics.CacheHitRate, rawMetrics.KVThrashingRate)

		// Print per-SLO metrics. With goodput targets configured, the section prints
		// even for a single class (#1413, BC-5). Without goodput, the legacy
		// suppression for ≤1 class is preserved.
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

		// Print PD disaggregation metrics if disaggregation was active (PR4)
		printPDMetrics(os.Stdout, rawMetrics.PD)

		// Build and print trace summary if requested (BC-9)
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

		logrus.Info("Simulation complete.")
	},
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
		os.Exit(1)
	}
}

// init sets up CLI flags and subcommands
func init() {
	registerSimConfigFlags(runCmd)

	// Workload generation flags (run-only)
	runCmd.Flags().StringVar(&workloadType, "workload", "distribution", "Workload type (chatbot, summarization, contentgen, multidoc, distribution)")

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
	runCmd.Flags().BoolVar(&lazyGeneration, "lazy-generation", false, "Alpha (#1441): stream requests from the workload generator instead of pre-generating the full slice. Default off. Supports every workload class — single-shot, single- and multi-session reasoning (#1458), concurrency clients (#1459), and time-varying / per-window workloads (#1460); no eager fallback.")
	runCmd.Flags().IntVar(&requestTimeoutSecs, "timeout", 300, "Per-request deadline in seconds (default 300s matches the session-client default in computeDeadline). Negative = disabled; 0 is rejected. Consistent with blis observe: both commands reject 0.")
	runCmd.Flags().StringVar(&goodputSLOTTFT, "slo-ttft", "", "Per-class TTFT goodput thresholds (e.g. \"critical=100ms,standard=500ms\"). Precedence: CLI > trace header > workload spec.")
	runCmd.Flags().StringVar(&goodputSLOITL, "slo-itl", "", "Per-class mean ITL goodput thresholds (e.g. \"critical=50ms,standard=150ms\").")
	runCmd.Flags().StringVar(&goodputSLOE2E, "slo-e2e", "", "Per-class E2E goodput thresholds (e.g. \"critical=5s,standard=30s\").")

	// Run-specific export
	runCmd.Flags().StringVar(&traceOutput, "trace-output", "", "Export workload as TraceV2 files (<prefix>.yaml + <prefix>.csv)")
	runCmd.Flags().StringVar(&metricsPath, "metrics-path", "", "File to write MetricsOutput JSON (aggregate P50/P95/P99 TTFT, E2E, throughput stats). Use --results-path on blis replay for per-request SimResult JSON.")

	// Saturation trace flags (#1516): --detectors + --saturation-config + --saturation-report.
	registerDetectorFlags(runCmd)

	// Attach `run` as a subcommand to `root`
	rootCmd.AddCommand(runCmd)
}

// prefixHitOf is how the pool's KV cache groups cut a prefix hit. It records the resolution
// for the results file and logs when the model's layout is not one BLIS maps.
func prefixHitOf(what string, m *engineModel) sim.KVPrefixHit {
	r := m.PrefixHit(swaBoundedReplay.v)
	if r.Note != "" {
		logrus.Warnf("prefix cache (%s): the hit is the full-attention prefix alone: %s", what, r.Note)
	}
	if prefixCacheRecords == nil {
		prefixCacheRecords = map[string]sim.PrefixCacheRecord{}
	}
	prefixCacheRecords[string(m.Role())] = sim.PrefixCacheRecord{
		ModelRunner: r.Runner, ModelRunnerForcedBy: r.RunnerForcedBy,
		SWABoundedReplay: r.BoundedReplay, SWABoundedReplayFrom: r.BoundedReplayFrom,
		Layout: r.Hit, Note: r.Note,
	}
	return r.Hit
}

// prefixCacheRecords are each opened pool's prefix-hit resolution, keyed by role, for the
// results file.
var prefixCacheRecords map[string]sim.PrefixCacheRecord

// optionalBool is a boolean flag that tells "unset" from false.
type optionalBool struct{ v *bool }

func (o *optionalBool) String() string {
	if o.v == nil {
		return ""
	}
	return strconv.FormatBool(*o.v)
}

func (o *optionalBool) Set(s string) error {
	b, err := strconv.ParseBool(s)
	if err != nil {
		return err
	}
	o.v = &b
	return nil
}

func (o *optionalBool) Type() string { return "bool" }

// IsBoolFlag lets the flag be passed bare (--swa-bounded-replay) as true.
func (o *optionalBool) IsBoolFlag() bool { return true }

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
