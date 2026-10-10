package cmd

import (
	"bytes"
	"fmt"
	"io"
	"math"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-schemas/spec/deployment"
	sim "github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/cluster"
	"github.com/inference-sim/inference-sim/sim/kernelmodel"
	_ "github.com/inference-sim/inference-sim/sim/lora" // registers sim.NewAdapterRegistryFunc via init()
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
	seed                      int64              // Seed for random token generation
	simulationHorizon         int64              // Total simulation time (in ticks)
	logLevel                  string             // Log verbosity level
	totalKVBlocks             int64              // KV blocks per rank, sized by the kernel
	maxNumSeqs                int64              // Maximum number of requests in the Running batch (vLLM: --max-num-seqs)
	maxNumBatchedTokens       int64              // Maximum total number of tokens across requests in the Running batch (vLLM: --max-num-batched-tokens)
	noEnablePrefixCaching     bool               // --no-enable-prefix-caching: disable cross-request GPU prefix reuse (vLLM parity, #1867)
	blockSizeTokens           int64              // Number of tokens per KV block
	defaultsFilePath          string             // Path to default constants: the LoRA cost coefficients
	catalogPath               string             // --catalog: catalog clone root (models/, hardware/, networks/, ...). No default; BLIS_CATALOG is the fallback (#1731)
	resolvedCatalogRoot       string             // Catalog ROOT this run read (side effect of resolveLatencyConfig); recorded as results-file provenance (#1732)
	workloadType              string             // Workload type (chatbot, summarization, contentgen, multidoc, distribution)
	longPrefillTokenThreshold int64              // Max length of prefill beyond which chunked prefill is triggered
	rate                      float64            // Requests arrival per second
	numRequests               int                // Number of requests
	concurrency               int                // Number of concurrent virtual users (closed-loop)
	thinkTimeMs               int                // Think time between response and next request (ms)
	prefixTokens              int                // Prefix Token Count
	promptTokensMean          int                // Average Prompt Token Count
	promptTokensStdev         int                // Stdev Prompt Token Count
	promptTokensMin           int                // Min Prompt Token Count
	promptTokensMax           int                // Max Prompt Token Count
	outputTokensMean          int                // Average Output Token Count
	outputTokensStdev         int                // Stdev Output Token Count
	outputTokensMin           int                // Min Output Token Count
	outputTokensMax           int                // Max Output Token Count
	kernelScenario            string             // CLI --scenario: scenario file name (required)
	kernelScenarioDir         string             // CLI --scenarios: directory of scenario files (required)
	kernelRegistry            string             // CLI --registry: blis-registry clone root (required)
	kernelDeploymentExperts   int                // routed expert count from the model graph; 0 for a dense model
	kernelDeploymentTopK      int                // routed experts per token from the model graph; 0 for a dense model
	kernelOpened              *kernelmodel.Model // the kernel adoptKernelDeployment opened; nil until it runs
	maxModelLen               int64              // the scenario's engine.max_model_len: max total sequence length (input + output)
	// The deployment: model, GPU, TP, DP and EP, all from the scenario
	model                string // LLM name
	gpu                  string // GPU type
	tensorParallelism    int    // TP value
	dataParallelism      int    // DP value (MoE only), from the kernel scenario
	enableExpertParallel bool   // EP mode (MoE only), from the kernel scenario

	// cluster config
	numInstances int // Number of instances in the cluster

	// online routing pipeline config
	admissionPolicy       string             // Admission policy name
	admissionLatency      int64              // Admission latency in microseconds
	routingLatency        int64              // Routing latency in microseconds
	tokenBucketCapacity   float64            // Token bucket capacity
	tokenBucketRefillRate float64            // Token bucket refill rate (tokens/second)
	tierShedThreshold     int                // Tier-shed overload threshold (0 = any load)
	tierShedMinPriority   int                // Tier-shed minimum admitted priority under overload
	tenantBudgets         map[string]float64 // Per-tenant fraction of total capacity (nil = no enforcement)
	sloPriorityOverrides  map[string]int     // SLO class → priority overrides (nil = GAIE defaults)
	sloTargetsMap         map[string]int64   // SLO class → TTFT target µs for slo-deadline ordering (nil = disabled)
	gaieQDThreshold       float64            // GAIE-legacy queue depth threshold per instance (default 5)
	gaieKVThreshold       float64            // GAIE-legacy KV cache utilization threshold (default 0.8)

	// routing policy config (PR 6, evolved in PR17)
	routingPolicy    string  // Routing policy name
	routingScorers   string  // Comma-separated name:weight pairs for weighted routing
	loraScorerWeight float64 // Weight of the lora-affinity scorer; 0 (default) ⇒ off (#1469)

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
	// feature inert, output byte-identical (INV-6). --speculative-acceptance-rate, a workload
	// property, is required (Changed-gated) when K > 0 to prevent the α=0 "pure slowdown"
	// footgun.
	numSpeculativeTokens  int     // K, from the scenario
	speculativeAcceptance float64 // --speculative-acceptance-rate (α ∈ [0,1])
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
	kvCPUBlocks             int64
	kvOffloadThreshold      float64
	kvTransferBandwidth     float64 // no flag sets it: the kernel prices the legacy CPU tier (KVTransferTicksPerBlock)
	kvTransferBaseLatency   int64   // no flag sets it, for the same reason
	snapshotRefreshInterval int64
	cacheSignalDelay        int64

	// PD disaggregation config
	prefillInstances       int    // Number of instances dedicated to prefill
	decodeInstances        int    // Number of instances dedicated to decode
	prefillDecodeInstances int    // Number of shared-role instances (both prefill and decode), issue #1276
	pdDecider              string // Disaggregation decider name
	pdTransferContention   bool   // Fair-share contention model; no flag sets it, dormant while the kernel prices the handoff (INV-P2-2)
	pdPrefixThreshold      int    // Non-cached token threshold for prefix-threshold decider
	prefillRoutingScorers  string // Scorer weights for prefill pool routing
	decodeRoutingScorers   string // Scorer weights for decode pool routing

	// E/P/D disaggregation config (GAP-4, issue #1264)
	encodeInstances int    // Number of instances dedicated to encoding multimodal input (0 = disabled)
	encodeDecider   string // Encode decider name: "never" (default), "always", "multimodal"

	// Autoscaler config (Phase 1C)
	modelAutoscalerIntervalUs float64 // tick interval in μs; 0 = disabled

	// Flow control config (issue #882, GIE parity)
	flowControlEnabled              bool
	flowControlDetector             string
	flowControlDispatchOrder        string
	flowControlSLOTargets           string
	flowControlMaxQueueDepth        int
	flowControlQueueDepthThreshold  float64
	flowControlKVCacheUtilThreshold float64
	flowControlMaxConcurrency       int
	flowControlPerBandCapacity      int
	flowControlUsageLimitThreshold  float64
	flowControlFairnessPolicy       string
	flowControlRequestTTL           int64
	flowControlQueueShedding        bool
	flowControlDispatchTickInterval int64
	flowControlInFlightEviction     bool

	// per-request timeout override for blis run (seconds; negative = disabled, 0 is rejected)
	requestTimeoutSecs int

	// Goodput SLO targets (issue #1413). Each is "class=duration[,class=duration...]" using
	// Go duration syntax. Distinct from --slo-targets (dispatch ordering, µs); these gate
	// goodput emission. Precedence: CLI > trace header > workload spec.
	goodputSLOTTFT string
	goodputSLOITL  string
	goodputSLOE2E  string

	// output file paths
	metricsPath string // File to write MetricsOutput JSON for blis run (--metrics-path)
	resultsPath string // File to write []SimResult JSON for blis replay (--results-path)
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
// DP-as-placement applies to an MoE model with dp>1 and no autoscaler; each replica is
// then a standalone TP engine sized per-rank (DP=1). vLLM data parallelism is N independent
// EngineCores with an internal load balancer distributing requests disjointly — the BLIS
// equivalent is N real instances behind the existing cluster router. The lumped
// single-instance DP model divided token work by dp precisely because it held every
// request; once the router splits requests across N instances each replica must be DP=1 or
// the /dp factor double-counts.
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
	return dpPlacementPlan{Active: true, Replicas: dp, PerRankDP: 1}, nil
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
	TotalKVBlocks int64 // KV blocks per instance (the dp-multiplied aggregate on the way in when autoScaledKV)
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
// autoScaledKV=true means the incoming TotalKVBlocks is a dp-multiplied aggregate, which
// is divided back to one rank and max_model_len re-capped to that rank's budget. No
// production caller passes it: the kernel sizes KV blocks and max_model_len per rank, so
// resolveDPPlacement passes false and both are left unchanged. A non-Active plan is the
// identity.
func applyDPPlacement(plan dpPlacementPlan, dp int, dep dpPlacementDeployment, autoScaledKV bool, blockSizeTokens int64) (dpPlacementDeployment, error) {
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
	if autoScaledKV {
		out.TotalKVBlocks = dep.TotalKVBlocks / int64(dp)
	}
	// A replica with no KV blocks would panic NewSimulator, and its derived kvFeasibleMax
	// of 0 would silently mean "unlimited" in the re-cap below (the inverse of a cap), so
	// it is reported as a clean error instead (R1). The kernel refuses a non-positive
	// per-rank budget, so this is defense in depth.
	if out.TotalKVBlocks <= 0 {
		return dep, fmt.Errorf("dp %d leaves %d KV blocks on each of %d engine replicas; "+
			"the scenario's per-rank KV budget must be positive",
			dp, out.TotalKVBlocks, plan.Replicas)
	}
	// A divided aggregate leaves each replica only the per-rank budget, so max_model_len
	// is re-capped to the per-rank KV-feasible maximum; otherwise per-replica
	// NewSimulator would panic ("KV cache too small for MaxModelLen").
	if autoScaledKV && out.MaxModelLen > 0 && blockSizeTokens > 0 {
		kvFeasibleMax := out.TotalKVBlocks * blockSizeTokens
		if out.MaxModelLen > kvFeasibleMax {
			logrus.Warnf("max_model_len %d exceeds per-rank KV capacity (%d blocks × %d tokens) under "+
				"DP-as-placement; capping to %d tokens", out.MaxModelLen, out.TotalKVBlocks, blockSizeTokens, kvFeasibleMax)
			out.MaxModelLen = kvFeasibleMax
		}
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
	// The kernel sized totalKVBlocks per rank already, so the placement never divides it.
	autoScaledKV := false
	dep, err := applyDPPlacement(plan, dataParallelism, dpPlacementDeployment{
		NumInstances:     numInstances,
		TotalKVBlocks:    totalKVBlocks,
		MaxModelLen:      maxModelLen,
		PrefillInstances: prefillInstances,
		DecodeInstances:  decodeInstances,
		SharedInstances:  prefillDecodeInstances,
		EncodeInstances:  encodeInstances,
	}, autoScaledKV, blockSizeTokens)
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
func adoptSchedulingPolicy(cmd *cobra.Command, m *kernelmodel.Model) {
	eng, err := m.Engine()
	if err != nil || eng.SchedulingPolicy == "" {
		return
	}
	stated, ok := map[string]string{"fcfs": "fcfs", "priority": "priority-fcfs"}[eng.SchedulingPolicy]
	if !ok {
		logrus.Fatalf("scenario %q states engine.scheduling_policy %q; BLIS simulates vLLM's "+
			"fcfs and priority", kernelScenario, eng.SchedulingPolicy)
	}
	if !cmd.Flags().Changed("scheduler") {
		scheduler = stated
		return
	}
	if scheduler != stated {
		logrus.Warnf("--scheduler %s overrides scenario %q's engine.scheduling_policy %s (BLIS %s)",
			scheduler, kernelScenario, eng.SchedulingPolicy, stated)
	}
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
	missing := []string{}
	if kernelScenario == "" {
		missing = append(missing, "--scenario")
	}
	if kernelScenarioDir == "" {
		missing = append(missing, "--scenarios")
	}
	if kernelRegistry == "" {
		missing = append(missing, "--registry")
	}
	if len(missing) > 0 {
		logrus.Fatalf("blis run requires %s: the kernel prices a step from a "+
			"committed scenario plus the catalog and registry it names",
			strings.Join(missing, ", "))
	}
	// The scenario's roles and the CLI topology describe one deployment. A disaggregated
	// scenario run without a P/D topology would serve every request on its first pool's
	// engine, as though colocated; the opposite mismatch is refused when the pools open.
	shape, err := kernelmodel.ShapeOf(kernelScenario, kernelmodel.Repos{Scenarios: kernelScenarioDir})
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	// A colocated run simulates one pool's engine; a second colocated pool would be dropped.
	if !slices.Contains(shape.Roles, deployment.RolePrefill) && len(shape.Roles) > 1 {
		logrus.Fatalf("scenario %q states %d colocated pools; a run simulates one colocated "+
			"engine, so state one pool and scale it with --num-instances", kernelScenario, len(shape.Roles))
	}
	// The scenario's offload block and --kv-offload-config describe the same tiers in two
	// formats that are not yet one (#1910). Simulating without a hierarchy the scenario states
	// would drop it silently (R1), so it is refused until the scenario is the one source.
	if shape.StatesOffload {
		logrus.Fatalf("scenario %q states a deployment offload block, which BLIS does not "+
			"read yet (#1910); describe the tiers with --kv-offload-config and remove the block "+
			"from the scenario", kernelScenario)
	}
	if slices.Contains(shape.Roles, deployment.RolePrefill) && prefillInstances == 0 && decodeInstances == 0 {
		logrus.Fatalf("scenario %q is disaggregated (it states prefill and decode "+
			"pools), so the run needs a P/D topology: pass --prefill-instances and --decode-instances",
			kernelScenario)
	}
	catalogRoot, err := resolveCatalogRoot()
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	m, err := kernelmodel.Open(kernelScenario, kernelmodel.Repos{
		Scenarios: kernelScenarioDir, Catalog: catalogRoot, Registry: kernelRegistry,
	})
	if err != nil {
		// Fatal rather than fallen back on: there is no other latency model to serve the run.
		logrus.Fatalf("scenario %q: %v",
			kernelScenario, err)
	}
	settings, err := m.Settings()
	if err != nil {
		logrus.Fatalf("scenario %q: %v", kernelScenario, err)
	}
	kernelOpened = m
	adoptSchedulingPolicy(cmd, m)
	dep := m.Deployment()
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
		"tokens, max_num_seqs %d, max_num_batched_tokens %d, max_model_len %d, prefix caching %t, "+
		"speculative tokens %d",
		kernelScenario, model, gpu,
		tensorParallelism, dataParallelism, enableExpertParallel,
		totalKVBlocks, blockSizeTokens, maxNumSeqs, maxNumBatchedTokens, maxModelLen,
		!noEnablePrefixCaching, numSpeculativeTokens)
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
		m, err = kernelmodel.Open(kernelScenario, kernelmodel.Repos{
			Scenarios: kernelScenarioDir, Catalog: catalogRoot, Registry: kernelRegistry,
		})
		if err != nil {
			logrus.Fatalf("scenario %q: %v", kernelScenario, err)
		}
	}
	// The expert geometry the graph stated. Nothing else of ModelConfig is populated: no HF
	// config is parsed, and a half-filled struct read as complete is how a dense model comes
	// to be treated as MoE.
	var modelConfig sim.ModelConfig
	modelConfig.NumLocalExperts = kernelDeploymentExperts
	modelConfig.NumExpertsPerTok = kernelDeploymentTopK

	if dataParallelism < 1 {
		logrus.Fatalf("scenario %q states dp %d; it must be >= 1", kernelScenario, dataParallelism)
	}
	if dataParallelism > 1 && !modelConfig.IsMoE() {
		// Dense DP is the router-replica mechanism, not a latency divisor: vLLM's
		// --data-parallel-size on a dense model runs N independent engines behind an
		// internal load balancer, which BLIS expresses as N instances.
		logrus.Fatalf("scenario %q states dp %d for a dense model; DP placement is modelled for MoE "+
			"models only. Express N independent dense engines as --num-instances N with dp 1.",
			kernelScenario, dataParallelism)
	}
	if enableExpertParallel && !modelConfig.IsMoE() {
		// vLLM fatally rejects --enable-expert-parallel on dense models
		// (config/model.py:_verify_with_expert_parallelism).
		logrus.Fatalf("scenario %q enables expert parallelism for dense model %q, which vLLM refuses",
			kernelScenario, model)
	}
	return latencyResolution{
		KernelModel: m,
		ModelConfig: modelConfig,
	}
}

// composeLoRAScorer appends the lora-affinity scorer at the given weight to the
// effective weighted-routing profile (#1469). When base is empty (no explicit
// --routing-scorers or bundle), it materializes the default profile first so the
// LoRA scorer composes alongside the standard dimensions rather than replacing
// them. Returns an error for a non-finite/non-positive weight or when base already
// declares lora-affinity (double-specification via both --routing-scorers and
// --lora-scorer-weight). The returned slice never aliases base's backing array.
func composeLoRAScorer(base []sim.ScorerConfig, weight float64) ([]sim.ScorerConfig, error) {
	if weight <= 0 || math.IsNaN(weight) || math.IsInf(weight, 0) {
		return nil, fmt.Errorf("weight must be a finite positive number, got %v", weight)
	}
	if len(base) == 0 {
		base = sim.DefaultScorerConfigs()
	}
	for _, sc := range base {
		if sc.Name == "lora-affinity" {
			return nil, fmt.Errorf("lora-affinity already present in the scorer profile; set its weight via --routing-scorers OR --lora-scorer-weight, not both")
		}
	}
	composed := make([]sim.ScorerConfig, 0, len(base)+1)
	composed = append(composed, base...)
	composed = append(composed, sim.ScorerConfig{Name: "lora-affinity", Weight: weight})
	return composed, nil
}

// resolvePolicies resolves admission/routing/priority/scheduler policy configuration
// from CLI flags and an optional policy bundle YAML file. It is called by both runCmd
// and replayCmd to ensure a single validation code path (R23: code path parity).
//
// Precondition: adoptKernelDeployment must be called first. blockSizeTokens comes from the
// scenario, validated by the kernel when it opens; resolvePolicies does not re-validate it.
//
// Side effects: may write admissionPolicy, routingPolicy, scheduler,
// tokenBucketCapacity, tokenBucketRefillRate, tierShedThreshold, tierShedMinPriority,
// tenantBudgets package-level vars (from policy bundle).
//
// Returns the parsed scorer configs for weighted routing (caller uses these in
// DeploymentConfig.RoutingScorerConfigs) and the loaded policy bundle (nil if none).
// Per-pool scorer configs (PD disaggregation) are NOT handled here — they remain inline
// in runCmd.
func resolvePolicies(cmd *cobra.Command) ([]sim.ScorerConfig, *sim.PolicyBundle) {
	var bundleScorerConfigs []sim.ScorerConfig
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
		// Apply bundle values as defaults; CLI flags override via Changed().
		if bundle.Admission.Policy != "" && !cmd.Flags().Changed("admission-policy") {
			admissionPolicy = bundle.Admission.Policy
		}
		if bundle.Admission.TokenBucketCapacity != nil && !cmd.Flags().Changed("token-bucket-capacity") {
			tokenBucketCapacity = *bundle.Admission.TokenBucketCapacity
		}
		if bundle.Admission.TokenBucketRefillRate != nil && !cmd.Flags().Changed("token-bucket-refill-rate") {
			tokenBucketRefillRate = *bundle.Admission.TokenBucketRefillRate
		}
		if bundle.Admission.TierShedThreshold != nil {
			tierShedThreshold = *bundle.Admission.TierShedThreshold
		}
		if bundle.Admission.TierShedMinPriority != nil {
			tierShedMinPriority = *bundle.Admission.TierShedMinPriority
		} else if bundle.Admission.Policy == "tier-shed" && bundle.Admission.TierShedMinPriority == nil {
			tierShedMinPriority = 3 // default: protect Critical (4) and Standard (3)
		}
		if bundle.TenantBudgets != nil {
			tenantBudgets = bundle.TenantBudgets
		}
		if bundle.Admission.SLOPriorities != nil {
			sloPriorityOverrides = bundle.Admission.SLOPriorities
			// Validate at CLI boundary (R3) before reaching library-level panic in NewSLOPriorityMap.
			for k, v := range sloPriorityOverrides {
				if v < -100 || v > 100 {
					logrus.Fatalf("slo_priorities override for %q: value %d out of range [-100, 100]", k, v)
				}
			}
		}
		if bundle.Admission.SLOTargets != nil && sloTargetsMap == nil {
			sloTargetsMap = bundle.Admission.SLOTargets
		}
		if bundle.Admission.GAIEQDThreshold != nil {
			gaieQDThreshold = *bundle.Admission.GAIEQDThreshold
		}
		if bundle.Admission.GAIEKVThreshold != nil {
			gaieKVThreshold = *bundle.Admission.GAIEKVThreshold
		}
		if bundle.Routing.Policy != "" && !cmd.Flags().Changed("routing-policy") {
			routingPolicy = bundle.Routing.Policy
		}
		bundleScorerConfigs = bundle.Routing.Scorers
		if bundle.Priority.Policy != "" && bundle.Priority.Policy != "constant" {
			logrus.Warnf("bundle priority.policy=%q has no effect: --priority-policy was removed in PR #1216. "+
				"Request priority is now static (set at enqueue via SLOPriorityMap.InvertForVLLM). "+
				"Use SLOClass in your workload spec for tier-based priority.", bundle.Priority.Policy)
		}
		if bundle.Scheduler != "" && !cmd.Flags().Changed("scheduler") {
			scheduler = bundle.Scheduler
		}
		if bundle.Preemption.Policy != "" && !cmd.Flags().Changed("preemption-policy") {
			preemptionPolicy = bundle.Preemption.Policy
		}
	}

	// Apply defaults for GAIE-legacy thresholds (not set via CLI flags, only via bundle).
	if gaieQDThreshold == 0 {
		gaieQDThreshold = 5
	}
	if gaieKVThreshold == 0 {
		gaieKVThreshold = 0.8
	}

	// Policy name validation (R3: validate at CLI boundary before passing to library)
	if admissionPolicy == "token-bucket" {
		if tokenBucketCapacity <= 0 || math.IsNaN(tokenBucketCapacity) || math.IsInf(tokenBucketCapacity, 0) {
			logrus.Fatalf("--token-bucket-capacity must be a finite value > 0, got %v", tokenBucketCapacity)
		}
		if tokenBucketRefillRate <= 0 || math.IsNaN(tokenBucketRefillRate) || math.IsInf(tokenBucketRefillRate, 0) {
			logrus.Fatalf("--token-bucket-refill-rate must be a finite value > 0, got %v", tokenBucketRefillRate)
		}
	}
	if admissionPolicy == "gaie-legacy" {
		if gaieQDThreshold <= 0 || math.IsNaN(gaieQDThreshold) || math.IsInf(gaieQDThreshold, 0) {
			logrus.Fatalf("gaie_qd_threshold must be > 0, got %v", gaieQDThreshold)
		}
		if gaieKVThreshold <= 0 || gaieKVThreshold > 1.0 || math.IsNaN(gaieKVThreshold) || math.IsInf(gaieKVThreshold, 0) {
			logrus.Fatalf("gaie_kv_threshold must be in (0, 1.0], got %v", gaieKVThreshold)
		}
	}
	if !sim.IsValidAdmissionPolicy(admissionPolicy) {
		logrus.Fatalf("Unknown admission policy %q. Valid: %s", admissionPolicy, strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
	}
	if !sim.IsValidRoutingPolicy(routingPolicy) {
		logrus.Fatalf("Unknown routing policy %q. Valid: %s", routingPolicy, strings.Join(sim.ValidRoutingPolicyNames(), ", "))
	}
	if !sim.IsValidScheduler(scheduler) {
		logrus.Fatalf("Unknown scheduler %q. Valid: %s", scheduler, strings.Join(sim.ValidSchedulerNames(), ", "))
	}
	if !sim.IsValidPreemptionPolicy(preemptionPolicy) {
		logrus.Fatalf("Unknown preemption policy %q. Valid: %s", preemptionPolicy, strings.Join(sim.ValidPreemptionPolicyNames(), ", "))
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
	if snapshotRefreshInterval < 0 {
		logrus.Fatalf("--snapshot-refresh-interval must be >= 0, got %d", snapshotRefreshInterval)
	}
	if cacheSignalDelay < 0 {
		logrus.Fatalf("--cache-signal-delay must be >= 0, got %d", cacheSignalDelay)
	}
	if admissionLatency < 0 {
		logrus.Fatalf("--admission-latency must be >= 0, got %d", admissionLatency)
	}
	if routingLatency < 0 {
		logrus.Fatalf("--routing-latency must be >= 0, got %d", routingLatency)
	}
	// Flow control validation (R3: validate at CLI boundary before passing to library)
	if flowControlEnabled {
		if !sim.IsValidSaturationDetector(flowControlDetector) {
			logrus.Fatalf("Unknown saturation detector %q. Valid: %s", flowControlDetector, strings.Join(sim.ValidSaturationDetectorNames(), ", "))
		}
		if flowControlDispatchOrder != "fifo" && flowControlDispatchOrder != "priority" && flowControlDispatchOrder != "slo-deadline" {
			logrus.Fatalf("--dispatch-order must be 'fifo', 'priority', or 'slo-deadline', got %q", flowControlDispatchOrder)
		}
		if flowControlMaxQueueDepth < 0 {
			logrus.Fatalf("--max-gateway-queue-depth must be >= 0, got %d", flowControlMaxQueueDepth)
		}
		if flowControlPerBandCapacity < 0 {
			logrus.Fatalf("--per-band-capacity must be >= 0, got %d", flowControlPerBandCapacity)
		}
		if flowControlUsageLimitThreshold <= 0 || flowControlUsageLimitThreshold > 1.0 {
			logrus.Fatalf("--usage-limit-threshold must be in (0, 1.0], got %v", flowControlUsageLimitThreshold)
		}
		if flowControlFairnessPolicy != "global-strict" && flowControlFairnessPolicy != "round-robin" {
			logrus.Fatalf("--fairness-policy must be 'global-strict' or 'round-robin', got %q", flowControlFairnessPolicy)
		}
		if flowControlUsageLimitThreshold < 1.0 && flowControlDispatchOrder == "fifo" {
			logrus.Warnf("--usage-limit-threshold < 1.0 with --dispatch-order fifo: HoL blocking uses priority-order iteration, FIFO semantics will not apply to gating decisions")
		}
		// Parse --slo-targets into map (e.g. "critical=100000,standard=500000")
		if flowControlSLOTargets != "" {
			sloTargetsMap = make(map[string]int64)
			for _, pair := range strings.Split(flowControlSLOTargets, ",") {
				parts := strings.SplitN(strings.TrimSpace(pair), "=", 2)
				if len(parts) != 2 {
					logrus.Fatalf("--slo-targets: invalid format %q (expected key=value)", pair)
				}
				key := strings.TrimSpace(parts[0])
				if key == "" {
					logrus.Fatalf("--slo-targets: empty key in pair %q (expected key=value)", pair)
				}
				v, parseErr := strconv.ParseInt(strings.TrimSpace(parts[1]), 10, 64)
				if parseErr != nil || v <= 0 {
					logrus.Fatalf("--slo-targets: value for %q must be a positive integer (µs), got %s", key, strings.TrimSpace(parts[1]))
				}
				sloTargetsMap[key] = v
			}
		}
		if sloTargetsMap != nil && flowControlDispatchOrder != "slo-deadline" {
			logrus.Warnf("--slo-targets has no effect without --dispatch-order slo-deadline")
		}
		// Validate only parameters consumed by the selected detector
		switch flowControlDetector {
		case "utilization":
			if flowControlQueueDepthThreshold <= 0 || math.IsNaN(flowControlQueueDepthThreshold) || math.IsInf(flowControlQueueDepthThreshold, 0) {
				logrus.Fatalf("--queue-depth-threshold must be a finite value > 0, got %v", flowControlQueueDepthThreshold)
			}
			if flowControlKVCacheUtilThreshold <= 0 || math.IsNaN(flowControlKVCacheUtilThreshold) || math.IsInf(flowControlKVCacheUtilThreshold, 0) {
				logrus.Fatalf("--kv-cache-util-threshold must be a finite value > 0, got %v", flowControlKVCacheUtilThreshold)
			}
		case "concurrency":
			if flowControlMaxConcurrency <= 0 {
				logrus.Fatalf("--max-concurrency must be > 0, got %d", flowControlMaxConcurrency)
			}
		case "", "never":
			logrus.Warnf("--flow-control enabled but --saturation-detector is %q (pass-through); specify 'utilization' or 'concurrency' for actual gating", flowControlDetector)
		}
	}
	if flowControlRequestTTL < 0 {
		logrus.Fatalf("--request-ttl must be >= 0, got %d", flowControlRequestTTL)
	}
	if flowControlRequestTTL > 0 && !flowControlEnabled {
		logrus.Warnf("--request-ttl %d has no effect without --flow-control", flowControlRequestTTL)
	}
	if flowControlDispatchTickInterval < 0 {
		logrus.Fatalf("--dispatch-tick-interval must be >= 0, got %d", flowControlDispatchTickInterval)
	}
	if flowControlQueueShedding && !flowControlEnabled {
		logrus.Warnf("--queue-shedding has no effect without --flow-control")
	}
	if flowControlInFlightEviction && !flowControlEnabled {
		logrus.Warnf("--in-flight-eviction has no effect without --flow-control")
	}

	logrus.Infof("Policy config: admission=%s, routing=%s, scheduler=%s, preemption=%s",
		admissionPolicy, routingPolicy, scheduler, preemptionPolicy)

	// Parse scorer configuration for weighted routing
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
		// Compose the lora-affinity scorer (#1469). Left unset the flag is inert, so
		// routing is byte-identical to today (INV-6). When set (to a positive weight;
		// an explicit non-positive value is rejected by composeLoRAScorer), append it
		// to the effective profile — materializing the default base when no explicit
		// --routing-scorers/bundle profile was given — so the LoRA scorer participates
		// alongside the existing dimensions.
		if cmd.Flags().Changed("lora-scorer-weight") {
			composed, err := composeLoRAScorer(parsedScorerConfigs, loraScorerWeight)
			if err != nil {
				logrus.Fatalf("Invalid --lora-scorer-weight: %v", err)
			}
			parsedScorerConfigs = composed
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
	if routingPolicy != "weighted" && cmd.Flags().Changed("lora-scorer-weight") {
		logrus.Warnf("--lora-scorer-weight has no effect when routing policy is %q (only applies to 'weighted')", routingPolicy)
	}
	if admissionPolicy == "token-bucket" {
		logrus.Infof("Token bucket: capacity=%.0f, refill-rate=%.0f", tokenBucketCapacity, tokenBucketRefillRate)
	}

	return parsedScorerConfigs, loadedBundle
}

// batchConfigFromCLI is the single run/replay wiring seam for vLLM batch settings.
// Keeping both commands on this helper makes the prefix-caching toggle structurally
// symmetric for INV-13 rather than relying on two call sites to stay in sync.
func batchConfigFromCLI() sim.BatchConfig {
	return sim.NewBatchConfig(maxNumSeqs, maxNumBatchedTokens, longPrefillTokenThreshold,
		sim.WithPrefixCachingDisabled(noEnablePrefixCaching))
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

	// Engine scheduling knobs the deployment schema does not state. Every other engine
	// setting -- block size, sequence and token caps, max_model_len, prefix caching, cache
	// dtype, speculation -- is the scenario's, and the KV budget is the kernel's.
	cmd.Flags().Int64Var(&longPrefillTokenThreshold, "long-prefill-token-threshold", 0, "Max length of prefill beyond which chunked prefill is triggered")

	// The deployment: a blis-schemas scenario (model, hardware, fabric, pools, engines)
	// priced by blis-latency-kernel with the coefficient sets it names.
	cmd.Flags().StringVar(&kernelScenario, "scenario", "", "Scenario FILE NAME within --scenarios, e.g. gpt-oss-120b-h200-fp4-vllm-tp4.yaml (REQUIRED): the blis-schemas Scenario and Deployment that state the model, hardware, fabric, each pool's parallelism and engine settings, and the coefficient sets that price it")
	cmd.Flags().StringVar(&kernelScenarioDir, "scenarios", "", "Directory holding scenario YAML files (REQUIRED)")
	cmd.Flags().StringVar(&kernelRegistry, "registry", "", "blis-registry clone root, holding the coefficient sets a scenario names (REQUIRED)")

	// Cluster config
	cmd.Flags().IntVar(&numInstances, "num-instances", 1, "Number of instances in the cluster")

	// Online routing pipeline config
	cmd.Flags().StringVar(&admissionPolicy, "admission-policy", "always-admit", "Admission policy: "+strings.Join(sim.ValidAdmissionPolicyNames(), ", "))
	cmd.Flags().Int64Var(&admissionLatency, "admission-latency", 0, "Admission latency in microseconds")
	cmd.Flags().Int64Var(&routingLatency, "routing-latency", 0, "Routing latency in microseconds")
	cmd.Flags().Float64Var(&tokenBucketCapacity, "token-bucket-capacity", 10000, "Token bucket capacity")
	cmd.Flags().Float64Var(&tokenBucketRefillRate, "token-bucket-refill-rate", 1000, "Token bucket refill rate (tokens/second)")

	// Routing policy config
	cmd.Flags().StringVar(&routingPolicy, "routing-policy", "round-robin", "Routing policy: round-robin, least-loaded, weighted, always-busiest")
	cmd.Flags().StringVar(&routingScorers, "routing-scorers", "", "Scorer weights for weighted routing (e.g., queue-depth:2,kv-utilization:2,load-balance:1). Default: precise-prefix-cache:2,queue-depth:1,kv-utilization:1")
	cmd.Flags().Float64Var(&loraScorerWeight, "lora-scorer-weight", 0, "Weight of the lora-affinity routing scorer, composed into the weighted profile. Leave unset to keep routing unchanged; must be a finite positive number when set. Requires --routing-policy weighted (#1469)")

	// Scheduler and preemption config
	cmd.Flags().StringVar(&scheduler, "scheduler", "fcfs", "Instance scheduler: fcfs, priority-fcfs, sjf, reverse-priority")
	cmd.Flags().StringVar(&preemptionPolicy, "preemption-policy", "fcfs", "Preemption victim selection: fcfs (tail-of-batch), priority (least-urgent SLO tier)")

	// Policy bundle config
	cmd.Flags().StringVar(&policyConfigPath, "policy-config", "", "Path to YAML policy configuration file")

	// Fitness evaluation config (PR9)
	cmd.Flags().StringVar(&fitnessWeights, "fitness-weights", "", "Fitness weights as key:value pairs (e.g., throughput:0.5,p99_ttft:0.3)")

	// Decision trace config (PR13)
	cmd.Flags().StringVar(&traceLevel, "trace-level", "none", "Trace verbosity: none, decisions")
	cmd.Flags().IntVar(&counterfactualK, "counterfactual-k", 0, "Number of counterfactual candidates per routing decision")
	cmd.Flags().BoolVar(&summarizeTrace, "summarize-trace", false, "Print trace summary after simulation")

	// Tiered KV cache (PR12)
	cmd.Flags().Int64Var(&kvCPUBlocks, "kv-cpu-blocks", 0, "CPU tier KV cache blocks (0 = disabled, single-tier mode). Typical: 1/3 of the kernel's per-rank GPU KV blocks")
	cmd.Flags().Float64Var(&kvOffloadThreshold, "kv-offload-threshold", 0.9, "GPU utilization (0-1) above which blocks are offloaded to CPU. Default: offload when GPU >90% full")
	cmd.Flags().Int64Var(&snapshotRefreshInterval, "snapshot-refresh-interval", 50000, "Prometheus snapshot refresh interval for all instance metrics in microseconds (0 = immediate/oracle mode, default 50ms = llm-d parity)")
	cmd.Flags().Int64Var(&cacheSignalDelay, "cache-signal-delay", cluster.DefaultCacheSignalDelay, "Propagation delay for prefix cache signals in microseconds. Only affects precise-prefix-cache and no-hit-lru scorers; no effect on other routing policies. Default 50ms. Set to 0 for oracle mode (live cache state).")
	cmd.Flags().Float64Var(&modelAutoscalerIntervalUs, "model-autoscaler-interval-us", 0, "Autoscaler tick interval in microseconds (0 = disabled). Overrides policy-config autoscaler.interval_us when non-zero.")

	// PD disaggregation config
	cmd.Flags().IntVar(&prefillInstances, "prefill-instances", 0, "Number of instances dedicated to prefill (0 = disabled)")
	cmd.Flags().IntVar(&decodeInstances, "decode-instances", 0, "Number of instances dedicated to decode (0 = disabled)")
	cmd.Flags().IntVar(&prefillDecodeInstances, "prefill-decode-instances", 0, "Number of shared-role instances serving both prefill and decode (llm-d 'prefill-decode'/'both' parity; 0 = disabled). Must satisfy --prefill-instances + --decode-instances + --prefill-decode-instances <= --num-instances.")
	cmd.Flags().StringVar(&pdDecider, "pd-decider", "never", "PD disaggregation decider: never (default), always, prefix-threshold")
	cmd.Flags().IntVar(&pdPrefixThreshold, "pd-prefix-threshold", 16, "Non-cached token threshold for prefix-threshold decider (>= 0); disaggregate when non-cached tokens exceed this value. Default 16 matches llm-d's shipped P/D configs (deploy/config/pd-epp-config.yaml).")
	cmd.Flags().StringVar(&prefillRoutingScorers, "prefill-routing-scorers", "", "Scorer weights for prefill pool routing (e.g., queue-depth:2,kv-utilization:2)")
	cmd.Flags().StringVar(&decodeRoutingScorers, "decode-routing-scorers", "", "Scorer weights for decode pool routing (e.g., queue-depth:2,kv-utilization:2)")

	// E/P/D disaggregation (GAP-4, issue #1264). Registered on both run and replay.
	cmd.Flags().IntVar(&encodeInstances, "encode-instances", 0, "Number of instances dedicated to encoding multimodal input (0 = encode pool disabled, default)")
	cmd.Flags().StringVar(&encodeDecider, "encode-decider", "never", "Encode decider: never (default), always, multimodal")

	// Flow control config (issue #882, GIE parity)
	cmd.Flags().BoolVar(&flowControlEnabled, "flow-control", false, "Enable gateway queue with saturation-gated dispatch (GIE flow control)")
	cmd.Flags().StringVar(&flowControlDetector, "saturation-detector", "never", "Saturation detector: "+strings.Join(sim.ValidSaturationDetectorNames(), ", "))
	cmd.Flags().StringVar(&flowControlDispatchOrder, "dispatch-order", "fifo", "Gateway queue dispatch order: fifo, priority, slo-deadline")
	cmd.Flags().StringVar(&flowControlSLOTargets, "slo-targets", "", "Per-SLO-class TTFT targets in µs for slo-deadline ordering (e.g., critical=100000,standard=500000)")
	cmd.Flags().IntVar(&flowControlMaxQueueDepth, "max-gateway-queue-depth", 0, "Max gateway queue depth (0=unlimited)")
	cmd.Flags().Float64Var(&flowControlQueueDepthThreshold, "queue-depth-threshold", 5, "Queue depth threshold for utilization detector")
	cmd.Flags().Float64Var(&flowControlKVCacheUtilThreshold, "kv-cache-util-threshold", 0.8, "KV cache utilization threshold for utilization detector")
	cmd.Flags().IntVar(&flowControlMaxConcurrency, "max-concurrency", 100, "Max concurrency per instance for concurrency detector")
	cmd.Flags().IntVar(&flowControlPerBandCapacity, "per-band-capacity", 0, "Max requests per priority band when --flow-control is enabled (0=unlimited)")
	cmd.Flags().Float64Var(&flowControlUsageLimitThreshold, "usage-limit-threshold", 1.0, "Per-band saturation ceiling for HoL blocking (1.0=no HoL, <1.0 gates lower-priority bands earlier)")
	cmd.Flags().StringVar(&flowControlFairnessPolicy, "fairness-policy", "global-strict", "Intra-band dispatch fairness: global-strict, round-robin")
	cmd.Flags().Int64Var(&flowControlRequestTTL, "request-ttl", 0, "Gateway queue request TTL in microseconds (0=disabled). Requires --flow-control.")
	cmd.Flags().BoolVar(&flowControlQueueShedding, "queue-shedding", false, "Enable cross-band victim shedding when gateway queue is full (BLIS-extra, not in llm-d; default: reject)")
	cmd.Flags().Int64Var(&flowControlDispatchTickInterval, "dispatch-tick-interval", 1000, "Microseconds between periodic gateway dispatch ticks (0 = use default 1ms; llm-d parity)")
	cmd.Flags().BoolVar(&flowControlInFlightEviction, "in-flight-eviction", false, "Enable in-flight eviction of sheddable requests when saturated (BLIS-extra, not in llm-d; requires --flow-control)")

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
	cmd.Flags().Float64Var(&speculativeAcceptance, "speculative-acceptance-rate", 0.0, "Speculative decoding: mean fraction of draft tokens accepted, in [0,1] -- a property of the workload, not the deployment. Required when the scenario's engine drafts tokens (engine.speculative).")

	// KV-cache offload config surface (H5, #1587). One flag: a strict-YAML file with a
	// single top-level kv_offload: block (CPU tier + ordered secondary tiers, per-tier
	// device physics). Registered on run and replay (INV-13). Absent => the offload
	// subsystem is inert and output is byte-identical to a build without the feature
	// (BC-G5). On replay the trace header is authoritative; a passed flag must match the
	// header (see resolveKVOffloadConfig / the replay wiring). device_class names resolve
	// against the CATALOG's storage-device table, <catalog>/devices/storage.yaml (#1770),
	// read only when some tier actually names a class.
	cmd.Flags().StringVar(&kvOffloadConfigPath, "kv-offload-config", "", "Path to a YAML file with a top-level kv_offload: block (multi-tier KV-cache offload config: cpu_bytes_to_use, block_size/blocks_per_chunk, eviction_policy, offload_prompt_only, secondary_tiers[] with per-tier device_class/direct_io). Every secondary tier names a device_class from the catalog's storage-device table at <catalog>/"+catalogStorageDevicesRelPath+" (located by --catalog / "+catalogEnvVar+"); the kernel prices its transfers from that device, so a tier stating its own read_bandwidth/write_bandwidth/base_latency is refused. Absent => offload subsystem inert. On replay the trace header is authoritative.")
}

// loraConfigFile is the on-disk shape of a --lora-config YAML file: a single
// top-level lora: block matching contracts/config-schema.md. Strict-parsed (R10).
type loraConfigFile struct {
	LoRA sim.LoRAConfig `yaml:"lora"`
}

// loadLoRAConfigFile parses a --lora-config YAML file's lora: block into a
// sim.LoRAConfig. Strict field checking (R10). CLI boundary => logrus.Fatalf on any
// read/parse error so a typo never silently no-ops.
func loadLoRAConfigFile(path string) sim.LoRAConfig {
	data, err := os.ReadFile(path)
	if err != nil {
		logrus.Fatalf("Failed to read --lora-config file %q: %v", path, err)
	}
	var f loraConfigFile
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(&f); err != nil {
		logrus.Fatalf("Failed to parse --lora-config file %q: %v", path, err)
	}
	return f.LoRA
}

// resolveLoRAConfig assembles the final sim.LoRAConfig from three composable sources,
// in increasing precedence: defaults.yaml (cost-coefficient fallback), the optional
// --lora-config file (adapter registry + any coefficients it sets), and the scalar
// --lora-* flags (applied only when Changed, R18). It is the single LoRAConfig
// construction site (R4), called by BOTH runCmd and replayCmd (INV-13 parity).
//
// The resolved config is validated at the CLI boundary; an invalid config aborts with
// logrus.Fatalf (Principle V) — e.g. adapters declared with adapter_capacity 0.
//
// With no --lora-config and no --lora-* flags set, the returned config still carries
// the defaults.yaml cost coefficients but declares no adapters, so HasAdapters() is
// false and the subsystem is inert (INV-6 no-op default; coefficients are unused
// until adapters exist).
func resolveLoRAConfig(cmd *cobra.Command) sim.LoRAConfig {
	var cfg sim.LoRAConfig
	if loraConfigPath != "" {
		cfg = loadLoRAConfigFile(loraConfigPath)
	}

	// defaults.yaml cost-coefficient fallback: fill only fields the file did not set,
	// so an unset flag defers to the file/defaults rather than clobbering it (R18).
	if defs := loadRunDefaults(cmd.Flags().Changed("defaults-filepath")).LoRADefaults; defs != nil {
		if cfg.LoadBaseLatencyUs == nil {
			v := defs.LoadBaseLatencyUs
			cfg.LoadBaseLatencyUs = &v
		}
		if cfg.LoadBandwidthBytesUs == nil {
			v := defs.LoadBandwidthBytesUs
			cfg.LoadBandwidthBytesUs = &v
		}
		if cfg.FootprintBytesPerRank == nil {
			v := defs.FootprintBytesPerRank
			cfg.FootprintBytesPerRank = &v
		}
		if len(cfg.StepOverheadTiers) == 0 && len(defs.StepOverheadTiers) > 0 {
			cfg.StepOverheadTiers = make(map[int]sim.StepOverheadTier, len(defs.StepOverheadTiers))
			for rank, t := range defs.StepOverheadTiers {
				k6, k7 := t.K6, t.K7
				cfg.StepOverheadTiers[rank] = sim.StepOverheadTier{K6: &k6, K7: &k7}
			}
		}
	}

	// Scalar flag overrides (R18: only when explicitly set).
	if cmd.Flags().Changed("lora-adapter-capacity") {
		v := loraAdapterCapacity
		cfg.AdapterCapacity = &v
	}
	if cmd.Flags().Changed("lora-load-base-latency-us") {
		v := loraLoadBaseLatencyUs
		cfg.LoadBaseLatencyUs = &v
	}
	if cmd.Flags().Changed("lora-load-bandwidth-bytes-us") {
		v := loraLoadBandwidthBytesUs
		cfg.LoadBandwidthBytesUs = &v
	}
	if cmd.Flags().Changed("lora-footprint-bytes-per-rank") {
		v := loraFootprintBytesPerRank
		cfg.FootprintBytesPerRank = &v
	}

	if err := cfg.Validate(); err != nil {
		logrus.Fatalf("Invalid LoRA configuration: %v", err)
	}
	return cfg
}

// resolveSpeculativeConfig builds the speculative-decoding / MTP config from CLI
// flags (#1528). Shared by run and replay via the shared flag set, so a trace
// round-trips under identical flags (INV-13). Fatalf on invalid config (CLI boundary,
// R6).
func resolveSpeculativeConfig(cmd *cobra.Command) sim.SpeculativeConfig {
	// Footgun guard: a scenario drafting k>0 tokens with an unsupplied
	// --speculative-acceptance-rate would default α=0, modeling spec-decode as PURE
	// SLOWDOWN (verify width k+1 raises per-step cost while g=1 gives no throughput
	// gain) — the opposite of the feature's intent. Require α to be supplied
	// explicitly when k>0 (α=0 stays legal, but must be a deliberate choice). Mirrors
	// the codebase's Changed()-gated required-flag idiom.
	if numSpeculativeTokens > 0 && !cmd.Flags().Changed("speculative-acceptance-rate") {
		logrus.Fatalf("--speculative-acceptance-rate is required when scenario %q drafts %d tokens "+
			"(engine.speculative): set it explicitly, e.g. --speculative-acceptance-rate 0.7; use 0 only "+
			"to deliberately model 0%% acceptance", kernelScenario, numSpeculativeTokens)
	}
	c, err := sim.NewSpeculativeConfig(numSpeculativeTokens, speculativeAcceptance, speculativeMethod)
	if err != nil {
		logrus.Fatalf("%v", err)
	}
	if c.IsEnabled() {
		logrus.Infof("speculative decoding enabled: method=%q k=%d acceptance=%.3f (mean %.3f tokens/step, verify width %d)",
			c.Method, c.K, c.Acceptance, c.EffectiveTokensPerStep(), c.VerifyWidth())
		if c.Acceptance == 0 {
			logrus.Warnf("speculative decoding: acceptance-rate=0 => no throughput gain but verify-width cost applies (net slowdown); this is usually not intended")
		}
	}
	return c
}

// adapterReservedBytesFor returns the static LoRA HBM reservation (bytes) to carve
// out of the KV budget for a resolved config, obtained through the sim/lora cost
// model's pure AdapterReservedBytes() query (design boundary #4 — the memory path
// never reaches into sim/lora internals). It routes through sim.BuildAdapterCost so
// the activation condition matches NewSimulator's resident-set/cost wiring exactly
// (R4): 0 when the subsystem is inert (no adapters, no capacity, or sim/lora not
// linked), leaving KV capacity byte-identical to today (INV-6). A malformed cost
// config aborts at the CLI boundary (Principle V), the same check NewSimulator makes.
//
// This deliberately builds an adapter-cost model that sim.NewSimulator (the
// cold-load gate) and sim/cluster.NewInstanceSimulator (the latency-model overhead
// wrapper, #1467) each also build from the same config via sim.BuildAdapterCost. The model
// is a pure, stateless value object, so independent builds from one config are
// behaviorally identical — the extra one-time O(adapters) construction at startup is
// the established BuildAdapterCost pattern, not a caching bug.
func adapterReservedBytesFor(cfg sim.LoRAConfig) int64 {
	ac, err := sim.BuildAdapterCost(sim.SimConfig{LoRAConfig: cfg})
	if err != nil {
		logrus.Fatalf("Invalid LoRA configuration (HBM reservation): %v", err)
	}
	if ac == nil {
		return 0
	}
	// Defense in depth: NewCostModel already rejects a non-finite reservation and
	// caps it below maxReservedBytes (< math.MaxInt64), so this conversion is exact
	// and non-negative for any model built through it. Guard the CLI boundary
	// explicitly anyway — so a future construction path that bypasses that check can
	// never silently truncate a huge/±Inf float64 to a garbage int64 (Go's
	// out-of-range float→int conversion is implementation-defined). Principle V:
	// fail at the CLI boundary, not deep in the KV-capacity library.
	//
	// int64(x) is well-defined and exact for every representable float64 strictly
	// below 2^63. The trap is float64(math.MaxInt64): MaxInt64 (2^63-1) is not
	// representable in float64 and rounds UP to 2^63, so it must NOT be used as the
	// bound (int64(2^63) overflows). We reject at the comfortably-conservative,
	// exactly-representable 2^62 (≈4.6e18) — far above any real reservation (the cost
	// model caps at 1e18) — so the cast is provably safe without relying on the exact
	// 2^63 edge.
	const maxSafeReservedBytes = float64(int64(1) << 62)
	reserved := ac.AdapterReservedBytes()
	if math.IsNaN(reserved) || math.IsInf(reserved, 0) || reserved < 0 || reserved >= maxSafeReservedBytes {
		logrus.Fatalf("Invalid LoRA configuration (HBM reservation): %v bytes is outside the representable range", reserved)
	}
	return int64(reserved)
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
		if longPrefillTokenThreshold < 0 {
			logrus.Fatalf("--long-prefill-token-threshold must be >= 0, got %d", longPrefillTokenThreshold)
		}
		// Changed() guard: unlike peer flags (default always positive), --horizon defaults
		// to math.MaxInt64 which would fail <= 0. Only validate when user explicitly sets it.
		if cmd.Flags().Changed("horizon") && simulationHorizon <= 0 {
			logrus.Fatalf("--horizon must be > 0, got %d", simulationHorizon)
		}

		// Resolve policy configuration (single code path shared with replayCmd).
		// Per-pool scorer configs (PD disaggregation) remain inline below.
		parsedScorerConfigs, bundle := resolvePolicies(cmd)

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
						"instance is priced for the scenario's hardware, so node pools must match it",
						np.Name, np.GPUType, kernelScenario, gpu)
				}
				// The kernel sizes every instance's KV from the catalog chip's memory; a pool
				// stating other memory would describe a device the run does not simulate.
				if chip := kernelOpened.Deployment().DeviceMemoryGiB; np.GPUMemoryGiB > 0 &&
					math.Abs(np.GPUMemoryGiB-chip) > 1e-9 {
					logrus.Fatalf("policy bundle node pool %q states gpu_memory_gib %g, but the catalog "+
						"chip %q has %g GiB, from which the kernel sizes every instance's KV; state %g or "+
						"omit it", np.Name, np.GPUMemoryGiB, gpu, chip, chip)
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
		if !sim.IsValidDisaggregationDecider(pdDecider) {
			logrus.Fatalf("Unknown PD decider %q. Valid: %s", pdDecider, strings.Join(sim.ValidDisaggregationDeciderNames(), ", "))
		}
		if err := cluster.ValidatePoolTopology(prefillInstances, decodeInstances, prefillDecodeInstances, encodeInstances, numInstances); err != nil {
			logrus.Fatalf("Invalid PD pool topology: %v", err)
		}
		if pdDecider == "prefix-threshold" && pdPrefixThreshold < 0 {
			logrus.Fatalf("--pd-prefix-threshold must be >= 0, got %d", pdPrefixThreshold)
		}
		if pdDecider != "prefix-threshold" && cmd.Flags().Changed("pd-prefix-threshold") {
			logrus.Warnf("--pd-prefix-threshold=%d is ignored when --pd-decider=%q (only applies to the prefix-threshold decider)", pdPrefixThreshold, pdDecider)
		}
		if pdDecider != "" && pdDecider != "never" && prefillInstances == 0 {
			logrus.Warnf("--pd-decider=%q has no effect because --prefill-instances=0 (disaggregation is disabled); set --prefill-instances and --decode-instances to enable", pdDecider)
		}

		// E/P/D disaggregation validation (GAP-4, issue #1264).
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

		// Each P/D role's engine is its own pool's, from its own kernel.
		kernelPD := openKernelPools(resolvedCatalogRoot)
		if kernelPD != nil {
			prefillOverrides, decodeOverrides = kernelPD.overrides()
		}

		// Parse per-pool scorer configs (PD disaggregation — not in resolvePolicies)
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
		// LoRA control-plane (#1464). loraCfg was resolved once at the top of RunE (for
		// the KV HBM reservation); here we cross-validate every workload adapter
		// reference against the declared registry (unknown id / base-model mismatch =>
		// Fatalf, never a silent no-op). With no adapters and no workload adapter
		// references this is inert (INV-6).
		var loraRegistry sim.AdapterRegistry
		if loraCfg.HasAdapters() {
			r, regErr := sim.NewAdapterRegistryFunc(loraCfg.Adapters)
			if regErr != nil {
				logrus.Fatalf("Invalid LoRA adapter registry: %v", regErr)
			}
			loraRegistry = r
		}
		if err := workload.ValidateAdapterReferences(spec, loraRegistry); err != nil {
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
		var pdTransferTime func(int64, cluster.InstanceID, cluster.InstanceID) int64
		if kernelPD != nil {
			pdTransferTime = kernelPD.transferTime()
		}
		config := cluster.DeploymentConfig{
			SimConfig: sim.SimConfig{
				Horizon: simulationHorizon,
				Seed:    seed,
				KVCacheConfig: sim.NewKVCacheConfig(totalKVBlocks, blockSizeTokens, kvCPUBlocks,
					kvOffloadThreshold, kvTransferBandwidth, kvTransferBaseLatency,
					sim.WithKVOffload(kvOffloadCfg), sim.WithKVTransferTicksPerBlock(kernelCPUTierTicks)),
				BatchConfig: batchConfigFromCLI(),
				// DP-as-placement (#1531): dpPlan.PerRankDP is the per-replica DP — 1 when
				// the plan is active (each replica is one rank), else the CLI dataParallelism
				// unchanged. Passing it through the canonical constructor (R4) keeps the config
				// authoritative from the start (no construct-then-override). Since #1556 replay
				// wires the SAME dpPlan.PerRankDP from the SAME resolveDPPlacement, so the two
				// paths agree for every config both support (INV-13).
				ModelHardwareConfig:  sim.NewModelHardwareConfig(lr.ModelConfig, model, gpu, tensorParallelism, dpPlan.PerRankDP, enableExpertParallel, maxModelLen),
				PolicyConfig:         sim.NewPolicyConfig(scheduler, preemptionPolicy),
				LoRAConfig:           loraCfg,
				SpeculativeConfig:    resolveSpeculativeConfig(cmd),
				SLOPriorityOverrides: sloPriorityOverrides,
				// The kernel prices every step.
				LatencyModel: lr.KernelModel,
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
			PDTransferContention:            pdTransferContention,
			PDTransferTime:                  pdTransferTime,
			PrefillScorerConfigs:            prefillScorerCfgs,
			DecodeScorerConfigs:             decodeScorerCfgs,
			PrefillOverrides:                prefillOverrides,
			DecodeOverrides:                 decodeOverrides,
			TierShedThreshold:               tierShedThreshold,
			TierShedMinPriority:             tierShedMinPriority,
			GAIEQDThreshold:                 gaieQDThreshold,
			GAIEKVThreshold:                 gaieKVThreshold,
			TenantBudgets:                   tenantBudgets,
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
			ModelAutoscalerIntervalUs:       bundleAutoscalerIntervalUs,
			ScaleUpStabilizationWindowUs:    bundleScaleUpStabilizationWindowUs,
			ScaleDownStabilizationWindowUs:  bundleScaleDownStabilizationWindowUs,
			HPAScrapeDelay:                  cluster.DelaySpec{Mean: bundleHPAScrapeDelayMean, Stddev: bundleHPAScrapeDelayStddev},
			AutoscalerAnalyzerConfig:        bundleAnalyzerCfg,
			NodePools:                       bundleNodePools,
			InstanceLifecycle:               bundleInstanceLifecycle,
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
		if err := aggregated.EmitOutput(clusterOutput, metricsPath,
			catalogProvenanceEmitOptions(metricsPath, resolvedCatalogRoot)...); err != nil {
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
		printPDMetrics(os.Stdout, rawMetrics.PD, config.PDTransferContention)

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
// No-op when pd is nil (disaggregation inactive). When contentionEnabled, also prints
// peak concurrent transfers and mean transfer queue depth.
func printPDMetrics(w io.Writer, pd *cluster.PDMetrics, contentionEnabled bool) {
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
	if contentionEnabled {
		_, _ = fmt.Fprintf(w, "Peak Concurrent Transfers: %d\n", pd.PeakConcurrentTransfers)
		_, _ = fmt.Fprintf(w, "Mean Transfer Queue Depth: %.4f\n", pd.MeanTransferQueueDepth)
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
