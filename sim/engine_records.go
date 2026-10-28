// Moved from blis-vllm/compat/engine at 6b1d026: the host report's records of what an engine
// did, which the driver fills from engine.Engine's results (phase 2c). Sources: Completion from
// engine_seams.go, PrefixLookup from batch_formation.go, EngineMetrics from engine_metrics.go,
// KVEvent from kv_store.go, the server series' block from server_series.go, the kernel
// provenance records from provenance.go, KVPrefixHit from config.go, and the scheduler and
// preemption policy names from scheduler.go.

package sim

import (
	"maps"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/vocab"
)

// completion is one finished request's report values, computed by the engine.
type Completion struct {
	Req          *Request
	TTFT         int64 // FirstTokenTime
	OutputTokens int   // emitted output tokens (>= 0)
	E2E          int64 // end-to-end latency in ticks
	ITL          float64
	Steps        int   // FinishedStepIdx - ScheduledStepIdx
	Clock        int64 // the clock at which the completion was processed
	Overhead     int64 // the post-decode overhead e2e includes
}

// PrefixLookup is one admission's prefix-cache lookup in prompt tokens, as vLLM v0.31.0
// counts it: recorded when a new request is admitted, not when a lookup is attempted
// (vllm/v1/core/sched/scheduler.py:1252-1256 "Record at admission so unscheduled lookups are
// not counted"), with queries = the request's tokens and hits = its locally computed prefix
// (vllm/v1/core/kv_cache_manager.py:253-262). A previously preempted request is kept out of
// these counts, as PrefixCacheStats keeps it in separate preempted_* fields
// (vllm/v1/metrics/stats.py:132-143). CPUHitTokens is the prefix the offload tier supplies
// beyond the GPU hit, vLLM's external computed tokens.
type PrefixLookup struct {
	QueryTokens  int64
	HitTokens    int64
	CPUHitTokens int64
}

// EngineMetrics is the state a model server publishes on its Prometheus endpoint, as vLLM's
// PrometheusStatLogger records it from the SchedulerStats of one engine step
// (vllm/v1/core/sched/scheduler.py make_stats, vllm/v1/metrics/loggers.py record).
//
// The values describe the engine at the end of the step, and they become visible to a scraper
// only when that step ends: a scrape that lands while a step runs sees the previous step's
// values, exactly as a vLLM scrape does.
type EngineMetrics struct {
	// PublishedAt is the simulated time (µs) at which these values were recorded.
	PublishedAt int64
	// Waiting is len(scheduler.waiting), including preempted requests (vllm:num_requests_waiting).
	Waiting int
	// Running is len(scheduler.running) (vllm:num_requests_running).
	Running int
	// KVUsage is the fraction of KV blocks in use (vllm:kv_cache_usage_perc). The block pool
	// holds only usable blocks, so used/total is vLLM's BlockPool.get_usage().
	KVUsage float64
	// Preemptions is the cumulative preemption count (vllm:num_preemptions_total).
	Preemptions int64
	// RunningAdapters and WaitingAdapters are the LoRA adapters with at least one running or
	// waiting request (vllm:lora_requests_info), sorted.
	RunningAdapters []string
	WaitingAdapters []string
}

// KVEvent is a change to the set of block hashes an engine's prefix cache can serve, as vLLM
// publishes it (a stored block, or a removed one). A GPU store carries the block's parent hash (""
// for a first block) and its tokens.
type KVEvent struct {
	Removed bool
	Hash    string
	Parent  string
	Tokens  []TokenID
	// Tier is where the block is cached: "" for the GPU prefix cache, "cpu" for a CPU offload
	// tier. A CPU event moves or drops one location of a block the GPU events introduced; it
	// carries no parent or tokens, as vLLM's offloading events do not.
	Tier string
}

// ServerSeriesOutput is the file-only server_series block.
type ServerSeriesOutput struct {
	IntervalS float64              `json:"interval_s"`
	Origin    string               `json:"origin"`
	Engines   []ServerSeriesEngine `json:"engines"`
}

// ServerSeriesEngine is one engine's binned series. Bin k covers [k, k+1) x interval of
// simulated time from time 0.
type ServerSeriesEngine struct {
	ID   string `json:"id"`
	Bins int    `json:"bins"`

	KVCacheUsage           []float64 `json:"kv_cache_usage"`
	KVCacheUsageLast       []float64 `json:"kv_cache_usage_last"`
	NumRequestsRunning     []float64 `json:"num_requests_running"`
	NumRequestsRunningLast []float64 `json:"num_requests_running_last"`
	NumRequestsWaiting     []float64 `json:"num_requests_waiting"`
	NumRequestsWaitingLast []float64 `json:"num_requests_waiting_last"`

	PrefixCacheQueries             []int64 `json:"prefix_cache_queries"`
	PrefixCacheHits                []int64 `json:"prefix_cache_hits"`
	PromptTokens                   []int64 `json:"prompt_tokens"`
	PromptTokensLocalCompute       []int64 `json:"prompt_tokens_local_compute"`
	PromptTokensLocalCacheHit      []int64 `json:"prompt_tokens_local_cache_hit"`
	PromptTokensExternalKVTransfer []int64 `json:"prompt_tokens_external_kv_transfer"`
	GenerationTokens               []int64 `json:"generation_tokens"`
}

// The kernel provenance and routing records the latency kernel's adapter (sim/kernelmodel)
// produces and the host's report embeds unchanged (Metrics' --metrics-path output). They are
// the engine's output, so they are declared with the engine; package sim names them.

// KernelPoolProvenance is the evidence behind one pool's kernel (#1942): how many of the
// coefficients it priced with rest on each evidence method, the registry entries that rest on
// none (copied from another scope, or assumed), the assumptions the kernel made itself, and
// the release rules no one applied to the scenario. Entries are blis-latency-kernel's own
// kernel.CoefficientOrigin records, so their keys are that type's field names.
type KernelPoolProvenance struct {
	// MethodCounts counts the Provenance entries by evidence method; Evidenced and Total are
	// the kernel's Evidence() figures.
	MethodCounts map[vocab.Method]int `json:"method_counts"`
	Evidenced    int                  `json:"evidenced"`
	Total        int                  `json:"total"`
	// EvidenceSource says where Evidenced came from: the kernel's Evidence(), or, for a
	// kernel that does not report it, nothing (Evidenced is 0 and Total counts the entries).
	EvidenceSource string `json:"evidence_source"`
	// NotEvidenced are the registry entries whose method is neither evidenced nor
	// not_charged: copied from another scope (a fallback) or assumed.
	NotEvidenced []kernel.CoefficientOrigin `json:"fallback_or_assumed"`
	// Assumptions are the values the kernel supplied itself (Set == KernelAssumptionSet).
	Assumptions []kernel.CoefficientOrigin `json:"assumptions"`
	// SkippedRules are the scenario's release rules that neither the kernel nor BLIS
	// enforces, and SkippedRuleFindings what they report on this scenario.
	SkippedRules        []string      `json:"skipped_rules"`
	SkippedRuleFindings []RuleFinding `json:"skipped_rule_findings"`
	// Routing is the pool's resolved expert-parallel routing imbalance: which layer
	// supplied it, from which set and source, and why the default applied when it did.
	Routing *RoutingRecord `json:"routing,omitempty"`
}

// RoutingRecord is the provenance of a pool's routing-imbalance resolution
// (blis-schemas routing.Resolve), recorded the way acceptance's source layer is.
type RoutingRecord struct {
	SourceLayer string `json:"source_layer"`
	Set         string `json:"set,omitempty"`
	// Keys are the decode and prefill profiles a profile layer resolved to.
	Keys         []string `json:"keys,omitempty"`
	Source       string   `json:"source,omitempty"`
	Reason       string   `json:"reason,omitempty"`
	MeasuredOn   string   `json:"measured_on,omitempty"`
	FallbackFrom string   `json:"fallback_from,omitempty"`
	// TokenImbalance and ExpertImbalance are an inline value's ratios.
	TokenImbalance  float64 `json:"token_imbalance,omitempty"`
	ExpertImbalance float64 `json:"expert_imbalance,omitempty"`
}

// The values of KernelPoolProvenance.EvidenceSource.
const (
	EvidenceFromKernel     = "kernel"
	EvidenceFromProvenance = "provenance_entries"
)

// RuleFinding is one finding of a blis-schemas release rule.
type RuleFinding struct {
	Rule     string `json:"rule"`
	Severity string `json:"severity"`
	Path     string `json:"path,omitempty"`
	Message  string `json:"message"`
}

// KVPrefixHit is how vLLM v0.31.0's KV cache coordinator reconciles a prefix-cache hit
// across a model's KV cache groups (vllm/v1/core/kv_cache_coordinator.py
// find_longest_cache_hit), beyond the full-attention prefix BLIS hashes. Every length is in
// tokens and the full-attention block is the store's block, which must be the engine's
// scheduler block (the LCM of the group block sizes) for the hit to land where vLLM's does.
type KVPrefixHit struct {
	// FullEagle drops the last block of every full-attention hit: the full-attention group is
	// an EAGLE group (FullAttentionManager.find_longest_cache_hit,
	// single_type_kv_cache_manager.py:834-836).
	FullEagle bool `json:"full_eagle,omitempty"`
	// Window is a prefix-cacheable sliding-window group (SlidingWindowManager), 0 when the
	// model has none or its window caches do not take part in prefix caching. WindowBlock is
	// its page in tokens.
	Window      int64 `json:"window,omitempty"`
	WindowBlock int64 `json:"window_block,omitempty"`
	// WindowEagle: the window group is an EAGLE group, which peeks one window block past a
	// boundary and drops it (single_type_kv_cache_manager.py:960-971, 1056-1067).
	WindowEagle bool `json:"window_eagle,omitempty"`
	// EagleBoundaries: some group is an EAGLE group, so the replay boundaries a request
	// retains window tails at sit one scheduler block below its prompt
	// (KVCacheCoordinator.get_replay_boundaries, kv_cache_coordinator.py:316-338).
	EagleBoundaries bool `json:"eagle_boundaries,omitempty"`
	// ReplayTokens is DeepSeek-V4.1's SWA bounded replay window (prefix_replay_tokens,
	// kv_cache_interface.py:953-955): a hit no longer than it is dropped, and the last
	// min(ReplayTokens, hit) tokens of a hit are computed again (sched/scheduler.py:1016-1026,
	// 1059-1079, 3038-3050). 0: off.
	ReplayTokens int64 `json:"replay_tokens,omitempty"`
}

// The scheduler and preemption policy names an engine accepts (--scheduler,
// --preemption-policy, a policy bundle's), as the compatibility engine's scheduler names them.
var (
	validSchedulers         = map[string]bool{"": true, "fcfs": true, "priority-fcfs": true, "sjf": true, "reverse-priority": true}
	validPreemptionPolicies = map[string]bool{"": true, "fcfs": true, "priority": true}
)

// SchedulerNameTable and PreemptionPolicyNameTable return copies of the name tables, so a caller
// that edits one changes nothing here.
func SchedulerNameTable() map[string]bool        { return maps.Clone(validSchedulers) }
func PreemptionPolicyNameTable() map[string]bool { return maps.Clone(validPreemptionPolicies) }

// IsValidScheduler returns true if name is a recognized scheduler.
func IsValidScheduler(name string) bool { return validSchedulers[name] }
