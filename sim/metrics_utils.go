// sim/metrics_utils.go
package sim

import (
	"math"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/vocab"
)

type IntOrFloat64 interface {
	int | int64 | float64
}

// Individual request metrics for the JSON log
type RequestMetrics struct {
	ArrivedAt         float64 `json:"arrived_at"`
	ID                string  `json:"requestID"`
	NumPrefillTokens  int     `json:"num_prefill_tokens"`
	NumDecodeTokens   int     `json:"num_decode_tokens"`
	TTFT              float64 `json:"ttft_ms"`
	ITL               float64 `json:"itl_ms"`
	E2E               float64 `json:"e2e_ms"`
	SchedulingDelay   float64 `json:"scheduling_delay_ms"`
	SLOClass          string  `json:"slo_class,omitempty"`              // PR10: for per-SLO-class metrics
	TenantID          string  `json:"tenant_id,omitempty"`              // PR10: for per-tenant fairness
	HandledBy         string  `json:"handled_by,omitempty"`             // #181: instance that processed this request
	Model             string  `json:"model,omitempty"`                  // W0-1: model tag for per-model metrics
	Adapter           string  `json:"adapter,omitempty"`                // #1464: LoRA adapter id serving this request ("" = base model)
	LengthCapped      bool    `json:"length_capped,omitempty"`          // #588: per-request indicator for BC-5 force-completion
	GatewayQueueDelay float64 `json:"gateway_queue_delay_ms,omitempty"` // #882: time spent in gateway queue (ms)
	SessionID         string  `json:"session_id,omitempty"`             // #1058: session context for multi-turn metrics
	RoundIndex        int     `json:"round_index"`                      // #1058: 0 for first round, N for Nth follow-up
	Phase             Phase   `json:"phase,omitempty"`                  // #1938: benchmark phase; "" = measured
	// Agentic is the request's agentic-replay tag (#1937), shared with the Request, so its
	// Outcome reads as the request ended. Absent for every other workload.
	Agentic *AgenticTag `json:"agentic,omitempty"`
	// OutputTokens is the count the request emitted (aiperf's OSL), set at completion. It
	// differs from NumDecodeTokens, the oracle count, for a length-capped request.
	OutputTokens int `json:"output_tokens,omitempty"`

	// CompletionIndex is this request's 1-based position in the run's completion order,
	// or 0 for a request that never completed (#1902). Assigned by EmitOutput, so it exists
	// only in the --metrics-path file (Requests[] never reaches stdout). A consumer that
	// discards a leading warm-up phase by completion -- the accuracy harness does -- needs
	// the order the simulator completed requests in, which neither arrival order nor
	// ArrivedAt+E2E recovers.
	CompletionIndex int `json:"completion_index,omitempty"`

	// completionSeq is the order in which the serving instance completed this request
	// (1-based, per instance; 0 = not completed) and completionClock the simulation clock
	// when it did, both recorded by Metrics.recordCompletionOrder. They survive the
	// cluster's value-copy merge of Requests, and EmitOutput derives CompletionIndex from
	// them. Unexported, so never serialized.
	completionSeq   int
	completionClock int64
	// completionOverheadTicks is the fixed host time the latency model charged after the
	// request's last streamed token (PostDecodeFixedOverhead), part of E2E but not of
	// aiperf's request latency, which ends at the last content chunk (#1938). timedOut marks
	// a request cancelled by its client timeout. Both survive the merge and the P/D
	// projection (WithCompletionOf) like the completion stamp.
	completionOverheadTicks int64
	timedOut                bool
}

// NewRequestMetrics creates a RequestMetrics from a Request and its arrival time.
// This is the canonical constructor — all production code MUST use this instead of
// inline RequestMetrics{} literals. Test code may still use literals for partial construction.
func NewRequestMetrics(req *Request, arrivedAt float64) RequestMetrics {
	rm := RequestMetrics{
		ID:               req.ID,
		ArrivedAt:        arrivedAt,
		NumPrefillTokens: int(req.InputLen()),
		NumDecodeTokens:  len(req.OutputTokens),
		SLOClass:         req.SLOClass,
		TenantID:         req.TenantID,
		HandledBy:        req.AssignedInstance,
		Model:            req.Model,
		Adapter:          req.Adapter,
		LengthCapped:     req.LengthCapped,
		SessionID:        req.SessionID,
		RoundIndex:       req.RoundIndex,
		Phase:            req.Phase,
		Agentic:          req.Agentic,
	}
	// Flow control: compute gateway queue delay when timestamps are set (#882)
	if req.GatewayDispatchTime > 0 && req.GatewayEnqueueTime > 0 {
		rm.GatewayQueueDelay = float64(req.GatewayDispatchTime-req.GatewayEnqueueTime) / 1000.0
	}
	return rm
}

// MetricsOutput defines the JSON structure for the saved metrics
type MetricsOutput struct {
	InstanceID           string  `json:"instance_id"`
	CompletedRequests    int     `json:"completed_requests"`
	StillQueued          int     `json:"still_queued"`
	StillRunning         int     `json:"still_running"`
	InjectedRequests     int     `json:"injected_requests"`
	TotalInputTokens     int     `json:"total_input_tokens"`
	TotalOutputTokens    int     `json:"total_output_tokens"`
	VllmDurationSec      float64 `json:"vllm_estimated_duration_s"`
	ResponsesPerSec      float64 `json:"responses_per_sec"`
	TokensPerSec         float64 `json:"tokens_per_sec"`
	E2EMeanMs            float64 `json:"e2e_mean_ms"`
	E2EP90Ms             float64 `json:"e2e_p90_ms"`
	E2EP95Ms             float64 `json:"e2e_p95_ms"`
	E2EP99Ms             float64 `json:"e2e_p99_ms"`
	TTFTMeanMs           float64 `json:"ttft_mean_ms"`
	TTFTP90Ms            float64 `json:"ttft_p90_ms"`
	TTFTP95Ms            float64 `json:"ttft_p95_ms"`
	TTFTP99Ms            float64 `json:"ttft_p99_ms"`
	ITLMeanMs            float64 `json:"itl_mean_ms"`
	ITLP90Ms             float64 `json:"itl_p90_ms"`
	ITLP95Ms             float64 `json:"itl_p95_ms"`
	ITLP99Ms             float64 `json:"itl_p99_ms"`
	SchedulingDelayP99Ms float64 `json:"scheduling_delay_p99_ms"`
	KVAllocationFailures int64   `json:"kv_allocation_failures,omitempty"`
	PreemptionCount      int64   `json:"preemption_count"`
	// CacheHitRate is the aggregate KV-cache hit rate (a fraction in [0,1]) at
	// finalization (#1583). It is populated ONLY into the --metrics-path file (the
	// file branch of EmitOutput), alongside the file-only Requests[] — never onto
	// stdout, so a run's stdout stays byte-identical to a pre-feature build (INV-6,
	// BC-9). A nil pointer (default) is dropped by omitempty; the human-readable rate
	// is still printed to stdout via printKVCacheMetrics. `blis calibrate --sim-metrics`
	// reads it as the simulator's hit-rate for the real-vs-sim comparison.
	CacheHitRate         *float64         `json:"cache_hit_rate,omitempty"`
	DroppedUnservable    int              `json:"dropped_unservable"`
	LengthCappedRequests int              `json:"length_capped_requests"`
	TimedOutRequests     int              `json:"timed_out_requests"`
	Requests             []RequestMetrics `json:"requests,omitempty"`
	Saturation           interface{}      `json:"saturation,omitempty"` // saturation.Result, using interface{} to avoid import cycle
	// Goodput fields (issue #1409). Populated by cmd/-side goodput wiring when
	// --slo-ttft/itl/e2e flags or workload-spec/trace-header goodput_slo_targets
	// are configured; otherwise left zero and suppressed by omitempty.
	// PerClass holds map[string]map[string]any (per-class goodput summary);
	// typed as interface{} to mirror the Saturation field above and avoid a
	// sim → sim/cluster import cycle.
	GoodputRPS    float64     `json:"goodput_rps,omitempty"`
	SLOAttainment float64     `json:"slo_attainment,omitempty"`
	PerClass      interface{} `json:"per_class,omitempty"`

	// Adapters holds per-LoRA-adapter aggregate metrics, keyed by adapter id.
	// omitempty: absent when no request is attributed to an adapter, so an
	// adapter-blind run adds no stdout fields (INV-6, SC-001). encoding/json emits
	// map string keys in sorted order, giving deterministic output (R2).
	Adapters map[string]AdapterMetrics `json:"adapters,omitempty"`

	// Catalog records WHICH model catalog produced this result (#1732, R1/S5).
	// Populated ONLY into the --metrics-path file, via EmitOutput's
	// WithCatalogProvenance option — never onto stdout, exactly like CacheHitRate
	// above (#1583). Catalog git state legitimately varies between two otherwise
	// identical runs, so it must not enter the byte-identical stdout channel INV-6
	// protects. A nil pointer (the default, and every non-file emit) is dropped by
	// omitempty, so a results file written without the option is byte-identical to
	// a pre-feature build.
	Catalog *CatalogProvenance `json:"catalog,omitempty"`

	// Overrides lists every setting this run simulated differently from its scenario -- an
	// explicit flag or policy bundle chose another value than the scenario's deployment
	// states -- so a results file always says when it does not reproduce its scenario.
	// File-only, like Catalog (WithScenarioOverrides); omitted when the run follows its
	// scenario exactly.
	// EPP is the ingested llm-d EndpointPickerConfig (--epp-config, #1930); file-only
	// via WithEPPProvenance, absent otherwise.
	EPP *EPPProvenance `json:"epp,omitempty"`

	Overrides []ScenarioOverride `json:"overrides,omitempty"`

	// Agentic is an agentic replay's own counts (#1937; workload.AgentXSummary, held as
	// any to avoid an import cycle). File-only (WithAgentic); absent for other workloads.
	Agentic any `json:"agentic,omitempty"`

	// PD is the P/D block (PDOutput), file-only; absent for a colocated run.
	PD *PDOutput `json:"pd,omitempty"`

	// MeasuredMetrics are the aiperf-defined measured-window aggregates (#1938), flattened
	// into the top level. File-only, like CacheHitRate: EmitOutput sets it in the
	// --metrics-path branch, and the nil pointer drops every key from stdout (INV-6).
	*MeasuredMetrics

	// WholeRun keeps the whole-run values of the keys the file restates over the measured
	// window (ttft_*, e2e_*, cache_hit_rate). File-only.
	WholeRun *WholeRunLatency `json:"whole_run,omitempty"`

	// KernelProvenance is blis-latency-kernel's per-pool evidence, keyed by pool role
	// (#1942), {} when no kernel priced the run. File-only (WithKernelProvenance); a pointer
	// so stdout omits it while the file can state an empty map.
	KernelProvenance *map[string]KernelPoolProvenance `json:"kernel_provenance,omitempty"`

	// BuildVersions pins the build that produced the file. File-only (WithBuildVersions).
	*BuildVersions
}

// BuildVersions are the module versions a results file was produced by: BLIS itself and the
// blis-latency-kernel and blis-schemas releases it was built against.
type BuildVersions struct {
	BLIS    string `json:"blis_version"`
	Kernel  string `json:"kernel_version"`
	Schemas string `json:"schemas_version"`
}

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

// PDOutput is the results file's P/D block (#1922, #1934, #1940): the KV handoffs, how long
// contended links held them up, the prefill KV held for decode pulls, and each pool's CPU-tier
// hit rate. File-only, like Catalog. Times in milliseconds; percentiles numpy-linear.
type PDOutput struct {
	Transfers                int                `json:"transfers"`
	TransferTimeP50Ms        float64            `json:"transfer_time_p50_ms"`
	TransferTimeP90Ms        float64            `json:"transfer_time_p90_ms"`
	TransferQueueDelayMeanMs float64            `json:"transfer_queue_delay_mean_ms"`
	TransferQueueDelayP90Ms  float64            `json:"transfer_queue_delay_p90_ms"`
	PrefillKVHoldPeakTokens  int64              `json:"prefill_kv_hold_peak_tokens"`
	LinkPeakConcurrent       map[string]int     `json:"link_peak_concurrent_transfers,omitempty"`
	CPUTierHitRate           map[string]float64 `json:"cpu_tier_hit_rate"`
}

// ScenarioOverride is one setting a run took from somewhere other than its scenario.
type ScenarioOverride struct {
	Setting  string `json:"setting"`  // what was overridden, e.g. "scheduler"
	Scenario string `json:"scenario"` // the value the scenario states (in BLIS's terms)
	Run      string `json:"run"`      // the value the run simulated
	Source   string `json:"source"`   // what chose it, e.g. "--scheduler" or "--policy-config"
}

// CatalogProvenance attributes a results file to the model catalog that produced it
// (#1732, R1/S5). Once the catalog is a versioned directory (#1731 + blis-catalog), a
// result must be attributable, and an EXPERIMENT (uncommitted edits to a config.json)
// must be distinguishable from a REPRODUCIBLE run — which is what Dirty answers.
//
// Revision is the catalog's git commit, or UnknownCatalogRevision when the catalog is
// not a git checkout (or git could not be consulted): a non-git catalog degrades to
// "path recorded, revision unknown", never to a plausible-but-wrong revision. Dirty is
// meaningful only alongside a known revision — it is false whenever Revision is
// UnknownCatalogRevision, since there is no committed state to differ from.
type CatalogProvenance struct {
	Path     string `json:"path"`
	Revision string `json:"revision"`
	Dirty    bool   `json:"dirty"`
}

// UnknownCatalogRevision is the CatalogProvenance.Revision value recorded when the
// catalog's git revision could not be determined — an absent directory, a directory
// outside any git checkout, a repository with no commits, or no usable git binary. It
// is a recorded sentinel rather than an empty string so a reader can distinguish
// "asked, and the answer is unknown" from "never populated".
const UnknownCatalogRevision = "unknown"

// AdapterMetrics is the per-adapter aggregate section
// (specs/007-lora-control-plane/contracts/metrics.md). TTFT
// percentiles are in microseconds (ticks); throughput is completed output tokens per
// second. LoadCount/EvictionCount are cumulative resident-set event counts populated
// by the per-instance resident set: LoadCount per cold load, EvictionCount per
// eviction. They are 0 for an adapter whose requests all hit a warm slot (touch
// only — no cold loads or evictions), and absent entirely on an adapter-blind run
// (INV-6).
type AdapterMetrics struct {
	LoadCount         int64   `json:"load_count"`
	EvictionCount     int64   `json:"eviction_count"`
	TTFTP50Us         float64 `json:"ttft_p50_us"`
	TTFTP99Us         float64 `json:"ttft_p99_us"`
	ThroughputTokPerS float64 `json:"throughput_tok_per_s"`
}

// CalculatePercentile is a util function that calculates the p-th percentile of a data list
// return values are in milliseconds
func CalculatePercentile[T IntOrFloat64](data []T, p float64) float64 {
	n := len(data)
	if n == 0 {
		return 0
	}

	rank := p / 100.0 * float64(n-1)
	lowerIdx := int(math.Floor(rank))
	upperIdx := int(math.Ceil(rank))

	if lowerIdx == upperIdx {
		return float64(data[lowerIdx]) / 1000
	} else {
		lowerVal := data[lowerIdx]
		upperVal := data[upperIdx]
		if upperIdx >= n {
			return float64(data[n-1]) / 1000
		}
		return float64(lowerVal)/1000 + float64(upperVal-lowerVal)*(rank-float64(lowerIdx))/1000
	}
}

// CalculateMean is a util function that calculates the mean of a data list
// return values are in milliseconds
func CalculateMean[T IntOrFloat64](numbers []T) float64 {
	if len(numbers) == 0 {
		return 0.0
	}

	sum := 0.0
	for _, number := range numbers {
		sum += float64(number)
	}

	return (sum / float64(len(numbers))) / 1000
}
