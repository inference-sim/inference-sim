// sim/metrics_utils.go
package sim

import (
	"math"
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

	// CachedLocalTokens and CachedExternalTokens are the prompt tokens the device prefix cache and
	// the KV offload tier supplied at the request's first admission (its Result row's
	// cached_local_tokens and cached_external_tokens). Not part of the legacy file.
	CachedLocalTokens    int `json:"-"`
	CachedExternalTokens int `json:"-"`

	// CompletionIndex is this request's 1-based position in the run's completion order,
	// or 0 for a request that never completed (#1902). Assigned by RequestRows. A consumer that
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
	InstanceID           string           `json:"instance_id"`
	CompletedRequests    int              `json:"completed_requests"`
	StillQueued          int              `json:"still_queued"`
	StillRunning         int              `json:"still_running"`
	InjectedRequests     int              `json:"injected_requests"`
	TotalInputTokens     int              `json:"total_input_tokens"`
	TotalOutputTokens    int              `json:"total_output_tokens"`
	VllmDurationSec      float64          `json:"vllm_estimated_duration_s"`
	ResponsesPerSec      float64          `json:"responses_per_sec"`
	TokensPerSec         float64          `json:"tokens_per_sec"`
	E2EMeanMs            float64          `json:"e2e_mean_ms"`
	E2EP90Ms             float64          `json:"e2e_p90_ms"`
	E2EP95Ms             float64          `json:"e2e_p95_ms"`
	E2EP99Ms             float64          `json:"e2e_p99_ms"`
	TTFTMeanMs           float64          `json:"ttft_mean_ms"`
	TTFTP90Ms            float64          `json:"ttft_p90_ms"`
	TTFTP95Ms            float64          `json:"ttft_p95_ms"`
	TTFTP99Ms            float64          `json:"ttft_p99_ms"`
	ITLMeanMs            float64          `json:"itl_mean_ms"`
	ITLP90Ms             float64          `json:"itl_p90_ms"`
	ITLP95Ms             float64          `json:"itl_p95_ms"`
	ITLP99Ms             float64          `json:"itl_p99_ms"`
	SchedulingDelayP99Ms float64          `json:"scheduling_delay_p99_ms"`
	KVAllocationFailures int64            `json:"kv_allocation_failures,omitempty"`
	PreemptionCount      int64            `json:"preemption_count"`
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
}

// PrefixCacheRecord is one pool's prefix-hit resolution.
type PrefixCacheRecord struct {
	ModelRunner          string      `json:"model_runner"`
	ModelRunnerForcedBy  []string    `json:"model_runner_v1_forced_by,omitempty"`
	SWABoundedReplay     *bool       `json:"swa_bounded_replay,omitempty"`
	SWABoundedReplayFrom string      `json:"swa_bounded_replay_from,omitempty"`
	Layout               KVPrefixHit `json:"layout"`
	Note                 string      `json:"note,omitempty"`
}

// BuildVersions are the module versions a results file was produced by: BLIS itself and the
// blis-latency-kernel and blis-schemas releases it was built against.
type BuildVersions struct {
	BLIS    string `json:"blis_version"`
	Kernel  string `json:"kernel_version"`
	Schemas string `json:"schemas_version"`
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

// AcceptanceRecord is a speculating run's acceptance as simulated: the mean acceptance length
// AL each verification step advanced a request by (1 + accepted drafts, so in [1, K+1]), its
// rate (AL-1)/K, the precedence layer that supplied it (registry, workload, scenario or cli),
// and the evidence behind it.
type AcceptanceRecord struct {
	Length      float64         `json:"length"`
	Rate        float64         `json:"rate"`
	SourceLayer string          `json:"source_layer"`
	Entry       AcceptanceEntry `json:"entry"`
}

// AcceptanceEntry is the provenance of a resolved acceptance. Set, Key and the profile fields
// are filled when a profile layer resolved it; Source always cites the evidence when one was
// stated.
type AcceptanceEntry struct {
	// Set and Key name the profile set and the (model/method/k/thinking/category) profile.
	Set string `json:"set,omitempty"`
	Key string `json:"key,omitempty"`
	// Method is how the profile's value was obtained (measured, copied, ...), and CopiedFrom
	// the profile a copied value came from.
	Method     string `json:"method,omitempty"`
	CopiedFrom string `json:"copied_from,omitempty"`
	// Source is the document cited: the profile's provenance source or the value's source.
	Source string `json:"source,omitempty"`
	// Run is the job that measured a profile.
	Run string `json:"run,omitempty"`
	// FallbackFrom is the workload category that had no profile when a requested fallback
	// category replaced it.
	FallbackFrom string `json:"fallback_from,omitempty"`
	// MeasuredOn is the hardware a profile was measured on when it differs from the run's.
	MeasuredOn string `json:"measured_on,omitempty"`
	// PerPositionRates are the per-position acceptance rates p_1..p_K, and RatesDerived
	// whether they were derived from AL by vLLM's minimum-variance schedule rather than stated.
	PerPositionRates []float64 `json:"per_position_rates"`
	RatesDerived     bool      `json:"rates_derived"`
}

// ScenarioOverride is one setting a run took from somewhere other than its scenario.
type ScenarioOverride struct {
	Setting  string `json:"setting"`  // what was overridden, e.g. "scheduler"
	Scenario string `json:"scenario"` // the value the scenario states (in BLIS's terms)
	Run      string `json:"run"`      // the value the run simulated
	Source   string `json:"source"`   // what chose it, e.g. "--scheduler" or "--policy-config"
}

// CatalogProvenance attributes a --metrics-path file to the catalog that produced it (#1732).
// Path is the catalog directory a path named, or "embedded" for the copy compiled into the
// pinned blis-substrate release; Revision is the catalog's combined content hash (design §3.3);
// Dirty is true when an overlay changed the catalog.
type CatalogProvenance struct {
	Path     string `json:"path"`
	Revision string `json:"revision"`
	Dirty    bool   `json:"dirty"`
}

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

	// float64(...) rounds the product before rank is used, so no platform fuses it into the
	// subtraction below: arm64 fused p/100*(n-1) - lowerIdx into one FNMSUBD and amd64 did not,
	// which moved the last bit of a percentile between platforms (design X6).
	rank := float64(p / 100.0 * float64(n-1))
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
