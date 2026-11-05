// Package trace defines the BLIS trace: a request stream written as a YAML header and a CSV
// of one row per request (SUB-19). `blis run --trace-output` and `blis replay` write one;
// `blis convert` writes one from another tool's capture; `blis replay` reads one. A
// scenario's workload binding refers to one by path (workload.TraceRef), with this header.
//
// The fields are transcribed from inference-sim's sim/workload/tracev2.go at be78af83,
// which defined the format before it moved here. Version 4 drops what only the removed
// `blis observe` command wrote or read: the `real` mode; the header's created_at,
// warm_up_requests, workload_spec, server, network and observed_kv_metrics; and the
// columns vllm_priority, x_request_id, server_input_tokens, send_time_us, num_chunks,
// error_message and finish_reason. It adds the `converted` mode, for a trace converted from
// another tool's capture, which records no BLIS deployment (inference-sim #2067).
package trace

import "sort"

// Version is the trace format this package reads and writes.
const Version = 4

// Header is the trace's YAML header: what produced the trace, in what units, under what
// seed, and the deployment inputs a replay rebuilds when the trace records a BLIS run.
type Header struct {
	// Version is the trace format version; it must equal Version.
	Version int `yaml:"trace_version"`
	// TimeUnit names the unit of the data file's *_us columns.
	TimeUnit TimeUnit `yaml:"time_unit"`
	// Mode is the pipeline that produced the trace.
	Mode Mode `yaml:"mode"`
	// WorkloadSeed is the workload generator's seed, when one was recorded. A seed of 0 is a
	// real seed, distinct from absence.
	WorkloadSeed *int64 `yaml:"workload_seed,omitempty"`
	// SessionContextGrowth says how a closed-loop replay assembles each session round's
	// input. Empty: each round's input is drawn independently. ContextAccumulate: each
	// round's input is the previous round's input and output followed by its new tokens.
	SessionContextGrowth ContextGrowth `yaml:"session_context_growth,omitempty"`
	// GoodputSLOTargets are the per-class latency targets the trace's goodput is measured
	// against, keyed by SLO class name. The key space is open: a class is whatever the
	// operator configured.
	GoodputSLOTargets map[string]SLODimTargets `yaml:"goodput_slo_targets,omitempty"`
	// KVOffload is the resolved KV-cache offload configuration of the run that wrote the
	// trace, so a replay reproduces it. Absent when the run used none.
	KVOffload *KVOffload `yaml:"kv_offload,omitempty"`
	// MaxNodesSpanned is the largest number of nodes one model instance occupied in the
	// run that wrote the trace, when it exceeded one. A replay refuses a trace whose value
	// it cannot reproduce. Zero when no instance crossed a node.
	MaxNodesSpanned int `yaml:"max_nodes_spanned,omitempty"`
	// Router records the llm-d router inputs the run routed with, so a replay rebuilds the
	// same router.
	Router *Router `yaml:"router,omitempty"`
}

// RecordsDeployment reports whether the trace records a BLIS deployment, whose header a
// replay must keep authoritative (inference-sim's INV-13). A converted trace records none,
// so a replay takes its deployment options from the scenario and the command line.
func (h *Header) RecordsDeployment() bool {
	return h.Mode == ModeGenerated || h.Mode == ModeReplayed
}

// Mode is the pipeline that produced a trace.
type Mode string

const (
	// ModeGenerated: `blis run` generated the requests and simulated them.
	ModeGenerated Mode = "generated"
	// ModeReplayed: `blis replay` simulated a trace and wrote what it ran.
	ModeReplayed Mode = "replayed"
	// ModeConverted: `blis convert` wrote it from another tool's capture. It records no
	// BLIS deployment.
	ModeConverted Mode = "converted"
)

var modes = map[Mode]bool{ModeGenerated: true, ModeReplayed: true, ModeConverted: true}

// Valid reports whether m is a recognised mode.
func (m Mode) Valid() bool { return modes[m] }

// AllModes returns every mode, sorted.
func AllModes() []string { return enumerate(modes) }

// TimeUnit is the unit of a trace's timestamp columns. The set carries the spellings the
// producing tools use for one unit; any other string is refused.
type TimeUnit string

const (
	TimeUnitNanoseconds      TimeUnit = "ns"
	TimeUnitNanosecondsLong  TimeUnit = "nanoseconds"
	TimeUnitMicroseconds     TimeUnit = "us"
	TimeUnitMicrosecondsLong TimeUnit = "microseconds"
	TimeUnitMilliseconds     TimeUnit = "ms"
	TimeUnitMillisecondsLong TimeUnit = "milliseconds"
	TimeUnitSeconds          TimeUnit = "s"
	TimeUnitSecondsLong      TimeUnit = "seconds"
)

var timeUnits = map[TimeUnit]bool{
	TimeUnitNanoseconds: true, TimeUnitNanosecondsLong: true,
	TimeUnitMicroseconds: true, TimeUnitMicrosecondsLong: true,
	TimeUnitMilliseconds: true, TimeUnitMillisecondsLong: true,
	TimeUnitSeconds: true, TimeUnitSecondsLong: true,
}

// Valid reports whether u is a recognised time unit.
func (u TimeUnit) Valid() bool { return timeUnits[u] }

// AllTimeUnits returns every recognised time unit, sorted.
func AllTimeUnits() []string { return enumerate(timeUnits) }

// ContextGrowth is how a session's rounds assemble their input.
type ContextGrowth string

// ContextAccumulate: each round's input is byte-identical to the previous round's input
// and output, followed by its new tokens.
const ContextAccumulate ContextGrowth = "accumulate"

// SLODimTargets is one SLO class's latency thresholds, in milliseconds. Zero on a
// dimension leaves it unconstrained.
type SLODimTargets struct {
	TTFTMs float64 `yaml:"ttft_ms,omitempty"`
	ITLMs  float64 `yaml:"itl_ms,omitempty"`
	E2EMs  float64 `yaml:"e2e_ms,omitempty"`
}

// KVOffload is a resolved KV-cache offload configuration, as the run that wrote the trace
// used it.
type KVOffload struct {
	CPUBytesToUse          int64  `yaml:"cpu_bytes_to_use"`
	PerBlockBytes          int64  `yaml:"per_block_bytes"`
	BlockSize              int64  `yaml:"block_size"`
	BlocksPerChunk         int64  `yaml:"blocks_per_chunk"`
	TokensPerHash          int64  `yaml:"tokens_per_hash"`
	EvictionPolicy         string `yaml:"eviction_policy"`
	OffloadPromptOnly      bool   `yaml:"offload_prompt_only"`
	SelfDescribingKVEvents bool   `yaml:"self_describing_kv_events"`
	// Tiers are the secondary tiers below the CPU tier.
	Tiers []KVOffloadTier `yaml:"secondary_tiers,omitempty"`
	// SimpleCPUOffload and LazyOffload record vLLM's SimpleCPUOffloadConnector and its
	// mode.
	SimpleCPUOffload bool `yaml:"simple_cpu_offload,omitempty"`
	LazyOffload      bool `yaml:"lazy_offload,omitempty"`
}

// KVOffloadTier is one resolved secondary offload tier. Bandwidths are bytes per
// microsecond and latencies microseconds, as resolved; DeviceClass is a label only.
type KVOffloadTier struct {
	Type           string  `yaml:"type"`
	RootDir        string  `yaml:"root_dir"`
	NReadThreads   int64   `yaml:"n_read_threads"`
	NWriteThreads  int64   `yaml:"n_write_threads"`
	Locality       string  `yaml:"locality,omitempty"`
	EnableKVEvents bool    `yaml:"enable_kv_events"`
	DirectIO       bool    `yaml:"direct_io"`
	DeviceClass    string  `yaml:"device_class,omitempty"`
	ReadBandwidth  float64 `yaml:"read_bandwidth"`
	WriteBandwidth float64 `yaml:"write_bandwidth"`
	BaseLatency    float64 `yaml:"base_latency"`
	// SaturationQueueDepth and SingleTransferFraction are the device model resolved for the
	// selected regime; 1 and 1.0 turn the ramp off.
	SaturationQueueDepth   int64   `yaml:"saturation_queue_depth"`
	SingleTransferFraction float64 `yaml:"single_transfer_fraction"`
	// CapacityBytes is the tier's resolved size; zero is unbounded.
	CapacityBytes       int64   `yaml:"capacity_bytes,omitempty"`
	LatencyJitterStddev float64 `yaml:"latency_jitter_stddev"`
}

// Router is the llm-d router's inputs as the run that wrote the trace gave them.
type Router struct {
	// EPPFlags are the router's command-line flags, with the EndpointPickerConfig inlined as
	// --config-text.
	EPPFlags []string `yaml:"epp_flags"`
	// Manifests are the Kubernetes manifests, as one YAML stream.
	Manifests string `yaml:"manifests,omitempty"`
	// PodLabels are the extra labels the engines' pods carry.
	PodLabels map[string]string `yaml:"pod_labels,omitempty"`
	// GatewayLatencyUs, ScrapeJitterUs and KVEventDelayUs model the environment around the
	// router.
	GatewayLatencyUs int64 `yaml:"gateway_latency_us,omitempty"`
	ScrapeJitterUs   int64 `yaml:"scrape_jitter_us,omitempty"`
	KVEventDelayUs   int64 `yaml:"kv_event_delay_us,omitempty"`
	// Seed is the router's seed.
	Seed *int64 `yaml:"seed,omitempty"`
}

// Record is one row of the data CSV: one request. The yaml tags are the column names. A
// timestamp is in the header's time unit, on the trace's own origin.
type Record struct {
	RequestID int    `yaml:"request_id"`
	ClientID  string `yaml:"client_id"`
	TenantID  string `yaml:"tenant_id"`
	SLOClass  string `yaml:"slo_class"`
	SessionID string `yaml:"session_id"`
	// RoundIndex is the round within the session: 0 for the first.
	RoundIndex   int    `yaml:"round_index"`
	PrefixGroup  string `yaml:"prefix_group"`
	PrefixLength int    `yaml:"prefix_length"`
	Streaming    bool   `yaml:"streaming"`
	// InputTokens is the prompt length after the shared prefix: the full prompt is
	// PrefixLength tokens of the group's prefix followed by InputTokens tokens. In an
	// accumulating session it is the round's new tokens.
	InputTokens  int     `yaml:"input_tokens"`
	OutputTokens int     `yaml:"output_tokens"`
	TextTokens   int     `yaml:"text_tokens"`
	ImageTokens  int     `yaml:"image_tokens"`
	AudioTokens  int     `yaml:"audio_tokens"`
	VideoTokens  int     `yaml:"video_tokens"`
	ReasonRatio  float64 `yaml:"reason_ratio"`
	// Model is the model tag; empty for the scenario's model.
	Model string `yaml:"model"`
	// DeadlineUs is the absolute deadline on the arrival origin; 0 is no deadline.
	DeadlineUs    int64 `yaml:"deadline_us"`
	ArrivalTimeUs int64 `yaml:"arrival_time_us"`
	// FirstChunkTimeUs and LastChunkTimeUs are the first and last token's delivery times in
	// the run that wrote the trace; 0 when it delivered none.
	FirstChunkTimeUs int64 `yaml:"first_chunk_time_us"`
	LastChunkTimeUs  int64 `yaml:"last_chunk_time_us"`
	// Status is how the request ended in the run or capture that wrote the trace.
	Status Status `yaml:"status"`

	// The optional columns. Each is written only when some record carries it.

	// SLOTargetUs is the request's TTFT target; 0 is none.
	SLOTargetUs int64 `yaml:"slo_target_us,omitempty"`
	// Adapter is the LoRA adapter serving the request; empty for the base model.
	Adapter string `yaml:"adapter,omitempty"`
	// ThinkTimeUs is the recorded client think time before this round, preferred over the
	// arrival gap in a closed-loop replay. Nil when not recorded; a recorded 0 is not nil.
	ThinkTimeUs *int64 `yaml:"think_time_us,omitempty"`
	// InputTokensReset marks a round whose recorded absolute input is smaller than the
	// previous round's input and output (a compacted context); an accumulating replay
	// restarts the session's buffer at this length. Nil on every other round.
	InputTokensReset *int64 `yaml:"input_tokens_reset,omitempty"`
}

// Status is how a request ended in the run or capture that wrote the trace.
type Status string

const (
	StatusOK      Status = "ok"
	StatusError   Status = "error"
	StatusTimeout Status = "timeout"
	// StatusIncomplete: the run stopped before the request ended.
	StatusIncomplete Status = "incomplete"
)

var statuses = map[Status]bool{StatusOK: true, StatusError: true, StatusTimeout: true, StatusIncomplete: true}

// Valid reports whether s is a recognised status.
func (s Status) Valid() bool { return statuses[s] }

// AllStatuses returns every status, sorted.
func AllStatuses() []string { return enumerate(statuses) }

func enumerate[T ~string](set map[T]bool) []string {
	out := make([]string, 0, len(set))
	for v := range set {
		out = append(out, string(v))
	}
	sort.Strings(out)
	return out
}
