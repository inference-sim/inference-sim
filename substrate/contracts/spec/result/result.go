// Package result describes what one BLIS run predicts: the document `blis run` and
// `blis replay` emit, one per run of one configuration at one load with one seed.
//
// A Result states its metrics in the vocabulary a measured run uses (package
// measurement): each metric is a name defined by a measurement.MetricDef, and the load is
// a measurement.Load. A prediction and a measurement of the same configuration therefore
// compare by metric name, with the two definitions checked equal, and with no table that
// translates one tool's keys into another's.
//
// A Result embeds the input it was produced from, resolved, and the revision of every
// component and data repository it read. Given that input at the same revisions, the
// simulator reproduces the document byte for byte outside Runtime, RerunOf and
// RevisionChanges. Runtime is the only part of the document that depends on the host or
// the wall clock.
//
// Bulk data (per-request rows, an exported trace, engine time series) is referenced by
// path and sha256, never inlined.
//
// The simulator writes a Result as JSON by default and as YAML on request, with Marshal;
// Unmarshal reads either. The yaml tags define the keys of both encodings, so a Result is
// written with Marshal, not with encoding/json, which would read no tag here.
package result

import (
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

// Kind is the value of Result.Kind.
const Kind = "Result"

// SchemaVersion is the Result schema this package reads and writes.
const SchemaVersion = 1

// Result is one simulator run's predicted output.
type Result struct {
	Kind          string `yaml:"kind"`
	SchemaVersion int    `yaml:"schema_version"`
	// Name identifies the run, a lowercase slug, for example "qwen3-32b-h100-pd.c64-s1".
	Name string `yaml:"name"`

	// Input is everything the run was given, after resolution.
	Input Input `yaml:"input"`
	// Revisions are the versions of every component and data repository the run read.
	Revisions Revisions `yaml:"revisions"`

	// RerunOf names the Result whose input this run was produced from, when it was.
	RerunOf *Reference `yaml:"rerun_of,omitempty"`
	// RevisionChanges lists each component whose revision differs from RerunOf's. Empty
	// unless RerunOf is set.
	RevisionChanges []RevisionChange `yaml:"revision_changes,omitempty"`

	// MetricDefs defines every key of Metrics, in the measured-run vocabulary.
	MetricDefs map[string]measurement.MetricDef `yaml:"metric_defs"`
	// Window is the measured window every metric is computed over.
	Window Window `yaml:"window"`
	// Metrics are the predicted values, keyed by metric name.
	Metrics map[string]float64 `yaml:"metrics"`

	// Requests is the request-conservation ledger (INV-1) over the whole run.
	Requests Requests `yaml:"requests"`

	// Goodput is present when the input states SLO targets.
	Goodput *Goodput `yaml:"goodput,omitempty"`
	// Breakdowns split the run along the axes it exercised.
	Breakdowns *Breakdowns `yaml:"breakdowns,omitempty"`

	// Engine, Router and Kernel are each reported by that component, keyed by pool role
	// where a component is per pool. The document carries them as reported.
	Engine map[string]any `yaml:"engine,omitempty"`
	Router map[string]any `yaml:"router,omitempty"`
	Kernel map[string]any `yaml:"kernel,omitempty"`
	// Saturation is the post-hoc saturation report, when detectors ran.
	Saturation map[string]any `yaml:"saturation,omitempty"`

	// Overrides lists every setting the run simulated differently from Input, and what
	// chose the value.
	Overrides []Override `yaml:"overrides,omitempty"`
	// Artifacts are bulk outputs, by reference.
	Artifacts map[string]Artifact `yaml:"artifacts,omitempty"`

	// Runtime is the only part of the document that depends on the host or the wall
	// clock. Determinism and round-trip comparisons skip it.
	Runtime *Runtime `yaml:"runtime,omitempty"`
}

// Input is the resolved input of a run.
type Input struct {
	Scenario   scenario.Scenario     `yaml:"scenario"`
	Deployment deployment.Deployment `yaml:"deployment"`
	// Policy is the simulator's policy bundle (autoscaler, node pools, instance
	// lifecycle, priority, preemption), verbatim, when one was given.
	Policy map[string]any `yaml:"policy,omitempty"`
	Run    Run            `yaml:"run"`
}

// Run holds the run-level controls: the inputs a scenario and a deployment do not state.
type Run struct {
	// Load is the offered load: exactly one of a concurrency and a request rate.
	Load measurement.Load `yaml:"load"`
	// NumRequests is the number of requests the run generated.
	NumRequests int   `yaml:"num_requests"`
	Seed        int64 `yaml:"seed"`
	// HorizonS bounds simulated time, in seconds, when set.
	HorizonS *float64 `yaml:"horizon_s,omitempty"`
	// Warmup is excluded from the measured window.
	Warmup *Warmup `yaml:"warmup,omitempty"`
	// Workload is the workload the run used: the scenario's binding unless one replaced it.
	Workload *workload.Binding `yaml:"workload,omitempty"`
}

// Warmup is the leading part of a run excluded from the measured window: a number of
// requests or a duration, not both.
type Warmup struct {
	Requests  int     `yaml:"requests,omitempty"`
	DurationS float64 `yaml:"duration_s,omitempty"`
}

// Revisions are the versions a run read, each a release tag, a commit, or both.
type Revisions struct {
	BLIS              string    `yaml:"blis"`
	BLISSchemas       string    `yaml:"blis_schemas"`
	BLISLatencyKernel string    `yaml:"blis_latency_kernel,omitempty"`
	LLMDRouter        string    `yaml:"llm_d_router,omitempty"`
	Engine            *Engine   `yaml:"engine,omitempty"`
	Catalog           *DataRepo `yaml:"catalog,omitempty"`
	Registry          *DataRepo `yaml:"registry,omitempty"`
}

// Engine is the inference-engine model a run used.
type Engine struct {
	// Kind names the engine model, for example "vllm".
	Kind string `yaml:"kind"`
	// UpstreamVersion is the release of the real engine it models, for example "0.32.0".
	UpstreamVersion string `yaml:"upstream_version"`
	// Release is the engine model's own release.
	Release string `yaml:"release,omitempty"`
}

// DataRepo is a data repository read at a release, a commit, or both.
type DataRepo struct {
	Release  string `yaml:"release,omitempty"`
	Revision string `yaml:"revision,omitempty"`
}

// Reference names another Result document.
type Reference struct {
	Name   string `yaml:"name"`
	SHA256 string `yaml:"sha256"`
}

// RevisionChange is one component whose revision differs from the Result a run reran.
type RevisionChange struct {
	// Component is a key of Revisions, for example "blis" or "catalog".
	Component string `yaml:"component"`
	From      string `yaml:"from"`
	To        string `yaml:"to"`
}

// Window is the measured window the metrics are over.
type Window struct {
	// Start and Stop say what bounds the window, for example "first_send_after_warmup"
	// and "last_completion".
	Start     string  `yaml:"start"`
	Stop      string  `yaml:"stop"`
	DurationS float64 `yaml:"duration_s"`
	// Requests is the number of requests that completed inside the window.
	Requests int `yaml:"requests"`
}

// Requests is the request-conservation ledger. Every injected request ends in exactly
// one terminal bucket; each bucket is named as in the simulator's INV-1.
type Requests struct {
	// Generated is every request the workload produced: Injected plus Rejected.
	Generated int `yaml:"generated"`
	// Rejected requests were refused at admission and never injected.
	Rejected int `yaml:"rejected"`
	Injected int `yaml:"injected"`

	Completed         int `yaml:"completed"`
	StillQueued       int `yaml:"still_queued"`
	StillRunning      int `yaml:"still_running"`
	DroppedUnservable int `yaml:"dropped_unservable"`
	TimedOut          int `yaml:"timed_out"`

	// The router's outcomes; zero for a single instance.
	RoutingRejections       int `yaml:"routing_rejections"`
	GatewayQueueDepth       int `yaml:"gateway_queue_depth"`
	GatewayQueueRejected    int `yaml:"gateway_queue_rejected"`
	GatewayEvicted          int `yaml:"gateway_evicted"`
	GatewayExpired          int `yaml:"gateway_expired"`
	EncodeRoutingRejections int `yaml:"encode_routing_rejections"`

	// LengthCapped counts completed requests whose output was cut at the length limit.
	// It is not a terminal bucket.
	LengthCapped int `yaml:"length_capped"`
}

// Terminal returns the sum of the terminal buckets, which INV-1 requires to equal
// Injected.
func (r Requests) Terminal() int {
	return r.Completed + r.StillQueued + r.StillRunning + r.DroppedUnservable + r.TimedOut +
		r.RoutingRejections + r.GatewayQueueDepth + r.GatewayQueueRejected +
		r.GatewayEvicted + r.GatewayExpired + r.EncodeRoutingRejections
}

// Goodput is the share of requests that met their SLO targets.
type Goodput struct {
	Targets       map[string]workload.SLODimTargets `yaml:"targets"`
	GoodputRPS    float64                           `yaml:"goodput_rps"`
	SLOAttainment float64                           `yaml:"slo_attainment"`
	PerClass      map[string]ClassGoodput           `yaml:"per_class,omitempty"`
}

// ClassGoodput is one SLO class's outcome.
type ClassGoodput struct {
	Completed int `yaml:"completed"`
	Met       int `yaml:"met"`
}

// Breakdowns split a run's metrics along the axes it exercised. Each slice is a map of
// metric name to value, in the units the matching MetricDefs entry states.
type Breakdowns struct {
	// PD holds the prefill/decode split of a disaggregated run.
	PD         map[string]float64            `yaml:"pd,omitempty"`
	PerModel   map[string]map[string]float64 `yaml:"per_model,omitempty"`
	PerTenant  map[string]map[string]float64 `yaml:"per_tenant,omitempty"`
	PerAdapter map[string]map[string]float64 `yaml:"per_adapter,omitempty"`
	Sessions   map[string]float64            `yaml:"sessions,omitempty"`
}

// Override is one setting the run simulated differently from its input.
type Override struct {
	Setting string `yaml:"setting"`
	// Input is the value Input states; Run is the value the run simulated.
	Input string `yaml:"input"`
	Run   string `yaml:"run"`
	// Source is what chose Run, for example a command-line flag.
	Source string `yaml:"source"`
}

// Artifact is a bulk output, by path relative to the Result document.
type Artifact struct {
	Ref    string `yaml:"ref"`
	SHA256 string `yaml:"sha256"`
	// Rows is the row count, when the artifact is a table.
	Rows *int `yaml:"rows,omitempty"`
}

// Runtime is the host- and wall-clock-dependent part of a run.
type Runtime struct {
	WallClockS float64 `yaml:"wall_clock_s,omitempty"`
	// StartedUTC is an RFC 3339 timestamp.
	StartedUTC string            `yaml:"started_utc,omitempty"`
	Host       map[string]string `yaml:"host,omitempty"`
}
