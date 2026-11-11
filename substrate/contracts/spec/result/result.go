// Package result describes what one BLIS run predicts: the document `blis run` and
// `blis replay` emit, one per run of one configuration under one experimental frame.
//
// A Result names every figure by its identifier in the metric registry
// (contracts/metrics, DEV-8) and carries no definition of its own, so a reader looks the
// definition up and two figures compare only when their definitions are the same.
//
// A Result records what the run read and how: the engine it ran (SIM-1), the catalog and
// registry it loaded, with their content hashes and overlays, and whether that is released
// data (design §3.3, X8); the registry entries its lookups selected and the coefficient
// files no lookup read (VAL-6); every option that changes a byte of the document (SIM-10),
// with the source of each workload property (D8); and its experimental frame (design §8.6,
// DEV-7). Each request it generated has one fate (DEV-9), and the document counts the
// fates and may list one row per request.
//
// Given the same input, options and data, the simulator reproduces the document byte for
// byte outside Runtime, RerunOf and RevisionChanges. Runtime is the only part that depends
// on the host or the wall clock.
//
// The simulator writes a Result as JSON by default and as YAML on request, with Marshal;
// Unmarshal reads either. The yaml tags define the keys of both encodings, so a Result is
// written with Marshal, not with encoding/json, which would read no tag here.
package result

import (
	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

// Kind is the value of Result.Kind.
const Kind = "Result"

// SchemaVersion is the Result schema this package reads and writes. Version 2 added the
// engine, data, selected entries, options, workload properties, frame, fates and request
// rows, and replaced metric_defs by registry identifiers. Version 3 requires each figure to
// state its scope, and its member and aggregate where it has them (DEV-8, design §8.7): a
// version 2 figure states none, and a definition is never inferred, so it is not read.
const SchemaVersion = 3

// Result is one simulator run's predicted output.
type Result struct {
	Kind          string `yaml:"kind"`
	SchemaVersion int    `yaml:"schema_version"`
	// Name identifies the run, a lowercase slug, for example "qwen3-32b-h100-pd.c64-s1".
	Name string `yaml:"name"`

	// Engine is the engine the run used, chosen by the scenario's engine (SIM-1).
	Engine Engine `yaml:"engine"`
	// Data is the catalog and registry the run read, and whether they are released data.
	Data Data `yaml:"data"`
	// SelectedEntries are the registry entries the run's lookups returned, and the
	// coefficient files no lookup read (VAL-6).
	SelectedEntries SelectedEntries `yaml:"selected_entries"`

	// Input is everything the run was given, after resolution.
	Input Input `yaml:"input"`
	// Frame is the run's experimental frame (design §8.6).
	Frame Frame `yaml:"frame"`
	// Revisions are the versions of the BLIS modules the run was built from.
	Revisions Revisions `yaml:"revisions"`

	// RerunOf names the Result whose input this run was produced from, when it was.
	RerunOf *Reference `yaml:"rerun_of,omitempty"`
	// RevisionChanges lists each component whose revision differs from RerunOf's. Empty
	// unless RerunOf is set.
	RevisionChanges []RevisionChange `yaml:"revision_changes,omitempty"`

	// Window is the measured window as it fell in this run.
	Window Window `yaml:"window"`
	// Metrics are the predicted figures, keyed by metric-registry identifier. A run that
	// fails an acceptance condition of its frame reports none (DEV-7).
	Metrics map[string]Figure `yaml:"metrics,omitempty"`

	// Requests counts the fate of every request the run generated (DEV-9).
	Requests Requests `yaml:"requests"`
	// RequestRows, when present, holds one row per generated request.
	RequestRows []RequestRow `yaml:"request_rows,omitempty"`

	// Goodput is present when the input states SLO targets.
	Goodput *Goodput `yaml:"goodput,omitempty"`
	// Breakdowns split the run along the axes it exercised.
	Breakdowns *Breakdowns `yaml:"breakdowns,omitempty"`
	// Reports are each component's own report, carried as reported.
	Reports Reports `yaml:"reports,omitempty"`

	// Overrides lists every setting the run simulated differently from Input, and what
	// chose the value. These are the host's choices.
	Overrides []Override `yaml:"overrides,omitempty"`
	// Resolutions are the engine's own resolution of the settings it resolves: for each, the
	// value it was asked for and the value it resolved to. They sit beside Overrides: an
	// override is what the host changed, a resolution is what the engine then did with the
	// value (design §11.2, which grades a point G2 where a resolved value that a benchmark
	// run logged differs from the engine's resolution of the value passed).
	Resolutions []Resolution `yaml:"resolutions,omitempty"`
	// Artifacts are bulk outputs, by reference.
	Artifacts map[string]Artifact `yaml:"artifacts,omitempty"`

	// Runtime is the only part of the document that depends on the host or the wall
	// clock. Determinism and round-trip comparisons skip it.
	Runtime *Runtime `yaml:"runtime,omitempty"`
}

// Engine is the engine a run used: its kind and the upstream release it models.
type Engine struct {
	// Kind names the engine, for example "vllm".
	Kind string `yaml:"kind"`
	// Version is the upstream release, for example "0.32.0".
	Version string `yaml:"version"`
}

// Data records which catalog and registry a run read (design §3.3, X8).
type Data struct {
	Catalog  DataSource `yaml:"catalog"`
	Registry DataSource `yaml:"registry"`
	// ReleasedData is true when the simulator is a tagged release whose stack.json names a
	// tagged substrate release, the catalog and registry hashes equal that release's
	// embedded hashes, and no overlay is used. Only such a run's figures are published.
	ReleasedData bool `yaml:"released_data"`
	// Build is the simulator build: a release tag such as "v0.16.0", or BuildUnreleased.
	Build string `yaml:"build"`
}

// BuildUnreleased is Data.Build for any simulator that is not a tagged release: a local
// build, go run, or stack CI.
const BuildUnreleased = "unreleased"

// DataSource is one of the catalog and the registry as a run read it.
type DataSource struct {
	// Source is where the base data came from.
	Source Source `yaml:"source"`
	// Hash is the content hash of the base data, 64 lowercase hex characters, computed as
	// design §3.3 defines it.
	Hash string `yaml:"hash"`
	// Overlays are the overlay directories applied on top, in the order applied.
	Overlays []Overlay `yaml:"overlays,omitempty"`
	// CombinedHash is the content hash of the data after every overlay. With no overlay it
	// equals Hash.
	CombinedHash string `yaml:"combined_hash"`
}

// Source says where a run's base catalog or registry came from.
type Source string

const (
	// SourceEmbedded: the copy compiled into the substrate the simulator was built with.
	SourceEmbedded Source = "embedded"
	// SourcePath: a directory named by --catalog, --registry, BLIS_CATALOG or
	// BLIS_REGISTRY.
	SourcePath Source = "path"
)

// Overlay is one overlay directory a run applied.
type Overlay struct {
	// Hash is the overlay's content hash.
	Hash string `yaml:"hash"`
}

// ClosenessVersion is the version of the closeness data a SelectedEntries can carry.
const ClosenessVersion = 1

// SelectedEntries is the selected-entry report (VAL-6). Every reader behind the engine's
// Environment contributes to it, the acceptance and routing profile readers included.
type SelectedEntries struct {
	// ClosenessVersion says the recorder passed on everything each coefficient's entry states:
	// its kernel implementations, fit range and interval, and the flags found from them. It is
	// 1 for the layout of this package, and absent in a report written before the data existed
	// or by a recorder that held only identities. It does not say the conditions were decided.
	// Under version 1, extrapolated is evaluated only for a coefficient that has a fit_range,
	// unidentified only for one that has a ci95, and the kernel hash only for one that has
	// kernel_impls; where the data is absent the entry did not state it, and the flag's false is
	// "not evaluated". Without the version, nothing is said.
	ClosenessVersion int `yaml:"closeness_version,omitempty"`
	// Coefficients are the coefficient entries a lookup returned, each once, sorted by
	// set, name, method and scope.
	Coefficients []SelectedCoefficient `yaml:"coefficients,omitempty"`
	// AcceptanceProfiles are the acceptance profiles a lookup returned.
	AcceptanceProfiles []SelectedProfile `yaml:"acceptance_profiles,omitempty"`
	// RoutingProfiles are the routing profiles a lookup returned.
	RoutingProfiles []SelectedProfile `yaml:"routing_profiles,omitempty"`
	// UnreadFiles are the registry's coefficient, acceptance and routing files, relative
	// to the registry directory with forward slashes, from which no lookup returned an
	// entry. Sorted.
	UnreadFiles []string `yaml:"unread_files,omitempty"`
}

// SelectedCoefficient identifies one registry coefficient as design §3.3 does: by its set,
// its name, its scope and its method, and records what a reader needs to judge how far the
// run's price rests on it (design §11.3, the G1 conditions): the kernel implementations the
// entry is valid for, the range its fit covered, and whether the run left that range or the
// entry is not identified by its data.
type SelectedCoefficient struct {
	Set    string            `yaml:"set"`
	Name   string            `yaml:"name"`
	Method vocab.Method      `yaml:"method"`
	Scope  coefficient.Scope `yaml:"scope"`
	// File is the registry file the entry was read from, relative to the registry
	// directory.
	File string `yaml:"file"`
	// KernelImpls are the kernel implementations the entry states it is valid for: the
	// library, its variant and the hash of the kernel's source (design D5, VAL-8), copied
	// from the entry. A reader maps each library version a benchmark run recorded to a
	// kernel hash and compares it with these. Empty when the entry states none, which is
	// allowed for an entry that is not tied to one implementation; a reader then cannot
	// compare a hash for it.
	KernelImpls []coefficient.KernelImpl `yaml:"kernel_impls,omitempty"`
	// FitRange is the range of inputs the entry's fit covered (design VAL-10), copied from
	// the entry. Empty when the entry records none, and then the extrapolation condition cannot
	// be evaluated for it.
	FitRange []coefficient.Bound `yaml:"fit_range,omitempty"`
	// CI95 is the entry's 95% interval, copied from the entry. Absent when the entry states
	// none, and then the identifiability condition cannot be evaluated for it.
	CI95 *coefficient.Interval `yaml:"ci95,omitempty"`
	// Extrapolated is true when at least one step the run priced read the entry at an input
	// outside FitRange. The cost model reports it from the trust verdict of each priced step
	// (kernel.Estimate.Trust), so a Result written before any step ran says false. It
	// requires a fit range.
	Extrapolated bool `yaml:"extrapolated,omitempty"`
	// Unidentified is true when the entry's data do not determine it: its 95% interval is
	// unbounded or wider than half its value (design VAL-10). It is a property of the entry,
	// known when the cost model is built. It requires CI95: without an interval the entry has
	// not said, and false means only that nothing was found.
	Unidentified bool `yaml:"unidentified,omitempty"`
}

// SelectedProfile identifies one acceptance or routing profile by its set and its key, as
// the profile's Key renders.
type SelectedProfile struct {
	Set  string `yaml:"set"`
	Key  string `yaml:"key"`
	File string `yaml:"file"`
}

// Input is the resolved input of a run.
type Input struct {
	Scenario   scenario.Scenario     `yaml:"scenario"`
	Deployment deployment.Deployment `yaml:"deployment"`
	// Packaging is the scenario cluster's packaging as the run resolved it
	// (scenario.Cluster.Packaging): the GPUs per node and per rack it simulated, and
	// whether each was stated by the scenario or taken from the chip's catalog entry
	// (SIM-10). The scenario above is as written, so where it leaves a value to the
	// catalog this is the only record of the value the run used. Required when the
	// scenario states no gpus_per_node.
	Packaging *scenario.Packaging `yaml:"packaging,omitempty"`
	// Policy is the simulator's policy bundle (autoscaler, node pools, instance
	// lifecycle, priority, preemption), verbatim, when one was given.
	Policy map[string]any `yaml:"policy,omitempty"`
	// Options are every command-line option and environment variable that changes a byte
	// of this document, with the value the run used (SIM-10). Sorted by name.
	Options []Option `yaml:"options,omitempty"`
	// WorkloadProperties are the values of quantities that depend on the request stream
	// and that BLIS does not simulate, such as speculative acceptance or expert routing
	// imbalance, each with what supplied it (D8). Sorted by name.
	WorkloadProperties []WorkloadProperty `yaml:"workload_properties,omitempty"`
}

// Option is one option that changes the Result.
type Option struct {
	// Name is the flag without its dashes, or the environment variable's name.
	Name string `yaml:"name"`
	// Value is the value the run used, as text.
	Value string `yaml:"value"`
	// Source says whether the value was given or is the option's default.
	Source OptionSource `yaml:"source"`
}

// OptionSource says where an option's value came from.
type OptionSource string

const (
	OptionCommandLine OptionSource = "command_line"
	OptionEnvironment OptionSource = "environment"
	// OptionDefault: the option was not given and its default applied.
	OptionDefault OptionSource = "default"
)

// WorkloadProperty is one workload property's value and its source (D8).
type WorkloadProperty struct {
	// Name is the property, for example "speculative_acceptance" or "expert_routing".
	Name string `yaml:"name"`
	// Values are the property's figures, keyed by name, for example
	// {acceptance_length: 3.8}. Empty only when Source is PropertyDefault.
	Values map[string]float64 `yaml:"values,omitempty"`
	// Source is what supplied the values, in D8's order of precedence.
	Source PropertySource `yaml:"source"`
	// Profile names the registry profile, as set and key, when Source is
	// PropertyProfile.
	Profile string `yaml:"profile,omitempty"`
	// Step is how a profile lookup found its value (D8): "exact" for the run's exact key, or
	// the borrowing step that supplied it: "same_drafter_other_k", "other_drafter" or
	// "other_target". Present exactly when Source is PropertyProfile and the property is a
	// profile lookup that can borrow (speculative acceptance).
	Step string `yaml:"step,omitempty"`
	// Differed names each key field in which a borrowed profile differs from the run's:
	// model, drafter, speculative_method or num_spec_tokens. Empty for an exact match.
	Differed []string `yaml:"differed,omitempty"`
	// KRule says how a borrowed value was brought to the run's K when Step is
	// "same_drafter_other_k": "interpolated", "above_largest", "below_smallest_rates" or
	// "below_smallest_proportional".
	KRule string `yaml:"k_rule,omitempty"`
	// Profiles lists every profile the value was read from, as set and key, the one that sets
	// it first. An interpolation lists the two it lies between. Present with Step.
	Profiles []string `yaml:"profiles,omitempty"`
}

// PropertySource is what supplied a workload property, highest precedence first (D8).
type PropertySource string

const (
	PropertyCommandLine PropertySource = "command_line"
	PropertyScenario    PropertySource = "scenario"
	// PropertyWorkload: the workload the scenario binds states the value, or it was
	// derived from the trace.
	PropertyWorkload PropertySource = "workload"
	// PropertyProfile: the substrate's profile for the workload's identity.
	PropertyProfile PropertySource = "profile"
	// PropertyDefault: no source states a value and the consumer's documented neutral
	// value applies, such as evenly loaded expert ranks.
	PropertyDefault PropertySource = "default"
)

// Frame is a run's experimental frame (design §8.6): what it feeds in, when it measures,
// and what it measures.
type Frame struct {
	// Kind says whether the run is terminating or steady-state (design §8.8).
	Kind FrameKind `yaml:"kind"`
	// Generator is the workload and its seeds.
	Generator Generator `yaml:"generator"`
	// InitialState is the engines' state when the run starts.
	InitialState InitialState `yaml:"initial_state"`
	// Phases define warm-up, the measured window and drain, each by a rule.
	Phases Phases `yaml:"phases"`
	// Stop is the stop rule.
	Stop StopRule `yaml:"stop"`
	// Acceptance are the conditions under which the run counts for this frame.
	Acceptance []AcceptanceCondition `yaml:"acceptance,omitempty"`
	// Metrics are the registry identifiers the frame measures, sorted. Every key of
	// Result.Metrics is one of them.
	Metrics []string `yaml:"metrics"`
}

// FrameKind is terminating or steady-state.
type FrameKind string

const (
	// FrameTerminating measures a fixed amount of work, such as a benchmark of a set
	// number of prompts at a set concurrency.
	FrameTerminating FrameKind = "terminating"
	// FrameSteadyState measures a system that has settled.
	FrameSteadyState FrameKind = "steady_state"
)

// Generator is the frame's input: a workload with its offered load, or a trace to replay,
// and the seeds.
type Generator struct {
	// Workload is the workload the run used: the scenario's binding unless one replaced
	// it.
	Workload *workload.Binding `yaml:"workload,omitempty"`
	// Load is the offered load: exactly one of a concurrency and a request rate. A replay
	// of fixed timestamps states neither.
	Load measurement.Load `yaml:"load,omitempty"`
	// Seed is the scenario seed every substream is derived from (DEV-6).
	Seed int64 `yaml:"seed"`
	// RNG names the random generator and its version, because a new generator changes
	// every draw (design §8.4).
	RNG string `yaml:"rng"`
}

// InitialState is the engines' state at the start of a run.
type InitialState struct {
	Caches CacheState `yaml:"caches"`
	// Rule states how caches were warmed, when Caches is CachesWarmed.
	Rule string `yaml:"rule,omitempty"`
}

// CacheState is the engines' caches at the start of a run.
type CacheState string

const (
	CachesEmpty  CacheState = "empty"
	CachesWarmed CacheState = "warmed"
)

// Phases are the frame's warm-up, measured window and drain.
type Phases struct {
	Warmup   WarmupPhase   `yaml:"warmup"`
	Measured MeasuredPhase `yaml:"measured"`
	Drain    DrainPhase    `yaml:"drain"`
}

// WarmupPhase is the leading part of a run excluded from the measured window.
type WarmupPhase struct {
	Rule WarmupRule `yaml:"rule"`
	// Requests is the number of leading requests excluded, for WarmupRequests.
	Requests int `yaml:"requests,omitempty"`
	// DurationS is the excluded leading time in simulated seconds, for WarmupDuration.
	DurationS float64 `yaml:"duration_s,omitempty"`
}

// WarmupRule is how a warm-up is defined.
type WarmupRule string

const (
	WarmupNone     WarmupRule = "none"
	WarmupRequests WarmupRule = "requests"
	WarmupDuration WarmupRule = "duration"
	// WarmupMarked: the workload marks its warm-up requests, as an agentic replay's
	// warm-up phase does.
	WarmupMarked WarmupRule = "marked"
	// WarmupMSER5: the cut that minimises the variance of the remaining mean, per output
	// series (design §8.8).
	WarmupMSER5 WarmupRule = "mser5"
)

// MeasuredPhase says what opens and closes the measured window.
type MeasuredPhase struct {
	Start WindowStart `yaml:"start"`
	Stop  WindowStop  `yaml:"stop"`
}

// WindowStart is what opens the measured window.
type WindowStart string

const (
	// StartFirstArrivalAfterWarmup: the first arrival of a measured request.
	StartFirstArrivalAfterWarmup WindowStart = "first_arrival_after_warmup"
	// StartProfiling: the workload's own profiling phase starts, as aiperf's does.
	StartProfiling WindowStart = "profiling_start"
	// StartSimulation: the start of the run.
	StartSimulation WindowStart = "simulation_start"
)

// WindowStop is what closes the measured window.
type WindowStop string

const (
	StopLastCompletion WindowStop = "last_completion"
	StopProfiling      WindowStop = "profiling_end"
	StopSimulation     WindowStop = "simulation_end"
)

// DrainPhase is what happens after the measured window closes.
type DrainPhase struct {
	Rule DrainRule `yaml:"rule"`
	// DurationS bounds the drain in simulated seconds, for DrainGrace.
	DurationS float64 `yaml:"duration_s,omitempty"`
}

// DrainRule is how a run ends its work after the measured window.
type DrainRule string

const (
	// DrainNone: the run stops when the window closes.
	DrainNone DrainRule = "none"
	// DrainUntilComplete: the run continues until every request has a fate.
	DrainUntilComplete DrainRule = "until_complete"
	// DrainGrace: the run continues for a stated grace period.
	DrainGrace DrainRule = "grace"
)

// StopRule ends a run: a request count, a horizon, or both, whichever comes first.
type StopRule struct {
	// Requests caps the requests the workload generator emits.
	Requests int `yaml:"requests,omitempty"`
	// HorizonS is the simulated time, in seconds, at which the run stops.
	HorizonS *float64 `yaml:"horizon_s,omitempty"`
}

// AcceptanceCondition is one condition under which a run counts for its frame.
type AcceptanceCondition struct {
	Condition ConditionKind `yaml:"condition"`
	// Threshold is the condition's bound, in the condition's terms.
	Threshold float64 `yaml:"threshold"`
	// Observed is what the run achieved.
	Observed float64 `yaml:"observed"`
	// Met says whether Observed satisfies Threshold.
	Met bool `yaml:"met"`
}

// ConditionKind names an acceptance condition.
type ConditionKind string

const (
	// ConditionConcurrencyTolerance: the achieved mean concurrency differs from the
	// target by at most Threshold, a fraction of the target.
	ConditionConcurrencyTolerance ConditionKind = "achieved_concurrency_within"
	// ConditionUnfinishedAtMost: the fraction of generated requests in flight or queued
	// at the horizon is at most Threshold.
	ConditionUnfinishedAtMost ConditionKind = "unfinished_fraction_at_most"
)

// Revisions are the versions of the BLIS modules a run was built from, each a release tag,
// a Go pseudo-version or a commit.
type Revisions struct {
	// Simulator is the inference-sim build.
	Simulator string `yaml:"simulator"`
	// Substrate is the blis-substrate module version the build selected.
	Substrate string `yaml:"substrate"`
	// Router is the llm-d router fork's version.
	Router string `yaml:"router,omitempty"`
	// EngineModule is the engine repository's module version, such as blis-vllm's.
	EngineModule string `yaml:"engine_module,omitempty"`
}

// Reference names another Result document.
type Reference struct {
	Name   string `yaml:"name"`
	SHA256 string `yaml:"sha256"`
}

// RevisionChange is one component whose revision differs from the Result a run reran.
type RevisionChange struct {
	// Component is a key of Revisions, or "catalog" or "registry" for the data's combined
	// hash.
	Component string `yaml:"component"`
	From      string `yaml:"from"`
	To        string `yaml:"to"`
}

// Window is the measured window as it fell: its bounds on the simulated clock and the
// requests that completed inside it.
type Window struct {
	// StartUs and EndUs bound the window in simulated microseconds.
	StartUs int64 `yaml:"start_us"`
	EndUs   int64 `yaml:"end_us"`
	// DurationS is EndUs - StartUs in seconds.
	DurationS float64 `yaml:"duration_s"`
	// Requests is the number of requests that completed inside the window.
	Requests int `yaml:"requests"`
}

// Figure is one predicted value, the number of samples behind it, and the scope it is at. Its
// metric is the key it is held under in Result.Metrics, so a Result holds one figure of a
// metric: the one its frame measures, at the scope and aggregate it states.
type Figure struct {
	Value float64 `yaml:"value"`
	// Samples is the size of the population the statistic was taken over: requests,
	// token gaps or steps, as the metric's definition says. For a rate, the count summed.
	Samples int `yaml:"samples"`
	// Scope, Member, Aggregate and Over are the figure's Basis (design §8.7): the scope it is
	// at, the pool or instance it is of, and, for a figure that combines the figures of a
	// narrower scope, the aggregate and the scope it ranges over. Scope is required; the
	// others are stated where the figure has them. Basis returns them together.
	Scope     metrics.Scope     `yaml:"scope"`
	Member    string            `yaml:"member,omitempty"`
	Aggregate metrics.Aggregate `yaml:"aggregate,omitempty"`
	Over      metrics.Scope     `yaml:"over,omitempty"`
}

// Requests counts each request's fate (DEV-9). Generated is the generator's own count of
// the requests it created, every round a session planned included; it equals the sum of
// the fates.
type Requests struct {
	Generated int `yaml:"generated"`

	Completed int `yaml:"completed"`
	// Dropped counts dropped requests by reason.
	Dropped           map[DropReason]int `yaml:"dropped,omitempty"`
	TimedOut          int                `yaml:"timed_out"`
	InFlightAtHorizon int                `yaml:"in_flight_at_horizon"`
	QueuedAtHorizon   int                `yaml:"queued_at_horizon"`
	NotStarted        int                `yaml:"not_started"`

	// LengthCapped counts completed requests that stopped at a length limit (max_tokens or
	// the context length) rather than at end of sequence. It is not a fate.
	LengthCapped int `yaml:"length_capped"`
}

// Fate is how a request ended (design §8.7).
type Fate string

const (
	FateCompleted         Fate = "completed"
	FateDropped           Fate = "dropped"
	FateTimedOut          Fate = "timed_out"
	FateInFlightAtHorizon Fate = "in_flight_at_horizon"
	FateQueuedAtHorizon   Fate = "queued_at_horizon"
	// FateNotStarted: a request the workload planned, such as a later round of a session,
	// that was never issued.
	FateNotStarted Fate = "not_started"
)

// DropReason is why a request was dropped.
type DropReason string

const (
	// DropAdmissionRejected: refused at admission and never routed.
	DropAdmissionRejected DropReason = "admission_rejected"
	// DropRoutingRejected: the router found no endpoint to serve it.
	DropRoutingRejected DropReason = "routing_rejected"
	// DropEncodeRoutingRejected: refused at an encode stage.
	DropEncodeRoutingRejected DropReason = "encode_routing_rejected"
	// DropGatewayQueueRejected: the router's flow control refused it at capacity.
	DropGatewayQueueRejected DropReason = "gateway_queue_rejected"
	// DropGatewayEvicted: the router revoked it in flight.
	DropGatewayEvicted DropReason = "gateway_evicted"
	// DropGatewayExpired: it waited past the router's queue-wait budget.
	DropGatewayExpired DropReason = "gateway_expired"
	// DropUnservable: the engine refused it because it can never fit in the KV cache
	// (SIM-7).
	DropUnservable DropReason = "unservable"
	// DropContextOverflow: the engine refused it because prompt plus output exceed the
	// context length.
	DropContextOverflow DropReason = "context_overflow"
	// DropDecodeKV: a disaggregated request whose decode instance could not hold its KV.
	DropDecodeKV DropReason = "decode_kv_unavailable"
)

// Sum returns the number of fates counted, which DEV-9 requires to equal Generated.
func (q Requests) Sum() int {
	n := q.Completed + q.TimedOut + q.InFlightAtHorizon + q.QueuedAtHorizon + q.NotStarted
	for _, v := range q.Dropped {
		n += v
	}
	return n
}

// Unfinished returns the requests in flight or queued at the horizon.
func (q Requests) Unfinished() int { return q.InFlightAtHorizon + q.QueuedAtHorizon }

// RequestRow is one generated request. Times are on the simulated clock in microseconds.
// A latency is present only when the request produced the sample: no token, no TTFT; fewer
// than two output tokens, no TPOT; not completed, no E2E.
type RequestRow struct {
	// ID is the request's identifier, unique within the run (SIM-8).
	ID   string `yaml:"id"`
	Fate Fate   `yaml:"fate"`
	// DropReason is set exactly when Fate is dropped.
	DropReason DropReason `yaml:"drop_reason,omitempty"`

	// ArrivalUs is when the request arrived at the client.
	ArrivalUs int64  `yaml:"arrival_us"`
	TTFTUs    *int64 `yaml:"ttft_us,omitempty"`
	// TPOTUs is (E2E - TTFT) / (output tokens - 1).
	TPOTUs *float64 `yaml:"tpot_us,omitempty"`
	E2EUs  *int64   `yaml:"e2e_us,omitempty"`
	// SchedulingDelayUs is the time from arrival at an engine to the first scheduled step.
	SchedulingDelayUs *int64 `yaml:"scheduling_delay_us,omitempty"`
	// GatewayQueueDelayUs is the time spent in the router's flow-control queue.
	GatewayQueueDelayUs *int64 `yaml:"gateway_queue_delay_us,omitempty"`
	// CompletionIndex is the request's 1-based position in the run's order of completion,
	// for a completed request.
	CompletionIndex int `yaml:"completion_index,omitempty"`

	// The token record the engine emitted (SIM-12).
	PromptTokens         int `yaml:"prompt_tokens"`
	CachedLocalTokens    int `yaml:"cached_local_tokens,omitempty"`
	CachedExternalTokens int `yaml:"cached_external_tokens,omitempty"`
	ComputedTokens       int `yaml:"computed_tokens,omitempty"`
	OutputTokens         int `yaml:"output_tokens"`
	// LengthCapped: the request completed at a length limit rather than at end of
	// sequence.
	LengthCapped bool `yaml:"length_capped,omitempty"`

	// Phase is the request's benchmark phase, when the workload marks phases.
	Phase Phase `yaml:"phase,omitempty"`
	// Instance is the engine instance that served it.
	Instance   string `yaml:"instance,omitempty"`
	Model      string `yaml:"model,omitempty"`
	Adapter    string `yaml:"adapter,omitempty"`
	SLOClass   string `yaml:"slo_class,omitempty"`
	TenantID   string `yaml:"tenant_id,omitempty"`
	SessionID  string `yaml:"session_id,omitempty"`
	RoundIndex int    `yaml:"round_index,omitempty"`
	// Agentic is the request's place in an agentic replay, for such a workload.
	Agentic *AgenticTag `yaml:"agentic,omitempty"`
}

// Phase is a request's benchmark phase, after aiperf's benchmark_phase tag. Only
// PhaseWarmup is outside the measured window.
type Phase string

const (
	PhaseWarmup  Phase = "warmup"
	PhaseProfile Phase = "profile"
	PhaseGrace   Phase = "grace"
)

// AgenticTag places a request in an agentic replay's session trees.
type AgenticTag struct {
	// Lane is the concurrency lane whose session-tree slot the request runs under.
	Lane int `yaml:"lane"`
	// Tree numbers the tree instance, one per lane at start and one per recycle.
	Tree int `yaml:"tree"`
	// TraceID is the corpus trace the tree replays.
	TraceID string `yaml:"trace_id"`
	// Stream is the conversation: the trace id for the root, or the child session id.
	Stream string `yaml:"stream"`
	// Kind is root, subagent or flat.
	Kind AgenticKind `yaml:"kind"`
	// StreamInstance numbers the stream's instance within the run.
	StreamInstance int `yaml:"stream_instance"`
	// Turn is the turn index within the stream.
	Turn int `yaml:"turn"`
	// Outcome is how the load generator would record the request: completed, error,
	// context_overflow or cancelled. It is the client's classification, beside the fate.
	Outcome string `yaml:"outcome,omitempty"`
}

// AgenticKind is a stream's place in its tree.
type AgenticKind string

const (
	AgenticRoot     AgenticKind = "root"
	AgenticSubagent AgenticKind = "subagent"
	AgenticFlat     AgenticKind = "flat"
)

// Goodput is the SLO outcome: the targets and each class's result. The goodput_rps and
// slo_attainment figures are metrics.
type Goodput struct {
	Targets  map[string]workload.SLODimTargets `yaml:"targets"`
	PerClass map[string]ClassGoodput           `yaml:"per_class,omitempty"`
}

// ClassGoodput is one SLO class's outcome.
type ClassGoodput struct {
	Completed int `yaml:"completed"`
	Met       int `yaml:"met"`
}

// Breakdowns split a run's metrics along the axes it exercised. Each per-key slice maps a
// metric identifier to its value over that key's requests.
type Breakdowns struct {
	// PD holds the prefill/decode split of a disaggregated run, by quantity name.
	PD         map[string]float64            `yaml:"pd,omitempty"`
	PerModel   map[string]map[string]float64 `yaml:"per_model,omitempty"`
	PerTenant  map[string]map[string]float64 `yaml:"per_tenant,omitempty"`
	PerAdapter map[string]map[string]float64 `yaml:"per_adapter,omitempty"`
	// Sessions holds the multi-turn session split, by quantity name.
	Sessions map[string]float64 `yaml:"sessions,omitempty"`
}

// Reports are the components' own reports, keyed as each component keys them. The
// document carries them as reported; their contents are not figures under the registry.
type Reports struct {
	Engine     map[string]any `yaml:"engine,omitempty"`
	Router     map[string]any `yaml:"router,omitempty"`
	Kernel     map[string]any `yaml:"kernel,omitempty"`
	Saturation map[string]any `yaml:"saturation,omitempty"`
}

// IsZero reports whether no component reported, so the key is omitted.
func (r Reports) IsZero() bool {
	return len(r.Engine) == 0 && len(r.Router) == 0 && len(r.Kernel) == 0 && len(r.Saturation) == 0
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

// Resolution is the engine's resolution of one setting, beside the value the host gave it.
type Resolution struct {
	// Setting is the setting's name. It is the same name an Override uses for the same
	// setting, when the host overrode it.
	Setting string `yaml:"setting"`
	// Pool says which pool of the deployment resolved the setting, as PoolLabels names it:
	// the pool's role, or the role and its position among the pools of that role when
	// several share one. Absent when the resolution is of the whole deployment, or the
	// deployment has one pool. A disaggregated run's prefill and decode pools resolve
	// differently, so each states its own.
	Pool string `yaml:"pool,omitempty"`
	// Host is the value the host gave the engine: the host's value when Overrides names this
	// setting, else Input's. Empty when the host gave none and the engine chose.
	Host string `yaml:"host,omitempty"`
	// Engine is the value the engine resolved to, and the run simulated.
	Engine string `yaml:"engine"`
	// Reason says why Engine differs from Host. Required exactly when both are set and
	// differ.
	Reason string `yaml:"reason,omitempty"`
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
