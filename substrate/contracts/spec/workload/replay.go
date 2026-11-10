package workload

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/routing"
)

// The citations in this file are to aiperf 0.12.0's source, under src/aiperf/: the
// scenario declaration common/scenario/inferencex_agentx_mvp.py (mvp.py below) and the
// lock that applies it, common/scenario/validator.py.

// Replay is a Scenario's third "what traffic" arm: an agentic trace-replay workload and
// the run's lane count. It takes one of two forms.
//
// Inline, it states the whole definition beside Concurrency. As a reference, Workload
// names a catalog replay workload (a workloads/<name>.yaml whose top-level key is
// replay:, see CatalogReplay), and the binding states only the run's choices: Concurrency,
// and optionally the seed, the window, the grace period, the context filter and the
// acceptance, which override the catalog's. ValidateAgainstCatalog loads the entry, and
// Resolve merges the two into an inline Replay.
type Replay struct {
	// Workload names a catalog replay workload by its file stem. Empty for an inline
	// replay.
	Workload         string `yaml:"workload,omitempty"`
	ReplayDefinition `yaml:",inline"`
	// Concurrency is the number of session-tree lanes: live trees, not requests.
	// Subagents fan out beside their parent without taking a lane, so requests in flight
	// can far exceed it. One value: a scenario run is not a sweep (validator.py:825-837).
	// It is the run's choice, so a catalog entry does not state it.
	Concurrency int `yaml:"concurrency"`
}

// CatalogReplay is a catalog replay workload: a definition named by its file, with no
// lane count, which each run chooses.
type CatalogReplay struct {
	Name string
	ReplayDefinition
}

// Validate checks the definition and that the entry has a name.
func (c *CatalogReplay) Validate() *validate.Problems {
	p := c.ReplayDefinition.Validate()
	if c.Name == "" {
		p.Field("name", "required")
	}
	return p
}

// Validate performs field-level validation of a replay. An inline one is validated in
// full; a reference states a catalog name, a lane count, and only run choices, and the
// definition it resolves to is validated when it is resolved.
func (r *Replay) Validate() *validate.Problems {
	p := &validate.Problems{}
	if r.Concurrency < 1 {
		p.Field("concurrency", "must be at least 1: the number of session-tree lanes")
	}
	if r.Workload == "" {
		r.ReplayDefinition.validateInto(p)
		return p
	}
	if fixed := r.catalogFixedFields(); len(fixed) > 0 {
		p.Errorf("a reference to catalog replay %q may state only the run's choices "+
			"(concurrency, random_seed, benchmark_duration_s, grace_period_s, max_context_length, "+
			"acceptance, routing); it also states %v, which the catalog entry fixes", r.Workload, fixed)
	}
	r.validateRunChoices(p)
	return p
}

// catalogFixedFields names the keys a reference states that only a definition may.
func (r *Replay) catalogFixedFields() []string {
	d := r.ReplayDefinition
	var out []string
	add := func(set bool, key string) {
		if set {
			out = append(out, key)
		}
	}
	add(d.Scenario != "", "scenario")
	add(d.HarnessVersion != "", "harness_version")
	add(d.Corpus != (Corpus{}), "corpus")
	add(d.CacheBust != "", "cache_bust")
	add(d.SystemIdleGapCapS != nil, "system_idle_gap_cap_s")
	add(d.TrajectoryStart != nil, "trajectory_start")
	add(d.Warmup != nil, "warmup")
	add(d.SplitFlattenedAgents != nil, "split_flattened_agents")
	add(d.Streaming != nil, "streaming")
	add(d.IgnoreEOS != nil, "ignore_eos")
	add(d.IgnoreTraceDelays != nil, "ignore_trace_delays")
	add(d.SynthesisMaxISL != 0, "synthesis_max_isl")
	add(d.TraceIdleGapCapS != nil, "trace_idle_gap_cap_s")
	add(d.InterTurnDelayCapS != nil, "inter_turn_delay_cap_s")
	add(d.Category != "", "category")
	add(d.Thinking != "", "thinking")
	return out
}

// validateRunChoices checks the overrides a reference may state.
func (r *Replay) validateRunChoices(p *validate.Problems) {
	d := r.ReplayDefinition
	if p.FiniteField("benchmark_duration_s", d.BenchmarkDurationS) {
		switch {
		case d.BenchmarkDurationS < 0:
			p.Field("benchmark_duration_s", "must not be negative")
		case d.BenchmarkDurationS > 0 && d.BenchmarkDurationS < 900:
			p.Field("benchmark_duration_s",
				"%v is below the scenario's 900 s floor (mvp.py:32; validator.py:589-602)", d.BenchmarkDurationS)
		}
	}
	if g := d.GracePeriodS; g != nil && (math.IsNaN(*g) || *g < 0) {
		p.Field("grace_period_s", "must be a non-negative number of seconds or .inf, got %v", *g)
	}
	if d.Acceptance != nil {
		p.Merge("acceptance", d.Acceptance.Validate())
	}
	if d.Routing != nil {
		p.Merge("routing", d.Routing.Validate())
	}
}

// Resolve merges a reference with the catalog entry it names: the entry's definition, with
// the reference's run choices over it. It returns an inline Replay, validated, or an error
// naming what is wrong. An inline replay resolves to itself.
func (r *Replay) Resolve(entry *CatalogReplay) (*Replay, error) {
	if r.Workload == "" {
		return r, nil
	}
	if entry == nil || entry.Name != r.Workload {
		return nil, fmt.Errorf("replay workload %q: the catalog entry was not supplied", r.Workload)
	}
	out := &Replay{ReplayDefinition: entry.ReplayDefinition, Concurrency: r.Concurrency}
	run := r.ReplayDefinition
	if run.RandomSeed != nil {
		out.RandomSeed = run.RandomSeed
	}
	if run.BenchmarkDurationS != 0 {
		out.BenchmarkDurationS = run.BenchmarkDurationS
	}
	if run.GracePeriodS != nil {
		out.GracePeriodS = run.GracePeriodS
	}
	if run.MaxContextLength != 0 {
		out.MaxContextLength = run.MaxContextLength
	}
	if run.Acceptance != nil {
		out.Acceptance = run.Acceptance
	}
	// A run's own measured routing imbalance is a run choice, as its acceptance is: it outranks
	// the catalog entry's, and stating it is never dropped (a reference used to validate with it
	// and resolve without it).
	if run.Routing != nil {
		out.Routing = run.Routing
	}
	if p := r.Validate(); !p.OK() {
		return nil, fmt.Errorf("replay workload %q: %s", r.Workload, p.Error())
	}
	if p := out.Validate(); !p.OK() {
		return nil, fmt.Errorf("replay workload %q resolves to an invalid replay:\n%s", r.Workload, p.Error())
	}
	return out, nil
}

// ReplayDefinition is an agentic trace-replay workload apart from the run's lane count: a
// named trace-replay scenario run over a pinned corpus, stated exactly enough that a simulator reproduces what the
// harness would send, such as "aiperf 0.12.0 --scenario inferencex-agentx-mvp over
// semianalysisai/cc-traces-weka-062126-256k at revision R". It describes the run and
// carries no replay logic: the corpus stays in its file, and how a session tree is rebuilt
// and dispatched is the harness's behaviour, which a simulator implements.
//
// Every knob the scenario locks has a field, so a document can state it, and validation
// refuses a value the harness's scenario validator would refuse. An omitted field takes
// the value the harness applies; the accessors return it.
type ReplayDefinition struct {
	// Scenario names the harness scenario. See ReplayScenario.
	Scenario ReplayScenario `yaml:"scenario"`
	// HarnessVersion is the harness release whose behaviour the scenario reproduces.
	HarnessVersion string `yaml:"harness_version"`
	Corpus         Corpus `yaml:"corpus"`

	// RandomSeed seeds lane snapshots and prompt content. Omitted, it is 42, the value
	// the target command and InferenceX use; aiperf itself would draw a random one
	// (validator.py:788-816), which no simulation can reproduce.
	RandomSeed *int64 `yaml:"random_seed,omitempty"`
	// BenchmarkDurationS is the profiling phase's sending window, at least 900 s
	// (mvp.py:32; validator.py:589-602). Zero means the scenario default, 1800 s
	// (mvp.py:33; validator.py:577-586).
	BenchmarkDurationS float64 `yaml:"benchmark_duration_s,omitempty"`
	// GracePeriodS is how long requests in flight at the end of the window may finish
	// before they are cancelled without a record. Omitted, it is aiperf's
	// --benchmark-grace-period default, 30 s (config/flags/cli_config.py:1937-1949);
	// .inf waits indefinitely. It is the one field of a Result that holds infinity, and says
	// so with the infinity tag the strict loader reads (check.InfinityTag, SUB-12).
	GracePeriodS *float64 `yaml:"grace_period_s,omitempty" infinity:"allowed"`
	// MaxContextLength drops every trace whose recorded peak prompt plus output exceeds
	// it, at load time. Omitted, no trace is dropped.
	MaxContextLength ContextLimit `yaml:"max_context_length,omitempty"`

	// CacheBust is how a tree's prompts are made unique to it. The scenario requires
	// first_turn_prefix (mvp.py:39; validator.py:503-544); omitted, it is that.
	CacheBust CacheBust `yaml:"cache_bust,omitempty"`
	// SystemIdleGapCapS caps globally idle replay time: when nothing is in flight, every
	// pending timer is pulled in so the system is never idle longer. The scenario locks
	// it to 10 s (mvp.py:36; validator.py:748-785); omitted, it is that.
	SystemIdleGapCapS *float64 `yaml:"system_idle_gap_cap_s,omitempty"`
	// TrajectoryStart is the window of a trace's recorded span each lane's snapshot is
	// drawn from. The scenario fills [0, 1], the whole span (mvp.py:34-35); unlike the
	// locks above, aiperf honours an explicit window (validator.py:610-639).
	TrajectoryStart *TrajectoryStart `yaml:"trajectory_start,omitempty"`
	// Warmup is how the replay primes the server before profiling. Omitted, it is the
	// synthesized warmup.
	Warmup *Warmup `yaml:"warmup,omitempty"`
	// SplitFlattenedAgents is whether the loader splits top-level requests a capture
	// flattened into separate agent chains by hash-prefix detection, which changes the
	// streams and their concurrency. Omitted, it is true, aiperf's
	// WEKA_SPLIT_FLATTENED_AGENTS default (common/environment.py:383-384).
	SplitFlattenedAgents *bool `yaml:"split_flattened_agents,omitempty"`

	// The knobs the scenario forbids or forces to one value. Each may be stated only as
	// that value.
	//
	// Streaming must be true (mvp.py:11; validator.py:234-263), IgnoreEOS true (mvp.py:10;
	// validator.py:266-300) and IgnoreTraceDelays false (mvp.py:12; validator.py:303-327).
	// SynthesisMaxISL and InterTurnDelayCapS must be unset: the scenario forbids input
	// truncation (mvp.py:13; validator.py:364-396) and a per-turn delay cap (mvp.py:38;
	// validator.py:642-693).
	//
	// TraceIdleGapCapS is a recorded version extension of the 0.12.0 lock. The scenario's
	// default forbids per-trace idle compression (mvp.py:37; validator.py:696-745), but the
	// InferenceX fork run states --trace-idle-gap-cap-seconds, a per-trajectory idle
	// watchdog that bounds observed runtime idle across a session tree without rewriting its
	// timestamps or spawn/join gates. The schema allows an explicit positive value;
	// cross-repo-changes.md §"Version discipline" records such an extension deliberately.
	Streaming          *bool    `yaml:"streaming,omitempty"`
	IgnoreEOS          *bool    `yaml:"ignore_eos,omitempty"`
	IgnoreTraceDelays  *bool    `yaml:"ignore_trace_delays,omitempty"`
	SynthesisMaxISL    int      `yaml:"synthesis_max_isl,omitempty"`
	TraceIdleGapCapS   *float64 `yaml:"trace_idle_gap_cap_s,omitempty"`
	InterTurnDelayCapS *float64 `yaml:"inter_turn_delay_cap_s,omitempty"`

	// Category, Thinking and Acceptance state the replay's acceptance terms, as a Shape's
	// do. A run that forced its acceptance carries the forced value here, or in the
	// scenario's pin.
	Category   string              `yaml:"category,omitempty"`
	Thinking   acceptance.Thinking `yaml:"thinking,omitempty"`
	Acceptance *acceptance.Value   `yaml:"acceptance,omitempty"`
	// Routing is the traffic's own measured expert-parallel routing imbalance, which
	// outranks the registry's profile for its Category (see routing.Resolve).
	Routing *routing.Value `yaml:"routing,omitempty"`
}

// ReplayScenario names a trace-replay scenario a harness implements. The set is closed:
// a scenario is a bundle of locked behaviour, and one not transcribed here cannot be
// validated.
type ReplayScenario string

// ScenarioInferenceXAgentXMVP is aiperf's inferencex-agentx-mvp: agentic replay of
// SemiAnalysis's Weka Claude Code traces, closed loop per stream, session trees counted
// against concurrency.
const ScenarioInferenceXAgentXMVP ReplayScenario = "inferencex-agentx-mvp"

var replayScenarios = map[ReplayScenario]bool{ScenarioInferenceXAgentXMVP: true}

// Valid reports whether s is a recognized replay scenario.
func (s ReplayScenario) Valid() bool { return replayScenarios[s] }

// AllReplayScenarios returns every replay scenario, sorted.
func AllReplayScenarios() []string { return enumerate(replayScenarios) }

// harnessVersions are the harness releases whose scenario behaviour is transcribed here.
var harnessVersions = map[ReplayScenario]map[string]bool{
	ScenarioInferenceXAgentXMVP: {"0.12.0": true},
}

// AgentXLoaders are the loaders inferencex-agentx-mvp accepts (mvp.py:14-31), each with
// the Hugging Face repository it reads. A public-dataset alias fixes its own repository
// (aiperf plugin/plugins.yaml, metadata.hf_dataset_name), so a corpus naming an alias and
// another repository describes a run that cannot have happened. weka_hf reads the one
// repository the scenario allows (validator.py:56, 485-500), and weka_trace, a local
// directory, reads none and is refused, since it cannot pin the corpus (validator.py:
// 466-483).
var AgentXLoaders = map[string]string{
	"semianalysis_cc_traces_weka_with_subagents":             "semianalysisai/cc-traces-weka-062126",
	"semianalysis_cc_traces_weka_with_subagents_256k":        "semianalysisai/cc-traces-weka-062126-256k",
	"semianalysis_cc_traces_weka_with_subagents_060226":      "semianalysisai/cc-traces-weka-with-subagents-060226",
	"semianalysis_cc_traces_weka_with_subagents_060226_256k": "semianalysisai/cc-traces-weka-with-subagents-060226-256k",
	"semianalysis_cc_traces_weka_with_subagents_060526":      "semianalysisai/cc-traces-weka-with-subagents-060526",
	"semianalysis_cc_traces_weka_with_subagents_060526_256k": "semianalysisai/cc-traces-weka-with-subagents-060526-256k",
	"semianalysis_cc_traces_weka_with_subagents_060826":      "semianalysisai/cc-traces-weka-with-subagents-060826",
	"semianalysis_cc_traces_weka_with_subagents_060826_256k": "semianalysisai/cc-traces-weka-with-subagents-060826-256k",
	"semianalysis_cc_traces_weka_061326":                     "semianalysisai/cc-traces-weka-061326",
	"semianalysis_cc_traces_weka_061326_256k":                "semianalysisai/cc-traces-weka-061326-256k",
	"semianalysis_cc_traces_weka_061526":                     "semianalysisai/cc-traces-weka-061526",
	"semianalysis_cc_traces_weka_061526_256k":                "semianalysisai/cc-traces-weka-061526-256k",
	"semianalysis_cc_traces_weka_062126":                     "semianalysisai/cc-traces-weka-062126",
	"semianalysis_cc_traces_weka_062126_256k":                "semianalysisai/cc-traces-weka-062126-256k",
	loaderWekaHF:    agentXWekaHFRepo,
	loaderWekaTrace: "",
}

// AgentXRollingLoaders are the aliases plugins.yaml describes as a "generic rolling alias
// that tracks the current default" corpus: the repository they read today is not the one
// they will read after the default moves, so a run named by one is dated only by its
// revision and digest.
var AgentXRollingLoaders = map[string]bool{
	"semianalysis_cc_traces_weka_with_subagents":      true,
	"semianalysis_cc_traces_weka_with_subagents_256k": true,
}

const (
	loaderWekaTrace = "weka_trace"
	loaderWekaHF    = "weka_hf"
	// agentXWekaHFRepo is the one repository weka_hf may name (validator.py:56, 485-500).
	agentXWekaHFRepo = "semianalysisai/cc-traces-weka-062126"
)

// Corpus pins the traces a replay reads, by reference: the rows stay in the file.
type Corpus struct {
	// Loader is the harness loader, or --public-dataset alias, that reads the corpus,
	// such as semianalysis_cc_traces_weka_062126_256k.
	Loader string `yaml:"loader"`
	// Format is the record schema. See CorpusFormat.
	Format CorpusFormat `yaml:"format"`
	// Encoding is how records are stored. See CorpusEncoding.
	Encoding CorpusEncoding `yaml:"encoding"`
	Source   CorpusSource   `yaml:"source"`
	// SHA256 is the digest of the corpus file, 64 hex characters: the Hugging Face LFS
	// object id (X-Linked-ETag) of the file at the revision. Required, because the harness
	// reads whatever the default branch holds and only the digest can show it read these
	// bytes.
	SHA256 string `yaml:"sha256"`
}

// CorpusSource is where a corpus is published, pinned to one revision.
type CorpusSource struct {
	// HuggingFace is the dataset repository id, such as
	// semianalysisai/cc-traces-weka-062126-256k.
	HuggingFace string `yaml:"huggingface"`
	// Revision is the repository commit the corpus was observed at. It is a record, not a
	// pin aiperf applies: aiperf 0.12.0's Weka loaders set no hf_revision, so they read
	// the repository's default branch whatever this says (dataset/loader/
	// base_hf_dataset.py:28, 77). What ties a run to these bytes is Corpus.SHA256, which
	// is therefore required.
	Revision string `yaml:"revision"`
	// File is the path of the corpus file in the repository, and Split the dataset split
	// the loader reads. Both optional.
	File  string `yaml:"file,omitempty"`
	Split string `yaml:"split,omitempty"`
}

// CorpusFormat is a corpus's record schema.
type CorpusFormat string

// CorpusWekaTraceV7 is SemiAnalysis's Weka trace format, version 7: one trace per record,
// with request, subagent and hash-id fields (aiperf dataset/loader/weka_trace_models.py).
const CorpusWekaTraceV7 CorpusFormat = "weka-trace-v7"

var corpusFormats = map[CorpusFormat]bool{CorpusWekaTraceV7: true}

// Valid reports whether f is a recognized corpus format.
func (f CorpusFormat) Valid() bool { return corpusFormats[f] }

// AllCorpusFormats returns every corpus format, sorted.
func AllCorpusFormats() []string { return enumerate(corpusFormats) }

// CorpusEncoding is how a corpus's records are stored.
type CorpusEncoding string

// CorpusJSONL is one JSON record per line.
const CorpusJSONL CorpusEncoding = "jsonl"

var corpusEncodings = map[CorpusEncoding]bool{CorpusJSONL: true}

// Valid reports whether e is a recognized corpus encoding.
func (e CorpusEncoding) Valid() bool { return corpusEncodings[e] }

// AllCorpusEncodings returns every corpus encoding, sorted.
func AllCorpusEncodings() []string { return enumerate(corpusEncodings) }

// CacheBust is how a replay makes a tree's prompts unique to it.
type CacheBust string

// CacheBustFirstTurnPrefix prefixes one marker per session tree to its first user
// message, so prefix reuse happens only within a tree (aiperf
// timing/strategies/cache_bust.py).
const CacheBustFirstTurnPrefix CacheBust = "first_turn_prefix"

var cacheBusts = map[CacheBust]bool{CacheBustFirstTurnPrefix: true}

// Valid reports whether c is a recognized cache-bust mode.
func (c CacheBust) Valid() bool { return cacheBusts[c] }

// AllCacheBusts returns every cache-bust mode, sorted.
func AllCacheBusts() []string { return enumerate(cacheBusts) }

// TrajectoryStart is the fraction of a trace's recorded span, from its first request's
// start to its last's, a lane's snapshot time is drawn from uniformly. Each bound is in
// [0, 1] and an omitted one takes the scenario's.
type TrajectoryStart struct {
	MinRatio *float64 `yaml:"min_ratio,omitempty"`
	MaxRatio *float64 `yaml:"max_ratio,omitempty"`
}

// Warmup is how a replay primes the server before profiling.
type Warmup struct {
	Mode WarmupMode `yaml:"mode"`
	// DurationS bounds a cache_pressure warmup by wall-clock time
	// (--agentic-cache-warmup-duration); no other mode takes it. A cache_pressure warmup
	// states exactly one of DurationS or RequestsPerLane.
	DurationS float64 `yaml:"duration_s,omitempty"`
	// RequestsPerLane bounds a cache_pressure warmup by request count: after the snapshot
	// primers, each lane sends this many additional warmup wire requests, each capped to one
	// output token (aiperf --warmup-requests-per-lane). The request-bounded alternative to
	// DurationS; no other mode takes it.
	RequestsPerLane int `yaml:"requests_per_lane,omitempty"`
}

// WarmupMode is a replay's warmup.
type WarmupMode string

const (
	// WarmupSynthesized: for every stream mid-flight at its lane's snapshot, its last
	// earlier request is sent with max_tokens 1 and its full prior context, aligned so
	// every snapshot lands at one instant; profiling starts when all have returned
	// (aiperf timing/config.py _build_agentic_warmup_config, unmeasured).
	WarmupSynthesized WarmupMode = "synthesized"
	// WarmupCachePressure: the accelerated cache-pressure warmup, which runs for a
	// stated duration rather than one request per stream
	// (--agentic-cache-warmup-duration).
	WarmupCachePressure WarmupMode = "cache_pressure"
)

var warmupModes = map[WarmupMode]bool{WarmupSynthesized: true, WarmupCachePressure: true}

// Valid reports whether m is a recognized warmup mode.
func (m WarmupMode) Valid() bool { return warmupModes[m] }

// AllWarmupModes returns every warmup mode, sorted.
func AllWarmupModes() []string { return enumerate(warmupModes) }

// ContextLimit is a replay's trace filter, which has three states: unstated (zero, no
// filter), a stated count of tokens, or the deployment's own context length, written
// max_model_len, which a resolver replaces with the deployment's resolved max_model_len.
type ContextLimit int

// ContextLimitMaxModelLen filters to the deployment's resolved max_model_len.
const ContextLimitMaxModelLen ContextLimit = -1

// FromDeployment reports whether the limit is the deployment's max_model_len.
func (c ContextLimit) FromDeployment() bool { return c == ContextLimitMaxModelLen }

// Resolve returns the token limit given the deployment's resolved max_model_len, and
// whether a limit applies. It refuses max_model_len with no resolved length.
func (c ContextLimit) Resolve(maxModelLen int) (int, bool, error) {
	switch {
	case c > 0:
		return int(c), true, nil
	case c == ContextLimitMaxModelLen && maxModelLen < 1:
		return 0, false, fmt.Errorf("max_context_length: max_model_len needs the deployment's resolved max_model_len")
	case c == ContextLimitMaxModelLen:
		return maxModelLen, true, nil
	default:
		return 0, false, nil
	}
}

// UnmarshalYAML accepts a positive whole number or max_model_len.
func (c *ContextLimit) UnmarshalYAML(n *yaml.Node) error {
	if n.Kind == yaml.ScalarNode {
		if n.Value == "max_model_len" {
			*c = ContextLimitMaxModelLen
			return nil
		}
		var v int
		if err := validate.DecodeNode(n, &v); err == nil && v > 0 {
			*c = ContextLimit(v)
			return nil
		}
	}
	return fmt.Errorf("line %d: max_context_length must be a positive token count or max_model_len", n.Line)
}

// MarshalYAML writes max_model_len for the deployment's length, so a round trip is stable.
func (c ContextLimit) MarshalYAML() (any, error) {
	if c == ContextLimitMaxModelLen {
		return "max_model_len", nil
	}
	return int(c), nil
}

// Seed returns the random seed, 42 when unstated.
func (r *ReplayDefinition) Seed() int64 {
	if r.RandomSeed != nil {
		return *r.RandomSeed
	}
	return 42
}

// Duration returns the profiling window in seconds, 1800 when unstated.
func (r *ReplayDefinition) Duration() float64 {
	if r.BenchmarkDurationS > 0 {
		return r.BenchmarkDurationS
	}
	return 1800
}

// Grace returns the grace period in seconds, 30 when unstated.
func (r *ReplayDefinition) Grace() float64 {
	if r.GracePeriodS != nil {
		return *r.GracePeriodS
	}
	return 30
}

// IdleCap returns the global idle cap in seconds, 10.
func (r *ReplayDefinition) IdleCap() float64 {
	if r.SystemIdleGapCapS != nil {
		return *r.SystemIdleGapCapS
	}
	return 10
}

// TraceIdleCap returns the per-trace idle cap in seconds and whether it is set. Unlike the
// global cap it has no default: the scenario's default leaves it off, so a caller must
// distinguish an unset cap from one stated as a value.
func (r *ReplayDefinition) TraceIdleCap() (float64, bool) {
	if r.TraceIdleGapCapS != nil {
		return *r.TraceIdleGapCapS, true
	}
	return 0, false
}

// StartWindow returns the snapshot window's bounds, [0, 1] where unstated.
func (r *ReplayDefinition) StartWindow() (float64, float64) {
	lo, hi := 0.0, 1.0
	if r.TrajectoryStart != nil {
		if r.TrajectoryStart.MinRatio != nil {
			lo = *r.TrajectoryStart.MinRatio
		}
		if r.TrajectoryStart.MaxRatio != nil {
			hi = *r.TrajectoryStart.MaxRatio
		}
	}
	return lo, hi
}

// CheckStartWindow records the problems of a snapshot window stated as optional bounds:
// each stated bound finite and in [0, 1], and the lower not above the upper, an omitted
// bound taking the scenario's 0 or 1. Other documents that record a window the replay
// used (a point record's trajectory_start) call it so one rule holds everywhere. path
// names the window in the document.
func CheckStartWindow(p *validate.Problems, path string, minRatio, maxRatio *float64) {
	for field, v := range map[string]*float64{"min_ratio": minRatio, "max_ratio": maxRatio} {
		if v != nil && p.FiniteField(path+"."+field, *v) && (*v < 0 || *v > 1) {
			p.Field(path+"."+field, "must lie in [0, 1], got %v", *v)
		}
	}
	lo, hi := 0.0, 1.0
	if minRatio != nil {
		lo = *minRatio
	}
	if maxRatio != nil {
		hi = *maxRatio
	}
	if lo > hi {
		p.Field(path, "min_ratio %v exceeds max_ratio %v", lo, hi)
	}
}

// WarmupMode returns the warmup mode, synthesized when unstated.
func (r *ReplayDefinition) WarmupMode() WarmupMode {
	if r.Warmup != nil {
		return r.Warmup.Mode
	}
	return WarmupSynthesized
}

// SplitsFlattenedAgents reports whether flattened agent chains are split, true when
// unstated.
func (r *ReplayDefinition) SplitsFlattenedAgents() bool {
	return r.SplitFlattenedAgents == nil || *r.SplitFlattenedAgents
}

// RoutingTerms returns what the replay states about routing, for routing.Resolve.
func (r *ReplayDefinition) RoutingTerms() routing.Workload {
	return routing.Workload{Category: r.Category, Value: r.Routing}
}

// AcceptanceTerms returns what the replay states about acceptance, for acceptance.Resolve.
func (r *ReplayDefinition) AcceptanceTerms() acceptance.Workload {
	return acceptance.Workload{Category: r.Category, Thinking: r.Thinking, Value: r.Acceptance}
}

// Validate performs field-level validation of a definition, refusing what the harness's
// scenario validator refuses for the named scenario.
func (r *ReplayDefinition) Validate() *validate.Problems {
	p := &validate.Problems{}
	r.validateInto(p)
	return p
}

func (r *ReplayDefinition) validateInto(p *validate.Problems) {
	switch {
	case r.Scenario == "":
		p.Field("scenario", "required: one of %v", AllReplayScenarios())
	case !r.Scenario.Valid():
		p.Field("scenario", "%q is not one of %v", r.Scenario, AllReplayScenarios())
	case r.HarnessVersion == "":
		p.Field("harness_version", "required: the harness release whose behaviour the replay reproduces")
	case !harnessVersions[r.Scenario][r.HarnessVersion]:
		p.Field("harness_version", "%s is not transcribed for %s; known: %v",
			r.HarnessVersion, r.Scenario, enumerate(harnessVersions[r.Scenario]))
	}
	p.Merge("corpus", r.Corpus.Validate())

	if p.FiniteField("benchmark_duration_s", r.BenchmarkDurationS) {
		switch {
		case r.BenchmarkDurationS < 0:
			p.Field("benchmark_duration_s", "must not be negative")
		case r.BenchmarkDurationS > 0 && r.BenchmarkDurationS < 900:
			p.Field("benchmark_duration_s",
				"%v is below the scenario's 900 s floor (mvp.py:32; validator.py:589-602)", r.BenchmarkDurationS)
		}
	}
	if g := r.GracePeriodS; g != nil && (math.IsNaN(*g) || *g < 0) {
		p.Field("grace_period_s", "must be a non-negative number of seconds or .inf, got %v", *g)
	}
	if r.CacheBust != "" && r.CacheBust != CacheBustFirstTurnPrefix {
		p.Field("cache_bust", "the scenario requires %s, got %q (validator.py:503-544)",
			CacheBustFirstTurnPrefix, r.CacheBust)
	}
	if c := r.SystemIdleGapCapS; c != nil && *c != 10 {
		p.Field("system_idle_gap_cap_s", "the scenario locks it to 10 s, got %v (validator.py:748-785)", *c)
	}
	if t := r.TrajectoryStart; t != nil {
		CheckStartWindow(p, "trajectory_start", t.MinRatio, t.MaxRatio)
	}
	if w := r.Warmup; w != nil {
		// Reject a non-finite duration first: NaN/±Inf must not slip past the
		// exclusivity/positivity checks below (NaN != 0 is true, NaN < 0 is false), which
		// would otherwise validate a cache_pressure warmup the reference promises is positive.
		switch {
		case !w.Mode.Valid():
			p.Field("warmup.mode", "%q is not one of %v", w.Mode, AllWarmupModes())
		case !p.FiniteField("warmup.duration_s", w.DurationS):
		case w.Mode != WarmupCachePressure && w.DurationS != 0:
			p.Field("warmup.duration_s", "only a cache_pressure warmup takes a duration")
		case w.Mode != WarmupCachePressure && w.RequestsPerLane != 0:
			p.Field("warmup.requests_per_lane", "only a cache_pressure warmup takes a per-lane request count")
		case w.Mode == WarmupCachePressure && (w.DurationS != 0) == (w.RequestsPerLane != 0):
			// A cache_pressure warmup is bounded by time or by request count, never both and
			// never neither: --agentic-cache-warmup-duration and --warmup-requests-per-lane are
			// mutually exclusive.
			p.Field("warmup", "a cache_pressure warmup states exactly one of duration_s or requests_per_lane")
		case w.DurationS < 0:
			p.Field("warmup.duration_s", "a cache_pressure warmup duration must be positive")
		case w.RequestsPerLane < 0:
			p.Field("warmup.requests_per_lane", "a cache_pressure per-lane request count must be positive")
		}
	}
	forced := func(field string, v *bool, want bool, cite string) {
		if v != nil && *v != want {
			p.Field(field, "the scenario requires %v (%s)", want, cite)
		}
	}
	forced("streaming", r.Streaming, true, "mvp.py:11; validator.py:234-263")
	forced("ignore_eos", r.IgnoreEOS, true, "mvp.py:10; validator.py:266-300")
	forced("ignore_trace_delays", r.IgnoreTraceDelays, false, "mvp.py:12; validator.py:303-327")
	if r.SynthesisMaxISL != 0 {
		p.Field("synthesis_max_isl", "the scenario forbids input truncation (mvp.py:13; validator.py:364-396)")
	}
	if c := r.TraceIdleGapCapS; c != nil && p.FiniteField("trace_idle_gap_cap_s", *c) && *c <= 0 {
		p.Field("trace_idle_gap_cap_s", "must be a positive number of seconds, got %v", *c)
	}
	if r.InterTurnDelayCapS != nil {
		p.Field("inter_turn_delay_cap_s", "the scenario forbids a per-turn delay cap (mvp.py:38; validator.py:642-693)")
	}
	acceptance.ValidateWorkloadTerms(p, r.Category, r.Thinking, r.Acceptance)
	if r.Routing != nil {
		p.Merge("routing", r.Routing.Validate())
	}
}

// Validate checks a corpus reference: a loader the scenario accepts that pins the corpus,
// a known format and encoding, and a source pinned to a revision.
func (c *Corpus) Validate() *validate.Problems {
	p := &validate.Problems{}
	repo, accepted := AgentXLoaders[c.Loader]
	switch {
	case c.Loader == "":
		p.Field("loader", "required: the harness loader or public-dataset alias")
	case !accepted:
		p.Field("loader", "%q is not a loader the scenario accepts (mvp.py:14-31): %v",
			c.Loader, loaderNames())
	case c.Loader == loaderWekaTrace:
		p.Field("loader", "a local weka_trace directory cannot pin the corpus; use a public-dataset alias or weka_hf (validator.py:466-483)")
	case c.Loader == loaderWekaHF && c.Source.HuggingFace != repo:
		p.Field("source.huggingface", "weka_hf may only read %s, got %q (validator.py:485-500)",
			repo, c.Source.HuggingFace)
	case c.Source.HuggingFace != "" && c.Source.HuggingFace != repo:
		p.Field("source.huggingface", "loader %s reads %s (aiperf plugin/plugins.yaml), not %q",
			c.Loader, repo, c.Source.HuggingFace)
	}
	if AgentXRollingLoaders[c.Loader] {
		w := &validate.Problems{}
		w.Warnf("%s is a generic rolling alias that tracks the current default corpus; name the dated alias so the run stays reproducible", c.Loader)
		p.Merge("loader", w)
	}
	switch {
	case c.Format == "":
		p.Field("format", "required: one of %v", AllCorpusFormats())
	case !c.Format.Valid():
		p.Field("format", "%q is not one of %v", c.Format, AllCorpusFormats())
	}
	switch {
	case c.Encoding == "":
		p.Field("encoding", "required: one of %v", AllCorpusEncodings())
	case !c.Encoding.Valid():
		p.Field("encoding", "%q is not one of %v", c.Encoding, AllCorpusEncodings())
	}
	if c.Source.HuggingFace == "" || strings.Count(c.Source.HuggingFace, "/") != 1 {
		p.Field("source.huggingface", "required: a dataset repository id, owner/name, got %q", c.Source.HuggingFace)
	}
	if c.Source.Revision == "" {
		p.Field("source.revision", "required: the repository commit the corpus was observed at, recorded beside its sha256")
	}
	switch {
	case c.SHA256 == "":
		p.Field("sha256", "required: aiperf 0.12.0 reads the default branch, not source.revision, so only the corpus file's digest ties a run to its bytes")
	case len(c.SHA256) != 64 || !isHex(c.SHA256):
		p.Field("sha256", "must be a 64-character hex digest of the corpus file")
	}
	return p
}

// CatalogWorkload is one catalog workloads/<name>.yaml, which is either a traffic shape
// or, when its top-level key is replay:, a replay workload. Exactly one is set.
type CatalogWorkload struct {
	Shape  *Shape
	Replay *CatalogReplay
}

// Name is the entry's name, its file stem.
func (w *CatalogWorkload) Name() string {
	if w.Replay != nil {
		return w.Replay.Name
	}
	if w.Shape != nil {
		return w.Shape.Name
	}
	return ""
}

// Validate validates whichever form the entry takes.
func (w *CatalogWorkload) Validate() *validate.Problems {
	switch {
	case w.Shape != nil && w.Replay != nil:
		p := &validate.Problems{}
		p.Errorf("a catalog workload is a shape or a replay, not both")
		return p
	case w.Replay != nil:
		return w.Replay.Validate()
	case w.Shape != nil:
		return w.Shape.Validate()
	}
	p := &validate.Problems{}
	p.Errorf("a catalog workload states neither a shape nor a replay")
	return p
}

func loaderNames() []string {
	out := make([]string, 0, len(AgentXLoaders))
	for k := range AgentXLoaders {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
