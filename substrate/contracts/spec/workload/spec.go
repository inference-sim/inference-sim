package workload

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/check"
)

// Spec is the workload generator's specification: the document `blis run --workload-spec`
// reads and `blis convert` writes. It describes clients, each with an arrival process and
// token-length distributions, or populations of them (cohorts), or a source a converter
// expands into clients (ServeGen data, an inference-perf specification).
//
// The fields are transcribed from inference-sim's sim/workload/spec.go at be78af83, which
// defined the document before it moved here, with the two features v0.15.0 deprecated
// removed: the v1 SLO class names realtime and interactive, which were mapped to critical and
// standard with a warning, and ServeGen's single-period time_window with its derived
// span_start and span_end.
type Spec struct {
	// Version is the specification's version; it must be SpecVersion.
	Version string `yaml:"version"`
	// Seed seeds every random substream the generator draws.
	Seed int64 `yaml:"seed"`
	// Category is the workload's kind, or empty.
	Category SpecCategory `yaml:"category"`
	Clients  []ClientSpec `yaml:"clients"`
	Cohorts  []CohortSpec `yaml:"cohorts,omitempty"`
	// AggregateRate is the total request rate in requests per second, which rate-based
	// clients share by RateFraction. Zero selects absolute mode, where every rate-based
	// client's lifecycle window states its own trace_rate.
	AggregateRate float64 `yaml:"aggregate_rate"`
	// Horizon bounds generation in microseconds; zero leaves it to the run.
	Horizon int64 `yaml:"horizon,omitempty"`
	// NumRequests caps the requests generated; zero is no cap.
	NumRequests   int64              `yaml:"num_requests,omitempty"`
	ServeGenData  *ServeGenDataSpec  `yaml:"servegen_data,omitempty"`
	InferencePerf *InferencePerfSpec `yaml:"inference_perf,omitempty"`
	// GoodputSLOTargets are per-class latency targets for goodput.
	GoodputSLOTargets map[string]SLODimTargets `yaml:"goodput_slo_targets,omitempty"`
}

// SpecVersion is the generator specification's version. Version 1, whose SLO class names
// realtime and interactive were mapped on load, is refused.
const SpecVersion = "2"

// SpecCategory is a workload's kind.
type SpecCategory string

const (
	CategoryLanguage   SpecCategory = "language"
	CategoryMultimodal SpecCategory = "multimodal"
	CategoryReasoning  SpecCategory = "reasoning"
)

// SLOClass is a request's service class, as the router's admission policy reads it.
type SLOClass string

const (
	SLOCritical   SLOClass = "critical"
	SLOStandard   SLOClass = "standard"
	SLOSheddable  SLOClass = "sheddable"
	SLOBatch      SLOClass = "batch"
	SLOBackground SLOClass = "background"
)

var sloClasses = map[SLOClass]bool{"": true, SLOCritical: true, SLOStandard: true, SLOSheddable: true, SLOBatch: true, SLOBackground: true}

// Valid reports whether c is empty or one of the five classes.
func (c SLOClass) Valid() bool { return sloClasses[c] }

// ClientSpec is one client's behaviour.
type ClientSpec struct {
	ID       string   `yaml:"id"`
	TenantID string   `yaml:"tenant_id"`
	SLOClass SLOClass `yaml:"slo_class"`
	Model    string   `yaml:"model,omitempty"`
	// Adapter is the LoRA adapter the client's requests use; empty for the base model.
	Adapter string `yaml:"adapter,omitempty"`
	// RateFraction is the client's share of AggregateRate. A concurrency client states
	// Concurrency instead.
	RateFraction float64 `yaml:"rate_fraction"`
	// Concurrency is the number of requests the client keeps in flight.
	Concurrency int `yaml:"concurrency,omitempty"`
	// ThinkTimeUs is a concurrency client's pause between a response and its next request.
	ThinkTimeUs int64       `yaml:"think_time_us,omitempty"`
	Arrival     ArrivalSpec `yaml:"arrival"`
	InputDist   DistSpec    `yaml:"input_distribution"`
	OutputDist  DistSpec    `yaml:"output_distribution"`
	PrefixGroup string      `yaml:"prefix_group,omitempty"`
	// PrefixLength is the shared prefix's token count; the generator's default applies
	// when zero.
	PrefixLength int             `yaml:"prefix_length,omitempty"`
	Streaming    bool            `yaml:"streaming"`
	Network      *NetworkSpec    `yaml:"network,omitempty"`
	Lifecycle    *LifecycleSpec  `yaml:"lifecycle,omitempty"`
	Multimodal   *MultimodalSpec `yaml:"multimodal,omitempty"`
	Reasoning    *ReasoningSpec  `yaml:"reasoning,omitempty"`
	// Timeout is the per-request timeout in microseconds: nil for the generator's default,
	// 0 for none.
	Timeout *int64 `yaml:"timeout,omitempty"`
	// SLOTargetUs is a per-request TTFT target; nil or 0 is none.
	SLOTargetUs *int64 `yaml:"slo_target_us,omitempty"`
	// ClosedLoop: nil for the default (closed for reasoning or multi-turn clients), false
	// to pre-generate every round.
	ClosedLoop *bool `yaml:"closed_loop,omitempty"`
}

// CohortSpec is a population of clients sharing arrival behaviour and distributions. The
// generator expands it into Population clients; it states no client ID and no lifecycle,
// which Diurnal, Spike and Drain synthesise.
type CohortSpec struct {
	ID            string          `yaml:"id"`
	Population    int             `yaml:"population"`
	TenantID      string          `yaml:"tenant_id,omitempty"`
	SLOClass      SLOClass        `yaml:"slo_class,omitempty"`
	Model         string          `yaml:"model,omitempty"`
	Adapter       string          `yaml:"adapter,omitempty"`
	Arrival       ArrivalSpec     `yaml:"arrival"`
	InputDist     DistSpec        `yaml:"input_distribution"`
	OutputDist    DistSpec        `yaml:"output_distribution"`
	PrefixGroup   string          `yaml:"prefix_group,omitempty"`
	PrefixSharing PrefixSharing   `yaml:"prefix_sharing,omitempty"`
	Streaming     bool            `yaml:"streaming,omitempty"`
	RateFraction  float64         `yaml:"rate_fraction"`
	Diurnal       *DiurnalSpec    `yaml:"diurnal,omitempty"`
	Spike         *SpikeSpec      `yaml:"spike,omitempty"`
	Drain         *DrainSpec      `yaml:"drain,omitempty"`
	PrefixLength  int             `yaml:"prefix_length,omitempty"`
	Reasoning     *ReasoningSpec  `yaml:"reasoning,omitempty"`
	ClosedLoop    *bool           `yaml:"closed_loop,omitempty"`
	Timeout       *int64          `yaml:"timeout,omitempty"`
	SLOTargetUs   *int64          `yaml:"slo_target_us,omitempty"`
	Network       *NetworkSpec    `yaml:"network,omitempty"`
	Multimodal    *MultimodalSpec `yaml:"multimodal,omitempty"`
}

// PrefixSharing is whether a cohort's members share one prefix or each have their own.
type PrefixSharing string

const (
	PrefixShared    PrefixSharing = "shared"
	PrefixPerMember PrefixSharing = "per_member"
)

// MaxCohortPopulation caps a cohort's population, so a document cannot exhaust memory.
const MaxCohortPopulation = 100_000

// DiurnalSpec modulates a cohort's rate sinusoidally over 24 hours.
type DiurnalSpec struct {
	PeakHour          int     `yaml:"peak_hour"`
	PeakToTroughRatio float64 `yaml:"peak_to_trough_ratio"`
}

// SpikeSpec activates a cohort over [StartTimeUs, StartTimeUs + DurationUs).
type SpikeSpec struct {
	StartTimeUs int64 `yaml:"start_time_us"`
	DurationUs  int64 `yaml:"duration_us"`
	// TraceRate is the cohort's rate in absolute mode.
	TraceRate *float64 `yaml:"trace_rate,omitempty"`
}

// DrainSpec ramps a cohort's rate linearly to zero.
type DrainSpec struct {
	StartTimeUs    int64 `yaml:"start_time_us"`
	RampDurationUs int64 `yaml:"ramp_duration_us"`
}

// ArrivalSpec is the inter-arrival process.
type ArrivalSpec struct {
	Process ArrivalProcess `yaml:"process"`
	// CV is the inter-arrival coefficient of variation.
	CV *float64 `yaml:"cv,omitempty"`
	// Shape and Scale are fitted parameters (gamma alpha and theta, or Weibull k and lambda
	// in microseconds); when both are present they override CV.
	Shape *float64 `yaml:"shape,omitempty"`
	Scale *float64 `yaml:"scale,omitempty"`
}

// ArrivalProcess is an inter-arrival distribution family.
type ArrivalProcess string

const (
	ArrivalPoisson  ArrivalProcess = "poisson"
	ArrivalGamma    ArrivalProcess = "gamma"
	ArrivalWeibull  ArrivalProcess = "weibull"
	ArrivalConstant ArrivalProcess = "constant"
)

var arrivalProcesses = map[ArrivalProcess]bool{ArrivalPoisson: true, ArrivalGamma: true, ArrivalWeibull: true, ArrivalConstant: true}

// DistSpec parameterises a token-length distribution.
type DistSpec struct {
	Type DistType `yaml:"type"`
	// Params are the family's parameters, by name; for empirical, token count to
	// probability.
	Params map[string]float64 `yaml:"params,omitempty"`
	// File is an empirical distribution's file.
	File string `yaml:"file,omitempty"`
}

// DistType is a token-length distribution family.
type DistType string

const (
	DistGaussian        DistType = "gaussian"
	DistExponential     DistType = "exponential"
	DistParetoLognormal DistType = "pareto_lognormal"
	DistLognormal       DistType = "lognormal"
	DistEmpirical       DistType = "empirical"
	DistConstant        DistType = "constant"
)

// distParams are each family's required parameters, as inference-sim's NewLengthSampler
// requires them (sim/workload/distribution.go:236-300).
var distParams = map[DistType][]string{
	DistGaussian:        {"mean", "std_dev", "min", "max"},
	DistExponential:     {"mean"},
	DistParetoLognormal: {"alpha", "xm", "mu", "sigma", "mix_weight"},
	DistLognormal:       {"mu", "sigma"},
	DistConstant:        {"value"},
	DistEmpirical:       nil,
}

// NetworkSpec is a client's network.
type NetworkSpec struct {
	RTTMs         float64 `yaml:"rtt_ms"`
	BandwidthMbps float64 `yaml:"bandwidth_mbps,omitempty"`
}

// LifecycleSpec is the windows in which a client is active.
type LifecycleSpec struct {
	Windows []ActiveWindow `yaml:"windows"`
}

// ActiveWindow is one window of activity, with optional overrides of the client's
// parameters while it lasts.
type ActiveWindow struct {
	StartUs int64 `yaml:"start_us"`
	EndUs   int64 `yaml:"end_us"`
	// TraceRate is a weight for proportional allocation when AggregateRate is positive,
	// and the client's absolute rate in requests per second when it is zero.
	TraceRate  *float64     `yaml:"trace_rate,omitempty"`
	Arrival    *ArrivalSpec `yaml:"arrival,omitempty"`
	InputDist  *DistSpec    `yaml:"input_distribution,omitempty"`
	OutputDist *DistSpec    `yaml:"output_distribution,omitempty"`
}

// MultimodalSpec is a multimodal client's per-modality distributions.
type MultimodalSpec struct {
	TextDist       DistSpec `yaml:"text_distribution"`
	ImageDist      DistSpec `yaml:"image_distribution"`
	ImageCountDist DistSpec `yaml:"image_count_distribution"`
	AudioDist      DistSpec `yaml:"audio_distribution"`
	AudioCountDist DistSpec `yaml:"audio_count_distribution"`
	VideoDist      DistSpec `yaml:"video_distribution"`
	VideoCountDist DistSpec `yaml:"video_count_distribution"`
}

// ReasoningSpec is a reasoning client's output split and turns.
type ReasoningSpec struct {
	ReasonRatioDist DistSpec       `yaml:"reason_ratio_distribution"`
	MultiTurn       *MultiTurnSpec `yaml:"multi_turn,omitempty"`
}

// MultiTurnSpec is a multi-turn conversation.
type MultiTurnSpec struct {
	MaxRounds   int   `yaml:"max_rounds"`
	ThinkTimeUs int64 `yaml:"think_time_us"`
	// ContextGrowth is "accumulate" (each round resends the conversation) or empty (each
	// round's input is drawn independently).
	ContextGrowth string `yaml:"context_growth"`
	// SingleSession keeps one session per client.
	SingleSession bool `yaml:"single_session,omitempty"`
}

// ServeGenDataSpec names ServeGen data a converter expands into clients, in multi-period
// mode: each period lasts WindowDurationSecs, with DrainTimeoutSecs between periods.
type ServeGenDataSpec struct {
	Path               string `yaml:"path,omitempty"`
	WindowDurationSecs int    `yaml:"window_duration_secs,omitempty"`
	DrainTimeoutSecs   int    `yaml:"drain_timeout_secs,omitempty"`
}

// InferencePerfSpec is an inference-perf load specification a converter expands into
// clients.
type InferencePerfSpec struct {
	Stages       []StageSpec       `yaml:"stages"`
	SharedPrefix *SharedPrefixSpec `yaml:"shared_prefix"`
}

// StageSpec is one stage: a rate in requests per second for a duration in seconds.
type StageSpec struct {
	Rate     float64 `yaml:"rate"`
	Duration int64   `yaml:"duration"`
}

// SharedPrefixSpec is inference-perf's shared-prefix workload.
type SharedPrefixSpec struct {
	NumUniqueSystemPrompts  int  `yaml:"num_unique_system_prompts"`
	NumUsersPerSystemPrompt int  `yaml:"num_users_per_system_prompt"`
	SystemPromptLen         int  `yaml:"system_prompt_len"`
	QuestionLen             int  `yaml:"question_len"`
	OutputLen               int  `yaml:"output_len"`
	EnableMultiTurnChat     bool `yaml:"enable_multi_turn_chat"`
}

// Validate checks a specification as inference-sim's WorkloadSpec.Validate did, plus the
// version: every rule here is one that loader enforced.
func (s *Spec) Validate() *validate.Problems {
	p := &validate.Problems{}
	switch s.Version {
	case SpecVersion:
	case "", "1":
		p.Field("version", "is %q: version 1 and its SLO class names realtime and interactive are removed; state version %q with critical, standard, sheddable, batch or background", s.Version, SpecVersion)
	default:
		p.Field("version", "must be %q, got %q", SpecVersion, s.Version)
	}
	switch s.Category {
	case "", CategoryLanguage, CategoryMultimodal, CategoryReasoning:
	default:
		p.Field("category", "%q is not language, multimodal, reasoning, or empty", s.Category)
	}
	if len(s.Clients) == 0 && len(s.Cohorts) == 0 && s.ServeGenData == nil && s.InferencePerf == nil {
		p.Errorf("at least one client, cohort, servegen_data or inference_perf is required")
	}
	rateBased := len(s.Cohorts) > 0
	for _, c := range s.Clients {
		rateBased = rateBased || c.Concurrency == 0
	}
	if rateBased {
		if s.AggregateRate == 0 {
			for i, c := range s.Clients {
				if c.Concurrency != 0 {
					continue
				}
				if c.Lifecycle == nil || len(c.Lifecycle.Windows) == 0 {
					p.Field(fmt.Sprintf("clients[%d].lifecycle", i), "aggregate_rate is 0 (absolute mode), so a rate-based client states windows with trace_rate")
					continue
				}
				for j, w := range c.Lifecycle.Windows {
					if w.TraceRate == nil {
						p.Field(fmt.Sprintf("clients[%d].lifecycle.windows[%d].trace_rate", i, j), "required: aggregate_rate is 0 (absolute mode)")
					}
				}
			}
			for i, c := range s.Cohorts {
				at := fmt.Sprintf("cohorts[%d]", i)
				switch {
				case c.Diurnal != nil:
					p.Field(at+".diurnal", "a diurnal pattern is not supported in absolute mode; use spike with trace_rate")
				case c.Drain != nil:
					p.Field(at+".drain", "a drain pattern is not supported in absolute mode; use spike with trace_rate")
				case c.Spike == nil:
					p.Field(at+".spike", "required in absolute mode, with trace_rate: a cohort with no pattern generates nothing")
				case c.Spike.TraceRate == nil:
					p.Field(at+".spike.trace_rate", "required: aggregate_rate is 0 (absolute mode)")
				default:
					positive(p, at+".spike.trace_rate", *c.Spike.TraceRate)
				}
			}
		} else {
			positive(p, "aggregate_rate", s.AggregateRate)
		}
	}
	if s.Horizon < 0 {
		p.Field("horizon", "must not be negative")
	}
	if s.NumRequests < 0 {
		p.Field("num_requests", "must not be negative")
	}
	hasConcurrency, hasMultiTurn := false, false
	for i := range s.Clients {
		c := &s.Clients[i]
		c.validate(p, fmt.Sprintf("clients[%d]", i))
		hasConcurrency = hasConcurrency || c.Concurrency > 0
		hasMultiTurn = hasMultiTurn || (c.Reasoning != nil && c.Reasoning.MultiTurn != nil)
	}
	for i := range s.Cohorts {
		c := &s.Cohorts[i]
		c.validate(p, fmt.Sprintf("cohorts[%d]", i))
		hasMultiTurn = hasMultiTurn || (c.Reasoning != nil && c.Reasoning.MultiTurn != nil)
	}
	if hasConcurrency && hasMultiTurn {
		p.Errorf("concurrency clients and multi-turn clients cannot be mixed in one specification: the follow-up budget does not account for both")
	}
	// An empty slo_class reads as standard in the metrics, so a specification that states
	// a class for some clients must state one for all.
	explicit, empty := 0, 0
	if s.InferencePerf == nil && s.ServeGenData == nil {
		for _, c := range s.Clients {
			if c.SLOClass != "" {
				explicit++
			} else {
				empty++
			}
		}
	}
	for _, c := range s.Cohorts {
		if c.SLOClass != "" {
			explicit++
		} else {
			empty++
		}
	}
	if explicit > 0 && empty > 0 {
		p.Errorf("mixed slo_class: %d clients and cohorts state a class and %d do not; state one for all or none", explicit, empty)
	}
	if ip := s.InferencePerf; ip != nil {
		ip.validate(p)
	}
	if sg := s.ServeGenData; sg != nil {
		if sg.WindowDurationSecs < 0 {
			p.Field("servegen_data.window_duration_secs", "must not be negative")
		}
		if sg.DrainTimeoutSecs < 0 {
			p.Field("servegen_data.drain_timeout_secs", "must not be negative")
		}
	}
	for _, class := range sortedKeys(s.GoodputSLOTargets) {
		if class == "" {
			p.Field("goodput_slo_targets", "an SLO class key must not be empty")
			continue
		}
		p.Merge("goodput_slo_targets."+class, s.GoodputSLOTargets[class].Validate())
	}
	return p
}

func (c *ClientSpec) validate(p *validate.Problems, at string) {
	if !c.SLOClass.Valid() {
		p.Field(at+".slo_class", "%q is not critical, standard, sheddable, batch, background, or empty", c.SLOClass)
	}
	if c.Concurrency < 0 {
		p.Field(at+".concurrency", "must not be negative")
	}
	if c.ThinkTimeUs < 0 {
		p.Field(at+".think_time_us", "must not be negative")
	}
	switch {
	case c.Concurrency > 0 && c.RateFraction > 0:
		p.Field(at, "concurrency and rate_fraction are mutually exclusive")
	case c.Concurrency == 0:
		positive(p, at+".rate_fraction", c.RateFraction)
		c.Arrival.validate(p, at+".arrival")
	}
	if c.Concurrency == 0 && c.Arrival.CV != nil {
		positive(p, at+".arrival.cv", *c.Arrival.CV)
	}
	if c.PrefixLength < 0 {
		p.Field(at+".prefix_length", "must not be negative")
	}
	c.InputDist.validate(p, at+".input_distribution")
	c.OutputDist.validate(p, at+".output_distribution")
	common(p, at, c.Timeout, c.SLOTargetUs, c.Reasoning)
	if l := c.Lifecycle; l != nil {
		if len(l.Windows) == 0 {
			p.Field(at+".lifecycle.windows", "a lifecycle states at least one window")
		}
		for j, w := range l.Windows {
			wat := fmt.Sprintf("%s.lifecycle.windows[%d]", at, j)
			if w.StartUs < 0 {
				p.Field(wat+".start_us", "must not be negative")
			}
			if w.EndUs <= w.StartUs {
				p.Field(wat+".end_us", "is %d, not after start_us %d", w.EndUs, w.StartUs)
			}
			if w.TraceRate != nil {
				positive(p, wat+".trace_rate", *w.TraceRate)
			}
			if w.InputDist != nil {
				w.InputDist.validate(p, wat+".input_distribution")
			}
			if w.OutputDist != nil {
				w.OutputDist.validate(p, wat+".output_distribution")
			}
		}
	}
}

func (c *CohortSpec) validate(p *validate.Problems, at string) {
	if c.Population <= 0 {
		p.Field(at+".population", "must be positive")
	} else if c.Population > MaxCohortPopulation {
		p.Field(at+".population", "%d exceeds the maximum %d", c.Population, MaxCohortPopulation)
	}
	positive(p, at+".rate_fraction", c.RateFraction)
	if !c.SLOClass.Valid() {
		p.Field(at+".slo_class", "%q is not critical, standard, sheddable, batch, background, or empty", c.SLOClass)
	}
	c.Arrival.validate(p, at+".arrival")
	if c.Arrival.CV != nil {
		positive(p, at+".arrival.cv", *c.Arrival.CV)
	}
	c.InputDist.validate(p, at+".input_distribution")
	c.OutputDist.validate(p, at+".output_distribution")
	if d := c.Diurnal; d != nil {
		if math.IsNaN(d.PeakToTroughRatio) || math.IsInf(d.PeakToTroughRatio, 0) || d.PeakToTroughRatio < 1 {
			p.Field(at+".diurnal.peak_to_trough_ratio", "must be a finite number at least 1, got %v", d.PeakToTroughRatio)
		}
		if d.PeakHour < 0 || d.PeakHour > 23 {
			p.Field(at+".diurnal.peak_hour", "must lie in [0, 23], got %d", d.PeakHour)
		}
	}
	if s := c.Spike; s != nil {
		if s.StartTimeUs < 0 {
			p.Field(at+".spike.start_time_us", "must not be negative")
		}
		if s.DurationUs <= 0 {
			p.Field(at+".spike.duration_us", "must be positive")
		}
	}
	if d := c.Drain; d != nil {
		if d.StartTimeUs < 0 {
			p.Field(at+".drain.start_time_us", "must not be negative")
		}
		if d.RampDurationUs <= 0 {
			p.Field(at+".drain.ramp_duration_us", "must be positive")
		}
	}
	if c.PrefixLength < 0 {
		p.Field(at+".prefix_length", "must not be negative")
	}
	switch c.PrefixSharing {
	case "", PrefixShared:
	case PrefixPerMember:
		if c.PrefixGroup == "" {
			p.Field(at+".prefix_group", "required: prefix_sharing per_member shares within a group")
		}
	default:
		p.Field(at+".prefix_sharing", "%q is not shared or per_member", c.PrefixSharing)
	}
	common(p, at, c.Timeout, c.SLOTargetUs, c.Reasoning)
}

// common checks the fields a client and a cohort share.
func common(p *validate.Problems, at string, timeout, sloTarget *int64, r *ReasoningSpec) {
	if timeout != nil && *timeout < 0 {
		p.Field(at+".timeout", "must not be negative")
	}
	if sloTarget != nil && *sloTarget < 0 {
		p.Field(at+".slo_target_us", "must not be negative")
	}
	if r != nil && r.MultiTurn != nil {
		mt := r.MultiTurn
		if mt.MaxRounds < 1 {
			p.Field(at+".reasoning.multi_turn.max_rounds", "must be at least 1")
		}
		if mt.ContextGrowth != "" && mt.ContextGrowth != "accumulate" {
			p.Field(at+".reasoning.multi_turn.context_growth", "%q is not accumulate or empty", mt.ContextGrowth)
		}
	}
}

func (a ArrivalSpec) validate(p *validate.Problems, at string) {
	if !arrivalProcesses[a.Process] {
		p.Field(at+".process", "%q is not poisson, gamma, weibull or constant", a.Process)
	}
	fitted := a.Shape != nil && a.Scale != nil && *a.Shape > 0 && *a.Scale > 0
	if a.Process == ArrivalWeibull && a.CV != nil && !fitted && (*a.CV < 0.01 || *a.CV > 10.4) {
		p.Field(at+".cv", "a Weibull CV lies in [0.01, 10.4], got %v", *a.CV)
	}
	if a.Shape != nil {
		positive(p, at+".shape", *a.Shape)
	}
	if a.Scale != nil {
		positive(p, at+".scale", *a.Scale)
	}
}

func (d DistSpec) validate(p *validate.Problems, at string) {
	req, ok := distParams[d.Type]
	if !ok {
		p.Field(at+".type", "%q is not gaussian, exponential, pareto_lognormal, lognormal, empirical or constant", d.Type)
		return
	}
	for _, k := range sortedKeys(d.Params) {
		p.FiniteField(at+".params."+k, d.Params[k])
	}
	for _, k := range req {
		if _, ok := d.Params[k]; !ok {
			p.Field(at+".params", "%s requires parameter %q", d.Type, k)
		}
	}
	switch d.Type {
	case DistEmpirical:
		if d.File == "" && len(d.Params) == 0 {
			p.Field(at, "an empirical distribution states a file or inline params")
		}
		for _, k := range sortedKeys(d.Params) {
			if _, err := strconv.Atoi(k); err != nil {
				p.Field(at+".params."+k, "an empirical key is a token count, an integer")
			}
		}
	case DistExponential, DistLognormal:
		lo, hasLo := d.Params["min"]
		hi, hasHi := d.Params["max"]
		if hasLo && hasHi && lo > 0 && hi > 0 && lo > hi {
			p.Field(at+".params", "min (%v) must not exceed max (%v)", lo, hi)
		}
	}
}

func (ip *InferencePerfSpec) validate(p *validate.Problems) {
	if len(ip.Stages) == 0 {
		p.Field("inference_perf.stages", "at least one stage is required")
	}
	for i, st := range ip.Stages {
		at := fmt.Sprintf("inference_perf.stages[%d]", i)
		if st.Duration <= 0 {
			p.Field(at+".duration", "must be positive")
		}
		positive(p, at+".rate", st.Rate)
	}
	sp := ip.SharedPrefix
	if sp == nil {
		p.Field("inference_perf.shared_prefix", "required")
		return
	}
	if sp.NumUniqueSystemPrompts <= 0 {
		p.Field("inference_perf.shared_prefix.num_unique_system_prompts", "must be positive")
	}
	if sp.NumUsersPerSystemPrompt <= 0 {
		p.Field("inference_perf.shared_prefix.num_users_per_system_prompt", "must be positive")
	}
	for f, v := range map[string]int{"system_prompt_len": sp.SystemPromptLen, "question_len": sp.QuestionLen, "output_len": sp.OutputLen} {
		if v < 0 {
			p.Field("inference_perf.shared_prefix."+f, "must not be negative")
		}
	}
}

// positive records a problem unless v is a finite positive number.
func positive(p *validate.Problems, at string, v float64) {
	if p.FiniteField(at, v) && v <= 0 {
		p.Field(at, "must be positive, got %v", v)
	}
}

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
