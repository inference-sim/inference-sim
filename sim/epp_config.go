package sim

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"os"
	"sort"
	"strings"
	"time"

	"gopkg.in/yaml.v3"
)

// This file ingests an llm-d EndpointPickerConfig (#1930) verbatim and maps it
// onto BLIS's weighted-scoring pipeline, RoutingPipelineConfig, the P/D
// machinery (decode-first routing, deciders) and the flow-control gateway queue.
//
// Reference implementation (MANDATORY): llm-d-router main@981acb8a and v0.10.0
// (see routing_epp.go). Schema: apix/config/v1{,alpha1}/endpointpickerconfig_types.go.
// Defaulting: pkg/epp/config/loader/defaults.go (unnamed plugin ⇒ name = type; scorer
// weight default 1; max-score-picker and single-profile-handler injected).
// Parameters decode strictly (DisallowUnknownFields:
// pkg/epp/framework/interface/plugin/registry.go). Flow control is gated by
// featureGates: [flowControl] (pkg/epp/flowcontrol/config.go:27).
//
// Every plugin type is mapped onto a BLIS implementation, accepted and recorded as
// inert (infrastructure with no simulation meaning), or refused by name. A
// scheduling-affecting type or parameter BLIS does not model is never ignored.

// EPPIngest is the BLIS-side result of ingesting an EndpointPickerConfig.
type EPPIngest struct {
	Version EPPVersion
	// Scorers is the default (or decode) profile's scorer list.
	Scorers []ScorerConfig
	// Disaggregated is set for a disagg-profile-handler config: PrefillScorers and
	// DecodeScorers then hold the two profiles, and PDDecider/PDPrefixBased the
	// prefill decider.
	Disaggregated  bool
	PrefillScorers []ScorerConfig
	DecodeScorers  []ScorerConfig
	PDDecider      string // "never", "always" (always-disagg-pd-decider) or "prefix-based"
	PDPrefixBased  *LLMDPrefixPDDeciderConfig
	Pipeline       *RoutingPipelineConfig
	// FlowControl is nil when the flowControl feature gate is off (llm-d then
	// runs no flow controller).
	FlowControl *EPPFlowControl
	// MetricsRefreshIntervalUs is metrics-data-source's interval rounded to the EPP
	// base tick; 0 when not configured (llm-d then scrapes every 50ms tick, BLIS's
	// default --snapshot-refresh-interval).
	MetricsRefreshIntervalUs int64
	Provenance               EPPProvenance
}

// EPPFlowControl is the flowControl block mapped onto BLIS's gateway queue.
type EPPFlowControl struct {
	Detector           ConcurrencyDetectorConfig
	MaxRequests        int   // global queue capacity (0 = unlimited)
	MaxBytes           int64 // global estimated-byte capacity (0 = unlimited)
	PerBandMaxRequests int   // defaults for a band without its own config (registry/config.go)
	PerBandMaxBytes    int64
	Bands              []EPPPriorityBand
	RequestTTLUs       int64 // defaultRequestTTL (controller/config.go, default 60s; 0 disables)
	// UniformCeiling is static-usage-limit-policy's threshold: one saturation ceiling
	// for every band (main:…/flowcontrol/usagelimits/usagelimitpolicy.go:35-66);
	// 1.0 = no gating below saturation 1.
	UniformCeiling float64
}

// EPPPriorityBand is one flowControl.priorityBands entry.
type EPPPriorityBand struct {
	Priority    int    `json:"priority"`
	MaxRequests int    `json:"max_requests"`
	MaxBytes    int64  `json:"max_bytes"`
	Fairness    string `json:"fairness"` // "global-strict" or "round-robin"
}

// EPPProvenance is the `epp` output block: what was ingested and how it maps.
type EPPProvenance struct {
	ConfigSHA256      string                `json:"config_sha256"`
	LLMDRouterVersion string                `json:"llm_d_router_version"`
	Plugins           []EPPPluginRecord     `json:"plugins"`
	Profile           EPPProfileRecord      `json:"profile"`
	Profiles          []EPPProfileRecord    `json:"profiles,omitempty"`
	ProfileHandler    string                `json:"profile_handler"`
	PDDecider         string                `json:"pd_decider,omitempty"`
	FlowControl       *EPPFlowControlRecord `json:"flow_control,omitempty"`
	Inert             []string              `json:"inert,omitempty"`
	Notes             []string              `json:"notes,omitempty"`
	// Deviations are BLIS options that change llm-d's behavior (opt-ins such as the
	// CPU-tier view); empty for a run that follows the config as llm-d does.
	Deviations []string `json:"deviations,omitempty"`
}

// EPPPluginRecord describes one plugin instance.
type EPPPluginRecord struct {
	Type        string `json:"type"`
	Name        string `json:"name"`
	Role        string `json:"role"` // filter | role-filter | scorer | picker | profile-handler | decider | producer | inert
	BLIS        string `json:"blis"` // the BLIS implementation it maps onto
	AutoCreated bool   `json:"auto_created,omitempty"`
}

// EPPProfileRecord is one scheduling profile as run.
type EPPProfileRecord struct {
	Name    string            `json:"name"`
	Filters []string          `json:"filters"`
	Scorers []EPPScorerRecord `json:"scorers"`
	Picker  string            `json:"picker"`
}

// EPPScorerRecord is one weighted scorer of a profile.
type EPPScorerRecord struct {
	Ref    string  `json:"ref"`
	Type   string  `json:"type"`
	BLIS   string  `json:"blis"`
	Weight float64 `json:"weight"`
}

// EPPFlowControlRecord echoes the flow-control mapping.
type EPPFlowControlRecord struct {
	SaturationDetector string            `json:"saturation_detector"`
	MaxRequests        int               `json:"max_requests"`
	MaxBytes           int64             `json:"max_bytes"`
	PerBandMaxRequests int               `json:"per_band_max_requests"`
	PerBandMaxBytes    int64             `json:"per_band_max_bytes"`
	Bands              []EPPPriorityBand `json:"priority_bands,omitempty"`
	DefaultRequestTTLs float64           `json:"default_request_ttl_s"`
	EnableEviction     bool              `json:"enable_eviction"`
}

type eppFile struct {
	APIVersion         string          `yaml:"apiVersion"`
	Kind               string          `yaml:"kind"`
	FeatureGates       []string        `yaml:"featureGates"`
	Plugins            []eppPluginSpec `yaml:"plugins"`
	SchedulingProfiles []eppProfile    `yaml:"schedulingProfiles"`
	DataLayer          yaml.Node       `yaml:"dataLayer"`
	FlowControl        *eppFlowControl `yaml:"flowControl"`
	RequestHandler     yaml.Node       `yaml:"requestHandler"`
	SaturationDetector *eppPluginRef   `yaml:"saturationDetector"` // v0.10.0 only
	Parser             yaml.Node       `yaml:"parser"`             // v0.10.0 only
}

type eppPluginSpec struct {
	Type       string    `yaml:"type"`
	Name       string    `yaml:"name"`
	Parameters yaml.Node `yaml:"parameters"`
}

type eppProfile struct {
	Name    string             `yaml:"name"`
	Plugins []eppProfilePlugin `yaml:"plugins"`
}

type eppProfilePlugin struct {
	PluginRef string   `yaml:"pluginRef"`
	Weight    *float64 `yaml:"weight"`
}

type eppPluginRef struct {
	PluginRef string `yaml:"pluginRef"`
}

type eppFlowControl struct {
	MaxBytes                     interface{}       `yaml:"maxBytes"`
	MaxRequests                  interface{}       `yaml:"maxRequests"`
	DefaultRequestTTL            *string           `yaml:"defaultRequestTTL"`
	NoEndpointRequestTTL         *string           `yaml:"noEndpointRequestTTL"` // main only
	DefaultPriorityBand          *eppPriorityBand  `yaml:"defaultPriorityBand"`
	DefaultNegativePriorityBand  *eppPriorityBand  `yaml:"defaultNegativePriorityBand"`
	PriorityBands                []eppPriorityBand `yaml:"priorityBands"`
	UsageLimitPolicyPluginRef    string            `yaml:"usageLimitPolicyPluginRef"`
	BandSelectionPolicyPluginRef string            `yaml:"bandSelectionPolicyPluginRef"` // main only
	SaturationDetector           *eppPluginRef     `yaml:"saturationDetector"`
	EnableEviction               bool              `yaml:"enableEviction"`
}

type eppPriorityBand struct {
	Priority          int         `yaml:"priority"`
	MaxBytes          interface{} `yaml:"maxBytes"`
	MaxRequests       interface{} `yaml:"maxRequests"`
	FairnessPolicyRef string      `yaml:"fairnessPolicyRef"`
	OrderingPolicyRef string      `yaml:"orderingPolicyRef"`
	DefaultRequestTTL *string     `yaml:"defaultRequestTTL"` // main only
}

// eppInertTypes are plugin types with no simulation meaning: tokenization (BLIS
// requests carry token IDs), endpoint discovery and metrics plumbing, request
// parsing and header reporting. Accepted and recorded as inert.
var eppInertTypes = map[string]string{
	"token-producer":                    "tokenization: BLIS requests carry token IDs",
	"tokenizer":                         "tokenization (deprecated alias of token-producer)",
	"endpoint-notification-source":      "endpoint lifecycle plumbing",
	"k8s-notification-source":           "Kubernetes notification plumbing",
	"metrics-data-source":               "metrics scraping plumbing (BLIS snapshots model the signals)",
	"core-metrics-extractor":            "metrics scraping plumbing (BLIS snapshots model the signals)",
	"models-data-source":                "model discovery plumbing",
	"models-extractor":                  "model discovery plumbing",
	"openai-parser":                     "request parsing",
	"anthropic-parser":                  "request parsing",
	"vllmhttp-parser":                   "request parsing",
	"vllmgrpc-parser":                   "request parsing",
	"passthrough-parser":                "request parsing",
	"vertexai-parser":                   "request parsing",
	"request-attribute-reporter":        "response header reporting",
	"disaggregatedset-rollout-screener": "rollout revision pinning: the simulated deployment has one revision",
	// Flow-control policies: FCFS ordering, global-strict fairness and the static
	// ceiling are BLIS's gateway-queue defaults; round-robin fairness takes effect
	// through a priority band's fairnessPolicyRef (mapped below).
	"fcfs-ordering-policy":          "flow-control policy (BLIS FIFO dispatch)",
	"global-strict-fairness-policy": "flow-control policy (BLIS global-strict fairness)",
	"round-robin-fairness-policy":   "flow-control policy (BLIS round-robin fairness when a band references it)",
	"static-usage-limit-policy":     "flow-control default (BLIS ceiling 1.0)",
}

// eppSimpleScorers maps parameterless llm-d scorers onto BLIS scorers whose
// semantics already match (#1930 table).
var eppSimpleScorers = map[string]string{
	"queue-scorer":                 "queue-depth",
	"kv-cache-utilization-scorer":  "kv-utilization",
	"active-request-scorer":        "active-requests",
	"running-requests-size-scorer": "running-requests",
	"load-aware-scorer":            "load-aware",
	"lora-affinity-scorer":         "lora-affinity",
}

// eppKnownFeatureGates are the gates llm-d registers (main:cmd/epp/runner/runner.go).
var eppKnownFeatureGates = map[string]bool{"flowControl": true, "haPopulateNonLeaderDatastore": true}

// EPPOption is a BLIS-side ingestion option. Options are BLIS deviations from the
// verbatim config and are recorded in the epp block's deviations.
type EPPOption func(*eppOptions)

type eppOptions struct {
	cpuTierView string
}

// WithCPUTierView makes the named approx-prefix-cache-producer report each
// instance's CPU-offload-tier-resident prefix (the tier's actual contents, delayed
// by --cache-signal-delay) instead of llm-d's router-side LRU estimate (#1912).
// A BLIS opt-in: llm-d itself only has the LRU estimate.
func WithCPUTierView(producer string) EPPOption {
	return func(o *eppOptions) { o.cpuTierView = producer }
}

// LoadEndpointPickerConfig reads and ingests an EndpointPickerConfig file under
// the given llm-d version. engineBlockSize is the engine's KV block size (scenario).
func LoadEndpointPickerConfig(path string, engineBlockSize int, v EPPVersion, opts ...EPPOption) (*EPPIngest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("reading EndpointPickerConfig: %w", err)
	}
	return ParseEndpointPickerConfig(data, engineBlockSize, v, opts...)
}

// ParseEndpointPickerConfig ingests EndpointPickerConfig bytes (YAML, comments allowed).
func ParseEndpointPickerConfig(data []byte, engineBlockSize int, v EPPVersion, opts ...EPPOption) (*EPPIngest, error) {
	var o eppOptions
	for _, opt := range opts {
		opt(&o)
	}
	if v == "" {
		v = EPPVersionMain
	}
	if v != EPPVersionMain && v != EPPVersionV0_10_0 {
		return nil, fmt.Errorf("unknown llm-d EPP version %q", v)
	}
	var f eppFile
	dec := yaml.NewDecoder(bytes.NewReader(data))
	dec.KnownFields(true)
	if err := dec.Decode(&f); err != nil {
		return nil, fmt.Errorf("parsing EndpointPickerConfig: %w", err)
	}
	sum := sha256.Sum256(data)
	b := &eppBuilder{
		f:               &f,
		v:               v,
		engineBlockSize: engineBlockSize,
		byName:          make(map[string]*eppPluginSpec),
		detectors:       make(map[string]ConcurrencyDetectorConfig),
		filters:         make(map[string]FilterConfig),
		scorers:         make(map[string]ScorerConfig),
		roles:           make(map[string]string),
		deciders:        make(map[string]*eppDecider),
		staticThreshold: make(map[string]float64),
		out: &EPPIngest{Version: v, Provenance: EPPProvenance{
			ConfigSHA256:      hex.EncodeToString(sum[:]),
			LLMDRouterVersion: string(v),
		}},
		pipeline: &RoutingPipelineConfig{Version: v},
		opts:     o,
	}
	if err := b.build(); err != nil {
		return nil, err
	}
	return b.out, nil
}

type eppBuilder struct {
	f               *eppFile
	v               EPPVersion
	engineBlockSize int
	byName          map[string]*eppPluginSpec
	out             *EPPIngest
	pipeline        *RoutingPipelineConfig
	detectors       map[string]ConcurrencyDetectorConfig
	filters         map[string]FilterConfig
	scorers         map[string]ScorerConfig
	roles           map[string]string
	disagg          *eppDisaggParams
	deciders        map[string]*eppDecider
	opts            eppOptions
	tokenProducer   bool
	staticThreshold map[string]float64
}

type eppDisaggParams struct {
	name       string
	stageOrder string
	decode     string
	prefill    string
	encode     string
	decider    string
}

type eppDecider struct {
	typ    string
	prefix *LLMDPrefixPDDeciderConfig
}

func (b *eppBuilder) note(format string, args ...interface{}) {
	b.out.Provenance.Notes = append(b.out.Provenance.Notes, fmt.Sprintf(format, args...))
}

func (b *eppBuilder) inert(what string) {
	b.out.Provenance.Inert = append(b.out.Provenance.Inert, what)
}

func (b *eppBuilder) record(r EPPPluginRecord) {
	b.out.Provenance.Plugins = append(b.out.Provenance.Plugins, r)
}

func (b *eppBuilder) build() error {
	f := b.f
	if f.Kind != "EndpointPickerConfig" {
		return fmt.Errorf("EndpointPickerConfig: kind must be EndpointPickerConfig, got %q", f.Kind)
	}
	switch {
	case f.APIVersion == "llm-d.ai/v1alpha1":
	case f.APIVersion == "llm-d.ai/v1" && b.v == EPPVersionMain:
	default:
		return fmt.Errorf("EndpointPickerConfig: apiVersion %q is not served by llm-d-router %s", f.APIVersion, b.v)
	}
	if b.v == EPPVersionMain && (f.SaturationDetector != nil || !isEmptyNode(f.Parser)) {
		return fmt.Errorf("EndpointPickerConfig: top-level saturationDetector/parser were removed after v0.10.0; use flowControl.saturationDetector")
	}
	if f.SaturationDetector != nil {
		return fmt.Errorf("EndpointPickerConfig: top-level saturationDetector is not supported; use flowControl.saturationDetector")
	}
	flowGate := false
	for _, g := range f.FeatureGates {
		if !eppKnownFeatureGates[g] {
			return fmt.Errorf("EndpointPickerConfig: unknown feature gate %q", g)
		}
		if g == "flowControl" {
			flowGate = true
		} else {
			b.inert("featureGate " + g)
		}
	}
	if !isEmptyNode(f.DataLayer) {
		b.inert("dataLayer (endpoint data plumbing)")
	}
	if !isEmptyNode(f.RequestHandler) || !isEmptyNode(f.Parser) {
		b.inert("requestHandler/parser (request parsing)")
	}

	for i := range f.Plugins {
		p := &f.Plugins[i]
		if p.Type == "" {
			return fmt.Errorf("EndpointPickerConfig: plugins[%d] has no type", i)
		}
		if p.Name == "" {
			p.Name = p.Type // loader/defaults.go applyStaticDefaults
		}
		if _, dup := b.byName[p.Name]; dup {
			return fmt.Errorf("EndpointPickerConfig: duplicate plugin name %q", p.Name)
		}
		b.byName[p.Name] = p
	}
	for i := range f.Plugins {
		p := &f.Plugins[i]
		role, blis, err := b.ingestPlugin(p)
		if err != nil {
			return err
		}
		b.roles[p.Name] = role
		b.record(EPPPluginRecord{Type: p.Type, Name: p.Name, Role: role, BLIS: blis})
	}

	if n := b.opts.cpuTierView; n != "" {
		if p, ok := b.byName[n]; !ok || p.Type != LLMDApproxPrefixProducer {
			return fmt.Errorf("--epp-cpu-tier-view %q names no approx-prefix-cache-producer in the config", n)
		}
	}
	if !b.tokenProducer {
		b.note("no token-producer: llm-d's default estimate backend hashes byte-packed pseudo-tokens; BLIS uses the request's token IDs (token counts and block boundaries approximate llm-d's)")
	}

	var handlers []string
	for name, role := range b.roles {
		if role == "profile-handler" {
			handlers = append(handlers, name)
		}
	}
	sort.Strings(handlers)
	if len(handlers) > 1 {
		return fmt.Errorf("EndpointPickerConfig: more than one profile handler (%v)", handlers)
	}
	if b.disagg != nil {
		if err := b.buildDisaggregated(); err != nil {
			return err
		}
	} else {
		if len(f.SchedulingProfiles) != 1 {
			return fmt.Errorf("EndpointPickerConfig: %d schedulingProfiles need a disagg-profile-handler; a single-profile config has exactly one", len(f.SchedulingProfiles))
		}
		b.out.Provenance.ProfileHandler = "single-profile-handler"
		rec, scorers, filters, err := b.buildProfile(f.SchedulingProfiles[0], "")
		if err != nil {
			return err
		}
		b.out.Scorers = scorers
		b.pipeline.Filters = filters
		b.out.Provenance.Profile = rec
	}

	all := append(append(append([]ScorerConfig{}, b.out.Scorers...), b.out.PrefillScorers...), b.out.DecodeScorers...)
	if d := b.out.PDPrefixBased; d != nil {
		// The decider consumes PrefixCacheMatchInfo: route it through the
		// default-producer rule like a prefix-cache scorer.
		all = append(all, ScorerConfig{Name: "prefix-cache", Params: ScorerParams{PrefixMatchInfoProducerName: d.PrefixMatchInfoProducerName}})
	}
	resolved, err := ResolveRoutingPipeline(all, b.pipeline)
	if err != nil {
		return fmt.Errorf("EndpointPickerConfig: %w", err)
	}
	if err := b.resolveFlowControl(flowGate, resolved); err != nil {
		return err
	}
	for _, pc := range resolved.Producers[len(b.pipeline.Producers):] {
		b.record(EPPPluginRecord{Type: pc.Type, Name: pc.Name, Role: "producer", BLIS: "router-side approximate prefix index", AutoCreated: true})
	}
	for _, lc := range resolved.InFlightLoads[len(b.pipeline.InFlightLoads):] {
		b.record(EPPPluginRecord{Type: LLMDInFlightLoadProducer, Name: lc.Name, Role: "producer", BLIS: "router-side in-flight token tracker", AutoCreated: true})
	}
	b.out.Pipeline = resolved
	return nil
}

// buildProfile maps one scheduling profile. role is "" (single profile),
// "prefill" or "decode": a P/D profile must select its pool with the matching
// role filter, which BLIS implements as pool membership.
func (b *eppBuilder) buildProfile(prof eppProfile, role string) (EPPProfileRecord, []ScorerConfig, []FilterConfig, error) {
	rec := EPPProfileRecord{Name: prof.Name, Filters: []string{}, Scorers: []EPPScorerRecord{}, Picker: "max-score-picker"}
	var scorers []ScorerConfig
	var filters []FilterConfig
	pickers, roleFilters := 0, 0
	for i, ref := range prof.Plugins {
		spec, ok := b.byName[ref.PluginRef]
		if !ok {
			return rec, nil, nil, fmt.Errorf("schedulingProfile %q: pluginRef %q names no plugin", prof.Name, ref.PluginRef)
		}
		switch b.roles[ref.PluginRef] {
		case "role-filter":
			if role == "" || spec.Type != role+"-filter" {
				return rec, nil, nil, fmt.Errorf("schedulingProfile %q: %s (%s) is not supported here: role filters are modeled as the P/D pool of the matching profile", prof.Name, ref.PluginRef, spec.Type)
			}
			if i != 0 {
				return rec, nil, nil, fmt.Errorf("schedulingProfile %q: role filter %q must be the profile's first plugin (BLIS applies pool membership before other filters)", prof.Name, ref.PluginRef)
			}
			roleFilters++
			rec.Filters = append(rec.Filters, ref.PluginRef)
		case "filter":
			if ref.Weight != nil {
				return rec, nil, nil, fmt.Errorf("schedulingProfile %q: filter %q must not carry a weight", prof.Name, ref.PluginRef)
			}
			fc := b.filters[ref.PluginRef]
			if role == "prefill" && fc.Type == LLMDConcurrencyDetector && b.v == EPPVersionV0_10_0 {
				return rec, nil, nil, fmt.Errorf("schedulingProfile %q: a concurrency-detector in the prefill profile is not supported under llm-d %s (an empty prefill result falls back to decode-only there, which BLIS does not model)", prof.Name, b.v)
			}
			filters = append(filters, fc)
			rec.Filters = append(rec.Filters, ref.PluginRef)
		case "scorer":
			w := 1.0 // DefaultScorerWeight (loader/defaults.go)
			if ref.Weight != nil {
				w = *ref.Weight
			}
			if w <= 0 || math.IsNaN(w) || math.IsInf(w, 0) {
				return rec, nil, nil, fmt.Errorf("schedulingProfile %q: scorer %q weight must be a finite positive number, got %v (zero-weight scorers are not supported by the simulator)", prof.Name, ref.PluginRef, w)
			}
			sc := b.scorers[ref.PluginRef]
			sc.Weight = w
			scorers = append(scorers, sc)
			rec.Scorers = append(rec.Scorers, EPPScorerRecord{Ref: ref.PluginRef, Type: spec.Type, BLIS: sc.Name, Weight: w})
		case "picker":
			pickers++
			rec.Picker = ref.PluginRef
		default:
			return rec, nil, nil, fmt.Errorf("schedulingProfile %q: plugin %q (type %q) is not a filter, scorer or picker", prof.Name, ref.PluginRef, spec.Type)
		}
	}
	if pickers > 1 {
		return rec, nil, nil, fmt.Errorf("schedulingProfile %q: more than one picker", prof.Name)
	}
	if role != "" && roleFilters != 1 {
		return rec, nil, nil, fmt.Errorf("schedulingProfile %q: a %s profile must start with %s-filter (BLIS routes it over the %s pool)", prof.Name, role, role, role)
	}
	if len(scorers) == 0 {
		return rec, nil, nil, fmt.Errorf("schedulingProfile %q: a profile without scorers is not supported by the simulator", prof.Name)
	}
	return rec, scorers, filters, nil
}

// buildDisaggregated maps a disagg-profile-handler config onto BLIS's decode-first
// P/D routing (main:scheduling/profilehandler/disagg/disagg_profile_handler.go:371-428).
func (b *eppBuilder) buildDisaggregated() error {
	d := b.disagg
	if d.stageOrder != "decode-first" {
		return fmt.Errorf("plugin %q: stageOrder %s is not supported (BLIS routes decode first)", d.name, d.stageOrder)
	}
	if d.encode != "" {
		return fmt.Errorf("plugin %q: deciders.encode (E/P/D) is not supported by the simulator's EPP ingestion", d.name)
	}
	byProfile := make(map[string]eppProfile, len(b.f.SchedulingProfiles))
	for _, p := range b.f.SchedulingProfiles {
		if p.Name != d.decode && p.Name != d.prefill {
			return fmt.Errorf("schedulingProfile %q: not the handler's decode (%q) or prefill (%q) profile", p.Name, d.decode, d.prefill)
		}
		byProfile[p.Name] = p
	}
	decodeProf, ok := byProfile[d.decode]
	if !ok {
		return fmt.Errorf("plugin %q: decode profile %q is missing", d.name, d.decode)
	}
	b.out.Disaggregated = true
	b.out.Provenance.ProfileHandler = "disagg-profile-handler (decode-first)"
	rec, scorers, filters, err := b.buildProfile(decodeProf, "decode")
	if err != nil {
		return err
	}
	b.out.DecodeScorers, b.out.Scorers = scorers, scorers
	b.pipeline.Filters = filters
	b.out.Provenance.Profile = rec
	b.out.Provenance.Profiles = append(b.out.Provenance.Profiles, rec)

	b.out.PDDecider = "never" // no deciders.prefill: P/D disaggregation disabled
	if d.decider != "" {
		dec, ok := b.deciders[d.decider]
		if !ok {
			return fmt.Errorf("plugin %q: deciders.prefill %q names no decider plugin", d.name, d.decider)
		}
		switch dec.typ {
		case "always-disagg-pd-decider":
			b.out.PDDecider = "always"
		case "prefix-based-pd-decider":
			b.out.PDDecider = "prefix-based"
			b.out.PDPrefixBased = dec.prefix
		}
	}
	b.out.Provenance.PDDecider = b.out.PDDecider
	prefillProf, ok := byProfile[d.prefill]
	if !ok {
		if d.decider != "" {
			return fmt.Errorf("plugin %q: prefill profile %q is missing", d.name, d.prefill)
		}
		return nil
	}
	rec, scorers, filters, err = b.buildProfile(prefillProf, "prefill")
	if err != nil {
		return err
	}
	b.out.PrefillScorers = scorers
	b.pipeline.PrefillFilters = filters
	b.out.Provenance.Profiles = append(b.out.Provenance.Profiles, rec)
	return nil
}

func (b *eppBuilder) resolveFlowControl(gate bool, resolved *RoutingPipelineConfig) error {
	fc := b.f.FlowControl
	if !gate {
		if fc != nil {
			b.inert("flowControl block (featureGate flowControl is not enabled; llm-d runs no flow controller)")
		}
		return nil
	}
	if fc == nil {
		fc = &eppFlowControl{}
	}
	if b.v == EPPVersionV0_10_0 && (fc.NoEndpointRequestTTL != nil || fc.BandSelectionPolicyPluginRef != "") {
		return fmt.Errorf("flowControl: noEndpointRequestTTL/bandSelectionPolicyPluginRef do not exist in llm-d %s", b.v)
	}
	if fc.BandSelectionPolicyPluginRef != "" {
		return fmt.Errorf("flowControl: bandSelectionPolicyPluginRef is not supported by the simulator")
	}
	if fc.NoEndpointRequestTTL != nil {
		b.inert("flowControl.noEndpointRequestTTL (the simulated pool always has endpoints)")
	}
	if fc.DefaultNegativePriorityBand != nil {
		return fmt.Errorf("flowControl: defaultNegativePriorityBand is not supported by the simulator's EPP ingestion")
	}
	// The usage-limit policy in force: usageLimitPolicyPluginRef, else the plugin
	// named static-usage-limit-policy (DefaultUsageLimitPolicyRef,
	// pkg/epp/flowcontrol/registry/config.go), else the 1.0 default.
	ceiling := 1.0
	ref0 := fc.UsageLimitPolicyPluginRef
	if ref0 != "" {
		p, ok := b.byName[ref0]
		if !ok || p.Type != "static-usage-limit-policy" {
			return fmt.Errorf("flowControl: usageLimitPolicyPluginRef %q is not supported (only static-usage-limit-policy)", ref0)
		}
	} else if p, ok := b.byName["static-usage-limit-policy"]; ok && p.Type == "static-usage-limit-policy" {
		ref0 = p.Name
	}
	if t, ok := b.staticThreshold[ref0]; ok {
		ceiling = t
	}
	if fc.EnableEviction {
		return fmt.Errorf("flowControl: enableEviction true is not supported by the simulator (llm-d's in-flight evictor is not modeled)")
	}
	ref := ""
	if fc.SaturationDetector != nil {
		ref = fc.SaturationDetector.PluginRef
	}
	if ref == "" {
		// loader/defaults.go ensureSaturationDetector defaults to utilization-detector.
		return fmt.Errorf("flowControl: the default saturation detector (utilization-detector) is not supported by the simulator's EPP ingestion; set flowControl.saturationDetector.pluginRef to a concurrency-detector")
	}
	det, ok := b.detectors[ref]
	if !ok {
		if p, exists := b.byName[ref]; exists {
			return fmt.Errorf("flowControl: saturationDetector %q has type %q, not supported (only concurrency-detector)", ref, p.Type)
		}
		return fmt.Errorf("flowControl: saturationDetector pluginRef %q names no plugin", ref)
	}
	found := false
	for _, l := range resolved.InFlightLoads {
		if l.Name == InFlightProducerName(det.InFlightLoadProducerName) {
			found = true
		}
	}
	if !found {
		if det.InFlightLoadProducerName != "" && det.InFlightLoadProducerName != LLMDInFlightLoadProducer {
			return fmt.Errorf("flowControl: saturationDetector %q: inFlightLoadProducerName %q names no inflight-load-producer", ref, det.InFlightLoadProducerName)
		}
		resolved.InFlightLoads = append(resolved.InFlightLoads, DefaultInFlightLoadConfig(LLMDInFlightLoadProducer, b.v))
	}
	maxReq, err := parseQuantity(fc.MaxRequests, "flowControl.maxRequests")
	if err != nil {
		return err
	}
	maxBytes, err := parseQuantity(fc.MaxBytes, "flowControl.maxBytes")
	if err != nil {
		return err
	}
	ttl, err := parseTTL(fc.DefaultRequestTTL, 60*time.Second, "flowControl.defaultRequestTTL")
	if err != nil {
		return err
	}
	if maxReq > math.MaxInt32 {
		return fmt.Errorf("flowControl.maxRequests too large: %d", maxReq)
	}
	// Per-band defaults: defaultPriorityBandMaxRequests 5000 and MaxBytes 1e9
	// (pkg/epp/flowcontrol/registry/config.go:388-399). defaultPriorityBand overrides them
	// only for the bands provisioned from that template, not for a listed priorityBands
	// entry, whose unset caps take the constants (v0.10.0:pkg/epp/config/loader/flowcontrol.go:139-185).
	const constBandReq, constBandBytes = int64(5000), int64(1_000_000_000)
	bandReq, bandBytes := constBandReq, constBandBytes
	if db := fc.DefaultPriorityBand; db != nil {
		if db.FairnessPolicyRef != "" || db.OrderingPolicyRef != "" || db.DefaultRequestTTL != nil {
			return fmt.Errorf("flowControl.defaultPriorityBand: only maxRequests/maxBytes are supported")
		}
		v, err := parseQuantity(db.MaxRequests, "defaultPriorityBand.maxRequests")
		if err != nil {
			return err
		}
		if v > 0 {
			bandReq = v
		}
		if v, err = parseQuantity(db.MaxBytes, "defaultPriorityBand.maxBytes"); err != nil {
			return err
		}
		if v > 0 {
			bandBytes = v
		}
	}
	var bands []EPPPriorityBand
	seen := map[int]bool{}
	for _, pb := range fc.PriorityBands {
		if seen[pb.Priority] {
			return fmt.Errorf("flowControl.priorityBands: duplicate priority %d", pb.Priority)
		}
		seen[pb.Priority] = true
		if pb.DefaultRequestTTL != nil {
			if b.v == EPPVersionV0_10_0 {
				return fmt.Errorf("flowControl.priorityBands[%d].defaultRequestTTL does not exist in llm-d %s", pb.Priority, b.v)
			}
			return fmt.Errorf("flowControl.priorityBands[%d]: per-band defaultRequestTTL is not supported by the simulator", pb.Priority)
		}
		band := EPPPriorityBand{Priority: pb.Priority, MaxRequests: int(constBandReq), MaxBytes: constBandBytes, Fairness: "global-strict"}
		v, err := parseQuantity(pb.MaxRequests, "priorityBands.maxRequests")
		if err != nil {
			return err
		}
		if v > 0 {
			band.MaxRequests = int(v)
		}
		if v, err = parseQuantity(pb.MaxBytes, "priorityBands.maxBytes"); err != nil {
			return err
		}
		if v > 0 {
			band.MaxBytes = v
		}
		if pb.OrderingPolicyRef != "" {
			if p, ok := b.byName[pb.OrderingPolicyRef]; !ok || p.Type != "fcfs-ordering-policy" {
				return fmt.Errorf("flowControl.priorityBands[%d]: orderingPolicyRef %q is not supported (only fcfs-ordering-policy)", pb.Priority, pb.OrderingPolicyRef)
			}
		}
		if pb.FairnessPolicyRef != "" {
			p, ok := b.byName[pb.FairnessPolicyRef]
			switch {
			case ok && p.Type == "round-robin-fairness-policy":
				band.Fairness = "round-robin"
			case ok && p.Type == "global-strict-fairness-policy":
			default:
				return fmt.Errorf("flowControl.priorityBands[%d]: fairnessPolicyRef %q is not supported", pb.Priority, pb.FairnessPolicyRef)
			}
		}
		bands = append(bands, band)
	}
	b.out.FlowControl = &EPPFlowControl{
		Detector:           det,
		MaxRequests:        int(maxReq),
		MaxBytes:           maxBytes,
		PerBandMaxRequests: int(bandReq),
		PerBandMaxBytes:    bandBytes,
		Bands:              bands,
		RequestTTLUs:       ttl.Microseconds(),
		UniformCeiling:     ceiling,
	}
	b.out.Provenance.FlowControl = &EPPFlowControlRecord{
		SaturationDetector: ref,
		MaxRequests:        int(maxReq),
		MaxBytes:           maxBytes,
		PerBandMaxRequests: int(bandReq),
		PerBandMaxBytes:    bandBytes,
		Bands:              bands,
		DefaultRequestTTLs: ttl.Seconds(),
	}
	b.note("flowControl byte limits count estimated request bytes (%d bytes per prompt token)", gatewayBytesPerTokenEstimate)
	if len(bands) > 0 {
		b.note("flowControl priority bands match a request's BLIS priority (SLO class → priority); other priorities use the default band")
	}
	return nil
}

// gatewayBytesPerTokenEstimate is the request-body size estimate the gateway queue
// uses for byte limits (llm-d's averageCharactersPerToken, approximateprefix/plugin.go).
const gatewayBytesPerTokenEstimate = 4

func parseTTL(s *string, def time.Duration, field string) (time.Duration, error) {
	if s == nil {
		return def, nil
	}
	d, err := time.ParseDuration(*s)
	if err != nil || d < 0 {
		return 0, fmt.Errorf("%s: invalid duration %q", field, *s)
	}
	return d, nil
}

// ingestPlugin maps one plugin and returns its role and the BLIS implementation.
func (b *eppBuilder) ingestPlugin(p *eppPluginSpec) (string, string, error) {
	if why, ok := eppInertTypes[p.Type]; ok {
		if err := b.checkInertParams(p); err != nil {
			return "", "", err
		}
		b.inert(fmt.Sprintf("%s %q (%s)", p.Type, p.Name, why))
		return "inert", "inert: " + why, nil
	}
	if blis, ok := eppSimpleScorers[p.Type]; ok {
		if !isEmptyNode(p.Parameters) {
			return "", "", fmt.Errorf("plugin %q (%s): parameters are not supported by the simulator (BLIS %s models llm-d's defaults only)", p.Name, p.Type, blis)
		}
		b.scorers[p.Name] = ScorerConfig{Name: blis, Instance: p.Name}
		return "scorer", blis, nil
	}
	switch p.Type {
	case "prefill-filter", "decode-filter":
		if !isEmptyNode(p.Parameters) {
			return "", "", fmt.Errorf("plugin %q (%s): parameters are not supported", p.Name, p.Type)
		}
		return "role-filter", "P/D pool membership", nil
	case LLMDApproxPrefixProducer:
		return b.ingestApprox(p)
	case LLMDPrecisePrefixProducer:
		return b.ingestPrecise(p)
	case LLMDInFlightLoadProducer:
		return b.ingestInFlight(p)
	case LLMDConcurrencyDetector:
		return b.ingestConcurrency(p)
	case LLMDPrefixAffinityFilter:
		return b.ingestAffinity(p)
	case "token-load-scorer":
		var raw struct {
			QueueThresholdTokens     int64  `json:"queueThresholdTokens"`
			InFlightLoadProducerName string `json:"inFlightLoadProducerName"`
		}
		if err := decodeParams(p, &raw); err != nil {
			return "", "", err
		}
		b.scorers[p.Name] = ScorerConfig{Name: "token-load", Instance: p.Name, Params: ScorerParams{
			QueueThresholdTokens: raw.QueueThresholdTokens, InFlightLoadProducerName: raw.InFlightLoadProducerName}}
		return "scorer", "token-load", nil
	case "prefix-cache-scorer":
		raw := struct {
			PrefixMatchInfoProducerName string  `json:"prefixMatchInfoProducerName"`
			MatchLengthWeight           float64 `json:"matchLengthWeight"`
			MatchLengthScaleTokens      *int    `json:"matchLengthScaleTokens"`
		}{}
		if err := decodeParams(p, &raw); err != nil {
			return "", "", err
		}
		scale := prefixScorerDefaultScaleToken
		setInt(&scale, raw.MatchLengthScaleTokens)
		if raw.MatchLengthWeight < 0 || raw.MatchLengthWeight > 1 {
			return "", "", fmt.Errorf("plugin %q: matchLengthWeight must be between 0.0 and 1.0, got %f", p.Name, raw.MatchLengthWeight)
		}
		if raw.MatchLengthWeight > 0 && scale <= 0 {
			return "", "", fmt.Errorf("plugin %q: matchLengthScaleTokens must be greater than 0 when matchLengthWeight is greater than 0", p.Name)
		}
		b.scorers[p.Name] = ScorerConfig{Name: "prefix-cache", Instance: p.Name, Params: ScorerParams{
			PrefixMatchInfoProducerName: raw.PrefixMatchInfoProducerName, MatchLengthWeight: raw.MatchLengthWeight, MatchLengthScaleTokens: scale}}
		return "scorer", "prefix-cache", nil
	case "no-hit-lru-scorer":
		var raw struct {
			PrefixPluginType            string `json:"prefixPluginType"`
			PrefixPluginName            string `json:"prefixPluginName"`
			LRUSize                     int    `json:"lruSize"`
			PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
		}
		if err := decodeParams(p, &raw); err != nil {
			return "", "", err
		}
		if raw.PrefixPluginType != "" || raw.PrefixPluginName != "" {
			return "", "", fmt.Errorf("plugin %q: prefixPluginType/prefixPluginName are deprecated and not supported; use prefixMatchInfoProducerName", p.Name)
		}
		b.scorers[p.Name] = ScorerConfig{Name: "no-hit-lru", Instance: p.Name, Params: ScorerParams{
			PrefixMatchInfoProducerName: raw.PrefixMatchInfoProducerName, ProducerBound: true, LRUSize: raw.LRUSize}}
		return "scorer", "no-hit-lru (producer-bound)", nil
	case "max-score-picker":
		var raw struct {
			MaxNumOfEndpoints int `json:"maxNumOfEndpoints"`
		}
		if err := decodeParams(p, &raw); err != nil {
			return "", "", err
		}
		if raw.MaxNumOfEndpoints > 1 {
			return "", "", fmt.Errorf("plugin %q: maxNumOfEndpoints %d > 1 is not supported by the simulator (one target per request)", p.Name, raw.MaxNumOfEndpoints)
		}
		return "picker", "argmax with seeded uniform tie-break", nil
	case "single-profile-handler":
		if !isEmptyNode(p.Parameters) {
			return "", "", fmt.Errorf("plugin %q (%s): parameters are not supported", p.Name, p.Type)
		}
		return "profile-handler", "single profile", nil
	case "disagg-profile-handler":
		var raw struct {
			StageOrder string `json:"stageOrder"`
			Profiles   struct {
				Decode  string `json:"decode"`
				Prefill string `json:"prefill"`
				Encode  string `json:"encode"`
			} `json:"profiles"`
			Deciders struct {
				Prefill string `json:"prefill"`
				Encode  string `json:"encode"`
			} `json:"deciders"`
		}
		if err := decodeParams(p, &raw); err != nil {
			return "", "", err
		}
		order, err := parseStageOrder(raw.StageOrder)
		if err != nil {
			return "", "", fmt.Errorf("plugin %q: %w", p.Name, err)
		}
		d := &eppDisaggParams{name: p.Name, stageOrder: order, decode: raw.Profiles.Decode, prefill: raw.Profiles.Prefill,
			encode: raw.Deciders.Encode, decider: raw.Deciders.Prefill}
		// DisaggProfileHandlerConfigParser (main:…/disagg_profile_handler.go:172-205).
		if d.decode == "" {
			d.decode = "decode"
		}
		if d.prefill == "" {
			d.prefill = "prefill"
		}
		b.disagg = d
		return "profile-handler", "decode-first P/D routing", nil
	case "always-disagg-pd-decider":
		if !isEmptyNode(p.Parameters) {
			return "", "", fmt.Errorf("plugin %q (%s): parameters are not supported", p.Name, p.Type)
		}
		b.deciders[p.Name] = &eppDecider{typ: p.Type}
		return "decider", "always disaggregate", nil
	case "prefix-based-pd-decider":
		var raw struct {
			NonCachedTokens             int    `json:"nonCachedTokens"`
			PromptTokens                int    `json:"promptTokens"`
			PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName"`
		}
		if err := decodeParams(p, &raw); err != nil {
			return "", "", err
		}
		if raw.NonCachedTokens < 0 || raw.PromptTokens < 0 {
			return "", "", fmt.Errorf("plugin %q: nonCachedTokens and promptTokens must be >= 0", p.Name)
		}
		b.deciders[p.Name] = &eppDecider{typ: p.Type, prefix: &LLMDPrefixPDDeciderConfig{
			NonCachedTokens: raw.NonCachedTokens, PromptTokens: raw.PromptTokens, PrefixMatchInfoProducerName: raw.PrefixMatchInfoProducerName}}
		return "decider", "prefix-based P/D decider", nil
	}
	return "", "", fmt.Errorf("plugin %q: type %q is not supported by the simulator", p.Name, p.Type)
}

// eppInertParams lists, per inert plugin type, the parameters with no simulation
// meaning (endpoints, credentials, metric names, headers, rollout scoping). Any other
// parameter is refused; parameters with simulation meaning are mapped in
// checkInertParams.
var eppInertParams = map[string][]string{
	"token-producer":                    {"modelName", "vllm", "estimate"},
	"tokenizer":                         {"modelName", "vllm", "estimate"},
	"metrics-data-source":               {"scheme", "path", "port", "insecureSkipVerify", "caCertPath", "clientCertPath", "clientKeyPath", "families", "interval"},
	"core-metrics-extractor":            {"engineConfigs", "defaultEngine", "engineLabelKey"},
	"k8s-notification-source":           {"group", "version", "kind"},
	"request-attribute-reporter":        {"attributes"},
	"disaggregatedset-rollout-screener": {"scope", "revisionGating"},
	"static-usage-limit-policy":         {"threshold"},
}

// checkInertParams refuses parameters outside the type's allowlist and maps the
// ones that matter: metrics-data-source interval -> snapshot refresh interval,
// static-usage-limit-policy threshold -> the uniform band ceiling, token-producer
// backend -> a tokenization note.
func (b *eppBuilder) checkInertParams(p *eppPluginSpec) error {
	if isEmptyNode(p.Parameters) {
		if p.Type == "token-producer" || p.Type == "tokenizer" {
			b.tokenProducer = true
			b.note("%q: no backend configured, llm-d's estimate backend hashes byte-packed pseudo-tokens; BLIS uses the request's token IDs", p.Name)
		}
		return nil
	}
	var raw map[string]json.RawMessage
	if err := decodeParams(p, &raw); err != nil {
		return err
	}
	allowed := map[string]bool{}
	for _, k := range eppInertParams[p.Type] {
		allowed[k] = true
	}
	keys := make([]string, 0, len(raw))
	for k := range raw {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		if !allowed[k] {
			return fmt.Errorf("plugin %q (%s): parameter %q is not supported by the simulator", p.Name, p.Type, k)
		}
	}
	switch p.Type {
	case "token-producer", "tokenizer":
		b.tokenProducer = true
		if _, ok := raw["vllm"]; !ok {
			b.note("%q: estimate backend — llm-d hashes byte-packed pseudo-tokens; BLIS uses the request's token IDs", p.Name)
		}
	case "metrics-data-source":
		if iv, ok := raw["interval"]; ok {
			var str string
			if err := json.Unmarshal(iv, &str); err != nil {
				return fmt.Errorf("plugin %q: interval: %w", p.Name, err)
			}
			d, err := time.ParseDuration(str)
			if err != nil || d < 0 {
				return fmt.Errorf("plugin %q: invalid interval %q", p.Name, str)
			}
			// Rounded to a multiple of the EPP's --refresh-metrics-interval base tick
			// (default 50ms; main:…/datalayer/source/metrics/datasource.go:66-70).
			const tick = 50 * time.Millisecond
			n := (d + tick/2) / tick
			if n < 1 {
				n = 1
			}
			b.out.MetricsRefreshIntervalUs = (n * tick).Microseconds()
		}
	case "static-usage-limit-policy":
		var t float64
		if err := json.Unmarshal(raw["threshold"], &t); err != nil || t <= 0 || t > 1 || math.IsNaN(t) {
			return fmt.Errorf("plugin %q: threshold must be in (0, 1], got %s", p.Name, raw["threshold"])
		}
		b.staticThreshold[p.Name] = t
	}
	return nil
}

// parseStageOrder mirrors ParseStageOrder (main:…/disagg_profile_handler.go:69-78).
func parseStageOrder(s string) (string, error) {
	switch strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(s, "-", ""), "_", "")) {
	case "", "decodefirst", "decode":
		return "decode-first", nil
	case "prefillfirst", "prefill":
		return "prefill-first", nil
	}
	return "", fmt.Errorf("invalid stage order %q, must be \"decode-first\" or \"prefill-first\"", s)
}

func (b *eppBuilder) ingestApprox(p *eppPluginSpec) (string, string, error) {
	cfg := DefaultApproxPrefixProducerConfig(p.Name, b.v)
	var raw struct {
		AutoTune               *bool `json:"autoTune"`
		BlockSizeTokens        *int  `json:"blockSizeTokens"`
		BlockSize              *int  `json:"blockSize"`
		MaxPrefixBlocksToMatch *int  `json:"maxPrefixBlocksToMatch"`
		MaxPrefixTokensToMatch *int  `json:"maxPrefixTokensToMatch"`
		LRUCapacityPerServer   *int  `json:"lruCapacityPerServer"`
	}
	if err := decodeParams(p, &raw); err != nil {
		return "", "", err
	}
	def := cfg.BlockSizeTokens
	setBool(&cfg.AutoTune, raw.AutoTune)
	setInt(&cfg.BlockSizeTokens, raw.BlockSizeTokens)
	setInt(&cfg.MaxPrefixBlocksToMatch, raw.MaxPrefixBlocksToMatch)
	setInt(&cfg.MaxPrefixTokensToMatch, raw.MaxPrefixTokensToMatch)
	setInt(&cfg.LRUCapacityPerServer, raw.LRUCapacityPerServer)
	// Deprecated blockSize maps into blockSizeTokens left at its default
	// (main:…/approximateprefix/plugin.go:421-445).
	if raw.BlockSize != nil && *raw.BlockSize > 0 && cfg.BlockSizeTokens == def {
		cfg.BlockSizeTokens = *raw.BlockSize
	}
	if !cfg.AutoTune && cfg.BlockSizeTokens <= 0 {
		return "", "", fmt.Errorf("plugin %q: blockSizeTokens must be > 0 when autoTune is disabled", p.Name)
	}
	if cfg.MaxPrefixTokensToMatch < 0 {
		return "", "", fmt.Errorf("plugin %q: maxPrefixTokensToMatch must be >= 0", p.Name)
	}
	if b.v == EPPVersionMain && cfg.MaxPrefixBlocksToMatch < 0 {
		return "", "", fmt.Errorf("plugin %q: maxPrefixBlocksToMatch must be >= 0", p.Name)
	}
	// llm-d models every approx producer — including a CPU-tier one such as the
	// tiered guide's cpu-prefix-cache-producer — as a router-side LRU sized by
	// lruCapacityPerServer or, under autoTune, the GPU block count (makeserver,
	// main:…/approximateprefix/plugin.go:361-372). The live CPU-tier view is an
	// explicit BLIS opt-in (WithCPUTierView), recorded as a deviation.
	cfg.CPUTier = b.opts.cpuTierView != "" && p.Name == b.opts.cpuTierView
	b.pipeline.Producers = append(b.pipeline.Producers, cfg)
	blis := fmt.Sprintf("router-side approximate prefix index (block %d tokens)", resolveApproxBlockSize(cfg, b.engineBlockSize))
	if cfg.CPUTier {
		blis = "BLIS opt-in: CPU-offload-tier prefix view (with an offload tier), else " + blis
		b.out.Provenance.Deviations = append(b.out.Provenance.Deviations,
			fmt.Sprintf("%q reports each instance's CPU-offload-tier-resident prefix (--epp-cpu-tier-view) instead of llm-d's router-side LRU", p.Name))
	}
	return "producer", blis, nil
}

func (b *eppBuilder) ingestPrecise(p *eppPluginSpec) (string, string, error) {
	var raw struct {
		TokenProcessorConfig json.RawMessage `json:"tokenProcessorConfig"`
		IndexerConfig        json.RawMessage `json:"indexerConfig"`
		KVEventsConfig       json.RawMessage `json:"kvEventsConfig"`
		SpeculativeIndexing  bool            `json:"speculativeIndexing"`
		SpeculativeTTL       string          `json:"speculativeTTL"`
	}
	if err := decodeParams(p, &raw); err != nil {
		return "", "", err
	}
	bs := 16 // kvblock/token_processor.go defaultBlockSize
	if len(raw.TokenProcessorConfig) > 0 {
		var tp struct {
			BlockSize       int    `json:"blockSize"`
			BlockSizeTokens int    `json:"blockSizeTokens"`
			HashSeed        string `json:"hashSeed"`
		}
		if err := strictJSON(raw.TokenProcessorConfig, &tp); err != nil {
			return "", "", fmt.Errorf("plugin %q: tokenProcessorConfig: %w", p.Name, err)
		}
		switch {
		case tp.BlockSizeTokens > 0:
			bs = tp.BlockSizeTokens
		case tp.BlockSize > 0:
			bs = tp.BlockSize
		}
	}
	if b.engineBlockSize > 0 && bs != b.engineBlockSize {
		return "", "", fmt.Errorf("plugin %q: tokenProcessorConfig.blockSizeTokens %d differs from the engine block size %d; KV-event block keys would never match", p.Name, bs, b.engineBlockSize)
	}
	cfg := PrefixProducerConfig{Name: p.Name, Type: LLMDPrecisePrefixProducer, BlockSizeTokens: bs}
	// Tier weights: DefaultKVCacheBackendConfig (gpu 1.0, cpu 0.8) unless
	// kvCacheBackendConfigs is set (pkg/kvcache/backend.go); a tier absent from the
	// map weighs 1.0 on v0.10.0 and 0.0 on main (pkg/kvcache/prefix_match.go:44-45).
	cfg.TierWeights = map[string]float64{"gpu": 1.0, "cpu": 0.8}
	cfg.UnknownTierWeight = 0.0
	if b.v == EPPVersionV0_10_0 {
		cfg.UnknownTierWeight = 1.0
	}
	if len(raw.IndexerConfig) > 0 {
		var ic struct {
			KVCacheBackendConfigs []struct {
				Name   string  `json:"name"`
				Weight float64 `json:"weight"`
			} `json:"kvCacheBackendConfigs"`
		}
		if err := json.Unmarshal(raw.IndexerConfig, &ic); err != nil {
			return "", "", fmt.Errorf("plugin %q: indexerConfig: %w", p.Name, err)
		}
		if len(ic.KVCacheBackendConfigs) > 0 {
			cfg.TierWeights = map[string]float64{}
			for _, be := range ic.KVCacheBackendConfigs {
				cfg.TierWeights[strings.ToLower(be.Name)] = be.Weight
			}
		}
		b.inert(fmt.Sprintf("%s %q indexerConfig storage sizing (BLIS reads each instance's KV view directly)", p.Type, p.Name))
	}
	if len(raw.KVEventsConfig) > 0 {
		b.inert(fmt.Sprintf("%s %q kvEventsConfig (ZMQ sockets and pod discovery; event delay is --cache-signal-delay)", p.Type, p.Name))
	}
	if raw.SpeculativeIndexing {
		cfg.Speculative = true
		cfg.SpeculativeTTLUs = preciseDefaultSpeculativeTTL
		if raw.SpeculativeTTL != "" {
			d, err := time.ParseDuration(raw.SpeculativeTTL)
			if err != nil || d <= 0 {
				return "", "", fmt.Errorf("plugin %q: invalid speculativeTTL %q", p.Name, raw.SpeculativeTTL)
			}
			cfg.SpeculativeTTLUs = d.Microseconds()
		}
	}
	b.pipeline.Producers = append(b.pipeline.Producers, cfg)
	b.note("%q maps to each instance's GPU KV view delayed by --cache-signal-delay (KV events) and, with an offload tier, the live CPU-tier view weighted %v", p.Name, weightOf(cfg, "cpu"))
	return "producer", fmt.Sprintf("per-instance KV cache view (block %d tokens)", bs), nil
}

func weightOf(c PrefixProducerConfig, tier string) float64 {
	if w, ok := c.TierWeights[tier]; ok {
		return w
	}
	return c.UnknownTierWeight
}

func (b *eppBuilder) ingestInFlight(p *eppPluginSpec) (string, string, error) {
	cfg := DefaultInFlightLoadConfig(p.Name, b.v)
	var raw struct {
		AddEstimatedOutputTokens    bool     `json:"addEstimatedOutputTokens"`
		OutputRatio                 *float64 `json:"outputRatio"` // v0.10.0 only
		MaxEstimatedOutputTokens    *int64   `json:"maxEstimatedOutputTokens"`
		PrefixMatchInfoProducerName string   `json:"prefixMatchInfoProducerName"`
		SyncCrossReplicaState       *bool    `json:"syncCrossReplicaState"`
	}
	if err := decodeParams(p, &raw); err != nil {
		return "", "", err
	}
	cfg.AddEstimatedOutput = raw.AddEstimatedOutputTokens
	if raw.OutputRatio != nil {
		if b.v != EPPVersionV0_10_0 {
			return "", "", fmt.Errorf("plugin %q: outputRatio was removed after llm-d v0.10.0 (main estimates output by outlen bucket)", p.Name)
		}
		if *raw.OutputRatio < 0 {
			return "", "", fmt.Errorf("plugin %q: outputRatio must be non-negative", p.Name)
		}
		cfg.OutputRatio = *raw.OutputRatio
	}
	if raw.MaxEstimatedOutputTokens != nil {
		if *raw.MaxEstimatedOutputTokens < 0 {
			return "", "", fmt.Errorf("plugin %q: maxEstimatedOutputTokens must be non-negative", p.Name)
		}
		cfg.MaxEstimatedOutput = *raw.MaxEstimatedOutputTokens
	}
	cfg.PrefixMatchInfoProducerName = raw.PrefixMatchInfoProducerName
	if raw.SyncCrossReplicaState != nil {
		b.inert(fmt.Sprintf("%s %q syncCrossReplicaState (one EPP replica is simulated)", p.Type, p.Name))
	}
	if cfg.AddEstimatedOutput && b.v == EPPVersionMain {
		b.note("%q: addEstimatedOutputTokens estimates %d output tokens per request (llm-d main's unknown outlen bucket; BLIS requests carry no bucket header)", p.Name, inflightUnknownOutputEstimate)
	}
	b.pipeline.InFlightLoads = append(b.pipeline.InFlightLoads, cfg)
	return "producer", "router-side in-flight token tracker", nil
}

func (b *eppBuilder) ingestConcurrency(p *eppPluginSpec) (string, string, error) {
	cfg := DefaultConcurrencyDetectorConfig()
	var raw struct {
		MaxConcurrency           *int64   `json:"maxConcurrency"`
		Headroom                 *float64 `json:"headroom"`
		ConcurrencyMode          *string  `json:"concurrencyMode"`
		MaxTokenConcurrency      *int64   `json:"maxTokenConcurrency"`
		InFlightLoadProducerName string   `json:"inFlightLoadProducerName"`
		FailOpen                 *bool    `json:"failOpen"` // main only
	}
	if err := decodeParams(p, &raw); err != nil {
		return "", "", err
	}
	if raw.MaxConcurrency != nil {
		cfg.MaxConcurrency = *raw.MaxConcurrency
	}
	if raw.Headroom != nil {
		cfg.Headroom = *raw.Headroom
	}
	if raw.ConcurrencyMode != nil {
		cfg.Mode = ConcurrencyMode(*raw.ConcurrencyMode)
	}
	if raw.MaxTokenConcurrency != nil {
		cfg.MaxTokenConcurrency = *raw.MaxTokenConcurrency
	}
	cfg.InFlightLoadProducerName = raw.InFlightLoadProducerName
	if b.v == EPPVersionMain {
		cfg.FailOpen = true
		setBool(&cfg.FailOpen, raw.FailOpen)
		cfg.CountIncomingTokens = true
		cfg.DispatchReservations = true
	} else if raw.FailOpen != nil {
		return "", "", fmt.Errorf("plugin %q: failOpen does not exist in llm-d %s", p.Name, b.v)
	}
	if err := cfg.Validate(); err != nil {
		return "", "", fmt.Errorf("plugin %q: invalid concurrency detector configuration: %w", p.Name, err)
	}
	b.detectors[p.Name] = cfg
	b.filters[p.Name] = FilterConfig{Type: LLMDConcurrencyDetector, Name: p.Name, Concurrency: cfg}
	return "filter", "concurrency detector (filter + flow-control saturation)", nil
}

func (b *eppBuilder) ingestAffinity(p *eppPluginSpec) (string, string, error) {
	cfg := DefaultPrefixAffinityFilterConfig(b.v)
	raw := struct {
		AffinityThreshold                 *float64 `json:"affinityThreshold"`
		ExplorationProbability            *float64 `json:"explorationProbability"`
		MaxTTFTPenaltyMs                  *float64 `json:"maxTTFTPenaltyMs"`
		TTFTSource                        string   `json:"ttftSource"`
		PeakPrefillThroughput             *float64 `json:"peakPrefillThroughput"`
		PrefixMatchInfoProducerName       string   `json:"prefixMatchInfoProducerName"`
		LatencyPredictionInfoProducerName string   `json:"latencyPredictionInfoProducerName"`
		InFlightLoadProducerName          string   `json:"inFlightLoadProducerName"`
	}{}
	if err := decodeParams(p, &raw); err != nil {
		return "", "", err
	}
	setFloat(&cfg.AffinityThreshold, raw.AffinityThreshold)
	setFloat(&cfg.ExplorationProbability, raw.ExplorationProbability)
	setFloat(&cfg.MaxTTFTPenaltyMs, raw.MaxTTFTPenaltyMs)
	setFloat(&cfg.PeakPrefillThroughput, raw.PeakPrefillThroughput)
	switch raw.TTFTSource {
	case "", "prefillThroughput":
	case "latencyPredictor":
		return "", "", fmt.Errorf("plugin %q: ttftSource latencyPredictor is not supported by the simulator (no predicted-latency producer)", p.Name)
	default:
		return "", "", fmt.Errorf("plugin %q: ttftSource must be \"latencyPredictor\" or \"prefillThroughput\", got %q", p.Name, raw.TTFTSource)
	}
	cfg.PrefixMatchInfoProducerName = raw.PrefixMatchInfoProducerName
	cfg.InFlightLoadProducerName = raw.InFlightLoadProducerName
	if err := cfg.Validate(b.v); err != nil {
		return "", "", fmt.Errorf("plugin %q: invalid config: %w", p.Name, err)
	}
	b.filters[p.Name] = FilterConfig{Type: LLMDPrefixAffinityFilter, Name: p.Name, Affinity: cfg}
	return "filter", "prefix-cache-affinity filter", nil
}

func isEmptyNode(n yaml.Node) bool {
	return n.Kind == 0 || (n.Kind == yaml.ScalarNode && n.Tag == "!!null")
}

// decodeParams decodes a plugin's parameters strictly (DisallowUnknownFields),
// as llm-d's plugin registry does.
func decodeParams(p *eppPluginSpec, out interface{}) error {
	if isEmptyNode(p.Parameters) {
		return nil
	}
	var generic interface{}
	if err := p.Parameters.Decode(&generic); err != nil {
		return fmt.Errorf("plugin %q: parameters: %w", p.Name, err)
	}
	raw, err := json.Marshal(generic)
	if err != nil {
		return fmt.Errorf("plugin %q: parameters: %w", p.Name, err)
	}
	if err := strictJSON(raw, out); err != nil {
		return fmt.Errorf("plugin %q (%s): parameters: %w", p.Name, p.Type, err)
	}
	return nil
}

func strictJSON(raw []byte, out interface{}) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	return dec.Decode(out)
}

func setBool(dst *bool, v *bool) {
	if v != nil {
		*dst = *v
	}
}

func setInt(dst *int, v *int) {
	if v != nil {
		*dst = *v
	}
}

func setFloat(dst *float64, v *float64) {
	if v != nil {
		*dst = *v
	}
}

// parseQuantity parses a Kubernetes resource.Quantity (plain number, decimal SI
// suffix k/M/G/T/P/E, or binary suffix Ki/Mi/Gi/Ti/Pi/Ei) to an integer, rounding
// up like resource.Quantity.Value. Nil ⇒ 0 (unset).
func parseQuantity(v interface{}, field string) (int64, error) {
	if v == nil {
		return 0, nil
	}
	var s string
	switch t := v.(type) {
	case int:
		s = fmt.Sprint(t)
	case int64:
		s = fmt.Sprint(t)
	case float64:
		s = fmt.Sprint(t)
	case string:
		s = strings.TrimSpace(t)
	default:
		return 0, fmt.Errorf("%s: invalid quantity %v", field, v)
	}
	mult := big.NewRat(1, 1)
	suffixes := []struct {
		s string
		m int64
		e int
	}{
		{"Ki", 1 << 10, 0}, {"Mi", 1 << 20, 0}, {"Gi", 1 << 30, 0}, {"Ti", 1 << 40, 0}, {"Pi", 1 << 50, 0}, {"Ei", 1 << 60, 0},
		{"k", 0, 3}, {"M", 0, 6}, {"G", 0, 9}, {"T", 0, 12}, {"P", 0, 15}, {"E", 0, 18},
	}
	for _, suf := range suffixes {
		if strings.HasSuffix(s, suf.s) {
			s = strings.TrimSuffix(s, suf.s)
			if suf.m > 0 {
				mult = new(big.Rat).SetInt64(suf.m)
			} else {
				mult = new(big.Rat).SetInt(new(big.Int).Exp(big.NewInt(10), big.NewInt(int64(suf.e)), nil))
			}
			break
		}
	}
	num, ok := new(big.Rat).SetString(s)
	if !ok {
		return 0, fmt.Errorf("%s: invalid quantity %v", field, v)
	}
	num.Mul(num, mult)
	if num.Sign() < 0 {
		return 0, fmt.Errorf("%s must be non-negative, got %v", field, v)
	}
	i := new(big.Int).Quo(num.Num(), num.Denom())
	if new(big.Rat).SetInt(i).Cmp(num) < 0 {
		i.Add(i, big.NewInt(1))
	}
	if !i.IsInt64() {
		return 0, fmt.Errorf("%s: quantity %v overflows int64", field, v)
	}
	return i.Int64(), nil
}
