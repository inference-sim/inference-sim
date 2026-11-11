package result

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

var (
	slug       = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)
	lowerHex64 = regexp.MustCompile(`^[0-9a-f]{64}$`)
	releaseTag = regexp.MustCompile(`^v[0-9]+\.[0-9]+\.[0-9]+$`)
)

// Validate performs field-level validation of a Result on its own: its shape, its
// embedded input and frame, its figures against the metric registry, and its fates.
func (r *Result) Validate() *validate.Problems {
	p := &validate.Problems{}
	if r.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, r.Kind)
	}
	if r.SchemaVersion != SchemaVersion {
		p.Field("schema_version", "must be %d, got %d", SchemaVersion, r.SchemaVersion)
	}
	if !slug.MatchString(r.Name) {
		p.Field("name", "must be a lowercase slug, got %q", r.Name)
	}
	r.validateEngine(p)
	r.Data.validate(p)
	r.SelectedEntries.validate(p)
	r.Input.validate(p)
	r.Frame.validate(p)
	r.Revisions.validate(p)
	r.validateRerun(p)
	r.Window.validate(p)
	r.validateMetrics(p)
	r.Requests.validate(p)
	r.validateRows(p)
	if r.Goodput != nil {
		r.Goodput.validate(p)
	}
	if r.Breakdowns != nil {
		r.validateBreakdowns(p)
	}
	for i, o := range r.Overrides {
		at := fmt.Sprintf("overrides[%d]", i)
		if o.Setting == "" {
			p.Field(at+".setting", "required")
		}
		if o.Source == "" {
			p.Field(at+".source", "required: what chose the simulated value")
		}
	}
	r.validateResolutions(p)
	for _, name := range sortedKeys(r.Artifacts) {
		a := r.Artifacts[name]
		at := "artifacts." + name
		if !slug.MatchString(name) {
			p.Field(at, "an artifact role is a lowercase slug")
		}
		if a.Ref == "" {
			p.Field(at+".ref", "required: the path of the artifact")
		}
		p.SHA256Field(at+".sha256", a.SHA256)
		if a.Rows != nil && *a.Rows < 0 {
			p.Field(at+".rows", "must not be negative")
		}
	}
	if r.Runtime != nil {
		r.Runtime.validate(p)
	}
	return p
}

// validateEngine checks the engine against the scenario that chose it (SIM-1): a document
// stating another engine than its scenario names describes a run that did not happen.
func (r *Result) validateEngine(p *validate.Problems) {
	e := r.Engine
	if e.Kind == "" {
		p.Field("engine.kind", "required: the engine the run used, for example vllm")
	}
	if e.Version == "" {
		p.Field("engine.version", "required: the upstream release the engine models")
	}
	sc := r.Input.Scenario
	if v := sc.EngineRelease(); v != "" && e.Version != "" && v != e.Version {
		p.Field("engine.version", "is %q but the scenario names release %q: the run models the release its scenario names", e.Version, v)
	}
	if sc.Engine != nil && sc.Engine.Kind != "" && e.Kind != "" && sc.Engine.Kind != e.Kind {
		p.Field("engine.kind", "is %q but input.scenario.engine.kind is %q", e.Kind, sc.Engine.Kind)
	}
}

func (d Data) validate(p *validate.Problems) {
	d.Catalog.validate(p, "data.catalog")
	d.Registry.validate(p, "data.registry")
	switch {
	case d.Build == "":
		p.Field("data.build", "required: a release tag such as v0.16.0, or %q", BuildUnreleased)
	case d.Build != BuildUnreleased && !releaseTag.MatchString(d.Build):
		p.Field("data.build", "is %q: a release tag vMAJOR.MINOR.PATCH, or %q", d.Build, BuildUnreleased)
	}
	if !d.ReleasedData {
		return
	}
	// The loader alone cannot say a hash equals the substrate release's embedded hash; what
	// the document can say is that nothing it records contradicts it.
	if d.Build == BuildUnreleased {
		p.Field("data.released_data", "is true but data.build is %q: only a tagged simulator release runs released data", BuildUnreleased)
	}
	for name, s := range map[string]DataSource{"catalog": d.Catalog, "registry": d.Registry} {
		if len(s.Overlays) > 0 {
			p.Field("data.released_data", "is true but data.%s has overlays: a run with an overlay is experimental", name)
		}
	}
}

func (s DataSource) validate(p *validate.Problems, at string) {
	switch s.Source {
	case SourceEmbedded, SourcePath:
	case "":
		p.Field(at+".source", "required: %q or %q", SourceEmbedded, SourcePath)
	default:
		p.Field(at+".source", "%q is not %q or %q", s.Source, SourceEmbedded, SourcePath)
	}
	hashField(p, at+".hash", s.Hash)
	hashField(p, at+".combined_hash", s.CombinedHash)
	for i, o := range s.Overlays {
		hashField(p, fmt.Sprintf("%s.overlays[%d].hash", at, i), o.Hash)
	}
	if len(s.Overlays) == 0 && s.CombinedHash != "" && s.Hash != "" && s.CombinedHash != s.Hash {
		p.Field(at+".combined_hash", "differs from hash, but no overlay is applied")
	}
}

// hashField checks a content hash: 64 lowercase hex characters, as design §3.3 computes it.
func hashField(p *validate.Problems, at, v string) {
	switch {
	case v == "":
		p.Field(at, "required: the content hash, 64 lowercase hex characters")
	case !lowerHex64.MatchString(v):
		p.Field(at, "must be 64 lowercase hex characters, got %q", v)
	}
}

// validateResolutions checks the engine's resolutions: each names a setting once, in order,
// resolves to a value, explains a difference from the value the host gave, and starts from
// the value the host simulated where the host overrode the setting.
func (r *Result) validateResolutions(p *validate.Problems) {
	hostRun := map[string]string{}
	for _, o := range r.Overrides {
		hostRun[o.Setting] = o.Run
	}
	pools := PoolLabels(r.Input.Deployment)
	prev := ""
	for i, x := range r.Resolutions {
		at := fmt.Sprintf("resolutions[%d]", i)
		key := x.Setting + "\x00" + x.Pool
		switch {
		case x.Setting == "":
			p.Field(at+".setting", "required")
		case i > 0 && key <= prev:
			p.Field(at+".setting", "resolutions are listed once each, sorted by setting and then pool; %q (pool %q) follows %q", x.Setting, x.Pool, strings.ReplaceAll(prev, "\x00", " pool "))
		}
		prev = key
		if x.Pool != "" && !contains(pools, x.Pool) {
			p.Field(at+".pool", "%q is not a pool of the deployment; its pools are %v", x.Pool, pools)
		}
		if x.Engine == "" {
			p.Field(at+".engine", "required: the value the engine resolved to")
		}
		differs := x.Host != "" && x.Engine != "" && x.Host != x.Engine
		switch {
		case differs && x.Reason == "":
			p.Field(at+".reason", "required: the host gave %q and the engine resolved %q", x.Host, x.Engine)
		case !differs && x.Reason != "":
			p.Field(at+".reason", "only where the engine's value differs from the host's")
		}
		if run, ok := hostRun[x.Setting]; ok && x.Host != "" && x.Host != run {
			p.Field(at+".host", "the host overrode %q to %q, so the engine was given that value, not %q",
				x.Setting, run, x.Host)
		}
	}
}

func (s SelectedEntries) validate(p *validate.Problems) {
	if s.ClosenessVersion != 0 && s.ClosenessVersion != ClosenessVersion {
		p.Field("selected_entries.closeness_version", "must be %d or absent, got %d", ClosenessVersion, s.ClosenessVersion)
	}
	read := map[string]bool{}
	var prev string
	for i, c := range s.Coefficients {
		at := fmt.Sprintf("selected_entries.coefficients[%d]", i)
		for f, v := range map[string]string{"set": c.Set, "name": c.Name, "method": string(c.Method), "file": c.File} {
			if v == "" {
				p.Field(at+"."+f, "required")
			}
		}
		if c.Method != "" && !c.Method.Valid() {
			p.Field(at+".method", "%q is not a registry method", c.Method)
		}
		key := c.Set + "\x00" + c.Name + "\x00" + string(c.Method) + "\x00" + c.Scope.Key()
		if i > 0 && key <= prev {
			p.Field(at, "coefficients are listed once each, sorted by set, name, method and scope")
		}
		prev = key
		read[c.File] = true
		coefficient.ValidateKernelImpls(p, at+".kernel_impls", c.KernelImpls)
		coefficient.ValidateFitRange(p, at+".fit_range", c.FitRange)
		if iv := c.CI95; iv != nil && (math.IsNaN(iv.Low) || math.IsNaN(iv.High) || iv.Low > iv.High) {
			p.Field(at+".ci95", "must be an interval [low, high] with low not above high, got [%v, %v]", iv.Low, iv.High)
		}
		if len(c.KernelImpls) > 0 && !coefficient.MayStateKernelImpls(c.Method) {
			p.Field(at+".kernel_impls", "method %q states nothing about a kernel implementation, so the entry states none", c.Method)
		}
		if len(c.FitRange) > 0 && !coefficient.MayStateFitRange(c.Method) {
			p.Field(at+".fit_range", "method %q was not fitted to data, so the entry states no fit range", c.Method)
		}
		if c.Unidentified && c.CI95 == nil {
			p.Field(at+".unidentified", "is true, but the entry records no ci95 that leaves it unidentified")
		}
		if s.ClosenessVersion == 0 && (len(c.KernelImpls) > 0 || len(c.FitRange) > 0 || c.CI95 != nil || c.Extrapolated || c.Unidentified) {
			p.Field(at, "states closeness data, but selected_entries.closeness_version is absent: without it a reader cannot tell an omitted flag from false")
		}
		if c.Extrapolated && len(c.FitRange) == 0 {
			p.Field(at+".extrapolated", "is true, but the entry records no fit_range to leave")
		}
	}
	for name, list := range map[string][]SelectedProfile{"acceptance_profiles": s.AcceptanceProfiles, "routing_profiles": s.RoutingProfiles} {
		prev := ""
		for i, sp := range list {
			at := fmt.Sprintf("selected_entries.%s[%d]", name, i)
			for f, v := range map[string]string{"set": sp.Set, "key": sp.Key, "file": sp.File} {
				if v == "" {
					p.Field(at+"."+f, "required")
				}
			}
			key := sp.Set + "\x00" + sp.Key
			if i > 0 && key <= prev {
				p.Field(at, "profiles are listed once each, sorted by set and key")
			}
			prev = key
			read[sp.File] = true
		}
	}
	for i, f := range s.UnreadFiles {
		at := fmt.Sprintf("selected_entries.unread_files[%d]", i)
		switch {
		case f == "" || strings.HasPrefix(f, "/") || strings.Contains(f, `\`):
			p.Field(at, "is a path relative to the registry directory, with forward slashes, got %q", f)
		case i > 0 && f <= s.UnreadFiles[i-1]:
			p.Field(at, "files are listed once each, sorted")
		case read[f]:
			p.Field(at, "%q is listed as unread, but a selected entry was read from it", f)
		}
	}
}

func (in Input) validate(p *validate.Problems) {
	p.Merge("input.scenario", in.Scenario.Validate())
	p.Merge("input.deployment", in.Deployment.Validate())
	c := in.Scenario.Cluster
	gpusPerNode := c.GPUsPerNode
	switch {
	case in.Packaging != nil:
		p.Merge("input.packaging", in.Packaging.Validate())
		p.Merge("input.packaging", in.Packaging.ValidateAgainst(c))
		gpusPerNode = in.Packaging.GPUsPerNode
	case c.GPUsPerNode == 0:
		p.Field("input.packaging", "required: the scenario leaves gpus_per_node to the catalog entry of %q, so the Result records the value the run resolved",
			c.Hardware)
	}
	p.Merge("input.deployment", in.Deployment.ValidateAgainstCluster(deployment.ClusterConstraints{
		Nodes: c.Nodes, GPUsPerNode: gpusPerNode, Storage: c.Storage}))
	for i, o := range in.Options {
		at := fmt.Sprintf("input.options[%d]", i)
		switch {
		case o.Name == "" || strings.HasPrefix(o.Name, "-"):
			p.Field(at+".name", "is the flag without its dashes, or an environment variable, got %q", o.Name)
		case i > 0 && o.Name <= in.Options[i-1].Name:
			p.Field(at+".name", "options are listed once each, sorted by name")
		}
		switch o.Source {
		case OptionCommandLine, OptionEnvironment, OptionDefault:
		default:
			p.Field(at+".source", "%q is not one of command_line, environment, default", o.Source)
		}
	}
	for i, w := range in.WorkloadProperties {
		at := fmt.Sprintf("input.workload_properties[%d]", i)
		switch {
		case !slug.MatchString(w.Name):
			p.Field(at+".name", "must be a lowercase slug, got %q", w.Name)
		case i > 0 && w.Name <= in.WorkloadProperties[i-1].Name:
			p.Field(at+".name", "properties are listed once each, sorted by name")
		}
		switch w.Source {
		case PropertyCommandLine, PropertyScenario, PropertyWorkload, PropertyProfile, PropertyDefault:
		default:
			p.Field(at+".source", "%q is not one of command_line, scenario, workload, profile, default", w.Source)
		}
		if (w.Source == PropertyProfile) != (w.Profile != "") {
			p.Field(at+".profile", "is stated exactly when source is profile")
		}
		w.validateBorrowing(p, at)
		if len(w.Values) == 0 && w.Source != PropertyDefault {
			p.Field(at+".values", "required: the values %s supplied", w.Source)
		}
		for _, k := range sortedKeys(w.Values) {
			p.FiniteField(at+".values."+k, w.Values[k])
		}
	}
}

func (f Frame) validate(p *validate.Problems) {
	switch f.Kind {
	case FrameTerminating, FrameSteadyState:
	default:
		p.Field("frame.kind", "%q is not %q or %q", f.Kind, FrameTerminating, FrameSteadyState)
	}
	g := f.Generator
	if g.Workload != nil {
		p.Merge("frame.generator.workload", g.Workload.Validate())
	}
	l := g.Load
	if l.Concurrency < 0 {
		p.Field("frame.generator.load.concurrency", "must not be negative")
	}
	if p.FiniteField("frame.generator.load.request_rate", l.RequestRate) && l.RequestRate < 0 {
		p.Field("frame.generator.load.request_rate", "must not be negative")
	}
	if l.Concurrency > 0 && l.RequestRate > 0 {
		p.Field("frame.generator.load", "states a concurrency or a request rate, not both: one run is driven one way")
	}
	if g.Workload == nil && l.Concurrency == 0 && l.RequestRate == 0 {
		p.Field("frame.generator", "states a workload, an offered load, or both")
	}
	if g.RNG == "" {
		p.Field("frame.generator.rng", "required: the random generator and its version (design §8.4)")
	}
	switch f.InitialState.Caches {
	case CachesEmpty:
		if f.InitialState.Rule != "" {
			p.Field("frame.initial_state.rule", "states how caches were warmed; empty caches take no rule")
		}
	case CachesWarmed:
		if f.InitialState.Rule == "" {
			p.Field("frame.initial_state.rule", "required: the rule that warmed the caches")
		}
	default:
		p.Field("frame.initial_state.caches", "%q is not %q or %q", f.InitialState.Caches, CachesEmpty, CachesWarmed)
	}
	f.Phases.validate(p)
	s := f.Stop
	if s.Requests < 0 {
		p.Field("frame.stop.requests", "must not be negative")
	}
	if h := s.HorizonS; h != nil && p.FiniteField("frame.stop.horizon_s", *h) && *h <= 0 {
		p.Field("frame.stop.horizon_s", "must be positive when set")
	}
	if s.Requests == 0 && s.HorizonS == nil {
		p.Field("frame.stop", "states a request count, a horizon, or both")
	}
	for i, c := range f.Acceptance {
		at := fmt.Sprintf("frame.acceptance[%d]", i)
		switch c.Condition {
		case ConditionConcurrencyTolerance, ConditionUnfinishedAtMost:
		default:
			p.Field(at+".condition", "%q is not %q or %q", c.Condition, ConditionConcurrencyTolerance, ConditionUnfinishedAtMost)
		}
		if p.FiniteField(at+".threshold", c.Threshold) && c.Threshold < 0 {
			p.Field(at+".threshold", "must not be negative")
		}
		if p.FiniteField(at+".observed", c.Observed) && c.Observed < 0 {
			p.Field(at+".observed", "must not be negative")
		}
		if want := c.Observed <= c.Threshold; c.Met != want {
			p.Field(at+".met", "is %v but observed %v against threshold %v gives %v", c.Met, c.Observed, c.Threshold, want)
		}
	}
	if len(f.Metrics) == 0 {
		p.Field("frame.metrics", "required: the metric identifiers the frame measures")
	}
	for i, id := range f.Metrics {
		at := fmt.Sprintf("frame.metrics[%d]", i)
		if _, ok := metrics.Lookup(id); !ok {
			p.Field(at, "%q is not in the metric registry (contracts/metrics)", id)
		}
		if i > 0 && id <= f.Metrics[i-1] {
			p.Field(at, "identifiers are listed once each, sorted")
		}
	}
}

// Accepted reports whether the run met every acceptance condition of its frame.
func (f Frame) Accepted() bool {
	for _, c := range f.Acceptance {
		if !c.Met {
			return false
		}
	}
	return true
}

func (ph Phases) validate(p *validate.Problems) {
	w := ph.Warmup
	switch w.Rule {
	case WarmupNone, WarmupMarked, WarmupMSER5:
		if w.Requests != 0 || w.DurationS != 0 {
			p.Field("frame.phases.warmup", "rule %s takes neither requests nor duration_s", w.Rule)
		}
	case WarmupRequests:
		if w.Requests < 1 || w.DurationS != 0 {
			p.Field("frame.phases.warmup", "rule requests takes a positive requests and no duration_s")
		}
	case WarmupDuration:
		if !p.FiniteField("frame.phases.warmup.duration_s", w.DurationS) {
			break
		}
		if w.DurationS <= 0 || w.Requests != 0 {
			p.Field("frame.phases.warmup", "rule duration takes a positive duration_s and no requests")
		}
	default:
		p.Field("frame.phases.warmup.rule", "%q is not one of none, requests, duration, marked, mser5", w.Rule)
	}
	switch ph.Measured.Start {
	case StartFirstArrivalAfterWarmup, StartProfiling, StartSimulation:
	default:
		p.Field("frame.phases.measured.start", "%q is not one of first_arrival_after_warmup, profiling_start, simulation_start", ph.Measured.Start)
	}
	switch ph.Measured.Stop {
	case StopLastCompletion, StopProfiling, StopSimulation:
	default:
		p.Field("frame.phases.measured.stop", "%q is not one of last_completion, profiling_end, simulation_end", ph.Measured.Stop)
	}
	d := ph.Drain
	switch d.Rule {
	case DrainNone, DrainUntilComplete:
		if d.DurationS != 0 {
			p.Field("frame.phases.drain.duration_s", "only a grace drain takes a duration")
		}
	case DrainGrace:
		if p.FiniteField("frame.phases.drain.duration_s", d.DurationS) && d.DurationS <= 0 {
			p.Field("frame.phases.drain.duration_s", "must be positive for a grace drain")
		}
	default:
		p.Field("frame.phases.drain.rule", "%q is not one of none, until_complete, grace", d.Rule)
	}
}

func (v Revisions) validate(p *validate.Problems) {
	if v.Simulator == "" {
		p.Field("revisions.simulator", "required: the inference-sim build that produced the result")
	}
	if v.Substrate == "" {
		p.Field("revisions.substrate", "required: the blis-substrate version the build selected")
	}
}

// revisionOf returns the value a revision change names for a component, and whether the run states
// one.
func (r *Result) revisionOf(component string) (string, bool) {
	v := map[string]string{
		"simulator": r.Revisions.Simulator, "substrate": r.Revisions.Substrate,
		"router": r.Revisions.Router, "engine_module": r.Revisions.EngineModule,
		"catalog": r.Data.Catalog.CombinedHash, "registry": r.Data.Registry.CombinedHash,
	}
	s, known := v[component]
	return s, known && s != ""
}

func (r *Result) validateRerun(p *validate.Problems) {
	if r.RerunOf == nil {
		if len(r.RevisionChanges) > 0 {
			p.Field("revision_changes", "only a rerun has revision changes; set rerun_of")
		}
		return
	}
	if !slug.MatchString(r.RerunOf.Name) {
		p.Field("rerun_of.name", "must be a lowercase slug, got %q", r.RerunOf.Name)
	}
	p.SHA256Field("rerun_of.sha256", r.RerunOf.SHA256)
	seen := map[string]bool{}
	for i, c := range r.RevisionChanges {
		at := fmt.Sprintf("revision_changes[%d]", i)
		own, ok := r.revisionOf(c.Component)
		known := c.Component == "simulator" || c.Component == "substrate" || c.Component == "router" ||
			c.Component == "engine_module" || c.Component == "catalog" || c.Component == "registry"
		switch {
		case !known:
			p.Field(at+".component", "%q is not simulator, substrate, router, engine_module, catalog or registry", c.Component)
		case seen[c.Component]:
			p.Field(at+".component", "%q is listed twice", c.Component)
		case c.From == c.To:
			p.Field(at, "from and to are both %q: an unchanged revision is not a change", c.From)
		case !ok:
			p.Field(at+".to", "this run states no %s revision to have changed to", c.Component)
		case own != c.To:
			p.Field(at+".to", "is %q but this run's %s is %q", c.To, c.Component, own)
		}
		seen[c.Component] = true
	}
}

func (w Window) validate(p *validate.Problems) {
	if w.StartUs < 0 {
		p.Field("window.start_us", "must not be negative")
	}
	if w.EndUs < w.StartUs {
		p.Field("window.end_us", "is %d, before start_us %d", w.EndUs, w.StartUs)
	}
	if p.FiniteField("window.duration_s", w.DurationS) {
		if want := float64(w.EndUs-w.StartUs) / 1e6; w.DurationS != want {
			p.Field("window.duration_s", "is %v but end_us - start_us is %v s", w.DurationS, want)
		}
	}
	if w.Requests < 0 {
		p.Field("window.requests", "must not be negative")
	}
}

func (r *Result) validateMetrics(p *validate.Problems) {
	measured := map[string]bool{}
	for _, id := range r.Frame.Metrics {
		measured[id] = true
	}
	if !r.Frame.Accepted() && len(r.Metrics) > 0 {
		p.Field("metrics", "a run that fails an acceptance condition of its frame reports no figure (DEV-7)")
	}
	for _, id := range sortedKeys(r.Metrics) {
		at := "metrics." + id
		f := r.Metrics[id]
		d, ok := metrics.Lookup(id)
		if !ok {
			p.Field(at, "%q is not in the metric registry (contracts/metrics): a figure without a definition cannot be compared", id)
			continue
		}
		if !measured[id] {
			p.Field(at, "is not one of frame.metrics")
		}
		f.Basis().Validate(p, at, d)
		if p.FiniteField(at+".value", f.Value) && f.Value < 0 {
			p.Field(at+".value", "must not be negative: every metric is a latency, a rate, a fraction or a count")
		}
		if d.Unit == metrics.UnitFraction && f.Value > 1 {
			p.Field(at+".value", "is a fraction, at most 1, got %v", f.Value)
		}
		if f.Samples < 0 {
			p.Field(at+".samples", "must not be negative")
		}
		if d.Unfinished == metrics.UnfinishedExcluded && f.Samples == 0 {
			p.Field(at+".samples", "is 0: a statistic over no sample is not reported")
		}
		if d.Statistic == metrics.StatRate && d.Window == metrics.WindowMeasured && !(r.Window.DurationS > 0) {
			p.Field("window.duration_s", "must be positive: metric %s is a rate over the window", id)
		}
	}
	if r.Window.Requests > r.Requests.Completed {
		p.Field("window.requests", "is %d but only %d requests completed in the whole run", r.Window.Requests, r.Requests.Completed)
	}
}

var dropReasons = map[DropReason]bool{
	DropAdmissionRejected: true, DropRoutingRejected: true, DropEncodeRoutingRejected: true,
	DropGatewayQueueRejected: true, DropGatewayEvicted: true, DropGatewayExpired: true,
	DropUnservable: true, DropContextOverflow: true, DropDecodeKV: true,
}

// Valid reports whether d is a recognised drop reason.
func (d DropReason) Valid() bool { return dropReasons[d] }

// AllDropReasons returns every drop reason, sorted.
func AllDropReasons() []string {
	out := make([]string, 0, len(dropReasons))
	for d := range dropReasons {
		out = append(out, string(d))
	}
	sort.Strings(out)
	return out
}

var fates = map[Fate]bool{FateCompleted: true, FateDropped: true, FateTimedOut: true,
	FateInFlightAtHorizon: true, FateQueuedAtHorizon: true, FateNotStarted: true}

// Valid reports whether f is one of the six fates of design §8.7.
func (f Fate) Valid() bool { return fates[f] }

func (q Requests) validate(p *validate.Problems) {
	for name, v := range map[string]int{
		"generated": q.Generated, "completed": q.Completed, "timed_out": q.TimedOut,
		"in_flight_at_horizon": q.InFlightAtHorizon, "queued_at_horizon": q.QueuedAtHorizon,
		"not_started": q.NotStarted, "length_capped": q.LengthCapped,
	} {
		if v < 0 {
			p.Field("requests."+name, "must not be negative")
		}
	}
	for _, reason := range sortedKeys(q.Dropped) {
		at := "requests.dropped." + string(reason)
		if !reason.Valid() {
			p.Field(at, "%q is not one of %v", reason, AllDropReasons())
		}
		if q.Dropped[reason] < 1 {
			p.Field(at, "must be positive; omit a reason no request was dropped for")
		}
	}
	if s := q.Sum(); s != q.Generated {
		p.Field("requests", "generated is %d but the fates sum to %d: every generated request has exactly one fate (DEV-9)", q.Generated, s)
	}
	if q.LengthCapped > q.Completed {
		p.Field("requests.length_capped", "is %d but only %d requests completed", q.LengthCapped, q.Completed)
	}
}

// validateRows checks the per-request rows against each other and against the fate
// counts: every generated request has one row, and each row's fate agrees with the
// samples it carries.
func (r *Result) validateRows(p *validate.Problems) {
	rows := r.RequestRows
	if len(rows) == 0 {
		return
	}
	if len(rows) != r.Requests.Generated {
		p.Field("request_rows", "has %d rows but requests.generated is %d: one row per generated request", len(rows), r.Requests.Generated)
	}
	counts := map[Fate]int{}
	dropped := map[DropReason]int{}
	ids := map[string]bool{}
	completion := map[int]bool{}
	capped := 0
	for i, row := range rows {
		at := fmt.Sprintf("request_rows[%d]", i)
		if row.ID == "" {
			p.Field(at+".id", "required")
		} else if ids[row.ID] {
			p.Field(at+".id", "%q appears twice: a request identifier is never reused in a run (SIM-8)", row.ID)
		}
		ids[row.ID] = true
		if !row.Fate.Valid() {
			p.Field(at+".fate", "%q is not one of the six fates of design §8.7", row.Fate)
		}
		counts[row.Fate]++
		switch {
		case row.Fate == FateDropped && !row.DropReason.Valid():
			p.Field(at+".drop_reason", "a dropped request names one of %v", AllDropReasons())
		case row.Fate != FateDropped && row.DropReason != "":
			p.Field(at+".drop_reason", "only a dropped request has a drop reason")
		}
		if row.Fate == FateDropped {
			dropped[row.DropReason]++
		}
		completed := row.Fate == FateCompleted
		for f, v := range map[string]int{"prompt_tokens": row.PromptTokens, "output_tokens": row.OutputTokens,
			"cached_local_tokens": row.CachedLocalTokens, "cached_external_tokens": row.CachedExternalTokens,
			"computed_tokens": row.ComputedTokens} {
			if v < 0 {
				p.Field(at+"."+f, "must not be negative")
			}
		}
		if row.ArrivalUs < 0 {
			p.Field(at+".arrival_us", "must not be negative")
		}
		// A sample exists only where the request produced it (DEV-9).
		if row.TTFTUs != nil && row.OutputTokens < 1 {
			p.Field(at+".ttft_us", "a request that emitted no token has no time to first token")
		}
		if row.TPOTUs != nil && (!completed || row.OutputTokens < 2) {
			p.Field(at+".tpot_us", "only a completed request with at least two output tokens has a time per output token")
		}
		if row.E2EUs != nil && !completed {
			p.Field(at+".e2e_us", "only a completed request has a request latency")
		}
		if completed && row.OutputTokens >= 1 && (row.TTFTUs == nil || row.E2EUs == nil) {
			p.Field(at, "a completed request that emitted a token carries ttft_us and e2e_us")
		}
		if completed && row.OutputTokens >= 2 && row.TPOTUs == nil {
			p.Field(at+".tpot_us", "required for a completed request with at least two output tokens")
		}
		for f, v := range map[string]*int64{"ttft_us": row.TTFTUs, "e2e_us": row.E2EUs,
			"scheduling_delay_us": row.SchedulingDelayUs, "gateway_queue_delay_us": row.GatewayQueueDelayUs} {
			if v != nil && *v < 0 {
				p.Field(at+"."+f, "must not be negative")
			}
		}
		if row.TPOTUs != nil && (math.IsNaN(*row.TPOTUs) || math.IsInf(*row.TPOTUs, 0) || *row.TPOTUs < 0) {
			p.Field(at+".tpot_us", "must be a finite number, not negative")
		}
		if row.TTFTUs != nil && row.E2EUs != nil && *row.E2EUs < *row.TTFTUs {
			p.Field(at+".e2e_us", "is %d, less than ttft_us %d", *row.E2EUs, *row.TTFTUs)
		}
		switch {
		case row.CompletionIndex != 0 && !completed:
			p.Field(at+".completion_index", "only a completed request has a completion index")
		case completed && (row.CompletionIndex < 1 || row.CompletionIndex > r.Requests.Completed):
			p.Field(at+".completion_index", "is %d: a completed request's index lies in [1, %d]", row.CompletionIndex, r.Requests.Completed)
		case completed && completion[row.CompletionIndex]:
			p.Field(at+".completion_index", "%d is given to two requests", row.CompletionIndex)
		}
		completion[row.CompletionIndex] = true
		if row.LengthCapped {
			capped++
			if !completed {
				p.Field(at+".length_capped", "only a completed request stops at a length limit")
			}
		}
		switch row.Phase {
		case "", PhaseWarmup, PhaseProfile, PhaseGrace:
		default:
			p.Field(at+".phase", "%q is not one of warmup, profile, grace", row.Phase)
		}
		if a := row.Agentic; a != nil {
			switch a.Kind {
			case AgenticRoot, AgenticSubagent, AgenticFlat:
			default:
				p.Field(at+".agentic.kind", "%q is not one of root, subagent, flat", a.Kind)
			}
			if a.TraceID == "" || a.Stream == "" {
				p.Field(at+".agentic", "states trace_id and stream")
			}
		}
	}
	q := r.Requests
	for f, want := range map[Fate]int{FateCompleted: q.Completed, FateTimedOut: q.TimedOut,
		FateInFlightAtHorizon: q.InFlightAtHorizon, FateQueuedAtHorizon: q.QueuedAtHorizon, FateNotStarted: q.NotStarted} {
		if counts[f] != want {
			p.Field("request_rows", "%d rows have fate %s but requests counts %d", counts[f], f, want)
		}
	}
	for _, reason := range sortedKeys(dropped) {
		if dropped[reason] != q.Dropped[reason] {
			p.Field("request_rows", "%d rows were dropped for %s but requests.dropped counts %d", dropped[reason], reason, q.Dropped[reason])
		}
	}
	for _, reason := range sortedKeys(q.Dropped) {
		if dropped[reason] == 0 {
			p.Field("request_rows", "no row was dropped for %s but requests.dropped counts %d", reason, q.Dropped[reason])
		}
	}
	if capped != q.LengthCapped {
		p.Field("request_rows", "%d rows are length-capped but requests.length_capped is %d", capped, q.LengthCapped)
	}
}

func (g *Goodput) validate(p *validate.Problems) {
	if len(g.Targets) == 0 {
		p.Field("goodput.targets", "required: goodput is measured against SLO targets")
	}
	for _, class := range sortedKeys(g.Targets) {
		if class == "" {
			p.Field("goodput.targets", "an SLO class key must not be empty")
			continue
		}
		p.Merge("goodput.targets."+class, g.Targets[class].Validate())
	}
	for _, class := range sortedKeys(g.PerClass) {
		c := g.PerClass[class]
		if _, ok := g.Targets[class]; !ok {
			p.Field("goodput.per_class."+class, "is not a class in goodput.targets")
		}
		if c.Met < 0 || c.Completed < 0 || c.Met > c.Completed {
			p.Field("goodput.per_class."+class, "met (%d) is between 0 and completed (%d)", c.Met, c.Completed)
		}
	}
}

// validateBreakdowns checks each split's values. A per-model, per-tenant or per-adapter
// slice splits the result's own metrics, so each of its keys is a reported metric's
// identifier; the prefill/decode and session slices name quantities of their own.
func (r *Result) validateBreakdowns(p *validate.Problems) {
	b := r.Breakdowns
	flat := func(at string, m map[string]float64, split bool) {
		for _, name := range sortedKeys(m) {
			kat := at + "." + name
			if !slug.MatchString(name) {
				p.Field(kat, "a quantity name is a lowercase slug")
			}
			if _, ok := r.Metrics[name]; split && !ok {
				p.Field(kat, "is not a key of metrics: a split restates a reported metric over the key's requests")
			}
			if v := m[name]; p.FiniteField(kat, v) && v < 0 {
				p.Field(kat, "must not be negative")
			}
		}
	}
	flat("breakdowns.pd", b.PD, false)
	flat("breakdowns.sessions", b.Sessions, false)
	for axis, m := range map[string]map[string]map[string]float64{
		"per_model": b.PerModel, "per_tenant": b.PerTenant, "per_adapter": b.PerAdapter,
	} {
		for _, key := range sortedKeys(m) {
			if key == "" {
				p.Field("breakdowns."+axis, "a key must not be empty")
			}
			flat("breakdowns."+axis+"."+key, m[key], true)
		}
	}
}

func (rt *Runtime) validate(p *validate.Problems) {
	if p.FiniteField("runtime.wall_clock_s", rt.WallClockS) && rt.WallClockS < 0 {
		p.Field("runtime.wall_clock_s", "must not be negative")
	}
	if rt.StartedUTC != "" {
		if _, err := time.Parse(time.RFC3339, rt.StartedUTC); err != nil || !strings.HasSuffix(rt.StartedUTC, "Z") {
			p.Field("runtime.started_utc", "must be an RFC 3339 timestamp in UTC (ending Z), got %q", rt.StartedUTC)
		}
	}
}

func sortedKeys[K ~string, V any](m map[K]V) []K {
	out := make([]K, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// validateBorrowing checks the fields that say how a profile lookup found its value (D8). They
// describe a profile, so none is stated for another source; a borrowed value names the key
// fields that differ and the profiles it read; an exact match differs in none; and a K rule
// appears only for the step that changes K.
func (w WorkloadProperty) validateBorrowing(p *validate.Problems, at string) {
	if w.Step == "" {
		for field, set := range map[string]bool{"differed": len(w.Differed) > 0, "k_rule": w.KRule != "", "profiles": len(w.Profiles) > 0} {
			if set {
				p.Field(at+"."+field, "is stated only with step")
			}
		}
		return
	}
	if w.Source != PropertyProfile {
		p.Field(at+".step", "is stated only when source is profile, because only a profile lookup borrows")
	}
	steps := map[string]bool{"exact": true, "same_drafter_other_k": true, "other_drafter": true, "other_target": true}
	if !steps[w.Step] {
		p.Field(at+".step", "%q is not one of exact, same_drafter_other_k, other_drafter, other_target", w.Step)
	}
	if len(w.Profiles) == 0 {
		p.Field(at+".profiles", "required with step: the profiles the value was read from")
	}
	if w.Step == "exact" && (len(w.Differed) > 0 || w.KRule != "") {
		p.Field(at+".differed", "an exact match differs in no key field and brings no K to the run's")
	}
	if w.Step != "exact" && len(w.Differed) == 0 {
		p.Field(at+".differed", "required for a borrowed value: the key fields in which it differs")
	}
	for i, d := range w.Differed {
		if !IsProfileKeyField(d) {
			p.Field(fmt.Sprintf("%s.differed[%d]", at, i), "%q is not one of %s", d, strings.Join(ProfileKeyFields(), ", "))
		}
	}
	rules := map[string]bool{"interpolated": true, "above_largest": true, "below_smallest_rates": true, "below_smallest_proportional": true}
	switch {
	case w.KRule != "" && !rules[w.KRule]:
		p.Field(at+".k_rule", "%q is not one of interpolated, above_largest, below_smallest_rates, below_smallest_proportional", w.KRule)
	case w.KRule != "" && w.Step == "exact":
		p.Field(at+".k_rule", "an exact match brings no K to the run's")
	case w.Step == "same_drafter_other_k" && w.KRule == "":
		p.Field(at+".k_rule", "required for step same_drafter_other_k")
	}
}

var profileKeyFields = []string{"model", "drafter", "speculative_method", "num_spec_tokens"}

// ProfileKeyFields returns the key fields in which a borrowed acceptance profile can differ
// from the run's (design D8): the target model, the drafter, the speculative method and the
// number of speculative tokens. A point record's profile substitution names them from the same
// list.
func ProfileKeyFields() []string { return append([]string(nil), profileKeyFields...) }

// IsProfileKeyField reports whether name is one of ProfileKeyFields.
func IsProfileKeyField(name string) bool {
	for _, f := range profileKeyFields {
		if f == name {
			return true
		}
	}
	return false
}
