package result

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
)

var slug = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]*$`)

// Validate performs field-level validation of a Result on its own: its shape, its
// embedded input, the definitions behind its metrics, and the request ledger.
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
	r.Input.validate(p)
	r.Revisions.validate(p)
	r.validateEngineRevision(p)
	r.validateRerun(p)
	r.validateMetrics(p)
	r.Window.validate(p)
	r.Requests.validate(p)
	r.validateCounts(p)
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

func (in Input) validate(p *validate.Problems) {
	p.Merge("input.scenario", in.Scenario.Validate())
	p.Merge("input.deployment", in.Deployment.Validate())
	c := in.Scenario.Cluster
	p.Merge("input.deployment", in.Deployment.ValidateAgainstCluster(deployment.ClusterConstraints{
		Nodes: c.Nodes, GPUsPerNode: c.GPUsPerNode, Storage: c.Storage}))
	in.Run.validate(p)
}

func (run Run) validate(p *validate.Problems) {
	l := run.Load
	if l.Concurrency < 0 {
		p.Field("input.run.load.concurrency", "must not be negative")
	}
	if p.FiniteField("input.run.load.request_rate", l.RequestRate) && l.RequestRate < 0 {
		p.Field("input.run.load.request_rate", "must not be negative")
	}
	switch {
	case l.Concurrency > 0 && l.RequestRate > 0:
		p.Field("input.run.load", "states a concurrency or a request rate, not both: one run is driven one way")
	case l.Concurrency == 0 && l.RequestRate == 0:
		p.Field("input.run.load", "states a concurrency or a request rate")
	}
	if run.NumRequests <= 0 {
		p.Field("input.run.num_requests", "must be positive")
	}
	if h := run.HorizonS; h != nil && p.FiniteField("input.run.horizon_s", *h) && *h <= 0 {
		p.Field("input.run.horizon_s", "must be positive when set")
	}
	if w := run.Warmup; w != nil {
		switch {
		case w.Requests < 0:
			p.Field("input.run.warmup.requests", "must not be negative")
		case run.NumRequests > 0 && w.Requests >= run.NumRequests:
			p.Field("input.run.warmup.requests", "is %d but the run generates %d: a warmup leaves requests to measure", w.Requests, run.NumRequests)
		case !p.FiniteField("input.run.warmup.duration_s", w.DurationS):
		case w.DurationS < 0:
			p.Field("input.run.warmup.duration_s", "must not be negative")
		case w.Requests > 0 && w.DurationS > 0:
			p.Field("input.run.warmup", "a number of requests or a duration, not both")
		case w.Requests == 0 && w.DurationS == 0:
			p.Field("input.run.warmup", "states a number of requests or a duration; omit it for no warmup")
		}
	}
	if run.Workload != nil {
		p.Merge("input.run.workload", run.Workload.Validate())
	}
}

func (v Revisions) validate(p *validate.Problems) {
	if v.BLIS == "" {
		p.Field("revisions.blis", "required: the simulator build that produced the result")
	}
	if v.BLISSchemas == "" {
		p.Field("revisions.blis_schemas", "required: the schema release the document was written against")
	}
	if e := v.Engine; e != nil {
		if e.Kind == "" {
			p.Field("revisions.engine.kind", "required")
		}
		if e.UpstreamVersion == "" {
			p.Field("revisions.engine.upstream_version", "required")
		}
	}
	for name, d := range map[string]*DataRepo{"catalog": v.Catalog, "registry": v.Registry} {
		if d != nil && d.Release == "" && d.Revision == "" {
			p.Field("revisions."+name, "states a release, a revision, or both")
		}
	}
}

// components are the keys of Revisions, which RevisionChange.Component names.
var components = map[string]bool{
	"blis": true, "blis_schemas": true, "blis_latency_kernel": true, "llm_d_router": true,
	"engine": true, "catalog": true, "registry": true,
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
		switch {
		case !components[c.Component]:
			p.Field(at+".component", "%q is not a key of revisions", c.Component)
		case seen[c.Component]:
			p.Field(at+".component", "%q is listed twice", c.Component)
		}
		seen[c.Component] = true
		if c.From == c.To {
			p.Field(at, "from and to are both %q: an unchanged revision is not a change", c.From)
		}
		if own, ok := r.Revisions.of(c.Component); ok && !own[c.To] {
			p.Field(at+".to", "is %q but this run's revisions.%s is %s", c.To, c.Component, describe(own))
		} else if !ok && components[c.Component] {
			p.Field(at+".to", "this run states no revisions.%s to have changed to", c.Component)
		}
	}
}

// of returns the values revisions.<component> may be named by in a revision change: the
// string itself, or a data repository's release or revision, or an engine model's release
// or upstream version. ok is false when the run states no such revision.
func (v Revisions) of(component string) (map[string]bool, bool) {
	set := func(vals ...string) (map[string]bool, bool) {
		out := map[string]bool{}
		for _, s := range vals {
			if s != "" {
				out[s] = true
			}
		}
		return out, len(out) > 0
	}
	switch component {
	case "blis":
		return set(v.BLIS)
	case "blis_schemas":
		return set(v.BLISSchemas)
	case "blis_latency_kernel":
		return set(v.BLISLatencyKernel)
	case "llm_d_router":
		return set(v.LLMDRouter)
	case "engine":
		if v.Engine != nil {
			return set(v.Engine.Release, v.Engine.UpstreamVersion)
		}
	case "catalog", "registry":
		d := v.Catalog
		if component == "registry" {
			d = v.Registry
		}
		if d != nil {
			return set(d.Release, d.Revision)
		}
	}
	return nil, false
}

func describe(vals map[string]bool) string {
	keys := sortedKeys(vals)
	for i, k := range keys {
		keys[i] = fmt.Sprintf("%q", k)
	}
	return strings.Join(keys, " or ")
}

// validateEngineRevision ties the engine model's upstream version to the scenario's: the
// simulator runs the engine release the scenario names, so a document stating another
// would describe a run that did not happen.
func (r *Result) validateEngineRevision(p *validate.Problems) {
	e := r.Revisions.Engine
	if e == nil || e.UpstreamVersion == "" || r.Input.Scenario.EngineVersion == "" {
		return
	}
	if e.UpstreamVersion != r.Input.Scenario.EngineVersion {
		p.Field("revisions.engine.upstream_version", "is %q but input.scenario.engine_version is %q: the run models the release its scenario names",
			e.UpstreamVersion, r.Input.Scenario.EngineVersion)
	}
}

func (r *Result) validateMetrics(p *validate.Problems) {
	if len(r.Metrics) == 0 {
		p.Field("metrics", "a result predicts at least one metric")
	}
	for _, name := range sortedKeys(r.Metrics) {
		at := "metrics." + name
		if _, ok := r.MetricDefs[name]; !ok {
			p.Field(at, "has no entry in metric_defs: a value without a definition cannot be compared")
		}
		if v := r.Metrics[name]; p.FiniteField(at, v) && v < 0 {
			p.Field(at, "must not be negative: every metric is a latency, a rate or a count")
		}
	}
	for _, name := range sortedKeys(r.MetricDefs) {
		at := "metric_defs." + name
		if !slug.MatchString(name) {
			p.Field(at, "a metric name is a lowercase slug")
		}
		if _, ok := r.Metrics[name]; !ok {
			p.Field(at, "defines a metric the result does not report")
		}
		d := r.MetricDefs[name]
		p.Merge(at, measurement.ValidateMetricDef(d))
		// A prediction compares with a measurement only if both definitions are complete:
		// a field left unstated is one the comparison cannot check equal.
		for _, f := range d.Unstated() {
			p.Field(at+"."+f, "required: a result states every field a comparison checks")
		}
		if d.Clock != "" && d.Clock != measurement.ClockSimulated {
			p.Field(at+".clock", "a simulated run's timestamps are on the simulated clock, not %q", d.Clock)
		}
		if want, ok := standardDef(name); ok {
			checkStandard(p, at, d, want)
		}
	}
}

// standard is what a standard metric name fixes about its definition.
type standard struct {
	formula     measurement.Formula
	statistic   measurement.Statistic
	denominator measurement.Denominator
}

// standardNames are the names docs/reference/result.md defines, other than the latency
// families standardDef builds.
var standardNames = map[string]standard{
	"output_tok_s":         {formula: measurement.FormulaOutputThroughput, denominator: measurement.DenominatorProfilingWindow},
	"output_tok_s_per_gpu": {formula: measurement.FormulaOutputThroughput, denominator: measurement.DenominatorPerGPU},
	"request_throughput":   {formula: measurement.FormulaRequestThroughput, denominator: measurement.DenominatorProfilingWindow},
	"interactivity_mean":   {formula: measurement.FormulaUserTPS, statistic: measurement.StatMean},
	"interactivity_p50":    {formula: measurement.FormulaInteractivity, statistic: "p50"},
	"interactivity_p90":    {formula: measurement.FormulaInteractivity, statistic: "p90"},
}

// latencyFamilies are the standard latency names: <prefix>_<statistic>_ms.
var latencyFamilies = map[string]measurement.Formula{
	"ttft": measurement.FormulaTTFT,
	"itl":  measurement.FormulaTPOTExclFirst,
	"e2e":  measurement.FormulaE2EL,
}

var latencyStatistics = []measurement.Statistic{measurement.StatMean, "p50", "p90", "p99"}

// standardDef returns what a standard metric name fixes, and whether name is one.
func standardDef(name string) (standard, bool) {
	if s, ok := standardNames[name]; ok {
		return s, true
	}
	for prefix, f := range latencyFamilies {
		for _, st := range latencyStatistics {
			if name == prefix+"_"+string(st)+"_ms" {
				return standard{formula: f, statistic: st}, true
			}
		}
	}
	return standard{}, false
}

// checkStandard refuses a definition that gives a standard name another meaning: the
// name is what a reader of either kind of document joins on.
func checkStandard(p *validate.Problems, at string, d measurement.MetricDef, want standard) {
	if d.Formula != want.formula {
		p.Field(at+".formula", "the standard name means formula %s, not %q", want.formula, d.Formula)
	}
	if d.Statistic != want.statistic {
		p.Field(at+".statistic", "the standard name means statistic %q, not %q", want.statistic, d.Statistic)
	}
	if d.Denominator != want.denominator {
		p.Field(at+".denominator", "the standard name means denominator %q, not %q", want.denominator, d.Denominator)
	}
}

// validateCounts checks the counts the document states in more than one place against
// each other.
func (r *Result) validateCounts(p *validate.Problems) {
	// The workload generates at most num_requests; fewer when a horizon or a closed-loop
	// workload ends generation first.
	if n := r.Input.Run.NumRequests; n > 0 && r.Requests.Generated > n {
		p.Field("requests.generated", "is %d but input.run.num_requests is %d: a run generates at most that many", r.Requests.Generated, n)
	}
	if r.Window.Requests > r.Requests.Completed {
		p.Field("window.requests", "is %d but only %d requests completed in the whole run", r.Window.Requests, r.Requests.Completed)
	}
	for _, name := range sortedKeys(r.MetricDefs) {
		if r.MetricDefs[name].Formula.Throughput() && !(r.Window.DurationS > 0) {
			p.Field("window.duration_s", "must be positive: metric %s is a rate over the window", name)
			break
		}
	}
}

// validateBreakdowns checks each split's values. A per-model, per-tenant or per-adapter
// slice splits the result's own metrics, so each of its keys is a metrics key and takes
// that definition; the prefill/decode and session slices also name quantities with no
// measurement formula, so their keys need only be metric-shaped names.
func (r *Result) validateBreakdowns(p *validate.Problems) {
	b := r.Breakdowns
	flat := func(at string, m map[string]float64, defined bool) {
		for _, name := range sortedKeys(m) {
			kat := at + "." + name
			if !slug.MatchString(name) {
				p.Field(kat, "a metric name is a lowercase slug")
			}
			if _, ok := r.MetricDefs[name]; defined && !ok {
				p.Field(kat, "has no entry in metric_defs: a split metric takes the definition of the metric it splits")
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
			at := "breakdowns." + axis + "." + key
			if key == "" {
				p.Field("breakdowns."+axis, "a key must not be empty")
			}
			flat(at, m[key], true)
		}
	}
}

func (w Window) validate(p *validate.Problems) {
	if w.Start == "" {
		p.Field("window.start", "required: what opens the measured window")
	}
	if w.Stop == "" {
		p.Field("window.stop", "required: what closes the measured window")
	}
	if p.FiniteField("window.duration_s", w.DurationS) && w.DurationS < 0 {
		p.Field("window.duration_s", "must not be negative")
	}
	if w.Requests < 0 {
		p.Field("window.requests", "must not be negative")
	}
}

func (q Requests) validate(p *validate.Problems) {
	for name, v := range map[string]int{
		"generated": q.Generated, "rejected": q.Rejected, "injected": q.Injected,
		"completed": q.Completed, "still_queued": q.StillQueued, "still_running": q.StillRunning,
		"dropped_unservable": q.DroppedUnservable, "timed_out": q.TimedOut,
		"routing_rejections": q.RoutingRejections, "gateway_queue_depth": q.GatewayQueueDepth,
		"gateway_queue_rejected": q.GatewayQueueRejected, "gateway_evicted": q.GatewayEvicted,
		"gateway_expired": q.GatewayExpired, "encode_routing_rejections": q.EncodeRoutingRejections,
		"length_capped": q.LengthCapped,
	} {
		if v < 0 {
			p.Field("requests."+name, "must not be negative")
		}
	}
	if t := q.Terminal(); t != q.Injected {
		p.Field("requests", "injected is %d but the terminal buckets sum to %d: every injected request ends in exactly one (INV-1)", q.Injected, t)
	}
	if q.Generated != q.Injected+q.Rejected {
		p.Field("requests.generated", "is %d but injected + rejected is %d", q.Generated, q.Injected+q.Rejected)
	}
	if q.LengthCapped > q.Completed {
		p.Field("requests.length_capped", "is %d but only %d requests completed", q.LengthCapped, q.Completed)
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
	if p.FiniteField("goodput.goodput_rps", g.GoodputRPS) && g.GoodputRPS < 0 {
		p.Field("goodput.goodput_rps", "must not be negative")
	}
	if p.FiniteField("goodput.slo_attainment", g.SLOAttainment) && (g.SLOAttainment < 0 || g.SLOAttainment > 1) {
		p.Field("goodput.slo_attainment", "is a fraction in [0, 1], got %v", g.SLOAttainment)
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

func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
