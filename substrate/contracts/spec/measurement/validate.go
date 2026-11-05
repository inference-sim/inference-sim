package measurement

import (
	"fmt"
	"strings"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
)

// Validate performs field-level validation of a run on its own.
//
// A stated verdict is checked as far as the document allows: the licence is not in it,
// so a run whose own facts make it EXACT may state EXACT or PARTIAL here.
// ValidateWithSources settles which.
func (r *MeasuredRun) Validate() *validate.Problems {
	p := &validate.Problems{}
	r.validateFields(p)
	exact, denied := Assess(r, true), Assess(r, false)
	checkStated(p, r, exact, denied)
	return p
}

// ValidateWithSources validates a run against the licence registry: its licence_ref must
// name an entry, and a stated verdict must equal the computed one.
func (r *MeasuredRun) ValidateWithSources(reg *Registry) *validate.Problems {
	p := &validate.Problems{}
	r.validateFields(p)
	if r.Source.LicenseRef != "" && reg.Lookup(r.Source.LicenseRef) == nil {
		p.Field("source.license_ref", "%q is not in the licence registry", r.Source.LicenseRef)
	}
	a := Assess(r, reg.Permits(r.Source.LicenseRef, UseScore))
	checkStated(p, r, a)
	return p
}

// checkStated rejects a stated verdict or resolution that no allowed assessment yields.
func checkStated(p *validate.Problems, r *MeasuredRun, allowed ...Assessment) {
	if r.Verdict != "" && !r.Verdict.Valid() {
		p.Field("verdict", "%q is not one of %v", r.Verdict, AllVerdicts())
		return
	}
	if r.Resolution != "" && !r.Resolution.Valid() {
		p.Field("resolution", "%q is not one of %v", r.Resolution, AllResolutions())
		return
	}
	if r.Verdict == "" && r.Resolution == "" {
		return
	}
	if r.Verdict == VerdictExact && r.Resolution == "" {
		p.Field("resolution", "required with verdict EXACT: STRICT or RESOLVED")
		return
	}
	for _, a := range allowed {
		if (r.Verdict == "" || r.Verdict == a.Verdict) &&
			(r.Resolution == "" || r.Resolution == a.Resolution) {
			return
		}
	}
	a := allowed[0]
	p.Field("verdict", "states %s%s, but the run computes to %s%s (%s); a verdict is computed, not asserted",
		r.Verdict, suffix(r.Resolution), a.Verdict, suffix(a.Resolution), reasons(a))
}

func suffix(r Resolution) string {
	if r == "" {
		return ""
	}
	return "/" + string(r)
}

func reasons(a Assessment) string {
	if len(a.Reasons) == 0 {
		return "nothing keeps it from EXACT"
	}
	return strings.Join(a.Reasons, "; ")
}

func (r *MeasuredRun) validateFields(p *validate.Problems) {
	if r.Kind != "MeasuredRun" {
		p.Field("kind", "must be %q, got %q", "MeasuredRun", r.Kind)
	}
	if !slug.MatchString(r.Name) {
		p.Field("name", "must be a lowercase slug, got %q", r.Name)
	}
	if r.SchemaVersion != SchemaVersion {
		p.Field("schema_version", "must be %d, got %d", SchemaVersion, r.SchemaVersion)
	}
	r.Source.validate(p)
	if r.Cluster != nil {
		p.Merge("cluster", r.Cluster.Validate())
	}
	if r.Deployment != nil {
		p.Merge("deployment", r.Deployment.Validate())
		if c := r.Cluster; c != nil {
			// The fit uses the node size the run states. A run whose cluster leaves it to
			// the chip's catalog entry (scenario.Cluster.Packaging) is fitted only by a
			// validator that holds the entry, and says so rather than pass unchecked.
			if c.GPUsPerNode == 0 {
				p.Warnf("cluster.gpus_per_node is left to the catalog entry of %q, so the deployment's per-node fit is not checked here",
					c.Hardware)
			}
			p.Merge("deployment", r.Deployment.ValidateAgainstCluster(deployment.ClusterConstraints{
				Nodes: c.Nodes, GPUsPerNode: c.GPUsPerNode, Storage: c.Storage}))
		}
	}
	if r.Router != nil {
		p.Merge("router", r.Router.Validate())
	}
	if r.Workload != nil {
		p.Merge("workload", r.Workload.Validate())
	}
	r.Client.validate(p)
	if len(r.MetricDefs) == 0 {
		p.Field("metric_defs", "a run defines at least one metric")
	}
	for _, name := range sortedKeys(r.MetricDefs) {
		at := "metric_defs." + name
		if !slug.MatchString(name) {
			p.Field(at, "a metric name is a lowercase slug")
		}
		d := r.MetricDefs[name]
		d.validate(p, at)
		if d.Clock == ClockSimulated {
			p.Field(at+".clock", "a measured run's timestamps are a client's or a server's, not simulated")
		}
	}
	r.validatePoints(p)
	r.validateProvenance(p)
	for i, d := range r.Diff {
		d.validate(p, fmt.Sprintf("diff[%d]", i))
	}
	if r.DuplicateOf != "" && r.DuplicateOf == r.Name {
		p.Field("duplicate_of", "a run is not a duplicate of itself")
	}
	if r.Supersedes != "" && r.Supersedes == r.Name {
		p.Field("supersedes", "a run does not supersede itself")
	}
	scanSecrets(p, r)
}

func (s Source) validate(p *validate.Problems) {
	if !s.Kind.Valid() {
		p.Field("source.kind", "%q is not one of %v", s.Kind, AllSourceKinds())
	}
	if s.Adapter.Name == "" {
		p.Field("source.adapter.name", "required")
	}
	if s.Adapter.Version == "" {
		p.Field("source.adapter.version", "required")
	}
	p.SHA256Field("source.adapter.sha256", s.Adapter.SHA256)
	if len(s.Inputs) == 0 {
		p.Field("source.inputs", "a run cites at least one input")
	}
	report := false
	for i, in := range s.Inputs {
		at := fmt.Sprintf("source.inputs[%d]", i)
		if !in.Role.Valid() {
			p.Field(at+".role", "%q is not one of %v", in.Role, AllInputRoles())
		}
		report = report || in.Role == RoleReport
		if in.URL == "" && in.Path == "" {
			p.Field(at, "names its file by url or path")
		}
		if in.URL != "" && !strings.HasPrefix(in.URL, "https://") && !strings.HasPrefix(in.URL, "http://") {
			p.Field(at+".url", "must be an http(s) URL, got %q", in.URL)
		}
		if in.Size < 0 {
			p.Field(at+".size", "must not be negative")
		}
		p.SHA256Field(at+".sha256", in.SHA256)
		utcField(p, at+".fetched_utc", in.FetchedUTC)
	}
	if len(s.Inputs) > 0 && !report {
		p.Field("source.inputs", "one input has role %q: the document the numbers were published in", RoleReport)
	}
	if !s.Extraction.Method.Valid() {
		p.Field("source.extraction.method", "%q is not one of %v", s.Extraction.Method, AllExtractions())
	}
	utcField(p, "source.retrieved_utc", s.RetrievedUTC)
	if !slug.MatchString(s.LicenseRef) {
		p.Field("source.license_ref", "must name a licence registry entry, got %q", s.LicenseRef)
	}
}

// utcField checks an optional timestamp is RFC 3339 in UTC, which is how fetch manifests
// record it and what keeps a re-extraction byte-identical across time zones.
func utcField(p *validate.Problems, at, v string) {
	if v == "" {
		return
	}
	if _, err := time.Parse("2006-01-02T15:04:05Z", v); err != nil {
		p.Field(at, "must be an RFC 3339 UTC time such as 2026-10-15T13:58:06Z, got %q", v)
	}
}

func (c ClientSpec) validate(p *validate.Problems) {
	if c.ImageDigest != "" {
		d := c.ImageDigest
		if i := strings.LastIndex(d, "@"); i >= 0 {
			d = d[i+1:]
		}
		if !strings.HasPrefix(d, "sha256:") || !validate.IsSHA256(strings.TrimPrefix(d, "sha256:")) {
			p.Field("client.image_digest", "must pin an image by sha256:<64 hex>, got %q", c.ImageDigest)
		}
	}
	if c.NumPrompts < 0 {
		p.Field("client.num_prompts", "must not be negative")
	}
	if c.NumWarmups < 0 {
		p.Field("client.num_warmups", "must not be negative")
	}
	if c.Burstiness != nil && p.FiniteField("client.burstiness", *c.Burstiness) && *c.Burstiness <= 0 {
		p.Field("client.burstiness", "must be positive")
	}
	if r := c.RampUp; r != nil {
		if !r.Strategy.Valid() {
			p.Field("client.ramp_up.strategy", "%q is not one of %v", r.Strategy, AllRampStrategies())
		}
		for field, v := range map[string]float64{"start": r.Start, "end": r.End} {
			if p.FiniteField("client.ramp_up."+field, v) && v < 0 {
				p.Field("client.ramp_up."+field, "must not be negative")
			}
		}
	}
	if t := c.Tokenizer; t != nil {
		if t.Repo == "" {
			p.Field("client.tokenizer.repo", "required")
		}
		if t.SHA256 != "" {
			p.SHA256Field("client.tokenizer.sha256", t.SHA256)
		}
	}
	for field, v := range map[string]float64{"duration_s": c.DurationSec, "grace_s": c.GraceSec} {
		if p.FiniteField("client."+field, v) && v < 0 {
			p.Field("client."+field, "must not be negative")
		}
	}
}

// ValidateMetricDef performs field-level validation of one metric definition, for a
// document outside this package that defines its metrics in this vocabulary. Paths are
// relative to the definition, as Merge expects.
func ValidateMetricDef(d MetricDef) *validate.Problems {
	p := &validate.Problems{}
	d.validate(p, defRoot)
	return p.Rebase(defRoot)
}

// defRoot is the placeholder location ValidateMetricDef validates a definition under.
const defRoot = "def"

func (d MetricDef) validate(p *validate.Problems, at string) {
	if d.Formula != "" && !d.Formula.Valid() {
		p.Field(at+".formula", "%q is not one of %v", d.Formula, AllFormulas())
	}
	if d.Statistic != "" && !d.Statistic.Valid() {
		p.Field(at+".statistic", "%q is not one of %v", d.Statistic, AllStatistics())
	}
	if d.Formula.Throughput() && d.Statistic != "" {
		p.Field(at+".statistic", "a throughput is a rate over a window; it has a denominator, not a statistic")
	}
	if d.Formula.Valid() && !d.Formula.Throughput() && d.Denominator != "" {
		p.Field(at+".denominator", "only a throughput has a denominator")
	}
	if d.Statistic.Percentile() && d.PercentileMethod == "" {
		p.Field(at+".percentile_method", "required for a percentile: interpolation changes its value")
	}
	for field, ok := range map[string]bool{
		"population":        d.Population == "" || d.Population.Valid(),
		"denominator":       d.Denominator == "" || d.Denominator.Valid(),
		"clock":             d.Clock == "" || d.Clock.Valid(),
		"start_event":       d.StartEvent == "" || d.StartEvent.Valid(),
		"stop_event":        d.StopEvent == "" || d.StopEvent.Valid(),
		"output_len_source": d.OutputLenSource == "" || d.OutputLenSource.Valid(),
		"percentile_method": d.PercentileMethod == "" || d.PercentileMethod.Valid(),
	} {
		if !ok {
			p.Field(at+"."+field, "not a recognized value")
		}
	}
	switch {
	case !d.Unit.Valid():
		p.Field(at+".unit", "%q is not one of %v", d.Unit, AllMetricUnits())
	case d.Formula.Valid() && d.Unit != d.Formula.Unit():
		p.Field(at+".unit", "formula %s is measured in %s, not %s", d.Formula, d.Formula.Unit(), d.Unit)
	}
	if d.Cite == "" {
		p.Field(at+".cite", "required: where the definition is read from")
	}
}

func (r *MeasuredRun) validatePoints(p *validate.Problems) {
	if len(r.Points) == 0 {
		p.Field("points", "a run with no points measured nothing")
		return
	}
	ids := map[string]bool{}
	mode := loadMode(r.Points[0].Load)
	for i, pt := range r.Points {
		at := fmt.Sprintf("points[%d]", i)
		switch {
		case pt.ID == "":
			p.Field(at+".id", "required")
		case ids[pt.ID]:
			p.Field(at+".id", "duplicate point %q", pt.ID)
		}
		ids[pt.ID] = true
		l := pt.Load
		if l.Concurrency < 0 {
			p.Field(at+".load.concurrency", "must not be negative")
		}
		if p.FiniteField(at+".load.request_rate", l.RequestRate) && l.RequestRate < 0 {
			p.Field(at+".load.request_rate", "must not be negative")
		}
		switch {
		case l.Concurrency == 0 && l.RequestRate == 0:
			p.Field(at+".load", "states a concurrency, a request rate, or both")
		case loadMode(l) != mode:
			p.Field(at+".load", "every point of a run is loaded the same way: by %s, not %s", mode, loadMode(l))
		case i > 0 && !loadLess(r.Points[i-1].Load, l):
			p.Field(at+".load", "points are strictly ordered by increasing load")
		}
		if len(pt.Metrics) == 0 {
			p.Field(at+".metrics", "a point measures at least one metric")
		}
		for _, name := range sortedKeys(pt.Metrics) {
			mat := at + ".metrics." + name
			if _, ok := r.MetricDefs[name]; !ok {
				p.Field(mat, "has no definition in metric_defs")
			}
			if v := pt.Metrics[name].Value; p.FiniteField(mat+".value", v) && v < 0 {
				p.Field(mat+".value", "must not be negative")
			}
		}
		for field, v := range map[string]*int{"completed": pt.Completed, "failed": pt.Failed,
			"preemptions": pt.Preemptions} {
			if v != nil && *v < 0 {
				p.Field(at+"."+field, "must not be negative")
			}
		}
		for field, v := range map[string]*float64{"kv_utilization": pt.KVUtilization,
			"prefix_hit_rate": pt.PrefixHitRate} {
			if v != nil && p.FiniteField(at+"."+field, *v) && (*v < 0 || *v > 1) {
				p.Field(at+"."+field, "must lie in [0, 1], got %v", *v)
			}
		}
		if re := pt.Realized; re != nil {
			for field, v := range map[string]float64{"mean_isl": re.MeanISL, "mean_osl": re.MeanOSL} {
				if p.FiniteField(at+".realized."+field, v) && v < 0 {
					p.Field(at+".realized."+field, "must not be negative")
				}
			}
		}
		if n := pt.Noise; n != nil {
			if n.Seeds < 2 {
				p.Field(at+".noise.seeds", "a spread needs at least two seeds, got %d", n.Seeds)
			}
			if p.FiniteField(at+".noise.cv", n.CV) && n.CV < 0 {
				p.Field(at+".noise.cv", "must not be negative")
			}
		}
	}
	used := map[string]bool{}
	for _, pt := range r.Points {
		for name := range pt.Metrics {
			used[name] = true
		}
	}
	for _, name := range sortedKeys(r.MetricDefs) {
		if !used[name] {
			p.Warnf("metric_defs.%s: defined but measured at no point", name)
		}
	}
}

func loadMode(l Load) string {
	switch {
	case l.Concurrency > 0 && l.RequestRate > 0:
		return "concurrency and request rate"
	case l.RequestRate > 0:
		return "request rate"
	default:
		return "concurrency"
	}
}

// loadLess orders loads by rate, then concurrency. Within one load mode only one of them
// varies in practice, and the order is total either way.
func loadLess(a, b Load) bool {
	if a.RequestRate != b.RequestRate {
		return a.RequestRate < b.RequestRate
	}
	return a.Concurrency < b.Concurrency
}

func (r *MeasuredRun) validateProvenance(p *validate.Problems) {
	seen := map[string]bool{}
	for i, row := range r.Provenance {
		at := fmt.Sprintf("provenance[%d]", i)
		switch {
		case row.Setting == "":
			p.Field(at+".setting", "required")
		case seen[row.Setting]:
			p.Field(at+".setting", "duplicate setting %q", row.Setting)
		}
		seen[row.Setting] = true
		measured := 0
		for field, o := range map[string]*Observed{"report": row.Report, "script": row.Script,
			"artifact": row.Artifact, "blis": row.BLIS} {
			if o == nil {
				continue
			}
			if o.Value == "" {
				p.Field(at+"."+field+".value", "required when the source is listed")
			}
			if field != "blis" {
				measured++
				if o.Ref == "" {
					p.Field(at+"."+field+".ref", "required: where this source states the value")
				}
			}
		}
		observed := measured
		if row.BLIS != nil {
			observed++
		}
		switch row.Verdict {
		case RowMatch, RowMismatch:
			if observed < 2 {
				p.Field(at+".verdict", "%s compares two sources; this row lists %d", row.Verdict, observed)
			}
		case RowUnstated:
			if measured > 1 {
				p.Field(at+".verdict", "UNSTATED means at most one source states the setting; this row lists %d", measured)
			}
		case RowUnverifiable:
			if row.Report == nil || row.Script == nil || row.Artifact != nil {
				p.Field(at+".verdict", "UNVERIFIABLE is a report and a script that disagree with no artifact to decide")
			}
		default:
			p.Field(at+".verdict", "%q is not one of %v", row.Verdict, AllRowVerdicts())
		}
		if row.ResolvedBy != "" && row.Verdict != RowUnstated {
			p.Field(at+".resolved_by", "only an UNSTATED setting is resolved by a rule")
		}
		materiality(p, at, row.Material, row.Bound)
	}
}

func (d DiffEntry) validate(p *validate.Problems, at string) {
	if d.Path == "" {
		p.Field(at+".path", "required: the field that differs")
	}
	if !d.Kind.Valid() {
		p.Field(at+".kind", "%q is not one of %v", d.Kind, AllDiffKinds())
	}
	if d.Removable && d.OwnerRepo == "" {
		p.Field(at+".owner_repo", "required for a removable difference: the repository whose change removes it")
	}
	materiality(p, at, d.Material, d.Bound)
}

func materiality(p *validate.Problems, at string, m Materiality, bound string) {
	switch {
	case !m.Valid():
		p.Field(at+".material", "%q is not one of %v", m, AllMaterialities())
	case m == MaterialBounded && bound == "":
		p.Field(at+".bound", "required when material is bounded")
	case m != MaterialBounded && bound != "":
		p.Field(at+".bound", "only a bounded difference states a bound")
	}
}
