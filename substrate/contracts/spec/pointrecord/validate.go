package pointrecord

import (
	"fmt"
	"regexp"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/check"
	"github.com/inference-sim/blis-substrate/contracts/metrics"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

var (
	hex64  = regexp.MustCompile(`^[0-9a-f]{64}$`)
	hex40  = regexp.MustCompile(`^[0-9a-f]{40}$`)
	digest = regexp.MustCompile(`^sha256:[0-9a-f]{64}$`)
)

var g5Codes = map[G5Code]bool{CodeNotVisible: true, RouterUnsupported: true, ValueNotRecorded: true,
	FieldNotModelled: true, FieldUnclassified: true, EngineUnsupported: true, NotInCatalog: true,
	FieldNotInRelease: true, BuildUnparseable: true}

// Valid reports whether c is one of the reasons of design §11.4.
func (c G5Code) Valid() bool { return g5Codes[c] }

// substitutionGrade is the grade each substitution sets at least.
var substitutionGrade = map[SubstitutionKind]Grade{
	SubCoefficientMethod: G2, SubCoefficientRange: G2, SubExecutionMode: G2, SubKernelHash: G2,
	SubValueClass: G2, SubEstimated: G2, SubInterimStatus: G2,
	SubResolvedDiffers: G2, SubClientDifference: G2, SubLaterRecord: G2, SubRouterDiffers: G2,
	SubOtherRelease: G3, SubChangedFunctions: G4, SubNightlyBuild: G4,
}

var matches = map[Match]bool{MatchExact: true, MatchMapped: true, MatchInert: true, MatchEstimated: true,
	MatchDiffers: true, MatchAbsentInBLIS: true, MatchUnreadable: true, MatchConflict: true}

// Floor returns the lowest grade the record's substitutions allow: the grade is the
// highest-numbered whose conditions apply (design §11.3), so a record stating a lower one
// hides a substitution.
func (r *Record) Floor() Grade {
	g := G1
	for _, s := range r.Substitutions {
		if sg := substitutionGrade[s.Kind]; sg > g {
			g = sg
		}
	}
	return g
}

// Validate checks a record on its own.
func (r *Record) Validate() *validate.Problems {
	p := &validate.Problems{}
	if r.Kind != Kind {
		p.Field("kind", "must be %q, got %q", Kind, r.Kind)
	}
	if r.SchemaVersion != SchemaVersion {
		p.Field("schema_version", "must be %d, got %d", SchemaVersion, r.SchemaVersion)
	}
	if !hex64.MatchString(r.PointID) {
		p.Field("point_id", "must be 64 lowercase hex characters, got %q", r.PointID)
	}
	if !hex64.MatchString(r.UnitID) {
		p.Field("unit_id", "must be 64 lowercase hex characters, got %q", r.UnitID)
	}
	r.Identity.validate(p)
	r.validateGrade(p)
	r.Build.validate(p)
	if r.Simulator.Build == "" {
		p.Field("simulator.build", "required: the simulator release tag or commit that scored the point")
	}
	if r.Simulator.Engine.Kind == "" || r.Simulator.Engine.Version == "" {
		p.Field("simulator.engine", "states the kind and version the simulator runs")
	}
	if !digest.MatchString(r.HarnessDigest) {
		p.Field("harness_digest", "must be sha256:<64 lowercase hex>, got %q", r.HarnessDigest)
	}
	r.validateFigures(p)
	for i, s := range r.Settings {
		at := fmt.Sprintf("settings[%d]", i)
		if s.Arg == "" {
			p.Field(at+".arg", "required")
		} else if i > 0 && s.Arg <= r.Settings[i-1].Arg {
			p.Field(at+".arg", "settings are sorted by arg, each once")
		}
		if !matches[s.Match] {
			p.Field(at+".match", "%q is not a match", s.Match)
		}
		scalar(p, at+".measured", s.Measured)
		scalar(p, at+".blis", s.BLIS)
		switch {
		case s.Match == MatchConflict && len(s.Values) < 2:
			p.Field(at+".values", "a conflict lists every value found, at least two")
		case s.Match == MatchEstimated && s.Rule == "":
			p.Field(at+".rule", "required: the estimation rule that set the value")
		case s.Match == MatchMapped && s.Field == "":
			p.Field(at+".field", "required: the scenario field a mapped value was written to")
		}
		if s.Occurrences == 1 || s.Occurrences < 0 {
			p.Field(at+".occurrences", "is stated only when the value was found at least twice")
		}
	}
	for i, q := range r.Resolved {
		at := fmt.Sprintf("resolved[%d]", i)
		if q.Quantity == "" {
			p.Field(at+".quantity", "required")
		} else if i > 0 && q.Quantity <= r.Resolved[i-1].Quantity {
			p.Field(at+".quantity", "resolved quantities are sorted, each once")
		}
		if !matches[q.Match] {
			p.Field(at+".match", "%q is not a match", q.Match)
		}
		scalar(p, at+".measured", q.Measured)
		scalar(p, at+".blis", q.BLIS)
	}
	if rp := r.Replay; rp != nil {
		if rp.Client == "" || rp.Scenario == "" || rp.Corpus == "" {
			p.Field("replay", "states client, scenario and corpus")
		}
		if rp.Concurrency < 1 {
			p.Field("replay.concurrency", "must be at least 1")
		}
		if !hex64.MatchString(rp.CorpusSHA256) {
			p.Field("replay.corpus_sha256", "must be 64 lowercase hex characters")
		}
		// duration_s is omitempty, so zero reads back as unstated; a stated one is a
		// finite number of seconds, as the workload's benchmark_duration_s is.
		if p.FiniteField("replay.duration_s", rp.DurationS) && rp.DurationS < 0 {
			p.Field("replay.duration_s", "must not be negative")
		}
		if t := rp.TrajectoryStart; t != nil {
			workload.CheckStartWindow(p, "replay.trajectory_start", t.MinRatio, t.MaxRatio)
		}
	}
	f := r.Facets
	if f.SweepID == "" || f.Scenario == "" || f.Model == "" {
		p.Field("facets", "states sweep_id, scenario and model")
	}
	seen := map[string]bool{}
	for i, n := range r.Notes {
		at := fmt.Sprintf("notes[%d]", i)
		if n.Subject == "" || n.Text == "" {
			p.Field(at, "states a subject and its text")
		}
		if seen[n.Subject] {
			p.Field(at+".subject", "%q has two notes", n.Subject)
		}
		seen[n.Subject] = true
	}
	return p
}

func (id Identity) validate(p *validate.Problems) {
	for f, v := range map[string]string{"source": id.Source, "model": id.Model, "hardware": id.Hardware,
		"framework": id.Framework, "benchmark_type": id.BenchmarkType, "recipe": id.Recipe} {
		if v == "" {
			p.Field("identity."+f, "required")
		}
	}
	if id.Concurrency < 1 {
		p.Field("identity.conc", "must be at least 1")
	}
	if _, err := time.Parse("2006-01-02", id.MeasuredDate); err != nil {
		p.Field("identity.measured_date", "must be YYYY-MM-DD, got %q", id.MeasuredDate)
	}
}

func (r *Record) validateGrade(p *validate.Problems) {
	switch r.Grade {
	case G1, G2, G3, G4:
		if r.G5Reason != nil {
			p.Field("g5_reason", "only a G5 point has a reason it was not scored")
		}
		if floor := r.Floor(); r.Grade < floor {
			p.Field("grade", "is %s but the substitutions set it at least %s: a point takes the highest-numbered grade whose conditions apply", r.Grade, floor)
		}
		if r.Grade == G2 && r.Floor() != G2 {
			p.Field("substitutions", "a G2 point names the substitution that makes it G2")
		}
		if (r.Grade == G3 || r.Grade == G4) && !r.has(SubOtherRelease) && !r.has(SubNightlyBuild) {
			p.Field("substitutions", "a %s point names its other release or its nightly or custom build", r.Grade)
		}
		if r.Grade == G4 && r.Floor() != G4 {
			p.Field("substitutions", "a G4 point names the changed functions or the build")
		}
	case G5:
		switch {
		case r.G5Reason == nil:
			p.Field("g5_reason", "required for a G5 point: one of the reasons of design §11.4")
		case !r.G5Reason.Code.Valid():
			p.Field("g5_reason.code", "%q is not a reason of design §11.4 (COV-5)", r.G5Reason.Code)
		}
		if len(r.Figures) > 0 {
			p.Field("figures", "a G5 point is not scored and has no figures")
		}
	default:
		p.Field("grade", "%q is not G1, G2, G3, G4 or G5", r.Grade)
	}
	for i, s := range r.Substitutions {
		at := fmt.Sprintf("substitutions[%d]", i)
		if _, ok := substitutionGrade[s.Kind]; !ok {
			p.Field(at+".kind", "%q is not a condition of design §11.3", s.Kind)
		}
		if s.Subject == "" {
			p.Field(at+".subject", "required: what was substituted")
		}
		s.validateKeys(p, at)
		if i > 0 {
			prev := r.Substitutions[i-1]
			if s.Kind < prev.Kind || (s.Kind == prev.Kind && s.Subject <= prev.Subject) {
				p.Field(at, "substitutions are sorted by kind and subject, each once")
			}
		}
	}
}

// validateKeys checks the keys a kind names (design 11.3). A later_record names the field, the
// record and its date; a router_differs the router the run used, the policy BLIS used and the
// router table's evidence; a client_difference and a resolved_differs say what differed. A key
// belongs to its own kind only, so a record cannot carry a router's evidence on a coefficient.
func (s Substitution) validateKeys(p *validate.Problems, at string) {
	need := map[string]*string{}
	switch s.Kind {
	case SubLaterRecord:
		need["record"], need["date"] = &s.Record, &s.Date
	case SubRouterDiffers:
		need["policy"], need["evidence"] = &s.Policy, &s.Evidence
	case SubClientDifference, SubResolvedDiffers:
		need["detail"] = &s.Detail
	}
	why := map[string]string{
		"record":   "the record the value was taken from, with its revision",
		"date":     "the date of that record, YYYY-MM-DD",
		"policy":   "the llm-d policy BLIS used in place of the run's router",
		"evidence": "the router table's evidence for the match",
		"detail":   "what differed",
	}
	for _, k := range []string{"record", "date", "policy", "evidence", "detail"} {
		if v, ok := need[k]; ok {
			if *v == "" {
				p.Field(at+"."+k, "required for %s: %s", s.Kind, why[k])
			}
		}
	}
	for k, v := range map[string]string{"record": s.Record, "date": s.Date, "policy": s.Policy, "evidence": s.Evidence} {
		if _, ok := need[k]; !ok && v != "" {
			p.Field(at+"."+k, "is not a key of %s", s.Kind)
		}
	}
	if s.Kind == SubLaterRecord && s.Date != "" {
		if _, err := time.Parse("2006-01-02", s.Date); err != nil {
			p.Field(at+".date", "must be YYYY-MM-DD, got %q", s.Date)
		}
	}
}

func (r *Record) has(k SubstitutionKind) bool {
	for _, s := range r.Substitutions {
		if s.Kind == k {
			return true
		}
	}
	return false
}

func (b Build) validate(p *validate.Problems) {
	if b.Image == "" {
		p.Field("build.image", "required: the engine image that measured the point")
	}
	if b.Commit != "" && !hex40.MatchString(b.Commit) {
		p.Field("build.commit", "must be a full 40-character lowercase commit, got %q", b.Commit)
	}
	switch b.Source {
	case "", "image_label", "label_by_tag", "server_log_banner", "image_tag", "release_tag":
	default:
		p.Field("build.source", "%q is not image_label, label_by_tag, server_log_banner, image_tag or release_tag", b.Source)
	}
	if b.Commit != "" && b.Source == "" {
		p.Field("build.source", "required with a commit: where it was read")
	}
}

func (r *Record) validateFigures(p *validate.Problems) {
	if r.Grade != G5 && r.Grade != "" && len(r.Figures) == 0 {
		p.Field("figures", "a graded point has at least one figure")
	}
	for i, f := range r.Figures {
		at := fmt.Sprintf("figures[%d]", i)
		if _, ok := metrics.Lookup(f.Metric); !ok {
			p.Field(at+".metric", "%q is not in the metric registry: both sides must name one definition (DEV-8)", f.Metric)
		}
		if i > 0 && f.Metric <= r.Figures[i-1].Metric {
			p.Field(at+".metric", "figures are sorted by metric, each once")
		}
		for name, v := range map[string]float64{"predicted": f.Predicted, "measured": f.Measured} {
			if p.FiniteField(at+"."+name, v) && v < 0 {
				p.Field(at+"."+name, "must not be negative")
			}
		}
		if f.PredictedSamples < 1 || f.MeasuredSamples < 1 {
			p.Field(at, "states the samples behind each side, at least one")
		}
		if iv := f.Interval; iv != nil {
			if !(iv.Low <= f.Predicted && f.Predicted <= iv.High) {
				p.Field(at+".interval", "[%v, %v] does not contain the predicted figure %v", iv.Low, iv.High, f.Predicted)
			}
			if !(iv.Level > 0 && iv.Level < 1) {
				p.Field(at+".interval.level", "is a confidence level in (0, 1), got %v", iv.Level)
			}
			if iv.Replications < 2 {
				p.Field(at+".interval.replications", "an interval needs at least two replications")
			}
		}
	}
}

// scalar records a problem unless v is null, a string, an integer or a boolean: the
// values a record carries are as the run printed them, and a float is kept as its text.
func scalar(p *validate.Problems, at string, v any) {
	switch v.(type) {
	case nil, string, bool, int, int64, uint64:
	default:
		p.Field(at, "is a %T: a value is a string (a float as printed), an integer, a boolean or null", v)
	}
}
