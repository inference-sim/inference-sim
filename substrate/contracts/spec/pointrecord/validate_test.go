package pointrecord

import (
	"math"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/result"
)

var h64 = strings.Repeat("a", 64)

func valid() *Record {
	return &Record{
		Kind: Kind, SchemaVersion: SchemaVersion, PointID: h64, UnitID: strings.Repeat("b", 64),
		Identity: Identity{Source: "inferencex", Model: "dsv41flash", Hardware: "h200", Precision: "fp8",
			Framework: "vllm", BenchmarkType: "single_turn", Recipe: "row:1", Concurrency: 64, MeasuredDate: "2026-10-21"},
		Grade:         G2,
		Substitutions: []Substitution{{Kind: SubEstimated, Subject: "max_model_len", Detail: "leave_unset"}},
		Build:         Build{Release: "0.32.0", Image: "vllm/vllm-openai:v0.32.0", Commit: strings.Repeat("c", 40), Source: "image_label"},
		Simulator: Simulator{Build: "v0.16.0", Engine: result.Engine{Kind: "vllm", Version: "0.32.0"},
			Data: result.Data{Build: "v0.16.0", ReleasedData: true}},
		HarnessDigest: "sha256:" + strings.Repeat("d", 64),
		Figures: []Figure{
			{Metric: "tpot_mean_ms", Predicted: 11.0, Measured: 11.2, PredictedSamples: 640, MeasuredSamples: 640,
				Interval: &Interval{Low: 10.8, High: 11.3, Level: 0.95, Replications: 5}},
			{Metric: "ttft_mean_ms", Predicted: 850, Measured: 880, PredictedSamples: 640, MeasuredSamples: 640},
		},
		Settings: []Setting{
			{Arg: "max_model_len", Measured: nil, BLIS: nil, Match: MatchEstimated, Rule: "leave_unset"},
			{Arg: "max_num_seqs", Measured: 256, BLIS: 256, Match: MatchExact},
		},
		Facets: Facets{SweepID: "s1", Scenario: "dsv41-h200", Label: "8k1k", Workload: "8192:1024", Model: "deepseek-v4.1-flash",
			GPU: "h200", Precision: "fp8", Framework: "vllm", Serving: "aggregated", SpecMethod: "none",
			Parallelism: Parallelism{TP: 8, PP: 1, DP: 1, EP: 1}},
	}
}

func ptr(v float64) *float64 { return &v }

// replay is a well-formed Replay, adjusted by f.
func replay(f func(*Replay)) *Replay {
	r := &Replay{Client: "aiperf", ClientVersion: "1", Scenario: "s", Corpus: "c", Concurrency: 1,
		CorpusSHA256: h64}
	f(r)
	return r
}

func TestAWellFormedReplayPasses(t *testing.T) {
	r := valid()
	r.Replay = replay(func(x *Replay) {
		x.DurationS = 900
		x.TrajectoryStart = &TrajectoryStart{MinRatio: ptr(0), MaxRatio: ptr(1)}
	})
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a well-formed replay was refused:\n%s", p.Error())
	}
}

func TestValidRecordPasses(t *testing.T) {
	if p := valid().Validate(); !p.OK() {
		t.Fatalf("the fixture must validate:\n%s", p.Error())
	}
	g5 := valid()
	g5.Grade, g5.Figures, g5.Substitutions = G5, nil, nil
	g5.G5Reason = &G5Reason{Code: RouterUnsupported, Subject: "dynamo"}
	if p := g5.Validate(); !p.OK() {
		t.Fatalf("a G5 record must validate:\n%s", p.Error())
	}
}

func TestEachFailureModeIsRefused(t *testing.T) {
	cases := []struct {
		name   string
		break_ func(*Record)
		path   string
		msg    string
	}{
		{"wrong kind", func(r *Record) { r.Kind = "Point" }, "kind", "must be"},
		{"short point id", func(r *Record) { r.PointID = "abc" }, "point_id", "64 lowercase"},
		{"no unit id", func(r *Record) { r.UnitID = "" }, "unit_id", "64 lowercase"},
		{"identity without model", func(r *Record) { r.Identity.Model = "" }, "identity.model", "required"},
		{"bad date", func(r *Record) { r.Identity.MeasuredDate = "21/10/2026" }, "identity.measured_date", "YYYY-MM-DD"},
		{"unknown grade", func(r *Record) { r.Grade = "G6" }, "grade", "is not G1"},
		{"grade below its substitutions", func(r *Record) { r.Grade = G1 }, "grade", "at least G2"},
		{"G2 without a substitution", func(r *Record) { r.Substitutions = nil }, "substitutions", "names the substitution"},
		{"G3 without the other release", func(r *Record) { r.Grade = G3 }, "substitutions", "other release"},
		{"G4 without changed functions", func(r *Record) {
			r.Grade = G4
			r.Substitutions = []Substitution{{Kind: SubOtherRelease, Subject: "0.31.0"}}
		}, "substitutions", "changed functions"},
		{"G5 without a reason", func(r *Record) { r.Grade, r.Figures = G5, nil }, "g5_reason", "required"},
		{"G5 with an unlisted reason", func(r *Record) {
			r.Grade, r.Figures, r.G5Reason = G5, nil, &G5Reason{Code: "blis_refused"}
		}, "g5_reason.code", "COV-5"},
		{"G5 with figures", func(r *Record) { r.Grade, r.G5Reason = G5, &G5Reason{Code: NotInCatalog} }, "figures", "not scored"},
		{"reason on a graded point", func(r *Record) { r.G5Reason = &G5Reason{Code: NotInCatalog} }, "g5_reason", "only a G5"},
		{"unknown substitution", func(r *Record) { r.Substitutions[0].Kind = "guess" }, "substitutions[0].kind", "§11.3"},
		{"unnamed substitution", func(r *Record) { r.Substitutions[0].Subject = "" }, "substitutions[0].subject", "required"},
		{"substitutions unsorted", func(r *Record) {
			r.Substitutions = append(r.Substitutions, Substitution{Kind: SubCoefficientMethod, Subject: "x"})
		}, "substitutions[1]", "sorted"},
		{"build without image", func(r *Record) { r.Build.Image = "" }, "build.image", "required"},
		{"short commit", func(r *Record) { r.Build.Commit = "abc123" }, "build.commit", "40-character"},
		{"commit without source", func(r *Record) { r.Build.Source = "" }, "build.source", "required with a commit"},
		{"no simulator build", func(r *Record) { r.Simulator.Build = "" }, "simulator.build", "required"},
		{"bad harness digest", func(r *Record) { r.HarnessDigest = "latest" }, "harness_digest", "sha256:"},
		{"graded without figures", func(r *Record) { r.Figures = nil }, "figures", "at least one"},
		{"unregistered metric", func(r *Record) { r.Figures[0].Metric = "mean_tpot" }, "figures[0].metric", "DEV-8"},
		{"figures unsorted", func(r *Record) { r.Figures[0], r.Figures[1] = r.Figures[1], r.Figures[0] }, "figures[1].metric", "sorted"},
		{"non-finite figure", func(r *Record) { r.Figures[1].Predicted = math.NaN() }, "figures[1].predicted", "finite"},
		{"figure without samples", func(r *Record) { r.Figures[1].MeasuredSamples = 0 }, "figures[1]", "samples"},
		{"interval misses the figure", func(r *Record) { r.Figures[0].Interval.High = 10.9 }, "figures[0].interval", "does not contain"},
		{"interval of one replication", func(r *Record) { r.Figures[0].Interval.Replications = 1 }, "figures[0].interval.replications", "at least two"},
		{"setting with a float", func(r *Record) { r.Settings[1].Measured = 0.9 }, "settings[1].measured", "float as printed"},
		{"estimated without rule", func(r *Record) { r.Settings[0].Rule = "" }, "settings[0].rule", "required"},
		{"conflict without values", func(r *Record) { r.Settings[1].Match = MatchConflict }, "settings[1].values", "at least two"},
		{"unknown match", func(r *Record) { r.Settings[1].Match = "close" }, "settings[1].match", "is not"},
		{"settings unsorted", func(r *Record) { r.Settings[0].Arg = "z" }, "settings[1].arg", "sorted"},
		{"replay without corpus digest", func(r *Record) {
			r.Replay = &Replay{Client: "aiperf", Scenario: "s", Corpus: "c", Concurrency: 1}
		}, "replay.corpus_sha256", "64 lowercase"},
		{"replay negative duration", func(r *Record) { r.Replay = replay(func(x *Replay) { x.DurationS = -5 }) },
			"replay.duration_s", "negative"},
		{"replay NaN duration", func(r *Record) { r.Replay = replay(func(x *Replay) { x.DurationS = math.NaN() }) },
			"replay.duration_s", "finite"},
		{"replay min ratio above one", func(r *Record) {
			r.Replay = replay(func(x *Replay) { x.TrajectoryStart = &TrajectoryStart{MinRatio: ptr(2.0)} })
		}, "replay.trajectory_start.min_ratio", "[0, 1]"},
		{"replay max ratio below zero", func(r *Record) {
			r.Replay = replay(func(x *Replay) { x.TrajectoryStart = &TrajectoryStart{MaxRatio: ptr(-1.0)} })
		}, "replay.trajectory_start.max_ratio", "[0, 1]"},
		{"replay min above max", func(r *Record) {
			r.Replay = replay(func(x *Replay) { x.TrajectoryStart = &TrajectoryStart{MinRatio: ptr(0.8), MaxRatio: ptr(0.2)} })
		}, "replay.trajectory_start", "exceeds"},
		{"replay NaN ratio", func(r *Record) {
			r.Replay = replay(func(x *Replay) { x.TrajectoryStart = &TrajectoryStart{MinRatio: ptr(math.NaN())} })
		}, "replay.trajectory_start.min_ratio", "finite"},
		{"facets without sweep", func(r *Record) { r.Facets.SweepID = "" }, "facets", "sweep_id"},
		{"two notes on one subject", func(r *Record) {
			r.Notes = []Note{{Subject: "a", Text: "x"}, {Subject: "a", Text: "y"}}
		}, "notes[1].subject", "two notes"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			r := valid()
			c.break_(r)
			p := r.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}

// TestEveryDesignReasonIsAccepted pins the closed list of design §11.4.
func TestEveryDesignReasonIsAccepted(t *testing.T) {
	for _, c := range []G5Code{"code_not_visible", "router_unsupported", "value_not_recorded", "field_not_modelled",
		"field_unclassified", "engine_unsupported", "not_in_catalog", "field_not_in_release", "build_unparseable"} {
		if !c.Valid() {
			t.Errorf("%s is a reason of design §11.4", c)
		}
	}
	if len(g5Codes) != 9 {
		t.Errorf("design §11.4 lists nine reasons; the code lists %d", len(g5Codes))
	}
}

// The three conditions design 11.3 lists beside the earlier ones, and resolved_differs, each
// name what the point must say.
func TestTheClosenessKindsAreAcceptedWithTheirKeys(t *testing.T) {
	for _, s := range []Substitution{
		{Kind: SubClientDifference, Subject: "send_gap", Detail: "the next request is sent after the response stream closes"},
		{Kind: SubLaterRecord, Subject: "cluster.gpus_per_node", Record: "runners.yaml at 0123456789abcdef0123456789abcdef01234567", Date: "2026-03-02"},
		{Kind: SubRouterDiffers, Subject: "dynamo-round-robin", Policy: "round-robin", Evidence: "catalog/routers/dynamo-round-robin.yaml"},
		{Kind: SubResolvedDiffers, Subject: "max_num_batched_tokens", Detail: "the run resolved 8192, the engine resolves 2048"},
	} {
		r := valid()
		r.Substitutions = []Substitution{s}
		if p := r.Validate(); !p.OK() {
			t.Errorf("%s: %s", s.Kind, p.Error())
		}
		// Each is a G2 condition: a G1 record naming it is refused.
		r.Grade = G1
		if p := r.Validate(); p.OK() {
			t.Errorf("%s: a G1 record naming it was accepted", s.Kind)
		}
		if got := r.Floor(); got != G2 {
			t.Errorf("%s: floor %s, want G2", s.Kind, got)
		}
	}
}

// A known-bad record per kind and per key (SUB-2): what the point must name, a key on a kind
// that does not take it, and a malformed date.
func TestTheClosenessKindsAreRefusedWithoutWhatTheyName(t *testing.T) {
	later := Substitution{Kind: SubLaterRecord, Subject: "cluster.gpus_per_node", Record: "runners.yaml at abc", Date: "2026-03-02"}
	router := Substitution{Kind: SubRouterDiffers, Subject: "dynamo", Policy: "round-robin", Evidence: "catalog/routers/dynamo.yaml"}
	client := Substitution{Kind: SubClientDifference, Subject: "send_gap", Detail: "x"}
	resolved := Substitution{Kind: SubResolvedDiffers, Subject: "s", Detail: "x"}
	with := func(s Substitution, edit func(*Substitution)) func(*Record) {
		edit(&s)
		return func(r *Record) { r.Substitutions = []Substitution{s} }
	}
	for _, c := range []struct {
		name   string
		break_ func(*Record)
		path   string
		msg    string
	}{
		{"later_record without a record", with(later, func(s *Substitution) { s.Record = "" }), "substitutions[0].record", "required for later_record"},
		{"later_record without a date", with(later, func(s *Substitution) { s.Date = "" }), "substitutions[0].date", "required for later_record"},
		{"later_record with a malformed date", with(later, func(s *Substitution) { s.Date = "2 March 2026" }), "substitutions[0].date", "YYYY-MM-DD"},
		{"later_record without the field", with(later, func(s *Substitution) { s.Subject = "" }), "substitutions[0].subject", "required"},
		{"later_record naming a policy", with(later, func(s *Substitution) { s.Policy = "round-robin" }), "substitutions[0].policy", "not a key of later_record"},
		{"router_differs without a policy", with(router, func(s *Substitution) { s.Policy = "" }), "substitutions[0].policy", "required for router_differs"},
		{"router_differs without evidence", with(router, func(s *Substitution) { s.Evidence = "" }), "substitutions[0].evidence", "required for router_differs"},
		{"router_differs without the router", with(router, func(s *Substitution) { s.Subject = "" }), "substitutions[0].subject", "required"},
		{"router_differs naming a record", with(router, func(s *Substitution) { s.Record = "r" }), "substitutions[0].record", "not a key of router_differs"},
		{"router_differs naming a date", with(router, func(s *Substitution) { s.Date = "2026-03-02" }), "substitutions[0].date", "not a key of router_differs"},
		{"client_difference without a detail", with(client, func(s *Substitution) { s.Detail = "" }), "substitutions[0].detail", "required for client_difference"},
		{"client_difference naming evidence", with(client, func(s *Substitution) { s.Evidence = "e" }), "substitutions[0].evidence", "not a key of client_difference"},
		{"resolved_differs without a detail", with(resolved, func(s *Substitution) { s.Detail = "" }), "substitutions[0].detail", "required for resolved_differs"},
		{"an estimate naming a policy", func(r *Record) { r.Substitutions[0].Policy = "p" }, "substitutions[0].policy", "not a key of estimated"},
	} {
		t.Run(c.name, func(t *testing.T) {
			r := valid()
			c.break_(r)
			p := r.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}

// The kinds the record lists are the ones the dashboard's grading code emits for G2 (its
// G2_KINDS), no fewer and no more. The list is restated here because the dashboard is another
// repository; a change to either side is a change to the other.
func TestTheG2KindsAreTheOnesTheDashboardEmits(t *testing.T) {
	var g2 []string
	for k, g := range substitutionGrade {
		if g == G2 {
			g2 = append(g2, string(k))
		}
	}
	sort.Strings(g2)
	want := []string{"client_difference", "coefficient_method", "coefficient_range", "estimated", "execution_mode",
		"interim_status", "kernel_hash", "later_record", "resolved_differs", "router_differs", "value_class"}
	if !reflect.DeepEqual(g2, want) {
		t.Errorf("G2 kinds %v, want %v", g2, want)
	}
}
