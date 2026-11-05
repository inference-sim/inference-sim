package result

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

func ptr[T any](v T) *T { return &v }

// valid returns a Result every validator accepts; each case below breaks one thing.
func valid() *Result {
	return &Result{
		Kind:          Kind,
		SchemaVersion: SchemaVersion,
		Name:          "llama-h100-tp2.c64-s1",
		Input: Input{
			Scenario: scenario.Scenario{
				Kind: "Scenario", Name: "llama-3.1-8b-on-h100", Model: "llama-3.1-8b-instruct",
				Coefficients: []string{"cost-model-primitives"}, EngineVersion: "0.29.0",
				Cluster: scenario.Cluster{Hardware: "h100", Nodes: 1, GPUsPerNode: 8},
			},
			Deployment: deployment.Deployment{
				Kind: "Deployment", Name: "tp2-dp4",
				Pools: []deployment.Pool{{
					Role: deployment.RoleColocated, Nodes: 1,
					Parallel: deployment.Parallelism{TP: 2, PP: 1, DP: 4},
				}},
			},
			Run: Run{
				Load:        measurement.Load{Concurrency: 64},
				NumRequests: 2000,
				Seed:        1,
				Warmup:      &Warmup{Requests: 128},
				Workload:    &workload.Binding{Shape: "chatbot"},
			},
		},
		Revisions: Revisions{BLIS: "v0.13.0", BLISSchemas: "v0.8.0"},
		MetricDefs: map[string]measurement.MetricDef{
			"ttft_p99_ms": {
				Formula: measurement.FormulaTTFT, Statistic: "p99",
				Population: measurement.PopulationPerRequest, Clock: measurement.ClockSimulated,
				PercentileMethod: measurement.PercentileNumpyLinear, Unit: measurement.UnitMs,
				Cite: "sim/metrics_measured.go TTFTP99Ms",
			},
		},
		Window:  Window{Start: "first_send_after_warmup", Stop: "last_completion", DurationS: 300, Requests: 1990},
		Metrics: map[string]float64{"ttft_p99_ms": 431.5},
		Requests: Requests{
			Generated: 2000, Injected: 2000, Completed: 1990, StillRunning: 8, GatewayQueueDepth: 2,
		},
	}
}

func TestValidResultPasses(t *testing.T) {
	if p := valid().Validate(); !p.OK() {
		t.Fatalf("the fixture must validate:\n%s", p.Error())
	}
}

// TestEachFailureModeIsRefused breaks one rule per case and expects a problem at exactly
// the named path whose message contains msg, so a case cannot pass on another rule's
// finding.
func TestEachFailureModeIsRefused(t *testing.T) {
	rerun := func(r *Result, changes ...RevisionChange) {
		r.RerunOf = &Reference{Name: "old.c64-s1", SHA256: strings.Repeat("a", 64)}
		r.RevisionChanges = changes
	}
	def := func(r *Result, edit func(d *measurement.MetricDef)) {
		d := r.MetricDefs["ttft_p99_ms"]
		edit(&d)
		r.MetricDefs["ttft_p99_ms"] = d
	}
	goodput := func(g Goodput) func(r *Result) {
		return func(r *Result) {
			if g.Targets == nil {
				g.Targets = map[string]workload.SLODimTargets{"a": {TTFTMs: 500}}
			}
			r.Goodput = &g
		}
	}
	nan := math.NaN()
	cases := []struct {
		name   string
		break_ func(r *Result)
		path   string
		msg    string
	}{
		{"wrong kind", func(r *Result) { r.Kind = "SimResult" }, "kind", "must be"},
		{"wrong schema version", func(r *Result) { r.SchemaVersion = 2 }, "schema_version", "must be"},
		{"name not a slug", func(r *Result) { r.Name = "Has Spaces" }, "name", "slug"},
		{"embedded scenario invalid", func(r *Result) { r.Input.Scenario.Kind = "" }, "input.scenario.kind", "Scenario"},
		{"deployment does not fill its cluster", func(r *Result) { r.Input.Scenario.Cluster.Nodes = 2 }, "input.deployment.pools", "sum to 1"},
		{"both concurrency and rate", func(r *Result) { r.Input.Run.Load.RequestRate = 5 }, "input.run.load", "not both"},
		{"neither concurrency nor rate", func(r *Result) { r.Input.Run.Load.Concurrency = 0 }, "input.run.load", "states a concurrency"},
		{"negative concurrency", func(r *Result) { r.Input.Run.Load.Concurrency = -1 }, "input.run.load.concurrency", "negative"},
		{"negative request rate", func(r *Result) {
			r.Input.Run.Load = measurement.Load{RequestRate: -1}
		}, "input.run.load.request_rate", "negative"},
		{"non-finite request rate", func(r *Result) {
			r.Input.Run.Load = measurement.Load{RequestRate: math.Inf(1)}
		}, "input.run.load.request_rate", "finite"},
		{"no requests", func(r *Result) { r.Input.Run.NumRequests = 0 }, "input.run.num_requests", "positive"},
		{"non-positive horizon", func(r *Result) { r.Input.Run.HorizonS = ptr(0.0) }, "input.run.horizon_s", "positive"},
		{"non-finite horizon", func(r *Result) { r.Input.Run.HorizonS = ptr(nan) }, "input.run.horizon_s", "finite"},
		{"warmup by both", func(r *Result) { r.Input.Run.Warmup.DurationS = 10 }, "input.run.warmup", "not both"},
		{"empty warmup", func(r *Result) { r.Input.Run.Warmup = &Warmup{} }, "input.run.warmup", "omit it"},
		{"negative warmup requests", func(r *Result) { r.Input.Run.Warmup = &Warmup{Requests: -1} }, "input.run.warmup.requests", "negative"},
		{"warmup takes every request", func(r *Result) { r.Input.Run.Warmup = &Warmup{Requests: 2000} }, "input.run.warmup.requests", "leaves requests"},
		{"negative warmup duration", func(r *Result) { r.Input.Run.Warmup = &Warmup{DurationS: -1} }, "input.run.warmup.duration_s", "negative"},
		{"non-finite warmup duration", func(r *Result) { r.Input.Run.Warmup = &Warmup{DurationS: nan} }, "input.run.warmup.duration_s", "finite"},
		{"workload with two arms", func(r *Result) {
			r.Input.Run.Workload.Trace = &workload.TraceRef{Data: "t.csv"}
		}, "input.run.workload", "exactly one"},
		{"no blis revision", func(r *Result) { r.Revisions.BLIS = "" }, "revisions.blis", "required"},
		{"no schema revision", func(r *Result) { r.Revisions.BLISSchemas = "" }, "revisions.blis_schemas", "required"},
		{"engine without kind", func(r *Result) { r.Revisions.Engine = &Engine{UpstreamVersion: "0.29.0"} }, "revisions.engine.kind", "required"},
		{"engine without upstream version", func(r *Result) { r.Revisions.Engine = &Engine{Kind: "vllm"} }, "revisions.engine.upstream_version", "required"},
		{"engine models another release than the scenario", func(r *Result) {
			r.Revisions.Engine = &Engine{Kind: "vllm", UpstreamVersion: "0.32.0"}
		}, "revisions.engine.upstream_version", "engine_version"},
		{"empty catalog revision", func(r *Result) { r.Revisions.Catalog = &DataRepo{} }, "revisions.catalog", "release, a revision"},
		{"revision changes without rerun_of", func(r *Result) {
			r.RevisionChanges = []RevisionChange{{Component: "blis", From: "v1", To: "v0.13.0"}}
		}, "revision_changes", "set rerun_of"},
		{"rerun_of without sha256", func(r *Result) { r.RerunOf = &Reference{Name: "old.c64-s1"} }, "rerun_of.sha256", "required"},
		{"rerun_of name not a slug", func(r *Result) {
			r.RerunOf = &Reference{Name: "Old Run", SHA256: strings.Repeat("a", 64)}
		}, "rerun_of.name", "slug"},
		{"unknown component", func(r *Result) {
			rerun(r, RevisionChange{Component: "kernel", From: "v1", To: "v2"})
		}, "revision_changes[0].component", "not a key"},
		{"component listed twice", func(r *Result) {
			rerun(r, RevisionChange{Component: "blis", From: "v0.12.0", To: "v0.13.0"},
				RevisionChange{Component: "blis", From: "v0.11.0", To: "v0.13.0"})
		}, "revision_changes[1].component", "twice"},
		{"unchanged revision listed", func(r *Result) {
			rerun(r, RevisionChange{Component: "blis", From: "v0.13.0", To: "v0.13.0"})
		}, "revision_changes[0]", "unchanged"},
		{"change to a revision this run does not have", func(r *Result) {
			rerun(r, RevisionChange{Component: "blis", From: "v0.12.0", To: "v0.14.0"})
		}, "revision_changes[0].to", "revisions.blis is"},
		{"change to a revision this run does not state", func(r *Result) {
			rerun(r, RevisionChange{Component: "catalog", From: "0.2.0", To: "0.3.0"})
		}, "revision_changes[0].to", "states no revisions.catalog"},
		{"no metrics", func(r *Result) { r.Metrics = nil }, "metrics", "at least one"},
		{"metric without a definition", func(r *Result) { r.Metrics["itl_p99_ms"] = 28 }, "metrics.itl_p99_ms", "no entry"},
		{"non-finite metric", func(r *Result) { r.Metrics["ttft_p99_ms"] = nan }, "metrics.ttft_p99_ms", "finite"},
		{"negative metric", func(r *Result) { r.Metrics["ttft_p99_ms"] = -1 }, "metrics.ttft_p99_ms", "negative"},
		{"definition without a metric", func(r *Result) {
			r.MetricDefs["ttft_p50_ms"] = r.MetricDefs["ttft_p99_ms"]
		}, "metric_defs.ttft_p50_ms", "does not report"},
		{"metric name not a slug", func(r *Result) {
			r.MetricDefs["TTFT"] = r.MetricDefs["ttft_p99_ms"]
			r.Metrics["TTFT"] = 1
		}, "metric_defs.TTFT", "slug"},
		{"invalid definition", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.PercentileMethod = "" })
		}, "metric_defs.ttft_p99_ms.percentile_method", "required for a percentile"},
		{"client clock", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Clock = measurement.ClockClient })
		}, "metric_defs.ttft_p99_ms.clock", "simulated clock"},
		{"clock unstated", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Clock = "" })
		}, "metric_defs.ttft_p99_ms.clock", "required"},
		{"population unstated", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Population = "" })
		}, "metric_defs.ttft_p99_ms.population", "required"},
		{"formula unstated", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Formula = "" })
		}, "metric_defs.ttft_p99_ms.formula", "required"},
		{"statistic unstated", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Statistic = "" })
		}, "metric_defs.ttft_p99_ms.statistic", "required"},
		{"standard name with another formula", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Formula = measurement.FormulaE2EL })
		}, "metric_defs.ttft_p99_ms.formula", "standard name"},
		{"standard name with another statistic", func(r *Result) {
			def(r, func(d *measurement.MetricDef) { d.Statistic = "p90" })
		}, "metric_defs.ttft_p99_ms.statistic", "standard name"},
		{"standard throughput with another denominator", func(r *Result) {
			r.MetricDefs["output_tok_s"] = measurement.MetricDef{
				Formula: measurement.FormulaOutputThroughput, Population: measurement.PopulationPooledTokens,
				Denominator: measurement.DenominatorPerGPU, Clock: measurement.ClockSimulated,
				Unit: measurement.UnitTokensPerSec, Cite: "x",
			}
			r.Metrics["output_tok_s"] = 1
		}, "metric_defs.output_tok_s.denominator", "standard name"},
		{"window without start", func(r *Result) { r.Window.Start = "" }, "window.start", "required"},
		{"window without stop", func(r *Result) { r.Window.Stop = "" }, "window.stop", "required"},
		{"negative window duration", func(r *Result) { r.Window.DurationS = -1 }, "window.duration_s", "negative"},
		{"non-finite window duration", func(r *Result) { r.Window.DurationS = nan }, "window.duration_s", "finite"},
		{"negative window requests", func(r *Result) { r.Window.Requests = -1 }, "window.requests", "negative"},
		{"window holds more than completed", func(r *Result) { r.Window.Requests = 1991 }, "window.requests", "completed"},
		{"rate over an empty window", func(r *Result) {
			r.Window.DurationS = 0
			r.MetricDefs["request_throughput"] = measurement.MetricDef{
				Formula: measurement.FormulaRequestThroughput, Population: measurement.PopulationPerRequest,
				Denominator: measurement.DenominatorProfilingWindow, Clock: measurement.ClockSimulated,
				Unit: measurement.UnitRequestsSec, Cite: "x",
			}
			r.Metrics["request_throughput"] = 6
		}, "window.duration_s", "rate over the window"},
		{"negative bucket", func(r *Result) { r.Requests.TimedOut = -1 }, "requests.timed_out", "negative"},
		{"INV-1 does not close", func(r *Result) { r.Requests.Completed-- }, "requests", "INV-1"},
		{"generated is not injected plus rejected", func(r *Result) { r.Requests.Rejected = 3 }, "requests.generated", "injected + rejected"},
		{"generated more than num_requests", func(r *Result) {
			r.Requests.Generated, r.Requests.Injected, r.Requests.StillQueued = 2001, 2001, 1
		}, "requests.generated", "num_requests"},
		{"more length-capped than completed", func(r *Result) { r.Requests.LengthCapped = 1991 }, "requests.length_capped", "completed"},
		{"goodput without targets", func(r *Result) { r.Goodput = &Goodput{SLOAttainment: 0.5} }, "goodput.targets", "required"},
		{"negative target", goodput(Goodput{Targets: map[string]workload.SLODimTargets{"a": {TTFTMs: -5}}}),
			"goodput.targets.a.ttft_ms", "negative"},
		{"empty target", goodput(Goodput{Targets: map[string]workload.SLODimTargets{"a": {}}}),
			"goodput.targets.a", "sets no target"},
		{"empty class key", goodput(Goodput{Targets: map[string]workload.SLODimTargets{"": {TTFTMs: 1}}}),
			"goodput.targets", "must not be empty"},
		{"negative goodput", goodput(Goodput{GoodputRPS: -1}), "goodput.goodput_rps", "negative"},
		{"non-finite goodput", goodput(Goodput{GoodputRPS: nan}), "goodput.goodput_rps", "finite"},
		{"attainment above one", goodput(Goodput{SLOAttainment: 1.5}), "goodput.slo_attainment", "[0, 1]"},
		{"attainment below zero", goodput(Goodput{SLOAttainment: -0.1}), "goodput.slo_attainment", "[0, 1]"},
		{"class met above completed", goodput(Goodput{PerClass: map[string]ClassGoodput{"a": {Completed: 1, Met: 2}}}),
			"goodput.per_class.a", "between 0 and completed"},
		{"class met negative", goodput(Goodput{PerClass: map[string]ClassGoodput{"a": {Completed: 1, Met: -1}}}),
			"goodput.per_class.a", "between 0 and completed"},
		{"class completed negative", goodput(Goodput{PerClass: map[string]ClassGoodput{"a": {Completed: -1, Met: -2}}}),
			"goodput.per_class.a", "between 0 and completed"},
		{"class without a target", goodput(Goodput{PerClass: map[string]ClassGoodput{"b": {Completed: 1, Met: 1}}}),
			"goodput.per_class.b", "not a class in goodput.targets"},
		{"split metric without a definition", func(r *Result) {
			r.Breakdowns = &Breakdowns{PerModel: map[string]map[string]float64{"m": {"e2e_p99_ms": 1}}}
		}, "breakdowns.per_model.m.e2e_p99_ms", "no entry in metric_defs"},
		{"non-finite split value", func(r *Result) {
			r.Breakdowns = &Breakdowns{PD: map[string]float64{"prefill_ttft_mean_ms": nan}}
		}, "breakdowns.pd.prefill_ttft_mean_ms", "finite"},
		{"negative split value", func(r *Result) {
			r.Breakdowns = &Breakdowns{PerTenant: map[string]map[string]float64{"t": {"ttft_p99_ms": -1}}}
		}, "breakdowns.per_tenant.t.ttft_p99_ms", "negative"},
		{"split key not a slug", func(r *Result) {
			r.Breakdowns = &Breakdowns{Sessions: map[string]float64{"Turns Mean": 1}}
		}, "breakdowns.sessions.Turns Mean", "slug"},
		{"empty split key", func(r *Result) {
			r.Breakdowns = &Breakdowns{PerAdapter: map[string]map[string]float64{"": {"ttft_p99_ms": 1}}}
		}, "breakdowns.per_adapter", "must not be empty"},
		{"override without setting", func(r *Result) { r.Overrides = []Override{{Source: "--scheduler"}} }, "overrides[0].setting", "required"},
		{"override without source", func(r *Result) { r.Overrides = []Override{{Setting: "scheduler"}} }, "overrides[0].source", "required"},
		{"artifact without sha256", func(r *Result) {
			r.Artifacts = map[string]Artifact{"per_request": {Ref: "r.csv"}}
		}, "artifacts.per_request.sha256", "required"},
		{"artifact without ref", func(r *Result) {
			r.Artifacts = map[string]Artifact{"per_request": {SHA256: strings.Repeat("b", 64)}}
		}, "artifacts.per_request.ref", "required"},
		{"artifact with negative rows", func(r *Result) {
			r.Artifacts = map[string]Artifact{"per_request": {Ref: "r.csv", SHA256: strings.Repeat("b", 64), Rows: ptr(-1)}}
		}, "artifacts.per_request.rows", "negative"},
		{"artifact role not a slug", func(r *Result) {
			r.Artifacts = map[string]Artifact{"Per Request": {Ref: "r.csv", SHA256: strings.Repeat("b", 64)}}
		}, "artifacts.Per Request", "slug"},
		{"negative wall clock", func(r *Result) { r.Runtime = &Runtime{WallClockS: -1} }, "runtime.wall_clock_s", "negative"},
		{"non-finite wall clock", func(r *Result) { r.Runtime = &Runtime{WallClockS: nan} }, "runtime.wall_clock_s", "finite"},
		{"bad start time", func(r *Result) { r.Runtime = &Runtime{StartedUTC: "yesterday"} }, "runtime.started_utc", "RFC 3339"},
		{"start time not in UTC", func(r *Result) {
			r.Runtime = &Runtime{StartedUTC: "2026-10-28T15:02:11+05:00"}
		}, "runtime.started_utc", "ending Z"},
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

func TestTerminalSumsEveryBucket(t *testing.T) {
	q := Requests{
		Completed: 1, StillQueued: 2, StillRunning: 3, DroppedUnservable: 4, TimedOut: 5,
		RoutingRejections: 6, GatewayQueueDepth: 7, GatewayQueueRejected: 8, GatewayEvicted: 9,
		GatewayExpired: 10, EncodeRoutingRejections: 11, LengthCapped: 100,
	}
	if got, want := q.Terminal(), 66; got != want {
		t.Errorf("Terminal() = %d, want %d (every terminal bucket, and not length_capped)", got, want)
	}
}

// TestFullResultPasses: every optional block, filled consistently, validates.
func TestFullResultPasses(t *testing.T) {
	r := valid()
	r.Revisions.Engine = &Engine{Kind: "vllm", UpstreamVersion: "0.29.0", Release: "v0.29.0-blis.1"}
	r.Input.Run.HorizonS = ptr(600.0)
	r.Goodput = &Goodput{
		Targets:    map[string]workload.SLODimTargets{"interactive": {TTFTMs: 500, ITLMs: 50}},
		GoodputRPS: 5.7, SLOAttainment: 0.95,
		PerClass: map[string]ClassGoodput{"interactive": {Completed: 1990, Met: 1890}},
	}
	r.Breakdowns = &Breakdowns{
		PD:       map[string]float64{"kv_transfer_mean_ms": 7.2},
		PerModel: map[string]map[string]float64{"llama-3.1-8b-instruct": {"ttft_p99_ms": 431.5}},
	}
	r.Overrides = []Override{{Setting: "scheduler", Input: "fcfs", Run: "priority", Source: "--scheduler"}}
	r.Artifacts = map[string]Artifact{"per_request": {Ref: "r.csv", SHA256: strings.Repeat("b", 64), Rows: ptr(2000)}}
	r.Runtime = &Runtime{WallClockS: 18.4, StartedUTC: "2026-10-28T15:02:11Z"}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a full result must validate:\n%s", p.Error())
	}
}

func TestRerunWithChangesPasses(t *testing.T) {
	r := valid()
	r.RerunOf = &Reference{Name: "llama-h100-tp2.c64-s1", SHA256: strings.Repeat("c", 64)}
	r.Revisions.Catalog = &DataRepo{Release: "0.4.0"}
	r.RevisionChanges = []RevisionChange{
		{Component: "blis", From: "v0.12.0", To: "v0.13.0"},
		{Component: "catalog", From: "0.3.0", To: "0.4.0"},
	}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a rerun with changes must validate:\n%s", p.Error())
	}
}
