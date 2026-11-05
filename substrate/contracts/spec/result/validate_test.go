package result

import (
	"math"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/measurement"
	"github.com/inference-sim/blis-substrate/contracts/spec/scenario"
	"github.com/inference-sim/blis-substrate/contracts/spec/workload"
)

func ptr[T any](v T) *T { return &v }

var (
	hashA = strings.Repeat("a", 64)
	hashB = strings.Repeat("b", 64)
	hashC = strings.Repeat("c", 64)
)

// valid returns a Result every validator accepts; each case below breaks one thing.
func valid() *Result {
	return &Result{
		Kind:          Kind,
		SchemaVersion: SchemaVersion,
		Name:          "llama-h100-tp2.c64-s1",
		Engine:        Engine{Kind: "vllm", Version: "0.32.0"},
		Data: Data{
			Catalog:  DataSource{Source: SourceEmbedded, Hash: hashA, CombinedHash: hashA},
			Registry: DataSource{Source: SourcePath, Hash: hashB, CombinedHash: hashB},
			Build:    BuildUnreleased,
		},
		SelectedEntries: SelectedEntries{
			Coefficients: []SelectedCoefficient{
				{Set: "cost-model-primitives", Name: "gemm_efficiency", Method: "measured",
					Scope: coefficient.Scope{Hardware: []string{"h100"}}, File: "coefficients/cost-model-primitives.yaml"},
			},
			UnreadFiles: []string{"coefficients/cost-model-recurrent.yaml"},
		},
		Input: Input{
			Scenario: scenario.Scenario{
				Kind: "Scenario", Name: "llama-3.1-8b-on-h100", Model: "llama-3.1-8b-instruct",
				Coefficients: []string{"cost-model-primitives"},
				Engine:       &scenario.Engine{Kind: "vllm", Version: "0.32.0"},
				Cluster:      scenario.Cluster{Hardware: "h100", Nodes: 1, GPUsPerNode: 8},
			},
			Deployment: deployment.Deployment{
				Kind: "Deployment", Name: "tp2-dp4",
				Pools: []deployment.Pool{{
					Role: deployment.RoleColocated, Nodes: 1,
					Parallel: deployment.Parallelism{TP: 2, PP: 1, DP: 4},
				}},
			},
			Options: []Option{
				{Name: "concurrency", Value: "64", Source: OptionCommandLine},
				{Name: "seed", Value: "1", Source: OptionDefault},
			},
			WorkloadProperties: []WorkloadProperty{
				{Name: "expert_routing", Source: PropertyDefault},
			},
		},
		Frame: Frame{
			Kind: FrameTerminating,
			Generator: Generator{Workload: &workload.Binding{Shape: "chatbot"},
				Load: measurement.Load{Concurrency: 64}, Seed: 1, RNG: "pcg64-sha256/1"},
			InitialState: InitialState{Caches: CachesEmpty},
			Phases: Phases{
				Warmup:   WarmupPhase{Rule: WarmupRequests, Requests: 2},
				Measured: MeasuredPhase{Start: StartFirstArrivalAfterWarmup, Stop: StopLastCompletion},
				Drain:    DrainPhase{Rule: DrainUntilComplete},
			},
			Stop:       StopRule{Requests: 4},
			Acceptance: []AcceptanceCondition{{Condition: ConditionUnfinishedAtMost, Threshold: 0.5, Observed: 0.25, Met: true}},
			Metrics:    []string{"tpot_mean_ms", "ttft_p99_ms"},
		},
		Revisions: Revisions{Simulator: "v0.16.0", Substrate: "v0.11.0"},
		Window:    Window{StartUs: 1_000_000, EndUs: 4_000_000, DurationS: 3, Requests: 2},
		Metrics: map[string]Figure{
			"ttft_p99_ms":  {Value: 431.5, Samples: 2},
			"tpot_mean_ms": {Value: 20, Samples: 2},
		},
		Requests: Requests{Generated: 4, Completed: 2, InFlightAtHorizon: 1, Dropped: map[DropReason]int{DropRoutingRejected: 1}},
		RequestRows: []RequestRow{
			{ID: "request_0", Fate: FateCompleted, ArrivalUs: 0, TTFTUs: ptr(int64(100)), TPOTUs: ptr(20.0),
				E2EUs: ptr(int64(140)), CompletionIndex: 1, PromptTokens: 10, OutputTokens: 3},
			{ID: "request_1", Fate: FateCompleted, ArrivalUs: 5, TTFTUs: ptr(int64(90)), TPOTUs: ptr(20.0),
				E2EUs: ptr(int64(130)), CompletionIndex: 2, PromptTokens: 10, OutputTokens: 3},
			{ID: "request_2", Fate: FateInFlightAtHorizon, ArrivalUs: 7, TTFTUs: ptr(int64(95)), PromptTokens: 10, OutputTokens: 1},
			{ID: "request_3", Fate: FateDropped, DropReason: DropRoutingRejected, ArrivalUs: 9, PromptTokens: 10},
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
// finding (SUB-2: a failing example for each way a field can be wrong).
func TestEachFailureModeIsRefused(t *testing.T) {
	rerun := func(r *Result, changes ...RevisionChange) {
		r.RerunOf = &Reference{Name: "old.c64-s1", SHA256: hashA}
		r.RevisionChanges = changes
	}
	row := func(i int, edit func(*RequestRow)) func(*Result) {
		return func(r *Result) { edit(&r.RequestRows[i]) }
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
		{"old schema version", func(r *Result) { r.SchemaVersion = 1 }, "schema_version", "must be 2"},
		{"name not a slug", func(r *Result) { r.Name = "Has Spaces" }, "name", "slug"},

		// engine (SIM-1)
		{"engine without kind", func(r *Result) { r.Engine.Kind = "" }, "engine.kind", "required"},
		{"engine without version", func(r *Result) { r.Engine.Version = "" }, "engine.version", "required"},
		{"engine other than the scenario's release", func(r *Result) { r.Engine.Version = "0.31.0" }, "engine.version", "scenario names"},
		{"engine other than the scenario's kind", func(r *Result) { r.Engine.Kind = "sglang" }, "engine.kind", "input.scenario.engine.kind"},
		{"engine other than the deprecated engine_version", func(r *Result) {
			r.Input.Scenario.Engine, r.Input.Scenario.EngineVersion = nil, "0.31.0"
		}, "engine.version", "scenario names"},

		// data (design §3.3, X8)
		{"catalog without source", func(r *Result) { r.Data.Catalog.Source = "" }, "data.catalog.source", "required"},
		{"catalog with unknown source", func(r *Result) { r.Data.Catalog.Source = "remote" }, "data.catalog.source", "is not"},
		{"registry without hash", func(r *Result) { r.Data.Registry.Hash = "" }, "data.registry.hash", "required"},
		{"hash in upper case", func(r *Result) { r.Data.Registry.Hash = strings.Repeat("A", 64) }, "data.registry.hash", "lowercase hex"},
		{"short combined hash", func(r *Result) { r.Data.Catalog.CombinedHash = "abc" }, "data.catalog.combined_hash", "64 lowercase"},
		{"combined hash differs without overlay", func(r *Result) { r.Data.Catalog.CombinedHash = hashC }, "data.catalog.combined_hash", "no overlay"},
		{"overlay without hash", func(r *Result) {
			r.Data.Catalog.Overlays = []Overlay{{}}
			r.Data.Catalog.CombinedHash = hashC
		}, "data.catalog.overlays[0].hash", "required"},
		{"no build", func(r *Result) { r.Data.Build = "" }, "data.build", "required"},
		{"build not a tag", func(r *Result) { r.Data.Build = "main" }, "data.build", "release tag"},
		{"released data from an unreleased build", func(r *Result) { r.Data.ReleasedData = true }, "data.released_data", "tagged simulator release"},
		{"released data with an overlay", func(r *Result) {
			r.Data.Build, r.Data.ReleasedData = "v0.16.0", true
			r.Data.Registry.Overlays = []Overlay{{Hash: hashC}}
			r.Data.Registry.CombinedHash = hashA
		}, "data.released_data", "overlays"},

		// selected entries (VAL-6)
		{"selected coefficient without set", func(r *Result) { r.SelectedEntries.Coefficients[0].Set = "" }, "selected_entries.coefficients[0].set", "required"},
		{"selected coefficient with unknown method", func(r *Result) { r.SelectedEntries.Coefficients[0].Method = "guessed" }, "selected_entries.coefficients[0].method", "not a registry method"},
		{"selected coefficient listed twice", func(r *Result) {
			r.SelectedEntries.Coefficients = append(r.SelectedEntries.Coefficients, r.SelectedEntries.Coefficients[0])
		}, "selected_entries.coefficients[1]", "once each"},
		{"selected profile without key", func(r *Result) {
			r.SelectedEntries.AcceptanceProfiles = []SelectedProfile{{Set: "s", File: "acceptance/s.yaml"}}
		}, "selected_entries.acceptance_profiles[0].key", "required"},
		{"unread file that was read", func(r *Result) {
			r.SelectedEntries.UnreadFiles = []string{"coefficients/cost-model-primitives.yaml"}
		}, "selected_entries.unread_files[0]", "was read from it"},
		{"unread files unsorted", func(r *Result) {
			r.SelectedEntries.UnreadFiles = []string{"b.yaml", "a.yaml"}
		}, "selected_entries.unread_files[1]", "sorted"},
		{"unread file absolute", func(r *Result) { r.SelectedEntries.UnreadFiles = []string{"/x.yaml"} }, "selected_entries.unread_files[0]", "relative"},

		// input
		{"embedded scenario invalid", func(r *Result) { r.Input.Scenario.Kind = "" }, "input.scenario.kind", "Scenario"},
		{"deployment does not fill its cluster", func(r *Result) { r.Input.Scenario.Cluster.Nodes = 2 }, "input.deployment.pools", "sum to 1"},
		{"option with dashes", func(r *Result) { r.Input.Options[0].Name = "--concurrency" }, "input.options[0].name", "without its dashes"},
		{"options unsorted", func(r *Result) { r.Input.Options[1].Name = "a" }, "input.options[1].name", "sorted"},
		{"option without source", func(r *Result) { r.Input.Options[0].Source = "" }, "input.options[0].source", "is not one of"},
		{"property with unknown source", func(r *Result) { r.Input.WorkloadProperties[0].Source = "guess" }, "input.workload_properties[0].source", "is not one of"},
		{"property from a profile without its name", func(r *Result) {
			r.Input.WorkloadProperties[0] = WorkloadProperty{Name: "speculative_acceptance", Source: PropertyProfile,
				Values: map[string]float64{"acceptance_length": 3}}
		}, "input.workload_properties[0].profile", "exactly when"},
		{"property without values", func(r *Result) { r.Input.WorkloadProperties[0].Source = PropertyScenario }, "input.workload_properties[0].values", "required"},
		{"property with a non-finite value", func(r *Result) {
			r.Input.WorkloadProperties[0] = WorkloadProperty{Name: "x", Source: PropertyCommandLine, Values: map[string]float64{"v": nan}}
		}, "input.workload_properties[0].values.v", "finite"},

		// frame (DEV-7)
		{"frame without kind", func(r *Result) { r.Frame.Kind = "" }, "frame.kind", "is not"},
		{"both concurrency and rate", func(r *Result) { r.Frame.Generator.Load.RequestRate = 5 }, "frame.generator.load", "not both"},
		{"negative concurrency", func(r *Result) { r.Frame.Generator.Load.Concurrency = -1 }, "frame.generator.load.concurrency", "negative"},
		{"non-finite request rate", func(r *Result) {
			r.Frame.Generator.Load = measurement.Load{RequestRate: math.Inf(1)}
		}, "frame.generator.load.request_rate", "finite"},
		{"no workload and no load", func(r *Result) {
			r.Frame.Generator.Workload, r.Frame.Generator.Load = nil, measurement.Load{}
		}, "frame.generator", "workload, an offered load"},
		{"workload with two arms", func(r *Result) {
			r.Frame.Generator.Workload.Trace = &workload.TraceRef{Data: "t.csv"}
		}, "frame.generator.workload", "exactly one"},
		{"no generator name", func(r *Result) { r.Frame.Generator.RNG = "" }, "frame.generator.rng", "required"},
		{"warmed caches without a rule", func(r *Result) { r.Frame.InitialState.Caches = CachesWarmed }, "frame.initial_state.rule", "required"},
		{"empty caches with a rule", func(r *Result) { r.Frame.InitialState.Rule = "x" }, "frame.initial_state.rule", "no rule"},
		{"unknown cache state", func(r *Result) { r.Frame.InitialState.Caches = "hot" }, "frame.initial_state.caches", "is not"},
		{"warm-up by requests without a count", func(r *Result) { r.Frame.Phases.Warmup.Requests = 0 }, "frame.phases.warmup", "positive requests"},
		{"warm-up by duration with requests", func(r *Result) {
			r.Frame.Phases.Warmup = WarmupPhase{Rule: WarmupDuration, DurationS: 1, Requests: 1}
		}, "frame.phases.warmup", "positive duration_s"},
		{"warm-up none with a count", func(r *Result) { r.Frame.Phases.Warmup.Rule = WarmupNone }, "frame.phases.warmup", "neither"},
		{"non-finite warm-up duration", func(r *Result) {
			r.Frame.Phases.Warmup = WarmupPhase{Rule: WarmupDuration, DurationS: nan}
		}, "frame.phases.warmup.duration_s", "finite"},
		{"unknown warm-up rule", func(r *Result) { r.Frame.Phases.Warmup.Rule = "skip" }, "frame.phases.warmup.rule", "is not one of"},
		{"unknown window start", func(r *Result) { r.Frame.Phases.Measured.Start = "x" }, "frame.phases.measured.start", "is not one of"},
		{"unknown window stop", func(r *Result) { r.Frame.Phases.Measured.Stop = "x" }, "frame.phases.measured.stop", "is not one of"},
		{"grace drain without duration", func(r *Result) { r.Frame.Phases.Drain.Rule = DrainGrace }, "frame.phases.drain.duration_s", "positive"},
		{"drain duration without grace", func(r *Result) { r.Frame.Phases.Drain.DurationS = 1 }, "frame.phases.drain.duration_s", "only a grace"},
		{"no stop rule", func(r *Result) { r.Frame.Stop = StopRule{} }, "frame.stop", "request count, a horizon"},
		{"non-positive horizon", func(r *Result) { r.Frame.Stop.HorizonS = ptr(0.0) }, "frame.stop.horizon_s", "positive"},
		{"non-finite horizon", func(r *Result) { r.Frame.Stop.HorizonS = ptr(nan) }, "frame.stop.horizon_s", "finite"},
		{"acceptance with a wrong verdict", func(r *Result) { r.Frame.Acceptance[0].Observed = 0.9 }, "frame.acceptance[0].met", "gives false"},
		{"unknown acceptance condition", func(r *Result) { r.Frame.Acceptance[0].Condition = "x" }, "frame.acceptance[0].condition", "is not"},
		{"frame measures nothing", func(r *Result) { r.Frame.Metrics = nil; r.Metrics = nil }, "frame.metrics", "required"},
		{"frame names an unregistered metric", func(r *Result) { r.Frame.Metrics = []string{"tpot_mean_ms", "ttft_p999_ms"} }, "frame.metrics[1]", "not in the metric registry"},
		{"frame metrics unsorted", func(r *Result) { r.Frame.Metrics = []string{"ttft_p99_ms", "tpot_mean_ms"} }, "frame.metrics[1]", "sorted"},

		// revisions and reruns
		{"no simulator revision", func(r *Result) { r.Revisions.Simulator = "" }, "revisions.simulator", "required"},
		{"no substrate revision", func(r *Result) { r.Revisions.Substrate = "" }, "revisions.substrate", "required"},
		{"revision changes without rerun_of", func(r *Result) {
			r.RevisionChanges = []RevisionChange{{Component: "simulator", From: "v1", To: "v0.16.0"}}
		}, "revision_changes", "set rerun_of"},
		{"rerun_of without sha256", func(r *Result) { r.RerunOf = &Reference{Name: "old.c64-s1"} }, "rerun_of.sha256", "required"},
		{"unknown component", func(r *Result) {
			rerun(r, RevisionChange{Component: "kernel", From: "v1", To: "v2"})
		}, "revision_changes[0].component", "is not"},
		{"component listed twice", func(r *Result) {
			rerun(r, RevisionChange{Component: "simulator", From: "v0.15.0", To: "v0.16.0"},
				RevisionChange{Component: "simulator", From: "v0.14.0", To: "v0.16.0"})
		}, "revision_changes[1].component", "twice"},
		{"unchanged revision listed", func(r *Result) {
			rerun(r, RevisionChange{Component: "simulator", From: "v0.16.0", To: "v0.16.0"})
		}, "revision_changes[0]", "unchanged"},
		{"change to a revision this run does not have", func(r *Result) {
			rerun(r, RevisionChange{Component: "simulator", From: "v0.15.0", To: "v0.17.0"})
		}, "revision_changes[0].to", "this run's simulator"},
		{"change to data this run did not read", func(r *Result) {
			rerun(r, RevisionChange{Component: "registry", From: hashA, To: hashC})
		}, "revision_changes[0].to", "this run's registry"},
		{"change to a revision this run does not state", func(r *Result) {
			rerun(r, RevisionChange{Component: "router", From: "v1", To: "v2"})
		}, "revision_changes[0].to", "states no router"},

		// window and metrics (DEV-8)
		{"window ends before it starts", func(r *Result) { r.Window.EndUs = 0 }, "window.end_us", "before start_us"},
		{"window duration disagrees", func(r *Result) { r.Window.DurationS = 2 }, "window.duration_s", "end_us - start_us"},
		{"negative window requests", func(r *Result) { r.Window.Requests = -1 }, "window.requests", "negative"},
		{"window holds more than completed", func(r *Result) { r.Window.Requests = 3 }, "window.requests", "completed"},
		{"unregistered metric", func(r *Result) { r.Metrics["ttft_p999_ms"] = Figure{Value: 1, Samples: 1} }, "metrics.ttft_p999_ms", "not in the metric registry"},
		{"metric outside the frame", func(r *Result) { r.Metrics["e2e_mean_ms"] = Figure{Value: 1, Samples: 1} }, "metrics.e2e_mean_ms", "frame.metrics"},
		{"non-finite metric", func(r *Result) { r.Metrics["ttft_p99_ms"] = Figure{Value: nan, Samples: 1} }, "metrics.ttft_p99_ms.value", "finite"},
		{"negative metric", func(r *Result) { r.Metrics["ttft_p99_ms"] = Figure{Value: -1, Samples: 1} }, "metrics.ttft_p99_ms.value", "negative"},
		{"latency over no sample", func(r *Result) { r.Metrics["ttft_p99_ms"] = Figure{Value: 1} }, "metrics.ttft_p99_ms.samples", "no sample"},
		{"fraction above one", func(r *Result) {
			r.Frame.Metrics = []string{"cache_hit_rate", "tpot_mean_ms", "ttft_p99_ms"}
			r.Metrics["cache_hit_rate"] = Figure{Value: 1.2, Samples: 4}
		}, "metrics.cache_hit_rate.value", "fraction"},
		{"rate over an empty window", func(r *Result) {
			r.Window = Window{StartUs: 5, EndUs: 5, Requests: 0}
			r.Frame.Metrics = []string{"request_throughput", "tpot_mean_ms", "ttft_p99_ms"}
			r.Metrics["request_throughput"] = Figure{Value: 6, Samples: 2}
		}, "window.duration_s", "rate over the window"},
		{"figures from a run that failed its frame", func(r *Result) {
			r.Frame.Acceptance[0] = AcceptanceCondition{Condition: ConditionUnfinishedAtMost, Threshold: 0.1, Observed: 0.25}
		}, "metrics", "DEV-7"},

		// fates (DEV-9)
		{"negative fate count", func(r *Result) { r.Requests.TimedOut = -1 }, "requests.timed_out", "negative"},
		{"fates do not sum to generated", func(r *Result) { r.Requests.Generated = 5; r.RequestRows = nil }, "requests", "DEV-9"},
		{"unknown drop reason", func(r *Result) {
			r.Requests.Dropped = map[DropReason]int{"bored": 1}
			r.RequestRows = nil
		}, "requests.dropped.bored", "is not one of"},
		{"zero drop count", func(r *Result) {
			r.Requests.Dropped[DropUnservable] = 0
		}, "requests.dropped.unservable", "omit a reason"},
		{"more length-capped than completed", func(r *Result) { r.Requests.LengthCapped = 3; r.RequestRows = nil }, "requests.length_capped", "completed"},

		// request rows
		{"rows short of generated", func(r *Result) {
			r.RequestRows = r.RequestRows[:3]
		}, "request_rows", "one row per generated request"},
		{"row id reused", row(1, func(x *RequestRow) { x.ID = "request_0" }), "request_rows[1].id", "never reused"},
		{"row without id", row(1, func(x *RequestRow) { x.ID = "" }), "request_rows[1].id", "required"},
		{"row with a seventh fate", row(2, func(x *RequestRow) { x.Fate = "length_capped" }), "request_rows[2].fate", "six fates"},
		{"dropped row without reason", row(3, func(x *RequestRow) { x.DropReason = "" }), "request_rows[3].drop_reason", "names one of"},
		{"reason on a row not dropped", row(2, func(x *RequestRow) { x.DropReason = DropUnservable }), "request_rows[2].drop_reason", "only a dropped"},
		{"ttft without a token", row(3, func(x *RequestRow) { x.TTFTUs = ptr(int64(1)) }), "request_rows[3].ttft_us", "emitted no token"},
		{"tpot on an unfinished row", row(2, func(x *RequestRow) { x.TPOTUs = ptr(1.0) }), "request_rows[2].tpot_us", "completed request"},
		{"tpot with one output token", row(0, func(x *RequestRow) { x.OutputTokens = 1 }), "request_rows[0].tpot_us", "at least two"},
		{"e2e on an unfinished row", row(2, func(x *RequestRow) { x.E2EUs = ptr(int64(1)) }), "request_rows[2].e2e_us", "only a completed"},
		{"completed row without latencies", row(0, func(x *RequestRow) { x.E2EUs = nil }), "request_rows[0]", "carries ttft_us and e2e_us"},
		{"completed row without tpot", row(0, func(x *RequestRow) { x.TPOTUs = nil }), "request_rows[0].tpot_us", "required"},
		{"e2e below ttft", row(0, func(x *RequestRow) { x.E2EUs = ptr(int64(50)) }), "request_rows[0].e2e_us", "less than ttft_us"},
		{"negative scheduling delay", row(0, func(x *RequestRow) { x.SchedulingDelayUs = ptr(int64(-1)) }), "request_rows[0].scheduling_delay_us", "negative"},
		{"non-finite tpot", row(0, func(x *RequestRow) { x.TPOTUs = ptr(nan) }), "request_rows[0].tpot_us", "finite"},
		{"negative prompt tokens", row(0, func(x *RequestRow) { x.PromptTokens = -1 }), "request_rows[0].prompt_tokens", "negative"},
		{"completion index on an unfinished row", row(2, func(x *RequestRow) { x.CompletionIndex = 3 }), "request_rows[2].completion_index", "only a completed"},
		{"completion index out of range", row(0, func(x *RequestRow) { x.CompletionIndex = 3 }), "request_rows[0].completion_index", "lies in [1, 2]"},
		{"completion index given twice", row(1, func(x *RequestRow) { x.CompletionIndex = 1 }), "request_rows[1].completion_index", "two requests"},
		{"row fates disagree with the counts", func(r *Result) {
			r.RequestRows[2].Fate, r.RequestRows[2].TTFTUs = FateQueuedAtHorizon, nil
		}, "request_rows", "fate in_flight_at_horizon"},
		{"row drop reasons disagree with the counts", row(3, func(x *RequestRow) { x.DropReason = DropUnservable }), "request_rows", "dropped for"},
		{"length-capped row disagrees with the count", row(0, func(x *RequestRow) { x.LengthCapped = true }), "request_rows", "length-capped"},
		{"length-capped row not completed", row(2, func(x *RequestRow) { x.LengthCapped = true }), "request_rows[2].length_capped", "only a completed"},
		{"unknown phase", row(0, func(x *RequestRow) { x.Phase = "main" }), "request_rows[0].phase", "is not one of"},
		{"agentic tag with unknown kind", row(0, func(x *RequestRow) {
			x.Agentic = &AgenticTag{TraceID: "t", Stream: "t", Kind: "child"}
		}), "request_rows[0].agentic.kind", "is not one of"},

		// goodput
		{"goodput without targets", func(r *Result) { r.Goodput = &Goodput{} }, "goodput.targets", "required"},
		{"negative target", goodput(Goodput{Targets: map[string]workload.SLODimTargets{"a": {TTFTMs: -5}}}), "goodput.targets.a.ttft_ms", "negative"},
		{"empty class key", goodput(Goodput{Targets: map[string]workload.SLODimTargets{"": {TTFTMs: 1}}}), "goodput.targets", "must not be empty"},
		{"class met above completed", goodput(Goodput{PerClass: map[string]ClassGoodput{"a": {Completed: 1, Met: 2}}}), "goodput.per_class.a", "between 0 and completed"},
		{"class without a target", goodput(Goodput{PerClass: map[string]ClassGoodput{"b": {Completed: 1, Met: 1}}}), "goodput.per_class.b", "not a class"},

		// breakdowns
		{"split metric not reported", func(r *Result) {
			r.Breakdowns = &Breakdowns{PerModel: map[string]map[string]float64{"m": {"e2e_p99_ms": 1}}}
		}, "breakdowns.per_model.m.e2e_p99_ms", "not a key of metrics"},
		{"non-finite split value", func(r *Result) {
			r.Breakdowns = &Breakdowns{PD: map[string]float64{"kv_transfer_mean_ms": nan}}
		}, "breakdowns.pd.kv_transfer_mean_ms", "finite"},
		{"split key not a slug", func(r *Result) {
			r.Breakdowns = &Breakdowns{Sessions: map[string]float64{"Turns Mean": 1}}
		}, "breakdowns.sessions.Turns Mean", "slug"},
		{"empty split key", func(r *Result) {
			r.Breakdowns = &Breakdowns{PerAdapter: map[string]map[string]float64{"": {"ttft_p99_ms": 1}}}
		}, "breakdowns.per_adapter", "must not be empty"},

		// the rest
		{"override without source", func(r *Result) { r.Overrides = []Override{{Setting: "scheduler"}} }, "overrides[0].source", "required"},
		{"artifact without sha256", func(r *Result) { r.Artifacts = map[string]Artifact{"trace": {Ref: "r.csv"}} }, "artifacts.trace.sha256", "required"},
		{"artifact with negative rows", func(r *Result) {
			r.Artifacts = map[string]Artifact{"trace": {Ref: "r.csv", SHA256: hashB, Rows: ptr(-1)}}
		}, "artifacts.trace.rows", "negative"},
		{"negative wall clock", func(r *Result) { r.Runtime = &Runtime{WallClockS: -1} }, "runtime.wall_clock_s", "negative"},
		{"start time not in UTC", func(r *Result) { r.Runtime = &Runtime{StartedUTC: "2026-10-28T15:02:11+05:00"} }, "runtime.started_utc", "ending Z"},
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

// TestSumCountsEveryFate: Sum adds the six fates and every drop reason, and not
// length_capped, which is a property of completed requests.
func TestSumCountsEveryFate(t *testing.T) {
	q := Requests{Completed: 1, TimedOut: 2, InFlightAtHorizon: 3, QueuedAtHorizon: 4, NotStarted: 5,
		Dropped: map[DropReason]int{DropUnservable: 6, DropGatewayExpired: 7}, LengthCapped: 100}
	if got, want := q.Sum(), 28; got != want {
		t.Errorf("Sum() = %d, want %d", got, want)
	}
	if got := q.Unfinished(); got != 7 {
		t.Errorf("Unfinished() = %d, want 7", got)
	}
}

// TestFullResultPasses: every optional block, filled consistently, validates.
func TestFullResultPasses(t *testing.T) {
	r := valid()
	r.Data.Registry.Overlays = []Overlay{{Hash: hashC}}
	r.Data.Registry.CombinedHash = hashA
	r.Frame.Stop.HorizonS = ptr(600.0)
	r.Input.WorkloadProperties = []WorkloadProperty{
		{Name: "expert_routing", Source: PropertyDefault},
		{Name: "speculative_acceptance", Source: PropertyProfile, Profile: "acceptance-default/llama/eagle3/k3/thinking-off/coding",
			Values: map[string]float64{"acceptance_length": 2.9}},
	}
	r.SelectedEntries.AcceptanceProfiles = []SelectedProfile{{Set: "acceptance-default", Key: "llama/eagle3/k3/thinking-off/coding", File: "acceptance/acceptance-default.yaml"}}
	r.Goodput = &Goodput{
		Targets:  map[string]workload.SLODimTargets{"interactive": {TTFTMs: 500, ITLMs: 50}},
		PerClass: map[string]ClassGoodput{"interactive": {Completed: 2, Met: 1}},
	}
	r.Breakdowns = &Breakdowns{
		PD:       map[string]float64{"kv_transfer_mean_ms": 7.2},
		PerModel: map[string]map[string]float64{"llama-3.1-8b-instruct": {"ttft_p99_ms": 431.5}},
	}
	r.Reports = Reports{Engine: map[string]any{"colocated": map[string]any{"preemptions": 3}}}
	r.RequestRows[0].Agentic = &AgenticTag{Lane: 0, TraceID: "t1", Stream: "t1", Kind: AgenticRoot, Outcome: "completed"}
	r.RequestRows[0].Phase = PhaseProfile
	r.Overrides = []Override{{Setting: "scheduler", Input: "fcfs", Run: "priority", Source: "--scheduler"}}
	r.Artifacts = map[string]Artifact{"trace": {Ref: "r.csv", SHA256: hashB, Rows: ptr(4)}}
	r.Runtime = &Runtime{WallClockS: 18.4, StartedUTC: "2026-10-28T15:02:11Z"}
	r.RerunOf = &Reference{Name: "llama-h100-tp2.c64-s1", SHA256: hashC}
	r.RevisionChanges = []RevisionChange{
		{Component: "simulator", From: "v0.15.0", To: "v0.16.0"},
		{Component: "registry", From: hashB, To: hashA},
	}
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a full result must validate:\n%s", p.Error())
	}
}

// TestDeprecatedEngineVersionStillReads: until M4 a scenario may name its release only by
// engine_version (plan §3.3); the Result's engine must equal it.
func TestDeprecatedEngineVersionStillReads(t *testing.T) {
	r := valid()
	r.Input.Scenario.Engine, r.Input.Scenario.EngineVersion = nil, "0.32.0"
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a scenario with only engine_version must validate:\n%s", p.Error())
	}
}

// TestRowsAreOptional: the fate counts stand alone; rows, when present, must agree.
func TestRowsAreOptional(t *testing.T) {
	r := valid()
	r.RequestRows = nil
	if p := r.Validate(); !p.OK() {
		t.Fatalf("a result without rows must validate:\n%s", p.Error())
	}
}
