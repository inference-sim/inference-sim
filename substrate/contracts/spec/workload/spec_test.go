package workload

import (
	"math"
	"strings"
	"testing"
)

func f64(v float64) *float64 { return &v }

// spec returns a specification that uses every block, valid as it stands.
func spec() *Spec {
	return &Spec{
		Version: SpecVersion, Seed: 42, Category: CategoryLanguage, AggregateRate: 10, NumRequests: 100,
		Clients: []ClientSpec{
			{ID: "chat", TenantID: "t", SLOClass: SLOCritical, RateFraction: 0.7,
				Arrival:    ArrivalSpec{Process: ArrivalGamma, CV: f64(2)},
				InputDist:  DistSpec{Type: DistGaussian, Params: map[string]float64{"mean": 512, "std_dev": 64, "min": 1, "max": 2048}},
				OutputDist: DistSpec{Type: DistExponential, Params: map[string]float64{"mean": 128}},
				Lifecycle:  &LifecycleSpec{Windows: []ActiveWindow{{StartUs: 0, EndUs: 1_000_000}}},
				Reasoning: &ReasoningSpec{ReasonRatioDist: DistSpec{Type: DistConstant, Params: map[string]float64{"value": 0}},
					MultiTurn: &MultiTurnSpec{MaxRounds: 3, ThinkTimeUs: 1000, ContextGrowth: "accumulate"}}},
		},
		Cohorts: []CohortSpec{
			{ID: "bulk", Population: 10, SLOClass: SLOBatch, RateFraction: 0.3,
				Arrival:     ArrivalSpec{Process: ArrivalPoisson},
				InputDist:   DistSpec{Type: DistEmpirical, Params: map[string]float64{"100": 0.5, "200": 0.5}},
				OutputDist:  DistSpec{Type: DistLognormal, Params: map[string]float64{"mu": 4, "sigma": 1}},
				Diurnal:     &DiurnalSpec{PeakHour: 14, PeakToTroughRatio: 3},
				PrefixGroup: "sys", PrefixSharing: PrefixPerMember},
		},
		GoodputSLOTargets: map[string]SLODimTargets{"critical": {TTFTMs: 500}},
	}
}

func TestValidSpecPasses(t *testing.T) {
	if p := spec().Validate(); !p.OK() {
		t.Fatalf("the fixture must validate:\n%s", p.Error())
	}
}

// TestSpecRefusals: one case per rule inference-sim's WorkloadSpec.Validate enforced, plus
// the removed v1 class names.
func TestSpecRefusals(t *testing.T) {
	cases := []struct {
		name   string
		break_ func(*Spec)
		path   string
		msg    string
	}{
		{"version 1", func(s *Spec) { s.Version = "1" }, "version", "realtime and interactive are removed"},
		{"no version", func(s *Spec) { s.Version = "" }, "version", "removed"},
		{"v1 class name", func(s *Spec) { s.Clients[0].SLOClass = "realtime" }, "clients[0].slo_class", "is not critical"},
		{"unknown category", func(s *Spec) { s.Category = "video" }, "category", "is not language"},
		{"nothing to generate", func(s *Spec) { s.Clients, s.Cohorts = nil, nil }, "", "at least one client"},
		{"non-positive aggregate rate", func(s *Spec) { s.AggregateRate = -1 }, "aggregate_rate", "positive"},
		{"absolute mode without windows", func(s *Spec) { s.AggregateRate = 0; s.Cohorts = nil; s.Clients[0].Lifecycle = nil }, "clients[0].lifecycle", "absolute mode"},
		{"absolute mode window without rate", func(s *Spec) { s.AggregateRate = 0; s.Cohorts = nil }, "clients[0].lifecycle.windows[0].trace_rate", "required"},
		{"absolute mode diurnal cohort", func(s *Spec) {
			s.AggregateRate, s.Clients = 0, nil
		}, "cohorts[0].diurnal", "not supported in absolute mode"},
		{"absolute mode cohort without pattern", func(s *Spec) {
			s.AggregateRate, s.Clients = 0, nil
			s.Cohorts[0].Diurnal = nil
		}, "cohorts[0].spike", "required in absolute mode"},
		{"concurrency and rate", func(s *Spec) { s.Clients[0].Concurrency = 4 }, "clients[0]", "mutually exclusive"},
		{"negative concurrency", func(s *Spec) { s.Clients[0].Concurrency = -1 }, "clients[0].concurrency", "negative"},
		{"no rate fraction", func(s *Spec) { s.Clients[0].RateFraction = 0 }, "clients[0].rate_fraction", "positive"},
		{"unknown arrival", func(s *Spec) { s.Clients[0].Arrival.Process = "burst" }, "clients[0].arrival.process", "is not poisson"},
		{"weibull CV out of range", func(s *Spec) {
			s.Clients[0].Arrival = ArrivalSpec{Process: ArrivalWeibull, CV: f64(20)}
		}, "clients[0].arrival.cv", "[0.01, 10.4]"},
		{"non-positive shape", func(s *Spec) { s.Clients[0].Arrival.Shape = f64(0) }, "clients[0].arrival.shape", "positive"},
		{"unknown distribution", func(s *Spec) { s.Clients[0].InputDist.Type = "uniform" }, "clients[0].input_distribution.type", "is not gaussian"},
		{"missing parameter", func(s *Spec) { delete(s.Clients[0].InputDist.Params, "std_dev") }, "clients[0].input_distribution.params", `"std_dev"`},
		{"non-finite parameter", func(s *Spec) { s.Clients[0].InputDist.Params["mean"] = math.NaN() }, "clients[0].input_distribution.params.mean", "finite"},
		{"empirical key not a count", func(s *Spec) { s.Cohorts[0].InputDist.Params["x"] = 0.1 }, "cohorts[0].input_distribution.params.x", "token count"},
		{"min above max", func(s *Spec) {
			s.Clients[0].OutputDist.Params["min"], s.Clients[0].OutputDist.Params["max"] = 10, 5
		}, "clients[0].output_distribution.params", "must not exceed"},
		{"window ends before it starts", func(s *Spec) { s.Clients[0].Lifecycle.Windows[0].EndUs = 0 }, "clients[0].lifecycle.windows[0].end_us", "not after"},
		{"empty lifecycle", func(s *Spec) { s.Clients[0].Lifecycle.Windows = nil }, "clients[0].lifecycle.windows", "at least one"},
		{"zero rounds", func(s *Spec) { s.Clients[0].Reasoning.MultiTurn.MaxRounds = 0 }, "clients[0].reasoning.multi_turn.max_rounds", "at least 1"},
		{"unknown context growth", func(s *Spec) { s.Clients[0].Reasoning.MultiTurn.ContextGrowth = "grow" }, "clients[0].reasoning.multi_turn.context_growth", "is not accumulate"},
		{"negative timeout", func(s *Spec) { s.Clients[0].Timeout = new(int64); *s.Clients[0].Timeout = -1 }, "clients[0].timeout", "negative"},
		{"concurrency with multi-turn", func(s *Spec) {
			s.Clients = append(s.Clients, ClientSpec{ID: "c", SLOClass: SLOStandard, Concurrency: 2,
				InputDist:  DistSpec{Type: DistConstant, Params: map[string]float64{"value": 1}},
				OutputDist: DistSpec{Type: DistConstant, Params: map[string]float64{"value": 1}}})
		}, "", "cannot be mixed"},
		{"mixed slo classes", func(s *Spec) { s.Cohorts[0].SLOClass = "" }, "", "mixed slo_class"},
		{"zero population", func(s *Spec) { s.Cohorts[0].Population = 0 }, "cohorts[0].population", "positive"},
		{"huge population", func(s *Spec) { s.Cohorts[0].Population = MaxCohortPopulation + 1 }, "cohorts[0].population", "exceeds"},
		{"peak hour out of range", func(s *Spec) { s.Cohorts[0].Diurnal.PeakHour = 24 }, "cohorts[0].diurnal.peak_hour", "[0, 23]"},
		{"ratio below one", func(s *Spec) { s.Cohorts[0].Diurnal.PeakToTroughRatio = 0.5 }, "cohorts[0].diurnal.peak_to_trough_ratio", "at least 1"},
		{"per-member without group", func(s *Spec) { s.Cohorts[0].PrefixGroup = "" }, "cohorts[0].prefix_group", "required"},
		{"unknown sharing", func(s *Spec) { s.Cohorts[0].PrefixSharing = "pooled" }, "cohorts[0].prefix_sharing", "is not shared"},
		{"zero spike", func(s *Spec) { s.Cohorts[0].Spike = &SpikeSpec{} }, "cohorts[0].spike.duration_us", "positive"},
		{"zero drain", func(s *Spec) { s.Cohorts[0].Drain = &DrainSpec{} }, "cohorts[0].drain.ramp_duration_us", "positive"},
		{"inference-perf without stages", func(s *Spec) {
			s.InferencePerf = &InferencePerfSpec{SharedPrefix: &SharedPrefixSpec{NumUniqueSystemPrompts: 1, NumUsersPerSystemPrompt: 1}}
		}, "inference_perf.stages", "at least one"},
		{"inference-perf without prefix", func(s *Spec) {
			s.InferencePerf = &InferencePerfSpec{Stages: []StageSpec{{Rate: 1, Duration: 1}}}
		}, "inference_perf.shared_prefix", "required"},
		{"empty target class", func(s *Spec) { s.GoodputSLOTargets[""] = SLODimTargets{TTFTMs: 1} }, "goodput_slo_targets", "must not be empty"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := spec()
			c.break_(s)
			p := s.Validate()
			for _, it := range p.Errors() {
				if it.Path == c.path && strings.Contains(it.Message, c.msg) {
					return
				}
			}
			t.Errorf("expected a problem at %q containing %q, got:\n%s", c.path, c.msg, p.Error())
		})
	}
}
