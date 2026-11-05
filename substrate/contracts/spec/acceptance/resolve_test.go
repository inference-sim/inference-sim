package acceptance

import (
	"strings"
	"testing"
)

// baseRequest asks for DeepSeek-V4.1-Flash with DSpark at K=5 on coding traffic with
// thinking off, against the registry set alone.
func baseRequest() Request {
	return Request{
		Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: 5,
		Hardware: "b300",
		Registry: []*ProfileSet{validProfileSet()},
		Workload: Workload{Category: "coding", Thinking: ThinkingOff},
	}
}

func TestResolvePrecedence(t *testing.T) {
	workloadV := &Value{NumSpecTokens: 5, AcceptanceLength: f(3.9), Source: "chat trace"}
	scenarioV := &Value{NumSpecTokens: 5, AcceptanceLength: f(3.2), Source: "forced AL"}
	cliV := &Value{NumSpecTokens: 5, AcceptanceRate: f(0.5)}
	pinnedSet := &ProfileSet{Kind: Kind, Name: "pinned", Profiles: []Profile{{
		Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: 5,
		Thinking: ThinkingOff, Category: "coding", AcceptanceLength: 2.5,
		Method: "assumed", Rationale: "what-if"}}}

	cases := []struct {
		name   string
		mutate func(*Request)
		al     float64
		layer  Layer
		source string
	}{
		{"registry default", func(*Request) {}, 4.07, LayerRegistry,
			measuredProvenance().Source},
		{"workload outranks registry", func(r *Request) { r.Workload.Value = workloadV },
			3.9, LayerWorkload, "chat trace"},
		{"scenario value outranks workload", func(r *Request) {
			r.Workload.Value = workloadV
			r.Scenario = &Pin{Value: scenarioV}
		}, 3.2, LayerScenario, "forced AL"},
		{"scenario set outranks workload", func(r *Request) {
			r.Workload.Value = workloadV
			r.Scenario = &Pin{Profiles: "pinned"}
			r.ScenarioSet = pinnedSet
		}, 2.5, LayerScenario, ""},
		{"cli outranks everything", func(r *Request) {
			r.Workload.Value = workloadV
			r.Scenario = &Pin{Value: scenarioV}
			r.CLI = cliV
		}, 3.5, LayerCLI, ""},
		{"a fallback-only pin leaves the registry deciding", func(r *Request) {
			r.Scenario = &Pin{FallbackCategory: "coding"}
		}, 4.07, LayerRegistry, measuredProvenance().Source},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := baseRequest()
			c.mutate(&req)
			got, err := Resolve(req)
			if err != nil {
				t.Fatal(err)
			}
			if !near(got.AcceptanceLength, c.al) || got.Layer != c.layer || got.Source != c.source {
				t.Errorf("got AL %v from %s (source %q), want %v from %s (source %q)",
					got.AcceptanceLength, got.Layer, got.Source, c.al, c.layer, c.source)
			}
			if len(got.PerPositionRates) != 5 {
				t.Errorf("rates %v, want 5 positions", got.PerPositionRates)
			}
		})
	}
}

// With no rates stated, the resolution carries vLLM's minimum-variance schedule and says
// it was derived; stated rates pass through as stated.
func TestResolveRates(t *testing.T) {
	got, err := Resolve(baseRequest())
	if err != nil {
		t.Fatal(err)
	}
	if !got.RatesDerived || !sameRates(got.PerPositionRates, []float64{1, 1, 1, 0.07, 0}) {
		t.Errorf("derived rates = %v (derived %v)", got.PerPositionRates, got.RatesDerived)
	}
	if !near(got.AcceptanceRate(), 0.614) {
		t.Errorf("AcceptanceRate() = %v, want 0.614", got.AcceptanceRate())
	}

	req := baseRequest()
	stated := []float64{0.9, 0.8, 0.6, 0.4, 0.2}
	req.CLI = &Value{NumSpecTokens: 5, PerPositionRates: stated}
	got, err = Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.RatesDerived || !sameRates(got.PerPositionRates, stated) || !near(got.AcceptanceLength, 3.9) {
		t.Errorf("stated rates resolved to %+v", got)
	}
}

func TestResolveRecordsProvenance(t *testing.T) {
	req := baseRequest()
	req.Hardware = "h200"
	got, err := Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if got.Set != "inferencex-golden-al" || got.Profile == nil || got.Profile.Thinking != ThinkingOff {
		t.Errorf("did not record the set and profile: %+v", got)
	}
	if got.MeasuredOn != "b300" {
		t.Errorf("MeasuredOn = %q; a curve measured on other hardware must be recorded", got.MeasuredOn)
	}
	req.Hardware = "B300"
	if got, _ := Resolve(req); got.MeasuredOn != "" {
		t.Errorf("MeasuredOn = %q on the hardware it was measured on", got.MeasuredOn)
	}
}

func TestResolveFallback(t *testing.T) {
	for _, c := range []struct {
		name     string
		mutate   func(*Request)
		from     string
		errorHas string
	}{
		{"uncovered category is refused", func(r *Request) { r.Workload.Category = "chat" },
			"", "none is applied silently"},
		{"no category is refused", func(r *Request) { r.Workload.Category = "" },
			"", "states no category"},
		{"cli fallback is recorded", func(r *Request) {
			r.Workload.Category, r.FallbackCategory = "chat", "coding"
		}, "chat", ""},
		{"scenario fallback is recorded", func(r *Request) {
			r.Workload.Category = "chat"
			r.Scenario = &Pin{FallbackCategory: "coding"}
		}, "chat", ""},
		{"fallback for no category is recorded", func(r *Request) {
			r.Workload.Category, r.FallbackCategory = "", "coding"
		}, "(none)", ""},
		{"scenario fallback overrides the cli's", func(r *Request) {
			r.Workload.Category, r.FallbackCategory = "chat", "math"
			r.Scenario = &Pin{FallbackCategory: "coding"}
		}, "chat", ""},
		{"a fallback with no curve is refused", func(r *Request) {
			r.Workload.Category, r.FallbackCategory = "chat", "math"
		}, "", `fallback category "math" has none`},
		{"a covered category is not replaced", func(r *Request) { r.FallbackCategory = "math" },
			"", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			req := baseRequest()
			c.mutate(&req)
			got, err := Resolve(req)
			if c.errorHas != "" {
				if err == nil || !strings.Contains(err.Error(), c.errorHas) {
					t.Fatalf("want an error containing %q, got %v", c.errorHas, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if got.FallbackFrom != c.from {
				t.Errorf("FallbackFrom = %q, want %q", got.FallbackFrom, c.from)
			}
		})
	}
}

func TestResolveRefuses(t *testing.T) {
	for _, c := range []struct {
		name     string
		mutate   func(*Request)
		errorHas string
	}{
		{"no speculation", func(r *Request) { r.NumSpecTokens = 0 }, "at least 1"},
		{"K with no curve", func(r *Request) { r.NumSpecTokens = 6 }, "no profile for deepseek-v4.1-flash/dspark/k6"},
		{"another method", func(r *Request) { r.SpeculativeMethod = "mtp" }, "no profile"},
		{"no thinking mode", func(r *Request) { r.Workload.Thinking = "" }, "no thinking mode"},
		{"unknown thinking mode", func(r *Request) { r.Workload.Thinking = "maybe" }, "not one of"},
		{"no model", func(r *Request) { r.Model = "" }, "needs the model"},
		{"no registry", func(r *Request) { r.Registry = nil }, "none was supplied"},
		{"two sets match", func(r *Request) {
			other := validProfileSet()
			other.Name = "other"
			r.Registry = append(r.Registry, other)
		}, "matches in both"},
		{"workload value at another K", func(r *Request) {
			r.Workload.Value = &Value{NumSpecTokens: 3, AcceptanceLength: f(2.5)}
		}, "stated for num_spec_tokens 3"},
		{"scenario value at another K", func(r *Request) {
			r.Scenario = &Pin{Value: &Value{NumSpecTokens: 3, AcceptanceLength: f(2.5)}}
		}, "stated for num_spec_tokens 3"},
		{"cli value at another K", func(r *Request) {
			r.CLI = &Value{NumSpecTokens: 4, AcceptanceRate: f(0.5)}
		}, "stated for num_spec_tokens 4"},
		{"invalid cli value", func(r *Request) {
			r.CLI = &Value{NumSpecTokens: 5, AcceptanceLength: f(7)}
		}, "outside [1, 6]"},
		{"invalid pin", func(r *Request) { r.Scenario = &Pin{} }, "pins nothing"},
		{"named set not supplied", func(r *Request) {
			r.Scenario = &Pin{Profiles: "pinned"}
		}, `"pinned", which was not supplied`},
		{"named set has no curve, and the registry is not consulted", func(r *Request) {
			r.Scenario = &Pin{Profiles: "empty"}
			r.ScenarioSet = &ProfileSet{Kind: Kind, Name: "empty"}
		}, "scenario acceptance: no profile"},
	} {
		t.Run(c.name, func(t *testing.T) {
			req := baseRequest()
			c.mutate(&req)
			_, err := Resolve(req)
			if err == nil || !strings.Contains(err.Error(), c.errorHas) {
				t.Fatalf("want an error containing %q, got %v", c.errorHas, err)
			}
		})
	}
}
