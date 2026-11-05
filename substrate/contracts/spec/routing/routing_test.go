package routing

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

func fp(x float64) *float64 { return &x }

func curve(model string, phase Phase, ep int, pts ...Point) Profile {
	return Profile{Model: model, Category: "chat", Phase: phase, Placement: PlacementLinear,
		ExpertParallel: ep, Points: pts, Method: "measured",
		Provenance: &Provenance{Source: "hf://example@rev", Hardware: "b200", Tier: 2}}
}

func set(name string, role Role, ps ...Profile) *ProfileSet {
	return &ProfileSet{Kind: Kind, Name: name, Role: role, Profiles: ps}
}

func dsv41() *ProfileSet {
	return set("capture", RoleSensitivity,
		curve("m", PhaseDecode, 8, Point{BatchTokens: 16, TokenImbalance: 1.7, ExpertImbalance: 1.1},
			Point{BatchTokens: 256, TokenImbalance: 1.6, ExpertImbalance: 0.76}),
		curve("m", PhasePrefill, 8, Point{BatchTokens: 1024, TokenImbalance: 1.86, ExpertImbalance: 0.5}))
}

func TestValidSetPasses(t *testing.T) {
	if p := dsv41().Validate(); !p.OK() {
		t.Fatalf("a valid set was rejected:\n%s", p.Error())
	}
}

// A busiest rank below the mean is impossible, a curve must be ordered for interpolation,
// a width of one has no cross-rank term, and a sensitivity must say it is one.
func TestValidateRefuses(t *testing.T) {
	for name, mutate := range map[string]func(s *ProfileSet){
		"token below 1":  func(s *ProfileSet) { s.Profiles[0].Points[0].TokenImbalance = 0.9 },
		"unordered":      func(s *ProfileSet) { s.Profiles[0].Points[1].BatchTokens = 8 },
		"ep 1":           func(s *ProfileSet) { s.Profiles[0].ExpertParallel = 1 },
		"no role":        func(s *ProfileSet) { s.Role = "" },
		"eplb placement": func(s *ProfileSet) { s.Profiles[0].Placement = "eplb" },
		"repeated key":   func(s *ProfileSet) { s.Profiles[1].Phase = PhaseDecode },
		"no provenance":  func(s *ProfileSet) { s.Profiles[0].Provenance = nil },
	} {
		s := dsv41()
		mutate(s)
		if p := s.Validate(); p.OK() {
			t.Errorf("%s: accepted", name)
		}
	}
}

func TestResolvePrecedence(t *testing.T) {
	cli := &Value{TokenImbalance: 2.0, Source: "--flag"}
	wl := &Value{TokenImbalance: 1.5}
	pin := &Value{TokenImbalance: 1.8}
	def := set("default", RoleDefault, dsv41().Profiles...)
	base := Request{Model: "m", ExpertParallel: 8, Registry: []*ProfileSet{def},
		Workload: Workload{Category: "chat"}}

	cases := []struct {
		name  string
		edit  func(r *Request)
		layer Layer
		tok   float64
	}{
		{"registry default set", func(r *Request) {}, LayerRegistry, 1.6},
		{"workload over registry", func(r *Request) { r.Workload.Value = wl }, LayerWorkload, 1.5},
		{"scenario over workload", func(r *Request) {
			r.Workload.Value = wl
			r.Scenario = &Pin{Value: pin}
		}, LayerScenario, 1.8},
		{"cli over scenario", func(r *Request) {
			r.Scenario = &Pin{Value: pin}
			r.CLI = cli
		}, LayerCLI, 2.0},
	}
	for _, c := range cases {
		req := base
		c.edit(&req)
		r, err := Resolve(req)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if r.Layer != c.layer {
			t.Errorf("%s: layer %s, want %s", c.name, r.Layer, c.layer)
		}
		if got := r.At(PhaseDecode, 256).TokenImbalance; got != c.tok {
			t.Errorf("%s: token imbalance %v, want %v", c.name, got, c.tok)
		}
	}
}

// A sensitivity set is never a silent default: the registry layer skips it, and only a
// scenario that names it applies it.
func TestSensitivitySetOnlyByName(t *testing.T) {
	req := Request{Model: "m", ExpertParallel: 8, Registry: []*ProfileSet{dsv41()},
		Workload: Workload{Category: "chat"}, Hardware: "h100"}
	r, err := Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if r.Layer != LayerDefault || r.At(PhaseDecode, 64).TokenImbalance != 1 {
		t.Fatalf("an unnamed sensitivity set applied: %+v", r)
	}
	if !strings.Contains(r.Reason, "m/chat/decode/linear/ep8") {
		t.Errorf("the default does not say what was looked up: %q", r.Reason)
	}
	req.Scenario, req.ScenarioSet = &Pin{Profiles: "capture"}, dsv41()
	r, err = Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if r.Layer != LayerScenario || r.Set != "capture" || r.MeasuredOn != "b200" {
		t.Fatalf("named set not applied or not recorded: %+v", r)
	}
}

func TestResolveDefaults(t *testing.T) {
	for name, req := range map[string]Request{
		"no expert parallelism": {Model: "m", ExpertParallel: 1, CLI: &Value{TokenImbalance: 3}},
		"eplb":                  {Model: "m", ExpertParallel: 8, EPLB: true, Workload: Workload{Category: "chat"}, Registry: []*ProfileSet{set("d", RoleDefault, dsv41().Profiles...)}},
		"no category":           {Model: "m", ExpertParallel: 8, Registry: []*ProfileSet{set("d", RoleDefault, dsv41().Profiles...)}},
		"other width":           {Model: "m", ExpertParallel: 16, Workload: Workload{Category: "chat"}, Registry: []*ProfileSet{set("d", RoleDefault, dsv41().Profiles...)}},
	} {
		r, err := Resolve(req)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if r.Layer != LayerDefault || r.Reason == "" {
			t.Errorf("%s: want the recorded default, got %+v", name, r)
		}
		if l := r.At(PhaseDecode, 64); l.TokenImbalance != 1 || l.ExpertImbalance != 1 {
			t.Errorf("%s: default is not evenly loaded: %+v", name, l)
		}
	}
}

func TestResolveRefuses(t *testing.T) {
	for name, req := range map[string]Request{
		"named set not supplied": {Model: "m", ExpertParallel: 8, Scenario: &Pin{Profiles: "capture"}},
		"named set lacks key":    {Model: "m", ExpertParallel: 16, Workload: Workload{Category: "chat"}, Scenario: &Pin{Profiles: "capture"}, ScenarioSet: dsv41()},
		"named set under eplb":   {Model: "m", ExpertParallel: 8, EPLB: true, Workload: Workload{Category: "chat"}, Scenario: &Pin{Profiles: "capture"}, ScenarioSet: dsv41()},
		"value below 1":          {Model: "m", ExpertParallel: 8, CLI: &Value{TokenImbalance: 0.5}},
		"ambiguous":              {Model: "m", ExpertParallel: 8, Workload: Workload{Category: "chat"}, Registry: []*ProfileSet{set("a", RoleDefault, dsv41().Profiles...), set("b", RoleDefault, dsv41().Profiles...)}},
	} {
		if _, err := Resolve(req); err == nil {
			t.Errorf("%s: resolved", name)
		}
	}
}

// Between points the curve interpolates in log(batch); outside, it holds the end and
// reports the clamp. Prefill reads its own curve; a missing phase borrows the other's.
func TestLookup(t *testing.T) {
	r, err := Resolve(Request{Model: "m", ExpertParallel: 8, Workload: Workload{Category: "chat"},
		Scenario: &Pin{Profiles: "capture"}, ScenarioSet: dsv41()})
	if err != nil {
		t.Fatal(err)
	}
	mid := r.At(PhaseDecode, 64) // log-midpoint of 16 and 256
	if math.Abs(mid.TokenImbalance-1.65) > 1e-9 || mid.Clamped {
		t.Errorf("log interpolation: %+v", mid)
	}
	if lo := r.At(PhaseDecode, 4); lo.TokenImbalance != 1.7 || !lo.Clamped {
		t.Errorf("below the curve: %+v", lo)
	}
	if pf := r.At(PhasePrefill, 1024); pf.TokenImbalance != 1.86 || pf.ExpertImbalance != 0.5 {
		t.Errorf("prefill did not read its own curve: %+v", pf)
	}
	r.Prefill = nil
	if pf := r.At(PhasePrefill, 256); pf.TokenImbalance != 1.6 {
		t.Errorf("prefill without a curve did not borrow decode's: %+v", pf)
	}
}

// A coding workload may borrow another category's curve only when the pin asks, and the
// substitution is recorded.
func TestScenarioFallbackCategory(t *testing.T) {
	req := Request{Model: "m", ExpertParallel: 8, Workload: Workload{Category: "coding"},
		Scenario: &Pin{Profiles: "capture"}, ScenarioSet: dsv41()}
	if _, err := Resolve(req); err == nil {
		t.Fatal("a category with no curve resolved without a fallback")
	}
	req.Scenario.FallbackCategory = "chat"
	r, err := Resolve(req)
	if err != nil {
		t.Fatal(err)
	}
	if r.FallbackFrom != "coding" || r.Decode == nil || r.Decode.Category != "chat" {
		t.Fatalf("fallback not applied or not recorded: %+v", r)
	}
}

func TestValueUnknownKeyRejected(t *testing.T) {
	var v Value
	if err := yaml.Unmarshal([]byte("token_imbalance: 1.2\ntoken_imbalence: 2\n"), &v); err == nil {
		t.Fatal("a misspelt key was accepted")
	}
	if err := yaml.Unmarshal([]byte("token_imbalance: 1.2\n"), &v); err != nil || v.Expert() != 1 {
		t.Fatalf("expert imbalance should default to 1: %v %v", err, v.Expert())
	}
	_ = fp
}
