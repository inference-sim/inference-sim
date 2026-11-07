package acceptance

import (
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func f(x float64) *float64 { return &x }
func i(x int) *int         { return &x }

func measuredProvenance() *Provenance {
	return &Provenance{
		Source:      "https://github.com/SemiAnalysisAI/InferenceX/blob/main/inferencex-e2e/infx/golden_al_distribution/dsv41flash_dspark.yaml",
		Run:         "https://github.com/SemiAnalysisAI/InferenceX/actions/runs/34493175056",
		Hardware:    "b300",
		EngineImage: "vllm/vllm-openai:deepseekv41-flash-0909",
		TP:          4,
		Sampling:    &Sampling{Temperature: f(1.0), MaxTokens: 4096},
	}
}

func validProfileSet() *ProfileSet {
	return &ProfileSet{
		Kind: Kind, Name: "inferencex-golden-al",
		Caveats: []string{"coding only"},
		Profiles: []Profile{
			{Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: 5,
				Thinking: ThinkingOff, Category: "coding", AcceptanceLength: 4.07,
				Method: vocab.MethodMeasured, Provenance: measuredProvenance()},
			{Model: "deepseek-v4.1-flash", SpeculativeMethod: "dspark", NumSpecTokens: 5,
				Thinking: ThinkingOn, Category: "coding", AcceptanceLength: 3.51,
				Method: vocab.MethodMeasured, Provenance: measuredProvenance()},
			{Model: "glm-5.3", SpeculativeMethod: "mtp", NumSpecTokens: 3,
				Thinking: ThinkingOn, Category: "coding", AcceptanceLength: 2.99,
				Method: vocab.MethodTransferred, TransferredFrom: "glm-5.2-fp8/mtp/k3/thinking-on/coding",
				TransferRule: "the value measured on GLM-5.2 is carried unchanged, because GLM-5.3 shares its base and layer topology",
				Rationale:  "provisional; shares GLM-5.2's topology",
				Provenance: &Provenance{Source: "glm5.3_mtp.yaml", Hardware: "b300"}},
		},
	}
}

func TestValidProfileSetPasses(t *testing.T) {
	if p := validProfileSet().Validate(); !p.OK() {
		t.Fatalf("a valid set was rejected:\n%s", p.Error())
	}
}

func TestProfileSetAcceptsStatedRates(t *testing.T) {
	s := validProfileSet()
	// Rounded the way vLLM logs them: within RatesSumTolerance of AL-1, not exactly on it.
	s.Profiles[0].PerPositionRates = []float64{0.951, 0.802, 0.663, 0.451, 0.201}
	if p := s.Validate(); !p.OK() {
		t.Fatalf("rates summing to AL-1 within tolerance were rejected:\n%s", p.Error())
	}
}

func TestProfileSetRejects(t *testing.T) {
	cases := []struct {
		name   string
		field  string
		mutate func(*ProfileSet)
	}{
		{"wrong kind", "kind", func(s *ProfileSet) { s.Kind = "CoefficientSet" }},
		{"no name", "name", func(s *ProfileSet) { s.Name = "" }},
		{"no profiles", "profiles", func(s *ProfileSet) { s.Profiles = nil }},
		{"empty caveat", "caveats[0]", func(s *ProfileSet) { s.Caveats = []string{""} }},
		{"duplicate key", "profiles[3]", func(s *ProfileSet) {
			s.Profiles = append(s.Profiles, s.Profiles[0])
		}},
		{"no model", "profiles[0].model", func(s *ProfileSet) { s.Profiles[0].Model = "" }},
		{"no method", "profiles[0].speculative_method", func(s *ProfileSet) {
			s.Profiles[0].SpeculativeMethod = ""
		}},
		{"no thinking", "profiles[0].thinking", func(s *ProfileSet) { s.Profiles[0].Thinking = "" }},
		{"unknown thinking", "profiles[0].thinking", func(s *ProfileSet) {
			s.Profiles[0].Thinking = "high"
		}},
		{"no category", "profiles[0].category", func(s *ProfileSet) { s.Profiles[0].Category = "" }},
		{"capitalised category", "profiles[0].category", func(s *ProfileSet) {
			s.Profiles[0].Category = "Coding"
		}},
		{"K zero", "profiles[0].num_spec_tokens", func(s *ProfileSet) { s.Profiles[0].NumSpecTokens = 0 }},
		{"AL below one", "profiles[0].acceptance_length", func(s *ProfileSet) {
			s.Profiles[0].AcceptanceLength = 0.9
		}},
		{"AL above K+1", "profiles[0].acceptance_length", func(s *ProfileSet) {
			s.Profiles[0].AcceptanceLength = 6.01
		}},
		{"AL NaN", "profiles[0].acceptance_length", func(s *ProfileSet) {
			s.Profiles[0].AcceptanceLength = math.NaN()
		}},
		{"rates of the wrong length", "profiles[0].per_position_rates", func(s *ProfileSet) {
			s.Profiles[0].PerPositionRates = []float64{1, 1, 1.07}
		}},
		{"rising rates", "profiles[0].per_position_rates", func(s *ProfileSet) {
			s.Profiles[0].PerPositionRates = []float64{1, 1, 0.07, 1, 0}
		}},
		{"rates not summing to AL-1", "profiles[0].per_position_rates", func(s *ProfileSet) {
			s.Profiles[0].PerPositionRates = []float64{1, 1, 1, 0, 0}
		}},
		{"unknown evidence method", "profiles[0].method", func(s *ProfileSet) {
			s.Profiles[0].Method = "divined"
		}},
		{"not_charged", "profiles[0].method", func(s *ProfileSet) {
			s.Profiles[0].Method = vocab.MethodNotCharged
		}},
		{"calibrated", "profiles[0].method", func(s *ProfileSet) {
			s.Profiles[0].Method = vocab.MethodCalibrated
		}},
		{"transferred without transferred_from", "profiles[2].transferred_from", func(s *ProfileSet) {
			s.Profiles[2].TransferredFrom = ""
		}},
		{"transferred without a rule", "profiles[2].transfer_rule", func(s *ProfileSet) {
			s.Profiles[2].TransferRule = ""
		}},
		{"transferred_from on a measured profile", "profiles[0].transferred_from", func(s *ProfileSet) {
			s.Profiles[0].TransferredFrom = "x"
		}},
		{"transferred without rationale", "profiles[2].rationale", func(s *ProfileSet) {
			s.Profiles[2].Rationale = ""
		}},
		{"measured without provenance", "profiles[0].provenance", func(s *ProfileSet) {
			s.Profiles[0].Provenance = nil
		}},
		{"literature without provenance", "profiles[0].provenance", func(s *ProfileSet) {
			s.Profiles[0].Method, s.Profiles[0].Rationale = vocab.MethodLiterature, "paper"
			s.Profiles[0].Provenance = nil
		}},
		{"provenance without source", "profiles[0].provenance.source", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Source = ""
		}},
		{"measured without hardware", "profiles[0].provenance.hardware", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Hardware = ""
		}},
		{"measured without image", "profiles[0].provenance.engine_image", func(s *ProfileSet) {
			s.Profiles[0].Provenance.EngineImage = ""
		}},
		{"measured without sampling", "profiles[0].provenance.sampling", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Sampling = nil
		}},
		{"sampling without temperature", "profiles[0].provenance.sampling.temperature", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Sampling.Temperature = nil
		}},
		{"negative temperature", "profiles[0].provenance.sampling.temperature", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Sampling.Temperature = f(-1)
		}},
		{"top_p zero", "profiles[0].provenance.sampling.top_p", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Sampling.TopP = f(0)
		}},
		{"top_k below -1", "profiles[0].provenance.sampling.top_k", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Sampling.TopK = i(-2)
		}},
		{"negative prompts", "profiles[0].provenance.prompts", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Prompts = -1
		}},
		{"empty setting", "profiles[0].provenance.settings", func(s *ProfileSet) {
			s.Profiles[0].Provenance.Settings = map[string]string{"rejection_sample_method": ""}
		}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := validProfileSet()
			c.mutate(s)
			p := s.Validate()
			if p.OK() {
				t.Fatal("expected rejection, got none")
			}
			for _, it := range p.Errors() {
				if it.Path == c.field {
					return
				}
			}
			t.Errorf("no error at %s:\n%s", c.field, p.Error())
		})
	}
}

// An assumed profile needs a rationale but no provenance: it rests on reasoning, which
// the rationale states, and has no measurement to cite.
func TestAssumedProfileNeedsNoProvenance(t *testing.T) {
	s := validProfileSet()
	s.Profiles[0].Method = vocab.MethodAssumed
	s.Profiles[0].Provenance = nil
	s.Profiles[0].Rationale = "scaled from the thinking-on curve"
	if p := s.Validate(); !p.OK() {
		t.Fatalf("an assumed profile with a rationale was rejected:\n%s", p.Error())
	}
}

func TestValue(t *testing.T) {
	cases := []struct {
		name   string
		v      Value
		ok     bool
		length float64
	}{
		{"length", Value{NumSpecTokens: 5, AcceptanceLength: f(4.07)}, true, 4.07},
		{"rate converts at its K", Value{NumSpecTokens: 4, AcceptanceRate: f(0.5)}, true, 3},
		{"rates alone state AL", Value{NumSpecTokens: 2, PerPositionRates: []float64{0.8, 0.5}}, true, 2.3},
		{"length with agreeing rates", Value{NumSpecTokens: 2, AcceptanceLength: f(2.3),
			PerPositionRates: []float64{0.8, 0.5}}, true, 2.3},
		{"rate with agreeing rates", Value{NumSpecTokens: 2, AcceptanceRate: f(0.65),
			PerPositionRates: []float64{0.8, 0.5}}, true, 2.3},
		{"no K", Value{AcceptanceLength: f(2)}, false, 0},
		{"nothing stated", Value{NumSpecTokens: 3}, false, 0},
		{"length and rate both", Value{NumSpecTokens: 3, AcceptanceLength: f(2), AcceptanceRate: f(0.3)}, false, 0},
		{"length above K+1", Value{NumSpecTokens: 3, AcceptanceLength: f(4.5)}, false, 0},
		{"length below 1", Value{NumSpecTokens: 3, AcceptanceLength: f(0.5)}, false, 0},
		{"rate above 1", Value{NumSpecTokens: 3, AcceptanceRate: f(1.2)}, false, 0},
		{"rate NaN", Value{NumSpecTokens: 3, AcceptanceRate: f(math.NaN())}, false, 0},
		{"disagreeing rates", Value{NumSpecTokens: 2, AcceptanceLength: f(2.9),
			PerPositionRates: []float64{0.8, 0.5}}, false, 0},
		{"rising rates alone", Value{NumSpecTokens: 2, PerPositionRates: []float64{0.5, 0.8}}, false, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			p := c.v.Validate()
			if p.OK() != c.ok {
				t.Fatalf("Validate OK = %v, want %v:\n%s", p.OK(), c.ok, p.Error())
			}
			if c.ok && !near(c.v.Length(), c.length) {
				t.Errorf("Length() = %v, want %v", c.v.Length(), c.length)
			}
		})
	}
}

func TestPin(t *testing.T) {
	val := &Value{NumSpecTokens: 3, AcceptanceLength: f(2.8)}
	cases := []struct {
		name string
		pin  Pin
		ok   bool
	}{
		{"set", Pin{Profiles: "inferencex-golden-al"}, true},
		{"set with fallback", Pin{Profiles: "inferencex-golden-al", FallbackCategory: "coding"}, true},
		{"fallback alone", Pin{FallbackCategory: "coding"}, true},
		{"value", Pin{Value: val}, true},
		{"empty", Pin{}, false},
		{"set and value", Pin{Profiles: "x", Value: val}, false},
		{"value with fallback", Pin{Value: val, FallbackCategory: "coding"}, false},
		{"malformed fallback", Pin{FallbackCategory: "Coding"}, false},
		{"invalid value", Pin{Value: &Value{NumSpecTokens: 3}}, false},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if p := c.pin.Validate(); p.OK() != c.ok {
				t.Errorf("Validate OK = %v, want %v:\n%s", p.OK(), c.ok, p.Error())
			}
		})
	}
}

// YAML 1.1 reads a bare off as false. Thinking is a string type, so yaml.v3 keeps the
// text either way, and an unquoted mode decodes as written.
func TestThinkingDecodesUnquoted(t *testing.T) {
	var pr Profile
	if err := yaml.Unmarshal([]byte("thinking: off\n"), &pr); err != nil {
		t.Fatal(err)
	}
	if pr.Thinking != ThinkingOff {
		t.Errorf("thinking: off decoded as %q", pr.Thinking)
	}
}

func TestValueRejectsUnknownKeys(t *testing.T) {
	var v Value
	err := yaml.Unmarshal([]byte("num_spec_tokens: 3\nacceptance_lenght: 2.5\n"), &v)
	if err == nil || !strings.Contains(err.Error(), "acceptance_lenght") {
		t.Fatalf("a misspelled key was not rejected: %v", err)
	}
}

func TestKeyString(t *testing.T) {
	got := validProfileSet().Profiles[0].Key().String()
	if want := "deepseek-v4.1-flash/dspark/k5/thinking-off/coding"; got != want {
		t.Errorf("Key().String() = %q, want %q", got, want)
	}
}
