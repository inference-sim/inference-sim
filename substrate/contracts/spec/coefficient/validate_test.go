package coefficient

import (
	"fmt"
	"math"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func validSet() *Set {
	return &Set{
		Kind: "CoefficientSet", Name: "cost-model-primitives-h200",
		Coefficients: []Entry{
			{Name: "gemm_eps_max_bf16", Value: 0.72,
				Units: vocab.UnitDimensionless, Method: vocab.MethodMeasured,
				Fitted: true, Scope: Scope{Hardware: []string{"h200"}},
				Sources: []Source{{Kind: vocab.SourceModel, Cite: "operator table",
					Role: vocab.RolePrimary}}},
			{Name: "gemm_m_half_bf16", Value: 96, Units: vocab.UnitTokens,
				Method: vocab.MethodMeasured, Fitted: true,
				Scope: Scope{Hardware: []string{"h200"}},
				Sources: []Source{{Kind: vocab.SourceModel, Cite: "same fit",
					Role: vocab.RolePrimary}}},
			{Name: "host_output_token", Value: 45.9,
				Units: vocab.UnitMicrosecondsPerToken, Method: vocab.MethodAssumed,
				Scope:     Scope{Hardware: []string{"h200"}},
				Rationale: "scaled from a related fit; see the realization document"},
		},
	}
}

func TestValidSetPasses(t *testing.T) {
	if p := validSet().Validate(); !p.OK() {
		t.Fatalf("a valid set was rejected:\n%s", p.Error())
	}
}

func TestRejects(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*Set)
	}{
		{"wrong kind", func(s *Set) { s.Kind = "Coefficients" }},
		{"no name", func(s *Set) { s.Name = "" }},
		{"no entries", func(s *Set) { s.Coefficients = nil }},
		{"duplicate entry name", func(s *Set) {
			s.Coefficients = append(s.Coefficients, s.Coefficients[0])
		}},
		{"unknown unit", func(s *Set) { s.Coefficients[0].Units = "parsecs" }},
		{"unknown method", func(s *Set) { s.Coefficients[0].Method = "divined" }},
		{"empty scope", func(s *Set) { s.Coefficients[0].Scope = Scope{} }},
		{"not_charged with a value", func(s *Set) {
			s.Coefficients[0].Method = vocab.MethodNotCharged
		}},
		{"copied without a source scope", func(s *Set) {
			s.Coefficients[0].Method = vocab.MethodCopied
		}},
		{"unknown source kind", func(s *Set) {
			s.Coefficients[0].Sources[0].Kind = "rumour"
		}},
		{"source without a citation", func(s *Set) {
			s.Coefficients[0].Sources[0].Cite = ""
		}},
		{"inverted interval", func(s *Set) {
			s.Coefficients[0].CI95 = &Interval{Low: 1, High: 0}
		}},
		{"value outside its interval", func(s *Set) {
			s.Coefficients[0].CI95 = &Interval{Low: 0.8, High: 0.9}
		}},
		// A non-finite coefficient value poisons the latency model, and a NaN would also
		// escape the interval checks, so it is rejected like a non-finite hardware figure.
		{"NaN value", func(s *Set) { s.Coefficients[0].Value = math.NaN() }},
		{"Inf value", func(s *Set) { s.Coefficients[0].Value = math.Inf(1) }},
		{"non-finite interval endpoint", func(s *Set) {
			s.Coefficients[0].CI95 = &Interval{Low: math.Inf(-1), High: math.Inf(1)}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := validSet()
			tc.mutate(s)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
}

// The provenance/evidence rules below were migrated from blis-registry's Python validator
// (validator/schema.py) so the registry can drop it and validation lives in one place.
// Each is tested positive and negative, matching schema.py's exact scoping.

// Every method but measured must carry its reasoning. This was a warning (assumed only);
// it is now an error for all non-measured methods, as the Python validator required.
func TestRationaleRequiredForEveryMethodButMeasured(t *testing.T) {
	for _, m := range []vocab.Method{
		vocab.MethodLiterature, vocab.MethodVendorSpec, vocab.MethodCopied,
		vocab.MethodAssumed, vocab.MethodNotCharged,
	} {
		s := validSet()
		e := &s.Coefficients[2] // the assumed entry
		e.Method, e.Rationale, e.Fitted = m, "", false
		// Satisfy the OTHER companions each method needs, so rationale is the sole failure.
		switch m {
		case vocab.MethodLiterature, vocab.MethodVendorSpec:
			e.Sources = []Source{{Kind: vocab.SourceDatasheet, Cite: "x", Role: vocab.RolePrimary}}
		case vocab.MethodCopied:
			e.CopiedFrom = "hardware=h100"
		case vocab.MethodNotCharged:
			e.Value = 0
		}
		if p := s.Validate(); p.OK() {
			t.Errorf("method %q without a rationale should be rejected", m)
		}
		e.Rationale = "a weighable reason"
		if p := s.Validate(); !p.OK() {
			t.Errorf("method %q with a rationale should pass:\n%s", m, p.Error())
		}
	}
}

func TestMeasuredNeedsNoRationale(t *testing.T) {
	s := validSet()
	s.Coefficients[0].Rationale = "" // entry[0] is measured
	if p := s.Validate(); !p.OK() {
		t.Errorf("a measured value needs no rationale:\n%s", p.Error())
	}
}

// sources are required for literature and vendor_spec specifically — not for measured.
func TestSourcesRequiredForLiteratureAndVendorSpec(t *testing.T) {
	for _, m := range []vocab.Method{vocab.MethodLiterature, vocab.MethodVendorSpec} {
		s := validSet()
		e := &s.Coefficients[0]
		e.Method, e.Fitted, e.Rationale, e.Sources = m, false, "cited below", nil
		if p := s.Validate(); p.OK() {
			t.Errorf("method %q without sources should be rejected", m)
		}
		e.Sources = []Source{{Kind: vocab.SourcePublication, Cite: "arXiv:1", Role: vocab.RolePrimary}}
		if p := s.Validate(); !p.OK() {
			t.Errorf("method %q with a source should pass:\n%s", m, p.Error())
		}
	}
}

// A measured value may be uncited — the registry carries such entries — so it is neither
// an error nor a warning now (the broad "evidenced without source" warning is gone).
func TestMeasuredWithoutSourceIsClean(t *testing.T) {
	s := validSet()
	s.Coefficients[0].Sources = nil
	p := s.Validate()
	if !p.OK() {
		t.Errorf("an uncited measured value should pass:\n%s", p.Error())
	}
	for _, pr := range p.All() {
		if strings.Contains(pr.Message, "source") {
			t.Errorf("measured-without-source should be silent, got: %s", pr)
		}
	}
}

// fitted: true is coherent only with a measured method.
func TestFittedTrueRequiresMeasured(t *testing.T) {
	s := validSet()
	s.Coefficients[2].Fitted = true // entry[2] is assumed
	if p := s.Validate(); p.OK() {
		t.Error("fitted: true on a non-measured method should be rejected")
	}
	s.Coefficients[2].Fitted = false
	if p := s.Validate(); !p.OK() {
		t.Errorf("the fixture should otherwise be valid:\n%s", p.Error())
	}
}

// A zero value is reserved for method not_charged (and not_charged must be zero).
func TestZeroValueRequiresNotCharged(t *testing.T) {
	s := validSet()
	s.Coefficients[0].Value = 0 // measured, so this must be rejected
	if p := s.Validate(); p.OK() {
		t.Error("a zero value under a non-not_charged method should be rejected")
	}
	s2 := validSet()
	e := &s2.Coefficients[2] // assumed, has a rationale
	e.Method, e.Value = vocab.MethodNotCharged, 0
	if p := s2.Validate(); !p.OK() {
		t.Errorf("not_charged with a zero value and a rationale should pass:\n%s", p.Error())
	}
}

// Strict parse: an unknown top-level key or an unknown scope key is rejected at decode —
// a custom UnmarshalYAML would otherwise drop them silently (the KnownFields loss).
func TestStrictUnknownKeysRejectedAtDecode(t *testing.T) {
	cases := map[string]string{
		"unknown top-level key": `kind: CoefficientSet
name: s
backend: vllm
coefficients:
  - x: {value: 1, units: dimensionless, method: measured, fitted: false, scope: {tp: [8]}}
`,
		"fabric as a scalar, not a list": `kind: CoefficientSet
name: s
coefficients:
  - x: {value: 1, units: dimensionless, method: measured, fitted: false, scope: {fabric: roce-200g}}
`,
		"unknown scope key": `kind: CoefficientSet
name: s
coefficients:
  - x: {value: 1, units: dimensionless, method: measured, fitted: false, scope: {dtype: [bf16]}}
`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			var s Set
			if err := yaml.Unmarshal([]byte(body), &s); err == nil {
				t.Fatal("expected a decode error for the unknown key, got none")
			}
		})
	}
}

// fitted is a required field whose Go zero value (false) is valid, so its absence is
// caught at decode rather than slipping through as an intentional false — matching the
// Python validator, which requires it present. Present (either value) decodes cleanly.
func TestFittedPresenceRequiredAtDecode(t *testing.T) {
	const tmpl = `kind: CoefficientSet
name: s
coefficients:
  - x:
      value: 1
      units: dimensionless
      method: measured
%s      scope:
        hardware: [h200]
`
	t.Run("absent fitted is rejected", func(t *testing.T) {
		var s Set
		if err := yaml.Unmarshal([]byte(fmt.Sprintf(tmpl, "")), &s); err == nil {
			t.Fatal("an entry without a fitted key should fail to decode, got none")
		}
	})
	for _, present := range []string{"      fitted: true\n", "      fitted: false\n"} {
		t.Run("present fitted decodes: "+strings.TrimSpace(present), func(t *testing.T) {
			var s Set
			if err := yaml.Unmarshal([]byte(fmt.Sprintf(tmpl, present)), &s); err != nil {
				t.Fatalf("an entry with a fitted key should decode, got: %v", err)
			}
		})
	}
}

// sources parity with the registry's _check_sources: an unknown field inside a source
// object, and a present-but-empty sources list, are both rejected at decode — the custom
// unmarshal would otherwise drop the stray field and ignore the empty list.
func TestSourcesStrictAtDecode(t *testing.T) {
	cases := map[string]string{
		"unknown field in a source object": `kind: CoefficientSet
name: s
coefficients:
  - x:
      value: 1
      units: dimensionless
      method: literature
      fitted: false
      rationale: from a paper
      scope: {tp: [8]}
      sources:
        - {kind: publication, cite: "arXiv:1", role: primary, bogus: 1}
`,
		"empty sources list present": `kind: CoefficientSet
name: s
coefficients:
  - x:
      value: 1
      units: dimensionless
      method: measured
      fitted: false
      scope: {tp: [8]}
      sources: []
`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			var s Set
			if err := yaml.Unmarshal([]byte(body), &s); err == nil {
				t.Fatalf("expected a decode error, got none")
			}
		})
	}
}

func TestScopeEmpty(t *testing.T) {
	if !(Scope{}).Empty() {
		t.Error("a zero scope should report empty")
	}
	if (Scope{TP: []int{8}}).Empty() {
		t.Error("a scope with one dimension should not report empty")
	}
}

// TestNonFiniteCoefficientRejectedFromYAML pins the non-finite check end-to-end from the
// wire form: YAML's own `.nan`, `.inf` and `-.inf` literals are refused when the set is
// decoded (SUB-12), naming the field, since a non-finite coefficient poisons the latency
// model the registry feeds. Validate refuses the same values in a set built in Go
// (TestRejects, "NaN value" and "Inf value").
func TestNonFiniteCoefficientRejectedFromYAML(t *testing.T) {
	const tmpl = `kind: CoefficientSet
name: cost-model-primitives-h200
coefficients:
  - gemm_eps_max_bf16:
      value: %s
      units: dimensionless
      method: measured
      fitted: true
      scope:
        hardware: [h200]
      sources:
        - {kind: model, cite: operator table, role: primary}
`
	for _, lit := range []string{".nan", ".inf", "-.inf", ".NaN"} {
		t.Run(lit, func(t *testing.T) {
			var s Set
			err := yaml.Unmarshal([]byte(fmt.Sprintf(tmpl, lit)), &s)
			if err == nil || !strings.Contains(err.Error(), "value: "+lit+" is not a finite number") {
				t.Fatalf("a %s coefficient value must be refused at decode, naming the field; got %v", lit, err)
			}
		})
	}
}

func calibratedEntry() Entry {
	return Entry{Name: "host_launch_eager_per_kernel", Value: 4.2,
		Units: vocab.UnitMicrosecondsPerLayer, Method: vocab.MethodCalibrated, Fitted: true,
		Scope:     Scope{Hardware: []string{"h100"}, Model: []string{"gpt-oss-120b"}},
		Rationale: "fitted on output throughput; holdout grouped by model x hardware x source",
		Sources: []Source{{Kind: vocab.SourceMeasuredCase,
			Cite: "synthetic-h100-tp4@0123abcd", Role: vocab.RolePrimary}},
		Validated: MetricList{"output_tps", "tpot"}}
}

// A calibrated value is fitted through the cost model to measured cases, so it must name
// them and the held-out metrics it was checked on; `derived` is a catalog provenance and
// is not a method at all.
func TestCalibratedRequiresMeasuredCasesAndHoldout(t *testing.T) {
	ok := validSet()
	ok.Coefficients = append(ok.Coefficients, calibratedEntry())
	if p := ok.Validate(); !p.OK() {
		t.Fatalf("a well-formed calibrated entry was rejected:\n%s", p.Error())
	}
	cases := []struct {
		name   string
		mutate func(*Entry)
	}{
		{"no sources", func(e *Entry) { e.Sources = nil }},
		{"sources without a measured case", func(e *Entry) {
			e.Sources = []Source{{Kind: vocab.SourcePublication, Cite: "x", Role: vocab.RolePrimary}}
		}},
		{"no held-out metrics", func(e *Entry) { e.Validated = nil }},
		{"no rationale", func(e *Entry) { e.Rationale = "" }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := validSet()
			e := calibratedEntry()
			tc.mutate(&e)
			s.Coefficients = append(s.Coefficients, e)
			if p := s.Validate(); p.OK() {
				t.Fatal("expected rejection, got none")
			}
		})
	}
	if vocab.Method("derived").Valid() {
		t.Error("derived is a catalog provenance, not a coefficient method")
	}
}
