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
		{"transferred without a source scope or rule", func(s *Set) {
			s.Coefficients[0].Method = vocab.MethodTransferred
		}},
		{"transferred without a rule", func(s *Set) {
			s.Coefficients[0].Method = vocab.MethodTransferred
			s.Coefficients[0].TransferredFrom = "hardware=h100"
		}},
		{"transferred_from on a measured entry", func(s *Set) {
			s.Coefficients[0].TransferredFrom = "hardware=h100"
		}},
		{"roofline without the facts it was computed from", func(s *Set) {
			e := &s.Coefficients[0]
			e.Method, e.Fitted, e.Sources = vocab.MethodRoofline, false, nil
			e.Rationale = "larger of flops over peak and bytes over bandwidth"
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
		vocab.MethodLiterature, vocab.MethodVendorSpec, vocab.MethodTransferred,
		vocab.MethodRoofline, vocab.MethodAssumed, vocab.MethodNotCharged,
	} {
		s := validSet()
		e := &s.Coefficients[2] // the assumed entry
		e.Method, e.Rationale, e.Fitted = m, "", false
		// Satisfy the OTHER companions each method needs, so rationale is the sole failure.
		switch m {
		case vocab.MethodLiterature, vocab.MethodVendorSpec:
			e.Sources = []Source{{Kind: vocab.SourceDatasheet, Cite: "x", Role: vocab.RolePrimary}}
		case vocab.MethodTransferred:
			e.TransferredFrom, e.TransferRule = "hardware=h100", "scaled by the ratio of link bandwidths"
		case vocab.MethodRoofline:
			e.Sources = []Source{{Kind: vocab.SourceDatasheet, Cite: "catalog hardware/h200.yaml", Role: vocab.RolePrimary}}
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
		Units: vocab.UnitMicrosecondsPerXfer, Method: vocab.MethodCalibrated, Fitted: true,
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

// rate_basis (SUB-5): a closed word, only on a bytes_per_us entry, and a work rate must say
// why it is not a memory rate. Each case fails for exactly one reason.
func TestRateBasis(t *testing.T) {
	rate := func(mutate func(*Entry)) *Set {
		s := validSet()
		e := Entry{Name: "attention_decode_rate", Value: 1e6, Units: vocab.UnitBytesPerMicrosecond,
			Method: vocab.MethodMeasured, Fitted: true, Scope: Scope{Hardware: []string{"h200"}}}
		mutate(&e)
		s.Coefficients = append(s.Coefficients, e)
		return s
	}
	ok := []struct {
		name   string
		mutate func(*Entry)
	}{
		{"absent means memory", func(e *Entry) {}},
		{"memory", func(e *Entry) { e.RateBasis = vocab.RateBasisMemory }},
		{"work with a rationale", func(e *Entry) {
			e.RateBasis, e.Rationale = vocab.RateBasisWork, "a work rate over cached bytes"
		}},
	}
	for _, c := range ok {
		if p := rate(c.mutate).Validate(); !p.OK() {
			t.Errorf("%s was rejected:\n%s", c.name, p.Error())
		}
	}
	bad := []struct {
		name, want string
		mutate     func(*Entry)
	}{
		{"unknown word", "is not one of", func(e *Entry) { e.RateBasis = "cache" }},
		{"on a non-rate unit", "applies only to units", func(e *Entry) {
			e.Units, e.RateBasis = vocab.UnitDimensionless, vocab.RateBasisMemory
		}},
		{"work with no rationale", "requires a rationale", func(e *Entry) { e.RateBasis = vocab.RateBasisWork }},
		{"work with a blank rationale", "requires a rationale", func(e *Entry) {
			e.RateBasis, e.Rationale = vocab.RateBasisWork, "  "
		}},
	}
	for _, c := range bad {
		p := rate(c.mutate).Validate()
		if p.OK() || !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: want a problem containing %q, got ok=%v:\n%s", c.name, c.want, p.OK(), p.Error())
		}
	}
}

// D5, VAL-8, VAL-9, VAL-10: the optional evidence records are checked whenever present, and
// each way of being wrong has a failing input.
func TestEvidenceRecords(t *testing.T) {
	measured := func(mutate func(*Entry)) *Set {
		s := validSet()
		mutate(&s.Coefficients[0]) // a measured, fitted entry
		return s
	}
	impl := KernelImpl{Library: "flashinfer", Variant: "BatchDecodeWithPagedKVCache", CodeHash: "0123456789abcdef"}
	other := KernelImpl{Library: "flashinfer", Variant: "BatchDecodeWithPagedKVCache", CodeHash: "fedcba9876543210"}
	ok := []struct {
		name   string
		mutate func(*Entry)
	}{
		{"a kernel implementation", func(e *Entry) { e.KernelImpls = []KernelImpl{impl} }},
		// A transferred value names the implementation it is valid for and the one it was
		// measured on, which differ: the target of a transfer is not removed from the entry.
		{"a transferred value for a known target", func(e *Entry) {
			transferredTo(e, []KernelImpl{impl}, []KernelImpl{other})
		}},
		{"a transferred value measured on its own kernel (another chip)", func(e *Entry) {
			transferredTo(e, []KernelImpl{impl}, []KernelImpl{impl})
		}},
		{"a transferred value with no kernel key at all", func(e *Entry) { transferredTo(e, nil, nil) }},
		{"a measurement boundary", func(e *Entry) {
			e.Measures = &Measures{Time: vocab.TimeDevice, Execution: vocab.ExecutionCaptured, Harness: "aisimulate@2f0f4cd"}
		}},
		{"a fit range", func(e *Entry) { e.FitRange = []Bound{{Dim: "tokens", Low: 1, High: 8192}} }},
	}
	for _, c := range ok {
		if p := measured(c.mutate).Validate(); !p.OK() {
			t.Errorf("%s was rejected:\n%s", c.name, p.Error())
		}
	}
	bad := []struct {
		name, want string
		mutate     func(*Entry)
	}{
		{"kernel impl without a library", "library", func(e *Entry) {
			e.KernelImpls = []KernelImpl{{Variant: "v", CodeHash: "0123456789abcdef"}}
		}},
		{"kernel impl without a variant", "variant", func(e *Entry) {
			e.KernelImpls = []KernelImpl{{Library: "l", CodeHash: "0123456789abcdef"}}
		}},
		{"kernel impl with a short hash", "code_hash", func(e *Entry) {
			e.KernelImpls = []KernelImpl{{Library: "l", Variant: "v", CodeHash: "abc"}}
		}},
		{"kernel impl with an upper-case hash", "code_hash", func(e *Entry) {
			e.KernelImpls = []KernelImpl{{Library: "l", Variant: "v", CodeHash: "0123456789ABCDEF"}}
		}},
		{"duplicate kernel impl", "duplicate kernel implementation", func(e *Entry) { e.KernelImpls = []KernelImpl{impl, impl} }},
		{"kernel impl on an assumed entry", "states nothing about a kernel implementation", func(e *Entry) {
			e.Method, e.Fitted, e.Rationale = vocab.MethodAssumed, false, "reason"
			e.KernelImpls = []KernelImpl{impl}
		}},
		{"kernel impl on a roofline entry", "states nothing about a kernel implementation", func(e *Entry) {
			e.Method, e.Fitted, e.Rationale = vocab.MethodRoofline, false, "reason"
			e.Sources = []Source{{Kind: vocab.SourceModel, Role: vocab.RolePrimary, Cite: "catalog facts"}}
			e.KernelImpls = []KernelImpl{impl}
		}},
		{"kernel impl on a name that prices no lowered kernel", "binds to no primitive", func(e *Entry) {
			e.Name = "hbm_derate"
			e.KernelImpls = []KernelImpl{impl}
		}},
		{"measured_on on a measured entry", "states no other", func(e *Entry) {
			e.KernelImpls = []KernelImpl{impl}
			e.MeasuredOn = []KernelImpl{impl}
		}},
		{"transferred target with no source", "must name the ones it was measured on", func(e *Entry) {
			transferredTo(e, []KernelImpl{impl}, nil)
		}},
		{"duplicate measured_on", "duplicate kernel implementation", func(e *Entry) {
			transferredTo(e, []KernelImpl{impl}, []KernelImpl{other, other})
		}},
		{"unknown measurement time", "measures.time", func(e *Entry) {
			e.Measures = &Measures{Time: "wall", Execution: vocab.ExecutionEager, Harness: "h"}
		}},
		{"unknown measurement execution", "measures.execution", func(e *Entry) {
			e.Measures = &Measures{Time: vocab.TimeDevice, Execution: "replayed", Harness: "h"}
		}},
		{"measures without a harness", "measures.harness", func(e *Entry) {
			e.Measures = &Measures{Time: vocab.TimeDevice, Execution: vocab.ExecutionEager}
		}},
		{"measures on an assumed entry", "no measurement boundary", func(e *Entry) {
			e.Method, e.Fitted, e.Rationale = vocab.MethodAssumed, false, "reason"
			e.Measures = &Measures{Time: vocab.TimeDevice, Execution: vocab.ExecutionEager, Harness: "h"}
		}},
		{"fit range with low above high", "exceeds high", func(e *Entry) { e.FitRange = []Bound{{Dim: "tokens", Low: 9, High: 1}} }},
		{"fit range with a repeated dimension", "duplicate dimension", func(e *Entry) {
			e.FitRange = []Bound{{Dim: "tokens", Low: 1, High: 2}, {Dim: "tokens", Low: 1, High: 3}}
		}},
		{"fit range with no dimension", "dim", func(e *Entry) { e.FitRange = []Bound{{Low: 1, High: 2}} }},
		{"fit range on an assumed entry", "no fit range", func(e *Entry) {
			e.Method, e.Fitted, e.Rationale = vocab.MethodAssumed, false, "reason"
			e.FitRange = []Bound{{Dim: "tokens", Low: 1, High: 2}}
		}},
	}
	for _, c := range bad {
		p := measured(c.mutate).Validate()
		if p.OK() || !strings.Contains(p.Error(), c.want) {
			t.Errorf("%s: want a problem containing %q, got ok=%v:\n%s", c.name, c.want, p.OK(), p.Error())
		}
	}
}

// VAL-10 and VAL-3: an interval that is unbounded or wider than half the value marks the
// entry unidentified, which ranks with assumed; a lookup outside the fitted range is
// extrapolated, which ranks with transferred. The least trusted effect applies.
func TestUnidentifiedExtrapolatedAndEffectiveRank(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		name         string
		e            Entry
		unidentified bool
	}{
		{"no interval is not unidentified", Entry{Value: 10}, false},
		{"narrow interval", Entry{Value: 10, CI95: &Interval{Low: 9, High: 11}}, false},
		{"interval exactly half the value", Entry{Value: 10, CI95: &Interval{Low: 7.5, High: 12.5}}, false},
		{"interval wider than half the value", Entry{Value: 10, CI95: &Interval{Low: 7, High: 13}}, true},
		{"unbounded above", Entry{Value: 10, CI95: &Interval{Low: 9, High: inf}}, true},
		{"unbounded below", Entry{Value: 10, CI95: &Interval{Low: math.Inf(-1), High: 11}}, true},
	}
	for _, c := range cases {
		if got := c.e.Unidentified(); got != c.unidentified {
			t.Errorf("%s: Unidentified = %v, want %v", c.name, got, c.unidentified)
		}
	}
	e := Entry{Method: vocab.MethodMeasured, FitRange: []Bound{{Dim: "tokens", Low: 1, High: 100}, {Dim: "ranks", Low: 2, High: 8}}}
	for _, c := range []struct {
		name string
		in   map[string]float64
		want bool
	}{
		{"inside", map[string]float64{"tokens": 50, "ranks": 4}, false},
		{"on the bound", map[string]float64{"tokens": 100, "ranks": 8}, false},
		{"above on one dimension", map[string]float64{"tokens": 101, "ranks": 4}, true},
		{"below on the other", map[string]float64{"tokens": 50, "ranks": 1}, true},
		{"a recorded dimension not supplied is not judged", map[string]float64{"tokens": 50}, false},
	} {
		if got := e.Extrapolated(c.in); got != c.want {
			t.Errorf("%s: Extrapolated = %v, want %v", c.name, got, c.want)
		}
	}
	if (Entry{Method: vocab.MethodMeasured}).Extrapolated(map[string]float64{"tokens": 1e9}) {
		t.Error("an entry that recorded no range has not said, so it is never extrapolated")
	}
	rank := func(e Entry, extrapolated bool) int { r, _ := e.EffectiveRank(extrapolated); return r }
	wide := &Interval{Low: 0, High: 100}
	for _, c := range []struct {
		name string
		e    Entry
		x    bool
		want int
	}{
		{"measured, identified, in range", Entry{Method: vocab.MethodMeasured, Value: 10}, false, 1},
		{"measured and extrapolated ranks with transferred", Entry{Method: vocab.MethodMeasured, Value: 10}, true, 3},
		{"measured and unidentified ranks with assumed", Entry{Method: vocab.MethodMeasured, Value: 10, CI95: wide}, false, 7},
		{"calibrated and extrapolated", Entry{Method: vocab.MethodCalibrated, Value: 10}, true, 3},
		{"literature is already below transferred, so extrapolation does not move it", Entry{Method: vocab.MethodLiterature, Value: 10}, true, 4},
		{"assumed stays assumed", Entry{Method: vocab.MethodAssumed, Value: 10, CI95: wide}, true, 7},
	} {
		if got := rank(c.e, c.x); got != c.want {
			t.Errorf("%s: EffectiveRank = %d, want %d", c.name, got, c.want)
		}
	}
	if _, ranked := (Entry{Method: vocab.MethodNotCharged}).EffectiveRank(true); ranked {
		t.Error("not_charged has no rank")
	}
}

// transferredTo turns a measured entry into a transferred one that states the given target and
// source implementations.
func transferredTo(e *Entry, target, source []KernelImpl) {
	e.Method, e.Fitted = vocab.MethodTransferred, false
	e.Rationale = "reason"
	e.TransferredFrom, e.TransferRule = "hardware=h200", "scaled by the ratio of the chips' measured link bandwidths"
	e.KernelImpls, e.MeasuredOn = target, source
}
