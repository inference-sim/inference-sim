package workload

import (
	"testing"

	"gopkg.in/yaml.v3"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
)

func al(x float64) *float64 { return &x }

// The acceptance terms are optional on both a shape and a trace, and when stated they are
// checked where they are written.
func TestAcceptanceTerms(t *testing.T) {
	type terms struct {
		category string
		thinking acceptance.Thinking
		value    *acceptance.Value
	}
	cases := []struct {
		name string
		in   terms
		ok   bool
	}{
		{"none", terms{}, true},
		{"category and thinking", terms{"coding", acceptance.ThinkingOn, nil}, true},
		{"measured value", terms{"chat", acceptance.ThinkingOff,
			&acceptance.Value{NumSpecTokens: 3, AcceptanceLength: al(2.6)}}, true},
		{"capitalised category", terms{"Coding", "", nil}, false},
		{"unknown thinking", terms{"", "sometimes", nil}, false},
		{"value out of range", terms{"", "", &acceptance.Value{NumSpecTokens: 3, AcceptanceLength: al(4.5)}}, false},
	}
	for _, c := range cases {
		t.Run("shape/"+c.name, func(t *testing.T) {
			s := chatbot()
			s.Category, s.Thinking, s.Acceptance = c.in.category, c.in.thinking, c.in.value
			if p := s.Validate(); p.OK() != c.ok {
				t.Errorf("OK = %v, want %v:\n%s", p.OK(), c.ok, p.Error())
			}
			got := s.AcceptanceTerms()
			if got.Category != c.in.category || got.Thinking != c.in.thinking || got.Value != c.in.value {
				t.Errorf("AcceptanceTerms() = %+v", got)
			}
		})
		t.Run("trace/"+c.name, func(t *testing.T) {
			tr := traceRef()
			tr.Category, tr.Thinking, tr.Acceptance = c.in.category, c.in.thinking, c.in.value
			if p := tr.Validate(); p.OK() != c.ok {
				t.Errorf("OK = %v, want %v:\n%s", p.OK(), c.ok, p.Error())
			}
		})
	}
}

// A shape's custom unmarshaller must not swallow a typo inside its acceptance value.
func TestShapeAcceptanceDecodesStrictly(t *testing.T) {
	good := "prompt: {tokens: 10}\noutput: {tokens: 10}\ncategory: coding\nthinking: off\n" +
		"acceptance: {num_spec_tokens: 2, acceptance_length: 2.2}\n"
	var s Shape
	if err := yaml.Unmarshal([]byte(good), &s); err != nil {
		t.Fatal(err)
	}
	if s.Thinking != acceptance.ThinkingOff || s.Acceptance == nil || *s.Acceptance.AcceptanceLength != 2.2 {
		t.Errorf("decoded %+v", s)
	}
	bad := "prompt: {tokens: 10}\noutput: {tokens: 10}\nacceptance: {num_spec_tokens: 2, al: 2.2}\n"
	if err := yaml.Unmarshal([]byte(bad), &s); err == nil {
		t.Error("a misspelled acceptance key decoded silently")
	}
}
