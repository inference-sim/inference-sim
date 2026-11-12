package validate

import (
	"math"
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

// doc has one field that may hold +infinity, as grace_period_s does, and fields that may not.
type infDoc struct {
	Grace  *float64           `yaml:"grace,omitempty" infinity:"allowed"`
	Plain  float64            `yaml:"plain,omitempty"`
	PlainP *float64           `yaml:"plain_p,omitempty"`
	List   []float64          `yaml:"list,omitempty"`
	ByName map[string]float64 `yaml:"by_name,omitempty"`
	Limits []float64          `yaml:"limits,omitempty" infinity:"allowed"`
	Inner  struct {
		F float64 `yaml:"f,omitempty"`
	} `yaml:"inner,omitempty"`
}

// SUB-12, issue 28: a field that opts in holds +infinity, spelled .inf, +.inf or .Inf; no field
// holds -infinity or NaN; a field that does not opt in holds none of them.
func TestAFieldThatOptsInHoldsPlusInfinity(t *testing.T) {
	for _, in := range []string{"grace: .inf\n", "grace: +.inf\n", "grace: .Inf\n", "grace: .INF\n"} {
		var d infDoc
		if err := DecodeStrict("d.yaml", []byte(in), &d); err != nil || d.Grace == nil || !math.IsInf(*d.Grace, 1) {
			t.Errorf("%q: err=%v value=%v", in, err, d.Grace)
		}
	}
	var d infDoc
	if err := DecodeStrict("d.yaml", []byte("grace: 30\n"), &d); err != nil || *d.Grace != 30 {
		t.Errorf("a finite value: %v %v", err, d.Grace)
	}
}

func TestNoFieldHoldsNaNOrMinusInfinity(t *testing.T) {
	for _, field := range []string{"grace", "plain", "plain_p", "limits"} {
		for _, v := range []string{".nan", ".NaN", "-.inf", "-.Inf"} {
			in := field + ": " + v + "\n"
			if field == "limits" {
				in = field + ": [1, " + v + "]\n"
			}
			var d infDoc
			err := DecodeStrict("d.yaml", []byte(in), &d)
			if err == nil {
				t.Errorf("%q was accepted into %s (decoded %+v)", in, field, d)
			} else if !strings.Contains(err.Error(), "no field holds it") {
				t.Errorf("%q: the refusal does not say no field holds it: %v", in, err)
			}
		}
	}
}

func TestAFieldThatDoesNotOptInRefusesInfinity(t *testing.T) {
	for _, in := range []string{
		"plain: .inf\n", "plain_p: .inf\n", "plain_p: +.inf\n",
		"list: [1, .inf]\n", "by_name: {a: .inf}\n", "inner: {f: .inf}\n",
	} {
		var d infDoc
		err := DecodeStrict("d.yaml", []byte(in), &d)
		if err == nil || !strings.Contains(err.Error(), "does not allow infinity") {
			t.Errorf("%q: %v", in, err)
		}
	}
}

// An opt-in on a slice applies to its elements and nothing else is opened by it.
func TestAnOptInOnASliceCoversItsElements(t *testing.T) {
	var d infDoc
	if err := DecodeStrict("d.yaml", []byte("limits: [1, .inf, 3]\nlist: [1, 2]\n"), &d); err != nil || !math.IsInf(d.Limits[1], 1) {
		t.Fatalf("%v %v", err, d.Limits)
	}
	if err := DecodeStrict("d.yaml", []byte("limits: [1]\nlist: [1, .inf]\n"), &d); err == nil {
		t.Error("an opt-in on one field opened another")
	}
}

// Infinity written as a quoted string is a string, not infinity, and stays a type error.
func TestTheStringInfIsNotInfinityInYAML(t *testing.T) {
	var d infDoc
	if err := DecodeStrict("d.yaml", []byte("grace: \"inf\"\n"), &d); err == nil {
		t.Errorf("the string \"inf\" loaded into a float field: %+v", d)
	}
}

// With no target type, only the rule that holds for every number applies: infinity is refused.
func TestWithNoTargetTypeInfinityIsRefused(t *testing.T) {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("a: .inf\n"), &n); err != nil {
		t.Fatal(err)
	}
	if err := CheckNode(&n, nil); err == nil {
		t.Error("infinity was accepted with no field to opt in")
	}
}

// InfinityNodes finds exactly the scalars under opted-in fields.
func TestInfinityNodesFindsTheOptedInScalars(t *testing.T) {
	var n yaml.Node
	if err := yaml.Unmarshal([]byte("grace: .inf\nplain: 1\nlimits: [1, .inf]\nlist: [2]\n"), &n); err != nil {
		t.Fatal(err)
	}
	got := InfinityNodes(&n, reflect.TypeOf(&infDoc{}))
	if len(got) != 3 {
		t.Fatalf("found %d scalars, want 3 (grace, and the two limits)", len(got))
	}
}

// The inventory of fields that opt in is the list a test names: a new opt-in changes it.
func TestInfinityFieldsListsTheOptIns(t *testing.T) {
	got := InfinityFields(reflect.TypeOf(infDoc{}))
	if len(got) != 2 || got[0] != "infDoc.Grace" || got[1] != "infDoc.Limits" {
		t.Errorf("InfinityFields = %v", got)
	}
}
