// Original to blis-vllm.

package blisvllm

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// valueClassesDoc is fields/value_classes.json: for each modelled or proven-inert field, the
// value_classes object the substrate's field list records (design VLM-3) and the
// basis it was projected from, generated from the pinned vLLM source by fields/gen_classes.py.
// The Python test fields/test_classes.py reads the source and checks each cited line; this test
// needs no source, so a field gaining a status without classes fails go test.
type valueClassesDoc struct {
	Kind   string                       `json:"kind"`
	Limits string                       `json:"limits"`
	Fields map[string]valueClassesEntry `json:"fields"`
}

type valueClassesEntry struct {
	ValueClasses substrateClasses `json:"value_classes"`
	Basis        classesBasis     `json:"basis"`
}

// substrateClasses is the object blis-substrate's contracts/spec/fields.ValueClasses defines.
type substrateClasses struct {
	Type           string      `json:"type"`
	Values         []string    `json:"values"`
	Covered        []string    `json:"covered"`
	Thresholds     []threshold `json:"thresholds"`
	CoveredClasses []int       `json:"covered_classes"`
}

type threshold struct {
	Value  float64 `json:"value"`
	Closed string  `json:"closed"`
}

type classesBasis struct {
	Kind    string       `json:"kind"`
	Classes []classValue `json:"classes"`
	Sites   []classSite  `json:"sites"`
	Stated  []struct {
		Trace string `json:"trace"`
	} `json:"stated"`
}

type classValue struct {
	Lo         *float64 `json:"lo"`
	Hi         *float64 `json:"hi"`
	LoClosed   bool     `json:"lo_closed"`
	HiClosed   bool     `json:"hi_closed"`
	Other      bool     `json:"other"`
	Uncompared bool     `json:"uncompared"`
	Uncut      bool     `json:"uncut"`
	At         []string `json:"compared_at"`
}

type classSite struct {
	At string `json:"at"`
}

// classProblems applies the rules of fields/gen_classes.py check_document that need no source, and
// the rules of the substrate's value-class validator (which this repository's pin does not have yet).
func classProblems(l *fields.FieldList, doc valueClassesDoc) []string {
	var out []string
	want := map[string]fields.Status{}
	for _, f := range l.Fields {
		if f.Status == fields.StatusModelled || f.Status == fields.StatusProvenInert {
			want[f.Name] = f.Status
			if _, ok := doc.Fields[f.Name]; !ok {
				out = append(out, fmt.Sprintf("%s: a %s field has no value classes", f.Name, f.Status))
			}
		}
	}
	for n, rec := range doc.Fields {
		if _, ok := want[n]; !ok {
			out = append(out, n+": has value classes but is neither modelled nor proven inert")
			continue
		}
		vc, e := rec.ValueClasses, rec.Basis
		if len(e.Classes) == 0 {
			out = append(out, n+": no classes")
			continue
		}
		switch vc.Type {
		case "boolean", "enumerated":
			if len(vc.Covered) == 0 {
				out = append(out, n+": the evidence states no value (covered is empty)")
			}
		case "numeric":
			if len(vc.CoveredClasses) == 0 {
				out = append(out, n+": the evidence states no value in any class (covered_classes is empty)")
			}
			for i, k := range vc.CoveredClasses {
				if k < 0 || k > len(vc.Thresholds) || i > 0 && k <= vc.CoveredClasses[i-1] {
					out = append(out, fmt.Sprintf("%s: covered_classes[%d] = %d is outside the %d classes or out of order", n, i, k, len(vc.Thresholds)+1))
				}
			}
			for i, t := range vc.Thresholds {
				if t.Closed != "above" && t.Closed != "below" {
					out = append(out, fmt.Sprintf("%s: threshold %d has side %q", n, i, t.Closed))
				}
				if i > 0 && t.Value <= vc.Thresholds[i-1].Value {
					out = append(out, fmt.Sprintf("%s: thresholds are not strictly increasing at %d", n, i))
				}
			}
			if len(vc.Thresholds)+1 != numericClasses(e.Classes) {
				out = append(out, fmt.Sprintf("%s: %d thresholds for %d basis classes", n, len(vc.Thresholds), numericClasses(e.Classes)))
			}
		default:
			out = append(out, fmt.Sprintf("%s: value_classes.type %q", n, vc.Type))
		}
		if len(e.Stated) == 0 {
			out = append(out, n+": the basis names no trace that states a value")
		}
		sites := map[string]bool{}
		for _, s := range e.Sites {
			sites[s.At] = true
		}
		for i, c := range e.Classes {
			if len(c.At) == 0 && !c.Other && !c.Uncompared && !c.Uncut {
				out = append(out, fmt.Sprintf("%s: class %d is backed by no comparison", n, i))
			}
			for _, a := range c.At {
				if !sites[a] {
					out = append(out, fmt.Sprintf("%s: class %d cites %s, which is not a comparison of the entry", n, i, a))
				}
			}
		}
	}
	return out
}

func numericClasses(cl []classValue) int {
	n := 0
	for _, c := range cl {
		if c.Lo != nil || c.Hi != nil || c.Uncut {
			n++
		}
	}
	return n
}

func loadClasses(t *testing.T) (*fields.FieldList, valueClassesDoc) {
	t.Helper()
	l, err := fields.Load("fields.json")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("fields/value_classes.json")
	if err != nil {
		t.Fatal(err)
	}
	var doc valueClassesDoc
	if err := json.Unmarshal(raw, &doc); err != nil {
		t.Fatal(err)
	}
	return l, doc
}

// A modelled or proven-inert field records its value classes, and each class is backed by a
// comparison the generator found in the pinned source (fields/test_classes.py reads that source).
func TestEveryModelledFieldRecordsItsValueClasses(t *testing.T) {
	l, doc := loadClasses(t)
	if doc.Kind != "FieldValueClasses" {
		t.Fatalf("fields/value_classes.json kind = %q", doc.Kind)
	}
	if !strings.Contains(doc.Limits, "not followed") {
		t.Error("fields/value_classes.json does not state what the generator does not follow")
	}
	for _, p := range classProblems(l, doc) {
		t.Error(p)
	}
	// fields.json carries the classes themselves (the loader refuses a modelled or proven-inert
	// field without them); fields/value_classes.json is the basis they were projected from, so
	// the two must agree field by field.
	for _, f := range l.Fields {
		if f.Status != fields.StatusModelled && f.Status != fields.StatusProvenInert {
			continue
		}
		rec, ok := doc.Fields[f.Name]
		if !ok || f.ValueClasses == nil {
			continue // classProblems and the loader report these
		}
		// Through the one struct, so an absent list and an empty one are the same value.
		raw, err := json.Marshal(f.ValueClasses)
		if err != nil {
			t.Fatal(err)
		}
		var got substrateClasses
		if err := json.Unmarshal(raw, &got); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, rec.ValueClasses) {
			t.Errorf("%s: value_classes in fields.json differ from fields/value_classes.json\n fields.json: %+v\n basis file:  %+v", f.Name, got, rec.ValueClasses)
		}
	}
}

func TestTheClassCheckFailsOnEachKnownBadDocument(t *testing.T) {
	l, doc := loadClasses(t)
	var name string
	for n, rec := range doc.Fields {
		if rec.ValueClasses.Type == "numeric" && len(rec.ValueClasses.Thresholds) > 1 && len(rec.Basis.Classes[1].At) > 0 {
			name = n
			break
		}
	}
	if name == "" {
		t.Fatal("no numeric field with two thresholds to mutate")
	}
	clone := func() valueClassesDoc {
		raw, err := json.Marshal(doc)
		if err != nil {
			t.Fatal(err)
		}
		var d valueClassesDoc
		if err := json.Unmarshal(raw, &d); err != nil {
			t.Fatal(err)
		}
		return d
	}
	cases := map[string]func(d valueClassesDoc){
		"a field without classes": func(d valueClassesDoc) { delete(d.Fields, name) },
		"a field with an empty class list": func(d valueClassesDoc) {
			r := d.Fields[name]
			r.Basis.Classes = nil
			d.Fields[name] = r
		},
		"a class no comparison backs": func(d valueClassesDoc) { d.Fields[name].Basis.Classes[1].At = nil },
		"a class citing a place that is no comparison": func(d valueClassesDoc) {
			d.Fields[name].Basis.Classes[1].At = []string{"vllm/nowhere.py:1"}
		},
		"thresholds out of order": func(d valueClassesDoc) {
			th := d.Fields[name].ValueClasses.Thresholds
			th[1].Value = th[0].Value
		},
		"a bad side": func(d valueClassesDoc) { d.Fields[name].ValueClasses.Thresholds[0].Closed = "left" },
		"a covered class that does not exist": func(d valueClassesDoc) {
			r := d.Fields[name]
			r.ValueClasses.CoveredClasses = []int{len(r.ValueClasses.Thresholds) + 1}
			d.Fields[name] = r
		},
		"no value stated by the evidence": func(d valueClassesDoc) {
			r := d.Fields[name]
			r.ValueClasses.CoveredClasses = nil
			d.Fields[name] = r
		},
		"classes for a field that is not modelled": func(d valueClassesDoc) { d.Fields["VLLM_NOT_A_MODELLED_FIELD"] = d.Fields[name] },
	}
	for what, mutate := range cases {
		d := clone()
		mutate(d)
		if got := classProblems(l, d); len(got) == 0 {
			t.Errorf("%s: the check found no problem", what)
		}
	}
}
