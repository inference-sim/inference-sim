// Original to blis-sglang.

package blissglang

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/fields"
)

// valueClassesDoc is fields/value_classes.json: for each modelled or proven-inert field, the
// value_classes object the substrate's field list records (VLM-3, which SGL-3 applies) and the
// basis it was projected from, generated from the pinned SGLang source by fields/gen_classes.py.
// The Python test fields/test_classes.py reads source; this test needs none, so a field gaining a
// status without classes fails go test.
type valueClassesDoc struct {
	Kind   string                       `json:"kind"`
	Limits string                       `json:"limits"`
	Fields map[string]valueClassesEntry `json:"fields"`
}

type valueClassesEntry struct {
	ValueClasses fields.ValueClasses `json:"value_classes"`
	Basis        struct {
		Classes []struct {
			At    []string `json:"compared_at"`
			Uncut bool     `json:"uncut"`
			Other bool     `json:"other"`
		} `json:"classes"`
		Sites []struct {
			At string `json:"at"`
		} `json:"sites"`
	} `json:"basis"`
}

// classProblems applies the rules of fields/gen_classes.py check_document that need no source:
// every modelled or proven-inert field has classes, no other field has them, and each class is
// backed by a comparison the entry lists.
func classProblems(l *fields.FieldList, doc valueClassesDoc) []string {
	var out []string
	want := map[string]bool{}
	for _, f := range l.Fields {
		if f.Status == fields.StatusModelled || f.Status == fields.StatusProvenInert {
			want[f.Name] = true
			if _, ok := doc.Fields[f.Name]; !ok {
				out = append(out, fmt.Sprintf("%s: a %s field has no value classes", f.Name, f.Status))
			}
		}
	}
	for n, rec := range doc.Fields {
		if !want[n] {
			out = append(out, n+": has value classes but is neither modelled nor proven inert")
			continue
		}
		sites := map[string]bool{}
		for _, s := range rec.Basis.Sites {
			sites[s.At] = true
		}
		for i, c := range rec.Basis.Classes {
			if len(c.At) == 0 && !c.Uncut && !c.Other {
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

func TestEveryModelledFieldRecordsItsValueClasses(t *testing.T) {
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
	if doc.Kind != "ValueClasses" || doc.Limits == "" {
		t.Errorf("fields/value_classes.json: kind %q, limits %q; want ValueClasses with its limits stated", doc.Kind, doc.Limits)
	}
	for _, p := range classProblems(l, doc) {
		t.Error(p)
	}
}

// The check above passes while no field is modelled, so it is run here on documents that break
// each rule, to show it is not vacuous.
func TestTheClassCheckFailsOnEachKnownBadDocument(t *testing.T) {
	modelled := &fields.FieldList{Fields: []fields.Field{{Name: "page_size", Source: fields.SourceConfig, Status: fields.StatusModelled}}}
	notModelled := &fields.FieldList{Fields: []fields.Field{{Name: "page_size", Source: fields.SourceConfig, Status: fields.StatusNotModelled}}}
	var entry valueClassesEntry
	if err := json.Unmarshal([]byte(`{"value_classes":{"type":"numeric","covered_classes":[0]},
		"basis":{"classes":[{"compared_at":["a.py:1"]},{"compared_at":[]}],"sites":[{"at":"a.py:1"}]}}`), &entry); err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		l    *fields.FieldList
		doc  valueClassesDoc
		want string
	}{
		{"modelled without classes", modelled, valueClassesDoc{Fields: map[string]valueClassesEntry{}}, "has no value classes"},
		{"classes for a not-modelled field", notModelled, valueClassesDoc{Fields: map[string]valueClassesEntry{"page_size": entry}}, "neither modelled nor proven inert"},
		{"a class with no comparison", modelled, valueClassesDoc{Fields: map[string]valueClassesEntry{"page_size": entry}}, "class 1 is backed by no comparison"},
	}
	for _, c := range cases {
		probs := strings.Join(classProblems(c.l, c.doc), "\n")
		if !strings.Contains(probs, c.want) {
			t.Errorf("%s: want a problem containing %q, got %q", c.name, c.want, probs)
		}
	}
}
