package validate

import (
	"reflect"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

type mergeTarget struct {
	N int     `yaml:"n"`
	F float64 `yaml:"f"`
}

// The number rules reach fields that arrive through a merge key, whether the merge names
// one anchor (<<: *a) or a sequence of them (<<: [*a, *b]).
func TestNumberRulesReachMergedMembers(t *testing.T) {
	for name, c := range map[string]struct{ doc, want string }{
		"map form, fraction":      {"b: &b {n: 1.5}\nt:\n  <<: *b\n", "1.5"},
		"sequence form, fraction": {"b: &b {n: 1.5}\nt:\n  <<: [*b]\n", "1.5"},
		"sequence form, second":   {"a: &a {n: 1}\nb: &b {n: 1.5}\nt:\n  <<: [*a, *b]\n", "1.5"},
		"sequence form, inf":      {"b: &b {f: .inf}\nt:\n  <<: [*b]\n", ".inf"},
		"sequence form, nan":      {"b: &b {f: .nan}\nt:\n  <<: [*b]\n", ".nan"},
	} {
		t.Run(name, func(t *testing.T) {
			var doc struct {
				T mergeTarget `yaml:"t"`
			}
			var n yaml.Node
			if err := yaml.Unmarshal([]byte(c.doc), &n); err != nil {
				t.Fatal(err)
			}
			err := CheckNode(&n, reflect.TypeOf(&doc))
			if err == nil || !strings.Contains(err.Error(), c.want) {
				t.Fatalf("want a refusal naming %q, got %v", c.want, err)
			}
		})
	}
}

// A number rule is not skipped just because the sequence holds a clean member first, and a
// clean sequence merge is accepted.
func TestCleanSequenceMergeIsAccepted(t *testing.T) {
	var n yaml.Node
	doc := "a: &a {n: 1}\nb: &b {f: 2.5}\nt:\n  <<: [*a, *b]\n"
	if err := yaml.Unmarshal([]byte(doc), &n); err != nil {
		t.Fatal(err)
	}
	var d struct {
		T mergeTarget `yaml:"t"`
	}
	if err := CheckNode(&n, reflect.TypeOf(&d)); err != nil {
		t.Fatal(err)
	}
}
