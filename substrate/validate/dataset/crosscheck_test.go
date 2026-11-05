package dataset

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func entry(name string, s coefficient.Scope, m vocab.Method) coefficient.Entry {
	return coefficient.Entry{Name: name, Value: 1, Method: m, Scope: s}
}

// VAL-4 at load: a model-scoped entry is ordered before one that states no model, and two
// entries whose scopes cannot both match are not a pair. Any other matching pair is
// refused, whichever sets hold them.
func TestCheckOrdered(t *testing.T) {
	h100 := coefficient.Scope{Hardware: []string{"h100"}}
	for _, c := range []struct {
		name string
		a, b coefficient.Scope
		bad  bool
	}{
		{"same scope in two sets", h100, h100, true},
		{"one scope inside another", h100, coefficient.Scope{Hardware: []string{"h100"}, TP: []int{8}}, true},
		{"disjoint chips", h100, coefficient.Scope{Hardware: []string{"h200"}}, false},
		{"disjoint widths", coefficient.Scope{Hardware: []string{"h100"}, TP: []int{4}}, coefficient.Scope{Hardware: []string{"h100"}, TP: []int{8}}, false},
		{"model-scoped over not", h100, coefficient.Scope{Hardware: []string{"h100"}, Model: []string{"m"}}, false},
		{"case-insensitive chip", h100, coefficient.Scope{Hardware: []string{"H100"}}, true},
	} {
		sets := map[string]*coefficient.Set{
			"a": {Name: "a", Coefficients: []coefficient.Entry{entry("x", c.a, vocab.MethodMeasured)}},
			"b": {Name: "b", Coefficients: []coefficient.Entry{entry("x", c.b, vocab.MethodAssumed)}},
		}
		got := CheckOrdered(sets)
		if (len(got) > 0) != c.bad {
			t.Errorf("%s: problems %v, want refused=%v", c.name, got, c.bad)
		}
	}
}

// A registry must fit the catalog it runs with: every chip, model and fabric a coefficient
// names is in the catalog.
func TestCheckFits(t *testing.T) {
	c := &Catalog{Chips: map[string]*hardware.Chip{"h100": {}}, Models: map[string]*model.Graph{"m": {}},
		Fabrics: map[string]*hardware.Fabric{}}
	r := &Registry{Sets: map[string]*coefficient.Set{"s": {Coefficients: []coefficient.Entry{
		entry("ok", coefficient.Scope{Hardware: []string{"h100"}, Model: []string{"m"}}, vocab.MethodAssumed),
		entry("chip", coefficient.Scope{Hardware: []string{"h900"}}, vocab.MethodAssumed),
		entry("model", coefficient.Scope{Model: []string{"nope"}}, vocab.MethodAssumed),
		entry("fabric", coefficient.Scope{Fabric: []string{"ib-400g"}}, vocab.MethodAssumed),
	}}}}
	got := CheckFits(c, r)
	if len(got) != 3 {
		t.Fatalf("want three misses, got %v", got)
	}
	for i, want := range []string{`chip "h900"`, `fabric "ib-400g"`, `model "nope"`} {
		if !strings.Contains(got[i].Message, want) || got[i].File != "coefficients/s.yaml" {
			t.Errorf("problem %d: %v, want one naming %s", i, got[i], want)
		}
	}
}
