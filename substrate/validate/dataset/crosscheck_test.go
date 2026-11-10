package dataset

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/acceptance"
	"github.com/inference-sim/blis-substrate/contracts/spec/coefficient"
	"github.com/inference-sim/blis-substrate/contracts/spec/hardware"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func entry(name string, s coefficient.Scope, m vocab.Method) coefficient.Entry {
	return coefficient.Entry{Name: name, Value: 1, Method: m, Scope: s}
}

// VAL-4 at load: entries are ordered by method rank first and then by the scope precedence,
// and two entries whose scopes cannot both match are not a pair. A matching pair that neither
// orders is refused, whichever sets hold them.
func TestCheckOrdered(t *testing.T) {
	h100 := coefficient.Scope{Hardware: []string{"h100"}}
	for _, c := range []struct {
		name string
		a, b coefficient.Scope
		bad  bool
	}{
		{"same scope and rank in two sets", h100, h100, true},
		{"one scope inside another: the stated width decides", h100, coefficient.Scope{Hardware: []string{"h100"}, TP: []int{8}}, false},
		{"disjoint chips", h100, coefficient.Scope{Hardware: []string{"h200"}}, false},
		{"disjoint widths", coefficient.Scope{Hardware: []string{"h100"}, TP: []int{4}}, coefficient.Scope{Hardware: []string{"h100"}, TP: []int{8}}, false},
		{"model-scoped over not", h100, coefficient.Scope{Hardware: []string{"h100"}, Model: []string{"m"}}, false},
		{"case-insensitive chip, same rank", h100, coefficient.Scope{Hardware: []string{"H100"}}, true},
		{"overlapping lists that do not nest", coefficient.Scope{Hardware: []string{"h100", "h200"}},
			coefficient.Scope{Hardware: []string{"h200", "b200"}}, true},
		{"a list and one of its members", coefficient.Scope{Hardware: []string{"h100", "h200"}},
			coefficient.Scope{Hardware: []string{"h200"}}, true},
	} {
		sets := map[string]*coefficient.Set{
			"a": {Name: "a", Coefficients: []coefficient.Entry{entry("x", c.a, vocab.MethodMeasured)}},
			"b": {Name: "b", Coefficients: []coefficient.Entry{entry("x", c.b, vocab.MethodMeasured)}},
		}
		got := CheckOrdered(sets)
		if (len(got) > 0) != c.bad {
			t.Errorf("%s: problems %v, want refused=%v", c.name, got, c.bad)
		}
	}
}

// Two entries of one name at one scope under different methods are ordered by rank, in
// whichever sets they sit (VAL-4, design §9.2).
func TestCheckOrderedAcceptsDifferentRanksAtOneScope(t *testing.T) {
	h100 := coefficient.Scope{Hardware: []string{"h100"}}
	sets := map[string]*coefficient.Set{
		"a": {Name: "a", Coefficients: []coefficient.Entry{entry("x", h100, vocab.MethodMeasured)}},
		"b": {Name: "b", Coefficients: []coefficient.Entry{entry("x", h100, vocab.MethodAssumed)}},
	}
	if got := CheckOrdered(sets); len(got) != 0 {
		t.Errorf("a measured and an assumed entry at one scope are ordered by rank, got %v", got)
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

// A profile's drafter must be a catalog model, and the architecture it states must be the one
// its target's graph records, or a lookup that borrows across targets would borrow from a model
// of another architecture (design D8).
func TestCheckFitsReadsAProfilesDrafterAndArchitecture(t *testing.T) {
	graph := func(arch string) *model.Graph {
		g := &model.Graph{}
		g.Global.Architecture = arch
		return g
	}
	c := &Catalog{Chips: map[string]*hardware.Chip{}, Fabrics: map[string]*hardware.Fabric{},
		Models: map[string]*model.Graph{"target": graph("DeepseekV3ForCausalLM"), "drafter": graph("X")}}
	p := func(mut func(*acceptance.Profile)) *Registry {
		pr := acceptance.Profile{Model: "target", SpeculativeMethod: "mtp", NumSpecTokens: 3,
			Thinking: acceptance.ThinkingOff, Category: "coding", AcceptanceLength: 2,
			Architecture: "DeepseekV3ForCausalLM", Method: "measured"}
		mut(&pr)
		return &Registry{Acceptance: map[string]*acceptance.ProfileSet{"s": {Profiles: []acceptance.Profile{pr}}}}
	}
	if got := CheckFits(c, p(func(*acceptance.Profile) {})); len(got) != 0 {
		t.Errorf("a consistent profile was reported: %v", got)
	}
	if got := CheckFits(c, p(func(pr *acceptance.Profile) { pr.Drafter = "drafter" })); len(got) != 0 {
		t.Errorf("a profile naming a catalog drafter was reported: %v", got)
	}
	for name, c2 := range map[string]struct {
		mut  func(*acceptance.Profile)
		want string
	}{
		"a drafter that is no catalog model": {func(pr *acceptance.Profile) { pr.Drafter = "ghost" }, `drafter "ghost"`},
		"an architecture the graph does not record": {func(pr *acceptance.Profile) { pr.Architecture = "LlamaForCausalLM" },
			`states architecture "LlamaForCausalLM", but the catalog's graph for target records "DeepseekV3ForCausalLM"`},
	} {
		got := CheckFits(c, p(c2.mut))
		if len(got) != 1 || !strings.Contains(got[0].Message, c2.want) {
			t.Errorf("%s: %v, want one problem containing %q", name, got, c2.want)
		}
	}
	// A profile that states no architecture is never borrowed across targets, so nothing is
	// asserted about it.
	if got := CheckFits(c, p(func(pr *acceptance.Profile) { pr.Architecture = "" })); len(got) != 0 {
		t.Errorf("a profile with no architecture was reported: %v", got)
	}
}
