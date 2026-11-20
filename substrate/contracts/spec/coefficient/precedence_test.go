package coefficient

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func cand(set, name string, m vocab.Method, s Scope) Candidate {
	return Candidate{Set: set, Entry: Entry{Name: name, Method: m, Scope: s}}
}

// VAL-4: the smaller method rank wins; within a rank the scope precedence decides; entries
// the precedence does not order are refused. Each case is run in both load orders.
func TestSelect(t *testing.T) {
	h100 := Scope{Hardware: []string{"h100"}}
	h100m := Scope{Hardware: []string{"h100"}, Model: []string{"m"}}
	h100tp8 := Scope{Hardware: []string{"h100"}, TP: []int{8}}
	either := Scope{Hardware: []string{"h100", "h200"}}
	cases := []struct {
		name     string
		a, b     Candidate
		wantSet  string // "" means refused
		wantDisp int
	}{
		{"rank beats scope: measured over a narrower assumed",
			cand("a", "x", vocab.MethodMeasured, h100), cand("b", "x", vocab.MethodAssumed, h100m), "a", 1},
		{"rank beats scope: a wider measured over a narrower transferred",
			cand("a", "x", vocab.MethodTransferred, h100tp8), cand("b", "x", vocab.MethodCalibrated, either), "b", 1},
		{"one rank, model-scoped over not",
			cand("a", "x", vocab.MethodMeasured, h100), cand("b", "x", vocab.MethodMeasured, h100m), "b", 1},
		{"one rank, a stated width over an open one",
			cand("a", "x", vocab.MethodMeasured, h100), cand("b", "x", vocab.MethodMeasured, h100tp8), "b", 1},
		{"one rank, one chip inside a list of two: not ordered, a list that overlaps is no narrower",
			cand("a", "x", vocab.MethodMeasured, either), cand("b", "x", vocab.MethodMeasured, h100), "", 0},
		{"one rank, the same chips in another order do not differ",
			cand("a", "x", vocab.MethodMeasured, Scope{Hardware: []string{"h100", "h200"}}),
			cand("b", "x", vocab.MethodMeasured, Scope{Hardware: []string{"H200", "h100"}}), "", 0},
		{"not_charged has no rank, so scope decides",
			cand("a", "x", vocab.MethodNotCharged, h100m), cand("b", "x", vocab.MethodAssumed, h100), "a", 1},
		{"same rank, equal scope: refused",
			cand("a", "x", vocab.MethodMeasured, h100), cand("b", "x", vocab.MethodMeasured, h100), "", 0},
		{"same rank, each states what the other leaves open: the model decides first",
			cand("a", "x", vocab.MethodMeasured, h100m), cand("b", "x", vocab.MethodMeasured, h100tp8), "a", 1},
		{"same rank, lists that overlap without nesting: refused",
			cand("a", "x", vocab.MethodMeasured, Scope{Hardware: []string{"a", "b"}}),
			cand("b", "x", vocab.MethodMeasured, Scope{Hardware: []string{"b", "c"}}), "", 0},
	}
	for _, c := range cases {
		for _, order := range [][]Candidate{{c.a, c.b}, {c.b, c.a}} {
			w, disp, err := Select(order)
			if c.wantSet == "" {
				if err == nil || !strings.Contains(err.Error(), "VAL-4") {
					t.Errorf("%s: want a VAL-4 refusal, got winner %v, err %v", c.name, w.Describe(), err)
				}
				continue
			}
			if err != nil || w.Set != c.wantSet || len(disp) != c.wantDisp {
				t.Errorf("%s (order %s,%s): winner %q displaced %d err %v; want %q displaced %d",
					c.name, order[0].Set, order[1].Set, w.Set, len(disp), err, c.wantSet, c.wantDisp)
			}
		}
	}
}

// The model decides before the chip: a model-scoped entry for two chips is chosen over a
// chip-only entry for one, which is the order the cost model has always had.
func TestModelDecidesBeforeHardware(t *testing.T) {
	a := Scope{Hardware: []string{"h100"}}
	b := Scope{Hardware: []string{"h100", "h200"}, Model: []string{"m"}}
	if cmp, ok := ComparePrecedence(b, a); !ok || cmp >= 0 {
		t.Errorf("the model-scoped entry must be narrower, got %d, %v", cmp, ok)
	}
}

func TestAdmits(t *testing.T) {
	p := Point{Hardware: "H100", Model: "m", TP: 8, EP: 1, NodesSpanned: 1}
	for _, c := range []struct {
		name string
		s    Scope
		want bool
	}{
		{"empty", Scope{}, true},
		{"chip, any case", Scope{Hardware: []string{"h100"}}, true},
		{"other chip", Scope{Hardware: []string{"h200"}}, false},
		{"width listed", Scope{TP: []int{4, 8}}, true},
		{"width not listed", Scope{TP: []int{4}}, false},
		{"a fabric-scoped entry on a single node", Scope{Fabric: []string{"f"}}, false},
	} {
		if got := c.s.Admits(p); got != c.want {
			t.Errorf("%s: Admits = %v, want %v", c.name, got, c.want)
		}
	}
	if !(Scope{Fabric: []string{"F"}}).Admits(Point{Fabric: "f"}) {
		t.Error("fabric names match without regard to case")
	}
}
