package coefficient

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/vocab"
)

func entry(name string, hw []string, v float64) Entry {
	return Entry{
		Name: name, Value: v, Units: vocab.UnitDimensionless,
		Method: vocab.MethodMeasured, Scope: Scope{Hardware: hw},
		Sources: []Source{{Kind: vocab.SourceModel, Cite: "x", Role: vocab.RolePrimary}},
	}
}

func set(es ...Entry) Set {
	return Set{Kind: "CoefficientSet", Name: "t", Coefficients: es}
}

func hasDuplicate(s Set) bool {
	for _, p := range s.Validate().All() {
		if strings.Contains(p.Message, "duplicate") {
			return true
		}
	}
	return false
}

func TestOneNameAtTwoScopesIsNotADuplicate(t *testing.T) {
	s := set(entry("gemm_eps_max_bf16", []string{"h100"}, 0.903),
		entry("gemm_eps_max_bf16", []string{"h200"}, 0.890))
	if hasDuplicate(s) {
		t.Error("two scopes under one name were rejected; a resolver picks by scope")
	}
}

func TestOneNameAtOneScopeIsADuplicate(t *testing.T) {
	s := set(entry("gemm_eps_max_bf16", []string{"h200"}, 0.890),
		entry("gemm_eps_max_bf16", []string{"h200"}, 0.72))
	if !hasDuplicate(s) {
		t.Error("two entries a resolver cannot choose between were accepted")
	}
}

func TestScopeIdentityIgnoresListOrder(t *testing.T) {
	s := set(entry("hbm_derate", []string{"h100", "h200"}, 0.8),
		entry("hbm_derate", []string{"h200", "h100"}, 0.85))
	if !hasDuplicate(s) {
		t.Error("the same scope written in a different order was treated as distinct")
	}
}

func TestTwoUnscopedEntriesUnderOneNameAreADuplicate(t *testing.T) {
	s := set(entry("hbm_derate", nil, 0.8), entry("hbm_derate", nil, 0.85))
	if !hasDuplicate(s) {
		t.Error("two unscoped entries under one name were accepted")
	}
}

// A fabric scope is its own dimension: the same name on hardware is a different scope, and
// two fabrics are two scopes, so one entry name can carry a RoCE and an InfiniBand value.
func TestFabricIsAScopeDimension(t *testing.T) {
	if (Scope{Fabric: []string{"x"}}).Key() == (Scope{Hardware: []string{"x"}}).Key() {
		t.Error("a fabric scope and a hardware scope naming the same string collide")
	}
	if (Scope{Fabric: []string{"roce-200g", "ib-400g"}}).Key() !=
		(Scope{Fabric: []string{"ib-400g", "roce-200g"}}).Key() {
		t.Error("fabric order changed the scope identity")
	}
	s := set(
		Entry{Name: "fabric_bandwidth_efficiency", Value: 0.8, Units: vocab.UnitDimensionless,
			Method: vocab.MethodMeasured, Scope: Scope{Fabric: []string{"roce-200g"}}},
		Entry{Name: "fabric_bandwidth_efficiency", Value: 0.9, Units: vocab.UnitDimensionless,
			Method: vocab.MethodMeasured, Scope: Scope{Fabric: []string{"ib-400g"}}})
	if hasDuplicate(s) {
		t.Error("one name scoped to two fabrics was rejected as a duplicate")
	}
	if (Scope{Fabric: []string{"ib-400g"}}).Empty() {
		t.Error("a scope stating only a fabric reads as empty")
	}
}

// Each malformed fabric name is its own failure: blank, a path, a file name, a repeat.
func TestFabricScopeRejectsWhatNamesNoFabric(t *testing.T) {
	for name, fabrics := range map[string][]string{
		"blank":          {" "},
		"a path":         {"networks/roce-200g"},
		"a file name":    {"roce-200g.yaml"},
		"a repeated one": {"ib-400g", "IB-400G"},
	} {
		t.Run(name, func(t *testing.T) {
			s := set(Entry{Name: "x", Value: 1, Units: vocab.UnitDimensionless,
				Method: vocab.MethodMeasured, Scope: Scope{Fabric: fabrics}})
			found := false
			for _, p := range s.Validate().All() {
				found = found || strings.Contains(p.Path, "scope.fabric")
			}
			if !found {
				t.Errorf("fabric %q was accepted", fabrics)
			}
		})
	}
}

func TestScopeKeyDistinguishesDimensions(t *testing.T) {
	// A hardware scope and a model scope naming the same string are different scopes.
	a := Scope{Hardware: []string{"x"}}
	b := Scope{Model: []string{"x"}}
	if a.Key() == b.Key() {
		t.Errorf("scopes on different dimensions collide: %q", a.Key())
	}
}

// An entry's identity is its name, scope and method (design §3.3): the same name and scope
// under another method is a different coefficient, and only all three matching is refused.
func TestOneNameAndScopeAtTwoMethodsIsNotADuplicate(t *testing.T) {
	assumed := entry("fabric_bandwidth_efficiency", []string{"h200"}, 0.8)
	assumed.Method = vocab.MethodAssumed
	assumed.Rationale = "a judgement"
	assumed.Sources = nil
	measured := entry("fabric_bandwidth_efficiency", []string{"h200"}, 0.9)
	if hasDuplicate(set(assumed, measured)) {
		t.Error("one name and scope under two methods was refused as a duplicate")
	}
	again := measured
	again.Value = 0.7
	if !hasDuplicate(set(assumed, measured, again)) {
		t.Error("one name, scope and method twice was not refused")
	}
}
