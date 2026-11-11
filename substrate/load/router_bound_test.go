package load

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
	"github.com/inference-sim/blis-substrate/contracts/spec/routertable"
)

// A router_differs carries the entry's differences and bound as they stand, so a reader of the
// point sees what the mapping rests on and how far the policy moves each figure. A record that
// states fewer differences, a changed ratio or no bound is refused, whatever else it says.
func TestARouterSubstitutionCarriesTheEntrysDifferencesAndBound(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const covered = "1682167d3c97f0fc8cd405dee51f17288eeaeb55"
	e, out := d.LookupRouter("dynamo", "round-robin", nil, covered)
	if out != routertable.OutcomeMapped {
		t.Fatalf("dynamo round-robin at %s: %s", covered, out)
	}
	// carried builds a point's substitution from the entry, with its own copy of the bound.
	carried := func() pointrecord.Substitution {
		b := *e.Bound
		b.Ratios = append([]routertable.Ratio(nil), b.Ratios...)
		return pointrecord.Substitution{Kind: pointrecord.SubRouterDiffers, Subject: "dynamo", Entry: e.Name,
			Policy: string(e.Policy), Revision: covered, Evidence: "catalog/routers/" + e.Name + ".yaml",
			Difference: append([]routertable.Statement(nil), e.Statements()...), Bound: &b}
	}
	rec := func(s pointrecord.Substitution) *pointrecord.Record {
		return &pointrecord.Record{Substitutions: []pointrecord.Substitution{s}}
	}
	if err := d.CheckRouterSubstitutions(rec(carried())); err != nil {
		t.Fatalf("a substitution that carries the entry's own differences and bound was refused: %v", err)
	}
	for name, c := range map[string]struct {
		edit func(*pointrecord.Substitution)
		want string
	}{
		"a difference dropped":  {func(s *pointrecord.Substitution) { s.Difference = s.Difference[1:] }, "differences are not entry"},
		"a difference reworded": {func(s *pointrecord.Substitution) { s.Difference[0].Text += " (softened)" }, "differences are not entry"},
		"no differences":        {func(s *pointrecord.Substitution) { s.Difference = nil }, "differences are not entry"},
		"a ratio lowered":       {func(s *pointrecord.Substitution) { s.Bound.Ratios[0].MaxRatio = 1 }, "bound is not entry"},
		"a ratio dropped":       {func(s *pointrecord.Substitution) { s.Bound.Ratios = s.Bound.Ratios[1:] }, "bound is not entry"},
		"no bound":              {func(s *pointrecord.Substitution) { s.Bound = nil }, "bound is not entry"},
	} {
		s := carried()
		c.edit(&s)
		if err := d.CheckRouterSubstitutions(rec(s)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want an error containing %q", name, err, c.want)
		}
	}
}
