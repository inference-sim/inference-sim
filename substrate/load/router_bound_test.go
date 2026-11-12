package load

import (
	"gopkg.in/yaml.v3"
	"os"
	"reflect"
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
	// The point: a balanced run at 64 clients on the entry's deployment and the 2P3D shape.
	regime := routertable.Point{Workload: routertable.WorkloadBalanced, Concurrency: 64, Model: "minimax-m2.5",
		Hardware: "gb200-nvl72", EngineVersion: "0.32.0", Fabric: "ib-400g", Shape: "2P3D",
		Prefill: routertable.Parallel{TP: 1, PP: 1, DP: 1}, Decode: routertable.Parallel{TP: 4, PP: 1, DP: 1, ExpertParallel: true}}
	carried := func() pointrecord.Substitution {
		// The part of the entry's bound that covers the point's regime, and none for another.
		b := *e.Bound.For(regime)
		b.Ratios = append([]routertable.Ratio(nil), b.Ratios...)
		return pointrecord.Substitution{Kind: pointrecord.SubRouterDiffers, Subject: "dynamo", Entry: e.Name,
			Policy: string(e.Policy), Revision: covered, Regime: &regime, Evidence: "catalog/routers/" + e.Name + ".yaml",
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
		"a ratio lowered":       {func(s *pointrecord.Substitution) { s.Bound.Ratios[0].MaxRatio = 1 }, "bound is not the part"},
		"a ratio dropped":       {func(s *pointrecord.Substitution) { s.Bound.Ratios = s.Bound.Ratios[1:] }, "bound is not the part"},
		"the whole bound, ratios of other regimes included": {func(s *pointrecord.Substitution) { b := *e.Bound; s.Bound = &b }, "bound is not the part"},
		"no regime":          {func(s *pointrecord.Substitution) { s.Regime = nil }, "no regime"},
		"an unsampled count": {func(s *pointrecord.Substitution) { r := regime; r.Concurrency = 100; s.Regime = &r }, "no bound for this regime"},
		"another fabric":     {func(s *pointrecord.Substitution) { r := regime; r.Fabric = "roce-400g"; s.Regime = &r }, "no bound for this regime"},
		"no bound":           {func(s *pointrecord.Substitution) { s.Bound = nil }, "bound is not the part"},
	} {
		s := carried()
		c.edit(&s)
		if err := d.CheckRouterSubstitutions(rec(s)); err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got %v, want an error containing %q", name, err, c.want)
		}
	}
}

// End to end: the bound Entry.BoundFor returns for a covered point is what the record carries, what
// PointRecord.Validate accepts and what CheckRouterSubstitutions verifies against the entry; a
// point outside the entry's regime gets no bound, and a record that carries one anyway is refused
// by both.
func TestTheBoundForAPointSurvivesTheRecordAndItsCheck(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	e, err := d.Router("dynamo-round-robin")
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile("../docs/contracts/examples/point-record.json")
	if err != nil {
		t.Fatal(err)
	}
	var base pointrecord.Record
	if err := yaml.Unmarshal(raw, &base); err != nil {
		t.Fatal(err)
	}
	base.Grade = pointrecord.G2
	point := routertable.Point{Workload: routertable.WorkloadBalanced, Concurrency: base.Identity.Concurrency, Model: "minimax-m2.5",
		Hardware: "gb200-nvl72", EngineVersion: "0.32.0", Fabric: "ib-400g", Shape: "2P3D",
		Prefill: routertable.Parallel{TP: 1, PP: 1, DP: 1}, Decode: routertable.Parallel{TP: 4, PP: 1, DP: 1, ExpertParallel: true}}
	recordOf := func(p routertable.Point, bound *routertable.Bound) *pointrecord.Record {
		r := base
		r.Substitutions = []pointrecord.Substitution{{Kind: pointrecord.SubRouterDiffers, Subject: "dynamo", Entry: e.Name,
			Policy: string(e.Policy), Revision: e.Revisions[0], Evidence: "catalog/routers/" + e.Name + ".yaml",
			Regime: &p, Difference: e.Statements(), Bound: bound}}
		return &r
	}
	bound, out := e.BoundFor(point)
	if out != routertable.OutcomeMapped || bound == nil || len(bound.Ratios) == 0 || len(bound.Ratios) >= len(e.Bound.Ratios) {
		t.Fatalf("BoundFor(%+v) = %d ratios of %d, %s; want the balanced ratios only", point, len(bound.Ratios), len(e.Bound.Ratios), out)
	}
	r := recordOf(point, bound)
	if p := r.Validate(); !p.OK() {
		t.Fatalf("the record that carries BoundFor's bound is invalid:\n%s", p.Error())
	}
	if err := d.CheckRouterSubstitutions(r); err != nil {
		t.Fatalf("the bound BoundFor returned cannot be persisted: %v", err)
	}
	// Another point of the same entry takes the other regime's ratios, not these.
	other := point
	other.Workload = routertable.WorkloadPrefillBound
	b2, out2 := e.BoundFor(other)
	if out2 != routertable.OutcomeMapped || reflect.DeepEqual(b2, bound) {
		t.Errorf("a prefill-bound point: %s, same bound as the balanced one: %v", out2, reflect.DeepEqual(b2, bound))
	}
	if err := d.CheckRouterSubstitutions(recordOf(other, bound)); err == nil {
		t.Error("the balanced bound was accepted for a prefill-bound point")
	}
	// An unsampled count has no bound; a record carrying the whole bound for it is refused.
	unsampled := point
	unsampled.Concurrency = 100
	if b, out := e.BoundFor(unsampled); b != nil || out != routertable.OutcomeRegimeNotCovered {
		t.Errorf("BoundFor at 100 clients = %v, %s", b, out)
	}
	bad := recordOf(unsampled, e.Bound)
	bad.Identity.Concurrency = 100
	if p := bad.Validate(); p.OK() {
		t.Error("PointRecord.Validate accepted a bound for an unsampled client count")
	}
	if err := d.CheckRouterSubstitutions(bad); err == nil {
		t.Error("CheckRouterSubstitutions accepted a bound for an unsampled client count")
	}
}
