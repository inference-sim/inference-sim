package load

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/pointrecord"
	"github.com/inference-sim/blis-substrate/contracts/spec/routertable"
)

// A Dynamo recipe names its build by wheel version. The embedded table resolves the versions the
// P/D recipes use, and a point that names the commit the table gives for a build passes the
// record check. A version the table lacks stays unmapped, and a mapped build is still limited by
// the bound's regime: the table's bound is for one model on one chip, so another model is
// no_bound_for_regime however the build resolves.
func TestADynamoRecipeBuildResolvesThroughTheEmbeddedTable(t *testing.T) {
	d, err := Embedded()
	if err != nil {
		t.Fatal(err)
	}
	regime := routertable.Point{Workload: routertable.WorkloadBalanced, Concurrency: 64, Model: "minimax-m2.5",
		Hardware: "gb200-nvl72", EngineVersion: "0.32.0", Fabric: "ib-400g", Shape: "2P3D",
		Prefill: routertable.Parallel{TP: 1, PP: 1, DP: 1}, Decode: routertable.Parallel{TP: 4, PP: 1, DP: 1, ExpertParallel: true}}
	for _, c := range []struct {
		build string
		want  routertable.Outcome
	}{
		{"1.2.0.dev20260426", routertable.OutcomeMapped},
		{"1.3.0.dev20260601", routertable.OutcomeMapped},
		{"1.2.1", routertable.OutcomeMapped},
		{"1.3.0.dev20260614", routertable.OutcomeRevisionNotCovered},
		{"", routertable.OutcomeRevisionNotCovered},
	} {
		e, got := d.LookupRouterBuild("dynamo", "round-robin", nil, c.build)
		if got != c.want {
			t.Fatalf("build %q: %s, want %s", c.build, got, c.want)
		}
		if got != routertable.OutcomeMapped {
			continue
		}
		if _, o := e.BoundFor(regime); o != routertable.OutcomeMapped {
			t.Errorf("build %q on the bound's own regime: %s, want %s", c.build, o, routertable.OutcomeMapped)
		}
		other := regime
		other.Model = "minimax-m2.5-nvfp4"
		if _, o := e.BoundFor(other); o != routertable.OutcomeRegimeNotCovered {
			t.Errorf("build %q on another model: %s, want %s", c.build, o, routertable.OutcomeRegimeNotCovered)
		}
		// A router_differs that names the build's commit is accepted.
		b := e.BuildOf(c.build)
		s := pointrecord.Substitution{Kind: pointrecord.SubRouterDiffers, Subject: "dynamo", Entry: e.Name,
			Policy: string(e.Policy), Revision: b.Revision(), Regime: &regime, Evidence: "catalog/routers/" + e.Name + ".yaml",
			Difference: e.Statements()}
		bound := *e.Bound.For(regime)
		s.Bound = &bound
		if err := d.CheckRouterSubstitutions(&pointrecord.Record{Substitutions: []pointrecord.Substitution{s}}); err != nil {
			t.Errorf("build %q: a substitution naming the build's commit %s was refused: %v", c.build, b.Revision(), err)
		}
	}
}
