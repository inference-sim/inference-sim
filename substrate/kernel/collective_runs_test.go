package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// SUB-6, SUB-15: which collectives a layer launches is the engine's to say (Plan.Collectives.Runs),
// and the kernel prices what the plan says and evaluates no condition of its own.

func collectiveRunsCase(t *testing.T) Inputs {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	return planCase(t, d, "llama-3.1-70b-instruct", "h100", 1, "tp: 4, pp: 1, dp: 1", nil)
}

// editRuns is in with the plan's collective decisions edited by f, as stated and not completed.
func editRuns(in Inputs, f func(*plan.CollectiveRun)) Inputs {
	l := cloneLowered(*in.Lowered)
	for i := range l.Plan.Collectives.Runs {
		f(&l.Plan.Collectives.Runs[i])
	}
	in.Lowered = &l
	return in
}

// A collective the plan says does not run adds no time, and one it says runs adds some: the
// kernel prices the decision, whatever the layout's width would have said.
func TestACollectiveThePlanSaysDoesNotRunIsNotPriced(t *testing.T) {
	in := collectiveRunsCase(t)
	if len(in.Lowered.Plan.Collectives.Runs) == 0 {
		t.Fatal("the fixture states no collective decision, so the case tests nothing")
	}
	base := prices(t, in)
	for _, op := range []string{"AllReduce", "AllGather", "ReduceScatter", "All2All"} {
		flipped := 0
		moved := prices(t, editRuns(in, func(r *plan.CollectiveRun) {
			if r.Op == op && r.Runs {
				r.Runs, flipped = false, flipped+1
			}
		}))
		if flipped == 0 {
			continue // the layout runs none of this op
		}
		if moved == base {
			t.Errorf("the plan says the %d %s collectives do not run, and no price moves", flipped, op)
		}
	}
	// And the other way: an all-reduce the layout leaves out runs where the plan says so.
	tp1 := planCase(t, mustEmbedded(t), "llama-3.1-70b-instruct", "h100", 1, "tp: 1, pp: 1, dp: 1", nil)
	ran := 0
	forced := prices(t, editRuns(tp1, func(r *plan.CollectiveRun) {
		if r.Op == "AllReduce" && !r.Runs {
			r.Runs, ran = true, ran+1
		}
	}))
	if ran == 0 {
		t.Fatal("the tp 1 fixture leaves no all-reduce out")
	}
	if forced == prices(t, tp1) {
		t.Error("the plan says a collective runs at tp 1, and no price moves")
	}
}

func mustEmbedded(t *testing.T) *load.Data {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	return d
}

// A collective the graph conditions and the plan does not state is refused, naming the node:
// pricing it would either drop a cost or invent one.
func TestACollectiveThePlanDoesNotStateIsRefused(t *testing.T) {
	in := collectiveRunsCase(t)
	l := cloneLowered(*in.Lowered)
	gone := l.Plan.Collectives.Runs[0]
	l.Plan.Collectives.Runs = l.Plan.Collectives.Runs[1:]
	in.Lowered = &l
	_, err := New(in)
	if err == nil {
		t.Fatalf("a plan with no decision for %s %q in %s is priced", gone.Op, gone.Role, gone.Section)
	}
	for _, want := range []string{"Plan.Collectives.Runs", gone.Op, gone.Role} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal %q does not name %q", err, want)
		}
	}
}
