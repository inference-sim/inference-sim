package latencykernel

import (
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// SUB-15, SUB-11: which nodes of the model the engine runs on a rank's shard of the step's tokens
// under the model's own sequence parallelism is the engine's lowering (plan.PoolPlan.
// TokenShardedRoles), not a field of the model graph. A node the plan names runs on
// ceil(tokens / tp) rows while that parallelism is on, and on every row while it is off; a role no
// node of the model has prices nothing, since an engine's table spans the variants of an
// architecture.
func TestATokenShardedRoleThePlanNamesRunsOnItsRanksTokensUnderModelSequenceParallelism(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	price := func(sp bool, roles ...string) float64 {
		k := mustKernel(t, planCase(t, d, "deepseek-v4.1-flash", "b200", 1, "tp: 4, pp: 1, dp: 1", func(l *plan.Lowered) {
			l.Plan.Collectives.SequenceParallelModel = sp
			l.Plan.TokenShardedRoles = roles
		}))
		b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 2048, PromptLen: 2048}}}
		return float64(k.StepTime(b).NoOverlap)
	}
	routed := []string{"router", "experts", "shared_gate_up", "shared_down"}
	mhc := []string{"hc_attn_fn", "hc_ffn_fn", "mhc_attn_pre", "mhc_final_post"}
	if got, none := price(true, routed...), price(true); got == none {
		t.Errorf("sequence parallelism on: naming the routed-MoE roles prices a step %v, as naming none does", got)
	}
	if got, none := price(true, mhc...), price(true); got == none {
		t.Errorf("sequence parallelism on: naming the mHC roles prices a step %v, as naming none does", got)
	}
	if got, none := price(false, append(routed, mhc...)...), price(false); got != none {
		t.Errorf("sequence parallelism off: naming roles prices a step %v, but %v with none named", got, none)
	}
	if got, none := price(true, "no_such_role"), price(true); got != none {
		t.Errorf("a role no node has prices a step %v, but %v with none named", got, none)
	}
	a, b := price(true, routed[:2]...), price(true, routed...)
	if a == b {
		t.Errorf("naming the shared experts as well as the router and routed experts prices a step %v, as without them", b)
	}
}
