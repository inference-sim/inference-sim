package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	"github.com/inference-sim/inference-sim/substrate/load"
)

// SUB-15, SUB-11: which replicated GEMMs the engine splits over the tensor-parallel ranks, and for
// which steps, is the engine's lowering (plan.PoolPlan.DecodeShards), not a field of the model
// graph. A GEMM the plan names is priced on a tp-th of its columns plus the all-gather that
// reassembles them, for exactly the steps its range covers; a pool whose plan names none prices
// every replicated GEMM whole.
func TestADecodeShardThePlanStatesSplitsTheGEMMItNamesForTheStepsItCovers(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const pooler, unpooler = "routed_expert_down_proj", "routed_expert_up_proj"
	open := func(shards ...plan.DecodeShard) *Kernel {
		return mustKernel(t, planCase(t, d, "muse-spark-open-816b-a42b-nvfp4", "gb200-nvl72", 2,
			"tp: 8, pp: 1, dp: 1", func(l *plan.Lowered) { l.Plan.DecodeShards = shards }))
	}
	decodes := func(n int) kernel.Batch {
		b := kernel.Batch{DecodeThreshold: 1}
		for range n {
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024})
		}
		return b
	}
	whole := open()
	for _, c := range []struct {
		name   string
		shards []plan.DecodeShard
		tokens int
		split  bool
	}{
		{"pooler, inside its range", []plan.DecodeShard{{Role: pooler, FromTokens: 1, ToTokens: 256}}, 16, true},
		{"pooler, at its last step", []plan.DecodeShard{{Role: pooler, FromTokens: 1, ToTokens: 256}}, 256, true},
		{"pooler, past its range", []plan.DecodeShard{{Role: pooler, FromTokens: 1, ToTokens: 256}}, 257, false},
		{"unpooler, before its first step", []plan.DecodeShard{{Role: unpooler, FromTokens: 32, ToTokens: 256}}, 31, false},
		{"unpooler, at its first step", []plan.DecodeShard{{Role: unpooler, FromTokens: 32, ToTokens: 256}}, 32, true},
		{"a range of one step", []plan.DecodeShard{{Role: unpooler, FromTokens: 8, ToTokens: 8}}, 8, true},
		{"a range of one step, one more token", []plan.DecodeShard{{Role: unpooler, FromTokens: 8, ToTokens: 8}}, 9, false},
	} {
		got, want := open(c.shards...).StepTime(decodes(c.tokens)).NoOverlap, whole.StepTime(decodes(c.tokens)).NoOverlap
		if c.split == (got == want) {
			t.Errorf("%s: a %d-token step prices %v split and %v whole; want them %s", c.name, c.tokens, got, want,
				map[bool]string{true: "different", false: "equal"}[c.split])
		}
	}

	both := open(plan.DecodeShard{Role: pooler, FromTokens: 1, ToTokens: 256},
		plan.DecodeShard{Role: unpooler, FromTokens: 1, ToTokens: 256})
	one := open(plan.DecodeShard{Role: pooler, FromTokens: 1, ToTokens: 256})
	if b, o := both.StepTime(decodes(16)).NoOverlap, one.StepTime(decodes(16)).NoOverlap; b == o {
		t.Errorf("naming the unpooler as well as the pooler prices a step %v, as the pooler alone does", b)
	}
}

// A plan that names a role no replicated GEMM of the model has is refused, so a table that has
// drifted from the graph does not price as though it sharded.
func TestADecodeShardNamingNoReplicatedGEMMIsRefused(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, role := range []string{"no_such_role", "o_proj"} {
		in := planCase(t, d, "muse-spark-open-816b-a42b-nvfp4", "gb200-nvl72", 2, "tp: 8, pp: 1, dp: 1",
			func(l *plan.Lowered) {
				l.Plan.DecodeShards = []plan.DecodeShard{{Role: role, FromTokens: 1, ToTokens: 256}}
			})
		_, err := New(in)
		if err == nil || !strings.Contains(err.Error(), role) || !strings.Contains(err.Error(), "replicated") {
			t.Errorf("role %q: err %v, want a refusal naming the role and saying no replicated GEMM has it", role, err)
		}
	}
}
