// Original to blis-vllm.

package lowering

import (
	"slices"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/engine/plan"
	"github.com/inference-sim/inference-sim/substrate/contracts/units"

	"github.com/inference-sim/inference-sim/engines/vllm/core/vllmsettings"
)

// Muse Spark computes its MoE pooler and unpooler a tensor-parallel slice at a time and
// all-gathers the columns when the layer takes the replicated-token path (tep_default,
// vllm/models/muse_spark/nvidia/model.py:505-523@4ba787ec9959): a part of capability 10 or later,
// tensor parallelism above one, no data parallelism, no sequence parallelism and no mega-MoE. A
// step of at most 256 tokens splits both; where the tensor-parallel group spans hosts the
// unpooler splits only from 32 tokens (shards_unpooler, min_sharded_unpooler_tokens, :535-539).
// The plan states the result (SUB-15: how an engine shards a tensor is its lowering, not a fact
// of the checkpoint).
var (
	muse     = where{"muse-spark-open-816b-a42b-nvfp4", "b200", "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"}
	museTP1  = where{"muse-spark-open-816b-a42b-nvfp4", "b200", "tp: 1, pp: 1, dp: 1"}
	museDP   = where{"muse-spark-open-816b-a42b-nvfp4", "b200", "tp: 4, pp: 1, dp: 2, enable_expert_parallel: true"}
	museHop  = where{"muse-spark-open-816b-a42b-nvfp4", "h200", "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"}
	museOpen = where{"muse-spark-open-816b-a42b", "b200", "tp: 8, pp: 1, dp: 1, enable_expert_parallel: true"}
)

func shardOf(l *plan.Lowered, role string) (plan.DecodeShard, bool) {
	for _, d := range l.Plan.DecodeShards {
		if d.Role == role {
			return d, true
		}
	}
	return plan.DecodeShard{}, false
}

func TestMuseSparkSplitsPoolerAndUnpoolerOverTPForSmallBlackwellSteps(t *testing.T) {
	l := lowered(t, muse, vllmsettings.Settings{})
	pool, okP := shardOf(l, "routed_expert_down_proj")
	unpool, okU := shardOf(l, "routed_expert_up_proj")
	if !okP || !okU || len(l.Plan.DecodeShards) != 2 {
		t.Fatalf("DecodeShards = %+v, want the pooler and unpooler", l.Plan.DecodeShards)
	}
	if pool.FromTokens != 1 || pool.ToTokens != 256 || unpool.FromTokens != 1 || unpool.ToTokens != 256 {
		t.Errorf("one host: pooler %d..%d, unpooler %d..%d tokens, want 1..256 for both",
			pool.FromTokens, pool.ToTokens, unpool.FromTokens, unpool.ToTokens)
	}
	if !slices.Equal(roles(lowered(t, museOpen, vllmsettings.Settings{})), roles(l)) {
		t.Error("the BF16 checkpoint of the same architecture states different shards")
	}
}

func roles(l *plan.Lowered) []string {
	var r []string
	for _, d := range l.Plan.DecodeShards {
		r = append(r, d.Role)
	}
	return r
}

func TestMuseSparkUnpoolerSplitsOnlyFromThirtyTwoTokensWhereTheTPGroupSpansHosts(t *testing.T) {
	chip, err := data.Chip(muse.chip)
	if err != nil {
		t.Fatal(err)
	}
	chip.GPUsPerNode = 4
	l, err := tryLowerOn(t, muse, chip, vllmsettings.Settings{})
	if err != nil {
		t.Fatal(err)
	}
	pool, _ := shardOf(l, "routed_expert_down_proj")
	unpool, _ := shardOf(l, "routed_expert_up_proj")
	want := func(d plan.DecodeShard, from, to units.Tokens) {
		t.Helper()
		if d.FromTokens != from || d.ToTokens != to {
			t.Errorf("%s: %d..%d tokens, want %d..%d", d.Role, d.FromTokens, d.ToTokens, from, to)
		}
	}
	want(pool, 1, 256)
	want(unpool, 32, 256)
}

func TestMuseSparkStatesNoSplitOffTheReplicatedTokenPath(t *testing.T) {
	for name, at := range map[string]where{
		"tensor parallelism 1": museTP1,
		"data parallelism 2":   museDP,
		"Hopper":               museHop,
	} {
		l, err := tryLower(t, at, vllmsettings.Settings{})
		if err != nil {
			t.Logf("%s: refused (%v)", name, err)
			continue
		}
		if len(l.Plan.DecodeShards) != 0 {
			t.Errorf("%s: DecodeShards = %+v, want none", name, l.Plan.DecodeShards)
		}
	}
}

func TestOnlyMuseSparkStatesDecodeShards(t *testing.T) {
	for _, at := range []where{dense, denseB, moeOne, v41} {
		if l := lowered(t, at, vllmsettings.Settings{}); len(l.Plan.DecodeShards) != 0 {
			t.Errorf("%s: DecodeShards = %+v, want none", at.model, l.Plan.DecodeShards)
		}
	}
}

// Each condition of the replicated-token path turns the split off on its own.
func TestDecodeShardsEachConditionOfTheReplicatedTokenPathIsNeeded(t *testing.T) {
	const arch = "MuseSparkForConditionalGeneration"
	base := layout{TP: 8, DP: 1, GPUsPerNode: 8}
	if len(decodeShards(arch, 10, base, false)) != 2 {
		t.Fatal("the base layout does not split")
	}
	for name, got := range map[string][]plan.DecodeShard{
		"Hopper":               decodeShards(arch, 9, base, false),
		"tensor parallelism 1": decodeShards(arch, 10, layout{TP: 1, DP: 1}, false),
		"data parallelism 2":   decodeShards(arch, 10, layout{TP: 8, DP: 2, GPUsPerNode: 8}, false),
		"sequence-parallel MoE": decodeShards(arch, 10,
			layout{TP: 8, DP: 1, GPUsPerNode: 8, SequenceParallelMoE: true}, false),
		"model sequence parallelism": decodeShards(arch, 10,
			layout{TP: 8, DP: 1, GPUsPerNode: 8, ModelSequenceParallel: true}, false),
		"mega-MoE":      decodeShards(arch, 10, base, true),
		"another model": decodeShards("LlamaForCausalLM", 10, base, false),
	} {
		if len(got) != 0 {
			t.Errorf("%s: DecodeShards = %+v, want none", name, got)
		}
	}
}
