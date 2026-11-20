package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// A GLM DSA layer's indexer runs whole on every rank: vLLM builds its query projection as a
// ReplicatedLinear and its key projection with disable_tp, and its scorer computes every one of
// index_n_heads heads a rank (vllm/models/deepseek_v32/attention.py:66-85 at v0.33.0). Past the
// top-k, the scorer is the only part of a prefill step whose work grows with the cached prefix
// (the sparse attention reads the top-k, a constant), so that growth is the same on a TP4 rank
// as on a TP8 rank. Before, the graph left the scorer at the default split, a TP8 rank was charged
// half a TP4 rank's scoring, and AISimulate's GLM-5.3 attention-module table rose 6.6x faster with
// the prefix than the price did.
func TestGLMIndexerScoringDoesNotShardAcrossRanks(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	slope := func(tp string) float64 {
		in := planCase(t, d, "glm-5.3", "h200", 1, "tp: "+tp+", pp: 1, dp: 1", nil)
		in.Draft = nil
		k := mustKernel(t, in)
		device := func(prefix int) float64 {
			e := k.StepTime(kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
				{Scheduled: 2048, Computed: prefix, PromptLen: prefix + 2048}}})
			return (e.PerResource[kernel.ResourceSM] + e.PerResource[kernel.ResourceHBM]).Seconds()
		}
		return device(98304) - device(16384)
	}
	s4, s8 := slope("4"), slope("8")
	if !(s4 > 0 && s8 > 0) {
		t.Fatalf("a 2,048-token chunk's device price over 98,304 cached tokens less over 16,384: "+
			"TP4 %v s, TP8 %v s; both should be positive", s4, s8)
	}
	// Within 15%: what else in the step reads the prefix (a cache layout term) may shard.
	if r := s8 / s4; r < 0.85 || r > 1.15 {
		t.Errorf("the prefix's cost on a TP8 rank is %.3fx a TP4 rank's (TP8 %.3g s, TP4 %.3g s); "+
			"the indexer runs every head on every rank, so it should not shrink with TP", r, s8, s4)
	}
}

// The fused A-projection of a GLM DSA layer runs whole on every rank: vLLM builds it as
// DeepSeekV2FusedQkvAProjLinear, a MergedColumnParallelLinear with disable_tp=True (deepseek_v2.py:953-969,
// built at deepseek_v32/attention.py:247 and glm5next/common/attention.py:428, v0.33.0 4ba787ec). A graph that
// left it at the default split charged a TP8 rank an eighth of its weight and work. The resolved layout of the
// node is what the kernel prices from.
func TestGLMFusedAProjectionIsReplicated(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"glm-5", "glm-5.2", "glm-5.2-fp8", "glm-5.3", "glm-5.3-flash"} {
		g, err := d.Model(name)
		if err != nil {
			t.Fatal(err)
		}
		seen := 0
		for _, lk := range g.LayerKinds {
			for _, n := range lk.Nodes {
				if n.Role == "qkv_a_proj" {
					seen++
					if n.TP != model.TPReplicated {
						t.Errorf("%s layer kind %s: qkv_a_proj has layout %q; vLLM runs it whole on every rank", name, lk.ID, n.TP)
					}
				}
			}
		}
		if seen == 0 {
			t.Errorf("%s: no qkv_a_proj node found", name)
		}
	}
}
