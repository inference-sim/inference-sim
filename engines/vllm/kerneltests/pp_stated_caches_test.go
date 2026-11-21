package latencykernel_test

import (
	"errors"
	"strings"
	"testing"

	. "github.com/inference-sim/inference-sim/substrate/kernel"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	"github.com/inference-sim/inference-sim/substrate/contracts/enginefacts"
	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// Pipeline parallelism over a graph that states its caches: each stage's rank holds the caches
// of its own layers, in the pool the engine draws over the global KV cache groups projected onto
// them, and the stages split the layers as get_pp_indices does.

const mistralLarge4 = "mistral-large-4-1t-a52b"

// ppBlock is the block size the DSA-family models' indexers run, which these tests state.
const ppBlock = 64

// ppInputs is dcpMLAFixture laid out as pp stages of TP8 over pp 8-GPU nodes, serving the
// catalog model name.
func ppInputs(t testing.TB, name string, pp int) Inputs {
	t.Helper()
	in := pcpInputs(t, dcpMLAFixture, 1)
	withModel(t, &in, name)
	editInputsEngine(&in, func(e *engineconfig.Engine) { e.BlockSize = ppBlock })
	pool := &in.Deployment.Pools[0]
	pool.Parallel.PP = pp
	pool.Nodes, in.Scenario.Cluster.Nodes = pp, pp
	return in
}

// ppMistral is Mistral Large 4 laid out as pp stages of TP8 over pp 8-GPU H200 nodes.
func ppMistral(t testing.TB, pp int) Inputs {
	t.Helper()
	return ppInputs(t, mistralLarge4, pp)
}

// stackOf is the model's expanded layer-kind sequence.
func stackOf(in Inputs) []string { return in.Model.Stack.Expand() }

// truncated is in with its stack cut to the layers [lo, hi) of the original, on one rank: the
// single-stage model the stage [lo, hi) of a pipeline stands for.
func truncated(t testing.TB, in Inputs, lo, hi int) Inputs {
	t.Helper()
	out := ppMistral(t, 1)
	out.Model.Stack = model.Stack{Prologue: stackOf(in)[lo:hi]}
	return out
}

// ppSplit is get_pp_indices: layers // pp each, the remainder one layer apiece to the stages
// second-to-last backwards (vllm/distributed/utils.py:211-265).
func ppSplit(layers, pp int) []int {
	n := make([]int, pp)
	for i := range n {
		n[i] = layers / pp
	}
	for i := 2; i < layers%pp+2; i++ {
		n[pp-i]++
	}
	return n
}

// LAW: a stage's rank holds what a single-stage model of exactly that stage's layers holds of
// the stated caches -- one pool block costs the same, and so does a page of every KV layer --
// for any pp, so a pp > 1 pool is the layers' KV divided over the stages.
func TestPipelineStageHoldsItsLayersStatedCaches(t *testing.T) {
	for _, pp := range []int{2, 3} {
		in := ppMistral(t, pp)
		k := build(t, in)
		stages := k.PipelineStageMemory()
		if len(stages) != pp {
			t.Fatalf("pp %d: %d stages", pp, len(stages))
		}
		lo := 0
		for s, n := range ppSplit(len(stackOf(in)), pp) {
			one := build(t, truncated(t, in, lo, lo+n))
			lo += n
			block := one.SequenceVariableBytes(ppBlock)
			if got := stages[s].PerBlockBytes; got != block {
				t.Errorf("pp %d stage %d: a block holds %d B, a model of its %d layers holds %d B",
					pp, s, got, n, block)
			}
			if got, want := stages[s].LayerPageBytes, one.KVLayerPageBytes(ppBlock); got != want {
				t.Errorf("pp %d stage %d: a layer page holds %d B, a model of its %d layers holds "+
					"%d B", pp, s, got, n, want)
			}
		}
	}
}

// LAW: the stages' blocks sum to the whole model's, so no cache is dropped or charged twice.
func TestPipelineStatedCachesConserveAcrossStages(t *testing.T) {
	whole := build(t, ppMistral(t, 1)).SequenceVariableBytes(ppBlock)
	for _, pp := range []int{2, 3} {
		var sum int64
		for _, s := range build(t, ppMistral(t, pp)).PipelineStageMemory() {
			sum += s.PerBlockBytes
		}
		if sum != whole {
			t.Errorf("pp %d: stages hold %d B a block between them, the whole model %d B", pp,
				sum, whole)
		}
	}
}

// The sequence sizes are the binding stage's, and the binding stage is the one whose pool holds
// the fewest tokens, which the stated caches' per-block cost now enters.
func TestPipelineSequenceSizesAreTheBindingStagesStatedCaches(t *testing.T) {
	k := build(t, ppMistral(t, 2))
	var bound int64 = -1
	for _, s := range k.PipelineStageMemory() {
		if s.Binding {
			bound = s.PerBlockBytes
		}
	}
	if got := k.SequenceVariableBytes(ppBlock); got != bound {
		t.Errorf("a block holds %d B on the sequence sizes, %d B on the binding stage", got, bound)
	}
	if bound <= 0 {
		t.Errorf("the binding stage holds no stated cache: %d B a block", bound)
	}
}

// A layer that reads a cache whose owner sits on an earlier stage is refused by name: an engine
// worker holds only its own stage's caches. DeepSeek-V4.1-Flash's 40 layers split 20 and 20 at
// pp 2, every reader beside its owner, and 13, 14 and 13 at pp 3, where layer 13 reads the cache
// layer 8 owns on the stage before.
func TestPipelineRefusesACacheReadAcrossAStageBoundary(t *testing.T) {
	k, err := newKernel(ppInputs(t, "deepseek-v4.1-flash", 2))
	if err != nil {
		t.Fatalf("pp 2, every reader on its owner's stage: %v", err)
	}
	for s, m := range k.PipelineStageMemory() {
		if m.PerBlockBytes <= 0 {
			t.Errorf("pp 2 stage %d holds no stated cache: %d B a block", s, m.PerBlockBytes)
		}
	}
	_, err = newKernel(ppInputs(t, "deepseek-v4.1-flash", 3))
	if err == nil || !strings.Contains(err.Error(), "shared across a stage boundary") {
		t.Fatalf("pp 3, a reader across the stage boundary: got %v", err)
	}
}

// ungroupedRules is a rules pack that cannot group stated caches.
type ungroupedRules struct{ enginefacts.EngineRules }

func (ungroupedRules) KVCacheGroups(enginefacts.KVCacheGroupsInput) (enginefacts.KVCacheGrouping, error) {
	return enginefacts.KVCacheGrouping{}, errors.New("not transcribed")
}

// LAW: where the rules pack cannot group the caches and each is sized at its own page, a stage
// holds only the caches of its own layers, so the stages' blocks still sum to the whole model's.
func TestPipelineUngroupedCachesAreHeldOnTheirOwnStageOnly(t *testing.T) {
	ungrouped := func(pp int) Inputs {
		in := ppMistral(t, pp)
		in.Rules = ungroupedRules{in.Rules}
		return in
	}
	whole := build(t, ungrouped(1)).SequenceVariableBytes(ppBlock)
	var sum int64
	for s, m := range build(t, ungrouped(2)).PipelineStageMemory() {
		if m.PerBlockBytes <= 0 {
			t.Errorf("stage %d holds no stated cache: %d B a block", s, m.PerBlockBytes)
		}
		sum += m.PerBlockBytes
	}
	if sum != whole {
		t.Errorf("the stages hold %d B a block between them, the whole model %d B", sum, whole)
	}
}
