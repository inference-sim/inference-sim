package latencykernel

import (
	"reflect"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// Kimi-K3.1's sparse layers run a block-index scorer (a GQA attention that does not hold the
// engine's KV) beside the primary sparse-MLA attention, in one layer. A plan that lowers the two
// to distinct kernels states two entries for attention.decode in that layer, the scorer's marked
// Scorer; the kernel classes the primary as sparse_mla and the scorer as block_index_scores, so
// the registry's per-role pairs bind to the kernel of their own node. A layer without a scorer
// takes no scorer entry, and a scorer entry on such a layer is refused.
func TestKimiK31ScorerAndPrimaryAreClassedApartOnTheCommittedGraph(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	g, err := d.Model("kimi-k3.1")
	if err != nil {
		t.Fatal(err)
	}
	chip, err := d.Chip("b300")
	if err != nil {
		t.Fatal(err)
	}
	family, _ := chip.CapabilityMajor()
	g, err = g.ForCapability(family)
	if err != nil {
		t.Fatal(err)
	}
	primary := plan.KernelImpl{Library: "flashmla", Variant: "sparse_decode", CodeHash: "aaaaaaaaaaaa"}
	scorer := plan.KernelImpl{Library: "deepgemm", Variant: "paged_mqa_logits", CodeHash: "bbbbbbbbbbbb"}
	kinds := map[string]model.LayerKind{}
	for _, lk := range g.LayerKinds {
		kinds[lk.ID] = lk
	}
	var ops []plan.Operation
	var scoredLayers, plainLayer []int
	for i, id := range g.Stack.Expand() {
		hasScorer, sparse, attends := false, false, false
		for _, n := range kinds[id].Nodes {
			if n.Op != model.OpAttention {
				continue
			}
			attends = true
			if !n.ReadsEngineKV() {
				hasScorer = true
			} else if n.AttentionKind == model.AttentionSparseMLA {
				sparse = true
			}
		}
		if !attends {
			continue
		}
		ops = append(ops, plan.Operation{Layer: i, Primitive: plan.PrimAttentionDecode, Impl: primary})
		switch {
		case hasScorer:
			ops = append(ops, plan.Operation{Layer: i, Scorer: true, Primitive: plan.PrimAttentionDecode, Impl: scorer})
			scoredLayers = append(scoredLayers, len(ops)-1)
		case !sparse:
			plainLayer = append(plainLayer, len(ops)-1)
		}
	}
	if len(scoredLayers) == 0 {
		t.Fatal("the committed Kimi-K3.1 graph has no layer that runs a block-index scorer")
	}
	got, err := loweredOperations(g, ops, *chip, family)
	if err != nil {
		t.Fatal(err)
	}
	for _, i := range scoredLayers {
		if !got[i].Scorer || !reflect.DeepEqual(got[i].Chain, []string{"block_index_scores"}) || got[i].Impl.Variant != scorer.Variant {
			t.Errorf("scorer operation %d: %+v", i, got[i])
		}
		// The primary of the same layer is the entry before it.
		if p := got[i-1]; p.Scorer || !reflect.DeepEqual(p.Chain, []string{"sparse_mla"}) || p.Impl.Variant != primary.Variant {
			t.Errorf("primary operation %d, beside the scorer: %+v", i-1, p)
		}
	}
	for _, i := range plainLayer {
		if len(got[i].Chain) != 0 && got[i].Chain[0] == "sparse_mla" {
			t.Errorf("operation %d of a layer with no sparse attention is classed sparse_mla", i)
		}
	}
	// A scorer entry on a layer that runs none is refused, naming it.
	if len(plainLayer) > 0 {
		bad := append([]plan.Operation(nil), ops...)
		l := ops[plainLayer[0]].Layer
		bad = append(bad, plan.Operation{Layer: l, Scorer: true, Primitive: plan.PrimAttentionDecode, Impl: scorer})
		if _, err := loweredOperations(g, bad, *chip, family); err == nil {
			t.Errorf("a scorer entry on layer %d, which runs none, was accepted", l)
		}
	}
}
