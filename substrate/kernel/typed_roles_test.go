package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/load"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"gopkg.in/yaml.v3"
)

// typedRoleCase returns the first perturbation scenario of the named catalog model.
func typedRoleCase(t *testing.T, name string) perturbationCase {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range perturbationScenarios(t, d, name) {
		return c
	}
	t.Fatalf("no buildable scenario for %s", name)
	return perturbationCase{}
}

// priceWith prices the case's model after edit has changed a copy of its graph. A model the
// cost model refuses returns "refused: ..." so a refusal counts as a change.
func priceWith(t *testing.T, c perturbationCase, edit func(*model.Node)) string {
	t.Helper()
	b, err := yaml.Marshal(*c.in.Model)
	if err != nil {
		t.Fatal(err)
	}
	var g model.Graph
	if err := yaml.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	walk := func(nodes []model.Node) {
		for i := range nodes {
			edit(&nodes[i])
		}
	}
	for i := range g.LayerKinds {
		walk(g.LayerKinds[i].Nodes)
	}
	in := c.in
	in.Model = &g
	k, err := New(in)
	if err != nil {
		return "refused: " + err.Error()
	}
	return pricesOf(k)
}

// SUB-20: relabelling a block-index scorer, and the attention that reads the engine's cache,
// moves no price, because holds_engine_kv decides which is which.
func TestRelabellingAScorerOrAMainAttentionMovesNoPrice(t *testing.T) {
	c := typedRoleCase(t, "deepseek-v4.1-flash")
	want := priceWith(t, c, func(*model.Node) {})
	if strings.HasPrefix(want, "refused") {
		t.Fatal(want)
	}
	var scorers, mains int
	got := priceWith(t, c, func(n *model.Node) {
		if n.Op != model.OpAttention {
			return
		}
		if n.ReadsEngineKV() {
			mains++
			n.Role = "renamed_main"
		} else {
			scorers++
			n.Role = "renamed_scorer"
		}
	})
	if scorers == 0 || mains == 0 {
		t.Fatalf("the model has %d scorers and %d main attentions; the test needs both", scorers, mains)
	}
	if got != want {
		t.Error("relabelling a scorer and a main attention moved a price")
	}
	// A main attention given a scorer's old label, and a scorer left unlabelled, are the
	// same two nodes: the label never told them apart.
	swapped := priceWith(t, c, func(n *model.Node) {
		if n.Op == model.OpAttention {
			if n.ReadsEngineKV() {
				n.Role = "block_index_scores"
			} else {
				n.Role = ""
			}
		}
	})
	if swapped != want {
		t.Error("swapping the labels of scorers and main attentions moved a price")
	}
}

// Known bad for the check above: the field is read. Flipping holds_engine_kv on the scorers
// moves the price (or is refused), so the unchanged price above is not an artefact of the
// field being ignored.
func TestFlippingHoldsEngineKVOnAScorerMovesThePrice(t *testing.T) {
	c := typedRoleCase(t, "deepseek-v4.1-flash")
	want := priceWith(t, c, func(*model.Node) {})
	got := priceWith(t, c, func(n *model.Node) {
		if n.Op == model.OpAttention && !n.ReadsEngineKV() {
			n.HoldsEngineKV = model.Bool(true)
		}
	})
	if got == want {
		t.Error("stating holds_engine_kv true on every scorer moved no price")
	}
}

// The same for shared: relabelling a shared-expert group moves no price, and flipping its
// shared field does.
func TestSharedDecidesTheSharedExpertGroup(t *testing.T) {
	c := typedRoleCase(t, "nemotron-3-ultra-550b-a55b-nvfp4")
	want := priceWith(t, c, func(*model.Node) {})
	var shared int
	got := priceWith(t, c, func(n *model.Node) {
		if n.IsSharedExpert() {
			shared++
			n.Role = "renamed_shared"
		}
	})
	if shared == 0 {
		t.Fatal("the model has no shared-expert group")
	}
	if got != want {
		t.Error("relabelling a shared-expert group moved a price")
	}
	flipped := priceWith(t, c, func(n *model.Node) {
		if n.IsSharedExpert() {
			n.Shared = model.Bool(false)
		}
	})
	if flipped == want {
		t.Error("stating shared false on the shared-expert group moved no price")
	}
}
