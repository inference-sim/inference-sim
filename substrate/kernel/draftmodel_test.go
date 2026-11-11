package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

func draftGraphs() (target, draft *model.Graph) {
	norm := model.Node{Op: model.OpElementwise, Role: "input_norm"}
	target = &model.Graph{Kind: "ModelGraph", Name: "t",
		Global:     model.GlobalShape{HiddenSize: 64, VocabSize: 100, WeightDType: model.DTypeBF16},
		LayerKinds: []model.LayerKind{{ID: "layer", Nodes: []model.Node{norm}}},
		Stack:      model.Stack{Pattern: []string{"layer"}, Repeat: 4},
		Head:       []model.Node{{Op: model.OpGEMM, Role: "lm_head", N: 100, K: 64}}}
	draft = &model.Graph{Kind: "ModelGraph", Name: "d",
		Global:     model.GlobalShape{HiddenSize: 64, VocabSize: 100, WeightDType: model.DTypeBF16},
		LayerKinds: []model.LayerKind{{ID: "layer", Nodes: []model.Node{norm}}, {ID: "ctx", Nodes: []model.Node{norm}}},
		Stack:      model.Stack{Pattern: []string{"layer"}, Repeat: 2},
		Head:       []model.Node{{Op: model.OpGEMM, Role: "lm_head", N: 100, K: 64}},
		Speculator: &model.Speculator{Method: "dspark", NumSpec: 7, Drafting: model.DraftingParallelBlock,
			Stack:        model.Stack{Pattern: []string{"layer"}, Repeat: 2},
			ContextStack: &model.Stack{Pattern: []string{"ctx"}, Repeat: 1},
			Head:         []model.Node{{Op: model.OpGEMM, Role: "dspark_lm_head", N: 100, K: 64, Aliases: "head.lm_head"}}}}
	return target, draft
}

// A named drafter joins the target as its speculator: its layer kinds under a prefix that
// cannot collide with the target's (both graphs name a kind "layer"), its stacks renamed with
// them, and an alias of the drafter's own head dropped, since that head is the drafter's own
// weight rather than the target's.
func TestANamedDrafterBecomesTheTargetsSpeculator(t *testing.T) {
	target, draft := draftGraphs()
	g, err := withDrafter(target, draft)
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]bool{}
	for _, lk := range g.LayerKinds {
		if ids[lk.ID] {
			t.Errorf("layer kind %q appears twice", lk.ID)
		}
		ids[lk.ID] = true
	}
	if !ids["layer"] || !ids[draftKindPrefix+"layer"] || !ids[draftKindPrefix+"ctx"] {
		t.Errorf("kinds %v: want the target's and the drafter's, prefixed", ids)
	}
	sp := g.Speculator
	if sp == nil || sp.Stack.Pattern[0] != draftKindPrefix+"layer" ||
		sp.ContextStack.Pattern[0] != draftKindPrefix+"ctx" {
		t.Fatalf("speculator %+v: stacks not renamed", sp)
	}
	if a := sp.Head[0].Aliases; a != "" {
		t.Errorf("the drafter's own head aliases %q; it is the drafter's weight", a)
	}
	if target.Speculator != nil || len(target.LayerKinds) != 1 {
		t.Error("the target graph was modified")
	}
}

// What would leave no answer to which drafter runs is refused. (A target that states its own
// speculator is not here: a named drafter replaces it; see namedrafter_test.go.)
func TestADrafterThatCannotBeGraftedIsRefused(t *testing.T) {
	for name, c := range map[string]struct {
		edit func(target, draft *model.Graph)
		want string
	}{
		"a drafter that states none": {func(_, d *model.Graph) { d.Speculator = nil }, "states no speculator"},
		"a different hidden size":    {func(_, d *model.Graph) { d.Global.HiddenSize = 32 }, "hidden size"},
	} {
		t.Run(name, func(t *testing.T) {
			target, draft := draftGraphs()
			c.edit(target, draft)
			if _, err := withDrafter(target, draft); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err %v, want one containing %q", err, c.want)
			}
		})
	}
}

// Graph is the graph the kernel priced: with a named drafter, the target and the drafter's
// layers together, so an engine model sizing KV groups reads exactly what was priced.
func TestGraphIsWhatThePlanPriced(t *testing.T) {
	target, draft := draftGraphs()
	g, err := withDrafter(target, draft)
	if err != nil {
		t.Fatal(err)
	}
	k := &Kernel{graph: g}
	if got := k.Graph(); got.Speculator == nil || len(got.LayerKinds) != 3 {
		t.Errorf("Graph() %d kinds, speculator %v: want the grafted graph", len(got.LayerKinds), got.Speculator)
	}
}

// A grafted drafter's weights are its own checkpoint's: every weight node that states no dtype
// takes the drafter's global dtype and scale, not the target's, its own head included; an alias
// reads the target's weights and keeps no dtype of its own.
func TestADraftersNodesKeepItsOwnDType(t *testing.T) {
	target, draft := draftGraphs()
	target.Global.WeightDType = model.DTypeFP8
	target.Global.WeightScale = &model.WeightScale{Elements: 16384, Bytes: 4}
	draft.Speculator.Head = append(draft.Speculator.Head, model.Node{Op: model.OpGEMM, Role: "proj", N: 64, K: 64})
	draft.LayerKinds[0].Nodes = append(draft.LayerKinds[0].Nodes, model.Node{Op: model.OpGEMM, Role: "fc", N: 64, K: 64})
	draft.Head = nil
	g, err := withDrafter(target, draft)
	if err != nil {
		t.Fatal(err)
	}
	for _, lk := range g.LayerKinds {
		if !strings.HasPrefix(lk.ID, draftKindPrefix) {
			continue
		}
		for _, n := range lk.Nodes {
			if n.Op == model.OpGEMM && (n.WeightDType != model.DTypeBF16 || n.WeightScale != nil) {
				t.Errorf("drafter node %s: dtype %q scale %v, want the drafter's bf16", n.Role, n.WeightDType, n.WeightScale)
			}
		}
	}
	for _, n := range g.Speculator.Head {
		switch {
		case n.Aliases != "" && n.WeightDType != "":
			t.Errorf("aliased head node %s states dtype %q", n.Role, n.WeightDType)
		case n.Aliases == "" && n.WeightDType != model.DTypeBF16:
			t.Errorf("drafter head node %s: dtype %q, want bf16", n.Role, n.WeightDType)
		}
	}
	for _, n := range g.LayerKinds[0].Nodes {
		if n.WeightDType != "" {
			t.Errorf("the target's node %s was stamped %q", n.Role, n.WeightDType)
		}
	}
}

// A drafter that ships no head says it shares its target's. Grafted, its head alias then stays
// an alias of the target's head, whose weights are resident once, where an alias without the
// flag is the drafter's own weight (it is dropped, and the node counts as the drafter's).
func TestADrafterThatSharesItsTargetsHeadKeepsTheAlias(t *testing.T) {
	target, draft := draftGraphs()
	draft.Speculator.SharesTargetHead = true
	g, err := withDrafter(target, draft)
	if err != nil {
		t.Fatal(err)
	}
	if a := g.Speculator.Head[0].Aliases; a != "head.lm_head" {
		t.Fatalf("shared head alias %q, want head.lm_head", a)
	}
	if _, ok := g.AliasTarget(g.Speculator.Head[0].Aliases); !ok {
		t.Error("the kept alias resolves to no node of the grafted graph")
	}
	// The same drafter without the flag owns its head, as before.
	target, draft = draftGraphs()
	g, err = withDrafter(target, draft)
	if err != nil {
		t.Fatal(err)
	}
	if a := g.Speculator.Head[0].Aliases; a != "" {
		t.Errorf("an own head aliases %q", a)
	}
}

func TestASharedHeadTheTargetLacksIsRefused(t *testing.T) {
	target, draft := draftGraphs()
	draft.Speculator.SharesTargetHead = true
	target.Head = nil
	if _, err := withDrafter(target, draft); err == nil || !strings.Contains(err.Error(), "states no such node") {
		t.Fatalf("got %v, want a refusal naming the missing head", err)
	}
}

// The head a drafter shares must be the shape it was built for: another vocabulary or hidden
// width would price the draft head at one shape and read the target's weights at another.
func TestASharedHeadOfAnotherShapeIsRefused(t *testing.T) {
	for name, edit := range map[string]func(*model.Node){
		"another vocabulary": func(n *model.Node) { n.N = 120 },
		"another width":      func(n *model.Node) { n.K = 32 },
	} {
		target, draft := draftGraphs()
		draft.Speculator.SharesTargetHead = true
		edit(&target.Head[0])
		if _, err := withDrafter(target, draft); err == nil || !strings.Contains(err.Error(), "another vocabulary or width") {
			t.Errorf("%s: got %v, want a refusal naming the shape", name, err)
		}
	}
	target, draft := draftGraphs()
	draft.Speculator.SharesTargetHead = true
	if _, err := withDrafter(target, draft); err != nil {
		t.Errorf("a head of the same shape was refused: %v", err)
	}
}
