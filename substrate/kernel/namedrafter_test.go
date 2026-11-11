package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// A named separate drafter replaces the target graph's own speculator (its MTP layers), as
// vLLM runs only the named checkpoint when speculative_config.model is set. These tests price
// real catalog pairs: the target's in-graph speculator must neither be priced nor held in
// memory, and what is refused must still be refused.

// withNamedDrafter is the case with the deployment speculating through the drafter the catalog
// names, at the given method and draft length.
func withNamedDrafter(t *testing.T, d *load.Data, target, drafter, chip, method string, n int) Inputs {
	t.Helper()
	in := quantCase(t, d, target, chip)
	dr, err := d.Model(drafter)
	if err != nil {
		t.Fatal(err)
	}
	in.Draft = dr
	return withEngine(in, func(e *enginesettings.Settings) {
		e.Speculative = &enginesettings.Speculative{Method: method, NumSpecTokens: n, DraftModel: drafter}
	})
}

// strippedOfItsSpeculator is the same inputs with the target graph's own speculator removed
// first, which is the form that already priced a named drafter before the replacement rule.
func strippedOfItsSpeculator(in Inputs) Inputs {
	g := *in.Model
	g.Speculator = nil
	in.Model = &g
	return in
}

func TestANamedDrafterPricesInPlaceOfTheTargetsOwnSpeculator(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ target, drafter string }{
		{"minimax-m3-mxfp8", "minimax-m3-eagle3"},
		{"minimax-m3-mxfp8", "minimax-m3-eagle3-gqa"},
		{"minimax-m3", "minimax-m3-eagle3"},
	} {
		for _, chip := range []string{"h200", "b200"} {
			name := c.target + "+" + c.drafter + "/" + chip
			t.Run(name, func(t *testing.T) {
				in := withNamedDrafter(t, d, c.target, c.drafter, chip, "eagle3", 3)
				if in.Model.Speculator == nil || in.Model.Speculator.Method != "minimax_m3_mtp" {
					t.Fatalf("the target graph does not state the in-graph speculator this test replaces: %+v", in.Model.Speculator)
				}
				k := mustKernel(t, in)
				if got := k.plan.Draft.Method; got != "eagle3" {
					t.Fatalf("the priced speculator is %q, want the named drafter's eagle3", got)
				}
				if !k.speculating {
					t.Error("the deployment speculates and the kernel does not")
				}
				// Priced as the drafter alone: identical to a target that never stated a
				// speculator, memory and every step shape.
				want := pricesOf(mustKernel(t, strippedOfItsSpeculator(in)))
				if got := pricesOf(k); got != want {
					t.Errorf("the target's own speculator is priced beside the named drafter:\n got %s\nwant %s", got, want)
				}
				// And not as the in-graph MTP: the same target speculating through its own
				// layers (method mtp, no drafter named) prices differently.
				own := in
				own.Draft = nil
				own = withEngine(own, func(e *enginesettings.Settings) {
					e.Speculative = &enginesettings.Speculative{Method: "mtp", NumSpecTokens: 3}
				})
				if pricesOf(mustKernel(t, own)) == pricesOf(k) {
					t.Error("a named EAGLE3 drafter prices exactly as the target's own MTP")
				}
				// The MTP layer is not resident: the drafter's weights are, the MTP's are not.
				for _, a := range k.assumptions {
					if a.Name == "speculator" {
						t.Errorf("a speculator assumption was recorded: %s", a.Scope)
					}
				}
			})
		}
	}
}

// Without a named drafter the target's own speculator is still the one priced.
func TestWithNoNamedDrafterTheTargetsOwnSpeculatorIsStillPriced(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	in := quantCase(t, d, "minimax-m3-mxfp8", "b200")
	in = withEngine(in, func(e *enginesettings.Settings) {
		e.Speculative = &enginesettings.Speculative{Method: "mtp", NumSpecTokens: 1}
	})
	k := mustKernel(t, in)
	if k.plan.Draft == nil || k.plan.Draft.Method != "minimax_m3_mtp" || !k.speculating {
		t.Fatalf("the target's own MTP is not priced: %+v", k.plan.Draft)
	}
	if pricesOf(k) == pricesOf(mustKernel(t, strippedOfItsSpeculator(in))) {
		t.Error("stripping the in-graph speculator changes nothing, so the test proves nothing")
	}
}

// The combinations that stay refused with a drafter named.
func TestANamedDrafterStillRefusedWhereNoOneAnswerRemains(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	pair := func() Inputs {
		return withNamedDrafter(t, d, "minimax-m3-mxfp8", "minimax-m3-eagle3", "b200", "eagle3", 3)
	}
	for _, c := range []struct {
		name string
		in   func() Inputs
		want string
	}{
		{"a deployment method that is not the drafter's", func() Inputs {
			return withEngine(pair(), func(e *enginesettings.Settings) { e.Speculative.Method = "mtp" })
		}, "a mtp drafter is not described"},
		{"a drafter that states no speculator", func() Inputs {
			in := pair()
			g := *in.Draft
			g.Speculator = nil
			in.Draft = &g
			return in
		}, "states no speculator"},
		{"a drafter of another hidden size", func() Inputs {
			in := pair()
			g := *in.Draft
			g.Global.HiddenSize = 4096
			in.Draft = &g
			return in
		}, "hidden size"},
		{"a drafter graph not given", func() Inputs {
			in := pair()
			in.Draft = nil
			return in
		}, "whose graph was not given"},
		{"a drafter other than the one named", func() Inputs {
			in := pair()
			g := *in.Draft
			g.Name = "someone-else"
			in.Draft = &g
			return in
		}, "drafter graph given is"},
	} {
		t.Run(c.name, func(t *testing.T) {
			if _, err := New(c.in()); err == nil || !strings.Contains(err.Error(), c.want) {
				t.Errorf("err %v, want one containing %q", err, c.want)
			}
		})
	}
}

// A kind only the replaced speculator ran leaves with it; a kind the target also runs stays.
func TestTheReplacedSpeculatorsOwnKindsLeaveTheGraph(t *testing.T) {
	target, draft := draftGraphs()
	norm := model.Node{Op: model.OpElementwise, Role: "input_norm"}
	target.LayerKinds = append(target.LayerKinds, model.LayerKind{ID: "mtp_only", Nodes: []model.Node{norm}})
	target.Speculator = &model.Speculator{Method: "deepseek_mtp", NumSpec: 1,
		Stack: model.Stack{Pattern: []string{"mtp_only", "layer"}, Repeat: 1}}
	g, err := withDrafter(target, draft)
	if err != nil {
		t.Fatal(err)
	}
	have := map[string]bool{}
	for _, lk := range g.LayerKinds {
		have[lk.ID] = true
	}
	if have["mtp_only"] {
		t.Error("a kind only the replaced speculator ran is still in the graph")
	}
	if !have["layer"] || !have[draftKindPrefix+"layer"] {
		t.Errorf("kinds %v: the target's own kind or the drafter's is missing", have)
	}
	if g.Speculator.Method != "dspark" {
		t.Errorf("speculator %q, want the drafter's", g.Speculator.Method)
	}
	for _, pr := range g.Validate().All() {
		if strings.Contains(pr.Message, "never used") || strings.Contains(pr.Message, "names no layer kind") {
			t.Errorf("the grafted graph: %s", pr.Message)
		}
	}
	if target.Speculator.Method != "deepseek_mtp" || len(target.LayerKinds) != 2 {
		t.Error("the target graph was modified")
	}
}

// Every separate drafter the catalog names grafts onto every target it lists, and the grafted
// graph validates: an alias the drafter reads from its target (the embedding table, the head)
// must resolve in each target's graph, whatever handler derived that graph. A target whose
// handler states no embedding fails here, not in a deployment.
func TestEveryCatalogDrafterGraftsOntoEachTargetItNames(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	pairs := 0
	for _, name := range d.Models() {
		id, err := d.ModelIdentity(name)
		if err != nil {
			t.Fatal(err)
		}
		for _, targetName := range id.Targets {
			pairs++
			t.Run(targetName+"+"+name, func(t *testing.T) {
				draft, err := d.Model(name)
				if err != nil {
					t.Fatal(err)
				}
				target, err := d.Model(targetName)
				if err != nil {
					t.Fatal(err)
				}
				g, err := withDrafter(target, draft)
				if err != nil {
					t.Fatal(err)
				}
				if p := g.Validate(); !p.OK() {
					t.Errorf("the target graph with drafter %s does not validate:\n%v", name, p)
				}
			})
		}
	}
	if pairs < 5 {
		t.Errorf("only %d target and drafter pairs found; the catalog lists more", pairs)
	}
}

// The NVFP4 checkpoint InferenceX serves with an EAGLE3 head names the head: both must hold
// together, with the head priced in place of the in-graph MTP at the checkpoint's own layout. The
// engine resolves the checkpoint's quantization method itself and states no quantization to the
// kernel (the layout it serves is Settings.ServedWeights).
func TestTheMixedPrecisionTargetPricesWithEitherEagle3HeadAtItsOwnLayout(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, drafter := range []string{"minimax-m3-eagle3", "minimax-m3-eagle3-gqa"} {
		for _, chip := range []string{"b200", "b300"} {
			in := withNamedDrafter(t, d, "minimax-m3-nvfp4", drafter, chip, "eagle3", 3)
			k := mustKernel(t, in)
			if k.plan.Draft == nil || k.plan.Draft.Method != "eagle3" || !k.speculating {
				t.Fatalf("%s on %s: the named head is not the priced speculator", drafter, chip)
			}
			// The same head on the MXFP8 target prices differently: the experts are the NVFP4 ones.
			if pricesOf(k) == pricesOf(mustKernel(t, withNamedDrafter(t, d, "minimax-m3-mxfp8", drafter, chip, "eagle3", 3))) {
				t.Errorf("%s on %s: the NVFP4 target prices as the MXFP8 one", drafter, chip)
			}
		}
	}
}
