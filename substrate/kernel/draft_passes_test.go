package latencykernel

import (
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"
)

// An autoregressive drafter's first pass can process one row per request or every token the
// target scheduled, and the later passes and the head can run for the requests that complete
// their prompt or for every request. Which one runs is the engine's execution decision, not a
// fact of the catalog graph, so the engine states it in the pool's plan (plan.DraftPolicy's
// FirstPassRows and PassRequests, handed to the kernel as Inputs.DraftFirstPassRows and
// DraftPassRequests) and a deployment that speculates through an autoregressive drafter without
// stating both is refused with draft_passes_unstated. The kernel names no engine and no method:
// the in-graph MTP speculators and a named EAGLE3 head are refused alike. A parallel-block
// drafter (DSpark) states its target-wide work as its context stack and needs neither.
//
// The facts are not engine settings: enginesettings.Speculative has no field for them, so the
// cost model never reads them from the settings it is given.

// namedEagle3 builds the catalog pair of a target with no speculator of its own and a named
// EAGLE3 head, with the engine's speculative block and the pool plan's statement as given.
func namedEagle3(t *testing.T, name string, rows plan.FirstPassRows, reqs plan.PassRequests) (*Kernel, error) {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	draft, err := d.Model(name)
	if err != nil {
		t.Fatal(err)
	}
	c := typedRoleCase(t, "minimax-m3")
	in := c.in
	in.Model = withoutSpeculator(t, in.Model)
	in.Draft = draft
	eng := *in.Engine
	eng.Speculative = &enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, DraftModel: name}
	in.Engine = &eng
	return New(passesOf(in, rows, reqs))
}

func TestANamedEagle3DrafterWithUnstatedPassesIsRefused(t *testing.T) {
	for _, name := range []string{"minimax-m3-eagle3", "minimax-m3-eagle3-gqa"} {
		t.Run(name, func(t *testing.T) {
			_, err := namedEagle3(t, name, "", "")
			if err == nil || !strings.Contains(err.Error(), "draft_passes_unstated") ||
				!strings.Contains(err.Error(), name) || !strings.Contains(err.Error(), "DraftFirstPassRows") {
				t.Fatalf("err %v, want draft_passes_unstated naming %s and DraftFirstPassRows", err, name)
			}
			// One field is not enough.
			if _, err := namedEagle3(t, name, plan.FirstPassScheduled, ""); err == nil || !strings.Contains(err.Error(), "DraftPassRequests") {
				t.Errorf("err %v, want the missing DraftPassRequests named", err)
			}
			if _, err := namedEagle3(t, name, "", plan.PassRequestsAll); err == nil || !strings.Contains(err.Error(), "DraftFirstPassRows") {
				t.Errorf("err %v, want the missing DraftFirstPassRows named", err)
			}
		})
	}
}

func TestANamedEagle3DrafterWithStatedPassesIsAccepted(t *testing.T) {
	for _, c := range []struct {
		rows plan.FirstPassRows
		reqs plan.PassRequests
	}{
		{plan.FirstPassScheduled, plan.PassRequestsAll},
		{plan.FirstPassPerRequest, plan.PassRequestsSampling},
	} {
		if _, err := namedEagle3(t, "minimax-m3-eagle3", c.rows, c.reqs); err != nil {
			t.Fatalf("stated passes %q, %q refused: %v", c.rows, c.reqs, err)
		}
	}
}

func TestAnUnknownPassWordIsRefused(t *testing.T) {
	if _, err := namedEagle3(t, "minimax-m3-eagle3", "everything", plan.PassRequestsAll); err == nil || !strings.Contains(err.Error(), "draft_passes_unknown") {
		t.Errorf("err %v, want draft_passes_unknown", err)
	}
	if _, err := namedEagle3(t, "minimax-m3-eagle3", plan.FirstPassScheduled, "none"); err == nil || !strings.Contains(err.Error(), "draft_passes_unknown") {
		t.Errorf("err %v, want draft_passes_unknown", err)
	}
}

// The model graph's own autoregressive speculator (MiniMax-M3's MTP, several tokens) is refused
// the same way, and accepted once the plan states the passes.
func TestAnInGraphAutoregressiveSpeculatorWithUnstatedPassesIsRefused(t *testing.T) {
	c := typedRoleCase(t, "minimax-m3")
	if c.in.Model.Speculator == nil {
		t.Fatal("minimax-m3 states no speculator")
	}
	run := func(sp *enginesettings.Speculative, rows plan.FirstPassRows, reqs plan.PassRequests) error {
		in := c.in
		eng := *in.Engine
		eng.Speculative = sp
		in.Engine = &eng
		_, err := New(passesOf(in, rows, reqs))
		return err
	}
	for _, n := range []int{1, 3, 7} {
		sp := &enginesettings.Speculative{Method: "mtp", NumSpecTokens: n}
		err := run(sp, "", "")
		if err == nil || !strings.Contains(err.Error(), "draft_passes_unstated") {
			t.Errorf("K=%d: err %v, want draft_passes_unstated", n, err)
		}
		if err := run(sp, plan.FirstPassScheduled, plan.PassRequestsAll); err != nil {
			t.Errorf("K=%d: stated passes refused: %v", n, err)
		}
	}
	if err := run(nil, "", ""); err != nil {
		t.Errorf("no speculation was refused: %v", err)
	}
}

// The refusal names neither an engine nor a method.
func TestTheDraftPassesRefusalNamesNoEngineOrMethod(t *testing.T) {
	c := typedRoleCase(t, "minimax-m3")
	in := c.in
	eng := *in.Engine
	eng.Speculative = &enginesettings.Speculative{Method: "mtp", NumSpecTokens: 3}
	in.Engine = &eng
	_, err := New(in)
	if err == nil {
		t.Fatal("not refused")
	}
	for _, w := range []string{"vllm", "vLLM", "sglang", "mtp", "eagle"} {
		if strings.Contains(err.Error(), w) {
			t.Errorf("refusal %q names %q", err, w)
		}
	}
}

// A parallel-block drafter needs neither fact, and a deployment that does not speculate is
// unaffected whatever the plan says.
func TestAParallelBlockDrafterNeedsNoPassStatement(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	draft, err := d.Model("kimi-k3-dspark")
	if err != nil {
		t.Fatal(err)
	}
	c := typedRoleCase(t, "kimi-k3")
	in := c.in
	in.Draft = draft
	eng := *in.Engine
	eng.Speculative = &enginesettings.Speculative{Method: "dspark", NumSpecTokens: 7, DraftModel: "kimi-k3-dspark"}
	in.Engine = &eng
	if _, err := New(in); err != nil {
		t.Errorf("a parallel-block drafter with no pass statement was refused: %v", err)
	}
}

// The facts are the plan's, not the engine settings': the settings type has no such field, so a
// cost model that read them from settings would have nothing to read (X2, SUB-6).
func TestTheEngineSettingsCarryNoDraftPassFacts(t *testing.T) {
	for _, typ := range []reflect.Type{reflect.TypeOf(enginesettings.Settings{}), reflect.TypeOf(enginesettings.Speculative{})} {
		for i := 0; i < typ.NumField(); i++ {
			if n := typ.Field(i).Name; strings.Contains(n, "FirstPass") || strings.Contains(n, "PassRequests") {
				t.Errorf("%s.%s: the draft passes are stated in the pool plan (plan.DraftPolicy), not in the engine settings", typ.Name(), n)
			}
		}
	}
}

// withoutSpeculator returns a copy of g with its speculator and the layer kinds only the
// speculator ran removed.
func withoutSpeculator(t *testing.T, g *model.Graph) *model.Graph {
	t.Helper()
	b, err := yaml.Marshal(*g)
	if err != nil {
		t.Fatal(err)
	}
	var c model.Graph
	if err := yaml.Unmarshal(b, &c); err != nil {
		t.Fatal(err)
	}
	used := map[string]bool{}
	for _, id := range c.Stack.Expand() {
		used[id] = true
	}
	c.Speculator = nil
	if len(c.Embedding) == 0 {
		// An EAGLE3 head aliases the target's token embedding; the MiniMax-M3 graphs state none
		// in this stack, so the copy is given the one vLLM builds, to let the pair validate.
		var gather model.Node
		node := "op: Gather\nrole: embed_tokens\nrows: " + strconv.Itoa(c.Global.VocabSize) +
			"\nrow_bytes: " + strconv.Itoa(2*c.Global.HiddenSize) +
			"\nrows_per_token: 1\nout_bytes_per_token: " + strconv.Itoa(2*c.Global.HiddenSize) +
			"\nshard: rows\n"
		if err := yaml.Unmarshal([]byte(node), &gather); err != nil {
			t.Fatal(err)
		}
		c.Embedding = []model.Node{gather}
	}
	kinds := c.LayerKinds[:0]
	for _, lk := range c.LayerKinds {
		if used[lk.ID] {
			kinds = append(kinds, lk)
		}
	}
	c.LayerKinds = kinds
	return &c
}
