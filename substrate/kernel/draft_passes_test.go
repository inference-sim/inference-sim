package latencykernel

import (
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"
)

// An autoregressive drafter's first pass runs its layers over a row count the engine decides
// (one per request, or every token the target scheduled), and the head and later passes run for
// the requests the engine decides. The catalog graph states only the fc projection of that pass,
// and this layer does not price the rest. So every deployment that speculates through an
// autoregressive drafter, a named EAGLE3 head or the graph's own MTP, is refused with
// draft_first_pass_unpriced rather than priced one row per request and left short. The layer
// that prices the pass lifts the refusal in the commit that prices it. A parallel-block drafter
// (DSpark) states its target-wide work as its context stack and is not refused.

// namedEagle3 builds the catalog pair of a target with no speculator of its own and a named
// EAGLE3 head, with the engine's speculative block as given.
func namedEagle3(t *testing.T, name string, sp *enginesettings.Speculative) (*Kernel, error) {
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
	eng.Speculative = sp
	in.Engine = &eng
	return New(in)
}

func TestANamedEagle3DrafterIsRefusedUntilItsFirstPassIsPriced(t *testing.T) {
	for _, name := range []string{"minimax-m3-eagle3", "minimax-m3-eagle3-gqa"} {
		t.Run(name, func(t *testing.T) {
			sp := enginesettings.Speculative{Method: "eagle3", NumSpecTokens: 3, DraftModel: name}
			_, err := namedEagle3(t, name, &sp)
			if err == nil || !strings.Contains(err.Error(), "draft_first_pass_unpriced") ||
				!strings.Contains(err.Error(), name) {
				t.Fatalf("err %v, want draft_first_pass_unpriced naming %s", err, name)
			}
		})
	}
}

// The model graph's own autoregressive speculator (MiniMax-M3's MTP, several tokens) is refused
// the same way, at every draft length; no speculation is not.
func TestAnInGraphAutoregressiveSpeculatorIsRefusedUntilItsFirstPassIsPriced(t *testing.T) {
	c := typedRoleCase(t, "minimax-m3")
	if c.in.Model.Speculator == nil {
		t.Fatal("minimax-m3 states no speculator")
	}
	run := func(sp *enginesettings.Speculative) error {
		in := c.in
		eng := *in.Engine
		eng.Speculative = sp
		in.Engine = &eng
		_, err := New(in)
		return err
	}
	for _, n := range []int{1, 3, 7} {
		err := run(&enginesettings.Speculative{Method: "mtp", NumSpecTokens: n})
		if err == nil || !strings.Contains(err.Error(), "draft_first_pass_unpriced") {
			t.Errorf("K=%d: err %v, want draft_first_pass_unpriced", n, err)
		}
	}
	if err := run(nil); err != nil {
		t.Errorf("no speculation was refused: %v", err)
	}
}

// The refusal names neither an engine nor a method.
func TestTheFirstPassRefusalNamesNoEngineOrMethod(t *testing.T) {
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

// A parallel-block drafter is not refused: its target-wide work is its context stack.
func TestAParallelBlockDrafterIsNotRefusedForItsFirstPass(t *testing.T) {
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
		t.Errorf("a parallel-block drafter was refused: %v", err)
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
