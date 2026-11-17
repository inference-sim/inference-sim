package latencykernel

import (
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
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
	in = withPlan(in, func(l *plan.Lowered) { l.Drafter = draft.Name })
	in = drafting(in, 3)
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
	run := func(n int, rows plan.FirstPassRows, reqs plan.PassRequests) error {
		_, err := New(passesOf(drafting(c.in, n), rows, reqs))
		return err
	}
	for _, n := range []int{1, 3, 7} {
		sp := n
		err := run(sp, "", "")
		if err == nil || !strings.Contains(err.Error(), "draft_passes_unstated") {
			t.Errorf("K=%d: err %v, want draft_passes_unstated", n, err)
		}
		if err := run(sp, plan.FirstPassScheduled, plan.PassRequestsAll); err != nil {
			t.Errorf("K=%d: stated passes refused: %v", n, err)
		}
	}
	if err := run(0, "", ""); err != nil {
		t.Errorf("no speculation was refused: %v", err)
	}
}

// The refusal names neither an engine nor a method.
func TestTheDraftPassesRefusalNamesNoEngineOrMethod(t *testing.T) {
	c := typedRoleCase(t, "minimax-m3")
	in := c.in
	in = drafting(in, 3)
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

// A parallel-block drafter states no first pass, but it states for which requests its query pass
// and sequential head run: without it the deployment is refused, and with it alone it builds.
func TestAParallelBlockDrafterStatesOnlyForWhichRequestsItProposes(t *testing.T) {
	in := parallelBlockCase(t)
	in.DraftFirstPassRows, in.DraftPassRequests = "", ""
	if _, err := New(in); err == nil || !strings.Contains(err.Error(), "draft_passes_unstated") ||
		!strings.Contains(err.Error(), "DraftPassRequests") {
		t.Errorf("a parallel-block drafter stating no pass requests: %v, want draft_passes_unstated naming DraftPassRequests", err)
	}
	in.DraftPassRequests = plan.PassRequestsAll
	if _, err := New(in); err != nil {
		t.Errorf("a parallel-block drafter stating its pass requests and no first pass was refused: %v", err)
	}
}

// A parallel-block drafter whose engine runs its query pass for every request (vLLM's DFlash and
// DSpark, num_reqs * num_query_per_req tokens, dflash/speculator.py:463-467@4ba787ec9959e566d3b587a796dff07673324da5)
// pays for it over a request still in the middle of its prompt; one that runs it for the
// sampling requests does not. A step of decodes alone costs the same under both.
func TestAParallelBlockDrafterProposesForTheRequestsTheEngineStates(t *testing.T) {
	in := parallelBlockCase(t)
	price := func(r plan.PassRequests, reqs ...kernel.ReqShape) float64 {
		c := in
		c.DraftPassRequests = r
		return mustKernel(t, c).StepTime(kernel.Batch{DecodeThreshold: 8, Reqs: reqs}).Overlap.Seconds()
	}
	chunk := kernel.ReqShape{Scheduled: 2048, PromptLen: 8192}
	decode := kernel.ReqShape{Scheduled: 8, Computed: 4088, PromptLen: 4096}
	if all, sampling := price(plan.PassRequestsAll, chunk, decode), price(plan.PassRequestsSampling, chunk, decode); all <= sampling {
		t.Errorf("a chunk mid-prompt beside a decode: %.9g s proposing for every request, %.9g s for the sampling ones; "+
			"the chunk's query pass must be charged", all, sampling)
	}
	if all, sampling := price(plan.PassRequestsAll, decode, decode), price(plan.PassRequestsSampling, decode, decode); all != sampling {
		t.Errorf("two decodes: %.9g s proposing for every request, %.9g s for the sampling ones; they must match", all, sampling)
	}
}

// parallelBlockCase is kimi-k3 drafting 7 tokens through its DSpark drafter.
func parallelBlockCase(t *testing.T) Inputs {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	draft, err := d.Model("kimi-k3-dspark")
	if err != nil {
		t.Fatal(err)
	}
	in := typedRoleCase(t, "kimi-k3").in
	in.Draft = draft
	in = withPlan(in, func(l *plan.Lowered) { l.Drafter = draft.Name })
	return drafting(in, 7)
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
