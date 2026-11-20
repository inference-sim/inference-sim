package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"gopkg.in/yaml.v3"
)

// withoutRole returns a copy of the case's graph with every node of the given role removed
// from the named layer kind, the kind's edges re-chained over what is left: the graph as the
// deriver emitted it before the recurrent input projections were nodes.
func withoutRole(t *testing.T, g *model.Graph, kind, role string) (*model.Graph, model.Node, int) {
	t.Helper()
	b, err := yaml.Marshal(*g)
	if err != nil {
		t.Fatal(err)
	}
	var out model.Graph
	if err := yaml.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	var dropped model.Node
	found := 0
	for i := range out.LayerKinds {
		lk := &out.LayerKinds[i]
		if lk.ID != kind {
			continue
		}
		kept := lk.Nodes[:0]
		for _, n := range lk.Nodes {
			if n.Role == role {
				dropped = n
				found++
				continue
			}
			kept = append(kept, n)
		}
		lk.Nodes = kept
		lk.Edges = nil
		for j := 1; j < len(kept); j++ {
			lk.Edges = append(lk.Edges, [2]int{j - 1, j})
		}
	}
	if found != 1 {
		t.Fatalf("layer kind %q has %d nodes of role %q; the test needs exactly one", kind, found, role)
	}
	layers := 0
	for _, id := range out.Stack.Expand() {
		if id == kind {
			layers++
		}
	}
	return &out, dropped, layers
}

// A Mamba-2 layer's input projection is an ordinary GEMM node, so the cost model needs nothing
// new to price it: the catalog's Nemotron graph with its recurrent_in_proj node holds exactly
// that node's column-split weight more per rank, over every Mamba layer, and its decode and
// prefill steps take longer than the same graph without it (the graph before the node existed).
func TestMambaInputProjectionAddsItsWeightAndStepTime(t *testing.T) {
	c := typedRoleCase(t, "nemotron-3.5-lightning-30b-a3b-bf16")
	old, node, layers := withoutRole(t, c.in.Model, "mamba", "recurrent_in_proj")
	if node.Op != model.OpGEMM || node.TP != model.TPColumn {
		t.Fatalf("recurrent_in_proj is %s with tp %q; want a column-split GEMM", node.Op, node.TP)
	}
	pool := c.in.Deployment.Pools[c.in.PoolIndex].Parallel
	if pool.PP > 1 {
		t.Fatalf("the case runs pp %d; the weight figure below is for one stage holding every layer", pool.PP)
	}
	withNode, err := New(c.in)
	if err != nil {
		t.Fatal(err)
	}
	in := c.in
	in.Model = old
	without, err := New(in)
	if err != nil {
		t.Fatal(err)
	}

	// bf16: two bytes a parameter, n split over tp ranks.
	want := int64(node.N) * int64(node.K) * 2 * int64(layers) / int64(pool.TP)
	got := withNode.FixedBytes().Weights - without.FixedBytes().Weights
	if got != want {
		t.Errorf("resident weights rose by %d bytes per rank; want %d (%d x %d bf16 over %d layers at tp %d)",
			got, want, node.N, node.K, layers, pool.TP)
	}

	decode := kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096}
	prefill := kernel.ReqShape{Scheduled: 2048, PromptLen: 2048}
	for name, reqs := range map[string][]kernel.ReqShape{
		"decode x32": {decode, decode, decode, decode, decode, decode, decode, decode,
			decode, decode, decode, decode, decode, decode, decode, decode,
			decode, decode, decode, decode, decode, decode, decode, decode,
			decode, decode, decode, decode, decode, decode, decode, decode},
		"prefill 2048": {prefill},
	} {
		b := kernel.Batch{DecodeThreshold: 1, Reqs: reqs}
		with, base := withNode.StepTime(b).NoOverlap, without.StepTime(b).NoOverlap
		if with <= base {
			t.Errorf("%s: the step takes %d with the input projection and %d without; the GEMM added no time",
				name, with, base)
		}
	}
}
