package price

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"gopkg.in/yaml.v3"
)

// kimiK31Plan plans the committed Kimi-K3.1 graph and returns its DSA layer kind.
func kimiK31Plan(t *testing.T) PlannedLayer {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "..", "..", "catalog", "models", "kimi-k3.1", "graph.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	var g model.Graph
	if err := yaml.Unmarshal(b, &g); err != nil {
		t.Fatal(err)
	}
	p, err := BuildPlan(&g, emitAll{}, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	for _, l := range p.Layers {
		if l.ID == "dsa_moe" {
			return l
		}
	}
	t.Fatal("no dsa_moe layer kind in the Kimi-K3.1 plan")
	return PlannedLayer{}
}

// vLLM builds the K3.1 DSA layer's A-projection (fused_qkv_a_proj), the indexer's query
// projection (wq_index) and its key and weights projection (wkv_index) replicated: every rank
// holds and runs each whole (KimiK3DSAAttention and KimiK3DSAIndexer, vllm/models/kimi_k3/
// nvidia/dsa.py:150-166, 469-475 at v0.34.0rc0). The plan must say so: left to inference it
// shards the A-projection and the key projection by output width (k equals the hidden size)
// and the query projection by reduction, and a step divides both their work and their weight
// reads by tp, pricing a tp-th of matrices every rank runs whole.
//
// The packed forward (pack_projections, :688-734) launches [q_a | kv_a | wk | weights_proj] as
// one GEMM, the output gate as its own, and [q_b | wq_index] as one. The plan states the first
// as one replicated GEMM of 2112 + 160 rows and cannot state the third as one (a column-parallel
// part beside a replicated one), so wq_index is its own launch. The gate is column-parallel and
// must not fuse into the replicated A-projection.
func TestKimiK31AttentionFrontEndIsPlannedAsTheEnginePacksIt(t *testing.T) {
	l := kimiK31Plan(t)
	type shape struct {
		n, k       int
		replicated bool
	}
	want := []shape{
		{2272, 7168, true},   // [q_a | kv_a | wk | weights_proj], one launch
		{12288, 7168, false}, // g_proj, column-parallel, its own launch
		{18432, 1536, false}, // q_b_proj
		{4096, 1536, true},   // wq_index, replicated; the engine packs it with q_b, the plan cannot
		{24576, 512, false},  // kv_b_proj
		{7168, 12288, false}, // o_proj
	}
	if len(l.DenseGEMMs) < len(want) {
		t.Fatalf("the DSA layer plans %d dense GEMMs, want at least %d", len(l.DenseGEMMs), len(want))
	}
	for i, w := range want {
		g := l.DenseGEMMs[i]
		if g.N != w.n || g.K != w.k || g.Replicated != w.replicated {
			t.Errorf("dense GEMM %d is n=%d k=%d replicated=%v, want n=%d k=%d replicated=%v",
				i, g.N, g.K, g.Replicated, w.n, w.k, w.replicated)
		}
		if g.Replicated && g.ShardN {
			t.Errorf("dense GEMM %d is replicated and also shards its output", i)
		}
		if g.Replicated && g.WeightBytes != float64(g.N*g.K)*2 {
			t.Errorf("dense GEMM %d holds %v bytes, want %d (bf16, whole)", i, g.WeightBytes, g.N*g.K*2)
		}
	}
}

// The replicated dense weights of the layer are exactly the A-projection (with wk and
// weights_proj) and wq_index: no other projection of the layer is replicated, and these two
// are, whatever else changes.
func TestKimiK31ReplicatedProjectionsAreExactlyTheOnesVLLMReplicates(t *testing.T) {
	l := kimiK31Plan(t)
	var rows float64
	for _, g := range l.DenseGEMMs {
		if g.Replicated {
			rows += float64(g.N * g.K)
		}
	}
	if want := float64(2272*7168 + 4096*1536); rows != want {
		t.Errorf("replicated dense weights are %v, want %v (A-projection with wk and weights_proj, and wq_index)",
			rows, want)
	}
}
