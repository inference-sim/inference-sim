package latencykernel

import (
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
	"gopkg.in/yaml.v3"
)

// Muse Spark (vLLM v0.33.0) builds a RoPE cos/sin table in every layer that rotates:
// MuseSpark1xAttention constructs its own RotaryEmbedding rather than sharing one, 1,048,576 positions x 64
// fp32 = 256 MiB, replicated on every rank, 46 of the 62 layers, and the engine counts them in the weights it
// logs at start-up (inference-sim/blis-substrate#378). On GB200 at TP8 with expert parallelism that line reads
// 81.00 GiB a GPU (internal deployment measurements); the graph without the tables priced 69.30 (68.78 in this
// fixture, which carries no vision tower). The tables are resident memory and no step work: the QK-norm and RoPE
// kernel that reads them is a node the graph already counts. So the weights of a rank hold exactly the 46 tables
// at every tensor-parallel width, the engine's figure is met within a percent and a half, and no step is priced
// differently.
func TestMuseSparkHoldsTheEnginesRoPETablesAsMemoryOnly(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const role = "rope_cos_sin_cache"
	open := func(parallel string, strip bool) *Kernel {
		in := planCase(t, d, "muse-spark-open-816b-a42b-nvfp4", "gb200-nvl72", 2, parallel, nil)
		if strip {
			in.Model = withoutRoleInAllKinds(t, in.Model, role)
		}
		return mustKernel(t, in)
	}
	decode := kernel.Batch{DecodeThreshold: 1}
	for range 16 {
		decode.Reqs = append(decode.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 2048, PromptLen: 1024})
	}
	prefill := kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{{Scheduled: 4096, PromptLen: 4096}}}
	const tables = 46 * 1048576 * 64 * 4
	for _, parallel := range []string{"tp: 8, pp: 1, dp: 1, enable_expert_parallel: true", "tp: 4, pp: 1, dp: 1, enable_expert_parallel: true"} {
		with, without := open(parallel, false), open(parallel, true)
		if got := with.FixedBytes().Weights - without.FixedBytes().Weights; got != tables {
			t.Errorf("%s: the tables add %d bytes to a rank's weights, want the engine's 46 x 256 MiB = %d", parallel, got, int64(tables))
		}
		for name, b := range map[string]kernel.Batch{"decode": decode, "prefill": prefill} {
			if w, wo := with.StepTime(b).NoOverlap, without.StepTime(b).NoOverlap; w != wo {
				t.Errorf("%s: a %s step costs %v with the tables and %v without; they are memory, not step work", parallel, name, w, wo)
			}
		}
	}
	const engineTP8 = 81.00
	tp8 := float64(open("tp: 8, pp: 1, dp: 1, enable_expert_parallel: true", false).FixedBytes().Weights) / (1 << 30)
	t.Logf("weights a GPU at tp8+ep: %.2f GiB", tp8)
	if tp8 < engineTP8*0.985 || tp8 > engineTP8*1.015 {
		t.Errorf("tp8+ep weights %.2f GiB; the engine logs %.2f", tp8, engineTP8)
	}
}

// withoutRoleInAllKinds is a copy of g with every node of the role removed from every layer kind, each
// kind's edges re-chained over what is left (the graphs here are chains), so a graph can be priced as if the
// engine held no such table.
func withoutRoleInAllKinds(t *testing.T, g *model.Graph, role string) *model.Graph {
	t.Helper()
	b, err := yaml.Marshal(*g)
	if err != nil {
		t.Fatal(err)
	}
	var out model.Graph
	if err := yaml.Unmarshal(b, &out); err != nil {
		t.Fatal(err)
	}
	for i := range out.LayerKinds {
		lk := &out.LayerKinds[i]
		kept := lk.Nodes[:0]
		for _, n := range lk.Nodes {
			if n.Role != role {
				kept = append(kept, n)
			}
		}
		lk.Nodes, lk.Edges = kept, nil
		for j := 1; j < len(kept); j++ {
			lk.Edges = append(lk.Edges, [2]int{j - 1, j})
		}
	}
	return &out
}
