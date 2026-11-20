// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/models/glm5next/common/attention.py at vLLM v0.34.0+0fda86a35419 (0fda86a35419c9c70dbfada224c838c10fbb62ad).

package kernelmodel

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	"github.com/inference-sim/inference-sim/engines/vllm/core/vllm"
)

// glm5NextScenario is a one-pool vLLM scenario for a GLM-5.3-Flash catalog model on a chip at tp;
// kvDType "" leaves the cache dtype unstated and numSpec above 0 turns MTP on.
func glm5NextScenario(t *testing.T, modelName, chip string, tp int, kvDType string, numSpec int) (string, Repos) {
	t.Helper()
	extra := ""
	if kvDType != "" {
		extra += "\n      kv_cache_dtype: " + kvDType
	}
	if numSpec > 0 {
		extra += "\n      speculative_config: {method: mtp, num_speculative_tokens: " + strconv.Itoa(numSpec) + "}"
	}
	nodes := strconv.Itoa((tp + 7) / 8)
	src := `kind: Scenario
name: s
engine: {kind: vllm}
model: ` + modelName + `
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
cluster: {hardware: ` + chip + `, fabric: ib-400g, nodes: ` + nodes + `, gpus_per_node: 8}
---
kind: Deployment
name: s
pools:
  - role: colocated
    nodes: ` + nodes + `
    parallel: {tp: ` + strconv.Itoa(tp) + `, pp: 1, dp: 1, enable_expert_parallel: false}
    engine:
      block_size: 16
      max_num_batched_tokens: 8192
      max_num_seqs: 256
      max_model_len: 4096
      gpu_memory_utilization: 0.9
      compilation_config: {cudagraph_mode: PIECEWISE}` + extra + "\n"
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "s.yaml"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	r := repos()
	r.Scenarios = dir
	return "s.yaml", r
}

// GLM-5.3-Flash's sparse-MLA layer beside its recurrent (KDA) layers registers three caches, in
// the order its modules construct them (Indexer, then MultiHeadLatentAttentionWrapper,
// glm5next/common/attention.py:298-316, :620-637): the indexer's key cache, an MLAAttentionSpec
// of one 132-byte uint8 state per index_kpool tokens (Glm5NextIndexerCache, :93-140); the
// kpool tail ring, a CircularBufferSpec of two bf16 head slots of the indexer's key width
// (Glm5NextTailCache, :142-200); and the main MLA latent, one state a token
// (MLAAttentionSpec, mla_attention.py:1637-1672). It was refused as "sparse_mla attention beside
// recurrent layers is not mapped" (blis-vllm#308). Laws, over the chips and the cache dtypes:
// the model opens; per sparse-MLA layer of the graph there is one run of each of the three
// caches (per stack position, the drafter's marked as the drafter's); every cache states its
// element size, the main latent's following the cache dtype that runs (SM10x resolves an unstated
// dtype to fp8, Glm5NextModelConfig, models/config.py:209-224); the block size the hybrid
// alignment leaves is a multiple of 128 (an MLA model's floor, interface.py:1130-1138); and the
// pool's bytes a block do not follow the memory model's page.
func TestGlm5NextSparseMLABesideRecurrentLayers(t *testing.T) {
	for _, c := range []struct {
		chip    string
		tp      int
		kvDType string
		numSpec int
		elem    int
	}{
		{"h200", 8, "", 0, 2},
		{"h200", 4, "fp8", 0, 1},
		{"b200", 8, "", 0, 1},
		{"b200", 8, "bfloat16", 0, 2},
		{"h200", 8, "", 1, 2},
		{"b200", 4, "", 3, 1},
	} {
		name := c.chip + "/tp" + strconv.Itoa(c.tp) + "/" + c.kvDType + "/spec" + strconv.Itoa(c.numSpec)
		t.Run(name, func(t *testing.T) {
			scen, r := glm5NextScenario(t, "glm-5.3-flash", c.chip, c.tp, c.kvDType, c.numSpec)
			m, err := Open(scen, r)
			if err != nil {
				t.Fatal(err)
			}
			runs, err := m.KVLayers()
			if err != nil {
				t.Fatal(err)
			}
			sparse := 0
			for _, id := range m.graph.Stack.Expand() {
				for _, lk := range m.graph.LayerKinds {
					if lk.ID != id {
						continue
					}
					for _, n := range lk.Nodes {
						if n.ReadsEngineKV() && n.AttentionKind == "sparse_mla" {
							sparse++
						}
					}
				}
			}
			if sparse == 0 {
				t.Fatal("the graph has no sparse_mla layer")
			}
			ring := 4 * nextPow2(cdiv(4+c.numSpec, 4))
			var main, index, tail, drafts [2]int // [target, draft]
			vr := make([]vllm.KVLayerRun, len(runs))
			for i, run := range runs {
				vr[i] = vllm.KVLayerRun(run)
				if run.Spec == "mamba" {
					continue
				}
				if run.ElemBytes == 0 {
					t.Errorf("a %s run states no element size: %+v", run.Spec, run)
				}
				d := 0
				if run.Draft {
					d = 1
					drafts[1] += run.Count
				}
				switch {
				case run.Spec == "mla" && run.TokensPerState == 1:
					main[d] += run.Count
					if run.NumKVHeads != 1 || run.HeadSize != 512 || run.ElemBytes != c.elem {
						t.Errorf("main latent %+v, want 1 head of 512 x %d B", run, c.elem)
					}
				case run.Spec == "mla":
					index[d] += run.Count
					if run.TokensPerState != 4 || run.NumKVHeads != 1 || run.HeadSize != 132 || run.ElemBytes != 1 {
						t.Errorf("indexer cache %+v, want 1 head of 132 x 1 B at 4 tokens a state", run)
					}
				case run.Spec == "circular_buffer":
					tail[d] += run.Count
					if run.NumKVHeads != 2 || run.HeadSize != 128 || run.ElemBytes != 2 || run.BlockSize != ring {
						t.Errorf("tail ring %+v, want 2 heads of 128 x 2 B, %d rows", run, ring)
					}
				default:
					t.Errorf("unexpected run %+v", run)
				}
			}
			want := [2]int{sparse, 0}
			if c.numSpec > 0 {
				want[1] = 1
			}
			for what, got := range map[string][2]int{"main": main, "index": index, "tail": tail} {
				if got != want {
					t.Errorf("%s caches [target draft] = %v, want %v", what, got, want)
				}
			}
			if m.engine.BlockSize%128 != 0 {
				t.Errorf("block size %d is not a multiple of 128", m.engine.BlockSize)
			}
			var per [2]int64
			for i, page := range []int64{1 << 20, 3<<20 + 7} {
				if per[i], err = vllm.KVPoolBytesPerBlock(vr, false, m.engine.BlockSize, page); err != nil {
					t.Fatalf("memory model page %d: %v", page, err)
				}
			}
			if per[0] != per[1] || per[0] <= 0 {
				t.Errorf("the pool's bytes a block follow the memory model's page: %d and %d", per[0], per[1])
			}
		})
	}
}

// The tail ring must divide the attention block (Glm5NextTailCache asserts it, attention.py:174-181):
// a draft length whose ring outgrows the block is refused by name, not sized.
func TestGlm5NextRefusesARingThatDoesNotDivideTheBlock(t *testing.T) {
	scen, r := glm5NextScenario(t, "glm-5.3-flash", "h200", 8, "", 1)
	m, err := Open(scen, r)
	if err != nil {
		t.Fatal(err)
	}
	e := m.engine
	e.BlockSize = 128
	e.Speculative = &engineconfig.Speculative{Method: "mtp", NumSpecTokens: 200}
	var lk model.LayerKind
	var main model.Node
	for _, k := range m.graph.LayerKinds {
		for _, n := range k.Nodes {
			if n.ReadsEngineKV() && n.AttentionKind == model.AttentionSparseMLA {
				lk, main = k, n
			}
		}
	}
	if _, err := glm5NextLayerRuns(m.graph, lk.ID, lk, main, e, false); err == nil || !strings.Contains(err.Error(), "tail ring 256") {
		t.Fatalf("a 256-row ring over a 128-token block: %v", err)
	}
	e.Speculative.NumSpecTokens = 1
	if _, err := glm5NextLayerRuns(m.graph, lk.ID, lk, main, e, false); err != nil {
		t.Fatalf("a 4-row ring over a 128-token block: %v", err)
	}
}

func cdiv(a, b int) int { return (a + b - 1) / b }

func nextPow2(n int) int {
	p := 1
	for p < n {
		p *= 2
	}
	return p
}
