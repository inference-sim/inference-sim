package kernelmodel

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/inference-sim/inference-sim/substrate/contracts/spec/model"
)

// openTP opens a catalog model at tensor parallel size tp on one H200 node.
func openTP(t *testing.T, model string, tp int) (*Model, error) {
	t.Helper()
	body := strings.Join([]string{
		"kind: Scenario",
		"name: " + model + "-tp",
		"engine: {kind: vllm}",
		"model: " + model,
		"coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]",
		"cluster:",
		"  hardware: h200",
		"  fabric: ib-400g",
		"  nodes: 1",
		"  gpus_per_node: 8",
		"---",
		"kind: Deployment",
		"name: " + model + "-tp",
		"pools:",
		"  - role: colocated",
		"    nodes: 1",
		"    parallel: {tp: " + strconv.Itoa(tp) + ", pp: 1, dp: 1}",
		"    engine:",
		"      block_size: 16",
		"      max_num_seqs: 64",
		"      max_num_batched_tokens: 8192",
		"      max_model_len: 32768",
		"      enable_prefix_caching: true",
		"      gpu_memory_utilization: 0.9",
		"      async_scheduling: true",
		"",
	}, "\n")
	r := repos()
	r.Scenarios = t.TempDir()
	if err := os.WriteFile(filepath.Join(r.Scenarios, "v.yaml"), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	return Open("v.yaml", r)
}

// MiMo-V2-Flash's layers are the engine's two specs in model order: the 9 full-attention layers
// (4 KV heads, 2 a rank at tp 2) and the 39 sliding-window layers (8 KV heads, 4 a rank, a
// 128-token window), each with a value head of 128 against a key head of 192, so the page is
// kv heads x block x (192 + 128) x dtype, vllm/v1/kv_cache_interface.py:511-604@0fda86a35419.
// The runs state that page's content, 640 B a head and token in bf16 (320 x 2), not the
// symmetric 2 x 192 x 2 = 768 B the elements would give.
func TestMiMoKVLayersStateTheNarrowValueHeadsPage(t *testing.T) {
	m, err := openTP(t, "mimo-v2-flash", 2)
	if err != nil {
		t.Fatal(err)
	}
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	var full, swa int
	for _, r := range runs {
		if r.StateBytes != 640 {
			t.Errorf("a %s run of %d layers states %d B a head, want 640 (320 elements of bf16)", r.Spec, r.Count, r.StateBytes)
		}
		switch r.Spec {
		case "full":
			full += r.Count
			if r.NumKVHeads != 2 {
				t.Errorf("full run: %d KV heads a rank, want 2", r.NumKVHeads)
			}
		case "sliding_window":
			swa += r.Count
			if r.SlidingWindow != 128 || r.NumKVHeads != 4 {
				t.Errorf("sliding run: window %d, %d KV heads a rank, want 128 and 4", r.SlidingWindow, r.NumKVHeads)
			}
		default:
			t.Errorf("spec %q", r.Spec)
		}
	}
	if full != 9 || swa != 39 {
		t.Errorf("%d full and %d sliding layers, want 9 and 39", full, swa)
	}
}

// A model whose layers state no cache and have the symmetric entry (HY3: keys and values of
// head_dim elements) is one full-attention group the engine sizes from head_size: no runs.
func TestSymmetricModelHasNoRuns(t *testing.T) {
	m, err := openTP(t, "hy3-fp8", 8)
	if err != nil {
		t.Fatal(err)
	}
	runs, err := m.KVLayers()
	if err != nil {
		t.Fatal(err)
	}
	if runs != nil {
		t.Errorf("runs %+v, want nil", runs)
	}
}

// statedHeadBytes is a layer's stated page content a KV head and token, or 0 when the entry is the
// symmetric one the engine infers from head_size (or one this does not size).
func TestStatedHeadBytes(t *testing.T) {
	gqa := func(kind model.AttentionKind, kv, dh int) model.Node {
		return model.Node{AttentionKind: kind, NumKVHeads: kv, HeadDim: dh}
	}
	entry := func(b int) model.Cache { return model.Cache{Kind: model.CacheKV, Owned: true, BytesPerEntry: b} }
	for _, c := range []struct {
		name  string
		n     model.Node
		c     model.Cache
		dtype string
		want  int
	}{
		{"narrow value head", gqa(model.AttentionGQA, 4, 192), entry(4 * (192 + 128) * 2), "bfloat16", 640},
		{"narrow value head in a window layer", gqa(model.AttentionSWA, 8, 192), entry(8 * (192 + 128) * 2), "bfloat16", 640},
		{"symmetric entry", gqa(model.AttentionGQA, 8, 128), entry(2 * 8 * 128 * 2), "bfloat16", 0},
		{"symmetric entry of an fp8 cache", gqa(model.AttentionGQA, 8, 128), entry(2 * 8 * 128), "fp8", 0},
		{"entry not whole bytes a head", gqa(model.AttentionGQA, 3, 128), entry(1000), "bfloat16", 0},
		{"several tokens an entry", gqa(model.AttentionGQA, 4, 192), model.Cache{Kind: model.CacheKV, Owned: true, BytesPerEntry: 4 * 320 * 2, TokensPerEntry: 2}, "bfloat16", 0},
		{"a cache dtype with no plain width", gqa(model.AttentionGQA, 4, 192), entry(4 * 320 * 2), "fp8_ds_mla", 0},
		{"not a grouped-query node", gqa(model.AttentionMLA, 4, 192), entry(4 * 320 * 2), "bfloat16", 0},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := statedHeadBytes(c.n, c.c, c.dtype, nil); got != c.want {
				t.Errorf("statedHeadBytes = %d, want %d", got, c.want)
			}
		})
	}
}
