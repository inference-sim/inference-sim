// Original to blis-vllm.

package kv

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

// ocLayer, ocCase: the oracle's kv_pool cases (oracle/lower/extract.py kv_pool_cases).
type ocLayer struct {
	Count          int    `json:"count"`
	Spec           string `json:"spec"`
	SlidingWindow  int    `json:"sliding_window"`
	NumKVHeads     int    `json:"num_kv_heads"`
	HeadSize       int    `json:"head_size"`
	BlockSize      int    `json:"block_size"`
	TokensPerState int    `json:"tokens_per_state"`
	StateBytes     int    `json:"state_bytes"`
	ElemBytes      int    `json:"elem_bytes"`
	Alignment      int    `json:"alignment"`
	BoundedReplay  bool   `json:"bounded_replay"`
	PageSizeBytes  int64  `json:"page_size_bytes"`
	StrideAlign    int64  `json:"stride_alignment"`
}

type ocCase struct {
	Model          string           `json:"model"`
	TP             int              `json:"tp"`
	BlockSize      int              `json:"block_size"`
	Backend        string           `json:"backend"`
	Runs           []ocLayer        `json:"runs"`
	Layout         string           `json:"layout"`
	Groups         []int            `json:"groups"`
	GroupPages     [][][3]int64     `json:"group_pages"`
	BytesPerBlock  map[string]int64 `json:"bytes_per_block"`
	PerLayerEquals bool             `json:"per_layer_equals_once"`
}

func loadPoolOracle(t *testing.T) []ocCase {
	t.Helper()
	raw, err := os.ReadFile("../lower/testdata/vllm_lower.json")
	if err != nil {
		t.Fatal(err)
	}
	var f struct {
		KVPool []ocCase `json:"kv_pool"`
	}
	if err := json.Unmarshal(raw, &f); err != nil {
		t.Fatal(err)
	}
	if len(f.KVPool) < 300 {
		t.Fatalf("%d KV pool cases in the oracle file, want its full sweep", len(f.KVPool))
	}
	return f.KVPool
}

// layersOfCase is the case's layers in model order, as the engine block states them.
func layersOfCase(t *testing.T, c ocCase) []Layer {
	t.Helper()
	var out []Layer
	for _, r := range c.Runs {
		spec, err := ParseLayerSpec(r.Spec)
		if err != nil {
			t.Fatal(err)
		}
		for range r.Count {
			out = append(out, Layer{Spec: spec, Window: r.SlidingWindow, KVHeads: r.NumKVHeads, HeadSize: r.HeadSize,
				BlockSize: r.BlockSize, TokensPerState: r.TokensPerState, StateBytes: r.StateBytes,
				ElemBytes: r.ElemBytes, Alignment: r.Alignment, BoundedReplay: r.BoundedReplay,
				StrideAlignment: int(r.StrideAlign)})
		}
	}
	return out
}

// The KV pool's bytes per block, at a cache stored as fp8, is the figure vLLM's own
// _get_kv_cache_bytes_per_block gives for the layers each stated architecture builds
// (vllm/v1/core/kv_cache_utils.py:1515-1552@5df63a72). The expected numbers are not written here:
// oracle/lower/extract.py builds vLLM's spec classes for every layer, applies the kernel block
// sizes the attention backends declare through vLLM's customize_attention_spec
// (vllm/v1/worker/utils.py:476-486), groups them with get_kv_cache_groups and prints each spec's
// page_size_bytes and get_block_stride_alignment(), and the bytes per block under both layouts:
// layer-outermost (each layer's page rounded to its own alignment, then summed) and block-outermost
// (the pages summed, rounded once to the lcm of the alignments). core/lower/testdata/vllm_lower.json
// holds them, and CI checks the file against the pinned release.
//
// Does rounding each layer and rounding once agree? Per architecture, for every block size
// (16 to 256) and every kernel block size an attention backend declares (none, MultipleOf(16),
// MultipleOf(64), 64, 128, 32 or 64):
//
//	architecture          layers at fp8 (per rank)                          layout    per layer = once
//	gpt-oss-120b          18 sliding window + 18 full, tp 1 to 8            LBNHC     yes, every case
//	minimax-m2.5          62 full, tp 1 to 8                                LBNHC     yes, every case
//	minimax-m3            60 full (fp8) + 57 indexer (bf16), tp 2 to 8      LBNHC     yes, every case
//	kimi-k2.5             61 MLA latents of 576, tp 4 and 8                 LBNHC     yes, every case
//	kimi-k3 (MLA only)    24 MLA latents of 576                             LBNHC     yes, every case
//	deepseek-v4-pro       243 packed MLA / window / ring layers             BLHNC     yes by default; no with
//	                                                                                  indexer_sparse_logits on
//	deepseek-v4.1-flash   20 packed runs                                    BLHNC     yes, with or without it
//
// A kernel block size either divides the block (the alignment is 1) or is a fraction of the page
// that divides it, so no default case rounds anything. The one case that rounds is DeepSeek-V4's
// indexer with indexer_sparse_logits on (vllm/models/deepseek_v41/attention.py:1231-1236, default
// off): there the two rules differ (1453248 against 1437696 for V4-pro), and V4's layout is
// block-outermost, which rounds once, as the substrate's single least-common-multiple field
// (rules.KVCacheGroupsInput.BlockStrideAlignment) does. The substrate has no per-layer field, and
// no default setting needs one; compat/kernelmodel's TestSubstrateStatesNoPerLayerStrideAlignment records that.
//
// Limit: the sliding-window layers of a backend whose kernel block exceeds the engine's block
// size take that larger block (_largest_kernel_block_within,
// vllm/model_executor/layers/attention/attention.py:101-130); the cases keep the engine's block.
func TestPoolBytesPerBlockIsVLLMsAtFP8(t *testing.T) {
	cases := loadPoolOracle(t)
	models := map[string]int{}
	for _, c := range cases {
		models[c.Model]++
		name := c.Model + "/" + c.Backend
		layers := layersOfCase(t, c)

		// A layer's page: the spec's page_size_bytes, a fp8 element one byte (uint8) and the
		// indexer's bf16 two.
		for i, r := range c.Runs {
			one := layersOfCase(t, ocCase{Runs: []ocLayer{{Count: 1, Spec: r.Spec, SlidingWindow: r.SlidingWindow,
				NumKVHeads: r.NumKVHeads, HeadSize: r.HeadSize, BlockSize: r.BlockSize, TokensPerState: r.TokensPerState,
				StateBytes: r.StateBytes, ElemBytes: r.ElemBytes, Alignment: r.Alignment}}})
			bs := c.BlockSize
			if got, err := PoolBytesPerBlock(one, GroupOptions{}, bs); err != nil || got != r.PageSizeBytes {
				t.Fatalf("%s tp%d block %d run %d (%s): page %d (%v), vLLM's page_size_bytes %d", name, c.TP, bs, i, r.Spec, got, err, r.PageSizeBytes)
			}
		}

		// The pool, as the engine sizes it: the layers grouped, the layout the model implies.
		got, err := PoolBytesPerBlock(layers, GroupOptions{}, c.BlockSize)
		if want := c.BytesPerBlock[c.Layout]; err != nil || got != want {
			t.Errorf("%s %s tp%d block %d: PoolBytesPerBlock = %d (%v), vLLM's %s bytes per block %d",
				c.Model, c.Backend, c.TP, c.BlockSize, got, err, c.Layout, want)
		}
		pages, err := GroupPages(layers, GroupOptions{}, c.BlockSize)
		if err != nil || len(pages) != len(c.Groups) {
			t.Fatalf("%s tp%d block %d: %d groups (%v), vLLM forms %d", name, c.TP, c.BlockSize, len(pages), err, len(c.Groups))
		}

		// Both rules, on vLLM's own groups' pages and alignments.
		var groups []PageGroup
		for _, g := range c.GroupPages {
			var pg PageGroup
			for _, run := range g {
				for range run[0] {
					pg = append(pg, PageLayer{PageBytes: run[1], Alignment: run[2]})
				}
			}
			groups = append(groups, pg)
		}
		for layout, outermost := range map[string]bool{"LBNHC": false, "BLHNC": true} {
			per, err := BytesPerBlock(groups, outermost)
			if want := c.BytesPerBlock[layout]; err != nil || per != want {
				t.Errorf("%s %s tp%d block %d: BytesPerBlock(%s) = %d (%v), vLLM's %d", c.Model, c.Backend, c.TP, c.BlockSize, layout, per, err, want)
			}
		}
		// The table above: per-layer rounding and rounding once agree except where an alignment
		// is not 1 for a layer whose page it does not divide, which only V4-pro with
		// indexer_sparse_logits on has (V4.1-flash's 512-byte alignment divides its pages).
		want := c.Model != "deepseek-v4-pro" || !strings.Contains(c.Backend, "indexer_sparse_logits")
		if c.PerLayerEquals != want {
			t.Errorf("%s %s tp%d block %d: per-layer rounding equals rounding once = %v, the table says %v",
				c.Model, c.Backend, c.TP, c.BlockSize, c.PerLayerEquals, want)
		}
	}
	for _, m := range []string{"gpt-oss-120b", "minimax-m2.5", "minimax-m3", "kimi-k2.5", "kimi-k3 (MLA layers only)", "deepseek-v4-pro"} {
		if models[m] == 0 {
			t.Errorf("no oracle case for %s", m)
		}
	}
}
