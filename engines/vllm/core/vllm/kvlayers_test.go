// Original to blis-vllm.

package vllm

import (
	"reflect"
	"strings"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// An engine block's kv_layers become the scheduler's KV cache groups, as vLLM groups them:
// gpt-oss-120b's alternating window and full layers are two groups, window first, and with the
// hybrid manager disabled one full-attention group; a spec name it does not know is refused.
func TestKVLayersAreTheSchedulersGroups(t *testing.T) {
	const gptOSS = "kv_layers:\n" +
		"- {count: 1, spec: sliding_window, sliding_window: 128, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: sliding_window, sliding_window: 128, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n"
	for _, c := range []struct {
		extra string
		want  []kv.Group
	}{
		{gptOSS, []kv.Group{{Kind: kv.SlidingWindow, Window: 128}, {Kind: kv.FullAttention}}},
		{gptOSS + "disable_hybrid_kv_cache_manager: true\n", []kv.Group{{Kind: kv.FullAttention}}},
		{"", nil},
	} {
		cfg, err := decodeConfig(block(t, c.extra))
		if err != nil {
			t.Fatal(err)
		}
		s, err := resolve(cfg)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(s.sched.KVGroups, c.want) {
			t.Errorf("%q: groups %v, want %v", c.extra, s.sched.KVGroups, c.want)
		}
	}
	cfg, err := decodeConfig(block(t, "kv_layers:\n- {count: 2, spec: mamba, num_kv_heads: 1, head_size: 1}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := resolve(cfg); err == nil || !strings.Contains(err.Error(), `unknown KV layer spec "mamba"`) {
		t.Errorf("an unknown spec: %v, want it refused by name", err)
	}
}

// The pool's block count is the memory over the widest KV cache group's bytes per block, the
// layers' element size taken from the memory model's bytes for one block of every layer: at an
// fp8 cache (one byte an element), gpt-oss-120b's two groups of 18 hold available / (18 pages),
// one group of all 36 (the hybrid manager off) available / (36 pages).
func TestKVPoolBlocksAreTheMemoryOverTheWidestGroup(t *testing.T) {
	runs := []KVLayerRun{}
	for range 18 {
		runs = append(runs, KVLayerRun{Count: 1, Spec: "sliding_window", SlidingWindow: 128, NumKVHeads: 8, HeadSize: 64},
			KVLayerRun{Count: 1, Spec: "full", NumKVHeads: 8, HeadSize: 64})
	}
	const page = 16 * 2 * 8 * 64 // one layer's page at 16 tokens, an fp8 cache
	const available = 1_000_000_007
	for _, c := range []struct {
		disableHybrid bool
		want          int64
	}{{false, available / (18 * page)}, {true, available / (36 * page)}} {
		got, err := KVPoolBlocks(runs, c.disableHybrid, 16, 36*page, available)
		if err != nil || got != c.want {
			t.Errorf("disable_hybrid %t: KVPoolBlocks = %d (%v), want %d", c.disableHybrid, got, err, c.want)
		}
	}
	// A memory model whose block is not a whole number of bytes an element is refused.
	if _, err := KVPoolBlocks(runs, false, 16, 36*page+1, available); err == nil {
		t.Error("KVPoolBlocks sized a pool from a fractional element size")
	}
}

// A layer that states its element size keeps it, and the others share what the memory model's
// block leaves: MiniMax-M3's indexer keys at bf16, their own dtype, beside attention KV at fp8
// (vllm/models/minimax_m3/nvidia/model.py:505-509@5df63a72). Its 60 attention layers of two
// heads and 57 one-head indexer layers at 16 tokens are one uniform-type group of two page sizes,
// whose pool vLLM sizes at 72893 blocks from 52846898709 bytes (oracle/scenarios/
// dashboard-minimax-m3-fp8-kv-probe.json): 60 x 16 x 2x2x128 x 1 + 57 x 16 x 128 x 2 = 724992
// bytes a block. Taking the indexer at the attention dtype instead, as one element size for every
// layer, would refuse the memory model's block or size another pool.
func TestKVPoolBlocksKeepAStatedElementSize(t *testing.T) {
	runs := []KVLayerRun{
		{Count: 60, Spec: "full", NumKVHeads: 2, HeadSize: 128},
		{Count: 57, Spec: "indexer", NumKVHeads: 1, HeadSize: 128, ElemBytes: 2},
	}
	const perBlock = 60*16*2*2*128*1 + 57*16*128*2
	got, err := KVPoolBlocks(runs, false, 16, perBlock, 52846898709)
	if err != nil || got != 72893 {
		t.Errorf("KVPoolBlocks = %d (%v), want vLLM's 72893", got, err)
	}
	// The remainder after the stated layers must be whole bytes an element for the others.
	if _, err := KVPoolBlocks(runs, false, 16, perBlock+1, 52846898709); err == nil {
		t.Error("KVPoolBlocks sized a pool from a fractional element size")
	}
}

// A trailing drafter's group is flagged when the engine block resolves: vLLM locates the drafter's
// layers by position when they register after the target's (_uses_trailing_mtp_layers,
// vllm/v1/core/kv_cache_utils.py:1997-2013@5df63a72), which is MTP on any model and every
// EAGLE-family method on DeepSeek V4. Here two DeepSeek V4 windows and a compressed cache, the
// drafter's window last: packed, the windows are two groups beside the compressed one, and the
// second window group holds the drafter's layer, so only it is the drafter's; with no drafter no
// group is.
func TestTheTrailingDrafterGroupIsFlagged(t *testing.T) {
	const deepseek = "kv_layers:\n" +
		"- {count: 1, spec: sliding_window_mla, sliding_window: 128, num_kv_heads: 1, head_size: 512, block_size: 64, state_bytes: 584, alignment: 576}\n" +
		"- {count: 1, spec: mla, num_kv_heads: 1, head_size: 512, block_size: 256, tokens_per_state: 128, state_bytes: 584, alignment: 576}\n" +
		"- {count: 1, spec: sliding_window_mla, sliding_window: 128, num_kv_heads: 1, head_size: 512, block_size: 64, state_bytes: 584, alignment: 576}\n"
	for _, c := range []struct {
		name, extra string
		eagle       []bool
	}{
		{"mtp", "speculative: {method: mtp, num_spec_tokens: 1}\nmtp_nextn_predict_layers: 1\n", []bool{false, true, false}},
		{"dspark", "speculative: {method: dspark, num_spec_tokens: 2}\nuse_v2_model_runner: true\n", []bool{false, true, false}},
		{"none", "", []bool{false, false, false}},
	} {
		t.Run(c.name, func(t *testing.T) {
			cfg, err := decodeConfig(block(t, deepseek+"block_size: 256\n"+c.extra))
			if err != nil {
				t.Fatal(err)
			}
			s, err := resolve(cfg)
			if err != nil {
				t.Fatal(err)
			}
			if len(s.sched.KVGroups) != len(c.eagle) {
				t.Fatalf("groups %v, want %d", s.sched.KVGroups, len(c.eagle))
			}
			for i, g := range s.sched.KVGroups {
				if g.Eagle != c.eagle[i] {
					t.Errorf("group %d (%v) Eagle %t, want %t", i, g, g.Eagle, c.eagle[i])
				}
			}
		})
	}
}
