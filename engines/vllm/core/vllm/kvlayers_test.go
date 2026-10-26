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
