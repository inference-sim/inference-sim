package vllm

import (
	"strings"
	"testing"
)

// A packed layout whose full-attention group is at a block size other than the engine's publishes
// BlockStored at that group's block_size, while the router keys every pod's blocks at the
// engine's: the engine says so (KVEventsRefusal) so a simulator refuses precise-prefix routing
// rather than index blocks the router would never match. The layout here is constructed (a 64-token
// compressed group under 256-token windows). DeepSeek V4's own layout (compressed group at the
// engine's 256 beside 64-token windows) and Muse Spark's are not refused.
func TestKVEventsRefusedForAPackedFullAttentionGroup(t *testing.T) {
	const packed = "kv_layers:\n" +
		"- {count: 1, spec: sliding_window_mla, sliding_window: 512, num_kv_heads: 1, head_size: 512, block_size: 256, state_bytes: 584, alignment: 576}\n" +
		"- {count: 1, spec: mla, num_kv_heads: 1, head_size: 512, block_size: 64, tokens_per_state: 32, state_bytes: 584, alignment: 576}\n" +
		"block_size: 256\n"
	const deepseek = "kv_layers:\n" +
		"- {count: 1, spec: sliding_window_mla, sliding_window: 128, num_kv_heads: 1, head_size: 512, block_size: 64, state_bytes: 584, alignment: 576}\n" +
		"- {count: 1, spec: mla, num_kv_heads: 1, head_size: 512, block_size: 256, tokens_per_state: 128, state_bytes: 584, alignment: 576}\n" +
		"block_size: 256\n"
	e := mustEngine(t, packed)
	err := e.KVEventsRefusal()
	if err == nil || !strings.Contains(err.Error(), "block size") {
		t.Fatalf("KVEventsRefusal() = %v, want a refusal naming the block size", err)
	}
	const muse = "kv_layers:\n" +
		"- {count: 1, spec: full, num_kv_heads: 1, head_size: 64}\n" +
		"- {count: 1, spec: sliding_window, sliding_window: 32, num_kv_heads: 1, head_size: 64}\n"
	for _, extra := range []string{"", deepseek, muse} {
		if err := mustEngine(t, extra).KVEventsRefusal(); err != nil {
			t.Errorf("%q: KVEventsRefusal() = %v, want none", extra, err)
		}
	}
}
