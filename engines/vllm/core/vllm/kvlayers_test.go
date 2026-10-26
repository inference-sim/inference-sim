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
