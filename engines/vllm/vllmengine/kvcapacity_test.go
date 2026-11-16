// Original to blis-vllm.

package vllmengine

import (
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// kvCacheSizer is the engine's start-up KV figure (core/vllm Engine.KVCacheSize).
type kvCacheSizer interface {
	KVCacheSize() (kv.Capacity, error)
}

// The engine reports its KV pool as vLLM logs it at start-up ("GPU KV cache size: N tokens"),
// for Muse Spark Open's hybrid layout on GB200 with and without the DSpark drafter
// (blis-vllm#158): its pool's blocks over the blocks one 1,048,576-token request takes in every KV
// cache group, times 1,048,576 (get_kv_cache_capacity). A request's blocks here are vLLM's own,
// from vLLM's get_kv_cache_configs over the same layers and settings (core/kv/testdata/
// vllm_kv_capacity.json, the case named beside each): 65,536 in the full-attention group and a
// window's admission cap in each sliding-window group, three groups without the drafter and four
// with its five layers.
//
// Blocks times the block size, which is what Snapshot states, is not that figure: it counts every
// block as 16 tokens of context, as if no request held a window's blocks in each window group.
func TestEngineKVCacheSizeIsVLLMsStartupFigure(t *testing.T) {
	const maxModelLen = 1 << 20
	for _, c := range []struct {
		scenario, oracleCase string
		requestBlocks        int64
		groups               int
	}{
		{"muse-gb200-tp8.yaml", "tp8-bf16", 72067, 4},
		{"muse-gb200-tp8-mnbt32768.yaml", "tp8-bf16-mnbt32768", 78211, 4},
		{"muse-gb200-tp4-pp4.yaml", "tp4-pp4-bf16-async-v2", 73603, 4},
		{"muse-gb200-tp4-dspark7.yaml", "tp4-fp8-dspark", 74244, 5},
		{"muse-gb200-tp4-dspark3.yaml", "tp4-fp8-dspark-mnbt8192", 70148, 5},
	} {
		t.Run(c.scenario, func(t *testing.T) {
			e, err := newEngine(roots(t, c.scenario))
			if err != nil {
				t.Fatal(err)
			}
			s := e.Snapshot()
			blocks := s.KVBlocksTotal + 1 // the null block, which vLLM's num_blocks counts
			want := int64(float64(blocks) / float64(c.requestBlocks) * maxModelLen)
			got := s.KVBlocksTotal * int64(s.KVBlockSize)
			if k, ok := e.(kvCacheSizer); ok {
				capacity, err := k.KVCacheSize()
				if err != nil {
					t.Fatal(err)
				}
				got = capacity.Tokens
				if capacity.Groups != c.groups || capacity.RequestBlocks != c.requestBlocks {
					t.Errorf("%d groups, %d blocks a request; vLLM's %s: %d groups, %d blocks",
						capacity.Groups, capacity.RequestBlocks, c.oracleCase, c.groups, c.requestBlocks)
				}
			}
			if got != want {
				t.Errorf("the engine reports %d tokens of KV cache; vLLM reports %d for a pool of %d blocks (%s)",
					got, want, blocks, c.oracleCase)
			}
		})
	}
}
