// Original to blis-vllm.

package sched

import (
	"path/filepath"
	"testing"

	"github.com/inference-sim/blis-vllm/core/kv"
)

// TestUnequalPagesGroupAsVLLMDoes holds core/kv's handling of attention layers of unequal pages to vLLM v0.33.0's: the
// gemma4-unequal-pages scenarios (Gemma 4's layer pattern, the sliding layers' pages twice the full layers') are run
// by vLLM's own get_kv_cache_groups, which unifies the pages with unify_kv_cache_spec_page_size, and the golden records what
// the Scheduler resolved (oracle/run_scheduler.py block_config): the groups, each group's block size and page, the
// scheduler's block size (the LCM of the groups') and the hash block size (their GCD with prefix caching). The Go grouping, the
// pool's bytes per block and the manager must resolve the same values. TestDifferentialAgainstVLLM holds the trace itself.
func TestUnequalPagesGroupAsVLLMDoes(t *testing.T) {
	for _, name := range []string{"gemma4-unequal-pages-layers"} {
		t.Run(name, func(t *testing.T) {
			var sc scenario
			readJSON(t, filepath.Join("../../oracle/scenarios", name+".json"), &sc)
			var want struct {
				KVBlocks *struct {
					SchedulerBlockSize int `json:"scheduler_block_size"`
					HashBlockSize      int `json:"hash_block_size"`
					NumBlocks          int `json:"num_blocks"`
					Groups             []struct {
						Type          string `json:"type"`
						BlockSize     int    `json:"block_size"`
						PageSizeBytes int    `json:"page_size_bytes"`
						Layers        int    `json:"layers"`
					} `json:"groups"`
				} `json:"kv_blocks"`
			}
			readJSON(t, filepath.Join("testdata", "golden", name+".json"), &want)
			if want.KVBlocks == nil {
				t.Fatal("the golden records no kv_blocks: record it with oracle/run_scheduler.py (the scenario asks for engine_core_block_sizes)")
			}
			c := sc.Config
			layers := scenarioLayers(t, c.KVLayers)
			o := kv.GroupOptions{BlockSize: c.BlockSize}
			groups, err := kv.GroupLayers(layers, o)
			if err != nil {
				t.Fatalf("vLLM groups these layers, core/kv refuses them: %v", err)
			}
			if len(groups) != len(want.KVBlocks.Groups) {
				t.Fatalf("%d groups, vLLM has %d", len(groups), len(want.KVBlocks.Groups))
			}
			pages, err := kv.GroupPages(layers, o, c.BlockSize)
			if err != nil {
				t.Fatal(err)
			}
			kinds := map[string]kv.GroupKind{"FullAttentionSpec": kv.FullAttention, "SlidingWindowSpec": kv.SlidingWindow}
			for i, w := range want.KVBlocks.Groups {
				if k, ok := kinds[w.Type]; !ok || groups[i].Kind != k {
					t.Errorf("group %d: kind %v, vLLM's %s", i, groups[i].Kind, w.Type)
				}
				if pages[i].BlockTokens != w.BlockSize {
					t.Errorf("group %d: block size %d, vLLM's %d", i, pages[i].BlockTokens, w.BlockSize)
				}
				if got := pages[i].BytesPerBlock; got != int64(w.PageSizeBytes*w.Layers) {
					t.Errorf("group %d: %d bytes a block over its layers, vLLM's page %d x %d layers", i, got, w.PageSizeBytes, w.Layers)
				}
			}
			m := kv.NewManagerWith(c.NumBlocks, c.BlockSize, c.MaxModelLen, c.EnablePrefixCaching, 0, kv.Options{Groups: groups})
			if m.BlockSize() != want.KVBlocks.SchedulerBlockSize || m.HashBlockSize() != want.KVBlocks.HashBlockSize {
				t.Errorf("scheduler block %d and hash block %d, vLLM's %d and %d", m.BlockSize(), m.HashBlockSize(),
					want.KVBlocks.SchedulerBlockSize, want.KVBlocks.HashBlockSize)
			}
		})
	}
}
