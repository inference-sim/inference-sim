// Original to blis-vllm.

package kv

import (
	"strings"
	"testing"
)

// A sliding window below the scheduler block size caches only the blocks a hit at an aligned
// boundary can reach: each 256-token segment's last cdiv(window - 1, block) blocks, one block
// later and one longer with the EAGLE peek. Each row is vLLM v0.32.0's own
// SlidingWindowManager.reachable_block_mask over two segments at alignment 256 with dense
// retention (single_type_kv_cache_manager.py:1054-1133@5df63a72); "" is its None, every block.
func TestReachableWindowBlocksAreVLLMs(t *testing.T) {
	for _, c := range []struct {
		blockSize, window int
		eagle             bool
		want              string
	}{
		{64, 128, false, strings.Repeat(".", 2) + strings.Repeat("1", 2) + strings.Repeat(".", 2) + strings.Repeat("1", 2)},
		{64, 128, true, strings.Repeat(".", 2) + strings.Repeat("1", 3) + strings.Repeat(".", 1) + strings.Repeat("1", 2)},
		{8, 128, false, strings.Repeat(".", 16) + strings.Repeat("1", 16) + strings.Repeat(".", 16) + strings.Repeat("1", 16)},
		{4, 8, false, strings.Repeat(".", 62) + strings.Repeat("1", 2) + strings.Repeat(".", 62) + strings.Repeat("1", 2)},
		{4, 8, true, strings.Repeat(".", 62) + strings.Repeat("1", 3) + strings.Repeat(".", 61) + strings.Repeat("1", 2)},
		{32, 128, false, strings.Repeat(".", 4) + strings.Repeat("1", 4) + strings.Repeat(".", 4) + strings.Repeat("1", 4)},
		{256, 128, false, ""},
	} {
		m := NewManagerWith(100, 256, 4096, true, 0, Options{EagleBlockDrop: c.eagle, Groups: []Group{
			{Kind: FullAttention, BlockSize: 256},
			{Kind: SlidingWindowMLA, Window: c.window, BlockSize: c.blockSize, Eagle: c.eagle}}})
		mask := m.reachableWindowBlocks(m.groups[1], 0, 2*256/c.blockSize)
		got := ""
		for _, keep := range mask {
			if keep {
				got += "1"
			} else {
				got += "."
			}
		}
		if got != c.want {
			t.Errorf("block %d window %d eagle %t: mask %q, want %q", c.blockSize, c.window, c.eagle, got, c.want)
		}
	}
}

// A drafter's EAGLE block drop reaches the target groups that share the drafter group's spec,
// and no other: vLLM v0.32.0's coordinator flags the drafter's group (eagle_group_ids,
// kv_cache_coordinator.py:127-133@5df63a72), merges groups of one spec, and gives every member of
// a merged group holding a drafter's group use_eagle (:726-736, 761-765). The groups are
// deepseek-v4-pro's with its MTP layer, as get_kv_cache_groups packs them (group 1 holds the MTP
// layer), and want is each manager's use_eagle as vLLM's own scheduler sets it
// (dashboard-deepseek-v4-pro-mtp-layers): the window groups of the drafter's spec drop the
// block, the compressed MLA group and the windows at blocks 8 and 4 do not.
func TestEagleBlockDropReachesTheDraftersSpecGroup(t *testing.T) {
	m := NewManagerWith(1200, 256, 2048, true, 0, Options{EagleBlockDrop: true, Groups: []Group{
		{Kind: SlidingWindowMLA, Window: 128, BlockSize: 64},
		{Kind: SlidingWindowMLA, Window: 128, BlockSize: 64, Eagle: true},
		{Kind: FullAttention, BlockSize: 256},
		{Kind: SlidingWindowMLA, Window: 128, BlockSize: 8},
		{Kind: SlidingWindowMLA, Window: 8, BlockSize: 4},
	}})
	want := []bool{true, true, false, false, false}
	for i, g := range m.groups {
		if g.eagle != want[i] {
			t.Errorf("group %d use_eagle %t, want %t (vLLM's own manager's)", i, g.eagle, want[i])
		}
	}
}
