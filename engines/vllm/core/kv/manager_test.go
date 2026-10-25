// Original to blis-vllm.

package kv

import (
	"fmt"
	"testing"
)

func testHashes(n int) []BlockHash {
	hs := make([]BlockHash, n)
	var parent BlockHash
	for i := range hs {
		parent = HashBlock(parent, []int64{int64(i), int64(i) + 1, int64(i) + 2, int64(i) + 3}, nil)
		hs[i] = parent
	}
	return hs
}

// A model whose every attention layer is one sliding window has one KV cache group, and vLLM
// looks a prefix hit up through that group's own SlidingWindowManager
// (UnitaryKVCacheCoordinator.find_longest_cache_hit, vllm/v1/core/kv_cache_coordinator.py:
// 574-591@5df63a72), which needs only the window's blocks, not the whole prefix. With window 8
// and block size 4, a 7-block prompt whose block 1 was evicted hits 28 tokens: blocks 5 and 6
// cached, nulls below them; with the EAGLE drop, one block more is matched and dropped, 24
// tokens (single_type_kv_cache_manager.py:955-1051). Both expectations are what vLLM's own
// SlidingWindowManager.find_longest_cache_hit returns for this pool. The full-attention scan
// would stop at the evicted block, 4 tokens.
func TestOneSlidingWindowGroupLooksUpThroughTheWindow(t *testing.T) {
	for _, c := range []struct {
		eagle  bool
		tokens int
		cached []bool
	}{
		{false, 28, []bool{false, false, false, false, false, true, true}},
		{true, 24, []bool{false, false, false, false, true, true}},
	} {
		m := NewManagerWith(40, 4, 128, true, 0, Options{EagleBlockDrop: c.eagle,
			Groups: []Group{{Kind: SlidingWindow, Window: 8}}})
		hs := testHashes(7)
		blocks := m.Pool.New(7)
		m.Pool.CacheFull(blocks, hs, 0, 7, StoredRun{Tokens: make([]int64, 28)}, 4)
		m.Pool.Free(blocks)
		// Evict block 1's hash, as reuse of a free block would.
		m.Pool.evict(blocks[1])

		hit, n := m.ComputedBlocks(hs, 29)
		if n != c.tokens {
			t.Errorf("eagle %v: hit %d tokens, want %d: one sliding-window group looks up through the window", c.eagle, n, c.tokens)
			continue
		}
		if len(hit[0]) != len(c.cached) {
			t.Errorf("eagle %v: %d blocks, want %d", c.eagle, len(hit[0]), len(c.cached))
			continue
		}
		for i, b := range hit[0] {
			if got := !b.Null; got != c.cached[i] {
				t.Errorf("eagle %v: block %d cached %v, want %v (nulls below the window)", c.eagle, i, got, c.cached[i])
			}
		}
	}
}

// Under a multi-module MTP drafter vLLM tags every sliding-window spec with
// extra_retained_tokens = num_prefill_lookahead_tokens - 1 (get_kv_cache_configs,
// vllm/v1/core/kv_cache_utils.py:2582-2589@5df63a72), which keeps that many tokens below the
// window (get_num_skipped_tokens, single_type_kv_cache_manager.py:1135-1171) and widens the
// admission cap by them (max_admission_blocks_per_request, vllm/v1/kv_cache_interface.py:
// 859-885). Full attention is unchanged.
func TestMultiModuleMTPKeepsExtraTokensBelowTheWindow(t *testing.T) {
	for _, c := range []struct {
		lookahead, computed, wantSkipped int
	}{
		{0, 20, 13},  // no drafter: 20 - 8 + 1
		{1, 20, 13},  // EAGLE reads one token ahead: nothing extra
		{4, 20, 10},  // a 4-module MTP chain: 3 extra tokens kept
		{4, 9, 0},    // never negative
		{4, 12, 2},   // 12 - 8 + 1 - 3
		{3, 100, 91}, // 100 - 8 + 1 - 2
	} {
		t.Run(fmt.Sprintf("lookahead%d-computed%d", c.lookahead, c.computed), func(t *testing.T) {
			m := NewManagerWith(40, 4, 128, true, 0, Options{PrefillLookahead: c.lookahead,
				Groups: []Group{{Kind: FullAttention}, {Kind: SlidingWindow, Window: 8}}})
			if got := m.groups[1].skippedTokens(c.computed); got != c.wantSkipped {
				t.Errorf("sliding window skips %d tokens, want %d", got, c.wantSkipped)
			}
			if got := m.groups[0].skippedTokens(c.computed); got != 0 {
				t.Errorf("full attention skips %d tokens, want 0", got)
			}
		})
	}

	// The admission cap: min(window - 1 + extra + in_flight, max_model_len) tokens, in blocks
	// plus one. Window 8, extra 3, 16 tokens in flight: cdiv(26, 4) + 1 = 8 blocks, where
	// without the extra tokens it is cdiv(23, 4) + 1 = 7.
	for _, c := range []struct{ lookahead, want int }{{0, 7}, {4, 8}} {
		m := NewManagerWith(400, 4, 1024, true, 0, Options{PrefillLookahead: c.lookahead, MaxInFlightTokens: 16,
			Groups: []Group{{Kind: SlidingWindow, Window: 8}}})
		if got := m.toAllocate("r", 1000, [][]*Block{nil}, 0, true); got != c.want {
			t.Errorf("lookahead %d: a new 1000-token request reserves %d window blocks, want %d", c.lookahead, got, c.want)
		}
	}
}

// A model whose layers share one spec is one KV cache group, as get_kv_cache_groups gives it
// (vllm/v1/core/kv_cache_utils.py:2185-2189@5df63a72): vLLM's own call returns one
// FullAttentionSpec group for two full-attention layers and one SlidingWindowSpec group for
// two equal windows. Distinct specs stay one group each.
func TestUniformLayersAreOneGroup(t *testing.T) {
	full, sw8, sw12 := Group{Kind: FullAttention}, Group{Kind: SlidingWindow, Window: 8}, Group{Kind: SlidingWindow, Window: 12}
	for _, c := range []struct {
		layers []Group
		want   []Group
	}{
		{[]Group{full, full}, []Group{full}},
		{[]Group{sw8, sw8}, []Group{sw8}},
		{[]Group{sw8}, []Group{sw8}},
		{[]Group{full, sw8}, []Group{full, sw8}},
		{[]Group{sw8, sw12}, []Group{sw8, sw12}},
	} {
		got := ResolveGroups(c.layers, false)
		if fmt.Sprint(got) != fmt.Sprint(c.want) {
			t.Errorf("ResolveGroups(%v) = %v, want %v", c.layers, got, c.want)
		}
	}
}
