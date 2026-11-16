// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Derived from vllm/v1/kv_cache_interface.py at vLLM v0.33.0 (4ba787ec9959e566d3b587a796dff07673324da5).

package kv

import "testing"

// The Mamba manager's prompt checkpoint leaves the block pool's hash lookahead uncomputed
// (get_mamba_prefill_checkpoint_position, vllm/v1/kv_cache_interface.py:1169-1186@4ba787ec, read
// with block_pool.num_hash_lookahead_tokens at single_type_kv_cache_manager.py:1761-1766). v0.32.0
// left one token whatever the lookahead, so 98 tokens at block 16 checkpointed at 96.
func TestMambaCheckpointLeavesTheHashLookahead(t *testing.T) {
	for _, c := range []struct{ n, hbs, lookahead, want int }{
		{98, 16, 3, 80}, // (98 - 3) // 16 * 16
		{98, 16, 1, 96},
		{98, 16, 0, 96},
		{99, 16, 3, 96},
	} {
		if got := mambaPrefillCheckpointPosition(c.n, c.hbs, false, c.lookahead); got != c.want {
			t.Errorf("position(%d, %d, lookahead %d) = %d, want %d", c.n, c.hbs, c.lookahead, got, c.want)
		}
	}
	// With the block drop on, the drop already leaves tokens and the lookahead is not read
	// (the scheduler turns the drop off whenever a lookahead is hashed).
	if got := mambaPrefillCheckpointPosition(98, 16, true, 3); got != 80 {
		t.Errorf("position(98, 16, drop, lookahead 3) = %d, want 80", got)
	}
}

// A drafter whose layers fall in two groups flags both (vllm/v1/core/kv_cache_planning.py:
// 1650-1656@4ba787ec): here a full-attention and a sliding-window draft layer after the
// target's. v0.32.0 flagged only the group of the model's last layer, and only for MTP and
// DeepSeek V4 drafters (_annotate_eagle_groups, vllm/v1/core/kv_cache_utils.py:2029-2083@5df63a72).
func TestEveryGroupHoldingADraftLayerIsTheDrafters(t *testing.T) {
	var layers []Layer
	for range 4 {
		layers = append(layers, Layer{Spec: LayerFull, KVHeads: 8, HeadSize: 128},
			Layer{Spec: LayerSlidingWindow, Window: 64, KVHeads: 8, HeadSize: 128})
	}
	layers = append(layers, Layer{Spec: LayerFull, KVHeads: 8, HeadSize: 128, Draft: true},
		Layer{Spec: LayerSlidingWindow, Window: 64, KVHeads: 8, HeadSize: 128, Draft: true})
	groups, err := GroupLayers(layers, GroupOptions{TagDraftGroups: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(groups) != 2 {
		t.Fatalf("groups %v, want a full and a sliding-window group", groups)
	}
	for i, g := range groups {
		if !g.Eagle {
			t.Errorf("group %d (%v) is not flagged, though it holds a draft layer", i, g)
		}
	}
	// Without the tag (a drafter that drops no block) no group is flagged.
	groups, _ = GroupLayers(layers, GroupOptions{})
	for i, g := range groups {
		if g.Eagle {
			t.Errorf("untagged: group %d flagged", i)
		}
	}
}
