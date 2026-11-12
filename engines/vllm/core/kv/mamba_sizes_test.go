// Original to blis-vllm.

package kv

import (
	"reflect"
	"strings"
	"testing"
)

// The pool's block is the widest group's bytes (get_kv_cache_config_from_groups,
// kv_cache_utils.py:1521-1553@5df63a72). nemotronLike's four groups are 4 layers of a 1024-byte
// page each: 4096. Each group's own block size is its spec's: Mamba's is max_model_len in none
// mode.
func TestHybridMambaPoolAndGroupPages(t *testing.T) {
	layers := nemotronLike("none", 64)
	per, err := PoolBytesPerBlock(layers, GroupOptions{}, hybridBlock)
	if err != nil || per != 4096 {
		t.Fatalf("PoolBytesPerBlock = %d (%v), want 4096", per, err)
	}
	pages, err := GroupPages(layers, GroupOptions{}, hybridBlock)
	if err != nil {
		t.Fatal(err)
	}
	want := []GroupPage{{4096, 64, 0}, {4096, 64, 0}, {4096, 64, 0}, {4096, 16, 0}}
	if !reflect.DeepEqual(pages, want) {
		t.Errorf("GroupPages %+v, want %+v", pages, want)
	}
	on := make([]bool, len(layers))
	for i := range 8 {
		on[i] = true
	}
	stage, err := StagePoolBytesPerBlock(layers, on, GroupOptions{}, hybridBlock)
	// Layers 0-7 hold two layers of each Mamba group and of the attention group: 2 x 1024.
	if err != nil || stage != 2048 {
		t.Errorf("StagePoolBytesPerBlock = %d (%v), want 2048", stage, err)
	}
}

// The offload worker copies a block of a group layer by layer at the unpadded page
// (canonical_mapping.py:396-399, _fill_direct_ops, gpu_worker.py:395-416@5df63a72): a Mamba
// layer's is its states' 600 bytes, not the 1024 it is padded to. Stage layers 0-7 hold two
// layers of every group.
func TestMambaOffloadCopiesTheUnpaddedState(t *testing.T) {
	layers := nemotronLike("align", hybridBlock)
	on := make([]bool, len(layers))
	for i := range 8 {
		on[i] = true
	}
	got, err := StageGroupCopyBytes(layers, on, GroupOptions{}, hybridBlock)
	want := []int64{1200, 1200, 1200, 2048}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Errorf("StageGroupCopyBytes = %v (%v), want %v", got, err, want)
	}
}

func startupLayers(mode string) []Layer {
	ls := nemotronLike(mode, hybridBlock)
	for i := range ls {
		if ls[i].Spec == LayerMamba {
			ls[i].NumSpeculativeBlocks = 1
			ls[i].NumPrefillCheckpointBlocks = 1
			ls[i].PrefillCheckpointAlignment = 1
		}
	}
	return ls
}

// MambaSpec.max_memory_usage_bytes over its page (kv_cache_interface.py:1077-1083@5df63a72): in
// none mode 1 + num_speculative_blocks pages, in align mode 2 + num_speculative_blocks +
// num_prefill_checkpoint_blocks. Three Mamba groups and the attention group, max_model_len 1000
// at 16-token blocks (cdiv(1000, 16) = 63): none 63 + 3 x 2 = 69; align 63 + 3 x 4 = 75.
func TestMaxRequestBlocksCountsMambaStates(t *testing.T) {
	limits := RequestLimits{MaxModelLen: 1000}
	for _, c := range []struct {
		mode string
		want int64
	}{{"none", 69}, {"align", 75}} {
		got, err := MaxRequestBlocks(startupLayers(c.mode), GroupOptions{}, hybridBlock, limits)
		if err != nil || got != c.want {
			t.Errorf("%s: MaxRequestBlocks = %d (%v), want %d", c.mode, got, err, c.want)
		}
	}
}

// The start-up check and the longest model length that fits (kv_cache_utils.py:1003-1040,
// 2369-2412@5df63a72): in none mode the three Mamba groups take 6 blocks at any length, so a pool
// of 70 usable blocks leaves 64 for attention: cdiv(n, 16) <= 64, n <= 1024; one token more needs
// 65 blocks.
func TestEstimateMaxModelLenWithMambaStates(t *testing.T) {
	layers := startupLayers("none")
	limits := RequestLimits{MaxModelLen: 4096}
	got, err := EstimateMaxModelLen(layers, GroupOptions{}, hybridBlock, limits, 70)
	if err != nil || got != 1024 {
		t.Fatalf("EstimateMaxModelLen = %d (%v), want 1024", got, err)
	}
	limits.MaxModelLen = 1024
	if err := CheckEnoughKVCacheMemory(layers, GroupOptions{}, hybridBlock, limits, 71, 0, 0); err != nil {
		t.Errorf("71 blocks (70 beside the null block) hold the 70-block request: %v", err)
	}
	err = CheckEnoughKVCacheMemory(layers, GroupOptions{}, hybridBlock, limits, 70, 0, 0)
	if err == nil || !strings.Contains(err.Error(), "estimated maximum model length is 1008") {
		t.Errorf("70 blocks: err %v, want the refusal estimating 1008 (63 attention blocks + 6)", err)
	}
}

// MambaSpec.max_num_blocks_per_req (kv_cache_interface.py:1085-1092@5df63a72): none mode holds
// 1 + num_speculative_blocks pages; align mode's row is position-indexed over the sequence,
// cdiv(1000, 16) + num_speculative_blocks = 63 + 2.
func TestMaxNumBlocksPerReq(t *testing.T) {
	none, err := MaxNumBlocksPerReq(Group{Kind: Mamba, MambaCacheMode: "none", MambaSpeculativeBlocks: 2}, 1000, 1000)
	if err != nil || none != 3 {
		t.Errorf("none: %d (%v), want 3", none, err)
	}
	align, err := MaxNumBlocksPerReq(Group{Kind: Mamba, MambaCacheMode: "align", MambaSpeculativeBlocks: 2}, 16, 1000)
	if err != nil || align != 65 {
		t.Errorf("align: %d (%v), want 65", align, err)
	}
	if _, err := MaxNumBlocksPerReq(Group{Kind: FullAttention}, 16, 1000); err == nil {
		t.Error("an attention group has no Mamba block-table width")
	}
}
