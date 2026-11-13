// Original to blis-vllm.

package kv

import (
	"strings"
	"testing"
)

func noneGroup(blockSize, spec int) Group {
	return Group{Kind: Mamba, BlockSize: blockSize, MambaCacheMode: "none", MambaSpeculativeBlocks: spec}
}

func lens(m *Manager, id string) []int {
	var out []int
	for g := range m.NumGroups() {
		out = append(out, len(m.GroupBlocks(g, id)))
	}
	return out
}

// In none mode a Mamba group allocates by the base rules plus num_speculative_blocks
// (MambaManager.get_num_blocks_to_allocate and allocate_new_blocks, single_type_kv_cache_manager.py:
// 1692-1716, 1790-1800@5df63a72): block_size x num_speculative_blocks is added to the tokens that
// need a slot, so the group holds cdiv(tokens, block_size) + num_speculative_blocks blocks. Mamba's
// block is max_model_len (64): 20 tokens are one block and one speculative block, beside full
// attention's cdiv(20, 16) = 2. The full-sequence gate counts the same 4.
func TestMambaNoneAllocatesSpeculativeBlocks(t *testing.T) {
	m := NewManagerWith(20, 16, 64, false, 0, Options{Groups: []Group{{Kind: FullAttention}, noneGroup(64, 1)}})
	free := m.Pool.NumFree()
	if got := m.RemainingBlocks("a", 20, 0); got != 4 {
		t.Errorf("RemainingBlocks = %d, want 4 (2 attention + 1 + 1 speculative)", got)
	}
	if !m.Allocate("a", Alloc{New: 20, NumTokens: 20}) {
		t.Fatal("allocation refused")
	}
	if got := lens(m, "a"); got[0] != 2 || got[1] != 2 {
		t.Errorf("blocks per group %v, want [2 2]", got)
	}
	if got := free - m.Pool.NumFree(); got != 4 {
		t.Errorf("%d blocks taken, want 4", got)
	}
	// A decode step inside both groups' blocks takes none; crossing attention's 32nd token takes one.
	m.Allocate("a", Alloc{Computed: 20, New: 1, NumTokens: 21})
	m.Allocate("a", Alloc{Computed: 32, New: 1, NumTokens: 33})
	if got := lens(m, "a"); got[0] != 3 || got[1] != 2 {
		t.Errorf("after 33 tokens blocks per group %v, want [3 2]", got)
	}
}

// Mamba keeps only the last computed token's state: get_num_skipped_tokens is
// num_computed_tokens - 1 (single_type_kv_cache_manager.py:1954-1960@5df63a72), and
// remove_skipped_blocks frees the blocks wholly below it (:672-710). At block size 4, after 10
// computed tokens 9 are skipped: blocks 0 and 1 are freed into null slots, block 2 stays.
func TestMambaNoneFreesBlocksBelowTheLastToken(t *testing.T) {
	m := NewManagerWith(10, 4, 64, false, 0, Options{Groups: []Group{noneGroup(4, 0)}})
	if !m.Allocate("a", Alloc{New: 10, NumTokens: 10}) {
		t.Fatal("allocation refused")
	}
	free := m.Pool.NumFree()
	if !m.Allocate("a", Alloc{Computed: 10, New: 1, NumTokens: 11}) {
		t.Fatal("decode allocation refused")
	}
	bs := m.Blocks("a")
	if len(bs) != 3 || !bs[0].Null || !bs[1].Null || bs[2].Null {
		t.Fatalf("blocks %v, want two null slots then the live block", bs)
	}
	if got := m.Pool.NumFree() - free; got != 2 {
		t.Errorf("%d blocks freed, want 2", got)
	}
	// Freeing the request returns the live block only; the nulls are not blocks.
	m.Free("a")
	if m.Pool.NumFree() != 9 {
		t.Errorf("%d free after Free, want all 9 usable blocks", m.Pool.NumFree())
	}
}

// Hybrid groups run at the LCM of their block sizes when prefix caching is off
// (resolve_kv_cache_block_sizes, kv_cache_utils.py:804-904@5df63a72): 16 and Mamba's max_model_len
// 64. A speculative step's blocks leave out Mamba's, which it counts itself
// (scheduler.py:347-357): cdiv(3, 16) = 1 for the attention group alone.
func TestMambaNoneBlockSizeAndSpecDecodeBlocks(t *testing.T) {
	m := NewManagerWith(20, 16, 64, false, 0, Options{Groups: []Group{{Kind: FullAttention}, noneGroup(64, 1)}})
	if m.BlockSize() != 64 || m.HashBlockSize() != 64 {
		t.Errorf("block sizes %d and %d, want the LCM 64 for both", m.BlockSize(), m.HashBlockSize())
	}
	if got := m.SpecDecodeStepBlocks(3); got != 1 {
		t.Errorf("SpecDecodeStepBlocks(3) = %d, want 1", got)
	}
}

// A Mamba group in mode "none" with prefix caching on is unreachable in vLLM, whose config hook
// sets the mode to "align" (vllm/model_executor/models/config.py:672-706@5df63a72): refused by
// name, as an error from CheckGroups and a panic from the manager.
func TestMambaNoneWithPrefixCachingIsRefused(t *testing.T) {
	groups := []Group{{Kind: FullAttention}, noneGroup(64, 0)}
	if err := CheckGroups(groups, false); err != nil {
		t.Errorf("caching off: %v", err)
	}
	err := CheckGroups(groups, true)
	if err == nil || !strings.Contains(err.Error(), `mamba_cache_mode "none"`) {
		t.Fatalf("caching on: err %v, want a refusal naming mamba_cache_mode none", err)
	}
	defer func() {
		if recover() == nil {
			t.Error("NewManagerWith built a Mamba none group under prefix caching")
		}
	}()
	NewManagerWith(20, 16, 64, true, 0, Options{Groups: groups})
}
