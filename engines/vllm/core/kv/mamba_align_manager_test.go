// Original to blis-vllm.

package kv

import "testing"

func alignHybrid(t *testing.T) *Manager {
	t.Helper()
	return NewManagerWith(30, alignBS, 64, true, 0, Options{Groups: []Group{
		{Kind: FullAttention}, {Kind: Mamba, MambaCacheMode: "align"}}})
}

func hashesFor(tag string, n int) []BlockHash {
	out := make([]BlockHash, n)
	for i := range out {
		out[i] = BlockHash(tag + string(rune('a'+i)))
	}
	return out
}

func tableShapes(m *Manager, id string) (string, string) {
	return shape(m.GroupBlocks(0, id)), shape(m.GroupBlocks(1, id))
}

// Through Manager.Allocate and ComputedBlocks, a full-attention group and a Mamba align group:
// the second request hits the prompt the first cached, the full-attention group gets the three
// cached blocks and the Mamba group null blocks before its one state block (:1512-1517); a
// hit on a block cached in the same step is deferred until NewStepStarts.
func TestManagerMambaAlign_HybridHitDeferredThenAdmitted(t *testing.T) {
	m := alignHybrid(t)
	h := hashesFor("p", 3)
	if !m.Allocate("r1", Alloc{New: 12, Hashes: h, NumTokens: 12, FullSequenceMustFit: true,
		Run: StoredRun{NumPromptTokens: 12}}) {
		t.Fatal("r1 refused")
	}
	if f, mb := tableShapes(m, "r1"); f != "hhh" || mb != "nnh" {
		t.Fatalf("r1 tables = %q (full) and %q (mamba), want hhh and nnh", f, mb)
	}
	if m.Pool.NumFree() != 25 {
		t.Fatalf("free = %d, want 25", m.Pool.NumFree())
	}

	blocks, n, boundary := m.ComputedBlocksWithBoundary(append(hashesFor("p", 3), BlockHash("tail")), 14)
	if n != 12 || boundary != 0 || len(blocks[0]) != 3 || shape(blocks[1]) != "nnh" {
		t.Fatalf("hit = %d tokens, boundary %d, full %q, mamba %q; want 12, 0, 3 blocks, nnh",
			n, boundary, shape(blocks[0]), shape(blocks[1]))
	}
	alloc := Alloc{New: 2, NewComputed: blocks, NumNewComputed: n, Hashes: append(h, BlockHash("tail")), NumTokens: 14,
		FullSequenceMustFit: true, Run: StoredRun{NumPromptTokens: 14}}
	if m.Allocate("r2", alloc) {
		t.Fatal("r2 hit a block cached in the same step and was admitted")
	}
	m.NewStepStarts()
	if !m.Allocate("r2", alloc) {
		t.Fatal("r2 refused in the next step")
	}
	if f, mb := tableShapes(m, "r2"); f != "hhhb" || mb != "nnhb" {
		t.Errorf("r2 tables = %q and %q, want hhhb and nnhb", f, mb)
	}
	if m.Pool.NumFree() != 23 {
		t.Errorf("free = %d, want 23 (one new block per group)", m.Pool.NumFree())
	}
	m.Free("r1")
	m.Free("r2")
	if m.Pool.NumFree() != 29 {
		t.Errorf("free after both requests = %d, want 29", m.Pool.NumFree())
	}
}

// A Mamba group that stops short of a longer full-attention hit reports the gap as the shared
// prefix boundary (kv_cache_coordinator.py:922-930, kv_cache_manager.py:313-321): r1 cached
// three full blocks but only the last Mamba state, so a request that may use 11 tokens finds
// 8 tokens in the full-attention group, no state in the Mamba group, hit 0, boundary 8.
func TestManagerMambaAlign_SharedPrefixBoundaryIsTheLongestSingleGroupHit(t *testing.T) {
	m := alignHybrid(t)
	h := hashesFor("p", 3)
	if !m.Allocate("r1", Alloc{New: 12, Hashes: h, NumTokens: 12, Run: StoredRun{NumPromptTokens: 12}}) {
		t.Fatal("r1 refused")
	}
	m.NewStepStarts()
	blocks, n, boundary := m.ComputedBlocksWithBoundary(h, 12)
	if n != 0 || boundary != 8 || len(blocks[0]) != 0 || len(blocks[1]) != 0 {
		t.Errorf("hit = %d tokens, boundary %d, %d + %d blocks; want 0, 8, none", n, boundary, len(blocks[0]), len(blocks[1]))
	}
	if b2, n2 := m.ComputedBlocks(h, 12); n2 != 0 || len(b2[0]) != 0 {
		t.Errorf("ComputedBlocks changed its result: %d tokens", n2)
	}
}

// A Mamba-only model is a unitary coordinator: Manager.Allocate, RemoveSkipped and Free drive
// the align manager, and a decode crossing a block boundary retires the old state.
func TestManagerMambaAlign_MambaOnlyDecode(t *testing.T) {
	m := NewManagerWith(20, alignBS, 64, true, 0, Options{Groups: []Group{
		{Kind: Mamba, MambaCacheMode: "align", MambaSpeculativeBlocks: 2}}})
	h := hashesFor("p", 4)
	if !m.Allocate("r", Alloc{New: 8, Hashes: h, NumTokens: 8, FullSequenceMustFit: true, Run: StoredRun{NumPromptTokens: 8}}) {
		t.Fatal("prefill refused")
	}
	// required 2 + 2 = 4, num_skipped 1: [n A B C], A cached at 8 tokens.
	if got := shape(m.GroupBlocks(0, "r")); got != "nhbb" {
		t.Fatalf("table = %q, want nhbb", got)
	}
	if !m.Allocate("r", Alloc{Computed: 8, New: 1, Hashes: h, NumTokens: 9, Run: StoredRun{NumPromptTokens: 8}}) {
		t.Fatal("decode refused")
	}
	if got := shape(m.GroupBlocks(0, "r")); got != "nhbbb" {
		t.Errorf("table after crossing = %q, want nhbbb", got)
	}
	free := m.Pool.NumFree()
	m.RemoveSkipped("r", 9)
	if got := shape(m.GroupBlocks(0, "r")); got != "nnbbb" || m.Pool.NumFree() != free+1 {
		t.Errorf("after RemoveSkipped(9): table %q, free %d (was %d); want nnbbb and one more free", got, m.Pool.NumFree(), free)
	}
	m.Free("r")
	if m.Pool.NumFree() != 19 {
		t.Errorf("free after Free = %d, want 19", m.Pool.NumFree())
	}
}

func TestManagerMambaAlign_RefusesPartialHashHitsAtConstruction(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("a Mamba block larger than the hash block size was accepted")
		}
	}()
	NewManagerWith(20, 4, 64, true, 0, Options{Groups: []Group{
		{Kind: FullAttention, BlockSize: 4}, {Kind: Mamba, MambaCacheMode: "align", BlockSize: 8}}})
}
