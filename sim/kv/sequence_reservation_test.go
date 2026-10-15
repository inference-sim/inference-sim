package kv

import (
	"fmt"
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
)

// promptRequest is a request with an n-token prompt of distinct tokens, so no two requests
// share a prefix block.
func promptRequest(id string, n int) *sim.Request {
	toks := make([]sim.TokenID, n)
	for i := range toks {
		toks[i] = sim.TokenID(len(id)*100000 + i + 1)
	}
	for i, c := range id {
		toks[0] += sim.TokenID(c) * sim.TokenID(i+7)
	}
	return &sim.Request{ID: id, InputTokens: toks, OutputTokens: make([]sim.TokenID, 4)}
}

// admitUntilFull admits whole prompts until the pool refuses one and returns how many fit.
func admitUntilFull(t *testing.T, kvc *KVCacheState, promptTokens int) int {
	t.Helper()
	for n := 0; ; n++ {
		req := promptRequest(fmt.Sprintf("r%d", n), promptTokens)
		if !kvc.AllocateKVBlocks(req, 0, int64(promptTokens), nil) {
			assertBlockConservation(t, kvc)
			return n
		}
		req.ProgressIndex = int64(promptTokens)
	}
}

// #1941: at a fixed pool, charging every resident sequence F fixed blocks lowers the
// admitted-sequence ceiling from floor(N/P) to floor(N/(P+F)), and with F = 0 the ceiling is
// the pool's alone. The law, not a value: the fixed term is paid once per sequence, whatever
// its length.
func TestSequenceFixedBlocks_LowerTheAdmissionCeilingByTheFixedTerm(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		blockSize := rapid.IntRange(1, 16).Draw(rt, "blockSize")
		total := rapid.Int64Range(8, 400).Draw(rt, "total")
		promptBlocks := rapid.Int64Range(1, 6).Draw(rt, "promptBlocks")
		fixed := rapid.Int64Range(0, 6).Draw(rt, "fixed")

		kvc := NewKVCacheState(total, int64(blockSize))
		kvc.SequenceFixedBlocks = fixed
		got := admitUntilFull(t, kvc, int(promptBlocks)*blockSize)
		if want := int(total / (promptBlocks + fixed)); got != want {
			rt.Fatalf("pool %d, prompt %d blocks, fixed %d: admitted %d sequences, want %d",
				total, promptBlocks, fixed, got, want)
		}
		if used, want := kvc.UsedBlocks(), int64(got)*(promptBlocks+fixed); used != want {
			rt.Fatalf("used %d blocks for %d sequences, want %d", used, got, want)
		}
	})
}

// The fixed blocks are charged once, at a sequence's first allocation: later prefill chunks
// and decode tokens add only the blocks their tokens fill, and releasing the sequence returns
// every block it held, fixed ones included (INV-4).
func TestSequenceFixedBlocks_ChargedOnceAndFreedWithTheSequence(t *testing.T) {
	kvc := NewKVCacheState(64, 4)
	kvc.SequenceFixedBlocks = 3
	req := promptRequest("a", 10)

	if !kvc.AllocateKVBlocks(req, 0, 6, nil) {
		t.Fatal("first chunk refused")
	}
	if got := kvc.UsedBlocks(); got != 2+3 {
		t.Fatalf("after a 6-token chunk: %d blocks used, want 2 token blocks + 3 fixed", got)
	}
	req.ProgressIndex = 6
	if !kvc.AllocateKVBlocks(req, 6, 10, nil) {
		t.Fatal("second chunk refused")
	}
	req.ProgressIndex = 10
	for i := 0; i < 3; i++ {
		if !kvc.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, nil) {
			t.Fatalf("decode %d refused", i)
		}
		req.ProgressIndex++
	}
	if got := kvc.UsedBlocks(); got != 4+3 {
		t.Fatalf("13 tokens: %d blocks used, want 4 token blocks + 3 fixed", got)
	}
	for _, id := range kvc.FixedBlocks["a"] {
		if b := kvc.Blocks[id]; len(b.Tokens) != 0 || b.Hash != "" {
			t.Errorf("fixed block %d holds tokens %v / hash %q; it must hold neither", id, b.Tokens, b.Hash)
		}
	}
	kvc.ReleaseKVBlocks(req)
	if got := kvc.UsedBlocks(); got != 0 {
		t.Errorf("after release: %d blocks still used", got)
	}
	if len(kvc.FixedBlocks) != 0 {
		t.Errorf("after release: fixed blocks still recorded: %v", kvc.FixedBlocks)
	}
	assertBlockConservation(t, kvc)
}

// A refusal for want of the fixed blocks mutates nothing (check-then-act): the request holds
// no blocks afterwards, and a pool with room for the tokens but not the fixed term refuses.
func TestSequenceFixedBlocks_RefusalIsSideEffectFree(t *testing.T) {
	kvc := NewKVCacheState(4, 4)
	kvc.SequenceFixedBlocks = 2
	req := promptRequest("a", 12) // 3 token blocks + 2 fixed > 4
	if kvc.AllocateKVBlocks(req, 0, 12, nil) {
		t.Fatal("admitted a sequence whose fixed term does not fit")
	}
	if kvc.UsedBlocks() != 0 || len(kvc.RequestMap) != 0 || len(kvc.FixedBlocks) != 0 {
		t.Errorf("a refused allocation left state behind: used %d, map %v, fixed %v",
			kvc.UsedBlocks(), kvc.RequestMap, kvc.FixedBlocks)
	}
	assertBlockConservation(t, kvc)
}

// A fixed block is taken off the free list, so a prefix cached in a freed block must not be
// clobbered by it while that block is being claimed as a prefix hit in the same allocation.
func TestSequenceFixedBlocks_DoNotTakeTheCachedPrefixBeingClaimed(t *testing.T) {
	kvc := NewKVCacheState(5, 4)
	first := promptRequest("a", 8)
	if !kvc.AllocateKVBlocks(first, 0, 8, nil) {
		t.Fatal("first request refused")
	}
	filler := promptRequest("g", 12)
	if !kvc.AllocateKVBlocks(filler, 0, 12, nil) {
		t.Fatal("filler refused")
	}
	// Free list, head first: a's two hashed prefix blocks, then the filler's three.
	kvc.ReleaseKVBlocks(first)
	kvc.ReleaseKVBlocks(filler)

	again := &sim.Request{ID: "b", InputTokens: append([]sim.TokenID(nil), first.InputTokens...)}
	kvc.SequenceFixedBlocks = 2
	cached := kvc.GetCachedBlocks(again.InputTokens)
	if len(cached) != 2 {
		t.Fatalf("expected the 2-block prefix cached, got %v", cached)
	}
	again.InputTokens = append(again.InputTokens, 7, 7) // one uncached tail block
	if !kvc.AllocateKVBlocks(again, 8, 10, cached) {
		t.Fatal("the cached request was refused") // 2 cached + 1 tail + 2 fixed = 5 blocks
	}
	for _, c := range cached {
		if b := kvc.Blocks[c]; b.Hash == "" || b.RefCount != 1 {
			t.Errorf("cached block %d lost its prefix (hash %q, refcount %d)", c, b.Hash, b.RefCount)
		}
	}
	for _, id := range kvc.FixedBlocks["b"] {
		for _, c := range cached {
			if id == c {
				t.Fatalf("fixed block %d is the cached prefix block it was claiming", id)
			}
		}
	}
	assertBlockConservation(t, kvc)
}
