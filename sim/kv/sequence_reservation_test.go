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

// held is every block a request holds: its token blocks, spare slot blocks and fixed blocks.
func held(kvc *KVCacheState, id string) int64 {
	return int64(len(kvc.RequestMap[id]) + len(kvc.SpareBlocks[id]) + len(kvc.FixedBlocks[id]))
}

// #1627: a speculating request holds blocks for vLLM's slot count -- its computed tokens,
// everything scheduled this step (the whole verify width) and the drafter's lookahead -- not
// for the tokens it has stored. Over random prefill chunks and verify-width decode steps that
// advance by a random accepted count, after every successful allocation the request holds
// exactly max(token blocks, ceil((end + lookahead) / block)) besides its fixed blocks; a
// refusal changes nothing; the pool is conserved throughout (INV-4) and a release returns
// every block.
func TestSpeculativeSlots_HoldTheVerifyWidthAndLookahead(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		bs := rapid.Int64Range(1, 16).Draw(rt, "blockSize")
		k := rapid.Int64Range(1, 6).Draw(rt, "k")
		lookahead := rapid.SampledFrom([]int64{0, k, k + 1}).Draw(rt, "lookahead")
		total := rapid.Int64Range(4, 256).Draw(rt, "total")
		fixed := rapid.Int64Range(0, 2).Draw(rt, "fixed")
		kvc := NewKVCacheState(total, bs)
		kvc.SequenceFixedBlocks = fixed
		kvc.Speculative = sim.KVSpeculation{Enabled: true, LookaheadTokens: lookahead}

		input := rapid.IntRange(1, 40).Draw(rt, "input")
		output := rapid.IntRange(1, 40).Draw(rt, "output")
		req := promptRequest("s", input)
		req.OutputTokens = make([]sim.TokenID, output)
		end := int64(input + output - 1)
		for step := 0; req.ProgressIndex < end && step < 200; step++ {
			var hi, advance int64
			if req.ProgressIndex < int64(input) {
				hi = min(req.ProgressIndex+rapid.Int64Range(1, 12).Draw(rt, "chunk"), int64(input))
				advance = hi - req.ProgressIndex
			} else {
				hi = req.ProgressIndex + k + 1 // the verify width is scheduled
				advance = min(rapid.Int64Range(1, k+1).Draw(rt, "accepted"), end-req.ProgressIndex)
			}
			before := held(kvc, "s")
			freeBefore := kvc.countFreeBlocks()
			if !kvc.AllocateKVBlocks(req, req.ProgressIndex, hi, nil) {
				if held(kvc, "s") != before || kvc.countFreeBlocks() != freeBefore {
					rt.Fatalf("a refused allocation changed the hold %d->%d or free %d->%d",
						before, held(kvc, "s"), freeBefore, kvc.countFreeBlocks())
				}
				break
			}
			if err := kvc.verifyBlockConservation(); err != nil {
				rt.Fatal(err)
			}
			tokenBlocks := int64(len(kvc.RequestMap["s"]))
			want := max(tokenBlocks, (hi+lookahead+bs-1)/bs) + fixed
			if got := held(kvc, "s"); got != want {
				rt.Fatalf("step %d to end %d (lookahead %d, block %d): holds %d blocks, want %d",
					step, hi, lookahead, bs, got, want)
			}
			if got := kvc.UsedBlocks(); got != want {
				rt.Fatalf("pool reports %d used, the request holds %d", got, want)
			}
			req.ProgressIndex += advance
		}
		kvc.ReleaseKVBlocks(req)
		if kvc.UsedBlocks() != 0 || len(kvc.SpareBlocks) != 0 {
			rt.Fatalf("release left %d blocks used, spare %v", kvc.UsedBlocks(), kvc.SpareBlocks)
		}
		if err := kvc.verifyBlockConservation(); err != nil {
			rt.Fatal(err)
		}
	})
}

// Spare slot blocks are the first a request's next tokens use, so the reservation never
// double-counts: a decode that fills a block draws it from the request's own spares, leaving
// the free list untouched.
func TestSpeculativeSlots_TokensFillTheReservedBlocksFirst(t *testing.T) {
	kvc := NewKVCacheState(16, 4)
	kvc.Speculative = sim.KVSpeculation{Enabled: true, LookaheadTokens: 5}
	req := promptRequest("s", 4)
	req.OutputTokens = make([]sim.TokenID, 20)
	if !kvc.AllocateKVBlocks(req, 0, 4, nil) {
		t.Fatal("prefill refused")
	}
	// 4 prompt tokens + 5 lookahead slots = 9 slots = 3 blocks: 1 token block, 2 spare.
	if got := held(kvc, "s"); got != 3 {
		t.Fatalf("after prefill holds %d blocks, want 3", got)
	}
	spare := append([]int64(nil), kvc.SpareBlocks["s"]...)
	req.ProgressIndex = 4
	free := kvc.countFreeBlocks()
	// A decode at 4 scheduling 1 token: slots to 4+1+5 = 10, still 3 blocks; its token needs a
	// new block (the prompt block is full) and takes a spare.
	if !kvc.AllocateKVBlocks(req, 4, 5, nil) {
		t.Fatal("decode refused")
	}
	if kvc.countFreeBlocks() != free {
		t.Errorf("the decode took %d blocks off the free list; its spare should have served", free-kvc.countFreeBlocks())
	}
	if ids := kvc.RequestMap["s"]; ids[len(ids)-1] != spare[0] {
		t.Errorf("the decode token landed in block %d, not the first spare %d", ids[len(ids)-1], spare[0])
	}
}

// A prefix hash names one block in the index at a time: when a fully cached prompt
// recomputes its last block (#1943) the new block takes over the hash while the old, free
// copy still carries it. Reusing that stale copy must not drop the live mapping -- vLLM keys
// cached_block_hash_to_block by block id and evicts only the entry for the block it pops
// (vllm/v1/core/block_pool.py _maybe_evict_cached_block at v0.31.0). Two fully hit prompts,
// then a third, longer one with the same prefix still hits both of its blocks.
func TestReusingAStaleHashedBlockKeepsTheLiveMapping(t *testing.T) {
	kvc := NewKVCacheState(6, 4)
	prompt := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	a := &sim.Request{ID: "a", InputTokens: append([]sim.TokenID(nil), prompt...)}
	if !kvc.AllocateKVBlocks(a, 0, 8, nil) {
		t.Fatal("a refused")
	}
	kvc.ReleaseKVBlocks(a)

	// b hits only its first block (the hit is capped at InputLen-1) and recomputes the
	// second into a fresh block, which now owns the second block's hash.
	b := &sim.Request{ID: "b", InputTokens: append([]sim.TokenID(nil), prompt...)}
	cached := kvc.GetCachedBlocks(prompt)[:1]
	if !kvc.AllocateKVBlocks(b, 4, 8, cached) {
		t.Fatal("b refused")
	}
	live := kvc.RequestMap["b"][1]
	kvc.ReleaseKVBlocks(b)

	// A filler takes the stale copy of the second block off the free list.
	filler := promptRequest("filler", 16)
	if !kvc.AllocateKVBlocks(filler, 0, 16, nil) {
		t.Fatal("filler refused")
	}

	c := append(append([]sim.TokenID(nil), prompt...), 9, 10, 11, 12)
	got := kvc.GetCachedBlocks(c)
	if len(got) != 2 || got[1] != live {
		t.Errorf("a prompt extending the cached one hits %v; want both blocks, the second being %d", got, live)
	}
	assertBlockConservation(t, kvc)
}

// freeListIDs is the GPU free list, head first.
func freeListIDs(kvc *KVCacheState) []int64 {
	var ids []int64
	for b := kvc.FreeHead; b != nil; b = b.NextFree {
		ids = append(ids, b.ID)
	}
	return ids
}

// cpuResidentPrompt puts an 8-token prompt's two blocks on the legacy CPU tier and evicts
// them from the GPU, so admitting it means a CPU->GPU reload.
func cpuResidentPrompt(t *testing.T, tiered *TieredKVCache, gpu *KVCacheState) []sim.TokenID {
	t.Helper()
	prompt := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8}
	r := &sim.Request{ID: "seed", InputTokens: prompt}
	if !tiered.AllocateKVBlocks(r, 0, 8, nil) {
		t.Fatal("seed refused")
	}
	for _, id := range gpu.RequestMap["seed"] {
		tiered.cpu.store(gpu.Blocks[id].Hash, append([]sim.TokenID(nil), gpu.Blocks[id].Tokens...))
	}
	tiered.ReleaseKVBlocks(r)
	for _, b := range gpu.Blocks { // the GPU copies are gone
		gpu.dropHash(b)
	}
	return prompt
}

// Review I1: an admission refused for want of its fixed blocks (#1941) is refused before the
// CPU tier is consulted, as vLLM's allocate_slots refuses before it touches a block: no
// reload, no transfer latency, no cache hit, the free list as it was.
func TestARefusedReloadAdmissionHasNoSideEffects(t *testing.T) {
	gpu := NewKVCacheState(6, 4)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 10)
	prompt := cpuResidentPrompt(t, tiered, gpu)
	gpu.SequenceFixedBlocks = 5 // 2 token blocks + 5 fixed > 6

	free, latency, hits, cpuHits := freeListIDs(gpu), tiered.PendingTransferLatency(), gpu.CacheHits, tiered.cpuHitCount
	req := &sim.Request{ID: "r", InputTokens: append([]sim.TokenID(nil), prompt...)}
	if tiered.AllocateKVBlocks(req, 0, 8, nil) {
		t.Fatal("admitted a sequence whose fixed blocks do not fit")
	}
	if got := freeListIDs(gpu); fmt.Sprint(got) != fmt.Sprint(free) {
		t.Errorf("free list reordered: %v -> %v", free, got)
	}
	if tiered.PendingTransferLatency() != latency || gpu.CacheHits != hits || tiered.cpuHitCount != cpuHits {
		t.Errorf("side effects: latency %d->%d, cache hits %d->%d, cpu hits %d->%d", latency,
			tiered.PendingTransferLatency(), hits, gpu.CacheHits, cpuHits, tiered.cpuHitCount)
	}
	if len(gpu.RequestMap) != 0 || len(gpu.FixedBlocks) != 0 {
		t.Errorf("the refused request holds blocks: %v %v", gpu.RequestMap, gpu.FixedBlocks)
	}
	assertBlockConservation(t, gpu)
}

// Review I2: a reload that covers a request's whole allocation still charges its fixed
// blocks and its speculative slots -- for a new request and for a running one -- as vLLM
// passes num_lookahead_tokens to every allocate_slots (scheduler.py:745-751, 1214).
func TestAFullyReloadedAllocationHoldsItsSlotsAndFixedBlocks(t *testing.T) {
	gpu := NewKVCacheState(32, 4)
	tiered := NewTieredKVCache(gpu, 10, 0.0, 1.0, 10)
	prompt := cpuResidentPrompt(t, tiered, gpu)
	gpu.SequenceFixedBlocks = 2
	gpu.Speculative = sim.KVSpeculation{Enabled: true, LookaheadTokens: 5}
	reloadsBefore := tiered.cpuHitCount

	req := &sim.Request{ID: "r", InputTokens: append([]sim.TokenID(nil), prompt...)}
	if !tiered.AllocateKVBlocks(req, 0, 8, nil) {
		t.Fatal("refused")
	}
	if tiered.cpuHitCount-reloadsBefore != 2 {
		t.Fatalf("expected the whole prompt reloaded from CPU, %d blocks were", tiered.cpuHitCount-reloadsBefore)
	}
	// 8 tokens + 5 lookahead = 13 slots = 4 blocks: 2 reloaded token blocks + 2 spare, + 2 fixed.
	if got := held(gpu, "r"); got != 4+2 {
		t.Errorf("a fully reloaded admission holds %d blocks, want 4 slot blocks + 2 fixed", got)
	}
	assertBlockConservation(t, gpu)
}

// Review I3: the slot reservation never reaches past max_model_len, as vLLM caps
// num_tokens_need_slot = min(computed + new + lookahead, max_model_len)
// (kv_cache_manager.py:536-538 at v0.31.0).
func TestSpeculativeSlotsStopAtMaxModelLen(t *testing.T) {
	for _, tt := range []struct {
		maxModelLen, want int64
	}{{0, 4}, {10, 3}, {13, 4}} { // 8 tokens + 5 lookahead = 13 slots
		kvc := NewKVCacheState(32, 4)
		kvc.Speculative = sim.KVSpeculation{Enabled: true, LookaheadTokens: 5, MaxModelLen: tt.maxModelLen}
		if !kvc.AllocateKVBlocks(promptRequest("r", 8), 0, 8, nil) {
			t.Fatal("refused")
		}
		if got := held(kvc, "r"); got != tt.want {
			t.Errorf("max_model_len %d: holds %d blocks, want %d", tt.maxModelLen, got, tt.want)
		}
	}
}
