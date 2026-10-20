package kv

import (
	"testing"

	"pgregory.net/rapid"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/internal/kvkey"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// chunkCache is an offload cache over a bs-token GPU pool of gpuBlocks blocks with
// blocks_per_chunk bpc, a cpuChunks-chunk CPU tier and the given secondary tiers.
func chunkCache(gpuBlocks, bs, bpc, cpuChunks int64, tiers int) *OffloadCache {
	cfg := enabledOffloadCfg(cpuChunks*bpc*4096, 4096, tiers)
	cfg.BlocksPerChunk, cfg.BlockSize = bpc, bpc*bs
	return NewOffloadCache(NewKVCacheState(gpuBlocks, bs), cfg)
}

func seqTokens(from, n int) []sim.TokenID {
	out := make([]sim.TokenID, n)
	for i := range out {
		out[i] = sim.TokenID(from + i)
	}
	return out
}

// vLLM stores whole chunks only, each keyed by its last block's hash
// (offloading/scheduler.py:418-436, :448-479 at v0.31.0): a 13-block-and-a-bit prompt at
// blocks_per_chunk 4 offloads chunks 0-2 (blocks 0-11) under the hashes of blocks 3, 7 and
// 11, and never block 12. Each chunk is one CPU entry; the request's chunks go to each tier as
// one store job (tiering/manager.py:737-747), one file per chunk.
func TestOffloadChunks_StoreWholeChunksKeyedByTheirLastBlock(t *testing.T) {
	oc := chunkCache(64, 2, 4, 16, 1)
	tokens := seqTokens(1, 27)
	keys := blockKeysFor(tokens, 2)
	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(1)
	if !admitOffload(oc, req, 0, int64(len(tokens)), nil) {
		t.Fatal("allocation failed")
	}
	oc.MirrorToCPU([]*sim.Request{req})
	if got := oc.cpu.usedCount(); got != 3 {
		t.Fatalf("CPU holds %d entries, want 3 chunks", got)
	}
	for _, i := range []int{3, 7, 11} {
		if oc.cpu.lookup(keys[i]) != cpuHit {
			t.Errorf("chunk ending at block %d is not stored under that block's hash", i)
		}
	}
	for _, i := range []int{0, 1, 2, 12} {
		if oc.cpu.lookup(keys[i]) != cpuMiss {
			t.Errorf("block %d has its own CPU entry; only chunk-final hashes are keys", i)
		}
	}
	if got := oc.secondary[0].activeJobs(kvtransfer.Write); got != 1 {
		t.Errorf("%d cascade writes, want one store job for the request's 3 chunks", got)
	}
}

// vLLM loads whole chunks from the chunk holding the GPU-cached boundary
// (start_chunk_idx = num_computed_tokens // tokens_per_chunk, scheduler.py:841), and
// loads nothing when that comes to less than one chunk (:818-821).
func TestOffloadChunks_ReloadWholeChunksAndNeverLessThanOne(t *testing.T) {
	const bs, bpc = 2, 4
	tokens := seqTokens(1, 12*bs+1) // 12 full blocks = 3 chunks
	keys := blockKeysFor(tokens, bs)
	// GPU holds blocks 0-5 (a released request's cached prefix).
	setup := func(cpuChunks ...int) *OffloadCache {
		oc := chunkCache(64, bs, bpc, 16, 0)
		warm := &sim.Request{ID: "warm", InputTokens: tokens[:6*bs]}
		admitOffload(oc, warm, 0, 6*bs, nil)
		oc.ReleaseKVBlocks(warm)
		for _, c := range cpuChunks {
			oc.cpu.store(keys[(c+1)*bpc-1])
		}
		return oc
	}
	req := &sim.Request{ID: "r", InputTokens: tokens}

	oc := setup(1) // chunk 1 = blocks 4-7: only blocks 6, 7 are new, under one chunk
	if end, ok := oc.ReloadablePrefixEnd(req, 6*bs); ok || end != 6*bs {
		t.Errorf("half a chunk past the GPU prefix: reloadable to %d (ok=%v), want no load", end, ok)
	}
	oc = setup(1, 2) // blocks 6-11: six blocks, at least one chunk
	if end, ok := oc.ReloadablePrefixEnd(req, 6*bs); !ok || end != 12*bs {
		t.Errorf("chunks 1-2 resident: reloadable to %d (ok=%v), want %d", end, ok, 12*bs)
	}
	if !admitOffload(oc, req, 6*bs, int64(len(tokens)), oc.GetCachedBlocks(tokens)[:6]) {
		t.Fatal("allocation after reload failed")
	}
	if got := len(oc.GetCachedBlocks(tokens)); got != 12 {
		t.Errorf("after the reload %d blocks are GPU-cached, want 12", got)
	}
	oc = setup(2) // chunk 1 absent: the run stops at the GPU prefix
	if _, ok := oc.ReloadablePrefixEnd(req, 6*bs); ok {
		t.Error("a CPU chunk past a missing chunk was counted as reloadable")
	}
}

// The CPU tier evicts whole chunks in LRU order: a two-chunk tier offered three chunks
// keeps the two most recent.
func TestOffloadChunks_LRUEvictsWholeChunks(t *testing.T) {
	oc := chunkCache(64, 2, 4, 2, 0)
	tokens := seqTokens(1, 24)
	keys := blockKeysFor(tokens, 2)
	req := &sim.Request{ID: "r", InputTokens: tokens}
	admitOffload(oc, req, 0, 24, nil)
	oc.MirrorToCPU([]*sim.Request{req})
	if oc.cpu.usedCount() != 2 || oc.cpu.lookup(keys[3]) != cpuMiss ||
		oc.cpu.lookup(keys[7]) != cpuHit || oc.cpu.lookup(keys[11]) != cpuHit {
		t.Errorf("a two-chunk tier offered chunks 0-2 holds used=%d, chunk0=%v chunk1=%v chunk2=%v; want chunks 1-2",
			oc.cpu.usedCount(), oc.cpu.lookup(keys[3]), oc.cpu.lookup(keys[7]), oc.cpu.lookup(keys[11]))
	}
}

// Properties over random prompts that share prefixes, chunk widths 1-4 and random CPU
// residency of chunk keys:
//   - the pure query (ReloadablePrefixEnd) and the commit (AllocateKVBlocks) agree on the
//     cached prefix (with GPU room);
//   - a load ends on a chunk boundary and spans at least one chunk;
//   - the CPU tier never holds more chunks than it has room for, and every key it holds is
//     a chunk-final block hash.
func TestOffloadChunks_LoadsAreWholeChunksAndTheQueryIsTheCommit(t *testing.T) {
	rapid.Check(t, func(rt *rapid.T) {
		const bs = 2
		bpc := rapid.Int64Range(1, 4).Draw(rt, "bpc")
		cpuChunks := rapid.Int64Range(1, 8).Draw(rt, "cpuChunks")
		oc := chunkCache(256, bs, bpc, cpuChunks, 0)
		base := seqTokens(1, 40)
		chunkKeys := map[kvkey.BlockKey]bool{}
		for i, k := range blockKeysFor(base, bs) {
			if int64(i+1)%bpc == 0 {
				chunkKeys[k] = true
			}
		}
		for step := 0; step < rapid.IntRange(1, 6).Draw(rt, "steps"); step++ {
			n := rapid.IntRange(1, len(base)).Draw(rt, "len")
			tokens := base[:n]
			id := string(rune('a' + step))
			if rapid.Bool().Draw(rt, "warmGPU") {
				w := &sim.Request{ID: id + "w", InputTokens: tokens[:rapid.IntRange(0, n).Draw(rt, "warm")]}
				if w.InputLen() > 0 {
					admitOffload(oc, w, 0, w.InputLen(), nil)
					oc.MirrorToCPU([]*sim.Request{w})
					oc.ReleaseKVBlocks(w)
				}
			}
			req := &sim.Request{ID: id, InputTokens: tokens}
			cached := oc.GetCachedBlocks(tokens)
			start := int64(len(cached)) * bs
			end, ok := oc.ReloadablePrefixEnd(req, start)
			if ok {
				if endBlock, startBlock := end/bs, start/bs; endBlock%bpc != 0 || endBlock-startBlock < bpc {
					rt.Fatalf("bpc %d: load [%d, %d) is not whole chunks of at least one chunk", bpc, start, end)
				}
			} else if end != start {
				rt.Fatalf("no load, yet reloadable end %d != start %d", end, start)
			}
			if !admitOffload(oc, req, start, req.InputLen(), cached) {
				rt.Fatal("allocation with ample GPU room failed")
			}
			if got := int64(len(oc.GetCachedBlocks(tokens))) * bs; got < end {
				rt.Fatalf("bpc %d: query promised a cached prefix to %d, commit reached %d", bpc, end, got)
			}
			oc.MirrorToCPU([]*sim.Request{req})
			oc.ReleaseKVBlocks(req)
			if oc.cpu.usedCount() > cpuChunks {
				rt.Fatalf("CPU tier holds %d chunks, capacity %d", oc.cpu.usedCount(), cpuChunks)
			}
			for k := range oc.cpu.blocks {
				if !chunkKeys[k] {
					rt.Fatalf("bpc %d: CPU tier holds a key that is not a chunk-final block hash", bpc)
				}
			}
		}
	})
}

// Metamorphic: a chunk-aligned prompt fully offloaded and then evicted from the GPU reloads to
// the same cached prefix at every chunk width, and the router's CPU-tier view counts the
// same blocks.
func TestOffloadChunks_AlignedPromptReloadsTheSameAtEveryWidth(t *testing.T) {
	const bs = 2
	tokens := seqTokens(1, 24*bs+1) // 24 blocks: whole chunks at widths 1, 2, 3, 4
	var want int64 = -1
	for _, bpc := range []int64{1, 2, 3, 4} {
		oc := chunkCache(32, bs, bpc, 64, 0)
		a := &sim.Request{ID: "a", InputTokens: tokens}
		admitOffload(oc, a, 0, a.InputLen(), nil)
		oc.MirrorToCPU([]*sim.Request{a})
		oc.ReleaseKVBlocks(a)
		flush := &sim.Request{ID: "flush", InputTokens: seqTokens(1000, 32*bs)}
		if !admitOffload(oc, flush, 0, flush.InputLen(), nil) {
			t.Fatal("flush allocation failed")
		}
		oc.ReleaseKVBlocks(flush)
		if got := len(oc.GetCachedBlocks(tokens)); got != 0 {
			t.Fatalf("bpc %d: the flush left %d blocks GPU-cached", bpc, got)
		}
		if got := oc.CPUTierPrefixBlockCount(tokens); got != 24 {
			t.Errorf("bpc %d: router CPU view counts %d blocks, want 24", bpc, got)
		}
		b := &sim.Request{ID: "b", InputTokens: tokens}
		end, _ := oc.ReloadablePrefixEnd(b, 0)
		if want < 0 {
			want = end
		} else if end != want {
			t.Errorf("bpc %d: reloadable to %d, want %d as at blocks_per_chunk 1", bpc, end, want)
		}
	}
}

// Every tier transfer moves whole chunks, each its own file: a cascade write of two chunks and
// a promotion read of one, at blocks_per_chunk 4, price each file at blocks_per_chunk x
// per-block bytes.
func TestOffloadChunks_TransfersMoveWholeChunks(t *testing.T) {
	cfg := enabledOffloadCfg(16*4*4096, 4096, 1)
	cfg.BlocksPerChunk, cfg.BlockSize = 4, 8
	var priced []int64
	cfg.Tiers[0].ServiceTime = func(_ bool, bytes int64, _ int) int64 { priced = append(priced, bytes); return 10 }
	oc := NewOffloadCache(NewKVCacheState(64, 2), cfg)
	tokens := seqTokens(1, 17) // two chunks
	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(1)
	admitOffload(oc, req, 0, int64(len(tokens)), nil)
	oc.MirrorToCPU([]*sim.Request{req})
	keys := blockKeysFor(seqTokens(500, 16), 2)
	oc.secondary[0].store(keys[3])
	oc.consultAndReload(seqTokens(500, 16), 0, "")
	if len(priced) != 2 {
		t.Fatalf("priced %d jobs, want one 2-chunk write and one 1-chunk read", len(priced))
	}
	for _, b := range priced {
		if b != 4*4096 {
			t.Errorf("a transfer moved %d bytes, want one chunk (%d)", b, 4*4096)
		}
	}
}
