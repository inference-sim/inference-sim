package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kvtransfer"
)

// BC-C7a + BC-C3(pin): MirrorToCPU stores each new full block and cascades it to
// EVERY secondary tier, pinning the CPU block once per in-flight write so it is
// non-evictable until the writes complete.
func TestOffload_CascadePinsAllTiers(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 2)) // 2 secondary tiers
	tokens := []sim.TokenID{1, 2, 3, 4}                           // 2 prompt blocks
	keys := blockKeysFor(tokens, 2)

	req := &sim.Request{ID: "r", InputTokens: tokens}
	oc.SetClock(100)
	oc.AllocateKVBlocks(req, 0, 4, nil) // 2 GPU blocks with hashes
	oc.MirrorToCPU([]*sim.Request{req})

	// Both blocks landed in CPU and are pinned by 2 in-flight writes each -> evictable 0.
	if oc.cpu.lookup(keys[0]) != cpuHit || oc.cpu.lookup(keys[1]) != cpuHit {
		t.Fatalf("mirrored blocks must be CPU-resident HITs")
	}
	if oc.cpu.evictableCount() != 0 {
		t.Fatalf("cascade writes must pin every mirrored block (evictable 0), got %d", oc.cpu.evictableCount())
	}
	// 2 blocks × 2 tiers = 4 Write jobs; per tier, 2 write jobs in service.
	if w0, w1 := oc.station.ActiveJobs(0, kvtransfer.Write), oc.station.ActiveJobs(1, kvtransfer.Write); w0 != 2 || w1 != 2 {
		t.Fatalf("each tier must have 2 cascade Write jobs, got tier0=%d tier1=%d", w0, w1)
	}
}

// BC-C4 + I3: when the CPU tier is full and every block is pinned, MirrorToCPU
// SKIPS the store (counted) rather than force-evicting a locked block.
func TestOffload_MirrorSkipsWhenFullPinned(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(8192, 4096, 1)) // CPU capacity 2
	// Pre-fill + pin both CPU slots.
	filler := blockKeysFor([]sim.TokenID{90, 91, 92, 93}, 2)
	oc.cpu.store(filler[0])
	oc.cpu.store(filler[1])
	oc.cpu.pin(filler[0])
	oc.cpu.pin(filler[1])

	req := &sim.Request{ID: "r", InputTokens: []sim.TokenID{1, 2, 3, 4}}
	oc.SetClock(10)
	oc.AllocateKVBlocks(req, 0, 4, nil)
	oc.MirrorToCPU([]*sim.Request{req})

	if oc.mirrorSkipped != 2 {
		t.Fatalf("both mirror stores must be skipped when CPU is full-and-pinned, mirrorSkipped=%d", oc.mirrorSkipped)
	}
	// The pinned fillers survive; nothing new was force-stored.
	if oc.cpu.usedCount() != 2 {
		t.Fatalf("CPU must still hold exactly the 2 pinned fillers, used=%d", oc.cpu.usedCount())
	}
}

// OffloadPromptOnly (vLLM default TRUE): decode-generated blocks are NOT offloaded.
// The offloadable-token clamp truncates the computed-KV count to InputLen() when
// prompt-only, so a full, hashed block beyond the prompt range (index >= InputLen/bs)
// is offloaded only when OffloadPromptOnly is false.
func TestOffload_PromptOnlySkipsDecode(t *testing.T) {
	tokens := []sim.TokenID{1, 2, 3, 4} // InputLen 4 -> 2 prompt blocks

	build := func(promptOnly bool) *OffloadCache {
		gpu := NewKVCacheState(64, 2)
		cfg := enabledOffloadCfg(1<<20, 4096, 1)
		cfg.OffloadPromptOnly = promptOnly
		oc := NewOffloadCache(gpu, cfg)
		req := &sim.Request{ID: "r", InputTokens: tokens}
		oc.SetClock(1)
		oc.AllocateKVBlocks(req, 0, 4, nil) // 2 prompt blocks (full, hashed)

		// Append a FULL, HASHED "decode" block (index 2) beyond the prompt so the
		// only thing gating its mirror is OffloadPromptOnly (not the Hash/full guard).
		db := gpu.popFreeBlock()
		db.Tokens = []sim.TokenID{5, 6}
		db.Hash = "decodeblockhash000000000000000000000000000000000000000000000000"
		db.RefCount = 1
		db.InUse = true
		gpu.HashToBlock[db.Hash] = db.ID
		gpu.RequestMap["r"] = append(gpu.RequestMap["r"], db.ID)

		oc.MirrorToCPU([]*sim.Request{req})
		return oc
	}

	if oc := build(true); oc.cpu.usedCount() != 2 {
		t.Fatalf("OffloadPromptOnly=true must mirror only the 2 prompt blocks, got %d", oc.cpu.usedCount())
	}
	if oc := build(false); oc.cpu.usedCount() != 3 {
		t.Fatalf("OffloadPromptOnly=false must mirror the decode block too (3), got %d", oc.cpu.usedCount())
	}
}

// BC-1: the offloadable amount is a SINGLE token-count clamp (min(computed, InputLen)
// when prompt-only) floor-divided into whole chunks — not a per-block classification.
// With computed KV strictly greater than InputLen, prompt-only offloads exactly
// floor(InputLen/tokensPerChunk) chunks while !prompt-only offloads every full block;
// the ONLY difference between the two is the InputLen truncation (the single decision
// point).
func TestOffload_OffloadableTokenClamp_SingleDecisionPoint(t *testing.T) {
	promptTokens := []sim.TokenID{1, 2, 3, 4} // InputLen 4 -> 2 prompt blocks at bs=2

	build := func(promptOnly bool) *OffloadCache {
		gpu := NewKVCacheState(64, 2)
		cfg := enabledOffloadCfg(1<<20, 4096, 1) // blocks_per_chunk 1 -> tokensPerChunk == bs
		cfg.OffloadPromptOnly = promptOnly
		oc := NewOffloadCache(gpu, cfg)
		req := &sim.Request{ID: "r", InputTokens: promptTokens}
		oc.SetClock(1)
		oc.AllocateKVBlocks(req, 0, 4, nil) // 2 prompt blocks (full, hashed)

		// Append TWO full, hashed "decode" blocks (indices 2 and 3) so computed KV
		// (4 blocks) strictly exceeds InputLen/bs (2). Their mirror is gated only by
		// the offloadable-token clamp, not the Hash/full guard.
		for i, toks := range [][]sim.TokenID{{5, 6}, {7, 8}} {
			db := gpu.popFreeBlock()
			db.Tokens = toks
			db.Hash = "decodeblockhash" + string(rune('A'+i)) + "0000000000000000000000000000000000000000000000000"
			db.RefCount = 1
			db.InUse = true
			gpu.HashToBlock[db.Hash] = db.ID
			gpu.RequestMap["r"] = append(gpu.RequestMap["r"], db.ID)
		}

		oc.MirrorToCPU([]*sim.Request{req})
		return oc
	}

	// prompt-only: min(4 blocks*2, InputLen 4)=4 tokens -> floor(4/2)=2 chunks.
	if oc := build(true); oc.cpu.usedCount() != 2 {
		t.Fatalf("prompt-only must offload floor(InputLen/tokensPerChunk)=2 chunks, got %d", oc.cpu.usedCount())
	}
	// !prompt-only: no clamp -> all 4 full hashed blocks offloaded.
	if oc := build(false); oc.cpu.usedCount() != 4 {
		t.Fatalf("!prompt-only must offload every full block (4), got %d", oc.cpu.usedCount())
	}
}

// BC-3: a prompt of 1.5 x tokens_per_chunk offloads exactly ONE chunk. The half-chunk
// tail block — even after a decode token completes AND hashes it via the partial-fill
// path (cache.go:288-305) — is outside the floor-divided offloadable range, so it is
// never offered for store. This is the truncate-then-floor-divide rule: the clamp keeps
// InputLen=3 tokens, and floor(3/2)=1 chunk.
func TestOffload_PromptOnly_PartialTail_OffloadsExactlyOneChunk(t *testing.T) {
	gpu := NewKVCacheState(64, 2) // bs = 2 -> tokens_per_chunk = 2 (blocks_per_chunk 1)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))

	// InputLen = 3 = 1.5 x tokens_per_chunk. OutputTokens supply the decode token that
	// completes the half-full tail block.
	req := &sim.Request{ID: "r", InputTokens: []sim.TokenID{1, 2, 3}, OutputTokens: []sim.TokenID{9}}
	oc.SetClock(1)
	oc.AllocateKVBlocks(req, 0, 3, nil) // prefill: block0 [1,2] full+hashed, block1 [3] partial
	req.ProgressIndex = 3               // prefill complete; next AllocateKVBlocks is a decode step
	oc.AllocateKVBlocks(req, 3, 4, nil) // decode: appends output[0]=9 -> block1 [3,9] full, hashed

	// The tail block IS full and hashed (partial-fill path fired)...
	keys := blockKeysFor([]sim.TokenID{1, 2, 3, 9}, 2) // keys[0]=block0, keys[1]=block1
	if _, hashed := gpu.HashToBlock[string(keys[1])]; !hashed {
		t.Fatalf("tail block completed by decode must be hashed via the partial-fill path")
	}

	oc.MirrorToCPU([]*sim.Request{req})

	// ...but only 1 chunk (block0) is offloaded; the tail (block1) is truncated away.
	if oc.cpu.usedCount() != 1 {
		t.Fatalf("1.5x-chunk prompt must offload exactly 1 chunk, got %d", oc.cpu.usedCount())
	}
	if oc.cpu.lookup(keys[0]) != cpuHit {
		t.Fatalf("the single full prompt chunk (block0) must be CPU-resident")
	}
	if oc.cpu.lookup(keys[1]) != cpuMiss {
		t.Fatalf("the half-chunk tail (block1) must NOT be offloaded (outside offloadable range)")
	}
}
