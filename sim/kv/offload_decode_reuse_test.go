package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// buildDecodeReq drives a request through the offload chain's real prefill + one-token-
// per-step decode path (the way the simulator does), so decode blocks are formed and
// hashed exactly as in a live run — no synthetic hashes.
func buildDecodeReq(gpu *KVCacheState, oc *OffloadCache, id string, prompt, output []sim.TokenID) *sim.Request {
	req := &sim.Request{ID: id, InputTokens: prompt, OutputTokens: output}
	oc.SetClock(1)
	oc.AllocateKVBlocks(req, 0, int64(len(prompt)), nil) // prefill
	req.ProgressIndex = int64(len(prompt))
	for i := range output { // decode, one token per step
		si := int64(len(prompt)) + int64(i)
		oc.AllocateKVBlocks(req, si, si+1, nil)
		req.ProgressIndex = si + 1
	}
	return req
}

// CORRECTION (Deviation Log): the issue's premise that "decode blocks currently get no
// hash" is incorrect for block_size > 1. A decode block is completed one token at a time
// via the partial-fill path (cache.go:288-305), which hashes it prefix-consistently with
// NO ProgressIndex guard (the guard at cache.go:352 lives in the new-block path, which
// never forms a full decode block since decode advances one token per step). This test
// pins that fact — decode offload's reuse depends on it, so a regression must fail loudly.
func TestOffload_DecodeBlockGetsPrefixConsistentHash(t *testing.T) {
	gpu := NewKVCacheState(64, 2) // bs = 2
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 0))
	_ = buildDecodeReq(gpu, oc, "r", []sim.TokenID{1, 2}, []sim.TokenID{9, 10})

	ids := gpu.RequestMap["r"]
	if len(ids) != 2 {
		t.Fatalf("expected 2 full blocks (1 prompt + 1 decode), got %d", len(ids))
	}
	// The decode block's hash must equal the prefix-chained key a later request derives
	// for the same content — this is what makes cross-request reuse possible.
	want := blockKeysFor([]sim.TokenID{1, 2, 9, 10}, 2) // want[1] = decode block key
	if got := gpu.Blocks[ids[1]].Hash; got != string(want[1]) {
		t.Fatalf("decode block must carry the prefix-consistent hash %q, got %q", want[1], got)
	}
}

// BC-4 (offload side) + BC-2: the offloadable-token clamp is the single policy switch —
// under prompt-only (default) a full decode block is NOT offered to the CPU tier; under
// offload_prompt_only=false it IS.
func TestOffload_DecodeOffload_ResidencyByPolicy(t *testing.T) {
	build := func(promptOnly bool) *OffloadCache {
		gpu := NewKVCacheState(64, 2)
		cfg := enabledOffloadCfg(1<<20, 4096, 0)
		cfg.OffloadPromptOnly = promptOnly
		oc := NewOffloadCache(gpu, cfg)
		req := buildDecodeReq(gpu, oc, "r", []sim.TokenID{1, 2}, []sim.TokenID{9, 10})
		oc.MirrorToCPU([]*sim.Request{req})
		return oc
	}
	decodeKey := blockKeysFor([]sim.TokenID{1, 2, 9, 10}, 2)[1]

	ocTrue := build(true)
	if ocTrue.cpu.usedCount() != 1 || ocTrue.cpu.lookup(decodeKey) == cpuHit {
		t.Fatalf("prompt-only: only the prompt block offloads; decode block must NOT be CPU-resident (used=%d)", ocTrue.cpu.usedCount())
	}
	ocFalse := build(false)
	if ocFalse.cpu.usedCount() != 2 || ocFalse.cpu.lookup(decodeKey) != cpuHit {
		t.Fatalf("!prompt-only: the decode block must be CPU-resident too (used=%d)", ocFalse.cpu.usedCount())
	}
}

// BC-4 (full done-when scenario): a multi-turn workload where turn N+1's INPUT contains
// turn N's OUTPUT. After turn N's GPU blocks are evicted, turn N+1 reloads the reused
// region from the CPU tier only when decode KV was offloaded — so hit-rate reflects the
// policy. Under prompt-only (default) the output tokens produce NO CPU hit (1 reload =
// the prompt); under offload_prompt_only=false they DO (2 reloads = prompt + output).
func TestOffload_MultiTurn_OutputReuse_HitByPolicy(t *testing.T) {
	reloadsForTurnN1 := func(promptOnly bool) int64 {
		gpu := NewKVCacheState(6, 2) // small GPU so a filler can evict turn N's blocks
		cfg := enabledOffloadCfg(1<<20, 4096, 0)
		cfg.OffloadPromptOnly = promptOnly
		oc := NewOffloadCache(gpu, cfg)

		// Turn N: input [1,2], output [9,10] -> block0=[1,2] (prompt), block1=[9,10] (decode).
		reqN := buildDecodeReq(gpu, oc, "N", []sim.TokenID{1, 2}, []sim.TokenID{9, 10})
		oc.MirrorToCPU([]*sim.Request{reqN})
		oc.ReleaseKVBlocks(reqN)

		// Evict turn N's GPU blocks: one filler consuming all 6 GPU blocks reuses them
		// (popFreeBlock lazy-deletes their hashes). Release it so turn N+1 has room.
		filler := &sim.Request{ID: "F", InputTokens: []sim.TokenID{20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}}
		oc.SetClock(2)
		oc.AllocateKVBlocks(filler, 0, 12, nil)
		oc.ReleaseKVBlocks(filler)

		before := oc.reloadCount
		// Turn N+1: input [1,2,9,10] contains turn N's output [9,10].
		reqN1 := &sim.Request{ID: "N1", InputTokens: []sim.TokenID{1, 2, 9, 10}}
		oc.SetClock(100)
		cached := oc.GetCachedBlocks(reqN1.FullInputTokens())
		if len(cached) != 0 {
			t.Fatalf("precondition: turn N's blocks must be evicted from GPU, but %d prefix blocks are still cached", len(cached))
		}
		oc.AllocateKVBlocks(reqN1, 0, 4, cached)
		return oc.reloadCount - before
	}

	if got := reloadsForTurnN1(true); got != 1 {
		t.Fatalf("prompt-only: turn N+1 must reload only the prompt block from CPU (1), got %d", got)
	}
	if got := reloadsForTurnN1(false); got != 2 {
		t.Fatalf("!prompt-only: turn N+1 must reload prompt AND reused-output blocks from CPU (2), got %d", got)
	}
}

// BC-5 (INV-6 isolation): MirrorToCPU is a pure CONSUMER of GPU block hashes — under
// EITHER policy it never adds an entry to gpu.HashToBlock. Decode blocks are hashed by
// the GPU tier's own partial-fill path (cache.go), never by the offload path, so the
// offload chain cannot perturb GPU-tier prefix-cache behavior. This underpins the
// byte-identity of offload-disabled and prompt-only runs.
func TestOffload_MirrorToCPU_DoesNotMutateGPUHashes(t *testing.T) {
	for _, promptOnly := range []bool{true, false} {
		gpu := NewKVCacheState(64, 2)
		cfg := enabledOffloadCfg(1<<20, 4096, 1)
		cfg.OffloadPromptOnly = promptOnly
		oc := NewOffloadCache(gpu, cfg)
		req := buildDecodeReq(gpu, oc, "r", []sim.TokenID{1, 2}, []sim.TokenID{9, 10})

		before := len(gpu.HashToBlock)
		oc.MirrorToCPU([]*sim.Request{req})
		if after := len(gpu.HashToBlock); after != before {
			t.Fatalf("promptOnly=%v: MirrorToCPU must not add GPU hashes (before=%d after=%d)", promptOnly, before, after)
		}
	}
}

// Documented limitation (regression guard): at block_size == 1, decode blocks take the
// GUARDED new-block path (cache.go:352, `req.ProgressIndex < req.InputLen()`) instead of the
// partial-fill path, so they never receive a hash and decode-KV offload is inert even under
// offload_prompt_only=false. This test pins that behavior — if a future change to cache.go's
// decode-hash guard makes block_size==1 decode blocks hashable, BC-4 would silently change and
// this test would flag it.
func TestOffload_DecodeOffload_InertAtBlockSizeOne(t *testing.T) {
	gpu := NewKVCacheState(64, 1) // bs = 1 -> decode blocks are full via the guarded new-block path
	cfg := enabledOffloadCfg(1<<20, 4096, 0)
	cfg.OffloadPromptOnly = false // decode-offload requested...
	oc := NewOffloadCache(gpu, cfg)
	req := buildDecodeReq(gpu, oc, "r", []sim.TokenID{1}, []sim.TokenID{9}) // 1 prompt block, 1 decode block

	ids := gpu.RequestMap["r"]
	if len(ids) != 2 {
		t.Fatalf("expected 1 prompt + 1 decode block, got %d", len(ids))
	}
	if gpu.Blocks[ids[1]].Hash != "" {
		t.Fatalf("at block_size==1 the decode block must be unhashed (guarded new-block path)")
	}
	oc.MirrorToCPU([]*sim.Request{req})
	// ...but only the prompt block is offloaded; the unhashed decode block is inert.
	if oc.cpu.usedCount() != 1 {
		t.Fatalf("block_size==1 decode offload must be inert (only the prompt block offloads), got %d", oc.cpu.usedCount())
	}
}
