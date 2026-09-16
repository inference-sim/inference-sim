package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// enabledOffloadCfg builds a resolved, enabled offload config for tests.
// tiers = number of secondary fs tiers.
func enabledOffloadCfg(cpuBytes, perBlockBytes int64, tiers int) sim.KVOffloadConfig {
	cfg := sim.KVOffloadConfig{
		Enabled: true, CPUBytesToUse: cpuBytes, PerBlockBytes: perBlockBytes,
		BlockSize: 16, BlocksPerChunk: 1, TokensPerHash: 16,
		EvictionPolicy: "lru", OffloadPromptOnly: true,
	}
	for i := 0; i < tiers; i++ {
		cfg.Tiers = append(cfg.Tiers, sim.KVOffloadTier{
			Type: "fs", RootDir: "/mnt", NReadThreads: 16, NWriteThreads: 16,
			DirectIO: true, ReadBandwidth: 7000, WriteBandwidth: 5000, BaseLatency: 80,
		})
	}
	return cfg
}

func mustPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Fatalf("%s: expected panic, got none", name)
		}
	}()
	f()
}

// The factory selects the offload chain iff Offload.IsEnabled(); otherwise the
// legacy paths are unchanged (BC-N1: disabled offload never becomes an OffloadCache).
func TestNewKVStore_OffloadGating(t *testing.T) {
	// Disabled + no CPU blocks -> single-tier.
	single := NewKVStore(sim.KVCacheConfig{TotalKVBlocks: 64, BlockSizeTokens: 16}, 0)
	if _, ok := single.(*KVCacheState); !ok {
		t.Fatalf("disabled offload + no CPU blocks must be single-tier *KVCacheState, got %T", single)
	}
	// Disabled + legacy CPU blocks -> legacy TieredKVCache (unchanged).
	legacy := NewKVStore(sim.KVCacheConfig{TotalKVBlocks: 64, BlockSizeTokens: 16, KVCPUBlocks: 8, KVTransferBandwidth: 100}, 0)
	if _, ok := legacy.(*TieredKVCache); !ok {
		t.Fatalf("legacy KVCPUBlocks path must be *TieredKVCache, got %T", legacy)
	}
	// Enabled offload -> OffloadCache.
	off := NewKVStore(sim.KVCacheConfig{TotalKVBlocks: 64, BlockSizeTokens: 16, Offload: enabledOffloadCfg(1<<20, 4096, 1)}, 0)
	if _, ok := off.(*OffloadCache); !ok {
		t.Fatalf("enabled offload must be *OffloadCache, got %T", off)
	}
}

// Both offload models set at once is refused loudly (R1/R22).
func TestNewKVStore_BothOffloadModelsPanics(t *testing.T) {
	mustPanic(t, "both-set", func() {
		NewKVStore(sim.KVCacheConfig{
			TotalKVBlocks: 64, BlockSizeTokens: 16, KVCPUBlocks: 8, KVTransferBandwidth: 100,
			Offload: enabledOffloadCfg(1<<20, 4096, 1),
		}, 0)
	})
}

// H1 restrictions and the derived-field requirement are enforced loudly at
// construction (defense-in-depth; the CLI validates first).
func TestNewOffloadCache_Validation(t *testing.T) {
	gpu := NewKVCacheState(64, 16)

	mustPanic(t, "perBlockBytes<=0", func() {
		c := enabledOffloadCfg(1<<20, 0, 1)
		NewOffloadCache(gpu, c)
	})
	mustPanic(t, "blocksPerChunk>1", func() {
		c := enabledOffloadCfg(1<<20, 4096, 1)
		c.BlocksPerChunk = 2
		NewOffloadCache(gpu, c)
	})
	mustPanic(t, "arc eviction", func() {
		c := enabledOffloadCfg(1<<20, 4096, 1)
		c.EvictionPolicy = "arc"
		NewOffloadCache(gpu, c)
	})
	mustPanic(t, "cpu budget too small", func() {
		c := enabledOffloadCfg(100, 4096, 1) // 100 bytes < one 4096-byte block
		NewOffloadCache(gpu, c)
	})
	mustPanic(t, "disabled cfg", func() {
		NewOffloadCache(gpu, sim.KVOffloadConfig{})
	})

	// A valid config builds: capacity = cpu_bytes / per_block_bytes.
	oc := NewOffloadCache(gpu, enabledOffloadCfg(40960, 4096, 2)) // 10 CPU blocks, 2 tiers
	if oc.cpu.capacity != 10 {
		t.Fatalf("CPU capacity must be cpu_bytes/per_block_bytes=10, got %d", oc.cpu.capacity)
	}
	if len(oc.secondary) != 2 || oc.station == nil {
		t.Fatalf("2 secondary tiers must build 2 tiers + a station, got %d tiers station=%v", len(oc.secondary), oc.station)
	}

	// CPU-only offload (no secondary tiers) builds with a nil station.
	ocpu := NewOffloadCache(gpu, enabledOffloadCfg(40960, 4096, 0))
	if len(ocpu.secondary) != 0 || ocpu.station != nil {
		t.Fatalf("no secondary tiers must yield 0 tiers + nil station")
	}
}

// I1 (routing): OffloadCache must satisfy the SnapshotCachedBlocksFn capability the
// cluster router type-asserts, so routing keeps frozen-snapshot semantics rather
// than falling back to a live query.
func TestOffload_SnapshotCachedBlocksFn(t *testing.T) {
	var _ interface {
		SnapshotCachedBlocksFn() func([]sim.TokenID) int
	} = (*OffloadCache)(nil)

	gpu := NewKVCacheState(64, 16)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	fn := oc.SnapshotCachedBlocksFn()
	if fn == nil {
		t.Fatalf("SnapshotCachedBlocksFn must return a non-nil closure")
	}
	if n := fn([]sim.TokenID{1, 2, 3, 4}); n != 0 {
		t.Fatalf("empty cache snapshot must count 0 cached blocks, got %d", n)
	}
}

// #1699/#1706: the offload store reports, via a PURE pre-allocation query
// (ReloadReportingKVStore.ReloadablePrefixEnd), the token boundary to which a same-step
// CPU->GPU reload WOULD extend a new request's GPU-cached prefix — the analogue of vLLM's
// get_num_new_matched_tokens. Batch formation folds this boundary in BEFORE the chunk cap
// so the WHOLE reloadable prefix is credited (not just one committed chunk). The query
// mutates nothing (repeatable) and only counts the CPU-resident (same-step reloadable)
// run; a secondary-tier-only run stays on the H3 deferral path (not counted here).
func TestOffload_ReloadablePrefixEnd_ReportsBoundary(t *testing.T) {
	// Compile-time capability assertion.
	var _ sim.ReloadReportingKVStore = (*OffloadCache)(nil)

	t.Run("full CPU-resident prefix reports InputLen and is a pure repeatable query", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4} // 2 blocks, all CPU-resident
		keys := blockKeysFor(tokens, 2)
		oc.cpu.store(keys[0])
		oc.cpu.store(keys[1])

		req := &sim.Request{ID: "r", InputTokens: tokens}
		reloadableEnd, ok := oc.ReloadablePrefixEnd(req, 0)
		if !ok || reloadableEnd != 4 {
			t.Fatalf("full CPU-resident prefix must report reloadableEnd=4, ok=true; got %d ok=%v", reloadableEnd, ok)
		}
		// PURE query: a second call (no commit in between) must report the same boundary,
		// and it must not have mutated any tier (no CPU block reloaded to GPU).
		if again, ok := oc.ReloadablePrefixEnd(req, 0); !ok || again != 4 {
			t.Fatalf("ReloadablePrefixEnd must be a pure repeatable query; got %d ok=%v", again, ok)
		}
		if oc.reloadCount != 0 {
			t.Fatalf("a pure query must not perform any CPU->GPU reload; reloadCount=%d", oc.reloadCount)
		}
	})

	t.Run("partial CPU-resident prefix reports the reloadable boundary", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		// 3 blocks of input; only the first block is CPU-resident. The tail (blocks
		// 1..2) is a genuine miss.
		tokens := []sim.TokenID{1, 2, 3, 4, 5, 6}
		keys := blockKeysFor(tokens, 2)
		oc.cpu.store(keys[0])

		req := &sim.Request{ID: "p", InputTokens: tokens}
		reloadableEnd, ok := oc.ReloadablePrefixEnd(req, 0)
		if !ok || reloadableEnd != 2 {
			t.Fatalf("1 CPU-resident block must report reloadableEnd=2, ok=true; got %d ok=%v", reloadableEnd, ok)
		}
	})

	t.Run("nothing CPU-resident reports ok=false", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4} // nothing CPU-resident: genuine miss
		req := &sim.Request{ID: "m", InputTokens: tokens}
		if reloadableEnd, ok := oc.ReloadablePrefixEnd(req, 0); ok {
			t.Fatalf("a request with no CPU-resident prefix must report ok=false; got %d ok=%v", reloadableEnd, ok)
		}
	})

	t.Run("a running continuation is not reported (bills incrementally)", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4}
		keys := blockKeysFor(tokens, 2)
		oc.cpu.store(keys[0])
		oc.cpu.store(keys[1])

		// Admit the request so it becomes a running continuation (present in RequestMap).
		req := &sim.Request{ID: "run", InputTokens: tokens}
		if ok := oc.AllocateKVBlocks(req, 0, 4, nil); !ok {
			t.Fatalf("allocation should succeed")
		}
		if _, ok := oc.ReloadablePrefixEnd(req, 0); ok {
			t.Fatalf("a running continuation must not be reported (it bills incrementally against ProgressIndex)")
		}
	})

	// L2 (reviewer coverage gap): the boundary is an ABSOLUTE token index, not relative to
	// startIndex — the one thing an implementer of this interface could plausibly get
	// wrong. Blocks 0-1 are GPU-resident (matched via GetCachedBlocks, startIndex=4), block
	// 2 [4,6) is CPU-resident, block 3 uncached. The reloadable run extends to block 2, so
	// reloadableEnd=6 (absolute).
	t.Run("query composed with non-zero startIndex reports absolute boundary", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8} // 4 blocks
		keys := blockKeysFor(tokens, 2)

		// Warm blocks 0-1 onto the GPU via a prior resident request.
		warm := &sim.Request{ID: "warm", InputTokens: tokens[:4]} // [1,2,3,4] = blocks 0-1
		if ok := oc.AllocateKVBlocks(warm, 0, 4, nil); !ok {
			t.Fatalf("warm allocation should succeed")
		}
		// Block 2 ([5,6] = tokens[4:6]) is CPU-resident only.
		oc.cpu.store(keys[2])

		cached := oc.GetCachedBlocks(tokens) // GPU-matched prefix: blocks 0-1
		startIndex := int64(len(cached)) * oc.BlockSize()
		if startIndex != 4 {
			t.Fatalf("precondition: startIndex must be 4 (blocks 0-1 GPU-resident), got %d", startIndex)
		}

		req := &sim.Request{ID: "r", InputTokens: tokens}
		reloadableEnd, ok := oc.ReloadablePrefixEnd(req, startIndex)
		if !ok || reloadableEnd != 6 {
			t.Fatalf("reloadableEnd must be the ABSOLUTE boundary 6 (blocks 0-2), not relative to startIndex=4; got %d ok=%v", reloadableEnd, ok)
		}
	})
}

// TestOffload_ComposedReloadCommit_ConservesBlocks is the companion COMMIT-PATH test to
// the pure-query TestOffload_ReloadablePrefixEnd_ReportsBoundary. The #1706 pure query
// deliberately commits nothing, so it cannot verify block ownership — this test restores
// the INV-4 GPU-conservation coverage the pre-#1706 one-shot ReloadedPrefixEnd subtest
// carried (reviewer coverage gap): the pure query is what batch formation READS, but
// AllocateKVBlocks is what actually reloads the CPU-resident block and allocates the tail,
// and the two must agree — after the composed (non-zero startIndex) reload+tail alloc the
// request owns every block of its input and no GPU block is lost or double-owned.
func TestOffload_ComposedReloadCommit_ConservesBlocks(t *testing.T) {
	gpu := NewKVCacheState(64, 2)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
	tokens := []sim.TokenID{1, 2, 3, 4, 5, 6, 7, 8} // 4 blocks
	keys := blockKeysFor(tokens, 2)

	// Warm blocks 0-1 onto the GPU via a prior resident request (startIndex will be 4).
	warm := &sim.Request{ID: "warm", InputTokens: tokens[:4]} // [1,2,3,4] = blocks 0-1
	if ok := oc.AllocateKVBlocks(warm, 0, 4, nil); !ok {
		t.Fatalf("warm allocation should succeed")
	}
	// Block 2 ([5,6] = tokens[4:6]) is CPU-resident only.
	oc.cpu.store(keys[2])

	cached := oc.GetCachedBlocks(tokens) // GPU-matched prefix: blocks 0-1
	startIndex := int64(len(cached)) * oc.BlockSize()
	if startIndex != 4 {
		t.Fatalf("precondition: startIndex must be 4 (blocks 0-1 GPU-resident), got %d", startIndex)
	}

	// AllocateKVBlocks is called with the UNCHANGED startIndex (C3): the CPU-resident
	// block 2 is reloaded to GPU and the uncached tail (block 3) is allocated.
	req := &sim.Request{ID: "r", InputTokens: tokens}
	if ok := oc.AllocateKVBlocks(req, startIndex, 8, cached); !ok {
		t.Fatalf("composed reload+tail allocation should succeed")
	}
	if oc.reloadCount != 1 {
		t.Fatalf("exactly one CPU-resident block (block 2) must be reloaded to GPU; reloadCount=%d", oc.reloadCount)
	}
	if got := len(oc.gpu.RequestMap["r"]); got != 4 {
		t.Fatalf("request must own all 4 blocks after reload+tail alloc, got %d", got)
	}
	if err := oc.gpu.verifyBlockConservation(); err != nil {
		t.Fatalf("INV-4 GPU conservation must hold after a composed reload: %v", err)
	}
}
