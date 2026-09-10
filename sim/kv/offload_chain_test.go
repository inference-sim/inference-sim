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

// #1699: after a CPU->GPU reload during AllocateKVBlocks, the offload store must
// report the enlarged (post-reload) GPU-cached prefix boundary via the
// ReloadReportingKVStore capability, so batch formation can re-bill prefill work.
// The record is one-shot (consumed on read) and recorded only for NEW admissions.
func TestOffload_ReloadedPrefixEnd_ReportsBoundary(t *testing.T) {
	// Compile-time capability assertion.
	var _ sim.ReloadReportingKVStore = (*OffloadCache)(nil)

	t.Run("full-prefix reload reports InputLen and is one-shot", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4} // 2 blocks, all CPU-resident
		keys := blockKeysFor(tokens, 2)
		oc.cpu.store(keys[0])
		oc.cpu.store(keys[1])

		req := &sim.Request{ID: "r", InputTokens: tokens}
		if ok := oc.AllocateKVBlocks(req, 0, 4, nil); !ok {
			t.Fatalf("allocation should succeed")
		}
		newStart, ok := oc.ReloadedPrefixEnd("r")
		if !ok || newStart != 4 {
			t.Fatalf("full CPU-resident prefix reload must report newStart=4, ok=true; got newStart=%d ok=%v", newStart, ok)
		}
		// One-shot: the second read must not see a stale boundary.
		if _, ok := oc.ReloadedPrefixEnd("r"); ok {
			t.Fatalf("ReloadedPrefixEnd must be one-shot (consumed on read)")
		}
	})

	t.Run("partial reload reports the reloaded boundary", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		// 3 blocks of input; only the first block is CPU-resident. The tail (blocks
		// 1..2) is a genuine miss and is computed fresh.
		tokens := []sim.TokenID{1, 2, 3, 4, 5, 6}
		keys := blockKeysFor(tokens, 2)
		oc.cpu.store(keys[0])

		req := &sim.Request{ID: "p", InputTokens: tokens}
		if ok := oc.AllocateKVBlocks(req, 0, 6, nil); !ok {
			t.Fatalf("allocation should succeed")
		}
		newStart, ok := oc.ReloadedPrefixEnd("p")
		if !ok || newStart != 2 {
			t.Fatalf("partial reload of 1 block must report newStart=2, ok=true; got newStart=%d ok=%v", newStart, ok)
		}
	})

	t.Run("no reload reports ok=false", func(t *testing.T) {
		gpu := NewKVCacheState(64, 2)
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4} // nothing CPU-resident: genuine miss
		req := &sim.Request{ID: "m", InputTokens: tokens}
		if ok := oc.AllocateKVBlocks(req, 0, 4, nil); !ok {
			t.Fatalf("allocation should succeed")
		}
		if newStart, ok := oc.ReloadedPrefixEnd("m"); ok {
			t.Fatalf("a request with no CPU reload must report ok=false; got newStart=%d ok=%v", newStart, ok)
		}
	})

	t.Run("failed tail alloc records no boundary (no leak)", func(t *testing.T) {
		// A tiny GPU: 1 reloadable prefix block fits, but the uncached tail cannot be
		// allocated, so AllocateKVBlocks returns false. The request is NOT admitted, so
		// batch formation never reads the boundary — recording it would leak a stale
		// entry into a later step. Assert no boundary is recorded on the failure path.
		gpu := NewKVCacheState(1, 2) // room for exactly 1 block
		oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<20, 4096, 1))
		tokens := []sim.TokenID{1, 2, 3, 4, 5, 6} // 3 blocks; block 0 CPU-resident, tail uncached
		keys := blockKeysFor(tokens, 2)
		oc.cpu.store(keys[0])

		req := &sim.Request{ID: "f", InputTokens: tokens}
		if ok := oc.AllocateKVBlocks(req, 0, 6, nil); ok {
			t.Fatalf("allocation must fail: the uncached tail cannot fit a 1-block GPU")
		}
		if newStart, ok := oc.ReloadedPrefixEnd("f"); ok {
			t.Fatalf("a failed admission must record no boundary (no leak); got newStart=%d ok=%v", newStart, ok)
		}
	})
}
