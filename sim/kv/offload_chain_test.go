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
	single := NewKVStore(sim.KVCacheConfig{TotalKVBlocks: 64, BlockSizeTokens: 16})
	if _, ok := single.(*KVCacheState); !ok {
		t.Fatalf("disabled offload + no CPU blocks must be single-tier *KVCacheState, got %T", single)
	}
	// Disabled + legacy CPU blocks -> legacy TieredKVCache (unchanged).
	legacy := NewKVStore(sim.KVCacheConfig{TotalKVBlocks: 64, BlockSizeTokens: 16, KVCPUBlocks: 8, KVTransferBandwidth: 100})
	if _, ok := legacy.(*TieredKVCache); !ok {
		t.Fatalf("legacy KVCPUBlocks path must be *TieredKVCache, got %T", legacy)
	}
	// Enabled offload -> OffloadCache.
	off := NewKVStore(sim.KVCacheConfig{TotalKVBlocks: 64, BlockSizeTokens: 16, Offload: enabledOffloadCfg(1<<20, 4096, 1)})
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
		})
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
