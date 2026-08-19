package kv

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// BenchmarkOffloadAllocate_GPUCachedPrefix measures the hot path (allocate a
// request whose prefix is GPU-resident) with the offload chain active. Compare
// against BenchmarkSingleTierAllocate_GPUCachedPrefix / BenchmarkTieredAllocate_*
// to confirm no material per-allocation regression (objective P): the offload
// bookkeeping (kvkey derive + O(1) CPU-tier probes, BC-C8) must not dominate.
func BenchmarkOffloadAllocate_GPUCachedPrefix(b *testing.B) {
	const blockSize = int64(16)
	const prefixBlocks = int64(32)
	gpu := NewKVCacheState(1<<20, blockSize)
	oc := NewOffloadCache(gpu, enabledOffloadCfg(1<<30, 4096, 1))

	toks := make([]sim.TokenID, (prefixBlocks+1)*blockSize)
	for i := range toks {
		toks[i] = sim.TokenID(i + 1)
	}
	oc.SetClock(1)
	seed := &sim.Request{ID: "seed", InputTokens: toks}
	oc.AllocateKVBlocks(seed, 0, int64(len(toks)), nil)
	oc.MirrorToCPU([]*sim.Request{seed})
	oc.ReleaseKVBlocks(seed)

	req := &sim.Request{ID: "r", InputTokens: toks}
	suffixPos := int(prefixBlocks * blockSize)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		toks[suffixPos] = sim.TokenID(1000000 + i)
		cached := oc.GetCachedBlocks(req.FullInputTokens())
		startIndex := int64(len(cached)) * blockSize
		oc.AllocateKVBlocks(req, startIndex, int64(len(toks)), cached)
		b.StopTimer()
		oc.ReleaseKVBlocks(req)
		b.StartTimer()
	}
}

// BenchmarkOffloadCPUTier_Ops confirms the CPU-tier ref_cnt operations are O(1)
// (BC-C8): store/pin/unpin/lookup at a large residency have no per-op cost that
// scales with the number of resident blocks.
func BenchmarkOffloadCPUTier_Ops(b *testing.B) {
	c := newOffloadCPUTier(1 << 20)
	for i := 0; i < (1 << 20); i++ {
		c.store(cpuTestKey(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		k := cpuTestKey(i & ((1 << 20) - 1))
		c.pin(k)
		c.unpin(k)
		_ = c.lookup(k)
	}
}
