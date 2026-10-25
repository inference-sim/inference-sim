package sim

import "fmt"

// The engine's state queries and host-initiated actions: what the cluster (sim/cluster) asks of
// an instance's engine, so it never reaches into the engine's KV store, wait queue or running
// batch (#2021 B3: blis-vllm answers state queries; PLAN-E2 E2.3).

// KVThrashing returns the KV store's thrashing rate and, for a store that counts them, the
// thrash events and opportunities behind it (ok false otherwise).
func (sim *EngineCore) KVThrashing() (rate float64, counts ThrashCounts, ok bool) {
	rate = sim.KVCache.KVThrashingRate()
	if c, isCounter := sim.KVCache.(ThrashCounter); isCounter {
		return rate, c.ThrashCounts(), true
	}
	return rate, ThrashCounts{}, false
}

// KVCapacity is the device KV pool's total blocks; KVBlockSize is tokens per block; KVUsed is
// the blocks in use; KVCacheHitRate the store's cumulative hit rate. A nil store reports 0.
func (sim *EngineCore) KVCapacity() int64 {
	if sim.KVCache == nil {
		return 0
	}
	return sim.KVCache.TotalCapacity()
}

func (sim *EngineCore) KVBlockSize() int64 {
	if sim.KVCache == nil {
		return 0
	}
	return sim.KVCache.BlockSize()
}

func (sim *EngineCore) KVUsed() int64 {
	if sim.KVCache == nil {
		return 0
	}
	return sim.KVCache.UsedBlocks()
}

func (sim *EngineCore) KVCacheHitRate() float64 { return sim.KVCache.CacheHitRate() }

// CachedBlockCount is how many consecutive leading blocks of tokens the prefix cache holds.
func (sim *EngineCore) CachedBlockCount(tokens []TokenID) int {
	return len(sim.KVCache.GetCachedBlocks(tokens))
}

// SnapshotCacheQuery returns a query over a frozen copy of the prefix cache, or ok false for a
// store that cannot snapshot. KVCacheState and TieredKVCache can; the cluster uses it for
// stale cache signals (issue #919).
func (sim *EngineCore) SnapshotCacheQuery() (fn func([]TokenID) int, ok bool) {
	type snapshotter interface {
		SnapshotCachedBlocksFn() func([]TokenID) int
	}
	if s, isSnap := sim.KVCache.(snapshotter); isSnap {
		return s.SnapshotCachedBlocksFn(), true
	}
	return nil, false
}

// ReleaseKV frees every KV block req holds.
func (sim *EngineCore) ReleaseKV(req *Request) { sim.KVCache.ReleaseKVBlocks(req) }

// RequestBlocks returns the device blocks req holds (the store's own slice), ok false for a
// store that cannot report them.
func (sim *EngineCore) RequestBlocks(req *Request) ([]int64, bool) {
	r, ok := sim.KVCache.(RequestBlockReporter)
	if !ok {
		return nil, false
	}
	return r.RequestBlocks(req.ID), true
}

// KVStoreKind names the KV store's type, for diagnostics.
func (sim *EngineCore) KVStoreKind() string { return typeName(sim.KVCache) }

// CachedBlockHashes, CPUBlockHashes and TierBlockHashes are the hashes the prefix cache, the
// CPU tier and a named secondary tier hold, sorted; nil for a store that does not track them.
func (sim *EngineCore) CachedBlockHashes() []string {
	if src, ok := sim.KVCache.(KVEventSource); ok {
		return src.CachedBlockHashes()
	}
	return nil
}

func (sim *EngineCore) CPUBlockHashes() []string {
	if src, ok := sim.KVCache.(KVEventSource); ok {
		return src.CPUBlockHashes()
	}
	return nil
}

func (sim *EngineCore) TierBlockHashes(name string) []string {
	if src, ok := sim.KVCache.(interface{ TierBlockHashes(string) []string }); ok {
		return src.TierBlockHashes(name)
	}
	return nil
}

// Evict removes req from the engine for a gateway eviction: from the wait queue (clearing any
// offload deferral) or the running batch, freeing its KV and its adapter pin, and marks it
// completed as a tombstone (INV-2: the request produced no output and is counted as
// gateway-evicted, never through CompletedRequests). It reports whether req was there.
func (sim *EngineCore) Evict(req *Request) bool {
	if sim.WaitQ.Remove(req) {
		sim.ClearDeferredKV(req.ID) // H3 (#1591): a gateway-evicted queued request may be mid-deferral
		sim.KVCache.ReleaseKVBlocks(req)
		// Release the LoRA adapter pin (#1466): a queued request is not pinned, so
		// this is a no-op here, but calling it keeps eviction symmetric with the
		// running-batch branch below and robust if pinning semantics change.
		sim.ReleaseAdapterPin(req)
		req.State = StateCompleted
		return true
	}
	if sim.RunningBatch != nil {
		for idx, r := range sim.RunningBatch.Requests {
			if r.ID == req.ID {
				sim.RunningBatch.Requests = append(
					sim.RunningBatch.Requests[:idx],
					sim.RunningBatch.Requests[idx+1:]...,
				)
				sim.KVCache.ReleaseKVBlocks(req)
				// Release the LoRA adapter pin (#1466): a running request WAS pinned
				// on batch entry, and this gateway eviction is a terminal path outside
				// completion/timeout, so it must free the slot or the pin leaks and
				// eventually stalls all cold loads on this instance (INV-L5).
				sim.ReleaseAdapterPin(req)
				req.State = StateCompleted
				if len(sim.RunningBatch.Requests) == 0 {
					sim.RunningBatch = nil
				}
				return true
			}
		}
	}
	return false
}

func typeName(v any) string { return fmt.Sprintf("%T", v) }
