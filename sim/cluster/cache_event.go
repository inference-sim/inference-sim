package cluster

import "github.com/sirupsen/logrus"

// PriorityCacheEvent is the cluster event priority for cache signal propagation events.
// Set to 10 — after all existing event types (0=Arrival through 9=ScaleActuation).
// This ensures the cache snapshot is taken after the instance has fully processed its step.
const PriorityCacheEvent = 10

// CacheEventArrivalEvent models the arrival of a KV cache state change signal
// from an instance to the router's cache index. In production llm-d, this corresponds
// to a ZMQ event traveling from vLLM to the router's KVBlockIndex.
//
// BLIS simplification: one event per step that ran AllocateKVBlocks, not per individual
// block mutation. This is deliberate — all block mutations from one step would arrive
// within the same propagation window, and evictions only happen inside AllocateKVBlocks.
// The behavioral difference from per-block events is negligible.
//
// Known gap: TieredKVCache.reloadPrefixFromCPU modifies gpu.HashToBlock directly
// without going through AllocateKVBlocks, so full-reload successes (where the entire
// requested range is satisfied from CPU cache) do not increment AllocationEpoch and
// therefore do not trigger a CacheEventArrivalEvent. This is a rare tiered-cache-only
// path with negligible routing impact. See issue #1056.
//
// Scheduled by ClusterSimulator's main loop when it detects AllocationEpoch() changed
// after ProcessNextEvent(). Fires CacheEventDelay µs later.
type CacheEventArrivalEvent struct {
	time       int64
	instanceID InstanceID
}

func (e *CacheEventArrivalEvent) Timestamp() int64 { return e.time }
func (e *CacheEventArrivalEvent) Priority() int     { return PriorityCacheEvent }

// Execute refreshes the stale cache snapshot for the target instance.
func (e *CacheEventArrivalEvent) Execute(cs *ClusterSimulator) {
	if cs.staleCache == nil {
		return // oracle mode — should not happen, but defensive
	}
	logrus.Debugf("[cluster] cache event arrival for instance %s at tick %d", e.instanceID, e.time)
	cs.staleCache.RefreshInstance(e.instanceID)
}
