package cluster

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

// StaleCacheIndex manages per-instance frozen snapshots of KV cache hash maps.
// When cache-signal-delay > 0, the cacheQueryFn closures delegate to this index
// instead of querying live instance state, simulating asynchronous KV event
// propagation from production llm-d (issue #919).
//
// Signal freshness (R17, INV-7):
//
//	Reads: KVCache.SnapshotCachedBlocksFn snapshots — Periodic staleness.
//	Refresh interval controlled by DeploymentConfig.CacheSignalDelay.
//	When delay=0, this type is not used (oracle mode).
//	Default delay is 2s, matching llm-d's speculative TTL.
type StaleCacheIndex struct {
	instances   map[InstanceID]*InstanceSimulator
	staleFns    map[string]func([]int) int // instanceID → frozen snapshot closure
	interval    int64                      // refresh interval (microseconds)
	lastRefresh int64                      // sim clock at last refresh
}

// NewStaleCacheIndex creates a StaleCacheIndex and takes an initial snapshot of all instances.
// interval is the refresh interval in simulated microseconds. Panics if interval <= 0.
func NewStaleCacheIndex(instances map[InstanceID]*InstanceSimulator, interval int64) *StaleCacheIndex {
	if interval <= 0 {
		panic(fmt.Sprintf("NewStaleCacheIndex: interval must be > 0, got %d", interval))
	}
	idx := &StaleCacheIndex{
		instances:   make(map[InstanceID]*InstanceSimulator),
		staleFns:    make(map[string]func([]int) int),
		interval:    interval,
		lastRefresh: 0,
	}
	for id, inst := range instances {
		idx.instances[id] = inst
		idx.staleFns[string(id)] = inst.SnapshotCacheQueryFn()
	}
	return idx
}

// RefreshIfNeeded updates all stale snapshots if the refresh interval has elapsed.
// No-op if clock - lastRefresh < interval.
func (s *StaleCacheIndex) RefreshIfNeeded(clock int64) {
	if clock-s.lastRefresh < s.interval {
		return
	}
	for id, inst := range s.instances {
		s.staleFns[string(id)] = inst.SnapshotCacheQueryFn()
	}
	s.lastRefresh = clock
}

// Query returns the cached block count for the given instance and tokens,
// using the stale snapshot. Returns 0 if the instance is unknown.
func (s *StaleCacheIndex) Query(instanceID string, tokens []int) int {
	if fn, ok := s.staleFns[instanceID]; ok {
		return fn(tokens)
	}
	logrus.Warnf("[stale-cache] Query for unknown instance %q — returning 0", instanceID)
	return 0
}

// RemoveInstance unregisters an instance (e.g., on termination) and frees its
// snapshot closure. No-op if the instance is not registered.
func (s *StaleCacheIndex) RemoveInstance(id InstanceID) {
	delete(s.instances, id)
	delete(s.staleFns, string(id))
}

// AddInstance registers a new instance (e.g., from NodeReadyEvent) and takes
// an initial snapshot. Panics if the instance ID is already registered.
func (s *StaleCacheIndex) AddInstance(id InstanceID, inst *InstanceSimulator) {
	if _, exists := s.instances[id]; exists {
		panic("StaleCacheIndex.AddInstance: instance " + string(id) + " already registered")
	}
	s.instances[id] = inst
	s.staleFns[string(id)] = inst.SnapshotCacheQueryFn()
}

// BuildCacheQueryFn returns a cacheQueryFn map where each closure delegates to the
// stale snapshot. The returned closures read the current staleFns[id] at call time
// (not a captured copy), so they automatically use the latest snapshot after refresh.
func (s *StaleCacheIndex) BuildCacheQueryFn() map[string]func([]int) int {
	result := make(map[string]func([]int) int, len(s.instances))
	for id := range s.instances {
		idStr := string(id)
		result[idStr] = func(tokens []int) int {
			return s.Query(idStr, tokens)
		}
	}
	return result
}
