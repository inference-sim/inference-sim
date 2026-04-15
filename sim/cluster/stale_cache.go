package cluster

import (
	"fmt"

	"github.com/sirupsen/logrus"
)

// instanceCacheEntry holds a live instance reference and its current stale snapshot closure.
// The entry is owned exclusively by StaleCacheIndex.
type instanceCacheEntry struct {
	inst    *InstanceSimulator
	staleFn func([]int) int
}

// StaleCacheIndex manages per-instance frozen snapshots of KV cache hash maps.
// When cache-event-delay > 0, the cacheQueryFn closures delegate to this index
// instead of querying live instance state, simulating asynchronous KV event
// propagation from production llm-d (issues #919, #1029).
//
// Signal freshness (R17, INV-7):
//
//	Reads: InstanceSimulator.SnapshotCacheQueryFn() snapshots (delegates to
//	KVCacheState.SnapshotCachedBlocksFn via cacheSnapshotCapable).
//	Primary refresh path: event-driven via RefreshInstance(id), called by
//	CacheEventArrivalEvent after each step that allocates KV blocks (issue #1029).
//	Legacy refresh path: RefreshIfNeeded() (periodic, retained for test use).
//	Controlled by DeploymentConfig.CacheEventDelay.
//	When delay=0, this type is not used (oracle mode).
//	Default delay is 50ms (DefaultCacheEventDelay).
type StaleCacheIndex struct {
	entries     map[InstanceID]instanceCacheEntry
	interval    int64 // refresh interval (microseconds). Used by RefreshIfNeeded() only (deprecated path). Event-driven refresh via RefreshInstance() does not use this value.
	lastRefresh int64 // sim clock at last refresh
}

// NewStaleCacheIndex creates a StaleCacheIndex and takes an initial snapshot of all instances.
// interval is the refresh interval in simulated microseconds. Panics if interval <= 0.
// instances may be nil or empty.
func NewStaleCacheIndex(instances map[InstanceID]*InstanceSimulator, interval int64) *StaleCacheIndex {
	if interval <= 0 {
		panic(fmt.Sprintf("NewStaleCacheIndex: interval must be > 0, got %d", interval))
	}
	idx := &StaleCacheIndex{
		entries:     make(map[InstanceID]instanceCacheEntry, len(instances)),
		interval:    interval,
		lastRefresh: 0,
	}
	for id, inst := range instances {
		warnIfNotSnapshotCapable(id, inst)
		idx.entries[id] = instanceCacheEntry{
			inst:    inst,
			staleFn: inst.SnapshotCacheQueryFn(),
		}
	}
	return idx
}

// RefreshIfNeeded updates all stale snapshots if the refresh interval has elapsed.
// No-op if clock - lastRefresh < interval. Refreshes when clock - lastRefresh >= interval.
func (s *StaleCacheIndex) RefreshIfNeeded(clock int64) {
	if clock-s.lastRefresh < s.interval {
		return
	}
	for id, e := range s.entries {
		e.staleFn = e.inst.SnapshotCacheQueryFn()
		s.entries[id] = e // re-assign: map stores value type, not pointer
	}
	s.lastRefresh = clock
}

// RefreshInstance updates the stale snapshot for a single instance.
// No-op if the instance ID is not registered. Does not affect other instances' snapshots.
func (s *StaleCacheIndex) RefreshInstance(id InstanceID) {
	e, ok := s.entries[id]
	if !ok {
		return
	}
	e.staleFn = e.inst.SnapshotCacheQueryFn()
	s.entries[id] = e
}

// Query returns the cached block count for the given instance and tokens,
// using the stale snapshot. Returns 0 if the instance is unknown.
func (s *StaleCacheIndex) Query(instanceID string, tokens []int) int {
	if e, ok := s.entries[InstanceID(instanceID)]; ok {
		return e.staleFn(tokens)
	}
	logrus.Warnf("[stale-cache] Query for unknown instance %q — returning 0", instanceID)
	return 0
}

// RemoveInstance unregisters an instance (e.g., on termination) and frees its
// snapshot closure. No-op if the instance is not registered.
func (s *StaleCacheIndex) RemoveInstance(id InstanceID) {
	delete(s.entries, id)
}

// AddInstance registers a new instance (e.g., from NodeReadyEvent) and takes
// an initial snapshot. Panics if the instance ID is already registered.
func (s *StaleCacheIndex) AddInstance(id InstanceID, inst *InstanceSimulator) {
	if _, exists := s.entries[id]; exists {
		panic("StaleCacheIndex.AddInstance: instance " + string(id) + " already registered")
	}
	warnIfNotSnapshotCapable(id, inst)
	s.entries[id] = instanceCacheEntry{
		inst:    inst,
		staleFn: inst.SnapshotCacheQueryFn(),
	}
}

// BuildCacheQueryFn returns a cacheQueryFn map where each closure delegates to the
// stale snapshot. The returned closures call s.Query at call time (not a captured copy),
// so they automatically use the latest snapshot after RefreshIfNeeded.
func (s *StaleCacheIndex) BuildCacheQueryFn() map[string]func([]int) int {
	result := make(map[string]func([]int) int, len(s.entries))
	for id := range s.entries {
		idStr := string(id)
		result[idStr] = func(tokens []int) int {
			return s.Query(idStr, tokens)
		}
	}
	return result
}

// warnIfNotSnapshotCapable logs a warning if inst's KVCache does not implement
// cacheSnapshotCapable. Called once at registration (not on every refresh) to avoid log spam.
func warnIfNotSnapshotCapable(id InstanceID, inst *InstanceSimulator) {
	if inst.sim == nil {
		return
	}
	if _, ok := inst.sim.KVCache.(cacheSnapshotCapable); !ok {
		logrus.Warnf("[stale-cache] instance %s: KVCache does not implement cacheSnapshotCapable — falling back to live query; stale-cache semantics not honored", id)
	}
}
