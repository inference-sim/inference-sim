package cluster

import (
	"fmt"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim"
)

// UpdateMode controls when a snapshot field is refreshed.
type UpdateMode int

const (
	Immediate UpdateMode = iota // Re-read from instance on every Snapshot() call
	Periodic                    // Re-read only after Interval has elapsed
	OnDemand                    // Only refreshed via explicit RefreshAll()
)

// FieldConfig configures refresh behavior for a single snapshot field.
type FieldConfig struct {
	Mode     UpdateMode
	Interval int64 // Only used when Mode == Periodic (microseconds)
}

// ObservabilityConfig configures refresh behavior for all snapshot fields.
type ObservabilityConfig struct {
	QueueDepth    FieldConfig
	BatchSize     FieldConfig
	KVUtilization FieldConfig
	CacheBlocks   FieldConfig // cache block hash map staleness (precise-prefix-cache, no-hit-lru)
}

// DefaultObservabilityConfig returns a config where all fields use Immediate mode.
func DefaultObservabilityConfig() ObservabilityConfig {
	return ObservabilityConfig{
		QueueDepth:    FieldConfig{Mode: Immediate},
		BatchSize:     FieldConfig{Mode: Immediate},
		KVUtilization: FieldConfig{Mode: Immediate},
		CacheBlocks:   FieldConfig{Mode: Immediate},
	}
}

// newObservabilityConfig creates an ObservabilityConfig based on the refresh intervals.
// refreshInterval controls Prometheus-sourced signals (QueueDepth, BatchSize, KVUtilization);
// 0 = Immediate. cacheDelay controls cache block hash map staleness; 0 = Immediate (oracle mode).
func newObservabilityConfig(refreshInterval int64, cacheDelay int64) ObservabilityConfig {
	config := DefaultObservabilityConfig()
	if refreshInterval > 0 {
		periodic := FieldConfig{Mode: Periodic, Interval: refreshInterval}
		config.QueueDepth = periodic
		config.BatchSize = periodic
		config.KVUtilization = periodic
	}
	if cacheDelay > 0 {
		config.CacheBlocks = FieldConfig{Mode: Periodic, Interval: cacheDelay}
	}
	return config
}

// SnapshotProvider produces instance snapshots with configurable staleness.
// Returns sim.RoutingSnapshot directly — no intermediate type translation needed.
type SnapshotProvider interface {
	Snapshot(id InstanceID, clock int64) sim.RoutingSnapshot
	RefreshAll(clock int64)
	// HasInstance returns true if the given ID is registered and routable.
	HasInstance(id InstanceID) bool
}

// fieldTimestamps tracks the last refresh time per field per instance.
type fieldTimestamps struct {
	QueueDepth    int64
	BatchSize     int64
	KVUtilization int64
}

// cacheEntry holds a live instance reference and its current stale snapshot closure.
type cacheEntry struct {
	inst    *InstanceSimulator
	staleFn func([]int) int
}

// CachedSnapshotProvider implements SnapshotProvider with configurable caching.
// Fields configured as Immediate are re-read on every call.
// Fields configured as Periodic are re-read when the interval has elapsed.
// Fields configured as OnDemand are only refreshed via RefreshAll().
//
// When config.CacheBlocks.Mode == Periodic, the provider also manages stale
// snapshots of per-instance KV cache hash maps (previously StaleCacheIndex).
type CachedSnapshotProvider struct {
	instances   map[InstanceID]*InstanceSimulator
	config      ObservabilityConfig
	cache       map[InstanceID]sim.RoutingSnapshot
	lastRefresh map[InstanceID]fieldTimestamps

	// Cache block snapshot management (replaces StaleCacheIndex).
	cacheEntries     map[InstanceID]cacheEntry
	cacheLastRefresh int64
}

// NewCachedSnapshotProvider creates a CachedSnapshotProvider from instances and config.
// When config.CacheBlocks.Mode == Periodic, initial stale snapshots are taken for all instances.
func NewCachedSnapshotProvider(instances map[InstanceID]*InstanceSimulator, config ObservabilityConfig) *CachedSnapshotProvider {
	if instances == nil {
		instances = make(map[InstanceID]*InstanceSimulator)
	}
	cache := make(map[InstanceID]sim.RoutingSnapshot, len(instances))
	lastRefresh := make(map[InstanceID]fieldTimestamps, len(instances))
	for id := range instances {
		cache[id] = sim.NewRoutingSnapshot(string(id))
		lastRefresh[id] = fieldTimestamps{}
	}

	cacheEntries := make(map[InstanceID]cacheEntry, len(instances))
	if config.CacheBlocks.Mode == Periodic {
		for id, inst := range instances {
			warnIfNotSnapshotCapable(id, inst)
			cacheEntries[id] = cacheEntry{
				inst:    inst,
				staleFn: inst.SnapshotCacheQueryFn(),
			}
		}
	}

	return &CachedSnapshotProvider{
		instances:    instances,
		config:       config,
		cache:        cache,
		lastRefresh:  lastRefresh,
		cacheEntries: cacheEntries,
	}
}

// Snapshot returns a RoutingSnapshot, refreshing fields based on their configured mode.
// InFlightRequests is NOT set here — it is injected by buildRouterState() which has
// access to the cluster-level in-flight request tracking.
func (p *CachedSnapshotProvider) Snapshot(id InstanceID, clock int64) sim.RoutingSnapshot {
	inst := p.instances[id]
	snap := p.cache[id]
	lr := p.lastRefresh[id]

	snap.ID = string(id)
	snap.PreemptionCount = inst.PreemptionCount() // always Immediate — monotonically increasing counter

	if p.shouldRefresh(p.config.QueueDepth, lr.QueueDepth, clock) {
		snap.QueueDepth = inst.QueueDepth()
		lr.QueueDepth = clock
	}
	if p.shouldRefresh(p.config.BatchSize, lr.BatchSize, clock) {
		snap.BatchSize = inst.BatchSize()
		lr.BatchSize = clock
	}
	if p.shouldRefresh(p.config.KVUtilization, lr.KVUtilization, clock) {
		snap.KVUtilization = inst.KVUtilization()
		snap.FreeKVBlocks = inst.FreeKVBlocks()
		snap.CacheHitRate = inst.CacheHitRate()
		snap.TotalKvCapacityTokens = inst.TotalKvCapacityTokens()
		snap.KvTokensInUse = inst.KvTokensInUse()
		lr.KVUtilization = clock
	}

	p.cache[id] = snap
	p.lastRefresh[id] = lr
	return snap
}

// RefreshAll refreshes all fields for all instances regardless of mode.
func (p *CachedSnapshotProvider) RefreshAll(clock int64) {
	for id, inst := range p.instances {
		snap := sim.NewRoutingSnapshot(string(id))
		snap.QueueDepth = inst.QueueDepth()
		snap.BatchSize = inst.BatchSize()
		snap.KVUtilization = inst.KVUtilization()
		snap.FreeKVBlocks = inst.FreeKVBlocks()
		snap.CacheHitRate = inst.CacheHitRate()
		snap.TotalKvCapacityTokens = inst.TotalKvCapacityTokens()
		snap.KvTokensInUse = inst.KvTokensInUse()
		p.cache[id] = snap
		p.lastRefresh[id] = fieldTimestamps{
			QueueDepth:    clock,
			BatchSize:     clock,
			KVUtilization: clock,
		}
	}
}

// AddInstance registers a new instance with the provider so that subsequent
// Snapshot calls will include it. Panics if the ID is already registered.
func (p *CachedSnapshotProvider) AddInstance(id InstanceID, inst *InstanceSimulator) {
	if _, exists := p.instances[id]; exists {
		panic(fmt.Sprintf("CachedSnapshotProvider.AddInstance: instance %s already registered", id))
	}
	p.instances[id] = inst
	p.cache[id] = sim.NewRoutingSnapshot(string(id))
	p.lastRefresh[id] = fieldTimestamps{}
}

// HasInstance returns true if the given instance ID is registered with this provider.
// Used by tests to verify routability without accessing internal fields.
func (p *CachedSnapshotProvider) HasInstance(id InstanceID) bool {
	_, ok := p.instances[id]
	return ok
}

// RefreshCacheIfNeeded updates all stale cache snapshots if the CacheBlocks interval
// has elapsed. No-op when CacheBlocks.Mode != Periodic.
func (p *CachedSnapshotProvider) RefreshCacheIfNeeded(clock int64) {
	if p.config.CacheBlocks.Mode != Periodic {
		return
	}
	if clock-p.cacheLastRefresh < p.config.CacheBlocks.Interval {
		return
	}
	for id, e := range p.cacheEntries {
		e.staleFn = e.inst.SnapshotCacheQueryFn()
		p.cacheEntries[id] = e
	}
	p.cacheLastRefresh = clock
}

// CacheQuery returns the cached block count for the given instance and tokens.
// When CacheBlocks.Mode == Periodic, returns stale snapshot data.
// When CacheBlocks.Mode == Immediate, queries live instance state.
// Returns 0 if the instance is unknown.
func (p *CachedSnapshotProvider) CacheQuery(instanceID string, tokens []int) int {
	id := InstanceID(instanceID)
	if p.config.CacheBlocks.Mode == Periodic {
		if e, ok := p.cacheEntries[id]; ok {
			return e.staleFn(tokens)
		}
		logrus.Warnf("[cache-snapshot] Query for unknown instance %q — returning 0", instanceID)
		return 0
	}
	// Immediate mode: query live instance state.
	if inst, ok := p.instances[id]; ok {
		return inst.GetCachedBlockCount(tokens)
	}
	return 0
}

// BuildCacheQueryFn returns a cacheQueryFn map where each closure delegates to
// CacheQuery. The returned closures use the latest snapshot after RefreshCacheIfNeeded.
func (p *CachedSnapshotProvider) BuildCacheQueryFn() map[string]func([]int) int {
	var ids []InstanceID
	if p.config.CacheBlocks.Mode == Periodic {
		ids = make([]InstanceID, 0, len(p.cacheEntries))
		for id := range p.cacheEntries {
			ids = append(ids, id)
		}
	} else {
		ids = make([]InstanceID, 0, len(p.instances))
		for id := range p.instances {
			ids = append(ids, id)
		}
	}
	result := make(map[string]func([]int) int, len(ids))
	for _, id := range ids {
		idStr := string(id)
		result[idStr] = func(tokens []int) int {
			return p.CacheQuery(idStr, tokens)
		}
	}
	return result
}

// AddCacheInstance registers a new instance for cache snapshot tracking.
// Panics if the instance ID is already registered in cacheEntries.
func (p *CachedSnapshotProvider) AddCacheInstance(id InstanceID, inst *InstanceSimulator) {
	if _, exists := p.cacheEntries[id]; exists {
		panic(fmt.Sprintf("CachedSnapshotProvider.AddCacheInstance: instance %s already registered", id))
	}
	warnIfNotSnapshotCapable(id, inst)
	p.cacheEntries[id] = cacheEntry{
		inst:    inst,
		staleFn: inst.SnapshotCacheQueryFn(),
	}
}

// RemoveCacheInstance unregisters an instance from cache snapshot tracking.
// No-op if the instance is not registered.
func (p *CachedSnapshotProvider) RemoveCacheInstance(id InstanceID) {
	delete(p.cacheEntries, id)
}

// IsStaleCacheMode returns true when stale cache mode is active (CacheBlocks.Mode == Periodic).
func (p *CachedSnapshotProvider) IsStaleCacheMode() bool {
	return p.config.CacheBlocks.Mode == Periodic
}

// warnIfNotSnapshotCapable logs a warning if inst's KVCache does not implement
// cacheSnapshotCapable. Called once at registration (not on every refresh) to avoid log spam.
func warnIfNotSnapshotCapable(id InstanceID, inst *InstanceSimulator) {
	if inst.sim == nil {
		return
	}
	if _, ok := inst.sim.KVCache.(cacheSnapshotCapable); !ok {
		logrus.Warnf("[cache-snapshot] instance %s: KVCache does not implement cacheSnapshotCapable — falling back to live query; stale-cache semantics not honored", id)
	}
}

// shouldRefresh returns true if a field should be refreshed based on its config.
func (p *CachedSnapshotProvider) shouldRefresh(fc FieldConfig, lastTime int64, clock int64) bool {
	switch fc.Mode {
	case Immediate:
		return true
	case Periodic:
		return clock-lastTime >= fc.Interval
	case OnDemand:
		return false
	default:
		return false
	}
}
