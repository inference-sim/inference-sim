package cluster

import "github.com/inference-sim/inference-sim/sim"

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
}

// DefaultObservabilityConfig returns a config where all fields use Immediate mode.
func DefaultObservabilityConfig() ObservabilityConfig {
	return ObservabilityConfig{
		QueueDepth:    FieldConfig{Mode: Immediate},
		BatchSize:     FieldConfig{Mode: Immediate},
		KVUtilization: FieldConfig{Mode: Immediate},
	}
}

// newObservabilityConfig creates an ObservabilityConfig based on the refresh interval.
// If interval is 0, all fields use Immediate mode (default behavior).
// If interval > 0, all fields use Periodic mode with the given interval,
// matching real vLLM Prometheus endpoint behavior where all gauges share the
// same scrape interval.
func newObservabilityConfig(refreshInterval int64) ObservabilityConfig {
	if refreshInterval <= 0 {
		return DefaultObservabilityConfig()
	}
	periodic := FieldConfig{Mode: Periodic, Interval: refreshInterval}
	return ObservabilityConfig{
		QueueDepth:    periodic,
		BatchSize:     periodic,
		KVUtilization: periodic,
	}
}

// SnapshotProvider produces instance snapshots with configurable staleness.
// Returns sim.RoutingSnapshot directly — no intermediate type translation needed.
type SnapshotProvider interface {
	Snapshot(id InstanceID, clock int64) sim.RoutingSnapshot
	RefreshAll(clock int64)
}

// fieldTimestamps tracks the last refresh time per field per instance.
type fieldTimestamps struct {
	QueueDepth    int64
	BatchSize     int64
	KVUtilization int64
}

// CachedSnapshotProvider implements SnapshotProvider with configurable caching.
// Fields configured as Immediate are re-read on every call.
// Fields configured as Periodic are re-read when the interval has elapsed.
// Fields configured as OnDemand are only refreshed via RefreshAll().
type CachedSnapshotProvider struct {
	instances   map[InstanceID]*InstanceSimulator
	config      ObservabilityConfig
	cache       map[InstanceID]sim.RoutingSnapshot
	lastRefresh map[InstanceID]fieldTimestamps
}

// NewCachedSnapshotProvider creates a CachedSnapshotProvider from instances and config.
func NewCachedSnapshotProvider(instances map[InstanceID]*InstanceSimulator, config ObservabilityConfig) *CachedSnapshotProvider {
	cache := make(map[InstanceID]sim.RoutingSnapshot, len(instances))
	lastRefresh := make(map[InstanceID]fieldTimestamps, len(instances))
	for id := range instances {
		cache[id] = sim.NewRoutingSnapshot(string(id))
		lastRefresh[id] = fieldTimestamps{}
	}
	return &CachedSnapshotProvider{
		instances:   instances,
		config:      config,
		cache:       cache,
		lastRefresh: lastRefresh,
	}
}

// Snapshot returns a RoutingSnapshot, refreshing fields based on their configured mode.
// PendingRequests is NOT set here — it is injected by buildRouterState() which has
// access to the cluster-level pending request tracking.
func (p *CachedSnapshotProvider) Snapshot(id InstanceID, clock int64) sim.RoutingSnapshot {
	inst := p.instances[id]
	snap := p.cache[id]
	lr := p.lastRefresh[id]

	snap.ID = string(id)

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
		p.cache[id] = snap
		p.lastRefresh[id] = fieldTimestamps{
			QueueDepth:    clock,
			BatchSize:     clock,
			KVUtilization: clock,
		}
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
