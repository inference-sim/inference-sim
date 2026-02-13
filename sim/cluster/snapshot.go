package cluster

// InstanceSnapshot is an immutable value-type snapshot of an instance's state.
// Taken at a point in time, subsequent instance state changes do not affect it.
type InstanceSnapshot struct {
	ID            InstanceID
	Timestamp     int64
	QueueDepth    int
	BatchSize     int
	KVUtilization float64
	FreeKVBlocks  int64
}

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

// SnapshotProvider produces instance snapshots with configurable staleness.
type SnapshotProvider interface {
	Snapshot(id InstanceID, clock int64) InstanceSnapshot
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
	cache       map[InstanceID]InstanceSnapshot
	lastRefresh map[InstanceID]fieldTimestamps
}

// NewCachedSnapshotProvider creates a CachedSnapshotProvider from instances and config.
func NewCachedSnapshotProvider(instances map[InstanceID]*InstanceSimulator, config ObservabilityConfig) *CachedSnapshotProvider {
	cache := make(map[InstanceID]InstanceSnapshot, len(instances))
	lastRefresh := make(map[InstanceID]fieldTimestamps, len(instances))
	for id := range instances {
		cache[id] = InstanceSnapshot{ID: id}
		lastRefresh[id] = fieldTimestamps{}
	}
	return &CachedSnapshotProvider{
		instances:   instances,
		config:      config,
		cache:       cache,
		lastRefresh: lastRefresh,
	}
}

// Snapshot returns an InstanceSnapshot, refreshing fields based on their configured mode.
func (p *CachedSnapshotProvider) Snapshot(id InstanceID, clock int64) InstanceSnapshot {
	inst := p.instances[id]
	snap := p.cache[id]
	lr := p.lastRefresh[id]

	snap.ID = id
	snap.Timestamp = clock

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
		lr.KVUtilization = clock
	}

	p.cache[id] = snap
	p.lastRefresh[id] = lr
	return snap
}

// RefreshAll refreshes all fields for all instances regardless of mode.
func (p *CachedSnapshotProvider) RefreshAll(clock int64) {
	for id, inst := range p.instances {
		snap := InstanceSnapshot{
			ID:            id,
			Timestamp:     clock,
			QueueDepth:    inst.QueueDepth(),
			BatchSize:     inst.BatchSize(),
			KVUtilization: inst.KVUtilization(),
			FreeKVBlocks:  inst.FreeKVBlocks(),
		}
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
