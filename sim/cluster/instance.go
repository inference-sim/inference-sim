// Package cluster provides multi-replica cluster simulation capabilities.
//
// This package wraps the single-instance simulator (sim.Simulator) to enable
// multi-replica coordination via ClusterSimulator.
package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/kv"
	"github.com/inference-sim/inference-sim/sim/latency"
)

// InstanceID uniquely identifies a simulator instance within a cluster.
// Uses distinct type (not alias) to prevent accidental string mixing.
type InstanceID string

// InstanceSimulator wraps a Simulator for use in multi-replica clusters.
// Provides an interception point for cluster-level coordination.
//
// Thread-safety: NOT thread-safe. All methods must be called from the same goroutine.
type InstanceSimulator struct {
	id     InstanceID
	sim    *sim.Simulator
	hasRun bool
}

// NewInstanceSimulator creates an InstanceSimulator from a SimConfig struct.
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
// Failure modes: Panics if internal Simulator creation fails (matches existing behavior).
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
	// Create KV store (single-tier or tiered based on config)
	kvStore := kv.NewKVStore(cfg.KVCacheConfig)
	latencyModel, err := latency.NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		panic(fmt.Sprintf("NewInstanceSimulator(%s): NewLatencyModel: %v", id, err))
	}
	s, err := sim.NewSimulator(cfg, kvStore, latencyModel)
	if err != nil {
		panic(fmt.Sprintf("NewInstanceSimulator(%s): %v", id, err))
	}
	return &InstanceSimulator{
		id:  id,
		sim: s,
	}
}

// Run executes the simulation to completion.
// Delegates directly to wrapped Simulator.Run().
//
// Postconditions:
//   - Metrics() returns populated metrics
//   - Clock() returns final simulation time
//
// Panics if called more than once (run-once semantics).
func (i *InstanceSimulator) Run() {
	if i.hasRun {
		panic("InstanceSimulator.Run() called more than once")
	}
	i.hasRun = true
	i.sim.Run()
}

// ID returns the instance identifier.
func (i *InstanceSimulator) ID() InstanceID {
	return i.id
}

// Clock returns the current simulation clock (in ticks).
func (i *InstanceSimulator) Clock() int64 {
	return i.sim.CurrentClock()
}

// Metrics returns the simulation metrics.
// Returns pointer to wrapped Simulator's Metrics (not a copy).
func (i *InstanceSimulator) Metrics() *sim.Metrics {
	return i.sim.Metrics
}

// Horizon returns the simulation horizon (in ticks).
func (i *InstanceSimulator) Horizon() int64 {
	return i.sim.SimHorizon()
}


// InjectRequest delegates to sim.InjectArrival. Panics if called after Run().
func (i *InstanceSimulator) InjectRequest(req *sim.Request) {
	if i.hasRun {
		panic("InstanceSimulator.InjectRequest() called after Run()")
	}
	i.sim.InjectArrival(req)
}

// SetRequestRate sets the request rate on the underlying simulator.
func (i *InstanceSimulator) SetRequestRate(rate float64) {
	i.sim.SetRequestRate(rate)
}

// HasPendingEvents returns true if the instance has pending events.
func (i *InstanceSimulator) HasPendingEvents() bool { return i.sim.HasPendingEvents() }

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first; panics on empty queue.
func (i *InstanceSimulator) PeekNextEventTime() int64 { return i.sim.PeekNextEventTime() }

// ProcessNextEvent pops and executes the earliest event, returning it.
// Caller MUST check HasPendingEvents() first; panics on empty queue.
func (i *InstanceSimulator) ProcessNextEvent() sim.Event { return i.sim.ProcessNextEvent() }

// Finalize sets SimEndedTime, captures KV metrics, and logs completion.
func (i *InstanceSimulator) Finalize() {
	i.sim.Finalize()
	// Capture KV metrics at finalization for CollectRawMetrics
	i.sim.Metrics.CacheHitRate = i.sim.KVCache.CacheHitRate()
	i.sim.Metrics.KVThrashingRate = i.sim.KVCache.KVThrashingRate()
}

// QueueDepth returns the number of requests in the wait queue.
func (i *InstanceSimulator) QueueDepth() int {
	return i.sim.QueueDepth()
}

// BatchSize returns the number of requests in the running batch, or 0 if nil.
func (i *InstanceSimulator) BatchSize() int {
	return i.sim.BatchSize()
}

// KVUtilization returns the fraction of KV cache blocks in use.
func (i *InstanceSimulator) KVUtilization() float64 {
	return float64(i.sim.KVCache.UsedBlocks()) / float64(i.sim.KVCache.TotalCapacity())
}

// FreeKVBlocks returns the number of free KV cache blocks.
func (i *InstanceSimulator) FreeKVBlocks() int64 {
	return i.sim.KVCache.TotalCapacity() - i.sim.KVCache.UsedBlocks()
}

// CacheHitRate returns the cumulative cache hit rate.
func (i *InstanceSimulator) CacheHitRate() float64 {
	return i.sim.KVCache.CacheHitRate()
}

// PendingTransferLatency returns the accumulated CPU→GPU transfer latency (ticks).
// Non-zero when tiered KV cache has pending reloads. Zero for single-tier.
func (i *InstanceSimulator) PendingTransferLatency() int64 {
	return i.sim.KVCache.PendingTransferLatency()
}

// KVThrashingRate returns the fraction of CPU→GPU reloads that were thrashing
// (block was offloaded and reloaded within a short interval).
func (i *InstanceSimulator) KVThrashingRate() float64 {
	return i.sim.KVCache.KVThrashingRate()
}

// InjectRequestOnline injects a request during the event loop (online routing mode).
// Unlike InjectRequest, this does NOT check hasRun, allowing injection during simulation.
func (i *InstanceSimulator) InjectRequestOnline(req *sim.Request, eventTime int64) {
	i.sim.InjectArrivalAt(req, eventTime)
}
