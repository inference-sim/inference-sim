// Package cluster provides multi-replica cluster simulation capabilities.
//
// This package wraps the single-instance simulator (sim.Simulator) to enable
// multi-replica coordination via ClusterSimulator.
package cluster

import (
	"github.com/inference-sim/inference-sim/sim"
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

// NewInstanceSimulator creates an InstanceSimulator wrapping a new Simulator.
// All parameters except `id` are passed directly to sim.NewSimulator.
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
// Failure modes: Panics if internal Simulator creation fails (matches existing behavior).
func NewInstanceSimulator(
	id InstanceID,
	horizon int64,
	seed int64,
	totalKVBlocks int64,
	blockSizeTokens int64,
	maxRunningReqs int64,
	maxScheduledTokens int64,
	longPrefillTokenThreshold int64,
	betaCoeffs []float64,
	alphaCoeffs []float64,
	guideLLMConfig *sim.GuideLLMConfig,
	modelConfig sim.ModelConfig,
	hwConfig sim.HardwareCalib,
	model string,
	GPU string,
	tp int,
	roofline bool,
	tracesWorkloadFilePath string,
) *InstanceSimulator {
	s := sim.NewSimulator(
		horizon,
		seed,
		totalKVBlocks,
		blockSizeTokens,
		maxRunningReqs,
		maxScheduledTokens,
		longPrefillTokenThreshold,
		betaCoeffs,
		alphaCoeffs,
		guideLLMConfig,
		modelConfig,
		hwConfig,
		model,
		GPU,
		tp,
		roofline,
		tracesWorkloadFilePath,
	)
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
	return i.sim.Clock
}

// Metrics returns the simulation metrics.
// Returns pointer to wrapped Simulator's Metrics (not a copy).
func (i *InstanceSimulator) Metrics() *sim.Metrics {
	return i.sim.Metrics
}

// Horizon returns the simulation horizon (in ticks).
func (i *InstanceSimulator) Horizon() int64 {
	return i.sim.Horizon
}

// NewInstanceSimulatorWithoutWorkload creates an InstanceSimulator with no workload generation.
// Caller injects requests via InjectRequest before running.
func NewInstanceSimulatorWithoutWorkload(
	id InstanceID,
	horizon, seed, totalKVBlocks, blockSizeTokens,
	maxRunningReqs, maxScheduledTokens, longPrefillTokenThreshold int64,
	betaCoeffs, alphaCoeffs []float64,
	modelConfig sim.ModelConfig, hwConfig sim.HardwareCalib,
	model, GPU string, tp int, roofline bool,
) *InstanceSimulator {
	s := sim.NewSimulatorWithoutWorkload(horizon, seed, totalKVBlocks,
		blockSizeTokens, maxRunningReqs, maxScheduledTokens,
		longPrefillTokenThreshold, betaCoeffs, alphaCoeffs,
		modelConfig, hwConfig, model, GPU, tp, roofline)
	return &InstanceSimulator{id: id, sim: s}
}

// InjectRequest delegates to sim.InjectArrival. Panics if called after Run().
func (i *InstanceSimulator) InjectRequest(req *sim.Request) {
	if i.hasRun {
		panic("InstanceSimulator.InjectRequest() called after Run()")
	}
	i.sim.InjectArrival(req)
}

// SetRequestRate sets the request rate on the instance's metrics.
func (i *InstanceSimulator) SetRequestRate(rate float64) {
	i.sim.Metrics.RequestRate = rate
}

// HasPendingEvents returns true if the instance has pending events.
func (i *InstanceSimulator) HasPendingEvents() bool { return i.sim.HasPendingEvents() }

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first; panics on empty queue.
func (i *InstanceSimulator) PeekNextEventTime() int64 { return i.sim.PeekNextEventTime() }

// ProcessNextEvent pops and executes the earliest event.
// Caller MUST check HasPendingEvents() first; panics on empty queue.
func (i *InstanceSimulator) ProcessNextEvent() { i.sim.ProcessNextEvent() }

// Finalize sets SimEndedTime and logs completion.
func (i *InstanceSimulator) Finalize() { i.sim.Finalize() }
