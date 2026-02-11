// Package cluster provides multi-replica cluster simulation capabilities.
//
// This package wraps the single-instance simulator (sim.Simulator) to enable
// multi-replica coordination in ClusterSimulator (PR3).
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
	id  InstanceID
	sim *sim.Simulator
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
func (i *InstanceSimulator) Run() {
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
