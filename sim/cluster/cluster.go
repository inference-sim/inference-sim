package cluster

import (
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/sirupsen/logrus"
)

// ClusterSimulator orchestrates N InstanceSimulator replicas behind a shared clock.
// Events from all instances are processed in global timestamp order;
// ties are broken by lowest instance index for determinism.
type ClusterSimulator struct {
	config            DeploymentConfig
	instances         []*InstanceSimulator
	rng               *sim.PartitionedRNG
	clock             int64
	workload          *sim.GuideLLMConfig
	tracesPath        string
	hasRun            bool
	aggregatedMetrics *sim.Metrics
}

// NewClusterSimulator creates a ClusterSimulator with N instances.
// Panics if config.NumInstances < 1 or if both workload and tracesPath are unset.
func NewClusterSimulator(config DeploymentConfig, workload *sim.GuideLLMConfig,
	tracesPath string) *ClusterSimulator {
	if config.NumInstances < 1 {
		panic("ClusterSimulator: NumInstances must be >= 1")
	}
	if workload == nil && tracesPath == "" {
		panic("ClusterSimulator: workload config is nil and no traces path provided")
	}
	instances := make([]*InstanceSimulator, config.NumInstances)
	for idx := range instances {
		instances[idx] = NewInstanceSimulatorWithoutWorkload(
			InstanceID(fmt.Sprintf("instance_%d", idx)),
			config.Horizon,
			config.Seed,
			config.TotalKVBlocks,
			config.BlockSizeTokens,
			config.MaxRunningReqs,
			config.MaxScheduledTokens,
			config.LongPrefillTokenThreshold,
			config.BetaCoeffs,
			config.AlphaCoeffs,
			config.ModelConfig,
			config.HWConfig,
			config.Model,
			config.GPU,
			config.TP,
			config.Roofline,
		)
	}
	return &ClusterSimulator{
		config:     config,
		instances:  instances,
		rng:        sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed)),
		workload:   workload,
		tracesPath: tracesPath,
	}
}

// Run executes the cluster simulation: generates requests centrally,
// dispatches them round-robin, runs a shared-clock event loop across
// all instances, then finalizes and aggregates metrics.
// Panics if called more than once.
func (c *ClusterSimulator) Run() {
	if c.hasRun {
		panic("ClusterSimulator.Run() called more than once")
	}
	c.hasRun = true

	// 1. Generate requests centrally
	requests := c.generateRequests()

	// 2. Dispatch round-robin and set request rate.
	// Each instance gets the global rate for metrics compatibility;
	// the actual per-instance arrival rate is Rate/N due to round-robin dispatch.
	for i, req := range requests {
		c.instances[i%c.config.NumInstances].InjectRequest(req)
	}
	if c.workload != nil {
		for _, inst := range c.instances {
			inst.SetRequestRate(c.workload.Rate)
		}
	}

	// 3. Shared-clock event loop
	for {
		// Find instance with earliest pending event.
		// Ties: lowest index wins because we use strict < and iterate 0..N-1.
		earliestTime := int64(math.MaxInt64)
		earliestIdx := -1
		for idx, inst := range c.instances {
			if inst.HasPendingEvents() {
				t := inst.PeekNextEventTime()
				if t < earliestTime {
					earliestTime = t
					earliestIdx = idx
				}
			}
		}
		if earliestIdx == -1 {
			break // all instances drained
		}
		c.clock = earliestTime
		c.instances[earliestIdx].ProcessNextEvent()
		if c.clock > c.config.Horizon {
			break
		}
	}

	// 4. Finalize all instances. Each instance's SimEndedTime = min(its Clock, Horizon).
	// Instances that never advanced past time 0 (no events) get SimEndedTime = 0.
	for _, inst := range c.instances {
		inst.Finalize()
	}
	c.aggregatedMetrics = c.aggregateMetrics()
}

// Clock returns the cluster's current simulation clock.
func (c *ClusterSimulator) Clock() int64 {
	return c.clock
}

// Instances returns the slice of InstanceSimulators.
func (c *ClusterSimulator) Instances() []*InstanceSimulator {
	return c.instances
}

// AggregatedMetrics returns the merged metrics across all instances.
// Panics if called before Run() has completed.
func (c *ClusterSimulator) AggregatedMetrics() *sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.AggregatedMetrics() called before Run()")
	}
	return c.aggregatedMetrics
}

func (c *ClusterSimulator) aggregateMetrics() *sim.Metrics {
	merged := sim.NewMetrics()
	for _, inst := range c.instances {
		m := inst.Metrics()
		merged.CompletedRequests += m.CompletedRequests
		merged.TotalInputTokens += m.TotalInputTokens
		merged.TotalOutputTokens += m.TotalOutputTokens
		merged.TTFTSum += m.TTFTSum
		merged.ITLSum += m.ITLSum
		if m.SimEndedTime > merged.SimEndedTime {
			merged.SimEndedTime = m.SimEndedTime
		}
		// KVBlocksUsed: sum of time-integrated block usage across all instances.
		merged.KVBlocksUsed += m.KVBlocksUsed
		// PeakKVBlocksUsed: max across instances (each instance has independent KV cache memory).
		if m.PeakKVBlocksUsed > merged.PeakKVBlocksUsed {
			merged.PeakKVBlocksUsed = m.PeakKVBlocksUsed
		}
		// Step-indexed time series are concatenated, not interleaved by time.
		merged.NumWaitQRequests = append(merged.NumWaitQRequests, m.NumWaitQRequests...)
		merged.NumRunningBatchRequests = append(merged.NumRunningBatchRequests, m.NumRunningBatchRequests...)
		// Merge per-request maps (IDs are globally unique — centrally generated as "request_N").
		// Log a warning if duplicate IDs are detected, as this indicates a workload generation bug.
		for k, v := range m.RequestTTFTs {
			if _, exists := merged.RequestTTFTs[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in RequestTTFTs", k)
			}
			merged.RequestTTFTs[k] = v
		}
		for k, v := range m.RequestE2Es {
			if _, exists := merged.RequestE2Es[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in RequestE2Es", k)
			}
			merged.RequestE2Es[k] = v
		}
		for k, v := range m.RequestITLs {
			if _, exists := merged.RequestITLs[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in RequestITLs", k)
			}
			merged.RequestITLs[k] = v
		}
		for k, v := range m.RequestSchedulingDelays {
			if _, exists := merged.RequestSchedulingDelays[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in RequestSchedulingDelays", k)
			}
			merged.RequestSchedulingDelays[k] = v
		}
		for k, v := range m.RequestCompletionTimes {
			if _, exists := merged.RequestCompletionTimes[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in RequestCompletionTimes", k)
			}
			merged.RequestCompletionTimes[k] = v
		}
		for k, v := range m.Requests {
			if _, exists := merged.Requests[k]; exists {
				logrus.Warnf("aggregateMetrics: duplicate request ID %q in Requests", k)
			}
			merged.Requests[k] = v
		}
		merged.AllITLs = append(merged.AllITLs, m.AllITLs...)
		merged.RequestStepCounters = append(merged.RequestStepCounters, m.RequestStepCounters...)
	}
	if c.workload != nil {
		merged.RequestRate = c.workload.Rate
	}
	return merged
}
