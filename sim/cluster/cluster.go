package cluster

import (
	"container/heap"
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

	// Online routing pipeline fields
	clusterEvents      ClusterEventQueue
	seqCounter         int64
	admissionLatency   int64
	routingLatency     int64
	admissionPolicy    sim.AdmissionPolicy
	snapshotProvider   SnapshotProvider
	routingPolicy      sim.RoutingPolicy
	rejectedRequests   int // EC-2: count of requests rejected by admission policy
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
	simCfg := config.ToSimConfig()
	instances := make([]*InstanceSimulator, config.NumInstances)
	for idx := range instances {
		instances[idx] = NewInstanceSimulator(
			InstanceID(fmt.Sprintf("instance_%d", idx)),
			simCfg,
		)
	}
	// Build instance map for snapshot provider
	instanceMap := make(map[InstanceID]*InstanceSimulator, len(instances))
	for _, inst := range instances {
		instanceMap[inst.ID()] = inst
	}

	return &ClusterSimulator{
		config:           config,
		instances:        instances,
		rng:              sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed)),
		workload:         workload,
		tracesPath:       tracesPath,
		clusterEvents:    make(ClusterEventQueue, 0),
		admissionLatency: config.AdmissionLatency,
		routingLatency:   config.RoutingLatency,
		admissionPolicy:  sim.NewAdmissionPolicy(config.AdmissionPolicy, config.TokenBucketCapacity, config.TokenBucketRefillRate),
		snapshotProvider: NewCachedSnapshotProvider(instanceMap, DefaultObservabilityConfig()),
		routingPolicy:    sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingCacheWeight, config.RoutingLoadWeight),
	}
}

// Run executes the cluster simulation using online routing pipeline:
// generates requests centrally, schedules ClusterArrivalEvents, runs a shared-clock
// event loop processing cluster events before instance events, then finalizes.
// Panics if called more than once.
func (c *ClusterSimulator) Run() {
	if c.hasRun {
		panic("ClusterSimulator.Run() called more than once")
	}
	c.hasRun = true

	// 1. Generate requests centrally
	requests := c.generateRequests()

	// 2. Schedule ClusterArrivalEvents (NC-1: no pre-dispatch before event loop)
	heap.Init(&c.clusterEvents)
	for _, req := range requests {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &ClusterArrivalEvent{time: req.ArrivalTime, request: req},
			seqID: c.nextSeqID(),
		})
	}

	// Set request rate on all instances for metrics compatibility
	if c.workload != nil {
		for _, inst := range c.instances {
			inst.SetRequestRate(c.workload.Rate)
		}
	}

	// 3. Shared-clock event loop (BC-4: cluster events before instance events)
	for {
		// Find earliest cluster event time
		clusterTime := int64(math.MaxInt64)
		if len(c.clusterEvents) > 0 {
			clusterTime = c.clusterEvents[0].event.Timestamp()
		}

		// Find earliest instance event time
		instanceTime := int64(math.MaxInt64)
		instanceIdx := -1
		for idx, inst := range c.instances {
			if inst.HasPendingEvents() {
				t := inst.PeekNextEventTime()
				if t < instanceTime {
					instanceTime = t
					instanceIdx = idx
				}
			}
		}

		// Both queues empty: done
		if clusterTime == math.MaxInt64 && instanceIdx == -1 {
			break
		}

		// BC-4: Cluster events at time T processed before instance events at time T
		// Using <= ensures cluster events drain first when timestamps are equal
		if clusterTime <= instanceTime {
			entry := heap.Pop(&c.clusterEvents).(clusterEventEntry)
			c.clock = entry.event.Timestamp()
			if c.clock > c.config.Horizon {
				break
			}
			entry.event.Execute(c)
		} else {
			c.clock = instanceTime
			if c.clock > c.config.Horizon {
				break
			}
			c.instances[instanceIdx].ProcessNextEvent()
		}
	}

	// 4. Finalize all instances
	for _, inst := range c.instances {
		inst.Finalize()
	}
	c.aggregatedMetrics = c.aggregateMetrics()
}

// nextSeqID returns the next monotonically increasing sequence ID for event ordering.
func (c *ClusterSimulator) nextSeqID() int64 {
	id := c.seqCounter
	c.seqCounter++
	return id
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

// RejectedRequests returns the count of requests rejected by the admission policy (EC-2).
// Returns 0 if AlwaysAdmit is used or if no requests were rejected by TokenBucket.
func (c *ClusterSimulator) RejectedRequests() int {
	return c.rejectedRequests
}

// mergeFloat64Map merges src into dst, logging a warning on duplicate keys.
func mergeFloat64Map(dst, src map[string]float64, mapName string) {
	for k, v := range src {
		if _, exists := dst[k]; exists {
			logrus.Warnf("aggregateMetrics: duplicate request ID %q in %s", k, mapName)
		}
		dst[k] = v
	}
}

// mergeInt64Map merges src into dst, logging a warning on duplicate keys.
func mergeInt64Map(dst, src map[string]int64, mapName string) {
	for k, v := range src {
		if _, exists := dst[k]; exists {
			logrus.Warnf("aggregateMetrics: duplicate request ID %q in %s", k, mapName)
		}
		dst[k] = v
	}
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
		merged.KVBlocksUsed += m.KVBlocksUsed
		if m.PeakKVBlocksUsed > merged.PeakKVBlocksUsed {
			merged.PeakKVBlocksUsed = m.PeakKVBlocksUsed
		}
		merged.NumWaitQRequests = append(merged.NumWaitQRequests, m.NumWaitQRequests...)
		merged.NumRunningBatchRequests = append(merged.NumRunningBatchRequests, m.NumRunningBatchRequests...)

		// Merge per-request maps. IDs are globally unique (centrally generated as "request_N").
		// Duplicate IDs indicate a workload generation bug.
		mergeFloat64Map(merged.RequestTTFTs, m.RequestTTFTs, "RequestTTFTs")
		mergeFloat64Map(merged.RequestE2Es, m.RequestE2Es, "RequestE2Es")
		mergeFloat64Map(merged.RequestITLs, m.RequestITLs, "RequestITLs")
		mergeInt64Map(merged.RequestSchedulingDelays, m.RequestSchedulingDelays, "RequestSchedulingDelays")
		mergeFloat64Map(merged.RequestCompletionTimes, m.RequestCompletionTimes, "RequestCompletionTimes")

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
