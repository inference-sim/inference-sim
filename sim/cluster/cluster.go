package cluster

import (
	"container/heap"
	"fmt"
	"math"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
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
	hasRun            bool
	aggregatedMetrics *sim.Metrics

	// Online routing pipeline fields
	clusterEvents        ClusterEventQueue
	seqCounter           int64
	admissionLatency     int64
	routingLatency       int64
	admissionPolicy      sim.AdmissionPolicy
	snapshotProvider     SnapshotProvider
	routingPolicy        sim.RoutingPolicy
	rejectedRequests     int                    // EC-2: count of requests rejected by admission policy
	trace                *trace.SimulationTrace // nil when trace-level is "none" (BC-1: zero overhead)
	preGeneratedRequests    []*sim.Request              // Pre-generated requests (all workload paths unified)
	inFlightRequests        map[string]int              // instance ID → dispatched-but-not-completed count (#463)
	poolMembership          map[string]PoolRole         // instance ID → pool role (nil when disaggregation disabled)
	disaggregationDecider   sim.DisaggregationDecider   // PD disaggregation decider (nil when disabled)

	droppedKVAllocations      int               // R1: count of decode requests dropped due to KV OOM

	// PD disaggregation state (PR2)
	parentRequests            map[string]*ParentRequest // parent request ID → tracking record
	pendingPrefillCompletions map[string]string         // prefill sub-req ID → parent ID
	transfersInitiated        int
	transfersCompleted        int
	prefillRoutingPolicy      sim.RoutingPolicy // nil = use main routingPolicy
	decodeRoutingPolicy       sim.RoutingPolicy // nil = use main routingPolicy
}

// NewClusterSimulator creates a ClusterSimulator with N instances.
// All workload generation now happens externally — requests are passed in directly.
// Panics if config.NumInstances < 1.
func NewClusterSimulator(config DeploymentConfig, requests []*sim.Request) *ClusterSimulator {
	if config.NumInstances < 1 {
		panic("ClusterSimulator: NumInstances must be >= 1")
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

	// Initialize trace collector if tracing is enabled (BC-1: nil when none)
	var simTrace *trace.SimulationTrace
	if config.TraceLevel != "" && trace.TraceLevel(config.TraceLevel) != trace.TraceLevelNone {
		simTrace = trace.NewSimulationTrace(trace.TraceConfig{
			Level:           trace.TraceLevel(config.TraceLevel),
			CounterfactualK: config.CounterfactualK,
		})
	}

	// Extract PartitionedRNG before struct literal so routing policy can use SubsystemRouter.
	// The routing policy exclusively owns the SubsystemRouter partition — do not reuse
	// cs.rng.ForSubsystem(SubsystemRouter) elsewhere to avoid interleaving RNG draws.
	rng := sim.NewPartitionedRNG(sim.NewSimulationKey(config.Seed))

	cs := &ClusterSimulator{
		config:               config,
		instances:            instances,
		rng:                  rng,
		preGeneratedRequests: requests,
		clusterEvents:        make(ClusterEventQueue, 0),
		admissionLatency:     config.AdmissionLatency,
		routingLatency:       config.RoutingLatency,
		admissionPolicy:      sim.NewAdmissionPolicy(config.AdmissionPolicy, config.TokenBucketCapacity, config.TokenBucketRefillRate),
		snapshotProvider:     NewCachedSnapshotProvider(instanceMap, newObservabilityConfig(config.SnapshotRefreshInterval)),
		routingPolicy:        sim.NewRoutingPolicy(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem(sim.SubsystemRouter)),
		trace:                simTrace,
		inFlightRequests:     make(map[string]int, config.NumInstances),
	}

	// PD disaggregation: validate topology and build pool membership
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
		if err := ValidatePoolTopology(config.PrefillInstances, config.DecodeInstances, config.NumInstances); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		cs.poolMembership = BuildPoolMembership(instances, config.PrefillInstances, config.DecodeInstances)
		if config.PDDecider == "prefix-threshold" {
			cs.disaggregationDecider = sim.NewPrefixThresholdDecider(config.PDPrefixThreshold, int(config.BlockSizeTokens))
		} else {
			cs.disaggregationDecider = sim.NewDisaggregationDecider(config.PDDecider)
		}
		cs.parentRequests = make(map[string]*ParentRequest)
		cs.pendingPrefillCompletions = make(map[string]string)

		// Per-pool routing policies (use separate RNG partitions to avoid fragile coupling)
		if len(config.PrefillScorerConfigs) > 0 {
			cs.prefillRoutingPolicy = sim.NewRoutingPolicy("weighted", config.PrefillScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("prefill-router"))
		}
		if len(config.DecodeScorerConfigs) > 0 {
			cs.decodeRoutingPolicy = sim.NewRoutingPolicy("weighted", config.DecodeScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("decode-router"))
		}

		logrus.Infof("[cluster] PD disaggregation enabled: %d prefill, %d decode instances, decider=%q",
			config.PrefillInstances, config.DecodeInstances, config.PDDecider)
	}

	// Startup warning: horizon too small for pipeline (BC-1)
	pipelineLatency := cs.admissionLatency + cs.routingLatency
	if cs.config.Horizon > 0 && cs.config.Horizon < pipelineLatency {
		logrus.Warnf("[cluster] horizon (%d) < pipeline latency (%d); no requests can complete — increase --horizon or reduce admission/routing latency",
			cs.config.Horizon, pipelineLatency)
	}

	return cs
}

// Run executes the cluster simulation using online routing pipeline:
// generates requests centrally, schedules ClusterArrivalEvents, runs a shared-clock
// event loop processing cluster events before instance events, then finalizes.
// Panics if called more than once.
func (c *ClusterSimulator) Run() error {
	if c.hasRun {
		panic("ClusterSimulator.Run() called more than once")
	}
	c.hasRun = true

	// 1. Use pre-generated requests (all workload paths now pre-generate)
	requests := c.preGeneratedRequests
	if len(requests) == 0 {
		logrus.Warn("[cluster] no requests provided — simulation will produce zero results")
	}

	// 2. Schedule ClusterArrivalEvents (NC-1: no pre-dispatch before event loop)
	heap.Init(&c.clusterEvents)
	for _, req := range requests {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &ClusterArrivalEvent{time: req.ArrivalTime, request: req},
			seqID: c.nextSeqID(),
		})
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
			inst := c.instances[instanceIdx]
			instID := string(inst.ID())

			// Snapshot counters BEFORE processing the event
			completedBefore := inst.Metrics().CompletedRequests
			droppedBefore := inst.Metrics().DroppedUnservable

			ev := inst.ProcessNextEvent()
			_ = ev // Event type no longer used for decrement

			// Completion-based decrement (#463, BC-3, BC-7): InFlightRequests tracks the full
			// dispatch-to-completion window. Decrement by the number of newly completed OR
			// dropped-unservable requests. DroppedUnservable requests never reach CompletedRequests
			// but still exit the in-flight window (they were rejected during EnqueueRequest).
			completedAfter := inst.Metrics().CompletedRequests
			droppedAfter := inst.Metrics().DroppedUnservable
			delta := (completedAfter - completedBefore) + (droppedAfter - droppedBefore)
			if delta > 0 {
				c.inFlightRequests[instID] -= delta
				if c.inFlightRequests[instID] < 0 {
					logrus.Warnf("inFlightRequests[%s] went negative (%d) after delta=%d (completed=%d, dropped=%d) — bookkeeping bug",
						instID, c.inFlightRequests[instID], delta, completedAfter-completedBefore, droppedAfter-droppedBefore)
					c.inFlightRequests[instID] = 0
				}

				// PD disaggregation: detect prefill sub-request completions and schedule KV transfer
				if c.poolsConfigured() && c.poolMembership[instID] == PoolRolePrefill {
					c.detectPrefillCompletions(inst)
				}
			}
		}
	}

	// 4. Finalize all instances (populates StillQueued/StillRunning)
	for _, inst := range c.instances {
		inst.Finalize()
	}

	// 5. Post-simulation invariant: inFlightRequests should match StillQueued + StillRunning
	// MUST be after Finalize() — StillQueued/StillRunning are zero until Finalize populates them.
	// NOTE: A mismatch can occur legitimately if requests were routed near the horizon but their
	// ArrivalEvent/QueuedEvent hadn't fired yet (request is in the instance event queue, not in
	// WaitQ or RunningBatch). This is an edge case, not a bookkeeping bug.
	for _, inst := range c.instances {
		instID := string(inst.ID())
		inflight := c.inFlightRequests[instID]
		m := inst.Metrics()
		expectedInFlight := m.StillQueued + m.StillRunning
		if inflight != expectedInFlight {
			logrus.Warnf("post-simulation: inFlightRequests[%s] = %d, expected %d (StillQueued=%d + StillRunning=%d) — may indicate bookkeeping bug or requests in event pipeline at horizon",
				instID, inflight, expectedInFlight, m.StillQueued, m.StillRunning)
		}
	}

	c.aggregatedMetrics = c.aggregateMetrics()

	// Post-simulation diagnostic warnings (BC-2, BC-3)
	if c.aggregatedMetrics.CompletedRequests == 0 {
		if c.rejectedRequests > 0 {
			logrus.Warnf("[cluster] all %d requests rejected by admission policy %q — no requests completed",
				c.rejectedRequests, c.config.AdmissionPolicy)
		} else {
			logrus.Warnf("[cluster] no requests completed — horizon may be too short or workload too small")
		}
	}

	return nil
}

// nextSeqID returns the next monotonically increasing sequence ID for event ordering.
func (c *ClusterSimulator) nextSeqID() int64 {
	id := c.seqCounter
	c.seqCounter++
	return id
}

// buildPoolFilteredSnapshots constructs routing snapshots filtered to a specific pool role.
// Preserves instance order from c.instances for determinism (R2).
func (c *ClusterSimulator) buildPoolFilteredSnapshots(role PoolRole) []sim.RoutingSnapshot {
	allSnapshots := make([]sim.RoutingSnapshot, len(c.instances))
	for i, inst := range c.instances {
		snap := c.snapshotProvider.Snapshot(inst.ID(), c.clock)
		snap.InFlightRequests = c.inFlightRequests[string(inst.ID())]
		allSnapshots[i] = snap
	}
	return FilterSnapshotsByPool(allSnapshots, c.poolMembership, role)
}

// detectPrefillCompletions checks for newly completed prefill sub-requests on the given instance
// and schedules KV transfer events for each.
func (c *ClusterSimulator) detectPrefillCompletions(inst *InstanceSimulator) {
	for subReqID, parentID := range c.pendingPrefillCompletions {
		if _, completed := inst.Metrics().RequestCompletionTimes[subReqID]; completed {
			parent := c.parentRequests[parentID]
			if parent == nil {
				continue
			}
			parent.PrefillCompleteTime = c.clock
			delete(c.pendingPrefillCompletions, subReqID)

			// Schedule KV transfer
			heap.Push(&c.clusterEvents, clusterEventEntry{
				event: &KVTransferStartedEvent{
					time:      c.clock,
					parentReq: parent,
				},
				seqID: c.nextSeqID(),
			})
		}
	}
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

// DroppedKVAllocations returns the count of decode sub-requests dropped due to
// insufficient KV capacity at the decode instance (R1: count dropped, never silent).
func (c *ClusterSimulator) DroppedKVAllocations() int {
	return c.droppedKVAllocations
}

// poolsConfigured returns true if PD disaggregation pool topology is active.
func (c *ClusterSimulator) poolsConfigured() bool {
	return c.poolMembership != nil
}

// notifyDisaggregationObserver calls ObserveRouting on the disaggregationDecider if it
// implements DisaggregationObserver. Called after each routing decision (both standard and
// prefill paths) to keep the decider's prefix cache current (BC-PD-28, R17, INV-7).
func (c *ClusterSimulator) notifyDisaggregationObserver(req *sim.Request, instanceID string) {
	if c.disaggregationDecider == nil {
		return
	}
	if obs, ok := c.disaggregationDecider.(sim.DisaggregationObserver); ok {
		obs.ObserveRouting(req, instanceID)
	}
}

// PoolMembership returns a copy of the pool role membership map (R8: no exported mutable maps).
// Returns nil when disaggregation is disabled.
func (c *ClusterSimulator) PoolMembership() map[string]PoolRole {
	if c.poolMembership == nil {
		return nil
	}
	result := make(map[string]PoolRole, len(c.poolMembership))
	for k, v := range c.poolMembership {
		result[k] = v
	}
	return result
}

// Trace returns the decision trace collected during simulation.
// Returns nil if trace-level was "none" (default).
func (c *ClusterSimulator) Trace() *trace.SimulationTrace {
	return c.trace
}

// PerInstanceMetrics returns the metrics for each individual instance.
// Panics if called before Run() has completed.
func (c *ClusterSimulator) PerInstanceMetrics() []*sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.PerInstanceMetrics() called before Run()")
	}
	metrics := make([]*sim.Metrics, len(c.instances))
	for i, inst := range c.instances {
		metrics[i] = inst.Metrics()
	}
	return metrics
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
		merged.PreemptionCount += m.PreemptionCount
		merged.KVAllocationFailures += m.KVAllocationFailures
		merged.DroppedUnservable += m.DroppedUnservable
		merged.CacheHitRate += m.CacheHitRate
		merged.KVThrashingRate += m.KVThrashingRate
		merged.StillQueued += m.StillQueued
		merged.StillRunning += m.StillRunning
	}
	if n := len(c.instances); n > 0 {
		merged.CacheHitRate /= float64(n)
		merged.KVThrashingRate /= float64(n)
	}
	return merged
}
