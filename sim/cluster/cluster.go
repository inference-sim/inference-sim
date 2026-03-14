package cluster

import (
	"container/heap"
	"fmt"
	"math"
	"sort"

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

	// PD disaggregation state (PR2)
	// parentRequests is intentionally never pruned during simulation; entries are retained for
	// post-simulation access via ParentRequests() and CollectPDMetrics(). The map grows by one
	// entry per disaggregated parent request. For typical simulation objects (< 100K requests),
	// memory growth is bounded and acceptable. Long-lived or high-volume callers should use
	// short-lived ClusterSimulator instances rather than accumulating requests across runs.
	parentRequests            map[string]*ParentRequest // parent request ID → tracking record
	pendingPrefillCompletions   map[string]string            // prefill sub-req ID → parent ID (global index)
	pendingDecodeCompletions    map[string]string            // decode sub-req ID → parent ID (global index)
	pendingPrefillByInstance    map[string]map[string]string // instance ID → (sub-req ID → parent ID); O(K) detection
	pendingDecodeByInstance     map[string]map[string]string // instance ID → (sub-req ID → parent ID); O(K) detection
	transfersInitiated        int
	transfersCompleted        int
	droppedAtDecodeKV         int               // decode sub-requests dropped due to KV allocation failure (R1, INV-1)
	prefillRoutingPolicy      sim.RoutingPolicy    // nil = use main routingPolicy
	decodeRoutingPolicy       sim.RoutingPolicy    // nil = use main routingPolicy
	prefillInstanceSlice      []*InstanceSimulator // pre-computed for O(K) pool snapshot; nil when disabled
	decodeInstanceSlice       []*InstanceSimulator // pre-computed for O(K) pool snapshot; nil when disabled

	// Transfer contention state (--pd-transfer-contention flag, INV-P2-2)
	activeTransfers                int   // currently in-flight transfers
	peakConcurrentTransfers        int   // max observed concurrent transfers
	transferDepthSum               int64 // running sum of activeTransfers (post-increment) at each transfer start
	transferStartCount             int64 // number of transfer start events (for mean calculation)
	contentionBookkeepingCorrupted bool  // set when activeTransfers goes negative (R1); triggers error in Run()
}

// NewClusterSimulator creates a ClusterSimulator with N instances.
// All workload generation now happens externally — requests are passed in directly.
// Panics if config.NumInstances < 1.
func NewClusterSimulator(config DeploymentConfig, requests []*sim.Request) *ClusterSimulator {
	if config.NumInstances < 1 {
		panic("ClusterSimulator: NumInstances must be >= 1")
	}
	// Validate pool topology BEFORE building membership or constructing instances,
	// so invalid configs fail fast before any allocation (R3).
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
		if err := ValidatePoolTopology(config.PrefillInstances, config.DecodeInstances, config.NumInstances); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		// Warn when instances are unassigned (prefill + decode < total).
		// Unassigned instances receive NO requests when PD is active: all admitted requests
		// route through pool-filtered snapshots (disaggregated → prefill pool,
		// non-disaggregated → decode pool). Unassigned instances sit idle.
		// This is allowed for warm-standby or future-use capacity, but is unusual and
		// likely a misconfiguration. Users intending fully-disaggregated clusters should
		// set prefill + decode == num-instances.
		if unassigned := config.NumInstances - config.PrefillInstances - config.DecodeInstances; unassigned > 0 {
			logrus.Warnf("[cluster] %d instance(s) are unassigned to any pool (prefill=%d, decode=%d, total=%d). "+
				"Unassigned instances will be idle when PD disaggregation is active. "+
				"Set --prefill-instances + --decode-instances == --num-instances to use all capacity.",
				unassigned, config.PrefillInstances, config.DecodeInstances, config.NumInstances)
		}
	}

	// R3: validate interference factors BEFORE instance construction so the
	// authored error messages are reachable and no partial allocation occurs.
	// Upper bound (MaxInterferenceFactor) prevents silent int64 overflow when
	// factor * stepTime exceeds MaxInt64 after math.Round (see interference.go).
	if config.PDInterferencePrefill < 0 || math.IsNaN(config.PDInterferencePrefill) || math.IsInf(config.PDInterferencePrefill, 0) || config.PDInterferencePrefill > MaxInterferenceFactor {
		panic(fmt.Sprintf("ClusterSimulator: PDInterferencePrefill must be a finite number in [0, %.0f], got %f", MaxInterferenceFactor, config.PDInterferencePrefill))
	}
	if config.PDInterferenceDecode < 0 || math.IsNaN(config.PDInterferenceDecode) || math.IsInf(config.PDInterferenceDecode, 0) || config.PDInterferenceDecode > MaxInterferenceFactor {
		panic(fmt.Sprintf("ClusterSimulator: PDInterferenceDecode must be a finite number in [0, %.0f], got %f", MaxInterferenceFactor, config.PDInterferenceDecode))
	}
	// R20: warn when interference factors are non-zero but the deployment is fully
	// disaggregated (all instances pool-assigned AND decider is "always"). In that case,
	// pool instances only receive phase-pure batches (INV-PD-2), so the interference
	// multiplier is always 1.0 and these parameters have no effect.
	// When direct-to-decode or prefix-threshold is active, non-disaggregated requests
	// may reach decode instances, creating mixed batches where interference applies.
	if (config.PDInterferencePrefill > 0 || config.PDInterferenceDecode > 0) &&
		config.PrefillInstances > 0 && config.DecodeInstances > 0 &&
		config.PDDecider == "always" {
		logrus.Warnf("[cluster] pd-interference-prefill/decode are non-zero but all instances are pool-assigned "+
			"(prefill-instances=%d, decode-instances=%d) with decider=%q. Pool instances serve only phase-pure batches "+
			"(INV-PD-2), so the interference multiplier is always 1.0. These parameters have no effect "+
			"in fully disaggregated deployments.",
			config.PrefillInstances, config.DecodeInstances, config.PDDecider)
	}

	// R3: validate per-pool overrides BEFORE instance construction so invalid configs
	// fail with a clear message rather than a cryptic panic inside the latency factory.
	if err := config.PrefillOverrides.Validate("prefill pool"); err != nil {
		panic(fmt.Sprintf("ClusterSimulator: %v", err))
	}
	if err := config.DecodeOverrides.Validate("decode pool"); err != nil {
		panic(fmt.Sprintf("ClusterSimulator: %v", err))
	}

	// Build pool membership from indices BEFORE instance construction
	// so we can resolve per-pool configs for each instance (INV-P2-1).
	var prePoolMembership map[string]PoolRole
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
		prePoolMembership = BuildPoolMembershipFromIndices(
			config.NumInstances, config.PrefillInstances, config.DecodeInstances,
		)
	}

	instances := make([]*InstanceSimulator, config.NumInstances)
	for idx := range instances {
		id := InstanceID(fmt.Sprintf("instance_%d", idx))
		role := PoolRole(0)
		if prePoolMembership != nil {
			role = prePoolMembership[string(id)]
		}
		simCfg := config.resolveConfigForRole(role)
		instances[idx] = newInstanceSimulatorCore(id, simCfg, config.PDInterferencePrefill, config.PDInterferenceDecode)
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

	// R3: reject PDTransferContention when PD disaggregation is not active.
	if config.PDTransferContention && config.PrefillInstances == 0 && config.DecodeInstances == 0 {
		panic("ClusterSimulator: PDTransferContention requires PD disaggregation (--prefill-instances and --decode-instances must be set)")
	}
	// PD disaggregation: validate transfer parameters and build runtime state.
	// Pool topology already validated above (before instance construction).
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
		// R3: validate PD transfer parameters at construction time.
		if config.PDKVBytesPerToken <= 0 {
			panic(fmt.Sprintf("ClusterSimulator: PDKVBytesPerToken must be > 0 when PD is enabled, got %d", config.PDKVBytesPerToken))
		}
		if config.PDTransferBandwidthGBps <= 0 || math.IsNaN(config.PDTransferBandwidthGBps) || math.IsInf(config.PDTransferBandwidthGBps, 0) {
			panic(fmt.Sprintf("ClusterSimulator: PDTransferBandwidthGBps must be a finite positive number when PD is enabled, got %f", config.PDTransferBandwidthGBps))
		}
		if config.PDTransferBaseLatencyMs < 0 || math.IsNaN(config.PDTransferBaseLatencyMs) || math.IsInf(config.PDTransferBaseLatencyMs, 0) {
			panic(fmt.Sprintf("ClusterSimulator: PDTransferBaseLatencyMs must be a finite non-negative number when PD is enabled, got %f", config.PDTransferBaseLatencyMs))
		}
		// C1/R3: guard against int64 overflow in KVTransferStartedEvent.Execute().
		// baseLatUs = PDTransferBaseLatencyMs * 1000; int64(math.Ceil(1e19)) wraps to MinInt64
		// on amd64, then the duration<1 floor silently clamps to 1 µs (wrong result, no error).
		// 3.6e9 ms = 1000 hours is an extreme but finite upper bound.
		const maxTransferBaseLatencyMs = 3_600_000_000.0 // 1000 hours
		if config.PDTransferBaseLatencyMs > maxTransferBaseLatencyMs {
			panic(fmt.Sprintf("ClusterSimulator: PDTransferBaseLatencyMs must be <= %.0f ms (1000 hours) to prevent int64 overflow in transfer duration, got %f", maxTransferBaseLatencyMs, config.PDTransferBaseLatencyMs))
		}
		// C6/R3: BlockSizeTokens must be > 0 when PD is enabled; NewParentRequest panics
		// if blockSizeTokens <= 0, and that panic fires inside an event handler during Run()
		// (R6 violation). Validate here so the error appears at construction time.
		// Note: NewSimulator also validates BlockSizeTokens > 0 per instance, so this check
		// is reached first only when called before instance construction.
		if config.BlockSizeTokens <= 0 {
			panic(fmt.Sprintf("ClusterSimulator: KVCacheConfig.BlockSizeTokens must be > 0 when PD is enabled, got %d", config.BlockSizeTokens))
		}
		// R3: guard against int64 overflow in KVTransferStartedEvent.Execute().
		// blockSizeBytes = BlockSizeTokens * PDKVBytesPerToken; if this overflows int64,
		// transferBytes becomes negative and the duration clamps to 1 µs (silent wrong result).
		if config.BlockSizeTokens > 0 && config.PDKVBytesPerToken > math.MaxInt64/config.BlockSizeTokens {
			panic(fmt.Sprintf("ClusterSimulator: BlockSizeTokens (%d) * PDKVBytesPerToken (%d) overflows int64 (R3)",
				config.BlockSizeTokens, config.PDKVBytesPerToken))
		}
		cs.poolMembership = prePoolMembership
		switch config.PDDecider {
		case "prefix-threshold":
			cs.disaggregationDecider = sim.NewPrefixThresholdDecider(config.PDPrefixThreshold, int(config.BlockSizeTokens))
		case "direct-to-decode":
			cs.disaggregationDecider = sim.NewDirectToDecodeDecider(config.PDDirectDecodeThreshold)
		default:
			cs.disaggregationDecider = sim.NewDisaggregationDecider(config.PDDecider)
		}
		cs.parentRequests = make(map[string]*ParentRequest)
		cs.pendingPrefillCompletions = make(map[string]string)
		cs.pendingDecodeCompletions = make(map[string]string)
		cs.pendingPrefillByInstance = make(map[string]map[string]string, config.PrefillInstances)
		cs.pendingDecodeByInstance = make(map[string]map[string]string, config.DecodeInstances)

		// Per-pool routing policies (use separate RNG partitions to avoid fragile coupling)
		if len(config.PrefillScorerConfigs) > 0 {
			cs.prefillRoutingPolicy = sim.NewRoutingPolicy("weighted", config.PrefillScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("prefill-router"))
		}
		if len(config.DecodeScorerConfigs) > 0 {
			cs.decodeRoutingPolicy = sim.NewRoutingPolicy("weighted", config.DecodeScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("decode-router"))
		}

		// Pre-compute per-pool instance slices for O(K) snapshot filtering in
		// buildPoolFilteredSnapshots (avoids O(N) linear scan per routing event).
		// Instances are appended in order (R2: deterministic).
		cs.prefillInstanceSlice = make([]*InstanceSimulator, 0, config.PrefillInstances)
		cs.decodeInstanceSlice = make([]*InstanceSimulator, 0, config.DecodeInstances)
		for _, inst := range instances {
			switch cs.poolMembership[string(inst.ID())] {
			case PoolRolePrefill:
				cs.prefillInstanceSlice = append(cs.prefillInstanceSlice, inst)
			case PoolRoleDecode:
				cs.decodeInstanceSlice = append(cs.decodeInstanceSlice, inst)
			}
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
					logrus.Errorf("inFlightRequests[%s] went negative (%d) after delta=%d (completed=%d, dropped=%d) — bookkeeping bug",
						instID, c.inFlightRequests[instID], delta, completedAfter-completedBefore, droppedAfter-droppedBefore)
					c.inFlightRequests[instID] = 0
				}

				// PD disaggregation: detect prefill/decode sub-request completions
				if c.poolsConfigured() {
					switch c.poolMembership[instID] {
					case PoolRolePrefill:
						c.detectPrefillCompletions(inst)
					case PoolRoleDecode:
						c.detectDecodeCompletions(inst)
					}
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
	// R1/INV-1: decode sub-requests dropped at KV allocation are not tracked by instance
	// metrics. Account for them in the aggregated DroppedUnservable count so that
	// injected_requests == completed + queued + running + dropped holds at cluster level.
	c.aggregatedMetrics.DroppedUnservable += c.droppedAtDecodeKV

	// Post-simulation PD diagnostics
	if c.poolsConfigured() {
		// INV-PD-3: transfer conservation — initiated_transfers == completed_transfers.
		// This is a hard invariant: a mismatch means the transfer pipeline lost or
		// duplicated a transfer, producing corrupt PD metrics. Return an error so
		// the caller can fail visibly rather than report misleading numbers.
		// Note: horizon truncation (completion events past the horizon) also violates INV-PD-3
		// and is caught here first, before any contention checks below.
		if c.transfersInitiated != c.transfersCompleted {
			msg := fmt.Sprintf("INV-PD-3 violated: transfersInitiated=%d != transfersCompleted=%d",
				c.transfersInitiated, c.transfersCompleted)
			if c.contentionBookkeepingCorrupted {
				msg += "; additionally, contention bookkeeping was corrupted (activeTransfers went negative)"
			}
			return fmt.Errorf("%s", msg)
		}
		// Contention bookkeeping corruption: if activeTransfers went negative during the run
		// (see negative guard in KVTransferCompletedEvent), contention metrics are invalid.
		// This can only be reached after INV-PD-3 passes (all transfers accounted for), so
		// horizon truncation is not the cause — a programming error in the event pipeline is.
		if c.contentionBookkeepingCorrupted {
			return fmt.Errorf("[cluster] transfer contention bookkeeping corrupted: activeTransfers went negative during simulation (R1); contention metrics are invalid — examine earlier logrus.Errorf output for root cause")
		}
		// Defense-in-depth: activeTransfers should be 0 when INV-PD-3 holds and no corruption
		// occurred. A non-zero residual here indicates an undetected bookkeeping imbalance.
		if c.config.PDTransferContention && c.activeTransfers != 0 {
			logrus.Warnf("[cluster] activeTransfers = %d at simulation end (unexpected: INV-PD-3 holds and no negative-guard correction was recorded — undetected bookkeeping imbalance)",
				c.activeTransfers)
		}
		// Orphaned pending completions at horizon — in-flight disaggregated requests
		// that never completed their pipeline phase. This includes requests dropped at the
		// instance level (MaxModelLen violations) and requests horizon-truncated mid-pipeline.
		// Both cases are expected: dropped requests are counted in DroppedUnservable; horizon-
		// truncated requests remain in StillQueued/StillRunning. Warn so operators can diagnose.
		// Clear both maps after logging: the detection maps are only useful during simulation,
		// and orphaned entries for dropped/truncated requests would accumulate across long
		// simulations. Clearing here releases the memory and prevents stale state (R1).
		if n := len(c.pendingPrefillCompletions); n > 0 {
			logrus.Warnf("[cluster] %d prefill sub-requests still pending at horizon (dropped at instance or horizon-truncated)", n)
			c.pendingPrefillCompletions = nil
			c.pendingPrefillByInstance = nil
		}
		if n := len(c.pendingDecodeCompletions); n > 0 {
			logrus.Warnf("[cluster] %d decode sub-requests still pending at horizon (horizon-truncated)", n)
			c.pendingDecodeCompletions = nil
			c.pendingDecodeByInstance = nil
		}
	}

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

// buildPoolFilteredSnapshots constructs routing snapshots for instances in the given pool role.
// Uses pre-computed per-pool instance slices for O(K) lookup (K = pool size) instead of
// scanning all instances (O(N)). Slices are populated at construction in sorted order (R2).
func (c *ClusterSimulator) buildPoolFilteredSnapshots(role PoolRole) []sim.RoutingSnapshot {
	var poolInstances []*InstanceSimulator
	switch role {
	case PoolRolePrefill:
		poolInstances = c.prefillInstanceSlice
	case PoolRoleDecode:
		poolInstances = c.decodeInstanceSlice
	default:
		// Fallback: linear scan for unrecognized roles (defensive, should not occur in practice).
		poolInstances = make([]*InstanceSimulator, 0, len(c.instances)/2+1)
		for _, inst := range c.instances {
			if c.poolMembership[string(inst.ID())] == role {
				poolInstances = append(poolInstances, inst)
			}
		}
	}
	snapshots := make([]sim.RoutingSnapshot, 0, len(poolInstances))
	for _, inst := range poolInstances {
		snap := c.snapshotProvider.Snapshot(inst.ID(), c.clock)
		snap.InFlightRequests = c.inFlightRequests[string(inst.ID())]
		snapshots = append(snapshots, snap)
	}
	return snapshots
}

// detectPrefillCompletions checks for newly completed prefill sub-requests on the given instance
// and schedules KV transfer events for each.
// Uses pendingPrefillByInstance[inst.ID()] for O(K) detection (K = sub-requests routed to this
// instance) instead of scanning all pending completions cluster-wide (old O(N) approach).
// Keys are processed in sorted order (R2) to ensure deterministic seqID assignment (INV-6).
func (c *ClusterSimulator) detectPrefillCompletions(inst *InstanceSimulator) {
	instID := string(inst.ID())
	instancePending, ok := c.pendingPrefillByInstance[instID]
	if !ok || len(instancePending) == 0 {
		return // no pending prefills for this instance
	}
	completionTimes := inst.Metrics().RequestCompletionTimes

	// R2/INV-6: collect and sort keys before processing so seqIDs are deterministic
	// across runs regardless of Go's map iteration order.
	// Only iterate sub-requests assigned to this instance (O(K) not O(all pending)).
	subReqIDs := make([]string, 0, len(instancePending))
	for subReqID := range instancePending {
		if _, completed := completionTimes[subReqID]; completed {
			subReqIDs = append(subReqIDs, subReqID)
		}
	}
	sort.Strings(subReqIDs)

	for _, subReqID := range subReqIDs {
		parentID := c.pendingPrefillCompletions[subReqID]
		parent := c.parentRequests[parentID]
		if parent == nil {
			// R1: nil parent is a programming error — parentRequests is populated in
			// DisaggregationDecisionEvent.Execute() and pendingPrefillCompletions in
			// PrefillRoutingEvent.Execute(). Both are guaranteed set before this fires.
			panic(fmt.Sprintf("detectPrefillCompletions: parentID %q not found in parentRequests (programming error)", parentID))
		}
		parent.PrefillCompleteTime = c.clock
		delete(c.pendingPrefillCompletions, subReqID)
		delete(instancePending, subReqID)

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

// detectDecodeCompletions checks for newly completed decode sub-requests on the given instance
// and records CompletionTime on the parent request, completing INV-PD-4.
// Uses pendingDecodeByInstance[inst.ID()] for O(K) detection (K = sub-requests routed to this
// instance) instead of scanning all pending completions cluster-wide.
// Keys are processed in sorted order (R2) to ensure deterministic processing (INV-6).
func (c *ClusterSimulator) detectDecodeCompletions(inst *InstanceSimulator) {
	instID := string(inst.ID())
	instancePending, ok := c.pendingDecodeByInstance[instID]
	if !ok || len(instancePending) == 0 {
		return // no pending decodes for this instance
	}
	completionTimes := inst.Metrics().RequestCompletionTimes

	// R2/INV-6: collect and sort keys before processing.
	// Only iterate sub-requests assigned to this instance (O(K) not O(all pending)).
	subReqIDs := make([]string, 0, len(instancePending))
	for subReqID := range instancePending {
		if _, completed := completionTimes[subReqID]; completed {
			subReqIDs = append(subReqIDs, subReqID)
		}
	}
	sort.Strings(subReqIDs)

	for _, subReqID := range subReqIDs {
		parentID := c.pendingDecodeCompletions[subReqID]
		parent := c.parentRequests[parentID]
		if parent == nil {
			// R1: nil parent is a programming error — parentRequests is populated in
			// DisaggregationDecisionEvent.Execute() and pendingDecodeCompletions in
			// DecodeRoutingEvent.Execute().
			panic(fmt.Sprintf("detectDecodeCompletions: parentID %q not found in parentRequests (programming error)", parentID))
		}
		parent.CompletionTime = c.clock
		delete(c.pendingDecodeCompletions, subReqID)
		delete(instancePending, subReqID)
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
	return c.droppedAtDecodeKV
}

// poolsConfigured returns true if PD disaggregation pool topology is active.
func (c *ClusterSimulator) poolsConfigured() bool {
	return c.poolMembership != nil
}

// PeakConcurrentTransfers returns the maximum number of concurrent KV transfers
// observed during the simulation. Returns 0 when contention is disabled.
func (c *ClusterSimulator) PeakConcurrentTransfers() int {
	return c.peakConcurrentTransfers
}

// MeanTransferQueueDepth returns the average number of concurrent transfers
// at each transfer initiation. Returns 0 when contention is disabled or no transfers occurred.
func (c *ClusterSimulator) MeanTransferQueueDepth() float64 {
	if c.transferStartCount == 0 {
		return 0
	}
	return float64(c.transferDepthSum) / float64(c.transferStartCount)
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

// ParentRequests returns a copy of all ParentRequest records sorted by ID.
// Panics if called before Run() completes (BC-11).
// Returns an empty slice when PD disaggregation is not active.
func (c *ClusterSimulator) ParentRequests() []*ParentRequest {
	if !c.hasRun {
		panic("ClusterSimulator.ParentRequests() called before Run()")
	}
	result := make([]*ParentRequest, 0, len(c.parentRequests))
	for _, pr := range c.parentRequests {
		result = append(result, pr)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}

// PerInstanceMetricsByID returns a map of instance ID → *sim.Metrics.
// Panics if called before Run() completes (BC-12).
// The returned map is a new map (not a reference to internal state), consistent with R8.
func (c *ClusterSimulator) PerInstanceMetricsByID() map[string]*sim.Metrics {
	if !c.hasRun {
		panic("ClusterSimulator.PerInstanceMetricsByID() called before Run()")
	}
	result := make(map[string]*sim.Metrics, len(c.instances))
	for _, inst := range c.instances {
		result[string(inst.ID())] = inst.Metrics()
	}
	return result
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
		merged.LengthCappedRequests += m.LengthCappedRequests
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
