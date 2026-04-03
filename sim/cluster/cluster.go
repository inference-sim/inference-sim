package cluster

import (
	"container/heap"
	"fmt"
	"math"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/latency"
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
	routingRejections    int                    // I13: count of requests rejected at routing (no routable instances)
	shedByTier           map[string]int         // per-SLOClass rejection counts (Phase 1B-1a)
	deferredQueue        []*sim.Request         // Batch/Background requests awaiting idle capacity (Phase 1B-1b)
	trace                *trace.SimulationTrace // nil when trace-level is "none" (BC-1: zero overhead)
	preGeneratedRequests []*sim.Request         // Pre-generated requests (all workload paths unified)
	inFlightRequests     map[string]int         // instance ID → dispatched-but-not-completed count (#463)
	poolMembership       map[string]PoolRole    // instance ID → pool role (nil when disaggregation disabled)
	disaggregationDecider sim.DisaggregationDecider // PD disaggregation decider (nil when disabled)

	// PD disaggregation state (PR2)
	parentRequests            map[string]*ParentRequest // parent request ID → tracking record
	pendingPrefillCompletions map[string]string         // prefill sub-req ID → parent ID
	pendingDecodeCompletions  map[string]string         // decode sub-req ID → parent ID
	transfersInitiated        int
	transfersCompleted        int
	pdPrefillCompletedCount   int                       // prefill sub-requests that completed (for INV-1 correction)
	pdDecodeCompletedCount    int                       // decode sub-requests that completed (for INV-1 in-flight tracking)
	pdDecodeTimedOutCount     int                       // decode sub-requests that timed out (for INV-1 in-flight tracking)
	droppedAtDecodeKV         int                       // requests dropped due to insufficient KV at decode
	prefillRoutingPolicy      sim.RoutingPolicy         // nil = use main routingPolicy
	decodeRoutingPolicy       sim.RoutingPolicy         // nil = use main routingPolicy

	// Transfer contention state (--pd-transfer-contention flag, INV-P2-2)
	activeTransfers                int
	peakConcurrentTransfers        int
	transferDepthSum               int64
	transferStartCount             int64
	contentionBookkeepingCorrupted bool

	// Phase 1A: node/GPU placement manager. Nil when NodePools is empty (backward-compat).
	placement *PlacementManager

	// Phase 1B-2a: per-tenant fair-share tracker. Nil when TenantBudgets is nil (backward-compat).
	tenantTracker *TenantTracker

	// sessionCallback is the raw onRequestDone parameter for session follow-up
	// generation in PD mode. Called from detectDecodeCompletions with the original
	// request (which carries SessionID). Separate from the per-instance closure to
	// avoid double-notifying tenantTracker (issue #884). Nil for non-session workloads.
	sessionCallback func(*sim.Request, int64) []*sim.Request

	// cacheQueryFn maps instance IDs to KV cache query functions for precise
	// prefix cache scoring. Built after instance construction; deferred instances
	// are added in NodeReadyEvent.Execute. Nil when no instances exist yet.
	cacheQueryFn map[string]func([]int) int

	// staleCache manages periodic snapshots of per-instance KV cache hash maps
	// for stale prefix cache scoring (issue #919). Nil when CacheSignalDelay == 0 (oracle mode).
	// Default CacheSignalDelay is 2s, matching llm-d's speculative TTL.
	staleCache *StaleCacheIndex

	// Flow control state (issue #882, GIE parity).
	// When flowControlEnabled is false, these fields are nil/zero (BC-1 pass-through).
	flowControlEnabled bool
	saturationDetector sim.SaturationDetector
	gatewayQueue       *GatewayQueue
}

// NewClusterSimulator creates a ClusterSimulator with N instances.
// All workload generation now happens externally — requests are passed in directly.
// onRequestDone is an optional callback invoked when a request reaches a terminal state
// (completed, length-capped, timed out, or dropped). The callback returns follow-up
// requests which are routed through the cluster pipeline (not injected locally).
// Pass nil for non-session workloads.
// Panics if config.NumInstances < 1.
func NewClusterSimulator(config DeploymentConfig, requests []*sim.Request, onRequestDone func(*sim.Request, int64) []*sim.Request) *ClusterSimulator {
	if config.NumInstances < 1 {
		panic("ClusterSimulator: NumInstances must be >= 1")
	}

	// Validate pool topology and overrides early (before instance construction).
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
		if err := ValidatePoolTopology(config.PrefillInstances, config.DecodeInstances, config.NumInstances); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		if err := config.PrefillOverrides.Validate("prefill pool"); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		if err := config.DecodeOverrides.Validate("decode pool"); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
	}

	// Validate KV bytes per token derivation early so KVTransferStartedEvent never
	// encounters a configuration error at runtime (the panic there is now unreachable).
	if config.PrefillInstances > 0 {
		if config.EffectivePrefillTP() <= 0 {
			panic("ClusterSimulator: PD disaggregation requires prefill TP > 0 (set --tp or --prefill-tp)")
		}
		if _, err := latency.KVBytesPerToken(config.ModelConfig, config.EffectivePrefillTP()); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: PD disaggregation requires valid ModelConfig for KV transfer sizing: %v", err))
		}
	}

	if config.PDTransferContention && config.PrefillInstances == 0 && config.DecodeInstances == 0 {
		panic("ClusterSimulator: PDTransferContention requires PD disaggregation (--prefill-instances and --decode-instances must be set)")
	}

	// Build pre-construction pool membership so instance construction can resolve per-pool config.
	// When disaggregation is disabled (PrefillInstances==0), prePoolMembership is nil and
	// all instances use the global config (backward-compatible).
	var prePoolMembership map[string]PoolRole
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
		prePoolMembership = BuildPoolMembershipFromIndices(config.NumInstances, config.PrefillInstances, config.DecodeInstances)
	}

	// instances and instanceMap are populated by the unified construction+placement loop below.
	// Declared here so they are available throughout NewClusterSimulator.
	instanceMap := make(map[InstanceID]*InstanceSimulator, config.NumInstances)

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

	// Bypass generic factory for "tier-shed": factory signature is float64-only and
	// cannot carry the int fields TierShedAdmission requires (research.md D-2).
	var admissionPolicy sim.AdmissionPolicy
	if config.AdmissionPolicy == "tier-shed" {
		if config.TierShedMinPriority == 0 {
			logrus.Warn("[cluster] tier-shed: TierShedMinPriority=0 admits all tiers under overload — policy behaves like AlwaysAdmit; set tier_shed_min_priority: 3 for Standard-and-above protection")
		}
		admissionPolicy = sim.NewTierShedAdmission(config.TierShedThreshold, config.TierShedMinPriority)
	} else {
		admissionPolicy = sim.NewAdmissionPolicy(config.AdmissionPolicy, config.TokenBucketCapacity, config.TokenBucketRefillRate)
	}

	cs := &ClusterSimulator{
		config:               config,
		instances:            make([]*InstanceSimulator, 0, config.NumInstances),
		rng:                  rng,
		preGeneratedRequests: requests,
		clusterEvents:        make(ClusterEventQueue, 0),
		admissionLatency:     config.AdmissionLatency,
		routingLatency:       config.RoutingLatency,
		admissionPolicy:      admissionPolicy,
		snapshotProvider: nil, // set after unified construction loop below
		routingPolicy:    nil, // set after instance construction (needs cacheQueryFn from instances)
		trace:                simTrace,
		inFlightRequests:     make(map[string]int, config.NumInstances),
		shedByTier:           make(map[string]int),
	}

	// PD disaggregation: set pool membership (topology already validated above)
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 {
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

		// Per-pool routing policies are created after the construction loop
		// (need cacheQueryFn from instances).

		logrus.Infof("[cluster] PD disaggregation enabled: %d prefill, %d decode instances, decider=%q",
			config.PrefillInstances, config.DecodeInstances, config.PDDecider)
	}

	// Phase 1A: initialize PlacementManager when node pools are configured.
	// Must happen BEFORE the unified construction loop so cs.placement is set.
	if len(config.NodePools) > 0 {
		provisionRng := rng.ForSubsystem(subsystemNodeProvisioning)
		loadingRng := rng.ForSubsystem(subsystemInstanceLoading)
		cs.placement = NewPlacementManager(config.NodePools, provisionRng, loadingRng, 0)
	}

	// Unified construction+placement loop: construct each InstanceSimulator AFTER placement
	// so the pool's GPU type (authoritative) is used instead of the CLI flag (SC-004).
	// TP=0 in ModelHardwareConfig means "not configured" — treat as 1 GPU per instance.
	tpDegree := config.TP
	if tpDegree < 1 {
		tpDegree = 1 // default to TP=1 when not explicitly set (R3: defensive correction with comment)
	}
	for idx := 0; idx < config.NumInstances; idx++ {
		id := InstanceID(fmt.Sprintf("instance_%d", idx))
		role := PoolRole(0)
		if prePoolMembership != nil {
			role = prePoolMembership[string(id)]
		}
		simCfg := config.resolveConfigForRole(role)

		if cs.placement != nil {
			// NodePools path: placement determines GPU type (authoritative).
			// Pass "" as gpuType so PlacementManager selects any available pool
			// (the pool's gpu_type is the authoritative source, not the CLI --gpu flag).
			nodeID, gpuIDs, matchedGPUType, err := cs.placement.PlaceInstance(id, config.Model, "", tpDegree)
			if err != nil {
				// No capacity — defer construction until NodeReadyEvent.
				// Pass "" as gpuType (any pool) to match AddPending's placement semantics.
				cs.placement.AddPending(id, config.Model, "", tpDegree, simCfg)
				continue
			}
			// Placement succeeded: use pool's GPU type (SC-004: pool-authoritative, not CLI flag).
			// Set GPU label and, when HWConfigByGPU is provided, override HWConfig so that
			// roofline/trained-roofline backends use the pool's hardware coefficients (issue #893).
			simCfg.GPU = matchedGPUType
			if hc, ok := config.HWConfigByGPU[matchedGPUType]; ok {
				if hc.TFlopsPeak <= 0 || hc.BwPeakTBs <= 0 {
					panic(fmt.Sprintf("HWConfigByGPU[%q]: TFlopsPeak and BwPeakTBs must be positive, got TFlopsPeak=%v BwPeakTBs=%v",
						matchedGPUType, hc.TFlopsPeak, hc.BwPeakTBs))
				}
				simCfg.HWConfig = hc
			}
			inst := NewInstanceSimulator(id, simCfg)
			inst.Model = config.Model
			inst.nodeID = nodeID
			inst.allocatedGPUIDs = gpuIDs
			inst.warmUpRemaining = config.InstanceLifecycle.WarmUpRequestCount
			inst.TransitionTo(InstanceStateLoading)
			cs.scheduleInstanceLoadedEvent(inst)
			cs.instances = append(cs.instances, inst)
			instanceMap[id] = inst
			cs.inFlightRequests[string(id)] = 0
		} else {
			// No NodePools: the CLI --gpu flag (config.ModelHardwareConfig.GPU, accessed via
			// DeploymentConfig's embedded SimConfig) is the authoritative source (backward-compat).
			// simCfg.GPU is already set — resolveConfigForRole returns config.SimConfig as-is
			// for the default role, preserving ModelHardwareConfig.GPU from the CLI flag.
			inst := NewInstanceSimulator(id, simCfg)
			inst.Model = config.Model
			inst.warmUpRemaining = config.InstanceLifecycle.WarmUpRequestCount
			if inst.warmUpRemaining > 0 {
				inst.TransitionTo(InstanceStateWarmingUp)
			} else {
				inst.TransitionTo(InstanceStateActive)
			}
			cs.instances = append(cs.instances, inst)
			instanceMap[id] = inst
			cs.inFlightRequests[string(id)] = 0
		}
	}

	// Initialize snapshot provider with exactly the placed instances.
	// Deferred instances are registered via CachedSnapshotProvider.AddInstance
	// when NodeReadyEvent.Execute constructs them (Phase 4, T017).
	cs.snapshotProvider = NewCachedSnapshotProvider(instanceMap, newObservabilityConfig(config.SnapshotRefreshInterval))

	// Build cacheQueryFn from constructed instances for precise prefix cache scoring.
	if config.CacheSignalDelay > 0 {
		// Stale mode: scorers query periodically-refreshed snapshots (issue #919).
		cs.staleCache = NewStaleCacheIndex(instanceMap, config.CacheSignalDelay)
		cs.cacheQueryFn = cs.staleCache.BuildCacheQueryFn()
	} else {
		// Oracle mode (default): scorers query live KV cache state.
		cs.cacheQueryFn = make(map[string]func([]int) int, len(cs.instances))
		for _, inst := range cs.instances {
			id := string(inst.ID())
			inst := inst // capture for closure
			cs.cacheQueryFn[id] = func(tokens []int) int {
				return inst.GetCachedBlockCount(tokens)
			}
		}
	}

	// Create routing policies now that cacheQueryFn is available.
	cs.routingPolicy = sim.NewRoutingPolicyWithCache(config.RoutingPolicy, config.RoutingScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem(sim.SubsystemRouter), cs.cacheQueryFn)
	if len(config.PrefillScorerConfigs) > 0 {
		cs.prefillRoutingPolicy = sim.NewRoutingPolicyWithCache("weighted", config.PrefillScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("prefill-router"), cs.cacheQueryFn)
	}
	if len(config.DecodeScorerConfigs) > 0 {
		cs.decodeRoutingPolicy = sim.NewRoutingPolicyWithCache("weighted", config.DecodeScorerConfigs, config.BlockSizeTokens, rng.ForSubsystem("decode-router"), cs.cacheQueryFn)
	}

	// Phase 1B-2a: initialize TenantTracker when TenantBudgets is configured (issue #811).
	// totalCapacity = NumInstances × MaxRunningReqs (batch size proxy for cluster-wide capacity).
	if config.TenantBudgets != nil {
		totalCapacity := config.NumInstances * int(config.MaxRunningReqs)
		if len(config.TenantBudgets) > 0 && totalCapacity == 0 {
			logrus.Warnf("[cluster] tenant_budgets configured but totalCapacity=0 (NumInstances=%d, MaxRunningReqs=%d); all budgeted tenants will be immediately over-budget — set max_running_reqs > 0",
				config.NumInstances, config.MaxRunningReqs)
		}
		cs.tenantTracker = NewTenantTracker(config.TenantBudgets, totalCapacity)
	}

	// Flow control: gateway queue with saturation-gated dispatch (issue #882).
	// When disabled (default), the pipeline is unchanged — requests flow directly
	// from admission to routing (BC-1 pass-through equivalence).
	if config.FlowControlEnabled {
		dispatchOrder := config.FlowControlDispatchOrder
		if dispatchOrder == "" {
			dispatchOrder = "fifo"
		}
		cs.flowControlEnabled = true
		cs.gatewayQueue = NewGatewayQueue(dispatchOrder, config.FlowControlMaxQueueDepth)
		cs.saturationDetector = sim.NewSaturationDetector(
			config.FlowControlDetector,
			config.FlowControlQueueDepthThreshold,
			config.FlowControlKVCacheUtilThreshold,
			config.FlowControlMaxConcurrency,
		)
		logrus.Infof("[cluster] flow control enabled: detector=%q, dispatch=%q, maxDepth=%d",
			config.FlowControlDetector, dispatchOrder, config.FlowControlMaxQueueDepth)
	}

	// Startup warning: horizon too small for pipeline (BC-1)
	pipelineLatency := cs.admissionLatency + cs.routingLatency
	if cs.config.Horizon > 0 && cs.config.Horizon < pipelineLatency {
		logrus.Warnf("[cluster] horizon (%d) < pipeline latency (%d); no requests can complete — increase --horizon or reduce admission/routing latency",
			cs.config.Horizon, pipelineLatency)
	}

	// Store raw callback for PD session follow-up (issue #884).
	cs.sessionCallback = onRequestDone

	// Wire OnRequestDone callback on each instance (BC-9: follow-ups route through cluster pipeline).
	// The callback pushes follow-up requests as ClusterArrivalEvents, ensuring they go through
	// admission → routing → instance injection. The callback returns nil so the per-instance
	// simulator does not inject locally.
	// Phase 1B-2a: also notify tenantTracker on completion when budgets are configured.
	if onRequestDone != nil || cs.tenantTracker != nil {
		for _, inst := range cs.instances {
			inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
				// Phase 1B-2a: release tenant in-flight slot on every terminal state.
				if cs.tenantTracker != nil {
					cs.tenantTracker.OnComplete(req.TenantID)
				}
				if onRequestDone == nil {
					return nil
				}
				nextReqs := onRequestDone(req, tick)
				for _, next := range nextReqs {
					heap.Push(&cs.clusterEvents, clusterEventEntry{
						event: &ClusterArrivalEvent{time: next.ArrivalTime, request: next},
						seqID: cs.nextSeqID(),
					})
				}
				return nil // don't inject locally — route through cluster pipeline
			}
		}
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
			timedOutBefore := inst.Metrics().TimedOutRequests

			ev := inst.ProcessNextEvent()
			_ = ev // Event type no longer used for decrement

			// Completion-based decrement (#463, BC-3, BC-7): InFlightRequests tracks the full
			// dispatch-to-completion window. Decrement by the number of newly completed,
			// dropped-unservable, or timed-out requests.
			completedAfter := inst.Metrics().CompletedRequests
			droppedAfter := inst.Metrics().DroppedUnservable
			timedOutAfter := inst.Metrics().TimedOutRequests
			delta := (completedAfter - completedBefore) + (droppedAfter - droppedBefore) + (timedOutAfter - timedOutBefore)
			if delta > 0 {
				c.inFlightRequests[instID] -= delta
				if c.inFlightRequests[instID] < 0 {
					// Warn-and-clamp: inFlightRequests is a best-effort routing signal
					// (INV-7); it recovers from delta mis-accounting and does not corrupt
					// deterministic metrics. Contrast with activeTransfers (contention
					// subsystem) which uses a hard error because contention metrics are
					// meaningless once the counter is wrong.
					logrus.Warnf("inFlightRequests[%s] went negative (%d) after delta=%d (completed=%d, dropped=%d, timedOut=%d) — bookkeeping bug",
						instID, c.inFlightRequests[instID], delta, completedAfter-completedBefore, droppedAfter-droppedBefore, timedOutAfter-timedOutBefore)
					c.inFlightRequests[instID] = 0
				}
				// T042: consume warm-up slots for newly completed requests (Phase 1A).
				// Each completion on a WarmingUp instance counts against the warm-up budget.
				completionDelta := int(completedAfter - completedBefore)
				for i := 0; i < completionDelta; i++ {
					if inst.IsWarmingUp() {
						inst.ConsumeWarmUpRequest()
					}
				}

				// Flow control: completion-triggered dispatch (BC-4).
				// Each completion opens capacity — try to dequeue from gateway queue.
				// Loop up to delta times so batch completions can dispatch multiple requests.
				// Early-exit when saturated or queue empty to avoid redundant buildRouterState calls.
				if c.flowControlEnabled {
					for i := 0; i < delta; i++ {
						if c.gatewayQueue.Len() == 0 {
							break
						}
						if !c.tryDispatchFromGatewayQueue() {
							break // saturated — no point rebuilding state for remaining iterations
						}
					}
				}
			}

			// T042: drain completion accounting (Phase 1A).
			// When a Draining instance has no more queued or running requests,
			// transition it to Terminated and release its GPU allocations.
			if inst.State == InstanceStateDraining && inst.QueueDepth() == 0 && inst.BatchSize() == 0 {
				inst.TransitionTo(InstanceStateTerminated)
				c.releaseInstanceGPUs(inst)
				if c.staleCache != nil {
					c.staleCache.RemoveInstance(inst.ID())
				}
				delete(c.cacheQueryFn, string(inst.ID()))
				// I1: a non-zero inFlightRequests at termination time indicates a bookkeeping bug.
				// This would cause isBusy() to permanently return true, silently stranding
				// all deferred Batch/Background requests until horizon.
				if c.inFlightRequests[instID] != 0 {
					logrus.Warnf("[cluster] instance %s terminated with inFlightRequests=%d — bookkeeping bug; deferred queue may stall",
						instID, c.inFlightRequests[instID])
				}
			}

			// PD disaggregation: detect prefill/decode sub-request completions
			if c.poolsConfigured() {
				if c.poolMembership[instID] == PoolRolePrefill {
					c.detectPrefillCompletions(inst)
				}
				if c.poolMembership[instID] == PoolRoleDecode {
					c.detectDecodeCompletions(inst)
				}
			}
		}

		// Phase 1B-1b: after each event, promote deferred Batch/Background requests
		// if the cluster has become idle. INV-8: ensures no stall while deferred work waits.
		if len(c.deferredQueue) > 0 && !c.isBusy() {
			c.promoteDeferred()
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

	// R1/INV-1: PD disaggregation conservation correction.
	// Each disaggregated request generates two sub-requests (prefill + decode) that
	// complete on separate instances. aggregateMetrics() naively sums CompletedRequests
	// across all instances, double-counting: prefill completion + decode completion = 2
	// for each original request. Subtract prefill completions to restore correct count.
	if c.pdPrefillCompletedCount > 0 {
		c.aggregatedMetrics.CompletedRequests -= c.pdPrefillCompletedCount
	}
	// Requests dropped at decode KV allocation: the prefill sub-request already
	// completed (counted above and subtracted), but the original request is lost.
	// Count as DroppedUnservable for INV-1 conservation.
	if c.droppedAtDecodeKV > 0 {
		c.aggregatedMetrics.DroppedUnservable += c.droppedAtDecodeKV
	}
	// In-flight PD transfers: requests whose prefill completed but decode hasn't
	// finished or been dropped yet (e.g., simulation ended at bounded horizon while
	// KV transfer was in progress). These requests were subtracted from CompletedRequests
	// but don't appear in any instance's StillQueued/StillRunning/DroppedUnservable.
	// Count them as StillRunning for conservation.
	//
	// Distinguish four sub-states of "prefill completed but decode not done":
	// - pendingDecodeCompletions: decode sub-requests already injected into instances
	//   (appear in instance StillQueued/StillRunning via Finalize — do NOT add again)
	// - pdInTransfer: requests still in KV transfer or cluster event queue
	//   (not on any instance — must be added to StillRunning)
	// - timed-out prefills: entries may remain in pendingPrefillCompletions but
	//   pdPrefillCompletedCount was NOT incremented; the timeout is already counted
	//   in instance TimedOutRequests → aggregated via aggregateMetrics(). No correction needed.
	// - timed-out decodes: counted in pdDecodeTimedOutCount; already in instance
	//   TimedOutRequests via aggregateMetrics(). Subtracted here to keep pdInTransfer = 0.
	pdInTransfer := c.pdPrefillCompletedCount - c.pdDecodeCompletedCount - c.pdDecodeTimedOutCount - c.droppedAtDecodeKV - len(c.pendingDecodeCompletions)
	if pdInTransfer > 0 {
		c.aggregatedMetrics.StillRunning += pdInTransfer
	} else if pdInTransfer < 0 {
		logrus.Warnf("[cluster] pdInTransfer = %d (negative): prefillCompleted=%d, decodeCompleted=%d, decodeTimedOut=%d, droppedAtDecodeKV=%d, pendingDecode=%d — bookkeeping bug in PD disaggregation accounting",
			pdInTransfer, c.pdPrefillCompletedCount, c.pdDecodeCompletedCount, c.pdDecodeTimedOutCount, c.droppedAtDecodeKV, len(c.pendingDecodeCompletions))
	}

	// INV-PD-6: Project sub-request metrics to parent-request granularity.
	// aggregateMetrics() merges per-instance maps keyed by sub-request IDs
	// (req_N_prefill, req_N_decode). Replace with parent-keyed entries so
	// user-facing distributions reflect the full request lifecycle.
	c.projectPDMetrics()

	// Post-simulation contention bookkeeping checks (INV-P2-2)
	if c.contentionBookkeepingCorrupted {
		return fmt.Errorf("contention bookkeeping corrupted: activeTransfers went negative during simulation — contention metrics are invalid")
	}
	if c.config.PDTransferContention && c.activeTransfers != 0 {
		logrus.Warnf("[cluster] post-simulation: activeTransfers = %d (expected 0), initiated=%d completed=%d — contention metrics (PeakConcurrentTransfers, MeanTransferQueueDepth) may be inflated if horizon cut off in-flight transfers",
			c.activeTransfers, c.transfersInitiated, c.transfersCompleted)
	}

	// Flow control: log gateway queue state at simulation end
	if c.flowControlEnabled && c.gatewayQueue.Len() > 0 {
		logrus.Warnf("[cluster] %d requests remain in gateway queue at simulation end", c.gatewayQueue.Len())
	}

	// Post-simulation diagnostic warnings (BC-2, BC-3)
	if c.aggregatedMetrics.CompletedRequests == 0 {
		if c.rejectedRequests > 0 {
			logrus.Warnf("[cluster] all %d requests rejected by admission policy %q — no requests completed",
				c.rejectedRequests, c.config.AdmissionPolicy)
		} else if c.aggregatedMetrics.TimedOutRequests > 0 {
			logrus.Warnf("[cluster] no requests completed — %d of %d requests timed out (client timeout exceeded, likely KV pressure)",
				c.aggregatedMetrics.TimedOutRequests,
				c.aggregatedMetrics.TimedOutRequests+c.aggregatedMetrics.DroppedUnservable)
		} else if len(c.deferredQueue) > 0 {
			logrus.Warnf("[cluster] no requests completed — %d batch/background requests remain deferred at horizon (cluster never became idle; mix in standard/critical traffic to trigger promotion)", len(c.deferredQueue))
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

// poolsConfigured returns true if PD disaggregation pool topology is active.
func (c *ClusterSimulator) poolsConfigured() bool {
	return c.poolMembership != nil
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

// ParentRequests returns a sorted slice of defensive copies of parent request tracking records.
// Each ParentRequest struct is copied by value so callers cannot mutate lifecycle timestamps (R8).
// Note: OriginalRequest and DecodeSubReq are shared *sim.Request pointers — callers must not mutate via them.
// Panics if called before Run() completes. Returns an empty (non-nil) slice when disaggregation is disabled,
// allowing callers to range over the result without a nil check.
func (c *ClusterSimulator) ParentRequests() []*ParentRequest {
	if !c.hasRun {
		panic("ClusterSimulator.ParentRequests() called before Run()")
	}
	result := make([]*ParentRequest, 0, len(c.parentRequests))
	for _, pr := range c.parentRequests {
		cp := *pr
		result = append(result, &cp)
	}
	sort.Slice(result, func(i, j int) bool { return result[i].ID < result[j].ID })
	return result
}

// buildPoolFilteredSnapshots constructs routing snapshots filtered to a specific pool role.
// Filters by IsRoutable() for parity with buildRouterState (R23), then by pool role.
// Model filter is intentionally omitted: all instances in a DeploymentConfig share config.Model,
// so pool-role filtering is sufficient. If multi-model PD clusters are added, add model filtering here.
// Preserves instance order from c.instances for determinism (R2).
func (c *ClusterSimulator) buildPoolFilteredSnapshots(role PoolRole) []sim.RoutingSnapshot {
	allSnapshots := make([]sim.RoutingSnapshot, 0, len(c.instances))
	for _, inst := range c.instances {
		if !inst.IsRoutable() {
			continue
		}
		snap := c.snapshotProvider.Snapshot(inst.ID(), c.clock)
		snap.InFlightRequests = c.inFlightRequests[string(inst.ID())]
		allSnapshots = append(allSnapshots, snap)
	}
	return FilterSnapshotsByPool(allSnapshots, c.poolMembership, role)
}

// detectPrefillCompletions checks for newly completed prefill sub-requests on the given instance
// and schedules KV transfer events for each.
// R2/INV-6: Collects completed IDs into a sorted slice before processing to ensure
// deterministic nextSeqID() assignment regardless of Go's random map iteration order.
func (c *ClusterSimulator) detectPrefillCompletions(inst *InstanceSimulator) {
	instID := string(inst.ID())
	// Phase 1: collect completed sub-request IDs (sorted for determinism)
	var completedIDs []string
	for subReqID, parentID := range c.pendingPrefillCompletions {
		parent := c.parentRequests[parentID]
		if parent == nil || string(parent.PrefillInstanceID) != instID {
			continue
		}
		if _, completed := inst.Metrics().RequestCompletionTimes[subReqID]; completed {
			completedIDs = append(completedIDs, subReqID)
		}
	}
	sort.Strings(completedIDs)

	// Phase 2: process in deterministic order
	for _, subReqID := range completedIDs {
		parentID := c.pendingPrefillCompletions[subReqID]
		parent := c.parentRequests[parentID]
		parent.PrefillCompleteTime = c.clock
		delete(c.pendingPrefillCompletions, subReqID)
		c.pdPrefillCompletedCount++

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

// detectDecodeCompletions checks for newly completed or timed-out decode sub-requests
// on the given instance and sets the parent request's CompletionTime.
// R2/INV-6: Collects IDs into sorted slices before processing for determinism.
func (c *ClusterSimulator) detectDecodeCompletions(inst *InstanceSimulator) {
	instID := string(inst.ID())
	// Phase 1: collect completed and timed-out sub-request IDs (sorted for determinism)
	var completedIDs []string
	var timedOutIDs []string
	for subReqID, parentID := range c.pendingDecodeCompletions {
		parent := c.parentRequests[parentID]
		if parent == nil || string(parent.DecodeInstanceID) != instID {
			continue
		}
		if _, completed := inst.Metrics().RequestCompletionTimes[subReqID]; completed {
			completedIDs = append(completedIDs, subReqID)
		} else if parent.DecodeSubReq != nil && parent.DecodeSubReq.State == sim.StateTimedOut {
			timedOutIDs = append(timedOutIDs, subReqID)
		}
	}
	sort.Strings(completedIDs)
	sort.Strings(timedOutIDs)

	// Phase 2: process completions in deterministic order
	for _, subReqID := range completedIDs {
		parent := c.parentRequests[c.pendingDecodeCompletions[subReqID]]
		// Include PostDecodeFixedOverhead so parent.CompletionTime represents the
		// client-visible completion time, matching non-PD E2E semantics (issue #846).
		// For blackbox/roofline/cross-model (overhead=0), value is byte-identical to before.
		// No zero-output guard needed: decode sub-requests always carry the full
		// output token list from the original request (set in KVTransferCompletedEvent.Execute).
		parent.CompletionTime = c.clock + inst.PostDecodeFixedOverhead()
		delete(c.pendingDecodeCompletions, subReqID)
		c.pdDecodeCompletedCount++

		// Issue #884: trigger session follow-up for the original (parent) request.
		// The per-instance OnRequestDone fires for the decode sub-request (no
		// SessionID), so SessionManager never sees PD completions. We call
		// sessionCallback directly with the original request to generate follow-ups.
		if c.sessionCallback != nil {
			// Value copy to avoid mutating the shared *sim.Request pointer
			// (contract at ParentRequests: callers must not mutate via OriginalRequest).
			origCopy := *parent.OriginalRequest
			origCopy.State = sim.StateCompleted
			// Use the decode sub-request's actual ProgressIndex for accurate context
			// accumulation (session.go:163). For length-capped decode sub-requests
			// (BC-5 force-completion), MaxOutputLen overstates the actual output;
			// DecodeSubReq.ProgressIndex reflects the true final position.
			// (blis replay passes onRequestDone=nil, so this code never runs in replay mode.)
			origCopy.ProgressIndex = parent.DecodeSubReq.ProgressIndex
			nextReqs := c.sessionCallback(&origCopy, parent.CompletionTime)
			for _, next := range nextReqs {
				heap.Push(&c.clusterEvents, clusterEventEntry{
					event: &ClusterArrivalEvent{time: next.ArrivalTime, request: next},
					seqID: c.nextSeqID(),
				})
			}
		}
	}

	// Phase 3: process timed-out decode sub-requests (INV-11 session completeness).
	// Non-PD equivalent: TimeoutEvent.Execute calls OnRequestDone with StateTimedOut →
	// SessionManager cancels the session. The PD path needs the same treatment.
	for _, subReqID := range timedOutIDs {
		parent := c.parentRequests[c.pendingDecodeCompletions[subReqID]]
		parent.CompletionTime = c.clock
		delete(c.pendingDecodeCompletions, subReqID)
		c.pdDecodeTimedOutCount++

		if c.sessionCallback != nil {
			origCopy := *parent.OriginalRequest
			origCopy.State = sim.StateTimedOut
			origCopy.ProgressIndex = parent.DecodeSubReq.ProgressIndex
			// SessionManager.OnComplete cancels the session for StateTimedOut (session.go:112).
			// No follow-ups expected, but handle defensively.
			nextReqs := c.sessionCallback(&origCopy, parent.CompletionTime)
			for _, next := range nextReqs {
				heap.Push(&c.clusterEvents, clusterEventEntry{
					event: &ClusterArrivalEvent{time: next.ArrivalTime, request: next},
					seqID: c.nextSeqID(),
				})
			}
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
// Note (Phase 1B-1b): INV-1 conservation at cluster level requires callers to also add
// DeferredQueueLen() for the deferred-horizon-interrupted bucket. AggregatedMetrics alone
// does not include deferred-at-horizon requests.
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

// RoutingRejections returns the count of requests rejected at routing because no
// routable instances were available (I13). Distinct from admission rejections.
func (c *ClusterSimulator) RoutingRejections() int {
	return c.routingRejections
}

// ShedByTier returns a copy of per-SLOClass rejection counts recorded during tier-shed admission.
// The map is populated only when AdmissionPolicy is "tier-shed"; returns an empty map otherwise.
// Returns a defensive copy so callers cannot mutate the internal counter (R8).
// Panics if called before Run() completes.
func (c *ClusterSimulator) ShedByTier() map[string]int {
	if !c.hasRun {
		panic("ClusterSimulator.ShedByTier() called before Run()")
	}
	result := make(map[string]int, len(c.shedByTier))
	for k, v := range c.shedByTier {
		result[k] = v
	}
	return result
}

// isBusy returns true when any non-terminated instance has non-zero effective load.
// Uses the three-component definition: QueueDepth + BatchSize + InFlightRequests > 0.
// Skips instances in InstanceStateTerminated state — stale inFlightRequests on terminated
// instances must not count as load (otherwise a recently terminated instance with residual
// accounting would permanently block deferred-queue promotion).
// An empty instance pool returns false (not busy).
// Called by the deferred queue pre-admission intercept and the idle-capacity promotion check.
func (c *ClusterSimulator) isBusy() bool {
	for _, inst := range c.instances {
		if inst.State == InstanceStateTerminated {
			continue // stale inFlightRequests on terminated instances must not count as load
		}
		if inst.QueueDepth()+inst.BatchSize()+c.inFlightRequests[string(inst.ID())] > 0 {
			return true
		}
	}
	return false
}

// promoteDeferred injects all deferred requests as ClusterArrivalEvents at the current clock.
// Called when isBusy() transitions to false. Truncates deferredQueue after injection.
// INV-8: ensures work-conserving behaviour — deferred requests re-enter the pipeline
// within the same scheduling step as the idle transition.
//
// Re-deferral: with non-zero admission latency, standard traffic arriving in the
// [clock, clock+admissionLatency] window may make isBusy() return true before a
// promoted request reaches AdmissionDecisionEvent, causing it to be re-deferred.
// This is intentional (Decision 4 in research.md) but may inflate DeferredHorizonInterrupted
// counts under continuous light standard load.
func (c *ClusterSimulator) promoteDeferred() {
	logrus.Debugf("[cluster] promoting %d deferred requests at tick %d", len(c.deferredQueue), c.clock)
	for _, req := range c.deferredQueue {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &ClusterArrivalEvent{time: c.clock, request: req},
			seqID: c.nextSeqID(),
		})
	}
	c.deferredQueue = c.deferredQueue[:0]
}

// DeferredQueueLen returns the number of Batch/Background requests still in the
// deferred queue at simulation end (i.e., deferred_horizon_interrupted count).
// Panics if called before Run() completes.
// Used by cmd/ to populate RawMetrics.DeferredHorizonInterrupted (Phase 1B-1b).
func (c *ClusterSimulator) DeferredQueueLen() int {
	if !c.hasRun {
		panic("ClusterSimulator.DeferredQueueLen() called before Run()")
	}
	return len(c.deferredQueue)
}

// GatewayQueueDepth returns the number of requests still in the gateway queue
// at simulation end. Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueDepth() int {
	if c.gatewayQueue == nil {
		return 0
	}
	return c.gatewayQueue.Len()
}

// GatewayQueueShed returns the number of requests shed from the gateway queue
// due to capacity limits. Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueShed() int {
	if c.gatewayQueue == nil {
		return 0
	}
	return c.gatewayQueue.ShedCount()
}

// tryDispatchFromGatewayQueue attempts to dispatch one request from the gateway queue.
// Called after each completion (BC-4) and after each enqueue (for NeverSaturated pass-through).
// Builds fresh RouterState at dispatch time for late binding (BC-3).
// Returns true if a request was dispatched, false if saturated or queue empty.
func (c *ClusterSimulator) tryDispatchFromGatewayQueue() bool {
	if c.gatewayQueue == nil || c.gatewayQueue.Len() == 0 {
		return false
	}
	// Build fresh state for late binding (BC-3)
	state := buildRouterState(c, nil)
	sat := c.saturationDetector.Saturation(state)
	if sat >= 1.0 {
		logrus.Debugf("[cluster] tryDispatch: held (saturation=%.2f, snapshots=%d, queueLen=%d)",
			sat, len(state.Snapshots), c.gatewayQueue.Len())
		return false // hold until next completion
	}
	req := c.gatewayQueue.Dequeue()
	if req == nil {
		return false
	}
	req.GatewayDispatchTime = c.clock

	// Schedule routing or disaggregation event (BC-9)
	if c.poolsConfigured() {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &DisaggregationDecisionEvent{
				time:    c.clock,
				request: req,
			},
			seqID: c.nextSeqID(),
		})
	} else {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &RoutingDecisionEvent{
				time:    c.clock + c.routingLatency,
				request: req,
			},
			seqID: c.nextSeqID(),
		})
	}
	return true
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

// PerInstanceMetricsByID returns a map of instance ID → *sim.Metrics.
// Panics if called before Run() completes (R1).
// The returned map is a new map (R8), but the *sim.Metrics values are live pointers to
// instance-owned structs — callers must not mutate fields through them.
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

// notifyDisaggregationObserver calls ObserveRouting on the disaggregationDecider if it
// implements sim.DisaggregationObserver. Called synchronously within the event loop,
// so the prefix cache is always current at the next Decide() call.
func (c *ClusterSimulator) notifyDisaggregationObserver(req *sim.Request, instanceID string) {
	if c.disaggregationDecider == nil {
		return
	}
	if obs, ok := c.disaggregationDecider.(sim.DisaggregationObserver); ok {
		obs.ObserveRouting(req, instanceID)
	}
}

// PeakConcurrentTransfers returns the maximum number of KV transfers in flight simultaneously.
// Returns 0 when --pd-transfer-contention is disabled (backward-compat).
func (c *ClusterSimulator) PeakConcurrentTransfers() int {
	return c.peakConcurrentTransfers
}

// MeanTransferQueueDepth returns the mean number of active concurrent transfers sampled at each
// transfer initiation event (arrival-weighted mean, not a time-average). Specifically:
//
//	sum(activeTransfers at each start event) / count(start events)
//
// The activeTransfers count is taken post-increment, so it includes the initiating transfer
// itself. For example, with fully sequential transfers the mean is exactly 1.0.
//
// This is not equivalent to a time-averaged queue depth (Little's Law denominator); it measures
// how many transfers were in flight at the moment each new transfer began, including the new one.
// Returns 0 when --pd-transfer-contention is disabled or no transfers occurred.
func (c *ClusterSimulator) MeanTransferQueueDepth() float64 {
	if c.transferStartCount == 0 {
		return 0
	}
	return float64(c.transferDepthSum) / float64(c.transferStartCount)
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
		merged.TimedOutRequests += m.TimedOutRequests
		merged.CacheHitRate += m.CacheHitRate
		merged.KVThrashingRate += m.KVThrashingRate
		merged.StillQueued += m.StillQueued
		merged.StillRunning += m.StillRunning
	}
	if n := len(c.instances); n > 0 {
		merged.CacheHitRate /= float64(n)
		merged.KVThrashingRate /= float64(n)
	}

	// T042: apply warm-up TTFT factor to requests served during warm-up (Phase 1A, R23).
	// C4 (known simplification): The penalty is applied post-hoc to recorded TTFTs rather than
	// during token generation. This means scheduling decisions during warm-up don't see inflated
	// TTFTs. Acceptable for Phase 1A; a pre-hoc model would require latency model integration.
	// Applied uniformly across all TTFT recording paths.
	// warmUpRequestIDs is cleared unconditionally to prevent unbounded memory growth,
	// even when factor <= 1.0 (e.g., default config where effectiveWarmUpFactor returns 1.0).
	factor := c.config.InstanceLifecycle.effectiveWarmUpFactor()
	for _, inst := range c.instances {
		if factor > 1.0 {
			for _, reqID := range inst.WarmUpRequestIDs() {
				if ttft, ok := merged.RequestTTFTs[reqID]; ok {
					// Guard against propagating corrupt TTFT values (R3, R11)
					if !math.IsNaN(ttft) && !math.IsInf(ttft, 0) {
						newTTFT := ttft * factor
						// I34: Guard against Inf from large factor * large TTFT
						if math.IsInf(newTTFT, 0) {
							continue
						}
						// I1: Keep TTFTSum consistent with per-request TTFT adjustments.
						// Convert the TTFT delta (microseconds) to int64 ticks for TTFTSum.
						merged.TTFTSum += int64(newTTFT - ttft)
						merged.RequestTTFTs[reqID] = newTTFT
					}
				}
			}
		}
		inst.clearWarmUpRequestIDs()
	}

	return merged
}

// projectPDMetrics replaces sub-request entries in per-request metric maps
// with parent-level entries. For each ParentRequest:
//   - Completed parents (CompletionTime > 0, DecodeInstanceID != ""):
//     sub-request entries are replaced with parent-keyed entries using
//     true user-facing values (e.g., E2E = CompletionTime - ArrivalTime).
//   - Incomplete/dropped parents: sub-request entries are removed
//     (these requests did not complete successfully).
//
// This is a no-op when disaggregation is not active (parentRequests is empty).
func (c *ClusterSimulator) projectPDMetrics() {
	if len(c.parentRequests) == 0 {
		return
	}
	m := c.aggregatedMetrics

	for _, parent := range c.parentRequests {
		pfx := parent.PrefillSubReqID // "req_N_prefill"
		dec := parent.DecodeSubReqID  // "req_N_decode"
		pid := parent.ID              // "req_N"
		completed := parent.CompletionTime > 0 && parent.DecodeInstanceID != ""

		// E2E = parent.CompletionTime - parent.ArrivalTime
		// (arrival → prefill → transfer → decode → completion).
		delete(m.RequestE2Es, pfx)
		delete(m.RequestE2Es, dec)
		if completed {
			e2e := parent.CompletionTime - parent.ArrivalTime
			if e2e < 0 {
				// INV-3/INV-5 violation: completion before arrival. Should never occur
				// after the clusterTime fix in EnqueueDecodeSubRequest.
				logrus.Errorf("[cluster] projectPDMetrics: negative E2E for %s (completionTime=%d arrivalTime=%d); skipping",
					pid, parent.CompletionTime, parent.ArrivalTime)
			} else {
				m.RequestE2Es[pid] = float64(e2e)
			}
		}

		// TTFT: rekey from prefill sub-request ID to parent ID (value unchanged).
		// Only prefill sub-requests record TTFT; decode sub-requests never trigger it.
		// Gate on completed: dropped-request TTFTs must not enter the distribution.
		// Source keys are always deleted, regardless of completion status.
		if completed {
			if ttft, ok := m.RequestTTFTs[pfx]; ok {
				m.RequestTTFTs[pid] = ttft
			} else {
				logrus.Warnf("[cluster] projectPDMetrics: completed parent %s has no prefill TTFT (key %s)", pid, pfx)
			}
		}
		delete(m.RequestTTFTs, pfx)
		delete(m.RequestTTFTs, dec)

		// Scheduling delay = prefill sub-request's delay
		// (the real user-facing delay, not the decode pipeline cumulative latency).
		prefillDelay, hasPrefillDelay := m.RequestSchedulingDelays[pfx]
		delete(m.RequestSchedulingDelays, pfx)
		delete(m.RequestSchedulingDelays, dec)
		if completed && hasPrefillDelay {
			m.RequestSchedulingDelays[pid] = prefillDelay
		}

		// Requests metadata keyed by parent ID, HandledBy set to decode instance.
		delete(m.Requests, pfx)
		delete(m.Requests, dec)
		if completed {
			if parent.OriginalRequest == nil {
				panic(fmt.Sprintf("projectPDMetrics: parent %s has nil OriginalRequest", pid))
			}
			rm := sim.NewRequestMetrics(parent.OriginalRequest, float64(parent.ArrivalTime)/1e6)
			rm.HandledBy = string(parent.DecodeInstanceID)
			m.Requests[pid] = rm
		}

		// ITL from decode sub-request (prefill ITL is 0 noise).
		decodeITL, hasDecodeITL := m.RequestITLs[dec]
		delete(m.RequestITLs, pfx)
		delete(m.RequestITLs, dec)
		if completed && hasDecodeITL {
			m.RequestITLs[pid] = decodeITL
		}

		// Completion time from parent lifecycle tracking.
		delete(m.RequestCompletionTimes, pfx)
		delete(m.RequestCompletionTimes, dec)
		if completed {
			m.RequestCompletionTimes[pid] = float64(parent.CompletionTime)
		}
	}
}
