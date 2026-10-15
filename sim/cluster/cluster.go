package cluster

import (
	"container/heap"
	"fmt"
	"math"
	"reflect"
	"sort"

	"github.com/inference-sim/inference-sim/sim"
	"github.com/inference-sim/inference-sim/sim/trace"
	"github.com/sirupsen/logrus"
)

// ClusterSimulator orchestrates N InstanceSimulator replicas behind a shared clock.
// Events from all instances are processed in global timestamp order;
// ties are broken by lowest instance index for determinism.
type ClusterSimulator struct {
	// abortErr is the error the gateway aborted the run with (GatewayHost.Abort).
	abortErr          error
	config            DeploymentConfig
	instances         []*InstanceSimulator
	rng               *sim.PartitionedRNG
	clock             int64
	hasRun            bool
	aggregatedMetrics *sim.Metrics

	// maxNodesSpanned is the largest number of physical nodes any single instance has
	// occupied (#1530). 0/1 = every instance is single-node. Exported via
	// MaxNodesSpanned() so `blis run --trace-output` can record it in the trace header;
	// replay refuses a trace whose fleet spanned nodes, because it cannot reconstruct
	// one (node pools are run-only) and would otherwise silently replay the workload at
	// single-node speed.
	maxNodesSpanned int

	// Request pipeline: arrivals go to the gateway, which dispatches or rejects them.
	clusterEvents     ClusterEventQueue
	seqCounter        int64
	gateway           Gateway
	daemonQueued      int            // daemon gateway callbacks in clusterEvents (they do not keep the run alive)
	rejectedRequests  int            // INV-1: requests an admission check refused
	routingRejections int            // INV-1: requests no endpoint could serve
	gatewayRejected   int            // INV-1: requests flow control rejected at capacity
	shedByTier        map[string]int // per-SLOClass shedding: admission rejections + flow-control rejections and expiries + in-flight evictions
	// injectedByClass: per-SLOClass arrival counter. Incremented in ClusterArrivalEvent.Execute
	// before any drop/route/admission decision. Goodput denominator (issue #1409, BC-5).
	injectedByClass  map[string]int64
	trace            *trace.SimulationTrace // nil when trace-level is "none" (BC-1: zero overhead)
	requestSource    RequestSource          // Source of requests to inject as arrival events. Drained once by Run().
	inFlightRequests map[string]int         // instance ID → dispatched-but-not-completed count (#463)
	gatewayEvicted   int                    // INV-1: requests the gateway evicted in flight
	gatewayExpired   int                    // INV-1: requests that expired in flow control
	gatewayTimedOut  int                    // INV-1: requests whose client deadline passed in the gateway (counted in TimedOutRequests)
	poolMembership   map[string]PoolRole    // instance ID → pool role (nil when disaggregation disabled)

	// PD disaggregation state (PR2)
	parentRequests            map[string]*ParentRequest // parent request ID → tracking record
	pendingPrefillCompletions map[string]string         // prefill sub-req ID → parent ID
	pendingDecodeCompletions  map[string]string         // decode sub-req ID → parent ID
	// transfersInitiated counts every request that reaches
	// KVTransferStartedEvent (issue #1343: both successful reservations
	// that enter the transfer pipeline AND drop-at-start cases where the
	// decode pod was unroutable or reservation failed — dropAtStart also
	// bumps this counter). This preserves the pre-#1343 "attempts"
	// semantics so INV-PD-3 (initiated == completed) still holds: every
	// drop-at-start also schedules a zero-duration
	// KVTransferCompletedEvent that increments transfersCompleted.
	transfersInitiated int
	// transfersCompleted counts every KVTransferCompletedEvent that fires —
	// successful promotions, late drops (decode pod became non-routable
	// mid-transfer), and degenerate completions scheduled by dropAtStart.
	transfersCompleted      int
	pdPrefillCompletedCount int // prefill sub-requests that completed (for INV-1 correction)
	pdDecodeCompletedCount  int // decode sub-requests that completed (for INV-1 in-flight tracking)
	pdDecodeTimedOutCount   int // decode sub-requests that timed out (for INV-1 in-flight tracking)
	droppedAtDecodeKV       int // requests dropped due to insufficient KV at decode

	encodeRoutingRejections int // INV-1 term: requests rejected at encode routing (empty encode pool)

	// Phase 1A: node/GPU placement manager. Nil when NodePools is empty (backward-compat).
	placement *PlacementManager

	// Phase 1C: model autoscaler pipeline. Nil when ModelAutoscalerIntervalUs == 0 (backward-compat, INV-6).
	autoscaler      *autoscalerPipeline
	pendingArrivals int // count of ClusterArrivalEvents not yet executed; used by scheduleNextTick to stop ticking when all work is done

	// sessionCallback is the raw onRequestDone parameter for session follow-up
	// generation in PD mode. Called from detectDecodeCompletions with the original
	// request (which carries SessionID). Separate from the per-instance closure to
	// avoid double-notifying tenantTracker (issue #884). Nil for non-session workloads.
	sessionCallback func(*sim.Request, int64) []*sim.Request

	progressHook               sim.ProgressHook
	simClockProgressIntervalUs int64
	nextSnapshotClockUs        int64

	// arrivalHook fires once per fresh arrival (initial workload and
	// follow-ups from closed-loop sessions). It does NOT fire for requests
	// re-injected by the REDIRECT drain policy — whether or not a prior
	// ClusterArrivalEvent fired, emitting here would duplicate or create
	// a spurious record. Nil unless SetArrivalHook was called.
	//
	// Contract:
	//   - Fires at most once per (logical) request.
	//   - Called from ClusterArrivalEvent.Execute on the cluster's single
	//     Run() goroutine (no concurrency). Firing at execute time — not
	//     push time — is what makes the hook's stream non-decreasing in
	//     arrival time (INV-15; the clock invariant it used to cite is INV-3,
	//     which is about the clock the loop advances, not this input stream).
	//   - Must be cheap (recording-only). Heavy work belongs in Run finalization.
	//   - Receives the *sim.Request pointer the cluster will inject. The hook
	//     must not mutate the request — it is shared with the cluster pipeline.
	//
	// Determinism (INV-6): the hook sees requests in the same monotonic
	// non-decreasing ArrivalTime order the cluster enqueues them.
	// fireArrivalHook() panics on a regression.
	arrivalHook         func(*sim.Request)
	lastArrivalHookTime int64 // monotonicity guard for arrivalHook (us)
}

// effectiveAnalyzerConfig applies WVA reference defaults to zero-valued fields.
// Zero values mean "not configured by caller" — fill with defaults so callers
// only need to set ModelAutoscalerIntervalUs to enable the autoscaler.
func effectiveAnalyzerConfig(cfg V2SaturationAnalyzerConfig) V2SaturationAnalyzerConfig {
	if cfg.KvCacheThreshold == 0 {
		cfg.KvCacheThreshold = 0.8
	}
	if cfg.ScaleUpThreshold == 0 {
		cfg.ScaleUpThreshold = 0.8
	}
	if cfg.ScaleDownBoundary == 0 {
		cfg.ScaleDownBoundary = 0.4
	}
	if cfg.AvgInputTokens == 0 {
		cfg.AvgInputTokens = 512
	}
	return cfg
}

// NewClusterSimulator creates a ClusterSimulator with N instances.
// Requests are pulled from requestSource (which yields in non-decreasing
// ArrivalTime order, exactly once each) at the start of Run().
//
// onRequestDone is an optional callback invoked when a request reaches a terminal state
// (completed, length-capped, timed out, or dropped). The callback returns follow-up
// requests which are routed through the cluster pipeline (not injected locally).
// Pass nil for non-session workloads.
//
// Panics if config.NumInstances < 1 or if requestSource is nil.
func NewClusterSimulator(config DeploymentConfig, requestSource RequestSource, onRequestDone func(*sim.Request, int64) []*sim.Request) *ClusterSimulator {
	if config.NumInstances < 1 {
		panic("ClusterSimulator: NumInstances must be >= 1")
	}
	if requestSource == nil {
		panic("ClusterSimulator: requestSource must not be nil (use NewSliceRequestSource(nil) for an empty workload)")
	}

	// Validate pool topology and overrides early (before instance construction).
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 || config.EncodeInstances > 0 {
		if err := ValidatePoolTopology(config.PrefillInstances, config.DecodeInstances, config.SharedInstances, config.EncodeInstances, config.NumInstances); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		if err := config.PrefillOverrides.Validate("prefill pool"); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
		if err := config.DecodeOverrides.Validate("decode pool"); err != nil {
			panic(fmt.Sprintf("ClusterSimulator: %v", err))
		}
	}

	// A KV handoff is priced by the latency backend (DeploymentConfig.PDTransferTime); the
	// simulator owns only when it happens and has no transfer formula of its own. Refuse a
	// PD-enabled deployment without a pricer here, so KVTransferStartedEvent never meets one
	// at runtime (R1). Covers pure-shared clusters too (issue #1276): a shared-role pod can
	// perform prefill and therefore source a KV transfer.
	if config.PrefillInstances > 0 || config.SharedInstances > 0 {
		if config.PDTransferTime == nil {
			panic("ClusterSimulator: PD disaggregation requires DeploymentConfig.PDTransferTime: " +
				"the latency backend prices the KV handoff, and the simulator has no formula of its own")
		}
	}
	// Concurrent handoffs are priced independently: overlapping transfers do not share fabric
	// bandwidth. Per-link contention is not modelled yet (INV-P2-2, #1922).

	// Build pre-construction pool membership so instance construction can resolve per-pool config.
	// When disaggregation is disabled (all pool counts are 0), prePoolMembership is nil
	// and all instances use the global config (backward-compatible).
	var prePoolMembership map[string]PoolRole
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 || config.EncodeInstances > 0 {
		prePoolMembership = BuildPoolMembershipFromIndices(config.NumInstances, config.PrefillInstances, config.DecodeInstances, config.SharedInstances, config.EncodeInstances)
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

	cs := &ClusterSimulator{
		config:           config,
		instances:        make([]*InstanceSimulator, 0, config.NumInstances),
		rng:              rng,
		requestSource:    requestSource,
		clusterEvents:    make(ClusterEventQueue, 0),
		trace:            simTrace,
		inFlightRequests: make(map[string]int, config.NumInstances),
		shedByTier:       make(map[string]int),
		injectedByClass:  make(map[string]int64),
	}

	// PD disaggregation: set pool membership (topology already validated above). The gateway
	// decides per request whether and where to disaggregate.
	if config.PrefillInstances > 0 || config.DecodeInstances > 0 || config.SharedInstances > 0 {
		cs.poolMembership = prePoolMembership
		cs.parentRequests = make(map[string]*ParentRequest)
		cs.pendingPrefillCompletions = make(map[string]string)
		cs.pendingDecodeCompletions = make(map[string]string)

		// Per-pool routing policies and the disaggregation decider are created after
		// the construction loop (both need cacheQueryFn from instances).

		logrus.Infof("[cluster] PD disaggregation enabled: %d prefill, %d decode, %d shared (prefill-decode) instances",
			config.PrefillInstances, config.DecodeInstances, config.SharedInstances)

		// Issue #1276 D-2: shared-role pods resolve config via decode-wins precedence.
		// Warn when PrefillOverrides would have been applicable but is silently discarded.
		if config.SharedInstances > 0 && !reflect.DeepEqual(config.PrefillOverrides, config.DecodeOverrides) {
			logrus.Infof("[cluster] shared-role pods use DecodeOverrides (PrefillOverrides ignored per PR1276 decode-wins rule)")
		}
	}

	// Phase 1A: initialize PlacementManager when node pools are configured.
	// Must happen BEFORE the unified construction loop so cs.placement is set.
	if len(config.NodePools) > 0 {
		// #1529: placement uses the GLOBAL config.TP to decide single-node vs
		// whole-node spanning and to bill nodes-spanned × cost. A per-role TP
		// override (PoolOverrides.TP, from a P/D scenario's per-pool parallelism) would make the simulated TP diverge
		// from the placed/billed TP — wrong span decision, wrong cost, wrong KV
		// capacity. Per-role placement does not exist yet, so reject the combination
		// loudly rather than produce silently-wrong numbers (mirrors the INV-13
		// Fatalf-on-unsupported-combination policy). Lifting this guard (a per-role
		// placement path) is tracked by #1543.
		for _, ov := range []struct {
			name string
			po   PoolOverrides
		}{{"prefill", config.PrefillOverrides}, {"decode", config.DecodeOverrides}} {
			if ov.po.TP != nil && *ov.po.TP != config.TP {
				panic(fmt.Sprintf("ClusterSimulator: per-role tensor parallelism (%s pool TP=%d) is not supported with "+
					"node_pools (global TP=%d): placement, node-span, and cost use the global TP, so a differing per-role "+
					"TP would be simulated at one degree but placed/billed at another. Use a uniform TP across pools, or drop node_pools.",
					ov.name, *ov.po.TP, config.TP))
			}
		}
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
	// Deferral accounting (#1529): count instances that fail startup placement so we
	// warn once after the loop instead of once per instance. NodeReadyEvent (the only
	// path that retries pending instances) has no production caller in blis run, so a
	// startup deferral means the instance is effectively dropped — surface it (R1).
	var unplacedCount int
	var unplacedFirstErr error
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
				// Collect deferrals here and warn ONCE after the loop (a cluster with
				// InitialNodes=0 defers every instance — one summary, not N copies).
				// Keep the FIRST error: instance_0's error is the most actionable (e.g.
				// the structurally-unsatisfiable "regardless of capacity" message would
				// otherwise be masked by a later instance's generic capacity error).
				unplacedCount++
				if unplacedFirstErr == nil {
					unplacedFirstErr = err
				}
				cs.placement.AddPending(id, config.Model, "", tpDegree, simCfg)
				continue
			}
			// Placement succeeded: use pool's GPU type (SC-004: pool-authoritative, not CLI flag).
			simCfg.GPU = matchedGPUType
			// Phase 1C: look up CostPerHour for this matched GPU type (issue #692).
			var poolCostPerHour float64
			for i := range config.NodePools {
				if config.NodePools[i].GPUType == matchedGPUType {
					poolCostPerHour = config.NodePools[i].CostPerHour
					break
				}
			}
			// Record how many nodes this instance spans, for the trace header (#1530).
			cs.recordNodeSpan(gpuIDs)
			inst := NewInstanceSimulator(id, simCfg)
			inst.Model = config.Model
			inst.nodeID = nodeID
			inst.allocatedGPUIDs = gpuIDs
			inst.TPDegree = tpDegree
			// #1529: cost = distinct-nodes-spanned × pool cost_per_hour. Single-node
			// instances are 1× (unchanged); a multi-node TP instance is billed per node.
			inst.CostPerHour = cs.placement.InstanceCostPerHour(gpuIDs, poolCostPerHour)
			inst.warmUpRemaining = config.InstanceLifecycle.WarmUpRequestCount
			if config.InstanceLifecycle.WarmStartInitialInstances {
				// Pre-deployed cluster: startup instances skip loading delay and start Active.
				// Autoscaler-added instances (via direct_actuator.go) still use LoadingDelay.
				if inst.warmUpRemaining > 0 {
					inst.TransitionTo(sim.InstanceStateWarmingUp)
				} else {
					inst.TransitionTo(sim.InstanceStateActive)
				}
			} else {
				inst.TransitionTo(sim.InstanceStateLoading)
				cs.scheduleInstanceLoadedEvent(inst)
			}
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
				inst.TransitionTo(sim.InstanceStateWarmingUp)
			} else {
				inst.TransitionTo(sim.InstanceStateActive)
			}
			cs.instances = append(cs.instances, inst)
			instanceMap[id] = inst
			cs.inFlightRequests[string(id)] = 0
		}
	}

	// One summary warning for all startup placement deferrals (#1529). The first
	// error text distinguishes a structurally-unsatisfiable request ("unsatisfiable
	// regardless of capacity" — a config error worth fixing) from a transient
	// shortfall; either way, NodeReady is not wired in blis run, so these instances
	// will not be placed later.
	if unplacedCount > 0 {
		logrus.Warnf("[cluster] %d of %d instance(s) not placed at startup (first error: %v) — deferred to "+
			"pending, but NodeReady is not wired in blis run, so they will not be placed later",
			unplacedCount, config.NumInstances, unplacedFirstErr)
	}

	// Phase 1C: initialize autoscaler pipeline when ModelAutoscalerIntervalUs > 0 (issue #692).
	// Zero interval disables the autoscaler entirely (INV-6 backward-compat).
	// The default WVA pipeline (DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine →
	// DirectActuator) is wired here. Tests that need custom components replace cs.autoscaler
	// after construction via same-package access.
	// R3: validate autoscaler float64 fields — NaN/Inf/negative values are configuration errors.
	if math.IsNaN(config.ModelAutoscalerIntervalUs) || math.IsInf(config.ModelAutoscalerIntervalUs, 0) {
		panic("ModelAutoscalerIntervalUs must not be NaN or Inf")
	}
	if config.ModelAutoscalerIntervalUs < 0 {
		panic("ModelAutoscalerIntervalUs must be ≥0 (0 = disabled)")
	}
	if math.IsNaN(config.ScaleUpStabilizationWindowUs) || math.IsInf(config.ScaleUpStabilizationWindowUs, 0) || config.ScaleUpStabilizationWindowUs < 0 {
		panic("ScaleUpStabilizationWindowUs must be a finite non-negative number")
	}
	if math.IsNaN(config.ScaleDownStabilizationWindowUs) || math.IsInf(config.ScaleDownStabilizationWindowUs, 0) || config.ScaleDownStabilizationWindowUs < 0 {
		panic("ScaleDownStabilizationWindowUs must be a finite non-negative number")
	}
	if math.IsNaN(config.HPAScrapeDelay.Mean) || math.IsInf(config.HPAScrapeDelay.Mean, 0) || config.HPAScrapeDelay.Mean < 0 {
		panic("HPAScrapeDelay.Mean must be a finite non-negative number")
	}
	if math.IsNaN(config.HPAScrapeDelay.Stddev) || math.IsInf(config.HPAScrapeDelay.Stddev, 0) || config.HPAScrapeDelay.Stddev < 0 {
		panic("HPAScrapeDelay.Stddev must be a finite non-negative number")
	}
	if config.ModelAutoscalerIntervalUs > 0 {
		// Wire the default WVA pipeline: DefaultCollector → V2SaturationAnalyzer → UnlimitedEngine → DirectActuator.
		// effectiveAnalyzerConfig fills zero fields with WVA reference defaults so callers only need interval_us.
		// Tests that need custom components (stubs, nopActuator) replace cs.autoscaler after construction (same-package access).
		analyzerCfg := effectiveAnalyzerConfig(config.AutoscalerAnalyzerConfig)
		cs.autoscaler = newAutoscalerPipeline(
			&DefaultCollector{},
			NewV2SaturationAnalyzer(analyzerCfg),
			&UnlimitedEngine{},
			NewDirectActuator(cs),
			rng.ForSubsystem(subsystemAutoscaler),
		)
	}

	// Store raw callback for PD session follow-up (issue #884).
	cs.sessionCallback = onRequestDone

	for _, inst := range cs.instances {
		cs.wireInstance(inst)
	}
	return cs
}

// wireInstance connects an instance's request lifecycle to the gateway and to session
// follow-ups. Follow-ups route through the cluster pipeline, never injected locally (BC-9).
func (cs *ClusterSimulator) wireInstance(inst *InstanceSimulator) {
	inst.sim.OnFirstToken = func(req *sim.Request, at int64) {
		if parent := cs.parentOf(req); parent != nil {
			// A disaggregated response streams from the decode instance.
			if req.IsDecodeSubRequest && parent.OriginalRequest != nil {
				cs.gateway.OnFirstToken(parent.OriginalRequest, at)
			}
			return
		}
		cs.gateway.OnFirstToken(req, at)
	}
	inst.sim.OnRequestDone = func(req *sim.Request, tick int64) []*sim.Request {
		if parent := cs.parentOf(req); parent != nil {
			// Decode completions report in detectDecodeCompletions, at parent granularity; a
			// prefill that does not complete ends its parent here.
			if !req.IsDecodeSubRequest && req.State != sim.StateCompleted {
				cs.parentDone(parent, tick, doneOutcome(req))
			}
		} else {
			cs.gateway.OnDone(req, doneTime(inst, req, tick), doneOutcome(req))
		}
		if cs.sessionCallback == nil {
			return nil
		}
		for _, next := range cs.sessionCallback(req, tick) {
			cs.pushArrival(next, next.ArrivalTime)
		}
		return nil // don't inject locally — route through cluster pipeline
	}
	inst.onTransition = func(inst *InstanceSimulator) {
		if cs.gateway != nil {
			cs.gateway.OnInstanceChange(inst)
		}
	}
}

// doneTime is when the client observes the end of a request that reached a terminal state at
// tick: a completed response ends when its last token is delivered.
func doneTime(inst *InstanceSimulator, req *sim.Request, tick int64) int64 {
	if req.State == sim.StateCompleted {
		if t, ok := inst.Metrics().RequestCompletionTimes[req.ID]; ok {
			return max(tick, int64(t))
		}
	}
	return tick
}

func doneOutcome(req *sim.Request) DoneOutcome {
	switch req.State {
	case sim.StateCompleted:
		return DoneCompleted
	case sim.StateTimedOut:
		return DoneTimedOut
	default:
		return DoneDropped
	}
}

// pushArrival enqueues a ClusterArrivalEvent and increments pendingArrivals
// as a paired operation. All ClusterArrivalEvent pushes MUST go through this
// method — it is the single enforcement point for the pendingArrivals
// co-invariant used by scheduleNextTick (autoscaler.go). Direct heap.Push of
// ClusterArrivalEvent outside this method is prohibited.
// The paired decrement occurs in ClusterArrivalEvent.Execute (cluster_event.go).
//
// NOTE: When ClusterSubsystem hooks are introduced (see discussion #1033 PR D),
// OnRequestDone hooks that generate follow-ups should return them to the
// coordinator rather than calling pushArrival directly, preserving this method
// as the single enforcement point. See also issue #1041.
func (cs *ClusterSimulator) pushArrival(req *sim.Request, timeUs int64) {
	heap.Push(&cs.clusterEvents, clusterEventEntry{
		event: &ClusterArrivalEvent{time: timeUs, request: req},
		seqID: cs.nextSeqID(),
	})
	cs.pendingArrivals++
}

// fireArrivalHook is called from ClusterArrivalEvent.Execute on the single
// path that all fresh arrivals (initial workload and closed-loop follow-ups)
// traverse at their effective arrival time. Firing here — rather than at
// pushArrival — gives the hook a stream that is non-decreasing in arrival
// time (INV-15), so trace records emerge already in arrival order without a
// downstream sort.
// REDIRECT re-injections are skipped: req.Redirected=true marks requests
// the drain policy is rerouting internally. Whether or not a prior
// ClusterArrivalEvent fired for this request, emitting a trace record
// here would either duplicate an existing record or create a spurious
// one for internally-rerouted work.
//
// Horizon semantics: ClusterArrivalEvents whose timestamp exceeds
// config.Horizon never execute (cluster.go event loop short-circuits past
// Horizon), so the hook does NOT see beyond-horizon follow-ups. This is
// intentional — a request that never arrived to the cluster has no place
// in the exported trace, and excluding it strengthens INV-13 (replay reads
// the same trace the run produced).
func (cs *ClusterSimulator) fireArrivalHook(req *sim.Request, timeUs int64) {
	if cs.arrivalHook == nil || req.Redirected {
		return
	}
	if timeUs < cs.lastArrivalHookTime {
		// INV-15, not INV-3 or INV-6 (the IDs this guard cited before #1772):
		// this is a precondition on the arrival stream the cluster is GIVEN, so
		// the repair is in the arrival source, not in the clock or in output
		// ordering. See docs/contributing/standards/invariants.md#inv-15.
		panic(fmt.Sprintf("ClusterSimulator: arrival hook received out-of-order request %q (timeUs=%d < last=%d) — INV-15 violation: arrivals must be non-decreasing in ArrivalTime",
			req.ID, timeUs, cs.lastArrivalHookTime))
	}
	cs.lastArrivalHookTime = timeUs
	cs.arrivalHook(req)
}

// SetArrivalHook installs a callback fired once per fresh request arrival
// (initial workload + closed-loop session follow-ups). The hook does NOT
// fire for requests re-injected by the REDIRECT drain policy.
//
// Must be called before Run(); panics otherwise. Pass nil to clear; the
// monotonicity guard (lastArrivalHookTime) is reset to zero on clear so
// a subsequently installed hook starts from a clean baseline.
//
// Used by `blis run` to capture TraceV2 records at the arrival boundary
// (issue #1440), replacing the eager post-run RequestsToTraceRecords pass.
func (cs *ClusterSimulator) SetArrivalHook(hook func(*sim.Request)) {
	if cs.hasRun {
		panic("ClusterSimulator: SetArrivalHook must be called before Run()")
	}
	cs.arrivalHook = hook
	if hook == nil {
		// Reset the monotonicity floor so a future hook installation does
		// not inherit the previous hook's timestamp watermark.
		cs.lastArrivalHookTime = 0
	}
}

// MaxNodesSpanned returns the largest number of physical nodes any single model
// instance in this cluster occupies (#1530). Returns 0 when no node pools are
// configured (no placement), and 1 when every instance fits on one node.
//
// `blis run --trace-output` records it in the trace header so `blis replay` can refuse
// a trace whose fleet spanned nodes: cross-node collective traffic is charged to step
// time, but replay cannot reconstruct a multi-node fleet (node pools are run-only), so
// replaying such a trace would silently be faster than the run it came from.
func (cs *ClusterSimulator) MaxNodesSpanned() int { return cs.maxNodesSpanned }

// Run executes the cluster simulation using online routing pipeline: drains the
// configured RequestSource into ClusterArrivalEvents, runs a shared-clock event
// loop processing cluster events before instance events, then finalizes.
// Panics if called more than once.
func (c *ClusterSimulator) Run() error {
	if c.hasRun {
		panic("ClusterSimulator.Run() called more than once")
	}
	c.hasRun = true

	// 1. Schedule ClusterArrivalEvents (NC-1: no pre-dispatch before event loop)
	heap.Init(&c.clusterEvents)
	if c.gateway == nil && newDefaultGateway != nil {
		c.gateway = newDefaultGateway()
	}
	if c.gateway == nil {
		panic("ClusterSimulator.Run: no gateway installed (SetGateway)")
	}
	if err := c.gateway.Start(c); err != nil {
		return fmt.Errorf("gateway: %w", err)
	}
	defer c.gateway.Stop()

	// Phase 1C: schedule the first ScalingTickEvent when the autoscaler is enabled (T015).
	// The autoscaler is enabled when ModelAutoscalerIntervalUs > 0 AND cs.autoscaler is non-nil.
	// Zero-interval guard: no tick is ever scheduled when interval is 0 (INV-6).
	if c.autoscaler != nil && c.config.ModelAutoscalerIntervalUs > 0 {
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &ScalingTickEvent{At: c.clock},
			seqID: c.nextSeqID(),
		})
	}

	// 2. Drain the request source to schedule arrival events. The source is
	// required to yield in non-decreasing ArrivalTime order (RequestSource
	// contract — caller obligation, not verified here); we count emissions to
	// preserve today's "no requests" warning.
	arrivalCount := 0
	for {
		req, ok := c.requestSource.Next()
		if !ok {
			break
		}
		if req == nil {
			panic("ClusterSimulator: RequestSource.Next() returned (nil, true) — implementation contract violation (Next must never return ok=true with a nil request)")
		}
		c.pushArrival(req, req.ArrivalTime)
		arrivalCount++
	}
	if arrivalCount == 0 {
		logrus.Warn("[cluster] no requests provided — simulation will produce zero results")
	}

	// 3. Shared-clock event loop (BC-4: cluster events before instance events)
	var lastProgress int64
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

		// Done when no instance work and no request work remain. Daemon events (router
		// background loops) never keep the run alive on their own.
		if instanceIdx == -1 && len(c.clusterEvents) == c.daemonQueued {
			if c.gateway.Pending() == 0 {
				break
			}
			// The gateway still holds requests and only its own timers can release them. A run that
			// cannot make progress is an error, never a result (R1).
			if len(c.clusterEvents) == 0 {
				return fmt.Errorf("gateway holds %d request(s) and no event can release them", c.gateway.Pending())
			}
			if c.clock-lastProgress > gatewayStallLimitUs {
				return fmt.Errorf("gateway holds %d request(s) and nothing but its own timers has happened for %d µs of simulated time",
					c.gateway.Pending(), gatewayStallLimitUs)
			}
		} else if clusterTime == math.MaxInt64 && instanceIdx == -1 {
			break
		}

		// BC-4: Cluster events at time T processed before instance events at time T
		// Using <= ensures cluster events drain first when timestamps are equal
		if instanceIdx == -1 && clusterTime != math.MaxInt64 {
			c.maybeFastForward()
			clusterTime = c.clusterEvents[0].event.Timestamp()
		}
		if c.abortErr != nil {
			return fmt.Errorf("gateway: %w", c.abortErr)
		}
		if clusterTime <= instanceTime {
			entry := heap.Pop(&c.clusterEvents).(clusterEventEntry)
			c.clock = entry.event.Timestamp()
			if c.clock > c.config.Horizon {
				break
			}
			if cb, ok := entry.event.(*CallbackEvent); !ok || !cb.daemon {
				lastProgress = c.clock
			}
			entry.event.Execute(c)
		} else {
			prevClusterClock := c.clock
			c.clock = instanceTime
			if c.clock > c.config.Horizon {
				break
			}
			inst := c.instances[instanceIdx]
			instID := string(inst.ID())
			lastProgress = c.clock

			// Snapshot counters BEFORE processing the event
			completedBefore := inst.Metrics().CompletedRequests
			droppedBefore := inst.Metrics().DroppedUnservable
			timedOutBefore := inst.Metrics().TimedOutRequests

			ev := inst.ProcessNextEvent()

			// If ProcessNextEvent() skipped a cancelled TimeoutEvent (lazy
			// cancellation — inst.Clock was not advanced), restore c.clock.
			// A no-op orphaned timeout must not advance the cluster clock.
			if te, ok := ev.(*sim.TimeoutEvent); ok && te.Request.State == sim.StateCompleted {
				c.clock = prevClusterClock
			}

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
					// deterministic metrics.
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

			}

			// T042: drain completion accounting (Phase 1A).
			// When a Draining instance has no more queued or running requests,
			// transition it to Terminated and release its GPU allocations.
			if inst.State == sim.InstanceStateDraining && inst.QueueDepth() == 0 && inst.BatchSize() == 0 {
				inst.TransitionTo(sim.InstanceStateTerminated)
				c.releaseInstanceGPUs(inst)
				// I1: a non-zero inFlightRequests at termination time indicates a bookkeeping bug —
				// a missing completion event or an early-termination race.
				if c.inFlightRequests[instID] != 0 {
					logrus.Warnf("[cluster] instance %s terminated with inFlightRequests=%d — bookkeeping bug",
						instID, c.inFlightRequests[instID])
				}
			}

			// PD disaggregation: detect prefill/decode sub-request completions.
			// Set-membership (.Has) so a shared-role pod (PoolRolePrefillDecode)
			// fires both detectors — issue #1276 BC-5.
			if c.poolsConfigured() {
				role := c.poolMembership[instID]
				if role.Has(PoolRolePrefill) {
					c.detectPrefillCompletions(inst)
				}
				if role.Has(PoolRoleDecode) {
					c.detectDecodeCompletions(inst)
				}
			}
		}

		c.maybeDeliverProgressSnapshot(false)
	}
	if c.abortErr != nil {
		return fmt.Errorf("gateway: %w", c.abortErr)
	}

	c.maybeDeliverProgressSnapshot(true)

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
	c.aggregatedMetrics.TimedOutRequests += c.gatewayTimedOut // client deadlines that passed while the gateway held the request

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

	if n := c.GatewayQueueDepth(); n > 0 {
		logrus.Warnf("[cluster] %d requests remain in the gateway at simulation end", n)
	}

	// Post-simulation diagnostic warnings (BC-2, BC-3)
	if c.aggregatedMetrics.CompletedRequests == 0 {
		if c.rejectedRequests > 0 {
			logrus.Warnf("[cluster] all %d requests rejected by the router's admission checks — no requests completed",
				c.rejectedRequests)
		} else if c.aggregatedMetrics.TimedOutRequests > 0 {
			logrus.Warnf("[cluster] no requests completed — %d of %d requests timed out (client timeout exceeded, likely KV pressure)",
				c.aggregatedMetrics.TimedOutRequests,
				c.aggregatedMetrics.TimedOutRequests+c.aggregatedMetrics.DroppedUnservable)
		} else {
			logrus.Warnf("[cluster] no requests completed — horizon may be too short or workload too small")
		}
	}

	return nil
}

// maybeFastForward lets an idle gateway advance its background work to the next request work
// when nothing but gateway background events comes before it.
func (c *ClusterSimulator) maybeFastForward() {
	ff, ok := c.gateway.(IdleFastForwarder)
	if !ok {
		return
	}
	if cb, ok := c.clusterEvents[0].event.(*CallbackEvent); !ok || !cb.daemon {
		return
	}
	next := int64(math.MaxInt64)
	for _, e := range c.clusterEvents {
		if cb, ok := e.event.(*CallbackEvent); ok && cb.daemon {
			continue
		}
		next = min(next, e.event.Timestamp())
	}
	if next != math.MaxInt64 && next > c.clock {
		ff.FastForwardIdle(next)
	}
}

// gatewayStallLimitUs bounds how long a run continues on router timers alone while the gateway
// holds requests and nothing else can happen: one simulated hour.
const gatewayStallLimitUs = int64(3600 * 1e6)

// nextSeqID returns the next monotonically increasing sequence ID for event ordering.
func (c *ClusterSimulator) nextSeqID() int64 {
	id := c.seqCounter
	c.seqCounter++
	return id
}

// SetProgressHook registers an optional hook that receives periodic state
// snapshots during cluster simulation execution. Must be called before Run().
// When hook is nil (default), there is zero behavioral or performance impact.
// simClockIntervalUs controls the minimum simulation-clock interval (microseconds)
// between periodic snapshots. If simClockIntervalUs <= 0, only the final snapshot
// is delivered.
func (c *ClusterSimulator) SetProgressHook(hook sim.ProgressHook, simClockIntervalUs int64) {
	c.progressHook = hook
	if simClockIntervalUs > 0 {
		c.simClockProgressIntervalUs = simClockIntervalUs
		c.nextSnapshotClockUs = simClockIntervalUs
	}
}

func (c *ClusterSimulator) maybeDeliverProgressSnapshot(isFinal bool) {
	if c.progressHook == nil {
		return
	}
	if !isFinal && (c.simClockProgressIntervalUs <= 0 || c.clock < c.nextSnapshotClockUs) {
		return
	}

	activeCount := 0
	instanceSnaps := make([]sim.InstanceSnapshot, 0, len(c.instances))
	for _, inst := range c.instances {
		if inst.State == sim.InstanceStateTerminated {
			continue
		}
		if inst.State == sim.InstanceStateActive || inst.State == sim.InstanceStateWarmingUp {
			activeCount++
		}
		instanceSnaps = append(instanceSnaps, sim.InstanceSnapshot{
			ID:                string(inst.ID()),
			QueueDepth:        inst.QueueDepth(),
			BatchSize:         inst.BatchSize(),
			KVUtilization:     inst.KVUtilization(),
			KVFreeBlocks:      inst.FreeKVBlocks(),
			KVTotalBlocks:     inst.TotalKVBlocks(),
			CacheHitRate:      inst.CacheHitRate(),
			PreemptionCount:   inst.PreemptionCount(),
			CompletedRequests: inst.Metrics().CompletedRequests,
			InFlightRequests:  c.inFlightRequests[string(inst.ID())],
			TimedOutRequests:  inst.Metrics().TimedOutRequests,
			State:             inst.State,
			Model:             inst.Model,
		})
	}

	clock := c.clock
	if isFinal {
		clock = min(c.clock, c.config.Horizon)
	}

	snap := sim.ProgressSnapshot{
		Clock:             clock,
		TotalCompleted:    c.completedRequestsTotal(),
		TotalTimedOut:     c.timedOutRequestsTotal(),
		TotalDropped:      c.droppedRequestsTotal(),
		TotalInputTokens:  c.inputTokensTotal(),
		TotalOutputTokens: c.outputTokensTotal(),
		TotalPreemptions:  c.preemptionsTotal(),
		InstanceSnapshots: instanceSnaps,
		RejectedRequests:  c.rejectedRequests,
		RoutingRejections: c.routingRejections,
		GatewayQueueDepth: c.GatewayQueueDepth(),
		GatewayEvicted:    c.gatewayEvicted,
		GatewayExpired:    c.gatewayExpired,
		ActiveInstances:   activeCount,
		TotalInstances:    len(c.instances),
		IsFinal:           isFinal,
	}
	if len(c.shedByTier) > 0 {
		snap.ShedByTier = make(map[string]int, len(c.shedByTier))
		for k, v := range c.shedByTier {
			snap.ShedByTier[k] = v
		}
	}
	c.progressHook.OnProgress(snap)
	if !isFinal {
		c.nextSnapshotClockUs += c.simClockProgressIntervalUs
	}
}

func (c *ClusterSimulator) completedRequestsTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().CompletedRequests
	}
	return total
}

func (c *ClusterSimulator) timedOutRequestsTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().TimedOutRequests
	}
	return total
}

func (c *ClusterSimulator) droppedRequestsTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().DroppedUnservable
	}
	return total
}

func (c *ClusterSimulator) inputTokensTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().TotalInputTokens
	}
	return total
}

func (c *ClusterSimulator) outputTokensTotal() int {
	total := 0
	for _, inst := range c.instances {
		total += inst.Metrics().TotalOutputTokens
	}
	return total
}

func (c *ClusterSimulator) preemptionsTotal() int64 {
	var total int64
	for _, inst := range c.instances {
		total += inst.Metrics().PreemptionCount
	}
	return total
}

// addLiveInstance constructs, registers, and activates an InstanceSimulator for a
// placement that succeeded while the cluster is already running.
// Called from NodeReadyEvent.Execute (deferred placement) and DirectActuator.scaleUp
// (autoscaler direct placement). NOT used by the NewClusterSimulator startup path,
// which bulk-initialises the snapshot provider with a full instance map.
//
// simCfg must already have GPU set (pool-authoritative, SC-004).
// Returns true on success. On false, GPU allocations have been released — callers
// must not touch the instance and should skip/continue.
//
// Maintenance note: if you add a new field to instance initialisation here, also
// check NewClusterSimulator's construction loop (which does NOT call this method).
func (cs *ClusterSimulator) addLiveInstance(
	id InstanceID,
	model string,
	simCfg sim.SimConfig,
	nodeID string,
	gpuIDs []string,
	tpDegree int,
	costPerHour float64,
) bool {
	inst := NewInstanceSimulator(id, simCfg)
	inst.Model = model
	inst.nodeID = nodeID
	inst.allocatedGPUIDs = gpuIDs
	inst.TPDegree = tpDegree
	inst.CostPerHour = costPerHour
	inst.warmUpRemaining = cs.config.InstanceLifecycle.WarmUpRequestCount
	cs.wireInstance(inst)
	inst.TransitionTo(sim.InstanceStateLoading)

	cs.scheduleInstanceLoadedEvent(inst)
	cs.instances = append(cs.instances, inst)
	cs.inFlightRequests[string(id)] = 0
	if cs.gateway != nil {
		cs.gateway.OnInstanceChange(inst)
	}

	return true
}

// poolsConfigured returns true if PD disaggregation pool topology is active.
func (c *ClusterSimulator) poolsConfigured() bool {
	return c.poolMembership != nil
}

// PoolMembership returns a copy of the pool role membership map (R8: no exported mutable maps).
// Returns nil when disaggregation is disabled.
//
// The copy is also how INV-PD-5 (pool membership fixed at construction) holds against
// callers outside this package: none of them can reach c.poolMembership to reassign a
// role mid-run. In-package callers do receive the raw map (see buildPoolFilteredSnapshots)
// and read it only.
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
		// The prefill completes at the END of the step that computed its KV, not at the
		// cluster clock, which reads that step's START when this detector runs right after
		// the step event (#1903): the handoff cannot begin before the KV it moves exists.
		parent.PrefillCompleteTime = parent.PrefillSubReq.DepartureTime
		delete(c.pendingPrefillCompletions, subReqID)
		c.pdPrefillCompletedCount++

		// Schedule KV transfer
		heap.Push(&c.clusterEvents, clusterEventEntry{
			event: &KVTransferStartedEvent{
				time:      parent.PrefillCompleteTime,
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
		// The end of the decode sub-request's final step -- not the cluster clock, which
		// reads that step's START when this detector runs (#1903) -- plus
		// PostDecodeFixedOverhead, so parent.CompletionTime is the client-visible completion,
		// matching non-PD E2E semantics (issue #846). No zero-output guard needed: decode
		// sub-requests always carry the full output token list (set in
		// KVTransferCompletedEvent.Execute).
		//
		// This line IS INV-PD-6b (parent completion includes post-decode overhead).
		parent.CompletionTime = parent.DecodeSubReq.DepartureTime + inst.PostDecodeFixedOverhead()
		delete(c.pendingDecodeCompletions, subReqID)
		c.pdDecodeCompletedCount++
		c.parentDone(parent, parent.CompletionTime, DoneCompleted)

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
				c.pushArrival(next, next.ArrivalTime)
			}
		}
	}

	// Phase 3: process timed-out decode sub-requests (INV-11 session completeness).
	// Non-PD equivalent: TimeoutEvent.Execute calls OnRequestDone with StateTimedOut →
	// SessionManager cancels the session. The PD path needs the same treatment.
	for _, subReqID := range timedOutIDs {
		parent := c.parentRequests[c.pendingDecodeCompletions[subReqID]]
		// No PostDecodeFixedOverhead here, unlike the INV-PD-6b line above: a
		// timed-out parent never finished decoding, so it never pays the
		// post-decode overhead.
		//
		// Note for anyone reconciling this against the doc: these timed-out parents
		// DO satisfy INV-PD-6b's parenthetical `DecodeInstanceID != ""`
		// (detectDecodeCompletions reaches them by decode-instance match), so read
		// on its own that gloss appears to cover this line. It does not: the
		// statement's governing qualifier is "successfully decoded" parent
		// requests, and a parent whose decode sub-request timed out is by
		// definition not successfully decoded. The parenthetical is loose; the
		// statement is correctly scoped and must NOT be narrowed on the strength
		// of this line.
		parent.CompletionTime = c.clock
		delete(c.pendingDecodeCompletions, subReqID)
		c.pdDecodeTimedOutCount++
		c.parentDone(parent, c.clock, DoneTimedOut)

		if c.sessionCallback != nil {
			origCopy := *parent.OriginalRequest
			origCopy.State = sim.StateTimedOut
			origCopy.ProgressIndex = parent.DecodeSubReq.ProgressIndex
			// SessionManager.OnComplete cancels the session for StateTimedOut (session.go:112).
			// No follow-ups expected, but handle defensively.
			nextReqs := c.sessionCallback(&origCopy, parent.CompletionTime)
			for _, next := range nextReqs {
				c.pushArrival(next, next.ArrivalTime)
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

// EncodeRoutingRejections returns the count of requests rejected at the encode
// routing stage because the encode pool has zero routable instances (GAP-4,
// issue #1264). Always zero when --encode-instances 0.
func (c *ClusterSimulator) EncodeRoutingRejections() int {
	return c.encodeRoutingRejections
}

// ShedByTier returns a copy of per-SLOClass rejection counts recorded during admission.
// Populated unconditionally for every admission rejection, regardless of policy.
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

// InjectedByClass returns a defensive copy of the per-SLOClass arrival counter.
// Incremented in ClusterArrivalEvent.Execute before any drop/route/admission
// decision; used as the goodput denominator (issue #1409, BC-5/BC-6).
// Panics if called before Run() completes (R8 mirror of ShedByTier).
func (c *ClusterSimulator) InjectedByClass() map[string]int64 {
	if !c.hasRun {
		panic("ClusterSimulator.InjectedByClass() called before Run()")
	}
	result := make(map[string]int64, len(c.injectedByClass))
	for k, v := range c.injectedByClass {
		result[k] = v
	}
	return result
}

// gpuInventory computes the current GPU inventory for Engine.Optimize().
// Phase 1C (T012): returns free GPU slots per VariantSpec.
//
// Free slots for a variant = total GPUs of that GPU type on Ready nodes
//   - GPUs held by Loading instances of that GPU type
//   - GPUs held by Active/WarmingUp instances of that GPU type
//   - GPUs held by Draining instances of that GPU type (hold GPUs until drain completes)
//
// Pending (Scheduling) instances are NOT subtracted.
// Terminated instances are NOT subtracted.
//
// Returns an empty inventory when cs.placement is nil (no NodePools configured, backward-compat).
func (c *ClusterSimulator) gpuInventory() GPUInventory {
	if c.placement == nil {
		return GPUInventory{byVariant: make(map[VariantSpec]int)}
	}

	// Step 1: count total GPUs on Ready nodes per GPUType.
	totalByGPUType := make(map[string]int)
	for _, node := range c.placement.nodesByID {
		if node.State == NodeStateReady {
			totalByGPUType[node.GPUType] += node.TotalGPUs
		}
	}

	// Step 2: subtract GPUs used by Loading, Active (incl. WarmingUp), and Draining instances.
	// Also populate seenVariants so every GPU type with Ready capacity appears in the result,
	// even when there are no active instances of that GPU type (enables scale-up from zero).
	clusterTPDegree := c.config.TP
	if clusterTPDegree < 1 {
		clusterTPDegree = 1
	}
	usedByGPUType := make(map[string]int)
	seenVariants := make(map[VariantSpec]struct{})
	// Seed from Ready node GPU types so zero-instance pools appear in inventory.
	for gpuType, total := range totalByGPUType {
		if total > 0 {
			seenVariants[VariantSpec{GPUType: gpuType, TPDegree: clusterTPDegree}] = struct{}{}
		}
	}
	for _, inst := range c.instances {
		switch inst.State {
		case sim.InstanceStateLoading, sim.InstanceStateWarmingUp, sim.InstanceStateActive, sim.InstanceStateDraining:
			if inst.GPU() != "" {
				usedByGPUType[inst.GPU()] += inst.TPDegree
				if inst.TPDegree > 0 {
					seenVariants[VariantSpec{GPUType: inst.GPU(), TPDegree: inst.TPDegree}] = struct{}{}
				}
			}
		}
	}

	// Step 3: build byVariant — same raw free count for each variant of the same GPUType.
	// Callers must use Variants() to iterate (R2: map iteration is non-deterministic).
	byVariant := make(map[VariantSpec]int, len(seenVariants))
	for v := range seenVariants {
		free := totalByGPUType[v.GPUType] - usedByGPUType[v.GPUType]
		if free < 0 {
			logrus.Warnf("[autoscaler] gpuInventory: variant %+v has negative free slots (%d) — bookkeeping inconsistency; clamping to 0", v, free)
			free = 0
		}
		byVariant[v] = free
	}
	return GPUInventory{byVariant: byVariant}
}

// GatewayQueueDepth returns the number of requests still in the gateway queue
// at simulation end. Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueDepth() int {
	if c.gateway == nil {
		return 0
	}
	return c.gateway.Pending()
}

// GatewayQueueRejected returns the number of requests rejected from the gateway queue
// (queue full, incoming could not displace any entry). Returns 0 when flow control is disabled.
func (c *ClusterSimulator) GatewayQueueRejected() int {
	return c.gatewayRejected
}

// GatewayEvicted returns the number of requests evicted in-flight from instances
// due to gateway-level eviction (INV-1: gw_evicted bucket).
func (c *ClusterSimulator) GatewayEvicted() int {
	return c.gatewayEvicted
}

// GatewayExpired returns the number of requests expired from the gateway queue
// via TTL (INV-1: gw_expired bucket).
func (c *ClusterSimulator) GatewayExpired() int {
	return c.gatewayExpired
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

		// Per-adapter resident-set counts are keyed by adapter id, which — unlike the
		// globally-unique request ids above — legitimately recurs across instances (the
		// same adapter can be loaded on many instances). Sum them for a cluster-wide
		// total per adapter rather than merging (which would warn and overwrite).
		for k, v := range m.AdapterLoadCounts {
			merged.AdapterLoadCounts[k] += v
		}
		for k, v := range m.AdapterEvictionCounts {
			merged.AdapterEvictionCounts[k] += v
		}
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
//   - Served parents (terminal AND the decode sub-request emitted ≥1 token):
//     sub-request entries are replaced with parent-keyed entries using
//     true user-facing values (e.g., E2E = decodeSchedulingDelay + decodeOwnE2E).
//   - Incomplete parents, and terminal-but-no-token parents (decode-side drops
//     with nil DecodeSubReq, or decode-timeout-while-queued with empty ITL): the
//     sub-request entries are removed and NO parent entry is created. These
//     requests produced no output; they are accounted for by the drop/timeout
//     counters only (DroppedUnservable / TimedOutRequests), never by the latency
//     distributions (issue #1511, INV-PD-6).
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

		// servedFirstToken: the decode sub-request has ≥1 un-discarded ITL entry at
		// finalization — i.e. a client-visible first-token measurement survived to the end.
		// It is false for the no-token TERMINAL outcomes (issue #1511):
		//   - decode-side drop at transfer start / transfer complete — DecodeSubReq is
		//     nil (nilled by dropAtStart / the late-drop path before the decode sub-request
		//     runs);
		//   - decode-timeout-while-queued — DecodeSubReq != nil but ITL is empty because it
		//     never ran a decode step; and
		//   - preempted-then-timed-out — a decode sub-request that emitted a token, was then
		//     preempted (batch_formation.go resets ProgressIndex/TTFTSet and clears ITL on
		//     preempt), and timed out while re-queued before re-emitting. Its ITL is empty at
		//     finalization, so no first-token measurement survives — correctly excluded, same
		//     as the others.
		// Each reaches a terminal CompletionTime with a DecodeInstanceID assigned (so
		// `completed` is true) yet has no surviving first token. Such a parent belongs to the
		// drop/timeout counters (DroppedUnservable via droppedAtDecodeKV; TimedOutRequests via
		// pdDecodeTimedOutCount) ONLY and must contribute NO per-request metric entry —
		// otherwise it is double-represented as both a drop and a latency sample, polluting
		// the TTFT/E2E/scheduling-delay distributions (INV-PD-6). `served` folds that
		// requirement into the projection gate below. A decode sub-request that emitted a
		// first token and THEN timed out mid-generation WITHOUT an intervening preemption
		// keeps a non-empty ITL, so it is still served: it keeps its TTFT, scheduling delay
		// and request row (the user did receive that token), but gets no E2E or completion
		// stamp, since it did not complete (the E2E block below).
		servedFirstToken := parent.DecodeSubReq != nil && len(parent.DecodeSubReq.ITL) > 0
		served := completed && servedFirstToken

		// Read the decode sub-request's own per-instance measurements before the
		// per-metric delete/rekey blocks below consume them (R1: no silent data
		// loss). Both the E2E fix (issue #1513) and the TTFT fix (issue #1510) are
		// built from these instance-frame values:
		//   - decodeDelay  = RequestSchedulingDelays[dec] = arrival → decode-schedule span
		//   - decodeOwnE2E = RequestE2Es[dec]             = decode-schedule → completion span
		// Because the decode sub-request's ArrivalTime is the parent's original
		// arrival (pd_events.go), decodeDelay spans prefill_queue + prefill_step +
		// kv_transfer + decode_queue_wait. RequestSchedulingDelays[dec] is not
		// deleted until the scheduling-delay block below; read it here so the E2E
		// and TTFT blocks share one value.
		decodeDelay, hasDecodeDelay := m.RequestSchedulingDelays[dec]
		decodeOwnE2E, hasDecodeOwnE2E := m.RequestE2Es[dec]

		// E2E: user-visible arrival → last-token span for PD disaggregation
		// (issue #1513). The decode sub-request's own per-instance E2E
		// (decodeOwnE2E = FirstTokenTime + Σ ITL + PostDecodeFixedOverhead, all
		// measured relative to the decode sub-request's ArrivalTime = the parent's
		// original arrival, pd_events.go) correctly captures the decode execution
		// span INCLUDING the decode step's own advance. Two clock frames apply,
		// discriminated by whether the decode sub-request ever ran prefill:
		//
		//   - Normal PD decode sub-request: it starts at ProgressIndex == InputLen
		//     (batch_formation.go), so the FirstTokenTime block (simulator.go) never
		//     fires and FirstTokenTime stays 0. Its own E2E is therefore measured
		//     from the decode SCHEDULE instant and omits the arrival → decode-schedule
		//     wait. Add decodeDelay to reconstitute the full arrival → completion span:
		//
		//       E2E = decodeSchedulingDelay + decodeOwnE2E
		//
		//   - Preempted-and-re-prefilled decode sub-request: preemption resets
		//     ProgressIndex to 0 and clears TTFTSet (batch_formation.go), so on
		//     re-prefill the FirstTokenTime block fires and stamps FirstTokenTime as
		//     an ARRIVAL-relative offset (simulator.go: now + step + OTPT − ArrivalTime).
		//     decodeOwnE2E is then ALREADY the full arrival → completion span, and
		//     decodeDelay (re-stamped to reschedule − arrival on re-admission,
		//     simulator.go) must NOT be added or E2E double-counts the pre-decode wait.
		//     Discriminate on FirstTokenTime != 0 and use decodeOwnE2E directly.
		//
		// This is the E2E-analog of the TTFT fix below and guarantees INV-5 by
		// construction: decodeOwnE2E ≥ ITL[0] = firstDecodeStep, so E2E ≥ TTFT. Only a
		// decode sub-request that completed has an own E2E, so only a completed parent
		// gets one.
		//
		// The previous formula (parent.CompletionTime − ArrivalTime) under-counted:
		// parent.CompletionTime was then stamped on the CLUSTER clock at the
		// completion-DETECTION tick and omitted the decode step's own advance, so for
		// short outputs the reported E2E fell below a single decode step (ITL[0]) — and
		// below the parent TTFT — violating INV-5. (Since #1903 it is stamped at the
		// final step's end.)
		//
		// parentE2E / haveParentE2E are captured for reuse by the completion-time
		// block below (metric consistency: completion == arrival + E2E).
		delete(m.RequestE2Es, pfx)
		delete(m.RequestE2Es, dec)
		var parentE2E float64
		var haveParentE2E bool
		if served {
			if hasDecodeDelay && hasDecodeOwnE2E {
				// decodeOwnE2E is schedule-relative on the normal path (FirstTokenTime
				// unset) and arrival-relative after a re-prefill (FirstTokenTime set);
				// only add decodeDelay in the former case (issue #1513 preemption fix).
				e2e := decodeOwnE2E
				preempted := parent.DecodeSubReq != nil && parent.DecodeSubReq.FirstTokenTime != 0
				if !preempted {
					e2e += float64(decodeDelay)
				}
				if e2e < 0 {
					// Defensive parity with the TTFT block: never emit a negative E2E
					// (a headline SLO metric). Unreachable in normal operation
					// (decodeDelay ≥ 0 by shared-clock event ordering, decodeOwnE2E > 0),
					// but guards a hypothetical clock regression rather than silently
					// reporting a negative value.
					logrus.Errorf("[cluster] projectPDMetrics: negative reconstructed E2E for %s (decodeDelay=%d decodeOwnE2E=%.0f preempted=%v); skipping",
						pid, decodeDelay, decodeOwnE2E, preempted)
				} else {
					parentE2E = e2e
					haveParentE2E = true
				}
			} else if hasDecodeOwnE2E {
				// A completed decode sub-request is always scheduled first, so it has a
				// scheduling delay; without one the span cannot be reconstructed.
				logrus.Errorf("[cluster] projectPDMetrics: completed decode sub-request %s has no scheduling delay; no E2E for %s",
					dec, pid)
			}
			// A decode sub-request with no own E2E did not complete: it emitted a first
			// token and then timed out mid-generation (recordRequestCompletion never ran).
			// Like a non-P/D request timed out mid-generation (simulator.go records E2E only
			// on completion), it keeps its TTFT -- the user did receive that token -- but no
			// E2E, completion time or completion index: it is a timeout (TimedOutRequests),
			// not a completion, and its deadline is not an end-to-end latency.
			if haveParentE2E {
				m.RequestE2Es[pid] = parentE2E
			}
		}

		// TTFT: user-visible time-to-first-token for PD disaggregation (issue #1510,
		// correcting the earlier #930 composition). In llm-d the first token reaches the
		// user from the decode pod, not prefill: prefill completes → KV transfers →
		// decode pod queues, recomputes the last prompt token, and samples the first
		// output token. The correct user-visible TTFT is the arrival → first-token-emitted
		// span:
		//
		//   TTFT = decodeSchedulingDelay + firstDecodeStep
		//
		// where decodeSchedulingDelay = RequestSchedulingDelays[decodeSubReqID] = the
		// decode sub-request's (schedule − arrival). Because the decode sub-request's
		// ArrivalTime is the parent's original arrival time (pd_events.go), that delay
		// already spans prefill_queue + prefill_step + kv_transfer + decode_queue_wait —
		// including the decode-queue wait that the previous
		// (prefillTTFT + transferDuration + firstDecodeStep) formula omitted. It also
		// carries exactly ONE OutputTokenProcessingTime: prefillTTFT (dropped here) held a
		// second, phantom copy; firstDecodeStep = ITL[0] contributes the single legitimate
		// one (the decode pod streams the first real token exactly once). This is
		// structurally identical to the non-PD TTFT (scheduling_delay + first-token step).
		//
		// Read prefill TTFT before deleting sub-request keys (R1: no silent data loss).
		// prefillTTFT is retained as the TTFTSum baseline: on the normal path pre-projection
		// TTFTSum holds exactly prefillTTFT for this parent (a normal decode sub-request never
		// sets FirstTokenTime, so it contributes 0). A preempted-and-re-prefilled decode
		// sub-request is the exception — it re-stamps FirstTokenTime and adds a decode term to
		// TTFTSum that this projection does not roll back; that pre-existing PD TTFTSum residual
		// is tracked in #1628 and is out of scope for the #1511 no-token fix here.
		// decodeDelay/hasDecodeDelay were read
		// above (shared with the E2E block); RequestSchedulingDelays[dec] is not
		// deleted until the scheduling-delay block below.
		//
		// Gate on `served` (terminal AND the decode sub-request emitted ≥1 token). A
		// terminal-but-no-token parent — a decode-side drop (nil DecodeSubReq, both the
		// transfer-start and late-drop paths) or a decode-timeout-while-queued (non-nil
		// DecodeSubReq with empty ITL) — contributes NO parent TTFT (issue #1511,
		// INV-PD-6). Before this fix such a parent was `completed == true` and fell to the
		// prefill-only fallback, polluting the TTFT distribution with a phantom entry.
		prefillTTFT, hasPrefillTTFT := m.RequestTTFTs[pfx]
		delete(m.RequestTTFTs, pfx)
		delete(m.RequestTTFTs, dec)
		if served {
			// `served` guarantees DecodeSubReq != nil && len(ITL) > 0, so the decode
			// sub-request emitted a first token — including the case where it emitted that
			// token and THEN timed out mid-generation (a genuine arrival→first-token span;
			// the user did receive the token). The primary branch additionally needs the
			// recorded decode scheduling delay; without it, fall back to the prefill-only
			// TTFT (defensive — decode data partially unavailable).
			if hasPrefillTTFT && hasDecodeDelay {
				firstDecodeStep := float64(parent.DecodeSubReq.ITL[0])
				newTTFT := float64(decodeDelay) + firstDecodeStep
				if newTTFT < 0 {
					// Defensive parity with the E2E block above: never emit a negative
					// TTFT (a headline SLO metric). Unreachable in normal operation —
					// decodeDelay ≥ 0 by shared-clock event ordering and firstDecodeStep > 0
					// — but guards against a hypothetical clock regression rather than
					// silently reporting a negative value.
					logrus.Errorf("[cluster] projectPDMetrics: negative TTFT for %s (decodeDelay=%d firstDecodeStep=%.0f); using prefill TTFT",
						pid, decodeDelay, firstDecodeStep)
					m.RequestTTFTs[pid] = prefillTTFT // delta 0 vs baseline: TTFTSum untouched
				} else {
					m.RequestTTFTs[pid] = newTTFT
					// BC-3: Keep TTFTSum consistent with the TTFT adjustment. The baseline
					// prefillTTFT is replaced by newTTFT for this parent.
					m.TTFTSum += int64(newTTFT - prefillTTFT)
				}
			} else if hasPrefillTTFT {
				// Defensive fallback: served a token but the decode scheduling delay is
				// unavailable; use the prefill-only TTFT. TTFTSum unchanged (projected
				// value equals the baseline).
				m.RequestTTFTs[pid] = prefillTTFT
				logrus.Warnf("[cluster] projectPDMetrics: parent %s served a token but is missing its decode scheduling delay; using prefill TTFT", pid)
			} else {
				logrus.Warnf("[cluster] projectPDMetrics: served parent %s has no prefill TTFT (key %s)", pid, pfx)
			}
		} else if completed && hasPrefillTTFT {
			// Terminal-but-no-token parent (decode-side drop / timeout-while-queued /
			// preempted-then-timed-out, #1511): it has no surviving first token, so it
			// contributes NO parent TTFT. The prefill sub-request's TTFT key was deleted
			// above, but its value is still summed into the aggregate TTFTSum; roll the
			// PREFILL term back so the reported mean TTFT (TTFTSum / n) does not count a
			// request that produced no token.
			//
			// Scope: this rolls back the prefill term only, which is exactly what the three
			// no-token outcomes this fix restores require. A decode sub-request that emitted a
			// token, was preempted (re-stamping FirstTokenTime and adding a decode term to
			// TTFTSum, simulator.go), then timed out while re-queued also leaves that decode
			// term in TTFTSum, which neither this rollback nor the unconditional
			// delete(RequestTTFTs, dec) removes — a pre-existing PD TTFTSum residual tracked in
			// #1628, not introduced here. So TTFTSum == Σ RequestTTFTs holds for the no-token
			// terminal outcomes but is NOT a general PD law (see #1628).
			m.TTFTSum -= int64(prefillTTFT)
		}

		// Scheduling delay = prefill sub-request's delay
		// (the real user-facing delay, not the decode pipeline cumulative latency).
		// Gated on `served`: a no-token terminal parent contributes no scheduling-delay
		// sample either, so it does not pollute the scheduling-delay distribution (#1511).
		prefillDelay, hasPrefillDelay := m.RequestSchedulingDelays[pfx]
		delete(m.RequestSchedulingDelays, pfx)
		delete(m.RequestSchedulingDelays, dec)
		if served && hasPrefillDelay {
			m.RequestSchedulingDelays[pid] = prefillDelay
		}

		// Requests metadata keyed by parent ID, HandledBy set to decode instance.
		// Gated on `served`: a no-token terminal parent (drop/timeout) contributes no
		// entry (INV-PD-6); it is represented by the drop/timeout counters (#1511).
		decodeEntry := m.Requests[dec]
		delete(m.Requests, pfx)
		delete(m.Requests, dec)
		if served {
			if parent.OriginalRequest == nil {
				panic(fmt.Sprintf("projectPDMetrics: parent %s has nil OriginalRequest", pid))
			}
			rm := sim.NewRequestMetrics(parent.OriginalRequest, float64(parent.ArrivalTime)/1e6)
			rm.HandledBy = string(parent.DecodeInstanceID)
			// The parent completes when its decode sub-request does, on the decode instance,
			// so it takes that completion's place in the completion order.
			m.Requests[pid] = rm.WithCompletionOf(decodeEntry)
		}

		// ITL from decode sub-request (prefill ITL is 0 noise).
		decodeITL, hasDecodeITL := m.RequestITLs[dec]
		delete(m.RequestITLs, pfx)
		delete(m.RequestITLs, dec)
		if served && hasDecodeITL {
			m.RequestITLs[pid] = decodeITL
		}

		// Completion-time METRIC. Kept consistent with the projected E2E so the
		// non-PD identity completion_metric == ArrivalTime + E2E holds (non-PD sets
		// both from the same `lat`, simulator.go). Without this, the E2E fix (issue
		// #1513) and the parent.CompletionTime-based completion metric would disagree
		// by the decode step advance, and computeSessionMetrics (metrics.go), which
		// derives session duration from RequestCompletionTimes, would inherit the same
		// under-count.
		//
		// This adjusts only the METRIC, not the lifecycle field parent.CompletionTime,
		// which is intentionally left untouched: it drives session follow-up arrival
		// scheduling (sessionCallback in detectDecodeCompletions) and phase-causality
		// checks (INV-10). A parent with no E2E (it timed out mid-generation) gets no
		// completion time either, as in the non-P/D path.
		delete(m.RequestCompletionTimes, pfx)
		delete(m.RequestCompletionTimes, dec)
		if served && haveParentE2E {
			m.RequestCompletionTimes[pid] = float64(parent.ArrivalTime) + parentE2E
		}
	}
}
