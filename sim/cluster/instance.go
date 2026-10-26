// Package cluster provides multi-replica cluster simulation capabilities.
//
// This package wraps the single-instance simulator (sim.Simulator) to enable
// multi-replica coordination via ClusterSimulator.
package cluster

import (
	"fmt"
	"sort"

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
	// onTransition, when set, is called after every lifecycle transition.
	onTransition func(*InstanceSimulator)
	id           InstanceID
	sim          *sim.Simulator
	hasRun       bool

	// Phase 1A: lifecycle and placement fields.
	// All zero-value safe (backward-compatible with no-node-pool mode).
	Model            string            // target model identifier (empty = default/single-model)
	State            sim.InstanceState // lifecycle state; empty = untracked (backward-compat)
	warmUpRemaining  int               // requests remaining in warm-up phase; 0 = no warm-up
	warmUpRequestIDs []string          // IDs of requests served during warm-up (for TTFT factor)
	nodeID           string            // primary node this instance is placed on (empty = unplaced). For a multi-node TP instance (#1529) this is only the lowest-index node; the full node set is PlacementManager.distinctNodesForGPUs(allocatedGPUIDs).
	allocatedGPUIDs  []string          // GPU IDs allocated to this instance (spans >1 node for multi-node TP)
	endpointTopology EndpointTopology  // captured at placement; never inferred from the primary nodeID
	gpu              string            // GPU type used for this instance (set from pool gpu_type, or config.GPU)

	// Phase 1C: hardware variant fields (set at placement time in cluster.go).
	// Used by buildRouterState() to populate RoutingSnapshot for the autoscaler Collector.
	// GPUType is available via inst.GPU(); TPDegree and CostPerHour are autoscaler-specific.
	TPDegree    int     // tensor-parallel degree; 0 = unplaced/unknown
	CostPerHour float64 // $/hr from NodePool.CostPerHour; 0 = unplaced/free tier

	// maxNumSeqs stores cfg.BatchConfig.MaxNumSeqs at construction time.
	// Exposed via MaxBatchSize() for the autoscaler pipeline.
	maxNumSeqs int64
}

// NewInstanceSimulator creates an InstanceSimulator from a SimConfig struct.
//
// Thread-safety: NOT thread-safe. Must be called from single goroutine.
// Failure modes: Panics if internal Simulator creation fails (matches existing behavior).
func NewInstanceSimulator(id InstanceID, cfg sim.SimConfig) *InstanceSimulator {
	// The engine (its KV store, adapter-cost wrapper and simulator) is built in one place,
	// sim.NewEngine (#2021 E2.2).
	s, err := sim.NewEngine(string(id), cfg)
	if err != nil {
		panic(fmt.Sprintf("NewInstanceSimulator(%s): %v", id, err))
	}
	return &InstanceSimulator{
		id:         id,
		sim:        s,
		gpu:        cfg.GPU,
		maxNumSeqs: cfg.MaxNumSeqs,
	}
}

// GPU returns the GPU type this instance was constructed with.
// When NodePools are configured, this reflects the pool's gpu_type (authoritative).
// When NodePools are absent, this reflects config.GPU (the CLI flag).
func (i *InstanceSimulator) GPU() string { return i.gpu }

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

// PostDecodeFixedOverhead returns the fixed per-request post-decode overhead (µs)
// from the instance's underlying latency model. Used by detectDecodeCompletions
// to stamp parent.CompletionTime with the correct client-visible completion time.
func (i *InstanceSimulator) PostDecodeFixedOverhead() int64 {
	return i.sim.PostDecodeFixedOverhead()
}

// InjectRequest delegates to sim.InjectArrival. Panics if called after Run().
func (i *InstanceSimulator) InjectRequest(req *sim.Request) {
	if i.hasRun {
		panic("InstanceSimulator.InjectRequest() called after Run()")
	}
	i.sim.InjectArrival(req)
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
	rate, counts, counted := i.sim.KVThrashing()
	i.sim.Metrics.KVThrashingRate = rate
	// The thrash counts behind the rate, so the cluster aggregate weights each instance
	// by its opportunities (#1904). The hit rate is the instance's prompt-token lookups
	// (Metrics.recordPrefixLookup), the quantity vLLM's prefix-cache stats count, rather
	// than the store's block tally, which also counts decode growth.
	if counted {
		i.sim.Metrics.KVThrashEvents = counts.Events
		i.sim.Metrics.KVThrashOpportunities = counts.Opportunities
	}
	i.sim.Metrics.CacheHitRate = i.sim.Metrics.PromptCacheHitRate()
}

// QueueDepth returns the number of requests in the wait queue.
func (i *InstanceSimulator) QueueDepth() int {
	return i.sim.QueueDepth()
}

// BatchSize returns the number of requests in the running batch, or 0 if nil.
func (i *InstanceSimulator) BatchSize() int {
	return i.sim.BatchSize()
}

// ResidentAdapterIDs returns the ids of LoRA adapters currently resident on this
// instance, or nil when the LoRA subsystem is inert.
func (i *InstanceSimulator) ResidentAdapterIDs() []string {
	if i.sim == nil {
		return nil
	}
	return i.sim.ResidentAdapterIDs()
}

// KVUtilization returns the fraction of KV cache blocks in use.
// Returns 0 when TotalCapacity is 0 to avoid division by zero (R11 defensive guard).
func (i *InstanceSimulator) KVUtilization() float64 {
	total := i.sim.KVCapacity()
	if total <= 0 {
		return 0
	}
	return float64(i.sim.KVUsed()) / float64(total)
}

// FreeKVBlocks returns the number of free KV cache blocks.
func (i *InstanceSimulator) FreeKVBlocks() int64 {
	return i.sim.KVCapacity() - i.sim.KVUsed()
}

// CacheHitRate returns the cumulative cache hit rate.
func (i *InstanceSimulator) CacheHitRate() float64 {
	return i.sim.KVCacheHitRate()
}

// TotalKvCapacityTokens returns total KV cache capacity in tokens.
// Returns 0 if i.sim or its KV store is nil (construction defect), or if
// TotalCapacity() * BlockSize() is zero (misconfigured KV store).
// Loading instances have correct non-zero capacity from construction —
// NewInstanceSimulator always initializes the KV store before any lifecycle transition.
func (i *InstanceSimulator) TotalKvCapacityTokens() int64 {
	if i.sim == nil {
		return 0
	}
	return i.sim.KVCapacity() * i.sim.KVBlockSize()
}

// KvTokensInUse returns current KV cache occupancy in tokens.
// Returns 0 when the KV cache is not yet initialized.
func (i *InstanceSimulator) KvTokensInUse() int64 {
	if i.sim == nil {
		return 0
	}
	return i.sim.KVUsed() * i.sim.KVBlockSize()
}

// TotalKVBlocks returns the total number of KV cache blocks for this instance.
func (i *InstanceSimulator) TotalKVBlocks() int64 {
	if i.sim == nil {
		return 0
	}
	return i.sim.KVCapacity()
}

// PreemptionCount returns the cumulative number of preemption events on this instance.
func (i *InstanceSimulator) PreemptionCount() int64 {
	return i.sim.Metrics.PreemptionCount
}

// InstanceLatencyStats holds cumulative averages of per-instance latency and throughput from completed requests.
// Units: TTFT and ITL are in microseconds (ticks = µs in the simulator clock).
// DispatchRate is in req/s; AvgInTokens and AvgOutTokens are per-request averages.
// All fields are zero when no requests have completed.
type InstanceLatencyStats struct {
	TTFT         float64 // µs
	ITL          float64 // µs
	DispatchRate float64 // req/s
	AvgInTokens  float64
	AvgOutTokens float64
}

// LatencyStats returns aggregate latency and throughput statistics from completed requests.
// All fields are 0 when no requests have completed.
// TTFT and ITL are in microseconds; DispatchRate is in req/s.
func (i *InstanceSimulator) LatencyStats() InstanceLatencyStats {
	if i.sim == nil {
		return InstanceLatencyStats{}
	}
	m := i.sim.Metrics
	if m == nil || m.CompletedRequests == 0 {
		return InstanceLatencyStats{}
	}
	n := float64(m.CompletedRequests)

	// DispatchRate: when simulation has ended, use SimEndedTime as the denominator
	// (exact throughput, matches ResponsesPerSec in metrics.go:SaveResults).
	// Mid-simulation fallback: span between first and last completion time (growing window —
	// rate decreases over time even at steady throughput). Final fallback: current clock.
	// INV-6 note: min/max reduction is order-independent (unlike float sums), so this map
	// range is exempt from the R2 sort requirement.
	var minCT, maxCT float64
	first := true
	for _, ct := range m.RequestCompletionTimes {
		if first || ct < minCT {
			minCT = ct
		}
		if first || ct > maxCT {
			maxCT = ct
		}
		first = false
	}
	var dispatchRate float64
	if m.SimEndedTime > 0 {
		dispatchRate = n / (float64(m.SimEndedTime) / 1e6)
	} else if span := maxCT - minCT; span > 0 {
		dispatchRate = n / (span / 1e6)
	} else if clockUs := i.Clock(); clockUs > 0 {
		dispatchRate = n / (float64(clockUs) / 1e6)
	}

	// Compute ITL from RequestITLs (per-request average ITL in µs).
	// Sort keys before accumulating to satisfy INV-6 determinism (R2).
	// ITLSum is never populated by the simulator; RequestITLs is the authoritative source.
	reqIDs := make([]string, 0, len(m.RequestITLs))
	for id := range m.RequestITLs {
		reqIDs = append(reqIDs, id)
	}
	sort.Strings(reqIDs)
	var itlSum float64
	for _, id := range reqIDs {
		itlSum += m.RequestITLs[id]
	}

	return InstanceLatencyStats{
		TTFT:         float64(m.TTFTSum) / n,
		ITL:          itlSum / n,
		DispatchRate: dispatchRate,
		AvgInTokens:  float64(m.TotalInputTokens) / n,
		AvgOutTokens: float64(m.TotalOutputTokens) / n,
	}
}

// MaxBatchSize returns the simulator's configured maximum number of concurrent requests
// (BatchConfig.MaxNumSeqs). Returns 0 when the instance has no underlying simulator.
func (i *InstanceSimulator) MaxBatchSize() int {
	if i.sim == nil {
		return 0
	}
	return int(i.maxNumSeqs)
}

// GetCachedBlockCount returns the number of consecutive cached prefix blocks
// matching the given token sequence. Used by precise prefix cache scoring.
func (i *InstanceSimulator) GetCachedBlockCount(tokens []sim.TokenID) int {
	if i.sim == nil {
		return 0
	}
	return i.sim.CachedBlockCount(tokens)
}

// SnapshotCacheQueryFn returns a function that queries a frozen copy of this
// instance's KV cache hash map. The returned function is safe to call after
// the live cache state has changed — it always returns results as of snapshot time.
// Returns a zero-returning function if the simulator is nil or the KV cache
// does not support snapshotting.
func (i *InstanceSimulator) SnapshotCacheQueryFn() func([]sim.TokenID) int {
	if i.sim == nil {
		return func([]sim.TokenID) int { return 0 }
	}
	if fn, ok := i.sim.SnapshotCacheQuery(); ok {
		return fn
	}
	// Fallback: live query (for KVStore implementations without snapshot support).
	// Callers (CachedSnapshotProvider) are responsible for warning about stale-cache
	// semantics not being honored — this function is intentionally side-effect-free.
	return func(tokens []sim.TokenID) int {
		return i.GetCachedBlockCount(tokens)
	}
}

// InjectRequestOnline injects a request during the event loop (online routing mode).
// Unlike InjectRequest, this does NOT check hasRun, allowing injection during simulation.
func (i *InstanceSimulator) InjectRequestOnline(req *sim.Request, eventTime int64) {
	i.sim.InjectArrivalAt(req, eventTime)
}

// IsRoutable returns true if this instance should appear in routing snapshots.
// Active and WarmingUp instances are routable.
// When State is empty (no lifecycle tracking), all instances are treated as routable
// for backward compatibility with pre-Phase-1A cluster tests.
func (i *InstanceSimulator) IsRoutable() bool {
	switch i.State {
	case sim.InstanceStateActive, sim.InstanceStateWarmingUp:
		return true
	case "": // untracked — backward-compat
		return true
	default:
		return false
	}
}

// HasSim returns true if the instance has an underlying simulator (false in test-only scenarios).
func (i *InstanceSimulator) HasSim() bool {
	return i.sim != nil
}

// IsWarmingUp returns true if the warm-up TTFT penalty should be applied.
func (i *InstanceSimulator) IsWarmingUp() bool {
	return i.State == sim.InstanceStateWarmingUp && i.warmUpRemaining > 0
}

// RecordWarmUpRequest marks a request ID as having been served during warm-up.
// Called at routing time when the instance IsWarmingUp(). The TTFT for these
// requests will be multiplied by WarmUpTTFTFactor in aggregateMetrics().
func (i *InstanceSimulator) RecordWarmUpRequest(reqID string) {
	i.warmUpRequestIDs = append(i.warmUpRequestIDs, reqID)
}

// WarmUpRequestIDs returns the IDs of requests that were routed during warm-up.
func (i *InstanceSimulator) WarmUpRequestIDs() []string {
	return i.warmUpRequestIDs
}

// clearWarmUpRequestIDs frees the warm-up request ID slice after the TTFT factor
// has been applied in aggregateMetrics(), preventing unbounded memory growth.
func (i *InstanceSimulator) clearWarmUpRequestIDs() {
	i.warmUpRequestIDs = nil
}

// ConsumeWarmUpRequest decrements the warm-up counter.
// When it reaches zero, automatically transitions WarmingUp → Active.
func (i *InstanceSimulator) ConsumeWarmUpRequest() {
	if i.warmUpRemaining <= 0 {
		return
	}
	i.warmUpRemaining--
	if i.warmUpRemaining == 0 && i.State == sim.InstanceStateWarmingUp {
		i.TransitionTo(sim.InstanceStateActive)
	}
}

// validInstanceTransitions maps valid source → target pairs for instance lifecycle.
// This table IS INV-14 (see docs/contributing/standards/invariants.md#inv-14): the
// legal edge set, which is also monotone — every edge advances the instance along
// Scheduling < Loading < WarmingUp < Active < Draining < Terminated and none regresses.
// Adding a backward edge here fails TestInstanceStateMachine_NoBackwardTransitions.
var validInstanceTransitions = map[sim.InstanceState]map[sim.InstanceState]struct{}{
	sim.InstanceStateScheduling: {sim.InstanceStateLoading: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateLoading:    {sim.InstanceStateWarmingUp: {}, sim.InstanceStateActive: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateWarmingUp:  {sim.InstanceStateActive: {}, sim.InstanceStateDraining: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateActive:     {sim.InstanceStateDraining: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateDraining:   {sim.InstanceStateTerminated: {}},
	sim.InstanceStateTerminated: {},
}

// TransitionTo validates and applies an instance state transition. It is the only
// writer of State outside construction, which is what makes validInstanceTransitions
// load-bearing rather than advisory.
//
// Panics on an invalid transition — INV-14 clause 1 (invariant violation per
// Principle V). Initializes State on first call when State is empty: INV-14
// clause 2, the seeding carve-out for runs with lifecycle tracking disabled, so
// clause 1 constrains an instance's second and later transitions.
func (i *InstanceSimulator) TransitionTo(state sim.InstanceState) {
	if i.State == "" {
		// INV-14 clause 2: lifecycle tracking not enabled — accept this one
		// transition unvalidated to seed State. Subsequent calls are validated.
		i.State = state
		return
	}
	targets, ok := validInstanceTransitions[i.State]
	if !ok {
		panic(fmt.Sprintf("TransitionTo %s: unknown source state %q — INV-14 violation", i.id, i.State))
	}
	if _, valid := targets[state]; !valid {
		panic(fmt.Sprintf("TransitionTo %s: invalid transition %q → %q — INV-14 violation", i.id, i.State, state))
	}
	i.State = state
	if i.onTransition != nil {
		i.onTransition(i)
	}
}

// ReserveTransferredKV reserves KV cache blocks on this instance for a decode
// sub-request whose KV transfer is in flight (issue #1343, vLLM
// WAITING_FOR_REMOTE_KVS parity), and returns how many prompt tokens the pull must
// move: those this instance's prefix cache does not already hold (sim.Simulator.
// ReserveRemoteKV). Blocks are allocated immediately so they reduce available capacity
// for other requests during the transfer window. ProgressIndex is advanced past the
// input so batch formation treats the request as decode-ready once the transfer
// completes and the state is promoted to StateQueued. Returns ok=false if this instance
// lacks the KV capacity.
//
// Caller is responsible for calling ReleaseReservedKV (or the equivalent
// KVCache.ReleaseKVBlocks) if the reservation must be undone — e.g., when
// the decode instance becomes non-routable between KVTransferStartedEvent
// and KVTransferCompletedEvent.
func (i *InstanceSimulator) ReserveTransferredKV(req *sim.Request) (pulledTokens int64, ok bool) {
	return i.sim.ReserveRemoteKV(req)
}

// CacheRemoteKV registers a decode sub-request's pulled blocks in this instance's prefix
// cache once the pull has landed (sim.Simulator.CacheRemoteKV).
func (i *InstanceSimulator) CacheRemoteKV(req *sim.Request) { i.sim.CacheRemoteKV(req) }

// ReleaseReservedKV frees KV blocks previously reserved via ReserveTransferredKV.
// Used when a reservation must be undone (e.g., decode instance transitioned
// to a non-routable state mid-transfer). Safe to call when the request holds
// no blocks.
func (i *InstanceSimulator) ReleaseReservedKV(req *sim.Request) {
	i.sim.ReleaseKV(req)
}

// ReleaseHeldKV frees the KV blocks a finished request kept past completion
// (sim.Request.RetainKVOnFinish): a P/D prefill sub-request's, once the decode side has pulled
// them (INV-PD-7). The blocks go back to the free list with their prefix hashes, as any
// finished request's do.
func (i *InstanceSimulator) ReleaseHeldKV(req *sim.Request) {
	if !req.RetainKVOnFinish || req.State != sim.StateCompleted {
		panic(fmt.Sprintf("ReleaseHeldKV %s: request %s is not a finished request that retains its KV",
			i.id, req.ID))
	}
	i.sim.ReleaseKV(req)
}

// WakeIfParked wakes a parked engine for KV freed at cluster time t (sim.Simulator.WakeIfParked).
func (i *InstanceSimulator) WakeIfParked(t int64) { i.sim.WakeIfParked(t) }

// KVBlocksOf returns the GPU blocks req holds on this instance (the store's own slice).
func (i *InstanceSimulator) KVBlocksOf(req *sim.Request) []int64 {
	blocks, ok := i.sim.RequestBlocks(req)
	if !ok {
		panic(fmt.Sprintf("instance %s: KV store %s does not report a request's blocks", i.id, i.sim.KVStoreKind()))
	}
	return blocks
}

// AbortReservedKV cancels a reservation whose transfer was evicted before it completed at time
// now: its blocks are freed (their hashes were never registered, ReserveTransferredKV), and the
// engine publishes its state after the abort.
func (i *InstanceSimulator) AbortReservedKV(req *sim.Request, now int64) {
	i.sim.ReleaseKV(req)
	i.sim.NoteAbort(now) // also wakes a parked engine (scheduleStepAt)
}

// InjectDecodeOnline injects a decode sub-request with pre-allocated KV.
// Bypasses the normal ArrivalEvent → QueuedEvent → EnqueueRequest chain to avoid
// the oversized-request guard (KV already allocated) and TotalInputTokens double-counting.
// Registers request in metrics and directly enqueues into wait queue.
// clusterTime is the cluster clock at the time of injection (from KVTransferCompletedEvent).
func (i *InstanceSimulator) InjectDecodeOnline(req *sim.Request, clusterTime int64) {
	i.sim.Metrics.Requests[req.ID] = sim.NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
	i.sim.EnqueueDecodeSubRequest(req, clusterTime)
}

// DrainWaitQueue extracts all pending (queued but not yet scheduled) requests
// from the instance's wait queue and returns them.
// Used by DrainRedirect to re-inject requests elsewhere.
func (i *InstanceSimulator) DrainWaitQueue() []*sim.Request {
	return i.sim.DrainWaitQueue()
}

// EvictRequest removes a request from this instance due to gateway-level eviction.
// Searches WaitQ first, then RunningBatch. Frees KV blocks if allocated.
// Sets req.State to StateCompleted to prevent dangling TimeoutEvents from double-counting.
// Returns true if found and removed, false otherwise (idempotent for already-completed).
//
// INV-2 (request lifecycle): this is the one semantically anomalous StateCompleted
// write in the codebase — a TOMBSTONE, not a completion. The request produced no
// output and is counted as gatewayEvicted (sim/cluster/cluster_event.go), never via
// the sole CompletedRequests++ in sim/simulator.go. So INV-2's `-> completed` edge
// does not describe this line; reading StateCompleted as "completed" here would
// double-count an evicted request.
func (i *InstanceSimulator) EvictRequest(req *sim.Request, now int64) bool {
	if !i.evictRequest(req) {
		return false
	}
	i.sim.NoteAbort(now) // the engine's next step publishes the state after the eviction
	return true
}

func (i *InstanceSimulator) evictRequest(req *sim.Request) bool { return i.sim.Evict(req) }

// AdmitRemoteKV hands the instance's waiting handoffs, in the order they arrived, to its engine,
// which admits, in its own order, those whose pull may start (sim.EngineCore.AdmitRemoteKV,
// INV-PD-9).
func (i *InstanceSimulator) AdmitRemoteKV(waiting []*sim.Request) []sim.RemoteKVAdmission {
	return i.sim.AdmitRemoteKV(waiting)
}

// RemoteKVLanded tells the engine a handoff's pull has landed, or was dropped, so its first-step
// blocks no longer count against new pulls.
func (i *InstanceSimulator) RemoteKVLanded(req *sim.Request) { i.sim.RemoteKVLanded(req) }

// UsedKVBlocks returns the KV blocks the instance's engine has allocated.
func (i *InstanceSimulator) UsedKVBlocks() int64 { return i.sim.KVUsed() }

// EngineMetricsAt returns the metrics a scrape of this instance's model server at time t sees.
func (i *InstanceSimulator) EngineMetricsAt(t int64) sim.EngineMetrics {
	return i.sim.EngineMetricsAt(t)
}

// CachedBlockHashes returns the hashes of the blocks the instance's prefix cache can serve,
// sorted (nil for a KV cache that does not track them).
func (i *InstanceSimulator) CachedBlockHashes() []string {
	return i.sim.CachedBlockHashes()
}

// CPUBlockHashes returns the hashes of the blocks the instance's CPU KV tier holds, sorted (nil
// without one).
func (i *InstanceSimulator) CPUBlockHashes() []string {
	return i.sim.CPUBlockHashes()
}

// TierBlockHashes returns the hashes of the blocks the secondary KV-offload tiers publishing as
// medium (e.g. "shared_storage" for a filesystem tier) hold, sorted (nil without one).
func (i *InstanceSimulator) TierBlockHashes(name string) []string {
	return i.sim.TierBlockHashes(name)
}

// SetKVEventSink makes the instance's engine publish its KV events (sim.Simulator.SetKVEventSink).
func (i *InstanceSimulator) SetKVEventSink(sink func(at int64, events []sim.KVEvent)) {
	i.sim.SetKVEventSink(sink)
}

// SetRemoteKVAdmission installs the engine's admission of remote-KV pulls (sim.Simulator's).
func (i *InstanceSimulator) SetRemoteKVAdmission(f func(now int64)) { i.sim.SetRemoteKVAdmission(f) }

// SetRemoteKVWaiting records how many handoffs wait for this engine's KV.
func (i *InstanceSimulator) SetRemoteKVWaiting(n int) { i.sim.SetRemoteKVWaiting(n) }

// HasWork reports whether the engine has running or waiting requests.
func (i *InstanceSimulator) HasWork() bool { return i.sim.HasWork() }

// BusyUntil is when the engine's last executed step ends.
func (i *InstanceSimulator) BusyUntil() int64 { return i.sim.BusyUntil() }

// SetCallbacks installs the engine's first-token and request-done callbacks.
func (i *InstanceSimulator) SetCallbacks(onFirstToken func(*sim.Request, int64), onRequestDone func(*sim.Request, int64) []*sim.Request) {
	i.sim.OnFirstToken = onFirstToken
	i.sim.OnRequestDone = onRequestDone
}

// ForgetRequest drops a request's per-request record from the engine's metrics, for a request
// drained away to another instance (the destination re-registers it).
func (i *InstanceSimulator) ForgetRequest(id string) {
	if i.sim != nil {
		delete(i.sim.Metrics.Requests, id)
	}
}

// engineCore is the instance's engine for the replica and report builders that still take the
// concrete type (sim.NewLockstepGroup, sim.ShareRenderer, sim.BuildServerSeries); nil for an
// instance with no engine. It is the one place sim/cluster names the concrete engine type
// outside this file, and goes when those builders move with the engine (#2027, addendum 4 item 2).
func (i *InstanceSimulator) engineCore() *sim.Simulator { return i.sim }

// AdapterCapacity returns the resident LoRA adapter slots (vLLM max_loras), 0 when LoRA is inert.
func (i *InstanceSimulator) AdapterCapacity() int { return i.sim.AdapterCapacity() }
