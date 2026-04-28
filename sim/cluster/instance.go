// Package cluster provides multi-replica cluster simulation capabilities.
//
// This package wraps the single-instance simulator (sim.Simulator) to enable
// multi-replica coordination via ClusterSimulator.
package cluster

import (
	"fmt"
	"sort"

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

	// Phase 1A: lifecycle and placement fields.
	// All zero-value safe (backward-compatible with no-node-pool mode).
	Model            string        // target model identifier (empty = default/single-model)
	State            InstanceState // lifecycle state; empty = untracked (backward-compat)
	warmUpRemaining  int           // requests remaining in warm-up phase; 0 = no warm-up
	warmUpRequestIDs []string      // IDs of requests served during warm-up (for TTFT factor)
	nodeID           string        // node this instance is placed on (empty = unplaced)
	allocatedGPUIDs  []string      // GPU IDs allocated to this instance
	gpu              string        // GPU type used for this instance (set from pool gpu_type, or config.GPU)

	// Phase 1C: hardware variant fields (set at placement time in cluster.go).
	// Used by buildRouterState() to populate RoutingSnapshot for the autoscaler Collector.
	// GPUType is available via inst.GPU(); TPDegree and CostPerHour are autoscaler-specific.
	TPDegree    int     // tensor-parallel degree; 0 = unplaced/unknown
	CostPerHour float64 // $/hr from NodePool.CostPerHour; 0 = unplaced/free tier

	// maxRunningReqs stores cfg.BatchConfig.MaxRunningReqs at construction time.
	// Exposed via MaxBatchSize() for the autoscaler pipeline (Task 3).
	maxRunningReqs int64
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
		id:             id,
		sim:            s,
		gpu:            cfg.GPU,
		maxRunningReqs: cfg.MaxRunningReqs,
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
// Returns 0 for roofline; non-zero for trained-physics (BC-2, #846).
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
// Returns 0 when TotalCapacity is 0 to avoid division by zero (R11 defensive guard).
func (i *InstanceSimulator) KVUtilization() float64 {
	total := i.sim.KVCache.TotalCapacity()
	if total <= 0 {
		return 0
	}
	return float64(i.sim.KVCache.UsedBlocks()) / float64(total)
}

// FreeKVBlocks returns the number of free KV cache blocks.
func (i *InstanceSimulator) FreeKVBlocks() int64 {
	return i.sim.KVCache.TotalCapacity() - i.sim.KVCache.UsedBlocks()
}

// CacheHitRate returns the cumulative cache hit rate.
func (i *InstanceSimulator) CacheHitRate() float64 {
	return i.sim.KVCache.CacheHitRate()
}

// TotalKvCapacityTokens returns total KV cache capacity in tokens.
// Returns 0 when the KV cache is not yet initialized (e.g., instance still loading).
func (i *InstanceSimulator) TotalKvCapacityTokens() int64 {
	if i.sim == nil || i.sim.KVCache == nil {
		return 0
	}
	return i.sim.KVCache.TotalCapacity() * i.sim.KVCache.BlockSize()
}

// KvTokensInUse returns current KV cache occupancy in tokens.
// Returns 0 when the KV cache is not yet initialized.
func (i *InstanceSimulator) KvTokensInUse() int64 {
	if i.sim == nil || i.sim.KVCache == nil {
		return 0
	}
	return i.sim.KVCache.UsedBlocks() * i.sim.KVCache.BlockSize()
}

// PreemptionCount returns the cumulative number of preemption events on this instance.
func (i *InstanceSimulator) PreemptionCount() int64 {
	return i.sim.Metrics.PreemptionCount
}

// InstanceLatencyStats holds rolling-window averages of per-instance latency and throughput.
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

	// DispatchRate: use the span between first and last completion time so that
	// idle time at simulation start does not dilute the rate (rolling-window approximation).
	// Falls back to total-elapsed-time when all completions share the same tick.
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
	if span := maxCT - minCT; span > 0 {
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
// (BatchConfig.MaxRunningReqs). Returns 0 when the instance has no underlying simulator.
func (i *InstanceSimulator) MaxBatchSize() int {
	if i.sim == nil {
		return 0
	}
	return int(i.maxRunningReqs)
}

// GetCachedBlockCount returns the number of consecutive cached prefix blocks
// matching the given token sequence. Used by precise prefix cache scoring.
func (i *InstanceSimulator) GetCachedBlockCount(tokens []int) int {
	if i.sim == nil {
		return 0
	}
	return len(i.sim.KVCache.GetCachedBlocks(tokens))
}

// cacheSnapshotCapable is satisfied by KVStore implementations that can produce
// a frozen snapshot query function. Both KVCacheState and TieredKVCache implement this.
// Used for stale cache signal simulation (issue #919).
type cacheSnapshotCapable interface {
	SnapshotCachedBlocksFn() func([]int) int
}

// SnapshotCacheQueryFn returns a function that queries a frozen copy of this
// instance's KV cache hash map. The returned function is safe to call after
// the live cache state has changed — it always returns results as of snapshot time.
// Returns a zero-returning function if the simulator is nil or the KV cache
// does not support snapshotting.
func (i *InstanceSimulator) SnapshotCacheQueryFn() func([]int) int {
	if i.sim == nil {
		return func([]int) int { return 0 }
	}
	if cs, ok := i.sim.KVCache.(cacheSnapshotCapable); ok {
		return cs.SnapshotCachedBlocksFn()
	}
	// Fallback: live query (for KVStore implementations without snapshot support).
	// Callers (CachedSnapshotProvider) are responsible for warning about stale-cache
	// semantics not being honored — this function is intentionally side-effect-free.
	return func(tokens []int) int {
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
	case InstanceStateActive, InstanceStateWarmingUp:
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
	return i.State == InstanceStateWarmingUp && i.warmUpRemaining > 0
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
	if i.warmUpRemaining == 0 && i.State == InstanceStateWarmingUp {
		i.TransitionTo(InstanceStateActive)
	}
}

// validInstanceTransitions maps valid source → target pairs for instance lifecycle.
var validInstanceTransitions = map[InstanceState]map[InstanceState]struct{}{
	InstanceStateScheduling: {InstanceStateLoading: {}, InstanceStateTerminated: {}},
	InstanceStateLoading:    {InstanceStateWarmingUp: {}, InstanceStateActive: {}, InstanceStateTerminated: {}},
	InstanceStateWarmingUp:  {InstanceStateActive: {}, InstanceStateDraining: {}, InstanceStateTerminated: {}},
	InstanceStateActive:     {InstanceStateDraining: {}, InstanceStateTerminated: {}},
	InstanceStateDraining:   {InstanceStateTerminated: {}},
	InstanceStateTerminated: {},
}

// TransitionTo validates and applies an instance state transition.
// Panics on invalid transition (invariant violation per Principle V).
// No-op when State is empty (backward-compat: lifecycle not tracked).
func (i *InstanceSimulator) TransitionTo(state InstanceState) {
	if i.State == "" {
		// Lifecycle tracking not enabled — silently accept transition to initialize state.
		i.State = state
		return
	}
	targets, ok := validInstanceTransitions[i.State]
	if !ok {
		panic(fmt.Sprintf("TransitionTo %s: unknown source state %q", i.id, i.State))
	}
	if _, valid := targets[state]; !valid {
		panic(fmt.Sprintf("TransitionTo %s: invalid transition %q → %q", i.id, i.State, state))
	}
	i.State = state
}

// AllocateTransferredKV simulates receiving transferred KV cache data from a prefill instance.
// Pre-allocates KV blocks for the request's input tokens and sets ProgressIndex past input.
// Returns false if insufficient KV capacity on this instance.
func (i *InstanceSimulator) AllocateTransferredKV(req *sim.Request) bool {
	inputLen := int64(len(req.InputTokens))
	if inputLen == 0 {
		req.ProgressIndex = 0
		return true
	}
	ok := i.sim.KVCache.AllocateKVBlocks(req, 0, inputLen, nil)
	if ok {
		req.ProgressIndex = inputLen
	}
	return ok
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
