// Package cluster provides multi-replica cluster simulation capabilities.
//
// This package wraps the single-instance simulator (sim.Simulator) to enable
// multi-replica coordination via ClusterSimulator.
package cluster

import (
	"fmt"

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
	State            sim.InstanceState // lifecycle state; empty = untracked (backward-compat)
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
		id:  id,
		sim: s,
		gpu: cfg.GPU,
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

// TotalKVBlocks returns the total number of KV cache blocks for this instance.
func (i *InstanceSimulator) TotalKVBlocks() int64 {
	if i.sim == nil || i.sim.KVCache == nil {
		return 0
	}
	return i.sim.KVCache.TotalCapacity()
}

// PreemptionCount returns the cumulative number of preemption events on this instance.
func (i *InstanceSimulator) PreemptionCount() int64 {
	return i.sim.Metrics.PreemptionCount
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
var validInstanceTransitions = map[sim.InstanceState]map[sim.InstanceState]struct{}{
	sim.InstanceStateScheduling: {sim.InstanceStateLoading: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateLoading:    {sim.InstanceStateWarmingUp: {}, sim.InstanceStateActive: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateWarmingUp:  {sim.InstanceStateActive: {}, sim.InstanceStateDraining: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateActive:     {sim.InstanceStateDraining: {}, sim.InstanceStateTerminated: {}},
	sim.InstanceStateDraining:   {sim.InstanceStateTerminated: {}},
	sim.InstanceStateTerminated: {},
}

// TransitionTo validates and applies an instance state transition.
// Panics on invalid transition (invariant violation per Principle V).
// No-op when State is empty (backward-compat: lifecycle not tracked).
func (i *InstanceSimulator) TransitionTo(state sim.InstanceState) {
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
