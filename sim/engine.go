// sim/engine.go: the engine half of an instance (EngineCore): every scheduling, KV-cache and
// metric decision, reaching its host only through engineHost and its report only through
// recorder (#2021 B3, PLAN-E2 E2.3).
package sim

import (
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim/internal/util"
)

const MaxTokenID = 128000 // Max token ID in request input/output

// Compile-time guard: MaxTokenID must fit in TokenID's underlying int32 range.
// If MaxTokenID is ever raised above math.MaxInt32 (~2.1B), the int→TokenID
// cast in GenerateRandomTokenIDs would silently truncate. This expression
// evaluates to a negative integer constant when MaxTokenID > MaxInt32, which
// is not representable as uint and fails to compile.
const _ = uint(math.MaxInt32 - MaxTokenID)

// EngineCore is the engine: the wait queue, the KV store, the running batch, batch formation,
// the LoRA and speculative-decoding state, and the report counters the engine updates. It
// holds no clock and no event queue (#2021 B3); what it needs from the host it gets from the
// Simulator it is embedded in until the split is finished.
type EngineCore struct {
	// host is the simulator this engine runs in (engineHost), set at construction.
	Host engineHost
	// WaitQ aka request waiting queue before it is scheduled
	WaitQ   *WaitQueue
	KVCache KVStore
	// Running batch contains the set of requests that go into the model for execution per Step.
	// In vLLM, running is a list (not queue) of requests, hence we don't call it RunningQ here.
	// Requests are ordered by First-Come-First-Served in WaitQ, and the same order is maintained
	// while adding requests to RunningBatch
	RunningBatch *Batch
	// rec is where the engine writes the report (recorder): the host's Metrics.
	Rec recorder
	// max number of requests RunningBatch can hold
	maxNumSeqs int64
	// max total number of new tokens across all requests in RunningBatch
	maxNumBatchedTokens       int64
	longPrefillTokenThreshold int64
	prefixCachingDisabled     bool
	stepCount                 int
	// lastStepEmpty is whether the last executed step scheduled no tokens.
	lastStepEmpty bool
	// lastStepPreempted is whether the last step's schedule preempted a request. Such a step
	// skipped its WAITING phase (vllm/v1/core/sched/scheduler.py:900@5df63a72), so the next
	// poll may admit what this one could not; the engine must not park on it (canPark).
	lastStepPreempted bool
	// pollMissesAtStep is the KV store's PollMisses at the start of the last step.
	pollMissesAtStep int64
	// serverSeries records the vLLM /metrics series per engine; nil when off.
	serverSeries *serverSeries
	// abortPending is set by NoteAbort until the next step publishes the engine's state.
	abortPending bool
	// map of request IDs to total num computed tokens (including cached tokens)
	reqNumComputedTokens map[string]int64
	batchFormation       BatchFormation
	ModelName            string
	gpu                  string
	maxModelLen          int64 // max total sequence length (0 = unlimited)
	// Speculative decoding / MTP (#1528). specEnabled gates ALL spec-decode behavior;
	// when false every decode path is byte-identical to a pre-feature build (INV-6).
	// specTokensPerStep is the mean accepted tokens/step (1+α·K) consumed by the
	// per-request fractional carry.
	specEnabled       bool
	specTokensPerStep float64
	// specVerifyWidth is what a speculating decode schedules a step, K+1 (#1627).
	specVerifyWidth int64
	// specDraftSlots is the drafter's extra query slots per request a step (vLLM
	// max_num_new_slots_for_drafting).
	specDraftSlots int64
	rng            *PartitionedRNG // partitioned RNG for deterministic multi-subsystem simulation
	sloMap         *SLOPriorityMap // vLLM-convention priority mapping for instance-level scheduling
	scheduler      InstanceScheduler
	latencyModel   LatencyModel
	// renderer is the endpoint's render stage (INV-R1), shared by the ranks of an internal-balancer
	// DP replica; nil when FrontendConfig is inert.
	renderer *Renderer
	// residentAdapters tracks this instance's finite resident LoRA adapter slots
	// (capacity-bounded LRU). nil when the LoRA subsystem is inert (no adapters /
	// capacity configured, or sim/lora not imported), in which case adapter handling
	// is a no-op and output is byte-identical to a pre-feature build (INV-6).
	residentAdapters ResidentAdapterSet
	// adapterCost derives the cold-load latency charged by the pre-admission gate
	// (#1466). Non-nil exactly when residentAdapters is (both wired together from
	// the same sim/lora registration); nil ⇒ no gating.
	adapterCost AdapterCost
	// loadingAdapter is the id of the adapter whose load is currently in flight on
	// this instance, or "" when none. Loads serialize per instance: the gate starts
	// a new load only when this is "" (§7 serialization).
	loadingAdapter string

	// remoteKVAdmission admits requests whose KV a remote pull will fill (a P/D decode
	// engine's handoffs), at the point of vLLM's schedule() where the WAITING phase runs: see
	// SetRemoteKVAdmission. Nil on every instance that receives no handoff.
	remoteKVAdmission func(now int64)
	// remoteKVWaiting is how many handoffs wait for this engine's KV before their pull starts
	// (SetRemoteKVWaiting): requests in vLLM's waiting queue that BLIS holds outside WaitQ.
	remoteKVWaiting int

	// engineMetrics is the state this engine publishes on its metrics endpoint.
	engineMetrics engineMetricsLog
	// kvEventSink receives the KV events of each step (SetKVEventSink); nil publishes none.
	kvEventSink     func(at int64, events []KVEvent)
	adapterCapacity int
}

// newEngineCore builds the engine from its configuration and the KV store and latency model
// it runs on; NewSimulator gives it its host and its recorder. It refuses what NewSimulator
// always refused, with the same messages.
func NewEngineCore(cfg SimConfig, kvStore KVStore, latencyModel LatencyModel) (*EngineCore, error) {
	if kvStore == nil {
		return nil, fmt.Errorf("NewSimulator: kvStore must not be nil")
	}
	if latencyModel == nil {
		return nil, fmt.Errorf("NewSimulator: latencyModel must not be nil")
	}
	if cfg.MaxNumSeqs <= 0 {
		return nil, fmt.Errorf("NewSimulator: MaxNumSeqs must be > 0, got %d", cfg.MaxNumSeqs)
	}
	if cfg.MaxNumBatchedTokens <= 0 {
		return nil, fmt.Errorf("NewSimulator: MaxNumBatchedTokens must be > 0, got %d", cfg.MaxNumBatchedTokens)
	}
	if cfg.LongPrefillTokenThreshold < 0 {
		return nil, fmt.Errorf("NewSimulator: LongPrefillTokenThreshold must be >= 0, got %d", cfg.LongPrefillTokenThreshold)
	}
	if cfg.MaxModelLen < 0 {
		return nil, fmt.Errorf("NewSimulator: MaxModelLen must be >= 0, got %d", cfg.MaxModelLen)
	}
	if cfg.MaxModelLen > 0 {
		if cfg.BlockSizeTokens <= 0 {
			return nil, fmt.Errorf("NewSimulator: BlockSizeTokens must be > 0 when MaxModelLen is set, got %d", cfg.BlockSizeTokens)
		}
		// Ceiling division for MaxModelLen → block count (R11).
		blocksForMaxLen := cfg.MaxModelLen / cfg.BlockSizeTokens
		if cfg.MaxModelLen%cfg.BlockSizeTokens != 0 {
			blocksForMaxLen++
		}
		if blocksForMaxLen > cfg.TotalKVBlocks {
			return nil, fmt.Errorf("NewSimulator: KV cache too small for MaxModelLen: need %d blocks (ceil(%d/%d)) but TotalKVBlocks=%d",
				blocksForMaxLen, cfg.MaxModelLen, cfg.BlockSizeTokens, cfg.TotalKVBlocks)
		}
	}
	batchFormation := NewBatchFormation(cfg.PreemptionPolicy)

	s := &EngineCore{
		WaitQ:                     &WaitQueue{},
		KVCache:                   kvStore,
		RunningBatch:              &Batch{},
		maxNumSeqs:                cfg.MaxNumSeqs,
		maxNumBatchedTokens:       cfg.MaxNumBatchedTokens,
		longPrefillTokenThreshold: cfg.LongPrefillTokenThreshold,
		prefixCachingDisabled:     cfg.PrefixCachingDisabled,
		stepCount:                 0,
		reqNumComputedTokens:      make(map[string]int64),
		batchFormation:            batchFormation,
		ModelName:                 cfg.Model,
		gpu:                       cfg.GPU,
		maxModelLen:               cfg.MaxModelLen,
		specEnabled:               cfg.IsEnabled(),
		specTokensPerStep:         cfg.EffectiveTokensPerStep(),
		specVerifyWidth:           int64(cfg.VerifyWidth()),
		specDraftSlots:            cfg.DraftSlots(),
		latencyModel:              latencyModel,
		sloMap:                    NewSLOPriorityMap(cfg.SLOPriorityOverrides),
	}
	if cfg.ServerSeriesIntervalUs > 0 {
		s.serverSeries = newServerSeries(cfg.ServerSeriesIntervalUs)
	}
	s.rng = NewPartitionedRNG(NewSimulationKey(cfg.Seed))
	s.scheduler = NewScheduler(cfg.Scheduler)

	if err := cfg.Frontend.Validate(); err != nil {
		return nil, fmt.Errorf("NewSimulator: %w", err)
	}
	if cfg.Frontend.Enabled() {
		r, err := NewRenderer(max(cfg.Frontend.APIServerCount, 1), cfg.Frontend.RendererNumWorkers)
		if err != nil {
			return nil, err
		}
		s.renderer = r
	}

	// Defense-in-depth: reject a non-positive adapter capacity here rather than
	// letting it reach newResidentSet as a panic. cmd/ validates via LoRAConfig.Validate,
	// but NewSimulator is library code and must return an error, not terminate, for a
	// caller that bypasses the CLI (R6). Mirrors the BatchConfig re-validation above.
	if cfg.HasAdapters() && cfg.AdapterCapacity != nil && *cfg.AdapterCapacity <= 0 {
		return nil, fmt.Errorf("NewSimulator: adapter_capacity must be > 0 when adapters are declared, got %d", *cfg.AdapterCapacity)
	}
	// Wire the per-instance resident-adapter set only when the LoRA subsystem is
	// active — adapters declared with a positive capacity — and sim/lora is linked
	// (NewResidentAdapterSetFunc registered). Otherwise it stays nil and adapter
	// handling is a no-op (INV-6).
	// Wire the resident set AND the cost model together (both from sim/lora's single
	// init, so both registration funcs are non-nil or both nil). Requiring both here
	// guarantees the invariant the gate relies on: whenever residentAdapters != nil,
	// adapterCost != nil too. If only the resident set were wired, FormBatch would
	// gate cold requests (AdapterResident predicate set) but maybeStartAdapterLoad
	// could never start a load (adapterCost nil) — stranding them. A malformed cost
	// config is a library-boundary error (R6), not a panic.
	// BuildAdapterCost centralizes the activation condition (R4) so NewSimulator and
	// the sim/cluster latency-model wrapper (WithAdapterOverhead) agree on exactly when adapter costs apply; it
	// returns (nil, nil) when the LoRA subsystem is inert (no adapters, no capacity,
	// or sim/lora unlinked). A non-nil ac therefore stands in for the full
	// HasAdapters && capacity != nil && factories-registered condition.
	ac, err := BuildAdapterCost(cfg)
	if err != nil {
		return nil, fmt.Errorf("NewSimulator: adapter cost model: %w", err)
	}
	if ac != nil {
		// Guard against a factory that returns a nil interface value: the concrete
		// sim/lora set never does (it returns a valid set or panics on bad capacity),
		// but a test double or future implementation might, and a typed-nil interface
		// would slip past the sim.residentAdapters == nil guard and panic on first use.
		rs := NewResidentAdapterSetFunc(*cfg.AdapterCapacity)
		if rs == nil {
			return nil, fmt.Errorf("NewSimulator: NewResidentAdapterSetFunc returned nil for capacity %d", *cfg.AdapterCapacity)
		}
		s.residentAdapters = rs
		s.adapterCapacity = *cfg.AdapterCapacity
		s.adapterCost = ac
	} else if cfg.HasAdapters() && cfg.AdapterCapacity == nil {
		// Adapters declared but no capacity: the resident set stays inert and every
		// adapter metric reports zero. Warn rather than fail silently (R1) — a run
		// that completes with zeroed adapter counts is otherwise indistinguishable
		// from a working one.
		logrus.Warnf("adapters declared but adapter_capacity is not set; per-instance resident-adapter tracking is disabled and adapter metrics will be zero")
	} else if cfg.HasAdapters() && cfg.AdapterCapacity != nil && (NewResidentAdapterSetFunc == nil || NewAdapterCostFunc == nil) {
		// Adapters + capacity configured but sim/lora was never linked, so the seam
		// factories are unregistered. Only reachable from a non-CLI caller (cmd/root.go
		// imports sim/lora); warn so that path does not silently drop adapter tracking.
		// This branch is not unit-testable from within package sim: the blank import in
		// lora_import_test.go always registers the factories, so there is no way to
		// observe them nil in a sim test.
		logrus.Warnf("adapters and adapter_capacity are configured but sim/lora is not linked (NewResidentAdapterSetFunc/NewAdapterCostFunc unregistered); resident-adapter tracking is disabled and adapter metrics will be zero")
	}

	return s, nil
}

// WorkloadRNG returns the RNG for workload generation.
// This maintains backward compatibility with the original single-RNG implementation.
func (sim *EngineCore) WorkloadRNG() *rand.Rand {
	return sim.rng.ForSubsystem(SubsystemWorkload)
}

// QueueDepth returns the number of requests in the wait queue.
func (sim *EngineCore) QueueDepth() int { return sim.WaitQ.Len() }

// ResidentAdapterIDs returns the ids of LoRA adapters currently resident on this
// instance, in a deterministic order (INV-6). Returns nil when the LoRA subsystem
// is inert (no adapters/capacity configured, or sim/lora not imported), so the
// snapshot's ResidentAdapters stays nil and the lora-affinity scorer is neutral
// (INV-6). Read by the cluster snapshot provider to populate
// RoutingSnapshot.ResidentAdapters (#1469).
func (sim *EngineCore) ResidentAdapterIDs() []string {
	if sim.residentAdapters == nil {
		return nil
	}
	return sim.residentAdapters.ResidentIDs()
}

// DrainWaitQueue removes and returns all requests currently in the wait queue.
// Used by DrainRedirect policy to re-inject queued requests into the cluster router.
// After this call, WaitQ.Len() == 0.
func (sim *EngineCore) DrainWaitQueue() []*Request {
	items := sim.WaitQ.Items()
	// Forget any deferral state for drained requests (H3, #1591): they leave this
	// instance's WaitQ for re-injection elsewhere, so their offload deferral must not
	// leak here. No-op when the store does not support deferral.
	for _, req := range items {
		sim.ClearDeferredKV(req.ID)
	}
	sim.WaitQ = &WaitQueue{}
	return items
}

// ClearDeferredKV forgets any step-boundary deferral state (H3, #1591) the KV store
// holds for a request leaving the WaitQ by a non-admit path (timeout, gateway
// eviction, drain-redirect), so the offload chain's deferred map does not leak. A
// no-op unless the store supports deferral (the offload chain with secondary tiers)
// and the request is tracked; idempotent.
func (sim *EngineCore) ClearDeferredKV(id string) {
	if d, ok := sim.KVCache.(DeferrableKVStore); ok {
		d.ClearDeferred(id)
	}
}

// BatchSize returns the number of requests in the running batch, or 0 if nil.
func (sim *EngineCore) BatchSize() int {
	if sim.RunningBatch == nil {
		return 0
	}
	return len(sim.RunningBatch.Requests)
}

// NoteAbort records that a request was aborted between steps (a gateway eviction). As in vLLM,
// whose scheduler keeps a finished request until its next step, the engine runs one more step at
// time now (or when the step still executing ends, #1944), so its published metrics and KV events
// show the state after the abort even when nothing is left to run. A parked engine is woken
// (the abort may have freed KV); a lockstep rank asks its group (scheduleStepAt).
func (sim *EngineCore) NoteAbort(now int64) {
	sim.abortPending = true
	if !sim.Host.stepArmed() {
		sim.Host.armStep(now)
	}
}

// ScheduleStepIfIdle schedules a batch-formation step if the instance is idle
// with pending work. Used by gateway eviction to maintain INV-8 after removing
// the last request from RunningBatch.
func (sim *EngineCore) ScheduleStepIfIdle(time int64) {
	if sim.WaitQ.Len() > 0 {
		sim.Host.wakeStep(time)
	}
}

// idleDeferralWake returns the next tick at which a deferred request can change
// state, when every waiting request is deferred. Until then an empty step admits
// nothing, so the scheduler can sleep instead of re-polling every tick.
func (sim *EngineCore) IdleDeferralWake(now int64) (int64, bool) {
	d, ok := sim.KVCache.(DeferrableKVStore)
	if !ok {
		return 0, false
	}
	for i := 0; i < sim.WaitQ.Len(); i++ {
		if !d.IsDeferred(sim.WaitQ.PeekAt(i).ID) {
			return 0, false
		}
	}
	return d.NextDeferralWake(now)
}

// PostDecodeFixedOverhead returns the latency model's fixed per-request post-decode
// overhead in microseconds. Used by the cluster layer to include overhead in
// parent.CompletionTime when disaggregated decode sub-requests complete.
func (sim *EngineCore) PostDecodeFixedOverhead() int64 {
	return sim.latencyModel.PostDecodeFixedOverhead()
}

// EnqueueRequest adds a newly arrived request to the waiting queue.
//
// Preprocessing: auto-fills MaxOutputLen when the client doesn't set a budget
// (MaxOutputLen == 0) and maxModelLen > 0. Sets MaxOutputLen = maxModelLen - len(InputTokens),
// mirroring vLLM's input_processor.py:554 (max_tokens = max_model_len - seq_len).
// Workload generators normally set MaxOutputLen = len(OutputTokens) (tight budget);
// this auto-fill is a safety net for requests that bypass generators.
//
// Three guards then prevent unservable requests from entering the queue:
//  0. MaxOutputLen validation (R3): drops requests with negative MaxOutputLen.
//  1. MaxModelLen guard (when maxModelLen > 0): validates the request fits within
//     the model's context window. First checks input >= maxModelLen (vLLM uses >=:
//     input filling the entire context leaves no room for output). Then, when
//     MaxOutputLen > 0 (client budget), checks input + budget <= maxModelLen.
//  2. KV capacity guard (defense-in-depth, always active): drops requests whose input
//     tokens alone require more KV blocks than total cache capacity (R19: livelock protection).
//
// All guards mirror real vLLM behavior where oversized requests are rejected
// before entering the engine. The control plane never peeks at len(OutputTokens) —
// respecting the oracle knowledge boundary (INV-9, #567).
func (sim *EngineCore) EnqueueRequest(r *Request) {
	// Guard -1: Already timed out (race: TimeoutEvent fired before QueuedEvent).
	// Request was timed out during the queueing delay (alpha overhead) before the server
	// processed the input. TotalInputTokens is NOT counted for this path.
	// INV-1 holds: request is counted in timed_out bucket.
	if r.State == StateTimedOut {
		return
	}

	// Auto-fill: if client didn't set a budget, cap at remaining context window.
	if r.MaxOutputLen == 0 && sim.maxModelLen > 0 && r.InputLen() < sim.maxModelLen {
		r.MaxOutputLen = int(sim.maxModelLen) - int(r.InputLen())
	}

	// Guard 0: Negative MaxOutputLen check (R3)
	if r.MaxOutputLen < 0 {
		logrus.Warnf("dropping request %s: MaxOutputLen %d is negative",
			r.ID, r.MaxOutputLen)
		sim.Rec.requestDropped(r, false)
		// Callback for dropped requests (R1: don't silently discard, BC-17)
		sim.Host.requestDone(r, sim.Host.now())
		return
	}

	// Guard 1: MaxModelLen check
	if sim.maxModelLen > 0 {
		if r.InputLen() >= sim.maxModelLen {
			logrus.Warnf("dropping request %s: input length %d >= MaxModelLen %d (no room for output)",
				r.ID, r.InputLen(), sim.maxModelLen)
			sim.Rec.requestDropped(r, true)
			sim.Host.requestDone(r, sim.Host.now())
			return
		}
		if r.MaxOutputLen > 0 {
			totalSeqLen := r.InputLen() + int64(r.MaxOutputLen)
			if totalSeqLen > sim.maxModelLen {
				logrus.Warnf("dropping request %s: total sequence length %d (input=%d + budget=%d) exceeds MaxModelLen %d",
					r.ID, totalSeqLen, r.InputLen(), r.MaxOutputLen, sim.maxModelLen)
				sim.Rec.requestDropped(r, true)
				sim.Host.requestDone(r, sim.Host.now())
				return
			}
		}
	}

	// Guard 2: KV capacity check (defense-in-depth, always active)
	blocksNeeded := (r.InputLen() + sim.KVCache.BlockSize() - 1) / sim.KVCache.BlockSize()
	if blocksNeeded > sim.KVCache.TotalCapacity() {
		logrus.Warnf("dropping request %s: input requires %d KV blocks but cache has only %d total",
			r.ID, blocksNeeded, sim.KVCache.TotalCapacity())
		sim.Rec.requestDropped(r, false)
		sim.Host.requestDone(r, sim.Host.now())
		return
	}

	// Input tokens counted BEFORE past-due check (request was received)
	sim.Rec.inputReceived(r.InputLen())

	// Past-due guard (EC-2): check BEFORE enqueue to avoid enqueue-then-remove.
	// Request is counted as timed_out, not dropped_unservable.
	if r.Deadline > 0 && r.Deadline <= sim.Host.now() {
		r.State = StateTimedOut
		sim.Rec.timedOut(r)
		sim.Host.requestDone(r, sim.Host.now())
		return
	}

	// Pre-processor: convert cluster-convention SLO priority to vLLM instance convention.
	// Mirrors the llm-d → vLLM dispatch boundary in production (lower = more urgent).
	// Overwrites any routing-hint Priority (which was always transient; see routing.go:59).
	if sim.sloMap == nil {
		// sloMap should always be set by NewSimulator; this path indicates manual struct construction.
		logrus.Warnf("Simulator.sloMap not initialized — using DefaultSLOPriorityMap; prefer NewSimulator()")
		sim.sloMap = DefaultSLOPriorityMap()
	}
	r.Priority = float64(sim.sloMap.InvertForVLLM(r.SLOClass))

	sim.WaitQ.Enqueue(r)

	// Schedule timeout event (after all guards + enqueue — BC-5)
	// Skip scheduling when deadline > horizon (perf: avoids orphaned events)
	if r.Deadline > 0 && r.Deadline <= sim.Host.horizon() {
		sim.Host.scheduleTimeout(r, r.Deadline)
	}
}

// EnqueueDecodeSubRequest enqueues a decode sub-request that already has KV blocks
// pre-allocated (via PD disaggregation transfer). Bypasses the oversized-request guard
// (blocks already allocated, guard would leak them) and does NOT increment TotalInputTokens
// (input tokens were already counted by the prefill sub-request).
// clusterTime is the cluster-level clock when this request is injected (from
// KVTransferCompletedEvent.Execute()). The StepEvent is scheduled at
// max(sim.Clock, clusterTime) to prevent the instance from processing the
// decode sub-request at a stale internal time that precedes the request's arrival.
// Triggers StepEvent if the instance is idle (INV-8: work-conserving).
func (sim *EngineCore) EnqueueDecodeSubRequest(r *Request, clusterTime int64) {
	// Pre-processor: decode sub-requests inherit SLOClass from parent (pd_events.go:215).
	// Apply the same vLLM-convention priority as EnqueueRequest.
	if sim.sloMap == nil {
		logrus.Warnf("Simulator.sloMap not initialized — using DefaultSLOPriorityMap; prefer NewSimulator()")
		sim.sloMap = DefaultSLOPriorityMap()
	}
	r.Priority = float64(sim.sloMap.InvertForVLLM(r.SLOClass))

	sim.WaitQ.Enqueue(r)
	// Do NOT add len(r.InputTokens) to TotalInputTokens — already counted by prefill sub-request.

	// Schedule timeout for decode sub-request (R23: parity with EnqueueRequest)
	if r.Deadline > 0 && r.Deadline <= sim.Host.horizon() {
		sim.Host.scheduleTimeout(r, r.Deadline)
	}

	// Trigger StepEvent if idle (work-conserving: INV-8).
	// Use max(sim.Clock, clusterTime) so the decode sub-request is not processed
	// at a stale instance time that precedes the cluster time when it was injected.
	if sim.RunningBatch == nil || len(sim.RunningBatch.Requests) == 0 {
		sim.Host.wakeStep(max(sim.Host.now(), clusterTime))
	}
}

// recordQueueSnapshots records the wait queue and running batch sizes at this step.
// Called after batch formation, before execution.
func (sim *EngineCore) recordQueueSnapshots() {
	sim.Rec.queueSnapshot(sim.WaitQ.Len(), len(sim.RunningBatch.Requests))
}

// recordKVUsageMetrics records peak and time-weighted KV block usage.
// Called after execution, before completion processing.
func (sim *EngineCore) recordKVUsageMetrics(stepDuration int64) {
	sim.Rec.kvUsage(sim.KVCache.UsedBlocks(), stepDuration, true)
}

// recordRequestCompletion records per-request metrics for a completed request.
// Called after state transitions (req.State, req.ITL, req.FinishedStepIdx)
// and KV cleanup are done.
//
// NOTE: E2E (lat) includes PostDecodeFixedOverhead and OutputTokenProcessingTime, both of
// which model non-blocking CPU overhead (concurrent with GPU execution). These inflate
// E2E and RequestCompletionTimes beyond the RequestLeftEvent timestamp by the overhead
// amount. This is architecturally intentional: real vLLM's post-processing (detokenization,
// response serialization) is non-blocking but still contributes to client-perceived latency.
func (sim *EngineCore) recordRequestCompletion(req *Request) {
	// Release this request's adapter pin (cold-load gate, #1466): a completed
	// request no longer uses its adapter, so the slot becomes evictable. Covers the
	// normal and length-capped completion paths (both funnel through here); the
	// running-request timeout path releases separately. No-op when inert (INV-6).
	sim.releaseAdapterPin(req)

	// INV-1 conservation: Always increment CompletedRequests.
	// For redirected requests: the source instance drained the request from its WaitQ
	// (StillQueued=0 at end), so source contributes 0 to InjectedRequests.
	// The destination is the sole completion site. Skipping CompletedRequests++ here
	// would cause the request to vanish from conservation accounting entirely.

	// Count output tokens at completion time (not inline per step) to avoid
	// double-counting under preemption (ProgressIndex reset to 0 on eviction).
	// PI - InputLen counts decode-step increments (= OutputLen - 1 for normal completion).
	// Add 1 for the prefill-generated first token (#1097) when the decode-step count falls
	// short of OutputLen. PD 1-output decode sub-requests are the exception: their PI_final
	// lands at InputLen+1 (one step past the InputLen threshold), so the decode-step count
	// == OutputLen already — the guard prevents double-counting in that case.
	//
	// R23 (single source of truth): this is THE output-token count for the request, and
	// every derived per-request metric below must read it rather than re-derive one. The
	// length-capped mean-ITL denominator re-derived its own from len(req.ITL) and came out
	// a token short of vLLM's maxModelLen-input (#1891).
	outputTokens := int(req.ProgressIndex) - int(req.InputLen())
	if outputTokens < len(req.OutputTokens) {
		outputTokens++ // prefill-generated first token (vLLM parity)
	}
	// Speculative decoding / MTP overshoot clamp (#1528, BC-4): DEFENSE-IN-DEPTH.
	// Since #1657, FormBatch caps each decode grant at the request's completion
	// boundary, so a multi-token step no longer carries ProgressIndex past the target
	// and this clamp is not expected to bind. It stays as a last line of defense for
	// INV-1 conservation: a request must never count MORE output tokens than it was
	// assigned, whatever a future step-sizing path does. No-op for g=1 (feature off).
	// For a LENGTH-CAPPED request this clamp provably never binds: processCompletions
	// reaches the cap branch only via its `else if` (not natural completion), so
	// ProgressIndex-InputLen < len(OutputTokens) strictly, hence outputTokens == PI-InputLen+1
	// — it can fire only on a non-capped spec-decode overshoot, which is what keeps the
	// shared `outputTokens` safe for the length-capped denominator below.
	if outputTokens > len(req.OutputTokens) {
		outputTokens = len(req.OutputTokens)
	}
	req.GeneratedTokens = max(outputTokens, 0)

	var itlSum int64
	for _, v := range req.ITL {
		itlSum += v
	}
	// PostDecodeFixedOverhead: fixed per-request overhead at completion (e.g., response setup).
	// Only applied to requests that went through a decode phase. Zero-output-token requests
	// (prefill-only) skip this overhead since they never entered the post-decode path.
	var postDecodeOverhead int64
	if len(req.OutputTokens) > 0 {
		postDecodeOverhead = sim.latencyModel.PostDecodeFixedOverhead()
	}
	lat := req.FirstTokenTime + itlSum + postDecodeOverhead
	logrus.Debugf("Finished req: ID: %s at time: %d", req.ID, lat+req.ArrivalTime)
	var itl float64 // the request's mean inter-token latency (0 without output tokens)
	if len(req.OutputTokens) > 0 {
		// Compute average ITL from itlSum directly (not from lat - FirstTokenTime)
		// to avoid contaminating per-token ITL with the fixed post-decode overhead.
		reqTotalOutput := itlSum
		if req.LengthCapped {
			// #588: len(req.OutputTokens) is the oracle count, which a length cap never
			// reaches, so the denominator comes from what the request actually emitted —
			// `outputTokens` above, minus one for the TPOT convention of excluding the
			// first generated token.
			//
			// NOT len(req.ITL) (#1891): an ITL entry is appended per decode STEP, and the
			// first output token is charged to prefill (its latency is TTFT, not an ITL
			// entry), so len(ITL) is already outputTokens-1 — fewer still under speculative
			// decoding / MTP, where one entry covers g tokens. max(len(ITL)-1, 1) therefore
			// divided by outputTokens-2, reporting TPOT as if the request had emitted one
			// token fewer than vLLM's maxModelLen-input, and disagreed with this branch's
			// former spec-decode arm, which already used the token-derived count — so a
			// feature-off (g=1) run and a k=0 spec-decode run reported different TPOT for
			// the same request. One expression now serves both: max(outputTokens-1, 1) is
			// identical to that arm's max((PI-InputLen+1)-1, 1), because a length-capped
			// request is always capped short of its oracle output and so always takes the
			// prefill-token adjustment above (outputTokens == PI-InputLen+1).
			//
			// Caveat (PD): a PD decode sub-request at the MaxModelLen boundary emits its
			// one final token even past MaxModelLen-1 (batch_formation.go floors the cap at
			// 1, matching the pre-feature PD path, which never capped), so its
			// ProgressIndex can land at MaxModelLen — one higher than a non-PD
			// length-capped request, giving outputTokens one larger. The asymmetry is
			// pre-existing and affects only the length-capped PD denominator, by one token.
			itl = float64(reqTotalOutput) / float64(max(outputTokens-1, 1))
		} else {
			// TPOT calculation in vLLM excludes the first generated token.
			// No spec-decode branch needed here: a normally-completed request generated
			// exactly len(OutputTokens) tokens, so len(OutputTokens)-1 is already the
			// correct per-token denominator regardless of how many decode STEPS it took
			// (steps < tokens under spec-decode). Only the LengthCapped branch above
			// must switch to a token-derived count, because there len(ITL)=steps≠tokens.
			itl = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1))
		}
	}
	sim.Rec.completed(Completion{
		Req:          req,
		TTFT:         req.FirstTokenTime,
		OutputTokens: max(outputTokens, 0),
		E2E:          lat,
		ITL:          itl,
		Steps:        req.FinishedStepIdx - req.ScheduledStepIdx,
		Clock:        sim.Host.now(),
		Overhead:     postDecodeOverhead,
	})
	// Terminal state: reset the spec-decode carry so no stale fraction survives if
	// this Request struct is ever reused (#1528). No-op when the feature is off.
	if req.eng != nil {
		req.eng.specDecodeCarry = 0
	}
}

// nothingToStep reports whether a step would find nothing to run: no running batch and an
// empty wait queue.
func (sim *EngineCore) NothingToStep() bool { return sim.RunningBatch == nil && sim.WaitQ.Len() == 0 }

// keepRunning sets the running batch to the requests a step left running.
func (sim *EngineCore) KeepRunning(remaining []*Request) { sim.RunningBatch.Requests = remaining }

// clearRunning drops the running batch after a step that finished every request in it.
func (sim *EngineCore) ClearRunning() { sim.RunningBatch = nil }

// adapterLoading reports whether a LoRA adapter load is in flight (the cold-load gate holds
// the wait queue's head until it completes, #1466).
func (sim *EngineCore) AdapterLoading() bool { return sim.loadingAdapter != "" }

// recordStillHeld records the requests the engine still holds at the end of the run, waiting
// and running (BC-8, BC-9): in Finalize, because cluster mode drives events through
// ProcessNextEvent and never calls Run.
func (sim *EngineCore) RecordStillHeld() {
	running := 0
	if sim.RunningBatch != nil {
		running = len(sim.RunningBatch.Requests)
	}
	sim.Rec.stillHeld(sim.WaitQ.Len(), running)
}

// idleStep is a step that found nothing to run. The step after an abort that left nothing to
// run publishes the engine's state after the abort (NoteAbort).
func (sim *EngineCore) IdleStep(now int64) {
	if sim.abortPending {
		sim.abortPending = false
		sim.publishEngineMetrics(now)
		sim.publishKVEvents(now)
	}
}

// executeStep schedules and executes one engine step at now and returns its duration.
func (sim *EngineCore) ExecuteStep(now int64) int64 {
	sim.abortPending = false
	sim.scheduleBatch(now)
	return sim.executeBatchStep(now)
}

// finishStep completes the step that started at now and lasted advance: it mirrors in-use
// blocks to the CPU tier (a no-op for single-tier KVCacheState; after execution, so new full
// blocks exist, and before completions, so completing requests' blocks are still in use),
// processes completions, records the server series' step end, and returns the requests
// still running.
func (sim *EngineCore) FinishStep(now, advance int64) []*Request {
	sim.KVCache.MirrorToCPU(sim.RunningBatch.Requests)
	remaining := sim.processCompletions(now, advance)
	sim.recordServerStepEnd(now+advance, len(remaining))
	return remaining
}

// publishStep publishes the engine's metrics and KV events at the end of a step, at.
func (sim *EngineCore) PublishStep(at int64) {
	sim.publishEngineMetrics(at)
	sim.publishKVEvents(at)
}

// scheduleBatch handles Phase 1: priority assignment, queue reordering, batch formation,
// and event scheduling for preemptions and newly scheduled requests.
func (sim *EngineCore) scheduleBatch(now int64) {
	sim.stepCount += 1
	if q, ok := sim.KVCache.(QuiescentKVStore); ok {
		sim.pollMissesAtStep = q.PollMisses()
	}

	// Synchronize KV cache clock for thrashing detection (no-op for single-tier KVCacheState)
	sim.KVCache.SetClock(now)

	// Order queue per scheduler policy. Priorities are static — set once at
	// EnqueueRequest/EnqueueDecodeSubRequest via SLOPriorityMap.InvertForVLLM
	// (vLLM static priority model; per-step recomputation removed).
	sim.WaitQ.Reorder(func(reqs []*Request) {
		sim.scheduler.OrderQueue(reqs, now)
		holdingKVFirst(reqs)
	})

	// Cold-load pre-admission gate (LoRA, #1466): if the wait-queue head is a cold
	// adapter request and no load is in flight, start a serialized per-instance
	// adapter load now (committing an eviction victim and scheduling the load
	// completion). FormBatch then holds any not-yet-resident adapter out of the
	// batch via the AdapterResident predicate below. No-op when the subsystem is
	// inert (INV-6).
	sim.maybeStartAdapterLoad(now)

	// Delegate batch composition to the pluggable BatchFormation strategy.
	// Event scheduling and metrics recording happen after FormBatch returns (kernel concerns).
	batchCtx := BatchContext{
		RunningBatch:          sim.RunningBatch,
		WaitQ:                 sim.WaitQ,
		KVCache:               sim.KVCache,
		MaxNumBatchedTokens:   sim.maxNumBatchedTokens,
		MaxNumSeqs:            sim.maxNumSeqs,
		PrefillTokenThreshold: sim.longPrefillTokenThreshold,
		PrefixCachingDisabled: sim.prefixCachingDisabled,
		MaxModelLen:           sim.maxModelLen,
		Now:                   now,
		StepCount:             sim.stepCount,
		ComputedTokens:        sim.reqNumComputedTokens,
	}
	if sim.residentAdapters != nil {
		batchCtx.AdapterResident = sim.residentAdapters.IsResident
	}
	if sim.specEnabled {
		// Pure peek: FormBatch sizes the decode step and caps g; the carry is
		// committed with the granted count in executeBatchStep (#1528).
		batchCtx.DecodeTokensPerStep = sim.peekDecodeTokens
		batchCtx.SpecVerifyWidth = sim.specVerifyWidth
		batchCtx.DraftSlots = sim.specDraftSlots
	}
	batchResult := sim.batchFormation.FormBatch(batchCtx)

	// Apply result: update running batch
	sim.RunningBatch = batchResult.RunningBatch

	// Record preemption metrics and emit debug log for each preempted request
	for _, p := range batchResult.Preempted {
		logrus.Debugf("<< Preemption: %s at %d ticks", p.Request.ID, now)
		sim.Rec.preempted()
	}
	sim.lastStepPreempted = len(batchResult.Preempted) > 0

	sim.admitRemoteKV(now, len(batchResult.Preempted) > 0)

	// Schedule events for newly scheduled requests and record scheduling metrics
	for _, s := range batchResult.NewlyScheduled {
		sim.Host.scheduleScheduled(s.Request, now)
		sim.Rec.scheduled(s.Request, now, s.Lookup)
		if sim.serverSeries != nil {
			sim.serverSeries.recordAdmission(now, s.Request, s.Lookup)
		}
		sim.recordAdapterResidency(s.Request)
	}

	// Record queue depth observations after batch formation
	sim.recordQueueSnapshots()
}

// recordAdapterResidency updates the per-instance resident-adapter set when a
// request enters the running batch. Under the cold-load gate (#1466) a request is
// admitted only once its adapter is resident (a cold request is held out of the
// batch until its load completes), so here the adapter is always resident: touch
// it to most-recently-used and pin it for the request's in-flight lifetime so it
// cannot be evicted while in use (INV-L5). The pin is taken exactly once per
// request (a preempted-then-re-admitted request keeps its existing pin) and
// released at a terminal state by releaseAdapterPin. No-op when the LoRA subsystem
// is inert or the request targets the base model, preserving byte-identity (INV-6).
func (sim *EngineCore) recordAdapterResidency(req *Request) {
	if sim.residentAdapters == nil || req.Adapter == "" {
		return
	}
	sim.residentAdapters.Touch(req.Adapter)
	if !req.AdapterPinned() {
		sim.residentAdapters.Pin(req.Adapter)
		req.state().adapterPinned = true
	}
}

// ReleaseAdapterPin releases req's adapter pin if it holds one. Exported for the
// cluster layer's gateway-eviction / drain paths (InstanceSimulator.EvictRequest),
// which remove a running request from this instance outside the normal
// completion/timeout terminal paths and must still free its pinned adapter slot —
// otherwise the pin leaks and eventually blocks all future cold loads (INV-L5,
// #1466). Idempotent and a no-op for unpinned (e.g. queued) or base-model requests.
func (sim *EngineCore) ReleaseAdapterPin(req *Request) {
	sim.releaseAdapterPin(req)
}

// releaseAdapterPin drops this request's pin on its adapter when it reaches a
// terminal state (completed, length-capped, or timed-out while running), letting
// the adapter become evictable once no in-flight request references it. Idempotent
// via the per-request adapterPinned flag; no-op when inert or base-model (INV-6).
func (sim *EngineCore) releaseAdapterPin(req *Request) {
	if sim.residentAdapters == nil || req.Adapter == "" || !req.AdapterPinned() {
		return
	}
	sim.residentAdapters.Unpin(req.Adapter)
	req.state().adapterPinned = false
}

// maybeStartAdapterLoad begins a serialized cold-adapter load when the wait-queue
// head is a new prefill request whose adapter is not yet resident (§7). It runs
// before batch formation each step. Loads serialize per instance: it starts at
// most one at a time (guarded by loadingAdapter). At load-start it commits the
// eviction victim and reserves a slot (EvictLRU when at capacity), then schedules
// an AdapterLoadCompletionEvent at now + LoadLatency; residency is committed at
// completion, so the gate keeps holding the request until then. No-op when the
// subsystem is inert (INV-6).
func (sim *EngineCore) maybeStartAdapterLoad(now int64) {
	if sim.residentAdapters == nil || sim.adapterCost == nil || sim.loadingAdapter != "" {
		return
	}
	head := sim.WaitQ.Peek()
	if head == nil || head.IsDecodeSubRequest || head.Adapter == "" || sim.residentAdapters.IsResident(head.Adapter) {
		return
	}
	// Cold head: reserve a slot by committing the LRU non-pinned victim now (§7).
	// The capacity bound itself (INV-L2, |resident| <= capacity) is structural in
	// residentSet.Store; committing here rather than at completion is what makes the
	// Store in completeAdapterLoad unable to fail, and §12's no-deadlock argument sound.
	if sim.residentAdapters.AtCapacity() {
		evicted, ok := sim.residentAdapters.EvictLRU()
		if !ok {
			// Every slot is pinned by an in-flight request; cannot start a load this
			// step. A running request will complete and unpin, and the INV-8 guard
			// will re-form a step to retry. (Guaranteed reachable: pins come from
			// running requests, which make progress.)
			return
		}
		sim.Rec.adapterEvicted(evicted)
	}
	sim.loadingAdapter = head.Adapter
	loadTicks := max(1, int64(math.Ceil(sim.adapterCost.LoadLatency(head.Adapter))))
	sim.Host.scheduleAdapterLoaded(head.Adapter, now+loadTicks)
}

// completeAdapterLoad finishes a cold-adapter load: it makes the adapter resident
// (a slot was reserved at load-start, so Store adds without further eviction),
// charges the one-time load (INV-L3 — exactly once per cold (adapter, instance)
// transition), clears the in-flight marker so the next serialized load can begin,
// and ensures a step forms so the gated request is admitted this tick (INV-8; the
// completion is ordered ahead of a co-timed step by PriorityAdapterLoad).
func (sim *EngineCore) CompleteAdapterLoad(now int64, adapter string) {
	if sim.residentAdapters == nil {
		return
	}
	// A slot was reserved at load-start (EvictLRU when at capacity), so Store adds
	// the adapter without further eviction and must succeed. A false result would
	// mean the set filled and fully pinned during the load — impossible under the
	// blocking model (no admissions occur mid-load) — so surface it loudly (R1)
	// rather than silently dropping the load accounting. Store's evicted-id return
	// is intentionally discarded: the eviction (and its AdapterEvictionCounts
	// increment) already happened at load-start, so it is always "" here; any future
	// path that calls Store at capacity would need to account for that eviction.
	if _, admitted := sim.residentAdapters.Store(adapter); admitted {
		sim.Rec.adapterLoaded(adapter) // charged once per cold transition (INV-L3)
	} else {
		logrus.Errorf("[tick %07d] adapter %q load completed but could not be made resident (set full and fully pinned) — resident-set accounting bug", now, adapter)
	}
	// Clear the in-flight marker and ensure a step forms REGARDLESS of the Store
	// outcome, so the gated request is retried and the simulator never wedges with
	// queued work and no pending step (INV-8) — even on the unreachable error path.
	sim.loadingAdapter = ""
	sim.ScheduleStepIfIdle(now)
}

// peekDecodeTokens returns this step's PROPOSED accepted-token count g for a decode
// request under speculative decoding / MTP (#1528): floor(carry + mean-rate), floored
// at 1 (the always-generated bonus token, so INV-12 NumNewTokens>0 holds). PURE — it
// does NOT mutate the carry, so FormBatch may call it while sizing the step and then
// cap g by KV budget / MaxModelLen / the distance to the completion boundary (#1657)
// without desyncing the carry (the Simulator commits the granted count afterward in
// executeBatchStep). Oracle-blind: reads only the mean rate and the request's carry,
// never OutputTokens (INV-9) — the output-budget cap is applied by the caller. Only
// wired into FormBatch when specEnabled; feature off ⇒ the caller uses the literal 1
// (byte-identical).
func (sim *EngineCore) peekDecodeTokens(req *Request) int64 {
	proposed := int64(req.specDecodeCarryOf() + sim.specTokensPerStep) // floor of accumulated
	if proposed < 1 {
		proposed = 1 // guarantee the bonus token (INV-12)
	}
	return proposed
}

// commitDecodeTokens advances the carry by the mean rate and debits the tokens
// ACTUALLY granted this step (req.NumNewTokens, after FormBatch's KV/MaxModelLen caps).
// The net carry delta is (rate − granted): a capped step keeps the ungranted fraction
// for later, so the long-run mean advance is exactly EffectiveTokensPerStep (=1+α·K)
// with no drift and no RNG (INV-6). Called once per decode step in executeBatchStep,
// only for requests that actually decoded this step (past prefill, NumNewTokens>0).
func (sim *EngineCore) commitDecodeTokens(req *Request) {
	req.state().specDecodeCarry += sim.specTokensPerStep - float64(req.NumNewTokens)
	if req.state().specDecodeCarry < 0 {
		// Defensive: granted more than proposed (no current cap raises g). Clamp to 0
		// so the carry can never fund a phantom future token.
		req.state().specDecodeCarry = 0
	}
}

// executeBatchStep handles Phase 2: model execution (prefill + decode) for all requests
// in the running batch. Returns the step time advance in ticks.
func (sim *EngineCore) executeBatchStep(now int64) int64 {
	// Match vLLM's scheduled_running_reqs: only requests that were allocated
	// tokens by FormBatch participate in the forward pass latency computation.
	// Requests with NumNewTokens=0 (past Phase 1 break point, token budget
	// exhaustion, or MaxModelLen boundary) retain their KV blocks and remain
	// in RunningBatch for the next step, but do not contribute to this step's
	// compute time. See vllm/v1/core/sched/scheduler.py scheduled_running_reqs.
	// Note: scheduled may be empty when all requests are idle (e.g., after
	// Phase 1 preemption cascade). The LatencyModel contract requires StepTime to
	// handle an empty batch (return >= 1), and the max(1, ...) floor below guarantees INV-3.
	currStepAdvance := sim.latencyModel.StepTime(sim.scheduledRequests())

	// Add transfer latency from CPU→GPU reloads (0 for single-tier)
	currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()

	// INV-3 defense-in-depth: guarantee clock advancement regardless of the latency model.
	// All LatencyModel implementations must return >= 1 per interface contract;
	// this floor catches violations that would cause infinite livelock.
	currStepAdvance = max(1, currStepAdvance)
	sim.lastStepEmpty = len(sim.scheduledRequests()) == 0
	sim.advanceBatch(now, currStepAdvance)
	return currStepAdvance
}

// scheduledRequests is the running batch's members FormBatch allocated tokens to this
// step: the requests the forward pass runs.
func (sim *EngineCore) scheduledRequests() []*Request {
	scheduled := make([]*Request, 0, len(sim.RunningBatch.Requests))
	for _, req := range sim.RunningBatch.Requests {
		if req.NumNewTokens > 0 {
			scheduled = append(scheduled, req)
		}
	}
	return scheduled
}

// advanceBatch applies one step of currStepAdvance ticks to the running batch: progress,
// ITL, TTFT and the KV usage observation.
func (sim *EngineCore) advanceBatch(now, currStepAdvance int64) {
	var emitted int64

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	// Note: Per-request TTFT fields (FirstTokenTime, RequestTTFTs) are recorded inline
	// because they are tightly coupled to the prefill/decode state transitions in this
	// loop. The !req.TTFTSet guard below fires once: a request preempted after its first token
	// keeps it (batch_formation.go), as vLLM keeps a preempted request's output, and one
	// preempted before it prefills again. TTFTSum and TotalOutputTokens are computed at
	// completion time in recordRequestCompletion to avoid double-counting when a preempted
	// request re-runs from ProgressIndex=0.
	for _, req := range sim.RunningBatch.Requests {
		if end := req.PrefillEnd(); req.ProgressIndex < end {
			req.ProgressIndex = sim.reqNumComputedTokens[req.ID]
			if req.resumeEndOf() > req.InputLen() && req.ProgressIndex == end {
				// A resumed prefill samples the request's next token (sched/scheduler.py:1084):
				// its inter-token gap runs from the last token before the preemption.
				emitted++
				emittedAt := now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime()
				last := req.ArrivalTime + req.FirstTokenTime
				for _, itl := range req.ITL {
					last += itl
				}
				req.ITL = append(req.ITL, max(emittedAt-last, 0))
			}
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// Decode phase: only generate tokens if FormBatch allocated any.
			// Without this guard, a request at the MaxModelLen boundary (NumNewTokens=0
			// from proactive cap) would get a phantom ProgressIndex increment.
			// Also prevents phantom tokens from token budget exhaustion (pre-existing edge case).
			if req.NumNewTokens > 0 {
				// Advance by the accepted-token count (1 for normal decode; g=1+accepted
				// under speculative decoding / MTP, #1528). Detokenization overhead is
				// per output token, so a step emitting g tokens pays g×OTPT. At the
				// feature-off default NumNewTokens==1 ⇒ +=1 and 1×OTPT ⇒ byte-identical.
				req.ProgressIndex += int64(req.NumNewTokens)
				emitted += int64(req.NumNewTokens)
				req.ITL = append(req.ITL, currStepAdvance+int64(req.NumNewTokens)*sim.latencyModel.OutputTokenProcessingTime())
				if sim.specEnabled {
					// Commit the carry with the GRANTED count (after FormBatch caps), so a
					// budget/MaxModelLen-capped step keeps the ungranted fraction — no drift.
					sim.commitDecodeTokens(req)
				}
			}
		}
		// !req.TTFTSet guard: fires at the first prefill completion only; a preemption after it
		// leaves the first token time as it was. TTFTSum is not accumulated here;
		// it is accumulated exactly once at completion time in recordRequestCompletion.
		if req.ProgressIndex == req.InputLen() && !req.TTFTSet {
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime() - req.ArrivalTime
			sim.Rec.firstToken(req)
			emitted++ // the prefill-completing step samples the first token
			if sim.serverSeries != nil {
				sim.serverSeries.recordFirstToken(now+currStepAdvance, req)
			}
			sim.Host.firstToken(req, req.ArrivalTime+req.FirstTokenTime)
		}
	}

	if sim.serverSeries != nil {
		sim.serverSeries.recordGeneration(now+currStepAdvance, emitted)
	}
	// Record KV cache usage observations after execution
	sim.recordKVUsageMetrics(currStepAdvance)
}

// processCompletions handles Phase 3: identifies completed requests, performs state
// transitions, releases KV blocks, and records completion metrics.
// Returns the remaining (non-completed) requests.
//
// IMPORTANT: This MUST run as a separate pass after executeBatchStep (BC-5).
// For zero-output-token requests, both "prefill completed" and "request completed"
// conditions are true in the same step. The two-pass design ensures prefill metrics
// (TTFT) are recorded before completion metrics (E2E). If these were ever
// consolidated into a single pass, both branches would fire for the same request
// in the same step.
func (sim *EngineCore) processCompletions(now, currStepAdvance int64) []*Request {
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex >= req.completionProgressIndex() {
			// State transitions
			// INV-2 (request lifecycle): the terminal running -> completed edge on
			// the normal path, taken once the request reaches its output-token target.
			req.State = StateCompleted
			// Zero-output requests complete at prefill end with no decode phase.
			// The guard below has two distinct roles depending on output length:
			//
			// 1-output-token PD decode sub-requests: FormBatch Phase 2 already
			// allocated the single decode token's KV block. After executeBatchStep
			// runs, ProgressIndex = inputLen+1, so the guard
			// (req.ProgressIndex < inputLen+outputLen) evaluates to
			// (inputLen+1) < (inputLen+1) = false — preventing a duplicate allocation.
			//
			// Requests with 2+ output tokens (PD or non-PD): after executeBatchStep
			// runs, ProgressIndex = inputLen+outputLen-1 on the final decode step,
			// so the guard evaluates to true — this is the first and only allocation
			// for the final token.
			// ITL is NOT appended here — executeBatchStep already recorded it
			// for this decode step (fix for #524 phantom ITL entry).
			if len(req.OutputTokens) > 0 && req.ProgressIndex < req.InputLen()+util.Len64(req.OutputTokens) {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
				if !ok {
					logrus.Errorf("[tick %07d] KV allocation failed for completing request %s (request will still complete) — this indicates a cache accounting bug", now, req.ID)
					sim.Rec.kvAllocationFailure()
				}
			}
			// ReleaseKVBlocks is safe even when the final-token allocation failed:
			// the decode pre-check returns false before any state mutation (check-then-act
			// pattern, matching vLLM kv_cache_manager.py:334-336), so RequestMap is
			// preserved and Release frees all blocks from prior successful allocations.
			sim.releaseFinishedKV(req)
			req.FinishedStepIdx = sim.stepCount
			req.DepartureTime = now + currStepAdvance
			sim.Host.scheduleLeft(req, req.DepartureTime)

			// Record completion metrics
			sim.recordRequestCompletion(req)

			// Invoke completion callback for session management
			sim.Host.requestDone(req, now+currStepAdvance)
		} else if sim.maxModelLen > 0 && req.ProgressIndex >= sim.maxModelLen-1 {
			// BC-5: Proactive MaxModelLen cap — force-complete at boundary.
			// After the proactive cap in FormBatch prevents scheduling tokens beyond
			// maxModelLen-1, and the decode guard in executeBatchStep prevents phantom
			// ProgressIndex increments, the request reaches PI=maxModelLen-1 and needs
			// a completion path. This matches vLLM's effective behavior where the scheduler
			// cap at max_model_len-1-num_computed prevents further scheduling.
			//
			// This predicate IS vLLM's check_stop (#1891). vLLM stops a length-capped
			// request when num_tokens >= max_model_len, evaluated after the model appends
			// the generated token, which emits max_model_len - input output tokens.
			// ProgressIndex is BLIS's num_computed_tokens, and it lags the generated-token
			// count by exactly one — BLIS charges output token #1 to prefill completion,
			// just as vLLM's num_computed lags num_tokens by the token the forward pass has
			// just appended. So PI >= maxModelLen-1 is num_tokens >= max_model_len, the
			// boundary token has been generated by the step that lands PI here (this pass
			// runs in the same Step as the executeBatchStep that advanced it), and
			// recordRequestCompletion counts maxModelLen - input output tokens (the
			// prefill-charged token is added back there, #1097). Testing PI >= maxModelLen
			// instead would emit one token too many and run the sequence past the context
			// window. TestLengthCap_VLLMParityOutputTokens pins all three quantities.
			//
			// NOTE (R23 exception): Final-token KV allocation is intentionally skipped here.
			// The normal completion path's AllocateKVBlocks for the last token is not useful
			// for a force-terminated request whose blocks are immediately released — and it
			// is unobservable either way, since recordKVUsageMetrics samples usage at the end
			// of executeBatchStep, before this pass runs. Allocating here would only risk a
			// spurious KVAllocationFailures bump under a tight cache.
			logrus.Warnf("[tick %07d] force-completing request %s: ProgressIndex %d >= MaxModelLen-1 %d (length-capped)",
				now, req.ID, req.ProgressIndex, sim.maxModelLen-1)
			req.LengthCapped = true
			sim.Rec.lengthCapped(req)
			// INV-2 (request lifecycle): the terminal running -> completed edge on
			// the length-capped path. A force-completed request is still completed,
			// not a distinct terminal state — the LengthCapped flag records why.
			// Scoped to THIS path: do not generalize to StateCompleted as a whole.
			// InstanceSimulator.EvictRequest writes StateCompleted as a tombstone
			// rather than a completion — see the note there.
			req.State = StateCompleted
			sim.releaseFinishedKV(req)
			req.FinishedStepIdx = sim.stepCount
			req.DepartureTime = now + currStepAdvance
			sim.Host.scheduleLeft(req, req.DepartureTime)
			sim.recordRequestCompletion(req)

			// Invoke completion callback for session management (length-capped)
			sim.Host.requestDone(req, now+currStepAdvance)
		} else {
			remaining = append(remaining, req)
		}
	}
	return remaining
}

// ReserveRemoteKV allocates the prompt KV of a P/D decode sub-request whose KV a NIXL pull is
// about to bring (vLLM's WAITING_FOR_REMOTE_KVS) and returns how many prompt tokens the pull
// must move. As vLLM's decode engine does, it first takes the prompt's local prefix-cache hit
// -- full blocks, capped below the prompt's last token (kv_cache_manager.py:290-299 at
// v0.31.0) -- and the connector is then asked only for the rest (sched/scheduler.py:525-537
// and 949): NIXL pulls the blocks the decode cache lacks (nixl/pull_scheduler.py:149-177), and
// the decode engine's own CPU offload tier is not consulted, nor are the blocks visible to the
// prefix cache until the pull lands (sim.RemoteKVReserver). On
// success ProgressIndex is set past the prompt so the request decodes once the pull lands.
// Returns ok=false, allocating nothing, when the blocks do not fit.
func (sim *EngineCore) ReserveRemoteKV(req *Request) (pulledTokens int64, ok bool) {
	inputLen := req.InputLen()
	if inputLen == 0 {
		req.ProgressIndex = 0
		return 0, true
	}
	bs := sim.KVCache.BlockSize()
	var cached []int64
	if !sim.prefixCachingDisabled {
		cached = sim.KVCache.GetCachedBlocks(req.FullInputTokens())
		for int64(len(cached))*bs > inputLen-1 {
			cached = cached[:len(cached)-1]
		}
		// The hybrid groups cut the hit as on any admission (kv_cache_manager.py:295-300).
		if r, ok := sim.KVCache.(PrefixHitReconciler); ok {
			cached = r.ReconcilePrefixHit(req, cached)
		}
	}
	local := int64(len(cached)) * bs
	if !sim.remoteKVReserver().AllocateRemoteKV(req, local, inputLen, cached) {
		return 0, false
	}
	req.ProgressIndex = inputLen
	return inputLen - local, true
}

// CacheRemoteKV registers a decode sub-request's pulled blocks in the prefix cache once its
// pull has landed (sim.RemoteKVReserver). Until then no other request hits on them.
func (sim *EngineCore) CacheRemoteKV(req *Request) { sim.remoteKVReserver().CacheRemoteKV(req.ID) }

func (sim *EngineCore) remoteKVReserver() RemoteKVReserver {
	r, ok := sim.KVCache.(RemoteKVReserver)
	if !ok {
		panic(fmt.Sprintf("KV store %T cannot reserve KV for a remote pull (sim.RemoteKVReserver)", sim.KVCache))
	}
	return r
}

// releaseFinishedKV frees a finished request's KV blocks, unless the request's owner keeps
// them past completion (Request.RetainKVOnFinish) and will release them itself.
func (sim *EngineCore) releaseFinishedKV(req *Request) {
	if req.RetainKVOnFinish {
		return
	}
	sim.KVCache.ReleaseKVBlocks(req)
}

// SetRemoteKVAdmission installs f as the engine's admission of remote-KV pulls (INV-PD-9).
// vLLM v0.31.0 admits a request whose KV loads asynchronously inside schedule()'s WAITING
// phase (sched/scheduler.py:872-894, 1195-1229, 1258-1287): after the running requests have taken their
// slots, only in a step that preempted nothing, behind every waiting request ahead of it, and
// only while fewer than max_num_seqs run. The engine calls f at exactly that point of each
// schedule; f allocates the pulls that fit, in arrival order, and the engine with no work at
// all admits whenever one arrives (HasWork).
func (sim *EngineCore) SetRemoteKVAdmission(f func(now int64)) { sim.remoteKVAdmission = f }

// SetRemoteKVWaiting records how many handoffs wait for this engine's KV before their pull can
// start. vLLM holds them in the decode engine's waiting queue (sched/scheduler.py:1195-1229 at
// v0.31.0), so they count in the engine's published vllm:num_requests_waiting, the series the
// router reads; BLIS queues them beside the engine (cluster decodeKVWaiters), and adds them here.
func (sim *EngineCore) SetRemoteKVWaiting(n int) { sim.remoteKVWaiting = n }

// admitRemoteKV runs the remote-KV admission at the end of a schedule, when vLLM's WAITING
// phase would reach it: no preemption this step (scheduler.py:873), the waiting queue drained
// ahead of it (the queue is FCFS, holding-KV requests first, :885-889), and a running slot
// free (max_num_active_reqs, :891-894).
func (sim *EngineCore) admitRemoteKV(now int64, preempted bool) {
	if sim.remoteKVAdmission == nil || preempted || sim.WaitQ.Len() > 0 {
		return
	}
	if sim.RunningBatch != nil && int64(len(sim.RunningBatch.Requests)) >= sim.maxNumSeqs {
		return
	}
	sim.remoteKVAdmission(now)
}

// HasWork reports whether the engine has running or waiting requests, as vLLM's
// has_unfinished_requests does. An engine without work schedules a newly arrived request at
// once.
func (sim *EngineCore) HasWork() bool { return HasWorkOf(sim) }

// holdingKVFirst moves the waiting requests that already hold their KV -- P/D decode
// sub-requests whose pull has landed (ProgressIndex set at reservation; a waiting request
// otherwise holds none) -- ahead of the rest, keeping each group's order. vLLM v0.31.0 drains
// its kv_holding_waiting queue before waiting (sched/scheduler.py:886-887). Without it a
// preempted request at the head that cannot be readmitted blocks the handoffs holding the
// blocks it needs, and the decode engine deadlocks (INV-PD-9). No-op when none holds KV.
func holdingKVFirst(reqs []*Request) {
	holds := func(r *Request) bool { return r.IsDecodeSubRequest && r.ProgressIndex > 0 }
	for _, r := range reqs {
		if holds(r) {
			sort.SliceStable(reqs, func(i, j int) bool { return holds(reqs[i]) && !holds(reqs[j]) })
			return
		}
	}
}

// FirstDecodeStepBlocks is the blocks a remote-KV request's first decode step takes beyond
// the prompt it holds: one when the prompt fills its last block, else none. A speculating
// engine's reservation already holds the step's slots (AllocateRemoteKV sizes them).
func (sim *EngineCore) FirstDecodeStepBlocks(req *Request) int64 {
	if sim.specEnabled || req.InputLen()%sim.KVCache.BlockSize() != 0 {
		return 0
	}
	return 1
}

// WaitingHolderBlocks is FirstDecodeStepBlocks summed over the waiting requests that hold
// their KV (pulls that landed and have not run): the blocks they need before they can start.
func (sim *EngineCore) WaitingHolderBlocks() int64 {
	var n int64
	for _, r := range sim.WaitQ.Items() {
		if r.IsDecodeSubRequest && r.ProgressIndex > 0 {
			n += sim.FirstDecodeStepBlocks(r)
		}
	}
	return n
}

// ReserveRemoteKVBehind is ReserveRemoteKV gated as vLLM gates an async-load admission
// (sched/scheduler.py:1195-1205, kv_cache_manager.py:570-576 at v0.31.0): the allocation must
// leave reserved blocks free -- what the other admitted remote-KV requests still need to run --
// plus the request's own first-step blocks, "to avoid deadlock and predictable preemptions".
// A refused reservation leaves no allocation behind.
func (sim *EngineCore) ReserveRemoteKVBehind(req *Request, reserved int64) (pulledTokens int64, ok bool) {
	free := func() int64 { return sim.KVCache.TotalCapacity() - sim.KVCache.UsedBlocks() }
	if free() < reserved+sim.FirstDecodeStepBlocks(req) {
		return 0, false
	}
	pulled, ok := sim.ReserveRemoteKV(req)
	if !ok {
		return 0, false
	}
	if free() < reserved+sim.FirstDecodeStepBlocks(req) {
		sim.KVCache.ReleaseKVBlocks(req)
		req.ProgressIndex = 0
		return 0, false
	}
	return pulled, true
}
