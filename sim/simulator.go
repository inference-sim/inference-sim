// sim/simulator.go
package sim

import (
	"container/heap"
	"fmt"
	"math/rand"

	"github.com/sirupsen/logrus"

	"github.com/inference-sim/inference-sim/sim/internal/util"
)

const MaxTokenID = 128000 // Max token ID in request input/output

// eventEntry wraps an Event with a sequence ID for deterministic ordering.
// The EventQueue orders by (Timestamp, Priority, seqID), matching the
// cluster event queue's scheme. seqID breaks ties within same-type same-timestamp events.
type eventEntry struct {
	event Event
	seqID int64
}

// EventQueue implements heap.Interface and orders events by (timestamp, priority, seqID).
// This ensures deterministic same-tick event ordering (INV-6 improvement).
type EventQueue []eventEntry

func (eq EventQueue) Len() int { return len(eq) }

func (eq EventQueue) Less(i, j int) bool {
	if eq[i].event.Timestamp() != eq[j].event.Timestamp() {
		return eq[i].event.Timestamp() < eq[j].event.Timestamp()
	}
	if eq[i].event.Priority() != eq[j].event.Priority() {
		return eq[i].event.Priority() < eq[j].event.Priority()
	}
	return eq[i].seqID < eq[j].seqID
}

func (eq EventQueue) Swap(i, j int) { eq[i], eq[j] = eq[j], eq[i] }

func (eq *EventQueue) Push(x any) {
	*eq = append(*eq, x.(eventEntry))
}

func (eq *EventQueue) Pop() any {
	old := *eq
	n := len(old)
	item := old[n-1]
	*eq = old[:n-1]
	return item
}

// SimConfig holds all configuration for creating a Simulator.
// Sub-configs are embedded so fields are accessible via promotion
// (e.g., cfg.TotalKVBlocks resolves to cfg.KVCacheConfig.TotalKVBlocks).
type SimConfig struct {
	// Simulation control (no sub-config — no factory uses only these)
	Horizon int64
	Seed    int64

	// Module-scoped sub-configs (R16)
	KVCacheConfig
	BatchConfig
	LatencyCoeffs
	ModelHardwareConfig
	PolicyConfig
	WorkloadConfig
}

// Simulator is the core object that holds simulation time, system state, and the event loop.
type Simulator struct {
	Clock   int64
	Horizon int64
	// eventQueue has all the simulator events, like arrival and step events
	eventQueue EventQueue
	// WaitQ aka request waiting queue before it is scheduled
	WaitQ   *WaitQueue
	KVCache KVStore
	// Running batch contains the set of requests that go into the model for execution per Step.
	// In vLLM, running is a list (not queue) of requests, hence we don't call it RunningQ here.
	// Requests are ordered by First-Come-First-Served in WaitQ, and the same order is maintained
	// while adding requests to RunningBatch
	RunningBatch *Batch
	Metrics *Metrics
	// max number of requests RunningBatch can hold
	maxRunningReqs int64
	// max total number of new tokens across all requests in RunningBatch
	maxScheduledTokens        int64
	longPrefillTokenThreshold int64
	stepEvent                 Event
	stepCount                 int
	// map of request IDs to total num computed tokens (including cached tokens)
	reqNumComputedTokens map[string]int64
	batchFormation       BatchFormation
	model                  string
	gpu                    string
	maxModelLen            int64 // max total sequence length (0 = unlimited)
	rng                    *PartitionedRNG // partitioned RNG for deterministic multi-subsystem simulation
	priorityPolicy         PriorityPolicy
	scheduler              InstanceScheduler
	latencyModel           LatencyModel
	seqCounter             int64 // monotonic counter for event queue seqID (deterministic ordering)
	// OnRequestDone is an optional callback invoked when a request reaches a terminal
	// state (completed, length-capped, or timed out). Returns follow-up requests to inject.
	// Set by the caller (cmd/root.go or ClusterSimulator). Nil = no callback.
	OnRequestDone func(req *Request, tick int64) []*Request
}

// NewSimulator creates a Simulator from a SimConfig struct and pre-built dependencies.
// All workload generation now happens externally — callers inject requests via InjectArrival.
func NewSimulator(cfg SimConfig, kvStore KVStore, latencyModel LatencyModel) (*Simulator, error) {
	if kvStore == nil {
		return nil, fmt.Errorf("NewSimulator: kvStore must not be nil")
	}
	if latencyModel == nil {
		return nil, fmt.Errorf("NewSimulator: latencyModel must not be nil")
	}
	if cfg.MaxRunningReqs <= 0 {
		return nil, fmt.Errorf("NewSimulator: MaxRunningReqs must be > 0, got %d", cfg.MaxRunningReqs)
	}
	if cfg.MaxScheduledTokens <= 0 {
		return nil, fmt.Errorf("NewSimulator: MaxScheduledTokens must be > 0, got %d", cfg.MaxScheduledTokens)
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
	batchFormation := NewBatchFormation()

	s := &Simulator{
		Clock:                     0,
		Horizon:                   cfg.Horizon,
		eventQueue:                make(EventQueue, 0),
		WaitQ:                     &WaitQueue{},
		KVCache:                   kvStore,
		RunningBatch:              &Batch{},
		Metrics:                   NewMetrics(),
		maxRunningReqs:            cfg.MaxRunningReqs,
		maxScheduledTokens:        cfg.MaxScheduledTokens,
		longPrefillTokenThreshold: cfg.LongPrefillTokenThreshold,
		stepEvent:                 nil,
		stepCount:                 0,
		reqNumComputedTokens:      make(map[string]int64),
		batchFormation:            batchFormation,
		model:                     cfg.Model,
		gpu:                       cfg.GPU,
		maxModelLen:               cfg.MaxModelLen,
		latencyModel:              latencyModel,
	}
	s.rng = NewPartitionedRNG(NewSimulationKey(cfg.Seed))
	s.priorityPolicy = NewPriorityPolicy(cfg.PriorityPolicy)
	s.scheduler = NewScheduler(cfg.Scheduler)

	return s, nil
}

// WorkloadRNG returns the RNG for workload generation.
// This maintains backward compatibility with the original single-RNG implementation.
func (sim *Simulator) WorkloadRNG() *rand.Rand {
	return sim.rng.ForSubsystem(SubsystemWorkload)
}

// Schedule pushes an event into the simulator's EventQueue with a monotonic seqID.
// Note, this has nothing to do with vLLM's scheduler.schedule().
func (sim *Simulator) Schedule(ev Event) {
	sim.seqCounter++
	heap.Push(&sim.eventQueue, eventEntry{event: ev, seqID: sim.seqCounter})
}

// HasPendingEvents returns true if the EventQueue is non-empty.
func (sim *Simulator) HasPendingEvents() bool {
	return len(sim.eventQueue) > 0
}

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
func (sim *Simulator) PeekNextEventTime() int64 {
	return sim.eventQueue[0].event.Timestamp()
}

// ProcessNextEvent pops the earliest event, advances Clock, executes it, and returns it.
// The returned Event lets callers react to what happened (e.g., detect QueuedEvent for
// pending-request tracking) without maintaining fragile before/after heuristics.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
// Does NOT check horizon — caller is responsible.
func (sim *Simulator) ProcessNextEvent() Event {
	entry := heap.Pop(&sim.eventQueue).(eventEntry)
	ev := entry.event
	sim.Clock = ev.Timestamp()
	logrus.Debugf("[tick %07d] Executing %T", sim.Clock, ev)
	ev.Execute(sim)
	return ev
}

// Finalize records end-of-run state and sets SimEndedTime.
// Call once after the event loop ends. Called by both sim.Run() (single-instance)
// and ClusterSimulator.Run() (cluster mode via inst.Finalize()).
func (sim *Simulator) Finalize() {
	// Record conservation fields (BC-8, BC-9) — must happen in Finalize
	// because cluster mode drives events via ProcessNextEvent() directly
	// and never calls sim.Run().
	sim.Metrics.StillQueued = sim.WaitQ.Len()
	if sim.RunningBatch != nil {
		sim.Metrics.StillRunning = len(sim.RunningBatch.Requests)
	}
	sim.Metrics.SimEndedTime = min(sim.Clock, sim.Horizon)
	logrus.Infof("[tick %07d] Simulation ended", sim.Clock)
}

// InjectArrival schedules an ArrivalEvent for req and registers it in Metrics.Requests.
func (sim *Simulator) InjectArrival(req *Request) {
	if req.ArrivalTime > sim.Horizon {
		logrus.Warnf("InjectArrival: request %s has ArrivalTime %d > Horizon %d; "+
			"ArrivalEvent will not fire (INV-1 conservation may be affected)",
			req.ID, req.ArrivalTime, sim.Horizon)
	}
	sim.Schedule(&ArrivalEvent{time: req.ArrivalTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

// InjectArrivalAt schedules an ArrivalEvent at eventTime (not req.ArrivalTime).
// Metrics.Requests uses req.ArrivalTime for ArrivedAt to preserve original arrival time.
// Used by cluster-mode online routing where event time differs from original arrival.
func (sim *Simulator) InjectArrivalAt(req *Request, eventTime int64) {
	sim.Schedule(&ArrivalEvent{time: eventTime, Request: req})
	sim.Metrics.Requests[req.ID] = NewRequestMetrics(req, float64(req.ArrivalTime)/1e6)
}

func (sim *Simulator) Run() {
	for sim.HasPendingEvents() {
		sim.ProcessNextEvent()
		if sim.Clock > sim.Horizon {
			break
		}
	}
	sim.Finalize()
}

// QueueDepth returns the number of requests in the wait queue.
func (sim *Simulator) QueueDepth() int { return sim.WaitQ.Len() }

// DrainWaitQueue removes and returns all requests currently in the wait queue.
// Used by DrainRedirect policy to re-inject queued requests into the cluster router.
// After this call, WaitQ.Len() == 0.
func (sim *Simulator) DrainWaitQueue() []*Request {
	items := sim.WaitQ.Items()
	sim.WaitQ = &WaitQueue{}
	return items
}

// BatchSize returns the number of requests in the running batch, or 0 if nil.
func (sim *Simulator) BatchSize() int {
	if sim.RunningBatch == nil {
		return 0
	}
	return len(sim.RunningBatch.Requests)
}

// CurrentClock returns the current simulation clock (in ticks).
func (sim *Simulator) CurrentClock() int64 { return sim.Clock }

// SimHorizon returns the simulation horizon (in ticks).
func (sim *Simulator) SimHorizon() int64 { return sim.Horizon }

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
func (sim *Simulator) EnqueueRequest(r *Request) {
	// Guard -1: Already timed out (race: TimeoutEvent fired before QueuedEvent).
	// Request was timed out during the queueing delay (alpha overhead) before the server
	// processed the input. TotalInputTokens is NOT counted for this path.
	// INV-1 holds: request is counted in timed_out bucket.
	if r.State == StateTimedOut {
		return
	}

	// Auto-fill: if client didn't set a budget, cap at remaining context window.
	if r.MaxOutputLen == 0 && sim.maxModelLen > 0 && int64(len(r.InputTokens)) < sim.maxModelLen {
		r.MaxOutputLen = int(sim.maxModelLen) - len(r.InputTokens)
	}

	// Guard 0: Negative MaxOutputLen check (R3)
	if r.MaxOutputLen < 0 {
		logrus.Warnf("dropping request %s: MaxOutputLen %d is negative",
			r.ID, r.MaxOutputLen)
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		// Callback for dropped requests (R1: don't silently discard, BC-17)
		if sim.OnRequestDone != nil {
			for _, next := range sim.OnRequestDone(r, sim.Clock) {
				sim.InjectArrival(next)
			}
		}
		return
	}

	// Guard 1: MaxModelLen check
	if sim.maxModelLen > 0 {
		if int64(len(r.InputTokens)) >= sim.maxModelLen {
			logrus.Warnf("dropping request %s: input length %d >= MaxModelLen %d (no room for output)",
				r.ID, len(r.InputTokens), sim.maxModelLen)
			sim.Metrics.DroppedUnservable++
			delete(sim.Metrics.Requests, r.ID)
			if sim.OnRequestDone != nil {
				for _, next := range sim.OnRequestDone(r, sim.Clock) {
					sim.InjectArrival(next)
				}
			}
			return
		}
		if r.MaxOutputLen > 0 {
			totalSeqLen := int64(len(r.InputTokens)) + int64(r.MaxOutputLen)
			if totalSeqLen > sim.maxModelLen {
				logrus.Warnf("dropping request %s: total sequence length %d (input=%d + budget=%d) exceeds MaxModelLen %d",
					r.ID, totalSeqLen, len(r.InputTokens), r.MaxOutputLen, sim.maxModelLen)
				sim.Metrics.DroppedUnservable++
				delete(sim.Metrics.Requests, r.ID)
				if sim.OnRequestDone != nil {
					for _, next := range sim.OnRequestDone(r, sim.Clock) {
						sim.InjectArrival(next)
					}
				}
				return
			}
		}
	}

	// Guard 2: KV capacity check (defense-in-depth, always active)
	blocksNeeded := (int64(len(r.InputTokens)) + sim.KVCache.BlockSize() - 1) / sim.KVCache.BlockSize()
	if blocksNeeded > sim.KVCache.TotalCapacity() {
		logrus.Warnf("dropping request %s: input requires %d KV blocks but cache has only %d total",
			r.ID, blocksNeeded, sim.KVCache.TotalCapacity())
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		if sim.OnRequestDone != nil {
			for _, next := range sim.OnRequestDone(r, sim.Clock) {
				sim.InjectArrival(next)
			}
		}
		return
	}

	// Input tokens counted BEFORE past-due check (request was received)
	sim.Metrics.TotalInputTokens += len(r.InputTokens)

	// Past-due guard (EC-2): check BEFORE enqueue to avoid enqueue-then-remove.
	// Request is counted as timed_out, not dropped_unservable.
	if r.Deadline > 0 && r.Deadline <= sim.Clock {
		r.State = StateTimedOut
		sim.Metrics.TimedOutRequests++
		if sim.OnRequestDone != nil {
			for _, next := range sim.OnRequestDone(r, sim.Clock) {
				sim.InjectArrival(next)
			}
		}
		return
	}

	sim.WaitQ.Enqueue(r)

	// Schedule timeout event (after all guards + enqueue — BC-5)
	// Skip scheduling when deadline > horizon (perf: avoids orphaned events)
	if r.Deadline > 0 && r.Deadline <= sim.Horizon {
		sim.Schedule(&TimeoutEvent{time: r.Deadline, Request: r})
	}
}

// recordQueueSnapshots records the wait queue and running batch sizes at this step.
// Called after batch formation, before execution.
func (sim *Simulator) recordQueueSnapshots() {
	sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, sim.WaitQ.Len())
	sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, len(sim.RunningBatch.Requests))
}

// recordKVUsageMetrics records peak and time-weighted KV block usage.
// Called after execution, before completion processing.
func (sim *Simulator) recordKVUsageMetrics(stepDuration int64) {
	used := sim.KVCache.UsedBlocks()
	if used > sim.Metrics.PeakKVBlocksUsed {
		sim.Metrics.PeakKVBlocksUsed = used
	}
	sim.Metrics.KVBlocksUsed += float64(used) * float64(stepDuration)
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
// For trained-roofline, PostDecodeFixedOverhead adds ~1.85ms to E2E; for other backends it's 0.
func (sim *Simulator) recordRequestCompletion(req *Request) {
	// INV-1 conservation: Always increment CompletedRequests.
	// For redirected requests: the source instance drained the request from its WaitQ
	// (StillQueued=0 at end), so source contributes 0 to InjectedRequests.
	// The destination is the sole completion site. Skipping CompletedRequests++ here
	// would cause the request to vanish from conservation accounting entirely.
	sim.Metrics.CompletedRequests++

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
	sim.Metrics.RequestE2Es[req.ID] = float64(lat)
	logrus.Debugf("Finished req: ID: %s at time: %d", req.ID, lat+req.ArrivalTime)
	if len(req.OutputTokens) > 0 {
		// Compute average ITL from itlSum directly (not from lat - FirstTokenTime)
		// to avoid contaminating per-token ITL with the fixed post-decode overhead.
		reqTotalOutput := itlSum
		if req.LengthCapped {
			// #588: Use actual decode step count for length-capped requests.
			// len(req.OutputTokens) is the pre-determined count; len(req.ITL) is actual.
			// TPOT convention: exclude first generated token → denominator is len(ITL)-1.
			sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.ITL)-1, 1))
		} else {
			// TPOT calculation in vLLM excludes the first generated token.
			sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1))
		}
	} else {
		sim.Metrics.RequestITLs[req.ID] = 0
	}
	sim.Metrics.RequestStepCounters = append(sim.Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
	sim.Metrics.RequestCompletionTimes[req.ID] = float64(lat + req.ArrivalTime)
	sim.Metrics.AllITLs = append(sim.Metrics.AllITLs, req.ITL...)
}

// Step simulates a single vllm step(): batch scheduling, model execution, mirroring, and completion.
// Phases: (1) schedule batch, (2) execute prefill/decode, (2.5) mirror to CPU, (3) process completions, (4) schedule next step.
//
// Orphaned StepEvent guard: when a TimeoutEvent empties the RunningBatch and nils stepEvent,
// a previously-scheduled StepEvent may still be in the heap. If the INV-8 guard also scheduled
// a new StepEvent, two StepEvents fire at the same tick. The second finds RunningBatch nil and
// WaitQ empty (already processed by the first). This guard prevents the phantom double-step.
func (sim *Simulator) Step(now int64) {
	if sim.RunningBatch == nil && sim.WaitQ.Len() == 0 {
		return // orphaned StepEvent — nothing to process
	}
	sim.scheduleBatch(now)
	currStepAdvance := sim.executeBatchStep(now)
	// Mirror in-use blocks to CPU tier (no-op for single-tier KVCacheState).
	// Runs after execution (new full blocks exist) and before completions
	// (completing requests' blocks are still in-use and can be mirrored).
	sim.KVCache.MirrorToCPU(sim.RunningBatch.Requests)
	remaining := sim.processCompletions(now, currStepAdvance)
	sim.scheduleNextStep(now, currStepAdvance, remaining)
}

// scheduleBatch handles Phase 1: priority assignment, queue reordering, batch formation,
// and event scheduling for preemptions and newly scheduled requests.
func (sim *Simulator) scheduleBatch(now int64) {
	sim.stepCount += 1

	// Synchronize KV cache clock for thrashing detection (no-op for single-tier KVCacheState)
	sim.KVCache.SetClock(now)

	// Assign priorities to queued requests and order queue per scheduler policy
	for _, req := range sim.WaitQ.Items() {
		req.Priority = sim.priorityPolicy.Compute(req, now)
	}
	sim.WaitQ.Reorder(func(reqs []*Request) {
		sim.scheduler.OrderQueue(reqs, now)
	})

	// Delegate batch composition to the pluggable BatchFormation strategy.
	// Event scheduling and metrics recording happen after FormBatch returns (kernel concerns).
	batchCtx := BatchContext{
		RunningBatch:          sim.RunningBatch,
		WaitQ:                 sim.WaitQ,
		KVCache:               sim.KVCache,
		MaxScheduledTokens:    sim.maxScheduledTokens,
		MaxRunningReqs:        sim.maxRunningReqs,
		PrefillTokenThreshold: sim.longPrefillTokenThreshold,
		MaxModelLen:           sim.maxModelLen,
		Now:                   now,
		StepCount:             sim.stepCount,
		ComputedTokens:        sim.reqNumComputedTokens,
	}
	batchResult := sim.batchFormation.FormBatch(batchCtx)

	// Apply result: update running batch
	sim.RunningBatch = batchResult.RunningBatch

	// Record preemption metrics and emit debug log for each preempted request
	for _, p := range batchResult.Preempted {
		logrus.Debugf("<< Preemption: %s at %d ticks", p.Request.ID, now)
		sim.Metrics.PreemptionCount++
	}

	// Schedule events for newly scheduled requests and record scheduling metrics
	for _, s := range batchResult.NewlyScheduled {
		sim.Schedule(&ScheduledEvent{
			time:    now,
			Request: s.Request,
		})
		sim.Metrics.RequestSchedulingDelays[s.Request.ID] = now - s.Request.ArrivalTime
	}

	// Record queue depth observations after batch formation
	sim.recordQueueSnapshots()
}

// executeBatchStep handles Phase 2: model execution (prefill + decode) for all requests
// in the running batch. Returns the step time advance in ticks.
func (sim *Simulator) executeBatchStep(now int64) int64 {
	// Estimate step time via LatencyModel (blackbox or roofline, selected at construction)
	currStepAdvance := sim.latencyModel.StepTime(sim.RunningBatch.Requests)

	// Add transfer latency from CPU→GPU reloads (0 for single-tier)
	currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()

	// INV-3 defense-in-depth: guarantee clock advancement regardless of backend.
	// All LatencyModel implementations must return >= 1 per interface contract;
	// this floor catches violations that would cause infinite livelock.
	currStepAdvance = max(1, currStepAdvance)

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	// Note: TotalOutputTokens++ and TTFT metrics are recorded inline (not extracted to helpers)
	// because they are tightly coupled to the prefill/decode state transitions in this loop.
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex < util.Len64(req.InputTokens) {
			req.ProgressIndex = sim.reqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// Decode phase: only generate a token if FormBatch allocated one.
			// Without this guard, a request at the MaxModelLen boundary (NumNewTokens=0
			// from proactive cap) would get a phantom ProgressIndex increment.
			// Also prevents phantom tokens from token budget exhaustion (pre-existing edge case).
			if req.NumNewTokens > 0 {
				req.ProgressIndex++
				sim.Metrics.TotalOutputTokens++
				req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
			}
		}
		if req.ProgressIndex == util.Len64(req.InputTokens) { // prefill complete, first token is generated
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime() - req.ArrivalTime
			sim.Metrics.TTFTSum += req.FirstTokenTime // in microsec
			sim.Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime)
		}
	}

	// Record KV cache usage observations after execution
	sim.recordKVUsageMetrics(currStepAdvance)

	return currStepAdvance
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
func (sim *Simulator) processCompletions(now, currStepAdvance int64) []*Request {
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex == util.Len64(req.InputTokens)+max(util.Len64(req.OutputTokens), 1)-1 {
			// State transitions
			req.State = StateCompleted
			// Zero-output requests complete at prefill end with no decode phase.
			// The final-token KV allocation only applies to requests that have
			// output tokens. ITL is NOT appended here — executeBatchStep already
			// recorded it for this decode step (fix for #524 phantom ITL entry).
			if len(req.OutputTokens) > 0 {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
				if !ok {
					logrus.Errorf("[tick %07d] KV allocation failed for completing request %s (request will still complete) — this indicates a cache accounting bug", now, req.ID)
					sim.Metrics.KVAllocationFailures++
				}
			}
			// ReleaseKVBlocks is safe even when the final-token allocation failed:
			// AllocateKVBlocks only modifies RequestMap on success, so Release
			// frees exactly the blocks from prior successful allocations.
			sim.KVCache.ReleaseKVBlocks(req)
			req.FinishedStepIdx = sim.stepCount
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})

			// Record completion metrics
			sim.recordRequestCompletion(req)

			// Invoke completion callback for session management
			if sim.OnRequestDone != nil {
				for _, next := range sim.OnRequestDone(req, now+currStepAdvance) {
					sim.InjectArrival(next)
				}
			}
		} else if sim.maxModelLen > 0 && req.ProgressIndex >= sim.maxModelLen-1 {
			// BC-5: Proactive MaxModelLen cap — force-complete at boundary.
			// After the proactive cap in FormBatch prevents scheduling tokens beyond
			// maxModelLen-1, and the decode guard in executeBatchStep prevents phantom
			// ProgressIndex increments, the request reaches PI=maxModelLen-1 and needs
			// a completion path. This matches vLLM's effective behavior where the scheduler
			// cap at max_model_len-1-num_computed prevents further scheduling.
			// Note: vLLM completes length-capped requests via check_stop (num_tokens >= max_model_len)
			// which fires AFTER the model appends the generated token, producing maxModelLen-input
			// output tokens. BLIS completes at PI >= maxModelLen-1 (before the final token),
			// producing maxModelLen-1-input tokens (1 fewer). This is because BLIS lacks vLLM's
			// post-execution check_stop loop; processCompletions is the DES equivalent.
			//
			// NOTE (R23 exception): Final-token KV allocation is intentionally skipped here.
			// The normal completion path's AllocateKVBlocks for the last token is not useful
			// for a force-terminated request whose blocks are immediately released.
			logrus.Warnf("[tick %07d] force-completing request %s: ProgressIndex %d >= MaxModelLen-1 %d (length-capped)",
				now, req.ID, req.ProgressIndex, sim.maxModelLen-1)
			sim.Metrics.LengthCappedRequests++
			req.LengthCapped = true
			// Refresh Metrics.Requests: NewRequestMetrics was called at enqueue before
			// LengthCapped was known. Update so per-request JSON reflects the flag.
			if rm, ok := sim.Metrics.Requests[req.ID]; ok {
				rm.LengthCapped = true
				sim.Metrics.Requests[req.ID] = rm
			}
			req.State = StateCompleted
			sim.KVCache.ReleaseKVBlocks(req)
			req.FinishedStepIdx = sim.stepCount
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})
			sim.recordRequestCompletion(req)

			// Invoke completion callback for session management (length-capped)
			if sim.OnRequestDone != nil {
				for _, next := range sim.OnRequestDone(req, now+currStepAdvance) {
					sim.InjectArrival(next)
				}
			}
		} else {
			remaining = append(remaining, req)
		}
	}
	return remaining
}

// scheduleNextStep handles Phase 4: schedules the next step event based on
// remaining requests, or starts a new batch if only WaitQ has pending work
// (work-conserving property, INV-8).
func (sim *Simulator) scheduleNextStep(now, currStepAdvance int64, remaining []*Request) {
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
		// estimate queue overhead from LR (sim.features)
		//
		pbe := StepEvent{time: now + currStepAdvance}
		sim.Schedule(&pbe)
		sim.stepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.stepEvent = nil
		// Work-conserving: if WaitQ has pending requests, immediately
		// schedule a new step to form the next batch. Without this,
		// queued requests are stranded until the next arrival event
		// triggers a QueuedEvent — violating the work-conserving
		// property that real vLLM maintains.
		if sim.WaitQ.Len() > 0 {
			pbe := StepEvent{time: now + currStepAdvance}
			sim.Schedule(&pbe)
			sim.stepEvent = &pbe
		}
	}
}
