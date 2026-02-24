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
// EventQueue implements heap.Interface and orders events by timestamp.
// See canonical Golang example here: https://pkg.go.dev/container/heap#example-package-IntHeap
type EventQueue []Event

func (eq EventQueue) Len() int           { return len(eq) }
func (eq EventQueue) Less(i, j int) bool { return eq[i].Timestamp() < eq[j].Timestamp() }
func (eq EventQueue) Swap(i, j int)      { eq[i], eq[j] = eq[j], eq[i] }

func (eq *EventQueue) Push(x any) {
	*eq = append(*eq, x.(Event))
}

func (eq *EventQueue) Pop() any {
	old := *eq
	n := len(old)
	item := old[n-1]
	*eq = old[0 : n-1]
	return item
}

type PrefillRequestConfig struct {
	ProgressIndex       int64 `json:"progress_index"`
	NumNewPrefillTokens int   `json:"num_new_prefill_tokens"`
}

type DecodeRequestConfig struct {
	ProgressIndex      int64 `json:"progress_index"`
	NumNewDecodeTokens int   `json:"num_new_decode_tokens"`
}

type StepConfig struct {
	PrefillRequests []PrefillRequestConfig `json:"prefill_requests"`
	DecodeRequests  []DecodeRequestConfig  `json:"decode_requests"`
}

// GuideLLMConfig supports GuideLLM style request generation
type GuideLLMConfig struct {
	Rate               float64 // Requests per second
	NumRequests        int     // Number of requests
	PrefixTokens       int     // Prefix Token Count
	PromptTokens       int     // Average Prompt Token Count
	PromptTokensStdDev int     // Stddev Prompt Token Count
	PromptTokensMin    int     // Min Prompt Token Count
	PromptTokensMax    int     // Max Prompt Token Count
	OutputTokens       int     // Average Output Token Count
	OutputTokensStdDev int     // Stddev Output Token Count
	OutputTokensMin    int     // Min Output Token Count
	OutputTokensMax    int     // Max Output Token Count
}

// NewGuideLLMConfig creates a GuideLLMConfig with all fields explicitly set.
// This is the canonical constructor — all construction sites must use it (R4).
// Parameter order matches struct field order: Rate, NumRequests, PrefixTokens,
// PromptTokens, PromptTokensStdDev, PromptTokensMin, PromptTokensMax,
// OutputTokens, OutputTokensStdDev, OutputTokensMin, OutputTokensMax.
func NewGuideLLMConfig(
	rate float64,
	numRequests int,
	prefixTokens int,
	promptTokens int,
	promptTokensStdDev int,
	promptTokensMin int,
	promptTokensMax int,
	outputTokens int,
	outputTokensStdDev int,
	outputTokensMin int,
	outputTokensMax int,
) *GuideLLMConfig {
	return &GuideLLMConfig{
		Rate:               rate,
		NumRequests:        numRequests,
		PrefixTokens:       prefixTokens,
		PromptTokens:       promptTokens,
		PromptTokensStdDev: promptTokensStdDev,
		PromptTokensMin:    promptTokensMin,
		PromptTokensMax:    promptTokensMax,
		OutputTokens:       outputTokens,
		OutputTokensStdDev: outputTokensStdDev,
		OutputTokensMin:    outputTokensMin,
		OutputTokensMax:    outputTokensMax,
	}
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
	guideLLMConfig       *GuideLLMConfig
	model                  string
	gpu                    string
	tracesWorkloadFilePath string
	rng                    *PartitionedRNG // partitioned RNG for deterministic multi-subsystem simulation
	priorityPolicy         PriorityPolicy
	scheduler              InstanceScheduler
	latencyModel           LatencyModel
	requestRate            float64 // arrival rate for workload generation (moved from Metrics — DES state/statistics separation, #243)
}

// NewSimulator creates a Simulator from a SimConfig struct.
// Workload mode is determined by the config fields:
//   - TracesWorkloadFilePath != "" → load workload from CSV traces
//   - GuideLLMConfig != nil → generate workload from distribution
//   - Both zero-valued → no workload (caller injects via InjectArrival)
func NewSimulator(cfg SimConfig) (*Simulator, error) {
	if cfg.TotalKVBlocks <= 0 {
		panic(fmt.Sprintf("KVCacheConfig.TotalKVBlocks must be > 0, got %d", cfg.TotalKVBlocks))
	}
	if cfg.BlockSizeTokens <= 0 {
		panic(fmt.Sprintf("KVCacheConfig.BlockSizeTokens must be > 0, got %d", cfg.BlockSizeTokens))
	}
	latencyModel, err := NewLatencyModel(cfg.LatencyCoeffs, cfg.ModelHardwareConfig)
	if err != nil {
		return nil, fmt.Errorf("creating latency model: %w", err)
	}
	batchFormation := NewBatchFormation(latencyModel)

	s := &Simulator{
		Clock:                     0,
		Horizon:                   cfg.Horizon,
		eventQueue:                make(EventQueue, 0),
		WaitQ:                     &WaitQueue{},
		KVCache:                   NewKVStore(cfg.KVCacheConfig),
		RunningBatch:              &Batch{},
		Metrics:                   NewMetrics(),
		maxRunningReqs:            cfg.MaxRunningReqs,
		maxScheduledTokens:        cfg.MaxScheduledTokens,
		longPrefillTokenThreshold: cfg.LongPrefillTokenThreshold,
		stepEvent:                 nil,
		stepCount:                 0,
		reqNumComputedTokens:      make(map[string]int64),
		batchFormation:            batchFormation,
		guideLLMConfig:            cfg.GuideLLMConfig,
		tracesWorkloadFilePath:    cfg.TracesWorkloadFilePath,
		model:                     cfg.Model,
		gpu:                       cfg.GPU,
		latencyModel:              latencyModel,
	}
	s.rng = NewPartitionedRNG(NewSimulationKey(cfg.Seed))
	s.priorityPolicy = NewPriorityPolicy(cfg.PriorityPolicy)
	s.scheduler = NewScheduler(cfg.Scheduler)

	if cfg.TracesWorkloadFilePath != "" && cfg.GuideLLMConfig == nil {
		s.requestRate = 0.0
		if err := s.generateWorkloadFromCSV(); err != nil {
			return nil, fmt.Errorf("loading CSV workload: %w", err)
		}
	} else if cfg.GuideLLMConfig != nil {
		s.requestRate = cfg.GuideLLMConfig.Rate
		s.generateWorkloadDistribution()
	}
	// else: no workload — caller injects via InjectArrival

	return s, nil
}

// WorkloadRNG returns the RNG for workload generation.
// This maintains backward compatibility with the original single-RNG implementation.
func (sim *Simulator) WorkloadRNG() *rand.Rand {
	return sim.rng.ForSubsystem(SubsystemWorkload)
}

// Pushes an event (ArrivalEvent/StepEvent) into the simulator's EventQueue.
// Note, this has nothing to do with vLLM's scheduler.schedule().
func (sim *Simulator) Schedule(ev Event) {
	heap.Push(&sim.eventQueue, ev)
}

// HasPendingEvents returns true if the EventQueue is non-empty.
func (sim *Simulator) HasPendingEvents() bool {
	return len(sim.eventQueue) > 0
}

// PeekNextEventTime returns the timestamp of the earliest pending event.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
func (sim *Simulator) PeekNextEventTime() int64 {
	return sim.eventQueue[0].Timestamp()
}

// ProcessNextEvent pops the earliest event, advances Clock, executes it, and returns it.
// The returned Event lets callers react to what happened (e.g., detect QueuedEvent for
// pending-request tracking) without maintaining fragile before/after heuristics.
// Caller MUST check HasPendingEvents() first. Panics on empty queue.
// Does NOT check horizon — caller is responsible.
func (sim *Simulator) ProcessNextEvent() Event {
	ev := heap.Pop(&sim.eventQueue).(Event)
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

// SetRequestRate sets the arrival rate for workload generation.
// Used by cluster mode to propagate the per-instance rate.
// Precondition: rate >= 0. Callers are responsible for validation (R3).
func (sim *Simulator) SetRequestRate(rate float64) { sim.requestRate = rate }

// EnqueueRequest adds a newly arrived request to the waiting queue.
// Requests whose input tokens require more KV blocks than the total cache
// capacity are dropped with a warning (R19: livelock protection). This mirrors
// real vLLM behavior where oversized requests are rejected before entering
// the engine.
func (sim *Simulator) EnqueueRequest(r *Request) {
	blocksNeeded := (int64(len(r.InputTokens)) + sim.KVCache.BlockSize() - 1) / sim.KVCache.BlockSize()
	if blocksNeeded > sim.KVCache.TotalCapacity() {
		logrus.Warnf("dropping request %s: input requires %d KV blocks but cache has only %d total",
			r.ID, blocksNeeded, sim.KVCache.TotalCapacity())
		sim.Metrics.DroppedUnservable++
		delete(sim.Metrics.Requests, r.ID)
		return
	}
	sim.WaitQ.Enqueue(r)
	sim.Metrics.TotalInputTokens += len(r.InputTokens)
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
func (sim *Simulator) recordRequestCompletion(req *Request) {
	sim.Metrics.CompletedRequests++

	var itlSum int64
	for _, v := range req.ITL {
		itlSum += v
	}
	lat := req.FirstTokenTime + itlSum
	sim.Metrics.RequestE2Es[req.ID] = float64(lat)
	logrus.Debugf("Finished req: ID: %s at time: %d", req.ID, lat+req.ArrivalTime)
	if len(req.OutputTokens) > 0 {
		reqTotalOutput := lat - req.FirstTokenTime
		// TPOT calculation in vLLM excludes the first generated token
		sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1))
	} else {
		sim.Metrics.RequestITLs[req.ID] = 0
	}
	sim.Metrics.RequestStepCounters = append(sim.Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
	sim.Metrics.RequestCompletionTimes[req.ID] = float64(lat + req.ArrivalTime)
	sim.Metrics.AllITLs = append(sim.Metrics.AllITLs, req.ITL...)
}

// Step simulates a single vllm step(): batch scheduling, model execution, and completion.
// Phases: (1) schedule batch, (2) execute prefill/decode, (3) process completions, (4) schedule next step.
func (sim *Simulator) Step(now int64) {
	sim.scheduleBatch(now)
	currStepAdvance := sim.executeBatchStep(now)
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
		Now:                   now,
		StepCount:             sim.stepCount,
		ComputedTokens:        sim.reqNumComputedTokens,
	}
	batchResult := sim.batchFormation.FormBatch(batchCtx)

	// Apply result: update running batch
	sim.RunningBatch = batchResult.RunningBatch

	// Schedule events for preempted requests and record preemption metrics
	for _, p := range batchResult.Preempted {
		sim.Schedule(&PreemptionEvent{
			time:    now + p.PreemptionDelay,
			Request: p.Request,
		})
		sim.Metrics.PreemptionCount++
	}

	// Schedule events for newly scheduled requests and record scheduling metrics
	for _, s := range batchResult.NewlyScheduled {
		sim.Schedule(&ScheduledEvent{
			time:    now + s.ScheduledDelay,
			Request: s.Request,
		})
		sim.Metrics.RequestSchedulingDelays[s.Request.ID] = now + s.ScheduledDelay - s.Request.ArrivalTime
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
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Metrics.TotalOutputTokens++
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
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
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
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
