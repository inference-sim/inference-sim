// sim/simulator.go
package sim

import (
	"container/heap"
	"fmt"
	"math/rand"

	"github.com/sirupsen/logrus"
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
type SimConfig struct {
	Horizon                   int64
	Seed                      int64
	TotalKVBlocks             int64
	BlockSizeTokens           int64
	MaxRunningReqs            int64
	MaxScheduledTokens        int64
	LongPrefillTokenThreshold int64
	BetaCoeffs                []float64 // regression coefficients for step time (≥3 elements required)
	AlphaCoeffs               []float64 // regression coefficients for queueing time (≥3 elements required)
	ModelConfig               ModelConfig
	HWConfig                  HardwareCalib
	Model                     string
	GPU                       string
	TP                        int
	Roofline                  bool
	// Workload config (optional — nil/empty means no workload generation)
	GuideLLMConfig         *GuideLLMConfig
	TracesWorkloadFilePath string
	PriorityPolicy         string // "constant" (default) or "slo-based"
	Scheduler              string // "fcfs" (default), "priority-fcfs", "sjf"
	// Tiered KV cache configuration (PR12)
	KVCPUBlocks           int64   // CPU tier capacity (0 = single-tier, default)
	KVOffloadThreshold    float64 // GPU utilization threshold for offload (default 0.9)
	KVTransferBandwidth   float64 // blocks/tick transfer rate (default 100.0)
	KVTransferBaseLatency int64   // fixed cost per transfer (ticks, default 0)
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
	reqNumComputedTokens   map[string]int64
	preemptionHappened     bool
	guideLLMConfig         *GuideLLMConfig
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
		panic(fmt.Sprintf("SimConfig.TotalKVBlocks must be > 0, got %d", cfg.TotalKVBlocks))
	}
	if cfg.BlockSizeTokens <= 0 {
		panic(fmt.Sprintf("SimConfig.BlockSizeTokens must be > 0, got %d", cfg.BlockSizeTokens))
	}
	latencyModel, err := NewLatencyModel(cfg)
	if err != nil {
		return nil, fmt.Errorf("creating latency model: %w", err)
	}

	s := &Simulator{
		Clock:                     0,
		Horizon:                   cfg.Horizon,
		eventQueue:                make(EventQueue, 0),
		WaitQ:                     &WaitQueue{},
		KVCache:                   NewKVStore(cfg),
		RunningBatch:              &Batch{},
		Metrics:                   NewMetrics(),
		maxRunningReqs:            cfg.MaxRunningReqs,
		maxScheduledTokens:        cfg.MaxScheduledTokens,
		longPrefillTokenThreshold: cfg.LongPrefillTokenThreshold,
		stepEvent:                 nil,
		stepCount:                 0,
		reqNumComputedTokens:      make(map[string]int64),
		preemptionHappened:        false,
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

// Adds a newly arrived request to the waiting queue
func (sim *Simulator) EnqueueRequest(r *Request) {
	sim.WaitQ.Enqueue(r)
	sim.Metrics.TotalInputTokens += len(r.InputTokens)
}

func (sim *Simulator) preempt(req *Request, now int64, numNewTokens int64) bool {

	for {
		if ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// Could not allocate (e.g., no free blocks)

			// Circuit breaker: if batch is empty and allocation still fails,
			// the KV cache is too small for this request. Return false instead
			// of panicking on empty slice access. (#293, #297, R19)
			if len(sim.RunningBatch.Requests) == 0 {
				logrus.Warnf("[tick %07d] preemption: KV cache too small for request %s (need %d tokens, no running requests to evict)",
					now, req.ID, numNewTokens)
				return false
			}

			sim.preemptionHappened = true
			sim.Metrics.PreemptionCount++
			preemptionDelay := sim.latencyModel.PreemptionProcessingTime()
			preemptedRequest := sim.RunningBatch.Requests[len(sim.RunningBatch.Requests)-1]
			logrus.Warnf("[tick %07d] preemption: evicting %s to make room", now, preemptedRequest.ID)
			sim.RunningBatch.Requests = sim.RunningBatch.Requests[:len(sim.RunningBatch.Requests)-1]
			sim.Schedule(&PreemptionEvent{
				time:    now + preemptionDelay,
				Request: preemptedRequest,
			})

			preemptedRequest.State = StateQueued
			preemptedRequest.ProgressIndex = 0
			sim.KVCache.ReleaseKVBlocks(preemptedRequest)
			sim.WaitQ.PrependFront(preemptedRequest)

			if preemptedRequest == req {
				return false
			}
		} else {
			return true
		}
	}

}

func (sim *Simulator) makeRunningBatch(now int64) {
	if sim.RunningBatch == nil {
		sim.RunningBatch = &Batch{}
	}

	// clear PreemptionHappened if it doesn't exist
	sim.preemptionHappened = false

	// allocate a max token budget at the start of each Step
	tokenBudget := sim.maxScheduledTokens

	// First run requests in the RunningBatch.
	// Requests could be in either prefill or decode.
	for _, req := range sim.RunningBatch.Requests {
		if tokenBudget <= 0 {
			// Simulator has run out of token budget. Cannot run any more requests in this Step.
			// Wait for currently running requests to finish, and try again in next Step
			logrus.Warnf("[tick %07d] token budget exhausted, deferring remaining requests to next step", now)
			break
		}
		numNewTokens := Len64(req.InputTokens) - req.ProgressIndex
		// if a request is in running queue in this function and in prefill phase,
		// request must be doing chunked prefill
		// cache hits cannot happen here
		if numNewTokens > 0 {
			if 0 < sim.longPrefillTokenThreshold && sim.longPrefillTokenThreshold < numNewTokens {
				numNewTokens = sim.longPrefillTokenThreshold
			}
			numNewTokens = min(numNewTokens, tokenBudget)

			if can_schedule := sim.preempt(req, now, numNewTokens); !can_schedule {
				break
			}

			tokenBudget -= numNewTokens
			req.NumNewTokens = int(numNewTokens)
			sim.reqNumComputedTokens[req.ID] += numNewTokens

		}
		// if it is in decode phase, then allocate blocks for the token generated in the previous Step
		if req.ProgressIndex >= Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
			// Decode phase: exactly 1 new token per step. Compute explicitly
			// instead of reusing numNewTokens (which is negative during decode:
			// len(InputTokens) - ProgressIndex where ProgressIndex > len(InputTokens)).
			decodeTokens := int64(1)
			if can_schedule := sim.preempt(req, now, decodeTokens); !can_schedule {
				break
			}
			// currently each request produces 1 token per decode.
			// this needs to be updated with speculative decoding
			tokenBudget--

			req.NumNewTokens = 1
			sim.reqNumComputedTokens[req.ID] += 1
		}
	}

	// Next, attempt to dequeue requests in waiting queue, if batch size is not exceeded and not any preemption happened
	for len(sim.RunningBatch.Requests) < int(sim.maxRunningReqs) && sim.WaitQ.Len() > 0 && tokenBudget > 0 && !sim.preemptionHappened {
		// we will attempt to dequeue `next` request
		// if that attempt fails, we will break out of the loop

		next := sim.WaitQ.Peek()

		// first find cache hits. This only happens once per prefill (regardless of chunked)
		cachedBlocks := sim.KVCache.GetCachedBlocks(next.InputTokens)
		numNewTokens := Len64(next.InputTokens) - Len64(cachedBlocks)*sim.KVCache.BlockSize()

		// now check for chunked prefill
		if 0 < sim.longPrefillTokenThreshold && sim.longPrefillTokenThreshold < numNewTokens {
			numNewTokens = sim.longPrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := Len64(cachedBlocks) * sim.KVCache.BlockSize()
		endIndex := startIndex + numNewTokens

		// estimate the number of new blocks needed for the next request
		// and allocate if possible
		if ok := sim.KVCache.AllocateKVBlocks(next, startIndex, endIndex, cachedBlocks); !ok {
			// cannot allocate enough blocks for remaining tokens, do not schedule current request
			// vLLM maintains First-Come-First-Served order of requests, so we cannot move onto the
			// next request.
			break
		}

		// at this point: the `next` request is deemed schedulable

		// dequeue this request
		sim.WaitQ.DequeueBatch()
		// make it part of the running batch
		sim.RunningBatch.Requests = append(sim.RunningBatch.Requests, next)
		next.ScheduledStepIdx = sim.stepCount
		// create a scheduledevent for the request that just went into running batch
		scheduledDelay := sim.latencyModel.SchedulingProcessingTime() // ToDo: there are some minor processing time above - model it or constant?
		sim.Schedule(&ScheduledEvent{
			time:    now + scheduledDelay,
			Request: next,
		})
		// record request scheduling delay
		sim.Metrics.RequestSchedulingDelays[next.ID] = now + scheduledDelay - next.ArrivalTime

		// decrement the token budget
		tokenBudget = tokenBudget - numNewTokens
		// change the state of the request from queued to running
		next.State = StateRunning

		next.NumNewTokens = int(numNewTokens)
		sim.reqNumComputedTokens[next.ID] = numNewTokens + Len64(cachedBlocks)*sim.KVCache.BlockSize()
	}
}

// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
func (sim *Simulator) Step(now int64) {

	// increment Step counter
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

	// Subprocess: fill running batch from wait queue, similar to vLLM's scheduler.schedule()
	sim.makeRunningBatch(now)

	// save waitQ length for analysis
	sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, sim.WaitQ.Len())

	// save runningBatch length for analysis
	sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, len(sim.RunningBatch.Requests))

	// Estimate step time via LatencyModel (blackbox or roofline, selected at construction)
	currStepAdvance := sim.latencyModel.StepTime(sim.RunningBatch.Requests)

	// Add transfer latency from CPU→GPU reloads (0 for single-tier)
	currStepAdvance += sim.KVCache.ConsumePendingTransferLatency()

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex < Len64(req.InputTokens) {
			req.ProgressIndex = sim.reqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Metrics.TotalOutputTokens++
			req.ITL = append(req.ITL, currStepAdvance+sim.latencyModel.OutputTokenProcessingTime())
		}
		if req.ProgressIndex == Len64(req.InputTokens) { // prefill complete, first token is generated
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance + sim.latencyModel.OutputTokenProcessingTime() - req.ArrivalTime
			sim.Metrics.TTFTSum += req.FirstTokenTime // in microsec
			sim.Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime)
		}
	}

	// Subprocess: check completion and push next step event, similar to vLLM's
	// scheduler.update_from_output()

	// Write KVBlocks usage metrics

	if sim.KVCache.UsedBlocks() > sim.Metrics.PeakKVBlocksUsed {
		sim.Metrics.PeakKVBlocksUsed = sim.KVCache.UsedBlocks()
	}
	sim.Metrics.KVBlocksUsed += float64(sim.KVCache.UsedBlocks()) * float64(currStepAdvance)

	// IMPORTANT: This completion loop MUST run as a separate pass after the
	// prefill/decode execution loop above. For zero-output-token requests,
	// both "prefill completed" and "request completed" conditions are true
	// in the same step. The two-pass design ensures prefill metrics (TTFT)
	// are recorded before completion metrics (E2E). If these loops were ever
	// consolidated into a single pass, both branches would fire for the
	// same request in the same step.
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex == Len64(req.InputTokens)+max(Len64(req.OutputTokens), 1)-1 {
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
			sim.Metrics.CompletedRequests++
			// trigger the RequestLeftEvent with no delay
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance,
				Request: req,
			})

			// we need to add the token postprocessing time for all output tokens for E2E latency
			ITLSum := func(nums []int64) int64 {
				s := int64(0)
				for _, v := range nums {
					s += v
				}
				return s
			}(req.ITL)
			lat := req.FirstTokenTime + ITLSum
			sim.Metrics.RequestE2Es[req.ID] = float64(lat)
			logrus.Debugf("Finished req: ID: %s at time: %d", req.ID, lat+req.ArrivalTime)
			if len(req.OutputTokens) > 0 {
				reqTotalOutput := lat - req.FirstTokenTime
				// TPOT calculation in vLLM excludes the first generated token, calculated in ms
				sim.Metrics.RequestITLs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1))
			} else {
				sim.Metrics.RequestITLs[req.ID] = 0
			}
			req.FinishedStepIdx = sim.stepCount
			sim.Metrics.RequestStepCounters = append(sim.Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
			sim.Metrics.RequestCompletionTimes[req.ID] = float64(lat + req.ArrivalTime)
			sim.Metrics.AllITLs = append(sim.Metrics.AllITLs, req.ITL...)
		} else {
			remaining = append(remaining, req)
		}
	}

	// push the next step event as needed
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
