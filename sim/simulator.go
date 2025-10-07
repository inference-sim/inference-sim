// sim/simulator.go
package sim

import (
	"container/heap"

	"github.com/sirupsen/logrus"
)

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

type RegressionFeatures struct {
	TotalCacheMissTokens int `json:"num_cache_miss_tokens"`
	TotalDecodeTokens    int `json:"total_decode_tokens"`
	NumDecodeRequests    int `json:"num_decode_requests"`
	NumPrefillRequests   int `json:"num_prefill_requests"`
	TotalPrefillTokens   int `json:"total_prefill_tokens"`
	MaxPrefillTokens     int `json:"max_prefill_tokens"`
}

// Simulator is the core object that holds simulation time, system state, and the event loop.
type Simulator struct {
	Clock   int64
	Horizon int64
	// EventQueue has all the simulator events, like arrival and step events
	EventQueue EventQueue
	// WaitQ aka request waiting queue before it is scheduled
	WaitQ   *WaitQueue
	KVCache *KVCacheState
	// Running batch contains the set of requests that go into the model for execution per Step.
	// In vLLM, running is a list (not queue) of requests, hence we don't call it RunningQ here.
	// Requests are ordered by First-Come-First-Served in WaitQ, and the same order is maintained
	// while adding requests to RunningBatch
	// ToDo: Add vLLM logic for reordering requests in RunningBatch before model execution
	RunningBatch *Batch
	// ToDo: We have a data structure, but this is where we need to
	// make metrics calculations accurate
	Metrics *Metrics
	// max number of requests RunningBatch can hold
	MaxRunningReqs int64
	// max total number of new tokens across all requests in RunningBatch
	MaxScheduledTokens int
	RegressionCoeffs   []float64
	QueuingCoeffs      []float64
	FinishedCoeffs     []float64
	// RunningBatchFeatures is a map of form: {"num_decode_requests": a, "num_prefill_requests": b
	// , "total_decode_tokens": c, "total_prefill_tokens": d}
	RunningBatchFeatures      RegressionFeatures
	Requests                  []*Request
	LongPrefillTokenThreshold int
	QueuingDelay              int
	FinishedDelay             int
	StepEvent                 Event
	StepCount                 int
	// map of request IDs to total num computed tokens (including cached tokens)
	ReqNumComputedTokens map[string]int
	PreemptionHappened   bool
}

func NewSimulator(horizon int64, totalKVBlocks int, blockSizeTokens int, maxRunningReqs int64, maxScheduledTokens int, longPrefillTokenThreshold int,
	queuingDelay int, finishedDelay int, regressionCoeffs []float64, queuingCoeffs []float64, finishedCoeffs []float64, rate float64, requests []*Request) *Simulator {
	s := &Simulator{
		Clock:                     0,
		Horizon:                   horizon,
		EventQueue:                make(EventQueue, 0),
		WaitQ:                     &WaitQueue{},
		KVCache:                   NewKVCacheState(totalKVBlocks, blockSizeTokens),
		RunningBatch:              &Batch{},
		Metrics:                   NewMetrics(),
		MaxRunningReqs:            maxRunningReqs,
		MaxScheduledTokens:        maxScheduledTokens,
		RegressionCoeffs:          regressionCoeffs,
		QueuingCoeffs:             queuingCoeffs,
		FinishedCoeffs:            finishedCoeffs,
		RunningBatchFeatures:      RegressionFeatures{},
		Requests:                  requests,
		LongPrefillTokenThreshold: longPrefillTokenThreshold,
		QueuingDelay:              queuingDelay,
		FinishedDelay:             finishedDelay,
		StepEvent:                 nil,
		StepCount:                 0,
		ReqNumComputedTokens:      make(map[string]int),
		PreemptionHappened:        false,
	}

	s.Metrics.RequestRate = rate

	return s
}

// Pushes an event (ArrivalEvent/StepEvent) into the simulator's EventQueue.
// Note, this has nothing to do with vLLM's scheduler.schedule().
func (sim *Simulator) Schedule(ev Event) {
	heap.Push(&sim.EventQueue, ev)
}

func (sim *Simulator) Run() {
	for len(sim.EventQueue) > 0 {
		// get the next event to be simulated
		ev := heap.Pop(&sim.EventQueue).(Event)
		// advance the clock
		sim.Clock = ev.Timestamp()
		logrus.Infof("[tick %07d] Executing %T", sim.Clock, ev)
		// process the event
		ev.Execute(sim)
		// end the simulation if horizon is reached or if we've run out of events
		if sim.Clock > sim.Horizon {
			break
		}
	}
	sim.Metrics.SimEndedTime = min(sim.Clock, sim.Horizon)
	logrus.Infof("[tick %07d] Simulation ended", sim.Clock)
}

// Adds a newly arrived request to the waiting queue
func (sim *Simulator) EnqueueRequest(r *Request) {
	sim.WaitQ.Enqueue(r)
	sim.Metrics.TotalInputTokens += len(r.InputTokens)
}

// GeneratePoissonArrivals generates requests with arrival distributed as a Poisson process
func (sim *Simulator) GeneratePoissonArrivals(rate float64, horizon int64) {
	currentTime := int64(0)
	// keep track of how many requests in the data file have been processed
	reqIdx := 0

	// create request arrivals iteratively
	for currentTime < horizon && reqIdx < len(sim.Requests) {
		// In a Poisson process, the arrival rate is inversely proportional
		// to the mean interarrival time
		// go through the workload requests one by one
		// ToDo: create flags for max input and output lengths
		requestID := sim.Requests[reqIdx].ID
		input := sim.Requests[reqIdx].InputTokens
		output := sim.Requests[reqIdx].OutputTokens
		arrivalTime := sim.Requests[reqIdx].ArrivalTime

		currentTime = int64(arrivalTime)

		// form the request; it will be in the "queued" state when it arrives
		req := &Request{
			ID:               requestID,
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            "queued",
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}

		// push the request for arrival
		sim.Schedule(&ArrivalEvent{time: currentTime, Request: req})

		// move on to the next request
		reqIdx++

		if currentTime > horizon {
			break
		}
	}
}

// Queueing processing time estimation
func (sim *Simulator) getQueuedTime(req *Request) int64 {
	// ToDo: incorporate alpha_1 here
	var totalQueueTime float64
	totalQueueTime += sim.QueuingCoeffs[0] * float64(len(req.InputTokens))
	totalQueueTime += sim.QueuingCoeffs[1]
	return int64(totalQueueTime)
}

// Scheduling processing time estimation (step has been doing some book keeping and processing before scheduling the request)
func (sim *Simulator) getSchedulingProcessingTime() int64 {
	// ToDo: incorporate some alphas here or constant?
	return int64(0)

}

// Request Left processing time estimation
func (sim *Simulator) getRequestLeftProcessingTime(req *Request) int64 {
	// ToDo: incorporate some alphas here
	var totalFinishedTime float64
	totalFinishedTime += sim.FinishedCoeffs[0] * float64(len(req.InputTokens))
	totalFinishedTime += sim.FinishedCoeffs[1]
	return int64(totalFinishedTime) // convert from seconds to microseconds, need to verify with Satyam
}

// Request Preemption processing time estimation
func (sim *Simulator) getPreemptionProcessingTime() int64 {
	// ToDo: incorporate some alphas here or maybe constat
	return int64(0)
}

// Estimate Step Advance Time using regression features and coefficients
func (sim *Simulator) getStepTime() int64 {
	var totalStepTime float64
	totalStepTime += sim.RegressionCoeffs[0]
	totalStepTime += sim.RegressionCoeffs[1] * float64(sim.RunningBatchFeatures.TotalCacheMissTokens)
	totalStepTime += sim.RegressionCoeffs[2] * float64(sim.RunningBatchFeatures.TotalDecodeTokens)
	return int64(totalStepTime * 1e6) // convert from seconds to microseconds, need to verify with Satyam
}

func (sim *Simulator) preempt(req *Request, now int64, numNewTokens int) bool {

	for {
		if ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int{}); !ok {
			// ToDo: add while true here, because we will keep preempting until we are good
			// Could not allocate (e.g., no free blocks)
			logrus.Warnf("[Preemption]")
			sim.PreemptionHappened = true
			preemptionDelay := sim.getPreemptionProcessingTime() // model it or constant?
			preemptedRequest := sim.RunningBatch.Requests[len(sim.RunningBatch.Requests)-1]
			sim.RunningBatch.Requests = sim.RunningBatch.Requests[:len(sim.RunningBatch.Requests)-1]
			sim.Schedule(&PreemptionEvent{
				time:    now + preemptionDelay,
				Request: preemptedRequest,
			})

			preemptedRequest.State = "queued"
			preemptedRequest.ProgressIndex = 0
			sim.KVCache.ReleaseKVBlocks(preemptedRequest)
			sim.WaitQ.queue = append([]*Request{preemptedRequest}, sim.WaitQ.queue...)

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

	// allocate a max token budget at the start of each Step
	tokenBudget := sim.MaxScheduledTokens

	// First run requests in the RunningBatch.
	// Requests could be in either prefill or decode.
	for _, req := range sim.RunningBatch.Requests {
		if tokenBudget <= 0 {
			// Simulator has run out of token budget. Cannot run any more requests in this Step.
			// Wait for currently running requests to finish, and try again in next Step
			logrus.Warnf("Simulator has run out of token budget. Trying in next step.")
			break
		}
		numNewTokens := len(req.InputTokens) - req.ProgressIndex
		// if a request is in running queue in this function and in prefill phase,
		// request must be doing chunked prefill
		// cache hits cannot happen here
		if numNewTokens > 0 {
			if 0 < sim.LongPrefillTokenThreshold && sim.LongPrefillTokenThreshold < numNewTokens {
				numNewTokens = sim.LongPrefillTokenThreshold
			}
			numNewTokens = min(numNewTokens, tokenBudget)

			if can_schedule := sim.preempt(req, now, numNewTokens); !can_schedule {
				break
			}

			tokenBudget -= numNewTokens
			sim.RunningBatchFeatures.TotalCacheMissTokens += numNewTokens
			sim.RunningBatchFeatures.NumPrefillRequests += 1
			sim.RunningBatchFeatures.TotalPrefillTokens += numNewTokens
			sim.RunningBatchFeatures.MaxPrefillTokens = max(sim.RunningBatchFeatures.MaxPrefillTokens, numNewTokens)
			sim.ReqNumComputedTokens[req.ID] += numNewTokens

		}
		// if it is in decode phase, then allocate blocks for the token generated in the previous Step
		if req.ProgressIndex >= len(req.InputTokens) && len(req.OutputTokens) > 0 {
			// this request will go through decode phase in this batch
			if can_schedule := sim.preempt(req, now, numNewTokens); !can_schedule {
				break
			}
			// currently each request produces 1 token per decode.
			// this needs to be updated with speculative decoding
			tokenBudget--

			// update decode-related features in RunningBatchFeatures
			sim.RunningBatchFeatures.NumDecodeRequests += 1
			sim.RunningBatchFeatures.TotalDecodeTokens += 1
			sim.ReqNumComputedTokens[req.ID] += 1
		}
	}

	// Next, attempt to dequeue requests in waiting queue, if batch size is not exceeded and not any preemption happened
	for len(sim.RunningBatch.Requests) < int(sim.MaxRunningReqs) && len(sim.WaitQ.queue) > 0 && tokenBudget > 0 && !sim.PreemptionHappened {
		// we will attempt to dequeue `next` request
		// if that attempt fails, we will break out of the loop

		next := sim.WaitQ.queue[0]

		// first find cache hits. This only happens once per prefill (regardless of chunked)
		cachedBlocks := sim.KVCache.GetCachedBlocks(next.InputTokens)
		numNewTokens := len(next.InputTokens) - len(cachedBlocks)*sim.KVCache.BlockSizeTokens

		// now check for chunked prefill
		if 0 < sim.LongPrefillTokenThreshold && sim.LongPrefillTokenThreshold < numNewTokens {
			numNewTokens = sim.LongPrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := len(cachedBlocks) * sim.KVCache.BlockSizeTokens
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
		sim.WaitQ.queue = sim.WaitQ.queue[1:]
		// make it part of the running batch
		sim.RunningBatch.Requests = append(sim.RunningBatch.Requests, next)
		next.ScheduledStepIdx = sim.StepCount
		// create a scheduledevent for the request that just went into running batch
		scheduledDelay := sim.getSchedulingProcessingTime() // ToDo: there are some minor processing time above - model it or constant?
		sim.Schedule(&ScheduledEvent{
			time:    now + scheduledDelay,
			Request: next,
		})

		// decrement the token budget
		tokenBudget = tokenBudget - numNewTokens
		// change the state of the request from queued to running
		next.State = "running"

		// update prefill-related features in RunningBatchFeatures
		sim.RunningBatchFeatures.NumPrefillRequests += 1
		sim.RunningBatchFeatures.TotalPrefillTokens += numNewTokens
		sim.RunningBatchFeatures.TotalCacheMissTokens += numNewTokens
		sim.RunningBatchFeatures.MaxPrefillTokens = max(sim.RunningBatchFeatures.MaxPrefillTokens, numNewTokens)
		sim.ReqNumComputedTokens[next.ID] = numNewTokens + len(cachedBlocks)*sim.KVCache.BlockSizeTokens
	}
}

// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
// ToDo: Understand and handle pre-emption logic, if need be.
func (sim *Simulator) Step(now int64) {

	// increment Step counter
	sim.StepCount += 1

	// refreshing RunningBatchFeatures for current Step
	sim.RunningBatchFeatures = RegressionFeatures{
		TotalDecodeTokens:    0,
		TotalCacheMissTokens: 0,
	}
	// Subprocess: fill running batch from wait queue, similar to vLLM's scheduler.schedule()
	sim.makeRunningBatch(now)

	// save waitQ length for analysis
	sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, len(sim.WaitQ.queue))

	// save runningBatch length for analysis
	sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, len(sim.RunningBatch.Requests))

	// Estimate regression times based on runningBatch state
	currStepAdvance := sim.getStepTime()

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex < len(req.InputTokens) {
			req.ProgressIndex = sim.ReqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Metrics.TotalOutputTokens++
		}
		if req.ProgressIndex == len(req.InputTokens) { // prefill complete, first token is generated
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance - req.ArrivalTime
			sim.Metrics.TTFTSum += req.FirstTokenTime                             // in microsec
			sim.Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime) / 1000 // in ms
		}
	}

	// Subprocess: check completion and push next step event, similar to vLLM's
	// scheduler.update_from_output()

	// Write KVBlocks usage metrics

	if sim.KVCache.UsedBlockCnt > sim.Metrics.PeakKVBlocksUsed {
		sim.Metrics.PeakKVBlocksUsed = sim.KVCache.UsedBlockCnt
	}
	sim.Metrics.KVBlocksUsed += sim.KVCache.UsedBlockCnt * int(currStepAdvance)

	// handle completed and remaining requests
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex == len(req.InputTokens)+max(len(req.OutputTokens), 1)-1 {
			req.State = "completed"
			if len(req.OutputTokens) > 0 {
				ok := sim.KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int{})
				if !ok {
					// Could not allocate (e.g., no free blocks)
					logrus.Warnf("[THIS SHOULD NEVER HAPPEN]")
					continue
				}
			}
			sim.KVCache.ReleaseKVBlocks(req)
			sim.Metrics.CompletedRequests++
			// trigger the RequestLeftEvent
			requestLeftDelay := sim.getRequestLeftProcessingTime(req) // alpha params to estimate
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance + requestLeftDelay,
				Request: req,
			})

			lat := now + currStepAdvance - req.ArrivalTime
			sim.Metrics.RequestE2Es[req.ID] = float64(lat) / 1000 // in ms
			logrus.Infof("Finished req: ID: %s at time: %d\n", req.ID, lat+req.ArrivalTime)
			sim.Metrics.TotalLatency += lat
			if len(req.OutputTokens) > 0 {
				reqTotalOutput := lat - req.FirstTokenTime
				sim.Metrics.TPOTSum += reqTotalOutput // in microsec
				// TPOT calculation in vLLM excludes the first generated token, calculated in ms
				sim.Metrics.RequestTPOTs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1)) / 1000
			} else {
				sim.Metrics.RequestTPOTs[req.ID] = 0
			}
			req.FinishedStepIdx = sim.StepCount
			sim.Metrics.RequestStepCounters = append(sim.Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
			sim.Metrics.RequestCompletionTimes[req.ID] = float64(lat+req.ArrivalTime) / 1e6 // in seconds
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
		sim.StepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.StepEvent = nil
	}
}
