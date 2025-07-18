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
	NumDecodeRequests  int `json:"num_decode_requests"`
	NumPrefillRequests int `json:"num_prefill_requests"`
	TotalDecodeTokens  int `json:"total_decode_tokens"`
	TotalPrefillTokens int `json:"total_prefill_tokens"`
	MaxPrefillTokens   int `json:"max_prefill_tokens"`
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
	// regression coefficients for execute_model time prediction
	RegressionCoeffs []float64
	// RunningBatchFeatures is a map of form: {"num_decode_requests": a, "num_prefill_requests": b
	// , "total_decode_tokens": c, "total_prefill_tokens": d}
	RunningBatchFeatures RegressionFeatures
	Requests             []*Request
	ScheduleTime         int64
	UpdateTime           int64
	QueueOverheadTime    int64
	VLLMOverheadTime     int64
	StepEvent            Event
}

func NewSimulator(horizon int64, totalKVBlocks int, blockSizeTokens int, maxRunningReqs int64, maxScheduledTokens int,
	regressionCoeffs []float64, rate float64, requests []*Request, scheduleTime int64, updateTime int64, queueOverheadTime int64, vLLMOverheadTime int64) *Simulator {
	s := &Simulator{
		Clock:                0,
		Horizon:              horizon,
		EventQueue:           make(EventQueue, 0),
		WaitQ:                &WaitQueue{},
		KVCache:              NewKVCacheState(totalKVBlocks, blockSizeTokens),
		RunningBatch:         &Batch{},
		Metrics:              &Metrics{RequestTTFTs: []float64{}, RequestTPOTs: []float64{}, RequestE2Es: []float64{}, NumWaitQRequests: []int{}, NumRunningBatchRequests: []int{}},
		MaxRunningReqs:       maxRunningReqs,
		MaxScheduledTokens:   maxScheduledTokens,
		RegressionCoeffs:     regressionCoeffs,
		RunningBatchFeatures: RegressionFeatures{},
		Requests:             requests,
		ScheduleTime:         scheduleTime,
		UpdateTime:           updateTime,
		QueueOverheadTime:    queueOverheadTime,
		VLLMOverheadTime:     vLLMOverheadTime,
		StepEvent:            nil,
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
		arrivalDelta := sim.Requests[reqIdx].ArrivalDelta

		currentTime += int64(arrivalDelta)

		// form the request; it will be in the "queued" state when it arrives
		req := &Request{
			ID:           requestID,
			ArrivalTime:  currentTime,
			ArrivalDelta: arrivalDelta,
			InputTokens:  input,
			OutputTokens: output,
			State:        "queued",
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

// Estimate Step Advance Time using regression features and coefficients
func (sim *Simulator) getStepTime() int64 {
	var totalStepTime float64
	totalStepTime += sim.RegressionCoeffs[0] * float64(sim.RunningBatchFeatures.TotalDecodeTokens)
	totalStepTime += sim.RegressionCoeffs[1] * float64(sim.RunningBatchFeatures.TotalPrefillTokens)
	totalStepTime += sim.RegressionCoeffs[2] * float64(sim.RunningBatchFeatures.MaxPrefillTokens)
	totalStepTime += sim.RegressionCoeffs[3] * float64(sim.RunningBatchFeatures.NumPrefillRequests)
	totalStepTime += sim.RegressionCoeffs[4] * float64(sim.RunningBatchFeatures.TotalDecodeTokens*sim.RunningBatchFeatures.TotalDecodeTokens)
	totalStepTime += sim.RegressionCoeffs[5] * float64(sim.RunningBatchFeatures.TotalDecodeTokens*sim.RunningBatchFeatures.TotalPrefillTokens)
	totalStepTime += sim.RegressionCoeffs[6] * float64(sim.RunningBatchFeatures.TotalDecodeTokens*int(sim.RunningBatchFeatures.MaxPrefillTokens))
	totalStepTime += sim.RegressionCoeffs[7] * float64(sim.RunningBatchFeatures.TotalDecodeTokens*sim.RunningBatchFeatures.NumPrefillRequests)
	totalStepTime += sim.RegressionCoeffs[8] * float64(sim.RunningBatchFeatures.TotalPrefillTokens*sim.RunningBatchFeatures.TotalPrefillTokens)
	totalStepTime += sim.RegressionCoeffs[9] * float64(sim.RunningBatchFeatures.TotalPrefillTokens*int(sim.RunningBatchFeatures.MaxPrefillTokens))
	totalStepTime += sim.RegressionCoeffs[10] * float64(sim.RunningBatchFeatures.TotalPrefillTokens*int(sim.RunningBatchFeatures.NumPrefillRequests))
	totalStepTime += sim.RegressionCoeffs[11] * float64(int(sim.RunningBatchFeatures.MaxPrefillTokens)*int(sim.RunningBatchFeatures.MaxPrefillTokens))
	totalStepTime += sim.RegressionCoeffs[12] * float64(int(sim.RunningBatchFeatures.MaxPrefillTokens)*int(sim.RunningBatchFeatures.NumPrefillRequests))
	totalStepTime += sim.RegressionCoeffs[13] * float64(sim.RunningBatchFeatures.NumPrefillRequests*int(sim.RunningBatchFeatures.NumPrefillRequests))
	totalStepTime += (sim.RegressionCoeffs[14]) // intercept
	return int64(totalStepTime * 1e6)           // convert from seconds to microseconds, need to verify with Satyam
}

func (sim *Simulator) makeRunningBatch() {
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
		// if a request is in running queue in this function and in prefill phase, then nothing left to do,
		// blocks have already been allocated. if it is in decode phase, then allocate blocks for the
		// token generated in the previous Step
		if req.ProgressIndex >= len(req.InputTokens) && len(req.OutputTokens) > 0 {
			// this request will go through decode phase in this batch
			ok := sim.KVCache.AllocateKVBlocksDecode(req)
			if !ok {
				// Could not allocate (e.g., no free blocks)
				logrus.Warnf("[Preemption]")
				continue // ToDo: pre-empt this request
			}
			// currently each request produces 1 token per decode.
			// this needs to be updated with speculative decoding
			tokenBudget--

			// update decode-related features in RunningBatchFeatures
			sim.RunningBatchFeatures.NumDecodeRequests += 1
			sim.RunningBatchFeatures.TotalDecodeTokens += 1
		}
	}

	// Next, attempt to dequeue requests in waiting queue, if batch size is not exceeded
	for len(sim.RunningBatch.Requests) < int(sim.MaxRunningReqs) && len(sim.WaitQ.queue) > 0 && tokenBudget > 0 {
		// we will attempt to dequeue `next` request
		// if that attempt fails, we will break out of the loop

		next := sim.WaitQ.queue[0]

		cachedBlocks := sim.KVCache.GetCachedBlocks(next.InputTokens)
		numRemainingTokens := len(next.InputTokens) - len(cachedBlocks)*sim.KVCache.BlockSizeTokens

		// estimate the number of new blocks needed for the next request
		// and allocate if possible
		if ok := sim.KVCache.AllocateKVBlocksPrefill(next); !ok {
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
		// decrement the token budget
		tokenBudget = tokenBudget - numRemainingTokens
		// change the state of the request from queued to running
		next.State = "running"

		// update prefill-related features in RunningBatchFeatures
		sim.RunningBatchFeatures.NumPrefillRequests += 1
		sim.RunningBatchFeatures.TotalPrefillTokens += int(numRemainingTokens)
		sim.RunningBatchFeatures.MaxPrefillTokens = max(sim.RunningBatchFeatures.MaxPrefillTokens, numRemainingTokens)
	}
}

// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
// ToDo: Understand and handle pre-emption logic, if need be.
func (sim *Simulator) Step(now int64) {

	// refreshing RunningBatchFeatures for current Step
	sim.RunningBatchFeatures = RegressionFeatures{
		NumDecodeRequests:  0,
		NumPrefillRequests: 0,
		TotalDecodeTokens:  0,
		TotalPrefillTokens: 0,
		MaxPrefillTokens:   0,
	}
	// Subprocess: fill running batch from wait queue, similar to vLLM's scheduler.schedule()
	sim.makeRunningBatch()

	// save waitQ length for analysis
	sim.Metrics.NumWaitQRequests = append(sim.Metrics.NumWaitQRequests, len(sim.WaitQ.queue))

	// save runningBatch length for analysis
	sim.Metrics.NumRunningBatchRequests = append(sim.Metrics.NumRunningBatchRequests, len(sim.RunningBatch.Requests))

	// Estimate regression times based on runningBatch state
	currStepAdvance := sim.getStepTime()

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex == 0 {
			req.ProgressIndex = len(req.InputTokens)
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance - req.ArrivalTime + sim.VLLMOverheadTime + sim.ScheduleTime + sim.UpdateTime
			sim.Metrics.TTFTSum += req.FirstTokenTime
			sim.Metrics.RequestTTFTs = append(sim.Metrics.RequestTTFTs, float64(req.FirstTokenTime)/1000)
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else if req.ProgressIndex >= len(req.InputTokens) {
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Metrics.TotalOutputTokens++
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
				ok := sim.KVCache.AllocateKVBlocksDecode(req)
				if !ok {
					// Could not allocate (e.g., no free blocks)
					logrus.Warnf("[Preemption]")
					continue // ToDo: pre-empt this request
				}
			}
			sim.KVCache.ReleaseKVBlocks(req)
			sim.Metrics.CompletedRequests++
			lat := now + currStepAdvance - req.ArrivalTime + sim.VLLMOverheadTime + sim.ScheduleTime + sim.UpdateTime
			sim.Metrics.RequestE2Es = append(sim.Metrics.RequestE2Es, float64(lat)/1000)
			logrus.Infof("Finished req: ID: %s at time: %d\n", req.ID, now+currStepAdvance)
			sim.Metrics.TotalLatency += lat
			if len(req.OutputTokens) > 0 {
				reqTotalOutput := lat - req.FirstTokenTime + sim.VLLMOverheadTime
				sim.Metrics.TPOTSum += reqTotalOutput
				// TPOT calculation in vLLM excludes the first generated token
				sim.Metrics.RequestTPOTs = append(sim.Metrics.RequestTPOTs, float64(reqTotalOutput)/float64(max(len(req.OutputTokens)-1, 1))/1000)
			}
		} else {
			remaining = append(remaining, req)
		}
	}

	// push the next step event as needed
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
		pbe := StepEvent{time: now + currStepAdvance + sim.QueueOverheadTime + sim.ScheduleTime + sim.UpdateTime}
		sim.Schedule(&pbe)
		sim.StepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.StepEvent = nil
	}
}
