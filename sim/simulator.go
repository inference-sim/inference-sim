// sim/simulator.go
package sim

import (
	"container/heap"
	"fmt"
	"math"
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

type RegressionFeatures struct {
	TotalCacheMissTokens int64 `json:"num_cache_miss_tokens"`
	TotalDecodeTokens    int64 `json:"total_decode_tokens"`
	NumDecodeRequests    int64 `json:"num_decode_requests"`
	NumPrefillRequests   int64 `json:"num_prefill_requests"`
	TotalPrefillTokens   int64 `json:"total_prefill_tokens"`
	MaxPrefillTokens     int64 `json:"max_prefill_tokens"`
}

type Replica struct {
	// WaitQ aka request waiting queue before it is scheduled
	WaitQ   *WaitQueue
	KVCache *KVCacheState
	// Running batch contains the set of requests that go into the model for execution per Step.
	// In vLLM, running is a list (not queue) of requests, hence we don't call it RunningQ here.
	// Requests are ordered by First-Come-First-Served in WaitQ, and the same order is maintained
	// while adding requests to RunningBatch
	// ToDo: Add vLLM logic for reordering requests in RunningBatch before model execution
	RunningBatch *Batch
	// RunningBatchFeatures is a map of form: {"num_decode_requests": a, "num_prefill_requests": b
	// , "total_decode_tokens": c, "total_prefill_tokens": d}
	RunningBatchFeatures RegressionFeatures
	// Metrics for this replica
	// ToDo: We have a data structure, but this is where we need to
	// make metrics calculations accurate
	Metrics *Metrics
	// Local vllm event indicator
	StepEvent Event
}

// Simulator is the core object that holds simulation time, system state, and the event loop.
type Simulator struct {
	Clock   int64
	Horizon int64
	// Load balancer
	LoadBalancer LoadBalancer
	// Num Replicas
	NumReplicas int
	// Replicas are the states of each replica
	Replicas []Replica
	// Global metrics
	// ToDo: We have a data structure, but this is where we need to
	// make metrics calculations accurate
	GlobalMetrics *Metrics
	// EventQueue has all the simulator events, like arrival and step events
	EventQueue EventQueue
	// max number of requests RunningBatch can hold
	MaxRunningReqs int64
	// max total number of new tokens across all requests in RunningBatch
	MaxScheduledTokens        int64
	RegressionCoeffs          []float64
	QueuingCoeffs             []float64
	FinishedCoeffs            []float64
	LongPrefillTokenThreshold int64
	QueuingDelay              int
	FinishedDelay             int
	StepCount                 int
	// map of request IDs to total num computed tokens (including cached tokens)
	ReqNumComputedTokens  map[string]int64
	PreemptionHappened    bool
	RequestGenConfig      *RequestGenConfig
	randomNumberGenerator *rand.Rand // random number generator for request tokens
}

func NewSimulator(horizon int64, totalKVBlocks int64, blockSizeTokens int64, maxRunningReqs int64, maxScheduledTokens int64, longPrefillTokenThreshold int64,
	queuingDelay int, finishedDelay int, regressionCoeffs []float64, queuingCoeffs []float64, finishedCoeffs []float64, requestGenConfig *RequestGenConfig, numReplicas int, loadBalancerType string) *Simulator {
	s := &Simulator{
		Clock:                     0,
		Horizon:                   horizon,
		EventQueue:                make(EventQueue, 0),
		GlobalMetrics:             NewMetrics(),
		NumReplicas:               numReplicas,
		MaxRunningReqs:            maxRunningReqs,
		MaxScheduledTokens:        maxScheduledTokens,
		RegressionCoeffs:          regressionCoeffs,
		QueuingCoeffs:             queuingCoeffs,
		FinishedCoeffs:            finishedCoeffs,
		LongPrefillTokenThreshold: longPrefillTokenThreshold,
		QueuingDelay:              queuingDelay,
		FinishedDelay:             finishedDelay,
		StepCount:                 0,
		ReqNumComputedTokens:      make(map[string]int64),
		PreemptionHappened:        false,
		RequestGenConfig:          requestGenConfig,
	}

	switch loadBalancerType {
	case "random":
		s.LoadBalancer = NewRandomLoadBalancer(s.NumReplicas, 0)
	default:
		logrus.Panic("unknown load balancer type")
	}

	replicas := make([]Replica, s.NumReplicas)
	for i := range replicas {
		replicas[i].WaitQ = &WaitQueue{}
		replicas[i].KVCache = NewKVCacheState(totalKVBlocks, blockSizeTokens)
		replicas[i].RunningBatch = &Batch{}
		replicas[i].StepEvent = nil
		replicas[i].Metrics = NewMetrics()
		replicas[i].RunningBatchFeatures = RegressionFeatures{}
	}

	s.Replicas = replicas

	s.GlobalMetrics.RequestRate = requestGenConfig.GuideLLMConfig.RateConfig.Rate

	src := rand.NewSource(requestGenConfig.Seed)
	s.randomNumberGenerator = rand.New(src)
	s.generateRequestArrivals()

	return s
}

// generateLengthGauss generates input or output length satisfying DataConfig distribution
// The generated length is sampled from a Gaussian distribution with mean=lengthMean, std=lengthStd
// and is clamped between (lengthMin, lengthMax)
func (sim *Simulator) generateLengthGauss(lengthMean, lengthStd, lengthMin, lengthMax int) int {
	if lengthMin == lengthMax {
		return lengthMin
	}
	val := sim.randomNumberGenerator.NormFloat64()*float64(lengthStd) + float64(lengthMean)
	clampedVal := math.Min(float64(lengthMax), val)
	clampedVal = math.Max(float64(lengthMin), clampedVal)
	roundedVal := math.Round(clampedVal)
	return int(roundedVal)
}

// generateRandomTokenIDs creates a slice of 'length' random integers.
// each token ID ranges between 0 to 32000.
func (sim *Simulator) generateRandomTokenIDs(length int) []int {

	tokens := make([]int, length)

	for i := 0; i < length; i++ {
		tokens[i] = sim.randomNumberGenerator.Intn(MaxTokenID)
	}
	return tokens
}

// GenerateRequestArrivals generates request arrivals according to gen config
func (sim *Simulator) generateRequestArrivals() {

	currentTime := int64(0)
	// keep track of how many requests have been generated
	reqIdx := 0

	// generate prefix here; this is a random sequence of tokens of prefix len
	prefix := sim.generateRandomTokenIDs(sim.RequestGenConfig.GuideLLMConfig.DataConfig.PrefixTokens)

	// create request arrivals iteratively
	for currentTime < sim.Horizon && reqIdx < sim.RequestGenConfig.GuideLLMConfig.RateConfig.MaxPrompts {
		// In a Poisson process, the arrival rate is inversely proportional
		// to the mean interarrival time
		// go through the workload requests one by one
		// ToDo: create flags for max input and output lengths

		// get input token length given DataConfig distribution
		promptLen := sim.generateLengthGauss(sim.RequestGenConfig.GuideLLMConfig.DataConfig.PromptTokens, sim.RequestGenConfig.GuideLLMConfig.DataConfig.PromptTokensStdDev, sim.RequestGenConfig.GuideLLMConfig.DataConfig.PromptTokensMin, sim.RequestGenConfig.GuideLLMConfig.DataConfig.PromptTokensMax)
		// generate random input tokens of above promptLen
		prompt := sim.generateRandomTokenIDs(promptLen)
		// combine prefix and prompt
		input := append(prefix, prompt...)

		// get output token len given DataConfig distribution
		outputLen := sim.generateLengthGauss(sim.RequestGenConfig.GuideLLMConfig.DataConfig.OutputTokens, sim.RequestGenConfig.GuideLLMConfig.DataConfig.OutputTokensStdDev, sim.RequestGenConfig.GuideLLMConfig.DataConfig.OutputTokensMin, sim.RequestGenConfig.GuideLLMConfig.DataConfig.OutputTokensMax)
		// generate random output tokens of above outputLen
		output := sim.generateRandomTokenIDs(outputLen)

		// form the request; it will be in the "queued" state when it arrives
		req := &Request{
			ID:               fmt.Sprintf("request_%v", reqIdx),
			ArrivalTime:      currentTime,
			InputTokens:      input,
			OutputTokens:     output,
			State:            "queued",
			ScheduledStepIdx: 0,
			FinishedStepIdx:  0,
		}

		// push the request for arrival
		sim.Schedule(&ArrivalEvent{time: currentTime, Request: req})

		// estimate arrivalTime based on constant RPS
		currentTime += int64(1 / sim.GlobalMetrics.RequestRate)

		// move on to the next request
		reqIdx++

		if currentTime > sim.Horizon {
			break
		}
	}

}

// Pushes an event (ArrivalEvent/StepEvent) into the simulator's EventQueue.
// Note, this has nothing to do with vLLM's scheduler.schedule().
func (sim *Simulator) Schedule(ev Event) {
	heap.Push(&sim.EventQueue, ev) // ToDo: discuss if we need event queues per replica
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
	sim.GlobalMetrics.SimEndedTime = min(sim.Clock, sim.Horizon)
	logrus.Infof("[tick %07d] Simulation ended", sim.Clock)
}

// Adds a newly arrived request to the waiting queue
func (sim *Simulator) EnqueueRequest(r *Request, rIndex int) {
	sim.Replicas[rIndex].WaitQ.Enqueue(r)
	sim.Replicas[rIndex].Metrics.TotalInputTokens += len(r.InputTokens)
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
func (sim *Simulator) getStepTime(rIndex int) int64 {
	var totalStepTime float64
	totalStepTime += sim.RegressionCoeffs[0]
	totalStepTime += sim.RegressionCoeffs[1] * float64(sim.Replicas[rIndex].RunningBatchFeatures.TotalCacheMissTokens)
	totalStepTime += sim.RegressionCoeffs[2] * float64(sim.Replicas[rIndex].RunningBatchFeatures.TotalDecodeTokens)
	return int64(totalStepTime * 1e6) // convert from seconds to microseconds, need to verify with Satyam
}

func (sim *Simulator) preempt(req *Request, now int64, numNewTokens int64, rIndex int) bool {

	for {
		if ok := sim.Replicas[rIndex].KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+numNewTokens, []int64{}); !ok {
			// ToDo: add while true here, because we will keep preempting until we are good
			// Could not allocate (e.g., no free blocks)
			logrus.Warnf("[Preemption]")
			sim.PreemptionHappened = true
			preemptionDelay := sim.getPreemptionProcessingTime() // model it or constant?
			preemptedRequest := sim.Replicas[rIndex].RunningBatch.Requests[len(sim.Replicas[rIndex].RunningBatch.Requests)-1]
			sim.Replicas[rIndex].RunningBatch.Requests = sim.Replicas[rIndex].RunningBatch.Requests[:len(sim.Replicas[rIndex].RunningBatch.Requests)-1]
			sim.Schedule(&PreemptionEvent{
				time:    now + preemptionDelay,
				Request: preemptedRequest,
			})

			preemptedRequest.State = "queued"
			preemptedRequest.ProgressIndex = 0
			sim.Replicas[rIndex].KVCache.ReleaseKVBlocks(preemptedRequest)
			sim.Replicas[rIndex].WaitQ.queue = append([]*Request{preemptedRequest}, sim.Replicas[rIndex].WaitQ.queue...)

			if preemptedRequest == req {
				return false
			}
		} else {
			return true
		}
	}

}

func (sim *Simulator) makeRunningBatch(now int64, rIndex int) {
	if sim.Replicas[rIndex].RunningBatch == nil {
		sim.Replicas[rIndex].RunningBatch = &Batch{}
	}

	// allocate a max token budget at the start of each Step
	tokenBudget := sim.MaxScheduledTokens

	// First run requests in the RunningBatch.
	// Requests could be in either prefill or decode.
	for _, req := range sim.Replicas[rIndex].RunningBatch.Requests {
		if tokenBudget <= 0 {
			// Simulator has run out of token budget. Cannot run any more requests in this Step.
			// Wait for currently running requests to finish, and try again in next Step
			logrus.Warnf("Simulator has run out of token budget. Trying in next step.")
			break
		}
		numNewTokens := Len64(req.InputTokens) - req.ProgressIndex
		// if a request is in running queue in this function and in prefill phase,
		// request must be doing chunked prefill
		// cache hits cannot happen here
		if numNewTokens > 0 {
			if 0 < sim.LongPrefillTokenThreshold && sim.LongPrefillTokenThreshold < numNewTokens {
				numNewTokens = sim.LongPrefillTokenThreshold
			}
			numNewTokens = min(numNewTokens, tokenBudget)

			if can_schedule := sim.preempt(req, now, numNewTokens, rIndex); !can_schedule {
				break
			}

			tokenBudget -= numNewTokens
			sim.Replicas[rIndex].RunningBatchFeatures.TotalCacheMissTokens += numNewTokens
			sim.Replicas[rIndex].RunningBatchFeatures.NumPrefillRequests += 1
			sim.Replicas[rIndex].RunningBatchFeatures.TotalPrefillTokens += numNewTokens
			sim.Replicas[rIndex].RunningBatchFeatures.MaxPrefillTokens = max(sim.Replicas[rIndex].RunningBatchFeatures.MaxPrefillTokens, numNewTokens)
			sim.ReqNumComputedTokens[req.ID] += numNewTokens

		}
		// if it is in decode phase, then allocate blocks for the token generated in the previous Step
		if req.ProgressIndex >= Len64(req.InputTokens) && len(req.OutputTokens) > 0 {
			// this request will go through decode phase in this batch
			if can_schedule := sim.preempt(req, now, numNewTokens, rIndex); !can_schedule {
				break
			}
			// currently each request produces 1 token per decode.
			// this needs to be updated with speculative decoding
			tokenBudget--

			// update decode-related features in RunningBatchFeatures
			sim.Replicas[rIndex].RunningBatchFeatures.NumDecodeRequests += 1
			sim.Replicas[rIndex].RunningBatchFeatures.TotalDecodeTokens += 1
			sim.ReqNumComputedTokens[req.ID] += 1
		}
	}

	// Next, attempt to dequeue requests in waiting queue, if batch size is not exceeded and not any preemption happened
	for len(sim.Replicas[rIndex].RunningBatch.Requests) < int(sim.MaxRunningReqs) && len(sim.Replicas[rIndex].WaitQ.queue) > 0 && tokenBudget > 0 && !sim.PreemptionHappened {
		// we will attempt to dequeue `next` request
		// if that attempt fails, we will break out of the loop

		next := sim.Replicas[rIndex].WaitQ.queue[0]

		// first find cache hits. This only happens once per prefill (regardless of chunked)
		cachedBlocks := sim.Replicas[rIndex].KVCache.GetCachedBlocks(next.InputTokens)
		numNewTokens := Len64(next.InputTokens) - Len64(cachedBlocks)*sim.Replicas[rIndex].KVCache.BlockSizeTokens

		// now check for chunked prefill
		if 0 < sim.LongPrefillTokenThreshold && sim.LongPrefillTokenThreshold < numNewTokens {
			numNewTokens = sim.LongPrefillTokenThreshold
		}
		numNewTokens = min(numNewTokens, tokenBudget)
		startIndex := Len64(cachedBlocks) * sim.Replicas[rIndex].KVCache.BlockSizeTokens
		endIndex := startIndex + numNewTokens

		// estimate the number of new blocks needed for the next request
		// and allocate if possible
		if ok := sim.Replicas[rIndex].KVCache.AllocateKVBlocks(next, startIndex, endIndex, cachedBlocks); !ok {
			// cannot allocate enough blocks for remaining tokens, do not schedule current request
			// vLLM maintains First-Come-First-Served order of requests, so we cannot move onto the
			// next request.
			break
		}

		// at this point: the `next` request is deemed schedulable

		// dequeue this request
		sim.Replicas[rIndex].WaitQ.queue = sim.Replicas[rIndex].WaitQ.queue[1:]
		// make it part of the running batch
		sim.Replicas[rIndex].RunningBatch.Requests = append(sim.Replicas[rIndex].RunningBatch.Requests, next)
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
		sim.Replicas[rIndex].RunningBatchFeatures.NumPrefillRequests += 1
		sim.Replicas[rIndex].RunningBatchFeatures.TotalPrefillTokens += numNewTokens
		sim.Replicas[rIndex].RunningBatchFeatures.TotalCacheMissTokens += numNewTokens
		sim.Replicas[rIndex].RunningBatchFeatures.MaxPrefillTokens = max(sim.Replicas[rIndex].RunningBatchFeatures.MaxPrefillTokens, numNewTokens)
		sim.ReqNumComputedTokens[next.ID] = numNewTokens + Len64(cachedBlocks)*sim.Replicas[rIndex].KVCache.BlockSizeTokens
	}
}

// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
// ToDo: Understand and handle pre-emption logic, if need be.
func (sim *Simulator) Step(now int64, rIndex int) {

	// increment Step counter
	sim.StepCount += 1

	// refreshing RunningBatchFeatures for current Step
	sim.Replicas[rIndex].RunningBatchFeatures = RegressionFeatures{
		TotalDecodeTokens:    0,
		TotalCacheMissTokens: 0,
	}
	// Subprocess: fill running batch from wait queue, similar to vLLM's scheduler.schedule()
	sim.makeRunningBatch(now, rIndex)

	// save waitQ length for analysis
	sim.GlobalMetrics.NumWaitQRequests = append(sim.GlobalMetrics.NumWaitQRequests, len(sim.Replicas[rIndex].WaitQ.queue))

	// save runningBatch length for analysis
	sim.GlobalMetrics.NumRunningBatchRequests = append(sim.GlobalMetrics.NumRunningBatchRequests, len(sim.Replicas[rIndex].RunningBatch.Requests))

	// Estimate regression times based on runningBatch state
	currStepAdvance := sim.getStepTime(rIndex)

	// Subprocess: Model Execution - this could be prefill or decode depending on the request.
	// similar to vLLM's execute_model()
	for _, req := range sim.Replicas[rIndex].RunningBatch.Requests {
		if req.ProgressIndex < Len64(req.InputTokens) {
			req.ProgressIndex = sim.ReqNumComputedTokens[req.ID]
			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else {
			// this request goes through decode phase in this batch
			req.ProgressIndex++
			sim.Replicas[rIndex].Metrics.TotalOutputTokens++
		}
		if req.ProgressIndex == Len64(req.InputTokens) { // prefill complete, first token is generated
			req.TTFTSet = true
			req.FirstTokenTime = now + currStepAdvance - req.ArrivalTime
			sim.Replicas[rIndex].Metrics.TTFTSum += req.FirstTokenTime                             // in microsec
			sim.Replicas[rIndex].Metrics.RequestTTFTs[req.ID] = float64(req.FirstTokenTime) / 1000 // in ms
		}
	}

	// Subprocess: check completion and push next step event, similar to vLLM's
	// scheduler.update_from_output()

	// Write KVBlocks usage metrics

	if sim.Replicas[rIndex].KVCache.UsedBlockCnt > sim.Replicas[rIndex].Metrics.PeakKVBlocksUsed {
		sim.Replicas[rIndex].Metrics.PeakKVBlocksUsed = sim.Replicas[rIndex].KVCache.UsedBlockCnt
	}
	sim.Replicas[rIndex].Metrics.KVBlocksUsed += float64(sim.Replicas[rIndex].KVCache.UsedBlockCnt) * float64(currStepAdvance)

	// handle completed and remaining requests
	remaining := []*Request{}
	for _, req := range sim.Replicas[rIndex].RunningBatch.Requests {
		// in cases where there are 0 output tokens, set it to 1 manually to avoid errors
		if req.ProgressIndex == Len64(req.InputTokens)+max(Len64(req.OutputTokens), 1)-1 {
			req.State = "completed"
			if len(req.OutputTokens) > 0 {
				ok := sim.Replicas[rIndex].KVCache.AllocateKVBlocks(req, req.ProgressIndex, req.ProgressIndex+1, []int64{})
				if !ok {
					// Could not allocate (e.g., no free blocks)
					logrus.Warnf("[THIS SHOULD NEVER HAPPEN]")
					continue
				}
			}
			sim.Replicas[rIndex].KVCache.ReleaseKVBlocks(req)
			sim.Replicas[rIndex].Metrics.CompletedRequests++
			// trigger the RequestLeftEvent
			requestLeftDelay := sim.getRequestLeftProcessingTime(req) // alpha params to estimate
			sim.Schedule(&RequestLeftEvent{
				time:    now + currStepAdvance + requestLeftDelay,
				Request: req,
			})

			lat := now + currStepAdvance - req.ArrivalTime
			sim.Replicas[rIndex].Metrics.RequestE2Es[req.ID] = float64(lat) / 1000 // in ms
			logrus.Infof("Finished req: ID: %s at time: %d\n", req.ID, lat+req.ArrivalTime)
			if len(req.OutputTokens) > 0 {
				reqTotalOutput := lat - req.FirstTokenTime
				sim.Replicas[rIndex].Metrics.TPOTSum += reqTotalOutput // in microsec
				// TPOT calculation in vLLM excludes the first generated token, calculated in ms
				sim.Replicas[rIndex].Metrics.RequestTPOTs[req.ID] = float64(reqTotalOutput) / float64(max(len(req.OutputTokens)-1, 1)) / 1000
			} else {
				sim.Replicas[rIndex].Metrics.RequestTPOTs[req.ID] = 0
			}
			req.FinishedStepIdx = sim.StepCount
			sim.Replicas[rIndex].Metrics.RequestStepCounters = append(sim.Replicas[rIndex].Metrics.RequestStepCounters, req.FinishedStepIdx-req.ScheduledStepIdx)
			sim.Replicas[rIndex].Metrics.RequestCompletionTimes[req.ID] = float64(lat+req.ArrivalTime) / 1e6 // in seconds
		} else {
			remaining = append(remaining, req)
		}
	}

	// push the next step event as needed
	if len(remaining) > 0 {
		sim.Replicas[rIndex].RunningBatch.Requests = remaining
		// estimate queue overhead from LR (sim.features)
		//
		pbe := StepEvent{time: now + currStepAdvance}
		sim.Schedule(&pbe)
		sim.Replicas[rIndex].StepEvent = &pbe
	} else {
		sim.Replicas[rIndex].RunningBatch = nil
		sim.Replicas[rIndex].StepEvent = nil
	}
}
