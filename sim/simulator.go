// sim/simulator.go
package sim

import (
	"container/heap"
	"fmt"
	"log"
	"math/rand"
	"strconv"
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

// Simulator is the core object that holds simulation time, system state, and the event loop.
type Simulator struct {
	Clock   int64
	Horizon int64
	// Advance estimates the per-batch model execution time
	Advance int64
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
	Metrics      *Metrics
	MaxBatchSize int64
	StepEvent    Event
}

func NewSimulator(horizon int64, advance int64, totalKVBlocks int, blockSizeTokens int, maxBatchSize int64) *Simulator {
	s := &Simulator{
		Clock:        0,
		Horizon:      horizon,
		Advance:      advance,
		EventQueue:   make(EventQueue, 0),
		WaitQ:        &WaitQueue{},
		KVCache:      NewKVCacheState(totalKVBlocks, blockSizeTokens),
		RunningBatch: &Batch{},
		Metrics:      &Metrics{RequestLatencies: make(map[string]int64)},
		MaxBatchSize: maxBatchSize,
		StepEvent:    nil,
	}
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
		log.Printf("[tick %07d] Executing %T", sim.Clock, ev)
		// process the event
		ev.Execute(sim)
		// end the simulation if horizon is reached or if we've run out of events
		if sim.Clock > sim.Horizon {
			break
		}
	}
	log.Printf("[tick %07d] Simulation ended", sim.Clock)
}

// Adds a newly arrived request to the waiting queue
func (sim *Simulator) EnqueueRequest(r *Request) {
	sim.WaitQ.Enqueue(r)
}

// GeneratePoissonArrivals generates requests with arrival distributed as a Poisson process
func (sim *Simulator) GeneratePoissonArrivals(rate float64, horizon int64, seed int64) {
	currentTime := int64(0)
	// each request gets a unique id
	requestId := 0
	// initialize the random number generator
	rGen := rand.New(rand.NewSource(seed))

	// create request arrivals iteratively
	for currentTime < horizon {
		// In a Poisson process, the arrival rate is inversely proportional
		// to the mean interarrival time
		delta := int64(rGen.ExpFloat64() * (1.0 / rate))
		currentTime += delta
		if currentTime > horizon {
			break
		}
		// generate random input and output tokens; their lengths and contents are both random
		// ToDo: create flags for max input and output lengths
		// Plug them into rGen.Intn below, instead of hardcoded values
		input := make([]string, rGen.Intn(20)+10)
		output := make([]string, rGen.Intn(10)+5)
		for i := range input {
			input[i] = "tok" + strconv.Itoa(rGen.Intn(100000))
		}
		for i := range output {
			output[i] = "tok" + strconv.Itoa(rGen.Intn(100000))
		}

		// form the request; it will be in the "queued" state when it arrives
		req := &Request{
			ID:           fmt.Sprintf("req-%d", requestId),
			ArrivalTime:  currentTime,
			InputTokens:  input,
			OutputTokens: output,
			State:        "queued",
		}
		// push the request for arrival
		sim.Schedule(&ArrivalEvent{time: currentTime, Request: req})

		// move on to the next request
		requestId++
	}
}

// In vllm, the processing of requests proceeds iteratively in steps.
// Step simulates a single vllm step(), which roughly corresponds to a single scheduler.schedule()
// to construct a batch, model execution of the batch and scheduler.update().
// ToDo: Understand and handle pre-emption logic, if need be.
func (sim *Simulator) Step(now int64) {

	// Subprocess: fill running batch from wait queue, similar to vLLM's scheduler.schedule()

	if sim.RunningBatch == nil {
		sim.RunningBatch = &Batch{}
	}

	// attempt to dequeue, if batch size is not exceeded, and there is a request waiting
	for len(sim.RunningBatch.Requests) < int(sim.MaxBatchSize) && len(sim.WaitQ.queue) > 0 {
		// we will attempt to dequeue `next` request
		// if that attempt fails, we will break out of the loop

		// estimate the number of new blocks needed for the next request
		next := sim.WaitQ.queue[0]
		_, _, numRemainingBlocks := sim.KVCache.CacheStateFor(next)

		// ToDo: verify if the following checks are used by vLLM to determine schedulability
		if numRemainingBlocks > sim.KVCache.countFreeBlocks() {
			break
		}
		// note: in reality, some of the stuff like caching of newly created blocks
		// that happens inside AllocateKVBlocks
		// should really be happening inside model execution, or after
		// but we will take this shortcut for now
		if ok := sim.KVCache.AllocateKVBlocks(next); !ok {
			// this really should not happen
			// ToDo: Log an error here
			break
		}

		// at this point: the `next` request is deemed schedulable

		// dequeue this request
		sim.WaitQ.queue = sim.WaitQ.queue[1:]
		// make it part of the running batch
		sim.RunningBatch.Requests = append(sim.RunningBatch.Requests, next)
		// change the state of the request from queued to running
		next.State = "running"
	}

	// if there are no requests in RunningBatch at this point, we're done with this step
	if len(sim.RunningBatch.Requests) == 0 {
		return
	}

	// Subprocess: Model Execution - this could be prefill or decode depending on the request
	// We want to make this efficient so that the total simulation time is
	// O(TT), where TT is the total number of input and output tokens
	// across all requests
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex == 0 {
			// this request goes through prefill phase in this batch
			req.ProgressIndex = len(req.InputTokens)
			req.TTFTSet = true
			req.FirstTokenTime = now + sim.Advance
			sim.Metrics.TTFTSum += now + sim.Advance - req.ArrivalTime

			// ToDo: Go through the newly allocated blocks for this request;
			// Make sure they are cached, if they're full
		} else if req.ProgressIndex >= len(req.InputTokens) {
			// this request goes through decode phase in this batch
			nextTokenIndex := req.ProgressIndex - len(req.InputTokens)
			nextToken := req.OutputTokens[nextTokenIndex]
			ok := sim.KVCache.AppendToken(req.ID, nextToken)
			if !ok {
				// Could not allocate (e.g., no free blocks)
				continue // ToDo: pre-empt this request
			}
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
	sim.Metrics.KVBlocksUsed += sim.KVCache.UsedBlockCnt * int(sim.Advance)

	// handle completed and remaining requests
	remaining := []*Request{}
	for _, req := range sim.RunningBatch.Requests {
		if req.ProgressIndex == len(req.InputTokens)+len(req.OutputTokens) {
			req.State = "completed"
			sim.KVCache.ReleaseKVBlocks(req)
			sim.Metrics.CompletedRequests++
			lat := now + sim.Advance - req.ArrivalTime
			sim.Metrics.TotalLatency += lat
			sim.Metrics.RequestLatencies[req.ID] = lat
			if len(req.OutputTokens) > 0 {
				sim.Metrics.TPOTSum += now + sim.Advance - req.FirstTokenTime
			}
		} else {
			remaining = append(remaining, req)
		}
	}

	// push the next step event as needed
	if len(remaining) > 0 {
		sim.RunningBatch.Requests = remaining
		pbe := StepEvent{time: now + sim.Advance}
		sim.Schedule(&pbe)
		sim.StepEvent = &pbe
	} else {
		sim.RunningBatch = nil
		sim.StepEvent = nil
	}
}
