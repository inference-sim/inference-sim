// sim/engine_seams.go: the two interfaces the engine declares for what it needs from outside
// itself: its host (the clock, the event queue, the callbacks) and its recorder (the host's
// report record). The Simulator and *Metrics implement them (#2021 B2, B3).
package sim

// engineHost is what the engine needs from the simulator it runs in (#2021 B3: the clock,
// the event queue and the callbacks stay with the host). The engine reaches the host only
// through it, so the engine can move to blis-vllm with the host left behind (PLAN-E2 E2.3).
// The calls are made at exactly the points the engine made them before, so events are
// created in the same order with the same seqIDs (hazard 2).
type engineHost interface {
	// now is the simulation clock.
	now() int64
	// horizon is the simulation horizon.
	horizon() int64
	// scheduleTimeout queues the request's client-deadline timeout at t (PriorityTimeout).
	scheduleTimeout(req *Request, t int64)
	// scheduleScheduled and scheduleLeft queue the observational scheduled and request-left
	// events at t (PriorityScheduled, PriorityRequestLeft).
	scheduleScheduled(req *Request, t int64)
	scheduleLeft(req *Request, t int64)
	// scheduleAdapterLoaded queues the completion of an adapter's load at t
	// (PriorityAdapterLoad).
	scheduleAdapterLoaded(adapter string, t int64)
	// requestDone hands a request that reached a terminal state, at tick, to the host's
	// callback, and injects the follow-ups it returns. A no-op without a callback.
	requestDone(req *Request, tick int64)
	// firstToken reports a request's first token, received by the client at at.
	firstToken(req *Request, at int64)
	// wakeStep makes sure a step fires no later than t (scheduleStepBy).
	wakeStep(t int64)
	// armStep arms the next step at t (scheduleStepAt).
	armStep(t int64)
	// stepArmed reports whether a step is armed.
	stepArmed() bool
}

// recorder is how the engine writes the host's report record: every count, latency and series
// the engine produces goes through one of these calls, at exactly the point the engine used to
// write the field (#2021 B2: inference-sim owns its reports; the engine updates them through
// the seam, PLAN-E2 E2.3b). *Metrics implements it.
type recorder interface {
	// requestArrived registers a request the instance has received; arrivedAt is its arrival
	// in seconds.
	requestArrived(req *Request, arrivedAt float64)
	// requestDropped records a request dropped at enqueue: contextOverflow when it exceeds
	// the context window.
	requestDropped(req *Request, contextOverflow bool)
	// inputReceived counts a received request's prompt tokens.
	inputReceived(tokens int64)
	// timedOut records a request its client deadline cancelled.
	timedOut(req *Request)
	// preempted counts one preemption.
	preempted()
	// preemptions is the cumulative preemption count.
	preemptions() int64
	// scheduled records a request's admission into the running batch at now: its scheduling
	// delay and its prefix-cache lookup.
	scheduled(req *Request, now int64, lookup PrefixLookup)
	// firstToken records a request's time to first token, in ticks.
	firstToken(req *Request)
	// queueSnapshot records the wait queue and running batch sizes after a schedule.
	queueSnapshot(waiting, running int)
	// kvUsage records a step's KV blocks in use over duration ticks (the time-weighted
	// integral and the peak). With weight false it updates the peak only (a free poll).
	kvUsage(used, duration int64, weight bool)
	// completed records a finished request (see recordRequestCompletion for each value).
	completed(c completion)
	// lengthCapped records a request force-completed at the context limit.
	lengthCapped(req *Request)
	// kvAllocationFailure counts a final-token KV allocation that failed.
	kvAllocationFailure()
	// adapterLoaded and adapterEvicted count LoRA residency transitions.
	adapterLoaded(id string)
	adapterEvicted(id string)
	// stillHeld records the requests the engine still holds at the end of the run.
	stillHeld(waiting, running int)
}

// completion is one finished request's report values, computed by the engine.
type completion struct {
	req          *Request
	ttft         int64 // FirstTokenTime
	outputTokens int   // emitted output tokens (>= 0)
	e2e          int64 // end-to-end latency in ticks
	itl          float64
	steps        int   // FinishedStepIdx - ScheduledStepIdx
	clock        int64 // the clock at which the completion was processed
	overhead     int64 // the post-decode overhead e2e includes
}
