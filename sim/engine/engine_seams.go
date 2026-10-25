// sim/engine_seams.go: the two interfaces the engine declares for what it needs from outside
// itself: its host (the clock, the event queue, the callbacks) and its recorder (the host's
// report record). The Simulator and *Metrics implement them (#2021 B2, B3).
package engine

// EngineHost is what the engine needs from the simulator it runs in (#2021 B3: the clock,
// the event queue and the callbacks stay with the host). The engine reaches the host only
// through it, so the engine can move to blis-vllm with the host left behind (PLAN-E2 E2.3).
// The calls are made at exactly the points the engine made them before, so events are
// created in the same order with the same seqIDs (hazard 2).
type EngineHost interface {
	// CurrentClock is the simulation clock.
	CurrentClock() int64
	// SimHorizon is the simulation horizon.
	SimHorizon() int64
	// ScheduleTimeout queues the request's client-deadline timeout at t (PriorityTimeout).
	ScheduleTimeout(req *Request, t int64)
	// ScheduleScheduled and ScheduleLeft queue the observational scheduled and request-left
	// events at t (PriorityScheduled, PriorityRequestLeft).
	ScheduleScheduled(req *Request, t int64)
	ScheduleLeft(req *Request, t int64)
	// ScheduleAdapterLoaded queues the completion of an adapter's load at t
	// (PriorityAdapterLoad).
	ScheduleAdapterLoaded(adapter string, t int64)
	// RequestDone hands a request that reached a terminal state, at tick, to the host's
	// callback, and injects the follow-ups it returns. A no-op without a callback.
	RequestDone(req *Request, tick int64)
	// FirstTokenAt reports a request's first token, received by the client at at.
	FirstTokenAt(req *Request, at int64)
	// WakeStep makes sure a step fires no later than t (scheduleStepBy).
	WakeStep(t int64)
	// ArmStep arms the next step at t (scheduleStepAt).
	ArmStep(t int64)
	// StepArmed reports whether a step is armed.
	StepArmed() bool
}

// Recorder is how the engine writes the host's report record: every count, latency and series
// the engine produces goes through one of these calls, at exactly the point the engine used to
// write the field (#2021 B2: inference-sim owns its reports; the engine updates them through
// the seam, PLAN-E2 E2.3b). *Metrics implements it.
type Recorder interface {
	// RecordArrival registers a request the instance has received; arrivedAt is its arrival
	// in seconds.
	RecordArrival(req *Request, arrivedAt float64)
	// RecordDrop records a request dropped at enqueue: contextOverflow when it exceeds
	// the context window.
	RecordDrop(req *Request, contextOverflow bool)
	// RecordInputTokens counts a received request's prompt tokens.
	RecordInputTokens(tokens int64)
	// RecordTimedOut records a request its client deadline cancelled.
	RecordTimedOut(req *Request)
	// RecordPreemption counts one preemption.
	RecordPreemption()
	// PreemptionsSoFar is the cumulative preemption count.
	PreemptionsSoFar() int64
	// RecordScheduled records a request's admission into the running batch at now: its scheduling
	// delay and its prefix-cache lookup.
	RecordScheduled(req *Request, now int64, lookup PrefixLookup)
	// FirstTokenAt records a request's time to first token, in ticks.
	RecordFirstToken(req *Request)
	// RecordQueueSnapshot records the wait queue and running batch sizes after a schedule.
	RecordQueueSnapshot(waiting, running int)
	// RecordKVUsage records a step's KV blocks in use over duration ticks (the time-weighted
	// integral and the peak). With weight false it updates the peak only (a free poll).
	RecordKVUsage(used, duration int64, weight bool)
	// RecordCompletion records a finished request (see recordRequestCompletion for each value).
	RecordCompletion(c Completion)
	// RecordLengthCapped records a request force-completed at the context limit.
	RecordLengthCapped(req *Request)
	// RecordKVAllocationFailure counts a final-token KV allocation that failed.
	RecordKVAllocationFailure()
	// RecordAdapterLoad and RecordAdapterEviction count LoRA residency transitions.
	RecordAdapterLoad(id string)
	RecordAdapterEviction(id string)
	// RecordStillHeldAtEnd records the requests the engine still holds at the end of the run.
	RecordStillHeldAtEnd(waiting, running int)
}

// completion is one finished request's report values, computed by the engine.
type Completion struct {
	Req          *Request
	TTFT         int64 // FirstTokenTime
	OutputTokens int   // emitted output tokens (>= 0)
	E2E          int64 // end-to-end latency in ticks
	ITL          float64
	Steps        int   // FinishedStepIdx - ScheduledStepIdx
	Clock        int64 // the clock at which the completion was processed
	Overhead     int64 // the post-decode overhead e2e includes
}
