package sim

import (
	"github.com/sirupsen/logrus"
)

// Event priority constants. At equal timestamps, lower priority fires first.
// This ensures deterministic same-tick ordering (INV-6 improvement over
// timestamp-only heap). Matches cluster event queue's (timestamp, priority, seqID) scheme.
const (
	PriorityArrival      = 0 // External input enters system first
	PriorityQueued       = 1 // Request enters queue after arrival processing
	PriorityStep         = 2 // Batch processing (may complete requests)
	PriorityScheduled    = 3 // Observational (no state mutation)
	PriorityRequestLeft  = 4 // Observational (no state mutation)
	PriorityTimeout      = 5 // Client-side cancellation fires last (BC-12: completion wins)
)

// Event defines the interface for all simulation events.
// Each event must have a Timestamp (in ticks), a Priority for deterministic
// same-tick ordering, and an Execute method that advances simulation state.
type Event interface {
	Timestamp() int64
	Priority() int
	Execute(*Simulator)
}

// ArrivalEvent represents the arrival of a new inference request into the system.
type ArrivalEvent struct {
	time    int64    // Simulation time of arrival (in ticks)
	Request *Request // The incoming request associated with this event
}

func (e *ArrivalEvent) Timestamp() int64 { return e.time }
func (e *ArrivalEvent) Priority() int    { return PriorityArrival }

// Execute schedules the next StepEvent, if no such event is scheduled
func (e *ArrivalEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< Arrival: %s at %d ticks", e.Request.ID, e.time)

	// Trigger queued event with processing delay
	queued_delay := sim.latencyModel.QueueingTime(e.Request) // coming from alpha model
	sim.Schedule(&QueuedEvent{
		time:    e.time + queued_delay,
		Request: e.Request,
	})

}

// QueuedEvent represents the queue of a new inference request into the system.
type QueuedEvent struct {
	time    int64    // Simulation time of queued (in ticks)
	Request *Request // The incoming request associated with this event
}

func (e *QueuedEvent) Timestamp() int64 { return e.time }
func (e *QueuedEvent) Priority() int    { return PriorityQueued }

// Execute enqueues the request and triggers a StepEvent if needed.
// The WaitQ.Len() > 0 check prevents phantom empty-batch Steps after
// EnqueueRequest returns early (past-due timeout, StateTimedOut guard,
// dropped-unservable). Sets sim.stepEvent to prevent duplicate scheduling.
func (e *QueuedEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< Queued: %s at %d ticks", e.Request.ID, e.time)

	// Enqueue the arriving request into the waiting queue
	sim.EnqueueRequest(e.Request)

	// If there's no Step scheduled and WaitQ has work, trigger one immediately.
	if sim.stepEvent == nil && sim.WaitQ.Len() > 0 {
		pbe := &StepEvent{time: e.time}
		sim.Schedule(pbe)
		sim.stepEvent = pbe
	}
}

// ScheduledEvent represents the scheduling of a new inference request in the system.
type ScheduledEvent struct {
	time    int64    // Simulation time of Scheduled (in ticks)
	Request *Request // The incoming request associated with this event
}

func (e *ScheduledEvent) Timestamp() int64 { return e.time }
func (e *ScheduledEvent) Priority() int    { return PriorityScheduled }

// Execute does nothing
func (e *ScheduledEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< Schedule: %s at %d ticks", e.Request.ID, e.time)
}

// RequestLeftEvent represents the leaving of an inference request from the system.
type RequestLeftEvent struct {
	time    int64    // Simulation time of RequestLeftEvent (in ticks)
	Request *Request // The incoming request associated with this event
}

func (e *RequestLeftEvent) Timestamp() int64 { return e.time }
func (e *RequestLeftEvent) Priority() int    { return PriorityRequestLeft }

// Execute does nothing
func (e *RequestLeftEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< RequestLeft: %s at %d ticks", e.Request.ID, e.time)
}

// StepEvent represents a simulation step.
// It encapsulates the vLLM step function, consisting of the following:
//   - scheduler.schedule()
//   - execute_model()
//   - scheduler.update_from_output()
type StepEvent struct {
	time int64 // Scheduled execution time (in ticks)
}

func (e *StepEvent) Timestamp() int64 { return e.time }
func (e *StepEvent) Priority() int    { return PriorityStep }

// Execute the StepEvent
func (e *StepEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< StepEvent at %d ticks", e.time)
	sim.Step(e.time)
}

// TimeoutEvent models client-side request cancellation at the deadline tick.
// Classification: mixed exogenous/endogenous (round-0 exogenous, follow-up endogenous).
// Priority 5: fires after all other event types at equal timestamps (BC-12).
type TimeoutEvent struct {
	time    int64
	Request *Request
}

func (e *TimeoutEvent) Timestamp() int64 { return e.time }
func (e *TimeoutEvent) Priority() int    { return PriorityTimeout }

// Execute cancels the request if it hasn't already completed or timed out.
// Three paths: (1) running request — new-slice removal from RunningBatch (R21),
// (2) queued request — WaitQ.Remove, (3) pre-QueuedEvent race — request not in
// any container yet (WaitQ.Remove returns false, safe no-op).
func (e *TimeoutEvent) Execute(sim *Simulator) {
	// No-op guard: request already completed or timed out (BC-3)
	if e.Request.State == StateCompleted || e.Request.State == StateTimedOut {
		return
	}
	wasRunning := e.Request.State == StateRunning
	e.Request.State = StateTimedOut
	sim.Metrics.TimedOutRequests++

	// Release KV blocks (safe for zero-block queued requests per BC-15)
	sim.KVCache.ReleaseKVBlocks(e.Request)

	// Clean up computed-token tracking for ALL timed-out requests (prevents memory leak).
	// A preempted-then-queued request still has an entry from its prior running phase.
	delete(sim.reqNumComputedTokens, e.Request.ID)

	if wasRunning {
		// New-slice construction (R21): build excluding timed-out request.
		// Allocates O(batch_size) per running-request timeout. Acceptable because:
		// (a) running-request timeouts are infrequent, (b) batch size is typically O(100s),
		// (c) R21 compliance requires avoiding in-place mutation of iterated slices.
		remaining := make([]*Request, 0, len(sim.RunningBatch.Requests)-1)
		for _, r := range sim.RunningBatch.Requests {
			if r != e.Request {
				remaining = append(remaining, r)
			}
		}
		sim.RunningBatch.Requests = remaining
		// If batch is now empty, clear RunningBatch. Do NOT nil sim.stepEvent:
		// leaving it pointing to the already-scheduled StepEvent prevents the
		// INV-8 guard below from creating a duplicate StepEvent at the current
		// tick. At the same tick, StepEvents fire before TimeoutEvents (priority
		// 2 < 5), so a newly-created StepEvent would pull a queued request and
		// then its same-deadline TimeoutEvent would immediately time it out —
		// cascading once per queued seed and leaving N orphaned StepEvents.
		// The already-scheduled StepEvent fires at the correct future time and
		// the Step() orphan guard handles the empty-batch case.
		if len(remaining) == 0 {
			sim.RunningBatch = nil
		}
	} else {
		sim.WaitQ.Remove(e.Request)
	}

	// INV-8 work-conserving: if running batch is now empty but WaitQ has work,
	// schedule a StepEvent (defense-in-depth, BC-18)
	if (sim.RunningBatch == nil || len(sim.RunningBatch.Requests) == 0) &&
		sim.stepEvent == nil && sim.WaitQ.Len() > 0 {
		pbe := &StepEvent{time: e.time}
		sim.Schedule(pbe)
		sim.stepEvent = pbe
	}

	// Invoke completion callback for session management
	if sim.OnRequestDone != nil {
		for _, next := range sim.OnRequestDone(e.Request, e.time) {
			sim.InjectArrival(next)
		}
	}
}
