package sim

import (
	"github.com/sirupsen/logrus"
)

// Event priority constants. At equal timestamps, lower priority fires first.
// This ensures deterministic same-tick ordering (INV-6 improvement over
// timestamp-only heap). Matches cluster event queue's (timestamp, priority, seqID) scheme.
const (
	PriorityArrival     = 0 // External input enters system first
	PriorityQueued      = 1 // Request enters queue after arrival processing
	PriorityAdapterLoad = 2 // LoRA adapter-load completion: makes a gated adapter resident
	//                        BEFORE a co-timed step forms its batch (§7 ordering
	//                        contract), so the newly-loaded adapter's request is
	//                        batch-eligible that same tick.
	PriorityStep        = 3 // Batch processing (may complete requests)
	PriorityScheduled   = 4 // Observational (no state mutation)
	PriorityRequestLeft = 5 // Observational (no state mutation)
	PriorityTimeout     = 6 // Client-side cancellation fires last (BC-12: completion wins)
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
	sim.Schedule(&QueuedEvent{
		time:    sim.QueuedAt(e.Request, e.time),
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

	// If WaitQ has work, make sure a Step runs now: schedule one if none is pending,
	// or pull an idle wake forward to this arrival (#1886).
	if sim.HasWaiting() {
		sim.scheduleStepBy(e.time)
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
	// idleWake marks a step scheduled to wake an idle instance at the next tick a
	// deferred request can change state. New work may supersede it (scheduleStepBy).
	idleWake bool
	// cancelled marks a superseded idle wake. ProcessNextEvent drops it without
	// advancing the clock, like an orphaned TimeoutEvent.
	cancelled bool
}

func (e *StepEvent) Timestamp() int64 { return e.time }
func (e *StepEvent) Priority() int    { return PriorityStep }

// Cancelled reports whether this step was superseded and must not run.
func (e *StepEvent) Cancelled() bool { return e.cancelled }

// Execute the StepEvent
func (e *StepEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< StepEvent at %d ticks", e.time)
	if sim.lockstep != nil {
		sim.lockstep.step(e)
		return
	}
	sim.Step(e.time)
}

// AdapterLoadCompletionEvent fires when a per-instance LoRA adapter load finishes.
// It is endogenous — the cold-load pre-admission gate schedules it at
// now + LoadLatency when a cold request reaches the wait-queue head (§7). On
// completion the adapter becomes resident and the gated request(s) become
// batch-eligible; the event is ordered ahead of a co-timed StepEvent
// (PriorityAdapterLoad < PriorityStep) so the request is admitted that same tick.
// Loads serialize per instance: the gate starts at most one at a time.
type AdapterLoadCompletionEvent struct {
	time    int64  // Scheduled completion time (in ticks) = load-start + LoadLatency
	Adapter string // Adapter id being loaded
}

func (e *AdapterLoadCompletionEvent) Timestamp() int64 { return e.time }
func (e *AdapterLoadCompletionEvent) Priority() int    { return PriorityAdapterLoad }

// Execute makes the loaded adapter resident, charges the one-time load (INV-L3),
// clears the in-flight-load marker so the next serialized load can start, and
// ensures a step forms so the gated request is admitted (INV-8).
func (e *AdapterLoadCompletionEvent) Execute(sim *Simulator) {
	logrus.Debugf("<< AdapterLoadCompletion: %s at %d ticks", e.Adapter, e.time)
	sim.CompleteAdapterLoad(e.time, e.Adapter)
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
	if !sim.CancelOnTimeout(e.Request) {
		return
	}

	// INV-8 work-conserving: if running batch is now empty but WaitQ has work,
	// schedule a StepEvent (defense-in-depth, BC-18)
	//
	// Deliberately NOT scheduleStepBy (#1886): this is the one idle-scheduling site that
	// keeps the stepEvent == nil form. scheduleStepBy would pull a pending idle wake
	// forward to e.time, re-creating the cascade the comment in EngineCore.CancelOnTimeout describes.
	// Nothing is stranded by waiting: an idle wake exists only when every WaitQ entry is
	// deferred, and a queued timeout frees no CPU-tier capacity a deferral could use —
	// OffloadCache.ReleaseKVBlocks delegates to the GPU tier alone (kv/offload_chain.go)
	// and ClearDeferred leaves in-flight CPU slots to complete (kv/offload_deferral.go).
	// So the remaining deferrals are polled at the wake, at a tick where admission could
	// not have happened sooner anyway. Pinned by
	// TestIdleWake_TimeoutDuringWaitDoesNotDelayAdmission: dropping the stale wake here
	// instead of leaving it strands the sibling deferral.
	if !sim.HasRunning() && sim.stepEvent == nil && sim.HasWaiting() {
		sim.scheduleStepAt(e.time)
	}

	// Invoke completion callback for session management
	sim.requestDone(e.Request, e.time)
}
