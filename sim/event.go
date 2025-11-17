package sim

import (
	"github.com/sirupsen/logrus"
)

// Event defines the interface for all simulation events.
// Each event must have a Timestamp (in ticks) and an Execute method
// that advances simulation state when invoked.
type Event interface {
	Timestamp() int64
	Execute(*Simulator)
}

// ArrivalEvent represents the arrival of a new inference request into the system.
type ArrivalEvent struct {
	time    int64    // Simulation time of arrival (in ticks)
	Request *Request // The incoming request associated with this event
}

// Timestamp returns the scheduled time of the ArrivalEvent.
func (e *ArrivalEvent) Timestamp() int64 {
	return e.time
}

// Execute schedules the next StepEvent, if no such event is scheduled
func (e *ArrivalEvent) Execute(sim *Simulator) {
	logrus.Infof("<< Arrival: %s at %d ticks", e.Request.ID, e.time)

	// Trigger queued event with processing delay
	queued_delay := sim.getProcessingTime(e.Request) // coming from alpha model
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

// Timestamp returns the time of the QueuedEvent.
func (e *QueuedEvent) Timestamp() int64 {
	return e.time
}

// Execute normally just enqueues the request
// If this is the first step, Execute calls the StepEvent
func (e *QueuedEvent) Execute(sim *Simulator) {
	logrus.Infof("<< Queued: %s at %d ticks", e.Request.ID, e.time)

	// Enqueue the arriving request into the waiting queue
	sim.EnqueueRequest(e.Request)

	// If there's no Step scheduled, trigger one immediately
	if sim.StepEvent == nil {
		sim.Schedule(&StepEvent{
			time: e.time,
		})
	}
}

// ScheduledEvent represents the scheduling of a new inference request in the system.
type ScheduledEvent struct {
	time    int64    // Simulation time of Scheduled (in ticks)
	Request *Request // The incoming request associated with this event
}

// Timestamp returns the time of the ScheduledEvent.
func (e *ScheduledEvent) Timestamp() int64 {
	return e.time
}

// Execute does nothing
func (e *ScheduledEvent) Execute(sim *Simulator) {
	logrus.Infof("<< Schedule: %s at %d ticks", e.Request.ID, e.time)
}

// PreemptionEvent represents the pre-emption of an inference request in the system.
type PreemptionEvent struct {
	time    int64    // Simulation time of PreemptionEvent (in ticks)
	Request *Request // The incoming request associated with this event
}

// Timestamp returns the time of the PreemptionEvent.
func (e *PreemptionEvent) Timestamp() int64 {
	return e.time
}

// Execute does nothing
func (e *PreemptionEvent) Execute(sim *Simulator) {
	logrus.Infof("<< Preemption: %s at %d ticks", e.Request.ID, e.time)
}

// RequestLeftEvent represents the leaving of an inference request from the system.
type RequestLeftEvent struct {
	time    int64    // Simulation time of RequestLeftEvent (in ticks)
	Request *Request // The incoming request associated with this event
}

// Timestamp returns the time of the RequestLeftEvent.
func (e *RequestLeftEvent) Timestamp() int64 {
	return e.time
}

// Execute does nothing
func (e *RequestLeftEvent) Execute(sim *Simulator) {
	logrus.Infof("<< RequestLeft: %s at %d ticks", e.Request.ID, e.time)
}

// StepEvent represents a simulation step.
// It encapsulates the vLLM step function, consisting of the following:
//   - scheduler.schedule()
//   - execute_model()
//   - scheduler.update_from_output()
type StepEvent struct {
	time int64 // Scheduled execution time (in ticks)
}

// Timestamp returns the scheduled time of the StepEvent.
func (e *StepEvent) Timestamp() int64 {
	return e.time
}

// Execute the StepEvent
func (e *StepEvent) Execute(sim *Simulator) {
	logrus.Infof("<< StepEvent at %d ticks", e.time)
	sim.Step(e.time)
}
