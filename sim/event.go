package sim

import "github.com/sirupsen/logrus"

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

	// Enqueue the arriving request into the waiting queue
	sim.EnqueueRequest(e.Request)

	// If there's no Step scheduled, trigger one immediately
	if sim.StepEvent == nil {
		sim.Schedule(&StepEvent{
			time: e.time,
		})
	}
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
