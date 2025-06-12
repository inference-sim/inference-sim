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

// Execute schedules the next ProcessBatchEvent, if no such event is scheduled
func (e *ArrivalEvent) Execute(sim *Simulator) {
	logrus.Infof("<< Arrival: %s at %d ticks", e.Request.ID, e.time)

	// Enqueue the arriving request into the waiting queue
	sim.EnqueueRequest(e.Request)

	// If there's no forward pass scheduled, trigger one immediately
	if sim.ProcessBatchEvent == nil {
		sim.Schedule(&ProcessBatchEvent{
			time: e.time,
		})
	}
}

// ProcessBatchEvent represents a simulation step where a forward pass is executed.
// It encapsulates the vLLM scheduler step:
//   - Batch formation (prefill + decode)
//   - Forward pass
//   - Handle processing post forward pass
type ProcessBatchEvent struct {
	time int64 // Scheduled execution time (in ticks)
}

// Timestamp returns the scheduled time of the ProcessBatchEvent.
func (e *ProcessBatchEvent) Timestamp() int64 {
	return e.time
}

// Execute the ProcessBatchEvent
func (e *ProcessBatchEvent) Execute(sim *Simulator) {
	logrus.Infof("<< ProcessBatchEvent at %d ticks", e.time)
	sim.ProcessBatch(e.time)
}
