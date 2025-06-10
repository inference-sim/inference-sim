// sim/event.go
package sim

import "log"

type Event interface {
	Timestamp() int64
	Execute(*Simulator)
}

type ArrivalEvent struct {
	time    int64
	Request *Request
}

func (e *ArrivalEvent) Timestamp() int64 { return e.time }
func (e *ArrivalEvent) Execute(sim *Simulator) {
	log.Printf("<< Arrival: %s at %dµs", e.Request.ID, e.time)
	// if no process batch events are scheduled, create one now
	if sim.ProcessBatchEvent == nil {
		sim.Schedule(&ProcessBatchEvent{
			time: e.time,
		})
	}
}

type ProcessBatchEvent struct {
	time int64
}

func (e *ProcessBatchEvent) Timestamp() int64 { return e.time }
func (e *ProcessBatchEvent) Execute(sim *Simulator) {
	log.Printf("<< ProcessBatchEvent at %dµs", e.time)
	sim.ProcessBatch(e.time)
}
