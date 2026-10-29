// Moved from inference-sim's sim/simulator.go and sim/event.go at 02a7c6c8, when the
// compatibility engine's own host loop (sim.Simulator) was removed: every instance is an
// engine-driven one (EngineDriver), whose events this queue orders (phase 2c).

package sim

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
	Execute()
}

// eventEntry wraps an Event with a sequence ID for deterministic ordering.
// The EventQueue orders by (Timestamp, Priority, seqID), matching the
// cluster event queue's scheme. seqID breaks ties within same-type same-timestamp events.
type eventEntry struct {
	event Event
	seqID int64
}

// EventQueue implements heap.Interface and orders events by (timestamp, priority, seqID).
// This ensures deterministic same-tick event ordering (INV-6 improvement).
type EventQueue []eventEntry

func (eq EventQueue) Len() int { return len(eq) }

func (eq EventQueue) Less(i, j int) bool {
	if eq[i].event.Timestamp() != eq[j].event.Timestamp() {
		return eq[i].event.Timestamp() < eq[j].event.Timestamp()
	}
	if eq[i].event.Priority() != eq[j].event.Priority() {
		return eq[i].event.Priority() < eq[j].event.Priority()
	}
	return eq[i].seqID < eq[j].seqID
}

func (eq EventQueue) Swap(i, j int) { eq[i], eq[j] = eq[j], eq[i] }

func (eq *EventQueue) Push(x any) {
	*eq = append(*eq, x.(eventEntry))
}

func (eq *EventQueue) Pop() any {
	old := *eq
	n := len(old)
	item := old[n-1]
	*eq = old[:n-1]
	return item
}
