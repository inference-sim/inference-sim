package cluster

import (
	"fmt"

	"github.com/inference-sim/inference-sim/sim"
)

// FlowControlAdmission implements sim.AdmissionPolicy by delegating to a
// per-band gateway queue. When flow control is enabled, this policy replaces
// the legacy admission policy — the queue IS the admission decision.
// Matches llm-d's FlowControlAdmissionController.
//
// Admit() always returns admitted=true. Enqueue outcomes are exposed via
// LastOutcome() and LastShedVictim() for the handler to read.
// Queue rejections are a separate INV-1 bucket from admission rejections —
// they must NOT increment cs.rejectedRequests.
type FlowControlAdmission struct {
	queue       *GatewayQueue
	priorityMap *sim.SLOPriorityMap
	seqCounter  int64 // monotonic, separate from event queue seqIDs (DES single-threaded — safe without atomics)
	lastOutcome EnqueueOutcome
	lastVictim  *sim.Request
}

// NewFlowControlAdmission creates a FlowControlAdmission policy.
// Panics if queue is nil.
func NewFlowControlAdmission(queue *GatewayQueue, priorityMap *sim.SLOPriorityMap) *FlowControlAdmission {
	if queue == nil {
		panic("FlowControlAdmission: queue must not be nil")
	}
	if priorityMap == nil {
		priorityMap = sim.DefaultSLOPriorityMap()
	}
	return &FlowControlAdmission{queue: queue, priorityMap: priorityMap}
}

// Admit enqueues the request into the per-band gateway queue.
// Always returns admitted=true. The caller reads LastOutcome()/LastShedVictim()
// for queue-level outcome handling.
// INV-9: reads only SLOClass and TenantID — does NOT read OutputTokens.
func (fc *FlowControlAdmission) Admit(req *sim.Request, state *sim.RouterState) (bool, string) {
	fc.lastVictim = nil
	req.GatewayEnqueueTime = state.Clock
	fc.seqCounter++
	outcome, victim := fc.queue.Enqueue(req, fc.seqCounter)
	fc.lastOutcome = outcome

	switch outcome {
	case Rejected:
		req.GatewayEnqueueTime = 0
		return true, "flow-control-queue-rejected"
	case ShedVictim:
		if victim != nil {
			victim.GatewayEnqueueTime = 0
			fc.lastVictim = victim
		}
		return true, "flow-control-enqueued"
	case Enqueued:
		return true, "flow-control-enqueued"
	default:
		panic(fmt.Sprintf("FlowControlAdmission: unhandled EnqueueOutcome %d", outcome))
	}
}

// Queue returns the underlying gateway queue for dispatch and metrics access.
func (fc *FlowControlAdmission) Queue() *GatewayQueue { return fc.queue }

// LastOutcome returns the enqueue outcome from the most recent Admit() call.
func (fc *FlowControlAdmission) LastOutcome() EnqueueOutcome { return fc.lastOutcome }

// LastShedVictim returns the evicted request from the most recent Admit() call, or nil.
func (fc *FlowControlAdmission) LastShedVictim() *sim.Request { return fc.lastVictim }
