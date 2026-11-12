// The host's admission stage: the queue requests wait in before the engine admits them, whose
// shape the engine states in its plan (plan.HostPlan.Admission) and whose demands the cost model
// prices (env.Environment.HostDemand). The host schedules requests on it knowing nothing of what
// the servers are.

package sim

import (
	"fmt"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
)

// AdmissionStage is one endpoint's admission queue (INV-R1): q.Queues() independent FIFO queues,
// each served by q.WorkersPerServer servers. A request's demand on it is the time it holds a
// server (the cost model's HostDemand.Service); the stage only decides when that work can start.
//
// Requests join the queues in turn, in arrival order, which is a deterministic stand-in for how an
// endpoint spreads connections over its processes. Within one queue a request takes the
// earliest-free server, in arrival order (FIFO, an M/G/c queue).
type AdmissionStage struct {
	// freeAt[s][w] is when queue s's server w finishes its last request.
	freeAt [][]int64
	next   int
	// admitted is the number of requests that took a server; waitTicks their summed wait.
	admitted  int64
	waitTicks int64
}

// NewAdmissionStage builds the stage a plan's admission queue states. It refuses a queue with
// no servers: that is no stage (plan.AdmissionQueue.Enabled).
func NewAdmissionStage(q plan.AdmissionQueue) (*AdmissionStage, error) {
	if err := q.Validate(); err != nil {
		return nil, fmt.Errorf("NewAdmissionStage: %w", err)
	}
	if !q.Enabled() {
		return nil, fmt.Errorf("NewAdmissionStage: %+v has no servers", q)
	}
	s := &AdmissionStage{freeAt: make([][]int64, q.Queues())}
	for i := range s.freeAt {
		s.freeAt[i] = make([]int64, q.WorkersPerServer)
	}
	return s, nil
}

// Admit queues a request that reached the endpoint at arrival and holds a server for service
// ticks, and returns when it releases the server: max(arrival, the server's free time) +
// service. Callers admit in arrival order (the event loop's), which makes each queue FIFO.
func (s *AdmissionStage) Admit(arrival, service int64) int64 {
	servers := s.freeAt[s.next]
	s.next = (s.next + 1) % len(s.freeAt)
	w := 0
	for k := 1; k < len(servers); k++ {
		if servers[k] < servers[w] {
			w = k
		}
	}
	start := max(arrival, servers[w])
	done := start + service
	servers[w] = done
	s.admitted++
	s.waitTicks += start - arrival
	return done
}

// Admitted is the number of requests that have taken a server.
func (s *AdmissionStage) Admitted() int64 { return s.admitted }

// WaitTicks is the summed time requests waited for a server.
func (s *AdmissionStage) WaitTicks() int64 { return s.waitTicks }
