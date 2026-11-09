package workload

import "github.com/inference-sim/inference-sim/sim"

// CompletionHandler is the closed-loop seam consumed by the DES replay driver
// loop: given a request that just reached a terminal
// state at tick, it returns the follow-up requests to inject next (empty when
// the session/pool produces nothing further).
//
// Both *SessionManager (round-level closed loop) and *SessionPoolDriver
// (session-level fixed pool on top of it) satisfy this.
type CompletionHandler interface {
	OnComplete(req *sim.Request, tick int64) []*sim.Request
}

// Compile-time assertions.
var (
	_ CompletionHandler = (*SessionManager)(nil)
	_ CompletionHandler = (*SessionPoolDriver)(nil)
)
