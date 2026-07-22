package workload

import "github.com/inference-sim/inference-sim/sim"

// CompletionHandler is the closed-loop seam consumed by driver loops (the DES
// replay path and blis observe): given a request that just reached a terminal
// state at tick, it returns the follow-up requests to inject next (empty when
// the session/pool produces nothing further).
//
// Both *SessionManager (round-level closed loop) and *SessionPoolDriver
// (session-level fixed pool on top of it) satisfy this. Sharing one interface
// keeps blis replay and blis observe driving identical closed-loop semantics
// (R4) so calibration compares like with like.
type CompletionHandler interface {
	OnComplete(req *sim.Request, tick int64) []*sim.Request
}

// Compile-time assertions.
var (
	_ CompletionHandler = (*SessionManager)(nil)
	_ CompletionHandler = (*SessionPoolDriver)(nil)
)
