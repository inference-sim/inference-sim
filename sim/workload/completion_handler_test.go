package workload

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

// Compile-time + behavioral proof that both closed-loop drivers satisfy
// CompletionHandler, so observe's serializer can hold either.
func TestCompletionHandler_SatisfiedByBothDrivers(t *testing.T) {
	var _ CompletionHandler = (*SessionManager)(nil)
	var _ CompletionHandler = (*SessionPoolDriver)(nil)

	// Behavioral: a non-session request returns no follow-ups through the interface.
	var h CompletionHandler = NewSessionManager(nil)
	if got := h.OnComplete(&sim.Request{}, 0); got != nil {
		t.Errorf("OnComplete(non-session) = %v, want nil", got)
	}
}
