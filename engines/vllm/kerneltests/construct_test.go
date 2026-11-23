// Moved from blis-latency-kernel/construct_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	"testing"
)

func TestConstructsFromCommittedArtifacts(t *testing.T) {
	k := fixture(t, "minimax-m25-h200-ep8.yaml")
	r := k.Resolved()
	t.Logf("ep width %d, allreduce %s, sp-moe %v, overrides %d",
		r.ExpertParallelWidth, r.AllReduceBackend, r.SequenceParallelMoE,
		len(r.Overrides))
	measured, total, assumed := k.Evidence()
	t.Logf("coefficients: %d resolved, %d evidenced, %d assumed", total, measured,
		len(assumed))
}
