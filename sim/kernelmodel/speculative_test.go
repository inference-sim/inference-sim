package kernelmodel

import (
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"github.com/inference-sim/blis-schemas/spec/deployment"

	"github.com/inference-sim/inference-sim/sim"
)

// openSpeculative opens a committed scenario with its pool given a draft configuration of k
// tokens, through the same New the kernel's own OpenPool ends in.
func openSpeculative(t *testing.T, scenario string, k int) *Model {
	t.Helper()
	in, err := latencykernel.OpenInputs(scenario, repos(), 0)
	if err != nil {
		t.Fatal(err)
	}
	if k > 0 {
		in.Deployment.Pools[0].Engine.Speculative = &deployment.Speculative{Method: "mtp", NumSpecTokens: k}
	}
	kn, err := latencykernel.New(in)
	if err != nil {
		t.Fatalf("New with %d draft tokens: %v", k, err)
	}
	return newModel(kn, identityOf(in))
}

// Under speculation a decode costs its verify width -- 1 + the draft length -- whatever it
// later advanced by: the engine runs every draft position, and the blis-schemas contract
// states Scheduled that way. BLIS's NumNewTokens is the advance (1 + accepted), so passing
// it through would price an accepted-zero step as a plain decode.
//
// Two laws: the adapter's price for a decode batch equals the kernel's own price at
// Scheduled = 1 + K regardless of the advance, and a prefill chunk is untouched.
func TestASpeculativeDecodeIsPricedAtItsVerifyWidth(t *testing.T) {
	const scenario = "glm-5-h200-fp8-sglang-tp8.yaml"
	for _, k := range []int{1, 3} {
		m := openSpeculative(t, scenario, k)
		for _, advance := range []int{1, k + 1} {
			decode := &sim.Request{InputTokens: make([]sim.TokenID, 2048), ProgressIndex: 2100,
				NumNewTokens: advance}
			got := m.StepTime([]*sim.Request{decode})
			b := m.batchOf([]*sim.Request{decode})
			want := max(1, m.Kernel().StepTime(b).NoOverlap.Microseconds())
			if b.Reqs[0].Scheduled != 1+k {
				t.Errorf("k=%d advance=%d: decode scheduled %d, want verify width %d", k, advance,
					b.Reqs[0].Scheduled, 1+k)
			}
			if got != want {
				t.Errorf("k=%d advance=%d: adapter priced %d, kernel at the verify width %d", k, advance, got, want)
			}
		}
		prefill := &sim.Request{InputTokens: make([]sim.TokenID, 2048), NumNewTokens: 512}
		if s := m.batchOf([]*sim.Request{prefill}).Reqs[0].Scheduled; s != 512 {
			t.Errorf("k=%d: a prefill chunk scheduled %d, want its own 512", k, s)
		}
	}
	// Without a draft the decode is priced as one token, as before.
	plain := openSpeculative(t, scenario, 0)
	decode := &sim.Request{InputTokens: make([]sim.TokenID, 2048), ProgressIndex: 2100, NumNewTokens: 1}
	if s := plain.batchOf([]*sim.Request{decode}).Reqs[0].Scheduled; s != 1 {
		t.Errorf("no draft: decode scheduled %d, want 1", s)
	}
}
