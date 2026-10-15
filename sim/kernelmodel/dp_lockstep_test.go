package kernelmodel

import (
	"testing"

	"github.com/inference-sim/inference-sim/sim"
)

const dpLockstepScenario = "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml"

func decodes(n int, ctx int64) []*sim.Request {
	out := make([]*sim.Request, n)
	for i := range out {
		out[i] = &sim.Request{InputTokens: make([]sim.TokenID, ctx), ProgressIndex: ctx, NumNewTokens: 1}
	}
	return out
}

// A lockstep rank is priced at the group's largest token count when the step runs under a
// CUDA graph (vLLM's DP padding), so load on one rank raises the other's price; an idle rank
// runs a dummy batch, never cheaper than an empty step; and past the capture ceiling nothing
// is padded, so each rank costs its own batch.
func TestStepTimesAcrossDP_PadsToTheLargestRank(t *testing.T) {
	m := open(t, dpLockstepScenario)
	if m.pad.ceiling <= 0 {
		t.Fatalf("scenario %s runs CUDA graphs; the capture ceiling must be read from the kernel", dpLockstepScenario)
	}
	light := decodes(2, 4096)
	alone := m.StepTimesAcrossDP([][]*sim.Request{light, light})
	loaded := m.StepTimesAcrossDP([][]*sim.Request{light, decodes(m.pad.ceiling/2, 4096)})
	if loaded[0] <= alone[0] {
		t.Errorf("rank 0's price did not rise with rank 1's load: %d alone, %d beside a loaded rank", alone[0], loaded[0])
	}
	if alone[0] != m.StepTime(light) {
		t.Errorf("equal ranks need no padding: got %d, own price %d", alone[0], m.StepTime(light))
	}

	idle := m.StepTimesAcrossDP([][]*sim.Request{nil, light})
	if idle[0] < m.StepTime(nil) {
		t.Errorf("an idle rank's dummy batch priced %d, below an empty step %d", idle[0], m.StepTime(nil))
	}

	prefill := []*sim.Request{{InputTokens: make([]sim.TokenID, 4*m.pad.ceiling), NumNewTokens: 2 * m.pad.ceiling}}
	eager := m.StepTimesAcrossDP([][]*sim.Request{light, prefill})
	if eager[0] != m.StepTime(light) || eager[1] != m.StepTime(prefill) {
		t.Errorf("past the capture ceiling each rank must cost its own batch: got %v, own %d/%d",
			eager, m.StepTime(light), m.StepTime(prefill))
	}
}
