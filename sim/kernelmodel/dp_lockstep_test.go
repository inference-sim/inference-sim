package kernelmodel

import (
	"fmt"
	"os"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/blis-latency-kernel"
	"pgregory.net/rapid"

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
	// Unpadded, an idle rank still runs the dummy batch -- one uniform decode, all of it pad --
	// which costs more than a step with no forward at all.
	idleEager := m.StepTimesAcrossDP([][]*sim.Request{nil, prefill})
	if empty := m.StepTime(nil); idleEager[0] <= empty {
		t.Errorf("an idle rank in an unpadded step priced %d, no more than an empty step %d", idleEager[0], empty)
	}
}

// openWithGraphMode opens the lockstep scenario with its pool's cudagraph_mode replaced.
func openWithGraphMode(t *testing.T, mode string) (*Model, error) {
	t.Helper()
	in, err := latencykernel.OpenInputs(dpLockstepScenario, repos(), 0)
	if err != nil {
		t.Fatal(err)
	}
	in.Deployment.Pools[0].Engine.CUDAGraphMode = mode
	k, err := latencykernel.New(in)
	if err != nil {
		t.Fatalf("New with cudagraph_mode %q: %v", mode, err)
	}
	return newModel(k, identityOf(in))
}

// Every catalogued scenario resolves a capture ceiling exactly when the kernel runs CUDA
// graphs; one that runs graphs with no readable ceiling is refused at construction rather
// than priced unpadded. NONE pads nothing; FULL_DECODE_ONLY pads only uniform decodes.
func TestDPPadding_CeilingIffGraphs(t *testing.T) {
	entries, err := os.ReadDir(repos().Scenarios)
	if err != nil {
		t.Fatal(err)
	}
	opened := 0
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		m, err := Open(e.Name(), repos())
		if err != nil {
			if strings.Contains(err.Error(), "capture ceiling") {
				t.Errorf("%s: %v", e.Name(), err)
			}
			continue
		}
		opened++
		decode, other := m.k.(graphResolver).GraphModes()
		graphs := decode != "NONE" || other != "NONE"
		if graphs != (m.pad.ceiling > 0) {
			t.Errorf("%s: graph modes %s/%s but capture ceiling %d", e.Name(), decode, other, m.pad.ceiling)
		}
	}
	if opened == 0 {
		t.Fatal("no catalogued scenario opened")
	}
	none, err := openWithGraphMode(t, "NONE")
	if err != nil || none.pad.ceiling != 0 {
		t.Errorf("cudagraph_mode NONE: ceiling %v, err %v; want no padding", none, err)
	}
	fdo, err := openWithGraphMode(t, "FULL_DECODE_ONLY")
	if err != nil || fdo.pad.ceiling <= 0 || !fdo.pad.decodeOnly {
		t.Fatalf("FULL_DECODE_ONLY: %+v, err %v; want a ceiling and decode-only padding", fdo, err)
	}
	light := decodes(2, 4096)
	mixed := []*sim.Request{{InputTokens: make([]sim.TokenID, 256), NumNewTokens: 64}}
	if got := fdo.StepTimesAcrossDP([][]*sim.Request{light, mixed}); got[0] != fdo.StepTime(light) {
		t.Errorf("FULL_DECODE_ONLY pads a mixed group step: rank 0 priced %d, own %d", got[0], fdo.StepTime(light))
	}
}

func drawRankBatch(t *rapid.T, label string, ceiling int) []*sim.Request {
	n := rapid.IntRange(0, 6).Draw(t, label+"-n")
	out := make([]*sim.Request, n)
	for i := range out {
		ctx := rapid.Int64Range(1, 8192).Draw(t, label+"-ctx")
		if rapid.Bool().Draw(t, label+"-prefill") {
			chunk := rapid.IntRange(16, 2*ceiling).Draw(t, label+"-chunk")
			out[i] = &sim.Request{InputTokens: make([]sim.TokenID, ctx+int64(chunk)), ProgressIndex: ctx, NumNewTokens: chunk}
		} else {
			out[i] = &sim.Request{InputTokens: make([]sim.TokenID, ctx), ProgressIndex: ctx, NumNewTokens: 1}
		}
	}
	return out
}

// Laws of a lockstep group's pricing: no rank costs less than its own batch alone (an idle
// rank no less than an empty step); a rank that only runs the dummy batch never lowers the
// group's step; and the price is equivariant under permuting the ranks.
func TestStepTimesAcrossDP_Laws(t *testing.T) {
	m := open(t, dpLockstepScenario)
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 4).Draw(rt, "ranks")
		batches := make([][]*sim.Request, n)
		for i := range batches {
			batches[i] = drawRankBatch(rt, fmt.Sprintf("r%d", i), m.pad.ceiling)
		}
		out := m.StepTimesAcrossDP(batches)
		maxOf := func(d []int64) int64 {
			best := int64(0)
			for _, v := range d {
				best = max(best, v)
			}
			return best
		}
		for i, b := range batches {
			if own := m.StepTime(b); out[i] < own {
				rt.Fatalf("rank %d priced %d below its own batch %d", i, out[i], own)
			}
		}
		withIdle := m.StepTimesAcrossDP(append(append([][]*sim.Request{}, batches...), nil))
		if maxOf(withIdle) < maxOf(out) {
			rt.Fatalf("adding a dummy-only rank lowered the step from %d to %d", maxOf(out), maxOf(withIdle))
		}
		perm := rapid.Permutation(func() []int {
			p := make([]int, n)
			for i := range p {
				p[i] = i
			}
			return p
		}()).Draw(rt, "perm")
		permuted := make([][]*sim.Request, n)
		for i, j := range perm {
			permuted[i] = batches[j]
		}
		pout := m.StepTimesAcrossDP(permuted)
		for i, j := range perm {
			if pout[i] != out[j] {
				rt.Fatalf("permuting the ranks changed rank %d's price: %d vs %d", j, pout[i], out[j])
			}
		}
	})
}
