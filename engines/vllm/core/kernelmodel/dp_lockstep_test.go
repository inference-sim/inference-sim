// Moved from inference-sim/sim/kernelmodel/dp_lockstep_test.go at ca532abca209af41425c007974a5f3ef336f5981.

package kernelmodel

import (
	"fmt"
	"os"
	"strings"
	"testing"

	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"
	"pgregory.net/rapid"

	sim "github.com/inference-sim/inference-sim/engines/vllm/core/runconfig"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
)

const dpLockstepScenario = "minimax-m2.5-b200-fp4-vllm-tp2-ep4-dp2.yaml"

// priceInGroup prices batch as a rank whose group hands the experts groupTokens tokens in all.
func priceInGroup(m *Model, batch []*sim.Request, groupTokens int) int64 {
	b := m.batchOf(batch)
	b.GroupTokens = groupTokens
	return max(1, ticks(m.k.StepTime(b).NoOverlap))
}

func decodes(n int, ctx int64) []*sim.Request {
	out := make([]*sim.Request, n)
	for i := range out {
		out[i] = &sim.Request{InputTokens: make([]sim.TokenID, ctx), ProgressIndex: ctx, NumNewTokens: 1}
	}
	return out
}

// acrossDP prices one lockstep step through StepTimesAcrossDPOfShapes, the path the engine
// ships (vllmengine.go's kernelPricer.StepTimesAcrossDP calls StepTimesAcrossDPOfStep, which
// StepTimesAcrossDPOfShapes forwards to unconditionally). The dpLockstepScenario pool runs no
// speculative decoding, so shapeOf's plain translation is exactly what batchOf would have
// produced for these requests.
func acrossDP(m *Model, batches [][]*sim.Request) []int64 {
	ranks := make([][]ReqShape, len(batches))
	for i, b := range batches {
		shapes := make([]ReqShape, len(b))
		for j, r := range b {
			shapes[j] = shapeOf(r)
		}
		ranks[i] = shapes
	}
	return m.StepTimesAcrossDPOfShapes(ranks)
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
	alone := acrossDP(m, [][]*sim.Request{light, light})
	loaded := acrossDP(m, [][]*sim.Request{light, decodes(m.pad.ceiling/2, 4096)})
	if loaded[0] <= alone[0] {
		t.Errorf("rank 0's price did not rise with rank 1's load: %d alone, %d beside a loaded rank", alone[0], loaded[0])
	}
	if alone[0] != m.StepTime(light) {
		t.Errorf("equal ranks need no padding: got %d, own price %d", alone[0], m.StepTime(light))
	}

	idle := acrossDP(m, [][]*sim.Request{nil, light})
	if idle[0] < m.StepTime(nil) {
		t.Errorf("an idle rank's dummy batch priced %d, below an empty step %d", idle[0], m.StepTime(nil))
	}

	prefill := []*sim.Request{{InputTokens: make([]sim.TokenID, 4*m.pad.ceiling), NumNewTokens: 2 * m.pad.ceiling}}
	eager := acrossDP(m, [][]*sim.Request{light, prefill})
	if eager[0] <= m.StepTime(light) || eager[1] >= m.StepTime(prefill) {
		t.Errorf("past the capture ceiling a rank keeps its own attention but shares the experts' tokens with the group: got %v, own %d/%d",
			eager, m.StepTime(light), m.StepTime(prefill))
	}
	alike := acrossDP(m, [][]*sim.Request{prefill, prefill})
	if alike[0] != m.StepTime(prefill) || alike[1] != m.StepTime(prefill) {
		t.Errorf("ranks running alike must cost one rank's price: got %v, own %d", alike, m.StepTime(prefill))
	}
	// Unpadded, an idle rank still runs the dummy batch -- one uniform decode, all of it pad --
	// which costs more than a step with no forward at all.
	idleEager := acrossDP(m, [][]*sim.Request{nil, prefill})
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
	editPoolEngine(t, &in.Deployment.Pools[0], func(e *engineconfig.Engine) { e.CUDAGraphMode = mode })
	m, err := fromInputs(in)
	if err != nil {
		t.Fatalf("New with cudagraph_mode %q: %v", mode, err)
	}
	return m, nil
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
	unpadded := priceInGroup(fdo, light, fdo.batchOf(light).PaddedTokens()+fdo.batchOf(mixed).PaddedTokens())
	if got := acrossDP(fdo, [][]*sim.Request{light, mixed}); got[0] != unpadded {
		t.Errorf("FULL_DECODE_ONLY pads a mixed group step: rank 0 priced %d, unpadded %d", got[0], unpadded)
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

// Laws of a lockstep group's pricing: no rank costs less than its own batch beside idle ranks
// (an idle rank no less than an empty step); a rank that only runs the dummy batch never lowers the
// group's step; and the price is equivariant under permuting the ranks.
func TestStepTimesAcrossDP_Laws(t *testing.T) {
	m := open(t, dpLockstepScenario)
	rapid.Check(t, func(rt *rapid.T) {
		n := rapid.IntRange(2, 4).Draw(rt, "ranks")
		batches := make([][]*sim.Request, n)
		for i := range batches {
			batches[i] = drawRankBatch(rt, fmt.Sprintf("r%d", i), m.pad.ceiling)
		}
		out := acrossDP(m, batches)
		maxOf := func(d []int64) int64 {
			best := int64(0)
			for _, v := range d {
				best = max(best, v)
			}
			return best
		}
		for i, b := range batches {
			if alone := priceInGroup(m, b, m.batchOf(b).PaddedTokens()); out[i] < alone {
				rt.Fatalf("rank %d priced %d below its own batch %d", i, out[i], alone)
			}
		}
		withIdle := acrossDP(m, append(append([][]*sim.Request{}, batches...), nil))
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
		pout := acrossDP(m, permuted)
		for i, j := range perm {
			if pout[i] != out[j] {
				rt.Fatalf("permuting the ranks changed rank %d's price: %d vs %d", j, pout[i], out[j])
			}
		}
	})
}

// The routed experts see the sum of every rank's tokens, not dp times the rank's own: a rank
// that prefills beside light ranks hands the experts far fewer tokens than dp copies of itself,
// and each light rank still pays for the prefill rank's tokens. So a rank's price is monotone
// in the others' load, and the same unpadded step costs less than one where every rank runs the
// prefill.
func TestStepTimesAcrossDP_ExpertsSeeTheGroupsSummedTokens(t *testing.T) {
	m := open(t, dpLockstepScenario)
	light := decodes(2, 4096)
	prefill := []*sim.Request{{InputTokens: make([]sim.TokenID, 4*m.pad.ceiling), NumNewTokens: 2 * m.pad.ceiling}}
	longer := []*sim.Request{{InputTokens: make([]sim.TokenID, 4*m.pad.ceiling), NumNewTokens: 3 * m.pad.ceiling / 2}}

	mixed := acrossDP(m, [][]*sim.Request{light, prefill})
	both := acrossDP(m, [][]*sim.Request{prefill, prefill})
	if mixed[1] >= both[1] {
		t.Errorf("the prefill rank beside a light rank priced %d, no less than beside an equal prefill %d", mixed[1], both[1])
	}
	beside := acrossDP(m, [][]*sim.Request{longer, prefill})
	if mixed[1] >= beside[1] || beside[1] >= both[1] {
		t.Errorf("a rank's price must rise with the other rank's tokens: %d beside light, %d beside a smaller prefill, %d beside an equal one",
			mixed[1], beside[1], both[1])
	}
	if mixed[0] > mixed[1] {
		t.Errorf("a light rank priced %d, above the prefill rank %d in the same step: the experts' work does not depend on which rank sent the tokens",
			mixed[0], mixed[1])
	}
}
