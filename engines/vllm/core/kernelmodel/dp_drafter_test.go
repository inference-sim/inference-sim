package kernelmodel

import (
	"testing"

	"github.com/inference-sim/inference-sim/engines/vllm/core/engineconfig"
	"github.com/inference-sim/inference-sim/substrate/contracts/kernel"
	latencykernel "github.com/inference-sim/inference-sim/substrate/kernel"
)

// dsparkDP is the committed two-node DSpark scenario at data parallelism 8 (DEP8), K = 5.
const dsparkDP = "deepseek-v4.1-flash-h100-2n-dep8-dspark5.yaml"

func openDSparkDP(t *testing.T) *Model {
	t.Helper()
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	m, err := OpenPool(dsparkDP, r, 0)
	if err != nil {
		t.Fatalf("OpenPool(%s): %v", dsparkDP, err)
	}
	if m.specTokens <= 0 {
		t.Fatalf("%s does not speculate", dsparkDP)
	}
	return m
}

// verifies is n decode rows at verify width, each with a context of ctx tokens.
func verifies(m *Model, n int, ctx int) []ReqShape {
	out := make([]ReqShape, n)
	for i := range out {
		out[i] = ReqShape{Computed: ctx, Scheduled: 1 + m.specTokens, PromptLen: 1000}
	}
	return out
}

// A drafter's data-parallel width is the group's largest request count times its query rows,
// on every rank, and nothing without a group to agree it in (vLLM's dp_sync_state is None at one
// rank), without a drafter, or on a step whose drafters are skipped.
func TestDrafterPaddingIsTheGroupsWidth(t *testing.T) {
	m := openDSparkDP(t)
	k := m.specTokens
	reqs := func(n int) kernel.Batch { return kernel.Batch{Reqs: make([]kernel.ReqShape, n)} }
	for _, tt := range []struct {
		name  string
		ranks []kernel.Batch
		want  []int
	}{
		{"parallel-block: K rows per request of the largest rank",
			[]kernel.Batch{reqs(8), reqs(1), reqs(3)}, []int{8 * k, 8 * k, 8 * k}},
		{"an idle rank is padded like the others", []kernel.Batch{reqs(4), reqs(0)}, []int{4 * k, 4 * k}},
		{"one rank has no group", []kernel.Batch{reqs(4)}, []int{0}},
		{"nothing to launch", []kernel.Batch{reqs(0), reqs(0)}, []int{0, 0}},
	} {
		m.padDrafters(tt.ranks)
		for i, b := range tt.ranks {
			if b.DraftPaddedTokens != tt.want[i] {
				t.Errorf("%s: rank %d states %d drafter rows, want %d", tt.name, i, b.DraftPaddedTokens, tt.want[i])
			}
			if b.PadTokens != 0 {
				t.Errorf("%s: rank %d: the drafter's width moved the target's pad to %d", tt.name, i, b.PadTokens)
			}
		}
	}
	skipped := []kernel.Batch{{Reqs: make([]kernel.ReqShape, 4), SkipDrafts: true}, {Reqs: make([]kernel.ReqShape, 1), SkipDrafts: true}}
	m.padDrafters(skipped)
	for i, b := range skipped {
		if b.DraftPaddedTokens != 0 {
			t.Errorf("a skipped step: rank %d states %d drafter rows, want none", i, b.DraftPaddedTokens)
		}
	}

	// An autoregressive drafter (MTP) runs its multi-step decode at one row per request.
	mtp := openSpeculative(t, "glm-5-h200-fp8-sglang-tp8.yaml", 3)
	ranks := []kernel.Batch{reqs(6), reqs(2)}
	mtp.padDrafters(ranks)
	for i, b := range ranks {
		if b.DraftPaddedTokens != 6 {
			t.Errorf("MTP: rank %d states %d drafter rows, want the group's 6 requests x 1 row", i, b.DraftPaddedTokens)
		}
	}
	plain := openSpeculative(t, "glm-5-h200-fp8-sglang-tp8.yaml", 0)
	ranks = []kernel.Batch{reqs(6), reqs(2)}
	plain.padDrafters(ranks)
	for i, b := range ranks {
		if b.DraftPaddedTokens != 0 {
			t.Errorf("no speculation: rank %d states %d drafter rows", i, b.DraftPaddedTokens)
		}
	}
}

// openDSparkDPGraphed is dsparkDP on an all-to-all backend whose layout runs CUDA graphs, so
// the target pads its lockstep ranks to the group's largest token count.
func openDSparkDPGraphed(t *testing.T) *Model {
	t.Helper()
	r := repos()
	r.Scenarios = "../../testdata/scenarios"
	in, err := latencykernel.OpenInputs(dsparkDP, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	editPoolEngine(t, &in.Deployment.Pools[0], func(e *engineconfig.Engine) {
		e.All2AllBackend, e.MaxNumBatchedTokens = "deepep_low_latency", 1024
	})
	m, err := fromInputs(in)
	if err != nil {
		t.Fatal(err)
	}
	if m.pad.ceiling <= 0 {
		t.Fatalf("the layout runs eagerly: %+v", m.pad)
	}
	return m
}

// The lighter rank of a lockstep group pays the group's drafter padding, and the target's pad
// and the drafter's are independent. Each rank is priced as the kernel prices the batch that
// states both: the target padded to the group's largest token count (when the layout graphs
// it) and the drafter to the group's largest request count times K, two different widths. A
// rank that paid only its own drafter rows would cost less.
func TestALighterRankPaysTheGroupsDrafterPadding(t *testing.T) {
	for _, tt := range []struct {
		name         string
		open         func(*testing.T) *Model
		targetPadded bool
	}{
		{"target running eagerly: only the drafter is padded", openDSparkDP, false},
		{"target under a graph: both padded, to different widths", openDSparkDPGraphed, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.open(t)
			k := m.specTokens
			const heavyDecodes = 12
			heavy, light := verifies(m, heavyDecodes, 3000), verifies(m, 1, 3000)
			got := m.StepTimesAcrossDPOfStep([][]ReqShape{heavy, light}, false)

			widest := heavyDecodes * (1 + k) // target tokens of the heavy rank
			targetPad := func(own int) int {
				if tt.targetPadded {
					return widest - own
				}
				return 0
			}
			groupTokens := widest + 1 + k // each rank's own tokens
			if tt.targetPadded {
				groupTokens = 2 * widest // both ranks padded to the widest
			}
			price := func(reqs []ReqShape, own, draftRows int) int64 {
				b := kernel.Batch{Reqs: reqs, DecodeThreshold: m.decodeThreshold, SMBudget: m.smBudget,
					PadTokens: targetPad(own), GroupTokens: groupTokens, DraftPaddedTokens: draftRows}
				return max(1, ticks(m.k.StepTime(b).NoOverlap))
			}
			groupRows := heavyDecodes * k
			if want := price(light, 1+k, groupRows); got[1] != want {
				t.Errorf("light rank: %d ticks, want %d (the kernel's price of a rank padded to the group's "+
					"%d drafter rows and %d target tokens)", got[1], want, groupRows, targetPad(1+k)+1+k)
			}
			if own := price(light, 1+k, 0); got[1] <= own {
				t.Errorf("light rank: %d ticks, not above %d at its own drafter rows: it did not pay the "+
					"group's drafter padding", got[1], own)
			}
			if want := price(heavy, widest, groupRows); got[0] != want {
				t.Errorf("heavy rank: %d ticks, want %d", got[0], want)
			}
			if tt.targetPadded && groupRows == widest {
				t.Fatal("the fixture gives the two pads one width")
			}
		})
	}
}

// An idle rank runs the drafter's dummy propose so the group stays in step: it pays the group's
// drafter width on top of its dummy target batch, where an idle rank of a group without a
// drafter pays nothing for one. A step whose drafters are skipped launches no such pass.
func TestAnIdleRankLaunchesTheGroupsDrafter(t *testing.T) {
	m := openDSparkDP(t)
	busy := verifies(m, 8, 3000)
	idle := m.StepTimesAcrossDPOfStep([][]ReqShape{busy, nil}, false)
	skipped := m.StepTimesAcrossDPOfStep([][]ReqShape{busy, nil}, true)
	if idle[1] <= skipped[1] {
		t.Errorf("idle rank: %d ticks with the drafter, %d with its drafts skipped: the dummy propose "+
			"is not priced", idle[1], skipped[1])
	}
	// The same idle rank beside a lighter busy rank launches a narrower drafter.
	narrow := m.StepTimesAcrossDPOfStep([][]ReqShape{verifies(m, 2, 3000), nil}, false)
	if narrow[1] >= idle[1] {
		t.Errorf("idle rank beside 2 requests: %d ticks, not below %d beside 8: the drafter width "+
			"does not follow the group", narrow[1], idle[1])
	}
}
