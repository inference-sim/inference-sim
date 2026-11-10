package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/units"
	"github.com/inference-sim/blis-substrate/load"
)

// stepKernel opens a dense model on one H100-class part, the smallest kernel that prices a step.
func stepKernel(t *testing.T) *Kernel {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	body := `kind: Scenario
name: stepshape
model: qwen3-14b
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-recurrent, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
cluster:
  hardware: h100
  fabric: ib-400g
  nodes: 1
  gpus_per_node: 8
---
kind: Deployment
name: d
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 1, pp: 1, dp: 1}
`
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192, MaxModelLen: 8192}, nil
	}
	k, err := OpenPool(name, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	return k
}

// shape is one rank with one request of n scheduled tokens over a context of 4096.
func shape(n units.Tokens, decodeQueryLen units.Tokens) plan.StepShape {
	return plan.StepShape{
		Mode:           plan.StepMixed,
		Ranks:          []plan.RankStep{{Requests: []plan.RequestStep{{Scheduled: n, Computed: 4096, Prompt: 4096 + n}}}},
		Padded:         []plan.PaddedShape{{Tokens: n}},
		DecodeQueryLen: decodeQueryLen,
	}
}

func priceOf(t *testing.T, k *Kernel, s plan.StepShape) kernel.StepEstimate {
	t.Helper()
	b, err := BatchOf(s, 0)
	if err != nil {
		t.Fatal(err)
	}
	return k.StepTime(b)
}

// SUB-11: StepShape.DecodeQueryLen changes a price. A request that schedules 4 tokens is a
// decode where the engine's threshold is 4 (a verify of 1 + 3 drafts) and a prefill chunk where
// it is 1, and the two are priced with different attention. The same request, on the same kernel,
// costs a different amount.
func TestTheDecodeThresholdTheEngineStatesMovesTheStepPrice(t *testing.T) {
	k := stepKernel(t)
	asDecode := priceOf(t, k, shape(4, 4))
	asPrefill := priceOf(t, k, shape(4, 1))
	if asDecode.NoOverlap == asPrefill.NoOverlap && asDecode.Overlap == asPrefill.Overlap {
		t.Fatalf("a 4-token request costs %v whether the engine's threshold is 4 or 1", asDecode)
	}
	// The zero value is the neutral 1, not a default of the cost model's: it prices as a stated 1.
	if zero := priceOf(t, k, shape(4, 0)); zero.NoOverlap != asPrefill.NoOverlap || zero.Overlap != asPrefill.Overlap {
		t.Errorf("an unstated threshold prices %v, a stated 1 prices %v", zero, asPrefill)
	}
	// A threshold above the request's tokens changes nothing further.
	if more := priceOf(t, k, shape(4, 9)); more.NoOverlap != asDecode.NoOverlap {
		t.Errorf("a threshold of 9 prices %v, of 4 prices %v; both make the request a decode", more, asDecode)
	}
}

// SUB-6: the cost model supplies no threshold of its own. The converter passes the engine's
// figure through, and the engine's neutral value is 1, not the 8 older callers passed.
func TestTheCostModelAppliesNoDecodeThresholdOfItsOwn(t *testing.T) {
	for _, c := range []struct {
		stated units.Tokens
		want   int
	}{{0, 1}, {1, 1}, {4, 4}, {8, 8}} {
		b, err := BatchOf(shape(2, c.stated), 0)
		if err != nil {
			t.Fatal(err)
		}
		if b.DecodeThreshold != c.want {
			t.Errorf("stated %d: batch threshold %d, want %d", c.stated, b.DecodeThreshold, c.want)
		}
	}
	// A negative figure is the step shape's to refuse (StepShape.Validate); the converter reads
	// the contract's floored width and so never hands the kernel a negative threshold.
	if b, err := BatchOf(shape(2, -1), 0); err != nil || b.DecodeThreshold != 1 {
		t.Errorf("a negative threshold became %+v, %v; want the neutral 1", b, err)
	}
	if err := shape(2, -1).Validate(1, 1); err == nil || !strings.Contains(err.Error(), "DecodeQueryLen") {
		t.Errorf("a negative DecodeQueryLen was accepted: %v", err)
	}
	if _, err := BatchOf(shape(2, 1), 3); err == nil {
		t.Error("a rank the step does not have was accepted")
	}
}

// The size of the move, recorded once so a reviewer sees what the field is worth: the same
// 4-token request over a 4096-token context, as a decode (threshold 4) and as a prefill chunk
// (threshold 1).
func TestTheDecodeThresholdsPriceMoveIsRecorded(t *testing.T) {
	k := stepKernel(t)
	a, b := priceOf(t, k, shape(4, 4)), priceOf(t, k, shape(4, 1))
	t.Logf("a 4-token request over 4096 tokens: decode %v / %v, prefill chunk %v / %v (overlap / no overlap)",
		a.Overlap, a.NoOverlap, b.Overlap, b.NoOverlap)
}

// A rank's pad is the tokens the engine launched beyond the ones it scheduled (the data-parallel
// lockstep pad and a graph bucket's), which run the GEMMs and the collectives and no attention
// (Batch.PadTokens). The converter states it and the kernel prices it: a rank padded to 64 tokens
// costs more than one that launches only its 4.
func TestARankPaddedByTheEngineIsPricedAsPadded(t *testing.T) {
	k := stepKernel(t)
	s := shape(4, 1)
	s.Padded = []plan.PaddedShape{{Tokens: 64, Launch: plan.LaunchGraph}}
	b, err := BatchOf(s, 0)
	if err != nil {
		t.Fatal(err)
	}
	if b.PadTokens != 60 || b.Tokens() != 4 {
		t.Fatalf("pad %d tokens %d, want 60 and 4", b.PadTokens, b.Tokens())
	}
	if padded, bare := k.StepTime(b), priceOf(t, k, shape(4, 1)); padded.NoOverlap <= bare.NoOverlap {
		t.Errorf("a rank padded to 64 tokens costs %v, no more than the unpadded %v", padded.NoOverlap, bare.NoOverlap)
	}
	// An unpadded rank (the engine states no padded shape) carries no pad.
	s.Padded = nil
	if b, _ := BatchOf(s, 0); b.PadTokens != 0 {
		t.Errorf("a rank with no padded shape carries %d pad tokens", b.PadTokens)
	}
}

// specKernel opens a model whose graph states a parallel-block drafter (DSpark, K = 7) with the
// speculative settings the engine would resolve, on the chip and layout the perturbation builds.
func specKernel(t *testing.T) *Kernel {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range perturbationScenarios(t, d, "muse-spark-open-dspark") {
		in := c.in
		st := in.Engine
		st.Speculative = &enginesettings.Speculative{Method: "dspark", NumSpecTokens: 7}
		in.Engine = st
		k, err := New(in)
		if err != nil {
			continue
		}
		if k.speculating && k.plan.Draft != nil {
			return k
		}
	}
	t.Skip("no catalog scenario builds a speculating kernel here")
	return nil
}

// SUB-11: DraftPass.Padded changes a price. A rank whose drafter launches 128 tokens for 8
// draft rows pays for the GEMMs and the MoE paths of the padded rows, so its step costs more than
// the rank that launches only its own. DraftPass.Launch moves no price yet: the kernel charges a
// parallel-block drafter's host launches by its own capture rule (draftHost), not by a mode the
// engine states, so Launch is on testdata/unpriced_plan_fields.txt with that reason.
func TestADrafterPaddedByItsGroupIsPricedAsPadded(t *testing.T) {
	k := specKernel(t)
	step := func(padded units.Tokens, launch plan.Launch) plan.StepShape {
		s := shape(8, 8)
		s.Mode = plan.StepVerify
		s.Drafts = plan.DraftShape{Drafted: 7, Verified: 7}
		if padded > 0 {
			s.Drafts.Passes = []plan.DraftPass{{Padded: padded, Launch: launch}}
		}
		return s
	}
	bare := priceOf(t, k, step(0, plan.LaunchEager))
	own := priceOf(t, k, step(7, plan.LaunchGraph))
	padded := priceOf(t, k, step(128, plan.LaunchGraph))
	t.Logf("drafter alone %v / %v, at its own 7 rows %v / %v, padded to 128 %v / %v (overlap / no overlap)",
		bare.Overlap, bare.NoOverlap, own.Overlap, own.NoOverlap, padded.Overlap, padded.NoOverlap)
	// The launch mode is not read: every mode prices alike. This is the reason Launch is listed.
	for _, l := range []plan.Launch{plan.LaunchEager, plan.LaunchPiecewise} {
		if other := priceOf(t, k, step(128, l)); other.NoOverlap != padded.NoOverlap || other.Overlap != padded.Overlap {
			t.Errorf("DraftPass.Launch %q now moves the price (%v against %v): move it to priced_plan_fields.txt",
				l, other, padded)
		}
	}
	if padded.NoOverlap <= own.NoOverlap || padded.Overlap <= own.Overlap {
		t.Errorf("a drafter padded to 128 tokens costs %v, no more than at its own rows %v", padded, own)
	}
	if own.NoOverlap != bare.NoOverlap {
		t.Errorf("a drafter launching only its own rows costs %v, a step stating none %v", own, bare)
	}
}

func TestADrafterPassIsReadFromTheRankItBelongsTo(t *testing.T) {
	s := shape(8, 8)
	s.Mode = plan.StepVerify
	s.Ranks = append(s.Ranks, s.Ranks[0])
	s.Padded = append(s.Padded, s.Padded[0])
	s.Drafts.Passes = []plan.DraftPass{{Padded: 7}, {Padded: 128, Launch: plan.LaunchGraph}}
	b0, err := BatchOf(s, 0)
	if err != nil {
		t.Fatal(err)
	}
	b1, err := BatchOf(s, 1)
	if err != nil {
		t.Fatal(err)
	}
	if b0.DraftPaddedTokens != 7 || b1.DraftPaddedTokens != 128 {
		t.Errorf("drafter rows %d and %d, want 7 and 128 (the pass of each rank)", b0.DraftPaddedTokens, b1.DraftPaddedTokens)
	}
	if one, _ := BatchOf(shape(2, 1), 0); one.DraftPaddedTokens != 0 {
		t.Errorf("a step with no drafter pass carries %d", one.DraftPaddedTokens)
	}
}
