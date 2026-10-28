// Moved from blis-latency-kernel/step_behaviour_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"strings"
	"testing"
	"time"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// Step-time behaviour, as relations between steps, deployments and graphs.

func prefillOf(tokens, prefix int) kernel.Batch {
	return kernel.Batch{DecodeThreshold: 1, Reqs: []kernel.ReqShape{
		{Scheduled: tokens, Computed: prefix, PromptLen: prefix + tokens}}}
}

// A recurrent prefill runs the chunked scan, not the decode kernel's sequential one: the
// smaller GLM-5.3-Flash, 34 of whose layers are KDA, prefills no slower than GLM-5 on the same
// deployment, where pricing its prefill tokens at the decode scan's rate made it over twice as
// slow.
func TestRecurrentPrefillIsNotPricedAtTheDecodeRate(t *testing.T) {
	glm5 := fixture(t, "glm5-h200-tp8.yaml")
	in := fixtureInputs(t, "glm5-h200-tp8.yaml")
	withModel(t, &in, "glm-5.3-flash")
	flash := build(t, in)
	b := prefillOf(8192, 0)
	if f, g := flash.StepTime(b).NoOverlap, glm5.StepTime(b).NoOverlap; f > g {
		t.Errorf("an 8K GLM-5.3-Flash prefill takes %v, slower than the larger GLM-5's %v", f, g)
	}
}

// Running a step with no graph launches every kernel, so it is never cheaper on the host
// than replaying captured pieces: NONE >= PIECEWISE >= FULL for any batch.
func TestEagerLaunchesCostAtLeastAReplay(t *testing.T) {
	for _, scenario := range []string{"glm5-h200-tp8.yaml", "kimi-k3-h100-nospec.yaml", v41} {
		host := map[string]float64{}
		for _, mode := range []string{"NONE", "PIECEWISE", "FULL"} {
			in := fixtureInputs(t, scenario)
			editEngine(t, &in, func(e *engineconfig.Engine) { e.CUDAGraphMode = mode })
			host[mode] = build(t, in).StepTime(decodeBatch(1, 1, 8000)).
				PerResource[kernel.ResourceHost].Seconds()
		}
		if host["NONE"] < host["PIECEWISE"] || host["PIECEWISE"] < host["FULL"] {
			t.Errorf("%s: host time NONE %.3g, PIECEWISE %.3g, FULL %.3g; capturing more "+
				"cannot cost the host more", scenario, host["NONE"], host["PIECEWISE"], host["FULL"])
		}
	}
}

// A top-k layer's prefill query attends its selection, not its whole prefix: a chunk's
// attention barely grows with how much context precedes it, where a full-attention chunk's
// grows with the prefix.
func TestSparsePrefillIsBoundedByItsSelection(t *testing.T) {
	k := fixture(t, "glm5-h200-tp8.yaml") // DSA, index_topk 2048
	sm := func(prefix int) float64 {
		return k.StepTime(prefillOf(1024, prefix)).PerResource[kernel.ResourceSM].Seconds()
	}
	if r := sm(131072) / sm(8192); r > 1.5 {
		t.Errorf("a 1,024-token chunk on a 128K prefix costs %.2fx one on an 8K prefix; "+
			"each query reads at most its top-k", r)
	}
}

// The indexer's scoring scan runs in prefill too: removing it makes a prefill cheaper on SM,
// not only by its launches.
func TestTheIndexerScanIsPricedInPrefill(t *testing.T) {
	with := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	if editNodes(in.Model, role("block_index_scores"),
		func(n *model.Node) { n.NumQHeads /= 2 }) == 0 {
		t.Fatal("the graph states no indexer")
	}
	narrower := build(t, in)
	b := prefillOf(8192, 0)
	if w, n := with.StepTime(b).PerResource[kernel.ResourceSM],
		narrower.StepTime(b).PerResource[kernel.ResourceSM]; w <= n {
		t.Errorf("an 8K prefill takes %v of SM with the indexer's heads and %v with half "+
			"of them; the scan is not priced in prefill", w, n)
	}
}

// Speculating costs the drafter's passes and holds its weights: the same verify step is
// dearer, and the deployment holds more, than the deployment that does not speculate. A parallel-block drafter
// runs once where an autoregressive one runs once per drafted token, so it is no dearer.
func TestTheDrafterIsPriced(t *testing.T) {
	spec := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
	none := build(t, in)
	b := decodeBatch(32, 6, 8192)
	if spec.StepTime(b).NoOverlap <= none.StepTime(b).NoOverlap {
		t.Error("a speculating step costs no more than the same step with no drafter")
	}
	if spec.FixedBytes().Weights <= none.FixedBytes().Weights {
		t.Error("the drafter's weights are not resident")
	}

	in = fixtureInputs(t, v41)
	in.Model.Speculator.Drafting = model.DraftingAutoregressive
	in.Model.Speculator.SequentialHead = nil
	in.Model.Speculator.ContextStack = nil
	// An autoregressive draft reads only its ring, not the block positions beyond it.
	if editNodes(in.Model, func(n model.Node) bool {
		return n.Op == model.OpAttention && len(n.Caches) > 0 &&
			n.Caches[0].Kind == model.CacheWindow && n.Window > n.Caches[0].Window
	}, func(n *model.Node) { n.Window = n.Caches[0].Window }) == 0 {
		t.Fatal("the drafter states no read beyond its ring")
	}
	ar := build(t, in)
	in = fixtureInputs(t, v41)
	in.Model.Speculator.SequentialHead = nil
	in.Model.Speculator.ContextStack = nil
	pb := build(t, in)
	if pb.StepTime(b).NoOverlap > ar.StepTime(b).NoOverlap {
		t.Error("one parallel-block pass costs more than five autoregressive ones")
	}

	// An aliased weight is read, not held: the drafter's LM head reads the target's.
	in = fixtureInputs(t, v41)
	for i := range in.Model.Speculator.Head {
		in.Model.Speculator.Head[i].Aliases = ""
		if in.Model.Speculator.Head[i].Op == model.OpGEMM {
			in.Model.Speculator.Head[i].WeightDType = model.DTypeBF16
		}
	}
	own := build(t, in)
	if own.FixedBytes().Weights <= spec.FixedBytes().Weights {
		t.Error("an LM head of the drafter's own holds no more than one aliasing the target's")
	}
	// And computes alike: an alias runs at its target's format, here BF16, exactly as a
	// BF16 copy of its own would, on every chip.
	for _, chip := range []string{"h200", "b200"} {
		a, o := fixtureInputs(t, v41), in
		if chip != "h200" {
			a.Chip, a.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
			o = fixtureInputs(t, v41)
			for i := range o.Model.Speculator.Head {
				o.Model.Speculator.Head[i].Aliases = ""
				if o.Model.Speculator.Head[i].Op == model.OpGEMM {
					o.Model.Speculator.Head[i].WeightDType = model.DTypeBF16
				}
			}
			o.Chip, o.Scenario.Cluster.Hardware = a.Chip, chip
		}
		if x, y := build(t, a).StepTime(b).NoOverlap, build(t, o).StepTime(b).NoOverlap; x != y {
			t.Errorf("%s: a drafter aliasing the BF16 LM head steps in %v, its own BF16 copy in %v",
				chip, x, y)
		}
	}
}

// DeepSeek-V4-family FlashMLA pads each rank's query heads to 64, on SM90 and SM100 alike
// (V4.1's SM100 mega attention subclasses it): at TP8 a 64-head V4.1 rank computes 64 heads
// whether the model has 64 or 32, so halving the attention's heads leaves the prefill
// unchanged on both parts.
func TestFlashMLAPadsQueryHeads(t *testing.T) {
	halve := func(in *Inputs) {
		editNodes(in.Model, func(n model.Node) bool {
			return n.Op == model.OpAttention && n.Role == "" && n.NumQHeads == 64
		}, func(n *model.Node) { n.NumQHeads = 32 })
	}
	b := prefillOf(8192, 0)
	for _, chip := range []string{"h200", "b200"} {
		in := fixtureInputs(t, v41)
		half := fixtureInputs(t, v41)
		if chip != "h200" {
			in.Chip, in.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
			half.Chip, half.Scenario.Cluster.Hardware = chipInputs(t, chip), chip
		}
		halve(&half)
		if full, h := build(t, in).StepTime(b).NoOverlap, build(t, half).StepTime(b).NoOverlap; full != h {
			t.Errorf("%s: 64 heads prefill in %v and 32 in %v; both pad to 64 a rank",
				chip, full, h)
		}
	}
}

// The drafter's routed experts divide over the expert group by their own count, not the
// target's: what speculating adds to a rank's weights does not depend on how many experts
// the target routes over.
func TestDraftExpertsAreSizedByTheirOwnCount(t *testing.T) {
	added := func(targetExperts int) int64 {
		with := func(spec bool) int64 {
			in := fixtureInputs(t, v41)
			if !spec {
				editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
			}
			for i := range in.Model.LayerKinds {
				if strings.HasPrefix(in.Model.LayerKinds[i].ID, "dspark") {
					continue
				}
				for j := range in.Model.LayerKinds[i].Nodes {
					if n := &in.Model.LayerKinds[i].Nodes[j]; n.Role == "experts" {
						n.Experts = targetExperts
					}
				}
			}
			return build(t, in).FixedBytes().Weights
		}
		return with(true) - with(false)
	}
	if a, b := added(384), added(192); a != b {
		t.Errorf("speculating adds %d weight bytes beside 384 target experts and %d beside "+
			"192; the drafter's are its own", a, b)
	}
}

// The LM head runs over the logits rows -- one per prefill request, one per decode token --
// so widening the vocabulary costs an 8K prefill the same head work as a one-token decode.
func TestTheLMHeadRunsOverLogitsRows(t *testing.T) {
	widen := func(in *Inputs) { scaleVocab(in, 2) }
	wide := func() *Kernel {
		in := fixtureInputs(t, v41)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
		widen(&in)
		return build(t, in)
	}()
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
	narrow := build(t, in)
	sm := func(k *Kernel, b kernel.Batch) time.Duration { return k.StepTime(b).PerResource[kernel.ResourceSM] }
	prefill, decode := prefillOf(8192, 0), decodeBatch(1, 1, 8192)
	dp := sm(wide, prefill) - sm(narrow, prefill)
	dd := sm(wide, decode) - sm(narrow, decode)
	if dd <= 0 || dp > 2*dd {
		t.Errorf("doubling the vocabulary costs an 8K prefill %v of SM and a 1-token decode %v; "+
			"both run the head over one row", dp, dd)
	}
}

// Under tensor parallelism the vocabulary-parallel logits are gathered: a wider vocabulary
// costs link time at TP8.
func TestLogitsAreGatheredAcrossTP(t *testing.T) {
	link := func(scale int) time.Duration {
		in := fixtureInputs(t, v41)
		scaleVocab(&in, scale)
		editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
		return build(t, in).StepTime(decodeBatch(64, 1, 8192)).PerResource[kernel.ResourceNVLink]
	}
	if one, two := link(1), link(2); two <= one {
		t.Errorf("a doubled vocabulary moves %v of NVLink against %v; the logits are gathered",
			two, one)
	}
}

// scaleVocab widens the LM head, and the drafter's head that aliases it, by factor.
func scaleVocab(in *Inputs, factor int) {
	heads := [][]model.Node{in.Model.Head}
	if in.Model.Speculator != nil {
		heads = append(heads, in.Model.Speculator.Head)
	}
	for _, h := range heads {
		for i := range h {
			if h[i].Op == model.OpGEMM {
				h[i].N *= factor
			}
		}
	}
}
