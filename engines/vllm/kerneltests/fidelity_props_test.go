// Moved from blis-latency-kernel/fidelity_props_test.go at f556c19c1b1b7cb58c7f279a131c94b78a66b36b.

package latencykernel_test

import (
	. "github.com/inference-sim/blis-latency-kernel"

	"strings"
	"testing"

	"github.com/inference-sim/blis-schemas/kernel"
	"github.com/inference-sim/blis-schemas/spec/coefficient"
	"github.com/inference-sim/blis-schemas/spec/model"
	"github.com/inference-sim/blis-vllm/rules/engineconfig"
)

// (#31, stated-cache path) A stated window ring bounds its own decode read: halving
// every ring's window lowers a long-context decode's HBM time, and leaves a decode shorter
// than the narrower window unchanged.
func TestStatedWindowRingBoundsItsDecodeRead(t *testing.T) {
	narrow := func() *Kernel {
		in := fixtureInputs(t, v41)
		if editNodes(in.Model, func(n model.Node) bool {
			for _, c := range n.Caches {
				if c.Kind == model.CacheWindow {
					return true
				}
			}
			return false
		}, func(n *model.Node) {
			for c := range n.Caches {
				if w := n.Caches[c].Window; n.Caches[c].Kind == model.CacheWindow {
					n.Caches[c].Window = w / 2
					n.Window -= w - w/2 // the read keeps whatever it reads beyond the ring
				}
			}
		}) == 0 {
			t.Fatal("the graph states no window ring")
		}
		return build(t, in)
	}()
	wide := build(t, fixtureInputs(t, v41))
	hbm := func(k *Kernel, ctx int) float64 {
		return k.StepTime(decodeBatch(64, 1, ctx)).PerResource[kernel.ResourceHBM].Seconds()
	}
	if n, w := hbm(narrow, 32768), hbm(wide, 32768); n >= w {
		t.Errorf("halving the rings' window left a 32K decode's HBM at %v (was %v)", n, w)
	}
	if n, w := hbm(narrow, 32), hbm(wide, 32); n != w {
		t.Errorf("a 32-token decode, inside both windows, read %v and %v", n, w)
	}
}

// (#36) A layer kind outside the target stack does not decide which cache dtypes the
// target can run: a drafter declared last with plain MLA does not refuse fp8_ds_mla for an
// all-sparse target.
func TestDraftKindDoesNotGateTheTargetCacheDType(t *testing.T) {
	in := fixtureInputs(t, v41)
	draft := map[string]bool{}
	for _, id := range in.Model.Speculator.Stack.Expand() {
		draft[id] = true
	}
	for i := range in.Model.LayerKinds {
		if !draft[in.Model.LayerKinds[i].ID] {
			continue
		}
		for j := range in.Model.LayerKinds[i].Nodes {
			n := &in.Model.LayerKinds[i].Nodes[j]
			if n.Op == model.OpAttention && n.Role == "" {
				n.AttentionKind, n.IndexTopK, n.CompressRatio = model.AttentionMLA, 0, 0
			}
		}
	}
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
	editEngine(t, &in, func(e *engineconfig.Engine) { e.CacheDType = "fp8_ds_mla" })
	if _, err := newKernel(in); err != nil {
		t.Errorf("an MLA drafter declared last refused fp8_ds_mla for a sparse target: %v", err)
	}
}

// (#35) The drafter's passes are device work, in both composition modes; one parallel
// block pass is strictly cheaper than five autoregressive ones; the sequential head and the
// context stack each cost something.
func TestDraftPassesAreDeviceWork(t *testing.T) {
	b := decodeBatch(32, 6, 8192)
	step := func(edit func(*Inputs)) kernel.StepEstimate {
		in := fixtureInputs(t, v41)
		if edit != nil {
			edit(&in)
		}
		return build(t, in).StepTime(b)
	}
	base := step(nil)
	deeper := step(func(in *Inputs) {
		draft := map[string]bool{}
		for _, id := range in.Model.Speculator.Stack.Expand() {
			draft[id] = true
		}
		for i := range in.Model.LayerKinds {
			if !draft[in.Model.LayerKinds[i].ID] {
				continue
			}
			for j := range in.Model.LayerKinds[i].Nodes {
				if n := &in.Model.LayerKinds[i].Nodes[j]; n.Role == "experts" {
					n.TopK *= 2
				}
			}
		}
	})
	if deeper.Overlap <= base.Overlap || deeper.NoOverlap <= base.NoOverlap ||
		deeper.PerResource[kernel.ResourceHBM] <= base.PerResource[kernel.ResourceHBM] {
		t.Errorf("doubling the draft stack: Overlap %v->%v, NoOverlap %v->%v, HBM %v->%v",
			base.Overlap, deeper.Overlap, base.NoOverlap, deeper.NoOverlap,
			base.PerResource[kernel.ResourceHBM], deeper.PerResource[kernel.ResourceHBM])
	}
	bare := func(in *Inputs) {
		in.Model.Speculator.SequentialHead, in.Model.Speculator.ContextStack = nil, nil
	}
	ar := step(func(in *Inputs) {
		bare(in)
		in.Model.Speculator.Drafting = model.DraftingAutoregressive
		// An autoregressive draft reads only its ring: its read window is the ring's.
		editNodes(in.Model, func(n model.Node) bool { return len(n.Caches) > 0 },
			func(n *model.Node) {
				for _, c := range n.Caches {
					if c.Kind == model.CacheWindow && n.Window > c.Window {
						n.Window = c.Window
					}
				}
			})
	})
	if pb := step(bare); pb.NoOverlap >= ar.NoOverlap {
		t.Errorf("one parallel-block pass (%v) is not cheaper than five AR passes (%v)",
			pb.NoOverlap, ar.NoOverlap)
	}
	if s := step(func(in *Inputs) { in.Model.Speculator.SequentialHead = nil }); s.NoOverlap >= base.NoOverlap {
		t.Error("dropping the sequential head did not make the step cheaper")
	}
	if s := step(func(in *Inputs) { in.Model.Speculator.ContextStack = nil }); s.NoOverlap >= base.NoOverlap {
		t.Error("dropping the context stack did not make the step cheaper")
	}
}

// (#34) With expert parallelism (so nothing is padded), four-bit experts on a part with
// no four-bit peak compute strictly slower than FP8 experts: W4A16 runs at BF16.
func TestFourBitExpertsOnHopperComputeAtBF16(t *testing.T) {
	b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}}
	sm := func(edit func(*model.Node)) float64 {
		in := fixtureInputs(t, v41)
		in.Deployment.Pools[0].Parallel.EnableExpertParallel = true
		if edit != nil {
			editNodes(in.Model, role("experts"), edit)
		}
		return build(t, in).StepTime(b).PerResource[kernel.ResourceSM].Seconds()
	}
	fp4, fp8 := sm(nil), sm(func(n *model.Node) { n.WeightDType = model.DTypeFP8 })
	if fp4 <= fp8 {
		t.Errorf("unpadded MXFP4 experts take %v of SM, no more than FP8's %v", fp4, fp8)
	}
}

// (#37) Replicating a column-parallel GEMM strictly raises a prefill's SM time: every
// rank now runs the whole matmul.
func TestReplicatingAGEMMRaisesItsWork(t *testing.T) {
	b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 8192, PromptLen: 8192}}}
	base := build(t, fixtureInputs(t, v41)).StepTime(b).PerResource[kernel.ResourceSM]
	in := fixtureInputs(t, v41)
	editNodes(in.Model, role("wq_b"), func(n *model.Node) { n.TP = model.TPReplicated })
	if repl := build(t, in).StepTime(b).PerResource[kernel.ResourceSM]; repl <= base {
		t.Errorf("replicating wq_b left prefill SM at %v (was %v)", repl, base)
	}
}

// (#37) Under the model's own sequence parallelism, token sharding strictly lowers a
// prefill (ceil(tokens/tp) rows instead of all of them).
func TestTokenShardingStrictlyHelpsUnderSP(t *testing.T) {
	sp := func(clear bool) *Kernel {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.DP, pl.EnableExpertParallel = 4, 2, true
		if clear {
			editNodes(in.Model, func(n model.Node) bool { return n.UnderSequenceParallel != "" },
				func(n *model.Node) { n.UnderSequenceParallel = "" })
		}
		return build(t, in)
	}
	b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 4096, PromptLen: 4096}}}
	if on, off := sp(false).StepTime(b).NoOverlap, sp(true).StepTime(b).NoOverlap; on >= off {
		t.Errorf("token sharding under SP prices %v, not below unsharded %v", on, off)
	}
}

// (#30) A speculating pool's handoff also carries the drafter's caches, which share the
// target's KV cache groups and transfer with them. They are window rings, so what they add is
// fixed: the same for a long sequence as for any past the ring.
func TestHandoffCarriesTheDraftRings(t *testing.T) {
	spec := build(t, fixtureInputs(t, v41))
	in := fixtureInputs(t, v41)
	editEngine(t, &in, func(e *engineconfig.Engine) { e.Speculative = nil })
	plain := build(t, in)
	from, to := kernel.Placement{Node: 0}, kernel.Placement{Node: 1}
	extra := func(n int) int64 {
		return spec.PDTransferDemand(n, from, to).Bytes - plain.PDTransferDemand(n, from, to).Bytes
	}
	if extra(8192) <= 0 {
		t.Errorf("a speculating pool hands off %d more bytes than a plain one", extra(8192))
	}
	if a, b := extra(8192), extra(131072); a != b {
		t.Errorf("the drafter's rings add %d bytes to an 8K handoff and %d to a 128K one", a, b)
	}
}

// (#33) Context-proportional work is priced: doubling every stated context entry's
// bytes raises a long-prefix prefill chunk's HBM time.
func TestContextWorkIsPriced(t *testing.T) {
	b := prefillOf(2048, 65536)
	base := build(t, fixtureInputs(t, v41)).StepTime(b).PerResource[kernel.ResourceHBM]
	in := fixtureInputs(t, v41)
	if editNodes(in.Model, func(n model.Node) bool { return n.Context != nil },
		func(n *model.Node) { c := *n.Context; c.BytesPerEntry *= 2; n.Context = &c }) == 0 {
		t.Fatal("the graph states no context work")
	}
	if wide := build(t, in).StepTime(b).PerResource[kernel.ResourceHBM]; wide <= base {
		t.Errorf("doubling context entry bytes left HBM at %v (was %v)", wide, base)
	}
}

// (#27) The recurrent DECODE rate prices decode rows only: slowing it 10x moves a
// decode step and leaves a pure prefill step exactly where it was.
func TestRecurrentDecodeRateDoesNotPricePrefill(t *testing.T) {
	scale := func(sets []*coefficient.Set, prefix string, f float64) []*coefficient.Set {
		out := make([]*coefficient.Set, len(sets))
		for i, s := range sets {
			c := *s
			c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
			for j := range c.Coefficients {
				if strings.HasPrefix(c.Coefficients[j].Name, prefix) {
					c.Coefficients[j].Value *= f
				}
			}
			out[i] = &c
		}
		return out
	}
	in := fixtureInputs(t, "glm5-h200-tp8.yaml")
	withModel(t, &in, "glm-5.3-flash")
	base := build(t, in)
	in.Coefficients = scale(in.Coefficients, "recurrent_decode_rate_", 0.1)
	slow := build(t, in)
	p := prefillOf(8192, 0)
	if a, b := base.StepTime(p).NoOverlap, slow.StepTime(p).NoOverlap; a != b {
		t.Errorf("a 10x slower decode scan moved a pure prefill from %v to %v", a, b)
	}
	d := decodeBatch(64, 1, 8192)
	if a, b := base.StepTime(d).NoOverlap, slow.StepTime(d).NoOverlap; b <= a {
		t.Errorf("a 10x slower decode scan left a decode step at %v (was %v)", b, a)
	}
}

// P11 (#42), with no internal fields: sweep the step's token count one request at a time
// from 1 to 2048 and require it never to fall.
func TestStepNeverFallsAsRequestsArrive(t *testing.T) {
	for _, sc := range []string{"glm5-h200-tp8.yaml", "minimax-m25-h200-tp8.yaml",
		"kimi-k3-h100-nospec.yaml", v41} {
		k := fixture(t, sc)
		var b kernel.Batch
		b.DecodeThreshold = 1
		prev := k.StepTime(b).NoOverlap
		for i := 0; b.Tokens() < 2048; i++ {
			s := 1
			if i%5 == 4 {
				s = 7 // a short prefill now and then
			}
			b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: s, Computed: 4096, PromptLen: 4096 + s})
			if got := k.StepTime(b).NoOverlap; got < prev {
				t.Fatalf("%s: %d tokens priced %v, below %d tokens' %v", sc, b.Tokens(), got,
					b.Tokens()-s, prev)
			} else {
				prev = got
			}
		}
	}
}

// (#37) Token sharding of GEMMs specifically: clearing it on GEMM nodes alone makes an
// SP prefill strictly dearer.
func TestTokenShardedGEMMsRunFewerRows(t *testing.T) {
	sp := func(clear bool) *Kernel {
		in := fixtureInputs(t, v41)
		pl := &in.Deployment.Pools[0].Parallel
		pl.TP, pl.DP, pl.EnableExpertParallel = 4, 2, true
		if clear && editNodes(in.Model, func(n model.Node) bool {
			return n.Op == model.OpGEMM && n.UnderSequenceParallel != ""
		}, func(n *model.Node) { n.UnderSequenceParallel = "" }) == 0 {
			t.Fatal("no token-sharded GEMM in the graph")
		}
		return build(t, in)
	}
	b := kernel.Batch{Reqs: []kernel.ReqShape{{Scheduled: 4096, PromptLen: 4096}}}
	if on, off := sp(false).StepTime(b).PerResource[kernel.ResourceSM],
		sp(true).StepTime(b).PerResource[kernel.ResourceSM]; on >= off {
		t.Errorf("token-sharded GEMMs take %v of SM, not below unsharded %v", on, off)
	}
}

// (#42) The capture-ceiling law holds whatever the coefficients: with the eager
// launch cost made 1000x cheaper than the registry's, the step still never falls as
// requests arrive.
func TestCeilingLawSurvivesCheapEagerLaunches(t *testing.T) {
	in := fixtureInputs(t, "glm5-h200-tp8.yaml")
	for i, s := range in.Coefficients {
		c := *s
		c.Coefficients = append([]coefficient.Entry(nil), s.Coefficients...)
		for j := range c.Coefficients {
			if strings.HasPrefix(c.Coefficients[j].Name, "host_launch_eager") {
				c.Coefficients[j].Value *= 1e-3
			}
		}
		in.Coefficients[i] = &c
	}
	k := build(t, in)
	var b kernel.Batch
	b.DecodeThreshold = 1
	prev := k.StepTime(b).NoOverlap
	for b.Tokens() < 2048 {
		b.Reqs = append(b.Reqs, kernel.ReqShape{Scheduled: 1, Computed: 4096, PromptLen: 4097})
		got := k.StepTime(b).NoOverlap
		if got < prev {
			t.Fatalf("%d tokens priced %v, below %v one request fewer", b.Tokens(), got, prev)
		}
		prev = got
	}
}
