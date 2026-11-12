package latencykernel

import (
	"strings"
	"testing"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/rules"
	"github.com/inference-sim/blis-substrate/contracts/spec/deployment"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/load"
)

// An engine can run the layers after the last one that produces KV on each request's trailing
// window only (DeepSeek-V4.1 at vLLM v0.32.0: layers 21 to 39 of 40 on each request's last 128
// tokens, vllm/models/deepseek_v41/nvidia/model_state.py:438-540@v0.32.0). The kernel prices the
// rows the engine states: TailFrom and each request's TailRows. These tests read the price, not
// the batch.

// v41TailCase is a DeepSeek-V4.1-Flash kernel on H200 at tensor parallel 8, the deployment the
// layer is for, and the index of the model's last KV-producing layer.
func v41TailCase(t *testing.T) (*Kernel, Inputs, int) {
	t.Helper()
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range perturbationScenarios(t, d, "deepseek-v4.1-flash") {
		if strings.HasPrefix(c.label, "deepseek-v4.1-flash/h200/tp8/dp1/epfalse/dcp0/pcp0") {
			return mustKernel(t, c.in), c.in, lastKVLayer(t, c.in.Model)
		}
	}
	t.Fatal("no deepseek-v4.1-flash scenario builds on h200 at tp8")
	return nil, Inputs{}, 0
}

// lastKVLayer is the position in the stack of the last layer whose kind owns a compressed KV
// cache: the layer the engine writes every row's KV in before the layers that read it.
func lastKVLayer(t *testing.T, g *model.Graph) int {
	t.Helper()
	owns := map[string]bool{}
	for _, lk := range g.LayerKinds {
		for _, n := range lk.Nodes {
			for _, c := range n.Caches {
				if c.Owned && c.Kind == model.CacheKV {
					owns[lk.ID] = true
				}
			}
		}
	}
	last := -1
	for i, id := range g.Stack.Expand() {
		if owns[id] {
			last = i
		}
	}
	if last < 0 {
		t.Fatal("no layer owns a compressed KV cache")
	}
	return last
}

func tailStep(k *Kernel, from int, reqs ...kernel.ReqShape) float64 {
	return k.StepTime(kernel.Batch{DecodeThreshold: 1, TailFrom: from, Reqs: reqs}).NoOverlap.Seconds()
}

func prefill(scheduled, computed, tail int) kernel.ReqShape {
	return kernel.ReqShape{Scheduled: scheduled, Computed: computed, PromptLen: scheduled + computed, TailRows: tail}
}

// The layers after the last KV layer cost what 128 rows cost, not what the chunk costs: a chunk
// of 8192 tokens and one of 2048 differ in the layers before the cut, and not in the layers after.
func TestTheLayersAfterTheLastKVLayerAreChargedOnTheTrailingRowsNotTheChunk(t *testing.T) {
	k, _, cut := v41TailCase(t)
	// What trimming saves: the step with every layer on the chunk less the step with the layers
	// from the cut on 128 rows.
	save := func(s int) float64 {
		return tailStep(k, 0, prefill(s, 0, 0)) - tailStep(k, cut, prefill(s, 0, 128))
	}
	s2k, s8k := save(2048), save(8192)
	if s2k <= 0 || s8k <= 0 {
		t.Fatalf("trimming the tail to 128 rows saved %.4g s at 2048 and %.4g s at 8192; it must save time", s2k, s8k)
	}
	// Layers 20 to 39 are half the stack, so the layers they skip are about half the chunk's
	// token-proportional work: the saving grows with the chunk, by about its ratio (4x), not by
	// the fixed cost of 128 rows.
	if r := s8k / s2k; r < 2.5 || r > 5 {
		t.Errorf("the saving grew %.2fx from a 2048-token chunk to an 8192-token one; the rows skipped grow 4x", r)
	}
	// And the priced tail does not grow with the chunk the way an untrimmed one does: the step
	// is the same layers plus a tail on 128 rows, whose own cost moves by the context the rows
	// read, not by 4x.
	tail := func(s int) float64 { return tailStep(k, cut, prefill(s, 0, 128)) - tailStep(k, cut, prefill(s, 0, 1)) }
	if r := tail(8192) / tail(2048); r > 2 {
		t.Errorf("the tail on 128 rows grew %.2fx from a 2048-token chunk to an 8192-token one", r)
	}
}

// A request's rows are its own: the price falls as the engine states fewer rows for it, and
// stating all of them is the price of no tail at all.
func TestARequestsRowsAreItsOwnAndAllOfThemIsNoTail(t *testing.T) {
	k, _, cut := v41TailCase(t)
	none := tailStep(k, 0, prefill(4096, 0, 0))
	if got := tailStep(k, cut, prefill(4096, 0, 4096)); got != none {
		t.Errorf("a tail of every row prices %.9g s, no tail %.9g s", got, none)
	}
	if got := tailStep(k, cut, prefill(4096, 0, 0)); got != none {
		t.Errorf("a tail that states no rows for the request prices %.9g s, no tail %.9g s", got, none)
	}
	prev := none
	for _, rows := range []int{2048, 512, 128, 1} {
		got := tailStep(k, cut, prefill(4096, 0, rows))
		if got >= prev {
			t.Errorf("%d tail rows price %.6g s, not below %.6g s for more rows", rows, got, prev)
		}
		prev = got
	}
	// Rows beyond the request's tokens are not rows it has.
	if got := tailStep(k, cut, prefill(4096, 0, 9000)); got != none {
		t.Errorf("9000 tail rows of 4096 scheduled tokens price %.9g s, the whole chunk %.9g s", got, none)
	}
}

// A request with a cached prefix schedules the hit's rewind and the tokens after it; the engine
// states that request's own rows, and a short request in the same step keeps its own, whatever
// its neighbour keeps.
func TestEachRequestInAStepUsesItsOwnRows(t *testing.T) {
	k, _, cut := v41TailCase(t)
	hit := prefill(128+3000, 300_000, 128)
	short := func(rows int) float64 { return tailStep(k, cut, hit, prefill(90, 0, rows)) }
	if a, b := short(90), short(1); a <= b {
		t.Errorf("a 90-token request keeping its 90 rows prices %.6g s, keeping 1 row %.6g s", a, b)
	}
	swapped := tailStep(k, cut, prefill(90, 0, 90), hit)
	if got := short(90); got != swapped {
		t.Errorf("the same two requests in the other order price %.9g s and %.9g s", got, swapped)
	}
	// The hit's rows are its own: it keeps 128 of its 3128 and prices below keeping 1000.
	big := func(rows int) float64 {
		return tailStep(k, cut, prefill(128+3000, 300_000, rows), prefill(90, 0, 90))
	}
	if a, b := big(128), big(1000); a >= b {
		t.Errorf("the cached-prefix request keeping 128 rows prices %.6g s, 1000 rows %.6g s", a, b)
	}
}

// A decode step is unchanged: its requests schedule a verify width and keep it, so a tail changes
// no row of it.
func TestADecodeStepIsUnchangedByATail(t *testing.T) {
	k, _, cut := v41TailCase(t)
	decodes := make([]kernel.ReqShape, 16)
	for i := range decodes {
		decodes[i] = kernel.ReqShape{Scheduled: 1, Computed: 50_000 + i, PromptLen: 40_000}
	}
	if a, b := tailStep(k, 0, decodes...), tailStep(k, cut, decodes...); a != b {
		t.Errorf("a decode step prices %.9g s with a tail stated and %.9g s without", b, a)
	}
}

// An engine that does not run the layers on a subset states no TailFrom, and the step is priced
// on every scheduled token whatever rows a request names: rows with nothing to apply to are
// refused by Validate, and ignored by the kernel that is handed them anyway.
func TestWithoutTailFromEveryLayerRunsEveryToken(t *testing.T) {
	k, _, _ := v41TailCase(t)
	none := tailStep(k, 0, prefill(4096, 0, 0))
	if got := tailStep(k, 0, prefill(4096, 0, 128)); got != none {
		t.Errorf("rows without a layer to start at price %.9g s, no tail %.9g s", got, none)
	}
	if err := (kernel.Batch{Reqs: []kernel.ReqShape{prefill(4096, 0, 128)}}).Validate(); err == nil {
		t.Error("Validate accepted tail rows with no TailFrom")
	}
	if err := (kernel.Batch{TailFrom: 3, Reqs: []kernel.ReqShape{prefill(100, 0, 101)}}).Validate(); err == nil {
		t.Error("Validate accepted more tail rows than scheduled tokens")
	}
	// A TailFrom past the last layer leaves no layer in the tail.
	if got := tailStep(k, k.graph.Stack.Layers(), prefill(4096, 0, 128)); got != none {
		t.Errorf("a tail from past the last layer prices %.9g s, no tail %.9g s", got, none)
	}
}

// The tail is the layers from TailFrom and no others: starting it one layer later charges one
// more layer on every row, starting it at the first layer charges all of them on the rows.
func TestTheTailCoversTheLayersFromItsFirstLayer(t *testing.T) {
	k, _, cut := v41TailCase(t)
	at := func(from int) float64 { return tailStep(k, from, prefill(4096, 0, 128)) }
	if !(at(1) < at(cut) && at(cut) < at(cut+1) && at(cut+1) < at(cut+10)) {
		t.Errorf("tail starts at 1, %d, %d, %d price %.6g, %.6g, %.6g, %.6g s; a later start charges more layers on every row",
			cut, cut+1, cut+10, at(1), at(cut), at(cut+1), at(cut+10))
	}
}

// The rows the engine launches beyond the tail's own run the GEMMs and the experts, as a data
// parallel pad does: padding the tail to a graph's size costs more than not padding it.
func TestRowsThatPadTheTailArePriced(t *testing.T) {
	k, _, cut := v41TailCase(t)
	base := kernel.Batch{DecodeThreshold: 1, TailFrom: cut, Reqs: []kernel.ReqShape{prefill(4096, 0, 128)}}
	padded := base
	padded.TailPadTokens = 384
	if a, b := k.StepTime(base).NoOverlap, k.StepTime(padded).NoOverlap; b <= a {
		t.Errorf("a tail padded by 384 rows prices %v, unpadded %v", b, a)
	}
}

// A step shape carries the same statement as a batch: BatchOf maps TailFrom, each request's
// TailRows and the tail's launched rows, and the price follows.
func TestAStepShapeStatesTheTailInTheEnginesWords(t *testing.T) {
	k, _, cut := v41TailCase(t)
	shape := plan.StepShape{Mode: plan.StepExtendOnly, TailFrom: cut,
		Ranks:  []plan.RankStep{{Requests: []plan.RequestStep{{Scheduled: 4096, Prompt: 4096, TailRows: 128}}}},
		Padded: []plan.PaddedShape{{Tokens: 4096, TailTokens: 512}}}
	b, err := BatchOf(shape, 0)
	if err != nil {
		t.Fatal(err)
	}
	want := kernel.Batch{DecodeThreshold: 1, TailFrom: cut, TailPadTokens: 512 - 128,
		Reqs: []kernel.ReqShape{{Scheduled: 4096, PromptLen: 4096, TailRows: 128}}}
	if k.StepTime(b).NoOverlap != k.StepTime(want).NoOverlap {
		t.Errorf("BatchOf prices %v, the batch the statement means prices %v", k.StepTime(b).NoOverlap, k.StepTime(want).NoOverlap)
	}
	whole := shape
	whole.TailFrom, whole.Ranks = 0, []plan.RankStep{{Requests: []plan.RequestStep{{Scheduled: 4096, Prompt: 4096}}}}
	whole.Padded = []plan.PaddedShape{{Tokens: 4096}}
	bw, err := BatchOf(whole, 0)
	if err != nil {
		t.Fatal(err)
	}
	if k.StepTime(b).NoOverlap >= k.StepTime(bw).NoOverlap {
		t.Errorf("the step that runs its tail on 128 rows prices %v, not below the whole step's %v",
			k.StepTime(b).NoOverlap, k.StepTime(bw).NoOverlap)
	}
	bad := shape
	bad.Ranks = []plan.RankStep{{Requests: []plan.RequestStep{{Scheduled: 100, Prompt: 100, TailRows: 200}}}}
	if _, err := BatchOf(bad, 0); err == nil {
		t.Error("BatchOf accepted 200 tail rows of 100 scheduled tokens")
	}
}

// The tail's rows are the last rows of the chunk, at their own positions: they read the chunk's
// earlier tokens as context. The same 128 rows cost more when 3968 tokens precede them in the
// chunk than when they are the whole chunk, because the layers' indexers score the whole context.
func TestTheTailRowsReadTheChunkBeforeThem(t *testing.T) {
	k, _, cut := v41TailCase(t)
	// What 127 more rows cost in the tail, at the context each chunk puts them in.
	extra := func(chunk int) float64 {
		return tailStep(k, cut, prefill(chunk, 0, 128)) - tailStep(k, cut, prefill(chunk, 0, 1))
	}
	if deep, shallow := extra(4096), extra(128); deep <= shallow {
		t.Errorf("128 rows after 3968 chunk tokens cost %.6g s more than 1 row, 128 rows that are the whole chunk %.6g s", deep, shallow)
	}
}

// Under pipeline parallelism each stage prices the share of the tail it holds: a tail that starts
// inside the second stage trims that stage alone, and one that starts in the first trims both.
func TestAPipelineStageTrimsTheShareOfTheTailItHolds(t *testing.T) {
	d, err := load.Embedded()
	if err != nil {
		t.Fatal(err)
	}
	const body = `kind: Scenario
name: pp2
model: llama-3.1-8b-instruct
coefficients: [cost-model-primitives, cost-model-collectives, cost-model-host-overheads, cost-model-attention, cost-model-memory, cost-model-fabric]
engine_version: "0.32.0"
cluster:
  hardware: h100
  fabric: ib-400g
  nodes: 1
  gpus_per_node: 8
---
kind: Deployment
name: pp2
pools:
  - role: colocated
    nodes: 1
    parallel: {tp: 2, pp: 2, dp: 1}
`
	r, name := openRepos(t, d, body)
	r.Rules = func(string) rules.EngineRules { return inertRules{} }
	r.EngineSettings = func(*deployment.Deployment, int) (enginesettings.Settings, error) {
		return enginesettings.Settings{BlockSize: 16, MaxNumSeqs: 64, MaxNumBatchedTokens: 8192,
			MaxModelLen: 8192, GPUMemoryUtilization: 0.92}, nil
	}
	in, err := OpenInputs(name, r, 0)
	if err != nil {
		t.Fatal(err)
	}
	k := mustKernel(t, in)
	layers := k.graph.Stack.Layers()
	if len(k.stages) != 2 || k.stages[1].lo <= 0 || k.stages[1].hi != layers {
		t.Fatalf("stages %+v over %d layers", k.stages, layers)
	}
	second := k.stages[1].lo
	none := tailStep(k, 0, prefill(4096, 0, 0))
	late := tailStep(k, second+2, prefill(4096, 0, 128))
	early := tailStep(k, 2, prefill(4096, 0, 128))
	if !(early < late && late < none) {
		t.Errorf("tails from layer 2, %d and none price %.6g, %.6g, %.6g s; each earlier start trims more layers", second+2, early, late, none)
	}
	if got := tailStep(k, second+2, prefill(4096, 0, 4096)); got != none {
		t.Errorf("a tail of every row prices %.9g s, no tail %.9g s", got, none)
	}
}

// sizedRules resolves a release's capture sizes to a fixed ladder, as a release's rules do.
type sizedRules struct{ windowedRules }

func (sizedRules) CUDAGraphCaptureSizes(rules.CUDAGraphCaptureInput) (rules.CUDAGraphCapture, bool, error) {
	return rules.CUDAGraphCapture{Sizes: []int{1, 2, 4, 8, 256, 272, 512}, Max: 512}, true, nil
}

// The capture sizes the kernel prices from are readable by the engine that states a step's rows:
// ascending, the largest being the ceiling past which a step runs eager.
func TestTheCaptureSizesEndAtTheCeilingAStepRunsEagerPast(t *testing.T) {
	_, in, _ := v41TailCase(t)
	in.Rules = sizedRules{}
	k := mustKernel(t, in)
	sizes := k.CaptureSizes()
	want := []int{1, 2, 4, 8, 256, 272, 512}
	if len(sizes) != len(want) || sizes[len(sizes)-1] != k.CaptureTokens() {
		t.Fatalf("sizes %v with ceiling %d; the rules resolved %v", sizes, k.CaptureTokens(), want)
	}
	sizes[0] = -1
	if k.CaptureSizes()[0] == -1 {
		t.Error("CaptureSizes returned the kernel's own slice")
	}
}
