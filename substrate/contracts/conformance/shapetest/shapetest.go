// Package shapetest holds the unusual model shapes of design SUB-10, as valid model graphs,
// and the checks every engine and the cost model run over them.
//
// A model that is merely larger than the ones a cost model was written for rarely breaks it.
// One with a different structure does: a stack in which every layer is windowed, or recurrent,
// or has no attention at all; a single-layer model; a draft layer declared last. Each shape
// below is built to be valid under the graph schema and to differ from the catalog's models
// in exactly one structural way. An implementation either runs a shape or refuses it with its
// own reason; it never returns a size or a price that is non-finite or zero where a positive
// value is required.
package shapetest

import (
	"fmt"
	"math"
	"testing"
	"time"

	"github.com/inference-sim/blis-substrate/contracts/kernel"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// Shape is one unusual model.
type Shape struct {
	// Name identifies the shape in a failure.
	Name string
	// Why says what structural property it tests.
	Why string
	// Graph is the model, valid under model.Graph.Validate.
	Graph *model.Graph
}

const (
	hidden = 1024
	vocab  = 32000
)

func base(name string) *model.Graph {
	return &model.Graph{
		Kind: "ModelGraph", Name: name,
		DerivedFrom: model.Derivation{Format: "conformance", Path: "shapetest",
			SHA256: "0000000000000000000000000000000000000000000000000000000000000000", DeriverVersion: 1},
		Global: model.GlobalShape{HiddenSize: hidden, VocabSize: vocab, WeightDType: model.DTypeBF16},
		Head: []model.Node{
			{Op: model.OpElementwise, Role: "final_norm"},
			{Op: model.OpGEMM, Role: "lm_head", N: vocab, K: hidden},
		},
	}
}

func ffn() []model.Node {
	return []model.Node{
		{Op: model.OpElementwise, Role: "post_attn_norm"},
		{Op: model.OpGEMM, Role: "mlp_gate_up", N: 4 * hidden, K: hidden},
		{Op: model.OpGEMM, Role: "mlp_down", N: hidden, K: 2 * hidden},
		{Op: model.OpAllReduce, Role: "mlp_out", Emit: model.EmitTensorParallel},
	}
}

func chain(nodes []model.Node) [][2]int {
	var e [][2]int
	for i := 0; i+1 < len(nodes); i++ {
		e = append(e, [2]int{i, i + 1})
	}
	return e
}

func attention(kind model.AttentionKind, window int) []model.Node {
	n := model.Node{Op: model.OpAttention, AttentionKind: kind, NumQHeads: 16, NumKVHeads: 4, HeadDim: 64, Window: window}
	if kind == model.AttentionMLA {
		n = model.Node{Op: model.OpAttention, AttentionKind: kind, NumQHeads: 16, NumKVHeads: 1,
			HeadDim: 576, KVLoRARank: 512, QKRopeHeadDim: 64}
	}
	return []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpGEMM, Role: "qkv_proj", N: 16*64 + 2*4*64, K: hidden},
		n,
		{Op: model.OpGEMM, Role: "o_proj", N: hidden, K: 16 * 64},
		{Op: model.OpAllReduce, Role: "attn_out", Emit: model.EmitTensorParallel},
	}
}

func layerKind(id string, nodes []model.Node) model.LayerKind {
	return model.LayerKind{ID: id, Nodes: nodes, Edges: chain(nodes)}
}

func recurrentNodes() []model.Node {
	return []model.Node{
		{Op: model.OpElementwise, Role: "input_norm"},
		{Op: model.OpRecurrentUpdate, RecurrentKind: model.RecurrentGDN, NumHeads: 16,
			StateSize: 64, ConvKernel: 4, IntermediateSize: hidden, StateDType: model.DTypeFP32},
		{Op: model.OpGEMM, Role: "mixer_out", N: hidden, K: hidden},
		{Op: model.OpAllReduce, Role: "mixer_out_reduce", Emit: model.EmitTensorParallel},
	}
}

// Shapes returns the six shapes of SUB-10 and the extras that the same reasoning adds. Each
// call returns fresh graphs, so a test may change one without affecting another.
func Shapes() []Shape {
	var out []Shape
	add := func(name, why string, g *model.Graph) { out = append(out, Shape{name, why, g}) }

	g := base("every-layer-windowed")
	g.LayerKinds = []model.LayerKind{layerKind("swa", append(attention(model.AttentionSWA, 128), ffn()...))}
	g.Stack = model.Stack{Pattern: []string{"swa"}, Repeat: 8}
	add("every layer windowed", "no layer reads the whole context, so KV per sequence is bounded by the window", g)

	g = base("every-layer-recurrent")
	g.LayerKinds = []model.LayerKind{layerKind("rec", append(recurrentNodes(), ffn()...))}
	g.Stack = model.Stack{Pattern: []string{"rec"}, Repeat: 8}
	add("every layer recurrent", "no KV cache at all, only per-sequence state that does not grow with context", g)

	g = base("mla-only")
	g.LayerKinds = []model.LayerKind{layerKind("mla", append(attention(model.AttentionMLA, 0), ffn()...))}
	g.Stack = model.Stack{Pattern: []string{"mla"}, Repeat: 8}
	add("MLA only", "one latent vector per token and no value tensor", g)

	g = base("single-layer")
	g.LayerKinds = []model.LayerKind{layerKind("dense", append(attention(model.AttentionGQA, 0), ffn()...))}
	g.Stack = model.Stack{Pattern: []string{"dense"}, Repeat: 1}
	add("a single layer", "first and last layer are the same layer; pipeline splitting has nothing to split", g)

	g = base("no-attention")
	g.LayerKinds = []model.LayerKind{layerKind("mlp", append([]model.Node{{Op: model.OpElementwise, Role: "input_norm"}}, ffn()...))}
	g.Stack = model.Stack{Pattern: []string{"mlp"}, Repeat: 4}
	add("no attention at all", "no layer reads or writes a KV cache, so sequence bytes are zero", g)

	g = base("draft-layer-last")
	g.LayerKinds = []model.LayerKind{
		layerKind("dense", append(attention(model.AttentionGQA, 0), ffn()...)),
		layerKind("draft", append(attention(model.AttentionGQA, 0), ffn()...)),
	}
	g.Stack = model.Stack{Pattern: []string{"dense"}, Repeat: 6}
	g.Speculator = &model.Speculator{Method: "conformance_mtp", NumSpec: 1,
		Stack: model.Stack{Pattern: []string{"draft"}, Repeat: 1}}
	add("a draft layer declared last", "the speculator's layer comes after the target's in layer_kinds", g)
	return out
}

// Validate returns an error when the shape's graph is not a valid document. A shape that is
// not valid tests nothing about the implementation.
func (s Shape) Validate() error {
	if p := s.Graph.Validate(); !p.OK() {
		return fmt.Errorf("shape %q is not a valid graph:\n%s", s.Name, p.Error())
	}
	return nil
}

// Cache is the kind of per-sequence memory a shape holds, read from its graph.
type Cache int

const (
	// CacheNone: no attention and no recurrent state, so a sequence holds nothing.
	CacheNone Cache = iota
	// CacheState: recurrent state only, which does not grow with context.
	CacheState
	// CacheBounded: attention in which every layer is windowed, so the cache stops growing at
	// the window.
	CacheBounded
	// CacheGrowing: at least one attention layer reads the whole context.
	CacheGrowing
)

func (c Cache) String() string {
	return [...]string{"none", "state", "bounded", "growing"}[c]
}

// CacheOf reads the shape's cache kind from its graph, so an implementation cannot excuse a
// wrong size by declaring the shape different.
func CacheOf(g *model.Graph) Cache {
	c := CacheNone
	g.EachNode(func(_ string, n model.Node) {
		switch {
		case n.Op == model.OpAttention && n.AttentionKind != model.AttentionSWA:
			c = CacheGrowing
		case n.Op == model.OpAttention && c < CacheBounded:
			c = CacheBounded
		case n.Op == model.OpRecurrentUpdate && c < CacheState:
			c = CacheState
		}
	})
	return c
}

// CheckSizes checks the numbers a kernel reports for a shape. Every size is finite and not
// negative, the model has weights, and the per-sequence figures follow the cache kind: a
// shape with no cache holds nothing; a growing cache holds more at a longer context; a
// bounded cache holds some and stops growing; a state-only shape holds its state. The
// per-sequence total is the fixed part (state, window rings) plus the variable part.
func CheckSizes(k kernel.Kernel, c Cache) error {
	f := k.FixedBytes()
	for name, v := range map[string]int64{
		"Weights": f.Weights, "ActivationPeak": f.ActivationPeak, "CUDAGraph": f.CUDAGraph,
		"EPLBRedundant": f.EPLBRedundant, "CommBuffers": f.CommBuffers,
	} {
		if v < 0 {
			return fmt.Errorf("FixedBytes.%s is negative: %d", name, v)
		}
	}
	if f.Weights <= 0 {
		return fmt.Errorf("FixedBytes.Weights is %d; a model has weights", f.Weights)
	}
	fixed := k.SequenceFixedBytes()
	if fixed < 0 {
		return fmt.Errorf("SequenceFixedBytes is negative: %d", fixed)
	}
	var prev int64 = -1
	for _, tokens := range []int{1, 16, 4096, 131072} {
		v := k.SequenceVariableBytes(tokens)
		if v < 0 {
			return fmt.Errorf("SequenceVariableBytes(%d) is negative: %d", tokens, v)
		}
		if v < prev {
			return fmt.Errorf("SequenceVariableBytes(%d) = %d, below the %d of fewer tokens", tokens, v, prev)
		}
		prev = v
	}
	short, long := k.SequenceVariableBytes(4096), k.SequenceVariableBytes(131072)
	switch c {
	case CacheNone:
		if fixed != 0 || long != 0 {
			return fmt.Errorf("a shape with no cache holds %d fixed and %d variable bytes", fixed, long)
		}
	case CacheState:
		if fixed+long <= 0 {
			return fmt.Errorf("a state-only shape holds no state")
		}
	case CacheBounded:
		if fixed+long <= 0 {
			return fmt.Errorf("a windowed shape holds nothing per sequence")
		}
		if long > short {
			return fmt.Errorf("a windowed cache grows with context: %d bytes at 4096 tokens, %d at 131072", short, long)
		}
	case CacheGrowing:
		if short <= 0 || long <= short {
			return fmt.Errorf("a growing cache does not grow: %d bytes at 4096 tokens, %d at 131072", short, long)
		}
	}
	return nil
}

// CheckPrices checks the prices a kernel returns for a shape over decode, prefill and mixed
// batches: every duration is finite and positive where work is done, the overlap bound is
// not above the no-overlap bound, and no per-resource time is negative.
func CheckPrices(k kernel.Kernel) error {
	decode := kernel.ReqShape{Scheduled: 1, Computed: 4095, PromptLen: 4096}
	prefill := kernel.ReqShape{Scheduled: 512, PromptLen: 512}
	for name, b := range map[string]kernel.Batch{
		"decode":  {DecodeThreshold: 1, Reqs: []kernel.ReqShape{decode}},
		"prefill": {DecodeThreshold: 1, Reqs: []kernel.ReqShape{prefill}},
		"mixed":   {DecodeThreshold: 1, Reqs: []kernel.ReqShape{decode, prefill}},
	} {
		e := k.StepTime(b)
		for label, d := range map[string]time.Duration{"Overlap": e.Overlap, "NoOverlap": e.NoOverlap} {
			if d <= 0 || d > time.Hour {
				return fmt.Errorf("%s step: %s = %v, outside (0, 1h]", name, label, d)
			}
		}
		if e.Overlap > e.NoOverlap {
			return fmt.Errorf("%s step: Overlap %v exceeds NoOverlap %v", name, e.Overlap, e.NoOverlap)
		}
		for r, d := range e.PerResource {
			if d < 0 {
				return fmt.Errorf("%s step: resource %s time is negative: %v", name, r, d)
			}
		}
	}
	for _, f := range []float64{float64(k.AdmissionOverhead(1024)), float64(k.OutputTokenOverhead()), float64(k.CompletionOverhead())} {
		if math.IsNaN(f) || math.IsInf(f, 0) || f < 0 {
			return fmt.Errorf("a host overhead is %v", f)
		}
	}
	return nil
}

// Result is what an implementation did with one shape.
type Result struct {
	// Kernel is the implementation's pricer for the shape, nil when it refused.
	Kernel kernel.Kernel
	// Refusal is the implementation's own reason when it refused the shape. A shape is run or
	// refused, never both and never neither.
	Refusal error
}

// Run holds an implementation to SUB-10 over every shape. build prices one shape. The shape's
// KV cache is judged from its graph, so an implementation cannot excuse a wrong size by
// declaring the shape different.
func Run(t testing.TB, build func(Shape) Result) {
	t.Helper()
	for _, s := range Shapes() {
		if err := s.Validate(); err != nil {
			t.Fatal(err)
		}
		r := build(s)
		switch {
		case r.Kernel != nil && r.Refusal != nil:
			t.Errorf("%s: both ran and refused (%v)", s.Name, r.Refusal)
		case r.Kernel == nil && r.Refusal == nil:
			t.Errorf("%s: neither ran nor refused", s.Name)
		case r.Refusal != nil:
			if r.Refusal.Error() == "" {
				t.Errorf("%s: refused with an empty reason", s.Name)
			}
		default:
			if err := CheckSizes(r.Kernel, CacheOf(s.Graph)); err != nil {
				t.Errorf("%s: %v", s.Name, err)
			}
			if err := CheckPrices(r.Kernel); err != nil {
				t.Errorf("%s: %v", s.Name, err)
			}
		}
	}
}
