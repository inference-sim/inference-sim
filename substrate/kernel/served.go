package latencykernel

import (
	"fmt"
	"strings"

	"github.com/inference-sim/blis-substrate/contracts/kernel/enginesettings"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
)

// checkServedWeights proves that the target graph stores exactly the layout the engine says it
// serves (enginesettings.ServedWeights), class by class, and refuses a difference. The cost
// model resolves no engine setting: the engine model resolves its quantization setting, refuses
// what its engine refuses, and states the result; the kernel prices the graph, so it must show
// the graph is that layout rather than take the statement on trust.
//
// A class the graph does not contain is empty, so a graph with its router, head or embedding
// removed cannot match a layout that states them, and a class stored at another dtype, or at
// more than one, cannot match either. The nodes are told apart by what they are, never by their
// free-text role: the router is the GEMM with one row per routed expert of its layer kind, the
// shared experts are linears, the head is the head GEMM of the vocabulary's width, and the
// embedding is the Gather of that vocabulary, whose stored width is its row bytes over the
// hidden width.
func checkServedWeights(g *model.Graph, sw *enginesettings.ServedWeights) error {
	if sw == nil {
		return nil
	}
	got := storedLayout(g)
	want := *sw
	var diff []string
	for _, c := range []struct{ class, want, got string }{
		{"linear", want.Linear, got.Linear},
		{"routed_experts", want.RoutedExperts, got.RoutedExperts},
		{"router", want.Router, got.Router},
		{"head", want.Head, got.Head},
		{"embedding", want.Embedding, got.Embedding},
	} {
		if c.want != c.got {
			diff = append(diff, fmt.Sprintf("%s: the engine serves %q, the graph stores %q",
				c.class, c.want, c.got))
		}
	}
	if want.LinearScaleElements != got.LinearScaleElements {
		diff = append(diff, fmt.Sprintf("linear scale: the engine serves one scale per %d weights, "+
			"the graph stores one per %d", want.LinearScaleElements, got.LinearScaleElements))
	}
	if len(diff) == 0 {
		return nil
	}
	return fmt.Errorf("served_weights_mismatch: the graph %s does not store the layout the engine "+
		"serves (%s); an empty value is a class the graph does not contain", g.Name, strings.Join(diff, "; "))
}

// storedLayout reads the layout a graph stores, in the words ServedWeights uses. A class stored
// at more than one dtype reads "mixed", which no engine statement equals; an embedding of a width
// no dtype has reads "other".
func storedLayout(g *model.Graph) enginesettings.ServedWeights {
	var linear, experts, router oneDType
	for _, lk := range g.LayerKinds {
		routedRows := 0
		for _, n := range lk.Nodes {
			if n.Op == model.OpGroupedGEMM && !n.IsSharedExpert() && n.Aliases == "" {
				routedRows = n.Experts
			}
		}
		for _, n := range lk.Nodes {
			if (n.Op != model.OpGEMM && n.Op != model.OpGroupedGEMM) || n.Aliases != "" {
				continue
			}
			d := string(nodeDType(g, n))
			switch {
			case n.Op == model.OpGroupedGEMM && !n.IsSharedExpert():
				experts.add(d)
				if n.SharedExperts > 0 {
					// A routed node that also carries the shared expert stores the shared one at
					// the routed dtype, which is not a class of its own: report the mix.
					linear.add(d)
				}
			case n.Op == model.OpGEMM && routedRows > 0 && n.N == routedRows:
				router.add(d)
			default:
				linear.add(d)
			}
		}
	}
	out := enginesettings.ServedWeights{Linear: linear.get(), RoutedExperts: experts.get(), Router: router.get()}
	if ws := g.Global.WeightScale; ws != nil && out.Linear != "" {
		out.LinearScaleElements = ws.Elements
	}
	var head oneDType
	for _, n := range g.Head {
		if n.Op == model.OpGEMM && n.Aliases == "" && n.N == g.Global.VocabSize {
			head.add(string(nodeDType(g, n)))
		}
	}
	out.Head = head.get()
	var emb oneDType
	for _, n := range g.Embedding {
		if n.Op == model.OpGather && n.Aliases == "" && int(n.Rows) == g.Global.VocabSize &&
			g.Global.HiddenSize > 0 {
			emb.add(embeddingDType(n.RowBytes, g.Global.HiddenSize))
		}
	}
	out.Embedding = emb.get()
	return out
}

// embeddingDType names the dtype whose width is rowBytes over the hidden size.
func embeddingDType(rowBytes, hidden int) string {
	if rowBytes%hidden == 0 {
		switch rowBytes / hidden {
		case 4:
			return string(model.DTypeFP32)
		case 2:
			return string(model.DTypeBF16)
		case 1:
			return string(model.DTypeFP8)
		}
	}
	return "other"
}

// oneDType collects the dtypes of a class: none reads "", one reads itself, several "mixed".
type oneDType struct{ seen []string }

func (o *oneDType) add(d string) {
	for _, s := range o.seen {
		if s == d {
			return
		}
	}
	o.seen = append(o.seen, d)
}

func (o *oneDType) get() string {
	switch len(o.seen) {
	case 0:
		return ""
	case 1:
		return o.seen[0]
	}
	return "mixed"
}

// nodeDType is the dtype a node's parameters are stored at: its own, else the global one.
func nodeDType(g *model.Graph, n model.Node) model.DType {
	if n.WeightDType != "" {
		return n.WeightDType
	}
	return g.Global.WeightDType
}
