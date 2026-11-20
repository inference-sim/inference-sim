package latencykernel

import (
	"github.com/inference-sim/blis-substrate/contracts/kernel"

	"github.com/inference-sim/blis-substrate/kernel/internal/price"
)

// A layer suffix that runs on a subset of a step's rows.
//
// An engine can run the layers after the last one that produces KV on each request's trailing
// window only: the earlier layers write the KV of every row, and the later layers, which own
// nothing later layers read, are run for the rows whose outputs are used (the last window of
// each request's scheduled tokens). The graph describes the layers; how many rows each runs in a
// given step is the engine's execution decision, stated in Batch.TailFrom and ReqShape.TailRows
// (plan.StepShape.TailFrom, RequestStep.TailRows), and the kernel prices what it is told and
// resolves nothing (SUB-6). A step that states none prices every layer on every scheduled token,
// as before.
//
// The tail is priced as its own batch through the same per-layer pricing: request i schedules its
// last TailRows tokens, and the tokens before them are context it has already computed, so a
// windowed layer reads the same keys and a full-context layer the same context it reads for those
// queries in the whole batch. Only the rows change.
//
// What stays on the whole batch, deliberately: the embedding, the layers before TailFrom, the
// head (it reads only each request's last row), the drafter, and the host, launch and
// scheduler terms. The step launches every layer whatever rows it gives them.
//
// TailFrom names the first layer priced wholly on the subset. A layer that first writes KV for
// every row and then reruns on the subset (the last layer that produces KV, in an engine that
// does so) is priced on the subset alone when the engine states TailFrom at it: the work it does
// for the rows outside the subset, its input projections and cache writes, is not charged. An
// engine that would rather charge it states TailFrom one layer later and pays that layer's
// feed-forward and attention on every row instead. The engine's lowering says which it chose and
// how large the difference is.

// layerSpan is a run of the target's layers: each plan layer kind with the count of its layers
// that lie in the run, a kind with none absent, and the fits and cache views aligned to them.
type layerSpan struct {
	layers []price.PlannedLayer
	fits   [][]gemmFit
	kv     []layerKV
}

// spanOf is the target's layers at positions [lo, hi) of the expanded stack seq. A kind appears once
// in the plan with every layer of that kind, so the run's share of it is counted here.
func (k *Kernel) spanOf(seq []string, lo, hi int) layerSpan {
	lo, hi = max(lo, 0), min(hi, len(seq))
	counts := map[string]int{}
	for _, id := range seq[min(lo, hi):hi] {
		counts[id]++
	}
	var sp layerSpan
	for i, l := range k.plan.Layers {
		n := counts[l.ID]
		if n == 0 {
			continue
		}
		l.Count = n
		sp.layers = append(sp.layers, l)
		sp.fits = append(sp.fits, k.layerFits[i])
		sp.kv = append(sp.kv, k.layerKV[i])
	}
	return sp
}

// tailBatchOf is the batch the layers from b.TailFrom run, and whether b has a tail: a step that
// states TailFrom with a layer at or after it. Request i keeps its last TailRows tokens (all of
// them when it states none, or more than it schedules), at their own positions: the tokens
// before them join the computed context. The pad is the tail's own (Batch.TailPadTokens), not
// the target's, since the engine launches the tail at a size of its own.
func (k *Kernel) tailBatchOf(b kernel.Batch) (kernel.Batch, bool) {
	if b.TailFrom <= 0 || b.TailFrom >= k.graph.Stack.Layers() {
		return kernel.Batch{}, false
	}
	t := b
	t.PadTokens, t.TailFrom, t.TailPadTokens = max(b.TailPadTokens, 0), 0, 0
	t.Reqs = make([]kernel.ReqShape, len(b.Reqs))
	for i, r := range b.Reqs {
		rows := r.Scheduled
		if r.TailRows > 0 && r.TailRows < rows {
			rows = r.TailRows
		}
		t.Reqs[i] = r
		t.Reqs[i].Computed = r.Computed + r.Scheduled - rows
		t.Reqs[i].Scheduled = rows
		t.Reqs[i].TailRows = 0
	}
	return t, true
}

// priceTarget prices the target's layer stack for one step: every layer on the step's shape s,
// or, where the batch states a tail, the layers before it on s and the rest on the tail's own.
func (k *Kernel) priceTarget(b kernel.Batch, s *stepShape, per *[numResources]float64) float64 {
	tb, ok := k.tailBatchOf(b)
	if !ok {
		return k.priceLayers(k.plan.Layers, k.layerFits, k.layerKV, s, per)
	}
	ts := k.shapeOf(tb)
	return k.priceSplit(0, k.graph.Stack.Layers(), b.TailFrom, s, &ts, per)
}

// priceSplit prices the target's layers at positions [lo, hi) with those before from on head and
// the rest on tail.
func (k *Kernel) priceSplit(lo, hi, from int, head, tail *stepShape,
	per *[numResources]float64) float64 {
	seq := k.graph.Stack.Expand()
	var seconds float64
	if first := k.spanOf(seq, lo, min(hi, from)); len(first.layers) > 0 {
		seconds += k.priceLayers(first.layers, first.fits, first.kv, head, per)
	}
	if rest := k.spanOf(seq, max(lo, from), hi); len(rest.layers) > 0 {
		seconds += k.priceLayers(rest.layers, rest.fits, rest.kv, tail, per)
	}
	return seconds
}
