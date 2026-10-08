package sim

import (
	"time"

	"github.com/inference-sim/inference-sim/sim/lorascore"
	"github.com/inference-sim/inference-sim/sim/lorascore/residency"
)

// newLoRAResidencyTruthScorer builds "lora-residency-truth": the lora-residency
// scorer's policy fed ground-truth residency instead of its router-side estimate,
// so the gap between the two measures what the estimate costs. It is a
// simulation-only reference; no router can see this.
//
// Each Score rebuilds a residency model per instance from the snapshot's
// ResidentOrder and ResidentPinned: every resident adapter is replayed in true
// LRU→MRU order, the pinned ones left running (never evictable) and the rest
// finished; an adapter still loading is replayed last and left running, since its
// slot is taken and cannot be given up. Capacity is the instance's MaxLoras, and the CPU cache equals it,
// matching BLIS's single adapter tier. What a router could observe is kept as
// lora-residency has it: requests this scorer routed but which have not yet
// produced a first token count as pending, and demand comes from its own routing
// decisions, at the routing clock. Everything else (cost model, weights,
// normalization, base-model handling) is lorascore.Score, unchanged.
func newLoRAResidencyTruthScorer() scorerParts {
	demand, err := lorascore.NewDemand(loraResidencyDemandHalfLife)
	if err != nil {
		panic(err) // a constant half-life; unreachable
	}
	weights := lorascore.DefaultWeights()
	type routed struct{ instance, adapter string }
	pending := map[string]routed{} // request ID -> where it was routed, until it starts or ends
	at := func(tick int64) time.Time { return time.UnixMicro(tick) }
	var clock int64

	score := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		model := residency.NewModel(0)
		ids := make([]string, len(snapshots))
		replay := at(0)
		for i, snap := range snapshots {
			ids[i] = snap.ID
			model.ObserveCapacity(snap.ID, snap.MaxLoras)
			for _, a := range snap.ResidentOrder {
				replay = replay.Add(time.Microsecond)
				model.Routed(snap.ID, a)
				model.Started(snap.ID, a, replay)
				if !snap.ResidentPinned[a] {
					model.Finished(snap.ID, a, replay, true)
				}
			}
			if a := snap.LoadingAdapter; a != "" {
				replay = replay.Add(time.Microsecond)
				model.Routed(snap.ID, a)
				model.Started(snap.ID, a, replay)
			}
		}
		for _, p := range pending {
			model.Routed(p.instance, p.adapter)
		}
		adapter := ""
		if req != nil {
			adapter = req.Adapter
		}
		return lorascore.Score(model, demand, adapter, adapter == "", ids, weights, at(clock))
	}
	observe := func(req *Request, target string) {
		if req == nil || req.Adapter == "" {
			return
		}
		pending[req.ID] = routed{target, req.Adapter}
		demand.Observe(req.Adapter, at(clock))
	}
	settle := func(req *Request, _ string, _ int64) {
		if req != nil {
			delete(pending, req.ID)
		}
	}
	return scorerParts{score: score, observe: observe, onStart: settle, onComplete: settle,
		setClock: func(c int64) { clock = c }}
}
