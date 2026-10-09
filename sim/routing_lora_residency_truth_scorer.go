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
// finished. An adapter still loading is not resident yet: a request for it is
// pending (its load is under way), while for any other adapter its reserved slot
// is taken, modelled as a running placeholder that cannot be evicted. Capacity is
// the snapshot's ResidentCapacity, and the CPU cache equals it, matching BLIS's
// single adapter tier. The cluster pins these snapshot fields live whenever this
// scorer is configured. What a router could observe is kept as
// lora-residency has it: requests this scorer routed but which have not yet
// produced a first token count as pending, and demand comes from its own routing
// decisions, at the routing clock. Everything else (cost model, weights,
// normalization, base-model handling) is lorascore.Score, unchanged.
// reservedSlot stands for the slot an in-progress load has reserved. The NUL byte
// keeps it from colliding with an adapter ID; it is never demanded, so never priced.
const reservedSlot = "\x00reserved"

func newLoRAResidencyTruthScorer() scorerParts {
	demand, err := lorascore.NewDemand(loraResidencyDemandHalfLife)
	if err != nil {
		panic(err) // a constant half-life; unreachable
	}
	weights := lorascore.DefaultWeights()
	type routed struct{ instance, adapter string }
	pending := map[string]routed{} // request ID -> where it was routed, until it starts or ends
	// Admit asks only whether a pair has any pending request, so each pair is
	// replayed once per Score, however long the backlog.
	pendingPairs := map[routed]int{}
	at := func(tick int64) time.Time { return time.UnixMicro(tick) }
	var clock int64

	score := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		adapter := ""
		if req != nil {
			adapter = req.Adapter
		}
		model := residency.NewModel(0)
		ids := make([]string, len(snapshots))
		replay := at(0)
		for i, snap := range snapshots {
			ids[i] = snap.ID
			model.ObserveCapacity(snap.ID, snap.ResidentCapacity)
			for _, a := range snap.ResidentOrder {
				replay = replay.Add(time.Microsecond)
				model.Routed(snap.ID, a)
				model.Started(snap.ID, a, replay)
				if !snap.ResidentPinned[a] {
					model.Finished(snap.ID, a, replay, true)
				}
			}
			switch snap.LoadingAdapter {
			case "":
			case adapter:
				model.Routed(snap.ID, adapter)
			default:
				replay = replay.Add(time.Microsecond)
				model.Started(snap.ID, reservedSlot, replay)
			}
		}
		for p := range pendingPairs {
			model.Routed(p.instance, p.adapter)
		}
		return lorascore.Score(model, demand, adapter, adapter == "", ids, weights, at(clock))
	}
	unpend := func(id string) {
		p, ok := pending[id]
		if !ok {
			return
		}
		delete(pending, id)
		if pendingPairs[p]--; pendingPairs[p] == 0 {
			delete(pendingPairs, p)
		}
	}
	observe := func(req *Request, target string) {
		if req == nil || req.Adapter == "" {
			return
		}
		unpend(req.ID) // a re-routed request (e.g. drained and re-injected) moves
		p := routed{target, req.Adapter}
		pending[req.ID] = p
		pendingPairs[p]++
		demand.Observe(req.Adapter, at(clock))
	}
	settle := func(req *Request, _ string, _ int64) {
		if req != nil {
			unpend(req.ID)
		}
	}
	return scorerParts{score: score, observe: observe, onStart: settle, onComplete: settle,
		setClock: func(c int64) { clock = c }}
}
