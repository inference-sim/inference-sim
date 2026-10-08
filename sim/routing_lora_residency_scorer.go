package sim

import (
	"time"

	"github.com/inference-sim/inference-sim/sim/lorascore"
	"github.com/inference-sim/inference-sim/sim/lorascore/residency"
)

// loraResidencyDemandHalfLife matches the lora-residency-scorer EPP plugin's
// default demandHalfLife.
const loraResidencyDemandHalfLife = time.Minute

// newLoRAResidencyScorer builds the "lora-residency" scorer: the
// lora-residency-scorer EPP plugin's scoring core (sim/lorascore, vendored from
// lora-control) driven only by what a router observes. Routing decisions feed
// Routed, first tokens (RequestStartObserver) feed Started, terminal requests
// (RequestCompletionObserver) feed Finished, and each snapshot's MaxLoras and
// ActiveAdapters feed capacity and protection. It never reads ResidentAdapters,
// the simulator's ground truth, which no router sees.
//
// It models aggregated serving only; NewClusterSimulator rejects it under PD/EPD
// disaggregation, where the routed request's lifecycle runs on sub-requests.
//
// The CPU cache size is unknown to a router; it is taken as vLLM's default,
// max_cpu_loras = max_loras. A base-model request (Adapter "") scores every
// instance 1 and is not tracked. Scores and demand are evaluated at the
// routing clock. Ticks are microseconds.
func newLoRAResidencyScorer() scorerParts {
	model := residency.NewModel(0)
	demand, err := lorascore.NewDemand(loraResidencyDemandHalfLife)
	if err != nil {
		panic(err) // a constant half-life; unreachable
	}
	weights := lorascore.DefaultWeights()
	routedTo := map[string]string{} // request ID -> instance, for tracked requests
	started := map[string]bool{}    // request IDs whose first token was seen

	at := func(tick int64) time.Time { return time.UnixMicro(tick) }
	// clock is the routing decision's time (RouterState.Clock), which gateway
	// queueing, admission and redirects can put well after req.ArrivalTime.
	var clock int64

	score := func(req *Request, snapshots []RoutingSnapshot) map[string]float64 {
		ids := make([]string, len(snapshots))
		for i, snap := range snapshots {
			ids[i] = snap.ID
			model.ObserveCapacity(snap.ID, snap.MaxLoras)
			active := make([]string, 0, len(snap.ActiveAdapters))
			for a := range snap.ActiveAdapters {
				active = append(active, a)
			}
			model.ObserveActive(snap.ID, active)
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
		routedTo[req.ID] = target
		model.Routed(target, req.Adapter)
		demand.Observe(req.Adapter, at(clock))
	}
	onStart := func(req *Request, instanceID string, tick int64) {
		if req == nil || routedTo[req.ID] != instanceID || started[req.ID] {
			return // untracked, another instance's copy, or a re-prefill after preemption
		}
		started[req.ID] = true
		model.Started(instanceID, req.Adapter, at(tick))
	}
	// A completion settles the request on the instance it was routed to, wherever
	// it is reported: a request that leaves another way (a disaggregated decode
	// pod, say) must not stay pending or running here forever.
	onComplete := func(req *Request, _ string, tick int64) {
		if req == nil {
			return
		}
		inst, tracked := routedTo[req.ID]
		if !tracked {
			return
		}
		model.Finished(inst, req.Adapter, at(tick), started[req.ID])
		delete(routedTo, req.ID)
		delete(started, req.ID)
	}
	return scorerParts{score: score, observe: observe, onStart: onStart, onComplete: onComplete,
		setClock: func(c int64) { clock = c }}
}
