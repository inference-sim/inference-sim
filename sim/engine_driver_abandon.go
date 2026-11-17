package sim

import (
	"fmt"
	"sort"

	"github.com/inference-sim/blis-substrate/contracts/engine"
)

// Abandon drops every request the engine holds, waiting or running, at now, for an instance that
// stops at once (an IMMEDIATE drain). The engine frees each request (Finish with FinishAbort), and
// the driver counts it dropped as unservable, as it does a request the engine fails before it emits
// anything (finish): the instance's books lose it without a completion, it is recorded unserved,
// and the done callback hears of it as a drop, so the cluster writes its fate (SIM-8) and a
// session does not go on to its next round. It returns how many requests it dropped, in request
// ID order (R2).
func (d *EngineDriver) Abandon(now int64) int {
	ids := make([]string, 0, len(d.reqs))
	for id := range d.reqs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	n := 0
	for _, id := range ids {
		r := d.reqs[id]
		if r == nil || r.req.State == StateCompleted || r.req.State == StateTimedOut {
			continue
		}
		if err := d.eng.Finish(id, engine.FinishAbort, now); err != nil {
			panic(fmt.Sprintf("engine driver: Finish(%s, abort): %v", id, err))
		}
		d.noteFreed(id)
		req := r.req
		d.Metrics.DroppedUnservable++
		d.Metrics.RecordUnserved(req, false)
		delete(d.Metrics.Requests, req.ID)
		delete(d.reqs, req.ID)
		if d.unserved == nil {
			d.unserved = map[string]bool{}
		}
		d.unserved[req.ID] = true
		// As finish: a drop reaches the done callback as StateQueued.
		req.State = StateQueued
		d.done(req, now)
		n++
	}
	return n
}
